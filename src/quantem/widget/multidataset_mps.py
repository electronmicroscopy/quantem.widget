"""Lazy multi-dataset MacBook handle: see dataset 0 in ~2s, browse the rest as
they decode behind a slider.

This is the MPS (Apple Silicon) implementation of ``load([masters])``. Because a
4-5 dataset 5D Metal stack is 12s+ to decode and may not fit 24 GB of unified
memory, the MPS path is LAZY: dataset 0 decodes synchronously, the viewer is
shown immediately over a frame slider spanning all N datasets, and a single
background GPU-worker thread decodes datasets 1..N-1 into the live container.
Sliding to a not-yet-decoded dataset shows the last ready one until its slot
fills (auto-updates). A progress line prints ``[k/N loaded]`` as each finishes.

``load([...])`` returns a :class:`LazyMacbookDatasets` handle (dataset 0 already
decoded); ``Show4DSTEM(handle)`` builds the viewer and starts the background
fill. One dedicated worker owns every Metal decode (the command queue is serial
- one owner is the safe + correct model). Memory is the same as loading all
upfront (~1.2 GB each at bin4); lazy hides the TIME, not the footprint. Run
``discover_masters(folder)`` and inspect representative metadata before loading
the full stack.

CUDA / CPU never reach this module: ``load([...])`` eager-stacks into one 5D
array there (big VRAM, instant dataset switch). Only MPS is lazy.

Usage::

    from quantem.widget import load, Show4DSTEM
    Show4DSTEM(load(master_paths, det_bin=4))   # dataset 0 shows now; slide across
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import queue
import threading
import time
from typing import Any, Callable


@dataclass
class _DecodeJob:
    """One FIFO decode request owned by :class:`LazyMacbookDatasets`."""

    path: str
    label: str
    slot: int | None
    index_hint: int
    done: threading.Event = field(default_factory=threading.Event)
    result_idx: int | None = None
    error: BaseException | None = None


class LazyMacbookDatasets:
    """MPS lazy multi-dataset handle returned by ``load([masters])``.

    Holds dataset 0 (already decoded) plus the spec to decode 1..N-1 on demand.
    :func:`quantem.widget.Show4DSTEM` consumes it: builds the 5D viewer over the
    live :class:`MultiChunkedFrames` container, then starts one background worker
    that fills the remaining datasets. Browsing is instant; the slider only spans
    decoded datasets and grows as each lands.
    """

    def __init__(
        self,
        masters,
        det_bin,
        names,
        multi,
        decode,
        verbose=True,
        validate_master: Callable[[str], None] | None = None,
    ):
        self.masters = [str(master) for master in masters]
        self.det_bin = det_bin
        self.names = [str(name) for name in names]
        self.multi = multi  # MultiChunkedFrames([ds0], n_total=N, names=names)
        self._decode = decode
        self._validate_master = validate_master
        self.verbose = bool(verbose)
        self._lock = threading.RLock()
        self._poll_lock = threading.Lock()
        self._decode_queue: queue.Queue[_DecodeJob | None] = queue.Queue()
        self._decode_stop = threading.Event()
        self._decode_thread: threading.Thread | None = None
        self._accept_decode_results = True
        self._pending_jobs: dict[str, _DecodeJob] = {}
        self._decode_errors: dict[str, str] = {}
        self._initial_slots = {
            self._master_key(master): idx for idx, master in enumerate(self.masters)
        }
        self._successful_master_keys = (
            {self._master_key(self.masters[0])} if self.masters else set()
        )
        self._ready_signatures: dict[str, dict[str, Any]] = {}
        self._waiting_detail = ""
        self._last_error_detail = ""
        self._status_state = "hidden"
        self._status_detail = ""
        self._status_callback: Callable[[str, str], None] | None = None
        self._watching = False
        self._watch_started = False
        self._watch_failed = False
        self._closed = False
        self._watch_stop: threading.Event | None = None
        self._watch_thread: threading.Thread | None = None

    def build_viewer(self, **viewer_kwargs):
        """Show dataset 0 now and queue 1..N-1 on one owned FIFO worker."""
        from quantem.widget.show4dstem_mps import Show4DSTEM_MACBOOK

        verbose = bool(viewer_kwargs.pop("verbose", self.verbose))
        viewer_kwargs.setdefault("frame_dim_label", "Dataset")
        viewer_kwargs.setdefault("frame_labels", list(self.names))
        viewer = Show4DSTEM_MACBOOK(self.multi, verbose=verbose, **viewer_kwargs)
        # Also bind non-folder ``Show4DSTEM(load([masters]))`` viewers so their
        # close/free lifecycle joins this owned initial-fill worker.
        viewer._mps_folder_live = self
        for idx in range(1, len(self.masters)):
            self._queue_master(
                self.masters[idx],
                label=self.names[idx],
                slot=idx,
                async_=True,
                validate=False,
            )
        return viewer

    @property
    def pending_masters(self) -> tuple[str, ...]:
        """Absolute paths queued or currently decoding."""
        with self._lock:
            return tuple(job.path for job in self._pending_jobs.values())

    @property
    def successful_masters(self) -> tuple[str, ...]:
        """Absolute paths whose decode and compatibility checks succeeded."""
        with self._lock:
            return tuple(sorted(self._successful_master_keys))

    def set_status_callback(
        self,
        callback: Callable[[str, str], None] | None,
    ) -> None:
        """Publish watch protocol state to the mounted viewer."""
        with self._lock:
            self._status_callback = callback
            state = self._status_state
            detail = self._status_detail
        if callback is not None:
            callback(state, detail)

    def _set_status(self, state: str, detail: str = "") -> None:
        text = " ".join(str(detail).split())
        compact: list[str] = []
        for token in text.split(" "):
            stripped = token.strip("()[]{}<>,.;:")
            if os.path.isabs(stripped) and os.sep in stripped[1:]:
                token = token.replace(
                    stripped,
                    os.path.basename(stripped) or "source file",
                )
            compact.append(token)
        detail = " ".join(compact)
        if len(detail) > 360:
            detail = f"{detail[:359].rstrip()}…"
        with self._lock:
            self._status_state = str(state)
            self._status_detail = detail
            callback = self._status_callback
        if callback is not None:
            try:
                callback(str(state), detail)
            except Exception:
                # Status is advisory; it must not terminate acquisition work.
                pass

    def _refresh_activity_status(self) -> None:
        with self._lock:
            watch_thread = self._watch_thread
            watch_requested = self._watching
            watching = (
                self._watching
                and watch_thread is not None
                and watch_thread.is_alive()
            )
            watch_started = self._watch_started
            watch_failed = self._watch_failed
            pending = bool(self._pending_jobs)
            error = self._last_error_detail
            waiting = self._waiting_detail
        if not watching:
            if watch_requested or watch_failed:
                self._set_status(
                    "error",
                    error
                    or "Folder watch worker is not running. Restart folder watching.",
                )
            elif watch_started:
                self._set_status("stopped", "Folder watching has stopped.")
            else:
                self._set_status("hidden", "")
        elif error:
            self._set_status("error", error)
        elif pending:
            self._set_status("updating", "Decoding newly completed 4D-STEM data.")
        elif waiting:
            self._set_status("waiting", waiting)
        else:
            self._set_status("watching", "Watching for completed 4D-STEM masters.")

    @staticmethod
    def _readiness_is_waiting(report: Any) -> bool:
        """Classify transient acquisition states separately from bad data."""
        actual = getattr(report, "actual_frames", None)
        expected = getattr(report, "expected_frames", None)
        if actual is not None and expected is not None:
            if int(actual) < int(expected):
                return True
            if int(actual) > int(expected):
                return False
        reason = str(getattr(report, "reason", "")).casefold()
        permanent_markers = (
            "inconsistent detector",
            "inconsistent dtype",
            "inconsistent scan_shape",
            "incompatible",
            "does not match",
            "conflicting",
            "expected at least (frame, det_row, det_col)",
        )
        if any(marker in reason for marker in permanent_markers):
            return False
        transient_markers = (
            "missing",
            "empty",
            "zero stored frames",
            "not readable hdf5",
            "cannot be inspected",
            "changed during readiness inspection",
            "headers are incomplete",
            "header to finish writing",
            "no entry/data",
            "has no entry/data/data",
        )
        return any(marker in reason for marker in transient_markers)

    def _ensure_decode_worker(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("This MPS folder source has been closed.")
            if self._decode_thread is not None and self._decode_thread.is_alive():
                return
            self._decode_stop = threading.Event()
            self._accept_decode_results = True
            thread = threading.Thread(
                target=self._decode_worker,
                name="Show4DSTEMMPS-decode-worker",
                daemon=True,
            )
            self._decode_thread = thread
            try:
                thread.start()
            except BaseException as exc:
                self._decode_thread = None
                self._decode_stop.set()
                self._accept_decode_results = False
                self._last_error_detail = (
                    "Could not start the MPS decode worker: "
                    f"{type(exc).__name__}: {str(exc)[:180]}"
                )
                raise

    def _finish_job(self, job: _DecodeJob) -> None:
        key = self._master_key(job.path)
        with self._lock:
            if self._pending_jobs.get(key) is job:
                self._pending_jobs.pop(key, None)
        job.done.set()
        self._refresh_activity_status()

    def _decode_worker(self) -> None:
        while True:
            job = self._decode_queue.get()
            if job is None:
                self._decode_queue.task_done()
                break
            try:
                if self._decode_stop.is_set():
                    raise RuntimeError("MPS folder decode stopped before completion.")
                if self.verbose:
                    print(f"[decode] loading {job.label} ...", flush=True)
                started = time.perf_counter()
                frames = self._decode(job.path)
                with self._lock:
                    accept = (
                        self._accept_decode_results
                        and not self._decode_stop.is_set()
                        and not self._closed
                    )
                if not accept:
                    raise RuntimeError(
                        "MPS folder decode finished after the viewer stopped; "
                        "the result was discarded."
                    )
                if job.slot is None:
                    idx = int(self.multi.append_dataset(frames, name=job.label))
                    with self._lock:
                        self.masters.append(job.path)
                        self.names.append(job.label)
                else:
                    idx = int(job.slot)
                    self.multi.set_dataset(idx, frames)
                key = self._master_key(job.path)
                with self._lock:
                    self._successful_master_keys.add(key)
                    prior_error = self._decode_errors.pop(key, None)
                    if prior_error and self._last_error_detail == prior_error:
                        self._last_error_detail = ""
                job.result_idx = idx
                if self.verbose:
                    print(
                        f"[decode] {job.label} ready in "
                        f"{time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
            except BaseException as exc:
                job.error = exc
                key = self._master_key(job.path)
                detail = (
                    f"{job.label}: {str(exc)[:180]}. This master remains "
                    "retryable; wait for file completion or correct the error, "
                    "then poll again."
                )
                with self._lock:
                    self._decode_errors[key] = detail
                    if not self._decode_stop.is_set() and not self._closed:
                        self._last_error_detail = detail
                if self.verbose and not self._decode_stop.is_set():
                    print(f"[decode] {job.label} FAILED: {str(exc)[:120]}", flush=True)
            finally:
                self._finish_job(job)
                self._decode_queue.task_done()

    def _queue_master(
        self,
        master,
        *,
        label: str | None = None,
        slot: int | None = None,
        async_: bool,
        validate: bool = True,
    ) -> _DecodeJob | None:
        path = self._master_key(master)
        resolved_label = str(label) if label is not None else os.path.basename(path)
        if resolved_label.endswith("_master.h5"):
            resolved_label = resolved_label[: -len("_master.h5")]
        key = self._master_key(path)
        with self._lock:
            if key in self._successful_master_keys:
                return None
            existing = self._pending_jobs.get(key)
        if existing is not None:
            if not async_:
                existing.done.wait()
                if existing.error is not None:
                    raise existing.error
            return existing
        if validate and self._validate_master is not None:
            self._validate_master(path)
        if slot is None:
            slot = self._initial_slots.get(key)
        duplicate: _DecodeJob | None = None
        with self._lock:
            if key in self._successful_master_keys:
                return None
            duplicate = self._pending_jobs.get(key)
            if duplicate is None:
                append_pending = sum(
                    1 for pending in self._pending_jobs.values() if pending.slot is None
                )
                index_hint = (
                    int(slot)
                    if slot is not None
                    else len(getattr(self.multi, "datasets", self.masters))
                    + append_pending
                )
                job = _DecodeJob(path, resolved_label, slot, index_hint)
                self._pending_jobs[key] = job
                self._decode_errors.pop(key, None)
        if duplicate is not None:
            if not async_:
                duplicate.done.wait()
                if duplicate.error is not None:
                    raise duplicate.error
            return duplicate
        try:
            self._ensure_decode_worker()
        except BaseException as exc:
            job.error = exc
            self._finish_job(job)
            raise
        self._decode_queue.put(job)
        self._refresh_activity_status()
        if not async_:
            job.done.wait()
            if job.error is not None:
                raise job.error
        return job

    def append_master(
        self,
        master,
        *,
        name: str | None = None,
        async_: bool = True,
    ) -> int:
        """Decode one newly discovered master and append it to the live dataset list.

        Parameters
        ----------
        master
            Path to a 4D-STEM master file compatible with the existing stack.
        name
            Optional dataset label. Defaults to the master filename stem.
        async_
            If ``True`` (default), decode in a background daemon thread and
            return the future slot index immediately. If ``False``, decode and
            append before returning.

        Returns
        -------
        int
            Dataset slot index assigned to the appended master.
        """
        job = self._queue_master(master, label=name, async_=async_)
        if job is None:
            key = self._master_key(master)
            for idx, known in enumerate(self.masters):
                if self._master_key(known) == key:
                    return idx
            raise RuntimeError(f"Decoded MPS master {master!s} has no dataset slot.")
        return int(job.result_idx if job.result_idx is not None else job.index_hint)

    @staticmethod
    def _master_key(master) -> str:
        return os.path.abspath(os.path.expanduser(str(master)))

    def append_new_masters(self, masters, *, async_: bool = True) -> list[int]:
        """Append only masters that are not already present in this live handle.

        This is the safe inner loop for microscope/live-folder workflows: callers
        can repeatedly pass the current discovered master list, and already loaded
        acquisitions are skipped without rebuilding the viewer.
        """
        added: list[int] = []
        for master in masters:
            key = self._master_key(master)
            with self._lock:
                known = (
                    key in self._successful_master_keys or key in self._pending_jobs
                )
            if known:
                continue
            try:
                added.append(self.append_master(key, async_=async_))
            except Exception as exc:
                with self._lock:
                    detail = self._decode_errors.get(key) or (
                        f"{_master_name(key)}: {str(exc)[:180]}. This master "
                        "remains retryable; wait for file completion or correct "
                        "the error, then poll again."
                    )
                    self._last_error_detail = detail
                self._refresh_activity_status()
        return added

    def poll_master_folder(
        self,
        folder,
        *,
        pattern: str = "*_master.h5",
        recursive: bool = True,
        scan_size: int | None = None,
        ready_only: bool = True,
        async_: bool = True,
        require_stable: bool = False,
    ) -> list[int]:
        """Discover and append once, returning immediately if a poll is active."""
        if not self._poll_lock.acquire(blocking=False):
            return []
        try:
            return self._poll_master_folder_once(
                folder,
                pattern=pattern,
                recursive=recursive,
                scan_size=scan_size,
                ready_only=ready_only,
                async_=async_,
                require_stable=require_stable,
            )
        finally:
            self._poll_lock.release()

    def _poll_master_folder_once(
        self,
        folder,
        *,
        pattern: str = "*_master.h5",
        recursive: bool = True,
        scan_size: int | None = None,
        ready_only: bool = True,
        async_: bool = True,
        require_stable: bool = False,
    ) -> list[int]:
        """Discover ready master files in *folder* and append new acquisitions.

        Parameters mirror :func:`quantem.widget.io.discover_masters`. When
        ``ready_only`` is true, partially written masters are ignored until their
        linked data files are present.
        """
        from quantem.widget.io import (
            discover_masters,
            inspect_master_readiness,
            is_master_ready,
        )

        scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
        with self._lock:
            watch_thread = self._watch_thread
            watching = bool(
                self._watching
                and watch_thread is not None
                and watch_thread.is_alive()
            )
        if watching:
            self._set_status(
                "updating",
                "Checking the folder for new 4D-STEM data.",
            )
        with self._lock:
            self._waiting_detail = ""
            self._last_error_detail = ""
        try:
            masters = discover_masters(
                os.path.expanduser(str(folder)),
                pattern=pattern,
                recursive=recursive,
                # A stability watch must see incomplete/mixed candidates so it
                # can report amber readiness and red contract failures itself.
                scan_shape=None if require_stable else scan_shape,
                verbose=False,
            )
        except ValueError as exc:
            if "no files matching" not in str(exc).casefold():
                with self._lock:
                    self._last_error_detail = (
                        f"Folder discovery failed: {str(exc)[:220]}. Correct the "
                        "folder pattern or scan-size request, then poll again."
                    )
                self._refresh_activity_status()
                raise
            masters = []
        except Exception as exc:
            with self._lock:
                self._last_error_detail = (
                    f"Folder discovery failed: {str(exc)[:220]}. Check folder "
                    "access and retry; restart watching if the error persists."
                )
            self._refresh_activity_status()
            raise

        discovered_keys = {self._master_key(master) for master in masters}
        with self._lock:
            represented = self._successful_master_keys | set(self._pending_jobs)
            self._ready_signatures = {
                key: signature
                for key, signature in self._ready_signatures.items()
                if key in discovered_keys and key not in represented
            }

        ready: list[Any] = []
        waiting: list[str] = []
        errors: list[str] = []
        for master in masters:
            key = self._master_key(master)
            with self._lock:
                if key in self._successful_master_keys or key in self._pending_jobs:
                    continue
            try:
                if require_stable:
                    report = inspect_master_readiness(master, scan_shape=scan_shape)
                    if not report.ready:
                        self._ready_signatures.pop(key, None)
                        detail = (
                            f"{_master_name(master)}: {report.reason}. {report.action}"
                        )
                        if self._readiness_is_waiting(report):
                            waiting.append(detail)
                        else:
                            errors.append(detail)
                        continue
                    previous = self._ready_signatures.get(key)
                    self._ready_signatures[key] = report.source_signature
                    if previous != report.source_signature:
                        waiting.append(
                            f"{_master_name(master)}: ready once; waiting for an "
                            "unchanged completion signature on the next poll."
                        )
                        continue
                elif ready_only and not is_master_ready(master):
                    waiting.append(
                        f"{_master_name(master)}: waiting for file completion."
                    )
                    continue
                if self._validate_master is not None:
                    self._validate_master(self._master_key(master))
                ready.append(master)
            except Exception as exc:
                errors.append(f"{_master_name(master)}: {str(exc)[:180]}")

        with self._lock:
            self._waiting_detail = (
                "Waiting for file completion: " + " | ".join(waiting[:3])
                if waiting
                else ""
            )
            self._last_error_detail = " | ".join(errors[:3])
        if ready and not errors:
            with self._lock:
                thread = self._watch_thread
                watching = bool(
                    self._watching
                    and thread is not None
                    and thread.is_alive()
                )
            if watching:
                self._set_status(
                    "updating",
                    "Decoding newly completed 4D-STEM data.",
                )
        added = self.append_new_masters(ready, async_=async_)
        for master in ready:
            self._ready_signatures.pop(self._master_key(master), None)
        self._refresh_activity_status()
        return added

    def watch_master_folder(
        self,
        folder,
        *,
        interval: float = 2.0,
        pattern: str = "*_master.h5",
        recursive: bool = True,
        scan_size: int | None = None,
        ready_only: bool = True,
        async_: bool = True,
    ) -> "LazyMacbookDatasets":
        """Poll a live acquisition folder and append new ready masters.

        The existing Show4DSTEM viewer stays mounted; newly completed masters
        are appended to the dataset slider as they decode. Call
        :meth:`stop_watch` before starting a different watcher.
        """
        interval = float(interval)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError(
                "interval must be a finite positive number of seconds, got "
                f"{interval!r}."
            )
        if self._watch_thread is not None:
            self.stop_watch()
        if self._closed:
            raise RuntimeError("This MPS folder source has been closed.")
        self._ensure_decode_worker()
        stop = threading.Event()
        self._watch_stop = stop
        self._watching = True
        self._watch_started = True
        self._watch_failed = False

        def _worker() -> None:
            unexpected_exit = True
            unexpected_detail = ""
            try:
                while not stop.wait(interval):
                    try:
                        added = self.poll_master_folder(
                            folder,
                            pattern=pattern,
                            recursive=recursive,
                            scan_size=scan_size,
                            ready_only=ready_only,
                            async_=async_,
                            require_stable=True,
                        )
                        if self.verbose and added:
                            print(
                                f"[watch] appended {len(added)} new master(s)",
                                flush=True,
                            )
                    except Exception as exc:
                        with self._lock:
                            self._last_error_detail = (
                                "Folder discovery failed: "
                                + str(exc)[:180]
                                + ". Check folder access and restart watching if "
                                "the error persists."
                            )
                        self._refresh_activity_status()
                        if self.verbose:
                            print(
                                "[watch] master folder poll failed: "
                                f"{str(exc)[:120]}",
                                flush=True,
                            )
                unexpected_exit = False
            except BaseException as exc:
                unexpected_detail = (
                    "The folder watch worker stopped unexpectedly "
                    f"({type(exc).__name__}: "
                    f"{str(exc)[:120]}). Restart watching after checking the "
                    "notebook kernel log."
                )
            finally:
                with self._lock:
                    if self._watch_stop is stop:
                        self._watch_stop = None
                    if self._watch_thread is threading.current_thread():
                        self._watch_thread = None
                    self._watching = False
                if unexpected_exit and not stop.is_set():
                    with self._lock:
                        self._watch_failed = True
                        self._last_error_detail = unexpected_detail or (
                            "The folder watch worker stopped unexpectedly. Restart "
                            "watching after checking the notebook kernel log."
                        )
                    self._set_status("error", self._last_error_detail)
                elif stop.is_set():
                    self._set_status("stopped", "Folder watching has stopped.")

        self._watch_thread = threading.Thread(
            target=_worker,
            name="Show4DSTEMMPS-watch-master-folder",
            daemon=True,
        )
        try:
            self._watch_thread.start()
        except BaseException as exc:
            self._watching = False
            self._watch_failed = True
            self._watch_thread = None
            self._watch_stop = None
            self._stop_decode_worker()
            self._set_status(
                "error",
                "Could not start folder watching: "
                f"{type(exc).__name__}: {str(exc)[:180]}",
            )
            raise
        self._refresh_activity_status()
        return self

    def stop_watch(self) -> None:
        """Stop and join both folder discovery and the owned decode worker."""
        was_started = bool(self._watch_started)
        stop = getattr(self, "_watch_stop", None)
        thread = getattr(self, "_watch_thread", None)
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self._watching = False
        self._watch_failed = False
        self._watch_stop = None
        self._watch_thread = None
        with self._lock:
            self._ready_signatures.clear()
        self._stop_decode_worker()
        if was_started:
            self._set_status("stopped", "Folder watching has stopped.")
        else:
            self._set_status("hidden", "")

    def _stop_decode_worker(self) -> None:
        with self._lock:
            thread = self._decode_thread
            if thread is None:
                return
            self._accept_decode_results = False
            self._decode_stop.set()
            self._decode_queue.put(None)
        if thread is not threading.current_thread():
            thread.join()
        with self._lock:
            if self._decode_thread is thread:
                self._decode_thread = None
            self._decode_queue = queue.Queue()

    def wait_for_decodes(self, timeout: float | None = None) -> bool:
        """Wait until the FIFO has no queued/in-flight masters."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            with self._lock:
                jobs = list(self._pending_jobs.values())
            if not jobs:
                return True
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            if remaining == 0.0:
                return False
            jobs[0].done.wait(remaining)

    def shutdown(self) -> None:
        """Permanently stop this handle and suppress future viewer callbacks."""
        self.stop_watch()
        with self._lock:
            self._closed = True
            self._accept_decode_results = False
            self._status_callback = None
        try:
            self.multi.on_ready = None
        except Exception:
            pass


def _master_name(master: Any) -> str:
    name = os.path.basename(str(master))
    return name[: -len("_master.h5")] if name.endswith("_master.h5") else name


def load_macbook_datasets(
    masters,
    *,
    det_bin: int = 4,
    scan_size: int | None = None,
    verbose: bool = True,
    skip_mps_memory_check: bool | None = None,
    validate_master: Callable[[str], None] | None = None,
) -> LazyMacbookDatasets:
    """Decode dataset 0, return a lazy handle over all N (MPS only).

    ``masters`` is either a folder (every ``*_master.h5`` in it is discovered +
    sorted, no hardcoding) or an explicit list of master paths. ``scan_size``
    (e.g. 512 or 256) keeps only masters whose scan is that NxN size - a mixed
    folder holding both 512 and 256 acquisitions is filtered to one, so the 5D
    stack is uniform. Reads HDF5 headers only, no decode, for discovery.
    """
    from quantem.widget.io import discover_masters, load
    # MPS imports kept inside this function so CUDA / CPU never pull pyobjc Metal.
    from quantem.gpu.compute.mps import ChunkedFrames, MultiChunkedFrames

    # folder -> auto-discover (optionally filtered to one scan size); list -> as given
    if isinstance(masters, (str, os.PathLike)) and os.path.isdir(os.path.expanduser(str(masters))):
        scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
        masters = discover_masters(os.path.expanduser(str(masters)),
                                   scan_shape=scan_shape, verbose=False)
    masters = [str(m) for m in masters]
    n = len(masters)
    if n == 0:
        raise ValueError("no master files found")
    names = [os.path.basename(m)[:-len("_master.h5")]
             if m.endswith("_master.h5") else os.path.basename(m) for m in masters]

    def _decode(path):
        # load() returns a LoadResult(data, meta); data is the MPSChunked4DSTEM
        # (chunks + metadata). Wrap in the compute container so MultiChunkedFrames
        # sees a uniform ChunkedFrames.
        data, _meta = load(
            path,
            backend="mps",
            det_bin=det_bin,
            verbose=False,
            skip_mps_memory_check=skip_mps_memory_check,
        )
        row_prefix = bool(getattr(data, "row_prefix", False)
                          or getattr(data, "metadata", {}).get("row_prefix", False))
        return ChunkedFrames(data, row_prefix=row_prefix)

    if verbose:
        print(f"[1/{n}] loading {names[0]} ...", flush=True)
    t0 = time.perf_counter()
    ds0 = _decode(masters[0])
    if verbose:
        print(f"[1/{n}] {names[0]} ready in {time.perf_counter() - t0:.1f}s", flush=True)
    multi = MultiChunkedFrames([ds0], n_total=n, names=names)
    return LazyMacbookDatasets(
        masters,
        det_bin,
        names,
        multi,
        _decode,
        verbose=verbose,
        validate_master=validate_master,
    )


def load_4dstem_macbook(
    masters,
    *,
    det_bin: int = 4,
    scan_size: int | None = None,
    verbose: bool = True,
    skip_mps_memory_check: bool | None = None,
    **viewer_kwargs,
):
    """Convenience wrapper: build the MPS lazy handle AND return a mounted Show4DSTEM viewer.

    Same discovery + decode behavior as :func:`load_macbook_datasets`, but
    additionally hands the returned :class:`LazyMacbookDatasets` to
    :func:`Show4DSTEM` so a caller who wants "one line, see it now" doesn't have
    to construct the viewer separately. Extra keyword arguments are forwarded to
    the viewer.
    """
    from quantem.widget import Show4DSTEM
    lazy = load_macbook_datasets(
        masters,
        det_bin=det_bin,
        scan_size=scan_size,
        verbose=verbose,
        skip_mps_memory_check=skip_mps_memory_check,
    )
    return Show4DSTEM(lazy, **viewer_kwargs)
