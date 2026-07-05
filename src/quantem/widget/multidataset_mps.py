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

import os
import threading
import time


class LazyMacbookDatasets:
    """MPS lazy multi-dataset handle returned by ``load([masters])``.

    Holds dataset 0 (already decoded) plus the spec to decode 1..N-1 on demand.
    :func:`quantem.widget.Show4DSTEM` consumes it: builds the 5D viewer over the
    live :class:`MultiChunkedFrames` container, then starts one background worker
    that fills the remaining datasets. Browsing is instant; the slider only spans
    decoded datasets and grows as each lands.
    """

    def __init__(self, masters, det_bin, names, multi, decode, verbose=True):
        self.masters = masters
        self.det_bin = det_bin
        self.names = names
        self.multi = multi  # MultiChunkedFrames([ds0], n_total=N, names=names)
        self._decode = decode
        self.verbose = bool(verbose)
        self._watch_stop: threading.Event | None = None
        self._watch_thread: threading.Thread | None = None

    def build_viewer(self, **viewer_kwargs):
        """Show dataset 0 now, fill 1..N-1 in a daemon thread."""
        from quantem.widget.show4dstem_mps import Show4DSTEM_MACBOOK
        verbose = bool(viewer_kwargs.pop("verbose", self.verbose))
        viewer_kwargs.setdefault("frame_dim_label", "Dataset")
        viewer_kwargs.setdefault("frame_labels", list(self.names))
        viewer = Show4DSTEM_MACBOOK(self.multi, verbose=verbose, **viewer_kwargs)
        n = len(self.masters)
        if n > 1:
            def _worker():
                for i in range(1, n):
                    try:
                        if verbose:
                            print(f"[{i + 1}/{n}] loading {self.names[i]} ...", flush=True)
                        t = time.perf_counter()
                        self.multi.set_dataset(i, self._decode(self.masters[i]))
                        if verbose:
                            print(f"[{i + 1}/{n}] {self.names[i]} ready in "
                                  f"{time.perf_counter() - t:.1f}s", flush=True)
                    except Exception as exc:
                        if verbose:
                            print(f"[{i + 1}/{n}] {self.names[i]} FAILED: {str(exc)[:80]}",
                                  flush=True)
            threading.Thread(target=_worker, daemon=True).start()
        return viewer

    def append_master(self, master, *, name: str | None = None, async_: bool = True) -> int:
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
        path = str(master)
        label = str(name) if name is not None else os.path.basename(path)
        if label.endswith("_master.h5"):
            label = label[:-len("_master.h5")]

        idx = len(self.masters)
        self.masters.append(path)
        self.names.append(label)

        def _decode_and_append():
            try:
                if self.verbose:
                    print(f"[append {idx + 1}] loading {label} ...", flush=True)
                t = time.perf_counter()
                self.multi.append_dataset(self._decode(path), name=label)
                if self.verbose:
                    print(f"[append {idx + 1}] {label} ready in "
                          f"{time.perf_counter() - t:.1f}s", flush=True)
            except Exception as exc:
                if self.verbose:
                    print(f"[append {idx + 1}] {label} FAILED: {str(exc)[:80]}",
                          flush=True)
                raise

        if async_:
            threading.Thread(
                target=_decode_and_append,
                name="Show4DSTEMMPS-append-master",
                daemon=True,
            ).start()
        else:
            _decode_and_append()
        return idx

    @staticmethod
    def _master_key(master) -> str:
        return os.path.abspath(os.path.expanduser(str(master)))

    def append_new_masters(self, masters, *, async_: bool = True) -> list[int]:
        """Append only masters that are not already present in this live handle.

        This is the safe inner loop for microscope/live-folder workflows: callers
        can repeatedly pass the current discovered master list, and already loaded
        acquisitions are skipped without rebuilding the viewer.
        """
        known = {self._master_key(master) for master in self.masters}
        added: list[int] = []
        for master in masters:
            key = self._master_key(master)
            if key in known:
                continue
            added.append(self.append_master(key, async_=async_))
            known.add(key)
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
    ) -> list[int]:
        """Discover ready master files in *folder* and append new acquisitions.

        Parameters mirror :func:`quantem.widget.io.discover_masters`. When
        ``ready_only`` is true, partially written masters are ignored until their
        linked data files are present.
        """
        from quantem.widget.io import discover_masters, is_master_ready

        scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
        masters = discover_masters(
            os.path.expanduser(str(folder)),
            pattern=pattern,
            recursive=recursive,
            scan_shape=scan_shape,
            verbose=False,
        )
        if ready_only:
            masters = [master for master in masters if is_master_ready(master)]
        return self.append_new_masters(masters, async_=async_)

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
        self.stop_watch()
        stop = threading.Event()
        self._watch_stop = stop

        def _worker() -> None:
            while not stop.wait(float(interval)):
                try:
                    added = self.poll_master_folder(
                        folder,
                        pattern=pattern,
                        recursive=recursive,
                        scan_size=scan_size,
                        ready_only=ready_only,
                        async_=async_,
                    )
                    if self.verbose and added:
                        print(f"[watch] appended {len(added)} new master(s)", flush=True)
                except ValueError:
                    # Empty folders are normal at the start of a live session.
                    continue
                except Exception as exc:
                    if self.verbose:
                        print(f"[watch] master folder poll failed: {str(exc)[:120]}",
                              flush=True)

        self._watch_thread = threading.Thread(
            target=_worker,
            name="Show4DSTEMMPS-watch-master-folder",
            daemon=True,
        )
        self._watch_thread.start()
        return self

    def stop_watch(self) -> None:
        """Stop the live master-folder watcher if one is running."""
        stop = getattr(self, "_watch_stop", None)
        if stop is not None:
            stop.set()
        self._watch_stop = None
        self._watch_thread = None


def load_macbook_datasets(masters, *, det_bin: int = 4, scan_size: int | None = None,
                          verbose: bool = True) -> LazyMacbookDatasets:
    """Decode dataset 0, return a lazy handle over all N (MPS only).

    ``masters`` is either a folder (every ``*_master.h5`` in it is discovered +
    sorted, no hardcoding) or an explicit list of master paths. ``scan_size``
    (e.g. 512 or 256) keeps only masters whose scan is that NxN size - a mixed
    folder holding both 512 and 256 acquisitions is filtered to one, so the 5D
    stack is uniform. Reads HDF5 headers only, no decode, for discovery.
    """
    from quantem.widget.io import discover_masters, load
    # MPS imports kept inside this function so CUDA / CPU never pull pyobjc Metal.
    from quantem.widget.kernels.compute.mps import ChunkedFrames, MultiChunkedFrames

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
        data, _meta = load(path, backend="mps", det_bin=det_bin, verbose=False)
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
    return LazyMacbookDatasets(masters, det_bin, names, multi, _decode, verbose=verbose)


def load_4dstem_macbook(masters, *, det_bin: int = 4, scan_size: int | None = None,
                        verbose: bool = True, **viewer_kwargs):
    """Convenience wrapper: build the MPS lazy handle AND return a mounted Show4DSTEM viewer.

    Same discovery + decode behavior as :func:`load_macbook_datasets`, but
    additionally hands the returned :class:`LazyMacbookDatasets` to
    :func:`Show4DSTEM` so a caller who wants "one line, see it now" doesn't have
    to construct the viewer separately. Extra keyword arguments are forwarded to
    the viewer.
    """
    from quantem.widget import Show4DSTEM
    lazy = load_macbook_datasets(
        masters, det_bin=det_bin, scan_size=scan_size, verbose=verbose,
    )
    return Show4DSTEM(lazy, **viewer_kwargs)
