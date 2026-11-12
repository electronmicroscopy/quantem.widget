"""Live notebook resource monitoring and explicit scan lifecycle controls."""

import math
import os
import threading
import time
import weakref
from pathlib import Path
from typing import Literal

import anywidget
import psutil
import traitlets


def sample(*, users: bool = True) -> dict:
    """Collect driver telemetry and host memory, without allocating GPU arrays."""
    try:
        from quantem.gpu.device.telemetry import status
    except ImportError:
        snapshot = {
            "gpus": [],
            "error": "GPU telemetry needs a quantem.gpu version with device.status().",
            "timestamp": time.time(),
            "query_ms": 0.0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    else:
        snapshot = status(processes=users)
    if users and "processes" in snapshot:
        kernel = psutil.Process()
        protected = {os.getpid(), *(parent.pid for parent in kernel.parents())}
        username = kernel.username()
        identities = {}
        groups = {}
        for process in snapshot.pop("processes"):
            pid = process["pid"]
            if pid not in identities:
                try:
                    identities[pid] = psutil.Process(pid).as_dict(
                        attrs=["username", "name", "create_time"], ad_value=None
                    )
                except (psutil.Error, OSError):
                    identities[pid] = {"username": None, "name": None}
            identity = identities[pid]
            key = (process["gpu_uuid"], identity["username"])
            group = groups.setdefault(
                key,
                {
                    "name": identity["username"],
                    "used_bytes": 0,
                    "processes": 0,
                    "process_details": [],
                },
            )
            group["processes"] += 1
            group["process_details"].append(
                {
                    "pid": pid,
                    "name": identity["name"],
                    "used_bytes": process["used_bytes"],
                    "created_at": identity.get("create_time"),
                    "can_terminate": (pid not in protected and pid > 1 and
                                      identity.get("create_time") is not None and
                                      identity["username"] == username),
                }
            )
            if process["used_bytes"] is None or group["used_bytes"] is None:
                group["used_bytes"] = None
            else:
                group["used_bytes"] += process["used_bytes"]
        for group in groups.values():
            group["process_details"].sort(key=lambda process: process["pid"])
        for gpu in snapshot["gpus"]:
            gpu["users"] = sorted(
                (group for (uuid, _), group in groups.items() if uuid == gpu["uuid"]),
                key=lambda group: (group["name"] is None, group["name"] or ""),
            )
    memory = psutil.virtual_memory()
    snapshot["system"] = {
        "total_bytes": memory.total,
        "available_bytes": memory.available,
        "used_bytes": memory.total - memory.available,
        "kernel_rss_bytes": psutil.Process().memory_info().rss,
    }
    return snapshot


def poll(reference: weakref.ReferenceType, stop: threading.Event) -> None:
    """Sample independently of the kernel execution loop, without retaining it."""
    while not stop.is_set():
        monitor = reference()
        if monitor is None:
            return
        interval = monitor.interval
        del monitor
        if stop.wait(interval):
            return
        monitor = reference()
        if monitor is None:
            return
        if monitor.running:
            monitor.refresh()
        del monitor


class ShowGPU(anywidget.AnyWidget):
    """Monitor live GPU and system memory on the notebook's server.

    Readings continue from a background thread while notebook code runs.
    Sampling never allocates GPU memory, synchronizes compute, releases
    caches or changes device selection. Driver rows include other processes.
    Explicit controls can unload watched scans or terminate an owned process.
    A timestamp and stale indicator distinguish live readings from saved output.

    Parameters
    ----------
    users : bool, default True
        Show compute users, their reported VRAM and process counts per GPU.
        Expand a user to inspect process names, IDs and individual VRAM use.
        Disable to avoid querying process ownership. Graphics and driver
        allocations remain included in device totals, not this breakdown.
    expand : bool, default True
        Start user process lists expanded. Each user row can still be opened
        or closed individually in the widget.
    size : {"small", "medium", "large", "xlarge"}, default "medium"
        Text and control size. Cards wrap to fit the notebook width. The
        widget's S/M/L/XL buttons change this without rerunning the cell.
    interval : float, default 1.0
        Seconds between driver samples, between 0.2 and 60. Polling is bounded
        and consecutive queries never overlap.
    running : bool, default True
        Start automatic sampling. Pause and Resume are also available in the
        widget; Refresh takes one sample while paused.

    Notes
    -----
    GPU telemetry currently supports NVIDIA drivers through quantem.gpu.
    System RAM remains available without NVIDIA. Physical driver indices may
    differ from CUDA logical indices; the compute visibility mask is shown.
    Close the widget when finished, or before replacing it in a rerun cell.
    A native extension holding Python's GIL can delay background sampling;
    stale readings are visibly marked. Browser commands may wait while the
    kernel is executing a cell.

    Examples
    --------
    >>> from quantem.widget import ShowGPU
    >>> monitor = ShowGPU(users=True, expand=True, size="medium")
    >>> monitor
    >>> monitor.close()  # stop sampling when finished
    """

    _esm = Path(__file__).parent / "static" / "showgpu.js"
    snapshot = traitlets.Dict().tag(sync=True)
    action_status = traitlets.Dict().tag(sync=True)
    interval = traitlets.Float(1.0).tag(sync=True)
    running = traitlets.Bool(True).tag(sync=True)
    users = traitlets.Bool(True).tag(sync=True)
    expand = traitlets.Bool(True).tag(sync=True)
    size = traitlets.Enum(
        ["small", "medium", "large", "xlarge"], default_value="medium"
    ).tag(sync=True)

    def __init__(
        self,
        *,
        users: bool = True,
        expand: bool = True,
        size: Literal["small", "medium", "large", "xlarge"] = "medium",
        interval: float = 1.0,
        running: bool = True,
    ) -> None:
        self._stop = threading.Event()
        self._sample_lock = threading.Lock()
        self._sequence = 0
        self._worker = None
        self._recording = self._viewer = None
        self._stop_acquisition = None
        self._watch_id = 0
        self._action_thread = None
        self._watch_lock = threading.RLock()
        self._scan_states = {}
        self._activity = []
        self._terminating = {}
        super().__init__(
            users=users, expand=expand, size=size, interval=interval, running=running
        )
        self.on_msg(self.handle_message)
        self.refresh()
        self._worker = threading.Thread(
            target=poll,
            args=(weakref.ref(self), self._stop),
            name="ShowGPU telemetry",
            daemon=True,
        )
        weakref.finalize(self, self._stop.set)
        self._worker.start()

    def watch(self, recording, *, viewer=None, stop=None) -> None:
        """Show the scan inventory of an acquisition already started by the user.

        Parameters
        ----------
        recording : object or None
            Recording or acquisition controller exposing ``memory_snapshot()``
            and ``unload(scan_id)``. None detaches it without releasing data.
        viewer : Show4DSTEM, optional
            Its streaming view updates after a scan is unloaded.
        stop : callable, optional
            Stop the acquisition trigger before closing the recording, for
            example ``replay.close``. Runs only after an explicit Stop action.

        Examples
        --------
        >>> monitor.watch(run, viewer=viewer, stop=replay.close)
        """
        if recording is not None and not (callable(getattr(recording, "memory_snapshot", None))
                                          and callable(getattr(recording, "unload", None))):
            raise TypeError("Watch a live Recording or acquisition controller with memory_snapshot() and unload().")
        with self._watch_lock:
            self._watch_id += 1
            self._recording = None if recording is None else weakref.ref(recording)
            self._viewer = None if viewer is None else weakref.ref(viewer)
            self._stop_acquisition = stop
            self._scan_states = {}
        self.action_status = {}
        self.refresh()

    @traitlets.validate("interval")
    def validate_interval(self, proposal):
        value = proposal["value"]
        if not math.isfinite(value) or not 0.2 <= value <= 60:
            raise traitlets.TraitError(
                "interval must be 0.2 to 60 seconds; use interval=1.0."
            )
        return value

    @traitlets.observe("users")
    def users_changed(self, change) -> None:
        """Refresh the ownership display even when automatic polling is paused."""
        if self._worker is not None:
            self.refresh()

    def refresh(self) -> dict:
        """Read one current sample, also when automatic sampling is paused.

        Returns
        -------
        dict
            Latest GPU and system-memory snapshot. Concurrent requests share
            the existing in-flight sample instead of starting another query.

        Examples
        --------
        >>> monitor.refresh()["system"]["available_bytes"]
        """
        if self._stop.is_set() or not self._sample_lock.acquire(blocking=False):
            return self.snapshot
        try:
            try:
                snapshot = sample(users=self.users)
            except (OSError, RuntimeError, psutil.Error) as exc:
                # Report a failed sample and retry on the next interval. Never
                # leave an old successful reading labelled as current.
                snapshot = {
                    "gpus": [],
                    "system": {},
                    "timestamp": time.time(),
                    "error": f"Telemetry unavailable: {exc}",
                    "query_ms": 0.0,
                }
            self._sequence += 1
            snapshot["sequence"] = self._sequence
            with self._watch_lock:
                recording = None if self._recording is None else self._recording()
                watch_id = self._watch_id
                viewer = None if self._viewer is None else self._viewer()
            if recording is not None:
                try:
                    snapshot["recording"] = recording.memory_snapshot()
                except Exception as exc:  # noqa: BLE001 - expose provider failures without losing driver readings
                    snapshot["recording"] = {"scans": [], "error": str(exc)}
                snapshot["recording"]["watch_id"] = watch_id
                snapshot["recording"]["can_select"] = viewer is not None
            with self._watch_lock:
                report = snapshot.get("recording", {})
                if watch_id != self._watch_id:
                    snapshot.pop("recording", None)
                elif not report.get("error"):
                    rows = [*report.get("scans", []), *report.get("releasing", [])]
                    states = {row["id"]: (row.get("name", row["id"]), row["state"]) for row in rows}
                    for identifier, (name, state) in states.items():
                        if self._scan_states.get(identifier) != (name, state):
                            self._record_activity(f"Scan {name}: {state.lower()}.")
                    for identifier, (name, state) in self._scan_states.items():
                        if identifier not in states:
                            message = "viewer buffers released" if state == "Waiting for viewers" else "no longer retained"
                            self._record_activity(f"Scan {name}: {message}.")
                    self._scan_states = states
                for pid, created_at in list(self._terminating.items()):
                    try:
                        process = psutil.Process(pid)
                        exited = (process.create_time() != created_at or
                                  process.status() in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD})
                    except psutil.NoSuchProcess:
                        exited = True
                    except psutil.Error:
                        continue  # Keep the request pending until exit can be verified.
                    if exited:
                        del self._terminating[pid]
                        message = f"PID {pid} exited."
                        self._record_activity(message)
                        if self.action_status.get("pid") == pid:
                            self.action_status = {"message": message}
                snapshot["terminating"] = list(self._terminating)
                snapshot["activity"] = list(self._activity)
            if not self._stop.is_set():
                self.snapshot = snapshot
            return self.snapshot
        finally:
            self._sample_lock.release()

    def _record_activity(self, message: str) -> None:
        """Keep a bounded history of observed changes without retaining source objects."""
        with self._watch_lock:
            self._activity.append({"timestamp": time.time(), "message": message})
            del self._activity[:-32]

    def handle_message(self, widget, content: dict, buffers) -> None:
        """Apply explicit, identified actions to the current watched resources."""
        if self._stop.is_set():
            return
        action = content.get("action")
        if action == "refresh":
            self.refresh()
            return
        if self._action_thread is not None and self._action_thread.is_alive():
            return
        try:
            if action == "terminate":
                self.terminate_process(content.get("pid"), content.get("created_at"))
                with self._watch_lock:
                    self._terminating[content["pid"]] = content["created_at"]
                    message = f"Termination requested for PID {content['pid']}; waiting for exit."
                    self.action_status = {"message": message, "pid": content["pid"]}
                    self._record_activity(message)
            elif action in {"unload", "stop_acquisition", "select_scan"}:
                recording = None if self._recording is None else self._recording()
                if recording is None or content.get("watch_id") != self._watch_id:
                    raise ValueError("The watched acquisition changed; refresh and select it again.")
                if action == "select_scan":
                    viewer = None if self._viewer is None else self._viewer()
                    if viewer is not None:
                        viewer.refresh_stream(content.get("scan_id"))
                elif action == "unload":
                    recording.unload(content.get("scan_id"))
                    viewer = None if self._viewer is None else self._viewer()
                    if viewer is not None:
                        viewer.refresh_stream()
                    self.action_status = {"message": "Scan removed from the acquisition. Viewers may still hold its buffers; driver VRAM also includes allocator reserves."}
                else:
                    self.action_status = {"message": "Stopping acquisition and draining pending work...", "busy": True}
                    self._record_activity("Acquisition stop requested; draining pending work.")
                    stop, watch_id = self._stop_acquisition, self._watch_id

                    def finish():
                        try:
                            if stop is not None:
                                stop()
                            recording.close()
                            while recording.receiving and not self._stop.wait(0.2):
                                if self._watch_id != watch_id:
                                    return
                            if self._stop.is_set():
                                return
                            status = {"message": "Acquisition stopped; retained scans remain available."}
                        except Exception as exc:  # noqa: BLE001 - report user-controller errors in the widget
                            status = {"message": str(exc), "error": True}
                        with self._watch_lock:
                            if self._watch_id == watch_id:
                                self.action_status = status
                                self._record_activity(status["message"])
                        self.refresh()

                    self._action_thread = threading.Thread(target=finish, name="ShowGPU stop acquisition", daemon=True)
                    self._action_thread.start()
        except Exception as exc:  # noqa: BLE001 - report explicit action failures to their requester
            self.action_status = {"message": str(exc), "error": True}
        self.refresh()

    def terminate_process(self, pid: int, created_at: float) -> None:
        """Request SIGTERM for a displayed process owned by this operating-system user.

        The notebook kernel and its ancestors are protected. Creation time is
        checked immediately before signalling so a reused PID is not targeted.

        Examples
        --------
        >>> monitor.terminate_process(process["pid"], process["created_at"])
        """
        shown = [process for gpu in self.snapshot.get("gpus", [])
                 for user in gpu.get("users", []) for process in user.get("process_details", [])]
        if not any(item["pid"] == pid and item.get("created_at") == created_at
                   and item.get("can_terminate") for item in shown):
            raise ValueError("Select an owned process from the current live sample; this process cannot be terminated here.")
        process = psutil.Process(pid)
        protected = {os.getpid(), *(parent.pid for parent in psutil.Process().parents())}
        if (pid in protected or pid <= 1 or process.username() != psutil.Process().username()
                or process.create_time() != created_at):
            raise ValueError("The process identity or ownership changed; refresh before trying again.")
        process.terminate()

    def close(self) -> None:
        """Stop automatic sampling and close the notebook display.

        Examples
        --------
        >>> monitor.close()
        """
        if hasattr(self, "_stop"):
            self._stop.set()
        worker = getattr(self, "_worker", None)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=4.5)
        if hasattr(self, "_watch_lock"):
            with self._watch_lock:
                self._watch_id += 1
                self._recording = self._viewer = self._stop_acquisition = None
        super().close()
