"""Internal polling lifecycle for a viewer borrowing live acquisition data."""

import threading
import time
import traceback
from contextlib import nullcontext
from typing import TYPE_CHECKING

from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.adapters.gpu import AcquisitionSeries
from quantem.widget.show4dstem.session import SessionSeries

if TYPE_CHECKING:
    from quantem.widget.show4dstem.widget import Show4DSTEM


# Seconds after the user last touched the viewer before the probe follows the newest pattern again.
STREAM_FOLLOW_PAUSE_S = 1.5


class StreamFollower:
    """Follow one recording without taking ownership of its acquisitions."""

    def __init__(self, viewer: "Show4DSTEM", recording, sources: list, *,
                 labels: list | None, refresh_s: float, options: dict):
        self.viewer = viewer
        self.recording = recording
        self.partial = hasattr(recording, "acquisitions")
        self.lock = getattr(recording, "lock", nullcontext())
        self.shown_sources = sources[:1]
        self.previous = None
        self.initialized = bool(self.shown_sources)
        self.last_refresh = 0.0
        self.labels = labels
        self.refresh_s = refresh_s
        self.options = options
        self.title = options.get("title") or "Live acquisition"

    def snapshot(self) -> list:
        return list(self.recording.acquisitions if self.partial else self.recording.datasets)

    def start(self) -> None:
        """Publish independently of blocking cells; register only this viewer's hooks."""
        from IPython import get_ipython

        viewer = self.viewer
        viewer._stream_update = self.update
        viewer._stream_recording = self.recording
        shell = get_ipython()
        if shell is not None and getattr(shell, "kernel", None) is not None:
            def before_cell(*_args, **_kwargs):
                viewer._stream_cell_running = True

            def after_cell(*_args, **_kwargs):
                viewer._stream_cell_running = False

            viewer._stream_cell_running = shell.kernel.get_parent().get("header", {}).get("msg_type") == "execute_request"
            shell.events.register("pre_run_cell", before_cell)
            shell.events.register("post_run_cell", after_cell)
            viewer._stream_cell_hooks = (shell, before_cell, after_cell)
        viewer._stream_thread = threading.Thread(target=self.follow, name="Show4DSTEM-stream", daemon=True)
        viewer._stream_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Join before releasing any sessions, and detach borrowed recording references."""
        viewer = self.viewer
        viewer._stream_stop.set()
        thread = viewer._stream_thread
        if thread is not None and thread.is_alive():
            if thread is not threading.current_thread():
                thread.join(timeout)
            if thread.is_alive():
                raise RuntimeError("Show4DSTEM is still refreshing. Wait for the query to finish, then call close() again.")
        hooks = viewer._stream_cell_hooks
        if hooks is not None:
            shell, before, after = hooks
            shell.events.unregister("pre_run_cell", before)
            shell.events.unregister("post_run_cell", after)
            viewer._stream_cell_hooks = None
        with viewer._stream_lock:
            viewer._stream_update = None
            viewer._stream_recording = None
            self.shown_sources = []
            self.recording = None
            _release_error_frames(viewer.stream_error)

    def update_state(self, force: bool = False) -> bool:
        """Adopt retained sources and refresh only when counts or topology change."""
        viewer = self.viewer
        if viewer._stream_stop.is_set():
            return False
        try:
            if self.recording.error is not None and not force:
                raise self.recording.error
            # Sample completion first: if it changes during this update,
            # the next update must still publish the final counts.
            complete = not self.recording.receiving
            now = time.monotonic()
            # Full scan images share Jupyter's ordered output channel with
            # the small DP replies. Keep room for direct manipulation;
            # detector gestures still recompute immediately, and the final
            # acquisition refresh is never deferred.
            if (not force and not complete and not viewer._stream_deferred_state
                    and now - viewer._last_interaction_at < 0.15
                    and now - self.last_refresh < 0.1):
                return True
            with self.lock, viewer._defer_stream_observers():
                sources = self.snapshot()
                shown = len(self.shown_sources)
                if not sources:
                    if self.shown_sources:
                        viewer._replace_stream_sources(self.shown_sources, [])
                        self.last_refresh = time.monotonic()
                    self.shown_sources, self.previous = [], None
                    return not complete
                if not self.initialized and self.partial and not sources[0].received:
                    return not complete
                if viewer._data is None:
                    with viewer.hold_sync(), viewer.hold_trait_notifications():
                        viewer._replace_stream_sources([], sources, labels=self.labels)
                        if not self.initialized:
                            viewer._initialize_stream_geometry(sources[0], self.options)
                            self.initialized = True
                    self.previous = None
                elif sources[:shown] == self.shown_sources:
                    if len(sources) > shown:
                        viewer.append(sources[shown:], self.labels[shown:len(sources)] if self.labels else None)
                        self.previous = None
                elif sources and self.partial:
                    viewer._replace_stream_sources(self.shown_sources, sources)
                    self.previous = None
                self.shown_sources = sources
                shown = len(sources)
                if self.partial and isinstance(viewer._data, SessionSeries):
                    # Live sessions remain open as leases until eviction. Their
                    # completed datasets can already use batched GPU queries;
                    # there is no need to wait for continuous acquisition to end.
                    completed = {dataset.metadata.get("acquisition_id"): dataset
                                 for dataset in self.recording.datasets
                                 if gpu_adapter.is_acquisition(dataset) and dataset.metadata.get("acquisition_id")}
                    with viewer._compare_lock:
                        viewer._stream_completed_sources = {
                            idx: completed[source.identifier] for idx, source in enumerate(sources)
                            if getattr(source, "identifier", None) in completed
                        }
            progress = tuple(source.received for source in sources[:shown]) if self.partial else ()
            if self.partial and complete and isinstance(viewer._data, SessionSeries) and len(self.recording.datasets) == shown and all(
                    gpu_adapter.is_acquisition(dataset) for dataset in self.recording.datasets):
                # Borrow the same ANS storage through the regular joint
                # query path once every acquisition is immutable.
                completed = AcquisitionSeries(self.recording.datasets)
                with viewer._compare_lock, viewer._session_lock:
                    viewer._close_sessions()
                    viewer._data = completed
                self.previous = None
            if force or progress != self.previous:
                viewer.refresh()
                if self.partial:
                    current = sources[shown - 1]
                    self.follow_probe(current)
                    scan = current.metadata.get("series", shown)
                    # One short line that never wraps beside a control panel; the counts are in the stats bar.
                    done = " ✓" if current.received == current.total else ""
                    viewer.title = f"Live · {scan} · {current.received:,}/{current.total:,}{done} · {shown} kept"
                else:
                    viewer.title = self.title
                self.previous = progress
                self.last_refresh = time.monotonic()
            return not complete
        except Exception as error:
            viewer.stream_error = error
            viewer.title = f"Live acquisition stopped: {error}"
            if force:
                raise
            return False

    def follow_probe(self, current) -> None:
        """Point the diffraction pattern at the newest arrived pattern of the scan being received, unless the
        user interacted within STREAM_FOLLOW_PAUSE_S: their probe stays until they let go."""
        viewer = self.viewer
        if not viewer.stream_follow or not current.received or current.received >= current.total:
            return
        if time.monotonic() - viewer._last_interaction_at < STREAM_FOLLOW_PAUSE_S:
            return
        row, col = divmod(int(current.received) - 1, int(viewer.shape_cols))
        if (row, col) != (viewer.pos_row, viewer.pos_col) and row < viewer.shape_rows:
            with viewer.hold_trait_notifications():
                viewer.pos_row, viewer.pos_col = row, col

    def update(self, force: bool = False) -> bool:
        """Coalesce display traffic without deferring acquisition retirement."""
        viewer = self.viewer
        with viewer._stream_lock:
            if viewer._stream_stop.is_set():
                return False
            if force:
                self.previous = None
            # Jupyter carries images and interactive replies on the same
            # ordered channel. Bound unacknowledged acquisition images to
            # the credit window. A busy user cell cannot dispatch ACKs.
            now = time.monotonic()
            absent = viewer._stream_display_negotiated and viewer._stream_display_view is None
            outstanding = viewer._stream_display_seq - viewer._stream_display_acknowledged
            waiting = outstanding >= viewer._stream_display_credits and not (
                viewer._stream_cell_running and now - viewer._stream_display_sent_at >= 2.0)
            # Also bound the initial cell before its frontend-ready message
            # can dispatch (for example, display(viewer); recording.wait()).
            busy = viewer._stream_cell_running and self.last_refresh and now - self.last_refresh < 2.0
            if (not force and viewer._stream_display_pending and now - max(
                    viewer._stream_display_sent_at, viewer._stream_display_probe_at) >= 2.0):
                # A final update may occupy only part of the window. A
                # surviving view must still recover a hard-closed owner.
                viewer._stream_display_probe_at = now
                viewer._stream_display_probe_seq = viewer._stream_display_pending
                viewer.send({"type": "show4dstem_stream_probe", "version": 1,
                             "seq": viewer._stream_display_probe_seq})
            if not force and (absent or waiting or busy) and self.recording.error is None:
                # Retention is independent of display delivery. A closed
                # browser must not keep retired acquisitions in GPU memory.
                try:
                    with self.lock:
                        sources = self.snapshot()
                        if any(source not in sources for source in self.shown_sources):
                            if viewer._stream_deferred_snapshot is None:
                                viewer._stream_deferred_snapshot = viewer.get_state(key=viewer.keys)
                            with viewer.hold_sync(), viewer._defer_stream_observers():
                                viewer._replace_stream_sources(self.shown_sources, sources, labels=self.labels)
                                viewer._stream_deferred_state.update(viewer._states_to_send)
                                viewer._states_to_send.clear()
                            self.shown_sources, self.previous = sources, None
                except Exception as error:
                    viewer.stream_error = error
                    viewer.title = f"Live acquisition stopped: {error}"
                    return False
                return True
            before = self.last_refresh
            deferred = bool(viewer._stream_deferred_state)
            old_images = viewer.compare_virtual_image_bytes
            old_revision = viewer.compare_image_revision
            published = viewer._stream_deferred_snapshot
            old_ids = (published or {}).get("_stream_source_ids", viewer._stream_source_ids)
            old_indices = (published or {}).get("compare_panel_indices", viewer.compare_panel_indices)
            old_panel_ids = [old_ids[index] for index in old_indices] if old_ids else []
            old_topology = (tuple(map(id, self.shown_sources)), tuple(viewer.compare_panel_indices),
                            viewer.shape_rows, viewer.shape_cols)
            delta = None
            with viewer.hold_sync():
                receiving = self.update_state(force)
                viewer._states_to_send.update(viewer._stream_deferred_state)
                viewer._stream_deferred_state.clear()
                viewer._stream_deferred_snapshot = None
                topology = (tuple(map(id, self.shown_sources)), tuple(viewer.compare_panel_indices),
                            viewer.shape_rows, viewer.shape_cols)
                if (not force and not deferred and viewer._stream_display_delta
                        and old_topology == topology):
                    delta = viewer._stream_compare_delta(old_images, old_revision)
                elif (not force and viewer._stream_display_views
                      and viewer._stream_display_views.keys() <= viewer._stream_layout_views
                      and old_topology[2:] == topology[2:]):
                    viewer._stream_compare_layout(old_images, old_revision, old_panel_ids)
            if delta is not None:
                message, buffers = delta
                viewer.send(message, buffers=buffers)
            if deferred or self.last_refresh != before:
                viewer._stream_display_barrier()
            return viewer.stream_error is None and (receiving or bool(viewer._stream_display_pending))

    def follow(self) -> None:
        """Keep background failures visible; yield between expensive refreshes."""
        viewer = self.viewer
        try:
            while not viewer._stream_stop.is_set():
                started = time.monotonic()
                if not self.update():
                    break
                viewer._stream_stop.wait(max(0.005, started + self.refresh_s - time.monotonic()))
        except Exception as error:
            viewer.stream_error = error
            viewer.title = f"Live acquisition stopped: {error}"


def _release_error_frames(error: BaseException | None) -> None:
    """Keep failure identities and stack locations without retaining scan locals."""
    pending, seen = [error], set()
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        # Called only after this viewer's worker and explicit refresh have ended.
        # clear_frames also leaves any still-executing receiver frame untouched.
        traceback.clear_frames(error.__traceback__)
        pending.extend((error.__cause__, error.__context__))
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
