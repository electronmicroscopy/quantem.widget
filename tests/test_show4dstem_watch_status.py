"""Focused live-folder state tests for the Dataset5dstem viewer path."""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import traitlets

from quantem.widget.data import Dataset5dstem
from quantem.widget.io import MasterReadiness
from quantem.widget.show4dstem import Show4DSTEM


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true before timeout")


def _report(
    *,
    ready: bool,
    revision: str,
    reason: str = "complete",
    action: str = "Ready to open with Show4DSTEM.",
    actual_frames: int | None = 1,
    expected_frames: int | None = 1,
) -> MasterReadiness:
    return MasterReadiness(
        ready=ready,
        reason=reason,
        action=action,
        source_kind="external",
        actual_frames=actual_frames,
        expected_frames=expected_frames,
        detector_shape=(2, 2),
        dtype="<u2",
        source_signature={"revision": revision},
    )


def _folder_widget(tmp_path: Path, *, view_mode: str = "single"):
    frame = torch.ones((1, 1, 2, 2), dtype=torch.uint16)
    initial = tmp_path / "scan_00_master.h5"
    materialized: list[str] = []
    incompatible: set[str] = set()

    data = Dataset5dstem.from_lazy_loaders(
        [lambda: frame],
        shape=(1, 1, 1, 2, 2),
        dtype=torch.uint16,
        initial_frames={0: frame},
    )
    widget = Show4DSTEM(
        data,
        view_mode=view_mode,
        compare_max_panels=4,
        precompute_virtual_images=False,
        verbose=False,
    )

    def make_loader(master, _idx: int):
        path = str(master)

        def load():
            materialized.append(path)
            return frame

        return Path(path).stem.removesuffix("_master"), load

    def validate(master) -> None:
        if str(master) in incompatible:
            raise ValueError("detector shape (3, 2) does not match (2, 2)")

    widget._attach_folder_source(
        folder=tmp_path,
        pattern="*_master.h5",
        recursive=False,
        scan_shape=(1, 1),
        ready_only=True,
        known_masters=[initial],
        make_loader=make_loader,
        validate_master=validate,
        preload_all_if_fits=False,
        warm_cache=False,
    )
    return widget, initial, materialized, incompatible


def _write_external_master(folder: Path, index: int) -> Path:
    stem = f"scan_{index:03d}"
    chunk = folder / f"{stem}_data_000001.h5"
    master = folder / f"{stem}_master.h5"
    with h5py.File(chunk, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.full((16, 8, 8), index + 1, dtype=np.uint16),
        )
    with h5py.File(master, "w") as handle:
        group = handle.require_group("entry/data")
        group["data_000001"] = h5py.ExternalLink(
            chunk.name,
            "entry/data/data",
        )
        detector = handle.require_group(
            "entry/instrument/detector/detectorSpecific"
        )
        detector.create_dataset("ntrigger", data=16)
        detector.create_dataset("nimages", data=1)
    return master


def test_show4dstem_watch_requires_stable_ready_signature(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import quantem.widget.io as widget_io

    widget, initial, materialized, _ = _folder_widget(tmp_path)
    arriving = tmp_path / "scan_01_master.h5"
    discovered = [str(initial), str(arriving)]
    current = {
        "report": _report(
            ready=False,
            revision="partial",
            reason="stored frame count is 0; expected 1",
            action="Wait for the remaining detector frames to finish writing.",
            actual_frames=0,
        )
    }
    scan_shapes: list[tuple[int, int] | None] = []

    monkeypatch.setattr(
        widget_io,
        "discover_masters",
        lambda *args, **kwargs: list(discovered),
    )

    def inspect(_master, *, scan_shape=None):
        scan_shapes.append(scan_shape)
        return current["report"]

    monkeypatch.setattr(widget_io, "inspect_master_readiness", inspect)
    transitions: list[str] = []
    widget.observe(
        lambda change: transitions.append(change["new"]),
        names="folder_watch_state",
    )
    widget.watch_folder(interval=60)
    try:
        # C1: a live worker is green but a full serialized model cannot retain
        # that daemon, expect its snapshot to say Stopped without mutating the
        # live trait used by the mounted frontend.
        snapshot = widget.get_state()
        assert widget.folder_watch_state == "watching"
        assert snapshot["folder_watch_state"] == "stopped"
        assert "Re-run the cell" in snapshot["folder_watch_detail"]
        widget._initial_live_mount_state = True
        try:
            assert widget.get_state()["folder_watch_state"] == "watching"
        finally:
            widget._initial_live_mount_state = False
        with pytest.raises(traitlets.TraitError):
            widget.folder_watch_state = "false-green"

        # C2: a discovered master is incomplete, expect an amber waiting state
        # with corrective detail and no lazy slot append.
        assert widget.poll_folder() == []
        assert widget.folder_watch_state == "waiting"
        assert "stored frame count is 0" in widget.folder_watch_detail
        assert widget.n_frames == 1

        # C3: a complete signature changes between polls, expect probation to
        # restart and the source to remain unrepresented.
        current["report"] = _report(ready=True, revision="ready-a")
        assert widget.poll_folder() == []
        current["report"] = _report(ready=True, revision="ready-b")
        assert widget.poll_folder() == []
        assert "unchanged follow-up" in widget.folder_watch_detail

        # C4: the same complete signature appears again, expect exactly one cold
        # lazy append and a green state only after the successful update.
        assert widget.poll_folder() == [1]
        assert widget.poll_folder() == []
        assert widget.n_frames == 2
        assert widget._data.loaded_indices() == [0]
        assert materialized == []
        assert widget.folder_watch_state == "watching"
        assert "updating" in transitions
        assert scan_shapes and set(scan_shapes) == {(1, 1)}
    finally:
        widget.close()


def test_show4dstem_bad_candidate_does_not_block_later_compatible(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import quantem.widget.io as widget_io

    widget, initial, materialized, incompatible = _folder_widget(tmp_path)
    bad = tmp_path / "scan_01_master.h5"
    good = tmp_path / "scan_02_master.h5"
    incompatible.add(str(bad))
    reports = {
        str(bad): _report(ready=True, revision="bad-stable"),
        str(good): _report(ready=True, revision="good-stable"),
    }
    monkeypatch.setattr(
        widget_io,
        "discover_masters",
        lambda *args, **kwargs: [str(initial), str(bad), str(good)],
    )
    monkeypatch.setattr(
        widget_io,
        "inspect_master_readiness",
        lambda master, **kwargs: reports[str(master)],
    )
    widget.watch_folder(interval=60)
    try:
        # C1: two complete candidates first enter probation, expect neither to
        # append until a second identical readiness observation.
        assert widget.poll_folder() == []
        assert widget.folder_watch_state == "waiting"

        # C2: the earlier candidate violates the established data contract,
        # expect a red corrective state while the later compatible master still
        # appends as a cold slot in the same poll.
        assert widget.poll_folder() == [1]
        assert widget.n_frames == 2
        assert list(widget.frame_labels)[-1] == "scan_02"
        assert widget.folder_watch_state == "error"
        assert "detector shape" in widget.folder_watch_detail
        assert materialized == []

        # C3: correcting the contract makes the retained stable candidate
        # retryable, expect one append and a return to Watching.
        incompatible.clear()
        assert widget.poll_folder() == [2]
        assert widget.poll_folder() == []
        assert widget.n_frames == 3
        assert widget.folder_watch_state == "watching"
        assert materialized == []
    finally:
        widget.close()


def test_show4dstem_watch_error_restart_stop_and_close_lifecycle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import quantem.widget.io as widget_io

    widget, initial, _, _ = _folder_widget(tmp_path)
    failing = {"value": True}

    def discover(*args, **kwargs):
        if failing["value"]:
            raise RuntimeError(f"acquisition mount unavailable at {tmp_path}")
        return [str(initial)]

    monkeypatch.setattr(widget_io, "discover_masters", discover)
    first = None
    second = None
    try:
        # C1: an unexpected discovery failure occurs in a live worker, expect
        # a red actionable state without terminating the retry loop.
        widget.watch_folder(interval=0.01)
        first = widget._folder_watch_thread
        _wait_until(lambda: widget.folder_watch_state == "error")
        assert first is not None and first.is_alive()
        assert "acquisition mount unavailable" in widget.folder_watch_detail
        assert str(tmp_path) not in widget.folder_watch_detail

        # C2: storage recovers on the next cycle, expect the same live worker to
        # clear the false-red state and report Watching.
        failing["value"] = False
        _wait_until(lambda: widget.folder_watch_state == "watching")
        assert widget._folder_watch_thread is first

        # C3: an invalid restart interval is rejected before stopping the good
        # worker, then repeated stop calls remain safe and publish Stopped.
        with pytest.raises(ValueError, match="finite value > 0"):
            widget.watch_folder(interval=0)
        assert widget._folder_watch_thread is first and first.is_alive()
        widget.stop_folder_watch()
        widget.stop_folder_watch()
        assert not first.is_alive()
        assert widget.folder_watch_state == "stopped"

        # C4: restart creates a fresh live worker; repeated close calls stop it
        # without leaving a false green badge or a background thread.
        widget.watch_folder(interval=60)
        second = widget._folder_watch_thread
        assert second is not None and second is not first and second.is_alive()
        assert widget.folder_watch_state == "watching"
        widget.close()
        widget.close()
        assert not second.is_alive()
        assert widget._folder_watch_thread is None
        assert widget.folder_watch_state == "stopped"
    finally:
        widget.stop_folder_watch()
        widget.close()


def test_show4dstem_fixed_folder_snapshot_keeps_watch_status_hidden(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import quantem.widget.io as widget_io

    widget, initial, _, _ = _folder_widget(tmp_path)
    monkeypatch.setattr(
        widget_io,
        "discover_masters",
        lambda *args, **kwargs: [str(initial)],
    )
    try:
        # C1: a fixed snapshot polls without ever starting a worker, expect no
        # live-watch badge and no false Stopped state.
        assert widget.folder_watch_state == "hidden"
        assert widget.poll_folder() == []
        widget.stop_folder_watch()
        assert widget.folder_watch_state == "hidden"
        assert widget.folder_watch_detail == ""
    finally:
        widget.close()


def test_show4dstem_live_append_paints_active_partial_page_before_green(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import quantem.widget.io as widget_io

    widget, initial, materialized, _ = _folder_widget(
        tmp_path,
        view_mode="multiple",
    )
    arriving = tmp_path / "scan_01_master.h5"
    monkeypatch.setattr(
        widget_io,
        "discover_masters",
        lambda *args, **kwargs: [str(initial), str(arriving)],
    )
    monkeypatch.setattr(
        widget_io,
        "inspect_master_readiness",
        lambda master, **kwargs: _report(ready=True, revision="stable"),
    )
    transitions: list[str] = []
    widget.observe(
        lambda change: transitions.append(change["new"]),
        names="folder_watch_state",
    )
    widget.watch_folder(interval=60)
    try:
        # C1: a complete master enters probation, expect the shared protocol to
        # show Updating during discovery, then amber while awaiting confirmation.
        assert widget.poll_folder() == []
        assert widget.folder_watch_state == "waiting"
        assert "updating" in transitions
        assert transitions[-1] == "waiting"
        assert materialized == []

        # C2: the stable follow-up is accepted on the visible partial page,
        # expect the progressive worker to materialize and publish both slots
        # before the watcher returns to green.
        assert widget.poll_folder() == [1]
        widget.wait_for_compare_page(timeout=2)
        assert widget.compare_page_loading is False
        assert widget.compare_panel_indices == [0, 1]
        assert len(widget.compare_virtual_image_bytes) == 2 * 1 * 1 * 4
        assert materialized == [str(arriving)]
        assert widget.folder_watch_state == "watching"
        assert "updating" in transitions
        assert transitions.index("updating") < len(transitions) - 1
        assert transitions[-1] == "watching"
    finally:
        widget.close()


def test_public_show4dstem_from_folder_paints_real_external_arrival(
    tmp_path: Path,
) -> None:
    from quantem.widget import Show4DSTEM as PublicShow4DSTEM

    _write_external_master(tmp_path, 0)
    widget = PublicShow4DSTEM.from_folder(
        tmp_path,
        backend="cpu",
        scan_size=4,
        det_bin=1,
        dtype="u8",
        page_budget=1,
        page_size=4,
        view_mode="multiple",
        watch=True,
        watch_interval=60,
        preload_all_if_fits=False,
        warm_cache=False,
        preview_cache=False,
        precompute_virtual_images=False,
        verbose=False,
    )
    model_id = widget.model_id
    transitions: list[str] = []
    widget.observe(
        lambda change: transitions.append(change["new"]),
        names="folder_watch_state",
    )
    try:
        _write_external_master(tmp_path, 1)

        # C1: a real external-link master first enters header probation, expect
        # the mounted public viewer to remain unchanged and amber.
        assert widget.poll_folder() == []
        assert widget.model_id == model_id
        assert widget.n_frames == 1
        assert widget.folder_watch_state == "waiting"

        # C2: the unchanged header contract is accepted, expect the same public
        # viewer to stream the new active-page tile before returning to green.
        assert widget.poll_folder() == [1]
        widget.wait_for_compare_page(timeout=5)
        assert widget.model_id == model_id
        assert widget.n_frames == 2
        assert list(widget.frame_labels) == ["scan_000", "scan_001"]
        assert widget.compare_panel_indices == [0, 1]
        assert widget.compare_page_loading is False
        assert len(widget.compare_virtual_image_bytes) == 2 * 4 * 4 * 4
        assert transitions[-1] == "watching"
        assert "updating" in transitions
        assert transitions.index("updating") < len(transitions) - 1

        # C3: exporting/saving cannot retain the daemon, expect a gray stopped
        # snapshot without mutating the still-live mounted model.
        snapshot = widget.get_state()
        assert widget.folder_watch_state == "watching"
        assert snapshot["folder_watch_state"] == "stopped"
    finally:
        widget.close()
