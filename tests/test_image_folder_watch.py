from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from quantem.widget import Show2D, Show3D


def _save(path: Path, value: float, shape: tuple[int, int] = (6, 8)) -> None:
    np.save(path, np.full(shape, value, dtype=np.float32))


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before the watcher timeout")


def test_show2d_folder_poll_remaps_panel_state_by_path(tmp_path: Path) -> None:
    _save(tmp_path / "frame_2.npy", 2)
    _save(tmp_path / "frame_10.npy", 10)
    widget = Show2D.from_folder(tmp_path, watch=False, show_fft=True)
    try:
        assert widget.labels == ["frame_2", "frame_10"]
        assert widget._data.shape == (2, 6, 8)

        widget.selected_idx = 1
        widget.star_panel(0)
        widget.hide_panel(0)
        widget.set_panel_order([1, 0])
        widget.rotate(1, 180)
        widget.roi_active = True
        widget.roi_list = [{"shape": "circle", "row": 2, "col": 3, "radius": 1}]
        widget.roi_selected_idx = 0
        widget.profile_line = [{"row": 1, "col": 1}, {"row": 4, "col": 6}]
        widget.view_box = [0, 5, 0, 7]
        widget.zoom_row = 2.5
        widget.zoom_col = 3.5

        _save(tmp_path / "frame_1.npy", 1)
        _save(tmp_path / "frame_10.npy", 99)
        changed = widget.poll_folder()

        assert changed == [0]
        assert widget.labels == ["frame_1", "frame_2", "frame_10"]
        np.testing.assert_array_equal(widget._data[:, 0, 0], [1, 2, 10])
        assert widget.selected_idx == 2
        assert widget.starred == [0, 1, 0]
        assert widget.hidden_panels == [1]
        assert widget.panel_order == [2, 1, 0]
        assert widget.image_rotations == [0, 0, 2]
        assert widget.roi_active is True
        assert widget.roi_list[0]["radius"] == 1
        assert widget.roi_selected_idx == 0
        assert widget.profile_line == [{"row": 1, "col": 1}, {"row": 4, "col": 6}]
        assert widget.view_box == [0, 5, 0, 7]
        assert widget.zoom_row == 2.5
        assert widget.zoom_col == 3.5
        assert widget.show_fft is True
    finally:
        widget.close()


def test_show3d_folder_poll_remaps_frame_state_without_stopping_playback(
    tmp_path: Path,
) -> None:
    _save(tmp_path / "frame_2.npy", 2)
    _save(tmp_path / "frame_10.npy", 10)
    widget = Show3D.from_folder(tmp_path, watch=False, show_fft=True)
    try:
        widget.slice_idx = 1
        widget.bookmarked_frames = [0, 1]
        widget.loop_start = 0
        widget.loop_end = 1
        widget.starred = [1]
        widget.roi_active = True
        widget.roi_list = [{"shape": "circle", "row": 2, "col": 3, "radius": 1}]
        widget.roi_selected_idx = 0
        widget.profile_line = [{"row": 1, "col": 1}, {"row": 4, "col": 6}]
        widget.playing = True

        _save(tmp_path / "frame_1.npy", 1)
        _save(tmp_path / "frame_10.npy", 99)
        changed = widget.poll_folder()

        assert changed == [0]
        assert widget.labels == ["frame_1", "frame_2", "frame_10"]
        np.testing.assert_array_equal(widget._data[:, 0, 0], [1, 2, 10])
        assert widget.slice_idx == 2
        assert widget.bookmarked_frames == [1, 2]
        assert widget.loop_start == 1
        assert widget.loop_end == 2
        assert widget.starred == [2]
        assert widget.roi_active is True
        assert widget.roi_list[0]["radius"] == 1
        assert widget.roi_selected_idx == 0
        assert widget.profile_line == [{"row": 1, "col": 1}, {"row": 4, "col": 6}]
        assert widget.playing is True
        assert widget.show_fft is True
    finally:
        widget.close()


@pytest.mark.parametrize(
    ("viewer", "count_attr"),
    [(Show2D, "n_images"), (Show3D, "n_slices")],
)
def test_folder_background_watcher_appends_full_resolution_images(
    tmp_path: Path,
    viewer,
    count_attr: str,
) -> None:
    _save(tmp_path / "frame_1.npy", 1, shape=(17, 23))
    widget = viewer.from_folder(tmp_path, watch_interval=0.02)
    source = widget._folder_source
    thread = source._watch_thread
    try:
        assert thread is not None and thread.is_alive()
        _save(tmp_path / "frame_2.npy", 2, shape=(17, 23))
        _wait_until(lambda: int(getattr(widget, count_attr)) == 2)
        assert widget._data.shape[-2:] == (17, 23)

        _save(tmp_path / "frame_2.npy", 7, shape=(17, 23))
        time.sleep(0.08)
        assert float(widget._data[1, 0, 0]) == 2.0
    finally:
        widget.stop_folder_watch()
        widget.stop_folder_watch()
        assert not thread.is_alive()
        assert source._watch_thread is None
        widget.close()
        widget.close()


def test_show3d_free_stops_folder_watcher(tmp_path: Path) -> None:
    _save(tmp_path / "frame_1.npy", 1)
    widget = Show3D.from_folder(tmp_path, watch_interval=0.02)
    source = widget._folder_source
    thread = source._watch_thread

    assert thread is not None and thread.is_alive()
    widget.free()

    assert not thread.is_alive()
    assert source._watch_thread is None
    widget.close()


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
def test_folder_failed_and_mismatched_files_remain_retryable(
    tmp_path: Path,
    viewer,
) -> None:
    _save(tmp_path / "frame_1.npy", 1)
    widget = viewer.from_folder(tmp_path, watch=False)
    try:
        partial = tmp_path / "frame_2.npy"
        partial.write_bytes(b"not a complete npy file")
        assert widget.poll_folder() == []
        assert partial.resolve() in widget.folder_errors
        assert [path.name for path in widget.folder_paths] == ["frame_1.npy"]

        _save(partial, 2)
        assert widget.poll_folder() == [1]

        mismatch = tmp_path / "frame_3.npy"
        _save(mismatch, 3, shape=(3, 4))
        with pytest.raises(
            ValueError,
            match=r"frame_3\.npy.*expected \(6, 8\), got \(3, 4\)",
        ):
            widget.poll_folder()
        assert mismatch.resolve() in widget.folder_errors
        assert [path.name for path in widget.folder_paths] == ["frame_1.npy", "frame_2.npy"]

        _save(mismatch, 3)
        assert widget.poll_folder() == [2]
    finally:
        widget.close()


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
def test_folder_calibration_is_preserved_until_files_disagree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    viewer,
) -> None:
    from quantem.widget import io as widget_io

    class Dataset:
        def __init__(self, value: float, sampling: float) -> None:
            self.array = np.full((5, 7), value, dtype=np.float32)
            self.sampling = (sampling, sampling)
            self.units = ("nm", "nm")

    sampling_by_name = {"frame_1.fake.npy": 0.2, "frame_2.fake.npy": 0.2}

    def read_image(path: Path) -> Dataset:
        return Dataset(float(path.name[6]), sampling_by_name[path.name])

    monkeypatch.setattr(widget_io, "read_image", read_image)
    for name in sampling_by_name:
        (tmp_path / name).write_bytes(name.encode())

    widget = viewer.from_folder(tmp_path, watch=False)
    try:
        assert widget.pixel_size == pytest.approx(0.2)
        assert widget.pixel_unit == "nm"
        assert widget.scale_bar_visible is True

        sampling_by_name["frame_3.fake.npy"] = 0.3
        (tmp_path / "frame_3.fake.npy").write_bytes(b"new mixed calibration")
        assert widget.poll_folder() == [2]

        assert widget.pixel_size == 0.0
        assert widget.scale_bar_visible is False
        assert "different sampling or units" in widget._folder_calibration_status
    finally:
        widget.close()
