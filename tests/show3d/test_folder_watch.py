from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from quantem.widget import Show3D


def _save(path: Path, value: float, shape: tuple[int, int] = (6, 8)) -> None:
    np.save(path, np.full(shape, value, dtype=np.float32))


def test_live_folder_creates_missing_path_and_appends_png(tmp_path: Path) -> None:
    folder = tmp_path / "new" / "microscope-session"
    widget = Show3D.from_folder(
        folder,
        file_types=".PNG",
        watch=True,
        watch_interval=60,
    )
    try:
        assert folder.is_dir()
        assert widget.n_slices == 0
        widget.stop_folder_watch()

        frame = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
        Image.fromarray(frame).save(folder / "frame_001.png")
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [0]
        assert widget.n_slices == 1
        assert widget.labels == ["frame_001"]
        np.testing.assert_array_equal(widget._data[0], frame)
    finally:
        widget.close()


def test_live_folder_filters_new_arrivals_by_file_type(tmp_path: Path) -> None:
    folder = tmp_path / "mixed-session"
    widget = Show3D.from_folder(
        folder,
        file_types=["png", ".tif"],
        watch=True,
        watch_interval=60,
    )
    try:
        widget.stop_folder_watch()
        frame = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
        Image.fromarray(frame).save(folder / "frame_001.png")
        Image.fromarray(frame + 1).save(folder / "frame_002.tif")
        np.save(folder / "frame_003.npy", frame + 2)

        assert widget.poll_folder() == []
        assert widget.poll_folder() == [0, 1]
        assert widget.labels == ["frame_001", "frame_002"]
        assert [path.suffix for path in widget.folder_paths] == [".png", ".tif"]
    finally:
        widget.close()


@pytest.mark.parametrize("file_types", [[], "", ["png", "csv"], ["png", 1]])
def test_folder_rejects_invalid_file_types(
    tmp_path: Path,
    file_types: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="file_types|Unsupported"):
        Show3D.from_folder(tmp_path, file_types=file_types, watch=False)


def test_one_shot_folder_keeps_missing_path_strict(tmp_path: Path) -> None:
    folder = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        Show3D.from_folder(folder, watch=False)

    assert not folder.exists()


def test_folder_poll_remaps_frame_state_without_stopping_playback(
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
        assert widget.poll_folder() == []
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


def test_folder_many_files_remain_one_unpaged_stack(tmp_path: Path) -> None:
    for index in range(25):
        _save(tmp_path / f"frame_{index:03d}.npy", index)

    widget = Show3D.from_folder(tmp_path, watch=False)
    try:
        assert widget.n_slices == 25
        assert widget.n_panels == 1
        assert widget.n_pages == 1
        assert widget.page_idx == 0
        assert widget.panels_per_page == 0
        assert widget.page_labels == []
        assert widget.visible_panels == [0]
        assert widget.labels[0] == "frame_000"
        assert widget.labels[-1] == "frame_024"
        np.testing.assert_array_equal(widget._data[:, 0, 0], np.arange(25))
    finally:
        widget.close()

    with pytest.raises(TypeError, match="appends every file as a frame"):
        Show3D.from_folder(tmp_path, watch=False, page_size=20)
    with pytest.raises(TypeError, match="does not accept"):
        Show3D.from_folder(tmp_path, watch=False, page_labels=["Page 1"])


def test_folder_live_append_crosses_twenty_without_pages(tmp_path: Path) -> None:
    for index in range(20):
        _save(tmp_path / f"frame_{index:03d}.npy", index)
    widget = Show3D.from_folder(tmp_path, watch=False)
    try:
        widget_id = id(widget)
        widget.slice_idx = 19
        widget.playing = True
        widget.bookmarked_frames = [19]

        _save(tmp_path / "frame_020.npy", 20)
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [20]

        assert id(widget) == widget_id
        assert widget.n_slices == 21
        assert widget.n_panels == 1
        assert widget.n_pages == 1
        assert widget.panels_per_page == 0
        assert widget.page_idx == 0
        assert widget.slice_idx == 19
        assert widget.bookmarked_frames == [19]
        assert widget.playing is True
    finally:
        widget.close()


def test_free_stops_folder_watcher(tmp_path: Path) -> None:
    _save(tmp_path / "frame_1.npy", 1)
    widget = Show3D.from_folder(tmp_path, watch_interval=0.02)
    source = widget._folder_source
    thread = source._watch_thread

    assert thread is not None and thread.is_alive()
    widget.free()

    assert not thread.is_alive()
    assert source._watch_thread is None
    widget.close()
