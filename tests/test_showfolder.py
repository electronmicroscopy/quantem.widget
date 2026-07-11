import json
from pathlib import Path
import threading

import h5py
import numpy as np
import pytest

from quantem.widget import ShowFolder, prebuild_showfolder_cache, show_folder


def _metadata(rotation: float = 0.0) -> np.ndarray:
    payload = json.dumps({
        "Scan": {"ScanRotation": str(rotation)},
        "BinaryResult": {"PixelSize": {"height": 1e-9, "width": 1e-9}},
    }).encode()
    arr = np.zeros((len(payload) + 1, 1), dtype=np.uint8)
    arr[:len(payload), 0] = np.frombuffer(payload, dtype=np.uint8)
    return arr


def _image_emd(path: Path, *, shape: tuple[int, int] = (16, 16), rotation: float = 0.0) -> None:
    with h5py.File(path, "w") as h:
        group = h.create_group("Data/Image/uid")
        group.create_dataset("Data", data=np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape))
        group.create_dataset("Metadata", data=_metadata(rotation))


def _eds_emd(path: Path, *, shape: tuple[int, int] = (8, 8)) -> None:
    with h5py.File(path, "w") as h:
        group = h.create_group("Data/SpectrumImage/uid")
        group.create_dataset("Data", data=np.zeros((*shape, 4), dtype=np.uint16))
        group.create_dataset("Metadata", data=_metadata(1.57079632679))
        h.create_group("Data/SpectrumStream")


def test_show_folder_surveys_microscopy_folder_and_delegates_selection(tmp_path: Path) -> None:
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd")
    _image_emd(tmp_path / "0011 - HAADF 15Mx Nano 90deg.emd", rotation=1.57079632679)
    _eds_emd(tmp_path / "0020 - HAADF 15Mx Nano EDS.emd")

    widget = show_folder(tmp_path, thumb=8, group_by="none")

    assert isinstance(widget, ShowFolder)
    assert widget.folder == tmp_path.resolve()
    assert len(widget.items) == 3
    assert widget.inventory_rows[0]["kind"] == "image"
    assert widget.inventory_rows[2]["kind"] == "EDS"

    assert widget.browser is not None
    assert widget.browser.gallery is not None
    widget.browser.gallery.star_panel(1)
    assert widget.browser.eds_widgets == []
    assert widget.browser.eds_selection_controls == {}

    selected = widget.selection()
    assert selected["selected_image_ids"] == ["0011"]
    assert "selected_eds_ids" not in selected
    assert [path.name for path in widget.selected_paths("image")] == ["0011 - HAADF 15Mx Nano 90deg.emd"]
    assert widget.selected_folders() == [tmp_path.resolve()]


def test_show_folder_save_and_load_selection(tmp_path: Path) -> None:
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd")
    _image_emd(tmp_path / "0011 - HAADF 15Mx Nano.emd")

    widget = ShowFolder(tmp_path, thumb=8, group_by="none")
    assert widget.browser is not None
    widget.browser.gallery.star_panel(0)

    selection_path = widget.save(tmp_path / "selection.json")
    restored = ShowFolder(tmp_path, thumb=8, group_by="none")
    restored.load(selection_path)

    assert [item.file_id for item in restored.selected("image")] == ["0010"]


def test_show_folder_opens_selected_images_as_show2d_and_show3d(tmp_path: Path) -> None:
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd")
    _image_emd(tmp_path / "0011 - HAADF 15Mx Nano.emd")
    _image_emd(tmp_path / "0012 - HAADF 15Mx Nano.emd")

    widget = ShowFolder(tmp_path, thumb=8, group_by="none")
    assert widget.browser is not None
    assert widget.browser.gallery is not None
    widget.browser.gallery.set_starred_panels([0, 2])

    selected_2d = widget.show_selected()
    selected_3d = widget.show_selected_stack()
    selected_both = widget.browser.show_selected_both()

    assert selected_2d.__class__.__name__ == "Show2D"
    assert selected_3d.__class__.__name__ == "Show3D"
    assert selected_both.__class__.__name__ == "VBox"
    assert [child.__class__.__name__ for child in selected_both.children] == ["Show2D", "Show3D"]
    assert selected_2d._data.shape[0] == 2
    assert selected_3d.n_slices == 2
    assert [label.split(" | ", 1)[0] for label in selected_2d.labels] == ["0010", "0012"]
    assert [label.split(" | ", 1)[0] for label in selected_3d.labels] == ["0010", "0012"]


def test_show_folder_opens_all_images_as_live_show2d_and_show3d(tmp_path: Path) -> None:
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd")
    _image_emd(tmp_path / "0011 - HAADF 15Mx Nano.emd")

    widget = ShowFolder(tmp_path, thumb=8, group_by="none")
    assert widget.browser is not None

    all_2d = widget.show_all_as_show2d()
    all_3d = widget.show_all_as_show3d_stack()

    assert all_2d.__class__.__name__ == "Show2D"
    assert all_3d.__class__.__name__ == "Show3D"
    assert all_2d._data.shape[0] == 2
    assert all_3d.n_slices == 2
    assert [label.split(" | ", 1)[0] for label in all_2d.labels] == ["0010", "0011"]
    assert [label.split(" | ", 1)[0] for label in all_3d.labels] == ["0010", "0011"]


def test_show_folder_public_open_show4dstem_delegates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd")
    widget = ShowFolder(tmp_path, thumb=8, group_by="none")
    assert widget.browser is not None
    calls = []
    sentinel = object()

    def fake_open_show4dstem(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(widget.browser, "open_show4dstem", fake_open_show4dstem)

    result = widget.open_show4dstem(page_budget=2, det_bin=4)

    assert result is sentinel
    assert calls == [{"page_budget": 2, "det_bin": 4}]


def test_show_folder_renders_master_only_folder_for_show4dstem(tmp_path: Path) -> None:
    (tmp_path / "scan_000_master.h5").touch()
    (tmp_path / "scan_001_master.h5").touch()

    widget = ShowFolder(tmp_path, thumb=8, group_by="none")

    assert widget.browser is not None
    assert widget.items == []
    assert [row["status"] for row in widget.master_qc_rows] == ["bad", "bad"]
    assert widget.browser.gallery is None
    assert widget.browser.selection_panel is not None
    assert "4D-STEM master" in widget.browser.widget.children[0].value


def test_show_folder_watch_signature_tracks_4dstem_masters(tmp_path: Path) -> None:
    first_master = tmp_path / "scan_000_master.h5"
    first_master.touch()

    widget = ShowFolder(tmp_path, thumb=8, group_by="none")
    before = widget._folder_signature()

    second_master = tmp_path / "scan_001_master.h5"
    second_master.touch()
    after = widget._folder_signature()

    assert {row[0] for row in before} == {first_master.name}
    assert {row[0] for row in after} == {first_master.name, second_master.name}


def test_show_folder_reuses_thumbnail_cache(tmp_path: Path, monkeypatch) -> None:
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd")
    _image_emd(tmp_path / "0011 - HAADF 15Mx Nano.emd")

    import quantem.widget.io as widget_io

    real_read_image = widget_io.read_image
    calls = {"count": 0}

    def counted_read_image(path):
        calls["count"] += 1
        return real_read_image(path)

    monkeypatch.setattr(widget_io, "read_image", counted_read_image)

    first = ShowFolder(
        tmp_path,
        thumb=8,
        group_by="none",
        cache_dir=tmp_path / "cache",
    )
    assert calls["count"] == 2
    assert first.cache_info["misses"] == 2
    assert first.cache_info["hits"] == 0
    assert first.cache_path is not None
    assert (first.cache_path / "manifest.json").exists()
    assert (first.cache_path / "thumbnails.npz").exists()

    calls["count"] = 0
    second = ShowFolder(
        tmp_path,
        thumb=8,
        group_by="none",
        cache_dir=tmp_path / "cache",
    )

    assert calls["count"] == 0
    assert second.cache_info["hits"] == 2
    assert second.cache_info["misses"] == 0
    assert len(second.items) == 2


def test_show_folder_watch_once_adds_and_removes_files_in_place(tmp_path: Path) -> None:
    img0 = tmp_path / "0010 - HAADF 15Mx Nano.emd"
    img1 = tmp_path / "0011 - HAADF 15Mx Nano.emd"
    _image_emd(img0)
    _image_emd(img1)

    widget = ShowFolder(
        tmp_path,
        thumb=8,
        group_by="none",
        cache_dir=tmp_path / "cache",
    )
    assert widget.browser is not None
    assert widget.browser.gallery is not None
    displayed = widget.widget
    widget.browser.gallery.star_panel(0)
    opened = widget.open_both()
    selected_2d, selected_3d = opened.children
    assert widget.browser._active_selected_modes == {"show2d", "show3d"}
    assert widget.browser._selection_viewer_output.children == (
        selected_2d,
        selected_3d,
    )
    assert selected_2d._data.shape[0] == 1
    assert selected_3d.n_slices == 1

    widget.watch(start=False)
    img2 = tmp_path / "0012 - HAADF 15Mx Nano.emd"
    _image_emd(img2)

    assert widget.watch_once() is True
    assert widget.widget is displayed
    assert [item.file_id for item in widget.items] == ["0010", "0011", "0012"]
    assert [item.file_id for item in widget.selected("image")] == ["0010"]
    assert widget.browser._selected_show2d_widget is selected_2d
    assert widget.browser._selected_show3d_widget is selected_3d
    assert widget.browser._selection_viewer_output.children == (
        selected_2d,
        selected_3d,
    )
    assert selected_2d._data.shape[0] == 1
    assert selected_3d.n_slices == 1
    assert "1 new" in widget._watch_status.value

    widget.browser.gallery.star_panel(2)
    assert [item.file_id for item in widget.selected("image")] == ["0010", "0012"]
    assert widget.browser._selected_show2d_widget is selected_2d
    assert widget.browser._selected_show3d_widget is selected_3d
    assert selected_2d._data.shape[0] == 2
    assert selected_3d.n_slices == 2

    img0.unlink()

    assert widget.watch_once() is True
    assert [item.file_id for item in widget.items] == ["0011", "0012"]
    assert [item.file_id for item in widget.selected("image")] == ["0012"]
    assert widget.browser._selected_show2d_widget is selected_2d
    assert widget.browser._selected_show3d_widget is selected_3d
    assert selected_2d._data.shape[0] == 1
    assert selected_3d.n_slices == 1
    assert "1 removed" in widget._watch_status.value

    manifest = widget.cache_path / "manifest.json"
    payload = json.loads(manifest.read_text())
    cached_rel_paths = [entry["relative_path"] for entry in payload["entries"]]
    assert cached_rel_paths == [img1.name, img2.name]


def test_show_folder_watch_once_appends_all_image_viewers_in_place(tmp_path: Path) -> None:
    img0 = tmp_path / "0010 - HAADF 15Mx Nano.emd"
    img1 = tmp_path / "0011 - HAADF 15Mx Nano.emd"
    _image_emd(img0)
    _image_emd(img1)

    widget = ShowFolder(
        tmp_path,
        thumb=8,
        group_by="none",
        cache_dir=tmp_path / "cache",
    )
    assert widget.browser is not None
    assert widget.browser.selection_panel is not None
    button_row = widget.browser.selection_panel.children[2]
    open_all_both = next(
        button
        for button in button_row.children
        if getattr(button, "description", "") == "Open All Both"
    )
    open_all_both.click()
    holder = widget.browser._selection_viewer_output
    all_2d = widget.browser._selected_show2d_widget
    all_3d = widget.browser._selected_show3d_widget
    assert widget.browser._active_selected_modes == {"show2d_all", "show3d_all"}
    assert widget.browser._selection_viewer_output.children == (all_2d, all_3d)
    assert all_2d._data.shape[0] == 2
    assert all_3d.n_slices == 2

    widget.watch(start=False)
    img2 = tmp_path / "0012 - HAADF 15Mx Nano.emd"
    _image_emd(img2)

    assert widget.watch_once() is True
    assert widget.browser._selection_viewer_output is holder
    assert widget.browser._selected_show2d_widget is all_2d
    assert widget.browser._selected_show3d_widget is all_3d
    assert widget.browser._selection_viewer_output.children == (all_2d, all_3d)
    assert all_2d._data.shape[0] == 3
    assert all_3d.n_slices == 3
    assert [label.split(" | ", 1)[0] for label in all_2d.labels] == ["0010", "0011", "0012"]
    assert [label.split(" | ", 1)[0] for label in all_3d.labels] == ["0010", "0011", "0012"]


def test_show_folder_viewer_observers_see_published_inventory(tmp_path: Path) -> None:
    # C1: an all-image append notifies both preserved viewers, expect callbacks
    # to observe the new ShowFolder inventory rather than the previous browser.
    _image_emd(tmp_path / "0010 - HAADF live.emd")
    _image_emd(tmp_path / "0011 - HAADF live.emd")
    widget = ShowFolder(tmp_path, thumb=8, group_by="none", cache=False)
    holder = widget.open_both(all_images=True)
    show2d, show3d = holder.children
    observed: list[tuple[str, int, int]] = []

    show2d.observe(
        lambda change: observed.append(
            ("show2d", len(widget.items), int(change["new"]))
        ),
        names="n_images",
    )
    show3d.observe(
        lambda change: observed.append(
            ("show3d", len(widget.items), int(change["new"]))
        ),
        names="n_slices",
    )
    widget.watch(start=False)
    _image_emd(tmp_path / "0012 - HAADF live.emd")

    assert widget.watch_once() is True
    assert ("show2d", 3, 3) in observed
    assert ("show3d", 3, 3) in observed


def test_show_folder_open_both_holder_survives_empty_transitions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # C1: an empty acquisition folder is mounted before its first file, expect
    # one durable holder to transition placeholders -> viewers -> placeholders.
    widget = ShowFolder(tmp_path, thumb=8, group_by="none", cache=False)
    holder = widget.open_both(all_images=True)
    assert holder is widget.browser._selection_viewer_output
    assert [child.__class__.__name__ for child in holder.children] == ["HTML", "HTML"]
    assert widget.browser.selection_panel.children[3].__class__.__name__ == "HTML"

    widget.watch(start=False)
    image = tmp_path / "0010 - HAADF live.emd"
    _image_emd(image)

    assert widget.watch_once() is True
    assert widget.browser._selection_viewer_output is holder
    first_show2d, first_show3d = holder.children
    assert first_show2d.__class__.__name__ == "Show2D"
    assert first_show3d.__class__.__name__ == "Show3D"
    close_counts = {"show2d": 0, "show3d": 0}
    original_show2d_close = first_show2d.close
    original_show3d_close = first_show3d.close

    def close_show2d() -> None:
        close_counts["show2d"] += 1
        original_show2d_close()

    def close_show3d() -> None:
        close_counts["show3d"] += 1
        original_show3d_close()

    monkeypatch.setattr(first_show2d, "close", close_show2d)
    monkeypatch.setattr(first_show3d, "close", close_show3d)

    # C2: removing the final file and adding a later file crosses an empty
    # signature, expect both changes to remain discoverable with the same holder.
    image.unlink()
    assert widget.watch_once() is True
    assert widget.browser._selection_viewer_output is holder
    assert [child.__class__.__name__ for child in holder.children] == ["HTML", "HTML"]
    assert close_counts == {"show2d": 1, "show3d": 1}
    _image_emd(tmp_path / "0011 - HAADF live.emd")
    assert widget.watch_once() is True
    assert widget.browser._selection_viewer_output is holder
    assert [child.__class__.__name__ for child in holder.children] == ["Show2D", "Show3D"]


def test_show_folder_closes_replaced_placeholders_and_inactive_viewers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # C1: an empty holder replaces two HTML placeholders with live viewers,
    # expect both orphan comms to close while the durable holder remains.
    widget = ShowFolder(tmp_path, thumb=8, group_by="none", cache=False)
    holder = widget.open_both(all_images=True)
    placeholders = tuple(holder.children)
    closed: list[int] = []
    for placeholder in placeholders:
        original = placeholder.close

        def close(original=original, placeholder=placeholder) -> None:
            closed.append(id(placeholder))
            original()

        monkeypatch.setattr(placeholder, "close", close)
    widget.watch(start=False)
    _image_emd(tmp_path / "0010 - HAADF live.emd")
    assert widget.watch_once() is True
    assert set(closed) == {id(child) for child in placeholders}

    # C2: switching from dual view to one viewer makes the other inactive,
    # expect its heavy data and comm to be released instead of retained off-DOM.
    show2d, show3d = holder.children
    show3d_closed: list[bool] = []
    original_show3d_close = show3d.close

    def close_show3d() -> None:
        show3d_closed.append(True)
        original_show3d_close()

    monkeypatch.setattr(show3d, "close", close_show3d)
    assert widget.open_show2d(all_images=True) is show2d
    assert show3d_closed
    assert widget.browser._selected_show3d_widget is None
    assert holder.children == (show2d,)

    # C3: wrapper close includes the remaining selected model as well as the
    # visible browser tree, expect no inactive selected reference to leak.
    show2d_closed: list[bool] = []
    original_show2d_close = show2d.close

    def close_show2d() -> None:
        show2d_closed.append(True)
        original_show2d_close()

    monkeypatch.setattr(show2d, "close", close_show2d)
    widget.close()
    assert show2d_closed


def test_show_folder_rebuild_closes_orphans_but_keeps_live_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # C1: one live dual-view rebuild replaces a source gallery and control panel,
    # expect their models and the temporary root to close while live models stay.
    _image_emd(tmp_path / "0010 - HAADF live.emd")
    widget = ShowFolder(tmp_path, thumb=8, group_by="none", cache=False)
    assert widget.browser is not None
    holder = widget.open_both(all_images=True)
    show2d, show3d = holder.children
    old_panel = widget.browser.selection_panel
    old_button = old_panel.children[2].children[0]
    old_gallery = widget.browser.gallery
    closed: set[int] = set()

    def track_close(model) -> None:
        original = model.close

        def close() -> None:
            closed.add(id(model))
            original()

        monkeypatch.setattr(model, "close", close)

    for model in (holder, show2d, show3d, old_panel, old_button, old_gallery):
        track_close(model)

    import quantem.widget.showfolder_core as showfolder_core

    real_build = showfolder_core.build_showfolder
    temporary_roots = []

    def tracked_build(*args, **kwargs):
        browser = real_build(*args, **kwargs)
        temporary_roots.append(browser.widget)
        track_close(browser.widget)
        return browser

    monkeypatch.setattr(showfolder_core, "build_showfolder", tracked_build)
    widget.watch(start=False)
    _image_emd(tmp_path / "0011 - HAADF live.emd")

    assert widget.watch_once() is True
    assert id(old_panel) in closed
    assert id(old_button) in closed
    assert id(old_gallery) in closed
    assert id(temporary_roots[-1]) in closed
    assert id(holder) not in closed
    assert id(show2d) not in closed
    assert id(show3d) not in closed


def test_show_folder_retries_change_that_lands_during_rebuild(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # C1: a second file lands after the triggering signature but during rebuild,
    # expect the following deterministic poll to discover it.
    _image_emd(tmp_path / "0010 - HAADF live.emd")
    widget = ShowFolder(tmp_path, thumb=8, group_by="none", cache=False)
    widget.watch(start=False)
    _image_emd(tmp_path / "0011 - HAADF live.emd")
    original_replace = widget._replace_browser
    changed_during_build = False

    def replace_browser(**kwargs) -> None:
        nonlocal changed_during_build
        original_replace(**kwargs)
        if not changed_during_build:
            changed_during_build = True
            _image_emd(tmp_path / "0012 - HAADF live.emd")

    monkeypatch.setattr(widget, "_replace_browser", replace_browser)

    assert widget.watch_once() is True
    assert [item.file_id for item in widget.items] == ["0010", "0011"]
    assert widget.watch_once() is True
    assert [item.file_id for item in widget.items] == ["0010", "0011", "0012"]


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_show_folder_rejects_invalid_watch_intervals(
    tmp_path: Path,
    interval: float,
) -> None:
    # C1: a non-positive or non-finite interval could busy-loop, expect a clear
    # validation error before any watcher state changes.
    _image_emd(tmp_path / "0010 - HAADF live.emd")
    widget = ShowFolder(tmp_path, thumb=8, group_by="none", cache=False)

    with pytest.raises(ValueError, match="finite value > 0"):
        widget.watch(interval=interval)

    assert widget._watch_thread is None
    assert widget._watch_stop is None


def test_show_folder_watch_stop_restart_and_close_are_idempotent(tmp_path: Path) -> None:
    # C1: switching a background watcher to deterministic polling, expect the
    # previous daemon to finish before its references are cleared.
    _image_emd(tmp_path / "0010 - HAADF live.emd")
    widget = ShowFolder(tmp_path, thumb=8, group_by="none", cache=False)
    widget.watch(interval=60.0)
    first_thread = widget._watch_thread
    first_stop = widget._watch_stop

    widget.watch(interval=60.0, start=False)

    assert first_stop.is_set()
    assert not first_thread.is_alive()
    assert widget._watch_thread is None
    assert widget._watch_stop is None

    # C2: repeated stop/close calls after a restart, expect truthful empty state
    # and no surviving ShowFolder watch thread.
    widget.watch(interval=60.0)
    second_thread = widget._watch_thread
    widget.stop_watch()
    widget.stop_watch()
    assert not second_thread.is_alive()
    assert widget._watch_thread is None
    assert widget._watch_stop is None
    widget.watch(interval=60.0)
    final_thread = widget._watch_thread
    widget.close()
    widget.close()
    assert not final_thread.is_alive()
    assert widget._watch_thread is None
    assert widget._watch_stop is None


def test_show_folder_background_poll_uses_kernel_callback(tmp_path: Path, monkeypatch) -> None:
    # C1: a kernel IOLoop is available, expect the daemon to enqueue rather than
    # execute widget mutation and a stopped queued callback to become a no-op.
    _image_emd(tmp_path / "0010 - HAADF live.emd")
    widget = ShowFolder(tmp_path, thumb=8, group_by="none", cache=False)
    callbacks = []
    scheduled = threading.Event()
    polled_on: list[threading.Thread] = []

    class FakeLoop:
        def add_callback(self, callback) -> None:
            callbacks.append(callback)
            scheduled.set()

    monkeypatch.setattr(widget, "_watch_ioloop", lambda: FakeLoop())
    monkeypatch.setattr(
        widget,
        "watch_once",
        lambda: polled_on.append(threading.current_thread()) or False,
    )
    widget.watch(interval=0.01)

    assert scheduled.wait(timeout=1.0)
    assert polled_on == []
    callback = callbacks[0]
    widget.stop_watch()
    callback()
    assert polled_on == []


def test_show_folder_cache_invalidates_changed_file(tmp_path: Path, monkeypatch) -> None:
    img0 = tmp_path / "0010 - HAADF 15Mx Nano.emd"
    img1 = tmp_path / "0011 - HAADF 15Mx Nano.emd"
    _image_emd(img0)
    _image_emd(img1)
    ShowFolder(tmp_path, thumb=8, group_by="none", cache_dir=tmp_path / "cache")

    _image_emd(img1, shape=(20, 20))

    import quantem.widget.io as widget_io

    real_read_image = widget_io.read_image
    read_names: list[str] = []

    def counted_read_image(path):
        read_names.append(Path(path).name)
        return real_read_image(path)

    monkeypatch.setattr(widget_io, "read_image", counted_read_image)

    refreshed = ShowFolder(tmp_path, thumb=8, group_by="none", cache_dir=tmp_path / "cache")

    assert read_names == [img1.name]
    assert refreshed.cache_info["hits"] == 1
    assert refreshed.cache_info["misses"] == 1


def test_show_folder_cache_modes_and_clear_cache(tmp_path: Path) -> None:
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd")

    disabled = ShowFolder(tmp_path, thumb=8, group_by="none", cache=False)
    assert disabled.cache_info["enabled"] is False
    assert disabled.cache_path is None

    local = ShowFolder(tmp_path, thumb=8, group_by="none", cache="folder")
    assert local.cache_path is not None
    assert tmp_path / ".quantem" / "showfolder-cache" in local.cache_path.parents
    assert (local.cache_path / "manifest.json").exists()

    local.clear_cache()
    assert not (local.cache_path / "manifest.json").exists()
    assert not (local.cache_path / "thumbnails.npz").exists()


def test_show_folder_export_html_writes_nested_widget_state(tmp_path: Path) -> None:
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd")
    _image_emd(tmp_path / "0011 - HAADF 15Mx Nano.emd")

    widget = ShowFolder(tmp_path, thumb=8, group_by="none", cache_dir=tmp_path / "cache")

    out = widget.export_html(tmp_path / "showfolder.html", title="ShowFolder export")

    text = out.read_text(encoding="utf-8")
    assert out.exists()
    assert "ShowFolder export" in text
    assert "application/vnd.jupyter.widget-state+json" in text
    assert "0010" in text


def test_prebuild_showfolder_cache_returns_cache_info(tmp_path: Path) -> None:
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd")

    info = prebuild_showfolder_cache(tmp_path, thumb=8, cache_dir=tmp_path / "cache")

    assert info["enabled"] is True
    assert info["misses"] == 1
    assert Path(info["path"]).exists()
