import json
from pathlib import Path

import h5py
import numpy as np

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
