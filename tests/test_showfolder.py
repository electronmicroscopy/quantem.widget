import json
from pathlib import Path

import h5py
import numpy as np

from quantem.widget import ShowFolder, show_folder


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

    assert widget.survey is not None
    assert widget.survey.gallery is not None
    widget.survey.gallery.star_panel(1)
    widget.survey.eds_selection_controls["0020"].value = True

    selected = widget.selection()
    assert selected["selected_image_ids"] == ["0011"]
    assert selected["selected_eds_ids"] == ["0020"]
    assert [path.name for path in widget.selected_paths("image")] == ["0011 - HAADF 15Mx Nano 90deg.emd"]
    assert widget.selected_folders() == [tmp_path.resolve()]


def test_show_folder_save_and_load_selection(tmp_path: Path) -> None:
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd")
    _image_emd(tmp_path / "0011 - HAADF 15Mx Nano.emd")

    widget = ShowFolder(tmp_path, thumb=8, group_by="none")
    assert widget.survey is not None
    widget.survey.gallery.star_panel(0)

    selection_path = widget.save(tmp_path / "selection.json")
    restored = ShowFolder(tmp_path, thumb=8, group_by="none")
    restored.load(selection_path)

    assert [item.file_id for item in restored.selected("image")] == ["0010"]

