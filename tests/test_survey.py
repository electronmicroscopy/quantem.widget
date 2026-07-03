import json
from pathlib import Path

import h5py
import numpy as np

from quantem.widget import survey
from quantem.widget.survey import has_eds, scan_rotation_deg, write_survey_notebook


def _metadata(rotation: float, *, stage=None, fov=None):
    metadata = {
        "Scan": {"ScanRotation": str(rotation)},
        "BinaryResult": {"PixelSize": {"height": 1e-9, "width": 1e-9}},
    }
    if stage is not None:
        metadata["Stage"] = {"Position": {"x": stage[0], "y": stage[1], "z": stage[2]}}
    if fov is not None:
        metadata["Optics"] = {
            "FullScanFieldOfView": {"height": fov[0], "width": fov[1]},
        }
    text = json.dumps(metadata).encode()
    arr = np.zeros((len(text) + 1, 1), dtype=np.uint8)
    arr[:len(text), 0] = np.frombuffer(text, dtype=np.uint8)
    return arr


def _image_emd(path: Path, *, shape=(16, 16), rotation=0.0, stage=None, fov=None):
    with h5py.File(path, "w") as h:
        group = h.create_group("Data/Image/uid")
        group.create_dataset("Data", data=np.arange(shape[0] * shape[1], dtype=np.float32).reshape(*shape))
        group.create_dataset("Metadata", data=_metadata(rotation, stage=stage, fov=fov))


def _eds_emd(path: Path, *, shape=(8, 8), rotation=1.57079632679, stage=None, fov=None):
    with h5py.File(path, "w") as h:
        group = h.create_group("Data/SpectrumImage/uid")
        group.create_dataset("Data", data=np.zeros((*shape, 4), dtype=np.uint16))
        group.create_dataset("Metadata", data=_metadata(rotation, stage=stage, fov=fov))
        h.create_group("Data/SpectrumStream")


def test_survey_detects_eds_and_scan_rotation(tmp_path):
    img = tmp_path / "0010 - HAADF 15Mx Nano.emd"
    eds = tmp_path / "0020 - HAADF 10.5Mx Nano EDS.emd"
    _image_emd(img, rotation=0.25)
    _eds_emd(eds)

    assert not has_eds(img)
    assert has_eds(eds)
    assert scan_rotation_deg(img) == np.degrees(0.25)
    assert round(scan_rotation_deg(eds), 1) == 90.0


def test_survey_builds_non_eds_gallery_and_inventory(tmp_path):
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd", shape=(16, 16), rotation=0.0)
    _image_emd(tmp_path / "0011 - HAADF 3.7Mx Nano.emd", shape=(20, 12), rotation=0.5)
    _eds_emd(tmp_path / "0020 - HAADF 10.5Mx Nano EDS.emd")

    result = survey(tmp_path, thumb=8, group_by="fov")

    assert len(result.items) == 3
    assert len(result.image_items) == 2
    assert len(result.eds_items) == 1
    assert result.gallery is not None
    assert result.gallery._data.shape == (2, 8, 8)
    assert result.gallery.pixel_sizes == [2.0, 2.0]
    assert result.gallery.pixel_unit == "nm"
    labels = result.gallery.labels
    assert "0010" in labels[0] and "15Mx" in labels[0]
    assert "downsample 2x" in labels[0]
    assert "EDS" not in " ".join(labels)
    assert result.inventory_rows[0]["downsample"] == 2
    assert result.inventory_rows[0]["pixel_size"] == "2 nm/px"
    assert result.inventory_rows[2]["kind"] == "EDS"


def test_survey_save_state_is_opt_in_for_docs(tmp_path):
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd", shape=(16, 16))

    lightweight = survey(tmp_path, thumb=8)
    embedded = survey(tmp_path, thumb=8, save_state=True)

    assert lightweight.gallery._save_state is False
    assert embedded.gallery._save_state is True


def test_survey_selection_roundtrip_and_selected_gallery(tmp_path):
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd", shape=(16, 16), rotation=0.0)
    _image_emd(tmp_path / "0011 - HAADF 3.7Mx Nano.emd", shape=(16, 16), rotation=0.5)
    _eds_emd(tmp_path / "0020 - HAADF 10.5Mx Nano EDS.emd")

    result = survey(tmp_path, thumb=8, group_by="fov")
    result.gallery.star_panel(1)
    result.eds_selection_controls["0020"].value = True

    selected = result.selection()
    assert selected["selected_image_ids"] == ["0011"]
    assert selected["selected_eds_ids"] == ["0020"]
    assert result.selection_file.name == ".quantem-survey.json"
    assert [item.file_id for item in result.selected("image")] == ["0011"]
    assert [p.name for p in result.paths("image")] == ["0011 - HAADF 3.7Mx Nano.emd"]

    selected_gallery = result.show_selected()
    assert selected_gallery._data.shape == (1, 8, 8)
    assert selected_gallery.labels == [result.gallery.labels[1]]

    path = result.save()
    saved = json.loads(path.read_text())
    assert saved["selected_files"][0]["id"] == "0011"

    restored = survey(tmp_path, thumb=8)
    restored.load(path)
    assert restored.gallery.starred_panels == [1]
    assert restored.eds_selection_controls["0020"].value is True
    assert restored.selection()["selected_eds_ids"] == ["0020"]
    assert [p.name for p in restored.selected_paths("eds")] == ["0020 - HAADF 10.5Mx Nano EDS.emd"]


def test_survey_groups_same_field_of_view_rows(tmp_path):
    shared_stage = (1e-6, 2e-6, 3e-6)
    shared_fov = (6.7e-9, 6.7e-9)
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd", stage=shared_stage, fov=shared_fov, rotation=0.0)
    _image_emd(
        tmp_path / "0011 - HAADF 15Mx Nano.emd",
        stage=(shared_stage[0] + 2e-9, shared_stage[1], shared_stage[2]),
        fov=shared_fov,
        rotation=1.57079632679,
    )
    _eds_emd(tmp_path / "0020 - HAADF 15Mx Nano EDS.emd", stage=shared_stage, fov=shared_fov)
    _image_emd(
        tmp_path / "0030 - HAADF 15Mx Nano.emd",
        stage=(shared_stage[0] + 20e-9, shared_stage[1], shared_stage[2]),
        fov=shared_fov,
    )

    result = survey(tmp_path, thumb=8, group_by="fov")

    assert len(result.fov_groups) == 1
    assert [item.file_id for item in result.fov_groups[0]] == ["0010", "0011", "0020"]
    assert result.inventory_rows[0]["fov_group"] == "FOV 1"
    assert len(result.image_galleries) == 2
    assert result.group_view == "stack"
    assert result.image_galleries[0][0].n_slices == 2
    result.image_galleries[0][0].star_panel(0, frame=1)
    assert [item.file_id for item in result.selected("image")] == ["0011"]


def test_survey_can_render_fov_groups_as_show2d_gallery(tmp_path):
    shared_stage = (1e-6, 2e-6, 3e-6)
    shared_fov = (6.7e-9, 6.7e-9)
    _image_emd(tmp_path / "0010 - HAADF 15Mx Nano.emd", stage=shared_stage, fov=shared_fov)
    _image_emd(tmp_path / "0011 - HAADF 15Mx Nano.emd", stage=shared_stage, fov=shared_fov)

    result = survey(tmp_path, thumb=8, group_by="fov", group_view="gallery")

    assert result.group_view == "gallery"
    assert result.image_galleries[0][0].ncols == 2
    result.image_galleries[0][0].star_panel(1)
    assert [item.file_id for item in result.selected("image")] == ["0011"]


def test_survey_default_groups_by_session_order_and_magnification(tmp_path):
    for i in range(1, 7):
        _image_emd(tmp_path / f"{i:04d} - HAADF 15Mx Nano.emd")
    _image_emd(tmp_path / "0007 - HAADF 3.7Mx Nano.emd")

    result = survey(tmp_path, thumb=8)

    assert len(result.fov_groups) == 2
    assert [item.file_id for item in result.fov_groups[0]] == ["0001", "0002", "0003", "0004"]
    assert [item.file_id for item in result.fov_groups[1]] == ["0005", "0006"]
    assert result.inventory_rows[0]["fov_group"] == "Session 1"
    assert result.inventory_rows[4]["fov_group"] == "Session 2"


def test_write_survey_notebook_uses_public_api(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    _image_emd(folder / "0010 - HAADF 15Mx Nano.emd")
    _eds_emd(folder / "0020 - HAADF 10.5Mx Nano EDS.emd")
    out = tmp_path / "survey.ipynb"

    written = write_survey_notebook(folder, out, thumb=8, eds_backend="auto")

    nb = json.loads(written.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    assert "from quantem.widget import survey" in source
    assert "survey(" in source
    assert "group_view='stack'" in source
    assert "result =" not in source
    assert "print(" not in source
    assert "ShowEDS.from_emd" not in source
    assert len(nb["cells"]) == 2


def test_cli_survey_writes_notebook(tmp_path):
    from quantem.widget import cli

    folder = tmp_path / "data"
    folder.mkdir()
    _image_emd(folder / "0010 - HAADF 15Mx Nano.emd")
    out = tmp_path / "cli_survey.ipynb"

    code = cli.main(["survey", str(folder), "--notebook", str(out), "--thumb", "8", "--no-open"])

    assert code == 0
    assert out.exists()
    source = "\n".join("".join(cell.get("source", [])) for cell in json.loads(out.read_text())["cells"])
    assert "survey(" in source
    assert "result =" not in source
    assert "print(" not in source
