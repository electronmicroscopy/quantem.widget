import io
import json
import zipfile

import numpy as np
import pytest

from quantem.widget import Bin4D, Show3D, Show3DVolume, Show4D, Show4DSTEM

REQUIRED_EXPORT_KEYS = {
    "metadata_version",
    "widget_name",
    "widget_version",
    "format",
    "export_kind",
}

def _require_image_deps() -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("matplotlib")

def _assert_export_contract(
    metadata: dict,
    *,
    widget_name: str,
    format_name: str,
    export_kind: str,
) -> None:
    assert REQUIRED_EXPORT_KEYS.issubset(metadata.keys())
    assert metadata["metadata_version"] == "1.0"
    assert metadata["widget_name"] == widget_name
    assert isinstance(metadata["widget_version"], str)
    assert metadata["format"] == format_name
    assert metadata["export_kind"] == export_kind
    assert "scan_position" not in metadata

def _read_metadata_from_zip_bytes(payload: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
        assert "metadata.json" in zf.namelist()
        return json.loads(zf.read("metadata.json").decode("utf-8"))

def _read_metadata_from_zip_path(path) -> dict:
    with zipfile.ZipFile(path, "r") as zf:
        assert "metadata.json" in zf.namelist()
        return json.loads(zf.read("metadata.json").decode("utf-8"))

def test_show3d_gif_metadata_contract():
    _require_image_deps()
    data = np.random.rand(6, 12, 12).astype(np.float32)
    w = Show3D(data, cmap="viridis")
    w.loop_start = 1
    w.loop_end = 3
    w._generate_gif()
    metadata = json.loads(w._gif_metadata_json)
    _assert_export_contract(
        metadata,
        widget_name="Show3D",
        format_name="gif",
        export_kind="animated_frames",
    )

def test_show3d_zip_metadata_contract():
    _require_image_deps()
    data = np.random.rand(6, 12, 12).astype(np.float32)
    w = Show3D(data)
    w.loop_start = 0
    w.loop_end = 2
    w._generate_zip()
    metadata = _read_metadata_from_zip_bytes(w._zip_data)
    _assert_export_contract(
        metadata,
        widget_name="Show3D",
        format_name="zip",
        export_kind="png_frames",
    )

def test_show3dvolume_gif_metadata_contract():
    _require_image_deps()
    data = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(data)
    w._export_axis = 2
    w._generate_gif()
    metadata = json.loads(w._gif_metadata_json)
    _assert_export_contract(
        metadata,
        widget_name="Show3DVolume",
        format_name="gif",
        export_kind="animated_slices",
    )

def test_show3dvolume_zip_metadata_contract():
    _require_image_deps()
    data = np.random.rand(6, 6, 6).astype(np.float32)
    w = Show3DVolume(data)
    w._export_axis = 1
    w._generate_zip()
    metadata = _read_metadata_from_zip_bytes(w._zip_data)
    _assert_export_contract(
        metadata,
        widget_name="Show3DVolume",
        format_name="zip",
        export_kind="png_slices",
    )

def test_show4d_gif_metadata_contract():
    _require_image_deps()
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data, cmap="viridis")
    w.set_path([(0, 0), (1, 1), (2, 2)], autoplay=False)
    w._generate_gif()
    metadata = json.loads(w._gif_metadata_json)
    _assert_export_contract(
        metadata,
        widget_name="Show4D",
        format_name="gif",
        export_kind="path_animation",
    )

def test_show4dstem_gif_metadata_contract():
    _require_image_deps()
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.set_path([(0, 0), (1, 1), (2, 2)], autoplay=False)
    w._generate_gif()
    metadata = json.loads(w._gif_metadata_json)
    _assert_export_contract(
        metadata,
        widget_name="Show4DSTEM",
        format_name="gif",
        export_kind="path_animation",
    )

def test_show4dstem_save_image_sidecar_contract(tmp_path):
    _require_image_deps()
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    out = tmp_path / "show4dstem_view.png"
    w.save_image(out, view="all", include_metadata=True)
    metadata = json.loads(out.with_suffix(".json").read_text())
    _assert_export_contract(
        metadata,
        widget_name="Show4DSTEM",
        format_name="png",
        export_kind="single_view_image",
    )
    assert "position" in metadata
    assert set(metadata["position"].keys()) == {"row", "col"}
    assert "scan_position" not in metadata

def test_bin_save_image_sidecar_contract(tmp_path):
    _require_image_deps()
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    w = Bin4D(data, device="cpu")
    out = tmp_path / "bin_grid.png"
    w.save_image(out, view="grid", include_metadata=True)
    metadata = json.loads(out.with_suffix(".json").read_text())
    _assert_export_contract(
        metadata,
        widget_name="Bin4D",
        format_name="png",
        export_kind="single_view_image",
    )

def test_bin_save_zip_and_gif_contract(tmp_path):
    _require_image_deps()
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    w = Bin4D(data, device="cpu")

    zip_path = tmp_path / "bin_bundle.zip"
    w.save_zip(zip_path, include_arrays=True)
    zip_meta = _read_metadata_from_zip_path(zip_path)
    _assert_export_contract(
        zip_meta,
        widget_name="Bin4D",
        format_name="zip",
        export_kind="multi_panel_bundle",
    )

    gif_path = tmp_path / "bin_compare.gif"
    w.save_gif(gif_path, channel="bf")
    gif_meta = json.loads(gif_path.with_suffix(".json").read_text())
    _assert_export_contract(
        gif_meta,
        widget_name="Bin4D",
        format_name="gif",
        export_kind="before_after_animation",
    )
