import json

import numpy as np
import pytest

import quantem.widget.showeds as showeds_module
from quantem.widget import ShowEDS
from quantem.widget.showeds import (
    _estimate_spectrum_image_sidecar_bytes,
    _sum_bin_spectrum_image_lazy,
    bin_spectrum_image,
    eds_line_hints,
    load_spectrum_image_sidecar,
    prepare_spectrum_image_sidecar,
)


def test_showeds_constructor_sets_shape_and_state():
    cube = np.zeros((5, 6, 7), dtype=np.float32)
    energy = np.linspace(0, 6, 7, dtype=np.float32)

    widget = ShowEDS(
        cube,
        energy,
        title="EDS",
        band=(1.0, 2.5),
        roi=(1, 2, 3, 4),
        panel_width_px=240,
        spectrum_width_px=520,
        spectrum_height_px=180,
        smooth=True,
        pixel_size=0.25,
        pixel_unit="nm",
        scale_bar_visible=False,
        map_vmin_pct=5,
        map_vmax_pct=95,
        selected_elements=["au", "Si", "Au"],
        auto_identify=False,
        show_debug=True,
        saved_rois=[{"name": "particle", "row": 1, "col": 2, "height": 3, "width": 4}],
        saved_bands=[{"name": "Au M", "start": 1, "end": 3}],
        export_presets=[{"label": "Small demo", "mode": "single", "binning": 4}],
    )

    assert widget.n_rows == 5
    assert widget.n_cols == 6
    assert widget.n_energy == 7
    assert widget.band_start == 1
    assert widget.band_end == 3
    assert widget.roi_row == 1
    assert widget.roi_col == 2
    assert widget.roi_height == 3
    assert widget.roi_width == 4
    assert widget.roi_shape == "rect"
    assert widget.panel_width_px == 240
    assert widget.spectrum_width_px == 520
    assert widget.spectrum_height_px == 180
    assert widget.smooth is True
    assert widget.pixel_size == 0.25
    assert widget.pixel_unit == "nm"
    assert widget.scale_bar_visible is False
    assert widget.map_vmin_pct == 5
    assert widget.map_vmax_pct == 95
    assert widget.selected_elements == ["Au", "Si"]
    assert widget.auto_identify is False
    assert widget.show_debug is True
    assert widget.debug_control_visible is True
    assert widget.saved_rois == [{"name": "particle", "row": 1, "col": 2, "height": 3, "width": 4, "shape": "rect"}]
    assert widget.saved_bands == [{"name": "Au M", "start": 1, "end": 3}]
    assert widget.export_presets == [{"label": "Small demo", "mode": "single", "binning": 4}]
    assert widget.show_line_hints is True
    assert any(line["element"] == "Si" for line in widget.line_hints)
    assert len(widget.cube_bytes) == cube.size * 4
    assert len(widget.base_image_bytes) == cube.shape[0] * cube.shape[1] * 4


def test_showeds_preserves_uint16_counts_for_browser_backend():
    cube = np.arange(5 * 6 * 7, dtype=np.uint16).reshape(5, 6, 7)

    widget = ShowEDS(cube)

    assert widget.compute_backend == "browser"
    assert widget.cube_dtype == "uint16"
    assert widget.show_debug is False
    assert widget.debug_control_visible is False
    assert len(widget.cube_bytes) == cube.size * 2
    assert np.frombuffer(widget.cube_bytes, dtype=np.uint16).reshape(cube.shape).tolist() == cube.tolist()


def test_showeds_preserves_uint32_counts_for_browser_backend():
    cube = (np.arange(3 * 4 * 5, dtype=np.uint32).reshape(3, 4, 5) + np.iinfo(np.uint16).max + 1)

    widget = ShowEDS(cube)

    assert widget.compute_backend == "browser"
    assert widget.cube_dtype == "uint32"
    assert len(widget.cube_bytes) == cube.size * 4
    assert np.frombuffer(widget.cube_bytes, dtype=np.uint32).reshape(cube.shape).tolist() == cube.tolist()


def test_showeds_can_hide_debug_control_when_debug_hud_is_enabled():
    cube = np.zeros((3, 4, 5), dtype=np.uint16)

    widget = ShowEDS(cube, show_debug=True, debug_control_visible=False)

    assert widget.show_debug is True
    assert widget.debug_control_visible is False


def test_showeds_state_roundtrip():
    cube = np.ones((8, 9, 10), dtype=np.float32)
    widget = ShowEDS(
        cube,
        band=(2, 5),
        roi=(2, 3, 4, 5),
        roi_shape="circle",
        log_spectrum=True,
        smooth=True,
        element_label="Au",
        spectrum_width_px=500,
        spectrum_height_px=260,
        sampling=(0.2, 0.4),
        units=["nm", "nm"],
        scale_bar_visible=True,
        map_vmin_pct=4,
        map_vmax_pct=99,
        selected_elements=["Au", "Cu"],
        auto_identify=False,
        show_debug=True,
        saved_rois=[{"name": "A", "row": 2, "col": 3, "height": 4, "width": 5, "shape": "circle"}],
        saved_bands=[{"name": "B", "start": 2, "end": 5}],
        export_presets=[{"label": "Portable", "mode": "single", "binning": 2}],
    )
    widget.map_zoom = 2.0
    widget.map_view_row = 1.5
    widget.map_view_col = 2.5
    widget.spectrum_view_start = 1.0
    widget.spectrum_view_end = 9.0
    state = widget.state_dict()

    restored = ShowEDS(cube, state=state)

    assert restored.state_dict() == state


def test_showeds_accepts_ellipse_roi_shape():
    cube = np.ones((8, 9, 10), dtype=np.uint16)

    widget = ShowEDS(
        cube,
        roi=(2, 3, 4, 5),
        roi_shape="oval",
        saved_rois=[{"name": "ellipse", "row": 2, "col": 3, "height": 4, "width": 5, "shape": "oval"}],
    )

    assert widget.roi_shape == "ellipse"
    assert widget.roi_row == 2
    assert widget.roi_col == 3
    assert widget.roi_height == 4
    assert widget.roi_width == 5
    assert widget.saved_rois == [{"name": "ellipse", "row": 2, "col": 3, "height": 4, "width": 5, "shape": "ellipse"}]


def test_showeds_export_html_writes_standalone_state(tmp_path):
    cube = np.ones((4, 5, 6), dtype=np.float32)
    widget = ShowEDS(cube, title="EDS Export", band=(1, 4), roi=(1, 1, 2, 3), log_spectrum=True)

    out = widget.export_html(tmp_path / "showeds.html")

    html = out.read_text()
    assert "EDS Export" in html
    assert "application/vnd.jupyter.widget-state+json" in html
    assert "ShowEDS" in html


def test_showeds_frontend_export_request_prepares_payload():
    cube = np.ones((3, 4, 5), dtype=np.uint16)
    widget = ShowEDS(cube, title="EDS GUI Export")

    widget.export_request = json.dumps(
        {"mode": "single", "id": "req-1", "filename": "eds_gui.html", "download": True}
    )

    assert widget.export_payload_id == "req-1"
    assert widget.export_filename == "eds_gui.html"
    assert b"EDS GUI Export" in widget.export_payload
    assert widget.export_status.startswith("Ready eds_gui.html")

    widget.export_request = json.dumps({"mode": "clear", "id": "clear"})

    assert widget.export_payload == b""
    assert widget.export_payload_id == ""
    assert widget.export_filename == ""


def test_showeds_sidecar_binned_export_request_prepares_portable_payload(tmp_path):
    cube = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
    energy = np.linspace(0, 7, 8, dtype=np.float32)
    sidecar = prepare_spectrum_image_sidecar(cube, energy, tmp_path / "eds")
    widget = ShowEDS.from_sidecar(
        "/files/eds/",
        sidecar_dir=sidecar,
        title="EDS Binned Export",
        band=(2, 4),
        roi=(0, 0, 4, 4),
    )

    widget.export_request = json.dumps(
        {"mode": "single", "downsample": 2, "id": "req-binned", "filename": "eds_binned.html", "download": True}
    )

    assert widget.export_sidecar_bytes > 0
    assert widget.export_payload_id == "req-binned"
    assert widget.export_filename == "eds_binned.html"
    assert b"EDS Binned Export sum-binned 2x/2x" in widget.export_payload
    assert widget.export_status.startswith("Ready eds_binned.html")
    assert "single sum-binned 2x" in widget.export_status


def test_showeds_binned_export_scales_saved_rois_and_bands(tmp_path):
    cube = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
    energy = np.linspace(0, 7, 8, dtype=np.float32)
    sidecar = prepare_spectrum_image_sidecar(cube, energy, tmp_path / "eds")
    widget = ShowEDS.from_sidecar(
        "/files/eds/",
        sidecar_dir=sidecar,
        title="EDS Binned Presets",
        band=(2, 4),
        roi=(0, 0, 4, 4),
        saved_rois=[{"name": "corner", "row": 2, "col": 2, "height": 2, "width": 4, "shape": "circle"}],
        saved_bands=[{"name": "peak", "start": 2, "end": 6}],
        pixel_size=0.5,
        pixel_unit="nm",
        smooth=True,
        selected_elements=["Au", "Cu"],
        auto_identify=False,
    )
    widget.map_zoom = 2.0
    widget.map_view_row = 2.0
    widget.map_view_col = 4.0
    widget.spectrum_view_start = 2.0
    widget.spectrum_view_end = 6.0

    binned, _label = widget._export_widget_for_mode("single", binning=2)

    assert binned.saved_rois == [{"name": "corner", "row": 1, "col": 1, "height": 1, "width": 1, "shape": "circle"}]
    assert binned.saved_bands == [{"name": "peak", "start": 1, "end": 3}]
    assert binned.pixel_size == 1.0
    assert binned.pixel_unit == "nm"
    assert binned.smooth is widget.smooth
    assert binned.map_zoom == 2.0
    assert binned.map_view_row == 1.0
    assert binned.map_view_col == 1.5
    assert binned.spectrum_view_start == 0.0
    assert binned.spectrum_view_end == 4.0
    assert binned.selected_elements == ["Au", "Cu"]
    assert binned.auto_identify is False


def test_showeds_export_html_keeps_legacy_mode_aliases(tmp_path):
    cube = np.ones((4, 5, 6), dtype=np.float32)
    widget = ShowEDS(cube, title="EDS Legacy Export")

    out = widget.export_html(tmp_path / "legacy.html", mode="embedded")

    assert out.exists()
    assert widget.export_status.startswith("Exported legacy.html")

    binned = widget.export_html(tmp_path / "legacy_binned.html", mode="binned-2")

    assert binned.exists()
    assert "single sum-binned 2x" in widget.export_status

    downsampled = widget.export_html(tmp_path / "downsampled.html", downsample=2)

    assert downsampled.exists()
    assert "single sum-binned 2x" in widget.export_status


def test_showeds_accepts_custom_line_hints_and_candidate_filter():
    cube = np.ones((4, 4, 16), dtype=np.float32)
    energy = np.linspace(0, 4, 16, dtype=np.float32)
    widget = ShowEDS(
        cube,
        energy,
        line_hints=[{"element": "Xx", "line": "Ka1", "energy_keV": 1.23, "intensity": 1.0}],
        show_line_hints=False,
    )

    assert widget.show_line_hints is False
    assert widget.line_hints == [{"element": "Xx", "line": "Ka1", "energy_keV": 1.23, "intensity": 1.0}]

    filtered = ShowEDS(cube, energy, candidate_elements=["Au"])
    assert filtered.line_hints
    assert {line["element"] for line in filtered.line_hints} == {"Au"}


def test_eds_line_hints_include_common_synthetic_peaks():
    lines = eds_line_hints(0.0, 4.0, elements=["O", "Si", "Ca", "Au"])
    labels = {(line["element"], line["line"]) for line in lines}

    assert ("O", "Ka1") in labels
    assert ("Si", "Ka1") in labels
    assert ("Ca", "Ka1") in labels
    assert ("Au", "Ma") in labels


def test_bin_spectrum_image_preserves_counts_and_bins_axes():
    cube = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    energy = np.linspace(0, 7, 8, dtype=np.float32)
    base = np.ones((4, 6), dtype=np.float32)

    binned, binned_energy, binned_base = bin_spectrum_image(
        cube,
        energy,
        base_image=base,
        spatial_bin=2,
        energy_bin=4,
    )

    assert binned.shape == (2, 3, 2)
    assert np.isclose(binned.sum(), cube.sum())
    assert binned_energy.tolist() == [1.5, 5.5]
    assert binned_base.shape == (2, 3)
    assert np.isclose(binned_base.sum(), base.sum())


def test_showeds_rejects_non_cube():
    with pytest.raises(ValueError, match="3D cube"):
        ShowEDS(np.zeros((4, 5), dtype=np.float32))


def test_showeds_rejects_oversized_embedded_widget_state():
    cube = np.zeros((4, 5, 6), dtype=np.uint16)

    with pytest.raises(ValueError, match="in-browser copy.*widget buffer"):
        ShowEDS(cube, max_state_bytes=32)


def test_showeds_kernel_backend_uses_initial_buffers_without_cube_bytes():
    initial_map = np.ones((4, 5), dtype=np.float32)
    initial_spectrum = np.arange(6, dtype=np.float32)

    widget = ShowEDS(None, initial_map=initial_map, initial_spectrum=initial_spectrum)

    assert widget.compute_backend == "kernel"
    assert widget.n_rows == 4
    assert widget.n_cols == 5
    assert widget.n_energy == 6
    assert widget.cube_bytes == b""
    assert len(widget.initial_map_bytes) == initial_map.size * 4
    assert len(widget.initial_spectrum_bytes) == initial_spectrum.size * 4


def test_prepare_spectrum_image_sidecar_writes_exact_prefix_and_integrals(tmp_path):
    cube = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    energy = np.linspace(0, 1, 5, dtype=np.float32)

    out = prepare_spectrum_image_sidecar(cube, energy, tmp_path / "eds", energy_chunk=2)

    meta = __import__("json").loads((out / "meta.json").read_text())
    assert meta["rows"] == 3
    assert meta["cols"] == 4
    assert meta["n_energy"] == 5

    prefix = np.memmap(out / "energy_prefix_u32.bin", dtype="<u4", mode="r", shape=(6, 3, 4))
    assert np.array_equal(prefix[0], np.zeros((3, 4), dtype=np.uint32))
    assert np.array_equal(prefix[3] - prefix[1], cube[:, :, 1:3].sum(axis=2, dtype=np.uint32))

    sat = np.memmap(out / meta["spatial_prefix"], dtype="<u4", mode="r", shape=(4, 5, 5))
    plane = cube[:, :, 1].astype(np.uint32)
    assert int(sat[3, 4, 1]) == int(plane.sum())
    assert int(sat[3, 4, 1] - sat[1, 4, 1] - sat[3, 2, 1] + sat[1, 2, 1]) == int(plane[1:3, 2:4].sum())


def test_prepare_spectrum_image_sidecar_refuses_oversized_prefix_cache(tmp_path):
    class SparseVendorCube:
        ndim = 3
        shape = (2048, 2048, 4096)
        dtype = np.dtype(np.uint16)

    energy = np.arange(4096, dtype=np.float32)

    with pytest.raises(ValueError, match="prefix-cache data folder would be too large"):
        prepare_spectrum_image_sidecar(SparseVendorCube(), energy, tmp_path / "eds", max_sidecar_bytes=1024)

    estimated = _estimate_spectrum_image_sidecar_bytes(SparseVendorCube.shape, include_base_image=False)
    assert estimated > 100 * 1024**3


def test_lazy_spatial_binning_matches_eager_binning_without_materializing_first():
    da = pytest.importorskip("dask.array")
    cube = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
    base = cube.sum(axis=2)
    energy = np.linspace(0, 7, 8, dtype=np.float32)
    lazy_cube = da.from_array(cube, chunks=(2, 3, 4))
    lazy_base = da.from_array(base, chunks=(2, 3))

    binned, axis, binned_base = _sum_bin_spectrum_image_lazy(
        lazy_cube,
        energy,
        base_image=lazy_base,
        spatial_bin=2,
        energy_bin=2,
    )
    expected, expected_axis, expected_base = bin_spectrum_image(cube, energy, base_image=base, spatial_bin=2, energy_bin=2)

    assert hasattr(binned, "compute")
    assert np.array_equal(binned.compute(), expected)
    assert np.allclose(axis, expected_axis)
    assert np.array_equal(binned_base.compute(), expected_base)


def test_showeds_from_emd_auto_uses_native_lazy_for_no_bin_emd(monkeypatch, tmp_path):
    cube = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
    energy = np.linspace(0, 5, 6, dtype=np.float32)
    base = cube.sum(axis=2)
    sidecar_dir = tmp_path / "sidecar"

    monkeypatch.setattr(
        showeds_module,
        "_read_emd_spectrum_image",
        lambda *args, **kwargs: {
            "cube": cube,
            "energy_keV": energy,
            "base_image": base,
            "title": "Sparse EMD",
            "candidate_elements": ["Cu"],
            "source_shape": cube.shape,
            "path": str(tmp_path / "sparse.emd"),
        },
    )

    widget = ShowEDS.from_emd(
        tmp_path / "sparse.emd",
        sidecar_dir=sidecar_dir,
        max_sidecar_bytes=1024,
        energy=2.0,
        width=2.0,
    )

    assert widget.compute_backend == "kernel"
    assert widget.cube_bytes == b""
    assert not sidecar_dir.exists()
    assert widget.n_rows == 4
    assert widget.n_cols == 5
    assert widget.n_energy == 6
    assert np.frombuffer(widget.initial_map_bytes, dtype=np.float32).reshape(4, 5).sum() > 0


def test_showeds_kernel_emd_export_refuses_misleading_exact_html():
    initial_map = np.ones((4, 5), dtype=np.float32)
    initial_spectrum = np.arange(6, dtype=np.float32)
    widget = ShowEDS(
        None,
        np.arange(6, dtype=np.float32),
        initial_map=initial_map,
        initial_spectrum=initial_spectrum,
        lazy_path="sparse.emd",
    )

    with pytest.raises(ValueError, match="no Python kernel"):
        widget._export_widget_for_mode("single")


def test_showeds_from_emd_explicit_sidecar_refuses_oversized_prefix_cache(monkeypatch, tmp_path):
    cube = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
    energy = np.linspace(0, 5, 6, dtype=np.float32)

    monkeypatch.setattr(
        showeds_module,
        "_read_emd_spectrum_image",
        lambda *args, **kwargs: {
            "cube": cube,
            "energy_keV": energy,
            "base_image": None,
            "title": "Sparse EMD",
            "candidate_elements": ["Cu"],
            "source_shape": cube.shape,
            "path": str(tmp_path / "sparse.emd"),
        },
    )
    monkeypatch.setattr(
        showeds_module,
        "_estimate_spectrum_image_sidecar_bytes",
        lambda *args, **kwargs: 100 * 1024**3,
    )

    with pytest.raises(ValueError, match="exact prefix-cache data folder"):
        ShowEDS.from_emd(
            tmp_path / "sparse.emd",
            backend="sidecar",
            sidecar_dir=tmp_path / "sidecar",
            max_sidecar_bytes=1024,
        )


def test_showeds_sidecar_backend_uses_initial_buffers_without_cube_bytes():
    initial_map = np.ones((4, 5), dtype=np.float32)
    initial_spectrum = np.arange(6, dtype=np.float32)

    widget = ShowEDS.from_sidecar(
        "files/eds_sidecar",
        initial_map=initial_map,
        initial_spectrum=initial_spectrum,
        energy_keV=np.arange(6, dtype=np.float32),
    )

    assert widget.compute_backend == "sidecar"
    assert widget.sidecar_url == "files/eds_sidecar/"
    assert widget.cube_bytes == b""


def test_showeds_from_sidecar_can_load_startup_state_from_sidecar_dir(tmp_path):
    cube = np.arange(3 * 4 * 8, dtype=np.uint16).reshape(3, 4, 8)
    energy = np.linspace(0, 7, 8, dtype=np.float32)
    sidecar = prepare_spectrum_image_sidecar(cube, energy, tmp_path / "eds")

    widget = ShowEDS.from_sidecar(
        "/files/eds/",
        sidecar_dir=sidecar,
        energy=2.5,
        width=2.0,
        roi=(1, 1, 2, 2),
    )

    assert widget.compute_backend == "sidecar"
    assert widget.band_start == 2
    assert widget.band_end == 4
    assert widget.roi_row == 1
    assert widget.roi_col == 1
    assert widget.roi_height == 2
    assert widget.roi_width == 2
    expected_map = cube[:, :, 2:4].sum(axis=2).astype(np.float32)
    expected_spectrum = cube[1:3, 1:3, :].sum(axis=(0, 1)).astype(np.float32)
    assert np.frombuffer(widget.initial_map_bytes, dtype=np.float32).reshape(3, 4).tolist() == expected_map.tolist()
    assert np.frombuffer(widget.initial_spectrum_bytes, dtype=np.float32).tolist() == expected_spectrum.tolist()


def test_showeds_sidecar_circle_roi_uses_exact_pixel_mask(tmp_path):
    cube = np.arange(5 * 5 * 4, dtype=np.uint16).reshape(5, 5, 4)
    energy = np.arange(4, dtype=np.float32)
    sidecar = prepare_spectrum_image_sidecar(cube, energy, tmp_path / "eds")

    loaded = load_spectrum_image_sidecar(sidecar, roi=(1, 1, 3, 3), roi_shape="circle")

    yy, xx = np.ogrid[:3, :3]
    mask = ((yy + 0.5 - 1.5) ** 2 + (xx + 0.5 - 1.5) ** 2) <= 1.5**2
    expected = (cube[1:4, 1:4, :] * mask[:, :, None]).sum(axis=(0, 1)).astype(np.float32)
    assert loaded["roi"] == (1, 1, 3, 3)
    assert loaded["initial_spectrum"].tolist() == expected.tolist()


def test_showeds_sidecar_ellipse_roi_uses_exact_pixel_mask(tmp_path):
    cube = np.arange(5 * 6 * 4, dtype=np.uint16).reshape(5, 6, 4)
    energy = np.arange(4, dtype=np.float32)
    sidecar = prepare_spectrum_image_sidecar(cube, energy, tmp_path / "eds")

    loaded = load_spectrum_image_sidecar(sidecar, roi=(1, 1, 3, 4), roi_shape="ellipse")

    yy, xx = np.ogrid[:3, :4]
    cy = 1.5
    cx = 2.0
    mask = (((yy + 0.5 - cy) / 1.5) ** 2 + ((xx + 0.5 - cx) / 2.0) ** 2) <= 1.0
    expected = (cube[1:4, 1:5, :] * mask[:, :, None]).sum(axis=(0, 1)).astype(np.float32)
    assert loaded["roi"] == (1, 1, 3, 4)
    assert loaded["initial_spectrum"].tolist() == expected.tolist()


def test_showeds_stream_sidecar_startup_spectrum_matches_sparse_events(tmp_path):
    rows, cols, n_energy = 4, 5, 6
    events_by_pixel: list[list[int]] = []
    for pixel in range(rows * cols):
        row, col = divmod(pixel, cols)
        events_by_pixel.append([row % n_energy, (col + 1) % n_energy, (row + col) % n_energy])

    pixel_offsets = np.zeros(rows * cols + 1, dtype=np.uint32)
    pixel_channels_list: list[int] = []
    for pixel, channels in enumerate(events_by_pixel):
        pixel_channels_list.extend(channels)
        pixel_offsets[pixel + 1] = len(pixel_channels_list)
    pixel_channels = np.asarray(pixel_channels_list, dtype=np.uint16)

    channel_pixels_list: list[int] = []
    channel_offsets = np.zeros(n_energy + 1, dtype=np.uint32)
    for channel in range(n_energy):
        channel_offsets[channel] = len(channel_pixels_list)
        for pixel, channels in enumerate(events_by_pixel):
            channel_pixels_list.extend([pixel] * channels.count(channel))
    channel_offsets[n_energy] = len(channel_pixels_list)
    channel_pixels = np.asarray(channel_pixels_list, dtype=np.uint32)

    sidecar = tmp_path / "stream"
    sidecar.mkdir()
    (sidecar / "pixel_offsets_u32.bin").write_bytes(pixel_offsets.astype("<u4", copy=False).tobytes())
    (sidecar / "pixel_channels_u16.bin").write_bytes(pixel_channels.astype("<u2", copy=False).tobytes())
    (sidecar / "channel_offsets_u32.bin").write_bytes(channel_offsets.astype("<u4", copy=False).tobytes())
    (sidecar / "channel_pixels_u32.bin").write_bytes(channel_pixels.astype("<u4", copy=False).tobytes())
    (sidecar / "base_f32.bin").write_bytes(np.ones((rows, cols), dtype="<f4").tobytes())
    (sidecar / "meta.json").write_text(
        json.dumps(
            {
                "format": "quantem.widget.showeds.stream-sidecar.v1",
                "rows": rows,
                "cols": cols,
                "n_energy": n_energy,
                "n_events": int(pixel_channels.size),
                "energy_keV": np.arange(n_energy, dtype=np.float32).tolist(),
                "channel_offsets": "channel_offsets_u32.bin",
                "channel_pixels": "channel_pixels_u32.bin",
                "pixel_offsets": "pixel_offsets_u32.bin",
                "pixel_channels": "pixel_channels_u16.bin",
                "base_image": "base_f32.bin",
            }
        )
    )

    loaded = load_spectrum_image_sidecar(sidecar, band=(2, 4), roi=(1, 1, 2, 3))

    expected_channels: list[int] = []
    for row in range(1, 3):
        for col in range(1, 4):
            expected_channels.extend(events_by_pixel[row * cols + col])
    expected_spectrum = np.bincount(expected_channels, minlength=n_energy).astype(np.float32)
    expected_map = np.zeros(rows * cols, dtype=np.float32)
    for pixel, channels in enumerate(events_by_pixel):
        expected_map[pixel] = sum(1 for channel in channels if 2 <= channel < 4)

    assert loaded["roi"] == (1, 1, 2, 3)
    assert loaded["initial_spectrum"].tolist() == expected_spectrum.tolist()
    assert loaded["initial_map"].reshape(-1).tolist() == expected_map.tolist()
