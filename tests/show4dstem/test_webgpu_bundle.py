import json
import re
import stat

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

from quantem.widget import Show4DSTEM
from quantem.widget.cli import _prepare_show4dstem_webgpu_output_dir
from quantem.widget.show4dstem_webgpu_export import (
    build_lazy_show4dstem_sidecar,
    bundle_master_urls,
    export_show4dstem_webgpu_bundle,
    export_show4dstem_hdf5_viewer,
)


def _write_arina_family(folder, stem, n_frames=16, det=32, bad_pixels=None, write_mask=True):
    frames = np.zeros((n_frames, det, det), np.uint16)
    frames[:, det // 2, det // 2] = 500  # a >255 count so uint16 matters
    pixel_mask = np.zeros((det, det), np.uint32)
    for row, col in bad_pixels or []:
        pixel_mask[int(row), int(col)] = 1
    with h5py.File(folder / f"{stem}_data_000001.h5", "w") as f:
        f.create_group("entry/data").create_dataset(
            "data", data=frames, chunks=(1, det, det),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
    with h5py.File(folder / f"{stem}_master.h5", "w") as f:
        g = f.create_group("entry/instrument/detector/detectorSpecific")
        g.create_dataset("ntrigger", data=n_frames)
        g.create_dataset("nimages", data=1)
        if write_mask:
            g.create_dataset("pixel_mask", data=pixel_mask)


def test_bundle_export_writes_launcher_viewer_and_vendored_page(tmp_path):
    _write_arina_family(tmp_path, "tilt_a")
    _write_arina_family(tmp_path, "tilt_b")
    urls = bundle_master_urls(tmp_path)
    assert urls == ["../tilt_a_master.h5", "../tilt_b_master.h5"]
    widget = Show4DSTEM(
        np.zeros((1, 1, 1, 1), np.uint8), h5_urls=urls,
        scan_shape=(4, 4), detector_shape=(32, 32), backend="webgpu",
        view_mode="multiple", compare_max_panels=3, compare_group_mode="all",
        precompute_virtual_images=False, verbose=False,
    )
    try:
        launcher = export_show4dstem_webgpu_bundle(widget, tmp_path, port=8899)
    finally:
        widget.close()
    assert launcher.name == "Show4DSTEM.command"
    assert launcher.stat().st_mode & stat.S_IXUSR
    command = launcher.read_text()
    assert "8899" in command and "serve_range.py" in command
    assert "/index.html" in command
    assert "__quantem_viewer_root__" in command
    assert "EXISTING_ROOT" in command
    assert "PORT=$((PORT + 1))" in command
    assert "Reusing local Show4DSTEM viewer server for this folder" in command
    viewer = tmp_path / ".viewer"
    root_index = tmp_path / "index.html"
    assert root_index.exists()
    assert ".viewer/Show4DSTEM.html" in root_index.read_text(encoding="utf-8")
    assert "window.location.protocol === \"file:\"" in root_index.read_text(encoding="utf-8")
    assert "This Show4DSTEM package must be opened through its local range server" not in root_index.read_text(encoding="utf-8")
    for name in ("Show4DSTEM.html", "require.min.js", "embed-amd.js", "anywidget.min.js", "serve_range.py"):
        assert (viewer / name).exists(), name
    page = (viewer / "Show4DSTEM.html").read_text(encoding="utf-8")
    assert "cdnjs.cloudflare.com" not in page and "cdn.jsdelivr.net" not in page
    assert "__QT_REQUIRE_LOCAL_H5_FILES" in page
    assert "__QT_H5_DECODE_DTYPE" in page and "__BSLZ4_FRAME_WG" in page
    assert 'globalThis.__QT_H5_DECODE_DTYPE ??= "u2"' in page
    assert "globalThis.__QT_H5_FORCE_LOW8 ??= false" in page
    assert "globalThis.__BSLZ4_PIPELINE_STAGING ??= false" in page
    assert "../tilt_a_master.h5" in page


def test_bundle_export_uses_low8_for_audited_uint8_h5(tmp_path):
    _write_arina_family(tmp_path, "tilt_a")
    urls = bundle_master_urls(tmp_path)
    widget = Show4DSTEM(
        np.zeros((1, 1, 1, 1), np.uint8),
        h5_urls=urls,
        h5_uint8_lossless=True,
        scan_shape=(4, 4),
        detector_shape=(32, 32),
        backend="webgpu",
        precompute_virtual_images=False,
        verbose=False,
    )
    try:
        export_show4dstem_webgpu_bundle(widget, tmp_path, port=8899, h5_decode_dtype="uint8")
    finally:
        widget.close()
    page = (tmp_path / ".viewer" / "Show4DSTEM.html").read_text(encoding="utf-8")
    assert 'globalThis.__QT_H5_DECODE_DTYPE ??= "uint8"' in page
    assert "globalThis.__QT_H5_FORCE_LOW8 ??= true" in page
    assert "globalThis.__BSLZ4_LOW8_ONLY ??= true" in page
    assert "globalThis.__QT_H5_MAX_RESIDENT ??= 1" not in page


def test_h5_export_sets_file_protocol_local_file_guard(tmp_path):
    _write_arina_family(tmp_path, "tilt_a")
    widget = Show4DSTEM(
        np.zeros((1, 1, 1, 1), np.uint8),
        h5_url="sources/tilt_a_master.h5",
        scan_shape=(4, 4),
        detector_shape=(32, 32),
        backend="webgpu",
        precompute_virtual_images=False,
        verbose=False,
    )
    try:
        out = tmp_path / "index.html"
        widget.export_html(out, title="h5 direct", dtype="uint8", det_bin=1)
    finally:
        widget.close()
    page = out.read_text(encoding="utf-8")
    assignment = "__QT_REQUIRE_LOCAL_H5_FILES = true"
    assert assignment in page
    assert page.find(assignment) < page.find("__QT_H5_DECODE_DTYPE")
    assert page.find(assignment) < page.find("sources/tilt_a_master.h5")


def test_direct_hdf5_viewer_links_source_without_copying(tmp_path):
    """C1: project viewer, expect direct HTML and the same raw file inodes."""

    source = tmp_path / "source"
    project = tmp_path / "project"
    source.mkdir()
    _write_arina_family(source, "tilt_a")

    html = export_show4dstem_hdf5_viewer(
        source / "tilt_a_master.h5",
        project,
        scan_shape=(4, 4),
        detector_shape=(32, 32),
        title="Tilt A Show4DSTEM",
    )

    assert html == project / ".viewer" / "Show4DSTEM.html"
    assert html.is_file()
    assert (project / "tilt_a_master.h5").samefile(
        source / "tilt_a_master.h5"
    )
    assert (project / "tilt_a_data_000001.h5").samefile(
        source / "tilt_a_data_000001.h5"
    )
    assert "../tilt_a_master.h5" in html.read_text(encoding="utf-8")


def test_direct_hdf5_viewer_can_anonymize_family_names(tmp_path):
    """C1: anonymous project, expect generic aliases with identical raw bytes."""

    source = tmp_path / "source"
    project = tmp_path / "project"
    source.mkdir()
    _write_arina_family(source, "private_scan")
    project.mkdir()
    (project / "stale_name_master.h5").write_bytes(b"stale")
    (project / "stale_name_data_000001.h5").write_bytes(b"stale")

    html = export_show4dstem_hdf5_viewer(
        source / "private_scan_master.h5",
        project,
        scan_shape=(4, 4),
        detector_shape=(32, 32),
        title="Dataset 001 Show4DSTEM",
        target_stem="dataset-001",
    )

    assert (project / "dataset-001_master.h5").samefile(
        source / "private_scan_master.h5"
    )
    assert (project / "dataset-001_data_000001.h5").samefile(
        source / "private_scan_data_000001.h5"
    )
    page = html.read_text(encoding="utf-8")
    assert "../dataset-001_master.h5" in page
    assert "private_scan" not in page
    assert not (project / "stale_name_master.h5").exists()
    assert not (project / "stale_name_data_000001.h5").exists()


def test_show4dstem_lazy_urls_export_uses_lazy_source(tmp_path):
    widget = Show4DSTEM(
        np.zeros((1, 1, 1, 1), np.uint8),
        lazy_urls=["tilt_00_lazy/", "tilt_01_lazy/"],
        scan_shape=(4, 4),
        detector_shape=(32, 32),
        backend="webgpu",
        precompute_virtual_images=False,
        verbose=False,
    )
    try:
        out = tmp_path / "lazy.html"
        widget.export_html(str(out), title="lazy", dtype="uint8", det_bin=1)
    finally:
        widget.close()
    page = out.read_text(encoding="utf-8")
    assert '"_lazy_urls": "[\\"tilt_00_lazy/\\", \\"tilt_01_lazy/\\"]"' in page
    assert '"_h5_url": ""' in page
    assert '"_h5_urls": ""' in page
    assert '"gpu_memory_label": "Browser WebGPU lazy source"' in page
    state_match = re.search(
        r'<script type="application/vnd\.jupyter\.widget-state\+json">\s*(.*?)\s*</script>',
        page,
        re.DOTALL,
    )
    assert state_match is not None
    state = json.loads(state_match.group(1))
    offline_buffers = [
        buffer
        for buffer in state["state"].values()
        for buffer in buffer.get("buffers", [])
        if buffer.get("path") == ["_offline_stack"]
    ]
    assert offline_buffers == [{"encoding": "base64", "path": ["_offline_stack"], "data": ""}]


def test_show4dstem_webgpu_cli_replaces_existing_export_dir(tmp_path):
    source = tmp_path / "source_data_000001.h5"
    source.write_bytes(b"source")
    out_dir = tmp_path / "seven-tilts_show4dstem_webgpu"
    out_dir.mkdir()
    (out_dir / "index.html").write_text("old html")
    (out_dir / "tilt_00_lazy").mkdir()
    (out_dir / "tilt_00_lazy" / "meta.json").write_text('{"old": true}')
    (out_dir / "tilt_00_data_000001.h5").symlink_to(source)

    assert _prepare_show4dstem_webgpu_output_dir(out_dir) is True

    assert out_dir.is_dir()
    assert list(out_dir.iterdir()) == []
    assert source.read_bytes() == b"source"


def test_build_lazy_show4dstem_sidecar_indexes_h5_ranges(tmp_path):
    _write_arina_family(tmp_path, "tilt_00", n_frames=16, det=32)

    url = build_lazy_show4dstem_sidecar(
        tmp_path,
        label="tilt_00",
        scan_shape=(4, 4),
        detector_shape=(32, 32),
    )

    lazy = tmp_path / "tilt_00_lazy"
    assert url == "tilt_00_lazy/"
    meta = __import__("json").loads((lazy / "meta.json").read_text())
    assert meta["SR"] == 4
    assert meta["SC"] == 4
    assert meta["D"] == 32
    assert meta["sourceDtype"] == "uint16"
    assert meta["files"] == ["../tilt_00_data_000001.h5"]
    index = np.memmap(lazy / "index.bin", mode="r", dtype=np.uint32, shape=(16, 3))
    assert np.all(index[:, 0] == 0)
    assert np.all(index[:, 1] > 0)
    assert np.all(index[:, 2] > 0)
    profile = np.memmap(
        lazy / "profile.bin",
        mode="r",
        dtype=np.float32,
        shape=(16, meta["NB"]),
    )
    assert np.all(profile[:, 0] == 500)
    com = np.memmap(lazy / "com.bin", mode="r", dtype=np.float32, shape=(2, 16))
    assert np.allclose(com[0], 16)
    assert np.allclose(com[1], 16)


def test_build_lazy_show4dstem_sidecar_applies_h5_hot_pixel_mask(tmp_path):
    det = 32
    center = (det // 2, det // 2)
    _write_arina_family(tmp_path, "tilt_00", n_frames=16, det=det, bad_pixels=[center])

    build_lazy_show4dstem_sidecar(
        tmp_path,
        label="tilt_00",
        scan_shape=(4, 4),
        detector_shape=(det, det),
    )

    lazy = tmp_path / "tilt_00_lazy"
    meta = json.loads((lazy / "meta.json").read_text())
    assert meta["badPixels"] == [center[0] * det + center[1]]
    profile = np.memmap(
        lazy / "profile.bin",
        mode="r",
        dtype=np.float32,
        shape=(16, meta["NB"]),
    )
    assert np.all(profile[:, 0] == 0)
    com = np.memmap(lazy / "com.bin", mode="r", dtype=np.float32, shape=(2, 16))
    assert np.allclose(com, 0)


def test_build_lazy_show4dstem_sidecar_masks_saturated_pixels_without_h5_mask(tmp_path):
    det = 32
    center = (det // 2, det // 2)
    _write_arina_family(tmp_path, "tilt_00", n_frames=16, det=det, write_mask=False)
    with h5py.File(tmp_path / "tilt_00_data_000001.h5", "r+") as f:
        f["entry/data/data"][0, center[0], center[1]] = np.iinfo(np.uint16).max

    build_lazy_show4dstem_sidecar(
        tmp_path,
        label="tilt_00",
        scan_shape=(4, 4),
        detector_shape=(det, det),
    )

    lazy = tmp_path / "tilt_00_lazy"
    meta = json.loads((lazy / "meta.json").read_text())
    assert meta["badPixels"] == [center[0] * det + center[1]]
    profile = np.memmap(
        lazy / "profile.bin",
        mode="r",
        dtype=np.float32,
        shape=(16, meta["NB"]),
    )
    assert np.all(profile[:, 0] == 0)


def test_bundle_export_requires_masters(tmp_path):
    widget = Show4DSTEM(np.zeros((2, 2, 4, 4), np.uint8), verbose=False)
    try:
        with pytest.raises(ValueError, match="master"):
            export_show4dstem_webgpu_bundle(widget, tmp_path)
    finally:
        widget.close()
