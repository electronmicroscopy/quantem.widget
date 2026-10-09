"""Show4DSTEM over dense arrays, files and encoded acquisitions: what the tutorials rely on.

Every reduction the viewer shows is checked against a NumPy sum over the same
detector mask on the same counts. Files come from a small synthetic Arina-style
master written with h5py: read densely by the widget (no quantem.gpu needed), or
loaded through ``quantem.gpu.io.load`` when quantem.gpu and a GPU are present,
where the two paths must agree exactly.
"""

import gzip
import json
from pathlib import Path

import comm
import h5py
import hdf5plugin
import numpy as np
import pytest
import torch
from PIL import Image

from quantem.widget import Show4DSTEM, device
from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.show4dstem import preview, reader
from quantem.widget.show4dstem.detector import detector_mask, fit_probe, roi_mask
from quantem.widget.counts import tensor_dtype
from quantem.widget.show4dstem import export as html_export
from quantem.widget.show4dstem.export import bundle_master_urls, check_html_payload, export_data_array, write_webgpu_bundle

SCAN, DET = (6, 6), (16, 16)


def _counts(seed: int, scan=SCAN, det=DET) -> np.ndarray:
    """A bright disk of varying intensity on a weak background, as uint16 counts."""
    rows, cols = np.indices(det)
    disk = ((rows - 7.5) ** 2 + (cols - 7.5) ** 2) <= 3.0**2
    scale = np.random.default_rng(seed).integers(20, 60, scan, dtype=np.uint16)
    pattern = np.where(disk, 10, 1).astype(np.uint16)
    return (pattern[None, None] * scale[..., None, None]).astype(np.uint16)


def _write_master(folder: Path, stem: str, values: np.ndarray) -> Path:
    """One Arina master linking its frames from an external bitshuffle-LZ4 data file."""
    rows, cols, det_rows, det_cols = values.shape
    with h5py.File(folder / f"{stem}_data_000001.h5", "w") as handle:
        handle.create_dataset("entry/data/data", data=values.reshape(rows * cols, det_rows, det_cols),
                              chunks=(1, det_rows, det_cols), **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    master = folder / f"{stem}_master.h5"
    with h5py.File(master, "w") as handle:
        handle.require_group("entry/data")["data_000001"] = h5py.ExternalLink(f"{stem}_data_000001.h5", "entry/data/data")
        specific = handle.require_group("entry/instrument/detector/detectorSpecific")
        specific.create_dataset("ntrigger", data=rows * cols)
        specific.create_dataset("nimages", data=1)
    return master


def _write_flagged_master(folder: Path, stem: str, values: np.ndarray, flagged: list[tuple[int, int]]) -> Path:
    """A uint32 Arina master over two linked data files whose ``pixel_mask`` flags ``flagged``.

    Flagged pixels hold 2**32 - 1 on disk, as a Dectris detector writes them, so a
    reader that forgets the mask cannot narrow the counts and gives wrong sums.
    """
    rows, cols, det_rows, det_cols = values.shape
    raw = values.astype(np.uint32).reshape(rows * cols, det_rows, det_cols)
    mask = np.zeros((det_rows, det_cols), np.uint32)
    for row, col in flagged:
        mask[row, col] = 1
        raw[:, row, col] = 2**32 - 1
    half = len(raw) // 2
    master = folder / f"{stem}_master.h5"
    with h5py.File(master, "w") as handle:
        for index, frames in enumerate((raw[:half], raw[half:]), start=1):
            name = f"{stem}_data_{index:06d}.h5"
            with h5py.File(folder / name, "w") as data_file:
                data_file.create_dataset("entry/data/data", data=frames, chunks=(1, det_rows, det_cols),
                                         **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
            handle.require_group("entry/data")[f"data_{index:06d}"] = h5py.ExternalLink(name, "entry/data/data")
        specific = handle.require_group("entry/instrument/detector/detectorSpecific")
        specific.create_dataset("ntrigger", data=rows * cols)
        specific.create_dataset("nimages", data=1)
        specific.create_dataset("pixel_mask", data=mask)
    return master


def _median_corrected(values: np.ndarray, flagged: list[tuple[int, int]]) -> np.ndarray:
    """Reference hot-pixel rule: the median of the unflagged 3x3 neighbours, rounded down."""
    corrected = values.astype(np.int64)
    rows, cols = values.shape[-2:]
    for row, col in flagged:
        neighbours = [(r, c) for r in range(row - 1, row + 2) for c in range(col - 1, col + 2)
                      if (r, c) != (row, col) and 0 <= r < rows and 0 <= c < cols and (r, c) not in flagged]
        stack = np.stack([values[..., r, c] for r, c in neighbours], axis=-1).astype(np.int64)
        corrected[..., row, col] = np.floor(np.median(stack, axis=-1)).astype(np.int64)
    return corrected


@pytest.fixture
def accelerator():
    """Encoded acquisitions need quantem.gpu and a CUDA or MPS device; there is no CPU route."""
    if not gpu_adapter.accelerator_ready():
        pytest.skip("encoded acquisitions need quantem.gpu (quantem.widget[cuda] or [mps]) and a CUDA or MPS GPU")


def _bf(counts: np.ndarray, center, radius: float, inner: float = 0.0) -> np.ndarray:
    """NumPy reference virtual image: the exact sum of the detector pixels between two radii."""
    mask = detector_mask(center, inner, radius, counts.shape[-2:])
    return counts[..., mask].sum(axis=-1, dtype=np.uint64).astype(np.float32)


def _mean_dp(counts: np.ndarray) -> np.ndarray:
    """NumPy reference mean pattern: exact integer total divided once in float64."""
    flat = counts.reshape(-1, *counts.shape[-2:])
    return (flat.sum(axis=0, dtype=np.uint64) / len(flat)).astype(np.float32)


def _image(widget) -> np.ndarray:
    return np.frombuffer(widget.virtual_image_bytes, np.float32).reshape(widget.shape_rows, widget.shape_cols)


def _pattern(widget) -> np.ndarray:
    return np.frombuffer(widget.frame_bytes, np.float32).reshape(widget.det_rows, widget.det_cols)


# ---------------------------------------------------------------------------
# Encoded acquisition through quantem.gpu.io.load
# ---------------------------------------------------------------------------


def test_encoded_master_opens_with_detector_products(tmp_path, accelerator):
    counts = _counts(0)
    master = _write_master(tmp_path, "gold", counts)
    with gpu_adapter.load_acquisition(master) as loaded:
        widget = Show4DSTEM(loaded)
        try:
            center, radius = fit_probe(_mean_dp(counts))
            assert (widget.center_row, widget.center_col) == pytest.approx(center)
            assert widget.bf_radius == pytest.approx(radius)
            np.testing.assert_array_equal(_image(widget), _bf(counts, center, radius))
            widget.apply_preset("adf")
            np.testing.assert_array_equal(_image(widget), _bf(counts, center, 2 * radius, inner=radius))
            widget.pos_row, widget.pos_col = 2, 3
            np.testing.assert_array_equal(_pattern(widget), counts[2, 3])
            exported = export_data_array(widget, "uint16", 1, 1)
            np.testing.assert_array_equal(exported, counts)
        finally:
            widget.close()
        # offline packing reads the acquisition in scan windows into the bytes a dense array packs
        packed = Show4DSTEM(loaded, offline=True, offline_dtype="uint16", verbose=False)
        dense = Show4DSTEM(counts, offline=True, offline_dtype="uint16", precompute_virtual_images=False, verbose=False)
        try:
            assert packed.offline and gzip.decompress(packed._offline_stack) == gzip.decompress(dense._offline_stack)
        finally:
            packed.close()
            dense.close()
        with pytest.raises(ValueError, match="h5_urls"):
            Show4DSTEM(loaded, h5_urls=["../scan_master.h5"], scan_shape=SCAN, detector_shape=DET)


def test_a_path_opened_on_the_gpu_with_device_auto_names_the_device_once(tmp_path, accelerator, monkeypatch, capsys):
    monkeypatch.setattr(device, "_printed", set())
    master = _write_master(tmp_path, "auto", _counts(3))
    for _ in range(2):
        widget = Show4DSTEM(master)
        widget.close()
    assert capsys.readouterr().out == f'quantem.widget: device="auto" selected {device.device_name(widget._device)}.\n'


def test_two_acquisitions_compare_under_one_detector(tmp_path, accelerator):
    counts = [_counts(seed) for seed in range(2)]
    masters = [_write_master(tmp_path, f"scan_{idx}", values) for idx, values in enumerate(counts)]
    loaded = gpu_adapter.load_acquisition(masters)
    widget = Show4DSTEM(loaded, compare_dp_mode="all")
    try:
        assert widget.view_mode == "multiple"
        assert list(widget.frame_labels) == ["scan_0", "scan_1"]
        mask = widget._current_detector_mask()
        panels = np.frombuffer(widget.compare_virtual_image_bytes, np.float32).reshape(2, *SCAN)
        for panel, values in zip(panels, counts, strict=True):
            np.testing.assert_allclose(panel, values[..., mask].sum(axis=-1) / mask.sum(), rtol=1e-6)
        widget.pos_row, widget.pos_col = 1, 4
        patterns = np.frombuffer(widget.compare_diffraction_bytes, np.float32).reshape(2, *DET)
        np.testing.assert_array_equal(patterns, np.stack([values[1, 4] for values in counts]))
        widget.frame_idx = 1
        np.testing.assert_array_equal(_pattern(widget), counts[1][1, 4])
    finally:
        widget.close()
        for acquisition in loaded:
            acquisition.close()


def test_from_folder_fills_two_masters_and_free_closes_them(tmp_path, accelerator):
    counts = [_counts(seed) for seed in range(2)]
    for idx, values in enumerate(counts):
        _write_master(tmp_path, f"scan_{idx:02d}", values)
    widget = Show4DSTEM.from_folder(tmp_path, watch=False)
    try:
        widget.wait_for_folder()
        assert widget.n_frames == 2
        assert list(widget.frame_labels) == ["scan_00", "scan_01"]
        assert widget.compare_panel_indices == [0, 1]
        acquisitions = list(widget._owned_acquisitions)
        widget.free()
        assert widget._owned_acquisitions == []
        with pytest.raises((RuntimeError, ValueError)):
            gpu_adapter.prepare_session(acquisitions[0]).masked_sum(np.ones(DET, bool))
    finally:
        widget.close()
    with pytest.raises(ValueError, match="at least 3"):
        Show4DSTEM.from_folder(tmp_path, watch=False, min_masters=3)


def test_a_watched_folder_emptied_of_masters_polls_without_error(tmp_path, accelerator, monkeypatch):
    _write_master(tmp_path, "scan_00", _counts(0))
    widget = Show4DSTEM.from_folder(tmp_path, watch=False)
    try:
        widget.wait_for_folder()
        for path in tmp_path.iterdir():
            path.unlink()
        # the poll lists the folder itself; quantem.gpu's discover raises on an empty match
        monkeypatch.setattr(gpu_adapter, "discover_masters", lambda *args, **kwargs: pytest.fail("discover called"))
        assert widget.poll_folder() == []
        assert widget.n_frames == 1
    finally:
        widget.close()


# ---------------------------------------------------------------------------
# Dense arrays and tensors
# ---------------------------------------------------------------------------


def test_numpy_viewer_matches_detector_and_pattern_selection():
    counts = _counts(1)
    widget = Show4DSTEM(counts, center=(7.5, 7.5), bf_radius=3.0, precompute_virtual_images=False)
    try:
        assert widget.roi_mode == "circle" and widget.roi_radius == 3.0
        np.testing.assert_array_equal(_image(widget), _bf(counts, (7.5, 7.5), 3.0))
        widget.roi_mode = "point"
        widget.roi_center = [7.0, 9.0]
        np.testing.assert_array_equal(_image(widget), counts[:, :, 7, 9])
        widget.pos_row, widget.pos_col = 5, 0
        np.testing.assert_array_equal(_pattern(widget), counts[5, 0])
        widget.vi_roi_mode = "circle"
        widget.vi_roi_center = [2.0, 2.0]
        widget.vi_roi_radius = 1.0
        inside = [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]
        expected = np.mean([counts[row, col] for row, col in inside], axis=0)
        np.testing.assert_allclose(np.frombuffer(widget.vi_roi_dp_bytes, np.float32).reshape(DET), expected, rtol=1e-6)
    finally:
        widget.close()


def test_scan_position_outside_the_scan_stops_at_its_edge():
    counts = _counts(4)
    widget = Show4DSTEM(counts, precompute_virtual_images=False)
    try:
        widget.pos_row, widget.pos_col = 10, -3
        assert (widget.pos_row, widget.pos_col) == (5, 0)
        np.testing.assert_array_equal(_pattern(widget), counts[5, 0])
        widget.pos_row, widget.pos_col = 0, 9  # not the next row's pattern at flat index 9
        assert (widget.pos_row, widget.pos_col) == (0, 5)
        np.testing.assert_array_equal(_pattern(widget), counts[0, 5])
        np.testing.assert_array_equal(widget.pattern(5, 0), counts[5, 0])
        with pytest.raises(IndexError, match="outside the 6 x 6 scan"):
            widget.pattern(0, 6)
    finally:
        widget.close()


def test_mask_goldens_shared_with_the_browser_are_the_python_masks():
    # js/show4dstem/detectorInteraction.test.ts checks the drawn detector overlay against these pixels
    goldens = json.loads((Path(__file__).resolve().parents[2] / "js" / "show4dstem" / "maskGoldens.json").read_text())
    for name, case in goldens["cases"].items():
        mask = roi_mask(case["mode"], tuple(goldens["detector_shape"]), center_row=case["center"][0], center_col=case["center"][1],
                        radius=case["radius"], radius_inner=case["radius_inner"], width=0.0, height=0.0)
        assert np.flatnonzero(mask).tolist() == case["selected"], name


def test_detector_near_a_preset_radius_sums_its_own_pixels():
    # A radius within a pixel of the BF preset selects fewer pixels; the cached BF image must not stand in for it.
    counts = _counts(2)
    widget = Show4DSTEM(counts, center=(7.5, 7.5), bf_radius=3.0)
    try:
        np.testing.assert_array_equal(_image(widget), _bf(counts, (7.5, 7.5), 3.0))
        widget.roi_radius = 2.2
        np.testing.assert_array_equal(_image(widget), _bf(counts, (7.5, 7.5), 2.2))
        assert not np.array_equal(_bf(counts, (7.5, 7.5), 2.2), _bf(counts, (7.5, 7.5), 3.0))
    finally:
        widget.close()


def test_five_dimensional_tensor_pages_and_averages_patterns():
    data_t = torch.stack([torch.as_tensor(_counts(seed).astype(np.float32)) for seed in range(3)])
    widget = Show4DSTEM(data_t, view_mode="multiple", compare_max_panels=2, frame_labels=["a", "b", "c"])
    try:
        assert widget.compare_page_count == 2
        assert widget.compare_panel_indices == [0, 1]
        average = data_t[:2, widget.pos_row, widget.pos_col].mean(0).numpy()
        np.testing.assert_allclose(_pattern(widget), average, rtol=1e-6)
        widget.compare_page_idx = 1
        assert widget.compare_panel_indices == [2]
        assert widget.frame_idx == 2
        widget.compare_hidden_panels = [2]
        assert widget.compare_panel_indices == [] and "hidden" in widget.compare_status
        widget.compare_group_mode = "all"
        assert widget.compare_panel_indices == [0, 1]
        assert widget.frame_idx == 0  # the selected dataset follows the visible panels
        widget.compare_dp_mode = "selected"
        widget.frame_idx = 1
        np.testing.assert_array_equal(_pattern(widget), data_t[1, widget.pos_row, widget.pos_col].numpy())
    finally:
        widget.close()


def test_free_releases_the_packed_browser_counts():
    widget = Show4DSTEM(_counts(1), offline=True, precompute_virtual_images=False, verbose=False)
    try:
        assert widget.offline and len(widget._offline_stack) > 0
        widget.free()
        assert not widget.offline and widget._offline_stack == b"" and widget.export_payload == b""
    finally:
        widget.close()


def test_state_round_trips_through_dict_and_file(tmp_path):
    counts = _counts(2)
    widget = Show4DSTEM(counts, title="round trip", precompute_virtual_images=False)
    try:
        widget.apply_preset("haadf")
        widget.dp_colormap = "viridis"
        widget.vi_scale_mode = "log"
        widget.pos_row, widget.pos_col = 1, 2
        state = widget.state_dict()
        widget.save(tmp_path / "state.json")
    finally:
        widget.close()
    assert json.loads((tmp_path / "state.json").read_text())["widget_name"] == "Show4DSTEM"
    for restored in (Show4DSTEM(counts, state=state, precompute_virtual_images=False),
                     Show4DSTEM(counts, state=tmp_path / "state.json", precompute_virtual_images=False)):
        try:
            assert restored.state_dict() == state
            assert restored.roi_mode == "annular" and restored.dp_colormap == "viridis"
            np.testing.assert_array_equal(_pattern(restored), counts[1, 2])
        finally:
            restored.close()


def test_product_map_from_compute_ssb_becomes_the_image(monkeypatch):
    counts = _counts(3)
    phase = np.linspace(-1, 1, counts[..., 0, 0].size, dtype=np.float32).reshape(SCAN)
    widget = Show4DSTEM(counts, precompute_virtual_images=False)
    try:
        monkeypatch.setattr(type(widget), "_ssb_phase", lambda self, **kwargs: (phase, phase * 2, phase * 3))
        np.testing.assert_array_equal(widget.compute_ssb(), phase)
        assert widget.vi_source == "SSB"
        assert widget.vi_product_labels == ["DPC_row", "DPC_col", "SSB"]
        np.testing.assert_array_equal(widget.virtual_image(), phase)
        stack = np.frombuffer(widget.vi_product_maps_bytes, np.float32).reshape(3, *SCAN)
        np.testing.assert_array_equal(stack[2], phase)
        assert widget.ssb_compute_status.startswith("SSB ready")
        widget.vi_source = "roi"
        np.testing.assert_array_equal(_image(widget), _bf(counts, (widget.center_row, widget.center_col), widget.bf_radius))
    finally:
        widget.close()


# ---------------------------------------------------------------------------
# Files read densely, without quantem.gpu
# ---------------------------------------------------------------------------

FLAGGED = [(0, 0), (4, 5), (9, 9), (9, 10)]  # a corner and two neighbouring flagged pixels


def test_master_read_densely_matches_numpy(tmp_path, monkeypatch, capsys):
    counts = _counts(7, scan=(6, 6))
    master = _write_flagged_master(tmp_path, "flagged", counts, FLAGGED)
    expected = _median_corrected(counts, FLAGGED)
    monkeypatch.setattr(gpu_adapter, "accelerator_ready", lambda: False)
    widget = Show4DSTEM(master, device="cpu")
    try:
        # uint16 counts; torch 2.2 (Intel Macs) has no uint16 and stores them as int32
        stored = str(tensor_dtype(np.uint16)).removeprefix("torch.")
        assert f"read flagged_master.h5 densely on the CPU into cpu: (6, 6, 16, 16) {stored}" in capsys.readouterr().out
        assert widget.title == "flagged" and widget._device.type == "cpu"
        np.testing.assert_array_equal(widget._session(0).mean_dp(), _mean_dp(expected))
        center, radius = fit_probe(_mean_dp(expected))
        assert (widget.center_row, widget.center_col, widget.bf_radius) == pytest.approx((*center, radius))
        np.testing.assert_array_equal(_image(widget), _bf(expected, center, radius))
        widget.apply_preset("adf")
        np.testing.assert_array_equal(_image(widget), _bf(expected, center, 2 * radius, inner=radius))
        widget.pos_row, widget.pos_col = 3, 4
        np.testing.assert_array_equal(_pattern(widget), expected[3, 4])
    finally:
        widget.close()


def test_dense_read_refuses_above_the_memory_ceiling(tmp_path, monkeypatch):
    master = _write_flagged_master(tmp_path, "big", _counts(8), FLAGGED)
    with pytest.raises(MemoryError, match="Nothing was binned or cropped"):
        reader.read_4dstem(master, "cpu", max_bytes=1000)
    monkeypatch.setattr(gpu_adapter, "accelerator_ready", lambda: False)
    monkeypatch.setattr(reader, "ceiling", lambda device=None: 1000)
    # the refusal offers [cuda] only where an NVIDIA GPU is present
    monkeypatch.setattr(device, "host_platform", lambda: ("linux", "x86_64"))
    monkeypatch.setattr(device, "nvidia_present", lambda: True)
    # The read and the argument checks run before the widget model exists, so a refusal sends nothing to the frontend.
    opened = []
    create_comm = comm.create_comm
    monkeypatch.setattr(comm, "create_comm", lambda **kwargs: opened.append(kwargs["target_name"]) or create_comm(**kwargs))
    with pytest.raises(MemoryError, match="pip install"):
        Show4DSTEM(master, device="cpu")
    with pytest.raises(ValueError, match="offline_dtype"):
        Show4DSTEM(_counts(1), offline_dtype="float32")
    assert opened == []
    # float32 on the CPU needs no copy, so the patched ceiling does not apply
    Show4DSTEM(_counts(1).astype(np.float32), device="cpu", precompute_virtual_images=False).close()
    assert "jupyter.widget" in opened  # the same probe sees a widget that does open


def test_refusal_states_the_size_the_read_needs(tmp_path):
    # A gold_512-sized scan on a 6.4 GB machine. The stored type depends on the largest count, so before
    # reading only the range is known: the file dtype and the detector bit depth bound it from above.
    cases = {
        (np.uint32, 32): "needs between 19.3 GB and 77.3 GB, above the 6.4 GB ceiling (80% of the available memory). It is "
                         "stored as the narrowest type that holds every count: 19.3 GB as uint16 or 77.3 GB as int64 "
                         "for this file (uint32, 32-bit counts); the largest count is known only after reading.",
        (np.uint32, 16): "needs 19.3 GB as uint16, above the 6.4 GB ceiling (80% of the available memory). Nothing",
        (np.uint8, None): "needs 9.7 GB as uint8, above the 6.4 GB ceiling (80% of the available memory). Nothing",
    }
    for (dtype, bits), expected in cases.items():
        path = tmp_path / f"{np.dtype(dtype).name}_{bits}.h5"
        with h5py.File(path, "w") as handle:  # chunks are allocated on write, so the file stays a few KB
            handle.create_dataset("entry/data/data", shape=(512, 512, 192, 192), dtype=dtype, chunks=(1, 1, 192, 192))
            if bits:
                handle["entry/instrument/detector/bit_depth_image"] = bits
        with pytest.raises(MemoryError) as refusal:
            reader.read_4dstem(path, "cpu", max_bytes=6_400_000_000)
        assert expected in str(refusal.value)


def test_from_folder_reads_masters_densely_without_quantem_gpu(tmp_path, monkeypatch, capsys):
    counts = [_counts(seed) for seed in range(2)]
    for idx, values in enumerate(counts):
        _write_master(tmp_path, f"scan_{idx:02d}", values)
    monkeypatch.setattr(gpu_adapter, "accelerator_ready", lambda: False)
    widget = Show4DSTEM.from_folder(tmp_path, device="cpu", verbose=False)
    try:
        assert "watching a folder for new masters needs quantem.gpu" in capsys.readouterr().out
        assert widget.n_frames == 2 and list(widget.frame_labels) == ["scan_00", "scan_01"]
        widget.frame_idx = 1
        widget.pos_row, widget.pos_col = 2, 1
        np.testing.assert_array_equal(_pattern(widget), counts[1][2, 1])
    finally:
        widget.close()


@pytest.mark.parametrize("first, second", [
    (_counts(5).astype(np.uint32) + 70000, _counts(6).astype(np.float32) + 0.25),  # int64 counts, then fractions
    (_counts(5), _counts(6).astype(np.float16) + 0.5),  # uint16 counts, then float16 of the same width
])
def test_from_folder_holds_a_float_master_after_integer_counts_exactly(tmp_path, monkeypatch, first, second):
    for idx, values in enumerate((first, second)):
        _write_master(tmp_path, f"scan_{idx:02d}", values)
    monkeypatch.setattr(gpu_adapter, "accelerator_ready", lambda: False)
    widget = Show4DSTEM.from_folder(tmp_path, device="cpu", watch=False, verbose=False)
    try:
        for idx, values in enumerate((first, second)):
            widget.frame_idx = idx
            np.testing.assert_array_equal(widget.pattern(2, 1), values[2, 1].astype(np.float32))
    finally:
        widget.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dense_and_encoded_paths_agree(tmp_path, accelerator, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    master = _write_flagged_master(tmp_path, "pair", _counts(9), FLAGGED)
    dense = Show4DSTEM(reader.read_4dstem(master, "cpu", verbose=False), device=device)
    encoded = Show4DSTEM(master)
    try:
        assert encoded._owned_acquisitions and isinstance(dense._data, torch.Tensor) and dense._device.type == device
        np.testing.assert_array_equal(dense._session(0).mean_dp(), encoded._session(0).mean_dp())
        for preset in ("bf", "abf", "adf", "haadf"):
            dense.apply_preset(preset)
            encoded.apply_preset(preset)
            np.testing.assert_array_equal(_image(dense), _image(encoded))
        for widget in (dense, encoded):
            widget.pos_row, widget.pos_col = 4, 1
            widget.vi_roi_mode = "circle"
        np.testing.assert_array_equal(_pattern(dense), _pattern(encoded))
        np.testing.assert_array_equal(dense.vi_roi_pattern(), encoded.vi_roi_pattern())
    finally:
        dense.close()
        encoded.close()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_interactive_export_packs_binned_counts_and_a_launcher(tmp_path):
    counts = _counts(4, scan=(4, 4))
    widget = Show4DSTEM(counts, title="export", sampling=(0.5, 0.5, 0.1, 0.1), center=(8, 8), bf_radius=2,
                        precompute_virtual_images=False)
    try:
        binned = export_data_array(widget, "uint8", 2, 2)
        expected = counts.reshape(2, 2, 2, 2, 8, 2, 8, 2).mean(axis=(1, 3, 5, 7))
        np.testing.assert_array_equal(binned, np.clip(np.round(expected), 0, 255).astype(np.uint8))
        out = widget.export_html(tmp_path / "viewer.html", dtype="uint16", det_bin=2, scan_bin=2)
        html = out.read_text(errors="ignore")
        assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
        assert "_offline_stack" in html and "application/vnd.jupyter.widget-state+json" in html
        assert (tmp_path / "Show4DSTEM.command").exists()
        assert "interactive raw 4D, uint16, scan bin 2x, detector bin 2x" in widget.export_status
        with pytest.raises(ValueError, match="det_bin"):
            widget.export_html(tmp_path / "bad.html", det_bin=3)
        widget.export_request = json.dumps({"export_kind": "report", "dtype": "uint8", "det_bin": 2, "scan_bin": 1,
                                            "dataset_scope": "all", "id": "r1", "filename": "report.html", "download": True})
        assert widget.export_payload_id == "r1"
        assert b"Static report export" in bytes(widget.export_payload)
        report = widget.export_html(tmp_path / "report.html", export_kind="report")
        assert "Static report export" in report.read_text()
    finally:
        widget.close()


def test_interactive_export_refuses_counts_no_page_can_load_before_reading(tmp_path, monkeypatch):
    # the shape alone decides: a gold-sized 512 x 512 x 192 x 192 scan is 9.7 GB of uint8 at bin 1
    with pytest.raises(ValueError, match=r"9\.7 GB of uint8 .* pass det_bin=2, scan_bin=2 .*Nothing was binned"):
        check_html_payload(1, (512, 512), (192, 192), "uint8", 1, 1)
    check_html_payload(1, (512, 512), (192, 192), "uint8", 4, 2)
    widget = Show4DSTEM(_counts(4, scan=(4, 4)), precompute_virtual_images=False)
    try:
        monkeypatch.setattr(html_export, "HTML_PAYLOAD_LIMIT", 4 * 4 * 16 * 16 - 1)
        monkeypatch.setattr(html_export, "HTML_PAYLOAD_ADVISED", 4 * 4 * 16 * 16 // 4)
        monkeypatch.setattr(Show4DSTEM, "_frame_array", lambda *args: pytest.fail("the export read a frame"))
        with pytest.raises(ValueError, match="pass det_bin=2, scan_bin=1"):
            widget.export_html(tmp_path / "viewer.html")
        assert not (tmp_path / "viewer.html").exists()
    finally:
        widget.close()


def test_saved_notebook_preview_enlarges_panels_with_sharp_pixels(monkeypatch):
    # The live pattern canvas is pixelated and the virtual image interpolates only with vi_smooth;
    # the static preview of a saved notebook shows the same pixels.
    counts = _counts(4, scan=(8, 8))
    widget = Show4DSTEM(counts, precompute_virtual_images=False)
    try:
        for overlay in ("_draw_virtual_overlays", "_draw_diffraction_overlays", "_draw_scalebar"):
            monkeypatch.setattr(preview, overlay, lambda *args: None)
        pattern, image = preview.diffraction_rgb(widget), preview.virtual_rgb(widget)
        np.testing.assert_array_equal(np.asarray(preview._panel(widget, "diffraction", 64)), pattern.repeat(4, 0).repeat(4, 1))
        np.testing.assert_array_equal(np.asarray(preview._panel(widget, "virtual", 64)), image.repeat(8, 0).repeat(8, 1))
        widget.vi_smooth = True
        assert not np.array_equal(np.asarray(preview._panel(widget, "virtual", 64)), image.repeat(8, 0).repeat(8, 1))
    finally:
        widget.close()


def test_saved_notebook_preview_draws_overlays_on_pixel_centers():
    """Profile points and the scan cursor center pixel i at i, as the live canvases draw them."""
    widget = Show4DSTEM(np.ones((8, 8, 16, 16), np.uint16), precompute_virtual_images=False, verbose=False)
    try:
        widget.profile_line = [{"row": 5, "col": 2}, {"row": 5, "col": 13}]
        widget.pos_row, widget.pos_col = 3, 6
        pattern, image = Image.new("RGB", (160, 160)), Image.new("RGB", (80, 80))
        preview._draw_diffraction_overlays(widget, pattern)
        preview._draw_virtual_overlays(widget, image)
        cyan = np.asarray(pattern, dtype=np.int16)
        rows = np.nonzero((cyan[..., 2] > 200) & (cyan[..., 1] > 150) & (cyan[..., 0] < 60))[0]
        assert rows.size and abs(np.median(rows) - 5.5 * 10) <= 0.5  # 10 preview px per detector pixel
        red = np.asarray(image, dtype=np.int16)
        cross_rows, cross_cols = np.nonzero((red[..., 0] > 200) & (red[..., 1] < 150))
        assert abs(np.median(cross_rows) - 3.5 * 10) <= 0.5 and abs(np.median(cross_cols) - 6.5 * 10) <= 0.5
    finally:
        widget.close()


def test_uint8_pack_announces_the_counts_it_clips(capsys):
    """uint8 is the compact default; counts above 255 change, so the pack says how many and how to keep them."""
    widget = Show4DSTEM(np.full((2, 2, 4, 4), 300, dtype=np.uint16), precompute_virtual_images=False, verbose=False)
    try:
        capsys.readouterr()
        assert export_data_array(widget, "uint8", 1, 1).max() == 255
        assert "uint8 pack clipped 64 values outside 0..255 (100.000%); pass dtype='uint16'" in capsys.readouterr().out
        assert export_data_array(widget, "uint16", 1, 1).max() == 300 and capsys.readouterr().out == ""
    finally:
        widget.close()


def test_webgpu_bundle_links_masters_and_writes_vendored_viewer(tmp_path):
    for stem in ("tilt_a", "tilt_b"):
        _write_master(tmp_path, stem, _counts(5, scan=(2, 2)))
    urls = bundle_master_urls(tmp_path)
    assert urls == ["../tilt_a_master.h5", "../tilt_b_master.h5"]
    widget = Show4DSTEM(np.zeros((1, 1, 1, 1), np.uint8), h5_urls=urls, scan_shape=(2, 2), detector_shape=DET,
                        backend="webgpu", view_mode="multiple", compare_group_mode="all")
    try:
        assert widget.n_frames == 2 and widget.offline
        assert json.loads(widget._h5_urls) == urls
        launcher = write_webgpu_bundle(widget, tmp_path, port=8899, h5_decode_dtype="uint8")
    finally:
        widget.close()
    assert launcher.name == "Show4DSTEM.command" and "8899" in launcher.read_text()
    page = (tmp_path / ".viewer" / "Show4DSTEM.html").read_text()
    assert '__QT_H5_DECODE_DTYPE ??= "uint8"' in page
    assert "__QT_REQUIRE_LOCAL_H5_FILES = true" in page
    assert "cdn.jsdelivr.net" not in page and (tmp_path / ".viewer" / "anywidget.min.js").exists()
    assert ".viewer/Show4DSTEM.html" in (tmp_path / "index.html").read_text()
