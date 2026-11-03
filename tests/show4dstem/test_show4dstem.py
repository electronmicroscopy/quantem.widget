"""Show4DSTEM over dense arrays, files and encoded acquisitions: what the tutorials rely on.

Every reduction the viewer shows is checked against a NumPy sum over the same
detector mask on the same counts. Files come from a small synthetic Arina-style
master written with h5py: read densely by the widget (no quantem.gpu needed), or
loaded through ``quantem.gpu.io.load`` when quantem.gpu and a GPU are present,
where the two paths must agree exactly.
"""

import gzip
import json
import re
import time
from pathlib import Path

import comm
import h5py
import hdf5plugin
import numpy as np
import pytest
import torch
from PIL import Image

from quantem.widget import Show4DSTEM, ShowDiffraction, device
from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.adapters.core import make_dataset
from quantem.widget.show4dstem import preview, reader
from quantem.widget.show4dstem.detector import detector_mask, fit_bright_field_disk, roi_mask
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
    """Reference hot-pixel rule: the median of the unflagged 3x3 neighbours, half to even (np.round)."""
    corrected = values.astype(np.int64)
    rows, cols = values.shape[-2:]
    for row, col in flagged:
        neighbours = [(r, c) for r in range(row - 1, row + 2) for c in range(col - 1, col + 2)
                      if (r, c) != (row, col) and 0 <= r < rows and 0 <= c < cols and (r, c) not in flagged]
        stack = np.stack([values[..., r, c] for r, c in neighbours], axis=-1).astype(np.int64)
        corrected[..., row, col] = np.round(np.median(stack, axis=-1)).astype(np.int64)
    return corrected


def test_the_dense_fill_is_unbiased_and_matches_quantem_gpu():
    """A .5 median rounds half to even and a flagged cluster is filled from its ring, never 0, as quantem.gpu does."""
    from quantem.widget.show4dstem.reader import correct_flagged_pixels

    valid = np.ones((12, 12), bool)
    valid[0, 2:4] = False  # (0, 2) keeps four neighbours
    valid[5:8, 5:8] = False  # (6, 6) has no valid 3x3 neighbour
    block = np.random.default_rng(1).integers(0, 6, (40, 12, 12)).astype(np.uint16)
    block[:, ~valid] = 65535
    filled = correct_flagged_pixels(block, valid)
    ring = [(r, c) for r in range(4, 9) for c in range(4, 9) if not (5 <= r <= 7 and 5 <= c <= 7)]
    np.testing.assert_array_equal(filled[:, 6, 6], np.round(np.median(np.stack([block[:, r, c] for r, c in ring], 1), 1)))
    near = [(0, 1), (1, 1), (1, 2), (1, 3)]
    np.testing.assert_array_equal(filled[:, 0, 2], np.round(np.median(np.stack([block[:, r, c] for r, c in near], 1), 1)))
    hot_pixels = pytest.importorskip("quantem.gpu.resident.hot_pixels")
    expected = hot_pixels.fill_flagged(block.copy(), hot_pixels.median_plan(~valid), "median")
    np.testing.assert_array_equal(filled, expected)


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
            center, radius = fit_bright_field_disk(_mean_dp(counts))
            assert (widget.bright_field_center_row_px, widget.bright_field_center_col_px) == pytest.approx(center)
            assert widget.bright_field_radius_px == pytest.approx(radius)
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


def test_show4dstem_presets_use_the_recorded_bright_field_disk(tmp_path, accelerator):
    """data.calibration.fit_detector_sampling() is the one disk fit: the viewer reads it and its calibrated axes.

    The presets are multiples of the recorded edge radius around the recorded centre; a disk given by hand wins.
    """
    counts = _counts(3)
    master = _write_master(tmp_path, "gold", counts)
    with gpu_adapter.load_acquisition(master) as loaded:
        loaded.calibration.semiangle_mrad = 30
        fit = loaded.calibration.fit_detector_sampling()
        assert (fit.bright_field_center_px, fit.bright_field_edge_radius_px) == fit_bright_field_disk(_mean_dp(counts))
        # a disk recorded with the dataset is read, not fitted again
        loaded.metadata["bright_field_disk"] = {**loaded.metadata["bright_field_disk"],
                                                "bright_field_center_px": [7.0, 8.0], "bright_field_edge_radius_px": 3.25}
        widget = Show4DSTEM(loaded, precompute_virtual_images=False)
        given = Show4DSTEM(loaded, bright_field_center_px=(7.5, 7.5), bright_field_radius_px=2.0,
                           precompute_virtual_images=False)
        try:
            assert (widget.bright_field_center_row_px, widget.bright_field_center_col_px,
                    widget.bright_field_radius_px) == (7.0, 8.0, 3.25)
            assert widget.detector_sampling_mrad == pytest.approx(list(fit.detector_sampling_mrad))
            widget.apply_preset("adf")
            assert (widget.roi_radius_inner, widget.roi_radius) == (3.25, 6.5)
            np.testing.assert_array_equal(_image(widget), _bf(counts, (7.0, 8.0), 6.5, inner=3.25))
            assert (given.bright_field_center_row_px, given.bright_field_radius_px) == (7.5, 2.0)
        finally:
            widget.close()
            given.close()
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


@pytest.mark.parametrize("count", [7, 10])
def test_many_encoded_acquisitions_drag_reorder_hide_and_retain_exact_images(tmp_path, accelerator, count):
    counts = [_counts(seed) for seed in range(count)]
    paths = [_write_master(tmp_path, f"scan_{idx}", values) for idx, values in enumerate(counts)]
    loaded = gpu_adapter.load_acquisition(paths)
    viewer = Show4DSTEM(loaded, compare_group_mode="all", compare_max_panels=count)
    try:
        assert viewer._compare_session is not None
        previous = viewer.compare_virtual_image_bytes
        frozen = bytes(previous)
        for mode, center in [("circle", [5.2, 6.1]), ("annulus", [9.4, 10.2]), ("circle", [7.5, 7.5])]:
            with viewer.hold_trait_notifications():
                viewer.roi_mode = mode
                viewer.roi_center = center
                viewer.roi_radius_inner = 1.0
                viewer.roi_radius = 3.0
            mask = viewer._current_detector_mask()
            actual = np.frombuffer(viewer.compare_virtual_image_bytes, np.float32).reshape(count, *SCAN)
            expected = np.stack([values[..., mask].sum(-1).astype(np.float32) for values in counts])
            expected /= max(1, np.count_nonzero(mask))
            np.testing.assert_array_equal(actual, expected)
            assert bytes(previous) == frozen
        viewer.compare_panel_order = list(reversed(range(count)))
        viewer.compare_hidden_panels = [1, count - 1]
        indices = viewer.compare_panel_indices
        assert indices == [idx for idx in reversed(range(count)) if idx not in {1, count - 1}]
        actual = np.frombuffer(viewer.compare_virtual_image_bytes, np.float32).reshape(len(indices), *SCAN)
        np.testing.assert_array_equal(actual, expected[indices])
        assert viewer._compare_session_indices == tuple(indices)
        viewer.free()
        assert viewer._compare_session is None
        # Releasing the viewer's query workspace must leave borrowed data readable.
        session = gpu_adapter.prepare_session(loaded[0])
        try:
            np.testing.assert_array_equal(session.masked_sum(mask), counts[0][..., mask].sum(-1).astype(np.float32))
        finally:
            session.close()
    finally:
        viewer.close()
        for source in loaded:
            source.close()


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
    widget = Show4DSTEM(counts, bright_field_center_px=(7.5, 7.5), bright_field_radius_px=3.0, precompute_virtual_images=False)
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


def test_one_detector_message_reduces_the_image_once():
    # traitlets calls an observer once per changed trait, so a drag message (row, column and the pair) used to reduce three times
    counts = _counts(3)
    widget = Show4DSTEM(counts, bright_field_center_px=(7.5, 7.5), bright_field_radius_px=3.0, precompute_virtual_images=False)
    try:
        reduced = []
        reduce_image = widget._compute_virtual_image_from_roi

        def counted() -> None:
            reduced.append((widget.roi_mode, widget.roi_center_row, widget.roi_center_col, widget.roi_radius))
            reduce_image()

        widget._compute_virtual_image_from_roi = counted
        # set_state is what the comm handler calls with one frontend message
        widget.set_state({"roi_center_row": 6.0, "roi_center_col": 9.0, "roi_center": [6.0, 9.0]})  # a drag step
        assert reduced == [("circle", 6.0, 9.0, 3.0)]
        np.testing.assert_array_equal(_image(widget), _bf(counts, (6.0, 9.0), 3.0))
        reduced.clear()
        widget.set_state({"roi_center_row": 7.5, "roi_center_col": 7.5, "roi_center": [7.5, 7.5]})  # the DP Reset button
        assert reduced == [("circle", 7.5, 7.5, 3.0)]
        reduced.clear()
        widget.set_state({"roi_mode": "annular", "roi_radius_inner": 2.0, "roi_radius": 5.0})  # an annulus edit of both radii
        assert reduced == [("annular", 7.5, 7.5, 5.0)]
        np.testing.assert_array_equal(_image(widget), _bf(counts, (7.5, 7.5), 5.0, 2.0))
        reduced.clear()
        widget.roi_center = [8.0, 6.0]  # the pair alone moves both halves, never the row with the old column
        assert reduced == [("annular", 8.0, 6.0, 5.0)] and (widget.roi_center_row, widget.roi_center_col) == (8.0, 6.0)
        reduced.clear()
        widget.roi_center_row = 5.0  # one trait from Python is still one reduction
        assert reduced == [("annular", 5.0, 6.0, 5.0)]
        np.testing.assert_array_equal(_image(widget), _bf(counts, (5.0, 6.0), 5.0, 2.0))
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


def test_a_removed_keyword_raises_and_a_trait_keyword_still_sets():
    # traitlets only warns about a keyword it does not know, so a removed one did nothing
    with pytest.raises(TypeError, match="unexpected keyword argument 'compare_panel_gap_px'"):
        Show4DSTEM(_counts(1), compare_panel_gap_px=0)
    widget = Show4DSTEM(_counts(1), precompute_virtual_images=False, dp_scale_mode="linear")
    try:
        assert widget.dp_scale_mode == "linear"
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
    widget = Show4DSTEM(counts, bright_field_center_px=(7.5, 7.5), bright_field_radius_px=3.0)
    try:
        np.testing.assert_array_equal(_image(widget), _bf(counts, (7.5, 7.5), 3.0))
        widget.roi_radius = 2.2
        np.testing.assert_array_equal(_image(widget), _bf(counts, (7.5, 7.5), 2.2))
        assert not np.array_equal(_bf(counts, (7.5, 7.5), 2.2), _bf(counts, (7.5, 7.5), 3.0))
    finally:
        widget.close()


class _RecordingComm:
    """A comm that keeps every message the widget sends to its frontend."""

    comm_id = "recording-comm"
    kernel = object()

    def __init__(self):
        self.sent = []

    def on_msg(self, callback):
        pass

    def send(self, data=None, metadata=None, buffers=None):
        self.sent.append(data)

    def close(self, *args, **kwargs):
        pass


def test_a_moved_cursor_reads_its_pattern_once():
    # A frontend move writes the row and the column in one message: one pattern read, at the new position.
    counts = _counts(3)
    widget = Show4DSTEM(counts, precompute_virtual_images=False)
    try:
        reads = []
        session = widget._session(0)
        frame = session.frame
        session.frame = lambda index: (reads.append(index), frame(index))[1]
        widget.set_state({"pos_row": 1, "pos_col": 4})
        assert reads == [1 * SCAN[1] + 4]
        np.testing.assert_array_equal(_pattern(widget), counts[1, 4].astype(np.float32))
        widget.pos_col = 2  # one trait moved from Python still sends its pattern
        assert reads == [1 * SCAN[1] + 4, 1 * SCAN[1] + 2]
        np.testing.assert_array_equal(_pattern(widget), counts[1, 2].astype(np.float32))
    finally:
        widget.close()


def test_a_preset_click_sends_one_update_and_reuses_its_image():
    counts = _counts(4)
    widget = Show4DSTEM(counts, bright_field_center_px=(7.5, 7.5), bright_field_radius_px=3.0, precompute_virtual_images=False)
    try:
        sums = []
        session = widget._session(0)
        masked_sum = session.masked_sum
        session.masked_sum = lambda mask: (sums.append(1), masked_sum(mask))[1]
        recorder = _RecordingComm()
        widget.comm = recorder
        widget.set_state({"_preset_request": "adf"})
        updates = [message for message in recorder.sent if message["method"] == "update"]
        assert len(updates) == 1  # detector traits, image and the cleared request in one message
        assert {"roi_mode", "roi_radius", "roi_radius_inner", "_preset_request"} <= set(updates[0]["state"])
        assert ["virtual_image_bytes"] in updates[0]["buffer_paths"]
        adf = _image(widget).copy()
        np.testing.assert_array_equal(adf, masked_sum(widget._current_detector_mask()).astype(np.float32))
        widget.apply_preset("bf")  # computed when the widget opened
        computed = len(sums)
        widget.apply_preset("adf")
        assert len(sums) == computed  # the ADF image computed on demand is reused
        np.testing.assert_array_equal(_image(widget), adf)
        widget.bright_field_radius_px = 2.5  # a new calibration drops every cached preset
        widget.apply_preset("adf")
        assert len(sums) == computed + 1
        np.testing.assert_array_equal(_image(widget), masked_sum(widget._current_detector_mask()).astype(np.float32))
    finally:
        widget.comm = None
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


def test_float_data_is_packed_as_rounded_counts_and_says_so(capsys):
    # the uint8/uint16 packs cast float data to integers, truncating: 2.6 became 2 and data in [0, 1) became 0, silently
    values = np.full((4, 4, 8, 8), 2.6, np.float32)
    values[..., :4] = 0.4
    widget = Show4DSTEM(values, offline=True, precompute_virtual_images=False, verbose=False)
    try:
        inline = np.frombuffer(gzip.decompress(widget._offline_stack), np.uint8).reshape(values.shape)
        assert set(np.unique(inline)) == {0, 3}
        assert "uint8 pack rounded 1024 non-integer values to whole counts (100.000%, largest change 0.4)" in capsys.readouterr().out
        np.testing.assert_array_equal(export_data_array(widget, "uint16", 1, 1), inline)
        assert "uint16 pack rounded 1024 non-integer values" in capsys.readouterr().out
    finally:
        widget.close()
    counts = Show4DSTEM(_counts(1), offline=True, precompute_virtual_images=False, verbose=False)
    try:
        export_data_array(counts, "uint16", 1, 1)
        assert "rounded" not in capsys.readouterr().out  # whole counts are packed as they are, without a line
    finally:
        counts.close()


def test_presets_stop_at_the_detector_edge_and_refuse_a_ring_beyond_it(capsys):
    # on a 192 px detector with a 49 px disk, HAADF began at 98.7 px, past the edge (a near-empty
    # image), and the ADF ring ran past the edge, keeping only its corners
    rows, cols = np.indices((32, 32))
    pattern = np.where(np.hypot(rows - 15.5, cols - 15.5) <= 9.0, 50, 2).astype(np.uint16)
    widget = Show4DSTEM(np.broadcast_to(pattern, (4, 4, 32, 32)).copy(), precompute_virtual_images=True, verbose=False)
    try:
        radius = widget.bright_field_radius_px
        assert 8.5 < radius < 9.5 and (widget.bright_field_center_row_px, widget.bright_field_center_col_px) == (15.5, 15.5)
        capsys.readouterr()
        widget.apply_preset("adf")
        assert (widget.roi_radius_inner, widget.roi_radius) == (pytest.approx(radius), 16.0)  # the edge, not 2x the radius
        assert f"ADF outer radius is 16.0 px, the detector edge, not {2 * radius:.1f} px" in capsys.readouterr().out
        widget.apply_preset("haadf")  # 2x the radius is past the edge: refused, the ROI stays on ADF
        assert (widget.roi_mode, widget.roi_radius_inner, widget.roi_radius) == ("annular", pytest.approx(radius), 16.0)
        printed = capsys.readouterr().out
        assert f"HAADF starts {2 * radius:.1f} px from the bright-field centre, but this detector reaches 16.0 px" in printed
        assert "haadf" not in widget._preset_images
        widget.apply_preset("bf")
        assert (widget.roi_radius_inner, widget.roi_radius) == (0.0, pytest.approx(radius)) and capsys.readouterr().out == ""
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


def test_annular_after_bf_is_the_bright_disk_not_a_zero_area_ring():
    """ADF then BF then Annular: the BF disk must not keep ADF's inner radius."""
    counts = _counts(4)
    widget = Show4DSTEM(counts, precompute_virtual_images=False)
    try:
        widget.apply_preset("adf").apply_preset("bf")
        assert widget.roi_radius_inner == 0.0
        widget.roi_mode = "annular"
        bright = widget.virtual_image()
        assert bright.min() > 0
        widget.roi_mode = "circle"
        np.testing.assert_array_equal(widget.virtual_image(), bright)
    finally:
        widget.close()


def test_refused_argument_raises_without_a_cleanup_traceback():
    """ipywidgets closes the half-built widget in __del__; that close must not fail on missing state."""
    import gc
    import sys

    unraisable = []
    previous = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        with pytest.raises(TypeError):
            Show4DSTEM(_counts(5), detector_radius=3)
        gc.collect()
    finally:
        sys.unraisablehook = previous
    assert [str(hook.exc_value) for hook in unraisable] == []


def test_widget_calls_only_functions_its_export_module_defines():
    """compute_ssb replaces _ssb_phase in the tests below, so a stale export name there never ran in CI."""
    import inspect
    import re

    from quantem.widget.show4dstem import export as html_export
    from quantem.widget.show4dstem import widget as widget_module

    called = set(re.findall(r"\bhtml_export\.(\w+)", inspect.getsource(widget_module)))
    assert called and called <= set(dir(html_export)), sorted(called - set(dir(html_export)))


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
        np.testing.assert_array_equal(_image(widget), _bf(counts, (widget.bright_field_center_row_px, widget.bright_field_center_col_px), widget.bright_field_radius_px))
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
        center, radius = fit_bright_field_disk(_mean_dp(expected))
        assert (widget.bright_field_center_row_px, widget.bright_field_center_col_px, widget.bright_field_radius_px) == pytest.approx((*center, radius))
        np.testing.assert_array_equal(_image(widget), _bf(expected, center, radius))
        widget.apply_preset("adf")
        np.testing.assert_array_equal(_image(widget), _bf(expected, center, 2 * radius, inner=radius))
        widget.pos_row, widget.pos_col = 3, 4
        np.testing.assert_array_equal(_pattern(widget), expected[3, 4])
    finally:
        widget.close()


def test_widget_loads_say_how_many_flagged_pixels_they_replace_and_can_keep_them(tmp_path, monkeypatch, capsys):
    # every widget load forced the median replacement with no way to keep the stored counts, and said nothing
    counts = _counts(7, scan=(6, 6))
    master = _write_flagged_master(tmp_path, "flagged", counts, FLAGGED)
    monkeypatch.setattr(device, "_printed", set())
    corrected = reader.read_4dstem(master, "cpu", verbose=False)
    np.testing.assert_array_equal(np.asarray(corrected.array), _median_corrected(counts, FLAGGED))
    assert ('replaced 4 flagged detector pixels with the median of their unflagged 3x3 neighbours '
            '(hot_pixel_correction="median"); pass hot_pixel_correction="none"') in capsys.readouterr().out
    zeroed = reader.read_4dstem(master, "cpu", verbose=False, hot_pixel_correction="zero")
    assert all(int(np.asarray(zeroed.array)[..., row, col].max()) == 0 for row, col in FLAGGED)
    assert zeroed.metadata["hot_pixel_correction"]["method"] == "zero"
    assert 'replaced 4 flagged detector pixels with 0 (hot_pixel_correction="zero")' in capsys.readouterr().out
    raw = reader.read_dense(master, None, verbose=False, hot_pixel_correction="none")
    assert all(int(raw[..., row, col].min()) == 2**32 - 1 for row, col in FLAGGED)
    assert '4 detector pixels are flagged in the file and left as stored (hot_pixel_correction="none")' in capsys.readouterr().out
    if gpu_adapter.accelerator_ready():
        # an acquisition the caller loaded raw shows its flagged pixels as 0, which one line says
        from quantem.gpu import io as gpu_io

        with gpu_io.load(str(master), hot_pixel_correction="none", verbose=False) as acquisition:
            Show4DSTEM(acquisition, precompute_virtual_images=False).close()
        assert "4 detector pixels the acquisition flags are shown as 0" in capsys.readouterr().out
    monkeypatch.setattr(gpu_adapter, "accelerator_ready", lambda: False)
    widget = Show4DSTEM(master, device="cpu", hot_pixel_correction="none", precompute_virtual_images=False)
    try:
        widget.pos_row, widget.pos_col = 3, 4
        assert _pattern(widget)[4, 5] == np.float32(2**32 - 1)
    finally:
        widget.close()
    with pytest.raises(ValueError, match="hot_pixel_correction applies when Show4DSTEM reads a file"):
        Show4DSTEM(counts, hot_pixel_correction="none")
    with pytest.raises(ValueError, match="hot_pixel_correction must be one of"):
        reader.read_dense(master, None, hot_pixel_correction="mean")


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


def test_interactive_export_packs_binned_counts_in_one_self_contained_page(tmp_path):
    counts = _counts(4, scan=(4, 4))
    widget = Show4DSTEM(counts, title="export", scan_sampling_A=(0.4, 0.6), detector_sampling_mrad=(0.1, 0.2),
                        bright_field_center_px=(8, 8), bright_field_radius_px=2, precompute_virtual_images=False)
    try:
        # the page's clone keeps each axis its own step, grown by the binning along it (the row step was
        # replaced by the column step before)
        clone = html_export.export_clone(widget, "uint8", 2, 2)
        assert (clone.scan_sampling_A, clone.detector_sampling_mrad) == ([0.8, 1.2], [0.2, 0.4])
        clone.close()
        binned = export_data_array(widget, "uint8", 2, 2)
        expected = counts.reshape(2, 2, 2, 2, 8, 2, 8, 2).mean(axis=(1, 3, 5, 7))
        np.testing.assert_array_equal(binned, np.clip(np.round(expected), 0, 255).astype(np.uint8))
        out = widget.export_html(tmp_path / "viewer.html", dtype="uint16", detector_bin=2, scan_bin=2)
        html = out.read_text(errors="ignore")
        assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
        assert "_offline_stack" in html and "application/vnd.jupyter.widget-state+json" in html
        # rc41: a packed page holds its counts, yet a launcher was written and the status said it read a data file over HTTP
        assert not (tmp_path / "Show4DSTEM.command").exists()
        assert "interactive raw 4D, uint16, scan bin 2x, detector bin 2x" in widget.export_status
        assert widget.export_status.endswith("self-contained HTML - double-click to open, no server needed")
        with pytest.raises(ValueError, match="detector_bin"):
            widget.export_html(tmp_path / "bad.html", detector_bin=3)
        widget.export_request = json.dumps({"export_kind": "report", "dtype": "uint8", "detector_bin": 2, "scan_bin": 1,
                                            "dataset_scope": "all", "id": "r1", "filename": "report.html", "download": True})
        assert widget.export_payload_id == "r1"
        assert b"Static report export" in bytes(widget.export_payload)
        report = widget.export_html(tmp_path / "report.html", export_kind="report")
        assert "Static report export" in report.read_text()
    finally:
        widget.close()


def test_interactive_export_refuses_counts_no_page_can_load_before_reading(tmp_path, monkeypatch):
    # the shape alone decides: a gold-sized 512 x 512 x 192 x 192 scan is 9.7 GB of uint8 at bin 1
    with pytest.raises(ValueError, match=r"9\.7 GB of uint8 .* pass detector_bin=2, scan_bin=2 .*Nothing was binned"):
        check_html_payload(1, (512, 512), (192, 192), "uint8", 1, 1)
    check_html_payload(1, (512, 512), (192, 192), "uint8", 4, 2)
    widget = Show4DSTEM(_counts(4, scan=(4, 4)), precompute_virtual_images=False)
    try:
        monkeypatch.setattr(html_export, "HTML_PAYLOAD_LIMIT", 4 * 4 * 16 * 16 - 1)
        monkeypatch.setattr(html_export, "HTML_PAYLOAD_ADVISED", 4 * 4 * 16 * 16 // 4)
        monkeypatch.setattr(Show4DSTEM, "_frame_array", lambda *args: pytest.fail("the export read a frame"))
        with pytest.raises(ValueError, match="pass detector_bin=2, scan_bin=1"):
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


def test_webgpu_bundle_links_masters_and_writes_an_offline_viewer(tmp_path):
    for stem in ("tilt_a", "tilt_b"):
        _write_master(tmp_path, stem, _counts(5, scan=(2, 2)))
    urls = bundle_master_urls(tmp_path)
    assert urls == ["../tilt_a_master.h5", "../tilt_b_master.h5"]
    widget = Show4DSTEM(None, h5_urls=urls, scan_shape=(2, 2), detector_shape=DET,
                        view_mode="multiple", compare_group_mode="all")
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
    # the widget runtime travels inside the page: no CDN and no script file to fetch from disk
    assert "<script src=" not in page and 'id="quantem-widget-runtime-2"' in page
    assert ".viewer/Show4DSTEM.html" in (tmp_path / "index.html").read_text()


def test_qem_bundle_links_acquisitions_for_a_streaming_viewer(tmp_path):
    """A .qem folder viewer: relative links beside the page, the files named in its state, no HDF5 tuning."""
    reference = pytest.importorskip("quantem.gpu.formats.qem.reference")
    from quantem.widget.show4dstem.export import write_qem_bundle

    data = tmp_path / "data"
    data.mkdir()
    for name, seed in (("tilt_a.qem", 1), ("tilt_b.qem", 2)):
        reference.save_array(data / name, _counts(seed, scan=(3, 4)),
                             {"voltage_kV": 300.0, "scan_sampling_A": [0.4, 0.6], "detector_sampling_mrad": [1.5, 2.0]})
    launcher = write_qem_bundle([data / "tilt_a.qem", data / "tilt_b.qem"], tmp_path / "viewer",
                                frame_labels=["-17 mrad", "+17 mrad"], port=8898)
    viewer = tmp_path / "viewer"
    assert launcher == viewer / "Show4DSTEM.command" and "8898" in launcher.read_text()
    for name in ("tilt_a.qem", "tilt_b.qem"):
        link = viewer / "qem" / name
        assert link.is_symlink() and not Path(link.readlink()).is_absolute()
        assert link.read_bytes() == (data / name).read_bytes()
    page = (viewer / ".viewer" / "Show4DSTEM.html").read_text()
    assert '\\"../qem/tilt_a.qem\\", \\"../qem/tilt_b.qem\\"' in page
    assert "+17 mrad" in page and "__QT_H5_DECODE_DTYPE ??=" not in page
    assert '\"compare_cols\": 2' in page
    # the page draws the calibration the headers record (it showed 1.0 px scale bars before)
    assert re.search(r'"scan_sampling_A": \[\s*0\.4,\s*0\.6\s*\]', page)
    assert re.search(r'"detector_sampling_mrad": \[\s*1\.5,\s*2(\.0)?\s*\]', page)
    reference.save_array(data / "wide.qem", _counts(3, scan=(3, 5)))
    with pytest.raises(ValueError, match="one viewer compares acquisitions of one shape"):
        write_qem_bundle([data / "tilt_a.qem", data / "wide.qem"], tmp_path / "other")


def test_compute_ssb_accepts_a_1024_scan(monkeypatch):
    # B-1: the scan guard allowed 128, 256 and 512, although quantem.gpu reconstructs 1024 x 1024 scans.
    pytest.importorskip("quantem.gpu", reason="the SSB phase also takes quantem.gpu's centre of mass")
    import types

    from quantem.widget.show4dstem import widget as widget_module

    class Session:
        num_bright_field_pixels = 4
        bright_field_intensity_threshold = 0.0
        bright_field_radius_px = 1.5
        scan_detector_rotation_flipped = False

        def __init__(self, source, **kwargs):
            self.detector_sampling_mrad = kwargs.get("detector_sampling_mrad")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def reconstruct(self):
            return types.SimpleNamespace(phase=np.ones((1024, 1024), np.float32), aberrations={"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0},
                                         scan_detector_rotation_deg=0.0, loss=None, elapsed_s=None, sample_tilt_mrad=None)

    monkeypatch.setattr(widget_module.gpu_adapter, "SSB", Session)  # stands in for quantem.gpu's class behind the adapter
    widget = Show4DSTEM(np.ones((1024, 1024, 4, 4), np.uint16), precompute_virtual_images=False, voltage_kV=300,
                        semiangle_mrad=20, scan_sampling_A=0.5, detector_sampling_mrad=1.0)
    try:
        phase, dpc_row, _ = widget._ssb_phase(aberrations={"C10_nm": 0.0}, lock_aberrations=True)
        assert phase.shape == dpc_row.shape == (1024, 1024)
    finally:
        widget.close()


# ---------------------------------------------------------------------------
# Calibration: the dataset or file is the source
# ---------------------------------------------------------------------------


def test_a_dataset_is_the_source_of_the_calibration():
    counts = _counts(1, scan=(4, 4))
    dataset = make_dataset(counts, sampling=(0.04, 0.06, 0.02, 0.03), units=("nm", "nm", "1/angstrom", "1/angstrom"))
    dataset.metadata.update(voltage_kV=300.0, semiangle_mrad=20.0)
    widget = Show4DSTEM(dataset, precompute_virtual_images=False)
    # the .qem unit strings and quantem core's are read through one table, under the format's flat names
    assert widget.scan_sampling_A == pytest.approx([0.4, 0.6])
    assert (widget.detector_sampling_inv_A, widget.detector_sampling_mrad) == ([0.02, 0.03], [])
    assert (widget.voltage_kV, widget.semiangle_mrad) == (300.0, 20.0)
    widget.close()
    # a keyword may repeat what the dataset records, never contradict it
    Show4DSTEM(dataset, voltage_kV=300, precompute_virtual_images=False).close()
    with pytest.raises(ValueError, match="voltage_kV=200 contradicts the value the input records"):
        Show4DSTEM(dataset, voltage_kV=200)
    with pytest.raises(ValueError, match="scan_sampling_A"):
        Show4DSTEM(dataset, scan_sampling_A=(0.5, 0.5))
    # an uncalibrated dataset is absent, not 1.0 "pixels"; an array takes its calibration from the keywords
    plain = Show4DSTEM(make_dataset(counts), precompute_virtual_images=False)
    stated = Show4DSTEM(counts, scan_sampling_A=0.5, detector_sampling_mrad=(1.0, 2.0), precompute_virtual_images=False)
    assert (plain.scan_sampling_A, plain.detector_sampling_mrad, plain.detector_sampling_inv_A) == ([], [], [])
    assert (stated.scan_sampling_A, stated.detector_sampling_mrad) == ([0.5, 0.5], [1.0, 2.0])
    plain.close()
    stated.close()
    with pytest.raises(ValueError, match="Unknown unit 'furlong'"):
        Show4DSTEM(make_dataset(counts, sampling=(1, 1, 1, 1), units=("furlong",) * 4))
    # the removed spellings are refused by name; backend="webgpu" was checked and then never used
    for removed in ("ssb_voltage_kV", "ssb_semiangle_mrad", "ssb_scan_sampling_A", "ssb_det_sampling_mrad", "sampling", "units",
                    "backend"):
        with pytest.raises(TypeError, match=removed):
            Show4DSTEM(counts, **{removed: 1.0})


def test_compute_ssb_reads_the_optics_the_dataset_records(monkeypatch, tmp_path):
    # Compute SSB raised "needs ssb_voltage_kV" for a dataset that records the voltage
    pytest.importorskip("quantem.gpu", reason="the SSB phase also takes quantem.gpu's centre of mass")
    import types

    from quantem.widget.show4dstem import widget as widget_module

    seen = {}

    class Session:
        num_bright_field_pixels = 4
        bright_field_intensity_threshold = 0.0
        bright_field_radius_px = 1.5
        scan_detector_rotation_flipped = True  # the record states the session's flag, whatever it is

        def __init__(self, source, **kwargs):
            seen.update(kwargs)
            self.detector_sampling_mrad = kwargs.get("detector_sampling_mrad")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def reconstruct(self):
            # a fit on the reversed branch: quantem.gpu reports the physical angle, here above 180
            return types.SimpleNamespace(phase=np.ones((128, 128), np.float32), aberrations={"C10_nm": 12.0, "C12_nm": 3.0, "phi12_deg": 30.0},
                                         scan_detector_rotation_deg=345.5, loss=0.25, elapsed_s=1.5, sample_tilt_mrad=None)

    monkeypatch.setattr(widget_module.gpu_adapter, "SSB", Session)  # stands in for quantem.gpu's class behind the adapter
    dataset = make_dataset(np.ones((128, 128, 4, 4), np.uint16), sampling=(0.4, 0.6, 1.5, 1.5), units=("angstrom", "angstrom", "mrad", "mrad"))
    dataset.metadata.update(voltage_kV=300.0, semiangle_mrad=20.0)
    widget = Show4DSTEM(dataset, precompute_virtual_images=False)
    widget._ssb_phase(aberrations={"C10_nm": 0.0}, lock_aberrations=True)
    assert (seen["voltage_kV"], seen["semiangle_mrad"], seen["scan_sampling_A"]) == (300.0, 20.0, (0.4, 0.6))
    # a missing coefficient is 0, as compute_ssb documents: quantem.gpu refuses a mapping without all three
    assert seen["aberrations"] == {"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0}
    # the trait and the calibration file hold the physical angle, so the reversal survives: a manual
    # recompute sent the angle below 180 back without the flag, 180 degrees off
    assert widget.ssb_compute_scan_detector_rotation_deg == 345.5
    saved = json.loads(widget.ssb_compute_calibration_json)
    # its own schema: the record differs from ShowPtycho's calibration.json (source, no session_identity)
    assert saved["schema"] == "quantem.show4dstem-ssb-calibration/3"
    assert (saved["scan_detector_rotation_deg"], saved["scan_detector_rotation_flipped"]) == (345.5, True)
    assert saved["aberrations"] == {"C10_nm": 12.0, "C12_nm": 3.0, "phi12_deg": 30.0}
    assert (saved["scan_sampling_A"], saved["detector_sampling_mrad"], saved["voltage_kV"]) == ([0.4, 0.6], [1.5, 1.5], 300.0)
    assert (saved["num_bright_field_pixels"], saved["loss"], saved["elapsed_s"]) == (4, 0.25, 1.5)
    # a manual recompute runs no search; the record names the refinement as quantem.gpu does, never a bool "refine"
    assert saved["refinement"] is None and "refine" not in saved
    assert not {"rotation_angle_deg", "com_reversed", "aberration_units", "calibration", "run"} & set(saved)
    # ShowPtycho does not restore this record as its own calibration.json
    from quantem.widget.showptycho.calibration import read_calibration
    from quantem.widget.showptycho.widget import CALIBRATION_SCHEMA

    downloaded = tmp_path / "calibration.json"
    downloaded.write_text(widget.ssb_compute_calibration_json)
    with pytest.raises(ValueError, match="quantem.show4dstem-ssb-calibration/3' is not ShowPtycho's"):
        read_calibration(downloaded, {}, (0, 128, 0, 128), None, CALIBRATION_SCHEMA)
    widget._ssb_phase(aberrations=saved["aberrations"], scan_detector_rotation_deg=saved["scan_detector_rotation_deg"], lock_aberrations=True)
    assert seen["scan_detector_rotation_deg"] == 345.5 and seen["aberrations"] == {"C10_nm": 12.0, "C12_nm": 3.0, "phi12_deg": 30.0}
    widget.close()
    bare = Show4DSTEM(np.ones((128, 128, 4, 4), np.uint16), precompute_virtual_images=False)
    with pytest.raises(ValueError, match="needs the scan step in angstrom and the data record none"):
        bare._ssb_phase()
    bare.close()


def _scan_gradient(scan: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """``(row, col)`` gradient of ``sin(a row) cos(b col)`` in the scan frame, at most 1.5 detector pixels.

    Unequal row and column periods keep the flip identifiable: with ``a == b``
    the swapped gradient is curl-free too.
    """
    rows, cols = np.mgrid[0:scan, 0:scan].astype(np.float64)
    a, b = 2 * np.pi / 32, 2 * np.pi / 20
    scale = 1.5 / max(a, b)
    return scale * a * np.cos(a * rows) * np.cos(b * cols), -scale * b * np.sin(a * rows) * np.sin(b * cols)


def _rotated_counts(rotation_deg: float, *, flipped: bool = False, scan: int = 128, det: int = 24) -> np.ndarray:
    """uint16 counts of a soft bright-field disk shifted by ``_scan_gradient`` seen through a rotated detector.

    The detector shift is the scan gradient rotated by ``-rotation_deg``, so
    rotating the centre of mass by ``rotation_deg`` makes it curl-free again;
    ``flipped`` swaps the detector axes on top.
    """
    gradient_row, gradient_col = _scan_gradient(scan)
    angle = np.radians(rotation_deg)
    shift_row = np.cos(angle) * gradient_row + np.sin(angle) * gradient_col
    shift_col = -np.sin(angle) * gradient_row + np.cos(angle) * gradient_col
    rows, cols = np.indices((det, det)).astype(np.float32)
    centre = (det - 1) / 2
    distance = np.hypot(rows - (centre + shift_row[..., None, None]), cols - (centre + shift_col[..., None, None]))
    values = np.round(40 * np.clip(6.5 - distance, 0, 1)).astype(np.uint16) + 1
    return values.swapaxes(-1, -2).copy() if flipped else values


def _require_ssb() -> None:
    """Skip without quantem.gpu or without CUDA or MPS."""
    pytest.importorskip("quantem.gpu", reason="Compute SSB runs quantem.gpu")
    try:
        pytest.importorskip("quantem.gpu.device").detect()
    except RuntimeError as exc:
        pytest.skip(f"SSB needs CUDA or MPS: {exc}")


def _ssb_widget(counts: np.ndarray) -> Show4DSTEM:
    """A calibrated Show4DSTEM over ``counts``, or a skip without CUDA or MPS."""
    _require_ssb()
    return Show4DSTEM(counts, precompute_virtual_images=False, voltage_kV=300, semiangle_mrad=20,
                      scan_sampling_A=0.5, detector_sampling_mrad=20 / 6)


def test_compute_ssb_uses_the_estimated_rotation_rather_than_zero():
    # Compute SSB passed scan_detector_rotation_deg=0.0, so the fit could only pick 0 or 180 degrees
    widget = _ssb_widget(_rotated_counts(30.0))
    try:
        widget.compute_ssb(num_trials=4, refinement=None)
        rotation = widget.ssb_compute_scan_detector_rotation_deg
        assert min(abs(rotation - 30.0), abs(rotation - 210.0)) < 1.0
        assert json.loads(widget.ssb_compute_calibration_json)["scan_detector_rotation_flipped"] is False
    finally:
        widget.close()


@pytest.mark.parametrize("flipped", [False, True])
def test_compute_ssb_dpc_maps_follow_the_scan_axes_for_a_flipped_detector(flipped):
    # the DPC alignment copied quantem.gpu's rotation without its axis swap, so a flipped
    # detector gave DPC_row correlated -0.87 with the scan-row gradient and 0.50 with the column one
    _require_ssb()
    from quantem.gpu import dpc

    counts = _rotated_counts(30.0, flipped=flipped)
    widget = _ssb_widget(counts)
    try:
        widget.compute_ssb(num_trials=4, refinement=None)
        record = json.loads(widget.ssb_compute_calibration_json)
        assert record["scan_detector_rotation_flipped"] is flipped
        maps = np.frombuffer(widget.vi_product_maps_bytes, np.float32).reshape(-1, 128, 128)
        dpc_row, dpc_col = (maps[widget.vi_product_labels.index(label)] for label in ("DPC_row", "DPC_col"))
        expected = dpc.run(counts, scan_detector_rotation_deg=record["scan_detector_rotation_deg"],
                           scan_detector_rotation_flipped=flipped)
        np.testing.assert_array_equal(dpc_row, expected.com_row_aligned)
        np.testing.assert_array_equal(dpc_col, expected.com_col_aligned)
        gradient_row, gradient_col = _scan_gradient()
        sign = 1.0 if abs(record["scan_detector_rotation_deg"] - 30.0) < 90 else -1.0
        assert sign * np.corrcoef(dpc_row.ravel(), gradient_row.ravel())[0, 1] > 0.99
        assert sign * np.corrcoef(dpc_col.ravel(), gradient_col.ravel())[0, 1] > 0.99
    finally:
        widget.close()


def test_compute_ssb_selects_bright_field_pixels_with_the_quantem_gpu_default_threshold():
    # Show4DSTEM passed threshold 0.5 where quantem.gpu defaults to 0.0: with a given 7 px radius on this
    # soft-edged disk it kept 120 of the 156 bright-field pixels a plain SSB session uses
    _require_ssb()
    from quantem.gpu import SSB

    rows, cols = np.indices((24, 24))
    pattern = np.round(40 * np.clip((8 - np.hypot(rows - 11.5, cols - 11.5)) / 4, 0, 1)).astype(np.uint16) + 1
    counts = np.broadcast_to(pattern, (128, 128, 24, 24)).copy()
    widget = _ssb_widget(counts)
    try:
        widget.compute_ssb(aberrations={"C10_nm": 0.0}, lock_aberrations=True, scan_detector_rotation_deg=0.0,
                           bright_field_radius_px=7.0)
        with SSB(counts, voltage_kV=300, semiangle_mrad=20, scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(20 / 6, 20 / 6),
                 scan_detector_rotation_deg=0.0, bright_field_radius_px=7.0) as plain:
            assert widget.ssb_compute_num_bright_field_pixels == plain.num_bright_field_pixels == 156
        record = json.loads(widget.ssb_compute_calibration_json)
        assert (record["bright_field_intensity_threshold"], record["num_bright_field_pixels"]) == (0.0, 156)
    finally:
        widget.close()


def test_compute_ssb_derives_a_missing_semiangle_with_the_quantem_gpu_disk_edge_rule(capsys):
    # the fallback multiplied the widget's equal-area disk radius by the column detector step only:
    # 20.62 mrad for this 20 mrad aperture imaged at (2.0, 2.2) mrad per pixel, now 20.22
    _require_ssb()
    from quantem.gpu import detector

    sampling = (2.0, 2.2)
    rows, cols = np.indices((32, 32))
    distance = np.hypot((rows - 15.5) * sampling[0], (cols - 15.5) * sampling[1])  # mrad from the beam centre
    pattern = np.round(40 * np.clip((20.0 - distance) / 2.0 + 0.5, 0, 1)).astype(np.uint16) + 1
    counts = np.broadcast_to(pattern, (128, 128, 32, 32)).copy()
    widget = Show4DSTEM(counts, precompute_virtual_images=False, voltage_kV=300, scan_sampling_A=0.5,
                        detector_sampling_mrad=sampling)
    try:
        assert widget.semiangle_mrad is None
        capsys.readouterr()
        widget.compute_ssb(aberrations={"C10_nm": 0.0}, lock_aberrations=True, scan_detector_rotation_deg=0.0)
        semiangle = json.loads(widget.ssb_compute_calibration_json)["semiangle_mrad"]
        edge = detector.disk_edge_radius(np.asarray(pattern, np.float64))
        assert semiangle == pytest.approx(edge * np.sqrt(sampling[0] * sampling[1]))
        assert semiangle == pytest.approx(20.0, rel=0.02)
        # the viewer's disk is the same edge (it was the equal-area circle: 20.62 mrad along the columns)
        assert widget.bright_field_radius_px == pytest.approx(edge)
        assert "semiangle_mrad is not recorded" in capsys.readouterr().out
    finally:
        widget.close()


def test_compute_ssb_converts_a_detector_step_in_reciprocal_angstrom(monkeypatch):
    # a recorded detector_sampling_inv_A was dropped: SSB then estimated the step from the disk edge instead
    pytest.importorskip("quantem.gpu", reason="the conversion uses quantem.gpu's wavelength")
    import types

    from quantem.gpu.optics import wavelength_A_from_kV
    from quantem.widget.show4dstem import widget as widget_module

    seen = {}

    class Session:
        num_bright_field_pixels = 4
        bright_field_intensity_threshold = 0.0
        bright_field_radius_px = 1.5
        scan_detector_rotation_flipped = False

        def __init__(self, source, **kwargs):
            seen.update(kwargs)
            self.detector_sampling_mrad = kwargs.get("detector_sampling_mrad")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def reconstruct(self):
            return types.SimpleNamespace(phase=np.ones((128, 128), np.float32), aberrations={"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0},
                                         scan_detector_rotation_deg=0.0, loss=None, elapsed_s=None, sample_tilt_mrad=None)

    monkeypatch.setattr(widget_module.gpu_adapter, "SSB", Session)
    dataset = make_dataset(np.ones((128, 128, 4, 4), np.uint16), sampling=(0.5, 0.5, 0.02, 0.025),
                           units=("angstrom", "angstrom", "1/angstrom", "1/angstrom"))
    dataset.metadata.update(voltage_kV=300.0, semiangle_mrad=20.0)
    widget = Show4DSTEM(dataset, precompute_virtual_images=False)
    try:
        assert (widget.detector_sampling_mrad, widget.detector_sampling_inv_A) == ([], [0.02, 0.025])
        widget._ssb_phase(aberrations={"C10_nm": 0.0}, lock_aberrations=True, scan_detector_rotation_deg=0.0)
        expected = (0.02 * wavelength_A_from_kV(300.0) * 1e3, 0.025 * wavelength_A_from_kV(300.0) * 1e3)
        assert seen["detector_sampling_mrad"] == pytest.approx(expected)
        assert expected[0] == pytest.approx(0.3937, abs=1e-4)
        assert json.loads(widget.ssb_compute_calibration_json)["detector_sampling_mrad"] == pytest.approx(list(expected))
    finally:
        widget.close()


def test_compute_ssb_reconstructs_a_rectangular_scan_that_quantem_gpu_pads():
    # Compute SSB refused every scan but square 128, 256, 512 and 1024, although quantem.gpu pads other shapes
    _require_ssb()
    from quantem.gpu import SSB

    counts = _rotated_counts(30.0)[:96, :80].copy()
    widget = _ssb_widget(counts)
    optics = {"C10_nm": 2.0, "C12_nm": 0.5, "phi12_deg": 10.0}
    try:
        phase = widget.compute_ssb(aberrations=optics, lock_aberrations=True, scan_detector_rotation_deg=30.0)
        assert phase.shape == (96, 80)
        with SSB(counts, voltage_kV=300, semiangle_mrad=20, scan_sampling_A=0.5, detector_sampling_mrad=20 / 6,
                 aberrations=optics, scan_detector_rotation_deg=30.0) as ssb:
            full = np.asarray(torch.as_tensor(ssb.reconstruct().phase).cpu())
            top, left = ((padded - measured) // 2 for padded, measured in zip(ssb.scan_shape, ssb.input_scan_shape))
        np.testing.assert_array_equal(phase, full[top:top + 96, left:left + 80])
    finally:
        widget.close()


def test_compute_ssb_uses_the_given_bright_field_disk_and_records_the_one_used():
    # bright_field_center_px= and bright_field_radius_px= never reached SSB, and the record wrote
    # None for the radius and the detector step whenever quantem.gpu found them itself
    _require_ssb()
    from quantem.gpu import SSB, detector

    counts = _rotated_counts(30.0)
    optics = {"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0}
    _require_ssb()
    found = Show4DSTEM(counts, precompute_virtual_images=False, voltage_kV=300, semiangle_mrad=20, scan_sampling_A=0.5)
    given = Show4DSTEM(counts, precompute_virtual_images=False, voltage_kV=300, semiangle_mrad=20, scan_sampling_A=0.5,
                       bright_field_center_px=(11.0, 12.0), bright_field_radius_px=4.0)
    try:
        found.compute_ssb(aberrations=optics, lock_aberrations=True, scan_detector_rotation_deg=30.0)
        record = json.loads(found.ssb_compute_calibration_json)
        with SSB(counts, voltage_kV=300, semiangle_mrad=20, scan_sampling_A=0.5, scan_detector_rotation_deg=30.0) as plain:
            plain.num_bright_field_pixels
            expected_radius = plain.browser_state().brightfield.bright_field_radius_px
        assert record["bright_field_radius_px"] == pytest.approx(expected_radius)
        mean = np.asarray(counts, np.float64).mean(axis=(0, 1))
        assert record["detector_sampling_mrad"] == pytest.approx(list(detector.detector_sampling_mrad(mean, 20.0)))
        given.compute_ssb(aberrations=optics, lock_aberrations=True, scan_detector_rotation_deg=30.0)
        with SSB(counts, voltage_kV=300, semiangle_mrad=20, scan_sampling_A=0.5, scan_detector_rotation_deg=30.0,
                 bright_field_center_px=(11.0, 12.0), bright_field_radius_px=4.0) as plain:
            assert given.ssb_compute_num_bright_field_pixels == plain.num_bright_field_pixels
        assert given.ssb_compute_num_bright_field_pixels != found.ssb_compute_num_bright_field_pixels
        assert json.loads(given.ssb_compute_calibration_json)["bright_field_radius_px"] == 4.0
    finally:
        found.close()
        given.close()


def test_compute_ssb_names_its_refinement_as_quantem_gpu_does():
    # the trait ssb_compute_refine and the record key "refine" were booleans; quantem.gpu calls it refinement,
    # and "auto" (its default) is recorded as the optimizer it selected
    widget = _ssb_widget(_rotated_counts(30.0))
    try:
        assert (widget.ssb_compute_refinement, widget.ssb_compute_tilt) == ("auto", True)
        widget.compute_ssb(num_trials=2)
        record = json.loads(widget.ssb_compute_calibration_json)
        assert record["refinement"] == "newton" and record["tilt"] is True  # the tilt-corrected fit refines with Newton
        assert len(record["sample_tilt_mrad"]) == 2 and record["depth_spread_nm"] >= 0.0
        widget.ssb_compute_refinement = None
        widget.compute_ssb(num_trials=2, tilt=False)
        record = json.loads(widget.ssb_compute_calibration_json)
        assert record["refinement"] is None and record["tilt"] is False and record["sample_tilt_mrad"] is None
        assert record["objective"] == "agreement"
        widget.ssb_compute_refinement = "auto"
        widget.compute_ssb(num_trials=2, tilt=False, objective="phase_variance")
        record = json.loads(widget.ssb_compute_calibration_json)
        assert (record["objective"], record["refinement"], record["tilt"]) == ("phase_variance", "nelder-mead", False)
        with pytest.raises(ValueError, match="refinement must be 'auto', 'newton', 'nelder-mead' or None"):
            widget.compute_ssb(num_trials=2, refinement="lbfgs")
        assert not hasattr(widget, "ssb_compute_refine")
    finally:
        widget.close()


def test_compute_ssb_fits_tilt_corrected_by_default_and_tilt_false_on_request(monkeypatch):
    """Compute SSB is quantem.gpu's default fit (model with sample tilt and depth spread); tilt=False fits at tilt 0.

    It ran find_aberrations(tilt=False, refinement="nelder-mead") whatever the user asked, the one SSB entry point
    that did not default to the tilt-corrected fit.
    """
    pytest.importorskip("quantem.gpu", reason="the SSB phase also takes quantem.gpu's centre of mass")
    import types

    from quantem.widget.show4dstem import widget as widget_module

    calls = []

    class Session:
        num_bright_field_pixels = 4
        bright_field_intensity_threshold = 0.0
        bright_field_radius_px = 1.5
        scan_detector_rotation_flipped = False

        def __init__(self, source, **kwargs):
            self.detector_sampling_mrad = kwargs.get("detector_sampling_mrad")
            self.aberrations = kwargs.get("aberrations")
            self.sample_tilt_mrad = self.depth_spread_nm = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def _result(self, tilt, refinement):
            return types.SimpleNamespace(
                phase=np.ones((128, 128), np.float32), aberrations={"C10_nm": 8.0, "C12_nm": 1.0, "phi12_deg": 20.0},
                scan_detector_rotation_deg=30.0, loss=0.5, elapsed_s=1.0, refinement=refinement,
                sample_tilt_mrad=(1.5, -2.0) if tilt else None, depth_spread_nm=6.0 if tilt else None,
                tilt_fit_gain=0.1 if tilt else None)

        def find_aberrations(self, **options):
            calls.append(("fit", options))
            result = self._result(options["tilt"], "newton" if options["refinement"] == "auto" else options["refinement"])
            result.objective = options["objective"]
            return result

        def reconstruct(self):
            calls.append(("reconstruct", {"aberrations": self.aberrations, "sample_tilt_mrad": self.sample_tilt_mrad,
                                          "depth_spread_nm": self.depth_spread_nm}))
            return self._result(self.sample_tilt_mrad is not None, None)

    monkeypatch.setattr(widget_module.gpu_adapter, "SSB", Session)  # stands in for quantem.gpu's class behind the adapter
    widget = Show4DSTEM(np.ones((128, 128, 4, 4), np.uint16), precompute_virtual_images=False, voltage_kV=300,
                        semiangle_mrad=20, scan_sampling_A=0.5, detector_sampling_mrad=1.0)
    try:
        widget.compute_ssb()
        kind, options = calls[-1]
        assert kind == "fit" and options["tilt"] is True and options["refinement"] == "auto" and options["num_trials"] == 200
        assert options["objective"] == "agreement"
        assert not {"search_ranges", "refine_lock", "fixed_aberrations"} & set(options)
        record = json.loads(widget.ssb_compute_calibration_json)
        assert record["schema"] == "quantem.show4dstem-ssb-calibration/3" and record["objective"] == "agreement"
        assert (record["tilt"], record["sample_tilt_mrad"], record["depth_spread_nm"], record["tilt_fit_gain"]) == (True, [1.5, -2.0], 6.0, 0.1)
        assert record["refinement"] == "newton"  # what quantem.gpu selected, never "auto"
        assert "tilt-corrected: (+1.5, -2.0) mrad" in widget.ssb_compute_status
        # a manual recompute from the sliders keeps the fitted sample model
        widget.compute_ssb(aberrations={"C10_nm": 9.0, "C12_nm": 1.0, "phi12_deg": 20.0}, lock_aberrations=True)
        kind, state = calls[-1]
        assert kind == "reconstruct" and state["sample_tilt_mrad"] == (1.5, -2.0) and state["depth_spread_nm"] == 6.0
        assert json.loads(widget.ssb_compute_calibration_json)["tilt"] is True
        # in the joint fit a lock is quantem.gpu's fixed_aberrations: C10 alone, C12 = 0 alone, or all three;
        # the held values are the sliders' (a fit moves the sliders to its result)
        def sliders(C10_nm, C12_nm, phi12_deg):
            widget.ssb_compute_C10_nm, widget.ssb_compute_C12_nm, widget.ssb_compute_phi12_deg = C10_nm, C12_nm, phi12_deg

        sliders(5.0, 0.0, 0.0)
        widget.compute_ssb(lock_C10=True)
        assert calls[-1][1]["fixed_aberrations"] == {"C10_nm": 5.0} and "check_rotation" not in calls[-1][1]
        sliders(5.0, 0.0, 0.0)
        widget.compute_ssb(lock_C12=True)
        assert calls[-1][1]["fixed_aberrations"] == {"C12_nm": 0.0}
        sliders(5.0, 2.0, 15.0)
        widget.compute_ssb(lock_C10=True, lock_C12=True)
        assert calls[-1][1]["fixed_aberrations"] == {"C10_nm": 5.0, "C12_nm": 2.0, "phi12_deg": 15.0}
        assert calls[-1][1]["check_rotation"] is False
        fits = len(calls)
        sliders(5.0, 2.0, 15.0)
        with pytest.raises(ValueError, match="holds C12 alone only at 0"):
            widget.compute_ssb(lock_C12=True)
        assert len(calls) == fits and widget.ssb_compute_status.startswith("SSB failed")
        # tilt=False: the model at tilt 0, with the standard fit's search ranges and refinement locks
        sliders(5.0, 2.0, 15.0)
        widget.compute_ssb(tilt=False, lock_C10=True)
        kind, options = calls[-1]
        assert options["tilt"] is False and options["refine_lock"] == ["C10_nm"]
        assert options["search_ranges"] == {"C10_nm": 5.0, "C12_nm": (0.0, 10.0), "phi12_deg": (-90.0, 90.0)}
        record = json.loads(widget.ssb_compute_calibration_json)
        assert (record["tilt"], record["sample_tilt_mrad"], record["depth_spread_nm"]) == (False, None, None)
        # after a tilt-0 fit a manual recompute has no sample model to keep
        widget.compute_ssb(aberrations={"C10_nm": 5.0}, lock_aberrations=True)
        assert calls[-1][1]["sample_tilt_mrad"] is None
        assert json.loads(widget.ssb_compute_calibration_json)["objective"] is None
        # the third method, quantem.gpu's names: tilt=False, objective="phase_variance"; it cannot fit a tilt
        widget.compute_ssb(tilt=False, objective="phase_variance")
        assert (calls[-1][1]["tilt"], calls[-1][1]["objective"]) == (False, "phase_variance")
        assert json.loads(widget.ssb_compute_calibration_json)["objective"] == "phase_variance"
        fits = len(calls)
        with pytest.raises(ValueError, match="earlier SSB search, a backup at tilt 0"):
            widget.compute_ssb(objective="phase_variance")
        with pytest.raises(ValueError, match="objective must be 'agreement' or 'phase_variance'"):
            widget.compute_ssb(tilt=False, objective="variance")
        assert len(calls) == fits
        # the frontend sends the switch with its request
        widget.ssb_compute_request = json.dumps({"action": "compute_ssb", "num_trials": 20, "tilt": False,
                                                 "objective": "phase_variance", "refinement": None})
        deadline = time.time() + 10
        while widget.ssb_compute_busy or not calls[-1][1].get("num_trials") == 20:
            assert time.time() < deadline
            time.sleep(0.01)
        assert (widget.ssb_compute_tilt, widget.ssb_compute_objective) == (False, "phase_variance")
        assert calls[-1][1]["tilt"] is False and calls[-1][1]["objective"] == "phase_variance"
        assert calls[-1][1]["refinement"] is None
    finally:
        widget.close()


def test_compute_ssb_refuses_an_unknown_aberration_key():
    # an aberration under another name ({"C10": 5.0}) was dropped without a message and the phase reconstructed at C10 0
    pytest.importorskip("quantem.gpu", reason="the session that refuses the key is quantem.gpu's")
    try:
        pytest.importorskip("quantem.gpu.device").detect()
    except RuntimeError as exc:
        pytest.skip(f"SSB needs CUDA or MPS: {exc}")
    widget = Show4DSTEM(np.ones((128, 128, 4, 4), np.uint16), precompute_virtual_images=False, voltage_kV=300,
                        semiangle_mrad=20, scan_sampling_A=0.5, detector_sampling_mrad=1.0)
    with pytest.raises(ValueError, match=r"unknown \['C10'\]"):
        widget.compute_ssb(aberrations={"C10": 5.0}, lock_aberrations=True)
    widget.close()


def test_the_bright_field_radius_has_one_name_in_keyword_trait_and_state():
    # keyword, trait and state key were bf_radius; the name every package uses is bright_field_radius_px (detector pixels)
    counts = _counts(3, scan=(4, 4))
    viewer = Show4DSTEM(counts, bright_field_center_px=(7.5, 7.5), bright_field_radius_px=3.0, precompute_virtual_images=False)
    pattern = ShowDiffraction(counts[0, 0].astype(np.float32), bright_field_center_px=(7.5, 7.5), bright_field_radius_px=3.0, verbose=False)
    for widget in (viewer, pattern):
        assert widget.bright_field_radius_px == 3.0 and widget.state_dict()["bright_field_radius_px"] == 3.0
        assert "bright_field_radius_px" in widget.get_state() and not widget.has_trait("bf_radius")
        with pytest.raises(ValueError, match=r"does not save: \['bf_radius'\]"):
            widget.load_state_dict({"bf_radius": 2.0})
    viewer.close()
    for Widget, data in ((Show4DSTEM, counts), (ShowDiffraction, counts[0, 0].astype(np.float32))):
        with pytest.raises(TypeError, match="unexpected keyword argument 'bf_radius'"):
            Widget(data, bf_radius=3.0)


def test_the_bright_field_centre_has_one_name_in_keyword_traits_and_state(tmp_path):
    # keyword center=(row, col) with traits and state keys center_row, center_col; the vocabulary name is bright_field_center_px
    counts = _counts(3, scan=(4, 4))
    viewer = Show4DSTEM(counts, bright_field_center_px=(6.5, 8.5), bright_field_radius_px=3.0, precompute_virtual_images=False)
    pattern = ShowDiffraction(counts[0, 0].astype(np.float32), bright_field_center_px=(6.5, 8.5), bright_field_radius_px=3.0, verbose=False)
    for widget in (viewer, pattern):
        assert (widget.bright_field_center_row_px, widget.bright_field_center_col_px) == (6.5, 8.5)
        state = widget.state_dict()
        assert (state["bright_field_center_row_px"], state["bright_field_center_col_px"]) == (6.5, 8.5)
        assert {"bright_field_center_row_px", "bright_field_center_col_px"} <= set(widget.get_state())
        assert not widget.has_trait("center_row") and not widget.has_trait("center_col")
        with pytest.raises(ValueError, match=r"does not save: \['center_row', 'center_col'\]"):
            widget.load_state_dict({"center_row": 2.0, "center_col": 2.0})
    viewer.close()
    for Widget, data in ((Show4DSTEM, counts), (ShowDiffraction, counts[0, 0].astype(np.float32))):
        with pytest.raises(TypeError, match="unexpected keyword argument 'center'"):
            Widget(data, center=(6.5, 8.5))
    # the measurements file names the two coordinates the same way
    pattern.save(tmp_path / "state.json")
    metadata = json.loads(ShowDiffraction.measurements_from_state(tmp_path / "state.json", tmp_path / "measurements.json").read_text())["metadata"]
    assert (metadata["bright_field_center_row_px"], metadata["bright_field_center_col_px"]) == (6.5, 8.5) and "center_row" not in metadata


def test_static_preview_labels_the_detector_bar_in_its_own_unit(monkeypatch):
    # the bar read "mrad" for every calibrated detector, also one calibrated in 1/angstrom
    dataset = make_dataset(_counts(2, scan=(4, 4)), sampling=(0.4, 0.6, 0.02, 0.03), units=("angstrom", "angstrom", "1/angstrom", "1/angstrom"))
    widget = Show4DSTEM(dataset, precompute_virtual_images=False)
    drawn = []
    monkeypatch.setattr(preview, "_draw_scalebar", lambda image, step, unit: drawn.append((image.width * step, unit)))
    preview.static_png_b64(widget)
    widget.close()
    assert drawn == [(pytest.approx(4 * 0.6), "angstrom"), (pytest.approx(DET[1] * 0.03), "1/angstrom")]


def test_a_dense_read_records_the_file_and_the_flagged_pixels(tmp_path):
    # the CPU route replaced flagged pixels and left no record of it
    master = _write_flagged_master(tmp_path, "flagged", _counts(5), FLAGGED)
    dataset = reader.read_4dstem(master, device="cpu", verbose=False)
    record = dataset.metadata["hot_pixel_correction"]
    assert dataset.metadata["source_path"] == str(master)
    assert (record["applied"], record["method"], record["pixel_count"]) == (True, "median", len(FLAGGED))
    assert [tuple(point) for point in record["coordinates_row_col"]] == sorted(FLAGGED)
    # no calibration in the file: every axis is absent, as on the GPU route
    widget = Show4DSTEM(dataset, precompute_virtual_images=False)
    assert (widget.scan_sampling_A, widget.detector_sampling_mrad, widget.detector_sampling_inv_A) == ([], [], [])
    widget.close()
