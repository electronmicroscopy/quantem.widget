"""Where Show4DSTEM computes: an acquisition reduces in its quantem.gpu session, an array in the widget's own.

The viewer wraps an acquisition from ``quantem.gpu.io.load`` (encoded counts or
float intensities), or a GPU tensor held as a dense acquisition, in an
``AcquisitionView`` and asks quantem.gpu for the mean pattern, every virtual
image and every scan-ROI reduction. The view is read only for the one pattern
under the cursor: reducing bounded reads instead took seconds per image. Host
arrays and tensors keep the widget's ``DenseSession``. Every number shown is
checked against NumPy on the same values.
"""

import numpy as np
import pytest
import torch

from quantem.widget import Show4DSTEM
from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.adapters.gpu import AcquisitionView
from quantem.widget.show4dstem.dense import DenseSession
from quantem.widget.show4dstem.detector import detector_mask, fit_bright_field_disk

SCAN, DET = (7, 9), (16, 16)


def _values(dtype) -> np.ndarray:
    """A bright disk whose intensity changes with scan position; the float one is fractional."""
    rows, cols = np.indices(DET)
    disk = ((rows - 7.5) ** 2 + (cols - 7.5) ** 2) <= 3.0**2
    scale = np.random.default_rng(17).integers(20, 60, SCAN, dtype=np.uint16)
    counts = (np.where(disk, 10, 1).astype(np.uint16)[None, None] * scale[..., None, None]).astype(np.uint16)
    return counts if dtype == np.uint16 else (counts / 8 + 0.125).astype(np.float32)


def _mean_dp(values: np.ndarray) -> np.ndarray:
    """The mean pattern rule: the float64 total divided once and rounded to float32."""
    return (values.reshape(-1, *DET).sum(axis=0, dtype=np.float64) / (values.shape[0] * values.shape[1])).astype(np.float32)


def _image(values: np.ndarray, center, inner: float, outer: float) -> np.ndarray:
    return values[..., detector_mask(center, inner, outer, DET)].sum(axis=-1, dtype=np.float64).astype(np.float32)


@pytest.fixture
def read_sizes(monkeypatch):
    """Scan positions covered by each bounded read of any ``AcquisitionView``."""
    sizes = []
    read = AcquisitionView.read

    def counted(view, *, scan_region):
        sizes.append((scan_region[1] - scan_region[0]) * (scan_region[3] - scan_region[2]))
        return read(view, scan_region=scan_region)

    monkeypatch.setattr(AcquisitionView, "read", counted)
    return sizes


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
@pytest.mark.parametrize("stored", ["encoded", "dense"])
def test_gpu_acquisition_reduces_in_quantem_gpu_without_reading_the_scan(tmp_path, read_sizes, dtype, stored):
    if not gpu_adapter.accelerator_ready():
        pytest.skip("GPU acquisitions need quantem.gpu (quantem.widget[cuda] or [mps]) and a CUDA or MPS GPU")
    values = _values(dtype)
    np.save(tmp_path / "scan.npy", values)
    loaded = gpu_adapter.load_acquisition(tmp_path / "scan.npy")
    if stored == "dense":
        # a tensor already on the GPU keeps the GPU dataset type (a strided preview of a loaded file)
        with loaded:
            values_t = loaded.read()
        loaded = gpu_adapter.acquisition_from_tensor(values_t, {"name": "preview"})
    widget = Show4DSTEM(loaded)
    try:
        assert isinstance(widget._data, AcquisitionView) and widget._device.type in ("cuda", "mps")
        assert not isinstance(widget._session(0), DenseSession)
        center, radius = fit_bright_field_disk(_mean_dp(values))
        assert (widget.bright_field_center_row_px, widget.bright_field_center_col_px, widget.bright_field_radius_px) == (*center, radius)
        np.testing.assert_allclose(widget.virtual_image(), _image(values, center, 0.0, radius), rtol=0 if dtype == np.uint16 else 1e-6)
        widget.apply_preset("adf")
        np.testing.assert_allclose(widget.virtual_image(), _image(values, center, radius, 2 * radius), rtol=0 if dtype == np.uint16 else 1e-6)
        widget.vi_roi_mode = "square"
        inside = (np.abs(np.arange(SCAN[0])[:, None] - widget.vi_roi_center_row) <= widget.vi_roi_radius) & (
            np.abs(np.arange(SCAN[1])[None, :] - widget.vi_roi_center_col) <= widget.vi_roi_radius)
        np.testing.assert_array_equal(widget.vi_roi_pattern(), _mean_dp(values[inside][:, None]))
        widget.pos_row, widget.pos_col = SCAN[0] - 1, SCAN[1] - 1
        np.testing.assert_array_equal(widget.pattern(SCAN[0] - 1, SCAN[1] - 1), values[-1, -1])
        assert read_sizes and set(read_sizes) == {1}
    finally:
        widget.close()
        loaded.close()


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_scan_region_of_an_acquisition_keeps_its_own_coordinates(tmp_path, read_sizes, dtype):
    if not gpu_adapter.accelerator_ready():
        pytest.skip("GPU acquisitions need quantem.gpu (quantem.widget[cuda] or [mps]) and a CUDA or MPS GPU")
    values = _values(dtype)
    np.save(tmp_path / "scan.npy", values)
    region = (1, 6, 2, 8)
    viewed = values[1:6, 2:8]
    with gpu_adapter.load_acquisition(tmp_path / "scan.npy") as loaded:
        widget = Show4DSTEM(loaded, scan_region=region)
        try:
            assert (widget.shape_rows, widget.shape_cols) == viewed.shape[:2]
            center, radius = fit_bright_field_disk(_mean_dp(viewed))
            assert (widget.bright_field_center_row_px, widget.bright_field_center_col_px, widget.bright_field_radius_px) == (*center, radius)
            np.testing.assert_allclose(widget.virtual_image(), _image(viewed, center, 0.0, radius), rtol=0 if dtype == np.uint16 else 1e-6)
            widget.pos_row, widget.pos_col = 0, 5
            np.testing.assert_array_equal(widget.pattern(0, 5), viewed[0, 5])
            assert set(read_sizes) == {1}
        finally:
            widget.close()


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
@pytest.mark.parametrize("kind", ["numpy", "torch cpu", "cpu acquisition"])
def test_host_data_stays_on_the_cpu(tmp_path, dtype, kind):
    """An array, a CPU tensor and a CPU load result all show the same numbers from the host."""
    values = _values(dtype)
    if kind == "cpu acquisition":
        if not gpu_adapter.available():
            pytest.skip("a CPU load result comes from quantem.gpu")
        np.save(tmp_path / "scan.npy", values)
        data = gpu_adapter.gpu_io.load(tmp_path / "scan.npy", backend="cpu", verbose=False)
    else:
        data = values if kind == "numpy" else torch.from_numpy(values)
    widget = Show4DSTEM(data, device="cpu", precompute_virtual_images=False)
    try:
        assert widget._device.type == "cpu"
        # only a load result is an acquisition; arrays and tensors reduce in the widget's own torch code
        assert isinstance(widget._session(0), DenseSession) == (kind != "cpu acquisition")
        center, radius = fit_bright_field_disk(_mean_dp(values))
        assert (widget.bright_field_center_row_px, widget.bright_field_center_col_px, widget.bright_field_radius_px) == (*center, radius)
        np.testing.assert_array_equal(widget.virtual_image(), _image(values, center, 0.0, radius))
        widget.pos_row, widget.pos_col = 6, 8
        np.testing.assert_array_equal(widget.pattern(6, 8), values[6, 8])
    finally:
        widget.close()


@pytest.mark.parametrize("reduce", ["sum", "max", "mean"])
def test_scan_roi_of_signed_integer_data_keeps_its_sign(reduce):
    """Dark-subtracted int16 patterns have negative totals: as uint64 the scan-ROI sum showed about 1.8e19."""
    values = (_values(np.uint16).astype(np.int32) - 300).astype(np.int16)
    widget = Show4DSTEM(values, device="cpu", precompute_virtual_images=False)
    try:
        widget.vi_roi_mode, widget.vi_roi_reduce = "square", reduce
        inside = (np.abs(np.arange(SCAN[0])[:, None] - widget.vi_roi_center_row) <= widget.vi_roi_radius) & (
            np.abs(np.arange(SCAN[1])[None, :] - widget.vi_roi_center_col) <= widget.vi_roi_radius)
        selected = values[inside].astype(np.int64)
        expected = {"sum": selected.sum(axis=0), "max": selected.max(axis=0), "mean": selected.sum(axis=0) / len(selected)}[reduce]
        assert expected.min() < 0
        np.testing.assert_array_equal(widget.vi_roi_pattern(), expected.astype(np.float32))
    finally:
        widget.close()
