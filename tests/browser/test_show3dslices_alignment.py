"""Show3DSlices slice alignment on the hardware adapter equals the CPU path and alignment.py.

The Align switch fits the drift in the browser: on a hardware WebGPU device the
whole estimate (blur, window, FFT, cross-power, peak, subpixel refinement)
stays in device buffers; without one the CPU path runs the same algorithm.
``js/show3dslices/sliceAlignment.ts`` is bundled with esbuild into a page that
runs both paths on the same stacks, and both must return the shifts the Python
estimator returns for those stacks.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from quantem.widget.show3dslices.alignment import estimate_global_slice_alignment

pytestmark = pytest.mark.browser

REPO = Path(__file__).resolve().parents[2]
ENTRY = """
import { estimateSliceAlignment } from "./show3dslices/sliceAlignment";
import { requireHardwareGPUDevice } from "./display/device";
window.estimate = async ({ depth, height, width, values }) => {
  const volume = Float32Array.from(values);
  const device = await requireHardwareGPUDevice("Slice alignment parity");
  const gpu = await estimateSliceAlignment(volume, width, height, depth, device);
  const cpu = await estimateSliceAlignment(volume, width, height, depth, null);
  return { gpu, cpu };
};
"""


@pytest.fixture(scope="module")
def alignment_page(browser_out) -> str:
    bundle = subprocess.run(
        ["npx", "esbuild", "--bundle", "--format=iife", "--loader=ts", f"--sourcefile={REPO / 'js' / 'entry.ts'}"],
        input=ENTRY, capture_output=True, text=True, cwd=REPO / "js", check=True,
    ).stdout
    (browser_out / "show3dslices_alignment.html").write_text(f'<!doctype html><meta charset="utf-8"><script>{bundle}</script>')
    return "show3dslices_alignment.html"


def drifting_stack(depth: int, height: int, width: int, row_drift: float, col_drift: float, seed: int) -> np.ndarray:
    """A textured plane on a background ramp, shifted by an exact sub-pixel amount per slice (FFT phase ramp)."""
    rng = np.random.default_rng(seed)
    rows, cols = np.indices((height, width)).astype(np.float64)
    plane = (np.sin(rows / 3.1) + 0.7 * np.cos(cols / 4.7) + 0.35 * np.sin((rows + cols) / 5.3)
             + 4.0 * np.exp(-((rows - height * 0.35) ** 2 + (cols - width * 0.42) ** 2) / 80.0)
             + 0.08 * rows + 0.05 * cols + 0.4 * rng.standard_normal((height, width)))
    spectrum = np.fft.fft2(plane)
    row_freq, col_freq = np.fft.fftfreq(height)[:, None], np.fft.fftfreq(width)[None, :]
    return np.stack([np.fft.ifft2(spectrum * np.exp(-2j * np.pi * (row_freq * z * row_drift + col_freq * z * col_drift))).real
                     for z in range(depth)]).astype(np.float32)


@pytest.mark.parametrize(("shape", "drift"), [((7, 64, 64), (0.75, -0.4)), ((6, 80, 96), (1.3, 0.55)), ((5, 128, 128), (-2.0, 1.15))],
                         ids=["64x64", "80x96", "128x128"])
def test_gpu_resident_alignment_equals_cpu_and_python(shape, drift, alignment_page, serve, open_page, hardware_adapter, browser_out):
    stack = drifting_stack(*shape, *drift, seed=sum(shape))
    expected = estimate_global_slice_alignment(stack)
    page = open_page(f"{serve(browser_out)}/{alignment_page}", webgpu=True)
    hardware_adapter(page.adapter())
    depth, height, width = shape
    result = page.evaluate("job => window.estimate(job)", {"depth": depth, "height": height, "width": width, "values": stack.ravel().tolist()})
    (browser_out / f"show3dslices_alignment_{height}x{width}.json").write_text(json.dumps({"python": expected, **result}, indent=1))
    assert result["gpu"]["backend"] == "webgpu" and result["cpu"]["backend"] == "cpu"
    for path in ("gpu", "cpu"):
        estimate = result[path]
        # shifts land on the 1/20 px refinement grid, so equal estimates agree exactly
        assert estimate["adjacentShiftPx"] == expected["adjacent_shift_apply_px"], path
        assert estimate["rowShiftPxPerSlice"] == pytest.approx(expected["row_shift_px_per_slice"], abs=1e-9)
        assert estimate["colShiftPxPerSlice"] == pytest.approx(expected["col_shift_px_per_slice"], abs=1e-9)
        # float32 transforms and reductions in the browser, float64 FFTs in numpy (measured: under 1e-6)
        np.testing.assert_allclose(estimate["quality"], expected["quality"], atol=2e-6)
    assert page.errors == [], page.errors
