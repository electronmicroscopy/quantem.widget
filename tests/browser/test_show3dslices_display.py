"""Show3DSlices in an exported page on the hardware adapter.

The overlays drawn over the slice panels (oblique cut and crosshair, scale bar,
colorbar) keep a backing store at the screen's device pixel ratio, so they stay
sharp on HiDPI screens. Align fits the drift of the volume on WebGPU and
reports the slopes alignment.py computes for the same volume. The 3D view
says that it samples an 8-bit copy of the volume.
"""

import re

import numpy as np
import pytest

from quantem.widget.show3dslices.alignment import estimate_global_slice_alignment

pytestmark = pytest.mark.browser

SLICE = 'canvas[data-quantem-scientific-output="show3dslices-slice-{axis}"]'
ROW_DRIFT, COL_DRIFT = 0.8, -0.45


@pytest.fixture(scope="module")
def drifting_volume() -> np.ndarray:
    """A textured plane on a ramp, shifted by an exact sub-pixel amount per slice (FFT phase ramp)."""
    depth, size = 12, 96
    rows, cols = np.indices((size, size)).astype(np.float64)
    plane = (np.sin(rows / 3.1) + 0.7 * np.cos(cols / 4.7) + 4.0 * np.exp(-((rows - 34) ** 2 + (cols - 40) ** 2) / 80.0)
             + 0.05 * rows + 0.3 * np.random.default_rng(0).standard_normal((size, size)))
    spectrum = np.fft.fft2(plane)
    freq = np.fft.fftfreq(size)
    return np.stack([np.fft.ifft2(spectrum * np.exp(-2j * np.pi * (freq[:, None] * z * ROW_DRIFT + freq[None, :] * z * COL_DRIFT))).real
                     for z in range(depth)]).astype(np.float32)


@pytest.fixture(scope="module")
def exported(drifting_volume, browser_out) -> str:
    from quantem.widget import Show3DSlices

    Show3DSlices(drifting_volume, title="drift", sampling=0.5, units="A").export_html(browser_out / "show3dslices_drift.html")
    return "show3dslices_drift.html"


def test_overlays_keep_a_device_pixel_backing_store(exported, chrome, serve, browser_out):
    context = chrome.new_context(viewport={"width": 1600, "height": 1050}, device_scale_factor=2)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)[:300]))
    page.goto(f"{serve(browser_out)}/{exported}")
    page.wait_for_selector(SLICE.format(axis=0), timeout=60_000)
    page.wait_for_timeout(4_000)
    page.screenshot(path=str(browser_out / "show3dslices_dpr2.png"))
    sizes = page.evaluate("""() => [0, 1].flatMap(axis => {
      const image = document.querySelector(`canvas[data-quantem-scientific-output="show3dslices-slice-${axis}"]`);
      return [...image.parentElement.querySelectorAll("canvas")].slice(1).map(overlay => {
        const box = overlay.getBoundingClientRect();
        return { axis, cssWidth: box.width, cssHeight: box.height, width: overlay.width, height: overlay.height };
      });
    })""")
    # two overlays per panel: the oblique cut and crosshair, then the scale bar and colorbar
    assert len(sizes) == 4
    for size in sizes:
        assert size["width"] == round(2 * size["cssWidth"]) and size["height"] == round(2 * size["cssHeight"]), size
    assert errors == [], errors
    context.close()


def test_align_fits_the_drift_alignment_py_fits(exported, drifting_volume, serve, open_page, hardware_adapter, browser_out):
    expected = estimate_global_slice_alignment(drifting_volume)
    page = open_page(f"{serve(browser_out)}/{exported}", webgpu=True)
    page.wait_for_selector(SLICE.format(axis=0), timeout=60_000)
    hardware_adapter(page.adapter())
    page.get_by_role("button", name="Advanced").click()
    page.get_by_label("Align slices with automatic global slice alignment").click()
    status = page.get_by_text(re.compile(r"^Aligned row"))
    status.wait_for(timeout=60_000)
    text = status.inner_text()
    page.screenshot(path=str(browser_out / "show3dslices_aligned.png"))
    assert text.endswith("(WebGPU)"), text
    # the 3D view's 8-bit copy of the volume is stated where the view is
    assert page.get_by_text("3D view samples an 8-bit copy; the slice panels show the float32 values.").is_visible()
    row, col = (float(value) for value in re.findall(r"[-+]\d+\.\d{3}", text))
    assert row == pytest.approx(expected["row_shift_px_per_slice"], abs=5e-4)
    assert col == pytest.approx(expected["col_shift_px_per_slice"], abs=5e-4)
    assert page.errors == [], page.errors
