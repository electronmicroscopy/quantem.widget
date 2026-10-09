"""Show2D's live contrast drag shows the pixels its release paints, with WebGPU on and off.

While a histogram handle is held, Show2D repaints only the dragged panels'
canvases (and without WebGPU colors only the source pixels the canvas shows);
the release repaints the offscreens at full resolution. A 1024 x 1024 image is
dragged at the fitted view and again zoomed in off centre: each time the canvas
held at the last drag position must equal, pixel for pixel, the canvas after
the release. A gallery built with ``denoise=`` shows the filtered panel
smoother than its raw twin from the first paint. A click on the FFT of nm
data reads its d-spacing and |g| in nm.
"""

import re

import numpy as np
import pytest

pytestmark = pytest.mark.browser

IMAGE = 'canvas[data-quantem-scientific-output="show2d-image-0"]'
READ_PANEL = """
(panel) => {
  const canvas = document.querySelector(`canvas[data-quantem-scientific-output="show2d-image-${panel}"]`);
  return Array.from(canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data);
}
"""
READ_IMAGE = f"() => ({READ_PANEL})(0)"
MAX_HANDLE = """
() => {
  const root = Array.from(document.querySelectorAll('.MuiSlider-root'))
    .find(r => r.querySelector('input[aria-label="Histogram intensity clip range"]'));
  root.scrollIntoView({block: 'center'});
  const box = root.querySelectorAll('.MuiSlider-thumb')[1].getBoundingClientRect();
  return [box.left + box.width / 2, box.top + box.height / 2];
}
"""


@pytest.fixture(scope="module")
def large_image(browser_out) -> str:
    from quantem.widget import Show2D

    rows, cols = np.indices((1024, 1024))
    noise = np.random.default_rng(0).standard_normal((1024, 1024))
    image = (np.sin(cols / 23.0) * np.cos(rows / 31.0) + 0.3 * noise).astype(np.float32)
    Show2D(image, cmap="viridis").export_html(browser_out / "show2d_large.html")
    return "show2d_large.html"


def held_and_released(page, step_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Drag the max handle left and hold it; the canvas while held, then after the release."""
    x, y = page.evaluate(MAX_HANDLE)
    page.mouse.move(x, y)
    page.mouse.down()
    for step in range(1, 7):
        page.mouse.move(x - step_px * step, y)
        page.wait_for_timeout(40)
    page.wait_for_timeout(600)
    held = np.asarray(page.evaluate(READ_IMAGE), dtype=np.uint8)
    page.mouse.up()
    page.wait_for_timeout(1_500)
    released = np.asarray(page.evaluate(READ_IMAGE), dtype=np.uint8)
    return held, released


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_contrast_drag_shows_the_pixels_its_release_paints(webgpu, large_image, serve, open_page, hardware_adapter, browser_out):
    page = open_page(f"{serve(browser_out)}/{large_image}", webgpu=webgpu)
    page.wait_for_selector(IMAGE, timeout=60_000)
    page.wait_for_timeout(6_000)
    if webgpu:
        hardware_adapter(page.adapter())
    assert page.render_paths() == ["WebGPU" if webgpu else "CPU"]
    tag = "webgpu" if webgpu else "cpu"
    before = np.asarray(page.evaluate(READ_IMAGE), dtype=np.uint8)
    held, released = held_and_released(page, 3)
    page.screenshot(path=str(browser_out / f"show2d_contrast_fit_{tag}.png"))
    assert np.any(held != before), "the held drag did not repaint the image"
    assert np.array_equal(held, released), int(np.count_nonzero(held != released))
    # Zoomed in about a point off centre: the canvas edge cuts through data pixels.
    box = page.locator(IMAGE).bounding_box()
    page.mouse.move(box["x"] + 0.3 * box["width"], box["y"] + 0.6 * box["height"])
    for _ in range(3):
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(150)
    page.wait_for_timeout(1_000)
    zoomed = np.asarray(page.evaluate(READ_IMAGE), dtype=np.uint8)
    held, released = held_and_released(page, 4)
    page.screenshot(path=str(browser_out / f"show2d_contrast_zoom_{tag}.png"))
    assert np.any(held != zoomed), "the held drag did not repaint the zoomed image"
    assert np.array_equal(held, released), int(np.count_nonzero(held != released))
    assert page.errors == [], page.errors


@pytest.fixture(scope="module")
def denoised_gallery(browser_out) -> str:
    from quantem.widget import Show2D

    counts = np.random.default_rng(1).poisson(3.0, (256, 256)).astype(np.float32)
    Show2D([counts, counts], labels=["raw", "anscombe"], cmap="gray", denoise=["none", "anscombe"],
           denoise_sigma=4.0).export_html(browser_out / "show2d_denoise.html")
    return "show2d_denoise.html"


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_denoise_keyword_filters_the_panel_it_names(webgpu, denoised_gallery, serve, open_page, hardware_adapter, browser_out):
    page = open_page(f"{serve(browser_out)}/{denoised_gallery}", webgpu=webgpu)
    page.wait_for_selector('canvas[data-quantem-scientific-output="show2d-image-1"]', timeout=60_000)
    page.wait_for_timeout(6_000)
    if webgpu:
        hardware_adapter(page.adapter())
    page.screenshot(path=str(browser_out / f"show2d_denoise_{'webgpu' if webgpu else 'cpu'}.png"))
    raw, filtered = (np.asarray(page.evaluate(READ_PANEL, panel), dtype=float).reshape(-1, 4)[:, 0] for panel in (0, 1))
    side = int(np.sqrt(raw.size))
    roughness = [float(np.abs(np.diff(values.reshape(side, side), axis=1)).mean()) for values in (raw, filtered)]
    # Poisson counts at 3: neighbouring raw pixels differ by most of the gray range; the smoothed twin barely changes.
    assert roughness[1] < 0.25 * roughness[0], roughness
    assert page.errors == [], page.errors


def test_fft_click_reads_d_spacing_in_the_pixel_unit(serve, open_page, browser_out):
    from quantem.widget import Show2D

    rows, cols = np.indices((256, 256))
    lattice = (np.cos(2 * np.pi * cols / 8.0) + np.cos(2 * np.pi * rows / 8.0)).astype(np.float32)
    Show2D(lattice, sampling=0.2, units="nm", show_fft=True).export_html(browser_out / "show2d_fft_nm.html")
    page = open_page(f"{serve(browser_out)}/show2d_fft_nm.html", webgpu=False)
    fft = 'canvas[data-quantem-scientific-output="show2d-fft-0"]'
    page.wait_for_selector(fft, timeout=60_000)
    page.wait_for_timeout(4_000)
    box = page.locator(fft).bounding_box()
    # the Bragg spot 32 FFT pixels right of the centre, at the default 2x FFT zoom
    page.mouse.click(box["x"] + box["width"] / 2 + 2 * 32 * box["width"] / 256, box["y"] + box["height"] / 2)
    page.wait_for_timeout(800)
    page.screenshot(path=str(browser_out / "show2d_fft_nm.png"))
    match = re.search(r"d = ([\d.]+) (nm|Å) \| \|g\| = ([\d.]+) (\S+)", page.inner_text("body"))
    assert match, page.inner_text("body")[-400:]
    # 8 px period at 0.2 nm/px: d = 1.6 nm and |g| = 1/d in nm^-1 (the Angstrom-only labels read 1.60 Å)
    assert (float(match.group(1)), match.group(2), match.group(4)) == (1.6, "nm", "nm⁻¹"), match.group(0)
    assert float(match.group(3)) == pytest.approx(0.625, abs=1e-4), match.group(0)
    assert page.errors == [], page.errors
