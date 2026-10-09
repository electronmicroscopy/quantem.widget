"""Show3D draws sharp pixels until Smooth is switched on, with WebGPU on and off.

A two-level checkerboard is shown at exactly two screen pixels per data pixel.
With nearest-neighbour drawing the screen holds only the two colormap end
colors; bilinear interpolation would add intermediate grays along every block
edge, which is what the Smooth switch must then produce. The status line above
the image uses the normal text color on both display paths. A moving average
keeps Auto's stack-wide window, so averaging only reduces noise. A wheel over
the FFT overlay inset zooms the inset alone, through the one native listener
that can stop the page from scrolling. Blink compare alternates its two frames.
A held arrow key steps one frame per key repeat on the WebGPU fast path too.
Denoise and the frequency filter change the image only: the statistics describe the data.
Compare difference paints frame B minus frame A; the stats follow blink compare and the moving
average during playback; a zoomed multi-panel difference view paints the difference on WebGPU too.
"""

import io

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.browser

IMAGE = 'canvas[data-quantem-scientific-output="show3d-image"]'
BLACK, WHITE = (0, 0, 0), (255, 255, 255)


@pytest.fixture(scope="module")
def checkerboard(browser_out) -> str:
    from quantem.widget import Show3D

    rows, cols = np.indices((128, 128))
    stack = np.stack([((rows // 8 + (cols + 2 * frame) // 8) % 2).astype(np.float32) for frame in range(4)])
    Show3D(stack, panel_width_px=256, cmap="gray").export_html(browser_out / "show3d_checkerboard.html")
    return "show3d_checkerboard.html"


def screen_pixels(page, box) -> np.ndarray:
    """What the screen shows in the top-left 128 x 128 CSS pixels of the image (clear of the scale bar)."""
    clip = {"x": box["x"], "y": box["y"], "width": 128, "height": 128}
    return np.asarray(Image.open(io.BytesIO(page.screenshot(clip=clip))).convert("RGB"))


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_draws_sharp_pixels_until_smooth_is_on(webgpu, checkerboard, serve, open_page, hardware_adapter, browser_out):
    page = open_page(f"{serve(browser_out)}/{checkerboard}", webgpu=webgpu)
    page.wait_for_selector(IMAGE, timeout=60_000)
    page.wait_for_timeout(6_000)
    if webgpu:
        hardware_adapter(page.adapter())
    assert page.render_paths() == ["WebGPU" if webgpu else "CPU"]
    box = page.locator(IMAGE).bounding_box()
    assert round(box["width"]) == 256 and round(box["height"]) == 256  # 2 screen pixels per data pixel
    tag = "webgpu" if webgpu else "cpu"
    sharp = screen_pixels(page, box)
    Image.fromarray(sharp).save(browser_out / f"show3d_sharp_{tag}.png")
    assert {tuple(color) for color in sharp.reshape(-1, 3)} == {BLACK, WHITE}
    # along a row each 8-pixel block is 16 screen pixels wide and flips black to white in one step
    flips = np.flatnonzero(np.diff(sharp[4, :, 0].astype(int)))
    assert len(flips) >= 6 and np.all(np.diff(flips) == 16), flips
    page.get_by_label("Toggle bilinear smoothing").click()
    page.wait_for_timeout(1_500)
    smooth = screen_pixels(page, box)
    Image.fromarray(smooth).save(browser_out / f"show3d_smooth_{tag}.png")
    grays = {tuple(color) for color in smooth.reshape(-1, 3)} - {BLACK, WHITE}
    assert len(grays) >= 2, grays
    assert page.errors == [], page.errors


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_status_line_is_normal_text_on_both_paths(webgpu, checkerboard, serve, open_page, browser_out):
    # The CPU canvas is a supported display path, so its status line is information, not error red.
    page = open_page(f"{serve(browser_out)}/{checkerboard}", webgpu=webgpu)
    status = page.locator(f'[data-show3d-gpu-residency="{"ready" if webgpu else "fallback"}"]')
    status.wait_for(timeout=60_000)
    page.screenshot(path=str(browser_out / f"show3d_status_{'webgpu' if webgpu else 'cpu'}.png"))
    expected = "native float32 display · WebGPU resident · 4/4 frames" if webgpu else "WebGPU unavailable · using CPU/canvas: no WebGPU adapter"
    assert expected in status.inner_text()
    assert status.evaluate("element => getComputedStyle(element).color") in ("rgb(30, 30, 30)", "rgb(224, 224, 224)")  # theme text, light or dark


@pytest.fixture(scope="module")
def brightening(browser_out) -> dict:
    """Nine frames of one gradient, each brighter than the last, opened with a 3-frame moving average."""
    from quantem.widget import Show3D

    rows, cols = np.indices((128, 128))
    gradient = 1.0 + (rows + cols) / 254.0
    stack = np.stack([gradient * (1.0 + 0.5 * frame) for frame in range(9)]).astype(np.float32)
    viewer = Show3D(stack, panel_width_px=256, cmap="gray", avg_window=3)
    viewer.export_html(browser_out / "show3d_average.html")
    low, high = np.float32(viewer.auto_vmins[0]), np.float32(viewer.auto_vmaxs[0])
    # the shown frame (the middle one) averages itself and its two neighbours
    shown = viewer.slice_idx
    mean = stack[shown - 1:shown + 2].sum(axis=0, dtype=np.float32) / np.float32(3)
    levels = np.minimum(np.trunc(np.clip((mean - low) / (high - low), 0, 1) * np.float32(255)), 255)
    return {"page": "show3d_average.html", "levels": levels, "window": (float(low), float(high))}


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_moving_average_keeps_the_stack_auto_window(webgpu, brightening, serve, open_page, hardware_adapter, browser_out):
    page = open_page(f"{serve(browser_out)}/{brightening['page']}", webgpu=webgpu)
    page.wait_for_selector(IMAGE, timeout=60_000)
    page.wait_for_timeout(6_000)
    if webgpu:
        hardware_adapter(page.adapter())
    shot = screen_pixels(page, page.locator(IMAGE).bounding_box())
    Image.fromarray(shot).save(browser_out / f"show3d_average_{'webgpu' if webgpu else 'cpu'}.png")
    # two screen pixels per data pixel: the top-left 128 CSS px show data rows and columns 0..63
    drawn = shot[1::2, 1::2, 0].astype(int)
    expected = brightening["levels"][:64, :64].astype(int)
    assert np.abs(drawn - expected).max() <= 1, (brightening["window"], drawn.min(), drawn.max(), expected.min(), expected.max())
    assert page.errors == [], page.errors


INSET = '[data-show3d-fft-inset="true"]'
SCROLL = "() => [window.scrollX, window.scrollY, document.scrollingElement.scrollTop]"
# The React props of the inset element: React keeps them on the DOM node in production builds too.
INSET_WHEEL_PROP = """
(selector) => {
  const inset = document.querySelector(selector);
  const key = Object.keys(inset).find(name => name.startsWith('__reactProps'));
  return typeof inset[key].onWheel;
}
"""


@pytest.fixture(scope="module")
def fft_overlay(browser_out) -> str:
    from quantem.widget import Show3D

    rows, cols = np.indices((128, 128))
    lattice = np.cos(2 * np.pi * cols / 6) + np.cos(2 * np.pi * rows / 9)
    stack = np.stack([lattice + 0.1 * frame * np.cos(2 * np.pi * (rows + cols) / 13) for frame in range(3)]).astype(np.float32)
    Show3D(stack, panel_width_px=384, show_fft=True, fft_layout="overlay").export_html(browser_out / "show3d_fft_overlay.html")
    return "show3d_fft_overlay.html"


def screen(page, box) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(page.screenshot(clip=box))).convert("RGB"))


def spot_radius(inset: np.ndarray) -> float:
    """Mean distance of the bright FFT spots from the inset centre, inside its border and below its title."""
    rows, cols = np.nonzero(inset[20:-4, 4:-4].sum(axis=2) > 3 * 128)
    center_row, center_col = inset.shape[0] / 2 - 20, inset.shape[1] / 2 - 4
    return float(np.hypot(rows - center_row, cols - center_col).mean())


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_fft_inset_wheel_zooms_the_inset_alone(webgpu, fft_overlay, serve, open_page, hardware_adapter, browser_out):
    page = open_page(f"{serve(browser_out)}/{fft_overlay}", webgpu=webgpu)
    page.wait_for_selector(INSET, timeout=60_000)
    page.wait_for_timeout(6_000)
    if webgpu:
        hardware_adapter(page.adapter())
    # The root's capture listener is the only wheel path; React's passive onWheel could not stop the scroll.
    assert page.evaluate(INSET_WHEEL_PROP, INSET) == "undefined"
    inset = page.locator(INSET).first.bounding_box()
    image = page.locator(IMAGE).bounding_box()
    # The image quadrant diagonally opposite the top-left inset
    beside = {"x": image["x"] + image["width"] / 2, "y": image["y"] + image["height"] / 2,
              "width": image["width"] / 2 - 2, "height": image["height"] / 4}
    before_inset, before_image, scroll = screen(page, inset), screen(page, beside), page.evaluate(SCROLL)
    page.mouse.move(inset["x"] + inset["width"] / 2, inset["y"] + inset["height"] / 2)
    page.mouse.wheel(0, -300)
    page.wait_for_timeout(1_500)
    after_inset, after_image = screen(page, inset), screen(page, beside)
    tag = "webgpu" if webgpu else "cpu"
    Image.fromarray(np.concatenate([before_inset, after_inset], axis=1)).save(browser_out / f"show3d_fft_inset_wheel_{tag}.png")
    assert spot_radius(after_inset) > 1.2 * spot_radius(before_inset)  # the inset zoomed in about the cursor
    assert np.array_equal(before_image, after_image)  # the image under it did not
    assert page.evaluate(SCROLL) == scroll
    assert page.errors == [], page.errors


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_blink_compare_alternates_its_two_frames(webgpu, serve, open_page, browser_out):
    # A black and a white frame at 2 Hz: the image must show both, switching several times in 4 s.
    from quantem.widget import Show3D

    stack = np.stack([np.zeros((64, 64), np.float32), np.ones((64, 64), np.float32)])
    widget = Show3D(stack, panel_width_px=256, cmap="gray")
    widget.compare_mode, widget.blink_fps = "blink", 2.0
    widget.export_html(browser_out / "show3d_blink.html")
    page = open_page(f"{serve(browser_out)}/show3d_blink.html", webgpu=webgpu)
    page.wait_for_selector(IMAGE, timeout=60_000)
    page.wait_for_timeout(4_000)
    box = page.locator(IMAGE).bounding_box()
    clip = {"x": box["x"] + 64, "y": box["y"] + 64, "width": 16, "height": 16}
    levels = []
    for _ in range(40):
        levels.append(float(np.asarray(Image.open(io.BytesIO(page.screenshot(clip=clip))).convert("L")).mean()))
        page.wait_for_timeout(100)
    switches = int(np.count_nonzero(np.abs(np.diff(levels)) > 128))
    assert min(levels) < 64 and max(levels) > 192 and switches >= 4, (switches, levels)
    assert page.errors == [], page.errors


@pytest.fixture(scope="module")
def numbered(browser_out) -> dict:
    """60 uniform 64 x 64 frames whose value is the frame number, at 4 screen pixels per data pixel.

    The integer enlargement keeps the WebGPU run on the resident fast path, which draws a scrubbed frame
    at once and commits React state only after the input settles.
    """
    from quantem.widget import Show3D

    stack = np.broadcast_to(np.arange(60, dtype=np.float32)[:, None, None], (60, 64, 64)).copy()
    viewer = Show3D(stack, panel_width_px=256, cmap="gray")
    viewer.export_html(browser_out / "show3d_numbered.html")
    low, high = np.float32(viewer.auto_vmins[0]), np.float32(viewer.auto_vmaxs[0])
    levels = np.minimum(np.trunc(np.clip((np.arange(60, dtype=np.float32) - low) / (high - low), 0, 1) * np.float32(255)), 255)
    return {"page": "show3d_numbered.html", "levels": levels.astype(int)}


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_held_arrow_key_steps_one_frame_per_repeat(webgpu, numbered, serve, open_page, hardware_adapter, browser_out):
    # ArrowRight held for 1 s at a 30 Hz key repeat: 30 frames on, on both display paths
    page = open_page(f"{serve(browser_out)}/{numbered['page']}", webgpu=webgpu)
    page.locator(f'[data-show3d-gpu-residency="{"ready" if webgpu else "fallback"}"]').wait_for(timeout=60_000)
    page.wait_for_timeout(2_000)
    if webgpu:
        hardware_adapter(page.adapter())
    count = page.locator("[data-show3d-playback-count]")
    box = page.locator(IMAGE).bounding_box()
    center = {"x": box["x"] + box["width"] / 2 - 8, "y": box["y"] + box["height"] / 2 - 8, "width": 16, "height": 16}
    page.focus(".show3d-root")
    page.keyboard.press("Home")
    page.wait_for_timeout(1_000)
    assert count.inner_text() == "1/60"
    for _ in range(30):
        page.keyboard.down("ArrowRight")  # a repeat keydown while the key stays down
        page.wait_for_timeout(33)
    page.keyboard.up("ArrowRight")
    held = count.inner_text(), int(screen(page, center)[..., 0].mean().round())
    page.wait_for_timeout(1_000)
    settled = count.inner_text(), int(screen(page, center)[..., 0].mean().round())
    page.screenshot(path=str(browser_out / f"show3d_held_key_{'webgpu' if webgpu else 'cpu'}.png"))
    expected = numbered["levels"][30]
    assert held[0] == settled[0] == "31/60" and abs(held[1] - expected) <= 1 and abs(settled[1] - expected) <= 1, (
        held, settled, numbered["levels"][28:33])
    assert page.errors == [], page.errors


STATS_STD = 'p:has-text("Std") span'


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_stats_describe_the_data_under_a_view_filter(webgpu, serve, open_page, hardware_adapter, browser_out):
    # Denoise smooths the noise on screen; the Std readout still reads the unfiltered frame.
    from quantem.widget import Show3D

    stack = (5.0 + 1.37 * np.random.default_rng(0).standard_normal((3, 64, 64))).astype(np.float32)
    viewer = Show3D(stack, panel_width_px=256, cmap="gray", show_stats=True)
    viewer.denoise, viewer.denoise_sigma, viewer.denoise_enabled = "gaussian", 3.0, True
    viewer.export_html(browser_out / "show3d_view_filter_stats.html")
    page = open_page(f"{serve(browser_out)}/show3d_view_filter_stats.html", webgpu=webgpu)
    page.wait_for_selector(IMAGE, timeout=60_000)
    page.wait_for_timeout(6_000)
    if webgpu:
        hardware_adapter(page.adapter())
    shot = screen(page, page.locator(IMAGE).bounding_box())
    Image.fromarray(shot).save(browser_out / f"show3d_view_filter_stats_{'webgpu' if webgpu else 'cpu'}.png")
    # one sample per data pixel (4 screen pixels each), clear of the scale bar
    levels = shot[2:192:4, 2:192:4, 0].astype(int)
    assert np.abs(np.diff(levels, axis=1)).mean() < 20  # smoothed: unfiltered noise steps by about 56 levels
    shown = stack[viewer.slice_idx]
    assert abs(float(page.locator(STATS_STD).first.inner_text()) - float(shown.std())) < 0.006
    assert page.errors == [], page.errors


def open_ready(page_url, webgpu, open_page, hardware_adapter):
    page = open_page(page_url, webgpu=webgpu)
    page.locator(f'[data-show3d-gpu-residency="{"ready" if webgpu else "fallback"}"]').wait_for(timeout=60_000)
    page.wait_for_timeout(2_000)
    if webgpu:
        hardware_adapter(page.adapter())
    return page


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_compare_difference_paints_b_minus_a(webgpu, serve, open_page, hardware_adapter, browser_out):
    # Frame B is frame A plus 1 everywhere: the signed difference map is uniform full magenta.
    # Before, frame A was read back from the displayed frame (already B - A), which painted A itself.
    from quantem.widget import Show3D

    rows, cols = np.indices((64, 64))
    ramp = (1.0 + (rows + cols) / 126.0).astype(np.float32)
    viewer = Show3D(np.stack([ramp, ramp + 1, ramp + 3]), panel_width_px=256, cmap="gray", show_stats=True)
    viewer.slice_idx, viewer.compare_pair, viewer.compare_mode = 0, [0, 1], "difference"
    viewer.export_html(browser_out / "show3d_compare_difference.html")
    page = open_ready(f"{serve(browser_out)}/show3d_compare_difference.html", webgpu, open_page, hardware_adapter)
    shot = screen(page, page.locator(IMAGE).bounding_box())
    Image.fromarray(shot).save(browser_out / f"show3d_compare_difference_{'webgpu' if webgpu else 'cpu'}.png")
    inner = shot[8:192, 8:248].reshape(-1, 3)  # clear of the scale bar
    assert np.abs(inner - np.array([255, 0, 255])).max() <= 2, np.unique(inner, axis=0)[:5]
    assert float(page.locator(STATS_STD).first.inner_text()) < 0.01  # the stats describe B - A = 1, not frame A
    assert page.errors == [], page.errors


# Every Std text the page shows, read at the start of each frame for `ms` milliseconds.
STD_EVERY_FRAME = """
(ms) => new Promise(resolve => {
  const seen = [];
  const end = performance.now() + ms;
  const read = () => {
    const span = [...document.querySelectorAll('p')].find(p => p.textContent.startsWith('Std'))?.querySelector('span');
    if (span) seen.push(parseFloat(span.textContent));
    if (performance.now() < end) requestAnimationFrame(read); else resolve(seen);
  };
  requestAnimationFrame(read);
})
"""


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_stats_follow_the_moving_average_while_playing(webgpu, serve, open_page, hardware_adapter, browser_out):
    # Independent noise per frame: a 3-frame mean has std 1.37 / sqrt(3) = 0.79, a single frame 1.37.
    # Before, the playback loop wrote single-frame stats that showed until the frame effect replaced them.
    from quantem.widget import Show3D

    stack = (5.0 + 1.37 * np.random.default_rng(1).standard_normal((24, 64, 64))).astype(np.float32)
    viewer = Show3D(stack, panel_width_px=256, cmap="gray", show_stats=True, avg_window=3, fps=10)
    viewer.export_html(browser_out / "show3d_playing_average.html")
    page = open_ready(f"{serve(browser_out)}/show3d_playing_average.html", webgpu, open_page, hardware_adapter)
    page.get_by_label("Play", exact=True).first.click()
    page.wait_for_timeout(500)
    readings = page.evaluate(STD_EVERY_FRAME, 2_500)
    counts = page.locator("[data-show3d-playback-count]").inner_text()
    assert counts != "1/24" and len(readings) > 60 and max(readings) < 1.0, (counts, sorted(set(readings)))
    assert page.errors == [], page.errors


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_stats_follow_blink_compare(webgpu, serve, open_page, hardware_adapter, browser_out):
    # Blink alternates a std-1 and a std-3 frame at 2 Hz: the stats must alternate with them.
    from quantem.widget import Show3D

    noise = np.random.default_rng(2).standard_normal((2, 64, 64)).astype(np.float32)
    viewer = Show3D(np.stack([noise[0], 3 * noise[1]]), panel_width_px=256, cmap="gray", show_stats=True)
    viewer.compare_pair, viewer.compare_mode, viewer.blink_fps = [0, 1], "blink", 2.0
    viewer.export_html(browser_out / "show3d_blink_stats.html")
    page = open_ready(f"{serve(browser_out)}/show3d_blink_stats.html", webgpu, open_page, hardware_adapter)
    readings = page.evaluate(STD_EVERY_FRAME, 3_000)
    assert min(readings) < 1.2 and max(readings) > 2.8, sorted(set(readings))
    assert page.errors == [], page.errors


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_panels_difference_view_paints_the_difference(webgpu, serve, open_page, hardware_adapter, browser_out):
    # Two panels, Diff "first": every frame minus frame 0 is a uniform offset, so each panel is flat, also
    # after a wheel zoom. Before, the WebGPU zoom repaint drew the raw resident frame (the ramp).
    from quantem.widget import Show3D

    rows, cols = np.indices((64, 64))
    ramp = (rows + cols).astype(np.float32)  # integers: frame k minus frame 0 is exactly k
    frames = np.stack([ramp + 10 * k for k in range(4)])
    viewer = Show3D(frames, frames[:, ::-1].copy(), panel_width_px=256, cmap="gray")
    viewer.slice_idx, viewer.diff_mode = 2, "first"
    viewer.export_html(browser_out / "show3d_panels_difference.html")
    page = open_ready(f"{serve(browser_out)}/show3d_panels_difference.html", webgpu, open_page, hardware_adapter)
    box = page.locator(IMAGE).bounding_box()
    page.mouse.move(box["x"] + 128, box["y"] + 128)
    page.mouse.wheel(0, -200)
    page.wait_for_timeout(1_500)
    shot = screen(page, box)
    Image.fromarray(shot).save(browser_out / f"show3d_panels_difference_{'webgpu' if webgpu else 'cpu'}.png")
    first_panel = shot[40:180, 40:220, 0].astype(int)  # clear of the title and the scale bar
    assert np.ptp(first_panel) <= 2, np.ptp(first_panel)
    assert page.errors == [], page.errors


FFT = 'canvas[data-quantem-scientific-output="show3d-fft"]'


@pytest.mark.parametrize("layout", ["right", "bottom"])
@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_zoomed_fft_keeps_zero_frequency_at_the_centre(webgpu, layout, serve, open_page, hardware_adapter, browser_out):
    # The tutorial's first viewer: fft_overlay_zoom=2.0 must zoom about the centre, where the zero frequency
    # sits. A lattice of period 8 px puts its four peaks symmetric about that centre.
    from quantem.widget import Show3D

    rows, cols = np.indices((128, 128))
    lattice = np.cos(2 * np.pi * cols / 8) + np.cos(2 * np.pi * rows / 8)
    stack = np.stack([lattice * (1 + 0.1 * frame) for frame in range(3)]).astype(np.float32)
    Show3D(stack, panel_width_px=256, show_fft=True, fft_layout=layout, fft_overlay_zoom=2.0).export_html(
        browser_out / f"show3d_fft_zoom_{layout}.html")
    page = open_ready(f"{serve(browser_out)}/show3d_fft_zoom_{layout}.html", webgpu, open_page, hardware_adapter)
    page.wait_for_timeout(2_000)
    box = page.locator(FFT).bounding_box()
    shot = screen(page, box)
    Image.fromarray(shot).save(browser_out / f"show3d_fft_zoom_{layout}_{'webgpu' if webgpu else 'cpu'}.png")
    level = shot.astype(int).sum(axis=2)
    level[:24] = 0  # the quality caption along the top
    # the canvas can sit on a half pixel, so the clip may hold one row or column of page background
    level[-2:], level[:, :2], level[:, -2:] = 0, 0, 0
    bright_rows, bright_cols = np.nonzero(level > 0.8 * level.max())
    centre = np.array([shot.shape[0] / 2, shot.shape[1] / 2])
    offset = np.array([bright_rows.mean(), bright_cols.mean()]) - centre
    assert np.all(np.abs(offset) < 0.08 * shot.shape[0]), (offset, shot.shape)
    assert page.errors == [], page.errors
