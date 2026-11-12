"""Show3D draws true-color stacks on WebGPU whenever an adapter exists, and says which path drew.

Each RGB frame is resident on the GPU and drawn by the direct true-color pass during scrub and
playback; the JavaScript pack loop runs only on a page without WebGPU. The title badge and the
status line name the path that drew the frame, and both paths read back the same bytes:
round(value * 255) per channel, the color the frame was given. One and two panels, as an exported
page and live in a notebook.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

IMAGE = 'canvas[data-quantem-scientific-output="show3d-image"]'
FRAMES, SIZE = 3, 256

# Count WebGPU presents into page canvases and Canvas2D putImageData calls (the CPU pack loop).
COUNTERS = """
window.__draws = {webgpu: 0, cpu: 0};
const currentTexture = GPUCanvasContext.prototype.getCurrentTexture;
GPUCanvasContext.prototype.getCurrentTexture = function () {
  if (this.canvas instanceof HTMLCanvasElement) window.__draws.webgpu++;
  return currentTexture.call(this);
};
const putImageData = CanvasRenderingContext2D.prototype.putImageData;
CanvasRenderingContext2D.prototype.putImageData = function (...args) {
  window.__draws.cpu++;
  return putImageData.apply(this, args);
};
"""
READBACK = f"""() => {{
  const canvas = document.querySelector('{IMAGE}');
  const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  return {{width: canvas.width, height: canvas.height, pixels: Array.from(pixels)}};
}}"""
DRAWS = "() => ({...window.__draws})"
PLAY_FOR_TWO_SECONDS = """() => new Promise(resolve => {
  let frames = 0;
  const start = performance.now();
  const tick = () => {
    frames++;
    if (performance.now() - start < 2000) requestAnimationFrame(tick);
    else resolve({frames, seconds: (performance.now() - start) / 1000});
  };
  requestAnimationFrame(tick);
})"""


def rgb_stacks() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    return tuple(rng.integers(0, 256, size=(FRAMES, SIZE, SIZE, 3), dtype=np.uint8) for _ in range(2))


@pytest.fixture(scope="module")
def rgb_pages(browser_out) -> dict:
    from quantem.widget import Show3D

    left, right = rgb_stacks()
    Show3D(left, panel_width_px=SIZE, fps=60).export_html(browser_out / "show3d_rgb_one.html")
    Show3D([left, right], panel_width_px=SIZE, fps=60).export_html(browser_out / "show3d_rgb_two.html")
    return {1: ("show3d_rgb_one.html", [left]), 2: ("show3d_rgb_two.html", [left, right])}


def panel_bytes(page, panels: int) -> np.ndarray:
    """The drawn panels side by side, the gap between two panels left out."""
    read = page.evaluate(READBACK)
    image = np.asarray(read["pixels"], dtype=np.uint8).reshape(read["height"], read["width"], 4)[..., :3]
    assert read["height"] == SIZE and read["width"] >= panels * SIZE, (read["width"], read["height"])
    return np.concatenate([image[:, :SIZE], image[:, -SIZE:]], axis=1) if panels == 2 else image


def expected_bytes(stacks: list[np.ndarray], frame: int) -> np.ndarray:
    return np.concatenate([stack[frame] for stack in stacks], axis=1)


def check_scrub_and_playback(page, panels: int, stacks: list[np.ndarray], webgpu: bool) -> dict:
    """Scrub to frames 0 and 1, comparing the readback to the given colors, then play for two seconds."""
    page.focus(".show3d-root")
    page.keyboard.press("Home")
    page.wait_for_timeout(1_000)
    assert np.array_equal(panel_bytes(page, panels), expected_bytes(stacks, 0))
    before = page.evaluate(DRAWS)
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(1_000)
    after = page.evaluate(DRAWS)
    assert np.array_equal(panel_bytes(page, panels), expected_bytes(stacks, 1))
    scrub = {path: after[path] - before[path] for path in after}
    # the step draws at once and again when its slice commits to the model; both on the one path
    drawn, other = ("webgpu", "cpu") if webgpu else ("cpu", "webgpu")
    assert scrub[drawn] >= 1 and scrub[other] == 0, scrub

    page.get_by_label("Play forward").click()
    before = page.evaluate(DRAWS)
    clock = page.evaluate(PLAY_FOR_TWO_SECONDS)
    after = page.evaluate(DRAWS)
    rates = {path: (after[path] - before[path]) / clock["seconds"] for path in after}
    display_rate = min(60.0, clock["frames"] / clock["seconds"])
    # every displayed frame of a 60 fps playback is a new frame, drawn on one path only
    assert rates[drawn] >= 0.9 * display_rate and rates[other] == 0, (rates, display_rate)
    return {"rates": rates, "display_rate": display_rate}


@pytest.mark.parametrize("panels", [1, 2], ids=["one-panel", "two-panels"])
@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show3d_rgb_draws_on_the_path_its_badge_names(webgpu, panels, rgb_pages, serve, chrome, hardware_adapter, browser_out):
    name, stacks = rgb_pages[panels]
    context = chrome.new_context(viewport={"width": 1600, "height": 1050}, device_scale_factor=1)
    context.add_init_script(COUNTERS)
    if not webgpu:
        context.add_init_script("Object.defineProperty(Navigator.prototype, 'gpu', {get: () => undefined, configurable: true});")
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)[:300]))
    try:
        page.goto(f"{serve(browser_out)}/{name}")
        status = page.locator(f'[data-show3d-gpu-residency="{"ready" if webgpu else "fallback"}"]')
        status.wait_for(timeout=60_000)
        page.wait_for_timeout(2_000)
        if webgpu:
            hardware_adapter(page.evaluate(
                "async () => { const a = await navigator.gpu.requestAdapter(); return a && {vendor: a.info.vendor, architecture: a.info.architecture}; }"))
        expected = (f"native true-color display · WebGPU resident · {FRAMES}/{FRAMES} frames" if webgpu
                    else "WebGPU unavailable · using CPU/canvas: no WebGPU adapter")
        assert expected in status.inner_text()
        badges = "() => Array.from(document.querySelectorAll('[data-render-path]')).map(e => e.dataset.renderPath)"
        assert page.evaluate(badges) == ["WebGPU" if webgpu else "CPU"]
        check_scrub_and_playback(page, panels, stacks, webgpu)
        assert page.evaluate(badges) == ["WebGPU" if webgpu else "CPU"]
        page.screenshot(path=str(browser_out / f"show3d_rgb_{panels}_{'webgpu' if webgpu else 'cpu'}.png"))
        assert errors == [], errors
    finally:
        context.close()


def test_show3d_rgb_live_notebook_draws_on_webgpu(jupyter, chrome, hardware_adapter, browser_out):
    np.save(jupyter.root / "rgb_left.npy", rgb_stacks()[0])
    np.save(jupyter.root / "rgb_right.npy", rgb_stacks()[1])
    jupyter.write_notebook("show3d_rgb_live.ipynb", (
        "import numpy as np\nfrom quantem.widget import Show3D\n\n"
        f"Show3D([np.load('rgb_left.npy'), np.load('rgb_right.npy')], panel_width_px={SIZE}, fps=60)"))

    def open_page(url, *, webgpu):
        context = chrome.new_context(viewport={"width": 1600, "height": 1050}, device_scale_factor=1)
        context.add_init_script(COUNTERS)
        page = context.new_page()
        page.errors, page.console = [], []
        page.on("pageerror", lambda error: page.errors.append(str(error)[:300]))
        page.goto(url)
        page.adapter = lambda: page.evaluate(
            "async () => { const a = await navigator.gpu.requestAdapter(); return a && {vendor: a.info.vendor, architecture: a.info.architecture}; }")
        page.context_handle = context
        return page

    page = jupyter.open(open_page, "show3d_rgb_live.ipynb", "show3d-rgb-live", webgpu=True,
                        ready='[data-show3d-gpu-residency="ready"]', out=browser_out)
    try:
        hardware_adapter(page.adapter())
        status = page.locator('[data-show3d-gpu-residency="ready"]').inner_text()
        assert f"native true-color display · WebGPU resident · {FRAMES}/{FRAMES} frames" in status
        assert page.evaluate("() => Array.from(document.querySelectorAll('[data-render-path]')).map(e => e.dataset.renderPath)") == ["WebGPU"]
        check_scrub_and_playback(page, 2, list(rgb_stacks()), webgpu=True)
        page.screenshot(path=str(browser_out / "show3d_rgb_live.png"))
        # JupyterLab's debugger extension reports a missing session on its own.
        assert [error for error in page.errors if "debugger session" not in error] == [], page.errors
    finally:
        page.context_handle.close()
