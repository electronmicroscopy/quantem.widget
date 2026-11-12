"""Show2D packs a color panel on WebGPU whenever an adapter exists, beside a grayscale panel.

The true-color panel goes through the colormap engine like the grayscale one: its WebGPU pass when
the browser has an adapter, the Canvas2D engine otherwise. Both read back the bytes
round(value * 255) per channel, and the title badge names the path that packed them.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

COLOR = 'canvas[data-quantem-scientific-output="show2d-image-0"]'
SIZE = 256
# Count CPU-made pixels: putImageData (the JavaScript pack) and bitmaps made from ImageData (the Canvas2D engine).
COUNTERS = """
window.__cpuPixels = 0;
for (const proto of [CanvasRenderingContext2D.prototype, OffscreenCanvasRenderingContext2D.prototype]) {
  const put = proto.putImageData;
  proto.putImageData = function (...args) { window.__cpuPixels++; return put.apply(this, args); };
}
const bitmap = window.createImageBitmap;
window.createImageBitmap = function (source, ...rest) {
  if (source instanceof ImageData) window.__cpuPixels++;
  return bitmap.call(this, source, ...rest);
};
"""
READBACK = f"""() => {{
  const canvas = document.querySelector('{COLOR}');
  const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  return {{width: canvas.width, height: canvas.height, pixels: Array.from(pixels)}};
}}"""


@pytest.mark.parametrize("second", ["gray", "color"])
@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_show2d_color_panel_draws_on_the_path_its_badge_names(webgpu, second, jupyter, chrome, hardware_adapter, browser_out):
    rng = np.random.default_rng(3)
    np.save(jupyter.root / "show2d_color.npy", rng.integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8))
    np.save(jupyter.root / "show2d_gray.npy", rng.random((SIZE, SIZE)).astype(np.float32))
    np.save(jupyter.root / "show2d_color2.npy", rng.integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8))
    tag = f"{'webgpu' if webgpu else 'cpu'}_{second}"
    notebook = f"show2d_rgb_{tag}.ipynb"
    jupyter.write_notebook(notebook, (
        "import numpy as np\nfrom quantem.widget import Show2D\n\n"
        f"Show2D([np.load('show2d_color.npy'), np.load('show2d_{'gray' if second == 'gray' else 'color2'}.npy')], panel_width_px={SIZE})"))
    contexts = []

    def open_page(url, *, webgpu):
        context = chrome.new_context(viewport={"width": 1600, "height": 1050}, device_scale_factor=1)
        contexts.append(context)
        context.add_init_script(COUNTERS)
        if not webgpu:
            context.add_init_script("Object.defineProperty(Navigator.prototype, 'gpu', {get: () => undefined, configurable: true});")
        page = context.new_page()
        page.errors, page.console = [], []
        page.on("pageerror", lambda error: page.errors.append(str(error)[:300]))
        page.goto(url)
        return page

    try:
        page = jupyter.open(open_page, notebook, f"show2d-rgb-{tag.replace('_', '-')}", webgpu=webgpu,
                            ready=COLOR, out=browser_out)
        if webgpu:
            hardware_adapter(page.evaluate(
                "async () => { const a = await navigator.gpu.requestAdapter(); return a && {vendor: a.info.vendor, architecture: a.info.architecture}; }"))
        badges = page.evaluate("() => Array.from(document.querySelectorAll('[data-render-path]')).map(e => e.dataset.renderPath)")
        assert badges == ["WebGPU" if webgpu else "CPU"], badges
        read = page.evaluate(READBACK)
        assert (read["width"], read["height"]) == (SIZE, SIZE), read["width"]
        drawn = np.asarray(read["pixels"], dtype=np.uint8).reshape(SIZE, SIZE, 4)[..., :3]
        assert np.array_equal(drawn, np.load(jupyter.root / "show2d_color.npy"))
        packed_on_cpu = page.evaluate("() => window.__cpuPixels")
        assert (packed_on_cpu == 0) if webgpu else (packed_on_cpu > 0), packed_on_cpu
        page.screenshot(path=str(browser_out / f"show2d_rgb_{tag}.png"))
        # JupyterLab's debugger extension reports a missing session on its own.
        assert [error for error in page.errors if "debugger session" not in error] == [], page.errors
    finally:
        for context in contexts:
            context.close()
