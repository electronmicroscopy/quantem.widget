"""Exported widgets draw their data, and the same pixels with WebGPU and with Canvas2D.

Each tutorial-sized export opens twice in one headed Chrome: with WebGPU on a
hardware adapter, and with ``navigator.gpu`` removed so the widget takes its
JavaScript path. Equal pixels are not enough on their own (two blank canvases
are equal), so every page must first show its data: each image and FFT panel
visible, on top and non-blank, each histogram with bars that follow the data,
and the panels repainting when a histogram handle moves. Show2D and Show3D
canvases must then match within 1 LSB per channel. The Show4DSTEM export
reduces the 4D data on WebGPU only, so without it the page must keep the saved
views and say what it needs.
"""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.browser

READ_CANVASES = """
() => Array.from(document.querySelectorAll('canvas')).filter(canvas => {
  const rect = canvas.getBoundingClientRect();
  return canvas.width && canvas.height && rect.width && rect.height;
}).map(canvas => {
  let context = canvas.getContext('2d');
  if (!context) {
    const copy = document.createElement('canvas');
    copy.width = canvas.width; copy.height = canvas.height;
    context = copy.getContext('2d');
    context.drawImage(canvas, 0, 0);
  }
  return {width: canvas.width, height: canvas.height,
          pixels: Array.from(context.getImageData(0, 0, canvas.width, canvas.height).data)};
})
"""


# Per visible canvas: owner panel, whether it is the topmost element at its
# centre, pixel spread, distinct colors and a checksum; histogram canvases also
# report the bar height of each pixel column (pixels that differ from the
# background color).
APPEARANCE = """
() => Array.from(document.querySelectorAll('canvas')).filter(canvas => {
  const rect = canvas.getBoundingClientRect();
  return canvas.width && canvas.height && rect.width && rect.height;
}).map(canvas => {
  canvas.scrollIntoView({block: 'center', inline: 'center'});
  const rect = canvas.getBoundingClientRect();
  let context = canvas.getContext('2d');
  const readable = !!context;
  if (!context) {
    const copy = document.createElement('canvas');
    copy.width = canvas.width; copy.height = canvas.height;
    context = copy.getContext('2d');
    context.drawImage(canvas, 0, 0);
  }
  const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const counts = new Map();
  let sum = 0, sum2 = 0, checksum = 0;
  for (let i = 0; i < data.length; i += 4) {
    const color = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
    counts.set(color, (counts.get(color) || 0) + 1);
    const value = data[i] + data[i + 1] + data[i + 2];
    sum += value; sum2 += value * value; checksum = (checksum * 31 + value + data[i + 3]) % 2147483647;
  }
  const n = data.length / 4;
  const owner = canvas.closest('[data-quantem-scientific-output]');
  const histogram = !!canvas.parentElement && !!canvas.parentElement.querySelector('input[aria-label="Histogram intensity clip range"]');
  let columns = null;
  if (histogram) {
    let background = 0, most = -1;
    for (const [color, count] of counts) if (count > most) { most = count; background = color; }
    columns = [];
    for (let x = 0; x < canvas.width; x++) {
      let bar = 0;
      for (let y = 0; y < canvas.height; y++) {
        const i = 4 * (y * canvas.width + x);
        if (((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]) !== background) bar++;
      }
      columns.push(bar);
    }
  }
  const top = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
  return {width: canvas.width, height: canvas.height, css: [rect.width, rect.height],
          owner: owner ? owner.dataset.quantemScientificOutput : null, histogram, readable,
          on_top: top === canvas, std: Math.sqrt(Math.max(0, sum2 / n - (sum / n) ** 2)),
          colors: counts.size, checksum, columns};
})
"""
# Moves the max handle of the first histogram to 50 % through React's own input path.
MOVE_FIRST_HISTOGRAM_MAX = """
() => {
  const canvas = Array.from(document.querySelectorAll('canvas'))
    .find(c => c.parentElement && c.parentElement.querySelector('input[aria-label="Histogram intensity clip range"]'));
  const inputs = canvas.parentElement.querySelectorAll('input[type=range]');
  const input = inputs[inputs.length - 1];
  Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(input, '50');
  input.dispatchEvent(new Event('input', {bubbles: true}));
  input.dispatchEvent(new Event('change', {bubbles: true}));
}
"""


def assert_shows_data(page, result: dict, name: str) -> None:
    """Every panel visible, on top and non-blank; every histogram has bars."""
    panels = [canvas for canvas in result if canvas["owner"]]
    histograms = [canvas for canvas in result if canvas["histogram"]]
    assert panels and histograms, (name, len(panels), len(histograms))
    for canvas in panels:
        assert canvas["on_top"], (name, canvas["owner"], "covered")
        if canvas["readable"]:
            assert canvas["std"] > 1 and canvas["colors"] >= 8, (name, canvas["owner"], canvas["std"], canvas["colors"])
        else:
            # A WebGPU canvas reads back blank through drawImage; read what the screen shows
            # (a clip, not an element screenshot, which waits for a still frame).
            owner = f'[data-quantem-scientific-output="{canvas["owner"]}"]'
            box = page.locator(f"canvas{owner}, {owner} canvas").first.bounding_box()
            shot = np.asarray(Image.open(io.BytesIO(page.screenshot(clip=box))).convert("RGB"))
            assert shot.reshape(-1, 3).std(axis=0).max() > 1, (name, canvas["owner"], "blank on screen")
    for index, canvas in enumerate(histograms):
        filled = sum(1 for bar in canvas["columns"] if bar > 0)
        assert filled >= 4, (name, f"histogram {index} draws {filled} bar columns")


def histogram_bars(columns: list[int], bars: int = 64) -> np.ndarray:
    """Bar heights of the shared Histogram: ``bars`` bars over the canvas width, one column each."""
    width = len(columns)
    return np.array([columns[min(width - 1, int(index * width / bars + 0.5))] for index in range(bars)], dtype=float)


def render(open_page, url: str, screenshot: Path, *, webgpu: bool) -> dict:
    page = open_page(url, webgpu=webgpu)
    page.wait_for_selector("canvas", timeout=60_000)
    # Bundle mount, colormap, FFT and histogram settle asynchronously.
    page.wait_for_timeout(8_000)
    result = {"adapter": page.adapter(), "paths": page.render_paths(), "canvases": page.evaluate(READ_CANVASES),
              "text": page.inner_text("body"), "errors": page.errors}
    page.screenshot(path=str(screenshot))
    return result


@pytest.fixture(scope="module")
def exports(browser_out) -> dict[str, str]:
    from quantem.widget import Show2D, Show3D, Show4DSTEM
    from quantem.widget.datasets import show2d_gold, show3d_gold, show4dstem_gold

    from quantem.widget import Show3DSlices

    gold = show2d_gold(size="small")
    Show2D(gold, cmap="inferno", show_fft=True).export_html(browser_out / "show2d.html")
    images = [gold.array, np.rot90(gold.array).copy(), gold.array[::-1].copy()]
    Show2D(images, show_fft=True, link_contrast=False).export_html(browser_out / "show2d_gallery.html")
    stack = show3d_gold(size="small")
    Show3D(stack, show_fft=True, fft_layout="right").export_html(browser_out / "show3d.html")
    frames = np.asarray(stack.array)
    Show3D(frames, frames[:, ::-1].copy(), panel_titles=["gold", "flipped"]).export_html(browser_out / "show3d_panels.html")
    Show3DSlices(np.asarray(stack.array)).export_html(browser_out / "show3dslices.html")
    (browser_out / "show4dstem").mkdir(exist_ok=True)
    Show4DSTEM(show4dstem_gold(size="medium"), dp_scale_mode="linear", precompute_virtual_images=False).export_html(
        browser_out / "show4dstem" / "show4dstem.html", dtype="uint16")
    np.save(browser_out / "show2d_image.npy", np.asarray(gold.array, dtype=np.float32))
    return {"show2d": "show2d.html", "show2d_gallery": "show2d_gallery.html", "show3d": "show3d.html",
            "show3d_panels": "show3d_panels.html", "show3dslices": "show3dslices.html",
            "show4dstem": "show4dstem/show4dstem.html"}


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
@pytest.mark.parametrize("name", ["show2d", "show2d_gallery", "show3d", "show3dslices"])
def test_export_shows_its_data_and_responds(name, webgpu, exports, serve, open_page, hardware_adapter, browser_out):
    url = f"{serve(browser_out)}/{exports[name]}"
    page = open_page(url, webgpu=webgpu)
    page.wait_for_selector("canvas", timeout=60_000)
    page.wait_for_timeout(8_000)
    tag = f"{name}_{'webgpu' if webgpu else 'cpu'}"
    if webgpu:
        hardware_adapter(page.adapter())
    else:
        assert page.adapter() is None
    before = page.evaluate(APPEARANCE)
    page.screenshot(path=str(browser_out / f"appearance_{tag}.png"), full_page=True)
    assert page.errors == [], page.errors
    assert_shows_data(page, before, tag)
    if name == "show2d":
        # The image histogram follows the data: its bars against a numpy histogram of the same pixels.
        image = np.load(browser_out / "show2d_image.npy")
        reference = np.histogram(image, bins=64, range=(float(image.min()), float(image.max())))[0].astype(float)
        bars = histogram_bars(next(c for c in before if c["histogram"])["columns"])
        correlation = float(np.corrcoef(bars, reference)[0, 1])
        assert correlation > 0.9, correlation
    page.evaluate(MOVE_FIRST_HISTOGRAM_MAX)
    page.wait_for_timeout(2_000)
    after = page.evaluate(APPEARANCE)
    page.screenshot(path=str(browser_out / f"appearance_{tag}_after.png"), full_page=True)
    changed = [b["owner"] for b, a in zip(before, after) if b["owner"] and b["checksum"] != a["checksum"]]
    assert changed, (tag, "no panel repainted after the histogram handle moved")
    summary = {"panels": [(c["owner"], round(c["std"], 1), c["colors"]) for c in before if c["owner"]],
               "histogram_bar_columns": [sum(1 for bar in c["columns"] if bar) for c in before if c["histogram"]],
               "repainted_after_handle": changed}
    (browser_out / f"appearance_{tag}.json").write_text(json.dumps(summary, indent=1))


# Counts 2D draws per scientific canvas from now on (the prototypes are read at call time).
COUNT_DRAWS = """
() => {
  window.__draws = {};
  for (const name of ['drawImage', 'putImageData', 'clearRect']) {
    const original = CanvasRenderingContext2D.prototype[name];
    CanvasRenderingContext2D.prototype[name] = function (...args) {
      const owner = this.canvas && this.canvas.dataset ? this.canvas.dataset.quantemScientificOutput : null;
      if (owner) window.__draws[owner] = (window.__draws[owner] || 0) + 1;
      return original.apply(this, args);
    };
  }
}
"""
# Center of the max handle of the n-th histogram slider, scrolled into view.
HISTOGRAM_MAX_HANDLE = """
(n) => {
  const root = Array.from(document.querySelectorAll('.MuiSlider-root'))
    .filter(r => r.querySelector('input[aria-label="Histogram intensity clip range"]'))[n];
  root.scrollIntoView({block: 'center'});
  const box = root.querySelectorAll('.MuiSlider-thumb')[1].getBoundingClientRect();
  return [box.left + box.width / 2, box.top + box.height / 2];
}
"""


@pytest.mark.parametrize("webgpu", [True, False], ids=["webgpu", "cpu"])
def test_unlinked_gallery_fft_contrast_drag_repaints_only_its_panel(webgpu, exports, serve, open_page, hardware_adapter, browser_out):
    """Dragging one FFT histogram of an unlinked gallery redraws that FFT panel and no other."""
    page = open_page(f"{serve(browser_out)}/{exports['show2d_gallery']}", webgpu=webgpu)
    page.wait_for_selector('[data-quantem-scientific-output="show2d-fft-2"]', timeout=60_000)
    page.wait_for_timeout(8_000)
    if webgpu:
        hardware_adapter(page.adapter())
    before = page.evaluate(APPEARANCE)
    page.evaluate(COUNT_DRAWS)
    # three image histograms, then one FFT histogram per panel
    x, y = page.evaluate(HISTOGRAM_MAX_HANDLE, 3)
    page.wait_for_timeout(500)
    for drag in range(2):  # the first drag and a later one
        page.mouse.move(x, y)
        page.mouse.down()
        for step in range(1, 9):
            page.mouse.move(x - 4 * step, y)
            page.wait_for_timeout(30)
        page.mouse.up()
        page.wait_for_timeout(1_500)
        x, y = page.evaluate(HISTOGRAM_MAX_HANDLE, 3)
    draws = page.evaluate("() => window.__draws")
    after = page.evaluate(APPEARANCE)
    page.screenshot(path=str(browser_out / f"gallery_fft_drag_{'webgpu' if webgpu else 'cpu'}.png"), full_page=True)
    assert page.errors == [], page.errors
    checksum = {canvas["owner"]: (b["checksum"], canvas["checksum"]) for b, canvas in zip(before, after) if canvas["owner"]}
    assert draws.get("show2d-fft-0", 0) > 0 and checksum["show2d-fft-0"][0] != checksum["show2d-fft-0"][1], (draws, checksum)
    assert draws.get("show2d-fft-1", 0) == 0 and draws.get("show2d-fft-2", 0) == 0, draws
    assert checksum["show2d-fft-1"][0] == checksum["show2d-fft-1"][1] and checksum["show2d-fft-2"][0] == checksum["show2d-fft-2"][1]


@pytest.mark.parametrize("name", ["show2d", "show3d", "show3d_panels"])
def test_export_pixels_match_with_and_without_webgpu(name, exports, serve, open_page, hardware_adapter, browser_out):
    url = f"{serve(browser_out)}/{exports[name]}"
    gpu = render(open_page, url, browser_out / f"{name}_webgpu.png", webgpu=True)
    cpu = render(open_page, url, browser_out / f"{name}_cpu.png", webgpu=False)
    hardware_adapter(gpu["adapter"])
    assert cpu["adapter"] is None
    assert gpu["errors"] == [] and cpu["errors"] == [], (gpu["errors"], cpu["errors"])
    assert gpu["paths"] == ["WebGPU"] and cpu["paths"] == ["CPU"], (gpu["paths"], cpu["paths"])
    assert len(gpu["canvases"]) == len(cpu["canvases"]) > 0
    differences = []
    for gpu_canvas, cpu_canvas in zip(gpu["canvases"], cpu["canvases"]):
        assert (gpu_canvas["width"], gpu_canvas["height"]) == (cpu_canvas["width"], cpu_canvas["height"])
        gpu_pixels = np.asarray(gpu_canvas["pixels"], dtype=np.int16)
        cpu_pixels = np.asarray(cpu_canvas["pixels"], dtype=np.int16)
        differences.append(int(np.abs(gpu_pixels - cpu_pixels).max()))
    (browser_out / f"{name}_parity.json").write_text(json.dumps({"adapter": gpu["adapter"], "max_difference": differences}))
    assert max(differences) <= 1, differences


def test_show4dstem_export_needs_webgpu_and_says_so(exports, serve, open_page, hardware_adapter, browser_out):
    url = f"{serve(browser_out)}/{exports['show4dstem']}"
    gpu = render(open_page, url, browser_out / "show4dstem_webgpu.png", webgpu=True)
    cpu = render(open_page, url, browser_out / "show4dstem_cpu.png", webgpu=False)
    hardware_adapter(gpu["adapter"])
    assert gpu["paths"] == ["WebGPU"] and cpu["paths"] == ["CPU"]
    assert "Needs WebGPU" not in gpu["text"]
    assert "Needs WebGPU: this page computes virtual images" in cpu["text"]
    # The saved diffraction pattern and virtual image stay drawn under the notice.
    saved = [canvas for canvas in cpu["canvases"] if canvas["width"] == canvas["height"] and canvas["width"] in (24, 64)]
    assert any(np.asarray(canvas["pixels"]).reshape(-1, 4)[:, :3].std() > 0 for canvas in saved)
