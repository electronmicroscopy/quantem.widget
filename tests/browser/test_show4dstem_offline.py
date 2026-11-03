"""Show4DSTEM exported pages driven like a user on the hardware WebGPU adapter.

Small synthetic scans are exported once per module: a bright-field disk whose
intensity carries fringes and whose center follows the scan position, so the
virtual images, the DPC sources and the FFT all hold structure. Each test opens
one page, acts on it through the mouse and keyboard, and checks what the screen
shows (or the canvas a test or screenshot tool reads) against the data.
"""

import io
import re

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.browser

DP = 'canvas[data-quantem-scientific-output="show4dstem-diffraction-pattern"]'
VI = 'canvas[data-quantem-scientific-output="show4dstem-virtual-image"]'
FFT = 'canvas[data-quantem-scientific-output="show4dstem-fft"]'
TILE = '[aria-label="Show4DSTEM multiple panel 1"]'
# Every string drawn on a canvas, so labels painted into overlays can be read back.
RECORD_CANVAS_TEXT = """
window.__canvasText = [];
const fillText = CanvasRenderingContext2D.prototype.fillText;
CanvasRenderingContext2D.prototype.fillText = function (text, ...rest) {
  window.__canvasText.push(String(text));
  return fillText.call(this, text, ...rest);
};
"""

# A WebGPU canvas cannot also supply a 2D context. Retain the configured device
# only in the test page so visibility probes paint the real presentation canvas.
RECORD_GPU_CANVAS = """
window.__gpuCanvasConfigurations = new WeakMap();
const configure = GPUCanvasContext.prototype.configure;
GPUCanvasContext.prototype.configure = function (configuration) {
  window.__gpuCanvasConfigurations.set(this.canvas, configuration);
  return configure.call(this, configuration);
};
"""


def paint_visibility_probe(page, selector: str, fraction: float) -> None:
    """Paint green into the actual output canvas, using its existing backend."""
    page.evaluate("""async ({selector, fraction}) => {
      const canvas = document.querySelector(selector);
      const configuration = window.__gpuCanvasConfigurations.get(canvas);
      if (!configuration) {
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Output canvas has no recorded rendering context');
        context.fillStyle = '#00ff00';
        context.fillRect(0, 0, canvas.width * fraction, canvas.height * fraction);
        return;
      }
      const {device, format} = configuration;
      const shader = device.createShaderModule({code: `
        @vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
          let positions = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
          return vec4f(positions[i], 0, 1);
        }
        @fragment fn fragment() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
      `});
      const pipeline = device.createRenderPipeline({layout: 'auto',
        vertex: {module: shader, entryPoint: 'vertex'},
        fragment: {module: shader, entryPoint: 'fragment', targets: [{format}]}});
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({colorAttachments: [{
        view: canvas.getContext('webgpu').getCurrentTexture().createView(),
        loadOp: 'load', storeOp: 'store'}]});
      pass.setPipeline(pipeline);
      pass.setScissorRect(0, 0, Math.floor(canvas.width * fraction), Math.floor(canvas.height * fraction));
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }""", {"selector": selector, "fraction": fraction})


def counts(frames: int, scan: int, det: int, seed: int = 0) -> np.ndarray:
    """Poisson counts (frames, scan, scan, det, det): a disk shifted with the scan position over a weak background."""
    rng = np.random.default_rng(seed)
    scan_row, scan_col = np.mgrid[0:scan, 0:scan].astype(np.float32)
    det_row, det_col = (np.mgrid[0:det, 0:det].astype(np.float32) - det / 2)
    stacks = []
    for frame in range(frames):
        fringes = 1 + 0.3 * np.sin(2 * np.pi * scan_row / 8 + frame) * np.cos(2 * np.pi * scan_col / 5)
        shift_row = 1.5 * np.sin(2 * np.pi * scan_row / 16)
        shift_col = 1.5 * np.cos(2 * np.pi * scan_col / 16)
        rows = det_row[None, None] - shift_row[..., None, None]
        cols = det_col[None, None] - shift_col[..., None, None]
        disk = (rows ** 2 + cols ** 2) <= (det * 0.2) ** 2
        mean = 40 * fringes[..., None, None] * disk + 2.0
        stacks.append(rng.poisson(mean).astype(np.uint16))
    return np.stack(stacks)


@pytest.fixture(scope="module")
def pages(serve, browser_out) -> dict[str, str]:
    from quantem.widget import Show4DSTEM

    root = browser_out / "show4dstem_offline"
    single = counts(1, 32, 32)[0]
    bright = single.copy()
    bright[16, 16] *= 20  # the pattern shown at load is 20x brighter than the one at scan (0, 0)
    exports = {
        "single": Show4DSTEM(single, show_fft=True),
        "bright": Show4DSTEM(bright),
        "series": Show4DSTEM(counts(3, 16, 16, seed=1), view_mode="multiple"),
        "series_single": Show4DSTEM(counts(3, 16, 16, seed=2)),
    }
    narrow = Show4DSTEM(single)
    narrow.panel_width_px = 300
    exports["narrow"] = narrow
    url = serve(root)
    for name, viewer in exports.items():
        viewer.export_html(root / name / f"{name}.html", dtype="uint16")
    return {name: f"{url}/{name}/{name}.html" for name in exports}


def open_export(open_page, hardware_adapter, url: str, ready: str = DP, init_script: str | None = None):
    page = open_page("about:blank", webgpu=True)
    if init_script:
        page.add_init_script(init_script)
    page.goto(url)
    hardware_adapter(page.adapter())
    page.wait_for_selector(ready, timeout=60_000)
    page.wait_for_timeout(4000)
    return page


def shot(page, selector: str) -> np.ndarray:
    """What the screen shows over an element (a clip, not an element screenshot, which waits for a still frame)."""
    box = page.locator(selector).first.bounding_box()
    return np.asarray(Image.open(io.BytesIO(page.screenshot(clip=box))).convert("RGB"), dtype=np.int16)


def green_outline_pixels(image: np.ndarray) -> int:
    return int(((image[..., 1] > image[..., 0] + 60) & (image[..., 1] > image[..., 2] + 60)).sum())


def choose(page, combobox, option: str) -> None:
    combobox.click()
    page.locator('li[role="option"]', has_text=re.compile(f"^{option}$")).first.click()
    page.wait_for_timeout(1500)


def image_scale_select(page, panel_selector: str):
    """The Lin/Log select under a panel (the first one below it, within its column)."""
    index = page.evaluate("""(selector) => {
      const panel = document.querySelector(selector).getBoundingClientRect();
      const combos = Array.from(document.querySelectorAll('[role="combobox"]'));
      return combos.findIndex(combo => {
        const box = combo.getBoundingClientRect();
        return combo.textContent.trim() === 'Lin' && box.top > panel.bottom && box.left >= panel.left - 2 && box.left < panel.right;
      });
    }""", panel_selector)
    return page.locator('[role="combobox"]').nth(index)


# ---------------------------------------------------------------------------
# Multiple view
# ---------------------------------------------------------------------------


def test_compare_tile_output_is_the_canvas_on_screen(pages, open_page, hardware_adapter):
    page = open_export(open_page, hardware_adapter, pages["series"], ready=TILE,
                       init_script=RECORD_GPU_CANVAS)
    output = '[data-quantem-scientific-output="show4dstem-compare-0"]'
    page.wait_for_selector(output, timeout=30_000)
    # A mark painted into the canvas tests and screenshot tools read must be what the screen shows.
    paint_visibility_probe(page, output, 0.25)
    page.wait_for_timeout(300)
    box = page.locator(output).bounding_box()
    clip = {"x": box["x"] + 0.12 * box["width"], "y": box["y"] + 0.15 * box["height"],
            "width": 0.08 * box["width"], "height": 0.08 * box["height"]}
    region = np.asarray(Image.open(io.BytesIO(page.screenshot(clip=clip))).convert("RGB"), dtype=np.int16)
    marked = (region[..., 1] > 200) & (region[..., 0] < 60) & (region[..., 2] < 60)
    assert marked.mean() > 0.9, marked.mean()


def test_compare_zoom_and_pan_keep_the_painted_tiles(pages, open_page, hardware_adapter):
    page = open_export(open_page, hardware_adapter, pages["series"], ready=TILE,
                       init_script=RECORD_GPU_CANVAS)
    page.wait_for_timeout(2000)
    output = '[data-quantem-scientific-output="show4dstem-compare-0"]'
    paint_visibility_probe(page, output, 1.0)
    box = page.locator(TILE).bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    for _ in range(3):
        page.mouse.wheel(0, -100)
        page.wait_for_timeout(150)
    page.keyboard.down("Shift")
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2 + 30, box["y"] + box["height"] / 2 + 20, steps=6)
    page.mouse.up()
    page.keyboard.up("Shift")
    page.wait_for_timeout(1000)
    zoom = float(page.locator(TILE).get_attribute("data-comparison-zoom"))
    # CSS zoom/pan must preserve the painted canvas on screen, not merely an
    # offscreen byte-copy canvas that the scientist cannot see.
    shown = shot(page, TILE)
    center = shown[shown.shape[0] // 3:2 * shown.shape[0] // 3,
                   shown.shape[1] // 3:2 * shown.shape[1] // 3]
    green = (center[..., 1] > 200) & (center[..., 0] < 60) & (center[..., 2] < 60)
    assert zoom > 1.2 and green.mean() > 0.9, (zoom, green.mean())


def test_detector_outline_survives_a_device_pixel_ratio_change(pages, open_page, hardware_adapter, browser_out):
    page = open_export(open_page, hardware_adapter, pages["series"], ready=TILE)
    assert green_outline_pixels(shot(page, DP)) > 50
    client = page.context.new_cdp_session(page)
    client.send("Emulation.setDeviceMetricsOverride", {"width": 1600, "height": 1050, "deviceScaleFactor": 2, "mobile": False})
    page.wait_for_timeout(500)
    # a hover readout re-renders the widget, which resizes the HiDPI overlay canvases
    box = page.locator(DP).bounding_box()
    page.mouse.move(box["x"] + 10, box["y"] + 10)
    page.wait_for_timeout(300)
    page.mouse.move(5, 300)
    page.wait_for_timeout(1000)
    after = shot(page, DP)
    Image.fromarray(after.astype(np.uint8)).save(browser_out / "show4dstem_offline_dpr2_dp.png")
    assert green_outline_pixels(after) > 50, green_outline_pixels(after)


def test_dpc_log_tiles_keep_negative_values(pages, open_page, hardware_adapter):
    page = open_export(open_page, hardware_adapter, pages["series"], ready=TILE)
    page.locator('[aria-label="Show DPC row virtual detector"]').click()
    page.wait_for_timeout(2000)
    choose(page, image_scale_select(page, TILE), "Log")
    page.wait_for_timeout(1500)
    # The signed log keeps a DPC map's negative half: only its minimum takes the first colormap entry.
    darkest = page.evaluate("""() => {
      const canvas = document.querySelector('[data-quantem-scientific-output="show4dstem-compare-0"]');
      const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      const brightness = Array.from({length: data.length / 4}, (_, i) => data[4 * i] + data[4 * i + 1] + data[4 * i + 2]);
      const darkestValue = Math.min(...brightness);
      return brightness.filter(value => value === darkestValue).length / brightness.length;
    }""")
    assert darkest < 0.05, darkest


# ---------------------------------------------------------------------------
# Single view
# ---------------------------------------------------------------------------


def test_rect_point_and_square_detectors_change_the_virtual_image(pages, open_page, hardware_adapter, browser_out):
    page = open_export(open_page, hardware_adapter, pages["single"])
    page.wait_for_timeout(1500)
    shots = {"bf": shot(page, VI)}
    detector = page.locator('[role="combobox"]', has_text=re.compile("^Circle$")).first
    for mode in ("Rect", "Point", "Square", "Circle"):
        choose(page, detector, mode)
        detector = page.locator('[role="combobox"]', has_text=re.compile(f"^{mode}$")).first
        shots[mode.lower()] = shot(page, VI)
        Image.fromarray(shots[mode.lower()].astype(np.uint8)).save(browser_out / f"show4dstem_offline_detector_{mode.lower()}.png")
    differences = {mode: int(np.abs(shots[mode] - shots["bf"]).max()) for mode in ("rect", "point", "square", "circle")}
    # each shape sums other detector pixels; returning to the BF circle shows the BF image again
    assert min(differences["rect"], differences["point"], differences["square"]) > 20 and differences["circle"] <= 1, differences


def test_fft_click_that_fails_to_refine_ends_the_drag(pages, open_page, hardware_adapter):
    page = open_export(open_page, hardware_adapter, pages["single"], ready=FFT)
    page.wait_for_timeout(1500)
    box = page.locator(FFT).bounding_box()
    # a lost device fails the GPU peak refine: every readback rejects during the click
    page.evaluate("""() => {
      window.__mapAsync = GPUBuffer.prototype.mapAsync;
      GPUBuffer.prototype.mapAsync = () => Promise.reject(new Error('device lost'));
    }""")
    page.mouse.click(box["x"] + box["width"] * 0.7, box["y"] + box["height"] * 0.3)
    page.wait_for_timeout(500)
    page.evaluate("() => { GPUBuffer.prototype.mapAsync = window.__mapAsync; }")
    # read the FFT canvas in the page: a screenshot here would itself fire a pointer leave
    pixels = f"""() => Array.from(document.querySelector('{FFT}').getContext('2d').getImageData(0, 0, 32, 32).data)"""
    before = page.evaluate(pixels)
    page.mouse.move(box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.6, steps=8)
    page.wait_for_timeout(500)
    # with no button held, moving over the FFT must not pan it
    assert page.evaluate(pixels) == before


def test_uncalibrated_scan_labels_read_px(pages, open_page, hardware_adapter):
    page = open_export(open_page, hardware_adapter, pages["single"], ready=FFT, init_script=RECORD_CANVAS_TEXT)
    page.wait_for_timeout(1500)
    box = page.locator(FFT).bounding_box()
    # a click far from the center snaps to a spot there, not to the zero-frequency peak
    page.mouse.click(box["x"] + box["width"] * 0.85, box["y"] + box["height"] * 0.25)
    page.wait_for_timeout(1500)
    drawn = page.evaluate("window.__canvasText")
    readout = page.locator("text=/^d = /").first.inner_text()
    assert not [text for text in drawn if "pixels" in text], drawn
    assert any(text.endswith(" px") for text in drawn), drawn
    assert readout.endswith(" px") and "Å" not in readout, readout


def test_dp_histogram_axis_reads_the_shown_pattern(pages, open_page, hardware_adapter):
    shown = counts(1, 32, 32)[0][16, 16].astype(np.float64) * 20  # the bright page's pattern at load
    page = open_export(open_page, hardware_adapter, pages["bright"])
    page.wait_for_timeout(1500)
    labels = page.evaluate("""() => {
      const histogram = document.querySelector('canvas[aria-label^="Histogram of intensity"]');
      const row = histogram.parentElement.parentElement.lastElementChild;
      return Array.from(row.children).map(label => label.textContent);
    }""")
    low, high = (float(label) for label in labels)
    assert low == pytest.approx(shown.min(), abs=0.05) and high == pytest.approx(shown.max(), rel=0.05), (labels, shown.min(), shown.max())


def test_source_buttons_stay_visible_on_a_narrow_panel(pages, open_page, hardware_adapter):
    page = open_export(open_page, hardware_adapter, pages["narrow"])
    button = page.locator('[aria-label="Show iDPC virtual detector"]')
    button.wait_for(timeout=30_000)
    inside = page.evaluate("""() => {
      const button = document.querySelector('[aria-label="Show iDPC virtual detector"]');
      const bar = button.parentElement.getBoundingClientRect();
      const box = button.getBoundingClientRect();
      const top = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
      return {inBar: box.right <= bar.right + 0.5 && box.bottom <= bar.bottom + 0.5, onTop: button.contains(top)};
    }""")
    assert inside == {"inBar": True, "onTop": True}, inside


def test_space_plays_the_frame_series(pages, open_page, hardware_adapter):
    page = open_export(open_page, hardware_adapter, pages["series_single"])
    box = page.locator(VI).bounding_box()
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    counter = page.locator("text=/^[123]\\/3$/").first
    assert counter.inner_text() == "1/3"
    page.keyboard.press(" ")
    page.wait_for_timeout(1500)
    assert counter.inner_text() != "1/3"
