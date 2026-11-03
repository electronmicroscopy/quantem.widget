"""Show2D tools driven like a user, on exported pages with WebGPU on.

An unlinked gallery histogram is computed once per panel frame, not on every
hover. A gallery FFT of a panel above 2048 px says it is binned and measures
its lattice band at the binned sampling. A profile drag samples each panel at
most once per pointer move. The frequency-filter ring sits on the frequency
the filter cuts at the FFT's zoom, and dragging it sets that frequency. A ROI
edge moves by as much as the pointer moves after the press. Overlay editing
lets pointers through the More menu's backdrop only, so every other menu still
closes on an outside click. The SVG export draws what the screen shows. The
lens shows the image's colors with Log on, signed data included. Every switch
has an accessible name.
"""

import re

import numpy as np
import pytest

pytestmark = pytest.mark.browser

COUNT_LOG1P = """
window.__log1pCalls = 0;
const log1p = Math.log1p;
Math.log1p = (value) => { window.__log1pCalls += 1; return log1p(value); };
"""
# One entry per histogram (its range slider has two inputs): the min and max labels.
# One profile sample is one compute pass of the line-profile kernel (its pipeline is built once and reused).
COUNT_PROFILE_SAMPLES = """
window.__profileSamples = 0;
const profileModules = new WeakSet(), profilePipelines = new WeakSet();
const createShaderModule = GPUDevice.prototype.createShaderModule;
GPUDevice.prototype.createShaderModule = function (descriptor) {
  const module = createShaderModule.call(this, descriptor);
  if (descriptor.code.includes('profile_width: u32')) profileModules.add(module);
  return module;
};
const createComputePipeline = GPUDevice.prototype.createComputePipeline;
GPUDevice.prototype.createComputePipeline = function (descriptor) {
  const pipeline = createComputePipeline.call(this, descriptor);
  if (profileModules.has(descriptor.compute.module)) profilePipelines.add(pipeline);
  return pipeline;
};
const setPipeline = GPUComputePassEncoder.prototype.setPipeline;
GPUComputePassEncoder.prototype.setPipeline = function (pipeline) {
  if (profilePipelines.has(pipeline)) window.__profileSamples += 1;
  return setPipeline.call(this, pipeline);
};
"""
# The value of the ROI slider that follows a label ("Size", "W", "H") in the selected-ROI row.
ROI_SLIDER = """
(label) => {
  const text = Array.from(document.querySelectorAll('p, span')).find(element => element.textContent.trim() === label);
  return Number(text.nextElementSibling.querySelector('input[type="range"]').value);
}
"""
HISTOGRAM_LABELS = """
() => Array.from(new Set(Array.from(document.querySelectorAll('input[aria-label="Histogram intensity clip range"]'))
  .map(input => input.closest('.MuiBox-root').parentElement)))
  .map(histogram => histogram.innerText.split('\\n').filter(Boolean).slice(0, 2))
"""


def open_export(page_name, serve, open_page, hardware_adapter, browser_out, ready, init_script=None):
    page = open_page(f"{serve(browser_out)}/{page_name}", webgpu=True)
    if init_script:
        page.add_init_script(init_script)
        page.reload()
    page.wait_for_selector(ready, timeout=120_000)
    page.wait_for_timeout(5_000)
    hardware_adapter(page.adapter())
    return page


def test_unlinked_gallery_hover_reuses_each_panel_histogram(serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show2D

    rng = np.random.default_rng(2)
    panels = [(rng.standard_normal((256, 256)) * 40 * (k + 1)).astype(np.float32) for k in range(4)]
    Show2D(panels, link_contrast=False, log_scale=True).export_html(browser_out / "show2d_unlinked_log.html")
    page = open_export("show2d_unlinked_log.html", serve, open_page, hardware_adapter, browser_out,
                       'canvas[data-quantem-scientific-output="show2d-image-3"]', COUNT_LOG1P)
    box = page.locator('canvas[data-quantem-scientific-output="show2d-image-0"]').bounding_box()
    page.evaluate("() => { window.__log1pCalls = 0; }")
    for step in range(8):
        page.mouse.move(box["x"] + box["width"] * (0.2 + 0.08 * step), box["y"] + box["height"] * 0.5)
        page.wait_for_timeout(80)
    page.screenshot(path=str(browser_out / "show2d_unlinked_log_hover.png"))
    # A hover changes no pixel: it must not log-transform any panel again.
    assert page.evaluate("() => window.__log1pCalls") == 0
    # Each histogram spans its own panel's signed log1p range (float32, as the browser stores it).
    expected = []
    for panel in panels:
        logged = (np.sign(panel) * np.log1p(np.abs(panel).astype(np.float64))).astype(np.float32)
        expected.append([f"{float(logged.min()):.1f}", f"{float(logged.max()):.1f}"])
    assert page.evaluate(HISTOGRAM_LABELS) == expected
    assert page.errors == [], page.errors


def test_gallery_fft_of_a_binned_panel_says_so_and_measures_at_the_binned_sampling(serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show2D

    rows, cols = np.indices((4096, 4096))
    # A 6 px lattice at 0.25 A/px: a 1.5 A period, inside the 1-6 A band the
    # FFT metrics count peaks in. On the 2x binned FFT (0.5 A/px) the band is
    # only right if the metrics use the binned sampling.
    lattice = (np.cos(2 * np.pi * cols / 6.0) + np.cos(2 * np.pi * rows / 6.0)).astype(np.float32)
    Show2D([lattice, lattice], sampling=0.25, units="Å", show_fft=True).export_html(
        browser_out / "show2d_fft_binned.html", encoding="uint8", max_mb=None)
    page = open_export("show2d_fft_binned.html", serve, open_page, hardware_adapter, browser_out,
                       '.quantem-fft-quality-label')
    page.wait_for_timeout(3_000)
    page.screenshot(path=str(browser_out / "show2d_fft_binned.png"))
    labels = page.locator(".quantem-fft-quality-label").all_inner_texts()
    peaks = [int(re.search(r"peaks (\d+)", label).group(1)) for label in labels]
    # +-g along rows and columns: four reflections in the band.
    assert peaks == [4, 4], labels
    binning = page.locator(".quantem-fft-binning-label")
    assert binning.all_inner_texts() == ["FFT preview, binned 2x"] * 2
    assert page.errors == [], page.errors


def test_profile_drag_samples_each_panel_at_most_once_per_move(serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show2D

    rows, cols = np.indices((1024, 1024))
    ramp = (cols + 0.5 * rows).astype(np.float32)
    widget = Show2D([ramp, ramp[::-1].copy()])
    widget.profile_line = [{"row": 200.0, "col": 100.0}, {"row": 200.0, "col": 800.0}]
    widget.export_html(browser_out / "show2d_profile_drag.html", max_mb=None)
    page = open_export("show2d_profile_drag.html", serve, open_page, hardware_adapter, browser_out,
                       'canvas[data-quantem-scientific-output="show2d-image-1"]', COUNT_PROFILE_SAMPLES)
    box = page.locator('canvas[data-quantem-scientific-output="show2d-image-0"]').bounding_box()
    scale = box["width"] / 1024
    x, y = box["x"] + 800 * scale, box["y"] + 200 * scale
    page.mouse.move(x, y)
    page.evaluate("() => { window.__profileSamples = 0; }")
    page.mouse.down()
    moves = 20
    for step in range(1, moves + 1):
        page.mouse.move(x - 10 * step * scale, y + 8 * step * scale)
        page.wait_for_timeout(16)
    page.mouse.up()
    page.wait_for_timeout(1_500)
    page.screenshot(path=str(browser_out / "show2d_profile_drag.png"))
    samples = page.evaluate("() => window.__profileSamples")
    # Two panels: never more than one sample per panel per move, and the last line is sampled.
    assert 2 <= samples <= 2 * moves, samples
    assert page.errors == [], page.errors


def test_frequency_ring_marks_and_sets_the_cutoff_on_the_zoomed_fft(serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show2D

    noise = np.random.default_rng(3).standard_normal((512, 512)).astype(np.float32)
    widget = Show2D(noise, show_fft=True)
    widget.frequency_filter_enabled = True
    widget.frequency_filter_modes = ["lowpass"]
    widget.frequency_filter_cutoffs = [0.25]
    widget.export_html(browser_out / "show2d_frequency_ring.html")
    page = open_export("show2d_frequency_ring.html", serve, open_page, hardware_adapter, browser_out,
                       ".quantem-frequency-filter-ring")
    fft = page.locator('canvas[data-quantem-scientific-output="show2d-fft-0"]').bounding_box()
    ring = page.locator(".quantem-frequency-filter-ring")
    box = ring.bounding_box()
    center_x, center_y = fft["x"] + fft["width"] / 2, fft["y"] + fft["height"] / 2
    # The FFT opens at 2x zoom: Nyquist is a full panel width from the center,
    # so a 0.25 Nyquist cutoff is a ring half the panel wide.
    assert box["width"] == pytest.approx(0.5 * fft["width"], abs=2)
    assert box["x"] + box["width"] / 2 == pytest.approx(center_x, abs=1)
    page.mouse.move(box["x"] + box["width"] - 1, center_y)
    page.mouse.down()
    for step in range(1, 6):
        page.mouse.move(box["x"] + box["width"] - 1 + step * (center_x + 0.3 * fft["width"] - box["x"] - box["width"] + 1) / 5, center_y)
        page.wait_for_timeout(60)
    page.mouse.up()
    page.wait_for_timeout(1_500)
    page.screenshot(path=str(browser_out / "show2d_frequency_ring.png"))
    label = ring.get_attribute("aria-label")
    cutoff = float(re.search(r"at ([\d.]+) Nyq", label).group(1))
    # Released 0.3 panel widths from the center at 2x zoom: 0.3 Nyquist.
    assert cutoff == pytest.approx(0.3, abs=0.005), label
    assert ring.bounding_box()["width"] == pytest.approx(0.6 * fft["width"], abs=2)
    assert page.errors == [], page.errors


def test_roi_resize_follows_the_pointer_from_the_press(serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show2D

    widget = Show2D(np.random.default_rng(4).standard_normal((512, 512)).astype(np.float32))
    widget.roi_active = True
    roi = {"radius_inner": 20, "color": "#4fc3f7", "line_width": 2, "highlight": False}
    widget.roi_list = [
        {**roi, "row": 256, "col": 150, "shape": "circle", "radius": 50, "width": 100, "height": 100},
        {**roi, "row": 256, "col": 380, "shape": "rectangle", "radius": 50, "width": 100, "height": 60},
    ]
    widget.roi_selected_idx = 0
    widget.export_html(browser_out / "show2d_roi_resize.html")
    page = open_export("show2d_roi_resize.html", serve, open_page, hardware_adapter, browser_out,
                       'canvas[data-quantem-scientific-output="show2d-image-0"]')
    box = page.locator('canvas[data-quantem-scientific-output="show2d-image-0"]').bounding_box()
    scale = box["width"] / 512

    def drag(row, cols):
        page.mouse.move(box["x"] + cols[0] * scale, box["y"] + row * scale)
        page.mouse.down()
        for col in cols[1:]:
            page.mouse.move(box["x"] + col * scale, box["y"] + row * scale)
            page.wait_for_timeout(60)
        page.mouse.up()
        page.wait_for_timeout(300)

    # Press 8 px outside the circle's edge, then move 1 px and 20 px outward:
    # the radius grows by 20, it does not jump to the press distance first.
    drag(256, [208, 209, 228])
    # (the pointer lands on whole screen pixels, about 1 image pixel each)
    assert page.evaluate(ROI_SLIDER, "Size") == pytest.approx(70, abs=1)
    # The rectangle's right edge, pressed at mid height and moved 10 px right:
    # the width grows by 20 and the height stays.
    drag(256, [433, 438, 443])
    page.screenshot(path=str(browser_out / "show2d_roi_resize.png"))
    assert page.evaluate(ROI_SLIDER, "W") == pytest.approx(120, abs=2)
    assert page.evaluate(ROI_SLIDER, "H") == 60
    assert page.errors == [], page.errors


def test_overlay_edit_leaves_other_menus_closable(serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show2D

    image = np.random.default_rng(5).standard_normal((256, 256)).astype(np.float32)
    Show2D(image, panel_overlays=[{"shape": "circle", "center": (0.5, 0.5), "radius": 0.1, "coords": "relative"}]).export_html(
        browser_out / "show2d_overlay_edit.html")
    page = open_export("show2d_overlay_edit.html", serve, open_page, hardware_adapter, browser_out,
                       'canvas[data-quantem-scientific-output="show2d-image-0"]')
    page.get_by_role("button", name="More tools").click()
    page.get_by_role("menuitem").filter(has_text="Overlay Edit").dispatch_event("mousedown")
    page.wait_for_timeout(500)
    page.get_by_role("combobox").filter(has_text="Inferno").click()
    page.wait_for_selector('[role="listbox"]')
    page.mouse.click(1500, 1000)
    page.wait_for_timeout(500)
    page.screenshot(path=str(browser_out / "show2d_overlay_edit.png"))
    assert page.locator('[role="listbox"]').count() == 0, "the colormap menu stayed open after an outside click"
    # The More menu itself keeps letting pointers through to the image while editing.
    page.get_by_role("button", name="More tools").click()
    backdrop = page.locator("#show2d-more-menu .MuiBackdrop-root")
    assert backdrop.evaluate("element => getComputedStyle(element).pointerEvents") == "none"
    assert page.errors == [], page.errors


def test_svg_export_draws_what_the_screen_shows(serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show2D

    rows, cols = np.indices((256, 256))
    image = (np.sin(cols / 9.0) * np.cos(rows / 13.0)).astype(np.float32)
    inset = {"x": [0, 2, 4, 6, 8, 10], "y": [0, 0.4, 0.9, 1, 0.6, 0.2], "show_ticks": True, "xticks": [0, 5, 10],
             "xlabel": "q (1/nm)", "annotation": "fit", "position": "top-right"}
    Show2D(image, panel_annotations=[[{"text": "scan 3"}]], inset_plots=[inset]).export_html(browser_out / "show2d_svg.html")
    page = open_export("show2d_svg.html", serve, open_page, hardware_adapter, browser_out,
                       'canvas[data-quantem-scientific-output="show2d-image-0"]', "delete window.showSaveFilePicker;")
    page.get_by_role("button", name="Export").first.click()
    with page.expect_download() as download:
        page.get_by_role("menuitem", name="SVG", exact=True).click()
    svg = open(download.value.path()).read()
    (browser_out / "show2d_export.svg").write_text(svg)
    # The annotation badge keeps its 0.72 alpha; the inset keeps its ticks, axis label and annotation.
    assert 'fill="rgb(0, 0, 0)" fill-opacity="0.72"' in svg
    assert all(f">{text}</text>" in svg for text in ("0", "5", "10", "q (1/nm)", "fit"))
    # A single image has no panel title and, without marker_colors, no panel marker on screen.
    assert ">Image 1<" not in svg
    assert 'width="5"' not in svg
    assert page.errors == [], page.errors


def test_lens_shows_the_image_colors_of_signed_data_in_log_scale(serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show2D

    # Columns ramp from -50 to 50: half the image is below -1, where an unsigned log1p has no value.
    ramp = np.tile(np.linspace(-50, 50, 250, dtype=np.float32), (250, 1))
    Show2D(ramp, log_scale=True, cmap="viridis").export_html(browser_out / "show2d_lens_log.html")
    page = open_export("show2d_lens_log.html", serve, open_page, hardware_adapter, browser_out,
                       'canvas[data-quantem-scientific-output="show2d-image-0"]')
    page.evaluate("""() => Array.from(document.querySelectorAll('p, span')).find(element => element.textContent.trim() === 'Lens')
      .nextElementSibling.querySelector('input').click()""")
    image = page.locator('canvas[data-quantem-scientific-output="show2d-image-0"]')
    box = image.bounding_box()
    scale = box["width"] / 250
    page.mouse.move(box["x"] + 50.5 * scale, box["y"] + 125.5 * scale)
    page.wait_for_timeout(500)
    page.screenshot(path=str(browser_out / "show2d_lens_log.png"))
    # 4x lens, 128 px inset at its default bottom-left corner: 10 px right of and below
    # its center (clear of the crosshair) is image pixel (127, 52).
    lens, main = page.evaluate("""() => {
      const image = document.querySelector('canvas[data-quantem-scientific-output="show2d-image-0"]');
      const lens = image.parentElement.querySelectorAll('canvas')[image.parentElement.querySelectorAll('canvas').length - 1];
      const cssHeight = image.getBoundingClientRect().height;
      const lensPixel = lens.getContext('2d').getImageData(12 + 64 + 10, cssHeight - 128 - 12 - 20 + 64 + 10, 1, 1).data;
      const scale = image.width / 250;
      const mainPixel = image.getContext('2d').getImageData(Math.floor(52.5 * scale), Math.floor(127.5 * scale), 1, 1).data;
      return [Array.from(lensPixel.slice(0, 3)), Array.from(mainPixel.slice(0, 3))];
    }""")
    assert np.abs(np.subtract(lens, main)).max() <= 2, (lens, main)
    assert page.errors == [], page.errors


def test_every_switch_has_an_accessible_name(serve, open_page, hardware_adapter, browser_out):
    from quantem.widget import Show2D

    images = np.random.default_rng(6).standard_normal((2, 128, 128)).astype(np.float32)
    Show2D(images, show_fft=True, link_contrast=False).export_html(browser_out / "show2d_switches.html")
    Show2D(images[0], show_fft=True).export_html(browser_out / "show2d_switches_single.html")
    unnamed = {}
    for name in ("show2d_switches.html", "show2d_switches_single.html"):
        page = open_export(name, serve, open_page, hardware_adapter, browser_out,
                           'canvas[data-quantem-scientific-output="show2d-fft-0"]')
        page.get_by_role("button", name="More tools").click()
        page.wait_for_timeout(300)
        switches = page.locator(".MuiSwitch-input")
        assert switches.count() > 10
        unnamed[name] = [index for index in range(switches.count()) if not switches.nth(index).get_attribute("aria-label")]
        assert page.errors == [], page.errors
    assert unnamed == {"show2d_switches.html": [], "show2d_switches_single.html": []}
