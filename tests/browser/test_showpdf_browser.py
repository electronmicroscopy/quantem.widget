"""Browser tests of ShowPDF's standalone HTML export.

Each test exports a widget, opens the file in headless Chromium (CHROME_EXECUTABLE, else Google
Chrome) and drives its controls, so the curves are recomputed in the browser. Opt in with
QUANTEM_WIDGET_BROWSER=1. No display or GPU is needed, but the page loads its scripts from CDNs.
"""

from __future__ import annotations

import inspect
import os
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.browser

_diffraction = pytest.importorskip("quantem.diffraction")
_PDF = getattr(_diffraction, "PairDistributionFunction", None)
if _PDF is None or not {"density", "max_iter"} <= set(inspect.signature(_PDF.calculate_Gr).parameters):
    pytest.skip(
        "ShowPDF needs PairDistributionFunction.calculate_Gr(density=..., max_iter=...)",
        allow_module_level=True,
    )
sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

from quantem.core.datastructures.dataset4dstem import Dataset4dstem  # noqa: E402
from quantem.diffraction import PairDistributionFunction  # noqa: E402
from quantem.widget import ShowPDF  # noqa: E402
from quantem.widget.showpdf import _DENSITY_HINT_GR, _WINDOW_HINT  # noqa: E402

DENSITY_FIELD = 'input[aria-label="Number density in atoms per cubic ångström"]'
PLOT_CANVAS = '[data-quantem-scientific-output="showpdf-curves"] canvas'
NAV_CANVAS = '[data-quantem-scientific-output="showpdf-navigation"] canvas'
R_CUT_SLIDER = "r cut (upper end of the low-r correction)"
PARAMETER_SLIDERS = [
    "k fit range (background fit)",
    "k window range",
    "r max",
    "k lowpass filter",
    "k highpass filter",
]


# ---------------------------------------------------------------------------
# JS run in the page
# ---------------------------------------------------------------------------

# Reading the page.

# Text of the status line in the header (empty when nothing is shown).
STATUS_JS = """() => {
  const header = document.querySelector(".showpdf-root").firstElementChild;
  const el = header.children[1].querySelector("[title]");
  return el ? el.getAttribute("title") : "";
}"""

# Whether the element at a page point (x, y) is part of a slider thumb.
ON_THUMB_JS = "([x, y]) => !!document.elementFromPoint(x, y).closest('.MuiSlider-thumb')"

# Fraction of each scan cell (its inner 80 %, at scan zoom 1) that the scan
# image's overlay paints in the translucent band colour rgba(100,200,255,0.2).
BAND_COVER_JS = """([rows, cols]) => {
  const canvas = document.querySelectorAll('[data-quantem-scientific-output="showpdf-navigation"] canvas')[1];
  const { width, height } = canvas;
  const data = canvas.getContext("2d").getImageData(0, 0, width, height).data;
  const sx = width / cols, sy = height / rows;
  return Array.from({ length: rows }, (_, r) => Array.from({ length: cols }, (_, c) => {
    let n = 0, hit = 0;
    for (let y = Math.ceil((r + 0.1) * sy); y < (r + 0.9) * sy; y++) {
      for (let x = Math.ceil((c + 0.1) * sx); x < (c + 0.9) * sx; x++) {
        const i = (y * width + x) * 4;
        n++;
        if (Math.abs(data[i + 3] - 51) <= 2 && Math.abs(data[i] - 100) <= 8
            && Math.abs(data[i + 1] - 200) <= 8 && data[i + 2] >= 245) hit++;
      }
    }
    return hit / n;
  }));
}"""

# The widget's model: HOOK_MODEL_JS finds it, the snippets after it use it.

# Finds the anywidget model from the @anywidget/react context provider above the element
# `sel`, keeps it as window.__model and counts the g(r) updates from then on (__grUpdates).
HOOK_MODEL_JS = """(sel) => {
  const el = document.querySelector(sel);
  const key = Object.keys(el).find((k) => k.startsWith("__reactFiber"));
  for (let f = el[key]; f; f = f.return) {
    const v = f.memoizedProps && f.memoizedProps.value;
    if (v && v.model && typeof v.model.get === "function") {
      window.__model = v.model;
      window.__grUpdates = 0;
      v.model.on("change:pdf_y_bytes", () => { window.__grUpdates += 1; });
      return true;
    }
  }
  return false;
}"""

# Counts the F(k) and G(r) updates of the page's model (__curveUpdates).
CURVE_UPDATES_JS = """() => {
  window.__curveUpdates = 0;
  for (const t of ["fk_y_bytes", "gr_y_bytes"]) {
    window.__model.on(`change:${t}`, () => { window.__curveUpdates += 1; });
  }
}"""

# Records the values the model takes for a trait from now on (__changes).
CHANGES_JS = """(t) => {
  window.__changes = [];
  window.__model.on(`change:${t}`, () => window.__changes.push(window.__model.get(t)));
}"""

# Sets the given line traits of the page's model, all in one call.
SET_LINE_JS = "(line) => Object.entries(line).forEach(([t, v]) => window.__model.set(t, v))"

# Init scripts, added before the page loads.

# Records the strings drawn on each canvas since its last redraw (a redraw
# starts by setting canvas.width), with their text alignment.
FILLTEXT_JS = """(() => {
  const texts = new Map();
  window.__canvasTexts = () => [...texts.values()].flat().map((d) => d.text);
  window.__canvasDraws = (sel) => (texts.get(document.querySelector(sel)) || []).map((d) => [d.text, d.align]);
  const fillText = CanvasRenderingContext2D.prototype.fillText;
  CanvasRenderingContext2D.prototype.fillText = function (t, ...rest) {
    if (!texts.has(this.canvas)) texts.set(this.canvas, []);
    texts.get(this.canvas).push({ text: String(t), align: this.textAlign });
    return fillText.call(this, t, ...rest);
  };
  const width = Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, "width");
  Object.defineProperty(HTMLCanvasElement.prototype, "width", {
    get() { return width.get.call(this); },
    set(v) { texts.set(this, []); width.set.call(this, v); },
  });
})();"""

# Counts the ImageData buffers created (one per band image built) and the scan
# overlay's redraws in line mode (each draws the start label "A").
OVERLAY_BUILDS_JS = """(() => {
  window.__imageData = 0;
  window.__lineRedraws = 0;
  const create = CanvasRenderingContext2D.prototype.createImageData;
  CanvasRenderingContext2D.prototype.createImageData = function (...args) {
    window.__imageData += 1;
    return create.apply(this, args);
  };
  const fillText = CanvasRenderingContext2D.prototype.fillText;
  CanvasRenderingContext2D.prototype.fillText = function (t, ...rest) {
    if (t === "A") window.__lineRedraws += 1;
    return fillText.call(this, t, ...rest);
  };
})();"""


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

# Browser and page.


def _chrome_executable() -> str | None:
    """CHROME_EXECUTABLE, else the first Chrome/Chromium found in a usual place."""
    candidates = [
        os.environ.get("CHROME_EXECUTABLE"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/opt/google/chrome/chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


@pytest.fixture(scope="module")
def browser():
    chrome = _chrome_executable()
    if chrome is None:
        pytest.skip("Chrome/Chromium executable not found")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, headless=True)
        yield browser
        browser.close()


@pytest.fixture
def open_export(browser):
    """open_export(path, init_script=None, has_touch=False) opens an exported file in a new context
    and waits for the widget; at teardown, any error a page raised fails the test."""
    pages = []

    def open_(path, init_script=None, has_touch=False):
        context = browser.new_context(viewport={"width": 1400, "height": 1100}, has_touch=has_touch)
        page = context.new_page()
        page.errors = []
        page.on("pageerror", lambda error: page.errors.append(str(error)))
        if init_script:
            page.add_init_script(init_script)
        page.goto(Path(path).as_uri())
        page.wait_for_selector(".showpdf-root", timeout=60_000)
        pages.append(page)
        return page

    yield open_
    for page in pages:
        assert page.errors == []
        page.context.close()


def _poll(page, read, done, timeout=5.0):
    """read() every 100 ms until done(value) or the timeout; returns the last value."""
    deadline = time.monotonic() + timeout
    while True:
        value = read()
        if done(value) or time.monotonic() > deadline:
            return value
        page.wait_for_timeout(100)


def _poll_status(page, done):
    """_poll the status line's text."""
    return _poll(page, lambda: page.evaluate(STATUS_JS), done)


# Test data.


def _calibrated_pdf(scan=(4, 4), det=64, dq=0.05, radial_min=0.0):
    """k-calibrated amorphous-like data (dq 1/A per detector pixel), as in tests/showpdf/test_showpdf.py.

    The k axis starts at radial_min detector pixels."""
    rr, cc = np.mgrid[0:det, 0:det].astype(np.float64)
    c = (det - 1) / 2.0
    rad = np.hypot(rr - c, cc - c)
    beam = 400.0 * np.exp(-(rad**2) / (2 * 2.0**2)) + 30.0 * np.exp(-rad / 9.0) + 0.2
    rings = (
        6.0 * np.exp(-((rad - 9.0) ** 2) / 5.0)
        + 2.5 * np.exp(-((rad - 17.0) ** 2) / 9.0)
        + 1.2 * np.exp(-((rad - 25.0) ** 2) / 12.0)
    )
    rng = np.random.default_rng(0)
    data = np.empty((*scan, det, det), dtype=np.float32)
    for iy in range(scan[0]):
        for ix in range(scan[1]):
            amp = 1.0 + 0.15 * (iy + 2 * ix)
            data[iy, ix] = beam + amp * rings + 0.01 * rng.standard_normal((det, det))
    ds = Dataset4dstem.from_array(
        data, sampling=(1.0, 1.0, dq, dq), units=("A", "A", "A^-1", "A^-1")
    )
    return PairDistributionFunction.from_data(
        ds,
        find_origin=False,
        origin_row=c,
        origin_col=c,
        num_annular_bins=90,
        radial_min=radial_min,
        device="cpu",
    )


def _r_grid(w):
    """First and last r of the widget's G(r)."""
    r = np.frombuffer(w.gr_x_bytes, dtype=np.float32)
    return float(r[0]), float(r[-1])


# The page's model (after HOOK_MODEL_JS).


def _model_value(page, trait):
    """A trait of the page's model."""
    return page.evaluate("(t) => window.__model.get(t)", trait)


def _model_set(page, trait, value):
    """Set a trait of the page's model, as Python would."""
    page.evaluate("([t, v]) => window.__model.set(t, v)", [trait, value])


def _poll_model(page, trait, done):
    """_poll a trait of the page's model."""
    return _poll(page, lambda: _model_value(page, trait), done)


# Sliders: MUI renders one input per thumb, labelled with the slider's aria-label.


def _input(page, label, nth=0):
    """The nth input labelled `label` (the nth thumb of a slider)."""
    return page.locator(f'input[aria-label="{label}"]').nth(nth)


def _press(page, label, key, times=1, nth=0):
    """Focus the nth input labelled `label` and press `key` `times` times."""
    _input(page, label, nth).focus()
    for _ in range(times):
        page.keyboard.press(key)


def _slider_value(page, label, nth=0):
    """The value a slider thumb shows."""
    return float(_input(page, label, nth).get_attribute("aria-valuenow"))


def _poll_slider(page, label, done, nth=0):
    """_poll the value a slider thumb shows."""
    return _poll(page, lambda: _slider_value(page, label, nth), done)


def _set_trait_and_wait(page, trait, value, label, nth=0):
    """Set a trait as Python would, wait until its slider shows it, and let the thumb's move finish."""
    _model_set(page, trait, value)
    assert _poll_slider(page, label, lambda v: v == value, nth) == value
    page.wait_for_timeout(300)


def _slider_track(page, label, nth):
    """The slider's track box (MUI maps the pointer's x on it to a value), min and max."""
    thumb = _input(page, label, nth)
    track = thumb.locator("xpath=ancestor::span[contains(@class, 'MuiSlider-root')]").bounding_box()
    return track, float(thumb.get_attribute("min")), float(thumb.get_attribute("max"))


def _x_of_value(track, lo, hi, value):
    """The page x of a slider value on its track."""
    return track["x"] + (value - lo) / (hi - lo) * track["width"]


def _value_at_x(track, lo, hi, x):
    """The slider value under page x, before MUI rounds it to a step."""
    return lo + (x - track["x"]) / track["width"] * (hi - lo)


def _thumb_centre(page, label, nth):
    thumb = _input(page, label, nth)
    box = thumb.locator("xpath=ancestor::span[contains(@class, 'MuiSlider-thumb')]").bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def _value_on_pixel(page, label, near, step=1):
    """The value within 5 steps of `near` whose position on the slider's track is nearest a whole
    pixel (a mouse position), for a slider with less than a pixel per step."""
    track, lo, hi = _slider_track(page, label, 0)

    def off_pixel(value):
        x = _x_of_value(track, lo, hi, value)
        return abs(x - round(x))

    candidates = [round(near + i * step, 10) for i in range(-5, 6)]
    return min(candidates, key=off_pixel)


def _drag_thumb(page, label, nth, value):
    """Mouse drag of a slider thumb to `value`."""
    track, lo, hi = _slider_track(page, label, nth)
    x, y = _thumb_centre(page, label, nth)
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(_x_of_value(track, lo, hi, value), y, steps=10)
    page.mouse.up()


def _drag_away_and_back(page, x, y, away_x, steps):
    """Mouse drag from (x, y) to away_x and back to x, then release."""
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(away_x, y, steps=steps)
    page.mouse.move(x, y, steps=steps)
    page.mouse.up()


def _readout(page, label):
    """The value readout next to a slider."""
    return _input(page, label).locator(
        "xpath=ancestor::span[contains(@class, 'MuiSlider-root')]/following-sibling::p[1]"
    )


# The curve plot (after FILLTEXT_JS).


def _canvas_texts(page):
    """The strings drawn on every canvas since its last redraw."""
    return page.evaluate("() => window.__canvasTexts()")


def _plot_texts(page):
    """The strings drawn in the curve plot's last redraw."""
    return [text for text, _ in page.evaluate("(sel) => window.__canvasDraws(sel)", PLOT_CANVAS)]


def _plot_ticks(page):
    """Tick values of the curve plot's last redraw: (x ticks, y ticks).

    The x tick labels are centred, the y tick labels right aligned; the axis labels
    and messages are not numbers."""
    ticks = {"center": [], "right": []}
    for text, align in page.evaluate("(sel) => window.__canvasDraws(sel)", PLOT_CANVAS):
        try:
            ticks[align].append(float(text))
        except (KeyError, ValueError):
            pass
    return ticks["center"], ticks["right"]


def _wheel_plot(page, notches):
    """Zoom the curve plot in about its centre."""
    box = page.locator(PLOT_CANVAS).bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    for _ in range(notches):
        page.mouse.wheel(0, -100)
        page.wait_for_timeout(50)
    page.mouse.move(5, 5)  # off the plot: no hover readout


# Touch.


def _touch(cdp, kind, points=()):
    """Send one touch event through CDP, which gives every touch the same id (browsers reuse ids too)."""
    touch_points = [{"x": x, "y": y} for x, y in points]
    cdp.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": touch_points})


def _below_widget(page):
    """A page point 100 px below the widget, off it."""
    root = page.locator(".showpdf-root").bounding_box()
    return root["x"] + root["width"] / 2, root["y"] + root["height"] + 100


# Exports shared by several tests.


def _open_density_export(open_export, tmp_path):
    """A g(r) export with density 0.0456789 and its model hooked; returns the page and the field."""
    w = ShowPDF(_calibrated_pdf(), k_min_fit=0.3, density_value=0.0456789, plot_mode="gr")
    page = open_export(w.export_html(tmp_path / "d.html", encoding="uint8"))
    assert page.evaluate(HOOK_MODEL_JS, DENSITY_FIELD)
    field = page.locator(DENSITY_FIELD)
    assert field.input_value() == "0.04568"  # shown rounded to 4 significant figures
    return page, field


def _assert_density_kept(page, field):
    """The density is still the exported 0.0456789, g(r) was not recomputed, no status."""
    page.wait_for_timeout(300)
    assert _model_value(page, "density_value") == 0.0456789
    assert page.evaluate("() => window.__grUpdates") == 0
    assert field.input_value() == "0.04568"
    assert page.evaluate(STATUS_JS) == ""


def _open_window_export(open_export, tmp_path):
    """An F(k) export whose k window ends both follow the fit range [0.3, 1.2]."""
    w = ShowPDF(_calibrated_pdf(), k_min_fit=0.3, k_max_fit=1.2, density_value=0.06, plot_mode="Fk")
    assert (w.k_min_window, w.k_max_window) == (0.0, 0.0)  # both ends follow the fit range
    assert w.k_min_available == 0.0
    return open_export(w.export_html(tmp_path / "win.html", encoding="uint8"))


def _open_slider_export(open_export, tmp_path, label):
    """A uint8 export that shows the slider, its model hooked: damping on (for r cut), in
    probe mode for Probe size, else with a line (for Line width)."""
    w = ShowPDF(
        _calibrated_pdf(scan=(6, 6)),
        k_min_fit=0.3,
        k_max_fit=1.2,
        damp_origin_oscillations=True,
        density_value=0.06,
    )
    if label == "Probe size":
        w.analysis_mode = "probe"
    else:
        w.set_line(0, 0, 5, 5, width=3)
    page = open_export(w.export_html(tmp_path / "set.html", encoding="uint8"))
    assert page.evaluate(HOOK_MODEL_JS, f'input[aria-label="{label}"]')
    return page


# ---------------------------------------------------------------------------
# Status line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ["full", "uint8"])
def test_showpdf_export_shows_baked_error_until_recompute(open_export, tmp_path, encoding):
    w = ShowPDF(_calibrated_pdf(scan=(6, 6)), k_min_fit=0.3)
    w.k_lowpass = 0.02
    w.k_highpass = 0.05  # highpass above lowpass: calculate_Gr refuses
    assert w.status_message.startswith("Error:")
    page = open_export(w.export_html(tmp_path / "err.html", encoding=encoding))
    # The file shows the error Python reported when it was exported ...
    assert _poll_status(page, lambda s: s == w.status_message) == w.status_message
    # ... until the browser recomputes; from then on its own status.
    _press(page, "k highpass filter", "Home")  # highpass off: valid
    assert _poll_status(page, lambda s: s == "") == ""
    page.keyboard.press("End")  # highpass 0.1 above lowpass 0.02
    assert _poll_status(page, lambda s: s != "").startswith("Error: k highpass must not exceed")


def test_showpdf_export_clearing_line_clears_linescan_error(open_export, tmp_path):
    w = ShowPDF(_calibrated_pdf(scan=(6, 6)), k_min_fit=0.3)
    w.set_line(0, 0, 5, 5, width=3)
    w.line_mode = "linescan"
    assert w.status_message == ""
    page = open_export(w.export_html(tmp_path / "ls.html", encoding="uint8"))
    _press(page, "k lowpass filter", "ArrowRight", times=20)  # lowpass 0.02
    _press(page, "k highpass filter", "End")  # highpass 0.1: the line scan is refused
    assert _poll_status(page, lambda s: s != "").startswith("Line-scan error: k highpass must not exceed")
    page.get_by_role("button", name="Clear", exact=True).click()
    # As in Python, where clearing the line drops the line-scan error.
    w.k_lowpass, w.k_highpass = 0.02, 0.1
    assert w.status_message.startswith("Line-scan error:")
    w.clear_line()
    assert _poll_status(page, lambda s: s == w.status_message) == w.status_message == ""


def _failed_from_the_start():
    """No compute succeeded: the widget has no I(k) at all."""
    return ShowPDF(_calibrated_pdf(scan=(6, 6)), k_min_fit=0.3, k_lowpass=0.02, k_highpass=0.05)


def _failed_for_a_new_selection():
    """A compute succeeded for one mask and then failed for another, so Python cleared the curves."""
    w = ShowPDF(_calibrated_pdf(scan=(6, 6)), k_min_fit=0.3, plot_mode="Gr")
    left = np.zeros((6, 6), dtype=bool)
    left[:, :2] = True
    w.set_mask(left)
    w.k_lowpass, w.k_highpass = 0.02, 0.05  # refused: highpass above lowpass
    w.set_mask(~left)  # refused too, for other scan positions
    return w


@pytest.mark.parametrize(
    "widget", [_failed_from_the_start, _failed_for_a_new_selection], ids=["from the start", "for a new selection"]
)
def test_showpdf_export_keeps_baked_error_without_curves_to_recompute(open_export, tmp_path, widget):
    # A curves-only file without the I(k) of the selection it shows cannot recompute (and must
    # not show G(r) of another mask's I(k)): its parameter controls are locked, so the file keeps
    # its exported state and no control contradicts the error.
    w = widget()
    w.damp_origin_oscillations = True  # shows the r cut slider and the density field
    assert w.status_message.startswith("Error:") and len(w.ik_y_bytes) == 0
    page = open_export(w.export_html(tmp_path / "noik.html", encoding="full"))
    assert _poll_status(page, lambda s: s == w.status_message) == w.status_message
    for label in PARAMETER_SLIDERS + [R_CUT_SLIDER, "Damp low-r oscillations"]:
        assert _input(page, label).is_disabled(), label
    assert page.locator(DENSITY_FIELD).is_disabled()
    for mode in ["I(k)", "F(k)", "G(r)", "g(r)"]:
        assert page.get_by_role("button", name=mode, exact=True).is_disabled(), mode
    assert page.evaluate(HOOK_MODEL_JS, DENSITY_FIELD)
    page.locator(".showpdf-root").focus()
    page.keyboard.press("1")  # the I(k) key changes nothing
    _press(page, "k highpass filter", "Home")  # highpass off would be valid: changes nothing
    page.wait_for_timeout(500)
    assert _model_value(page, "plot_mode") == w.plot_mode
    # Nothing was recomputed, so the file still shows why there are no curves.
    assert page.evaluate(STATUS_JS) == w.status_message


def test_showpdf_export_status_says_window_empty(open_export, tmp_path):
    # The lower window end follows the fit range (0.6), above the upper end 0.4.
    w = ShowPDF(_calibrated_pdf(), k_min_fit=0.6, k_max_fit=1.2, k_max_window=0.4, plot_mode="gr")
    assert w.status_message == _WINDOW_HINT  # the browser decides as Python does
    page = open_export(w.export_html(tmp_path / "win.html", encoding="uint8"))
    # The window hint comes before the density hint (g(r) without a density) ...
    assert _poll_status(page, lambda s: s == _WINDOW_HINT) == _WINDOW_HINT
    # ... and returns once an error is gone.
    _press(page, "k lowpass filter", "ArrowRight", times=20)  # lowpass 0.02
    _press(page, "k highpass filter", "End")  # highpass 0.1: refused
    error = _poll_status(page, lambda s: s.startswith("Error:"))
    assert error.startswith("Error: k highpass must not exceed")
    page.keyboard.press("Home")  # highpass off
    assert _poll_status(page, lambda s: s == _WINDOW_HINT) == _WINDOW_HINT
    _press(page, "k fit range (background fit)", "ArrowLeft", times=30)  # fit min and window start 0.3
    assert _poll_status(page, lambda s: s == _DENSITY_HINT_GR) == _DENSITY_HINT_GR
    # Python, in the same state, shows the same hint.
    w.k_lowpass = 0.02
    w.k_min_fit = 0.3
    assert w.status_message == _DENSITY_HINT_GR


# ---------------------------------------------------------------------------
# Density field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["blur", "Enter", "1e"])
def test_showpdf_export_density_field_keeps_density_unless_a_new_one_is_typed(open_export, tmp_path, entry):
    page, field = _open_density_export(open_export, tmp_path)
    field.click()
    if entry == "blur":
        field.blur()
    else:
        if entry != "Enter":  # a malformed number, which reads as an empty field
            page.keyboard.press("ControlOrMeta+A")
            page.keyboard.type(entry)
            assert field.evaluate("el => el.validity.badInput")
        page.keyboard.press("Enter")
    # Not rounded to the shown 0.04568 or unset, g(r) not recomputed, and the field restored.
    _assert_density_kept(page, field)


def test_showpdf_export_density_field_typing_the_shown_value_sets_it(open_export, tmp_path):
    page, field = _open_density_export(open_export, tmp_path)
    field.click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.type("0.04568")  # exactly the rounded value the field shows
    page.keyboard.press("Enter")
    assert _poll_model(page, "density_value", lambda v: v == 0.04568) == 0.04568
    assert page.evaluate("() => window.__grUpdates") >= 1
    assert field.input_value() == "0.04568"


# ---------------------------------------------------------------------------
# g(r) plot
# ---------------------------------------------------------------------------


def test_showpdf_export_empty_gr_plot_says_density_missing(open_export, tmp_path):
    w = ShowPDF(_calibrated_pdf(), k_min_fit=0.3, plot_mode="Ik")  # no density
    page = open_export(w.export_html(tmp_path / "gr.html", encoding="uint8"), init_script=FILLTEXT_JS)
    page.get_by_role("button", name="g(r)", exact=True).click()
    drawn = _poll(page, lambda: _canvas_texts(page), lambda texts: _DENSITY_HINT_GR in texts)
    assert _DENSITY_HINT_GR in drawn
    # On the r grid of G(r) and a neutral g(r) range, not the I(k) axes.
    r0, r1 = _r_grid(w)
    x, y = _plot_ticks(page)
    assert x and r0 - 1e-6 <= min(x) and max(x) <= r1 + 1e-6 and max(x) - min(x) >= 0.75 * (r1 - r0)
    assert y and min(y) == 0 and max(y) == 2


def test_showpdf_export_clearing_density_clears_gr_plot(open_export, tmp_path):
    w = ShowPDF(_calibrated_pdf(), k_min_fit=0.3, density_value=0.06, plot_mode="gr")
    page = open_export(w.export_html(tmp_path / "grc.html", encoding="uint8"), init_script=FILLTEXT_JS)
    _wheel_plot(page, 1)  # zoomed: clearing the density does not refit (and redraw) the plot
    page.locator(DENSITY_FIELD).click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.press("Backspace")
    page.keyboard.press("Enter")
    # The plot is redrawn without the old g(r), with the message, at once.
    drawn = _poll(page, lambda: _canvas_texts(page), lambda texts: _DENSITY_HINT_GR in texts)
    assert _DENSITY_HINT_GR in drawn


# ---------------------------------------------------------------------------
# k window
# ---------------------------------------------------------------------------


def test_showpdf_export_window_end_follows_fit_range(open_export, tmp_path):
    page = _open_window_export(open_export, tmp_path)
    # An end at 0 follows the fit range: the slider shows the window in effect.
    _press(page, "k window range", "Home")  # the lower end to 0, which it already is
    assert _poll_slider(page, "k window range", lambda v: v == pytest.approx(0.3)) == pytest.approx(0.3)
    assert _slider_value(page, "k window range", 1) == pytest.approx(1.2)
    _press(page, "k window range", "ArrowLeft", times=10)  # lower end set to 0.2
    lower = _poll_slider(page, "k window range", lambda v: v == pytest.approx(0.2))
    assert lower == pytest.approx(0.2)
    _press(page, "k fit range (background fit)", "ArrowLeft", times=20, nth=1)  # fit max 1.0
    assert _slider_value(page, "k fit range (background fit)", 1) == pytest.approx(1.0)
    # The upper end was left alone, so it still follows the fit range.
    upper = _poll_slider(page, "k window range", lambda v: v == pytest.approx(1.0), nth=1)
    assert upper == pytest.approx(1.0)
    assert _slider_value(page, "k window range", 0) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "ends, action", [({"k_max_window": 1.546}, "End"), ({"k_max_fit": 1.546}, "drag")], ids=["set", "follows-fit"]
)
def test_showpdf_export_window_end_below_max_moves_to_max(open_export, tmp_path, ends, action):
    # The upper window end is less than a step (0.01) below the slider's max.
    w = ShowPDF(_calibrated_pdf(), k_min_fit=0.3, **ends, density_value=0.06, plot_mode="Fk")
    top = w.k_max_available
    page = open_export(w.export_html(tmp_path / "top.html", encoding="uint8"))
    assert page.evaluate(HOOK_MODEL_JS, 'input[aria-label="k window range"]')
    if action == "drag":
        _drag_thumb(page, "k window range", 1, 1.7)  # past the max
    else:
        _press(page, "k window range", action, nth=1)
    assert _poll_model(page, "k_max_window", lambda v: v == top) == top
    assert _slider_value(page, "k window range", 1) == top


def test_showpdf_export_negative_window_end_shows_fit_range(open_export, tmp_path):
    # Python takes a window end <= 0, as 0, to follow the fit range.
    w = ShowPDF(
        _calibrated_pdf(), k_min_fit=0.3, k_max_fit=1.2, k_min_window=-0.5, density_value=0.06, plot_mode="Fk"
    )
    page = open_export(w.export_html(tmp_path / "neg.html", encoding="uint8"), init_script=FILLTEXT_JS)
    label = "k window range"
    assert page.evaluate(HOOK_MODEL_JS, f'input[aria-label="{label}"]')
    shown = _readout(page, label)
    # The slider, its readout and the F(k) plot's window marker show the window in effect.
    assert _poll(page, shown.inner_text, lambda t: t == "[0.30, 1.20]") == "[0.30, 1.20]"
    assert _slider_value(page, label, 0) == pytest.approx(0.3)
    drawn = _poll(page, lambda: _plot_texts(page), lambda texts: "k window" in texts)
    assert "k window" in drawn
    # So does a drag of the upper end away and back, which changes nothing ...
    x, y = _thumb_centre(page, label, 1)
    _drag_away_and_back(page, x, y, x - 60, steps=5)
    page.wait_for_timeout(300)
    ends = (_model_value(page, "k_min_window"), _model_value(page, "k_max_window"))
    assert (ends, shown.inner_text()) == ((-0.5, 0), "[0.30, 1.20]")
    # ... and a key on the upper end leaves the lower one following the fit range.
    _press(page, label, "ArrowLeft", nth=1)
    assert _poll_model(page, "k_max_window", lambda v: v != 0) == pytest.approx(1.19)
    page.wait_for_timeout(300)
    assert (_model_value(page, "k_min_window"), shown.inner_text()) == (-0.5, "[0.30, 1.19]")


# ---------------------------------------------------------------------------
# Sliders: mouse and keys
# ---------------------------------------------------------------------------

# Every slider that commits to a trait: label, thumb, trait, a value to drag
# it to, and a value Python sets (on the slider's step grid and, for a mouse
# click at its thumb's centre, on the pixel of that value; for Probe size, a
# size near it on a pixel is picked from the layout).
SLIDER_CASES = [
    ("r max", 0, "r_max", 30.0, 12.5),
    ("k fit range (background fit)", 1, "k_max_fit", 1.0, 1.4),
    ("k window range", 1, "k_max_window", 1.0, 1.4),
    ("k lowpass filter", 0, "k_lowpass", 0.05, 0.02),
    ("k highpass filter", 0, "k_highpass", 0.05, 0.02),
    (R_CUT_SLIDER, 0, "r_cut", 3.0, 1.5),
    ("Line width", 0, "line_width", 9, 5),
    ("Probe size", 0, "probe_size", 30, 20),
]


def _slider_cases(*labels):
    """The SLIDER_CASES of the given sliders."""
    return [case for case in SLIDER_CASES if case[0] in labels]


# A range slider's upper thumb, and Probe size, with less than a pixel per step.
@pytest.mark.parametrize(
    ("label", "nth", "trait", "drag_to", "python_value"), _slider_cases("k fit range (background fit)", "Probe size")
)
def test_showpdf_export_slider_click_keeps_value_set_from_python(
    open_export, tmp_path, label, nth, trait, drag_to, python_value
):
    page = _open_slider_export(open_export, tmp_path, label)
    if label == "Probe size":  # less than a pixel per size: not every size has a pixel
        python_value = _value_on_pixel(page, label, python_value)
    exported = _model_value(page, trait)
    _drag_thumb(page, label, nth, drag_to)
    dragged = _poll_model(page, trait, lambda v: v != exported)
    assert dragged not in (exported, python_value)
    # Python sets the trait (here from the page's model) ...
    _set_trait_and_wait(page, trait, python_value, label, nth)
    # ... and a click on the thumb, on the pixel of that value, moves nothing. MUI
    # commits its last onChange value (the drag) also then, which must not undo the change.
    x, y = _thumb_centre(page, label, nth)
    page.mouse.move(round(x), y)  # the whole pixel nearest the thumb's centre
    page.mouse.down()
    pressed = _slider_value(page, label, nth)
    page.mouse.up()
    assert pressed == python_value, (x, _slider_track(page, label, nth))  # the press is on the value
    page.wait_for_timeout(300)
    clicked = _model_value(page, trait)
    # Nor does End at the end of the range, once Python set the trait there: MUI
    # then changes nothing, but commits that value again.
    top = float(_input(page, label, nth).get_attribute("max"))
    _model_set(page, trait, top)
    assert _poll_slider(page, label, lambda v: v == top, nth) == top
    _press(page, label, "End", nth=nth)
    page.wait_for_timeout(300)
    assert (clicked, _model_value(page, trait)) == (python_value, top)


# A range slider's upper thumb, and r cut, with more than one step per pixel.
@pytest.mark.parametrize(
    ("label", "nth", "trait", "drag_to"),
    [case[:4] for case in _slider_cases("k fit range (background fit)", R_CUT_SLIDER)],
)
def test_showpdf_export_slider_commit_without_change_recomputes_nothing(
    open_export, tmp_path, label, nth, trait, drag_to
):
    page = _open_slider_export(open_export, tmp_path, label)
    if label == R_CUT_SLIDER:  # more than one r step per pixel: start from an r cut on a pixel
        step = float(_input(page, label, nth).get_attribute("step"))
        start = _value_on_pixel(page, label, _model_value(page, trait), step)
        _set_trait_and_wait(page, trait, start, label, nth)
    page.evaluate(CURVE_UPDATES_JS)
    exported = _model_value(page, trait)
    # A drag away and back to where it started commits the value the trait has.
    track, lo, hi = _slider_track(page, label, nth)
    x, y = _thumb_centre(page, label, nth)
    x = round(x)  # the whole pixel nearest the thumb's centre (a fractional one is truncated)
    _drag_away_and_back(page, x, y, _x_of_value(track, lo, hi, drag_to), steps=10)
    page.wait_for_timeout(500)
    assert _model_value(page, trait) == exported
    # The file keeps its exact curves (a recompute starts from the 8-bit profiles).
    assert page.evaluate("() => window.__curveUpdates") == 0


@pytest.mark.parametrize(("label", "nth", "trait", "drag_to", "python_value"), SLIDER_CASES)
def test_showpdf_export_slider_rail_click_near_thumb_commits_pointer_value(
    open_export, tmp_path, label, nth, trait, drag_to, python_value
):
    # A click 8 or 15 px from the 10 px thumb's centre, within MUI's 42 px hit circle around
    # the thumb, commits the value under the pointer. Every slider, so each one's commit reaches
    # its trait.
    page = _open_slider_export(open_export, tmp_path, label)
    track, lo, hi = _slider_track(page, label, nth)
    step = float(_input(page, label, nth).get_attribute("step"))
    pixel = (hi - lo) / track["width"]  # the value one pixel spans (a mouse position is a whole pixel)
    for dx in (8, 15):
        _set_trait_and_wait(page, trait, python_value, label, nth)
        x, y = _thumb_centre(page, label, nth)
        x += -dx if python_value > lo + 0.7 * (hi - lo) else dx  # towards the middle
        assert page.evaluate(ON_THUMB_JS, [x, y]), (label, dx)
        page.mouse.click(x, y)
        under = _value_at_x(track, lo, hi, x)
        value = _poll_model(page, trait, lambda v: v != python_value)
        assert abs(value - under) <= step / 2 + pixel + 1e-9, (label, dx, value, under)
        assert _slider_value(page, label, nth) == value, (label, dx)


def test_showpdf_export_small_thumb_drag_commits(open_export, tmp_path):
    # A 2 px drag moves the thumb: a step of this slider is less than 2 px.
    label, trait, start = "Line width", "line_width", 21
    page = _open_slider_export(open_export, tmp_path, label)
    track, lo, hi = _slider_track(page, label, 0)
    step = float(_input(page, label).get_attribute("step"))
    assert step / (hi - lo) * track["width"] < 2  # Line width: about 1.63 px
    _set_trait_and_wait(page, trait, start, label)
    x, y = _thumb_centre(page, label, 0)
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 2, y)
    shown = _slider_value(page, label)
    page.mouse.up()
    assert shown != start
    assert _poll_model(page, trait, lambda v: v == shown) == shown
    page.wait_for_timeout(300)
    assert (_model_value(page, trait), _slider_value(page, label)) == (shown, shown)


@pytest.mark.parametrize(
    ("label", "trait", "value", "readout"),
    [("r max", "r_max", 60.0, "60.0 Å"), ("r max", "r_max", 0.5, "0.5 Å")],
    ids=["above", "below"],
)
def test_showpdf_export_key_towards_slider_bound_keeps_value_beyond_it(
    open_export, tmp_path, label, trait, value, readout
):
    # Python set the trait beyond the slider's fixed bound: the thumb sits at the bound.
    page = _open_slider_export(open_export, tmp_path, label)
    slider = _input(page, label)
    lo = float(slider.get_attribute("min"))
    hi = float(slider.get_attribute("max"))
    step = float(slider.get_attribute("step"))
    shown = _readout(page, label)
    _model_set(page, trait, value)
    assert _poll(page, shown.inner_text, lambda t: t == readout) == readout
    page.wait_for_timeout(300)
    page.evaluate(CURVE_UPDATES_JS)
    above = value > hi
    if above:
        keys = ["ArrowRight", "ArrowUp", "PageUp", "End"]
    else:
        keys = ["ArrowLeft", "ArrowDown", "PageDown", "Home"]
    for key in keys:
        _press(page, label, key)
        page.wait_for_timeout(300)
        # A key towards the bound does not set the bound, and the readout keeps the value.
        assert (_model_value(page, trait), shown.inner_text()) == (value, readout), key
    assert page.evaluate("() => window.__curveUpdates") == 0
    page.keyboard.press("ArrowLeft" if above else "ArrowRight")  # away from the bound: a step inside it
    inside = _poll_model(page, trait, lambda v: v != value)
    assert inside == pytest.approx(hi - step if above else lo + step)


# ---------------------------------------------------------------------------
# Touch and pen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["rail", "thumb"])
def test_showpdf_export_touch_tap_commits_once(open_export, tmp_path, where):
    # A tap commits the value under the finger, once. The browser's mouse events
    # repeating the tap, at a whole pixel and off a thumb moved towards it by its
    # touch adjustment, would commit a second value.
    label, nth, trait, start = "Probe size", 0, "probe_size", 50
    w = ShowPDF(_calibrated_pdf(scan=(6, 6)), k_min_fit=0.3, k_max_fit=1.2, analysis_mode="probe")
    page = open_export(w.export_html(tmp_path / "tap.html", encoding="uint8"), has_touch=True)
    assert page.evaluate(HOOK_MODEL_JS, f'input[aria-label="{label}"]')
    page.evaluate(CHANGES_JS, trait)
    track, lo, hi = _slider_track(page, label, nth)
    step = float(_input(page, label, nth).get_attribute("step"))
    pixel = (hi - lo) / track["width"]  # the value one pixel spans (a mouse position is a whole pixel)

    def step_at(px):  # the step of MUI's value at a position
        return round((px - track["x"]) / track["width"] * (hi - lo) / step)

    def reset():
        _set_trait_and_wait(page, trait, start, label, nth)
        page.evaluate("() => { window.__changes = []; }")

    reset()
    x0, y = _thumb_centre(page, label, nth)
    if where == "rail":
        taps = [x0 - 25 - 0.37 * i for i in range(5)]
    else:
        # Five taps across the 10 px thumb, off the start and at another step than the
        # nearest whole pixel, where the browser's mouse events land.
        start_step = round((start - lo) / step)
        across_thumb = [x0 - 4.5 + 0.05 * i for i in range(181)]
        taps = [x for x in across_thumb if step_at(x) not in (step_at(round(x)), start_step)]
        taps = taps[:: max(1, len(taps) // 5)][:5]
        assert len(taps) == 5, taps
    for i, x in enumerate(taps):
        reset()
        page.touchscreen.tap(x, y)
        under = _value_at_x(track, lo, hi, x)
        _poll(page, lambda: page.evaluate("() => window.__changes"), lambda c: len(c) > 0)
        page.wait_for_timeout(300)  # time for a second commit
        changes = page.evaluate("() => window.__changes")
        assert len(changes) == 1, (i, under, changes)
        assert abs(changes[0] - under) <= step / 2 + pixel, (i, under, changes)
        assert _slider_value(page, label, nth) == changes[0], i


@pytest.mark.parametrize("gesture", ["touch drag", "pen drag"])
def test_showpdf_export_touch_and_pen_drag_commit(open_export, tmp_path, gesture):
    w = ShowPDF(_calibrated_pdf(scan=(6, 6)), k_min_fit=0.3, analysis_mode="probe")
    page = open_export(w.export_html(tmp_path / "probe.html", encoding="uint8"), has_touch=True)
    assert page.evaluate(HOOK_MODEL_JS, 'input[aria-label="Probe size"]')
    _set_trait_and_wait(page, "probe_size", 20, "Probe size")
    track, lo, hi = _slider_track(page, "Probe size", 0)
    x, y = _thumb_centre(page, "Probe size", 0)
    end = x + 30
    cdp = page.context.new_cdp_session(page)
    path = [x + (end - x) * i / 5 for i in range(6)]
    if gesture == "touch drag":
        for i, px in enumerate(path):
            _touch(cdp, "touchStart" if i == 0 else "touchMove", [(px, y)])
        _touch(cdp, "touchEnd")
    else:  # a pen's only mouse events are the compatibility ones
        pen = {"button": "left", "clickCount": 1, "pointerType": "pen"}
        for i, px in enumerate(path):
            kind = "mousePressed" if i == 0 else "mouseMoved"
            cdp.send("Input.dispatchMouseEvent", {"type": kind, "x": px, "y": y, "buttons": 1, **pen})
        cdp.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": end, "y": y, "buttons": 0, **pen})
    size = _poll_model(page, "probe_size", lambda v: v != 20)
    under = _value_at_x(track, lo, hi, end)
    assert abs(size - under) <= 0.5 + (hi - lo) / track["width"], (size, under)
    assert _slider_value(page, "Probe size") == size


@pytest.mark.parametrize(
    "then", ["tap on the thumb", "click on the thumb", "End at the max", "tap elsewhere", "swipe elsewhere"]
)
def test_showpdf_export_cancelled_touch_press_commits_nothing(open_export, tmp_path, then):
    # A touch press that moves the thumb and ends in touchcancel commits nothing (MUI
    # ignores touchcancel), and the slider shows its old value again. Once Python has
    # changed the trait, an interaction that changes nothing must keep it, although MUI
    # then commits the cancelled press's value and a swipe reusing its touch id would move it.
    label, trait, start, later = "Probe size", "probe_size", 50, 20
    w = ShowPDF(_calibrated_pdf(scan=(6, 6)), k_min_fit=0.3, k_max_fit=1.2, analysis_mode="probe")
    page = open_export(w.export_html(tmp_path / "cancel.html", encoding="uint8"), has_touch=True)
    assert page.evaluate(HOOK_MODEL_JS, f'input[aria-label="{label}"]')
    _set_trait_and_wait(page, trait, start, label)
    page.evaluate(CHANGES_JS, trait)
    x, y = _thumb_centre(page, label, 0)
    cdp = page.context.new_cdp_session(page)
    _touch(cdp, "touchStart", [(x + 25, y)])
    pressed = _slider_value(page, label)  # the thumb moves to the finger on the rail
    _touch(cdp, "touchCancel")
    page.wait_for_timeout(300)
    assert pressed != start and _model_value(page, trait) == start
    assert _poll_slider(page, label, lambda v: v == start) == start
    if then == "End at the max":
        later = float(_input(page, label).get_attribute("max"))
    elif label == "Probe size":  # a click on the thumb then lands on its size
        later = _value_on_pixel(page, label, later)
    _set_trait_and_wait(page, trait, later, label)
    if then == "tap on the thumb":
        page.touchscreen.tap(*_thumb_centre(page, label, 0))
    elif then == "click on the thumb":
        x, y = _thumb_centre(page, label, 0)
        page.mouse.click(round(x), y)
    elif then == "End at the max":
        _press(page, label, "End")
    elif then == "tap elsewhere":  # off the widget, where MUI would still listen for the touch
        page.touchscreen.tap(*_below_widget(page))
    else:  # a swipe off the widget with the cancelled touch's id (CDP's default id)
        sx, sy = _below_widget(page)
        _touch(cdp, "touchStart", [(sx, sy)])
        for k in range(1, 8):
            _touch(cdp, "touchMove", [(sx - 40 * k, sy - 5 * k)])
        _touch(cdp, "touchEnd")
    page.wait_for_timeout(500)
    assert (_model_value(page, trait), page.evaluate("() => window.__changes")) == (later, [later]), then


# ---------------------------------------------------------------------------
# Line band and line width
# ---------------------------------------------------------------------------


def test_showpdf_export_band_overlay_is_the_selection_with_bisector_end_half_way(open_export, tmp_path):
    # The bisector of (0, 0) -> (12, 7) has an end exactly half way between two scan
    # positions, so the last bit of the line's length decides which way it snaps.
    w = ShowPDF(_calibrated_pdf(scan=(16, 16)), k_min_fit=0.3)
    w.set_line(0, 0, 12, 7, width=3)
    w.line_perpendicular = True
    assert w._effective_line() == (9, 0, 2, 10)
    band = w._line_band_mask()
    selected = int(band.sum())
    page = open_export(w.export_html(tmp_path / "perp.html", encoding="uint8"))
    assert page.evaluate(HOOK_MODEL_JS, 'input[aria-label="Line width"]')
    cover = _poll(
        page,
        lambda: np.array(page.evaluate(BAND_COVER_JS, [16, 16])),
        lambda c: c[band].min() > 0.5 and c[~band].max() < 0.1,
    )
    assert cover[band].min() > 0.5 and cover[~band].max() < 0.1, np.round(cover, 2)
    # The file's own recompute averages the same scan positions as Python.
    _press(page, "Line width", "ArrowRight")  # width 5
    assert _poll_model(page, "mask_pixel_count", lambda n: n != selected) != selected
    page.keyboard.press("ArrowLeft")  # width 3 again
    assert _poll_model(page, "mask_pixel_count", lambda n: n == selected) == selected


def test_showpdf_export_band_overlay_is_the_selection_at_every_angle(open_export, tmp_path):
    w = ShowPDF(_calibrated_pdf(scan=(16, 16)), k_min_fit=0.3)
    w.set_line(8, 8, 8, 14, width=3)
    page = open_export(w.export_html(tmp_path / "angles.html", encoding="full"))
    assert page.evaluate(HOOK_MODEL_JS, 'input[aria-label="Line width"]')
    for i, deg in enumerate(range(0, 360, 10)):
        th = np.deg2rad(deg)
        w.set_line(7.6, 8.3, 7.6 - 6.5 * np.sin(th), 8.3 + 6.5 * np.cos(th), width=(1, 3, 5)[i % 3])
        band = w._line_band_mask()
        line = {t: getattr(w, t) for t in ("line_row0", "line_col0", "line_row1", "line_col1", "line_width")}
        page.evaluate(SET_LINE_JS, line)
        # The overlay of each new line paints every selected scan position and no other one
        # (on these smaller cells the line and its end markers cover part of some band cells).
        cover = _poll(
            page,
            lambda: np.array(page.evaluate(BAND_COVER_JS, [16, 16])),
            lambda c: c[band].min() > 0.5 and c[~band].max() < 0.1,
        )
        assert cover[band].min() > 0.5 and cover[~band].max() < 0.1, (deg, np.round(cover, 2))


def test_showpdf_export_scan_zoom_reuses_band_overlay(open_export, tmp_path):
    w = ShowPDF(_calibrated_pdf(scan=(8, 8)), k_min_fit=0.3)
    w.set_line(1, 1, 6, 4, width=3)
    page = open_export(w.export_html(tmp_path / "zoom.html", encoding="full"), init_script=OVERLAY_BUILDS_JS)

    def counts():
        return page.evaluate("() => [window.__imageData, window.__lineRedraws]")

    built, redraws = _poll(page, counts, lambda c: c[1] > 0)
    assert built >= 1
    box = page.locator('[data-quantem-scientific-output="showpdf-navigation"]').bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    for _ in range(3):
        page.mouse.wheel(0, -100)
        page.wait_for_timeout(50)
    # Each zoom step redraws the overlay, from the band image already built.
    redraws_after = _poll(page, counts, lambda c: c[1] >= redraws + 3)[1]
    assert redraws_after >= redraws + 3
    assert counts()[0] == built


@pytest.mark.parametrize("leave", ["probe", "clear"])
def test_showpdf_export_band_overlay_released_without_line(open_export, tmp_path, leave):
    w = ShowPDF(_calibrated_pdf(scan=(8, 8)), k_min_fit=0.3)
    w.set_line(1, 1, 6, 4, width=3)
    line = {t: getattr(w, t) for t in ("line_row0", "line_col0", "line_row1", "line_col1")}
    page = open_export(w.export_html(tmp_path / "rel.html", encoding="uint8"), init_script=OVERLAY_BUILDS_JS)
    assert page.evaluate(HOOK_MODEL_JS, 'input[aria-label="Line width"]')
    built = _poll(page, lambda: page.evaluate("() => window.__imageData"), lambda n: n >= 1)
    assert built >= 1  # the band image exists before it is released
    if leave == "probe":  # probe mode draws no band; then back to the line
        page.get_by_role("button", name="probe", exact=True).click()
        page.wait_for_timeout(300)
        page.get_by_role("button", name="line", exact=True).click()
    else:  # no line; then the same line again (line_active stays off, so no empty band is drawn between)
        page.get_by_role("button", name="Clear", exact=True).click()
        page.wait_for_timeout(300)
        page.evaluate(SET_LINE_JS, line)
    # The band image was not kept while no band was drawn, so it is built again.
    rebuilt = _poll(page, lambda: page.evaluate("() => window.__imageData"), lambda n: n > built)
    assert rebuilt == built + 1


def test_showpdf_export_line_width_range_keeps_the_widest_width(open_export, tmp_path):
    w = ShowPDF(_calibrated_pdf(scan=(6, 6)), k_min_fit=0.3)
    w.set_line(0, 0, 5, 5, width=151)  # set from Python, above the slider's usual 99
    page = open_export(w.export_html(tmp_path / "wide.html", encoding="uint8"))
    assert page.evaluate(HOOK_MODEL_JS, 'input[aria-label="Line width"]')
    slider = _input(page, "Line width")
    assert _slider_value(page, "Line width") == 151  # the thumb at the width, not at 99
    assert slider.get_attribute("aria-valuemax") == "151"
    _press(page, "Line width", "ArrowLeft")  # narrower
    assert _poll_model(page, "line_width", lambda v: v == 149) == 149
    # The range still ends at 151 (the thumb does not jump to its end), so 151 can be set again.
    assert slider.get_attribute("aria-valuemax") == "151"
    page.keyboard.press("ArrowRight")
    assert _poll_model(page, "line_width", lambda v: v == 151) == 151


# ---------------------------------------------------------------------------
# Mask painting
# ---------------------------------------------------------------------------


def test_showpdf_export_brush_stroke_that_leaves_the_image_is_kept(open_export, tmp_path):
    # A brush stroke released outside the scan image still selects the positions it painted.
    w = ShowPDF(
        _calibrated_pdf(scan=(8, 8)),
        k_min_fit=0.3,
        analysis_mode="mask",
        mask_tool="freeform",
        mask_brush_size=1,
    )
    page = open_export(w.export_html(tmp_path / "paint.html", encoding="uint8"))
    assert page.evaluate(HOOK_MODEL_JS, NAV_CANVAS)
    page.evaluate(CURVE_UPDATES_JS)
    before = _model_value(page, "mask_pixel_count")
    box = page.locator(NAV_CANVAS).first.bounding_box()
    cell = box["width"] / 8
    y = box["y"] + 2.5 * cell
    page.mouse.move(box["x"] + 2.5 * cell, y)
    page.mouse.down()
    page.mouse.move(box["x"] + 4.5 * cell, y, steps=4)
    page.mouse.move(box["x"] + box["width"] + 40, y, steps=8)  # out of the image, button still down
    page.mouse.up()
    painted = _poll_model(page, "mask_pixel_count", lambda n: n != before)
    assert 0 < painted < 64, (before, painted)
    assert page.evaluate("() => window.__curveUpdates") > 0
