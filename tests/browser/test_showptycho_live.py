"""ShowPtycho in a live JupyterLab: keyboard scope and two viewers on one page.

A notebook holds a direct ptychography session in ShowPtycho and an empty code cell. Text
typed into the cell, with the letters and keys the widget binds (P, R, S,
Space, arrows), must land in the cell editor untouched while the widget sits
on the page. After a click on the widget the same keys must reach it: P pins
the current phase and Space starts and stops the aberration sweep, and the
cell text stays as it was. Dragging one pin onto another reorders them.

Starring a viewed pin with S, and again to unstar it, reaches the kernel as
exactly what the pin shows.

Two ShowPtycho viewers of different images on one page each keep drawing their
own image when one of them changes colormap.
"""

import json
import re
import uuid

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.browser

PHASE = '[data-quantem-scientific-output="Phase"]'
PINS = "[data-showptycho-pin-id]"
SESSION = """import numpy as np
from quantem.gpu import DirectPtycho
from quantem.widget import ShowPtycho

rows, cols = np.mgrid[0:32, 0:32]
disk = (np.hypot(rows - 16, cols - 16) < 6).astype(np.float32)
scan = np.random.default_rng(0).normal(size=(24, 24, 1, 1)).astype(np.float32)
data = disk * (100 + scan) * (1 + 0.02 * np.cos(rows / 3.0))
viewer = ShowPtycho(DirectPtycho(data, voltage_kV=300, semiangle_mrad=20, scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(20 / 6, 20 / 6)))
viewer"""


def test_shortcuts_need_widget_focus_and_leave_cell_typing_alone(jupyter, open_page, hardware_adapter, browser_out):
    pytest.importorskip("quantem.gpu")
    jupyter.write_notebook("showptycho_keys.ipynb", SESSION, "")
    page = jupyter.open(open_page, "showptycho_keys.ipynb", "keys", webgpu=True, ready=PHASE, out=browser_out)
    hardware_adapter(page.adapter())
    editor = page.locator(".jp-Cell .cm-content").nth(1)
    editor.click()
    page.keyboard.type("pp rs")
    page.keyboard.press("ArrowLeft")
    page.keyboard.press("ArrowLeft")
    page.keyboard.type("x")
    page.wait_for_timeout(500)
    page.screenshot(path=str(browser_out / "showptycho_keys_typed.png"))
    assert editor.inner_text() == "pp xrs"
    assert page.locator(PINS).count() == 0
    assert page.get_by_label("Play sweep").count() == 1
    page.locator(PHASE).first.click()
    page.keyboard.press("p")
    page.wait_for_timeout(300)
    page.keyboard.press("p")
    page.wait_for_timeout(300)
    assert page.locator(PINS).count() == 2
    page.keyboard.press(" ")
    page.wait_for_timeout(300)
    assert page.get_by_label("Pause sweep").count() == 1
    page.keyboard.press(" ")
    page.wait_for_timeout(300)
    assert page.get_by_label("Play sweep").count() == 1
    page.screenshot(path=str(browser_out / "showptycho_keys_widget.png"))
    assert editor.inner_text() == "pp xrs"
    # dragging the first pin onto the second swaps them
    order = page.eval_on_selector_all(PINS, "pins => pins.map(pin => pin.dataset.showptychoPinId)")
    page.locator(PINS).nth(1).scroll_into_view_if_needed()
    first, second = page.locator(PINS).nth(0).bounding_box(), page.locator(PINS).nth(1).bounding_box()
    page.mouse.move(first["x"] + first["width"] / 2, first["y"] + first["height"] / 2)
    page.mouse.down()
    page.mouse.move(second["x"] + second["width"] / 2, second["y"] + second["height"] / 2, steps=8)
    page.mouse.up()
    page.wait_for_timeout(500)
    assert page.eval_on_selector_all(PINS, "pins => pins.map(pin => pin.dataset.showptychoPinId)") == order[::-1]
    # JupyterLab's debugger panel reports "No active debugger session" on its own
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors


STARRED = "import json; print(json.dumps([pin['starred'] for pin in viewer._pinned]))"


def test_star_key_sends_the_state_the_pin_shows(jupyter, open_page, hardware_adapter, browser_out):
    jupyter.write_notebook("showptycho_star.ipynb", SESSION)
    page = jupyter.open(open_page, "showptycho_star.ipynb", "star", webgpu=True, ready=PHASE, out=browser_out)
    hardware_adapter(page.adapter())
    page.locator(PHASE).first.click()
    for _ in range(2):
        page.keyboard.press("p")
        page.wait_for_timeout(300)
    page.keyboard.press("ArrowRight")  # view the first pin
    page.wait_for_timeout(500)
    states = []
    for _ in range(2):
        page.keyboard.press("s")
        page.wait_for_timeout(800)
        states.append(json.loads(jupyter.run_in_kernel("showptycho_star.ipynb", STARRED)))
    assert states == [[True, False], [False, False]], states
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors


TWO_VIEWERS = """import numpy as np
from IPython.display import display
from quantem.widget import ShowPtycho

rows, cols = np.indices((64, 64))
across, down = ShowPtycho((cols / 63).astype(np.float32)), ShowPtycho((rows / 63).astype(np.float32))
display(across, down)"""


def gradient_axis(path) -> str:
    """Whether the drawn gray image changes along its columns or its rows."""
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    inner = gray[gray.shape[0] // 8:-gray.shape[0] // 8, gray.shape[1] // 8:-gray.shape[1] // 8]
    return "cols" if np.abs(np.diff(inner, axis=1)).sum() > np.abs(np.diff(inner, axis=0)).sum() else "rows"


def test_two_viewers_each_keep_their_own_image(jupyter, open_page, hardware_adapter, browser_out):
    jupyter.write_notebook("showptycho_two.ipynb", TWO_VIEWERS)
    page = jupyter.open(open_page, "showptycho_two.ipynb", "two", webgpu=True, ready=PHASE, out=browser_out)
    hardware_adapter(page.adapter())
    phases = page.locator(PHASE)
    assert phases.count() == 2
    # the first viewer switches to a gray colormap; its canvas must still show its own gradient
    combos = page.locator('.jp-OutputArea-output [role="combobox"]')
    first = next(index for index in range(combos.count()) if combos.nth(index).inner_text().strip() == "Inferno")
    combos.nth(first).click()
    page.locator('li[role="option"]', has_text="Gray").first.click()
    page.mouse.move(5, 300)
    page.wait_for_timeout(1500)
    shots = []
    for index in range(2):
        path = browser_out / f"showptycho_two_{index}.png"
        phases.nth(index).scroll_into_view_if_needed()
        phases.nth(index).screenshot(path=str(path))
        shots.append(path)
    assert [gradient_axis(path) for path in shots] == ["cols", "rows"]
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors


def test_tilt_auto_save_and_exact_export_reopen(jupyter, open_page, hardware_adapter, browser_out, serve):
    """Real controls retain fitted science; an exported integer-count session reopens."""
    setup = SESSION.replace("size=(24, 24, 1, 1)", "size=(128, 128, 1, 1)")
    setup = setup.replace("viewer = ShowPtycho(DirectPtycho(data,", "data = np.rint(data).astype(np.uint16)\nviewer = ShowPtycho(DirectPtycho(data,")
    setup = setup.rsplit("\nviewer", 1)[0] + """
import json
from pathlib import Path
viewer.sample_json = json.dumps({'sample_tilt_mrad': [3., -4.], 'depth_spread_nm': 12.})
viewer.higher_order_aberrations_json = json.dumps({'C30_nm': 2.})
viewer
"""
    notebook = "showptycho_tilt_export.ipynb"
    jupyter.write_notebook(notebook, setup)
    page = jupyter.open(open_page, notebook, "tilt-export", webgpu=True, ready=PHASE, out=browser_out)
    hardware_adapter(page.adapter())
    # the output sampling select lives in the View settings popover
    page.get_by_role("button", name="View settings").click()
    sampling = page.get_by_role("combobox", name="Upsampling")
    sampling.click()
    page.get_by_role("option", name="Auto", exact=False).click()
    page.keyboard.press("Escape")
    page.wait_for_timeout(1500)
    toggle = page.get_by_role("checkbox", name="Tilt correction")
    toggle.uncheck()
    page.wait_for_timeout(1000)
    page.locator("button").filter(has_text=re.compile(r"^Save$")).click()
    page.wait_for_timeout(1000)
    proof = jupyter.run_in_kernel(notebook, """
saved = json.loads(Path(viewer.calibration_path).read_text())
assert saved['sample_tilt_mrad'] == [3., -4.]
assert saved['depth_spread_nm'] == 12.
assert saved['display']['tilt_correction'] is False
assert saved['upsample_mode'] == 'auto'
assert viewer.upsample == 'auto'
assert json.loads(viewer.higher_order_aberrations_json)['C30_nm'] == 2.
expected, _ = viewer._direct.preview({**viewer._current(), 'C30_nm': 2.}, upsample=viewer.auto_upsample,
    sample_tilt_mrad=(0., 0.), depth_spread_nm=12.)
np.testing.assert_array_equal(np.frombuffer(viewer.phase_bytes, np.float32).reshape(expected.shape), expected)
print('saved-fitted-science')
""")
    assert "saved-fitted-science" in proof, proof
    toggle.check()
    page.get_by_role("button", name="View settings").click()
    sampling.click()
    page.get_by_role("option", name="1×", exact=True).click()
    page.wait_for_timeout(1000)
    page.locator(PHASE).first.screenshot(path=str(browser_out / "showptycho_tilt_native.png"))
    export = browser_out / "showptycho_exact_reopen"
    # A unique test artifact path prevents re-export from replacing an earlier run.
    export = export / uuid.uuid4().hex
    proof = jupyter.run_in_kernel(notebook, f"""
viewer.export({str(export)!r})
print('exported-exact-counts')
""")
    assert "exported-exact-counts" in proof, proof
    url = serve(export)
    offline = open_page(url + "/index.html", webgpu=True)
    hardware_adapter(offline.adapter())
    offline.locator(PHASE).wait_for(timeout=60000)
    offline.wait_for_timeout(8000)
    assert offline.get_by_role("checkbox", name="Tilt correction").is_checked()
    from playwright.sync_api import expect
    expect(offline.locator(PHASE).first).to_have_attribute("data-source-width", "128", timeout=60000)
    expect(offline.locator(PHASE).first).to_have_attribute("data-source-height", "128")
    screenshot = browser_out / "showptycho_export_reopened.png"
    offline.locator(PHASE).first.screenshot(path=str(screenshot))
    assert np.asarray(Image.open(screenshot).convert("L")).std() > 10
    assert not offline.errors, offline.errors
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors


def test_direct_methods_and_regularization_follow_live_slider_requests(jupyter, open_page, browser_out):
    """Drag a stabilization strength and compare the shown phase with the backend."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Regularized direct reconstruction requires CUDA")
    notebook = "showptycho_direct_methods.ipynb"
    jupyter.write_notebook(notebook, SESSION)
    page = jupyter.open(open_page, notebook, "direct-methods", webgpu=True, ready=PHASE, out=browser_out)
    # This exercises live CUDA reconstruction; hardware browser rendering has
    # a separate release gate. Canvas fallback must retain the same controls.
    select = page.get_by_role("combobox", name="Reconstruction method")
    select.click()
    page.get_by_role("option", name="Phase · experimental", exact=True).click()
    slider = page.get_by_role("slider", name="Phase damping", exact=True)
    slider.wait_for(state="visible")
    slider.focus()
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(1500)
    select.click()
    page.get_by_role("option", name="Amplitude + phase · experimental", exact=True).click()
    amplitude = page.get_by_role("slider", name="Amplitude damping", exact=True)
    amplitude.wait_for(state="visible")
    amplitude.focus()
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(1500)
    proof = jupyter.run_in_kernel(notebook, """
import json
request = json.loads(viewer.result_json)
assert request['method'] == 'complex', request
assert request['phase_damp'] > .03, request
assert request['amplitude_damp'] > .03, request
expected, _ = viewer._direct.preview(viewer.ptycho_parameters['aberrations'], compute_loss=False,
    upsample=viewer._output_factor, **viewer._reconstruction_options())
shown = np.frombuffer(viewer.phase_bytes, np.float32).reshape(expected.shape)
np.testing.assert_array_equal(shown, expected)
assert not viewer.webgpu_preview_enabled
print('live direct method and strengths match backend')
""")
    assert "live direct method and strengths match backend" in proof
    page.screenshot(path=str(browser_out / "showptycho_direct_methods.png"))
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors


def test_phase_draws_on_open_and_save_writes_the_shown_png(jupyter, open_page, hardware_adapter, browser_out):
    """The phase shows before any control is touched, and Save writes it as PNG in the shown colormap."""
    notebook = "showptycho_save_png.ipynb"
    jupyter.write_notebook(notebook, SESSION)
    page = jupyter.open(open_page, notebook, "save-png", webgpu=True, ready=PHASE, out=browser_out)
    hardware_adapter(page.adapter())
    page.wait_for_timeout(1500)
    first_draw = browser_out / "showptycho_first_draw.png"
    page.locator(PHASE).first.screenshot(path=str(first_draw))
    assert np.asarray(Image.open(first_draw).convert("L")).std() > 10, "phase blank until a control changes"
    combos = page.locator('.jp-OutputArea-output [role="combobox"]')
    first = next(index for index in range(combos.count()) if combos.nth(index).inner_text().strip() == "Inferno")
    combos.nth(first).click()
    page.locator('li[role="option"]', has_text="Gray").first.click()
    page.mouse.move(5, 300)
    page.wait_for_timeout(1000)
    page.locator("button").filter(has_text=re.compile(r"^Save$")).click()
    page.wait_for_timeout(1500)
    proof = jupyter.run_in_kernel(notebook, """
from PIL import Image
image = np.asarray(Image.open(viewer.calibration_image_path).convert('RGB'))
assert viewer.cmap == 'gray' and viewer.vmin is not None and viewer.vmin < viewer.vmax
assert image.shape[:2] == (viewer.phase_height, viewer.phase_width)
assert (image[..., 0] == image[..., 1]).all() and image.std() > 10
print('saved-shown-png')
""")
    assert "saved-shown-png" in proof, proof
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors


def test_reset_all_returns_model_and_display_to_launch(jupyter, open_page, hardware_adapter, browser_out):
    """Reset All undoes colormap, output sampling and tilt-correction edits in one click."""
    notebook = "showptycho_reset_all.ipynb"
    setup = SESSION.replace("size=(24, 24, 1, 1)", "size=(128, 128, 1, 1)").rsplit("\nviewer", 1)[0] + """
import json
viewer.sample_json = json.dumps({'sample_tilt_mrad': [3., -4.], 'depth_spread_nm': 12.})
viewer.fitted_sample_json = viewer.sample_json
launch = (viewer.cmap, viewer.upsample, viewer.method, viewer.phase_damp)
viewer
"""
    jupyter.write_notebook(notebook, setup)
    page = jupyter.open(open_page, notebook, "reset-all", webgpu=True, ready=PHASE, out=browser_out)
    hardware_adapter(page.adapter())
    page.wait_for_timeout(1500)
    combos = page.locator('.jp-OutputArea-output [role="combobox"]')
    first = next(index for index in range(combos.count()) if combos.nth(index).inner_text().strip() == "Inferno")
    combos.nth(first).click()
    page.locator('li[role="option"]', has_text="Gray").first.click()
    page.mouse.move(5, 300)
    # the output sampling select lives in the View settings popover
    page.get_by_role("button", name="View settings").click()
    sampling = page.get_by_role("combobox", name="Upsampling")
    sampling.click()
    page.get_by_role("option", name="1×", exact=True).click()
    page.keyboard.press("Escape")
    page.wait_for_timeout(1000)
    page.get_by_role("checkbox", name="Tilt correction").uncheck()
    page.wait_for_timeout(1000)
    edited = jupyter.run_in_kernel(notebook, "print(viewer.cmap, viewer.upsample, viewer.tilt_correction)")
    assert "gray 1 False" in edited, edited
    page.locator("button").filter(has_text=re.compile(r"^Reset All$")).click()
    page.wait_for_timeout(2000)
    proof = jupyter.run_in_kernel(notebook, """
assert (viewer.cmap, viewer.upsample, viewer.method, viewer.phase_damp) == launch, (viewer.cmap, viewer.upsample, launch)
assert viewer.tilt_correction is True
print('reset-all-launch')
""")
    assert "reset-all-launch" in proof, proof
    page.locator(PHASE).first.screenshot(path=str(browser_out / "showptycho_reset_all.png"))
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors
