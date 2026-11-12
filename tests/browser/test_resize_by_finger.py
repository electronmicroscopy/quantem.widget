"""Every viewer's panel resize grip follows a finger, not only a mouse.

A phone or a folding phone sends touch input: a grip that listens to mouse events alone never moves under
a finger. The drag here is a trusted touch (CDP ``Input.dispatchTouchEvent``) in a live notebook.
"""

import json

import pytest

pytestmark = pytest.mark.browser

WIDGETS = {
    "show2d": "from quantem.widget import Show2D\nShow2D(np.random.default_rng(0).random((64, 64)))",
    "show3d": "from quantem.widget import Show3D\nShow3D(np.random.default_rng(0).random((6, 64, 64)))",
    "show3dslices": "from quantem.widget import Show3DSlices\nShow3DSlices(np.random.default_rng(0).random((32, 48, 40)))",
    "show4dstem": "from quantem.widget import Show4DSTEM\nShow4DSTEM(np.random.default_rng(0).random((8, 8, 32, 32)).astype(np.float32))",
    "showptycho": "from quantem.widget import ShowPtycho\nShowPtycho(np.random.default_rng(0).random((64, 64)))",
}

# The visible panel resize grips (the bottom-right corners drawn with the nwse-resize cursor), in page order.
GRIPS = """() => [...document.querySelectorAll('.jp-OutputArea-output *')]
  .filter(el => getComputedStyle(el).cursor === 'nwse-resize' && el.getBoundingClientRect().width > 0)"""
# Scroll grip ``index`` into view and return its centre and the element under that point (the grip, not a cover).
GRIP_AT = "index => { const el = (" + GRIPS + ")()[index]; el.scrollIntoView({block: 'center', inline: 'center'}); " \
    "const r = el.getBoundingClientRect(); const x = r.x + r.width / 2, y = r.y + r.height / 2; " \
    "return [x, y, el.contains(document.elementFromPoint(x, y))]; }"
# Where grip ``index`` is, without scrolling.
GRIP_NOW = "index => { const r = (" + GRIPS + ")()[index].getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }"


@pytest.mark.parametrize("widget", list(WIDGETS))
def test_each_resize_grip_follows_a_finger(widget, jupyter, open_page, browser_out):
    notebook = f"finger_resize_{widget}.ipynb"
    jupyter.write_notebook(notebook, "import numpy as np\n" + WIDGETS[widget])
    page = jupyter.open(open_page, notebook, f"finger-{widget}", webgpu=True,
                        ready=".jp-OutputArea-output canvas", out=browser_out, collapse_input=True)
    cdp = page.context.new_cdp_session(page)
    cdp.send("Emulation.setTouchEmulationEnabled", {"enabled": True, "maxTouchPoints": 5})
    grips = page.evaluate("() => (" + GRIPS + ")().length")
    assert grips, f"{widget}: no resize grip on the page"
    moved = []
    for index in range(grips):
        x, y, reachable = page.evaluate(GRIP_AT, index)
        assert reachable, f"{widget}: grip {index} is covered at its centre"
        cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
        for step in range(1, 11):
            cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x + 6 * step, "y": y + 6 * step}]})
            page.wait_for_timeout(16)
        cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        page.wait_for_timeout(500)
        moved.append(round(page.evaluate(GRIP_NOW, index)[0] - x))
    page.screenshot(path=str(browser_out / f"finger_resize_{widget}.png"))
    (browser_out / f"finger_resize_{widget}.json").write_text(json.dumps({"grips": grips, "moved_px": moved}))
    # JupyterLab's debugger extension reports a missing session on its own.
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors
    # The finger moved 60 px right and down; each grip (the panel's corner) follows by most of it.
    assert all(dx >= 30 for dx in moved), f"{widget}: grips moved {moved} px under a 60 px finger drag"
