"""Multiple recordings stay interactive while their detector is moved."""

import json

import pytest

pytestmark = pytest.mark.browser

DP = '[data-quantem-scientific-output="show4dstem-diffraction-pattern"]'
COMPARISON = '[data-quantem-scientific-output="show4dstem-compare-0"]'
SOURCE = """
import torch
from quantem.widget import Show4DSTEM

scan_t = torch.arange(16 * 16).reshape(1, 16, 16, 1, 1)
row_t = torch.arange(32).reshape(1, 1, 1, 32, 1)
col_t = torch.arange(32).reshape(1, 1, 1, 1, 32)
frame_t = torch.arange(3).reshape(3, 1, 1, 1, 1)
data_t = ((scan_t * 5 + row_t * 3 + col_t * 7 + frame_t * 11) % 97 + 1).float()
viewer = Show4DSTEM(
    data_t, device="cpu", offline=False, precompute_virtual_images=False,
    bright_field_center_px=(15.5, 15.5), bright_field_radius_px=5,
    view_mode="multiple", compare_max_panels=3, compare_group_mode="all",
)
viewer
"""
REPORT = """
import hashlib, json
mask_t = torch.as_tensor(viewer._current_detector_mask(), dtype=torch.float64)
expected_t = (data_t[viewer.compare_panel_indices].double() * mask_t).sum((-2, -1)).float()
expected_t /= max(1, int(torch.count_nonzero(mask_t)))
actual_t = torch.frombuffer(bytearray(viewer.compare_virtual_image_bytes), dtype=torch.float32)
print(json.dumps({
    "center": list(viewer.roi_center),
    "pos": [viewer.pos_row, viewer.pos_col],
    "region": list(viewer.vi_roi_center),
    "panels": len(viewer.compare_panel_indices),
    "equal": torch.equal(actual_t, expected_t.flatten()),
    "sha256": hashlib.sha256(viewer.compare_virtual_image_bytes).hexdigest(),
}))
"""


def _position(page, selector, row, col, size):
    box = page.locator(selector).first.bounding_box()
    return (box["x"] + (col + 0.5) * box["width"] / size,
            box["y"] + (row + 0.5) * box["height"] / size)


def _wait_state(page, jupyter, notebook, predicate):
    for _ in range(30):
        state = json.loads(jupyter.run_in_kernel(notebook, REPORT).splitlines()[-1])
        if predicate(state):
            return state
        page.wait_for_timeout(100)
    pytest.fail(f"Comparison did not update: {state}")


def test_comparison_detector_drag_and_scan_region_switch(
    jupyter, open_page, hardware_adapter, browser_out,
):
    notebook = "show4dstem_compare_pointer.ipynb"
    jupyter.write_notebook(notebook, SOURCE)
    page = jupyter.open(open_page, notebook, "compare_pointer", webgpu=True,
                        ready=COMPARISON, out=browser_out)
    adapter = page.adapter()
    hardware_adapter(adapter)
    before = _wait_state(page, jupyter, notebook, lambda state: state["equal"])
    before_png = page.locator(COMPARISON).first.screenshot(
        path=str(browser_out / "compare_pointer_before.png"))
    page.locator(DP).first.scroll_into_view_if_needed()
    page.mouse.move(*_position(page, DP, 15.5, 15.5, 32))
    page.mouse.down()
    try:
        page.mouse.move(*_position(page, DP, 9, 22, 32), steps=12)
        held = _wait_state(page, jupyter, notebook, lambda state: (
            state["center"] != before["center"] and state["equal"]
            and state["sha256"] != before["sha256"]
        ))
        assert held["panels"] == 3
        held_png = page.locator(COMPARISON).first.screenshot(
            path=str(browser_out / "compare_pointer_held.png"))
        assert held_png != before_png
    finally:
        page.mouse.up()

    jupyter.run_in_kernel(notebook,
        "viewer.vi_roi_mode = 'circle'; viewer.vi_roi_radius = 2; "
        "viewer.vi_roi_center = [3.0, 3.0]")
    page.wait_for_timeout(300)
    page.locator(COMPARISON).first.scroll_into_view_if_needed()
    page.mouse.click(*_position(page, COMPARISON, 11, 10, 16))
    region = _wait_state(page, jupyter, notebook, lambda state: (
        abs(state["region"][0] - 11) < 0.05
        and abs(state["region"][1] - 10) < 0.05
    ))
    assert region["pos"] == held["pos"]
    jupyter.run_in_kernel(notebook, "viewer.vi_roi_mode = 'off'")
    page.wait_for_timeout(300)
    page.mouse.click(*_position(page, COMPARISON, 6, 4, 16))
    point = _wait_state(page, jupyter, notebook, lambda state: state["pos"] == [6, 4])
    assert page.adapter() == adapter
    (browser_out / "compare_pointer.json").write_text(json.dumps({
        "adapter": adapter, "before": before, "held": held,
        "region": region, "point": point,
    }, indent=2))
    assert [error for error in page.errors if "debugger session" not in error] == []
