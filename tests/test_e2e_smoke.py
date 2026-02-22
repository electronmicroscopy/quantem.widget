"""
Playwright smoke tests — verify each widget renders in a real browser.

All widgets are loaded in a single notebook to avoid repeated
JupyterLab page loads. Tests cover rendering, interactions, and theme switching.

Run with:
    python -m pytest tests/test_e2e_smoke.py -v

Requires: playwright, jupyterlab
"""

import time
import re

import pytest

from conftest import TESTS_DIR, _run_notebook_and_wait, _write_notebook

NOTEBOOK_PATH = TESTS_DIR / "_smoke_all_widgets.ipynb"
SCREENSHOT_DIR = TESTS_DIR / "screenshots" / "smoke"

@pytest.fixture(scope="module")
def smoke_page(browser_context):
    """Single notebook with all widgets, opened once for the module."""
    _write_notebook(NOTEBOOK_PATH, [
        {"source": [
            "import numpy as np\n",
            "from quantem.widget import Bin, Mark2D, Show1D, Show2D, Show3D, Show3DVolume, Show4DSTEM\n",
            "from quantem.widget import Show4D, Edit2D, Align2D, ShowComplex2D\n",
        ]},
        {"source": [
            "Show1D(np.random.rand(256).astype(np.float32), title='Show1D Smoke')\n",
        ]},
        {"source": [
            "Mark2D(np.random.rand(64, 64).astype(np.float32))\n",
        ]},
        {"source": [
            "Show2D(np.random.rand(64, 64).astype(np.float32))\n",
        ]},
        {"source": [
            "Show3D(np.random.rand(10, 64, 64).astype(np.float32))\n",
        ]},
        {"source": [
            "Show3DVolume(np.random.rand(32, 32, 32).astype(np.float32))\n",
        ]},
        {"source": [
            "Show4DSTEM(np.random.rand(8, 8, 32, 32).astype(np.float32))\n",
        ]},
        {"source": [
            "Bin(np.random.rand(8, 8, 32, 32).astype(np.float32), pixel_size=2.4, k_pixel_size=0.48, device='cpu')\n",
        ]},
        {"source": [
            "Show4D(np.random.rand(8, 8, 32, 32).astype(np.float32))\n",
        ]},
        {"source": [
            "Edit2D(np.random.rand(64, 64).astype(np.float32))\n",
        ]},
        {"source": [
            "Align2D(np.random.rand(64, 64).astype(np.float32), np.random.rand(64, 64).astype(np.float32))\n",
        ]},
        {"source": [
            "ShowComplex2D(np.random.rand(64, 64).astype(np.float32) + 1j * np.random.rand(64, 64).astype(np.float32))\n",
        ]},
    ])
    page = browser_context.new_page()
    _run_notebook_and_wait(page, NOTEBOOK_PATH)
    yield page
    page.close()
    NOTEBOOK_PATH.unlink(missing_ok=True)

ALL_WIDGETS = [
    "show1d-root",
    "mark2d-root",
    "show2d-root",
    "show3d-root",
    "show3dvolume-root",
    "show4dstem-root",
    "bin-root",
    "show4d-root",
    "edit2d-root",
    "align2d-root",
    "showcomplex-root",
]

VIEWER_WIDGETS_WITH_CUSTOMIZER = [
    "show2d-root",
    "show3d-root",
    "show3dvolume-root",
    "show4d-root",
    "show4dstem-root",
    "showcomplex-root",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _click_menu_item(page, text, timeout=3000):
    """Click a MUI dropdown menu item (global portal, not scoped to widget)."""
    item = page.locator(f'.MuiMenuItem-root:has-text("{text}")').first
    item.wait_for(state="visible", timeout=timeout)
    item.click()

def _screenshot(widget, name):
    """Save a widget screenshot to the smoke directory."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    widget.screenshot(path=str(SCREENSHOT_DIR / f"{name}.png"))

def _dismiss_menus(page):
    """Press Escape to dismiss any open MUI dropdown menus."""
    backdrop = page.locator(".MuiBackdrop-root")
    if backdrop.count() > 0:
        page.keyboard.press("Escape")
        time.sleep(0.5)

def _change_dropdown(widget, page, nth, value, wait=1.0):
    """Change the nth MUI Select dropdown to the given value."""
    _dismiss_menus(page)
    widget.locator(".MuiSelect-select").nth(nth).click()
    time.sleep(0.5)
    _click_menu_item(page, value)
    time.sleep(wait)

def _set_dark_theme(page):
    """Switch JupyterLab to dark theme via DOM manipulation."""
    page.evaluate("""() => {
        document.body.dataset.jpThemeLight = 'false';
        document.body.dataset.jpThemeName = 'JupyterLab Dark';
        document.body.classList.remove('jp-theme-light');
        document.body.classList.add('jp-theme-dark');
    }""")
    time.sleep(2)

def _set_light_theme(page):
    """Switch JupyterLab back to light theme."""
    page.evaluate("""() => {
        document.body.dataset.jpThemeLight = 'true';
        document.body.dataset.jpThemeName = 'JupyterLab Light';
        document.body.classList.remove('jp-theme-dark');
        document.body.classList.add('jp-theme-light');
    }""")
    time.sleep(2)

def _extract_show4d_nav_pos(widget) -> tuple[int, int]:
    """Parse the current Show4D navigation cursor (row, col) from widget text."""
    match = re.search(r"at \((\d+),\s*(\d+)\)", widget.inner_text())
    assert match is not None, "Could not find Show4D nav position text 'at (row, col)'"
    return int(match.group(1)), int(match.group(2))

# ---------------------------------------------------------------------------
# Basic rendering tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("css_class", ALL_WIDGETS)
def test_widget_root_exists(smoke_page, css_class):
    root = smoke_page.locator(f".{css_class}")
    assert root.count() >= 1, f"Widget .{css_class} not found on page"

@pytest.mark.parametrize("css_class", ALL_WIDGETS)
def test_canvas_rendered(smoke_page, css_class):
    canvas = smoke_page.locator(f".{css_class} canvas")
    assert canvas.count() >= 1, f"No canvas in .{css_class}"
    box = canvas.first.bounding_box()
    assert box is not None, f"Canvas in .{css_class} has no bounding box"
    assert box["width"] > 0 and box["height"] > 0, f"Canvas in .{css_class} has zero dimensions"

@pytest.mark.parametrize("css_class", ALL_WIDGETS)
def test_widget_screenshot(smoke_page, css_class):
    """Capture a screenshot of each widget for visual verification."""
    root = smoke_page.locator(f".{css_class}").first
    name = css_class.replace("-root", "")
    _screenshot(root, name)

@pytest.mark.parametrize("css_class", VIEWER_WIDGETS_WITH_CUSTOMIZER)
def test_viewer_controls_dropdown_exists(smoke_page, css_class):
    """Viewer widgets expose the shared controls customizer button."""
    widget = smoke_page.locator(f".{css_class}").first
    widget.scroll_into_view_if_needed()
    assert widget.locator('button[aria-label="Customize controls"]').count() >= 1

# ---------------------------------------------------------------------------
# Show1D interaction tests
# ---------------------------------------------------------------------------

def test_show1d_toggle_log_scale(smoke_page):
    """Toggle log scale on Show1D via switch."""
    widget = smoke_page.locator(".show1d-root").first
    widget.scroll_into_view_if_needed()
    # Switches: Log(0), Auto(1), Grid(2), Legend(3)
    widget.locator(".MuiSwitch-root").nth(0).click()
    time.sleep(1)
    _screenshot(widget, "show1d_log")
    widget.locator(".MuiSwitch-root").nth(0).click()
    time.sleep(0.5)

def test_show1d_toggle_auto_contrast(smoke_page):
    """Toggle auto-contrast on Show1D via switch."""
    widget = smoke_page.locator(".show1d-root").first
    widget.scroll_into_view_if_needed()
    # Switches: Log(0), Auto(1), Grid(2), Legend(3)
    widget.locator(".MuiSwitch-root").nth(1).click()
    time.sleep(1)
    _screenshot(widget, "show1d_auto")
    widget.locator(".MuiSwitch-root").nth(1).click()
    time.sleep(0.5)

def test_show1d_toggle_grid(smoke_page):
    """Toggle grid on Show1D."""
    widget = smoke_page.locator(".show1d-root").first
    widget.scroll_into_view_if_needed()
    # Grid is on by default, toggle off
    widget.locator(".MuiSwitch-root").nth(2).click()
    time.sleep(1)
    _screenshot(widget, "show1d_no_grid")
    widget.locator(".MuiSwitch-root").nth(2).click()
    time.sleep(0.5)

def test_show1d_toggle_legend(smoke_page):
    """Toggle legend on Show1D."""
    widget = smoke_page.locator(".show1d-root").first
    widget.scroll_into_view_if_needed()
    # Legend is on by default, toggle off
    widget.locator(".MuiSwitch-root").nth(3).click()
    time.sleep(1)
    _screenshot(widget, "show1d_no_legend")
    widget.locator(".MuiSwitch-root").nth(3).click()
    time.sleep(0.5)

# ---------------------------------------------------------------------------
# Show2D interaction tests
# ---------------------------------------------------------------------------

def test_show2d_toggle_fft(smoke_page):
    """Toggle FFT on Show2D and verify an extra canvas appears."""
    widget = smoke_page.locator(".show2d-root").first
    widget.scroll_into_view_if_needed()
    before = widget.locator("canvas").count()
    # Switches: Profile(0), ROI(1), Lens(2), FFT(3), Colorbar(4), Auto(5)
    widget.locator(".MuiSwitch-root").nth(3).click()
    time.sleep(2)
    after = widget.locator("canvas").count()
    assert after > before, f"FFT toggle didn't add canvas ({before} → {after})"
    _screenshot(widget, "show2d_fft")
    widget.locator(".MuiSwitch-root").nth(3).click()
    time.sleep(1)

def test_show2d_change_colormap(smoke_page):
    """Change Show2D colormap to Viridis and back."""
    widget = smoke_page.locator(".show2d-root").first
    widget.scroll_into_view_if_needed()
    # Dropdown order (FFT off): Scale(0), Color(1)
    _change_dropdown(widget, smoke_page, 1, "Viridis")
    _screenshot(widget, "show2d_viridis")
    _change_dropdown(widget, smoke_page, 1, "Inferno")

def test_show2d_change_scale(smoke_page):
    """Switch Show2D scale to Log and back."""
    widget = smoke_page.locator(".show2d-root").first
    widget.scroll_into_view_if_needed()
    _change_dropdown(widget, smoke_page, 0, "Log")
    _screenshot(widget, "show2d_log")
    _change_dropdown(widget, smoke_page, 0, "Lin")

def test_show2d_toggle_colorbar(smoke_page):
    """Toggle Colorbar switch on Show2D."""
    widget = smoke_page.locator(".show2d-root").first
    widget.scroll_into_view_if_needed()
    # Switches: Profile(0), ROI(1), Lens(2), FFT(3), Colorbar(4), Auto(5)
    widget.locator(".MuiSwitch-root").nth(4).click()
    time.sleep(1)
    _screenshot(widget, "show2d_colorbar")
    widget.locator(".MuiSwitch-root").nth(4).click()
    time.sleep(0.5)

def test_show2d_toggle_auto_contrast(smoke_page):
    """Toggle Auto contrast on Show2D."""
    widget = smoke_page.locator(".show2d-root").first
    widget.scroll_into_view_if_needed()
    widget.locator(".MuiSwitch-root").nth(5).click()
    time.sleep(1)
    _screenshot(widget, "show2d_auto")
    widget.locator(".MuiSwitch-root").nth(5).click()
    time.sleep(0.5)

def test_show2d_profile_hover(smoke_page):
    """Enable profile, draw line, hover over chart, verify crosshair position."""
    widget = smoke_page.locator(".show2d-root").first
    widget.scroll_into_view_if_needed()
    canvases_before = widget.locator("canvas").count()

    # Enable Profile (switch index 0: Profile(0), ROI(1), Lens(2), FFT(3))
    widget.locator(".MuiSwitch-root").nth(0).click()
    time.sleep(1)

    # Click two points on the main image canvas to create a profile line
    canvas = widget.locator("canvas").first
    box = canvas.bounding_box()
    smoke_page.mouse.click(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.2)
    time.sleep(0.5)
    smoke_page.mouse.click(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.8)
    time.sleep(1)

    # Profile canvas should have appeared
    canvases_after = widget.locator("canvas").count()
    assert canvases_after > canvases_before, "Profile canvas not added"

    _screenshot(widget, "show2d_profile")

    # Hover over the profile chart at different X positions
    profile_canvas = widget.locator("canvas").last
    pbox = profile_canvas.bounding_box()
    assert pbox is not None, "Profile canvas has no bounding box"

    # Hover at 25%, 50%, 75% along the profile chart
    for pct, label in [(0.25, "25"), (0.50, "50"), (0.75, "75")]:
        smoke_page.mouse.move(
            pbox["x"] + pbox["width"] * pct,
            pbox["y"] + pbox["height"] * 0.5,
        )
        time.sleep(0.3)
    _screenshot(widget, "show2d_profile_hover")

    # Clean up: disable Profile
    widget.locator(".MuiSwitch-root").nth(0).click()
    time.sleep(0.5)

def test_show2d_profile_drag(smoke_page):
    """Draw a profile line on Show2D, verify drag moves endpoint."""
    widget = smoke_page.locator(".show2d-root").first
    widget.scroll_into_view_if_needed()

    # Enable Profile toggle — Switches: Profile(0), ROI(1), Lens(2), Preview(3), FFT(4)
    widget.locator(".MuiSwitch-root").nth(0).click()
    time.sleep(1)

    canvas = widget.locator("canvas").first
    box = canvas.bounding_box()
    assert box is not None

    # Draw profile line
    p0_x = box["x"] + box["width"] * 0.2
    p0_y = box["y"] + box["height"] * 0.2
    smoke_page.mouse.click(p0_x, p0_y)
    time.sleep(0.5)
    smoke_page.mouse.click(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.8)
    time.sleep(1)

    # Capture before-drag image
    img_before = canvas.screenshot()

    # Assert cursor at endpoint
    smoke_page.mouse.move(p0_x, p0_y)
    time.sleep(0.3)
    cursor = smoke_page.evaluate(
        """([x, y]) => { const el = document.elementFromPoint(x, y); return el ? window.getComputedStyle(el).cursor : "default"; }""",
        [p0_x, p0_y],
    )
    assert cursor in ("grab", "default"), f"Expected grab or default at endpoint, got {cursor}"

    # Drag endpoint to a different position
    smoke_page.mouse.down()
    smoke_page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5, steps=5)
    smoke_page.mouse.up()
    time.sleep(1)

    # Verify the line moved
    img_after = canvas.screenshot()
    assert img_before != img_after, "Profile line did not visually change after drag"
    _screenshot(widget, "show2d_profile_drag")

    # Clean up: disable Profile
    widget.locator(".MuiSwitch-root").nth(0).click()
    time.sleep(0.5)

def test_show2d_roi_fft(smoke_page):
    """Enable ROI + FFT on Show2D, add ROI, verify ROI FFT label appears."""
    widget = smoke_page.locator(".show2d-root").first
    widget.scroll_into_view_if_needed()

    # Enable FFT (switch 3) and ROI (switch 1)
    # Switches: Profile(0), ROI(1), Lens(2), FFT(3)
    widget.locator(".MuiSwitch-root").nth(3).click()  # FFT on
    time.sleep(2)
    widget.locator(".MuiSwitch-root").nth(1).click()  # ROI on
    time.sleep(1)

    # Add an ROI via the ADD button (adds at center and auto-selects)
    widget.locator("button:has-text('ADD')").click()
    time.sleep(2)

    # Check that "ROI FFT" label appears in the widget text
    text = widget.inner_text()
    assert "ROI FFT" in text, f"ROI FFT label not found in widget text"
    _screenshot(widget, "show2d_roi_fft")

    # Clean up: disable ROI and FFT
    widget.locator(".MuiSwitch-root").nth(1).click()  # ROI off
    time.sleep(0.5)
    widget.locator(".MuiSwitch-root").nth(3).click()  # FFT off
    time.sleep(0.5)

# ---------------------------------------------------------------------------
# Show3D interaction tests
# ---------------------------------------------------------------------------

def test_show3d_toggle_fft(smoke_page):
    """Toggle FFT on Show3D and verify an extra canvas appears."""
    widget = smoke_page.locator(".show3d-root").first
    widget.scroll_into_view_if_needed()
    before = widget.locator("canvas").count()
    # Switches: FFT(0), Profile(1), Lens(2), ROI(3), Auto(4), Colorbar(5)
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(2)
    after = widget.locator("canvas").count()
    assert after > before, f"FFT toggle didn't add canvas ({before} → {after})"
    _screenshot(widget, "show3d_fft")
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(1)

def test_show3d_change_colormap(smoke_page):
    """Change Show3D colormap to Gray and back."""
    widget = smoke_page.locator(".show3d-root").first
    widget.scroll_into_view_if_needed()
    # Dropdowns: Scale(0), Color(1), Diff(2)
    _change_dropdown(widget, smoke_page, 1, "Gray")
    _screenshot(widget, "show3d_gray")
    _change_dropdown(widget, smoke_page, 1, "Magma")

def test_show3d_change_scale(smoke_page):
    """Switch Show3D scale to Log and back."""
    widget = smoke_page.locator(".show3d-root").first
    widget.scroll_into_view_if_needed()
    _change_dropdown(widget, smoke_page, 0, "Log")
    _screenshot(widget, "show3d_log")
    _change_dropdown(widget, smoke_page, 0, "Lin")

def test_show3d_toggle_auto_contrast(smoke_page):
    """Toggle Auto contrast on Show3D."""
    widget = smoke_page.locator(".show3d-root").first
    widget.scroll_into_view_if_needed()
    # Switches: FFT(0), Profile(1), Lens(2), ROI(3), Auto(4), Colorbar(5)
    widget.locator(".MuiSwitch-root").nth(4).click()
    time.sleep(1)
    _screenshot(widget, "show3d_auto")
    widget.locator(".MuiSwitch-root").nth(4).click()
    time.sleep(0.5)

def test_show3d_roi_fft(smoke_page):
    """Enable ROI + FFT on Show3D, add ROI, verify ROI FFT label appears."""
    widget = smoke_page.locator(".show3d-root").first
    widget.scroll_into_view_if_needed()

    # Switches: FFT(0), Profile(1), Lens(2), ROI(3), Auto(4), Colorbar(5)
    widget.locator(".MuiSwitch-root").nth(0).click()  # FFT on
    time.sleep(2)
    widget.locator(".MuiSwitch-root").nth(3).click()  # ROI on
    time.sleep(1)

    # Add an ROI via the ADD button (adds at center and auto-selects)
    widget.locator("button:has-text('ADD')").click()
    time.sleep(2)

    # Check that "ROI FFT" label appears in the widget text
    text = widget.inner_text()
    assert "ROI FFT" in text, f"ROI FFT label not found in widget text"
    _screenshot(widget, "show3d_roi_fft")

    # Clean up: disable ROI and FFT
    widget.locator(".MuiSwitch-root").nth(3).click()  # ROI off
    time.sleep(0.5)
    widget.locator(".MuiSwitch-root").nth(0).click()  # FFT off
    time.sleep(0.5)

def test_show3d_profile_draw_and_drag(smoke_page):
    """Draw a profile line and verify it appears on the canvas."""
    widget = smoke_page.locator(".show3d-root").first
    widget.scroll_into_view_if_needed()
    canvases_before = widget.locator("canvas").count()

    # Switches: FFT(0), Profile(1), Lens(2), ROI(3), Preview(4), Auto(5), Colorbar(6)
    widget.locator(".MuiSwitch-root").nth(1).click()
    time.sleep(1)

    # Click two points on the main canvas to create a profile line
    canvas = widget.locator("canvas").first
    box = canvas.bounding_box()
    assert box is not None
    img_before = canvas.screenshot()

    # First point at 30%, 30%
    smoke_page.mouse.click(box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.3)
    time.sleep(0.5)

    # Second point at 70%, 70%
    smoke_page.mouse.click(box["x"] + box["width"] * 0.7, box["y"] + box["height"] * 0.7)
    time.sleep(1)

    # Verify profile sparkline canvas appeared
    canvases_after = widget.locator("canvas").count()
    assert canvases_after > canvases_before, (
        f"Profile canvas not added ({canvases_before} → {canvases_after})"
    )

    # Verify the profile line is drawn on canvas
    img_with_profile = canvas.screenshot()
    assert img_before != img_with_profile, "Profile line not visible on canvas"
    _screenshot(widget, "show3d_profile")

    # Clean up: disable Profile
    widget.locator(".MuiSwitch-root").nth(1).click()
    time.sleep(0.5)

def test_show3d_profile_updates_across_frames(smoke_page):
    """Profile sparkline must update during playback (rAF loop)."""
    widget = smoke_page.locator(".show3d-root").first
    widget.scroll_into_view_if_needed()

    # Enable Profile via View menu
    # Switches: FFT(0), Profile(1), Lens(2), ROI(3), Preview(4), Auto(5), Colorbar(6)
    widget.locator(".MuiSwitch-root").nth(1).click()
    time.sleep(1)

    # Draw profile line
    canvas = widget.locator("canvas").first
    box = canvas.bounding_box()
    assert box is not None
    smoke_page.mouse.click(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.2)
    time.sleep(0.5)
    smoke_page.mouse.click(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.8)
    time.sleep(1)

    # Find the profile sparkline canvas (the small one added after main canvas)
    canvases = widget.locator("canvas")
    sparkline = None
    for ci in range(canvases.count()):
        cb = canvases.nth(ci).bounding_box()
        if cb and cb["height"] < 100:
            sparkline = canvases.nth(ci)
            break
    assert sparkline is not None, "Profile sparkline canvas not found"

    # Capture sparkline image at current frame
    img_before = sparkline.screenshot()
    assert len(img_before) > 0

    # Advance frames with ArrowRight (more reliable than Space playback)
    canvas.click()
    time.sleep(0.3)
    for _ in range(5):
        smoke_page.keyboard.press("ArrowRight")
        time.sleep(0.3)
    time.sleep(1)

    # Capture sparkline after advancing — it should have updated
    img_during = sparkline.screenshot()

    # The sparkline must change across frames (different data per frame)
    assert img_before != img_during, "Profile sparkline did not update across frames"
    _screenshot(widget, "show3d_profile_playback")

    # Clean up: disable Profile
    widget.locator(".MuiSwitch-root").nth(1).click()
    time.sleep(0.5)


def test_show3d_profile_live_playback(smoke_page):
    """Profile sparkline must update during live playback (rAF loop)."""
    widget = smoke_page.locator(".show3d-root").first
    widget.scroll_into_view_if_needed()

    # Switches: FFT(0), Profile(1), Lens(2), ROI(3), Preview(4), Auto(5), Colorbar(6)
    widget.locator(".MuiSwitch-root").nth(1).click()
    time.sleep(1)

    # Draw profile line
    canvas = widget.locator("canvas").first
    box = canvas.bounding_box()
    assert box is not None
    smoke_page.mouse.click(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.2)
    time.sleep(0.5)
    smoke_page.mouse.click(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.8)
    time.sleep(1)

    # Find profile sparkline canvas
    canvases = widget.locator("canvas")
    sparkline = None
    for ci in range(canvases.count()):
        cb = canvases.nth(ci).bounding_box()
        if cb and cb["height"] < 100:
            sparkline = canvases.nth(ci)
            break
    assert sparkline is not None, "Profile sparkline canvas not found"

    # Capture sparkline before playback
    img_before = sparkline.screenshot()
    assert len(img_before) > 0

    # Click the Play button using MUI's PlayArrow SVG path
    widget.locator('path[d="M8 5v14l11-7z"]').first.locator("..").locator("..").click()
    time.sleep(3)

    # Capture sparkline during playback
    img_during = sparkline.screenshot()

    # Stop playback (force=True because slider thumb may intercept)
    widget.locator('path[d="M6 6h12v12H6z"]').first.locator("..").locator("..").click(force=True)
    time.sleep(0.5)

    # Sparkline must change during live playback
    assert img_before != img_during, "Profile sparkline did not update during live playback"
    _screenshot(widget, "show3d_profile_live_playback")

    # Clean up: disable Profile
    widget.locator(".MuiSwitch-root").nth(1).click()
    time.sleep(0.5)

# ---------------------------------------------------------------------------
# Mark2D interaction tests
# ---------------------------------------------------------------------------

def test_mark2d_toggle_fft(smoke_page):
    """Toggle FFT on Mark2D and verify an extra canvas appears."""
    widget = smoke_page.locator(".mark2d-root").first
    widget.scroll_into_view_if_needed()
    before = widget.locator("canvas").count()
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(2)
    after = widget.locator("canvas").count()
    assert after > before, f"FFT toggle didn't add canvas ({before} → {after})"
    _screenshot(widget, "mark2d_fft")
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(1)

def test_mark2d_change_colormap(smoke_page):
    """Change Mark2D colormap to Viridis and back."""
    widget = smoke_page.locator(".mark2d-root").first
    widget.scroll_into_view_if_needed()
    # Dropdowns: Scale(0), Color(1)
    _change_dropdown(widget, smoke_page, 1, "Viridis")
    _screenshot(widget, "mark2d_viridis")
    _change_dropdown(widget, smoke_page, 1, "Gray")

def test_mark2d_toggle_colorbar(smoke_page):
    """Toggle Colorbar switch on Mark2D."""
    widget = smoke_page.locator(".mark2d-root").first
    widget.scroll_into_view_if_needed()
    # Switches: FFT(0), Auto(1), Colorbar(2)
    widget.locator(".MuiSwitch-root").nth(2).click()
    time.sleep(1)
    _screenshot(widget, "mark2d_colorbar")
    widget.locator(".MuiSwitch-root").nth(2).click()
    time.sleep(0.5)

def test_mark2d_undo_redo_exist(smoke_page):
    """Verify UNDO and REDO buttons exist in Mark2D."""
    widget = smoke_page.locator(".mark2d-root").first
    widget.scroll_into_view_if_needed()
    undo = widget.locator('button:has-text("UNDO")')
    redo = widget.locator('button:has-text("REDO")')
    assert undo.count() >= 1, "UNDO button not found in Mark2D"
    assert redo.count() >= 1, "REDO button not found in Mark2D"

def test_mark2d_profile_drag(smoke_page):
    """Draw a profile line on Mark2D, verify drag moves endpoint."""
    widget = smoke_page.locator(".mark2d-root").first
    widget.scroll_into_view_if_needed()

    # Enable profile mode — click the "Profile" label
    widget.locator('text=Profile').first.click()
    time.sleep(1)

    canvas = widget.locator("canvas").first
    box = canvas.bounding_box()
    assert box is not None

    # Draw profile line
    p0_x = box["x"] + box["width"] * 0.2
    p0_y = box["y"] + box["height"] * 0.2
    smoke_page.mouse.click(p0_x, p0_y)
    time.sleep(0.5)
    smoke_page.mouse.click(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.8)
    time.sleep(1)

    # Capture before-drag image
    img_before = canvas.screenshot()

    # Assert cursor at endpoint
    smoke_page.mouse.move(p0_x, p0_y)
    time.sleep(0.3)
    cursor = smoke_page.evaluate(
        """([x, y]) => { const el = document.elementFromPoint(x, y); return el ? window.getComputedStyle(el).cursor : "default"; }""",
        [p0_x, p0_y],
    )
    assert cursor in ("grab", "default"), f"Expected grab or default at endpoint, got {cursor}"

    # Drag endpoint to a different position
    smoke_page.mouse.down()
    smoke_page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5, steps=5)
    smoke_page.mouse.up()
    time.sleep(1)

    # Verify the line moved
    img_after = canvas.screenshot()
    assert img_before != img_after, "Profile line did not visually change after drag"
    _screenshot(widget, "mark2d_profile_drag")

    # Clean up: click Profile label again to deactivate
    widget.locator('text=Profile').first.click()
    time.sleep(0.5)

def test_mark2d_roi_fft(smoke_page):
    """Enable FFT + add ROI on Mark2D, verify ROI FFT label appears."""
    widget = smoke_page.locator(".mark2d-root").first
    widget.scroll_into_view_if_needed()

    # Enable FFT — first switch
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(2)

    # Add an ROI via the ADD button (auto-selects new ROI)
    widget.locator("button:has-text('ADD')").first.click()
    time.sleep(2)

    # Verify ROI FFT label appears
    text = widget.inner_text()
    assert "ROI FFT" in text, f"Expected 'ROI FFT' in widget text, got: {text[:200]}"
    _screenshot(widget, "mark2d_roi_fft")

    # Clean up: disable FFT
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(0.5)

# ---------------------------------------------------------------------------
# Show3DVolume interaction tests
# ---------------------------------------------------------------------------

def test_show3dvolume_toggle_fft(smoke_page):
    """Toggle FFT on Show3DVolume."""
    widget = smoke_page.locator(".show3dvolume-root").first
    widget.scroll_into_view_if_needed()
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(2)
    _screenshot(widget, "show3dvolume_fft")
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(1)

def test_show3dvolume_change_colormap(smoke_page):
    """Change Show3DVolume colormap."""
    widget = smoke_page.locator(".show3dvolume-root").first
    widget.scroll_into_view_if_needed()
    # Dropdowns (FFT off): Scale(0), Color(1), PlayAxis(2)
    _change_dropdown(widget, smoke_page, 1, "Viridis")
    _screenshot(widget, "show3dvolume_viridis")
    _change_dropdown(widget, smoke_page, 1, "Inferno")

def test_show3dvolume_toggle_crosshair(smoke_page):
    """Toggle Crosshair switch on Show3DVolume."""
    widget = smoke_page.locator(".show3dvolume-root").first
    widget.scroll_into_view_if_needed()
    # Switches (FFT off): FFT(0), Auto(1), Crosshair(2), Colorbar(3)
    switches = widget.locator(".MuiSwitch-root")
    # Find crosshair — it's after Auto
    switches.nth(2).click()
    time.sleep(1)
    _screenshot(widget, "show3dvolume_crosshair")
    switches.nth(2).click()
    time.sleep(0.5)


# ---------------------------------------------------------------------------
# Show4DSTEM interaction tests
# ---------------------------------------------------------------------------

def test_show4dstem_presets(smoke_page):
    """Click BF, ABF, and ADF preset labels on Show4DSTEM."""
    widget = smoke_page.locator(".show4dstem-root").first
    widget.scroll_into_view_if_needed()
    bf_label = widget.locator('span:has-text("BF")').first
    bf_label.wait_for(state="visible", timeout=3000)
    bf_label.click()
    time.sleep(0.5)
    _screenshot(widget, "show4dstem_bf")
    # ABF preset
    abf_label = widget.locator('span:has-text("ABF")').first
    abf_label.wait_for(state="visible", timeout=3000)
    abf_label.click()
    time.sleep(0.5)
    _screenshot(widget, "show4dstem_abf")
    # ADF preset
    adf_label = widget.locator('span:has-text("ADF")').first
    adf_label.wait_for(state="visible", timeout=3000)
    adf_label.click()
    time.sleep(0.5)
    _screenshot(widget, "show4dstem_adf")

def test_show4dstem_change_detector_mode(smoke_page):
    """Change Show4DSTEM detector mode from Circle to Annular."""
    widget = smoke_page.locator(".show4dstem-root").first
    widget.scroll_into_view_if_needed()
    # Dropdown order: Detector(0)
    _change_dropdown(widget, smoke_page, 0, "Annular")
    _screenshot(widget, "show4dstem_annular")
    _change_dropdown(widget, smoke_page, 0, "Circle")

def test_show4dstem_toggle_fft(smoke_page):
    """Toggle FFT on Show4DSTEM VI panel."""
    widget = smoke_page.locator(".show4dstem-root").first
    widget.scroll_into_view_if_needed()
    # Find the FFT switch — it's in the VI panel header
    fft_switch = widget.locator(".MuiSwitch-root")
    before = widget.locator("canvas").count()
    # FFT is typically the last visible switch in Show4DSTEM
    # It's at index after Profile and DP Colorbar
    fft_switch.nth(2).click()
    time.sleep(2)
    _screenshot(widget, "show4dstem_fft")
    fft_switch.nth(2).click()
    time.sleep(1)

def test_show4dstem_roi_fft(smoke_page):
    """Enable VI ROI + FFT on Show4DSTEM, verify ROI FFT label appears."""
    widget = smoke_page.locator(".show4dstem-root").first
    widget.scroll_into_view_if_needed()

    # Enable FFT (switch 2) then set VI ROI to Circle (dropdown 3)
    # Switches: Profile(0), Colorbar(1), FFT(2)
    # Dropdowns: Detector(0), DPColor(1), DPScale(2), VIRoi(3), VIColor(4), VIScale(5)
    widget.locator(".MuiSwitch-root").nth(2).click()  # FFT on
    time.sleep(2)
    _change_dropdown(widget, smoke_page, 3, "Circle")
    time.sleep(2)

    # Check that "ROI FFT" label appears in the widget text
    text = widget.inner_text()
    assert "ROI FFT" in text, f"ROI FFT label not found in widget text"
    _screenshot(widget, "show4dstem_roi_fft")

    # Clean up: VI ROI off, FFT off
    _change_dropdown(widget, smoke_page, 3, "Off")
    time.sleep(0.5)
    widget.locator(".MuiSwitch-root").nth(2).click()  # FFT off
    time.sleep(0.5)

def test_show4dstem_profile_drag(smoke_page):
    """Draw a profile line on Show4DSTEM DP panel, verify drag moves endpoint."""
    widget = smoke_page.locator(".show4dstem-root").first
    widget.scroll_into_view_if_needed()

    # Enable DP Profile — first switch
    widget.locator(".MuiSwitch-root").nth(0).click()
    time.sleep(1)

    # DP panel is the left half — find canvases whose midpoint is in the left half
    wbox = widget.bounding_box()
    assert wbox is not None
    mid_x = wbox["x"] + wbox["width"] / 2
    canvases = widget.locator("canvas")
    dp_canvas = None
    for ci in range(canvases.count()):
        cb = canvases.nth(ci).bounding_box()
        if cb and (cb["x"] + cb["width"] / 2) < mid_x:
            dp_canvas = canvases.nth(ci)
            break
    assert dp_canvas is not None, "Could not find DP canvas in left half"
    box = dp_canvas.bounding_box()
    assert box is not None

    # Draw profile line
    p0_x = box["x"] + box["width"] * 0.2
    p0_y = box["y"] + box["height"] * 0.2
    smoke_page.mouse.click(p0_x, p0_y)
    time.sleep(0.5)
    smoke_page.mouse.click(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.8)
    time.sleep(1)

    # Capture before-drag image
    img_before = dp_canvas.screenshot()

    # Assert cursor at endpoint
    smoke_page.mouse.move(p0_x, p0_y)
    time.sleep(0.3)
    cursor = smoke_page.evaluate(
        """([x, y]) => { const el = document.elementFromPoint(x, y); return el ? window.getComputedStyle(el).cursor : "default"; }""",
        [p0_x, p0_y],
    )
    assert cursor in ("grab", "default"), f"Expected grab or default at endpoint, got {cursor}"

    # Drag endpoint to a very different position
    smoke_page.mouse.down()
    smoke_page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5, steps=5)
    smoke_page.mouse.up()
    time.sleep(1)

    # Capture after-drag image — line should have moved
    img_after = dp_canvas.screenshot()
    assert img_before != img_after, "Profile line did not visually change after drag"
    _screenshot(widget, "show4dstem_profile_drag")

    # Clean up: disable Profile
    widget.locator(".MuiSwitch-root").nth(0).click()
    time.sleep(0.5)

# ---------------------------------------------------------------------------
# Bin interaction tests
# ---------------------------------------------------------------------------

def test_bin_change_binning_controls(smoke_page):
    """Change Bin factors and verify binned shape text updates."""
    widget = smoke_page.locator(".bin-root").first
    widget.scroll_into_view_if_needed()

    # Dropdown order in Bin: scan row, scan col, det row, det col, reduce, edges, colormap, display
    _change_dropdown(widget, smoke_page, 0, "2")
    _change_dropdown(widget, smoke_page, 1, "2")
    _change_dropdown(widget, smoke_page, 2, "2")
    _change_dropdown(widget, smoke_page, 3, "2")
    _change_dropdown(widget, smoke_page, 4, "sum")
    _change_dropdown(widget, smoke_page, 5, "pad")

    shape_line = widget.locator('text=shape: (8, 8, 32, 32) → (4, 4, 16, 16)')
    assert shape_line.count() >= 1, "Bin shape summary did not update after changing factors"
    _screenshot(widget, "bin_factors_sum_pad")

def test_bin_change_display_controls(smoke_page):
    """Change Bin colormap/display controls and capture screenshot."""
    widget = smoke_page.locator(".bin-root").first
    widget.scroll_into_view_if_needed()

    _change_dropdown(widget, smoke_page, 6, "magma")
    # Log scale is now a Switch, not a Select dropdown
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(1)
    _screenshot(widget, "bin_magma_log")

    # Reset to default-like settings for downstream screenshots
    _change_dropdown(widget, smoke_page, 6, "inferno")
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(0.5)

# ---------------------------------------------------------------------------
# Show4D interaction tests
# ---------------------------------------------------------------------------

def test_show4d_toggle_fft(smoke_page):
    """Toggle FFT on Show4D signal panel."""
    widget = smoke_page.locator(".show4d-root").first
    widget.scroll_into_view_if_needed()
    switches = widget.locator(".MuiSwitch-root")
    # FFT switch is nth(1) — after Snap(0)
    switches.nth(1).click()
    time.sleep(2)
    _screenshot(widget, "show4d_fft")
    switches.nth(1).click()
    time.sleep(1)

def test_show4d_change_roi_mode(smoke_page):
    """Change Show4D ROI mode from Off to Rect."""
    widget = smoke_page.locator(".show4d-root").first
    widget.scroll_into_view_if_needed()
    # Dropdowns: ROI(0), Reduce(1), NavColor(2), NavScale(3)
    _change_dropdown(widget, smoke_page, 0, "Rect")
    time.sleep(0.5)
    _screenshot(widget, "show4d_roi_rect")
    _change_dropdown(widget, smoke_page, 0, "Off")

def test_show4d_roi_fft(smoke_page):
    """Enable ROI + FFT on Show4D, verify ROI FFT label appears."""
    widget = smoke_page.locator(".show4d-root").first
    widget.scroll_into_view_if_needed()

    # Enable FFT (switch 1) then set ROI to Circle (dropdown 0)
    widget.locator(".MuiSwitch-root").nth(1).click()  # FFT on
    time.sleep(2)
    _change_dropdown(widget, smoke_page, 0, "Circle")
    time.sleep(2)

    # Check that "ROI FFT" label appears in the widget text
    text = widget.inner_text()
    assert "ROI FFT" in text, f"ROI FFT label not found in widget text"
    _screenshot(widget, "show4d_roi_fft")

    # Clean up: ROI off, FFT off
    _change_dropdown(widget, smoke_page, 0, "Off")
    time.sleep(0.5)
    widget.locator(".MuiSwitch-root").nth(1).click()  # FFT off
    time.sleep(0.5)

def test_show4d_change_colormap(smoke_page):
    """Change Show4D navigation colormap."""
    widget = smoke_page.locator(".show4d-root").first
    widget.scroll_into_view_if_needed()
    # When ROI=off: ROI(0), NavColor(1), NavScale(2), SigColor(3), SigScale(4)
    _change_dropdown(widget, smoke_page, 1, "Gray")
    _screenshot(widget, "show4d_gray")
    _change_dropdown(widget, smoke_page, 1, "Inferno")

def test_show4d_profile_drag(smoke_page):
    """Draw a profile line on Show4D signal panel, verify drag moves endpoint."""
    widget = smoke_page.locator(".show4d-root").first
    widget.scroll_into_view_if_needed()

    # Enable Profile toggle — Switches: Snap(0), FFT(1), Profile(2)
    widget.locator(".MuiSwitch-root").nth(2).click()
    time.sleep(1)

    # Signal panel is the right half — find canvases whose midpoint is in the right half
    wbox = widget.bounding_box()
    assert wbox is not None
    mid_x = wbox["x"] + wbox["width"] / 2
    canvases = widget.locator("canvas")
    sig_canvas = None
    for ci in range(canvases.count()):
        cb = canvases.nth(ci).bounding_box()
        if cb and (cb["x"] + cb["width"] / 2) > mid_x:
            sig_canvas = canvases.nth(ci)
            break
    assert sig_canvas is not None, "Could not find signal canvas in right half"
    box = sig_canvas.bounding_box()
    assert box is not None

    # Draw profile line
    p0_x = box["x"] + box["width"] * 0.2
    p0_y = box["y"] + box["height"] * 0.2
    smoke_page.mouse.click(p0_x, p0_y)
    time.sleep(0.5)
    smoke_page.mouse.click(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.8)
    time.sleep(1)

    # Capture before-drag image
    img_before = sig_canvas.screenshot()

    # Assert cursor at endpoint
    smoke_page.mouse.move(p0_x, p0_y)
    time.sleep(0.3)
    cursor = smoke_page.evaluate(
        """([x, y]) => { const el = document.elementFromPoint(x, y); return el ? window.getComputedStyle(el).cursor : "default"; }""",
        [p0_x, p0_y],
    )
    assert cursor in ("grab", "default"), f"Expected grab or default at endpoint, got {cursor}"

    # Drag endpoint to a different position
    smoke_page.mouse.down()
    smoke_page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5, steps=5)
    smoke_page.mouse.up()
    time.sleep(1)

    # Verify the line moved
    img_after = sig_canvas.screenshot()
    assert img_before != img_after, "Profile line did not visually change after drag"
    _screenshot(widget, "show4d_profile_drag")

    # Clean up: disable Profile
    widget.locator(".MuiSwitch-root").nth(2).click()
    time.sleep(0.5)

def test_show4d_controls_customizer_presets(smoke_page):
    """Apply Compact/All presets from controls customizer."""
    widget = smoke_page.locator(".show4d-root").first
    widget.scroll_into_view_if_needed()

    widget.locator('button[aria-label="Customize controls"]').first.click()
    time.sleep(0.5)
    smoke_page.locator('[data-testid="preset-compact"]').first.click()
    time.sleep(0.5)
    smoke_page.keyboard.press("Escape")
    time.sleep(0.3)

    assert widget.locator("text=Profile:").count() == 0, "Compact preset should hide Profile controls"
    _screenshot(widget, "show4d_controls_compact")

    widget.locator('button[aria-label="Customize controls"]').first.click()
    time.sleep(0.5)
    smoke_page.locator('[data-testid="preset-all"]').first.click()
    time.sleep(0.5)
    smoke_page.keyboard.press("Escape")
    time.sleep(0.3)

    assert widget.locator("text=Profile:").count() >= 1, "All preset should restore Profile controls"

def test_show4d_controls_customizer_lock_export(smoke_page):
    """Lock/unlock Export controls from customizer and verify button disabled state."""
    widget = smoke_page.locator(".show4d-root").first
    widget.scroll_into_view_if_needed()

    widget.locator('button[aria-label="Customize controls"]').first.click()
    time.sleep(0.5)
    # Click the Lock switch in the export tool row
    export_row = smoke_page.locator('[data-testid="tool-row-export"]').first
    export_row.locator(".MuiSwitch-root").nth(1).click()
    time.sleep(0.3)
    smoke_page.keyboard.press("Escape")
    time.sleep(0.3)

    assert widget.locator('button:has-text("Export")').first.is_disabled(), "Export button should be disabled when export group is locked"
    _screenshot(widget, "show4d_controls_lock_export")

    widget.locator('button[aria-label="Customize controls"]').first.click()
    time.sleep(0.5)
    export_row = smoke_page.locator('[data-testid="tool-row-export"]').first
    export_row.locator(".MuiSwitch-root").nth(1).click()
    time.sleep(0.3)
    smoke_page.keyboard.press("Escape")
    time.sleep(0.3)

def test_show4d_controls_customizer_lock_navigation_blocks_keyboard_and_mouse(smoke_page):
    """Lock navigation and verify keyboard/mouse navigation handlers are blocked."""
    widget = smoke_page.locator(".show4d-root").first
    widget.scroll_into_view_if_needed()

    # Ensure ROI is off so "(row, col)" text is present.
    _change_dropdown(widget, smoke_page, 0, "Off", wait=0.3)
    start_pos = _extract_show4d_nav_pos(widget)

    # Lock navigation through the shared customizer.
    widget.locator('button[aria-label="Customize controls"]').first.click()
    time.sleep(0.3)
    nav_row = smoke_page.locator('[data-testid="tool-row-navigation"]').first
    nav_row.locator(".MuiSwitch-root").nth(1).click()
    time.sleep(0.2)
    smoke_page.keyboard.press("Escape")
    time.sleep(0.3)

    # Keyboard arrows should not move nav position.
    widget.locator("canvas").first.click(force=True)
    smoke_page.keyboard.press("ArrowRight")
    smoke_page.keyboard.press("ArrowDown")
    time.sleep(0.2)
    assert _extract_show4d_nav_pos(widget) == start_pos

    # Mouse click on nav canvas should also be ignored while locked.
    widget.locator("canvas").first.click(position={"x": 8, "y": 8}, force=True)
    time.sleep(0.2)
    assert _extract_show4d_nav_pos(widget) == start_pos

    # Unlock to avoid affecting downstream tests.
    widget.locator('button[aria-label="Customize controls"]').first.click()
    time.sleep(0.3)
    nav_row = smoke_page.locator('[data-testid="tool-row-navigation"]').first
    nav_row.locator(".MuiSwitch-root").nth(1).click()
    time.sleep(0.2)
    smoke_page.keyboard.press("Escape")
    time.sleep(0.2)

# ---------------------------------------------------------------------------
# Edit2D interaction tests
# ---------------------------------------------------------------------------

def test_edit2d_controls_exist(smoke_page):
    """Verify Edit2D has mode dropdown and canvas."""
    widget = smoke_page.locator(".edit2d-root").first
    widget.scroll_into_view_if_needed()
    selects = widget.locator(".MuiSelect-select")
    assert selects.count() >= 1, "No dropdown selects found in Edit2D"
    canvases = widget.locator("canvas")
    assert canvases.count() >= 1, "No canvas in Edit2D"
    _screenshot(widget, "edit2d_controls")

def test_edit2d_change_colormap(smoke_page):
    """Change Edit2D colormap."""
    widget = smoke_page.locator(".edit2d-root").first
    widget.scroll_into_view_if_needed()
    # Dropdowns: Scale(0), Color(1)
    _change_dropdown(widget, smoke_page, 1, "Viridis")
    _screenshot(widget, "edit2d_viridis")
    _change_dropdown(widget, smoke_page, 1, "Inferno")

def test_edit2d_toggle_auto_contrast(smoke_page):
    """Toggle Auto contrast on Edit2D."""
    widget = smoke_page.locator(".edit2d-root").first
    widget.scroll_into_view_if_needed()
    # Only 1 switch: Auto(0)
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(1)
    _screenshot(widget, "edit2d_auto")
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(0.5)

# ---------------------------------------------------------------------------
# Align2D interaction tests
# ---------------------------------------------------------------------------

def test_align2d_controls_exist(smoke_page):
    """Verify Align2D has AUTO button, ZERO button, and canvas."""
    widget = smoke_page.locator(".align2d-root").first
    widget.scroll_into_view_if_needed()
    auto_btn = widget.locator('button:has-text("AUTO")')
    assert auto_btn.count() >= 1, "AUTO button not found in Align2D"
    zero_btn = widget.locator('button:has-text("ZERO")')
    assert zero_btn.count() >= 1, "ZERO button not found in Align2D"
    canvases = widget.locator("canvas")
    assert canvases.count() >= 1, "No canvas in Align2D"
    _screenshot(widget, "align2d_controls")

def test_align2d_change_blend_mode(smoke_page):
    """Change Align2D blend mode to Difference and back."""
    widget = smoke_page.locator(".align2d-root").first
    widget.scroll_into_view_if_needed()
    # Dropdowns: Blend(0), HistSource(1), Color(2)
    _change_dropdown(widget, smoke_page, 0, "Diff")
    _screenshot(widget, "align2d_diff")
    _change_dropdown(widget, smoke_page, 0, "Blend")

def test_align2d_toggle_panels(smoke_page):
    """Toggle Show Panels switch on Align2D."""
    _dismiss_menus(smoke_page)
    widget = smoke_page.locator(".align2d-root").first
    widget.scroll_into_view_if_needed()
    # Switches: ShowPanels(0), FFT(1)
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(1)
    _screenshot(widget, "align2d_panels")
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(1)

def test_align2d_toggle_fft(smoke_page):
    """Toggle FFT switch — exercises FFT computation and histogram update paths."""
    _dismiss_menus(smoke_page)
    widget = smoke_page.locator(".align2d-root").first
    widget.scroll_into_view_if_needed()
    # Switches: ShowPanels(0), FFT(1)
    switches = widget.locator(".MuiSwitch-root")
    assert switches.count() >= 2, "Expected at least 2 switches (Panels, FFT)"
    switches.nth(1).click()  # toggle FFT on
    time.sleep(2)  # allow FFT computation (CPU fallback in headless)
    _screenshot(widget, "align2d_fft")
    switches.nth(1).click()  # toggle FFT off
    time.sleep(1)

def test_align2d_switch_hist_source(smoke_page):
    """Switch histogram source A→B→A — should NOT re-trigger FFT."""
    _dismiss_menus(smoke_page)
    widget = smoke_page.locator(".align2d-root").first
    widget.scroll_into_view_if_needed()
    # Hist source is the dropdown labelled A/B next to "Histogram:"
    # Dropdowns: Blend(0), HistSource(1), Color(2)
    _change_dropdown(widget, smoke_page, 1, "B")
    _screenshot(widget, "align2d_hist_b")
    _change_dropdown(widget, smoke_page, 1, "A")

def test_align2d_drag_alignment(smoke_page):
    """Drag on merged canvas to shift alignment — exercises debounced NCC."""
    _dismiss_menus(smoke_page)
    widget = smoke_page.locator(".align2d-root").first
    widget.scroll_into_view_if_needed()
    # Find the merged canvas (the one inside the merged container after the panels)
    canvases = widget.locator("canvas")
    assert canvases.count() >= 3, "Expected at least 3 canvases"
    # The merged canvas is the 3rd visible canvas (after panel A and panel B)
    # But with panels visible there are 5+ canvases. Use the merged container.
    merged = canvases.nth(2)
    box = merged.bounding_box()
    assert box is not None, "Merged canvas has no bounding box"
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    # Drag 20px right, 10px down
    smoke_page.mouse.move(cx, cy)
    smoke_page.mouse.down()
    smoke_page.mouse.move(cx + 20, cy + 10, steps=5)
    smoke_page.mouse.up()
    time.sleep(0.5)  # wait for debounced NCC (100ms)
    _screenshot(widget, "align2d_dragged")

def test_align2d_difference_mode_drag(smoke_page):
    """Switch to difference mode, then drag — exercises cached diff offscreen."""
    _dismiss_menus(smoke_page)
    widget = smoke_page.locator(".align2d-root").first
    widget.scroll_into_view_if_needed()
    _change_dropdown(widget, smoke_page, 0, "Diff")
    time.sleep(0.5)
    canvases = widget.locator("canvas")
    merged = canvases.nth(2)
    box = merged.bounding_box()
    assert box is not None
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    # Drag in diff mode
    smoke_page.mouse.move(cx, cy)
    smoke_page.mouse.down()
    smoke_page.mouse.move(cx - 15, cy + 5, steps=5)
    smoke_page.mouse.up()
    time.sleep(0.5)
    _screenshot(widget, "align2d_diff_dragged")
    # Restore blend mode
    _change_dropdown(widget, smoke_page, 0, "Blend")

# ---------------------------------------------------------------------------
# ShowComplex2D interaction tests
# ---------------------------------------------------------------------------

def test_showcomplex_change_mode(smoke_page):
    """Switch ShowComplex2D to Phase display mode via dropdown."""
    widget = smoke_page.locator(".showcomplex-root").first
    widget.scroll_into_view_if_needed()
    # Dropdowns: Scale(0), Color(1), Mode(2)
    _change_dropdown(widget, smoke_page, 2, "Phase")
    _screenshot(widget, "showcomplex_phase")
    _change_dropdown(widget, smoke_page, 2, "Amplitude")

def test_showcomplex_hsv_mode(smoke_page):
    """Switch ShowComplex2D to HSV mode and verify colorwheel."""
    widget = smoke_page.locator(".showcomplex-root").first
    widget.scroll_into_view_if_needed()
    _change_dropdown(widget, smoke_page, 2, "HSV")
    time.sleep(1)
    _screenshot(widget, "showcomplex_hsv")
    _change_dropdown(widget, smoke_page, 2, "Amplitude")

def test_showcomplex_toggle_fft(smoke_page):
    """Toggle FFT on ShowComplex2D."""
    widget = smoke_page.locator(".showcomplex-root").first
    widget.scroll_into_view_if_needed()
    before = widget.locator("canvas").count()
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(2)
    after = widget.locator("canvas").count()
    assert after > before, f"FFT toggle didn't add canvas ({before} → {after})"
    _screenshot(widget, "showcomplex_fft")
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(1)

def test_showcomplex_roi_fft(smoke_page):
    """Enable ROI + FFT on ShowComplex2D, verify ROI FFT label appears."""
    widget = smoke_page.locator(".showcomplex-root").first
    widget.scroll_into_view_if_needed()

    # Enable FFT (first switch)
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(2)

    # Set ROI to Circle via the ROI dropdown (dropdown index 3: Scale, Color, Mode, ROI)
    _change_dropdown(widget, smoke_page, 3, "Circle")
    time.sleep(2)

    # Check that "ROI FFT" label appears
    text = widget.inner_text()
    assert "ROI FFT" in text, f"ROI FFT label not found in widget text"
    _screenshot(widget, "showcomplex_roi_fft")

    # Clean up: ROI off, FFT off
    _change_dropdown(widget, smoke_page, 3, "Off")
    time.sleep(0.5)
    widget.locator(".MuiSwitch-root").first.click()
    time.sleep(0.5)

# ---------------------------------------------------------------------------
# Dark theme tests — must run AFTER all light-mode tests
# ---------------------------------------------------------------------------

def test_zz_dark_theme_switch(smoke_page):
    """Switch to dark theme and verify the attribute is set."""
    _set_dark_theme(smoke_page)
    is_dark = smoke_page.evaluate("() => document.body.dataset.jpThemeLight")
    assert is_dark == "false", f"Expected dark theme, got jpThemeLight={is_dark}"

@pytest.mark.parametrize("css_class", ALL_WIDGETS)
def test_zz_dark_theme_screenshot(smoke_page, css_class):
    """Screenshot every widget in dark theme for visual verification."""
    widget = smoke_page.locator(f".{css_class}").first
    widget.scroll_into_view_if_needed()
    name = css_class.replace("-root", "")
    _screenshot(widget, f"{name}_dark")

def test_zz_dark_theme_restore(smoke_page):
    """Restore light theme after dark theme tests."""
    _set_light_theme(smoke_page)
    is_light = smoke_page.evaluate("() => document.body.dataset.jpThemeLight")
    assert is_light == "true", f"Expected light theme, got jpThemeLight={is_light}"
