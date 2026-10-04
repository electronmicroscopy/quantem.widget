"""Opt-in Show3D exported-HTML playback regression tests.

Run after ``npm run build`` with ``QUANTEM_TEST_PLAYBACK=1 pytest -q
tests/show3d/test_playback_browser.py``. Requires Playwright Chromium and WebGPU.
Set ``QUANTEM_SHOW3D_PLAYBACK_HTML`` to test an existing multi-frame export
instead of generating a small moving-pattern fixture. Set
``QUANTEM_HEADLESS=1`` for CI; headed is the default. Screenshots go to pytest's
temporary directory. These are behavior checks, not an FPS benchmark.
"""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import re
import time

import pytest
from PIL import Image, ImageChops


pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_TEST_PLAYBACK") != "1"
    and not os.environ.get("QUANTEM_SHOW3D_PLAYBACK_HTML"),
    reason="Opt in with QUANTEM_TEST_PLAYBACK=1 (requires browser/WebGPU)",
)


@pytest.fixture(scope="module")
def export_path(tmp_path_factory):
    existing = os.environ.get("QUANTEM_SHOW3D_PLAYBACK_HTML")
    if existing:
        path = Path(existing).expanduser().resolve()
        assert path.is_file(), f"Export does not exist: {path}"
        return path

    import numpy as np
    from quantem.widget import Show3D

    row, col = np.mgrid[:96, :96]
    stack = np.stack([
        np.sin((col - frame * 2) / 7) + np.cos((row + frame) / 5)
        for frame in range(24)
    ]).astype(np.float32)
    path = tmp_path_factory.mktemp("playback-export") / "playback.html"
    Show3D(
        stack, stack[:, ::-1, :].copy(), panel_titles=["Raw", "Comparison"],
        fps=6, loop=True, save_state=False, verbose=False,
    ).export_html(path)
    return path


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=os.environ.get("QUANTEM_HEADLESS") == "1",
            executable_path=os.environ.get("CHROME_EXECUTABLE"),
        )
        yield browser
        browser.close()


@pytest.fixture
def page(browser, export_path, tmp_path):
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(export_path.as_uri(), wait_until="domcontentloaded")
    page.get_by_role("button", name="Play", exact=True).wait_for(timeout=60000)
    assert page.evaluate("async () => !!(await navigator.gpu?.requestAdapter())"), (
        "WebGPU is unavailable. Run headed Chromium on a GPU-capable host; "
        "CPU fallback does not qualify this playback check."
    )
    # The accessible image canvas has the GPU presentation canvas immediately
    # above it. Do not silently accept CPU fallback as production coverage.
    page.wait_for_function("""() => {
        const image = document.querySelector('canvas[aria-label^="Slice image"]');
        const gpu = image?.nextElementSibling;
        return gpu?.tagName === 'CANVAS' && getComputedStyle(gpu).opacity === '1';
    }""", timeout=30000)
    fps = page.get_by_role("slider", name="Playback frames per second", exact=True)
    fps.focus()
    fps.press("Home")
    for _ in range(5):
        fps.press("ArrowRight")
    fps.press("Tab")
    yield page
    page.screenshot(path=str(tmp_path / "playback-final.png"))
    page.close()
    assert not errors, f"Browser errors: {errors}"


def _sliders(page):
    return page.get_by_role("slider", name=re.compile("Loop range and current"))


def _frame(page):
    # Read the visible frame counter, not the Python trait (which is intentionally
    # committed less often than GPU paints).
    text = page.locator("[data-show3d-playback-count='true']").inner_text()
    return int(re.match(r"\s*(\d+)\s*/", text).group(1))


def _pixels(page):
    box = page.locator('canvas[aria-label^="Slice image"]').bounding_box()
    assert box is not None
    # Sample an interior patch so cursor, frame labels and toolbar changes cannot
    # make frozen scientific pixels look like successful playback.
    clip = {"x": box["x"] + box["width"] * .25,
            "y": box["y"] + box["height"] * .35,
            "width": min(120, box["width"] * .2),
            "height": min(120, box["height"] * .2)}
    return Image.open(BytesIO(page.screenshot(clip=clip))).convert("RGB")


def _assert_advancing(page, *, bounds=None):
    assert page.get_by_role("button", name="Pause playback", exact=True).is_visible()
    first_frame, first_pixels = _frame(page), _pixels(page)
    frames = {first_frame}
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        frame = _frame(page)
        frames.add(frame)
        if bounds:
            assert bounds[0] <= frame <= bounds[1]
        if len(frames) >= 3:
            delta = ImageChops.difference(first_pixels, _pixels(page))
            if delta.getbbox() is not None:
                return
        page.wait_for_timeout(70)
    pytest.fail("Playback did not advance both frame counter and image pixels")


def _drag(page, handle, fraction):
    inputs = _sliders(page)
    assert inputs.count() == 3, "Expected start/current/end playback handles"
    root = inputs.nth(handle).locator("xpath=ancestor::span[contains(@class, 'MuiSlider-root')]")
    root.scroll_into_view_if_needed()
    rail = root.bounding_box()
    thumb = inputs.nth(handle).locator("..").bounding_box()
    assert rail is not None and thumb is not None
    page.mouse.move(thumb["x"] + thumb["width"] / 2, thumb["y"] + thumb["height"] / 2)
    page.mouse.down()
    page.mouse.move(rail["x"] + rail["width"] * fraction,
                    rail["y"] + rail["height"] / 2, steps=12)
    page.mouse.up()
    page.mouse.move(1, 1)


def test_scrubbing_during_playback_keeps_advancing(page):
    """Repeated seeks in either direction preserve Play and repaint the image."""
    page.get_by_role("button", name="Play", exact=True).click()
    _assert_advancing(page)
    for fraction in (.7, .3, .6):
        _drag(page, 1, fraction)
        _assert_advancing(page)


def test_loop_endpoint_drags_keep_playing_inside_selected_range(page):
    """Range edits preserve playback and constrain subsequent frames."""
    from playwright.sync_api import expect

    _drag(page, 1, .5)
    page.get_by_role("button", name="Play", exact=True).click()
    _drag(page, 0, .2)
    _drag(page, 2, .8)
    inputs = _sliders(page)
    # React publishes range edits after the pointer release. Wait for that
    # visible state, rather than racing a still-pending render on large exports.
    expect(inputs.nth(0)).not_to_have_value("0")
    expect(inputs.nth(2)).not_to_have_value(inputs.nth(2).get_attribute("max"))
    low, high = int(inputs.nth(0).input_value()), int(inputs.nth(2).input_value())
    assert low > 0 and high < int(inputs.nth(2).get_attribute("max"))
    _assert_advancing(page, bounds=(low + 1, high + 1))


def test_paused_scrubbing_does_not_start_playback(page):
    """Seeking while paused changes the displayed image but leaves it paused."""
    _drag(page, 1, .3)
    before = _pixels(page)
    _drag(page, 1, .7)
    assert page.get_by_role("button", name="Play", exact=True).is_visible()
    after = _pixels(page)
    assert ImageChops.difference(before, after).getbbox() is not None
    frame = _frame(page)
    page.wait_for_timeout(700)
    assert _frame(page) == frame
    assert ImageChops.difference(after, _pixels(page)).getbbox() is None


def test_keyboard_seek_keeps_playback_running(page):
    """Accessible keyboard seeking preserves the same playback intent."""
    page.get_by_role("button", name="Play", exact=True).click()
    _sliders(page).nth(1).focus()
    _sliders(page).nth(1).press("ArrowRight")
    _assert_advancing(page)


def test_clicking_timeline_keeps_playback_running(page):
    """Clicking the track uses the same seek path as dragging the handle."""
    page.get_by_role("button", name="Play", exact=True).click()
    root = _sliders(page).nth(1).locator(
        "xpath=ancestor::span[contains(@class, 'MuiSlider-root')]"
    )
    root.scroll_into_view_if_needed()
    rail = root.bounding_box()
    assert rail is not None
    page.mouse.click(rail["x"] + rail["width"] * .3,
                     rail["y"] + rail["height"] / 2)
    page.mouse.move(1, 1)
    _assert_advancing(page)
