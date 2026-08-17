"""Browser regression tests for the dedicated Mask2D interface."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from quantem.widget import Mask2D


sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright


def _chrome_executable() -> str | None:
    candidates = [
        os.environ.get("CHROME_EXECUTABLE"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/opt/google/chrome/chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    return next((path for path in candidates if path and Path(path).exists()), None)


@pytest.mark.skipif(
    os.environ.get("QT_RUN_BROWSER_TESTS") != "1",
    reason="set QT_RUN_BROWSER_TESTS=1 to run Mask2D browser regression tests",
)
def test_mask2d_export_draws_replaces_and_clears_selection(tmp_path):
    chrome = _chrome_executable()
    if chrome is None:
        pytest.skip("Chrome/Chromium executable not found")

    row, col = np.mgrid[:96, :128]
    image = (
        np.exp(-((row - 45) ** 2 + (col - 70) ** 2) / 450) * 800
        + row * 2
        + col
    ).astype(np.float32)
    html_path = tmp_path / "mask2d.html"
    Mask2D(image, title="Choose region", size=500).export_html(html_path)

    page_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, headless=True)
        page = browser.new_page(viewport={"width": 900, "height": 800})
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(html_path.as_uri())

        canvas = page.get_by_label("Mask2D image")
        canvas.wait_for(state="visible")
        box = canvas.bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + 80, box["y"] + 100)
        page.mouse.down()
        page.mouse.move(box["x"] + 260, box["y"] + 280, steps=8)
        page.mouse.up()

        assert page.get_by_text("rectangle center", exact=False).is_visible()
        assert page.get_by_role("button", name="Clear").is_enabled()

        page.get_by_label("Region shape").click()
        page.get_by_role("option", name="Circle").click()
        page.mouse.move(box["x"] + 180, box["y"] + 170)
        page.mouse.down()
        page.mouse.move(box["x"] + 320, box["y"] + 310, steps=8)
        page.mouse.up()

        assert page.get_by_text("circle center", exact=False).is_visible()
        page.get_by_role("button", name="Clear").click()
        assert page.get_by_role("button", name="Clear").is_disabled()
        assert page.get_by_text("circle center", exact=False).count() == 0
        assert page_errors == []
        browser.close()
