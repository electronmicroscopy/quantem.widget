"""Opt-in visible-pixel gate on an already loaded resident Show4DSTEM page.

Set QT_RESIDENT_DRAG_CDP to the owned browser's debugging URL and
QT_RESIDENT_DRAG_PAGE to its exact viewer URL. Screenshots and evidence go to
QT_RESIDENT_DRAG_REPORT_DIR. Load and independently qualify native parity first.
This test changes detector/view controls but never loads or copies source data.
"""

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("QT_RESIDENT_DRAG_CDP"),
    reason="requires an explicitly selected loaded resident browser",
)


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_visible_images_change_before_detector_release(scale: str):
    """Center and radius moves must change image interiors before pointer-up."""
    from PIL import Image
    from playwright.sync_api import sync_playwright

    url = os.environ["QT_RESIDENT_DRAG_PAGE"]
    report = Path(os.environ["QT_RESIDENT_DRAG_REPORT_DIR"]) / scale
    report.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(
            os.environ["QT_RESIDENT_DRAG_CDP"]
        )
        page = next(p for context in browser.contexts for p in context.pages if p.url == url)
        page.bring_to_front()
        assert page.evaluate("!!window.__sh4d?.residentSource?.()")
        count = page.evaluate("__sh4d.residentSource().loadedAcquisitions")
        assert 2 <= count <= 12, "Use two to twelve complete acquisitions for this gate."
        assert page.evaluate("[__sh4d.model.get('det_rows'), __sh4d.model.get('det_cols')]") == [192, 192]
        keys = ["view_mode", "vi_source", "roi_active", "roi_mode", "roi_radius",
                "roi_radius_inner", "roi_center_row", "roi_center_col", "vi_scale_mode"]
        saved = page.evaluate("keys => Object.fromEntries(keys.map(k=>[k,__sh4d.model.get(k)]))", keys)
        pressed = False
        page.evaluate("""() => {
            window.heldRegression = {events: [], errors: []};
            window.heldRegressionListener = e => heldRegression.events.push({
                type:e.type, buttons:e.buttons, trusted:e.isTrusted});
            for (const type of ['pointerdown','pointerup','pointercancel','pointermove'])
                window.addEventListener(type, heldRegressionListener, true);
            window.heldRegressionError = e => heldRegression.errors.push(e.error.message);
            __sh4d.residentSource().device.addEventListener('uncapturederror', heldRegressionError);
        }""")
        try:
            for view in ("single", "multiple"):
                for mode in ("circle", "annular"):
                    for gesture in ("center", "outer", "inner"):
                        if gesture == "inner" and mode == "circle":
                            continue
                        settings = dict(view_mode=view, vi_source="roi", roi_active=True,
                                        roi_mode=mode, roi_radius=24 if mode == "circle" else 94,
                                        roi_radius_inner=0 if mode == "circle" else 60,
                                        roi_center_row=95.5, roi_center_col=95.5, vi_scale_mode=scale)
                        page.evaluate("s => __sh4d.model.set(s)", settings)
                        page.wait_for_timeout(250)
                        dp = page.locator('[data-quantem-scientific-output="show4dstem-diffraction-pattern"]').bounding_box()
                        assert dp
                        values = ([95.5, 120, 145, 80] if gesture == "center" else
                                  [60, 30, 50, 20] if gesture == "inner" else
                                  [24, 35, 45, 20] if mode == "circle" else [94, 80, 70, 90])
                        previous = None
                        for pose, value in enumerate(values):
                            row, col = ((95.5, value) if gesture == "center" else
                                        (95.5 + value / 2**0.5, 95.5 + value / 2**0.5))
                            x, y = dp["x"] + col / 192 * dp["width"], dp["y"] + row / 192 * dp["height"]
                            page.mouse.move(x, y)
                            if pose == 0:
                                page.evaluate("heldRegression.events = []")
                                page.mouse.down()
                                pressed = True
                            page.wait_for_timeout(200)
                            # Full viewport only: clipped screenshots can resize the
                            # viewport and synthesize zero-button pointer events.
                            raw = page.screenshot(full_page=False, scale="css")
                            name = f"{view}-{mode}-{gesture}-{pose}"
                            (report / f"{name}.png").write_bytes(raw)
                            image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                            selectors = (["show4dstem-virtual-image"] if view == "single" else
                                         [f"show4dstem-compare-{i}" for i in range(count)])
                            pixel_scale = image.shape[1] / page.evaluate("innerWidth")
                            crops = []
                            for output in selectors:
                                rect = page.locator(f'[data-quantem-scientific-output="{output}"]').bounding_box()
                                assert rect
                                left, top = int(rect["x"] + 25), int(rect["y"] + 25)
                                right, bottom = int(rect["x"] + rect["width"] - 25), int(rect["y"] + rect["height"] - 25)
                                left, top, right, bottom = (round(v * pixel_scale) for v in (left, top, right, bottom))
                                crop = image[top:bottom, left:right].copy()
                                assert crop.size
                                assert np.count_nonzero(crop) > 100, name
                                crops.append(crop)
                            events = page.evaluate("heldRegression.events")
                            assert events[0]["type"] == "pointerdown"
                            assert all(e["trusted"] and e["buttons"] == 1 for e in events), events
                            changes = ([] if previous is None else
                                       [int(np.count_nonzero(np.any(a != b, axis=2))) for a, b in zip(crops, previous)])
                            results.append(dict(case=name, changed_pixels=changes, events=events))
                            assert all(n > 100 for n in changes), results[-1]
                            previous = crops
                        page.mouse.up()
                        pressed = False
            assert not page.evaluate("heldRegression.errors")
        finally:
            if pressed:
                page.mouse.up()
            page.evaluate("""s => {
                for (const type of ['pointerdown','pointerup','pointercancel','pointermove'])
                    window.removeEventListener(type, heldRegressionListener, true);
                __sh4d.residentSource().device.removeEventListener('uncapturederror', heldRegressionError);
                __sh4d.model.set(s);
            }""", saved)
            (report / "held-drag.json").write_text(json.dumps(results, indent=2))
            # Disconnect this driver; preserve the user-owned page and browser.


def test_copy_exports_visible_resident_pixels():
    """Exercise Copy without overwriting the user's clipboard; decode its PNG."""
    from PIL import Image
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(os.environ["QT_RESIDENT_DRAG_CDP"])
        page = next(p for c in browser.contexts for p in c.pages
                    if p.url == os.environ["QT_RESIDENT_DRAG_PAGE"])
        saved = page.evaluate("__sh4d.model.get('view_mode')")
        page.evaluate("__sh4d.model.set('view_mode', 'single')")
        page.wait_for_timeout(250)
        page.evaluate("""() => {
            window.savedCopyWrite = Object.getOwnPropertyDescriptor(navigator.clipboard, 'write');
            window.copyResult = null;
            Object.defineProperty(navigator.clipboard, 'write', {configurable: true,
                value: async items => {copyResult = await items[0].getType('image/png');}});
        }""")
        try:
            page.get_by_role("button", name="Copy", exact=True).nth(1).click()
            page.wait_for_function("window.copyResult !== null", timeout=5000)
            raw = bytes(page.evaluate("async()=>Array.from(new Uint8Array(await copyResult.arrayBuffer()))"))
            image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            assert image.shape == (512, 512, 3)
            assert np.count_nonzero(np.any(image != 0, axis=2)) > 250_000
            report = Path(os.environ["QT_RESIDENT_DRAG_REPORT_DIR"])
            report.mkdir(parents=True, exist_ok=True)
            (report / "copied-image.png").write_bytes(raw)
        finally:
            page.evaluate("""view => {
                if (savedCopyWrite) Object.defineProperty(navigator.clipboard, 'write', savedCopyWrite);
                else delete navigator.clipboard.write;
                delete window.copyResult;
                __sh4d.model.set('view_mode', view);
            }""", saved)
