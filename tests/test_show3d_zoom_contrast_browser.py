"""Browser regression for Show3D linked zoom with independent auto contrast."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")


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


def _contrast_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    shape = (3, 128, 128)
    bf = rng.normal(78_000.0, 500.0, shape).astype(np.float32)
    df = rng.normal(6_000.0, 260.0, shape).astype(np.float32)
    phase = rng.normal(0.0, 0.025, shape).astype(np.float32)
    for panel in (bf, df, phase):
        panel[:, 0, 0] = float(panel.min()) - float(np.ptp(panel)) * 8
        panel[:, 0, 1] = float(panel.max()) + float(np.ptp(panel)) * 8
    return bf, df, phase


def _anchor_patch(page) -> dict[str, float | str | None]:
    return page.evaluate(
        """() => {
          const canvas = document.querySelectorAll("canvas")[0];
          const ctx = canvas.getContext("2d", { willReadFrequently: true });
          const x = Math.round(canvas.width / 4);
          const y = Math.round(canvas.height / 4);
          const data = ctx.getImageData(x - 2, y - 2, 5, 5).data;
          const values = [];
          for (let i = 0; i < data.length; i += 4) values.push(data[i]);
          values.sort((a, b) => a - b);
          const perf = window.__quantemShow3DPerf || {};
          return {
            median: values[12],
            latency: perf.lastInteractionLatencyMs ?? null,
            path: perf.lastInteractionRenderPath ?? null,
          };
        }"""
    )


@pytest.mark.skipif(
    os.environ.get("QT_RUN_BROWSER_TESTS") != "1",
    reason="set QT_RUN_BROWSER_TESTS=1 to run Show3D browser regression tests",
)
def test_linked_zoom_preserves_unlinked_panel_auto_contrast(tmp_path):
    from playwright.sync_api import sync_playwright

    from quantem.widget import Show3D

    chrome = _chrome_executable()
    if chrome is None:
        pytest.skip("Chrome/Chromium executable not found")

    bf, df, phase = _contrast_fixture()
    widget = Show3D(
        bf,
        df,
        phase,
        panel_titles=["BF", "DF", "phase"],
        auto_contrast=True,
        link_contrast=False,
        link_panels=True,
        percentile_low=0.5,
        percentile_high=99.5,
        max_cols=2,
        verbose=False,
    )
    widget.panel_cmaps = ["gray", "inferno", "viridis"]
    html_path = tmp_path / "show3d-zoom-contrast.html"
    widget.export_html(html_path, encoding="uint8")

    page_errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=chrome,
            headless=True,
            args=["--enable-unsafe-swiftshader", "--enable-webgpu"],
        )
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(html_path.as_uri())
        page.wait_for_timeout(2_500)

        canvas = page.locator("canvas").nth(0)
        box = canvas.bounding_box()
        assert box is not None
        before = _anchor_patch(page)
        page.mouse.move(box["x"] + box["width"] / 4, box["y"] + box["height"] / 4)
        page.mouse.wheel(0, -320)
        page.wait_for_timeout(700)
        after = _anchor_patch(page)

        assert abs(float(after["median"]) - float(before["median"])) <= 12
        assert after["path"] == "canvas-panel-transform"
        assert after["latency"] is not None
        assert float(after["latency"]) < 50
        assert page_errors == []

        page.get_by_role("button", name="Reset").click()
        page.wait_for_timeout(300)
        reset = _anchor_patch(page)
        assert reset["median"] == before["median"]

        # Hovering a panel reveals its top-left hide action without a Python
        # option. Hiding repacks the visible gallery and the Panels menu can
        # always restore it.
        page.mouse.move(box["x"] + box["width"] / 4, box["y"] + box["height"] / 4)
        page.wait_for_timeout(200)
        hide_bf = page.get_by_role("button", name="Hide BF")
        assert hide_bf.count() == 1
        assert hide_bf.evaluate("el => getComputedStyle(el).opacity") == "1"
        hide_bf.click()
        page.wait_for_function(
            "() => window.__quantemShow3DPerf?.embeddedPackedViewportPaint?.visibleCount === 2"
        )

        page.get_by_role("button", name="Choose visible panels").click()
        page.get_by_text("Show all panels", exact=True).click()
        page.wait_for_function(
            "() => window.__quantemShow3DPerf?.embeddedPackedViewportPaint?.visibleCount === 3"
        )
        assert page_errors == []
        browser.close()
