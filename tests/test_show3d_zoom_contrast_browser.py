"""Browser regression for Show3D linked zoom with independent auto contrast."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from PIL import Image
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


def _visible_panel_color_spread(page) -> list[float]:
    canvas = page.locator("canvas").nth(0)
    box = canvas.bounding_box()
    assert box is not None
    screenshot = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
    left = round(box["x"])
    top = round(box["y"])
    right = round(box["x"] + box["width"])
    bottom = round(box["y"] + box["height"])
    pixels = np.asarray(screenshot.crop((left, top, right, bottom)), dtype=np.float32)
    height, width, _ = pixels.shape
    spreads: list[float] = []
    for panel in range(3):
        panel_left = round(panel * width / 3 + width * 0.015)
        panel_right = round((panel + 1) * width / 3 - width * 0.015)
        region = pixels[round(height * 0.08):round(height * 0.92), panel_left:panel_right]
        spreads.append(float((region.max(axis=2) - region.min(axis=2)).mean()))
    return spreads


def _visible_canvas_pixels(page) -> np.ndarray:
    """Capture the actual composited scientific canvas as RGB pixels."""
    canvas = page.locator('canvas[role="img"]').first
    screenshot = Image.open(io.BytesIO(canvas.screenshot())).convert("RGB")
    return np.asarray(screenshot, dtype=np.float32)


def _panel_pixel_differences(
    before: np.ndarray,
    after: np.ndarray,
    count: int,
    cols: int,
) -> list[float]:
    """Return mean absolute RGB changes for each panel in a wrapped grid."""
    assert before.shape == after.shape
    rows = int(np.ceil(count / cols))
    height, width = before.shape[:2]
    differences: list[float] = []
    for panel in range(count):
        row, col = divmod(panel, cols)
        top, bottom = round(row * height / rows), round((row + 1) * height / rows)
        left, right = round(col * width / cols), round((col + 1) * width / cols)
        before_panel = before[top:bottom, left:right]
        after_panel = after[top:bottom, left:right]
        delta = np.abs(after_panel - before_panel)
        active = np.minimum(before_panel, after_panel).min(axis=2) < 250
        differences.append(float(delta[active].mean()) if np.any(active) else 0.0)
    return differences


def _panel_spatial_std(
    pixels: np.ndarray,
    count: int,
    cols: int,
) -> list[float]:
    """Measure scientific texture while excluding titles and panel edges."""
    rows = int(np.ceil(count / cols))
    height, width = pixels.shape[:2]
    spreads: list[float] = []
    for panel in range(count):
        row, col = divmod(panel, cols)
        y0 = round((row + 0.12) * height / rows)
        y1 = round((row + 0.88) * height / rows)
        x0 = round((col + 0.08) * width / cols)
        x1 = round((col + 0.92) * width / cols)
        region = pixels[y0:y1, x0:x1]
        spreads.append(float(np.std(region.mean(axis=2))))
    return spreads


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
    widget.export_html(html_path, encoding="full")

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

        page.wait_for_function(
            "() => window.__quantemShow3DPerf?.offlineFramePrewarmDone === 3"
        )

        canvas = page.locator("canvas").nth(0)
        box = canvas.bounding_box()
        assert box is not None
        before = _anchor_patch(page)
        page.mouse.move(box["x"] + box["width"] / 4, box["y"] + box["height"] / 4)
        page.mouse.wheel(0, -320)
        page.wait_for_timeout(700)
        after = _anchor_patch(page)

        assert abs(float(after["median"]) - float(before["median"])) <= 12
        assert after["path"] == "canvas-packed-transform"
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
            "() => window.__quantemShow3DPerf?.layoutVisiblePanels === 2"
        )

        page.get_by_role("button", name="Choose visible panels").click()
        page.get_by_text("Show all panels", exact=True).click()
        page.wait_for_function(
            "() => window.__quantemShow3DPerf?.layoutVisiblePanels === 3"
        )

        assert page_errors == []
        browser.close()


@pytest.mark.skipif(
    os.environ.get("QT_RUN_BROWSER_TESTS") != "1",
    reason="set QT_RUN_BROWSER_TESTS=1 to run Show3D browser regression tests",
)
def test_smooth_toggle_keeps_playback_residency(tmp_path):
    """Smooth and play/pause reuse WebGPU residency or the canvas fallback."""
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
        link_contrast=False,
        max_cols=2,
        verbose=False,
    )
    html_path = tmp_path / "show3d-smooth-playback.html"
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
        page.wait_for_function(
            """() => {
              return document.querySelector(
                '[data-show3d-gpu-residency="ready"], [data-show3d-gpu-residency="fallback"]'
              ) !== null;
            }"""
        )
        page.wait_for_timeout(100)

        cache_before = page.evaluate(
            """() => ({
              gpu: window.__quantemShow3DPerf.embeddedGpuResident === true,
              uploads: window.__quantemShow3DPerf.gpuFrameCacheUploaded || 0,
            })"""
        )
        if cache_before["gpu"]:
            assert cache_before["uploads"] == 3
        page.get_by_role("checkbox", name="Toggle bilinear smoothing").check()
        page.wait_for_timeout(150)
        cache_after_smooth = page.evaluate(
            """() => ({
              gpu: window.__quantemShow3DPerf.embeddedGpuResident === true,
              uploads: window.__quantemShow3DPerf.gpuFrameCacheUploaded || 0,
              imageRendering: getComputedStyle(document.querySelector("canvas")).imageRendering,
            })"""
        )
        assert cache_after_smooth["gpu"] == cache_before["gpu"]
        assert cache_after_smooth["uploads"] == cache_before["uploads"]
        assert cache_after_smooth["imageRendering"] == "auto"

        page.get_by_role("button", name="Play forward").click()
        page.wait_for_timeout(350)
        page.get_by_role("button", name="Pause playback").click()
        page.wait_for_timeout(150)
        cache_after_playback = page.evaluate(
            """() => ({
              gpu: window.__quantemShow3DPerf.embeddedGpuResident === true,
              uploads: window.__quantemShow3DPerf.gpuFrameCacheUploaded || 0,
            })"""
        )
        assert cache_after_playback["gpu"] == cache_before["gpu"]
        assert cache_after_playback["uploads"] == cache_before["uploads"]
        assert page_errors == []
        browser.close()


@pytest.mark.skipif(
    os.environ.get("QT_RUN_BROWSER_TESTS") != "1",
    reason="set QT_RUN_BROWSER_TESTS=1 to run Show3D browser regression tests",
)
def test_scrub_commit_preserves_mixed_panel_colormaps(tmp_path):
    """Held drags repaint raw resident frames without blanking or early commits."""
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
        cmap=["gray", "inferno", "viridis"],
        link_contrast=False,
        max_cols=3,
        verbose=False,
    )
    html_path = tmp_path / "show3d-mixed-colormap-scrub.html"
    widget.export_html(html_path, encoding="full")

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

        average = page.get_by_role("slider", name="Moving average window")
        assert average.input_value() == "1"
        baseline = _visible_panel_color_spread(page)
        baseline_pixels = _visible_canvas_pixels(page)
        sliders = page.locator('input[aria-label^="Loop range and current frame"]')
        assert sliders.count() == 3
        current_thumb = sliders.nth(1)
        initial_frame = int(current_thumb.input_value())
        page.evaluate(
            """() => {
              const perf = window.__quantemShow3DPerf;
              perf.scrubPointerEvents = 0;
              perf.scrubRafPaints = 0;
              perf.scrubModelCommits = 0;
              perf.lastScrubFrame = null;
              perf.lastScrubCommitFrame = null;
              perf.lastScrubRenderPath = null;
              perf.lastScrubInputLatencyMs = null;
            }"""
        )
        thumb_box = current_thumb.bounding_box()
        slider_box = current_thumb.locator(
            "xpath=ancestor::*[contains(@class, 'MuiSlider-root')]"
        ).bounding_box()
        assert thumb_box is not None and slider_box is not None
        page.mouse.move(
            thumb_box["x"] + thumb_box["width"] / 2,
            thumb_box["y"] + thumb_box["height"] / 2,
        )
        page.mouse.down()
        page.mouse.move(
            slider_box["x"] + slider_box["width"] * 0.8,
            slider_box["y"] + slider_box["height"] / 2,
        )
        page.wait_for_timeout(35)
        during_frame = int(current_thumb.input_value())
        during_drag = _visible_panel_color_spread(page)
        during_pixels = _visible_canvas_pixels(page)
        during_perf = page.evaluate("() => ({ ...window.__quantemShow3DPerf })")

        assert during_frame != initial_frame
        assert during_perf["lastScrubFrame"] == during_frame
        assert during_perf["scrubPointerEvents"] >= 1
        assert during_perf["scrubRafPaints"] >= 1
        assert during_perf["scrubModelCommits"] == 0
        assert during_perf["lastScrubCommitFrame"] is None
        assert "average" not in str(during_perf["lastScrubRenderPath"]).lower()
        if during_perf.get("embeddedGpuResident") is True:
            assert "webgpu" in str(during_perf["lastScrubRenderPath"]).lower()
            assert during_perf["lastScrubInputLatencyMs"] < 16.7
        assert min(_panel_spatial_std(during_pixels, 3, 3)) > 3.0
        assert any(
            difference > 0.5
            for difference in _panel_pixel_differences(
                baseline_pixels,
                during_pixels,
                3,
                3,
            )
        )

        page.mouse.up()
        page.wait_for_timeout(50)
        after_release = _visible_panel_color_spread(page)
        after_pixels = _visible_canvas_pixels(page)
        after_perf = page.evaluate("() => ({ ...window.__quantemShow3DPerf })")

        assert after_perf["scrubModelCommits"] == 1
        assert after_perf["lastScrubCommitFrame"] == during_frame
        assert min(_panel_spatial_std(after_pixels, 3, 3)) > 3.0

        for panel in (1, 2):
            assert baseline[panel] > 20
            assert during_drag[panel] >= baseline[panel] * 0.25
            assert after_release[panel] >= baseline[panel] * 0.25
        assert page_errors == []
        browser.close()


@pytest.mark.skipif(
    os.environ.get("QT_RUN_BROWSER_TESTS") != "1",
    reason="set QT_RUN_BROWSER_TESTS=1 to run Show3D browser regression tests",
)
def test_display_controls_repaint_pixels_immediately_during_playback(tmp_path):
    """Display controls repaint now, including while the movie is playing."""
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
        cmap="gray",
        auto_contrast=False,
        link_contrast=False,
        max_cols=3,
        fps=60,
        debug=True,
        verbose=False,
    )
    html_path = tmp_path / "show3d-live-display-controls.html"
    widget.export_html(html_path, encoding="full")

    page_errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=chrome,
            headless=True,
            args=["--enable-unsafe-swiftshader", "--enable-webgpu"],
        )
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(html_path.as_uri())
        page.wait_for_function("() => window.__quantemShow3DPerf?.offlineFramePrewarmDone === 3")
        page.wait_for_timeout(200)

        initial_frame = page.evaluate(
            "() => window.__quantemShow3DPerf.lastPlaybackLiveCountText"
        )
        layout_cols = page.evaluate("() => window.__quantemShow3DPerf.layoutCols")
        assert isinstance(layout_cols, int) and layout_cols > 0
        gray = _visible_canvas_pixels(page)

        page.get_by_role("combobox", name="Shared colormap for all panels").click()
        page.get_by_role("option", name="Viridis", exact=True).click()
        page.wait_for_timeout(100)
        viridis = _visible_canvas_pixels(page)
        assert np.abs(viridis - gray).mean() > 5
        assert page.evaluate(
            "() => window.__quantemShow3DPerf.lastPlaybackLiveCountText"
        ) == initial_frame

        page.get_by_role("combobox", name="Intensity scale (linear or logarithmic)").click()
        page.get_by_role("option", name="Log", exact=True).click()
        page.wait_for_timeout(100)
        log_pixels = _visible_canvas_pixels(page)
        log_delta = np.abs(log_pixels - viridis)
        assert np.quantile(log_delta, 0.99) >= 3
        assert np.mean(log_delta.max(axis=2) > 2) > 0.01
        assert page.evaluate(
            "() => window.__quantemShow3DPerf.lastPlaybackLiveCountText"
        ) == initial_frame

        page.get_by_role("checkbox", name="Toggle bilinear smoothing").check()
        page.wait_for_timeout(100)
        assert page.locator('canvas[role="img"]').first.evaluate(
            "canvas => getComputedStyle(canvas).imageRendering"
        ) == "auto"
        assert page.evaluate(
            "() => window.__quantemShow3DPerf.lastPlaybackLiveCountText"
        ) == initial_frame

        # With contrast unlinked, the first histogram changes only BF while the
        # pointer is still held. A release is not needed to trigger the paint.
        unlinked_before = _visible_canvas_pixels(page)
        first_histogram_thumb = page.locator(
            'input[aria-label="Histogram intensity clip range"]'
        ).first
        first_slider = first_histogram_thumb.locator(
            "xpath=ancestor::*[contains(@class, 'MuiSlider-root')]"
        )
        first_slider.scroll_into_view_if_needed()
        thumb_box = first_slider.locator(".MuiSlider-thumb").first.bounding_box()
        slider_box = first_slider.bounding_box()
        assert thumb_box is not None and slider_box is not None
        page.mouse.move(
            thumb_box["x"] + thumb_box["width"] / 2,
            thumb_box["y"] + thumb_box["height"] / 2,
        )
        page.mouse.down()
        page.mouse.move(
            slider_box["x"] + slider_box["width"] * 0.35,
            slider_box["y"] + slider_box["height"] / 2,
        )
        page.wait_for_timeout(50)
        unlinked_during_drag = _visible_canvas_pixels(page)
        page.mouse.up()
        unlinked_changes = _panel_pixel_differences(
            unlinked_before,
            unlinked_during_drag,
            3,
            layout_cols,
        )
        assert unlinked_changes[0] > 1
        assert unlinked_changes[1] < unlinked_changes[0] * 0.25
        assert unlinked_changes[2] < unlinked_changes[0] * 0.25

        # Manual per-panel ranges are stored in the current intensity domain.
        # Switching Log -> Lin -> Log must convert those absolute limits with
        # the pixels. Reusing limits from the previous domain makes every pixel
        # hit the first LUT color and presents a uniform blue/purple canvas.
        scale = page.get_by_role(
            "combobox", name="Intensity scale (linear or logarithmic)"
        )
        scale.click()
        page.get_by_role("option", name="Lin", exact=True).click()
        page.wait_for_timeout(100)
        linear_after_manual = _visible_canvas_pixels(page)
        assert min(_panel_spatial_std(linear_after_manual, 3, layout_cols)) > 3
        scale.click()
        page.get_by_role("option", name="Log", exact=True).click()
        page.wait_for_timeout(100)
        log_after_manual = _visible_canvas_pixels(page)
        assert min(_panel_spatial_std(log_after_manual, 3, layout_cols)) > 3

        page.get_by_role("checkbox", name="Link contrast across panels").check()
        page.wait_for_timeout(100)
        linked_before = _visible_canvas_pixels(page)
        linked_thumb = page.locator(
            'input[aria-label="Histogram intensity clip range"]'
        ).first
        linked_slider = linked_thumb.locator(
            "xpath=ancestor::*[contains(@class, 'MuiSlider-root')]"
        )
        linked_slider.scroll_into_view_if_needed()
        linked_thumb_box = linked_slider.locator(".MuiSlider-thumb").first.bounding_box()
        linked_slider_box = linked_slider.bounding_box()
        assert linked_thumb_box is not None and linked_slider_box is not None
        page.mouse.move(
            linked_thumb_box["x"] + linked_thumb_box["width"] / 2,
            linked_thumb_box["y"] + linked_thumb_box["height"] / 2,
        )
        page.mouse.down()
        page.mouse.move(
            linked_slider_box["x"] + linked_slider_box["width"] * 0.2,
            linked_slider_box["y"] + linked_slider_box["height"] / 2,
        )
        page.wait_for_timeout(50)
        linked_during_drag = _visible_canvas_pixels(page)
        page.mouse.up()
        assert min(
            _panel_pixel_differences(
                linked_before,
                linked_during_drag,
                3,
                layout_cols,
            )
        ) > 1

        # Changing the colormap during playback must keep the movie running and
        # repaint before any pause or scrub gesture.
        page.get_by_role("button", name="Play", exact=True).click()
        page.wait_for_timeout(150)
        frame_before_style = page.evaluate(
            "() => window.__quantemShow3DPerf.lastPlaybackLiveCountText"
        )
        playing_before_style = _visible_canvas_pixels(page)
        page.get_by_role("combobox", name="Shared colormap for all panels").click()
        page.get_by_role("option", name="Inferno", exact=True).click()
        page.wait_for_function(
            "previous => window.__quantemShow3DPerf.lastPlaybackLiveCountText !== previous",
            arg=frame_before_style,
        )
        playing_after_style = _visible_canvas_pixels(page)
        assert page.get_by_role("button", name="Pause playback").count() == 1
        assert np.abs(playing_after_style - playing_before_style).mean() > 2
        assert min(_panel_spatial_std(playing_after_style, 3, layout_cols)) > 3
        page.get_by_role("button", name="Pause playback").click()

        assert page_errors == []
        browser.close()
