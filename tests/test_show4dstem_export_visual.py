"""Opt-in exported Show4DSTEM visual smoke tests.

These tests intentionally drive standalone exported HTML with a real 4D-STEM
fixture. They are not default CI because they launch a browser and require a
local real-data fixture.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("playwright.sync_api")


DEFAULT_REAL_MEMMAP = Path("/tmp/quantem-widget-thin-gold-mobile/real_gold_16panel_32x32_uint16.memmap")
DEFAULT_REAL_SHAPE = (16, 32, 32, 24, 24)


def _chrome_executable() -> str | None:
    candidates = [
        os.environ.get("CHROME_EXECUTABLE"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/opt/google/chrome/chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(port: int, proc: subprocess.Popen) -> None:
    import urllib.request

    url = f"http://127.0.0.1:{port}/"
    for _ in range(80):
        if proc.poll() is not None:
            raise RuntimeError(f"HTTP server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("HTTP server did not become ready")


def _load_real_4dstem_fixture() -> np.ndarray:
    npz_path = os.environ.get("QT_SHOW4DSTEM_VISUAL_NPZ")
    if npz_path:
        path = Path(npz_path)
        if not path.exists():
            pytest.skip(f"real-data fixture not found: {path}")
        with np.load(path) as archive:
            key = "data" if "data" in archive.files else archive.files[0]
            data = np.asarray(archive[key])
    else:
        memmap_path = Path(os.environ.get("QT_SHOW4DSTEM_VISUAL_MEMMAP", str(DEFAULT_REAL_MEMMAP)))
        if not memmap_path.exists():
            pytest.skip(
                "set QT_SHOW4DSTEM_VISUAL_NPZ or QT_SHOW4DSTEM_VISUAL_MEMMAP "
                "to run exported Show4DSTEM visual tests with real data"
            )
        shape_text = os.environ.get("QT_SHOW4DSTEM_VISUAL_SHAPE")
        shape = tuple(int(part) for part in shape_text.split(",")) if shape_text else DEFAULT_REAL_SHAPE
        dtype = os.environ.get("QT_SHOW4DSTEM_VISUAL_DTYPE", "uint16")
        data = np.memmap(memmap_path, dtype=np.dtype(dtype), mode="r", shape=shape)

    if data.ndim != 5:
        pytest.skip(f"real visual fixture must be 5D, got shape {data.shape!r}")
    if min(data.shape) < 4:
        pytest.skip(f"real visual fixture is too small for interaction smoke tests: {data.shape!r}")

    frames = min(6, data.shape[0])
    scan_rows = min(24, data.shape[1])
    scan_cols = min(24, data.shape[2])
    det_rows = min(24, data.shape[3])
    det_cols = min(24, data.shape[4])
    return np.asarray(data[:frames, :scan_rows, :scan_cols, :det_rows, :det_cols]).copy()


def _artifact_root(tmp_path: Path) -> Path:
    root = Path(os.environ.get("QT_SHOW4DSTEM_VISUAL_ARTIFACTS", tmp_path / "visual-artifacts"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _export_show4dstem_pages(root: Path, data: np.ndarray) -> dict[str, Path]:
    from quantem.widget import Show4DSTEM

    pages: dict[str, Path] = {}
    cases = {
        "single": {
            "data": data[0],
            "kwargs": {"view_mode": "single", "compare_dp_mode": "average"},
        },
        "multiple-cols2": {
            "data": data,
            "kwargs": {"view_mode": "multiple", "compare_cols": 2, "compare_dp_mode": "average"},
        },
        "multiple-cols4": {
            "data": data,
            "kwargs": {"view_mode": "multiple", "compare_cols": 4, "compare_dp_mode": "average"},
        },
        "multiple-selected": {
            "data": data,
            "kwargs": {"view_mode": "multiple", "compare_cols": 3, "compare_dp_mode": "selected"},
        },
    }
    for name, case in cases.items():
        title = f"Visual Smoke Show4DSTEM {name}"
        widget = Show4DSTEM(
            case["data"],
            title=title,
            show_fft=True,
            show_stats=True,
            show_scale_bar=True,
            compare_panel_gap_px=0,
            compare_grid_width_px=920,
            compare_max_panels=min(6, int(data.shape[0])),
            precompute_virtual_images=True,
            verbose=False,
            **case["kwargs"],
        )
        try:
            out = root / f"{name}.html"
            widget.export_html(out, title=title, encoding="full", downsample=1)
            pages[name] = out
        finally:
            widget.close()
    _write_touch_logger_page(root, pages["multiple-cols4"].name)
    return pages


def _write_touch_logger_page(root: Path, target_name: str) -> Path:
    page = root / "iphone-touch-logger.html"
    page.write_text(
        f"""<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Show4DSTEM iPhone Touch Logger</title>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
#log {{ position: sticky; top: 0; z-index: 2; background: #111; color: #fff; padding: 8px; font-size: 12px; }}
iframe {{ width: 100vw; height: calc(100vh - 88px); border: 0; display: block; }}
</style>
<div id="log">waiting for touch...</div>
<iframe src="{target_name}"></iframe>
<script>
const log = document.getElementById("log");
const lines = [];
function note(kind, event) {{
  const touch = event.touches && event.touches[0];
  const x = touch ? touch.clientX : event.clientX;
  const y = touch ? touch.clientY : event.clientY;
  lines.unshift(`${{kind}} x=${{Math.round(x || 0)}} y=${{Math.round(y || 0)}} `
    + `vw=${{innerWidth}} vh=${{innerHeight}} scroll=${{Math.round(scrollY)}}`);
  log.textContent = lines.slice(0, 4).join(" | ");
}}
["touchstart", "touchmove", "touchend", "pointerdown", "pointermove", "pointerup", "scroll"].forEach(
  kind => addEventListener(kind, event => note(kind, event), {{ passive: true }})
);
</script>
""",
        encoding="utf-8",
    )
    return page


def _dp_at(page) -> str:
    text = page.evaluate("document.body.innerText")
    match = re.search(r"DP at \([^)]*\)", text)
    assert match, "DP coordinate label was not found"
    return match.group(0)


def _large_canvas_rects(page) -> list[dict]:
    rects = page.evaluate(
        """() => {
          const seen = new Set();
          const out = [];
          for (const canvas of document.querySelectorAll('canvas')) {
            const r = canvas.getBoundingClientRect();
            if (r.width < 90 || r.height < 90) continue;
            const key = [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)].join(',');
            if (seen.has(key)) continue;
            seen.add(key);
            out.push({x:r.x, y:r.y, w:r.width, h:r.height});
          }
          return out.sort((a, b) => (a.y - b.y) || (a.x - b.x));
        }"""
    )
    assert len(rects) >= 2, f"expected at least two large canvas panels, got {rects!r}"
    return rects


def _drag_rect(page, rect: dict, start_frac: tuple[float, float], end_frac: tuple[float, float]) -> None:
    start = (rect["x"] + rect["w"] * start_frac[0], rect["y"] + rect["h"] * start_frac[1])
    end = (rect["x"] + rect["w"] * end_frac[0], rect["y"] + rect["h"] * end_frac[1])
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, steps=8)
    page.mouse.move(*end, steps=8)
    page.mouse.up()


def _drag_single_vi(page) -> None:
    rect = _large_canvas_rects(page)[1]
    before = _dp_at(page)
    _drag_rect(page, rect, (0.32, 0.35), (0.78, 0.75))
    page.wait_for_timeout(400)
    after = _dp_at(page)
    assert after != before, f"single VI drag did not update scan position: {before}"


def _drag_multiple_panel(page, panel_number: int = 1) -> None:
    panel = page.locator(f'[role="button"][aria-label="Show4DSTEM multiple panel {panel_number}"]')
    assert panel.count() == 1
    rect = panel.bounding_box()
    assert rect and rect["width"] > 40 and rect["height"] > 40
    before = _dp_at(page)
    _drag_rect(page, {"x": rect["x"], "y": rect["y"], "w": rect["width"], "h": rect["height"]}, (0.30, 0.35), (0.78, 0.75))
    page.wait_for_timeout(400)
    after = _dp_at(page)
    assert after != before, f"multiple panel drag did not update scan position: {before}"


def _toggle_fft(page) -> None:
    result = page.evaluate(
        """() => {
          const labels = [...document.querySelectorAll('p,span,label,div')]
            .filter(node => (node.textContent || '').trim() === 'FFT');
          for (const label of labels) {
            let root = label.parentElement;
            for (let depth = 0; root && depth < 5; depth++, root = root.parentElement) {
              const input = root.querySelector('input[type="checkbox"]');
              if (!input) continue;
              const before = input.checked;
              input.click();
              return {before, after: input.checked};
            }
          }
          return null;
        }"""
    )
    assert result is not None, "FFT toggle was not found"
    assert result["before"] != result["after"], "FFT toggle did not change state"
    page.wait_for_timeout(500)


def _first_panel_signature(page) -> str:
    return str(
        page.evaluate(
            """() => {
              const groups = new Map();
              for (const canvas of document.querySelectorAll('canvas')) {
                const rect = canvas.getBoundingClientRect();
                if (rect.width < 90 || rect.height < 90 || canvas.width <= 0 || canvas.height <= 0) continue;
                const key = [Math.round(rect.x), Math.round(rect.y), Math.round(rect.width), Math.round(rect.height)].join(',');
                let group = groups.get(key);
                if (!group) {
                  group = {x:rect.x, y:rect.y, signature: 0};
                  groups.set(key, group);
                }
                const ctx = canvas.getContext('2d', {willReadFrequently: true}) || canvas.getContext('2d');
                if (!ctx) continue;
                let data;
                try {
                  data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
                } catch {
                  continue;
                }
                const pixels = canvas.width * canvas.height;
                const step = Math.max(1, Math.floor(pixels / 6000));
                for (let p = 0; p < pixels; p += step) {
                  const i = p * 4;
                  group.signature += data[i] * 3 + data[i + 1] * 5 + data[i + 2] * 7 + data[i + 3];
                }
              }
              const ordered = [...groups.values()].sort((a, b) => (a.y - b.y) || (a.x - b.x));
              if (!ordered.length) return "";
              return String(Math.round(ordered[0].signature));
            }"""
        )
    )


def _wheel_zoom_does_not_scroll(page, rect: dict) -> None:
    x = rect["x"] + rect["w"] * 0.5
    y = rect["y"] + rect["h"] * 0.5
    before = int(page.evaluate("window.scrollY"))
    page.mouse.move(x, y)
    page.mouse.wheel(0, 420)
    page.wait_for_timeout(300)
    after = int(page.evaluate("window.scrollY"))
    assert after == before, f"wheel zoom scrolled the page from {before} to {after}"


def _assert_canvas_pixels(page) -> dict:
    stats = page.evaluate(
        """() => {
          const visible = [];
          for (const canvas of document.querySelectorAll('canvas')) {
            const rect = canvas.getBoundingClientRect();
            if (rect.width < 20 || rect.height < 20 || canvas.width <= 0 || canvas.height <= 0) continue;
            const ctx = canvas.getContext('2d', {willReadFrequently: true}) || canvas.getContext('2d');
            if (!ctx) continue;
            let data;
            try {
              data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
            } catch {
              continue;
            }
            const pixels = canvas.width * canvas.height;
            const step = Math.max(1, Math.floor(pixels / 20000));
            let nonblank = 0, white = 0, crosshair = 0;
            for (let p = 0; p < pixels; p += step) {
              const i = p * 4;
              const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];
              if (a > 20 && (r > 8 || g > 8 || b > 8)) nonblank++;
              if (a > 80 && r > 220 && g > 220 && b > 220) white++;
              if (a > 80 && ((r > 180 && g < 190 && b < 190) || (r > 180 && g > 160 && b < 120))) crosshair++;
            }
            visible.push({x:rect.x, y:rect.y, w:rect.width, h:rect.height, nonblank, white, crosshair});
          }
          return {
            count: visible.length,
            nonblank: visible.filter(c => c.nonblank > 20).length,
            white: visible.reduce((sum, c) => sum + c.white, 0),
            crosshair: visible.reduce((sum, c) => sum + c.crosshair, 0),
            visible,
          };
        }"""
    )
    assert stats["count"] >= 4, f"too few visible canvases: {stats!r}"
    assert stats["nonblank"] >= 2, f"expected nonblank scientific canvases: {stats!r}"
    assert stats["white"] > 10, f"expected visible white scale-bar/text pixels: {stats!r}"
    assert stats["crosshair"] > 4, f"expected visible red/yellow crosshair pixels: {stats!r}"
    return stats


def _assert_multiple_grid_layout(page, expected_cols: int, max_gap_above_grid: float = 240) -> dict:
    layout = page.evaluate(
        """([expectedCols, maxGapAboveGrid]) => {
          const panels = [...document.querySelectorAll('[role="button"][aria-label^="Show4DSTEM multiple panel"]')]
            .map(node => {
              const r = node.getBoundingClientRect();
              return {left:r.left, right:r.right, top:r.top, bottom:r.bottom, width:r.width, height:r.height};
            })
            .filter(r => r.width > 20 && r.height > 20)
            .sort((a, b) => (a.top - b.top) || (a.left - b.left));
          const grid = panels.length ? document.querySelector('[role="button"][aria-label="Show4DSTEM multiple panel 1"]').parentElement : null;
          const gr = grid ? grid.getBoundingClientRect() : {right:0, bottom:0};
          const firstRowTop = panels.length ? panels[0].top : 0;
          const firstRow = panels.filter(r => Math.abs(r.top - firstRowTop) < 3);
          let maxGap = 0;
          for (let i = 1; i < firstRow.length; i++) {
            maxGap = Math.max(maxGap, firstRow[i].left - firstRow[i - 1].right);
          }
          const maxRight = panels.reduce((v, r) => Math.max(v, r.right), 0);
          const maxBottom = panels.reduce((v, r) => Math.max(v, r.bottom), 0);
          const firstPanelTop = panels.length ? panels[0].top : 0;
          const upperLargeCanvases = [...document.querySelectorAll('canvas')]
            .map(canvas => canvas.getBoundingClientRect())
            .filter(r => r.width > 90 && r.height > 90 && r.bottom <= firstPanelTop + 1);
          const upperBottom = upperLargeCanvases.reduce((v, r) => Math.max(v, r.bottom), 0);
          const gapAboveGrid = firstPanelTop - upperBottom;
          return {
            panelCount: panels.length,
            firstRowCount: firstRow.length,
            maxGap,
            rightWaste: gr.right - maxRight,
            bottomWaste: gr.bottom - maxBottom,
            gapAboveGrid,
            maxGapAboveGrid,
            scrollWidth: document.documentElement.scrollWidth,
            clientWidth: document.documentElement.clientWidth,
            expectedCols,
          };
        }""",
        [expected_cols, max_gap_above_grid],
    )
    assert layout["panelCount"] >= expected_cols, f"not enough multiple panels: {layout!r}"
    assert layout["firstRowCount"] == expected_cols, f"compare_cols layout mismatch: {layout!r}"
    assert layout["maxGap"] <= 4, f"unexpected gap between multiple panels: {layout!r}"
    assert layout["rightWaste"] <= 6 and layout["bottomWaste"] <= 6, f"unexpected grid whitespace: {layout!r}"
    assert layout["gapAboveGrid"] <= max_gap_above_grid, f"unexpected whitespace above grid: {layout!r}"
    assert layout["scrollWidth"] <= layout["clientWidth"] + 8, f"horizontal overflow: {layout!r}"
    return layout


def _click_multiple_panel(page, panel_number: int) -> None:
    panel = page.locator(f'[role="button"][aria-label="Show4DSTEM multiple panel {panel_number}"]')
    assert panel.count() == 1
    panel.click()
    page.wait_for_timeout(500)
    text = page.evaluate("document.body.innerText")
    assert f"(Frame {panel_number}/" in text, f"clicking panel {panel_number} did not select its DP frame"


def _assert_selected_dp_mode_changes_dp(page, panel_number: int) -> None:
    before = _first_panel_signature(page)
    _click_multiple_panel(page, panel_number)
    after = _first_panel_signature(page)
    assert after and after != before, "selected compare DP mode did not redraw the DP panel"


def _open_page(page, url: str, title: str, min_canvases: int) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_function(
        """([title, minCanvases]) =>
          document.body.innerText.includes(title)
          && document.querySelectorAll('canvas').length >= minCanvases""",
        arg=[title, min_canvases],
        timeout=120_000,
    )
    page.wait_for_timeout(1200)


@pytest.mark.skipif(
    os.environ.get("QT_RUN_SHOW4DSTEM_EXPORT_VISUAL_TESTS") != "1",
    reason="set QT_RUN_SHOW4DSTEM_EXPORT_VISUAL_TESTS=1 to run exported Show4DSTEM visual smoke tests",
)
def test_exported_show4dstem_real_data_visual_smoke(tmp_path: Path) -> None:
    from playwright.sync_api import Error, sync_playwright

    chrome = _chrome_executable()
    data = _load_real_4dstem_fixture()
    root = tmp_path / "show4dstem-export-visual"
    root.mkdir()
    pages = _export_show4dstem_pages(root, data)
    artifacts = _artifact_root(tmp_path)
    saved_html = artifacts / "html"
    shutil.rmtree(saved_html, ignore_errors=True)
    shutil.copytree(root, saved_html)

    port = int(os.environ.get("QT_SHOW4DSTEM_VISUAL_PORT", "0")) or _free_port()
    server = subprocess.Popen(
        ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(root)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    summary: dict[str, object] = {
        "data_shape": list(data.shape),
        "artifacts": str(artifacts),
        "touch_logger": f"http://127.0.0.1:{port}/iphone-touch-logger.html",
        "saved_html": str(saved_html),
        "saved_touch_logger": str(saved_html / "iphone-touch-logger.html"),
    }
    try:
        _wait_for_http(port, server)
        with sync_playwright() as pw:
            launch_kwargs = {
                "headless": os.environ.get("QT_SHOW4DSTEM_VISUAL_HEADED") != "1",
                "args": ["--no-first-run", "--no-default-browser-check"],
            }
            if chrome is not None:
                launch_kwargs["executable_path"] = chrome
            try:
                browser = pw.chromium.launch(**launch_kwargs)
            except Error as exc:
                pytest.skip(f"Chromium could not be launched: {exc}")
            try:
                context = browser.new_context(viewport={"width": 1280, "height": 950})
                page = context.new_page()

                single_title = "Visual Smoke Show4DSTEM single"
                _open_page(page, f"http://127.0.0.1:{port}/{pages['single'].name}", single_title, 6)
                _drag_single_vi(page)
                single_rect = _large_canvas_rects(page)[1]
                _wheel_zoom_does_not_scroll(page, single_rect)
                _toggle_fft(page)
                single_pixels = _assert_canvas_pixels(page)
                page.screenshot(path=str(artifacts / "single-desktop.png"), full_page=False)

                multiple2_title = "Visual Smoke Show4DSTEM multiple-cols2"
                _open_page(page, f"http://127.0.0.1:{port}/{pages['multiple-cols2'].name}", multiple2_title, 12)
                layout2 = _assert_multiple_grid_layout(page, expected_cols=2)
                _drag_multiple_panel(page, panel_number=1)
                _click_multiple_panel(page, panel_number=3)
                panel3 = page.locator('[role="button"][aria-label="Show4DSTEM multiple panel 3"]').bounding_box()
                assert panel3 is not None
                _wheel_zoom_does_not_scroll(
                    page,
                    {"x": panel3["x"], "y": panel3["y"], "w": panel3["width"], "h": panel3["height"]},
                )
                _toggle_fft(page)
                multiple2_pixels = _assert_canvas_pixels(page)
                page.screenshot(path=str(artifacts / "multiple-cols2-desktop.png"), full_page=False)

                multiple4_title = "Visual Smoke Show4DSTEM multiple-cols4"
                _open_page(page, f"http://127.0.0.1:{port}/{pages['multiple-cols4'].name}", multiple4_title, 12)
                layout4 = _assert_multiple_grid_layout(page, expected_cols=4)
                _drag_multiple_panel(page, panel_number=1)
                _click_multiple_panel(page, panel_number=4)
                multiple4_pixels = _assert_canvas_pixels(page)
                page.screenshot(path=str(artifacts / "multiple-cols4-desktop.png"), full_page=False)

                selected_title = "Visual Smoke Show4DSTEM multiple-selected"
                _open_page(page, f"http://127.0.0.1:{port}/{pages['multiple-selected'].name}", selected_title, 12)
                selected_layout = _assert_multiple_grid_layout(page, expected_cols=3)
                _assert_selected_dp_mode_changes_dp(page, panel_number=3)
                _drag_multiple_panel(page, panel_number=2)
                selected_pixels = _assert_canvas_pixels(page)
                page.screenshot(path=str(artifacts / "multiple-selected-desktop.png"), full_page=False)

                mobile = context.new_page()
                mobile.set_viewport_size({"width": 390, "height": 844})
                _open_page(mobile, f"http://127.0.0.1:{port}/{pages['multiple-cols4'].name}", multiple4_title, 12)
                mobile_layout = _assert_multiple_grid_layout(mobile, expected_cols=4, max_gap_above_grid=80)
                _drag_multiple_panel(mobile, panel_number=1)
                mobile_pixels = _assert_canvas_pixels(mobile)
                mobile.screenshot(path=str(artifacts / "multiple-cols4-mobile.png"), full_page=False)
                mobile.close()

                summary.update(
                    {
                        "single_pixels": single_pixels,
                        "multiple_cols2_layout": layout2,
                        "multiple_cols2_pixels": multiple2_pixels,
                        "multiple_cols4_layout": layout4,
                        "multiple_cols4_pixels": multiple4_pixels,
                        "multiple_selected_layout": selected_layout,
                        "multiple_selected_pixels": selected_pixels,
                        "mobile_layout": mobile_layout,
                        "mobile_pixels": mobile_pixels,
                    }
                )
                print(json.dumps(summary, indent=2))
                context.close()
            finally:
                browser.close()
    finally:
        with suppress(Exception):
            server.terminate()
            server.wait(timeout=10)
        with suppress(Exception):
            server.kill()
        shutil.rmtree(root, ignore_errors=True)
