#!/usr/bin/env python3
"""Drive generated widget HTML exports in a real browser.

This script consumes the artifact directory created by ``widget_html_smoke.py``.
It opens every exported HTML page, verifies visible canvases are nonblank,
performs basic pointer/control interactions, captures screenshots, and writes a
browser smoke report next to the export matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import http.server
import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


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


@dataclass
class _StaticServer:
    root: Path
    port: int
    httpd: http.server.ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None

    def __enter__(self) -> str:
        root = self.root

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root), **kwargs)

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                return

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


def _image_nonblank(png_bytes: bytes, *, min_unique: int = 4, min_span: int = 8) -> tuple[bool, dict[str, Any]]:
    image = Image.open(BytesIO(png_bytes)).convert("RGB")
    # Downsample for cheap uniqueness checks while preserving blank/flat failures.
    image.thumbnail((160, 160))
    colors = image.getcolors(maxcolors=160 * 160 + 1) or []
    extrema = image.getextrema()
    span = max(hi - lo for lo, hi in extrema)
    return len(colors) >= min_unique and span >= min_span, {
        "width": image.width,
        "height": image.height,
        "unique_colors": len(colors),
        "max_channel_span": span,
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in name)


def _visible_canvas_boxes(page) -> list[dict[str, float]]:
    return page.evaluate(
        """() => [...document.querySelectorAll('canvas')].map((canvas, index) => {
          const rect = canvas.getBoundingClientRect();
          return {
            index,
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            backingWidth: canvas.width,
            backingHeight: canvas.height
          };
        }).filter(item =>
          item.width >= 24 && item.height >= 24 &&
          item.backingWidth > 0 && item.backingHeight > 0
        )"""
    )


def _click_text_controls(page, labels: list[str]) -> list[str]:
    clicked: list[str] = []
    for label in labels:
        did_click = page.evaluate(
            """(label) => {
              const wanted = label.toLowerCase();
              const nodes = [...document.querySelectorAll('button,[role="button"],label,span,div')]
                .filter(node => (node.textContent || '').trim().toLowerCase() === wanted);
              for (const node of nodes) {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                if (rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none') {
                  node.click();
                  return true;
                }
              }
              return false;
            }""",
            label,
        )
        if did_click:
            clicked.append(label)
            page.wait_for_timeout(120)
    return clicked


def _click_switches(page, limit: int) -> int:
    return int(
        page.evaluate(
            """(limit) => {
              let clicked = 0;
              for (const input of [...document.querySelectorAll('input[type="checkbox"]')]) {
                const host = input.closest('label') || input.parentElement || input;
                const rect = host.getBoundingClientRect();
                const style = getComputedStyle(host);
                if (rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none') {
                  input.click();
                  clicked++;
                  if (clicked >= limit) break;
                }
              }
              return clicked;
            }""",
            limit,
        )
    )


def _drag_first_slider(page) -> bool:
    slider = page.evaluate(
        """() => {
          const roots = [...document.querySelectorAll('.MuiSlider-root')];
          for (const root of roots) {
            const rect = root.getBoundingClientRect();
            if (rect.width > 40 && rect.height > 8) {
              return {x: rect.x, y: rect.y, width: rect.width, height: rect.height};
            }
          }
          return null;
        }"""
    )
    if not slider:
        return False
    y = slider["y"] + slider["height"] / 2
    page.mouse.move(slider["x"] + slider["width"] * 0.25, y)
    page.mouse.down()
    page.mouse.move(slider["x"] + slider["width"] * 0.72, y, steps=10)
    page.mouse.up()
    page.wait_for_timeout(180)
    return True


def _drive_canvas(page, box: dict[str, float]) -> None:
    x = box["x"] + box["width"] * 0.52
    y = box["y"] + box["height"] * 0.52
    page.mouse.move(x, y)
    page.mouse.wheel(0, -450)
    page.wait_for_timeout(140)
    page.mouse.down()
    page.mouse.move(x + min(40, box["width"] * 0.18), y + min(30, box["height"] * 0.18), steps=10)
    page.mouse.up()
    page.wait_for_timeout(180)


def _write_html_report(artifact_dir: Path, report: dict[str, Any]) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['widget'])}</td>"
        f"<td>{html.escape(row['variant'])}</td>"
        f"<td>{'pass' if row['passed'] else 'fail'}</td>"
        f"<td>{html.escape(str(row['canvas_count']))}</td>"
        f"<td>{html.escape(str(row['switches_clicked']))}</td>"
        f"<td>{html.escape(str(row['slider_dragged']))}</td>"
        f"<td>{html.escape(str(row['canvas_changed']))}</td>"
        f"<td><a href='{html.escape(row['screenshot'])}'>{html.escape(row['screenshot'])}</a></td>"
        f"<td>{html.escape('; '.join(row['errors']))}</td>"
        f"<td>{html.escape('; '.join(row.get('console_warnings', [])))}</td>"
        "</tr>"
        for row in report["pages"]
    )
    report_json = html.escape(json.dumps(report, indent=2))
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>quantem.widget browser smoke</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; line-height: 1.45; }}
    table {{ border-collapse: collapse; margin-top: 12px; min-width: 960px; }}
    th, td {{ border: 1px solid #ccd3db; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f5f7; }}
    code, pre {{ background: #f5f7f9; border-radius: 4px; }}
    code {{ padding: 2px 4px; }}
    pre {{ overflow: auto; padding: 12px; max-width: 1180px; }}
  </style>
</head>
<body>
  <h1>quantem.widget browser smoke</h1>
  <p>This report opens exported HTML in Chromium, checks nonblank rendering, and
  drives basic widget interactions.</p>
  <p>Passed: <strong>{report['passed']}</strong> / {len(report['pages'])}</p>
  <table>
    <thead><tr><th>Widget</th><th>Variant</th><th>Status</th><th>Canvases</th><th>Switches</th><th>Slider</th><th>Canvas changed</th><th>Screenshot</th><th>Errors</th><th>Warnings</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Machine-readable report</h2>
  <p><a href="browser-smoke-report.json">browser-smoke-report.json</a></p>
  <pre>{report_json}</pre>
</body>
</html>
"""
    (artifact_dir / "browser-smoke.html").write_text(page, encoding="utf-8")


def _check_page(context, base_url: str, artifact_dir: Path, row: dict[str, Any], timeout_ms: int) -> dict[str, Any]:
    page = context.new_page()
    browser_errors: list[str] = []
    console_errors: list[str] = []
    console_warnings: list[str] = []
    http_errors: list[str] = []
    page.on("pageerror", lambda exc: browser_errors.append(str(exc)))

    def _handle_console(msg) -> None:
        if msg.type != "error" or "Failed to load resource:" in msg.text:
            return
        if "Unable to preventDefault inside passive event listener invocation." in msg.text:
            console_warnings.append(msg.text)
            return
        console_errors.append(msg.text)

    page.on("console", _handle_console)
    page.on(
        "response",
        lambda response: http_errors.append(f"{response.status} {response.url}")
        if response.status >= 400
        and not response.url.endswith("/favicon.ico")
        and not response.url.endswith("/anywidget.js")
        and not response.url.endswith(".map")
        else None,
    )
    variant = str(row["variant"])
    widget = str(row["widget"])
    screenshot_name = f"screenshots/{_safe_name(variant)}.png"
    canvas_name = f"screenshots/{_safe_name(variant)}-canvas.png"
    result: dict[str, Any] = {
        "widget": widget,
        "variant": variant,
        "url": f"{base_url}/{Path(str(row['path'])).name}",
        "screenshot": screenshot_name,
        "canvas_screenshot": canvas_name,
        "canvas_count": 0,
        "canvas_nonblank": False,
        "canvas_changed": False,
        "switches_clicked": 0,
        "text_controls_clicked": [],
        "slider_dragged": False,
        "errors": [],
        "passed": False,
    }
    try:
        page.goto(result["url"], wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_function("document.body && document.body.innerText.length > 0", timeout=timeout_ms)
        page.wait_for_timeout(700)

        boxes = _visible_canvas_boxes(page)
        result["canvas_count"] = len(boxes)
        if widget != "showfolder" and not boxes:
            result["errors"].append("no visible canvas")

        if boxes:
            # Use the largest visible canvas as the primary render target.
            box = max(boxes, key=lambda item: item["width"] * item["height"])
            locator = page.locator("canvas").nth(int(box["index"]))
            before = locator.screenshot(timeout=timeout_ms)
            (artifact_dir / canvas_name).write_bytes(before)
            nonblank, image_stats = _image_nonblank(before)
            result["canvas_nonblank"] = nonblank
            result["canvas_stats"] = image_stats
            if not nonblank:
                result["errors"].append("primary canvas is blank or flat")

            _drive_canvas(page, box)
            after = locator.screenshot(timeout=timeout_ms)
            result["canvas_changed"] = _sha256(before) != _sha256(after)

        labels = ["Profile", "FFT", "ROI", "Lens", "Panels", "Stats", "Export"]
        result["text_controls_clicked"] = _click_text_controls(page, labels)
        result["switches_clicked"] = _click_switches(page, 3)
        result["slider_dragged"] = _drag_first_slider(page)
        page.wait_for_timeout(300)

        page.screenshot(path=str(artifact_dir / screenshot_name), full_page=True, timeout=timeout_ms)
        result["browser_errors"] = browser_errors
        result["console_errors"] = console_errors
        result["console_warnings"] = console_warnings
        result["http_errors"] = http_errors
        result["errors"].extend(browser_errors)
        result["errors"].extend(console_errors)
        result["errors"].extend(http_errors)

        if widget == "showfolder":
            has_folder_marker = page.evaluate("document.body.innerText.includes('0010')")
            if not has_folder_marker:
                result["errors"].append("showfolder marker 0010 not visible")
        elif boxes and not result["canvas_nonblank"]:
            result["errors"].append("render check failed")

        result["passed"] = not result["errors"]
        return result
    finally:
        page.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Directory created by widget_html_smoke.py.")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--headed", action="store_true", help="Show the Chromium window while driving.")
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument("--max-pages", type=int, default=0, help="Limit pages for debugging; 0 means all.")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("playwright is required for browser smoke testing") from exc

    artifact_dir = args.artifact_dir.resolve()
    report_path = artifact_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"missing {report_path}; run scripts/widget_html_smoke.py first")
    (artifact_dir / "screenshots").mkdir(parents=True, exist_ok=True)
    export_report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = list(export_report["exports"])
    if args.max_pages:
        rows = rows[: args.max_pages]

    chrome = _chrome_executable()
    launch_kwargs: dict[str, Any] = {
        "headless": not args.headed,
        "args": [
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-search-engine-choice-screen",
        ],
    }
    if chrome is not None:
        launch_kwargs["executable_path"] = chrome

    port = args.port or _free_port()
    started_at = time.time()
    with _StaticServer(artifact_dir, port) as base_url:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs)
            context = browser.new_context(viewport={"width": 1280, "height": 950})
            try:
                pages = [_check_page(context, base_url, artifact_dir, row, args.timeout_ms) for row in rows]
            finally:
                context.close()
                browser.close()

    report = {
        "artifact_dir": str(artifact_dir),
        "created_at_unix": started_at,
        "base_url": f"http://127.0.0.1:{port}",
        "headed": bool(args.headed),
        "passed": sum(1 for page in pages if page["passed"]),
        "pages": pages,
    }
    (artifact_dir / "browser-smoke-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_html_report(artifact_dir, report)
    print(json.dumps(report, indent=2))
    print(f"Browser smoke report: {artifact_dir / 'browser-smoke.html'}")
    if report["passed"] != len(pages):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
