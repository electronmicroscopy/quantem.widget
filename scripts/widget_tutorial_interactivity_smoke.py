#!/usr/bin/env python3
"""Verify that rendered tutorial HTML keeps widget interactions alive."""

from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


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


class _StaticServer:
    def __init__(self, root: Path, port: int) -> None:
        self.root = root
        self.port = port
        self.httpd: http.server.ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> str:
        root = self.root

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, directory=str(root), **kwargs)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def copyfile(self, source: Any, outputfile: Any) -> None:
                try:
                    shutil.copyfileobj(source, outputfile)
                except (BrokenPipeError, ConnectionResetError):
                    return

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


def _wait_for_http(url: str) -> None:
    import urllib.request

    for _ in range(80):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"server did not respond: {url}")


def _render_notebook(notebook: Path, artifact_dir: Path, timeout: int) -> Path:
    if shutil.which("jupyter") is None:
        raise RuntimeError("jupyter was not found")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_stem = notebook.stem
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "--execute",
        str(notebook),
        "--output",
        output_stem,
        "--output-dir",
        str(artifact_dir),
        f"--ExecutePreprocessor.timeout={timeout}",
    ]
    subprocess.run(cmd, check=True)
    html = artifact_dir / f"{output_stem}.html"
    if not html.exists():
        raise RuntimeError(f"nbconvert did not write {html}")
    return html


def _dp_at(text: str) -> str:
    match = re.search(r"DP at \([^)]*\)", text)
    if not match:
        raise AssertionError("DP coordinate label was not found")
    return match.group(0)


def _widget_dp_at(panel: Any) -> str:
    text = panel.evaluate(
        """node => {
          const root = node.closest('.show4dstem-root') || node;
          return root.innerText || root.textContent || '';
        }"""
    )
    return _dp_at(text)


def _drag_box(page: Any, box: dict[str, float]) -> None:
    start_x = box["x"] + box["width"] * 0.30
    start_y = box["y"] + box["height"] * 0.35
    end_x = box["x"] + box["width"] * 0.76
    end_y = box["y"] + box["height"] * 0.72
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(end_x, end_y, steps=16)
    page.mouse.up()


def _verify_show4dstem_multiple_interaction(url: str, artifact_dir: Path) -> dict[str, Any]:
    from playwright.sync_api import Error, sync_playwright

    console: list[str] = []
    screenshots = {
        "before": artifact_dir / "show4dstem-tutorial-before-drag.png",
        "after": artifact_dir / "show4dstem-tutorial-after-drag.png",
    }

    with sync_playwright() as pw:
        launch_kwargs: dict[str, Any] = {
            "headless": os.environ.get("QT_TUTORIAL_SMOKE_HEADED") != "1",
            "args": ["--no-first-run", "--no-default-browser-check"],
        }
        chrome = _chrome_executable()
        if chrome is not None:
            launch_kwargs["executable_path"] = chrome
        try:
            browser = pw.chromium.launch(**launch_kwargs)
        except Error as exc:
            raise RuntimeError(f"Chromium could not be launched: {exc}") from exc
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.on("console", lambda msg: console.append(f"{msg.type}: {msg.text}"))
            page.on("pageerror", lambda exc: console.append(f"pageerror: {exc}"))
            page.goto(url, wait_until="networkidle", timeout=120_000)
            panel = page.locator('[role="button"][aria-label="Show4DSTEM multiple panel 1"]').first
            panel.scroll_into_view_if_needed(timeout=120_000)
            page.wait_for_timeout(1500)
            before = _widget_dp_at(panel)
            box = panel.bounding_box()
            if box is None or box["width"] < 40 or box["height"] < 40:
                raise AssertionError(f"multiple panel 1 is not visible: {box!r}")
            page.screenshot(path=str(screenshots["before"]), full_page=False)
            _drag_box(page, box)
            page.wait_for_timeout(1200)
            after = _widget_dp_at(panel)
            page.screenshot(path=str(screenshots["after"]), full_page=False)
            if after == before:
                raise AssertionError(f"drag did not update the rendered widget: {before}")
            canvas_count = page.locator("canvas").count()
            return {
                "url": url,
                "before": before,
                "after": after,
                "canvas_count": canvas_count,
                "screenshots": {key: str(path) for key, path in screenshots.items()},
                "console_tail": console[-20:],
            }
        finally:
            browser.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook",
        nargs="?",
        default="docs/tutorials/show4dstem.ipynb",
        help="Tutorial notebook to execute and render.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="/tmp/quantem-widget-tutorial-interactivity-smoke",
        help="Directory for rendered HTML, screenshots, and report JSON.",
    )
    parser.add_argument("--timeout", type=int, default=240, help="Per-cell nbconvert timeout in seconds.")
    parser.add_argument("--port", type=int, default=0, help="Local HTTP port. Default: choose a free port.")
    args = parser.parse_args(argv)

    notebook = Path(args.notebook).expanduser().resolve()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    html = _render_notebook(notebook, artifact_dir, timeout=args.timeout)
    port = int(args.port) or _free_port()
    with _StaticServer(artifact_dir, port) as base_url:
        url = f"{base_url}/{html.name}"
        _wait_for_http(url)
        result = _verify_show4dstem_multiple_interaction(url, artifact_dir)

    result.update(
        {
            "notebook": str(notebook),
            "html": str(html),
            "artifact_dir": str(artifact_dir),
        }
    )
    report = artifact_dir / "tutorial-interactivity-report.json"
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
