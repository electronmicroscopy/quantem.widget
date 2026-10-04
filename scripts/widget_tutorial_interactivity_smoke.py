#!/usr/bin/env python3
"""Verify that rendered tutorial HTML keeps widget interactions alive."""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import sys
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


def _wait_for_scientific_pixels(output: Any, timeout_ms: int) -> None:
    """Wait for an asynchronous widget render to replace its loading frame."""
    from widget_browser_smoke import _image_nonblank

    deadline = time.monotonic() + min(timeout_ms, 15_000) / 1000
    last_stats: dict[str, Any] = {}
    while time.monotonic() < deadline:
        passed, last_stats = _image_nonblank(output.screenshot(timeout=timeout_ms))
        if passed:
            return
        time.sleep(0.75)
    raise AssertionError(f"scientific output did not become ready: {last_stats}")


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


def _widget_pages(book_dir: Path) -> list[tuple[str, Path, int]]:
    """Return every rendered tutorial page that embeds live widget views."""

    pages: list[tuple[str, Path, int]] = []
    tutorials = book_dir / "tutorials"
    for path in sorted(tutorials.glob("*.html")):
        source = path.read_text(encoding="utf-8", errors="ignore")
        view_count = source.count("application/vnd.jupyter.widget-view+json")
        if view_count:
            pages.append((path.stem, path, view_count))
    return pages


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _set_first_slider(page: Any) -> bool:
    sliders = page.get_by_role("slider")
    for index in range(sliders.count()):
        slider = sliders.nth(index)
        if not slider.is_visible():
            continue
        before = slider.get_attribute("aria-valuenow") or slider.get_attribute("value")
        slider.press("ArrowRight")
        page.wait_for_timeout(100)
        after = slider.get_attribute("aria-valuenow") or slider.get_attribute("value")
        if before != after:
            return True
        slider.press("ArrowLeft")
        page.wait_for_timeout(100)
        after = slider.get_attribute("aria-valuenow") or slider.get_attribute("value")
        if before != after:
            return True
    return False


def _ignorable_docs_framework_message(message: str) -> bool:
    """Return true only for known Jupyter Book/theme bootstrap noise."""

    return message in {
        "Got invalid theme mode: . Resetting to auto.",
        "Identifier 'THEBE_JS_URL' has already been declared",
    }


def _drive_first_output(page: Any, output: Any) -> str:
    """Change one live control, falling back to a real pointer drag."""

    if _set_first_slider(page):
        return "slider"
    box = output.bounding_box()
    if box is None:
        return "none"
    _drag_box(page, box)
    return "pointer-drag"


def _verify_book_interactions(
    book_dir: Path,
    artifact_dir: Path,
    *,
    timeout_ms: int,
    headed: bool,
    require_hardware_webgpu: bool,
    page_names: set[str] | None = None,
) -> dict[str, Any]:
    """Drive every tutorial page containing a baked anywidget view."""

    from playwright.sync_api import Error, sync_playwright

    from widget_browser_smoke import (
        _scientific_output_screenshots,
        _webgpu_adapter_info,
    )

    pages_to_check = _widget_pages(book_dir)
    if page_names:
        pages_to_check = [page for page in pages_to_check if page[0] in page_names]
    if not pages_to_check:
        raise AssertionError(f"no rendered widget tutorial pages under {book_dir}")
    screenshots = artifact_dir / "tutorial-screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "screenshots").mkdir(parents=True, exist_ok=True)
    port = _free_port()
    results: list[dict[str, Any]] = []

    with _StaticServer(book_dir, port) as base_url:
        with sync_playwright() as pw:
            launch_kwargs: dict[str, Any] = {
                "headless": not headed,
                "args": [
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--ignore-gpu-blocklist",
                    "--enable-unsafe-webgpu",
                ],
            }
            if sys.platform != "darwin":
                launch_kwargs["args"].extend(
                    ["--enable-features=Vulkan,WebGPU", "--use-angle=vulkan", "--disable-gpu-sandbox"]
                )
            chrome = _chrome_executable()
            if chrome is not None:
                launch_kwargs["executable_path"] = chrome
            try:
                browser = pw.chromium.launch(**launch_kwargs)
            except Error as exc:
                raise RuntimeError(f"Chromium could not be launched: {exc}") from exc
            try:
                for page_index, (name, path, expected_views) in enumerate(pages_to_check):
                    print(
                        f"tutorial page {name}: loading {expected_views} widget view(s)",
                        flush=True,
                    )
                    page = browser.new_page(viewport={"width": 1440, "height": 1100})
                    console_errors: list[str] = []
                    page_errors: list[str] = []
                    ignored_framework_messages: list[str] = []
                    page.on(
                        "console",
                        lambda msg, errors=console_errors, ignored=ignored_framework_messages: (
                            ignored.append(msg.text)
                            if msg.type == "error" and _ignorable_docs_framework_message(msg.text)
                            else errors.append(msg.text)
                        )
                        if msg.type == "error" and "Failed to load resource" not in msg.text
                        else None,
                    )
                    page.on(
                        "pageerror",
                        lambda exc, errors=page_errors, ignored=ignored_framework_messages: (
                            ignored.append(str(exc))
                            if _ignorable_docs_framework_message(str(exc))
                            else errors.append(str(exc))
                        ),
                    )
                    relative = path.relative_to(book_dir).as_posix()
                    url = f"{base_url}/{relative}"
                    record: dict[str, Any] = {
                        "name": name,
                        "url": url,
                        "expected_widget_views": expected_views,
                        "scientific_outputs": [],
                        "errors": [],
                    }
                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                        page.wait_for_function(
                            "document.querySelectorAll('canvas, [data-quantem-scientific-output]').length > 0",
                            timeout=timeout_ms,
                        )
                        page.wait_for_timeout(1200)
                        adapter = _webgpu_adapter_info(page)
                        record["webgpu_adapter"] = adapter
                        if require_hardware_webgpu and (
                            not adapter.get("available") or adapter.get("software")
                        ):
                            record["errors"].append(
                                f"hardware WebGPU adapter required: {adapter}"
                            )

                        fallback_count = page.locator("img.quantem-static-fallback").count()
                        record["static_fallback_count"] = fallback_count
                        if fallback_count:
                            record["errors"].append(
                                f"found {fallback_count} duplicate static fallback image(s)"
                            )

                        visible_load_errors = page.locator(
                            '[data-quantem-load-error="true"]:visible'
                        ).count()
                        record["visible_load_error_count"] = visible_load_errors
                        if visible_load_errors:
                            record["errors"].append(
                                f"found {visible_load_errors} visible widget load error(s)"
                            )

                        # Let notebook pages with several independent WebGPU
                        # widgets finish device creation before screenshots;
                        # rapid capture can otherwise monopolize compositing
                        # while the first durable frame is still queued.
                        page.wait_for_timeout(1800)
                        output_locator = page.locator("[data-quantem-scientific-output]")
                        record["scientific_output_count"] = output_locator.count()
                        if not output_locator.count():
                            record["errors"].append("no marked scientific output rendered")
                        else:
                            for output_index in range(output_locator.count()):
                                output = output_locator.nth(output_index)
                                output.scroll_into_view_if_needed(timeout=timeout_ms)
                                _wait_for_scientific_pixels(output, timeout_ms)
                            first = output_locator.first
                            first.scroll_into_view_if_needed(timeout=timeout_ms)
                            before = first.screenshot(timeout=timeout_ms)
                            action = _drive_first_output(page, first)
                            page.wait_for_timeout(900)
                            after = first.screenshot(timeout=timeout_ms)
                            record["interaction"] = action
                            record["interaction_changed_pixels"] = _sha256(before) != _sha256(after)
                            if not record["interaction_changed_pixels"]:
                                box = first.bounding_box()
                                if box is not None and action != "pointer-drag":
                                    _drag_box(page, box)
                                    page.wait_for_timeout(700)
                                    after = first.screenshot(timeout=timeout_ms)
                                    record["interaction"] = f"{action}+pointer-drag"
                                    record["interaction_changed_pixels"] = _sha256(before) != _sha256(after)
                            if not record["interaction_changed_pixels"]:
                                record["errors"].append(
                                    "real slider/pointer interaction did not change scientific pixels"
                                )
                            for output_index in range(output_locator.count()):
                                output = output_locator.nth(output_index)
                                output.scroll_into_view_if_needed(timeout=timeout_ms)
                                _wait_for_scientific_pixels(output, timeout_ms)

                        outputs = _scientific_output_screenshots(
                            page,
                            artifact_dir,
                            f"tutorial-{name}",
                            timeout_ms,
                        )
                        record["scientific_outputs"] = outputs
                        for output in outputs:
                            if not output.get("passed"):
                                record["errors"].append(
                                    f"scientific output {output.get('name')!r} is black, blank, or flat: "
                                    f"{output.get('stats', {})}"
                                )
                        full_page = screenshots / f"{name}.png"
                        page.screenshot(path=str(full_page), full_page=True, timeout=timeout_ms)
                        record["screenshot"] = str(full_page)
                        record["console_errors"] = console_errors
                        record["page_errors"] = page_errors
                        record["ignored_framework_messages"] = ignored_framework_messages
                        record["errors"].extend(console_errors)
                        record["errors"].extend(page_errors)
                    except Exception as exc:
                        record["errors"].append(str(exc))
                    finally:
                        page.close()
                        if page_index + 1 < len(pages_to_check):
                            # AnyWidget pages can own several GPU devices and
                            # large resident textures. A fresh browser process
                            # per tutorial proves each page independently and
                            # prevents a previous stress page from exhausting
                            # the next page's WebGPU resources.
                            browser.close()
                            time.sleep(5.0)
                            browser = pw.chromium.launch(**launch_kwargs)
                    record["passed"] = not record["errors"]
                    results.append(record)
                    print(
                        f"tutorial page {name}: {'PASS' if record['passed'] else 'FAIL'} "
                        f"({record.get('scientific_output_count', 0)} scientific outputs)",
                        flush=True,
                    )
            finally:
                browser.close()

    return {
        "mode": "jupyter-book",
        "book_dir": str(book_dir),
        "artifact_dir": str(artifact_dir),
        "require_hardware_webgpu": require_hardware_webgpu,
        "page_count": len(results),
        "passed_count": sum(1 for result in results if result["passed"]),
        "passed": all(result["passed"] for result in results),
        "pages": results,
    }


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
    parser.add_argument(
        "--book-dir",
        help="Drive every rendered tutorial page containing live widget views instead of executing one notebook.",
    )
    parser.add_argument("--headed", action="store_true", help="Show Chrome while driving rendered tutorial pages.")
    parser.add_argument(
        "--require-hardware-webgpu",
        action="store_true",
        help="Fail unless each rendered tutorial page acquires a non-software WebGPU adapter.",
    )
    parser.add_argument(
        "--page",
        action="append",
        default=[],
        help="With --book-dir, check only this tutorial stem (repeatable).",
    )
    args = parser.parse_args(argv)

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    if args.book_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        result = _verify_book_interactions(
            Path(args.book_dir).expanduser().resolve(),
            artifact_dir,
            timeout_ms=max(30_000, int(args.timeout) * 1000),
            headed=bool(args.headed),
            require_hardware_webgpu=bool(args.require_hardware_webgpu),
            page_names=set(args.page) or None,
        )
        failed_pages = {
            page["name"] for page in result["pages"] if not page.get("passed")
        }
        if failed_pages:
            result["initial_failures"] = {
                page["name"]: page.get("errors", [])
                for page in result["pages"]
                if page["name"] in failed_pages
            }
            retry = _verify_book_interactions(
                Path(args.book_dir).expanduser().resolve(),
                artifact_dir,
                timeout_ms=max(30_000, int(args.timeout) * 1000),
                headed=bool(args.headed),
                require_hardware_webgpu=bool(args.require_hardware_webgpu),
                page_names=failed_pages,
            )
            retry_by_name = {page["name"]: page for page in retry["pages"]}
            result["pages"] = [
                retry_by_name.get(page["name"], page) for page in result["pages"]
            ]
            result["retried_pages"] = sorted(failed_pages)
            result["passed_count"] = sum(
                1 for page in result["pages"] if page.get("passed")
            )
            result["passed"] = all(
                page.get("passed") for page in result["pages"]
            )
    else:
        notebook = Path(args.notebook).expanduser().resolve()
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
                "passed": True,
            }
        )
    report = artifact_dir / "tutorial-interactivity-report.json"
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
