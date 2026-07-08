#!/usr/bin/env python3
"""Profile an already exported QuantEM widget HTML page in Chromium.

This is the local-only gate for real exported HTML pages that already exist
outside the repository, for example a Tailscale-served report from a lab
workstation. It does not generate data or exports; it opens the provided URL,
checks canvas rendering, drives common widget controls, samples
requestAnimationFrame FPS, and writes a human-readable report.
"""

from __future__ import annotations

import argparse
import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from widget_browser_smoke import _chrome_executable, _image_nonblank, _measure_fps, _visible_canvas_boxes


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_artifact_dir() -> Path:
    return Path("/tmp") / "quantem-widget-external-html-profile" / _timestamp()


def _escape(value: object) -> str:
    return html.escape(str(value))


def _text_summary(page) -> dict[str, Any]:
    return page.evaluate(
        r"""() => {
          const text = document.body.innerText || "";
          const lines = text.split("\n").map((line) => line.trim()).filter(Boolean).slice(0, 24);
          const buttons = [...document.querySelectorAll("button")]
            .map((button) => (button.textContent || button.getAttribute("aria-label") || "").trim())
            .filter(Boolean);
          return { lines, buttons };
        }"""
    )


def _canvas_signature(page) -> list[dict[str, Any]]:
    """Return a cheap visual/layout signature for visible canvases."""

    return page.evaluate(
        r"""() => [...document.querySelectorAll("canvas")].map((canvas, index) => {
          const rect = canvas.getBoundingClientRect();
          if (rect.width < 24 || rect.height < 24 || canvas.width <= 0 || canvas.height <= 0) {
            return null;
          }
          let data = "";
          try {
            data = canvas.toDataURL("image/png").slice(0, 220);
          } catch (error) {
            data = `${canvas.width}x${canvas.height}:${rect.x},${rect.y}`;
          }
          return {
            index,
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            data,
          };
        }).filter(Boolean)"""
    )


def _click_button(page, label: str) -> bool:
    return bool(
        page.evaluate(
            r"""(label) => {
              const visible = (node) => {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return rect.width > 0 && rect.height > 0 &&
                  style.display !== "none" && style.visibility !== "hidden";
              };
              const wanted = label.toLowerCase();
              const matches = (button) => {
                const text = (button.textContent || "").trim().toLowerCase();
                const aria = (button.getAttribute("aria-label") || "").trim().toLowerCase();
                if (text === wanted || aria === wanted) {
                  return true;
                }
                if (wanted === "panels") {
                  return text.startsWith("panels ") || aria === "choose visible panels";
                }
                return false;
              };
              const candidates = [...document.querySelectorAll("button,[role='menuitem'],li")]
                .filter((button) => matches(button) && visible(button));
              if (!candidates.length) {
                return false;
              }
              candidates[0].click();
              return true;
            }""",
            label,
        )
    )


def _button_present(summary: dict[str, Any], label: str) -> bool:
    wanted = label.lower()
    return any(str(button).strip().lower() == wanted for button in summary.get("buttons", []))


def _first_hide_button(summary: dict[str, Any]) -> str | None:
    for button in summary.get("buttons", []):
        label = str(button).strip()
        if label.startswith("Hide "):
            return label
    return None


def _screenshot(page, path: Path, *, full_page: bool = False) -> dict[str, Any]:
    png = page.screenshot(path=str(path), full_page=full_page)
    nonblank, stats = _image_nonblank(png, min_unique=8, min_span=8)
    return {
        "path": str(path),
        "rel": f"screenshots/{path.name}",
        "nonblank": bool(nonblank),
        "stats": stats,
    }


def _primary_canvas_screenshot(page, path: Path) -> dict[str, Any]:
    boxes = _visible_canvas_boxes(page)
    if not boxes:
        return {"error": "no visible canvas"}
    box = sorted(boxes, key=lambda item: item["width"] * item["height"], reverse=True)[0]
    png = page.locator("canvas").nth(int(box["index"])).screenshot(path=str(path))
    nonblank, stats = _image_nonblank(png, min_unique=8, min_span=8)
    return {
        "box": box,
        "path": str(path),
        "rel": f"screenshots/{path.name}",
        "nonblank": bool(nonblank),
        "stats": stats,
    }


def _run_play_button_step(page, label: str, pause_labels: list[str], wait_ms: int, fps_ms: int) -> dict[str, Any]:
    before_summary = _text_summary(page)
    before_signature = _canvas_signature(page)
    clicked = _click_button(page, label)
    page.wait_for_timeout(wait_ms)
    after_summary = _text_summary(page)
    after_signature = _canvas_signature(page)
    fps = round(float(_measure_fps(page, fps_ms)), 1)
    paused_clicked = False
    for pause_label in pause_labels:
        if _click_button(page, pause_label):
            paused_clicked = True
            page.wait_for_timeout(250)
            break
    return {
        "label": label,
        "clicked": clicked,
        "paused_clicked": paused_clicked,
        "text_changed": before_summary != after_summary,
        "canvas_signature_changed": before_signature != after_signature,
        "fps": fps,
    }


def _run_hide_step(page, label: str, fps_ms: int) -> dict[str, Any]:
    before_count = len(_visible_canvas_boxes(page))
    before_signature = _canvas_signature(page)
    clicked = _click_button(page, label)
    page.wait_for_timeout(700)
    after_count = len(_visible_canvas_boxes(page))
    after_signature = _canvas_signature(page)
    fps = round(float(_measure_fps(page, fps_ms)), 1)
    restore_label = "Show " + label.removeprefix("Hide ")
    plain_label = label.removeprefix("Hide ")
    restore_clicked = _click_button(page, restore_label)
    if not restore_clicked and _click_button(page, "Panels"):
        page.wait_for_timeout(250)
        restore_clicked = _click_button(page, restore_label)
        if not restore_clicked:
            restore_clicked = _click_button(page, plain_label)
        if not restore_clicked:
            restore_clicked = _click_button(page, "Show all panels")
    page.wait_for_timeout(300)
    return {
        "label": label,
        "clicked": clicked,
        "restore_label": restore_label,
        "plain_restore_label": plain_label,
        "restore_clicked": restore_clicked,
        "canvas_count_before": before_count,
        "canvas_count_after": after_count,
        "canvas_signature_changed": before_signature != after_signature,
        "fps": fps,
    }


def _write_report(artifact_dir: Path, metrics: dict[str, Any]) -> None:
    errors = metrics.get("errors", [])
    status_html = (
        '<p class="pass">No blocking errors found.</p>'
        if not errors
        else '<ul class="fail">' + "".join(f"<li>{_escape(error)}</li>" for error in errors) + "</ul>"
    )
    rows = [
        ("URL", metrics.get("url")),
        ("HTTP status", metrics.get("status")),
        ("Ready time", f"{metrics.get('load_to_ready_s')} s"),
        ("Initial FPS", metrics.get("initial_fps")),
        ("Final FPS", metrics.get("final_fps")),
        ("Canvas count", metrics.get("initial_canvas_count")),
        ("Passed", metrics.get("passed")),
    ]
    summary_rows = "".join(f"<tr><th>{_escape(key)}</th><td>{_escape(value)}</td></tr>" for key, value in rows)
    steps_html = []
    for step in metrics.get("steps", []):
        shot = step.get("screenshot", {})
        image = f'<img src="{_escape(shot.get("rel", ""))}">' if shot.get("rel") else ""
        payload = {key: value for key, value in step.items() if key != "screenshot"}
        steps_html.append(
            f'<section class="card"><h2>{_escape(step.get("name"))}</h2>{image}'
            f"<pre>{_escape(json.dumps(payload, indent=2))}</pre></section>"
        )
    warnings = metrics.get("console_warnings_errors", [])
    warnings_html = "".join(
        f"<li>{_escape(item.get('type'))}: {_escape(item.get('text'))}</li>" for item in warnings
    ) or "<li>None captured</li>"
    initial = metrics.get("initial_screenshot", {})
    primary = metrics.get("initial_primary_canvas", {})
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>QuantEM external HTML profile</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172033; }}
    table {{ border-collapse: collapse; }}
    th {{ text-align: left; padding: 6px 16px 6px 0; vertical-align: top; white-space: nowrap; }}
    td {{ padding: 6px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 14px; margin: 14px 0; background: white; }}
    img {{ max-width: 100%; border: 1px solid #d8dee9; background: #f7f8fa; }}
    pre {{ white-space: pre-wrap; background: #f6f8fa; padding: 10px; border-radius: 6px; max-height: 360px; overflow: auto; }}
    code {{ background: #f0f2f5; padding: 1px 4px; border-radius: 4px; }}
    .pass {{ color: #087f23; font-weight: 700; }}
    .fail {{ color: #b00020; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>QuantEM external HTML profile</h1>
  {status_html}
  <section class="card"><h2>Summary</h2><table>{summary_rows}</table></section>
  <div class="grid">
    <section class="card"><h2>Initial viewport</h2><img src="{_escape(initial.get('rel', ''))}"></section>
    <section class="card"><h2>Initial primary canvas</h2><img src="{_escape(primary.get('rel', ''))}"><pre>{_escape(json.dumps(primary.get('stats'), indent=2))}</pre></section>
  </div>
  <div class="grid">{''.join(steps_html)}</div>
  <section class="card"><h2>Console warnings/errors</h2><ul>{warnings_html}</ul></section>
  <section class="card"><h2>Raw data</h2><p><a href="metrics.json">metrics.json</a></p></section>
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(doc, encoding="utf-8")


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - depends on local browser extras
        raise SystemExit("playwright is required for external HTML profiling") from exc

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    screenshot_dir = artifact_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    console: list[dict[str, str]] = []
    page_errors: list[str] = []
    metrics: dict[str, Any] = {
        "url": args.url,
        "artifact_dir": str(artifact_dir),
        "min_fps": args.min_fps,
        "viewport": {"width": args.viewport_width, "height": args.viewport_height},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not args.headed,
            executable_path=_chrome_executable(),
            args=["--disable-dev-shm-usage"],
        )
        try:
            context = browser.new_context(
                viewport={"width": args.viewport_width, "height": args.viewport_height},
                device_scale_factor=1,
            )
            page = context.new_page()
            page.on("console", lambda msg: console.append({"type": msg.type, "text": msg.text[:800]}))
            page.on("pageerror", lambda exc: page_errors.append(str(exc)[:1200]))

            started = time.perf_counter()
            response = page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_ms)
            page.wait_for_selector("canvas", state="visible", timeout=args.timeout_ms)
            page.wait_for_timeout(args.settle_ms)
            metrics["status"] = response.status if response else None
            metrics["load_to_ready_s"] = round(time.perf_counter() - started, 3)
            metrics["initial_summary"] = _text_summary(page)
            metrics["initial_canvas_count"] = len(_visible_canvas_boxes(page))
            metrics["initial_fps"] = round(float(_measure_fps(page, args.fps_sample_ms)), 1)
            metrics["initial_screenshot"] = _screenshot(page, screenshot_dir / "00-initial.png")
            metrics["initial_primary_canvas"] = _primary_canvas_screenshot(page, screenshot_dir / "00-primary-canvas.png")

            summary = metrics["initial_summary"]
            if _button_present(summary, "Play pages"):
                step = _run_play_button_step(
                    page,
                    "Play pages",
                    ["Pause pages", "Play pages"],
                    wait_ms=args.autoplay_wait_ms,
                    fps_ms=args.fps_sample_ms,
                )
                step["name"] = "page_autoplay"
                step["screenshot"] = _screenshot(page, screenshot_dir / "01-page-autoplay.png")
                metrics["steps"].append(step)

            if _button_present(summary, "Play forward"):
                step = _run_play_button_step(
                    page,
                    "Play forward",
                    ["Pause playback", "Stop and rewind to start"],
                    wait_ms=args.playback_wait_ms,
                    fps_ms=args.fps_sample_ms,
                )
                step["name"] = "frame_playback"
                step["screenshot"] = _screenshot(page, screenshot_dir / "02-frame-playback.png")
                metrics["steps"].append(step)

            hide_label = _first_hide_button(_text_summary(page))
            if hide_label:
                step = _run_hide_step(page, hide_label, fps_ms=args.fps_sample_ms)
                step["name"] = "hide_restore_panel"
                step["screenshot"] = _screenshot(page, screenshot_dir / "03-hide-panel.png")
                metrics["steps"].append(step)

            metrics["final_canvas_count"] = len(_visible_canvas_boxes(page))
            metrics["final_fps"] = round(float(_measure_fps(page, args.fps_sample_ms)), 1)
            metrics["final_summary"] = _text_summary(page)
            context.close()
        finally:
            browser.close()

    metrics["console_warnings_errors"] = [
        item for item in console if item["type"] in {"warning", "error"}
    ][: args.max_console_items]
    metrics["page_errors"] = page_errors

    errors: list[str] = []
    if metrics.get("status") != 200:
        errors.append(f"HTTP status was {metrics.get('status')}")
    if metrics.get("initial_canvas_count", 0) < args.min_canvases:
        errors.append(f"visible canvas count below {args.min_canvases}: {metrics.get('initial_canvas_count')}")
    if not metrics.get("initial_screenshot", {}).get("nonblank"):
        errors.append("initial viewport screenshot was blank or flat")
    if not metrics.get("initial_primary_canvas", {}).get("nonblank"):
        errors.append("initial primary canvas was blank or flat")
    if metrics.get("initial_fps", 0) < args.min_fps:
        errors.append(f"initial FPS {metrics.get('initial_fps')} below {args.min_fps}")
    if metrics.get("final_fps", 0) < args.min_fps:
        errors.append(f"final FPS {metrics.get('final_fps')} below {args.min_fps}")
    for step in metrics.get("steps", []):
        if not step.get("clicked"):
            errors.append(f"{step['name']} control was not clicked")
        if step.get("fps", args.min_fps) < args.min_fps:
            errors.append(f"{step['name']} FPS {step.get('fps')} below {args.min_fps}")
        if step["name"] in {"page_autoplay", "frame_playback"} and not (
            step.get("text_changed") or step.get("canvas_signature_changed")
        ):
            errors.append(f"{step['name']} did not change visible text or canvas signature")
        if step["name"] == "hide_restore_panel" and not step.get("canvas_signature_changed"):
            errors.append("hide/restore panel did not change visible canvas signature")
        if step["name"] == "hide_restore_panel" and not step.get("restore_clicked"):
            errors.append("hide/restore panel did not restore the hidden panel")
    if page_errors:
        errors.append("page JavaScript errors were reported")
    if args.fail_console_errors and metrics["console_warnings_errors"]:
        errors.append("console warnings/errors were reported")

    metrics["errors"] = errors
    metrics["passed"] = not errors
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_report(artifact_dir, metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Standalone exported widget HTML URL to profile.")
    parser.add_argument("--artifact-dir", default=str(_default_artifact_dir()), help="Directory for index.html and metrics.")
    parser.add_argument("--min-fps", type=float, default=30.0, help="Minimum rAF FPS for driven browser states.")
    parser.add_argument("--min-canvases", type=int, default=1, help="Minimum visible canvas count.")
    parser.add_argument("--fps-sample-ms", type=int, default=1500, help="Milliseconds for each rAF FPS sample.")
    parser.add_argument("--timeout-ms", type=int, default=180_000, help="Navigation and first-canvas timeout.")
    parser.add_argument("--settle-ms", type=int, default=2000, help="Delay after first canvas before measuring.")
    parser.add_argument("--autoplay-wait-ms", type=int, default=5000, help="Wait after clicking Play pages.")
    parser.add_argument("--playback-wait-ms", type=int, default=3500, help="Wait after clicking Play forward.")
    parser.add_argument("--viewport-width", type=int, default=1440)
    parser.add_argument("--viewport-height", type=int, default=980)
    parser.add_argument("--max-console-items", type=int, default=40)
    parser.add_argument("--fail-console-errors", action="store_true", help="Treat console warnings/errors as failures.")
    parser.add_argument("--headed", action="store_true", help="Run Chromium headed for local debugging.")
    args = parser.parse_args()

    metrics = run_profile(args)
    print(f"External HTML profile: {metrics['artifact_dir']}/index.html")
    if metrics["passed"]:
        print("PASS")
        return 0
    print("FAIL")
    for error in metrics["errors"]:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
