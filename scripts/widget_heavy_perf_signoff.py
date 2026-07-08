#!/usr/bin/env python3
"""Run the local-only real-data widget performance signoff.

This is intentionally not a normal CI script. It uses local HPC/workstation real
microscopy data when available, writes artifacts under ``/tmp`` by default, and
drives the exported HTML in Chromium. Do not commit generated data, screenshots,
or reports from this script unless a release explicitly asks for them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from widget_browser_smoke import _StaticServer, _chrome_executable, _free_port, _measure_fps


def _timestamp_dir() -> Path:
    return Path("/tmp/quantem-widget-heavy-signoff") / time.strftime("%Y%m%d-%H%M%S")


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _write_command_log(path: Path, result: subprocess.CompletedProcess[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "$ " + " ".join(result.args if isinstance(result.args, list) else [str(result.args)]),
                f"exit_code={result.returncode}",
                "",
                result.stdout,
            ]
        ),
        encoding="utf-8",
    )


def _show3d_export(report: dict[str, Any]) -> dict[str, Any]:
    show3d_exports = [
        item for item in report.get("exports", [])
        if item.get("widget") == "show3d" and "bin" in str(item.get("encoding", ""))
    ]
    if not show3d_exports:
        show3d_exports = [item for item in report.get("exports", []) if item.get("widget") == "show3d"]
    if not show3d_exports:
        raise RuntimeError("performance smoke did not produce a Show3D export")
    return min(show3d_exports, key=lambda item: float(item.get("size_mb", 0) or 0))


def _smallest_export(report: dict[str, Any], widget: str) -> dict[str, Any]:
    exports = [item for item in report.get("exports", []) if item.get("widget") == widget]
    if not exports:
        raise RuntimeError(f"performance smoke did not produce a {widget} export")
    return min(exports, key=lambda item: float(item.get("size_mb", 0) or 0))


def _chrome_launch_kwargs(*, headed: bool) -> dict[str, Any]:
    chrome = _chrome_executable()
    launch_kwargs: dict[str, Any] = {
        "headless": not headed,
        "args": [
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-search-engine-choice-screen",
        ],
    }
    if chrome is not None:
        launch_kwargs["executable_path"] = chrome
    return launch_kwargs


def _text_button(page, label_prefix: str) -> str:
    return str(
        page.evaluate(
            """(labelPrefix) => {
              const visible = (node) => {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return rect.width > 0 && rect.height > 0 &&
                  style.visibility !== 'hidden' && style.display !== 'none';
              };
              const button = [...document.querySelectorAll('button')]
                .find((node) => visible(node) && (node.textContent || '').trim().startsWith(labelPrefix));
              return button ? (button.textContent || '').trim().replace(/\\s+/g, ' ') : '';
            }""",
            label_prefix,
        )
    )


def _hide_first_panel_from_menu(page) -> dict[str, Any]:
    before_text = _text_button(page, "Panels")
    opened = bool(
        page.evaluate(
            """() => {
              const visible = (node) => {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return rect.width > 0 && rect.height > 0 &&
                  style.visibility !== 'hidden' && style.display !== 'none';
              };
              const button = [...document.querySelectorAll('button')]
                .find((node) =>
                  visible(node) &&
                  node.getAttribute('aria-label') === 'Choose visible panels' &&
                  (node.textContent || '').trim().startsWith('Panels')
                );
              if (!button) return false;
              button.click();
              return true;
            }"""
        )
    )
    if not opened:
        return {"opened": False, "clicked": False, "before": before_text, "after": before_text}
    page.wait_for_timeout(150)
    clicked = bool(
        page.evaluate(
            """() => {
              const visible = (node) => {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return rect.width > 0 && rect.height > 0 &&
                  style.visibility !== 'hidden' && style.display !== 'none';
              };
              const items = [...document.querySelectorAll('[role="menuitem"], .MuiMenuItem-root')]
                .filter((node) => {
                  const text = (node.textContent || '').trim();
                  const disabled = node.getAttribute('aria-disabled') === 'true' ||
                    node.classList.contains('Mui-disabled');
                  return visible(node) && !disabled && text &&
                    !/show all panels|reset order/i.test(text);
                });
              if (!items.length) return false;
              items[0].click();
              return true;
            }"""
        )
    )
    page.wait_for_timeout(120)
    for _ in range(3):
        if int(page.locator('[role="menu"], .MuiPopover-root, .MuiModal-root').count()) == 0:
            break
        page.keyboard.press("Escape")
        page.wait_for_timeout(100)
    return {"opened": opened, "clicked": clicked, "before": before_text, "after": _text_button(page, "Panels")}


def _page_slider_value(page) -> str | None:
    value = page.evaluate(
        """() => {
          const input = document.querySelector('input[aria-label="Page"]');
          return input ? String(input.value ?? '') : null;
        }"""
    )
    return None if value is None else str(value)


def _page_slider_max(page) -> int:
    value = page.evaluate(
        """() => {
          const input = document.querySelector('input[aria-label="Page"]');
          const raw = input ? Number(input.max || 0) : 0;
          return Number.isFinite(raw) ? Math.max(0, Math.round(raw)) : 0;
        }"""
    )
    return int(value or 0)


def _set_page_slider_by_pointer(page, target_idx: int, *, timeout_ms: int) -> None:
    target = page.evaluate(
        """(targetIdx) => {
          const input = document.querySelector('input[aria-label="Page"]');
          if (!input) throw new Error('Page slider input not found');
          const root = input.closest('.MuiSlider-root') || input.parentElement;
          if (!root) throw new Error('Page slider root not found');
          const rect = root.getBoundingClientRect();
          const max = Math.max(0, Math.round(Number(input.max || 0)));
          const clamped = Math.max(0, Math.min(max, Math.round(Number(targetIdx || 0))));
          if (max <= 0) throw new Error('Page slider has no second page');
          const bucket = (clamped + 0.5) / (max + 1);
          return {
            x: rect.left + (rect.width * bucket),
            y: rect.top + rect.height / 2,
            value: String(clamped),
          };
        }""",
        target_idx,
    )
    page.mouse.click(float(target["x"]), float(target["y"]))
    page.wait_for_function(
        """(targetValue) => {
          const input = document.querySelector('input[aria-label="Page"]');
          return input && String(Math.round(Number(input.value || 0))) === String(targetValue);
        }""",
        arg=target["value"],
        timeout=min(1000, timeout_ms),
    )


def _check_paged_widget_scrub(
    artifact_dir: Path,
    export_file: str,
    *,
    widget: str,
    timeout_ms: int,
    headed: bool,
) -> dict[str, Any]:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("playwright is required for heavy browser signoff") from exc

    port = _free_port()
    screenshot = artifact_dir / f"{widget}-paged-scrub.png"
    console_errors: list[str] = []
    page_errors: list[str] = []
    probe_errors: list[str] = []
    page_durations_ms: list[float] = []
    fps = 0.0
    slider_count = 0
    slider_before: str | None = None
    slider_after: str | None = None
    hide_probe: dict[str, Any] = {}
    panels_after_scrub = ""
    perf_before: dict[str, Any] | None = None
    perf_after: dict[str, Any] | None = None
    try:
        with _StaticServer(artifact_dir, port) as base_url:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(**_chrome_launch_kwargs(headed=headed))
                try:
                    page = browser.new_page(viewport={"width": 1440, "height": 1050})
                    page.on("pageerror", lambda exc: page_errors.append(str(exc)))
                    page.on(
                        "console",
                        lambda msg: console_errors.append(msg.text)
                        if msg.type == "error" and "Failed to load resource:" not in msg.text
                        else None,
                    )
                    page.goto(f"{base_url}/{Path(export_file).name}", wait_until="domcontentloaded", timeout=timeout_ms)
                    page.wait_for_function("document.querySelectorAll('canvas').length > 0", timeout=timeout_ms)
                    page.wait_for_timeout(1200)
                    slider_count = int(page.locator('input[aria-label="Page"]').count())
                    if slider_count != 1:
                        raise RuntimeError(f"{widget} expected one Page slider, found {slider_count}")
                    slider = page.locator('input[aria-label="Page"]').first
                    slider.focus(timeout=timeout_ms)
                    page.keyboard.press("Home")
                    page.wait_for_timeout(160)
                    slider_before = _page_slider_value(page)
                    hide_probe = _hide_first_panel_from_menu(page)
                    if not hide_probe.get("clicked"):
                        raise RuntimeError(f"{widget} could not hide a panel from the Panels menu: {hide_probe}")
                    fps = _measure_fps(page, 900)
                    perf_before = page.evaluate("() => window.__quantemShow3DPerf || null")
                    max_page_idx = _page_slider_max(page)
                    if max_page_idx < 1:
                        raise RuntimeError(f"{widget} Page slider has no second page")
                    scrub_targets = [max_page_idx, 0, max_page_idx, 0]
                    for _ in range(4):
                        before = _page_slider_value(page)
                        target = scrub_targets[len(page_durations_ms) % len(scrub_targets)]
                        t0 = time.perf_counter()
                        try:
                            _set_page_slider_by_pointer(page, target, timeout_ms=timeout_ms)
                        except PlaywrightTimeoutError:
                            probe_errors.append(
                                f"{widget} Page slider did not move from {before!r} to {target!r}"
                            )
                        page.wait_for_timeout(60)
                        page_durations_ms.append((time.perf_counter() - t0) * 1000)
                    slider_after = _page_slider_value(page)
                    panels_after_scrub = _text_button(page, "Panels")
                    perf_after = page.evaluate("() => window.__quantemShow3DPerf || null")
                    page.screenshot(path=str(screenshot), full_page=True, timeout=timeout_ms)
                finally:
                    browser.close()
    except Exception as exc:  # pragma: no cover - browser/runtime guard
        probe_errors.append(f"{widget} paged scrub probe failed: {exc}")

    errors = probe_errors + list(page_errors) + list(console_errors)
    hidden_before = str(hide_probe.get("before") or "")
    hidden_after = str(hide_probe.get("after") or "")
    max_page_ms = max(page_durations_ms) if page_durations_ms else None
    avg_page_ms = sum(page_durations_ms) / len(page_durations_ms) if page_durations_ms else None
    if hide_probe and hidden_before == hidden_after:
        errors.append(f"{widget} Panels button did not reflect hidden panel state: {hide_probe}")
    if hidden_after and panels_after_scrub and hidden_after != panels_after_scrub:
        errors.append(f"{widget} hidden panel state changed after page scrubs: {hidden_after} -> {panels_after_scrub}")
    if max_page_ms is not None and max_page_ms > 500:
        errors.append(f"{widget} page scrub max latency {max_page_ms:.1f} ms exceeded 500 ms")
    if widget == "show3d" and perf_after:
        cache_size = int(perf_after.get("offlineFrameCacheSize") or 0)
        cache_limit = int(perf_after.get("offlineFrameCacheLimit") or 0)
        if cache_limit > 0 and cache_size > cache_limit:
            errors.append(f"Show3D offline frame cache exceeded limit: {cache_size} > {cache_limit}")

    return {
        "widget": widget,
        "file": Path(export_file).name,
        "screenshot": screenshot.name if screenshot.exists() else None,
        "slider_count": slider_count,
        "slider_before": slider_before,
        "slider_after": slider_after,
        "hide_probe": hide_probe,
        "panels_after_scrub": panels_after_scrub,
        "fps": round(float(fps), 1),
        "page_scrub_avg_ms": round(avg_page_ms, 1) if avg_page_ms is not None else None,
        "page_scrub_max_ms": round(max_page_ms, 1) if max_page_ms is not None else None,
        "perf_before": perf_before,
        "perf_after": perf_after,
        "errors": errors,
        "passed": not errors,
    }


def _check_show3d_fft_idle(
    artifact_dir: Path,
    show3d_file: str,
    *,
    timeout_ms: int,
    min_fps: float,
    idle_seconds: float,
    headed: bool,
) -> dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("playwright is required for heavy browser signoff") from exc

    port = _free_port()
    screenshot = artifact_dir / "show3d-fft-idle.png"
    with _StaticServer(artifact_dir, port) as base_url:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**_chrome_launch_kwargs(headed=headed))
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1050})
                console_errors: list[str] = []
                page_errors: list[str] = []
                page.on("pageerror", lambda exc: page_errors.append(str(exc)))
                page.on(
                    "console",
                    lambda msg: console_errors.append(msg.text)
                    if msg.type == "error" and "Failed to load resource:" not in msg.text
                    else None,
                )
                page.goto(f"{base_url}/{Path(show3d_file).name}", wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_function("document.querySelectorAll('canvas').length > 0", timeout=timeout_ms)
                page.wait_for_timeout(1000)
                before = page.evaluate("() => window.__quantemShow3DPerf || null")
                start_hits = int((before or {}).get("fftCacheHits") or 0)
                start_computes = int((before or {}).get("fftComputes") or 0)
                fps = _measure_fps(page, 1500)
                page.wait_for_timeout(int(idle_seconds * 1000))
                after = page.evaluate("() => window.__quantemShow3DPerf || null")
                end_hits = int((after or {}).get("fftCacheHits") or 0)
                end_computes = int((after or {}).get("fftComputes") or 0)
                page.screenshot(path=str(screenshot), full_page=True, timeout=timeout_ms)
            finally:
                browser.close()

    hit_growth = end_hits - start_hits
    compute_growth = end_computes - start_computes
    errors = list(page_errors) + list(console_errors)
    if fps < min_fps:
        errors.append(f"Show3D FFT page FPS {fps:.1f} below {min_fps:.1f}")
    if hit_growth > 2:
        errors.append(f"Show3D FFT cache hits grew while idle: {start_hits} -> {end_hits}")
    if compute_growth > 0:
        errors.append(f"Show3D FFT recomputed while idle: {start_computes} -> {end_computes}")
    return {
        "file": Path(show3d_file).name,
        "screenshot": screenshot.name,
        "fps": round(float(fps), 1),
        "min_fps": float(min_fps),
        "idle_seconds": float(idle_seconds),
        "perf_before": before,
        "perf_after": after,
        "fft_cache_hit_growth": hit_growth,
        "fft_compute_growth": compute_growth,
        "errors": errors,
        "passed": not errors,
    }


def _check_show3d_fft_stats_toggle(
    artifact_dir: Path,
    show3d_file: str,
    *,
    timeout_ms: int,
    toggles: int,
    headed: bool,
) -> dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("playwright is required for heavy browser signoff") from exc

    port = _free_port()
    screenshot = artifact_dir / "show3d-fft-stats-toggle.png"
    durations_ms: list[float] = []
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    labels_before: list[str] = []
    labels_after: list[str] = []
    console_errors: list[str] = []
    page_errors: list[str] = []
    probe_errors: list[str] = []
    computing_text = False
    try:
        with _StaticServer(artifact_dir, port) as base_url:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(**_chrome_launch_kwargs(headed=headed))
                try:
                    page = browser.new_page(viewport={"width": 1440, "height": 1050})
                    page.on("pageerror", lambda exc: page_errors.append(str(exc)))
                    page.on(
                        "console",
                        lambda msg: console_errors.append(msg.text)
                        if msg.type == "error" and "Failed to load resource:" not in msg.text
                        else None,
                    )
                    page.goto(f"{base_url}/{Path(show3d_file).name}", wait_until="domcontentloaded", timeout=timeout_ms)
                    page.wait_for_function(
                        """
                        () => document.querySelectorAll('.quantem-fft-quality-label').length > 0
                          || Boolean(window.__quantemShow3DPerf?.lastFftMetricLabel)
                        """,
                        timeout=timeout_ms,
                    )
                    page.wait_for_timeout(500)
                    before = page.evaluate("() => window.__quantemShow3DPerf || null")
                    labels_before = page.evaluate(
                        """
                        () => {
                          const labels = [...document.querySelectorAll('.quantem-fft-quality-label')]
                            .map(node => node.textContent.trim())
                            .filter(Boolean);
                          const debugLabel = window.__quantemShow3DPerf?.lastFftMetricLabel;
                          return labels.length ? labels : (debugLabel ? [debugLabel] : []);
                        }
                        """
                    )

                    stats = page.locator('input[aria-label="Toggle statistics readout"]')
                    if stats.count() != 1:
                        raise RuntimeError("Show3D stats switch was not uniquely found")
                    for _ in range(toggles):
                        target = not bool(stats.is_checked())
                        t0 = time.perf_counter()
                        stats.set_checked(target, force=True, timeout=timeout_ms)
                        page.wait_for_timeout(120)
                        durations_ms.append((time.perf_counter() - t0) * 1000)

                    after = page.evaluate("() => window.__quantemShow3DPerf || null")
                    labels_after = page.evaluate(
                        """
                        () => {
                          const labels = [...document.querySelectorAll('.quantem-fft-quality-label')]
                            .map(node => node.textContent.trim())
                            .filter(Boolean);
                          const debugLabel = window.__quantemShow3DPerf?.lastFftMetricLabel;
                          return labels.length ? labels : (debugLabel ? [debugLabel] : []);
                        }
                        """
                    )
                    computing_text = page.evaluate("() => document.body.innerText.includes('Computing FFT')")
                    page.screenshot(path=str(screenshot), full_page=True, timeout=timeout_ms)
                finally:
                    browser.close()
    except Exception as exc:  # pragma: no cover - browser/runtime guard
        probe_errors.append(f"Show3D stats-toggle probe failed: {exc}")

    start_fft = int((before or {}).get("fftComputes") or 0)
    end_fft = int((after or {}).get("fftComputes") or 0)
    start_metric = int((before or {}).get("fftMetricComputes") or 0)
    end_metric = int((after or {}).get("fftMetricComputes") or 0)
    errors = probe_errors + list(page_errors) + list(console_errors)
    if not labels_before:
        errors.append("Show3D FFT metric label was not visible before stats toggles")
    if labels_before != labels_after:
        errors.append(f"Show3D FFT metric label changed during stats toggles: {labels_before} -> {labels_after}")
    if end_fft - start_fft > 0:
        errors.append(f"Show3D FFT recomputed during stats toggles: {start_fft} -> {end_fft}")
    if end_metric - start_metric > 0:
        errors.append(f"Show3D FFT metric recomputed during stats toggles: {start_metric} -> {end_metric}")
    if computing_text:
        errors.append("Show3D showed 'Computing FFT' after stats toggles")

    return {
        "file": Path(show3d_file).name,
        "screenshot": screenshot.name,
        "toggles": int(toggles),
        "avg_toggle_ms": round(sum(durations_ms) / len(durations_ms), 1) if durations_ms else None,
        "max_toggle_ms": round(max(durations_ms), 1) if durations_ms else None,
        "labels_before": labels_before,
        "labels_after": labels_after,
        "perf_before": before,
        "perf_after": after,
        "fft_compute_growth": end_fft - start_fft,
        "fft_metric_compute_growth": end_metric - start_metric,
        "errors": errors,
        "passed": not errors,
    }


def _write_index(artifact_dir: Path, report: dict[str, Any]) -> None:
    links = "\n".join(
        f"<li><a href='{Path(str(item['path'])).name}'>{item['widget']} {item['encoding']}</a> "
        f"({item['size_mb']:.2f} MB, {item['seconds']:.2f}s)</li>"
        for item in report["performance"].get("exports", [])
    )
    browser_link = (
        "<li><a href='browser-smoke.html'>Browser smoke report</a></li>"
        if (artifact_dir / "browser-smoke.html").exists()
        else ""
    )
    fft_link = (
        "<li><a href='show3d-fft-idle.png'>Show3D FFT idle screenshot</a></li>"
        if (artifact_dir / "show3d-fft-idle.png").exists()
        else ""
    )
    fft_stats_link = (
        "<li><a href='show3d-fft-stats-toggle.png'>Show3D FFT stats-toggle screenshot</a></li>"
        if (artifact_dir / "show3d-fft-stats-toggle.png").exists()
        else ""
    )
    page_links = "\n".join(
        f"<li><a href='{name}'>{name}</a></li>"
        for name in ("show2d-paged-scrub.png", "show3d-paged-scrub.png")
        if (artifact_dir / name).exists()
    )
    report_json = json.dumps(report, indent=2)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>quantem.widget heavy performance signoff</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; line-height: 1.45; }}
    code, pre {{ background: #f5f7f9; border-radius: 4px; }}
    code {{ padding: 2px 4px; }}
    pre {{ overflow: auto; padding: 12px; max-width: 1180px; }}
    .warn {{ border-left: 4px solid #b54708; padding: 8px 12px; background: #fff7ed; }}
  </style>
</head>
<body>
  <h1>quantem.widget heavy performance signoff</h1>
  <p class="warn">Local-only real-data report. Keep lab workstation paths, screenshots,
  generated HTML, and timing JSON out of GitHub unless explicitly approved.</p>
  <p>Result: <strong>{'PASS' if report['passed'] else 'FAIL'}</strong></p>
  <h2>Artifacts</h2>
  <ul>
    {links}
    {browser_link}
    {fft_link}
    {fft_stats_link}
    {page_links}
    <li><a href="heavy-signoff-report.json">Machine-readable report</a></li>
  </ul>
  <h2>Machine-readable report</h2>
  <pre>{report_json}</pre>
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--quick", action="store_true", help="Use smaller real-derived targets for script iteration.")
    parser.add_argument("--show2d-panels", type=int, default=None)
    parser.add_argument("--show2d-size", type=int, default=None)
    parser.add_argument("--show3d-panels", type=int, default=None)
    parser.add_argument("--show3d-frames", type=int, default=None)
    parser.add_argument("--show3d-size", type=int, default=None)
    parser.add_argument("--show3d-export-downsample", type=int, default=4)
    parser.add_argument("--min-fps", type=float, default=30.0)
    parser.add_argument("--timeout-ms", type=int, default=90_000)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--stats-toggles", type=int, default=8)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--skip-browser", action="store_true", help="Generate exports only; do not claim UI performance signoff.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    artifact_dir = (args.artifact_dir or _timestamp_dir()).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    perf_cmd = [
        sys.executable,
        "scripts/widget_performance_smoke.py",
        "--artifact-dir",
        str(artifact_dir),
        "--show3d-export-downsample",
        str(args.show3d_export_downsample),
        "--skip-show3d-full-export",
        "--max-single-mb",
        "350",
    ]
    if args.quick:
        perf_cmd.append("--quick")
    for root_path in args.search_root:
        perf_cmd.extend(["--search-root", str(root_path)])
    for flag, value in [
        ("--show2d-panels", args.show2d_panels),
        ("--show2d-size", args.show2d_size),
        ("--show3d-panels", args.show3d_panels),
        ("--show3d-frames", args.show3d_frames),
        ("--show3d-size", args.show3d_size),
    ]:
        if value is not None:
            perf_cmd.extend([flag, str(value)])

    performance_result = _run(perf_cmd, cwd=root)
    _write_command_log(artifact_dir / "performance-smoke.log", performance_result)
    if performance_result.returncode != 0:
        print(performance_result.stdout)
        return performance_result.returncode

    performance_report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    browser_report: dict[str, Any] | None = None
    fft_idle: dict[str, Any] | None = None
    fft_stats_toggle: dict[str, Any] | None = None
    show2d_paged_scrub: dict[str, Any] | None = None
    show3d_paged_scrub: dict[str, Any] | None = None
    errors: list[str] = []

    if args.skip_browser:
        errors.append("browser checks skipped; this is not a full UI performance signoff")
    else:
        browser_cmd = [
            sys.executable,
            "scripts/widget_browser_smoke.py",
            "--artifact-dir",
            str(artifact_dir),
            "--min-fps",
            str(args.min_fps),
            "--timeout-ms",
            str(args.timeout_ms),
        ]
        if args.headed:
            browser_cmd.append("--headed")
        browser_result = _run(browser_cmd, cwd=root)
        _write_command_log(artifact_dir / "browser-smoke.log", browser_result)
        if (artifact_dir / "browser-smoke-report.json").exists():
            browser_report = json.loads((artifact_dir / "browser-smoke-report.json").read_text(encoding="utf-8"))
        if browser_result.returncode != 0:
            errors.append("browser smoke failed; see browser-smoke.log")

        show2d = _smallest_export(performance_report, "show2d")
        show2d_paged_scrub = _check_paged_widget_scrub(
            artifact_dir,
            str(show2d["path"]),
            widget="show2d",
            timeout_ms=args.timeout_ms,
            headed=args.headed,
        )
        errors.extend(show2d_paged_scrub["errors"])
        show3d = _show3d_export(performance_report)
        show3d_paged_scrub = _check_paged_widget_scrub(
            artifact_dir,
            str(show3d["path"]),
            widget="show3d",
            timeout_ms=args.timeout_ms,
            headed=args.headed,
        )
        errors.extend(show3d_paged_scrub["errors"])
        fft_idle = _check_show3d_fft_idle(
            artifact_dir,
            str(show3d["path"]),
            timeout_ms=args.timeout_ms,
            min_fps=args.min_fps,
            idle_seconds=args.idle_seconds,
            headed=args.headed,
        )
        errors.extend(fft_idle["errors"])
        fft_stats_toggle = _check_show3d_fft_stats_toggle(
            artifact_dir,
            str(show3d["path"]),
            timeout_ms=args.timeout_ms,
            toggles=args.stats_toggles,
            headed=args.headed,
        )
        errors.extend(fft_stats_toggle["errors"])

    final_report = {
        "passed": not errors,
        "local_only": True,
        "normal_ci": False,
        "artifact_dir": str(artifact_dir),
        "real_data_policy": (
            "Generated artifacts and local lab workstation data paths are for local signoff only; "
            "do not upload real data or heavy generated HTML to GitHub."
        ),
        "thresholds": {
            "min_fps": float(args.min_fps),
            "fft_idle_seconds": float(args.idle_seconds),
            "stats_toggles": int(args.stats_toggles),
        },
        "performance": performance_report,
        "browser": browser_report,
        "show2d_paged_scrub": show2d_paged_scrub,
        "show3d_paged_scrub": show3d_paged_scrub,
        "show3d_fft_idle": fft_idle,
        "show3d_fft_stats_toggle": fft_stats_toggle,
        "errors": errors,
    }
    (artifact_dir / "heavy-signoff-report.json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    _write_index(artifact_dir, final_report)
    print(json.dumps(final_report, indent=2))
    print(f"Heavy performance signoff report: {artifact_dir / 'index.html'}")
    return 0 if final_report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
