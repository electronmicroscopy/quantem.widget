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

    with _StaticServer(artifact_dir, port) as base_url:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs)
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

        show3d = _show3d_export(performance_report)
        fft_idle = _check_show3d_fft_idle(
            artifact_dir,
            str(show3d["path"]),
            timeout_ms=args.timeout_ms,
            min_fps=args.min_fps,
            idle_seconds=args.idle_seconds,
            headed=args.headed,
        )
        errors.extend(fft_idle["errors"])

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
        },
        "performance": performance_report,
        "browser": browser_report,
        "show3d_fft_idle": fft_idle,
        "errors": errors,
    }
    (artifact_dir / "heavy-signoff-report.json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    _write_index(artifact_dir, final_report)
    print(json.dumps(final_report, indent=2))
    print(f"Heavy performance signoff report: {artifact_dir / 'index.html'}")
    return 0 if final_report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
