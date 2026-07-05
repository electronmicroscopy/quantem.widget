#!/usr/bin/env python3
"""Run the local-only real-data Show4DSTEM heavy performance signoff.

This gate is intentionally not normal CI. It uses local 4D-STEM master files,
measures the lazy MPS/chunking path, exports a standalone HTML viewer, then
drives that exported viewer in Chromium. Generated reports, screenshots, and
private lab paths stay under ``/tmp`` unless a maintainer explicitly asks for
them.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path
from typing import Any

from widget_browser_smoke import (
    _StaticServer,
    _chrome_executable,
    _free_port,
    _measure_fps,
    _visible_canvas_boxes,
)


DEFAULT_ROOTS = [
    Path("/data"),
    Path("/mnt/data"),
    Path("/Volumes"),
]


def _timestamp_dir() -> Path:
    return Path("/tmp/quantem-widget-show4dstem-heavy-signoff") / time.strftime("%Y%m%d-%H%M%S")


def _env_roots() -> list[Path]:
    raw = os.environ.get("QUANTEM_WIDGET_4DSTEM_ROOTS") or os.environ.get("QUANTEM_WIDGET_REAL_DATA_ROOTS") or ""
    return [Path(item).expanduser() for item in raw.split(os.pathsep) if item.strip()]


def _memory_snapshot(label: str) -> dict[str, Any]:
    snap: dict[str, Any] = {"label": label, "time": time.time()}
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        snap["rss_mb"] = round(proc.memory_info().rss / 1024**2, 1)
    except Exception:
        scale = 1024**2 if platform.system() == "Darwin" else 1024
        snap["rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale, 1)
        snap["rss_source"] = "resource_ru_maxrss"
    try:
        import torch

        if hasattr(torch, "mps"):
            snap["mps_available"] = bool(torch.backends.mps.is_available())
            for name in ["current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"]:
                fn = getattr(torch.mps, name, None)
                if callable(fn):
                    snap[f"torch_mps_{name}_mb"] = round(float(fn()) / 1024**2, 1)
        if torch.cuda.is_available():
            snap["cuda_allocated_mb"] = round(float(torch.cuda.memory_allocated()) / 1024**2, 1)
            snap["cuda_reserved_mb"] = round(float(torch.cuda.memory_reserved()) / 1024**2, 1)
    except Exception as exc:
        snap["gpu_memory_error"] = str(exc)[:160]
    return snap


def _roots_from_args(paths: list[Path]) -> list[Path]:
    roots: list[Path] = []
    for root in [*paths, *_env_roots(), *DEFAULT_ROOTS]:
        root = root.expanduser()
        if root.exists() and root not in roots:
            roots.append(root)
    return roots


def _discover_real_masters(
    roots: list[Path],
    *,
    pattern: str,
    scan_size: int | None,
    limit: int,
    ready_only: bool,
) -> tuple[list[Path], list[str]]:
    from quantem.widget.io import discover_masters, is_master_ready

    notes: list[str] = []
    masters: list[Path] = []
    seen: set[str] = set()
    scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
    for root in roots:
        if len(masters) >= limit:
            break
        try:
            found = discover_masters(
                root,
                pattern=pattern,
                recursive=True,
                scan_shape=scan_shape,
                verbose=False,
            )
        except Exception as exc:
            notes.append(f"{root}: discovery skipped ({str(exc)[:120]})")
            continue
        notes.append(f"{root}: discovered {len(found)} master candidate(s)")
        for item in found:
            path = Path(item).expanduser().resolve()
            key = str(path)
            if key in seen:
                continue
            if ready_only:
                try:
                    if not is_master_ready(path):
                        notes.append(f"{path.name}: not ready yet")
                        continue
                except Exception as exc:
                    notes.append(f"{path.name}: readiness check failed ({str(exc)[:120]})")
                    continue
            masters.append(path)
            seen.add(key)
            if len(masters) >= limit:
                break
    return masters, notes


def _describe_chunks(live: Any) -> dict[str, Any]:
    multi = getattr(live, "multi", None)
    datasets = list(getattr(multi, "datasets", []) or [])
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for idx, dataset in enumerate(datasets):
        if dataset is None:
            rows.append({"index": idx, "ready": False})
            continue
        chunks = list(getattr(dataset, "chunks", []) or [])
        chunk_shapes = [list(getattr(chunk, "shape", ())) for chunk in chunks[:8]]
        chunk_bytes = [int(getattr(chunk, "nbytes", 0) or 0) for chunk in chunks]
        total_bytes += sum(chunk_bytes)
        rows.append(
            {
                "index": idx,
                "ready": True,
                "name": (getattr(multi, "names", []) or [None])[idx],
                "shape": list(getattr(dataset, "shape", ())),
                "det_bin": int(getattr(dataset, "det_bin", 1) or 1),
                "fast_bin": int(getattr(dataset, "fast_bin", 0) or 0),
                "chunk_count": len(chunks),
                "chunk_shapes_sample": chunk_shapes,
                "chunk_bytes_mb": [round(value / 1024**2, 1) for value in chunk_bytes[:8]],
                "resident_mb": round(sum(chunk_bytes) / 1024**2, 1),
                "fast_sidecar_ready": getattr(dataset, "fast_vi", None) is not None,
            }
        )
    return {
        "type": type(multi).__name__ if multi is not None else None,
        "shape": list(getattr(multi, "shape", ())) if multi is not None else [],
        "n_total": len(datasets),
        "n_ready": int(getattr(multi, "n_ready", 0) or 0),
        "det_bin": int(getattr(multi, "det_bin", 1) or 1) if multi is not None else None,
        "resident_mb": round(total_bytes / 1024**2, 1),
        "datasets": rows,
    }


def _timed(label: str, records: list[dict[str, Any]], func):
    t0 = time.perf_counter()
    before = _memory_snapshot(f"{label}:before")
    result = func()
    after = _memory_snapshot(f"{label}:after")
    records.append(
        {
            "label": label,
            "seconds": round(time.perf_counter() - t0, 3),
            "memory_before": before,
            "memory_after": after,
        }
    )
    return result


def _export_widget(widget: Any, artifact_dir: Path, *, dtype: str, det_bin: int) -> dict[str, Any]:
    path = artifact_dir / f"show4dstem-real-{dtype}-bin{det_bin}.html"
    t0 = time.perf_counter()
    widget.export_html(path, encoding=dtype, downsample=det_bin, title="Show4DSTEM heavy signoff")
    seconds = time.perf_counter() - t0
    return {
        "widget": "show4dstem",
        "variant": f"show4dstem-real-{dtype}-bin{det_bin}",
        "encoding": dtype,
        "det_bin": det_bin,
        "path": str(path),
        "seconds": round(seconds, 3),
        "size_mb": round(path.stat().st_size / 1024**2, 2),
    }


def _browser_gpu_info(page) -> dict[str, Any]:
    return page.evaluate(
        """async () => {
          const info = { userAgent: navigator.userAgent, webgpu: Boolean(navigator.gpu) };
          if (!navigator.gpu) return info;
          try {
            const adapter = await navigator.gpu.requestAdapter();
            info.adapter = adapter ? (adapter.info || {}) : null;
            info.adapterName = adapter?.info?.description || adapter?.info?.vendor || null;
          } catch (err) {
            info.error = String(err && err.message ? err.message : err);
          }
          return info;
        }"""
    )


def _drag_box(page, box: dict[str, float], *, steps: int = 16) -> float:
    x0 = box["x"] + box["width"] * 0.35
    y0 = box["y"] + box["height"] * 0.35
    x1 = box["x"] + box["width"] * 0.68
    y1 = box["y"] + box["height"] * 0.64
    t0 = time.perf_counter()
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x1, y1, steps=steps)
    page.mouse.up()
    page.wait_for_timeout(120)
    return round((time.perf_counter() - t0) * 1000, 1)


def _drive_browser_export(
    artifact_dir: Path,
    export: dict[str, Any],
    *,
    min_fps: float,
    timeout_ms: int,
    headed: bool,
) -> dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("playwright is required for Show4DSTEM browser signoff") from exc

    port = _free_port()
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

    results: dict[str, Any] = {
        "file": Path(str(export["path"])).name,
        "errors": [],
        "passed": False,
        "min_fps": float(min_fps),
    }
    screenshot = artifact_dir / "show4dstem-browser-signoff.png"
    with _StaticServer(artifact_dir, port) as base_url:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1050})
                console_errors: list[str] = []
                console_warnings: list[str] = []
                page_errors: list[str] = []
                page.on("pageerror", lambda exc: page_errors.append(str(exc)))

                def _handle_console(msg) -> None:
                    if msg.type != "error" or "Failed to load resource:" in msg.text:
                        return
                    if "Unable to preventDefault inside passive event listener invocation." in msg.text:
                        console_warnings.append(msg.text)
                        return
                    console_errors.append(msg.text)

                page.on("console", _handle_console)
                page.goto(f"{base_url}/{Path(str(export['path'])).name}", wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_function("document.querySelectorAll('canvas').length >= 2", timeout=timeout_ms)
                page.wait_for_timeout(1000)
                results["browser_gpu"] = _browser_gpu_info(page)
                boxes = _visible_canvas_boxes(page)
                results["canvas_count"] = len(boxes)
                if len(boxes) < 2:
                    results["errors"].append("expected at least two canvases for diffraction + virtual image")
                else:
                    ranked = sorted(boxes, key=lambda item: item["width"] * item["height"], reverse=True)
                    virtual_box = ranked[0]
                    detector_box = ranked[1]
                    results["initial_fps"] = round(float(_measure_fps(page, 1200)), 1)
                    results["scan_position_drag_ms"] = _drag_box(page, virtual_box)
                    results["scan_position_fps"] = round(float(_measure_fps(page, 1200)), 1)
                    t0 = time.perf_counter()
                    results["detector_drag_ms"] = _drag_box(page, detector_box)
                    results["virtual_detector_recompute_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    results["detector_drag_fps"] = round(float(_measure_fps(page, 1200)), 1)
                    page.mouse.wheel(0, -400)
                    page.wait_for_timeout(140)
                    results["wheel_zoom_fps"] = round(float(_measure_fps(page, 1200)), 1)
                page.screenshot(path=str(screenshot), full_page=True, timeout=timeout_ms)
                results["screenshot"] = screenshot.name
                results["console_errors"] = console_errors
                results["console_warnings"] = console_warnings
                results["page_errors"] = page_errors
                results["errors"].extend(page_errors)
                results["errors"].extend(console_errors)
            finally:
                browser.close()

    for key in ["initial_fps", "scan_position_fps", "detector_drag_fps", "wheel_zoom_fps"]:
        value = float(results.get(key, 0) or 0)
        if value < min_fps:
            results["errors"].append(f"{key} {value:.1f} below {min_fps:.1f}")
    results["passed"] = not results["errors"]
    return results


def _write_index(artifact_dir: Path, report: dict[str, Any]) -> None:
    report_json = html.escape(json.dumps(report, indent=2))
    exports = "\n".join(
        f"<li><a href='{html.escape(Path(item['path']).name)}'>{html.escape(item['variant'])}</a> "
        f"({item['size_mb']:.2f} MB, {item['seconds']:.2f}s)</li>"
        for item in report.get("exports", [])
    )
    browser = report.get("browser") or {}
    screenshot = browser.get("screenshot")
    shot_html = f"<p><a href='{html.escape(screenshot)}'>Browser screenshot</a></p>" if screenshot else ""
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Show4DSTEM heavy performance signoff</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; line-height: 1.45; }}
    table {{ border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #cbd5df; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f5f7; }}
    pre {{ background: #f5f7f9; padding: 12px; overflow: auto; max-width: 1180px; }}
    .warn {{ border-left: 4px solid #b54708; padding: 8px 12px; background: #fff7ed; }}
  </style>
</head>
<body>
  <h1>Show4DSTEM heavy performance signoff</h1>
  <p class="warn">Local-only real-data report. Do not commit private data paths,
  generated HTML, screenshots, or timing JSON unless explicitly approved.</p>
  <p>Result: <strong>{'PASS' if report['passed'] else 'FAIL'}</strong></p>
  <h2>Exports</h2>
  <ul>{exports}</ul>
  {shot_html}
  <h2>Machine-readable report</h2>
  <p><a href="show4dstem-heavy-signoff-report.json">show4dstem-heavy-signoff-report.json</a></p>
  <pre>{report_json}</pre>
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--pattern", default="*_master.h5")
    parser.add_argument("--scan-size", type=int, default=None)
    parser.add_argument("--max-masters", type=int, default=2)
    parser.add_argument("--det-bin", type=int, default=4)
    parser.add_argument("--export-det-bin", type=int, default=4)
    parser.add_argument("--encoding", choices=["uint8", "uint16"], default="uint8")
    parser.add_argument("--min-fps", type=float, default=30.0)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--skip-browser", action="store_true", help="Measure backend/export only; do not claim UI performance signoff.")
    parser.add_argument("--allow-unready", action="store_true", help="Include discovered masters even if readiness checks fail.")
    parser.add_argument("--quick", action="store_true", help="Use one master and browser-suitable binning for script iteration.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    artifact_dir = (args.artifact_dir or _timestamp_dir()).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    max_masters = 1 if args.quick else max(1, int(args.max_masters))
    roots = _roots_from_args(args.search_root)
    masters, discovery_notes = _discover_real_masters(
        roots,
        pattern=args.pattern,
        scan_size=args.scan_size,
        limit=max_masters,
        ready_only=not args.allow_unready,
    )
    if not masters:
        report = {
            "passed": False,
            "reason": "no real 4D-STEM master files found",
            "local_only": True,
            "normal_ci": False,
            "search_roots": [str(root) for root in roots],
            "discovery_notes": discovery_notes,
        }
        (artifact_dir / "show4dstem-heavy-signoff-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_index(artifact_dir, report)
        print(f"No real Show4DSTEM masters found. Report: {artifact_dir / 'index.html'}")
        return 2

    from quantem.widget import Show4DSTEM
    from quantem.widget.multidataset_mps import load_macbook_datasets

    timing: list[dict[str, Any]] = []
    errors: list[str] = []
    lazy = _timed(
        "load_first_master_lazy_mps",
        timing,
        lambda: load_macbook_datasets([masters[0]], det_bin=args.det_bin, scan_size=args.scan_size, verbose=True),
    )
    widget = _timed(
        "build_show4dstem_viewer",
        timing,
        lambda: Show4DSTEM(
            lazy,
            title="Show4DSTEM heavy signoff",
            save_state=False,
            verbose=False,
            show_controls=True,
        ),
    )

    append_results: list[dict[str, Any]] = []
    for master in masters[1:]:
        label = master.name
        t0 = time.perf_counter()
        before = _memory_snapshot(f"append:{label}:before")
        try:
            indices = lazy.append_new_masters([master], async_=False)
            append_results.append(
                {
                    "master": str(master),
                    "indices": indices,
                    "seconds": round(time.perf_counter() - t0, 3),
                    "memory_before": before,
                    "memory_after": _memory_snapshot(f"append:{label}:after"),
                    "chunking_after": _describe_chunks(lazy),
                }
            )
        except Exception as exc:
            errors.append(f"append {label} failed: {exc}")
            append_results.append({"master": str(master), "error": str(exc)[:300]})

    chunking = _describe_chunks(lazy)
    export = _timed(
        f"export_html_{args.encoding}_bin{args.export_det_bin}",
        timing,
        lambda: _export_widget(widget, artifact_dir, dtype=args.encoding, det_bin=args.export_det_bin),
    )

    browser: dict[str, Any] | None = None
    if args.skip_browser:
        errors.append("browser checks skipped; this is not a full Show4DSTEM UI performance signoff")
    else:
        try:
            browser = _drive_browser_export(
                artifact_dir,
                export,
                min_fps=args.min_fps,
                timeout_ms=args.timeout_ms,
                headed=args.headed,
            )
            if not browser.get("passed"):
                errors.extend(browser.get("errors", []))
        except Exception as exc:
            errors.append(f"browser signoff failed: {exc}")
            browser = {"passed": False, "errors": [str(exc)]}

    if hasattr(widget, "close"):
        widget.close()
    if hasattr(lazy, "stop_watch"):
        lazy.stop_watch()

    report = {
        "passed": not errors,
        "local_only": True,
        "normal_ci": False,
        "artifact_dir": str(artifact_dir),
        "repo": str(root),
        "commit": os.popen("git rev-parse HEAD").read().strip(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "policy": {
            "real_data_not_committed": True,
            "normal_ci_excluded": True,
            "browser_and_backend_timings_are_separate": True,
        },
        "targets": {
            "masters": [str(master) for master in masters],
            "det_bin": args.det_bin,
            "export_det_bin": args.export_det_bin,
            "encoding": args.encoding,
            "min_fps": args.min_fps,
        },
        "discovery_notes": discovery_notes,
        "timing": timing,
        "append_results": append_results,
        "chunking": chunking,
        "exports": [export],
        "browser": browser,
        "memory_final": _memory_snapshot("final"),
        "errors": errors,
    }
    (artifact_dir / "show4dstem-heavy-signoff-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_index(artifact_dir, report)
    print(f"Show4DSTEM heavy signoff report: {artifact_dir / 'index.html'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
