#!/usr/bin/env python3
"""Run the local-only real-data Show4DSTEM heavy performance signoff.

This gate is intentionally not normal CI. It uses local 4D-STEM master files,
measures the selected backend path (CUDA/NVIDIA by default, MPS only when
requested), exports a standalone HTML viewer, then drives that exported viewer
in Chromium. Generated reports, screenshots, and private lab paths stay under
``/tmp`` unless a maintainer explicitly asks for them.
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


def _seven_tilt_dir_arg(value: str | None) -> Path | None:
    raw = (
        value
        or os.environ.get("QUANTEM_WIDGET_SHOW4DSTEM_FIXTURE_DIR", "")
        or os.environ.get("QUANTEM_WIDGET_SHOW4DSTEM_7TILT_DIR", "")
    )
    return Path(raw).expanduser() if raw.strip() else None


def _load_dtype_token(value: str) -> str | None:
    token = str(value or "auto").strip().lower()
    if token in {"", "auto", "native", "full", "exact"}:
        return None
    if token in {"u8", "uint8"}:
        return "u8"
    if token in {"u16", "uint16"}:
        return "u16"
    raise ValueError(f"unsupported --load-dtype {value!r}; use auto, u8, or u16")


def _anonymous_master_labels(masters: list[Path]) -> dict[str, str]:
    return {str(path): f"tilt_{idx:02d}" for idx, path in enumerate(masters)}


def _master_label_for_report(master: Path, labels: dict[str, str]) -> str:
    return labels.get(str(master), master.name)


def _private_h5_links(
    artifact_dir: Path,
    masters: list[Path],
    labels: dict[str, str],
) -> list[str]:
    """Create anonymous H5-family symlinks for browser tests without copying data."""
    h5_dir = artifact_dir / "h5"
    h5_dir.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    for master in masters:
        label = _master_label_for_report(master, labels)
        link = h5_dir / f"{label}_master.h5"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(master)
        source_prefix = master.name[: -len("_master.h5")]
        for data_file in sorted(master.parent.glob(f"{source_prefix}_data_*.h5")):
            data_link = h5_dir / data_file.name.replace(source_prefix, label, 1)
            if data_link.exists() or data_link.is_symlink():
                data_link.unlink()
            data_link.symlink_to(data_file.resolve())
        urls.append(f"h5/{link.name}")
    return urls


def _scrub_private_report(value: Any, labels: dict[str, str]) -> Any:
    """Replace private master paths and file names with anonymous tilt labels."""
    if not labels:
        return value
    replacements: dict[str, str] = {}
    for raw, label in labels.items():
        replacements[raw] = label
        replacements[Path(raw).name] = label
        stem = Path(raw).name
        if stem.endswith("_master.h5"):
            replacements[stem[: -len("_master.h5")]] = label
    if isinstance(value, str):
        out = value
        for raw, label in replacements.items():
            out = out.replace(raw, label)
        return out
    if isinstance(value, list):
        return [_scrub_private_report(item, labels) for item in value]
    if isinstance(value, tuple):
        return [_scrub_private_report(item, labels) for item in value]
    if isinstance(value, dict):
        return {
            _scrub_private_report(key, labels): _scrub_private_report(item, labels)
            for key, item in value.items()
        }
    return value


def _discover_real_masters(
    roots: list[Path],
    *,
    pattern: str,
    scan_size: int | None,
    limit: int,
    ready_only: bool,
) -> tuple[list[Path], list[str]]:
    from quantem.gpu.io import discover, inspect

    notes: list[str] = []
    masters: list[Path] = []
    seen: set[str] = set()
    scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
    for root in roots:
        if len(masters) >= limit:
            break
        try:
            found = discover(
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
                    if not inspect(path).ready:
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
    multi = live
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


def _loadresult_payload(data: Any) -> Any:
    return data.data if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()) else data


def _describe_backend_data(data: Any, *, backend: str) -> dict[str, Any]:
    payload = _loadresult_payload(data)
    if backend == "mps" and hasattr(data, "multi"):
        return _describe_chunks(data)
    if isinstance(payload, dict):
        shards = []
        total = 0
        for device, shard in payload.items():
            nbytes = int(getattr(shard, "nbytes", 0) or 0)
            total += nbytes
            shards.append(
                {
                    "device": str(device),
                    "shape": list(getattr(shard, "shape", ())),
                    "dtype": str(getattr(shard, "dtype", "")),
                    "nbytes_mb": round(nbytes / 1024**2, 1),
                }
            )
        return {
            "type": type(payload).__name__,
            "backend": backend,
            "shape": "sharded",
            "resident_mb": round(total / 1024**2, 1),
            "shards": shards,
        }
    nbytes = int(getattr(payload, "nbytes", 0) or 0)
    return {
        "type": type(payload).__name__,
        "backend": backend,
        "shape": list(getattr(payload, "shape", ())),
        "dtype": str(getattr(payload, "dtype", "")),
        "device": str(getattr(payload, "device", "")),
        "resident_mb": round(nbytes / 1024**2, 1),
        "chunk_count": len(getattr(payload, "chunks", []) or []),
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


def _timed_maybe(label: str, records: list[dict[str, Any]], func):
    t0 = time.perf_counter()
    before = _memory_snapshot(f"{label}:before")
    try:
        result = func()
    except Exception as exc:
        records.append(
            {
                "label": label,
                "seconds": round(time.perf_counter() - t0, 3),
                "memory_before": before,
                "memory_after": _memory_snapshot(f"{label}:after_error"),
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
        )
        return None, exc
    after = _memory_snapshot(f"{label}:after")
    records.append(
        {
            "label": label,
            "seconds": round(time.perf_counter() - t0, 3),
            "memory_before": before,
            "memory_after": after,
        }
    )
    return result, None


def _cleanup_backend_memory(label: str, records: list[dict[str, Any]]) -> None:
    before = _memory_snapshot(f"{label}:before")
    t0 = time.perf_counter()
    try:
        from quantem.widget import free_gpu

        released_gb = float(free_gpu(verbose=True))
        error = ""
    except Exception as exc:
        released_gb = 0.0
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    records.append(
        {
            "label": label,
            "seconds": round(time.perf_counter() - t0, 3),
            "released_gb": round(released_gb, 3),
            "memory_before": before,
            "memory_after": _memory_snapshot(f"{label}:after"),
            "error": error,
        }
    )


def _export_widget(widget: Any, artifact_dir: Path, *, dtype: str, det_bin: int) -> dict[str, Any]:
    path = artifact_dir / f"show4dstem-real-{dtype}-bin{det_bin}.html"
    t0 = time.perf_counter()
    widget.export_html(path, encoding=dtype, det_bin=det_bin, title="Show4DSTEM heavy signoff")
    seconds = time.perf_counter() - t0
    return {
        "widget": "show4dstem",
        "variant": f"show4dstem-real-{dtype}-bin{det_bin}",
        "encoding": dtype,
        "det_bin": det_bin,
        "path": str(path),
        "seconds": round(seconds, 3),
        "size_mb": round(path.stat().st_size / 1024**2, 2),
        "n_frames": int(getattr(widget, "n_frames", 1) or 1),
        "frame_dim_label": str(getattr(widget, "frame_dim_label", "Frame") or "Frame"),
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


def _dataset_slider_box(page) -> dict[str, float] | None:
    return page.evaluate(
        """() => {
          const visible = (el) => {
            const rect = el.getBoundingClientRect();
            const style = getComputedStyle(el);
            return rect.width > 50 && rect.height > 6 && style.visibility !== 'hidden' && style.display !== 'none';
          };
          const candidates = [];
          for (const root of [...document.querySelectorAll('.MuiSlider-root')]) {
            if (!visible(root)) continue;
            const thumbs = [...root.querySelectorAll('.MuiSlider-thumb')];
            if (thumbs.length !== 1) continue;
            const inputs = [...root.querySelectorAll('input')];
            const maxVals = inputs
              .map(input => Number(input.getAttribute('aria-valuemax') || input.max || '0'))
              .filter(Number.isFinite);
            const max = maxVals.length ? Math.max(...maxVals) : 0;
            if (max < 1) continue;
            const host = root.closest('.MuiBox-root') || root.parentElement;
            const text = (host?.innerText || '').toLowerCase();
            const score =
              (text.includes('dataset') ? 5 : 0) +
              (text.includes('frame') ? 3 : 0) +
              (text.includes('tilt') ? 3 : 0) +
              (text.includes('time') ? 3 : 0) +
              (text.includes('/') ? 1 : 0);
            const rect = root.getBoundingClientRect();
            candidates.push({x: rect.x, y: rect.y, width: rect.width, height: rect.height, max, score});
          }
          if (!candidates.length) return null;
          candidates.sort((a, b) => (b.score - a.score) || (b.y - a.y) || (b.max - a.max));
          return candidates[0];
        }"""
    )


def _drag_dataset_slider(page) -> dict[str, Any]:
    box = _dataset_slider_box(page)
    if not box:
        return {"found": False, "drag_ms": 0.0}
    y = box["y"] + box["height"] / 2
    t0 = time.perf_counter()
    page.mouse.move(box["x"] + box["width"] * 0.1, y)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.9, y, steps=14)
    page.mouse.up()
    page.wait_for_timeout(180)
    return {
        "found": True,
        "drag_ms": round((time.perf_counter() - t0) * 1000, 1),
        "slider_max": int(box.get("max") or 0),
    }


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
        "n_frames": int(export.get("n_frames", 1) or 1),
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
                bad_responses: list[dict[str, Any]] = []
                page.on("pageerror", lambda exc: page_errors.append(str(exc)))
                page.on(
                    "response",
                    lambda response: bad_responses.append(
                        {"status": int(response.status), "url": response.url}
                    )
                    if response.status >= 400 and not response.url.endswith("/favicon.ico")
                    else None,
                )

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
                    if int(export.get("n_frames", 1) or 1) > 1:
                        flip = _drag_dataset_slider(page)
                        results["dataset_flip"] = flip
                        if flip.get("found"):
                            results["dataset_flip_fps"] = round(float(_measure_fps(page, 1200)), 1)
                        else:
                            results["errors"].append("dataset/frame slider not found for multi-frame export")
                    else:
                        results["dataset_flip"] = {"found": False, "skipped": "single frame export"}
                page.screenshot(path=str(screenshot), full_page=True, timeout=timeout_ms)
                results["screenshot"] = screenshot.name
                results["console_errors"] = console_errors
                results["console_warnings"] = console_warnings
                results["page_errors"] = page_errors
                results["bad_responses"] = bad_responses[:100]
                results["errors"].extend(page_errors)
                results["errors"].extend(console_errors)
                results["errors"].extend(
                    f"HTTP {item['status']} {item['url']}" for item in bad_responses[:20]
                )
                body_text = page.locator("body").inner_text(timeout=timeout_ms).lower()
                if "show4dstem load failed" in body_text or "load failed" in body_text:
                    results["errors"].append("Show4DSTEM load failed text is visible in browser")
            finally:
                browser.close()

    for key in ["initial_fps", "scan_position_fps", "detector_drag_fps", "wheel_zoom_fps", "dataset_flip_fps"]:
        if key not in results:
            continue
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
    targets = report.get("targets", {})
    target_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in [
            ("backend", targets.get("backend", "")),
            ("devices", targets.get("devices", "")),
            ("requested_master_count", targets.get("requested_master_count", "")),
            ("max_successful_masters", targets.get("max_successful_masters", "")),
            ("det_bin", targets.get("det_bin", "")),
            ("export_det_bin", targets.get("export_det_bin", "")),
            ("min_fps", targets.get("min_fps", "")),
        ]
    )
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
  <h2>Targets</h2>
  <table><tbody>{target_rows}</tbody></table>
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
    parser.add_argument("--seven-tilt", action="store_true", help="Use the private seven-tilt local-data gate and anonymize data labels.")
    parser.add_argument("--seven-tilt-dir", default="", help="Private seven-tilt folder; alternatively set QUANTEM_WIDGET_SHOW4DSTEM_7TILT_DIR.")
    parser.add_argument("--pattern", default="*_master.h5")
    parser.add_argument("--scan-size", type=int, default=None)
    parser.add_argument("--max-masters", type=int, default=None)
    parser.add_argument("--backend", choices=["cuda", "mps", "webgpu", "auto"], default="cuda")
    parser.add_argument("--devices", default="", help="Comma-separated CUDA device IDs for sharded multi-GPU load, e.g. 0,1.")
    parser.add_argument("--det-bin", type=int, default=1)
    parser.add_argument("--load-dtype", choices=["auto", "u8", "uint8", "u16", "uint16"], default="auto")
    parser.add_argument("--export-det-bin", type=int, default=1)
    parser.add_argument("--encoding", choices=["uint8", "uint16"], default="uint8")
    parser.add_argument("--min-fps", type=float, default=30.0)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--skip-export", action="store_true", help="Measure backend/widget construction only; do not write an HTML export.")
    parser.add_argument("--skip-browser", action="store_true", help="Measure backend/export only; do not claim UI performance signoff.")
    parser.add_argument("--allow-unready", action="store_true", help="Include discovered masters even if readiness checks fail.")
    parser.add_argument("--quick", action="store_true", help="Use one master and browser-suitable binning for script iteration.")
    parser.add_argument("--no-free-gpu-before", action="store_true", help="Do not clear Torch/CuPy/MPS allocator caches before loading.")
    parser.add_argument("--no-free-gpu-after", action="store_true", help="Do not clear Torch/CuPy/MPS allocator caches before exiting.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    artifact_dir = (args.artifact_dir or _timestamp_dir()).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    seven_tilt_dir = _seven_tilt_dir_arg(args.seven_tilt_dir)
    private_seven_tilt = bool(args.seven_tilt or seven_tilt_dir is not None)
    max_masters = (
        1
        if args.quick
        else max(1, int(args.max_masters))
        if args.max_masters is not None
        else 7
        if private_seven_tilt
        else 2
    )
    roots = [seven_tilt_dir] if seven_tilt_dir is not None else _roots_from_args(args.search_root)
    masters, discovery_notes = _discover_real_masters(
        roots,
        pattern=args.pattern,
        scan_size=args.scan_size,
        limit=max_masters,
        ready_only=not args.allow_unready,
    )
    if not masters:
        master_labels = {}
        report = {
            "passed": False,
            "reason": "no real 4D-STEM master files found",
            "local_only": True,
            "normal_ci": False,
            "search_roots": ["private-seven-tilt-dir"] if private_seven_tilt else [str(root) for root in roots],
            "discovery_notes": ["private seven-tilt discovery found no usable masters"] if private_seven_tilt else discovery_notes,
        }
        report = _scrub_private_report(report, master_labels)
        (artifact_dir / "show4dstem-heavy-signoff-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_index(artifact_dir, report)
        print(f"No real Show4DSTEM masters found. Report: {artifact_dir / 'index.html'}")
        return 2

    from quantem.gpu.device import resolve
    from quantem.gpu.io import load
    from quantem.widget import Show4DSTEM

    timing: list[dict[str, Any]] = []
    cleanup_records: list[dict[str, Any]] = []
    errors: list[str] = []
    backend = resolve(args.backend)
    load_dtype = _load_dtype_token(args.load_dtype)
    scan_shape = (int(args.scan_size), int(args.scan_size)) if args.scan_size else None
    devices = [int(item.strip()) for item in args.devices.split(",") if item.strip()] or None
    master_labels = _anonymous_master_labels(masters) if private_seven_tilt else {}
    lazy = None
    active_data = None
    append_strategy = "none"
    if not args.no_free_gpu_before:
        _cleanup_backend_memory("free_gpu_before", cleanup_records)

    if backend == "webgpu":
        if args.det_bin != 1 or args.export_det_bin != 1:
            errors.append(
                "WebGPU H5-source signoff is full-detector only; use "
                "--det-bin 1 --export-det-bin 1."
            )
        from quantem.widget.show4dstem_factory import _master_file_contract

        contract = _master_file_contract(masters[0])
        source_scan_shape = scan_shape or contract.get("scan_shape")
        detector_shape = contract.get("detector_shape")
        if source_scan_shape is None or detector_shape is None:
            errors.append("WebGPU H5-source signoff could not infer scan/detector shape")
        h5_urls = _private_h5_links(artifact_dir, masters, master_labels)
        export = None
        browser = None
        widget = None
        if not errors:
            import numpy as np

            labels = [_master_label_for_report(master, master_labels) for master in masters]
            widget = _timed(
                "build_show4dstem_webgpu_h5_viewer",
                timing,
                lambda: Show4DSTEM(
                    np.zeros((1, 1, 1, 1), dtype=np.uint8),
                    h5_urls=h5_urls,
                    backend="webgpu",
                    scan_shape=tuple(int(value) for value in source_scan_shape),
                    detector_shape=tuple(int(value) for value in detector_shape),
                    frame_dim_label="Dataset",
                    frame_labels=labels,
                    title="Show4DSTEM heavy signoff",
                    save_state=False,
                    verbose=False,
                    show_controls=True,
                    debug=True,
                ),
            )
            export = _timed(
                f"export_html_{args.encoding}_bin{args.export_det_bin}",
                timing,
                lambda: _export_widget(
                    widget,
                    artifact_dir,
                    dtype=args.encoding,
                    det_bin=args.export_det_bin,
                ),
            )
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
            if widget is not None and hasattr(widget, "close"):
                widget.close()
        if not args.no_free_gpu_after:
            _cleanup_backend_memory("free_gpu_after", cleanup_records)
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
                "browser_required": True,
                "private_data_labels_anonymized": private_seven_tilt,
                "private_h5_sources_symlinked_not_copied": True,
                "full_detector_no_downsample": True,
            },
            "targets": {
                "masters": [_master_label_for_report(master, master_labels) for master in masters],
                "requested_master_count": len(masters),
                "max_successful_masters": len(masters) if not errors else 0,
                "backend": backend,
                "append_strategy": "webgpu_h5_source",
                "devices": devices,
                "det_bin": args.det_bin,
                "load_dtype": "source-h5",
                "export_det_bin": args.export_det_bin,
                "encoding": args.encoding,
                "min_fps": args.min_fps,
            },
            "discovery_notes": [f"private seven-tilt folder: discovered {len(masters)} master(s)"] if private_seven_tilt else discovery_notes,
            "timing": timing,
            "cleanup": cleanup_records,
            "append_results": [],
            "chunking": {
                "type": "WebGPUH5Source",
                "shape": [len(masters), *list(source_scan_shape or ()), *list(detector_shape or ())],
                "det_bin": 1,
                "resident_mb": 0.0,
                "h5_urls": h5_urls,
            },
            "exports": [] if export is None else [export],
            "browser": browser,
            "skipped": [],
            "memory_final": _memory_snapshot("final"),
            "errors": errors,
        }
        report = _scrub_private_report(report, master_labels)
        (artifact_dir / "show4dstem-heavy-signoff-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_index(artifact_dir, report)
        print(f"Show4DSTEM heavy signoff report: {artifact_dir / 'index.html'}")
        return 0 if report["passed"] else 1

    if backend == "mps":
        loaded, load_error = _timed_maybe(
            "load_first_master_lazy_mps",
            timing,
            lambda: load(
                [masters[0]],
                backend="mps",
                det_bin=args.det_bin,
                scan_shape=scan_shape,
                dtype=load_dtype or "auto",
                verbose=True,
            ),
        )
        if load_error is not None:
            errors.append(f"initial {backend} load failed: {load_error}")
        lazy = None if loaded is None else loaded.data
        active_data = loaded
        append_strategy = "mps_live_lazy_append"
    else:
        active_data, load_error = _timed_maybe(
            f"load_first_master_{backend}",
            timing,
            lambda: load(
                str(masters[0]),
                backend=backend,
                det_bin=args.det_bin,
                dtype=load_dtype or "auto",
                scan_shape=scan_shape,
                series_type="generic" if backend == "cuda" else None,
                verbose=True,
            ),
        )
        if load_error is not None:
            errors.append(f"initial {backend} load failed: {load_error}")
        append_strategy = "cuda_eager_stack_reload" if backend == "cuda" else "eager_stack_reload"

    if active_data is None:
        if not args.no_free_gpu_after:
            _cleanup_backend_memory("free_gpu_after_error", cleanup_records)
        report = {
            "passed": False,
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
                "private_data_labels_anonymized": private_seven_tilt,
            },
            "targets": {
                "masters": [_master_label_for_report(master, master_labels) for master in masters],
                "backend": backend,
                "append_strategy": append_strategy,
                "devices": devices,
                "det_bin": args.det_bin,
                "load_dtype": args.load_dtype,
                "export_det_bin": args.export_det_bin,
                "encoding": args.encoding,
                "min_fps": args.min_fps,
            },
            "discovery_notes": [f"private seven-tilt folder: discovered {len(masters)} master(s)"] if private_seven_tilt else discovery_notes,
            "timing": timing,
            "cleanup": cleanup_records,
            "append_results": [],
            "chunking": {},
            "exports": [],
            "browser": None,
            "memory_final": _memory_snapshot("final"),
            "errors": errors,
        }
        (artifact_dir / "show4dstem-heavy-signoff-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_index(artifact_dir, report)
        print(f"Show4DSTEM heavy signoff report: {artifact_dir / 'index.html'}")
        return 1

    widget = _timed(
        "build_show4dstem_viewer",
        timing,
        lambda: Show4DSTEM(
            active_data,
            title="Show4DSTEM heavy signoff",
            save_state=False,
            verbose=False,
            show_controls=True,
        ),
    )

    append_results: list[dict[str, Any]] = []
    if backend == "cuda" and len(masters) > 1 and hasattr(widget, "close"):
        widget.close()
        widget = None
    last_good_count = 1
    for idx, master in enumerate(masters[1:], start=2):
        label = _master_label_for_report(master, master_labels)
        t0 = time.perf_counter()
        before = _memory_snapshot(f"append:{label}:before")
        try:
            if backend == "mps" and lazy is not None:
                indices = lazy.poll(
                    master.parent,
                    pattern=master.name,
                    recursive=False,
                    ready_only=True,
                    async_=False,
                )
                chunking_after = _describe_chunks(lazy)
                active_data = lazy
                last_good_count = idx
                result = {"indices": indices}
            else:
                previous_data = active_data
                if hasattr(previous_data, "free"):
                    previous_data.free()
                    active_data = None
                active_data = load(
                    [str(path) for path in masters[:idx]],
                    backend=backend,
                    det_bin=args.det_bin,
                    dtype=load_dtype or "auto",
                    scan_shape=scan_shape,
                    series_type="generic" if backend == "cuda" else None,
                    verbose=True,
                    devices=devices,
                )
                chunking_after = _describe_backend_data(active_data, backend=backend)
                last_good_count = idx
                result = {"loaded_masters": idx}
            append_results.append(
                {
                    "master": label,
                    "strategy": append_strategy,
                    **result,
                    "seconds": round(time.perf_counter() - t0, 3),
                    "memory_before": before,
                    "memory_after": _memory_snapshot(f"append:{label}:after"),
                    "backend_after": chunking_after,
                }
            )
        except Exception as exc:
            errors.append(f"append {label} failed: {exc}")
            append_results.append({"master": label, "strategy": append_strategy, "error": str(exc)[:300]})
            try:
                import gc
                import traceback

                traceback.clear_frames(exc.__traceback__)
                gc.collect()
            except Exception:
                pass
            if backend == "cuda" and last_good_count > 0:
                _cleanup_backend_memory("free_gpu_after_append_failure", cleanup_records)
                try:
                    active_data = load(
                        [str(path) for path in masters[:last_good_count]],
                        backend=backend,
                        det_bin=args.det_bin,
                        dtype=load_dtype or "auto",
                        scan_shape=scan_shape,
                        series_type="generic",
                        verbose=True,
                        devices=devices,
                    )
                except Exception as reload_exc:
                    active_data = None
                    errors.append(f"reload last successful {last_good_count} master(s) failed: {reload_exc}")
            break

    if len(masters) > 1 and active_data is not None:
        if hasattr(widget, "close"):
            widget.close()
        rebuild_label = (
            "build_show4dstem_viewer_after_lazy_append"
            if backend == "mps"
            else "build_show4dstem_viewer_after_stack_growth"
        )
        widget = _timed(
            rebuild_label,
            timing,
            lambda: Show4DSTEM(
                active_data,
                title="Show4DSTEM heavy signoff",
                save_state=False,
                verbose=False,
                show_controls=True,
            ),
        )

    exports: list[dict[str, Any]] = []
    browser: dict[str, Any] | None = None
    skipped: list[str] = []
    if active_data is None or widget is None:
        chunking = {}
        errors.append("no active Show4DSTEM data remained after append/capacity probe; export and browser checks skipped")
    else:
        chunking = _describe_chunks(lazy) if backend == "mps" and lazy is not None else _describe_backend_data(active_data, backend=backend)
        if args.skip_export:
            skipped.append("HTML export skipped by request; backend/widget checks only")
        else:
            export = _timed(
                f"export_html_{args.encoding}_bin{args.export_det_bin}",
                timing,
                lambda: _export_widget(widget, artifact_dir, dtype=args.encoding, det_bin=args.export_det_bin),
            )
            exports.append(export)

        if args.skip_browser:
            skipped.append("browser checks skipped by request")
        elif args.skip_export:
            errors.append("browser signoff requires an HTML export; remove --skip-export or add --skip-browser")
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

    if widget is not None and hasattr(widget, "close"):
        widget.close()
    if lazy is not None:
        lazy.stop()
    if widget is not None:
        del widget
    if not args.no_free_gpu_after:
        _cleanup_backend_memory("free_gpu_after", cleanup_records)

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
            "browser_required": not args.skip_browser,
            "private_data_labels_anonymized": private_seven_tilt,
        },
        "targets": {
            "masters": [_master_label_for_report(master, master_labels) for master in masters],
            "requested_master_count": len(masters),
            "max_successful_masters": last_good_count,
            "backend": backend,
            "append_strategy": append_strategy,
            "devices": devices,
            "det_bin": args.det_bin,
            "load_dtype": args.load_dtype,
            "export_det_bin": args.export_det_bin,
            "encoding": args.encoding,
            "min_fps": args.min_fps,
        },
        "discovery_notes": [f"private seven-tilt folder: discovered {len(masters)} master(s)"] if private_seven_tilt else discovery_notes,
        "timing": timing,
        "cleanup": cleanup_records,
        "append_results": append_results,
        "chunking": chunking,
        "exports": exports,
        "browser": browser,
        "skipped": skipped,
        "memory_final": _memory_snapshot("final"),
        "errors": errors,
    }
    report = _scrub_private_report(report, master_labels)
    (artifact_dir / "show4dstem-heavy-signoff-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_index(artifact_dir, report)
    print(f"Show4DSTEM heavy signoff report: {artifact_dir / 'index.html'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
