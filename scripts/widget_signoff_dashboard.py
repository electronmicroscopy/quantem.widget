#!/usr/bin/env python3
"""Write the top-level widget automation dashboard.

The dashboard is intentionally an aggregator. Individual scripts own their
measurements; this file turns their JSON/HTML artifacts into one page that says
what passed, what failed, what was skipped, and where to inspect the evidence.
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"could not read {path.name}: {exc}"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _flatten_errors(value: Any) -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key in ("errors", "page_errors", "console_errors"):
            for item in _as_list(value.get(key)):
                if item:
                    errors.append(f"{key}: {item}")
        for nested_key in (
            "browser",
            "show2d_paged_scrub",
            "show3d_paged_scrub",
            "show3d_fft_idle",
            "show3d_fft_stats_toggle",
        ):
            errors.extend(_flatten_errors(value.get(nested_key)))
    elif isinstance(value, list):
        for item in value:
            errors.extend(_flatten_errors(item))
    return errors


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_seconds(value: Any) -> str:
    seconds = _safe_float(value)
    if seconds is None:
        return ""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def _format_mb(value: Any) -> str:
    mb = _safe_float(value)
    if mb is None:
        return ""
    return f"{mb:.2f} MB"


def _summarize_exports(exports: list[dict[str, Any]]) -> dict[str, str]:
    if not exports:
        return {}
    total_mb = sum(_safe_float(item.get("size_mb")) or 0.0 for item in exports)
    slowest = max(exports, key=lambda item: _safe_float(item.get("seconds")) or -1.0)
    largest = max(exports, key=lambda item: _safe_float(item.get("size_mb")) or -1.0)
    widgets = sorted({str(item.get("widget", "unknown")) for item in exports})
    return {
        "Exports": str(len(exports)),
        "Widgets": ", ".join(widgets),
        "Total HTML": _format_mb(total_mb),
        "Slowest export": f"{slowest.get('variant', slowest.get('widget', 'unknown'))} {_format_seconds(slowest.get('seconds'))}",
        "Largest export": f"{largest.get('variant', largest.get('widget', 'unknown'))} {_format_mb(largest.get('size_mb'))}",
    }


def _fps_values_from_browser_report(report: dict[str, Any]) -> list[float]:
    fps_values: list[float] = []
    for page in report.get("pages", []):
        fps = _safe_float(page.get("fps"))
        if fps is not None:
            fps_values.append(fps)
        for key in ("initial_fps", "final_fps"):
            nested = _safe_float(page.get(key))
            if nested is not None:
                fps_values.append(nested)
        for step in page.get("steps", []) if isinstance(page.get("steps"), list) else []:
            step_fps = _safe_float(step.get("fps"))
            if step_fps is not None:
                fps_values.append(step_fps)
    return fps_values


def _summarize_browser_report(report: dict[str, Any]) -> dict[str, str]:
    pages = report.get("pages", [])
    passed = int(report.get("passed", 0) or 0)
    fps_values = _fps_values_from_browser_report(report)
    screenshots = sum(1 for page in pages if page.get("screenshot"))
    metrics = {
        "Pages": f"{passed}/{len(pages)} passed",
        "Min FPS": f"{min(fps_values):.1f}" if fps_values else "",
        "Median-ish FPS": f"{sorted(fps_values)[len(fps_values) // 2]:.1f}" if fps_values else "",
        "Screenshots": str(screenshots),
    }
    if report.get("mobile"):
        metrics["Mobile precheck"] = "390x844 Chromium/touch"
    return {key: value for key, value in metrics.items() if value}


def _summarize_timing_report(report: dict[str, Any]) -> dict[str, str]:
    timings = report.get("timings", [])
    exports = report.get("exports", [])
    metrics = _summarize_exports(exports)
    if timings:
        slowest = max(timings, key=lambda item: _safe_float(item.get("seconds")) or -1.0)
        metrics["Slowest backend step"] = f"{slowest.get('case', 'unknown')} {_format_seconds(slowest.get('seconds'))}"
    return metrics


def _summarize_heavy_report(report: dict[str, Any]) -> dict[str, str]:
    metrics: dict[str, str] = {}
    browser = report.get("browser") if isinstance(report.get("browser"), dict) else {}
    fps_values = _fps_values_from_browser_report(browser)
    if fps_values:
        metrics["Browser min FPS"] = f"{min(fps_values):.1f}"
    for label, key in [
        ("Show2D page max", "show2d_paged_scrub"),
        ("Show3D page max", "show3d_paged_scrub"),
    ]:
        scrub = report.get(key)
        if isinstance(scrub, dict) and scrub.get("page_scrub_max_ms") is not None:
            metrics[label] = f"{scrub.get('page_scrub_max_ms')} ms"
    fft_idle = report.get("show3d_fft_idle")
    if isinstance(fft_idle, dict) and fft_idle.get("fps") is not None:
        metrics["FFT idle FPS"] = f"{fft_idle.get('fps')}"
    stats = report.get("show3d_fft_stats_toggle")
    if isinstance(stats, dict) and stats.get("toggles") is not None:
        metrics["Stats toggles"] = str(stats.get("toggles"))
    return metrics


def _summarize_show4dstem_heavy(report: dict[str, Any]) -> dict[str, str]:
    metrics: dict[str, str] = {}
    timing = report.get("timing") if isinstance(report.get("timing"), dict) else {}
    for key in ("first_load_seconds", "widget_build_seconds", "append_seconds"):
        if key in timing:
            metrics[key.replace("_", " ").title()] = _format_seconds(timing.get(key))
    browser = report.get("browser") if isinstance(report.get("browser"), dict) else {}
    for key in (
        "initial_fps",
        "scan_position_fps",
        "detector_drag_fps",
        "wheel_zoom_fps",
        "dataset_flip_fps",
    ):
        if browser.get(key) is not None:
            metrics[key.replace("_", " ").title()] = f"{browser.get(key)}"
    targets = report.get("targets") if isinstance(report.get("targets"), dict) else {}
    if targets.get("max_successful_masters") is not None:
        metrics["Masters loaded"] = str(targets.get("max_successful_masters"))
    return metrics


def _summarize_external_profile(metrics: dict[str, Any]) -> dict[str, str]:
    summary = {
        "URL": str(metrics.get("url", ""))[:80],
        "Ready": _format_seconds(metrics.get("load_to_ready_s")),
        "Initial FPS": f"{metrics.get('initial_fps')}" if metrics.get("initial_fps") is not None else "",
        "Final FPS": f"{metrics.get('final_fps')}" if metrics.get("final_fps") is not None else "",
        "Canvases": str(metrics.get("initial_canvas_count", "")),
        "Steps": str(len(metrics.get("steps", []))),
    }
    return {key: value for key, value in summary.items() if value}


def _summarize_gif_report(report: dict[str, Any]) -> dict[str, str]:
    exports = report.get("exports", [])
    metrics = {
        "Source": str((report.get("source") or {}).get("kind", report.get("source", ""))),
        "Input": "x".join(str(item) for item in report.get("input_shape", [])),
        "FPS": str(report.get("fps", "")),
        "Panel gap": str(report.get("panel_gap", "")),
        "Total GIF": _format_mb(report.get("total_size_mb")),
        "Exports": str(len(exports)),
    }
    if exports:
        largest = max(exports, key=lambda item: _safe_float(item.get("size_mb")) or -1.0)
        metrics["Largest GIF"] = f"{largest.get('quality', 'unknown')} {_format_mb(largest.get('size_mb'))}"
    return {key: value for key, value in metrics.items() if value}


def _status_cell(status: str) -> str:
    colors = {
        "pass": "#0f766e",
        "fail": "#b91c1c",
        "skipped": "#6b7280",
        "missing": "#92400e",
        "info": "#1d4ed8",
    }
    color = colors.get(status, "#374151")
    return f"<span class='pill' style='background:{color}'>{html.escape(status.upper())}</span>"


def _link(path: str, label: str) -> str:
    return f"<a href='{html.escape(path)}'>{html.escape(label)}</a>"


def _metrics_text(metrics: dict[str, str]) -> str:
    if not metrics:
        return ""
    return "; ".join(
        f"{key}: {value}"
        for key, value in metrics.items()
        if value not in {"", None}
    )


def _gate(
    gate: str,
    status: str,
    evidence: str = "",
    notes: str = "",
    metrics: dict[str, str] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "gate": gate,
        "status": status,
        "evidence": evidence,
        "notes": notes,
        "metrics": metrics or {},
        "errors": errors or [],
    }


def _report_gate(
    root: Path,
    *,
    gate: str,
    report_path: Path,
    html_path: Path | None = None,
    optional: bool = False,
    summary: str = "",
    metrics_fn: Any = None,
) -> dict[str, Any]:
    report = _read_json(report_path)
    if report is None:
        return _gate(
            gate,
            "skipped" if optional else "missing",
            notes=summary or ("Optional gate not run" if optional else "Expected report was not found"),
        )
    evidence_parts = []
    if html_path is not None and html_path.exists():
        evidence_parts.append(_link(html_path.relative_to(root).as_posix(), "HTML report"))
    evidence_parts.append(_link(report_path.relative_to(root).as_posix(), report_path.name))
    errors = _flatten_errors(report)
    status = "pass" if bool(report.get("passed", True)) and not errors else "fail"
    return _gate(
        gate,
        status,
        evidence=" · ".join(evidence_parts),
        notes=summary,
        metrics=metrics_fn(report) if metrics_fn else {},
        errors=errors,
    )


def _find_optional_reports(root: Path, name: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(name)
        if path.is_file() and ".git" not in path.parts
    )


def _artifact_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest = _read_json(root / "signoff-manifest.json") or {}
    if manifest:
        status = str(manifest.get("status", "info"))
        rows.append(_gate(
            "Overall local/CI signoff",
            "pass" if status == "pass" else "fail" if status == "fail" else "info",
            evidence=_link("signoff-manifest.json", "signoff-manifest.json"),
            notes=f"mode={manifest.get('mode', 'unknown')}; duration={manifest.get('duration_seconds', '?')} s",
            metrics={
                "Branch": str(manifest.get("branch", "")),
                "Commit": str(manifest.get("commit", "")),
                "Docs": str(manifest.get("docs_build", "")),
            },
        ))

    html_smoke = root / "html-smoke" / "index.html"
    html_report = _read_json(root / "html-smoke" / "report.json")
    if html_smoke.exists():
        exports = (html_report or {}).get("exports", [])
        rows.append(_gate(
            "HTML export smoke",
            "pass" if exports else "fail",
            evidence=_link("html-smoke/index.html", "HTML matrix"),
            notes="Standalone HTML export protocol and widget coverage",
            metrics=_summarize_exports(exports),
            errors=_flatten_errors(html_report),
        ))
    else:
        rows.append(_gate("HTML export smoke", "missing", notes="No HTML smoke report found"))

    browser_html = root / "html-smoke" / "browser-smoke.html"
    browser_report = _read_json(root / "html-smoke" / "browser-smoke-report.json")
    if browser_html.exists():
        pages = (browser_report or {}).get("pages", [])
        passed = (browser_report or {}).get("passed", 0)
        total = len(pages)
        status = "pass" if total and passed == total else "fail"
        rows.append(_gate(
            "Browser HTML smoke",
            status,
            evidence=_link("html-smoke/browser-smoke.html", "Browser report"),
            notes="Chromium render, controls, nonblank canvas, screenshots, FPS",
            metrics=_summarize_browser_report(browser_report or {}),
            errors=_flatten_errors(browser_report),
        ))
    else:
        rows.append(_gate(
            "Browser HTML smoke",
            "skipped",
            notes="Run with --browser for exported HTML interaction/FPS proof",
        ))

    live_html = root / "showfolder-live" / "index.html"
    live_report = _read_json(root / "showfolder-live" / "report.json")
    if live_html.exists():
        steps = (live_report or {}).get("steps", [])
        passed = bool((live_report or {}).get("passed"))
        rows.append(_gate(
            "ShowFolder live-folder smoke",
            "pass" if passed else "fail",
            evidence=_link("showfolder-live/index.html", "Live-folder report"),
            notes="Folder watcher and selected widget handoff",
            metrics={"Scenarios": str(len(steps))},
            errors=_flatten_errors(live_report),
        ))
    else:
        rows.append(_gate(
            "ShowFolder live-folder smoke",
            "missing",
            notes="Expected from local signoff; proves live Show2D/Show3D/Show4DSTEM handoff",
        ))

    perf_html = root / "performance" / "index.html"
    perf_report = _read_json(root / "performance" / "report.json")
    if perf_html.exists():
        too_large = (perf_report or {}).get("too_large", [])
        rows.append(_gate(
            "Real-data Show2D/Show3D performance smoke",
            "fail" if too_large else "pass",
            evidence=_link("performance/index.html", "Performance report"),
            notes="Backend load/construct/export timing and real-data payload sizes",
            metrics=_summarize_timing_report(perf_report or {}),
            errors=[f"too_large: {item}" for item in too_large] + _flatten_errors(perf_report),
        ))
    else:
        rows.append(_gate(
            "Real-data Show2D/Show3D performance smoke",
            "skipped",
            notes="Run with --performance for backend/export timing",
        ))

    perf_browser_html = root / "performance" / "browser-smoke.html"
    perf_browser_report = _read_json(root / "performance" / "browser-smoke-report.json")
    if perf_browser_html.exists():
        pages = (perf_browser_report or {}).get("pages", [])
        passed = (perf_browser_report or {}).get("passed", 0)
        total = len(pages)
        rows.append(_gate(
            "Real-data Show2D/Show3D browser smoke",
            "pass" if total and passed == total else "fail",
            evidence=_link("performance/browser-smoke.html", "Performance browser report"),
            notes="Browser render, interaction, screenshot, and FPS checks on real-data exports",
            metrics=_summarize_browser_report(perf_browser_report or {}),
            errors=_flatten_errors(perf_browser_report),
        ))
    else:
        rows.append(_gate(
            "Real-data Show2D/Show3D browser smoke",
            "skipped",
            notes="Run --browser together with --performance to drive real-data exports in Chromium",
        ))

    rows.append(_report_gate(
        root,
        gate="Heavy Show2D/Show3D signoff",
        report_path=root / "heavy-signoff-report.json",
        html_path=None,
        optional=True,
        summary="Local-only real-data FPS, page scrub, cache, FFT metric evidence",
        metrics_fn=_summarize_heavy_report,
    ))
    rows.append(_report_gate(
        root,
        gate="Heavy Show4DSTEM signoff",
        report_path=root / "show4dstem-heavy-signoff-report.json",
        html_path=None,
        optional=True,
        summary="Local-only CUDA/MPS/backend/browser interaction evidence",
        metrics_fn=_summarize_show4dstem_heavy,
    ))

    tutorial_report = root / "tutorial-interactivity-report.json"
    rows.append(_report_gate(
        root,
        gate="Rendered tutorial interactivity smoke",
        report_path=tutorial_report,
        html_path=None,
        optional=True,
        summary="Executed tutorial HTML still responds to browser interaction",
        metrics_fn=lambda report: {
            "Canvas count": str(report.get("canvas_count", "")),
            "Before": str(report.get("before", "")),
            "After": str(report.get("after", "")),
        },
    ))

    external_profiles = [
        path for path in _find_optional_reports(root, "metrics.json")
        if path.name == "metrics.json" and "external" in path.as_posix()
    ]
    if external_profiles:
        passed = 0
        errors: list[str] = []
        metrics: dict[str, str] = {"Profiles": str(len(external_profiles))}
        evidence = []
        for index, path in enumerate(external_profiles, start=1):
            report = _read_json(path) or {}
            if report.get("passed"):
                passed += 1
            errors.extend(_flatten_errors(report))
            metrics.update({f"Profile {index} {key}": value for key, value in _summarize_external_profile(report).items()})
            report_html = path.with_name("index.html")
            if report_html.exists():
                evidence.append(_link(report_html.relative_to(root).as_posix(), f"profile {index}"))
        rows.append(_gate(
            "External exported HTML profile(s)",
            "pass" if passed == len(external_profiles) and not errors else "fail",
            evidence=" · ".join(evidence),
            notes="Tailscale/hosted standalone exports opened and driven in Chromium",
            metrics=metrics,
            errors=errors,
        ))
    else:
        rows.append(_gate(
            "External exported HTML profile(s)",
            "skipped",
            notes="Run widget_external_html_profile.py for an already-existing report URL",
        ))

    gif_reports = [
        path for path in _find_optional_reports(root, "report.json")
        if path.parent.name not in {"html-smoke", "performance", "showfolder-live"}
        and isinstance(_read_json(path), dict)
        and "planned_exports" in (_read_json(path) or {})
        and "playback" in (_read_json(path) or {})
    ]
    if gif_reports:
        errors: list[str] = []
        metrics: dict[str, str] = {"Reports": str(len(gif_reports))}
        evidence = []
        passed = 0
        for index, path in enumerate(gif_reports, start=1):
            report = _read_json(path) or {}
            # GIF dry-runs are valid planning evidence; non-dry runs fail only
            # when the generator exits before writing a report.
            passed += 1
            errors.extend(_flatten_errors(report))
            metrics.update({f"GIF {index} {key}": value for key, value in _summarize_gif_report(report).items()})
            report_html = path.with_name("index.html")
            if report_html.exists():
                evidence.append(_link(report_html.relative_to(root).as_posix(), f"GIF report {index}"))
        rows.append(_gate(
            "Show3D GIF/MP4 presentation smoke",
            "pass" if passed == len(gif_reports) and not errors else "fail",
            evidence=" · ".join(evidence),
            notes="Presentation animation quality, size, dry-run, and frame-change evidence",
            metrics=metrics,
            errors=errors,
        ))
    else:
        rows.append(_gate(
            "Show3D GIF/MP4 presentation smoke",
            "skipped",
            notes="Run widget_show3d_animation_smoke.py when animation export changes",
        ))

    phone_log = root / "phone-events.ndjson"
    phone_probe = root / "phone-probe.html"
    if phone_log.exists() or phone_probe.exists():
        rows.append(_gate(
            "Physical phone handoff",
            "info",
            evidence=_link("phone-probe.html", "Phone probe") if phone_probe.exists() else html.escape(phone_log.name),
            notes="Physical device evidence present",
            metrics={"Events log": "present" if phone_log.exists() else "not present"},
        ))
    else:
        rows.append(_gate(
            "Physical phone handoff",
            "skipped",
            notes="Manual Tailscale/iPhone gate only",
        ))

    return rows


def _next_actions(rows: list[dict[str, Any]]) -> list[str]:
    statuses = {row["gate"]: row["status"] for row in rows}
    actions: list[str] = []
    if statuses.get("HTML export smoke") in {"missing", "fail"}:
        actions.append("Run `scripts/widget_local_signoff.sh --quick` before claiming normal readiness.")
    if statuses.get("Browser HTML smoke") == "skipped":
        actions.append("Run `scripts/widget_local_signoff.sh --quick --browser` for exported HTML UI/FPS evidence.")
    if statuses.get("Real-data Show2D/Show3D performance smoke") == "skipped":
        actions.append("Run `scripts/widget_local_signoff.sh --full --performance` when load/export timing matters.")
    if statuses.get("Real-data Show2D/Show3D browser smoke") == "skipped":
        actions.append("Run `scripts/widget_local_signoff.sh --full --browser --performance` when real-data exported HTML must also be driven in Chromium.")
    if statuses.get("Heavy Show2D/Show3D signoff") == "skipped":
        actions.append("Run `PYTHONPATH=src:. python scripts/widget_heavy_perf_signoff.py` for local-only heavy Show2D/Show3D claims.")
    if statuses.get("Heavy Show4DSTEM signoff") == "skipped":
        actions.append("Run `PYTHONPATH=src:. python scripts/widget_show4dstem_heavy_signoff.py --backend cuda` for local-only heavy Show4DSTEM claims.")
    if statuses.get("Physical phone handoff") == "skipped":
        actions.append("Use `scripts/widget_phone_handoff.py` only when the claim depends on physical iPhone/iPad Safari behavior.")
    if any(row["status"] == "fail" for row in rows):
        actions.insert(0, "Fix failed gates first; do not replace browser or heavy-data failures with weaker unit-test evidence.")
    return actions


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def _performance_rows(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    perf_rows: list[tuple[str, str, str]] = []
    interesting = {"Min FPS", "Browser min FPS", "Initial FPS", "Final FPS", "Ready", "Slowest backend step", "Slowest export", "Largest export", "Show2D page max", "Show3D page max", "FFT idle FPS"}
    for row in rows:
        for key, value in row.get("metrics", {}).items():
            if key in interesting or key.endswith("FPS") or "page max" in key.lower():
                perf_rows.append((row["gate"], key, str(value)))
    return perf_rows


def _metric(rows: list[dict[str, Any]], gate: str, key: str) -> str:
    for row in rows:
        if row["gate"] == gate:
            value = row.get("metrics", {}).get(key)
            return str(value) if value not in {None, ""} else ""
    return ""


def _summary_card_html(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    counts = _status_counts(rows)
    cards = [
        ("Result", str(manifest.get("status", "unknown")).upper(), "Overall local/CI signoff status."),
        (
            "Evidence Gates",
            f"{counts.get('pass', 0)} pass / {counts.get('fail', 0)} fail / {counts.get('skipped', 0)} skipped",
            "Skipped gates are stronger optional checks, not failures.",
        ),
        (
            "Browser Smoke",
            _metric(rows, "Browser HTML smoke", "Pages") or "not run",
            f"Min FPS {_metric(rows, 'Browser HTML smoke', 'Min FPS') or 'n/a'}",
        ),
        (
            "HTML Export",
            _metric(rows, "HTML export smoke", "Exports") or "not run",
            _metric(rows, "HTML export smoke", "Total HTML") or "No export size recorded.",
        ),
        (
            "Runtime",
            f"{manifest.get('duration_seconds', '?')} s",
            f"mode={manifest.get('mode', 'unknown')}; docs={manifest.get('docs_build', 'unknown')}",
        ),
    ]
    return "\n".join(
        "<section class='kpi'>"
        f"<div class='kpi-label'>{html.escape(label)}</div>"
        f"<div class='kpi-value'>{html.escape(value)}</div>"
        f"<div class='kpi-note'>{html.escape(note)}</div>"
        "</section>"
        for label, value, note in cards
    )


def _write_dashboard(root: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    run_status = str(manifest.get("status", "unknown"))
    title_status = "PASS" if run_status == "pass" else "FAIL" if run_status == "fail" else run_status.upper()
    status_counts = _status_counts(rows)
    failed_rows = [row for row in rows if row["status"] == "fail"]
    summary_bits = " · ".join(
        f"{count} {status}"
        for status, count in sorted(status_counts.items())
    )
    row_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['gate'])}</td>"
        f"<td>{_status_cell(row['status'])}</td>"
        f"<td>{row['evidence']}</td>"
        f"<td>{html.escape(row['notes'])}</td>"
        f"<td>{html.escape(_metrics_text(row.get('metrics', {})))}</td>"
        f"<td>{'<br>'.join(html.escape(item) for item in row.get('errors', [])[:8])}</td>"
        "</tr>"
        for row in rows
    )
    failure_html = "".join(
        f"<li><strong>{html.escape(row['gate'])}</strong>: "
        f"{html.escape('; '.join(row.get('errors', [])[:4]) or row['notes'])}</li>"
        for row in failed_rows
    ) or "<li>No failed evidence gates in this dashboard.</li>"
    manifest_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in manifest.items()
        if key not in {"extra"}
    )
    performance_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(gate)}</td>"
        f"<td>{html.escape(metric)}</td>"
        f"<td>{html.escape(value)}</td>"
        "</tr>"
        for gate, metric, value in _performance_rows(rows)
    ) or "<tr><td colspan='3'>No measured browser/load performance rows were found in this run.</td></tr>"
    action_html = "\n".join(
        f"<li>{html.escape(action)}</li>"
        for action in _next_actions(rows)
    ) or "<li>No follow-up command suggested by the dashboard.</li>"
    summary_cards = _summary_card_html(rows, manifest)
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>quantem.widget everything dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; line-height: 1.45; }}
    h1 {{ margin-bottom: 4px; }}
    table {{ border-collapse: collapse; margin: 14px 0 28px; width: min(1180px, 100%); }}
    th, td {{ border: 1px solid #ccd3db; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f5f7; }}
    code {{ background: #f5f7f9; padding: 2px 4px; border-radius: 4px; }}
    .pill {{ color: white; display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; font-weight: 700; }}
    .summary {{ border-left: 4px solid #2563eb; background: #eff6ff; padding: 10px 14px; width: min(1120px, 100%); }}
    .grid {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); width: min(1180px, 100%); }}
    .panel {{ border: 1px solid #d8dee6; border-radius: 8px; padding: 12px 14px; background: #fff; }}
    .kpis {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); width: min(1180px, 100%); margin: 14px 0; }}
    .kpi {{ border: 1px solid #d8dee6; border-radius: 8px; padding: 10px 12px; background: #fbfdff; min-height: 92px; }}
    .kpi-label {{ color: #5b6472; font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    .kpi-value {{ font-size: 22px; font-weight: 800; margin: 4px 0; }}
    .kpi-note {{ color: #4b5563; font-size: 13px; }}
    .table-scroll {{ overflow-x: auto; width: min(1180px, 100%); }}
    .table-scroll table {{ width: 1180px; }}
    td:nth-child(5), td:nth-child(6) {{ font-size: 13px; }}
  </style>
</head>
<body>
  <h1>quantem.widget everything dashboard: {html.escape(title_status)}</h1>
  <p class="summary">Single-page evidence index for widget automation. Summary:
  {html.escape(summary_bits)}. Open this artifact first, then drill into the
  linked reports only when a gate failed or a metric needs visual review.</p>

  <div class="kpis">
    {summary_cards}
  </div>

  <div class="grid">
    <section class="panel">
      <h2>Failures</h2>
      <ul>{failure_html}</ul>
    </section>
    <section class="panel">
      <h2>Next Recommended Runs</h2>
      <ul>{action_html}</ul>
    </section>
  </div>

  <h2>Run Metadata</h2>
  <div class="table-scroll"><table><tbody>{manifest_rows}</tbody></table></div>

  <h2>Evidence Gates</h2>
  <div class="table-scroll"><table>
    <thead><tr><th>Gate</th><th>Status</th><th>Evidence</th><th>Notes</th><th>Metrics</th><th>Errors</th></tr></thead>
    <tbody>{row_html}</tbody>
  </table></div>

  <h2>Measured Performance</h2>
  <div class="table-scroll"><table>
    <thead><tr><th>Gate</th><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{performance_html}</tbody>
  </table></div>

  <h2>Rules</h2>
  <ul>
    <li>Browser-sensitive claims need browser evidence, not Python tests alone.</li>
    <li>Real-data performance claims need local real-data reports.</li>
    <li>Physical iPhone Safari claims need phone handoff logs or explicit human confirmation.</li>
    <li>Keep private data paths, screenshots, heavy HTML, and generated notebook outputs out of git.</li>
  </ul>
</body>
</html>
"""
    (root / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    root = args.artifact_dir.resolve()
    manifest_path = args.manifest or root / "signoff-manifest.json"
    manifest = _read_json(manifest_path) or {}
    manifest.setdefault("created_at_unix", int(time.time()))
    manifest.setdefault("artifact_dir", str(root))
    rows = _artifact_rows(root)
    summary = {
        "status_counts": {status: sum(1 for row in rows if row["status"] == status) for status in sorted({row["status"] for row in rows})},
        "failed_gates": [row["gate"] for row in rows if row["status"] == "fail"],
        "next_actions": _next_actions(rows),
    }
    (root / "signoff-dashboard.json").write_text(
        json.dumps({"manifest": manifest, "summary": summary, "gates": rows}, indent=2),
        encoding="utf-8",
    )
    _write_dashboard(root, manifest, rows)
    print(f"Signoff dashboard: {root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
