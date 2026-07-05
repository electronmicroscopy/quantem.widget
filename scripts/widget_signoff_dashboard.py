#!/usr/bin/env python3
"""Write a single signoff dashboard for local and CI automation artifacts."""

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


def _artifact_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    html_smoke = root / "html-smoke" / "index.html"
    html_report = _read_json(root / "html-smoke" / "report.json")
    if html_smoke.exists():
        exports = len((html_report or {}).get("exports", []))
        rows.append({
            "gate": "HTML export smoke",
            "status": "pass",
            "evidence": _link("html-smoke/index.html", "HTML matrix"),
            "notes": f"{exports} exported widget page(s)" if exports else "Standalone export report",
        })
    else:
        rows.append({
            "gate": "HTML export smoke",
            "status": "missing",
            "evidence": "",
            "notes": "No HTML smoke report found",
        })

    browser_html = root / "html-smoke" / "browser-smoke.html"
    browser_report = _read_json(root / "html-smoke" / "browser-smoke-report.json")
    if browser_html.exists():
        pages = (browser_report or {}).get("pages", [])
        passed = (browser_report or {}).get("passed", 0)
        total = len(pages)
        status = "pass" if total and passed == total else "fail"
        rows.append({
            "gate": "Browser HTML smoke",
            "status": status,
            "evidence": _link("html-smoke/browser-smoke.html", "Browser report"),
            "notes": f"{passed}/{total} pages passed",
        })
    else:
        rows.append({
            "gate": "Browser HTML smoke",
            "status": "skipped",
            "evidence": "",
            "notes": "Run with --browser for exported HTML interaction/FPS proof",
        })

    perf_html = root / "performance" / "index.html"
    perf_report = _read_json(root / "performance" / "report.json")
    if perf_html.exists():
        exports = len((perf_report or {}).get("exports", []))
        rows.append({
            "gate": "Real-data Show2D/Show3D performance smoke",
            "status": "pass",
            "evidence": _link("performance/index.html", "Performance report"),
            "notes": f"{exports} real-data export(s)",
        })
    else:
        rows.append({
            "gate": "Real-data Show2D/Show3D performance smoke",
            "status": "skipped",
            "evidence": "",
            "notes": "Run with --performance for backend/export timing",
        })

    for report_name, label in [
        ("heavy-signoff-report.json", "Heavy Show2D/Show3D signoff"),
        ("show4dstem-heavy-signoff-report.json", "Heavy Show4DSTEM signoff"),
    ]:
        report = _read_json(root / report_name)
        if report is None:
            rows.append({
                "gate": label,
                "status": "skipped",
                "evidence": "",
                "notes": "Local-only real-data gate",
            })
            continue
        status = "pass" if report.get("passed") else "fail"
        rows.append({
            "gate": label,
            "status": status,
            "evidence": _link(report_name, report_name),
            "notes": "Local-only heavy real-data evidence",
        })

    phone_log = root / "phone-events.ndjson"
    phone_probe = root / "phone-probe.html"
    if phone_log.exists() or phone_probe.exists():
        rows.append({
            "gate": "Physical phone handoff",
            "status": "info",
            "evidence": _link("phone-probe.html", "Phone probe") if phone_probe.exists() else html.escape(phone_log.name),
            "notes": "Physical device evidence present",
        })
    else:
        rows.append({
            "gate": "Physical phone handoff",
            "status": "skipped",
            "evidence": "",
            "notes": "Manual Tailscale/iPhone gate only",
        })

    return rows


def _write_dashboard(root: Path, manifest: dict[str, Any], rows: list[dict[str, str]]) -> None:
    run_status = str(manifest.get("status", "unknown"))
    title_status = "PASS" if run_status == "pass" else "FAIL" if run_status == "fail" else run_status.upper()
    row_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['gate'])}</td>"
        f"<td>{_status_cell(row['status'])}</td>"
        f"<td>{row['evidence']}</td>"
        f"<td>{html.escape(row['notes'])}</td>"
        "</tr>"
        for row in rows
    )
    manifest_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in manifest.items()
        if key not in {"extra"}
    )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>quantem.widget signoff dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; line-height: 1.45; }}
    h1 {{ margin-bottom: 4px; }}
    table {{ border-collapse: collapse; margin: 14px 0 28px; width: min(1180px, 100%); }}
    th, td {{ border: 1px solid #ccd3db; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f5f7; }}
    code {{ background: #f5f7f9; padding: 2px 4px; border-radius: 4px; }}
    .pill {{ color: white; display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; font-weight: 700; }}
    .summary {{ border-left: 4px solid #2563eb; background: #eff6ff; padding: 10px 14px; width: min(1120px, 100%); }}
  </style>
</head>
<body>
  <h1>quantem.widget signoff dashboard: {html.escape(title_status)}</h1>
  <p class="summary">This is the single-page evidence index for the automation run.
  Download this artifact from GitHub Actions or open it locally, then follow the
  links below to inspect the generated widget reports.</p>

  <h2>Run Metadata</h2>
  <table><tbody>{manifest_rows}</tbody></table>

  <h2>Evidence Gates</h2>
  <table>
    <thead><tr><th>Gate</th><th>Status</th><th>Evidence</th><th>Notes</th></tr></thead>
    <tbody>{row_html}</tbody>
  </table>

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
    (root / "signoff-dashboard.json").write_text(
        json.dumps({"manifest": manifest, "gates": rows}, indent=2),
        encoding="utf-8",
    )
    _write_dashboard(root, manifest, rows)
    print(f"Signoff dashboard: {root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
