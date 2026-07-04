#!/usr/bin/env python3
"""Generate standalone HTML exports for every export-capable widget."""

from __future__ import annotations

import argparse
import html
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from quantem.widget.export import supports_html_export
from quantem.widget.show2d import Show2D
from quantem.widget.show3d import Show3D
from quantem.widget.show3dslices import Show3DSlices
from quantem.widget.show4dstem import Show4DSTEM
from quantem.widget.showdiffraction import ShowDiffraction
from quantem.widget.showeds import ShowEDS
from quantem.widget.showfolder import ShowFolder


def _metadata() -> np.ndarray:
    payload = json.dumps({
        "Scan": {"ScanRotation": "0"},
        "BinaryResult": {"PixelSize": {"height": 1e-9, "width": 1e-9}},
    }).encode()
    arr = np.zeros((len(payload) + 1, 1), dtype=np.uint8)
    arr[: len(payload), 0] = np.frombuffer(payload, dtype=np.uint8)
    return arr


def _image_emd(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        group = h5.create_group("Data/Image/uid")
        group.create_dataset("Data", data=np.arange(16 * 16, dtype=np.float32).reshape(16, 16))
        group.create_dataset("Metadata", data=_metadata())


def _cases(folder_root: Path) -> list[tuple[str, object, dict[str, object], str]]:
    rng = np.random.default_rng(0)
    showfolder_dir = folder_root / "showfolder-session"
    showfolder_dir.mkdir(parents=True, exist_ok=True)
    _image_emd(showfolder_dir / "0010 - HAADF 15Mx Nano.emd")
    _image_emd(showfolder_dir / "0011 - HAADF 15Mx Nano.emd")

    return [
        (
            "show2d",
            Show2D(rng.random((3, 32, 32), dtype=np.float32), title="Smoke Show2D", verbose=False),
            {"encoding": "uint8"},
            "Smoke Show2D",
        ),
        (
            "show3d",
            Show3D(rng.random((8, 32, 32), dtype=np.float32), title="Smoke Show3D"),
            {"encoding": "uint8"},
            "Smoke Show3D",
        ),
        (
            "show3dslices",
            Show3DSlices(rng.random((8, 32, 32), dtype=np.float32), title="Smoke Show3DSlices"),
            {"encoding": "uint8"},
            "Smoke Show3DSlices",
        ),
        (
            "show4dstem",
            Show4DSTEM(rng.integers(0, 64, size=(4, 4, 8, 8), dtype=np.uint16), title="Smoke Show4DSTEM", verbose=False),
            {"encoding": "uint8", "downsample": 1},
            "Smoke Show4DSTEM",
        ),
        (
            "showeds",
            ShowEDS(rng.integers(0, 32, size=(5, 6, 12), dtype=np.uint16), title="Smoke ShowEDS", band=(2, 8), roi=(1, 1, 3, 3)),
            {"mode": "single", "encoding": "full"},
            "Smoke ShowEDS",
        ),
        (
            "showdiffraction",
            ShowDiffraction(rng.random((48, 48), dtype=np.float32), title="Smoke ShowDiffraction", verbose=False),
            {"encoding": "full"},
            "Smoke ShowDiffraction",
        ),
        (
            "showfolder",
            ShowFolder(showfolder_dir, thumb=8, group_by="none", cache_dir=folder_root / "cache"),
            {},
            "0010",
        ),
    ]


def _write_browser_plan(artifact_dir: Path, report: dict[str, Any]) -> None:
    pages = [
        {
            "widget": item["widget"],
            "file": Path(str(item["path"])).name,
            "url_path": Path(str(item["path"])).name,
            "required_interactions": [
                "open the page and confirm the widget renders",
                "click or drag the primary image/canvas where available",
                "toggle FFT, profile, ROI, or related toolbar controls where available",
                "open Export and confirm the downloaded HTML path still works",
            ],
        }
        for item in report["exports"]
    ]
    plan = {
        "version": 1,
        "description": "Open these small standalone exports in the in-app browser for visual HTML export signoff.",
        "artifact_dir": str(artifact_dir),
        "pages": pages,
    }
    (artifact_dir / "browser-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")


def _write_html_report(artifact_dir: Path, report: dict[str, Any]) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item['widget']))}</td>"
        f"<td>{html.escape(str(item['seconds']))}</td>"
        f"<td>{html.escape(f'{float(item["size_mb"]):.3f}')}</td>"
        f"<td><a href='{html.escape(Path(str(item['path'])).name)}'>"
        f"{html.escape(Path(str(item['path'])).name)}</a></td>"
        "</tr>"
        for item in report["exports"]
    )
    report_json = html.escape(json.dumps(report, indent=2))
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>quantem.widget HTML export smoke</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; line-height: 1.45; }}
    table {{ border-collapse: collapse; margin-top: 12px; min-width: 640px; }}
    th, td {{ border: 1px solid #ccd3db; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f5f7; }}
    code, pre {{ background: #f5f7f9; border-radius: 4px; }}
    code {{ padding: 2px 4px; }}
    pre {{ overflow: auto; padding: 12px; max-width: 1100px; }}
  </style>
</head>
<body>
  <h1>quantem.widget HTML export smoke</h1>
  <p>This report is generated by <code>scripts/widget_html_smoke.py</code>. Open
  each linked export in the in-app browser and follow <code>browser-plan.json</code>
  when a visual signoff is needed.</p>
  <p>Total export size: <strong>{html.escape(f'{float(report["total_size_mb"]):.3f} MB')}</strong></p>
  <table>
    <thead><tr><th>Widget</th><th>Export seconds</th><th>Size MB</th><th>Standalone HTML</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Machine-readable report</h2>
  <p><a href="report.json">report.json</a> · <a href="browser-plan.json">browser-plan.json</a></p>
  <pre>{report_json}</pre>
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--max-total-mb", type=float, default=25.0)
    args = parser.parse_args()

    if args.artifact_dir is None:
        artifact_dir = Path(tempfile.mkdtemp(prefix="quantem-widget-html-smoke-"))
    else:
        artifact_dir = args.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, object]] = []
    total_size = 0
    for name, widget, kwargs, marker in _cases(artifact_dir):
        if not supports_html_export(widget):
            raise RuntimeError(f"{name} does not satisfy supports_html_export")
        start = time.perf_counter()
        out = widget.export_html(artifact_dir / f"{name}.html", title=f"Smoke {name}", **kwargs)
        elapsed = time.perf_counter() - start
        text = out.read_text(encoding="utf-8")
        size = out.stat().st_size
        total_size += size
        required = [
            "application/vnd.jupyter.widget-state+json",
            "quantem-widget-export-layout",
            str(marker),
        ]
        missing = [item for item in required if item not in text]
        if missing:
            raise RuntimeError(f"{name} export missing markers: {missing}")
        report_rows.append({
            "widget": name,
            "path": str(out),
            "seconds": round(elapsed, 3),
            "size_mb": round(size / 1024 / 1024, 3),
        })

    max_total = args.max_total_mb * 1024 * 1024
    report = {
        "artifact_dir": str(artifact_dir),
        "total_size_mb": round(total_size / 1024 / 1024, 3),
        "exports": report_rows,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_browser_plan(artifact_dir, report)
    _write_html_report(artifact_dir, report)
    print(json.dumps(report, indent=2))
    print(f"HTML smoke report: {artifact_dir / 'index.html'}")
    if total_size > max_total:
        raise RuntimeError(
            f"HTML smoke exports total {total_size / 1024 / 1024:.2f} MB "
            f"> {args.max_total_mb:.2f} MB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
