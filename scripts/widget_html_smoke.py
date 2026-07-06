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


def _mos2_lattice_stack(rng: np.random.Generator, frames: int, rows: int, cols: int) -> np.ndarray:
    """Small 1H-MoS2-like HAADF phantom for human-readable smoke reports.

    The smoke must stay tiny enough for CI, but the visual report should still
    read like atomic-resolution microscopy data. A projected 1H-MoS2 HAADF view
    has a honeycomb-like pair of column sites: bright Mo columns and dimmer
    projected S2 columns. Keeping sulfur as one projected column avoids the
    generic three-dot motif that does not look like MoS2.
    """
    y, x = np.mgrid[:rows, :cols].astype(np.float32)
    spacing = max(6.5, min(rows, cols) / 8.5)
    sigma_mo = max(0.65, spacing * 0.115)
    sigma_s2 = max(0.72, spacing * 0.135)
    angle = np.deg2rad(8.0)
    a1 = spacing * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    a2 = spacing * np.array([np.cos(angle + np.pi / 3.0), np.sin(angle + np.pi / 3.0)], dtype=np.float32)
    # Approximate HAADF Z contrast: Mo is dominant, projected S2 is visible but
    # much dimmer. The exact exponent is not important for this smoke; the visual
    # contract is the strong Mo/S2 contrast and honeycomb geometry.
    basis = [
        (np.array([0.0, 0.0], dtype=np.float32), 1.00, sigma_mo),
        ((a1 + a2) / 3.0, 0.42, sigma_s2),
    ]

    stack = []
    for idx in range(frames):
        frame = np.full((rows, cols), 0.035, dtype=np.float32)
        drift = np.array([0.16 * idx, -0.11 * idx], dtype=np.float32)
        origin = np.array([cols * 0.08, rows * 0.10], dtype=np.float32) + drift
        for row_idx in range(-2, int(rows / spacing) + 4):
            for col_idx in range(-2, int(cols / spacing) + 4):
                cell = origin + col_idx * a1 + row_idx * a2
                for offset, amp, sigma in basis:
                    cx, cy = cell + offset
                    if (
                        -3 * sigma <= cx < cols + 3 * sigma
                        and -3 * sigma <= cy < rows + 3 * sigma
                    ):
                        r2 = (x - cx) ** 2 + (y - cy) ** 2
                        frame += amp * np.exp(-r2 / (2.0 * sigma**2)).astype(np.float32)
        scan_modulation = 1.0 + 0.035 * np.sin((y + 0.7 * idx) / max(rows, 1) * 2.0 * np.pi)
        thickness = 0.88 + 0.12 * np.sin((x + y * 0.35) / max(cols, 1) * 2.0 * np.pi)
        noise = rng.normal(0, 0.014, size=(rows, cols)).astype(np.float32)
        frame = frame * scan_modulation.astype(np.float32) * thickness.astype(np.float32) + noise
        frame -= float(frame.min())
        frame /= float(frame.max()) + 1e-6
        stack.append(frame.astype(np.float32))
    return np.stack(stack, axis=0)


def _cases(folder_root: Path) -> list[tuple[str, str, object, dict[str, object], str]]:
    rng = np.random.default_rng(0)
    showfolder_dir = folder_root / "showfolder-session"
    showfolder_dir.mkdir(parents=True, exist_ok=True)
    _image_emd(showfolder_dir / "0010 - HAADF 15Mx Nano.emd")
    _image_emd(showfolder_dir / "0011 - HAADF 15Mx Nano.emd")

    show2d_single = _mos2_lattice_stack(rng, 1, 160, 192)[0]
    show2d_gallery3 = _mos2_lattice_stack(rng, 3, 160, 192)
    show2d_gallery6 = _mos2_lattice_stack(rng, 6, 144, 168)
    show2d_gallery8 = _mos2_lattice_stack(rng, 8, 128, 144)
    show3d_stack = _mos2_lattice_stack(rng, 10, 160, 192)
    show3d_short = _mos2_lattice_stack(rng, 5, 144, 168)
    show3d_panel_a = _mos2_lattice_stack(rng, 8, 144, 168)
    show3d_panel_b = _mos2_lattice_stack(rng, 8, 144, 168) * 0.7
    show3d_panel_c = _mos2_lattice_stack(rng, 8, 144, 168) + 0.2
    show3d_panel_d = _mos2_lattice_stack(rng, 8, 144, 168)

    cases: list[tuple[str, str, object, dict[str, object], str]] = [
        (
            "show2d",
            "show2d-single",
            Show2D(show2d_single, title="Smoke Show2D Single", sampling=0.2, units="nm", verbose=False),
            {"encoding": "uint8"},
            "Smoke Show2D Single",
        ),
        (
            "show2d",
            "show2d-gallery-3",
            Show2D(
                show2d_gallery3,
                labels=["original", "shifted", "noisy"],
                title="Smoke Show2D Gallery 3",
                ncols=3,
                sampling=0.2,
                units="nm",
                verbose=False,
            ),
            {"encoding": "uint8"},
            "Smoke Show2D Gallery 3",
        ),
        (
            "show2d",
            "show2d-gallery-6-fft",
            Show2D(
                show2d_gallery6,
                labels=[f"panel {idx + 1}" for idx in range(6)],
                title="Smoke Show2D Gallery 6 FFT",
                ncols=3,
                show_fft=True,
                link_zoom=True,
                link_pan=True,
                link_contrast=True,
                verbose=False,
            ),
            {"encoding": "uint8"},
            "Smoke Show2D Gallery 6 FFT",
        ),
        (
            "show2d",
            "show2d-hidden-starred",
            Show2D(
                show2d_gallery6,
                labels=[f"panel {idx + 1}" for idx in range(6)],
                title="Smoke Show2D Hidden Starred",
                ncols=3,
                hidden_panels=[1, "panel 5"],
                starred=[0, "panel 3"],
                verbose=False,
            ),
            {"encoding": "uint8"},
            "Smoke Show2D Hidden Starred",
        ),
        (
            "show2d",
            "show2d-compact-no-titles",
            Show2D(
                show2d_gallery8,
                labels=[f"compact {idx + 1}" for idx in range(8)],
                title="Smoke Show2D Compact No Titles",
                ncols=4,
                show_panel_titles=False,
                show_stats=False,
                display_bin=2,
                verbose=False,
            ),
            {"encoding": "uint8"},
            "Smoke Show2D Compact No Titles",
        ),
        (
            "show3d",
            "show3d-single-stack",
            Show3D(show3d_stack, title="Smoke Show3D Single Stack", sampling=0.2, units="nm"),
            {"encoding": "uint8"},
            "Smoke Show3D Single Stack",
        ),
        (
            "show3d",
            "show3d-single-fft-bottom",
            Show3D(show3d_short, title="Smoke Show3D FFT Bottom", show_fft=True, fft_layout="bottom", fps=12),
            {"encoding": "uint8"},
            "Smoke Show3D FFT Bottom",
        ),
        (
            "show3d",
            "show3d-single-fft-overlay",
            Show3D(
                show3d_short,
                title="Smoke Show3D FFT Overlay",
                show_fft=True,
                fft_layout="overlay",
                fft_overlay_position="bottom-right",
                fps=12,
            ),
            {"encoding": "uint8"},
            "Smoke Show3D FFT Overlay",
        ),
        (
            "show3d",
            "show3d-three-panels",
            Show3D(
                show3d_panel_a,
                show3d_panel_b,
                show3d_panel_c,
                title="Smoke Show3D Three Panels",
                panel_titles=["SSB reconstruction", "Mean DP", "Probe"],
                max_cols=3,
                hideable=True,
            ),
            {"encoding": "uint8"},
            "Smoke Show3D Three Panels",
        ),
        (
            "show3d",
            "show3d-hidden-panel",
            Show3D(
                show3d_panel_a,
                show3d_panel_b,
                show3d_panel_c,
                title="Smoke Show3D Hidden Panel",
                panel_titles=["SSB reconstruction", "Mean DP", "Probe"],
                hidden_panels=["Mean DP"],
                hideable=True,
                max_cols=3,
                show_stats=False,
            ),
            {"encoding": "uint8"},
            "Smoke Show3D Hidden Panel",
        ),
        (
            "show3d",
            "show3d-four-panel-downsample",
            Show3D(
                show3d_panel_a,
                show3d_panel_b,
                show3d_panel_c,
                show3d_panel_d,
                title="Smoke Show3D Four Panel Downsample",
                panel_titles=["A", "B", "C", "D"],
                max_cols=4,
                avg_window=2,
                fps=18,
            ),
            {"encoding": "uint8", "downsample": 2},
            "Smoke Show3D Four Panel Downsample",
        ),
        (
            "show3dslices",
            "show3dslices",
            Show3DSlices(rng.random((8, 32, 32), dtype=np.float32), title="Smoke Show3DSlices"),
            {"encoding": "uint8"},
            "Smoke Show3DSlices",
        ),
        (
            "show4dstem",
            "show4dstem",
            Show4DSTEM(rng.integers(0, 64, size=(4, 4, 8, 8), dtype=np.uint16), title="Smoke Show4DSTEM", verbose=False),
            {"encoding": "uint8", "downsample": 1},
            "Smoke Show4DSTEM",
        ),
        (
            "showeds",
            "showeds",
            ShowEDS(rng.integers(0, 32, size=(5, 6, 12), dtype=np.uint16), title="Smoke ShowEDS", band=(2, 8), roi=(1, 1, 3, 3)),
            {"mode": "single", "encoding": "full"},
            "Smoke ShowEDS",
        ),
        (
            "showdiffraction",
            "showdiffraction",
            ShowDiffraction(rng.random((48, 48), dtype=np.float32), title="Smoke ShowDiffraction", verbose=False),
            {"encoding": "full"},
            "Smoke ShowDiffraction",
        ),
        (
            "showfolder",
            "showfolder",
            ShowFolder(showfolder_dir, thumb=8, group_by="none", cache_dir=folder_root / "cache"),
            {},
            "0010",
        ),
    ]
    return cases


def _write_browser_plan(artifact_dir: Path, report: dict[str, Any]) -> None:
    pages = [
        {
            "widget": item["widget"],
            "variant": item["variant"],
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
        f"<td>{html.escape(str(item['variant']))}</td>"
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
  <p><strong>Show2D</strong> and <strong>Show3D</strong> examples use a small
  synthetic MoS2-like HAADF lattice so CI stays lightweight while the visual
  checks still show microscopy-style atomic contrast and FFT peaks.</p>
  <p>Total export size: <strong>{html.escape(f'{float(report["total_size_mb"]):.3f} MB')}</strong></p>
  <table>
    <thead><tr><th>Widget</th><th>Variant</th><th>Export seconds</th><th>Size MB</th><th>Standalone HTML</th></tr></thead>
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
    for widget_name, variant, widget, kwargs, marker in _cases(artifact_dir):
        if not supports_html_export(widget):
            raise RuntimeError(f"{variant} does not satisfy supports_html_export")
        start = time.perf_counter()
        out = widget.export_html(artifact_dir / f"{variant}.html", title=f"Smoke {variant}", **kwargs)
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
            raise RuntimeError(f"{variant} export missing markers: {missing}")
        report_rows.append({
            "widget": widget_name,
            "variant": variant,
            "path": str(out),
            "seconds": round(elapsed, 3),
            "size_mb": round(size / 1024 / 1024, 3),
            "options": kwargs,
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
