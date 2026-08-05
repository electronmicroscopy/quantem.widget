#!/usr/bin/env python3
"""Generate a Show3D GIF presentation smoke report.

This report is for the slide/share workflow: a scientist wants a small animated
asset for PowerPoint, email, or a manuscript supplement, not a full interactive
HTML widget. It times ``Show3D.save_gif()`` across quality tiers, decodes the
written GIFs, and writes a visual ``index.html`` with file-size and playback
metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
import time
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageSequence

from quantem.widget import Show3D
from quantem.widget.render import gif as gif_utils


DEFAULT_TIMESERIES_DIR = Path(os.environ.get("QUANTEM_SMOKE_TIMESERIES_DIR", str(Path.home() / "data" / "reference_timeseries")))


def _crop_center(image: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim > 2:
        arr = arr.reshape((-1, *arr.shape[-2:]))[0]
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    if h < size or w < size:
        reps = (math.ceil(size / max(h, 1)), math.ceil(size / max(w, 1)))
        arr = np.tile(arr, reps)
        h, w = arr.shape
    r0 = max(0, (h - size) // 2)
    c0 = max(0, (w - size) // 2)
    return np.ascontiguousarray(arr[r0 : r0 + size, c0 : c0 + size], dtype=np.float32)


def _load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile  # noqa: PLC0415

        return np.asarray(tifffile.imread(path))
    with Image.open(path) as image:
        return np.asarray(image)


def _load_timeseries_stack(data_root: Path, *, crop_size: int) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    paths = sorted(
        path
        for path in data_root.glob("*")
        if path.suffix.lower() in {".png", ".tif", ".tiff"}
    )
    if len(paths) < 2:
        raise FileNotFoundError(
            f"{data_root} does not contain at least two PNG/TIFF frames for an animation"
        )
    frames = [_crop_center(_load_image(path), crop_size) for path in paths]
    labels = []
    for path in paths:
        match = re.search(r"Raw(\d+)", path.stem)
        labels.append(f"Raw {match.group(1)}" if match else path.stem[-12:])
    return np.stack(frames, axis=0), labels, {
        "kind": "local reference Show3D PNG/TIFF time series",
        "root": str(data_root),
        "paths": [str(path) for path in paths],
    }


def _load_tutorial_stack(*, crop_size: int, frames: int) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    from quantem.widget.data import load_tutorial_show3d  # noqa: PLC0415

    dataset = load_tutorial_show3d(
        n_frames=frames,
        stride=8,
        crop_size=crop_size,
        verbose=False,
    )
    stack = np.asarray(dataset.array, dtype=np.float32)
    labels = [f"frame {idx + 1}" for idx in range(stack.shape[0])]
    return stack, labels, {
        "kind": "public tutorial gold HAADF moving-crop stack",
        "name": getattr(dataset, "name", "Gold HAADF moving-crop stack"),
    }


def _synthetic_stack(*, crop_size: int, frames: int) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Small fallback with moving lattice-like features for CI."""
    y, x = np.mgrid[:crop_size, :crop_size].astype(np.float32)
    out: list[np.ndarray] = []
    spacing = max(9.0, crop_size / 7.0)
    sigma = max(1.4, spacing * 0.12)
    for idx in range(frames):
        image = np.full((crop_size, crop_size), 0.03, dtype=np.float32)
        drift = np.array([0.45 * idx, -0.30 * idx], dtype=np.float32)
        for row in range(-2, int(crop_size / spacing) + 4):
            for col in range(-2, int(crop_size / spacing) + 4):
                cx = col * spacing + 0.5 * row * spacing + 0.15 * crop_size + drift[0]
                cy = row * spacing * np.sqrt(3.0) / 2.0 + 0.12 * crop_size + drift[1]
                if -3 * sigma <= cx < crop_size + 3 * sigma and -3 * sigma <= cy < crop_size + 3 * sigma:
                    image += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2)).astype(np.float32)
        image += 0.05 * np.sin((x + idx) / 8.0) + 0.03 * np.cos((y - idx) / 11.0)
        image -= float(image.min())
        image /= float(image.max()) + 1e-6
        out.append(image.astype(np.float32))
    return np.stack(out, axis=0), [f"t={idx}" for idx in range(frames)], {
        "kind": "synthetic CI fallback",
    }


def _box_blur_stack(stack: np.ndarray) -> np.ndarray:
    """Return a small 3x3 smoothed companion stack without optional deps."""
    arr = np.asarray(stack, dtype=np.float32)
    padded = np.pad(arr, ((0, 0), (1, 1), (1, 1)), mode="edge")
    blurred = np.zeros_like(arr, dtype=np.float32)
    for row in range(3):
        for col in range(3):
            blurred += padded[:, row : row + arr.shape[1], col : col + arr.shape[2]]
    return blurred / 9.0


def _multi_panel_stacks(stack: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    """Build a compact multi-panel time-series for presentation export checks."""
    raw = np.asarray(stack, dtype=np.float32)
    smoothed = _box_blur_stack(raw)
    change = np.abs(raw - raw[:1])
    return [raw, smoothed, change.astype(np.float32)], ["Raw", "Smoothed", "Change"]


def _load_stack(args: argparse.Namespace) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    source = args.source
    if source in {"auto", "timeseries"}:
        root = args.data_root or DEFAULT_TIMESERIES_DIR
        try:
            return _load_timeseries_stack(root.expanduser(), crop_size=args.crop_size)
        except Exception:
            if source == "timeseries":
                raise
    if source in {"auto", "tutorial"}:
        try:
            return _load_tutorial_stack(crop_size=args.crop_size, frames=args.frames)
        except Exception:
            if source == "tutorial":
                raise
    return _synthetic_stack(crop_size=args.crop_size, frames=args.frames)


def _animation_frame_count(n_frames: int, playback: str) -> int:
    if playback == "forward" or n_frames <= 1:
        return int(n_frames)
    return int(n_frames * 2 - 2)


def _source_file_mb(source: dict[str, Any]) -> float | None:
    paths = source.get("paths")
    if not isinstance(paths, list):
        return None
    total = 0
    for path in paths:
        try:
            total += Path(path).stat().st_size
        except OSError:
            return None
    return round(total / 1024 / 1024, 3)


def _planned_exports(
    stack: np.ndarray,
    qualities: list[str],
    *,
    playback: str,
    panel_count: int,
    max_cols: int,
    panel_gap: int,
) -> list[dict[str, Any]]:
    """Estimate output geometry and in-memory RGB work before GIF encoding."""
    n_slices, panel_h, panel_w = stack.shape
    n_frames = _animation_frame_count(int(n_slices), playback)
    cols = panel_count if max_cols <= 0 else min(max(1, max_cols), panel_count)
    rows = int(math.ceil(panel_count / cols))
    planned: list[dict[str, Any]] = []
    for quality in qualities:
        if quality not in gif_utils.QUALITY_SCALE:
            raise ValueError(f"quality must be one of {list(gif_utils.QUALITY_SCALE)}, got {quality!r}.")
        scale = gif_utils.QUALITY_SCALE[quality]
        gap = max(0, int(round(float(panel_gap) * scale)))
        out_w = max(1, int(panel_w * scale))
        out_h = max(1, int(panel_h * scale))
        grid_w = cols * out_w + gap * (cols - 1)
        grid_h = rows * out_h + gap * (rows - 1)
        rgb_mb = grid_w * grid_h * n_frames * 3 / 1024 / 1024
        planned.append({
            "quality": quality,
            "width": int(grid_w),
            "height": int(grid_h),
            "n_frames": int(n_frames),
            "uncompressed_rgb_mb": round(float(rgb_mb), 3),
        })
    return planned


def _dry_run_decision(
    *,
    panel_native_mb: float,
    planned_exports: list[dict[str, Any]],
    max_native_mb: float,
    max_work_mb: float,
) -> dict[str, Any]:
    total_rgb_mb = round(sum(float(item["uncompressed_rgb_mb"]) for item in planned_exports), 3)
    warnings: list[str] = []
    if panel_native_mb > max_native_mb:
        warnings.append(
            f"Derived panel arrays are {panel_native_mb:.1f} MB, above --max-native-mb={max_native_mb:.1f}."
        )
    if total_rgb_mb > max_work_mb:
        warnings.append(
            f"Planned uncompressed RGB work is {total_rgb_mb:.1f} MB, above --max-work-mb={max_work_mb:.1f}."
        )
    if warnings:
        summary = "Reduce crop size, frame count, or quality tiers before exporting."
        reasons = warnings
        should_run = False
    else:
        summary = "Run the full GIF export."
        reasons = ["Native data and projected RGB work are within the dry-run limits."]
        should_run = True
    return {
        "should_run": should_run,
        "summary": summary,
        "reasons": reasons,
        "total_uncompressed_rgb_mb": total_rgb_mb,
    }


def _gif_metrics(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        frames = [np.asarray(frame.convert("RGB"), dtype=np.int16) for frame in ImageSequence.Iterator(image)]
        if len(frames) >= 2:
            deltas = [
                float(np.mean(np.abs(frames[idx] - frames[idx - 1])))
                for idx in range(1, len(frames))
            ]
            mean_abs_delta = round(float(np.mean(deltas)), 3)
        else:
            mean_abs_delta = 0.0
        return {
            "n_frames": int(getattr(image, "n_frames", len(frames))),
            "width": int(image.size[0]),
            "height": int(image.size[1]),
            "mean_abs_delta": mean_abs_delta,
        }


def _format_size(path: Path) -> str:
    mb = path.stat().st_size / 1024 / 1024
    if mb >= 1:
        return f"{mb:.2f} MB"
    return f"{path.stat().st_size / 1024:.1f} KB"


def _write_html(artifact_dir: Path, report: dict[str, Any]) -> None:
    export_rows = "\n".join(
        "<tr>"
        f"<td>{item['quality']}</td>"
        f"<td>{item['seconds']:.3f}</td>"
        f"<td>{item['size_mb']:.3f}</td>"
        f"<td>{item['width']} x {item['height']}</td>"
        f"<td>{item['n_frames']}</td>"
        f"<td>{item['mean_abs_delta']:.3f}</td>"
        f"<td><a href=\"{Path(item['path']).name}\">{Path(item['path']).name}</a></td>"
        "</tr>"
        for item in report["exports"]
    )
    planned_rows = "\n".join(
        "<tr>"
        f"<td>{item['quality']}</td>"
        f"<td>{item['width']} x {item['height']}</td>"
        f"<td>{item['n_frames']}</td>"
        f"<td>{item['uncompressed_rgb_mb']:.3f}</td>"
        "</tr>"
        for item in report["planned_exports"]
    )
    if report["exports"]:
        cards = "\n".join(
            "<section class=\"card\">"
            f"<h2>GIF {item['quality']}</h2>"
            f"<img src=\"{Path(item['path']).name}\" alt=\"Show3D GIF {item['quality']}\">"
            f"<p>{_format_size(Path(item['path']))} · {item['width']} x {item['height']} · "
            f"{item['n_frames']} frames · {item['seconds']:.2f} s export</p>"
            "</section>"
            for item in report["exports"]
        )
        export_table = f"""
  <table>
    <thead><tr><th>Quality</th><th>Export seconds</th><th>Size MB</th><th>Dimensions</th><th>Frames</th><th>Mean frame delta</th><th>File</th></tr></thead>
    <tbody>{export_rows}</tbody>
  </table>"""
    else:
        cards = "<section class=\"notice\">Dry run only: no GIF files were written.</section>"
        export_table = ""
    decision = report["dry_run_decision"]
    title = "Show3D GIF presentation dry run" if report["dry_run"] else "Show3D GIF presentation smoke"
    source_file_mb = report.get("source_file_mb")
    source_size_text = (
        f"{source_file_mb:.3f} MB source files, "
        if isinstance(source_file_mb, int | float)
        else ""
    )
    overlay_controls = (
        f"Panel gap: {int(report['panel_gap'])} px; "
        f"panel labels: {'on' if report['panel_labels'] else 'off'}; "
        f"frame labels: {'on' if report['frame_labels'] else 'off'}; "
        f"scale bar: {'on' if report['scale_bar']['visible'] else 'off'}; "
        f"zoom readout: {'on' if report['zoom_indicator'] else 'off'}."
    )
    report_json = json.dumps(report, indent=2)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; }}
    .cards {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; align-items: start; }}
    .card {{ border: 1px solid #d4dae2; padding: 12px; background: #fff; max-width: 1220px; }}
    .card h2 {{ font-size: 16px; margin: 0 0 8px; }}
    .card img {{ display: block; max-width: 100%; background: #111; border: 1px solid #222; }}
    .card p {{ margin: 8px 0 0; color: #526070; font-size: 13px; }}
    table {{ border-collapse: collapse; margin: 18px 0; }}
    th, td {{ border: 1px solid #ccd3db; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f5f7; }}
    .notice {{ border: 1px solid #d4dae2; padding: 12px; max-width: 1220px; background: #f8fafc; }}
    .decision {{ border-left: 4px solid #2680d9; padding: 10px 12px; background: #f3f8ff; max-width: 1100px; }}
    pre {{ background: #f5f7f9; padding: 12px; overflow: auto; max-width: 1100px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p><strong>Source:</strong> {report['source']['kind']}</p>
  <p><strong>Panels:</strong> {", ".join(report['panels'])}. This report keeps
  the presentation overlays configurable on purpose: live-style top labels,
  bottom-right scale bars, and bottom-left zoom readouts should all be visible
  when enabled.</p>
  <p><strong>Controls:</strong> {overlay_controls}</p>
  <section class="decision">
    <strong>Decision:</strong> {decision['summary']}<br>
    <strong>Data size:</strong> {source_size_text}{report['native_mb']:.3f} MB input,
    {report['panel_native_mb']:.3f} MB after derived panels.
    <strong>Projected RGB work:</strong> {decision['total_uncompressed_rgb_mb']:.3f} MB.
  </section>
  <p>GIF is the simplest animated format for PowerPoint and email. Use
  <strong>medium</strong> when size matters; use <strong>high</strong> when the
  slide needs the sharpest image. For long or high-resolution animations, MP4 is
  usually smaller and smoother, while HTML remains the right format for
  interactive sharing.</p>
  <h2>Dry-run size plan</h2>
  <table>
    <thead><tr><th>Quality</th><th>Planned dimensions</th><th>Frames</th><th>Uncompressed RGB MB</th></tr></thead>
    <tbody>{planned_rows}</tbody>
  </table>
  <div class="cards">{cards}</div>
  {export_table}
  <h2>Raw report</h2>
  <pre>{report_json}</pre>
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--source", choices=["auto", "timeseries", "tutorial", "synthetic"], default="auto")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--frames", type=int, default=12, help="Frame count for tutorial/synthetic fallback data.")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--playback", choices=["forward", "bounce"], default="bounce")
    parser.add_argument("--qualities", nargs="+", default=["low", "medium", "high"])
    parser.add_argument("--panel-gap", type=int, default=0, help="Pixel gap between panels in the exported GIF grid. Use 0 for edge-to-edge panels.")
    parser.add_argument("--sampling-nm", type=float, default=0.05, help="Display sampling used for the GIF report scale bar.")
    parser.add_argument("--panel-labels", dest="panel_labels", action="store_true", default=True, help="Burn live-style panel labels into the GIF.")
    parser.add_argument("--no-panel-labels", dest="panel_labels", action="store_false", help="Do not burn top panel labels into the GIF.")
    parser.add_argument("--frame-labels", dest="frame_labels", action="store_true", default=True, help="Burn short frame labels into the GIF.")
    parser.add_argument("--no-frame-labels", dest="frame_labels", action="store_false", help="Do not burn frame labels into the GIF.")
    parser.add_argument("--no-scale-bar", action="store_true", help="Do not burn scale bars into the GIF.")
    parser.add_argument("--no-zoom-label", action="store_true", help="Do not burn the bottom-left zoom readout into the GIF.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect source data and projected export sizes without writing GIFs.")
    parser.add_argument("--max-total-mb", type=float, default=60.0)
    parser.add_argument("--max-native-mb", type=float, default=512.0, help="Dry-run warning threshold for derived panel arrays.")
    parser.add_argument("--max-work-mb", type=float, default=1024.0, help="Dry-run warning threshold for planned uncompressed RGB work.")
    args = parser.parse_args()

    if args.crop_size < 32:
        raise ValueError("--crop-size must be >= 32")
    if args.frames < 2:
        raise ValueError("--frames must be >= 2")
    if args.panel_gap < 0:
        raise ValueError("--panel-gap must be >= 0")

    artifact_dir = args.artifact_dir or Path(tempfile.mkdtemp(prefix="quantem-widget-show3d-gif-"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stack, labels, source = _load_stack(args)
    labels = labels[: stack.shape[0]]
    panels, panel_titles = _multi_panel_stacks(stack)
    max_cols = len(panels)
    panel_gap = int(args.panel_gap)
    scale_bar_visible = not bool(args.no_scale_bar)
    zoom_indicator_visible = scale_bar_visible and not bool(args.no_zoom_label)
    panel_native_mb = round(sum(panel.nbytes for panel in panels) / 1024 / 1024, 3)
    planned_exports = _planned_exports(
        stack,
        list(args.qualities),
        playback=args.playback,
        panel_count=len(panels),
        max_cols=max_cols,
        panel_gap=panel_gap,
    )
    dry_run_decision = _dry_run_decision(
        panel_native_mb=panel_native_mb,
        planned_exports=planned_exports,
        max_native_mb=float(args.max_native_mb),
        max_work_mb=float(args.max_work_mb),
    )

    report_base = {
        "artifact_dir": str(artifact_dir),
        "dry_run": bool(args.dry_run),
        "source": source,
        "source_file_mb": _source_file_mb(source),
        "input_shape": list(stack.shape),
        "input_dtype": str(stack.dtype),
        "native_mb": round(stack.nbytes / 1024 / 1024, 3),
        "panel_native_mb": panel_native_mb,
        "panels": panel_titles,
        "panel_gap": panel_gap,
        "panel_labels": bool(args.panel_labels),
        "scale_bar": {"visible": scale_bar_visible, "sampling_nm": float(args.sampling_nm)},
        "zoom_indicator": zoom_indicator_visible,
        "fps": float(args.fps),
        "playback": args.playback,
        "frame_labels": bool(args.frame_labels),
        "planned_exports": planned_exports,
        "dry_run_decision": dry_run_decision,
    }
    if args.dry_run:
        report = {
            **report_base,
            "total_size_mb": 0.0,
            "exports": [],
            "recommendation": dry_run_decision["summary"],
        }
        (artifact_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_html(artifact_dir, report)
        print(json.dumps(report, indent=2))
        return 0

    widget = Show3D(
        *panels,
        title="Show3D GIF presentation smoke",
        panel_titles=panel_titles,
        panel_frame_labels=[labels for _ in panels],
        max_cols=max_cols,
        panel_gap=panel_gap,
        sampling=args.sampling_nm,
        units="nm",
        fps=args.fps,
        cmap="inferno",
        show_controls=False,
        show_panel_titles=bool(args.panel_labels),
        show_scale_bar=scale_bar_visible,
        show_zoom_indicator=zoom_indicator_visible,
        save_state=False,
        verbose=False,
    )

    exports: list[dict[str, Any]] = []
    for quality in args.qualities:
        path = artifact_dir / f"show3d-timeseries-{quality}.gif"
        start = time.perf_counter()
        written = widget.save_gif(
            path,
            quality=quality,
            fps=args.fps,
            playback=args.playback,
            show_frame_labels=bool(args.frame_labels),
            background="black",
        )
        seconds = time.perf_counter() - start
        metrics = _gif_metrics(written)
        exports.append({
            "quality": quality,
            "path": str(written),
            "seconds": round(seconds, 3),
            "size_mb": round(written.stat().st_size / 1024 / 1024, 3),
            **metrics,
        })

    total_size_mb = sum(item["size_mb"] for item in exports)
    report = {
        **report_base,
        "total_size_mb": round(total_size_mb, 3),
        "exports": exports,
        "recommendation": "Use GIF medium for most PowerPoint/email loops; use GIF high when sharpness matters and the file size is acceptable.",
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_html(artifact_dir, report)
    print(json.dumps(report, indent=2))
    if total_size_mb > args.max_total_mb:
        raise RuntimeError(f"GIF report exceeded --max-total-mb={args.max_total_mb}: {total_size_mb:.3f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
