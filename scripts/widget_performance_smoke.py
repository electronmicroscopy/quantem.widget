#!/usr/bin/env python3
"""Generate real-data Show2D/Show3D performance UI artifacts.

This script owns the backend half of the performance UI gate. It creates
standalone HTML pages, timing JSON, a browser-drive plan, and a compact report
that agents can open in the in-app browser and measure. Browser FPS is measured
by driving the exported pages; Python timings here are only backend/export
timings.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from quantem.widget import Show2D, Show3D


DEFAULT_SEARCH_ROOTS = [
    Path.home() / "reports",
    Path.home() / "quantem-data" / "datasets",
    Path.home() / "share" / "denova_test_data",
    Path("/data/quantem"),
    Path("/scratch/quantem"),
]


@dataclass
class TimerRows:
    rows: list[dict[str, Any]]

    def record(self, case: str, start: float, detail: Any) -> None:
        self.rows.append({
            "case": case,
            "seconds": round(time.perf_counter() - start, 3),
            "detail": detail,
        })


def _native_mb(array: np.ndarray) -> float:
    return round(float(array.nbytes) / 1024 / 1024, 3)


def _as_float32(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim > 2:
        arr = arr.reshape((-1, *arr.shape[-2:]))[0]
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _center_crop_or_tile(image: np.ndarray, size: int) -> np.ndarray:
    """Return a square float32 image of ``size`` using real pixels only."""
    arr = _as_float32(image)
    h, w = arr.shape
    if h >= size and w >= size:
        r0 = (h - size) // 2
        c0 = (w - size) // 2
        return np.ascontiguousarray(arr[r0 : r0 + size, c0 : c0 + size], dtype=np.float32)
    reps = (int(np.ceil(size / max(h, 1))), int(np.ceil(size / max(w, 1))))
    tiled = np.tile(arr, reps)
    return np.ascontiguousarray(tiled[:size, :size], dtype=np.float32)


def _discover_real_image_paths(search_roots: list[Path], limit: int) -> list[Path]:
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in ("**/merged_cropped.tif", "**/*pair_fullres.npz", "**/*.tif", "**/*.tiff"):
            candidates.extend(path for path in root.glob(pattern) if path.is_file())
        if len(candidates) >= limit:
            break
    # Prefer drift outputs and processed gold pairs, which are common real-data
    # sources, then keep deterministic ordering for reports.
    candidates = sorted(
        dict.fromkeys(candidates),
        key=lambda path: (
            "merged_cropped" not in path.name,
            "quantem-data" not in str(path),
            str(path),
        ),
    )
    return candidates[:limit]


def _read_npz_images(path: Path) -> list[np.ndarray]:
    data = np.load(path, allow_pickle=False)
    arrays: list[np.ndarray] = []
    try:
        for key in data.files:
            value = np.asarray(data[key])
            if value.ndim >= 2 and np.issubdtype(value.dtype, np.number):
                if value.ndim == 2:
                    arrays.append(_as_float32(value))
                elif value.ndim == 3:
                    arrays.extend(_as_float32(frame) for frame in value[: min(value.shape[0], 8)])
    finally:
        data.close()
    return arrays


def _load_real_images(paths: list[Path], count: int, size: int) -> tuple[list[np.ndarray], list[str]]:
    images: list[np.ndarray] = []
    sources: list[str] = []
    for path in paths:
        try:
            if path.suffix.lower() == ".npz":
                for image in _read_npz_images(path):
                    images.append(_center_crop_or_tile(image, size))
                    sources.append(str(path))
                    if len(images) >= count:
                        return images, sources
            else:
                images.append(_center_crop_or_tile(tifffile.imread(path), size))
                sources.append(str(path))
                if len(images) >= count:
                    return images, sources
        except Exception as exc:  # pragma: no cover - only for local data drift
            print(f"warning: skipped {path}: {exc}")
    if not images:
        raise RuntimeError("no readable real microscopy images found")
    while len(images) < count:
        idx = len(images) % len(images)
        images.append(np.ascontiguousarray(np.rot90(images[idx], len(images) % 4), dtype=np.float32))
        sources.append(f"{sources[idx]} [derived rotation]")
    return images, sources


def _make_show3d_stack(images: list[np.ndarray], frames: int, size: int) -> np.ndarray:
    panels = []
    for panel_idx, image in enumerate(images):
        base = _center_crop_or_tile(image, size)
        movie = []
        for frame in range(frames):
            shift_r = (frame * (panel_idx + 1)) % max(size, 1)
            shift_c = (frame * (panel_idx + 3)) % max(size, 1)
            rolled = np.roll(base, shift=(shift_r, shift_c), axis=(0, 1))
            weight = 0.92 + 0.08 * np.sin((frame + panel_idx) / max(frames, 1) * 2 * np.pi)
            movie.append(np.ascontiguousarray(rolled * weight, dtype=np.float32))
        panels.append(np.stack(movie, axis=0))
    return np.stack(panels, axis=0)


def _page_shape(requested_panels: int, preferred_pages: int) -> tuple[int, int, int]:
    """Return ``(n_pages, panels_per_page, total_panels)`` for page smoke data."""

    requested = max(1, int(requested_panels))
    if requested < 2:
        return 1, requested, requested
    n_pages = min(max(2, int(preferred_pages)), requested)
    panels_per_page = int(np.ceil(requested / n_pages))
    return n_pages, panels_per_page, n_pages * panels_per_page


def _write_html_report(artifact_dir: Path, report: dict[str, Any]) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{item.get('widget', '')}</td>"
        f"<td>{item.get('encoding', '')}</td>"
        f"<td>{item.get('seconds', '')}</td>"
        f"<td>{item.get('size_mb', ''):.3f}</td>"
        f"<td><a href='{Path(str(item.get('path'))).name}'>{Path(str(item.get('path'))).name}</a></td>"
        "</tr>"
        for item in report["exports"]
    )
    timing = json.dumps(report, indent=2)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>quantem.widget performance UI smoke</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; }}
    table {{ border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #ccd3db; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f5f7; }}
    code, pre {{ background: #f5f7f9; padding: 2px 4px; border-radius: 4px; }}
    pre {{ overflow: auto; padding: 12px; max-width: 1100px; }}
  </style>
</head>
<body>
  <h1>quantem.widget performance UI smoke</h1>
  <p>Open each export below in the in-app browser and drive the browser plan in
  <code>browser-plan.json</code>. Python timings here are backend/export
  timings; browser FPS must be measured by interacting with the pages.</p>
  <table>
    <thead><tr><th>Widget</th><th>Encoding</th><th>Export seconds</th><th>Size MB</th><th>File</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Run summary</h2>
  <pre>{timing}</pre>
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(html, encoding="utf-8")


def _write_browser_plan(artifact_dir: Path, report: dict[str, Any]) -> None:
    pages = []
    for item in report["exports"]:
        pages.append({
            "widget": item["widget"],
            "encoding": item["encoding"],
            "file": Path(str(item["path"])).name,
            "url_path": Path(str(item["path"])).name,
            "required_interactions": [
                "wait for nonblank canvas",
                "measure requestAnimationFrame FPS",
                "wheel zoom repeatedly over the image",
                "drag pan repeatedly over the image",
                "open Export menu and confirm labels",
                "toggle FFT where available",
                "exercise columns/panels controls where available",
            ],
        })
    plan = {
        "version": 1,
        "description": "Open these exported pages through a local static server and drive with the in-app browser.",
        "target_fps": 30,
        "target_first_paint_seconds": 10,
        "artifact_dir": str(artifact_dir),
        "pages": pages,
    }
    (artifact_dir / "browser-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--quick", action="store_true", help="Use smaller real-derived targets for local iteration.")
    parser.add_argument("--show2d-panels", type=int, default=None)
    parser.add_argument("--show2d-size", type=int, default=None)
    parser.add_argument("--show3d-panels", type=int, default=None)
    parser.add_argument("--show3d-frames", type=int, default=None)
    parser.add_argument("--show3d-size", type=int, default=None)
    parser.add_argument("--show3d-export-downsample", type=int, default=4)
    parser.add_argument(
        "--skip-show3d-full-export",
        action="store_true",
        help="Skip the unbinned Show3D uint8 export for local heavy browser signoff.",
    )
    parser.add_argument("--max-single-mb", type=float, default=250.0)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or Path(tempfile.mkdtemp(prefix="quantem-widget-perf-ui-"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timers = TimerRows([])
    exports: list[dict[str, Any]] = []

    show2d_panels = args.show2d_panels or (4 if args.quick else 8)
    show2d_size = args.show2d_size or (1024 if args.quick else 4096)
    show3d_panels = args.show3d_panels or (3 if args.quick else 12)
    show3d_frames = args.show3d_frames or (12 if args.quick else 32)
    show3d_size = args.show3d_size or (512 if args.quick else 2048)

    env_roots = [
        Path(value).expanduser()
        for value in os.environ.get("QUANTEM_WIDGET_REAL_DATA_ROOTS", "").split(os.pathsep)
        if value
    ]
    search_roots = args.search_root or env_roots or DEFAULT_SEARCH_ROOTS
    show2d_pages, show2d_panels_per_page, show2d_total_panels = _page_shape(show2d_panels, 2)
    show3d_pages, show3d_panels_per_page, show3d_total_panels = _page_shape(
        show3d_panels,
        3 if show3d_panels >= 6 else 2,
    )
    source_paths = _discover_real_image_paths(search_roots, max(show2d_total_panels, show3d_total_panels, 8))

    start = time.perf_counter()
    show2d_images, show2d_sources = _load_real_images(source_paths, show2d_total_panels, show2d_size)
    show2d_stack = np.stack(show2d_images, axis=0)
    show2d_pages_stack = show2d_stack.reshape(
        show2d_pages,
        show2d_panels_per_page,
        show2d_size,
        show2d_size,
    )
    timers.record("load_show2d_real_images", start, {
        "shape": list(show2d_stack.shape),
        "paged_shape": list(show2d_pages_stack.shape),
        "native_mb": _native_mb(show2d_stack),
        "sources": show2d_sources,
    })

    start = time.perf_counter()
    show2d = Show2D(
        show2d_pages_stack,
        title=f"Performance Show2D real pages {show2d_pages}x{show2d_panels_per_page}x{show2d_size}",
        labels=[f"P{i + 1:02d}" for i in range(show2d_panels_per_page)],
        page_labels=[f"page {i + 1:02d}" for i in range(show2d_pages)],
        ncols=min(4, show2d_panels_per_page),
        display_bin="auto",
        show_fft=False,
        verbose=False,
    )
    timers.record("construct_show2d", start, {
        "n_pages": show2d_pages,
        "panels_per_page": show2d_panels_per_page,
        "n_panels": show2d_total_panels,
        "size": show2d_size,
        "display_bin": getattr(show2d, "display_bin_factor", None),
    })
    for encoding in ("uint8", "full"):
        start = time.perf_counter()
        out = show2d.export_html(artifact_dir / f"show2d-real-{encoding}.html", encoding=encoding)
        elapsed = time.perf_counter() - start
        exports.append({
            "widget": "show2d",
            "variant": f"show2d-real-{encoding}",
            "encoding": encoding,
            "path": str(out),
            "seconds": round(elapsed, 3),
            "size_mb": round(out.stat().st_size / 1024 / 1024, 3),
        })

    start = time.perf_counter()
    show3d_images, show3d_sources = _load_real_images(source_paths, show3d_total_panels, show3d_size)
    show3d_stack = _make_show3d_stack(show3d_images, show3d_frames, show3d_size)
    show3d_pages_stack = show3d_stack.reshape(
        show3d_pages,
        show3d_panels_per_page,
        show3d_frames,
        show3d_size,
        show3d_size,
    )
    timers.record("load_show3d_real_derived_stack", start, {
        "shape": list(show3d_stack.shape),
        "paged_shape": list(show3d_pages_stack.shape),
        "native_mb": _native_mb(show3d_stack),
        "sources": show3d_sources,
    })

    start = time.perf_counter()
    show3d = Show3D(
        show3d_pages_stack,
        title=(
            "Performance Show3D real-derived pages "
            f"{show3d_pages}x{show3d_panels_per_page}x{show3d_frames}x{show3d_size}"
        ),
        panel_titles=[f"P{i + 1:02d}" for i in range(show3d_panels_per_page)],
        page_labels=[f"page {i + 1:02d}" for i in range(show3d_pages)],
        max_cols=min(4, show3d_panels_per_page),
        show_fft=True,
        fft_layout="overlay",
        fft_overlay_size=0.25,
        save_state=False,
    )
    timers.record("construct_show3d", start, {
        "n_pages": show3d_pages,
        "panels_per_page": show3d_panels_per_page,
        "n_panels": show3d_total_panels,
        "frames": show3d_frames,
        "size": show3d_size,
    })
    show3d_exports = []
    if not args.skip_show3d_full_export:
        show3d_exports.append(("uint8", 1))
    show3d_exports.append(("uint8", args.show3d_export_downsample))
    for encoding, downsample in show3d_exports:
        suffix = "uint8" if downsample == 1 else f"uint8-{downsample}xbin"
        start = time.perf_counter()
        out = show3d.export_html(
            artifact_dir / f"show3d-real-derived-{suffix}.html",
            encoding=encoding,
            downsample=downsample,
        )
        elapsed = time.perf_counter() - start
        exports.append({
            "widget": "show3d",
            "variant": f"show3d-real-derived-{suffix}",
            "encoding": suffix,
            "path": str(out),
            "seconds": round(elapsed, 3),
            "size_mb": round(out.stat().st_size / 1024 / 1024, 3),
        })

    too_large = [item for item in exports if float(item["size_mb"]) > args.max_single_mb]
    report = {
        "artifact_dir": str(artifact_dir),
        "created_at_unix": time.time(),
        "backend": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "pid": os.getpid(),
        },
        "targets": {
            "show2d": {
                "requested_panels": show2d_panels,
                "pages": show2d_pages,
                "panels_per_page": show2d_panels_per_page,
                "total_panels": show2d_total_panels,
                "size": show2d_size,
            },
            "show3d": {
                "requested_panels": show3d_panels,
                "pages": show3d_pages,
                "panels_per_page": show3d_panels_per_page,
                "total_panels": show3d_total_panels,
                "frames": show3d_frames,
                "size": show3d_size,
            },
        },
        "timings": timers.rows,
        "exports": exports,
        "too_large": too_large,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_browser_plan(artifact_dir, report)
    _write_html_report(artifact_dir, report)
    print(json.dumps(report, indent=2))
    if too_large:
        raise RuntimeError(f"export(s) exceeded {args.max_single_mb:.1f} MB: {too_large}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
