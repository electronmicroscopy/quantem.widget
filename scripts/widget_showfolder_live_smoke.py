#!/usr/bin/env python3
"""Exercise ShowFolder live-folder refresh and selected-viewer handoff."""

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
import torch

from quantem.widget import ShowFolder
from quantem.widget.render import save_thumbnail


def _metadata() -> np.ndarray:
    payload = json.dumps({
        "Scan": {"ScanRotation": "0"},
        "BinaryResult": {"PixelSize": {"height": 1e-9, "width": 1e-9}},
    }).encode()
    arr = np.zeros((len(payload) + 1, 1), dtype=np.uint8)
    arr[: len(payload), 0] = np.frombuffer(payload, dtype=np.uint8)
    return arr


def _image_emd(path: Path, *, offset: int) -> None:
    data = np.arange(20 * 24, dtype=np.float32).reshape(20, 24) + float(offset)
    with h5py.File(path, "w") as h5:
        group = h5.create_group("Data/Image/uid")
        group.create_dataset("Data", data=data)
        group.create_dataset("Metadata", data=_metadata())


def _write_master(path: Path) -> None:
    idx = int(path.name.split("_master.h5", 1)[0].rsplit("_", 1)[-1])
    data = np.full((4, 4, 8, 8), idx + 1, dtype=np.uint16)
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry/data")
        entry.create_dataset("data", data=data)


def _export(widget: Any, path: Path, *, title: str) -> Path | None:
    export = getattr(widget, "export_html", None)
    if export is None:
        return None
    return Path(export(path, title=title))


def _write_thumbnail_previews(artifact_dir: Path, browser: Any, *, prefix: str) -> list[dict[str, Any]]:
    preview_dir = artifact_dir / "previews"
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for gallery, image_items in browser.image_galleries:
        data = np.asarray(getattr(gallery, "_data", np.empty((0, 0, 0))))
        if data.ndim < 3:
            continue
        for idx, item in enumerate(image_items):
            if idx >= data.shape[0] or item.file_id in seen:
                continue
            seen.add(item.file_id)
            out = preview_dir / f"{prefix}-{item.file_id}.webp"
            save_thumbnail(data[idx], out, size=96, cmap="inferno")
            rows.append({
                "id": item.file_id,
                "file": item.path.name,
                "webp": out.relative_to(artifact_dir).as_posix(),
                "bytes": out.stat().st_size,
            })
    return rows


def _run_image_live_smoke(artifact_dir: Path) -> dict[str, Any]:
    folder = artifact_dir / "live-images"
    folder.mkdir(parents=True, exist_ok=True)
    _image_emd(folder / "0010 - HAADF live.emd", offset=0)
    _image_emd(folder / "0011 - HAADF live.emd", offset=100)

    widget = ShowFolder(
        folder,
        thumb=10,
        group_by="none",
        cache_dir=artifact_dir / "cache-images",
    )
    assert widget.browser is not None
    assert widget.browser.gallery is not None
    widget.browser.gallery.star_panel(0)
    selected_2d = widget.show_selected()
    selected_3d = widget.show_selected_stack()
    widget.browser._active_selected_modes = {"show2d", "show3d"}
    widget.browser._selected_show2d_widget = selected_2d
    widget.browser._selected_show3d_widget = selected_3d
    before_widget_id = id(widget.widget)
    before_items = [item.file_id for item in widget.items]

    widget.watch(start=False)
    _image_emd(folder / "0012 - HAADF live.emd", offset=200)
    changed = widget.watch_once()
    after_items = [item.file_id for item in widget.items]

    assert changed is True
    assert id(widget.widget) == before_widget_id
    assert after_items == ["0010", "0011", "0012"]
    assert widget.browser._selected_show2d_widget is selected_2d
    assert widget.browser._selected_show3d_widget is selected_3d
    assert selected_2d._data.shape[0] == 1
    assert selected_3d.n_slices == 1

    widget.browser.gallery.star_panel(2)
    selected_paths = widget.selected_paths("image")
    selected_folders = widget.selected_folders()
    assert [path.name for path in selected_paths] == [
        "0010 - HAADF live.emd",
        "0012 - HAADF live.emd",
    ]
    assert selected_folders == [folder.resolve()]
    assert selected_2d._data.shape[0] == 2
    assert selected_3d.n_slices == 2
    thumbnail_previews = _write_thumbnail_previews(artifact_dir, widget.browser, prefix="live-image")

    exports = {
        "showfolder": _export(
            widget,
            artifact_dir / "showfolder-live-images.html",
            title="ShowFolder live images",
        ),
        "show2d": _export(
            selected_2d,
            artifact_dir / "showfolder-live-show2d.html",
            title="ShowFolder live Show2D",
        ),
        "show3d": _export(
            selected_3d,
            artifact_dir / "showfolder-live-show3d.html",
            title="ShowFolder live Show3D",
        ),
    }
    return {
        "name": "ShowFolder live images -> Show2D/Show3D",
        "passed": True,
        "before_items": before_items,
        "after_items": after_items,
        "watch_changed": changed,
        "widget_id_preserved": id(widget.widget) == before_widget_id,
        "selected_path_names": [path.name for path in selected_paths],
        "selected_folders": [str(path) for path in selected_folders],
        "show2d_panels": int(selected_2d._data.shape[0]),
        "show3d_slices": int(selected_3d.n_slices),
        "thumbnail_previews": thumbnail_previews,
        "exports": {key: None if value is None else value.name for key, value in exports.items()},
    }


def _run_master_live_smoke(artifact_dir: Path) -> dict[str, Any]:
    import quantem.widget as qw
    import quantem.widget.io as wio

    folder = artifact_dir / "live-4dstem"
    folder.mkdir(parents=True, exist_ok=True)
    _write_master(folder / "scan_000_master.h5")

    real_load = qw.load
    real_discover_masters = wio.discover_masters
    real_is_master_ready = wio.is_master_ready

    def fake_discover_masters(
        path: str,
        *,
        scan_shape=None,
        verbose: bool = False,
        **kwargs,
    ):
        return sorted(str(item) for item in Path(path).glob("*_master.h5"))

    def fake_is_master_ready(path: str) -> bool:
        return Path(path).exists()

    class _LoadResult:
        def __init__(self, path: str) -> None:
            stem = Path(path).name.split("_master.h5", 1)[0]
            idx = int(stem.rsplit("_", 1)[-1])
            self.data = torch.full((4, 4, 8, 8), idx + 1, dtype=torch.uint8)

    def fake_load(path: str, *, det_bin=4, dtype="u8", verbose: bool = False, **kwargs):
        return _LoadResult(path)

    qw.load = fake_load
    wio.discover_masters = fake_discover_masters
    wio.is_master_ready = fake_is_master_ready
    try:
        widget = ShowFolder(
            folder,
            thumb=10,
            group_by="none",
            cache_dir=artifact_dir / "cache-4dstem",
        )
        assert widget.browser is not None
        first = widget.browser.open_show4dstem(gpus=None, page_budget=1, det_bin=4, dtype="u8")
        assert first is not None
        assert first.n_frames == 1
        assert widget.master_qc_rows[0]["status"] == "ready"

        widget.watch(start=False)
        _write_master(folder / "scan_001_master.h5")
        changed = widget.watch_once()
        second = widget.browser._selected_show4dstem_widget
        assert changed is True
        assert second is not None
        assert second is not first
        assert second.n_frames == 2
        assert list(second.frame_labels) == ["scan_000", "scan_001"]

        export_path = _export(
            second,
            artifact_dir / "showfolder-live-show4dstem.html",
            title="ShowFolder live Show4DSTEM",
        )
        return {
            "name": "ShowFolder live 4D-STEM masters -> Show4DSTEM",
            "passed": True,
            "watch_changed": changed,
            "first_frames": 1,
            "after_frames": int(second.n_frames),
            "reused_old_widget": second is first,
            "frame_labels": list(second.frame_labels),
            "master_qc": widget.master_qc_rows,
            "exports": {"show4dstem": None if export_path is None else export_path.name},
        }
    finally:
        qw.load = real_load
        wio.discover_masters = real_discover_masters
        wio.is_master_ready = real_is_master_ready


def _write_report(artifact_dir: Path, report: dict[str, Any]) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(step['name'])}</td>"
        f"<td>{'pass' if step['passed'] else 'fail'}</td>"
        f"<td>{html.escape(json.dumps(step, sort_keys=True))}</td>"
        "</tr>"
        for step in report["steps"]
    )
    links = []
    for step in report["steps"]:
        for name in (step.get("exports") or {}).values():
            if name:
                links.append(f"<li><a href='{html.escape(name)}'>{html.escape(name)}</a></li>")
    preview_cards = []
    for step in report["steps"]:
        for preview in step.get("thumbnail_previews") or []:
            src = html.escape(str(preview["webp"]))
            label = html.escape(f"{preview['id']} · {preview['file']}")
            preview_cards.append(
                "<figure>"
                f"<img src='{src}' alt='{label}'>"
                f"<figcaption>{label}</figcaption>"
                "</figure>"
            )
    qc_rows = []
    for step in report["steps"]:
        for row in step.get("master_qc") or []:
            scan = row.get("scan_shape")
            det = row.get("detector_shape")
            qc_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('file', '')))}</td>"
                f"<td>{html.escape(str(row.get('status', '')))}</td>"
                f"<td>{'' if scan is None else html.escape('x'.join(str(v) for v in scan))}</td>"
                f"<td>{'' if det is None else html.escape('x'.join(str(v) for v in det))}</td>"
                f"<td>{html.escape(str(row.get('n_frames') or ''))}</td>"
                f"<td>{html.escape(str(row.get('dtype') or ''))}</td>"
                f"<td>{html.escape(str(row.get('reason', '')))}</td>"
                f"<td>{html.escape(str(row.get('action', '')))}</td>"
                "</tr>"
            )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ShowFolder live-folder smoke</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; line-height: 1.45; }}
    table {{ border-collapse: collapse; margin-top: 12px; width: min(1180px, 100%); }}
    th, td {{ border: 1px solid #ccd3db; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f5f7; }}
    code, pre {{ background: #f5f7f9; border-radius: 4px; }}
    code {{ padding: 2px 4px; }}
    pre {{ overflow: auto; padding: 12px; max-width: 1180px; }}
    .previews {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; }}
    figure {{ margin: 0; border: 1px solid #ccd3db; border-radius: 6px; padding: 8px; background: #f8fafc; }}
    figure img {{ display: block; width: 96px; height: 96px; object-fit: contain; background: #111827; }}
    figcaption {{ max-width: 160px; margin-top: 6px; font-size: 12px; color: #3b4654; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>ShowFolder live-folder smoke: {'PASS' if report['passed'] else 'FAIL'}</h1>
  <p>This report proves the folder watcher owns live updates while Show2D,
  Show3D, and Show4DSTEM remain display widgets. It also writes compact WebP
  previews for visual review while keeping the numeric thumbnail cache as
  arrays for widget handoff. It uses tiny generated files; heavy real-data
  performance is covered by the separate local-only signoffs.</p>
  <h2>Review Exports</h2>
  <ul>{''.join(links)}</ul>
  <h2>Thumbnail Previews</h2>
  <div class="previews">{''.join(preview_cards) if preview_cards else '<p>No thumbnail previews.</p>'}</div>
  <h2>4D-STEM Master QC</h2>
  <table>
    <thead><tr><th>Master</th><th>Status</th><th>Scan</th><th>Detector</th><th>Frames</th><th>Dtype</th><th>Reason</th><th>Next step</th></tr></thead>
    <tbody>{''.join(qc_rows) if qc_rows else '<tr><td colspan="8">No master QC rows.</td></tr>'}</tbody>
  </table>
  <h2>Checks</h2>
  <table>
    <thead><tr><th>Scenario</th><th>Status</th><th>Evidence</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Machine-readable report</h2>
  <p><a href="report.json">report.json</a></p>
  <pre>{html.escape(json.dumps(report, indent=2))}</pre>
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(page, encoding="utf-8")
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or Path(tempfile.mkdtemp(prefix="quantem-widget-showfolder-live-"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = artifact_dir.resolve()

    started = time.perf_counter()
    steps = [
        _run_image_live_smoke(artifact_dir),
        _run_master_live_smoke(artifact_dir),
    ]
    report = {
        "artifact_dir": str(artifact_dir),
        "created_at_unix": int(time.time()),
        "seconds": round(time.perf_counter() - started, 3),
        "passed": all(step["passed"] for step in steps),
        "steps": steps,
    }
    _write_report(artifact_dir, report)
    print(json.dumps(report, indent=2))
    print(f"ShowFolder live-folder smoke report: {artifact_dir / 'index.html'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
