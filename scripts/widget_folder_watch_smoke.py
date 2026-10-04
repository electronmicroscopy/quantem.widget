#!/usr/bin/env python3
"""Exercise direct viewer folder watching and file arrival."""

from __future__ import annotations

import argparse
import html
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np

from quantem.widget import Show2D, Show3D, Show4DSTEM


def _image_npy(path: Path, *, value: float) -> None:
    """Write one tiny exact image used by direct from_folder lifecycle checks."""
    np.save(path, np.full((12, 16), float(value), dtype=np.float32))


def _write_external_master(folder: Path, *, index: int) -> Path:
    """Write one tiny external-link master accepted by the GPU loader."""
    data_path = folder / f"scan_{index:03d}_data_000001.h5"
    master_path = folder / f"scan_{index:03d}_master.h5"
    with h5py.File(data_path, "w") as h5:
        h5.create_dataset(
            "entry/data/data",
            data=np.full((16, 4, 4), index + 1, dtype=np.uint16),
            chunks=(1, 4, 4),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
    with h5py.File(master_path, "w") as h5:
        h5.require_group("entry/data")["data_000001"] = h5py.ExternalLink(
            data_path.name,
            "entry/data/data",
        )
        detector = h5.require_group(
            "entry/instrument/detector/detectorSpecific"
        )
        detector.create_dataset("ntrigger", data=16)
    return master_path


def _export(widget: Any, path: Path, *, title: str) -> Path | None:
    export = getattr(widget, "export_html", None)
    if export is None:
        return None
    return Path(export(path, title=title))


def _browser_export_row(
    path: Path,
    *,
    widget: str,
    variant: str,
    seconds: float = 0.0,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the row shape consumed by widget_browser_smoke.py."""
    return {
        "widget": str(widget),
        "variant": str(variant),
        "path": str(path.resolve()),
        "seconds": round(float(seconds), 3),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "options": dict(options or {}),
    }


def _embedded_watch_contract(path: Path) -> dict[str, Any]:
    """Inspect exported widget model state without trusting visible label text."""
    text = path.read_text(encoding="utf-8")
    states = sorted(
        set(
            re.findall(
                r'"folder_watch_state"\s*:\s*"([^"]+)"',
                text,
            )
        )
    )
    return {
        "embedded_states": states,
        "watching_embedded": "watching" in states,
        "hidden_snapshot": states == ["hidden"],
    }


def _watch_snapshot(
    widget: Any,
    *,
    event: str,
    started: float,
    count_attr: str,
) -> dict[str, Any]:
    """Capture one JSON-safe direct-viewer lifecycle observation."""
    return {
        "event": str(event),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "python_id": int(id(widget)),
        "model_id": str(getattr(widget, "model_id", "")),
        "state": str(getattr(widget, "folder_watch_state", "")),
        "detail": str(getattr(widget, "folder_watch_detail", "")),
        "count": int(getattr(widget, count_attr)),
        "compare_page_loading": bool(
            getattr(widget, "compare_page_loading", False)
        ),
        "compare_page_loaded_count": int(
            getattr(widget, "compare_page_loaded_count", 0)
        ),
        "compare_panel_count": int(getattr(widget, "compare_panel_count", 0)),
        "compare_panel_indices": [
            int(value)
            for value in getattr(widget, "compare_panel_indices", [])
        ],
        "compare_cache_state": str(
            getattr(widget, "compare_page_cache_state", "")
        ),
        "virtual_image_bytes": len(
            bytes(getattr(widget, "compare_virtual_image_bytes", b""))
        ),
    }


def _run_direct_image_live_smoke(
    artifact_dir: Path,
    *,
    viewer_class: Any,
    widget_name: str,
    count_attr: str,
) -> dict[str, Any]:
    """Exercise one public Show2D/Show3D from_folder lifecycle in place."""
    folder = artifact_dir / f"direct-{widget_name}"
    folder.mkdir(parents=True, exist_ok=True)
    _image_npy(folder / "frame_000.npy", value=1)

    started = time.perf_counter()
    widget = viewer_class.from_folder(
        folder,
        pattern="*.npy",
        watch=True,
        watch_interval=60,
    )
    timeline = [
        _watch_snapshot(
            widget,
            event="mounted_with_initial_candidate_on_probation",
            started=started,
            count_attr=count_attr,
        )
    ]

    def record_state(change: dict[str, Any]) -> None:
        timeline.append(
            _watch_snapshot(
                widget,
                event=f"state:{change['new']}",
                started=started,
                count_attr=count_attr,
            )
        )

    widget.observe(record_state, names="folder_watch_state")
    initial_python_id = id(widget)
    initial_model_id = str(widget.model_id)
    try:
        assert int(getattr(widget, count_attr)) == 0
        assert widget.folder_watch_state == "waiting"

        initial_added = widget.poll_folder()
        assert initial_added == [0]
        timeline.append(
            _watch_snapshot(
                widget,
                event="initial_candidate_confirmed",
                started=started,
                count_attr=count_attr,
            )
        )
        assert widget.folder_watch_state == "watching"

        _image_npy(folder / "frame_001.npy", value=2)
        probation_added = widget.poll_folder()
        assert probation_added == []
        assert widget.folder_watch_state == "waiting"
        timeline.append(
            _watch_snapshot(
                widget,
                event="arrival_probation",
                started=started,
                count_attr=count_attr,
            )
        )

        stable_added = widget.poll_folder()
        assert stable_added == [1]
        timeline.append(
            _watch_snapshot(
                widget,
                event="stable_arrival_applied",
                started=started,
                count_attr=count_attr,
            )
        )
        assert int(getattr(widget, count_attr)) == 2
        assert id(widget) == initial_python_id
        assert str(widget.model_id) == initial_model_id
        np.testing.assert_array_equal(
            np.asarray(widget._data)[:, 0, 0],
            np.asarray([1, 2], dtype=np.float32),
        )

        widget.stop_folder_watch()
        timeline.append(
            _watch_snapshot(
                widget,
                event="stopped_before_static_export",
                started=started,
                count_attr=count_attr,
            )
        )
        assert widget.folder_watch_state == "stopped"
        saved_watch_state = widget.state_dict().get(
            "folder_watch_state",
            "absent",
        )
        assert saved_watch_state != "watching"

        export_path = artifact_dir / f"{widget_name}-from-folder-stopped.html"
        export_started = time.perf_counter()
        exported = _export(
            widget,
            export_path,
            title=f"{widget_name} from_folder stopped snapshot",
        )
        export_seconds = time.perf_counter() - export_started
        assert exported is not None
        static_contract = _embedded_watch_contract(exported)
        assert static_contract["watching_embedded"] is False
        assert static_contract["hidden_snapshot"] is True

        states = [str(item["state"]) for item in timeline]
        for required in ("waiting", "updating", "watching", "stopped"):
            assert required in states
        same_model = (
            {int(item["python_id"]) for item in timeline} == {initial_python_id}
            and {str(item["model_id"]) for item in timeline} == {initial_model_id}
        )
        assert same_model
        export_row = _browser_export_row(
            exported,
            widget=widget_name,
            variant=f"{widget_name}-folder-watch-static",
            seconds=export_seconds,
        )
        return {
            "name": f"Direct {widget_name}.from_folder live lifecycle",
            "kind": "direct_public_from_folder",
            "uses_monkeypatch": False,
            "passed": True,
            "same_mounted_model": same_model,
            "initial_probation_added": initial_added,
            "arrival_probation_added": probation_added,
            "stable_arrival_added": stable_added,
            "final_count": int(getattr(widget, count_attr)),
            "labels": list(widget.labels),
            "saved_state_watch": saved_watch_state,
            "static_watch_contract": static_contract,
            "timeline": timeline,
            "exports": {widget_name: exported.name},
            "export_rows": [export_row],
        }
    finally:
        widget.close()


def _run_direct_show4dstem_live_smoke(
    artifact_dir: Path,
) -> dict[str, Any]:
    """Exercise the production public GPU Show4DSTEM folder watcher."""
    folder = artifact_dir / "direct-show4dstem"
    folder.mkdir(parents=True, exist_ok=True)
    _write_external_master(folder, index=0)

    started = time.perf_counter()
    try:
        widget = Show4DSTEM.from_folder(
            folder,
            gpus=None,
            scan_size=4,
            det_bin=1,
            dtype="native",
            watch=True,
            watch_interval=60,
            view_mode="multiple",
            columns=2,
            page_size=4,
            page_budget=1,
            preload_all_if_fits=False,
            warm_cache=False,
            preview_cache=False,
            precompute_virtual_images=False,
            verbose=False,
        )
    except RuntimeError as exc:
        if "No QuantEM GPU backend is available" not in str(exc):
            raise
        return {
            "name": "Direct Show4DSTEM.from_folder live lifecycle",
            "kind": "direct_public_from_folder",
            "uses_monkeypatch": False,
            "passed": True,
            "skipped": True,
            "skip_reason": "No native CUDA or MPS backend is available.",
            "exports": {},
            "export_rows": [],
        }
    timeline = [
        _watch_snapshot(
            widget,
            event="mounted_with_initial_master",
            started=started,
            count_attr="n_frames",
        )
    ]

    def record_state(change: dict[str, Any]) -> None:
        timeline.append(
            _watch_snapshot(
                widget,
                event=f"state:{change['new']}",
                started=started,
                count_attr="n_frames",
            )
        )

    widget.observe(record_state, names="folder_watch_state")
    initial_python_id = id(widget)
    initial_model_id = str(widget.model_id)
    try:
        assert widget.n_frames == 1
        assert widget.folder_watch_state == "watching"
        arrival_start = len(timeline)
        _write_external_master(folder, index=1)

        probation_added = widget.poll_folder()
        assert probation_added == []
        assert widget.folder_watch_state == "waiting"
        timeline.append(
            _watch_snapshot(
                widget,
                event="arrival_headers_on_probation",
                started=started,
                count_attr="n_frames",
            )
        )

        stable_added = widget.poll_folder()
        assert stable_added == [1]
        timeline.append(
            _watch_snapshot(
                widget,
                event="arrival_registered_page_refresh_pending",
                started=started,
                count_attr="n_frames",
            )
        )
        widget.wait_for_compare_page(timeout=10)
        timeline.append(
            _watch_snapshot(
                widget,
                event="active_page_authoritative",
                started=started,
                count_attr="n_frames",
            )
        )

        assert widget.n_frames == 2
        assert id(widget) == initial_python_id
        assert str(widget.model_id) == initial_model_id
        assert widget.compare_page_loading is False
        is_mps = hasattr(widget, "_mps_folder_live")
        if is_mps:
            widget.frame_idx = 1
            virtual_image = np.frombuffer(
                widget.virtual_image_bytes,
                dtype=np.float32,
            ).reshape(4, 4)
            assert np.all(np.isfinite(virtual_image))
            image_means = [float(np.mean(virtual_image))]
            assert image_means[0] > 0
        else:
            assert widget.compare_page_loaded_count == 2
            assert widget.compare_panel_count == 2
            assert list(widget.compare_panel_indices) == [0, 1]
            virtual_images = np.frombuffer(
                widget.compare_virtual_image_bytes,
                dtype=np.float32,
            ).reshape(2, 4, 4)
            image_means = [float(np.mean(image)) for image in virtual_images]
            assert np.all(np.isfinite(virtual_images))
            assert image_means[0] > 0
            assert image_means[1] > image_means[0]
        assert widget.folder_watch_state == "watching"

        arrival_timeline = timeline[arrival_start:]
        arrival_states = [str(item["state"]) for item in arrival_timeline]
        assert "waiting" in arrival_states
        assert "updating" in arrival_states
        green_points = [
            item for item in arrival_timeline if item["state"] == "watching"
        ]
        assert green_points
        authoritative_green = green_points[-1]
        assert authoritative_green["compare_page_loading"] is False
        if not is_mps:
            assert authoritative_green["compare_page_loaded_count"] == 2
            assert authoritative_green["compare_panel_indices"] == [0, 1]
        assert authoritative_green["virtual_image_bytes"] > 0

        widget.stop_folder_watch()
        timeline.append(
            _watch_snapshot(
                widget,
                event="stopped_before_static_export",
                started=started,
                count_attr="n_frames",
            )
        )
        assert widget.folder_watch_state == "stopped"
        saved_watch_state = widget.state_dict().get(
            "folder_watch_state",
            "absent",
        )
        assert saved_watch_state != "watching"

        export_path = artifact_dir / "show4dstem-from-folder-stopped.html"
        export_started = time.perf_counter()
        exported = Path(
            widget.export_html(
                export_path,
                title="Show4DSTEM from_folder stopped snapshot",
                encoding="uint8",
                downsample=1,
            )
        )
        export_seconds = time.perf_counter() - export_started
        static_contract = _embedded_watch_contract(exported)
        assert static_contract["watching_embedded"] is False
        assert static_contract["hidden_snapshot"] is True

        same_model = (
            {int(item["python_id"]) for item in timeline} == {initial_python_id}
            and {str(item["model_id"]) for item in timeline} == {initial_model_id}
        )
        assert same_model
        export_options = {"encoding": "uint8", "downsample": 1}
        export_row = _browser_export_row(
            exported,
            widget="show4dstem",
            variant="show4dstem-folder-watch-static",
            seconds=export_seconds,
            options=export_options,
        )
        return {
            "name": "Direct Show4DSTEM.from_folder live lifecycle",
            "kind": "direct_public_from_folder",
            "backend": "mps" if is_mps else "cuda",
            "uses_monkeypatch": False,
            "loader_note": (
                "Production native-GPU path over tiny bitshuffle-LZ4 external-link "
                "HDF5 masters; no fake loader."
            ),
            "passed": True,
            "same_mounted_model": same_model,
            "arrival_probation_added": probation_added,
            "stable_arrival_added": stable_added,
            "authoritative_before_green": True,
            "final_count": int(widget.n_frames),
            "frame_labels": list(widget.frame_labels),
            "active_page_indices": list(widget.compare_panel_indices),
            "active_page_loaded_count": int(widget.compare_page_loaded_count),
            "virtual_image_means": image_means,
            "saved_state_watch": saved_watch_state,
            "static_watch_contract": static_contract,
            "timeline": timeline,
            "exports": {"show4dstem": exported.name},
            "export_rows": [export_row],
        }
    finally:
        widget.close()


def _write_browser_plan(artifact_dir: Path, report: dict[str, Any]) -> None:
    """Write standalone-export review instructions for existing browser smoke."""
    pages = []
    for item in report["exports"]:
        direct_static = str(item["variant"]).endswith("folder-watch-static")
        interactions = [
            "open the standalone export and confirm its scientific canvas is nonblank",
            "confirm pan/zoom or the primary image interaction still responds",
        ]
        if direct_static:
            interactions.append(
                "confirm no green Watching badge is present in this stopped static snapshot"
            )
        pages.append(
            {
                "widget": item["widget"],
                "variant": item["variant"],
                "file": Path(str(item["path"])).name,
                "url_path": Path(str(item["path"])).name,
                "expected_folder_watch_state": (
                    "hidden" if direct_static else "not asserted"
                ),
                "required_interactions": interactions,
            }
        )
    plan = {
        "version": 1,
        "description": (
            "Browser-drive the tiny static exports after the Python report has "
            "already proven direct live watcher transitions. Static pages must "
            "never claim that their stopped Python watcher is still Watching."
        ),
        "artifact_dir": str(artifact_dir),
        "pages": pages,
    }
    (artifact_dir / "browser-plan.json").write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )


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
    timeline_rows = []
    for step in report["steps"]:
        for point in step.get("timeline") or []:
            timeline_rows.append(
                "<tr>"
                f"<td>{html.escape(str(step['name']))}</td>"
                f"<td>{html.escape(str(point.get('event', '')))}</td>"
                f"<td>{html.escape(str(point.get('elapsed_ms', '')))}</td>"
                f"<td>{html.escape(str(point.get('state', '')))}</td>"
                f"<td>{html.escape(str(point.get('count', '')))}</td>"
                f"<td>{html.escape(str(point.get('compare_page_loading', '')))}</td>"
                f"<td>{html.escape(str(point.get('compare_page_loaded_count', '')))}</td>"
                f"<td>{html.escape(str(point.get('compare_panel_indices', '')))}</td>"
                f"<td>{html.escape(str(point.get('detail', '')))}</td>"
                "</tr>"
            )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Folder-watching direct-viewer smoke</title>
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
  <h1>Direct viewer folder-watch smoke: {'PASS' if report['passed'] else 'FAIL'}</h1>
  <p>This lightweight report covers the direct
  public <code>Show2D.from_folder</code>, <code>Show3D.from_folder</code>, and
  <code>Show4DSTEM.from_folder</code> lifecycles. Direct viewers retain one Python
  object and widget model through probation, stable arrival, update, and stop.
  The Show4DSTEM direct step uses the native GPU loader and waits for fresh
  visible-page pixels before accepting green Watching. Heavy real-data and GPU
  performance remain separate local-only signoffs.</p>
  <h2>Review Exports</h2>
  <ul>{''.join(links)}</ul>
  <h2>Direct Viewer Lifecycle Timeline</h2>
  <p>The state text—not color alone—is authoritative. Open
  <a href="browser-plan.json">browser-plan.json</a> to drive the stopped static
  exports; they must not retain a green Watching claim.</p>
  <table>
    <thead><tr><th>Viewer</th><th>Event</th><th>ms</th><th>State</th><th>Count</th><th>Page loading</th><th>Fresh panels</th><th>Panel indices</th><th>Detail</th></tr></thead>
    <tbody>{''.join(timeline_rows) if timeline_rows else '<tr><td colspan="9">No direct-viewer timeline.</td></tr>'}</tbody>
  </table>
  <h2>Machine-readable report</h2>
  <p><a href="report.json">report.json</a> · <a href="browser-plan.json">browser-plan.json</a></p>
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

    artifact_dir = args.artifact_dir or Path(tempfile.mkdtemp(prefix="quantem-widget-folder-watch-"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = artifact_dir.resolve()

    started = time.perf_counter()
    steps = [
        _run_direct_image_live_smoke(
            artifact_dir,
            viewer_class=Show2D,
            widget_name="show2d",
            count_attr="n_images",
        ),
        _run_direct_image_live_smoke(
            artifact_dir,
            viewer_class=Show3D,
            widget_name="show3d",
            count_attr="n_slices",
        ),
        _run_direct_show4dstem_live_smoke(artifact_dir),
    ]
    export_rows = [
        row
        for step in steps
        for row in step.get("export_rows", [])
    ]
    report = {
        "artifact_dir": str(artifact_dir),
        "created_at_unix": int(time.time()),
        "seconds": round(time.perf_counter() - started, 3),
        "passed": all(step["passed"] for step in steps),
        "steps": steps,
        "exports": export_rows,
        "total_size_mb": round(
            sum(float(row["size_mb"]) for row in export_rows),
            3,
        ),
    }
    _write_browser_plan(artifact_dir, report)
    _write_report(artifact_dir, report)
    print(json.dumps(report, indent=2))
    print(f"Direct viewer folder-watch smoke report: {artifact_dir / 'index.html'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
