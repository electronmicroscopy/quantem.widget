#!/usr/bin/env python3
"""Run the local widget stress-test workflow from notebook to browser report.

This script is intentionally local-data aware but not private-path hard-coded.
It looks for stress fixtures under ``QUANTEM_WIDGET_STRESS_DATA`` or the
repository-local ``.widget-stress-data`` symlink, executes a generated notebook
that builds a widget and exports standalone HTML, then drives the exported HTML
with ``widget_external_html_profile.py``.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import nbformat
from nbclient import NotebookClient


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_fixture_root() -> Path:
    env_path = os.environ.get("QUANTEM_WIDGET_STRESS_DATA")
    if env_path:
        return Path(env_path).expanduser()
    return _repo_root() / ".widget-stress-data"


def _default_artifact_dir() -> Path:
    env_path = os.environ.get("QUANTEM_WIDGET_STRESS_REPORT_DIR")
    if env_path:
        return Path(env_path).expanduser() / _timestamp()
    return Path("/tmp") / "quantem-widget-stress-tests" / _timestamp()


def _escape(value: object) -> str:
    return html.escape(str(value))


def _load_manifest(fixture_root: Path, fixture: str) -> tuple[Path, dict]:
    fixture_dir = (fixture_root / fixture).resolve()
    manifest_path = fixture_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing stress fixture manifest: {manifest_path}\n"
            "Create or link local data first. Expected convention: "
            "QUANTEM_WIDGET_STRESS_DATA/<fixture>/manifest.json or "
            ".widget-stress-data/<fixture>/manifest.json."
        )
    return fixture_dir, json.loads(manifest_path.read_text(encoding="utf-8"))


def _primary_array_path(manifest: dict) -> Path:
    primary = manifest.get("primary_array", {})
    path = Path(primary.get("path") or primary.get("repo_path") or "")
    if not path.exists():
        raise SystemExit(f"Primary stress array does not exist: {path}")
    return path


def _disk_warning(path: Path, estimated_bytes: int) -> str | None:
    usage = shutil.disk_usage(path)
    if usage.free < estimated_bytes * 2:
        return (
            f"Low free space for export/report directory {path}: "
            f"{usage.free / 1024**3:.1f} GiB free, estimated raw payload "
            f"{estimated_bytes / 1024**3:.1f} GiB."
        )
    return None


def _write_notebook(
    notebook_path: Path,
    *,
    repo_root: Path,
    array_path: Path,
    html_path: Path,
    panel_limit: int | None,
    ncols: int,
    cmap: str,
    title: str,
    mode: str,
    encoding: str,
    downsample: int | None,
    display_bin: str,
) -> None:
    downsample_code = "None" if downsample is None else str(downsample)
    panel_limit_code = "None" if panel_limit is None else str(panel_limit)
    display_bin_code = "\"auto\"" if display_bin == "auto" else str(int(display_bin))
    source = f"""
from pathlib import Path
import json
import sys
import time

import numpy as np

repo_root = Path(r"{repo_root}")
sys.path.insert(0, str(repo_root / "src"))

from quantem.widget import Show2D

array_path = Path(r"{array_path}")
html_path = Path(r"{html_path}")
panel_limit = {panel_limit_code}
downsample = {downsample_code}
skip_initial_frame_pack = {mode == "folder"!r}
skip_initial_stats = {mode == "folder"!r}
preserve_input_dtype_for_export = {mode == "folder" and encoding == "uint8"!r}

started = time.perf_counter()
stack = np.load(array_path, mmap_mode="r")
if stack.ndim != 3:
    raise ValueError(f"Expected a 3D stack, got {{stack.shape}}")
data = stack[:panel_limit] if panel_limit is not None else stack
labels = [f"gold {{i:02d}} · {{data.shape[1]}}x{{data.shape[2]}}" for i in range(data.shape[0])]
widget_title = f"{title} · exact {{data.shape[1]}}x{{data.shape[2]}}"

widget = Show2D(
    data,
    labels=labels,
    title=widget_title,
    ncols={ncols},
    cmap={cmap!r},
    display_bin={display_bin_code},
    verbose=False,
    _skip_initial_frame_pack=skip_initial_frame_pack,
    _skip_initial_stats=skip_initial_stats,
    _preserve_input_dtype_for_export=preserve_input_dtype_for_export,
)
html_path.parent.mkdir(parents=True, exist_ok=True)
export_started = time.perf_counter()
exported = widget.export_html(
    html_path,
    title=widget_title,
    mode={mode!r},
    encoding={encoding!r},
    downsample=downsample,
)
summary = {{
    "array_path": str(array_path),
    "array_shape": list(stack.shape),
    "data_shape": list(data.shape),
    "data_dtype": str(data.dtype),
    "display_shape": [int(widget.height), int(widget.width)],
    "display_bin_factor": int(widget._display_bin_factor),
    "html_path": str(exported),
    "html_size_bytes": Path(exported).stat().st_size,
    "load_and_build_seconds": round(export_started - started, 3),
    "export_seconds": round(time.perf_counter() - export_started, 3),
}}
print(json.dumps(summary, indent=2))
"""
    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# Widget stress notebook\n\n"
                "Generated by `scripts/widget_stress_tests.py`. This notebook "
                "loads a local ignored stress fixture, builds a `Show2D`, and "
                "exports a standalone HTML file. It does not require a live "
                "backend after export."
            ),
            nbformat.v4.new_code_cell(source.strip() + "\n"),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    notebook_path.write_text(nbformat.writes(nb), encoding="utf-8")


def _execute_notebook(notebook_path: Path, executed_path: Path, timeout: int, *, keep_widget_state: bool) -> dict:
    nb = nbformat.read(notebook_path, as_version=4)
    started = time.perf_counter()
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(notebook_path.parent)}})
    client.execute()
    if not keep_widget_state:
        nb.metadata.pop("widgets", None)
    executed_path.write_text(nbformat.writes(nb), encoding="utf-8")
    outputs = []
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                outputs.append(output.get("text", ""))
    return {
        "path": str(executed_path),
        "seconds": round(time.perf_counter() - started, 3),
        "stream_outputs": outputs,
    }


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_browser_profile(args: argparse.Namespace, html_path: Path, profile_dir: Path) -> dict:
    server = None
    if args.mode == "folder":
        port = _free_local_port()
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                str(port),
                "--bind",
                "127.0.0.1",
                "--directory",
                str(html_path.parent),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.5)
        url = f"http://127.0.0.1:{port}/{quote(html_path.name)}"
    else:
        url = html_path.resolve().as_uri()
    command = [
        sys.executable,
        str(_repo_root() / "scripts" / "widget_external_html_profile.py"),
        "--url",
        url,
        "--artifact-dir",
        str(profile_dir),
        "--min-fps",
        str(args.min_fps),
        "--min-canvases",
        str(args.min_canvases),
        "--viewport-width",
        str(args.viewport_width),
        "--viewport-height",
        str(args.viewport_height),
        "--settle-ms",
        str(args.settle_ms),
        "--fps-sample-ms",
        str(args.fps_sample_ms),
        "--hover-canvas-limit",
        str(args.hover_canvas_limit),
    ]
    if args.headed:
        command.append("--headed")
    if args.no_hover_stress:
        command.append("--no-hover-stress")
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    additions = [str(_repo_root() / "src"), str(_repo_root() / "scripts")]
    env["PYTHONPATH"] = os.pathsep.join(additions + ([existing] if existing else []))
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=_repo_root(), env=env, text=True, capture_output=True)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
    metrics_path = profile_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return {
        "command": command,
        "returncode": result.returncode,
        "seconds": round(time.perf_counter() - started, 3),
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "metrics_path": str(metrics_path) if metrics_path.exists() else None,
        "report_path": str(profile_dir / "index.html"),
        "metrics": metrics,
    }


def _write_report(
    artifact_dir: Path,
    *,
    args: argparse.Namespace,
    fixture_dir: Path,
    manifest: dict,
    array_path: Path,
    notebook_path: Path,
    executed_notebook_path: Path,
    html_path: Path,
    notebook_result: dict,
    browser_result: dict | None,
    warnings: list[str],
) -> dict:
    profile = browser_result or {}
    metrics = profile.get("metrics", {})
    checks = [
        ("Fixture manifest exists", True, str(fixture_dir / "manifest.json")),
        ("Primary array exists", array_path.exists(), str(array_path)),
        ("Notebook executed", executed_notebook_path.exists(), f"{notebook_result.get('seconds')} s"),
        ("HTML export exists", html_path.exists(), f"{html_path.stat().st_size / 1024**2:.1f} MiB" if html_path.exists() else ""),
    ]
    if browser_result is None:
        checks.append(("UI browser profile", False, "Skipped by --no-browser-profile"))
    else:
        checks.append(("UI browser profile", bool(metrics.get("passed")), profile.get("report_path", "")))

    passed = all(row[1] for row in checks)
    checks_html = "".join(
        f"<tr><td>{_escape(name)}</td><td class=\"{'pass' if ok else 'fail'}\">{'PASS' if ok else 'FAIL'}</td>"
        f"<td>{_escape(detail)}</td></tr>"
        for name, ok, detail in checks
    )
    warnings_html = "".join(f"<li>{_escape(warning)}</li>" for warning in warnings) or "<li>None</li>"

    images_html = ""
    if metrics:
        image_items = []
        for key, label in [
            ("initial_screenshot", "Initial exported widget"),
            ("initial_primary_canvas", "Primary canvas crop"),
        ]:
            shot = metrics.get(key, {})
            rel = shot.get("rel")
            if rel:
                image_items.append((label, f"profile/{rel}"))
        for step in metrics.get("steps", []):
            shot = step.get("screenshot") or {}
            rel = shot.get("rel")
            if rel:
                label = str(step.get("name", "step")).replace("_", " ")
                image_items.append((label, f"profile/{rel}"))
        if image_items:
            figures = "".join(
                f"<figure><img src=\"{_escape(rel)}\"><figcaption>{_escape(label)}</figcaption></figure>"
                for label, rel in image_items
            )
            images_html = f"<section class=\"card\"><h2>Visual evidence</h2><div class=\"shots\">{figures}</div></section>"

    artifacts = [
        ("Generated notebook", notebook_path),
        ("Executed notebook", executed_notebook_path),
        ("Standalone HTML export", html_path),
        ("UI profile report", Path(profile["report_path"]) if profile.get("report_path") else None),
        ("UI profile metrics", Path(profile["metrics_path"]) if profile.get("metrics_path") else None),
        ("Run summary JSON", artifact_dir / "run-summary.json"),
    ]
    artifacts_html = "".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(
            _escape(name),
            f'<a href="{_escape(path.relative_to(artifact_dir).as_posix())}">{_escape(path.name)}</a>'
            if path and path.exists() and path.is_relative_to(artifact_dir)
            else _escape(path or "not created"),
        )
        for name, path in artifacts
    )
    data_rows = [
        ("Fixture", args.fixture),
        ("Fixture directory", fixture_dir),
        ("Array path", array_path),
        ("Manifest name", manifest.get("name")),
        ("Manifest note", manifest.get("scientific_note")),
        ("Panel limit", args.panel_limit or "all"),
        ("Colormap", args.cmap),
        ("Export mode", args.mode),
        ("Export encoding", args.encoding),
        ("Export downsample", args.downsample or "none"),
        ("Display bin", args.display_bin),
        ("No backend after export", True),
    ]
    data_html = "".join(f"<tr><th>{_escape(k)}</th><td>{_escape(v)}</td></tr>" for k, v in data_rows)
    zoom_pan_step = next(
        (step for step in metrics.get("steps", []) if step.get("name") == "show2d_zoom_pan"),
        None,
    )
    profile_summary = {
        "passed": metrics.get("passed"),
        "load_to_ready_s": metrics.get("load_to_ready_s"),
        "initial_browser_raf_fps": metrics.get("initial_fps"),
        "final_browser_raf_fps": metrics.get("final_fps"),
        "initial_canvas_count": metrics.get("initial_canvas_count"),
        "show2d_zoom_pan_actual_user_timing": {
            "wheel_events": (zoom_pan_step or {}).get("wheel_event_probe"),
            "wheel_paint": (zoom_pan_step or {}).get("wheel_perf_debug"),
            "drag_events": (zoom_pan_step or {}).get("drag_event_probe"),
            "drag_paint": (zoom_pan_step or {}).get("drag_perf_debug"),
            "changed_panel_count": (zoom_pan_step or {}).get("changed_panel_count"),
        },
        "hover_sweep": next(
            (step for step in metrics.get("steps", []) if step.get("name") == "hover_sweep"),
            None,
        ),
        "errors": metrics.get("errors"),
    }
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Widget stress-test tutorial report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2328; }}
    h1, h2 {{ text-transform: none; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    .pass {{ color: #167a3b; font-weight: 700; }}
    .fail {{ color: #b42318; font-weight: 700; }}
    .card {{ border: 2px solid #ff2bbd; border-radius: 8px; padding: 16px; margin: 16px 0; background: white; box-shadow: 0 0 0 3px rgba(255, 43, 189, 0.10); }}
    .card h2 {{ display: inline-block; margin-top: 0; padding: 3px 8px; background: #ff2bbd; color: white; border-radius: 4px; }}
    .shots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 14px; }}
    figure {{ margin: 0; }}
    img {{ max-width: 100%; border: 3px solid #ff2bbd; background: #2a001e; box-sizing: border-box; }}
    figcaption {{ display: inline-block; margin-top: 3px; padding: 2px 6px; color: white; background: #ff2bbd; border-radius: 4px; font-size: 12px; }}
    pre, code {{ background: #f6f8fa; border-radius: 4px; }}
    pre {{ padding: 12px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Widget stress-test tutorial report</h1>
  <p>This run executed a generated notebook, wrote a standalone Show2D HTML export, and drove that exported HTML in Chromium. The exported HTML is static and does not require a live backend after creation.</p>
  <section class="card"><h2>Result</h2><p class="{'pass' if passed else 'fail'}">{'PASS' if passed else 'FAIL'}</p></section>
  <section class="card"><h2>Checks</h2><table><tr><th>Check</th><th>Status</th><th>Evidence</th></tr>{checks_html}</table></section>
  <section class="card"><h2>Data provenance</h2><table>{data_html}</table></section>
  <section class="card"><h2>UI profile summary</h2><pre>{_escape(json.dumps(profile_summary, indent=2))}</pre></section>
  <section class="card"><h2>Warnings</h2><ul>{warnings_html}</ul></section>
  {images_html}
  <section class="card"><h2>Artifacts</h2><table><tr><th>Artifact</th><th>Path</th></tr>{artifacts_html}</table></section>
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(doc, encoding="utf-8")
    summary = {
        "passed": passed,
        "artifact_dir": str(artifact_dir),
        "report_path": str(artifact_dir / "index.html"),
        "fixture_dir": str(fixture_dir),
        "array_path": str(array_path),
        "notebook_path": str(notebook_path),
        "executed_notebook_path": str(executed_notebook_path),
        "html_path": str(html_path),
        "notebook_result": notebook_result,
        "browser_result": browser_result,
        "warnings": warnings,
        "checks": [{"name": name, "passed": ok, "detail": detail} for name, ok, detail in checks],
    }
    (artifact_dir / "run-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> dict:
    repo_root = _repo_root()
    fixture_root = Path(args.fixture_root).expanduser().resolve()
    fixture_dir, manifest = _load_manifest(fixture_root, args.fixture)
    array_path = _primary_array_path(manifest)
    primary = manifest.get("primary_array", {})
    raw_nbytes = int(primary.get("nbytes") or 0)

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    warning = _disk_warning(artifact_dir, raw_nbytes)
    if warning:
        warnings.append(warning)
        print(f"WARNING: {warning}", file=sys.stderr)

    notebook_path = artifact_dir / "widget_stress_main.ipynb"
    executed_notebook_path = artifact_dir / "widget_stress_main.executed.ipynb"
    html_path = artifact_dir / "show2d_stress_export.html"
    title = args.title or f"Show2D stress: {args.fixture}"

    _write_notebook(
        notebook_path,
        repo_root=repo_root,
        array_path=array_path,
        html_path=html_path,
        panel_limit=args.panel_limit,
        ncols=args.ncols,
        cmap=args.cmap,
        title=title,
        mode=args.mode,
        encoding=args.encoding,
        downsample=args.downsample,
        display_bin=args.display_bin,
    )
    notebook_result = _execute_notebook(
        notebook_path,
        executed_notebook_path,
        args.notebook_timeout_s,
        keep_widget_state=args.keep_notebook_widget_state,
    )

    browser_result = None
    if args.browser_profile:
        browser_result = _run_browser_profile(args, html_path, artifact_dir / "profile")

    return _write_report(
        artifact_dir,
        args=args,
        fixture_dir=fixture_dir,
        manifest=manifest,
        array_path=array_path,
        notebook_path=notebook_path,
        executed_notebook_path=executed_notebook_path,
        html_path=html_path,
        notebook_result=notebook_result,
        browser_result=browser_result,
        warnings=warnings,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", default=str(_default_fixture_root()))
    parser.add_argument("--fixture", default="gold_4k")
    parser.add_argument("--artifact-dir", default=str(_default_artifact_dir()))
    parser.add_argument("--panel-limit", type=int, default=4, help="Panels to load/export; use 24 for the full gold_4k stress source.")
    parser.add_argument("--ncols", type=int, default=2)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--title")
    parser.add_argument("--mode", default="single", choices=["single", "folder"])
    parser.add_argument("--encoding", default="uint8", choices=["uint8", "full"])
    parser.add_argument("--downsample", type=int)
    parser.add_argument("--display-bin", default="1", help="Show2D display_bin; default 1 keeps native pixels, auto permits preview binning.")
    parser.add_argument("--notebook-timeout-s", type=int, default=900)
    parser.add_argument("--keep-notebook-widget-state", action="store_true", help="Keep large widget state in the executed notebook.")
    parser.add_argument("--browser-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-fps", type=float, default=20.0)
    parser.add_argument("--min-canvases", type=int, default=1)
    parser.add_argument("--fps-sample-ms", type=int, default=1000)
    parser.add_argument("--settle-ms", type=int, default=1500)
    parser.add_argument("--hover-canvas-limit", type=int, default=80)
    parser.add_argument("--no-hover-stress", action="store_true")
    parser.add_argument("--headed", action="store_true", help="Run the browser profile in a visible Chromium window.")
    parser.add_argument("--viewport-width", type=int, default=1800)
    parser.add_argument("--viewport-height", type=int, default=1200)
    args = parser.parse_args()

    summary = run(args)
    print(f"Widget stress tutorial report: {summary['report_path']}")
    print(f"Standalone HTML export: {summary['html_path']}")
    if summary["passed"]:
        print("PASS")
        return 0
    print("FAIL")
    for check in summary["checks"]:
        if not check["passed"]:
            print(f"- {check['name']}: {check['detail']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
