"""``quantem`` command-line interface: a folder (or file) of images or 4D-STEM
masters becomes a rendered, standalone HTML viewer in one command, no notebook.

    quantem show ./frames/                # PNG/TIFF folder -> Show3D scrub HTML
    quantem show scan.png                 # single image    -> Show2D HTML
    quantem show4dstem ./masters/ --backend webgpu --html --bin 1
                                            # *_master.h5 -> HDF5-backed WebGPU folder
    quantem showptycho scan_master.h5     # raw 4D-STEM    -> ShowPtycho WebGPU folder
    quantem showptycho ./masters/         # master folder  -> ShowPtycho project
    quantem showdiffraction pattern.npy   # diffraction     -> analyzed ShowDiffraction HTML
    quantem html tutorial.ipynb           # run a notebook  -> standalone shareable HTML

The CLI only orchestrates existing pieces: ``io.read_image`` / ``read_image_stack``
for images, ``quantem.gpu.io.discover`` + ``quantem.gpu.io.load(det_bin=...)``
for 4D-STEM and
ptychography review, the ``Show2D`` / ``Show3D`` / ``Show4DSTEM`` / ``ShowPtycho``
widgets, and each widget's export helpers. Show4DSTEM WebGPU HTML keeps the
compressed HDF5 family on disk and lets Chrome range-fetch/decompress H5 chunks
instead of preprocessing every frame before the viewer opens.
"""
import argparse
import email.utils
import http.server
import json
import mimetypes
import os
import pathlib
import posixpath
import re
import shutil
import socketserver
import sys
import threading
import urllib.parse
import webbrowser

from quantem.widget.showptycho_collection import (
    is_showptycho_collection as _is_showptycho_collection,
    showptycho_collection_folder as _showptycho_collection_folder,
    write_showptycho_collection as _write_showptycho_collection,
)

# Single image -> Show2D, a folder of frames -> Show3D, a folder of differently
# sized images -> a Show2D gallery. These are the formats read_image understands.
IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".dm3", ".dm4", ".emd", ".npy"}
MASTER_PATTERN = "*_master.h5"
SHOWPTYCHO_MASTER_PATTERNS = ("*_master.h5", "*_master_wrapper.h5")
SHOWPTYCHO_FOLDER_FORMAT = "quantem.showptycho.webgpu.folder"
DEFAULT_PTYCHO_SEMIANGLE_MRAD = 30.0
DEFAULT_PTYCHO_SCAN_SAMPLING_A = 0.5
DEFAULT_PTYCHO_VOLTAGE_KV = 300.0
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")
_RANGE_FALLBACK_CHUNK_BYTES = 16 * 1024 * 1024


def _package_source_state(
    *,
    version: str,
    module_file: str,
) -> dict[str, str | bool | None]:
    """Return reproducible package identity without recording local paths."""

    import subprocess

    source = pathlib.Path(module_file).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()),
        None,
    )
    if repository is None:
        return {"version": version, "commit": None, "dirty": None}

    commit_result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    status_result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    commit = (
        commit_result.stdout.strip()
        if commit_result.returncode == 0
        else None
    )
    dirty = (
        bool(status_result.stdout.strip())
        if status_result.returncode == 0
        else None
    )
    return {"version": version, "commit": commit, "dirty": dirty}


def _showptycho_software_provenance() -> dict[str, dict[str, str | bool | None]]:
    """Return the core, compute, and UI versions that produced an SSB fit."""

    import quantem
    import quantem.gpu
    import quantem.widget

    return {
        "quantem": _package_source_state(
            version=quantem.__version__,
            module_file=quantem.__file__,
        ),
        "quantem.gpu": _package_source_state(
            version=quantem.gpu.__version__,
            module_file=quantem.gpu.__file__,
        ),
        "quantem.widget": _package_source_state(
            version=quantem.widget.__version__,
            module_file=quantem.widget.__file__,
        ),
    }


def _anonymize_showptycho_payload(value):
    """Remove local acquisition identity while preserving scientific provenance."""

    source_keys = {"source_file", "source_path", "master_path", "source_stem"}
    if isinstance(value, dict):
        return {
            key: (
                "redacted_local_source"
                if key in source_keys and item is not None
                else _anonymize_showptycho_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_anonymize_showptycho_payload(item) for item in value]
    return value


def _showptycho_reused_fit_payload(
    fit_record: pathlib.Path,
    *,
    anonymize: bool,
) -> dict[str, object]:
    """Return a reused fit record with current export provenance."""

    payload = json.loads(fit_record.read_text(encoding="utf-8"))
    if anonymize:
        payload = _anonymize_showptycho_payload(payload)
    payload["export_software"] = _showptycho_software_provenance()
    return payload


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``quantem`` console script. Parse args, dispatch to the
    ``show`` subcommand, return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="quantem",
        description="Render images or 4D-STEM masters as a viewer (HTML, or a live notebook for 4D).",
    )
    sub = parser.add_subparsers(dest="command")
    # `show` auto-detects; show2d/show3d/show4dstem force the widget so the command
    # reads exactly like the widget it opens. All share the same options + engine.
    forced = {
        "show": "auto",
        "show2d": "2d",
        "show3d": "3d",
        "show4dstem": "4dstem",
        "showptycho": "showptycho",
    }
    helps = {
        "show": "Auto-detect PATH(s) and render the matching viewer.",
        "show2d": "Render an image (or a folder of images) as Show2D.",
        "show3d": "Render a folder of frames as a Show3D scrub.",
        "show4dstem": "Render 4D-STEM master(s) as Show4DSTEM (live notebook, or --html).",
        "showptycho": "Open or build a ShowPtycho project from 4D-STEM masters.",
    }
    for name in ("show", "show2d", "show3d", "show4dstem"):
        _add_show_args(sub.add_parser(name, help=helps[name]))
    _add_showptycho_args(
        sub.add_parser("showptycho", help=helps["showptycho"])
    )
    # `html` is a different shape (one .ipynb in, one HTML out), so it gets its own
    # parser rather than the shared show* options.
    _add_html_args(sub.add_parser(
        "html", help="Execute a notebook and export it to a standalone, offline shareable HTML."))
    # `github` shrinks a widget notebook to a form GitHub can display: drop the heavy offline
    # widget-state, keep the auto-snapshot widget render (re-encoded JPEG) + print outputs.
    _add_github_args(sub.add_parser(
        "github", help="Make a widget notebook GitHub-displayable (strip offline state, snapshots to JPEG)."))
    _add_showfolder_args(sub.add_parser(
        "showfolder", help="Browse a microscopy folder with ShowFolder: inventory, thumbnails, and selection state."))
    _add_showdiffraction_args(sub.add_parser(
        "showdiffraction",
        help="Analyze a diffraction pattern with ShowDiffraction: auto rings, phase, standalone HTML."))
    args = parser.parse_args(argv)
    try:
        if args.command == "html":
            return _render_html(args)
        if args.command == "github":
            return _prepare_github(args)
        if args.command == "showfolder":
            return _showfolder(args)
        if args.command == "showdiffraction":
            return _showdiffraction(args)
        if args.command not in forced:
            parser.print_help()
            return 0
        args.widget = forced[args.command]
        return _show(args)
    except (FileNotFoundError, ValueError) as err:
        print(f"quantem: {err}", file=sys.stderr)
        return 1


def _add_html_args(parser: argparse.ArgumentParser) -> None:
    """Attach options for the ``html`` subcommand."""
    parser.add_argument("path", help="The .ipynb to render.")
    parser.add_argument("--out", default=None,
                        help="Output path or directory for the HTML. Default: ~/Downloads.")
    parser.add_argument("--no-execute", action="store_true",
                        help="Export the notebook's already-saved outputs without re-running it.")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-cell execution timeout in seconds (default 600).")
    parser.add_argument("--no-open", action="store_true", help="Write the HTML but do not open it.")


def _add_showfolder_args(parser: argparse.ArgumentParser) -> None:
    """Attach options for the ``showfolder`` subcommand."""
    parser.add_argument("folder", help="Folder of microscopy files to browse.")
    parser.add_argument("--html", default=None, help="Execute the ShowFolder notebook and write this HTML file.")
    parser.add_argument("--notebook", default=None, help="Write this ShowFolder notebook path.")
    parser.add_argument("--thumb", type=int, default=512, help="Thumbnail size for the HAADF/STEM gallery.")
    parser.add_argument("--glob", default="*.emd", help="Glob within the folder (default '*.emd').")
    parser.add_argument("--title", default=None, help="ShowFolder title.")
    parser.add_argument("--group-by", default="session", choices=("session", "fov", "none"),
                        help="ShowFolder layout grouping mode (default 'session').")
    parser.add_argument("--group-view", default="stack", choices=("stack", "gallery"),
                        help="Grouped image display mode (default 'stack').")
    parser.add_argument("--timeout", type=int, default=900, help="Notebook execution timeout in seconds.")
    parser.add_argument("--no-open", action="store_true", help="Write outputs but do not launch/open them.")


def _fmt_bytes(value: int) -> str:
    """Format a byte count for concise CLI status output."""

    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1000 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1000.0
    return f"{size:.1f} TB"


def _showfolder(args: argparse.Namespace) -> int:
    """Generate a microscopy folder browser notebook, optionally render it to HTML."""
    import shutil
    import subprocess
    from quantem.widget.showfolder_core import write_showfolder_notebook

    folder = pathlib.Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"not a folder: {folder}")
    if shutil.which("jupyter") is None and args.html:
        raise ValueError("jupyter not found; install jupyter to render survey HTML")

    html_out = pathlib.Path(args.html).expanduser().resolve() if args.html else None
    if args.notebook:
        notebook = pathlib.Path(args.notebook).expanduser().resolve()
    elif html_out is not None:
        notebook = html_out.with_suffix(".ipynb")
    else:
        notebook = _default_out_dir() / f"{folder.name}_showfolder.ipynb"

    write_showfolder_notebook(
        folder,
        notebook,
        glob=args.glob,
        thumb=args.thumb,
        title=args.title,
        group_by=args.group_by,
        group_view=args.group_view,
    )
    print(f"notebook: {notebook}")

    if html_out is None:
        _launch_notebook(notebook, no_open=args.no_open)
        return 0

    html_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "--execute",
        str(notebook),
        "--output-dir",
        str(html_out.parent),
        "--output",
        html_out.stem,
        f"--ExecutePreprocessor.timeout={args.timeout}",
        # explicit store_widget_state: ambient nbconvert config must not strip
        # the ShowFolder hydration state out of the share artifact
        "--ExecutePreprocessor.store_widget_state=True",
    ]
    print(f"executing + rendering ShowFolder -> {html_out}")
    if subprocess.run(cmd).returncode != 0:
        raise ValueError("ShowFolder nbconvert failed (see output above)")
    size_mb = html_out.stat().st_size / 1e6
    print(f"HTML: {size_mb:.1f} MB")
    _open_html(html_out, serve=False, no_open=args.no_open)
    return 0


def _add_showdiffraction_args(parser: argparse.ArgumentParser) -> None:
    """Attach options for the ``showdiffraction`` subcommand."""
    parser.add_argument("path", nargs="?", default=None,
                        help="A diffraction pattern: .npy (2D, or a 3D stack), .emd/.dm3/.dm4, or a raster image.")
    parser.add_argument("--demo", action="store_true",
                        help="Analyze the real Fe3O4 nanoparticle SAED tutorial pattern instead of a file.")
    parser.add_argument("--phase", default=None,
                        help="Library phase for calibration and hkl indexing, e.g. Au or Fe3O4.")
    parser.add_argument("--no-auto", action="store_true",
                        help="Skip the Auto pipeline (center, rings, calibration, fit, indexing).")
    parser.add_argument("--max-rings", type=int, default=8,
                        help="Ring detection cap for the Auto pipeline (default 8).")
    parser.add_argument("--exclude-radius", type=float, default=None,
                        help="Ignore rings inside this radius in px, e.g. an amorphous halo.")
    parser.add_argument("--k-pixel-size", type=float, default=None,
                        help="Known detector calibration in 1/Å per pixel (kept even with --phase).")
    parser.add_argument("--out", default=None,
                        help="Output path or directory for the HTML. Default: ~/Downloads.")
    parser.add_argument("--title", default=None, help="Viewer page title.")
    parser.add_argument("--no-open", action="store_true", help="Write the HTML but do not open it.")
    parser.add_argument("--serve", action="store_true",
                        help="Open via a local HTTP server (tunnelable URL).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose progress.")


def _showdiffraction(args: argparse.Namespace) -> int:
    """Render one diffraction pattern as an analyzed ShowDiffraction HTML.

    The Auto pipeline (center, rings, profile fit) runs by default; ``--phase``
    adds calibration and hkl indexing, and an explicit ``--k-pixel-size`` is kept
    so the phase then only indexes. ``--demo`` analyzes the bundled Fe3O4 SAED
    with tutorial defaults."""
    import numpy as np
    from quantem.widget import ShowDiffraction

    if args.demo:
        if args.path is not None:
            raise ValueError("give either a pattern file or --demo, not both")
        from quantem.widget.data import tutorials
        try:
            data = tutorials.showdiffraction_fe3o4(verbose=args.verbose)
        except (OSError, ImportError) as err:
            raise ValueError(f"tutorial data unavailable ({err}); check access to huggingface.co")
        src = pathlib.Path("fe3o4_saed")
        args.title = args.title or "Fe3O4 nanoparticle SAED"
        args.phase = args.phase or "Fe3O4"
        if args.exclude_radius is None:
            args.exclude_radius = 70.0  # amorphous carbon support halo
    elif args.path is None:
        raise ValueError("provide a diffraction pattern file, or use --demo")
    else:
        src = pathlib.Path(args.path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"path does not exist: {src}")
        if not src.is_file():
            raise ValueError(f"not a file: {src}")
        if src.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(
                f"unsupported file type {src.suffix!r}; expected .npy, .emd, .dm3/.dm4, or a raster image")
        try:
            data = np.load(src) if src.suffix.lower() == ".npy" else _load_2d(src)
        except ImportError as err:
            raise ValueError(f"reading {src.suffix} needs an optional dependency ({err})")
        except (OSError, EOFError) as err:
            raise ValueError(f"could not read {src.name}: {err}")

    phase = None
    if args.phase is not None:
        from quantem.widget import library_phase
        phase = library_phase(args.phase)

    widget = ShowDiffraction(
        data,
        k_pixel_size=args.k_pixel_size,
        title=args.title or src.stem,
        offline=True,
        verbose=args.verbose,
    )
    if not args.no_auto:
        if phase is not None and args.k_pixel_size is not None:
            # explicit calibration wins; the phase indexes only
            widget.run_auto(max_rings=args.max_rings, exclude_radius=args.exclude_radius)
            if widget.rings:
                widget.index_rings(phase)
        else:
            widget.run_auto(phase, max_rings=args.max_rings, exclude_radius=args.exclude_radius)
        if widget.analysis_status:
            print(widget.analysis_status)
    if args.phase is not None:
        widget.phase_name = args.phase
    if not args.no_auto:
        widget.summary()

    out = _out_path(args.out, src, suffix="showdiffraction")
    widget.export_html(out, title=args.title or src.stem)
    _open_html(out, serve=args.serve, no_open=args.no_open)
    return 0


def _render_html(args: argparse.Namespace) -> int:
    """Execute a notebook and export it to a standalone, shareable HTML.

    Wraps ``jupyter nbconvert --to html [--execute]``: a finished notebook becomes a
    kernel-less HTML page whose saved widget state is hydrated by the ipywidgets HTML
    manager. Show2D / Show3D / Show3DSlices / ShowEDS controls remain interactive in the browser,
    but changes are browser-local and do not write back to the notebook or HTML file.
    The live ``.ipynb`` stays the editable surface; this is the share artifact.
    ``--no-execute`` exports the saved outputs as-is, which is what a notebook's own
    in-cell ``!jupyter nbconvert`` does after a run."""
    import shutil
    import subprocess
    notebook = pathlib.Path(args.path).expanduser().resolve()
    if not notebook.exists():
        raise FileNotFoundError(f"notebook not found: {notebook}")
    if notebook.suffix.lower() != ".ipynb":
        raise ValueError(f"expected a .ipynb, got {notebook.suffix!r}")
    if shutil.which("jupyter") is None:
        raise ValueError("jupyter not found; install jupyter to render a notebook")
    out_dir = _out_dir(args.out)
    cmd = ["jupyter", "nbconvert", "--to", "html", str(notebook),
           "--output-dir", str(out_dir), "--output", notebook.stem]
    if not args.no_execute:
        # explicit store_widget_state: a jupyter_nbconvert_config.py that disables it
        # for heavy notebook sweeps would otherwise silently strip the hydration state
        # this share artifact exists to carry
        cmd += ["--execute", f"--ExecutePreprocessor.timeout={args.timeout}",
                "--ExecutePreprocessor.store_widget_state=True"]
    print(f"{'rendering' if args.no_execute else 'executing + rendering'} {notebook.name} -> HTML")
    if subprocess.run(cmd).returncode != 0:
        raise ValueError("nbconvert failed (see output above)")
    out = out_dir / f"{notebook.stem}.html"
    # Report the file size so the audience knows how heavy the share artifact is: baked
    # widget images make these big (a Show2D gallery can be >100 MB), which matters for
    # email limits and browser open time.
    size_mb = out.stat().st_size / 1e6
    note = "large - widget images baked in; trim panels if emailing" if size_mb > 50 else "self-contained, offline"
    print(f"HTML: {size_mb:.1f} MB ({note})")
    print(f"  {out}")
    _open_html(out, serve=False, no_open=args.no_open)
    return 0


# ---------------------------------------------------------------------------
_WIDGET_CELL = ("Show2D(", "Show3D(", "Show4DSTEM(", "Show3DSlices(", "ShowEDS(")


def _add_github_args(parser: argparse.ArgumentParser) -> None:
    """Attach options for the ``github`` subcommand."""
    parser.add_argument("path", help="The .ipynb to make GitHub-displayable (edited in place).")
    parser.add_argument("--no-execute", action="store_true",
                        help="Use the notebook's existing outputs instead of re-running it.")
    parser.add_argument("--quality", type=int, default=92,
                        help="JPEG quality for the embedded renders (default 92).")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-cell execution timeout in seconds (default 600).")


def _strip_state(nb: dict) -> None:
    """Drop the heavy offline live-widget manager-state + the dead widget-view output refs."""
    nb.get("metadata", {}).pop("widgets", None)
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            (out.get("data") or {}).pop("application/vnd.jupyter.widget-view+json", None)


def _embed_jpeg(cell: dict, png_or_jpeg: bytes, quality: int) -> bool:
    """Replace a cell's visual output with one JPEG.

    Widget outputs usually have only ``application/vnd.jupyter.widget-view+json``,
    not an existing ``image/*`` slot.  For GitHub display we must add a normal
    image output before stripping the widget MIME bundle.
    """
    import base64
    from io import BytesIO
    from PIL import Image
    img = Image.open(BytesIO(png_or_jpeg)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    done = False
    for out in cell.get("outputs", []):
        data = out.get("data")
        if data and (
            any(k.startswith("image/") for k in data)
            or "application/vnd.jupyter.widget-view+json" in data
        ):
            for k in [k for k in data if k.startswith("image/")]:
                del data[k]
            data["image/jpeg"] = b64
            out.setdefault("metadata", {})
            done = True
            break
    if not done:
        cell.setdefault("outputs", []).append({
            "output_type": "display_data",
            "metadata": {},
            "data": {"image/jpeg": b64},
        })
        done = True
    return done


def _cell_has_image_output(cell: dict) -> bool:
    """Return true when a notebook cell already has a GitHub-renderable image."""
    for out in cell.get("outputs", []):
        data = out.get("data") or {}
        if any(key.startswith("image/") for key in data):
            return True
    return False


def _cell_has_widget_view_output(cell: dict) -> bool:
    """Return true when a notebook cell still depends on live widget MIME output."""
    for out in cell.get("outputs", []):
        data = out.get("data") or {}
        if "application/vnd.jupyter.widget-view+json" in data:
            return True
    return False


def _capture_full_ui(html: pathlib.Path, n_expected: int) -> list[bytes]:
    """Screenshot each widget's FULL UI (toolbar + toggles + panels + histograms) from the
    rendered live-widget HTML, deterministically, via Playwright on the real GPU. The widget
    UI is React+MUI+WebGPU, so a browser engine is required; Playwright manages the lifecycle
    (waits for mount + paint) and ``locator.screenshot`` grabs each widget element exactly."""
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("DISPLAY", ":1")
    from playwright.sync_api import sync_playwright
    shots: list[bytes] = []
    launch_kwargs = {}
    for candidate in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        executable = shutil.which(candidate)
        if executable:
            launch_kwargs["executable_path"] = executable
            print(f"  browser executable: {executable}")
            break
    with sync_playwright() as play:
        browser = play.chromium.launch(headless=False, args=[
            "--enable-unsafe-webgpu", "--use-angle=vulkan", "--enable-features=Vulkan",
            "--ignore-gpu-blocklist", "--disable-gpu-sandbox", "--no-sandbox"], **launch_kwargs)
        page = browser.new_page(viewport={"width": 1300, "height": 2400}, device_scale_factor=2)
        page.goto(html.as_uri(), wait_until="load", timeout=90000)
        page.wait_for_timeout(13000)  # anywidget mount + WebGPU paint
        arch = page.evaluate("async()=>{const a=await navigator.gpu?.requestAdapter();"
                             "return a?(a.info?.architecture||'?'):'none';}")
        print(f"  GPU adapter: {arch}")
        if arch == "swiftshader":
            print("  warning: WebGPU reported SwiftShader; continuing because GitHub snapshots only need pixels")
        outs = page.locator(".jp-OutputArea-output")
        for i in range(outs.count()):
            el = outs.nth(i)
            if el.locator("canvas").count() > 0:
                el.scroll_into_view_if_needed()
                page.wait_for_timeout(700)
                shots.append(el.screenshot())
        browser.close()
    if len(shots) != n_expected:
        print(f"  warning: captured {len(shots)} widget UIs for {n_expected} widget cells")
    return shots


def _prepare_github(args: argparse.Namespace) -> int:
    """Make a widget notebook GitHub/VS-Code-displayable: embed a screenshot of each widget's
    FULL live UI (toolbar+toggles+panels) - because the whole reason to use the widget over
    ``show_2d`` is the UI, so the static render shows it (captured deterministically via
    Playwright on the real GPU). Drops the offline ``metadata.widgets`` state (tens of MB;
    GitHub won't render it and can't run widgets anyway) and JPEG-encodes each render (noisy
    science images compress ~10x). Keeps every other output (matplotlib PNGs, prints).

    The interactive widget still comes from re-running the notebook or ``quantem html``. Needs
    Playwright + a real GPU (NVIDIA Vulkan ICD + a display); errors clearly if unavailable."""
    import json
    import shutil
    import subprocess
    notebook = pathlib.Path(args.path).expanduser().resolve()
    if not notebook.exists():
        raise FileNotFoundError(f"notebook not found: {notebook}")
    if notebook.suffix.lower() != ".ipynb":
        raise ValueError(f"expected a .ipynb, got {notebook.suffix!r}")
    if shutil.which("jupyter") is None:
        raise ValueError("jupyter not found; install jupyter")
    before = notebook.stat().st_size
    if not args.no_execute:
        print(f"executing {notebook.name} ...")
        if subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                           str(notebook), f"--ExecutePreprocessor.timeout={args.timeout}"]).returncode != 0:
            raise ValueError("nbconvert --execute failed (see output above)")
    nb = json.loads(notebook.read_text())
    widget_cells = [c for c in nb["cells"]
                    if c["cell_type"] == "code" and any(w in "".join(c["source"]) for w in _WIDGET_CELL)]
    capture_cells = [
        c for c in widget_cells
        if not _cell_has_image_output(c)
    ]
    if capture_cells:
        try:
            html = notebook.with_suffix(".fullui.html")
            subprocess.run(["jupyter", "nbconvert", "--to", "html", str(notebook),
                            "--output-dir", str(notebook.parent), "--output", notebook.stem + ".fullui"],
                           check=True)
            print(f"capturing {len(capture_cells)} widget UI(s) on the GPU ...")
            shots = _capture_full_ui(html, len(capture_cells))
            html.unlink(missing_ok=True)
            if len(shots) != len(capture_cells):
                raise ValueError(
                    f"captured {len(shots)} widget UI screenshot(s) for {len(capture_cells)} widget cell(s)"
                )
            for cell, png in zip(capture_cells, shots):
                _embed_jpeg(cell, png, args.quality)
            mode = f"{len(shots)} full-UI screenshots"
        except (ImportError, RuntimeError, OSError) as err:
            raise ValueError(
                "full-UI capture needs Playwright + a real GPU (NVIDIA Vulkan ICD + a display): "
                f"{err}") from err
    elif widget_cells:
        mode = f"{len(widget_cells)} existing image output(s)"
    else:
        mode = "no widget cells - state stripped only"
    _strip_state(nb)
    notebook.write_text(json.dumps(nb, indent=1))
    after = notebook.stat().st_size
    print(f"github-ready: {notebook.name}  {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB"
          f"  ({mode}, JPEG q{args.quality}, offline state stripped)")
    if after > 5e6:
        print("  warning: still > 5 MB - GitHub may not render. Lower --quality or the widget's size=.")
    return 0


# ---------------------------------------------------------------------------
def _add_viewer_path_args(
    parser: argparse.ArgumentParser,
    *,
    path_help: str | None = None,
    out_help: str = "Output file or directory.",
    include_serve: bool = True,
) -> None:
    """Attach path, output, and launch options shared by viewer commands."""

    parser.add_argument("path", nargs="+",
                        help=path_help or (
                            "An image, a folder of images, a 4D-STEM master, "
                            "a folder of masters, or several master files."
                        ))
    parser.add_argument("--out", default=None, help=out_help)
    parser.add_argument("--no-open", action="store_true", help="Write the file(s) but do not launch anything.")
    if include_serve:
        parser.add_argument("--serve", action="store_true",
                            help="Open via a local HTTP server even for self-contained files (tunnelable URL).")
    parser.add_argument("--port", type=int, default=None,
                        help="Folder exports: port for the local HTTP server (default: auto-pick).")
    parser.add_argument("--bind", default="127.0.0.1",
                        help="Folder exports: bind address for the local HTTP server (default: 127.0.0.1).")
    parser.add_argument("--title", default=None, help="Viewer page title.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose progress.")


def _add_show_args(parser: argparse.ArgumentParser) -> None:
    """Attach image and 4D-STEM options to a ``show*`` parser."""

    _add_viewer_path_args(parser)
    parser.add_argument("--bin", type=int, default=None, dest="det_bin",
                        help=(
                            "Detector binning factor. Show4DSTEM and ShowPtycho "
                            "default to 1, meaning full detector sampling."
                        ))
    parser.add_argument("--count", type=int, default=None,
                        help="Show4DSTEM: require and load this many compatible masters from the input.")
    parser.add_argument("--combined", action="store_true",
                        help="Many 4D masters -> one 5D HTML viewer (with --html; needs a local serve).")
    parser.add_argument("--quantized", action="store_true",
                        help="Image widgets: uint8 pack (smaller file).")
    parser.add_argument("--html", action="store_true",
                        help="4D-STEM: export a standalone offline-WebGPU HTML instead of a live notebook.")
    parser.add_argument("--watch", action="store_true",
                        help="Folder: write a live ShowFolder-watched notebook that appends new files.")
    parser.add_argument("--watch-interval", type=float, default=2.0,
                        help="Polling interval in seconds for --watch live folders (default 2).")
    parser.add_argument("--gpus", "--devices", dest="gpus", default=None,
                        help="4D-STEM CUDA devices, e.g. 0 or 0,1. Default preserves loader device.")
    parser.add_argument("--page-budget", default="auto",
                        help="4D-STEM --watch: resident dataset cache, e.g. auto, 1, 2, or none (default auto).")
    parser.add_argument("--dtype", default="u8", choices=("u8", "uint8", "u16", "uint16", "float32"),
                        help="4D-STEM browse dtype (default u8).")
    parser.add_argument("--scan-size", type=int, default=None,
                        help="4D-STEM --watch: only include masters with this square scan size.")
    parser.add_argument("--backend", default="auto",
                        choices=("auto", "cuda", "mps", "webgpu"),
                        help="Show4DSTEM backend. Use webgpu with --html.")


def _add_showptycho_args(parser: argparse.ArgumentParser) -> None:
    """Attach the focused ShowPtycho project options."""

    _add_viewer_path_args(
        parser,
        path_help=(
            "One or more *_master.h5 files, a folder containing masters, "
            "or an existing ShowPtycho project."
        ),
        out_help=(
            "Project directory. Default: ~/QuantEM/showptycho/<acquisition>."
        ),
        include_serve=False,
    )
    parser.set_defaults(det_bin=1)
    parser.add_argument("--dtype", default="u8", choices=("u8", "uint8", "u16", "uint16", "float32"),
                        help="ShowPtycho and Show4DSTEM browse dtype (default u8).")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write under SOURCE/quantem/showptycho instead of the user-owned default.",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help=(
            "ShowPtycho: redact the local acquisition name/path from saved "
            "calibration and optimization provenance."
        ),
    )
    parser.add_argument("--backend", default="auto", choices=("auto", "cuda", "mps"),
                        help="ShowPtycho master generation: HDF5 load backend (default auto).")
    parser.add_argument("--calibration", default="auto",
                        help=(
                            "ShowPtycho master generation: calibration JSON, "
                            "'auto' to search nearby QuantEM results, or 'none'."
                        ))
    parser.add_argument(
        "--trials",
        type=int,
        default=200,
        help=(
            "ShowPtycho master generation: run this many exact GPU Optuna "
            "trials through quantem.gpu.SSB before export (default 200). "
            "Set 0 only to reuse a resolved calibration without fitting."
        ),
    )
    parser.add_argument(
        "--refinement",
        default="nelder-mead",
        choices=("nelder-mead", "none"),
        help=(
            "Refinement after --trials: Nelder-Mead or none "
            "(default nelder-mead)."
        ),
    )
    parser.add_argument("--semiangle", "--semiangle-mrad", dest="semiangle_mrad",
                        type=float, default=None,
                        help="ShowPtycho master generation: probe semi-angle in mrad.")
    parser.add_argument("--scan-sampling", "--scan-sampling-A", dest="scan_sampling_A",
                        type=float, default=None,
                        help="ShowPtycho master generation: scan pixel size in Angstrom.")
    parser.add_argument("--det-sampling", "--det-sampling-mrad-px", dest="det_sampling_mrad_px",
                        type=float, default=None,
                        help="ShowPtycho master generation: detector angular sampling in mrad/pixel.")
    parser.add_argument("--voltage-kv", dest="voltage_kv", type=float, default=None,
                        help="ShowPtycho master generation: accelerating voltage in kV.")
    parser.add_argument("--drag-bf", type=float, default=1.0,
                        help="ShowPtycho browser BF fraction/count: 1.0 is full BF, 0.3 is 30%%, values greater than 1 are explicit BF-pixel counts (default 1.0).")
    parser.add_argument("--size", type=int, default=800,
                        help="ShowPtycho initial panel size in pixels (default 800).")
    parser.add_argument("--fft", action="store_true",
                        help="ShowPtycho opens with the FFT panel visible.")
    parser.add_argument("--force", action="store_true",
                        help="ShowPtycho master generation: rebuild an existing output folder.")


# ---------------------------------------------------------------------------
def _show(args: argparse.Namespace) -> int:
    """Resolve the content, render the matching widget(s), open the result.

    Images render to a standalone HTML (light, shareable, opens with a double-click).
    4D-STEM renders to a live Jupyter notebook by default (full real-time WebGPU, no
    large file); ``--html`` instead exports the self-contained offline-WebGPU HTML.
    One path can be a file or a folder; several paths are taken as a list of 4D-STEM
    masters and become one 5D multi-tilt viewer."""
    paths = [pathlib.Path(p).expanduser().resolve() for p in args.path]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("path does not exist: " + ", ".join(missing))
    if args.widget == "showptycho" and len(paths) > 1:
        masters = []
        for path in paths:
            if not path.is_file() or not _is_showptycho_master_name(path.name):
                raise ValueError(
                    "quantem showptycho accepts master HDF5 files or one folder "
                    "containing master HDF5 files."
                )
            masters.append(path)
        folder = _render_showptycho_collection(masters, args)
        _serve_showptycho_folder(
            folder, bind=args.bind, port=args.port, no_open=args.no_open
        )
        return 0
    # Several explicit paths: a list of masters -> one 5D viewer (multi-tilt), or a
    # set of image files -> a gallery. A single path falls through to _detect.
    if len(paths) > 1:
        if args.watch:
            raise ValueError("--watch requires one folder path, not multiple explicit paths.")
        if args.widget != "4dstem" and all(p.suffix.lower() in IMAGE_EXTS for p in paths):
            out = _render_gallery(paths, "gallery", args)
            _open_html(out, serve=args.serve, no_open=args.no_open)
            return 0
        masters = _select_show4dstem_masters([str(p) for p in paths], args)
        return _do_4dstem(masters, f"{len(masters)}_datasets", args, source_path=None)
    path = paths[0]
    kind = _detect(path, args.widget)
    if kind == "showptycho-master":
        folder = _render_showptycho_collection([path], args)
        _serve_showptycho_folder(
            folder, bind=args.bind, port=args.port, no_open=args.no_open
        )
        return 0
    if kind == "showptycho-masters":
        folder = _render_showptycho_collection(
            _showptycho_master_candidates(path), args, source_dir=path
        )
        _serve_showptycho_folder(
            folder, bind=args.bind, port=args.port, no_open=args.no_open
        )
        return 0
    if kind == "showptycho-collection":
        folder = _showptycho_collection_folder(path)
        _serve_showptycho_folder(folder, bind=args.bind, port=args.port, no_open=args.no_open)
        return 0
    if kind == "showptycho":
        folder = _showptycho_folder(path)
        _serve_showptycho_folder(folder, bind=args.bind, port=args.port, no_open=args.no_open)
        return 0
    if kind == "4dstem":
        if args.watch:
            if args.html:
                raise ValueError("--watch writes a live notebook; omit --html.")
            if not path.is_dir():
                raise ValueError("--watch requires a folder path containing *_master.h5 files.")
        from quantem.gpu.io import discover

        masters = [str(path)] if path.is_file() else discover(
            str(path), verbose=args.verbose
        )
        masters = [
            master
            for master in masters
            if not _is_show4dstem_generated_master_link(pathlib.Path(master))
        ]
        if not masters:
            raise ValueError(f"no *_master.h5 found in {path}")
        masters = _select_show4dstem_masters(masters, args)
        label = pathlib.Path(masters[0]).stem.replace("_master", "") if path.is_file() else path.name
        if args.watch:
            notebook = _render_4dstem_watch_notebook(path, label, args)
            _launch_notebook(notebook, no_open=args.no_open)
            return 0
        return _do_4dstem(masters, label, args, source_path=path)
    if args.watch:
        if args.html:
            raise ValueError("--watch writes a live notebook; omit --html.")
        if kind != "images" or not path.is_dir():
            raise ValueError("--watch requires one folder path.")
        widget = "show3d" if args.widget == "3d" else "show2d"
        notebook = _render_image_watch_notebook(path, path.name, args, widget=widget)
        _launch_notebook(notebook, no_open=args.no_open)
        return 0
    out = _render_images(path, kind, args)
    _open_html(out, serve=args.serve, no_open=args.no_open)
    return 0


def _do_4dstem(
    masters: list[str],
    label: str,
    args: argparse.Namespace,
    *,
    source_path: pathlib.Path | None = None,
) -> int:
    """Dispatch 4D-STEM master(s) to either a live notebook (default) or an offline
    HTML (``--html``), then launch/open it. One master loads alone; many load stacked
    into a 5D viewer with a dataset slider (the multi-tilt case)."""
    args.det_bin = _effective_det_bin(args, default=1)
    backend = _normalise_show4dstem_backend(args.backend)
    if args.html:
        if backend == "webgpu":
            output = _render_4dstem_webgpu_h5(masters, label, args)
            _open_show4dstem_command(output.parent / "Show4DSTEM.command", no_open=args.no_open)
            return 0
        outputs = _render_4dstem(masters, label, args)
        _open_html(outputs[0], serve=args.serve or args.combined, no_open=args.no_open)
        if len(outputs) > 1:
            print(f"wrote {len(outputs)} HTML files to {outputs[0].parent}")
        return 0
    if backend == "webgpu":
        raise ValueError("Show4DSTEM --backend webgpu writes browser HTML; add --html.")
    notebook = _render_4dstem_notebook(masters, label, args, source_path=source_path)
    _launch_notebook(notebook, no_open=args.no_open)
    return 0


def _select_show4dstem_masters(masters: list[str], args: argparse.Namespace) -> list[str]:
    """Apply Show4DSTEM ``--count`` as an exact compatible-master request."""

    count = getattr(args, "count", None)
    if count is None:
        return list(masters)
    count = int(count)
    if count < 1:
        raise ValueError(f"--count must be a positive integer, got {count}")
    if len(masters) < count:
        raise ValueError(
            f"--count {count} requested but only {len(masters)} master(s) were found."
        )
    return list(masters[:count])


def _is_show4dstem_generated_master_link(path: pathlib.Path) -> bool:
    """Return whether *path* is a symlink created by a CLI WebGPU export.

    Rerunning ``quantem show4dstem <source-folder> --out <source-folder>``
    must select the original masters, not the anonymous ``dataset_XX``
    links placed in its owned ``*_show4dstem_webgpu`` output folder.
    """
    return path.is_symlink() and path.parent.name.endswith("_show4dstem_webgpu")


_TILT_COORDINATE_RE = re.compile(
    r"_(?P<x>[+-]?\d+(?:\.\d+)?)x_(?P<y>[+-]?\d+(?:\.\d+)?)y_"
)


def _show4dstem_dataset_label(master: str, index: int) -> str:
    """Return a useful dataset label, including coordinates when available."""
    match = _TILT_COORDINATE_RE.search(pathlib.Path(master).name)
    if match is None:
        return f"Dataset {index + 1}"

    def _format(value: str) -> str:
        text = f"{float(value):+.2f}".rstrip("0").rstrip(".")
        return text if "." in text else f"{text}.0"

    return f"Tilt ({_format(match['x'])}, {_format(match['y'])})"


def _normalise_show4dstem_backend(value: str | None) -> str | None:
    """Return the Show4DSTEM backend token used by generated notebooks/exports."""

    token = str(value or "auto").strip().lower()
    if token in {"", "auto"}:
        return None
    if token == "webgpu":
        return "webgpu"
    if token in {"cuda", "mps"}:
        return token
    raise ValueError(f"unsupported Show4DSTEM backend {value!r}")


def _detect(path: pathlib.Path, forced: str) -> str:
    """Return the content kind: image, images, 4dstem, showptycho, or ptycho master.

    A single file is always 'image' unless it is a master or 4D is forced (a lone
    file can't be a 3D scrub). For a folder: the command's forced widget wins, else a
    ``*_master.h5`` makes it 4D and image files make it 'images'. The stack-vs-gallery
    split for 'images' is decided later from the forced widget."""
    if forced == "showptycho":
        if _is_showptycho_collection(path):
            return "showptycho-collection"
        if _is_showptycho_folder_export(path):
            return "showptycho"
        if path.is_file() and _is_showptycho_master_name(path.name):
            return "showptycho-master"
        if path.is_dir() and _showptycho_master_candidates(path):
            return "showptycho-masters"
        _showptycho_folder(path)
        return "showptycho"
    if _is_showptycho_folder_export(path):
        return "showptycho"
    if _is_showptycho_collection(path):
        return "showptycho-collection"
    if path.is_file():
        if forced == "4dstem" or path.name.endswith("_master.h5"):
            return "4dstem"
        if forced in ("2d", "3d", "auto") and path.suffix.lower() in IMAGE_EXTS:
            return "image"
        raise ValueError(f"unsupported file type {path.suffix!r}; expected an image or *_master.h5")
    if forced == "4dstem":
        return "4dstem"
    if forced in ("2d", "3d"):
        return "images"
    masters = sorted(path.glob(MASTER_PATTERN))
    if masters:
        return "4dstem"
    if any(p.suffix.lower() in IMAGE_EXTS for p in path.iterdir()):
        return "images"
    raise ValueError(f"no images or *_master.h5 found in {path}")


def _effective_det_bin(args: argparse.Namespace, *, default: int) -> int:
    """Return a positive detector bin, using the command-specific default."""

    raw = args.det_bin
    value = default if raw is None else raw
    try:
        det_bin = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--bin must be a positive integer, got {value!r}") from exc
    if det_bin < 1:
        raise ValueError(f"--bin must be a positive integer, got {det_bin}")
    return det_bin


def _showptycho_decode_dtype(args: argparse.Namespace) -> str:
    """Return the explicit browser decode dtype for ShowPtycho source HDF5."""

    raw = str(args.dtype).lower()
    if raw in {"u8", "uint8"}:
        return "uint8"
    if raw in {"u16", "uint16"}:
        return "uint16"
    if raw == "float32":
        return "float32"
    raise ValueError(f"ShowPtycho --dtype must be u8, u16, or float32; got {raw!r}")




def _is_showptycho_master_name(name: str) -> bool:
    """Return whether ``name`` is a supported ShowPtycho source master."""

    return name.endswith("_master.h5") or name.endswith("_master_wrapper.h5")


def _showptycho_master_candidates(path: pathlib.Path) -> list[pathlib.Path]:
    """Find supported ShowPtycho source masters in a folder."""

    masters: set[pathlib.Path] = set()
    for pattern in SHOWPTYCHO_MASTER_PATTERNS:
        masters.update(path.glob(pattern))
    return sorted(masters)


def _showptycho_folder(path: pathlib.Path) -> pathlib.Path:
    """Return the folder for a ShowPtycho WebGPU export, or raise with next steps."""
    folder = path.parent if path.is_file() and path.name == "index.html" else path
    if not folder.is_dir():
        raise FileNotFoundError(f"not a ShowPtycho folder export: {path}")
    index = folder / "index.html"
    manifest = _showptycho_manifest_path(folder)
    if not index.is_file():
        raise ValueError(f"ShowPtycho folder export is missing index.html: {folder}")
    if manifest is None:
        raise ValueError(
            f"ShowPtycho folder export is missing snapshots/manifest.json: {folder}"
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"ShowPtycho manifest is not valid JSON: {manifest}") from exc
    format_name = str(payload.get("format", ""))
    if not format_name.startswith(SHOWPTYCHO_FOLDER_FORMAT):
        raise ValueError(
            "not a ShowPtycho WebGPU folder export; expected manifest format "
            f"{SHOWPTYCHO_FOLDER_FORMAT!r}, got {format_name!r}"
        )
    return folder


def _is_showptycho_folder_export(path: pathlib.Path) -> bool:
    """Best-effort detector used by ``quantem show`` before normal image/4D routing."""
    folder = path.parent if path.is_file() and path.name == "index.html" else path
    manifest = _showptycho_manifest_path(folder)
    if manifest is None:
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return str(payload.get("format", "")).startswith(SHOWPTYCHO_FOLDER_FORMAT)


def _showptycho_manifest_path(folder: pathlib.Path) -> pathlib.Path | None:
    """Return the canonical ShowPtycho snapshot manifest, when present."""

    candidate = folder / "snapshots" / "manifest.json"
    return candidate if candidate.is_file() else None


def _showptycho_source_stem(master: pathlib.Path) -> str:
    """Return the microscope source stem without the Arina ``_master`` suffix."""

    stem = master.stem
    if stem.endswith("_master_wrapper"):
        return stem[:-len("_master_wrapper")]
    return stem[:-len("_master")] if stem.endswith("_master") else stem


def _default_showptycho_root() -> pathlib.Path:
    """Return the user-owned root for ShowPtycho projects."""

    return pathlib.Path.home() / "QuantEM" / "showptycho"


def _showptycho_target(path: pathlib.Path) -> pathlib.Path:
    """Validate and normalize one explicit ShowPtycho output directory."""

    target = path.expanduser().resolve()
    if target.suffix.lower() in {".html", ".htm", ".ipynb"}:
        raise ValueError(
            "ShowPtycho writes a project folder; pass --out as a directory."
        )
    target.mkdir(parents=True, exist_ok=True)
    return target


def _showptycho_collection_output_dir(
    masters: list[pathlib.Path],
    args: argparse.Namespace,
    source_dir: pathlib.Path | None,
) -> pathlib.Path:
    """Resolve one catalog root without copying the acquisition masters."""

    if args.out and args.in_place:
        raise ValueError("choose either --out or --in-place, not both")
    if args.out:
        return _showptycho_target(pathlib.Path(args.out))

    parents = {master.parent for master in masters}
    if source_dir is not None:
        project_name = source_dir.name
    elif len(masters) == 1:
        project_name = _showptycho_source_stem(masters[0])
    elif len(parents) == 1:
        project_name = next(iter(parents)).name
    else:
        project_name = "showptycho-collection"

    if args.in_place:
        if len(parents) != 1:
            raise ValueError(
                "--in-place requires all master files to share one source folder"
            )
        return _showptycho_target(
            next(iter(parents)) / "quantem" / "showptycho"
        )
    return _showptycho_target(_default_showptycho_root() / project_name)


def _write_show4dstem_viewer(
    master: pathlib.Path,
    folder: pathlib.Path,
    *,
    label: str,
    target_stem: str | None = None,
) -> pathlib.Path:
    """Write a direct browser Show4DSTEM viewer linked to the raw HDF5 family."""

    calibration_path = folder / "snapshots" / "cal.json"
    if not calibration_path.is_file():
        raise ValueError(
            f"ShowPtycho export is missing its calibration: {calibration_path}"
        )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    scan_shape = (
        calibration.get("scan_region", {}).get("shape")
        or calibration.get("phase_shape")
    )
    detector_shape = calibration.get("detector_shape")
    if not (
        isinstance(scan_shape, list)
        and len(scan_shape) == 2
        and isinstance(detector_shape, list)
        and len(detector_shape) == 2
    ):
        raise ValueError(
            f"ShowPtycho calibration is missing scan or detector shape: {calibration_path}"
        )

    from quantem.widget.show4dstem_webgpu_export import (
        export_show4dstem_hdf5_viewer,
    )

    return export_show4dstem_hdf5_viewer(
        master,
        folder / "show4dstem",
        scan_shape=(int(scan_shape[0]), int(scan_shape[1])),
        detector_shape=(int(detector_shape[0]), int(detector_shape[1])),
        title=f"{label} Show4DSTEM",
        target_stem=target_stem,
    )


def _render_showptycho_collection(
    masters: list[pathlib.Path],
    args: argparse.Namespace,
    *,
    source_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Build one project catalog and one isolated result folder per master."""

    if not masters:
        raise ValueError("no ShowPtycho master HDF5 files were found")
    root = _showptycho_collection_output_dir(masters, args, source_dir)
    datasets: list[dict[str, object]] = []
    used_names: set[str] = set()
    for index, master in enumerate(masters, start=1):
        stem = _showptycho_source_stem(master)
        base_name = f"dataset-{index:03d}" if args.anonymize else stem
        name = base_name
        suffix = 2
        while name in used_names:
            name = f"{base_name}-{suffix}"
            suffix += 1
        used_names.add(name)
        result = _render_showptycho_master(master, args, out_dir=root / name)
        raw_viewer = _write_show4dstem_viewer(
            master,
            result,
            label=(f"Dataset {index:03d}" if args.anonymize else stem),
            target_stem=(f"dataset-{index:03d}" if args.anonymize else None),
        )
        fit_path = result / "ssb_fit.json"
        fit = (
            json.loads(fit_path.read_text(encoding="utf-8"))
            if fit_path.is_file()
            else {}
        )
        entry: dict[str, object] = {
            "label": f"Dataset {index:03d}" if args.anonymize else stem,
            "viewer": f"{name}/index.html",
            "show4dstem": str(raw_viewer.relative_to(root)),
            "calibration": f"{name}/snapshots/cal.json",
            "fit": f"{name}/ssb_fit.json" if fit_path.is_file() else None,
            "backend": fit.get("backend"),
            "num_bf": fit.get("num_bf"),
            "loss": fit.get("loss"),
        }
        if not args.anonymize:
            entry["source"] = master.name
        datasets.append(entry)
    if args.anonymize:
        title = "ShowPtycho collection"
    elif args.title:
        title = args.title
    elif len(masters) == 1:
        title = f"{_showptycho_source_stem(masters[0])} ShowPtycho"
    elif source_dir is not None:
        title = f"{source_dir.name} ShowPtycho"
    else:
        title = "ShowPtycho collection"
    _write_showptycho_collection(root, datasets, title=title)
    return root


def _showptycho_calibration_search_paths(master: pathlib.Path) -> list[pathlib.Path]:
    """Nearby calibration files created by the QuantEM ptychography workflow."""

    stem = _showptycho_source_stem(master)
    root = master.parent
    return [
        root / "quantem" / "showptycho" / stem / "calibration.json",
    ]


def _showptycho_master_calib_paths(master: pathlib.Path) -> list[pathlib.Path]:
    """Nearby run metadata files that can fill microscope geometry."""

    stem = _showptycho_source_stem(master)
    root = master.parent
    return [
        root / "quantem" / "showptycho" / stem / "master_calib.json",
        root / "quantem" / "showptycho" / stem / "_runspec.json",
    ]


def _mapping_matches_showptycho_source(
    payload: dict,
    *,
    master: pathlib.Path,
) -> bool:
    """Return whether a calibration object belongs to ``master``."""

    stem = _showptycho_source_stem(master)
    candidates = {
        str(payload.get("source_stem", "")),
        str(payload.get("label", "")),
    }
    source_file = payload.get("source_file") or payload.get("master_path")
    if source_file:
        candidates.add(_showptycho_source_stem(pathlib.Path(str(source_file))))
    return stem in candidates or master.stem in candidates


def _calibration_from_showptycho_mapping(payload: dict):
    """Convert a calibration mapping to ``PtychoCalibration``."""

    from quantem.widget.showptycho import PtychoCalibration

    aberrations = {
        str(k): float(v) for k, v in (payload.get("aberrations") or {}).items()
    }
    if "C10" not in aberrations and "C10_nm" in payload:
        aberrations["C10"] = float(payload["C10_nm"])
    if "C12" not in aberrations and "C12_nm" in payload:
        aberrations["C12"] = float(payload["C12_nm"])
    if "phi12" not in aberrations and "phi12_rad" in payload:
        aberrations["phi12"] = float(payload["phi12_rad"])
    if "phi12" not in aberrations and "phi12_deg" in payload:
        import math

        aberrations["phi12"] = math.radians(float(payload["phi12_deg"]))
    if "rotation_angle_deg" not in payload:
        raise ValueError("calibration is missing rotation_angle_deg")
    return PtychoCalibration(
        rotation_angle_deg=float(payload["rotation_angle_deg"]),
        aberrations=aberrations,
        higher_order={
            str(k): float(v) for k, v in (payload.get("higher_order") or {}).items()
        },
        flip_phase=bool(payload.get("flip_phase", False)),
        voltage_kV=payload.get("voltage_kV") or payload.get("voltage_kv"),
        semiangle_mrad=payload.get("semiangle_mrad") or payload.get("semiangle"),
        scan_sampling_A=(
            payload.get("scan_sampling_A")
            or payload.get("scan_sampling")
            or payload.get("scan_sampling_A_per_px")
        ),
        det_sampling_mrad_px=(
            payload.get("det_sampling_mrad_px")
            or payload.get("det_sampling_mrad_per_px")
        ),
        loss=payload.get("loss"),
        source_file=payload.get("source_file"),
        source_stem=payload.get("source_stem"),
        label=payload.get("label"),
        notes=str(payload.get("notes", "")),
    )


def _load_showptycho_calibration(path: pathlib.Path, *, master: pathlib.Path):
    """Load a single calibration, choosing the matching entry from a list file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("calibrations"), list):
        payload = payload["calibrations"]
    if isinstance(payload, list):
        matches = [
            item for item in payload
            if isinstance(item, dict)
            and "rotation_angle_deg" in item
            and _mapping_matches_showptycho_source(item, master=master)
        ]
        if not matches:
            raise ValueError(f"no calibration in {path} matches {_showptycho_source_stem(master)}")

        def score(item: dict) -> float:
            loss = item.get("loss")
            try:
                return float(loss)
            except (TypeError, ValueError):
                return float("inf")

        payload = min(matches, key=score)
    if not isinstance(payload, dict):
        raise ValueError(f"calibration must be a JSON object or list: {path}")
    return _calibration_from_showptycho_mapping(payload)


def _resolve_showptycho_calibration(master: pathlib.Path, args: argparse.Namespace):
    """Resolve an explicit, automatic, or disabled ShowPtycho calibration."""

    raw = str(args.calibration).strip()
    if raw.lower() in {"none", "off", "false", "0"}:
        return None, None
    if raw.lower() != "auto":
        path = pathlib.Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"ShowPtycho calibration file not found: {path}")
        return _load_showptycho_calibration(path, master=master), path
    for path in _showptycho_calibration_search_paths(master):
        if not path.is_file():
            continue
        try:
            return _load_showptycho_calibration(path, master=master), path
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return None, None


def _read_showptycho_master_calib(master: pathlib.Path) -> dict:
    """Read optional ptychography run metadata next to a master."""

    for path in _showptycho_master_calib_paths(master):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _first_number(mapping: dict, *keys: str) -> float | None:
    """Return the first finite numeric value found under ``keys``."""

    for key in keys:
        value = mapping.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:
            return number
    return None


def _positive_cli_number(value: object, option: str) -> float | None:
    """Return a positive finite CLI number, or raise for an invalid explicit value."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{option} must be a positive finite number, got {value!r}") from exc
    if not (number == number and number > 0):
        raise ValueError(f"{option} must be a positive finite number, got {value!r}")
    return number


def _positive_optional_number(value: object) -> float | None:
    """Return ``value`` when it is positive and finite, otherwise ``None``."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number > 0 else None


def _first_positive_number(mapping: dict, *keys: str) -> float | None:
    """Return the first positive finite numeric value found under ``keys``."""

    number = _first_number(mapping, *keys)
    return number if number is not None and number > 0 else None


def _showptycho_default_warning(label: str, value: float, unit: str, option: str) -> str:
    """Format one visible quick-start geometry default warning."""

    return (
        f"using default ptychography {label} {value:g} {unit}; pass {option} "
        "or a calibration JSON for microscope-specific geometry"
    )


def _resolve_showptycho_geometry(
    args: argparse.Namespace,
    calibration,
    meta: dict,
) -> tuple[float, float, float, float | None, list[str]]:
    """Resolve ShowPtycho geometry from CLI args, calibration, metadata, or defaults."""

    semiangle = (
        _positive_cli_number(args.semiangle_mrad, "--semiangle")
        or _positive_optional_number(
            calibration.semiangle_mrad if calibration is not None else None
        )
        or _first_positive_number(meta, "semiangle_mrad", "semiangle")
    )
    scan_sampling = (
        _positive_cli_number(args.scan_sampling_A, "--scan-sampling")
        or _positive_optional_number(
            calibration.scan_sampling_A if calibration is not None else None
        )
        or _first_positive_number(
            meta, "scan_sampling_A", "scan_sampling", "scan_sampling_A_per_px"
        )
    )
    voltage = (
        _positive_cli_number(args.voltage_kv, "--voltage-kv")
        or _positive_optional_number(
            calibration.voltage_kV if calibration is not None else None
        )
        or _first_positive_number(meta, "voltage_kV", "voltage_kv", "voltage")
    )
    det_sampling = (
        _positive_cli_number(args.det_sampling_mrad_px, "--det-sampling")
        or _positive_optional_number(
            calibration.det_sampling_mrad_px if calibration is not None else None
        )
        or _first_positive_number(
            meta, "det_sampling_mrad_per_px", "det_sampling_mrad_px"
        )
    )

    warnings: list[str] = []
    if semiangle is None:
        semiangle = DEFAULT_PTYCHO_SEMIANGLE_MRAD
        warnings.append(_showptycho_default_warning(
            "semiangle", semiangle, "mrad", "--semiangle",
        ))
    if scan_sampling is None:
        scan_sampling = DEFAULT_PTYCHO_SCAN_SAMPLING_A
        warnings.append(_showptycho_default_warning(
            "scan sampling", scan_sampling, "A", "--scan-sampling",
        ))
    if voltage is None:
        voltage = DEFAULT_PTYCHO_VOLTAGE_KV
        warnings.append(_showptycho_default_warning(
            "voltage", voltage, "kV", "--voltage-kv",
        ))
    return semiangle, scan_sampling, voltage, det_sampling, warnings


def _render_showptycho_master(
    master: pathlib.Path,
    args: argparse.Namespace,
    *,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    """Build a ShowPtycho WebGPU folder from one raw ``*_master.h5`` file."""

    master = master.expanduser().resolve()
    out_dir = _showptycho_target(out_dir)
    if _is_showptycho_folder_export(out_dir) and not args.force:
        print(f"ShowPtycho folder already exists: {out_dir}")
        print("  using existing folder; pass --force to rebuild")
        return out_dir

    det_bin = _effective_det_bin(args, default=1)
    if det_bin != 1:
        raise ValueError(
            "ShowPtycho SSB requires native detector data; use --bin 1."
        )
    calibration, calibration_path = _resolve_showptycho_calibration(master, args)
    meta = _read_showptycho_master_calib(master)
    semiangle, scan_sampling, voltage, det_sampling, geometry_warnings = (
        _resolve_showptycho_geometry(args, calibration, meta)
    )

    from quantem.widget import ShowPtycho

    print(f"{master.name}: *_master.h5 -> ShowPtycho WebGPU folder")
    print(f"  output: {out_dir}")
    print(f"  detector bin: {det_bin} ({'native' if det_bin == 1 else 'downsampled'})")
    if calibration_path is not None:
        print(f"  calibration: {calibration_path}")
    else:
        print("  calibration: none; using resolved geometry and zero aberration start")
    print(
        f"  geometry: semiangle={semiangle:g} mrad, "
        f"scan_sampling={scan_sampling:g} A, voltage={voltage:g} kV"
    )
    if det_sampling is not None:
        print(f"  detector sampling: {det_sampling:g} mrad/pixel")
    for warning in geometry_warnings:
        print(f"warning: {warning}", file=sys.stderr)
    aberrations = dict(calibration.aberrations) if calibration is not None else None
    rotation = (
        float(calibration.rotation_angle_deg)
        if calibration is not None else 0.0
    )
    fit = None
    written_evidence_path = None
    bf_source_write_seconds = None
    trials = int(args.trials)
    if trials < 0:
        raise ValueError("--trials must be zero or a positive integer")

    from quantem.gpu import SSB

    workflow = SSB.open(
        str(master),
        backend=args.backend,
        dtype=_showptycho_decode_dtype(args),
        voltage_kV=float(voltage),
        semiangle_mrad=float(semiangle),
        scan_sampling_A=float(scan_sampling),
        det_sampling=float(det_sampling) if det_sampling is not None else None,
        aberrations=aberrations,
        rotation_angle_deg=rotation,
        bf_intensity_threshold=0.0,
        bf_radius=None,
        calibration=(
            str(calibration_path) if calibration_path is not None else None
        ),
        verbose=args.verbose,
    )
    print(
        f"  SSB source: {workflow.source_kind} "
        f"({workflow.source_storage_path})"
    )
    source_kind = workflow.source_kind
    source_path = workflow.source_storage_path
    source_dtype = workflow.source_dtype
    source_bytes = workflow.source_bytes
    source_load_seconds = workflow.source_load_seconds
    widget_input = workflow
    evidence_stem = out_dir.parent / f".{out_dir.name}_{os.getpid()}_bf_columns"
    written_evidence = workflow.export_brightfield(evidence_stem)
    if written_evidence is not None:
        written_evidence_path = pathlib.Path(written_evidence[0])
        bf_source_write_seconds = float(written_evidence[1])
        print(
            "  exact BF source prepared in "
            f"{bf_source_write_seconds:.2f}s ({written_evidence_path})"
        )

    if trials:
        from quantem.widget.showptycho import PtychoCalibration

        refine = None if args.refinement == "none" else str(args.refinement)
        print(
            f"  SSB fit: {trials} full-BF trials, "
            f"refine={refine or 'none'}, backend={workflow.backend.upper()}"
        )
        fit = workflow.fit(
            trials=trials,
            refinement=refine,
            verbose=args.verbose,
        )
        fit_det_sampling = (
            float(det_sampling)
            if det_sampling is not None
            else 2.0 * float(semiangle) / float(fit.detected_bf_radius)
        )
        aberrations = dict(fit.aberrations)
        rotation = float(fit.rotation_angle_deg)
        calibration = PtychoCalibration(
            rotation_angle_deg=fit.rotation_angle_deg,
            aberrations=aberrations,
            higher_order=(
                dict(calibration.higher_order)
                if calibration is not None else {}
            ),
            flip_phase=(
                bool(calibration.flip_phase)
                if calibration is not None else False
            ),
            voltage_kV=float(voltage),
            semiangle_mrad=float(semiangle),
            scan_sampling_A=float(scan_sampling),
            det_sampling_mrad_px=fit_det_sampling,
            loss=float(fit.loss) if fit.loss is not None else None,
            source_file=(
                "redacted_local_source" if args.anonymize else str(master)
            ),
            notes=(
                f"Exact {fit.backend.upper()} SSB fit: {trials} Optuna "
                f"trials followed by {refine or 'no refinement'}."
            ),
        )
        from dataclasses import asdict

        from quantem.widget.showptycho import (
            _atomic_write_json,
            save_ptycho_calibration,
        )

        save_ptycho_calibration(calibration, out_dir / "calibration.json")
        software = _showptycho_software_provenance()
        fit_payload = {
            "schema_version": 1,
            "software": software,
            "export_software": software,
            "backend": fit.backend,
            "source_kind": source_kind,
            "source_path": str(source_path),
            "source_dtype": source_dtype,
            "source_bytes": source_bytes,
            "source_load_seconds": source_load_seconds,
            "bf_source_write_seconds": bf_source_write_seconds,
            "objective": "full_bf_phase_variance",
            "n_trials": fit.n_trials,
            "refine_method": fit.refine_method,
            "num_bf": fit.num_bf,
            "loss": fit.loss,
            "elapsed_seconds": fit.elapsed,
            "timings": dict(fit.timings),
            "refine_nfev": fit.refine_nfev,
            "aberrations": fit.aberrations,
            "bf_center": list(fit.bf_center),
            "bf_radius": fit.bf_radius,
            "calibration": asdict(calibration),
            "trials": list(fit.optuna_trials or ()),
        }
        if args.anonymize:
            fit_payload = _anonymize_showptycho_payload(fit_payload)
        _atomic_write_json(out_dir / "ssb_fit.json", fit_payload)
        print(f"  optimized calibration: {out_dir / 'calibration.json'}")
        print(f"  optimization record: {out_dir / 'ssb_fit.json'}")
    widget = ShowPtycho(
        widget_input,
        backend=args.backend,
        semiangle_mrad=float(semiangle),
        scan_sampling_A=float(scan_sampling),
        det_sampling=float(det_sampling) if det_sampling is not None else None,
        voltage_kV=float(voltage),
        bf_intensity_threshold=0.0,
        bf_radius=None,
        aberrations=aberrations,
        rotation_angle_deg=rotation,
        calibration=calibration,
        source_file=str(master),
        drag_bf=float(args.drag_bf),
        size=int(args.size),
        fft_on=bool(args.fft),
    )
    try:
        exported = widget.export(
            out_dir,
            title=(
                args.title
                or (
                    "ShowPtycho"
                    if args.anonymize
                    else f"{_showptycho_source_stem(master)} ShowPtycho"
                )
            ),
            overwrite=True,
            decode_dtype=_showptycho_decode_dtype(args),
        )
    finally:
        if written_evidence_path is not None:
            written_evidence_path.unlink(missing_ok=True)
    if fit is None and calibration_path is not None:
        fit_record_candidates = [
            calibration_path.parent / "ssb_fit.json",
            calibration_path.parent.parent / "ssb_fit.json",
        ]
        fit_record = next(
            (path for path in fit_record_candidates if path.is_file()),
            None,
        )
        if fit_record is not None and fit_record.resolve() != (exported / "ssb_fit.json").resolve():
            from quantem.widget.showptycho import _atomic_write_json

            _atomic_write_json(
                exported / "ssb_fit.json",
                _showptycho_reused_fit_payload(
                    fit_record,
                    anonymize=bool(args.anonymize),
                ),
            )
            print(f"  optimization record: {exported / 'ssb_fit.json'}")
    return exported


# ---------------------------------------------------------------------------
def _render_images(path: pathlib.Path, kind: str, args: argparse.Namespace) -> pathlib.Path:
    """Render one image (Show2D), a same-size folder (Show3D scrub), or a mixed
    folder (Show2D gallery), and write the HTML. Returns the written path."""
    from quantem.widget import Show2D, Show3D
    from quantem.widget.io import read_image_stack
    title = args.title
    if kind == "image":
        print(f"{path.name}: 1 image -> Show2D")
        widget = Show2D(_load_2d(path), title=title or path.stem)
        out = _out_path(args.out, path, suffix="show2d")
        widget.export_html(out)
        return out
    # Folder of images: try to stack into a Show3D scrub; differently-sized frames
    # cannot stack (np.stack raises) so fall back to a Show2D gallery.
    if args.widget != "2d":
        try:
            stack = read_image_stack(path, progress=args.verbose)
            widget = Show3D(stack, title=title or path.name)
            out = _out_path(args.out, path, suffix="show3d", from_dir=True)
            widget.export_html(out, quantized=args.quantized)
            return out
        except ValueError:
            if args.verbose:
                print("frames differ in size; rendering a Show2D gallery instead")
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    arrays = [_load_2d(p) for p in files]
    widget = Show2D(arrays, title=title or path.name)
    out = _out_path(args.out, path, suffix="gallery", from_dir=True)
    widget.export_html(out)
    return out


def _load_2d(path: pathlib.Path):
    """Decode one image file to a 2D float32 array. ``.npy`` / ``.emd`` / Gatan go
    through ``read_image`` (calibration-aware); raster formats use the same frame
    decoder ``read_image_stack`` uses, since this repo's ``read_image`` only knows
    ``.emd`` / ``.npy``."""
    from quantem.widget.io import read_image
    from quantem.widget.io.image import _read_frame
    if path.suffix.lower() in (".npy", ".emd", ".dm3", ".dm4"):
        return read_image(path).array
    return _read_frame(path)


def _render_4dstem_notebook(
    masters: list[str],
    label: str,
    args: argparse.Namespace,
    *,
    source_path: pathlib.Path | None = None,
) -> pathlib.Path:
    """Write a live Jupyter notebook that loads the 4D-STEM master(s) and opens a
    kernel-backed ``Show4DSTEM`` (full real-time WebGPU, no baked HTML). One master
    loads on its own; many load stacked into a 5D viewer with a dataset slider (the
    multi-tilt case). The notebook is the editable, real-use surface; ``--html`` is
    the share artifact."""
    import json
    backend = _normalise_show4dstem_backend(args.backend)
    devices = _python_gpus(args.gpus)
    page_budget = _python_page_budget(args.page_budget)
    backend_label = backend or "auto"
    print(
        f"{len(masters)} master(s), backend {backend_label}, bin {args.det_bin}, "
        f"dtype {args.dtype}, devices {devices} -> Show4DSTEM (live notebook)"
    )
    if source_path is not None and source_path.is_dir():
        gpus_arg = "None" if backend == "mps" else devices
        backend_arg = "None" if backend is None else repr(backend)
        source = (
            "from quantem.widget import Show4DSTEM\n"
            "\n"
            f"folder = {str(source_path)!r}\n"
            f"backend = {backend_arg}\n"
            f"gpus = {gpus_arg}\n"
            "print('folder:', folder)\n"
            "print('backend:', backend or 'auto')\n"
            "print('gpus:', gpus)\n"
            "viewer = Show4DSTEM.from_folder(\n"
            "    folder,\n"
            f"    backend={backend_arg},\n"
            f"    max_masters={len(masters)},\n"
            f"    min_masters={len(masters)},\n"
            f"    det_bin={int(args.det_bin)},\n"
            f"    dtype={args.dtype!r},\n"
            "    gpus=gpus,\n"
            f"    page_budget={page_budget},\n"
            "    watch=False,\n"
            "    verbose=True,\n"
            ")\n"
            "viewer\n"
        )
    else:
        arg = repr(masters[0]) if len(masters) == 1 else repr(masters)
        backend_line = "" if backend is None else f"    backend={backend!r},\n"
        devices_line = (
            f"    devices={devices},\n    series_type='generic',\n"
            if backend == "cuda" and devices != "None"
            else ""
        )
        page_device = devices if backend == "cuda" and devices != "None" else "None"
        source = (
            "from quantem.gpu.io import load\n"
            "from quantem.widget import Show4DSTEM\n"
            "\n"
            f"masters = {arg}\n"
            "data = load(\n"
            "    masters,\n"
            f"    det_bin={int(args.det_bin)},\n"
            f"    dtype={args.dtype!r},\n"
            "    apply_mask=True,\n"
            f"{backend_line}"
            f"{devices_line}"
            "    verbose=True,\n"
            ")\n"
            "Show4DSTEM(\n"
            "    data,\n"
            f"    page_budget={page_budget},\n"
            f"    page_device={page_device},\n"
            "    verbose=True,\n"
            ")\n"
        )
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "title",
                "metadata": {},
                "source": [
                    f"# {label}\n",
                    f"\n{len(masters)} master(s), backend `{backend_label}`, "
                    f"detector bin {args.det_bin}, dtype `{args.dtype}`.",
                ],
            },
            {"cell_type": "code", "id": "viewer", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = _out_dir(args.out) / f"{label}.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    return out


def _python_page_budget(value: str | int | None) -> str:
    """Return a source literal for a Show4DSTEM page budget CLI value."""
    if value is None:
        return "None"
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text.lower() in {"none", "off", "false", "no"}:
        return "None"
    if text.isdigit():
        return str(int(text))
    return repr(text)


def _python_gpus(value: str | None) -> str:
    """Return a source literal for comma-separated CUDA GPU ids."""
    if value is None or not str(value).strip():
        return "None"
    try:
        ids = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--gpus must be a comma-separated list of integer ids, e.g. 0 or 0,1") from exc
    if not ids:
        return "None"
    return repr(ids)


def _render_4dstem_watch_notebook(folder: pathlib.Path, label: str, args: argparse.Namespace) -> pathlib.Path:
    """Write a live ShowFolder-watched notebook for a 4D-STEM acquisition folder."""
    import json

    print(
        f"{folder.name}: watched folder, bin {args.det_bin}, page_budget {args.page_budget} "
        "-> ShowFolder + lazy Show4DSTEM"
    )
    gpus = _python_gpus(args.gpus)
    page_budget = _python_page_budget(args.page_budget)
    scan_size = "None" if args.scan_size is None else str(int(args.scan_size))
    source = (
        "from quantem.widget import ShowFolder\n"
        "\n"
        f"folder = ShowFolder({str(folder)!r}, thumb=256, group_by='none')\n"
        "folder.browser.attach_selection_panel()\n"
        "folder.browser.open_show4dstem(\n"
        f"    gpus={gpus},\n"
        f"    page_budget={page_budget},\n"
        f"    det_bin={int(args.det_bin)},\n"
        f"    dtype={args.dtype!r},\n"
        f"    scan_size={scan_size},\n"
        ")\n"
        f"folder.watch(interval={float(args.watch_interval)!r})\n"
        "folder\n"
    )
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "title",
                "metadata": {},
                "source": [
                    f"# {label} live Show4DSTEM\n",
                    f"\nWatched folder: `{folder}`\n",
                    f"\nDetector bin {args.det_bin}; page budget `{args.page_budget}`; "
                    f"watch interval {args.watch_interval:g}s.",
                ],
            },
            {
                "cell_type": "code",
                "id": "live-viewer",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = _out_dir(args.out) / f"{label}_live.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    return out


def _render_image_watch_notebook(
    folder: pathlib.Path,
    label: str,
    args: argparse.Namespace,
    *,
    widget: str,
) -> pathlib.Path:
    """Write a live ShowFolder-watched notebook for image folder previews."""
    import json

    method = "open_show3d" if widget == "show3d" else "open_show2d"
    title = "Show3D" if widget == "show3d" else "Show2D"
    print(f"{folder.name}: watched folder -> ShowFolder + live all-image {title}")
    source = (
        "from quantem.widget import ShowFolder\n"
        "\n"
        f"folder = ShowFolder({str(folder)!r}, thumb=256, group_by='none')\n"
        "folder.browser.attach_selection_panel()\n"
        f"folder.browser.{method}(all_images=True)\n"
        f"folder.watch(interval={float(args.watch_interval)!r})\n"
        "folder\n"
    )
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "title",
                "metadata": {},
                "source": [
                    f"# {label} live {title}\n",
                    f"\nWatched folder: `{folder}`\n",
                    f"\nNew readable image files append on the next poll; "
                    f"watch interval {args.watch_interval:g}s.",
                ],
            },
            {
                "cell_type": "code",
                "id": "live-viewer",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = _out_dir(args.out) / f"{label}_{widget}_live.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    return out


def _render_gallery(files: list[pathlib.Path], label: str, args: argparse.Namespace) -> pathlib.Path:
    """Render several explicit image files as one Show2D gallery HTML."""
    from quantem.widget import Show2D
    print(f"{len(files)} images -> Show2D gallery")
    widget = Show2D([_load_2d(p) for p in files], title=args.title or label)
    out = _out_dir(args.out) / f"{label}.html"
    widget.export_html(out)
    return out


def _launch_notebook(notebook: pathlib.Path, *, no_open: bool) -> None:
    """Open the notebook for the user. Locally (a Mac or any box with a display) start
    ``jupyter lab`` on it, which opens the browser. On a headless/remote box a browser
    cannot be reached, so print the path plus the ``mj jupyter`` hint instead (never
    start a server the user cannot see)."""
    import shutil
    import subprocess
    headless = sys.platform != "darwin" and not os.environ.get("DISPLAY")
    if no_open or headless:
        print(f"wrote {notebook}")
        if headless:
            print(f"  open it from your Mac:  mj jupyter cuda-env quantem   (then open {notebook.name})")
        return
    jupyter = shutil.which("jupyter")
    if jupyter is None:
        probe = subprocess.run(
            [sys.executable, "-m", "jupyter", "lab", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        command = [sys.executable, "-m", "jupyter", "lab", str(notebook)] if probe.returncode == 0 else None
    else:
        command = [jupyter, "lab", str(notebook)]
    if command is None:
        print(f"wrote {notebook}")
        print("  jupyter not found in this Python; install it or open the notebook in your editor")
        return
    print(f"launching jupyter lab on {notebook}")
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _open_show4dstem_command(command: pathlib.Path, *, no_open: bool) -> None:
    """Open a generated Show4DSTEM folder launcher when a desktop is available."""

    if no_open:
        print(f"wrote {command}")
        return
    headless = sys.platform != "darwin" and not os.environ.get("DISPLAY")
    if headless or not command.is_file():
        print(f"wrote {command}")
        return
    import subprocess

    if sys.platform == "darwin":
        subprocess.Popen(
            ["open", str(command)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        webbrowser.open(command.as_uri())
    print(f"opened {command}")


def _render_4dstem_webgpu_h5(
    masters: list[str],
    label: str,
    args: argparse.Namespace,
) -> pathlib.Path:
    """Render Show4DSTEM WebGPU HTML over linked H5 data."""

    import numpy as np
    from quantem.widget import Show4DSTEM
    from quantem.widget.show4dstem_factory import _master_file_contract
    from quantem.widget.show4dstem_webgpu_export import (
        bundle_master_urls,
        export_show4dstem_webgpu_bundle,
    )

    if int(args.det_bin) != 1:
        raise ValueError(
            "Show4DSTEM --backend webgpu uses linked HDF5 files with browser "
            "range reads; keep --bin 1."
        )
    contracts = [_master_file_contract(master) for master in masters]
    first = contracts[0]
    scan_shape = first.get("scan_shape")
    detector_shape = first.get("detector_shape")
    n_frames = first.get("n_frames")
    if scan_shape is None or detector_shape is None or n_frames is None:
        raise ValueError("could not infer scan/detector shape from the first master")
    expected = {
        "scan_shape": tuple(int(value) for value in scan_shape),
        "detector_shape": tuple(int(value) for value in detector_shape),
        "n_frames": int(n_frames),
    }
    for master, contract in zip(masters[1:], contracts[1:], strict=True):
        observed = {
            "scan_shape": tuple(int(value) for value in contract.get("scan_shape") or ()),
            "detector_shape": tuple(int(value) for value in contract.get("detector_shape") or ()),
            "n_frames": int(contract.get("n_frames") or 0),
        }
        if observed != expected:
            raise ValueError(
                f"incompatible Show4DSTEM master {pathlib.Path(master).name!r}: "
                f"{observed}; expected {expected}"
            )

    out_dir = _out_dir(args.out) / f"{label}_show4dstem_webgpu"
    replaced = _prepare_show4dstem_webgpu_output_dir(out_dir)
    if replaced:
        print(f"replaced existing Show4DSTEM WebGPU export: {out_dir}")
    link_labels = [f"dataset_{idx:02d}" for idx in range(len(masters))]
    frame_labels = [_show4dstem_dataset_label(master, idx) for idx, master in enumerate(masters)]
    data_dir = out_dir / "data"
    data_dir.mkdir()
    for master, dataset_label in zip(masters, link_labels, strict=True):
        _link_show4dstem_h5_family(data_dir, pathlib.Path(master), dataset_label)

    widget = Show4DSTEM(
        np.zeros((1, 1, 1, 1), dtype=np.uint8),
        h5_urls=bundle_master_urls(data_dir, viewer_prefix="../data"),
        backend="webgpu",
        scan_shape=expected["scan_shape"],
        detector_shape=expected["detector_shape"],
        frame_dim_label="Dataset",
        frame_labels=frame_labels,
        view_mode="multiple" if len(masters) > 1 else "single",
        compare_max_panels=max(1, len(masters)),
        compare_group_mode="all",
        compare_dp_mode="selected" if len(masters) > 1 else "average",
        title=args.title or label,
        verbose=bool(args.verbose),
        show_controls=True,
    )
    decode_dtype = "uint8" if str(args.dtype).lower() in {"u8", "uint8"} else "uint16"
    export_show4dstem_webgpu_bundle(
        widget,
        out_dir,
        title=args.title or label,
        h5_decode_dtype=decode_dtype,
    )
    out = out_dir / "index.html"
    print(
        f"{len(masters)} master(s), backend webgpu, bin 1, dtype {args.dtype} "
        f"-> {out_dir / 'Show4DSTEM.command'}"
    )
    return out


def _prepare_show4dstem_webgpu_output_dir(out_dir: pathlib.Path) -> bool:
    """Create a fresh generated Show4DSTEM WebGPU export directory.

    The CLI owns ``*_show4dstem_webgpu`` directories. Reusing one is unsafe:
    stale ``index.html`` files, lazy metadata, or generated shards can make a
    launcher open yesterday's viewer while appearing to run today's command.
    """

    if not out_dir.name.endswith("_show4dstem_webgpu"):
        raise ValueError(
            "internal error: refusing to replace a non-Show4DSTEM WebGPU "
            f"output directory: {out_dir}"
        )
    existed = out_dir.exists() or out_dir.is_symlink()
    if out_dir.is_symlink() or out_dir.is_file():
        out_dir.unlink()
    elif out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    return existed


def _link_show4dstem_h5_family(out_dir: pathlib.Path, master: pathlib.Path, label: str) -> str:
    """Symlink a master and same-prefix data chunks under an anonymous label."""

    source_master = master.expanduser().resolve()
    source_prefix = source_master.name[: -len("_master.h5")]
    master_link = out_dir / f"{label}_master.h5"
    _replace_symlink(master_link, source_master)
    for data_file in sorted(source_master.parent.glob(f"{source_prefix}_data_*.h5")):
        data_link = out_dir / data_file.name.replace(source_prefix, label, 1)
        _replace_symlink(data_link, data_file.resolve())
    return master_link.name


def _replace_symlink(link: pathlib.Path, target: pathlib.Path) -> None:
    """Replace only symlink artifacts; never overwrite a real data file."""

    if link.exists() or link.is_symlink():
        if not link.is_symlink():
            raise ValueError(f"refusing to replace non-symlink artifact file: {link}")
        link.unlink()
    link.symlink_to(target)


def _show4dstem_export_dtype(args: argparse.Namespace) -> str:
    """Return the Show4DSTEM HTML export dtype requested by the CLI."""

    raw = str(getattr(args, "dtype", "u8") or "u8").strip().lower()
    if raw in {"u8", "uint8"}:
        return "uint8"
    if raw in {"u16", "uint16"}:
        return "uint16"
    raise ValueError(
        "Show4DSTEM --html export supports --dtype u8/uint8 or u16/uint16. "
        "Use a live notebook for float32 analysis."
    )


def _master_to_binned_numpy(master: str, det_bin: int, dtype: str = "u8"):
    """Load one master with detector binning and return a mean-binned 4D numpy array
    ``(scan_row, scan_col, det_row, det_col)``. Binning happens at LOAD time (so the
    full 19 GB stack never materializes - fits a laptop), and since the loader
    integer-SUMS over det_bin^2 we divide by that to get the MEAN, which keeps values
    in the raw range so the uint8 pack never clips. Works on CUDA / MPS (zero-copy
    ChunkedFrames, materialized via its chunks) / CPU."""
    import numpy as np
    import torch
    from quantem.gpu.io import load
    result = load(master, det_bin=det_bin, dtype=dtype)
    data = result.data if hasattr(result, "data") else result
    meta = getattr(result, "metadata", {}) or {}
    if hasattr(data, "chunks"):
        arr = np.concatenate([np.asarray(chunk) for chunk in data.chunks], axis=0)
    elif hasattr(data, "get"):
        arr = data.get()
    elif isinstance(data, torch.Tensor):
        arr = data.detach().to("cpu").numpy()
    else:
        arr = np.asarray(data)
    if arr.ndim == 3:
        scan = meta.get("scan_shape")
        rows, cols = scan if scan else (int(round(arr.shape[0] ** 0.5)),) * 2
        arr = arr.reshape(rows, cols, arr.shape[-2], arr.shape[-1])
    if det_bin > 1:
        arr = np.round(arr.astype(np.float32) / (det_bin * det_bin))  # loader summed -> mean
    return np.ascontiguousarray(arr.astype(np.float32))


def _render_4dstem(masters: list[str], label: str, args: argparse.Namespace) -> list[pathlib.Path]:
    """Render 4D-STEM master(s) as offline WebGPU Show4DSTEM HTML.

    Each master is loaded with the requested detector binning (``--bin``, default
    1 for full detector sampling). ``--combined`` instead
    stacks every master into one 5D viewer (a bslz4 companion folder + a local
    serve, or open through the file-grant browser path)."""
    import numpy as np
    from quantem.widget import Show4DSTEM
    out_dir = _out_dir(args.out)
    export_dtype = _show4dstem_export_dtype(args)
    if args.combined and len(masters) > 1:
        # Stack the masters into one 5D numpy array and pass THAT to the viewer. A
        # 5D array routes to the universal Show4DSTEM (which has the offline
        # multi-volume WebGPU frame-flip), not the MacBook live-Metal viewer (whose
        # offline export can't switch volumes kernel-lessly).
        volumes = [_master_to_binned_numpy(m, args.det_bin, args.dtype) for m in masters]
        stack = np.stack(volumes, axis=0)
        data_url = out_dir / "widget-data"
        widget = Show4DSTEM(
            stack, backend="webgpu", offline_codec="bslz4", data_url=str(data_url),
            frame_dim_label="Dataset",
            frame_labels=[pathlib.Path(m).stem.replace("_master", "") for m in masters],
        )
        out = out_dir / f"{label}_combined.html"
        widget.export_html(str(out), title=args.title, dtype=export_dtype)
        return [out]
    outputs = []
    iterator = masters
    if args.verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(masters, desc="export")
        except ImportError:
            pass
    for master in iterator:
        stem = pathlib.Path(master).stem.replace("_master", "")
        try:
            # Mean-bin at load (memory-safe: the full 19 GB stack never materializes)
            # so uint8 never clips the bright field. Data is already binned, so the
            # export does no further binning.
            arr = _master_to_binned_numpy(master, args.det_bin, args.dtype)
            widget = Show4DSTEM(arr, backend="webgpu")
            out = out_dir / f"{stem}.html"
            widget.export_html(str(out), title=args.title or stem, dtype=export_dtype)
            outputs.append(out)
        except (RuntimeError, ValueError, OSError, MemoryError) as err:
            print(f"quantem: skipped {stem}: {err}", file=sys.stderr)
    if not outputs:
        raise ValueError("every master failed to export (see messages above)")
    return outputs


# ---------------------------------------------------------------------------
def _out_path(out: str | None, src: pathlib.Path, *, suffix: str, from_dir: bool = False) -> pathlib.Path:
    """Resolve a single output HTML path from ``--out`` (file or dir) or default to
    ``<source-stem>_<suffix>.html`` beside the input."""
    base = (src.name if from_dir else src.stem)
    default_name = f"{base}_{suffix}.html"
    if out is None:
        return _default_out_dir() / default_name
    target = pathlib.Path(out).expanduser()
    if target.is_dir() or out.endswith("/"):
        target.mkdir(parents=True, exist_ok=True)
        return target / default_name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _out_dir(out: str | None) -> pathlib.Path:
    """Resolve the output directory (``--out`` or the default ``~/Downloads``)."""
    target = pathlib.Path(out).expanduser() if out else _default_out_dir()
    target.mkdir(parents=True, exist_ok=True)
    return target


def _default_out_dir() -> pathlib.Path:
    """Default save location: the user's ``~/Downloads`` (where a shareable artifact
    is expected and always writable), falling back to the current directory when no
    Downloads folder exists (servers, CI)."""
    downloads = pathlib.Path.home() / "Downloads"
    return downloads if downloads.is_dir() else pathlib.Path.cwd()


class _RangeRequestHandler(http.server.BaseHTTPRequestHandler):
    """Static file handler with single byte-range support for folder exports."""

    root: pathlib.Path

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - stdlib API name.
        if os.environ.get("QUANTEM_CLI_HTTP_LOG"):
            super().log_message(format, *args)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_common_headers()
        self.end_headers()

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def do_GET(self) -> None:
        self._serve(send_body=True)

    def do_PUT(self) -> None:
        path = self._resolve_snapshot_write_path(allow_json=True)
        if path is None:
            self.send_error(403, "writes are restricted to snapshots")
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            self.send_error(400, "invalid content length")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.rfile.read(max(0, length)))
        self.send_response(204)
        self._send_common_headers()
        self.end_headers()

    def do_DELETE(self) -> None:
        path = self._resolve_snapshot_write_path(allow_json=False)
        if path is None:
            self.send_error(403, "deletes are restricted to snapshot images")
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        self.send_response(204)
        self._send_common_headers()
        self.end_headers()

    def _serve(self, *, send_body: bool) -> None:
        path = self._resolve_path()
        if path is None:
            self.send_error(404, "file not found")
            return
        if path.is_dir():
            path = path / "index.html"
        if not path.is_file():
            self.send_error(404, "file not found")
            return

        size = path.stat().st_size
        start, end, partial = 0, size - 1, False
        range_header = self.headers.get("Range")
        if range_header:
            parsed = _parse_http_range(range_header, size)
            if parsed is None:
                self.send_response(416)
                self._send_common_headers()
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = parsed
            partial = True

        content_length = max(0, end - start + 1)
        self.send_response(206 if partial else 200)
        self._send_common_headers()
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Last-Modified", email.utils.formatdate(path.stat().st_mtime, usegmt=True))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        if not send_body:
            return
        with path.open("rb") as handle:
            self._send_file_body(handle, start=start, length=content_length)

    def _send_common_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")

    def _send_file_body(self, handle, *, start: int, length: int) -> None:
        """Send a file range with zero-copy when the local socket supports it."""
        if length <= 0:
            return
        if not os.environ.get("QUANTEM_DISABLE_HTTP_SENDFILE") and hasattr(self.connection, "sendfile"):
            try:
                self.connection.sendfile(handle, offset=start, count=length)
                return
            except BrokenPipeError:
                return
            except (OSError, ValueError):
                pass
        handle.seek(start)
        remaining = length
        chunk_bytes = _range_fallback_chunk_bytes()
        while remaining > 0:
            chunk = handle.read(min(chunk_bytes, remaining))
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
            except BrokenPipeError:
                break
            remaining -= len(chunk)

    def _resolve_path(self) -> pathlib.Path | None:
        parsed = urllib.parse.urlsplit(self.path)
        raw_path = urllib.parse.unquote(parsed.path)
        norm = posixpath.normpath(raw_path)
        rel = pathlib.Path(norm.lstrip("/"))
        root = self.root.resolve()
        candidate = root / rel
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    def _resolve_snapshot_write_path(self, *, allow_json: bool) -> pathlib.Path | None:
        parsed = urllib.parse.urlsplit(self.path)
        raw_path = urllib.parse.unquote(parsed.path)
        norm = posixpath.normpath(raw_path)
        rel_text = norm.lstrip("/")
        parts = pathlib.PurePosixPath(rel_text).parts
        if len(parts) == 2 and parts[0] == "snapshots":
            pass
        elif len(parts) == 3 and parts[1] == "snapshots":
            pass
        else:
            return None
        name = parts[-1]
        if allow_json and name == "snapshots.json":
            pass
        elif (
            name.startswith("snapshot_")
            and (name.endswith(".jpg") or name.endswith(".jpeg"))
        ):
            pass
        else:
            return None
        root = self.root.resolve()
        candidate = root.joinpath(*parts).resolve()
        try:
            candidate.parent.relative_to(root)
        except ValueError:
            return None
        if candidate.parent.name != "snapshots":
            return None
        return candidate


def _range_fallback_chunk_bytes() -> int:
    """Return the fallback HTTP chunk size for large local folder exports."""
    raw = os.environ.get("QUANTEM_HTTP_RANGE_CHUNK_MB", "")
    if raw:
        try:
            mb = int(raw)
        except ValueError:
            mb = 16
        return max(1, mb) * 1024 * 1024
    return _RANGE_FALLBACK_CHUNK_BYTES


def _parse_http_range(value: str, size: int) -> tuple[int, int] | None:
    """Parse a single HTTP byte range header."""
    match = _RANGE_RE.fullmatch(value.strip())
    if not match or size < 0:
        return None
    start_text, end_text = match.groups()
    if start_text == "" and end_text == "":
        return None
    if start_text == "":
        suffix = int(end_text)
        if suffix <= 0:
            return None
        start = max(0, size - suffix)
        end = size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    if start >= size or end < start:
        return None
    return start, min(end, size - 1)


def _showptycho_manifest(folder: pathlib.Path) -> dict:
    """Read a ShowPtycho folder manifest after validation."""
    manifest = _showptycho_manifest_path(folder)
    if manifest is None:
        raise ValueError(f"ShowPtycho folder export is missing snapshots/manifest.json: {folder}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def _host_for_url(bind: str) -> str:
    return "127.0.0.1" if bind in {"", "0.0.0.0", "::"} else bind


def _serve_showptycho_folder(
    folder: pathlib.Path,
    *,
    bind: str,
    port: int | None,
    no_open: bool,
) -> None:
    """Serve a ShowPtycho WebGPU folder export and open it for the user."""
    if _is_showptycho_collection(folder):
        folder = _showptycho_collection_folder(folder)
        collection = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        print(f"ShowPtycho collection: {folder}")
        print(f"  datasets: {len(collection.get('datasets', []))}")
    else:
        folder = _showptycho_folder(folder)
        manifest = _showptycho_manifest(folder)
        source = manifest.get("source", {})
        print(f"ShowPtycho folder: {folder}")
        if source.get("kind") == "hdf5":
            data_files = source.get("data_files") or []
            link_mode = ", ".join(source.get("link_mode") or []) or "linked"
            preferred = source.get("preferred_browser_source") or "compressed_hdf5"
            bf_columns = source.get("bf_columns") or {}
            print(
                "  source: compressed HDF5 "
                f"{source.get('master', 'source master')} + {len(data_files)} data file(s) "
                f"({link_mode}); no persistent BF-G cache"
            )
            print(f"  browser source: {preferred}")
            if preferred == "bf_columns" and bf_columns:
                print(
                    "  BF columns: "
                    f"{bf_columns.get('num_bf', '?')} BF x {bf_columns.get('plane', '?')} scan, "
                    f"{_fmt_bytes(int(bf_columns.get('bytes', 0) or 0))}"
                )
        else:
            raise ValueError(
                "ShowPtycho folder manifest has no compressed HDF5 detector source."
            )
    if no_open:
        print("  ready: run without --no-open to serve and open this folder.")
        return

    handler = type("QuantemShowPtychoRangeHandler", (_RangeRequestHandler,), {"root": folder})
    server = http.server.ThreadingHTTPServer((bind, port or 0), handler)
    actual_port = server.server_address[1]
    url = f"http://{_host_for_url(bind)}:{actual_port}/index.html"
    print(f"  open: {url}")
    print("  serving folder export; press Ctrl-C to stop")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    headless = sys.platform != "darwin" and not os.environ.get("DISPLAY")
    if not headless:
        webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()


def _open_html(path: pathlib.Path, *, serve: bool, no_open: bool) -> None:
    """Open the HTML for the user: a self-contained file via ``file://``, or behind
    a local HTTP server when serving (required for bslz4 companions, and the only
    way a remote/SSH user can tunnel in). On a headless box, just print the path."""
    headless = sys.platform != "darwin" and not os.environ.get("DISPLAY")
    if no_open:
        print(f"wrote {path}")
        return
    if serve:
        directory = str(path.parent)

        def handler(*args, **kwargs):
            return http.server.SimpleHTTPRequestHandler(
                *args,
                directory=directory,
                **kwargs,
            )

        httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{port}/{path.name}"
        print(f"serving {url}  (Ctrl-C to stop)")
        if not headless:
            webbrowser.open(url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            httpd.shutdown()
        return
    if headless:
        print(f"wrote {path}  (open it in a browser)")
        return
    webbrowser.open(path.as_uri())
    print(f"opened {path}")


if __name__ == "__main__":
    raise SystemExit(main())
