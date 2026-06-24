"""``quantem`` command-line interface: a folder (or file) of images or 4D-STEM
masters becomes a rendered, standalone HTML viewer in one command, no notebook.

    quantem show ./frames/                # PNG/TIFF folder -> Show3D scrub HTML
    quantem show scan.png                 # single image    -> Show2D HTML
    quantem show ./masters/ --bin 8       # *_master.h5     -> offline WebGPU Show4DSTEM
    quantem html tutorial.ipynb           # run a notebook  -> standalone shareable HTML

The CLI only orchestrates existing pieces: ``io.read_image`` / ``read_image_stack``
for images, ``io.discover_masters`` + ``io.load(det_bin=...)`` for 4D-STEM, the
``Show2D`` / ``Show3D`` / ``Show4DSTEM`` widgets, and each widget's ``export_html``.
4D-STEM is packed offline so the browser does all compute on WebGPU, which is what
lets a laptop browse data that never fit full resolution (bin the detector first).
"""
import argparse
import http.server
import os
import pathlib
import socketserver
import sys
import threading
import webbrowser

# Single image -> Show2D, a folder of frames -> Show3D, a folder of differently
# sized images -> a Show2D gallery. These are the formats read_image understands.
IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".dm3", ".dm4", ".emd", ".npy"}
MASTER_PATTERN = "*_master.h5"


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
    forced = {"show": "auto", "show2d": "2d", "show3d": "3d", "show4dstem": "4dstem"}
    helps = {
        "show": "Auto-detect PATH(s) and render the matching viewer.",
        "show2d": "Render an image (or a folder of images) as Show2D.",
        "show3d": "Render a folder of frames as a Show3D scrub.",
        "show4dstem": "Render 4D-STEM master(s) as Show4DSTEM (live notebook, or --html).",
    }
    for name in forced:
        _add_show_args(sub.add_parser(name, help=helps[name]))
    # `html` is a different shape (one .ipynb in, one HTML out), so it gets its own
    # parser rather than the shared show* options.
    _add_html_args(sub.add_parser(
        "html", help="Execute a notebook and export it to a standalone, offline shareable HTML."))
    # `jupyter` starts JupyterLab here on the GPU box and prints a URL to paste into the
    # laptop browser - kernel + GPU here, UI in your browser. You bring your own tunnel
    # (SSH -L / VS Code) the same way quantem.live does.
    _add_jupyter_args(sub.add_parser(
        "jupyter", help="Start JupyterLab on this GPU box and print a URL to open from your laptop."))
    args = parser.parse_args(argv)
    try:
        if args.command == "html":
            return _render_html(args)
        if args.command == "jupyter":
            return _launch_jupyter(args)
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


def _render_html(args: argparse.Namespace) -> int:
    """Execute a notebook and export it to a standalone, shareable HTML.

    Wraps ``jupyter nbconvert --to html [--execute]``: a finished notebook becomes a
    kernel-less HTML (interactive widgets such as Show2D bake in as static images) that
    opens in any browser with no Python. The live ``.ipynb`` stays the editable surface;
    this is the share artifact. ``--no-execute`` exports the saved outputs as-is, which
    is what a notebook's own in-cell ``!jupyter nbconvert`` does after a run."""
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
        cmd += ["--execute", f"--ExecutePreprocessor.timeout={args.timeout}"]
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
def _add_jupyter_args(parser: argparse.ArgumentParser) -> None:
    """Attach options for the ``jupyter`` subcommand."""
    parser.add_argument("path", nargs="?", default=None,
                        help="Notebook or directory to open (relative to where you run this, or "
                             "absolute). A .ipynb opens directly.")
    parser.add_argument("--env", default="live-env",
                        help="Conda/mamba env to activate before launching JupyterLab "
                             "(default: live-env). Pass --env '' to skip activation.")
    parser.add_argument("--port", type=int, default=None,
                        help="Port to serve on (default: auto-pick a free port).")
    parser.add_argument("--no-open", action="store_true",
                        help="Start JupyterLab but do not open a browser (the box is usually "
                             "headless - copy the printed URL into your laptop browser).")


def _free_port() -> int:
    """Pick a free local TCP port by binding to port 0 and reading back what the OS chose.
    Runs on the same box JupyterLab will, so the port is guaranteed free for the server."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _config_path() -> pathlib.Path:
    """Where the per-user jupyter config lives. Honors XDG_CONFIG_HOME, else ~/.config."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = pathlib.Path(base) if base else pathlib.Path.home() / ".config"
    return root / "quantem" / "jupyter.json"


def _ssh_target() -> str:
    """The `user@host` a laptop uses to SSH into this box, for the copy-paste tunnel line.

    Saved once in the per-user config so the printed `ssh -L` command is ready to paste
    with no placeholder to edit. On first interactive launch we auto-detect `whoami@fqdn`,
    show it, and let the user accept (Enter) or correct it, then persist it. Later launches
    read it silently. Non-interactive launches (no TTY) fall back to the auto-detected guess
    without prompting or saving, so a headless run still prints a usable command."""
    import getpass
    import json
    import socket
    cfg = _config_path()
    if cfg.exists():
        try:
            saved = json.loads(cfg.read_text()).get("ssh_target")
            if saved:
                return saved
        except (OSError, ValueError):
            pass
    default = f"{getpass.getuser()}@{socket.getfqdn()}"
    if not sys.stdin.isatty():
        return default
    answer = input(f"SSH target your laptop uses to reach this box [{default}]: ").strip()
    target = answer or default
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"ssh_target": target}, indent=2))
    print(f"  saved to {cfg} (edit it anytime to change)")
    return target


def _launch_jupyter(args: argparse.Namespace) -> int:
    """Start JupyterLab here on the GPU box and print a URL to paste into the laptop
    browser - kernel + GPU run here, UI in your browser. anywidget rides the Jupyter Comm
    channel, so one server serves every quantem widget (Show2D/3D/4DSTEM, SSB, any
    tutorial) with no per-widget setup. You bring your own laptop->box hop (SSH ``-L`` or
    VS Code Remote-SSH), same as quantem.live; this command does no SSH or tunnel work.

    Runs under a login shell so conda init is sourced, in the foreground so JupyterLab
    lives as long as the command does (Ctrl-C stops it)."""
    import secrets
    import shlex
    import subprocess
    import threading
    import webbrowser
    port = args.port or _free_port()
    token = secrets.token_hex(16)
    # --log-level=WARN silences the ~20 INFO log lines that otherwise scroll the
    # paste-line banner off-screen. Errors + warnings still surface.
    launch = (f"jupyter lab --no-browser --ip=127.0.0.1 --port={port} --log-level=WARN "
              f"--IdentityProvider.token={token} --ServerApp.token={token}")
    if args.env:
        # `conda` is usually NOT on a non-interactive login shell's PATH, so a bare
        # `conda activate` fails. Source conda.sh from the common install locations
        # first, then activate. Covers miniforge/miniconda/anaconda/mambaforge.
        src = ("for c in ~/miniforge3 ~/miniconda3 ~/anaconda3 ~/mambaforge; do "
               "[ -f \"$c/etc/profile.d/conda.sh\" ] && . \"$c/etc/profile.d/conda.sh\" && break; done")
        launch = f"{src} && conda activate {shlex.quote(args.env)} && {launch}"
    if args.path and not args.path.endswith(".ipynb"):
        launch = f"cd {shlex.quote(args.path)} && {launch}"
    sub = ""
    if args.path and args.path.endswith(".ipynb"):
        sub = "/tree/" + args.path.lstrip("/")
    url = f"http://localhost:{port}/lab{sub}?token={token}"
    # Resolve (and on first run, save) the SSH target BEFORE printing, so its one-time
    # prompt doesn't interrupt the URL block.
    target = _ssh_target()
    # Two one-liners — same tunnel command, OS-specific browser launcher. macOS
    # uses `open`; Linux uses `xdg-open`. Windows (PowerShell / Git Bash / cmd)
    # uses `start` (PowerShell alias for Start-Process; Git Bash forwards to
    # Windows start; in cmd the empty-title arg is needed but `start "" "URL"`
    # works in all three Windows shells).
    line_unix = (
        f'ssh -fN -L {port}:127.0.0.1:{port} {target} && '
        f'(open "{url}" 2>/dev/null || xdg-open "{url}" 2>/dev/null || echo "{url}")'
    )
    line_win = (
        f'ssh -fN -L {port}:127.0.0.1:{port} {target} ; start "" "{url}"'
    )
    YL = "\033[1;33m"      # bold yellow — frame
    CY = "\033[1;96m"      # bold bright cyan — the paste command
    GN = "\033[1;32m"      # bold green — status
    DM = "\033[2;37m"      # dim white — alternates
    AR = "\033[1;31m"      # bold red — arrows
    LB = "\033[1;35m"      # bold magenta — OS labels
    RST = "\033[0m"
    bar = "═" * 72
    arrow = "↓ ↓ ↓"
    print()
    print(f"{YL}╔{bar}╗{RST}")
    print(f"{YL}║{RST}    {AR}{arrow}{RST}  {YL}COPY-PASTE ONE LINE INTO YOUR LAPTOP TERMINAL{RST}  {AR}{arrow}{RST}        {YL}║{RST}")
    print(f"{YL}╚{bar}╝{RST}")
    print()
    print(f"  {LB}macOS / Linux:{RST}")
    print(f"    {CY}{line_unix}{RST}")
    print()
    print(f"  {LB}Windows (PowerShell or Git Bash):{RST}")
    print(f"    {CY}{line_win}{RST}")
    print()
    print(f"  {GN}✓ JupyterLab running on this box, port {port}. Ctrl-C here to stop.{RST}")
    print(f"  {DM}(URL alone): {url}{RST}")
    print(f"  {DM}(tunnel alone): ssh -L {port}:127.0.0.1:{port} {target}{RST}")
    print(f"  {DM}(no SSH key yet? https://github.com/ophusgroup/dev#appendix-c-ssh-for-github-and-gpu-servers){RST}")
    print()
    # Flush now: the foreground server below never returns, so block-buffered stdout
    # (when redirected to a file/pipe, not a TTY) would otherwise hide the URL forever.
    sys.stdout.flush()
    if not args.no_open:
        # If a browser is reachable (rare on a headless box), open it after the server
        # has had a moment to come up. The foreground process below keeps it alive.
        threading.Timer(3.0, lambda: webbrowser.open(url)).start()
    return subprocess.run(["bash", "-lc", launch]).returncode


def _add_show_args(parser: argparse.ArgumentParser) -> None:
    """Attach the shared options to a show* subparser (one source of truth)."""
    parser.add_argument("path", nargs="+",
                        help="An image, a folder of images, a 4D-STEM master, a folder of masters, "
                             "or several master files (-> one 5D multi-tilt viewer).")
    parser.add_argument("--bin", type=int, default=8, dest="det_bin",
                        help="Detector binning factor for 4D-STEM (default 8). Keeps a laptop-sized stack.")
    parser.add_argument("--combined", action="store_true",
                        help="Many 4D masters -> one 5D HTML viewer (with --html; needs a local serve).")
    parser.add_argument("--out", default=None,
                        help="Output path (single) or directory (batch). Default: ~/Downloads.")
    parser.add_argument("--quantized", action="store_true",
                        help="Image widgets: uint8 pack (smaller file).")
    parser.add_argument("--html", action="store_true",
                        help="4D-STEM: export a standalone offline-WebGPU HTML instead of a live notebook.")
    parser.add_argument("--no-open", action="store_true", help="Write the file(s) but do not launch anything.")
    parser.add_argument("--serve", action="store_true",
                        help="Open via a local HTTP server even for self-contained files (tunnelable URL).")
    parser.add_argument("--title", default=None, help="Viewer page title.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose progress.")


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
    # Several explicit paths: a list of masters -> one 5D viewer (multi-tilt), or a
    # set of image files -> a gallery. A single path falls through to _detect.
    if len(paths) > 1:
        if args.widget != "4dstem" and all(p.suffix.lower() in IMAGE_EXTS for p in paths):
            out = _render_gallery(paths, "gallery", args)
            _open_html(out, serve=args.serve, no_open=args.no_open)
            return 0
        masters = [str(p) for p in paths]
        return _do_4dstem(masters, f"{len(masters)}_datasets", args)
    path = paths[0]
    kind = _detect(path, args.widget)
    if kind == "4dstem":
        from quantem.widget.io import discover_masters
        masters = [str(path)] if path.is_file() else discover_masters(str(path), verbose=args.verbose)
        if not masters:
            raise ValueError(f"no *_master.h5 found in {path}")
        label = pathlib.Path(masters[0]).stem.replace("_master", "") if path.is_file() else path.name
        return _do_4dstem(masters, label, args)
    out = _render_images(path, kind, args)
    _open_html(out, serve=args.serve, no_open=args.no_open)
    return 0


def _do_4dstem(masters: list[str], label: str, args: argparse.Namespace) -> int:
    """Dispatch 4D-STEM master(s) to either a live notebook (default) or an offline
    HTML (``--html``), then launch/open it. One master loads alone; many load stacked
    into a 5D viewer with a dataset slider (the multi-tilt case)."""
    if args.html:
        outputs = _render_4dstem(masters, label, args)
        _open_html(outputs[0], serve=args.serve or args.combined, no_open=args.no_open)
        if len(outputs) > 1:
            print(f"wrote {len(outputs)} HTML files to {outputs[0].parent}")
        return 0
    notebook = _render_4dstem_notebook(masters, label, args)
    _launch_notebook(notebook, no_open=args.no_open)
    return 0


def _detect(path: pathlib.Path, forced: str) -> str:
    """Return the content kind: 'image' (single file), 'images' (folder), or '4dstem'.

    A single file is always 'image' unless it is a master or 4D is forced (a lone
    file can't be a 3D scrub). For a folder: the command's forced widget wins, else a
    ``*_master.h5`` makes it 4D and image files make it 'images'. The stack-vs-gallery
    split for 'images' is decided later from the forced widget."""
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


def _render_4dstem_notebook(masters: list[str], label: str, args: argparse.Namespace) -> pathlib.Path:
    """Write a live Jupyter notebook that loads the 4D-STEM master(s) and opens a
    kernel-backed ``Show4DSTEM`` (full real-time WebGPU, no baked HTML). One master
    loads on its own; many load stacked into a 5D viewer with a dataset slider (the
    multi-tilt case). The notebook is the editable, real-use surface; ``--html`` is
    the share artifact."""
    import json
    print(f"{len(masters)} master(s), bin {args.det_bin} -> Show4DSTEM (live notebook)")
    arg = repr(masters[0]) if len(masters) == 1 else repr(masters)
    source = (
        "from quantem.widget import load, Show4DSTEM\n"
        f"Show4DSTEM(load({arg}, det_bin={args.det_bin}))"
    )
    nb = {
        "cells": [
            {"cell_type": "markdown", "id": "title", "metadata": {}, "source": [f"# {label}\n", f"\n{len(masters)} master(s), detector bin {args.det_bin}."]},
            {"cell_type": "code", "id": "viewer", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = _out_dir(args.out) / f"{label}.ipynb"
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
    if no_open or headless or shutil.which("jupyter") is None:
        print(f"wrote {notebook}")
        if headless:
            print(f"  open it from your Mac:  mj jupyter cuda-env quantem   (then open {notebook.name})")
        elif shutil.which("jupyter") is None:
            print("  jupyter not found; install it or open the notebook in your editor")
        return
    print(f"launching jupyter lab on {notebook}")
    subprocess.Popen(["jupyter", "lab", str(notebook)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _master_to_binned_numpy(master: str, det_bin: int):
    """Load one master with detector binning and return a mean-binned 4D numpy array
    ``(scan_row, scan_col, det_row, det_col)``. Binning happens at LOAD time (so the
    full 19 GB stack never materializes - fits a laptop), and since the loader
    integer-SUMS over det_bin^2 we divide by that to get the MEAN, which keeps values
    in the raw range so the uint8 pack never clips. Works on CUDA / MPS (zero-copy
    ChunkedFrames, materialized via its chunks) / CPU."""
    import numpy as np
    import torch
    from quantem.widget import load
    result = load(master, det_bin=det_bin)
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

    Each master is loaded with detector binning (``--bin``, default 8) so the stack
    fits a laptop and packs inline as a self-contained file. ``--combined`` instead
    stacks every master into one 5D viewer (a bslz4 companion folder + a local
    serve, since file:// cannot fetch the companion)."""
    import numpy as np
    from quantem.widget import Show4DSTEM, load
    out_dir = _out_dir(args.out)
    if args.combined and len(masters) > 1:
        # Stack the masters into one 5D numpy array and pass THAT to the viewer. A
        # 5D array routes to the universal Show4DSTEM (which has the offline
        # multi-volume WebGPU frame-flip), not the MacBook live-Metal viewer (whose
        # offline export can't switch volumes kernel-lessly).
        volumes = [_master_to_binned_numpy(m, args.det_bin) for m in masters]
        stack = np.stack(volumes, axis=0)
        data_url = out_dir / "widget-data"
        widget = Show4DSTEM(
            stack, backend="web", offline_codec="bslz4", data_url=str(data_url),
            frame_dim_label="Dataset",
            frame_labels=[pathlib.Path(m).stem.replace("_master", "") for m in masters],
        )
        out = out_dir / f"{label}_combined.html"
        widget.export_html(str(out), title=args.title)
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
            arr = _master_to_binned_numpy(master, args.det_bin)
            widget = Show4DSTEM(arr, backend="web")
            out = out_dir / f"{stem}.html"
            widget.export_html(str(out), title=args.title or stem)
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
        handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=directory, **k)
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
