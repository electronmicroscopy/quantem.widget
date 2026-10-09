"""``quantem`` command-line interface for the two jobs that need no notebook.

    quantem show4dstem ./masters/                       # *_master.h5 -> live Show4DSTEM notebook
    quantem show4dstem ./masters/ --backend webgpu --html   # HDF5-backed WebGPU browser folder
    quantem show4dstem a_master.h5 b_master.h5 --html   # packed offline HTML per master
    quantem github tutorial_github.ipynb --no-execute   # GitHub-displayable notebook copy

``show4dstem`` orchestrates master discovery and loading (quantem.gpu when it
is installed, else the widget's dense reader), the ``Show4DSTEM`` widget and its
export helpers; WebGPU HTML keeps the compressed
HDF5 family on disk so Chrome range-fetches and decompresses chunks itself.
``github`` strips offline widget state from a notebook and embeds compressed
pictures of each widget so GitHub's notebook preview can display it.
"""
import argparse
import base64
import copy
import functools
import http.server
import json
import os
import pathlib
import re
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import webbrowser
from io import BytesIO


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``quantem`` console script. Parse args, dispatch to a
    subcommand, return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="quantem",
        description="Open 4D-STEM masters in Show4DSTEM, or make a widget notebook GitHub-displayable.",
    )
    # A missing command is a usage error (exit 2), as for quantem-gpu.
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_show4dstem_args(subparsers.add_parser(
        "show4dstem",
        help="Render 4D-STEM master(s) as Show4DSTEM (live notebook, or --html)."))
    # `github` shrinks a widget notebook to a form GitHub can display: drop the heavy offline
    # widget-state, keep the auto-snapshot widget render (re-encoded JPEG) + print outputs.
    _add_github_args(subparsers.add_parser(
        "github", help="Make a widget notebook GitHub-displayable (strip offline state, snapshots to JPEG)."))
    args = parser.parse_args(argv)
    # What a user can fix (a path, an argument, a missing package, too little memory) prints one
    # line and exits 1; anything else is a bug and keeps its traceback.
    try:
        if args.command == "github":
            return _prepare_github(args)
        return _show4dstem(args)
    except (OSError, ValueError, ImportError, MemoryError) as error:
        if isinstance(error, ImportError) and (error.name or "").startswith("quantem.widget"):
            raise  # a broken import inside this package, not a missing install
        print(f"quantem: {' '.join(str(error).split()) or type(error).__name__}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
_WIDGET_CELL = ("Show2D(", "Show3D(", "Show4DSTEM(", "Show3DSlices(")
_WIDGET_STATE_MIME = "application/vnd.jupyter.widget-state+json"
_WIDGET_VIEW_MIME = "application/vnd.jupyter.widget-view+json"


def _add_github_args(parser: argparse.ArgumentParser) -> None:
    """Attach options for the ``github`` subcommand."""
    parser.add_argument("path", help="The .ipynb to make GitHub-displayable (edited in place).")
    parser.add_argument("--no-execute", action="store_true",
                        help="Use the notebook's existing outputs instead of re-running it.")
    parser.add_argument("--quality", type=int, default=92,
                        help="JPEG quality for the embedded renders (default 92).")
    parser.add_argument("--max-width", type=int, default=1200,
                        help="Maximum embedded UI width in pixels (default 1200).")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-cell execution timeout in seconds (default 600).")


def _strip_state(nb: dict) -> None:
    """Drop the heavy offline live-widget manager-state + the dead widget-view output refs."""
    nb.get("metadata", {}).pop("widgets", None)
    for cell in nb.get("cells", []):
        kept = []
        for out in cell.get("outputs", []):
            (out.get("data") or {}).pop(_WIDGET_VIEW_MIME, None)
            if out.get("output_type") in {"display_data", "execute_result"} and not (
                out.get("data") or {}
            ):
                continue
            kept.append(out)
        if "outputs" in cell:
            cell["outputs"] = kept


def _embed_jpeg(
    cell: dict,
    png_or_jpeg: bytes,
    quality: int,
    max_width: int = 1200,
) -> bool:
    """Replace a cell's visual output with one JPEG.

    Widget outputs usually have only ``application/vnd.jupyter.widget-view+json``,
    not an existing ``image/*`` slot.  For GitHub display we must add a normal
    image output before stripping the widget MIME bundle.
    """
    from PIL import Image
    encoded, width = _encode_jpeg(Image.open(BytesIO(png_or_jpeg)).convert("RGB"), quality, max_width)
    github_metadata = {"github_full_ui": True, "github_quality": quality, "github_width": width}
    for out in cell.get("outputs", []):
        data = out.get("data")
        if data and (_has_image(data) or _WIDGET_VIEW_MIME in data):
            for key in [key for key in data if key.startswith("image/")]:
                del data[key]
            data["image/jpeg"] = encoded
            out.setdefault("metadata", {}).setdefault("quantem.widget", {}).update(github_metadata)
            return True
    cell.setdefault("outputs", []).append({
        "output_type": "display_data",
        "metadata": {"quantem.widget": github_metadata},
        "data": {"image/jpeg": encoded},
    })
    return True


def _encode_jpeg(image, quality: int, max_width: int) -> tuple[str, int]:
    """An RGB PIL image as base64 JPEG no wider than ``max_width`` (0 keeps its width), and the saved width.

    GitHub stops rendering a notebook above a few MB, so every embedded
    picture is a JPEG at a readable width rather than a full-size PNG.
    """
    from PIL import Image
    if max_width > 0 and image.width > max_width:
        height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), image.width


def _has_image(data: dict) -> bool:
    """Whether one output's MIME bundle holds an image GitHub can render."""
    return any(key.startswith("image/") for key in data)


def _quantem_metadata(out: dict) -> dict:
    """The ``quantem.widget`` metadata of one notebook output, {} when it has none."""
    return (out.get("metadata") or {}).get("quantem.widget") or {}


def _cell_has_marked_image(cell: dict, marker: str) -> bool:
    """Whether one of the cell's image outputs carries the ``quantem.widget`` metadata flag ``marker``."""
    return any(
        _has_image(out.get("data") or {}) and _quantem_metadata(out).get(marker) is True
        for out in cell.get("outputs", [])
    )


def _cell_has_image_output(cell: dict) -> bool:
    """Return true when a notebook cell already has a GitHub-renderable image."""
    return any(_has_image(out.get("data") or {}) for out in cell.get("outputs", []))


def _cell_has_widget_view_output(cell: dict) -> bool:
    """Return true when a notebook cell still depends on live widget MIME output."""
    return any(_WIDGET_VIEW_MIME in (out.get("data") or {}) for out in cell.get("outputs", []))


def _cell_has_full_ui_output(cell: dict) -> bool:
    """Return true when ``quantem github`` already embedded the full widget UI."""
    return _cell_has_marked_image(cell, "github_full_ui")


def _cell_has_static_preview_output(cell: dict) -> bool:
    """Return true when a verified static scientific preview was embedded."""
    return _cell_has_marked_image(cell, "github_static_preview")


def _github_widget_cells(nb: dict) -> list[dict]:
    """Find widget cells from runtime output first, with source as a fallback.

    Public APIs such as ``drift.show()`` return QuantEM widgets without naming
    ``Show2D`` in notebook source.  Runtime widget MIME is therefore the
    authoritative signal after execution.  The source check retains support
    for older or hand-edited notebooks, and the metadata marker recognizes a
    notebook already prepared by this command.
    """
    return [
        cell
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
        and (
            _cell_has_widget_view_output(cell)
            or _cell_has_full_ui_output(cell)
            or _cell_has_static_preview_output(cell)
            or any(widget in "".join(cell.get("source", [])) for widget in _WIDGET_CELL)
        )
    ]


def _github_capture_cells(nb: dict) -> list[dict]:
    """Return widget cells that still need a browser-captured full-UI image."""
    return [
        cell
        for cell in _github_widget_cells(nb)
        if not _cell_has_full_ui_output(cell)
        and not _cell_has_static_preview_output(cell)
        and (
            _cell_has_widget_view_output(cell)
            or not _cell_has_image_output(cell)
        )
    ]


def _image_has_scientific_pixels(image_bytes: bytes) -> bool:
    """Reject effectively blank canvas captures while tolerating any colormap.

    The full widget screenshot contains controls, labels, and borders, so it can
    look nonblank even when WebGPU failed to present the scientific raster. This
    helper is intentionally run on each canvas screenshot, not the surrounding
    UI. A nearly uniform canvas is not useful evidence and falls back to the
    widget's Python-rendered notebook preview.
    """
    from io import BytesIO
    from PIL import Image, ImageStat

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((256, 256), Image.Resampling.BILINEAR)
    if image.width < 2 or image.height < 2:
        return False

    # A range gate catches flat white/black captures. The standard-deviation
    # and dominant-color gates reject canvases containing only a few
    # antialiasing, resize-handle, or overlay pixels.
    channel_range = max(high - low for low, high in image.getextrema())
    channel_stddev = max(ImageStat.Stat(image).stddev)
    palette = image.quantize(colors=32)
    counts = palette.getcolors(maxcolors=32) or []
    dominant_fraction = max(
        (count for count, _ in counts), default=image.width * image.height
    )
    dominant_fraction /= image.width * image.height
    return (
        channel_range >= 12
        and channel_stddev >= 2.0
        and dominant_fraction <= 0.97
    )


def _promote_static_fallback(cell: dict) -> bool:
    """Mark the widget's Python-rendered sibling as the GitHub preview."""
    for out in cell.get("outputs", []):
        if _quantem_metadata(out).get("static_fallback") is not True:
            continue
        if not _has_image(out.get("data") or {}):
            continue
        quantem_metadata = out.setdefault("metadata", {}).setdefault(
            "quantem.widget", {}
        )
        quantem_metadata.pop("static_fallback", None)
        quantem_metadata["github_static_preview"] = True
        return True
    return False


def _widget_view_model_ids(cell: dict) -> list[str]:
    """Return root widget model IDs referenced by one output cell."""
    model_ids = []
    for out in cell.get("outputs", []):
        view = (out.get("data") or {}).get(_WIDGET_VIEW_MIME)
        model_id = view.get("model_id") if isinstance(view, dict) else None
        if model_id and model_id not in model_ids:
            model_ids.append(model_id)
    return model_ids


def _widget_model_closure(state: dict, roots: list[str]) -> set[str]:
    """Return each root model and every ``IPY_MODEL_`` dependency it references."""
    found: set[str] = set()
    pending = list(roots)

    def references(value):
        """Every ``IPY_MODEL_`` id inside one model state, at any depth."""
        if isinstance(value, str) and value.startswith("IPY_MODEL_"):
            yield value.removeprefix("IPY_MODEL_")
        elif isinstance(value, dict):
            for item in value.values():
                yield from references(item)
        elif isinstance(value, list):
            for item in value:
                yield from references(item)

    while pending:
        model_id = pending.pop()
        if model_id in found:
            continue
        if model_id not in state:
            raise ValueError(f"widget model {model_id!r} is absent from notebook state")
        found.add(model_id)
        pending.extend(ref for ref in references(state[model_id]) if ref not in found)
    return found


def _widget_capture_notebook(nb: dict, cell: dict) -> dict:
    """Build a minimal notebook containing one live widget and its dependencies.

    A scientific notebook can contain hundreds of megabytes of state per widget.
    Rendering every model into one HTML document can exceed the browser's JSON
    parser limit even though each widget renders correctly on its own.  This
    temporary notebook keeps exactly the state required by one output view.
    """
    widget_payload = (
        nb.get("metadata", {}).get("widgets", {}).get(_WIDGET_STATE_MIME)
    )
    state = widget_payload.get("state") if isinstance(widget_payload, dict) else None
    roots = _widget_view_model_ids(cell)
    if not roots:
        raise ValueError("widget output has no model_id to capture")
    if not isinstance(state, dict):
        raise ValueError("notebook has no saved widget state; execute it before capture")
    keep = _widget_model_closure(state, roots)

    capture_cell = copy.deepcopy(cell)
    capture_cell["source"] = []
    capture_cell["outputs"] = [
        {
            "output_type": out.get("output_type", "display_data"),
            "metadata": {},
            "data": {_WIDGET_VIEW_MIME: copy.deepcopy((out.get("data") or {})[_WIDGET_VIEW_MIME])},
        }
        for out in cell.get("outputs", [])
        if _WIDGET_VIEW_MIME in (out.get("data") or {})
    ]
    metadata = {
        key: copy.deepcopy(value)
        for key, value in nb.get("metadata", {}).items()
        if key != "widgets"
    }
    capture_payload = {
        key: copy.deepcopy(value)
        for key, value in widget_payload.items()
        if key != "state"
    }
    capture_payload["state"] = {
        model_id: copy.deepcopy(state[model_id]) for model_id in keep
    }
    metadata["widgets"] = {_WIDGET_STATE_MIME: capture_payload}
    return {
        "cells": [capture_cell],
        "metadata": metadata,
        "nbformat": nb.get("nbformat", 4),
        "nbformat_minor": nb.get("nbformat_minor", 5),
    }


def _capture_notebook_widget_uis(
    notebook: pathlib.Path,
    nb: dict,
    capture_cells: list[dict],
) -> list[bytes | None]:
    """Render and capture widget cells independently to bound temporary HTML size."""
    shots: list[bytes | None] = []
    with tempfile.TemporaryDirectory(
        prefix=f".{notebook.stem}-github-ui-", dir=notebook.parent
    ) as folder:
        temporary = pathlib.Path(folder)
        for index, cell in enumerate(capture_cells, start=1):
            capture_nb = temporary / f"widget-{index:02d}.ipynb"
            capture_nb.write_text(
                json.dumps(_widget_capture_notebook(nb, cell)), encoding="utf-8"
            )
            result = subprocess.run(
                [
                    "jupyter", "nbconvert", "--to", "html", str(capture_nb),
                    "--output-dir", str(temporary), "--output", capture_nb.stem,
                ]
            )
            if result.returncode != 0:
                raise ValueError(
                    f"nbconvert failed while preparing widget UI {index}"
                )
            html = temporary / f"{capture_nb.stem}.html"
            print(
                f"  widget {index}/{len(capture_cells)} temporary HTML: "
                f"{html.stat().st_size / 1e6:.1f} MB"
            )
            captured = _capture_full_ui(html, 1)
            if len(captured) != 1:
                raise ValueError(
                    f"captured {len(captured)} UI screenshot(s) for widget {index}"
                )
            shots.extend(captured)
    return shots


def _recompress_full_ui_outputs(nb: dict, quality: int, max_width: int) -> int:
    """Re-encode previously prepared full-UI images at the requested quality."""
    changed = 0
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            metadata = _quantem_metadata(out)
            data = out.get("data") or {}
            image = data.get("image/jpeg")
            if (
                metadata.get("github_full_ui") is True
                and image
                and (
                    metadata.get("github_quality") != quality
                    or metadata.get("github_width") != min(
                        max_width, metadata.get("github_width", max_width + 1)
                    )
                )
            ):
                _embed_jpeg(cell, base64.b64decode(image), quality, max_width)
                changed += 1
                break
    return changed


def _prune_widget_fallbacks(nb: dict) -> int:
    """Remove redundant auto-snapshots after a complete UI capture exists."""
    removed = 0
    for cell in _github_widget_cells(nb):
        if not _cell_has_full_ui_output(cell):
            continue
        kept = []
        for out in cell.get("outputs", []):
            quantem_metadata = _quantem_metadata(out)
            if (
                quantem_metadata.get("static_fallback") is True
                and quantem_metadata.get("github_full_ui") is not True
            ):
                removed += 1
                continue
            if quantem_metadata.get("github_full_ui") is True:
                data = out.get("data") or {}
                image = data.get("image/jpeg")
                out["data"] = {"image/jpeg": image} if image else {}
            kept.append(out)
        cell["outputs"] = kept
    return removed


def _validate_github_widget_outputs(widget_cells: list[dict]) -> None:
    """Require one nonblank GitHub preview and no duplicate widget render."""

    problems = []
    for index, cell in enumerate(widget_cells, start=1):
        full_ui = []
        static_previews = []
        fallbacks = 0
        widget_views = 0
        for out in cell.get("outputs", []):
            data = out.get("data") or {}
            quantem_metadata = _quantem_metadata(out)
            if quantem_metadata.get("github_full_ui") is True and _has_image(data):
                full_ui.append(out)
            if quantem_metadata.get("github_static_preview") is True and _has_image(data):
                static_previews.append(out)
            if quantem_metadata.get("static_fallback") is True:
                fallbacks += 1
            if _WIDGET_VIEW_MIME in data:
                widget_views += 1
        if len(full_ui) + len(static_previews) != 1 or fallbacks or widget_views:
            problems.append(
                f"cell {index}: full_ui={len(full_ui)}, "
                f"static_previews={len(static_previews)}, "
                f"fallbacks={fallbacks}, widget_views={widget_views}"
            )
    if problems:
        raise ValueError(
            "GitHub notebook preparation requires exactly one nonblank widget "
            "preview per widget cell and no fallback duplicates: "
            + "; ".join(problems)
        )


def _compress_large_raster_outputs(
    nb: dict,
    quality: int,
    max_width: int,
    threshold: int = 500_000,
) -> int:
    """JPEG-encode large ordinary PNG outputs while retaining readable dimensions."""
    from PIL import Image

    changed = 0
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if _quantem_metadata(out).get("github_full_ui") is True:
                continue
            data = out.get("data") or {}
            encoded = data.get("image/png")
            if not encoded or len(encoded) <= threshold:
                continue
            image = Image.open(BytesIO(base64.b64decode(encoded)))
            if image.mode in {"RGBA", "LA"}:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                image = background.convert("RGB")
            else:
                image = image.convert("RGB")
            jpeg, width = _encode_jpeg(image, quality, max_width)
            data.pop("image/png")
            data["image/jpeg"] = jpeg
            metadata = out.setdefault("metadata", {}).setdefault("quantem.widget", {})
            metadata["github_compressed_from"] = "image/png"
            metadata["github_quality"] = quality
            metadata["github_width"] = width
            changed += 1
    return changed


def _capture_full_ui(html: pathlib.Path, n_expected: int) -> list[bytes | None]:
    """Screenshot each widget's FULL UI (toolbar + toggles + panels + histograms) from the
    rendered live-widget HTML, deterministically, via Playwright on the real GPU. The widget
    UI is React+MUI+WebGPU, so a browser engine is required; Playwright manages the lifecycle
    (waits for mount + paint) and ``locator.screenshot`` grabs each widget element exactly."""
    # Point Vulkan at the NVIDIA driver when it exists, so WebGPU uses the GPU rather than a software adapter.
    nvidia_icd = pathlib.Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if nvidia_icd.is_file():
        os.environ.setdefault("VK_ICD_FILENAMES", str(nvidia_icd))
    # A headed browser needs a display; over SSH without one, Chromium renders headless.
    headless = sys.platform != "darwin" and not os.environ.get("DISPLAY")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError("quantem github needs Playwright to screenshot each widget: pip install playwright") from exc
    shots: list[bytes | None] = []
    launch_kwargs = {}
    for candidate in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        executable = shutil.which(candidate)
        if executable:
            launch_kwargs["executable_path"] = executable
            print(f"  browser executable: {executable}")
            break
    with sync_playwright() as playwright:
        # No --use-angle=vulkan: on a real X display it leaves Canvas2D panels black.
        browser = playwright.chromium.launch(headless=headless, args=[
            "--enable-unsafe-webgpu", "--enable-features=Vulkan",
            "--ignore-gpu-blocklist", "--disable-gpu-sandbox", "--no-sandbox"], **launch_kwargs)
        page = browser.new_page(viewport={"width": 1300, "height": 2400}, device_scale_factor=2)
        browser_errors = []
        page.on("pageerror", lambda error: browser_errors.append(f"page: {error}"))
        page.on(
            "console",
            lambda message: browser_errors.append(f"console: {message.text}")
            if message.type == "error"
            else None,
        )
        page.goto(html.as_uri(), wait_until="load", timeout=180000)
        # Large scientific notebooks can carry hundreds of megabytes of
        # temporary widget state. Wait for actual canvases instead of assuming
        # that every browser mounts and paints them within a fixed 13 seconds.
        canvas_count = 0
        for _ in range(120):
            canvas_count = page.locator(".jp-OutputArea-output canvas").count()
            if canvas_count >= n_expected:
                break
            page.wait_for_timeout(1000)
        if canvas_count < n_expected:
            for error in browser_errors[-10:]:
                print(f"  browser error: {error}")
            print(f"  mounted canvases: {canvas_count}/{n_expected}")
        else:
            page.wait_for_timeout(2000)  # allow the first WebGPU frame to present
        architecture = page.evaluate("async()=>{const a=await navigator.gpu?.requestAdapter();"
                                     "return a?(a.info?.architecture||'?'):'none';}")
        print(f"  GPU adapter: {architecture}")
        if architecture == "swiftshader":
            print("  warning: WebGPU reported SwiftShader; continuing because GitHub snapshots only need pixels")
        outputs = page.locator(".jp-OutputArea-output")
        for output_index in range(outputs.count()):
            output = outputs.nth(output_index)
            if output.locator("canvas").count() > 0:
                output.scroll_into_view_if_needed()
                page.wait_for_timeout(700)
                canvases = output.locator("canvas:visible")
                canvas_entries = []
                for index in range(canvases.count()):
                    canvas = canvases.nth(index)
                    box = canvas.bounding_box()
                    if box:
                        canvas_entries.append((canvas, box["width"] * box["height"]))
                largest_area = max((area for _, area in canvas_entries), default=0.0)
                scientific_canvases = [
                    canvas
                    for canvas, area in canvas_entries
                    if largest_area and area >= 0.25 * largest_area
                ]
                canvas_has_pixels = any(
                    _image_has_scientific_pixels(canvas.screenshot())
                    for canvas in scientific_canvases
                )
                if canvas_has_pixels:
                    shots.append(output.screenshot())
                else:
                    print(
                        "  warning: widget canvas contains no scientific pixels; "
                        "using its static notebook preview"
                    )
                    shots.append(None)
        browser.close()
    if len(shots) != n_expected:
        print(f"  warning: captured {len(shots)} widget UIs for {n_expected} widget cells")
    return shots


def _prepare_github(args: argparse.Namespace) -> int:
    """Make a widget notebook display correctly on GitHub and in VS Code.

    Prefer each widget's Python-rendered scientific preview. It is independent
    of WebGPU presentation and therefore cannot silently publish a white or
    black browser canvas. Widgets without a native preview use a browser-captured
    full UI only after the canvas passes a scientific-pixel check. Offline widget
    state is removed because GitHub cannot hydrate it.

    Re-running the source notebook or ``jupyter nbconvert --to html`` remains the path to
    the interactive widget.
    """
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
                           str(notebook), f"--ExecutePreprocessor.timeout={args.timeout}",
                           "--ExecutePreprocessor.store_widget_state=True"]).returncode != 0:
            raise ValueError("nbconvert --execute failed (see output above)")
    nb = json.loads(notebook.read_text())
    widget_cells = _github_widget_cells(nb)
    capture_cells = _github_capture_cells(nb)
    max_width = args.max_width
    recompressed = _recompress_full_ui_outputs(nb, args.quality, max_width)
    static_count = sum(_promote_static_fallback(cell) for cell in capture_cells)
    capture_cells = [
        cell for cell in capture_cells if not _cell_has_static_preview_output(cell)
    ]
    if capture_cells:
        try:
            print(f"capturing {len(capture_cells)} widget UI(s) on the GPU ...")
            shots = _capture_notebook_widget_uis(notebook, nb, capture_cells)
            full_ui_count = 0
            for cell, shot in zip(capture_cells, shots):
                if shot is not None:
                    _embed_jpeg(cell, shot, args.quality, max_width)
                    full_ui_count += 1
                else:
                    raise ValueError(
                        "widget canvas was blank and the widget provided no static "
                        "scientific preview; rerun after fixing its preview renderer"
                    )
            mode = (
                f"{full_ui_count} full-UI screenshot(s), "
                f"{static_count} verified static preview(s)"
            )
        except (ImportError, RuntimeError, OSError) as error:
            raise ValueError(
                "full-UI capture needs Playwright + a real GPU (NVIDIA Vulkan ICD + a display): "
                f"{error}") from error
    elif widget_cells:
        mode = (
            f"{static_count} verified static preview(s)"
            if static_count
            else f"{len(widget_cells)} existing image output(s)"
        )
    else:
        mode = "no widget cells - state stripped only"
    fallbacks = _prune_widget_fallbacks(nb)
    rasters = _compress_large_raster_outputs(nb, args.quality, max_width)
    _strip_state(nb)
    _validate_github_widget_outputs(widget_cells)
    notebook.write_text(json.dumps(nb, indent=1))
    after = notebook.stat().st_size
    print(f"github-ready: {notebook.name}  {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB"
          f"  ({mode}, JPEG q{args.quality}, max {max_width}px, offline state stripped)")
    if recompressed:
        print(f"  re-encoded {recompressed} existing full-UI image(s) at JPEG q{args.quality}")
    if fallbacks:
        print(f"  removed {fallbacks} redundant widget fallback image(s)")
    if rasters:
        print(f"  compressed {rasters} large raster output(s) for repository display")
    if after > 5e6:
        print("  warning: still > 5 MB - GitHub may not render. Lower --quality or the widget's size=.")
    return 0


# ---------------------------------------------------------------------------
def _add_show4dstem_args(parser: argparse.ArgumentParser) -> None:
    """Attach the ``show4dstem`` options."""

    parser.add_argument("path", nargs="+",
                        help="A 4D-STEM master, a folder of masters, or several master files.")
    parser.add_argument("--out", default=None, help="Output folder (default ~/Downloads).")
    parser.add_argument("--no-open", action="store_true", help="Write the file(s) but do not launch anything.")
    parser.add_argument("--serve", action="store_true",
                        help="Open via a local HTTP server even for self-contained files (tunnelable URL).")
    parser.add_argument("--title", default=None, help="Viewer page title.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose progress.")
    parser.add_argument("--bin", type=int, default=None, dest="det_bin",
                        help=(
                            "--html: detector binning factor (mean of each block). "
                            "Default 1, full detector sampling."
                        ))
    parser.add_argument("--max-gb", type=float, default=None, dest="max_gb",
                        help=("Largest dense array to read when quantem.gpu or a GPU is missing, in GB. "
                              "Default: 80%% of the available memory. Nothing is binned or cropped to fit."))
    parser.add_argument("--count", type=int, default=None,
                        help="Require and load this many compatible masters from the input.")
    parser.add_argument("--html", action="store_true",
                        help="Export a standalone offline-WebGPU HTML instead of a live notebook.")
    parser.add_argument("--watch", action="store_true",
                        help="Folder: write a live viewer notebook that appends new masters.")
    parser.add_argument("--watch-interval", type=float, default=2.0,
                        help="Polling interval in seconds for --watch live folders (default 2).")
    parser.add_argument("--dtype", default="auto", choices=("auto", "u8", "uint8", "u16", "uint16"),
                        help=("--html: packed dtype of the exported counts: auto (default; uint8 when every "
                              "count fits, else uint16, so nothing clips), u8 or u16."))
    parser.add_argument("--scan-size", type=int, default=None,
                        help="--watch: only include masters with this square scan size.")
    parser.add_argument("--backend", default="auto",
                        choices=("auto", "cuda", "mps", "webgpu"),
                        help="Show4DSTEM backend. Use webgpu with --html.")


def _show4dstem(args: argparse.Namespace) -> int:
    """Resolve the master(s), render Show4DSTEM, open the result.

    4D-STEM renders to a live Jupyter notebook by default (full real-time WebGPU, no
    large file); ``--html`` instead exports the self-contained offline-WebGPU HTML.
    One path can be a master file or a folder; several paths are taken as a list of
    masters and become one comparison viewer (the multi-tilt case)."""
    if args.serve and not args.html:
        raise ValueError("--serve opens an --html export over HTTP; add --html.")
    paths = [pathlib.Path(raw).expanduser().resolve() for raw in args.path]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("path does not exist: " + ", ".join(missing))
    if len(paths) > 1:
        if args.watch:
            raise ValueError("--watch requires one folder path, not multiple explicit paths.")
        masters = _select_show4dstem_masters([str(path) for path in paths], args)
        return _do_4dstem(masters, f"{len(masters)}_datasets", args, source_path=None)
    path = paths[0]
    if args.watch:
        if args.html:
            raise ValueError("--watch writes a live notebook; omit --html.")
        if not path.is_dir():
            raise ValueError("--watch requires a folder path containing *_master.h5 files.")
    from quantem.widget.adapters import gpu as gpu_adapter
    from quantem.widget.show4dstem.reader import find_masters

    discover = gpu_adapter.discover_masters if gpu_adapter.available() else find_masters
    masters = [str(path)] if path.is_file() else discover(path)
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


def _do_4dstem(
    masters: list[str],
    label: str,
    args: argparse.Namespace,
    *,
    source_path: pathlib.Path | None = None,
) -> int:
    """Dispatch 4D-STEM master(s) to either a live notebook (default) or an offline
    HTML (``--html``), then launch/open it. One master opens alone; many open as a
    dataset comparison (the multi-tilt case)."""
    args.det_bin = _effective_det_bin(args, default=1)
    backend = _normalise_show4dstem_backend(args.backend)
    if args.html:
        if backend == "webgpu":
            output = _render_4dstem_webgpu_h5(masters, label, args)
            _open_show4dstem_command(output.parent / "Show4DSTEM.command", no_open=args.no_open)
            return 0
        outputs = _render_4dstem(masters, args)
        if len(outputs) > 1:
            print(f"wrote {len(outputs)} HTML files to {outputs[0].parent}")
        _open_html(outputs[0], serve=args.serve, no_open=args.no_open)
        return 0
    if backend == "webgpu":
        raise ValueError("Show4DSTEM --backend webgpu writes browser HTML; add --html.")
    notebook = _render_4dstem_notebook(masters, label, args, source_path=source_path)
    _launch_notebook(notebook, no_open=args.no_open)
    return 0


def _select_show4dstem_masters(masters: list[str], args: argparse.Namespace) -> list[str]:
    """Apply Show4DSTEM ``--count`` as an exact compatible-master request."""

    count = args.count
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

    def format_coordinate(value: str) -> str:
        """Signed, at most two decimals, always one: ``+1.0``, ``-0.25``."""
        text = f"{float(value):+.2f}".rstrip("0").rstrip(".")
        return text if "." in text else f"{text}.0"

    return f"Tilt ({format_coordinate(match['x'])}, {format_coordinate(match['y'])})"


def _normalise_show4dstem_backend(value: str | None) -> str | None:
    """Return the Show4DSTEM backend token used by generated notebooks/exports."""

    token = str(value or "auto").strip().lower()
    if token in {"", "auto"}:
        return None
    if token in {"webgpu", "cuda", "mps"}:
        return token
    raise ValueError(f"unsupported Show4DSTEM backend {value!r}")


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


def _render_4dstem_notebook(
    masters: list[str],
    label: str,
    args: argparse.Namespace,
    *,
    source_path: pathlib.Path | None = None,
) -> pathlib.Path:
    """Write a live Jupyter notebook that loads the 4D-STEM master(s) and opens a
    kernel-backed ``Show4DSTEM`` (no baked HTML). Each master stays encoded on the
    GPU at full detector sampling; many open as a dataset comparison (the
    multi-tilt case). The notebook is the editable, real-use surface; ``--html`` is
    the share artifact."""
    backend = _normalise_show4dstem_backend(args.backend)
    if int(args.det_bin) != 1:
        raise ValueError(
            "A live Show4DSTEM notebook keeps full detector sampling in encoded GPU "
            "storage; --bin applies to --html exports only."
        )
    backend_label = backend or "auto"
    backend_arg = "'auto'" if backend is None else repr(backend)
    print(f"{len(masters)} master(s), backend {backend_label} -> Show4DSTEM (live notebook)")
    if source_path is not None and source_path.is_dir():
        source = (
            "from quantem.widget import Show4DSTEM\n"
            "\n"
            f"folder = {str(source_path)!r}\n"
            "viewer = Show4DSTEM.from_folder(\n"
            "    folder,\n"
            f"    backend={backend_arg},\n"
            f"    max_masters={len(masters)},\n"
            f"    min_masters={len(masters)},\n"
            "    watch=False,\n"
            "    verbose=True,\n"
            ")\n"
            "viewer\n"
        )
    elif len(masters) == 1:
        # a path opens through quantem.gpu when it is installed, else densely
        source = (
            "from quantem.widget import Show4DSTEM\n"
            "\n"
            f"Show4DSTEM({masters[0]!r})\n"
        )
    else:
        # one Dataset4dstemGPU per master on a GPU, one quantem core Dataset4dstem each elsewhere
        source = (
            "from quantem.widget import Show4DSTEM, read_4dstem\n"
            "\n"
            f"masters = {masters!r}\n"
            f"data = read_4dstem(masters, device={backend_arg})\n"
            "Show4DSTEM(data)\n"
        )
    title = [f"# {label}\n", f"\n{len(masters)} master(s), backend `{backend_label}`, full detector sampling."]
    return _write_viewer_notebook(_out_dir(args.out) / f"{label}.ipynb", title, "viewer", source)


def _render_4dstem_watch_notebook(folder: pathlib.Path, label: str, args: argparse.Namespace) -> pathlib.Path:
    """Write a live viewer notebook for a 4D-STEM acquisition folder."""
    if _effective_det_bin(args, default=1) != 1:
        raise ValueError(
            "A live folder viewer keeps full detector sampling in encoded GPU "
            "storage; --bin applies to --html exports only."
        )
    backend = _normalise_show4dstem_backend(args.backend)
    backend_arg = "'auto'" if backend is None else repr(backend)
    print(f"{folder.name}: watched folder -> Show4DSTEM over encoded masters")
    scan_size = "None" if args.scan_size is None else str(int(args.scan_size))
    source = (
        "from quantem.widget import Show4DSTEM\n\n"
        "viewer = Show4DSTEM.from_folder(\n"
        f"    {str(folder)!r},\n"
        f"    backend={backend_arg},\n"
        f"    scan_size={scan_size},\n"
        f"    watch=True, watch_interval={float(args.watch_interval)!r},\n"
        ")\nviewer\n"
    )
    title = [
        f"# {label} live Show4DSTEM\n",
        f"\nWatched folder: `{folder}`\n",
        f"\nFull detector sampling; watch interval {args.watch_interval:g}s.",
    ]
    return _write_viewer_notebook(_out_dir(args.out) / f"{label}_live.ipynb", title, "live-viewer", source)


def _write_viewer_notebook(path: pathlib.Path, title: list[str], cell_id: str, source: str) -> pathlib.Path:
    """Write a two-cell notebook, a markdown ``title`` and one code cell that opens the viewer, to ``path``."""
    nb = {
        "cells": [
            {"cell_type": "markdown", "id": "title", "metadata": {}, "source": title},
            {
                "cell_type": "code",
                "id": cell_id,
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
    path.write_text(json.dumps(nb, indent=1))
    return path


def _launch_notebook(notebook: pathlib.Path, *, no_open: bool) -> None:
    """Open the notebook for the user. Locally (a Mac or any box with a display) start
    ``jupyter lab`` on it, which opens the browser. On a headless/remote box a browser
    cannot be reached, so print the path plus the ``mj jupyter`` hint instead (never
    start a server the user cannot see)."""
    headless = sys.platform != "darwin" and not os.environ.get("DISPLAY")
    if no_open or headless:
        print(f"wrote {notebook}")
        if headless:
            print(f"  open it with:  jupyter lab {notebook}")
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
    """Open a generated Show4DSTEM WebGPU folder.

    The ``.command`` launcher is a macOS script, so macOS opens it; elsewhere the
    folder's stdlib Range server is printed as one command, with the page to
    open in Chrome (WebGPU needs a Chromium browser).
    """
    if sys.platform != "darwin":
        root, port = command.parent.resolve(), _free_port(8794)
        print(f"serve it, then open http://127.0.0.1:{port}/ in Chrome:\n"
              f"  python3 {root / '.viewer' / 'serve_range.py'} --root {root} --port {port}")
        return
    if no_open or not command.is_file():
        print(f"wrote {command}")
        return
    subprocess.Popen(["open", str(command)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"opened {command}")


def _free_port(start: int) -> int:
    """First loopback port from ``start`` that nothing listens on, as the macOS launcher advances."""
    import socket

    for port in range(start, start + 100):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def _render_4dstem_webgpu_h5(
    masters: list[str],
    label: str,
    args: argparse.Namespace,
) -> pathlib.Path:
    """Render Show4DSTEM WebGPU HTML over linked H5 data."""

    import numpy as np
    from quantem.widget import Show4DSTEM
    from quantem.widget.show4dstem.export import bundle_master_urls, write_webgpu_bundle
    from quantem.widget.show4dstem.widget import master_contract

    if int(args.det_bin) != 1:
        raise ValueError(
            "Show4DSTEM --backend webgpu uses linked HDF5 files with browser "
            "range reads; keep --bin 1."
        )
    contracts = [master_contract(master) for master in masters]
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
    # the browser decodes the HDF5 counts itself, so "auto" keeps the lossless uint16
    decode_dtype = _show4dstem_export_dtype(args)
    write_webgpu_bundle(widget, out_dir, title=args.title or label,
                        h5_decode_dtype="uint16" if decode_dtype == "auto" else decode_dtype)
    out = out_dir / "index.html"
    print(f"{len(masters)} master(s), backend webgpu, bin 1, dtype {args.dtype} -> {out_dir}")
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


def _link_show4dstem_h5_family(out_dir: pathlib.Path, master: pathlib.Path, label: str) -> None:
    """Symlink a master and same-prefix data chunks under an anonymous label."""

    source_master = master.expanduser().resolve()
    source_prefix = source_master.name[: -len("_master.h5")]
    master_link = out_dir / f"{label}_master.h5"
    _replace_symlink(master_link, source_master)
    for data_file in sorted(source_master.parent.glob(f"{source_prefix}_data_*.h5")):
        data_link = out_dir / data_file.name.replace(source_prefix, label, 1)
        _replace_symlink(data_link, data_file.resolve())


def _replace_symlink(link: pathlib.Path, target: pathlib.Path) -> None:
    """Replace only symlink artifacts; never overwrite a real data file."""

    if link.exists() or link.is_symlink():
        if not link.is_symlink():
            raise ValueError(f"refusing to replace non-symlink artifact file: {link}")
        link.unlink()
    link.symlink_to(target)


def _show4dstem_export_dtype(args: argparse.Namespace) -> str:
    """The ``--dtype`` choice as the export dtype name (float32 analysis stays in a live notebook).

    ``"auto"`` is resolved per master once its counts are known (``_lossless_pack_dtype``), so a
    pre-binned array whose counts exceed 255 is not clipped to a flat uint8 image.
    """
    if args.dtype in {"u8", "uint8"}:
        return "uint8"
    return "uint16" if args.dtype in {"u16", "uint16"} else "auto"


def _lossless_pack_dtype(counts, stem: str) -> str:
    """uint8 when every count fits 0..255, else uint16, saying so when the wider type is needed."""
    largest = float(counts.max())
    if largest <= 255:
        return "uint8"
    print(f"{stem}: counts reach {largest:.0f}, packed as uint16 (uint8 holds 0..255)")
    return "uint16"


_HTML_PAYLOAD_LIMIT = 2 << 30  # packed counts one offline page can hold; 0.6 GB of uint8 made a 0.3 GB page


def _check_html_payload(master: str, det_bin: int, dtype: str) -> None:
    """Refuse a master whose packed counts cannot fit one offline page, before reading it.

    The page embeds the gzip of ``frames x (det_rows / det_bin) x (det_cols / det_bin)``
    packed counts as one base64 string, and a browser string holds about 512 MiB, so a
    payload above ``_HTML_PAYLOAD_LIMIT`` cannot load. Without the check the export reads
    the whole acquisition for minutes and then fails on memory. The message names the
    smallest ``--bin`` whose payload is at most 768 MiB.
    """
    from quantem.widget.show4dstem.reader import master_layout

    layout = master_layout(master)
    if layout is None:
        return
    frames, (det_rows, det_cols), _ = layout
    itemsize = 1 if dtype == "uint8" else 2
    payload = frames * (det_rows // det_bin) * (det_cols // det_bin) * itemsize
    if payload <= _HTML_PAYLOAD_LIMIT:
        return
    fits = [
        factor for factor in range(det_bin + 1, min(det_rows, det_cols) + 1)
        if det_rows % factor == 0 and det_cols % factor == 0
        and frames * (det_rows // factor) * (det_cols // factor) * itemsize <= 3 << 28
    ]
    advice = f"pass --bin {fits[0]}, " if fits else ""
    raise ValueError(
        f"an offline HTML at --bin {det_bin} packs {payload / 1e9:.1f} GB of {dtype} counts, more than "
        f"one browser page can load; {advice}use --backend webgpu to read the HDF5 files in the browser, "
        "or omit --html for a live notebook. Nothing was binned."
    )


def _master_to_binned_numpy(master: str, det_bin: int, max_bytes: int | None = None):
    """Return one master as float32 ``(scan_row, scan_col, det_row, det_col)``.

    ``det_bin > 1`` replaces each ``det_bin`` x ``det_bin`` detector block by its
    mean (exact integer sum, one division), rounded to the nearest count: an offline
    HTML embeds the whole array, and the mean stays in the raw count range so
    uint8/uint16 packing never clips. The acquisition stays encoded on the GPU and
    is read about 256 MiB of scan rows at a time, so the dense cube (19 GB for
    512 x 512 x 192 x 192 uint16) never exists.
    """
    from quantem.widget.adapters import gpu as gpu_adapter
    from quantem.widget.show4dstem.reader import read_dense

    if gpu_adapter.accelerator_ready():
        with gpu_adapter.load_acquisition(master) as acquisition:
            cols = acquisition.shape[1]
            return _bin_scan_rows(
                acquisition.shape, lambda start, stop: acquisition.read(scan_region=(start, stop, 0, cols)), det_bin)
    # no quantem.gpu or no GPU: read the master densely on the host, then bin it in the same
    # scan-row blocks, since the exact int64 block sum of the whole cube would need 4x its size
    values_t = read_dense(master, "cpu", max_bytes=max_bytes, verbose=False)
    return _bin_scan_rows(values_t.shape, lambda start, stop: values_t[start:stop], det_bin)


def _bin_scan_rows(shape, read_rows, det_bin: int):
    """Detector-bin a ``(rows, cols, det_rows, det_cols)`` source about 256 MiB of scan rows at a time.

    ``read_rows(start, stop)`` returns the counts of scan rows ``start:stop``; each block is
    mean-binned (``detector_bin_mean``) and rounded to the nearest count when ``det_bin > 1``,
    so the transient memory is one block, not the whole cube.
    """
    import numpy as np

    from quantem.widget.counts import CHUNK_BYTES, detector_bin_mean

    rows, cols, det_rows, det_cols = shape
    if det_rows % det_bin or det_cols % det_bin:
        raise ValueError(
            f"--bin {det_bin} does not divide the {det_rows} x {det_cols} detector; "
            "choose a factor of both sides."
        )
    binned = np.empty((rows, cols, det_rows // det_bin, det_cols // det_bin), np.float32)
    block_rows = max(1, CHUNK_BYTES // (cols * det_rows * det_cols * 4))
    for row in range(0, rows, block_rows):
        stop = min(rows, row + block_rows)
        block = detector_bin_mean(read_rows(row, stop), det_bin)
        binned[row:stop] = np.round(block) if det_bin > 1 else block
    return binned


def _render_4dstem(masters: list[str], args: argparse.Namespace) -> list[pathlib.Path]:
    """Render each 4D-STEM master as one offline WebGPU Show4DSTEM HTML.

    Each master is loaded with the requested detector binning (``--bin``, default
    1 for full detector sampling) and packed into its own page."""
    from quantem.widget import Show4DSTEM
    out_dir = _out_dir(args.out)
    export_dtype = _show4dstem_export_dtype(args)
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
            # Mean-bin at load (memory-safe: the full 19 GB stack never materializes), so the
            # export does no further binning. "auto" packs at least one byte per count, so the
            # uint8 size is the bound to check before reading.
            _check_html_payload(master, args.det_bin, "uint8" if export_dtype == "auto" else export_dtype)
            binned = _master_to_binned_numpy(
                master, args.det_bin, None if args.max_gb is None else int(args.max_gb * 1e9))
            if args.det_bin > 1:
                # no silent reduction: name the factor and how to keep the full detector
                det_rows, det_cols = binned.shape[2:]
                print(f"{stem}: detector binned {args.det_bin}x{args.det_bin} (mean of each block), "
                      f"{det_rows * args.det_bin}x{det_cols * args.det_bin} -> {det_rows}x{det_cols}; "
                      "pass --bin 1 for the full detector")
            pack_dtype = _lossless_pack_dtype(binned, stem) if export_dtype == "auto" else export_dtype
            widget = Show4DSTEM(binned, backend="webgpu")
            out = out_dir / f"{stem}.html"
            widget.export_html(str(out), title=args.title or stem, dtype=pack_dtype)
            outputs.append(out)
        except (RuntimeError, ValueError, OSError, MemoryError) as error:
            print(f"quantem: skipped {stem}: {error}", file=sys.stderr)
    if not outputs:
        raise ValueError("every master failed to export (see messages above)")
    return outputs


# ---------------------------------------------------------------------------
def _out_dir(out: str | None) -> pathlib.Path:
    """Resolve the output directory (``--out`` or the default ``~/Downloads``)."""
    target = pathlib.Path(out).expanduser() if out else _default_out_dir()
    if target.suffix in {".html", ".ipynb"}:
        # --out is the folder the outputs are written into; a file name would become a folder
        raise ValueError(f"--out is the output folder; {target.name} looks like a file name, pass its folder")
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
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(path.parent))
        httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{port}/{path.name}"
        # flushed: the server blocks below, and a piped log would never show the URL
        print(f"serving {url}  (Ctrl-C to stop)", flush=True)
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
