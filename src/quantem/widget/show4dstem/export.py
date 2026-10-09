"""Standalone HTML from a Show4DSTEM: interactive page, static report, WebGPU folder bundle.

Three artifacts leave the kernel:

* ``interactive``: the 4D counts packed as a gzip ``uint8``/``uint16`` stack
  inside one HTML page; the browser's WebGPU detector kernels run the virtual
  image and ROI pattern with no Python.
* ``report``: PNG virtual-image pages plus one diffraction thumbnail, no raw
  counts, so a folder of many datasets can be reviewed from one small file.
* WebGPU bundle (``quantem show4dstem --backend webgpu --html``): the page sits
  beside the linked ``*_master.h5`` family and range-reads the compressed
  detector chunks itself; a ``Show4DSTEM.command`` starts the local server.
"""

import gzip
import html
import json
import math
import pathlib
import re

import numpy as np
import torch

from quantem.widget.command_launcher import write_command_launcher
from quantem.widget.export import ensure_mobile_viewport, export_slug, write_widget_html
from quantem.widget.show4dstem.detector import PRESET_RADII, preset_mask
from quantem.widget.show4dstem.preview import (
    colormap_rgb,
    display_range,
    png_data_uri,
    scale_values,
)
from quantem.widget.show4dstem.reader import read_pixel_mask

VALID_BINS = (1, 2, 4, 8)
# packed counts one interactive page can hold (as ``quantem show4dstem --html``): the page embeds their
# gzip as one base64 string and a browser string holds about 512 MiB; 0.6 GB of uint8 made a 0.3 GB page
HTML_PAYLOAD_LIMIT = 2 << 30
# a suggested binning must bring the counts under this, so the page loads with room to spare
HTML_PAYLOAD_ADVISED = 3 << 28


def export_options(widget, dtype: str, det_bin: int, scan_bin: int) -> tuple[str, int, int]:
    """Validate the packing choices: integer dtype and bins that divide both shapes."""
    dtype = str(dtype).strip().lower()
    dtype = {"u8": "uint8", "u16": "uint16"}.get(dtype, dtype)
    if dtype not in ("uint8", "uint16"):
        raise ValueError(f"dtype must be 'uint8' or 'uint16', got {dtype!r}")
    det_bin, scan_bin = int(det_bin), int(scan_bin)
    if det_bin not in VALID_BINS:
        raise ValueError(f"det_bin must be 1, 2, 4, or 8, got {det_bin}")
    if scan_bin not in VALID_BINS:
        raise ValueError(f"scan_bin must be 1, 2, 4, or 8, got {scan_bin}")
    if widget.det_rows % det_bin or widget.det_cols % det_bin:
        raise ValueError(
            f"Detector shape {widget.det_rows}x{widget.det_cols} is not divisible by det_bin={det_bin}"
        )
    if widget.shape_rows % scan_bin or widget.shape_cols % scan_bin:
        raise ValueError(
            f"Scan shape {widget.shape_rows}x{widget.shape_cols} is not divisible by scan_bin={scan_bin}"
        )
    return dtype, det_bin, scan_bin


def check_html_payload(n_frames: int, scan_shape: tuple[int, int], det_shape: tuple[int, int], dtype: str,
                       det_bin: int, scan_bin: int) -> None:
    """Refuse an interactive export whose packed counts no page can load, before any frame is read.

    The page packs ``n_frames x (scan_rows / scan_bin) x (scan_cols / scan_bin) x (det_rows /
    det_bin) x (det_cols / det_bin)`` counts of ``dtype``. Without the check a 512 x 512 x 192 x 192
    acquisition (9.7 GB of uint8) is read and packed in host memory first, which ran a shared
    machine into its out-of-memory killer, for a page that could not have loaded. The message
    names the least binning whose counts fit; nothing is binned unasked.
    """
    def payload(det: int, scan: int) -> int:
        return (n_frames * (scan_shape[0] // scan) * (scan_shape[1] // scan)
                * (det_shape[0] // det) * (det_shape[1] // det) * np.dtype(dtype).itemsize)

    size = payload(det_bin, scan_bin)
    if size <= HTML_PAYLOAD_LIMIT:
        return
    fits = sorted(
        (
            (det, scan) for det in VALID_BINS for scan in VALID_BINS
            if det >= det_bin and scan >= scan_bin
            and det_shape[0] % det == 0 and det_shape[1] % det == 0
            and scan_shape[0] % scan == 0 and scan_shape[1] % scan == 0
            and payload(det, scan) <= HTML_PAYLOAD_ADVISED
        ),
        # least reduction first, then the most even split; at a tie the scan sampling is kept
        key=lambda bins: (bins[0] * bins[1], max(bins), bins[1]),
    )
    advice = (f"pass det_bin={fits[0][0]}, scan_bin={fits[0][1]} ({payload(*fits[0]) / 1e9:.2f} GB), or "
              if fits else "")
    raise ValueError(
        f"An interactive HTML at det_bin={det_bin}, scan_bin={scan_bin} packs {size / 1e9:.1f} GB of {dtype} "
        f"counts, more than one browser page can load ({HTML_PAYLOAD_LIMIT / 2**30:.0f} GiB); {advice}"
        "use export_kind='report' for virtual-image pages without the raw counts. Nothing was binned."
    )


def export_kind(value) -> str:
    """``"interactive"`` or ``"report"`` from a caller or toolbar value; anything else is refused."""
    kind = str(value or "interactive").strip().lower()
    if kind not in ("interactive", "report"):
        raise ValueError("export_kind must be 'interactive' or 'report'")
    return kind


def mode_label(dtype: str, det_bin: int, scan_bin: int, kind: str) -> str:
    """Packing summary for the export status line, naming every active bin so a reduction is never silent."""
    parts = ["report" if kind == "report" else "interactive raw 4D", dtype]
    if scan_bin > 1:
        parts.append(f"scan bin {scan_bin}x")
    if det_bin > 1:
        parts.append(f"detector bin {det_bin}x")
    return ", ".join(parts)


def default_export_path(widget, dtype: str, det_bin: int, scan_bin: int, kind: str) -> pathlib.Path:
    """File in the kernel cwd named after the title, the exported shape and the packing, so exports do not overwrite."""
    shape = f"{widget.shape_rows // scan_bin}x{widget.shape_cols // scan_bin}x{widget.det_rows // det_bin}x{widget.det_cols // det_bin}"
    if widget.n_frames > 1:
        shape = f"{widget.n_frames}x{shape}"
    prefix = "report" if kind == "report" else dtype
    return pathlib.Path.cwd() / f"{export_slug(widget.title, 'show4dstem')}_{shape}_{prefix}_rbin{scan_bin}_kbin{det_bin}.html"


# ---------------------------------------------------------------------------
# Packing the counts
# ---------------------------------------------------------------------------


def mean_scan_bin(arr: np.ndarray, scan_bin: int) -> np.ndarray:
    """Average ``scan_bin x scan_bin`` scan positions; the two trailing axes stay.

    Mean rather than sum keeps every binned value inside the raw count range,
    so the result still packs into the chosen integer dtype without clipping.
    """
    arr = np.asarray(arr)
    if scan_bin <= 1:
        return arr
    lead = arr.shape[:-4] if arr.ndim >= 4 else arr.shape[:-2]
    rows, cols = arr.shape[len(lead)], arr.shape[len(lead) + 1]
    tail = arr.shape[len(lead) + 2 :]
    binned = arr.reshape(*lead, rows // scan_bin, scan_bin, cols // scan_bin, scan_bin, *tail)
    return binned.mean(axis=(len(lead) + 1, len(lead) + 3))


def mean_detector_bin(arr: np.ndarray, det_bin: int) -> np.ndarray:
    """Average ``det_bin x det_bin`` detector pixels of the two trailing axes."""
    arr = np.asarray(arr)
    if det_bin <= 1:
        return arr
    det_rows, det_cols = arr.shape[-2:]
    binned = arr.reshape(*arr.shape[:-2], det_rows // det_bin, det_bin, det_cols // det_bin, det_bin)
    return binned.mean(axis=(-3, -1))


def pack_counts(arr: np.ndarray, dtype: str, *, rounded: bool) -> np.ndarray:
    """Clip to the integer range of ``dtype``; binned means are rounded to whole counts first.

    Clipping changes data, so it is announced with the count and the way to avoid it.
    """
    if rounded:
        arr = np.round(arr)
    limit = np.iinfo(dtype).max
    announce_clipped(int(np.count_nonzero(arr > limit)) + int(np.count_nonzero(arr < 0)), arr.size, dtype)
    return np.clip(arr, 0, limit).astype(dtype, copy=False)


def export_data_array(widget, dtype: str, det_bin: int, scan_bin: int) -> np.ndarray:
    """Every frame of the viewer as one packed host array, binned by mean.

    Frames are read one at a time (GPU acquisitions in small scan windows,
    ``widget._frame_array``) and packed into the preallocated output, so the
    host holds one float32 frame besides the packed counts and the GPU never
    holds a dense cube. Clipping is announced once for every frame.
    """
    limit = np.iinfo(dtype).max
    packed, clipped = None, 0
    for frame_idx in range(widget.n_frames):
        values = mean_scan_bin(widget._frame_array(frame_idx, det_bin), scan_bin)
        if det_bin > 1 or scan_bin > 1:
            np.round(values, out=values)
        clipped += int(np.count_nonzero(values > limit)) + int(np.count_nonzero(values < 0))
        if packed is None:
            packed = np.empty((widget.n_frames, *values.shape), dtype=dtype)
        packed[frame_idx] = np.clip(values, 0, limit, out=values)
    announce_clipped(clipped, packed.size, dtype)
    return packed if widget.n_frames > 1 else packed[0]


def announce_clipped(clipped: int, total: int, dtype: str) -> None:
    """Print how many of ``total`` values a ``dtype`` pack clipped, and how to keep them."""
    if clipped:
        limit = np.iinfo(dtype).max
        hint = ("pass dtype='uint16' (--dtype u16 or auto on the command line) to keep counts up to 65535"
                if dtype == "uint8" else "the data exceed the uint16 count range")
        print(f"Show4DSTEM {dtype} pack clipped {clipped} values outside 0..{limit} ({clipped / total:.3%}); {hint}")


def export_state_for_bin(widget, det_bin: int, scan_bin: int) -> dict:
    """The widget's display state in the exported pixel grids.

    Detector coordinates shrink by ``det_bin`` and scan coordinates by
    ``scan_bin``; explicit count windows are dropped because binned means do
    not share the raw count scale.
    """
    state = widget.state_dict()
    if det_bin > 1:
        for key in ("center_row", "center_col", "bf_radius", "roi_center_row", "roi_center_col",
                    "roi_radius", "roi_radius_inner", "roi_width", "roi_height"):
            state[key] = float(state[key]) / det_bin
        state["k_pixel_size"] = float(state["k_pixel_size"]) * det_bin
        state["dp_vmin"] = None
        state["dp_vmax"] = None
    if scan_bin > 1:
        for key in ("pos_row", "pos_col", "vi_roi_center_row", "vi_roi_center_col",
                    "vi_roi_radius", "vi_roi_width", "vi_roi_height"):
            state[key] = float(state[key]) / scan_bin
        state["pixel_size"] = float(state["pixel_size"]) * scan_bin
        state["vi_vmin"] = None
        state["vi_vmax"] = None
    return state


def pack_inline(widget, dtype: str) -> None:
    """Gzip the viewer's own counts into the ``_offline_stack`` trait for a kernel-less page.

    A GPU acquisition is packed frame by frame from small scan windows
    (``export_data_array``): the host holds one float32 frame besides the
    packed counts.
    """
    if isinstance(widget._data, torch.Tensor):
        packed = pack_counts(widget._data.detach().to("cpu").numpy(), dtype, rounded=False)
    else:
        packed = export_data_array(widget, dtype, 1, 1)
    widget._offline_stack = gzip.compress(np.ascontiguousarray(packed).tobytes(), compresslevel=6)
    widget._offline_bad_px = ""
    widget.offline = True


# ---------------------------------------------------------------------------
# Interactive page
# ---------------------------------------------------------------------------


def export_clone(widget, dtype: str, det_bin: int, scan_bin: int):
    """A second viewer over the packed counts whose state is what the page embeds.

    The clone carries the binned geometry and the live display state, has its
    own export and SSB buttons disabled (there is no kernel behind the page),
    and keeps the heavy stack in its saved state.
    """
    data = export_data_array(widget, dtype, det_bin, scan_bin)
    clone = type(widget)(
        data,
        sampling=(widget.pixel_size * scan_bin, widget.pixel_size * scan_bin,
                  widget.k_pixel_size * det_bin, widget.k_pixel_size * det_bin),
        units=[widget.pixel_unit, widget.pixel_unit, widget.k_pixel_unit, widget.k_pixel_unit],
        center=(widget.center_row / det_bin, widget.center_col / det_bin),
        bf_radius=max(1.0, widget.bf_radius / det_bin),
        precompute_virtual_images=False,
        frame_dim_label=widget.frame_dim_label,
        frame_labels=list(widget.frame_labels),
        title=widget.title,
        show_title=widget.show_title,
        show_fft=widget.show_fft,
        show_controls=widget.show_controls,
        controls_collapsed=widget.controls_collapsed,
        show_stats=widget.show_stats,
        show_scale_bar=widget.show_scale_bar,
        view_mode=widget.view_mode,
        compare_cols=widget.compare_cols,
        compare_grid_width_px=widget.compare_grid_width_px,
        compare_max_panels=widget.compare_max_panels,
        compare_group_mode=widget.compare_group_mode,
        compare_dp_mode=widget.compare_dp_mode,
        verbose=False,
    )
    for label, product in widget._vi_product_maps.items():
        clone._set_vi_product_map(label, mean_scan_bin(product, scan_bin))
    clone.load_state_dict(export_state_for_bin(widget, det_bin, scan_bin))
    pack_inline(clone, dtype)
    clone.export_enabled = False
    clone.export_status = ""
    clone.export_payload = b""
    clone.export_payload_id = ""
    clone.export_filename = ""
    clone.ssb_compute_enabled = False
    clone.ssb_compute_status = ""
    clone.ssb_compute_request = ""
    clone.ssb_compute_busy = False
    clone.ssb_compute_calibration_json = ""
    clone.ssb_compute_calibration_filename = ""
    clone._save_state = True
    return clone


def write_interactive_html(widget, path, *, dtype: str, det_bin: int, scan_bin: int, title: str | None) -> pathlib.Path:
    """Write the kernel-less page: packed counts for arrays, linked HDF5 for a browser source."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    page_title = title or widget.title or "Show4DSTEM"
    if widget._h5_urls:
        if det_bin != 1 or scan_bin != 1:
            raise ValueError(
                "Binned interactive raw export is not available for browser-source "
                "Show4DSTEM exports because the browser reads the real H5 data "
                "directly. Use det_bin=1 and scan_bin=1 to preserve the full data."
            )
        urls = json.loads(widget._h5_urls)
        previous = (widget.export_enabled, widget._save_state, widget._offline_bad_px)
        widget.export_enabled = False
        widget._save_state = True
        try:
            if not widget._offline_bad_px and len(urls) == 1:
                widget._offline_bad_px = h5_bad_pixel_json(urls[0], out.parent) or ""
            write_widget_html(out, widget, page_title)
        finally:
            widget.export_enabled, widget._save_state, widget._offline_bad_px = previous
        inject_h5_tuning(out, dtype)
        return out
    check_html_payload(widget.n_frames, (widget.shape_rows, widget.shape_cols), (widget.det_rows, widget.det_cols),
                       dtype, det_bin, scan_bin)
    clone = export_clone(widget, dtype, det_bin, scan_bin)
    try:
        return write_widget_html(out, clone, page_title)
    finally:
        clone.close()
        clone.layout.close()  # Widget.close leaves the child Layout model open


# ---------------------------------------------------------------------------
# Static report
# ---------------------------------------------------------------------------

REPORT_SCOPES = {
    "unhidden": "unhidden", "visible": "unhidden",
    "current_page": "current-page", "current-page": "current-page", "current": "current-page", "page": "current-page",
    "starred": "starred", "star": "starred", "stars": "starred",
    "all": "all",
}


def report_scope(value) -> str:
    """Canonical report scope (``unhidden``, ``current-page``, ``starred``, ``all``) from any accepted spelling."""
    key = str(value or "unhidden").strip().lower()
    if key not in REPORT_SCOPES:
        raise ValueError("dataset_scope must be 'unhidden', 'current_page', 'starred', or 'all'")
    return REPORT_SCOPES[key]


def report_indices(widget, dataset_scope: str) -> list[int]:
    """Dataset indices a report covers, in display order."""
    scope = report_scope(dataset_scope)
    if widget.n_frames <= 1:
        return [0]
    ordered = widget._panel_order()
    if scope == "all":
        return ordered
    if scope == "starred":
        starred = widget._panel_set(widget.compare_starred_panels)
        return [idx for idx in ordered if idx in starred]
    hidden = widget._panel_set(widget.compare_hidden_panels)
    if scope == "current-page":
        start = max(0, int(widget.compare_page_idx)) * max(1, int(widget.compare_max_panels))
        ordered = ordered[start : start + max(1, int(widget.compare_max_panels))]
    return [idx for idx in ordered if idx not in hidden]


REPORT_CSS = """
:root {{ --cols: {cols}; color-scheme: light; }}
body {{ margin: 0; font: 13px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #202124; background: #fff; }}
main {{ padding: 14px; max-width: 1600px; }}
h1 {{ margin: 0 0 6px; color: #0759c9; font-size: 20px; }}
.sub {{ margin: 0 0 12px; color: #5f6368; max-width: 980px; }}
.toolbar {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 10px 0 12px; }}
.toolbar label {{ font-size: 11px; color: #5f6368; }}
select, button {{ font: inherit; height: 28px; border: 1px solid #d7dce2; border-radius: 4px; background: #f8f9fa; padding: 0 8px; }}
.layout {{ display: grid; grid-template-columns: minmax(220px, 360px) minmax(0, 1fr); gap: 12px; align-items: start; }}
.dp img {{ width: 100%; image-rendering: pixelated; background: #000; display: block; }}
.meta {{ margin-top: 8px; font-size: 11px; color: #5f6368; }}
details.meta summary {{ cursor: pointer; color: #0759c9; }}
details.meta code {{ display: block; white-space: pre-wrap; overflow-wrap: anywhere; margin-top: 4px; }}
.grid {{ display: grid; grid-template-columns: repeat(var(--cols), minmax(0, 1fr)); gap: 0; align-items: start; }}
.tile {{ position: relative; margin: 0; min-width: 0; background: #000; overflow: hidden; }}
.tile img {{ width: 100%; display: block; image-rendering: pixelated; }}
.tile figcaption {{ position: absolute; top: 3px; left: 4px; right: 4px; color: #fff; font-weight: 700; font-size: 11px; text-shadow: 0 1px 2px #000; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.report-page {{ display: none; }}
.report-page.active {{ display: block; }}
@media (max-width: 700px) {{
  main {{ padding: 8px; }}
  .layout {{ display: block; }}
  .dp {{ margin-bottom: 8px; }}
  .grid {{ grid-template-columns: repeat(min(var(--cols), 2), minmax(0, 1fr)); }}
}}
"""

REPORT_JS = """
const preset = document.getElementById("preset");
const page = document.getElementById("page");
const prev = document.getElementById("prev");
const next = document.getElementById("next");
const status = document.getElementById("page-status");
const maxPage = Number(page.dataset.max || "1");
function sync() {
  const p = preset.value;
  const pageIdx = Number(page.value || "0");
  document.querySelectorAll(".report-page").forEach((el) => {
    el.classList.toggle("active", el.dataset.preset === p && Number(el.dataset.page || "0") === pageIdx);
  });
  status.textContent = `Page ${pageIdx + 1} / ${maxPage}`;
  prev.disabled = pageIdx <= 0;
  next.disabled = pageIdx >= maxPage - 1;
}
preset.addEventListener("change", sync);
page.addEventListener("change", sync);
prev.addEventListener("click", () => { page.value = String(Math.max(0, Number(page.value) - 1)); sync(); });
next.addEventListener("click", () => { page.value = String(Math.min(maxPage - 1, Number(page.value) + 1)); sync(); });
sync();
"""


def report_html_bytes(widget, *, dtype: str, det_bin: int, scan_bin: int, dataset_scope: str, title: str | None) -> bytes:
    """One self-contained HTML: BF/ABF/ADF/HAADF virtual-image pages and the cursor pattern as PNGs."""
    dtype, det_bin, scan_bin = export_options(widget, dtype, det_bin, scan_bin)
    indices = report_indices(widget, dataset_scope)
    if not indices:
        raise ValueError("No datasets are available for the requested report export scope")
    report_title = title or widget.title or "Show4DSTEM Report"
    page_size = max(1, int(widget.compare_max_panels))
    pages = [indices[i : i + page_size] for i in range(0, len(indices), page_size)]
    cols = int(widget.compare_cols) or math.ceil(math.sqrt(min(page_size, len(indices))))
    cols = max(1, min(cols, page_size, len(indices)))
    det_shape = (widget.det_rows, widget.det_cols)
    center = (widget.center_row, widget.center_col)
    pattern = scale_values(mean_detector_bin(widget.pattern(widget.pos_row, widget.pos_col), det_bin), widget.dp_scale_mode)
    dp_uri = png_data_uri(colormap_rgb(pattern, widget.dp_colormap, *display_range(pattern, widget.dp_vmin_pct, widget.dp_vmax_pct)))
    sections = []
    for page_idx, page in enumerate(pages):
        for name in PRESET_RADII:
            mask = preset_mask(name, det_shape, center, widget.bf_radius)
            tiles = []
            for panel_idx, image in zip(page, widget._compare_images(page, mask), strict=True):
                scaled = scale_values(mean_scan_bin(image, scan_bin), widget.vi_scale_mode)
                uri = png_data_uri(colormap_rgb(scaled, widget.vi_colormap, *display_range(scaled, widget.vi_vmin_pct, widget.vi_vmax_pct)))
                label = html.escape(widget._panel_title(panel_idx))
                tiles.append(
                    f'<figure class="tile" data-panel="{panel_idx}" data-page="{page_idx}">'
                    f'<img alt="{name.upper()} virtual image for {label}" src="{uri}">'
                    f"<figcaption>{label}</figcaption></figure>"
                )
            sections.append(
                f'<section class="report-page" data-preset="{name}" data-page="{page_idx}">'
                f'<div class="grid">{"".join(tiles)}</div></section>'
            )
    metadata = {
        "scope": report_scope(dataset_scope), "datasets": len(indices), "pages": len(pages),
        "page_size": page_size, "scan_bin": scan_bin, "det_bin": det_bin, "dtype": dtype,
        "scan_shape": [widget.shape_rows, widget.shape_cols], "detector_shape": list(det_shape),
    }
    summary = (f"{len(indices)} dataset(s) · {len(pages)} page(s) · scope {metadata['scope']} · "
               f"rbin {scan_bin} · kbin {det_bin}")
    page_options = "\n".join(f'<option value="{idx}">Page {idx + 1} / {len(pages)}</option>' for idx in range(len(pages)))
    preset_options = "\n".join(f'<option value="{name}">{name.upper()}</option>' for name in PRESET_RADII)
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(report_title)}</title>
  <style>{REPORT_CSS.format(cols=cols)}</style>
</head>
<body>
  <main>
    <h1>{html.escape(report_title)}</h1>
    <p class="sub">Static report export: virtual-image PNGs only. Raw interactive 4D data is not embedded.</p>
    <div class="toolbar">
      <label for="preset">Virtual image</label>
      <select id="preset">{preset_options}</select>
      <button id="prev" type="button">Prev</button>
      <label for="page">Page</label>
      <select id="page" data-max="{len(pages)}">{page_options}</select>
      <button id="next" type="button">Next</button>
      <span id="page-status" class="meta"></span>
    </div>
    <div class="layout">
      <aside class="dp">
        <img alt="Representative diffraction pattern" src="{dp_uri}">
        <div class="meta">DP at ({int(widget.pos_row)}, {int(widget.pos_col)}) · detector bin {det_bin}x</div>
        <div class="meta">{html.escape(summary)}</div>
        <details class="meta">
          <summary>Details</summary>
          <code>{html.escape(json.dumps(metadata, separators=(",", ":")))}</code>
        </details>
      </aside>
      <div class="pages">
        {"".join(sections)}
      </div>
    </div>
  </main>
  <script>{REPORT_JS}</script>
</body>
</html>"""
    return body.encode("utf-8")


def write_report_html(widget, path, *, dtype: str, det_bin: int, scan_bin: int, dataset_scope: str, title: str | None) -> pathlib.Path:
    """Write the static report page with a mobile viewport, so it reads on a phone without zooming."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(report_html_bytes(widget, dtype=dtype, det_bin=det_bin, scan_bin=scan_bin,
                                      dataset_scope=dataset_scope, title=title))
    ensure_mobile_viewport(out)
    return out


# ---------------------------------------------------------------------------
# Browser WebGPU bundle over linked HDF5
# ---------------------------------------------------------------------------

VENDOR = pathlib.Path(__file__).parent.parent / "vendor"
# embed_minimal_html references CDNs; each is replaced by a vendored copy so the
# bundle opens with no network. Patterns, not exact URLs: the version specifiers
# drift across ipywidgets/anywidget releases.
CDN_REWRITES = (
    (re.compile(r"https://cdnjs\.cloudflare\.com/[^\"']*/require(\.min)?\.js"), "./require.min.js"),
    (re.compile(r"https://cdn\.jsdelivr\.net/[^\"']*html-manager[^\"']*/embed-amd\.js"), "./embed-amd.js"),
    (re.compile(r"\"https://cdn\.jsdelivr\.net/npm/anywidget@[^\"]*\""), '"./anywidget.min"'),
)


def h5_tuning_script(dtype: str) -> str:
    """Runtime settings the browser HDF5 loader reads before the widget bundle.

    ``uint8`` decodes the low byte only (compact browse); anything else keeps
    native ``uint16`` counts. The queue depths are the measured sweet spot for
    range-reading bitshuffle-LZ4 chunks over a local server.
    """
    decode = "uint8" if dtype == "uint8" else "u2"
    return (
        "<script>\n"
        "if (globalThis.location?.protocol === \"file:\") {\n"
        "  globalThis.__QT_REQUIRE_LOCAL_H5_FILES = true;\n"
        "}\n"
        f'globalThis.__QT_H5_DECODE_DTYPE ??= "{decode}";\n'
        "globalThis.__BSLZ4_FRAME_WG ??= 64;\n"
        "globalThis.__BSLZ4_PIPELINE_STAGING ??= false;\n"
        "globalThis.__QT_H5_FETCH_WINDOW ??= 8;\n"
        "globalThis.__QT_H5_DECODE_QUEUE ??= 8;\n"
        "globalThis.__QT_H5_PRELOAD_WINDOW ??= 1;\n"
        "globalThis.__QT_H5_LOCAL_GROUP ??= 8;\n"
        "globalThis.__QT_H5_LOCAL_WORKERS ??= 8;\n"
        "</script>\n"
    )


def inject_h5_tuning(path: pathlib.Path, dtype: str) -> None:
    """Put ``h5_tuning_script`` at the top of ``<head>``: the HDF5 loader reads those globals before the bundle runs."""
    text = path.read_text(encoding="utf-8")
    if "globalThis.__QT_H5_DECODE_DTYPE ??=" in text:  # the bundled JS mentions the name too
        return
    script = h5_tuning_script(dtype)
    text = text.replace("<head>", "<head>\n" + script, 1) if "<head>" in text else script + text
    path.write_text(text, encoding="utf-8")


def h5_bad_pixel_json(url: str, base_dir: pathlib.Path) -> str | None:
    """JSON list of flagged detector pixels of a local master, embedded so the page needs no second read."""
    if not url or "://" in url:
        return None
    path = pathlib.Path(url.split("?", 1)[0].split("#", 1)[0])
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        return None
    mask = read_pixel_mask(path)
    if mask is None:
        return None
    return json.dumps(np.flatnonzero(np.asarray(mask) > 0).tolist())


def bundle_master_urls(folder, names=None, *, viewer_prefix: str = "..") -> list[str]:
    """Viewer-relative URLs of the masters in a bundle folder.

    The page lives one level down in ``.viewer/``, so the data references climb
    back to the served root. ``names`` filters by substring in the given order.
    """
    masters = sorted(path.name for path in pathlib.Path(folder).glob("*_master.h5"))
    if names:
        picked = []
        for token in names:
            hits = [master for master in masters if token in master]
            if not hits:
                raise ValueError(f"no master matches {token!r} in {folder}")
            picked.append(hits[0])
        masters = picked
    return [f"{viewer_prefix}/{name}" for name in masters]


def write_webgpu_bundle(widget, out_dir, *, port: int = 8794, title: str | None = None, h5_decode_dtype: str = "uint16") -> pathlib.Path:
    """Write ``Show4DSTEM.command``, ``index.html`` and a vendored ``.viewer/`` page into the data folder.

    ``out_dir`` holds the linked ``*_master.h5`` family the widget references.
    The recipient double-clicks the command: a range-capable server starts over
    this folder and Chrome opens the page, with no Python install and no CDN.
    """
    root = pathlib.Path(out_dir)
    if not root.is_dir():
        raise ValueError(f"bundle out_dir must be an existing data folder: {root}")
    if not sorted(root.rglob("*_master.h5")):
        raise ValueError(f"no *_master.h5 files in {root}; the bundle serves the data folder itself")
    viewer = root / ".viewer"
    viewer.mkdir(exist_ok=True)
    page = viewer / "Show4DSTEM.html"
    write_interactive_html(widget, page, dtype=str(h5_decode_dtype).lower(), det_bin=1, scan_bin=1, title=title)
    text = page.read_text(encoding="utf-8")
    for pattern, local in CDN_REWRITES:
        text = pattern.sub(local, text)
    page.write_text(text, encoding="utf-8")
    for name in ("require.min.js", "embed-amd.js", "anywidget.min.js"):
        source = VENDOR / f"{name}.gz"
        if not source.is_file():
            raise FileNotFoundError(
                f"missing vendored Show4DSTEM browser asset: {source}; "
                "rebuild the package with src/quantem/widget/vendor included"
            )
        with gzip.open(source, "rb") as compressed, (viewer / name).open("wb") as unpacked:
            unpacked.write(compressed.read())
    (root / "index.html").write_text(
        '<!doctype html>\n<html><head><meta charset="utf-8"><title>Show4DSTEM</title></head>\n<body>\n<script>\n'
        'window.location.replace(".viewer/Show4DSTEM.html" + window.location.search + window.location.hash);\n'
        "</script>\n</body></html>\n",
        encoding="utf-8",
    )
    return write_command_launcher(root, "Show4DSTEM", viewer_html="index.html", port=int(port))
