"""
Bin2D: Interactive 2D image binning widget with side-by-side comparison.

Designed for quick downsampling of large TEM/HAADF images (2048×2048,
4096×4096) with real-time preview, calibration tracking, and export.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import bin2d, to_numpy
from quantem.widget.io import IOResult
from quantem.widget.json_state import build_json_header, resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)

_BIN2D_ESM = pathlib.Path(__file__).parent / "static" / "bin2d.js"
_BIN2D_CSS = pathlib.Path(__file__).parent / "static" / "bin2d.css"


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.1f} GB"


def _compute_stats(arr: np.ndarray) -> list[float]:
    if arr.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(arr.mean()), float(arr.min()), float(arr.max()), float(arr.std())]


class Bin2D(anywidget.AnyWidget):
    """
    Interactive 2D image binning widget with side-by-side comparison.

    Parameters
    ----------
    data : array_like
        2D ``(H, W)`` single image, 3D ``(N, H, W)`` stack, or list of 2D
        arrays. Also accepts IOResult and quantem Dataset2d objects.
    bin_factor : int, default 2
        Initial spatial bin factor.
    bin_mode : {"mean", "sum"}, default "mean"
        Reduction mode.
    edge_mode : {"crop", "pad"}, default "crop"
        How non-divisible dimensions are handled.
    pixel_size : float, optional
        Pixel size in Å/px for scale bar and calibration.
    title : str, default ""
        Widget title.
    cmap : str, default "inferno"
        Colormap name.
    labels : list of str, optional
        Per-image labels for gallery mode.
    state : dict or str or Path, optional
        Restore state from dict or JSON file.
    """

    _esm = _BIN2D_ESM if _BIN2D_ESM.exists() else "export function render() {}"
    _css = _BIN2D_CSS if _BIN2D_CSS.exists() else ""

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    n_images = traitlets.Int(1).tag(sync=True)
    selected_idx = traitlets.Int(0).tag(sync=True)

    bin_factor = traitlets.Int(2).tag(sync=True)
    max_bin_factor = traitlets.Int(16).tag(sync=True)
    bin_mode = traitlets.Unicode("mean").tag(sync=True)
    edge_mode = traitlets.Unicode("crop").tag(sync=True)
    binned_height = traitlets.Int(1).tag(sync=True)
    binned_width = traitlets.Int(1).tag(sync=True)

    original_bytes = traitlets.Bytes(b"").tag(sync=True)
    binned_bytes = traitlets.Bytes(b"").tag(sync=True)
    original_stats = traitlets.List(traitlets.Float()).tag(sync=True)
    binned_stats = traitlets.List(traitlets.Float()).tag(sync=True)

    pixel_size = traitlets.Float(0.0).tag(sync=True)
    binned_pixel_size = traitlets.Float(0.0).tag(sync=True)

    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)

    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups):
        return normalize_tool_groups("Bin2D", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_binning: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_navigation: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
    ):
        return build_tool_groups(
            "Bin2D",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "binning": disable_binning,
                "histogram": disable_histogram,
                "stats": disable_stats,
                "navigation": disable_navigation,
                "export": disable_export,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_binning: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_navigation: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
    ):
        return build_tool_groups(
            "Bin2D",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "binning": hide_binning,
                "histogram": hide_histogram,
                "stats": hide_stats,
                "navigation": hide_navigation,
                "export": hide_export,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    def __init__(
        self,
        data,
        *,
        bin_factor: int = 2,
        bin_mode: str = "mean",
        edge_mode: str = "crop",
        pixel_size: float | None = None,
        title: str = "",
        cmap: str = "inferno",
        labels: list[str] | None = None,
        log_scale: bool = False,
        auto_contrast: bool = False,
        show_stats: bool = True,
        show_controls: bool = True,
        disabled_tools: list[str] | None = None,
        disable_display: bool = False,
        disable_binning: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_navigation: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
        hidden_tools: list[str] | None = None,
        hide_display: bool = False,
        hide_binning: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_navigation: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        extracted_pixel_size = None
        extracted_title = None

        if isinstance(data, IOResult):
            if data.title:
                extracted_title = data.title
            if data.pixel_size:
                extracted_pixel_size = data.pixel_size
            if data.labels and labels is None:
                labels = data.labels
            data = data.data

        if hasattr(data, "data") and hasattr(data, "pixel_size"):
            ds = data
            if hasattr(ds, "title") and ds.title:
                extracted_title = str(ds.title)
            ps = getattr(ds, "pixel_size", None)
            if ps is not None and float(ps) > 0:
                unit = getattr(ds, "units", None) or getattr(ds, "unit", None) or ""
                ps_val = float(ps)
                if "nm" in str(unit).lower():
                    ps_val *= 10.0
                extracted_pixel_size = ps_val
            data = getattr(ds, "data", ds)

        self._data = self._prepare_data(data, labels)
        n, h, w = self._data.shape

        self.title = title or extracted_title or ""
        self.n_images = n
        self.height = h
        self.width = w
        self.labels = labels or []
        self.cmap = str(cmap)
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.show_stats = show_stats
        self.show_controls = show_controls

        if pixel_size is not None:
            self.pixel_size = float(pixel_size)
        elif extracted_pixel_size is not None:
            self.pixel_size = extracted_pixel_size

        self.bin_factor = bin_factor
        self.bin_mode = bin_mode
        self.edge_mode = edge_mode
        self.max_bin_factor = max(2, min(h, w) // 2)

        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_binning=disable_binning,
            disable_histogram=disable_histogram,
            disable_stats=disable_stats,
            disable_navigation=disable_navigation,
            disable_export=disable_export,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_binning=hide_binning,
            hide_histogram=hide_histogram,
            hide_stats=hide_stats,
            hide_navigation=hide_navigation,
            hide_export=hide_export,
            hide_all=hide_all,
        )

        self._update_current_frame()
        self._update_binned()

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = json.loads(pathlib.Path(state).read_text())
            state = unwrap_state_payload(state)
            self.load_state_dict(state)

    def _prepare_data(self, data, labels=None) -> np.ndarray:
        if isinstance(data, list):
            arrs = [to_numpy(d, dtype=np.float32) for d in data]
            for a in arrs:
                if a.ndim != 2:
                    raise ValueError(f"Each image must be 2D, got shape {a.shape}")
            return np.stack(arrs)
        arr = to_numpy(data, dtype=np.float32)
        if arr.ndim == 2:
            return arr[np.newaxis]
        if arr.ndim == 3:
            return arr
        raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D")

    def _update_current_frame(self):
        idx = max(0, min(self.selected_idx, self.n_images - 1))
        frame = self._data[idx]
        self.original_bytes = frame.astype(np.float32).tobytes()
        self.original_stats = _compute_stats(frame)

    def _update_binned(self):
        idx = max(0, min(self.selected_idx, self.n_images - 1))
        frame = self._data[idx]
        binned = bin2d(frame, factor=self.bin_factor, mode=self.bin_mode, edge_mode=self.edge_mode)
        bh, bw = binned.shape[-2:]
        self.binned_height = bh
        self.binned_width = bw
        self.binned_pixel_size = self.pixel_size * self.bin_factor if self.pixel_size > 0 else 0.0
        self.binned_bytes = binned.astype(np.float32).tobytes()
        self.binned_stats = _compute_stats(binned)

    @traitlets.observe("bin_factor", "bin_mode", "edge_mode")
    def _on_bin_params_changed(self, change):
        self._update_binned()

    @traitlets.observe("selected_idx")
    def _on_selected_idx_changed(self, change):
        self._update_current_frame()
        self._update_binned()

    def set_image(self, data, labels: list[str] | None = None) -> Self:
        self._data = self._prepare_data(data, labels)
        n, h, w = self._data.shape
        self.n_images = n
        self.height = h
        self.width = w
        self.max_bin_factor = max(2, min(h, w) // 2)
        if labels is not None:
            self.labels = labels
        self.selected_idx = 0
        self._update_current_frame()
        self._update_binned()
        return self

    def get_binned_data(self, copy: bool = True) -> np.ndarray:
        result = bin2d(self._data, factor=self.bin_factor, mode=self.bin_mode, edge_mode=self.edge_mode)
        if result.ndim == 3 and result.shape[0] == 1:
            result = result[0]
        return result.copy() if copy else result

    def get_binned_image(self, idx: int | None = None, copy: bool = True) -> np.ndarray:
        if idx is None:
            idx = self.selected_idx
        frame = self._data[idx]
        binned = bin2d(frame, factor=self.bin_factor, mode=self.bin_mode, edge_mode=self.edge_mode)
        return binned.copy() if copy else binned

    def save_image(self, path, *, view: str | None = None, format: str | None = None, dpi: int = 150) -> pathlib.Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        path = pathlib.Path(path)
        fmt = format or path.suffix.lstrip(".").lower() or "png"
        view = view or "binned"
        idx = self.selected_idx
        frame = self._data[idx]
        binned = bin2d(frame, factor=self.bin_factor, mode=self.bin_mode, edge_mode=self.edge_mode)

        if view == "original":
            img = frame
        elif view == "grid":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=dpi)
            ax1.imshow(frame, cmap=self.cmap)
            ax1.set_title(f"Original ({self.height}×{self.width})")
            ax1.axis("off")
            ax2.imshow(binned, cmap=self.cmap)
            ax2.set_title(f"Binned ×{self.bin_factor} ({binned.shape[0]}×{binned.shape[1]})")
            ax2.axis("off")
            fig.tight_layout()
            fig.savefig(str(path), format=fmt, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            meta_path = path.with_suffix(".json")
            meta_path.write_text(json.dumps({
                **build_json_header("Bin2D"),
                "format": fmt,
                "export_kind": "single_view_image",
                "render": {"view": view, "bin_factor": self.bin_factor, "bin_mode": self.bin_mode},
            }, indent=2))
            return path
        else:
            img = binned

        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
        ax.imshow(img, cmap=self.cmap)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(str(path), format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps({
            **build_json_header("Bin2D"),
            "format": fmt,
            "export_kind": "single_view_image",
            "render": {"view": view, "bin_factor": self.bin_factor, "bin_mode": self.bin_mode},
        }, indent=2))
        return path

    def state_dict(self) -> dict:
        return {
            "bin_factor": self.bin_factor,
            "bin_mode": self.bin_mode,
            "edge_mode": self.edge_mode,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "title": self.title,
            "selected_idx": self.selected_idx,
            "disabled_tools": list(self.disabled_tools),
            "hidden_tools": list(self.hidden_tools),
        }

    def save(self, path: str) -> None:
        save_state_file(path, "Bin2D", self.state_dict())

    def load_state_dict(self, state: dict) -> None:
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self) -> None:
        title_line = f"  Title: {self.title}" if self.title else "  Title: (none)"
        print(f"Bin2D Summary\n{title_line}")
        print(f"  Original: {self.height} × {self.width} ({self.n_images} image{'s' if self.n_images > 1 else ''})")
        print(f"  Bin factor: {self.bin_factor}× ({self.bin_mode}, {self.edge_mode})")
        print(f"  Binned: {self.binned_height} × {self.binned_width}")
        orig_bytes = self.n_images * self.height * self.width * 4
        binned_bytes = self.n_images * self.binned_height * self.binned_width * 4
        reduction = orig_bytes / binned_bytes if binned_bytes > 0 else 1
        print(f"  Data size: {_format_bytes(orig_bytes)} → {_format_bytes(binned_bytes)} ({reduction:.0f}× reduction)")
        if self.pixel_size > 0:
            print(f"  Pixel size: {self.pixel_size:.4g} Å/px → {self.binned_pixel_size:.4g} Å/px")
        print(f"  Display: cmap={self.cmap}, log_scale={self.log_scale}")
        if self.disabled_tools:
            print(f"  Locked: {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            print(f"  Hidden: {', '.join(self.hidden_tools)}")

    def __repr__(self) -> str:
        title_info = f", title='{self.title}'" if self.title else ""
        return (
            f"Bin2D({self.height}×{self.width}, "
            f"bin={self.bin_factor}×, "
            f"binned={self.binned_height}×{self.binned_width}, "
            f"mode={self.bin_mode}"
            f"{title_info})"
        )


bind_tool_runtime_api(Bin2D, "Bin2D")
