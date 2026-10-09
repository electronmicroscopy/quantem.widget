"""Show1D exports: the standalone HTML page, the matplotlib figure and the saved-notebook preview.

The HTML export keeps every trace sample exact; ``downsample`` only mean-bins
the linked 2D panels (snapshots, profile image) and rescales their calibration
so a smaller file stays physically correct.
"""

import base64
import io
import math
import pathlib
from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
from matplotlib.figure import Figure

from quantem.widget.export import export_slug

HTML_EXPORT_DOWNSAMPLES = (1, 2, 4, 8)


class Show1DExport:
    """Mixin over ``HtmlExportMixin``: Show1D's export options, clone, figure and preview."""

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | None = None,
    ) -> pathlib.Path:
        """Write a kernel-free interactive HTML viewer.

        Only ``mode="single"`` with ``encoding="full"`` exists for Show1D: the
        traces are float32 and small, so there is nothing to quantise.
        ``downsample`` of 2, 4 or 8 mean-bins the linked snapshot and profile
        images; trace samples and x coordinates stay exact.
        """
        factor = _export_downsample(mode, encoding, downsample)
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path(downsample=factor)
        self._write_html_export(export_path, title=title, downsample=factor)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, {self._export_mode_label(factor)})"
        return export_path

    def _html_export_options(self, payload: dict, mode: str) -> dict:
        """``export_html`` keywords from a toolbar request; only the image downsample is a choice."""
        return {"downsample": _export_downsample(mode, str(payload.get("encoding", "full")), payload.get("downsample"))}

    def _export_mode_label(self, downsample: int = 1) -> str:
        """How the export packed the data, for the toolbar status line."""
        return "full float32" + (f", {downsample}x downsampled images" if downsample > 1 else "")

    def _default_html_export_path(self, *, downsample: int = 1) -> pathlib.Path:
        """``<title>_<traces>x<points>[_Nxdownsampled]_single.html`` in the working directory."""
        suffix = f"_{downsample}xdownsampled" if downsample > 1 else ""
        return pathlib.Path.cwd() / f"{export_slug(self.title, 'show1d')}_{self.n_traces}x{self.n_points}{suffix}_single.html"

    def _clone_for_html_export(self, *, downsample: int = 1) -> Self:
        """An export-only copy: same traces and view state, linked images mean-binned by ``downsample``.

        The clone embeds everything (``_save_state``), paints with the offline
        theme and has no export or hand-off buttons, since there is no kernel
        behind the exported page.
        """
        clone = type(self)(
            self._data.copy(),
            x=None if self._x is None else self._x.copy(),
            labels=list(self.labels),
            title=self.title,
        )
        clone._snapshots = [_downsample_image(snap, downsample) for snap in self._snapshots]
        clone.snapshot_iterations = list(self.snapshot_iterations)
        clone.snapshot_labels = list(self.snapshot_labels)
        clone.snapshot_image_labels = list(self.snapshot_image_labels)
        clone.snapshot_group_indices = list(self.snapshot_group_indices)
        clone.snapshot_group_iterations = list(self.snapshot_group_iterations)
        clone.snapshot_group_labels = list(self.snapshot_group_labels)
        clone.n_snapshot_groups = self.n_snapshot_groups
        clone._trial_metrics = {label: dict(values) for label, values in self._trial_metrics.items()}
        clone._monitor_warnings = list(self._monitor_warnings)
        clone._update_snapshot_bytes()
        clone.load_state_dict(self.state_dict())
        if self._profile_image is not None:
            clone._set_profile_image(_downsample_image(self._profile_image, downsample))
            clone.profile_line = _scale_line(self.profile_line, downsample)
            clone.profile_width = max(1, math.ceil(self.profile_width / downsample))
        if self.pixel_size > 0:
            clone.pixel_size = self.pixel_size * downsample
        clone.snapshot_real_space_center = [value / downsample for value in self.snapshot_real_space_center]
        clone.snapshot_fft_center = [value / downsample for value in self.snapshot_fft_center]
        clone.snapshot_profile_line = _scale_line(self.snapshot_profile_line, downsample)
        clone._save_state = True
        clone._export_light = True
        clone.export_enabled = False
        clone.handoff_enabled = False
        return clone

    def save_image(self, path: str | pathlib.Path, *, format: str | None = None, dpi: int = 150) -> pathlib.Path:
        """Save a publication-style PNG or PDF line figure of the traces."""
        out = pathlib.Path(path)
        fmt = (format or out.suffix.lstrip(".") or "png").lower()
        if fmt not in {"png", "pdf"}:
            raise ValueError(f"Unsupported format {fmt!r}. Use 'png' or 'pdf'.")
        out.parent.mkdir(parents=True, exist_ok=True)
        x_values = self._effective_x()
        fig = Figure(figsize=(6, 3.5), dpi=dpi)
        ax = fig.add_subplot(111)
        for trace in range(self.n_traces):
            ax.plot(x_values, self._data[trace], color=self.colors[trace], label=self.labels[trace], linewidth=self.line_width)
        if self.log_scale:
            ax.set_yscale("log")
        if self.show_grid:
            ax.grid(True, alpha=0.3, linestyle="--")
        if self.show_legend and self.n_traces > 1:
            ax.legend(fontsize=8, framealpha=0.8)
        xlabel = self.x_label + (f" ({self.x_unit})" if self.x_label and self.x_unit else self.x_unit)
        ylabel = self.y_label + (f" ({self.y_unit})" if self.y_label and self.y_unit else self.y_unit)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if self.show_title and self.title:
            ax.set_title(self.title)
        fig.tight_layout()
        fig.savefig(out, format=fmt, bbox_inches="tight")
        return out

    def _static_png_b64(self, max_px: int = 512) -> str | None:
        """Matplotlib render of the traces for the saved-notebook preview.

        Mirrors what the live plot shows on mount: the first eight traces (a
        wide lambda sweep stays legible), the log-scale setting and the axis
        labels. Points are stride-decimated; this is a reopen preview, not data.
        """
        if self._data.size == 0:
            return None
        traces = np.atleast_2d(self._data)
        x_values = self._effective_x()
        stride = max(1, traces.shape[1] // 4096)
        fig = Figure(figsize=(max_px / 100.0, max_px * 0.62 / 100.0), dpi=100)
        ax = fig.add_subplot(111)
        for trace in range(min(len(traces), 8)):
            label = self.labels[trace] if trace < len(self.labels) and self.labels[trace].strip() else None
            ax.plot(x_values[::stride], traces[trace][::stride], linewidth=1.0, label=label)
        if self.log_scale:
            ax.set_yscale("log")
        if self.title:
            ax.set_title(self.title, fontsize=9)
        if self.x_label:
            ax.set_xlabel(self.x_label, fontsize=8)
        if self.y_label:
            ax.set_ylabel(self.y_label, fontsize=8)
        ax.tick_params(labelsize=7)
        if any(line.get_label() and not line.get_label().startswith("_") for line in ax.lines):
            ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _export_downsample(mode: str, encoding: str, downsample) -> int:
    """The validated image downsample factor; the only packing Show1D offers is single-file float32."""
    if mode == "folder":
        raise NotImplementedError("Show1D folder export is not available. Use mode='single' with downsample=2, 4, or 8 to reduce linked image panels.")
    if mode != "single":
        raise ValueError("Show1D HTML export supports mode='single'")
    if encoding == "uint8":
        raise NotImplementedError("Show1D encoding='uint8' is not available. Use encoding='full' with downsample=2, 4, or 8.")
    if encoding != "full":
        raise ValueError(f"unknown Show1D export encoding {encoding!r}; expected 'full'")
    if downsample in (None, 0):
        return 1
    if isinstance(downsample, bool) or int(downsample) != downsample or int(downsample) not in HTML_EXPORT_DOWNSAMPLES:
        raise ValueError(f"Show1D HTML export downsample must be one of {HTML_EXPORT_DOWNSAMPLES}, got {downsample!r}")
    return int(downsample)


def _downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """NaN-aware mean over ``factor`` x ``factor`` blocks; partial edge blocks average what they have."""
    if factor <= 1 or image.size == 0:
        return image.copy()
    height, width = image.shape
    out_height = math.ceil(height / factor)
    out_width = math.ceil(width / factor)
    padded = np.pad(image, ((0, out_height * factor - height), (0, out_width * factor - width)), constant_values=np.nan)
    blocks = padded.reshape(out_height, factor, out_width, factor)
    finite = np.isfinite(blocks)
    sums = np.where(finite, blocks, 0.0).sum(axis=(1, 3), dtype=np.float64)
    counts = finite.sum(axis=(1, 3))
    out = np.full((out_height, out_width), np.nan, dtype=np.float32)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _scale_line(points: Sequence[Mapping[str, float]], factor: int) -> list[dict[str, float]]:
    """``{row, col}`` line points divided by ``factor``, so a line drawn on the full image lands on the binned one."""
    return [{"row": float(point["row"]) / factor, "col": float(point["col"]) / factor} for point in points]
