"""Static PNG fallback for Show2D: the saved-notebook preview and the panel
plan that export_svg shares with it.

``_static_png_b64`` mirrors the live widget panel for panel (same colormap,
contrast window, zoom crop, overlays, scale bar) so a notebook reopened
without a kernel shows what the user saw. ``_static_panel_specs`` and
``_static_overlay_texts`` are the shared plan; tests assert the strings
without OCR-ing the PNG.
"""

import base64
import io
import math

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.patches
import matplotlib.patheffects
import numpy as np

from quantem.widget.colormap import colorize
from quantem.widget.render.figure import format_scale_label, round_to_nice, static_overlay_font


_IDENTITY_PALETTE = ("#2e7d32", "#c62828", "#d81b60", "#1565c0", "#f9a825", "#6a1b9a")


class Show2DStaticPng:
    """Mixin: the matplotlib static preview and the panel plan behind it."""

    # Colormaps the frontend treats as sequential; a signed diff panel switches
    # to a diverging map (RdBu) because zero must sit at the visual midpoint.
    _SEQUENTIAL_CMAPS = frozenset(
        {"inferno", "viridis", "plasma", "magma", "hot", "gray", "turbo"}
    )

    def _gallery_export_chrome(self) -> dict[str, int | float | str]:
        """Resolved gallery chrome (gaps, borders) for exports and previews.

        One place applies the fallbacks (an unset outer-border color takes the
        gap color) so the SVG export and the PNG preview draw the same frame.
        """
        gap_color = str(self.inter_panel_gap_color or "")
        return {
            "inter_panel_gap_px": max(0, int(self.inter_panel_gap_px)),
            "inter_panel_gap_color": gap_color,
            "gallery_outer_border_px": max(0, int(self.gallery_outer_border_px)),
            "gallery_outer_border_color": str(self.gallery_outer_border_color or gap_color),
            "panel_inner_border_px": max(0.0, float(self.panel_inner_border_px)),
            "panel_inner_border_color": str(self.panel_inner_border_color or "#d0d0d0"),
        }

    @staticmethod
    def _signed_log1p(values: np.ndarray | float) -> np.ndarray | float:
        """Signed log1p, matching the frontend's ``applyLogScale``.

        The widget maps negative intensities to ``-log1p(-v)`` so diff-like
        data keeps its sign under log scale; a plain ``log1p(clip(v, 0))``
        would collapse everything below zero and diverge from the live render.
        """
        return np.sign(values) * np.log1p(np.abs(values))

    @staticmethod
    def _format_stat(value: float) -> str:
        """Format a statistic like the widget's stats row (JS ``formatNumber``).

        ``0`` stays ``"0"``; magnitudes >= 1000 or < 0.01 use two-decimal
        scientific notation with an unpadded exponent (``5.85e+3``, matching
        JS ``toExponential(2)``); everything else uses two fixed decimals.
        """
        if value == 0:
            return "0"
        if abs(value) >= 1000 or abs(value) < 0.01:
            mantissa, exponent = f"{value:.2e}".split("e")
            return f"{mantissa}e{int(exponent):+d}"
        return f"{value:.2f}"

    def _resolve_panel_display_ranges(self, frames: list[np.ndarray]) -> list[tuple[float, float]]:
        """Per-panel ``(vmin, vmax)`` in display space, mirroring the frontend.

        Precedence (same as the JS colormap effect in ``js/show2d/index.tsx``):
        per-image ``vmins[i]/vmaxs[i]`` beat the scalar ``vmin/vmax``, which
        beats ``auto_contrast`` (2/98 percentiles), which beats the frame's
        min/max. With ``link_contrast`` on a gallery and no explicit ranges,
        panels share one merged range so cross-panel intensities compare
        directly. Under ``log_scale`` the limits live in signed-log1p space,
        exactly like the shader. Ranges are computed on the FULL-resolution
        display frames so the PNG's contrast matches the widget even though
        the PNG pixels are later area-binned.
        """
        n_frames = len(frames)
        # RGB panels bypass the contrast pipeline entirely: their range is the
        # fixed display-ready [0, 1], and they are excluded from linked-contrast
        # merging so a [0, 1] overlay never drags a counts-scaled panel's window.
        is_rgb = [i < len(self.is_rgb) and bool(self.is_rgb[i]) for i in range(n_frames)]
        gray = [i for i in range(n_frames) if not is_rgb[i]]

        def to_display(value: float) -> float:
            """A data-space limit in display space (signed log1p under ``log_scale``)."""
            return float(self._signed_log1p(float(value))) if self.log_scale else float(value)

        has_absolute = self.vmin is not None and self.vmax is not None
        panel_vmins = list(self.vmins) if self.vmins else [None] * n_frames
        panel_vmaxs = list(self.vmaxs) if self.vmaxs else [None] * n_frames
        has_panel_range = [panel_vmins[i] is not None and panel_vmaxs[i] is not None for i in range(n_frames)]
        base_ranges: list[tuple[float, float]] = []
        for i, frame in enumerate(frames):
            if is_rgb[i]:
                base_ranges.append((0.0, 1.0))
            elif has_panel_range[i]:
                base_ranges.append((to_display(panel_vmins[i]), to_display(panel_vmaxs[i])))
            elif has_absolute:
                base_ranges.append((to_display(self.vmin), to_display(self.vmax)))
            else:
                base_ranges.append((to_display(frame.min()), to_display(frame.max())))
        linked_shared = (
            self.link_contrast and len(gray) >= 2 and not has_absolute and not any(has_panel_range)
        )
        shared_base = None
        if linked_shared:
            shared_base = (min(base_ranges[i][0] for i in gray), max(base_ranges[i][1] for i in gray))
        auto_ranges: list[tuple[float, float]] = []
        use_auto = self.auto_contrast and not has_absolute
        if use_auto:
            for i, frame in enumerate(frames):
                if has_panel_range[i] or is_rgb[i]:
                    auto_ranges.append(base_ranges[i])
                    continue
                processed = self._signed_log1p(frame) if self.log_scale else frame
                low, high = (float(value) for value in np.percentile(processed, (2, 98)))
                # Sparse/clustered data can collapse the 2-98% window to a
                # point; fall back to full extrema like computeAutoRange does.
                full_low, full_high = float(processed.min()), float(processed.max())
                if high - low <= max(1e-12, abs(full_high - full_low) * 1e-6):
                    low, high = full_low, full_high
                auto_ranges.append((low, high))
            if linked_shared:
                shared_auto = (min(auto_ranges[i][0] for i in gray), max(auto_ranges[i][1] for i in gray))
        ranges: list[tuple[float, float]] = []
        for i in range(n_frames):
            if is_rgb[i]:
                ranges.append((0.0, 1.0))
            elif use_auto and not has_panel_range[i]:
                ranges.append(shared_auto if linked_shared else auto_ranges[i])
            else:
                ranges.append(shared_base if linked_shared else base_ranges[i])
        return ranges

    def _static_panel_rgb(
        self,
        frame: np.ndarray,
        vmin: float,
        vmax: float,
        cmap_name: str,
        *,
        apply_log: bool | None = None,
    ) -> np.ndarray:
        """Colormap one panel exactly as the live widget maps pixels to colors.

        Pipeline: optional signed-log1p on the data, clip-normalize into the
        display-space ``[vmin, vmax]`` window, then look up the matplotlib
        colormap (the same LUT the JS mirrors). Returning explicit RGB uint8
        keeps matplotlib's own norm machinery out of the loop, so the PNG's
        pixel mapping is byte-identical to what tests can compute independently.
        """
        apply_log = self.log_scale if apply_log is None else apply_log
        processed = self._signed_log1p(frame) if apply_log else frame
        if vmax > vmin:
            normalized = np.clip((processed - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            normalized = np.zeros(processed.shape, dtype=np.float64)
        return colorize(normalized, cmap_name)

    def _static_panel_specs(self) -> list[dict]:
        """Panel plan for the static PNG, mirroring the live widget's layout.

        One entry per visible image panel, plus one signed diff panel per
        non-reference image when ``diff_mode`` is on (the widget renders
        ``ref - other`` with a symmetric range around zero and a diverging
        colormap). Each spec carries the full-resolution frame plus the
        resolved display-space contrast window and a stats line, so the PNG
        renderer only has to bin and colormap.
        """
        frames_source = self._display_data if self._display_data is not None else self._data
        if frames_source is None or len(frames_source) == 0:
            return []
        frames = list(frames_source)
        ranges = self._resolve_panel_display_ranges(frames)

        def stats_line(mean: float, low: float, high: float, std: float) -> str:
            """The widget's stats row text for one panel."""
            format_stat = self._format_stat
            return f"Mean {format_stat(mean)}   Min {format_stat(low)}   Max {format_stat(high)}   Std {format_stat(std)}"

        specs: list[dict] = []
        display_rgb = self._display_rgb or [None] * len(frames)
        for i in self._visible_panels():
            if i >= len(frames):
                continue
            label = self._panel_title_for_index(i) if self.show_panel_titles else ""
            panel_rgb = display_rgb[i] if i < len(display_rgb) else None
            frame = panel_rgb if panel_rgb is not None else frames[i]
            specs.append({
                # RGB panels pass their display-ready pixels straight through:
                # no colormap, no log, no contrast window (stats stay luminance).
                "frame": frame,
                "rgb": panel_rgb is not None,
                "vmin": ranges[i][0],
                "vmax": ranges[i][1],
                "cmap": self._panel_cmap_for_index(i),
                "apply_log": self.log_scale and panel_rgb is None,
                "label": label,
                "stats": stats_line(self.stats_mean[i], self.stats_min[i], self.stats_max[i], self.stats_std[i]),
                "panel_index": i,
            })
        if self.diff_mode and len(frames) >= 2:
            reference = int(self.diff_reference)
            reference_cmap = self._panel_cmap_for_index(reference)
            diff_cmap = "RdBu" if reference_cmap in self._SEQUENTIAL_CMAPS else reference_cmap
            for other in range(len(frames)):
                # RGB panels never get a diff panel: a signed residual against
                # a display-ready color composite is meaningless.
                if other == reference or (other < len(self.is_rgb) and self.is_rgb[other]):
                    continue
                diff = frames[reference] - frames[other]
                # Symmetric window centers zero on the diverging map's midpoint,
                # so positive and negative residuals read with equal weight.
                half_range = float(max(abs(float(diff.min())), abs(float(diff.max())))) or 1.0
                label = ("Diff (A − B)" if len(frames) == 2
                         else f"Diff (#{reference + 1} − #{other + 1})")
                specs.append({
                    "frame": diff,
                    "vmin": -half_range,
                    "vmax": half_range,
                    "cmap": diff_cmap,
                    "apply_log": False,  # widget diffs raw data, never log-scaled
                    "label": label if self.show_panel_titles else "",
                    "stats": stats_line(float(diff.mean()), float(diff.min()),
                                        float(diff.max()), float(diff.std())),
                    "panel_index": other,
                })
        return specs

    def _panel_cmap_for_index(self, panel: int) -> str:
        """Return a panel-specific colormap or the widget fallback colormap."""
        if 0 <= int(panel) < len(self.panel_cmaps):
            value = str(self.panel_cmaps[int(panel)])
            if value:
                return value
        return str(self.cmap)

    def _static_roi_items(self) -> list[dict[str, object]]:
        """Return visible ROI dictionaries with defaults for static rendering."""
        if not self.roi_active:
            return []
        rois: list[dict[str, object]] = []
        defaults = {
            "shape": "circle",
            "row": float(self.height) / 2.0,
            "col": float(self.width) / 2.0,
            "radius": 10.0,
            "radius_inner": 5.0,
            "width": 20.0,
            "height": 20.0,
            "line_width": 2.0,
            "color": "#4fc3f7",
            "visible": True,
        }
        for roi in self.roi_list:
            if not isinstance(roi, dict):
                continue
            item = {**defaults, **roi}
            if item.get("visible") is False:
                continue
            rois.append(item)
        return rois

    @staticmethod
    def _static_roi_extent(roi: dict[str, object]) -> tuple[float, float]:
        """Return ROI half-height and half-width in source pixels."""
        shape = str(roi.get("shape", "circle")).lower()
        if shape == "rectangle":
            return (
                max(1.0, float(roi.get("height", 20.0)) / 2.0),
                max(1.0, float(roi.get("width", 20.0)) / 2.0),
            )
        radius = max(1.0, float(roi.get("radius", 10.0)))
        return radius, radius

    @staticmethod
    def _static_roi_crop_slices(
        roi: dict[str, object],
        height: int,
        width: int,
        *,
        half_shape: tuple[float, float] | None = None,
    ) -> tuple[slice, slice]:
        """Crop around an ROI for the saved-notebook zoom panel."""
        center_row = float(roi.get("row", height / 2.0))
        center_col = float(roi.get("col", width / 2.0))
        half_h, half_w = half_shape or Show2DStaticPng._static_roi_extent(roi)
        # The saved zoom panel should show the ROI evidence itself, not the
        # whole surrounding field. Keep a small outline margin so the ROI border
        # is visible while most pixels come from the selected region.
        pad_h = max(2.0, half_h * 1.08)
        pad_w = max(2.0, half_w * 1.08)
        crop_h = max(1, int(math.ceil(2.0 * pad_h)))
        crop_w = max(1, int(math.ceil(2.0 * pad_w)))
        row0 = int(round(center_row - crop_h / 2.0))
        col0 = int(round(center_col - crop_w / 2.0))
        row0 = min(max(0, row0), max(0, height - crop_h))
        col0 = min(max(0, col0), max(0, width - crop_w))
        row1 = min(height, row0 + crop_h)
        col1 = min(width, col0 + crop_w)
        return slice(row0, row1), slice(col0, col1)

    def _static_roi_zoom_specs(self, specs: list[dict]) -> list[dict]:
        """Build one right-side zoom panel per visible ROI."""
        if len(specs) != 1 or len(self._visible_panels()) != 1 or self.diff_mode:
            return []
        rois = self._static_roi_items()
        if not rois:
            return []
        source = specs[0]
        frame = source["frame"]
        # One common crop size so every zoom panel shows the same magnification.
        extents = [self._static_roi_extent(roi) for roi in rois]
        common_half_extent = max(max(half_h, half_w) for half_h, half_w in extents)
        common_half_shape = (common_half_extent, common_half_extent)
        zooms: list[dict] = []
        for roi_number, roi in enumerate(rois, start=1):
            rows, cols = self._static_roi_crop_slices(
                roi,
                frame.shape[0],
                frame.shape[1],
                half_shape=common_half_shape,
            )
            zoom_roi = dict(roi)
            zoom_roi["row"] = float(zoom_roi.get("row", 0.0)) - rows.start
            zoom_roi["col"] = float(zoom_roi.get("col", 0.0)) - cols.start
            zooms.append({
                **source,
                "frame": frame[rows, cols],
                "label": f"ROI {roi_number} zoom",
                "stats": "",
                "roi_items": [zoom_roi],
                "roi_zoom_panel": True,
                "source_crop": (rows.start, cols.start, rows.stop, cols.stop),
            })
        return zooms

    @staticmethod
    def _center_crop_slices(height: int, width: int, zoom: float) -> tuple[slice, slice]:
        """Central 1/zoom crop, matching the live widget's zoomed viewport.

        The widget at zoom z (pan 0) scales the full image about the canvas
        center, so the visible region is the central ``height/z x width/z``
        window. The static PNG must show the same pixels or the fallback
        looks nothing like the screenshot the user saved."""
        if zoom <= 1:
            return slice(0, height), slice(0, width)
        crop_h = max(1, math.floor(height / zoom + 0.5))
        crop_w = max(1, math.floor(width / zoom + 0.5))
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        return slice(top, top + crop_h), slice(left, left + crop_w)

    @staticmethod
    def _downsample_static_frame(frame: np.ndarray, max_px: int) -> np.ndarray:
        """Area-downsample a frame for notebook PNG fallback rendering."""
        if max_px <= 0:
            return frame
        height, width = frame.shape[-2:]
        factor = max(1, int(math.ceil(max(height, width) / max_px)))
        if factor == 1:
            return frame
        trimmed_h = max(1, height // factor) * factor
        trimmed_w = max(1, width // factor) * factor
        trimmed = frame[:trimmed_h, :trimmed_w]
        return trimmed.reshape(
            trimmed_h // factor,
            factor,
            trimmed_w // factor,
            factor,
        ).mean(axis=(1, 3))

    def _static_canvas_css_px(self) -> float:
        """CSS width of the live panel canvas, the length every JS overlay
        constant (16px font, 12px margin, 60px bar target) is relative to.

        Port of js/show2d/index.tsx: the ``size`` trait when set, else
        SINGLE_IMAGE_TARGET (500) for one image / GALLERY_IMAGE_TARGET (300)
        for a gallery. Without this the static overlays would be drawn for a
        fictitious canvas size and read visibly smaller than the widget's."""
        if self.size > 0:
            return float(self.size)
        return 300.0 if self.n_images > 1 else 500.0

    def _static_overlay_texts(self, specs: list[dict] | None = None,
                              *, css_px: float | None = None) -> list[tuple[str, str, str, float]]:
        """Per-panel overlay strings for the static PNG, one tuple
        ``(label, zoom_text, bar_text, bar_px)`` per panel.

        Pure port of js/figure.ts drawScaleBarHiDPI's math, evaluated on a
        ``css_px``-wide canvas (default: the live widget's own canvas CSS
        width from ``_static_canvas_css_px``): effectiveZoom =
        zoom * cssWidth / imageWidth, target bar 60 CSS px rounded to a nice
        physical length, label via formatScaleLabel. Uncalibrated data gets
        pixelSize 1 and unit "px" exactly like the widget (show2d/index.tsx
        overlay effect). ``bar_px`` is the bar length in panel CSS px;
        ``bar_text``/``bar_px`` are ""/0.0 when the scale bar is hidden.
        Exposed separately from the renderer so tests can assert the strings
        without OCR-ing the PNG."""
        if specs is None:
            specs = self._static_panel_specs()
        if css_px is None:
            css_px = self._static_canvas_css_px()
        # widget clamps initial_zoom to [MIN_ZOOM, MAX_ZOOM] (index.tsx)
        zoom = min(max(float(self.initial_zoom) or 1.0, 0.5), 20.0)
        zoom_text = f"{zoom:.1f}×" if self.show_zoom_indicator else ""  # JS: `${zoom.toFixed(1)}×`
        calibrated = self.pixel_size > 0
        pixel_size = self.pixel_size if calibrated else 1.0
        unit = self.pixel_unit if calibrated else "px"
        bar_panels = {int(panel) for panel in self.scale_bar_panels}
        texts: list[tuple[str, str, str, float]] = []
        for spec in specs:
            panel_index = int(spec.get("panel_index", len(texts)))
            if not self.scale_bar_visible or (bar_panels and panel_index not in bar_panels):
                texts.append((spec["label"], zoom_text, "", 0.0))
                continue
            full_w = spec["frame"].shape[1]
            effective_zoom = zoom * css_px / full_w
            # 60 CSS px target bar, rounded to a nice physical length
            if self.scale_bar_length is not None and float(self.scale_bar_length) > 0:
                nice = float(self.scale_bar_length)
            else:
                nice = round_to_nice(60.0 / effective_zoom * pixel_size)
            bar_px = nice / pixel_size * effective_zoom
            label = str(self.scale_bar_label or "") or format_scale_label(nice, unit)
            texts.append((spec["label"], zoom_text, label, bar_px))
        return texts

    @staticmethod
    def _draw_static_roi(
        ax: matplotlib.axes.Axes,
        roi: dict[str, object],
        *,
        source_rows: slice,
        source_cols: slice,
        bin_h: int,
        bin_w: int,
        points_per_css_px: float,
    ) -> None:
        """Draw one ROI in the saved PNG's image-pixel coordinate system.

        The ROI lives in source pixels; the panel shows ``source_rows`` x
        ``source_cols`` binned to ``bin_h`` x ``bin_w``, so the center and size
        scale by that ratio. A translucent black halo keeps the outline visible
        on bright and dark pixels alike.
        """
        crop_h = max(1, source_rows.stop - source_rows.start)
        crop_w = max(1, source_cols.stop - source_cols.start)
        scale_y = bin_h / crop_h
        scale_x = bin_w / crop_w
        row = float(roi.get("row", 0.0))
        col = float(roi.get("col", 0.0))
        center_x = (col - source_cols.start) * scale_x - 0.5
        center_y = (row - source_rows.start) * scale_y - 0.5
        if center_x < -bin_w or center_x > 2 * bin_w or center_y < -bin_h or center_y > 2 * bin_h:
            return
        hex_color = str(roi.get("color") or "#4fc3f7").lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(digit * 2 for digit in hex_color)
        color = tuple(int(hex_color[start:start + 2], 16) / 255.0 for start in (0, 2, 4)) if len(hex_color) == 6 else (0.31, 0.76, 0.97)
        line_width = max(1.0, float(roi.get("line_width", 2.0))) * points_per_css_px
        outline = {
            "fill": False,
            "edgecolor": color,
            "linewidth": line_width,
            "path_effects": [matplotlib.patheffects.withStroke(
                linewidth=line_width + 1.5 * points_per_css_px,
                foreground=(0, 0, 0, 0.65),
            )],
        }
        shape = str(roi.get("shape", "circle")).lower()
        if shape in {"rectangle", "square"}:
            if shape == "rectangle":
                half_h = max(1.0, float(roi.get("height", 20.0)) / 2.0) * scale_y
                half_w = max(1.0, float(roi.get("width", 20.0)) / 2.0) * scale_x
            else:
                radius = max(1.0, float(roi.get("radius", 10.0)))
                half_h, half_w = radius * scale_y, radius * scale_x
            ax.add_patch(matplotlib.patches.Rectangle(
                (center_x - half_w, center_y - half_h), 2 * half_w, 2 * half_h, **outline,
            ))
            return
        radius = max(1.0, float(roi.get("radius", 10.0)))
        ax.add_patch(matplotlib.patches.Ellipse(
            (center_x, center_y), 2 * radius * scale_x, 2 * radius * scale_y, **outline,
        ))
        if shape == "annular":
            inner = max(0.5, float(roi.get("radius_inner", 5.0)))
            ax.add_patch(matplotlib.patches.Ellipse(
                (center_x, center_y), 2 * inner * scale_x, 2 * inner * scale_y, linestyle="--", **outline,
            ))

    def _draw_static_inset_plot(
        self,
        ax: matplotlib.axes.Axes,
        spec: dict[str, object],
        *,
        panel_index: int,
        line_color: str,
    ) -> None:
        """Draw a compact per-panel calibration curve in the saved PNG.

        Placement reads the same ``box`` / ``position`` + ``margin`` keys as
        the SVG inset, as axes fractions, and a bottom-right inset lifts above
        the scale bar so the two never overlap.
        """
        if not spec:
            return
        x_values = np.asarray(spec.get("x"), dtype=float).ravel()
        y_values = np.asarray(spec.get("y"), dtype=float).ravel()
        if x_values.size != y_values.size or x_values.size < 2:
            return
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        if finite.sum() < 2:
            return
        x_values = x_values[finite]
        y_values = y_values[finite]
        xlim = tuple(float(value) for value in spec.get("xlim", (float(x_values.min()), float(x_values.max()))))
        ylim = tuple(float(value) for value in spec.get("ylim", (float(y_values.min()), float(y_values.max()))))
        if xlim[1] <= xlim[0]:
            pad = max(1.0, abs(xlim[0]) * 0.05)
            xlim = (xlim[0] - pad, xlim[1] + pad)
        if ylim[1] <= ylim[0]:
            pad = max(1.0, abs(ylim[0]) * 0.05)
            ylim = (ylim[0] - pad, ylim[1] + pad)
        position = str(spec.get("position", "bottom-right"))
        size = max(0.18, min(0.55, float(spec.get("size", 0.31))))
        box_w = min(0.62, size)
        box_h = min(0.55, float(spec.get("height", box_w * 0.68)))
        if "box" in spec:
            left, top, width, height = (float(value) for value in spec["box"])
            box_w = max(0.05, min(0.95, width))
            box_h = max(0.05, min(0.95, height))
            box_x = max(0.0, min(1.0 - box_w, left))
            box_y = max(0.0, min(1.0 - box_h, 1.0 - top - box_h))
        else:
            # ``margin`` is in CSS px; /300 maps it onto a 300 px gallery panel's axes fractions.
            margin = spec.get("margin", (0.035, 0.035))
            if isinstance(margin, (int, float)):
                margin_x = margin_y = float(margin) / 300.0
            else:
                margin_values = list(margin)
                margin_x = float(margin_values[0]) / 300.0
                margin_y = float(margin_values[1]) / 300.0
            margin_x = max(0.0, min(0.45, margin_x))
            margin_y = max(0.0, min(0.45, margin_y))
            if "right" in position:
                box_x = 1.0 - box_w - margin_x
            elif "center" in position:
                box_x = 0.5 - box_w / 2
            else:
                box_x = margin_x
            if "bottom" in position:
                box_y = margin_y + 0.08
            elif "center" in position:
                box_y = 0.5 - box_h / 2
            else:
                box_y = 1.0 - box_h - margin_y
            if self.scale_bar_visible and position == "bottom-right":
                box_y += 0.10
        inset = ax.inset_axes([box_x, box_y, box_w, box_h])
        background_alpha = max(0.0, min(1.0, float(spec.get("background_alpha", 0.68))))
        inset.set_facecolor(spec.get("background") or (0.04, 0.05, 0.07, background_alpha))
        for spine in inset.spines.values():
            spine.set_color(spec.get("border_color") or (1, 1, 1, 0.35))
            spine.set_linewidth(max(0.0, min(6.0, float(spec.get("border_width", 1.0)))) * 0.6)
        inset.plot(
            x_values,
            y_values,
            color=spec.get("color") or line_color,
            linewidth=max(1.4, float(spec.get("line_width", 2.0))),
            solid_capstyle="round",
        )
        if "point" in spec:
            point = np.asarray(spec["point"], dtype=float).ravel()
            if point.size == 2 and np.isfinite(point).all():
                inset.scatter(
                    [point[0]],
                    [point[1]],
                    s=18,
                    color=spec.get("point_color") or "white",
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=5,
                )
        inset.set_xlim(*xlim)
        inset.set_ylim(*ylim)
        show_ticks = bool(spec.get("show_ticks", False))
        tick_font_size = max(4.0, min(12.0, float(spec.get("tick_font_size", 4.5))))
        label_font_size = max(4.0, min(14.0, float(spec.get("label_font_size", 4.5))))
        legend_font_size = max(4.0, min(14.0, float(spec.get("legend_font_size", 5.5))))
        text_color = spec.get("text_color") or "white"
        tick_color = spec.get("tick_color") or (1, 1, 1, 0.72)
        if show_ticks:
            if "xticks" in spec:
                inset.set_xticks([float(tick) for tick in spec["xticks"]])
            else:
                inset.set_xticks([xlim[0], xlim[1]])
            if "yticks" in spec:
                inset.set_yticks([float(tick) for tick in spec["yticks"]])
            else:
                inset.set_yticks([ylim[0], ylim[1]])
            inset.tick_params(
                axis="both",
                colors=tick_color,
                labelsize=tick_font_size,
                length=1.5,
                width=0.4,
                pad=1,
            )
        else:
            inset.set_xticks([])
            inset.set_yticks([])
        if spec.get("title"):
            inset.set_title(str(spec["title"]), color=text_color, fontsize=legend_font_size, pad=1.5, weight="bold")
        if spec.get("xlabel"):
            inset.set_xlabel(str(spec["xlabel"]), color=tick_color, fontsize=label_font_size, labelpad=0.5)
        if spec.get("ylabel"):
            inset.set_ylabel(str(spec["ylabel"]), color=tick_color, fontsize=label_font_size, labelpad=0.5)
        for text_key, position_key, default_color in (
            ("legend", "legend_position", spec.get("text_color") or spec.get("color") or line_color),
            ("annotation", "annotation_position", text_color),
        ):
            if spec.get(text_key):
                text_position = str(spec.get(position_key, "top-left" if text_key == "legend" else "top-right"))
                text_x = 0.96 if "right" in text_position else 0.04
                text_y = 0.07 if "top" in text_position else 0.93
                inset.text(
                    text_x,
                    text_y,
                    str(spec[text_key]),
                    transform=inset.transAxes,
                    ha="right" if "right" in text_position else "left",
                    va="top" if "top" in text_position else "bottom",
                    color=default_color,
                    fontsize=legend_font_size,
                    weight="bold",
                    path_effects=[
                        matplotlib.patheffects.withStroke(
                            linewidth=0.8,
                            foreground=(0, 0, 0, 0.7),
                        )
                    ],
                )
        if spec.get("show_panel_index", False):
            inset.text(
                0.98,
                0.03,
                str(panel_index + 1),
                transform=inset.transAxes,
                ha="right",
                va="bottom",
                color=(1, 1, 1, 0.42),
                fontsize=4.0,
            )

    def _static_png_b64(self, *, max_px: int = 512, dpi: int = 160) -> str | None:
        """Base64 PNG of all panels, attached to the cell output.

        With ``save_state`` False the interactive widget state is not embedded,
        so a reopened notebook (GitHub, nbviewer, cold Lab) would show nothing.
        This render mirrors the live widget panel-for-panel: same colormap,
        same per-panel or linked contrast window (resolved on the
        full-resolution frame so percentile cuts match the widget, then
        applied to the binned pixels), the same central 1/zoom viewport,
        diff panel(s) when ``diff_mode`` is on, and the widget's own in-panel
        overlays - label top-center, zoom badge bottom-left, scale bar with
        its label bottom-right - at the exact CSS-pixel geometry the JS draws
        on a panel of this size. Panels are area-mean binned to ~``max_px``
        so atomic-lattice detail averages instead of aliasing, and the render
        stays cheap on every display.
        """
        specs = self._static_panel_specs()
        if not specs:
            return None
        chrome = self._gallery_export_chrome()
        gap_px = chrome["inter_panel_gap_px"]
        gap_color = chrome["inter_panel_gap_color"]
        outer_px = chrome["gallery_outer_border_px"]
        outer_color = chrome["gallery_outer_border_color"]
        panel_border_px = chrome["panel_inner_border_px"]
        panel_border_color = chrome["panel_inner_border_color"]
        base_roi_items = self._static_roi_items()
        if base_roi_items:
            specs = [{**spec, "roi_items": base_roi_items} for spec in specs]
            if len(specs) == 1:
                specs.extend(self._static_roi_zoom_specs(specs))
        n_panels = len(specs)
        # Total-pixel budget: a large survey gallery (e.g. 38 panels) at a fixed
        # 512 px/panel produced a ~27 MB PNG and a ~73 MB notebook (noisy STEM
        # content compresses poorly, ~2.7 bytes/px). The fallback is a reopen
        # preview, not the data, so shrink per-panel resolution as the panel
        # count grows (~2 MP total -> a few MB, floor 160 px) instead of
        # scaling the file linearly with the gallery. Small galleries keep the
        # full 512 px/panel.
        budget_px = int((2_000_000 / n_panels) ** 0.5)
        max_px = max(160, min(max_px, budget_px))
        css_w = self._static_canvas_css_px()
        overlays = self._static_overlay_texts(specs, css_px=css_w)
        zoom = min(max(float(self.initial_zoom) or 1.0, 0.5), 20.0)
        has_roi_zoom = any(bool(spec.get("roi_zoom_panel")) for spec in specs)
        ncols = min(n_panels, 4) if has_roi_zoom else max(1, min(self.ncols, n_panels))
        nrows = (n_panels + ncols - 1) // ncols
        # cells sized to the panels' cropped aspect so every image fills its
        # cell exactly: a taller cell would pad panels with white and make the
        # horizontal gutters read wider than the vertical ones
        first_height, first_width = specs[0]["frame"].shape[:2]
        first_rows, first_cols = self._center_crop_slices(first_height, first_width, zoom)
        aspect = (first_rows.stop - first_rows.start) / (first_cols.stop - first_cols.start)
        # inter-panel gutter is the widget's own gallery gap (CSS px of the
        # live canvas), identical horizontally and vertically
        gap_frac = gap_px / css_w
        outer_frac = outer_px / css_w
        cell_w_in = max_px / dpi  # cell width in inches so 1 panel = max_px device px
        cell_h_in = cell_w_in * aspect
        gap_in = gap_frac * cell_w_in
        outer_in = outer_frac * cell_w_in
        fig_w_in = ncols * cell_w_in + (ncols - 1) * gap_in + 2 * outer_in
        fig_h_in = nrows * cell_h_in + (nrows - 1) * gap_in + 2 * outer_in
        # Build an unmanaged Figure rather than registering one through
        # pyplot. In a live Jupyter kernel this renderer runs from a
        # ``post_execute`` callback, alongside matplotlib-inline's own figure
        # flushing callback. A pyplot-managed multi-panel figure can be
        # cleared by that callback before ``savefig`` draws it, leaving a
        # correctly sized but completely white saved-notebook preview. A
        # standalone Figure has the same Agg rendering path without entering
        # Jupyter's global figure-manager lifecycle.
        fig = matplotlib.figure.Figure(figsize=(fig_w_in, fig_h_in))
        bg_color = outer_color if outer_px > 0 and outer_color else gap_color if gap_px > 0 and gap_color else "white"
        fig.patch.set_facecolor(bg_color)
        left = outer_in / fig_w_in if fig_w_in > 0 else 0.0
        right = 1.0 - left
        bottom = outer_in / fig_h_in if fig_h_in > 0 else 0.0
        top = 1.0 - bottom
        if gap_px > 0 and gap_color:
            fig.patches.append(
                matplotlib.patches.Rectangle(
                    (left, bottom),
                    max(0.0, right - left),
                    max(0.0, top - bottom),
                    transform=fig.transFigure,
                    facecolor=gap_color,
                    edgecolor="none",
                    zorder=-10,
                )
            )
        # wspace/hspace are fractions of cell width/height; both resolve to
        # the same gap_in inches so the white gutters match to the pixel
        grid = fig.add_gridspec(nrows, ncols, wspace=gap_frac,
                                hspace=gap_frac / aspect,
                                left=left, right=right, bottom=bottom, top=top)
        font_family = static_overlay_font()
        # the live canvas is css_w CSS px wide but the PNG panel is max_px
        # device px wide, so every CSS-px size renders scaled by max_px/css_w
        points_per_css_px = (max_px / css_w) * 72.0 / dpi
        for idx, (spec, (label, zoom_text, bar_text, bar_css)) in enumerate(zip(specs, overlays)):
            ax = fig.add_subplot(grid[idx // ncols, idx % ncols])
            ax.axis("off")
            frame = spec["frame"]
            rows, cols = self._center_crop_slices(frame.shape[0], frame.shape[1], zoom)
            if spec.get("rgb"):
                # RGB panels bypass the colormap: bin each channel and pass the
                # display-ready pixels straight through, matching the live view.
                cropped = frame[rows, cols]
                binned_channels = [self._downsample_static_frame(cropped[..., channel], max_px=max_px)
                                   for channel in range(3)]
                rgb = (np.clip(np.stack(binned_channels, axis=-1), 0.0, 1.0) * 255).astype(np.uint8)
            else:
                binned = self._downsample_static_frame(frame[rows, cols], max_px=max_px)
                rgb = self._static_panel_rgb(binned, spec["vmin"], spec["vmax"],
                                             spec["cmap"], apply_log=spec["apply_log"])
            ax.imshow(rgb, interpolation="nearest")
            bin_h, bin_w = rgb.shape[:2]
            ax.set(xlim=(-0.5, bin_w - 0.5), ylim=(bin_h - 0.5, -0.5))
            if panel_border_px > 0:
                ax.add_patch(matplotlib.patches.Rectangle(
                    (-0.5, -0.5),
                    bin_w,
                    bin_h,
                    facecolor="none",
                    edgecolor=panel_border_color,
                    linewidth=panel_border_px * points_per_css_px,
                    joinstyle="miter",
                ))
            for roi in spec.get("roi_items", []):
                self._draw_static_roi(
                    ax,
                    roi,
                    source_rows=rows,
                    source_cols=cols,
                    bin_h=bin_h,
                    bin_w=bin_w,
                    points_per_css_px=points_per_css_px,
                )
            # CSS px -> data px: the panel canvas is css_w CSS px wide showing
            # bin_w image pixels, so overlay geometry scales by bin_w / css_w
            css_h = css_w * bin_h / bin_w
            data_px_per_css_px = bin_w / css_w

            def css_xy(x_css: float, y_css: float) -> tuple[float, float]:
                """Panel CSS px to image-pixel axes coordinates (pixel centers sit at integers)."""
                return x_css * data_px_per_css_px - 0.5, y_css * data_px_per_css_px - 0.5

            # widget textShadow "1px 1px 0 rgba(0,0,0,0.85), 0 0 3px ..." reads
            # as a soft dark outline; a thin translucent black stroke is its
            # closest matplotlib match (a full-opacity stroke looks stenciled)
            stroke = [matplotlib.patheffects.withStroke(
                linewidth=1.5 * points_per_css_px, foreground=(0, 0, 0, 0.8))]
            if label:
                # panel title Box (show2d/index.tsx): top 6px inside the image,
                # centered, bold max(8, panel_title_font_size) px white @ 95%
                title_px = max(8, int(self.panel_title_font_size or 11))
                ax.text(*css_xy(css_w / 2, 6), label, color=(1, 1, 1, 0.95),
                        fontsize=title_px * points_per_css_px, fontweight="bold",
                        fontfamily=font_family, ha="center", va="top",
                        path_effects=stroke)
            if self.scale_bar_visible:
                # drawScaleBarHiDPI (js/figure.ts): margin 12, bar 5 px thick,
                # 16px label centered 4px above the bar, optional zoom badge
                # on the opposite corner sharing the bar's bottom edge.
                scale_left = self.scale_bar_position == "bottom-left"
                bar_x = 12 if scale_left else css_w - bar_css - 12
                bar_y = css_h - 12
                ax.add_patch(matplotlib.patches.Rectangle(
                    css_xy(bar_x, bar_y), bar_css * data_px_per_css_px, 5 * data_px_per_css_px,
                    facecolor="white", edgecolor="none"))
                ax.text(*css_xy(bar_x + bar_css / 2, bar_y - 4), bar_text,
                        color="white", fontsize=16 * points_per_css_px,
                        fontfamily=font_family, ha="center", va="bottom",
                        path_effects=stroke)
                if self.show_zoom_indicator:
                    zoom_x = css_w - 12 if scale_left else 12
                    zoom_ha = "right" if scale_left else "left"
                    ax.text(*css_xy(zoom_x, css_h - 12 + 5), zoom_text,
                            color="white", fontsize=16 * points_per_css_px,
                            fontfamily=font_family, ha=zoom_ha, va="bottom",
                            path_effects=stroke)
            panel_index = int(spec.get("panel_index", idx))
            if panel_index < len(self.inset_plots):
                line_color = (
                    self.marker_colors[panel_index]
                    if panel_index < len(self.marker_colors)
                    else _IDENTITY_PALETTE[panel_index % len(_IDENTITY_PALETTE)]
                )
                self._draw_static_inset_plot(
                    ax,
                    self.inset_plots[panel_index],
                    panel_index=panel_index,
                    line_color=line_color,
                )
        png_buffer = io.BytesIO()
        fig.savefig(png_buffer, format="png", dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.05)
        return base64.b64encode(png_buffer.getvalue()).decode("ascii")
