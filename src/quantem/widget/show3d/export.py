"""Show3D exports: standalone HTML, single-frame images and GIF.

The widget toolbar sends JSON export requests over the ``export_request``
trait; the same methods serve notebook calls (``export_html``, ``save_gif``,
``save_image``).
"""

import pathlib
import tempfile
from typing import Self

import numpy as np
from PIL import Image

from quantem.widget.colormap import colorize
from quantem.widget.export import export_slug
from quantem.widget.render import gif as gif_utils
from quantem.widget.utils.array import bin2d

# A single HTML much larger than this often fails to open under Chrome
# file://; export_html refuses past it and names the smaller encodings.
_HTML_EXPORT_SAFE_MB = 80.0


def _optional_int(value) -> int | None:
    """An integer option from a toolbar payload or a caller, where empty, 0 and None all mean unset."""
    return None if value in (None, "", 0, "0") else int(value)


class Show3DExport:
    """HTML, image and animation export for Show3D."""

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | None = None,
        quantized: bool | None = None,
        max_mb: float | None = _HTML_EXPORT_SAFE_MB,
    ) -> pathlib.Path:
        """Write a standalone HTML viewer.

        ``encoding="full"`` embeds the exact float32 display stack;
        ``encoding="uint8"`` writes a quantized pack with per-panel ranges and
        accepts ``downsample`` 2, 4 or 8 for compact reports. ``quantized`` is
        an alias for ``encoding="uint8"``. Exports above ``max_mb`` are refused
        with the smaller options named; pass ``max_mb=None`` to force.
        """
        if self._data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        quantized, downsample_factor = self._normalise_html_export_options(
            mode=mode, encoding=encoding, downsample=downsample, quantized=quantized
        )
        if max_mb is not None:
            estimate_mb = self._estimate_html_export_mb(quantized=quantized, downsample=downsample_factor)
            if estimate_mb > float(max_mb):
                uint8_mb = self._estimate_html_export_mb(quantized=True, downsample=downsample_factor)
                raise ValueError(
                    f"This export would embed about {estimate_mb:.0f} MB into one HTML file, "
                    f"above the {float(max_mb):.0f} MB safe limit (large single-file exports often "
                    f"fail to open under Chrome file://). Options: encoding='uint8' "
                    f"(about {uint8_mb:.0f} MB), downsample=2 or 4 to shrink spatially, or pass "
                    f"max_mb={estimate_mb:.0f} to force this size."
                )
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path(quantized, downsample=downsample_factor)
        self._write_html_export(export_path, quantized=quantized, downsample=downsample_factor, title=title)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, {self._export_mode_label(quantized, downsample=downsample_factor)})"
        return export_path

    def _normalise_html_export_options(
        self,
        *,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | None = None,
        quantized: bool | None = None,
    ) -> tuple[bool, int]:
        """``(quantized, downsample)`` from the export arguments; downsampling is refused for exact float32."""
        raw_mode = str(mode or "single").strip().lower().replace("_", "-")
        if raw_mode in {"exact", "full"}:
            raw_mode, encoding = "single", "full"
        elif raw_mode in {"quantized", "uint8", "u8"}:
            raw_mode, encoding = "single", "uint8"
        if raw_mode != "single":
            raise ValueError("Show3D HTML export supports only mode='single'. Use encoding='uint8' and downsample=2, 4, or 8 for a smaller report.")
        if downsample in (None, "", 0, "0"):
            downsample_factor = 1
        else:
            if isinstance(downsample, bool):
                raise ValueError("Show3D HTML export downsample must be an integer factor, not bool")
            downsample_factor = int(downsample)
        if downsample_factor not in {1, 2, 4, 8}:
            raise ValueError("Show3D HTML export downsample must be one of 1, 2, 4, or 8")
        raw_encoding = str(encoding or "full").strip().lower().replace("_", "-")
        if quantized is True:
            raw_encoding = "uint8"
        if raw_encoding in {"full", "exact", "float32", "f32"}:
            if downsample_factor != 1:
                raise ValueError("Show3D exact float32 HTML export does not support downsample; use encoding='uint8'")
            return False, 1
        if raw_encoding in {"uint8", "u8", "quantized"}:
            return True, downsample_factor
        raise ValueError(f"unknown Show3D export encoding {encoding!r}; expected 'full' or 'uint8'")

    def _estimate_html_export_mb(self, *, quantized: bool, downsample: int) -> float:
        """Rough embedded size: base64 inflates the stack bytes by 4/3 plus a 2 MB runtime."""
        elements = int(np.prod(self._offline_stack_source().shape)) // max(1, int(downsample) ** 2)
        return elements * (1 if quantized else 4) * (4.0 / 3.0) / (1024 * 1024) + 2.0

    def _default_html_export_path(self, quantized: bool, *, downsample: int = 1) -> pathlib.Path:
        """File in the kernel cwd named after the title, stack shape and encoding (``HtmlExportMixin`` hook)."""
        slug = export_slug(self.title, "show3d")
        suffix = f"_{int(downsample)}xbin" if quantized and int(downsample) > 1 else ""
        return pathlib.Path.cwd() / f"{slug}_{self.n_slices}x{self.height}x{self.width}_{'quantized' if quantized else 'exact'}{suffix}.html"

    def _default_gif_export_path(self, quality: str = "medium") -> pathlib.Path:
        """File in the kernel cwd for a toolbar GIF, named after the title, stack shape and quality."""
        return pathlib.Path.cwd() / f"{export_slug(self.title, 'show3d')}_{self.n_slices}x{self.height}x{self.width}_{quality}.gif"

    def _export_data_args(self, *, downsample: int = 1) -> tuple[np.ndarray, ...]:
        """Display-shaped per-panel stacks so the export clone rebuilds the same panels."""
        if self._display_data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        downsample = int(downsample)

        def binned(panel: np.ndarray) -> np.ndarray:
            """One gray panel stack mean-binned by ``downsample``, as contiguous float32."""
            if downsample > 1:
                return np.ascontiguousarray(bin2d(panel, factor=downsample, mode="mean"), dtype=np.float32)
            return np.ascontiguousarray(panel, dtype=np.float32)

        def binned_rgb(stack: np.ndarray) -> np.ndarray:
            """One ``(N, H, W, 3)`` color stack mean-binned per channel by ``downsample``."""
            if downsample <= 1:
                return np.ascontiguousarray(stack, dtype=np.float32)
            frames = [np.stack([bin2d(stack[frame, ..., channel], factor=downsample, mode="mean") for channel in range(3)], axis=-1)
                      for frame in range(stack.shape[0])]
            return np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.float32)

        n_panels = int(self.n_panels)
        if self.is_rgb and self._rgb_data is not None:
            if n_panels > 1:
                panel_width = int(self.panel_width_px) or self._rgb_data.shape[2] // n_panels
                return tuple(binned_rgb(self._rgb_data[:, :, i * panel_width : (i + 1) * panel_width]) for i in range(n_panels))
            return (binned_rgb(self._rgb_data),)
        if self.shared_panel_source and n_panels > 1:
            # one stack object repeated, so the clone sends it once too
            shared = binned(self._display_data)
            return tuple(shared for _ in range(n_panels))
        panel_width = int(self.panel_width_px)
        if n_panels > 1 and panel_width > 0 and int(self.width) == panel_width * n_panels:
            return tuple(binned(self._display_data[:, :, i * panel_width : (i + 1) * panel_width]) for i in range(n_panels))
        return (binned(self._display_data),)

    def _clone_for_html_export(self, *, quantized: bool, downsample: int = 1) -> Self:
        """An export-only widget carrying the current view state and requested packing."""
        downsample = int(downsample)
        clone = type(self)(
            *self._export_data_args(downsample=downsample),
            labels=list(self.labels) if self.labels else None,
            panel_titles=list(self.panel_titles) if self.panel_titles else None,
            title=self.title,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            sampling=self.pixel_size * downsample if self.pixel_size > 0 else None,
            units=self.pixel_unit,
            smooth=self.smooth,
            auto_contrast=self.auto_contrast,
            fps=self.fps,
            avg_window=self.avg_window,
            show_title=self.show_title,
            show_fft=self.show_fft,
            fft_layout=self.fft_layout,
            fft_overlay_zoom=self.fft_overlay_zoom,
            show_stats=self.show_stats,
            panel_width_px=self.size,
            dim_label=self.dim_label,
            display_bin=1,
            max_cols=self.max_cols,
            panel_gap=self.panel_gap,
            show_scale_bar=self.scale_bar_visible,
            panel_annotations=list(self.panel_annotations),
            panel_overlays=list(self.panel_overlays),
            verbose=False,
        )
        clone.panel_real_frames = list(self.panel_real_frames)
        clone.n_pages = int(self.n_pages)
        clone.panels_per_page = int(self.panels_per_page)
        clone.page_labels = list(self.page_labels)
        clone.page_starred = list(self.page_starred)
        clone.page_idx = int(self.page_idx)
        clone.panel_title_spans = list(self.panel_title_spans)
        clone.load_state_dict(self.state_dict())
        # The saved state carries the live pixel size; a binned export needs it
        # scaled so the scale bar stays physically correct.
        if self.pixel_size > 0:
            clone.pixel_size = self.pixel_size * downsample
        if quantized:
            clone._pack_offline_u8_stack()
        else:
            clone._pack_exact_offline_stack()
        clone.playing = bool(self.playing)
        clone.export_enabled = False
        clone.export_status = ""
        clone._export_light = True
        # get_state() drops the pixel traits unless the clone opts into saving them.
        clone._save_state = True
        return clone

    def _release_export_clone(self, clone) -> None:
        """Free the clone's stack copy (observers pin a widget, so ``close`` alone would keep it), then close its model."""
        clone.free()
        clone.close()

    def _html_export_options(self, payload: dict, mode: str) -> dict:
        """The ``export_html`` keywords of a toolbar request (``HtmlExportMixin`` hook)."""
        quantized, downsample = self._normalise_html_export_options(
            mode=mode, encoding=str(payload.get("encoding", "full")), downsample=payload.get("downsample")
        )
        return {"quantized": quantized, "downsample": downsample}

    def _export_request(self, payload: dict, mode: str) -> None:
        """GIF toolbar requests; HTML goes through the shared export handshake."""
        if mode != "gif":
            super()._export_request(payload, mode)
            return
        quality = self._normalise_animation_quality(payload.get("quality", "medium"))

        def flag(key: str) -> bool | None:
            """A payload switch as a bool (the strings ``"1"``, ``"true"``, ``"yes"``, ``"on"`` count), None when absent."""
            if key not in payload:
                return None
            if isinstance(payload[key], str):
                return payload[key].strip().lower() in {"1", "true", "yes", "on"}
            return bool(payload[key])

        kwargs = {
            "fps": None if payload.get("fps") in (None, "", 0, "0") else float(payload["fps"]),
            "playback": str(payload.get("playback") or "forward"),
            "frame_start": _optional_int(payload.get("frame_start")),
            "frame_stop": _optional_int(payload.get("frame_stop")),
            "every_n": max(1, int(payload.get("every_n", 1) or 1)),
            "max_frames": _optional_int(payload.get("max_frames")),
            "downsample": self._normalise_animation_downsample(payload.get("downsample", 1)),
            "max_edge_px": _optional_int(payload.get("max_edge_px")),
            "show_panel_titles": flag("show_panel_titles"),
            "show_scale_bar": flag("show_scale_bar"),
            "show_zoom": flag("show_zoom"),
            "slides_preset": bool(payload.get("slides_preset")),
        }
        order = self._animation_frame_order(
            kwargs["playback"], frame_start=kwargs["frame_start"], frame_stop=kwargs["frame_stop"],
            every_n=kwargs["every_n"], max_frames=kwargs["max_frames"],
        )
        parts = [f"{len(order)} frames"]
        if kwargs["frame_start"] is not None or kwargs["frame_stop"] is not None:
            first = 1 if kwargs["frame_start"] is None else kwargs["frame_start"] + 1
            last = int(self.n_slices) if kwargs["frame_stop"] is None else kwargs["frame_stop"]
            parts.append(f"{first}-{last}")
        if kwargs["every_n"] > 1:
            parts.append(f"every {kwargs['every_n']}")
        if kwargs["max_frames"]:
            parts.append(f"max {kwargs['max_frames']}")
        if kwargs["downsample"] > 1:
            parts.append(f"{kwargs['downsample']}x downsample")
        if kwargs["max_edge_px"] is not None:
            parts.append(f"max edge {kwargs['max_edge_px']} px")
        if kwargs["slides_preset"]:
            parts.append("Slides preset")
        detail = ", ".join(parts)
        filename = str(payload.get("filename") or self._default_gif_export_path(quality).name)
        if payload.get("download"):
            self.export_status = f"Preparing {filename} ({detail})..."
            with tempfile.TemporaryDirectory(prefix="show3d-animation-export-") as tmp:
                path = pathlib.Path(tmp) / filename
                self.save_gif(path, quality=quality, **kwargs)
                media = path.read_bytes()
            self.export_filename = filename
            self.export_payload = media
            self.export_payload_id = str(payload.get("id") or "")
            self.export_status = f"Ready {filename} ({len(media) / (1024 * 1024):.1f} MB, {detail})"
        else:
            self.export_status = f"Exporting {filename} ({detail})..."
            path = self._default_gif_export_path(quality)
            self.save_gif(path, quality=quality, **kwargs)
            self.export_status = f"Exported {path.name} ({path.stat().st_size / (1024 * 1024):.1f} MB, {detail})"

    # --- single frame ------------------------------------------------------------

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Map one frame to uint8 with the current contrast, log scale and diff range.

        Signed log keeps negative values (diffs, phase) from collapsing to zero,
        matching the browser's ``slog``.
        """
        if self.log_scale:
            frame = np.sign(frame) * np.log1p(np.abs(frame))
        if self.vmin is not None or self.vmax is not None:
            vmin = float(self.vmin if self.vmin is not None else self.data_min)
            vmax = float(self.vmax if self.vmax is not None else self.data_max)
            if self.log_scale:
                vmin = float(np.sign(vmin) * np.log1p(abs(vmin)))
                vmax = float(np.sign(vmax) * np.log1p(abs(vmax)))
        elif self.auto_contrast:
            vmin = float(np.percentile(frame, self.percentile_low))
            vmax = float(np.percentile(frame, self.percentile_high))
        else:
            vmin, vmax = float(self.data_min), float(self.data_max)
        if vmax > vmin:
            return np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    def save_image(self, path: str | pathlib.Path, *, frame_idx: int | None = None, format: str | None = None, dpi: int = 150) -> pathlib.Path:
        """Save one frame as PNG, PDF or TIFF with the current colormap and contrast."""
        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")
        idx = self.slice_idx if frame_idx is None else frame_idx
        if not 0 <= idx < self.n_slices:
            raise IndexError(f"Frame index {idx} out of range [0, {self.n_slices})")
        if self.is_rgb and self._rgb_data is not None:
            rgba = np.clip(self._rgb_data[idx] * 255.0, 0, 255).astype(np.uint8)
        else:
            frame = self._data[idx]
            if self.diff_mode == "previous":
                frame = frame - self._data[idx - 1] if idx > 0 else np.zeros_like(frame)
            elif self.diff_mode == "first":
                frame = frame - self._data[0]
            rgba = colorize(self._normalize_frame(frame) / 255.0, self.cmap)
        image = Image.fromarray(rgba)
        if fmt == "pdf":
            image = image.convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(path), dpi=(dpi, dpi))
        return path

    # --- animations ----------------------------------------------------------------

    def save_gif(
        self,
        path: str | pathlib.Path,
        *,
        quality: str = "high",
        fps: float | None = None,
        playback: str = "forward",
        show_frame_labels: bool = False,
        background: str | tuple[int, int, int] = "dark",
        frame_start: int | None = None,
        frame_stop: int | None = None,
        every_n: int = 1,
        max_frames: int | None = None,
        downsample: int = 1,
        max_edge_px: int | None = None,
        show_panel_titles: bool | None = None,
        show_scale_bar: bool | None = None,
        show_zoom: bool | None = None,
        slides_preset: bool = False,
    ) -> pathlib.Path:
        """Save the visible panels as an animated GIF matching the live view.

        ``quality`` picks the resolution tier (high 1.0, medium 0.6, low 0.35).
        ``playback="bounce"`` plays forward then back without repeating the
        endpoints. ``slides_preset=True`` caps the export at 40 frames and
        512 px per panel unless overridden. Publication chrome from the live
        view (titles, scale bars, zoom labels, gaps) is kept; FFT, profiles,
        controls and hover or selection affordances are never rendered.
        """
        quality = self._normalise_animation_quality(quality)
        downsample = self._normalise_animation_downsample(downsample)
        if slides_preset:
            max_frames = 40 if max_frames is None else max_frames
            max_edge_px = 512 if max_edge_px is None and downsample == 1 else max_edge_px
        max_edge_px = _optional_int(max_edge_px)
        source_w = max(1, int(self.source_panel_width or self.panel_width_px or self.width))
        source_h = max(1, int(self.source_height or self.height))
        scale = gif_utils.animation_output_scale(source_w, source_h, quality, downsample=downsample, max_edge_px=max_edge_px)
        panel_gap = max(0, int(round(float(self.panel_gap) * scale)))
        include_scale_bar = bool(self.scale_bar_visible) if show_scale_bar is None else bool(show_scale_bar)
        include_zoom = bool(self.show_zoom_indicator) if show_zoom is None else bool(show_zoom)
        include_panel_titles = bool(self.show_panel_titles) if show_panel_titles is None else bool(show_panel_titles)
        pixel_size = float(self.pixel_size) / max(1, int(self.display_bin)) if include_scale_bar else 0.0
        unit = self.pixel_unit or "A"
        panel_indices = self._visible_panels
        if not panel_indices:
            raise ValueError("cannot export animation with every panel hidden")
        panel_titles = [self._panel_title_for_index(panel) for panel in panel_indices]
        frames = []
        for frame_idx in self._animation_frame_order(playback, frame_start=frame_start, frame_stop=frame_stop, every_n=every_n, max_frames=max_frames):
            panel_images = [
                gif_utils.finalize_frame(
                    gif_utils.colorize(self._normalize_frame(self._get_source_panel_frame(panel, frame_idx)), self.cmap),
                    quality,
                    pixel_size,
                    unit,
                    show_zoom_indicator=include_zoom,
                    downsample=downsample,
                    max_edge_px=max_edge_px,
                )
                for panel in panel_indices
            ]
            frames.append(
                gif_utils.compose_panel_grid(
                    panel_images,
                    panel_titles=[self._static_panel_title(panel, frame_idx) for panel in panel_indices] if show_frame_labels else panel_titles,
                    frame_labels=None,
                    show_panel_titles=include_panel_titles,
                    title_font_size=max(gif_utils.MIN_TITLE_FONT_SIZE, int(round(11 * scale))),
                    max_cols=int(self.max_cols),
                    panel_gap=panel_gap,
                    background=background,
                    outer_border=0,
                    outer_border_color=background,
                    panel_inner_border=0,
                    panel_inner_border_color="black",
                )
            )
        return gif_utils.write_gif(frames, path, fps=float(self.fps) if fps is None else float(fps))

    def _animation_frame_order(
        self,
        playback: str,
        *,
        frame_start: int | None = None,
        frame_stop: int | None = None,
        every_n: int = 1,
        max_frames: int | None = None,
    ) -> list[int]:
        """Frame indices in export order; ``max_frames`` samples evenly across the range."""
        n_slices = int(self.n_slices)
        start = max(0, min(n_slices, 0 if frame_start is None else int(frame_start)))
        stop = max(start, min(n_slices, n_slices if frame_stop is None else int(frame_stop)))
        frames = list(range(start, stop))[:: max(1, int(every_n))]
        if max_frames not in (None, 0, "", "0") and len(frames) > max(1, int(max_frames)):
            keep = np.linspace(0, len(frames) - 1, max(1, int(max_frames))).round().astype(int)
            frames = [frames[int(i)] for i in keep]
        if not frames:
            raise ValueError("animation export frame range does not include any frames")
        mode = str(playback).lower()
        if mode == "forward":
            return frames
        if mode in {"bounce", "boomerang"}:
            return frames + frames[-2:0:-1] if len(frames) > 1 else frames
        raise ValueError("playback must be 'forward' or 'bounce'")

    def _normalise_animation_quality(self, quality: object) -> str:
        """A GIF quality tier name (``gif_utils.QUALITY_SCALE``); unknown tiers are refused."""
        value = str(quality or "medium").lower()
        if value not in gif_utils.QUALITY_SCALE:
            raise ValueError(f"animation quality must be one of {list(gif_utils.QUALITY_SCALE)}, got {quality!r}")
        return value

    def _normalise_animation_downsample(self, downsample: int | None) -> int:
        """GIF downsample factor 1, 2, 4 or 8, with empty values meaning 1; a bool is refused as a factor."""
        if downsample in (None, "", 0, "0"):
            return 1
        if isinstance(downsample, bool) or int(downsample) not in {1, 2, 4, 8}:
            raise ValueError("animation export downsample must be one of 1, 2, 4, or 8")
        return int(downsample)
