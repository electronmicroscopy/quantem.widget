"""Saved-notebook state and static PNG preview for Show3D.

A saved notebook keeps a compact JPEG/PNG of the current frame instead of the
multi-hundred-MB stack; this module renders that preview through Show2D so
fonts, gutters and scale bars match the gallery widget pixel for pixel.
"""

import numpy as np
from comm import DummyComm


class Show3DFallback:
    """The static preview and the pixel traits ``SavedStateMixin`` keeps out of a saved notebook."""

    # Traits that carry pixels; dropped from the saved-notebook snapshot.
    _UNSAVED_HEAVY_KEYS = ("_offline_stack", "_offline_float_stack", "export_payload")

    def _static_panel_title(self, panel: int, idx: int) -> str:
        """Panel title as the live canvas draws it: title, frame label, count."""
        panel_title = self._panel_title_for_index(panel)
        if int(self.n_panels) == 1 and self.title and panel_title == "Panel 1":
            title_text = self.title
        elif int(self.n_panels) > 1 or (panel < len(self.panel_titles) and self.panel_titles[panel]):
            title_text = panel_title
        else:
            title_text = self.title or panel_title
        frame_label = str(self.labels[idx]).strip() if idx < len(self.labels) else ""
        if frame_label in {str(idx), str(idx + 1)}:
            frame_label = ""
        panel_real = self.panel_real_frames[panel] if panel < len(self.panel_real_frames) else 0
        shown = min(idx + 1, int(panel_real)) if panel_real else idx + 1
        total = int(panel_real) if panel_real else int(self.n_slices)
        return f"{title_text}{f' · {frame_label}' if frame_label else ''} {shown}/{total}"

    def _static_png_b64(self, *, max_px: int = 512, dpi: int = 160) -> str | None:
        """Base64 PNG of the current frame, rendered by a Show2D gallery.

        Delegating to Show2D keeps fonts, gutters, colormap and scale bars
        pixel-identical between the two widgets' saved previews. The Show2D is
        a renderer, not a view: on a ``DummyComm`` it never opens a frontend
        model (no message, no frame copy to the browser), and ``close()``
        drops it from the widget registry, so every display leaves nothing open.
        """
        if self.n_slices <= 0 or self._display_data is None:
            return None
        from quantem.widget.show2d import Show2D

        panels = self._visible_panels or [0]
        idx = max(0, min(int(self.slice_idx), int(self.n_slices) - 1))
        if self.is_rgb and self._rgb_data is not None:
            frames = [np.ascontiguousarray(self._rgb_data[idx], dtype=np.float32)]
        else:
            frames = [np.asarray(self._get_display_panel_frame(panel, idx), dtype=np.float32) for panel in panels]
        vmin = [self.vmin_per_panel[panel] for panel in panels] if any(bound is not None for bound in self.vmin_per_panel) else self.vmin
        vmax = [self.vmax_per_panel[panel] for panel in panels] if any(bound is not None for bound in self.vmax_per_panel) else self.vmax
        ncols = int(self.max_cols) if int(self.max_cols) > 0 else len(frames)
        preview = Show2D(
            frames,
            labels=[self._static_panel_title(panel, idx) for panel in panels],
            title=self.title,
            cmap=[self.panel_cmaps[panel] for panel in panels] if self.panel_cmaps else self.cmap,
            sampling=self.pixel_size if self.pixel_size > 0 else None,
            units=self.pixel_unit,
            show_scale_bar=self.scale_bar_visible,
            show_fft=False,
            show_controls=False,
            show_stats=False,
            verbose=False,
            log_scale=self.log_scale,
            auto_contrast=self.auto_contrast,
            vmin=vmin,
            vmax=vmax,
            ncols=max(1, min(ncols, len(frames))),
            size=int(self.size or 0),
            smooth=self.smooth,
            zoom=1.0,
            link_contrast=self.link_contrast,
            display_bin=1,
            show_panel_titles=self.show_panel_titles,
            panel_title_font_size=11,
            # Show2D draws a 1 px grey inner border by default; Show3D never did.
            inter_panel_gap_px=int(self.panel_gap),
            inter_panel_gap_color="",
            gallery_outer_border_px=0,
            gallery_outer_border_color="",
            panel_inner_border_px=0.0,
            panel_inner_border_color="#000000",
            save_state=False,
            comm=DummyComm(),
        )
        if self.roi_active and self.roi_list and len(panels) == 1:
            preview.roi_active = True
            preview.roi_list = [dict(roi) for roi in self.roi_list]
            preview.roi_selected_idx = int(self.roi_selected_idx)
        png = preview._static_png_b64(max_px=max_px, dpi=dpi)
        preview.close()
        return png
