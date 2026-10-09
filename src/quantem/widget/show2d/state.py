"""Show2D view state: the JSON-safe dict behind ``save`` / ``state=``."""

import math

from quantem.widget.show2d.options import (
    _normalize_inset_plot_specs,
    _normalize_marker_mapping,
    _normalize_panel_annotations,
    _normalize_panel_indices,
    _normalize_panel_overlays,
    _normalize_panel_title_style,
    _normalize_scale_bar_style,
)


class Show2DState:
    """Mixin: state dict round trip; the saved-notebook trim is ``SavedStateMixin``."""

    # Traits that carry the bulk pixel payload. Dropped from the saved-notebook
    # snapshot when save_state is False so a plain display stays a few MB, not GB.
    _UNSAVED_HEAVY_KEYS = (
        "frame_bytes",
        "panel_stack_bytes",
        "export_payload",
        "_detail_bytes",
    )
    _STATE_KEYS = (
        "title", "show_title", "cmap", "panel_cmaps", "panel_cmaps_memory",
        "log_scale", "auto_contrast", "contrast_preset", "vmin", "vmax", "vmins", "vmaxs",
        "marker_colors", "marker_style", "row_markers", "col_markers",
        "labels", "panel_title_spans", "panel_annotations", "panel_overlays",
        "inset_plots", "show_inset_plots",
        "starred", "n_pages", "page_idx", "panels_per_page", "page_kind",
        "page_labels", "page_starred", "hidden_panels", "hidden_page_slots", "panel_order",
        "panel_frame_indices", "panel_playback_fps",
        "show_panel_titles", "panel_title_font_size", "panel_title_style",
        "inter_panel_gap_px", "inter_panel_gap_color",
        "gallery_outer_border_px", "gallery_outer_border_color",
        "panel_inner_border_px", "panel_inner_border_color",
        "show_stats", "show_fft", "fft_window", "fft_metrics",
        "show_controls", "controls_collapsed",
        "pixel_size", "pixel_sizes", "pixel_unit",
        "scale_bar_visible", "scale_bar_position", "scale_bar_panels",
        "scale_bar_length", "scale_bar_label", "scale_bar_style", "show_zoom_indicator",
        "size", "smooth", "initial_zoom", "zoom_row", "zoom_col", "view_box",
        "link_zoom", "link_pan", "link_contrast",
        "diff_mode", "diff_reference", "ncols", "selected_idx", "selected_panels",
        "roi_active", "roi_list", "roi_selected_idx", "profile_line",
        "image_rotations", "rotation_scope", "image_flips_horizontal", "image_flips_vertical",
        "denoise", "denoise_sigma", "denoise_bin", "denoise_scope", "show_denoise",
        "denoise_enabled", "denoise_modes", "denoise_sigmas", "denoise_bins",
        "frequency_filter", "frequency_filter_enabled", "frequency_filter_cutoff",
        "frequency_filter_center", "frequency_filter_width", "frequency_filter_modes",
        "frequency_filter_cutoffs", "frequency_filter_centers", "frequency_filter_widths",
        "frequency_filter_scope", "show_frequency_filter",
    )

    def state_dict(self) -> dict:
        """JSON-safe view settings (no pixels): what ``save`` writes and
        ``Show2D(data, state=...)`` restores."""
        state = {key: getattr(self, key) for key in self._STATE_KEYS}
        for key, value in state.items():
            if isinstance(value, (list, dict)):
                state[key] = type(value)(value)
        state["folder_page_size"] = self._folder_page_size
        state["display_bin"] = self._display_bin
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore a ``state_dict``; entries that do not fit this widget's panel
        count or page layout are skipped rather than raising."""
        state = dict(state)
        n_images = int(self.n_images)
        page_kind = str(state.get("page_kind", self.page_kind))
        if page_kind not in {"comparison", "items"}:
            page_kind = str(self.page_kind)
            state.pop("page_kind", None)
        saved_n_pages = max(1, int(state.get("n_pages", self.n_pages)))
        saved_panels_per_page = max(0, int(state.get("panels_per_page", self.panels_per_page)))
        valid_pages = saved_n_pages == 1 and saved_panels_per_page == 0
        if saved_n_pages > 1 and saved_panels_per_page > 0:
            if page_kind == "items":
                valid_pages = saved_n_pages == math.ceil(n_images / saved_panels_per_page)
            else:
                valid_pages = n_images == saved_n_pages * saved_panels_per_page
        if valid_pages:
            # Page index, labels, and hidden-state normalization below all need
            # the saved layout installed before they are validated.
            self.page_kind = page_kind
            self.n_pages = saved_n_pages
            self.panels_per_page = saved_panels_per_page
        else:
            state.pop("n_pages", None)
            state.pop("panels_per_page", None)
            state.pop("page_kind", None)
        folder_page_size = state.pop("folder_page_size", None)
        if page_kind == "items" and self._folder_source is not None and folder_page_size is not None:
            self._folder_page_size = max(1, int(folder_page_size))
        n_pages = int(self.n_pages)
        if "page_idx" in state:
            state["page_idx"] = int(max(0, min(int(state["page_idx"]), n_pages - 1)))
        for key, expected in (
            ("page_starred", (n_pages,)),
            ("page_labels", (n_pages,)),
            ("starred", (n_images,)),
            ("panel_title_spans", (0, n_images)),
            ("panel_cmaps", (0, n_images)),
            ("panel_cmaps_memory", (0, n_images)),
            ("marker_colors", (0, n_images)),
            ("image_flips_horizontal", (0, n_images)),
            ("image_flips_vertical", (0, n_images)),
            ("inset_plots", (0, 1, n_images)),
            ("panel_annotations", (0, n_images)),
            ("panel_overlays", (0, n_images)),
        ):
            if key in state and isinstance(state[key], list) and len(state[key]) not in expected:
                state.pop(key)
        if "inset_plots" in state:
            state["inset_plots"] = _normalize_inset_plot_specs(state["inset_plots"], n_items=n_images)
        if "panel_annotations" in state:
            state["panel_annotations"] = _normalize_panel_annotations(
                state["panel_annotations"], n_items=n_images, labels=list(self.labels),
            )
        if "panel_overlays" in state:
            state["panel_overlays"] = _normalize_panel_overlays(
                state["panel_overlays"], n_items=n_images, labels=list(self.labels),
            )
        if "scale_bar_panels" in state:
            state["scale_bar_panels"] = _normalize_panel_indices(
                state["scale_bar_panels"], n_items=n_images, labels=list(self.labels),
            )
        if "panel_title_style" in state:
            state["panel_title_style"] = _normalize_panel_title_style(state["panel_title_style"])
        if "scale_bar_style" in state:
            state["scale_bar_style"] = _normalize_scale_bar_style(state["scale_bar_style"])
        if state.get("scale_bar_position") not in (None, "bottom-right", "bottom-left"):
            state.pop("scale_bar_position")
        if state.get("marker_style") not in (None, "left", "around"):
            state.pop("marker_style")
        for key in ("row_markers", "col_markers"):
            if key in state:
                state[key] = _normalize_marker_mapping(state[key], name=key)
        if "selected_panels" in state and isinstance(state["selected_panels"], list):
            state["selected_panels"] = [int(value) for value in state["selected_panels"] if 0 <= int(value) < n_images]
        if "hidden_panels" in state and isinstance(state["hidden_panels"], list):
            # A saved state that hid every panel would leave nothing on screen: keep the last one visible.
            hidden = sorted({int(value) for value in state["hidden_panels"] if 0 <= int(value) < n_images})
            if len(hidden) >= n_images:
                hidden = hidden[:-1]
            state["hidden_panels"] = self._normalize_item_page_hidden(hidden, drop_if_full=True)
        if "hidden_page_slots" in state and isinstance(state["hidden_page_slots"], list):
            state["hidden_page_slots"] = self._normalize_hidden_page_slots(state["hidden_page_slots"], drop_if_full=True)
        elif n_pages > 1 and "hidden_panels" in state and isinstance(state["hidden_panels"], list):
            state["hidden_page_slots"] = self._hidden_page_slots_from_panels(state["hidden_panels"], drop_if_full=True)
        if "panel_order" in state and isinstance(state["panel_order"], list):
            order = [int(value) for value in state["panel_order"]]
            if len(order) != n_images or sorted(order) != list(range(n_images)):
                state.pop("panel_order")
        if "panel_frame_indices" in state and isinstance(state["panel_frame_indices"], list):
            counts = list(self.panel_frame_counts)
            indices = [int(value) for value in state["panel_frame_indices"]]
            if len(indices) != n_images or any(
                value < 0 or value >= max(1, int(counts[panel])) for panel, value in enumerate(indices)
            ):
                state.pop("panel_frame_indices")
        for key, value in state.items():
            if key == "display_bin":
                self._display_bin = value
            elif self.has_trait(key):
                setattr(self, key, value)
