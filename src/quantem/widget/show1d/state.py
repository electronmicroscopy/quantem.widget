"""Show1D view state: the JSON-safe dict behind ``save`` / ``state=``."""


class Show1DState:
    """Mixin: state dict round trip; the saved-notebook trim is ``SavedStateMixin``."""

    # Bulk buffers dropped from the saved-notebook snapshot when save_state is
    # False. y_bytes / x_bytes stay: they are the trace itself (normally KBs)
    # and let a cold reopen paint the plot. snapshot_bytes is the whale (a
    # monitor run's full snapshot stack); profile_image_bytes can be a
    # full-resolution image in from_image mode.
    _UNSAVED_HEAVY_KEYS = (
        "snapshot_bytes",
        "profile_image_bytes",
        "export_payload",
    )
    _STATE_KEYS = (
        "title", "labels", "colors", "x_label", "x_integer", "y_label", "x_unit", "y_unit", "log_scale",
        "show_title", "show_stats", "show_review", "show_legend", "show_grid", "show_controls", "controls_collapsed",
        "line_width", "plot_height_px", "plot_width_px", "max_width", "side_panel_width_px",
        "focused_trace", "x_range", "y_range",
        "selected_snapshot_idx", "selected_snapshot_group_idx", "bookmarked_snapshot_groups",
        "show_snapshots", "show_snapshot_thumbnails", "snapshot_link_views", "show_snapshot_histogram",
        "show_snapshot_fft", "snapshot_fft_window", "snapshot_fft_cmap", "snapshot_fft_layout",
        "show_snapshot_profile", "snapshot_profile_line", "snapshot_profile_height",
        "snapshot_histogram_width", "snapshot_histogram_height",
        "snapshot_contrast_preset", "snapshot_contrast_range", "snapshot_panel_contrast_ranges",
        "snapshot_thumbnail_size", "snapshot_panel_width_px", "snapshot_columns", "snapshot_overlay_position",
        "snapshot_real_space_zoom", "snapshot_real_space_center", "snapshot_fft_zoom", "snapshot_fft_center",
        "image_cmap", "pixel_size", "pixel_unit", "scale_bar_visible",
        "snapshot_playing", "snapshot_fps", "snapshot_loop", "snapshot_bounce",
        "starred_snapshot_image_labels", "hidden_snapshot_image_labels", "trial_notes", "trial_tags",
        "show_trial_notes", "show_starred_only", "review_mode", "trial_sort_key", "trial_sort_descending",
        "trial_filter_text", "top_trial_count",
        "profile_line", "profile_width",
    )

    def state_dict(self) -> dict:
        """JSON-safe view settings (no trace or pixel buffers): what ``save`` writes and ``Show1D(..., state=)`` restores."""
        state = {key: getattr(self, key) for key in self._STATE_KEYS}
        for key, value in state.items():
            if isinstance(value, (list, dict)):
                state[key] = type(value)(value)
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore a ``state_dict``; the trait validators clamp and normalise every value, unknown keys are skipped."""
        for key, value in state.items():
            if key in self._STATE_KEYS:
                setattr(self, key, value)
        self._update_trial_analysis()
