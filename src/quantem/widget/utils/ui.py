"""Shared viewer UI chrome presets."""

from collections.abc import Mapping
from typing import Any, Literal

UiMode = Literal["interactive", "presentation", "report", "minimal"]

UI_MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "interactive": {},
    "presentation": {
        "controls_collapsed": True,
        "show_stats": False,
        "show_resize_handles": False,
        "show_zoom_indicator": False,
    },
    "report": {
        "show_controls": False,
        "controls_collapsed": False,
        "show_stats": False,
        "show_resize_handles": False,
        "show_zoom_indicator": False,
    },
    "minimal": {
        "show_title": False,
        "show_controls": False,
        "controls_collapsed": False,
        "show_stats": False,
        "show_panel_titles": False,
        "show_resize_handles": False,
        "show_zoom_indicator": False,
        "show_scale_bar": False,
        "show_legend": False,
        "show_grid": False,
        "show_crosshair": False,
    },
}


def resolve_ui_mode(
    ui_mode: UiMode | str,
    *,
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a viewer UI preset with explicit keyword overrides.

    ``ui_mode`` applies broad defaults first. Values in ``overrides`` win when
    they are not ``None`` so callers can use a preset and still opt individual
    controls back in.
    """
    mode = str(ui_mode or "interactive").strip().lower()
    if mode not in UI_MODE_DEFAULTS:
        choices = ", ".join(repr(choice) for choice in UI_MODE_DEFAULTS)
        raise ValueError(f"ui_mode must be one of {choices}, got {ui_mode!r}")

    resolved = dict(defaults)
    resolved.update(
        {key: value for key, value in UI_MODE_DEFAULTS[mode].items() if key in defaults}
    )
    resolved.update({key: value for key, value in overrides.items() if value is not None})
    return resolved
