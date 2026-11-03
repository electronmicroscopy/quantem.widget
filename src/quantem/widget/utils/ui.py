"""Shared viewer UI chrome presets, constructor keyword checks and display-limit validation."""

import math
from collections.abc import Mapping
from typing import Literal

import traitlets

UiMode = Literal["interactive", "presentation", "report", "minimal"]

UI_MODE_DEFAULTS: dict[str, dict[str, bool]] = {
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
    defaults: Mapping[str, object],
    overrides: Mapping[str, object],
) -> dict[str, object]:
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


def reject_unknown_kwargs(widget_class: type, kwargs: Mapping[str, object], owned: Mapping[str, str] | None = None) -> None:
    """Raise TypeError naming every keyword that is neither a parameter nor a trait of ``widget_class``.

    Widgets pass leftover keywords to ``anywidget.AnyWidget.__init__`` so any
    synced trait can be set at construction. traitlets only warns about a
    keyword it does not know, so without this check a removed parameter
    (``Show4DSTEM(..., compare_panel_gap_px=0)``) or a typo does nothing.
    ``owned`` maps the traits the constructor sets itself to what sets them (a
    parameter name, ``"data"`` or ``"state"``): a keyword for one of those would
    be overwritten without a word, so it raises and names the way in.
    """
    unknown = sorted(set(kwargs) - set(widget_class.class_trait_names()))
    if len(unknown) == 1:
        raise TypeError(f"{widget_class.__name__}() got an unexpected keyword argument {unknown[0]!r}")
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise TypeError(f"{widget_class.__name__}() got unexpected keyword arguments {names}")
    for name in sorted(set(kwargs) & set(owned or {})):
        owner = owned[name]
        if owner == "data":
            hint = "it follows the data"
        elif owner == "state":
            hint = f"pass state={{{name!r}: ...}} or set widget.{name} after construction"
        else:
            hint = f"pass {owner}= instead"
        raise TypeError(f"{widget_class.__name__}() sets {name} itself; {hint}")


def validate_display_limits(widget, proposal: dict) -> float | None:
    """``vmin`` / ``vmax`` trait validator: a manual limit is finite, with ``vmin <= vmax``."""
    name, value = proposal["trait"].name, proposal["value"]
    if value is not None:
        if not math.isfinite(value):
            raise traitlets.TraitError(f"{name} must be finite, got {value}")
        if name == "vmax" and widget.vmin is not None and value < widget.vmin:
            raise traitlets.TraitError(f"vmax ({value}) must be >= vmin ({widget.vmin})")
        if name == "vmin" and widget.vmax is not None and value > widget.vmax:
            raise traitlets.TraitError(f"vmin ({value}) must be <= vmax ({widget.vmax})")
    return value
