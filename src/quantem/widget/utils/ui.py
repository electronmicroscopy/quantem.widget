"""Shared viewer UI chrome presets, constructor keyword checks and display-limit validation."""

import json
import math
import numbers
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import traitlets

DisplayMode = Literal["interactive", "presentation", "report", "minimal"]
ViewerSize = Literal["small", "medium", "large", "xlarge"]
PANEL_WIDTHS: dict[str, int] = json.loads(
    (Path(__file__).parent.parent / "panel_sizes.json").read_text(encoding="utf-8")
)


def resolve_columns(count: int, *, columns: int | None = None, rows: int | None = None, default: int = 2) -> int:
    """Columns of a gallery of ``count`` panels: ``columns`` as given, else enough for ``rows``, else ``default``."""
    for name, value in (("columns", columns), ("rows", rows)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer; use {name}=2, got {value!r}.")
    if rows is not None and columns is not None and rows * columns < count:
        raise ValueError(
            f"rows={rows} and columns={columns} cannot hold {count} panels; "
            "increase the layout or omit rows to calculate it automatically."
        )
    if columns is None:
        columns = max(1, math.ceil(count / rows)) if rows is not None else default
    return int(columns)


def resolve_panel_width(size: ViewerSize, panel_width_px: int | None = None) -> int:
    """Resolve a named image-panel size, with an optional explicit CSS width.

    A positive ``panel_width_px`` overrides the preset. ``None`` or the legacy
    value 0 uses the preset. This changes display geometry, never source pixels.

    Parameters
    ----------
    size : {"small", "medium", "large", "xlarge"}
        Named panel width.
    panel_width_px : int or None, optional
        Explicit CSS width, or 0 to use the named preset.

    Returns
    -------
    int
        Resolved width in CSS pixels.
    """
    if not isinstance(size, str):
        raise TypeError("size must be 'small', 'medium', 'large' or 'xlarge'; use panel_width_px for a pixel width.")
    if size not in PANEL_WIDTHS:
        raise ValueError(f"Unknown size {size!r}; choose 'small', 'medium', 'large' or 'xlarge'.")
    if panel_width_px is None:
        return PANEL_WIDTHS[size]
    if isinstance(panel_width_px, bool) or not isinstance(panel_width_px, numbers.Integral) or panel_width_px < 0:
        raise ValueError("panel_width_px must be an integer >= 0; use size='medium' for the default.")
    return int(panel_width_px) or PANEL_WIDTHS[size]


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


def resolve_mode(
    mode: DisplayMode | str,
    *,
    defaults: Mapping[str, object],
    overrides: Mapping[str, object],
) -> dict[str, object]:
    """Resolve a viewer UI preset with explicit keyword overrides.

    ``mode`` applies broad defaults first. Values in ``overrides`` win when
    they are not ``None`` so callers can use a preset and still opt individual
    controls back in.
    """
    mode = str(mode or "interactive").strip().lower()
    if mode not in UI_MODE_DEFAULTS:
        choices = ", ".join(repr(choice) for choice in UI_MODE_DEFAULTS)
        raise ValueError(f"mode must be one of {choices}, got {mode!r}")

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
