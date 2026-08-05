"""
show2d: 2D comparison viewer with optional FFT and histogram analysis.

For displaying a single image or a gallery of multiple images. Individual list
items may be local frame stacks; unlike Show3D, Show2D does not impose one
shared frame axis across the whole gallery.
"""

import base64
import html
import io as _io
import json
import math
import pathlib
import re
import tempfile
import textwrap
import warnings
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable, Mapping, Self, Sequence

import anywidget
import ipywidgets
import matplotlib
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import traitlets
from quantem.widget._image_folder import (
    ImageFolderRecord,
    WatchedImageFolder,
    WatchedImageFolderMixin,
)
from quantem.widget._folder_watch_status import FOLDER_WATCH_STATE_VALUES
from quantem.widget.utils.array import _b64_safe, _resize_image, to_numpy
from quantem.widget.utils.roi_geometry import roi_geometries
from quantem.widget.utils.state_io import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.utils.static_fallback import StaticFallbackMixin
from quantem.widget.utils.ui import UiMode, resolve_ui_mode

_CORE_IMAGE_DATASET_IMPORT_ATTEMPTED = False
_CORE_IMAGE_DATASET_TYPES: tuple[type[Any], ...] = ()


def _core_image_dataset_types() -> tuple[type[Any], ...]:
    """Return core image dataset classes when quantem.core is importable."""
    global _CORE_IMAGE_DATASET_IMPORT_ATTEMPTED, _CORE_IMAGE_DATASET_TYPES
    if not _CORE_IMAGE_DATASET_IMPORT_ATTEMPTED:
        _CORE_IMAGE_DATASET_IMPORT_ATTEMPTED = True
        try:
            from quantem.core.datastructures import Dataset2d, Dataset3d
        except Exception:
            _CORE_IMAGE_DATASET_TYPES = ()
        else:
            _CORE_IMAGE_DATASET_TYPES = (Dataset2d, Dataset3d)
    return _CORE_IMAGE_DATASET_TYPES


# Sentinel for "caller did not pass this kwarg", so a deprecated alias can
# fill an unset denoise kwarg while an explicitly passed new kwarg wins even
# when its value equals the trait default (e.g. denoise="none").
_UNSET = object()

# Saved-state keys from the display_filter-era API mapped onto the denoise
# family, so old .qwstate files and notebooks keep loading.
_DENOISE_STATE_ALIASES = {
    "display_filter": "denoise",
    "display_sigma": "denoise_sigma",
    "spatial_bin": "denoise_bin",
    "display_filters": "denoise_modes",
    "display_sigmas": "denoise_sigmas",
    "spatial_bins": "denoise_bins",
    "display_filter_banner": "denoise_banner",
}

_DEFAULT_FOLDER_PAGE_SIZE = 20
_PANEL_TITLE_STYLE_KEYS = {
    "bg",
    "fg",
    "border_color",
    "border_width",
    "pad_x",
    "pad_y",
    "max_width",
    "radius",
    "font_weight",
    "font_family",
    "align",
    "opacity",
    "outline_color",
    "outline_width",
    "x",
    "y",
    "anchor",
    "offset",
}

_SCALE_BAR_STYLE_KEYS = {
    "offset",
    "label_gap",
    "font_family",
    "font_size",
    "font_weight",
    "color",
    "outline_color",
    "outline_width",
    "bar_height",
    "bar_width",
    "shadow_color",
}


def _nonnegative_int(value: object, *, name: str) -> int:
    """Return a nonnegative integer for pixel-width options."""
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}") from exc
    if result < 0:
        raise ValueError(f"{name} must be >= 0, got {result}")
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    """Return a nonnegative float for stroke-width options."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative number, got {value!r}") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite value >= 0, got {value!r}")
    return result


def _normalize_panel_title_style(style: Mapping[str, object] | None) -> dict[str, object]:
    """Normalize JSON-safe panel-title chrome options."""
    if style is None:
        return {}
    if not isinstance(style, Mapping):
        raise TypeError(f"panel_title_style must be a mapping, got {type(style).__name__}")
    out: dict[str, object] = {}
    for key, value in style.items():
        key_text = str(key)
        if key_text not in _PANEL_TITLE_STYLE_KEYS:
            raise ValueError(
                "panel_title_style keys must be one of "
                f"{sorted(_PANEL_TITLE_STYLE_KEYS)}, got {key_text!r}"
            )
        if value is None:
            continue
        if key_text in {"border_width", "pad_x", "pad_y", "radius", "opacity", "outline_width", "x", "y"}:
            out[key_text] = float(value)
        elif key_text == "offset":
            vals = np.asarray(value, dtype=np.float64).ravel()
            if vals.size != 2 or not np.isfinite(vals).all():
                raise ValueError("panel_title_style offset must contain two finite values")
            out[key_text] = [float(vals[0]), float(vals[1])]
        elif key_text == "font_weight":
            out[key_text] = int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
        else:
            out[key_text] = str(value)
    return out


def _normalize_scale_bar_style(style: Mapping[str, object] | None) -> dict[str, object]:
    """Normalize JSON-safe scale-bar style options."""
    if style is None:
        return {}
    if not isinstance(style, Mapping):
        raise TypeError(f"scale_bar_style must be a mapping, got {type(style).__name__}")
    out: dict[str, object] = {}
    for key, value in style.items():
        key_text = str(key)
        if key_text not in _SCALE_BAR_STYLE_KEYS:
            raise ValueError(
                "scale_bar_style keys must be one of "
                f"{sorted(_SCALE_BAR_STYLE_KEYS)}, got {key_text!r}"
            )
        if value is None:
            continue
        if key_text == "offset":
            vals = np.asarray(value, dtype=np.float64).ravel()
            if vals.size != 2 or not np.isfinite(vals).all():
                raise ValueError("scale_bar_style offset must contain two finite values")
            out[key_text] = [float(vals[0]), float(vals[1])]
        elif key_text in {"label_gap", "font_size", "outline_width", "bar_height", "bar_width"}:
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"scale_bar_style {key_text} must be finite, got {value!r}")
            if key_text in {"font_size", "bar_height", "bar_width"} and number <= 0:
                raise ValueError(f"scale_bar_style {key_text} must be > 0, got {value!r}")
            if key_text == "outline_width" and number < 0:
                raise ValueError(f"scale_bar_style outline_width must be >= 0, got {value!r}")
            out[key_text] = number
        elif key_text == "font_weight":
            out[key_text] = int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
        else:
            out[key_text] = str(value)
    return out


def _normalize_marker_mapping(markers: Mapping[object, object] | None, *, name: str) -> dict[str, str]:
    """Normalize row/column marker dictionaries to JSON-safe string keys."""
    if markers is None:
        return {}
    if not isinstance(markers, Mapping):
        raise TypeError(f"{name} must be a mapping from nonnegative index to color")
    out: dict[str, str] = {}
    for key, value in markers.items():
        if value is None or value == "":
            continue
        if isinstance(key, bool):
            raise ValueError(f"{name} index must be a nonnegative integer, got {key!r}")
        try:
            idx = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} index must be a nonnegative integer, got {key!r}") from exc
        if idx < 0:
            raise ValueError(f"{name} index must be >= 0, got {idx}")
        out[str(idx)] = str(value)
    return out


def _normalise_title_spans(value: object) -> tuple[str, list[dict[str, str]] | None]:
    """Return plain fallback text plus optional safe colored text spans.

    Panel-title spans are structured dictionaries, not HTML. The plain text
    fallback keeps existing string-based state, panel lookup, exports, and old
    notebooks unchanged while the synced span payload lets the frontend color
    status words such as ``low`` / ``cal`` / ``over``.
    """
    if value is None:
        return "", None
    if isinstance(value, str):
        return value, None
    if isinstance(value, Mapping):
        if value.get("math") not in (None, ""):
            text = str(value.get("math"))
            span: dict[str, str] = {"math": text}
        else:
            text = "" if value.get("text") is None else str(value.get("text"))
            span = {"text": text}
        color = value.get("color")
        if color not in (None, ""):
            span["color"] = str(color)
        return text, [span]
    if isinstance(value, (list, tuple)):
        spans: list[dict[str, str]] = []
        plain_parts: list[str] = []
        for idx, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise TypeError(
                    "rich panel title spans must be dictionaries like "
                    f"{{'text': 'low', 'color': '#60a5fa'}}, got {type(item).__name__} "
                    f"at span {idx}"
                )
            if item.get("math") not in (None, ""):
                text = str(item.get("math"))
                span = {"math": text}
            else:
                text = "" if item.get("text") is None else str(item.get("text"))
                span = {"text": text}
            color = item.get("color")
            if color not in (None, ""):
                span["color"] = str(color)
            spans.append(span)
            plain_parts.append(text)
        return "".join(plain_parts), spans
    return str(value), None


def _normalise_title_span_sequence(
    values: Sequence[object] | Mapping[str, object] | None,
) -> tuple[list[str] | None, list[list[dict[str, str]]]]:
    """Normalize a per-panel title sequence into plain labels plus spans."""
    if values is None:
        return None, []
    if isinstance(values, Mapping):
        values = [values]
    elif (
        isinstance(values, (list, tuple))
        and values
        and all(isinstance(item, Mapping) for item in values)
    ):
        values = [values]
    plain: list[str] = []
    rich: list[list[dict[str, str]]] = []
    has_rich = False
    for value in values:
        text, spans = _normalise_title_spans(value)
        plain.append(text)
        rich.append(spans or [])
        has_rich = has_rich or bool(spans)
    return plain, rich if has_rich else []


def _title_span_sequence_length(values: Sequence[object] | Mapping[str, object] | None) -> int:
    """Return the number of panel titles represented by a rich-title input."""
    if values is None:
        return 0
    if isinstance(values, Mapping):
        return 1
    if (
        isinstance(values, (list, tuple))
        and values
        and all(isinstance(item, Mapping) for item in values)
    ):
        return 1
    return len(values)


def _expand_title_spans_for_flattened_labels(
    spans: list[list[dict[str, str]]],
    *,
    original_len: int,
    n_panels: int,
    n_pages: int,
    panels_per_page: int,
) -> list[list[dict[str, str]]]:
    """Broadcast rich title spans across paged panel layouts."""
    if not spans:
        return []
    if len(spans) == n_panels:
        return spans
    if n_pages > 1 and panels_per_page > 0:
        if original_len == panels_per_page and len(spans) == panels_per_page:
            return [list(span) for _ in range(n_pages) for span in spans]
        if original_len == n_panels and len(spans) == n_panels:
            return spans
    return spans


def _rotation_to_quarter_turns(value: int | float) -> int:
    """Normalize a display rotation in degrees or quarter-turn units.

    Public APIs accept scientist-readable degrees (0, 90, 180, 270). For
    backwards compatibility with existing internal state, small integer values
    0..3 are also accepted as quarter turns.
    """
    angle = float(value)
    if angle in (0, 1, 2, 3):
        return int(angle) % 4
    if not np.isfinite(angle) or abs(angle / 90 - round(angle / 90)) > 1e-9:
        raise ValueError(
            f"rotation must be 0, 90, 180, 270, or a quarter-turn 0..3; got {value!r}"
        )
    return int(round(angle / 90)) % 4


def _normalize_rotation_list(
    *,
    n_items: int,
    rotation: int | float = 0,
    rotations: Sequence[int | float] | None = None,
) -> list[int]:
    """Return one normalized quarter-turn per item."""
    if rotations is None:
        return [_rotation_to_quarter_turns(rotation)] * int(n_items)
    values = [_rotation_to_quarter_turns(value) for value in rotations]
    if len(values) != int(n_items):
        raise ValueError(
            f"rotations length ({len(values)}) must match the number of items ({int(n_items)})"
        )
    return values


def _normalize_inset_plot_specs(
    inset_plots: Sequence[dict[str, Any] | None] | dict[str, Any] | None,
    *,
    n_items: int,
) -> list[dict[str, Any]]:
    """Return JSON-safe per-panel inset plot specifications.

    The public API intentionally mirrors the smallest useful slice of a
    matplotlib line plot: ``x``, ``y``, optional ``point``, optional
    ``xlim``/``ylim``, and simple style/placement keys.  Arrays are converted
    to plain lists so the same trait survives notebook state and HTML export.
    """
    if inset_plots is None:
        return []
    if isinstance(inset_plots, dict):
        raw_specs: list[dict[str, Any] | None] = [inset_plots]
    else:
        raw_specs = list(inset_plots)
    if len(raw_specs) == 1 and n_items > 1:
        raw_specs = raw_specs * n_items
    if len(raw_specs) != int(n_items):
        raise ValueError(
            f"inset_plots length ({len(raw_specs)}) must be 1 or match the "
            f"number of Show2D panels ({int(n_items)})"
        )

    normalized: list[dict[str, Any]] = []
    for panel, spec in enumerate(raw_specs):
        if spec is None:
            normalized.append({})
            continue
        if not isinstance(spec, dict):
            raise TypeError(f"inset_plots[{panel}] must be a dict or None")
        x_raw = spec.get("x")
        y_raw = spec.get("y")
        if y_raw is None and "points" in spec:
            points = np.asarray(spec["points"], dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError(
                    f"inset_plots[{panel}]['points'] must have shape (N, 2)"
                )
            x = points[:, 0]
            y = points[:, 1]
        else:
            if y_raw is None:
                raise ValueError(f"inset_plots[{panel}] must include 'y' or 'points'")
            y = np.asarray(y_raw, dtype=np.float64).ravel()
            x = (
                np.arange(y.size, dtype=np.float64)
                if x_raw is None
                else np.asarray(x_raw, dtype=np.float64).ravel()
            )
        if x.size != y.size or x.size < 2:
            raise ValueError(
                f"inset_plots[{panel}] x/y must have the same length >= 2; "
                f"got {x.size} and {y.size}"
            )
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"inset_plots[{panel}] contains NaN or inf")
        out: dict[str, Any] = {
            "x": [float(v) for v in x],
            "y": [float(v) for v in y],
        }
        for key in (
            "title",
            "legend",
            "legend_position",
            "annotation",
            "annotation_position",
            "xlabel",
            "ylabel",
            "color",
            "point_color",
            "border_color",
            "text_color",
            "tick_color",
            "position",
            "background",
        ):
            if key in spec and spec[key] is not None:
                out[key] = str(spec[key])
        for key in ("size", "height", "line_width", "border_width", "background_alpha", "tick_font_size", "label_font_size", "legend_font_size"):
            if key in spec and spec[key] is not None:
                out[key] = float(spec[key])
        for key in ("show_ticks", "show_panel_index"):
            if key in spec and spec[key] is not None:
                out[key] = bool(spec[key])
        for key in ("xlim", "ylim", "point"):
            if key in spec and spec[key] is not None:
                vals = np.asarray(spec[key], dtype=np.float64).ravel()
                if vals.size != 2 or not np.isfinite(vals).all():
                    raise ValueError(f"inset_plots[{panel}]['{key}'] must contain two finite values")
                out[key] = [float(vals[0]), float(vals[1])]
        for key in ("box",):
            if key in spec and spec[key] is not None:
                vals = np.asarray(spec[key], dtype=np.float64).ravel()
                if vals.size != 4 or not np.isfinite(vals).all():
                    raise ValueError(f"inset_plots[{panel}]['{key}'] must contain four finite values")
                left, top, width, height = (float(v) for v in vals)
                out[key] = [
                    max(0.0, min(1.0, left)),
                    max(0.0, min(1.0, top)),
                    max(0.05, min(1.0, width)),
                    max(0.05, min(1.0, height)),
                ]
        for key in ("xticks", "yticks"):
            if key in spec and spec[key] is not None:
                vals = np.asarray(spec[key], dtype=np.float64).ravel()
                if vals.size < 1 or not np.isfinite(vals).all():
                    raise ValueError(f"inset_plots[{panel}]['{key}'] must contain finite values")
                out[key] = [float(v) for v in vals]
        if "margin" in spec and spec["margin"] is not None:
            vals = np.asarray(spec["margin"], dtype=np.float64).ravel()
            if vals.size == 1:
                vals = np.repeat(vals, 2)
            if vals.size != 2 or not np.isfinite(vals).all():
                raise ValueError(
                    f"inset_plots[{panel}]['margin'] must be one number or two finite values"
                )
            out["margin"] = [max(0.0, float(vals[0])), max(0.0, float(vals[1]))]
        normalized.append(out)
    return normalized


_ANNOTATION_STYLE_KEYS = {
    "text",
    "math",
    "label",
    "title",
    "spans",
    "panel",
    "position",
    "anchor",
    "x",
    "y",
    "box",
    "region",
    "variant",
    "class_name",
    "class",
    "bg",
    "fg",
    "color",
    "border_color",
    "border_width",
    "font_size",
    "font_weight",
    "font_family",
    "pad_x",
    "pad_y",
    "radius",
    "opacity",
    "align",
    "max_width",
    "offset",
    "outline_color",
    "outline_width",
}
_ANNOTATION_POSITIONS = {
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
}
_ANNOTATION_VARIANTS = {"badge", "pill", "plain", "outline", "callout"}
_ANNOTATION_ANCHORS = {
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
}


def _is_annotation_spec(value: object) -> bool:
    """Return True when a mapping looks like one annotation spec."""
    return isinstance(value, Mapping) and any(str(key) in _ANNOTATION_STYLE_KEYS for key in value)


def _panel_annotation_index(panel: object, *, labels: Sequence[str] | None, n_items: int) -> int:
    """Resolve a panel annotation target from integer index or panel label."""
    if isinstance(panel, bool):
        raise ValueError(f"panel annotation panel must be an index or label, got {panel!r}")
    if isinstance(panel, str) and labels is not None and panel in labels:
        return list(labels).index(panel)
    try:
        idx = int(panel)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"panel annotation panel must be an index or label, got {panel!r}") from exc
    if idx < 0 or idx >= int(n_items):
        raise ValueError(f"panel annotation panel index {idx} is outside 0..{int(n_items) - 1}")
    return idx


def _normalize_panel_indices(
    panels: Sequence[object] | object | None,
    *,
    labels: Sequence[str] | None,
    n_items: int,
    name: str,
) -> list[int]:
    """Resolve optional panel index/label selectors to unique panel indices."""
    if panels is None:
        return []
    if isinstance(panels, (str, bytes)) or not isinstance(panels, Sequence):
        raw_values = [panels]
    else:
        raw_values = list(panels)
    out: list[int] = []
    seen: set[int] = set()
    for raw in raw_values:
        idx = _panel_annotation_index(raw, labels=labels, n_items=n_items)
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _normalize_panel_annotation_spec(spec: object, *, panel: int) -> dict[str, Any] | None:
    """Normalize one panel annotation into JSON-safe display state."""
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {"text": spec}
    if not isinstance(spec, Mapping):
        raise TypeError(
            f"panel_annotations[{panel}] entries must be strings or mappings, got {type(spec).__name__}"
        )
    unknown = sorted(str(key) for key in spec if str(key) not in _ANNOTATION_STYLE_KEYS)
    if unknown:
        raise ValueError(
            "panel_annotations entries only accept keys "
            f"{sorted(_ANNOTATION_STYLE_KEYS)}, got {unknown[0]!r}"
        )
    raw_text = spec.get("text", spec.get("label", spec.get("title", "")))
    raw_math = spec.get("math")
    spans_raw = spec.get("spans")
    if spans_raw is not None:
        text, spans = _normalise_title_spans(spans_raw)
    elif raw_math not in (None, ""):
        text = str(raw_math)
        spans = [{"math": text}]
    else:
        text, spans = _normalise_title_spans(raw_text)
    out: dict[str, Any] = {"text": text}
    if raw_math not in (None, ""):
        out["math"] = str(raw_math)
    if spans:
        out["spans"] = spans
    for key in (
        "position",
        "anchor",
        "variant",
        "class_name",
        "bg",
        "fg",
        "color",
        "border_color",
        "font_weight",
        "font_family",
        "align",
        "max_width",
        "outline_color",
    ):
        source_key = "class" if key == "class_name" and "class_name" not in spec else key
        if source_key in spec and spec[source_key] not in (None, ""):
            out[key] = str(spec[source_key])
    position = str(out.get("position", "top-left"))
    if position not in _ANNOTATION_POSITIONS:
        raise ValueError(f"panel annotation position must be one of {sorted(_ANNOTATION_POSITIONS)}, got {position!r}")
    out["position"] = position
    if "anchor" in out and out["anchor"] not in _ANNOTATION_ANCHORS:
        raise ValueError(f"panel annotation anchor must be one of {sorted(_ANNOTATION_ANCHORS)}, got {out['anchor']!r}")
    if "variant" in out and out["variant"] not in _ANNOTATION_VARIANTS:
        raise ValueError(f"panel annotation variant must be one of {sorted(_ANNOTATION_VARIANTS)}, got {out['variant']!r}")
    else:
        out.setdefault("variant", "badge")
    for key in ("x", "y", "border_width", "font_size", "pad_x", "pad_y", "radius", "opacity", "outline_width"):
        if key in spec and spec[key] is not None:
            value = float(spec[key])
            if not np.isfinite(value):
                raise ValueError(f"panel annotation {key} must be finite, got {value!r}")
            if key in {"x", "y", "opacity"}:
                value = max(0.0, min(1.0, value))
            out[key] = value
    box_raw = spec.get("box", spec.get("region"))
    if box_raw is not None:
        vals = np.asarray(box_raw, dtype=np.float64).ravel()
        if vals.size != 4 or not np.isfinite(vals).all():
            raise ValueError("panel annotation box/region must contain four finite values")
        left, top, width, height = (float(v) for v in vals)
        out["box"] = [
            max(0.0, min(1.0, left)),
            max(0.0, min(1.0, top)),
            max(0.01, min(1.0, width)),
            max(0.01, min(1.0, height)),
        ]
    if "offset" in spec and spec["offset"] is not None:
        vals = np.asarray(spec["offset"], dtype=np.float64).ravel()
        if vals.size != 2 or not np.isfinite(vals).all():
            raise ValueError("panel annotation offset must contain two finite values")
        out["offset"] = [float(vals[0]), float(vals[1])]
    return out


def _normalize_panel_annotations(
    panel_annotations: Sequence[object] | Mapping[object, object] | object | None,
    *,
    n_items: int,
    labels: Sequence[str] | None = None,
) -> list[list[dict[str, Any]]]:
    """Normalize per-panel annotation labels.

    Accepted forms are:
    - one annotation mapping/string, broadcast to all panels;
    - a per-panel sequence whose entries are an annotation, list of annotations,
      or ``None``;
    - a flat sequence of mappings that include ``panel=...``;
    - a mapping from panel index/label to one annotation or a list of them.
    """
    if panel_annotations is None:
        return []
    grouped: list[list[dict[str, Any]]] = [[] for _ in range(int(n_items))]

    def add(panel: int, value: object) -> None:
        values = value if isinstance(value, (list, tuple)) and not _is_annotation_spec(value) else [value]
        for item in values:
            normalized = _normalize_panel_annotation_spec(item, panel=panel)
            if normalized is not None and normalized.get("text", ""):
                grouped[panel].append(normalized)

    if isinstance(panel_annotations, Mapping) and not _is_annotation_spec(panel_annotations):
        for raw_panel, value in panel_annotations.items():
            add(_panel_annotation_index(raw_panel, labels=labels, n_items=n_items), value)
        return grouped
    if isinstance(panel_annotations, (str, Mapping)):
        for panel in range(int(n_items)):
            add(panel, panel_annotations)
        return grouped

    raw = list(panel_annotations)  # type: ignore[arg-type]
    if not raw:
        return []
    flat_with_panel = any(isinstance(item, Mapping) and "panel" in item for item in raw)
    if int(n_items) == 1 and not flat_with_panel:
        for item in raw:
            add(0, item)
        return grouped
    per_panel = len(raw) == int(n_items) and not flat_with_panel
    if per_panel:
        for panel, value in enumerate(raw):
            add(panel, value)
        return grouped
    for item in raw:
        if not isinstance(item, Mapping) or "panel" not in item:
            raise ValueError(
                "panel_annotations as a flat list must include panel=... on every entry, "
                "or pass a per-panel list/dict"
            )
        add(_panel_annotation_index(item["panel"], labels=labels, n_items=n_items), item)
    return grouped


_OVERLAY_SHAPES = {"circle", "rect", "rectangle", "square"}
_OVERLAY_COORDS = {"data", "relative"}
_OVERLAY_STYLE_KEYS = {
    "shape",
    "type",
    "kind",
    "coords",
    "coordinate_system",
    "panel",
    "center",
    "radius",
    "r",
    "size",
    "row",
    "col",
    "x",
    "y",
    "row0",
    "col0",
    "row1",
    "col1",
    "xyxy",
    "xywh",
    "box",
    "region",
    "stroke",
    "stroke_color",
    "border_color",
    "color",
    "stroke_width",
    "border_width",
    "line_width",
    "line_style",
    "stroke_style",
    "dash",
    "line_dash",
    "fill",
    "fill_color",
    "opacity",
    "alpha",
    "fill_opacity",
    "stroke_opacity",
    "z_order",
    "order",
    "class_name",
}


def _is_overlay_spec(value: object) -> bool:
    """Return True when a mapping looks like one geometric overlay spec."""
    return isinstance(value, Mapping) and any(str(key) in _OVERLAY_STYLE_KEYS for key in value)


def _panel_overlay_index(panel: object, *, labels: Sequence[str] | None, n_items: int) -> int:
    """Resolve an overlay target from integer index or panel label."""
    if isinstance(panel, bool):
        raise ValueError(f"panel overlay panel must be an index or label, got {panel!r}")
    if isinstance(panel, str) and labels is not None and panel in labels:
        return list(labels).index(panel)
    try:
        idx = int(panel)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"panel overlay panel must be an index or label, got {panel!r}") from exc
    if idx < 0 or idx >= int(n_items):
        raise ValueError(f"panel overlay panel index {idx} is outside 0..{int(n_items) - 1}")
    return idx


def _finite_float(value: object, *, name: str) -> float:
    """Return one finite float with a useful user-facing error."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"panel overlay {name} must be finite, got {value!r}")
    return out


def _finite_float_array(value: object, *, name: str, count: int) -> list[float]:
    """Return a fixed-length finite float list."""
    vals = np.asarray(value, dtype=np.float64).ravel()
    if vals.size != count or not np.isfinite(vals).all():
        raise ValueError(f"panel overlay {name} must contain {count} finite values")
    return [float(v) for v in vals]


def _finite_float_sequence(value: object, *, name: str) -> list[float]:
    """Return a non-empty finite float list for custom dash patterns."""
    vals = np.asarray(value, dtype=np.float64).ravel()
    if vals.size == 0 or not np.isfinite(vals).all():
        raise ValueError(f"panel overlay {name} must contain finite values")
    out = [float(v) for v in vals]
    if any(v < 0 for v in out):
        raise ValueError(f"panel overlay {name} values must be >= 0")
    if all(v == 0 for v in out):
        raise ValueError(f"panel overlay {name} must contain at least one positive value")
    return out


def _normalize_panel_overlay_spec(spec: object, *, panel: int) -> dict[str, Any] | None:
    """Normalize one circle/rect overlay into JSON-safe display state."""
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise TypeError(f"panel_overlays[{panel}] entries must be mappings, got {type(spec).__name__}")
    unknown = sorted(str(key) for key in spec if str(key) not in _OVERLAY_STYLE_KEYS)
    if unknown:
        raise ValueError(
            "panel_overlays entries only accept keys "
            f"{sorted(_OVERLAY_STYLE_KEYS)}, got {unknown[0]!r}"
        )
    shape = str(spec.get("shape", spec.get("type", spec.get("kind", "circle")))).lower()
    if shape not in _OVERLAY_SHAPES:
        raise ValueError(f"panel overlay shape must be one of {sorted(_OVERLAY_SHAPES)}, got {shape!r}")
    if shape == "rectangle":
        shape = "rect"
    coords = str(spec.get("coords", spec.get("coordinate_system", "data"))).lower()
    if coords not in _OVERLAY_COORDS:
        raise ValueError(f"panel overlay coords must be one of {sorted(_OVERLAY_COORDS)}, got {coords!r}")

    out: dict[str, Any] = {"shape": shape, "coords": coords}
    if shape == "circle":
        if "center" in spec and spec["center"] is not None:
            row, col = _finite_float_array(spec["center"], name="center", count=2)
        elif all(key in spec for key in ("row", "col")):
            row = _finite_float(spec["row"], name="row")
            col = _finite_float(spec["col"], name="col")
        elif all(key in spec for key in ("y", "x")):
            row = _finite_float(spec["y"], name="y")
            col = _finite_float(spec["x"], name="x")
        else:
            raise ValueError("circle overlays require center=(row, col) or row=... and col=...")
        radius = _finite_float(spec.get("radius", spec.get("r")), name="radius")
        if radius <= 0:
            raise ValueError(f"circle overlay radius must be > 0, got {radius}")
        out.update({"row": row, "col": col, "radius": radius})
    else:
        if "box" in spec or "region" in spec:
            row0, col0, row1, col1 = _finite_float_array(spec.get("box", spec.get("region")), name="box", count=4)
        elif "xyxy" in spec:
            col0, row0, col1, row1 = _finite_float_array(spec["xyxy"], name="xyxy", count=4)
        elif "xywh" in spec:
            col0, row0, width, height = _finite_float_array(spec["xywh"], name="xywh", count=4)
            row1 = row0 + height
            col1 = col0 + width
        elif all(key in spec for key in ("row0", "col0", "row1", "col1")):
            row0 = _finite_float(spec["row0"], name="row0")
            col0 = _finite_float(spec["col0"], name="col0")
            row1 = _finite_float(spec["row1"], name="row1")
            col1 = _finite_float(spec["col1"], name="col1")
        elif shape == "square" and "center" in spec and "size" in spec:
            row, col = _finite_float_array(spec["center"], name="center", count=2)
            half = _finite_float(spec["size"], name="size") / 2.0
            row0, col0, row1, col1 = row - half, col - half, row + half, col + half
        else:
            raise ValueError("rect/square overlays require box=(row0, col0, row1, col1), xyxy=..., or xywh=...")
        if row1 < row0:
            row0, row1 = row1, row0
        if col1 < col0:
            col0, col1 = col1, col0
        if row1 == row0 or col1 == col0:
            raise ValueError("rect/square overlays must have non-zero width and height")
        out.update({"row0": row0, "col0": col0, "row1": row1, "col1": col1})

    stroke = spec.get("stroke", spec.get("stroke_color", spec.get("border_color", spec.get("color", "#00e5ff"))))
    fill = spec.get("fill", spec.get("fill_color", None))
    out["stroke"] = str(stroke)
    has_fill = fill not in (None, "", "none", "None")
    if has_fill:
        out["fill"] = str(fill)
    out["stroke_width"] = _finite_float(
        spec.get("stroke_width", spec.get("border_width", spec.get("line_width", 2.0))),
        name="stroke_width",
    )
    if out["stroke_width"] < 0:
        raise ValueError(f"panel overlay stroke_width must be >= 0, got {out['stroke_width']}")
    line_style = str(spec.get("line_style", spec.get("stroke_style", "solid"))).lower().replace("_", "-")
    line_style_aliases = {
        "solid": "solid",
        "none": "solid",
        "dash": "dashed",
        "dashed": "dashed",
        "dot": "dotted",
        "dotted": "dotted",
        "dash-dot": "dashdot",
        "dashdot": "dashdot",
        "dash-dot-dot": "dashdot",
    }
    if line_style not in line_style_aliases:
        raise ValueError(
            "panel overlay line_style must be one of "
            "['solid', 'dashed', 'dotted', 'dashdot'] or use dash=[...]"
        )
    out["line_style"] = line_style_aliases[line_style]
    if "dash" in spec or "line_dash" in spec:
        out["dash"] = _finite_float_sequence(spec.get("dash", spec.get("line_dash")), name="dash")
    opacity = max(0.0, min(1.0, _finite_float(spec.get("opacity", spec.get("alpha", 1.0)), name="opacity")))
    out["opacity"] = opacity
    default_fill_opacity = 1.0 if has_fill else 0.0
    out["fill_opacity"] = max(
        0.0,
        min(1.0, _finite_float(spec.get("fill_opacity", default_fill_opacity), name="fill_opacity")),
    )
    out["stroke_opacity"] = max(0.0, min(1.0, _finite_float(spec.get("stroke_opacity", 1.0), name="stroke_opacity")))
    out["z_order"] = _finite_float(spec.get("z_order", spec.get("order", 0.0)), name="z_order")
    if "class_name" in spec and spec["class_name"] not in (None, ""):
        out["class_name"] = str(spec["class_name"])
    return out


def _normalize_panel_overlays(
    panel_overlays: Sequence[object] | Mapping[object, object] | object | None,
    *,
    n_items: int,
    labels: Sequence[str] | None = None,
) -> list[list[dict[str, Any]]]:
    """Normalize per-panel circle/rect overlays.

    Accepted forms mirror ``panel_annotations``:
    one overlay mapping broadcasts to all panels; a mapping keyed by panel
    index/label targets specific panels; a per-panel list aligns with panels;
    and a flat list of mappings with ``panel=...`` can target arbitrary panels.
    """
    if panel_overlays is None:
        return []
    grouped: list[list[dict[str, Any]]] = [[] for _ in range(int(n_items))]

    def add(panel: int, value: object) -> None:
        values = value if isinstance(value, (list, tuple)) and not _is_overlay_spec(value) else [value]
        for item in values:
            normalized = _normalize_panel_overlay_spec(item, panel=panel)
            if normalized is not None:
                grouped[panel].append(normalized)

    if isinstance(panel_overlays, Mapping) and not _is_overlay_spec(panel_overlays):
        for raw_panel, value in panel_overlays.items():
            add(_panel_overlay_index(raw_panel, labels=labels, n_items=n_items), value)
        return grouped
    if isinstance(panel_overlays, Mapping):
        if "panel" in panel_overlays:
            add(_panel_overlay_index(panel_overlays["panel"], labels=labels, n_items=n_items), panel_overlays)
        else:
            for panel in range(int(n_items)):
                add(panel, panel_overlays)
        return grouped

    raw = list(panel_overlays)  # type: ignore[arg-type]
    if not raw:
        return []
    flat_with_panel = any(isinstance(item, Mapping) and "panel" in item for item in raw)
    flat_overlay_specs = all(_is_overlay_spec(item) for item in raw)
    if flat_overlay_specs and not flat_with_panel:
        for panel in range(int(n_items)):
            for item in raw:
                add(panel, item)
        return grouped
    if int(n_items) == 1 and not flat_with_panel:
        for item in raw:
            add(0, item)
        return grouped
    per_panel = len(raw) == int(n_items) and not flat_with_panel
    if per_panel:
        for panel, value in enumerate(raw):
            add(panel, value)
        return grouped
    for item in raw:
        if not isinstance(item, Mapping) or "panel" not in item:
            raise ValueError(
                "panel_overlays as a flat list must include panel=... on every entry, "
                "or pass a per-panel list/dict"
            )
        add(_panel_overlay_index(item["panel"], labels=labels, n_items=n_items), item)
    return grouped


def _reject_unknown_kwargs(cls, kwargs: dict) -> None:
    """Raise TypeError if kwargs contains any key that isn't a declared trait.

    anywidget/traitlets silently accept unknown keys, which let stale notebooks
    pass obsolete params like ``pixel_size_angstrom=0.5`` with no warning.  This
    helper catches typos and renamed-trait references at construction time.
    """
    traits = set(cls.class_trait_names())
    unknown = [k for k in kwargs if k not in traits]
    if unknown:
        key = sorted(unknown)[0]
        raise TypeError(
            f"{cls.__name__}() got unexpected keyword argument {key!r}. "
            f"Check for typos or a renamed parameter (e.g. canvas_size → size, "
            f"image_width_px → size, pixel_size_angstrom → pixel_size)."
        )


def _validate_folder_page_size(page_size: int | None) -> int | None:
    """Return a normalized Show2D folder page size."""
    if page_size is None:
        return None
    if isinstance(page_size, (bool, np.bool_)) or not isinstance(
        page_size,
        (int, np.integer),
    ):
        raise TypeError(
            "page_size must be a positive integer or None; "
            "use None to display the folder as one unpaged gallery"
        )
    value = int(page_size)
    if value < 1:
        raise ValueError(
            f"page_size must be >= 1 or None, got {page_size!r}; "
            "use None to disable folder paging"
        )
    return value


def _is_show2d_page_dict(value: object) -> bool:
    """Return True for a dict-like Show2D page specification."""
    return isinstance(value, dict) and any(key in value for key in ("images", "data", "array"))


def _normalise_show2d_pages(
    data,
    *,
    labels: list[str | None] | None,
    page_labels: Sequence[str | None] | None,
) -> tuple[object, list[str | None] | None, int, int, list[str], list[int]]:
    """Flatten Show2D paged input into the established gallery stack.

    Public page data is ``pages x panels x rows x cols``. Internally this stays
    compatible with Show2D's existing gallery transport by flattening pages into
    one stack and syncing page metadata separately. The browser then renders
    only the active page.
    """
    explicit_page_dicts = (
        isinstance(data, (list, tuple))
        and len(data) > 0
        and all(_is_show2d_page_dict(item) for item in data)
    )
    if explicit_page_dicts:
        page_arrays: list[np.ndarray] = []
        flattened_labels: list[str | None] = []
        inferred_page_labels: list[str] = []
        panels_per_page: int | None = None
        for page_idx, page in enumerate(data):
            raw_images = page.get("images", page.get("data", page.get("array")))
            if raw_images is None:
                raise ValueError(
                    f"Show2D page {page_idx} is missing images/data/array. "
                    "Use {'title': '...', 'images': [...]}"
                )
            arr = to_numpy(raw_images)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            if arr.ndim != 3:
                raise ValueError(
                    f"Show2D page {page_idx} must contain a 2D image or 3D panel stack, "
                    f"got shape {arr.shape}"
                )
            if panels_per_page is None:
                panels_per_page = int(arr.shape[0])
            elif int(arr.shape[0]) != panels_per_page:
                raise ValueError(
                    "Every Show2D page must contain the same number of panels; "
                    f"page 0 has {panels_per_page}, page {page_idx} has {arr.shape[0]}"
                )
            page_arrays.append(arr)
            inferred_page_labels.append(str(page.get("title") or page.get("label") or f"Page {page_idx + 1}"))
            page_panel_labels = page.get("labels")
            if page_panel_labels is not None:
                if len(page_panel_labels) != int(arr.shape[0]):
                    raise ValueError(
                        f"Show2D page {page_idx} labels length ({len(page_panel_labels)}) "
                        f"must match its panel count ({arr.shape[0]})"
                    )
                flattened_labels.extend([None if label is None else str(label) for label in page_panel_labels])
            elif labels is not None:
                if len(labels) == int(arr.shape[0]):
                    flattened_labels.extend([None if label is None else str(label) for label in labels])
                elif panels_per_page is not None and len(labels) == len(data) * panels_per_page:
                    start = page_idx * panels_per_page
                    stop = start + panels_per_page
                    flattened_labels.extend([None if label is None else str(label) for label in labels[start:stop]])
                else:
                    raise ValueError(
                        "labels for paged Show2D must have length panels_per_page "
                        f"({arr.shape[0]}) or n_pages * panels_per_page "
                        f"({len(data) * arr.shape[0]}), got {len(labels)}"
                    )
            else:
                flattened_labels.extend([f"Panel {i + 1}" for i in range(int(arr.shape[0]))])
        stack = np.stack(page_arrays, axis=0)
        n_pages = int(stack.shape[0])
        panels = int(stack.shape[1])
        resolved_page_labels = [
            "" if label is None else str(label)
            for label in (page_labels if page_labels is not None else inferred_page_labels)
        ]
        if len(resolved_page_labels) != n_pages:
            raise ValueError(f"page_labels length ({len(resolved_page_labels)}) must match n_pages ({n_pages})")
        return (
            stack.reshape(n_pages * panels, *stack.shape[-2:]),
            flattened_labels,
            n_pages,
            panels,
            resolved_page_labels,
            [0] * n_pages,
        )

    arr = None
    if not isinstance(data, list):
        try:
            arr = to_numpy(data)
        except Exception:
            arr = None
    if arr is not None and arr.ndim == 4:
        n_pages, panels = int(arr.shape[0]), int(arr.shape[1])
        resolved_page_labels = [
            "" if label is None else str(label)
            for label in (page_labels if page_labels is not None else [f"Page {i + 1}" for i in range(n_pages)])
        ]
        if len(resolved_page_labels) != n_pages:
            raise ValueError(f"page_labels length ({len(resolved_page_labels)}) must match n_pages ({n_pages})")
        if labels is not None:
            if len(labels) == panels:
                resolved_labels = [
                    "" if label is None else str(label)
                    for _ in range(n_pages)
                    for label in labels
                ]
            elif len(labels) == n_pages * panels:
                resolved_labels = [None if label is None else str(label) for label in labels]
            else:
                raise ValueError(
                    "labels for paged Show2D must have length panels_per_page "
                    f"({panels}) or n_pages * panels_per_page ({n_pages * panels}), got {len(labels)}"
                )
        else:
            resolved_labels = [f"Panel {i + 1}" for _ in range(n_pages) for i in range(panels)]
        return (
            arr.reshape(n_pages * panels, *arr.shape[-2:]),
            resolved_labels,
            n_pages,
            panels,
            resolved_page_labels,
            [0] * n_pages,
        )

    return data, labels, 1, 0, [], []


def _round_to_nice(value: float) -> float:
    """Round a physical length to a 'nice' value (1, 2, 5, 10, 20, 50, ...)."""
    if value <= 0:
        return 1.0
    exp = math.floor(math.log10(value))
    base = 10 ** exp
    mantissa = value / base
    if mantissa < 1.5:
        return base
    elif mantissa < 3.5:
        return 2 * base
    elif mantissa < 7.5:
        return 5 * base
    else:
        return 10 * base


def _js_round(value: float) -> int:
    """JS ``Math.round`` (half away from zero for positives); Python's
    banker's rounding would format 2.5 as "2" where the widget shows "3"."""
    return math.floor(value + 0.5)


# Ports of js/figure.ts unit tables so the static PNG's scale bar label is
# character-identical to the live widget's canvas label.
# Length-unit ladder, each as its size in nm: a sub-1 value in one unit
# (e.g. 0.5 nm) displays as a clean integer in a smaller unit (5 Å), because
# microscopists read "5 Å", not "0.50 nm".
_LENGTH_UNITS_NM: tuple[tuple[str, float], ...] = (
    ("mm", 1e6), ("µm", 1e3), ("nm", 1.0), ("Å", 0.1), ("pm", 1e-3),
)
# Base unit (the trait's unit) -> nm. Only length units rescale; anything else
# (mrad, ps, px, ...) keeps its own unit and the decimal fallback.
_BASE_UNIT_NM: dict[str, float] = {
    "mm": 1e6, "µm": 1e3, "μm": 1e3, "micron": 1e3, "microns": 1e3, "um": 1e3,
    "nm": 1.0, "nanometer": 1.0, "nanometers": 1.0,
    "å": 0.1, "angstrom": 0.1, "angstroms": 0.1, "ang": 0.1, "a": 0.1,
    "pm": 1e-3, "picometer": 1e-3, "picometers": 1e-3,
}


def _unit_symbol(unit: str) -> str:
    """Display symbol for a unit string (port of js/figure.ts unitSymbol).

    Users pass units like "micron"/"A" on a Dataset; the widget renders the
    conventional glyph (µm, Å) so labels read like a journal figure. Unknown
    strings pass through unchanged."""
    u = (unit or "").strip()
    lc = u.lower()
    if lc in ("micron", "microns", "um") or u in ("μm", "µm"):
        return "µm"
    if lc in ("angstrom", "angstroms", "ang", "a") or u == "Å":
        return "Å"
    if lc in ("nanometer", "nanometers", "nm"):
        return "nm"
    if lc in ("picometer", "picometers", "pm"):
        return "pm"
    if lc in ("millimeter", "millimeters", "mm"):
        return "mm"
    if lc in ("picosecond", "picoseconds", "ps"):
        return "ps"
    if lc in ("femtosecond", "femtoseconds", "fs"):
        return "fs"
    if lc in ("nanosecond", "nanoseconds", "ns"):
        return "ns"
    return u


def _format_scale_label(value: float, unit: str) -> str:
    """Scale bar label (port of js/figure.ts formatScaleLabel).

    Length values auto-pick the unit that reads as a clean integer:
    0.5 nm -> "5 Å", 0.005 nm -> "5 pm". Non-length units (mrad, ps, px)
    keep their unit. ``_round_to_nice`` gives n*10^k and every ladder step
    is a power of 10, so the rescaled number is always exact."""
    nice = _round_to_nice(value)
    base_nm = _BASE_UNIT_NM.get((unit or "").strip().lower())
    if base_nm is None:
        sym = _unit_symbol(unit)
        return f"{_js_round(nice)} {sym}" if nice >= 1 else f"{nice:.2f} {sym}"
    value_nm = nice * base_nm
    # largest ladder unit where the value is >= 1 -> the fewest-digit integer
    for sym, unit_nm in _LENGTH_UNITS_NM:
        if value_nm / unit_nm >= 1:
            return f"{_js_round(value_nm / unit_nm)} {sym}"
    sym, unit_nm = _LENGTH_UNITS_NM[-1]
    return f"{_js_round(value_nm / unit_nm)} {sym}"


# Rec. 709 luma weights: the standard perceptual grayscale reduction of RGB.
_RGB_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_IDENTITY_PALETTE = ("#2e7d32", "#c62828", "#d81b60", "#1565c0", "#f9a825", "#6a1b9a")


def _is_rgb_item(item: np.ndarray) -> bool:
    """True when a gallery item is an ``(H, W, 3)`` / ``(H, W, 4)`` color image.

    Detection rule (documented in the Show2D docstring): inside a LIST input,
    any item with ``ndim == 3`` and a trailing dim of 3 or 4 is RGB(A). A bare
    3-D ARRAY input keeps ``(N, H, W)`` stack semantics unless its trailing dim
    is 3/4 AND its leading dim is > 4 (so a 3-frame stack of tiny images never
    silently flips to RGB)."""
    return item.ndim == 3 and item.shape[-1] in (3, 4)


def _normalize_rgb(item: np.ndarray) -> np.ndarray:
    """Display-ready ``(H, W, 3)`` float32 in [0, 1] from an RGB(A) item.

    uint8 input scales by 1/255; float input is clipped to [0, 1] (overlays
    like drift's green-magenta ``overlay_pair`` are already in that range).
    An alpha channel is dropped: RGB panels bypass the contrast pipeline and
    are composited on an opaque canvas, so alpha has nothing to blend with."""
    rgb = np.asarray(item)[..., :3]
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    return np.clip(rgb.astype(np.float32), 0.0, 1.0)


def _normalize_grayscale_panel_items(
    images: Sequence[np.ndarray],
    panel_frame_indices: Sequence[int] | None = None,
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    """Normalize a mixed static/stack gallery to per-panel ``(F, H, W)`` arrays.

    A 2-D item is a static one-frame panel. A 3-D item is a local frame stack
    for that panel. Spatial shapes are center-padded to one common gallery size,
    while frame counts remain independent.
    """
    if not images:
        raise ValueError("Show2D requires at least one image panel")

    stacks: list[np.ndarray] = []
    for panel, image in enumerate(images):
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim != 3:
            raise ValueError(
                "Show2D list items must be 2D images or 3D (frames, rows, cols) "
                f"stacks; panel {panel} has shape {arr.shape}"
            )
        if 0 in arr.shape:
            raise ValueError(
                f"Show2D panel {panel} is empty (shape {arr.shape}); all dimensions must be >= 1"
            )
        if np.iscomplexobj(arr):
            raise TypeError(
                f"Show2D panel {panel} contains complex data. Convert first with "
                "np.abs(arr) for magnitude or np.angle(arr) for phase."
            )
        stack = np.array(arr, dtype=np.float32, copy=True)
        if not np.isfinite(stack).all():
            raise ValueError(
                f"Show2D panel {panel} contains NaN or inf. Clean first with "
                "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
            )
        stacks.append(stack)

    target_h = max(int(stack.shape[-2]) for stack in stacks)
    target_w = max(int(stack.shape[-1]) for stack in stacks)
    normalized: list[np.ndarray] = []
    for stack in stacks:
        if stack.shape[-2:] == (target_h, target_w):
            normalized.append(stack)
        else:
            normalized.append(
                np.stack(
                    [_resize_image(frame, target_h, target_w) for frame in stack],
                    axis=0,
                ).astype(np.float32, copy=False)
            )

    if panel_frame_indices is None:
        indices = [0] * len(normalized)
    else:
        if len(panel_frame_indices) != len(normalized):
            raise ValueError(
                "panel_frame_indices length "
                f"({len(panel_frame_indices)}) must match panel count ({len(normalized)})"
            )
        indices = []
        for panel, (raw_index, stack) in enumerate(zip(panel_frame_indices, normalized)):
            index = int(raw_index)
            if index < 0:
                index += int(stack.shape[0])
            if index < 0 or index >= int(stack.shape[0]):
                raise ValueError(
                    f"panel_frame_indices[{panel}]={raw_index} is outside the valid "
                    f"range [0, {stack.shape[0]})"
                )
            indices.append(index)

    current = np.stack(
        [stack[index] for stack, index in zip(normalized, indices)],
        axis=0,
    ).astype(np.float32, copy=False)
    return normalized, indices, current


def _compose_overlay_pair(reference: np.ndarray, moving: np.ndarray, mode: str) -> np.ndarray:
    """Color overlay of two grayscale images for checking alignment.

    Math parity with ``quantem.imaging.drift.plot.overlay_pair`` (replicated
    locally: the widget must not depend on quantem's drift module). Both images
    share ONE 1/99-percentile scale so the two colors match in brightness;
    per-image scaling would unbalance them. ``green-magenta`` is the
    colorblind-safe default: reference -> magenta, moving -> green, aligned ->
    white. ``rgb`` gives classic red/green (aligned -> yellow) on request only.
    """
    ref = np.asarray(reference, dtype=np.float32)
    mov = np.asarray(moving, dtype=np.float32)
    lo = min(float(np.percentile(ref, 1)), float(np.percentile(mov, 1)))
    hi = max(float(np.percentile(ref, 99)), float(np.percentile(mov, 99)))
    def _norm(arr):
        return np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    ref, mov = _norm(ref), _norm(mov)
    if mode == "rgb":
        return np.stack([ref, mov, np.zeros_like(ref)], -1)  # reference red, moving green, aligned yellow
    return np.stack([ref, mov, ref], -1)                     # reference magenta, moving green, aligned white


def _compose_dual(
    map_a_01: np.ndarray, map_b_01: np.ndarray, gain: Sequence[float]
) -> np.ndarray:
    """Two-map magenta+green composite for underlay ``mode='dual'``.

    Same colorblind-safe convention as :func:`_compose_overlay_pair`
    ``green-magenta``: map A -> magenta (red+blue), map B -> green, co-located
    signal -> white. The inputs are already display-normalized to [0, 1]; each
    channel is then scaled by its own ``gain`` so a weak element can be lifted
    against a strong one. The stored arrays are never touched.
    """
    a = np.clip(np.asarray(map_a_01, dtype=np.float32) * float(gain[0]), 0.0, 1.0)
    b = np.clip(np.asarray(map_b_01, dtype=np.float32) * float(gain[1]), 0.0, 1.0)
    return np.stack([a, b, a], axis=-1).astype(np.float32)  # A magenta, B green, both -> white


_OVERLAY_FONT: list[str] | None = None


def _static_overlay_font() -> list[str]:
    """Closest installed match to the widget's ``-apple-system, BlinkMacSystemFont,
    'Segoe UI', sans-serif`` stack, resolved once. Filtering to installed fonts
    avoids matplotlib findfont warnings on machines without Helvetica/Arial."""
    global _OVERLAY_FONT
    if _OVERLAY_FONT is None:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        preferred = ("Helvetica Neue", "Segoe UI", "Arial", "Liberation Sans")
        _OVERLAY_FONT = [f for f in preferred if f in installed] + ["DejaVu Sans"]
    return _OVERLAY_FONT


def _hex_to_rgb01(value: str | None, fallback: str = "#4fc3f7") -> tuple[float, float, float]:
    """Convert a widget hex color to matplotlib RGB floats."""
    color = (value or fallback).strip()
    if color.startswith("#"):
        color = color[1:]
    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    if len(color) != 6:
        color = fallback.lstrip("#")
    try:
        return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return (0.31, 0.76, 0.97)


class Colormap(StrEnum):
    INFERNO = "inferno"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    PLASMA = "plasma"
    GRAY = "gray"


def _cmap_name(value: str | Colormap) -> str:
    """Return the frontend colormap name for a string or Colormap enum."""
    return value.value if isinstance(value, Colormap) else str(value)


def _is_cmap_sequence(value: object) -> bool:
    """Return True when ``value`` is a per-panel colormap sequence."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


_MAX_PANEL_PLAYBACK_FPS = 30.0


class Show2D(WatchedImageFolderMixin, StaticFallbackMixin, anywidget.AnyWidget):
    """
    2D image comparison viewer with optional local panel stacks and analysis.

    Display a single image or multiple images in a gallery layout. A 3-D item
    inside a list is an independent local stack for that panel, with its own
    in-panel slider and playback. A bare 3-D array remains a static gallery;
    use Show3D when every panel should share one global frame axis.

    Parameters
    ----------
    data : array_like
        2D array (height, width) for single image, or
        3D array (N, height, width) for multiple images displayed as gallery.
        A Python list may mix 2D images with 3D ``(frames, height, width)``
        arrays. Each grayscale 3D list item is one gallery panel with an
        independent slider, playback state, frame count, and current frame;
        local frame counts do not need to match between panels.
        RGB images are first-class: inside a LIST input, any item shaped
        ``(H, W, 3)`` / ``(H, W, 4)`` is treated as an RGB(A) color image
        (uint8 or float in [0, 1]; alpha is dropped). A bare 3-D ARRAY keeps
        the historical ``(N, H, W)`` stack semantics unless its trailing dim
        is 3/4 AND its leading dim is > 4, in which case it is a single RGB
        image. Mixed galleries (grayscale + RGB side by side) are supported:
        RGB panels are display-ready and bypass the colormap, contrast,
        auto/log pipeline; their stats row, histogram, FFT, and line profile
        read the Rec. 709 luminance; the hover readout shows the (r, g, b)
        triplet. ``offline=True``, ``export_html`` and ``rotate`` are not
        supported when RGB panels are present.
    labels : list of str, optional
        Labels for each image in gallery mode.
    panel_overlays : mapping or sequence, optional
        Reproducible per-panel geometric overlays. Each overlay is a mapping
        with ``shape`` equal to ``"circle"``, ``"rect"``/``"rectangle"``, or
        ``"square"``. Circle geometry uses ``center=(row, col)`` plus
        ``radius``; rectangles use ``box=(row0, col0, row1, col1)``,
        ``xyxy=(col0, row0, col1, row1)``, or ``xywh=(col, row, width,
        height)``; squares may use ``center`` plus ``size``. A dictionary
        keyed by panel index or label targets specific panels. Coordinates are
        data pixels by default; pass ``coords="relative"`` for normalized
        0-1 panel coordinates. Style keys include ``stroke``,
        ``stroke_width``, ``line_style``, ``dash``, ``stroke_opacity``,
        ``fill``, ``fill_opacity``, ``opacity``, and ``z_order``.
    overlays : mapping or sequence, optional
        Convenience alias for shared geometric overlays. A single overlay or a
        flat list without ``panel=`` is broadcast to every panel. Use either
        ``overlays`` or ``panel_overlays``, not both.
    title : str, optional
        Title to display above the image(s).
    cmap : str or sequence of str, default "inferno"
        Colormap name ("magma", "viridis", "gray", "inferno", "plasma").
        A sequence assigns one colormap per panel while preserving the first
        entry as the fallback/global colormap for older saved states.
    sampling : float or tuple of float, optional
        Pixel size per axis ``(row, col)``. Scalar broadcasts to both axes.
        Used for scale bar display. Defaults to ``(1, 1)``.
    units : str or list of str, optional
        Unit string per axis. Scalar broadcasts to both. Common: ``"A"``,
        ``"nm"``, ``"pixels"``. Defaults to ``["pixels", "pixels"]``.
    show_fft : bool, default False
        Show FFT and histogram panels. Every interactive FFT panel includes its
        current browser-side zoom multiplier (for example ``2.0×`` or
        ``5.0×``), including independently zoomed gallery FFTs.
    fft_metrics : bool, default True
        Show compact FFT quality labels inside FFT panels when FFT is visible.
        The frontend computes these labels from the cached FFT magnitude and
        does not trigger another FFT.
    show_controls : bool, default True
        Show the live control UI. Set ``False`` for a permanently clean display.
    controls_collapsed : bool, default False
        Start with the live control UI collapsed behind a small GUI toggle.
        Unlike ``show_controls=False``, users can expand the controls in the
        frontend and Python can call ``expand_controls()`` later.
    show_stats : bool, default True
        Show statistics (mean, min, max, std).
    debug : bool, default False
        Show a compact frontend FPS/debug badge in the widget title row.
    log_scale : bool, default False
        Use log scale for intensity mapping.
    auto_contrast : bool, default False
        Use percentile-based contrast.
    vmin : float, optional
        Absolute minimum intensity for color mapping. When both vmin and vmax
        are set, all gallery images share the same intensity scale: essential
        for A/B visual comparison.
    vmax : float, optional
        Absolute maximum intensity for color mapping.
    diff_mode : bool, default False
        Append a live signed-difference panel to a grayscale gallery (mirrors
        ``overlay``): the gallery becomes ``[a, b, a - b]`` for a pair, using a
        symmetric diverging colormap centered on zero so positive and negative
        residuals read with equal weight. The reference frame defaults to panel
        0 (change it with the ``diff_reference`` trait); the diff recomputes
        whenever either frame or the reference changes, so it tracks scrubbing
        stacks live. With more than two frames one diff panel is appended per
        non-reference grayscale frame (``#ref - #other``). RGB panels never get
        a diff panel. Combine with ``overlay=True`` for ``[a, b, overlay]`` plus
        the diff panel.
    overlay : bool or str, default False
        Compose an alignment overlay panel from a 2-image grayscale gallery
        (mirrors ``diff_mode``): the gallery becomes ``[a, b, overlay]`` with
        image 0 -> magenta, image 1 -> green, aligned -> white (the
        colorblind-safe default; same math as quantem drift's
        ``overlay_pair``: one shared 1/99-percentile scale across both
        images). ``overlay="rgb"`` gives classic red/green (aligned ->
        yellow). Requires exactly 2 grayscale images. Combine with
        ``diff_mode=True`` for ``[a, b, overlay]`` plus the dynamic signed
        diff panel of the pair.
    underlay : bool or str, default False
        Compose chemistry-on-structure from exactly two grayscale inputs
        ``[haadf, map]``: the gallery becomes ``[haadf, map, map on HAADF]``,
        blending the element map (magenta) onto the HAADF (gray structure) as
        a third RGB panel so sparse EDS signal reads against the lattice
        without washing bright columns to white. Pass ``True`` or ``"haadf"``.
        Requires exactly two single-frame grayscale images (no per-panel frame
        stacks) and is mutually exclusive with ``overlay``. The two sources are
        never modified; only the composed panel is added. See ``underlay_alpha``
        and ``underlay_haadf_gain`` to tune the blend.
    underlay_alpha : float, default 0.95
        Opacity of the element map over the HAADF in the ``underlay`` blend, in
        ``[0, 1]``. Higher means more chemistry, less structure; the slider
        re-blends live without touching the sources.
    underlay_haadf_gain : float, default 0.35
        Brightness of the HAADF structure showing through the ``underlay``
        blend, in ``[0, 1]``. Lower keeps the map colors saturated over a dim
        lattice; higher lets more of the gray structure through.
    underlay_mode : {"haadf", "dual"}, default "haadf"
        Composite recipe for the third ``underlay`` panel. ``"haadf"`` blends a
        single element map (magenta) onto the HAADF lattice. ``"dual"`` takes
        two grayscale maps ``[map A, map B]`` (no HAADF) and composes a
        colorblind-safe magenta+green panel: map A -> magenta, map B -> green,
        co-located signal -> white. Both modes need exactly two single-frame
        grayscale inputs.
    stretch_percentiles : sequence of float, default (4.0, 99.0)
        Low/high display-stretch percentiles applied to the element map(s)
        before colorizing, matching the drift-paper Fig4 sweep. The slider
        re-stretches live without touching the stored counts; must satisfy
        ``0 <= low < high <= 100``.
    display_gamma : float, default 0.75
        Presence gamma inside the ``"haadf"`` blend, > 0. Below 1 lifts
        mid-count columns into color; above 1 keeps only the brightest lit.
        Ignored in ``"dual"`` mode (the magenta+green composite has no ghost).
    dual_gain : sequence of float, default (1.0, 1.0)
        Per-channel brightness ``[gain A, gain B]`` for the ``"dual"`` composite,
        each >= 0. Raise one channel to balance a weak element against a strong
        one; ignored in ``"haadf"`` mode.
    ncols : int, default 3
        Number of columns in gallery mode.
    panel_frame_indices : sequence of int, optional
        Initial frame for every panel when ``data`` is a list containing local
        3-D stacks. Each value is resolved against that panel's own frame count.
        Static 2-D panels accept only ``0``. Negative indices follow Python
        indexing, so ``-1`` starts a stack on its final frame.
    panel_playback_fps : float, default 10
        Playback speed shared by the independent local-stack play buttons.
        Configure it at construction without adding another toolbar control.
        Values above 30 are capped at the browser playback budget.
    size : int, default 0
        Canvas rendering size in CSS pixels (the on-screen width of each image).
        ``0`` uses the frontend default: 500 px for a single image, 300 px per
        image in gallery mode.  Pass e.g. ``size=800`` to enlarge for a
        presentation, or ``size=200`` to compress alongside a control panel.
        This controls **display only**: the underlying image resolution is
        never resampled; zooming into a 4K image preserves every pixel.
    save_state : bool, default False
        When False, saved notebooks omit heavy image buffers and keep a compact
        static preview for cold reopen. Set True only for small widgets that
        must reopen interactively without rerunning the kernel.
    notebook_preview_format : {"jpeg", "webp", "png"} or None, default "jpeg"
        Static preview format used when ``save_state=False``. ``"jpeg"`` is the
        most portable notebook default, ``"webp"`` is smaller for local/report
        workflows, ``"png"`` is lossless but larger, and ``None`` disables the
        preview.
    notebook_preview_quality : int, default 88
        Lossy preview quality for JPEG/WebP, from 1 to 100. Ignored for PNG.
    notebook_preview_max_px : int, default 512
        Longest panel side for the saved-notebook preview. Lower values make
        notebooks smaller; higher values make the static fallback sharper.
    denoise : str or sequence of str, default "none"
        Display-only denoise method for sparse maps (EDS, low dose). Three
        orthogonal choices: ``"none"``, ``"gaussian"``, or ``"anscombe"``
        (Poisson count-respecting smoothing); binning is the separate
        ``denoise_bin`` knob. Recommendation ladder: sparse EDS ->
        ``"anscombe"`` with ``denoise_bin=2`` and sigma 6-10; very sparse ->
        ``"anscombe"`` with ``denoise_bin=4`` and sigma 8-12; decent-dose
        HAADF -> ``"gaussian"`` sigma 1-2 or ``"none"``; anything
        quantitative -> ``"none"``. The compound spellings ``"bin2"``,
        ``"bin2_anscombe"`` and ``"bin4_anscombe"`` stay accepted as aliases
        that fold into (mode, bin); ``"tv"`` remains available from Python
        (not in the UI menu). A scalar applies to every panel; a
        sequence (one entry per panel) gives each panel its own method, e.g.
        ``["none", "anscombe"]`` for a raw vs denoised A/B gallery. Pure view
        transform: the stored array, the stats row, and every export of raw
        data keep the original counts, and the lossless default is
        ``"none"``. When active, a one-line banner announces the reduction
        and how to get raw counts back. RGB panels are never filtered.
        Independent of ``display_bin`` (the GPU display budget knob). The
        per-panel ``denoise_modes`` / ``denoise_sigmas`` / ``denoise_bins``
        lists are the source of truth; the scalar ``denoise`` /
        ``denoise_sigma`` / ``denoise_bin`` traits are the UI editor and,
        while ``denoise_scope == "panel"``, mirror only the selected panel.
        Set per-panel values imperatively with :meth:`set_denoise`.
    denoise_sigma : float or sequence of float, default 4.0
        Smoothing scale in pixels for the Gaussian/Anscombe display filters.
        A sequence sets one sigma per panel.
    denoise_bin : {1, 2, 4} or sequence, default 1
        Display-side 2x bin passes for SNR, combined with ``denoise``.
        ``1`` (the default) is lossless. A sequence sets one bin factor per
        panel. This is the SNR knob for sparse maps (EDS, low dose): it trades
        resolution for counts. It is orthogonal to ``display_bin``, which is a
        performance-only downsample to fit the GPU display budget and does not
        change reported intensities. Reach for ``denoise_bin`` to see faint
        chemistry, ``display_bin`` only to render a huge gallery faster.
    show_denoise : bool, default False
        Shows the denoise controls row; does not itself denoise - use
        ``denoise=`` for that. Hidden by default to keep the widget clean;
        auto-enabled when any panel starts with an active denoise. An active
        reduction always shows its banner, even with the row hidden.
    denoise_scope : {"all", "panel"}, default "all"
        UI knob scope: "all" applies Denoise/σ/Bin edits to every panel,
        "panel" edits only the selected panel. Passing any per-panel sequence
        switches to the "panel" scope automatically. In gallery mode the
        toggle lives in the Link group (Link Zoom / Pan / Contrast / Denoise):
        checked means linked ("all"), unchecked means per panel.

        .. deprecated::
            The ``display_filter``-era kwargs ``display_filter``,
            ``display_sigma``, ``spatial_bin`` and ``filter_per_panel`` are
            still accepted for one release and map onto ``denoise``,
            ``denoise_sigma``, ``denoise_bin`` and ``denoise_scope``
            respectively. Passing any of them emits a ``DeprecationWarning``;
            if both a new kwarg and its deprecated alias are given, the new
            kwarg wins. In particular ``spatial_bin`` maps onto ``denoise_bin``
            (the SNR knob), not ``display_bin`` (the performance downsample).
    pad_ratio : float, default 0.0
        Ratio-based border added on each side of the displayed frame, as a
        fraction of the image's max(rows, cols). Valid range 0 to 1. The
        border value is the frame minimum, which keeps the colormap floor.
        Display-only and reversible (single-panel widgets only): combine
        with :meth:`crop_to_view` and undo with :meth:`reset_view_ops`.
        When active, a one-line ``view:`` banner announces the reduction.
    Attributes
    ----------
    render_total_ms : int or None
        End-to-end wall clock from constructor start to first browser paint,
        populated by a JS→Python round-trip after the first canvas render.
        ``None`` until the browser has actually painted; also printed to stdout
        when it fires.  Use to triage "is it Python, wire, or the browser?"
        during live acquisitions.
    render_python_build_ms : int or None
        Subset of ``render_total_ms`` covering Python ``__init__`` only.
    render_wire_js_ms : int or None
        Subset covering everything after Python returns: Comm transfer, JS
        decode, colormap, and canvas paint.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show2D

    Single 2D NumPy array:

    >>> Show2D(np.random.rand(512, 512))

    PyTorch tensor (CPU or GPU, any dtype):

    >>> import torch
    >>> Show2D(torch.rand(512, 512))

    3D NumPy stack ``(N, H, W)`` rendered as a gallery:

    >>> Show2D(np.random.rand(6, 256, 256), ncols=3)

    List of arrays with different shapes (center-padded to a common canvas):

    >>> Show2D([np.random.rand(256, 256), np.random.rand(300, 400)])

    One static map beside an independently scrubbed HAADF stack:

    >>> haadf = np.random.rand(26, 256, 256)
    >>> Show2D([np.random.rand(256, 256), haadf],
    ...        labels=["EDS map", "HAADF"], panel_frame_indices=[0, -1])

    Four tomography reconstructions with independent slice counts and sliders:

    >>> reconstructions = [
    ...     np.random.rand(n_slices, 64, 64)
    ...     for n_slices in (8, 12, 16, 20)
    ... ]
    >>> labels = ["baseline", "regularized", "fine z", "alternative"]
    >>> w = Show2D(reconstructions, labels=labels,
    ...            panel_frame_indices=[3, 5, 7, -1],
    ...            panel_playback_fps=4, ncols=2, show_fft=True)
    >>> _ = w.set_panel_frame("fine z", 8)

    quantem ``Dataset2d``: title, sampling, units auto-extracted:

    >>> from quantem.core.datastructures import Dataset2d
    >>> ds = Dataset2d.from_array(np.random.rand(512, 512))
    >>> Show2D(ds)

    quantem ``Dataset3d``: gallery view of N frames with calibration:

    >>> from quantem.core.datastructures import Dataset3d
    >>> ds = Dataset3d.from_array(np.random.rand(6, 256, 256))
    >>> Show2D(ds, ncols=3)

    A/B comparison with shared contrast and linked zoom/pan:

    >>> a, b = np.random.rand(512, 512), np.random.rand(512, 512)
    >>> Show2D([a, b], vmin=0, vmax=1, link_zoom=True, link_pan=True)

    Per-image absolute contrast (one ``vmin``/``vmax`` per image):

    >>> Show2D([a, b], vmin=[0.0, 0.2], vmax=[1.0, 0.8])

    Drift comparison: diff mode adds a ``A - B`` panel alongside the originals
    (gallery becomes ``[A, B, A - B]``):

    >>> Show2D([a, b], diff_mode=True, link_zoom=True, link_pan=True)

    Alignment overlay: append a composed green-magenta RGB panel
    (gallery becomes ``[a, b, overlay]``; aligned regions read white):

    >>> Show2D([a, b], overlay=True)

    RGB images are first-class gallery items (e.g. drift's green-magenta
    ``overlay_pair`` composite next to the grayscale merge):

    >>> rgb_overlay = np.stack([a, b, a], axis=-1)  # (H, W, 3) in [0, 1]
    >>> Show2D([a, rgb_overlay], labels=["merged", "overlay (RGB)"])

    Large image: display-only canvas size (full resolution preserved):

    >>> Show2D(np.random.rand(4096, 4096), size=800)

    Per-panel display width for galleries. Use ``ncols`` to choose an
    intentional gallery shape, for example ``ncols=2`` for a 2×2 comparison
    of four images or ``ncols=4`` for a single row:

    >>> Show2D(np.random.rand(13, 128, 128), ncols=13, panel_width_px=70)

    Static export to PDF or PNG (vector PDF for publication figures):

    >>> w = Show2D(np.random.rand(512, 512), sampling=0.5, units="nm")
    >>> w.save_image("figure.pdf", dpi=150)

    Denoise a sparse EDS map for display (raw counts stay untouched):

    >>> eds_map = np.random.poisson(0.3, (256, 256)).astype(np.float32)
    >>> Show2D(eds_map, denoise="anscombe", denoise_bin=2, denoise_sigma=8)

    Raw vs denoised A/B gallery from one call (per-panel ``denoise`` list):

    >>> Show2D([eds_map, eds_map], denoise=["none", "anscombe"], denoise_sigma=8)

    Chemistry on structure: blend an element map onto HAADF (magenta on gray):

    >>> haadf = np.random.rand(256, 256).astype(np.float32)
    >>> Show2D([haadf, eds_map], underlay=True)

    Zoom into a feature, commit that window as the display, then undo it:

    >>> w = Show2D(eds_map, view_box=(64, 64, 96))  # zoom to a 96x96 region
    >>> w.crop_to_view()      # the window becomes the whole displayed frame
    >>> w.reset_view_ops()    # back to the full frame, bit-identical
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show2d.js"

    # =========================================================================
    # Core State
    # GPU memory budget for display buffers (MB). Each 4K image needs ~192 MB.
    # 12×4K = 2304 MB fits. 16+ triggers auto-bin.
    _GPU_DISPLAY_BUDGET_MB = 2500
    # Wire budget for the initial frame payload, per panel. A 6144x6144 float32
    # panel is 151 MB and Jupyter's kernel->browser channel moves ~24 MB/s, so
    # shipping it wholesale stalls first paint for ~6 s while the ~1000 px
    # canvas can only show ~1 MP of it. Panels above this budget send a binned
    # preview instead; the browser streams full-res detail for the visible
    # window on zoom (maps-style). 16 MiB keeps a plain 2048x2048 float32
    # image (exactly 16 MiB) on the classic full-payload path.
    _WIRE_BUDGET_BYTES_PER_PANEL = 16 * 1024 * 1024
    # Cap on one detail reply so a zoom refetch stays sub-second on the slow
    # Tornado channel. The JS coarsens the tile bin until the crop fits.
    _DETAIL_BUDGET_BYTES = 8 * 1024 * 1024

    # =========================================================================
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    n_images = traitlets.Int(1).tag(sync=True)
    folder_waiting = traitlets.Bool(False).tag(sync=True)
    folder_status = traitlets.Unicode("").tag(sync=True)
    folder_watch_state = traitlets.Enum(
        values=FOLDER_WATCH_STATE_VALUES,
        default_value="hidden",
    ).tag(sync=True)
    folder_watch_detail = traitlets.Unicode("").tag(sync=True)
    n_pages = traitlets.Int(1).tag(sync=True)
    page_idx = traitlets.Int(0).tag(sync=True)
    panels_per_page = traitlets.Int(0).tag(sync=True)
    page_kind = traitlets.Enum(
        ["comparison", "items"],
        default_value="comparison",
    ).tag(sync=True)
    page_labels = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    page_starred = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    _display_bin_factor = traitlets.Int(1).tag(sync=True)  # 1 = full-res, 2/4/8 = binned
    # Display-only denoise/bin for sparse maps (EDS, low dose). View transform
    # applied while packing frame_bytes; the stored data is never modified and
    # the lossless default is "none". Independent of _display_bin_factor (GPU
    # budget); denoise_bin here is the EDS bin-for-SNR knob.
    denoise = traitlets.Unicode("none").tag(sync=True)
    denoise_sigma = traitlets.Float(4.0).tag(sync=True)
    denoise_bin = traitlets.Int(1).tag(sync=True)
    # Per-panel resolved knobs: one entry per panel, the packing source of
    # truth. Constructed from scalar (broadcast) or sequence kwargs; the
    # scalar traits above are the UI-facing editor whose scope is controlled
    # by denoise_scope ("all" broadcasts, "panel" edits the selected panel).
    denoise_modes = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    denoise_sigmas = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    denoise_bins = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    denoise_scope = traitlets.Unicode("all").tag(sync=True)
    denoise_banner = traitlets.Unicode("").tag(sync=True)
    # The denoise controls row is hidden by default; this toggle shows it.
    # Auto-enabled at construction when any panel starts with an active
    # denoise so the knobs that explain the view are immediately visible.
    show_denoise = traitlets.Bool(False).tag(sync=True)
    # Master ON/OFF of the denoise EFFECT (the "Denoise" toggle). Off shows the
    # raw view; the per-panel config (modes/sigmas/bins) is PRESERVED, just
    # gated, so toggling back on restores it. Distinct from show_denoise, which
    # is only the editor-row visibility.
    denoise_enabled = traitlets.Bool(True).tag(sync=True)
    # Frequency-domain view filter. Unlike Denoise this deliberately removes
    # real signal, so it has a separate master, settings row, and honest banner.
    # Frequencies are normalized to Nyquist (0..1) until physical sampling is
    # available in the browser; raw arrays/stats/exports never use these traits.
    frequency_filter = traitlets.Enum(
        ["none", "lowpass", "highpass", "bandpass"], default_value="none"
    ).tag(sync=True)
    frequency_filter_enabled = traitlets.Bool(False).tag(sync=True)
    frequency_filter_cutoff = traitlets.Float(0.15).tag(sync=True)
    frequency_filter_center = traitlets.Float(0.30).tag(sync=True)
    frequency_filter_width = traitlets.Float(0.12).tag(sync=True)
    frequency_filter_modes = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    frequency_filter_cutoffs = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    frequency_filter_centers = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    frequency_filter_widths = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    frequency_filter_scope = traitlets.Enum(["all", "panel"], default_value="all").tag(sync=True)
    frequency_filter_banner = traitlets.Unicode("").tag(sync=True)
    show_frequency_filter = traitlets.Bool(False).tag(sync=True)
    # Browser-side filter negotiation: JS sets this True when a real (non
    # software) WebGPU adapter is available. Python then ships RAW frames for
    # panels whose mode the browser can evaluate (gaussian/bin2/anscombe
    # stacks; see BROWSER_DISPLAY_FILTER_MODES) and the WGSL compute port in
    # js/displayFilter.ts filters client-side, so the sigma slider scrubs live
    # with no kernel round trip and kernel-less HTML exports keep working
    # knobs. tv panels always stay on this Python path.
    # Default True: the browser owns display denoise. js/displayFilter.ts ships
    # The browser has WGSL and CPU TypeScript filter paths that match NumPy, so
    # every viewer filters client-side and Python never needs the scipy round
    # trip, which re-sent the whole frame over comm on every knob edit. The
    # frontend downgrades this to False only on a software (SwiftShader) adapter.
    _webgpu_filter_ok = traitlets.Bool(True).tag(sync=True)
    # Chemistry-on-structure view: HAADF-modulated blend of an element map on
    # the HAADF lattice as a third RGB panel (haadf | map | blend). Enabled at
    # construction with underlay=True on exactly two grayscale inputs.
    underlay = traitlets.Bool(False).tag(sync=True)
    underlay_alpha = traitlets.Float(0.95).tag(sync=True)
    underlay_haadf_gain = traitlets.Float(0.35).tag(sync=True)
    # Fig4 parity knobs, all live-scrubbable: the map's display stretch
    # (low/high percentile), the presence gamma inside the HAADF blend, the
    # composite mode ('haadf' = map-on-HAADF, 'dual' = two-map magenta+green),
    # and per-channel gains for the dual composite.
    stretch_percentiles = traitlets.List(
        traitlets.Float(), default_value=[4.0, 99.0]
    ).tag(sync=True)
    display_gamma = traitlets.Float(0.75).tag(sync=True)
    underlay_mode = traitlets.Unicode("haadf").tag(sync=True)
    dual_gain = traitlets.List(traitlets.Float(), default_value=[1.0, 1.0]).tag(sync=True)
    _gpu_max_buffer_mb = traitlets.Int(0).tag(sync=True)  # GPU reports maxBufferSize (JS→Python)
    # Flipped True by JS after the first colormap pass has painted to canvas.
    # Used by the Python-side truthful timing print (end-to-end wall clock, not just __init__).
    _js_rendered = traitlets.Bool(False).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    frame_bytes_url = traitlets.Unicode("").tag(sync=True)
    frame_bytes_urls = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    # Optional per-panel frame stacks. Static panels keep count=1 and are not
    # duplicated in panel_stack_bytes; offsets are -1 for those panels.
    panel_frame_counts = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    panel_frame_indices = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    panel_playback_fps = traitlets.Float(10.0).tag(sync=True)
    panel_stack_offsets = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    panel_stack_bytes = traitlets.Bytes(b"").tag(sync=True)
    panel_stack_bytes_url = traitlets.Unicode("").tag(sync=True)
    _panel_stack_mins = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    _panel_stack_maxs = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    # Offline mode: stack quantized to uint8 against global (min, max). 4x
    # smaller than float32 — drops standalone HTML from ~200 MB to ~110 MB.
    # JS dequantizes on read. Eye can't tell uint8 from float32 after colormap
    # reduces to 256 levels anyway.
    offline = traitlets.Bool(False).tag(sync=True)
    # True only on a clone written by export_html: forces the standalone HTML to
    # render on a light/white background regardless of the viewer's OS theme.
    _export_light = traitlets.Bool(False).tag(sync=True)
    _static_fallback_jpeg = traitlets.Unicode("").tag(sync=True)
    _static_fallback_mime = traitlets.Unicode("image/jpeg").tag(sync=True)
    _offline_min = traitlets.Float(0.0).tag(sync=True)
    _offline_max = traitlets.Float(1.0).tag(sync=True)
    # Per-image quantization ranges. A gallery's panels can span very different
    # intensities; one GLOBAL (min, max) then wastes most of the 256 uint8 codes
    # on whichever panel is widest, so a narrow panel keeps only a handful of
    # levels and its histogram shows a coarse comb. Quantizing each panel against
    # its OWN (min, max) gives every panel the full 256 codes -> clean histogram.
    _offline_mins = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    _offline_maxs = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    # Maps-style detail streaming (active whenever the preview is binned, i.e.
    # _display_bin_factor > 1): JS writes a JSON request describing the visible
    # window per panel; Python replies with cropped + binned float32 tiles.
    # Request/response over traits, same pattern as export_request below.
    _detail_request = traitlets.Unicode("").tag(sync=True)
    _detail_meta = traitlets.Unicode("").tag(sync=True)
    _detail_bytes = traitlets.Bytes(b"").tag(sync=True)
    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)
    saved_view_states = traitlets.List([]).tag(sync=True)
    saved_view_request = traitlets.Unicode("").tag(sync=True)
    saved_view_status = traitlets.Unicode("").tag(sync=True)
    handoff_request = traitlets.Unicode("").tag(sync=True)
    handoff_status = traitlets.Unicode("").tag(sync=True)
    handoff_enabled = traitlets.Bool(True).tag(sync=True)
    prepared_view_widget = traitlets.Instance(ipywidgets.Widget, allow_none=True).tag(
        sync=True,
        **ipywidgets.widget_serialization,
    )
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    panel_title_spans = traitlets.List(default_value=[]).tag(sync=True)
    # Per-panel RGB flag. True panels carry display-ready (H, W, 3) pixels that
    # bypass the colormap/contrast pipeline in JS; False panels are grayscale.
    is_rgb = traitlets.List(traitlets.Bool(), default_value=[]).tag(sync=True)
    starred = traitlets.List(traitlets.Int()).tag(sync=True)
    hidden_panels = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    hidden_page_slots = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    panel_order = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    show_panel_titles = traitlets.Bool(True).tag(sync=True)
    panel_title_font_size = traitlets.Int(11).tag(sync=True)
    panel_title_style = traitlets.Dict(default_value={}).tag(sync=True)
    inter_panel_gap_px = traitlets.Int(0).tag(sync=True)
    inter_panel_gap_color = traitlets.Unicode("").tag(sync=True)
    gallery_outer_border_px = traitlets.Int(0).tag(sync=True)
    gallery_outer_border_color = traitlets.Unicode("").tag(sync=True)
    panel_inner_border_px = traitlets.Float(1.0).tag(sync=True)
    panel_inner_border_color = traitlets.Unicode("#d0d0d0").tag(sync=True)
    gallery_gap_px = traitlets.Int(0).tag(sync=True)
    gallery_gap_color = traitlets.Unicode("").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    panel_cmaps = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    panel_cmaps_memory = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    ncols = traitlets.Int(3).tag(sync=True)

    # =========================================================================
    # Display Options
    # =========================================================================
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    contrast_preset = traitlets.Unicode("custom").tag(sync=True)
    histogram_advanced = traitlets.Bool(False).tag(sync=True)
    show_histogram_advanced = traitlets.Bool(False).tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmins = traitlets.List(trait=traitlets.Float(allow_none=True), allow_none=True, default_value=None).tag(sync=True)
    vmaxs = traitlets.List(trait=traitlets.Float(allow_none=True), allow_none=True, default_value=None).tag(sync=True)
    identity_colors = traitlets.List(trait=traitlets.Unicode(), default_value=[]).tag(sync=True)
    marker_colors = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    marker_style = traitlets.Enum(["left", "around"], default_value="left").tag(sync=True)
    row_markers = traitlets.Dict(default_value={}).tag(sync=True)
    col_markers = traitlets.Dict(default_value={}).tag(sync=True)
    selected_panels = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    inset_plots = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    show_inset_plots = traitlets.Bool(True).tag(sync=True)
    panel_annotations = traitlets.List(traitlets.List(traitlets.Dict()), default_value=[]).tag(sync=True)
    panel_overlays = traitlets.List(traitlets.List(traitlets.Dict()), default_value=[]).tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_sizes = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    pixel_unit = traitlets.Unicode("pixels").tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    scale_bar_position = traitlets.Unicode("bottom-right").tag(sync=True)
    scale_bar_panels = traitlets.List(trait=traitlets.Int(), default_value=[]).tag(sync=True)
    scale_bar_length = traitlets.Float(None, allow_none=True).tag(sync=True)
    scale_bar_label = traitlets.Unicode("").tag(sync=True)
    scale_bar_style = traitlets.Dict(default_value={}).tag(sync=True)
    show_zoom_indicator = traitlets.Bool(False).tag(sync=True)
    size = traitlets.Int(0).tag(sync=True)  # Canvas rendering size in CSS pixels; 0 = frontend default
    smooth = traitlets.Bool(False).tag(sync=True)
    initial_zoom = traitlets.Float(1.0).tag(sync=True)
    flip_rows = traitlets.Bool(False).tag(sync=True)
    flip_cols = traitlets.Bool(False).tag(sync=True)
    zoom_row = traitlets.Float(None, allow_none=True).tag(sync=True)
    zoom_col = traitlets.Float(None, allow_none=True).tag(sync=True)
    # Live viewport (row0, row1, col0, col1) in image pixel coordinates.
    # Python -> JS at construction (the view_box= sugar below derives zoom +
    # zoom_row/zoom_col from it); JS -> Python on every pan/zoom (debounced
    # ~100 ms), so current_view always reflects what is on screen. The box is
    # axis-aligned: two corners (row0, col0)/(row1, col1) define all four.
    view_box = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    # Reversible display-window ops (single panel only). view_crop holds the
    # committed crop (row0, row1, col0, col1) in FULL-RESOLUTION image pixels
    # (empty = no crop); pad_ratio adds a constant border around the displayed
    # frame. The fill value follows pad_fill_mode (min/median/mean). Both are display-only view
    # transforms applied while packing frame_bytes: the stored data is never
    # modified and reset_view_ops() restores the full frame bit-identically.
    view_crop = traitlets.List(trait=traitlets.Int(), default_value=[]).tag(sync=True)
    pad_ratio = traitlets.Float(0.0).tag(sync=True)
    pad_ratios = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    pad_fill_mode = traitlets.Unicode("min").tag(sync=True)
    pad_fill_modes = traitlets.List(trait=traitlets.Unicode(), default_value=[]).tag(sync=True)
    pad_scope = traitlets.Unicode("all").tag(sync=True)
    # One-line announcement of an active crop/pad. Same house rule as
    # denoise_banner: an active view reduction is never silent.
    view_banner = traitlets.Unicode("").tag(sync=True)
    # (row, col) native-pixel offset of the packed frame's origin relative to
    # the full image. JS adds it to cursor readouts so displayed coordinates
    # stay full-image while a crop or pad is active.
    _view_crop_offset = traitlets.List(trait=traitlets.Int(), default_value=[0, 0]).tag(sync=True)
    link_zoom = traitlets.Bool(False).tag(sync=True)
    link_pan = traitlets.Bool(False).tag(sync=True)
    link_contrast = traitlets.Bool(True).tag(sync=True)
    diff_mode = traitlets.Bool(False).tag(sync=True)
    diff_reference = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # UI Visibility
    # =========================================================================
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    debug = traitlets.Bool(False).tag(sync=True)
    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)

    # =========================================================================
    # Analysis Panels (FFT + Histogram shown together)
    # =========================================================================
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)
    fft_metrics = traitlets.Bool(True).tag(sync=True)

    # =========================================================================
    # Selected Image (for single-image analysis display)
    # =========================================================================
    selected_idx = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # ROI Selection
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_list = traitlets.List([]).tag(sync=True)
    roi_selected_idx = traitlets.Int(-1).tag(sync=True)

    # =========================================================================
    # Line Profile
    # =========================================================================
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)

    # =========================================================================
    # Per-Image Rotation
    # =========================================================================
    image_rotations = traitlets.List(traitlets.Int(), []).tag(sync=True)
    rotation_scope = traitlets.Enum(["all", "panel"], default_value="all").tag(sync=True)
    image_flips_horizontal = traitlets.List(traitlets.Bool(), default_value=[]).tag(sync=True)
    image_flips_vertical = traitlets.List(traitlets.Bool(), default_value=[]).tag(sync=True)

    @classmethod
    def from_gif(
        cls,
        path: str | pathlib.Path,
        *,
        ncols: int | None = None,
        labels: list[str | None] | bool | None = None,
        title: str | None = None,
        **kwargs,
    ) -> Self:
        """Open a GIF as a static frame gallery.

        Animated GIFs become an ``(N, H, W)`` stack displayed as a contact sheet.
        Use ``ncols`` to choose an intentional layout, for example ``ncols=2``
        for a 2x2 denoising comparison or ``ncols=1`` for a vertical strip.
        """
        from quantem.widget.io import read_gif  # noqa: PLC0415

        ds = read_gif(path)
        frame_count = int(ds.array.shape[0])
        if ncols is None:
            ncols = max(1, math.ceil(math.sqrt(frame_count)))
        if labels is True:
            resolved_labels = [f"Frame {i + 1}" for i in range(frame_count)]
        elif labels:
            resolved_labels = list(labels)
        else:
            resolved_labels = None
        return cls(
            ds,
            labels=resolved_labels,
            title=ds.name if title is None else title,
            ncols=ncols,
            **kwargs,
        )

    @classmethod
    def from_folder(
        cls,
        path: str | pathlib.Path,
        *,
        pattern: str = "*",
        recursive: bool = False,
        watch: bool = True,
        watch_interval: float = 1.0,
        page_size: int | None = _DEFAULT_FOLDER_PAGE_SIZE,
        **kwargs,
    ) -> Self:
        """Display readable folder images as a paged full-resolution gallery.

        Files are ordered naturally (``image_2`` before ``image_10``) and read
        through :func:`quantem.widget.io.read_image`, including EMD, TIFF, PNG,
        NPY, and DM calibration paths. The same widget is updated when stable
        files are added. Unreadable files remain retryable. At most
        ``page_size`` panels are visible at once; pass ``None`` to keep one
        unpaged gallery.

        Parameters
        ----------
        path : str or pathlib.Path
            Folder containing independent 2D image files.
        pattern : str, default "*"
            Glob selecting files within ``path``.
        recursive : bool, default False
            Search matching files below subdirectories as well.
        watch : bool, default True
            Start background polling immediately.
        watch_interval : float, default 1.0
            Seconds between background polls.
        page_size : int or None, default 20
            Maximum visible panels per page. Paging appears only after the
            ready image count exceeds this value. Pass ``None`` to disable
            automatic folder paging.
        **kwargs
            Normal :class:`Show2D` options. File-derived labels and data are
            managed by the folder source.
        """
        if "labels" in kwargs:
            raise TypeError(
                "Show2D.from_folder() derives labels from file paths so new files "
                "remain identifiable; remove labels= or construct Show2D directly."
            )
        if "page_labels" in kwargs or "page_kind" in kwargs:
            raise TypeError(
                "Show2D.from_folder() manages page labels and folder-page "
                "semantics as files arrive; remove page_labels=/page_kind= and "
                "use page_size= to configure the gallery"
            )
        resolved_page_size = _validate_folder_page_size(page_size)
        source = WatchedImageFolder(
            path,
            pattern=pattern,
            recursive=recursive,
            interval=watch_interval,
            mode="panels",
        )
        arrays, records = source.read_initial(
            allow_empty=watch,
            require_unchanged_followup=watch,
        )
        explicit_calibration = any(
            key in kwargs
            for key in ("sampling", "units", "pixel_size", "pixel_sizes", "pixel_unit")
        )
        kwargs.setdefault("title", source.folder.name)
        kwargs.setdefault("verbose", False)
        if arrays:
            initial_data = arrays
            initial_labels = [source.label(record.path) for record in records]
        else:
            display_bin = kwargs.get("display_bin", "auto")
            placeholder_side = (
                max(1, int(display_bin))
                if isinstance(display_bin, int) and not isinstance(display_bin, bool)
                else 1
            )
            initial_data = [np.zeros((placeholder_side, placeholder_side), dtype=np.float32)]
            initial_labels = [""]
        widget = cls(initial_data, labels=initial_labels, **kwargs)
        widget._folder_display_bin_request = kwargs.get("display_bin", "auto")
        widget._folder_page_size = resolved_page_size
        with widget.hold_sync():
            widget.page_kind = "items"
            if not arrays:
                widget._set_folder_waiting_empty_state()
            widget._sync_folder_pages(anchor_panel=0)
        source.attach(widget, explicit_calibration=explicit_calibration)
        if watch:
            widget.watch_folder(interval=watch_interval)
        return widget

    @property
    def folder_page_size(self) -> int | None:
        """Maximum visible panels per page for this folder-backed gallery."""
        self._require_folder_source()
        value = getattr(self, "_folder_page_size", None)
        return None if value is None else int(value)

    def set_folder_page_size(self, page_size: int | None) -> Self:
        """Change automatic folder pagination without rebuilding the widget.

        Parameters
        ----------
        page_size : int or None
            Maximum visible panels per page, or ``None`` for one unpaged
            gallery.

        Returns
        -------
        Show2D
            This widget, for method chaining.
        """
        self._require_folder_source()
        self._folder_page_size = _validate_folder_page_size(page_size)
        self._sync_folder_pages(anchor_panel=int(self.selected_idx))
        return self

    def __init__(
        self,
        data: np.ndarray | list[np.ndarray],
        labels: list[str | Sequence[Mapping[str, object]] | None] = None,
        panel_title_spans: Sequence[Sequence[Mapping[str, object]] | Mapping[str, object] | str | None] | None = None,
        page_labels: Sequence[str | None] | None = None,
        title: str = "",
        ui_mode: UiMode = "interactive",
        show_title: bool | None = None,
        cmap: str | Colormap | Sequence[str | Colormap] = Colormap.INFERNO,
        sampling: float | tuple[float, float] | list[float] | None = None,
        units: str | list[str] | None = None,
        scale_bar_visible: bool | None = None,
        show_scale_bar: bool | None = None,
        scale_bar_position: str = "bottom-right",
        scale_bar_panels: Sequence[int | str] | int | str | None = None,
        scale_bar_length: float | None = None,
        scale_bar_label: str | None = None,
        scale_bar_style: Mapping[str, object] | None = None,
        show_zoom_indicator: bool = False,
        show_fft: bool = False,
        fft_window: bool = True,
        fft_metrics: bool = True,
        show_controls: bool | None = None,
        controls_collapsed: bool | None = None,
        show_stats: bool | None = None,
        debug: bool = False,
        verbose: bool = True,
        log_scale: bool = False,
        auto_contrast: bool = False,
        contrast_preset: str = "custom",
        histogram_advanced: bool = False,
        offline: bool = False,
        vmin: float | list | None = None,
        vmax: float | list | None = None,
        identity_colors: Sequence[str] | None = None,
        marker_colors: Sequence[str] | None = None,
        marker_style: str = "left",
        row_markers: Mapping[object, object] | None = None,
        col_markers: Mapping[object, object] | None = None,
        inset_plots: Sequence[dict[str, Any] | None] | dict[str, Any] | None = None,
        show_inset_plots: bool = True,
        panel_annotations: Sequence[object] | Mapping[object, object] | object | None = None,
        overlays: Sequence[object] | Mapping[object, object] | object | None = None,
        panel_overlays: Sequence[object] | Mapping[object, object] | object | None = None,
        ncols: int = 3,
        panel_frame_indices: Sequence[int] | None = None,
        panel_playback_fps: float = 10.0,
        size: int = 0,
        panel_width_px: int = 0,
        smooth: bool = False,
        zoom: float = 1.0,
        rotation: int | float = 0,
        rotations: Sequence[int | float] | None = None,
        rotation_scope: str = "all",
        flip_rows: bool = False,
        flip_cols: bool = False,
        zoom_row: float | None = None,
        zoom_col: float | None = None,
        center: tuple | list | None = None,
        link_zoom: bool | None = None,
        link_pan: bool | None = None,
        link_contrast: bool = True,
        diff_mode: bool = False,
        overlay: bool | str = False,
        show_histogram_advanced: bool | None = None,
        image_flips_horizontal: Sequence[bool] | None = None,
        image_flips_vertical: Sequence[bool] | None = None,
        view_box: tuple | list | None = None,
        pad_ratio: float | Sequence[float] = 0.0,
        pad_fill_mode: str | Sequence[str] = "min",
        pad_scope: str = "all",
        display_bin: int | str = "auto",
        hidden_panels: Sequence[int | str] | int | str | None = None,
        starred: Sequence[int | str] | int | str | None = None,
        panel_order: Sequence[int | str] | None = None,
        show_panel_titles: bool | None = None,
        panel_title_font_size: int = 11,
        panel_title_style: Mapping[str, object] | None = None,
        inter_panel_gap_px: int | None = None,
        inter_panel_gap_color: str | None = None,
        gallery_outer_border_px: int | None = None,
        gallery_outer_border_color: str | None = None,
        panel_inner_border_px: float | int | None = None,
        panel_inner_border_color: str | None = None,
        gallery_gap_px: int | None = None,
        gallery_gap_color: str | None = None,
        state=None,
        save_state: bool = False,
        notebook_preview_format: str | None = "jpeg",
        notebook_preview_quality: int = 88,
        notebook_preview_max_px: int = 512,
        denoise: str | Sequence[str] = _UNSET,
        denoise_sigma: float | Sequence[float] = _UNSET,
        denoise_bin: int | Sequence[int] = _UNSET,
        denoise_scope: str = _UNSET,
        show_denoise: bool = False,
        frequency_filter: str | Sequence[str] = "none",
        frequency_filter_enabled: bool | None = None,
        frequency_filter_cutoff: float | Sequence[float] = 0.15,
        frequency_filter_center: float | Sequence[float] = 0.30,
        frequency_filter_width: float | Sequence[float] = 0.12,
        show_frequency_filter: bool = False,
        display_filter: str | Sequence[str] | None = None,
        display_sigma: float | Sequence[float] | None = None,
        spatial_bin: int | Sequence[int] | None = None,
        filter_per_panel: bool | None = None,
        underlay: bool | str = False,
        underlay_alpha: float = 0.95,
        underlay_haadf_gain: float = 0.35,
        underlay_mode: str = "haadf",
        stretch_percentiles: Sequence[float] = (4.0, 99.0),
        display_gamma: float = 0.75,
        dual_gain: Sequence[float] = (1.0, 1.0),
        **kwargs,
    ):
        import time as _time
        _t0 = _time.perf_counter()
        skip_initial_frame_pack = bool(kwargs.pop("_skip_initial_frame_pack", False))
        skip_initial_stats = bool(kwargs.pop("_skip_initial_stats", False))
        preserve_input_dtype_for_export = bool(kwargs.pop("_preserve_input_dtype_for_export", False))
        requested_panel_cmaps = (
            [_cmap_name(item) for item in cmap]
            if _is_cmap_sequence(cmap)
            else []
        )
        base_cmap = requested_panel_cmaps[0] if requested_panel_cmaps else _cmap_name(cmap)
        # Reject typos and stale kwargs (e.g. image_width_px, pixel_size_angstrom).
        # anywidget/traitlets silently ignores unknown keys, which hid the
        # pixel_size_angstrom bug in show2d_all_features.ipynb for months.
        _reject_unknown_kwargs(type(self), kwargs)
        # Deprecated aliases from the display_filter-era API (one rc of
        # compatibility). Key off "was this kwarg supplied" via the _UNSET
        # sentinel so an explicit new kwarg wins even when its value equals the
        # trait default; a deprecated alias only fills an unset new kwarg. Each
        # supplied alias warns regardless of who wins the value.
        denoise_supplied = denoise is not _UNSET
        denoise_sigma_supplied = denoise_sigma is not _UNSET
        denoise_bin_supplied = denoise_bin is not _UNSET
        denoise_scope_supplied = denoise_scope is not _UNSET
        if not denoise_supplied:
            denoise = "none"
        if not denoise_sigma_supplied:
            denoise_sigma = 4.0
        if not denoise_bin_supplied:
            denoise_bin = 1
        def _is_panel_seq(v):
            return isinstance(v, (list, tuple)) and not isinstance(v, str)
        _per_panel_denoise = (
            (denoise_supplied and _is_panel_seq(denoise))
            or (denoise_sigma_supplied and _is_panel_seq(denoise_sigma))
            or (denoise_bin_supplied and _is_panel_seq(denoise_bin))
        )
        if not denoise_scope_supplied:
            # Per-panel denoise specified at construction implies per-panel
            # editing: the single denoise control edits the SELECTED panel
            # rather than broadcasting to all (which would clobber the author's
            # per-panel setup on the first interactive edit). Explicit
            # denoise_scope / filter_per_panel below still win.
            denoise_scope = "panel" if _per_panel_denoise else "all"
        if display_filter is not None:
            warnings.warn(
                "display_filter is deprecated; use denoise= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if not denoise_supplied:
                denoise = display_filter
        if display_sigma is not None:
            warnings.warn(
                "display_sigma is deprecated; use denoise_sigma= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if not denoise_sigma_supplied:
                denoise_sigma = display_sigma
        if spatial_bin is not None:
            warnings.warn(
                "spatial_bin is deprecated; use denoise_bin= instead (the SNR "
                "knob for sparse maps, not the display_bin performance "
                "downsample).",
                DeprecationWarning,
                stacklevel=2,
            )
            if not denoise_bin_supplied:
                denoise_bin = spatial_bin
        if filter_per_panel is not None:
            warnings.warn(
                "filter_per_panel is deprecated; use denoise_scope='all' or "
                "denoise_scope='panel' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if not denoise_scope_supplied:
                denoise_scope = "all" if filter_per_panel else "panel"
        plain_labels, title_spans_from_labels = _normalise_title_span_sequence(labels)
        explicit_plain_labels, explicit_title_spans = _normalise_title_span_sequence(panel_title_spans)
        if plain_labels is None and explicit_plain_labels is not None:
            plain_labels = explicit_plain_labels
        raw_title_span_len = (
            _title_span_sequence_length(panel_title_spans)
            if panel_title_spans is not None
            else (len(plain_labels or []) if title_spans_from_labels else 0)
        )
        data, labels, n_pages, panels_per_page, resolved_page_labels, resolved_page_starred = _normalise_show2d_pages(
            data,
            labels=plain_labels,
            page_labels=page_labels,
        )
        raw_title_spans = explicit_title_spans or title_spans_from_labels
        resolved_panel_title_spans = _expand_title_spans_for_flattened_labels(
            raw_title_spans,
            original_len=raw_title_span_len,
            n_panels=len(labels or []),
            n_pages=n_pages,
            panels_per_page=panels_per_page,
        )
        panel_width_px = int(panel_width_px)
        if panel_width_px < 0:
            raise ValueError(f"panel_width_px must be >= 0, got {panel_width_px}")
        if panel_width_px > 0:
            size = panel_width_px
        ncols = int(ncols)
        if ncols < 1:
            raise ValueError(f"ncols must be >= 1, got {ncols}")
        legacy_gap_used = gallery_gap_px is not None or gallery_gap_color is not None
        resolved_inter_panel_gap_px = _nonnegative_int(
            0 if inter_panel_gap_px is None and gallery_gap_px is None
            else gallery_gap_px if inter_panel_gap_px is None
            else inter_panel_gap_px,
            name="inter_panel_gap_px",
        )
        resolved_inter_panel_gap_color = (
            "" if (gallery_gap_color if inter_panel_gap_color is None else inter_panel_gap_color) is None
            else str(gallery_gap_color if inter_panel_gap_color is None else inter_panel_gap_color)
        )
        if gallery_outer_border_px is None:
            resolved_gallery_outer_border_px = (
                resolved_inter_panel_gap_px
                if legacy_gap_used and resolved_inter_panel_gap_px > 0 and resolved_inter_panel_gap_color
                else 0
            )
        else:
            resolved_gallery_outer_border_px = _nonnegative_int(
                gallery_outer_border_px,
                name="gallery_outer_border_px",
            )
        resolved_gallery_outer_border_color = (
            resolved_inter_panel_gap_color
            if gallery_outer_border_color is None
            else str(gallery_outer_border_color)
        )
        resolved_panel_inner_border_px = (
            1.0
            if panel_inner_border_px is None
            else _nonnegative_float(panel_inner_border_px, name="panel_inner_border_px")
        )
        resolved_panel_inner_border_color = (
            resolved_inter_panel_gap_color
            if panel_inner_border_color is None and legacy_gap_used and resolved_inter_panel_gap_color
            else "#d0d0d0" if panel_inner_border_color is None else str(panel_inner_border_color)
        )
        gallery_gap_px = resolved_inter_panel_gap_px
        gallery_gap_color = resolved_inter_panel_gap_color
        if (
            scale_bar_visible is not None
            and show_scale_bar is not None
            and bool(scale_bar_visible) != bool(show_scale_bar)
        ):
            raise ValueError("Use either show_scale_bar or scale_bar_visible, not conflicting values")
        if scale_bar_position not in {"bottom-right", "bottom-left"}:
            raise ValueError(
                "scale_bar_position must be 'bottom-right' or 'bottom-left'; "
                f"got {scale_bar_position!r}"
            )
        ui = resolve_ui_mode(
            ui_mode,
            defaults={
                "show_title": True,
                "show_controls": True,
                "controls_collapsed": False,
                "show_stats": True,
                "show_panel_titles": True,
                "show_scale_bar": True,
            },
            overrides={
                "show_title": show_title,
                "show_controls": show_controls,
                "controls_collapsed": controls_collapsed,
                "show_stats": show_stats,
                "show_panel_titles": show_panel_titles,
                "show_scale_bar": scale_bar_visible if scale_bar_visible is not None else show_scale_bar,
            },
        )
        show_title = bool(ui["show_title"])
        show_controls = bool(ui["show_controls"])
        controls_collapsed = bool(ui["controls_collapsed"])
        show_stats = bool(ui["show_stats"])
        show_panel_titles = bool(ui["show_panel_titles"])
        scale_bar_visible = bool(ui["show_scale_bar"])
        # save_state controls whether the heavy pixel buffers are persisted into
        # the notebook's metadata.widgets on save. Default False: a plain display
        # embeds only light traits + a static image preview, so a 5-panel 4k
        # gallery does not bake ~1 GB into the .ipynb. Set True to persist full
        # interactive state so a reopened notebook restores the widget without
        # a kernel.
        self._save_state = bool(save_state)
        self._configure_static_fallback(
            notebook_preview_format=notebook_preview_format,
            notebook_preview_quality=notebook_preview_quality,
            notebook_preview_max_px=notebook_preview_max_px,
        )
        super().__init__(**kwargs)
        self._static_fallback_mime = self._static_fallback_mime_type()
        # hold_sync() batches ALL traitlet assignments into a single comm message
        # sent when the context manager exits.  Without this, each self.x = y
        # fires a separate round-trip over the ZMQ/websocket channel, which
        # can add 20+ seconds for a 30-image gallery in VS Code Jupyter.
        if center is not None:
            # center=(row, col) sugar: the friendliest way to say where to look
            zoom_row, zoom_col = float(center[0]), float(center[1])
        if identity_colors is None and marker_colors is not None:
            identity_colors = marker_colors
        panel_title_style = _normalize_panel_title_style(panel_title_style)
        row_markers = _normalize_marker_mapping(row_markers, name="row_markers")
        col_markers = _normalize_marker_mapping(col_markers, name="col_markers")
        n_display_panels = len(labels or []) or (
            int(data.shape[0]) if getattr(data, "ndim", 0) >= 3 else 1
        )
        scale_bar_panels = _normalize_panel_indices(
            scale_bar_panels,
            n_items=n_display_panels,
            labels=labels,
            name="scale_bar_panels",
        )
        if scale_bar_length is not None and (not np.isfinite(float(scale_bar_length)) or float(scale_bar_length) <= 0):
            raise ValueError(f"scale_bar_length must be a positive finite value, got {scale_bar_length!r}")
        scale_bar_style = _normalize_scale_bar_style(scale_bar_style)
        panel_annotations = _normalize_panel_annotations(
            panel_annotations,
            n_items=n_display_panels,
            labels=labels,
        )
        if overlays is not None and panel_overlays is not None:
            raise ValueError("Use either overlays= or panel_overlays=, not both")
        panel_overlays = _normalize_panel_overlays(
            panel_overlays if panel_overlays is not None else overlays,
            n_items=n_display_panels,
            labels=labels,
        )
        if show_histogram_advanced:
            histogram_advanced = True
        if image_flips_horizontal:
            flip_cols = any(bool(value) for value in image_flips_horizontal)
        if image_flips_vertical:
            flip_rows = any(bool(value) for value in image_flips_vertical)
        with self.hold_sync():
            self._init_sync(
                data=data, labels=labels, panel_title_spans=resolved_panel_title_spans,
                title=title, cmap=base_cmap,
                panel_cmaps=requested_panel_cmaps,
                n_pages=n_pages, panels_per_page=panels_per_page,
                page_labels=resolved_page_labels, page_starred=resolved_page_starred,
                show_title=show_title,
                sampling=sampling, units=units, scale_bar_visible=scale_bar_visible,
                scale_bar_position=scale_bar_position,
                scale_bar_panels=scale_bar_panels,
                scale_bar_length=scale_bar_length,
                scale_bar_label=scale_bar_label,
                scale_bar_style=scale_bar_style,
                show_zoom_indicator=show_zoom_indicator,
                show_fft=show_fft, fft_window=fft_window, fft_metrics=fft_metrics,
                show_controls=show_controls, controls_collapsed=controls_collapsed,
                show_stats=show_stats, debug=debug,
                log_scale=log_scale, auto_contrast=auto_contrast, offline=offline,
                contrast_preset=contrast_preset, histogram_advanced=histogram_advanced,
                show_histogram_advanced=show_histogram_advanced,
                vmin=vmin, vmax=vmax, identity_colors=identity_colors, marker_colors=marker_colors,
                marker_style=marker_style, row_markers=row_markers,
                col_markers=col_markers, inset_plots=inset_plots,
                show_inset_plots=show_inset_plots,
                panel_annotations=panel_annotations,
                panel_overlays=panel_overlays,
                ncols=ncols, panel_frame_indices=panel_frame_indices,
                panel_playback_fps=panel_playback_fps,
                size=size, smooth=smooth, zoom=zoom,
                rotation=rotation, rotations=rotations, rotation_scope=rotation_scope,
                flip_rows=flip_rows, flip_cols=flip_cols,
                image_flips_horizontal=image_flips_horizontal,
                image_flips_vertical=image_flips_vertical,
                zoom_row=zoom_row, zoom_col=zoom_col,
                link_zoom=link_zoom, link_pan=link_pan, link_contrast=link_contrast,
                diff_mode=diff_mode, overlay=overlay, view_box=view_box,
                pad_ratio=pad_ratio, pad_fill_mode=pad_fill_mode, pad_scope=pad_scope,
                display_bin=display_bin, hidden_panels=hidden_panels, starred=starred,
                panel_order=panel_order,
                show_panel_titles=show_panel_titles, panel_title_font_size=panel_title_font_size,
                panel_title_style=panel_title_style,
                inter_panel_gap_px=resolved_inter_panel_gap_px,
                inter_panel_gap_color=resolved_inter_panel_gap_color,
                gallery_outer_border_px=resolved_gallery_outer_border_px,
                gallery_outer_border_color=resolved_gallery_outer_border_color,
                panel_inner_border_px=resolved_panel_inner_border_px,
                panel_inner_border_color=resolved_panel_inner_border_color,
                gallery_gap_px=gallery_gap_px,
                gallery_gap_color=gallery_gap_color,
                verbose=verbose, state=state, _t0=_t0,
                denoise=denoise, denoise_sigma=denoise_sigma,
                denoise_bin=denoise_bin, denoise_scope=denoise_scope,
                denoise_scope_explicit=denoise_scope_supplied,
                show_denoise=show_denoise,
                frequency_filter=frequency_filter,
                frequency_filter_enabled=frequency_filter_enabled,
                frequency_filter_cutoff=frequency_filter_cutoff,
                frequency_filter_center=frequency_filter_center,
                frequency_filter_width=frequency_filter_width,
                show_frequency_filter=show_frequency_filter,
                underlay=underlay, underlay_alpha=underlay_alpha,
                underlay_haadf_gain=underlay_haadf_gain,
                underlay_mode=underlay_mode, stretch_percentiles=stretch_percentiles,
                display_gamma=display_gamma, dual_gain=dual_gain,
                skip_initial_frame_pack=skip_initial_frame_pack,
                skip_initial_stats=skip_initial_stats,
                preserve_input_dtype_for_export=preserve_input_dtype_for_export)

    def _init_sync(self, *, data, labels, panel_title_spans, title, cmap, panel_cmaps, n_pages, panels_per_page,
                   page_labels, page_starred, show_title, sampling, units,
                   scale_bar_visible, scale_bar_position, scale_bar_panels, scale_bar_length, scale_bar_label, scale_bar_style, show_zoom_indicator,
                   show_fft, fft_window, fft_metrics,
                   show_controls, controls_collapsed, show_stats, debug, log_scale, auto_contrast, offline,
                   contrast_preset, histogram_advanced, show_histogram_advanced,
                   vmin, vmax, identity_colors, marker_colors, marker_style,
                   row_markers, col_markers, inset_plots, panel_annotations,
                   panel_overlays,
                   show_inset_plots,
                   ncols, panel_frame_indices, panel_playback_fps, size, smooth, zoom,
                   rotation, rotations, rotation_scope,
                   flip_rows, flip_cols, image_flips_horizontal, image_flips_vertical,
                   zoom_row, zoom_col,
                   link_zoom, link_pan, link_contrast, diff_mode, overlay, view_box,
                   pad_ratio, pad_fill_mode, pad_scope,
                   display_bin, hidden_panels, starred, panel_order, show_panel_titles,
                   panel_title_font_size, panel_title_style,
                   inter_panel_gap_px, inter_panel_gap_color,
                   gallery_outer_border_px, gallery_outer_border_color,
                   panel_inner_border_px, panel_inner_border_color,
                   gallery_gap_px, gallery_gap_color, verbose, state, _t0,
                   denoise="none", denoise_sigma=4.0, denoise_bin=1,
                   denoise_scope="all", denoise_scope_explicit=False,
                   show_denoise=False,
                   frequency_filter="none", frequency_filter_enabled=None,
                   frequency_filter_cutoff=0.15, frequency_filter_center=0.30,
                   frequency_filter_width=0.12, show_frequency_filter=False,
                   underlay=False, underlay_alpha=0.95,
                   underlay_haadf_gain=0.35, underlay_mode="haadf",
                   stretch_percentiles=(4.0, 99.0), display_gamma=0.75,
                   dual_gain=(1.0, 1.0), skip_initial_frame_pack=False,
                   skip_initial_stats=False,
                   preserve_input_dtype_for_export=False):
        import time as _time
        self._verbose = verbose
        self.widget_version = resolve_widget_version()
        self._display_data = None  # initialized after data setup
        self._display_bin = 1
        self.prepared_view = None
        self.prepared_view_widget = None
        self.n_pages = int(max(1, n_pages))
        self.panels_per_page = int(max(0, panels_per_page))
        self.page_idx = 0
        self.page_labels = list(page_labels or [])
        self.page_starred = list(page_starred or [])

        # First-class support for quantem Dataset2d / Dataset3d:
        # auto-extract array + sampling + units from the dataset object.
        core_image_dataset_types = _core_image_dataset_types()
        is_core_image_dataset = (
            bool(core_image_dataset_types)
            and isinstance(data, core_image_dataset_types)
        )
        has_image_dataset_shape = (
            hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling")
        )
        if is_core_image_dataset or has_image_dataset_shape:
            if not title and data.name:
                title = data.name
            if sampling is None:
                sampling = tuple(float(s) for s in data.sampling[-2:])
            if units is None and hasattr(data, "units"):
                units = list(data.units[-2:])
            data = data.array
        # Same auto-extract for list/tuple of Dataset2d (gallery from per-file load).
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            first = data[0]
            first_is_core_image_dataset = (
                bool(core_image_dataset_types)
                and isinstance(first, core_image_dataset_types)
            )
            first_has_image_dataset_shape = (
                hasattr(first, "array") and hasattr(first, "sampling")
            )
            if first_is_core_image_dataset or first_has_image_dataset_shape:
                if sampling is None:
                    sampling = tuple(float(s) for s in first.sampling[-2:])
                if units is None and hasattr(first, "units"):
                    units = list(first.units[-2:])
                data = [d.array for d in data]

        # Convert NumPy / PyTorch / list inputs to a NumPy array.
        # RGB detection rule: per-ITEM in a list input, an item with ndim == 3
        # and shape[-1] in (3, 4) is an RGB(A) image. A single non-list ndim==3
        # input is RGB only when shape[-1] in (3, 4) AND shape[0] > 4;
        # otherwise it keeps the historical (N, H, W) stack semantics.
        rgb_flags: list[bool] = []
        rgb_frames: list[np.ndarray | None] = []
        panel_stacks: list[np.ndarray] | None = None
        resolved_panel_frame_indices: list[int] | None = None
        if isinstance(data, (list, tuple)):
            images = [to_numpy(d) for d in data]
            rgb_flags = [_is_rgb_item(img) for img in images]
        else:
            arr = to_numpy(data)
            if arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] > 4:
                images, rgb_flags = [arr], [True]
            else:
                images, rgb_flags = None, []
                data = arr
        if images is not None and any(rgb_flags):
            stack_panels = [
                idx for idx, (img, is_rgb) in enumerate(zip(images, rgb_flags))
                if not is_rgb and img.ndim == 3
            ]
            if stack_panels:
                raise NotImplementedError(
                    "Show2D does not yet mix RGB panels with local grayscale frame "
                    f"stacks (stack panel indices: {stack_panels}). Convert the RGB "
                    "panel to grayscale or use a separate Show2D."
                )
            # Mixed gallery: normalize RGB items to display-ready [0, 1] and
            # reduce them to Rec. 709 luminance for the grayscale machinery
            # (stats, histogram, FFT, profile all read the luminance plane).
            normalized = [_normalize_rgb(img) if flag else np.asarray(img, dtype=np.float32)
                          for img, flag in zip(images, rgb_flags)]
            shapes = [img.shape[:2] for img in normalized]
            if len(set(shapes)) > 1:
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                normalized = [
                    np.stack([_resize_image(ch, max_h, max_w) for ch in img.transpose(2, 0, 1)], axis=-1)
                    if flag else _resize_image(img, max_h, max_w)
                    for img, flag in zip(normalized, rgb_flags)
                ]
            rgb_frames = [img if flag else None for img, flag in zip(normalized, rgb_flags)]
            data = np.stack([img @ _RGB_LUMA if flag else img
                             for img, flag in zip(normalized, rgb_flags)])
        elif images is not None:
            panel_stacks, resolved_panel_frame_indices, data = _normalize_grayscale_panel_items(
                images,
                panel_frame_indices,
            )

        # Ensure 3D shape (N, H, W)
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # Avoid redundant copy: np.asarray is a no-op when already float32 + contiguous.
        # Folder export can preserve uint8 stress fixtures because the browser
        # receives explicit external bytes and does not need an intermediate
        # float32 trait payload.
        if preserve_input_dtype_for_export and data.dtype == np.uint8:
            self._data = np.asarray(data)
        elif data.dtype == np.float32:
            self._data = np.array(data, dtype=np.float32, copy=True)
        else:
            self._data = np.asarray(data, dtype=np.float32)
        self._rgb_frames = rgb_frames or [None] * int(data.shape[0])
        self.is_rgb = [f is not None for f in self._rgb_frames]
        if panel_stacks is None:
            panel_stacks = [self._data[i:i + 1] for i in range(int(self._data.shape[0]))]
            if panel_frame_indices is None:
                resolved_panel_frame_indices = [0] * len(panel_stacks)
            else:
                if len(panel_frame_indices) != len(panel_stacks):
                    raise ValueError(
                        "panel_frame_indices length "
                        f"({len(panel_frame_indices)}) must match panel count ({len(panel_stacks)})"
                    )
                resolved_panel_frame_indices = []
                for panel, raw_index in enumerate(panel_frame_indices):
                    index = int(raw_index)
                    if index == -1:
                        index = 0
                    if index != 0:
                        raise ValueError(
                            f"panel_frame_indices[{panel}]={raw_index} is outside the valid range [0, 1)"
                        )
                    resolved_panel_frame_indices.append(0)
        # overlay sugar (mirrors diff_mode): compose an alignment overlay panel
        # from the two grayscale inputs so users never hand-build (H, W, 3).
        overlay_label = None
        if overlay:
            if any(stack.shape[0] > 1 for stack in panel_stacks):
                raise NotImplementedError(
                    "overlay= is not supported with local frame-stack panels because "
                    "the composed overlay would not follow frame changes"
                )
            if overlay is not True and str(overlay) not in ("green-magenta", "rgb"):
                raise ValueError(f"overlay must be True, 'green-magenta', or 'rgb', got {overlay!r}")
            if self._data.shape[0] != 2 or any(self.is_rgb):
                raise ValueError(
                    f"overlay requires exactly 2 grayscale images, got "
                    f"{self._data.shape[0]} image(s)"
                    + (" including RGB panels" if any(self.is_rgb) else "")
                )
            mode = "rgb" if str(overlay) == "rgb" else "green-magenta"
            composed = _compose_overlay_pair(self._data[0], self._data[1], mode)
            # gallery becomes [a, b, overlay]; the overlay's luminance plane
            # joins _data so stats/histogram/FFT machinery sees three panels
            self._data = np.concatenate([self._data, (composed @ _RGB_LUMA)[None]], axis=0)
            self._rgb_frames = [None, None, composed]
            self.is_rgb = [False, False, True]
            panel_stacks = [self._data[i:i + 1] for i in range(int(self._data.shape[0]))]
            resolved_panel_frame_indices = [0] * len(panel_stacks)
            overlay_label = f"overlay ({mode})"
            data = self._data  # n_images/height/width below read `data`
        # underlay sugar: chemistry-on-structure. Blend the element map onto
        # the HAADF lattice as a third RGB panel so bright columns render in
        # the map color, never as white HAADF cores.
        if underlay:
            if overlay:
                raise ValueError("Use either overlay or underlay, not both")
            if str(underlay).lower() not in ("true", "1", "haadf"):
                raise ValueError(f"underlay must be True or 'haadf', got {underlay!r}")
            mode = str(underlay_mode).strip().lower()
            if mode not in ("haadf", "dual"):
                raise ValueError(
                    f"underlay_mode must be 'haadf' or 'dual', got {underlay_mode!r}"
                )
            # Both modes need exactly two single-frame grayscale inputs: haadf
            # mode reads them as (haadf, map), dual mode as (map A, map B).
            inputs_desc = "(map A, map B)" if mode == "dual" else "(haadf, map)"
            if self._data.shape[0] != 2 or any(self.is_rgb):
                raise ValueError(
                    f"underlay requires exactly 2 grayscale images {inputs_desc}; got "
                    f"{self._data.shape[0]} image(s)"
                    + (" including RGB panels" if any(self.is_rgb) else "")
                )
            if any(stack.shape[0] > 1 for stack in panel_stacks):
                raise NotImplementedError(
                    "underlay does not support per-panel frame stacks; pass single frames"
                )
            self.underlay = True
            self.underlay_mode = mode
            self.underlay_alpha = float(underlay_alpha)
            self.underlay_haadf_gain = float(underlay_haadf_gain)
            self.stretch_percentiles = [float(stretch_percentiles[0]), float(stretch_percentiles[1])]
            self.display_gamma = float(display_gamma)
            self.dual_gain = [float(dual_gain[0]), float(dual_gain[1])]
            self.cmap = str(cmap)
            self._underlay_haadf_idx = 0
            self._underlay_map_idx = 1
            # Raw blend now; the display-filtered blend is recomputed in
            # _update_all_frames once the filter knobs are set below.
            composed = self._compute_underlay_blend()
            self._data = np.concatenate([self._data, (composed @ _RGB_LUMA)[None]], axis=0)
            self._rgb_frames = [None, None, composed]
            self.is_rgb = [False, False, True]
            panel_stacks = [self._data[i:i + 1] for i in range(int(self._data.shape[0]))]
            resolved_panel_frame_indices = [0] * len(panel_stacks)
            overlay_label = "dual composite" if mode == "dual" else "map on HAADF"
            data = self._data
        if offline and any(self.is_rgb):
            raise NotImplementedError(
                "offline=True is not supported for RGB panels; RGB frames are "
                "sent as full float32 and bypass the uint8 quantization path."
            )
        # Store originals for rotation reset: views into _data (no copy).
        # Only materialized as independent copies when a rotation is applied.
        self._data_original = [self._data[i] for i in range(self._data.shape[0])]
        self._originals_are_views = True
        self.n_images = int(data.shape[0])
        self._panel_stacks = panel_stacks
        self._panel_stacks_original = [stack for stack in panel_stacks]
        self._panel_stack_originals_are_views = True
        self._updating_panel_frames = True
        try:
            self.panel_frame_counts = [int(stack.shape[0]) for stack in panel_stacks]
            self.panel_frame_indices = list(resolved_panel_frame_indices or [0] * self.n_images)
        finally:
            self._updating_panel_frames = False
        self.panel_stack_offsets = [-1] * self.n_images
        self.inset_plots = _normalize_inset_plot_specs(inset_plots, n_items=self.n_images)
        self.show_inset_plots = bool(show_inset_plots)
        self.panel_annotations = list(panel_annotations or [])
        self.panel_overlays = list(panel_overlays or [])
        self.height = int(data.shape[1])
        self.width = int(data.shape[2])
        self.rotation_scope = str(rotation_scope).lower()
        self.image_rotations = _normalize_rotation_list(
            n_items=self.n_images,
            rotation=rotation,
            rotations=rotations,
        )
        if self.n_pages > 1:
            if self.panels_per_page <= 0:
                raise ValueError("panels_per_page must be > 0 when n_pages > 1")
            if self.n_images != self.n_pages * self.panels_per_page:
                raise ValueError(
                    f"paged Show2D expects n_images == n_pages * panels_per_page, "
                    f"got {self.n_images} != {self.n_pages} * {self.panels_per_page}"
                )
            if not self.page_labels:
                self.page_labels = [f"Page {i + 1}" for i in range(self.n_pages)]
            if len(self.page_labels) != self.n_pages:
                raise ValueError(
                    f"page_labels length ({len(self.page_labels)}) must equal n_pages ({self.n_pages})"
                )
            if not self.page_starred:
                self.page_starred = [0] * self.n_pages

        # Labels
        if labels is None:
            self.labels = [f"Image {i+1}" for i in range(self.n_images)]
            if overlay_label:
                self.labels = self.labels[:-1] + [overlay_label]
        else:
            resolved_labels = list(labels)
            # user labeled the two inputs; the composed overlay names itself
            if overlay_label and len(resolved_labels) == self.n_images - 1:
                resolved_labels.append(overlay_label)
            self.labels = resolved_labels
        if panel_title_spans and len(panel_title_spans) in (self.n_images - 1, self.n_images):
            resolved_spans = list(panel_title_spans)
            if overlay_label and len(resolved_spans) == self.n_images - 1:
                resolved_spans.append([])
            self.panel_title_spans = resolved_spans
        else:
            self.panel_title_spans = []
        self.starred = [0] * self.n_images
        self.hidden_panels = []
        self.hidden_page_slots = []
        self.show_panel_titles = bool(show_panel_titles)
        self.panel_title_font_size = int(panel_title_font_size)
        self.panel_title_style = dict(panel_title_style or {})
        self.inter_panel_gap_px = int(inter_panel_gap_px)
        self.inter_panel_gap_color = "" if inter_panel_gap_color is None else str(inter_panel_gap_color)
        self.gallery_outer_border_px = int(gallery_outer_border_px)
        self.gallery_outer_border_color = "" if gallery_outer_border_color is None else str(gallery_outer_border_color)
        self.panel_inner_border_px = float(panel_inner_border_px)
        self.panel_inner_border_color = (
            "" if panel_inner_border_color is None else str(panel_inner_border_color)
        )
        self.gallery_gap_px = int(gallery_gap_px)
        self.gallery_gap_color = "" if gallery_gap_color is None else str(gallery_gap_color)
        if starred is not None:
            self.set_starred_panels(starred)
        if hidden_panels is not None:
            self.set_hidden_panels(hidden_panels)
        if panel_order is not None:
            self.set_panel_order(panel_order)

        # Options
        self.title = title
        self.show_title = bool(show_title)
        self.cmap = str(cmap)
        if panel_cmaps:
            cmaps = [str(item) for item in panel_cmaps]
            if len(cmaps) == 1:
                cmaps = cmaps * self.n_images
            elif len(cmaps) != self.n_images:
                raise ValueError(
                    f"cmap sequence length ({len(cmaps)}) must be 1 or match "
                    f"the number of Show2D panels ({self.n_images})"
                )
            self.panel_cmaps = cmaps
            self.panel_cmaps_memory = list(cmaps)
        else:
            self.panel_cmaps = []
            self.panel_cmaps_memory = []
        # Resolve sampling + units to scalar pixel_size + pixel_unit (column axis).
        # Scalar shorthand: sampling=0.5 → (0.5, 0.5). units="nm" → ["nm", "nm"].
        if sampling is None:
            pass  # keep the trait: pixel_size= may already be set directly via kwargs
        elif isinstance(sampling, (int, float)):
            self.pixel_size = float(sampling)
        else:
            self.pixel_size = float(sampling[-1])
        if units is None:
            pass  # keep the trait: pixel_unit= may already be set directly via kwargs
        elif isinstance(units, str):
            self.pixel_unit = units
        else:
            self.pixel_unit = str(units[-1])
        self.scale_bar_visible = scale_bar_visible
        self.scale_bar_position = scale_bar_position
        self.scale_bar_panels = list(scale_bar_panels or [])
        self.scale_bar_length = None if scale_bar_length is None else float(scale_bar_length)
        self.scale_bar_label = "" if scale_bar_label is None else str(scale_bar_label)
        self.scale_bar_style = dict(scale_bar_style or {})
        self.show_zoom_indicator = bool(show_zoom_indicator)
        self.pixel_sizes = []
        self.size = size
        self.smooth = smooth
        self.contrast_preset = str(contrast_preset)
        advanced_histogram = bool(histogram_advanced if show_histogram_advanced is None else show_histogram_advanced)
        self.histogram_advanced = advanced_histogram
        self.show_histogram_advanced = advanced_histogram
        marker_source = marker_colors if marker_colors is not None else identity_colors
        if marker_source is None:
            colors = []
        else:
            colors = [str(value) for value in marker_source]
            if colors and len(colors) != self.n_images:
                raise ValueError(
                    f"marker_colors length ({len(colors)}) must match "
                    f"the number of Show2D panels ({self.n_images})"
                )
        self.identity_colors = colors
        self.marker_colors = list(colors)
        self.marker_style = str(marker_style).lower()
        self.row_markers = dict(row_markers or {})
        self.col_markers = dict(col_markers or {})
        self.selected_panels = []
        # view_box sugar: sets zoom + zoom_row/col to center on box.
        # Two forms: (r0, r1, c0, c1) explicit bounds, or the friendlier
        # (row0, col0, size) = top-left corner + square size.
        if view_box is not None:
            vb = [float(v) for v in view_box]
            if len(vb) == 3:
                vb = [vb[0], vb[0] + vb[2], vb[1], vb[1] + vb[2]]
            r0, r1, c0, c1 = vb
            box_h = max(1.0, r1 - r0)
            box_w = max(1.0, c1 - c0)
            zoom = float(min(self.height / box_h, self.width / box_w))
            zoom_row = (r0 + r1) / 2
            zoom_col = (c0 + c1) / 2
            self.view_box = [r0, r1, c0, c1]
        self.initial_zoom = zoom
        self.flip_rows = bool(flip_rows)
        self.flip_cols = bool(flip_cols)
        horizontal_flips = (
            [bool(flip_cols)] * int(self.n_images)
            if image_flips_horizontal is None
            else [bool(value) for value in image_flips_horizontal]
        )
        vertical_flips = (
            [bool(flip_rows)] * int(self.n_images)
            if image_flips_vertical is None
            else [bool(value) for value in image_flips_vertical]
        )
        if len(horizontal_flips) != int(self.n_images) or len(vertical_flips) != int(self.n_images):
            raise ValueError(
                "image_flips_horizontal and image_flips_vertical must match "
                f"the number of Show2D panels ({int(self.n_images)})"
            )
        self.image_flips_horizontal = horizontal_flips
        self.image_flips_vertical = vertical_flips
        self.zoom_row = zoom_row
        self.zoom_col = zoom_col
        # Auto-link zoom + pan in gallery (n_images >= 2) so dragging one panel
        # follows the other — typical compare/diff workflow. Single image: no-op.
        self.link_zoom = (self.n_images >= 2) if link_zoom is None else link_zoom
        self.link_pan = (self.n_images >= 2) if link_pan is None else link_pan
        self.link_contrast = link_contrast
        self.diff_mode = diff_mode if self.n_images >= 2 else False
        if show_fft and self.height * self.width > 2048 * 2048:
            warnings.warn(
                f"FFT on {self.height}×{self.width} image ({self.height * self.width / 1e6:.1f}M pixels) "
                f"may be slow. Consider using ROI FFT for a sub-region.",
                stacklevel=2,
            )
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.fft_metrics = bool(fft_metrics)
        self.show_controls = show_controls
        self.controls_collapsed = bool(controls_collapsed)
        self.show_stats = show_stats
        self.debug = bool(debug)
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.offline = offline
        # Standalone HTML export packs panels through the grayscale stack
        # machinery, which cannot carry (H, W, 3) pixels: hide the export menu
        # and reject programmatic export instead of corrupting RGB panels.
        if any(self.is_rgb):
            self.export_enabled = False
        # Accept scalar OR list for vmin/vmax. List → per-image (vmins/vmaxs).
        if isinstance(vmin, (list, tuple)) or isinstance(vmax, (list, tuple)):
            n = self.n_images
            def _expand(v):
                if v is None:
                    return [None] * n
                if isinstance(v, (list, tuple)):
                    if len(v) != n:
                        raise ValueError(f"vmin/vmax list has length {len(v)} but n_images is {n}. Pass a list of length {n} or a scalar to apply uniformly.")
                    return [None if x is None else float(x) for x in v]
                return [float(v)] * n
            self.vmins = _expand(vmin)
            self.vmaxs = _expand(vmax)
            self.vmin = None
            self.vmax = None
        else:
            self.vmin = vmin
            self.vmax = vmax
        self.ncols = ncols
        self.panel_playback_fps = panel_playback_fps

        # Auto-bin for display: keep full-res in _data, send binned to JS.
        # GPU memory budget: ~2 GB for display buffers (128 MB per image at 4K).
        # At 4K: max ~16 full-res. Beyond that, auto-downsample.
        if display_bin == "auto":
            # Each 4K image needs ~192 MB GPU buffers (float32 + RGBA + read)
            # Tested: 12×4K (2.3 GB) works, 24×4K (4.6 GB) OOMs
            # Budget: 2.5 GB allows 12×4K full-res, bins above that
            gpu_budget_mb = self._GPU_DISPLAY_BUDGET_MB
            per_image_mb = (self.height * self.width * 4 * 3) / (1024 * 1024)  # 3 buffers
            total_mb = self.n_images * per_image_mb
            if total_mb > gpu_budget_mb:
                # Find minimum bin factor to fit
                for bf in [2, 4, 8]:
                    binned_mb = self.n_images * per_image_mb / (bf * bf)
                    if binned_mb <= gpu_budget_mb:
                        self._display_bin = bf
                        break
                else:
                    self._display_bin = 8
                # no surprise binning: announce the reduction and the way out
                print(f"Show2D display auto-binned {self._display_bin}x to fit the GPU display budget "
                      f"({total_mb:.0f} MB > {gpu_budget_mb} MB); pass display_bin=1 for native pixels")
            # Wire budget (maps-style first paint): a panel whose float32 bytes
            # exceed the per-panel budget would stall first paint for seconds on
            # a remote Jupyter channel, while the screen canvas can only display
            # a tiny fraction of those pixels at initial zoom. Send a preview
            # binned so the longest side is ~2x the canvas CSS size; zooming
            # streams full-res crops of the visible window via _detail_request,
            # so full fidelity is preserved without ever shipping the full frame.
            # Large galleries need a lower preview floor: 36x1024x1024 float32
            # is still 144 MB before comm/base64/browser work, while the
            # default 200 px panel can only display a fraction of those pixels.
            panel_bytes = self.height * self.width * 4
            if any(self.is_rgb):
                panel_bytes *= 3  # RGB panels carry 3 interleaved float planes
            if panel_bytes > self._WIRE_BUDGET_BYTES_PER_PANEL:
                preview_floor_px = 512.0 if self.n_images >= 16 else 1024.0
                preview_px = max(preview_floor_px, 2.0 * self._static_canvas_css_px())
                wire_bin = max(2, math.ceil(max(self.height, self.width) / preview_px))
                while panel_bytes / (wire_bin * wire_bin) > self._WIRE_BUDGET_BYTES_PER_PANEL:
                    wire_bin += 1
                self._display_bin = max(self._display_bin, wire_bin)
        elif isinstance(display_bin, int) and display_bin > 1:
            self._display_bin = display_bin

        if self._display_bin > 1:
            from quantem.widget.utils.array import bin2d
            orig_h, orig_w = self._data.shape[1], self._data.shape[2]
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
            if self.pixel_size > 0:
                self.pixel_size = self.pixel_size * self._display_bin
            # User-facing view coordinates (center=, view_box=) are full-res
            # pixels, but JS interprets the traits in preview pixels once the
            # display is binned. Rescale so a zoom target lands on the same
            # physical feature at any bin factor.
            if self.zoom_row is not None:
                self.zoom_row = self.zoom_row / self._display_bin
            if self.zoom_col is not None:
                self.zoom_col = self.zoom_col / self._display_bin
            if self.view_box:
                self.view_box = [v / self._display_bin for v in self.view_box]
            self._display_bin_factor = self._display_bin
            # RGB panels bin per channel so the display copy matches the
            # luminance plane's resolution (JS derives offsets from one H×W).
            self._display_rgb = [
                None if f is None else
                bin2d(f.transpose(2, 0, 1), factor=self._display_bin, mode="mean").transpose(1, 2, 0)
                for f in self._rgb_frames
            ]
            self._display_panel_stacks = [
                bin2d(stack, factor=self._display_bin, mode="mean")
                for stack in self._panel_stacks
            ]
            if verbose:
                print(f"  Display bin {self._display_bin}×: {orig_h}×{orig_w} → {self.height}×{self.width} ({self._display_data.nbytes // 1024 // 1024} MB preview; full-res detail streams on zoom)")
        else:
            self._display_data = self._data
            self._display_bin_factor = 1
            self._display_rgb = self._rgb_frames
            self._display_panel_stacks = self._panel_stacks

        # Display-only filter knobs (view transforms; the stored raw data and
        # the stats row stay untouched). Scalars broadcast to every panel; a
        # sequence gives each panel its own filter/sigma/bin (e.g. a raw vs
        # filtered A/B gallery). Set before the first frame pack so a
        # constructor-selected filter shows from the first paint.
        self._display_filter_ready = False
        self._filter_knob_sync = False
        n_panels = int(self._data.shape[0])

        def per_panel(value, kind, cast):
            if isinstance(value, (list, tuple, np.ndarray)):
                values = [cast(v) for v in value]
                if len(values) != n_panels:
                    raise ValueError(
                        f"{kind} sequence length ({len(values)}) must equal the "
                        f"panel count ({n_panels})"
                    )
                return values, False
            return [cast(value)] * n_panels, True

        filters, filters_scalar = per_panel(denoise, "denoise", str)
        sigmas, sigmas_scalar = per_panel(denoise_sigma, "denoise_sigma", float)
        bins, bins_scalar = per_panel(denoise_bin, "denoise_bin", int)
        pad_ratios, pad_ratio_scalar = per_panel(pad_ratio, "pad_ratio", float)
        pad_modes, pad_mode_scalar = per_panel(pad_fill_mode, "pad_fill_mode", str)
        invalid_pad_ratio = next((value for value in pad_ratios if not 0.0 <= value <= 1.0), None)
        if invalid_pad_ratio is not None:
            raise ValueError(f"pad_ratio must be between 0 and 1; got {invalid_pad_ratio}")
        pad_modes = [mode.strip().lower() for mode in pad_modes]
        invalid_pad_mode = next((mode for mode in pad_modes if mode not in {"min", "median", "mean"}), None)
        if invalid_pad_mode is not None:
            raise ValueError(
                "pad_fill_mode must be one of 'min', 'median', or 'mean'; "
                f"got {invalid_pad_mode!r}"
            )
        if any(pad_ratios) and any(self.is_rgb):
            raise NotImplementedError("pad_ratio supports grayscale panels; RGB panels are not padded.")
        # Compound spellings (bin2, bin2_anscombe, bin4_anscombe) are aliases
        # for (mode, bin); the traits always hold the canonical trio.
        from quantem.widget.utils.display_filter import resolve_denoise_mode

        resolved = [resolve_denoise_mode(m, b) for m, b in zip(filters, bins)]
        filters = [m for m, _ in resolved]
        bins = [b for _, b in resolved]
        self.denoise_modes = filters
        self.denoise_sigmas = sigmas
        self.denoise_bins = bins
        # Scalar traits are the UI editor, mirroring the selected panel.
        self._filter_knob_sync = True
        self.denoise = self.denoise_modes[0]
        self.denoise_sigma = float(sigmas[0])
        self.denoise_bin = int(bins[0])
        self._filter_knob_sync = False
        # A sequence means independent per-panel knobs, so UI edits scope to
        # the selected panel; scalars keep the broadcast "all" scope. Passing a
        # per-panel sequence AND an explicit denoise_scope="all" is a
        # contradiction ("all" broadcasts one value, the sequence gives each
        # panel its own), so reject it instead of silently forcing "panel".
        scalar_knobs = filters_scalar and sigmas_scalar and bins_scalar
        if denoise_scope_explicit and denoise_scope == "all" and not scalar_knobs:
            raise ValueError(
                "denoise_scope='all' broadcasts one setting to every panel, but "
                "a per-panel denoise/denoise_sigma/denoise_bin sequence was also "
                "given. Pass scalar denoise knobs with denoise_scope='all', or "
                "drop denoise_scope and let the sequence select per-panel scope."
            )
        broadcast = denoise_scope != "panel" and scalar_knobs
        self.denoise_scope = "all" if broadcast else "panel"
        self._display_filter_ready = True
        self._refresh_display_filter_banner(announce=True)
        # The master denoise switch starts ON iff the widget was built with an
        # active denoise config; a clean widget starts OFF (the toggle enables it).
        self.denoise_enabled = self._has_denoise_config()
        # Denoise controls stay hidden on a clean widget; an active denoise
        # (or an explicit request) reveals them from the first paint.
        self.show_denoise = bool(show_denoise) or self._display_filter_active()
        frequency_modes, _ = per_panel(frequency_filter, "frequency_filter", str)
        frequency_cutoffs, _ = per_panel(
            frequency_filter_cutoff, "frequency_filter_cutoff", float
        )
        frequency_centers, _ = per_panel(
            frequency_filter_center, "frequency_filter_center", float
        )
        frequency_widths, _ = per_panel(
            frequency_filter_width, "frequency_filter_width", float
        )
        frequency_modes = [mode.strip().lower().replace("-", "") for mode in frequency_modes]
        invalid_modes = [mode for mode in frequency_modes if mode not in {"none", "lowpass", "highpass", "bandpass"}]
        if invalid_modes:
            raise ValueError(
                "frequency_filter must contain only 'none', 'lowpass', "
                f"'highpass', or 'bandpass'; got {invalid_modes[0]!r}"
            )
        for name, values in {
            "frequency_filter_cutoff": frequency_cutoffs,
            "frequency_filter_center": frequency_centers,
            "frequency_filter_width": frequency_widths,
        }.items():
            invalid = next((value for value in values if not 0.0 <= value <= 1.0), None)
            if invalid is not None:
                raise ValueError(f"{name} must be between 0 and 1 (Nyquist); got {invalid}")
        self.frequency_filter_modes = frequency_modes
        self.frequency_filter_cutoffs = frequency_cutoffs
        self.frequency_filter_centers = frequency_centers
        self.frequency_filter_widths = frequency_widths
        self.frequency_filter = frequency_modes[0]
        self.frequency_filter_enabled = (
            any(mode != "none" for mode in frequency_modes)
            if frequency_filter_enabled is None
            else bool(frequency_filter_enabled)
        )
        self.frequency_filter_cutoff = frequency_cutoffs[0]
        self.frequency_filter_center = frequency_centers[0]
        self.frequency_filter_width = frequency_widths[0]
        # A scientist comparing multiple panels normally adjusts one result at
        # a time. Linking remains available in the UI, but it must be explicit:
        # scalar constructor values are broadcast as initial settings, not as
        # permission for later edits to modify every panel.
        self.frequency_filter_scope = "panel" if self.n_images > 1 else "all"
        self.show_frequency_filter = bool(show_frequency_filter)

        # Reversible view ops: a crop is committed later via crop_to_view(),
        # but the pad kwarg can start active. Geometry (frame extent, cursor
        # offset, banner) is synced before the first frame pack; the observer
        # stays inert until _view_ops_ready so the constructor zoom survives.
        self._view_ops_ready = False
        self._pad_knob_sync = False
        self.pad_ratios = pad_ratios
        self.pad_fill_modes = pad_modes
        self.pad_ratio = float(pad_ratios[0])
        self.pad_fill_mode = pad_modes[0]
        self.pad_scope = "all" if pad_ratio_scalar and pad_mode_scalar and str(pad_scope).lower() != "panel" else "panel"
        self._refresh_view_ops(announce=True)
        self._view_ops_ready = True

        # Compute initial stats (from full-res data)
        if skip_initial_stats:
            axes = (1, 2) if self._data.ndim == 3 else None
            self.stats_mean = np.mean(self._data, axis=axes).ravel().tolist()
            self.stats_min = np.min(self._data, axis=axes).ravel().tolist()
            self.stats_max = np.max(self._data, axis=axes).ravel().tolist()
            self.stats_std = np.std(self._data, axis=axes).ravel().tolist()
        else:
            self._compute_all_stats()

        # Send display data to JS (possibly binned)
        if skip_initial_frame_pack:
            self.frame_bytes = b""
            self.panel_stack_bytes = b""
        else:
            self._update_all_frames()

        self.selected_idx = 0

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

        # Stash wall-clock start on the instance; the observer below prints the
        # TRUE end-to-end time after JS signals first paint.  The Python-only
        # __init__ number is misleading for widget UX: a widget is not "done"
        # until the browser has painted its first frame.
        self._init_t0 = _t0
        self._init_py_elapsed_ms = (_time.perf_counter() - _t0) * 1000
        self.observe(self._on_first_render, names=["_js_rendered"])
        self.observe(self._on_export_request_change, names=["export_request"])
        self.observe(self._on_saved_view_request_change, names=["saved_view_request"])
        self.observe(self._on_handoff_request_change, names=["handoff_request"])
        self.observe(self._on_detail_request_change, names=["_detail_request"])

    @traitlets.validate("starred")
    def _validate_starred(self, proposal: dict) -> list[int]:
        """Normalize per-image star flags."""
        val = list(proposal["value"])
        n_img = int(self.n_images)
        if not val:
            return [0] * n_img
        if len(val) != n_img:
            raise traitlets.TraitError(
                f"starred length ({len(val)}) must equal n_images ({n_img})"
            )
        return [1 if int(v) else 0 for v in val]

    @traitlets.validate("panel_frame_indices")
    def _validate_panel_frame_indices(self, proposal: dict) -> list[int]:
        """Validate one independent frame index for every source panel."""
        values = list(proposal["value"])
        counts = list(getattr(self, "panel_frame_counts", []))
        n_images = int(getattr(self, "n_images", 0))
        if not values and n_images > 0:
            return [0] * n_images
        if len(values) != n_images:
            raise traitlets.TraitError(
                f"panel_frame_indices length ({len(values)}) must equal n_images ({n_images})"
            )
        if len(counts) != n_images:
            counts = [1] * n_images
        normalized: list[int] = []
        for panel, (raw_index, count) in enumerate(zip(values, counts)):
            index = int(raw_index)
            count = max(1, int(count))
            if index < 0 or index >= count:
                raise traitlets.TraitError(
                    f"panel_frame_indices[{panel}]={index} is outside the valid range [0, {count})"
                )
            normalized.append(index)
        return normalized

    @traitlets.validate("panel_playback_fps")
    def _validate_panel_playback_fps(self, proposal: dict) -> float:
        """Reject invalid local-stack playback rates and cap browser work."""
        value = float(proposal["value"])
        if not math.isfinite(value):
            raise traitlets.TraitError(
                f"panel_playback_fps must be finite, got {value}"
            )
        if value <= 0:
            raise traitlets.TraitError(
                f"panel_playback_fps must be > 0, got {value}"
            )
        return min(value, _MAX_PANEL_PLAYBACK_FPS)

    @traitlets.observe("panel_frame_indices")
    def _on_panel_frame_indices_changed(self, change) -> None:
        """Keep Python analysis/detail state aligned with browser-local scrubbing."""
        if getattr(self, "_updating_panel_frames", False):
            return
        stacks = getattr(self, "_panel_stacks", None)
        if not stacks or len(stacks) != int(getattr(self, "n_images", 0)):
            return
        indices = list(change.get("new") or [])
        if len(indices) != len(stacks):
            return
        self._data = np.stack(
            [stack[index] for stack, index in zip(stacks, indices)],
            axis=0,
        ).astype(np.float32, copy=False)
        display_stacks = getattr(self, "_display_panel_stacks", None)
        if display_stacks and len(display_stacks) == len(stacks):
            self._display_data = np.stack(
                [stack[index] for stack, index in zip(display_stacks, indices)],
                axis=0,
            ).astype(np.float32, copy=False)
        elif int(getattr(self, "_display_bin", 1)) <= 1:
            self._display_data = self._data
        if hasattr(self, "stats_mean"):
            self._compute_all_stats()
        if hasattr(self, "_detail_meta"):
            self._detail_request = ""
            self._detail_meta = ""
            self._detail_bytes = b""

    @traitlets.validate("page_idx")
    def _validate_page_idx(self, proposal: dict) -> int:
        """Clamp the active page index to the available page range."""
        n_pages = max(1, int(getattr(self, "n_pages", 1)))
        return int(max(0, min(int(proposal["value"]), n_pages - 1)))

    @traitlets.validate("page_starred")
    def _validate_page_starred(self, proposal: dict) -> list[int]:
        """Normalize per-page star flags."""
        val = list(proposal["value"])
        n_pages = max(1, int(getattr(self, "n_pages", 1)))
        if not val:
            return [0] * n_pages
        if len(val) != n_pages:
            raise traitlets.TraitError(
                f"page_starred length ({len(val)}) must equal n_pages ({n_pages})"
            )
        return [1 if int(v) else 0 for v in val]

    @traitlets.validate("hidden_panels")
    def _validate_hidden_panels(self, proposal: dict) -> list[int]:
        """Normalize hidden image indices and keep at least one image visible."""
        n_img = int(self.n_images)
        if n_img <= 0:
            return []
        clean: list[int] = []
        seen: set[int] = set()
        for value in proposal["value"]:
            idx = int(value)
            if 0 <= idx < n_img and idx not in seen:
                clean.append(idx)
                seen.add(idx)
        clean.sort()
        if len(clean) >= n_img:
            raise traitlets.TraitError(
                "hidden_panels cannot hide every panel; at least one panel must remain visible"
            )
        return self._normalize_item_page_hidden(clean)

    @traitlets.validate("selected_panels")
    def _validate_selected_panels(self, proposal: dict) -> list[int]:
        """Normalize the UI multi-panel selection to existing panel indices."""
        n_img = int(getattr(self, "n_images", 0))
        clean: list[int] = []
        seen: set[int] = set()
        for value in proposal["value"]:
            if isinstance(value, bool):
                continue
            idx = int(value)
            if 0 <= idx < n_img and idx not in seen:
                clean.append(idx)
                seen.add(idx)
        return clean

    def _normalize_hidden_page_slots(
        self,
        values: Sequence[object],
        *,
        drop_if_full: bool = False,
    ) -> list[int]:
        """Normalize reusable hidden panel slots for paged galleries."""
        if (
            int(self.n_pages) <= 1
            or int(self.panels_per_page) <= 0
            or str(self.page_kind) == "items"
        ):
            return []
        n_slots = int(self.panels_per_page)
        clean_set: set[int] = set()
        for value in values or []:
            if isinstance(value, bool):
                continue
            try:
                slot = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= slot < n_slots:
                clean_set.add(slot)
        clean = sorted(clean_set)
        if len(clean) >= n_slots:
            if drop_if_full:
                clean = clean[:-1]
            else:
                raise traitlets.TraitError(
                    "hidden_page_slots cannot hide every page slot; at least one panel must remain visible"
                )
        return clean

    def _hidden_page_slots_from_panels(
        self,
        panels: Sequence[int],
        *,
        drop_if_full: bool = False,
    ) -> list[int]:
        """Map absolute hidden image indices to reusable page slots."""
        if (
            int(self.n_pages) <= 1
            or int(self.panels_per_page) <= 0
            or str(self.page_kind) == "items"
        ):
            return []
        n_img = int(self.n_images)
        per_page = int(self.panels_per_page)
        slots = [
            int(panel) % per_page
            for panel in panels
            if 0 <= int(panel) < n_img
        ]
        return self._normalize_hidden_page_slots(slots, drop_if_full=drop_if_full)

    @traitlets.validate("hidden_page_slots")
    def _validate_hidden_page_slots(self, proposal: dict) -> list[int]:
        """Normalize hidden page slots for paged galleries."""
        return self._normalize_hidden_page_slots(proposal["value"])

    @traitlets.validate("panel_order")
    def _validate_panel_order(self, proposal: dict) -> list[int]:
        """Normalize optional display order over source image indices."""
        n_img = int(self.n_images)
        values = list(proposal["value"] or [])
        if not values:
            return []
        clean: list[int] = []
        try:
            for value in values:
                if isinstance(value, bool):
                    raise TypeError
                clean.append(int(value))
        except (TypeError, ValueError) as exc:
            raise traitlets.TraitError("panel_order must contain integer panel indices") from exc
        expected = list(range(n_img))
        if len(clean) != n_img or sorted(clean) != expected:
            raise traitlets.TraitError(
                "panel_order must include every panel index exactly once "
                f"(expected a permutation of {expected!r})"
            )
        return clean

    def _panel_title_for_index(self, panel: int) -> str:
        """Return the user-facing label for a Show2D image panel."""
        if 0 <= panel < len(self.labels) and self.labels[panel]:
            return str(self.labels[panel])
        return f"Image {panel + 1}"

    def _resolve_panel_ref(self, panel: int | str) -> int:
        """Resolve a panel index or exact label into a zero-based image index."""
        if isinstance(panel, bool):
            raise TypeError("panel must be an integer index or exact label, not bool")
        if isinstance(panel, int):
            idx = int(panel)
            if 0 <= idx < int(self.n_images):
                return idx
            raise ValueError(f"panel index {idx} out of range [0, {self.n_images})")
        if isinstance(panel, str):
            titles = [self._panel_title_for_index(i) for i in range(int(self.n_images))]
            matches = [i for i, title in enumerate(titles) if title == panel]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"panel label {panel!r} is not unique; use a zero-based panel index instead"
                )
            available = ", ".join(repr(title) for title in titles)
            raise ValueError(f"unknown panel label {panel!r}; available labels: {available}")
        raise TypeError(
            f"panel must be an integer index or exact label, got {type(panel).__name__}"
        )

    def _normalize_panel_refs(
        self,
        panels: Sequence[int | str] | int | str,
        *,
        allow_empty: bool = False,
    ) -> list[int]:
        """Resolve and de-duplicate image panel references."""
        if isinstance(panels, (str, int)) and not isinstance(panels, bool):
            values: Sequence[int | str] = [panels]
        else:
            values = panels  # type: ignore[assignment]
        out: list[int] = []
        seen: set[int] = set()
        for panel in values:
            idx = self._resolve_panel_ref(panel)
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
        if not out and not allow_empty:
            raise ValueError("at least one panel index or label is required")
        return out

    def _on_first_render(self, change):
        import time as _time
        from quantem.widget._timing import format_widget_render_timing
        if not change.get("new"):
            return
        total_ms = (_time.perf_counter() - self._init_t0) * 1000
        py_ms = self._init_py_elapsed_ms
        shape = (
            (self.n_images, self.height, self.width)
            if self.n_images > 1
            else (self.height, self.width)
        )
        mem = self._data.nbytes
        # Expose as attributes so tests and notebooks can assert on them.
        # These are the ground truth for "did JS actually paint": if they're
        # None, the JS side never signaled first render.
        self.render_total_ms = int(total_ms)
        self.render_python_build_ms = int(py_ms)
        self.render_wire_js_ms = int(total_ms - py_ms)
        if not getattr(self, "_save_state", False) and (self.frame_bytes or self.panel_stack_bytes):
            # The frontend has decoded and painted the pixels by the time it
            # flips ``_js_rendered``. Clear the synced model buffer so a later
            # notebook save stores the static PNG fallback, not the full 4k
            # array, while targeted initial sync remains intact.
            self.frame_bytes = b""
            self.panel_stack_bytes = b""
        if not getattr(self, "_verbose", True):
            pass
        else:
            print(
                format_widget_render_timing(
                    "Show2D",
                    shape=shape,
                    dtype=str(self._data.dtype),
                    raw_bytes=int(mem),
                    total_ms=total_ms,
                    python_ms=py_ms,
                    wire_js_ms=total_ms - py_ms,
                ),
                flush=True,
            )
        # Detach observer: one-shot, we only care about the first paint.
        try:
            self.unobserve(self._on_first_render, names=["_js_rendered"])
        except (ValueError, KeyError):
            pass

    def _folder_ordered_panel_indices(self) -> list[int]:
        """Return the complete path-stable order for a folder item gallery."""
        n_images = int(getattr(self, "n_images", 0))
        order = list(getattr(self, "panel_order", []) or [])
        if (
            len(order) == n_images
            and sorted(order) == list(range(n_images))
        ):
            return order
        return list(range(n_images))

    def _normalize_item_page_hidden(
        self,
        values: Sequence[int],
        *,
        drop_if_full: bool = False,
    ) -> list[int]:
        """Keep at least one concrete item visible on every folder page."""
        hidden = {int(value) for value in values}
        if (
            int(getattr(self, "n_pages", 1)) <= 1
            or int(getattr(self, "panels_per_page", 0)) <= 0
            or str(getattr(self, "page_kind", "comparison")) != "items"
        ):
            return sorted(hidden)
        order = self._folder_ordered_panel_indices()
        page_size = int(self.panels_per_page)
        for page_index, start in enumerate(range(0, len(order), page_size)):
            page = order[start : start + page_size]
            if page and all(panel in hidden for panel in page):
                if not drop_if_full:
                    raise traitlets.TraitError(
                        "hidden_panels cannot hide every panel on folder page "
                        f"{page_index + 1}; "
                        "leave at least one source image visible"
                    )
                hidden.discard(page[-1])
        return sorted(hidden)

    def _validate_folder_item_pages_visible(
        self,
        hidden: Iterable[int],
        operation: str,
    ) -> None:
        """Reject item-page state that would make any folder page empty."""
        if str(self.page_kind) != "items" or int(self.panels_per_page) <= 0:
            return
        hidden_set = {int(panel) for panel in hidden}
        page_size = int(self.panels_per_page)
        order = self._folder_ordered_panel_indices()
        for page_idx, start in enumerate(range(0, len(order), page_size)):
            panels = order[start : start + page_size]
            if panels and all(panel in hidden_set for panel in panels):
                raise ValueError(
                    f"{operation} would hide every panel on folder page "
                    f"{page_idx + 1}; leave at least one visible per page"
                )

    def _sync_folder_pages(
        self,
        *,
        anchor_panel: int | None = None,
        preferred_page: int | None = None,
    ) -> None:
        """Recompute sequential folder pages without rebuilding the widget."""
        page_size = getattr(self, "_folder_page_size", None)
        n_images = int(getattr(self, "n_images", 0))
        old_page = (
            int(getattr(self, "page_idx", 0))
            if preferred_page is None
            else int(preferred_page)
        )
        old_starred = list(getattr(self, "page_starred", []) or [])

        if page_size is None or n_images <= int(page_size):
            n_pages = 1
            panels_per_page = 0
            page_labels: list[str] = []
            next_page = 0
        else:
            panels_per_page = int(page_size)
            n_pages = int(math.ceil(n_images / panels_per_page))
            page_labels = [
                f"Images {start + 1}\u2013{min(n_images, start + panels_per_page)}"
                for start in range(0, n_images, panels_per_page)
            ]
            next_page = max(0, min(old_page, n_pages - 1))
            if anchor_panel is not None:
                order = self._folder_ordered_panel_indices()
                try:
                    position = order.index(int(anchor_panel))
                except (TypeError, ValueError):
                    pass
                else:
                    next_page = min(n_pages - 1, position // panels_per_page)

        next_starred = [0] * n_pages
        for index in range(min(len(old_starred), n_pages)):
            next_starred[index] = 1 if int(old_starred[index]) else 0

        with self.hold_sync():
            self.page_kind = "items"
            self.n_pages = n_pages
            self.panels_per_page = panels_per_page
            self.page_labels = page_labels
            self.page_starred = next_starred
            self.page_idx = next_page
            # Folder pages hide concrete files, not a repeated comparison slot.
            self.hidden_page_slots = []

    def _set_folder_waiting_empty_state(self) -> None:
        """Represent an acquisition folder with no readable image yet."""
        empty = np.empty((0, 0, 0), dtype=np.float32)
        with self.hold_sync():
            self._data = empty
            self._display_data = empty
            self._data_original = []
            self._panel_stacks = []
            self._panel_stacks_original = []
            self._display_panel_stacks = []
            self._rgb_frames = []
            self._display_rgb = []
            self.n_images = 0
            self.height = 0
            self.width = 0
            self.labels = []
            self.is_rgb = []
            self.starred = []
            self.hidden_panels = []
            self.hidden_page_slots = []
            self.panel_order = []
            self.image_rotations = []
            self.panel_frame_counts = []
            self.panel_frame_indices = []
            self.panel_stack_offsets = []
            self.panel_stack_bytes = b""
            self._panel_stack_mins = []
            self._panel_stack_maxs = []
            self.frame_bytes = b""
            self.stats_mean = []
            self.stats_min = []
            self.stats_max = []
            self.stats_std = []
            self._offline_mins = []
            self._offline_maxs = []
            self._static_fallback_jpeg = ""
            self.selected_idx = 0
            self.n_pages = 1
            self.page_idx = 0
            self.panels_per_page = 0
            self.page_labels = []
            self.page_starred = [0]
            self.folder_waiting = True

    def set_image(
        self,
        data,
        labels: list[str | None] | None = None,
        *,
        panel_frame_indices: Sequence[int] | None = None,
    ) -> None:
        """Replace the displayed image stack without rebuilding the widget.

        This is the light-weight live-update path used by ShowFolder watched
        selections. It preserves display controls such as colormap, contrast,
        FFT/profile toggles, and gallery layout, while resetting per-panel state
        tied to the previous image count or dimensions.
        """
        if isinstance(data, (list, tuple)):
            images = [
                to_numpy(item.array if hasattr(item, "array") else item)
                for item in data
            ]
            rgb_panels = [idx for idx, image in enumerate(images) if _is_rgb_item(image)]
            if rgb_panels:
                raise NotImplementedError(
                    "Show2D.set_image does not yet replace RGB panels; "
                    f"RGB panel indices: {rgb_panels}"
                )
            panel_stacks, resolved_indices, data = _normalize_grayscale_panel_items(
                images,
                panel_frame_indices,
            )
        else:
            if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
                data = data.array
            data = to_numpy(data)
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            if data.ndim != 3:
                raise ValueError(f"Show2D.set_image expects a 2D image or 3D stack, got {data.ndim}D")
            if 0 in data.shape:
                raise ValueError(f"Empty image stack: shape {data.shape}. All dims must be >= 1.")
            if np.iscomplexobj(data):
                raise TypeError(
                    "Show2D does not accept complex data. Convert first: "
                    "np.abs(arr) for magnitude or np.angle(arr) for phase."
                )
            data = np.asarray(data, dtype=np.float32)
            if not np.isfinite(data).all():
                raise ValueError(
                    "Data contains NaN or inf. Clean first: "
                    "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
                )
            panel_stacks = [data[i:i + 1] for i in range(int(data.shape[0]))]
            if panel_frame_indices is None:
                resolved_indices = [0] * len(panel_stacks)
            else:
                _, resolved_indices, data = _normalize_grayscale_panel_items(
                    [data[i] for i in range(int(data.shape[0]))],
                    panel_frame_indices,
                )

        previous_shape = (int(self.n_images), int(self.height), int(self.width))
        self._updating_panel_frames = True
        try:
            with self.hold_sync():
                self._data = data
                self._panel_stacks = panel_stacks
                self._panel_stacks_original = [stack for stack in panel_stacks]
                self._panel_stack_originals_are_views = True
                self.panel_frame_counts = [int(stack.shape[0]) for stack in panel_stacks]
                self.n_images = int(data.shape[0])
                self.panel_frame_indices = list(resolved_indices)
                self.panel_stack_offsets = [-1] * self.n_images
                self._data_original = [self._data[i] for i in range(self._data.shape[0])]
                self._originals_are_views = True
                self._rgb_frames = [None] * int(data.shape[0])
                self._display_rgb = self._rgb_frames
                self.is_rgb = [False] * int(data.shape[0])
                self.n_pages = 1
                self.page_idx = 0
                self.panels_per_page = 0
                self.page_labels = []
                self.page_starred = [0]
                self.image_rotations = [0] * self.n_images
                self.starred = [0] * self.n_images
                self.hidden_panels = []
                self.panel_order = []
                if labels is None:
                    self.labels = [f"Image {i + 1}" for i in range(self.n_images)]
                else:
                    if len(labels) != self.n_images:
                        raise ValueError(
                            f"labels length ({len(labels)}) must match n_images ({self.n_images})"
                        )
                    self.labels = ["" if label is None else str(label) for label in labels]
                self.selected_idx = min(int(self.selected_idx), self.n_images - 1)
                self.roi_list = []
                self.roi_selected_idx = -1
                self.profile_line = []
                self._detail_request = ""
                self._detail_meta = ""
                self._detail_bytes = b""
                self.vmins = None
                self.vmaxs = None

                self._display_bin = max(1, int(getattr(self, "_display_bin", 1)))
                if self._display_bin > 1:
                    from quantem.widget.utils.array import bin2d

                    self._display_panel_stacks = [
                        bin2d(stack, factor=self._display_bin, mode="mean")
                        for stack in self._panel_stacks
                    ]
                else:
                    self._display_panel_stacks = self._panel_stacks
                self._display_data = np.stack(
                    [stack[index] for stack, index in zip(self._display_panel_stacks, resolved_indices)],
                    axis=0,
                ).astype(np.float32, copy=False)
                display = self._display_data
                self.height = int(display.shape[1])
                self.width = int(display.shape[2])
                self._display_bin_factor = self._display_bin
                if (self.n_images, self.height, self.width) != previous_shape:
                    self.view_box = []
                    self.zoom_row = None
                    self.zoom_col = None
                if list(self.view_crop) or float(self.pad_ratio) > 0.0:
                    # Replacing the pixels invalidates a committed crop/pad:
                    # the window described a view of the OLD data.
                    self._view_ops_ready = False
                    self.view_crop = []
                    self.pad_ratio = 0.0
                    self._view_ops_ready = True
                    self._refresh_view_ops(announce=False)
                self._compute_all_stats()
                self._update_all_frames()
        finally:
            self._updating_panel_frames = False

    def _apply_folder_image_records(
        self,
        old_records: list[ImageFolderRecord],
        new_records: list[ImageFolderRecord],
        changed_arrays: dict[pathlib.Path, np.ndarray],
    ) -> None:
        """Replace folder panels while remapping panel state through file paths."""
        old_paths = [record.path for record in old_records]
        new_paths = [record.path for record in new_records]
        old_page_idx = int(getattr(self, "page_idx", 0))
        old_index = {path: idx for idx, path in enumerate(old_paths)}
        new_index = {path: idx for idx, path in enumerate(new_paths)}
        old_display_bin = max(1, int(getattr(self, "_display_bin", 1)))
        if getattr(self, "_folder_display_bin_request", None) == "auto":
            sample = changed_arrays.get(new_paths[0])
            if sample is None and old_paths:
                sample = np.asarray(self._panel_stacks_original[old_index[new_paths[0]]][0])
            if sample is not None:
                self._display_bin = self._folder_auto_display_bin(
                    len(new_paths),
                    int(sample.shape[-2]),
                    int(sample.shape[-1]),
                )
        new_display_bin = max(1, int(getattr(self, "_display_bin", 1)))
        coordinate_scale = old_display_bin / new_display_bin

        if not old_paths:
            self.set_image(
                [changed_arrays[path] for path in new_paths],
                labels=[self._folder_source.label(path) for path in new_paths],
            )
            self._sync_folder_pages(anchor_panel=0)
            return

        originals = list(getattr(self, "_panel_stacks_original", []))
        arrays: list[np.ndarray] = []
        for path in new_paths:
            if path in changed_arrays:
                arrays.append(changed_arrays[path])
                continue
            idx = old_index[path]
            stack = originals[idx]
            arrays.append(np.asarray(stack[0]))

        selected_path = (
            old_paths[int(self.selected_idx)]
            if old_paths and 0 <= int(self.selected_idx) < len(old_paths)
            else old_paths[0]
        )
        starred_by_path = {
            path: bool(self.starred[idx])
            for idx, path in enumerate(old_paths)
            if idx < len(self.starred)
        }
        hidden_paths = {
            old_paths[idx]
            for idx in self.hidden_panels
            if 0 <= int(idx) < len(old_paths)
        }
        rotations_by_path = {
            path: int(self.image_rotations[idx])
            for idx, path in enumerate(old_paths)
            if idx < len(self.image_rotations)
        }

        ordered_paths: list[pathlib.Path] = []
        if self.panel_order:
            ordered_paths.extend(
                old_paths[idx]
                for idx in self.panel_order
                if 0 <= int(idx) < len(old_paths)
            )
            ordered_paths.extend(path for path in new_paths if path not in ordered_paths)

        roi_active = bool(self.roi_active)
        roi_list = list(self.roi_list)
        roi_selected_idx = int(self.roi_selected_idx)
        profile_line = list(self.profile_line)
        view_box = [float(value) * coordinate_scale for value in self.view_box]
        zoom_row = (
            None if self.zoom_row is None else float(self.zoom_row) * coordinate_scale
        )
        zoom_col = (
            None if self.zoom_col is None else float(self.zoom_col) * coordinate_scale
        )

        def remap_panel_values(values):
            if values is None or len(values) != len(old_paths):
                return None
            by_path = {path: values[idx] for idx, path in enumerate(old_paths)}
            return [by_path.get(path) for path in new_paths]

        vmins = remap_panel_values(self.vmins)
        vmaxs = remap_panel_values(self.vmaxs)

        with self.hold_sync():
            self.set_image(
                arrays,
                labels=[self._folder_source.label(path) for path in new_paths],
            )
            self.image_rotations = [rotations_by_path.get(path, 0) for path in new_paths]
            self.starred = [int(starred_by_path.get(path, False)) for path in new_paths]
            self.panel_order = [new_index[path] for path in ordered_paths if path in new_index]
            self.selected_idx = new_index.get(selected_path, 0)
            # ``set_image`` resets paging traits. Keep the page the scientist
            # was reviewing even when selection lives on a different page.
            self._sync_folder_pages(preferred_page=old_page_idx)
            self.hidden_panels = self._normalize_item_page_hidden(
                [
                    new_index[path]
                    for path in hidden_paths
                    if path in new_index
                ],
                drop_if_full=True,
            )
            self.roi_active = roi_active
            self.roi_list = roi_list
            self.roi_selected_idx = roi_selected_idx
            self.profile_line = profile_line
            self.view_box = view_box
            self.zoom_row = zoom_row
            self.zoom_col = zoom_col
            if vmins is not None:
                self.vmins = vmins
            if vmaxs is not None:
                self.vmaxs = vmaxs

    def _folder_auto_display_bin(
        self,
        n_images: int,
        height: int,
        width: int,
    ) -> int:
        """Resolve ``display_bin='auto'`` for the current watched inventory."""
        factor = 1
        per_image_mb = (height * width * 4 * 3) / (1024 * 1024)
        total_mb = int(n_images) * per_image_mb
        if total_mb > self._GPU_DISPLAY_BUDGET_MB:
            for candidate in (2, 4, 8):
                if total_mb / (candidate * candidate) <= self._GPU_DISPLAY_BUDGET_MB:
                    factor = candidate
                    break
            else:
                factor = 8
        panel_bytes = height * width * 4
        if panel_bytes > self._WIRE_BUDGET_BYTES_PER_PANEL:
            preview_floor_px = 512.0 if int(n_images) >= 16 else 1024.0
            canvas_css_px = (
                float(self.size)
                if int(self.size) > 0
                else 300.0
                if int(n_images) > 1
                else 500.0
            )
            preview_px = max(preview_floor_px, 2.0 * canvas_css_px)
            wire_bin = max(2, math.ceil(max(height, width) / preview_px))
            while (
                panel_bytes / (wire_bin * wire_bin)
                > self._WIRE_BUDGET_BYTES_PER_PANEL
            ):
                wire_bin += 1
            factor = max(factor, wire_bin)
        return factor

    def __repr__(self) -> str:
        if self.n_images > 1:
            shape = f"{self.n_images}×{self.height}×{self.width}"
            return f"Show2D({shape}, idx={self.selected_idx}, cmap={self.cmap})"
        return f"Show2D({self.height}×{self.width}, cmap={self.cmap})"

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.log_scale:
            frame = np.log1p(np.maximum(frame, 0))
        if self.vmin is not None and self.vmax is not None:
            vmin = float(self.vmin)
            vmax = float(self.vmax)
            if self.log_scale:
                vmin = float(np.log1p(max(vmin, 0)))
                vmax = float(np.log1p(max(vmax, 0)))
        elif self.auto_contrast:
            vmin = float(np.percentile(frame, 2))
            vmax = float(np.percentile(frame, 98))
        else:
            vmin = float(frame.min())
            vmax = float(frame.max())
        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)
            return normalized.astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    @staticmethod
    def _downsample_static_frame(frame: np.ndarray, max_px: int) -> np.ndarray:
        """Area-downsample a frame for notebook PNG fallback rendering."""
        if max_px <= 0:
            return frame
        h, w = frame.shape[-2:]
        factor = max(1, int(math.ceil(max(h, w) / max_px)))
        if factor == 1:
            return frame

        trimmed_h = max(1, h // factor) * factor
        trimmed_w = max(1, w // factor) * factor
        trimmed = frame[:trimmed_h, :trimmed_w]
        return trimmed.reshape(
            trimmed_h // factor,
            factor,
            trimmed_w // factor,
            factor,
        ).mean(axis=(1, 3))

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        idx: int | None = None,
        format: str | None = None,
        dpi: int = 150,
        title: bool | str = False,
        colorbar: bool = False,
        scalebar: bool = False,
    ) -> pathlib.Path:
        """Save current image as PNG, PDF, or TIFF.

        When ``title``, ``colorbar``, or ``scalebar`` are enabled, the output
        is a publication-quality figure rendered via matplotlib. Otherwise a
        raw colormapped image is saved directly (faster, exact pixel output).

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        idx : int, optional
            Image index in gallery mode. Defaults to current selected_idx.
        format : str, optional
            'png', 'pdf', or 'tiff'. If omitted, inferred from file extension.
        dpi : int, default 150
            Output DPI.
        title : bool or str, default False
            ``True`` uses the widget title, a string sets a custom title.
        colorbar : bool, default False
            Include a colorbar showing the intensity mapping.
        scalebar : bool, default False
            Include a scale bar (requires ``pixel_size > 0``).

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        from matplotlib import colormaps
        from PIL import Image

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        i = idx if idx is not None else self.selected_idx
        if i < 0 or i >= self.n_images:
            raise IndexError(f"Image index {i} out of range [0, {self.n_images})")

        panel_rgb = self._rgb_frames[i] if getattr(self, "_rgb_frames", None) else None
        frame = self._data[i]
        cmap_fn = colormaps.get_cmap(self.cmap)
        path.parent.mkdir(parents=True, exist_ok=True)
        if panel_rgb is not None:
            # RGB panels are display-ready: save the color pixels directly,
            # bypassing colormap and contrast. Colorbar is meaningless for an
            # RGB composite, so it is skipped.
            display_pixels = (np.clip(panel_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            colorbar = False
        else:
            normalized = self._normalize_frame(frame)
            display_pixels = None

        use_figure = title or colorbar or scalebar
        if not use_figure:
            if panel_rgb is not None:
                img = Image.fromarray(display_pixels, mode="RGB")
            else:
                rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)
                img = Image.fromarray(rgba)
                if fmt == "pdf":
                    Image.init()
                    img = img.convert("RGB")
            img.save(str(path), dpi=(dpi, dpi))
            return path

        # Publication-quality figure via matplotlib
        h, w = frame.shape
        aspect = h / w
        fig_w = 6
        fig, ax = plt.subplots(figsize=(fig_w, fig_w * aspect))
        if panel_rgb is not None:
            im = ax.imshow(display_pixels, origin="upper")
        else:
            im = ax.imshow(normalized, cmap=cmap_fn, vmin=0, vmax=255, origin="upper")
        ax.axis("off")

        if title:
            label = title if isinstance(title, str) else self.title
            if label:
                ax.set_title(label, fontsize=14, fontweight="bold", pad=8)

        if colorbar:
            # Map 0–255 back to data-space values for tick labels
            if self.log_scale:
                frame_proc = np.log1p(np.maximum(frame, 0))
            else:
                frame_proc = frame
            if self.vmin is not None and self.vmax is not None:
                dmin = float(self.vmin)
                dmax = float(self.vmax)
                if self.log_scale:
                    dmin = float(np.log1p(max(dmin, 0)))
                    dmax = float(np.log1p(max(dmax, 0)))
            elif self.auto_contrast:
                dmin = float(np.percentile(frame_proc, 2))
                dmax = float(np.percentile(frame_proc, 98))
            else:
                dmin = float(frame_proc.min())
                dmax = float(frame_proc.max())
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            n_ticks = 5
            tick_positions = np.linspace(0, 255, n_ticks)
            tick_labels = [f"{dmin + (dmax - dmin) * t / 255:.4g}" for t in tick_positions]
            cb.set_ticks(tick_positions)
            cb.set_ticklabels(tick_labels)

        if scalebar and self.pixel_size > 0:
            # Compute a nice scale bar length
            target_frac = 0.2  # ~20% of image width
            raw_length_px = target_frac * w
            raw_length_phys = raw_length_px * self.pixel_size  # in Å
            nice = _round_to_nice(raw_length_phys)
            bar_px = nice / self.pixel_size
            if nice >= 10:
                label_text = f"{nice / 10:.4g} nm"
            else:
                label_text = f"{nice:.4g} Å"
            margin = 0.03
            bar_y = h * (1 - margin) - 2
            bar_x = w * (1 - margin) - bar_px
            ax.plot([bar_x, bar_x + bar_px], [bar_y, bar_y],
                    color="white", linewidth=3, solid_capstyle="butt")
            ax.plot([bar_x, bar_x + bar_px], [bar_y, bar_y],
                    color="black", linewidth=1, solid_capstyle="butt")
            ax.text(bar_x + bar_px / 2, bar_y - h * 0.02, label_text,
                    color="white", fontsize=10, fontweight="bold",
                    ha="center", va="bottom",
                    path_effects=[
                        matplotlib.patheffects.withStroke(linewidth=2, foreground="black")
                    ])

        fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                    facecolor="white", pad_inches=0.1)
        plt.close(fig)
        return path

    def _gallery_export_chrome(self) -> dict[str, int | float | str]:
        """Return resolved publication chrome for gallery exports and previews."""
        gap_px = max(0, int(getattr(self, "inter_panel_gap_px", self.gallery_gap_px)))
        gap_color = str(getattr(self, "inter_panel_gap_color", self.gallery_gap_color) or "")
        outer_px = max(0, int(getattr(self, "gallery_outer_border_px", 0)))
        outer_color = str(getattr(self, "gallery_outer_border_color", "") or gap_color)
        panel_border_px = max(0.0, float(getattr(self, "panel_inner_border_px", 1.0)))
        panel_border_color = str(getattr(self, "panel_inner_border_color", "#d0d0d0") or "#d0d0d0")
        return {
            "inter_panel_gap_px": gap_px,
            "inter_panel_gap_color": gap_color,
            "gallery_outer_border_px": outer_px,
            "gallery_outer_border_color": outer_color,
            "panel_inner_border_px": panel_border_px,
            "panel_inner_border_color": panel_border_color,
        }

    def export_svg(
        self,
        path: str | pathlib.Path | None = None,
        *,
        scale: float = 3,
        include_scale_bar: bool = True,
        include_colorbar: bool = False,
        title: str | None = None,
    ) -> pathlib.Path:
        """Export the current Show2D gallery as a hybrid SVG figure.

        The SVG keeps figure chrome editable as vector elements: panel
        frames, marker bars, panel labels, title, and scale-bar text/line. The
        scientific image panels are embedded as PNG images at ``scale`` times
        the widget display size, which preserves the measured pixels while
        giving Illustrator or Inkscape sharp panels to place in a manuscript.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Output SVG path. Defaults to a descriptive filename in the current
            working directory.
        scale : float, default 3
            Embedded image scale relative to the widget display panel size.
            Values below 1 are clamped to 1; use 3 for the default
            high-resolution export, or a smaller value when file size matters.
        include_scale_bar : bool, default True
            Include the current scale bar when scale bars are visible on the
            widget. Set False to omit scale-bar chrome from the SVG.
        include_colorbar : bool, default False
            Include an editable SVG colorbar for single-panel exports. The
            live browser export uses the current Color switch state instead.
        title : str, optional
            Figure title override. Defaults to the widget title.

        Returns
        -------
        pathlib.Path
            The written SVG path.
        """
        from PIL import Image

        chrome = self._gallery_export_chrome()
        specs = self._static_panel_specs()
        if not specs:
            raise ValueError("Show2D has no visible panels to export")

        export_path = pathlib.Path(path) if path is not None else self._default_svg_export_path()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_scale = max(1.0, min(8.0, float(scale)))
        panel_w = int(round(self._static_canvas_css_px()))
        gap = int(chrome["inter_panel_gap_px"])
        gap_color = str(chrome["inter_panel_gap_color"])
        frame = int(chrome["gallery_outer_border_px"])
        frame_color = str(chrome["gallery_outer_border_color"])
        panel_border_px = float(chrome["panel_inner_border_px"])
        panel_border_color = str(chrome["panel_inner_border_color"])
        ncols = max(1, min(int(self.ncols), len(specs)))
        title_text = self.title if title is None else str(title)
        title_h = 30 if title_text and self.show_title else 0
        draw_scale = bool(include_scale_bar and self.scale_bar_visible)
        view = self.current_view
        row0, row1, col0, col1 = view["box"]

        def crop_slices(frame: np.ndarray) -> tuple[slice, slice]:
            height, width = frame.shape[:2]
            r0 = max(0, min(height - 1, int(math.floor(row0))))
            r1 = max(r0 + 1, min(height, int(math.ceil(row1))))
            c0 = max(0, min(width - 1, int(math.floor(col0))))
            c1 = max(c0 + 1, min(width, int(math.ceil(col1))))
            return slice(r0, r1), slice(c0, c1)

        resample = Image.Resampling.BILINEAR if self.smooth else Image.Resampling.NEAREST
        panels: list[dict[str, Any]] = []
        overlays = self._static_overlay_texts(specs, css_px=panel_w)
        for spec, overlay in zip(specs, overlays):
            frame_arr = np.asarray(spec["frame"])
            rows, cols = crop_slices(frame_arr)
            cropped = frame_arr[rows, cols]
            if spec.get("rgb"):
                rgb = (np.clip(cropped[..., :3], 0.0, 1.0) * 255).astype(np.uint8)
            else:
                rgb = self._static_panel_rgb(
                    cropped,
                    float(spec["vmin"]),
                    float(spec["vmax"]),
                    str(spec["cmap"]),
                    apply_log=bool(spec.get("apply_log")),
                )
            panel_h = max(1, int(round(panel_w * rgb.shape[0] / max(1, rgb.shape[1]))))
            image = Image.fromarray(rgb, mode="RGB")
            embed_w = max(1, int(round(panel_w * export_scale)))
            embed_h = max(1, int(round(panel_h * export_scale)))
            if image.size != (embed_w, embed_h):
                image = image.resize((embed_w, embed_h), resample=resample)
            buf = _io.BytesIO()
            image.save(buf, format="PNG")
            panel_index = int(spec.get("panel_index", len(panels)))
            panels.append({
                "panel_index": panel_index,
                "label": str(spec.get("label", "")),
                "height": panel_h,
                "png": base64.b64encode(buf.getvalue()).decode("ascii"),
                "bar_text": overlay[2] if draw_scale else "",
                "bar_px": float(overlay[3]) if draw_scale else 0.0,
                "vmin": float(spec.get("vmin", 0.0)),
                "vmax": float(spec.get("vmax", 1.0)),
                "cmap": str(spec.get("cmap", self.cmap)),
            })

        row_heights: list[int] = []
        for start in range(0, len(panels), ncols):
            row_heights.append(max(int(panel["height"]) for panel in panels[start:start + ncols]))
        svg_w = 2 * frame + ncols * panel_w + (ncols - 1) * gap
        svg_h = title_h + 2 * frame + sum(row_heights) + max(0, len(row_heights) - 1) * gap

        def esc_text(value: object) -> str:
            return html.escape(str(value), quote=False)

        def esc_attr(value: object) -> str:
            return html.escape(str(value), quote=True)

        def svg_color(value: object, fallback: str = "") -> str:
            text = str(value if value not in (None, "") else fallback).strip()
            match = re.fullmatch(
                r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(?:0|1|0?\.\d+)\s*\)",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                return f"rgb({int(match.group(1))}, {int(match.group(2))}, {int(match.group(3))})"
            return text

        def wrap_svg_label(value: object, font_size: int, max_width: float, max_lines: int = 3) -> list[str]:
            text = str(value or "").strip()
            if not text:
                return []
            width = max(1, int(max_width / max(1, font_size * 0.55)))
            return textwrap.wrap(text, width=width, break_long_words=True, max_lines=max_lines)

        latex_symbols = {
            r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
            r"\lambda": "λ", r"\mu": "μ", r"\sigma": "σ", r"\chi": "χ",
            r"\omega": "ω", r"\Delta": "Δ", r"\Theta": "Θ", r"\pm": "±",
            r"\times": "×", r"\cdot": "·", r"\degree": "°", r"\angstrom": "Å",
            r"\le": "≤", r"\ge": "≥", r"\neq": "≠", r"\approx": "≈",
            r"\infty": "∞",
        }

        def math_text(value: object) -> str:
            text = str(value or "").strip().strip("$")
            while "\\\\" in text:
                text = text.replace("\\\\", "\\")
            for key, symbol in latex_symbols.items():
                text = text.replace(key, symbol)
            superscript = str.maketrans({
                "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
                "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
                "n": "ⁿ", "i": "ⁱ",
            })
            subscript = str.maketrans({
                "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
                "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
                "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
                "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
                "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
                "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
                "v": "ᵥ", "x": "ₓ",
            })

            def read_group(start: int) -> tuple[str, int]:
                if start >= len(text) or text[start] != "{":
                    return (text[start] if start < len(text) else ""), start + 1
                depth = 0
                for idx in range(start, len(text)):
                    if text[idx] == "{":
                        depth += 1
                    elif text[idx] == "}":
                        depth -= 1
                        if depth == 0:
                            return text[start + 1:idx], idx + 1
                return text[start + 1:], len(text)

            out: list[str] = []
            idx = 0
            while idx < len(text):
                char = text[idx]
                if char in {"^", "_"} and idx + 1 < len(text):
                    raw, next_idx = read_group(idx + 1)
                    table = superscript if char == "^" else subscript
                    out.append(raw.translate(table))
                    idx = next_idx
                    continue
                if char in {"{", "}"}:
                    idx += 1
                    continue
                out.append(char)
                idx += 1
            return "".join(out)

        def span_text(span: Mapping[str, object]) -> str:
            if span.get("math") not in (None, ""):
                return math_text(span.get("math", ""))
            return str(span.get("text", ""))

        def dash_attr(spec: Mapping[str, object], width: float) -> str:
            raw_dash = spec.get("dash")
            if isinstance(raw_dash, Sequence) and not isinstance(raw_dash, (str, bytes, bytearray)):
                vals = [float(v) for v in raw_dash if float(v) >= 0]
                if any(vals):
                    return f' stroke-dasharray="{esc_attr(" ".join(f"{v:g}" for v in vals))}" stroke-linecap="round"'
            style = str(spec.get("line_style", "solid")).lower().replace("_", "-")
            w = max(1.0, float(width))
            patterns = {
                "dashed": [4 * w, 2 * w],
                "dash": [4 * w, 2 * w],
                "dotted": [w, 1.8 * w],
                "dot": [w, 1.8 * w],
                "dashdot": [4 * w, 2 * w, w, 2 * w],
                "dash-dot": [4 * w, 2 * w, w, 2 * w],
            }
            if style in patterns:
                return f' stroke-dasharray="{esc_attr(" ".join(f"{v:g}" for v in patterns[style]))}" stroke-linecap="round"'
            return ""

        def overlay_svg(spec: Mapping[str, object], x: float, y: float, panel_w: float, panel_h: float) -> str:
            shape = str(spec.get("shape", "circle")).lower()
            if shape == "rectangle":
                shape = "rect"
            coords = str(spec.get("coords", "data")).lower()
            view_h = max(1.0, float(row1) - float(row0))
            view_w = max(1.0, float(col1) - float(col0))

            def to_x(col: float) -> float:
                if coords == "relative":
                    return x + col * panel_w
                return x + (col - float(col0)) / view_w * panel_w

            def to_y(row: float) -> float:
                if coords == "relative":
                    return y + row * panel_h
                return y + (row - float(row0)) / view_h * panel_h

            def to_radius(radius: float) -> float:
                if coords == "relative":
                    return radius * min(panel_w, panel_h)
                return radius / max(view_w, view_h) * max(panel_w, panel_h)

            stroke = svg_color(spec.get("stroke"), "#00e5ff")
            stroke_width = max(0.0, float(spec.get("stroke_width", 2.0)))
            opacity = max(0.0, min(1.0, float(spec.get("opacity", 1.0))))
            stroke_opacity = opacity * max(0.0, min(1.0, float(spec.get("stroke_opacity", 1.0))))
            fill_value = spec.get("fill", "none")
            fill = "none" if fill_value in (None, "", "none", "None") else svg_color(fill_value)
            fill_opacity = opacity * max(0.0, min(1.0, float(spec.get("fill_opacity", 1.0 if fill != "none" else 0.0))))
            common = (
                f' fill="{esc_attr(fill)}" fill-opacity="{fill_opacity:g}"'
                f' stroke="{esc_attr(stroke)}" stroke-width="{stroke_width:g}"'
                f' stroke-opacity="{stroke_opacity:g}"{dash_attr(spec, stroke_width)}'
            )
            if shape == "circle":
                row = float(spec.get("row", 0.0))
                col = float(spec.get("col", 0.0))
                radius = max(0.0, float(spec.get("radius", 0.0)))
                cx = to_x(col)
                cy = to_y(row)
                r = to_radius(radius)
                return f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}"{common}/>'
            spec_row0 = float(spec.get("row0", 0.0))
            spec_col0 = float(spec.get("col0", 0.0))
            spec_row1 = float(spec.get("row1", spec_row0))
            spec_col1 = float(spec.get("col1", spec_col0))
            sx0 = to_x(min(spec_col0, spec_col1))
            sx1 = to_x(max(spec_col0, spec_col1))
            sy0 = to_y(min(spec_row0, spec_row1))
            sy1 = to_y(max(spec_row0, spec_row1))
            return f'<rect x="{sx0:g}" y="{sy0:g}" width="{max(0, sx1 - sx0):g}" height="{max(0, sy1 - sy0):g}"{common}/>'

        def annotation_anchor(position: str) -> tuple[float, float, str, str]:
            if "left" in position:
                ax = 0.0
                anchor = "start"
            elif "right" in position:
                ax = 1.0
                anchor = "end"
            else:
                ax = 0.5
                anchor = "middle"
            if "top" in position:
                ay = 0.0
                baseline = "hanging"
            elif "bottom" in position:
                ay = 1.0
                baseline = "baseline"
            else:
                ay = 0.5
                baseline = "middle"
            return ax, ay, anchor, baseline

        def annotation_svg(spec: Mapping[str, object], x: float, y: float, panel_w: float, panel_h: float) -> str:
            position = str(spec.get("position", "top-left"))
            font_size = max(6.0, float(spec.get("font_size", 10.0)))
            font_family = str(spec.get("font_family", "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"))
            pad_x = max(0.0, float(spec.get("pad_x", 6.0 if spec.get("variant", "badge") != "plain" else 0.0)))
            pad_y = max(0.0, float(spec.get("pad_y", 2.0 if spec.get("variant", "badge") != "plain" else 0.0)))
            opacity = max(0.0, min(1.0, float(spec.get("opacity", 1.0))))
            offset = spec.get("offset", (0.0, 0.0))
            off_x, off_y = (float(offset[0]), float(offset[1])) if isinstance(offset, Sequence) and len(offset) >= 2 else (0.0, 0.0)
            align = str(spec.get("align", "")).lower()
            align_anchor = {
                "left": "start",
                "start": "start",
                "center": "middle",
                "middle": "middle",
                "right": "end",
                "end": "end",
            }.get(align)
            if "box" in spec:
                left, top, width, height = (float(v) for v in spec["box"])
                if align_anchor == "start":
                    tx = x + left * panel_w + pad_x + off_x
                elif align_anchor == "end":
                    tx = x + (left + width) * panel_w - pad_x + off_x
                else:
                    tx = x + (left + width / 2.0) * panel_w + off_x
                ty = y + (top + height / 2.0) * panel_h + off_y
                anchor = align_anchor or "middle"
                baseline = "middle"
            elif "x" in spec and "y" in spec:
                tx = x + float(spec.get("x", 0.0)) * panel_w + off_x
                ty = y + float(spec.get("y", 0.0)) * panel_h + off_y
                _, _, anchor, baseline = annotation_anchor(str(spec.get("anchor", "center")))
                anchor = align_anchor or anchor
            else:
                ax, ay, anchor, baseline = annotation_anchor(position)
                anchor = align_anchor or anchor
                margin = 10.0
                tx = x + margin + ax * (panel_w - 2 * margin) + off_x
                ty = y + margin + ay * (panel_h - 2 * margin) + off_y
            spans = spec.get("spans")
            text = math_text(spec.get("math")) if spec.get("math") not in (None, "") else str(spec.get("text", ""))
            text_len = max(len(text), sum(len(span_text(span)) for span in spans) if isinstance(spans, Sequence) else 0)
            box_w = text_len * font_size * 0.62 + 2 * pad_x
            box_h = font_size * 1.25 + 2 * pad_y
            fg = svg_color(spec.get("fg", spec.get("color", "#fff")))
            variant = str(spec.get("variant", "badge"))
            outline_width = max(0.0, float(spec.get("outline_width", 0.0)))
            outline_color = svg_color(spec.get("outline_color"), "rgba(0,0,0,0.85)")
            parts = [f'<g opacity="{opacity:g}">']
            if variant != "plain":
                bg = svg_color(spec.get("bg"), "rgba(0,0,0,0.72)")
                border = svg_color(spec.get("border_color"), "rgba(255,255,255,0.5)")
                border_width = max(0.0, float(spec.get("border_width", 1.0 if variant in {"outline", "callout"} else 0.0)))
                rect_x = tx - box_w / 2 if anchor == "middle" else tx - box_w if anchor == "end" else tx
                rect_y = ty - box_h / 2 if baseline == "middle" else ty - font_size - pad_y if baseline == "baseline" else ty
                parts.append(
                    f'<rect x="{rect_x:g}" y="{rect_y:g}" width="{box_w:g}" height="{box_h:g}" '
                    f'rx="{float(spec.get("radius", 3.0)):g}" fill="{esc_attr(bg)}" '
                    f'stroke="{esc_attr(border)}" stroke-width="{border_width:g}"/>'
                )
            text_y = ty + (font_size * 0.4 if baseline == "middle" else 0.0)
            text_attrs = (
                f'x="{tx:g}" y="{text_y:g}" text-anchor="{anchor}" '
                f'font-family="{esc_attr(font_family)}" '
                f'font-size="{font_size:g}" font-weight="{esc_attr(spec.get("font_weight", 700))}"'
            )
            fill_parts = []
            if isinstance(spans, Sequence) and not isinstance(spans, (str, bytes, bytearray)):
                for span in spans:
                    if not isinstance(span, Mapping):
                        continue
                    color_attr = f' fill="{esc_attr(svg_color(span["color"]))}"' if span.get("color") else ""
                    fill_parts.append(f'<tspan{color_attr}>{esc_text(span_text(span))}</tspan>')
            else:
                fill_parts.append(esc_text(text))
            fill_body = "".join(fill_parts)
            if outline_width > 0:
                plain = esc_text(text if text else "".join(span_text(span) for span in spans or [] if isinstance(span, Mapping)))
                parts.append(
                    f'<text {text_attrs} fill="none" stroke="{esc_attr(outline_color)}" '
                    f'stroke-width="{outline_width:g}" stroke-linejoin="round">{plain}</text>'
                )
            parts.append(f'<text {text_attrs} fill="{esc_attr(fg)}">{fill_body}</text></g>')
            return "".join(parts)

        def inset_svg(spec: Mapping[str, object], x: float, y: float, panel_w: float, panel_h: float, fallback_color: str) -> str:
            if not spec:
                return ""
            xs = np.asarray(spec.get("x", []), dtype=float).ravel()
            ys = np.asarray(spec.get("y", []), dtype=float).ravel()
            finite = np.isfinite(xs) & np.isfinite(ys)
            if xs.size != ys.size or finite.sum() < 2:
                return ""
            xs = xs[finite]
            ys = ys[finite]
            xlim = tuple(float(v) for v in spec.get("xlim", (float(xs.min()), float(xs.max()))))
            ylim = tuple(float(v) for v in spec.get("ylim", (float(ys.min()), float(ys.max()))))
            if xlim[1] <= xlim[0]:
                xlim = (xlim[0] - 0.5, xlim[0] + 0.5)
            if ylim[1] <= ylim[0]:
                ylim = (ylim[0] - 0.5, ylim[0] + 0.5)
            size = max(0.18, min(0.62, float(spec.get("size", 0.31))))
            box_w = max(78.0, min(panel_w * 0.62, panel_w * size))
            box_h = max(50.0, min(panel_h * 0.55, panel_w * float(spec.get("height", size * 0.68))))
            if "box" in spec:
                left, top, width, height = (float(v) for v in spec["box"])
                box_w = max(48.0, min(panel_w, panel_w * width))
                box_h = max(34.0, min(panel_h, panel_h * height))
                x0 = x + max(0.0, min(panel_w - box_w, panel_w * left))
                y0 = y + max(0.0, min(panel_h - box_h, panel_h * top))
            else:
                margin_raw = spec.get("margin", (12.0, 12.0))
                if isinstance(margin_raw, (int, float)):
                    margin_x = margin_y = float(margin_raw)
                else:
                    margin_x, margin_y = (float(v) for v in list(margin_raw)[:2])
                pos = str(spec.get("position", "bottom-right"))
                x0 = x + (panel_w - box_w - margin_x if "right" in pos else panel_w / 2 - box_w / 2 if "center" in pos else margin_x)
                y0 = y + (panel_h - box_h - margin_y - (34.0 if self.scale_bar_visible and pos == "bottom-right" else 0.0) if "bottom" in pos else panel_h / 2 - box_h / 2 if "center" in pos else margin_y + 18.0)
            show_ticks = bool(spec.get("show_ticks", False))
            tick_font = max(5.0, min(14.0, float(spec.get("tick_font_size", 7.0))))
            label_font = max(6.0, min(16.0, float(spec.get("label_font_size", 8.0))))
            legend_font = max(6.0, min(18.0, float(spec.get("legend_font_size", 9.0))))
            pad_l = max(22.0, tick_font * 3.2) if show_ticks or spec.get("ylabel") else 10.0
            pad_r = 7.0
            pad_t = max(13.0, legend_font + 6.0) if spec.get("title") or spec.get("legend") else 7.0
            pad_b = max(16.0, tick_font + label_font + 4.0) if show_ticks or spec.get("xlabel") else 8.0
            plot_x0 = x0 + pad_l
            plot_y0 = y0 + pad_t
            plot_w = box_w - pad_l - pad_r
            plot_h = box_h - pad_t - pad_b
            if plot_w <= 8 or plot_h <= 8:
                return ""
            sx = lambda value: plot_x0 + (float(value) - xlim[0]) / (xlim[1] - xlim[0]) * plot_w
            sy = lambda value: plot_y0 + plot_h - (float(value) - ylim[0]) / (ylim[1] - ylim[0]) * plot_h
            points = " ".join(f"{sx(px):g},{sy(py):g}" for px, py in zip(xs, ys))
            line_color = svg_color(spec.get("color"), fallback_color)
            text_color = svg_color(spec.get("text_color"), "rgba(255,255,255,0.92)")
            tick_color = svg_color(spec.get("tick_color"), "rgba(255,255,255,0.72)")
            parts = [
                "<g>",
                f'<rect x="{x0:g}" y="{y0:g}" width="{box_w:g}" height="{box_h:g}" '
                f'fill="{esc_attr(svg_color(spec.get("background"), "#0a0c10"))}" fill-opacity="{max(0.0, min(1.0, float(spec.get("background_alpha", 0.68)))):g}" '
                f'stroke="{esc_attr(svg_color(spec.get("border_color"), "rgba(255,255,255,0.34)"))}" stroke-width="{float(spec.get("border_width", 1.0)):g}"/>',
                f'<path d="M {plot_x0:g} {plot_y0:g} V {plot_y0 + plot_h:g} H {plot_x0 + plot_w:g}" fill="none" stroke="{esc_attr(tick_color)}" stroke-opacity="0.45" stroke-width="1"/>',
                f'<polyline points="{points}" fill="none" stroke="{esc_attr(line_color)}" stroke-width="{max(1.4, float(spec.get("line_width", 2.0))):g}" stroke-linejoin="round" stroke-linecap="round"/>',
            ]
            if "point" in spec:
                point = np.asarray(spec["point"], dtype=float).ravel()
                if point.size == 2 and np.isfinite(point).all():
                    parts.append(f'<circle cx="{sx(point[0]):g}" cy="{sy(point[1]):g}" r="3.4" fill="{esc_attr(svg_color(spec.get("point_color"), "#fff"))}" stroke="#000" stroke-width="1.5"/>')
            if spec.get("title"):
                parts.append(f'<text x="{x0 + 6:g}" y="{y0 + 12:g}" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" font-size="{legend_font:g}" font-weight="700" fill="{esc_attr(text_color)}">{esc_text(spec["title"])}</text>')
            if spec.get("legend"):
                parts.append(f'<text x="{x0 + 6:g}" y="{y0 + box_h - 6:g}" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" font-size="{legend_font:g}" font-weight="700" fill="{esc_attr(line_color)}">{esc_text(spec["legend"])}</text>')
            if spec.get("xlabel"):
                parts.append(f'<text x="{x0 + box_w - 7:g}" y="{y0 + box_h - 3:g}" text-anchor="end" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" font-size="{label_font:g}" fill="{esc_attr(tick_color)}">{esc_text(spec["xlabel"])}</text>')
            if spec.get("ylabel"):
                parts.append(f'<text x="{x0 + 5:g}" y="{plot_y0 + 2:g}" transform="rotate(-90 {x0 + 5:g} {plot_y0 + 2:g})" text-anchor="end" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" font-size="{label_font:g}" fill="{esc_attr(tick_color)}">{esc_text(spec["ylabel"])}</text>')
            parts.append("</g>")
            return "".join(parts)

        def colorbar_svg(panel: Mapping[str, object], x: float, y: float, panel_w: float, panel_h: float, idx: int) -> tuple[str, str]:
            grad_id = f"show2d-svg-colorbar-{idx}"
            cmap_obj = matplotlib.colormaps.get_cmap(str(panel.get("cmap", self.cmap)))
            stops = []
            for step in range(9):
                frac = step / 8.0
                r, g, b, _ = cmap_obj(frac)
                stops.append(
                    f'<stop offset="{frac * 100:g}%" stop-color="rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"/>'
                )
            bar_h = min(160.0, panel_h * 0.62)
            bar_w = 10.0
            bx = x + panel_w - 22.0
            by = y + 18.0
            vmin = float(panel.get("vmin", 0.0))
            vmax = float(panel.get("vmax", 1.0))
            body = [
                "<g>",
                f'<rect x="{bx - 1:g}" y="{by - 1:g}" width="{bar_w + 2:g}" height="{bar_h + 2:g}" fill="#000" fill-opacity="0.45"/>',
                f'<rect x="{bx:g}" y="{by:g}" width="{bar_w:g}" height="{bar_h:g}" fill="url(#{grad_id})" stroke="#fff" stroke-opacity="0.75" stroke-width="0.75"/>',
                f'<text x="{bx - 4:g}" y="{by + 4:g}" text-anchor="end" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" font-size="9" fill="#fff">{esc_text(self._format_stat(vmax))}</text>',
                f'<text x="{bx - 4:g}" y="{by + bar_h:g}" text-anchor="end" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" font-size="9" fill="#fff">{esc_text(self._format_stat(vmin))}</text>',
                "</g>",
            ]
            return (
                f'<linearGradient id="{grad_id}" x1="0" x2="0" y1="1" y2="0">{"".join(stops)}</linearGradient>',
                "".join(body),
            )

        elements: list[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{svg_w}" '
                f'height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" '
                f'role="img" aria-label="{esc_attr(title_text or "Show2D SVG export")}">'
            ),
        ]
        defs: list[str] = []
        if title_h:
            elements.append(
                f'<text x="{svg_w / 2:g}" y="19" text-anchor="middle" '
                'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" '
                f'font-size="14" font-weight="700" fill="#111">{esc_text(title_text)}</text>'
            )
        if frame > 0 and frame_color:
            elements.append(
                f'<rect x="0" y="{title_h:g}" width="{svg_w:g}" height="{max(0, svg_h - title_h):g}" '
                f'fill="{esc_attr(svg_color(frame_color))}"/>'
            )
        if gap > 0 and gap_color:
            elements.append(
                f'<rect x="{frame:g}" y="{title_h + frame:g}" '
                f'width="{max(0, svg_w - 2 * frame):g}" height="{max(0, svg_h - title_h - 2 * frame):g}" '
                f'fill="{esc_attr(svg_color(gap_color))}"/>'
            )

        title_style = dict(getattr(self, "panel_title_style", {}) or {})
        title_font_family = str(title_style.get("font_family", "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"))
        title_fg = svg_color(title_style.get("fg"), "#fff")
        title_opacity = max(0.0, min(1.0, float(title_style.get("opacity", 0.95))))
        title_font_weight = title_style.get("font_weight", 700)
        title_outline_width = max(0.0, float(title_style.get("outline_width", 0.0)))
        title_outline_color = svg_color(title_style.get("outline_color"), "rgba(0,0,0,0.85)")

        def title_anchor_for(value: str) -> str:
            if "right" in value:
                return "end"
            if "center" in value:
                return "middle"
            return "start"

        def title_baseline_y(value: str, y: float, panel_h: float, font_size: float) -> float:
            if "bottom" in value:
                return y + panel_h
            if "center" in value:
                return y + panel_h / 2.0 + font_size * 0.35
            return y + font_size

        def title_position(x: float, y: float, panel_w: float, panel_h: float, font_size: float) -> tuple[float, float, str]:
            offset_raw = title_style.get("offset", (0.0, 0.0))
            if isinstance(offset_raw, Sequence) and not isinstance(offset_raw, (str, bytes, bytearray)):
                vals = list(offset_raw)
                off_x = float(vals[0]) if vals else 0.0
                off_y = float(vals[1]) if len(vals) > 1 else 0.0
            else:
                off_x = off_y = 0.0
            if "x" in title_style or "y" in title_style:
                rel_x = float(title_style.get("x", 0.5))
                rel_y = float(title_style.get("y", 0.0))
                anchor_value = str(title_style.get("anchor", "top-center")).lower()
                return (
                    x + rel_x * panel_w + off_x,
                    title_baseline_y(anchor_value, y + rel_y * panel_h + off_y, 0.0, font_size),
                    title_anchor_for(anchor_value),
                )
            align = str(title_style.get("align", "center")).lower()
            if align in {"left", "start"}:
                return x + 28.0, y + 6.0 + font_size, "start"
            if align in {"right", "end"}:
                return x + panel_w - 28.0, y + 6.0 + font_size, "end"
            return x + panel_w / 2.0, y + 6.0 + font_size, "middle"

        def title_text_attrs(anchor: str, *, shadow: bool = False) -> str:
            if shadow:
                return (
                    f'text-anchor="{anchor}" font-family="{esc_attr(title_font_family)}" '
                    f'font-size="{{font_size}}" font-weight="{esc_attr(title_font_weight)}" '
                    'fill="#000" fill-opacity="0.85"'
                )
            return (
                f'text-anchor="{anchor}" font-family="{esc_attr(title_font_family)}" '
                f'font-size="{{font_size}}" font-weight="{esc_attr(title_font_weight)}" '
                f'fill="{esc_attr(title_fg)}" fill-opacity="{title_opacity:g}"'
            )

        def title_outline_text(x_pos: float, y_pos: float, anchor: str, font_size: float, text: str) -> str:
            return (
                f'<text x="{x_pos:g}" y="{y_pos:g}" text-anchor="{anchor}" '
                f'font-family="{esc_attr(title_font_family)}" font-size="{font_size:g}" '
                f'font-weight="{esc_attr(title_font_weight)}" fill="none" '
                f'stroke="{esc_attr(title_outline_color)}" stroke-width="{title_outline_width:g}" '
                f'stroke-linejoin="round">{esc_text(text)}</text>'
            )

        y = title_h + frame
        for row_idx, row_h in enumerate(row_heights):
            row_panels = panels[row_idx * ncols:(row_idx + 1) * ncols]
            for col_idx, panel in enumerate(row_panels):
                x = frame + col_idx * (panel_w + gap)
                panel_h = int(panel["height"])
                panel_index = int(panel["panel_index"])
                clip_id = f"show2d-svg-panel-clip-{panel_index}-{row_idx}-{col_idx}"
                defs.append(
                    f'<clipPath id="{clip_id}"><rect x="{x:g}" y="{y:g}" width="{panel_w:g}" height="{panel_h:g}"/></clipPath>'
                )
                marker_color = (
                    svg_color(self.marker_colors[panel_index])
                    if panel_index < len(self.marker_colors) and self.marker_colors[panel_index]
                    else ""
                )
                panel_stroke = svg_color(panel_border_color, "#d0d0d0")
                elements.extend([
                    f'<g id="show2d-panel-{panel_index}">',
                    f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" fill="#000"/>',
                    (
                        f'<image x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" '
                        f'xlink:href="data:image/png;base64,{panel["png"]}" preserveAspectRatio="none"/>'
                    ),
                ])
                if panel_border_px > 0:
                    elements.append(
                        f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" fill="none" '
                        f'stroke="{esc_attr(panel_stroke)}" stroke-width="{panel_border_px:g}"/>'
                    )
                if marker_color and str(self.marker_style or "left") == "around":
                    elements.append(
                        f'<rect x="{x + 1.5:g}" y="{y + 1.5:g}" width="{max(0, panel_w - 3):g}" '
                        f'height="{max(0, panel_h - 3):g}" fill="none" '
                        f'stroke="{esc_attr(marker_color)}" stroke-width="3"/>'
                    )
                elif marker_color:
                    elements.append(
                        f'<rect x="{x}" y="{y}" width="5" height="{panel_h}" fill="{esc_attr(marker_color)}"/>'
                    )
                label = panel["label"]
                if label:
                    font_size = max(8, int(self.panel_title_font_size or 11))
                    spans = self.panel_title_spans[panel_index] if panel_index < len(self.panel_title_spans) else []
                    title_x, title_y, title_anchor = title_position(x, y, panel_w, panel_h, font_size)
                    if spans:
                        line_y = title_y
                        plain = "".join(span_text(span) for span in spans if isinstance(span, Mapping))
                        if title_outline_width <= 0:
                            elements.append(
                                f'<text x="{title_x + 1:g}" y="{line_y + 1:g}" '
                                + title_text_attrs(title_anchor, shadow=True).format(font_size=font_size)
                                + f'>{esc_text(plain)}</text>'
                            )
                        else:
                            elements.append(title_outline_text(title_x, line_y, title_anchor, font_size, plain))
                        elements.append(
                            f'<text x="{title_x:g}" y="{line_y:g}" '
                            + title_text_attrs(title_anchor).format(font_size=font_size)
                            + ">"
                        )
                        for span in spans:
                            if not isinstance(span, Mapping):
                                continue
                            color_attr = f' fill="{esc_attr(svg_color(span["color"]))}"' if span.get("color") else ""
                            elements.append(f'<tspan{color_attr}>{esc_text(span_text(span))}</tspan>')
                        elements.append("</text>")
                    else:
                        for line_idx, line in enumerate(wrap_svg_label(label, font_size, max(24, panel_w - 56), 3)):
                            line_y = title_y + line_idx * font_size * 1.2
                            if title_outline_width <= 0:
                                elements.append(
                                    f'<text x="{title_x + 1:g}" y="{line_y + 1:g}" '
                                    + title_text_attrs(title_anchor, shadow=True).format(font_size=font_size)
                                    + f'>{esc_text(line)}</text>'
                                )
                            else:
                                elements.append(title_outline_text(title_x, line_y, title_anchor, font_size, line))
                            elements.append(
                                f'<text x="{title_x:g}" y="{line_y:g}" '
                                + title_text_attrs(title_anchor).format(font_size=font_size)
                                + f'>{esc_text(line)}</text>'
                            )
                elements.append(f'<g clip-path="url(#{clip_id})">')
                if bool(self.show_inset_plots) and panel_index < len(self.inset_plots):
                    elements.append(inset_svg(self.inset_plots[panel_index], x, y, panel_w, panel_h, marker_color))
                if panel_index < len(self.panel_overlays):
                    for overlay_spec in self.panel_overlays[panel_index]:
                        if isinstance(overlay_spec, Mapping):
                            elements.append(overlay_svg(overlay_spec, x, y, panel_w, panel_h))
                if include_colorbar:
                    grad, bar = colorbar_svg(panel, x, y, panel_w, panel_h, panel_index)
                    defs.append(grad)
                    elements.append(bar)
                if panel_index < len(self.panel_annotations):
                    for annotation_spec in self.panel_annotations[panel_index]:
                        if isinstance(annotation_spec, Mapping):
                            elements.append(annotation_svg(annotation_spec, x, y, panel_w, panel_h))
                elements.append("</g>")
                bar_text = panel["bar_text"]
                bar_px = float(panel["bar_px"])
                if bar_text and bar_px > 0:
                    scale_style = dict(getattr(self, "scale_bar_style", {}) or {})
                    scale_offset = scale_style.get("offset", [0.0, 0.0])
                    try:
                        offset_x = float(scale_offset[0])
                        offset_y = float(scale_offset[1])
                    except (TypeError, ValueError, IndexError):
                        offset_x = 0.0
                        offset_y = 0.0
                    bar_height = float(scale_style.get("bar_height", 5.0))
                    label_gap = float(scale_style.get("label_gap", 4.0))
                    scale_font_size = float(scale_style.get("font_size", 16.0))
                    scale_font_family = str(
                        scale_style.get(
                            "font_family",
                            "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                        )
                    )
                    scale_font_weight = scale_style.get("font_weight", "")
                    scale_color = svg_color(scale_style.get("color"), "#fff")
                    scale_outline_color = svg_color(scale_style.get("outline_color"), "#000")
                    scale_outline_width = float(scale_style.get("outline_width", 0.0))
                    shadow_value = scale_style.get("shadow_color")
                    shadow_color = svg_color(shadow_value, "#000")
                    scale_left = self.scale_bar_position == "bottom-left"
                    bar_x = x + (12 if scale_left else panel_w - bar_px - 12) + offset_x
                    bar_y = y + panel_h - 12 + offset_y
                    font_weight_attr = f' font-weight="{esc_attr(scale_font_weight)}"' if scale_font_weight != "" else ""
                    if shadow_value is not None:
                        elements.append(
                            f'<rect x="{bar_x + 1:g}" y="{bar_y + 1:g}" width="{bar_px:g}" '
                            f'height="{bar_height:g}" fill="{esc_attr(shadow_color)}" fill-opacity="0.5"/>'
                        )
                    elements.append(
                        f'<rect x="{bar_x:g}" y="{bar_y:g}" width="{bar_px:g}" height="{bar_height:g}" fill="{esc_attr(scale_color)}"/>'
                    )
                    text_x = bar_x + bar_px / 2
                    text_y = bar_y - label_gap
                    if scale_outline_width > 0:
                        elements.extend([
                            f'<text x="{text_x:g}" y="{text_y:g}" text-anchor="middle" '
                            f'font-family="{esc_attr(scale_font_family)}" font-size="{scale_font_size:g}"{font_weight_attr} '
                            f'fill="none" stroke="{esc_attr(scale_outline_color)}" '
                            f'stroke-width="{scale_outline_width:g}" stroke-linejoin="round">{esc_text(bar_text)}</text>',
                            f'<text x="{text_x:g}" y="{text_y:g}" text-anchor="middle" '
                            f'font-family="{esc_attr(scale_font_family)}" font-size="{scale_font_size:g}"{font_weight_attr} '
                            f'fill="{esc_attr(scale_color)}">{esc_text(bar_text)}</text>',
                        ])
                    else:
                        elements.extend([
                            f'<text x="{text_x + 1:g}" y="{text_y + 1:g}" text-anchor="middle" '
                            f'font-family="{esc_attr(scale_font_family)}" font-size="{scale_font_size:g}"{font_weight_attr} '
                            f'fill="{esc_attr(shadow_color)}" fill-opacity="0.85">{esc_text(bar_text)}</text>',
                            f'<text x="{text_x:g}" y="{text_y:g}" text-anchor="middle" '
                            f'font-family="{esc_attr(scale_font_family)}" font-size="{scale_font_size:g}"{font_weight_attr} '
                            f'fill="{esc_attr(scale_color)}">{esc_text(bar_text)}</text>',
                        ])
                    if self.show_zoom_indicator:
                        zoom_text = f"{min(max(float(self.initial_zoom) or 1.0, 0.5), 20.0):.1f}×"
                        zoom_x = x + (panel_w - 12 if scale_left else 12)
                        anchor = "end" if scale_left else "start"
                        elements.extend([
                            f'<text x="{zoom_x + 1:g}" y="{y + panel_h - 6:g}" text-anchor="{anchor}" '
                            f'font-family="{esc_attr(scale_font_family)}" '
                            f'font-size="{scale_font_size:g}"{font_weight_attr} fill="{esc_attr(shadow_color)}" fill-opacity="0.85">{esc_text(zoom_text)}</text>',
                            f'<text x="{zoom_x:g}" y="{y + panel_h - 7:g}" text-anchor="{anchor}" '
                            f'font-family="{esc_attr(scale_font_family)}" '
                            f'font-size="{scale_font_size:g}"{font_weight_attr} fill="{esc_attr(scale_color)}">{esc_text(zoom_text)}</text>',
                        ])
                elements.append("</g>")
            y += row_h + (gap if row_idx < len(row_heights) - 1 else 0)

        marker_top = title_h + frame
        total_grid_h = sum(row_heights) + max(0, len(row_heights) - 1) * gap
        for raw_row, color in dict(self.row_markers or {}).items():
            color = svg_color(color)
            try:
                row_idx = int(raw_row)
            except (TypeError, ValueError):
                continue
            if row_idx < 0 or row_idx >= len(row_heights):
                continue
            row_y = marker_top + sum(row_heights[:row_idx]) + row_idx * gap
            row_count = min(ncols, max(0, len(panels) - row_idx * ncols))
            row_w = row_count * panel_w + max(0, row_count - 1) * gap
            elements.extend([
                f'<rect x="{frame:g}" y="{row_y:g}" width="{row_w:g}" height="{row_heights[row_idx]:g}" fill="none" stroke="{esc_attr(color)}" stroke-width="3"/>',
                f'<rect x="{frame + 3:g}" y="{row_y + 3:g}" width="{max(0, row_w - 6):g}" height="{max(0, row_heights[row_idx] - 6):g}" fill="none" stroke="#000" stroke-opacity="0.9" stroke-width="2"/>',
            ])
        for raw_col, color in dict(self.col_markers or {}).items():
            color = svg_color(color)
            try:
                col_idx = int(raw_col)
            except (TypeError, ValueError):
                continue
            if col_idx < 0 or col_idx >= ncols:
                continue
            slots = [slot for slot in range(len(panels)) if slot % ncols == col_idx]
            if not slots:
                continue
            row_min = min(slot // ncols for slot in slots)
            row_max = max(slot // ncols for slot in slots)
            col_x = frame + col_idx * (panel_w + gap)
            col_y = marker_top + sum(row_heights[:row_min]) + row_min * gap
            col_h = sum(row_heights[row_min:row_max + 1]) + max(0, row_max - row_min) * gap
            elements.extend([
                f'<rect x="{col_x:g}" y="{col_y:g}" width="{panel_w:g}" height="{col_h:g}" fill="none" stroke="{esc_attr(color)}" stroke-width="3"/>',
                f'<rect x="{col_x + 3:g}" y="{col_y + 3:g}" width="{max(0, panel_w - 6):g}" height="{max(0, col_h - 6):g}" fill="none" stroke="#000" stroke-opacity="0.9" stroke-width="2"/>',
            ])

        if defs:
            elements.insert(2, f'<defs>{"".join(defs)}</defs>')
        elements.append("</svg>")
        export_path.write_text("\n".join(elements), encoding="utf-8")
        return export_path

    def _default_svg_export_path(self) -> pathlib.Path:
        """Default local filename for scripted Show2D SVG export."""
        slug = "".join(c.lower() if c.isalnum() else "_" for c in (self.title or "show2d"))
        while "__" in slug:
            slug = slug.replace("__", "_")
        slug = slug.strip("_") or "show2d"
        shape = f"{self.n_images}x{self.height}x{self.width}" if int(self.n_images) > 1 else f"{self.height}x{self.width}"
        return pathlib.Path.cwd() / f"{slug}_{shape}.svg"

    @property
    def view_corner(self):
        """Current view as ``(row0, col0, size)`` - paste straight into ``view_box=``.

        ``view_box`` live-syncs from JS on every pan/zoom, so after interacting this
        reads the region on screen. Restore next session with
        ``Show2D(..., view_box=w.view_corner)``.
        """
        if not self.view_box:
            return (0.0, 0.0, float(max(self.height, self.width)))
        r0, r1, c0, c1 = self.view_box
        return (round(r0, 1), round(c0, 1), round(max(r1 - r0, c1 - c0), 1))

    @property
    def view_center(self):
        """Current view as ``dict(zoom=..., zoom_row=..., zoom_col=...)`` - paste as kwargs.

        The center/zoom flavor of :attr:`view_corner`: restore with
        ``Show2D(..., **w.view_center)``.
        """
        if not self.view_box:
            return dict(zoom=1.0, center=None)
        r0, r1, c0, c1 = self.view_box
        zoom = float(min(self.height / max(1.0, r1 - r0), self.width / max(1.0, c1 - c0)))
        return dict(zoom=round(zoom, 2), center=(round((r0 + r1) / 2, 1), round((c0 + c1) / 2, 1)))

    def _on_detail_request_change(self, change: dict) -> None:
        """Serve a maps-style detail request: crop the visible window from the
        FULL-resolution data, mean-bin it to near canvas resolution, and reply
        with a small float32 buffer. This is how a binned preview still shows
        true full-res pixels under zoom: the browser swaps the tile in over
        the preview, and the 100+ MB full frame never crosses the wire.
        Coordinates arrive in preview pixels (the JS-side image space) and are
        scaled back to full resolution here; the reply reports full-res
        coordinates plus the tile bin so JS can place it exactly."""
        raw = str(change.get("new") or "")
        if not raw or getattr(self, "_data", None) is None:
            return
        from quantem.widget.utils.array import bin2d
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            return
        factor = max(1, int(self._display_bin_factor))
        full_h, full_w = int(self._data.shape[1]), int(self._data.shape[2])
        # With a crop/pad active, JS coordinates are frame-local: shift into
        # full-image pixels here, clamp to the committed crop window so a
        # tile never un-crops the view, and reply in frame-local coordinates.
        offset_row, offset_col = (int(v) for v in self._view_crop_offset)
        if len(self.view_crop) == 4:
            win_row0, win_row1, win_col0, win_col1 = (int(v) for v in self.view_crop)
        else:
            win_row0, win_row1, win_col0, win_col1 = 0, full_h, 0, full_w
        tiles: list[dict] = []
        blocks: list[bytes] = []
        offset = 0
        for spec in request.get("tiles", []):
            panel = int(spec.get("panel", -1))
            # RGB panels keep preview-only rendering: a color tile would need a
            # second interleaved payload path for a rare panel type.
            if not (0 <= panel < self.n_images) or (panel < len(self.is_rgb) and self.is_rgb[panel]):
                continue
            bin_factor = max(1, int(spec.get("bin", 1)))
            # Snap the window outward to bin multiples anchored at the image
            # origin so mean-binning blocks tile the crop with no partial edges.
            row0 = max(win_row0, math.floor((float(spec["row0"]) * factor + offset_row) / bin_factor) * bin_factor)
            col0 = max(win_col0, math.floor((float(spec["col0"]) * factor + offset_col) / bin_factor) * bin_factor)
            row1 = min(win_row1, math.ceil((float(spec["row1"]) * factor + offset_row) / bin_factor) * bin_factor)
            col1 = min(win_col1, math.ceil((float(spec["col1"]) * factor + offset_col) / bin_factor) * bin_factor)
            if row1 <= row0 or col1 <= col0:
                continue
            crop = self._data[panel, row0:row1, col0:col1]  # view: never copies the full array
            tile = bin2d(crop, factor=bin_factor, mode="mean") if bin_factor > 1 else crop
            tile = np.ascontiguousarray(tile, dtype=np.float32)
            if self._panel_filter_active(panel):
                # A raw full-res tile over a filtered preview would silently
                # un-filter the zoomed view. Filtering the crop is edge-
                # approximate near the tile border, but honest about the knobs.
                # Detail tiles stay on this Python path even when the browser
                # filters the preview (_webgpu_filter_ok): tiles only exist in
                # kernel sessions and already round-trip, and the committed
                # knobs here match the browser's committed knobs.
                tile = np.ascontiguousarray(self._filter_display_frame(tile, panel=panel))
            blocks.append(tile.tobytes())
            tiles.append({"panel": panel, "row0": row0 - offset_row, "col0": col0 - offset_col,
                          "rows": int(tile.shape[0]), "cols": int(tile.shape[1]),
                          "bin": bin_factor, "offset": offset})
            offset += tile.nbytes
        # One comm message for bytes + meta so JS never pairs a fresh meta with
        # a stale buffer (or vice versa) mid-update.
        with self.hold_sync():
            self._detail_bytes = _b64_safe(b"".join(blocks))
            self._detail_meta = json.dumps({"id": str(request.get("id", "")), "tiles": tiles})

    # Traits that carry the bulk pixel payload. Dropped from the saved-notebook
    # snapshot when save_state is False so a plain display stays a few MB, not GB.
    _UNSAVED_HEAVY_KEYS = (
        "frame_bytes",
        "panel_stack_bytes",
        "export_payload",
        "_detail_bytes",
    )

    def get_state(self, key=None, drop_defaults=False):
        """Trait state for comm sync and notebook embedding.

        ipywidgets calls this with ``key=None`` to snapshot the FULL state that
        gets written into the saved notebook's ``metadata.widgets``. When
        ``save_state`` is False we drop the heavy image buffers from that
        snapshot, so a plain Show2D does not bake ~1 GB of pixels into the
        .ipynb. Targeted syncs (``key`` is a name or set, used by hold_sync /
        send_state during live rendering) are untouched, so the frontend still
        receives ``frame_bytes`` normally. ``save_state=True`` embeds everything
        so a reopened notebook restores the interactive widget without a kernel.
        """
        state = super().get_state(key=key, drop_defaults=drop_defaults)
        if key is None and not getattr(self, "_save_state", False):
            if not self._static_fallback_enabled():
                state.pop("_static_fallback_jpeg", None)
                state.pop("_static_fallback_mime", None)
            elif not self._static_fallback_jpeg:
                png = self._static_fallback_png_b64()
                if png:
                    self._store_static_fallback_preview(png)
                    state["_static_fallback_jpeg"] = self._static_fallback_jpeg
                    state["_static_fallback_mime"] = self._static_fallback_mime
            for heavy_key in self._UNSAVED_HEAVY_KEYS:
                state.pop(heavy_key, None)
        if (
            key is None
            and not getattr(self, "_initial_live_mount_state", False)
            and state.get("folder_watch_state") in {
            "watching",
            "updating",
            "waiting",
            "error",
            }
        ):
            # Saved widget state retains pixels, not the Python watcher thread.
            # Keep snapshots truthful even when exported while acquisition is
            # still running in the current kernel.
            state["folder_watch_state"] = "stopped"
            state["folder_watch_detail"] = (
                "Folder watcher is not running in saved widget state. "
                "Re-run the cell to resume live folder updates."
            )
            if state.get("folder_waiting"):
                state["folder_status"] = (
                    "No completed image was captured in this saved widget. "
                    "Re-run the cell to resume folder watching."
                )
        return state

    def _with_initial_live_mount_state(self, fn, *args, **kwargs):
        """Keep live watcher state truthful during the first display handshake."""
        self._initial_live_mount_state = True
        try:
            return fn(*args, **kwargs)
        finally:
            self._initial_live_mount_state = False

    def _repr_mimebundle_(self, **kwargs):
        return self._with_initial_live_mount_state(
            super()._repr_mimebundle_,
            **kwargs,
        )

    def _ipython_display_(self):
        return self._with_initial_live_mount_state(super()._ipython_display_)

    # Colormaps the frontend treats as sequential; a signed diff panel switches
    # to a diverging map (RdBu) because zero must sit at the visual midpoint.
    _SEQUENTIAL_CMAPS = frozenset(
        {"inferno", "viridis", "plasma", "magma", "hot", "gray", "turbo"}
    )

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
        num = len(frames)
        # RGB panels bypass the contrast pipeline entirely: their range is the
        # fixed display-ready [0, 1], and they are excluded from linked-contrast
        # merging so a [0, 1] overlay never drags a counts-scaled panel's window.
        rgb = list(self.is_rgb) if any(self.is_rgb) else [False] * num
        gray = [i for i in range(num) if not (i < len(rgb) and rgb[i])]
        def to_display(value: float) -> float:
            return float(self._signed_log1p(float(value))) if self.log_scale else float(value)
        has_absolute = self.vmin is not None and self.vmax is not None
        per_mins = list(self.vmins) if self.vmins else [None] * num
        per_maxs = list(self.vmaxs) if self.vmaxs else [None] * num
        has_per_image = [per_mins[i] is not None and per_maxs[i] is not None for i in range(num)]
        base_ranges: list[tuple[float, float]] = []
        for i, frame in enumerate(frames):
            if i < len(rgb) and rgb[i]:
                base_ranges.append((0.0, 1.0))
            elif has_per_image[i]:
                base_ranges.append((to_display(per_mins[i]), to_display(per_maxs[i])))
            elif has_absolute:
                base_ranges.append((to_display(self.vmin), to_display(self.vmax)))
            else:
                base_ranges.append((to_display(frame.min()), to_display(frame.max())))
        linked_shared = (
            self.link_contrast and len(gray) >= 2 and not has_absolute and not any(has_per_image)
        )
        shared_base = None
        if linked_shared:
            shared_base = (min(base_ranges[i][0] for i in gray), max(base_ranges[i][1] for i in gray))
        auto_ranges: list[tuple[float, float]] = []
        use_auto = self.auto_contrast and not has_absolute
        if use_auto:
            for i, frame in enumerate(frames):
                if has_per_image[i] or (i < len(rgb) and rgb[i]):
                    auto_ranges.append(base_ranges[i])
                    continue
                processed = self._signed_log1p(frame) if self.log_scale else frame
                lo, hi = (float(v) for v in np.percentile(processed, (2, 98)))
                # Sparse/clustered data can collapse the 2-98% window to a
                # point; fall back to full extrema like computeAutoRange does.
                full_lo, full_hi = float(processed.min()), float(processed.max())
                if hi - lo <= max(1e-12, abs(full_hi - full_lo) * 1e-6):
                    lo, hi = full_lo, full_hi
                auto_ranges.append((lo, hi))
            if linked_shared:
                shared_auto = (min(auto_ranges[i][0] for i in gray), max(auto_ranges[i][1] for i in gray))
        ranges: list[tuple[float, float]] = []
        for i in range(num):
            if i < len(rgb) and rgb[i]:
                ranges.append((0.0, 1.0))
            elif use_auto and not has_per_image[i]:
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
        from matplotlib import colormaps
        apply_log = self.log_scale if apply_log is None else apply_log
        processed = self._signed_log1p(frame) if apply_log else frame
        if vmax > vmin:
            normalized = np.clip((processed - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            normalized = np.zeros(processed.shape, dtype=np.float64)
        rgba = colormaps.get_cmap(cmap_name)(normalized)
        return (rgba[..., :3] * 255).astype(np.uint8)

    def _static_panel_specs(self) -> list[dict]:
        """Panel plan for the static PNG, mirroring the live widget's layout.

        One entry per visible image panel, plus one signed diff panel per
        non-reference image when ``diff_mode`` is on (the widget renders
        ``ref - other`` with a symmetric range around zero and a diverging
        colormap). Each spec carries the full-resolution frame plus the
        resolved display-space contrast window and a stats line, so the PNG
        renderer only has to bin and colormap.
        """
        frames_source = getattr(self, "_display_data", None)
        if frames_source is None:
            frames_source = getattr(self, "_data", None)
        if frames_source is None or len(frames_source) == 0:
            return []
        if not any(self.is_rgb):
            # Mirror the live canvas/view transport.  Padding and crop are
            # display-only view ops, but saved notebook previews must show the
            # same canvas a user saved after reviewing drift margins.
            frames_source = self._crop_view_stack(np.asarray(frames_source))
            frames_source = self._filtered_frames(frames_source)
            frames_source = self._pad_view_stack(frames_source)
        frames = [frames_source[i] for i in range(len(frames_source))]
        ranges = self._resolve_panel_display_ranges(frames)
        def stats_line(mean: float, lo: float, hi: float, std: float) -> str:
            fmt = self._format_stat
            return f"Mean {fmt(mean)}   Min {fmt(lo)}   Max {fmt(hi)}   Std {fmt(std)}"

        def panel_stats_line(index: int, frame: np.ndarray) -> str:
            stat_arrays = (self.stats_mean, self.stats_min, self.stats_max, self.stats_std)
            if all(index < len(values) for values in stat_arrays):
                return stats_line(
                    self.stats_mean[index],
                    self.stats_min[index],
                    self.stats_max[index],
                    self.stats_std[index],
                )
            data = np.asarray(frame, dtype=np.float64)
            return stats_line(
                float(np.nanmean(data)),
                float(np.nanmin(data)),
                float(np.nanmax(data)),
                float(np.nanstd(data)),
            )

        specs: list[dict] = []
        display_rgb = getattr(self, "_display_rgb", None) or [None] * len(frames)
        for i in self.visible_panels:
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
                "stats": panel_stats_line(i, frame),
                "panel_index": i,
            })
        if self.diff_mode and len(frames) >= 2:
            ref = int(self.diff_reference)
            ref_cmap = self._panel_cmap_for_index(ref)
            diff_cmap = "RdBu" if ref_cmap in self._SEQUENTIAL_CMAPS else ref_cmap
            for other in range(len(frames)):
                # RGB panels never get a diff panel: a signed residual against
                # a display-ready color composite is meaningless.
                if other == ref or (other < len(self.is_rgb) and self.is_rgb[other]):
                    continue
                diff = frames[ref] - frames[other]
                # Symmetric window centers zero on the diverging map's midpoint,
                # so positive and negative residuals read with equal weight.
                sym = float(max(abs(float(diff.min())), abs(float(diff.max())))) or 1.0
                label = ("Diff (A − B)" if len(frames) == 2
                         else f"Diff (#{ref + 1} − #{other + 1})")
                specs.append({
                    "frame": diff,
                    "vmin": -sym,
                    "vmax": sym,
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

    def _static_roi_items(self) -> list[dict[str, Any]]:
        """Return visible ROI dictionaries with defaults for static rendering."""
        if not self.roi_active:
            return []
        rois: list[dict[str, Any]] = []
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
    def _static_roi_extent(roi: dict[str, Any]) -> tuple[float, float]:
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
        roi: dict[str, Any],
        height: int,
        width: int,
        *,
        half_shape: tuple[float, float] | None = None,
    ) -> tuple[slice, slice]:
        """Crop around an ROI for the saved-notebook zoom panel."""
        center_row = float(roi.get("row", height / 2.0))
        center_col = float(roi.get("col", width / 2.0))
        half_h, half_w = half_shape or Show2D._static_roi_extent(roi)
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
        if len(specs) != 1 or len(self.visible_panels) != 1 or self.diff_mode:
            return []
        rois = self._static_roi_items()
        if not rois:
            return []
        source = specs[0]
        frame = source["frame"]
        if frame.ndim < 2:
            return []
        extents = [self._static_roi_extent(roi) for roi in rois]
        common_half_extent = max(max(half_h, half_w) for half_h, half_w in extents)
        common_half_shape = (common_half_extent, common_half_extent)
        zooms: list[dict] = []
        for idx, roi in enumerate(rois, start=1):
            rows, cols = self._static_roi_crop_slices(
                roi,
                frame.shape[0],
                frame.shape[1],
                half_shape=common_half_shape,
            )
            crop = frame[rows, cols] if not source.get("rgb") else frame[rows, cols, :]
            if crop.size == 0:
                continue
            zoom_roi = dict(roi)
            zoom_roi["row"] = float(zoom_roi.get("row", 0.0)) - rows.start
            zoom_roi["col"] = float(zoom_roi.get("col", 0.0)) - cols.start
            zooms.append({
                **source,
                "frame": crop,
                "label": f"ROI {idx} zoom",
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
        crop_h = max(1, _js_round(height / zoom))
        crop_w = max(1, _js_round(width / zoom))
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        return slice(top, top + crop_h), slice(left, left + crop_w)

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

    def _panel_has_scale_bar(self, panel_index: int) -> bool:
        """Return whether a scale bar should be drawn for one panel."""
        if not self.scale_bar_visible:
            return False
        panels = list(getattr(self, "scale_bar_panels", []) or [])
        return not panels or int(panel_index) in {int(panel) for panel in panels}

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
        texts: list[tuple[str, str, str, float]] = []
        for spec in specs:
            panel_index = int(spec.get("panel_index", len(texts)))
            if not self._panel_has_scale_bar(panel_index):
                texts.append((spec["label"], zoom_text, "", 0.0))
                continue
            full_w = spec["frame"].shape[1]
            effective_zoom = zoom * css_px / full_w
            # 60 CSS px target bar, rounded to a nice physical length
            if self.scale_bar_length is not None and float(self.scale_bar_length) > 0:
                nice = float(self.scale_bar_length)
            else:
                nice = _round_to_nice(60.0 / effective_zoom * pixel_size)
            bar_px = nice / pixel_size * effective_zoom
            label = str(self.scale_bar_label or "") or _format_scale_label(nice, unit)
            texts.append((spec["label"], zoom_text,
                          label, bar_px))
        return texts

    @staticmethod
    def _draw_static_roi(
        ax: matplotlib.axes.Axes,
        roi: dict[str, Any],
        *,
        source_rows: slice,
        source_cols: slice,
        bin_h: int,
        bin_w: int,
        point: float,
    ) -> None:
        """Draw one ROI in the saved PNG's image-pixel coordinate system."""
        crop_h = max(1, source_rows.stop - source_rows.start)
        crop_w = max(1, source_cols.stop - source_cols.start)
        scale_y = bin_h / crop_h
        scale_x = bin_w / crop_w
        row = float(roi.get("row", 0.0))
        col = float(roi.get("col", 0.0))
        x = (col - source_cols.start) * scale_x - 0.5
        y = (row - source_rows.start) * scale_y - 0.5
        if x < -bin_w or x > 2 * bin_w or y < -bin_h or y > 2 * bin_h:
            return
        color = _hex_to_rgb01(str(roi.get("color", "#4fc3f7")))
        line_width = max(1.0, float(roi.get("line_width", 2.0))) * point
        stroke = [matplotlib.patheffects.withStroke(
            linewidth=line_width + 1.5 * point,
            foreground=(0, 0, 0, 0.65),
        )]
        shape = str(roi.get("shape", "circle")).lower()
        if shape == "rectangle":
            half_h = max(1.0, float(roi.get("height", 20.0)) / 2.0) * scale_y
            half_w = max(1.0, float(roi.get("width", 20.0)) / 2.0) * scale_x
            patch = matplotlib.patches.Rectangle(
                (x - half_w, y - half_h),
                2 * half_w,
                2 * half_h,
                fill=False,
                edgecolor=color,
                linewidth=line_width,
                path_effects=stroke,
            )
            ax.add_patch(patch)
            return
        radius = max(1.0, float(roi.get("radius", 10.0)))
        width = 2 * radius * scale_x
        height = 2 * radius * scale_y
        if shape == "square":
            patch = matplotlib.patches.Rectangle(
                (x - width / 2.0, y - height / 2.0),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=line_width,
                path_effects=stroke,
            )
            ax.add_patch(patch)
            return
        outer = matplotlib.patches.Ellipse(
            (x, y),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=line_width,
            path_effects=stroke,
        )
        ax.add_patch(outer)
        if shape == "annular":
            inner = max(0.5, float(roi.get("radius_inner", 5.0)))
            ax.add_patch(matplotlib.patches.Ellipse(
                (x, y),
                2 * inner * scale_x,
                2 * inner * scale_y,
                fill=False,
                edgecolor=color,
                linewidth=line_width,
                linestyle="--",
                path_effects=stroke,
            ))

    def _draw_static_inset_plot(
        self,
        ax: matplotlib.axes.Axes,
        spec: dict[str, Any],
        *,
        panel_index: int,
        line_color: str,
    ) -> None:
        """Draw a compact per-panel calibration curve in the saved PNG."""
        if not spec:
            return
        try:
            x = np.asarray(spec.get("x"), dtype=float).ravel()
            y = np.asarray(spec.get("y"), dtype=float).ravel()
        except Exception:
            return
        if x.size != y.size or x.size < 2:
            return
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2:
            return
        x = x[finite]
        y = y[finite]
        xlim = tuple(float(v) for v in spec.get("xlim", (float(x.min()), float(x.max()))))
        ylim = tuple(float(v) for v in spec.get("ylim", (float(y.min()), float(y.max()))))
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
            left, top, width, height = (float(v) for v in spec["box"])
            box_w = max(0.05, min(0.95, width))
            box_h = max(0.05, min(0.95, height))
            x0 = max(0.0, min(1.0 - box_w, left))
            y0 = max(0.0, min(1.0 - box_h, 1.0 - top - box_h))
        else:
            margin = spec.get("margin", (0.035, 0.035))
            if isinstance(margin, (int, float)):
                margin_x = margin_y = float(margin) / 300.0
            else:
                vals = list(margin)
                margin_x = float(vals[0]) / 300.0
                margin_y = float(vals[1]) / 300.0
            margin_x = max(0.0, min(0.45, margin_x))
            margin_y = max(0.0, min(0.45, margin_y))
            if "right" in position:
                x0 = 1.0 - box_w - margin_x
            elif "center" in position:
                x0 = 0.5 - box_w / 2
            else:
                x0 = margin_x
            if "bottom" in position:
                y0 = margin_y + 0.08
            elif "center" in position:
                y0 = 0.5 - box_h / 2
            else:
                y0 = 1.0 - box_h - margin_y
            if self.scale_bar_visible and position == "bottom-right":
                y0 += 0.10
        inset = ax.inset_axes([x0, y0, box_w, box_h])
        background_alpha = max(0.0, min(1.0, float(spec.get("background_alpha", 0.68))))
        inset.set_facecolor(spec.get("background") or (0.04, 0.05, 0.07, background_alpha))
        for spine in inset.spines.values():
            spine.set_color(spec.get("border_color") or (1, 1, 1, 0.35))
            spine.set_linewidth(max(0.0, min(6.0, float(spec.get("border_width", 1.0)))) * 0.6)
        inset.plot(
            x,
            y,
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
                inset.set_xticks([float(v) for v in spec["xticks"]])
            else:
                inset.set_xticks([xlim[0], xlim[1]])
            if "yticks" in spec:
                inset.set_yticks([float(v) for v in spec["yticks"]])
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
        for text_key, pos_key, default_color in (
            ("legend", "legend_position", spec.get("text_color") or spec.get("color") or line_color),
            ("annotation", "annotation_position", text_color),
        ):
            if spec.get(text_key):
                pos = str(spec.get(pos_key, "top-left" if text_key == "legend" else "top-right"))
                x_txt = 0.96 if "right" in pos else 0.04
                y_txt = 0.07 if "top" in pos else 0.93
                inset.text(
                    x_txt,
                    y_txt,
                    str(spec[text_key]),
                    transform=inset.transAxes,
                    ha="right" if "right" in pos else "left",
                    va="top" if "top" in pos else "bottom",
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
        import base64
        import io as _io
        specs = self._static_panel_specs()
        if not specs:
            return None
        chrome = self._gallery_export_chrome()
        gap_px = int(chrome["inter_panel_gap_px"])
        gap_color = str(chrome["inter_panel_gap_color"])
        outer_px = int(chrome["gallery_outer_border_px"])
        outer_color = str(chrome["gallery_outer_border_color"])
        panel_border_px = float(chrome["panel_inner_border_px"])
        panel_border_color = str(chrome["panel_inner_border_color"])
        base_roi_items = self._static_roi_items()
        if base_roi_items:
            specs = [{**spec, "roi_items": base_roi_items} for spec in specs]
            if len(specs) == 1:
                specs.extend(self._static_roi_zoom_specs(specs))
        num = len(specs)
        # Total-pixel budget: a large survey gallery (e.g. 38 panels) at a fixed
        # 512 px/panel produced a ~27 MB PNG and a ~73 MB notebook (noisy STEM
        # content compresses poorly, ~2.7 bytes/px). The fallback is a reopen
        # preview, not the data, so shrink per-panel resolution as the panel
        # count grows (~2 MP total -> a few MB, floor 160 px) instead of
        # scaling the file linearly with the gallery. Small galleries keep the
        # full 512 px/panel.
        budget_px = int((2_000_000 / num) ** 0.5)
        max_px = max(160, min(max_px, budget_px))
        css_w = self._static_canvas_css_px()
        overlays = self._static_overlay_texts(specs, css_px=css_w)
        zoom = min(max(float(self.initial_zoom) or 1.0, 0.5), 20.0)
        has_roi_zoom = any(bool(spec.get("roi_zoom_panel")) for spec in specs)
        ncols = min(num, 4) if has_roi_zoom else max(1, min(self.ncols, num))
        nrows = (num + ncols - 1) // ncols
        # cells sized to the panels' cropped aspect so every image fills its
        # cell exactly: a taller cell would pad panels with white and make the
        # horizontal gutters read wider than the vertical ones
        h0, w0 = specs[0]["frame"].shape[:2]
        rows0, cols0 = self._center_crop_slices(h0, w0, zoom)
        aspect = (rows0.stop - rows0.start) / (cols0.stop - cols0.start)
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
        fig = matplotlib.figure.Figure(
            figsize=(
                fig_w_in,
                fig_h_in,
            )
        )
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
        font_family = _static_overlay_font()
        # the live canvas is css_w CSS px wide but the PNG panel is max_px
        # device px wide, so every CSS-px size renders scaled by max_px/css_w
        point = (max_px / css_w) * 72.0 / dpi  # matplotlib points per CSS px
        for idx, (spec, (label, zoom_text, bar_text, bar_css)) in enumerate(zip(specs, overlays)):
            ax = fig.add_subplot(grid[idx // ncols, idx % ncols])
            ax.axis("off")
            frame = spec["frame"]
            rows, cols = self._center_crop_slices(frame.shape[0], frame.shape[1], zoom)
            if spec.get("rgb"):
                # RGB panels bypass the colormap: bin each channel and pass the
                # display-ready pixels straight through, matching the live view.
                cropped = frame[rows, cols]
                binned_channels = [self._downsample_static_frame(cropped[..., ch], max_px=max_px)
                                   for ch in range(3)]
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
                    linewidth=panel_border_px * point,
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
                    point=point,
                )
            # CSS px -> data px: the panel canvas is css_w CSS px wide showing
            # bin_w image pixels, so overlay geometry scales by bin_w / css_w
            css_h = css_w * bin_h / bin_w
            k = bin_w / css_w

            def css_xy(x_css: float, y_css: float) -> tuple[float, float]:
                return x_css * k - 0.5, y_css * k - 0.5
            # widget textShadow "1px 1px 0 rgba(0,0,0,0.85), 0 0 3px ..." reads
            # as a soft dark outline; a thin translucent black stroke is its
            # closest matplotlib match (a full-opacity stroke looks stenciled)
            stroke = [matplotlib.patheffects.withStroke(
                linewidth=1.5 * point, foreground=(0, 0, 0, 0.8))]
            if label:
                # panel title Box (show2d/index.tsx): top 6px inside the image,
                # centered, bold max(8, panel_title_font_size) px white @ 95%
                title_px = max(8, int(self.panel_title_font_size or 11))
                ax.text(*css_xy(css_w / 2, 6), label, color=(1, 1, 1, 0.95),
                        fontsize=title_px * point, fontweight="bold",
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
                    css_xy(bar_x, bar_y), bar_css * k, 5 * k,
                    facecolor="white", edgecolor="none"))
                ax.text(*css_xy(bar_x + bar_css / 2, bar_y - 4), bar_text,
                        color="white", fontsize=16 * point,
                        fontfamily=font_family, ha="center", va="bottom",
                        path_effects=stroke)
                if self.show_zoom_indicator:
                    zoom_x = css_w - 12 if scale_left else 12
                    zoom_ha = "right" if scale_left else "left"
                    ax.text(*css_xy(zoom_x, css_h - 12 + 5), zoom_text,
                            color="white", fontsize=16 * point,
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
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.05)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # _repr_mimebundle_ / _ipython_display_ / static-fallback sibling plumbing
    # comes from StaticFallbackMixin (utils/static_fallback.py); this class only
    # supplies _static_png_b64 above.

    def _store_static_fallback_preview(self, png_b64: str) -> None:
        """Store a compact saved-notebook preview inside lightweight state.

        JupyterLab can rehydrate a saved anywidget model even when
        ``save_state=False`` stripped the heavy ``frame_bytes``. In the common
        last-expression output path there is no separate static sibling output,
        so the frontend needs a small in-model preview to avoid mounting a blank
        widget after Cmd+S and reopen.
        """
        if getattr(self, "_save_state", False):
            return
        encoded = self._encode_static_fallback_b64(png_b64)
        if not encoded:
            self._static_fallback_jpeg = ""
            self._static_fallback_mime = ""
            return
        mime, image_b64 = encoded
        self._static_fallback_jpeg = image_b64
        self._static_fallback_mime = mime

    @staticmethod
    def _utc_timestamp() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _saved_view_snapshot(self) -> dict[str, Any]:
        """Lightweight inspection state, excluding raw data and nested bookmarks."""
        state = dict(self.state_dict())
        state.pop("saved_view_states", None)
        return state

    def _saved_view_summary(self, state: dict[str, Any] | None = None) -> str:
        """Short microscope-stage style summary shown in menus and notebooks."""
        state = self.state_dict() if state is None else state
        parts: list[str] = []
        if state.get("roi_active") and state.get("roi_list"):
            parts.append(f"ROI {len(state.get('roi_list') or [])}")
        if state.get("show_fft"):
            parts.append("FFT")
        ratios = state.get("pad_ratios") if isinstance(state.get("pad_ratios"), list) else []
        pad = max([float(state.get("pad_ratio") or 0.0)] + [float(v or 0.0) for v in ratios])
        if pad > 0:
            parts.append(f"pad {pad:.0%}")
        if state.get("denoise_enabled") and state.get("denoise") not in (None, "", "none"):
            parts.append(f"denoise {state.get('denoise')}")
        if state.get("frequency_filter_enabled") and state.get("frequency_filter") not in (None, "", "none"):
            parts.append(f"filter {state.get('frequency_filter')}")
        hidden = state.get("hidden_panels") if isinstance(state.get("hidden_panels"), list) else []
        if hidden:
            parts.append(f"{len(hidden)} hidden")
        selected = state.get("selected_idx")
        try:
            parts.append(f"panel {int(selected) + 1}")
        except (TypeError, ValueError):
            pass
        return " · ".join(parts) if parts else "current view"

    @staticmethod
    def _saved_view_name(value: Any) -> str:
        name = str(value or "").strip()
        return name if name else "Untitled view"

    def _normalize_saved_view_states(self, states: Any) -> list[dict[str, Any]]:
        """Return JSON-safe named view states with stable ids and summaries."""
        if not isinstance(states, list):
            return []
        clean: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(states):
            if not isinstance(item, dict):
                continue
            state = item.get("state")
            if not isinstance(state, dict):
                continue
            state = dict(state)
            state.pop("saved_view_states", None)
            raw_id = str(item.get("id") or "").strip()
            entry_id = raw_id if raw_id and raw_id not in seen else f"view-{index + 1}"
            while entry_id in seen:
                entry_id = f"{entry_id}-copy"
            seen.add(entry_id)
            created = str(item.get("created_at") or item.get("updated_at") or self._utc_timestamp())
            updated = str(item.get("updated_at") or created)
            clean.append({
                "id": entry_id,
                "name": self._saved_view_name(item.get("name")),
                "created_at": created,
                "updated_at": updated,
                "summary": str(item.get("summary") or self._saved_view_summary(state)),
                "state": state,
            })
        return clean

    def _saved_view_index(self, name_or_index: str | int) -> int:
        states = self._normalize_saved_view_states(self.saved_view_states)
        if isinstance(name_or_index, int):
            if 0 <= name_or_index < len(states):
                return name_or_index
            raise KeyError(f"no saved Show2D view at index {name_or_index}")
        key = str(name_or_index)
        for idx, entry in enumerate(states):
            if entry["id"] == key or entry["name"] == key:
                return idx
        raise KeyError(f"no saved Show2D view named {key!r}")

    def save_view_state(self, name: str | None = None, *, update: bool = False) -> dict[str, Any]:
        """Save the current lightweight Show2D inspection state.

        This is the programmatic version of the More → Save State button. It
        stores widget/view settings such as ROI, zoom/view box, padding,
        denoise/filter, FFT, selected panel, hidden panels, contrast, and frame
        indices. It never stores raw image arrays.
        """
        states = self._normalize_saved_view_states(self.saved_view_states)
        view_name = self._saved_view_name(name or f"View {len(states) + 1}")
        now = self._utc_timestamp()
        state = self._saved_view_snapshot()
        entry = {
            "id": "",
            "name": view_name,
            "created_at": now,
            "updated_at": now,
            "summary": self._saved_view_summary(state),
            "state": state,
        }
        target_idx = next((idx for idx, item in enumerate(states) if item["name"] == view_name), None)
        if update and target_idx is not None:
            entry["id"] = states[target_idx]["id"]
            entry["created_at"] = states[target_idx].get("created_at", now)
            states[target_idx] = entry
            self.saved_view_status = f"Updated state {view_name}"
        else:
            slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in view_name).strip("-") or "view"
            existing = {item["id"] for item in states}
            entry_id = slug
            suffix = 2
            while entry_id in existing:
                entry_id = f"{slug}-{suffix}"
                suffix += 1
            entry["id"] = entry_id
            states.append(entry)
            self.saved_view_status = f"Saved state {view_name}"
        self.saved_view_states = states
        return dict(entry)

    def load_view_state(self, name_or_index: str | int) -> Self:
        """Restore a named saved inspection state on the current data."""
        states = self._normalize_saved_view_states(self.saved_view_states)
        idx = self._saved_view_index(name_or_index)
        entry = states[idx]
        state = dict(entry.get("state") or {})
        state.pop("saved_view_states", None)
        self.load_state_dict(state)
        self.saved_view_states = states
        self.saved_view_status = f"Loaded state {entry['name']}"
        return self

    def delete_view_state(self, name_or_index: str | int) -> Self:
        """Delete one saved inspection state."""
        states = self._normalize_saved_view_states(self.saved_view_states)
        idx = self._saved_view_index(name_or_index)
        name = states[idx]["name"]
        del states[idx]
        self.saved_view_states = states
        self.saved_view_status = f"Deleted state {name}"
        return self

    def clear_view_states(self) -> Self:
        """Delete all saved inspection states."""
        count = len(self._normalize_saved_view_states(self.saved_view_states))
        self.saved_view_states = []
        self.saved_view_status = f"Deleted {count} saved state{'s' if count != 1 else ''}"
        return self

    def state_dict(self):
        return {
            "title": self.title,
            "show_title": self.show_title,
            "cmap": self.cmap,
            "panel_cmaps": list(self.panel_cmaps),
            "panel_cmaps_memory": list(self.panel_cmaps_memory),
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "contrast_preset": self.contrast_preset,
            "histogram_advanced": self.histogram_advanced,
            "show_histogram_advanced": self.show_histogram_advanced,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "identity_colors": list(self.identity_colors),
            "marker_style": self.marker_style,
            "row_markers": dict(self.row_markers),
            "col_markers": dict(self.col_markers),
            "labels": list(self.labels),
            "panel_title_spans": list(self.panel_title_spans),
            "panel_annotations": list(self.panel_annotations),
            "panel_overlays": list(self.panel_overlays),
            "starred": list(self.starred),
            "n_pages": int(self.n_pages),
            "page_idx": int(self.page_idx),
            "panels_per_page": int(self.panels_per_page),
            "page_kind": str(self.page_kind),
            "folder_page_size": getattr(self, "_folder_page_size", None),
            "page_labels": list(self.page_labels),
            "page_starred": list(self.page_starred),
            "hidden_panels": list(self.hidden_panels),
            "hidden_page_slots": list(self.hidden_page_slots),
            "panel_order": list(self.panel_order),
            "panel_frame_indices": list(self.panel_frame_indices),
            "panel_playback_fps": float(self.panel_playback_fps),
            "show_panel_titles": self.show_panel_titles,
            "panel_title_font_size": self.panel_title_font_size,
            "panel_title_spans": list(self.panel_title_spans),
            "panel_title_style": dict(self.panel_title_style),
            "inter_panel_gap_px": int(self.inter_panel_gap_px),
            "inter_panel_gap_color": str(self.inter_panel_gap_color),
            "gallery_outer_border_px": int(self.gallery_outer_border_px),
            "gallery_outer_border_color": str(self.gallery_outer_border_color),
            "panel_inner_border_px": float(self.panel_inner_border_px),
            "panel_inner_border_color": str(self.panel_inner_border_color),
            "gallery_gap_px": int(self.gallery_gap_px),
            "gallery_gap_color": str(self.gallery_gap_color),
            "show_stats": self.show_stats,
            "debug": self.debug,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "fft_metrics": self.fft_metrics,
            "show_controls": self.show_controls,
            "controls_collapsed": self.controls_collapsed,
            "pixel_size": self.pixel_size,
            "pixel_sizes": list(self.pixel_sizes),
            "pixel_unit": self.pixel_unit,
            "scale_bar_visible": self.scale_bar_visible,
            "scale_bar_position": self.scale_bar_position,
            "scale_bar_panels": list(self.scale_bar_panels),
            "scale_bar_length": self.scale_bar_length,
            "scale_bar_label": self.scale_bar_label,
            "scale_bar_style": dict(self.scale_bar_style),
            "show_zoom_indicator": self.show_zoom_indicator,
            "size": self.size,
            "smooth": self.smooth,
            "initial_zoom": self.initial_zoom,
            "flip_rows": self.flip_rows,
            "flip_cols": self.flip_cols,
            "vmins": self.vmins,
            "vmaxs": self.vmaxs,
            "link_zoom": self.link_zoom,
            "link_pan": self.link_pan,
            "link_contrast": self.link_contrast,
            "zoom_row": self.zoom_row,
            "zoom_col": self.zoom_col,
            "view_box": list(self.view_box),
            "view_crop": list(self.view_crop),
            "pad_ratio": float(self.pad_ratio),
            "pad_ratios": list(self.pad_ratios),
            "pad_fill_mode": self.pad_fill_mode,
            "pad_fill_modes": list(self.pad_fill_modes),
            "pad_scope": self.pad_scope,
            "diff_mode": self.diff_mode,
            # Which panel the signed-diff panel subtracts from; without it a
            # saved non-default diff reference silently reverts to panel 0.
            "diff_reference": int(self.diff_reference),
            "ncols": self.ncols,
            "selected_idx": self.selected_idx,
            "marker_colors": list(self.marker_colors),
            "selected_panels": list(self.selected_panels),
            "inset_plots": list(self.inset_plots),
            "show_inset_plots": self.show_inset_plots,
            "roi_active": self.roi_active,
            "roi_list": self.roi_list,
            "roi_selected_idx": self.roi_selected_idx,
            "profile_line": self.profile_line,
            "image_rotations": list(self.image_rotations),
            "rotation_scope": self.rotation_scope,
            "image_flips_horizontal": list(self.image_flips_horizontal),
            "image_flips_vertical": list(self.image_flips_vertical),
            "display_bin": self._display_bin,
            "denoise": self.denoise,
            "denoise_sigma": self.denoise_sigma,
            "denoise_bin": self.denoise_bin,
            "denoise_scope": self.denoise_scope,
            "show_denoise": self.show_denoise,
            "denoise_enabled": self.denoise_enabled,
            "denoise_modes": list(self.denoise_modes),
            "denoise_sigmas": list(self.denoise_sigmas),
            "denoise_bins": list(self.denoise_bins),
            "frequency_filter": self.frequency_filter,
            "frequency_filter_enabled": self.frequency_filter_enabled,
            "frequency_filter_cutoff": self.frequency_filter_cutoff,
            "frequency_filter_center": self.frequency_filter_center,
            "frequency_filter_width": self.frequency_filter_width,
            "frequency_filter_modes": list(self.frequency_filter_modes),
            "frequency_filter_cutoffs": list(self.frequency_filter_cutoffs),
            "frequency_filter_centers": list(self.frequency_filter_centers),
            "frequency_filter_widths": list(self.frequency_filter_widths),
            "frequency_filter_scope": self.frequency_filter_scope,
            "show_frequency_filter": self.show_frequency_filter,
            # underlay= is construction-time sugar: the composed "map on HAADF"
            # RGB panel is rebuilt from the kernel, not stored, so it is NOT
            # restored on a kernel-less reopen. These blend knobs are kept so a
            # live (kernel-backed) widget re-blends to the saved look; on a cold
            # reopen they are inert (there is no underlay panel to tune).
            "underlay_alpha": self.underlay_alpha,
            "underlay_haadf_gain": self.underlay_haadf_gain,
            "underlay_mode": self.underlay_mode,
            "stretch_percentiles": list(self.stretch_percentiles),
            "display_gamma": self.display_gamma,
            "dual_gain": list(self.dual_gain),
            "saved_view_states": self._normalize_saved_view_states(self.saved_view_states),
        }

    def save(self, path: str):
        save_state_file(path, "Show2D", self.state_dict())

    _HTML_EXPORT_SAFE_MB = 80.0

    def _estimate_html_export_mb(self, *, quantized: bool, downsample: int) -> float:
        """Rough MB an embedded single-file Show2D HTML would occupy."""
        downsample_factor = max(1, int(downsample))
        has_local_stacks = any(count > 1 for count in self.panel_frame_counts)
        if has_local_stacks and getattr(self, "_display_panel_stacks", None):
            elements = sum(int(np.prod(stack.shape)) for stack in self._display_panel_stacks)
        else:
            data = self._display_data if self._display_data is not None else self._data
            elements = int(np.prod(data.shape)) if data is not None else 0
        elements //= downsample_factor**2
        bytes_per = 1 if quantized else 4
        payload_mb = elements * bytes_per * (4.0 / 3.0) / (1024 * 1024)
        return payload_mb + 2.0

    def export_html(self, path: str | pathlib.Path | None = None,
                    *,
                    title: str | None = None,
                    mode: str = "single",
                    encoding: str = "full",
                    downsample: int | None = None,
                    quantized: bool | None = None,
                    max_mb: float | None = _HTML_EXPORT_SAFE_MB) -> pathlib.Path:
        """Write a standalone HTML viewer for this widget.

        The exported file mounts the live anywidget JS bundle with the current
        widget state (data, labels, cmap, vmin/vmax, log_scale, sampling, ...).
        Opens in any browser without a Jupyter kernel.
        Preferred export options are ``mode="single"``, ``encoding="full"`` or
        ``encoding="uint8"``, and ``downsample=None``. Use ``downsample=2`` /
        ``4`` / ``8`` with ``encoding="uint8"`` for compact visual reports.
        ``quantized`` is kept as a compatibility alias for ``encoding="uint8"``.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination HTML path.
        quantized : bool, default False
            Store the displayed image stack as uint8 with min/max metadata.
            This is smaller and visually equivalent after colormapping. The
            default stores exact float32 display values.
        title : str, optional
            Browser page title. Defaults to widget ``title`` or "Show2D".
        """
        if self._data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        if any(self.is_rgb):
            raise NotImplementedError(
                "export_html is not supported when the gallery contains RGB panels; "
                "the export clone rebuilds from the grayscale stack and would drop the color channels."
            )

        export_mode, quantized, downsample_factor = self._normalise_html_export_options(
            mode=mode,
            encoding=encoding,
            downsample=downsample,
            quantized=quantized,
        )
        if export_mode == "single" and max_mb is not None:
            estimate_mb = self._estimate_html_export_mb(
                quantized=quantized,
                downsample=downsample_factor,
            )
            if estimate_mb > float(max_mb):
                uint8_mb = self._estimate_html_export_mb(
                    quantized=True,
                    downsample=downsample_factor,
                )
                raise ValueError(
                    f"This export would embed about {estimate_mb:.0f} MB into one HTML file, "
                    f"above the {float(max_mb):.0f} MB safe limit (large single-file exports often "
                    f"fail to open under Chrome file://). Options: encoding='uint8' "
                    f"(about {uint8_mb:.0f} MB), downsample=2 or 4 to shrink spatially, "
                    f"mode='folder' for a thin HTML plus nearby data folder, or pass "
                    f"max_mb={estimate_mb:.0f} to force this size."
                )
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path(
            quantized,
            downsample=downsample_factor,
        )
        self._write_html_export(
            export_path,
            quantized=quantized,
            title=title,
            mode=export_mode,
            downsample=downsample_factor,
        )
        size_mb = export_path.stat().st_size / (1024 * 1024)
        label = self._export_mode_label(quantized, downsample=downsample_factor)
        mode_label = "folder, " if export_mode == "folder" else ""
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, {mode_label}{label})"
        return export_path

    def _normalise_html_export_options(
        self,
        *,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | None = None,
        quantized: bool | None = None,
    ) -> tuple[str, bool, int]:
        raw_mode = str(mode or "single").strip().lower().replace("_", "-")
        if raw_mode in {"exact", "full"}:
            raw_mode = "single"
            encoding = "full"
        elif raw_mode in {"quantized", "uint8", "u8"}:
            raw_mode = "single"
            encoding = "uint8"
        elif raw_mode in {"sidecar", "linked-folder", "linked-data-folder"}:
            raw_mode = "folder"
        if raw_mode not in {"single", "folder"}:
            raise ValueError("Show2D HTML export supports mode='single' or mode='folder'")
        if downsample in (None, "", 0, "0"):
            downsample_factor = 1
        else:
            if isinstance(downsample, bool):
                raise ValueError("Show2D HTML export downsample must be an integer factor, not bool")
            downsample_factor = int(downsample)
        if downsample_factor < 1:
            raise ValueError(f"Show2D HTML export downsample must be >= 1, got {downsample!r}")
        if downsample_factor not in {1, 2, 4, 8}:
            raise ValueError("Show2D HTML export downsample must be one of 1, 2, 4, or 8")
        raw_encoding = str(encoding or "full").strip().lower().replace("_", "-")
        if quantized is True:
            raw_encoding = "uint8"
        elif quantized is False and raw_encoding in {"quantized", "uint8", "u8"}:
            raw_encoding = "uint8"
        if raw_encoding in {"full", "exact", "float32", "f32"}:
            if downsample_factor != 1:
                raise ValueError("Show2D exact float32 HTML export does not support downsample; use encoding='uint8'")
            return raw_mode, False, downsample_factor
        if raw_encoding in {"uint8", "u8", "quantized"}:
            return raw_mode, True, downsample_factor
        raise ValueError(f"unknown Show2D export encoding {encoding!r}; expected 'full' or 'uint8'")

    def _on_export_request_change(self, change: dict) -> None:
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            mode = str(payload.get("mode", "exact"))
            if mode == "clear":
                self.export_payload = b""
                self.export_payload_id = ""
                self.export_filename = ""
                return
            export_mode, quantized, downsample_factor = self._normalise_html_export_options(
                mode=mode,
                encoding=str(payload.get("encoding", "full")),
                downsample=payload.get("downsample"),
                quantized=None,
            )
            if payload.get("download"):
                filename = str(payload.get("filename") or self._default_html_export_path(
                    quantized,
                    downsample=downsample_factor,
                ).name)
                request_id = str(payload.get("id") or "")
                self.export_status = f"Preparing {filename}..."
                html = self._html_export_bytes(quantized=quantized, downsample=downsample_factor)
                self.export_filename = filename
                self.export_payload = html
                self.export_payload_id = request_id
                size_mb = len(html) / (1024 * 1024)
                label = self._export_mode_label(quantized, downsample=downsample_factor)
                self.export_status = f"Ready {filename} ({size_mb:.1f} MB, {label})"
            else:
                self.export_status = f"Exporting {mode} HTML..."
                self.export_html(mode=export_mode, quantized=quantized, downsample=downsample_factor)
        except Exception as exc:
            self.export_status = f"Export failed: {exc}"

    def _on_saved_view_request_change(self, change: dict) -> None:
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            action = str(payload.get("action", "")).lower()
            name = payload.get("name")
            key = payload.get("id") or name
            if action == "save":
                self.save_view_state(str(name or ""), update=False)
            elif action == "update":
                self.save_view_state(str(name or key or ""), update=True)
            elif action == "load":
                self.load_view_state(str(key))
            elif action in {"delete", "remove"}:
                self.delete_view_state(str(key))
            elif action in {"clear", "delete_all"}:
                self.clear_view_states()
            else:
                self.saved_view_status = f"State action failed: unknown action {action!r}"
        except Exception as exc:
            self.saved_view_status = f"State action failed: {exc}"

    def _default_html_export_path(self, quantized: bool, *, downsample: int = 1) -> pathlib.Path:
        label = self.title.strip() or "show2d"
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        if not slug:
            slug = "show2d"
        mode = "quantized" if quantized else "exact"
        suffix = f"_{int(downsample)}xdownsample" if quantized and int(downsample) > 1 else ""
        shape = f"{self.n_images}x{self.height}x{self.width}" if self.n_images > 1 else f"{self.height}x{self.width}"
        return pathlib.Path.cwd() / f"{slug}_{shape}_{mode}{suffix}.html"

    def _export_mode_label(self, quantized: bool, *, downsample: int = 1) -> str:
        if not quantized:
            return "full float32"
        if int(downsample) > 1:
            return f"uint8, {int(downsample)}x downsample"
        return "uint8"

    def _write_html_export(
        self,
        path: str | pathlib.Path,
        *,
        quantized: bool,
        title: str | None = None,
        mode: str = "single",
        downsample: int = 1,
    ) -> pathlib.Path:
        from ipywidgets.embed import dependency_state, embed_minimal_html

        from .export import ensure_mobile_viewport

        export_path = pathlib.Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        page_title = title or self.title or "Show2D"
        has_local_stacks = any(count > 1 for count in self.panel_frame_counts)
        if mode == "folder" and not has_local_stacks and not any(self.is_rgb):
            self._write_html_folder_export_fast(
                export_path,
                quantized=quantized,
                title=page_title,
            )
            ensure_mobile_viewport(export_path)
            return export_path
        export_widget = self._clone_for_html_export(quantized=quantized, downsample=downsample)
        try:
            if mode == "folder":
                data_dir = export_path.parent / f"{export_path.stem}_files"
                data_dir.mkdir(parents=True, exist_ok=True)
                frame_name = "frame_bytes.bin"
                frame_path = data_dir / frame_name
                frame_path.write_bytes(bytes(export_widget.frame_bytes))
                export_widget.frame_bytes = b""
                export_widget.frame_bytes_url = f"{data_dir.name}/{frame_name}"
                export_widget.frame_bytes_urls = []
                panel_stack_size = 0
                if export_widget.panel_stack_bytes:
                    stack_name = "panel_stack_bytes.bin"
                    stack_path = data_dir / stack_name
                    stack_path.write_bytes(bytes(export_widget.panel_stack_bytes))
                    panel_stack_size = stack_path.stat().st_size
                    export_widget.panel_stack_bytes = b""
                    export_widget.panel_stack_bytes_url = f"{data_dir.name}/{stack_name}"
                else:
                    export_widget.panel_stack_bytes_url = ""
                (data_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "format": "quantem.widget.show2d.folder.v1",
                            "n_images": int(export_widget.n_images),
                            "height": int(export_widget.height),
                            "width": int(export_widget.width),
                            "encoding": "uint8" if quantized else "float32",
                            "frame_bytes": frame_name,
                            "frame_bytes_size": frame_path.stat().st_size,
                            "panel_stack_bytes": "panel_stack_bytes.bin" if panel_stack_size else "",
                            "panel_stack_bytes_size": panel_stack_size,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            else:
                export_widget.frame_bytes_url = ""
                export_widget.frame_bytes_urls = []
                export_widget.panel_stack_bytes_url = ""
            state = dependency_state([export_widget], drop_defaults=False)
            embed_minimal_html(
                str(export_path),
                views=[export_widget],
                title=page_title,
                drop_defaults=False,
                state=state,
            )
        finally:
            export_widget.close()
        ensure_mobile_viewport(export_path)
        return export_path

    def _folder_frame_payload(self, *, quantized: bool) -> tuple[bytes, list[float], list[float]]:
        """Return packed display bytes and dequantization ranges for folder export."""
        data = self._display_data if self._display_data is not None else self._data
        data = self._crop_view_stack(data)
        data = self._filtered_frames(data)
        data = self._pad_view_stack(data)
        if quantized:
            if np.asarray(data).dtype == np.uint8:
                arr = np.ascontiguousarray(data)
                mins = [0.0] * int(arr.shape[0])
                maxs = [255.0] * int(arr.shape[0])
                return _b64_safe(arr.tobytes()), mins, maxs
            arr = np.ascontiguousarray(data, dtype=np.float32)
            flat = arr.reshape(arr.shape[0], -1)
            out = np.empty(flat.shape, dtype=np.uint8)
            mins, maxs = [], []
            for i in range(flat.shape[0]):
                finite = flat[i][np.isfinite(flat[i])]
                lo = float(finite.min()) if finite.size else 0.0
                hi = float(finite.max()) if finite.size else 1.0
                rng = hi - lo if hi > lo else 1.0
                out[i] = np.clip((flat[i] - lo) * (255.0 / rng), 0, 255).astype(np.uint8)
                mins.append(lo)
                maxs.append(hi)
            return _b64_safe(out.tobytes()), mins, maxs
        arr = np.ascontiguousarray(data, dtype=np.float32)
        return _b64_safe(arr.tobytes()), [], []

    def _write_html_folder_export_fast(
        self,
        export_path: pathlib.Path,
        *,
        quantized: bool,
        title: str,
    ) -> None:
        """Write folder-mode HTML without cloning or embedding pixel buffers."""
        from ipywidgets.embed import dependency_state, embed_minimal_html

        data_dir = export_path.parent / f"{export_path.stem}_files"
        data_dir.mkdir(parents=True, exist_ok=True)
        frame_name = "frame_bytes.bin"
        frame_bytes, mins, maxs = self._folder_frame_payload(quantized=quantized)
        per_frame_bytes = int(self.height) * int(self.width)
        identity_uint8 = (
            quantized
            and per_frame_bytes > 0
            and len(frame_bytes) >= int(self.n_images) * per_frame_bytes
            and len(mins) == int(self.n_images)
            and len(maxs) == int(self.n_images)
            and all(lo == 0.0 and hi == 255.0 for lo, hi in zip(mins, maxs, strict=False))
        )
        frame_file_names: list[str] = []
        frame_bytes_size = 0
        frame_path: pathlib.Path | None = None
        if identity_uint8 and int(self.n_images) > 1:
            view = memoryview(frame_bytes)
            for image_index in range(int(self.n_images)):
                name = f"frame_{image_index:06d}.bin"
                start = image_index * per_frame_bytes
                stop = start + per_frame_bytes
                (data_dir / name).write_bytes(view[start:stop])
                frame_file_names.append(name)
            frame_name = ""
            frame_bytes_size = sum((data_dir / name).stat().st_size for name in frame_file_names)
        else:
            frame_path = data_dir / frame_name
            frame_path.write_bytes(frame_bytes)
            frame_bytes_size = frame_path.stat().st_size

        old_values = {
            "_save_state": getattr(self, "_save_state", False),
            "offline": self.offline,
            "_export_light": self._export_light,
            "frame_bytes": self.frame_bytes,
            "frame_bytes_url": self.frame_bytes_url,
            "frame_bytes_urls": list(self.frame_bytes_urls),
            "panel_stack_bytes": self.panel_stack_bytes,
            "panel_stack_bytes_url": self.panel_stack_bytes_url,
            "_offline_mins": list(self._offline_mins),
            "_offline_maxs": list(self._offline_maxs),
            "_offline_min": self._offline_min,
            "_offline_max": self._offline_max,
            "_webgpu_filter_ok": self._webgpu_filter_ok,
            "export_enabled": self.export_enabled,
            "export_status": self.export_status,
            "export_payload": self.export_payload,
            "export_payload_id": self.export_payload_id,
            "export_filename": self.export_filename,
            "handoff_enabled": self.handoff_enabled,
            "handoff_status": self.handoff_status,
            "handoff_request": self.handoff_request,
        }
        try:
            self._save_state = True
            self.offline = quantized
            self._export_light = True
            self.frame_bytes = b""
            self.frame_bytes_url = f"{data_dir.name}/{frame_name}" if frame_name else ""
            self.frame_bytes_urls = [f"{data_dir.name}/{name}" for name in frame_file_names]
            self.panel_stack_bytes = b""
            self.panel_stack_bytes_url = ""
            self._offline_mins = mins if quantized else []
            self._offline_maxs = maxs if quantized else []
            self._offline_min = mins[0] if mins else 0.0
            self._offline_max = maxs[0] if maxs else 1.0
            self._webgpu_filter_ok = True
            self.export_enabled = False
            self.export_status = ""
            self.export_payload = b""
            self.export_payload_id = ""
            self.export_filename = ""
            self.handoff_enabled = False
            self.handoff_status = ""
            self.handoff_request = ""
            state = dependency_state([self], drop_defaults=False)
            embed_minimal_html(
                str(export_path),
                views=[self],
                title=title,
                drop_defaults=False,
                state=state,
            )
        finally:
            for name, value in old_values.items():
                setattr(self, name, value)
        (data_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "quantem.widget.show2d.folder.v1",
                    "n_images": int(self.n_images),
                    "height": int(self.height),
                    "width": int(self.width),
                    "encoding": "uint8" if quantized else "float32",
                    "frame_bytes": frame_name,
                    "frame_bytes_files": frame_file_names,
                    "frame_bytes_size": frame_bytes_size,
                    "panel_stack_bytes": "",
                    "panel_stack_bytes_size": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _html_export_bytes(self, *, quantized: bool, downsample: int = 1) -> bytes:
        with tempfile.TemporaryDirectory(prefix="show2d-export-") as tmp:
            path = pathlib.Path(tmp) / self._default_html_export_path(
                quantized,
                downsample=downsample,
            ).name
            self._write_html_export(path, quantized=quantized, downsample=downsample)
            return path.read_bytes()

    def _clone_for_html_export(self, *, quantized: bool, downsample: int = 1) -> Self:
        if any(self.is_rgb):
            raise NotImplementedError(
                "HTML export is not supported when the gallery contains RGB panels; "
                "the export clone rebuilds from the grayscale stack and would drop the color channels."
            )
        data = self._display_data if self._display_data is not None else self._data
        if data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        downsample = int(downsample)

        def downsample_frame(frame: np.ndarray) -> np.ndarray:
            arr = np.ascontiguousarray(frame, dtype=np.float32)
            if downsample <= 1:
                return arr
            from quantem.widget.utils.array import bin2d
            return np.ascontiguousarray(bin2d(arr, factor=downsample, mode="mean"), dtype=np.float32)

        def downsample_stack(stack: np.ndarray) -> np.ndarray:
            arr = np.ascontiguousarray(stack, dtype=np.float32)
            if downsample <= 1:
                return arr
            return np.ascontiguousarray(
                np.stack([downsample_frame(frame) for frame in arr], axis=0),
                dtype=np.float32,
            )

        has_local_stacks = any(count > 1 for count in self.panel_frame_counts)
        if has_local_stacks:
            display_stacks = getattr(self, "_display_panel_stacks", None)
            if not display_stacks:
                raise ValueError("Cannot export local panel stacks after their data has been freed")
            export_data = [
                downsample_stack(stack) if stack.shape[0] > 1 else downsample_frame(stack[0])
                for stack in display_stacks
            ]
        else:
            export_data = downsample_stack(data)
        export_pixel_size = self.pixel_size * downsample if self.pixel_size > 0 else self.pixel_size
        clone = type(self)(
            export_data,
            labels=list(self.labels),
            page_labels=list(self.page_labels) if self.n_pages > 1 else None,
            title=self.title,
            show_title=self.show_title,
            cmap=list(self.panel_cmaps) if self.panel_cmaps else self.cmap,
            sampling=export_pixel_size if export_pixel_size > 0 else None,
            units=self.pixel_unit,
            scale_bar_visible=self.scale_bar_visible,
            scale_bar_position=self.scale_bar_position,
            scale_bar_panels=list(self.scale_bar_panels),
            scale_bar_length=self.scale_bar_length,
            scale_bar_label=self.scale_bar_label,
            scale_bar_style=dict(self.scale_bar_style),
            show_zoom_indicator=self.show_zoom_indicator,
            show_fft=self.show_fft,
            fft_window=self.fft_window,
            fft_metrics=self.fft_metrics,
            show_controls=self.show_controls,
            controls_collapsed=self.controls_collapsed,
            show_stats=self.show_stats,
            debug=self.debug,
            verbose=False,
            log_scale=self.log_scale,
            auto_contrast=self.auto_contrast,
            offline=quantized,
            vmin=self.vmin if self.vmin is not None else self.vmins,
            vmax=self.vmax if self.vmax is not None else self.vmaxs,
            ncols=self.ncols,
            panel_frame_indices=list(self.panel_frame_indices),
            panel_playback_fps=self.panel_playback_fps,
            size=self.size,
            smooth=self.smooth,
            zoom=self.initial_zoom,
            zoom_row=self.zoom_row,
            zoom_col=self.zoom_col,
            link_zoom=self.link_zoom,
            link_pan=self.link_pan,
            link_contrast=self.link_contrast,
            diff_mode=self.diff_mode,
            hidden_panels=list(self.hidden_panels),
            starred=[i for i, value in enumerate(self.starred) if value],
            panel_order=list(self.panel_order),
            show_panel_titles=self.show_panel_titles,
            panel_title_font_size=self.panel_title_font_size,
            panel_title_style=dict(self.panel_title_style),
            inter_panel_gap_px=int(self.inter_panel_gap_px),
            inter_panel_gap_color=str(self.inter_panel_gap_color),
            gallery_outer_border_px=int(self.gallery_outer_border_px),
            gallery_outer_border_color=str(self.gallery_outer_border_color),
            panel_inner_border_px=float(self.panel_inner_border_px),
            panel_inner_border_color=str(self.panel_inner_border_color),
            gallery_gap_px=int(self.gallery_gap_px),
            gallery_gap_color=str(self.gallery_gap_color),
            row_markers=dict(self.row_markers),
            col_markers=dict(self.col_markers),
            panel_annotations=list(self.panel_annotations),
            panel_overlays=list(self.panel_overlays),
            display_bin=1,
        )
        clone.pixel_sizes = list(self.pixel_sizes)
        clone.page_kind = str(self.page_kind)
        clone.n_pages = int(self.n_pages)
        clone.panels_per_page = int(self.panels_per_page)
        clone.page_labels = list(self.page_labels)
        clone.page_idx = int(self.page_idx)
        clone.page_starred = list(self.page_starred)
        clone.load_state_dict(self.state_dict())
        clone.offline = quantized
        clone._export_light = True
        clone._save_state = True
        clone.export_enabled = False
        clone.export_status = ""
        clone.export_payload = b""
        clone.export_payload_id = ""
        clone.export_filename = ""
        clone.handoff_enabled = False
        clone.handoff_status = ""
        clone.handoff_request = ""
        clone.prepared_view = None
        clone.prepared_view_widget = None
        # Exported pages ship RAW frames plus the filter knobs: the browser
        # port (WGSL compute, CPU fallback without WebGPU) filters client-side
        # so sigma scrubs live with no kernel. Panels on non portable modes
        # (tv) still bake their Python-filtered pixels because
        # _panel_browser_filtered is per panel.
        clone._webgpu_filter_ok = True
        clone._update_all_frames()
        return clone

    def load_state_dict(self, state):
        state = dict(state)
        page_kind = str(state.get("page_kind", self.page_kind))
        if page_kind not in {"comparison", "items"}:
            page_kind = str(self.page_kind)
            state.pop("page_kind", None)
        try:
            saved_n_pages = max(1, int(state.get("n_pages", self.n_pages)))
            saved_panels_per_page = max(
                0,
                int(state.get("panels_per_page", self.panels_per_page)),
            )
        except (TypeError, ValueError):
            saved_n_pages = int(self.n_pages)
            saved_panels_per_page = int(self.panels_per_page)
            state.pop("n_pages", None)
            state.pop("panels_per_page", None)
        n_images = int(self.n_images)
        valid_pages = saved_n_pages == 1 and saved_panels_per_page == 0
        if saved_n_pages > 1 and saved_panels_per_page > 0:
            if page_kind == "items":
                valid_pages = (
                    saved_n_pages == math.ceil(n_images / saved_panels_per_page)
                )
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
        if page_kind == "items" and hasattr(self, "_folder_source"):
            try:
                self._folder_page_size = _validate_folder_page_size(
                    folder_page_size,
                )
            except (TypeError, ValueError):
                pass
        # A crop saved from a single-panel session cannot apply to a gallery.
        if int(self.n_images) != 1:
            state.pop("view_crop", None)
        for key in ("pad_ratios", "pad_fill_modes"):
            if key in state and isinstance(state[key], list) and len(state[key]) != int(self.n_images):
                state.pop(key)
        if "panel_title_spans" in state and isinstance(state["panel_title_spans"], list):
            if len(state["panel_title_spans"]) not in (0, int(self.n_images)):
                state.pop("panel_title_spans")
        if "panel_title_style" in state:
            try:
                state["panel_title_style"] = _normalize_panel_title_style(state["panel_title_style"])
            except (TypeError, ValueError):
                state.pop("panel_title_style")
        if "gallery_gap_px" in state:
            try:
                state["gallery_gap_px"] = max(0, int(state["gallery_gap_px"]))
            except (TypeError, ValueError):
                state.pop("gallery_gap_px")
        if "gallery_gap_color" in state and state["gallery_gap_color"] is not None:
            state["gallery_gap_color"] = str(state["gallery_gap_color"])
        legacy_gap_color = str(state.get("gallery_gap_color") or "")
        if "inter_panel_gap_px" not in state and "gallery_gap_px" in state:
            state["inter_panel_gap_px"] = state["gallery_gap_px"]
        if "inter_panel_gap_color" not in state and "gallery_gap_color" in state:
            state["inter_panel_gap_color"] = state["gallery_gap_color"]
        if "gallery_outer_border_px" not in state and legacy_gap_color and "gallery_gap_px" in state:
            state["gallery_outer_border_px"] = state["gallery_gap_px"]
        if "gallery_outer_border_color" not in state and legacy_gap_color:
            state["gallery_outer_border_color"] = legacy_gap_color
        if "panel_inner_border_color" not in state and legacy_gap_color:
            state["panel_inner_border_color"] = legacy_gap_color
        for key in ("inter_panel_gap_px", "gallery_outer_border_px"):
            if key in state:
                try:
                    state[key] = _nonnegative_int(state[key], name=key)
                except ValueError:
                    state.pop(key)
        if "panel_inner_border_px" in state:
            try:
                state["panel_inner_border_px"] = _nonnegative_float(
                    state["panel_inner_border_px"],
                    name="panel_inner_border_px",
                )
            except ValueError:
                state.pop("panel_inner_border_px")
        for key in (
            "inter_panel_gap_color",
            "gallery_outer_border_color",
            "panel_inner_border_color",
        ):
            if key in state and state[key] is not None:
                state[key] = str(state[key])
        if "inter_panel_gap_px" in state:
            state["gallery_gap_px"] = state["inter_panel_gap_px"]
        if "inter_panel_gap_color" in state:
            state["gallery_gap_color"] = state["inter_panel_gap_color"]
        if "scale_bar_style" in state:
            try:
                state["scale_bar_style"] = _normalize_scale_bar_style(state["scale_bar_style"])
            except (TypeError, ValueError):
                state.pop("scale_bar_style")
        if "saved_view_states" in state:
            state["saved_view_states"] = self._normalize_saved_view_states(state["saved_view_states"])
        if "page_idx" in state:
            try:
                state["page_idx"] = int(max(0, min(int(state["page_idx"]), int(self.n_pages) - 1)))
            except (TypeError, ValueError):
                state.pop("page_idx")
        if "page_starred" in state and isinstance(state["page_starred"], list):
            if len(state["page_starred"]) != int(self.n_pages):
                state.pop("page_starred")
        if "page_labels" in state and isinstance(state["page_labels"], list):
            if len(state["page_labels"]) != int(self.n_pages):
                state.pop("page_labels")
        if "starred" in state and isinstance(state["starred"], list) and len(state["starred"]) != int(self.n_images):
            state.pop("starred")
        if "panel_cmaps" in state and isinstance(state["panel_cmaps"], list):
            if len(state["panel_cmaps"]) not in (0, int(self.n_images)):
                state.pop("panel_cmaps")
        if "panel_cmaps_memory" in state and isinstance(state["panel_cmaps_memory"], list):
            if len(state["panel_cmaps_memory"]) not in (0, int(self.n_images)):
                state.pop("panel_cmaps_memory")
        if "inset_plots" in state:
            try:
                state["inset_plots"] = _normalize_inset_plot_specs(
                    state["inset_plots"],
                    n_items=int(self.n_images),
                )
            except (TypeError, ValueError):
                state.pop("inset_plots")
        if "panel_annotations" in state:
            try:
                state["panel_annotations"] = _normalize_panel_annotations(
                    state["panel_annotations"],
                    n_items=int(self.n_images),
                    labels=list(self.labels),
                )
            except (TypeError, ValueError):
                state.pop("panel_annotations")
        if "panel_overlays" in state:
            try:
                state["panel_overlays"] = _normalize_panel_overlays(
                    state["panel_overlays"],
                    n_items=int(self.n_images),
                    labels=list(self.labels),
                )
            except (TypeError, ValueError):
                state.pop("panel_overlays")
        if state.get("scale_bar_position") not in (None, "bottom-right", "bottom-left"):
            state.pop("scale_bar_position")
        if "scale_bar_panels" in state:
            try:
                state["scale_bar_panels"] = _normalize_panel_indices(
                    state["scale_bar_panels"],
                    n_items=int(self.n_images),
                    labels=list(self.labels),
                    name="scale_bar_panels",
                )
            except (TypeError, ValueError):
                state.pop("scale_bar_panels")
        if state.get("scale_bar_length") is not None:
            try:
                length = float(state["scale_bar_length"])
                if not np.isfinite(length) or length <= 0:
                    raise ValueError
                state["scale_bar_length"] = length
            except (TypeError, ValueError):
                state.pop("scale_bar_length")
        for key in ("marker_colors", "image_flips_horizontal", "image_flips_vertical"):
            if key in state and isinstance(state[key], list) and len(state[key]) not in (0, int(self.n_images)):
                state.pop(key)
        if state.get("marker_style") not in (None, "left", "around"):
            state.pop("marker_style")
        for key in ("row_markers", "col_markers"):
            if key in state:
                try:
                    state[key] = _normalize_marker_mapping(state[key], name=key)
                except (TypeError, ValueError):
                    state.pop(key)
        if "selected_panels" in state and isinstance(state["selected_panels"], list):
            selected: list[int] = []
            seen_selected: set[int] = set()
            for value in state["selected_panels"]:
                if isinstance(value, bool):
                    continue
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < int(self.n_images) and idx not in seen_selected:
                    selected.append(idx)
                    seen_selected.add(idx)
            state["selected_panels"] = selected
        if "hidden_panels" in state and isinstance(state["hidden_panels"], list):
            n_img = int(self.n_images)
            clean_set: set[int] = set()
            for value in state["hidden_panels"]:
                if isinstance(value, bool):
                    continue
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < n_img:
                    clean_set.add(idx)
            clean = sorted(clean_set)
            if len(clean) >= n_img:
                clean = clean[:-1]
            state["hidden_panels"] = self._normalize_item_page_hidden(
                clean,
                drop_if_full=True,
            )
        if "hidden_page_slots" in state and isinstance(state["hidden_page_slots"], list):
            state["hidden_page_slots"] = self._normalize_hidden_page_slots(
                state["hidden_page_slots"],
                drop_if_full=True,
            )
        elif int(self.n_pages) > 1 and "hidden_panels" in state and isinstance(state["hidden_panels"], list):
            state["hidden_page_slots"] = self._hidden_page_slots_from_panels(
                state["hidden_panels"],
                drop_if_full=True,
            )
        if "panel_order" in state and isinstance(state["panel_order"], list):
            try:
                order = [int(value) for value in state["panel_order"] if not isinstance(value, bool)]
            except (TypeError, ValueError):
                order = []
            if len(order) != int(self.n_images) or sorted(order) != list(range(int(self.n_images))):
                state.pop("panel_order")
            else:
                state["panel_order"] = order
        if "panel_frame_indices" in state and isinstance(state["panel_frame_indices"], list):
            counts = list(self.panel_frame_counts)
            raw_indices = state["panel_frame_indices"]
            try:
                indices = [
                    int(value) if not isinstance(value, bool) else -1
                    for value in raw_indices
                ]
            except (TypeError, ValueError):
                indices = []
            if len(indices) != int(self.n_images) or any(
                value < 0 or value >= max(1, int(counts[panel]))
                for panel, value in enumerate(indices)
            ):
                state.pop("panel_frame_indices")
            else:
                state["panel_frame_indices"] = indices
        for key, val in state.items():
            # Silent migrations for renamed keys in older saved state files.
            if key == "pixel_size_angstrom":
                key = "pixel_size"
            elif key == "canvas_size":
                key = "size"
            elif key in _DENOISE_STATE_ALIASES:
                key = _DENOISE_STATE_ALIASES[key]
            elif key == "filter_per_panel":
                key, val = "denoise_scope", ("all" if val else "panel")
            if key == "display_bin":
                self._display_bin = val
                continue
            if hasattr(self, key):
                setattr(self, key, val)

    @property
    def current_view(self) -> dict:
        """The field of view currently on screen, in image and calibrated coordinates.

        Every pan/zoom in the browser syncs the visible region back to Python
        (debounced ~100 ms), so after zooming into a feature you can capture
        exactly where you are for figure-making: ``view = w.current_view``.
        Feed ``view["box"]`` back as ``Show2D(data, view_box=view["box"])`` to
        reproduce the same crop later. The box is axis-aligned, so the two
        corners (row0, col0) and (row1, col1) define all four.

        Returns a dict with ``row0/col0/row1/col1`` (image pixel coordinates),
        ``height/width`` (visible extent in pixels), ``zoom``, ``box`` (the
        (row0, row1, col0, col1) tuple accepted by ``view_box=``), and when
        ``pixel_size > 0`` the calibrated ``*_cal`` extents plus ``unit``.
        """
        if len(self.view_box) == 4:
            row0, row1, col0, col1 = (float(v) for v in self.view_box)
        else:
            # No pan/zoom synced from the browser yet: derive the viewport
            # from the construction-time zoom center so the property is
            # usable immediately after Show2D(data, zoom=..., zoom_row=...).
            zoom = float(self.initial_zoom) if self.initial_zoom else 1.0
            center_row = float(self.zoom_row) if self.zoom_row is not None else self.height / 2
            center_col = float(self.zoom_col) if self.zoom_col is not None else self.width / 2
            half_h, half_w = self.height / (2 * zoom), self.width / (2 * zoom)
            row0, row1 = max(0.0, center_row - half_h), min(float(self.height), center_row + half_h)
            col0, col1 = max(0.0, center_col - half_w), min(float(self.width), center_col + half_w)
        view = {"row0": row0, "col0": col0, "row1": row1, "col1": col1,
                "height": row1 - row0, "width": col1 - col0,
                "zoom": float(self.initial_zoom),
                "box": (row0, row1, col0, col1)}
        if self.pixel_size > 0:
            scale = self.pixel_size
            view.update({"row0_cal": row0 * scale, "col0_cal": col0 * scale,
                         "row1_cal": row1 * scale, "col1_cal": col1 * scale,
                         "height_cal": (row1 - row0) * scale, "width_cal": (col1 - col0) * scale,
                         "unit": self.pixel_unit})
        return view

    def crop_to_view(self) -> Self:
        """Commit the current browser viewport as the display extent.

        Zoom into a feature (mouse wheel, or ``view_box=`` at construction),
        then call this to make that window the whole displayed frame: the
        widget repacks only the committed region, so an active denoise
        operates on the cropped view. Display-only and reversible: the
        stored array is never modified, the stats row keeps reporting the
        full raw data, and cursor coordinates stay full-image (row, col).
        :meth:`reset_view_ops` restores the full frame bit-identically.

        Returns
        -------
        Self
            The widget, for chaining.

        Raises
        ------
        NotImplementedError
            For galleries (crop-to-view is single panel only in this
            release) and for RGB panels.

        Examples
        --------
        >>> w = Show2D(image, view_box=(64, 64, 96))  # zoom into a feature
        >>> w.crop_to_view()      # the 96x96 window becomes the display extent
        >>> w.reset_view_ops()    # back to the full frame, bit-identical
        """
        if int(self.n_images) != 1:
            raise NotImplementedError(
                f"crop_to_view() supports a single panel; this widget shows "
                f"{int(self.n_images)} panels. Crop the arrays before display for galleries."
            )
        if any(self.is_rgb):
            raise NotImplementedError(
                "crop_to_view() supports grayscale panels; this panel is RGB."
            )
        view = self.current_view
        factor = max(1, int(self._display_bin_factor))
        offset_row, offset_col = (int(v) for v in self._view_crop_offset)
        self.view_crop = [
            int(math.floor(view["row0"])) * factor + offset_row,
            int(math.ceil(view["row1"])) * factor + offset_row,
            int(math.floor(view["col0"])) * factor + offset_col,
            int(math.ceil(view["col1"])) * factor + offset_col,
        ]
        return self

    def reset_view_ops(self) -> Self:
        """Restore the uncropped, unpadded display.

        Undoes :meth:`crop_to_view` and ``pad_ratio`` in one call. Both ops
        are display-only view transforms, so resetting returns the exact
        original frame bytes; the stored data was never touched.

        Returns
        -------
        Self
            The widget, for chaining.

        Examples
        --------
        >>> w = Show2D(image, pad_ratio=0.1)
        >>> w.reset_view_ops()  # full frame again, no border
        """
        with self.hold_sync():
            self.view_crop = []
            self.pad_ratio = 0.0
            self.pad_ratios = [0.0] * int(self.n_images)
            self.pad_fill_mode = "min"
            self.pad_fill_modes = ["min"] * int(self.n_images)
            self.pad_scope = "all"
        return self

    def set_padding(
        self,
        ratio: float,
        *,
        fill: str = "min",
        panels: Sequence[int] | int | str | None = None,
    ) -> Self:
        """Set display padding for every panel, or a chosen panel subset.

        Padding is a reversible display transform: the stored arrays are never
        modified, while frame bytes, histograms, saved state, and exports use
        the padded display frame. ``ratio`` is a fraction of the current display
        canvas size (0 to 1). ``fill`` chooses the constant border value from
        each panel: ``"min"``, ``"median"``, or ``"mean"``.
        """
        ratio = float(ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be between 0 and 1; got {ratio}")
        fill = str(fill).strip().lower()
        if fill not in {"min", "median", "mean"}:
            raise ValueError(f"fill must be 'min', 'median', or 'mean'; got {fill!r}")
        if any(self.is_rgb) and ratio > 0:
            raise NotImplementedError("set_padding() supports grayscale panels; RGB panels are not padded.")

        n_panels = int(self.n_images)
        if panels is None or (isinstance(panels, str) and panels.strip().lower() == "all"):
            target = list(range(n_panels))
            scope = "all"
        elif isinstance(panels, str):
            raise ValueError(
                "panels must be None, 'all', an integer panel index, "
                f"or a sequence of panel indices; got {panels!r}"
            )
        elif isinstance(panels, int) and not isinstance(panels, bool):
            target = [int(panels)]
            scope = "panel"
        else:
            target = [int(panel) for panel in panels]
            scope = "panel"
        invalid = [panel for panel in target if not 0 <= panel < n_panels]
        if invalid:
            raise ValueError(f"panels must be in [0, {n_panels - 1}]; got {invalid[0]}")

        ratios = list(self.pad_ratios) if len(self.pad_ratios) == n_panels else [float(self.pad_ratio)] * n_panels
        modes = list(self.pad_fill_modes) if len(self.pad_fill_modes) == n_panels else [str(self.pad_fill_mode)] * n_panels
        for panel in target:
            ratios[panel] = ratio
            modes[panel] = fill
        self._pad_knob_sync = True
        try:
            with self.hold_sync():
                self.pad_scope = scope
                self.pad_ratios = ratios
                self.pad_fill_modes = modes
                mirror_panel = target[0] if scope == "panel" and target else 0
                self.pad_ratio = ratios[mirror_panel]
                self.pad_fill_mode = modes[mirror_panel]
        finally:
            self._pad_knob_sync = False
        return self

    def set_denoise(
        self,
        mode: str,
        *,
        sigma: float | None = None,
        bin: int | None = None,
        panels: Sequence[int] | int | None = None,
    ) -> Self:
        """Set the display denoise for every panel, or a chosen panel subset.

        The imperative twin of the ``denoise=`` constructor kwarg, chainable
        like :meth:`crop_to_view` and :meth:`set_roi`. Denoise is a pure view
        transform: the stored array, the stats row, and every raw-data export
        keep the original counts, and an active reduction still announces
        itself with the one-line banner. The per-panel ``denoise_modes`` /
        ``denoise_sigmas`` / ``denoise_bins`` lists are the source of truth;
        the scalar ``denoise`` / ``denoise_sigma`` / ``denoise_bin`` traits
        mirror only the selected panel while ``denoise_scope == "panel"``.

        Parameters
        ----------
        mode : str
            Denoise method for the targeted panels: ``"none"``, ``"gaussian"``,
            or ``"anscombe"``. The compound spellings ``"bin2"``,
            ``"bin2_anscombe"`` and ``"bin4_anscombe"`` fold into a (mode, bin)
            pair; ``"tv"`` stays available from Python.
        sigma : float or None, optional
            Smoothing scale in pixels. ``None`` leaves each targeted panel's
            current sigma unchanged.
        bin : int or None, optional
            Display-side SNR bin factor (1, 2, or 4). ``None`` leaves each
            targeted panel's current bin unchanged, unless a compound ``mode``
            (e.g. ``"bin2_anscombe"``) sets it.
        panels : int, sequence of int, or None, optional
            Panel indices to change. ``None`` (the default) targets every
            panel and applies one uniform setting; passing a subset edits only
            those panels and switches ``denoise_scope`` to ``"panel"`` so later
            UI edits stay per panel.

        Returns
        -------
        Self
            The widget, for chaining.

        Raises
        ------
        ValueError
            When a ``panels`` index is out of range for the widget.

        Examples
        --------
        >>> import numpy as np
        >>> from quantem.widget import Show2D
        >>> a = np.random.poisson(0.3, (256, 256)).astype("float32")
        >>> b = np.random.poisson(0.3, (256, 256)).astype("float32")
        >>> w = Show2D([a, b])                             # a raw A/B gallery
        >>> _ = w.set_denoise("anscombe", sigma=8, bin=2)  # denoise both panels
        >>> _ = w.set_denoise("none", panels=[0])          # keep panel 0 raw
        """
        from quantem.widget.utils.display_filter import resolve_denoise_mode

        n_panels = int(self.n_images)
        if panels is None:
            targets = list(range(n_panels))
            subset = False
        else:
            if isinstance(panels, (int, np.integer)):
                panels = [int(panels)]
            targets = [int(p) for p in panels]
            for p in targets:
                if not (0 <= p < n_panels):
                    raise ValueError(
                        f"set_denoise panels index {p} is out of range for a "
                        f"{n_panels}-panel widget (valid 0..{n_panels - 1})."
                    )
            subset = True
        resolved_mode, resolved_bin = resolve_denoise_mode(
            str(mode), 1 if bin is None else int(bin)
        )
        modes = list(self.denoise_modes) or [str(self.denoise)] * n_panels
        sigmas = list(self.denoise_sigmas) or [float(self.denoise_sigma)] * n_panels
        bins = list(self.denoise_bins) or [int(self.denoise_bin)] * n_panels
        for p in targets:
            modes[p] = resolved_mode
            if sigma is not None:
                sigmas[p] = float(sigma)
            if bin is not None or resolved_bin > 1:
                bins[p] = int(resolved_bin)
        # Write the per-panel lists in one batch: suppress the per-list repack
        # observer and the scalar mirror during the write, then repack once
        # (mirrors the constructor). The scalar knobs re-sync to the selected
        # panel so the editor keeps showing what is on screen.
        with self.hold_sync():
            prev_ready = getattr(self, "_display_filter_ready", False)
            self._display_filter_ready = False
            self._filter_knob_sync = True
            try:
                if subset:
                    self.denoise_scope = "panel"
                self.denoise_modes = modes
                self.denoise_sigmas = sigmas
                self.denoise_bins = bins
                sel = min(int(self.selected_idx), max(0, n_panels - 1))
                self.denoise = modes[sel]
                self.denoise_sigma = float(sigmas[sel])
                self.denoise_bin = int(bins[sel])
            finally:
                self._display_filter_ready = prev_ready
                self._filter_knob_sync = False
            if prev_ready:
                if any(mode != "none" for mode in modes):
                    self.denoise_enabled = True
                if self._display_filter_active():
                    self.show_denoise = True
                self._refresh_display_filter_banner(announce=True)
                self._update_all_frames()
        return self

    def to_show3d(
        self,
        panels: Sequence[int | str] | int | str | None = None,
        *,
        title: str | None = None,
        copy: bool = True,
        include_hidden: bool = False,
    ):
        """Create a ``Show3D`` stack from this Show2D image gallery.

        The default uses currently visible panels as frames, preserving labels,
        colormap, sampling, scale-bar units, and contrast settings. Use
        ``include_hidden=True`` when hidden panels should also become frames.
        """
        from quantem.widget.show3d import Show3D

        if any(self.is_rgb):
            raise ValueError("Show2D.to_show3d() supports grayscale panels; RGB panels cannot become Show3D frames yet")
        if panels is None:
            panel_indices = list(range(int(self.n_images))) if include_hidden else self.visible_panels
        else:
            panel_indices = self._normalize_panel_refs(panels)
            if not include_hidden:
                hidden = set(int(i) for i in self.hidden_panels)
                panel_indices = [idx for idx in panel_indices if idx not in hidden]
        if not panel_indices:
            raise ValueError("Show2D.to_show3d() needs at least one visible panel")

        frames = [np.asarray(self._data[idx], dtype=np.float32) for idx in panel_indices]
        shapes = {frame.shape for frame in frames}
        if len(shapes) != 1:
            raise ValueError(
                "Show2D.to_show3d() requires selected panels to have the same shape; "
                f"got {sorted(shapes)!r}"
            )
        stack = np.stack(frames, axis=0)
        if copy:
            stack = np.array(stack, copy=True)

        vmin = self.vmin
        vmax = self.vmax
        if self.link_contrast and self.vmins and len(self.vmins) == int(self.n_images):
            chosen_vmins = [self.vmins[idx] for idx in panel_indices]
            chosen_vmaxs = [self.vmaxs[idx] for idx in panel_indices]
            if len({float(v) for v in chosen_vmins}) == 1:
                vmin = float(chosen_vmins[0])
            if len({float(v) for v in chosen_vmaxs}) == 1:
                vmax = float(chosen_vmaxs[0])

        return Show3D(
            stack,
            labels=[self._panel_title_for_index(idx) for idx in panel_indices],
            panel_titles=[title or self.title or "Show2D"],
            title=title if title is not None else self.title,
            show_title=self.show_title,
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            sampling=self.pixel_size if self.pixel_size > 0 else None,
            units=self.pixel_unit,
            smooth=self.smooth,
            log_scale=self.log_scale,
            auto_contrast=self.auto_contrast,
            show_fft=False,
            show_stats=self.show_stats,
            show_controls=self.show_controls,
            controls_collapsed=self.controls_collapsed,
            size=int(self.size or 0),
            panel_title_font_size=int(self.panel_title_font_size or 11),
            show_panel_titles=self.show_panel_titles,
            show_scale_bar=self.scale_bar_visible,
            save_state=False,
            verbose=False,
        )

    def _on_handoff_request_change(self, change: dict) -> None:
        """Build a Python-side prepared view from a frontend toolbar request."""
        raw = change.get("new") or ""
        if not raw:
            return
        try:
            request = json.loads(raw)
            mode = str(request.get("mode", "show3d")).lower()
            if mode == "clear":
                self.prepared_view = None
                self.prepared_view_widget = None
                self.handoff_status = ""
                return
            if mode != "show3d":
                raise ValueError(f"unsupported handoff mode {mode!r}")
            self.prepared_view = self.to_show3d(panels=request.get("panels", None))
            self.prepared_view_widget = self.prepared_view
            n_frames = int(getattr(self.prepared_view, "n_slices", 0))
            self.handoff_status = f"Showing 3D with {n_frames} frame{'s' if n_frames != 1 else ''}"
        except Exception as exc:  # pragma: no cover - defensive comm boundary
            self.prepared_view = None
            self.prepared_view_widget = None
            self.handoff_status = f"View failed: {exc}"

    @property
    def ordered_panels(self) -> list[int]:
        """Zero-based image panel indices in the current display order."""
        if int(self.n_pages) > 1 and int(self.panels_per_page) > 0:
            start = int(self.page_idx) * int(self.panels_per_page)
            stop = min(start + int(self.panels_per_page), int(self.n_images))
            if str(self.page_kind) == "items":
                return self._folder_ordered_panel_indices()[start:stop]
            return list(range(start, stop))
        order = list(self.panel_order or [])
        if len(order) == int(self.n_images) and sorted(order) == list(range(int(self.n_images))):
            return order
        return list(range(int(self.n_images)))

    @property
    def visible_panels(self) -> list[int]:
        """Zero-based image panel indices currently visible in the gallery."""
        if int(self.n_pages) > 1 and int(self.panels_per_page) > 0:
            if str(self.page_kind) == "items":
                hidden = set(self.hidden_panels)
                return [panel for panel in self.ordered_panels if panel not in hidden]
            hidden_slots = set(
                self.hidden_page_slots
                or self._hidden_page_slots_from_panels(self.hidden_panels, drop_if_full=True)
            )
            per_page = int(self.panels_per_page)
            return [i for i in self.ordered_panels if (i % per_page) not in hidden_slots]
        hidden = set(self.hidden_panels)
        return [i for i in self.ordered_panels if i not in hidden]

    @property
    def starred_panels(self) -> list[int]:
        """Zero-based image panel indices marked with a star."""
        return [i for i, value in enumerate(self.starred) if value]

    @property
    def starred_pages(self) -> list[int]:
        """Zero-based page indices marked with a star."""
        return [i for i, value in enumerate(self.page_starred) if value]

    def set_hidden_panels(self, panels: Sequence[int | str] | int | str) -> Self:
        """Replace the hidden panel set by index or exact label.

        Hidden panels remain in the widget state and standalone HTML export, but
        they are collapsed from the gallery until restored. At least one panel
        must stay visible.
        """
        hidden = self._normalize_panel_refs(panels, allow_empty=True)
        if int(self.n_pages) > 1 and str(self.page_kind) == "items":
            self._validate_folder_item_pages_visible(hidden, "set_hidden_panels")
        if int(self.n_pages) > 1 and str(self.page_kind) != "items":
            try:
                hidden_page_slots = self._hidden_page_slots_from_panels(hidden)
            except traitlets.TraitError as exc:
                raise ValueError(
                    "set_hidden_panels would hide every page slot; leave at least one visible"
                ) from exc
        else:
            hidden_page_slots = []
            if int(self.n_pages) > 1 and str(self.page_kind) == "items":
                try:
                    hidden = self._normalize_item_page_hidden(hidden)
                except traitlets.TraitError as exc:
                    raise ValueError(str(exc)) from exc
        if len(hidden) >= int(self.n_images):
            raise ValueError("set_hidden_panels would hide every panel; leave at least one visible")
        self.hidden_panels = sorted(hidden)
        self.hidden_page_slots = hidden_page_slots
        return self

    def hide_panel(self, *panels: int | str) -> Self:
        """Hide one or more image panels by zero-based index or exact label."""
        to_hide = set(self.hidden_panels)
        to_hide.update(self._normalize_panel_refs(list(panels)))
        if int(self.n_pages) > 1 and str(self.page_kind) == "items":
            self._validate_folder_item_pages_visible(to_hide, "hide_panel")
        if int(self.n_pages) > 1 and str(self.page_kind) != "items":
            try:
                hidden_page_slots = self._hidden_page_slots_from_panels(sorted(to_hide))
            except traitlets.TraitError as exc:
                raise ValueError("hide_panel would hide every page slot; leave at least one visible") from exc
        else:
            hidden_page_slots = []
            if int(self.n_pages) > 1 and str(self.page_kind) == "items":
                try:
                    to_hide = set(self._normalize_item_page_hidden(sorted(to_hide)))
                except traitlets.TraitError as exc:
                    raise ValueError(str(exc)) from exc
        if len(to_hide) >= int(self.n_images):
            raise ValueError("hide_panel would hide every panel; leave at least one visible")
        self.hidden_panels = sorted(to_hide)
        self.hidden_page_slots = hidden_page_slots
        return self

    def show_panel(self, *panels: int | str) -> Self:
        """Restore one or more hidden image panels by zero-based index or exact label."""
        to_show = set(self._normalize_panel_refs(list(panels)))
        hidden = sorted(set(self.hidden_panels) - to_show)
        self.hidden_panels = hidden
        self.hidden_page_slots = self._hidden_page_slots_from_panels(hidden)
        return self

    def show_all_panels(self) -> Self:
        """Restore every image panel in the gallery."""
        self.hidden_panels = []
        self.hidden_page_slots = []
        return self

    def set_panel_order(self, panels: Sequence[int | str]) -> Self:
        """Set the gallery display order by panel index or exact label.

        The order is display-only: source data, labels, hidden state, stars, and
        pixel sizes stay keyed by their original panel indices.
        """
        order = self._normalize_panel_refs(panels, allow_empty=True)
        if not order:
            self.panel_order = []
            return self
        expected = set(range(int(self.n_images)))
        if len(order) != int(self.n_images) or set(order) != expected:
            raise ValueError("set_panel_order requires every panel exactly once")
        self.panel_order = order
        return self

    def reset_panel_order(self) -> Self:
        """Restore the natural source-image order."""
        self.panel_order = []
        return self

    def move_panel(self, panel: int | str, position: int) -> Self:
        """Move one panel to a zero-based display position."""
        idx = self._resolve_panel_ref(panel)
        order = (
            self._folder_ordered_panel_indices()
            if int(self.n_pages) > 1 and str(self.page_kind) == "items"
            else self.ordered_panels
        )
        order.remove(idx)
        pos = int(position)
        if pos < 0:
            pos = 0
        if pos > len(order):
            pos = len(order)
        order.insert(pos, idx)
        self.panel_order = order
        return self

    def set_panel_frame(self, panel: int | str, frame: int) -> Self:
        """Set the displayed frame for one local stack panel.

        Static panels have one frame and therefore only accept frame ``0``.
        Negative indices follow normal Python indexing, so ``-1`` selects the
        final frame (useful for Velox/EDS acquisitions whose exported HAADF is
        the last survey frame).
        """
        panel_idx = self._resolve_panel_ref(panel)
        count = int(self.panel_frame_counts[panel_idx])
        frame_idx = int(frame)
        if frame_idx < 0:
            frame_idx += count
        if frame_idx < 0 or frame_idx >= count:
            if count == 1:
                raise IndexError(
                    f"panel {panel_idx} is static and has one frame; only frame 0 is valid"
                )
            raise IndexError(
                f"frame index {frame} out of range for panel {panel_idx} with {count} frame(s)"
            )
        indices = list(self.panel_frame_indices)
        indices[panel_idx] = frame_idx
        self.panel_frame_indices = indices
        return self

    def collapse_controls(self) -> Self:
        """Collapse the live control UI while leaving the GUI toggle available."""
        self.controls_collapsed = True
        return self

    def expand_controls(self) -> Self:
        """Expand the live control UI."""
        self.controls_collapsed = False
        return self

    def toggle_controls(self) -> Self:
        """Toggle the collapsed state of the live control UI."""
        self.controls_collapsed = not bool(self.controls_collapsed)
        return self

    def set_starred_panels(self, panels: Sequence[int | str] | int | str) -> Self:
        """Replace the set of starred image panels by index or exact label."""
        starred = [0] * int(self.n_images)
        for idx in self._normalize_panel_refs(panels, allow_empty=True):
            starred[idx] = 1
        self.starred = starred
        return self

    def star_panel(self, panel: int | str) -> Self:
        """Mark an image panel with a star."""
        idx = self._resolve_panel_ref(panel)
        starred = list(self.starred)
        if len(starred) != int(self.n_images):
            starred = [0] * int(self.n_images)
        starred[idx] = 1
        self.starred = starred
        return self

    def unstar_panel(self, panel: int | str) -> Self:
        """Clear the star on an image panel."""
        idx = self._resolve_panel_ref(panel)
        starred = list(self.starred)
        if len(starred) != int(self.n_images):
            starred = [0] * int(self.n_images)
        starred[idx] = 0
        self.starred = starred
        return self

    def star_page(self, page: int) -> Self:
        """Mark a page with a star."""
        idx = int(page)
        if idx < 0 or idx >= int(self.n_pages):
            raise ValueError(f"page index {idx} out of range [0, {self.n_pages})")
        starred = list(self.page_starred)
        if len(starred) != int(self.n_pages):
            starred = [0] * int(self.n_pages)
        starred[idx] = 1
        self.page_starred = starred
        return self

    def unstar_page(self, page: int) -> Self:
        """Clear the star on a page."""
        idx = int(page)
        if idx < 0 or idx >= int(self.n_pages):
            raise ValueError(f"page index {idx} out of range [0, {self.n_pages})")
        starred = list(self.page_starred)
        if len(starred) != int(self.n_pages):
            starred = [0] * int(self.n_pages)
        starred[idx] = 0
        self.page_starred = starred
        return self

    def summary(self):
        """Print a human-readable snapshot of the widget's current state.

        Reports image dimensions and pixel size, data min/max/mean, display
        settings (colormap, contrast, scale, FFT), active ROIs and profile
        line, per-image rotations, and the most recent render timings.
        """
        lines = [self.title or "Show2D", "═" * 32]
        if self.n_images > 1:
            lines.append(f"Image:    {self.n_images}×{self.height}×{self.width} ({self.ncols} cols)")
        else:
            lines.append(f"Image:    {self.height}×{self.width}")
        if self.pixel_size > 0:
            ps = self.pixel_size
            if ps >= 10:
                lines[-1] += f" ({ps / 10:.2f} nm/px)"
            else:
                lines[-1] += f" ({ps:.2f} Å/px)"
        if hasattr(self, "_data") and self._data is not None:
            arr = self._data
            lines.append(f"Data:     min={float(arr.min()):.4g}  max={float(arr.max()):.4g}  mean={float(arr.mean()):.4g}")
        cmap = self.cmap
        scale = "log" if self.log_scale else "linear"
        if self.vmin is not None and self.vmax is not None:
            contrast = f"vmin={self.vmin:.4g}, vmax={self.vmax:.4g}"
        elif self.auto_contrast:
            contrast = "auto contrast"
        else:
            contrast = "manual contrast"
        display = f"{cmap} | {contrast} | {scale}"
        if self.show_fft:
            display += " | FFT"
            if not self.fft_window:
                display += " (no window)"
        lines.append(f"Display:  {display}")
        if self.roi_active and self.roi_list:
            lines.append(f"ROI:      {len(self.roi_list)} region(s)")
        if self.profile_line:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) → ({p1['row']:.0f}, {p1['col']:.0f})")
        non_zero = [(i, r * 90) for i, r in enumerate(self.image_rotations) if r % 4 != 0]
        if non_zero:
            parts = [f"#{i}={deg}°" for i, deg in non_zero]
            lines.append(f"Rotated:  {', '.join(parts)}")
        rt = getattr(self, "render_total_ms", None)
        if rt is not None:
            pb = getattr(self, "render_python_build_ms", 0)
            wj = getattr(self, "render_wire_js_ms", 0)
            lines.append(f"Rendered: {rt} ms total (Python build {pb} ms, wire+JS {wj} ms)")
        else:
            lines.append("Rendered: (pending first browser paint)")
        print("\n".join(lines))

    def _compute_all_stats(self):
        """Compute statistics for all images (vectorized over all frames)."""
        # Vectorized reduction over (H, W) is faster than per-image loops
        # for large galleries (e.g. 12×4096×4096: 164ms vs 191ms).
        axes = (1, 2) if self._data.ndim == 3 else None
        self.stats_mean = np.mean(self._data, axis=axes).ravel().tolist()
        self.stats_min = np.min(self._data, axis=axes).ravel().tolist()
        self.stats_max = np.max(self._data, axis=axes).ravel().tolist()
        self.stats_std = np.std(self._data, axis=axes).ravel().tolist()

    @traitlets.validate("denoise")
    def _validate_display_filter(self, proposal: dict) -> str:
        """Normalize and reject unknown display filter modes early."""
        from quantem.widget.utils.display_filter import DISPLAY_FILTER_MODES, _normalize_mode

        mode = _normalize_mode(proposal["value"])
        if mode != "none" and mode not in DISPLAY_FILTER_MODES:
            raise traitlets.TraitError(
                "denoise must be one of "
                + "|".join(DISPLAY_FILTER_MODES)
                + f" (or 'off'/'raw'); got {proposal['value']!r}"
            )
        return mode

    @traitlets.validate("denoise_scope")
    def _validate_denoise_scope(self, proposal: dict) -> str:
        value = str(proposal["value"]).strip().lower()
        if value not in ("all", "panel"):
            raise traitlets.TraitError(
                f"denoise_scope must be 'all' or 'panel'; got {proposal['value']!r}"
            )
        return value

    @traitlets.validate("denoise_bin")
    def _validate_spatial_bin(self, proposal: dict) -> int:
        value = int(proposal["value"])
        if value not in (1, 2, 4):
            raise traitlets.TraitError(f"denoise_bin must be 1, 2, or 4; got {value}")
        return value

    @traitlets.validate("underlay_mode")
    def _validate_underlay_mode(self, proposal: dict) -> str:
        value = str(proposal["value"]).strip().lower()
        if value not in ("haadf", "dual"):
            raise traitlets.TraitError(
                f"underlay_mode must be 'haadf' or 'dual'; got {proposal['value']!r}"
            )
        return value

    @traitlets.validate("stretch_percentiles")
    def _validate_stretch_percentiles(self, proposal: dict) -> list[float]:
        value = [float(v) for v in proposal["value"]]
        if len(value) != 2:
            raise traitlets.TraitError(
                "stretch_percentiles must be [low, high]; got "
                f"{proposal['value']!r}"
            )
        lo, hi = value
        if not (0.0 <= lo < hi <= 100.0):
            raise traitlets.TraitError(
                f"stretch_percentiles must satisfy 0 <= low < high <= 100; got [{lo}, {hi}]"
            )
        return value

    @traitlets.validate("display_gamma")
    def _validate_display_gamma(self, proposal: dict) -> float:
        value = float(proposal["value"])
        if not math.isfinite(value) or value <= 0:
            raise traitlets.TraitError(f"display_gamma must be > 0; got {value}")
        return value

    @traitlets.validate("dual_gain")
    def _validate_dual_gain(self, proposal: dict) -> list[float]:
        value = [float(v) for v in proposal["value"]]
        if len(value) != 2 or any((not math.isfinite(v)) or v < 0 for v in value):
            raise traitlets.TraitError(
                "dual_gain must be two non-negative finite values [gain_a, gain_b]; "
                f"got {proposal['value']!r}"
            )
        return value

    @traitlets.validate("denoise_modes")
    def _validate_display_filters(self, proposal: dict) -> list[str]:
        from quantem.widget.utils.display_filter import DISPLAY_FILTER_MODES, _normalize_mode

        modes = [_normalize_mode(value) for value in proposal["value"]]
        for mode in modes:
            if mode != "none" and mode not in DISPLAY_FILTER_MODES:
                raise traitlets.TraitError(
                    "denoise_modes entries must be one of "
                    + "|".join(DISPLAY_FILTER_MODES)
                    + f" (or 'off'/'raw'); got {mode!r}"
                )
        return modes

    @traitlets.validate("denoise_bins")
    def _validate_spatial_bins(self, proposal: dict) -> list[int]:
        values = [int(value) for value in proposal["value"]]
        for value in values:
            if value not in (1, 2, 4):
                raise traitlets.TraitError(
                    f"denoise_bins entries must be 1, 2, or 4; got {value}"
                )
        return values

    @traitlets.validate("pad_ratio")
    def _validate_pad_ratio(self, proposal: dict) -> float:
        value = float(proposal["value"])
        if not 0.0 <= value <= 1.0:
            raise traitlets.TraitError(
                f"pad_ratio must be between 0 and 1 (border as a fraction of "
                f"max(rows, cols)); got {value}"
            )
        # RGB panels use a separate packed representation; keep pad scalar
        # grayscale-only until color-frame padding has a dedicated path.
        if value > 0 and getattr(self, "_data", None) is not None:
            if any(self.is_rgb):
                raise NotImplementedError(
                    "pad_ratio supports grayscale panels; RGB panels are not padded."
                )
        return value

    @traitlets.validate("pad_ratios")
    def _validate_pad_ratios(self, proposal: dict) -> list[float]:
        values = [float(value) for value in proposal["value"]]
        if getattr(self, "_data", None) is not None and values and len(values) != int(self.n_images):
            raise traitlets.TraitError(
                f"pad_ratios length ({len(values)}) must equal panel count ({int(self.n_images)})"
            )
        invalid = next((value for value in values if not 0.0 <= value <= 1.0), None)
        if invalid is not None:
            raise traitlets.TraitError(f"pad_ratios entries must be between 0 and 1; got {invalid}")
        if any(values) and getattr(self, "_data", None) is not None and any(self.is_rgb):
            raise NotImplementedError("pad_ratios supports grayscale panels; RGB panels are not padded.")
        return values

    @traitlets.validate("pad_fill_mode")
    def _validate_pad_fill_mode(self, proposal: dict) -> str:
        value = str(proposal["value"]).strip().lower()
        if value not in {"min", "median", "mean"}:
            raise traitlets.TraitError(
                f"pad_fill_mode must be 'min', 'median', or 'mean'; got {proposal['value']!r}"
            )
        return value

    @traitlets.validate("pad_fill_modes")
    def _validate_pad_fill_modes(self, proposal: dict) -> list[str]:
        values = [str(value).strip().lower() for value in proposal["value"]]
        if getattr(self, "_data", None) is not None and values and len(values) != int(self.n_images):
            raise traitlets.TraitError(
                f"pad_fill_modes length ({len(values)}) must equal panel count ({int(self.n_images)})"
            )
        invalid = next((value for value in values if value not in {"min", "median", "mean"}), None)
        if invalid is not None:
            raise traitlets.TraitError(f"pad_fill_modes entries must be 'min', 'median', or 'mean'; got {invalid!r}")
        return values

    @traitlets.validate("pad_scope")
    def _validate_pad_scope(self, proposal: dict) -> str:
        value = str(proposal["value"]).strip().lower()
        if value not in {"all", "panel"}:
            raise traitlets.TraitError(f"pad_scope must be 'all' or 'panel'; got {proposal['value']!r}")
        return value

    @traitlets.validate("view_crop")
    def _validate_view_crop(self, proposal: dict) -> list[int]:
        values = [int(v) for v in proposal["value"]]
        if not values:
            return []
        if len(values) != 4:
            raise traitlets.TraitError(
                f"view_crop must be empty or (row0, row1, col0, col1) in "
                f"full-resolution image pixels; got {len(values)} values"
            )
        if getattr(self, "_data", None) is None:
            return values
        if int(self.n_images) != 1:
            raise traitlets.TraitError(
                f"view_crop supports a single panel; this widget shows "
                f"{int(self.n_images)} panels"
            )
        full_h, full_w = int(self._data.shape[1]), int(self._data.shape[2])
        row0, row1, col0, col1 = values
        row0, col0 = max(0, row0), max(0, col0)
        row1, col1 = min(full_h, row1), min(full_w, col1)
        if row1 - row0 < 2 or col1 - col0 < 2:
            raise traitlets.TraitError(
                f"view_crop window must span at least 2x2 pixels inside the "
                f"{full_h}x{full_w} image; got {values}"
            )
        return [row0, row1, col0, col1]

    # =========================================================================
    # Reversible view ops: crop-to-view + pad (display window, single panel)
    # =========================================================================
    def _view_crop_geometry(self) -> tuple[int, int, int, int]:
        """Active crop window in DISPLAY pixels.

        The view_crop trait holds full-resolution coordinates while the
        packed frames live in display pixels (after any _display_bin), so
        the window is rescaled here. No crop returns the full display frame.
        """
        data = self._display_data if self._display_data is not None else self._data
        full_h, full_w = int(data.shape[1]), int(data.shape[2])
        single_scalar = int(self.n_images) == 1 and not any(self.is_rgb)
        factor = max(1, int(self._display_bin_factor))
        if len(self.view_crop) == 4 and single_scalar:
            row0, row1, col0, col1 = (int(v) for v in self.view_crop)
            return (
                max(0, row0 // factor),
                min(full_h, -(-row1 // factor)),
                max(0, col0 // factor),
                min(full_w, -(-col1 // factor)),
            )
        return (0, full_h, 0, full_w)

    def _panel_pad_ratio(self, panel: int) -> float:
        ratios = list(self.pad_ratios)
        if 0 <= panel < len(ratios):
            return float(ratios[panel])
        return float(self.pad_ratio)

    def _panel_pad_fill_mode(self, panel: int) -> str:
        modes = list(self.pad_fill_modes)
        if 0 <= panel < len(modes):
            return str(modes[panel]).lower()
        return str(self.pad_fill_mode).lower()

    @staticmethod
    def _pad_fill_value(frame: np.ndarray, mode: str) -> float:
        finite = np.asarray(frame, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return 0.0
        if mode == "mean":
            return float(np.mean(finite))
        if mode == "median":
            return float(np.median(finite))
        return float(np.min(finite))

    def _view_ops_active(self) -> bool:
        data = self._display_data if self._display_data is not None else self._data
        crop = self._view_crop_geometry()
        any_pad = any(self._panel_pad_ratio(panel) > 0 for panel in range(int(self.n_images)))
        return any_pad or crop != (0, int(data.shape[1]), 0, int(data.shape[2]))

    def _crop_view_stack(self, data: np.ndarray) -> np.ndarray:
        """Crop VIEW of the display stack (before denoise); never copies."""
        row0, row1, col0, col1 = self._view_crop_geometry()
        return data[:, row0:row1, col0:col1]

    def _pad_view_stack(self, data: np.ndarray) -> np.ndarray:
        """Constant border around each frame after denoise/filter display ops."""
        if not any(self._panel_pad_ratio(panel) > 0 for panel in range(int(data.shape[0]))):
            return data
        base_h, base_w = int(data.shape[1]), int(data.shape[2])
        pads = [
            max(1, round(self._panel_pad_ratio(panel) * max(base_h, base_w)))
            if self._panel_pad_ratio(panel) > 0
            else 0
            for panel in range(int(data.shape[0]))
        ]
        target_h = max(base_h + 2 * pad for pad in pads)
        target_w = max(base_w + 2 * pad for pad in pads)
        frames = []
        for panel, frame in enumerate(np.asarray(data, dtype=np.float32)):
            fill = self._pad_fill_value(frame, self._panel_pad_fill_mode(panel))
            out = np.full((target_h, target_w), fill, dtype=np.float32)
            top = (target_h - base_h) // 2
            left = (target_w - base_w) // 2
            out[top:top + base_h, left:left + base_w] = frame
            frames.append(out)
        return np.stack(frames, axis=0)

    def _refresh_view_ops(self, *, announce: bool) -> None:
        """Sync the frame extent, cursor offset, and crop/pad notice.

        An active view reduction is never silent (house rule): the banner
        names the crop window in full-image coordinates and the pad ratio,
        and says that reset_view_ops() restores the full frame.
        """
        row0, row1, col0, col1 = self._view_crop_geometry()
        factor = max(1, int(self._display_bin_factor))
        base_h = row1 - row0
        base_w = col1 - col0
        pads = [
            max(1, round(self._panel_pad_ratio(panel) * max(base_h, base_w)))
            if self._panel_pad_ratio(panel) > 0
            else 0
            for panel in range(int(self.n_images))
        ]
        max_pad = max(pads) if pads else 0
        self.height = base_h + 2 * max_pad
        self.width = base_w + 2 * max_pad
        self._view_crop_offset = [(row0 - max_pad) * factor, (col0 - max_pad) * factor]
        parts = []
        if len(self.view_crop) == 4:
            r0, r1, c0, c1 = (int(v) for v in self.view_crop)
            parts.append(f"cropped to ({r0},{c0})-({r1},{c1})")
        # Announce the pad only when it was actually applied. The geometry
        # gates the pad on single_scalar, so pad > 0 (not pad_ratio > 0) is the
        # honest test: a gallery/RGB widget never pads, so it must never claim
        # a border in the banner. Constructing such a widget with pad_ratio > 0
        # raises in _validate_pad_ratio, but a later trait assignment could
        # still reach here.
        active_ratios = [self._panel_pad_ratio(panel) for panel, pad in enumerate(pads) if pad > 0]
        if active_ratios:
            if len(set(round(value, 6) for value in active_ratios)) == 1:
                parts.append(f"pad {active_ratios[0]:.0%} {self.pad_fill_mode}")
            else:
                parts.append("pad per-panel")
        banner = (
            f"view: {' · '.join(parts)} (reset_view_ops() restores full frame)"
            if parts
            else ""
        )
        changed = banner != self.view_banner
        self.view_banner = banner
        if announce and banner and changed:
            print(banner)

    @traitlets.observe("view_crop", "pad_ratio", "pad_ratios", "pad_fill_mode", "pad_fill_modes")
    def _on_view_ops_change(self, change: dict) -> None:
        """Repack the display frames when the crop window or pad changes."""
        if not getattr(self, "_view_ops_ready", False):
            return
        with self.hold_sync():
            self._refresh_view_ops(announce=True)
            # The committed window now fills the frame: clear the stale
            # zoom/viewport so the browser paints the new extent at 1x.
            self.view_box = []
            self.initial_zoom = 1.0
            self.zoom_row = None
            self.zoom_col = None
            self._update_all_frames()

    @traitlets.observe("pad_ratio", "pad_fill_mode")
    def _on_pad_scalar_change(self, change: dict) -> None:
        """Mirror scalar pad edits to all panels or to the selected panel."""
        if not getattr(self, "_view_ops_ready", False) or getattr(self, "_pad_knob_sync", False):
            return
        n_panels = int(self.n_images)
        if n_panels <= 0:
            return
        idx = max(0, min(int(self.selected_idx), n_panels - 1))

        def updated(values, value):
            current = list(values) if len(values) == n_panels else [value] * n_panels
            if self.pad_scope == "panel":
                current[idx] = value
            else:
                current = [value] * n_panels
            return current

        self._pad_knob_sync = True
        try:
            if change["name"] == "pad_ratio":
                value = float(change["new"])
                self.pad_ratios = updated(self.pad_ratios, value)
            elif change["name"] == "pad_fill_mode":
                value = str(change["new"]).lower()
                self.pad_fill_modes = updated(self.pad_fill_modes, value)
        finally:
            self._pad_knob_sync = False

    @traitlets.observe("pad_ratios", "pad_fill_modes")
    def _on_pad_panel_knobs_change(self, change: dict) -> None:
        """Mirror selected per-panel pad knobs back to scalar editor traits."""
        if not getattr(self, "_view_ops_ready", False) or getattr(self, "_pad_knob_sync", False):
            return
        if self.pad_scope != "panel":
            return
        idx = max(0, min(int(self.selected_idx), int(self.n_images) - 1))
        self._pad_knob_sync = True
        try:
            if 0 <= idx < len(self.pad_ratios):
                self.pad_ratio = float(self.pad_ratios[idx])
            if 0 <= idx < len(self.pad_fill_modes):
                self.pad_fill_mode = str(self.pad_fill_modes[idx]).lower()
        finally:
            self._pad_knob_sync = False

    @traitlets.observe("selected_idx")
    def _on_pad_selected_panel_change(self, change: dict) -> None:
        """Keep scalar padding editor traits aligned to the selected panel."""
        if not getattr(self, "_view_ops_ready", False) or getattr(self, "_pad_knob_sync", False):
            return
        if self.pad_scope != "panel":
            return
        idx = max(0, min(int(self.selected_idx), int(self.n_images) - 1))
        self._pad_knob_sync = True
        try:
            if 0 <= idx < len(self.pad_ratios):
                self.pad_ratio = float(self.pad_ratios[idx])
            if 0 <= idx < len(self.pad_fill_modes):
                self.pad_fill_mode = str(self.pad_fill_modes[idx]).lower()
        finally:
            self._pad_knob_sync = False

    @traitlets.observe("denoise", "denoise_sigma", "denoise_bin")
    def _on_display_filter_scalar_change(self, change: dict) -> None:
        """UI editor knobs write through to the per-panel lists.

        Scope follows ``denoise_scope``: "all" broadcasts the edit to every
        panel, "panel" edits only the selected panel.
        """
        if not getattr(self, "_display_filter_ready", False) or self._filter_knob_sync:
            return
        idx = min(int(self.selected_idx), max(0, int(self.n_images) - 1))
        n_panels = int(self.n_images)

        def updated(current, value):
            values = list(current)
            if len(values) != n_panels:
                values = [value] * n_panels
            if self.denoise_scope != "panel":
                return [value] * n_panels
            values[idx] = value
            return values

        name = change["name"]
        if name == "denoise":
            from quantem.widget.utils.display_filter import resolve_denoise_mode

            mode, extra_bin = resolve_denoise_mode(str(change["new"]))
            if mode != str(change["new"]):
                # Compound alias: rewrite the scalar knobs to the canonical
                # (mode, bin) pair; the recursive observer runs write the
                # per-panel lists for each.
                self.denoise = mode
                if extra_bin > 1:
                    self.denoise_bin = max(int(self.denoise_bin), extra_bin)
                return
            self.denoise_modes = updated(self.denoise_modes, mode)
        elif name == "denoise_sigma":
            self.denoise_sigmas = updated(self.denoise_sigmas, float(change["new"]))
        else:
            self.denoise_bins = updated(self.denoise_bins, int(change["new"]))
        if self._has_denoise_config():
            self.denoise_enabled = True

    @traitlets.observe("denoise_modes", "denoise_sigmas", "denoise_bins")
    def _on_display_filter_change(self, change: dict) -> None:
        """Repack the display frames when per-panel knobs change (no disk I/O)."""
        if not getattr(self, "_display_filter_ready", False):
            return
        self._refresh_display_filter_banner(announce=True)
        self._update_all_frames()

    @traitlets.observe("selected_idx")
    def _on_selected_panel_filter_sync(self, change: dict) -> None:
        """Panel scope: the scalar editor knobs mirror the selected panel."""
        if not getattr(self, "_display_filter_ready", False) or self.denoise_scope != "panel":
            return
        idx = int(change["new"])
        if not (0 <= idx < len(self.denoise_modes)):
            return
        self._filter_knob_sync = True
        try:
            self.denoise = self.denoise_modes[idx]
            self.denoise_sigma = float(self.denoise_sigmas[idx])
            self.denoise_bin = int(self.denoise_bins[idx])
        finally:
            self._filter_knob_sync = False

    def _panel_filter_knobs(self, panel: int) -> tuple[str, float, int]:
        filters = self.denoise_modes
        if not filters or panel >= len(filters):
            return str(self.denoise), float(self.denoise_sigma), int(self.denoise_bin)
        return (
            str(filters[panel]),
            float(self.denoise_sigmas[panel]),
            int(self.denoise_bins[panel]),
        )

    def _panel_filter_active(self, panel: int) -> bool:
        from quantem.widget.utils.display_filter import _normalize_mode

        mode, _sigma, denoise_bin = self._panel_filter_knobs(panel)
        return _normalize_mode(mode) != "none" or denoise_bin > 1

    def _has_denoise_config(self) -> bool:
        """Any panel carries an active denoise config, regardless of the master
        on/off. This is the CONFIG check (used to seed denoise_enabled)."""
        return any(self._panel_filter_active(i) for i in range(int(self.n_images)))

    def _display_filter_active(self) -> bool:
        """Denoise is actually applied: has a config AND the master switch is on.
        Turning denoise_enabled off shows raw while preserving the config."""
        return self._has_denoise_config() and bool(self.denoise_enabled)

    def _panel_browser_filtered(self, panel: int) -> bool:
        """True when this panel's display filter runs in the browser.

        The WGSL port (js/displayFilter.ts) covers the gaussian/bin2/anscombe
        modes; when the frontend negotiated ``_webgpu_filter_ok`` those panels
        ship raw pixels and the browser filters. Non portable modes (tv) keep
        the Python path even in a WebGPU session.
        """
        from quantem.widget.utils.display_filter import (
            BROWSER_DISPLAY_FILTER_MODES,
            _normalize_mode,
        )

        if not bool(self._webgpu_filter_ok):
            return False
        mode, _sigma, _spatial_bin = self._panel_filter_knobs(panel)
        return _normalize_mode(mode) in BROWSER_DISPLAY_FILTER_MODES

    @traitlets.observe("_webgpu_filter_ok")
    def _on_webgpu_filter_ok_change(self, change: dict) -> None:
        """Repack frames when the browser filter negotiation flips.

        True: portable panels repack raw (the browser filters them). False
        (e.g. reopening in a browser without WebGPU): Python filtering packs
        the view again, so the fallback shows identical pixels.
        """
        if not getattr(self, "_display_filter_ready", False):
            return
        if self._display_filter_active():
            self._update_all_frames()

    def _refresh_display_filter_banner(self, *, announce: bool) -> None:
        """Sync the one-line reduction notice; print it when it changes.

        Announcing an active reduction is a house rule: the user must always
        know the view is filtered and that ``denoise='none'`` restores
        raw counts. Mixed per-panel setups summarize which panels are filtered.
        """
        from quantem.widget.utils.display_filter import format_display_filter_banner

        knobs = [self._panel_filter_knobs(i) for i in range(int(self.n_images))]
        active = [i for i in range(int(self.n_images)) if self._panel_filter_active(i)]
        if not active:
            banner = ""
        elif len(set(knobs)) == 1:
            mode, sigma, denoise_bin = knobs[0]
            banner = format_display_filter_banner(mode, sigma, denoise_bin)
        else:
            per_panel = ", ".join(
                f"p{i}:{knobs[i][0]} σ={knobs[i][1]:g}"
                + (f" bin{knobs[i][2]}" if knobs[i][2] > 1 else "")
                for i in active
            )
            banner = f"denoise: {per_panel} (set denoise='none' for raw counts)"
        changed = banner != self.denoise_banner
        self.denoise_banner = banner
        if announce and banner and changed:
            print(banner)

    def _filter_display_frame(self, frame: np.ndarray, panel: int | None = None) -> np.ndarray:
        from quantem.widget.utils.display_filter import apply_display_filter

        if panel is None:
            mode, sigma, denoise_bin = (
                str(self.denoise),
                float(self.denoise_sigma),
                int(self.denoise_bin),
            )
        else:
            mode, sigma, denoise_bin = self._panel_filter_knobs(panel)
        return apply_display_filter(
            np.asarray(frame), mode=mode, sigma=sigma, spatial_bin=denoise_bin
        )

    def _filtered_frames(self, data: np.ndarray) -> np.ndarray:
        """Filtered VIEW of the display stack; the input arrays are never touched.

        Each panel uses its own knobs from the per-panel lists. RGB panels are
        skipped: per-channel filtering of composed color panels changes hue
        balance, so only 2D scalar panels are filtered.
        """
        if not self._display_filter_active():
            return data
        frames = []
        for i in range(int(data.shape[0])):
            rgb = bool(self.is_rgb[i]) if i < len(self.is_rgb) else False
            # Browser-filtered panels ship raw: the WGSL port applies the same
            # math client-side (live sigma scrub, kernel-less HTML exports).
            skip = rgb or not self._panel_filter_active(i) or self._panel_browser_filtered(i)
            frames.append(
                np.asarray(data[i], dtype=np.float32)
                if skip
                else self._filter_display_frame(data[i], panel=i)
            )
        return np.stack(frames, axis=0)

    def _filtered_underlay_input(self, panel: int) -> np.ndarray:
        """A source panel with its active display filter applied, if any.

        The underlay/dual composite always matches the filtered map panels next
        to it; the stored arrays keep raw counts.
        """
        raw = np.asarray(self._data[panel], dtype=np.float32)
        if self._panel_filter_active(panel):
            return self._filter_display_frame(raw, panel=panel)
        return raw

    def _compute_underlay_blend(self) -> np.ndarray:
        """Chemistry composite from the two source panels, dispatched by mode.

        ``underlay_mode='haadf'`` blends one element map (magenta) onto the
        HAADF lattice; ``'dual'`` composes two maps into a magenta+green panel.
        Both stretch each map with ``stretch_percentiles`` for display; the
        stored arrays keep raw counts.
        """
        from quantem.widget.utils.display_filter import blend_map_on_haadf, magenta_cmap

        lo_pct = float(self.stretch_percentiles[0])
        hi_pct = float(self.stretch_percentiles[1])

        def norm01(image, low, high):
            lo, hi = np.percentile(image, [low, high])
            span = hi - lo if hi > lo else 1.0
            return np.clip((image - lo) / span, 0.0, 1.0)

        if str(self.underlay_mode).lower() == "dual":
            map_a = self._filtered_underlay_input(0)
            map_b = self._filtered_underlay_input(1)
            return _compose_dual(
                norm01(map_a, lo_pct, hi_pct),
                norm01(map_b, lo_pct, hi_pct),
                gain=(float(self.dual_gain[0]), float(self.dual_gain[1])),
            )

        haadf = np.asarray(self._data[self._underlay_haadf_idx], dtype=np.float32)
        map_view = self._filtered_underlay_input(self._underlay_map_idx)

        cmap_name = str(self.cmap).lower()
        if cmap_name in ("magenta", "eds_magenta", "mag"):
            cmap = magenta_cmap()
        else:
            import matplotlib.pyplot as plt

            try:
                cmap = plt.colormaps[str(self.cmap)]
            except KeyError:
                cmap = magenta_cmap()
        return blend_map_on_haadf(
            # stretch_percentiles defaults to (4, 99), the dominant stretch of
            # the drift-paper Fig4 sweep; sparse maps at (2, 99.5) render dark.
            norm01(map_view, lo_pct, hi_pct),
            norm01(haadf, 1.0, 99.9),
            alpha=float(self.underlay_alpha),
            haadf_gain=float(self.underlay_haadf_gain),
            gamma=float(self.display_gamma),
            cmap=cmap,
        ).astype(np.float32)

    @traitlets.observe(
        "underlay_alpha", "underlay_haadf_gain", "underlay_mode",
        "stretch_percentiles", "display_gamma", "dual_gain",
    )
    def _on_underlay_change(self, change: dict) -> None:
        """Re-blend the chemistry panel live when a Fig4 knob moves."""
        if getattr(self, "_underlay_map_idx", None) is None or not getattr(
            self, "_display_filter_ready", False
        ):
            return
        self._update_all_frames()

    def _update_all_frames(self):
        """Send display data to JS (possibly binned for large galleries)."""
        if getattr(self, "_underlay_map_idx", None) is not None and getattr(
            self, "_display_filter_ready", False
        ):
            # The blend depends on filter knobs + underlay sliders; recompute
            # the RGB panel here so every knob path repacks a fresh blend.
            composed = self._compute_underlay_blend()
            self._rgb_frames[-1] = composed
            display_rgb = list(getattr(self, "_display_rgb", self._rgb_frames))
            if self._display_bin > 1:
                from quantem.widget.utils.array import bin2d

                display_rgb[-1] = bin2d(
                    composed.transpose(2, 0, 1), factor=self._display_bin, mode="mean"
                ).transpose(1, 2, 0)
            else:
                display_rgb[-1] = composed
            self._display_rgb = display_rgb
        data = self._display_data if self._display_data is not None else self._data
        # Reversible view ops: crop the display window BEFORE denoise (so the
        # filter operates on the cropped region) and pad AFTER it (so the
        # border stays a flat minimum instead of smearing into the filter).
        data = self._crop_view_stack(data)
        data = self._filtered_frames(data)
        data = self._pad_view_stack(data)
        if any(self.is_rgb):
            # Mixed packing: each panel is one contiguous float32 block, W*H
            # floats for grayscale, 3*W*H interleaved floats for RGB. JS derives
            # per-panel offsets from the synced is_rgb flags.
            display_rgb = getattr(self, "_display_rgb", self._rgb_frames)
            blocks = [
                np.ascontiguousarray(display_rgb[i], dtype=np.float32).tobytes()
                if self.is_rgb[i] else
                np.ascontiguousarray(data[i], dtype=np.float32).tobytes()
                for i in range(int(data.shape[0]))
            ]
            self.frame_bytes = _b64_safe(b"".join(blocks))
        elif self.offline:
            # Quantize to uint8 PER IMAGE (4x smaller than float32). Each panel uses
            # its own (min, max) so every panel gets the full 256 codes - a global
            # range would starve narrow-range panels and comb their histograms.
            # Display-only: the colormap reduces to 256 levels anyway.
            arr = np.ascontiguousarray(data, dtype=np.float32)  # (n, H, W)
            flat = arr.reshape(arr.shape[0], -1)
            out = np.empty(flat.shape, dtype=np.uint8)
            mins, maxs = [], []
            for i in range(flat.shape[0]):
                finite = flat[i][np.isfinite(flat[i])]
                lo = float(finite.min()) if finite.size else 0.0
                hi = float(finite.max()) if finite.size else 1.0
                rng = hi - lo if hi > lo else 1.0
                out[i] = np.clip((flat[i] - lo) * (255.0 / rng), 0, 255).astype(np.uint8)
                mins.append(lo)
                maxs.append(hi)
            self._offline_mins = mins
            self._offline_maxs = maxs
            self._offline_min = mins[0]  # back-compat scalars (single-image readers)
            self._offline_max = maxs[0]
            self.frame_bytes = _b64_safe(out.tobytes())
        else:
            self.frame_bytes = _b64_safe(data.tobytes())
        self._update_panel_stack_bytes()

    def _update_panel_stack_bytes(self) -> None:
        """Pack only multi-frame panel data for browser-local frame changes."""
        stacks = getattr(self, "_display_panel_stacks", None)
        if not stacks or not any(int(stack.shape[0]) > 1 for stack in stacks):
            self.panel_stack_offsets = [-1] * int(getattr(self, "n_images", 0))
            self.panel_stack_bytes = b""
            self._panel_stack_mins = []
            self._panel_stack_maxs = []
            return

        offsets: list[int] = []
        blocks: list[bytes] = []
        mins = [0.0] * len(stacks)
        maxs = [1.0] * len(stacks)
        float_offset = 0
        filter_active = self._display_filter_active()
        for panel, stack in enumerate(stacks):
            if int(stack.shape[0]) <= 1:
                offsets.append(-1)
                continue
            rgb = bool(self.is_rgb[panel]) if panel < len(self.is_rgb) else False
            if not rgb:
                # View ops track the main frame so browser-local frame
                # scrubbing shows the same committed crop window.
                stack = self._crop_view_stack(stack)
            # Browser-filtered panels ship raw stacks; JS filters each frame
            # on scrub with the same per-panel knobs.
            skip = rgb or not self._panel_filter_active(panel) or self._panel_browser_filtered(panel)
            if filter_active and not skip:
                # Per-frame filtered VIEW so browser-local frame scrubbing shows
                # the same filter as the main frame; the stored stack is untouched.
                stack = np.stack(
                    [
                        self._filter_display_frame(stack[k], panel=panel)
                        for k in range(int(stack.shape[0]))
                    ],
                    axis=0,
                )
            if not rgb:
                stack = self._pad_view_stack(stack)
            arr = np.ascontiguousarray(stack, dtype=np.float32)
            offsets.append(float_offset)
            float_offset += int(arr.size)
            if self.offline:
                finite = arr[np.isfinite(arr)]
                lo = float(finite.min()) if finite.size else 0.0
                hi = float(finite.max()) if finite.size else 1.0
                rng = hi - lo if hi > lo else 1.0
                blocks.append(
                    np.clip((arr - lo) * (255.0 / rng), 0, 255)
                    .astype(np.uint8)
                    .tobytes()
                )
                mins[panel] = lo
                maxs[panel] = hi
            else:
                blocks.append(arr.tobytes())
        self.panel_stack_offsets = offsets
        self._panel_stack_mins = mins if self.offline else []
        self._panel_stack_maxs = maxs if self.offline else []
        self.panel_stack_bytes = _b64_safe(b"".join(blocks))

    def _apply_rotations(self):
        """Re-rotate each displayed image from its original by ``image_rotations[i] * 90°``.

        This is purely a display-time reorientation of each 2D image via
        ``np.rot90`` — it is NOT scan rotation (which would rotate the
        scan grid in a 4D-STEM dataset). Originals are kept in
        ``_data_original`` so successive rotations compose from the
        unrotated source rather than accumulating interpolation error.
        Mixed shapes after rotation are center-padded to a common size.
        """
        has_local_stacks = any(
            count > 1 for count in getattr(self, "panel_frame_counts", [])
        )
        if has_local_stacks:
            has_rotation = any(
                (self.image_rotations[i] if i < len(self.image_rotations) else 0) % 4 != 0
                for i in range(len(self._panel_stacks_original))
            )
            if not has_rotation and self._panel_stack_originals_are_views:
                return
            if self._panel_stack_originals_are_views and has_rotation:
                self._panel_stacks_original = [
                    stack.copy() for stack in self._panel_stacks_original
                ]
                self._panel_stack_originals_are_views = False
            rotated_stacks = []
            for panel, original in enumerate(self._panel_stacks_original):
                k = (
                    self.image_rotations[panel]
                    if panel < len(self.image_rotations)
                    else 0
                ) % 4
                rotated_stacks.append(
                    original if k == 0 else np.rot90(original, k=k, axes=(-2, -1))
                )
            target_h = max(int(stack.shape[-2]) for stack in rotated_stacks)
            target_w = max(int(stack.shape[-1]) for stack in rotated_stacks)
            normalized_stacks = []
            for stack in rotated_stacks:
                if stack.shape[-2:] == (target_h, target_w):
                    normalized_stacks.append(np.asarray(stack, dtype=np.float32))
                else:
                    normalized_stacks.append(
                        np.stack(
                            [_resize_image(frame, target_h, target_w) for frame in stack],
                            axis=0,
                        ).astype(np.float32, copy=False)
                    )
            self._panel_stacks = normalized_stacks
            indices = list(self.panel_frame_indices)
            self._data = np.stack(
                [stack[index] for stack, index in zip(normalized_stacks, indices)],
                axis=0,
            ).astype(np.float32, copy=False)
            if self._display_bin > 1:
                from quantem.widget.utils.array import bin2d

                self._display_panel_stacks = [
                    bin2d(stack, factor=self._display_bin, mode="mean")
                    for stack in normalized_stacks
                ]
            else:
                self._display_panel_stacks = normalized_stacks
            self._display_data = np.stack(
                [stack[index] for stack, index in zip(self._display_panel_stacks, indices)],
                axis=0,
            ).astype(np.float32, copy=False)
            self._data_original = [self._data[i] for i in range(self._data.shape[0])]
            self._originals_are_views = True
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
            self._compute_all_stats()
            self._update_all_frames()
            return

        # Materialize originals as independent copies only when a non-zero
        # rotation exists (they start as views into _data to avoid 800MB copy at init)
        has_rotation = any(
            (self.image_rotations[i] if i < len(self.image_rotations) else 0) % 4 != 0
            for i in range(len(self._data_original))
        )
        # No-rotation fast path: skip 30+ MB of redundant tobytes + stats recomputation
        # on every widget init.  The observer fires once when image_rotations = [0]*n
        # is assigned in __init__; without this guard that triggered a full frame
        # rebuild + stats recompute for a no-op.
        if not has_rotation and self._originals_are_views:
            return
        if self._originals_are_views and has_rotation:
            self._data_original = [img.copy() for img in self._data_original]
            self._originals_are_views = False
        rotated = []
        for i, orig in enumerate(self._data_original):
            k = self.image_rotations[i] if i < len(self.image_rotations) else 0
            k = k % 4
            if k == 0:
                rotated.append(orig)
            else:
                rotated.append(np.rot90(orig, k=k))
        # If shapes differ after rotation, center-pad all to max dims
        shapes = [img.shape for img in rotated]
        if len(set(shapes)) > 1:
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)
            padded = []
            for img in rotated:
                h, w = img.shape
                pad_top = (max_h - h) // 2
                pad_bot = max_h - h - pad_top
                pad_left = (max_w - w) // 2
                pad_right = max_w - w - pad_left
                padded.append(np.pad(img, ((pad_top, pad_bot), (pad_left, pad_right)), mode="constant", constant_values=0))
            rotated = padded
        self._data = np.stack(rotated).astype(np.float32)
        # Recompute display data if binning is active
        if self._display_bin > 1:
            from quantem.widget.utils.array import bin2d
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
        else:
            self._display_data = self._data
        display = self._display_data if self._display_data is not None else self._data
        self.height = int(display.shape[1])
        self.width = int(display.shape[2])
        self._compute_all_stats()
        self._update_all_frames()

    @traitlets.observe("image_rotations")
    def _on_image_rotations_changed(self, change):
        if hasattr(self, "_data_original"):
            self._apply_rotations()

    def rotate(self, idx: int, angle: int) -> Self:
        """Rotate image ``idx`` by ``angle`` degrees (CCW-positive, matches np.rot90).

        Rotation convention follows ``np.rot90``::

            angle | image_rotations | np.rot90 k | direction
            ------+-----------------+------------+----------
              90  |        1        |     1      | 90° CCW
             180  |        2        |     2      | 180°
             -90  |        3        |     3      | 90° CW
             360  |        0        |     0      | identity

        Parameters
        ----------
        idx : int
            Image index in the gallery (0-based).
        angle : int
            Rotation angle in degrees (must be a multiple of 90).
            Positive = counter-clockwise, negative = clockwise.

        Returns
        -------
        Self
        """
        if any(self.is_rgb):
            raise NotImplementedError(
                "rotate() is not supported when the gallery contains RGB panels; "
                "rotation rebuilds panels from the grayscale stack and would desync the color pixels."
            )
        if angle % 90 != 0:
            raise ValueError(f"Rotation angle must be a multiple of 90 (got {angle}). Use 0, 90, 180, 270, or -90, -180, -270.")
        if idx < 0 or idx >= self.n_images:
            raise IndexError(f"Image index {idx} out of range [0, {self.n_images})")
        k = (angle // 90) % 4
        rots = list(self.image_rotations)
        while len(rots) < self.n_images:
            rots.append(0)
        rots[idx] = (rots[idx] + k) % 4
        self.image_rotations = rots
        return self

    def _sample_profile(self, row0, col0, row1, col1):
        img = self._data[self.selected_idx]
        h, w = img.shape
        dc, dr = col1 - col0, row1 - row0
        length = (dc**2 + dr**2) ** 0.5
        n = max(2, int(np.ceil(length)))
        t = np.linspace(0, 1, n)
        cs = col0 + t * dc
        rs = row0 + t * dr
        ci = np.floor(cs).astype(int)
        ri = np.floor(rs).astype(int)
        cf = cs - ci
        rf = rs - ri
        c0c = np.clip(ci, 0, w - 1)
        c1c = np.clip(ci + 1, 0, w - 1)
        r0c = np.clip(ri, 0, h - 1)
        r1c = np.clip(ri + 1, 0, h - 1)
        return (img[r0c, c0c] * (1 - cf) * (1 - rf) +
                img[r0c, c1c] * cf * (1 - rf) +
                img[r1c, c0c] * (1 - cf) * rf +
                img[r1c, c1c] * cf * rf).astype(np.float32)

    def set_profile(self, start: tuple, end: tuple):
        """Set a line profile between two points (image pixel coordinates).

        Parameters
        ----------
        start : tuple of (row, col)
            Start point in pixel coordinates.
        end : tuple of (row, col)
            End point in pixel coordinates.
        """
        row0, col0 = start
        row1, col1 = end
        self.profile_line = [
            {"row": float(row0), "col": float(col0)},
            {"row": float(row1), "col": float(col1)},
        ]

    def clear_profile(self):
        """Clear the current line profile."""
        self.profile_line = []

    def _upsert_selected_roi(self, updates: dict):
        rois = list(self.roi_list)
        color_cycle = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"]
        defaults = {
            "shape": "square",
            "row": int(self.height // 2),
            "col": int(self.width // 2),
            "radius": 10,
            "radius_inner": 5,
            "width": 20,
            "height": 20,
            "line_width": 2,
            "highlight": False,
            "visible": True,
            "locked": False,
        }
        if self.roi_selected_idx >= 0 and self.roi_selected_idx < len(rois):
            current = {**defaults, **rois[self.roi_selected_idx]}
            if not current.get("color"):
                current["color"] = color_cycle[self.roi_selected_idx % len(color_cycle)]
            rois[self.roi_selected_idx] = {**current, **updates}
        else:
            rois.append({**defaults, "color": color_cycle[len(rois) % len(color_cycle)], **updates})
            self.roi_selected_idx = len(rois) - 1
        self.roi_list = rois
        self.roi_active = True

    def add_roi(self, row: int | None = None, col: int | None = None, shape: str = "square") -> Self:
        with self.hold_sync():
            self.roi_selected_idx = -1
            self._upsert_selected_roi({
                "shape": shape,
                "row": int(self.height // 2 if row is None else row),
                "col": int(self.width // 2 if col is None else col),
            })
        return self

    def clear_rois(self) -> Self:
        with self.hold_sync():
            self.roi_list = []
            self.roi_selected_idx = -1
            self.roi_active = False
        return self

    def delete_selected_roi(self) -> Self:
        idx = int(self.roi_selected_idx)
        if idx < 0 or idx >= len(self.roi_list):
            return self
        with self.hold_sync():
            rois = [roi for i, roi in enumerate(self.roi_list) if i != idx]
            self.roi_list = rois
            self.roi_selected_idx = min(idx, len(rois) - 1) if rois else -1
            if not rois:
                self.roi_active = False
        return self

    def get_roi_geometries(self, *, visible_only: bool = True) -> list[dict[str, Any]]:
        """Return normalized ROI geometry in image ``(row, col)`` coordinates.

        The raw ``roi_list`` trait remains synced for widget state, while this
        helper gives notebooks, reports, and agents a stable public shape for
        downstream measurements. Bounds are reported in the image coordinate
        system; ``bounds_clipped`` is clamped to the current image extent for
        code that wants to slice arrays safely.

        Parameters
        ----------
        visible_only : bool, default=True
            If ``True``, omit ROIs whose synced state has ``visible=False``.

        Returns
        -------
        list of dict
            JSON-friendly ROI descriptions. Circle ROIs include ``center`` and
            ``radius``. Rectangles and squares include ``corners`` in clockwise
            order from the top-left. Annular ROIs include ``radius_inner`` and
            ``radius_outer``.
        """
        return roi_geometries(
            list(self.roi_list),
            height=self.height,
            width=self.width,
            visible_only=visible_only,
        )

    roi_geometries = get_roi_geometries

    def set_roi(self, row: int, col: int, radius: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "circle", "row": int(row), "col": int(col), "radius": int(radius)})
        return self

    def roi_circle(self, radius: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "circle", "radius": int(radius)})
        return self

    def roi_square(self, half_size: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "square", "radius": int(half_size)})
        return self

    def roi_rectangle(self, width: int = 20, height: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "rectangle", "width": int(width), "height": int(height)})
        return self

    def roi_annular(self, inner: int = 5, outer: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "annular", "radius_inner": int(inner), "radius": int(outer)})
        return self

    @property
    def profile(self):
        """Get profile line endpoints as [(row0, col0), (row1, col1)] or [].

        Returns
        -------
        list of tuple
            Line endpoints in pixel coordinates, or empty list if no profile.
        """
        return [(p["row"], p["col"]) for p in self.profile_line]

    @property
    def profile_values(self):
        """Get intensity values along the profile line as a numpy array.

        Returns
        -------
        np.ndarray or None
            Float32 array of sampled intensities, or None if no profile.
        """
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        return self._sample_profile(p0["row"], p0["col"], p1["row"], p1["col"])

    @property
    def profile_distance(self):
        """Get total distance of the profile line in calibrated units.

        Returns
        -------
        float or None
            Distance in angstroms (if pixel_size > 0) or pixels.
            None if no profile line is set.
        """
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        dc = p1["col"] - p0["col"]
        dr = p1["row"] - p0["row"]
        dist_px = (dc**2 + dr**2) ** 0.5
        if self.pixel_size > 0:
            return dist_px * self.pixel_size
        return dist_px
