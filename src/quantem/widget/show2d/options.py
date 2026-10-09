"""Show2D constructor option normalisers: panel title spans and style, row and
column markers, inset plots, text annotations and shape overlays.

Each ``_normalize_*`` function turns the loosely typed constructor value (a
mapping keyed by panel index or label, a flat sequence, or one spec) into the
exact list shape the browser renders, raising ``ValueError`` on typos so a
misspelled key never silently drops an annotation from a figure. Show3D
imports several of these for its own gallery options.
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np

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


def _finite_values(value: object, *, count: int, message: str) -> list[float]:
    """``value`` flattened to exactly ``count`` finite floats, else ``ValueError(message)``.

    Offsets, axis limits, points and boxes all pass through here, so a NaN or a
    wrong length fails at construction under the option's own name instead of
    reaching the browser as a NaN coordinate.
    """
    values = np.asarray(value, dtype=np.float64).ravel()
    if values.size != count or not np.isfinite(values).all():
        raise ValueError(message)
    return [float(item) for item in values]


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
            out[key_text] = _finite_values(value, count=2, message="panel_title_style offset must contain two finite values")
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
            out[key_text] = _finite_values(value, count=2, message="scale_bar_style offset must contain two finite values")
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
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} index must be a nonnegative integer, got {key!r}") from exc
        if index < 0:
            raise ValueError(f"{name} index must be >= 0, got {index}")
        out[str(index)] = str(value)
    return out


def _title_span(item: Mapping[str, object]) -> tuple[str, dict[str, str]]:
    """One rich span: its plain text plus the ``{"text"|"math", "color"}`` dict the browser draws.

    A non-empty ``math`` entry wins over ``text`` because the browser typesets
    it; the plain text keeps the raw LaTeX so label lookup and exports still
    have a string.
    """
    if item.get("math") not in (None, ""):
        text = str(item.get("math"))
        span: dict[str, str] = {"math": text}
    else:
        text = "" if item.get("text") is None else str(item.get("text"))
        span = {"text": text}
    color = item.get("color")
    if color not in (None, ""):
        span["color"] = str(color)
    return text, span


def _is_one_rich_title(values: object) -> bool:
    """Whether a title input is ONE rich title (a span dict or a list of span dicts), not a per-panel list.

    ``labels=[{"text": "a"}, {"text": "b"}]`` is one two-span title, so the
    per-panel normalisers must wrap it instead of reading two panels.
    """
    return isinstance(values, Mapping) or (
        isinstance(values, (list, tuple)) and bool(values) and all(isinstance(item, Mapping) for item in values)
    )


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
        text, span = _title_span(value)
        return text, [span]
    if isinstance(value, (list, tuple)):
        spans: list[dict[str, str]] = []
        plain_parts: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise TypeError(
                    "rich panel title spans must be dictionaries like "
                    f"{{'text': 'low', 'color': '#60a5fa'}}, got {type(item).__name__} "
                    f"at span {index}"
                )
            text, span = _title_span(item)
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
    if _is_one_rich_title(values):
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
    if _is_one_rich_title(values):
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
    """Broadcast rich title spans across paged panel layouts.

    Titles given once per page slot repeat on every page, matching how plain
    ``labels`` of length ``panels_per_page`` broadcast; any other length is
    returned as is.
    """
    if n_pages > 1 and panels_per_page > 0 and original_len == len(spans) == panels_per_page:
        return [list(span) for _ in range(n_pages) for span in spans]
    return spans


def _normalize_inset_plot_specs(
    inset_plots: Sequence[dict[str, object] | None] | dict[str, object] | None,
    *,
    n_items: int,
) -> list[dict[str, object]]:
    """Return JSON-safe per-panel inset plot specifications.

    The public API intentionally mirrors the smallest useful slice of a
    matplotlib line plot: ``x``, ``y``, optional ``point``, optional
    ``xlim``/``ylim``, and simple style/placement keys.  Arrays are converted
    to plain lists so the same trait survives notebook state and HTML export.
    """
    if isinstance(inset_plots, dict):
        raw_specs: list[dict[str, object] | None] = [inset_plots]
    else:
        raw_specs = list(inset_plots or [])
    if not raw_specs:
        return []
    if len(raw_specs) == 1 and n_items > 1:
        raw_specs = raw_specs * n_items
    if len(raw_specs) != int(n_items):
        raise ValueError(
            f"inset_plots length ({len(raw_specs)}) must be 1 or match the "
            f"number of Show2D panels ({int(n_items)})"
        )

    normalized: list[dict[str, object]] = []
    for panel, spec in enumerate(raw_specs):
        if not spec:
            normalized.append({})
            continue
        if not isinstance(spec, dict):
            raise TypeError(f"inset_plots[{panel}] must be a dict or None")
        raw_x = spec.get("x")
        raw_y = spec.get("y")
        if raw_y is None and "points" in spec:
            points = np.asarray(spec["points"], dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError(
                    f"inset_plots[{panel}]['points'] must have shape (N, 2)"
                )
            x_values = points[:, 0]
            y_values = points[:, 1]
        else:
            if raw_y is None:
                raise ValueError(f"inset_plots[{panel}] must include 'y' or 'points'")
            y_values = np.asarray(raw_y, dtype=np.float64).ravel()
            x_values = (
                np.arange(y_values.size, dtype=np.float64)
                if raw_x is None
                else np.asarray(raw_x, dtype=np.float64).ravel()
            )
        if x_values.size != y_values.size or x_values.size < 2:
            raise ValueError(
                f"inset_plots[{panel}] x/y must have the same length >= 2; "
                f"got {x_values.size} and {y_values.size}"
            )
        if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            raise ValueError(f"inset_plots[{panel}] contains NaN or inf")
        out: dict[str, object] = {
            "x": [float(value) for value in x_values],
            "y": [float(value) for value in y_values],
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
                out[key] = _finite_values(spec[key], count=2, message=f"inset_plots[{panel}]['{key}'] must contain two finite values")
        if spec.get("box") is not None:
            left, top, width, height = _finite_values(spec["box"], count=4, message=f"inset_plots[{panel}]['box'] must contain four finite values")
            out["box"] = [
                max(0.0, min(1.0, left)),
                max(0.0, min(1.0, top)),
                max(0.05, min(1.0, width)),
                max(0.05, min(1.0, height)),
            ]
        for key in ("xticks", "yticks"):
            if key in spec and spec[key] is not None:
                ticks = np.asarray(spec[key], dtype=np.float64).ravel()
                if ticks.size < 1 or not np.isfinite(ticks).all():
                    raise ValueError(f"inset_plots[{panel}]['{key}'] must contain finite values")
                out[key] = [float(tick) for tick in ticks]
        if "margin" in spec and spec["margin"] is not None:
            margin = np.asarray(spec["margin"], dtype=np.float64).ravel()
            if margin.size == 1:
                margin = np.repeat(margin, 2)
            if margin.size != 2 or not np.isfinite(margin).all():
                raise ValueError(
                    f"inset_plots[{panel}]['margin'] must be one number or two finite values"
                )
            out["margin"] = [max(0.0, float(margin[0])), max(0.0, float(margin[1]))]
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
_ANNOTATION_ANCHORS = _ANNOTATION_POSITIONS


def _is_annotation_spec(value: object) -> bool:
    """Return True when a mapping looks like one annotation spec."""
    return isinstance(value, Mapping) and any(str(key) in _ANNOTATION_STYLE_KEYS for key in value)


def _panel_target_index(panel: object, *, labels: Sequence[str] | None, n_items: int, kind: str) -> int:
    """Resolve an annotation or overlay target from an integer index or a panel label.

    ``kind`` (``"annotation"`` / ``"overlay"``) names the option in the error,
    so a bad target points the user at the argument that carried it.
    """
    if isinstance(panel, bool):
        raise ValueError(f"panel {kind} panel must be an index or label, got {panel!r}")
    if isinstance(panel, str) and labels is not None and panel in labels:
        return list(labels).index(panel)
    try:
        index = int(panel)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"panel {kind} panel must be an index or label, got {panel!r}") from exc
    if index < 0 or index >= int(n_items):
        raise ValueError(f"panel {kind} panel index {index} is outside 0..{int(n_items) - 1}")
    return index


def _normalize_panel_indices(
    panels: Sequence[object] | object | None,
    *,
    labels: Sequence[str] | None,
    n_items: int,
) -> list[int]:
    """Resolve optional panel index/label selectors to unique panel indices."""
    if panels is None:
        return []
    if isinstance(panels, (str, bytes)) or not isinstance(panels, Sequence):
        raw_values = [panels]
    else:
        raw_values = list(panels)
    indices = [_panel_target_index(raw, labels=labels, n_items=n_items, kind="annotation") for raw in raw_values]
    return list(dict.fromkeys(indices))


def _normalize_panel_annotation_spec(spec: object, *, panel: int) -> dict[str, object] | None:
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
    raw_spans = spec.get("spans")
    if raw_spans is not None:
        text, spans = _normalise_title_spans(raw_spans)
    elif raw_math not in (None, ""):
        text = str(raw_math)
        spans = [{"math": text}]
    else:
        text, spans = _normalise_title_spans(raw_text)
    out: dict[str, object] = {"text": text}
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
    out.setdefault("variant", "badge")
    for key in ("x", "y", "border_width", "font_size", "pad_x", "pad_y", "radius", "opacity", "outline_width"):
        if key in spec and spec[key] is not None:
            value = float(spec[key])
            if not np.isfinite(value):
                raise ValueError(f"panel annotation {key} must be finite, got {value!r}")
            if key in {"x", "y", "opacity"}:
                value = max(0.0, min(1.0, value))
            out[key] = value
    raw_box = spec.get("box", spec.get("region"))
    if raw_box is not None:
        left, top, width, height = _finite_values(raw_box, count=4, message="panel annotation box/region must contain four finite values")
        out["box"] = [
            max(0.0, min(1.0, left)),
            max(0.0, min(1.0, top)),
            max(0.01, min(1.0, width)),
            max(0.01, min(1.0, height)),
        ]
    if "offset" in spec and spec["offset"] is not None:
        out["offset"] = _finite_values(spec["offset"], count=2, message="panel annotation offset must contain two finite values")
    return out


def _normalize_panel_annotations(
    panel_annotations: Sequence[object] | Mapping[object, object] | object | None,
    *,
    n_items: int,
    labels: Sequence[str] | None = None,
) -> list[list[dict[str, object]]]:
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
    grouped: list[list[dict[str, object]]] = [[] for _ in range(int(n_items))]

    def add(panel: int, value: object) -> None:
        """Append one annotation, or each of a list, to ``panel``; empty text draws nothing so it is dropped."""
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            normalized = _normalize_panel_annotation_spec(item, panel=panel)
            if normalized is not None and normalized.get("text", ""):
                grouped[panel].append(normalized)

    if isinstance(panel_annotations, Mapping) and not _is_annotation_spec(panel_annotations):
        for raw_panel, value in panel_annotations.items():
            add(_panel_target_index(raw_panel, labels=labels, n_items=n_items, kind="annotation"), value)
        return grouped
    if isinstance(panel_annotations, (str, Mapping)):
        for panel in range(int(n_items)):
            add(panel, panel_annotations)
        return grouped

    raw = list(panel_annotations)
    if not raw:
        return []
    _route_panel_list(raw, add, n_items=n_items, labels=labels, kind="annotation", option="panel_annotations")
    return grouped


def _route_panel_list(
    raw: list[object],
    add: Callable[[int, object], None],
    *,
    n_items: int,
    labels: Sequence[str] | None,
    kind: str,
    option: str,
) -> None:
    """Send each entry of a list-valued annotation or overlay option to its panel through ``add``.

    One panel takes every entry; a list as long as the gallery is per panel;
    otherwise every entry must name its ``panel=``. Without the explicit
    ``panel=`` rule a flat list of the wrong length would land on the wrong panels.
    """
    flat_with_panel = any(isinstance(item, Mapping) and "panel" in item for item in raw)
    if int(n_items) == 1 and not flat_with_panel:
        for item in raw:
            add(0, item)
        return
    if len(raw) == int(n_items) and not flat_with_panel:
        for panel, value in enumerate(raw):
            add(panel, value)
        return
    for item in raw:
        if not isinstance(item, Mapping) or "panel" not in item:
            raise ValueError(
                f"{option} as a flat list must include panel=... on every entry, "
                "or pass a per-panel list/dict"
            )
        add(_panel_target_index(item["panel"], labels=labels, n_items=n_items, kind=kind), item)


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
# Spellings users type for a stroke pattern, mapped to the four the browser draws.
_LINE_STYLE_ALIASES = {
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


def _is_overlay_spec(value: object) -> bool:
    """Return True when a mapping looks like one geometric overlay spec."""
    return isinstance(value, Mapping) and any(str(key) in _OVERLAY_STYLE_KEYS for key in value)


def _finite_float(value: object, *, name: str) -> float:
    """Return one finite float with a useful user-facing error."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"panel overlay {name} must be finite, got {value!r}")
    return out


def _finite_float_array(value: object, *, name: str, count: int) -> list[float]:
    """Return a fixed-length finite float list, with the overlay key ``name`` in the error."""
    return _finite_values(value, count=count, message=f"panel overlay {name} must contain {count} finite values")


def _finite_float_sequence(value: object, *, name: str) -> list[float]:
    """Return a non-empty finite float list for custom dash patterns.

    An all-zero or negative dash array makes the browser draw nothing or throw,
    so it is rejected here.
    """
    values = np.asarray(value, dtype=np.float64).ravel()
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"panel overlay {name} must contain finite values")
    out = [float(item) for item in values]
    if any(item < 0 for item in out):
        raise ValueError(f"panel overlay {name} values must be >= 0")
    if all(item == 0 for item in out):
        raise ValueError(f"panel overlay {name} must contain at least one positive value")
    return out


def _normalize_panel_overlay_spec(spec: object, *, panel: int) -> dict[str, object] | None:
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

    out: dict[str, object] = {"shape": shape, "coords": coords}
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
    if line_style not in _LINE_STYLE_ALIASES:
        raise ValueError(
            "panel overlay line_style must be one of "
            "['solid', 'dashed', 'dotted', 'dashdot'] or use dash=[...]"
        )
    out["line_style"] = _LINE_STYLE_ALIASES[line_style]
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
) -> list[list[dict[str, object]]]:
    """Normalize per-panel circle/rect overlays.

    Accepted forms mirror ``panel_annotations``:
    one overlay mapping broadcasts to all panels; a mapping keyed by panel
    index/label targets specific panels; a per-panel list aligns with panels;
    and a flat list of mappings with ``panel=...`` can target arbitrary panels.
    """
    if panel_overlays is None:
        return []
    grouped: list[list[dict[str, object]]] = [[] for _ in range(int(n_items))]

    def add(panel: int, value: object) -> None:
        """Append one overlay, or each of a list, to ``panel``."""
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            normalized = _normalize_panel_overlay_spec(item, panel=panel)
            if normalized is not None:
                grouped[panel].append(normalized)

    if isinstance(panel_overlays, Mapping) and not _is_overlay_spec(panel_overlays):
        for raw_panel, value in panel_overlays.items():
            add(_panel_target_index(raw_panel, labels=labels, n_items=n_items, kind="overlay"), value)
        return grouped
    if isinstance(panel_overlays, Mapping):
        if "panel" in panel_overlays:
            add(_panel_target_index(panel_overlays["panel"], labels=labels, n_items=n_items, kind="overlay"), panel_overlays)
        else:
            for panel in range(int(n_items)):
                add(panel, panel_overlays)
        return grouped

    raw = list(panel_overlays)
    if not raw:
        return []
    flat_with_panel = any(isinstance(item, Mapping) and "panel" in item for item in raw)
    if all(_is_overlay_spec(item) for item in raw) and not flat_with_panel:
        # A flat list of shapes without panel= draws every shape on every panel.
        for panel in range(int(n_items)):
            for item in raw:
                add(panel, item)
        return grouped
    _route_panel_list(raw, add, n_items=n_items, labels=labels, kind="overlay", option="panel_overlays")
    return grouped
