"""The versioned JSON envelope around a widget's ``state_dict``: ``save`` writes it, ``load_state_dict`` reads it."""

import importlib.metadata
import json
import pathlib

JSON_METADATA_VERSION = "1.0"


def resolve_widget_version() -> str:
    """Installed quantem.widget version for the envelope; a source checkout without metadata is ``0.0.0+local``."""
    try:
        return importlib.metadata.version("quantem.widget")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+local"


def build_json_header(widget_name: str) -> dict[str, str]:
    """Envelope fields that name the format version and the widget that wrote the file."""
    return {
        "metadata_version": JSON_METADATA_VERSION,
        "widget_name": widget_name,
        "widget_version": resolve_widget_version(),
    }


def wrap_state_dict(widget_name: str, state: dict) -> dict:
    """``state`` inside the versioned envelope, so a reader can refuse another widget's file."""
    envelope = build_json_header(widget_name)
    envelope["state"] = state
    return envelope


def unwrap_state_payload(
    payload: dict,
    *,
    require_envelope: bool = False,
    expected_widget: str | None = None,
) -> dict:
    """The state dict inside an envelope, or a bare state dict as given.

    ``require_envelope`` is for files, which ``save`` always wraps; a dict
    passed in Python may be bare. ``expected_widget`` refuses an envelope
    written by another widget class.
    """
    if not isinstance(payload, dict):
        raise ValueError("State payload must be a dict.")
    if "state" in payload:
        state = payload["state"]
        if not isinstance(state, dict):
            raise ValueError("State envelope field 'state' must be a dict.")
        # If caller passed the widget name, refuse cross-widget loads
        # (Show2D state into Show3D would silently load wrong subset of traits).
        written_by = payload.get("widget_name")
        if expected_widget is not None and written_by is not None and written_by != expected_widget:
            raise ValueError(
                f"State envelope is for {written_by!r}, cannot load into {expected_widget!r}"
            )
        return state
    if require_envelope:
        raise ValueError("State JSON file must be a versioned envelope with top-level 'state'.")
    return payload


def _numpy_safe(value):
    """JSON fallback for values ``json.dumps`` rejects: a scalar's ``.item()``, anything else as text.

    NumPy scalars in ROI dicts (np.int64 from ``shape[0] // 2``) would
    otherwise raise TypeError in ``json.dumps``.
    """
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_state_file(path: str | pathlib.Path, widget_name: str, state: dict) -> None:
    """Write ``state`` in its envelope as indented JSON, creating the parent folder."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(wrap_state_dict(widget_name, state), indent=2, default=_numpy_safe))
