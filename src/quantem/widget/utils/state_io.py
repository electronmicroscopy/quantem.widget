"""The versioned JSON envelope around a widget's ``state_dict``: ``save`` writes it, ``load_state_dict`` reads it."""

import importlib.metadata
import json
import pathlib

import numpy as np

from quantem.widget import __version__

# The one layout of a save() file: {"schema", "widget_name", "widget_version", "state"}.
STATE_SCHEMA = "quantem.widget-state/2"


def read_state(state: dict | str | pathlib.Path, widget_name: str, keys) -> dict:
    """The state dict to restore from a ``save`` file, its envelope, or a bare ``state_dict``.

    A file must be an envelope (``save`` always wraps). An envelope written by
    another widget class is refused: Show3D keys loaded into Show2D would set
    a wrong subset of traits without a word.

    An envelope must name ``STATE_SCHEMA`` under ``schema``, and every key of
    the state must be one of ``keys``, the keys the widget's ``state_dict``
    writes. Either failure raises ValueError before the widget changes. A file
    of an earlier release has no ``schema`` and names traits renamed since;
    read without these checks it set the keys that still exist and dropped
    the rest without a message. It is not converted.
    """
    source = "State"
    if isinstance(state, (str, pathlib.Path)):
        source = f"State file {state}"
        state = json.loads(pathlib.Path(state).read_text())
        if not isinstance(state, dict) or "state" not in state:
            raise ValueError(f"{source} must be a versioned envelope with top-level 'state'.")
    if not isinstance(state, dict):
        raise ValueError("State payload must be a dict.")
    if "state" in state:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError(f"{source} has schema {state.get('schema')!r}; {widget_name} reads {STATE_SCHEMA!r} only. "
                             "A state saved by another release is not converted: set the view again and save() it.")
        written_by = state.get("widget_name")
        if written_by is not None and written_by != widget_name:
            raise ValueError(f"State envelope is for {written_by!r}, cannot load into {widget_name!r}")
        if not isinstance(state["state"], dict):
            raise ValueError("State envelope field 'state' must be a dict.")
        state = state["state"]
    unknown = [key for key in state if key not in keys]
    if unknown:
        raise ValueError(f"{source} has keys {widget_name} does not save: {unknown}. Nothing was applied; "
                         f"remove them, or read the keys of {widget_name}.state_dict().")
    return state


def resolve_widget_version() -> str:
    """Installed quantem.widget version; a source checkout without metadata is ``0.0.0+local`` (ShowPDF's state)."""
    try:
        return importlib.metadata.version("quantem.widget")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+local"


def unwrap_state_payload(payload: dict, *, require_envelope: bool = False, expected_widget: str | None = None) -> dict:
    """The state dict inside a ShowPDF ``save`` envelope, or a bare state dict as given.

    ShowPDF writes and reads its own envelope (``metadata_version`` 1.0);
    the other widgets use ``read_state`` and ``STATE_SCHEMA``.
    ``require_envelope`` is for files, which ``save`` always wraps.
    ``expected_widget`` refuses an envelope written by another widget class.
    """
    if not isinstance(payload, dict):
        raise ValueError("State payload must be a dict.")
    if "state" in payload:
        state = payload["state"]
        if not isinstance(state, dict):
            raise ValueError("State envelope field 'state' must be a dict.")
        written_by = payload.get("widget_name")
        if expected_widget is not None and written_by is not None and written_by != expected_widget:
            raise ValueError(f"State envelope is for {written_by!r}, cannot load into {expected_widget!r}")
        return state
    if require_envelope:
        raise ValueError("State JSON file must be a versioned envelope with top-level 'state'.")
    return payload


def _numpy_safe(value):
    """JSON fallback for values ``json.dumps`` rejects: a NumPy scalar's Python value, anything else as text.

    NumPy scalars in ROI dicts (np.int64 from ``shape[0] // 2``) would
    otherwise raise TypeError in ``json.dumps``.
    """
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def save_state_file(path: str | pathlib.Path, widget_name: str, state: dict) -> None:
    """Write ``state`` in its versioned envelope as indented JSON, creating the parent folder.

    The envelope names its layout (``schema``) and the widget that wrote the
    file, so ``read_state`` can refuse another widget's file and a file of
    another layout.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"schema": STATE_SCHEMA, "widget_name": widget_name, "widget_version": __version__, "state": state}
    path.write_text(json.dumps(envelope, indent=2, default=_numpy_safe))
