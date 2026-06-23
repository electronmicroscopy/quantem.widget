"""Schema-version helpers for persisted formats.

Every JSON / YAML file the live stack writes carries a ``schema_version``
integer at the top so a future agent reading a file knows which writer
produced it. There is no migration shim: current writers stamp version 1 and
readers fail on missing or future versions instead of guessing.

Files covered (one-line per format):

- ``dataset.yaml``                      session layout (human-authored, reader-side check only)
- ``<trial>/config.json``               ptycho trial config (writer in engine/ptycho/save.py)
- ``<trial>/events.jsonl``              ptycho hot-loop events (writer in cli/ptycho_worker.py)
- ``<acquisition>/config.json``         screener acquisition config (writer in screen.py)
- ``~/quantem/calibrations.json``       calibration registry (writer in dashboard/calibration_registry.py)
- ``calibration.json``                  single locked calibration (writer in calibration.py)

Calibration libraries (``_calibrations.json``) stay as JSON arrays; the array
shape itself is the version signal.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

#: Highest schema version the running code knows how to read. Bump this
#: any time the writer changes shape so the warning becomes a no-op for
#: the new format and stays loud for anything stamped beyond it.
SCHEMA_VERSION = 1


def stamp_schema_version(data: dict, version: int = SCHEMA_VERSION) -> dict:
    """Return a new dict with ``schema_version`` as the FIRST key.

    Use this at every JSON/YAML write site so the version field appears
    at the top of the file, easy to spot in a quick ``cat`` or ``head``.
    Pass the original payload, get back a fresh dict ready to dump.
    """
    return {"schema_version": int(version), **data}


def check_schema_version(
    data: dict,
    file_kind: str,
    max_known: int = SCHEMA_VERSION,
) -> int:
    """Read ``schema_version`` from a parsed dict and warn on future values."""
    if not isinstance(data, dict) or "schema_version" not in data:
        raise ValueError(f"{file_kind} missing schema_version")
    raw = data["schema_version"]
    try:
        version = int(raw)
    except (TypeError, ValueError):
        version = 1
    if version > max_known:
        logger.warning(
            "%s schema_version=%d is newer than this code knows (max=%d) - best effort",
            file_kind, version, max_known,
        )
    return version


def atomic_write_json(path: Path, data: object, *, indent: int = 2) -> None:
    """Atomic JSON write via tmp + ``os.replace``.

    Crash mid-write leaves the real file untouched, so a SIGKILL
    overnight cannot corrupt config.json / events.jsonl / calibration
    registries the operator depends on. Mirrors the pattern in
    calibration_registry.py and trials_router.py.

    ``data`` accepts anything ``json.dumps`` accepts (dict, list, ...);
    most callers pass dicts but drift history is a list of run entries.

    Self-invalidates any cached ``/api/trials`` list that owns ``path``
    so callers writing to a trial's ``config.json`` never need to
    remember to bust the cache. No-op for paths outside any tracked
    trials_dir.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent))
    os.replace(tmp, path)
    # Lazy-import to avoid a state ↔ schema import cycle at package-load.
    try:
        try:
            from quantem.live.server.core.state import bust_trials_cache_for_path
        except Exception:
            return  # dashboard-only cache hook; no-op in standalone widget
        bust_trials_cache_for_path(path)
    except ImportError:
        pass
