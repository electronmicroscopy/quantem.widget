"""dataset.yaml schema helpers — single source of truth per session.

A session lives at ``<data-root>/<source>/<YYYYMMDD_sample>/`` and
owns ONE ``dataset.yaml`` file. Multi-condition sessions (e.g. light
on/off, dose-series, dark references) encode their tagging inside that
single yaml — never split into per-condition yamls.

Schema (additive, schema_version=1):

    session:
      name: mos2
      date: 2026-03-06
      operator: alice

    conditions:                        # NEW: condition labels
      light:    {description: "..."}
      light_5x: {description: "..."}
      dark:     {description: "..."}

    folders:                           # NEW: tag whole subfolder at once
      light1: {condition: light}
      light2: {condition: light}
      light3: {condition: light_5x}
      dark1:  {condition: dark, skip: true}

    files:                             # explicit per-master overrides folder
      '00':    {mag: 3p6, condition: light}
      '03-08': {condition: light_5x}

Resolution order at lookup time (most-specific wins):
  1. ``files:`` exact match (e.g. ``'07'``)
  2. ``files:`` range match (e.g. ``'03-08'``)
  3. ``folders:`` lookup by master's first-level folder under session_dir
  4. else: ``None`` (acquisition is unconditioned)

This module is the canonical reader. All callers (screen, ptycho_config,
acquisitions_router, dashboard) go through ``resolve_condition()`` so a
single yaml change updates every downstream surface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from quantem.widget.io.schema import check_schema_version


@dataclass
class ConditionInfo:
    """Resolved condition for a single master file."""
    label: str | None = None
    description: str = ""
    skip: bool = False
    folder: str | None = None              # which folder it came from, if folder-derived
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "none"                   # "files" | "folders" | "none"


def load_dataset_yaml(session_dir: Path) -> dict:
    """Read ``<session_dir>/dataset.yaml`` and return parsed dict.

    Returns ``{}`` if the file does not exist (caller decides whether
    that is fatal). Schema version is checked but not enforced — a
    higher version still parses, just emits a warning.
    """
    path = Path(session_dir) / "dataset.yaml"
    if not path.is_file():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    check_schema_version(data, str(path))
    return data


def _file_num_from_master(master_path: Path) -> int | None:
    """Extract integer file number from master filename.

    Matches the trailing ``_NNNN_master.h5`` convention that Arina writes.
    Returns None if the filename does not follow the convention.
    """
    m = re.search(r"_(\d+)_master\.h5$", master_path.name)
    return int(m.group(1)) if m else None


def _files_entry_for(files_block: dict, file_num: int) -> dict | None:
    """Find a ``files:`` entry matching the given file number.

    Exact match (``'07'`` or ``7``) preferred over range match
    (``'03-08'``). Returns the entry dict or None.
    """
    exact_hit = None
    range_hit = None
    for key, val in (files_block or {}).items():
        s = str(key).strip()
        if "-" in s:
            try:
                lo, hi = s.split("-", 1)
                if int(lo) <= file_num <= int(hi):
                    range_hit = val or {}
            except ValueError:
                continue
        else:
            try:
                if int(s) == file_num:
                    exact_hit = val or {}
                    break
            except ValueError:
                continue
    return exact_hit if exact_hit is not None else range_hit


def _folder_for_master(master_path: Path, session_dir: Path) -> str | None:
    """Return the first-level folder name under session_dir, or None.

    Example: session_dir=``/data/alice/20260101_session``, master at
    ``light3/MoS2_007_master.h5`` → returns ``"light3"``. A master sitting
    in the session root returns None.
    """
    try:
        rel = Path(master_path).resolve().relative_to(Path(session_dir).resolve())
    except (ValueError, OSError):
        return None
    parts = rel.parts
    return parts[0] if len(parts) >= 2 else None


def resolve_condition(
    dataset_yaml: dict,
    master_path: Path,
    session_dir: Path,
) -> ConditionInfo:
    """Resolve the condition tag for a single master file.

    See module docstring for resolution order. Always returns a
    ConditionInfo (never raises for missing config); ``label=None`` and
    ``source="none"`` indicates no condition was specified.
    """
    conditions = (dataset_yaml.get("conditions") or {}) if dataset_yaml else {}
    folders = (dataset_yaml.get("folders") or {}) if dataset_yaml else {}
    files_block = (dataset_yaml.get("files") or {}) if dataset_yaml else {}

    file_num = _file_num_from_master(Path(master_path))
    file_entry = _files_entry_for(files_block, file_num) if file_num is not None else None
    if file_entry and "condition" in file_entry:
        label = file_entry["condition"]
        cond_meta = conditions.get(label, {}) or {}
        return ConditionInfo(
            label=label,
            description=cond_meta.get("description", ""),
            skip=bool(file_entry.get("skip", cond_meta.get("skip", False))),
            metadata=cond_meta,
            source="files",
        )

    folder = _folder_for_master(Path(master_path), Path(session_dir))
    folder_entry = folders.get(folder) if folder else None
    if folder_entry and "condition" in folder_entry:
        label = folder_entry["condition"]
        cond_meta = conditions.get(label, {}) or {}
        return ConditionInfo(
            label=label,
            description=cond_meta.get("description", ""),
            skip=bool(folder_entry.get("skip", cond_meta.get("skip", False))),
            folder=folder,
            metadata=cond_meta,
            source="folders",
        )

    return ConditionInfo(folder=folder, source="none")


def list_conditions(dataset_yaml: dict) -> list[dict]:
    """Return condition labels + descriptions as a list of dicts.

    Used by the dashboard to render a condition legend / chip filter.
    """
    out = []
    for label, meta in (dataset_yaml.get("conditions") or {}).items():
        meta = meta or {}
        out.append({
            "label": str(label),
            "description": meta.get("description", ""),
            **{k: v for k, v in meta.items() if k != "description"},
        })
    return out
