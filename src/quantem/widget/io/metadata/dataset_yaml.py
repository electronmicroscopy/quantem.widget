"""dataset.yaml schema helpers: single source of truth per session.

A session lives at ``<data-root>/<source>/<YYYYMMDD_sample>/`` and
owns ONE ``dataset.yaml`` file. Multi-condition sessions (e.g. light
on/off, dose-series, dark references) encode their tagging inside that
single yaml, never split into per-condition yamls.

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

This module is the canonical reader. All acquisition and reconstruction callers
go through ``resolve_condition()`` so a
single yaml change updates every downstream surface.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Highest ``schema_version`` this reader knows; a newer file still parses with a warning.
SCHEMA_VERSION = 1


@dataclass
class ConditionInfo:
    """Resolved condition for a single master file."""
    label: str | None = None
    description: str = ""
    skip: bool = False
    folder: str | None = None              # which folder it came from, if folder-derived
    metadata: dict[str, object] = field(default_factory=dict)
    source: str = "none"                   # "files" | "folders" | "none"


def load_dataset_yaml(session_dir: Path) -> dict:
    """Read ``<session_dir>/dataset.yaml`` and return parsed dict.

    Returns ``{}`` if the file does not exist (caller decides whether
    that is fatal). Schema version is checked but not enforced: a
    higher version still parses, just emits a warning.
    """
    path = Path(session_dir) / "dataset.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml  # only session tooling (quantem.live) reads dataset.yaml; the widgets never do
    except ImportError as exc:
        raise ImportError("Reading dataset.yaml needs PyYAML: pip install pyyaml") from exc
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    if "schema_version" not in data:
        raise ValueError(f"{path} missing schema_version")
    if int(data["schema_version"]) > SCHEMA_VERSION:
        logger.warning("%s schema_version=%s is newer than this reader knows (max=%d)", path, data["schema_version"], SCHEMA_VERSION)
    return data


def _file_num_from_master(master_path: Path) -> int | None:
    """Extract integer file number from master filename.

    Matches the trailing ``_NNNN_master.h5`` convention that Arina writes.
    Returns None if the filename does not follow the convention.
    """
    match = re.search(r"_(\d+)_master\.h5$", master_path.name)
    return int(match.group(1)) if match else None


def _files_entry_for(files_block: dict, file_num: int) -> dict | None:
    """Find a ``files:`` entry matching the given file number.

    Exact match (``'07'`` or ``7``) preferred over range match
    (``'03-08'``). Returns the entry dict or None.
    """
    exact_hit = None
    range_hit = None
    for key, entry in (files_block or {}).items():
        key_text = str(key).strip()
        if "-" in key_text:
            try:
                first, last = key_text.split("-", 1)
                if int(first) <= file_num <= int(last):
                    range_hit = entry or {}
            except ValueError:
                continue
        else:
            try:
                if int(key_text) == file_num:
                    exact_hit = entry or {}
                    break
            except ValueError:
                continue
    return exact_hit if exact_hit is not None else range_hit


def _folder_for_master(master_path: Path, session_dir: Path) -> str | None:
    """Return the first-level folder name under session_dir, or None.

    Example: session_dir=``/data/alice/20260101_session``, master at
    ``light3/scan_007_master.h5`` → returns ``"light3"``. A master sitting
    in the session root returns None.
    """
    try:
        relative = Path(master_path).resolve().relative_to(Path(session_dir).resolve())
    except (ValueError, OSError):
        return None
    parts = relative.parts
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
    dataset_yaml = dataset_yaml or {}
    conditions = dataset_yaml.get("conditions") or {}
    folders = dataset_yaml.get("folders") or {}
    files_block = dataset_yaml.get("files") or {}

    file_num = _file_num_from_master(Path(master_path))
    file_entry = _files_entry_for(files_block, file_num) if file_num is not None else None
    if file_entry and "condition" in file_entry:
        return _tagged_condition(file_entry, conditions, source="files")

    folder = _folder_for_master(Path(master_path), Path(session_dir))
    folder_entry = folders.get(folder) if folder else None
    if folder_entry and "condition" in folder_entry:
        return _tagged_condition(folder_entry, conditions, source="folders", folder=folder)

    return ConditionInfo(folder=folder, source="none")


def _tagged_condition(entry: dict, conditions: dict, *, source: str, folder: str | None = None) -> ConditionInfo:
    """The ``ConditionInfo`` of a ``files:`` or ``folders:`` entry that names a condition.

    The description and metadata come from the ``conditions:`` block; the
    entry's own ``skip`` wins over the condition's, so one folder of a
    condition can be skipped without skipping the rest.
    """
    label = entry["condition"]
    condition = conditions.get(label, {}) or {}
    return ConditionInfo(
        label=label,
        description=condition.get("description", ""),
        skip=bool(entry.get("skip", condition.get("skip", False))),
        folder=folder,
        metadata=condition,
        source=source,
    )
