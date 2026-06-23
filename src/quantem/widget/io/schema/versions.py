"""Per-format schema version constants.

One named constant per persisted format so writers and readers agree on a
single source of truth. Bump when the on-disk shape changes; readers must
warn (not error) when they see a version greater than they know.

A future v2 reader can branch on the constant + a named migration hook
(e.g. ``Calibration.from_dict_v1``) instead of guessing from field shape.
"""
from __future__ import annotations

#: ``<screen_root>/_calibrations.json`` (per-screening-folder library) and
#: any other ``list[Calibration]`` library file.
CALIBRATIONS_SCHEMA_VERSION = 1

#: ``~/quantem/calibrations.json`` (global registry, one entry per
#: microscope preset). Already had a ``schema_version`` field before this
#: constant existed; pinning it here so the value lives in one place.
CALIBRATION_REGISTRY_SCHEMA_VERSION = 1
