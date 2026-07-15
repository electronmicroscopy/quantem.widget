"""Build + validate quantem-data metadata sidecars (``quantem_meta.json``).

Two modality schemas (``4dstem`` vs ``haadf``) sharing a provenance block. The
rules, in order of importance:

- **acquisition-only**: store what the scope/operator set, never recon/probe
  outputs (no ``rotation_deg``, no ``det_sampling_mrad_per_px``).
- **primary-only**: never store a value derivable from another (no
  ``num_positions`` - it is ``scan_shape[0]*scan_shape[1]``).
- **present = known**: a field is written only when known; absent means
  "unknown, resolve it yourself". Never a guessed default.

Field names match quantem ptycho ``config.json`` / ``dataset.yaml`` exactly, and
unit suffixes follow the unit symbol's official capitalization (``voltage_kV``,
``semiangle_mrad``, ``scan_sampling_A``, ``pixel_size_nm``).

``io.upload`` builds one per dataset and refuses to upload if the required basics
are missing (:func:`validate_meta`).
"""
from __future__ import annotations

import re
from pathlib import Path

REQUIRED = {
    "4dstem": ["modality", "scan_shape", "det_shape", "dtype",
               "voltage_kV", "semiangle_mrad", "source", "sample", "date"],
    "haadf":  ["modality", "shape", "dtype", "voltage_kV", "source", "sample", "date"],
}
OPTIONAL = {
    "4dstem": ["scan_sampling_A", "magnification_MX", "facility"],
    "haadf":  ["pixel_size_nm", "facility"],
}


def validate_meta(meta: dict) -> dict:
    """Refuse (raise ValueError) if the required basics are missing/wrong-typed."""
    mod = meta.get("modality")
    if mod not in REQUIRED:
        raise ValueError(f"modality must be '4dstem' or 'haadf', got {mod!r}")
    missing = [k for k in REQUIRED[mod] if meta.get(k) in (None, "", [])]
    if missing:
        raise ValueError(f"{mod} metadata missing required basics: {missing}")
    if mod == "4dstem":
        for key in ("scan_shape", "det_shape"):
            value = meta[key]
            if not (isinstance(value, (list, tuple)) and len(value) == 2
                    and all(isinstance(i, int) for i in value)):
                raise ValueError(f"{key} must be [int, int], got {value!r}")
        if meta["semiangle_mrad"] <= 0:
            raise ValueError("semiangle_mrad must be > 0")
    else:
        value = meta["shape"]
        if not (isinstance(value, (list, tuple)) and len(value) == 2
                and all(isinstance(i, int) for i in value)):
            raise ValueError(f"shape must be [int, int], got {value!r}")
    if meta["voltage_kV"] <= 0:
        raise ValueError("voltage_kV must be > 0")
    return meta


def _clean(meta: dict, mod: str) -> dict:
    """Keep only schema keys, drop None/empty (present = known)."""
    keys = REQUIRED[mod] + OPTIONAL[mod]
    return {k: meta[k] for k in keys if meta.get(k) not in (None, "", [])}


def build_4dstem_meta(master_path, **fields) -> dict:
    """Assemble + validate a 4dstem sidecar from the Arina master + explicit fields.

    Auto-fills ``scan_shape``/``det_shape`` from the master header and
    ``semiangle_mrad`` from a ``<N>mrad`` filename token; ``fields`` supplies the
    rest (``voltage_kV``, ``sample``, ``source``, ``date``, and any known optionals).
    """
    from quantem.gpu import io  # noqa: PLC0415
    header = io.get_metadata(str(master_path))
    meta = {
        "modality": "4dstem",
        "scan_shape": [int(s) for s in header.get("scan_shape", ())],
        "det_shape": [int(d) for d in header.get("detector_shape", ())],
        "dtype": "uint16",
    }
    token = re.search(r"(\d+)\s*mrad", Path(master_path).name)
    if token:
        meta["semiangle_mrad"] = float(token.group(1))
    meta.update(fields)
    return validate_meta(_clean(meta, "4dstem"))


def build_haadf_meta(*, shape, dtype, pixel_size_nm=None, **fields) -> dict:
    """Assemble + validate a haadf sidecar (shape/dtype/pixel size + explicit fields)."""
    meta = {"modality": "haadf", "shape": [int(s) for s in shape], "dtype": str(dtype)}
    if pixel_size_nm is not None:
        meta["pixel_size_nm"] = float(pixel_size_nm)
    meta.update(fields)
    return validate_meta(_clean(meta, "haadf"))
