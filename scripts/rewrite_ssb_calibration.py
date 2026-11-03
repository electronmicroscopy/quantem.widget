"""Rewrite SSB calibration files of an earlier quantem.widget release into the current layout, once.

quantem.widget reads and writes one layout per file kind and converts nothing
inside the package. A calibration a person chose by eye cannot be recomputed,
so this script carries the earlier file kinds over:

- ``calibration.json`` from ShowPtycho (``schema_version`` 3, ``aberration_unit`` "nm",
  or ``quantem.ssb-calibration/4`` or ``/5``)
- ``showptycho_stars.json`` from ShowPtycho (a list of stars, or ``quantem.ssb-stars/1`` or ``/2``)
- ``snapshots/snapshots.json`` and ``snapshots/cal.json`` of an exported ShowPtycho folder
  (``quantem.ssb-snapshots/1``, ``quantem.ssb-browser-calibration/2``)
- ``*_ssb_calibration.json`` from Show4DSTEM (``schema`` "quantem.ssb.calibration.v1", or
  ``quantem.show4dstem-ssb-calibration/1``, whose ``refine`` flag is now ``refinement``:
  ``"nelder-mead"`` or None, or ``/2``). Layouts up to /2 recorded the standard fit at tilt 0,
  the only one Compute SSB ran then; /3 records the sample model of the phase (``tilt``,
  ``sample_tilt_mrad``, ``depth_spread_nm``, ``tilt_fit_gain``), so an earlier file gets
  ``tilt`` false and the other three None. Compute SSB now fits tilt-corrected by default.

The calibration /4, stars /1, snapshots /1 and cal.json /2 layouts named the display-only phase sign ``flip_phase`` (now
``display_negate_phase``) and, in regional fits, the scan-frame tilt
``tilt_mrad`` (now ``sample_tilt_mrad``), the joint-fit score ``agreement``
(now ``sample_agreement``) and the trial count ``completed_trials`` (now
``num_completed_trials``). No number changes.

Calibration /5 and stars /2 named the acquisition by its absolute ``source_path`` and
identified it by path, inode and modification times. Calibration /6 and stars /3 name it
``source`` (its file name, and its path relative to the record's folder when the data lives
under it) and identify it by quantem.gpu's portable ``source_identity``: file names, sizes,
SHA-256 of each file's first MiB, scan and detector shapes and the flagged-pixel fill. That
identity is computed from the data, so a /5 calibration with an identity needs its acquisition:
beside the calibration (the file name it recorded), or given with ``--data``. The flagged-pixel
fill is recorded as quantem.gpu fills it now (half to even; a cluster from the ring around it).

    python scripts/rewrite_ssb_calibration.py --dry-run path/to/calibration.json
    python scripts/rewrite_ssb_calibration.py path/to/calibration.json path/to/showptycho_stars.json
    python scripts/rewrite_ssb_calibration.py --data path/to/scan_master.h5 path/to/calibration.json

``--dry-run`` prints what would be written and changes nothing. Without it each
file is replaced and the earlier one is kept beside it as ``<name>.before``.
A file already in the current layout is left alone. A ShowPtycho file without
``"aberration_unit": "nm"`` is refused: before that marker quantem.gpu reported
Angstrom under the nm label, so the unit of its numbers is not known.
"""

import argparse
import json
import math
import pathlib

CALIBRATION_SCHEMA = "quantem.ssb-calibration/6"
STARS_SCHEMA = "quantem.ssb-stars/3"
# the layouts with an absolute source_path, carried to the current ones by portable()
PATH_CALIBRATION_SCHEMA = "quantem.ssb-calibration/5"
PATH_STARS_SCHEMA = "quantem.ssb-stars/2"
SNAPSHOTS_SCHEMA = "quantem.ssb-snapshots/2"
BROWSER_CALIBRATION_SCHEMA = "quantem.ssb-browser-calibration/3"
SHOW4DSTEM_SCHEMA = "quantem.show4dstem-ssb-calibration/3"
# Show4DSTEM layouts up to /2 fitted at tilt 0 and recorded no sample model
SHOW4DSTEM_TILT_ZERO = {"tilt": False, "sample_tilt_mrad": None, "depth_spread_nm": None, "tilt_fit_gain": None}
RENAMED = {"flip_phase": "display_negate_phase", "tilt_mrad": "sample_tilt_mrad",
           "agreement": "sample_agreement", "completed_trials": "num_completed_trials"}


def renamed(record):
    """``record`` with the /4-era keys under their current names, at every depth (regional fits nest them)."""
    if isinstance(record, list):
        return [renamed(item) for item in record]
    if isinstance(record, dict):
        return {RENAMED.get(key, key): renamed(value) for key, value in record.items()}
    return record


def aberrations(earlier: dict, *, higher_order_radians: bool) -> dict:
    """``{C10, C12 (nm), phi12 (rad), ...}`` as ``{C10_nm, C12_nm, phi12_deg, C21_nm, phi21_deg, ...}``.

    ``calibration.json`` held higher orders as ``C21_mag`` (nm) and
    ``C21_angle`` (degrees), the stars file as ``C21`` (nm) and ``phi21``
    (radians); ``higher_order_radians`` says which file this is.
    """
    current = {"C10_nm": float(earlier.get("C10", 0.0)), "C12_nm": float(earlier.get("C12", 0.0)),
               "phi12_deg": math.degrees(float(earlier.get("phi12", 0.0)))}
    for key, value in earlier.items():
        if key in ("C10", "C12", "phi12"):
            continue
        if key.endswith("_mag"):
            current[f"{key[:-4]}_nm"] = float(value)
        elif key.endswith("_angle"):
            current[f"phi{key[1:-6]}_deg"] = float(value)
        elif key.startswith("phi"):
            current[f"{key}_deg"] = math.degrees(float(value)) if higher_order_radians else float(value)
        else:
            current[f"{key}_nm"] = float(value)
    return current


def showptycho_record(earlier: dict, *, higher_order_radians: bool) -> tuple[dict, list[str]]:
    """One ShowPtycho calibration or star in the current keys, with the notes a person should read."""
    if earlier.get("aberration_unit") != "nm":
        raise ValueError('no "aberration_unit": "nm": the unit of C10 and C12 in this file is not known')
    notes = []
    step = earlier.get("scan_sampling_A")
    if isinstance(step, (int, float)):
        notes.append(f"scan_sampling_A held one number, the row step {step}; written as [{step}, {step}]: correct the column step if it differs")
        step = [float(step), float(step)]
    if earlier.get("timestamp"):
        notes.append(f"timestamp {earlier['timestamp']!r} is a local time without a zone and is not carried into generated_at_utc")
    current = {
        "id": earlier.get("id"), "generated_at_utc": None,
        "aberrations": aberrations(earlier["aberrations"], higher_order_radians=higher_order_radians),
        "scan_detector_rotation_deg": float(earlier["rotation_angle_deg"]) % 360.0, "scan_detector_rotation_flipped": False,
        "display_negate_phase": bool(earlier.get("flip_phase", False)), "voltage_kV": earlier.get("voltage_kV"),
        "semiangle_mrad": earlier.get("semiangle_mrad"), "scan_sampling_A": step, "detector_sampling_mrad": None,
        "loss": earlier.get("loss"), "source_path": earlier.get("source_file"),
    }
    if "scan_region" in earlier:
        current["scan_region"] = list(earlier["scan_region"])
    return current, notes


def rewrite(earlier, path: pathlib.Path | None = None, data: pathlib.Path | None = None) -> tuple[dict | None, list[str]]:
    """The current layout of one earlier file's content, or None when it is current already.

    ``path`` is where the file is (the folder a relative data path is measured from), ``data``
    the acquisition when it is not beside the file under the name the file recorded.
    """
    current, notes = path_layout(earlier)
    if current is not None and current.get("schema") in (PATH_CALIBRATION_SCHEMA, PATH_STARS_SCHEMA):
        current = portable(current, pathlib.Path.cwd() / "calibration.json" if path is None else path, data)
    return current, notes


def portable(record: dict, path: pathlib.Path, data: pathlib.Path | None) -> dict:
    """Calibration /5 or stars /2 as /6 or /3: ``source`` for ``source_path``, a portable session identity."""
    from quantem.gpu.ssb.persistence import portable_source

    folder = path.resolve().parent

    def located(recorded: str | None) -> pathlib.Path | None:
        if data is not None:
            return data
        beside = None if recorded is None else folder / pathlib.Path(recorded).name
        return beside if beside is not None and beside.exists() else None

    def source(recorded: str | None) -> dict | None:
        if recorded is None:
            return None
        found = located(recorded)
        return portable_source(found, folder) if found is not None else {"file_name": pathlib.Path(recorded).name,
                                                                          "relative_path": None}

    if record["schema"] == PATH_STARS_SCHEMA:
        stars = [{**{key: value for key, value in star.items() if key != "source_path"}, "source": source(star.get("source_path"))}
                 for star in record["stars"]]
        return {**record, "schema": STARS_SCHEMA, "stars": stars}
    current = {key: value for key, value in record.items() if key != "source_path"}
    current.update(schema=CALIBRATION_SCHEMA, source=source(record.get("source_path")))
    if record.get("session_identity") is not None:
        found = located(record.get("source_path"))
        if found is None or not found.exists():
            name = pathlib.Path(record.get("source_path") or "the acquisition").name
            raise ValueError(f"its identity is computed from the data, and {name} is not beside {path.name}; "
                             f"pass --data <path to {name}>")
        current["session_identity"] = session_identity(record["session_identity"], found, record.get("scan_shape"))
    return current


def session_identity(earlier: dict, data: pathlib.Path, scan_shape) -> dict:
    """A /5 session identity (paths, inodes, times) as the portable one ShowPtycho now compares."""
    import numpy as np
    from quantem.gpu.resident.hot_pixels import median_plan
    from quantem.gpu.ssb.persistence import acquisition_identity, json_value, source_identity

    signature = earlier.get("source") or {}
    companions = {}
    for name in ("storage", "calibration", "manifest"):
        held = signature.get(name)
        companion = None if not held else pathlib.Path(held["path"])
        if companion is not None and not companion.exists():
            raise ValueError(f"its {name} file {companion.name} is gone; open the session and save the fit once")
        companions[name] = None if companion is None else str(companion)
    data_record = dict(earlier.get("data") or {})
    held = (data_record.pop("provenance", None) or {}).get("hot_pixel_correction")
    correction = None
    if held:
        window = None
        if held.get("method") == "median" and held.get("applied"):
            # the window quantem.gpu's fill now uses for these flagged pixels
            mask = np.zeros(acquisition_identity(data, scan_shape=scan_shape)["detector_shape"], np.uint8)
            for row, col in held.get("coordinates_row_col", []):
                mask[row, col] = 1
            window = median_plan(mask).largest_window_px
        correction = {"method": held.get("method"), "pixel_count": held.get("pixel_count"), "applied": held.get("applied"),
                      "largest_window_px": window}
    return json_value({
        **earlier,
        "source": source_identity(str(data), scan_shape=scan_shape, hot_pixel_correction=correction,
                                  storage_path=companions["storage"], calibration_path=companions["calibration"],
                                  manifest_path=companions["manifest"]),
        "data": data_record,
    })


def path_layout(earlier) -> tuple[dict | None, list[str]]:
    """An earlier file's content in the last layout before ``portable``, or None when it is current already."""
    if isinstance(earlier, list):
        records = [showptycho_record(star, higher_order_radians=True) for star in earlier]
        return {"schema": PATH_STARS_SCHEMA, "stars": [record for record, _ in records]}, [note for _, notes in records for note in notes]
    if earlier.get("schema") in (PATH_CALIBRATION_SCHEMA, PATH_STARS_SCHEMA):
        return earlier, []
    if earlier.get("schema") in (CALIBRATION_SCHEMA, STARS_SCHEMA, SNAPSHOTS_SCHEMA, BROWSER_CALIBRATION_SCHEMA, SHOW4DSTEM_SCHEMA):
        return None, []
    later = {"quantem.ssb-calibration/4": PATH_CALIBRATION_SCHEMA, "quantem.ssb-stars/1": PATH_STARS_SCHEMA,
             "quantem.ssb-snapshots/1": SNAPSHOTS_SCHEMA,
             "quantem.ssb-browser-calibration/2": BROWSER_CALIBRATION_SCHEMA}.get(earlier.get("schema"))
    if later is not None:
        return {**renamed(earlier), "schema": later}, []
    if earlier.get("schema") == "quantem.show4dstem-ssb-calibration/1":
        current = {key: value for key, value in earlier.items() if key != "refine"}
        return {**current, "schema": SHOW4DSTEM_SCHEMA, "refinement": "nelder-mead" if earlier.get("refine") else None,
                **SHOW4DSTEM_TILT_ZERO}, []
    if earlier.get("schema") == "quantem.show4dstem-ssb-calibration/2":
        return {**earlier, "schema": SHOW4DSTEM_SCHEMA, **SHOW4DSTEM_TILT_ZERO}, []
    if earlier.get("schema") == "quantem.ssb.calibration.v1":
        calibration, run, source = earlier["calibration"], earlier["run"], dict(earlier["source"])
        source["acquisition_index"] = source.pop("frame_idx")
        held = earlier["aberrations"]
        return {
            "schema": SHOW4DSTEM_SCHEMA, "generated_at_utc": earlier["created_utc"], "source": source,
            "aberrations": {"C10_nm": float(held["C10"]), "C12_nm": float(held["C12"]), "phi12_deg": math.degrees(float(held["phi12"]))},
            # the earlier file held the angle below 180 degrees and the reversal beside it
            "scan_detector_rotation_deg": (float(earlier["rotation_angle_deg"]) + (180.0 if earlier["com_reversed"] else 0.0)) % 360.0,
            "scan_detector_rotation_flipped": False, "voltage_kV": calibration["voltage_kV"], "semiangle_mrad": calibration["semiangle_mrad"],
            "scan_sampling_A": calibration["scan_sampling_A"], "detector_sampling_mrad": calibration["det_sampling_mrad"],
            "bright_field_radius_px": calibration["bf_radius"], "bright_field_intensity_threshold": calibration["bf_intensity_threshold"],
            "num_bright_field_pixels": calibration["bf_pixels"], "num_trials": run["n_trials"],
            "refinement": "nelder-mead" if run["refine"] else None,
            "manual_locked": run["manual_locked"], **SHOW4DSTEM_TILT_ZERO,
            "seed": run["seed"], "loss": run["loss"], "elapsed_s": run["elapsed_s"],
        }, []
    if earlier.get("schema_version") == 3:
        current, notes = showptycho_record(earlier, higher_order_radians=False)
        tilt = earlier.get("tilt_mrad")
        current.update(schema=PATH_CALIBRATION_SCHEMA, notes=earlier.get("notes"), sample_tilt_mrad=None if tilt is None else list(tilt),
                       propagator_tilt_mrad=earlier.get("tilt_object_mrad"), depth_spread_nm=earlier.get("depth_spread_nm"))
        return current, notes
    raise ValueError("not an SSB calibration file this script knows")


def main() -> None:
    """Rewrite each path given on the command line, or print the result with ``--dry-run``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true", help="print the rewritten content and change nothing")
    parser.add_argument("--data", type=pathlib.Path, help="the acquisition, when it is not beside the calibration")
    args = parser.parse_args()
    for path in args.paths:
        current, notes = rewrite(json.loads(path.read_text(encoding="utf-8")), path, args.data)
        if current is None:
            print(f"{path}: already in the current layout, left alone")
            continue
        for note in notes:
            print(f"{path}: {note}")
        text = json.dumps(current, indent=2)
        if args.dry_run:
            print(f"{path}: would be rewritten as\n{text}")
            continue
        path.with_name(path.name + ".before").write_bytes(path.read_bytes())
        path.write_text(text + "\n", encoding="utf-8")
        print(f"{path}: rewritten; the earlier file is {path.name}.before")


if __name__ == "__main__":
    main()
