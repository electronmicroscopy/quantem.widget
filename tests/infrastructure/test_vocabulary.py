"""Public names follow the maintainer's naming vocabulary: one name per concept, no retired spelling.

Every public signature (the exported widgets and functions, their parameters and public methods), every synced
widget trait (the same name is the JavaScript model key), every key a widget's ``state_dict`` saves and every
command-line flag is checked against the retired names below. A hit fails unless ``EXCEPTIONS`` lists it with a
one-line reason.
"""

import inspect
import re
from pathlib import Path

import pytest

import quantem.widget as qw

SOURCE = Path(__file__).resolve().parents[2] / "src" / "quantem" / "widget"

# The retired spelling of each concept; the comment names the one name used instead.
BANNED = {
    "energy_kV", "energy_keV", "kv", "ht",                      # voltage_kV
    "alpha_mrad", "convergence_angle_mrad",                     # semiangle_mrad
    "aberration_dict",                                          # aberrations
    "tilt_scan_mrad",                                           # sample_tilt_mrad
    "scan_size", "nscan", "step_A",                             # scan_shape, scan_sampling_A
    "detector_size", "det_size",                                # detector_shape
    "mrad_per_px",                                              # detector_sampling_mrad
    "binned", "binning",                                        # detector_bin
    "beam_center_px", "disk_radius_px",                         # bright_field_center_px, bright_field_radius_px
    "remove_hot_pixels", "fix_hot_pixels",                      # hot_pixel_correction
    "mean_dp", "mean_diffraction_pattern", "dp_mean",           # mean_pattern
    "virtual_bf", "vbf",                                        # bf
    "reconstruct", "fit_aberrations", "find_aberrations",       # run(), aberrations()
    "phase_estimator",                                          # method
    "upsampling", "upsampling_factor", "upsample_factor",       # upsample
    "phase_damping", "phase_regularization",                    # phase_damp
    "num_trials", "n_trials",                                   # trials
    "random_state",                                             # seed
    "SSBResult",                                                # DirectPtychoResult
    "num_slices", "n_modes", "learning_rate",                   # slices, modes, lr
    "gpu", "gpus", "gpu_id", "backend",                         # device, devices
    "save_to", "output_dir", "out_dir",                         # save_dir
    "fname", "filename",                                        # path
    "quiet", "print_progress",                                  # verbose
    "ui_mode", "show_static",                                   # show(mode=...), mode=
    "summary_df",                                               # report()
    "pixel_size", "px_size",                                    # pixel_size_A
    "niter", "n_iter", "num_iterations",                        # iterations
    "ncols", "cols", "max_cols", "compare_cols",                # columns
    "figsize",                                                  # size
    "settle_sec", "dwell_us",                                   # settle_s, dwell_s
}

# name -> one-line reason it stays. Keep this list short; a new entry needs the same scrutiny as a rename.
EXCEPTIONS = {
    "pixel_size": "a display calibration whose unit is the separate pixel_unit trait (nm, A, mrad, 1/A), so no suffix names it",
    "binned": "ScanControl's Arina 2x2 setting, read by arina-stream-gpu's scan runner; renamed with the runner, which needs the Arina PC",
    "backend": "quantem-widget show4dstem --backend picks the browser viewer (auto or webgpu), not where data is computed",
}


# The method is direct ptychography (maintainer, 11 Nov 2026): no public name spells SSB, in any case.
RETIRED_FRAGMENTS = ("ssb",)


def _banned(name: str) -> bool:
    return name not in EXCEPTIONS and (name in BANNED or any(part in name.lower() for part in RETIRED_FRAGMENTS))


def _signature_names(function, owner):
    """(name, where) for a function's own name and each named parameter."""
    found = [(function.__name__, owner)]
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return found
    return found + [(name, f"{owner}({name}=)") for name, parameter in parameters.items()
                    if name not in ("self", "cls")
                    and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]


def _public_names(obj, owner):
    """(name, where) for an exported class or function: its name, parameters, public methods and properties."""
    if not inspect.isclass(obj):
        return _signature_names(obj, owner)
    found = [(obj.__name__, owner)]
    for name, member in vars(obj).items():
        if name.startswith("_") and name != "__init__":
            continue
        if isinstance(member, property):
            found.append((name, f"{owner}.{name}"))
            continue
        function = member.__func__ if isinstance(member, (classmethod, staticmethod)) else member
        if inspect.isfunction(function):
            found += [(item, place) for item, place in _signature_names(function, f"{owner}.{name}")
                      if not (name == "__init__" and item == "__init__")]
    return found


def _exports():
    return {name: getattr(qw, name) for name in qw.__all__}


def _widgets():
    return {name: obj for name, obj in _exports().items() if inspect.isclass(obj) and hasattr(obj, "class_traits")}


def test_public_signatures_use_the_vocabulary():
    hits = set()
    for name, obj in _exports().items():
        if inspect.isclass(obj):
            for cls in obj.__mro__:
                if cls.__module__.startswith("quantem.widget"):
                    hits.update(f"{where}: {item}" for item, where in _public_names(cls, f"{name}<{cls.__name__}>")
                                if _banned(item))
        elif callable(obj):
            hits.update(f"{where}: {item}" for item, where in _public_names(obj, name) if _banned(item))
    assert not hits, "\n".join(sorted(hits))


@pytest.mark.parametrize("name", sorted(_widgets()))
def test_synced_traits_use_the_vocabulary(name):
    """A synced trait's name is also the JavaScript model key and the key of the notebook's saved widget state."""
    traits = _widgets()[name].class_traits(sync=True)
    hits = sorted(trait for trait in traits if _banned(trait.lstrip("_")))
    assert not hits, hits


@pytest.mark.parametrize("name", sorted(_widgets()))
def test_saved_state_keys_use_the_vocabulary(name):
    """``state_dict`` writes these keys into every ``save()`` file."""
    widget = _widgets()[name]
    keys = set(getattr(widget, "_STATE_KEYS", ())) | set(getattr(widget, "_LOADABLE_KEYS", ()))
    hits = sorted(key for key in keys if _banned(key))
    assert not hits, hits


def test_command_line_flags_use_the_vocabulary():
    flags = set()
    for path in SOURCE.rglob("*.py"):
        flags.update(re.findall(r"add_argument\(\s*[\"']--([a-z0-9-]+)[\"']", path.read_text(encoding="utf-8")))
    hits = sorted(flag for flag in flags
                  if _banned(flag.replace("-", "_")) or flag in {"out", "output-dir", "timeout", "interval"})
    assert not hits, hits
