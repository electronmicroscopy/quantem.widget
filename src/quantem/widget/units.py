"""The one unit table of the widgets: what the unit string of a calibrated axis means.

A ``.qem`` file writes three unit strings (``angstrom`` on the scan axes,
``mrad`` or ``1/angstrom`` on the detector axes) and quantem core datasets
carry the text their author typed (``"A"``, ``"nm"``, ``"pixels"``). Every
widget reads an axis through ``axis_step``: a unit in ``UNIT_TABLE`` gives its
kind and the step in that kind's reference unit, a unit in
``UNCALIBRATED_UNITS`` (or a missing step or unit) gives None, and any other
string raises, so an axis is never drawn in a unit the widget only guessed.
``js/units.ts`` holds the same table for the frontends; keep the two in step.
"""

import math

import numpy as np

# unit string -> (kind, size of one unit in the kind's reference unit, symbol a label prints).
# Reference units: angstrom (length), 1/angstrom (reciprocal_length), mrad (angle), second (time).
UNIT_TABLE: dict[str, tuple[str, float, str]] = {
    "angstrom": ("length", 1.0, "Å"), "Å": ("length", 1.0, "Å"), "A": ("length", 1.0, "Å"),
    "nm": ("length", 10.0, "nm"), "pm": ("length", 0.01, "pm"),
    "µm": ("length", 1e4, "µm"), "μm": ("length", 1e4, "µm"), "um": ("length", 1e4, "µm"), "mm": ("length", 1e7, "mm"), "m": ("length", 1e10, "m"),
    "1/angstrom": ("reciprocal_length", 1.0, "Å⁻¹"), "1/Å": ("reciprocal_length", 1.0, "Å⁻¹"),
    "1/A": ("reciprocal_length", 1.0, "Å⁻¹"), "A^-1": ("reciprocal_length", 1.0, "Å⁻¹"), "Å^-1": ("reciprocal_length", 1.0, "Å⁻¹"),
    "1/nm": ("reciprocal_length", 0.1, "nm⁻¹"), "nm^-1": ("reciprocal_length", 0.1, "nm⁻¹"),
    # a Gatan file's unit tag reaches a dataset as written (read_image through ncempy): a diffraction pattern may be in 1/µm
    "1/µm": ("reciprocal_length", 1e-4, "µm⁻¹"), "1/μm": ("reciprocal_length", 1e-4, "µm⁻¹"), "1/um": ("reciprocal_length", 1e-4, "µm⁻¹"),
    "mrad": ("angle", 1.0, "mrad"), "rad": ("angle", 1e3, "rad"), "µrad": ("angle", 1e-3, "µrad"),
    "urad": ("angle", 1e-3, "µrad"), "deg": ("angle", 1e3 * math.pi / 180.0, "°"),
    "s": ("time", 1.0, "s"), "ms": ("time", 1e-3, "ms"), "µs": ("time", 1e-6, "µs"), "us": ("time", 1e-6, "µs"),
    "ns": ("time", 1e-9, "ns"), "ps": ("time", 1e-12, "ps"), "fs": ("time", 1e-15, "fs"),
}
# What an axis without a calibration is called: quantem core's default ("pixels", "index"), the
# first axis of a stack ("frame", "slice") and the widgets' own label ("px").
UNCALIBRATED_UNITS = frozenset({"", "pixels", "pixel", "px", "index", "frame", "slice"})
# Length units a label may switch to, largest first, each as its size in angstrom: a value
# below 1 in one unit (0.5 nm) reads as a whole number in a smaller one (5 Å).
LENGTH_LADDER: tuple[tuple[str, float], ...] = (("mm", 1e7), ("µm", 1e4), ("nm", 10.0), ("Å", 1.0), ("pm", 0.01))


def per_axis(value, count: int) -> list:
    """``value`` once per axis: one scalar repeated ``count`` times, or the entries of a sequence.

    A scalar is anything with no axis of its own (``np.ndim(value) == 0``): a
    Python or NumPy number, a 0-d array, a string, None. A tuple, a list and a
    1-d array give their entries, and a sequence of one entry is that entry
    for every axis (``units=["nm"]``). Every ``sampling=`` and ``units=``
    keyword is read here. Tested with ``isinstance(value, (int, float))``, a
    step that came out of NumPy (``np.float32(0.5)``, ``array.mean()``) fell
    to the sequence branch and raised on indexing.
    """
    entries = [value] if np.ndim(value) == 0 else list(value)
    return entries * count if len(entries) == 1 else entries


def wrap_degrees(value) -> float:
    """``value`` degrees as the same direction in [0, 360), the range every file holds an angle in.

    ``value % 360.0`` alone answers 360.0 for a tiny negative angle (-1e-15),
    outside the range. An angle already in the range is returned bit for bit.
    ``wrapDegrees`` of ``js/units.ts`` gives the same number for the page.
    """
    wrapped = float(value) % 360.0
    return 0.0 if wrapped == 360.0 else wrapped


def unit_entry(unit) -> tuple[str, float, str]:
    """``(kind, size in the reference unit, symbol)`` of a unit string; an uncalibrated one is ``("pixels", 1.0, "px")``.

    A string in neither table raises ValueError. Without this check a dataset
    in a unit the widgets do not know would open with no scale bar and no
    message, or with a bar in the wrong unit.
    """
    if unit is None or str(unit) in UNCALIBRATED_UNITS:
        return ("pixels", 1.0, "px")
    if str(unit) not in UNIT_TABLE:
        raise ValueError(f"Unknown unit {str(unit)!r}. A calibrated axis uses one of: {', '.join(sorted(UNIT_TABLE))}; "
                         f"an uncalibrated axis uses 'pixels'. A dataset in {str(unit)!r} opens uncalibrated when its "
                         "array is passed (dataset.array); sampling= and units= then state the step in a unit above.")
    return UNIT_TABLE[str(unit)]


def axis_step(sampling, unit) -> tuple[str, float] | None:
    """Kind and step of one axis in the kind's reference unit, or None for an uncalibrated axis.

    ``sampling`` is the step in ``unit``. A missing step or unit, an
    uncalibrated unit or a step that is not positive means the axis carries no
    calibration: unknown is absent, never ``1.0`` or ``"px"``.
    """
    kind, size, _ = unit_entry(unit)
    if sampling is None or kind == "pixels" or float(sampling) <= 0:
        return None
    return kind, float(sampling) * size


def axis_pair(sampling, units, kinds: tuple[str, ...], axes: str) -> tuple[str, list] | None:
    """Kind and ``[row, col]`` steps of two axes in the reference unit; an uncalibrated axis is None, the pair is None when both are.

    ``kinds`` are the kinds the two axes may have (a scan is a ``length``, a
    detector an ``angle`` or a ``reciprocal_length``) and ``axes`` names them
    in the error. Both axes must share one kind; a scan in mrad or a detector
    calibrated in two kinds has no geometry a widget can draw, so it raises.

    Each axis is read on its own, so ``stated`` can hold a keyword against the
    one axis an input records. A pair with one calibrated axis used to count
    as recording nothing, and a keyword then replaced the recorded step
    without a comparison. A widget draws a pair only when neither step is None.
    """
    steps = [axis_step(step, unit) for step, unit in zip(sampling, units, strict=True)]
    found_kinds = {step[0] for step in steps if step is not None}
    if not found_kinds:
        return None
    if len(found_kinds) > 1 or not found_kinds <= set(kinds):
        raise ValueError(f"The {axes} axes are in {units[0]!r} and {units[1]!r}; both must be one "
                         f"{' or '.join(kinds)}.")
    return found_kinds.pop(), [None if step is None else step[1] for step in steps]


def acquisition_calibration(sampling, units, metadata) -> dict:
    """The calibration a 4D acquisition records, under the ``.qem`` format's flat names; an unknown value is None.

    ``sampling`` and ``units`` are the four axis values of a
    ``quantem.gpu.io.Dataset4dstemGPU`` (an unknown axis is None) or of a
    quantem core ``Dataset4dstem`` (an unknown axis is ``1.0 "pixels"``), and
    ``metadata`` its metadata dict. The scan step is ``scan_sampling_A`` and
    the detector step ``detector_sampling_mrad`` or ``detector_sampling_inv_A``,
    each ``[row, col]`` with None for an axis the input leaves open;
    ``voltage_kV`` and ``semiangle_mrad`` are the flat
    metadata keys. Every 4D widget reads a dataset here, so one file opens
    with one calibration whichever dataset type this machine read it into.
    """
    scan = axis_pair(sampling[:2], units[:2], ("length",), "scan")
    detector = axis_pair(sampling[2:4], units[2:4], ("angle", "reciprocal_length"), "detector")
    return {
        "scan_sampling_A": None if scan is None else scan[1],
        "detector_sampling_mrad": detector[1] if detector is not None and detector[0] == "angle" else None,
        "detector_sampling_inv_A": detector[1] if detector is not None and detector[0] == "reciprocal_length" else None,
        "voltage_kV": metadata.get("voltage_kV"),
        "semiangle_mrad": metadata.get("semiangle_mrad"),
    }


def image_axes(sampling, units, dataset=None) -> tuple:
    """``(sampling, units)`` of an image widget opened on ``dataset``: the dataset's, which the keywords may repeat.

    ``sampling`` and ``units`` are the constructor keywords, each None when
    omitted and then taken from the last two axes of the dataset. When the
    dataset records a calibrated column step and the keywords give another one,
    ValueError says so: the dataset is the source of its calibration, and
    before this check the keyword replaced it without a message.

    A ``sampling`` stated with no ``units`` is in angstrom, for an array
    (``dataset`` None) as for a dataset. Show3D, Show3DSlices and ShowPtycho
    read that call so; Show2D and Show1D read the missing unit as "px" and
    dropped the stated step without a message.

    ``units`` alone on a calibrated dataset asks for the dataset's step in
    that unit: the step is converted through the unit table when both units
    are of one kind (``step * size of the dataset's unit / size of units``),
    and a unit of another kind raises ValueError naming both. Without the
    conversion the call raised "contradict", and before that check Show2D
    relabelled the step unconverted.

    A calibrated unit stated for an input that has no step, with no
    ``sampling``, raises ValueError: a unit needs a step. Taken as it was, the
    unit vanished for an array (no bar, no message) and turned the ``1.0`` of
    an uncalibrated dataset into a step of 1 nm.
    """
    if sampling is not None and units is None:
        units = "A"
    held_sampling = None if dataset is None else tuple(float(step) for step in dataset.sampling[-2:])
    held_units = None if dataset is None else [str(unit) for unit in dataset.units[-2:]]
    recorded = None if dataset is None else axis_step(held_sampling[-1], held_units[-1])
    if sampling is None and units is not None and recorded is None and unit_entry(per_axis(units, 2)[-1])[0] != "pixels":
        raise ValueError(f"units={units!r} names a unit and the input has no step to give it: a unit needs a step. "
                         f"Pass sampling= (the step per pixel in {per_axis(units, 2)[-1]}), or leave units out for an uncalibrated axis.")
    if dataset is None:
        return sampling, units
    if sampling is None and units is not None and recorded is not None:
        wanted = [str(unit) for unit in per_axis(units, 2)[-2:]]
        if unit_entry(wanted[-1])[0] != recorded[0]:
            raise ValueError(f"units={units!r} cannot show a dataset calibrated in {held_units[-1]!r} ({held_sampling[-1]:g} {held_units[-1]} per pixel): "
                             f"{held_units[-1]!r} is a {recorded[0].replace('_', ' ')} and {wanted[-1]!r} is not. "
                             f"Pass a {recorded[0].replace('_', ' ')} unit to convert the step, or leave units out.")
        # The ratio of the two sizes is 1.0 for one unit under two spellings, so that step is returned bit for bit.
        # An axis recorded in another kind or in free text (the time axis of a kymograph) keeps its step.
        entries = [(UNIT_TABLE.get(held), unit_entry(unit)) for held, unit in zip(held_units, wanted)]
        return tuple(step * (held[1] / shown[1]) if held is not None and held[0] == shown[0] else step
                     for step, (held, shown) in zip(held_sampling, entries)), units
    sampling = held_sampling if sampling is None else sampling
    units = held_units if units is None else units
    given = axis_step(per_axis(sampling, 2)[-1], per_axis(units, 2)[-1])
    if recorded is not None and (given is None or given[0] != recorded[0] or not math.isclose(given[1], recorded[1], rel_tol=1e-6)):
        raise ValueError(f"sampling={sampling!r}, units={units!r} contradict the calibration the dataset records "
                         f"({held_sampling[-1]:g} {held_units[-1]} per pixel). Leave them out: the dataset is the source of its calibration.")
    return sampling, units


def stated(name: str, keyword, recorded):
    """The value of one calibration: ``recorded`` by the input, else the ``keyword`` the caller passed.

    A keyword equal to the recorded value is accepted. One that differs raises
    ValueError naming both: the dataset or file is the source of its own
    calibration, and a second value would leave two answers for one quantity.

    A calibration of several axes is read per axis: None inside ``recorded``
    or ``keyword`` is an axis without a value there and takes the other's, and
    only an axis both give is compared. Compared as whole vectors, a depth
    step was refused for a stack whose slice axis records none.
    """
    if keyword is None or recorded is None:
        return recorded if keyword is None else keyword
    held = per_axis(recorded, 1)
    given = per_axis(keyword, len(held))
    if len(given) != len(held) or any(None not in (passed, kept) and not math.isclose(passed, kept, rel_tol=1e-6) for passed, kept in zip(given, held)):
        raise ValueError(f"{name}={keyword!r} contradicts the value the input records ({recorded!r}). "
                         f"Leave {name} out: the dataset or file is the source of its calibration.")
    return [passed if kept is None else kept for passed, kept in zip(given, held)] if None in held else recorded
