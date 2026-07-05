"""Reference-line matching and local phase loading for diffraction patterns."""

import pathlib
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from .crystal import Phase, _canonical_hkl, _format_hkl

if TYPE_CHECKING:
    from pymatgen.core import Structure


def structure_reflections(
    structure: "Structure", d_min: float = 0.8, voltage: float = 200.0
) -> list[dict]:
    """Electron-reflection families for a structure, largest d first."""
    try:
        from pymatgen.analysis.diffraction.core import get_unique_families
        from pymatgen.analysis.diffraction.tem import TEMCalculator
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "structure_reflections needs pymatgen; install it with "
            "'pip install \"quantem.widget[phaseid]\"'"
        ) from exc
    calculator = TEMCalculator(voltage=float(voltage))
    reciprocal_lattice = structure.lattice.reciprocal_lattice_crystallographic
    reciprocal_points = reciprocal_lattice.get_points_in_sphere(
        [[0, 0, 0]], [0, 0, 0], 1.0 / float(d_min)
    )
    spacing_by_hkl = {}
    for hkl_float, _, _, _ in reciprocal_points:
        hkl = tuple(int(round(index)) for index in hkl_float)
        if hkl != (0, 0, 0):
            spacing_by_hkl[hkl] = float(structure.lattice.d_hkl(hkl))

    intensities_by_hkl = calculator.cell_intensity(
        structure, calculator.bragg_angles(spacing_by_hkl)
    )
    lines_by_spacing: dict[float, list] = {}
    for hkl, intensity in intensities_by_hkl.items():
        spacing_key = round(spacing_by_hkl[hkl], 4)
        lines_by_spacing.setdefault(spacing_key, []).append((hkl, float(intensity)))

    intensity_by_spacing = {
        spacing: sum(intensity for _, intensity in members)
        for spacing, members in lines_by_spacing.items()
    }
    max_intensity = max(intensity_by_spacing.values(), default=0.0)
    if max_intensity <= 0:
        return []

    reflections = []
    for spacing, members in lines_by_spacing.items():
        total_intensity = intensity_by_spacing[spacing]
        if total_intensity < 1e-3 * max_intensity:
            continue
        families = get_unique_families([hkl for hkl, _ in members])
        representative = max(_canonical_hkl(hkl) for hkl in families)
        reflections.append(
            {
                "d": float(spacing),
                "hkl": representative,
                "hkl_str": _format_hkl(representative),
                "multiplicity": int(sum(families.values())),
                "intensity": float(100.0 * total_intensity / max_intensity),
            }
        )
    return sorted(reflections, key=lambda reflection: -reflection["d"])


def match_candidate(observed_d: Sequence[float], lines: Sequence[dict], tol: float = 0.03) -> dict:
    """Match measured d-spacings against one reference phase."""
    observed = [float(spacing) for spacing in observed_d if spacing and float(spacing) > 0]
    references = [
        (float(line["d"]), float(line.get("i_rel", line.get("intensity")) or 0.0))
        for line in lines
        if float(line["d"]) > 0
    ]
    n_observed = len(observed)
    if n_observed == 0 or not references:
        return {
            "matched": 0,
            "n_obs": n_observed,
            "mean_err": None,
            "n_missing_strong": 0,
            "assignments": [],
        }
    observed_g = [1.0 / spacing for spacing in observed]
    reference_g = [1.0 / spacing for spacing, _ in references]

    candidates = []
    for obs_index, observed_value in enumerate(observed_g):
        for ref_index, reference_value in enumerate(reference_g):
            error = abs(reference_value - observed_value) / observed_value
            if error <= tol:
                d_error = (
                    abs(1.0 / observed_value - references[ref_index][0]) / references[ref_index][0]
                )
                candidates.append((error, obs_index, ref_index, d_error))

    assignments = [(obs_index, None) for obs_index in range(n_observed)]
    errors = []
    matched_observed = set()
    matched_refs = set()
    for _, obs_index, ref_index, d_error in sorted(candidates):
        if obs_index in matched_observed or ref_index in matched_refs:
            continue
        assignments[obs_index] = (obs_index, ref_index)
        errors.append(d_error)
        matched_observed.add(obs_index)
        matched_refs.add(ref_index)

    n_matched = len(errors)
    g_min, g_max = min(observed_g), max(observed_g)
    n_missing_strong = sum(
        1
        for ref_index, (_, rel_intensity) in enumerate(references)
        if rel_intensity >= 25.0
        and g_min <= reference_g[ref_index] <= g_max
        and ref_index not in matched_refs
    )

    return {
        "matched": n_matched,
        "n_obs": n_observed,
        "mean_err": (float(sum(errors) / n_matched) if n_matched else None),
        "n_missing_strong": int(n_missing_strong),
        "assignments": assignments,
    }


def match_sort_key(report: dict) -> tuple:
    """Sort phase reports from strongest to weakest match."""
    return (
        -report["matched"],
        report["n_missing_strong"],
        report["mean_err"] if report["mean_err"] is not None else 1.0,
    )


def _structure_from_cif(path: str | pathlib.Path) -> "Structure":
    try:
        from pymatgen.io.cif import CifParser
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "CIF import needs pymatgen; install it with 'pip install \"quantem.widget[phaseid]\"'"
        ) from exc
    try:
        parser = CifParser(str(path), occupancy_tolerance=2.0)
        return parser.parse_structures(primitive=False, on_error="warn")[0]
    except (ValueError, KeyError, AssertionError, IndexError) as exc:
        raise ValueError(f"could not parse CIF {pathlib.Path(path).name!r}: {exc}") from exc


def _resolve_cif_paths(
    paths: str | pathlib.Path | Iterable[str | pathlib.Path],
) -> list[pathlib.Path]:
    """CIF paths from a directory (every ``*.cif`` inside) or an explicit list."""
    if isinstance(paths, (str, pathlib.Path)):
        root = pathlib.Path(paths)
        return sorted(root.glob("*.cif")) if root.is_dir() else [root]
    return [pathlib.Path(path) for path in paths]


def phases_from_cifs(
    paths: str | pathlib.Path | Iterable[str | pathlib.Path], d_min: float = 0.7
) -> list[Phase]:
    """Load local CIF files as candidate phases for identification.

    ``paths`` is a directory (loads every ``*.cif`` inside) or an iterable of
    CIF file paths. Files that fail to parse are skipped.
    """
    phases = []
    for path in _resolve_cif_paths(paths):
        try:
            phases.append(Phase.from_structure(_structure_from_cif(path), d_min=d_min))
        except (ValueError, KeyError, OSError):
            continue
    if not phases:
        raise ValueError("no CIF files could be loaded from the given paths")
    return phases
