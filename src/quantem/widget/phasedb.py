"""Reference-line matching for diffraction patterns."""

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

# out-of-tolerance pad cost
_NO_MATCH_COST = 1.0e6


def match_candidate(observed_d: Sequence[float], lines: Sequence[dict], tol: float = 0.03) -> dict:
    """Match measured d-spacings against one reference phase."""
    observed = [float(spacing) for spacing in observed_d if spacing and float(spacing) > 0]
    references = [
        (float(line["d"]), float(line.get("i_rel", line.get("intensity")) or 0.0))
        for line in lines
        if float(line["d"]) > 0
    ]
    has_intensity = any(rel_intensity > 0 for _, rel_intensity in references)
    n_observed = len(observed)
    if n_observed == 0 or not references:
        return {
            "matched": 0,
            "n_obs": n_observed,
            "mean_err": None,
            "n_missing_strong": 0 if has_intensity else None,
            "assignments": [],
        }
    observed_g = [1.0 / spacing for spacing in observed]
    reference_g = [1.0 / spacing for spacing, _ in references]

    # in-tolerance pair costs
    cost = np.full((n_observed, len(references)), _NO_MATCH_COST)
    d_errors = np.zeros_like(cost)
    for obs_index, observed_value in enumerate(observed_g):
        for ref_index, reference_value in enumerate(reference_g):
            error = abs(reference_value - observed_value) / observed_value
            if error <= tol:
                cost[obs_index, ref_index] = error
                d_errors[obs_index, ref_index] = (
                    abs(1.0 / observed_value - references[ref_index][0]) / references[ref_index][0]
                )

    # maximum matches first, then lowest total error
    assignments = [(obs_index, None) for obs_index in range(n_observed)]
    errors = []
    matched_refs = set()
    for obs_index, ref_index in zip(*linear_sum_assignment(cost)):
        if cost[obs_index, ref_index] >= _NO_MATCH_COST:
            continue
        assignments[obs_index] = (int(obs_index), int(ref_index))
        errors.append(float(d_errors[obs_index, ref_index]))
        matched_refs.add(int(ref_index))

    n_matched = len(errors)
    n_missing_strong = None
    if has_intensity:
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
        "n_missing_strong": n_missing_strong,
        "assignments": assignments,
    }


def match_sort_key(report: dict) -> tuple:
    """Sort phase reports from strongest to weakest match."""
    return (
        -report["matched"],
        report["n_missing_strong"] or 0,
        report["mean_err"] if report["mean_err"] is not None else 1.0,
    )
