"""Reference-line matching for diffraction patterns."""

from collections.abc import Sequence


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
