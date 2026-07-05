import pytest

from quantem.widget.phasedb import match_candidate, match_sort_key

# --- Matcher ---


def _lines(entries):
    return [{"d": d, "i_rel": i} for d, i in entries]


def test_match_candidate_perfect_match():
    refs = _lines([(3.0, 100), (2.0, 80), (1.5, 60), (1.2, 40)])
    sc = match_candidate([3.0, 2.0, 1.5, 1.2], refs, tol=0.03)
    assert sc["matched"] == 4 and sc["n_obs"] == 4
    assert sc["mean_err"] == pytest.approx(0.0, abs=1e-9)
    assert sc["n_missing_strong"] == 0
    assert all(j is not None for _, j in sc["assignments"])


def test_match_candidate_uses_absolute_calibration():
    refs = _lines([(3.0, 100), (2.0, 80), (1.5, 60), (1.2, 40)])
    perfect = match_candidate([3.0, 2.0, 1.5, 1.2], refs, tol=0.03)
    off = match_candidate([3.0 * 1.05, 2.0 * 1.05, 1.5 * 1.05, 1.2 * 1.05], refs, tol=0.03)
    assert off["matched"] == 0
    assert match_sort_key(perfect) < match_sort_key(off)


def test_match_candidate_uses_each_reference_once():
    sc = match_candidate([2.0, 2.01], _lines([(2.0, 100)]), tol=0.03)
    assert sc["matched"] == 1
    assert [ref for _, ref in sc["assignments"]].count(0) == 1


def test_rank_unexplained_ring_below_full_explanation():
    # same observations, two candidate phases: one explains every ring,
    # the other leaves the 1.05 ring unexplained
    obs = [3.0, 2.0, 1.5, 1.05]
    explains_all = match_candidate(obs, _lines([(3.0, 100), (2.0, 80), (1.5, 60), (1.05, 40)]))
    leaves_one = match_candidate(obs, _lines([(3.0, 100), (2.0, 80), (1.5, 60), (1.2, 40)]))
    assert explains_all["matched"] == 4 and leaves_one["matched"] == 3
    assert match_sort_key(explains_all) < match_sort_key(leaves_one)


def test_rank_missing_strong_below_missing_weak():
    strong = _lines([(3.0, 100), (2.0, 90), (1.5, 80)])
    weak = _lines([(3.0, 100), (2.0, 10), (1.5, 80)])
    obs = [3.0, 1.5]  # the d=2.0 line is inside the observed range but unmatched
    sc_strong = match_candidate(obs, strong, tol=0.03)
    sc_weak = match_candidate(obs, weak, tol=0.03)
    assert sc_strong["n_missing_strong"] == 1 and sc_weak["n_missing_strong"] == 0
    assert match_sort_key(sc_weak) < match_sort_key(sc_strong)


def test_match_candidate_accepts_intensity_key():
    # reference cards may carry "intensity" instead of "i_rel"
    refs = [
        {"d": 3.0, "intensity": 100.0},
        {"d": 2.0, "intensity": 90.0},
        {"d": 1.5, "intensity": 80.0},
    ]
    sc = match_candidate([3.0, 1.5], refs, tol=0.03)
    assert sc["matched"] == 2
    assert sc["n_missing_strong"] == 1  # the strong d=2.0 line is unexplained


def test_match_candidate_keeps_plain_match_report():
    dense = _lines([(1.0 + 0.01 * i, 20) for i in range(200)])  # 200 lines, d 1.0-2.99
    sc = match_candidate([2.5, 1.7], dense, tol=0.03)
    assert sc["matched"] == 2
    assert set(sc) == {
        "matched",
        "n_obs",
        "mean_err",
        "n_missing_strong",
        "assignments",
    }


def test_match_candidate_no_reference_lines():
    sc = match_candidate([2.5, 1.7], [], tol=0.03)
    assert sc["matched"] == 0
    assert sc["assignments"] == []
