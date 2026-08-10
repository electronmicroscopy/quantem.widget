import pytest

from quantem.widget.showdiffraction import match_candidate, match_sort_key


def _lines(entries):
    return [{"d": d, "i_rel": i} for d, i in entries]


def test_match_candidate_matching():
    # perfect match: every observed line pairs to a reference
    refs = _lines([(3.0, 100), (2.0, 80), (1.5, 60), (1.2, 40)])
    sc = match_candidate([3.0, 2.0, 1.5, 1.2], refs, tol=0.03)
    assert sc["matched"] == 4 and sc["n_obs"] == 4
    assert sc["mean_err"] == pytest.approx(0.0, abs=1e-9)
    assert sc["n_missing_strong"] == 0
    assert all(j is not None for _, j in sc["assignments"])

    # each reference line is used at most once
    sc = match_candidate([2.0, 2.01], _lines([(2.0, 100)]), tol=0.03)
    assert sc["matched"] == 1
    assert [ref for _, ref in sc["assignments"]].count(0) == 1

    # optimal assignment: both lines pair even when greedy would strand one
    refs = _lines([(2.001, 100), (1.985, 80)])
    sc = match_candidate([2.0, 2.01], refs, tol=0.01)
    assert sc["matched"] == 2
    assert all(j is not None for _, j in sc["assignments"])
    # crossed pairing also matches both lines but with larger error
    refs = _lines([(2.0, 100), (2.01, 80)])
    sc = match_candidate([2.0, 2.01], refs, tol=0.03)
    assert sc["matched"] == 2
    assert dict(sc["assignments"]) == {0: 0, 1: 1}
    assert sc["mean_err"] == pytest.approx(0.0, abs=1e-9)

    # reference cards may carry "intensity" instead of "i_rel"
    refs = [
        {"d": 3.0, "intensity": 100.0},
        {"d": 2.0, "intensity": 90.0},
        {"d": 1.5, "intensity": 80.0},
    ]
    sc = match_candidate([3.0, 1.5], refs, tol=0.03)
    assert sc["matched"] == 2
    assert sc["n_missing_strong"] == 1  # the strong d=2.0 line is unexplained


def test_match_sort_key_ranking():
    # absolute calibration: a uniform 5% scale breaks every match
    refs = _lines([(3.0, 100), (2.0, 80), (1.5, 60), (1.2, 40)])
    perfect = match_candidate([3.0, 2.0, 1.5, 1.2], refs, tol=0.03)
    off = match_candidate([3.0 * 1.05, 2.0 * 1.05, 1.5 * 1.05, 1.2 * 1.05], refs, tol=0.03)
    assert off["matched"] == 0
    assert match_sort_key(perfect) < match_sort_key(off)

    # a phase that explains every ring ranks above one that leaves a ring unexplained
    obs = [3.0, 2.0, 1.5, 1.05]
    explains_all = match_candidate(obs, _lines([(3.0, 100), (2.0, 80), (1.5, 60), (1.05, 40)]))
    leaves_one = match_candidate(obs, _lines([(3.0, 100), (2.0, 80), (1.5, 60), (1.2, 40)]))
    assert explains_all["matched"] == 4 and leaves_one["matched"] == 3
    assert match_sort_key(explains_all) < match_sort_key(leaves_one)

    # a missing strong line is penalized more than a missing weak line
    strong = _lines([(3.0, 100), (2.0, 90), (1.5, 80)])
    weak = _lines([(3.0, 100), (2.0, 10), (1.5, 80)])
    obs = [3.0, 1.5]  # the d=2.0 line is inside the observed range but unmatched
    sc_strong = match_candidate(obs, strong, tol=0.03)
    sc_weak = match_candidate(obs, weak, tol=0.03)
    assert sc_strong["n_missing_strong"] == 1 and sc_weak["n_missing_strong"] == 0
    assert match_sort_key(sc_weak) < match_sort_key(sc_strong)

    # cards with no intensity data, and all-zero intensities, both read unknown
    sc = match_candidate([3.0, 1.5], [{"d": 3.0}, {"d": 2.0}, {"d": 1.5}], tol=0.03)
    assert sc["matched"] == 2
    assert sc["n_missing_strong"] is None
    sc_zero = match_candidate([3.0, 1.5], _lines([(3.0, 0), (2.0, 0), (1.5, 0)]), tol=0.03)
    assert sc_zero["n_missing_strong"] is None
    # unknown orders like zero in the sort key
    with_data = match_candidate([3.0, 1.5], _lines([(3.0, 100), (2.0, 10), (1.5, 80)]), tol=0.03)
    assert match_sort_key(with_data) == match_sort_key(sc_zero)


def test_match_candidate_non_finite():
    # non-finite observed or reference d-spacings are dropped, not fatal
    refs = _lines([(3.0, 100), (2.3, 80)])
    sc = match_candidate([float("inf"), 2.3], refs, tol=0.03)
    assert sc["matched"] == 1 and sc["n_obs"] == 1
    sc = match_candidate([float("nan"), 2.3], refs, tol=0.03)
    assert sc["matched"] == 1 and sc["n_obs"] == 1
    sc = match_candidate([2.3], _lines([(float("inf"), 100), (2.3, 80)]), tol=0.03)
    assert sc["matched"] == 1
    assert sc["n_missing_strong"] == 0


def test_match_report_shape():
    # plain report keys, even against a dense reference card
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

    # no reference lines: nothing matches, empty assignments
    sc = match_candidate([2.5, 1.7], [], tol=0.03)
    assert sc["matched"] == 0
    assert sc["assignments"] == []
