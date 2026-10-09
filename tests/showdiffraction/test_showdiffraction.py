"""ShowDiffraction: the science against known references (d-spacings, a known center,
a synthetic lattice), the saved-state round trip and the export page structure."""

import json
import re
import time

import numpy as np
import pytest
import torch

from quantem.widget import Phase, ShowDiffraction, library_phase
from quantem.widget.showdiffraction import lattice
from quantem.widget.showdiffraction.phases import PHASE_LIBRARY

FE3O4_D = {"220": 2.9687, "311": 2.5317, "400": 2.0992, "422": 1.7140, "440": 1.4844}
K_SYNTH = 0.004  # 1/Å per pixel of the synthetic ring patterns
K_SAED = 0.018  # 1/Å per pixel of the synthetic single-crystal pattern
SAED_SPACING_PX = 28.0  # one reciprocal lattice step: d = 1 / (28 * 0.018) = 1.984 Å
# k (1/Å per px) that Auto finds on ring_pattern(seed) with max_rings=5, frozen 2026-10-08. Before the
# fix every one of these seeds came out 37 to 109 % off with "Fe3O4 (spinel): 6/6 matched": the
# detector took the background between the central beam and the 220 ring as a ring and the 3 %
# assignment accepted a consistent but wrong scale.
FROZEN_AUTO_K = {
    0: 0.003999974250962634,
    1: 0.004000264041479711,
    2: 0.004000069799793147,
    3: 0.004000095050272704,
    4: 0.003999891564158511,
    5: 0.003999821288085391,
}


def ring_pattern(center=None, size=512, dose=100.0, ratio=1.0, angle_deg=0.0, seed=0, lines=FE3O4_D):
    """Debye-Scherrer rings of ``lines`` (Fe3O4 by default) at K_SYNTH with a bright central
    beam and shot noise; ``ratio`` / ``angle_deg`` stretch the rings into ellipses."""
    center = ((size - 1) / 2, (size - 1) / 2) if center is None else center
    rows, cols = np.mgrid[0:size, 0:size]
    d_row, d_col = rows - center[0], cols - center[1]
    phi = np.deg2rad(angle_deg)
    major = d_col * np.cos(phi) + d_row * np.sin(phi)
    minor = -d_col * np.sin(phi) + d_row * np.cos(phi)
    radius = np.hypot(major / np.sqrt(ratio), minor * np.sqrt(ratio))
    pattern = 4.0 * np.exp(-(radius**2) / 80.0)
    for (d_ref, strength, width) in zip(lines.values(), (1.7, 1.3, 1.0, 0.8, 0.55), (3.0, 3.5, 4.0, 4.5, 5.0)):
        pattern += strength * np.exp(-((radius - 1.0 / (d_ref * K_SYNTH)) ** 2) / (2 * width**2))
    return np.random.default_rng(seed).poisson(dose * (pattern + 0.05)).astype(np.float32)


def lattice_pattern(size=256, seed=0):
    """A [001] zone-axis pattern of a primitive cubic cell: spots on a square grid."""
    center = (size - 1) / 2
    rows, cols = np.mgrid[0:size, 0:size]
    pattern = np.zeros((size, size), np.float32)
    for h in range(-4, 5):
        for k in range(-4, 5):
            amplitude = 6.0 if h == 0 and k == 0 else 1.0 / (1 + 0.4 * (h * h + k * k))
            pattern += amplitude * np.exp(
                -((rows - center - h * SAED_SPACING_PX) ** 2 + (cols - center - k * SAED_SPACING_PX) ** 2) / 8.0
            )
    return np.random.default_rng(seed).poisson(1000.0 * (pattern + 0.05)).astype(np.float32)


@pytest.fixture(scope="module")
def magnetite():
    """Auto (center, rings, Fe3O4 calibration, indexing) on the synthetic Fe3O4 rings, run once per module."""
    widget = ShowDiffraction(ring_pattern(), title="Magnetite-like rings", verbose=False)
    widget.phase_name = "Fe3O4"
    widget.run_auto(max_rings=5)
    return widget


@pytest.fixture(scope="module")
def saed():
    """Detected and indexed spots of the synthetic [001] primitive-cubic pattern, run once per module."""
    center = (255 / 2, 255 / 2)
    widget = ShowDiffraction(lattice_pattern(), center=center, bf_radius=14, k_pixel_size=K_SAED, verbose=False)
    widget.detect_spots(max_spots=12)
    widget.index_spots(Phase.from_cubic("Cubic", 1.984, absences="none"))
    return widget


def numpy_radial_profile(frame, center, mask=None, angular_range=None, ellipse=None):
    """The float64 NumPy radial profile the widget used before it moved to torch (np.hypot,
    np.digitize, np.bincount): the reference the device profile must reproduce."""
    n_rows, n_cols = frame.shape
    max_radius = max(1.0, float(min(center[0], center[1], n_rows - 1 - center[0], n_cols - 1 - center[1])))
    n_bins = max(1, round(max_radius))
    d_row = np.arange(n_rows, dtype=np.float64)[:, None] - center[0]
    d_col = np.arange(n_cols, dtype=np.float64)[None, :] - center[1]
    radii = np.hypot(*lattice.ellipse_to_circle(d_row, d_col, *ellipse)) if ellipse else np.hypot(d_row, d_col)
    keep = np.ones(frame.shape, bool) if mask is None else ~mask
    if angular_range is not None:
        keep &= lattice.in_wedge(np.degrees(np.arctan2(d_row, d_col)) % 360.0, *angular_range)
    values = frame.astype(np.float64)[keep]
    edges = np.linspace(0.0, max_radius, n_bins + 1)
    indices = np.digitize(radii[keep], edges) - 1
    inside = indices < n_bins
    indices, values = indices[inside], values[inside]
    counts = np.bincount(indices, minlength=n_bins).astype(np.float64)
    sums = np.bincount(indices, weights=values, minlength=n_bins)
    sums_sq = np.bincount(indices, weights=values * values, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = np.where(counts > 0, sums / counts, 0.0)
        variance = np.maximum(sums_sq - counts * intensity**2, 0.0) / (counts - 1.0)
        standard_error = np.where(counts > 1, np.sqrt(variance / counts), np.inf)
    return 0.5 * (edges[:-1] + edges[1:]).astype(np.float32), intensity.astype(np.float32), standard_error


# --- Phases ---
def test_phase_geometry_matches_textbook_values():
    si = library_phase("Si")
    assert si.d_spacing((1, 1, 1)) == pytest.approx(5.4311 / np.sqrt(3), rel=1e-9)
    assert si.plane_angle((1, 0, 0), (1, 1, 0)) == pytest.approx(45.0)
    assert si.plane_angle((1, 1, 1), (1, -1, 0)) == pytest.approx(90.0)
    # diamond cubic: 111, 220, 311 allowed; 200 and 222 forbidden
    assert [r["hkl_str"] for r in si.reflections(d_min=1.6)] == ["111", "220", "311"]
    assert not si.is_allowed((2, 0, 0)) and not si.is_allowed((2, 2, 2))
    # hcp Ti: (001) forbidden, (002) allowed, c axis spacing
    ti = library_phase("Ti")
    assert not ti.is_allowed((0, 0, 1)) and ti.is_allowed((0, 0, 2))
    assert ti.d_spacing((0, 0, 2)) == pytest.approx(4.6860 / 2)
    assert ti.d_spacing((1, 0, 0)) == pytest.approx(2.9500 * np.sqrt(3) / 2)
    # a d-spacing card matches without a lattice
    card = Phase.from_dspacings("card", [(2.355, "111", 100), (2.039, "200", 50)])
    assert card.match_d(2.04)[0]["hkl_str"] == "200"
    with pytest.raises(ValueError):
        card.d_spacing((1, 1, 1))


def test_every_library_phase_builds():
    for name in PHASE_LIBRARY:
        phase = library_phase(name)
        assert phase.d_spacing((1, 1, 1)) > 0.5, name
    with pytest.raises(ValueError):
        library_phase("unobtainium")
    with pytest.raises(ValueError):
        Phase("flat", 4.0, 4.0, 4.0, 90.0, 90.0, 180.0)


# --- Science against references ---
def test_ring_radii_match_known_d_spacings(magnetite):
    assert magnetite.analysis_status == ""
    assert magnetite.k_calibrated
    assert magnetite.k_pixel_size == pytest.approx(K_SYNTH, rel=5e-3)
    assert magnetite.calibration_rms_px < 1.0
    by_hkl = {ring["hkl"]: ring for ring in magnetite.rings}
    assert set(FE3O4_D) <= set(by_hkl)
    for hkl, d_ref in FE3O4_D.items():
        assert by_hkl[hkl]["radius_px"] == pytest.approx(1.0 / (d_ref * K_SYNTH), abs=1.0)
        assert by_hkl[hkl]["d_spacing"] == pytest.approx(d_ref, rel=5e-3)
        assert by_hkl[hkl]["fit_quality"] > 0.9


def test_auto_indexes_the_magnetite_rings_at_every_seed():
    for seed, frozen_k in FROZEN_AUTO_K.items():
        widget = ShowDiffraction(ring_pattern(seed=seed), verbose=False)
        widget.phase_name = "Fe3O4"
        widget.run_auto(max_rings=5)
        assert [ring["hkl"] for ring in widget.rings] == list(FE3O4_D), seed
        assert widget.k_pixel_size == pytest.approx(K_SYNTH, rel=1e-3), seed
        assert widget.k_pixel_size == pytest.approx(frozen_k, rel=1e-6), seed
        assert widget.analysis_status == "" and widget.phase_match.startswith("Fe3O4 (spinel): 5/5 matched"), seed


def test_ring_detection_finds_each_ring_at_its_radius_and_nothing_in_the_background():
    """The azimuthal mean of the five-ring pattern: beam tail, flat background, rings. The flat
    background between the beam and the 220 ring must not become a ring (the earlier
    Gaussian detrend made it one on noisy profiles) and every ring keeps its own bin (the
    detrend shifted them by 1 to 2 px)."""
    radii = np.arange(256, dtype=np.float32) + 0.5
    profile = 100.0 * (4.0 * np.exp(-(radii**2) / 80.0) + 0.05)
    centers = (84.5, 98.5, 119.5, 145.5, 168.5)
    for center, height, width in zip(centers, (1.7, 1.3, 1.0, 0.8, 0.55), (3.0, 3.5, 4.0, 4.5, 5.0)):
        profile += 100.0 * height * np.exp(-((radii - center) ** 2) / (2 * width**2))
    profile += np.random.default_rng(0).normal(0.0, 0.05, radii.size)
    standard_error = np.full(radii.size, 0.1)
    found = lattice.detect_ring_radii(radii, profile, standard_error, exclude_radius=13.0)
    assert sorted(found) == list(centers) and found[0] == 84.5  # the strongest ring first
    # a flat profile has no ring, and a bump below five standard errors is not one
    assert lattice.detect_ring_radii(radii, np.full(256, 5.0), standard_error, exclude_radius=13.0) == []
    faint = 5.0 + 0.4 * np.exp(-((radii - 120.5) ** 2) / 18.0)
    assert lattice.detect_ring_radii(radii, faint, standard_error, exclude_radius=13.0) == []


def test_phase_calibration_is_verified_by_ring_ratios():
    au = library_phase("Au")
    inv_d = [1.0 / reflection["d"] for reflection in au.reflections(d_min=0.5)]
    labels = [reflection["hkl_str"] for reflection in au.reflections(d_min=0.5)]
    # 111, 200, 220, 311 at 250 px per 1/Å. Every g times sqrt(8) (422, 440, 800, 664) has the
    # same ratios, but that alias predicts some 20 rings between them that are not there.
    radii = [250.0 / au.d_spacing(hkl) for hkl in ((1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1))]
    solution = lattice.calibration_scale_from_phase(radii, inv_d)
    assert solution["scale"] == pytest.approx(250.0, rel=1e-9)
    assert [labels[i] for i in solution["assigned"]] == ["111", "200", "220", "311"]
    # a ring the phase does not have stays unexplained instead of bending the scale
    solution = lattice.calibration_scale_from_phase(radii + [140.0], inv_d)
    assert solution["scale"] == pytest.approx(250.0, rel=1e-9) and solution["assigned"][-1] is None
    # two rings in the ratio sqrt(2) fit 111/220-type, 200/220, 220/400 ...: no scale is decided
    with pytest.raises(ValueError, match="ambiguous"):
        lattice.calibration_scale_from_phase([200.0, 200.0 * np.sqrt(2.0)], inv_d)
    # non-cubic: ZnO (wurtzite) rings keep their own labels
    zno = library_phase("ZnO")
    zno_radii = [250.0 / zno.d_spacing(hkl) for hkl in ((1, 0, 0), (0, 0, 2), (1, 0, 1), (1, 0, 2), (1, 1, 0))]
    zno_labels = [reflection["hkl_str"] for reflection in zno.reflections(d_min=0.5)]
    solution = lattice.calibration_scale_from_phase(zno_radii, [1.0 / r["d"] for r in zno.reflections(d_min=0.5)])
    assert solution["scale"] == pytest.approx(250.0, rel=1e-9)
    assert [zno_labels[i] for i in solution["assigned"]] == ["100", "002", "101", "102", "110"]


def test_auto_with_the_wrong_phase_leaves_the_pattern_uncalibrated():
    widget = ShowDiffraction(ring_pattern(), verbose=False)
    widget.phase_name = "Au"
    widget.run_auto(max_rings=5)
    assert widget.analysis_status.startswith("Auto: calibration failed (ambiguous")
    assert not widget.k_calibrated and widget.phase_match == ""
    assert all(ring["hkl"] == "" and ring["d_spacing"] is None for ring in widget.rings)
    widget._quality_report()
    assert widget._quality["n_unexplained_rings"] == 5


def test_index_rings_labels_only_what_the_ring_ratios_verify():
    widget = ShowDiffraction(ring_pattern(), verbose=False)
    widget.phase_name = "Fe3O4"
    widget.run_auto(max_rings=5)
    # a ring whose d is 2 % from 400: a d-only candidate the ratios do not explain
    widget._request = {"action": "add_ring", "args": [1.0 / (FE3O4_D["400"] * 1.02 * K_SYNTH)], "seq": 1}
    widget._request = {"action": "index_rings", "args": [], "seq": 2}
    by_radius = sorted(widget.rings, key=lambda ring: ring["radius_px"])
    assert [ring["hkl"] for ring in by_radius] == ["220", "311", "", "400", "422", "440"]
    assert by_radius[2]["hkl_candidates"] == ["400"]
    assert widget.analysis_status.startswith("Indexed 5/6 rings against Fe3O4, verified by ring ratios; unverified (d within 3% only) at r = 116.")
    assert widget.phase_match.startswith("Fe3O4 (spinel): 5/6 matched")
    # calibrating the 220 ring as if it were 311 scales every d by 0.85: d-matching within 3 % still
    # finds candidates for several rings, the ratios put the scale 17 % away, so nothing is labelled
    widget._request = {"action": "remove_ring", "args": [by_radius[2]["id"]], "seq": 3}
    widget._request = {"action": "calibrate_from_ring", "args": [by_radius[0]["radius_px"], FE3O4_D["311"]], "seq": 4}
    widget._request = {"action": "index_rings", "args": [], "seq": 5}
    assert all(ring["hkl"] == "" for ring in widget.rings) and widget.phase_match == ""
    assert any(ring["hkl_candidates"] for ring in widget.rings)
    assert "verified by ring ratios; unverified" in widget.analysis_status
    assert "% from the current calibration" in widget.analysis_status and "+17." in widget.analysis_status


def test_center_detection_recovers_known_center():
    center = (255.5 + 3.3, 255.5 - 4.7)
    pattern = ring_pattern(center=center)
    widget = ShowDiffraction(pattern, verbose=False)
    # bright-field disk centroid, then the two symmetry estimators
    assert (widget.center_row, widget.center_col) == pytest.approx(center, abs=1.0)
    for method in ("symmetry", "phase_corr", "auto"):
        picked = lattice.pick_center(pattern, method=method, guess=(widget.center_row, widget.center_col))
        assert (picked["row"], picked["col"]) == pytest.approx(center, abs=0.3), method
    widget._request = {"action": "refine_center", "args": [], "seq": 1}
    assert (widget.center_row, widget.center_col) == pytest.approx(center, abs=0.3)
    assert widget.analysis_status.startswith("Center (")
    assert widget._request == {}


def test_hkl_assignment_on_synthetic_lattice(saed):
    assert len(saed.spots) == 12
    assert saed.zone_axis == "[001]"
    assert all(spot["hkl"] for spot in saed.spots)
    labels = {spot["hkl"] for spot in saed.spots}
    assert {"100", "110"} <= labels
    for spot in saed.spots:
        order = sum(int(i) ** 2 for i in spot["hkl"])
        assert spot["d_spacing"] == pytest.approx(1.984 / np.sqrt(order), rel=5e-3)
        assert spot["fit_quality"] > 0.9
    # inter-spot angles against the ideal lattice vector each spot sits on
    lattice_vectors = [
        (round((spot["row"] - 127.5) / SAED_SPACING_PX), round((spot["col"] - 127.5) / SAED_SPACING_PX))
        for spot in saed.spots
    ]
    for spot, vector in zip(saed.spots[1:], lattice_vectors[1:]):
        assert spot["angle_deg"] == pytest.approx(lattice.vector_angle(lattice_vectors[0], vector), abs=0.5)
    assert saed.phase_match.startswith("Cubic (none): 12/12 matched")


def test_identify_and_search_rank_the_true_phase_first(magnetite):
    expected = [library_phase(n) for n in ("Fe3O4", "γ-Fe2O3", "α-Fe2O3 (hematite)", "α-Fe")]
    ranked = magnetite.identify_phase(expected)
    assert ranked[0]["name"] == "Fe3O4"
    assert ranked[0]["matched"] == ranked[0]["n_obs"] == 5
    assert ranked[0]["mean_err"] < ranked[1]["mean_err"]
    assert [row["hkl"] for row in ranked[0]["lines"]] == ["440", "422", "400", "311", "220"]
    magnetite.identify_elements = "Fe, O"
    candidates = magnetite.search_phases()
    assert candidates[0]["name"] == "Fe3O4"
    assert all(set(re.findall(r"[A-Z][a-z]?", c["name"])) <= {"Fe", "O"} for c in candidates)
    assert magnetite._identify_results[0]["name"] == "Fe3O4"
    assert magnetite.phase_match.startswith("Fe3O4: 5/5 lines")


def test_identify_with_an_empty_element_field_searches_the_whole_library(magnetite):
    magnetite.identify_elements = ""
    magnetite._request = {"action": "identify", "args": [], "seq": 1}
    assert magnetite.analysis_status.startswith("Fe3O4: 5/5 lines")
    assert len(magnetite._identify_results) == 10
    magnetite.custom_phases = [{"name": "My spinel", "a": 8.3967, "absences": "spinel"}]
    assert [report["name"] for report in magnetite.search_phases(custom_only=True)] == ["My spinel"]
    magnetite.custom_phases = []


def test_search_phases_absurd_calibration_bounded():
    widget = ShowDiffraction(ring_pattern(size=256), verbose=False)
    widget._request = {"action": "detect_rings", "args": [], "seq": 1}
    assert len(widget.rings) >= 2
    # a 0.25 Å ring puts every reflection below d_min: the enumeration must still stop
    widget._request = {"action": "calibrate_from_ring", "args": [widget.rings[0]["radius_px"], 0.25], "seq": 2}
    assert widget.k_calibrated
    start = time.perf_counter()
    reports = widget.identify_phase([library_phase("Au"), library_phase("Fe3O4")])
    assert time.perf_counter() - start < 2.0
    assert len(reports) == 2
    # the enumeration floors d_min at 0.2 Å and caps the Miller index
    assert min(r["d"] for r in library_phase("Au").reflections(d_min=1e-6)) >= 0.2


def test_detection_denoise_picks_anscombe_and_display_denoise_is_view_only():
    sparse = ring_pattern(size=256, dose=0.5, seed=2)
    assert lattice.pick_detect_denoise(sparse) == "anscombe"
    assert lattice.pick_detect_denoise(ring_pattern(size=256, dose=1e4)) == "none"
    widget = ShowDiffraction(sparse, verbose=False)
    widget.phase_name = "Fe3O4"
    widget.run_auto()
    assert sum(1 for r in widget.rings if r["hkl"]) >= 3
    raw_bytes, rings = widget.frame_bytes, list(widget.rings)
    widget.denoise = "gaussian"
    assert widget.frame_bytes != raw_bytes
    assert widget.rings == rings


def test_fit_ellipse_recovers_distortion():
    # one ring, so the annulus around it holds no neighbour
    pattern = ring_pattern(ratio=1.06, angle_deg=30.0, lines={"220": FE3O4_D["220"]})
    widget = ShowDiffraction(pattern, center=(255.5, 255.5), bf_radius=10, verbose=False)
    widget._request = {"action": "add_ring", "args": [1.0 / (FE3O4_D["220"] * K_SYNTH)], "seq": 1}
    widget._request = {"action": "fit_ellipse", "args": [], "seq": 2}
    assert widget.ellipse_ratio == pytest.approx(1.06, abs=0.01)
    assert widget.ellipse_angle == pytest.approx(30.0, abs=5.0)
    assert widget.analysis_status == f"Ellipse ratio {widget.ellipse_ratio:.3f} at {widget.ellipse_angle:.1f}°"


def test_merge_aligns_shifted_frames():
    base = ring_pattern(size=128)
    stack = np.stack([base, np.roll(base, (2, -3), axis=(0, 1)), np.roll(base, (-1, 2), axis=(0, 1))])
    _, shifts, used = lattice.align_frames(stack.astype(np.float64))
    assert used == [True, True, True]
    # the shift that undoes each roll
    assert shifts[1] == pytest.approx((-2.0, 3.0), abs=0.2)
    assert shifts[2] == pytest.approx((1.0, -2.0), abs=0.2)
    widget = ShowDiffraction(stack, verbose=False)
    widget._request = {"action": "merge", "args": [], "seq": 1}
    assert widget.n_frames == 4 and widget.frame_idx == 3
    assert widget.analysis_status == "Merged 3/3 frames"
    widget._request = {"action": "merge", "args": [], "seq": 2}
    assert widget.n_frames == 4  # the merged frame is not merged again


@pytest.mark.parametrize(
    "device",
    ["numpy", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))],
)
def test_radial_profile_matches_the_float64_numpy_reference(device):
    rng = np.random.default_rng(3)
    counts = rng.poisson(100.0, (512, 512)).astype(np.float32)
    smooth = (rng.random((512, 512)) * 1000.0).astype(np.float32)
    mask = np.zeros((512, 512), bool)
    mask[:100, :170] = True
    # an integer center puts exact radii (hypot(54, 240) = 246) on the integer bin edges
    for center in [(256.0, 256.0), (255.5, 255.5), (243.37, 261.81)]:
        for options in [{}, {"mask": mask}, {"angular_range": (350.0, 10.0), "mask": mask}, {"angular_range": (45.0, 225.0)}, {"ellipse": (1.07, 23.0)}]:
            for frame in (counts, smooth):
                radii, intensity, standard_error = numpy_radial_profile(frame, center, **options)
                ellipse = options.get("ellipse")
                got = lattice.radial_profile_px(
                    frame if device == "numpy" else torch.from_numpy(frame).to(device),
                    center=center,
                    mask=options.get("mask"),
                    angular_range=options.get("angular_range"),
                    ellipse_ratio=ellipse[0] if ellipse else 1.0,
                    ellipse_angle=ellipse[1] if ellipse else 0.0,
                    ellipse_corrected=ellipse is not None,
                )
                np.testing.assert_array_equal(got[0], radii)
                np.testing.assert_array_equal(got[1], intensity)
                if frame is counts:
                    # integer counts sum exactly in float64 in any order
                    np.testing.assert_array_equal(got[2], standard_error)
                else:
                    np.testing.assert_allclose(got[2], standard_error, rtol=1e-12)


def test_ring_sectors_read_only_the_annulus_box_and_sum_the_same_pixels():
    frame = ring_pattern(ratio=1.08, angle_deg=35.0).astype(np.float64)
    # the whole-frame computation the box replaces
    rows, cols = np.arange(512.0)[:, None], np.arange(512.0)[None, :]
    for center, ratio, radius in [((255.5, 255.5), 1.08, 168.4), ((240.3, 270.9), 0.9, 120.0), ((255.5, 255.5), 1.0, 300.0)]:
        radii = lattice.corrected_radius(rows - center[0], cols - center[1], ellipse_ratio=ratio, ellipse_angle=35.0,
                                         ellipse_corrected=True)
        selected = np.abs(radii - radius) <= 6.0
        theta = np.degrees(np.arctan2(rows - center[0], cols - center[1])) % 360.0
        sector = np.minimum((theta[selected] / 2.0).astype(int), 179)
        _, counts, intensity_sum, _, weighted_radius_sum = lattice.ring_sectors(
            frame, center=center, radius_px=radius, half_width=6.0, n_theta=180,
            ellipse_ratio=ratio, ellipse_angle=35.0, ellipse_corrected=True)
        np.testing.assert_array_equal(counts, np.bincount(sector, minlength=180))
        np.testing.assert_array_equal(intensity_sum, np.bincount(sector, weights=frame[selected], minlength=180))
        weight = np.clip(frame[selected] - np.median(frame[selected]), 0.0, None)
        np.testing.assert_array_equal(weighted_radius_sum, np.bincount(sector, weights=weight * radii[selected], minlength=180))


def test_a_center_click_profiles_each_ring_frame_once(monkeypatch):
    stack = np.stack([ring_pattern(seed=0), ring_pattern(seed=1)])
    widget = ShowDiffraction(stack, verbose=False)
    widget.run_auto(max_rings=3)
    widget.frame_idx = 1
    widget._request = {"action": "add_ring", "args": [60.0], "seq": 1}
    widget.frame_idx = 0
    widget.show_profile = True
    calls = []
    profile = lattice.radial_profile_px
    monkeypatch.setattr(lattice, "radial_profile_px", lambda frame, **kwargs: calls.append(1) or profile(frame, **kwargs))
    # the page sends a click as one message holding both coordinates
    widget.set_state({"center_row": 254.0, "center_col": 257.5})
    assert len(calls) == 2  # frame 0 (its rings and the profile panel) and frame 1 (one ring)
    radii, intensity, _ = numpy_radial_profile(stack[1], (254.0, 257.5))
    assert widget.rings[-1]["intensity"] == float(intensity[np.argmin(np.abs(radii - 60.0))])
    radii, intensity, _ = numpy_radial_profile(stack[0], (254.0, 257.5))
    assert widget._profile_data == np.concatenate([radii, intensity]).astype(np.float32).tobytes()


# --- Widget state ---
def test_request_channel_edits_spots_and_reports_failures():
    widget = ShowDiffraction(lattice_pattern(), center=(127.5, 127.5), bf_radius=14, verbose=False)
    widget._request = {"action": "add_spot", "args": [127.5 + SAED_SPACING_PX, 127.5], "seq": 1}
    widget._request = {"action": "add_spot", "args": [127.5, 127.5 + SAED_SPACING_PX], "seq": 2}
    assert [s["id"] for s in widget.spots] == [1, 2]
    assert widget.spots[1]["angle_deg"] == pytest.approx(90.0, abs=0.5)
    assert widget.spots[0]["d_spacing"] is None  # uncalibrated
    widget._request = {"action": "calibrate_from_spot", "args": [widget.spots[0]["row"], widget.spots[0]["col"], 1.984], "seq": 3}
    assert widget.k_pixel_size == pytest.approx(K_SAED, rel=5e-3)
    assert widget.spots[1]["d_spacing"] == pytest.approx(1.984, rel=5e-3)
    widget._request = {"action": "remove_spot", "args": [1], "seq": 4}
    assert [s["id"] for s in widget.spots] == [2]
    widget._request = {"action": "undo_spot", "args": [], "seq": 5}
    assert widget.spots == []
    widget._request = {"action": "index_spots", "args": [], "seq": 6}
    assert widget.analysis_status == "Index spots failed: no phase selected; set phase_name or add a custom phase"
    widget._request = {"action": "add_ring", "args": [-3.0], "seq": 7}
    assert widget.analysis_status == "Add ring failed: radius_px must be positive, got -3.0"
    assert widget._request == {}


def test_state_round_trip_and_measurements(saed, tmp_path):
    path = tmp_path / "saed_state.json"
    saed.save(path)
    envelope = json.loads(path.read_text())
    assert envelope["widget_name"] == "ShowDiffraction"
    restored = ShowDiffraction(lattice_pattern(), center=(0.0, 0.0), bf_radius=1.0, verbose=False)
    restored.load_state_dict(envelope["state"])
    assert restored.state_dict() == saed.state_dict()
    assert restored.zone_axis == "[001]"
    records = ShowDiffraction.measurements_from_state(path)
    assert len(records) == 12
    assert records[0]["kind"] == "spot" and records[0]["hkl"] and records[0]["d_angstrom"] == pytest.approx(1.984, rel=5e-3)
    csv_path = ShowDiffraction.measurements_from_state(envelope["state"], tmp_path / "m.csv")
    header = csv_path.read_text().splitlines()[0]
    assert header.startswith("id,kind,raw_row,raw_col,row,col") and header.endswith("hkl,hkl_candidates,note")
    with pytest.raises(ValueError):
        ShowDiffraction.measurements_from_state({"widget_name": "Show2D", "state": {}})


def test_export_html_structure(saed, tmp_path):
    out = saed.export_html(tmp_path / "saed.html", title="SAED export")
    html = out.read_text()
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert "<title>SAED export</title>" in html
    start = html.index('<script type="application/vnd.jupyter.widget-state+json">')
    state = json.loads(html[html.index(">", start) + 1 : html.index("</script>", start)])
    (model,) = [m["state"] for m in state["state"].values() if "ShowDiffraction" in str(m["state"].get("_anywidget_id"))]
    assert model["_anywidget_id"] == "quantem.widget.showdiffraction.widget.ShowDiffraction"
    assert model["offline"] is True and model["export_enabled"] is False
    assert len(model["spots"]) == 12 and model["zone_axis"] == "[001]"
    assert model["k_pixel_size"] == K_SAED and model["_request"] == {}
    assert "_spot_add_request" not in model and "pixel_size" not in model
    assert saed.export_status.startswith("Exported saed.html (")
    widget = ShowDiffraction(np.ones((32, 32), np.float32), verbose=False)
    widget.free()
    with pytest.raises(ValueError):
        widget.export_html(tmp_path / "freed.html")


def test_hot_pixels_are_kept_unless_removal_is_asked_for(capsys):
    # A counting-detector frame with two hot pixels: the recorded counts stay by default.
    rng = np.random.default_rng(3)
    frame = rng.poisson(20, size=(64, 64)).astype(np.uint16)
    frame[5, 7] = frame[40, 41] = 60000
    kept = ShowDiffraction(frame, device="cpu", verbose=False)
    assert kept._frame(0)[5, 7] == 60000
    assert "2 pixels exceed" in capsys.readouterr().out
    removed = ShowDiffraction(frame, device="cpu", remove_hot_pixels=True, verbose=False)
    assert removed._frame(0)[5, 7] == 0 and removed._frame(0)[40, 41] == 0
    assert "zeroed 2 hot pixels" in capsys.readouterr().out
