# ShowDiffraction Auto: ring detection and a verified phase calibration (2026-10-08)

Branch `science-fixes`, from `platform` (`ca01d32`).

## Question

The all-widget driver ran the ShowDiffraction tutorial's magnetite (Fe3O4) ring
pattern with a different Poisson realization and Auto returned six rings, one of
them spurious, and a k calibration about 26 % off (every d too small, 2.203
instead of 2.967 Å for the first ring), while the widget reported
`Fe3O4 (spinel): 5/6 matched, 1.2% mean error`. Does Auto find the rings that
are there, and only those? Does it calibrate to the true scale, and say so
honestly when it cannot?

## Setup

- Generator: the tutorial's pattern, 512 x 512, center (255.5, 255.5), central
  beam `4 exp(-r^2 / 80)`, background 0.05, Gaussian rings at
  `r = 1 / (d k)` with `k = 0.004` 1/Å per px, Poisson counts at `dose` counts
  per unit (`rng = np.random.default_rng(seed); rng.poisson(dose * (pattern + 0.05))`).
- Phases (all in `PHASE_LIBRARY`):
  - Fe3O4, the tutorial's five rings 220, 311, 400, 422, 440 (d 2.967, 2.532,
    2.099, 1.715, 1.485 Å; strengths 1.7, 1.3, 1.0, 0.8, 0.55; widths 3 to 5 px).
  - Au (fcc): 111, 200, 220, 311, 222 (strengths 1.7, 1.0, 0.8, 0.9, 0.4; widths
    2.5 to 3 px).
  - ZnO (wurtzite, non-cubic): 100, 002, 101, 102, 110, 103 (widths 1.8 px for
    the 100/002/101 triplet, 2.5 to 3 px for the others).
- Doses 2, 20, 100 (the tutorial's), 1000 counts per unit; seeds 0 to 49 each;
  plus the tutorial's exact realization (the generator first draws the
  single-crystal pattern, then the rings).
- Calls: `run_auto(max_rings=<number of built rings>)` (the tutorial's call) and
  `run_auto()` (default `max_rings=8`), on CPU, 1202 runs per code version.
- Detection is scored on the radii `_detect_rings` returns (a ring is found
  when a detected radius is within 3 px of a built one). Calibration is scored on
  the final k: correct when within 0.5 % of 0.004. "Wrong k shown as success"
  counts runs whose k is more than 1 % off while the status and phase match
  contain no failure word. hkl is scored on the final ring nearest each built
  ring (within 3 px).
- Before: `ca01d32`. After: this branch.

## Root cause

Two independent faults, both visible on seed 0 at 100 counts per unit.

1. Detection made a ring out of the background. `detect_ring_radii` took
   `log1p(I - min(I))` of the radial profile and subtracted a Gaussian-smoothed
   copy (sigma 12.8 bins). Around strong features that high-pass leaves negative
   lobes: the detrended profile is -1.205 at r = 23.5 px (beam shoulder) and
   -1.440 at r = 71.6 px (just inside the 220 ring), so the flat background
   between them (-0.076 at r = 47.6 px) became a local maximum with prominence
   1.130, more than the real 311 ring at 98.7 px (1.009). With `max_rings=5` the
   gap took the 311 ring's slot. The same high-pass shifted real peaks by 1 to
   2.5 px (440 at 170.8 instead of 168.4).
2. The phase calibration accepted a consistent but wrong scale.
   `calibration_scale_from_phase` anchored the innermost ring (here the
   spurious one) on each reflection in turn and kept the assignment with the most
   rings within 3 %. Above a few 1/Å the spinel reflections are closer together
   than 3 % (N = 43, 44 are 1.2 % apart), so at almost any scale every ring is
   "matched"; a wrong scale that explains 6 of 6 rings beats the right one that
   leaves the spurious ring out. Indexing then labelled every ring within 3 %,
   which produced the confident `5/6 matched` line.

## Fix

Detection (`lattice.detect_ring_radii`):

- The profile (smoothed by one bin) minus its lower envelope, a grey opening
  over 1/8 of the profile (at least 9 bins). A flat background or a monotone
  beam tail is unchanged by the opening, so it leaves no residual and no lobes.
- A ring is a local maximum of that residual whose prominence is at least 5
  standard errors of the raw frame's azimuthal mean at that radius
  (`radial_profile_px` now returns `sqrt(s^2 / n)` per bin). The denoised
  detection view only locates peaks; significance stays tied to the counts.
- It returns every significant peak, most significant first; `max_rings` keeps
  the first N, and Auto later recovers the remaining significant peaks that sit
  within 4 px of a reflection the verified scale predicts (this replaces
  `recover_predicted_radii`).

Calibration (`lattice.calibration_scale_from_phase`):

- Scale-free first: every ring radius is `scale * g` for one phase, so the ring
  ratios must match the reflection ratios within 1 % (`RATIO_TOL`). Every
  (ring, reflection) pair proposes a scale; rings are assigned one per
  reflection; the scale is the median of `r / g` over the rings within 1 %.
- Each proposal is scored in nats: `-ln p` per explained ring with
  `p = min(1, 2 max(e, 0.1 %) / s)` (the chance of landing within the relative
  error `e` of a reflection whose nearest neighbour is a relative gap `s` away),
  minus `ln 2` per reflection predicted between the innermost and outermost
  explained ring, minus `ln 10` per unexplained ring.
- Refused (`ValueError`, calibration left as it was) when no scale explains two
  rings, when more rings are unexplained than explained, or when a scale more
  than 1 % away scores within `ln 100` (4.6 nats) of the best (ambiguous).

Status (`run_auto`, `_calibrate_phase`, quality panel):

- Ring labels and `phase_match` come from the verified assignment only; a failed
  or ambiguous calibration restores the calibration from before Auto, clears the
  labels and `phase_match`, and the status says
  `calibration failed (ambiguous: 5 rings fit 121.4 px per 1/Å and 4 fit 171.7; ...)`.
- Rings the phase does not explain are named:
  `ZnO explains 4 of 5 rings; not at r = 99.9 px (another phase or an artefact?)`.
- The quality panel's "Unexplained rings" counts every ring without hkl whenever
  a phase is selected (before, a failed calibration reported 0).

## Results: seed sweep, before -> after

Recall lists the built rings in the order of the phase column.

| phase | counts/unit | max_rings | runs | detection recall per ring (before -> after) | spurious rings per run | correct k (within 0.5 %) | wrong k shown as success | uncalibrated (refused) | true hkl per ring, after | median d error, after |
|---|---|---|---|---|---|---|---|---|---|---|
| Au (111, 200, 220, 311, 222) | 2 | 5 | 50 | 1.00 1.00 1.00 1.00 0.00 -> 1.00 1.00 1.00 1.00 0.00 | 1.00 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 0.00 | 0.021 % |
| Au (111, 200, 220, 311, 222) | 2 | 8 | 50 | 1.00 1.00 1.00 1.00 0.00 -> 1.00 1.00 1.00 1.00 0.00 | 1.96 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 0.00 | 0.021 % |
| Au (111, 200, 220, 311, 222) | 20 | 5 | 50 | 1.00 1.00 1.00 1.00 0.00 -> 1.00 1.00 1.00 1.00 0.00 | 1.00 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 0.00 | 0.006 % |
| Au (111, 200, 220, 311, 222) | 20 | 8 | 50 | 1.00 1.00 1.00 1.00 0.00 -> 1.00 1.00 1.00 1.00 0.00 | 2.00 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 0.00 | 0.006 % |
| Au (111, 200, 220, 311, 222) | 100 | 5 | 50 | 1.00 1.00 1.00 1.00 0.00 -> 1.00 1.00 1.00 1.00 0.00 | 1.00 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 0.00 | 0.003 % |
| Au (111, 200, 220, 311, 222) | 100 | 8 | 50 | 1.00 1.00 1.00 1.00 0.00 -> 1.00 1.00 1.00 1.00 0.00 | 2.00 -> 0.00 | 0.00 -> 1.00 | 0.96 -> 0.00 | 0.04 -> 0.00 | 1.00 1.00 1.00 1.00 0.00 | 0.003 % |
| Au (111, 200, 220, 311, 222) | 1000 | 5 | 50 | 1.00 1.00 1.00 1.00 0.00 -> 1.00 1.00 1.00 1.00 1.00 | 1.00 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 1.00 | 0.003 % |
| Au (111, 200, 220, 311, 222) | 1000 | 8 | 50 | 1.00 1.00 1.00 1.00 0.02 -> 1.00 1.00 1.00 1.00 1.00 | 3.98 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 1.00 | 0.003 % |
| Fe3O4 (220, 311, 400, 422, 440) | tutorial | 5 | 1 | 1.00 0.00 1.00 1.00 1.00 -> 1.00 1.00 1.00 1.00 1.00 | 1.00 -> 0.00 | 1.00 -> 1.00 | 0.00 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 1.00 | 0.032 % |
| Fe3O4 (220, 311, 400, 422, 440) | tutorial | 8 | 1 | 1.00 1.00 1.00 1.00 1.00 -> 1.00 1.00 1.00 1.00 1.00 | 1.00 -> 0.00 | 1.00 -> 1.00 | 0.00 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 1.00 | 0.032 % |
| Fe3O4 (220, 311, 400, 422, 440) | 2 | 5 | 50 | 1.00 1.00 1.00 1.00 1.00 -> 1.00 1.00 1.00 1.00 1.00 | 0.00 -> 0.00 | 1.00 -> 1.00 | 0.00 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 1.00 | 0.034 % |
| Fe3O4 (220, 311, 400, 422, 440) | 2 | 8 | 50 | 1.00 1.00 1.00 1.00 1.00 -> 1.00 1.00 1.00 1.00 1.00 | 1.00 -> 0.00 | 0.06 -> 1.00 | 0.94 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 1.00 | 0.034 % |
| Fe3O4 (220, 311, 400, 422, 440) | 20 | 5 | 50 | 1.00 1.00 1.00 1.00 1.00 -> 1.00 1.00 1.00 1.00 1.00 | 0.00 -> 0.00 | 1.00 -> 1.00 | 0.00 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 1.00 | 0.034 % |
| Fe3O4 (220, 311, 400, 422, 440) | 20 | 8 | 50 | 1.00 1.00 1.00 1.00 1.00 -> 1.00 1.00 1.00 1.00 1.00 | 1.00 -> 0.00 | 0.08 -> 1.00 | 0.92 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 1.00 | 0.034 % |
| Fe3O4 (220, 311, 400, 422, 440) | 100 | 5 | 50 | 1.00 0.02 1.00 1.00 1.00 -> 1.00 1.00 1.00 1.00 1.00 | 0.98 -> 0.00 | 0.02 -> 1.00 | 0.98 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 1.00 | 0.032 % |
| Fe3O4 (220, 311, 400, 422, 440) | 100 | 8 | 50 | 1.00 1.00 1.00 1.00 1.00 -> 1.00 1.00 1.00 1.00 1.00 | 1.00 -> 0.00 | 0.06 -> 1.00 | 0.94 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 1.00 | 0.032 % |
| Fe3O4 (220, 311, 400, 422, 440) | 1000 | 5 | 50 | 1.00 0.00 1.00 1.00 0.84 -> 1.00 1.00 1.00 1.00 1.00 | 1.16 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 1.00 | 0.029 % |
| Fe3O4 (220, 311, 400, 422, 440) | 1000 | 8 | 50 | 1.00 1.00 1.00 1.00 0.84 -> 1.00 1.00 1.00 1.00 1.00 | 3.16 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 1.00 1.00 1.00 1.00 | 0.029 % |
| ZnO (100, 002, 101, 102, 110, 103) | 2 | 6 | 50 | 1.00 0.00 1.00 1.00 1.00 1.00 -> 0.98 0.00 1.00 1.00 1.00 1.00 | 1.00 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 0.98 0.00 0.48 1.00 1.00 1.00 | 0.026 % |
| ZnO (100, 002, 101, 102, 110, 103) | 2 | 8 | 50 | 1.00 0.00 1.00 1.00 1.00 1.00 -> 0.98 0.00 1.00 1.00 1.00 1.00 | 1.00 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 0.98 0.00 0.48 1.00 1.00 1.00 | 0.026 % |
| ZnO (100, 002, 101, 102, 110, 103) | 20 | 6 | 50 | 1.00 0.00 1.00 1.00 1.00 1.00 -> 1.00 0.00 1.00 1.00 1.00 1.00 | 1.00 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 0.00 0.24 1.00 1.00 1.00 | 0.011 % |
| ZnO (100, 002, 101, 102, 110, 103) | 20 | 8 | 50 | 1.00 0.00 1.00 1.00 1.00 1.00 -> 1.00 0.00 1.00 1.00 1.00 1.00 | 1.00 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 0.00 0.24 1.00 1.00 1.00 | 0.011 % |
| ZnO (100, 002, 101, 102, 110, 103) | 100 | 6 | 50 | 1.00 0.00 1.00 1.00 1.00 1.00 -> 1.00 0.00 1.00 1.00 1.00 1.00 | 1.00 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 0.00 0.28 1.00 1.00 1.00 | 0.008 % |
| ZnO (100, 002, 101, 102, 110, 103) | 100 | 8 | 50 | 1.00 0.00 1.00 1.00 1.00 1.00 -> 1.00 0.00 1.00 1.00 1.00 1.00 | 1.00 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 0.00 0.28 1.00 1.00 1.00 | 0.008 % |
| ZnO (100, 002, 101, 102, 110, 103) | 1000 | 6 | 50 | 1.00 0.00 1.00 1.00 1.00 1.00 -> 1.00 0.00 1.00 1.00 1.00 1.00 | 1.00 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 0.00 0.36 1.00 1.00 1.00 | 0.007 % |
| ZnO (100, 002, 101, 102, 110, 103) | 1000 | 8 | 50 | 1.00 0.00 1.00 1.00 1.00 1.00 -> 1.00 0.00 1.00 1.00 1.00 1.00 | 2.98 -> 0.00 | 0.00 -> 1.00 | 1.00 -> 0.00 | 0.00 -> 0.00 | 1.00 0.00 0.36 1.00 1.00 1.00 | 0.007 % |

Summary: before, 1087 of the 1200 seeded runs returned a wrong scale presented
as a success (only Fe3O4 with `max_rings=5` at 2 and 20 counts came out right)
and the 1202 runs held 1613 spurious rings. After, 0 spurious rings, k within
0.5 % of the truth in all 1202 runs, and no wrong scale shown as success.

Why some cells stay below 1.00 after the fix (physical limits of the synthetic
profiles, not detector faults):

- Au 222 at 2 to 100 counts: 9.2 px (3 sigma) outside the stronger 311 ring and
  0.44 of its height, it is a shoulder without its own maximum once the profile
  is smoothed by one bin. At 1000 counts it is detected and indexed in every run.
- ZnO 002 is 5 px from the 101 ring, 1.7 times stronger: never a separate
  maximum. Its shoulder pulls the single-Gaussian fit of 101 by about 1 px
  (1 %), so 101 fails the 1 % ratio test in 52 to 76 % of the runs; those runs
  calibrate correctly from the other four rings and say
  `ZnO explains 4 of 5 rings; not at r = 99.9 px (another phase or an artefact?)`.

Scores: the gap between the chosen scale and the best other scale is 7.5 to
10.4 nats in every run (Fe3O4 7.5 to 8.2, Au 9.7 to 10.4, ZnO 8.3 to 9.7),
against the ambiguity margin of 4.6 nats.

Noise peaks (20 seeds per cell, Fe3O4, Au, ZnO and a beam-only pattern): the
largest noise-peak prominence over the raw standard error is 0.5 at 0.5 counts
per unit, 0.8 at 2, 2.0 at 100, 2.4 at 1000 and 2.9 at 10^4 (1.8 with no rings
at all). Every built ring that is a separate maximum stands at 5.1 or more
(Fe3O4 5.4 and Au 7.0 at 0.5 counts, ZnO 5.1 at 2 counts); at 0.5 counts the
ZnO 100 ring (1.5) is lost. Without the one-bin smoothing the noise maximum at
1000 counts was 5.4 (the profile is then the raw, undenoised one), which is why
the smoothing is part of the detector.

## Controls

| case | after | before |
|---|---|---|
| Fe3O4 pattern, phase Au (wrong) | uncalibrated, `calibration failed (ambiguous: 5 rings fit 121.4 px per 1/Å and 4 fit 171.7; ...)` | k +166 %, `Au (fcc): 6/7 matched, 1.3% mean error` |
| Au pattern, phase Fe3O4 (wrong; 111/220/311 are spinel reflections at a 2.06x cell) | k -51 %, `Fe3O4 explains 3 of 4 rings; not at r = 122.6 px`, `3/4 matched` | k +48 %, `6/6 matched`, no warning |
| Fe3O4 pattern, phase NiO (rocksalt) | uncalibrated, ambiguous | k +159 %, `6/7 matched` |
| Fe3O4 pattern, phase gamma-Fe2O3 (same spinel, a 0.6 % smaller) | k +0.62 %, 5/5 (the lattice parameter is the only difference; Identify separates them) | k +54 % |
| beam and background only | `ring detection failed (no rings found)` | 1 ring, `needs >= 2 rings` |
| two rings, 220 + 311 | k -0.03 %, 2/2 | k +55 %, 3 rings |
| two rings, 400 + 440 (an exact sqrt(2) pair) | uncalibrated, ambiguous | k +58 % |
| Fe3O4 + Au rings mixed (overlapping, blended), phase Fe3O4 or Au | uncalibrated, ambiguous | k +18 % / +143 %, `5/6 matched` |
| Fe3O4, 256 px crop (three rings inside) | k -0.03 %, 3/3 | k +81 %, `4/4 matched` |
| Fe3O4 at 0.5 counts per unit | k -0.04 %, 5/5 | k +86 %, `10/10 matched` |
| Fe3O4 at 0.1 counts per unit | one ring found, `needs >= 2 rings` | k +81 %, `7/7 matched` |
| the tutorial's detection-denoise cell (magnetite thinned to median 0.5 counts), `detect_denoise` none and auto | k +0.03 %, 220 311 400 422 440, rms 0.07 px | k +22 %, a 42 px ring labelled 111, rms 0.57 px |

The Au-as-Fe3O4 case is the one wrong scale left: three of the four rings are
exact spinel reflections at that scale. The calibration is applied with its
warning naming the ring Fe3O4 cannot explain; `identify_phase` / `search_phases`
rank Au first.

## Real data

None public. `datasets.showdiffraction_fe3o4` names
`widget-tutorials/showdiffraction/fe3o4-saed` in `bobleesj/quantem-data`, but
that folder is not in the dataset (neither at the pinned revision
`00179851c0015612bfb6e6438e02387f5ffff0ae` nor on `main`), so the tutorial falls
back to the synthetic control. The public gold 4D-STEM scans are convergent-beam
(30 mrad): their mean patterns have no separated Debye-Scherrer rings.

## Tests

- `test_auto_indexes_the_magnetite_rings_at_every_seed`: seeds 0 to 5 at 100
  counts, `max_rings=5`; every ring labelled 220, 311, 400, 422, 440, k within
  0.1 % of truth and frozen to 1e-6 (before the fix all six seeds were 37 to
  109 % off with `6/6 matched`).
- `test_ring_detection_finds_each_ring_at_its_radius_and_nothing_in_the_background`,
  `test_phase_calibration_is_verified_by_ring_ratios` (fcc sqrt(8) alias, an
  unexplained ring, the ambiguous sqrt(2) pair, ZnO labels),
  `test_auto_with_the_wrong_phase_leaves_the_pattern_uncalibrated`.
- The magnetite fixture no longer needs `exclude_radius=70` to keep the beam
  shoulder out.

## Rejected ideas

- Tuning the Gaussian detrend (larger sigma, linear instead of log): any
  high-pass of a profile with a strong beam and strong rings leaves lobes; the
  gap between them stays a relative maximum.
- A noise estimate from the profile itself (running MAD of the residual or of
  first differences): inside the dense Fe3O4 ring block (77 to 175 px) the
  window is mostly ring, the estimate rose from about 0.05 to 34 counts and the
  311, 400 and 422 rings fell to 2.2 to 2.5 sigma. The raw standard error of the azimuthal mean has no such bias.
- Keeping the 5 % relative prominence threshold next to the significance test:
  it rejected the real Au 222 ring at 1000 counts (41 against 1645).
- Least-squares scale over the assigned rings: one ring biased by an unresolved
  neighbour (Au 311 pulled 1.1 px by 222) spread a 0.2 to 0.3 % error over all
  rings and made the fcc sqrt(8) alias look as good. The median is robust to it.
- Ranking proposals by the number of rings within tolerance (the earlier rule,
  3 %): dense high-order reflections (spinel N = 43 and 44 are 1.2 % apart)
  match almost any ring by chance, so a wrong scale explains as many rings as
  the right one. A tighter tolerance only narrows the window; the score weighs
  each match by how unlikely it is instead.
- The chance score alone: exact cubic aliases (all g times 2 or sqrt(8)) tie with
  the truth; the predicted-reflection term breaks the tie. Without the
  unexplained-ring term the sqrt(2) alias of the Fe3O4 rings (3 of 5 explained)
  came within 3.6 nats of the truth and every Fe3O4 run was refused as
  ambiguous.
