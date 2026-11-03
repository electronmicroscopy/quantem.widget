// @ts-nocheck
import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { computeGofR, computeGr, dampOriginOscillations, kWindow } from "./pdf_compute";

// Python reference: quantem.diffraction.PairDistributionFunction on a
// k-calibrated synthetic 4D-STEM dataset. The JS must reproduce the
// library's background, F(k), S(k), G(r), damped F(k)/G(r) and g(r).
// Skips when the interpreter in $PYTHON (default "python") lacks the library
// the widget is written against (calculate_Gr with density and max_iter, and
// the Lorch k window).
function pdfReferenceFixture() {
  const python = process.env.PYTHON || "python";
  const code = String.raw`
import inspect
import json
import sys

import numpy as np
import torch

try:
    from quantem.core.datastructures.dataset4dstem import Dataset4dstem
    from quantem.diffraction import PairDistributionFunction
except ImportError:
    sys.exit(1)
if not {"density", "max_iter"} <= set(inspect.signature(PairDistributionFunction.calculate_Gr).parameters):
    sys.exit(1)

det = 64
rr, cc = np.mgrid[0:det, 0:det].astype(np.float64)
rad = np.hypot(rr - 31.5, cc - 31.5)
dp = (
    400.0 * np.exp(-(rad**2) / (2 * 2.0**2))
    + 30.0 * np.exp(-rad / 9.0)
    + 6.0 * np.exp(-((rad - 9.0) ** 2) / 5.0)
    + 2.5 * np.exp(-((rad - 17.0) ** 2) / 9.0)
    + 1.2 * np.exp(-((rad - 25.0) ** 2) / 12.0)
    + 0.2
).astype(np.float32)
data = np.broadcast_to(dp, (2, 2, det, det)).copy()
ds = Dataset4dstem.from_array(data, sampling=(1.0, 1.0, 0.05, 0.05), units=("A", "A", "A^-1", "A^-1"))
pdf = PairDistributionFunction.from_data(
    ds, find_origin=False, origin_row=31.5, origin_col=31.5, num_annular_bins=90, device="cpu"
)
# An older window (the sine taper is 1 mid-band) is not the one the JS follows.
w_mid = float(pdf._k_window(torch.tensor([0.75]), 0.5, 1.0)[0])
if abs(w_mid - np.sin(0.75 * np.pi) / (0.75 * np.pi)) > 1e-6:
    sys.exit(1)
qq = np.asarray(pdf.qq, dtype=float)
cases = [
    dict(name="defaults", k_min_fit=None, k_max_fit=None, k_min_window=None, k_max_window=None,
         k_lowpass=None, k_highpass=None, r_max=10.0, r_step=0.02,
         density=None, r_cut=0.8),
    dict(name="window", k_min_fit=0.3, k_max_fit=1.5, k_min_window=0.35, k_max_window=1.4,
         k_lowpass=None, k_highpass=None, r_max=12.0, r_step=0.05,
         density=None, r_cut=0.8),
    dict(name="lowpass", k_min_fit=0.3, k_max_fit=1.5, k_min_window=None, k_max_window=None,
         k_lowpass=0.05, k_highpass=None, r_max=10.0, r_step=0.02,
         density=None, r_cut=0.8),
    dict(name="bandpass", k_min_fit=0.3, k_max_fit=1.5, k_min_window=None, k_max_window=None,
         k_lowpass=0.08, k_highpass=0.03, r_max=10.0, r_step=0.02,
         density=None, r_cut=0.8),
    dict(name="damped", k_min_fit=0.3, k_max_fit=1.5, k_min_window=None, k_max_window=None,
         k_lowpass=None, k_highpass=None, r_max=10.0, r_step=0.02,
         density=0.05, r_cut=0.8),
    # r_cut on the r = 0.78 point, which torch's float32 arange (Apple silicon)
    # puts one ulp above float32(0.78): the damping leaves that point out.
    dict(name="damped_rcut_tie", k_min_fit=0.3, k_max_fit=1.5, k_min_window=None,
         k_max_window=None, k_lowpass=None, k_highpass=None, r_max=10.0, r_step=0.02,
         density=0.05, r_cut=0.78),
    # 3 * k_lowpass / dk sits just above an integer in float64 but below it with
    # the library's float32 dk: the kernel radius must follow the float32 value.
    dict(name="lowpass_kernel_edge", k_min_fit=0.3, k_max_fit=1.5, k_min_window=None, k_max_window=None,
         k_lowpass=0.05 * 4 / 3 * (1 + 3e-9), k_highpass=None, r_max=10.0,
         r_step=0.02, density=None, r_cut=0.8),
    # k_min_fit is the next float64 above the k = 0.15 bin, so that bin is
    # outside the fit range in float64 but inside it in float32; the damping
    # corrects it (library float32 comparison) and the window reaches it.
    dict(name="damping_kfit_edge", k_min_fit=float(np.nextafter(qq[3], np.inf)), k_max_fit=1.5,
         k_min_window=0.1, k_max_window=None, k_lowpass=None, k_highpass=None,
         r_max=10.0, r_step=0.02, density=0.05, r_cut=0.8),
]
assert qq[3] < cases[-1]["k_min_fit"] and np.float32(qq[3]) == np.float32(cases[-1]["k_min_fit"])
out = []
for case in cases:
    params = {k: v for k, v in case.items() if k not in ("name", "density")}
    density = case["density"]
    pdf.calculate_Gr(**params, damp_origin_oscillations=density is not None, density=density)
    ref = dict(
        name=case["name"],
        params=case,
        bg=np.asarray(pdf.bg, dtype=float).tolist(),
        fk=np.asarray(pdf.Fk, dtype=float).tolist(),
        fkWin=np.asarray(pdf.Fk_masked, dtype=float).tolist(),
        sk=np.asarray(pdf.Sk, dtype=float).tolist(),
        r=np.asarray(pdf.r, dtype=float).tolist(),
        gr=np.asarray(pdf.reduced_pdf, dtype=float).tolist(),
    )
    if density is not None:
        ref["fkDamped"] = np.asarray(pdf.Fk_damped, dtype=float).tolist()
        ref["grDamped"] = np.asarray(pdf.reduced_pdf_damped, dtype=float).tolist()
        pdf.calculate_gr(density=density, r_cut=case["r_cut"], set_pdf_positive=True)
        ref["gofr"] = np.asarray(pdf.pdf, dtype=float).tolist()
    out.append(ref)

# Degenerate settings the UI can reach: record whether the library raises,
# returns all-NaN curves, or finite ones, and the finite G(r) (the JS must do
# the same). The one-bin window has its edges one float64 ulp outside the
# k = 0.6 / 0.65 bins but on them in float32, so the library's float32 window
# has no positive weight: 0 at k = 0.6 and tiny and negative at k = 0.65.
one_lo, one_hi = float(np.nextafter(qq[12], -np.inf)), float(np.nextafter(qq[13], np.inf))
assert np.float32(one_lo) == np.float32(qq[12]) and np.float32(one_hi) == np.float32(qq[13])
edge_cases = [
    dict(name="window_below_fit_damp", k_min_fit=0.6, k_max_fit=1.2, k_min_window=0.2, k_max_window=0.4,
         density=0.05),
    dict(name="window_empty_damp", k_min_fit=0.3, k_max_fit=0.6, k_min_window=0.8, k_max_window=None,
         density=0.05),
    dict(name="window_one_bin_f32", k_min_fit=0.3, k_max_fit=1.5, k_min_window=one_lo, k_max_window=one_hi,
         density=None),
    dict(name="window_one_bin_f32_damp", k_min_fit=0.3, k_max_fit=1.5, k_min_window=one_lo,
         k_max_window=one_hi, density=0.05),
]
edges = []
for case in edge_cases:
    case = dict(case, k_lowpass=None, k_highpass=None, r_max=10.0, r_step=0.02,
                r_cut=0.8)
    params = {k: v for k, v in case.items() if k not in ("name", "density")}
    density = case["density"]
    try:
        pdf.calculate_Gr(**params, damp_origin_oscillations=density is not None, density=density)
    except Exception:
        edges.append(dict(name=case["name"], params=case, outcome="raises"))
        continue
    g = np.asarray(pdf.reduced_pdf_damped if density is not None else pdf.reduced_pdf, dtype=float)
    outcome = "nan" if np.isnan(g[1:]).all() else "finite" if np.isfinite(g).all() else "mixed"
    edge = dict(name=case["name"], params=case, outcome=outcome)
    if outcome == "finite":
        edge["gr"] = g.tolist()
    edges.append(edge)

# The window calculate_Gr builds (its resolved bounds, the library's _k_window
# on its float32 k), whether it has any positive weight, and the curves.
dk = float(qq[1] - qq[0])
window_cases = [
    dict(name="normal", k_min_fit=0.3, k_max_fit=1.5, k_min_window=0.35, k_max_window=1.4),
    dict(name="follows_fit", k_min_fit=0.3, k_max_fit=1.5, k_min_window=None, k_max_window=None),
    dict(name="follows_axis", k_min_fit=None, k_max_fit=None, k_min_window=None, k_max_window=None),
    dict(name="inverted", k_min_fit=0.3, k_max_fit=1.5, k_min_window=1.0, k_max_window=0.5),
    dict(name="equal", k_min_fit=0.3, k_max_fit=1.5, k_min_window=0.8, k_max_window=0.8),
    dict(name="inverted_by_fit", k_min_fit=0.6, k_max_fit=1.5, k_min_window=None, k_max_window=0.4),
    dict(name="narrower_than_step", k_min_fit=0.3, k_max_fit=1.5, k_min_window=float(qq[12]) + dk / 4,
         k_max_window=float(qq[12]) + dk / 2),
    dict(name="narrower_than_step_on_bin", k_min_fit=0.3, k_max_fit=1.5, k_min_window=float(qq[12]) - dk / 4,
         k_max_window=float(qq[12]) + dk / 4),
    dict(name="one_bin_f32", k_min_fit=0.3, k_max_fit=1.5, k_min_window=one_lo, k_max_window=one_hi),
]
windows = []
for case in window_cases:
    case = dict(case, k_lowpass=None, k_highpass=None, r_max=10.0, r_step=0.02,
                density=None)
    params = {k: v for k, v in case.items() if k not in ("name", "density")}
    pdf.calculate_Gr(**params)
    k32 = torch.from_numpy(qq.astype(np.float32))
    wk = pdf._k_window(k32, pdf.kmin_window, pdf.kmax_window)
    windows.append(dict(name=case["name"], params=case, empty=not bool((wk > 0).any()),
                        wk=wk.numpy().astype(float).tolist(),
                        kmin_window=float(pdf.kmin_window), kmax_window=float(pdf.kmax_window),
                        fkWin=np.asarray(pdf.Fk_masked, dtype=float).tolist(),
                        gr=np.asarray(pdf.reduced_pdf, dtype=float).tolist()))
print(json.dumps({"qq": qq.tolist(), "ik": np.asarray(pdf.Ik, dtype=float).tolist(), "cases": out,
                  "edges": edges, "windows": windows}))
`;
  try {
    const out = execFileSync(python, ["-c", code], {
      encoding: "utf8", maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "pipe", "ignore"],
    });
    return JSON.parse(out);
  } catch {
    return null;
  }
}

// max |a - b| relative to max |b|
function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  expect(a.length).toBe(b.length);
  let scale = 0;
  for (let i = 0; i < b.length; i++) scale = Math.max(scale, Math.abs(b[i]));
  scale = scale || 1;
  let err = 0;
  for (let i = 0; i < b.length; i++) err = Math.max(err, Math.abs(a[i] - b[i]) / scale);
  return err;
}

function grParams(p) {
  return {
    kMinFit: p.k_min_fit ?? 0, kMaxFit: p.k_max_fit ?? 0,
    kMinWindow: p.k_min_window ?? 0, kMaxWindow: p.k_max_window ?? 0,
    kLowpass: p.k_lowpass ?? 0, kHighpass: p.k_highpass ?? 0,
    rMax: p.r_max, rStep: p.r_step,
  };
}

const fixture = pdfReferenceFixture();

(fixture ? describe : describe.skip)("ShowPDF offline PDF compute matches PairDistributionFunction", () => {
  const qq = fixture ? Float64Array.from(fixture.qq) : null;
  const ik = fixture ? Float64Array.from(fixture.ik) : null;

  for (const ref of fixture ? fixture.cases : []) {
    it(`reproduces calculate_Gr (${ref.name})`, () => {
      const res = computeGr(qq, ik, grParams(ref.params));
      // the r grid is float32(i*r_step) in both, so r_cut ties fall the same way
      expect(Array.from(res.r)).toEqual(ref.r);
      // JS works in float64 and the library in float32 (torch), so the curves
      // agree to ~1e-6 of their scale; the bounds leave about a 10x margin.
      expect(relErr(res.bg, ref.bg)).toBeLessThan(1e-6);
      expect(relErr(res.fk, ref.fk)).toBeLessThan(1e-5);
      expect(relErr(res.fkWin, ref.fkWin)).toBeLessThan(1e-5);
      expect(relErr(res.sk, ref.sk)).toBeLessThan(1e-5);
      expect(relErr(res.r, ref.r)).toBeLessThan(1e-6);
      expect(relErr(res.gr, ref.gr)).toBeLessThan(2e-5);
      expect(res.gr[0]).toBe(0);
      if (ref.params.density !== null) {
        const damping = { density: ref.params.density, rCut: ref.params.r_cut };
        const damped = dampOriginOscillations(qq, res, damping);
        expect(relErr(damped.fkWin, ref.fkDamped)).toBeLessThan(2e-5);
        expect(relErr(damped.gr, ref.grDamped)).toBeLessThan(2e-5);
        const gofr = computeGofR(res.r, damped.gr, ref.params.density);
        expect(relErr(gofr, ref.gofr)).toBeLessThan(2e-5);
      }
    });
  }

  for (const ref of fixture ? fixture.edges : []) {
    it(`raises, returns NaN or computes as the library does (${ref.name}: ${ref.outcome})`, () => {
      let outcome, gr;
      try {
        const res = computeGr(qq, ik, grParams(ref.params));
        gr = ref.params.density !== null
          ? dampOriginOscillations(qq, res, { density: ref.params.density, rCut: ref.params.r_cut }).gr
          : res.gr;
        const tail = Array.from(gr).slice(1);
        outcome = tail.every(Number.isNaN) ? "nan" : gr.every(Number.isFinite) ? "finite" : "mixed";
      } catch {
        outcome = "raises";
      }
      expect(outcome).toBe(ref.outcome);
      // A degenerate window is not exactly zero, so compare finite curves too.
      if (ref.gr) expect(relErr(gr, ref.gr)).toBeLessThan(2e-5);
    });
  }

  it("has the empty-window reference cases", () => {
    expect(fixture.windows.map((w) => `${w.name}:${w.empty}`)).toEqual([
      "normal:false", "follows_fit:false", "follows_axis:false", "inverted:true", "equal:true",
      "inverted_by_fit:true", "narrower_than_step:true", "narrower_than_step_on_bin:false",
      "one_bin_f32:true",
    ]);
  });

  for (const ref of fixture ? fixture.windows : []) {
    it(`decides an empty k window as the library's window does (${ref.name}: ${ref.empty})`, () => {
      // kWindow on the bounds computeGr resolves, which are the library's (the
      // exported file's status line resolves them the same way).
      const res = computeGr(qq, ik, grParams(ref.params));
      expect([res.kMinWindow, res.kMaxWindow]).toEqual([ref.kmin_window, ref.kmax_window]);
      const wk = kWindow(qq, res.kMinWindow, res.kMaxWindow);
      // The same zero, negative (a bin at k max) and positive bins as the
      // library's float32 window, and its values to float32 rounding.
      const sign = (v) => (v > 0 ? 1 : v < 0 ? -1 : 0);
      expect(Array.from(wk, sign)).toEqual(ref.wk.map(sign));
      for (let i = 0; i < wk.length; i++) {
        expect(Math.abs(wk[i] - ref.wk[i])).toBeLessThanOrEqual(1e-6 * Math.abs(ref.wk[i]));
      }
      const empty = !wk.some((w) => w > 0);
      expect(empty).toBe(ref.empty);
      // Without positive weight the window can keep a tiny negative one, so the
      // curves are not always zero: compare them with the library's.
      expect(relErr(res.fkWin, ref.fkWin)).toBeLessThan(1e-5);
      expect(relErr(res.gr, ref.gr)).toBeLessThan(2e-5);
      expect(Array.from(res.gr).every((g) => g === 0)).toBe(ref.gr.every((g) => g === 0));
    });
  }

  it("refuses a highpass above the lowpass, as the library does", () => {
    const bandPass = { ...grParams(fixture.cases[0].params), kLowpass: 0.02, kHighpass: 0.05 };
    expect(() => computeGr(qq, ik, bandPass)).toThrow(/highpass/);
  });
});
