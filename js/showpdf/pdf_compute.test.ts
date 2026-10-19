import { describe, expect, it } from "vitest";
import {
  computeGofR, computeGr, dampOriginOscillations, kWindow, resolveWindowBounds, type GrParams,
} from "./pdf_compute";

// Synthetic calibrated I(k): a decaying background plus damped oscillations.
const N = 120;
const qq = Float64Array.from({ length: N }, (_, i) => i * 0.02);
const ik = Float64Array.from(qq, (k) => 50 * Math.exp(-(k * k) / 0.08) + 5 * Math.exp(-k / 0.6) + 0.3
  + 0.4 * Math.exp(-k / 1.5) * Math.sin(2 * Math.PI * 2.6 * k));
const base: GrParams = {
  kMinFit: 0.2, kMaxFit: 2.2, kMinWindow: 0, kMaxWindow: 0,
  kLowpass: 0, kHighpass: 0, rMax: 10, rStep: 0.05,
};

describe("ShowPDF offline PDF compute", () => {
  it("refuses inputs the library refuses", () => {
    expect(() => computeGr(qq, ik, { ...base, kLowpass: 0.02, kHighpass: 0.05 })).toThrow(/highpass/);
    expect(() => computeGr(qq, ik, { ...base, kMinFit: 5, kMaxFit: 6 })).toThrow(/fit range/);
    expect(() => computeGr(qq, ik, { ...base, rMax: 0 })).toThrow(/r grid/);
  });

  it("builds the ramped Lorch window of _k_window", () => {
    const kmin = 0.5, kmax = 1.5, e = 0.1 * (kmax - kmin);   // sin^2 ramp over 10% of the band
    const w = kWindow(qq, kmin, kmax);
    for (let i = 0; i < N; i++) {
      const k = qq[i];
      if (k < kmin || k > kmax) { expect(w[i]).toBe(0); continue; }
      const ramp = k < kmin + e ? Math.sin(0.5 * Math.PI * (k - kmin) / e) ** 2 : 1;
      const lorch = Math.sin(Math.PI * k / kmax) / (Math.PI * k / kmax);
      // evaluated in float32 like the library, so ~1e-7
      expect(w[i]).toBeCloseTo(ramp * lorch, 6);
      if (k > kmin && k < kmax) expect(w[i]).toBeGreaterThan(0);
    }
    // k = 0.5 = kmin: the ramp starts at 0. k = 1.5 = kmax: sin(float32(pi)) < 0
    // leaves a tiny negative weight, as in the library.
    expect(w[25]).toBe(0);
    expect(w[75]).toBeLessThan(0);
    expect(w[75]).toBeGreaterThan(-1e-7);
    // Equal ends on a bin: only that bin is in band, with no positive weight.
    const one = kWindow(qq, qq[40], qq[40]);
    expect(Array.from(one).filter((v) => v !== 0)).toEqual([one[40]]);
    expect(one[40]).toBeLessThan(0);
  });

  it("follows calculate_Gr: window defaults, r grid, G(0) = 0", () => {
    const res = computeGr(qq, ik, base);
    expect(res.kMinWindow).toBe(0.2);
    expect(res.kMaxWindow).toBe(2.2);
    const wk = kWindow(qq, 0.2, 2.2);
    for (let i = 0; i < N; i++) expect(res.fkWin[i]).toBeCloseTo(res.fk[i] * wk[i], 12);
    expect(res.r.length).toBe(Math.ceil(base.rMax / base.rStep));
    expect(res.r[0]).toBe(0);
    expect(res.gr[0]).toBe(0);
    expect(res.gr.every(Number.isFinite)).toBe(true);
    // Unset fit range: the data ends, rounded to float32 as the library does
    const def = computeGr(qq, ik, { ...base, kMinFit: 0, kMaxFit: 0 });
    expect(def.kMaxFit).toBe(Math.fround(qq[N - 1]));
    expect(def.kMaxWindow).toBe(def.kMaxFit);
  });

  it("resolves the window bounds as computeGr uses them", () => {
    // 0 fit ends: the float32 k axis ends; 0 window ends: the fit ends
    const axisEnds = { kMinFit: Math.fround(qq[0]), kMaxFit: Math.fround(qq[N - 1]) };
    expect(resolveWindowBounds(qq, { kMinFit: 0, kMaxFit: 0, kMinWindow: 0, kMaxWindow: 0 }))
      .toEqual({ ...axisEnds, kMinWindow: axisEnds.kMinFit, kMaxWindow: axisEnds.kMaxFit });
    expect(resolveWindowBounds(qq, { kMinFit: 0.6, kMaxFit: 0, kMinWindow: 0, kMaxWindow: 0.4 }))
      .toEqual({ kMinFit: 0.6, kMaxFit: axisEnds.kMaxFit, kMinWindow: 0.6, kMaxWindow: 0.4 });
    const settings = [
      base, { ...base, kMinFit: 0, kMaxFit: 0 }, { ...base, kMinWindow: 0.3, kMaxWindow: 1.9 },
    ];
    for (const p of settings) {
      const { kMinFit, kMaxFit, kMinWindow, kMaxWindow } = computeGr(qq, ik, p);
      expect(resolveWindowBounds(qq, p)).toEqual({ kMinFit, kMaxFit, kMinWindow, kMaxWindow });
    }
  });

  it("damping without iterations reproduces the undamped transform", () => {
    const res = computeGr(qq, ik, base);
    const none = dampOriginOscillations(qq, res, { density: 0.05, rCut: 0.8, maxIter: 0 });
    for (let i = 0; i < res.gr.length; i++) expect(none.gr[i]).toBeCloseTo(res.gr[i], 9);
    const damped = dampOriginOscillations(qq, res, { density: 0.05, rCut: 0.8 });
    expect(damped.gr.length).toBe(res.gr.length);
    expect(damped.fkWin.length).toBe(N);
    expect(damped.gr.every(Number.isFinite)).toBe(true);
    expect(damped.gr[0]).toBe(0);
  });

  it("computes g(r) = max(0, 1 + G/(4 pi r rho0)) with g(0) = 0", () => {
    const r = Float64Array.from([0, 1, 2]);
    const gr = Float64Array.from([5, -4 * Math.PI * 0.05 * 3, 4 * Math.PI * 0.05 * 2]);
    expect(Array.from(computeGofR(r, gr, 0.05))).toEqual([0, 0, 2]);
  });
});
