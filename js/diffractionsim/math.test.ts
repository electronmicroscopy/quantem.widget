// @vitest-environment node
import { describe, expect, it } from "vitest";

import {
  Vec3,
  directionIndices,
  eighComplex,
  fourToThree,
  matTVec,
  matVec,
  parseDirection,
  quatFromZoneAxis,
  quatToMatrix,
  threeToFour,
} from "./math";
import { normalizeMode } from "./mode";
import { colormapLut } from "./pattern";
import { electronWavelength, parseCrystal, relativisticGamma } from "./physics";
import { PRESETS } from "../diffractionsim-web/presets";

const CUBIC: number[][] = [[5.431, 0, 0], [0, 5.431, 0], [0, 0, 5.431]];

function expectClose(a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-9) {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(tol);
}

describe("diffsim orientation math", () => {
  it("maps the requested zone axis onto the beam (z)", () => {
    for (const uvw of [[1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, -1], [1, 2, 3]] as Vec3[]) {
      const d = uvw;
      const n = Math.hypot(...d);
      const R = quatToMatrix(quatFromZoneAxis(d));
      expectClose(matVec(R, [d[0] / n, d[1] / n, d[2] / n]), [0, 0, 1]);
      expect(directionIndices(CUBIC, matTVec(R, [0, 0, 1]))).toEqual(uvw);
    }
  });

  it("builds orthonormal rotation matrices", () => {
    const R = quatToMatrix(quatFromZoneAxis([1, 2, 3], 30));
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const dot = R[3 * i] * R[3 * j] + R[3 * i + 1] * R[3 * j + 1] + R[3 * i + 2] * R[3 * j + 2];
        expect(Math.abs(dot - (i === j ? 1 : 0))).toBeLessThan(1e-12);
      }
    }
  });

  it("converts between three- and four-index hexagonal directions", () => {
    expect(fourToThree([0, 0, 0, 1])).toEqual([0, 0, 1]);
    expect(fourToThree([2, -1, -1, 0])).toEqual([3, 0, 0]);
    expect(threeToFour([1, 0, 0])).toEqual([2, -1, -1, 0]);
    expect(threeToFour([0, 0, 1]).map((x) => x + 0)).toEqual([0, 0, 0, 1]); // t comes out as -0
  });

  it("parses zone-axis text", () => {
    expect(parseDirection("1 1 0")).toEqual([1, 1, 0]);
    expect(parseDirection("[1-10]")).toEqual([1, -1, 0]);
    expect(parseDirection("0001")).toEqual([0, 0, 1]);
    expect(parseDirection("1 0 0 1")).toBeNull(); // u + v + t must vanish
    expect(parseDirection("0 0 0")).toBeNull();
  });
});

describe("diffsim Hermitian eigensolver", () => {
  it("returns eigenpairs of a complex Hermitian matrix", () => {
    const n = 3;
    const re = Float64Array.from([2, 1, 0, 1, 3, 0.5, 0, 0.5, 1]);
    const im = Float64Array.from([0, 0.5, -0.2, -0.5, 0, 0.3, 0.2, -0.3, 0]);
    const { vals, vecRe, vecIm } = eighComplex(re, im, n);
    let trace = 0;
    for (let j = 0; j < n; j++) {
      trace += vals[j];
      for (let i = 0; i < n; i++) {
        // (A v)_i = sum_k A_ik v_k, complex
        let sr = 0, si = 0;
        for (let k = 0; k < n; k++) {
          const ar = re[i * n + k], ai = im[i * n + k];
          const vr = vecRe[k * n + j], vi = vecIm[k * n + j];
          sr += ar * vr - ai * vi;
          si += ar * vi + ai * vr;
        }
        expect(Math.abs(sr - vals[j] * vecRe[i * n + j])).toBeLessThan(1e-9);
        expect(Math.abs(si - vals[j] * vecIm[i * n + j])).toBeLessThan(1e-9);
      }
    }
    expect(Math.abs(trace - 6)).toBeLessThan(1e-9);
  });
});

describe("diffsim physics constants", () => {
  it("matches the relativistic wavelength used in Python at 200 keV", () => {
    // quantem electron_wavelength_angstrom(200e3) = 0.025079337... A
    expect(Math.abs(electronWavelength(200e3) - 0.0250793)).toBeLessThan(1e-6);
    expect(Math.abs(relativisticGamma(200e3) - 1.391390)).toBeLessThan(1e-5);
  });

  it("decodes the embedded silicon preset", () => {
    const si = parseCrystal(PRESETS["Si (diamond cubic)"]);
    expect(si).not.toBeNull();
    expect(si!.hkl.length).toBe(si!.F2.length);
    expect(si!.U_re.length).toBe(si!.F2.length);
    expect(si!.k_max).toBe(3);
    expect(si!.absorptive).toBe(true);
    // diamond glide: 200 is kinematically forbidden, 111 is not
    const f2 = (h: number[]) => si!.F2[si!.hkl.findIndex((x) => x.join() === h.join())];
    expect(f2([2, 0, 0])).toBe(0);
    expect(f2([1, 1, 1])).toBeGreaterThan(0);
  });
});

describe("diffsim modes and colormaps", () => {
  it("accepts kossel as the earlier name of kikuchi", () => {
    expect(normalizeMode("kossel")).toBe("kikuchi");
    expect(normalizeMode("Kikuchi")).toBe("kikuchi");
    expect(normalizeMode("cbed")).toBe("cbed");
    expect(normalizeMode("unknown", "cbed")).toBe("cbed");
    expect(normalizeMode(undefined)).toBe("nanobeam");
  });

  it("reverses a shared colormap for the _r suffix", () => {
    const gray = colormapLut("gray");
    const grayR = colormapLut("gray_r");
    expect(Array.from(grayR.subarray(0, 3))).toEqual(Array.from(gray.subarray(765, 768)));
    expect(Array.from(grayR.subarray(765, 768))).toEqual(Array.from(gray.subarray(0, 3)));
  });
});
