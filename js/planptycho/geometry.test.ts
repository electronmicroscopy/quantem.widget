import { describe, expect, it } from "vitest";

import goldens from "./goldens.json";
import {
  beamDiameterA, checkRows, checkStatuses, detectorSamplingMrad, planGeometry, probeIntensity, recommendedSettings, wavelengthA,
  type DetectorPresets, type PlanSettings,
} from "./geometry";

describe("PlanPtycho matches the Python goldens", () => {
  for (const [name, testCase] of Object.entries(goldens.cases)) {
    it(`geometry, grades and check text: ${name}`, () => {
      const settings = testCase.settings as PlanSettings;
      const geometry = planGeometry(settings);
      for (const [key, expected] of Object.entries(testCase.geometry)) {
        const actual = geometry[key as keyof typeof geometry];
        if (expected === null) expect(actual).toBeNull();
        else expect(actual as number).toBeCloseTo(expected as number, 9);
      }
      expect(checkStatuses(geometry)).toEqual(testCase.statuses);
      expect(checkRows(geometry, settings.detector_px, settings.scan_step_A, goldens.column_phase_rad_per_A)).toEqual(testCase.rows);
    });
  }

  it("thickness recommendations", () => {
    for (const r of goldens.recommended) {
      expect(recommendedSettings(r.thickness_nm, r.voltage_kV, r.semiangle_mrad, r.scan_step_A)).toEqual(r.expected);
    }
  });

  it("detector calibration", () => {
    for (const r of goldens.detector_sampling) {
      expect(detectorSamplingMrad(goldens.presets as DetectorPresets, r.detector, r.camera_length_mm)).toBeCloseTo(r.expected, 12);
    }
    expect(detectorSamplingMrad(undefined, "Arina", 91)).toBeNull();
    expect(detectorSamplingMrad({ detectors: {}, arina_mrad_per_px: {}, arina_mrad_mm: 50 }, "Arina", 91)).toBeNull();
  });
});

describe("probe on the model window", () => {
  const lam = wavelengthA(300);
  const window = lam / 0.554e-3;

  // diameter holding half the intensity (encircled energy): the standard probe size; an rms width is dominated by the Airy tails
  const halfEnergyDiameter = (defocus: number) => {
    const { intensity, n } = probeIntensity(window, 30, lam, defocus, 192);
    const pairs: [number, number][] = [];
    let total = 0;
    for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) {
      const value = intensity[row * n + col];
      pairs.push([Math.hypot(row - n / 2, col - n / 2), value]); total += value;
    }
    pairs.sort((a, b) => a[0] - b[0]);
    let running = 0;
    for (const [radius, value] of pairs) { running += value; if (running >= total / 2) return 2 * radius * (window / n); }
    return Infinity;
  };

  it("is narrowest at the focus and spreads like the cone", () => {
    // at the focus: Airy disk, half the intensity within ~0.5 lambda / alpha of the centre (0.66 A here)
    expect(halfEnergyDiameter(0)).toBeLessThan(1.2);
    // 100 A from the focus the cone is 2 alpha dz = 6 A across; half of a uniform disk lies within 1/sqrt(2) of its radius
    expect(halfEnergyDiameter(100)).toBeGreaterThan(0.75 * 6 / Math.SQRT2);
    expect(halfEnergyDiameter(100)).toBeLessThan(1.25 * 6 / Math.SQRT2);
  });

  it("keeps the total intensity when it propagates", () => {
    const sum = (a: Float32Array) => a.reduce((acc, v) => acc + v, 0);
    expect(sum(probeIntensity(window, 30, lam, 400, 192).intensity) / sum(probeIntensity(window, 30, lam, 0, 192).intensity)).toBeCloseTo(1, 4);
  });

  it("grows the grid to hold the aperture and cuts what the detector does not record", () => {
    // 60 mrad at 0.269 mrad/px: aperture radius 223 px, beyond a 256 grid
    expect(probeIntensity(lam / 0.269e-3, 60, lam, 0, 192).n).toBe(512);
    // a 64 px detector at 0.554 mrad/px reaches 17.7 mrad: a 30 mrad aperture is cut to the detector square, so less passes
    // Parseval: n^2 x summed intensity counts the Fourier pixels that pass, whatever the grid size
    const passed = (detector: number) => { const { intensity, n } = probeIntensity(window, 30, lam, 0, detector); return n * n * intensity.reduce((acc, v) => acc + v, 0); };
    expect(passed(64) / passed(192)).toBeGreaterThan(0.95 * (64 * 64) / (Math.PI * 54.2 ** 2));
    expect(passed(64) / passed(192)).toBeLessThan(1.05 * (64 * 64) / (Math.PI * 54.2 ** 2));
  });

  it("beam diameter is the cone plus the Airy disk", () => {
    expect(beamDiameterA(1100, 170, 30, lam)).toBeCloseTo(2 * 0.03 * 930 + (1.22 * lam) / 0.03, 9);
  });
});
