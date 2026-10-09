import { describe, expect, it } from "vitest";

import canonicalPoints from "../src/quantem/widget/colormaps.json";
import {
  COLORMAP_NAMES,
  COLORMAP_POINTS,
  COLORMAPS,
  applyColormap,
} from "./display/colormaps";
import {
  applyHannWindow2D,
  computeMagnitude,
  fft2d,
  fftshift,
  findFFTPeak,
  reciprocalCoordinatesFromShiftedOffset,
  shiftedMagnitude,
} from "./display/fft";
import {
  applyLogScale,
  computeStats,
  computeHistogramFromBytes,
  percentileClip,
  signedLog1p,
} from "./display/stats";

describe("display math parity", () => {
  it("uses the exact colormap control points shared with Metal and Python", () => {
    expect(COLORMAP_POINTS).toEqual(canonicalPoints);
    expect(COLORMAP_NAMES).toEqual(Object.keys(canonicalPoints));
    expect(COLORMAP_NAMES).toHaveLength(15);
  });

  it("interpolates every LUT with exact endpoints and byte range", () => {
    for (const name of COLORMAP_NAMES) {
      const points = canonicalPoints[name as keyof typeof canonicalPoints];
      const lut = COLORMAPS[name];
      expect(lut).toHaveLength(256 * 3);
      expect(Array.from(lut.slice(0, 3))).toEqual(points[0]);
      expect(Array.from(lut.slice(-3))).toEqual(points[points.length - 1]);
      expect(Array.from(lut).every(value => value >= 0 && value <= 255)).toBe(true);
    }
  });

  it("uses floor LUT indexing with exact RGBA endpoints", () => {
    const values = new Float32Array([0, 0.25, 0.5, 0.75, 1]);
    const rgba = new Uint8ClampedArray(values.length * 4);
    applyColormap(values, rgba, COLORMAPS.gray, 0, 1);
    expect(Array.from(rgba)).toEqual([
      0, 0, 0, 255,
      63, 63, 63, 255,
      127, 127, 127, 255,
      191, 191, 191, 255,
      255, 255, 255, 255,
    ]);
  });

  it("preserves negative difference-image signal under signed log1p", () => {
    const transformed = applyLogScale(new Float32Array([-7, -3, 0, 3, 7]));
    expect(Array.from(transformed)).toEqual([
      -Math.log1p(7),
      -Math.log1p(3),
      0,
      Math.log1p(3),
      Math.log1p(7),
    ].map(Math.fround));
    expect(signedLog1p(-7)).toBe(-Math.log1p(7));
  });

  it("matches the 256-bin edge convention", () => {
    const values = new Float32Array([0, 0.25, 0.5, 0.75, 1]);
    const bins = computeHistogramFromBytes(values, 256, 0, 1);
    for (const index of [0, 64, 128, 192, 255]) expect(bins[index]).toBe(1);
    expect(bins.reduce((sum, value) => sum + value, 0)).toBe(5);
  });

  it("defines constant and non-finite display behavior", () => {
    const constant = new Float32Array([-7, 0, 7]);
    const bins = computeHistogramFromBytes(constant, 256, 3, 3);
    expect(bins[128]).toBe(1);
    expect(bins.reduce((sum, value) => sum + value, 0)).toBe(1);

    const rgba = new Uint8ClampedArray(constant.length * 4);
    applyColormap(constant, rgba, COLORMAPS.gray, 3, 3);
    expect(Array.from(rgba)).toEqual([
      127, 127, 127, 255,
      127, 127, 127, 255,
      127, 127, 127, 255,
    ]);

    const mixed = new Float32Array([NaN, -Infinity, Infinity, -1, 0, 1]);
    const mixedRgba = new Uint8ClampedArray(mixed.length * 4);
    applyColormap(mixed, mixedRgba, COLORMAPS.gray, -1, 1);
    expect(Array.from(mixedRgba)).toEqual([
      0, 0, 0, 255,
      0, 0, 0, 255,
      255, 255, 255, 255,
      0, 0, 0, 255,
      127, 127, 127, 255,
      255, 255, 255, 255,
    ]);
    expect(computeStats(mixed)).toEqual({ mean: 0, min: -1, max: 1, std: Math.sqrt(2 / 3) });
    expect(percentileClip(mixed, 0, 100)).toEqual({ vmin: -1, vmax: 1, min: -1, max: 1 });
  });

  it("keeps FFT display geometry and magnitude deterministic", () => {
    const shifted = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7]);
    fftshift(shifted, 4, 2);
    expect(Array.from(shifted)).toEqual([6, 7, 4, 5, 2, 3, 0, 1]);

    const magnitude = computeMagnitude(
      new Float32Array([3, 5, 8]),
      new Float32Array([4, 12, 15]),
    );
    expect(Array.from(magnitude)).toEqual([5, 13, 17]);

    const oddFullGrid = shiftedMagnitude(
      Float32Array.from({ length: 15 }, (_, index) => index),
      new Float32Array(15),
      5,
      3,
      true,
    );
    expect(oddFullGrid).toHaveLength(15);
    const sorted = Array.from(oddFullGrid).sort((a, b) => a - b);
    sorted.forEach((value, index) => expect(value).toBeCloseTo(Math.log1p(index), 6));

    const windowed = new Float32Array(25).fill(1);
    applyHannWindow2D(windowed, 5, 5);
    expect(windowed[0]).toBe(0);
    expect(windowed[12]).toBe(1);
    expect(windowed[24]).toBe(0);
  });

  it("matches reciprocal coordinates for axis, anisotropic, fractional, and DC offsets", () => {
    expect(reciprocalCoordinatesFromShiftedOffset(0, 4, 8, 16, 0.5, 0.25)).toEqual({
      rowFrequency: 0,
      columnFrequency: 1,
      spatialFrequency: 1,
      dSpacing: 1,
    });
    const diagonal = reciprocalCoordinatesFromShiftedOffset(1.5, -2, 6, 8, 0.5, 0.25);
    expect(diagonal.rowFrequency).toBe(0.5);
    expect(diagonal.columnFrequency).toBe(-1);
    expect(diagonal.spatialFrequency).toBeCloseTo(Math.sqrt(1.25), 15);
    expect(diagonal.dSpacing).toBeCloseTo(1 / Math.sqrt(1.25), 15);
    expect(reciprocalCoordinatesFromShiftedOffset(0, 0, 7, 9, 1, 2).dSpacing).toBeNull();
  });

  it("pins FFT peak tie, centroid, border, and non-finite behavior", () => {
    const magnitude = new Float32Array(20);
    magnitude[1 * 5 + 2] = 4;
    magnitude[1 * 5 + 3] = 2;
    magnitude[2 * 5 + 2] = 2;
    expect(findFFTPeak(magnitude, 5, 4, 2, 1, 1)).toEqual({ row: 1.25, col: 2.25 });

    const tied = new Float32Array(20);
    tied[0] = 5;
    tied[2] = 5;
    expect(findFFTPeak(tied, 5, 4, 1, 0, 2)).toEqual({ row: 0, col: 0 });

    const nonfinite = new Float32Array(20).fill(NaN);
    nonfinite[0] = Infinity;
    expect(findFFTPeak(nonfinite, 5, 4, 99, -4, 3)).toEqual({ row: 0, col: 4 });
  });

  it("matches a direct 2D DFT and preserves the inverse round trip", () => {
    const width = 4;
    const height = 4;
    const input = Float32Array.from(
      { length: width * height },
      (_, index) => ((index * 7) % 11) - 5,
    );
    const real = input.slice();
    const imag = new Float32Array(input.length);
    fft2d(real, imag, width, height);

    for (let ky = 0; ky < height; ky++) {
      for (let kx = 0; kx < width; kx++) {
        let expectedReal = 0;
        let expectedImag = 0;
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const angle = -2 * Math.PI * (kx * x / width + ky * y / height);
            const value = input[y * width + x];
            expectedReal += value * Math.cos(angle);
            expectedImag += value * Math.sin(angle);
          }
        }
        const index = ky * width + kx;
        expect(real[index]).toBeCloseTo(expectedReal, 4);
        expect(imag[index]).toBeCloseTo(expectedImag, 4);
      }
    }

    fft2d(real, imag, width, height, true);
    for (let index = 0; index < input.length; index++) {
      expect(real[index]).toBeCloseTo(input[index], 5);
      expect(imag[index]).toBeCloseTo(0, 5);
    }
  });
});
