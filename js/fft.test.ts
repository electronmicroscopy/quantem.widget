import { describe, expect, it } from "vitest";

import { autoEnhanceFFT } from "./fft";

describe("FFT automatic display range", () => {
  it("resolves the useful percentile when a DC outlier spans many orders of magnitude", () => {
    const width = 128;
    const height = 128;
    const magnitude = new Float32Array(width * height);
    for (let index = 0; index < magnitude.length; index++) {
      magnitude[index] = 100 + (index % 901);
    }
    const center = (height / 2) * width + width / 2;
    magnitude[center] = 1e12;
    magnitude[center - 1] = 1e9;

    const range = autoEnhanceFFT(magnitude, width, height);

    expect(range.min).toBe(100);
    expect(range.max).toBeGreaterThan(900);
    expect(range.max).toBeLessThan(10_000);
    const neighborMean = (
      magnitude[center - 1]
      + magnitude[center + 1]
      + magnitude[center - width]
      + magnitude[center + width]
    ) / 4;
    expect(Math.abs(magnitude[center] - neighborMean)).toBeLessThan(16);
  });

  it("returns a stable range for an empty or flat spectrum", () => {
    expect(autoEnhanceFFT(new Float32Array(), 0, 0)).toEqual({ min: 0, max: 0 });
    expect(autoEnhanceFFT(new Float32Array(16).fill(7), 4, 4)).toEqual({ min: 7, max: 7 });
  });

  it("rejects several extreme DC-neighborhood outliers without hiding useful peaks", () => {
    const width = 256;
    const height = 256;
    const magnitude = new Float32Array(width * height);
    for (let index = 0; index < magnitude.length; index++) {
      magnitude[index] = 1_000 + (index % 40_001);
    }
    const center = (height / 2) * width + width / 2;
    for (const offset of [-width, -1, 0, 1, width]) {
      magnitude[center + offset] = 1e12;
    }

    const range = autoEnhanceFFT(magnitude, width, height);

    expect(Number.isFinite(range.min)).toBe(true);
    expect(Number.isFinite(range.max)).toBe(true);
    expect(range.max).toBeGreaterThan(35_000);
    expect(range.max).toBeLessThan(1_000_000);
  });

  it("changes only the display DC pixel while finding the automatic range", () => {
    const width = 32;
    const height = 32;
    const magnitude = Float32Array.from(
      { length: width * height },
      (_, index) => 10 + (index % 101),
    );
    const original = magnitude.slice();
    const center = (height / 2) * width + width / 2;
    magnitude[center] = 1e8;
    original[center] = 1e8;

    autoEnhanceFFT(magnitude, width, height);

    for (let index = 0; index < magnitude.length; index++) {
      if (index === center) continue;
      expect(magnitude[index]).toBe(original[index]);
    }
    expect(magnitude[center]).not.toBe(original[center]);
  });
});
