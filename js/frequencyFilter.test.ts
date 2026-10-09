import { describe, expect, it } from "vitest";
import {
  applyFrequencyFilterCPU,
  applyFrequencyFilterBrowser,
  frequencyMaskValue,
  getFrequencyFilterBackend,
  normalizeFrequencyFilterMode,
} from "./display/frequencyFilter";

describe("frequency filter", () => {
  it("normalizes scientist-facing mode spellings", () => {
    expect(normalizeFrequencyFilterMode("Low-pass")).toBe("lowpass");
    expect(normalizeFrequencyFilterMode("HIGH PASS")).toBe("highpass");
    expect(normalizeFrequencyFilterMode("band_pass")).toBe("bandpass");
  });

  it("builds smooth goal-oriented masks", () => {
    expect(frequencyMaskValue(0, { mode: "lowpass", cutoff: 0.2 })).toBeGreaterThan(0.9);
    expect(frequencyMaskValue(0.8, { mode: "lowpass", cutoff: 0.2 })).toBeLessThan(0.1);
    expect(frequencyMaskValue(0, { mode: "highpass", cutoff: 0.2 })).toBeLessThan(0.1);
    expect(frequencyMaskValue(0.5, { mode: "bandpass", center: 0.5, width: 0.2 })).toBeGreaterThan(0.8);
  });

  it("never mutates source data and None restores it exactly", () => {
    const source = Float32Array.from({ length: 64 }, (_, i) => Math.sin(i));
    const before = Float32Array.from(source);
    const result = applyFrequencyFilterCPU(source, 8, 8, { mode: "none" });
    expect(Array.from(source)).toEqual(Array.from(before));
    expect(Array.from(result)).toEqual(Array.from(before));
  });

  it("high-pass removes a constant background", () => {
    const source = new Float32Array(64).fill(7);
    const result = applyFrequencyFilterCPU(source, 8, 8, { mode: "highpass", cutoff: 0.15 });
    expect(Math.max(...Array.from(result).map(Math.abs))).toBeLessThan(0.1);
  });

  it("runs the scalar reference when the browser has no WebGPU", async () => {
    const image = Float32Array.from({ length: 15 }, (_, index) => 2 + (index % 4));
    const options = { mode: "highpass", cutoff: 0.15 };
    const browser = await applyFrequencyFilterBrowser(image, 5, 3, options);
    expect(Array.from(browser)).toEqual(Array.from(applyFrequencyFilterCPU(image, 5, 3, options)));
    expect(getFrequencyFilterBackend()).toBe("CPU");
  });
});
