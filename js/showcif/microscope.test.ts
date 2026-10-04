import { describe, expect, it } from "vitest";
import { calibratedFov, scaleBarWidth } from "./geometry";

describe("microscope physical field", () => {
  it("uses supplied fixed-camera calibration, not an assumed microscope scale", () => {
    expect(calibratedFov(2e6, [1e6, 100])).toBe(50);
    expect(calibratedFov(0.5e6, [1e6, 100])).toBe(200);
    for (const calibration of [[], [1e6], [1e6, 0], [NaN, 100]])
      expect(() => calibratedFov(1e6, calibration)).toThrow();
    expect(() => calibratedFov(0, [1e6, 100])).toThrow();
  });
  it("keeps scale bars within one quarter of the physical field", () => {
    expect(scaleBarWidth(16)).toBe(2);
    expect(scaleBarWidth(40)).toBe(10);
    for (const span of [0.1, 1, 7, 16, 40, 128, 10000]) {
      expect(scaleBarWidth(span)).toBeGreaterThan(0);
      expect(scaleBarWidth(span)).toBeLessThanOrEqual(span / 4);
    }
  });
});
