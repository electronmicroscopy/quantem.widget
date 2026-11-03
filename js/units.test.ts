import { describe, expect, it } from "vitest";
import { degreesWithin, wrapDegrees } from "./units";

describe("angles written to files", () => {
  it("wraps a slider value below 0 or at and past 360 into [0, 360)", () => {
    expect(wrapDegrees(-10)).toBe(350);
    expect(wrapDegrees(-180)).toBe(180);
    expect(wrapDegrees(360)).toBe(0);
    expect(wrapDegrees(725.5)).toBeCloseTo(5.5, 12);
    expect(wrapDegrees(-725.5)).toBeCloseTo(354.5, 12);
    expect(wrapDegrees(42.5)).toBe(42.5);
    // a slider value already in the range is written as it is, to the last bit
    for (const inRange of [0, 0.1, 21.3, 165.5, 359.9]) expect(wrapDegrees(inRange)).toBe(inRange);
    expect(wrapDegrees(-1e-15)).toBe(0);
    expect(wrapDegrees(-1e-15)).toBeGreaterThanOrEqual(0);
    expect(wrapDegrees(-1e-15)).toBeLessThan(360);
  });

  it("reads a stored angle back onto a slider whose range is not [0, 360)", () => {
    // every value of a [-180, 180] slider survives the wrap and the read
    for (const slider of [-180, -90.5, -10, 0, 10, 179.9, 180]) {
      expect(degreesWithin(wrapDegrees(slider), -180, 180)).toBeCloseTo(slider === -180 ? 180 : slider, 12);
    }
    expect(degreesWithin(200, -180, 270)).toBe(200);
    // an angle already inside the range stays where it is (0 on a [0, 360] slider, not 360)
    expect(degreesWithin(0, 0, 360)).toBe(0);
    expect(degreesWithin(0, 200, 500)).toBe(360);
    // no turn of 100 lies in [-180, 50]: the stored angle is kept
    expect(degreesWithin(100, -180, 50)).toBe(100);
  });
});
