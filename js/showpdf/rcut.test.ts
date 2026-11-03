import { describe, expect, it } from "vitest";
import { rCutSlider } from "./rcut";

describe("ShowPDF r cut slider", () => {
  it("steps on the default r grid (r_step 0.02) from 0.02 to 5", () => {
    expect(rCutSlider(0.02, 20)).toEqual({ min: 0.02, max: 5, step: 0.02, decimals: 2 });
    // r max below 5 Å ends the range at the grid point at or below it
    expect(rCutSlider(0.02, 3.01)).toEqual({ min: 0.02, max: 3, step: 0.02, decimals: 2 });
  });

  it("keeps grid points and excludes r = 0", () => {
    expect(rCutSlider(0.25, 20)).toMatchObject({ min: 0.25, max: 5, decimals: 2 });
    // 5 Å is not on the grid: the range ends at the grid point below it
    expect(rCutSlider(0.3, 20)).toMatchObject({ min: 0.3, max: 4.8, decimals: 1 });
    expect(rCutSlider(0.1, 20)).toEqual({ min: 0.1, max: 5, step: 0.1, decimals: 1 });
    expect(rCutSlider(0.0125, 4)).toMatchObject({ min: 0.0125, max: 4, decimals: 4 });
  });

  it("gives every r bin up to 5 Å as a slider value", () => {
    const { min, max, step } = rCutSlider(0.02, 20);
    expect(Math.round((max - min) / step) + 1).toBe(250);
  });
});
