import { describe, expect, it } from "vitest";
import { CPUFFT } from "../display/fft";
import { subpixelShiftEstimator } from "./registration";

/** Two Gaussian blobs and a weak lattice, displaced by (rowShift, colShift) pixels. */
function blobs(width: number, height: number, rowShift: number, colShift: number): Float32Array {
  return Float32Array.from({ length: width * height }, (_, index) => {
    const row = Math.floor(index / width) - rowShift;
    const col = (index % width) - colShift;
    return 100 * Math.exp(-((row - height * 0.4) ** 2 + (col - width * 0.55) ** 2) / 40)
      + 60 * Math.exp(-((row - height * 0.7) ** 2 + (col - width * 0.3) ** 2) / 15)
      + 20 * Math.sin(col * 0.7) * Math.cos(row * 0.45);
  });
}

describe("sub-pixel registration", () => {
  // power-of-two, non-square and padded (non-power-of-two) grids
  it.each([[128, 64], [64, 128], [100, 80]])("recovers integer and fractional drift on %ix%i", async (width, height) => {
    const estimator = subpixelShiftEstimator(blobs(width, height, 0, 0), width, height, new CPUFFT());
    for (const [rowShift, colShift] of [[3, -2], [-5, 0], [0, 4], [2.5, -1.5]]) {
      const shift = await estimator.estimate(blobs(width, height, rowShift, colShift));
      // the returned shift moves the frame back onto the reference
      expect(Math.abs(shift.row + rowShift)).toBeLessThan(0.35);
      expect(Math.abs(shift.col + colShift)).toBeLessThan(0.35);
    }
  });
});
