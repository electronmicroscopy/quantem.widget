import { describe, expect, it } from "vitest";
import { findDrift } from "./drift";

/** A random specimen of Gaussian blobs (no lattice to mistake), every blob moved by (row, col) px, periodic. */
function specimen(size: number, rowShift: number, colShift: number): Float32Array {
  let state = 7;
  const random = () => ((state = (state * 1664525 + 1013904223) % 4294967296) / 4294967296);
  const blobs = Array.from({ length: 120 }, () => [random() * size, random() * size, 1.5 + random() * 4, 50 + 100 * random()]);
  const out = new Float32Array(size * size);
  for (const [r0, c0, sigma, amplitude] of blobs) {
    for (let row = 0; row < size; row++) {
      const dr0 = (row - r0 - rowShift) % size;
      const dr = Math.min(Math.abs(dr0), size - Math.abs(dr0));
      for (let col = 0; col < size; col++) {
        const dc0 = (col - c0 - colShift) % size;
        const dc = Math.min(Math.abs(dc0), size - Math.abs(dc0));
        out[row * size + col] += amplitude * Math.exp(-(dr * dr + dc * dc) / (2 * sigma * sigma));
      }
    }
  }
  return out;
}

describe("Find drift", () => {
  const size = 128;
  const velocity: [number, number] = [0.37, -0.21];        // px per frame
  const stack = Array.from({ length: 6 }, (_, f) => specimen(size, velocity[0] * f, velocity[1] * f));

  it("measures specimen motion in quantem.gpu's sign and the velocity to a few hundredths of a pixel", async () => {
    const drift = await findDrift((i) => stack[i], [0, 1, 2, 3, 4, 5], size, size, null,
      { pixelSize: 0.5, pixelUnit: "nm", frameSampling: 2, frameUnit: "s" });
    expect(drift.backend).toBe("cpu");
    expect(drift.frames).toEqual([0, 1, 2, 3, 4, 5]);
    expect(drift.shift_px[0]).toEqual([0, 0]);
    // the Hann window and high-pass pull each measured step ~3% toward zero (0.36 for 0.37 px here);
    // the shifts come out within 5% + 0.02 px, the velocity within 5%
    drift.shift_px.forEach(([row, col], f) => {
      expect(Math.abs(row - velocity[0] * f)).toBeLessThan(0.02 + 0.05 * Math.abs(velocity[0] * f));
      expect(Math.abs(col - velocity[1] * f)).toBeLessThan(0.02 + 0.05 * Math.abs(velocity[1] * f));
    });
    expect(Math.abs(drift.velocity_px_per_frame[0] / velocity[0] - 1)).toBeLessThan(0.05);
    expect(Math.abs(drift.velocity_px_per_frame[1] / velocity[1] - 1)).toBeLessThan(0.05);
    // 0.5 nm per px, 2 s per frame
    expect(drift.velocity_unit).toBe("nm/s");
    expect(Math.abs(drift.velocity![0] / (velocity[0] * 0.5 / 2) - 1)).toBeLessThan(0.05);
    expect(drift.frame_interval_s).toBe(2);
  }, 60000);

  it("fits the chosen frames at their own times, and stays in pixels without calibration", async () => {
    const drift = await findDrift((i) => stack[i], [5, 0, 2], size, size, null,
      { pixelSize: 0, pixelUnit: "px", frameSampling: 1, frameUnit: "" });
    expect(drift.frames).toEqual([0, 2, 5]);
    expect(Math.abs(drift.velocity_px_per_frame[0] / velocity[0] - 1)).toBeLessThan(0.05);
    expect(drift.velocity).toBeNull();
    expect(drift.velocity_unit).toBe("px/frame");
  }, 60000);

  it("fits frames taken unevenly against their own times (frame_times_s)", async () => {
    // a steady drift in time, scans taken 23 to 56 s apart as a recorded series was
    const times = [0, 23.4, 56.6, 88.2, 128.1, 184.3];
    const perSecond: [number, number] = [0.012, -0.007];   // px per second
    const uneven = times.map((t) => specimen(size, perSecond[0] * t, perSecond[1] * t));
    const drift = await findDrift((i) => uneven[i], [0, 1, 2, 3, 4, 5], size, size, null,
      { pixelSize: 0.5, pixelUnit: "nm", frameSampling: 1, frameUnit: "", frameTimesS: times });
    expect(drift.times_s).toEqual(times);
    expect(Math.abs(drift.velocity_px_per_s![0] / perSecond[0] - 1)).toBeLessThan(0.05);
    expect(Math.abs(drift.velocity_px_per_s![1] / perSecond[1] - 1)).toBeLessThan(0.05);
    expect(drift.velocity_unit).toBe("nm/s");
    expect(Math.abs(drift.velocity![0] / (perSecond[0] * 0.5) - 1)).toBeLessThan(0.05);
    expect(Math.min(...drift.fit_r2)).toBeGreaterThan(0.99);   // a line in time, though not in frame index
    expect(drift.frame_interval_s).toBeCloseTo(184.3 / 5, 6);
    // picked frames are timed from the first picked one
    const later = await findDrift((i) => uneven[i], [2, 4, 5], size, size, null,
      { pixelSize: 0.5, pixelUnit: "nm", frameSampling: 1, frameUnit: "", frameTimesS: times });
    expect(later.times_s!.map((t) => +t.toFixed(6))).toEqual([0, 71.5, 127.7]);
    await expect(findDrift((i) => uneven[i], [0, 1], size, size, null,
      { pixelSize: 0, pixelUnit: "px", frameSampling: 1, frameUnit: "", frameTimesS: [0, 0, 1, 2, 3, 4] }))
      .rejects.toThrow("must increase");
  }, 60000);

  it("needs two frames", async () => {
    await expect(findDrift((i) => stack[i], [3], size, size, null,
      { pixelSize: 0, pixelUnit: "px", frameSampling: 1, frameUnit: "" })).rejects.toThrow("at least two frames");
  });
});
