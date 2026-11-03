import { describe, expect, it } from "vitest";

import type { WebGPUSSBBackend } from "../.generated/engine/ssb/webgpu/backend";
import { higherOrderAzimuthKey, higherOrderMagnitudeKey, reconstruct } from "./engineAdapter";

describe("the units at the engine boundary", () => {
  it("computes only the native diagnostic while retaining the displayed Auto phase", async () => {
    const calls: unknown[][] = [];
    const engine = { reconstruct: (...args: unknown[]) => { calls.push(args); return Promise.resolve(null); } } as unknown as WebGPUSSBBackend;
    const controller = new AbortController();
    await reconstruct(engine, {
      aberrations: { C10_nm: 3, C12_nm: 1, phi12_deg: 90 }, higherOrderAberrations: {},
      scanDetectorRotationDeg: 10, upsamplingFactor: 4, numBrightFieldPixels: 9000,
      reducedBrightField: false, computeLoss: true, diagnosticOnly: true, signal: controller.signal,
      sample: { tiltRowMrad: 3, tiltColMrad: -4, depthSpreadNm: 12 },
    });
    expect(calls[0][3]).toMatchObject({ upsamplingFactor: 1, phaseEstimator: "mean_phase",
      numBrightFieldPixels: 9000, computeLoss: true, signal: controller.signal,
      sample: { sampleTiltMrad: [3, -4], depthSpreadA: 120 } });
  });
  it("converts nm to Angstrom and degrees to radians once, at the call into the engine", async () => {
    const calls: unknown[][] = [];
    const engine = { reconstruct: (...args: unknown[]) => { calls.push(args); return Promise.resolve(null); } } as unknown as WebGPUSSBBackend;
    await reconstruct(engine, {
      aberrations: { C10_nm: 3, C12_nm: 1, phi12_deg: 90 },
      higherOrderAberrations: { [higherOrderMagnitudeKey("C21")]: 5, [higherOrderAzimuthKey("C21")]: 45, [higherOrderMagnitudeKey("C30")]: 2 },
      scanDetectorRotationDeg: 10, upsamplingFactor: 2, numBrightFieldPixels: 7, reducedBrightField: true, computeLoss: false,
      sample: { tiltRowMrad: 3, tiltColMrad: -4, depthSpreadNm: 12 },
    });
    const [c10, c12, phi12, options] = calls[0] as [number, number, number, Record<string, unknown>];
    expect([c10, c12]).toEqual([30, 10]);
    expect(phi12).toBeCloseTo(Math.PI / 2, 12);
    expect(options).toEqual({
      preview: true, numBrightFieldPixels: 7, computeLoss: false, scanDetectorRotationDeg: 10, upsamplingFactor: 2,
      // magnitudes in Angstrom, azimuths in degrees under the same names
      higherOrder: { C21_A: 50, phi21_deg: 45, C30_A: 20 },
      // the depth spread of the model in Angstrom, the tilt as [row, col]
      sample: { sampleTiltMrad: [3, -4], depthSpreadA: 120 },
    });
  });

  it("sends no sample for standard SSB (depth spread 0)", async () => {
    const calls: unknown[][] = [];
    const engine = { reconstruct: (...args: unknown[]) => { calls.push(args); return Promise.resolve(null); } } as unknown as WebGPUSSBBackend;
    await reconstruct(engine, {
      aberrations: { C10_nm: 0, C12_nm: 0, phi12_deg: 0 }, higherOrderAberrations: {}, scanDetectorRotationDeg: 0,
      upsamplingFactor: 1, numBrightFieldPixels: 7, reducedBrightField: false, computeLoss: true,
      sample: { tiltRowMrad: 3, tiltColMrad: -4, depthSpreadNm: 0 },
    });
    expect((calls[0][3] as Record<string, unknown>).sample).toBeUndefined();
  });
});
