/**
 * ShowPtycho's reconstruction requests to the vendored quantem.gpu WebGPU SSB engine.
 *
 * The engine under `js/.generated/engine` is copied from quantem.gpu and shares the widget's names:
 * it reads `snapshots/cal.json` and the bright-field source exactly as ShowPtycho.export writes them
 * (`num_bright_field_pixels`, `scan_detector_rotation_deg`, `detector_sampling_mrad`, ...), takes the
 * options `numBrightFieldPixels`, `scanDetectorRotationDeg`, `upsamplingFactor`, and reports
 * `numBrightFieldPixels` and `scanDetectorRotationDeg` back. Its arithmetic is in Angstrom and radians,
 * so `reconstruct` converts the aberrations once, here: nm to Angstrom for C10, C12, the higher-order
 * magnitudes and the depth spread, degrees to radians for phi12. `src/quantem/widget/adapters/gpu.py`
 * is its Python counterpart.
 */

import type { WebGPUSSBBackend } from "../.generated/engine/ssb/webgpu/backend";
import type { WebGPUReconstructionOptions } from "../.generated/engine/ssb/webgpu/protocol";

/** The layout of `snapshots/snapshots.json`, written by the exported page. */
export const SNAPSHOTS_SCHEMA = "quantem.ssb-snapshots/2";

// The engine evaluates chi = (pi / lambda[A]) alpha^2 (C10 + ...), so it takes Angstrom where the vocabulary says nm.
const ANGSTROM_PER_NM = 10;

/** Thick-sample model of one reconstruction: sample tilt (scan frame) and depth spread; a depth spread of 0 is standard SSB. */
export interface SampleValues { tiltRowMrad: number; tiltColMrad: number; depthSpreadNm: number }

/** One reconstruction request in the vocabulary's quantities. */
export interface ReconstructionRequest {
  aberrations: { C10_nm: number; C12_nm: number; phi12_deg: number };
  /** Higher orders keyed `C21_nm`, `phi21_deg`, ...; a missing key is 0. */
  higherOrderAberrations: Record<string, number>;
  scanDetectorRotationDeg: number;
  upsamplingFactor: number;
  numBrightFieldPixels: number;
  /** True while fewer than all bright-field pixels are used (a drag). */
  reducedBrightField: boolean;
  computeLoss: boolean;
  /** Native diagnostic only: the caller keeps its already displayed phase. */
  diagnosticOnly?: boolean;
  signal?: AbortSignal;
  sample: SampleValues | null;
}

/** The magnitude key of a higher-order term: `C21` -> `C21_nm`. */
export function higherOrderMagnitudeKey(name: string): string {
  return `${name}_nm`;
}

/** The azimuth key of a higher-order term: `C21` -> `phi21_deg`. */
export function higherOrderAzimuthKey(name: string): string {
  return `phi${name.slice(1)}_deg`;
}

/** Run one reconstruction on the engine, converting nm to Angstrom and degrees to radians at this boundary. */
export function reconstruct(engine: WebGPUSSBBackend, request: ReconstructionRequest) {
  const higherOrder: Record<string, number> = {};
  for (const [key, value] of Object.entries(request.higherOrderAberrations)) {
    // a magnitude goes in Angstrom (C21_nm -> C21_A); an azimuth keeps its name and its degrees
    if (key.endsWith("_nm")) higherOrder[`${key.slice(0, -3)}_A`] = value * ANGSTROM_PER_NM;
    else higherOrder[key] = value;
  }
  const sample = request.sample && request.sample.depthSpreadNm > 0
    ? { sampleTiltMrad: [request.sample.tiltRowMrad, request.sample.tiltColMrad] as const, depthSpreadA: request.sample.depthSpreadNm * ANGSTROM_PER_NM }
    : undefined;
  const options: WebGPUReconstructionOptions = {
    preview: request.reducedBrightField, numBrightFieldPixels: request.numBrightFieldPixels, computeLoss: request.computeLoss,
    scanDetectorRotationDeg: request.scanDetectorRotationDeg,
    // the engine itself rejects a factor other than 1, 2, 4 or 8, so the narrowing cast is safe
    upsamplingFactor: request.upsamplingFactor as WebGPUReconstructionOptions["upsamplingFactor"],
    higherOrder, ...(sample ? { sample } : {}),
    ...(request.diagnosticOnly ? { phaseEstimator: "mean_phase" as const, upsamplingFactor: 1 as const } : {}),
    ...(request.signal ? { signal: request.signal } : {}),
  };
  return engine.reconstruct(
    request.aberrations.C10_nm * ANGSTROM_PER_NM, request.aberrations.C12_nm * ANGSTROM_PER_NM,
    request.aberrations.phi12_deg * Math.PI / 180, options,
  );
}
