/**
 * Find drift: how far the specimen moved between the chosen frames of a stack, and how fast.
 *
 * The registration is Show3DSlices' (sliceAlignment.ts): each frame against the next after a
 * high-pass blur and a Hann window, FFT, cross-power, inverse FFT and peak search on WebGPU, then
 * a separable upsampled DFT around the peak (Guizar-Sicairos) to 1/DRIFT_UPSAMPLE px. On a
 * hardware WebGPU device the frames cross the bus once and only the peaks come back; without one
 * the same steps run on the CPU.
 *
 * The shifts follow quantem.gpu's convention: frame f sees the specimen moved by d_f from the first
 * chosen frame, I_f(r) = I_0(r - d_f). The velocity is a least-squares line through d_f against the
 * frames' times: the viewer's frame_times_s when it has one time per frame (scans taken unevenly),
 * else frame index x the frame axis sampling when that axis is a time. In pixels and, with a pixel
 * size, in physical units a microscope can be given.
 */
import { estimateSliceAlignment } from "../show3dslices/sliceAlignment";

/** Subpixel refinement grid: 1/100 px, as quantem.gpu's drift registration. */
export const DRIFT_UPSAMPLE = 100;

/** Seconds per unit of the frame axis, for the units a time axis is given in; null when not a time. */
const SECONDS_PER: Record<string, number> = { s: 1, ms: 1e-3, us: 1e-6, "µs": 1e-6, min: 60, h: 3600 };

export interface DriftResult {
  /** The chosen frames (stack indices), in order; the first is the reference. */
  frames: number[];
  /** Specimen motion (row, col) px from the first chosen frame, one per frame (the first is 0, 0). */
  shift_px: [number, number][];
  /** Normalised correlation peak per adjacent pair, 1 a perfect match. */
  quality: number[];
  /** Fitted motion (row, col) px per frame index. */
  velocity_px_per_frame: [number, number];
  /** Seconds from the first chosen frame, one per chosen frame; null when the frames have no time. */
  times_s: number[] | null;
  /** Fitted motion (row, col) px per second against times_s; null without times. */
  velocity_px_per_s: [number, number] | null;
  /** How well a straight line fits each axis (1: perfectly linear drift), against time when there is one. */
  fit_r2: [number, number];
  /** (row, col) in pixel_unit, when the viewer has a pixel size. */
  shift: [number, number][] | null;
  /** (row, col) in pixel_unit per second, when the frames have times. */
  velocity: [number, number] | null;
  pixel_size: number;
  pixel_unit: string;
  /** Mean seconds per frame index over the chosen frames; null without times. */
  frame_interval_s: number | null;
  velocity_unit: string;
  upsample: number;
  backend: "webgpu" | "cpu";
}

/**
 * Seconds from the first chosen frame for each chosen frame: from one time per stack frame when the
 * viewer has them, else frame index x the frame axis sampling when that axis is a time; else null.
 */
function frameTimes(chosen: number[], frameTimesS: number[] | null, frameSampling: number, frameUnit: string): number[] | null {
  if (frameTimesS && frameTimesS.length > 0) {
    const times = chosen.map((index) => frameTimesS[index]);
    if (times.some((time) => !Number.isFinite(time))) throw new Error("frame_times_s has no time for every picked frame");
    const relative = times.map((time) => time - times[0]);
    if (relative.some((time, i) => i > 0 && time <= relative[i - 1])) throw new Error("frame_times_s must increase frame by frame");
    return relative;
  }
  const perSecond = SECONDS_PER[frameUnit.trim()];
  if (perSecond === undefined || !(frameSampling > 0)) return null;
  return chosen.map((index) => (index - chosen[0]) * frameSampling * perSecond);
}

/** Least-squares slope and R^2 of y against x. */
function line(x: number[], y: number[]): { slope: number; r2: number } {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) ** 2;
    syy += (y[i] - my) ** 2;
  }
  const slope = sxx > 0 ? sxy / sxx : 0;
  const residual = y.reduce((sum, value, i) => sum + (value - (my + slope * (x[i] - mx))) ** 2, 0);
  return { slope, r2: syy > 0 ? 1 - residual / syy : 1 };
}

/**
 * Drift over the chosen frames of a stack.
 *
 * @param frame Returns stack frame `index` as rows x cols floats.
 * @param frames Stack indices to compare, at least two, in time order.
 * @param device The page's hardware WebGPU device, or null for the CPU path.
 * @param calibration The viewer's pixel size and unit, the frame axis sampling and unit, and one time
 *   per stack frame in seconds (frameTimesS) when the frames were not taken evenly.
 */
export async function findDrift(
  frame: (index: number) => Float32Array | null,
  frames: number[],
  width: number,
  height: number,
  device: GPUDevice | null,
  calibration: { pixelSize: number; pixelUnit: string; frameSampling: number; frameUnit: string; frameTimesS?: number[] | null },
  upsample: number = DRIFT_UPSAMPLE,
): Promise<DriftResult> {
  const chosen = [...new Set(frames.map((value) => Math.round(value)))].sort((a, b) => a - b);
  if (chosen.length < 2) throw new Error("Find drift needs at least two frames");
  const plane = width * height;
  const volume = new Float32Array(chosen.length * plane);
  chosen.forEach((index, slot) => {
    const data = frame(index);
    if (!data || data.length < plane) throw new Error(`frame ${index + 1} is not available in the browser`);
    volume.set(data.subarray(0, plane), slot * plane);
  });
  const estimate = await estimateSliceAlignment(volume, width, height, chosen.length, device, upsample);
  // sliceAlignment reports the shift that carries each frame back onto the reference; the specimen moved the other way
  const shift_px = estimate.cumulativeShiftPx.map(([row, col]) => [row === 0 ? 0 : -row, col === 0 ? 0 : -col] as [number, number]);
  const rows = shift_px.map((value) => value[0]);
  const cols = shift_px.map((value) => value[1]);
  const rowFit = line(chosen, rows);
  const colFit = line(chosen, cols);
  const { pixelSize, pixelUnit, frameSampling, frameUnit, frameTimesS } = calibration;
  const times_s = frameTimes(chosen, frameTimesS ?? null, frameSampling, frameUnit);
  const rowTime = times_s ? line(times_s, rows) : null;
  const colTime = times_s ? line(times_s, cols) : null;
  const velocity_px_per_s: [number, number] | null = rowTime && colTime ? [rowTime.slope, colTime.slope] : null;
  const span = chosen[chosen.length - 1] - chosen[0];
  const frameIntervalS = times_s && span > 0 ? times_s[times_s.length - 1] / span : null;
  const calibrated = pixelSize > 0;
  return {
    frames: chosen,
    shift_px,
    quality: estimate.quality,
    velocity_px_per_frame: [rowFit.slope, colFit.slope],
    times_s,
    velocity_px_per_s,
    fit_r2: rowTime && colTime ? [rowTime.r2, colTime.r2] : [rowFit.r2, colFit.r2],
    shift: calibrated ? shift_px.map(([row, col]) => [row * pixelSize, col * pixelSize] as [number, number]) : null,
    velocity: calibrated && velocity_px_per_s ? [velocity_px_per_s[0] * pixelSize, velocity_px_per_s[1] * pixelSize] : null,
    pixel_size: pixelSize,
    pixel_unit: calibrated ? pixelUnit : "px",
    frame_interval_s: frameIntervalS,
    velocity_unit: calibrated && velocity_px_per_s ? `${pixelUnit}/s` : "px/frame",
    upsample,
    backend: estimate.backend,
  };
}
