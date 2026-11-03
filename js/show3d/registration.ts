/**
 * Sub-pixel registration for the Sub-pixel align control: the shift between two frames from the peak of
 * their cross-correlation, refined per axis by a parabola, and the bilinear translation that applies it.
 */
import { CPUFFT, type DisplayFFT, applyHannWindow2D, nextPow2 } from "../display/fft";

export type SubpixelShift = {
  row: number;
  col: number;
  quality: number;
};

function finiteMean(data: Float32Array): number {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < data.length; i++) {
    const value = data[i];
    if (!Number.isFinite(value)) continue;
    sum += value;
    count++;
  }
  return count > 0 ? sum / count : 0;
}

export function finiteMedianSample(data: Float32Array, maxSamples = 8192): number {
  const step = Math.max(1, Math.floor(data.length / maxSamples));
  const values: number[] = [];
  for (let i = 0; i < data.length; i += step) {
    const value = data[i];
    if (Number.isFinite(value)) values.push(value);
  }
  if (!values.length) return 0;
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)];
}

/**
 * Mean-subtracted, Hann-windowed copy of a frame with non-finite pixels set to
 * 0. Without the mean removal and the taper the cross-correlation peak locks
 * onto the DC term and the frame edges instead of image features.
 */
function registrationImage(data: Float32Array, width: number, height: number): Float32Array {
  const out = new Float32Array(width * height);
  const mean = finiteMean(data);
  for (let i = 0; i < out.length; i++) {
    const value = data[i];
    out[i] = Number.isFinite(value) ? value - mean : 0;
  }
  applyHannWindow2D(out, width, height);
  return out;
}

/**
 * 2D complex FFT (inverse when `inverse`) zero-padded to the next power-of-two
 * grid, on the WebGPU FFT when the browser has one and the CPU reference
 * otherwise. The padded size is returned because the correlation peak is
 * measured on that grid.
 */
async function fft2dComplex(
  real: Float32Array,
  imag: Float32Array,
  width: number,
  height: number,
  inverse: boolean,
  fftEngine: DisplayFFT,
): Promise<{ real: Float32Array; imag: Float32Array; width: number; height: number }> {
  const paddedW = nextPow2(width);
  const paddedH = nextPow2(height);
  if (paddedW === width && paddedH === height) {
    return { ...(await fftEngine.fft2D(real, imag, width, height, inverse)), width, height };
  }
  const realPadded = new Float32Array(paddedW * paddedH);
  const imagPadded = new Float32Array(paddedW * paddedH);
  for (let row = 0; row < height; row++) {
    realPadded.set(real.subarray(row * width, row * width + width), row * paddedW);
    imagPadded.set(imag.subarray(row * width, row * width + width), row * paddedW);
  }
  return { ...(await fftEngine.fft2D(realPadded, imagPadded, paddedW, paddedH, inverse)), width: paddedW, height: paddedH };
}

/** Signed shift of a correlation peak on a periodic axis: indices past the midpoint are negative. */
function wrappedPeakOffset(index: number, size: number): number {
  return index > size / 2 ? index - size : index;
}

/**
 * Sub-pixel peak offset from three samples,
 * delta = (prev - next) / (2 (prev - 2 center + next)), clamped to [-0.5, 0.5];
 * 0 when the curvature vanishes.
 */
function parabolicPeakDelta(prev: number, center: number, next: number): number {
  const denom = prev - 2 * center + next;
  if (!Number.isFinite(denom) || Math.abs(denom) < 1e-12) return 0;
  const delta = 0.5 * (prev - next) / denom;
  if (!Number.isFinite(delta)) return 0;
  return Math.max(-0.5, Math.min(0.5, delta));
}

// Below this many padded pixels the GPU round trips cost more than the CPU transforms
// (measured at 256^2 on Blackwell with three transforms per frame: 16 ms against 9 ms).
const GPU_MIN_PADDED_PIXELS = 512 * 512;

/**
 * Shift (row, col) that registers each moving frame onto `reference`: the
 * cross-correlation IFFT(F_ref * conj(F_mov)), its integer peak, then a
 * parabolic refinement per axis. `quality` is the peak height over the mean
 * correlation magnitude away from the peak. The reference spectrum is computed
 * once for all frames; `fftEngine` runs the transforms unless the frames are
 * small enough that the CPU reference is faster.
 */
export function subpixelShiftEstimator(
  reference: Float32Array,
  width: number,
  height: number,
  fftEngine: DisplayFFT,
): { path: DisplayFFT["path"]; estimate: (moving: Float32Array) => Promise<SubpixelShift> } {
  const engine = nextPow2(width) * nextPow2(height) < GPU_MIN_PADDED_PIXELS ? new CPUFFT() : fftEngine;
  const referenceImage = registrationImage(reference, width, height);
  const referenceSpectrum = fft2dComplex(referenceImage, new Float32Array(referenceImage.length), width, height, false, engine);
  const estimate = async (moving: Float32Array): Promise<SubpixelShift> => {
    const movingImage = registrationImage(moving, width, height);
    const movingSpectrum = await fft2dComplex(movingImage, new Float32Array(movingImage.length), width, height, false, engine);
    const { real: referenceReal, imag: referenceImag, width: workW, height: workH } = await referenceSpectrum;
    const workSize = workW * workH;
    const crossReal = new Float32Array(workSize);
    const crossImag = new Float32Array(workSize);
    for (let i = 0; i < workSize; i++) {
      crossReal[i] = referenceReal[i] * movingSpectrum.real[i] + referenceImag[i] * movingSpectrum.imag[i];
      crossImag[i] = referenceImag[i] * movingSpectrum.real[i] - referenceReal[i] * movingSpectrum.imag[i];
    }
    const correlation = await fft2dComplex(crossReal, crossImag, workW, workH, true, engine);
    let peakIdx = 0;
    let peakValue = -Infinity;
    let total = 0;
    for (let i = 0; i < correlation.real.length; i++) {
      const value = Math.hypot(correlation.real[i], correlation.imag[i]);
      total += value;
      if (value > peakValue) {
        peakValue = value;
        peakIdx = i;
      }
    }
    const peakRow = Math.floor(peakIdx / correlation.width);
    const peakCol = peakIdx % correlation.width;
    const correlationMagnitudeAt = (row: number, col: number) => {
      const wrappedRow = (row + correlation.height) % correlation.height;
      const wrappedCol = (col + correlation.width) % correlation.width;
      const idx = wrappedRow * correlation.width + wrappedCol;
      return Math.hypot(correlation.real[idx], correlation.imag[idx]);
    };
    const rowDelta = parabolicPeakDelta(correlationMagnitudeAt(peakRow - 1, peakCol), peakValue, correlationMagnitudeAt(peakRow + 1, peakCol));
    const colDelta = parabolicPeakDelta(correlationMagnitudeAt(peakRow, peakCol - 1), peakValue, correlationMagnitudeAt(peakRow, peakCol + 1));
    const row = wrappedPeakOffset(peakRow, correlation.height) + rowDelta;
    const col = wrappedPeakOffset(peakCol, correlation.width) + colDelta;
    const background = total > 0 ? (total - peakValue) / Math.max(1, correlation.real.length - 1) : 0;
    const quality = background > 1e-12 ? peakValue / background : peakValue;
    return { row, col, quality };
  };
  return { path: engine.path, estimate };
}

/**
 * Frame translated by (rowShift, colShift) with bilinear interpolation,
 * out(row, col) = in(row - rowShift, col - colShift); pixels whose source falls
 * off the frame get fillValue. Shifts below 1e-4 px return the input.
 */
export function shiftFrameBilinear(
  frame: Float32Array,
  width: number,
  height: number,
  rowShift: number,
  colShift: number,
  fillValue: number,
): Float32Array {
  if (Math.abs(rowShift) < 1e-4 && Math.abs(colShift) < 1e-4) return frame;
  const out = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    const srcRow = row - rowShift;
    const rowFloor = Math.floor(srcRow);
    const rowFrac = srcRow - rowFloor;
    for (let col = 0; col < width; col++) {
      const srcCol = col - colShift;
      const colFloor = Math.floor(srcCol);
      const colFrac = srcCol - colFloor;
      const dst = row * width + col;
      if (rowFloor < 0 || colFloor < 0 || rowFloor >= height - 1 || colFloor >= width - 1) {
        out[dst] = fillValue;
        continue;
      }
      const idx = rowFloor * width + colFloor;
      const v00 = frame[idx];
      const v01 = frame[idx + 1];
      const v10 = frame[idx + width];
      const v11 = frame[idx + width + 1];
      out[dst] =
        v00 * (1 - rowFrac) * (1 - colFrac) +
        v01 * (1 - rowFrac) * colFrac +
        v10 * rowFrac * (1 - colFrac) +
        v11 * rowFrac * colFrac;
    }
  }
  return out;
}
