/**
 * Client-side 2D FFT and image processing utilities.
 * Replaces server-side /api/fft and /api/profile endpoints for instant interactivity.
 */

// 1D FFT (Cooley-Tukey, in-place)
function fft1d(re: Float64Array, im: Float64Array, n: number): void {
  // Bit reversal
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  // FFT butterfly
  for (let len = 2; len <= n; len *= 2) {
    const half = len / 2;
    const angle = -2 * Math.PI / len;
    const wRe = Math.cos(angle), wIm = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < half; k++) {
        const tRe = curRe * re[i + k + half] - curIm * im[i + k + half];
        const tIm = curRe * im[i + k + half] + curIm * re[i + k + half];
        re[i + k + half] = re[i + k] - tRe;
        im[i + k + half] = im[i + k] - tIm;
        re[i + k] += tRe;
        im[i + k] += tIm;
        const newCurRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCurRe;
      }
    }
  }
}

// Next power of 2
function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

// Shared shift + log-magnitude + DC suppression for FFT outputs.
// Inputs:
//   re/im  — complex FFT output of size (paddedRows × paddedCols).
//   rows/cols — output crop centered on DC (typically the original ROI size).
//   paddedRows/paddedCols — the FFT padded shape (= rows/cols when no padding).
// Output: log-magnitude buffer (rows*cols), DC region replaced with the median
// to suppress spectral leakage. Used by both fft2dMagnitude (CPU) and roiFFT
// (WebGPU) so the on-screen FFT is identical regardless of compute path.
function shiftLogDC(
  re: ArrayLike<number>, im: ArrayLike<number>,
  rows: number, cols: number,
  paddedRows: number, paddedCols: number,
): Float32Array {
  const result = new Float32Array(rows * cols);
  const halfR = Math.floor(rows / 2), halfC = Math.floor(cols / 2);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const sr = ((r - halfR) % paddedRows + paddedRows) % paddedRows;
      const sc = ((c - halfC) % paddedCols + paddedCols) % paddedCols;
      const reV = re[sr * paddedCols + sc];
      const imV = im[sr * paddedCols + sc];
      result[r * cols + c] = Math.log1p(Math.sqrt(reV * reV + imV * imV));
    }
  }
  const sorted = Float32Array.from(result).sort();
  const median = sorted[Math.floor(sorted.length / 2)];
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      const rr = halfR + dr, cc = halfC + dc;
      if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) {
        result[rr * cols + cc] = median;
      }
    }
  }
  return result;
}

// 2D FFT -> log magnitude, fftshifted
export function fft2dMagnitude(data: Float32Array, rows: number, cols: number): Float32Array {
  const pr = nextPow2(rows), pc = nextPow2(cols);

  // Pad to power of 2
  const re = new Float64Array(pr * pc);
  const im = new Float64Array(pr * pc);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      re[r * pc + c] = data[r * cols + c];
    }
  }

  // FFT along rows
  const rowRe = new Float64Array(pc);
  const rowIm = new Float64Array(pc);
  for (let r = 0; r < pr; r++) {
    for (let c = 0; c < pc; c++) { rowRe[c] = re[r * pc + c]; rowIm[c] = im[r * pc + c]; }
    fft1d(rowRe, rowIm, pc);
    for (let c = 0; c < pc; c++) { re[r * pc + c] = rowRe[c]; im[r * pc + c] = rowIm[c]; }
  }

  // FFT along columns
  const colRe = new Float64Array(pr);
  const colIm = new Float64Array(pr);
  for (let c = 0; c < pc; c++) {
    for (let r = 0; r < pr; r++) { colRe[r] = re[r * pc + c]; colIm[r] = im[r * pc + c]; }
    fft1d(colRe, colIm, pr);
    for (let r = 0; r < pr; r++) { re[r * pc + c] = colRe[r]; im[r * pc + c] = colIm[r]; }
  }

  return shiftLogDC(re, im, rows, cols, pr, pc);
}

// Display-oriented preprocessing for image FFTs. This keeps the FFT style
// consistent across PanelViewer, thumbnails, compare dialogs, and the
// aberration explorer: remove DC / row / column background, then window.
export function prepareFftInput(data: Float32Array, rows: number, cols: number, applyWindow: boolean = true): Float32Array {
  const n = rows * cols;
  const centered = new Float32Array(n);
  let sum = 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    sum += v;
    count++;
  }
  const mean = count > 0 ? sum / count : 0;
  const rowSums = new Float64Array(rows);
  const rowCounts = new Uint32Array(rows);
  const colSums = new Float64Array(cols);
  const colCounts = new Uint32Array(cols);
  let centeredEnergy = 0;
  for (let r = 0; r < rows; r++) {
    const rowOff = r * cols;
    for (let c = 0; c < cols; c++) {
      const idx = rowOff + c;
      const raw = data[idx];
      const v = Number.isFinite(raw) ? raw - mean : 0;
      centered[idx] = v;
      centeredEnergy += v * v;
      if (Number.isFinite(raw)) {
        rowSums[r] += v;
        rowCounts[r]++;
        colSums[c] += v;
        colCounts[c]++;
      }
    }
  }
  let detrendedEnergy = 0;
  let detrendedMaxAbs = 0;
  for (let r = 0; r < rows; r++) {
    const rowMean = rowCounts[r] > 0 ? rowSums[r] / rowCounts[r] : 0;
    const rowOff = r * cols;
    for (let c = 0; c < cols; c++) {
      const colMean = colCounts[c] > 0 ? colSums[c] / colCounts[c] : 0;
      const v = centered[rowOff + c] - rowMean - colMean;
      centered[rowOff + c] = v;
      detrendedEnergy += v * v;
      const abs = Math.abs(v);
      if (abs > detrendedMaxAbs) detrendedMaxAbs = abs;
    }
  }
  // Row/column detrending makes Com/DPC stripe panels look like SSB FFTs, but
  // some nearly rank-1 images lose almost all signal. In that rare case keep
  // the global-mean centered spectrum so the viewer never paints a blank FFT.
  if (
    detrendedMaxAbs < 1e-12 ||
    (centeredEnergy > 0 && detrendedEnergy / centeredEnergy < 1e-5)
  ) {
    const centerOnly = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const raw = data[i];
      centerOnly[i] = Number.isFinite(raw) ? raw - mean : 0;
    }
    return applyWindow ? applyHann2D(centerOnly, rows, cols) : centerOnly;
  }
  return applyWindow ? applyHann2D(centered, rows, cols) : centered;
}

// Extract line profile from raw data
export function extractProfile(
  data: Float32Array, rows: number, cols: number,
  r0: number, c0: number, r1: number, c1: number, numPoints: number = 256
): Float32Array {
  const result = new Float32Array(numPoints);
  for (let i = 0; i < numPoints; i++) {
    const t = i / (numPoints - 1);
    const r = r0 * rows + (r1 - r0) * rows * t;
    const c = c0 * cols + (c1 - c0) * cols * t;
    const ri = Math.min(Math.max(Math.round(r), 0), rows - 1);
    const ci = Math.min(Math.max(Math.round(c), 0), cols - 1);
    result[i] = data[ri * cols + ci];
  }
  return result;
}

// Hann window (2D, applied before FFT to reduce spectral leakage)
export function applyHann2D(data: Float32Array, rows: number, cols: number): Float32Array {
  const result = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const hr = 0.5 * (1 - Math.cos(2 * Math.PI * r / (rows - 1)));
    for (let c = 0; c < cols; c++) {
      const hc = 0.5 * (1 - Math.cos(2 * Math.PI * c / (cols - 1)));
      result[r * cols + c] = data[r * cols + c] * hr * hc;
    }
  }
  return result;
}

// Extract ROI from raw data (normalized [0,1] coordinates)
export function extractROI(
  data: Float32Array, rows: number, cols: number,
  r0: number, c0: number, r1: number, c1: number
): { data: Float32Array; rows: number; cols: number } {
  const startR = Math.max(0, Math.round(r0 * rows));
  const endR = Math.min(rows, Math.round(r1 * rows));
  const startC = Math.max(0, Math.round(c0 * cols));
  const endC = Math.min(cols, Math.round(c1 * cols));
  const roiRows = endR - startR;
  const roiCols = endC - startC;
  const roi = new Float32Array(roiRows * roiCols);
  for (let r = 0; r < roiRows; r++) {
    for (let c = 0; c < roiCols; c++) {
      roi[r * roiCols + c] = data[(startR + r) * cols + (startC + c)];
    }
  }
  return { data: roi, rows: roiRows, cols: roiCols };
}

// Full pipeline: extract ROI -> Hann window -> FFT -> log magnitude.
// WebGPU only — no silent CPU fallback. The WebGPU pipeline must fail visibly
// so the operator sees a banner instead of a stalled
// drag while a 80 ms main-thread FFT runs per frame. Caller catches and
// surfaces the error in the UI.
import { getWebGPUFFT } from "./webgpu-fft";

export class WebGPUUnavailableError extends Error {
  constructor(message = "WebGPU FFT failed; ROI FFT requires WebGPU. Open in a browser with WebGPU enabled.") {
    super(message);
    this.name = "WebGPUUnavailableError";
  }
}

export async function roiFFT(
  data: Float32Array, rows: number, cols: number,
  r0: number, c0: number, r1: number, c1: number
): Promise<{ mag: Float32Array; rows: number; cols: number }> {
  const roi = extractROI(data, rows, cols, r0, c0, r1, c1);
  if (roi.rows < 4 || roi.cols < 4) return { mag: new Float32Array(0), rows: 0, cols: 0 };
  const windowed = applyHann2D(roi.data, roi.rows, roi.cols);

  const gpu = await getWebGPUFFT();
  if (!gpu) {
    throw new WebGPUUnavailableError();
  }
  const imag = new Float32Array(windowed.length); // zero imaginary
  const result = await gpu.fft2D(windowed, imag, roi.cols, roi.rows);
  const mag = shiftLogDC(result.real, result.imag, roi.rows, roi.cols, roi.rows, roi.cols);
  return { mag, rows: roi.rows, cols: roi.cols };
}

// Apply colormap (inferno) to normalized float32 data -> RGBA Uint8ClampedArray
export function applyInferno(data: Float32Array, width: number, height: number): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4);
  // Find percentile range
  const sorted = Float32Array.from(data).sort();
  const p2 = sorted[Math.floor(sorted.length * 0.02)];
  const p98 = sorted[Math.floor(sorted.length * 0.98)];
  const range = p98 - p2 || 1;

  // Simplified inferno LUT (13 key colors)
  const inferno = [
    [0,0,4],[22,11,57],[66,10,104],[106,23,110],[143,41,102],
    [176,63,81],[204,90,56],[227,121,34],[244,155,22],[253,191,33],
    [249,225,68],[232,250,131],[252,255,164]
  ];

  for (let i = 0; i < data.length; i++) {
    const t = Math.max(0, Math.min(1, (data[i] - p2) / range));
    const idx = t * (inferno.length - 1);
    const lo = Math.floor(idx), hi = Math.min(lo + 1, inferno.length - 1);
    const f = idx - lo;
    rgba[i * 4] = inferno[lo][0] * (1 - f) + inferno[hi][0] * f;
    rgba[i * 4 + 1] = inferno[lo][1] * (1 - f) + inferno[hi][1] * f;
    rgba[i * 4 + 2] = inferno[lo][2] * (1 - f) + inferno[hi][2] * f;
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}
