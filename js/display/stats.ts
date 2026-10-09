/** Display statistics: data range, log scale, percentile clip, histogram. */

/** Find min/max range of a Float32Array, filtering out NaN and Infinity. */
export function findDataRange(data: Float32Array): { min: number; max: number } {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const value = data[i];
    if (!isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (min === Infinity) return { min: 0, max: 0 };
  return { min, max };
}

/** Signed log1p. For non-negative inputs identical to log1p(x); for negatives
 *  returns -log1p(|x|) so diff_mode frames don't collapse to zero. */
export function signedLog1p(value: number): number {
  return value >= 0 ? Math.log1p(value) : -Math.log1p(-value);
}

/** Signed log1p of every sample, into a new array. */
export function applyLogScale(data: Float32Array): Float32Array {
  const result = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) result[i] = signedLog1p(data[i]);
  return result;
}

/** Apply signed log1p scale into a pre-allocated buffer. Avoids per-frame allocation. */
export function applyLogScaleInPlace(data: Float32Array, out: Float32Array): Float32Array {
  for (let i = 0; i < data.length; i++) {
    out[i] = signedLog1p(data[i]);
  }
  return out;
}

type NumericArray = ArrayLike<number>;

/**
 * Data value where the cumulative histogram first reaches `targetCount`,
 * linearly interpolated inside the crossing bin so the result is continuous in
 * the data rather than snapped to bin edges; `fallback` when it never does.
 * Bin i spans [min + i * range / bins, min + (i + 1) * range / bins).
 */
function histogramQuantile(bins: Uint32Array, targetCount: number, min: number, range: number, fallback: number): number {
  let cumSum = 0;
  for (let i = 0; i < bins.length; i++) {
    const prevSum = cumSum;
    cumSum += bins[i];
    if (cumSum >= targetCount) {
      const frac = (targetCount - prevSum) / Math.max(1, cumSum - prevSum);
      return min + ((i + frac) / bins.length) * range;
    }
  }
  return fallback;
}

/** Percentile-based clipping using O(n) histogram approach.
 *  Also returns data min/max so callers can skip a redundant findDataRange scan. */
export function percentileClip(
  data: NumericArray, pLow: number, pHigh: number,
): { vmin: number; vmax: number; min: number; max: number } {
  const len = data.length;
  if (len === 0) return { vmin: 0, vmax: 0, min: 0, max: 0 };

  // Pass 1: find min/max
  let min = Infinity, max = -Infinity;
  let finiteCount = 0;
  for (let i = 0; i < len; i++) {
    const value = data[i];
    if (!isFinite(value)) continue;
    finiteCount++;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (finiteCount === 0) return { vmin: 0, vmax: 0, min: 0, max: 0 };
  if (min === max) return { vmin: min, vmax: max, min, max };

  // Pass 2: build histogram
  const NUM_BINS = 1024;
  const bins = new Uint32Array(NUM_BINS);
  const range = max - min;
  // Bin i spans [min + i*range/NUM_BINS, min + (i+1)*range/NUM_BINS); the walk
  // below maps bins back with the same width, and max lands in the last bin.
  const scale = NUM_BINS / range;
  for (let i = 0; i < len; i++) {
    const value = data[i];
    if (isFinite(value)) bins[Math.min(NUM_BINS - 1, Math.floor((value - min) * scale))]++;
  }

  const vmin = histogramQuantile(bins, finiteCount * (pLow / 100), min, range, min);
  const vmax = histogramQuantile(bins, finiteCount * (pHigh / 100), min, range, max);
  return { vmin, vmax, min, max };
}

/** Compute mean, min, max, and standard deviation of a Float32Array. */
export function computeStats(data: Float32Array): { mean: number; min: number; max: number; std: number } {
  if (data.length === 0) return { mean: 0, min: 0, max: 0, std: 0 };
  let sum = 0, min = Infinity, max = -Infinity;
  let count = 0;
  for (let i = 0; i < data.length; i++) {
    const value = data[i];
    if (!isFinite(value)) continue;
    count++;
    sum += value;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (count === 0) return { mean: 0, min: 0, max: 0, std: 0 };
  const mean = sum / count;
  let variance = 0;
  for (let i = 0; i < data.length; i++) {
    if (isFinite(data[i])) variance += (data[i] - mean) ** 2;
  }
  const std = Math.sqrt(variance / count);
  return { mean, min, max, std };
}

/** Convert histogram slider percentages (0-100) to vmin/vmax in data space. */
export function sliderRange(
  dataMin: number, dataMax: number, vminPct: number, vmaxPct: number,
): { vmin: number; vmax: number } {
  const range = dataMax - dataMin;
  return {
    vmin: dataMin + (vminPct / 100) * range,
    vmax: dataMin + (vmaxPct / 100) * range,
  };
}

/** Compute normalized histogram bins from Float32Array.
 *  fixedMin/fixedMax pin bin edges to a global range (so scrubbing through
 *  a stack doesn't rescale per-frame). Defaults to per-array min/max. */
export function computeHistogramFromBytes(
  data: NumericArray | null,
  numBins = 256,
  fixedMin?: number,
  fixedMax?: number,
): number[] {
  if (!data || data.length === 0) return new Array(numBins).fill(0);
  const bins = new Array(numBins).fill(0);
  let min: number, max: number;
  if (fixedMin !== undefined && fixedMax !== undefined && isFinite(fixedMin) && isFinite(fixedMax)) {
    min = fixedMin;
    max = fixedMax;
  } else {
    min = Infinity; max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const value = data[i];
      if (isFinite(value)) { if (value < min) min = value; if (value > max) max = value; }
    }
    if (!isFinite(min) || !isFinite(max)) return bins;
  }
  if (!(max > min)) {
    let count = 0;
    for (let i = 0; i < data.length; i++) if (isFinite(data[i])) count++;
    if (count > 0) bins[Math.floor(numBins / 2)] = 1;
    return bins;
  }
  const range = max - min;
  for (let i = 0; i < data.length; i++) {
    const value = data[i];
    if (isFinite(value)) {
      // Clamp into last bin so max-value pixels aren't silently dropped.
      let bin = Math.floor(((value - min) / range) * numBins);
      if (bin === numBins) bin = numBins - 1;
      if (bin >= 0 && bin < numBins) bins[bin]++;
    }
  }
  const maxCount = Math.max(...bins);
  if (maxCount > 0) for (let i = 0; i < numBins; i++) bins[i] /= maxCount;
  return bins;
}
