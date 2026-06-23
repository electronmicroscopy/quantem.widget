// Statistics utilities for client-side rendering.
// Mirrors ../../js/stats.ts - kept local so tsconfig "include": ["src"] resolves it.

/** Bin a Float32Array into a small fixed-width histogram. Returns an array
 *  normalized to [0, 1] so it can feed HistogramStrip directly. Any finite
 *  clamp range is accepted; NaN/Inf samples are skipped. When `logScale` is
 *  true (default) applies log1p on the bin counts so a handful of
 *  saturated-signal bins don't flatten the rest of the distribution; when
 *  false, bars are normalized linearly by the peak count. O(n) single pass. */
export function bucketize(
  data: Float32Array, vmin: number, vmax: number, nbins: number = 64,
  logScale: boolean = true,
): number[] {
  if (nbins <= 0 || data.length === 0 || !(vmax > vmin)) return [];
  const counts = new Uint32Array(nbins);
  const scale = nbins / (vmax - vmin);
  const maxBin = nbins - 1;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    let b = Math.floor((v - vmin) * scale);
    if (b < 0) b = 0;
    else if (b > maxBin) b = maxBin;
    counts[b]++;
  }
  if (logScale) {
    let peak = 0;
    for (let i = 0; i < nbins; i++) {
      const lc = Math.log1p(counts[i]);
      if (lc > peak) peak = lc;
    }
    if (peak === 0) return new Array(nbins).fill(0);
    const out = new Array(nbins);
    for (let i = 0; i < nbins; i++) out[i] = Math.log1p(counts[i]) / peak;
    return out;
  }
  let peak = 0;
  for (let i = 0; i < nbins; i++) if (counts[i] > peak) peak = counts[i];
  if (peak === 0) return new Array(nbins).fill(0);
  const out = new Array(nbins);
  for (let i = 0; i < nbins; i++) out[i] = counts[i] / peak;
  return out;
}

/** Percentile-based clipping using O(n) histogram approach. */
export function percentileClip(
  data: Float32Array, pLow: number, pHigh: number,
): { vmin: number; vmax: number } {
  const len = data.length;
  if (len === 0) return { vmin: 0, vmax: 0 };

  // Pass 1: find min/max
  let min = Infinity, max = -Infinity;
  let finiteCount = 0;
  for (let i = 0; i < len; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    finiteCount++;
  }
  if (finiteCount === 0) return { vmin: 0, vmax: 1 };
  if (min === max) return { vmin: min, vmax: max };

  // Pass 2: build histogram
  const NUM_BINS = 1024;
  const bins = new Uint32Array(NUM_BINS);
  const range = max - min;
  const scale = (NUM_BINS - 1) / range;
  for (let i = 0; i < len; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    bins[Math.floor((v - min) * scale)]++;
  }

  // Walk cumulative histogram to find percentile values
  const lowCount = Math.floor(finiteCount * (pLow / 100));
  const highCount = Math.ceil(finiteCount * (pHigh / 100));
  let cumSum = 0;
  let vmin = min, vmax = max;
  for (let i = 0; i < NUM_BINS; i++) {
    cumSum += bins[i];
    if (cumSum >= lowCount) { vmin = min + (i / (NUM_BINS - 1)) * range; break; }
  }
  cumSum = 0;
  for (let i = 0; i < NUM_BINS; i++) {
    cumSum += bins[i];
    if (cumSum >= highCount) { vmax = min + (i / (NUM_BINS - 1)) * range; break; }
  }
  return { vmin, vmax };
}

/** Percentile clip with optional center-pixel mask. Mirrors Show4DSTEM
 *  `mask_dc`: skips a 3×3 region around (centerRow, centerCol) so the
 *  central beam doesn't crush the percentile range on diffraction
 *  patterns. show4dstem.py:3991 ports the same 3×3 box. */
export function percentileClipMasked(
  data: Float32Array, w: number, h: number,
  pLow: number, pHigh: number,
  maskCenter: boolean, centerRow: number, centerCol: number,
): { vmin: number; vmax: number } {
  if (!maskCenter || w < 4 || h < 4) return percentileClip(data, pLow, pHigh);
  const cr = Math.max(1, Math.min(h - 2, Math.round(centerRow)));
  const cc = Math.max(1, Math.min(w - 2, Math.round(centerCol)));
  // Build a copy with center 3×3 replaced by NaN, then re-pack the
  // valid samples. Matches show4dstem.py masked_vals = frame[mask].
  const valid = new Float32Array(data.length - 9);
  let j = 0;
  for (let r = 0; r < h; r++) {
    const rowOff = r * w;
    const inMaskRow = r >= cr - 1 && r <= cr + 1;
    for (let c = 0; c < w; c++) {
      if (inMaskRow && c >= cc - 1 && c <= cc + 1) continue;
      valid[j++] = data[rowOff + c];
    }
  }
  return percentileClip(valid.subarray(0, j), pLow, pHigh);
}
