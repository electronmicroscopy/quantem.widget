/**
 * Sample intensity values along a line using bilinear interpolation.
 * Returns one float per pixel-step along the segment from (row0,col0) to (row1,col1).
 */
// ported from quantem.widget show3d/index.tsx:381
export function sampleSingleLine(
  data: Float32Array,
  w: number,
  h: number,
  row0: number,
  col0: number,
  row1: number,
  col1: number,
): Float32Array {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  const n = Math.max(2, Math.ceil(len));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const c = col0 + t * dc;
    const r = row0 + t * dr;
    const ci = Math.floor(c), ri = Math.floor(r);
    const cf = c - ci, rf = r - ri;
    const c0c = Math.max(0, Math.min(w - 1, ci));
    const c1c = Math.max(0, Math.min(w - 1, ci + 1));
    const r0c = Math.max(0, Math.min(h - 1, ri));
    const r1c = Math.max(0, Math.min(h - 1, ri + 1));
    out[i] = data[r0c * w + c0c] * (1 - cf) * (1 - rf) +
             data[r0c * w + c1c] * cf * (1 - rf) +
             data[r1c * w + c0c] * (1 - cf) * rf +
             data[r1c * w + c1c] * cf * rf;
  }
  return out;
}

/**
 * Sample intensity along a line, averaging over profileWidth perpendicular pixels for noise robustness.
 */
// ported from quantem.widget show3d/index.tsx:406
export function sampleLineProfile(
  data: Float32Array,
  w: number,
  h: number,
  row0: number,
  col0: number,
  row1: number,
  col1: number,
  profileWidth: number = 1,
): Float32Array {
  if (profileWidth <= 1) return sampleSingleLine(data, w, h, row0, col0, row1, col1);
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  if (len < 1e-8) return sampleSingleLine(data, w, h, row0, col0, row1, col1);
  const perpR = -dc / len;
  const perpC = dr / len;
  const half = (profileWidth - 1) / 2;
  let accumulated: Float32Array | null = null;
  for (let k = 0; k < profileWidth; k++) {
    const off = -half + k;
    const vals = sampleSingleLine(data, w, h, row0 + off * perpR, col0 + off * perpC, row1 + off * perpR, col1 + off * perpC);
    if (!accumulated) {
      accumulated = vals;
    } else {
      for (let i = 0; i < vals.length; i++) accumulated[i] += vals[i];
    }
  }
  if (accumulated) for (let i = 0; i < accumulated.length; i++) accumulated[i] /= profileWidth;
  return accumulated || new Float32Array(0);
}

/**
 * Distance from a point (col,row) to the line segment (col0,row0)-(col1,row1); used for "click near line" hit-testing.
 */
// ported from quantem.widget show3d/index.tsx:429
export function pointToSegmentDistance(
  col: number,
  row: number,
  col0: number,
  row0: number,
  col1: number,
  row1: number,
): number {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const lenSq = dc * dc + dr * dr;
  if (lenSq <= 1e-12) return Math.sqrt((col - col0) ** 2 + (row - row0) ** 2);
  const tRaw = ((col - col0) * dc + (row - row0) * dr) / lenSq;
  const t = Math.max(0, Math.min(1, tRaw));
  const projCol = col0 + t * dc;
  const projRow = row0 + t * dr;
  return Math.sqrt((col - projCol) ** 2 + (row - projRow) ** 2);
}
