// Line geometry and line-scan sampling shared by the display and the offline recompute. Mirrors
// ShowPDF._effective_line, _line_band_mask and _linescan_samples in showpdf.py.
//
// Scan positions are whole pixels. The sampled line has its ends on scan positions and is clipped
// to the scan, so its band, its samples and its tip all sit on scan positions with data. The
// samples are evenly spaced from the start A to the tip B, one per scan step along the line's
// longer axis (capped by linescan_max_bins), and each scan position in the band is averaged into
// its nearest sample.

export interface LineEnds { r0: number; c0: number; r1: number; c1: number }

// Nearest scan position; halves round up (Python: math.floor(v + 0.5)).
export function snapToScan(v: number): number {
  return Math.floor(v + 0.5);
}

// The line actually sampled: the drawn line or its perpendicular bisector, with both ends snapped
// to scan positions and clipped to the scan. A line wholly outside the scan collapses to a point,
// which samples nothing.
export function effectiveLineEnds(l: LineEnds, perpendicular: boolean, rows: number, cols: number): LineEnds {
  const line = perpendicular ? perpendicularBisector(l) : l;
  const snapped = {
    r0: snapToScan(line.r0),
    c0: snapToScan(line.c0),
    r1: snapToScan(line.r1),
    c1: snapToScan(line.c1),
  };
  return clipToScan(snapped, rows, cols);
}

// Same midpoint and length, rotated 90 degrees. The length is sqrt(dr * dr + dc * dc) as in Python,
// not Math.hypot (which can be an ulp off), so an end half way between scan positions snaps the same.
function perpendicularBisector(l: LineEnds) {
  const dr = l.r1 - l.r0;
  const dc = l.c1 - l.c0;
  const len = Math.sqrt(dr * dr + dc * dc);
  if (len > 0) {
    const mr = (l.r0 + l.r1) / 2;
    const mc = (l.c0 + l.c1) / 2;
    const pr = -dc / len;   // unit vector perpendicular to the line
    const pc = dr / len;
    const half = len / 2;
    return { r0: mr - pr * half, c0: mc - pc * half, r1: mr + pr * half, c1: mc + pc * half };
  }
  return l;
}

// Liang-Barsky clip to the scan [0, rows - 1] x [0, cols - 1], with the clipped ends snapped again.
function clipToScan(l: LineEnds, rows: number, cols: number) {
  const { r0, c0 } = l;
  const dr = l.r1 - r0;
  const dc = l.c1 - c0;
  const point = { r0, c0, r1: r0, c1: c0 };
  let t0 = 0;
  let t1 = 1;
  const edges = [[-dr, r0], [dr, rows - 1 - r0], [-dc, c0], [dc, cols - 1 - c0]];
  for (const [p, q] of edges) {
    if (p === 0) {
      if (q < 0) return point;   // parallel to this edge and outside it
    } else if (p < 0) {
      t0 = Math.max(t0, q / p);
    } else {
      t1 = Math.min(t1, q / p);
    }
  }
  if (t0 > t1) return point;
  return {
    r0: snapToScan(r0 + t0 * dr),
    c0: snapToScan(c0 + t0 * dc),
    r1: snapToScan(r0 + t1 * dr),
    c1: snapToScan(c0 + t1 * dc),
  };
}

// Membership test for the band of the sampled line (see effectiveLineEnds), in the operation order
// of ShowPDF._line_band_mask: t (0 = A, 1 = B) in [0, 1], so the ends are flat, and the distance to
// the line at most max(0.5, width / 2), width in scan px. null for a zero-length line (no band).
export function lineBandIncludes(ends: LineEnds, width: number): ((r: number, c: number) => boolean) | null {
  const { r0, c0, r1, c1 } = ends;
  const dr = r1 - r0;
  const dc = c1 - c0;
  const segSq = dr * dr + dc * dc;
  if (segSq <= 0) return null;
  const len = Math.sqrt(segSq);
  const halfW = Math.max(0.5, width / 2);
  return (r, c) => {
    const pr = r - r0;
    const pc = c - c0;
    const t = (pr * dr + pc * dc) / segSq;
    return t >= 0 && t <= 1 && Math.abs(pr * dc - pc * dr) / len <= halfW;
  };
}

// Candidate columns [lo, hi] of scan row r for the band of a line of nonzero length, within
// [0, cols - 1] (lo > hi if none): the two conditions of lineBandIncludes solved for the column,
// one column wider on each side against rounding. Testing only these columns with lineBandIncludes
// gives exactly the band, in time proportional to its area instead of its bounding box.
export function lineBandColumns(ends: LineEnds, width: number, r: number, cols: number): [number, number] {
  const { r0, c0, r1, c1 } = ends;
  const dr = r1 - r0;
  const dc = c1 - c0;
  const segSq = dr * dr + dc * dc;
  const len = Math.sqrt(segSq);
  const halfW = Math.max(0.5, width / 2);
  const pr = r - r0;
  let lo = 0;
  let hi = cols - 1;
  // Keeps the columns c0 + pc with pc between a and b.
  const keepBetween = (a: number, b: number) => {
    lo = Math.max(lo, Math.floor(c0 + Math.min(a, b)) - 1);
    hi = Math.min(hi, Math.ceil(c0 + Math.max(a, b)) + 1);
  };
  // Distance: |pr * dc - pc * dr| <= halfW * len, with pc = c - c0. On a horizontal line it does
  // not depend on the column and is tested as lineBandIncludes does.
  if (dr === 0) {
    if (Math.abs(pr * dc) / len > halfW) return [0, -1];
  } else {
    keepBetween((pr * dc - halfW * len) / dr, (pr * dc + halfW * len) / dr);
  }
  // Along the line: 0 <= pr * dr + pc * dc <= segSq; on a vertical line, tested as above.
  if (dc === 0) {
    const t = (pr * dr) / segSq;
    if (!(t >= 0 && t <= 1)) return [0, -1];
  } else {
    keepBetween((-pr * dr) / dc, (segSq - pr * dr) / dc);
  }
  return [lo, hi];
}

// Number of samples for a line with row/col extent (dr, dc) between snapped ends: one per scan
// step along the longer axis, capped at maxBins.
export function linescanSampleCount(dr: number, dc: number, maxBins: number): number {
  const steps = Math.round(Math.max(Math.abs(dr), Math.abs(dc)));
  return Math.max(1, Math.min(maxBins, steps + 1));
}

// Nearest sample of a position at fraction t (0 = A, 1 = B) along the line, as in Python: the
// offset is snapped to 1e-9 so exact ties are not split by rounding error, then floor(x + 0.5).
export function linescanSampleIndex(t: number, n: number): number {
  if (n <= 1) return 0;
  const offset = Math.min(Math.max(t, 0), 1) * (n - 1);
  const snapped = Math.floor(offset * 1e9 + 0.5) / 1e9;
  return Math.min(Math.floor(snapped + 0.5), n - 1);
}

// Fraction along the line (A = 0, B = 1) of sample i of N; the last is B.
export function sampleFraction(i: number, N: number): number {
  return N > 1 ? i / (N - 1) : 0.5;
}

// Evenly spaced subset of the N samples for the waterfall view, counted back from the tip B, which
// is always included. The step is the one whose curve count is closest to `count` (on a tie, more
// curves); any remainder is a gap of less than one step at the start A.
export function sampledBinIndices(N: number, count: number): number[] {
  if (N <= 0) return [];
  if (count >= N) return Array.from({ length: N }, (_, i) => i);
  if (count <= 1) return [N - 1];
  const curves = (step: number) => Math.floor((N - 1) / step) + 1;   // samples N - 1, N - 1 - step, ...
  let best = 1;
  for (let step = 2; step <= N - 1; step++) {
    if (Math.abs(curves(step) - count) < Math.abs(curves(best) - count)) best = step;
  }
  const n = curves(best);
  return Array.from({ length: n }, (_, j) => N - 1 - (n - 1 - j) * best);
}
