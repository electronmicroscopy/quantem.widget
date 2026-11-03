// Show1D plots traces of up to millions of points, and two costs grew with the
// point count on every redraw (each wheel notch): the y range of the visible x
// window and one lineTo per point. With x sorted ascending both become work per
// pixel column: a binary search finds a column's points, and a min/max pyramid
// over the point index gives their extremes, so a redraw costs about
// O(columns * log points) instead of O(points).

// Points per finest pyramid block: windows shorter than this are scanned.
const LEAF = 16;

// Pixel columns are split this many times when decimating: each sub-column keeps
// its first, lowest, highest and last point, so the stroked polyline reaches the
// same extremes and leaves every column where the full polyline does.
const COLUMN_SPLIT = 4;

type Pyramid = { argMin: Int32Array; argMax: Int32Array }[];

export type TraceIndex = {
  yData: Float32Array;
  xData: Float32Array;
  nTraces: number;
  nPoints: number;
  logScale: boolean;
  /** x is finite and non-decreasing, so x windows are index windows. */
  sorted: boolean;
  /** Per trace: level k holds, per block of LEAF * 2^k points, the index of the lowest and of the highest drawable y (-1: none). */
  pyramids: Pyramid[];
  /** Per trace: indices of the undrawable points, which break the line (ascending). */
  breaks: Int32Array[];
};

function drawable(y: number, logScale: boolean): boolean {
  return Number.isFinite(y) && (!logScale || y > 0);
}

function xAt(index: TraceIndex, point: number): number {
  return index.xData.length > point ? index.xData[point] : point;
}

/** Build the per-trace pyramids once per data or log-scale change (one pass over the points). */
export function indexTraces(yData: Float32Array, xData: Float32Array, nTraces: number, nPoints: number, logScale: boolean): TraceIndex {
  const index: TraceIndex = { yData, xData, nTraces, nPoints, logScale, sorted: true, pyramids: [], breaks: [] };
  let previous = -Infinity;
  for (let point = 0; point < nPoints && index.sorted; point += 1) {
    const x = xAt(index, point);
    index.sorted = Number.isFinite(x) && x >= previous;
    previous = x;
  }
  if (!index.sorted) return index;
  for (let trace = 0; trace < nTraces; trace += 1) {
    const ys = yData.subarray(trace * nPoints, (trace + 1) * nPoints);
    const blocks = Math.ceil(nPoints / LEAF);
    const leafMin = new Int32Array(blocks).fill(-1);
    const leafMax = new Int32Array(blocks).fill(-1);
    const breaks: number[] = [];
    for (let point = 0; point < nPoints; point += 1) {
      const y = ys[point];
      if (!drawable(y, logScale)) {
        breaks.push(point);
        continue;
      }
      const block = (point / LEAF) | 0;
      if (leafMin[block] < 0 || y < ys[leafMin[block]]) leafMin[block] = point;
      if (leafMax[block] < 0 || y > ys[leafMax[block]]) leafMax[block] = point;
    }
    const pyramid: Pyramid = [{ argMin: leafMin, argMax: leafMax }];
    for (let below = pyramid[0]; below.argMin.length > 1; below = pyramid[pyramid.length - 1]) {
      const size = Math.ceil(below.argMin.length / 2);
      const argMin = new Int32Array(size);
      const argMax = new Int32Array(size);
      for (let block = 0; block < size; block += 1) {
        const left = 2 * block;
        const right = Math.min(left + 1, below.argMin.length - 1);
        argMin[block] = lower(ys, below.argMin[left], below.argMin[right]);
        argMax[block] = higher(ys, below.argMax[left], below.argMax[right]);
      }
      pyramid.push({ argMin, argMax });
    }
    index.pyramids.push(pyramid);
    index.breaks.push(Int32Array.from(breaks));
  }
  return index;
}

function lower(ys: Float32Array, a: number, b: number): number {
  return a < 0 ? b : b < 0 ? a : ys[b] < ys[a] ? b : a;
}

function higher(ys: Float32Array, a: number, b: number): number {
  return a < 0 ? b : b < 0 ? a : ys[b] > ys[a] ? b : a;
}

/** Indices of the lowest and highest drawable y of a trace in points [start, end), -1 when there is none. */
function extremes(index: TraceIndex, trace: number, start: number, end: number): [number, number] {
  const ys = index.yData.subarray(trace * index.nPoints, (trace + 1) * index.nPoints);
  let argMin = -1;
  let argMax = -1;
  const scan = (from: number, to: number) => {
    for (let point = from; point < to; point += 1) {
      if (!drawable(ys[point], index.logScale)) continue;
      argMin = lower(ys, argMin, point);
      argMax = higher(ys, argMax, point);
    }
  };
  let left = Math.ceil(start / LEAF);
  let right = Math.floor(end / LEAF);
  if (left >= right) {
    scan(start, end);
    return [argMin, argMax];
  }
  scan(start, left * LEAF);
  scan(right * LEAF, end);
  const pyramid = index.pyramids[trace];
  for (let level = 0; left < right; level += 1, left >>= 1, right >>= 1) {
    if (left & 1) {
      argMin = lower(ys, argMin, pyramid[level].argMin[left]);
      argMax = higher(ys, argMax, pyramid[level].argMax[left]);
      left += 1;
    }
    if (right & 1) {
      right -= 1;
      argMin = lower(ys, argMin, pyramid[level].argMin[right]);
      argMax = higher(ys, argMax, pyramid[level].argMax[right]);
    }
  }
  return [argMin, argMax];
}

/** First point whose x is >= value (inclusive) or > value (exclusive): the bounds of an x window on sorted x. */
function searchX(index: TraceIndex, value: number, inclusive: boolean, start = 0, end = index.nPoints): number {
  while (start < end) {
    const middle = (start + end) >>> 1;
    const x = xAt(index, middle);
    if (x < value || (!inclusive && x === value)) start = middle + 1;
    else end = middle;
  }
  return start;
}

/** Points [start, end) with xMin <= x <= xMax, for sorted x. */
export function pointWindow(index: TraceIndex, xMin: number, xMax: number): [number, number] {
  const start = searchX(index, xMin, true);
  return [start, Math.max(start, searchX(index, xMax, false, start))];
}

/** y range of the drawable points of the visible traces whose x lies in xRange, padded for the plot frame. */
export function yExtent(index: TraceIndex, xRange: [number, number], hiddenTraces?: Set<number>): [number, number] {
  const { yData, nTraces, nPoints, logScale } = index;
  let lo = Infinity;
  let hi = -Infinity;
  const [start, end] = index.sorted ? pointWindow(index, xRange[0], xRange[1]) : [0, 0];
  for (let trace = 0; trace < nTraces; trace += 1) {
    if (hiddenTraces?.has(trace)) continue;
    const offset = trace * nPoints;
    if (index.sorted) {
      const [argMin, argMax] = extremes(index, trace, start, end);
      if (argMin >= 0) lo = Math.min(lo, yData[offset + argMin]);
      if (argMax >= 0) hi = Math.max(hi, yData[offset + argMax]);
      continue;
    }
    for (let point = 0; point < nPoints; point += 1) {
      const x = xAt(index, point);
      if (x < xRange[0] || x > xRange[1]) continue;
      const y = yData[offset + point];
      if (!drawable(y, logScale)) continue;
      if (y < lo) lo = y;
      if (y > hi) hi = y;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return logScale ? [1e-6, 1] : [0, 1];
  if (lo === hi) {
    const pad = Math.max(Math.abs(lo) * 0.05, logScale ? Math.max(lo * 0.5, 1e-9) : 1);
    return [Math.max(logScale ? Number.MIN_VALUE : -Infinity, lo - pad), hi + pad];
  }
  if (logScale) return [Math.max(lo / 1.25, Number.MIN_VALUE), hi * 1.25];
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

export type PlotMapping = {
  /** Plot frame in CSS px and the x range it shows. */
  left: number;
  plotW: number;
  xMin: number;
  xMax: number;
  /** Device pixels per CSS px: decimation works on device pixel columns. */
  dpr: number;
  toX: (x: number) => number;
  toY: (y: number) => number;
};

/**
 * Point index where each decimation column of the plot starts, for sorted x, plus
 * the end of the last one; shared by every trace, since they share x. Null when
 * no column holds more than four points: then every point is drawn.
 */
export function columnStarts(index: TraceIndex, plot: PlotMapping): Int32Array | null {
  if (!index.sorted) return null;
  const [start, end] = pointWindow(index, plot.xMin, plot.xMax);
  const span = Math.max(plot.xMax - plot.xMin, 1e-12);
  const first = Math.floor(plot.left * plot.dpr * COLUMN_SPLIT);
  const last = Math.ceil((plot.left + plot.plotW) * plot.dpr * COLUMN_SPLIT);
  const starts = new Int32Array(last - first + 2);
  starts[0] = start;
  let point = start;
  for (let column = first + 1; column <= last; column += 1) {
    const x = plot.xMin + ((column / (plot.dpr * COLUMN_SPLIT) - plot.left) / plot.plotW) * span;
    point = searchX(index, x, true, point, end);
    starts[column - first] = point;
  }
  starts[last - first + 1] = end;
  for (let column = 0; column + 1 < starts.length; column += 1) {
    if (starts[column + 1] - starts[column] > 4) return starts;
  }
  return null;
}

/**
 * Append one trace's polyline to the current path (moveTo / lineTo; the caller
 * strokes). Undrawable points break the line, as in the full polyline. With
 * decimation columns: per unbroken run of points in a column its first, lowest,
 * highest and last point (every point of a run of at most four). Without: every
 * point in the x range.
 */
export function traceToPath(ctx: CanvasRenderingContext2D, index: TraceIndex, trace: number, plot: PlotMapping, columns: Int32Array | null): void {
  const { nPoints, logScale } = index;
  const ys = index.yData.subarray(trace * nPoints, (trace + 1) * nPoints);
  let active = false;
  let previous = -1;
  const visit = (point: number) => {
    if (point === previous) return;
    previous = point;
    const px = plot.toX(xAt(index, point));
    const py = plot.toY(ys[point]);
    if (active) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
    active = true;
  };
  if (!columns) {
    for (let point = 0; point < nPoints; point += 1) {
      const x = xAt(index, point);
      if (!Number.isFinite(x) || !drawable(ys[point], logScale)) {
        active = false;
        continue;
      }
      if (x >= plot.xMin && x <= plot.xMax) visit(point);
    }
    return;
  }
  const breaks = index.breaks[trace];
  const decimate = (start: number, end: number) => {
    if (end - start <= 4) {
      for (let point = start; point < end; point += 1) visit(point);
      return;
    }
    const [argMin, argMax] = extremes(index, trace, start, end);
    visit(start);
    visit(Math.min(argMin, argMax));
    visit(Math.max(argMin, argMax));
    visit(end - 1);
  };
  let nextBreak = 0;
  while (nextBreak < breaks.length && breaks[nextBreak] < columns[0]) nextBreak += 1;
  for (let column = 0; column + 1 < columns.length; column += 1) {
    let start = columns[column];
    const end = columns[column + 1];
    for (; nextBreak < breaks.length && breaks[nextBreak] < end; nextBreak += 1) {
      decimate(start, breaks[nextBreak]);
      active = false;
      start = breaks[nextBreak] + 1;
    }
    decimate(start, end);
  }
}
