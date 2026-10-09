// The trace index replaces two passes over every point per redraw. Its y range
// must equal the full scan's, and its decimated polyline must reach every
// column's lowest and highest point and keep the line breaks of the full one.
import { expect, it } from "vitest";
import { columnStarts, indexTraces, traceToPath, yExtent } from "./traceIndex";

function noisyTraces(nTraces: number, nPoints: number, withGaps: boolean) {
  let seed = 7;
  const random = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const xData = new Float32Array(nPoints).map((_, point) => point * 0.01);
  const yData = new Float32Array(nTraces * nPoints).map((_, value) => Math.sin(value * 0.003) * 50 + random() * 10 - 2);
  if (withGaps) for (let value = 0; value < yData.length; value += 977) yData[value] = Number.NaN;
  return { xData, yData };
}

function scannedYRange(yData: Float32Array, xData: Float32Array, nTraces: number, nPoints: number, xRange: [number, number], logScale: boolean) {
  let lo = Infinity;
  let hi = -Infinity;
  for (let trace = 0; trace < nTraces; trace += 1) {
    for (let point = 0; point < nPoints; point += 1) {
      const y = yData[trace * nPoints + point];
      if (xData[point] < xRange[0] || xData[point] > xRange[1] || !Number.isFinite(y) || (logScale && y <= 0)) continue;
      lo = Math.min(lo, y);
      hi = Math.max(hi, y);
    }
  }
  return logScale ? [lo / 1.25, hi * 1.25] : [lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.08];
}

function recordPath() {
  const calls: { name: string; px: number; py: number }[] = [];
  const ctx = {
    moveTo: (px: number, py: number) => calls.push({ name: "moveTo", px, py }),
    lineTo: (px: number, py: number) => calls.push({ name: "lineTo", px, py }),
  } as unknown as CanvasRenderingContext2D;
  return { ctx, calls };
}

function plotFor(xRange: [number, number]) {
  const left = 64;
  const plotW = 500;
  return {
    left, plotW, xMin: xRange[0], xMax: xRange[1], dpr: 1,
    toX: (x: number) => left + ((x - xRange[0]) / (xRange[1] - xRange[0])) * plotW,
    toY: (y: number) => -y,
  };
}

it("finds the y range of the visible x window exactly as a scan of every point", () => {
  const nTraces = 3;
  const nPoints = 20_011;
  const { xData, yData } = noisyTraces(nTraces, nPoints, true);
  for (const logScale of [false, true]) {
    const index = indexTraces(yData, xData, nTraces, nPoints, logScale);
    expect(index.sorted).toBe(true);
    for (const xRange of [[0, 200.1], [3.217, 3.5], [17.5, 150], [-5, 0.05], [199.9, 250]] as [number, number][]) {
      expect(yExtent(index, xRange)).toEqual(scannedYRange(yData, xData, nTraces, nPoints, xRange, logScale));
    }
  }
  const shuffled = Float32Array.from(xData).reverse();
  const unsorted = indexTraces(yData, shuffled, nTraces, nPoints, false);
  expect(unsorted.sorted).toBe(false);
  expect(yExtent(unsorted, [17.5, 150])).toEqual(scannedYRange(yData, shuffled, nTraces, nPoints, [17.5, 150], false));
});

it("decimates a dense trace to its column extremes and keeps its breaks", () => {
  const nPoints = 200_000;
  const { xData, yData } = noisyTraces(1, nPoints, true);
  const index = indexTraces(yData, xData, 1, nPoints, false);
  const plot = plotFor([0, nPoints * 0.01]);
  const columns = columnStarts(index, plot)!;
  const { ctx, calls } = recordPath();
  traceToPath(ctx, index, 0, plot, columns);
  expect(calls.length).toBeLessThan(nPoints / 20);
  const reached = new Set(calls.map((call) => -call.py));
  const lineStarts = Array.from(yData).filter((y, point) => Number.isFinite(y) && (point === 0 || !Number.isFinite(yData[point - 1]))).length;
  expect(calls.filter((call) => call.name === "moveTo").length).toBe(lineStarts);
  for (let column = 0; column + 1 < columns.length; column += 1) {
    const values = Array.from(yData.subarray(columns[column], columns[column + 1])).filter(Number.isFinite);
    if (!values.length) continue;
    expect(reached.has(Math.min(...values))).toBe(true);
    expect(reached.has(Math.max(...values))).toBe(true);
  }
});

it("draws every point once a zoom leaves at most four per column", () => {
  const nPoints = 200_000;
  const { xData, yData } = noisyTraces(1, nPoints, true);
  const index = indexTraces(yData, xData, 1, nPoints, false);
  const plot = plotFor([1000, 1004]);
  expect(columnStarts(index, plot)).toBeNull();
  const { ctx, calls } = recordPath();
  traceToPath(ctx, index, 0, plot, null);
  const inRange = Array.from(xData.keys()).filter((point) => xData[point] >= 1000 && xData[point] <= 1004 && Number.isFinite(yData[point]));
  expect(calls.map((call) => -call.py)).toEqual(inRange.map((point) => yData[point]));
});
