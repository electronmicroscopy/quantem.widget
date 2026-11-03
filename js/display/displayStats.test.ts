import { describe, expect, it } from "vitest";
import { displayStatsWorkerSource, frameStats } from "./displayStats";
import { computeHistogramFromBytes, percentileClip, percentileHistogram, percentileWindow } from "./stats";

// The pre-split percentileClip, kept verbatim as the reference the split must reproduce bit for bit.
function referencePercentileClip(data: ArrayLike<number>, pLow: number, pHigh: number) {
  const len = data.length;
  if (len === 0) return { vmin: 0, vmax: 0, min: 0, max: 0 };
  let min = Infinity, max = -Infinity, finiteCount = 0;
  for (let i = 0; i < len; i++) {
    const value = data[i];
    if (!isFinite(value)) continue;
    finiteCount++;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (finiteCount === 0) return { vmin: 0, vmax: 0, min: 0, max: 0 };
  if (min === max) return { vmin: min, vmax: max, min, max };
  const bins = new Uint32Array(1024);
  const range = max - min;
  const scale = 1024 / range;
  for (let i = 0; i < len; i++) {
    const value = data[i];
    if (isFinite(value)) bins[Math.min(1023, Math.floor((value - min) * scale))]++;
  }
  const quantile = (target: number, fallback: number) => {
    let cumSum = 0;
    for (let i = 0; i < bins.length; i++) {
      const prevSum = cumSum;
      cumSum += bins[i];
      if (cumSum >= target) return min + ((i + (target - prevSum) / Math.max(1, cumSum - prevSum)) / bins.length) * range;
    }
    return fallback;
  };
  return { vmin: quantile(finiteCount * (pLow / 100), min), vmax: quantile(finiteCount * (pHigh / 100), max), min, max };
}

const phaseLike = (n: number) => Float32Array.from({ length: n * n }, (_, i) =>
  Math.sin(i * 0.0137) * 1.3 + 0.2 * Math.cos(i * 0.71) + (i % 97 === 0 ? 40 : 0));

const cases: [string, Float32Array][] = [
  ["phase-like with outliers", phaseLike(96)],
  ["log magnitude", Float32Array.from({ length: 128 * 64 }, (_, i) => Math.log1p(Math.abs(Math.sin(i * 0.3)) * 10 ** (i % 7)))],
  ["non-finite values", Float32Array.from({ length: 4096 }, (_, i) => (i % 31 === 0 ? NaN : i % 53 === 0 ? Infinity : i % 59 === 0 ? -Infinity : Math.cos(i)))],
  ["constant", new Float32Array(500).fill(3.25)],
  ["all NaN", new Float32Array(10).fill(NaN)],
  ["empty", new Float32Array(0)],
];
const windows: [number, number][] = [[1, 99], [0, 100], [0.5, 99.5], [5, 95], [20, 21], [37.2, 88.9]];

describe("percentile histogram and window", () => {
  it.each(cases)("reproduce percentileClip exactly: %s", (_name, data) => {
    const histogram = percentileHistogram(data);
    for (const [lo, hi] of windows) {
      const expected = referencePercentileClip(data, lo, hi);
      expect(percentileWindow(histogram, lo, hi)).toEqual(expected);
      expect(percentileClip(data, lo, hi)).toEqual(expected);
    }
  });
});

describe("display statistics worker", () => {
  it.each(cases)("bins exactly as the input-thread reference: %s", (_name, data) => {
    const messages: { id: number; stats: ReturnType<typeof frameStats> }[] = [];
    const worker = { onmessage: (_event: unknown) => {}, postMessage: (value: { id: number; stats: ReturnType<typeof frameStats> }) => messages.push(value) };
    new Function("self", displayStatsWorkerSource())(worker);
    worker.onmessage({ data: { id: 5, data: data.slice() } });
    expect(messages[0].id).toBe(5);
    expect(messages[0].stats).toEqual(frameStats(data));
    expect(messages[0].stats.displayBins).toEqual(computeHistogramFromBytes(data));
    for (const [lo, hi] of windows) {
      expect(percentileWindow(messages[0].stats.histogram, lo, hi)).toEqual(referencePercentileClip(data, lo, hi));
    }
  });
});
