import { afterEach, describe, expect, it, vi } from 'vitest';
import { readSettledCompareHistogram } from './settledHistogram';

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(done => { resolve = done; });
  return {promise, resolve};
}
function fixture() {
  const state = {drag: false, pendingSettle: false, generation: 1};
  const generation = state.generation;
  const isCurrent = () => !state.drag && !state.pendingSettle && state.generation === generation;
  const engine = {
    computeRangeBatch: vi.fn(async () => [{min: 2, max: 4}, {min: 6, max: 8}]),
    computeHistogramBatch: vi.fn(async () => [[1, 1], [0, 2]]),
  };
  return {state, engine, isCurrent};
}
afterEach(() => vi.useRealTimers());

describe('settled comparison histograms', () => {
  it('does not dispatch an old 120ms timer after a quick drag and release', async () => {
    vi.useFakeTimers();
    const f = fixture();
    const task = new Promise(resolve => setTimeout(() => {
      void readSettledCompareHistogram(f.engine, [0, 1], false, f.isCurrent).then(resolve);
    }, 120));
    await vi.advanceTimersByTimeAsync(40);
    f.state.drag = true; f.state.pendingSettle = true; f.state.generation++;
    await vi.advanceTimersByTimeAsync(40);
    f.state.drag = false; // Release does not imply that float conversion settled.
    await vi.advanceTimersByTimeAsync(40);
    expect(await task).toBeNull();
    expect(f.engine.computeRangeBatch).not.toHaveBeenCalled();
    expect(f.engine.computeHistogramBatch).not.toHaveBeenCalled();
    // A new log-change generation still cannot read the pending float image.
    expect(await readSettledCompareHistogram(f.engine, [0, 1], true,
      () => !f.state.drag && !f.state.pendingSettle)).toBeNull();
    expect(f.engine.computeRangeBatch).not.toHaveBeenCalled();
  });

  it('abandons a range read interrupted by drag before dispatching histograms', async () => {
    const f = fixture(), range = deferred<{min: number; max: number}[]>();
    f.engine.computeRangeBatch.mockImplementationOnce(() => range.promise);
    const task = readSettledCompareHistogram(f.engine, [0, 1], false, f.isCurrent);
    expect(f.engine.computeRangeBatch).toHaveBeenCalledTimes(1);
    f.state.drag = true; f.state.pendingSettle = true; f.state.generation++;
    range.resolve([{min: 2, max: 8}]);
    expect(await task).toBeNull();
    expect(f.engine.computeHistogramBatch).not.toHaveBeenCalled();
  });

  it('discards an in-flight histogram when a drag or log generation replaces it', async () => {
    const f = fixture(), bins = deferred<number[][]>();
    f.engine.computeHistogramBatch.mockImplementationOnce(() => bins.promise);
    const task = readSettledCompareHistogram(f.engine, [0, 1], false, f.isCurrent);
    await Promise.resolve();
    expect(f.engine.computeHistogramBatch).toHaveBeenCalledTimes(1);
    f.state.generation++;
    bins.resolve([[99]]);
    expect(await task).toBeNull();
  });

  it('resumes on settled float images with exact merged bins and current log ranges', async () => {
    const f = fixture();
    f.state.pendingSettle = true;
    const current = () => !f.state.drag && !f.state.pendingSettle;
    expect(await readSettledCompareHistogram(f.engine, [0, 1], false, current)).toBeNull();
    f.state.pendingSettle = false; // Settled float normalization has been queued.
    for (const log of [false, true]) {
      const result = await readSettledCompareHistogram(f.engine, [0, 1], log, current);
      expect(result?.min).toBe(log ? Math.log1p(2) : 2);
      expect(result?.max).toBe(log ? Math.log1p(8) : 8);
      expect(Array.from(result!.bins.slice(0, 3))).toEqual([1, 3, 0]);
      expect(result!.bins.reduce((sum, n) => sum + n, 0)).toBe(4);
      expect(f.engine.computeHistogramBatch).toHaveBeenLastCalledWith([0, 1],
        [{min: result!.min, max: result!.max}, {min: result!.min, max: result!.max}], log);
    }
  });
});
