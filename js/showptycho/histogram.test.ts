import { describe, expect, it } from 'vitest';
import { histogramBarHeights } from './histogram';

describe('FFT histogram count scaling', () => {
  it('reveals rare peaks without dropping background or moving intensity bins', () => {
    const bins = [1, 0, 0.001, 0.00001];
    const original = [...bins];
    const counts = 100000 + 100 + 1;
    const linear = histogramBarHeights(bins, 4, 'linear', counts);
    const log = histogramBarHeights(bins, 4, 'log', counts);
    expect(linear).toEqual(bins);
    expect(log[0]).toBe(1);
    expect(log[1]).toBe(0);
    expect(log[2]).toBeCloseTo(Math.log1p(100) / Math.log1p(100000));
    expect(log[3]).toBeCloseTo(Math.log1p(1) / Math.log1p(100000));
    expect(bins).toEqual(original);
  });

  it('includes all bins when reducing, and handles empty frames', () => {
    expect(histogramBarHeights([1, 2, 3, 4, 5], 2, 'linear', 15)).toEqual([6 / 9, 1]);
    expect(histogramBarHeights([0, 0], 2, 'log', 0)).toEqual([0, 0]);
    expect(histogramBarHeights([], 2, 'log', 0)).toEqual([0, 0]);
  });
});
