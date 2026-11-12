import { describe, expect, it } from 'vitest';
import { frameStats } from '../display/displayStats';
import { percentileWindow } from '../display/stats';
import { RECORD_BYTES, recordStats } from './gpuFrame';

/** The record the GPU writes for `data`: (min, max, sum, count), then percentileHistogram's 1024 bins. */
function record(data: Float32Array): ArrayBuffer {
  const reference = frameStats(data).histogram;
  const buffer = new ArrayBuffer(RECORD_BYTES);
  new Float32Array(buffer, 0, 4).set([reference.min, reference.max, data.reduce((a, b) => a + b, 0), data.length]);
  if (reference.bins) new Uint32Array(buffer, 16, 1024).set(reference.bins);
  return buffer;
}

describe('GPU frame statistics', () => {
  it('rebuild the CPU display statistics from the 1024 percentile bins', () => {
    const phase = Float32Array.from({ length: 1536 * 96 }, (_, i) => Math.sin(i * 0.0137) * 0.3 + 0.05 * Math.cos(i * 0.71));
    const expected = frameStats(phase);
    const stats = recordStats(record(phase), 0);
    expect(stats.histogram).toEqual(expected.histogram);
    expect(stats.displayBins).toEqual(expected.displayBins);
    expect(percentileWindow(stats.histogram, 1, 99)).toEqual(percentileWindow(expected.histogram, 1, 99));
  });

  it('give a flat image the CPU reference statistics', () => {
    const flat = new Float32Array(64).fill(0.25);
    const expected = frameStats(flat);
    const stats = recordStats(record(flat), 0);
    expect(stats.histogram).toEqual(expected.histogram);
    expect(stats.displayBins).toEqual(expected.displayBins);
  });
});
