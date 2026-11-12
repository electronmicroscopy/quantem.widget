import { expect, test } from 'vitest';
import { cropGridRegion, fftAutoFrameZoom, fftFrameRegion } from './fftFraming';

test('Native frequency range is identical at 1x, 2x, padded 3x and 4x', () => {
  const step = 0.373, q = 0.8;
  for (const factor of [1, 2, 3, 4]) {
    const fft = 2 ** Math.ceil(Math.log2(512 * factor));
    const sampling = step / factor;
    const zoom = fftAutoFrameZoom(fft, fft, [sampling, sampling], [step, step]);
    expect(zoom).toBeCloseTo(factor, 12);
    expect(0.5 / sampling / zoom).toBeCloseTo(0.5 / step, 12);
    expect(q * sampling * zoom).toBeCloseTo(q * step, 12);
  }
});

test('rectangular anisotropic scans retain their original frequency extents', () => {
  expect(fftAutoFrameZoom(512, 256, [0.06, 0.04], [0.06, 0.04])).toBe(1);
  expect(fftAutoFrameZoom(2048, 1024, [0.02, 0.04 / 3], [0.06, 0.04])).toBeCloseTo(3);
});

test('uncalibrated arrays and invalid metadata retain the full FFT', () => {
  for (const sampling of [[], [0, 0], [NaN, 1]]) {
    expect(fftAutoFrameZoom(512, 512, sampling, [0.4, 0.4])).toBe(1);
    expect(fftAutoFrameZoom(512, 512, [0.1, 0.1], sampling)).toBe(1);
  }
  expect(fftAutoFrameZoom(512, 512, [0.1, 0.1])).toBe(1);
  expect(fftAutoFrameZoom(0, 0, [0.1, 0.1], [0.4, 0.4])).toBe(1);
});

test('the contrast region is the auto frame\'s centred block, and none at 1x', () => {
  expect(fftFrameRegion(512, 512, 1)).toBeNull();
  expect(fftFrameRegion(1024, 1024, 8)).toEqual({ x0: 448, y0: 448, width: 128, height: 128 });
  expect(fftFrameRegion(1024, 512, 4)).toEqual({ x0: 384, y0: 192, width: 256, height: 128 });
  const grid = Float32Array.from({ length: 16 }, (_, i) => i);
  expect(Array.from(cropGridRegion(grid, 4, { x0: 1, y0: 1, width: 2, height: 2 }))).toEqual([5, 6, 9, 10]);
});
