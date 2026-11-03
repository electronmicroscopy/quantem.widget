import { describe, expect, it } from 'vitest';
import { applyHannWindow2D, fftshift, computeMagnitude, nextPow2, shiftedMagnitude } from '../display/fft';
import { displayFFTInput, fftMagnitudeWorkerSource, interleavedShiftedMagnitude } from './fftDisplay';
import { frameStats } from '../display/displayStats';

describe('full-resolution FFT display', () => {
  it('preserves every shifted log-magnitude pixel in the worker, including float32 rounding', () => {
    // A rectangular Fourier grid with large dynamic range and negative components.
    const width = 256, height = 128;
    const real = Float32Array.from({length: width * height}, (_, i) => Math.sin(i * 0.29) * 10 ** ((i % 13) - 6));
    const imag = Float32Array.from(real, (value, i) => value * Math.cos(i * 0.17));
    const reference = computeMagnitude(real, imag);
    fftshift(reference, width, height);
    for (let i = 0; i < reference.length; i++) reference[i] = Math.log1p(reference[i]);
    const messages: {id: number; magnitude: Float32Array}[] = [];
    const worker = {onmessage: (_event: unknown) => {}, postMessage: (value: {id: number; magnitude: Float32Array}) => messages.push(value)};
    new Function('self', fftMagnitudeWorkerSource())(worker);
    worker.onmessage({data: {id: 17, real, imag, width, height}});
    expect(messages[0].id).toBe(17);
    expect(messages[0].magnitude).toEqual(reference);
    expect(messages[0].magnitude).toEqual(shiftedMagnitude(real, imag, width, height, true));
  });
});

describe('display FFT preparation in the worker', () => {
  it('windows, pads and interleaves exactly as the input-thread reference', () => {
    const width = 96, height = 80;
    const phase = Float32Array.from({length: width * height}, (_, i) => Math.sin(i * 0.013) + 0.1 * Math.cos(i * 0.7));
    // Reference: the previous input-thread preparation followed by the FFT engine's interleave.
    let sum = 0;
    for (let i = 0; i < phase.length; i++) sum += phase[i];
    const centered = Float32Array.from(phase, value => value - sum / phase.length);
    applyHannWindow2D(centered, width, height);
    const pw = nextPow2(width), ph = nextPow2(height);
    const expected = new Float32Array(pw * ph * 2);
    for (let row = 0; row < height; row++)
      for (let col = 0; col < width; col++) expected[2 * (row * pw + col)] = centered[row * width + col];
    const messages: {id: number; complex: Float32Array; pw: number; ph: number}[] = [];
    const worker = {onmessage: (_event: unknown) => {}, postMessage: (value: {id: number; complex: Float32Array; pw: number; ph: number}) => messages.push(value)};
    new Function('self', fftMagnitudeWorkerSource())(worker);
    worker.onmessage({data: {id: 3, kind: 'prepare', data: phase, width, height}});
    expect([messages[0].pw, messages[0].ph]).toEqual([pw, ph]);
    expect(messages[0].complex).toEqual(expected);
    expect(displayFFTInput(phase, width, height).complex).toEqual(expected);
  });

  it('gives the split-plane magnitude for an interleaved spectrum', () => {
    const width = 64, height = 32;
    const complex = Float32Array.from({length: width * height * 2}, (_, i) => Math.sin(i * 0.31) * 10 ** ((i % 9) - 4));
    const real = new Float32Array(width * height), imag = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) { real[i] = complex[2 * i]; imag[i] = complex[2 * i + 1]; }
    expect(interleavedShiftedMagnitude(complex, width, height)).toEqual(shiftedMagnitude(real, imag, width, height, true));
  });

  it('bins the interleaved magnitude in the worker exactly as the input-thread reference', () => {
    const width = 64, height = 32;
    const complex = Float32Array.from({length: width * height * 2}, (_, i) => Math.sin(i * 0.31) * 10 ** ((i % 9) - 4));
    const expected = interleavedShiftedMagnitude(complex, width, height);
    const messages: {id: number; magnitude: Float32Array; stats: ReturnType<typeof frameStats>}[] = [];
    const worker = {onmessage: (_event: unknown) => {}, postMessage: (value: {id: number; magnitude: Float32Array; stats: ReturnType<typeof frameStats>}) => messages.push(value)};
    new Function('self', fftMagnitudeWorkerSource())(worker);
    worker.onmessage({data: {id: 9, kind: 'interleaved', complex, width, height}});
    expect(messages[0].magnitude).toEqual(expected);
    expect(messages[0].stats).toEqual(frameStats(expected));
  });
});
