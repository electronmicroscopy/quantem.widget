import {
  applyHannWindow2D, computeMagnitude, displayFFTInput, fftshift, interleavedShiftedMagnitude, nextPow2, shiftedMagnitude,
} from '../display/fft';
import { frameStats, frameStatsWorkerCall, frameStatsWorkerFunctions, type FrameStats } from '../display/displayStats';
import { cropGridRegion, type GridRegion } from './fftFraming';

export { displayFFTInput, interleavedShiftedMagnitude };

/** Reuse the display reference verbatim, including both float32 roundings. */
export function fftMagnitudeWorkerSource(): string {
  const functions = [applyHannWindow2D, nextPow2, computeMagnitude, fftshift, shiftedMagnitude,
    displayFFTInput, interleavedShiftedMagnitude, cropGridRegion];
  return `${functions.map(fn => fn.toString()).join('\n')}
${frameStatsWorkerFunctions()}
self.onmessage = ({data}) => {
  if (data.kind === 'prepare') {
    const input = ${displayFFTInput.name}(data.data, data.width, data.height);
    self.postMessage({id: data.id, complex: input.complex, pw: input.pw, ph: input.ph}, [input.complex.buffer]);
    return;
  }
  const magnitude = data.kind === 'interleaved'
    ? ${interleavedShiftedMagnitude.name}(data.complex, data.width, data.height)
    : ${shiftedMagnitude.name}(data.real, data.imag, data.width, data.height, true);
  // The panel's contrast window and histogram bins, binned here instead of on the input thread,
  // from the framed block only when a region is given.
  const statsInput = data.region ? ${cropGridRegion.name}(magnitude, data.width, data.region) : magnitude;
  const stats = data.kind === 'interleaved' ? ${frameStatsWorkerCall('statsInput')} : null;
  const moved = [magnitude.buffer];
  if (stats && stats.histogram.bins) moved.push(stats.histogram.bins.buffer);
  self.postMessage({id: data.id, magnitude, stats}, moved);
};`;
}

/** Keep full-grid FFT display postprocessing off the notebook's input thread. */
export class FFTMagnitudeDisplay {
  private worker: Worker | null = null;
  private serial = 0;
  private pending = new Map<number, {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    resolve: (reply: any) => void;
    reject: (error: Error) => void;
  }>();
  private disposed = false;

  constructor() {
    if (typeof Worker === 'undefined') return;
    const url = URL.createObjectURL(new Blob([fftMagnitudeWorkerSource()], { type: 'text/javascript' }));
    try {
      this.worker = new Worker(url);
      this.worker.onmessage = ({ data }) => {
        const job = this.pending.get(data.id);
        this.pending.delete(data.id);
        job?.resolve(data);
      };
      this.worker.onerror = event => {
        this.fail(new Error(`FFT display worker failed: ${event.message}`));
      };
    } catch {
      // Hosts that disallow blob workers retain the existing display reference.
      this.worker = null;
    } finally { URL.revokeObjectURL(url); }
  }

  magnitude(real: Float32Array, imag: Float32Array, width: number, height: number): Promise<Float32Array> {
    if (this.disposed) return Promise.reject(new Error('FFT display is closed.'));
    if (!this.worker) return Promise.resolve(shiftedMagnitude(real, imag, width, height, true));
    const id = ++this.serial;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: reply => resolve(reply.magnitude), reject });
      // The FFT owns these temporary spectra. Move their full buffers instead
      // of copying them; the original phase remains available to the notebook.
      this.worker!.postMessage({ id, real, imag, width, height }, [real.buffer, imag.buffer]);
    });
  }

  /** displayFFTInput off the input thread; the phase is copied, it stays the caller's. */
  prepare(data: Float32Array, width: number, height: number): Promise<{ complex: Float32Array; pw: number; ph: number }> {
    if (this.disposed) return Promise.reject(new Error('FFT display is closed.'));
    if (!this.worker) return Promise.resolve(displayFFTInput(data, width, height));
    const id = ++this.serial;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: reply => resolve({ complex: reply.complex, pw: reply.pw, ph: reply.ph }), reject });
      this.worker!.postMessage({ id, kind: 'prepare', data, width, height });
    });
  }

  /** interleavedShiftedMagnitude and its display statistics off the input thread; the spectrum
   *  buffer moves to the worker. The statistics cover `region` of the magnitude when given. */
  magnitudeInterleaved(complex: Float32Array, width: number, height: number, region: GridRegion | null = null): Promise<{ magnitude: Float32Array; stats: FrameStats }> {
    if (this.disposed) return Promise.reject(new Error('FFT display is closed.'));
    if (!this.worker) {
      const magnitude = interleavedShiftedMagnitude(complex, width, height);
      return Promise.resolve({ magnitude, stats: frameStats(region ? cropGridRegion(magnitude, width, region) : magnitude) });
    }
    const id = ++this.serial;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: reply => resolve({ magnitude: reply.magnitude, stats: reply.stats }), reject });
      this.worker!.postMessage({ id, kind: 'interleaved', complex, width, height, region }, [complex.buffer]);
    });
  }

  private fail(error: Error) {
    for (const job of this.pending.values()) job.reject(error);
    this.pending.clear();
    this.worker?.terminate();
    this.worker = null;
  }

  dispose() {
    this.disposed = true;
    this.fail(new Error('FFT display is closed.'));
  }
}
