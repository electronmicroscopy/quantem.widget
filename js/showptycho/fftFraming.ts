/** Preserve the native scan frequency range without cropping the underlying FFT. */
export function fftAutoFrameZoom(
  width: number, height: number, sampling: readonly number[],
  nativeSampling: readonly number[] = [],
): number {
  const valid = (steps: readonly number[]) => steps.length === 2
    && steps.every(step => Number.isFinite(step) && step > 0);
  if (!(width > 0 && height > 0) || !valid(sampling) || !valid(nativeSampling)) return 1;
  // Each axis retains its native Nyquist extent as output sampling increases.
  // The FFT may be padded; use its actual shape and the phase's physical sampling.
  const nativeWidth = width * sampling[1] / nativeSampling[1];
  const nativeHeight = height * sampling[0] / nativeSampling[0];
  return Math.max(1, Math.max(width, height) / Math.max(nativeWidth, nativeHeight));
}

/** A block of a row-major grid `stride` wide. */
export type GridRegion = { x0: number; y0: number; width: number; height: number };

/**
 * The centred block of a DC-centred pw x ph spectrum that the auto frame shows, or null when the
 * frame shows all of it. The FFT panel takes its contrast window from this block only, so an
 * upsampled spectrum's empty outer band (most of the grid at 4x and 8x) does not set the window.
 * Display statistics only: the spectrum itself is never cropped.
 */
export function fftFrameRegion(pw: number, ph: number, zoom: number): GridRegion | null {
  if (!(pw > 0 && ph > 0 && zoom > 1)) return null;
  const width = Math.min(pw, Math.max(1, Math.ceil(pw / zoom)));
  const height = Math.min(ph, Math.max(1, Math.ceil(ph / zoom)));
  if (width === pw && height === ph) return null;
  return { x0: (pw >> 1) - (width >> 1), y0: (ph >> 1) - (height >> 1), width, height };
}

/** A copy of `region` of a row-major grid `stride` wide. */
export function cropGridRegion(data: Float32Array, stride: number, region: GridRegion): Float32Array {
  const out = new Float32Array(region.width * region.height);
  for (let row = 0; row < region.height; row++) {
    const start = (region.y0 + row) * stride + region.x0;
    out.set(data.subarray(start, start + region.width), row * region.width);
  }
  return out;
}
