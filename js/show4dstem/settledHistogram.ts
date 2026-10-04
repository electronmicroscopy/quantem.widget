import type { GPUColormapEngine } from '../colormaps';

/** Read the settled float image summary, abandoning work when a drag or a newer
 * display generation makes those buffers stale. Never dispatch the next stage
 * or publish a result after an asynchronous boundary without checking again.
 */
export async function readSettledCompareHistogram(
  engine: Pick<GPUColormapEngine, 'computeRangeBatch' | 'computeHistogramBatch'>,
  slots: number[],
  logScale: boolean,
  isCurrent: () => boolean,
): Promise<{ min: number; max: number; bins: Float32Array } | null> {
  if (!isCurrent()) return null;
  const rawRanges = await engine.computeRangeBatch(slots);
  if (!isCurrent()) return null;
  const ranges = rawRanges.map(range => {
    if (!Number.isFinite(range.min) || !Number.isFinite(range.max)) return null;
    return logScale ? {
      min: Math.log1p(Math.max(0, range.min)),
      max: Math.log1p(Math.max(0, range.max)),
    } : range;
  }).filter((range): range is {min: number; max: number} => Boolean(range));
  if (!ranges.length) return null;
  let min = Number.POSITIVE_INFINITY, max = Number.NEGATIVE_INFINITY;
  for (const range of ranges) { min = Math.min(min, range.min); max = Math.max(max, range.max); }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
  if (max <= min) max = min + 1e-12;
  if (!isCurrent()) return null;
  const histograms = await engine.computeHistogramBatch(slots, slots.map(() => ({min, max})), logScale);
  if (!isCurrent() || !histograms.length) return null;
  const bins = new Float32Array(256);
  for (const histogram of histograms) {
    for (let i = 0; i < Math.min(256, histogram.length); i++) bins[i] += Number(histogram[i]) || 0;
  }
  return {min, max, bins};
}
