/**
 * Moving-average width as an integer in [1, 15], the avg slider range. A
 * missing or non-finite value means no averaging, so the cache keys and the
 * averaged frames always agree on one width.
 */
export function normalizedAverageWindow(value: unknown): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 1;
  return Math.max(1, Math.min(15, Math.round(parsed)));
}

/**
 * The embedded Show3D display stack holds raw frames, so the browser applies
 * the frame difference and the moving average.
 */
export function shouldApplyClientDifference(diffMode: string): boolean {
  return diffMode !== "off";
}

/** True when the shown frame is not the raw stack frame (difference or average). */
export function requiresClientFrameTransform({
  diffMode,
  avgWindow,
}: {
  diffMode: string;
  avgWindow: unknown;
}): boolean {
  return normalizedAverageWindow(avgWindow) > 1
    || shouldApplyClientDifference(diffMode);
}

/** Cache identity for an asynchronously browser-filtered live frame. */
export function browserFilterCacheKey({
  frameIndex,
  frameSeq,
  mode,
  sigma,
  bin,
  avgWindow,
  diffMode,
  panels = 1,
}: {
  frameIndex: number;
  frameSeq: number;
  mode: string;
  sigma: number;
  bin: number;
  avgWindow: unknown;
  diffMode: string;
  panels?: number;
}): string {
  return `${Math.round(frameIndex)}:${frameSeq}:${mode}:${sigma}:${bin}:${normalizedAverageWindow(avgWindow)}:${diffMode}:${Math.max(1, Math.round(panels))}`;
}

/** Show3D's centered, full-width window; edges slide inward without wrapping. */
export function temporalAverageFrameIndices(frameIndex: number, count: number, windowSize: number): number[] {
  const frameCount = Math.max(1, Math.round(count || 1));
  const windowLength = Math.min(frameCount, normalizedAverageWindow(windowSize));
  const center = Math.max(0, Math.min(frameCount - 1, Math.round(frameIndex)));
  const start = Math.max(0, Math.min(frameCount - windowLength, center - Math.floor(windowLength / 2)));
  return Array.from({length: windowLength}, (_, offset) => start + offset);
}
