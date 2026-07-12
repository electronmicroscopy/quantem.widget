export function normalizedAverageWindow(value: unknown): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 1;
  return Math.max(1, Math.min(15, Math.round(parsed)));
}

/**
 * Live Show3D transport is already differenced by Python/the frame server.
 * Standalone HTML stores raw frames, so only the offline path applies the
 * difference in JavaScript. Moving average remains a browser-side transform.
 */
export function shouldApplyClientDifference(
  offline: boolean,
  diffMode: string,
): boolean {
  return offline && diffMode !== "off";
}

export function requiresClientFrameTransform({
  offline,
  diffMode,
  avgWindow,
}: {
  offline: boolean;
  diffMode: string;
  avgWindow: unknown;
}): boolean {
  return normalizedAverageWindow(avgWindow) > 1
    || shouldApplyClientDifference(offline, diffMode);
}

/** Separate full-resolution panel endpoints currently expose one frame only. */
export function supportsClientAverage(separatePanelFrames: boolean): boolean {
  return !separatePanelFrames;
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
