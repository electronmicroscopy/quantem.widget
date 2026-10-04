import type { Source112ResidentSet } from "../.generated/engine/detector/compute/webgpu/source112";

/** Integrate exact counts, then retain their mean for painting or refresh floats. */
export function source112MeanDelta(
  source: Pick<Source112ResidentSet, "integrate" | "normalizeDisplayBuffers">,
  mask: Uint32Array,
  previous: GPUBuffer[],
  retainMean?: (area: number) => boolean,
) {
  const delta = source.integrate(mask);
  // The caller supplies the effective detector mask, including native validity.
  const area = Math.max(1, mask.reduce((sum, value) => sum + (value ? 1 : 0), 0));
  let retained = false;
  try { retained = retainMean?.(area) ?? false; }
  finally { if (!retained) source.normalizeDisplayBuffers(previous, area); }
  return {
    buffers: previous,
    path: "delta" as const,
    addedPixels: delta.added,
    removedPixels: delta.removed,
  };
}

export type CompareCountImages = {
  images: ReadonlyMap<number, import("../colormaps").Uint32ImageView>;
  isCurrent: () => boolean;
  refreshFloat: () => void;
};
/** Return accepted panels when deferred, submitted panels otherwise.
 * Null clears views after a full refresh; invalidate first preserves their mean
 * in stable float slots before another operation changes the shared counts.
 */
export type CompareGpuRenderer = (counts?: CompareCountImages | null | "invalidate", deferPaint?: boolean) => number;

/** One queued canvas paint; newer scientific submissions replace its callback.
 * The callback must select current borrowed views and synchronously submit the
 * render: this keeps their mean divisor ordered with their GPU count buffer.
 */
export function createComparePaintScheduler(
  requestFrame: (callback: FrameRequestCallback) => number = requestAnimationFrame,
  cancelFrame: (handle: number) => void = cancelAnimationFrame,
) {
  let handle: number | null = null;
  let latest: { paint: () => void; onError?: (error: unknown) => void } | null = null;
  return {
    schedule(paint: () => void, onError?: (error: unknown) => void) {
      latest = { paint, onError };
      if (handle !== null) return;
      handle = requestFrame(() => {
        handle = null;
        const draw = latest;
        latest = null;
        if (!draw) return;
        try { draw.paint(); }
        catch (error) {
          if (!draw.onError) throw error;
          draw.onError(error);
        }
      });
    },
    cancel() {
      if (handle !== null) cancelFrame(handle);
      handle = null;
      latest = null;
    },
  };
}

/** Retain a current borrowed image for repaints, or refresh its float fallback.
 * A replaced/disposed source is discarded without touching its retired buffers.
 */
export function countImagesForRender(counts: CompareCountImages | null, shared: boolean): CompareCountImages | null {
  if (!counts?.isCurrent()) return null;
  if (shared) return counts;
  counts.refreshFloat();
  return null;
}

/** Detector skip decisions depend on included pixels, not mask storage values. */
export function sameDetectorMaskSupport(left: Uint32Array, right: Uint32Array): boolean {
  if (left.length !== right.length) return false;
  for (let i = 0; i < left.length; i++) {
    if (Boolean(left[i]) !== Boolean(right[i])) return false;
  }
  return true;
}
