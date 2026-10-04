import { describe, expect, it, vi } from "vitest";
import { source112MeanDelta, sameDetectorMaskSupport, countImagesForRender, createComparePaintScheduler, type CompareCountImages } from "./source112MeanDelta";

describe("source112 mean display deltas", () => {
  it("reuses owned displays and normalizes new counts exactly once per update", () => {
    const rawPatterns = [[100, 20, 1], [200, 40, 2]];
    const counts = new Uint32Array(2);
    const display = new Float32Array([NaN, 999999]);
    const owned = {} as GPUBuffer;
    const previous = [owned];
    const events: string[] = [];
    let last = new Uint32Array(3);
    const source = {
      integrate: vi.fn((mask: Uint32Array) => {
        events.push("integrate");
        let added = 0, removed = 0;
        for (let q = 0; q < mask.length; q++) {
          if (mask[q] && !last[q]) added++;
          if (!mask[q] && last[q]) removed++;
        }
        rawPatterns.forEach((pattern, scan) => {
          counts[scan] = pattern.reduce((sum, value, q) => sum + (mask[q] ? value : 0), 0);
        });
        last = mask.slice();
        return { added, removed, full: false };
      }),
      normalizeDisplayBuffers: vi.fn((buffers: GPUBuffer[], area: number) => {
        events.push("normalize");
        expect(buffers).toBe(previous);
        counts.forEach((value, scan) => { display[scan] = value / area; });
      }),
      imageBuffersF32: vi.fn(() => { throw new Error("Unexpected preliminary conversion"); }),
    };
    const effective = new Uint32Array([1, 1, 0]);
    const first = source112MeanDelta(source, effective, previous);
    expect(first).toEqual({ buffers: previous, path: "delta", addedPixels: 2, removedPixels: 0 });
    expect(first.buffers[0]).toBe(owned);
    expect(Array.from(counts)).toEqual([120, 240]);
    expect(Array.from(display)).toEqual([60, 120]);
    expect(effective).toEqual(new Uint32Array([1, 1, 0]));

    // Change both the count image and its divisor; stale display values cannot
    // be incremented or divided again to obtain this independent result.
    source112MeanDelta(source, new Uint32Array([0, 0, 1]), previous);
    expect(Array.from(counts)).toEqual([1, 2]);
    expect(Array.from(display)).toEqual([1, 2]);
    expect(source.normalizeDisplayBuffers.mock.calls.map(call => call[1])).toEqual([2, 1]);
    expect(source.imageBuffersF32).not.toHaveBeenCalled();
    expect(events).toEqual(["integrate", "normalize", "integrate", "normalize"]);
  });

  it("retains the empty-mask divisor and does not normalize failed updates", () => {
    const previous = [{} as GPUBuffer];
    const source = {
      integrate: vi.fn(() => ({ added: 0, removed: 1, full: false })),
      normalizeDisplayBuffers: vi.fn(),
    };
    source112MeanDelta(source, new Uint32Array(3), previous);
    expect(source.normalizeDisplayBuffers).toHaveBeenCalledExactlyOnceWith(previous, 1);
    source.integrate.mockImplementationOnce(() => { throw new Error("Source closed"); });
    expect(() => source112MeanDelta(source, new Uint32Array(3), previous)).toThrow("Source closed");
    expect(source.normalizeDisplayBuffers).toHaveBeenCalledTimes(1);
  });
});

describe("borrowed count display lifecycle", () => {
  it("skips conversion only after all direct mean views are accepted", () => {
    const source = {integrate: vi.fn(() => ({added: 1, removed: 0, full: false})), normalizeDisplayBuffers: vi.fn()};
    const buffers = [{} as GPUBuffer];
    const paint = vi.fn(area => { expect(area).toBe(2); return true; });
    source112MeanDelta(source, new Uint32Array([1, 1, 0]), buffers, paint);
    expect(source.normalizeDisplayBuffers).not.toHaveBeenCalled();
    source112MeanDelta(source, new Uint32Array([1, 0, 0]), buffers, () => false);
    expect(source.normalizeDisplayBuffers).toHaveBeenCalledExactlyOnceWith(buffers, 1);
    expect(() => source112MeanDelta(source, new Uint32Array([1]), buffers, () => {throw Error("render failed");})).toThrow("render failed");
    expect(source.normalizeDisplayBuffers).toHaveBeenCalledTimes(2);
  });

  it("retains mean area for repaints and refreshes legacy views only for the current dataset", () => {
    let current = true;
    const refreshFloat = vi.fn();
    const view = {divisor: 2472} as import("../colormaps").Uint32ImageView;
    const counts: CompareCountImages = {images: new Map([[65, view]]), isCurrent: () => current, refreshFloat};
    expect(countImagesForRender(counts, true)?.images.get(65)?.divisor).toBe(2472);
    expect(countImagesForRender(counts, true)).toBe(counts);
    expect(refreshFloat).not.toHaveBeenCalled();
    expect(countImagesForRender(counts, false)).toBeNull();
    expect(refreshFloat).toHaveBeenCalledTimes(1);
    current = false; // The old source has been replaced/disposed.
    expect(countImagesForRender(counts, true)).toBeNull();
    expect(countImagesForRender(counts, false)).toBeNull();
    expect(refreshFloat).toHaveBeenCalledTimes(1);
  });
});

describe("source112 unchanged detector support", () => {
  it("skips equal support, including empty masks, without confusing equal areas", () => {
    const first = new Uint32Array([0, 1, 2, 0]);
    expect(sameDetectorMaskSupport(first, new Uint32Array([0, 5, 1, 0]))).toBe(true);
    expect(sameDetectorMaskSupport(new Uint32Array(36864), new Uint32Array(36864))).toBe(true);
    expect(sameDetectorMaskSupport(first, new Uint32Array([1, 0, 2, 0]))).toBe(false);
    expect(sameDetectorMaskSupport(first, new Uint32Array([0, 1, 2]))).toBe(false);
    expect(Array.from(first)).toEqual([0, 1, 2, 0]);
  });

  it("stops the skip comparison at the first moved detector pixel", () => {
    let reads = 0;
    const mask = new Proxy(new Uint32Array(36864), {get(target, key) {
      if (key === 'length') return target.length;
      if (typeof key === 'string' && /^\d+$/.test(key)) reads++;
      return Reflect.get(target, key, target);
    }});
    const previous = new Uint32Array(36864); previous[0] = 1;
    expect(sameDetectorMaskSupport(mask, previous)).toBe(false);
    expect(reads).toBe(1);
  });

  it("preserves exact native delta counts and float fallback after an unchanged pose", () => {
    let previous: Uint32Array = new Uint32Array([1, 0, 0]);
    const source = {
      integrate: vi.fn((next: Uint32Array) => {
        const added = Array.from(next).filter((value, i) => value && !previous[i]).length;
        const removed = Array.from(previous).filter((value, i) => value && !next[i]).length;
        return {added, removed, full: false};
      }),
      normalizeDisplayBuffers: vi.fn(),
    };
    const buffers = [{} as GPUBuffer];
    const update = (mask: Uint32Array) => {
      if (sameDetectorMaskSupport(mask, previous)) return null;
      const result = source112MeanDelta(source, mask, buffers, () => false);
      previous = mask;
      return result;
    };
    expect(update(new Uint32Array([1, 0, 0]))).toBeNull();
    expect(source.integrate).not.toHaveBeenCalled();
    const next = new Uint32Array([0, 1, 0]);
    expect(update(next)).toEqual({buffers, path: 'delta', addedPixels: 1, removedPixels: 1});
    expect(previous).toBe(next);
    expect(Array.from(next)).toEqual([0, 1, 0]);
    expect(source.normalizeDisplayBuffers).toHaveBeenCalledExactlyOnceWith(buffers, 1);
    expect(update(new Uint32Array([0, 1, 0]))).toBeNull();
    expect(source.integrate).toHaveBeenCalledTimes(1);
  });
});


describe("compare paint cadence", () => {
  function animationFrames() {
    const callbacks = new Map<number, FrameRequestCallback>();
    let next = 0;
    return {
      request: (callback: FrameRequestCallback) => { callbacks.set(++next, callback); return next; },
      cancel: (handle: number) => { callbacks.delete(handle); },
      tick: () => { const pending = [...callbacks.values()]; callbacks.clear(); pending.forEach(callback => callback(0)); },
      count: () => callbacks.size,
    };
  }

  it("computes every pose but paints the newest all66 mean with queue-ordered area", () => {
    const frames = animationFrames();
    const scheduler = createComparePaintScheduler(frames.request, frames.cancel);
    // Model ordered GPU submissions: compute, render and later compute are
    // enqueued synchronously, but execute only when the simulated queue drains.
    const queue: Array<() => void> = [];
    const counts = new Uint32Array(66);
    const patterns = Array.from({ length: 66 }, (_, frame) => [frame + 1, 100 + frame, 1000 + frame]);
    const output = {} as GPUBuffer;
    const displays = Array.from({ length: 66 }, () => ({} as GPUBuffer));
    const paints: number[][] = [];
    let latest: CompareCountImages | null = null;
    let completions = 0;
    const source = {
      integrate(mask: Uint32Array) {
        const exact = patterns.map(pattern => pattern.reduce((sum, value, q) => sum + (mask[q] ? value : 0), 0));
        queue.push(() => { counts.set(exact); completions++; });
        return { added: 1, removed: 1, full: false };
      },
      normalizeDisplayBuffers: vi.fn(),
    };
    const update = (mask: Uint32Array) => source112MeanDelta(source, mask, displays, area => {
      latest = {
        images: new Map(patterns.map((_, frame) => [frame, { buffer: output, divisor: area } as import("../colormaps").Uint32ImageView])),
        isCurrent: () => true, refreshFloat: vi.fn(),
      };
      scheduler.schedule(() => {
        const current = countImagesForRender(latest, true)!;
        const areas = [...current.images.values()].map(view => view.divisor);
        queue.push(() => { paints.push(Array.from(counts, (value, frame) => Math.fround(Math.fround(value) / areas[frame]))); });
      });
      return true;
    });
    update(new Uint32Array([1, 0, 0]));
    update(new Uint32Array([1, 1, 0]));
    expect(frames.count()).toBe(1);
    expect(completions).toBe(0);
    frames.tick(); // Paint uses second pose's counts and area2.
    update(new Uint32Array([0, 0, 1])); // Later compute must not corrupt that queued paint.
    queue.splice(0).forEach(command => command());
    expect(completions).toBe(3);
    expect(paints).toEqual([patterns.map(pattern => (pattern[0] + pattern[1]) / 2)]);
    expect(Array.from(counts)).toEqual(patterns.map(pattern => pattern[2]));
    frames.tick();
    queue.splice(0).forEach(command => command());
    expect(paints[1]).toEqual(patterns.map(pattern => pattern[2]));
    expect(source.normalizeDisplayBuffers).not.toHaveBeenCalled();
  });

  it("uses current display settings at paint and refreshes the newest area for legacy mode", () => {
    const frames = animationFrames();
    const scheduler = createComparePaintScheduler(frames.request, frames.cancel);
    let area = 2, shared = true, current = true;
    let retained: CompareCountImages;
    const refreshFloat = vi.fn();
    const paints: Array<string | null> = [];
    let scale = "linear";
    const update = () => {
      const divisor = area;
      retained = { images: new Map([[0, { divisor } as import("../colormaps").Uint32ImageView]]), isCurrent: () => current,
        refreshFloat: () => refreshFloat(divisor) };
      scheduler.schedule(() => {
        const view = countImagesForRender(retained, shared);
        paints.push(view ? `${scale}:${view.images.get(0)!.divisor}` : null);
      });
    };
    update(); area = 3; update(); scale = "log"; frames.tick();
    expect(paints).toEqual(["log:3"]);
    update(); shared = false; frames.tick();
    expect(refreshFloat).toHaveBeenCalledExactlyOnceWith(3);
    update(); current = false; frames.tick();
    expect(refreshFloat).toHaveBeenCalledTimes(1);
    expect(paints[paints.length - 1]).toBeNull();
  });

  it("cancels deferred work on settled replacement and unmount without owning source buffers", () => {
    const frames = animationFrames();
    const scheduler = createComparePaintScheduler(frames.request, frames.cancel);
    const retiredPaint = vi.fn();
    scheduler.schedule(retiredPaint);
    scheduler.cancel(); frames.tick();
    expect(retiredPaint).not.toHaveBeenCalled();
    const currentPaint = vi.fn();
    scheduler.schedule(currentPaint); frames.tick();
    expect(currentPaint).toHaveBeenCalledTimes(1);
    scheduler.schedule(currentPaint); scheduler.cancel(); frames.tick();
    expect(currentPaint).toHaveBeenCalledTimes(1);
  });

  it("freezes the drag mean before release replaces shared counts during an awaited readback", async () => {
    const frames = animationFrames();
    const scheduler = createComparePaintScheduler(frames.request, frames.cancel);
    const queue: Array<() => void> = [];
    let raw = 120, float = 0;
    const paints: number[] = [];
    let counts: CompareCountImages | null = {
      images: new Map([[0, { divisor: 2 } as import("../colormaps").Uint32ImageView]]),
      isCurrent: () => true,
      refreshFloat: () => { queue.push(() => { float = Math.fround(raw / 2); }); },
    };
    scheduler.schedule(() => {
      const view = countImagesForRender(counts, true);
      if (view) paints.push(raw / view.images.get(0)!.divisor);
    });
    // Renderer 'invalidate' executes before the single-VI release computation.
    scheduler.cancel();
    const previous = counts; counts = null;
    countImagesForRender(previous, false);
    queue.push(() => { raw = 9; }); // Final released mask has area1, not area2.
    await Promise.resolve(); // Single-image readback yields before compare refresh.
    queue.splice(0).forEach(command => command());
    frames.tick();
    expect(paints).toEqual([]); // Never paints new9 / stale2.
    expect(float).toBe(60); // Stable old mean remains valid for interim fallback.
    expect(raw).toBe(9);
    counts = { images: new Map([[0, { divisor: 1 } as import("../colormaps").Uint32ImageView]]),
      isCurrent: () => true, refreshFloat: vi.fn() };
    scheduler.schedule(() => { paints.push(raw / counts!.images.get(0)!.divisor); });
    frames.tick();
    expect(paints).toEqual([9]);
  });

  it("reports deferred rendering errors and allows subsequent paints", () => {
    const frames = animationFrames();
    const scheduler = createComparePaintScheduler(frames.request, frames.cancel);
    const failure = new Error("Compare canvas device lost");
    const visibleErrors = vi.fn();
    scheduler.schedule(() => { throw failure; }, visibleErrors);
    expect(() => frames.tick()).not.toThrow();
    expect(visibleErrors).toHaveBeenCalledExactlyOnceWith(failure);
    const paint = vi.fn();
    scheduler.schedule(paint, visibleErrors); frames.tick();
    expect(paint).toHaveBeenCalledTimes(1);
  });

});
