import { describe, expect, it } from "vitest";

import {
  clampPanelPlaybackFps,
  makeGalleryFftCacheKey,
  panelPlaybackIntervalMs,
  readGalleryFftCache,
  rememberGalleryFftCache,
  resolveVisibleDiffPlan,
  type GalleryFftCacheEntry,
} from "./localStack";

function entry(values: number[]): GalleryFftCacheEntry {
  return { mag: new Float32Array(values), fftWidth: values.length, fftHeight: 1 };
}

function key(panel: number, frame: number, roiKey = ""): string {
  return makeGalleryFftCacheKey({
    dataEpoch: 3,
    panel,
    frame,
    width: 16,
    height: 12,
    roiKey,
    fftWindow: true,
    overviewDownsample: 1,
  });
}

describe("Show2D local-stack playback", () => {
  it("uses the configured constructor cadence without adding a UI setting", () => {
    expect(clampPanelPlaybackFps(4)).toBe(4);
    expect(panelPlaybackIntervalMs(4)).toBe(250);
    expect(clampPanelPlaybackFps(0.25)).toBe(0.25);
    expect(panelPlaybackIntervalMs(0.25)).toBe(4000);
    expect(clampPanelPlaybackFps(100)).toBe(30);
    expect(clampPanelPlaybackFps(0)).toBe(10);
    expect(clampPanelPlaybackFps(Number.NaN)).toBe(10);
  });
});

describe("Show2D panel-frame FFT cache", () => {
  it("reuses a panel slice after visiting another slice", () => {
    const cache = new Map<string, GalleryFftCacheEntry>();
    const frame0 = entry([1, 2]);
    const frame1 = entry([3, 4]);
    rememberGalleryFftCache(cache, key(0, 0), frame0);
    rememberGalleryFftCache(cache, key(0, 1), frame1);

    expect(readGalleryFftCache(cache, key(0, 0))).toBe(frame0);
    expect(readGalleryFftCache(cache, key(0, 1))).toBe(frame1);
  });

  it("keeps panels, frames, and ROI configurations distinct", () => {
    expect(key(0, 0)).not.toBe(key(1, 0));
    expect(key(0, 0)).not.toBe(key(0, 1));
    expect(key(0, 0)).not.toBe(key(0, 0, "roi-a"));
  });

  it("evicts the oldest inactive result while protecting visible FFTs", () => {
    const cache = new Map<string, GalleryFftCacheEntry>();
    const a = key(0, 0);
    const b = key(0, 1);
    const c = key(0, 2);
    rememberGalleryFftCache(cache, a, entry([1]));
    rememberGalleryFftCache(cache, b, entry([2]));
    const stats = rememberGalleryFftCache(cache, c, entry([3]), {
      maxEntries: 2,
      maxBytes: 1024,
      protectedKeys: new Set([a]),
    });

    expect(cache.has(a)).toBe(true);
    expect(cache.has(b)).toBe(false);
    expect(cache.has(c)).toBe(true);
    expect(stats).toMatchObject({ entries: 2, evictions: 1 });
  });
});

describe("Show2D visible-panel difference plan", () => {
  it("compares exactly the two panels left visible in a larger gallery", () => {
    expect(resolveVisibleDiffPlan([2, 4], [false, false, false, false, false, false], 0)).toEqual({
      visibleGrayscale: [2, 4],
      reference: 2,
      others: [4],
    });
  });

  it("honors a visible configured reference and excludes RGB overlays", () => {
    expect(resolveVisibleDiffPlan([0, 2, 1], [false, true, false], 2)).toEqual({
      visibleGrayscale: [0, 2],
      reference: 2,
      others: [0],
    });
  });
});
