import { describe, expect, it } from "vitest";

import {
  browserFilterCacheKey,
  normalizedAverageWindow,
  temporalAverageFrameIndices,
  requiresClientFrameTransform,
  shouldApplyClientDifference,
  supportsClientAverage,
} from "./frameTransform";

describe("Show3D frame transform ownership", () => {
  it("applies difference once: server owns live data and JavaScript owns offline data", () => {
    expect(shouldApplyClientDifference(false, "previous")).toBe(false);
    expect(shouldApplyClientDifference(false, "first")).toBe(false);
    expect(shouldApplyClientDifference(true, "previous")).toBe(true);
    expect(shouldApplyClientDifference(true, "off")).toBe(false);
  });

  it("keeps averaging as a client transform", () => {
    expect(requiresClientFrameTransform({ offline: false, diffMode: "previous", avgWindow: 1 })).toBe(false);
    expect(requiresClientFrameTransform({ offline: true, diffMode: "previous", avgWindow: 1 })).toBe(true);
    expect(requiresClientFrameTransform({ offline: false, diffMode: "off", avgWindow: 5 })).toBe(true);
    expect(normalizedAverageWindow(99)).toBe(15);
  });

  it("marks separate-panel averaging unsupported until neighbor frames are fetched", () => {
    expect(supportsClientAverage(false)).toBe(true);
    expect(supportsClientAverage(true)).toBe(false);
  });

  it("separates replacement frame bytes at the same scrub index", () => {
    const base = {
      frameIndex: 3,
      mode: "gaussian",
      sigma: 8,
      bin: 1,
      avgWindow: 1,
      diffMode: "off",
    };
    expect(browserFilterCacheKey({ ...base, frameSeq: 11 }))
      .not.toBe(browserFilterCacheKey({ ...base, frameSeq: 12 }));
  });

  it("separates packed multi-panel browser filter cache entries", () => {
    const base = {
      frameIndex: 3,
      frameSeq: 11,
      mode: "gaussian",
      sigma: 8,
      bin: 1,
      avgWindow: 1,
      diffMode: "off",
    };
    expect(browserFilterCacheKey({ ...base, panels: 1 }))
      .not.toBe(browserFilterCacheKey({ ...base, panels: 3 }));
  });
});


describe("shared Show3D moving-average window", () => {
  it("slides through three adjacent slices and keeps full width at edges", () => {
    expect(temporalAverageFrameIndices(0,16,3)).toEqual([0,1,2]);
    expect(temporalAverageFrameIndices(1,16,3)).toEqual([0,1,2]);
    expect(temporalAverageFrameIndices(2,16,3)).toEqual([1,2,3]);
    expect(temporalAverageFrameIndices(15,16,3)).toEqual([13,14,15]);
  });
  it("handles one slice, even windows, and wider requests without wrapping", () => {
    expect(temporalAverageFrameIndices(4,16,1)).toEqual([4]);
    expect(temporalAverageFrameIndices(4,16,4)).toEqual([2,3,4,5]);
    expect(temporalAverageFrameIndices(0,1,15)).toEqual([0]);
    expect(temporalAverageFrameIndices(8,3,15)).toEqual([0,1,2]);
  });
});
