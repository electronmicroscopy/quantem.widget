import { describe, expect, it } from "vitest";

import {
  normalizedAverageWindow,
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
});
