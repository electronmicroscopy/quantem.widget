import { describe, expect, it } from "vitest";

import { packedPanelAutoByteRange } from "./packedPanelContrast";

describe("packed Show3D panel auto contrast", () => {
  it("computes percentiles from the selected panel only", () => {
    const frame = new Uint8Array([
      0, 1, 2, 3, 100, 101, 102, 103,
      4, 5, 6, 7, 104, 105, 106, 107,
    ]);

    expect(packedPanelAutoByteRange(frame, 8, 2, 0, 4, 0, 100))
      .toEqual({ lo: 0, hi: 7 });
    expect(packedPanelAutoByteRange(frame, 8, 2, 4, 4, 0, 100))
      .toEqual({ lo: 100, hi: 107 });
  });

  it("uses discrete percentile ranks without mixing neighboring panels", () => {
    const frame = new Uint8Array([
      0, 10, 20, 30, 200, 210, 220, 230,
      40, 50, 60, 70, 240, 250, 251, 252,
    ]);

    expect(packedPanelAutoByteRange(frame, 8, 2, 0, 4, 25, 75))
      .toEqual({ lo: 10, hi: 50 });
    expect(packedPanelAutoByteRange(frame, 8, 2, 4, 4, 25, 75))
      .toEqual({ lo: 210, hi: 250 });
  });

  it("clamps invalid geometry and rejects truncated frames", () => {
    const frame = new Uint8Array([1, 2, 3, 4]);

    expect(packedPanelAutoByteRange(frame, 0, 2, 0, 2, 0, 100)).toBeNull();
    expect(packedPanelAutoByteRange(frame, 3, 2, 0, 2, 0, 100)).toBeNull();
    expect(packedPanelAutoByteRange(frame, 2.5, 1, 0, 2, 0, 100)).toBeNull();
  });

  it("keeps a usable range for constant black and white panels", () => {
    expect(packedPanelAutoByteRange(new Uint8Array(8), 4, 2, 0, 4, 0.5, 99.5))
      .toEqual({ lo: 0, hi: 1 });
    expect(packedPanelAutoByteRange(new Uint8Array(8).fill(255), 4, 2, 0, 4, 0.5, 99.5))
      .toEqual({ lo: 254, hi: 255 });
  });

  it("reuses cached ranges across views of the same encoded frame", () => {
    const storage = new Uint8Array([9, 9, 0, 10, 20, 30, 40, 9]);
    const firstView = new Uint8Array(storage.buffer, 2, 4);
    const secondView = new Uint8Array(storage.buffer, 2, 4);

    const first = packedPanelAutoByteRange(firstView, 4, 1, 0, 4, 0, 100);
    const second = packedPanelAutoByteRange(secondView, 4, 1, 0, 4, 0, 100);

    expect(first).toEqual({ lo: 0, hi: 30 });
    expect(second).toBe(first);
  });

  it("keeps cache entries distinct for frame offsets and percentile settings", () => {
    const storage = new Uint8Array([0, 10, 20, 30, 100, 110, 120, 130]);
    const firstFrame = new Uint8Array(storage.buffer, 0, 4);
    const secondFrame = new Uint8Array(storage.buffer, 4, 4);

    expect(packedPanelAutoByteRange(firstFrame, 4, 1, 0, 4, 0, 100))
      .toEqual({ lo: 0, hi: 30 });
    expect(packedPanelAutoByteRange(secondFrame, 4, 1, 0, 4, 0, 100))
      .toEqual({ lo: 100, hi: 130 });
    expect(packedPanelAutoByteRange(firstFrame, 4, 1, 0, 4, 50, 100))
      .toEqual({ lo: 10, hi: 30 });
  });
});
