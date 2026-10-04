import { describe, expect, it } from "vitest";
import { sharedCanvasLayout } from "./sharedCanvasLayout";

describe("resident shared canvas layout", () => {
  it("keeps all 66 native panels and their gaps at arbitrary CSS size", () => {
    const tiles = Array.from({ length: 66 }, (_, i) => ({ left: 40 + (i % 11) * 103, top: 70 + Math.floor(i / 11) * 103, width: 100, height: 100 }));
    const layout = sharedCanvasLayout({ left: 40, top: 70, width: 1130, height: 615 }, tiles, 512, 512, 16384)!;
    expect(layout.rectangles).toHaveLength(66);
    for (const rect of layout.rectangles) {
      expect(Math.abs(rect.width - 512)).toBeLessThanOrEqual(1);
      expect(Math.abs(rect.height - 512)).toBeLessThanOrEqual(1);
      expect(rect.x + rect.width).toBeLessThanOrEqual(layout.width);
      expect(rect.y + rect.height).toBeLessThanOrEqual(layout.height);
    }
    expect(layout.rectangles[1].x).toBeGreaterThan(layout.rectangles[0].width);
  });
  it("follows tile order after hiding/reordering without changing source coordinates", () => {
    const a = { left: 0, top: 0, width: 100, height: 50 };
    const b = { left: 110, top: 0, width: 100, height: 50 };
    const grid = { left: 0, top: 0, width: 210, height: 50 };
    expect(sharedCanvasLayout(grid, [b, a], 256, 512, 16384)!.rectangles.map(r => r.x)).toEqual([563, 0]);
    expect(sharedCanvasLayout(grid, [a], 256, 512, 16384)!.rectangles).toHaveLength(1);
  });
  it("bounds presentation to the adapter limit for tall grids", () => {
    const tiles = Array.from({ length: 66 }, (_, i) => ({ left: 0, top: i * 100, width: 100, height: 100 }));
    const layout = sharedCanvasLayout({ left: 0, top: 0, width: 100, height: 6600 }, tiles, 512, 512, 8192)!;
    expect(layout.height).toBe(8192);
    expect(layout.rectangles).toHaveLength(66);
    expect(layout.rectangles[65].y + layout.rectangles[65].height).toBe(8192);
  });
  it("waits for measurable mounted tiles", () => {
    expect(sharedCanvasLayout({ left: 0, top: 0, width: 0, height: 0 }, [], 512, 512, 16384)).toBeNull();
  });
});
