import { describe, expect, it } from "vitest";
import { imageToScreen, screenToImage, zoomAt, type ImageViewport } from "./imageView";

const view: ImageViewport = { height: 48, width: 64, canvas: 512, zoom: 1, panX: 0, panY: 0 };

describe("imageView", () => {
  it("round-trips image and screen coordinates under pan and zoom", () => {
    const panned: ImageViewport = { ...view, zoom: 3.2, panX: -40, panY: 17 };
    const [x, y] = imageToScreen(panned, 12.5, 30.25);
    const [row, col] = screenToImage(panned, x, y);
    expect(row).toBeCloseTo(12.5);
    expect(col).toBeCloseTo(30.25);
  });

  it("keeps the point under the cursor fixed while zooming", () => {
    const before = screenToImage(view, 300, 200);
    const zoomed: ImageViewport = { ...view, ...zoomAt(view, 300, 200, -1) };
    const after = screenToImage(zoomed, 300, 200);
    expect(after[0]).toBeCloseTo(before[0]);
    expect(after[1]).toBeCloseTo(before[1]);
  });
});
