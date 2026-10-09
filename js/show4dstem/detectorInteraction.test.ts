import { describe, expect, it } from "vitest";
import { canvasToMask, clampDetectorCenter, grabDetector, maskToCanvas, resizeDetectorFromPointer } from "./detectorInteraction";
import goldens from "./maskGoldens.json";
import { sampleLineProfile } from "../display/geometry";

// A 24 px detector on a 480 px panel, zoomed and panned. The canvas paints the
// image after translate(pan) and scale(zoom), so pixel j covers [pan + j * zoom,
// pan + (j + 1) * zoom) and its center is drawn at pan + (j + 0.5) * zoom.
const ZOOM = 1.7;
const PAN_X = -3.2;
const PAN_Y = 5.1;
const CSS_PER_PIXEL = 480 / 24;

type MaskCase = { mode: string; center: number[]; radius: number; radius_inner: number; selected: number[] };

/** Flat indices of the pixels whose drawn center lies inside the overlay drawn with center (screenRow, screenCol). */
function pixelsUnderOverlay(testCase: MaskCase, screenRow: number, screenCol: number): number[] {
  const [rows, cols] = goldens.detector_shape;
  const outer = testCase.radius * ZOOM * CSS_PER_PIXEL;
  const inner = testCase.radius_inner * ZOOM * CSS_PER_PIXEL;
  const covered: number[] = [];
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const rowOffset = (PAN_Y + (row + 0.5) * ZOOM) * CSS_PER_PIXEL - screenRow;
      const colOffset = (PAN_X + (col + 0.5) * ZOOM) * CSS_PER_PIXEL - screenCol;
      const inside = testCase.mode === "square"
        ? Math.max(Math.abs(rowOffset), Math.abs(colOffset)) <= outer
        : Math.hypot(rowOffset, colOffset) <= outer && Math.hypot(rowOffset, colOffset) >= inner;
      if (inside) covered.push(row * cols + col);
    }
  }
  return covered;
}

describe("Show4DSTEM detector overlay sits on the pixels the Python mask selects", () => {
  for (const [name, testCase] of Object.entries(goldens.cases as Record<string, MaskCase>)) {
    it(name, () => {
      const [centerRow, centerCol] = testCase.center;
      const screenRow = maskToCanvas(centerRow, ZOOM, PAN_Y) * CSS_PER_PIXEL;
      const screenCol = maskToCanvas(centerCol, ZOOM, PAN_X) * CSS_PER_PIXEL;
      expect(pixelsUnderOverlay(testCase, screenRow, screenCol)).toEqual(testCase.selected);
    });
  }

  it("drawn without the half-pixel offset, the bright-field overlay misses pixels of its own mask", () => {
    const disk = goldens.cases["bin8 bright-field disk"] as MaskCase;
    const [centerRow, centerCol] = disk.center;
    const shifted = pixelsUnderOverlay(disk, (centerRow * ZOOM + PAN_Y) * CSS_PER_PIXEL, (centerCol * ZOOM + PAN_X) * CSS_PER_PIXEL);
    expect(shifted).not.toEqual(disk.selected);
  });

  it("a press on a drawn pixel maps back to that pixel's mask coordinate", () => {
    for (const index of [0, 7, 23]) {
      expect(canvasToMask(PAN_X + (index + 0.5) * ZOOM, ZOOM, PAN_X)).toBeCloseTo(index, 12);
      expect(Math.round(canvasToMask(PAN_X + (index + 0.9) * ZOOM, ZOOM, PAN_X))).toBe(index);
      expect(canvasToMask(maskToCanvas(index + 0.3, ZOOM, PAN_Y), ZOOM, PAN_Y)).toBeCloseTo(index + 0.3, 12);
    }
  });
});

describe("Show4DSTEM detector interaction geometry", () => {
  it("keeps subpixel detector centers while clamping to the diffraction plane", () => {
    expect(clampDetectorCenter(12.25, 18.75, 48, 48)).toEqual({
      row: 12.25,
      col: 18.75,
    });
    expect(clampDetectorCenter(-2, 50, 48, 48)).toEqual({ row: 0, col: 47 });
  });

  it("moves a circle or ring from the inner half of its radius and resizes it near the edge", () => {
    // the bright-field disk of a bin-8 gold scan: r = 6.08 px, hit margin 0.5 px
    const disk = { mode: "circle" as const, centerRow: 11.5, centerCol: 11.36, radius: 6.08, radiusInner: 0, width: 0, height: 0, hitMargin: 0.5 };
    const at = (distance: number) => grabDetector({ ...disk, pointerRow: 11.5 + distance * 0.6, pointerCol: 11.36 + distance * 0.8 }).action;
    expect([0, 0.7, 2.0, 3.0].map(at)).toEqual(["move", "move", "move", "move"]);
    expect([3.1, 5.0, 6.08, 6.5].map(at)).toEqual(["resize", "resize", "resize", "resize"]);
    expect(at(6.7)).toBe("outside");
    // an ADF ring (inner = r_bf, outer = 2 r_bf) moves from its hole, resizes on the ring
    const ring = { ...disk, mode: "annular" as const, radius: 12.16, radiusInner: 6.08 };
    expect(grabDetector({ ...ring, pointerRow: 11.5, pointerCol: 11.36 + 5.9 }).action).toBe("move");
    expect(grabDetector({ ...ring, pointerRow: 11.5, pointerCol: 11.36 + 6.2 }).action).toBe("resize");
    // a thin ring moves from anywhere inside its hole, not only from half its radius
    const thin = { ...ring, radiusInner: 10.0 };
    expect(grabDetector({ ...thin, pointerRow: 11.5, pointerCol: 11.36 + 9.0 }).action).toBe("move");
    // squares measure the Chebyshev distance
    const square = { ...disk, mode: "square" as const, radius: 4 };
    expect(grabDetector({ ...square, pointerRow: 11.5 + 1.9, pointerCol: 11.36 + 1.9 }).action).toBe("move");
    expect(grabDetector({ ...square, pointerRow: 11.5 + 3.9, pointerCol: 11.36 }).action).toBe("resize");
  });

  it("resizes a rectangle along the edge that was grabbed and moves it from inside", () => {
    const rect = { mode: "rect" as const, centerRow: 10, centerCol: 10, radius: 0, radiusInner: 0, width: 8, height: 4, hitMargin: 0.5 };
    expect(grabDetector({ ...rect, pointerRow: 10, pointerCol: 14.2 })).toEqual({ action: "resize", rows: false, cols: true });
    expect(grabDetector({ ...rect, pointerRow: 12.1, pointerCol: 9 })).toEqual({ action: "resize", rows: true, cols: false });
    expect(grabDetector({ ...rect, pointerRow: 12, pointerCol: 14 })).toEqual({ action: "resize", rows: true, cols: true });
    expect(grabDetector({ ...rect, pointerRow: 11, pointerCol: 12 })).toEqual({ action: "move" });
    expect(grabDetector({ ...rect, pointerRow: 10, pointerCol: 16 })).toEqual({ action: "outside" });
  });

  it("changes the size by how far the pointer moved, so a grab inside the band never jumps", () => {
    const center = { centerRow: 10, centerCol: 10 };
    const start = { pointerRow: 10, pointerCol: 14, radius: 8, radiusInner: 3, width: 8, height: 4, rows: true, cols: true };
    // pressed at 4 px from the center of an 8 px circle: no change until the pointer moves
    expect(resizeDetectorFromPointer({ ...center, mode: "circle", pointerRow: 10, pointerCol: 14, start })).toEqual({ radius: 8 });
    expect(resizeDetectorFromPointer({ ...center, mode: "circle", pointerRow: 10, pointerCol: 16, start })).toEqual({ radius: 10 });
    expect(resizeDetectorFromPointer({ ...center, mode: "circle", pointerRow: 10, pointerCol: 6, start })).toEqual({ radius: 8 });
    expect(resizeDetectorFromPointer({ ...center, mode: "square", pointerRow: 13, pointerCol: 15, start })).toEqual({ radius: 9 });
    // a rect grabbed on its right edge keeps its height while the pointer wanders vertically
    const edge = { ...start, rows: false };
    expect(resizeDetectorFromPointer({ ...center, mode: "rect", pointerRow: 11, pointerCol: 15, start: edge })).toEqual({ width: 10, height: 4 });
    expect(resizeDetectorFromPointer({
      ...center, mode: "rect", pointerRow: 11, pointerCol: 15, start: edge, preserveAspect: true,
    })).toEqual({ width: 10, height: 5 });
  });

  it("keeps annular inner and outer radii ordered during live resizing", () => {
    const common = { mode: "annular" as const, centerRow: 10, centerCol: 10 };
    const start = { pointerRow: 10, pointerCol: 17, radius: 10, radiusInner: 4, width: 0, height: 0, rows: true, cols: true };
    expect(resizeDetectorFromPointer({ ...common, pointerRow: 10, pointerCol: 9, start })).toEqual({ radius: 5 });
    expect(resizeDetectorFromPointer({
      ...common,
      pointerRow: 10,
      pointerCol: 25,
      start: { ...start, pointerCol: 14 },
      resizeInner: true,
    })).toEqual({ radiusInner: 9 });
  });
});

describe("Show4DSTEM profile points use the convention the line sampler reads", () => {
  it("a click on a pixel center stores that pixel's index and the profile reads that pixel", () => {
    // one bright detector pixel at (5, 9) on a background of 10
    const rows = 16;
    const cols = 16;
    const image = new Float32Array(rows * cols).fill(10);
    image[5 * cols + 9] = 1000;
    // clicks on the drawn centers of pixels (5, 9) and (5, 13), on the zoomed and panned canvas
    const click = (row: number, col: number) => ({ y: maskToCanvas(row, ZOOM, PAN_Y), x: maskToCanvas(col, ZOOM, PAN_X) });
    const [first, second] = [click(5, 9), click(5, 13)];
    const start = { row: canvasToMask(first.y, ZOOM, PAN_Y), col: canvasToMask(first.x, ZOOM, PAN_X) };
    const end = { row: canvasToMask(second.y, ZOOM, PAN_Y), col: canvasToMask(second.x, ZOOM, PAN_X) };
    expect([start.row, start.col, end.row, end.col].map((value) => Math.round(value * 1e9) / 1e9)).toEqual([5, 9, 5, 13]);
    const profile = sampleLineProfile(image, cols, rows, start.row, start.col, end.row, end.col);
    expect(profile[0]).toBeCloseTo(1000, 3);
    expect(profile[profile.length - 1]).toBeCloseTo(10, 3);
    // the same click as an image coordinate (pixel i spans [i, i + 1)) reads a quarter of the pixel
    const span = { row: (first.y - PAN_Y) / ZOOM, col: (first.x - PAN_X) / ZOOM };
    expect(sampleLineProfile(image, cols, rows, span.row, span.col, end.row, end.col)[0]).toBeCloseTo(257.5, 3);
  });
});
