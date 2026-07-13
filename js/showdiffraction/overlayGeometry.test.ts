import { describe, expect, it } from "vitest";
import {
  dataAngleToScreen,
  dataColToScreenX,
  dataRowToScreenY,
  frameStats,
  screenToData,
  staleFrameNote,
  viewTransform,
} from "./overlayGeometry";

describe("viewTransform", () => {
  it("scales each detector axis independently on non-square detectors", () => {
    // 128x256 detector on a 512 px square canvas: 2 px/col, 4 px/row
    const t = viewTransform(512, 1, 0, 0, 128, 256);
    expect(t.scX).toBe(2);
    expect(t.scY).toBe(4);
    // a ring point 40 px below the center lands 40 * scY screen px down
    const cy = dataRowToScreenY(64, t);
    expect(dataRowToScreenY(64 + 40, t) - cy).toBeCloseTo(40 * t.scY, 10);
  });

  it("applies zoom and pan offsets", () => {
    const t = viewTransform(384, 2, 10, -6, 64, 64);
    expect(t.scX).toBe(12);
    expect(t.offX).toBe((384 - 768) / 2 + 10);
    expect(t.offY).toBe((384 - 768) / 2 - 6);
  });
});

describe("pixel-center convention", () => {
  const t = viewTransform(384, 1, 0, 0, 256, 256);

  it("maps a click on the visual center of pixel (i, j) to exactly (i, j)", () => {
    const mx = t.offX + (100 + 0.5) * t.scX;
    const my = t.offY + (40 + 0.5) * t.scY;
    const { row, col } = screenToData(mx, my, t);
    expect(col).toBeCloseTo(100, 10);
    expect(row).toBeCloseTo(40, 10);
  });

  it("round-trips data coordinates through the screen", () => {
    const mx = dataColToScreenX(123.25, t);
    const my = dataRowToScreenY(45.75, t);
    const { row, col } = screenToData(mx, my, t);
    expect(col).toBeCloseTo(123.25, 10);
    expect(row).toBeCloseTo(45.75, 10);
  });

  it("draws integer pixels at the middle of their screen cell", () => {
    // pixel column j covers [offX + j*scX, offX + (j+1)*scX)
    const j = 7;
    expect(dataColToScreenX(j, t)).toBeCloseTo(t.offX + (j + 0.5) * t.scX, 10);
  });
});

describe("dataAngleToScreen", () => {
  it("is the identity on square detectors", () => {
    const t = viewTransform(384, 1, 0, 0, 256, 256);
    expect(dataAngleToScreen(45, t)).toBeCloseTo(Math.PI / 4, 10);
  });

  it("stretches data angles by the per-axis scales", () => {
    // scX=2, scY=4: a 45 deg data ray renders at atan2(4, 2)
    const t = viewTransform(512, 1, 0, 0, 128, 256);
    expect(dataAngleToScreen(45, t)).toBeCloseTo(Math.atan2(4, 2), 10);
    expect(dataAngleToScreen(0, t)).toBeCloseTo(0, 10);
    expect(dataAngleToScreen(90, t)).toBeCloseTo(Math.PI / 2, 10);
  });
});

describe("frameStats", () => {
  it("computes mean, min, max, std", () => {
    const [mean, min, max, std] = frameStats(new Float32Array([1, 2, 3, 4]));
    expect(mean).toBeCloseTo(2.5, 6);
    expect(min).toBe(1);
    expect(max).toBe(4);
    expect(std).toBeCloseTo(Math.sqrt(1.25), 6);
  });
});

describe("staleFrameNote", () => {
  it("labels offline panes scrubbed away from the baked frame", () => {
    expect(staleFrameNote(true, 4, 2, 0)).toBe("computed on frame 1");
    expect(staleFrameNote(true, 4, 0, 0)).toBeNull();
    expect(staleFrameNote(false, 4, 2, 0)).toBeNull();
    expect(staleFrameNote(true, 1, 0, 0)).toBeNull();
  });
});
