import { describe, expect, it } from "vitest";
import { buildDetectorMask } from "../.generated/engine/detector/compute/webgpu/backend";
import { circularDragRadius, liveRoiGeometry } from "./roiRadiusDrag";
import { sameDetectorMaskSupport } from "./source112MeanDelta";

function model(mode = "annular") {
  const traits: Record<string, string | number> = { roi_mode: mode, roi_center_row: 95.5,
    roi_center_col: 95.5, roi_radius: 80, roi_radius_inner: 40 };
  return { traits, get: (name: string) => traits[name] };
}

describe("fractional circular detector handles", () => {
  it("builds exact changing native masks from subpixel handles before any model flush", () => {
    for (const boundary of ["inner", "outer"] as const) {
      const state = model(boundary === "outer" ? "circle" : "annular");
      const start = boundary === "inner" ? 40 : 28;
      let previous: Uint32Array | null = null, changed = 0;
      for (let step = 0; step < 48; step++) {
        const radius = circularDragRadius(start + step / 8, boundary, boundary === "outer" ? 0 : 80);
        const live = liveRoiGeometry(state, null, boundary === "outer" ? radius : null,
          boundary === "inner" ? radius : null);
        const mask = buildDetectorMask(live, 192, 192);
        const outer = boundary === "outer" ? radius : 80;
        const inner = boundary === "inner" ? radius : 0;
        let mismatches = 0;
        for (let row = 0; row < 192; row++) for (let col = 0; col < 192; col++) {
          const distanceSquared = (row - 95.5) ** 2 + (col - 95.5) ** 2;
          if (mask[row * 192 + col] !== Number(distanceSquared <= outer ** 2 && distanceSquared >= inner ** 2)) mismatches++;
        }
        expect(mismatches).toBe(0);
        if (previous && !sameDetectorMaskSupport(mask, previous)) changed++;
        previous = mask;
      }
      expect(changed).toBeGreaterThan(6); // Integer rounding admits only six radius changes here.
      expect(state.traits.roi_radius).toBe(80);
      expect(state.traits.roi_radius_inner).toBe(40);
    }
  });

  it("preserves inclusive inner and outer boundaries and the annular handle gap", () => {
    const state = model();
    const view = liveRoiGeometry(state, [4.5, 4], 3.5, 1.5);
    const mask = buildDetectorMask(view, 10, 10);
    expect(mask[1 * 10 + 4]).toBe(1); // outer radius3.5, inclusive
    expect(mask[3 * 10 + 4]).toBe(1); // inner radius1.5, inclusive
    expect(mask[4 * 10 + 4]).toBe(0);
    expect(circularDragRadius(2.125, "inner", 5)).toBe(2.125);
    expect(circularDragRadius(8, "inner", 5)).toBe(4);
    expect(circularDragRadius(2, "outer", 4.25)).toBe(5.25);
  });

  it("keeps unchanged support uncounted and commits the same final fractional mask", () => {
    const state = model();
    const radius = 40.125;
    const pending = buildDetectorMask(liveRoiGeometry(state, null, null, radius), 192, 192);
    const repeated = buildDetectorMask(liveRoiGeometry(state, null, null, radius), 192, 192);
    expect(sameDetectorMaskSupport(pending, repeated)).toBe(true);
    state.traits.roi_radius_inner = radius;
    expect(liveRoiGeometry(state, null, null, null)).toBe(state);
    const settled = buildDetectorMask(liveRoiGeometry(state, null, null, null), 192, 192);
    expect(sameDetectorMaskSupport(pending, settled)).toBe(true);
  });

  it("clamps rapid opposite-handle gestures against committed geometry before React catches up", () => {
    const state = model();
    const staleRenderedInner = 40, staleRenderedOuter = 80;
    state.traits.roi_radius_inner = 70.125; // Inner gesture just released.
    const outer = circularDragRadius(65, "outer", Number(state.get("roi_radius_inner")));
    expect(outer).toBe(71.125);
    expect(outer).not.toBe(circularDragRadius(65, "outer", staleRenderedInner));
    state.traits.roi_radius_inner = 40;
    state.traits.roi_radius = 65.25; // Later outer gesture just released.
    const inner = circularDragRadius(70, "inner", Number(state.get("roi_radius")));
    expect(inner).toBe(64.25);
    expect(inner).not.toBe(circularDragRadius(70, "inner", staleRenderedOuter));
  });

});
