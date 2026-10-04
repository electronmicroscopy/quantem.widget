// @ts-expect-error Vitest provides raw text imports; this is never in the widget bundle.
import source from "./index.tsx?raw";
import ts from "typescript";
import { describe, expect, it, vi } from "vitest";
import { buildDetectorMask } from "../.generated/engine/detector/compute/webgpu/backend";
import { liveRoiGeometry } from "./roiRadiusDrag";

// Execute the component's callbacks with a canvas spy, without mounting its GPU
// engine. This catches state updates in the RAF path as well as stale geometry.
function callback(name: string, bindings: Record<string, unknown>) {
  const start = source.indexOf(`const ${name} = React.useCallback(`);
  const bodyStart = start + `const ${name} = React.useCallback(`.length;
  const end = source.indexOf("}, [", bodyStart) + 1;
  if (start < 0 || end <= bodyStart) throw new Error(`Missing callback ${name}`);
  const code = ts.transpileModule(`const callback = ${source.slice(bodyStart, end)};`, {
    compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.None },
  }).outputText;
  return new Function(...Object.keys(bindings), `${code}; return callback;`)(...Object.values(bindings));
}

function workflow() {
  const traits: Record<string, number | string> = {roi_mode: "annular", roi_center_row: 95.5,
    roi_center_col: 95.5, roi_radius: 80, roi_radius_inner: 40};
  const model = {get: (name: string) => traits[name],
    set: vi.fn((name: string, value: number) => { traits[name] = value; }), save_changes: vi.fn()};
  const overlay = vi.fn();
  const canvas = {width: 560, height: 560, getContext: () => ({clearRect: vi.fn()})};
  const state = {
    model, dpRoiInteractiveRef: {current: true}, roiCenterPendingRef: {current: null},
    roiRadiusPendingRef: {current: null as number | null}, roiRadiusInnerPendingRef: {current: null as number | null},
    roiRadiusRafRef: {current: 7 as number | null}, drawDpLiveRef: {current: null as (() => void) | null},
    isResidentCompareDrag: () => true, cancelAnimationFrame: vi.fn(),
    setLocalRoiRadius: vi.fn(() => {throw Error("React radius update during resident paint");}),
    setLocalRoiRadiusInner: vi.fn(() => {throw Error("React inner radius update during resident paint");}),
    dpUiRef: {current: canvas}, roiRadius: 80, roiRadiusInner: 40,
    roiVirtualDetectorActive: true, roiMode: "annular", localKCol: 95.5, localKRow: 95.5,
    DPR: 2, dpZoom: 1, dpPanX: 0, dpPanY: 0, detCols: 192, detRows: 192,
    roiWidth: 1, roiHeight: 1, isDraggingDP: false, isDraggingResize: true,
    isDraggingResizeInner: false, isHoveringResize: false, isHoveringResizeInner: false,
    roiColors: {}, kCalibrated: false, kPixelUnit: "px", showScaleBar: false,
    profileActive: false, showDpColorbar: false, drawRoiOverlayHiDPI: overlay,
  };
  state.drawDpLiveRef.current = callback("drawDpUi", state);
  return {state, traits, model, overlay, flush: callback("flushRoiRadius", state)};
}

describe("resident radius overlay paint", () => {
  it("keeps each scientific radius current while only painting at the scheduled frame", () => {
    const {state, model, overlay, flush} = workflow();
    for (const radius of [80.125, 80.25, 80.5, 80.875]) {
      state.roiRadiusPendingRef.current = radius;
      flush(false);
      expect(liveRoiGeometry(model, null, state.roiRadiusPendingRef.current, null).get("roi_radius")).toBe(radius);
    }
    expect(overlay).not.toHaveBeenCalled();
    expect(state.cancelAnimationFrame).not.toHaveBeenCalled();
    expect(state.roiRadiusRafRef.current).toBe(7);
    expect(model.set).not.toHaveBeenCalled();
    flush();
    expect(overlay).toHaveBeenCalledTimes(1);
    expect(overlay.mock.lastCall![5]).toBe(80.875);
    expect(state.roiRadiusRafRef.current).toBeNull();
  });

  it.each(["inner", "outer"] as const)("paints latest %s geometry and exact mean support before model publication", boundary => {
    const {state, model, overlay, flush} = workflow();
    for (const radius of [0.125, 0.25, 0.375, 0.5]) {
      const ref = boundary === "outer" ? state.roiRadiusPendingRef : state.roiRadiusInnerPendingRef;
      ref.current = (boundary === "outer" ? 80 : 40) + radius;
      flush();
      const args = overlay.mock.lastCall!;
      const drawn = liveRoiGeometry(model, [args[4], args[3]], args[5], args[6]);
      const scientific = liveRoiGeometry(model, null, state.roiRadiusPendingRef.current, state.roiRadiusInnerPendingRef.current);
      const mask = buildDetectorMask(scientific, 192, 192);
      const overlayMask = buildDetectorMask(drawn, 192, 192);
      expect(overlayMask).toEqual(mask);
      // Independent native support census checks the display mean denominator.
      let area = 0;
      for (let row = 0; row < 192; row++) for (let col = 0; col < 192; col++) {
        const d2 = (row - 95.5) ** 2 + (col - 95.5) ** 2;
        area += Number(d2 >= args[6] ** 2 && d2 <= args[5] ** 2);
      }
      expect(mask.reduce((sum, value) => sum + value, 0)).toBe(area);
      expect(model.set).not.toHaveBeenCalled();
      expect(ref.current).not.toBeNull();
    }
    // Exposure/layout repaints share this callback and retain the live radius.
    state.drawDpLiveRef.current!();
    expect(overlay.mock.lastCall![boundary === "outer" ? 5 : 6]).toBe(boundary === "outer" ? 80.5 : 40.5);
    state.dpRoiInteractiveRef.current = false;
    flush();
    expect(model.set).toHaveBeenCalledExactlyOnceWith(boundary === "outer" ? "roi_radius" : "roi_radius_inner", boundary === "outer" ? 80.5 : 40.5);
    expect(model.save_changes).toHaveBeenCalledOnce();
    expect(state.roiRadiusPendingRef.current).toBeNull();
    expect(state.roiRadiusInnerPendingRef.current).toBeNull();
  });

  it("keeps nonresident RAF publication and ignores pending geometry after interaction ends", () => {
    const {state, model, overlay} = workflow();
    state.roiRadiusPendingRef.current = 83.25;
    state.roiRadiusInnerPendingRef.current = 41.125;
    callback("flushRoiRadius", {...state, isResidentCompareDrag: () => false})();
    expect(model.set.mock.calls).toEqual([["roi_radius", 83.25], ["roi_radius_inner", 41.125]]);
    expect(overlay).not.toHaveBeenCalled();
    state.dpRoiInteractiveRef.current = false;
    state.roiRadiusPendingRef.current = 99;
    callback("drawDpUi", {...state, roiRadius: 83.25, roiRadiusInner: 41.125})();
    expect(overlay.mock.lastCall!.slice(5, 7)).toEqual([83.25, 41.125]);
  });
});
