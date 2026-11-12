// The panel-grid direct render writes each panel's placement and range into
// its slot's uniform before one render pass, so a slot listed for two panels
// would draw both with the second panel's values; it is refused instead.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { COLORMAPS, GPUColormapEngine } from "./colormaps";

function device(): GPUDevice {
  const buffer = () => ({ destroy: () => {} });
  return {
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBuffer: buffer,
    createCommandEncoder: () => ({}),
    queue: { writeBuffer: () => {} },
  } as unknown as GPUDevice;
}

beforeEach(() => vi.stubGlobal("GPUBufferUsage", { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 }));
afterEach(() => vi.unstubAllGlobals());

describe("panel-grid direct render", () => {
  it("refuses a slot listed for two panels", () => {
    const engine = new GPUColormapEngine(device());
    engine.uploadLUT("gray", COLORMAPS.gray);
    engine.uploadData(0, new Float32Array(16), 4, 4);
    const opts = { width: 8, height: 4, panelCount: 2, cols: 2, rows: 1, gap: 0, bgRgb: 0 };
    expect(() => engine.renderPanelSlotsDirectToCanvas([0, 0], { vmin: 0, vmax: 1 }, false, {} as GPUCanvasContext, opts))
      .toThrow("Supply one distinct resident slot for each panel.");
  });
});

describe("per-canvas panel direct render", () => {
  it("refuses a slot listed for two canvases", () => {
    const engine = new GPUColormapEngine(device());
    engine.uploadData(0, new Float32Array(16), 4, 4);
    const draw = { slot: 0, ctx: {} as GPUCanvasContext, range: { vmin: 0, vmax: 1 }, logScale: false, lut: { name: "gray", lut: COLORMAPS.gray } };
    expect(() => engine.renderSlotsDirectToCanvases([draw, draw], { width: 4, height: 4, bgRgb: 0 }))
      .toThrow("Supply one distinct resident slot for each canvas context.");
  });

  it("refuses a detail overlay drawn from its own panel's slot", () => {
    const engine = new GPUColormapEngine(device());
    engine.uploadData(0, new Float32Array(16), 4, 4);
    const draw = {
      slot: 0, ctx: {} as GPUCanvasContext, range: { vmin: 0, vmax: 1 }, logScale: false,
      lut: { name: "gray", lut: COLORMAPS.gray }, overlay: { slot: 0, x: 0, y: 0, width: 2, height: 2 },
    };
    expect(() => engine.renderSlotsDirectToCanvases([draw], { width: 4, height: 4, bgRgb: 0 }))
      .toThrow("Supply one distinct resident slot for each canvas context.");
  });

  it("draws nothing when a slot is missing, leaving the panel to the 2D path", () => {
    const engine = new GPUColormapEngine(device());
    const draw = { slot: 3, ctx: {} as GPUCanvasContext, range: { vmin: 0, vmax: 1 }, logScale: false, lut: { name: "gray", lut: COLORMAPS.gray } };
    expect(engine.renderSlotsDirectToCanvases([draw], { width: 4, height: 4, bgRgb: 0 })).toBe(false);
  });
});
