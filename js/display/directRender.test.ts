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
