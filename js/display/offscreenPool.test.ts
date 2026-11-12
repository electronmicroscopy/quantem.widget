// Durable frames (the retained copy behind a WebGPU canvas, gallery tiles) once built a new
// OffscreenCanvas, configured its context and allocated a blit params buffer and bind group for
// every frame: a 48-panel histogram drag did that 51 times per pointer move. The engine now
// pools configured canvases by size and keeps one blit binding per rgba buffer.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GPUColormapEngine } from "./colormaps";

function mockGpu() {
  const counts = { configure: 0, createBindGroup: 0, createBuffer: 0, canvases: 0 };
  class FakeOffscreenCanvas {
    width: number; height: number;
    constructor(width: number, height: number) { this.width = width; this.height = height; counts.canvases++; }
    getContext() {
      return { configure: () => { counts.configure++; }, unconfigure: () => {}, getCurrentTexture: () => ({ createView: () => ({}) }) };
    }
    transferToImageBitmap() { return { width: this.width, height: this.height, close: () => {} }; }
  }
  const device = {
    createShaderModule: () => ({}),
    createRenderPipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBuffer: ({ size }: { size: number }) => { counts.createBuffer++; return { size, destroy: () => {} }; },
    createBindGroup: (descriptor: unknown) => { counts.createBindGroup++; return descriptor; },
    createCommandEncoder: () => ({
      beginRenderPass: () => ({ setPipeline: () => {}, setBindGroup: () => {}, draw: () => {}, end: () => {} }),
      finish: () => ({}),
    }),
    queue: { writeBuffer: () => {}, submit: () => {} },
  };
  return { counts, device: device as unknown as GPUDevice, FakeOffscreenCanvas };
}

let gpu: ReturnType<typeof mockGpu>;
beforeEach(() => {
  gpu = mockGpu();
  vi.stubGlobal("GPUBufferUsage", { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 });
  vi.stubGlobal("OffscreenCanvas", gpu.FakeOffscreenCanvas);
});
afterEach(() => vi.unstubAllGlobals());

describe("durable frame canvases", () => {
  it("configure one canvas per size and bind each rgba buffer once across frames", () => {
    const engine = new GPUColormapEngine(gpu.device);
    engine.ensureBlitPipeline("bgra8unorm");
    const rgba = gpu.device.createBuffer({ size: 64 * 32 * 4, usage: 0 });
    gpu.counts.createBuffer = 0;
    for (let frame = 0; frame < 10; frame++) {
      const encoder = gpu.device.createCommandEncoder();
      const blit = engine.encodeBlitToOffscreen(encoder, rgba, 64, 32, "bgra8unorm")!;
      gpu.device.queue.submit([encoder.finish()]);
      expect(engine.transferOffscreen(blit.canvas).width).toBe(64);
    }
    expect(gpu.counts.canvases).toBe(1);
    expect(gpu.counts.configure).toBe(1);
    expect(gpu.counts.createBuffer).toBe(1);
    expect(gpu.counts.createBindGroup).toBe(1);
  });

  it("hand frames in flight at once their own canvases", () => {
    const engine = new GPUColormapEngine(gpu.device);
    const a = engine.acquireOffscreen(16, 16, "bgra8unorm")!;
    const b = engine.acquireOffscreen(16, 16, "bgra8unorm")!;
    expect(a.canvas).not.toBe(b.canvas);
    engine.transferOffscreen(a.canvas);
    expect(engine.acquireOffscreen(16, 16, "bgra8unorm")!.canvas).toBe(a.canvas);
    expect(engine.acquireOffscreen(32, 16, "bgra8unorm")!.canvas).not.toBe(b.canvas);
  });
});
