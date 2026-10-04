import { afterEach, describe, expect, it, vi } from "vitest";
import { captureGpuCanvas } from "./captureGpuCanvas";

afterEach(() => vi.unstubAllGlobals());

describe("explicit GPU image capture", () => {
  function setup(format: string, fail = false) {
    const bytes = new Uint8Array(512);
    bytes.set([10, 20, 30, 255, 40, 50, 60, 255], 0);
    bytes.set([70, 80, 90, 255, 100, 110, 120, 255], 256);
    const read = {mapState: "unmapped", destroy: vi.fn(), unmap: vi.fn(),
      mapAsync: vi.fn(async () => { if (fail) throw Error("device lost"); read.mapState = "mapped"; }),
      getMappedRange: () => bytes.buffer};
    const copy = vi.fn();
    const device = {createBuffer: () => read, queue: {submit: vi.fn()},
      createCommandEncoder: () => ({copyTextureToBuffer: copy, finish: vi.fn()})};
    const context = {getCurrentTexture: () => ({width: 2, height: 2, format})};
    const put = vi.fn();
    const canvas = {width: 0, height: 0, getContext: () => ({putImageData: put})};
    vi.stubGlobal("GPUBufferUsage", {COPY_DST: 8, MAP_READ: 1});
    vi.stubGlobal("GPUMapMode", {READ: 1});
    vi.stubGlobal("document", {createElement: () => canvas});
    vi.stubGlobal("ImageData", class { constructor(public data: Uint8ClampedArray, public width: number, public height: number) {} });
    const render = vi.fn();
    return {read, copy, put, render, run: () => captureGpuCanvas(device as unknown as GPUDevice, context as unknown as GPUCanvasContext, render)};
  }
  it.each(["rgba8unorm", "bgra8unorm"])("copies padded %s rows with correct channel order", async format => {
    const s = setup(format);
    await s.run();
    const expected = format === "bgra8unorm"
      ? [30,20,10,255,60,50,40,255,90,80,70,255,120,110,100,255]
      : [10,20,30,255,40,50,60,255,70,80,90,255,100,110,120,255];
    expect([...s.put.mock.calls[0][0].data]).toEqual(expected);
    expect(s.render.mock.invocationCallOrder[0]).toBeLessThan(s.copy.mock.invocationCallOrder[0]);
    expect(s.read.unmap).toHaveBeenCalledOnce();
    expect(s.read.destroy).toHaveBeenCalledOnce();
  });
  it("releases the readback when the device rejects capture", async () => {
    const s = setup("bgra8unorm", true);
    await expect(s.run()).rejects.toThrow("device lost");
    expect(s.read.destroy).toHaveBeenCalledOnce();
    expect(s.put).not.toHaveBeenCalled();
  });
});
