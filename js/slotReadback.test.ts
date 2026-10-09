// A slot uploaded with a capacity hint (Show3D sizes its rgba buffer for the
// canvas, not the frame) has a read buffer larger than its image. Reading the
// whole mapped buffer into an image-sized array threw a RangeError; the
// readbacks now take only the slot's own pixels.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { COLORMAPS, GPUColormapEngine } from "./display/colormaps";

type MockBuffer = { bytes: Uint8Array; mapAsync: () => Promise<void>; getMappedRange: (offset?: number, size?: number) => ArrayBuffer; unmap: () => void; destroy: () => void };

/** A device whose colormap pass writes a recognisable byte per pixel and whose copies run on submit. */
function mockDevice() {
  const buffer = (size: number): MockBuffer => {
    const bytes = new Uint8Array(size);
    return {
      bytes, mapAsync: async () => {}, unmap: () => {}, destroy: () => {},
      getMappedRange: (offset = 0, size = bytes.byteLength - offset) => bytes.buffer.slice(offset, offset + size),
    };
  };
  const device = {
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBuffer: ({ size }: { size: number }) => buffer(size),
    createBindGroup: (descriptor: { entries: { binding: number; resource: { buffer: MockBuffer } }[] }) => descriptor,
    createCommandEncoder: () => {
      const commands: (() => void)[] = [];
      return {
        beginComputePass: () => {
          let group: { entries: { binding: number; resource: { buffer: MockBuffer } }[] };
          return {
            setPipeline: () => {}, end: () => {},
            setBindGroup: (_index: number, next: typeof group) => { group = next; },
            dispatchWorkgroups: () => {
              const rgba = group.entries.find(entry => entry.binding === 3)!.resource.buffer;
              commands.push(() => rgba.bytes.fill(7));
            },
          };
        },
        copyBufferToBuffer: (source: MockBuffer, sourceOffset: number, target: MockBuffer, targetOffset: number, size: number) => {
          commands.push(() => target.bytes.set(source.bytes.subarray(sourceOffset, sourceOffset + size), targetOffset));
        },
        finish: () => commands,
      };
    },
    queue: {
      writeBuffer: () => {},
      submit: (buffers: (() => void)[][]) => buffers.forEach(commands => commands.forEach(run => run())),
    },
  };
  return device as unknown as GPUDevice;
}

beforeEach(() => {
  vi.stubGlobal("GPUBufferUsage", { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 });
  vi.stubGlobal("GPUMapMode", { READ: 1, WRITE: 2 });
});
afterEach(() => vi.unstubAllGlobals());

describe("colormap readback of a slot with spare capacity", () => {
  it("returns the slot's own pixels", async () => {
    const engine = new GPUColormapEngine(mockDevice());
    engine.uploadLUT("gray", COLORMAPS.gray);
    engine.uploadData(0, new Float32Array(16 * 8), 16, 8, 64 * 64);
    const rgba = await engine.applySingleWithLut(0, 0, 1, "gray", COLORMAPS.gray);
    expect(rgba).toHaveLength(16 * 8 * 4);
    expect(rgba!.every(value => value === 7)).toBe(true);
    const putImageData = vi.fn();
    const imageData = { data: new Uint8ClampedArray(16 * 8 * 4) } as ImageData;
    const offscreen = { getContext: () => ({ putImageData }) } as unknown as HTMLCanvasElement;
    expect(await engine.renderSlots([0], [{ vmin: 0, vmax: 1 }], [offscreen], [imageData])).toBe(1);
    expect(imageData.data.every(value => value === 7)).toBe(true);
    expect(putImageData).toHaveBeenCalledWith(imageData, 0, 0);
  });
});
