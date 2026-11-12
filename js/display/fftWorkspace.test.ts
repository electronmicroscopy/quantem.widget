import { afterEach, expect, it, vi } from "vitest";
import { WebGPUFFT } from "./fft";

function fixture() {
  vi.stubGlobal("GPUBufferUsage", { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8, UNIFORM: 16 });
  vi.stubGlobal("GPUMapMode", { READ: 1 });
  const buffers: Array<{ bytes: ArrayBuffer; destroy: ReturnType<typeof vi.fn>; mapAsync: ReturnType<typeof vi.fn>; unmap: ReturnType<typeof vi.fn> }> = [];
  const device = {
    limits: { minUniformBufferOffsetAlignment: 256 },
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createBuffer: ({ size }: { size: number }) => {
      const buffer = { bytes: new ArrayBuffer(size), destroy: vi.fn(), mapAsync: vi.fn(async () => {}), unmap: vi.fn(), getMappedRange() { return this.bytes; } };
      buffers.push(buffer);
      return buffer;
    },
    queue: {
      writeBuffer(buffer: { bytes: ArrayBuffer }, offset: number, source: ArrayBuffer, sourceOffset = 0, length = source.byteLength) {
        new Uint8Array(buffer.bytes, offset, length).set(new Uint8Array(source, sourceOffset, length));
      },
      submit() {},
    },
    createCommandEncoder: () => ({
      beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }),
      copyBufferToBuffer(source: { bytes: ArrayBuffer }, _so: number, target: { bytes: ArrayBuffer }) {
        new Uint8Array(target.bytes).set(new Uint8Array(source.bytes));
      },
      finish() {},
    }),
  };
  return { fft: new WebGPUFFT(device as unknown as GPUDevice), buffers };
}

afterEach(() => vi.unstubAllGlobals());

it("reuses one workspace and releases it on a shape or direction change", async () => {
  const { fft, buffers } = fixture();
  const data = new Float32Array(8 * 8 * 2);
  await fft.fft2DInterleaved(data, 8, 8);
  await fft.fft2DInterleaved(data, 8, 8);
  expect(buffers).toHaveLength(3);
  await fft.fft2DInterleaved(data, 8, 8, true);
  expect(buffers).toHaveLength(6);
  for (const buffer of buffers.slice(0, 3)) expect(buffer.destroy).toHaveBeenCalledOnce();
  await fft.fft2DInterleaved(new Float32Array(16 * 8 * 2), 16, 8);
  for (const buffer of buffers.slice(3, 6)) expect(buffer.destroy).toHaveBeenCalledOnce();
});

it("keeps simultaneous panels independent, then reuses the original workspace", async () => {
  const { fft, buffers } = fixture();
  const a = new Float32Array(8 * 8 * 2).fill(1);
  const b = new Float32Array(8 * 8 * 2).fill(2);
  const [left, right] = await Promise.all([fft.fft2DInterleaved(a, 8, 8), fft.fft2DInterleaved(b, 8, 8)]);
  expect(left).toEqual(a);
  expect(right).toEqual(b);
  expect(buffers).toHaveLength(6);
  for (const buffer of buffers.slice(0, 3)) expect(buffer.destroy).not.toHaveBeenCalled();
  for (const buffer of buffers.slice(3)) expect(buffer.destroy).toHaveBeenCalledOnce();
  await fft.fft2DInterleaved(a, 8, 8);
  expect(buffers).toHaveLength(6);
});

it("discards a failed mapping instead of reusing an invalid readback", async () => {
  const { fft, buffers } = fixture();
  const data = new Float32Array(8 * 8 * 2);
  await fft.fft2DInterleaved(data, 8, 8);
  buffers[1].mapAsync.mockRejectedValueOnce(new Error("device lost"));
  await expect(fft.fft2DInterleaved(data, 8, 8)).rejects.toThrow("device lost");
  for (const buffer of buffers) expect(buffer.destroy).toHaveBeenCalledOnce();
  await fft.fft2DInterleaved(data, 8, 8);
  expect(buffers).toHaveLength(6);
});
