import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { createResidentHoverReader } from "./residentHover";

type MockBuffer = { bytes: Uint8Array; mapAsync: () => Promise<void>; getMappedRange: () => ArrayBuffer; unmap: () => void; destroy: () => void; destroyed: boolean };

/** A device whose copies run on submit and whose maps resolve when `release()` is called. */
function mockDevice() {
  const copies: number[] = [];
  const waiting: (() => void)[] = [];
  const buffer = (size: number): MockBuffer => {
    const bytes = new Uint8Array(size);
    const mock: MockBuffer = {
      bytes, destroyed: false, unmap: () => {}, destroy: () => { mock.destroyed = true; },
      mapAsync: () => new Promise<void>(resolve => waiting.push(resolve)),
      getMappedRange: () => bytes.buffer.slice(0),
    };
    return mock;
  };
  const device = {
    createBuffer: ({ size }: { size: number }) => buffer(size),
    createCommandEncoder: () => {
      const commands: (() => void)[] = [];
      return {
        copyBufferToBuffer: (source: MockBuffer, sourceOffset: number, target: MockBuffer, targetOffset: number, size: number) => {
          copies.push(size);
          commands.push(() => target.bytes.set(source.bytes.subarray(sourceOffset, sourceOffset + size), targetOffset));
        },
        finish: () => commands,
      };
    },
    queue: { submit: (buffers: (() => void)[][]) => buffers.forEach(commands => commands.forEach(run => run())) },
  } as unknown as GPUDevice;
  const release = async () => { while (waiting.length) { waiting.shift()!(); await Promise.resolve(); await Promise.resolve(); } };
  return { device, copies, release, buffer };
}

beforeEach(() => {
  vi.stubGlobal("GPUBufferUsage", { MAP_READ: 1, COPY_DST: 8, STORAGE: 128 });
  vi.stubGlobal("GPUMapMode", { READ: 1 });
});
afterEach(() => vi.unstubAllGlobals());

test("a hover reads the exact count under the pointer, four bytes at a time, latest position only", async () => {
  const { device, copies, release, buffer } = mockDevice();
  // Two acquisitions' uint32 sums in one buffer; the view lends the second, which starts at byte 256.
  const counts = buffer(256 + 4 * 6);
  new Uint32Array(counts.bytes.buffer, 256, 6).set([10, 16777217, 4294967295, 7, 0, 99]);
  const view = { device, buffer: counts as unknown as GPUBuffer, byteOffset: 256, count: 6, divisor: 1 };
  const reader = createResidentHoverReader();
  const shown: number[] = [];
  reader.request(view, 1, () => true, count => shown.push(count));
  reader.request(view, 3, () => true, count => shown.push(count));
  reader.request(view, 2, () => true, count => shown.push(count));
  await release();
  await release();
  // The first read was already in flight; the second was replaced by the third.
  expect(copies).toEqual([4, 4]);
  expect(shown).toEqual([4294967295]);
  reader.request(view, 1, () => true, count => shown.push(count));
  await release();
  expect(shown).toEqual([4294967295, 16777217]);
});

test("a read is dropped when the image changed or the pointer left before it arrived", async () => {
  const { device, release, buffer } = mockDevice();
  const counts = buffer(16);
  const view = { device, buffer: counts as unknown as GPUBuffer, byteOffset: 0, count: 4, divisor: 1 };
  const reader = createResidentHoverReader();
  const shown: number[] = [];
  let current = true;
  reader.request(view, 0, () => current, count => shown.push(count));
  current = false;
  await release();
  reader.request(view, 1, () => true, count => shown.push(count));
  reader.clear();
  await release();
  expect(shown).toEqual([]);
});
