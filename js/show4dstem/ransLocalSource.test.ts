// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";
import { ransHttpSource, ransLocalSource, type RansDirectoryHandle } from "../.generated/engine/detector/compute/webgpu/rans-source";

function folder(entries: Record<string, Blob | RansDirectoryHandle>): RansDirectoryHandle {
  const missing = () => new DOMException("Missing file", "NotFoundError");
  return {
    async getDirectoryHandle(name) {
      const entry = entries[name];
      if (!entry || entry instanceof Blob) throw missing();
      return entry;
    },
    async getFileHandle(name) {
      const entry = entries[name];
      if (!(entry instanceof Blob)) throw missing();
      return { async getFile() { return entry as File; } };
    },
  };
}

afterEach(() => vi.unstubAllGlobals());
describe("local rANS export acquisition", () => {
  it("reads exact separate acquisitions and ranges with no network fallback", async () => {
    const fetch = vi.fn(() => { throw new Error("network forbidden"); });
    vi.stubGlobal("fetch", fetch);
    const bytes = new Uint8Array([0, 255, 128, 1, 254, 0, 17]);
    const root = folder({ "manifest.json": new Blob(['{"tilts":[]}']),
      a: folder({ "payload.bin": new Blob([bytes]) }),
      b: folder({ "payload.bin": new Blob([bytes.slice().reverse()]) }) });
    for (const selection of [root, folder({ rans: root })]) {
      const source = await ransLocalSource(selection);
      expect(source.mode).toBe("local-folder");
      expect(new Uint8Array(await source.read("a/payload.bin", 1, 5))).toEqual(bytes.slice(1, 5));
      expect(new Uint8Array(await source.read("b/payload.bin"))).toEqual(bytes.slice().reverse());
      await expect(source.read("missing.bin")).rejects.toThrow();
    }
    expect(fetch).not.toHaveBeenCalled();
  });
  it("rejects escaped paths and truncated byte ranges", async () => {
    const source = await ransLocalSource(folder({ "manifest.json": new Blob(["{}"]), "payload.bin": new Blob(["123"]) }));
    for (const name of ["../payload.bin", "/payload.bin", "a/../payload.bin", "https://example/payload.bin"]) {
      await expect(source.read(name)).rejects.toThrow("Invalid rANS relative");
    }
    for (const [start, end] of [[0, 4], [-1, 2], [2, 1], [0.5, 2]]) {
      await expect(source.read("payload.bin", start, end)).rejects.toThrow("invalid byte range");
    }
  });
  it("keeps permission failures explicit", async () => {
    const root = folder({});
    root.getFileHandle = async () => { throw new DOMException("Permission denied", "NotAllowedError"); };
    await expect(ransLocalSource(root)).rejects.toThrow("Permission denied");
  });
  it("requires exact HTTP ranges rather than silently uploading a full file", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(new Uint8Array([1, 2, 3]), { status: 200 }));
    vi.stubGlobal("fetch", fetch);
    await expect(ransHttpSource("/rans").read("payload.bin", 0, 2)).rejects.toThrow("Range-capable");
    fetch.mockResolvedValue(new Response(new Uint8Array([1]), { status: 206 }));
    await expect(ransHttpSource("/rans").read("payload.bin", 0, 2)).rejects.toThrow("length mismatch");
    fetch.mockResolvedValue(new Response(new Uint8Array([255, 0]), { status: 206 }));
    expect(new Uint8Array(await ransHttpSource("/rans").read("payload.bin", 1, 3))).toEqual(new Uint8Array([255, 0]));
  });
});

// This tests byte staging, not the GPU decoder's numerical parity.
it("stages unaligned blocks into exact packed buffers and reports complete readiness", async () => {
  const { RansResidentSet } = await import("../.generated/engine/detector/compute/webgpu/rans");
  vi.stubGlobal("GPUBufferUsage", { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, MAP_READ: 8, UNIFORM: 16 });
  vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 });
  vi.stubGlobal("GPUMapMode", { READ: 1 });
  const created: { bytes: ArrayBuffer; size: number; usage: number }[] = [];
  const device = {
    limits: { maxStorageBufferBindingSize: 16, maxBufferSize: 1024 },
    createBuffer({ size, usage }: { size: number; usage: number }) {
      const buffer = { size, usage, bytes: new ArrayBuffer(size), getMappedRange() { return this.bytes; },
        unmap() {}, destroy() {}, async mapAsync() {} };
      created.push(buffer); return buffer;
    },
    createBindGroupLayout() { return {}; }, createPipelineLayout() { return {}; }, createShaderModule() { return {}; },
    createComputePipeline() { return { getBindGroupLayout() { return {}; } }; }, createBindGroup() { return {}; },
    createCommandEncoder() { return {
      beginComputePass() { return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }; },
      copyBufferToBuffer(src: { bytes: ArrayBuffer }, start: number, dst: { bytes: ArrayBuffer }, offset: number, length: number) {
        new Uint8Array(dst.bytes, offset, length).set(new Uint8Array(src.bytes, start, length));
      }, finish() { return {}; },
    }; },
    queue: {
      writeBuffer(buffer: { bytes: ArrayBuffer }, offset: number, data: ArrayBufferView) {
        new Uint8Array(buffer.bytes, offset, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      }, submit() {}, async onSubmittedWorkDone() {},
    },
  };
  const manifest = { tilts: [{ tilt: 0, K: 1, frames: 256, blocks: 2, scale: 8, model_frames: 256,
    payload_url: "payload.bin", models: [{ index: 0, symbols: 1 }], blocks_meta: [
      { index: 0, bytes: 3, model: 0, byte_start: 0, byte_end: 3 },
      { index: 1, bytes: 5, model: 0, byte_start: 3, byte_end: 8 },
    ] }] };
  const files: Record<string, Blob> = {
    "manifest.json": new Blob([JSON.stringify(manifest)]), "payload.bin": new Blob([new Uint8Array([1, 2, 255, 128, 0, 254, 4, 5])]),
    "t0-ctx-0.u32": new Blob([new Uint32Array([0, 1])]), "t0-literal-0.u8": new Blob([new Uint8Array([0])]),
    "t0-entries-0.u32": new Blob([new Uint32Array([1])]), "t0-lut-0.u8": new Blob([new Uint8Array([0])]),
    "t0-offsets-00.u32": new Blob([new Uint32Array([0, 3])]), "t0-offsets-01.u32": new Blob([new Uint32Array([0, 5])]),
  };
  const fetch = vi.fn(() => { throw new Error("network forbidden"); }); vi.stubGlobal("fetch", fetch);
  const set = await RansResidentSet.loadLocal(device as unknown as GPUDevice, folder(files));
  const payload = created.find(b => b.size === 12 && b.usage === 3)!;
  expect(new Uint8Array(payload.bytes)).toEqual(new Uint8Array([1, 2, 255, 0, 128, 0, 254, 4, 5, 0, 0, 0]));
  expect(set.payloadBytes).toBe(8);
  expect(set.acquisitionMode).toBe("local-folder");
  expect(set.readyMs).toBeGreaterThanOrEqual(set.loadMs + set.checkpointMs);
  expect(fetch).not.toHaveBeenCalled();
});

it("accepts a browser directory-input FileList without collapsing relative paths", async () => {
  const { ransLocalFilesSource } = await import("../.generated/engine/detector/compute/webgpu/rans-source");
  const file = (path: string, bytes: number[]) => {
    const blob = new Blob([new Uint8Array(bytes)]);
    Object.defineProperties(blob, { name: { value: path.split("/").pop() }, webkitRelativePath: { value: path } });
    return blob as File;
  };
  const files = [file("export/rans/manifest.json", [123, 125]), file("export/rans/a/payload.bin", [255, 1]), file("export/rans/b/payload.bin", [128, 0])];
  const source = await ransLocalFilesSource(files);
  expect(new Uint8Array(await source.read("a/payload.bin"))).toEqual(new Uint8Array([255, 1]));
  expect(new Uint8Array(await source.read("b/payload.bin"))).toEqual(new Uint8Array([128, 0]));
  await expect(source.read("missing.bin")).rejects.toThrow("Missing local rANS file");
  await expect(ransLocalFilesSource([])).rejects.toThrow("Select one rANS export folder");
});
