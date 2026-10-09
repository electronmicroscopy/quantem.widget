// @ts-nocheck  (node:fs and __dirname read the shader sources; no @types/node here)
// The slot min/max reduction is the first work Show2D submits to the GPU, and
// on a virtual machine's D3D12 adapter that first render hung the GPU. Its old
// grid-stride loop took its stride from @builtin(num_workgroups), which D3D12
// backends emulate with root constants: read as 0 there, `i += stride` never
// ends. The loop now takes its stride from the uniform the host writes and caps
// its step count. These tests run computeRangeBatch against a recording
// device whose dispatch replays the WGSL loop in JavaScript, line for line.
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GPUColormapEngine } from "./display/colormaps";

const COLORMAPS_SOURCE = readFileSync(path.join(__dirname, "display", "colormaps.ts"), "utf8");
const RANGE_MAX_STEPS = Number(/const RANGE_MAX_STEPS = (\d+)u;/.exec(COLORMAPS_SOURCE)![1]);

/** Every WGSL template literal (`/* wgsl *\/ \`...\``) in the widget's own source files. */
function widgetShaders(): { file: string; code: string }[] {
  const files = readdirSync(__dirname, { recursive: true, encoding: "utf8" })
    .filter(file => /\.tsx?$/.test(file) && !file.includes(".generated") && !file.includes(".test."));
  return files.flatMap(file => {
    const source = readFileSync(path.join(__dirname, file), "utf8");
    return [...source.matchAll(/\/\* wgsl \*\/\s*`([^`]*)`/g)].map(match => ({ file, code: match[1] }));
  });
}

type MockBuffer = { size: number; bytes: Uint8Array; mapAsync: () => Promise<void>; getMappedRange: () => ArrayBuffer; unmap: () => void; destroy: () => void };
type MockPipeline = { code: string; entryPoint: string; getBindGroupLayout: () => object };
type MockGroup = { entries: { binding: number; resource: { buffer: MockBuffer } }[] };

/** The reduce entry point of both range shaders, replayed per invocation as the WGSL reads. */
function replayRangeReduce(pipeline: MockPipeline, group: MockGroup, workgroups: number): void {
  const resource = (binding: number) => group.entries.find(entry => entry.binding === binding)!.resource.buffer;
  const params = new Uint32Array(resource(2).bytes.buffer, 0, 4);
  const [count, groups] = params;
  const packed = pipeline.code.includes("packed_data");
  const input = resource(0).bytes;
  const floats = new Float32Array(input.buffer, 0, input.byteLength >> 2);
  const read = (index: number) => packed ? input[index] : floats[index];
  const out = new Float32Array(resource(1).bytes.buffer);
  const stride = Math.max(groups, 1) * 256;                       // range_stride()
  const steps = Math.min(Math.floor(count / stride) + 1, RANGE_MAX_STEPS);  // range_steps()
  for (let workgroup = 0; workgroup < workgroups; workgroup++) {
    let low = 1.0e38;
    let high = -1.0e38;
    for (let lane = 0; lane < 256; lane++) {
      for (let step = 0; step < steps; step++) {
        const index = workgroup * 256 + lane + step * stride;
        if (index >= count) break;
        const value = read(index);
        if (!Number.isFinite(value)) continue;
        low = Math.min(low, value);
        high = Math.max(high, value);
      }
    }
    out[workgroup * 2] = Math.fround(low);
    out[workgroup * 2 + 1] = Math.fround(high);
  }
}

/** A WebGPU device that records range uniforms and dispatches and replays the reduce shader. */
function recordingDevice(maxComputeWorkgroupsPerDimension: number) {
  const ranges: { params: number[]; workgroups: number }[] = [];
  const buffer = (size: number): MockBuffer => {
    const bytes = new Uint8Array(Math.ceil(size / 4) * 4);
    return { size, bytes, mapAsync: async () => {}, getMappedRange: () => bytes.buffer.slice(0), unmap: () => {}, destroy: () => {} };
  };
  const device = {
    limits: { maxComputeWorkgroupsPerDimension, maxStorageBufferBindingSize: 1 << 30 },
    createShaderModule: ({ code }: { code: string }) => ({ code }),
    createComputePipeline: ({ compute }: { compute: { module: { code: string }; entryPoint: string } }): MockPipeline => (
      { code: compute.module.code, entryPoint: compute.entryPoint, getBindGroupLayout: () => ({}) }),
    createBuffer: ({ size }: { size: number }) => buffer(size),
    createBindGroup: (descriptor: MockGroup) => descriptor,
    createCommandEncoder: () => {
      const commands: (() => void)[] = [];
      return {
        beginComputePass: () => {
          let pipeline: MockPipeline;
          let group: MockGroup;
          return {
            setPipeline: (next: MockPipeline) => { pipeline = next; },
            setBindGroup: (_index: number, next: MockGroup) => { group = next; },
            dispatchWorkgroups: (workgroups: number) => {
              const bound = group;
              const params = Array.from(new Uint32Array(bound.entries.find(entry => entry.binding === 2)!.resource.buffer.bytes.buffer, 0, 4));
              ranges.push({ params, workgroups });
              const reduce = pipeline;
              commands.push(() => replayRangeReduce(reduce, bound, workgroups));
            },
            end: () => {},
          };
        },
        copyBufferToBuffer: (source: MockBuffer, sourceOffset: number, target: MockBuffer, targetOffset: number, size: number) => {
          commands.push(() => target.bytes.set(source.bytes.subarray(sourceOffset, sourceOffset + size), targetOffset));
        },
        finish: () => commands,
      };
    },
    queue: {
      writeBuffer: (target: MockBuffer, offset: number, data: ArrayBuffer | ArrayBufferView, dataOffset = 0, size?: number) => {
        const view = ArrayBuffer.isView(data)
          ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
          : new Uint8Array(data, dataOffset, size ?? data.byteLength - dataOffset);
        target.bytes.set(view, offset);
      },
      submit: (buffers: (() => void)[][]) => buffers.forEach(commands => commands.forEach(run => run())),
    },
  };
  return { device: device as unknown as GPUDevice, ranges };
}

beforeEach(() => {
  vi.stubGlobal("GPUBufferUsage", { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 });
  vi.stubGlobal("GPUMapMode", { READ: 1, WRITE: 2 });
});
afterEach(() => vi.unstubAllGlobals());

describe("slot min/max reduction", () => {
  it("never loops on @builtin(num_workgroups) or with while, anywhere in the widget's WGSL", () => {
    const shaders = widgetShaders();
    expect(shaders.length).toBeGreaterThan(20);
    for (const { file, code } of shaders) {
      expect(code, file).not.toMatch(/num_workgroups/);
      expect(code, file).not.toMatch(/\bwhile\s*\(/);
    }
    // The bounds the replay below mirrors: a zero groups uniform still gives a
    // 256-wide stride, and no count or groups value gives more than the cap.
    expect(COLORMAPS_SOURCE).toContain("fn range_stride() -> u32 { return max(params.groups, 1u) * 256u; }");
    expect(COLORMAPS_SOURCE).toContain("fn range_steps(stride: u32) -> u32 { return min(params.count / stride + 1u, RANGE_MAX_STEPS); }");
    expect(COLORMAPS_SOURCE.match(/for \(var step = 0u; step < steps; step = step \+ 1u\)/g)).toHaveLength(2);
    expect(RANGE_MAX_STEPS).toBe(1024);
  });

  it("finds each slot's exact range in one step, in many steps at the dispatch cap, and for packed uint8", async () => {
    const { device, ranges } = recordingDevice(4);
    const engine = new GPUColormapEngine(device);
    const small = Float32Array.from({ length: 1000 }, (_, i) => Math.sin(i) * 50);  // not a multiple of 256
    const large = Float32Array.from({ length: 128 * 128 }, (_, i) => (i * 7919) % 10007 - 4000);
    large[large.length - 1] = 9999.5;  // the last element is the max: the final step must reach it
    large[3] = Number.NaN;
    large[5] = Number.POSITIVE_INFINITY;
    const bytes = Uint8Array.from({ length: 64 * 64 }, (_, i) => 20 + (i * 13) % 200);
    engine.uploadData(0, small, 40, 25);
    engine.uploadData(1, large, 128, 128);
    engine.uploadUint8Data(2, bytes, 64, 64, 64 * 64);
    const result = await engine.computeRangeBatch([0, 1, 2]);
    const finite = Array.from(large).filter(Number.isFinite);
    expect(result).toEqual([
      { min: Math.fround(Math.min(...small)), max: Math.fround(Math.max(...small)) },
      { min: Math.min(...finite), max: 9999.5 },
      { min: Math.min(...bytes), max: Math.max(...bytes) },
    ]);
    // The uniform carries the dispatched workgroup count: one step at 1000
    // values, 16 steps of 4 x 256 at 128 x 128, and the 65535-style cap.
    expect(ranges).toEqual([
      { params: [1000, 4, 0, 0], workgroups: 4 },
      { params: [128 * 128, 4, 0, 0], workgroups: 4 },
      { params: [64 * 64, 4, 0, 0], workgroups: 4 },
    ]);
  });
});
