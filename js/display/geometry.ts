/// <reference types="@webgpu/types" />
/** Reusable row/column geometry for browser-side scientific display math. */

import { getHardwareGPUDevice, requireHardwareGPUDevice } from "./device";
import { findFFTPeak } from "./fft";

type MaskedCropRegion = {
  row: number;
  col: number;
  shape?: string;
  radius: number;
  width: number;
  height: number;
};

type MaskedCrop = {
  cropped: Float32Array;
  cropW: number;
  cropH: number;
};

/** Crop a rectangle or an outer-disk-masked circle for an ROI-scoped FFT.
 *
 * ``annular`` intentionally uses only the outer radius: the operation selects
 * the spatial support sent to the FFT, rather than rasterizing a detector
 * annulus. The bounding-box upper edge is exclusive.
 */
export function cropMaskedRegion(
  data: Float32Array,
  imageWidth: number,
  imageHeight: number,
  region: MaskedCropRegion,
): MaskedCrop | null {
  const bounds = cropBounds(imageWidth, imageHeight, region);
  if (!bounds) return null;
  const { column0, row0, cropW, cropH } = bounds;
  const cropped = new Float32Array(cropW * cropH);
  if (bounds.disk) {
    const radiusSquared = region.radius * region.radius;
    for (let cropRow = 0; cropRow < cropH; cropRow++) {
      for (let cropColumn = 0; cropColumn < cropW; cropColumn++) {
        const imageColumn = column0 + cropColumn;
        const imageRow = row0 + cropRow;
        const columnOffset = imageColumn - region.col;
        const rowOffset = imageRow - region.row;
        if (columnOffset * columnOffset + rowOffset * rowOffset <= radiusSquared) {
          cropped[cropRow * cropW + cropColumn] = data[imageRow * imageWidth + imageColumn];
        }
      }
    }
  } else {
    for (let cropRow = 0; cropRow < cropH; cropRow++) {
      const sourceOffset = (row0 + cropRow) * imageWidth + column0;
      cropped.set(data.subarray(sourceOffset, sourceOffset + cropW), cropRow * cropW);
    }
  }
  return { cropped, cropW, cropH };
}

/**
 * Bounding box of an ROI clipped to the image, with an exclusive upper edge:
 * the rectangle's half extents, or the radius for every other shape. Null when
 * fewer than 2 x 2 pixels remain, too small for a meaningful FFT.
 */
function cropBounds(
  imageWidth: number,
  imageHeight: number,
  region: MaskedCropRegion,
): { column0: number; row0: number; cropW: number; cropH: number; disk: boolean } | null {
  const shape = region.shape || "circle";
  const halfWidth = shape === "rectangle" ? region.width / 2 : region.radius;
  const halfHeight = shape === "rectangle" ? region.height / 2 : region.radius;
  const column0 = Math.max(0, Math.floor(region.col - halfWidth));
  const row0 = Math.max(0, Math.floor(region.row - halfHeight));
  const column1 = Math.min(imageWidth, Math.ceil(region.col + halfWidth));
  const row1 = Math.min(imageHeight, Math.ceil(region.row + halfHeight));
  const cropW = column1 - column0;
  const cropH = row1 - row0;
  return cropW < 2 || cropH < 2
    ? null
    : { column0, row0, cropW, cropH, disk: shape === "circle" || shape === "annular" };
}

/** Bilinearly sample one line at unit spacing, clamping taps at the image edge. */
function sampleSingleLine(
  data: Float32Array,
  width: number,
  height: number,
  row0: number,
  column0: number,
  row1: number,
  column1: number,
): Float32Array {
  const columnDelta = column1 - column0;
  const rowDelta = row1 - row0;
  const length = Math.hypot(columnDelta, rowDelta);
  const sampleCount = Math.max(2, Math.ceil(length));
  const output = new Float32Array(sampleCount);
  for (let index = 0; index < sampleCount; index++) {
    const fraction = index / (sampleCount - 1);
    const column = column0 + fraction * columnDelta;
    const row = row0 + fraction * rowDelta;
    const baseColumn = Math.floor(column);
    const baseRow = Math.floor(row);
    const columnFraction = column - baseColumn;
    const rowFraction = row - baseRow;
    const left = Math.max(0, Math.min(width - 1, baseColumn));
    const right = Math.max(0, Math.min(width - 1, baseColumn + 1));
    const top = Math.max(0, Math.min(height - 1, baseRow));
    const bottom = Math.max(0, Math.min(height - 1, baseRow + 1));
    output[index] =
      data[top * width + left] * (1 - columnFraction) * (1 - rowFraction) +
      data[top * width + right] * columnFraction * (1 - rowFraction) +
      data[bottom * width + left] * (1 - columnFraction) * rowFraction +
      data[bottom * width + right] * columnFraction * rowFraction;
  }
  return output;
}

/** Bilinearly sample a line, optionally averaging integer-width parallel lines.
 *  A fractional width rounds to whole lines, as the WebGPU kernel and the uint8 path do. */
export function sampleLineProfile(
  data: Float32Array,
  width: number,
  height: number,
  row0: number,
  column0: number,
  row1: number,
  column1: number,
  profileWidth = 1,
): Float32Array {
  const integerWidth = Math.max(1, Math.round(profileWidth));
  if (integerWidth <= 1) {
    return sampleSingleLine(data, width, height, row0, column0, row1, column1);
  }
  const columnDelta = column1 - column0;
  const rowDelta = row1 - row0;
  const length = Math.hypot(columnDelta, rowDelta);
  if (length < 1e-8) {
    return sampleSingleLine(data, width, height, row0, column0, row1, column1);
  }
  const perpendicularRow = -columnDelta / length;
  const perpendicularColumn = rowDelta / length;
  const halfWidth = (integerWidth - 1) / 2;
  let output: Float32Array | null = null;
  for (let line = 0; line < integerWidth; line++) {
    const offset = -halfWidth + line;
    const samples = sampleSingleLine(
      data,
      width,
      height,
      row0 + offset * perpendicularRow,
      column0 + offset * perpendicularColumn,
      row1 + offset * perpendicularRow,
      column1 + offset * perpendicularColumn,
    );
    if (!output) output = samples;
    else for (let index = 0; index < samples.length; index++) output[index] += samples[index];
  }
  if (output) {
    for (let index = 0; index < output.length; index++) output[index] /= integerWidth;
  }
  return output || new Float32Array(0);
}

/** Sample a quantized image without expanding the whole image to float32. */
export function sampleLineProfileUint8(
  data: Uint8Array,
  low: number,
  high: number,
  width: number,
  height: number,
  row0: number,
  column0: number,
  row1: number,
  column1: number,
  profileWidth = 1,
): Float32Array {
  const finiteLow = Number.isFinite(low) ? low : 0;
  const finiteHigh = Number.isFinite(high) ? high : finiteLow;
  const scale = finiteHigh > finiteLow ? (finiteHigh - finiteLow) / 255 : 0;
  const sampleLine = (startRow: number, startColumn: number, endRow: number, endColumn: number) => {
    const columnDelta = endColumn - startColumn;
    const rowDelta = endRow - startRow;
    const sampleCount = Math.max(2, Math.ceil(Math.hypot(columnDelta, rowDelta)));
    const output = new Float32Array(sampleCount);
    for (let index = 0; index < sampleCount; index++) {
      const fraction = index / (sampleCount - 1);
      const column = startColumn + fraction * columnDelta;
      const row = startRow + fraction * rowDelta;
      const baseColumn = Math.floor(column);
      const baseRow = Math.floor(row);
      const columnFraction = column - baseColumn;
      const rowFraction = row - baseRow;
      const left = Math.max(0, Math.min(width - 1, baseColumn));
      const right = Math.max(0, Math.min(width - 1, baseColumn + 1));
      const top = Math.max(0, Math.min(height - 1, baseRow));
      const bottom = Math.max(0, Math.min(height - 1, baseRow + 1));
      const decode = (position: number) => data[position] * scale + finiteLow;
      output[index] =
        decode(top * width + left) * (1 - columnFraction) * (1 - rowFraction) +
        decode(top * width + right) * columnFraction * (1 - rowFraction) +
        decode(bottom * width + left) * (1 - columnFraction) * rowFraction +
        decode(bottom * width + right) * columnFraction * rowFraction;
    }
    return output;
  };
  const integerWidth = Math.max(1, Math.round(profileWidth));
  if (integerWidth <= 1) return sampleLine(row0, column0, row1, column1);
  const columnDelta = column1 - column0;
  const rowDelta = row1 - row0;
  const length = Math.hypot(columnDelta, rowDelta);
  if (length < 1e-8) return sampleLine(row0, column0, row1, column1);
  const perpendicularRow = -columnDelta / length;
  const perpendicularColumn = rowDelta / length;
  const halfWidth = (integerWidth - 1) / 2;
  let output: Float32Array | null = null;
  for (let line = 0; line < integerWidth; line++) {
    const offset = line - halfWidth;
    const samples = sampleLine(
      row0 + offset * perpendicularRow,
      column0 + offset * perpendicularColumn,
      row1 + offset * perpendicularRow,
      column1 + offset * perpendicularColumn,
    );
    if (!output) output = samples;
    else for (let index = 0; index < output.length; index++) output[index] += samples[index];
  }
  if (output) for (let index = 0; index < output.length; index++) output[index] /= integerWidth;
  return output || new Float32Array(0);
}

const CROP_MASK_WGSL = /* wgsl */ `
struct Params {
  image_width: u32,
  image_height: u32,
  crop_width: u32,
  crop_height: u32,
  column0: u32,
  row0: u32,
  shape: u32,
  _pad: u32,
  center_column: f32,
  center_row: f32,
  radius_squared: f32,
  _pad2: f32,
};
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.crop_width || gid.y >= p.crop_height) { return; }
  let column = p.column0 + gid.x;
  let row = p.row0 + gid.y;
  let output_index = gid.y * p.crop_width + gid.x;
  if (p.shape == 1u) {
    let dc = f32(column) - p.center_column;
    let dr = f32(row) - p.center_row;
    if (dc * dc + dr * dr > p.radius_squared) {
      output[output_index] = 0.0;
      return;
    }
  }
  output[output_index] = source[row * p.image_width + column];
}
`;

const LINE_PROFILE_WGSL = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  sample_count: u32,
  profile_width: u32,
  row0: f32,
  column0: f32,
  row1: f32,
  column1: f32,
};
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

fn sample(row: f32, column: f32) -> f32 {
  let base_column = i32(floor(column));
  let base_row = i32(floor(row));
  let column_fraction = column - f32(base_column);
  let row_fraction = row - f32(base_row);
  let left = u32(clamp(base_column, 0, i32(p.width) - 1));
  let right = u32(clamp(base_column + 1, 0, i32(p.width) - 1));
  let top = u32(clamp(base_row, 0, i32(p.height) - 1));
  let bottom = u32(clamp(base_row + 1, 0, i32(p.height) - 1));
  return source[top * p.width + left] * (1.0 - column_fraction) * (1.0 - row_fraction)
    + source[top * p.width + right] * column_fraction * (1.0 - row_fraction)
    + source[bottom * p.width + left] * (1.0 - column_fraction) * row_fraction
    + source[bottom * p.width + right] * column_fraction * row_fraction;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= p.sample_count) { return; }
  let fraction = f32(index) / f32(p.sample_count - 1u);
  let column_delta = p.column1 - p.column0;
  let row_delta = p.row1 - p.row0;
  let length = sqrt(column_delta * column_delta + row_delta * row_delta);
  var sum = 0.0;
  if (p.profile_width <= 1u || length < 1e-8) {
    sum = sample(
      p.row0 + fraction * row_delta,
      p.column0 + fraction * column_delta,
    );
  } else {
    let perpendicular_row = -column_delta / length;
    let perpendicular_column = row_delta / length;
    let half_width = f32(p.profile_width - 1u) * 0.5;
    for (var line = 0u; line < p.profile_width; line = line + 1u) {
      let offset = f32(line) - half_width;
      sum = sum + sample(
        p.row0 + offset * perpendicular_row + fraction * row_delta,
        p.column0 + offset * perpendicular_column + fraction * column_delta,
      );
    }
    sum = sum / f32(p.profile_width);
  }
  output[index] = sum;
}
`;

const UINT8_LINE_PROFILE_WGSL = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  sample_count: u32,
  profile_width: u32,
  low: f32,
  scale: f32,
  row0: f32,
  column0: f32,
  row1: f32,
  column1: f32,
  _pad0: f32,
  _pad1: f32,
};
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

fn decoded(index: u32) -> f32 {
  let word = packed[index >> 2u];
  let shift = (index & 3u) * 8u;
  return f32((word >> shift) & 255u) * p.scale + p.low;
}

fn sample(row: f32, column: f32) -> f32 {
  let base_column = i32(floor(column));
  let base_row = i32(floor(row));
  let column_fraction = column - f32(base_column);
  let row_fraction = row - f32(base_row);
  let left = u32(clamp(base_column, 0, i32(p.width) - 1));
  let right = u32(clamp(base_column + 1, 0, i32(p.width) - 1));
  let top = u32(clamp(base_row, 0, i32(p.height) - 1));
  let bottom = u32(clamp(base_row + 1, 0, i32(p.height) - 1));
  return decoded(top * p.width + left) * (1.0 - column_fraction) * (1.0 - row_fraction)
    + decoded(top * p.width + right) * column_fraction * (1.0 - row_fraction)
    + decoded(bottom * p.width + left) * (1.0 - column_fraction) * row_fraction
    + decoded(bottom * p.width + right) * column_fraction * row_fraction;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= p.sample_count) { return; }
  let fraction = f32(index) / f32(p.sample_count - 1u);
  let column_delta = p.column1 - p.column0;
  let row_delta = p.row1 - p.row0;
  let length = sqrt(column_delta * column_delta + row_delta * row_delta);
  var sum = 0.0;
  if (p.profile_width <= 1u || length < 1e-8) {
    sum = sample(p.row0 + fraction * row_delta, p.column0 + fraction * column_delta);
  } else {
    let perpendicular_row = -column_delta / length;
    let perpendicular_column = row_delta / length;
    let half_width = f32(p.profile_width - 1u) * 0.5;
    for (var line = 0u; line < p.profile_width; line = line + 1u) {
      let offset = f32(line) - half_width;
      sum = sum + sample(
        p.row0 + offset * perpendicular_row + fraction * row_delta,
        p.column0 + offset * perpendicular_column + fraction * column_delta,
      );
    }
    sum = sum / f32(p.profile_width);
  }
  output[index] = sum;
}
`;

const FFT_PEAK_WGSL = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  base_column: u32,
  base_row: u32,
  radius: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(0) var<storage, read> magnitude: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

fn finite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

@compute @workgroup_size(1)
fn main() {
  let column0 = max(0, i32(p.base_column) - i32(p.radius));
  let row0 = max(0, i32(p.base_row) - i32(p.radius));
  let column1 = min(i32(p.width) - 1, i32(p.base_column) + i32(p.radius));
  let row1 = min(i32(p.height) - 1, i32(p.base_row) + i32(p.radius));
  var best_column = i32(p.base_column);
  var best_row = i32(p.base_row);
  var best_value = -3.402823466e38;
  for (var row = row0; row <= row1; row = row + 1) {
    for (var column = column0; column <= column1; column = column + 1) {
      let value = magnitude[u32(row) * p.width + u32(column)];
      if (finite(value) && value > best_value) {
        best_value = value;
        best_column = column;
        best_row = row;
      }
    }
  }
  var weight = 0.0;
  var weighted_column = 0.0;
  var weighted_row = 0.0;
  for (var row = max(0, best_row - 1); row <= min(i32(p.height) - 1, best_row + 1); row = row + 1) {
    for (var column = max(0, best_column - 1); column <= min(i32(p.width) - 1, best_column + 1); column = column + 1) {
      let value = magnitude[u32(row) * p.width + u32(column)];
      if (finite(value) && value > 0.0) {
        weight = weight + value;
        weighted_column = weighted_column + f32(column) * value;
        weighted_row = weighted_row + f32(row) * value;
      }
    }
  }
  output[0] = select(f32(best_row), weighted_row / weight, weight > 0.0);
  output[1] = select(f32(best_column), weighted_column / weight, weight > 0.0);
}
`;

/**
 * Run a one-input compute kernel and read back its float32 output: upload
 * `source` and `params`, dispatch `shader` once, copy the output to a mappable
 * buffer. Every buffer is destroyed before returning, so callers own nothing.
 */
async function runFloatKernel(
  operation: string,
  source: Float32Array | Uint32Array,
  outputCount: number,
  params: ArrayBuffer,
  shader: string,
  dispatch: [number, number, number?],
): Promise<Float32Array> {
  const device = await requireHardwareGPUDevice(operation);
  const sourceBuffer = device.createBuffer({
    size: source.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    size: outputCount * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const readBuffer = device.createBuffer({
    size: outputCount * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  try {
    device.queue.writeBuffer(
      sourceBuffer,
      0,
      source.buffer as ArrayBuffer,
      source.byteOffset,
      source.byteLength,
    );
    device.queue.writeBuffer(paramsBuffer, 0, params);
    const module = device.createShaderModule({ code: shader });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: sourceBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatch[0], dispatch[1], dispatch[2] ?? 1);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputCount * 4);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const output = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    return output;
  } finally {
    sourceBuffer.destroy();
    outputBuffer.destroy();
    paramsBuffer.destroy();
    readBuffer.destroy();
  }
}

/** Crop/mask an FFT ROI: WebGPU on a hardware adapter, else the JS reference. */
export async function cropMaskedRegionBrowser(
  data: Float32Array,
  imageWidth: number,
  imageHeight: number,
  region: MaskedCropRegion,
): Promise<MaskedCrop | null> {
  if (!(await getHardwareGPUDevice())) return cropMaskedRegion(data, imageWidth, imageHeight, region);
  const bounds = cropBounds(imageWidth, imageHeight, region);
  if (!bounds) return null;
  const params = new ArrayBuffer(48);
  const u32 = new Uint32Array(params);
  const f32 = new Float32Array(params);
  u32[0] = imageWidth; u32[1] = imageHeight;
  u32[2] = bounds.cropW; u32[3] = bounds.cropH;
  u32[4] = bounds.column0; u32[5] = bounds.row0; u32[6] = bounds.disk ? 1 : 0;
  f32[8] = region.col; f32[9] = region.row; f32[10] = region.radius * region.radius;
  const cropped = await runFloatKernel(
    "ROI crop and mask",
    data,
    bounds.cropW * bounds.cropH,
    params,
    CROP_MASK_WGSL,
    [Math.ceil(bounds.cropW / 16), Math.ceil(bounds.cropH / 16)],
  );
  return { cropped, cropW: bounds.cropW, cropH: bounds.cropH };
}

/** Bilinearly sample a width-averaged profile: WebGPU on a hardware adapter, else the JS reference. */
export async function sampleLineProfileBrowser(
  data: Float32Array,
  width: number,
  height: number,
  row0: number,
  column0: number,
  row1: number,
  column1: number,
  profileWidth = 1,
): Promise<Float32Array> {
  if (!(await getHardwareGPUDevice())) return sampleLineProfile(data, width, height, row0, column0, row1, column1, profileWidth);
  const sampleCount = Math.max(2, Math.ceil(Math.hypot(column1 - column0, row1 - row0)));
  const integerWidth = Math.max(1, Math.round(profileWidth));
  const params = new ArrayBuffer(32);
  const u32 = new Uint32Array(params);
  const f32 = new Float32Array(params);
  u32[0] = width; u32[1] = height; u32[2] = sampleCount; u32[3] = integerWidth;
  f32[4] = row0; f32[5] = column0; f32[6] = row1; f32[7] = column1;
  return runFloatKernel(
    "Line-profile sampling",
    data,
    sampleCount,
    params,
    LINE_PROFILE_WGSL,
    [Math.ceil(sampleCount / 256), 1],
  );
}

/** Sample uint8+range scientific data: WebGPU on a hardware adapter, else the JS reference. */
export async function sampleLineProfileUint8Browser(
  data: Uint8Array,
  low: number,
  high: number,
  width: number,
  height: number,
  row0: number,
  column0: number,
  row1: number,
  column1: number,
  profileWidth = 1,
): Promise<Float32Array> {
  if (!(await getHardwareGPUDevice())) return sampleLineProfileUint8(data, low, high, width, height, row0, column0, row1, column1, profileWidth);
  if (data.length < width * height) throw new Error("Quantized line-profile image is shorter than width * height");
  const finiteLow = Number.isFinite(low) ? low : 0;
  const finiteHigh = Number.isFinite(high) ? high : finiteLow;
  const sampleCount = Math.max(2, Math.ceil(Math.hypot(column1 - column0, row1 - row0)));
  const params = new ArrayBuffer(48);
  const u32 = new Uint32Array(params);
  const f32 = new Float32Array(params);
  u32[0] = width; u32[1] = height; u32[2] = sampleCount; u32[3] = Math.max(1, Math.round(profileWidth));
  f32[4] = finiteLow; f32[5] = finiteHigh > finiteLow ? (finiteHigh - finiteLow) / 255 : 0;
  f32[6] = row0; f32[7] = column0; f32[8] = row1; f32[9] = column1;
  const source = data.subarray(0, width * height);
  // Four samples per storage word: WGSL storage buffers have no 8-bit type.
  const packed = new Uint32Array(Math.max(1, Math.ceil(source.length / 4)));
  for (let index = 0; index < source.length; index++) {
    packed[index >> 2] |= source[index] << ((index & 3) * 8);
  }
  return runFloatKernel(
    "Quantized line-profile sampling",
    packed,
    sampleCount,
    params,
    UINT8_LINE_PROFILE_WGSL,
    [Math.ceil(sampleCount / 256), 1],
  );
}

/** Find and centroid-refine a local FFT peak: WebGPU on a hardware adapter, else the JS reference. */
export async function findFFTPeakBrowser(
  magnitude: Float32Array,
  width: number,
  height: number,
  column: number,
  row: number,
  radius: number,
): Promise<{ row: number; col: number }> {
  if (!(await getHardwareGPUDevice())) return findFFTPeak(magnitude, width, height, column, row, radius);
  if (width < 1 || height < 1 || magnitude.length < width * height) return { row: 0, col: 0 };
  const params = new ArrayBuffer(32);
  const u32 = new Uint32Array(params);
  u32[0] = width; u32[1] = height;
  u32[2] = Math.max(0, Math.min(width - 1, Math.floor(column)));
  u32[3] = Math.max(0, Math.min(height - 1, Math.floor(row)));
  u32[4] = Math.max(0, Math.floor(radius));
  const output = await runFloatKernel(
    "FFT peak refinement",
    magnitude,
    2,
    params,
    FFT_PEAK_WGSL,
    [1, 1],
  );
  return { row: output[0], col: output[1] };
}
