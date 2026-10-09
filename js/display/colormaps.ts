// Display colormaps, ranges and canvas rendering owned by quantem.widget.
// The CPU path (applyColormap) is the reference; the WebGPU engine must match it.
import { validateUint32ImageView, type Uint32ImageView } from "./borrowedImage";
export type { Uint32ImageView } from "./borrowedImage";
// One table for Python and the browser: the package file Plot2D and static renders read.
import colormapPoints from "../../src/quantem/widget/colormaps.json";
import { getHardwareGPUDevice, onGPULost } from "./device";

export const COLORMAP_POINTS: Record<string, number[][]> = colormapPoints;

export const COLORMAP_NAMES = Object.keys(COLORMAP_POINTS);

/** 256-entry RGB LUT, linearly interpolated between evenly spaced control points and rounded to bytes. */
function createColormapLUT(points: number[][]): Uint8Array {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const position = (i / 255) * (points.length - 1);
    const index = Math.floor(position);
    const frac = position - index;
    const p0 = points[Math.min(index, points.length - 1)];
    const p1 = points[Math.min(index + 1, points.length - 1)];
    lut[i * 3] = Math.round(p0[0] + frac * (p1[0] - p0[0]));
    lut[i * 3 + 1] = Math.round(p0[1] + frac * (p1[1] - p0[1]));
    lut[i * 3 + 2] = Math.round(p0[2] + frac * (p1[2] - p0[2]));
  }
  return lut;
}

export const COLORMAPS: Record<string, Uint8Array> = Object.fromEntries(
  Object.entries(COLORMAP_POINTS).map(([name, points]) => [name, createColormapLUT(points)])
);

// ============================================================================
// CPU colormap (Float32 -> RGBA via 256-entry LUT)
// ============================================================================

/**
 * Map a display value to [0, 1] exactly as the WGSL display_normalize does, in
 * float32: NaN and -inf go to 0, +inf to 1, an empty window to 0.5, and a
 * window whose float32 span overflows is halved first (exact), so
 * (value - low) / (high - low) is computed the same way for every window.
 * Every step is a correctly rounded float32 operation on both paths (the WGSL
 * divides through display_divide, not the GPU's inexact `/`), which is what
 * lets the Canvas2D and WebGPU paths choose the same LUT entry for a value that
 * sits on a level edge.
 */
export function displayNormalize(value: number, low: number, high: number): number {
  const f32 = Math.fround;
  if (!Number.isFinite(value)) return value === Infinity ? 1 : 0;
  if (!(high > low)) return 0.5;
  const scale = Number.isFinite(f32(high - low)) ? 1 : 0.5;
  const span = f32(f32(high * scale) - f32(low * scale));
  return Math.min(1, Math.max(0, f32(f32(f32(value * scale) - f32(low * scale)) / span)));
}

/** Signed log1p in float32, the WGSL log_scale branch: log(1 + v), or -log(1 - v) below zero. */
export function displayLog(value: number): number {
  const f32 = Math.fround;
  return value >= 0 ? f32(Math.log(f32(1 + value))) : f32(-Math.log(f32(1 - value)));
}

/** Apply colormap LUT to float data, writing into an RGBA Uint8ClampedArray. */
export function applyColormap(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
): void {
  const low = Math.fround(vmin);
  const high = Math.fround(vmax);
  for (let i = 0; i < data.length; i++) {
    const entry = Math.min(255, Math.trunc(Math.fround(displayNormalize(data[i], low, high) * 255))) * 3;
    const pixel = i * 4;
    rgba[pixel] = lut[entry];
    rgba[pixel + 1] = lut[entry + 1];
    rgba[pixel + 2] = lut[entry + 2];
    rgba[pixel + 3] = 255;
  }
}

/** Create an offscreen canvas with colormapped data. Returns null if context unavailable. */
export function renderToOffscreen(
  data: Float32Array,
  width: number,
  height: number,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
): HTMLCanvasElement | null {
  const offscreen = document.createElement("canvas");
  offscreen.width = width;
  offscreen.height = height;
  const ctx = offscreen.getContext("2d");
  if (!ctx) return null;
  const imgData = ctx.createImageData(width, height);
  applyColormap(data, imgData.data, lut, vmin, vmax);
  ctx.putImageData(imgData, 0, 0);
  return offscreen;
}

/** Render colormapped data to a reusable offscreen canvas + ImageData (avoids per-frame allocation). */
export function renderToOffscreenReuse(
  data: Float32Array,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  offscreen: HTMLCanvasElement,
  imgData: ImageData,
): void {
  applyColormap(data, imgData.data, lut, vmin, vmax);
  offscreen.getContext("2d")!.putImageData(imgData, 0, 0);
}

// ============================================================================
// WebGPU colormap engine
// ============================================================================

// Per-pixel passes dispatch 2D 16 x 16 workgroups to stay within WebGPU's 65535
// workgroups per dimension: a 1D dispatch of 256-thread groups over a 4096 x 4096
// frame needs ceil(4096 * 4096 / 256) = 65536, one past the limit.

// Correctly rounded float32 quotient of two integers in [1, 2^24) (0 / d is 0).
// Hardware f32 division is not correctly rounded (WGSL allows 2.5 ULP, and
// Metal uses it), so `a / b` can differ from the CPU path's
// Math.fround(a / b) by an ULP. Both integers are shifted so bit 23 leads,
// which leaves the quotient in [0.5, 2); three u32 divisions of 8 (last 7 or 8)
// bits each give the 24-bit significand, and the remainder rounds it to
// nearest, ties to even, in straight-line code with no per-pixel loop.
const ROUNDED_QUOTIENT_WGSL = /* wgsl */ `
fn rounded_quotient(numerator: u32, denominator: u32) -> f32 {
  if (numerator == 0u) { return 0.0; }
  let numerator_shift = 23u - firstLeadingBit(numerator);
  let denominator_shift = 23u - firstLeadingBit(denominator);
  let dividend = numerator << numerator_shift;
  let divisor = denominator << denominator_shift;
  let below_one = dividend < divisor;
  let exponent = i32(denominator_shift) - i32(numerator_shift) - select(0, 1, below_one);
  var mantissa = select(1u, 0u, below_one);
  var remainder = select(dividend - divisor, dividend, below_one);
  remainder = remainder << 8u;
  mantissa = (mantissa << 8u) | (remainder / divisor);
  remainder = remainder % divisor;
  remainder = remainder << 8u;
  mantissa = (mantissa << 8u) | (remainder / divisor);
  remainder = remainder % divisor;
  let last_bits = select(7u, 8u, below_one);
  remainder = remainder << last_bits;
  mantissa = (mantissa << last_bits) | (remainder / divisor);
  remainder = remainder % divisor;
  if (remainder * 2u > divisor || (remainder * 2u == divisor && (mantissa & 1u) != 0u)) { mantissa++; }
  return bitcast<f32>((u32(exponent + 127) << 23u) + mantissa - 0x800000u);
}
`;

// Temporal mean of N window frames -> one output frame, on the GPU. One thread
// per output pixel sums that pixel across the N frames (loop on the GPU, parallel
// over pixels) and divides. Replaces a CPU per-pixel double-loop on the UI thread.
// Integer-count sums take the correctly rounded quotient, so the mean equals the
// CPU path's Math.fround(sum / n).
const AVERAGE_SHADER = ROUNDED_QUOTIENT_WGSL + /* wgsl */ `
struct AvgParams { n: u32, frameSize: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: AvgParams;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.frameSize) { return; }
  var s = 0.0;
  for (var j = 0u; j < p.n; j = j + 1u) {
    s = s + src[j * p.frameSize + i];
  }
  if (s >= 0.0 && s <= 16777215.0 && floor(s) == s && p.n <= 65535u) {
    dst[i] = rounded_quotient(u32(s), p.n);
  } else {
    dst[i] = s / f32(p.n);
  }
}
`;

// display_divide is the correctly rounded numerator / denominator for finite
// numerator >= 0 and denominator > 0 (quotient below 2^128): the significands
// (hidden bit included; a subnormal has none and the smallest normal exponent)
// go through rounded_quotient and the exponent difference is applied exactly.
const DISPLAY_NORMALIZE_WGSL = ROUNDED_QUOTIENT_WGSL + /* wgsl */ `
fn display_is_finite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn display_half(value: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) - 0x00800000u);
}

fn display_divide(numerator: f32, denominator: f32) -> f32 {
  let numerator_bits = bitcast<u32>(numerator);
  let denominator_bits = bitcast<u32>(denominator);
  let numerator_field = (numerator_bits >> 23u) & 0xffu;
  let denominator_field = (denominator_bits >> 23u) & 0xffu;
  let numerator_significand = (numerator_bits & 0x7fffffu) | select(0u, 0x800000u, numerator_field != 0u);
  let denominator_significand = (denominator_bits & 0x7fffffu) | select(0u, 0x800000u, denominator_field != 0u);
  let exponent = i32(max(numerator_field, 1u)) - i32(max(denominator_field, 1u));
  return ldexp(rounded_quotient(numerator_significand, denominator_significand), exponent);
}

fn display_normalize(value: f32, low: f32, high: f32) -> f32 {
  let bits = bitcast<u32>(value);
  if ((bits & 0x7f800000u) == 0x7f800000u) {
    let negative = (bits & 0x80000000u) != 0u;
    let nan = (bits & 0x007fffffu) != 0u;
    return select(1.0, 0.0, negative || nan);
  }
  if (!(high > low)) { return 0.5; }
  // A window wider than float32 holds (high - low overflows) is halved first.
  // Halving is exact, so the quotient keeps its meaning, and one display_divide
  // call serves every window: FXC (Chrome's D3D12 compiler below feature level
  // 12) fails to build a shader that inlines a second set of divisions. Such a
  // window has low < 0 < high, both far above the smallest normal, so they
  // halve in their exponent bits; a multiply by 0.5 there was factored by the
  // compiler into (high - low) * 0.5, which overflows again.
  let wide = !display_is_finite(high - low);
  let window_low = select(low, display_half(low), wide);
  let span = select(high, display_half(high), wide) - window_low;
  let offset = select(value, value * 0.5, wide) - window_low;
  if (!(offset > 0.0)) { return 0.0; }
  if (offset >= span) { return 1.0; }
  return display_divide(offset, span);
}
`;

// Grid-stride bounds shared by the slot min/max reductions. The stride is the
// dispatched workgroup count the host writes into the uniform, times 256; it is
// not read from @builtin(num_workgroups). D3D12 has no such system value, so
// Dawn emulates it with root constants, and a translating driver that reads
// them as 0 turns `i += stride` into an endless loop that hangs the GPU. The
// step count is capped as well, so no uniform value can make the loop
// unbounded: 256 steps already cover a 4 GiB uint8 buffer at the 65535-group
// dispatch limit (2^32 / (65535 * 256)).
const RANGE_REDUCE_WGSL = /* wgsl */ `
struct RangeParams { count: u32, groups: u32, _p0: u32, _p1: u32 };
const RANGE_MAX_STEPS = 1024u;
fn range_stride() -> u32 { return max(params.groups, 1u) * 256u; }
fn range_steps(stride: u32) -> u32 { return min(params.count / stride + 1u, RANGE_MAX_STEPS); }
`;

const COLORMAP_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  vmin: f32,
  vmax: f32,
  log_scale: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  var val = data[idx];
  if (params.log_scale == 1u) {
    if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); }
  }
  let t = display_normalize(val, params.vmin, params.vmax);
  let lutIdx = min(u32(t * 255.0), 255u);
  let rgb = lut[lutIdx];
  // Simplified: LUT is already packed as R|(G<<8)|(B<<16), just add alpha
  rgba[idx] = rgb | 0xFF000000u;
}
`;

const SCALED_COLORMAP_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct Params {
  src_width: u32,
  src_height: u32,
  out_width: u32,
  out_height: u32,
  vmin: f32,
  vmax: f32,
  log_scale: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.out_width || gid.y >= params.out_height) { return; }
  let src_x = min(u32((f32(gid.x) + 0.5) * f32(params.src_width) / f32(params.out_width)), params.src_width - 1u);
  let src_y = min(u32((f32(gid.y) + 0.5) * f32(params.src_height) / f32(params.out_height)), params.src_height - 1u);
  let src_idx = src_y * params.src_width + src_x;
  let out_idx = gid.y * params.out_width + gid.x;
  var val = data[src_idx];
  if (params.log_scale == 1u) {
    if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); }
  }
  let t = display_normalize(val, params.vmin, params.vmax);
  let lutIdx = min(u32(t * 255.0), 255u);
  let rgb = lut[lutIdx];
  rgba[out_idx] = rgb | 0xFF000000u;
}
`;

// Folder-mode Show2D keeps native uint8 frames packed at one byte per pixel.
// Read four pixels from each storage word so a 40 x 4K review does not expand
// 640 MB of source bytes into 2.5 GB of float32 GPU buffers merely to paint a
// 300 px preview. Sampling and LUT normalization remain identical to the
// float32 scaled shader above.
const SCALED_UINT8_COLORMAP_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct Params {
  src_width: u32,
  src_height: u32,
  out_width: u32,
  out_height: u32,
  vmin: f32,
  vmax: f32,
  log_scale: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> packed_data: array<u32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;

fn read_u8(index: u32) -> f32 {
  let word = packed_data[index >> 2u];
  let shift = (index & 3u) * 8u;
  return f32((word >> shift) & 255u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.out_width || gid.y >= params.out_height) { return; }
  let src_x = min(u32((f32(gid.x) + 0.5) * f32(params.src_width) / f32(params.out_width)), params.src_width - 1u);
  let src_y = min(u32((f32(gid.y) + 0.5) * f32(params.src_height) / f32(params.out_height)), params.src_height - 1u);
  let src_idx = src_y * params.src_width + src_x;
  let out_idx = gid.y * params.out_width + gid.x;
  var val = read_u8(src_idx);
  if (params.log_scale == 1u) { val = log(1.0 + val); }
  let t = display_normalize(val, params.vmin, params.vmax);
  let lut_idx = min(u32(t * 255.0), 255u);
  rgba[out_idx] = lut[lut_idx] | 0xFF000000u;
}
`;

const DIRECT_GRID_COLORMAP_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct Params {
  src_width: u32,
  src_height: u32,
  src_panel_width: u32,
  out_width: u32,
  out_height: u32,
  panel_count: u32,
  cols: u32,
  rows: u32,
  log_scale: u32,
  bg_rgb: u32,
  shared_source: u32,
  _pad0: u32,
  vmin: f32,
  vmax: f32,
  gap: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
  let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  return out;
}

fn unpack_rgb(rgb: u32) -> vec4f {
  let r = f32(rgb & 0xFFu) / 255.0;
  let g = f32((rgb >> 8u) & 0xFFu) / 255.0;
  let b = f32((rgb >> 16u) & 0xFFu) / 255.0;
  return vec4f(r, g, b, 1.0);
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  if (params.cols == 0u || params.rows == 0u || params.panel_count == 0u || params.src_width == 0u || params.src_height == 0u) {
    return unpack_rgb(params.bg_rgb);
  }

  let out_x = min(u32(in.uv.x * f32(params.out_width)), params.out_width - 1u);
  let out_y = min(u32(in.uv.y * f32(params.out_height)), params.out_height - 1u);
  let src_panel_w = max(1u, min(params.src_panel_width, params.src_width));
  let gap = params.gap;
  let panel_w = (f32(params.out_width) - gap * f32(params.cols - 1u)) / f32(params.cols);
  let panel_h = (f32(params.out_height) - gap * f32(params.rows - 1u)) / f32(params.rows);
  let stride_x = panel_w + gap;
  let stride_y = panel_h + gap;
  let px = f32(out_x) + 0.5;
  let py = f32(out_y) + 0.5;
  let col = u32(floor(px / stride_x));
  let row = u32(floor(py / stride_y));
  if (col >= params.cols || row >= params.rows) {
    return unpack_rgb(params.bg_rgb);
  }

  let local_x = px - f32(col) * stride_x;
  let local_y = py - f32(row) * stride_y;
  let panel_idx = row * params.cols + col;
  if (panel_idx >= params.panel_count || local_x < 0.0 || local_y < 0.0 || local_x >= panel_w || local_y >= panel_h) {
    return unpack_rgb(params.bg_rgb);
  }

  let src_panel_idx = select(panel_idx, 0u, params.shared_source == 1u);
  let src_local_x = min(u32(local_x * f32(src_panel_w) / panel_w), src_panel_w - 1u);
  let src_x = min(src_panel_idx * src_panel_w + src_local_x, params.src_width - 1u);
  let src_y = min(u32(local_y * f32(params.src_height) / panel_h), params.src_height - 1u);
  let src_idx = src_y * params.src_width + src_x;
  var val = data[src_idx];
  if (params.log_scale == 1u) {
    if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); }
  }
  let t = display_normalize(val, params.vmin, params.vmax);
  let lut_idx = min(u32(t * 255.0), 255u);
  return unpack_rgb(lut[lut_idx]);
}
`;

const DIRECT_SLOT_COLORMAP_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct Params {
  src_width: u32,
  src_height: u32,
  src_x0: u32,
  src_region_width: u32,
  out_height: u32,
  out_width: u32,
  origin_x: f32,
  origin_y: f32,
  log_scale: u32,
  bg_rgb: u32,
  zoom: f32,
  smooth_sample: u32,
  vmin: f32,
  vmax: f32,
  pan_x: f32,
  pan_y: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
  let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  return out;
}

fn unpack_rgb(rgb: u32) -> vec4f {
  let r = f32(rgb & 0xFFu) / 255.0;
  let g = f32((rgb >> 8u) & 0xFFu) / 255.0;
  let b = f32((rgb >> 16u) & 0xFFu) / 255.0;
  return vec4f(r, g, b, 1.0);
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  if (params.src_width == 0u || params.src_height == 0u) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  let region_w = max(1u, min(params.src_region_width, params.src_width));
  let region_x0 = min(params.src_x0, params.src_width - 1u);
  let out_w = f32(max(1u, params.out_width));
  let out_h = f32(max(1u, params.out_height));
  let local_x = in.pos.x - params.origin_x;
  let local_y = in.pos.y - params.origin_y;
  let image_x = (local_x - params.pan_x) / max(params.zoom, 1e-6);
  let image_y = (local_y - params.pan_y) / max(params.zoom, 1e-6);
  if (image_x < 0.0 || image_y < 0.0 || image_x >= out_w || image_y >= out_h) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  var val: f32;
  if (params.smooth_sample == 1u) {
    let src_fx = clamp((image_x + 0.5) * f32(region_w) / out_w - 0.5, 0.0, f32(region_w - 1u));
    let src_fy = clamp((image_y + 0.5) * f32(params.src_height) / out_h - 0.5, 0.0, f32(params.src_height - 1u));
    let x0_local = u32(floor(src_fx));
    let y0 = u32(floor(src_fy));
    let x1_local = min(x0_local + 1u, region_w - 1u);
    let y1 = min(y0 + 1u, params.src_height - 1u);
    let x0 = min(region_x0 + x0_local, params.src_width - 1u);
    let x1 = min(region_x0 + x1_local, params.src_width - 1u);
    let tx = src_fx - f32(x0_local);
    let ty = src_fy - f32(y0);
    let row0 = y0 * params.src_width;
    let row1 = y1 * params.src_width;
    let v00 = data[row0 + x0];
    let v10 = data[row0 + x1];
    let v01 = data[row1 + x0];
    let v11 = data[row1 + x1];
    let v0 = v00 + (v10 - v00) * tx;
    let v1 = v01 + (v11 - v01) * tx;
    val = v0 + (v1 - v0) * ty;
  } else {
    let src_local_x = min(u32(image_x * f32(region_w) / out_w), region_w - 1u);
    let src_x = min(region_x0 + src_local_x, params.src_width - 1u);
    let src_y = min(u32(image_y * f32(params.src_height) / out_h), params.src_height - 1u);
    let src_idx = src_y * params.src_width + src_x;
    val = data[src_idx];
  }
  if (params.log_scale == 1u) {
    if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); }
  }
  let t = display_normalize(val, params.vmin, params.vmax);
  let lut_idx = min(u32(t * 255.0), 255u);
  return unpack_rgb(lut[lut_idx]);
}
`;

function directSlotGpuRangeShader(integerCounts = false): string {
  return DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct Params {
  src_width: u32,
  src_height: u32,
  src_x0: u32,
  src_region_width: u32,
  out_height: u32,
  out_width: u32,
  origin_x: f32,
  origin_y: f32,
  log_scale: u32,
  bg_rgb: u32,
  zoom: f32,
  smooth_sample: u32,
  vmin_pct: f32,
  vmax_pct: f32,
  pan_x: f32,
  pan_y: f32,
};
struct RangeOut { vmin: f32, vmax: f32, _p0: f32, _p1: f32 };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<${integerCounts ? "u32" : "f32"}>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read> range_in: RangeOut;

fn display_value(index: u32) -> f32 {
  return ${integerCounts ? "f32(data[index]) / range_in._p0" : "data[index]"};
}
struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
  let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  return out;
}

fn unpack_rgb(rgb: u32) -> vec4f {
  let r = f32(rgb & 0xFFu) / 255.0;
  let g = f32((rgb >> 8u) & 0xFFu) / 255.0;
  let b = f32((rgb >> 16u) & 0xFFu) / 255.0;
  return vec4f(r, g, b, 1.0);
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  if (params.src_width == 0u || params.src_height == 0u) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  let region_w = max(1u, min(params.src_region_width, params.src_width));
  let region_x0 = min(params.src_x0, params.src_width - 1u);
  let out_w = f32(max(1u, params.out_width));
  let out_h = f32(max(1u, params.out_height));
  let local_x = in.pos.x - params.origin_x;
  let local_y = in.pos.y - params.origin_y;
  let image_x = (local_x - params.pan_x) / max(params.zoom, 1e-6);
  let image_y = (local_y - params.pan_y) / max(params.zoom, 1e-6);
  if (image_x < 0.0 || image_y < 0.0 || image_x >= out_w || image_y >= out_h) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  var val: f32;
  if (params.smooth_sample == 1u) {
    let src_fx = clamp((image_x + 0.5) * f32(region_w) / out_w - 0.5, 0.0, f32(region_w - 1u));
    let src_fy = clamp((image_y + 0.5) * f32(params.src_height) / out_h - 0.5, 0.0, f32(params.src_height - 1u));
    let x0_local = u32(floor(src_fx));
    let y0 = u32(floor(src_fy));
    let x1_local = min(x0_local + 1u, region_w - 1u);
    let y1 = min(y0 + 1u, params.src_height - 1u);
    let x0 = min(region_x0 + x0_local, params.src_width - 1u);
    let x1 = min(region_x0 + x1_local, params.src_width - 1u);
    let tx = src_fx - f32(x0_local);
    let ty = src_fy - f32(y0);
    let row0 = y0 * params.src_width;
    let row1 = y1 * params.src_width;
    let v00 = display_value(row0 + x0);
    let v10 = display_value(row0 + x1);
    let v01 = display_value(row1 + x0);
    let v11 = display_value(row1 + x1);
    let v0 = v00 + (v10 - v00) * tx;
    let v1 = v01 + (v11 - v01) * tx;
    val = v0 + (v1 - v0) * ty;
  } else {
    let src_local_x = min(u32(image_x * f32(region_w) / out_w), region_w - 1u);
    let src_x = min(region_x0 + src_local_x, params.src_width - 1u);
    let src_y = min(u32(image_y * f32(params.src_height) / out_h), params.src_height - 1u);
    val = display_value(src_y * params.src_width + src_x);
  }
  if (params.log_scale == 1u) {
    if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); }
  }
  let span = range_in.vmax - range_in.vmin;
  let vmin = range_in.vmin + span * clamp(params.vmin_pct, 0.0, 100.0) / 100.0;
  let vmax = range_in.vmin + span * clamp(params.vmax_pct, 0.0, 100.0) / 100.0;
  let t = display_normalize(val, vmin, vmax);
  let lut_idx = min(u32(t * 255.0), 255u);
  return unpack_rgb(lut[lut_idx]);
}
`;
}

function shouldSmoothDirectSample(
  smooth: boolean | undefined,
  zoom: number,
  sourceWidth: number,
  sourceHeight: number,
  panelWidth: number,
  panelHeight: number,
): boolean {
  if (!smooth) return false;
  const scale = Math.max(1e-6, zoom);
  const projectedW = Math.max(1, panelWidth) * scale;
  const projectedH = Math.max(1, panelHeight) * scale;
  return projectedW >= Math.max(1, sourceWidth) * 0.75
    || projectedH >= Math.max(1, sourceHeight) * 0.75;
}

const PACKED_PANEL_TRANSFORM_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct Params {
  dims0: vec4u,  // src_width, src_height, source_panel_width, out_width
  dims1: vec4u,  // out_height, panel_count, cols, rows
  flags: vec4u,  // log_scale, bg_rgb, smooth, use_source_indices
  geom: vec4f,   // gap, _pad0, _pad1, _pad2
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read> ranges: array<vec4f>;
@group(0) @binding(4) var<storage, read> transforms: array<vec4f>;
@group(0) @binding(5) var<storage, read> source_panels: array<u32>;
@group(0) @binding(6) var<storage, read_write> rgba: array<u32>;

fn pack_rgb(rgb: u32) -> u32 {
  return rgb | 0xFF000000u;
}

fn sample_value(src_x0: u32, region_w: u32, image_x: f32, image_y: f32, out_w: f32, out_h: f32) -> f32 {
  if (params.flags.z == 1u) {
    let src_fx = clamp((image_x + 0.5) * f32(region_w) / out_w - 0.5, 0.0, f32(region_w - 1u));
    let src_fy = clamp((image_y + 0.5) * f32(params.dims0.y) / out_h - 0.5, 0.0, f32(params.dims0.y - 1u));
    let x0_local = u32(floor(src_fx));
    let y0 = u32(floor(src_fy));
    let x1_local = min(x0_local + 1u, region_w - 1u);
    let y1 = min(y0 + 1u, params.dims0.y - 1u);
    let x0 = min(src_x0 + x0_local, params.dims0.x - 1u);
    let x1 = min(src_x0 + x1_local, params.dims0.x - 1u);
    let tx = src_fx - f32(x0_local);
    let ty = src_fy - f32(y0);
    let row0 = y0 * params.dims0.x;
    let row1 = y1 * params.dims0.x;
    let v00 = data[row0 + x0];
    let v10 = data[row0 + x1];
    let v01 = data[row1 + x0];
    let v11 = data[row1 + x1];
    let v0 = v00 + (v10 - v00) * tx;
    let v1 = v01 + (v11 - v01) * tx;
    return v0 + (v1 - v0) * ty;
  }
  let src_local_x = min(u32(image_x * f32(region_w) / out_w), region_w - 1u);
  let src_x = min(src_x0 + src_local_x, params.dims0.x - 1u);
  let src_y = min(u32(image_y * f32(params.dims0.y) / out_h), params.dims0.y - 1u);
  return data[src_y * params.dims0.x + src_x];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let out_w_px = params.dims0.w;
  let out_h_px = params.dims1.x;
  if (gid.x >= out_w_px || gid.y >= out_h_px) { return; }
  let out_idx = gid.y * out_w_px + gid.x;
  let bg = pack_rgb(params.flags.y);
  let panel_count = max(1u, params.dims1.y);
  let cols = max(1u, params.dims1.z);
  let rows = max(1u, params.dims1.w);
  let gap = max(0.0, params.geom.x);
  let panel_w = (f32(out_w_px) - gap * f32(cols - 1u)) / f32(cols);
  let panel_h = (f32(out_h_px) - gap * f32(rows - 1u)) / f32(rows);
  if (panel_w <= 0.0 || panel_h <= 0.0) {
    rgba[out_idx] = bg;
    return;
  }
  let pitch_x = panel_w + gap;
  let pitch_y = panel_h + gap;
  let col = u32(floor(f32(gid.x) / pitch_x));
  let row = u32(floor(f32(gid.y) / pitch_y));
  if (col >= cols || row >= rows) {
    rgba[out_idx] = bg;
    return;
  }
  let panel_idx = row * cols + col;
  if (panel_idx >= panel_count) {
    rgba[out_idx] = bg;
    return;
  }
  let local_x = f32(gid.x) - f32(col) * pitch_x;
  let local_y = f32(gid.y) - f32(row) * pitch_y;
  if (local_x < 0.0 || local_y < 0.0 || local_x >= panel_w || local_y >= panel_h) {
    rgba[out_idx] = bg;
    return;
  }
  let transform = transforms[panel_idx];
  let zoom = max(transform.x, 1e-6);
  let image_x = (local_x - transform.y) / zoom;
  let image_y = (local_y - transform.z) / zoom;
  if (image_x < 0.0 || image_y < 0.0 || image_x >= panel_w || image_y >= panel_h) {
    rgba[out_idx] = 0xFF000000u;
    return;
  }
  let source_panel_raw = select(panel_idx, source_panels[panel_idx], params.flags.w == 1u);
  let source_panel = min(source_panel_raw, max(1u, params.dims0.x / max(1u, params.dims0.z)) - 1u);
  let src_x0 = min(source_panel * params.dims0.z, params.dims0.x - 1u);
  let region_w = max(1u, min(params.dims0.z, params.dims0.x - src_x0));
  var val = sample_value(src_x0, region_w, image_x, image_y, panel_w, panel_h);
  let range_info = ranges[panel_idx];
  if (params.flags.x == 1u || range_info.z > 0.5) {
    if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); }
  }
  let t = display_normalize(val, range_info.x, range_info.y);
  let lut_idx = min(u32(t * 255.0), 255u);
  rgba[out_idx] = lut[lut_idx] | 0xFF000000u;
}
`;

// Fullscreen-quad blit shader: reads RGBA u32 buffer, renders to canvas texture
const BLIT_SHADER = /* wgsl */ `
struct BlitParams { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> params: BlitParams;
@group(0) @binding(1) var<storage, read> rgba: array<u32>;

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  // Fullscreen triangle (3 vertices, covers entire clip space)
  var out: VSOut;
  let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
  let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  return out;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let px = min(u32(in.uv.x * f32(params.width)), params.width - 1u);
  let py = min(u32(in.uv.y * f32(params.height)), params.height - 1u);
  let idx = py * params.width + px;
  let packed = rgba[idx];
  let r = f32(packed & 0xFFu) / 255.0;
  let g = f32((packed >> 8u) & 0xFFu) / 255.0;
  let b = f32((packed >> 16u) & 0xFFu) / 255.0;
  return vec4f(r, g, b, 1.0);
}
`;

// True-color passthrough: interleaved RGB float [0,1] -> packed rgba u32 (same
// packing the blit shader reads), no colormap. This is the GPU twin of the CPU
// per-pixel RGB pack loop, moving that work off the UI thread.
const RGB_PASSTHROUGH_SHADER = /* wgsl */ `
struct Params { width: u32, height: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rgb: array<f32>;
@group(0) @binding(2) var<storage, read_write> rgba: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let base = idx * 3u;
  let r = u32(clamp(rgb[base] * 255.0, 0.0, 255.0));
  let g = u32(clamp(rgb[base + 1u] * 255.0, 0.0, 255.0));
  let b = u32(clamp(rgb[base + 2u] * 255.0, 0.0, 255.0));
  rgba[idx] = r | (g << 8u) | (b << 16u) | 0xFF000000u;
}
`;

// Smallest zoom of a volume slice view, Show3DSlices' MIN_ZOOM. The area
// average reads every source pixel an output pixel covers, (fullW / canvasW /
// zoom) per axis, so the clamp is what bounds that work per pixel.
export const VOLUME_SLICE_MIN_ZOOM = 0.5;

// Volume-resident orthogonal slice + colormap in ONE compute pass. The whole 3D
// volume lives on the GPU (uploaded once), in a storage buffer or, for 4k and
// row-padded stacks, a 2D texture array; per scrub only a tiny uniform (axis +
// slice index + vmin/vmax) changes, so there is NO per-frame CPU slice
// extraction and NO per-frame volume re-upload. axis: 0=XY(z fixed), 1=XZ(y
// fixed), 2=YZ(x fixed), 3=oblique cut along the alignment segment. Order
// matches the CPU path: log THEN flip.
function volumeSliceShader(source: "buffer" | "texture"): string {
  const volumeBinding = source === "buffer"
    ? "var<storage, read> vol: array<f32>"
    : "var volTex: texture_2d_array<f32>";
  const volumeLoad = source === "buffer"
    ? "vol[z * p.ny * p.nx + y * p.nx + x]"
    : "textureLoad(volTex, vec2<i32>(i32(x), i32(y)), i32(z), 0).r";
  return DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct VParams {
  nx: u32, ny: u32, nz: u32, axis: u32,
  index: u32, outW: u32, outH: u32, logScale: u32,
  flip: u32, viewMode: u32, canvasW: u32, canvasH: u32,
  vmin: f32, vmax: f32, zoom: f32, panX: f32,
  panY: f32, rowShift: f32, colShift: f32, segStartX: f32,
  segStartY: f32, segStopX: f32, segStopY: f32, _p0: f32,
};
@group(0) @binding(0) var<uniform> p: VParams;
@group(0) @binding(1) ${volumeBinding};
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;

fn loadVolume(z: u32, x: u32, y: u32) -> f32 {
  return ${volumeLoad};
}

fn sampleAligned(zRaw: f32, xRaw: f32, yRaw: f32) -> f32 {
  let z = u32(clamp(round(zRaw), 0.0, f32(p.nz - 1u)));
  let center = (f32(max(p.nz, 1u)) - 1.0) * 0.5;
  let x = clamp(xRaw - (f32(z) - center) * p.colShift, 0.0, f32(p.nx - 1u));
  let y = clamp(yRaw - (f32(z) - center) * p.rowShift, 0.0, f32(p.ny - 1u));
  let x0 = u32(floor(x));
  let y0 = u32(floor(y));
  let x1 = min(p.nx - 1u, x0 + 1u);
  let y1 = min(p.ny - 1u, y0 + 1u);
  let tx = x - f32(x0);
  let ty = y - f32(y0);
  let v00 = loadVolume(z, x0, y0);
  let v10 = loadVolume(z, x1, y0);
  let v01 = loadVolume(z, x0, y1);
  let v11 = loadVolume(z, x1, y1);
  return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

fn sampleSlice(p_axis: u32, p_index: u32, sliceX: u32, sliceY: u32) -> f32 {
  var x: f32; var y: f32; var z: f32;
  if (p_axis == 0u) { x = f32(sliceX); y = f32(sliceY); z = f32(p_index); }          // XY
  else if (p_axis == 1u) { x = f32(sliceX); y = f32(p_index); z = f32(sliceY); }     // XZ
  else if (p_axis == 2u) { x = f32(p_index); y = f32(sliceX); z = f32(sliceY); }     // YZ
  else {
    let denom = max(f32(p.canvasW - 1u), 1.0);
    let t = f32(sliceX) / denom;
    x = mix(p.segStartX, p.segStopX, t);
    y = mix(p.segStartY, p.segStopY, t);
    z = f32(sliceY);
  }
  return sampleAligned(z, x, y);
}

fn signedLog1p(v: f32) -> f32 {
  if (v >= 0.0) { return log(1.0 + v); }
  return -log(1.0 - v);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.outW || gid.y >= p.outH) { return; }
  // Full slice dims for this axis (source resolution).
  var fullW: u32; var fullH: u32;
  if (p.axis == 0u) { fullW = p.nx; fullH = p.ny; }        // XY
  else if (p.axis == 1u) { fullW = p.nx; fullH = p.nz; }   // XZ
  else if (p.axis == 2u) { fullW = p.ny; fullH = p.nz; }    // YZ
  else { fullW = max(1u, p.canvasW); fullH = p.nz; }        // oblique
  var x0: u32; var y0: u32; var x1: u32; var y1: u32;
  if (p.viewMode == 0u) {
    // AREA AVERAGE downsample: this output pixel covers the source block
    // [x0,x1) x [y0,y1). Average every covered source value, so no source pixels
    // are silently skipped when a 4k slice is displayed in a smaller panel. When
    // outW==fullW the block is 1x1 = exact native pixel.
    x0 = (gid.x * fullW) / p.outW;
    y0 = (gid.y * fullH) / p.outH;
    x1 = max(x0 + 1u, ((gid.x + 1u) * fullW) / p.outW);
    y1 = max(y0 + 1u, ((gid.y + 1u) * fullH) / p.outH);
  } else {
    let cw = max(f32(p.canvasW), 1.0);
    let ch = max(f32(p.canvasH), 1.0);
    let z = max(p.zoom, ${VOLUME_SLICE_MIN_ZOOM});
    let cx = cw * 0.5;
    let cy = ch * 0.5;
    let sx0 = (((f32(gid.x) - cx - p.panX) / z) + cx) * f32(fullW) / cw;
    let sy0 = (((f32(gid.y) - cy - p.panY) / z) + cy) * f32(fullH) / ch;
    let sx1 = (((f32(gid.x + 1u) - cx - p.panX) / z) + cx) * f32(fullW) / cw;
    let sy1 = (((f32(gid.y + 1u) - cy - p.panY) / z) + cy) * f32(fullH) / ch;
    let loX = min(sx0, sx1);
    let hiX = max(sx0, sx1);
    let loY = min(sy0, sy1);
    let hiY = max(sy0, sy1);
    if (hiX <= 0.0 || hiY <= 0.0 || loX >= f32(fullW) || loY >= f32(fullH)) {
      rgba[gid.y * p.outW + gid.x] = 0xFF000000u;
      return;
    }
    x0 = u32(clamp(floor(loX), 0.0, f32(fullW - 1u)));
    y0 = u32(clamp(floor(loY), 0.0, f32(fullH - 1u)));
    x1 = max(x0 + 1u, u32(clamp(ceil(hiX), 1.0, f32(fullW))));
    y1 = max(y0 + 1u, u32(clamp(ceil(hiY), 1.0, f32(fullH))));
  }
  var sum = 0.0; var cnt = 0.0;
  for (var yy = y0; yy < y1; yy++) {
    for (var xx = x0; xx < x1; xx++) {
      sum = sum + sampleSlice(p.axis, p.index, min(xx, fullW - 1u), min(yy, fullH - 1u));
      cnt = cnt + 1.0;
    }
  }
  var val = sum / max(cnt, 1.0);
  if (p.logScale == 1u) { val = signedLog1p(val); }
  if (p.flip == 1u) { val = -val; }
  let t = display_normalize(val, p.vmin, p.vmax);
  let li = min(u32(t * 255.0), 255u);
  rgba[gid.y * p.outW + gid.x] = lut[li] | 0xFF000000u;
}
`;
}

const VOLUME_PARAMS_BYTES = 96;

interface VolumeSliceView {
  zoom: number;
  panX: number;
  panY: number;
  canvasW: number;
  canvasH: number;
}

interface VolumeSliceAlignment {
  rowShift: number;
  colShift: number;
  segment?: {
    start: { x: number; y: number };
    stop: { x: number; y: number };
  };
}

/** Clear color for a 0xBBGGRR packed background, the byte order the shaders' unpack_rgb reads. */
function clearColorFromRgb(bgRgb: number): GPUColorDict {
  return { r: (bgRgb & 0xFF) / 255, g: ((bgRgb >> 8) & 0xFF) / 255, b: ((bgRgb >> 16) & 0xFF) / 255, a: 1 };
}

// Tiny per-pass GPU buffers (e.g. 32B region uniforms) that must live until
// the GPU has consumed them. We push them here when recorded into an encoder
// and destroy them once the caller has submitted the work.
const paramsBufQueue: GPUBuffer[] = [];
function flushParamsBufQueue(start = 0): void {
  for (const buffer of paramsBufQueue.splice(start)) buffer.destroy();
}

type GPUBufferOwnership = "owned" | "borrowed";

type GPUSlot = {
  dataBuffer: GPUBuffer;
  dataOwnership: GPUBufferOwnership;
  rgbaBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  paramsBuffer: GPUBuffer;
  blitParamsBuffer: GPUBuffer;
  histBinsBuffer: GPUBuffer;
  // Lazily allocated per-slot 16-byte buffer holding { vmin, vmax, _p0, _p1 }.
  // Populated by computeRange* on GPU and consumed directly by the range-aware
  // colormap shader (no CPU readback between passes).
  rangeBuffer: GPUBuffer | null;
  rangePartialsBuffer: GPUBuffer | null;
  liveRange?: {
    groups: number;
    partials: GPUBuffer;
    parameters: GPUBuffer;
    values: Uint32Array<ArrayBuffer>;
    reduceGroup: GPUBindGroup;
    finishGroup: GPUBindGroup;
    renderGroup: GPUBindGroup | null;
  };
  directGridBindGroup: GPUBindGroup | null;
  directSlotBindGroup: GPUBindGroup | null;
  directRegionParamsBuffers: (GPUBuffer | null)[];
  directRegionBindGroups: (GPUBindGroup | null)[];
  directRegionLutNames: string[];
  count: number;
  rgbaCapacity: number;
  width: number;
  height: number;
  directOnly: boolean;
  dataKind: "f32" | "u8";
};

/**
 * WebGPU colormap engine. Holds persistent data buffers on the GPU; histogram
 * slider changes only update a small uniform, with no data re-upload.
 */
export class GPUColormapEngine {
  readonly path = "WebGPU" as const;
  private device: GPUDevice;
  private pipeline: GPUComputePipeline | null = null;
  private scaledPipeline: GPUComputePipeline | null = null;
  private scaledUint8Pipeline: GPUComputePipeline | null = null;
  private directGridPipeline: GPURenderPipeline | null = null;
  private directSlotPipeline: GPURenderPipeline | null = null;
  private directSlotGpuRangeU32Pipeline: GPURenderPipeline | null = null;
  private directSlotGpuRangePipeline: GPURenderPipeline | null = null;
  private packedPanelTransformPipeline: GPUComputePipeline | null = null;
  private blitPipeline: GPURenderPipeline | null = null;
  // GPU temporal-average state: a compute pipeline that means N window frames
  // into a slot's dataBuffer, plus a reused scratch buffer for the window frames.
  private avgPipeline: GPUComputePipeline | null = null;
  private avgScratch: GPUBuffer | null = null;
  private avgScratchSize = 0;
  private avgParamsBuffer: GPUBuffer | null = null;
  // True-color passthrough state: an RGB->rgba compute pipeline plus reused
  // input/output buffers, so RGB frames paint on the GPU like grayscale.
  private rgbPipeline: GPUComputePipeline | null = null;
  private rgbDataBuffer: GPUBuffer | null = null;
  private rgbDataCapacity = 0;
  private rgbRgbaBuffer: GPUBuffer | null = null;
  private rgbRgbaCapacity = 0;
  private rgbParamsBuffer: GPUBuffer | null = null;
  // Persistent per-image buffers: data, rgba, readback, params, histogram.
  private slots: GPUSlot[] = [];
  private lutBuffer: GPUBuffer | null = null;
  private currentLutName: string = "";
  private namedLutBuffers = new Map<string, GPUBuffer>();
  // Scratch for the 24-byte colormap params; writeBuffer copies it before reuse.
  private colormapParams = new ArrayBuffer(24);
  private directGridParams = new ArrayBuffer(64);
  private directGridParamsU32 = new Uint32Array(this.directGridParams);
  private directGridParamsF32 = new Float32Array(this.directGridParams);
  private packedPanelRangesBuffer: GPUBuffer | null = null;
  private packedPanelRangesCapacity = 0;
  private packedPanelTransformsBuffer: GPUBuffer | null = null;
  private packedPanelTransformsCapacity = 0;
  private packedPanelIndicesBuffer: GPUBuffer | null = null;
  private packedPanelIndicesCapacity = 0;
  // Volume-resident slice pipeline (Show3DSlices): volume uploaded once, slice +
  // colormap done on GPU per scrub - no per-frame CPU extract / re-upload.
  private volumePipeline: GPUComputePipeline | null = null;
  private volumeTexturePipeline: GPUComputePipeline | null = null;
  private volumeBuffer: GPUBuffer | null = null;
  private volumeTexture: GPUTexture | null = null;
  private volTextureView: GPUTextureView | null = null;
  private volUseTexture = false;
  private volTextureWidth = 0;
  private volNx = 0;
  private volNy = 0;
  private volNz = 0;
  private volCount = 0;
  private volParamsBuffer: GPUBuffer | null = null;
  private volRgbaBuffer: GPUBuffer | null = null;
  private volRgbaCapacity = 0;
  private volParams = new ArrayBuffer(VOLUME_PARAMS_BYTES);
  private volParamsU32 = new Uint32Array(this.volParams);
  private volParamsF32 = new Float32Array(this.volParams);
  private volBlitCanvas: OffscreenCanvas | null = null;
  private volBlitContext: GPUCanvasContext | null = null;
  private volBlitFormat: GPUTextureFormat | null = null;
  private volBlitWidth = 0;
  private volBlitHeight = 0;
  private volBlitParamsBuffer: GPUBuffer | null = null;
  private volBlitParams = new Uint32Array(2);
  private volBlitBindGroup: GPUBindGroup | null = null;
  private volComputeBindGroup: GPUBindGroup | null = null;
  private volTextureBindGroup: GPUBindGroup | null = null;
  private retiredSlots: GPUSlot[] = [];

  constructor(device: GPUDevice) { this.device = device; }

  getDevice(): GPUDevice { return this.device; }

  private destroySlot(slot: GPUSlot): void {
    if (slot.dataOwnership === "owned") slot.dataBuffer.destroy();
    slot.rgbaBuffer.destroy();
    slot.readBuffer.destroy();
    slot.paramsBuffer.destroy();
    slot.blitParamsBuffer.destroy();
    slot.histBinsBuffer.destroy();
    slot.rangeBuffer?.destroy();
    slot.rangePartialsBuffer?.destroy();
    slot.liveRange?.partials.destroy();
    slot.liveRange?.parameters.destroy();
    for (const buf of slot.directRegionParamsBuffers) buf?.destroy();
  }

  private createLutBuffer(lut: Uint8Array): GPUBuffer {
    const packed = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      packed[i] = lut[i * 3] | (lut[i * 3 + 1] << 8) | (lut[i * 3 + 2] << 16);
    }
    const buffer = this.device.createBuffer({
      size: packed.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, packed);
    return buffer;
  }

  private namedLutBuffer(name: string, lut: Uint8Array): GPUBuffer {
    const cached = this.namedLutBuffers.get(name);
    if (cached) return cached;
    const buffer = this.createLutBuffer(lut);
    this.namedLutBuffers.set(name, buffer);
    return buffer;
  }

  private retireSlot(slot: GPUSlot): void {
    this.retiredSlots.push(slot);
    void this.device.queue.onSubmittedWorkDone()
      .catch(() => {})
      .finally(() => {
        const idx = this.retiredSlots.indexOf(slot);
        if (idx < 0) return;
        this.retiredSlots.splice(idx, 1);
        this.destroySlot(slot);
      });
  }

  releaseSlot(idx: number): void {
    const slot = this.slots[idx];
    if (!slot) return;
    this.destroySlot(slot);
    this.slots[idx] = null as never;
  }

  private ensurePipeline(): void {
    if (this.pipeline) return;
    const module = this.device.createShaderModule({ code: COLORMAP_SHADER });
    this.pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  private ensureScaledPipeline(): void {
    if (this.scaledPipeline) return;
    const module = this.device.createShaderModule({ code: SCALED_COLORMAP_SHADER });
    this.scaledPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  private ensureScaledUint8Pipeline(): void {
    if (this.scaledUint8Pipeline) return;
    const module = this.device.createShaderModule({ code: SCALED_UINT8_COLORMAP_SHADER });
    this.scaledUint8Pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  private ensureAveragePipeline(): void {
    if (this.avgPipeline) return;
    const module = this.device.createShaderModule({ code: AVERAGE_SHADER });
    this.avgPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  /**
   * Average already-resident frame slots entirely on the GPU.
   *
   * The source buffers are copied GPU-to-GPU into the reused average scratch
   * buffer, then the same temporal-mean compute shader writes the result into
   * `idx`. This keeps Show3D scrubbing off the JS CPU after residency completes.
   */
  averageResidentSlotsInto(idx: number, sourceIndices: number[]): boolean {
    const sources = sourceIndices.map(sourceIdx => this.slots[sourceIdx]).filter(Boolean);
    if (sources.length !== sourceIndices.length || sources.length === 0) return false;
    const first = sources[0];
    if (!sources.every(slot => (
      slot.count === first.count && slot.width === first.width && slot.height === first.height
    ))) return false;

    while (this.slots.length <= idx) this.slots.push(null as never);
    let target = this.slots[idx];
    if (
      !target || target.count !== first.count ||
      target.width !== first.width || target.height !== first.height
    ) {
      this.uploadData(idx, new Float32Array(first.count), first.width, first.height, first.rgbaCapacity);
      target = this.slots[idx];
    }

    this.ensureAveragePipeline();
    const frameBytes = first.count * 4;
    const scratchBytes = sources.length * frameBytes;
    if (!this.avgScratch || this.avgScratchSize < scratchBytes) {
      this.avgScratch?.destroy();
      this.avgScratch = this.device.createBuffer({
        size: scratchBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.avgScratchSize = scratchBytes;
    }
    if (!this.avgParamsBuffer) {
      this.avgParamsBuffer = this.device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(
      this.avgParamsBuffer,
      0,
      new Uint32Array([sources.length, first.count]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.avgPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.avgScratch } },
        { binding: 1, resource: { buffer: target.dataBuffer } },
        { binding: 2, resource: { buffer: this.avgParamsBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    sources.forEach((slot, sourceIdx) => {
      encoder.copyBufferToBuffer(slot.dataBuffer, 0, this.avgScratch!, sourceIdx * frameBytes, frameBytes);
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.avgPipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(first.count / 64));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  private ensureDirectGridPipeline(format: GPUTextureFormat): void {
    if (this.directGridPipeline) return;
    const module = this.device.createShaderModule({ code: DIRECT_GRID_COLORMAP_SHADER });
    this.directGridPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: {
        module,
        entryPoint: "fs",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  private ensureDirectSlotPipeline(format: GPUTextureFormat): void {
    if (this.directSlotPipeline) return;
    const module = this.device.createShaderModule({ code: DIRECT_SLOT_COLORMAP_SHADER });
    this.directSlotPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: {
        module,
        entryPoint: "fs",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  private ensureDirectSlotGpuRangePipeline(format: GPUTextureFormat, integerCounts = false): void {
    if (integerCounts ? this.directSlotGpuRangeU32Pipeline : this.directSlotGpuRangePipeline) return;
    const module = this.device.createShaderModule({ code: directSlotGpuRangeShader(integerCounts) });
    const pipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: {
        module,
        entryPoint: "fs",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
    if (integerCounts) this.directSlotGpuRangeU32Pipeline = pipeline;
    else this.directSlotGpuRangePipeline = pipeline;
  }

  private ensurePackedPanelTransformPipeline(): void {
    if (this.packedPanelTransformPipeline) return;
    const module = this.device.createShaderModule({ code: PACKED_PANEL_TRANSFORM_SHADER });
    this.packedPanelTransformPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  /** Upload LUT to GPU (only when colormap name changes). */
  uploadLUT(lutName: string, lut: Uint8Array): void {
    if (this.currentLutName === lutName && this.lutBuffer) return;
    this.ensurePipeline();
    if (this.lutBuffer) {
      this.lutBuffer.destroy();
      this.volComputeBindGroup = null;
      this.volTextureBindGroup = null;
      for (const slot of this.slots) {
        if (!slot) continue;
        slot.directGridBindGroup = null;
        slot.directSlotBindGroup = null;
        if (slot.liveRange) slot.liveRange.renderGroup = null;
        slot.directRegionBindGroups = slot.directRegionBindGroups.map(() => null);
      }
    }
    this.lutBuffer = this.createLutBuffer(lut);
    this.currentLutName = lutName;
  }

  /**
   * Slot record around an already-written data buffer, with fresh per-slot
   * rgba, readback, params and histogram buffers and no cached bind groups.
   * The params buffer holds 64 bytes: the 24-byte colormap and histogram
   * structs, the 32-byte scaled colormap struct and the 64-byte direct grid struct.
   */
  private createSlot(
    dataBuffer: GPUBuffer, dataOwnership: GPUBufferOwnership, dataKind: "f32" | "u8",
    count: number, width: number, height: number, rgbaCapacity: number, readBytes: number, directOnly: boolean,
  ): GPUSlot {
    return {
      dataBuffer,
      dataOwnership,
      rgbaBuffer: this.device.createBuffer({ size: rgbaCapacity * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      readBuffer: this.device.createBuffer({ size: readBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      paramsBuffer: this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      blitParamsBuffer: this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      histBinsBuffer: this.device.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      rangeBuffer: null,
      rangePartialsBuffer: null,
      directGridBindGroup: null,
      directSlotBindGroup: null,
      directRegionParamsBuffers: [],
      directRegionBindGroups: [],
      directRegionLutNames: [],
      count,
      rgbaCapacity,
      width,
      height,
      directOnly,
      dataKind,
    };
  }

  /** Upload float32 image data for slot `idx`. Only call when data changes. */
  uploadData(idx: number, data: Float32Array, width?: number, height?: number, rgbaCapacityHint?: number, directOnly: boolean = false): void {
    this.ensurePipeline();
    while (this.slots.length <= idx) this.slots.push(null as never);
    // Stale mount-time dims (e.g. width=1 from a closure) do not match the data
    // length; derive near-square dims from the length instead.
    const validDims = width && height && width > 1 && height > 1 && width * height === data.length;
    const slotWidth = validDims ? width : Math.round(Math.sqrt(data.length));
    const slotHeight = validDims ? height : Math.round(data.length / slotWidth);
    const byteSize = data.byteLength;
    const rgbaCapacity = directOnly ? 1 : Math.max(1, Math.round(rgbaCapacityHint ?? data.length));
    const existing = this.slots[idx];
    if (existing && existing.dataOwnership === "owned" && existing.dataKind === "f32" && existing.directOnly === directOnly && existing.count === data.length && existing.width === slotWidth && existing.height === slotHeight && existing.rgbaCapacity >= rgbaCapacity) {
      this.device.queue.writeBuffer(existing.dataBuffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
      return;
    }
    if (existing) {
      this.destroySlot(existing);
    }
    const dataBuffer = this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.device.queue.writeBuffer(dataBuffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    this.slots[idx] = this.createSlot(dataBuffer, "owned", "f32", data.length, slotWidth, slotHeight, rgbaCapacity, rgbaCapacity * 4, directOnly);
  }

  /** Upload one native uint8 image without expanding it to float32. */
  uploadUint8Data(
    idx: number,
    data: Uint8Array | Uint8ClampedArray,
    width: number,
    height: number,
    rgbaCapacityHint: number,
  ): void {
    this.ensureScaledUint8Pipeline();
    while (this.slots.length <= idx) this.slots.push(null as never);
    if (width <= 0 || height <= 0 || width * height !== data.length) {
      throw new Error(`uint8 display dimensions ${width}x${height} do not match ${data.length} pixels`);
    }
    const packedBytes = Math.max(4, Math.ceil(data.byteLength / 4) * 4);
    const upload = packedBytes === data.byteLength ? data : (() => {
      const padded = new Uint8Array(packedBytes);
      padded.set(data);
      return padded;
    })();
    const rgbaCapacity = Math.max(1, Math.round(rgbaCapacityHint));
    const existing = this.slots[idx];
    if (
      existing
      && existing.dataOwnership === "owned"
      && existing.dataKind === "u8"
      && existing.count === data.length
      && existing.width === width
      && existing.height === height
      && existing.rgbaCapacity >= rgbaCapacity
    ) {
      this.device.queue.writeBuffer(existing.dataBuffer, 0, upload.buffer as ArrayBuffer, upload.byteOffset, upload.byteLength);
      return;
    }
    if (existing) this.destroySlot(existing);
    const dataBuffer = this.device.createBuffer({
      size: packedBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.device.queue.writeBuffer(dataBuffer, 0, upload.buffer as ArrayBuffer, upload.byteOffset, upload.byteLength);
    this.slots[idx] = this.createSlot(dataBuffer, "owned", "u8", data.length, width, height, rgbaCapacity, rgbaCapacity * 4, false);
  }

  /** Display caller-owned storage; the caller retains its lifetime. */
  borrowBuffer(idx: number, buffer: GPUBuffer, width: number, height: number): void {
    this.adoptBuffer(idx, buffer, width, height, "borrowed");
  }

  /**
   * Adopt an externally produced float32 GPU buffer as a display slot. The slot
   * owns the buffer after this call unless ownership is explicitly "borrowed".
   * Borrowed storage remains the caller's responsibility and must outlive all
   * queued display work. Slot replacement, release, and engine destruction never
   * destroy borrowed storage. If the same buffer is adopted again, keep
   * the existing display resources so in-place GPU updates can repaint without
   * retiring their own source.
   *
   * @example
   * engine.adoptBuffer(41, residentDisplay.buffer, 512, 512, "borrowed");
   */
  adoptBuffer(idx: number, buffer: GPUBuffer, width: number, height: number, ownership: GPUBufferOwnership = "owned"): void {
    while (this.slots.length <= idx) this.slots.push(null as never);
    const old = this.slots[idx];
    if (old && old.dataBuffer === buffer) {
      if (old.width === width && old.height === height) {
        old.dataOwnership = ownership;
        return;
      }
      // The new slot retains the same storage while old display resources retire.
      old.dataOwnership = "borrowed";
    }
    if (old) this.retireSlot(old);
    const count = Math.max(1, width * height);
    // The 16-byte readback buffer is a placeholder: adopted slots draw without rgba readback.
    this.slots[idx] = this.createSlot(buffer, ownership, "f32", count, width, height, count, 16, false);
  }

  /** Fill the 24-byte colormap (and histogram) params: { width, height, vmin, vmax, log_scale, _pad }. */
  private writeColormapParams(params: ArrayBuffer, width: number, height: number, vmin: number, vmax: number, logScale: boolean): void {
    const paramsU32 = new Uint32Array(params);
    const paramsF32 = new Float32Array(params);
    paramsU32[0] = width;
    paramsU32[1] = height;
    paramsF32[2] = vmin;
    paramsF32[3] = vmax;
    paramsU32[4] = logScale ? 1 : 0;
    paramsU32[5] = 0;
  }

  /**
   * Encode the colormap compute pass that fills `slot`'s packed rgba buffer from
   * its data, display range and `lutBuffer`. The params go through the slot's
   * own uniform, written now, so slots encoded into one submit keep their ranges.
   */
  private encodeColormapPass(
    encoder: GPUCommandEncoder, slot: GPUSlot, vmin: number, vmax: number, logScale: boolean, lutBuffer: GPUBuffer,
  ): void {
    this.writeColormapParams(this.colormapParams, slot.width, slot.height, vmin, vmax, logScale);
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, this.colormapParams);
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: lutBuffer } },
        { binding: 3, resource: { buffer: slot.rgbaBuffer } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
    pass.end();
  }

  /** Apply one slot with an explicitly named LUT without mutating the engine-wide LUT. */
  async applySingleWithLut(
    idx: number,
    vmin: number,
    vmax: number,
    lutName: string,
    lut: Uint8Array,
    logScale: boolean = false,
  ): Promise<Uint8ClampedArray | null> {
    const slot = this.slots[idx];
    if (!this.pipeline || !slot || slot.directOnly || slot.rgbaCapacity < slot.count) return null;
    const lutBuffer = this.namedLutBuffer(lutName, lut);
    const encoder = this.device.createCommandEncoder();
    this.encodeColormapPass(encoder, slot, vmin, vmax, logScale, lutBuffer);
    encoder.copyBufferToBuffer(slot.rgbaBuffer, 0, slot.readBuffer, 0, slot.count * 4);
    this.device.queue.submit([encoder.finish()]);
    await slot.readBuffer.mapAsync(GPUMapMode.READ);
    // The read buffer holds rgbaCapacity pixels; only the first count are this image.
    const rgba = new Uint8ClampedArray(slot.readBuffer.getMappedRange(0, slot.count * 4).slice(0));
    slot.readBuffer.unmap();
    return rgba;
  }

  /**
   * GPU colormap → offscreen canvas in one pass (zero intermediate allocation).
   * Writes from GPU mapped memory directly into ImageData, then putImageData,
   * so no intermediate RGBA array is allocated per frame.
   */
  async renderSlots(
    indices: number[],
    ranges: { vmin: number; vmax: number }[],
    offscreens: (HTMLCanvasElement | null)[],
    imgDatas: (ImageData | null)[],
    logScale: boolean = false,
  ): Promise<number> {
    if (!this.pipeline || !this.lutBuffer || indices.length === 0) return 0;

    const activeSlots: { k: number; slot: GPUSlot }[] = [];
    const encoder = this.device.createCommandEncoder();
    for (let k = 0; k < indices.length; k++) {
      const slot = this.slots[indices[k]];
      if (!slot || slot.directOnly || slot.rgbaCapacity < slot.count || !offscreens[k] || !imgDatas[k]) continue;
      const range = ranges[k] || { vmin: 0, vmax: 1 };
      this.encodeColormapPass(encoder, slot, range.vmin, range.vmax, logScale, this.lutBuffer);
      encoder.copyBufferToBuffer(slot.rgbaBuffer, 0, slot.readBuffer, 0, slot.count * 4);
      activeSlots.push({ k, slot });
    }
    this.device.queue.submit([encoder.finish()]);
    await Promise.all(activeSlots.map(active => active.slot.readBuffer.mapAsync(GPUMapMode.READ)));

    // Copy straight from the mapped range into each ImageData, then onto its
    // canvas; a slot's read buffer can hold more pixels than its image.
    let rendered = 0;
    for (const active of activeSlots) {
      const mapped = active.slot.readBuffer.getMappedRange(0, active.slot.count * 4);
      const imgData = imgDatas[active.k]!;
      imgData.data.set(new Uint8ClampedArray(mapped));
      active.slot.readBuffer.unmap();
      offscreens[active.k]!.getContext("2d")!.putImageData(imgData, 0, 0);
      rendered++;
    }
    return rendered;
  }

  private ensureBlitPipeline(format: GPUTextureFormat): void {
    if (this.blitPipeline) return;
    const module = this.device.createShaderModule({ code: BLIT_SHADER });
    this.blitPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: {
        module, entryPoint: "fs",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  /**
   * Encode the fullscreen blit of a packed rgba buffer into a new width x height
   * OffscreenCanvas. Returns the canvas and its blit params buffer, which the
   * caller destroys once the encoder is submitted, or null when the canvas has
   * no WebGPU context. The blit pipeline must already exist.
   */
  private encodeBlitToOffscreen(
    encoder: GPUCommandEncoder, rgbaBuffer: GPUBuffer, width: number, height: number, format: GPUTextureFormat,
  ): { canvas: OffscreenCanvas; blitParamsBuffer: GPUBuffer } | null {
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!context) return null;
    context.configure({ device: this.device, format, alphaMode: "opaque" });
    const blitParamsBuffer = this.device.createBuffer({
      size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(blitParamsBuffer, 0, new Uint32Array([width, height]));
    const blitGroup = this.device.createBindGroup({
      layout: this.blitPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: blitParamsBuffer } },
        { binding: 1, resource: { buffer: rgbaBuffer } },
      ],
    });
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    renderPass.setPipeline(this.blitPipeline!);
    renderPass.setBindGroup(0, blitGroup);
    renderPass.draw(3);
    renderPass.end();
    return { canvas, blitParamsBuffer };
  }

  /**
   * GPU colormap → OffscreenCanvas → ImageBitmap (zero mapAsync).
   * Compute shader writes RGBA, render pass blits to OffscreenCanvas texture,
   * transferToImageBitmap() returns ImageBitmap for drawImage on 2D canvas.
   * Eliminates the 35ms JS memcpy for 12×4K images.
   */
  renderSlotsToImageBitmap(
    indices: number[],
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean = false,
  ): ImageBitmap[] | null {
    if (!this.lutBuffer) return null;
    const canvases = this.encodeSlotsToOffscreen(indices, ranges, logScale, this.lutBuffer);
    if (!canvases) return null;
    return this.transferOffscreens(canvases);
  }

  private ensureRgbPipeline(): void {
    if (this.rgbPipeline) return;
    const module = this.device.createShaderModule({ code: RGB_PASSTHROUGH_SHADER });
    this.rgbPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  /**
   * Paint an interleaved RGB float frame (3 channels in [0, 1]) to an
   * ImageBitmap entirely on the GPU: a passthrough compute shader packs the
   * channels into rgba, then the existing blit renders it. Returns null (so the
   * caller falls back to the CPU pack loop) if WebGPU is unavailable, the frame
   * is too large for the device's storage-buffer limit, or a validation error
   * fires. The device already requests the adapter's max buffer limits, so a 4K
   * color slot fits where the 128 MB default would have silently failed.
   */
  renderRgbToImageBitmap(rgb: Float32Array, width: number, height: number): ImageBitmap | null {
    if (width <= 0 || height <= 0) return null;
    const count = width * height;
    if (rgb.length < count * 3) return null;
    const rgbBytes = count * 3 * 4;
    const rgbaBytes = count * 4;
    const maxBind = this.device.limits.maxStorageBufferBindingSize;
    if (rgbBytes > maxBind || rgbaBytes > maxBind) return null;  // too big for one buffer -> CPU path
    try {
      this.ensureRgbPipeline();
      const format = navigator.gpu.getPreferredCanvasFormat();
      this.ensureBlitPipeline(format);
      if (!this.rgbPipeline || !this.blitPipeline) return null;

      if (!this.rgbDataBuffer || this.rgbDataCapacity < rgbBytes) {
        this.rgbDataBuffer?.destroy();
        this.rgbDataBuffer = this.device.createBuffer({
          size: rgbBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.rgbDataCapacity = rgbBytes;
      }
      if (!this.rgbRgbaBuffer || this.rgbRgbaCapacity < rgbaBytes) {
        this.rgbRgbaBuffer?.destroy();
        this.rgbRgbaBuffer = this.device.createBuffer({
          size: rgbaBytes, usage: GPUBufferUsage.STORAGE,
        });
        this.rgbRgbaCapacity = rgbaBytes;
      }
      if (!this.rgbParamsBuffer) {
        this.rgbParamsBuffer = this.device.createBuffer({
          size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
      }
      const src = rgb.length === count * 3 ? rgb : rgb.subarray(0, count * 3);
      this.device.queue.writeBuffer(this.rgbDataBuffer, 0, src.buffer, src.byteOffset, count * 3 * 4);
      this.device.queue.writeBuffer(this.rgbParamsBuffer, 0, new Uint32Array([width, height, 0, 0]));

      const encoder = this.device.createCommandEncoder();
      const computeGroup = this.device.createBindGroup({
        layout: this.rgbPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.rgbParamsBuffer } },
          { binding: 1, resource: { buffer: this.rgbDataBuffer } },
          { binding: 2, resource: { buffer: this.rgbRgbaBuffer } },
        ],
      });
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(this.rgbPipeline);
      computePass.setBindGroup(0, computeGroup);
      computePass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
      computePass.end();
      const blit = this.encodeBlitToOffscreen(encoder, this.rgbRgbaBuffer, width, height, format);
      if (!blit) return null;
      this.device.queue.submit([encoder.finish()]);
      blit.blitParamsBuffer.destroy();
      return blit.canvas.transferToImageBitmap();
    } catch {
      return null;
    }
  }

  /**
   * Async twin of renderSlotsToImageBitmap: waits for the submitted GPU work to
   * complete BEFORE snapshotting each OffscreenCanvas. Without the wait,
   * transferToImageBitmap() can snapshot the render pass's clear color (black)
   * before the blit executes, which is what produces the flaky black canvas
   * when the GPU queue is backed up (e.g. a 128 MB frame still draining over a
   * slow SSH tunnel keeps writeBuffer work ahead of the colormap pass).
   * The paint names its colormap instead of reading the engine-wide LUT, so
   * pipelines that paint from separate animation frames (image and FFT panels)
   * cannot repaint each other's panels in the wrong colormap.
   */
  async renderSlotsToImageBitmapAsync(
    indices: number[],
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean,
    lutName: string,
    lut: Uint8Array,
  ): Promise<ImageBitmap[] | null> {
    this.ensurePipeline();
    const canvases = this.encodeSlotsToOffscreen(indices, ranges, logScale, this.namedLutBuffer(lutName, lut));
    if (!canvases) return null;
    await this.device.queue.onSubmittedWorkDone();
    return this.transferOffscreens(canvases);
  }

  /** Snapshot each encoded canvas as an ImageBitmap, keeping null slots in place. */
  private transferOffscreens(canvases: (OffscreenCanvas | null)[]): ImageBitmap[] {
    return canvases.map((canvas) => canvas ? canvas.transferToImageBitmap() : null as never);
  }

  /**
   * Encode colormap + blit for each slot into its own OffscreenCanvas in one
   * submit. Slots that are missing, direct-only or under capacity get null.
   */
  private encodeSlotsToOffscreen(
    indices: number[],
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean,
    lutBuffer: GPUBuffer,
  ): (OffscreenCanvas | null)[] | null {
    if (!this.pipeline || indices.length === 0) return null;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureBlitPipeline(format);
    if (!this.blitPipeline) return null;

    const encoder = this.device.createCommandEncoder();
    const canvases: (OffscreenCanvas | null)[] = [];
    const tempBuffers: GPUBuffer[] = [];
    for (let k = 0; k < indices.length; k++) {
      const slot = this.slots[indices[k]];
      if (!slot || slot.directOnly || slot.rgbaCapacity < slot.count) { canvases.push(null as never); continue; }
      const range = ranges[k] || { vmin: 0, vmax: 1 };
      this.encodeColormapPass(encoder, slot, range.vmin, range.vmax, logScale, lutBuffer);
      const blit = this.encodeBlitToOffscreen(encoder, slot.rgbaBuffer, slot.width, slot.height, format)!;
      tempBuffers.push(blit.blitParamsBuffer);
      canvases.push(blit.canvas);
    }
    this.device.queue.submit([encoder.finish()]);
    for (const buffer of tempBuffers) buffer.destroy();
    return canvases;
  }

  private ensureVolumePipeline(): void {
    if (this.volumePipeline) return;
    const module = this.device.createShaderModule({ code: volumeSliceShader("buffer") });
    this.volumePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    if (!this.volParamsBuffer) {
      this.volParamsBuffer = this.device.createBuffer({
        size: VOLUME_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
  }

  private ensureVolumeTexturePipeline(): void {
    if (this.volumeTexturePipeline) return;
    const module = this.device.createShaderModule({ code: volumeSliceShader("texture") });
    this.volumeTexturePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    if (!this.volParamsBuffer) {
      this.volParamsBuffer = this.device.createBuffer({
        size: VOLUME_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
  }

  /**
   * Upload a 3D volume (nz, ny, nx) row-major float32 into GPU memory once.
   * 4k and row-padded large stacks use a 2D texture array to avoid
   * storage-buffer binding limits. The texture path keeps original float32
   * values; unaligned rows are padded only in the upload stride and never
   * sampled. Other shapes use the storage-buffer path.
   */
  uploadVolume(vol: Float32Array, nx: number, ny: number, nz: number): boolean {
    const rowBytes = nx * 4;
    const paddedRowBytes = Math.ceil(rowBytes / 256) * 256;
    const textureWidth = paddedRowBytes / 4;
    const canTexture = textureWidth <= this.device.limits.maxTextureDimension2D &&
      ny <= this.device.limits.maxTextureDimension2D &&
      nz <= this.device.limits.maxTextureArrayLayers;
    if (canTexture) {
      try {
        this.ensureVolumeTexturePipeline();
        const needsTexture = !this.volumeTexture || this.volCount !== vol.length ||
          this.volNx !== nx || this.volNy !== ny || this.volNz !== nz ||
          this.volTextureWidth !== textureWidth;
        if (needsTexture) {
          this.volumeTexture?.destroy();
          this.volumeTexture = this.device.createTexture({
            size: { width: textureWidth, height: ny, depthOrArrayLayers: nz },
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
          });
          this.volTextureView = this.volumeTexture.createView({ dimension: "2d-array" });
          this.volTextureBindGroup = null;
          this.volCount = vol.length;
        }
        const texture = this.volumeTexture;
        if (!texture) return false;
        if (paddedRowBytes === rowBytes) {
          this.device.queue.writeTexture(
            { texture },
            vol.buffer as ArrayBuffer,
            { offset: vol.byteOffset, bytesPerRow: rowBytes, rowsPerImage: ny },
            { width: nx, height: ny, depthOrArrayLayers: nz },
          );
        } else {
          const layer = new Float32Array(textureWidth * ny);
          const sliceStride = nx * ny;
          for (let z = 0; z < nz; z++) {
            const sliceStart = z * sliceStride;
            for (let row = 0; row < ny; row++) {
              const rowStart = sliceStart + row * nx;
              layer.set(vol.subarray(rowStart, rowStart + nx), row * textureWidth);
            }
            this.device.queue.writeTexture(
              { texture, origin: { x: 0, y: 0, z } },
              layer.buffer,
              { bytesPerRow: paddedRowBytes, rowsPerImage: ny },
              { width: nx, height: ny, depthOrArrayLayers: 1 },
            );
          }
        }
        this.volUseTexture = true;
        this.volumeBuffer?.destroy();
        this.volumeBuffer = null;
        this.volComputeBindGroup = null;
        this.volTextureWidth = textureWidth;
        this.volNx = nx; this.volNy = ny; this.volNz = nz;
        return true;
      } catch {
        this.volumeTexture?.destroy();
        this.volumeTexture = null;
        this.volTextureView = null;
        this.volTextureBindGroup = null;
        this.volUseTexture = false;
        this.volTextureWidth = 0;
      }
    }
    this.ensureVolumePipeline();
    this.volumeTexture?.destroy();
    this.volumeTexture = null;
    this.volTextureView = null;
    this.volTextureBindGroup = null;
    this.volTextureWidth = 0;
    const maxBind = this.device.limits.maxStorageBufferBindingSize;
    if (vol.byteLength > maxBind) return false;
    if (!this.volumeBuffer || this.volCount !== vol.length) {
      this.volumeBuffer?.destroy();
      this.volumeBuffer = this.device.createBuffer({
        size: vol.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.volCount = vol.length;
      this.volComputeBindGroup = null;
    }
    this.device.queue.writeBuffer(this.volumeBuffer, 0, vol.buffer as ArrayBuffer, vol.byteOffset, vol.byteLength);
    this.volUseTexture = false;
    this.volNx = nx; this.volNy = ny; this.volNz = nz;
    return true;
  }

  /**
   * Slice the resident volume along `axis` (0=XY, 1=XZ, 2=YZ) at `index`,
   * colormap with the current LUT + vmin/vmax (logScale/flip applied in-shader to
   * match the CPU path), and blit to an ImageBitmap. Returns null if the volume
   * isn't uploaded or the LUT/pipeline isn't ready (caller falls back to CPU).
   */
  renderVolumeSliceToImageBitmap(
    axis: number, index: number,
    range: { vmin: number; vmax: number },
    logScale: boolean, flip: boolean,
    maxOut?: number,
    view?: VolumeSliceView,
    alignment?: VolumeSliceAlignment,
  ): ImageBitmap | null {
    const texturePipeline = this.volUseTexture ? this.volumeTexturePipeline : null;
    const textureView = this.volUseTexture ? this.volTextureView : null;
    const useTexture = texturePipeline != null && textureView != null;
    const bufferPipeline = this.volumePipeline;
    const useBuffer = !useTexture && this.volumeBuffer != null && bufferPipeline != null;
    if ((!useTexture && !useBuffer) || !this.lutBuffer || !this.volParamsBuffer) return null;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureBlitPipeline(format);
    if (!this.blitPipeline) return null;
    const nx = this.volNx, ny = this.volNy, nz = this.volNz;
    const segmentWidth = alignment?.segment
      ? Math.max(1, Math.ceil(Math.hypot(
        alignment.segment.stop.x - alignment.segment.start.x,
        alignment.segment.stop.y - alignment.segment.start.y,
      )) + 1)
      : 1;
    const fullW = axis === 0 ? nx : axis === 1 ? nx : axis === 2 ? ny : segmentWidth;
    const fullH = axis === 0 ? ny : nz;
    // If maxOut is supplied, cap the output raster while still sampling from
    // the full-resolution source. Callers that need native-pixel zoom leave it
    // undefined, so outW/outH stay at the full slice dimensions.
    const cap = maxOut && maxOut > 0 ? maxOut : Math.max(fullW, fullH);
    const scale = Math.min(1, cap / Math.max(fullW, fullH));
    const outW = view ? Math.max(1, Math.round(view.canvasW)) : Math.max(1, Math.round(fullW * scale));
    const outH = view ? Math.max(1, Math.round(view.canvasH)) : Math.max(1, Math.round(fullH * scale));
    const sliceIndex = Math.max(0, Math.min((axis === 2 ? nx : axis === 1 ? ny : nz) - 1, Math.round(index)));
    const rgbaCount = outW * outH;
    if (!this.volRgbaBuffer || this.volRgbaCapacity < rgbaCount) {
      this.volRgbaBuffer?.destroy();
      this.volRgbaBuffer = this.device.createBuffer({
        size: rgbaCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.volRgbaCapacity = rgbaCount;
      this.volBlitBindGroup = null;
      this.volComputeBindGroup = null;
      this.volTextureBindGroup = null;
    }
    const paramsBuffer = this.volParamsBuffer;
    const lutBuffer = this.lutBuffer;
    const rgbaBuffer = this.volRgbaBuffer;
    if (!paramsBuffer || !lutBuffer || !rgbaBuffer) return null;
    // VParams: u32 control block + float contrast / viewport block.
    const paramsU32 = this.volParamsU32; const paramsF32 = this.volParamsF32;
    paramsU32[0] = nx; paramsU32[1] = ny; paramsU32[2] = nz; paramsU32[3] = axis;
    paramsU32[4] = sliceIndex; paramsU32[5] = outW; paramsU32[6] = outH; paramsU32[7] = logScale ? 1 : 0;
    paramsU32[8] = flip ? 1 : 0;
    paramsU32[9] = view ? 1 : 0;
    paramsU32[10] = view ? Math.max(1, Math.round(view.canvasW)) : outW;
    paramsU32[11] = view ? Math.max(1, Math.round(view.canvasH)) : outH;
    paramsF32[12] = range.vmin; paramsF32[13] = range.vmax;
    paramsF32[14] = view ? Math.max(1e-6, view.zoom) : 1;
    paramsF32[15] = view ? view.panX : 0;
    paramsF32[16] = view ? view.panY : 0;
    paramsF32[17] = alignment && Number.isFinite(alignment.rowShift) ? alignment.rowShift : 0;
    paramsF32[18] = alignment && Number.isFinite(alignment.colShift) ? alignment.colShift : 0;
    paramsF32[19] = alignment?.segment ? alignment.segment.start.x : 0;
    paramsF32[20] = alignment?.segment ? alignment.segment.start.y : 0;
    paramsF32[21] = alignment?.segment ? alignment.segment.stop.x : 0;
    paramsF32[22] = alignment?.segment ? alignment.segment.stop.y : 0;
    paramsF32[23] = 0;
    this.device.queue.writeBuffer(paramsBuffer, 0, this.volParams);
    const encoder = this.device.createCommandEncoder();
    if (useTexture) {
      if (!this.volTextureBindGroup) {
        this.volTextureBindGroup = this.device.createBindGroup({
          layout: texturePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: textureView },
            { binding: 2, resource: { buffer: lutBuffer } },
            { binding: 3, resource: { buffer: rgbaBuffer } },
          ],
        });
      }
    } else if (!this.volComputeBindGroup && bufferPipeline && this.volumeBuffer) {
      this.volComputeBindGroup = this.device.createBindGroup({
        layout: bufferPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: this.volumeBuffer } },
          { binding: 2, resource: { buffer: lutBuffer } },
          { binding: 3, resource: { buffer: rgbaBuffer } },
        ],
      });
    }
    const computePass = encoder.beginComputePass();
    if (useTexture) {
      computePass.setPipeline(texturePipeline);
      computePass.setBindGroup(0, this.volTextureBindGroup!);
    } else {
      if (!bufferPipeline || !this.volComputeBindGroup) { computePass.end(); return null; }
      computePass.setPipeline(bufferPipeline);
      computePass.setBindGroup(0, this.volComputeBindGroup);
    }
    computePass.dispatchWorkgroups(Math.ceil(outW / 16), Math.ceil(outH / 16));
    computePass.end();
    const sizeChanged = !this.volBlitCanvas || this.volBlitWidth !== outW || this.volBlitHeight !== outH;
    const formatChanged = this.volBlitFormat !== format;
    if (sizeChanged || formatChanged || !this.volBlitContext) {
      this.volBlitCanvas = new OffscreenCanvas(outW, outH);
      this.volBlitContext = this.volBlitCanvas.getContext("webgpu") as GPUCanvasContext | null;
      if (!this.volBlitContext) return null;
      this.volBlitContext.configure({ device: this.device, format, alphaMode: "opaque" });
      this.volBlitWidth = outW;
      this.volBlitHeight = outH;
      this.volBlitFormat = format;
    }
    if (!this.volBlitParamsBuffer) {
      this.volBlitParamsBuffer = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    this.volBlitParams[0] = outW;
    this.volBlitParams[1] = outH;
    this.device.queue.writeBuffer(this.volBlitParamsBuffer, 0, this.volBlitParams);
    if (!this.volBlitBindGroup) {
      this.volBlitBindGroup = this.device.createBindGroup({
        layout: this.blitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.volBlitParamsBuffer } },
          { binding: 1, resource: { buffer: rgbaBuffer } },
        ],
      });
    }
    const blitContext = this.volBlitContext;
    const blitCanvas = this.volBlitCanvas;
    if (!blitContext || !blitCanvas) return null;
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{ view: blitContext.getCurrentTexture().createView(), loadOp: "clear" as GPULoadOp, storeOp: "store" as GPUStoreOp, clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    renderPass.setPipeline(this.blitPipeline);
    renderPass.setBindGroup(0, this.volBlitBindGroup);
    renderPass.draw(3);
    renderPass.end();
    this.device.queue.submit([encoder.finish()]);
    return blitCanvas.transferToImageBitmap();
  }

  /**
   * Encode the scaled colormap pass: each of the outWidth x outHeight output
   * pixels samples its nearest source pixel through `pipeline` (float32 or
   * packed uint8), writing the 32-byte scaled params into the slot's uniform.
   */
  private encodeScaledColormapPass(
    encoder: GPUCommandEncoder,
    slot: GPUSlot,
    pipeline: GPUComputePipeline,
    lutBuffer: GPUBuffer,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    outWidth: number,
    outHeight: number,
  ): void {
    const params = new ArrayBuffer(32);
    const paramsU32 = new Uint32Array(params);
    const paramsF32 = new Float32Array(params);
    paramsU32[0] = slot.width;
    paramsU32[1] = slot.height;
    paramsU32[2] = outWidth;
    paramsU32[3] = outHeight;
    paramsF32[4] = range.vmin;
    paramsF32[5] = range.vmax;
    paramsU32[6] = logScale ? 1 : 0;
    paramsU32[7] = 0;
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);
    const computeGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: lutBuffer } },
        { binding: 3, resource: { buffer: slot.rgbaBuffer } },
      ],
    });
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, computeGroup);
    computePass.dispatchWorkgroups(Math.ceil(outWidth / 16), Math.ceil(outHeight / 16));
    computePass.end();
  }

  /**
   * GPU colormap one slot, then blit the full-resolution RGBA buffer into a
   * smaller OffscreenCanvas. The fragment shader samples the source buffer by
   * UV, so values stay full-precision through the colormap step while playback
   * avoids creating a 4096x4096 ImageBitmap when the visible canvas is smaller.
   */
  renderSlotScaledToImageBitmap(
    idx: number,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    outW: number,
    outH: number,
  ): ImageBitmap | null {
    if (!this.pipeline || !this.lutBuffer) return null;
    const slot = this.slots[idx];
    if (!slot || slot.directOnly) return null;
    const canvasWidth = Math.max(1, Math.round(outW));
    const canvasHeight = Math.max(1, Math.round(outH));
    if (canvasWidth * canvasHeight > slot.rgbaCapacity) {
      if (slot.rgbaCapacity < slot.count) return null;
      const bitmaps = this.renderSlotsToImageBitmap([idx], [range], logScale);
      return bitmaps?.[0] ?? null;
    }
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureScaledPipeline();
    this.ensureBlitPipeline(format);
    if (!this.scaledPipeline || !this.blitPipeline) return null;
    const encoder = this.device.createCommandEncoder();
    this.encodeScaledColormapPass(encoder, slot, this.scaledPipeline, this.lutBuffer, range, logScale, canvasWidth, canvasHeight);
    const blit = this.encodeBlitToOffscreen(encoder, slot.rgbaBuffer, canvasWidth, canvasHeight, format);
    if (!blit) return null;
    this.device.queue.submit([encoder.finish()]);
    blit.blitParamsBuffer.destroy();
    return blit.canvas.transferToImageBitmap();
  }

  /**
   * Queue-safe scaled render for float32 or packed uint8 slots.
   *
   * Awaiting submitted work before `transferToImageBitmap()` is required on
   * Metal; otherwise a busy queue can snapshot the render pass clear color.
   */
  async renderSlotScaledToImageBitmapAsync(
    idx: number,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    outW: number,
    outH: number,
    lutName: string,
    lut: Uint8Array,
  ): Promise<ImageBitmap | null> {
    const activeLut = this.namedLutBuffer(lutName, lut);
    const slot = this.slots[idx];
    if (!slot || slot.directOnly) return null;
    const canvasWidth = Math.max(1, Math.round(outW));
    const canvasHeight = Math.max(1, Math.round(outH));
    if (canvasWidth * canvasHeight > slot.rgbaCapacity) return null;

    const format = navigator.gpu.getPreferredCanvasFormat();
    if (slot.dataKind === "u8") this.ensureScaledUint8Pipeline();
    else this.ensureScaledPipeline();
    this.ensureBlitPipeline(format);
    const computePipeline = slot.dataKind === "u8" ? this.scaledUint8Pipeline : this.scaledPipeline;
    if (!computePipeline || !this.blitPipeline) return null;
    const encoder = this.device.createCommandEncoder();
    this.encodeScaledColormapPass(encoder, slot, computePipeline, activeLut, range, logScale, canvasWidth, canvasHeight);
    const blit = this.encodeBlitToOffscreen(encoder, slot.rgbaBuffer, canvasWidth, canvasHeight, format);
    if (!blit) return null;
    this.device.queue.submit([encoder.finish()]);
    try {
      await this.device.queue.onSubmittedWorkDone();
      return blit.canvas.transferToImageBitmap();
    } finally {
      blit.blitParamsBuffer.destroy();
    }
  }

  renderSharedGridDirectToCanvas(
    idx: number,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth?: number;
      sharedSource?: boolean;
    },
  ): boolean {
    if (!this.lutBuffer) return false;
    const slot = this.slots[idx];
    if (!slot) return false;
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureDirectGridPipeline(format);
    const pipeline = this.directGridPipeline;
    if (!pipeline) return false;

    const params = new ArrayBuffer(64);
    const paramsU32 = new Uint32Array(params);
    const paramsF32 = new Float32Array(params);
    paramsU32[0] = slot.width;
    paramsU32[1] = slot.height;
    paramsU32[2] = Math.max(1, Math.min(slot.width, Math.round(opts.sourcePanelWidth ?? slot.width)));
    paramsU32[3] = outW;
    paramsU32[4] = outH;
    paramsU32[5] = Math.max(1, Math.round(opts.panelCount));
    paramsU32[6] = Math.max(1, Math.round(opts.cols));
    paramsU32[7] = Math.max(1, Math.round(opts.rows));
    paramsU32[8] = logScale ? 1 : 0;
    paramsU32[9] = opts.bgRgb & 0xFFFFFF;
    paramsU32[10] = opts.sharedSource ? 1 : 0;
    paramsU32[11] = 0;
    paramsF32[12] = range.vmin;
    paramsF32[13] = range.vmax;
    paramsF32[14] = Math.max(0, opts.gap);
    paramsF32[15] = 0;
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

    let bindGroup = slot.directGridBindGroup;
    if (!bindGroup) {
      bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.lutBuffer } },
        ],
      });
      slot.directGridBindGroup = bindGroup;
    }

    const texture = ctx.getCurrentTexture();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  /** Queue-safe durable-frame twin of renderSharedGridDirectToCanvas. */
  async renderSharedGridToImageBitmapAsync(
    idx: number,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth?: number;
      sharedSource?: boolean;
    },
  ): Promise<ImageBitmap | null> {
    const canvas = new OffscreenCanvas(
      Math.max(1, Math.round(opts.width)),
      Math.max(1, Math.round(opts.height)),
    );
    const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!context) return null;
    context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "opaque",
    });
    if (!this.renderSharedGridDirectToCanvas(idx, range, logScale, context, opts)) return null;
    await this.device.queue.onSubmittedWorkDone();
    return canvas.transferToImageBitmap();
  }

  renderPanelSlotsDirectToCanvas(
    indices: number[],
    range: { vmin: number; vmax: number } | { vmin: number; vmax: number }[],
    logScale: boolean | boolean[],
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      transforms?: { zoom: number; panX: number; panY: number }[];
      smooth?: boolean;
    },
  ): boolean {
    const encoder = this.device.createCommandEncoder();
    const rendered = this.encodePanelSlotsDirectToCanvas(encoder, indices, range, logScale, ctx, opts);
    if (rendered) this.device.queue.submit([encoder.finish()]);
    return rendered;
  }

  private encodePanelSlotsDirectToCanvas(
    encoder: GPUCommandEncoder,
    ...[indices, range, logScale, ctx, opts]: Parameters<GPUColormapEngine["renderPanelSlotsDirectToCanvas"]>
  ): boolean {
    if (!this.lutBuffer || indices.length === 0) return false;
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const n = Math.max(1, Math.min(indices.length, Math.round(opts.panelCount)));
    const cols = Math.max(1, Math.round(opts.cols));
    const rows = Math.max(1, Math.round(opts.rows));
    const gap = Math.max(0, opts.gap);
    const panelW = (outW - gap * (cols - 1)) / cols;
    const panelH = (outH - gap * (rows - 1)) / rows;
    if (panelW <= 0 || panelH <= 0) return false;

    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureDirectSlotPipeline(format);
    const pipeline = this.directSlotPipeline;
    if (!pipeline) return false;

    const params = this.directGridParams;
    const paramsU32 = this.directGridParamsU32;
    const paramsF32 = this.directGridParamsF32;
    for (let panel = 0; panel < n; panel++) {
      const slot = this.slots[indices[panel]];
      if (!slot) return false;
      const panelRange = Array.isArray(range) ? (range[panel] ?? range[0]) : range;
      const panelLogScale = Array.isArray(logScale) ? !!logScale[panel] : logScale;
      paramsU32[0] = slot.width;
      paramsU32[1] = slot.height;
      paramsU32[2] = 0;
      paramsU32[3] = slot.width;
      paramsU32[4] = Math.max(1, Math.round(panelH));
      paramsU32[5] = Math.max(1, Math.round(panelW));
      const col = panel % cols;
      const row = Math.floor(panel / cols);
      paramsF32[6] = col * (panelW + gap);
      paramsF32[7] = row * (panelH + gap);
      paramsU32[8] = panelLogScale ? 1 : 0;
      paramsU32[9] = opts.bgRgb & 0xFFFFFF;
      const transform = opts.transforms?.[panel];
      const zoomValue = Math.max(1e-6, transform?.zoom ?? 1);
      paramsF32[10] = zoomValue;
      paramsU32[11] = shouldSmoothDirectSample(opts.smooth, zoomValue, slot.width, slot.height, panelW, panelH) ? 1 : 0;
      paramsF32[12] = panelRange.vmin;
      paramsF32[13] = panelRange.vmax;
      paramsF32[14] = transform?.panX ?? 0;
      paramsF32[15] = transform?.panY ?? 0;
      this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);
      if (!slot.directSlotBindGroup) {
        slot.directSlotBindGroup = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: slot.paramsBuffer } },
            { binding: 1, resource: { buffer: slot.dataBuffer } },
            { binding: 2, resource: { buffer: this.lutBuffer } },
          ],
        });
      }
    }

    const texture = ctx.getCurrentTexture();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: clearColorFromRgb(opts.bgRgb),
      }],
    });
    pass.setPipeline(pipeline);
    for (let panel = 0; panel < n; panel++) {
      const slot = this.slots[indices[panel]];
      if (!slot?.directSlotBindGroup) continue;
      const col = panel % cols;
      const row = Math.floor(panel / cols);
      const x = col * (panelW + gap);
      const y = row * (panelH + gap);
      const sx = Math.max(0, Math.floor(x));
      const sy = Math.max(0, Math.floor(y));
      const sw = Math.max(1, Math.ceil(panelW));
      const sh = Math.max(1, Math.ceil(panelH));
      pass.setViewport(x, y, panelW, panelH, 0, 1);
      pass.setScissorRect(sx, sy, Math.min(sw, outW - sx), Math.min(sh, outH - sy));
      pass.setBindGroup(0, slot.directSlotBindGroup);
      pass.draw(3);
    }
    pass.end();
    return true;
  }

  /**
   * Queue-safe offscreen twin of renderPanelSlotsDirectToCanvas.
   *
   * Browser presentation textures may be discarded after compositing. Render
   * to an OffscreenCanvas and transfer only after the GPU queue completes when
   * callers need a durable scientific frame.
   */
  async renderPanelSlotsToImageBitmapAsync(
    indices: number[],
    range: { vmin: number; vmax: number } | { vmin: number; vmax: number }[],
    logScale: boolean | boolean[],
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      transforms?: { zoom: number; panX: number; panY: number }[];
      smooth?: boolean;
    },
  ): Promise<ImageBitmap | null> {
    const canvas = new OffscreenCanvas(
      Math.max(1, Math.round(opts.width)),
      Math.max(1, Math.round(opts.height)),
    );
    const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!context) return null;
    context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "opaque",
    });
    if (!this.renderPanelSlotsDirectToCanvas(indices, range, logScale, context, opts)) return null;
    await this.device.queue.onSubmittedWorkDone();
    return canvas.transferToImageBitmap();
  }

  renderSlotDirectWithGpuRangeToCanvas(
    idx: number,
    vminPct: number,
    vmaxPct: number,
    logScale: boolean,
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      bgRgb: number;
      transform?: { zoom: number; panX: number; panY: number };
      smooth?: boolean;
    },
  ): boolean {
    return this.renderSlotsDirectWithGpuRangeToCanvases([idx], [ctx], vminPct, vmaxPct, logScale, opts) === 1;
  }

  /** Present distinct resident slots with one queue submission. */
  renderSlotsDirectWithGpuRangeToCanvases(
    indices: number[], contexts: GPUCanvasContext[], vminPct: number, vmaxPct: number,
    logScale: boolean,
    opts: { width: number; height: number; bgRgb: number;
      transform?: { zoom: number; panX: number; panY: number }; smooth?: boolean },
  ): number {
    if (indices.length !== contexts.length || new Set(indices).size !== indices.length) {
      throw new Error("Supply one distinct resident slot for each canvas context.");
    }
    const encoder = this.device.createCommandEncoder();
    const paramsStart = paramsBufQueue.length;
    let rendered = 0;
    try {
      indices.forEach((idx, i) => {
        if (this.encodeSlotDirectWithGpuRangeToCanvas(encoder, idx, vminPct, vmaxPct, logScale, contexts[i], opts)) rendered++;
      });
      if (rendered) this.device.queue.submit([encoder.finish()]);
      return rendered;
    } finally {
      flushParamsBufQueue(paramsStart);
    }
  }

  /**
   * Draw distinct resident images into one canvas, retaining per-image GPU ranges.
   * Rectangles use canvas pixels. Pan uses source pixels, as in scientific views.
   * Source buffers remain unchanged; only display uniforms and ranges are written.
   * Optional count views are borrowed for this submission, keyed by slot index.
   * Their float32 means feed the same range/log/interpolation operations as float
   * slots. They never replace owned slot buffers and must outlive queued work.
   */
  renderSlotsDirectWithGpuRangeToCanvas(
    indices: number[],
    rectangles: { x: number; y: number; width: number; height: number }[],
    ctx: GPUCanvasContext,
    vminPct: number,
    vmaxPct: number,
    logScale: boolean,
    opts: { width: number; height: number; bgRgb: number;
      transform?: { zoom: number; panX: number; panY: number }; smooth?: boolean; counts?: ReadonlyMap<number, Uint32ImageView> },
  ): number {
    if (indices.length !== rectangles.length || new Set(indices).size !== indices.length) {
      throw new Error("Supply one rectangle for each distinct resident slot.");
    }
    if (!Number.isInteger(opts.width) || !Number.isInteger(opts.height) || opts.width < 1 || opts.height < 1) {
      throw new Error("Canvas dimensions must be positive integer pixels.");
    }
    for (const rect of rectangles) {
      if (![rect.x, rect.y, rect.width, rect.height].every(Number.isInteger)
        || rect.x < 0 || rect.y < 0 || rect.width < 1 || rect.height < 1
        || rect.x + rect.width > opts.width || rect.y + rect.height > opts.height) {
        throw new Error("Image rectangles must be positive integer pixel regions inside the canvas.");
      }
    }
    if (opts.counts) {
      if (opts.counts.size !== indices.length) throw new Error('Supply one count image for every shared display slot.');
      for (const idx of indices) {
        const slot = this.slots[idx], view = opts.counts.get(idx);
        if (!slot || !view) throw new Error('Every count image requires an initialized display slot.');
        validateUint32ImageView(view, this.device, slot.count);
      }
    }
    if (!this.lutBuffer || !indices.length) return 0;
    this.ensureRangeRegionPipeline(Boolean(opts.counts));
    this.ensureDirectSlotGpuRangePipeline(navigator.gpu.getPreferredCanvasFormat(), Boolean(opts.counts));
    const pipeline = opts.counts ? this.directSlotGpuRangeU32Pipeline : this.directSlotGpuRangePipeline;
    if (!pipeline) return 0;
    const encoder = this.device.createCommandEncoder();
    const paramsStart = paramsBufQueue.length;
    const draws: { rect: typeof rectangles[number]; group: GPUBindGroup }[] = [];
    try {
      indices.forEach((idx, i) => {
        const slot = this.slots[idx], counts = opts.counts?.get(idx);
        if (!slot || !this.recordComputeRangeRegion(encoder, idx, undefined, logScale, counts)) return;
        const rect = rectangles[i];
        const paramsU32 = this.directGridParamsU32;
        const paramsF32 = this.directGridParamsF32;
        paramsU32[0] = slot.width; paramsU32[1] = slot.height; paramsU32[2] = 0; paramsU32[3] = slot.width;
        paramsU32[4] = rect.height; paramsU32[5] = rect.width;
        paramsF32[6] = rect.x; paramsF32[7] = rect.y;
        paramsU32[8] = logScale ? 1 : 0; paramsU32[9] = opts.bgRgb & 0xFFFFFF;
        const zoom = Math.max(1e-6, opts.transform?.zoom ?? 1);
        paramsF32[10] = zoom;
        paramsU32[11] = shouldSmoothDirectSample(opts.smooth, zoom, slot.width, slot.height, rect.width, rect.height) ? 1 : 0;
        paramsF32[12] = vminPct; paramsF32[13] = vmaxPct;
        paramsF32[14] = (opts.transform?.panX ?? 0) * rect.width / slot.width;
        paramsF32[15] = (opts.transform?.panY ?? 0) * rect.height / slot.height;
        this.device.queue.writeBuffer(slot.paramsBuffer, 0, this.directGridParams);
        draws.push({ rect, group: this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: slot.paramsBuffer } },
            { binding: 1, resource: counts
              ? { buffer: counts.buffer, offset: counts.byteOffset, size: counts.count * 4 }
              : { buffer: slot.dataBuffer } },
            { binding: 2, resource: { buffer: this.lutBuffer! } },
            { binding: 3, resource: { buffer: this.ensureSlotRangeBuffer(slot) } },
          ],
        }) });
      });
      if (!draws.length) return 0;
      const pass = encoder.beginRenderPass({ colorAttachments: [{
        view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store",
        clearValue: clearColorFromRgb(opts.bgRgb),
      }] });
      pass.setPipeline(pipeline);
      for (const { rect, group } of draws) {
        pass.setViewport(rect.x, rect.y, rect.width, rect.height, 0, 1);
        pass.setScissorRect(rect.x, rect.y, rect.width, rect.height);
        pass.setBindGroup(0, group);
        pass.draw(3);
      }
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      return draws.length;
    } finally {
      flushParamsBufQueue(paramsStart);
    }
  }

  private encodeSlotDirectWithGpuRangeToCanvas(
    encoder: GPUCommandEncoder,
    idx: number,
    vminPct: number,
    vmaxPct: number,
    logScale: boolean,
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      bgRgb: number;
      transform?: { zoom: number; panX: number; panY: number };
      smooth?: boolean;
    },
  ): boolean {
    if (!this.lutBuffer) return false;
    const slot = this.slots[idx];
    // Direct scalar presentation consumes f32. Packed u8 slots retain their
    // existing uint8-aware scaled renderer and must not be interpreted as f32.
    if (!slot || slot.dataKind !== "f32") return false;
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureDirectSlotGpuRangePipeline(format);
    const pipeline = this.directSlotGpuRangePipeline;
    if (!pipeline) return false;

    this.recordLiveRange(encoder, slot, logScale);

    const params = this.directGridParams;
    const paramsU32 = this.directGridParamsU32;
    const paramsF32 = this.directGridParamsF32;
    paramsU32[0] = slot.width;
    paramsU32[1] = slot.height;
    paramsU32[2] = 0;
    paramsU32[3] = slot.width;
    paramsU32[4] = outH;
    paramsU32[5] = outW;
    paramsF32[6] = 0;
    paramsF32[7] = 0;
    paramsU32[8] = logScale ? 1 : 0;
    paramsU32[9] = opts.bgRgb & 0xFFFFFF;
    const transform = opts.transform;
    const zoomValue = Math.max(1e-6, transform?.zoom ?? 1);
    paramsF32[10] = zoomValue;
    paramsU32[11] = shouldSmoothDirectSample(opts.smooth, zoomValue, slot.width, slot.height, outW, outH) ? 1 : 0;
    paramsF32[12] = vminPct;
    paramsF32[13] = vmaxPct;
    paramsF32[14] = transform?.panX ?? 0;
    paramsF32[15] = transform?.panY ?? 0;
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

    const rangeBuffer = this.ensureSlotRangeBuffer(slot);
    const bindGroup = slot.liveRange!.renderGroup ??= this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: this.lutBuffer } },
        { binding: 3, resource: { buffer: rangeBuffer } },
      ],
    });

    const texture = ctx.getCurrentTexture();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: clearColorFromRgb(opts.bgRgb),
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    return true;
  }

  private renderPackedPanelTransformComputeToCanvas(
    slot: GPUSlot,
    range: { vmin: number; vmax: number } | { vmin: number; vmax: number }[],
    logScale: boolean | boolean[],
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth: number;
      transforms?: { zoom: number; panX: number; panY: number }[];
      sourcePanelIndices?: number[];
      smooth?: boolean;
    },
  ): boolean {
    if (!this.lutBuffer) return false;
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const n = Math.max(1, Math.round(opts.panelCount));
    const cols = Math.max(1, Math.round(opts.cols));
    const rows = Math.max(1, Math.round(opts.rows));
    const gap = Math.max(0, opts.gap);
    const panelW = (outW - gap * (cols - 1)) / cols;
    const panelH = (outH - gap * (rows - 1)) / rows;
    if (panelW <= 0 || panelH <= 0) return false;
    if (slot.rgbaCapacity < outW * outH) return false;

    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensurePackedPanelTransformPipeline();
    this.ensureBlitPipeline(format);
    const pipeline = this.packedPanelTransformPipeline;
    const blitPipeline = this.blitPipeline;
    if (!pipeline || !blitPipeline) return false;

    const sourcePanelW = Math.max(1, Math.min(slot.width, Math.round(opts.sourcePanelWidth)));
    const rangeBytes = n * 16;
    if (!this.packedPanelRangesBuffer || this.packedPanelRangesCapacity < rangeBytes) {
      this.packedPanelRangesBuffer?.destroy();
      this.packedPanelRangesBuffer = this.device.createBuffer({
        size: rangeBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.packedPanelRangesCapacity = rangeBytes;
    }
    if (!this.packedPanelTransformsBuffer || this.packedPanelTransformsCapacity < rangeBytes) {
      this.packedPanelTransformsBuffer?.destroy();
      this.packedPanelTransformsBuffer = this.device.createBuffer({
        size: rangeBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.packedPanelTransformsCapacity = rangeBytes;
    }
    const indexBytes = n * 4;
    if (!this.packedPanelIndicesBuffer || this.packedPanelIndicesCapacity < indexBytes) {
      this.packedPanelIndicesBuffer?.destroy();
      this.packedPanelIndicesBuffer = this.device.createBuffer({
        size: indexBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.packedPanelIndicesCapacity = indexBytes;
    }

    const packedRanges = new Float32Array(n * 4);
    const packedTransforms = new Float32Array(n * 4);
    const packedIndices = new Uint32Array(n);
    let useSmooth = false;
    let usesExplicitSourcePanels = false;
    for (let panel = 0; panel < n; panel++) {
      const panelRange = Array.isArray(range) ? (range[panel] ?? range[0]) : range;
      const panelLogScale = Array.isArray(logScale) ? !!logScale[panel] : logScale;
      const transform = opts.transforms?.[panel];
      const zoomValue = Math.max(1e-6, transform?.zoom ?? 1);
      packedRanges[panel * 4] = panelRange.vmin;
      packedRanges[panel * 4 + 1] = panelRange.vmax;
      packedRanges[panel * 4 + 2] = panelLogScale ? 1 : 0;
      packedRanges[panel * 4 + 3] = 0;
      packedTransforms[panel * 4] = zoomValue;
      packedTransforms[panel * 4 + 1] = transform?.panX ?? 0;
      packedTransforms[panel * 4 + 2] = transform?.panY ?? 0;
      packedTransforms[panel * 4 + 3] = 0;
      const sourcePanel = Math.max(0, Math.round(opts.sourcePanelIndices?.[panel] ?? panel));
      packedIndices[panel] = sourcePanel;
      usesExplicitSourcePanels = usesExplicitSourcePanels || sourcePanel !== panel;
      useSmooth = useSmooth
        || shouldSmoothDirectSample(opts.smooth, zoomValue, sourcePanelW, slot.height, panelW, panelH);
    }
    this.device.queue.writeBuffer(this.packedPanelRangesBuffer, 0, packedRanges);
    this.device.queue.writeBuffer(this.packedPanelTransformsBuffer, 0, packedTransforms);
    this.device.queue.writeBuffer(this.packedPanelIndicesBuffer, 0, packedIndices);

    const params = this.directGridParams;
    const paramsU32 = this.directGridParamsU32;
    const paramsF32 = this.directGridParamsF32;
    paramsU32[0] = slot.width;
    paramsU32[1] = slot.height;
    paramsU32[2] = sourcePanelW;
    paramsU32[3] = outW;
    paramsU32[4] = outH;
    paramsU32[5] = n;
    paramsU32[6] = cols;
    paramsU32[7] = rows;
    paramsU32[8] = 0;
    paramsU32[9] = opts.bgRgb & 0xFFFFFF;
    paramsU32[10] = useSmooth ? 1 : 0;
    paramsU32[11] = usesExplicitSourcePanels ? 1 : 0;
    paramsF32[12] = gap;
    paramsF32[13] = 0;
    paramsF32[14] = 0;
    paramsF32[15] = 0;
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);
    this.device.queue.writeBuffer(slot.blitParamsBuffer, 0, new Uint32Array([outW, outH]));

    const computeGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: this.lutBuffer } },
        { binding: 3, resource: { buffer: this.packedPanelRangesBuffer } },
        { binding: 4, resource: { buffer: this.packedPanelTransformsBuffer } },
        { binding: 5, resource: { buffer: this.packedPanelIndicesBuffer } },
        { binding: 6, resource: { buffer: slot.rgbaBuffer } },
      ],
    });
    const blitGroup = this.device.createBindGroup({
      layout: blitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.blitParamsBuffer } },
        { binding: 1, resource: { buffer: slot.rgbaBuffer } },
      ],
    });

    const texture = ctx.getCurrentTexture();
    const encoder = this.device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, computeGroup);
    computePass.dispatchWorkgroups(Math.ceil(outW / 16), Math.ceil(outH / 16));
    computePass.end();

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: clearColorFromRgb(opts.bgRgb),
      }],
    });
    renderPass.setPipeline(blitPipeline);
    renderPass.setBindGroup(0, blitGroup);
    renderPass.draw(3);
    renderPass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  renderCombinedPanelRegionsDirectToCanvas(
    slotIdx: number,
    range: { vmin: number; vmax: number } | { vmin: number; vmax: number }[],
    logScale: boolean | boolean[],
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth: number;
      transforms?: { zoom: number; panX: number; panY: number }[];
      sourcePanelIndices?: number[];
      panelLuts?: { name: string; lut: Uint8Array }[];
      smooth?: boolean;
    },
  ): boolean {
    if (!this.lutBuffer) return false;
    const slot = this.slots[slotIdx];
    if (!slot) return false;
    const hasActiveTransform = (opts.transforms || []).some((transform) => (
      Math.abs((transform?.zoom ?? 1) - 1) > 1e-6 ||
      Math.abs(transform?.panX ?? 0) > 1e-3 ||
      Math.abs(transform?.panY ?? 0) > 1e-3
    ));
    // The compute shortcut binds the one shared LUT. Any explicit panel LUT
    // contract must use the per-panel fragment bindings, even when all named
    // panel maps currently happen to be identical.
    const hasPanelLuts = Boolean(opts.panelLuts?.length);
    if (!hasActiveTransform && !hasPanelLuts && this.renderPackedPanelTransformComputeToCanvas(slot, range, logScale, ctx, opts)) {
      return true;
    }
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const n = Math.max(1, Math.round(opts.panelCount));
    const cols = Math.max(1, Math.round(opts.cols));
    const rows = Math.max(1, Math.round(opts.rows));
    const gap = Math.max(0, opts.gap);
    const panelW = (outW - gap * (cols - 1)) / cols;
    const panelH = (outH - gap * (rows - 1)) / rows;
    if (panelW <= 0 || panelH <= 0) return false;

    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureDirectSlotPipeline(format);
    const pipeline = this.directSlotPipeline;
    if (!pipeline) return false;

    const params = this.directGridParams;
    const paramsU32 = this.directGridParamsU32;
    const paramsF32 = this.directGridParamsF32;
    const sourcePanelW = Math.max(1, Math.min(slot.width, Math.round(opts.sourcePanelWidth)));
    while (slot.directRegionParamsBuffers.length < n) {
      slot.directRegionParamsBuffers.push(null);
      slot.directRegionBindGroups.push(null);
    }
    for (let panel = 0; panel < n; panel++) {
      let paramsBuffer = slot.directRegionParamsBuffers[panel];
      if (!paramsBuffer) {
        paramsBuffer = this.device.createBuffer({
          size: 64,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        slot.directRegionParamsBuffers[panel] = paramsBuffer;
      }
      const panelLut = opts.panelLuts?.[panel];
      const lutName = panelLut?.name || this.currentLutName;
      const lutBuffer = panelLut
        ? this.namedLutBuffer(panelLut.name, panelLut.lut)
        : this.lutBuffer;
      if (!lutBuffer) continue;
      const panelRange = Array.isArray(range) ? (range[panel] ?? range[0]) : range;
      const panelLogScale = Array.isArray(logScale) ? !!logScale[panel] : logScale;
      const sourcePanel = Math.max(0, Math.round(opts.sourcePanelIndices?.[panel] ?? panel));
      const srcX0 = Math.min(sourcePanel * sourcePanelW, Math.max(0, slot.width - 1));
      const sourceWidth = Math.max(1, Math.min(sourcePanelW, slot.width - srcX0));
      paramsU32[0] = slot.width;
      paramsU32[1] = slot.height;
      paramsU32[2] = srcX0;
      paramsU32[3] = sourceWidth;
      paramsU32[4] = Math.max(1, Math.round(panelH));
      paramsU32[5] = Math.max(1, Math.round(panelW));
      const col = panel % cols;
      const row = Math.floor(panel / cols);
      paramsF32[6] = col * (panelW + gap);
      paramsF32[7] = row * (panelH + gap);
      paramsU32[8] = panelLogScale ? 1 : 0;
      paramsU32[9] = opts.bgRgb & 0xFFFFFF;
      paramsU32[10] = 1;
      const transform = opts.transforms?.[panel];
      const zoomValue = Math.max(1e-6, transform?.zoom ?? 1);
      paramsF32[10] = zoomValue;
      paramsU32[11] = shouldSmoothDirectSample(opts.smooth, zoomValue, sourceWidth, slot.height, panelW, panelH) ? 1 : 0;
      paramsF32[12] = panelRange.vmin;
      paramsF32[13] = panelRange.vmax;
      paramsF32[14] = transform?.panX ?? 0;
      paramsF32[15] = transform?.panY ?? 0;
      this.device.queue.writeBuffer(paramsBuffer, 0, params);
      if (
        !slot.directRegionBindGroups[panel]
        || slot.directRegionLutNames[panel] !== lutName
      ) {
        slot.directRegionBindGroups[panel] = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: slot.dataBuffer } },
            { binding: 2, resource: { buffer: lutBuffer } },
          ],
        });
        slot.directRegionLutNames[panel] = lutName;
      }
    }

    const texture = ctx.getCurrentTexture();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: clearColorFromRgb(opts.bgRgb),
      }],
    });
    pass.setPipeline(pipeline);
    for (let panel = 0; panel < n; panel++) {
      const bindGroup = slot.directRegionBindGroups[panel];
      if (!bindGroup) continue;
      const col = panel % cols;
      const row = Math.floor(panel / cols);
      const x = col * (panelW + gap);
      const y = row * (panelH + gap);
      const sx = Math.max(0, Math.floor(x));
      const sy = Math.max(0, Math.floor(y));
      const sw = Math.max(1, Math.min(Math.ceil(panelW), outW - sx));
      const sh = Math.max(1, Math.min(Math.ceil(panelH), outH - sy));
      pass.setViewport(x, y, panelW, panelH, 0, 1);
      pass.setScissorRect(sx, sy, sw, sh);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  /** Queue-safe durable-frame twin of renderCombinedPanelRegionsDirectToCanvas. */
  async renderCombinedPanelRegionsToImageBitmapAsync(
    slotIdx: number,
    range: { vmin: number; vmax: number } | { vmin: number; vmax: number }[],
    logScale: boolean | boolean[],
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth: number;
      transforms?: { zoom: number; panX: number; panY: number }[];
      sourcePanelIndices?: number[];
      panelLuts?: { name: string; lut: Uint8Array }[];
      smooth?: boolean;
    },
  ): Promise<ImageBitmap | null> {
    const canvas = new OffscreenCanvas(
      Math.max(1, Math.round(opts.width)),
      Math.max(1, Math.round(opts.height)),
    );
    const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!context) return null;
    context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "opaque",
    });
    if (!this.renderCombinedPanelRegionsDirectToCanvas(
      slotIdx,
      range,
      logScale,
      context,
      opts,
    )) return null;
    await this.device.queue.onSubmittedWorkDone();
    return canvas.transferToImageBitmap();
  }

  /**
   * Configure a canvas for WebGPU zero-copy rendering.
   * Returns the GPUCanvasContext, or null if WebGPU canvas is not supported.
   */
  configureCanvas(canvas: HTMLCanvasElement, width: number, height: number): GPUCanvasContext | null {
    try {
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("webgpu") as GPUCanvasContext | null;
      if (!ctx) return null;
      ctx.configure({
        device: this.device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        alphaMode: "opaque",
      });
      return ctx;
    } catch {
      return null;
    }
  }

  /** Release all GPU resources. */
  destroy(): void {
    for (const slot of this.slots) {
      if (slot) this.destroySlot(slot);
    }
    this.slots = [];
    for (const slot of this.retiredSlots) this.destroySlot(slot);
    this.retiredSlots = [];
    this.lutBuffer?.destroy();
    this.lutBuffer = null;
    for (const buffer of this.namedLutBuffers.values()) buffer.destroy();
    this.namedLutBuffers.clear();
    this.currentLutName = "";
    for (const scratch of this.panelRgbaBuffers.values()) { scratch.rgba.destroy(); scratch.range.destroy(); }
    this.panelRgbaBuffers.clear();
    this.volumeBuffer?.destroy(); this.volumeBuffer = null;
    this.volumeTexture?.destroy(); this.volumeTexture = null; this.volTextureView = null;
    this.volParamsBuffer?.destroy(); this.volParamsBuffer = null;
    this.volRgbaBuffer?.destroy(); this.volRgbaBuffer = null;
    this.volBlitParamsBuffer?.destroy(); this.volBlitParamsBuffer = null;
    this.volBlitCanvas = null; this.volBlitContext = null; this.volBlitFormat = null;
    this.volBlitBindGroup = null; this.volComputeBindGroup = null; this.volTextureBindGroup = null; this.volBlitWidth = 0; this.volBlitHeight = 0;
    this.volUseTexture = false;
    this.volCount = 0; this.volRgbaCapacity = 0; this.volTextureWidth = 0;
  }

  /** Number of uploaded image slots. */
  get slotCount(): number { return this.slots.filter(slot => slot).length; }

  /** Resolve once all GPU work submitted so far has completed. */
  async waitForSubmittedWork(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
  }

  // ── GPU min/max reduction ──

  private rangePipeline: GPUComputePipeline | null = null;
  private rangeUint8Pipeline: GPUComputePipeline | null = null;
  private RANGE_WG_SIZE = 256;
  private rangeFinishPipeline: GPUComputePipeline | null = null;

  /** Reduce the current scalar frame and keep both passes and their storage resident. */
  private recordLiveRange(encoder: GPUCommandEncoder, slot: GPUSlot, logScale: boolean): void {
    this.ensureRangePipeline();
    if (!this.rangeFinishPipeline) {
      const module = this.device.createShaderModule({ code: /* wgsl */ `
struct Config { count: u32, groups: u32, logScale: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> partials: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> config: Config;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
var<workgroup> lower: array<f32, 256>;
var<workgroup> upper: array<f32, 256>;
fn scaled(value: f32) -> f32 {
  if (config.logScale == 0u) { return value; }
  if (value >= 0.0) { return log(1.0 + value); }
  return -log(1.0 - value);
}
@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_index) lane: u32) {
  var lo = 1.0e38;
  var hi = -1.0e38;
  for (var index = lane; index < config.groups; index += 256u) {
    lo = min(lo, partials[index].x);
    hi = max(hi, partials[index].y);
  }
  lower[lane] = lo;
  upper[lane] = hi;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      lower[lane] = min(lower[lane], lower[lane + stride]);
      upper[lane] = max(upper[lane], upper[lane + stride]);
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    if (upper[0] >= lower[0]) {
      output[0] = vec4<f32>(scaled(lower[0]), scaled(upper[0]), 0.0, 0.0);
    } else { output[0] = vec4<f32>(0.0); }
  }
}
` });
      this.rangeFinishPipeline = this.device.createComputePipeline({
        layout: "auto", compute: { module, entryPoint: "reduce" },
      });
    }
    if (!slot.liveRange) {
      const groups = Math.min(Math.ceil(slot.count / this.RANGE_WG_SIZE), this.device.limits.maxComputeWorkgroupsPerDimension);
      const partials = this.device.createBuffer({ size: Math.max(8, groups * 8), usage: GPUBufferUsage.STORAGE });
      let parameters: GPUBuffer | null = null;
      try {
        // One block for both passes: the partial pass reads it as RangeParams
        // { count, groups } (its grid stride), the finish pass as Config.
        parameters = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const values = new Uint32Array([slot.count, groups, logScale ? 1 : 0, 0]);
        this.device.queue.writeBuffer(parameters, 0, values);
        slot.liveRange = {
          groups, partials, parameters, values, renderGroup: null,
          reduceGroup: this.device.createBindGroup({
            layout: this.rangePipeline!.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: slot.dataBuffer } },
              { binding: 1, resource: { buffer: partials } },
              { binding: 2, resource: { buffer: parameters } },
            ],
          }),
          finishGroup: this.device.createBindGroup({
            layout: this.rangeFinishPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: partials } },
              { binding: 1, resource: { buffer: parameters } },
              { binding: 2, resource: { buffer: this.ensureSlotRangeBuffer(slot) } },
            ],
          }),
        };
      } catch (error) {
        partials.destroy();
        parameters?.destroy();
        throw error;
      }
    }
    const state = slot.liveRange;
    if (state.values[2] !== Number(logScale)) {
      state.values[2] = Number(logScale);
      this.device.queue.writeBuffer(state.parameters, 0, state.values);
    }
    const partialPass = encoder.beginComputePass();
    partialPass.setPipeline(this.rangePipeline!);
    partialPass.setBindGroup(0, state.reduceGroup);
    partialPass.dispatchWorkgroups(state.groups);
    partialPass.end();
    const finishPass = encoder.beginComputePass();
    finishPass.setPipeline(this.rangeFinishPipeline);
    finishPass.setBindGroup(0, state.finishGroup);
    finishPass.dispatchWorkgroups(1);
    finishPass.end();
  }

  private ensureRangePipeline(): void {
    if (this.rangePipeline) return;
    // Two-pass parallel reduction: each workgroup reduces a chunk to one min/max pair.
    // Output: array of [min, max] pairs (one per workgroup). JS reduces the partials.
    const code = DISPLAY_NORMALIZE_WGSL + RANGE_REDUCE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: RangeParams;

var<workgroup> sMin: array<f32, 256>;
var<workgroup> sMax: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  var local_min = 1.0e38;
  var local_max = -1.0e38;
  let stride = range_stride();
  let steps = range_steps(stride);
  for (var step = 0u; step < steps; step = step + 1u) {
    let i = gid.x + step * stride;
    if (i >= params.count) { break; }
    let value = data[i];
    if (display_is_finite(value)) {
      local_min = min(local_min, value);
      local_max = max(local_max, value);
    }
  }
  sMin[lid.x] = local_min;
  sMax[lid.x] = local_max;

  workgroupBarrier();

  // Tree reduction in shared memory
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid.x < s) {
      sMin[lid.x] = min(sMin[lid.x], sMin[lid.x + s]);
      sMax[lid.x] = max(sMax[lid.x], sMax[lid.x + s]);
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    out[wid.x * 2u] = sMin[0];
    out[wid.x * 2u + 1u] = sMax[0];
  }
}
`;
    const module = this.device.createShaderModule({ code });
    this.rangePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "reduce" },
    });
  }

  private ensureRangeUint8Pipeline(): void {
    if (this.rangeUint8Pipeline) return;
    const code = RANGE_REDUCE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<storage, read> packed_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: RangeParams;

var<workgroup> sMin: array<f32, 256>;
var<workgroup> sMax: array<f32, 256>;

fn read_u8(index: u32) -> f32 {
  let word = packed_data[index >> 2u];
  let shift = (index & 3u) * 8u;
  return f32((word >> shift) & 255u);
}

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  var local_min = 1.0e38;
  var local_max = -1.0e38;
  let stride = range_stride();
  let steps = range_steps(stride);
  for (var step = 0u; step < steps; step = step + 1u) {
    let i = gid.x + step * stride;
    if (i >= params.count) { break; }
    let value = read_u8(i);
    local_min = min(local_min, value);
    local_max = max(local_max, value);
  }
  sMin[lid.x] = local_min;
  sMax[lid.x] = local_max;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid.x < s) {
      sMin[lid.x] = min(sMin[lid.x], sMin[lid.x + s]);
      sMax[lid.x] = max(sMax[lid.x], sMax[lid.x + s]);
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    out[wid.x * 2u] = sMin[0];
    out[wid.x * 2u + 1u] = sMax[0];
  }
}
`;
    const module = this.device.createShaderModule({ code });
    this.rangeUint8Pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "reduce" },
    });
  }

  /**
   * Batch-compute min/max for multiple slots on GPU.
   * Returns { min, max } per slot. One GPU submission for all slots.
   */
  async computeRangeBatch(indices: number[]): Promise<{ min: number; max: number }[]> {
    this.ensureRangePipeline();
    this.ensureRangeUint8Pipeline();
    if (!this.rangePipeline || !this.rangeUint8Pipeline || indices.length === 0) return [];
    const encoder = this.device.createCommandEncoder();
    const jobs: { idx: number; nGroups: number; outBuf: GPUBuffer; readBuf: GPUBuffer; countBuf: GPUBuffer }[] = [];

    for (const idx of indices) {
      const slot = this.slots[idx];
      if (!slot) continue;
      const pipeline = slot.dataKind === "u8" ? this.rangeUint8Pipeline : this.rangePipeline;
      const nGroups = Math.min(
        Math.ceil(slot.count / this.RANGE_WG_SIZE),
        this.device.limits.maxComputeWorkgroupsPerDimension,
      );
      const outSize = nGroups * 2 * 4; // 2 floats (min, max) per workgroup
      const outBuf = this.device.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readBuf = this.device.createBuffer({ size: outSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      // RangeParams { count, groups }: the shader's grid stride is groups * 256.
      const countBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(countBuf, 0, new Uint32Array([slot.count, nGroups, 0, 0]));

      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.dataBuffer } },
          { binding: 1, resource: { buffer: outBuf } },
          { binding: 2, resource: { buffer: countBuf } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(nGroups);
      pass.end();
      encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outSize);
      jobs.push({ idx, nGroups, outBuf, readBuf, countBuf });
    }

    this.device.queue.submit([encoder.finish()]);
    await Promise.all(jobs.map(job => job.readBuf.mapAsync(GPUMapMode.READ)));

    const results: { min: number; max: number }[] = [];
    for (const job of jobs) {
      const partials = new Float32Array(job.readBuf.getMappedRange().slice(0));
      job.readBuf.unmap();
      job.outBuf.destroy(); job.readBuf.destroy(); job.countBuf.destroy();
      // The partials are at most 65535 pairs, so finishing the reduction in JS is cheap.
      let dmin = Infinity, dmax = -Infinity;
      for (let k = 0; k < job.nGroups; k++) {
        if (partials[k * 2] < dmin) dmin = partials[k * 2];
        if (partials[k * 2 + 1] > dmax) dmax = partials[k * 2 + 1];
      }
      if (!(dmax >= dmin)) { dmin = 0; dmax = 0; }
      results.push({ min: dmin, max: dmax });
    }
    return results;
  }

  // ── GPU region min/max → range-aware colormap (no CPU readback) ──
  //
  // Used by Show3D per-panel contrast: each panel is a sub-region of one full
  // frame buffer. We avoid the JS slab-extract + findDataRange loop entirely
  // by reducing on GPU and feeding the result straight into the colormap pass
  // via a small storage buffer (no mapAsync between the two passes).

  private rangeU32Pipelines: [GPUComputePipeline, GPUComputePipeline, GPUComputePipeline] | null = null;
  private rangeRegionPipeline: GPUComputePipeline | null = null;
  private rangePartialsPipeline: GPUComputePipeline | null = null;
  private rangeFinalizePipeline: GPUComputePipeline | null = null;
  private colormapRangePipeline: GPUComputePipeline | null = null;
  // Per-panel scratch state for renderPerPanelGpuExplicit and
  // computeHistogramRegions when N panels share ONE GPU slot (full frame).
  // Each entry holds the panel-sized rgba output buffer and the 16-byte range
  // buffer. Keyed by panel index.
  private panelRgbaBuffers: Map<number, { rgba: GPUBuffer; range: GPUBuffer; size: number }> = new Map();

  private ensurePanelScratch(panel: number, panelPixels: number): { rgba: GPUBuffer; range: GPUBuffer } {
    const want = panelPixels * 4;
    const existing = this.panelRgbaBuffers.get(panel);
    if (existing && existing.size === want) return existing;
    if (existing) { existing.rgba.destroy(); existing.range.destroy(); }
    const rgba = this.device.createBuffer({
      size: want,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const range = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const entry = { rgba, range, size: want };
    this.panelRgbaBuffers.set(panel, entry);
    return entry;
  }

  private ensureRangeRegionPipeline(integerCounts = false): void {
    if (integerCounts ? this.rangeU32Pipelines : this.rangeRegionPipeline) return;
    // Keep the single-workgroup entry point for existing direct callers. Large
    // slot ranges use the same finite/log transform and tree reduction in two
    // stages, distributing the image reads across many workgroups.
    const code = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct RangeOut { vmin: f32, vmax: f32, _p0: f32, _p1: f32 };
struct RegionParams { region: vec4u, fullWidth: u32, log_scale: u32, partial_count: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read> data: array<${integerCounts ? "u32" : "f32"}>;
@group(0) @binding(1) var<uniform> params: RegionParams;
@group(0) @binding(2) var<storage, read_write> out: RangeOut;
@group(0) @binding(3) var<storage, read_write> partials: array<vec2f>;
var<workgroup> sMin: array<f32, 256>;
var<workgroup> sMax: array<f32, 256>;
fn reduce_pair(lid: u32, low: f32, high: f32) -> vec2f {
  sMin[lid] = low; sMax[lid] = high;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lid < stride) {
      sMin[lid] = min(sMin[lid], sMin[lid + stride]);
      sMax[lid] = max(sMax[lid], sMax[lid + stride]);
    }
    workgroupBarrier();
  }
  return vec2f(sMin[0], sMax[0]);
}
fn region_range(lid: u32, first: u32, step: u32) -> vec2f {
  var low = 1.0e38; var high = -1.0e38;
  let width = params.region.z;
  let count = width * params.region.w;
  for (var index = first; index < count; index += step) {
    let row = index / width; let col = index - row * width;
    let at = (params.region.y + row) * params.fullWidth + params.region.x + col;
    var value = ${integerCounts ? "f32(data[at]) / bitcast<f32>(params._pad2)" : "data[at]"};
    if (!display_is_finite(value)) { continue; }
    if (params.log_scale == 1u) {
      if (value >= 0.0) { value = log(1.0 + value); } else { value = -log(1.0 - value); }
    }
    if (value < low) { low = value; }
    if (value > high) { high = value; }
  }
  return reduce_pair(lid, low, high);
}
fn store_range(lid: u32, result: vec2f) {
  if (lid == 0u) {
    if (result.y >= result.x) { out.vmin = result.x; out.vmax = result.y; }
    else { out.vmin = 0.0; out.vmax = 0.0; }
    out._p0 = ${integerCounts ? "bitcast<f32>(params._pad2)" : "0.0"}; out._p1 = 0.0;
  }
}
@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_index) lid: u32) {
  store_range(lid, region_range(lid, lid, 256u));
}
@compute @workgroup_size(256)
fn reduce_partials(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) group: vec3u) {
  // A zero step would never leave the region loop; the host writes >= 1.
  let result = region_range(lid, group.x * 256u + lid, max(params.partial_count, 1u) * 256u);
  if (lid == 0u) { partials[group.x] = result; }
}
@compute @workgroup_size(256)
fn finalize(@builtin(local_invocation_index) lid: u32) {
  var low = 1.0e38; var high = -1.0e38;
  for (var index = lid; index < params.partial_count; index += 256u) {
    low = min(low, partials[index].x); high = max(high, partials[index].y);
  }
  store_range(lid, reduce_pair(lid, low, high));
}
`;
    const module = this.device.createShaderModule({ code });
    const pipeline = (entryPoint: string) => this.device.createComputePipeline({
      layout: "auto", compute: { module, entryPoint },
    });
    if (integerCounts) {
      this.rangeU32Pipelines = [pipeline("reduce"), pipeline("reduce_partials"), pipeline("finalize")];
      return;
    }
    this.rangeRegionPipeline = pipeline("reduce");
    this.rangePartialsPipeline = pipeline("reduce_partials");
    this.rangeFinalizePipeline = pipeline("finalize");
  }

  private ensureColormapRangePipeline(): void {
    if (this.colormapRangePipeline) return;
    // Same as COLORMAP_SHADER but reads vmin/vmax from a storage buffer
    // (filled by recordComputeRangeRegion) and applies the user slider
    // percentages on GPU so those scalars never round-trip through JS.
    // Also accepts a region (offset + size into the full data buffer) and a
    // stride so the colormap output is the panel sub-image, sourced from
    // the full frame in place, with no slab extraction in JS.
    const code = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct Params {
  width: u32,        // output (panel) width
  height: u32,       // output (panel) height
  vmin_pct: f32,
  vmax_pct: f32,
  log_scale: u32,
  src_x: u32,        // region offset x in source data
  src_y: u32,        // region offset y in source data
  src_stride: u32,   // row stride of source data
};
struct RangeOut { vmin: f32, vmax: f32, _p0: f32, _p1: f32 };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;
@group(0) @binding(4) var<storage, read> range_in: RangeOut;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let out_idx = gid.y * params.width + gid.x;
  let src_idx = (params.src_y + gid.y) * params.src_stride + (params.src_x + gid.x);
  var val = data[src_idx];
  if (params.log_scale == 1u) {
    if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); }
  }
  let span = range_in.vmax - range_in.vmin;
  let vmin = range_in.vmin + span * (params.vmin_pct / 100.0);
  let vmax = range_in.vmin + span * (params.vmax_pct / 100.0);
  let t = display_normalize(val, vmin, vmax);
  let lutIdx = min(u32(t * 255.0), 255u);
  let rgb = lut[lutIdx];
  rgba[out_idx] = rgb | 0xFF000000u;
}
`;
    const module = this.device.createShaderModule({ code });
    this.colormapRangePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  private ensureSlotRangeBuffer(slot: GPUSlot): GPUBuffer {
    if (!slot.rangeBuffer) {
      slot.rangeBuffer = this.device.createBuffer({
        // 4 floats: vmin, vmax, _p0, _p1
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    }
    return slot.rangeBuffer;
  }

  /**
   * Reduce a rectangular region of slot `idx`'s data buffer to (vmin, vmax)
   * on GPU and stash the result in `slot.rangeBuffer`. The caller chains a
   * range-aware colormap pass that reads it directly, with no CPU sync.
   *
   * `region` is { x, y, width, height } in pixels into the slot's full frame
   * (which has stride `slot.width`). Omit `region` to scan the whole slot.
   *
   * Records into the supplied encoder so callers can fuse multiple panels
   * into a single submit.
   */
  recordComputeRangeRegion(
    encoder: GPUCommandEncoder,
    idx: number,
    region?: { x: number; y: number; width: number; height: number },
    logScale: boolean = false,
    counts?: Uint32ImageView,
  ): boolean {
    const slot = this.slots[idx];
    if (!slot) return false;
    if (counts) validateUint32ImageView(counts, this.device, slot.count);
    this.ensureRangeRegionPipeline(Boolean(counts));
    const [rangePipeline, partialsPipeline, finalizePipeline] = counts ? this.rangeU32Pipelines!
      : [this.rangeRegionPipeline!, this.rangePartialsPipeline!, this.rangeFinalizePipeline!];
    const input: GPUBufferBinding = counts
      ? { buffer: counts.buffer, offset: counts.byteOffset, size: counts.count * 4 }
      : { buffer: slot.dataBuffer };
    const scanRegion = region ?? { x: 0, y: 0, width: slot.width, height: slot.height };
    const rangeBuf = this.ensureSlotRangeBuffer(slot);
    const partialCount = Math.max(1, Math.min(1024, Math.ceil(scanRegion.width * scanRegion.height / 1024)));
    // These parameters belong to this recorded region. They are never rewritten
    // while another region or an earlier queued frame still references them.
    const paramsBuf = this.device.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const params = new Uint32Array([scanRegion.x, scanRegion.y, scanRegion.width, scanRegion.height, slot.width, logScale ? 1 : 0, partialCount, 0]);
    if (counts) new Float32Array(params.buffer)[7] = counts.divisor;
    this.device.queue.writeBuffer(paramsBuf, 0, params);
    const pass = encoder.beginComputePass();
    if (partialCount === 1) {
      pass.setPipeline(rangePipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: rangePipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: input },
          { binding: 1, resource: { buffer: paramsBuf } },
          { binding: 2, resource: { buffer: rangeBuf } },
        ],
      }));
      pass.dispatchWorkgroups(1);
    } else {
      if (!slot.rangePartialsBuffer) {
        slot.rangePartialsBuffer = this.device.createBuffer({
          size: Math.max(1, Math.min(1024, Math.ceil(slot.count / 1024))) * 8,
          usage: GPUBufferUsage.STORAGE,
        });
      }
      const partials = slot.rangePartialsBuffer;
      pass.setPipeline(partialsPipeline!);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: partialsPipeline!.getBindGroupLayout(0), entries: [
          { binding: 0, resource: input },
          { binding: 1, resource: { buffer: paramsBuf } },
          { binding: 3, resource: { buffer: partials } },
        ],
      }));
      pass.dispatchWorkgroups(partialCount);
      pass.setPipeline(finalizePipeline!);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: finalizePipeline!.getBindGroupLayout(0), entries: [
          { binding: 1, resource: { buffer: paramsBuf } },
          { binding: 2, resource: { buffer: rangeBuf } },
          { binding: 3, resource: { buffer: partials } },
        ],
      }));
      pass.dispatchWorkgroups(1);
    }
    pass.end();
    // The encoder still references paramsBuf; the caller destroys it through
    // flushParamsBufQueue once the encoder is submitted.
    paramsBufQueue.push(paramsBuf);
    return true;
  }

  /**
   * Async range-aware slot rendering that keeps the min/max range on the GPU.
   * This is the no-readback variant used by GPU-resident Show4DSTEM compare
   * panels: range reduction, colormap, and blit are submitted together, then
   * the OffscreenCanvas is snapshotted only after the queue drains.
   */
  async renderSlotsWithComputedGpuRangeAsync(
    indices: number[],
    vminPct: number[],
    vmaxPct: number[],
    logScale: boolean = false,
  ): Promise<ImageBitmap[] | null> {
    this.ensureColormapRangePipeline();
    if (!this.colormapRangePipeline || !this.lutBuffer || indices.length === 0) return null;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureBlitPipeline(format);
    if (!this.blitPipeline) return null;

    const encoder = this.device.createCommandEncoder();
    const params = new ArrayBuffer(32);
    const canvases: (OffscreenCanvas | null)[] = [];
    const tempBuffers: GPUBuffer[] = [];
    let encoded = 0;

    for (const idx of indices) {
      if (this.recordComputeRangeRegion(encoder, idx, undefined, logScale)) encoded++;
    }
    if (encoded === 0) {
      flushParamsBufQueue();
      return null;
    }

    for (let k = 0; k < indices.length; k++) {
      const slot = this.slots[indices[k]];
      if (!slot || slot.directOnly || slot.rgbaCapacity < slot.count || !slot.rangeBuffer) {
        canvases.push(null);
        continue;
      }
      const lowPct = vminPct[k] ?? 0;
      const highPct = vmaxPct[k] ?? 100;

      const paramsU32 = new Uint32Array(params);
      const paramsF32 = new Float32Array(params);
      paramsU32[0] = slot.width;
      paramsU32[1] = slot.height;
      paramsF32[2] = lowPct;
      paramsF32[3] = highPct;
      paramsU32[4] = logScale ? 1 : 0;
      paramsU32[5] = 0;
      paramsU32[6] = 0;
      paramsU32[7] = slot.width;
      this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

      const computeGroup = this.device.createBindGroup({
        layout: this.colormapRangePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.lutBuffer } },
          { binding: 3, resource: { buffer: slot.rgbaBuffer } },
          { binding: 4, resource: { buffer: slot.rangeBuffer } },
        ],
      });
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(this.colormapRangePipeline);
      computePass.setBindGroup(0, computeGroup);
      computePass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
      computePass.end();
      const blit = this.encodeBlitToOffscreen(encoder, slot.rgbaBuffer, slot.width, slot.height, format);
      if (!blit) {
        canvases.push(null);
        continue;
      }
      tempBuffers.push(blit.blitParamsBuffer);
      canvases.push(blit.canvas);
    }

    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    for (const buffer of tempBuffers) buffer.destroy();
    flushParamsBufQueue();
    return this.transferOffscreens(canvases);
  }

  /**
   * Render panel sub-regions with explicit per-panel ranges. Used when
   * Show3D contrast is unlinked and each histogram owns its own clip state.
   */
  renderPerPanelGpuExplicit(
    slotIdx: number,
    regions: { x: number; y: number; width: number; height: number }[],
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean | boolean[] = false,
  ): ImageBitmap[] | null {
    this.ensureColormapRangePipeline();
    if (!this.colormapRangePipeline || !this.lutBuffer) return null;
    const slot = this.slots[slotIdx];
    if (!slot || regions.length === 0) return null;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureBlitPipeline(format);
    if (!this.blitPipeline) return null;

    const encoder = this.device.createCommandEncoder();
    const colormapParams = new ArrayBuffer(32);
    const canvases: (OffscreenCanvas | null)[] = [];
    const tempBuffers: GPUBuffer[] = [];

    for (let k = 0; k < regions.length; k++) {
      const region = regions[k];
      const panelRange = ranges[k] ?? ranges[0];
      if (!region || !panelRange) { canvases.push(null); continue; }
      const scratch = this.ensurePanelScratch(k, region.width * region.height);
      this.device.queue.writeBuffer(
        scratch.range,
        0,
        new Float32Array([panelRange.vmin, panelRange.vmax, 0, 0]),
      );

      const paramsU32 = new Uint32Array(colormapParams);
      const paramsF32 = new Float32Array(colormapParams);
      paramsU32[0] = region.width; paramsU32[1] = region.height;
      paramsF32[2] = 0; paramsF32[3] = 100;
      paramsU32[4] = Array.isArray(logScale) ? (logScale[k] ? 1 : 0) : (logScale ? 1 : 0);
      paramsU32[5] = region.x; paramsU32[6] = region.y; paramsU32[7] = slot.width;
      const colormapParamsBuffer = this.device.createBuffer({
        size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(colormapParamsBuffer, 0, colormapParams);
      tempBuffers.push(colormapParamsBuffer);

      const colormapGroup = this.device.createBindGroup({
        layout: this.colormapRangePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: colormapParamsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.lutBuffer } },
          { binding: 3, resource: { buffer: scratch.rgba } },
          { binding: 4, resource: { buffer: scratch.range } },
        ],
      });
      const colormapPass = encoder.beginComputePass();
      colormapPass.setPipeline(this.colormapRangePipeline);
      colormapPass.setBindGroup(0, colormapGroup);
      colormapPass.dispatchWorkgroups(Math.ceil(region.width / 16), Math.ceil(region.height / 16));
      colormapPass.end();
      const blit = this.encodeBlitToOffscreen(encoder, scratch.rgba, region.width, region.height, format)!;
      tempBuffers.push(blit.blitParamsBuffer);
      canvases.push(blit.canvas);
    }

    this.device.queue.submit([encoder.finish()]);
    for (const buffer of tempBuffers) buffer.destroy();
    return this.transferOffscreens(canvases);
  }

  // ── GPU histogram ──

  private histPipeline: GPUComputePipeline | null = null;
  private histUint8Pipeline: GPUComputePipeline | null = null;
  private histClearPipeline: GPUComputePipeline | null = null;
  private histRegionPipeline: GPUComputePipeline | null = null;

  private ensureHistPipeline(): void {
    if (this.histPipeline) return;
    const code = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct HistParams {
  width: u32,
  height: u32,
  dmin: f32,
  dmax: f32,
  log_scale: u32,
  _pad: u32,
};
@group(0) @binding(0) var<uniform> params: HistParams;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> bins: array<atomic<u32>>;

@compute @workgroup_size(16, 16)
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  var val = data[idx];
  if (!display_is_finite(val)) { return; }
  if (params.log_scale == 1u) { if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); } }
  let t = display_normalize(val, params.dmin, params.dmax);
  let bin = min(u32(t * 256.0), 255u);
  atomicAdd(&bins[bin], 1u);
}

@compute @workgroup_size(256)
fn clear_bins(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < 256u) { atomicStore(&bins[gid.x], 0u); }
}
`;
    const module = this.device.createShaderModule({ code });
    this.histPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "histogram" },
    });
    this.histClearPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "clear_bins" },
    });
  }

  private ensureHistUint8Pipeline(): void {
    if (this.histUint8Pipeline) return;
    const code = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct HistParams {
  width: u32,
  height: u32,
  dmin: f32,
  dmax: f32,
  log_scale: u32,
  _pad: u32,
};
@group(0) @binding(0) var<uniform> params: HistParams;
@group(0) @binding(1) var<storage, read> packed_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> bins: array<atomic<u32>>;

fn read_u8(index: u32) -> f32 {
  let word = packed_data[index >> 2u];
  let shift = (index & 3u) * 8u;
  return f32((word >> shift) & 255u);
}

@compute @workgroup_size(16, 16)
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  var val = read_u8(idx);
  if (params.log_scale == 1u) { val = log(1.0 + val); }
  let t = display_normalize(val, params.dmin, params.dmax);
  let bin = min(u32(t * 256.0), 255u);
  atomicAdd(&bins[bin], 1u);
}
`;
    const module = this.device.createShaderModule({ code });
    this.histUint8Pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "histogram" },
    });
  }

  private ensureHistRegionPipeline(): void {
    if (this.histRegionPipeline) return;
    const code = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct RegionParams {
  region: vec4u,
  full_width: u32,
  log_scale: u32,
  _pad0: u32,
  _pad1: u32,
};
struct RangeIn { vmin: f32, vmax: f32, _p0: f32, _p1: f32 };

@group(0) @binding(0) var<uniform> params: RegionParams;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> range_in: RangeIn;
@group(0) @binding(3) var<storage, read_write> bins: array<atomic<u32>>;

@compute @workgroup_size(16, 16)
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.region.z || gid.y >= params.region.w) { return; }
  let idx = (params.region.y + gid.y) * params.full_width + params.region.x + gid.x;
  var val = data[idx];
  if (!display_is_finite(val)) { return; }
  if (params.log_scale == 1u) { if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); } }
  let t = display_normalize(val, range_in.vmin, range_in.vmax);
  let bin = min(u32(t * 256.0), 255u);
  atomicAdd(&bins[bin], 1u);
}
`;
    const module = this.device.createShaderModule({ code });
    this.histRegionPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "histogram" },
    });
  }

  /**
   * Compute ranges and 256-bin histograms for rectangular regions of one slot.
   *
   * Show3D packs independent panels side-by-side in one resident frame slot.
   * This keeps playback histogram refreshes on WebGPU: range reduction and bin
   * accumulation happen in one submission, with only the small ranges and bins
   * read back for drawing the histogram controls.
   */
  async computeHistogramRegions(
    idx: number,
    regions: { x: number; y: number; width: number; height: number }[],
    logScale: boolean = false,
  ): Promise<{ range: { min: number; max: number }; bins: number[] }[]> {
    this.ensureRangeRegionPipeline();
    this.ensureHistRegionPipeline();
    const slot = this.slots[idx];
    if (!slot || !this.rangeRegionPipeline || !this.histRegionPipeline || regions.length === 0) {
      return [];
    }

    const validRegions = regions.map(region => {
      const x = Math.max(0, Math.min(slot.width - 1, Math.round(region.x)));
      const y = Math.max(0, Math.min(slot.height - 1, Math.round(region.y)));
      return {
        x,
        y,
        width: Math.max(1, Math.min(slot.width - x, Math.round(region.width))),
        height: Math.max(1, Math.min(slot.height - y, Math.round(region.height))),
      };
    });
    const binsBytes = validRegions.length * 256 * 4;
    const rangesBytes = validRegions.length * 16;
    const binsBuffer = this.device.createBuffer({
      size: binsBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const binsReadBuffer = this.device.createBuffer({
      size: binsBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const rangesReadBuffer = this.device.createBuffer({
      size: rangesBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const paramsBuffers: GPUBuffer[] = [];
    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(binsBuffer);

    for (let k = 0; k < validRegions.length; k++) {
      const region = validRegions[k];
      const scratch = this.ensurePanelScratch(k, region.width * region.height);
      const paramsBuffer = this.device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(
        paramsBuffer,
        0,
        new Uint32Array([
          region.x,
          region.y,
          region.width,
          region.height,
          slot.width,
          logScale ? 1 : 0,
          0,
          0,
        ]),
      );
      paramsBuffers.push(paramsBuffer);

      const rangeGroup = this.device.createBindGroup({
        layout: this.rangeRegionPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.dataBuffer } },
          { binding: 1, resource: { buffer: paramsBuffer } },
          { binding: 2, resource: { buffer: scratch.range } },
        ],
      });
      const rangePass = encoder.beginComputePass();
      rangePass.setPipeline(this.rangeRegionPipeline);
      rangePass.setBindGroup(0, rangeGroup);
      rangePass.dispatchWorkgroups(1);
      rangePass.end();

      const histogramGroup = this.device.createBindGroup({
        layout: this.histRegionPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: scratch.range } },
          {
            binding: 3,
            resource: { buffer: binsBuffer, offset: k * 256 * 4, size: 256 * 4 },
          },
        ],
      });
      const histogramPass = encoder.beginComputePass();
      histogramPass.setPipeline(this.histRegionPipeline);
      histogramPass.setBindGroup(0, histogramGroup);
      histogramPass.dispatchWorkgroups(
        Math.ceil(region.width / 16),
        Math.ceil(region.height / 16),
      );
      histogramPass.end();
      encoder.copyBufferToBuffer(scratch.range, 0, rangesReadBuffer, k * 16, 16);
    }

    encoder.copyBufferToBuffer(binsBuffer, 0, binsReadBuffer, 0, binsBytes);
    this.device.queue.submit([encoder.finish()]);

    try {
      await Promise.all([
        binsReadBuffer.mapAsync(GPUMapMode.READ),
        rangesReadBuffer.mapAsync(GPUMapMode.READ),
      ]);
      const rawBins = new Uint32Array(binsReadBuffer.getMappedRange().slice(0));
      const rawRanges = new Float32Array(rangesReadBuffer.getMappedRange().slice(0));
      binsReadBuffer.unmap();
      rangesReadBuffer.unmap();

      return validRegions.map((_, k) => {
        const offset = k * 256;
        let maxCount = 0;
        for (let bin = 0; bin < 256; bin++) {
          maxCount = Math.max(maxCount, rawBins[offset + bin]);
        }
        const bins = new Array<number>(256);
        for (let bin = 0; bin < 256; bin++) {
          bins[bin] = maxCount > 0 ? rawBins[offset + bin] / maxCount : 0;
        }
        return {
          range: { min: rawRanges[k * 4], max: rawRanges[k * 4 + 1] },
          bins,
        };
      });
    } finally {
      for (const buffer of paramsBuffers) buffer.destroy();
      binsBuffer.destroy();
      binsReadBuffer.destroy();
      rangesReadBuffer.destroy();
    }
  }

  /**
   * Batch-compute 256-bin histograms for multiple slots in ONE GPU submission.
   * Uses persistent per-slot histogram buffers (zero create/destroy overhead).
   * Returns normalized bins per image.
   */
  async computeHistogramBatch(
    indices: number[],
    ranges: { min: number; max: number }[],
    logScale: boolean = false,
  ): Promise<number[][]> {
    this.ensureHistPipeline();
    this.ensureHistUint8Pipeline();
    if (!this.histPipeline || !this.histUint8Pipeline || !this.histClearPipeline || indices.length === 0) return [];

    const encoder = this.device.createCommandEncoder();
    const activeSlots: { k: number; slot: GPUSlot; readBuffer: GPUBuffer }[] = [];
    const params = new ArrayBuffer(24);

    for (let k = 0; k < indices.length; k++) {
      const slot = this.slots[indices[k]];
      if (!slot) continue;
      const histogramPipeline = slot.dataKind === "u8" ? this.histUint8Pipeline : this.histPipeline;
      const range = ranges[k] || { min: 0, max: 1 };
      // The histogram params share the colormap params layout.
      this.writeColormapParams(params, slot.width, slot.height, range.min, range.max, logScale);
      this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

      const clearGroup = this.device.createBindGroup({
        layout: this.histClearPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 2, resource: { buffer: slot.histBinsBuffer } },
        ],
      });
      const clearPass = encoder.beginComputePass();
      clearPass.setPipeline(this.histClearPipeline!);
      clearPass.setBindGroup(0, clearGroup);
      clearPass.dispatchWorkgroups(1);
      clearPass.end();

      const histGroup = this.device.createBindGroup({
        layout: histogramPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: slot.histBinsBuffer } },
        ],
      });
      const histPass = encoder.beginComputePass();
      histPass.setPipeline(histogramPipeline);
      histPass.setBindGroup(0, histGroup);
      histPass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
      histPass.end();

      // A per-call readback buffer: one persistent buffer per slot made
      // overlapping histogram requests fail with "outstanding map pending"
      // when Show2D refreshed auto-contrast and the visible histogram during
      // fast interaction.
      const readBuffer = this.device.createBuffer({
        size: 256 * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      encoder.copyBufferToBuffer(slot.histBinsBuffer, 0, readBuffer, 0, 256 * 4);
      activeSlots.push({ k, slot, readBuffer });
    }

    this.device.queue.submit([encoder.finish()]);

    const results: number[][] = indices.map(() => new Array(256).fill(0));
    try {
      await Promise.all(activeSlots.map(active => active.readBuffer.mapAsync(GPUMapMode.READ)));
      for (const active of activeSlots) {
        const rawBins = new Uint32Array(active.readBuffer.getMappedRange().slice(0));
        active.readBuffer.unmap();
        let maxCount = 0;
        for (let bin = 0; bin < 256; bin++) if (rawBins[bin] > maxCount) maxCount = rawBins[bin];
        const norm = new Array(256);
        for (let bin = 0; bin < 256; bin++) norm[bin] = maxCount > 0 ? rawBins[bin] / maxCount : 0;
        results[active.k] = norm;
      }
    } finally {
      for (const active of activeSlots) active.readBuffer.destroy();
    }
    return results;
  }

  /**
   * Compute a 256-bin histogram for slot `idx` on GPU, given known data range.
   * Returns normalized bins (0–1) matching `computeHistogramFromBytes`.
   */
  async computeHistogramWithRange(
    idx: number, dmin: number, dmax: number, logScale: boolean = false,
  ): Promise<number[]> {
    this.ensureHistPipeline();
    this.ensureHistUint8Pipeline();
    const slot = this.slots[idx];
    if (!slot || !this.histPipeline || !this.histUint8Pipeline || !this.histClearPipeline) return new Array(256).fill(0);
    const histogramPipeline = slot.dataKind === "u8" ? this.histUint8Pipeline : this.histPipeline;
    const binsBuffer = this.device.createBuffer({
      size: 256 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readBuffer = this.device.createBuffer({
      size: 256 * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const paramsBuf = this.device.createBuffer({
      size: 24,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const params = new ArrayBuffer(24);
    this.writeColormapParams(params, slot.width, slot.height, dmin, dmax, logScale);
    this.device.queue.writeBuffer(paramsBuf, 0, params);

    const encoder = this.device.createCommandEncoder();
    const clearGroup = this.device.createBindGroup({
      layout: this.histClearPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 2, resource: { buffer: binsBuffer } },
      ],
    });
    const clearPass = encoder.beginComputePass();
    clearPass.setPipeline(this.histClearPipeline);
    clearPass.setBindGroup(0, clearGroup);
    clearPass.dispatchWorkgroups(1);
    clearPass.end();
    const histGroup = this.device.createBindGroup({
      layout: histogramPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: binsBuffer } },
      ],
    });
    const histPass = encoder.beginComputePass();
    histPass.setPipeline(histogramPipeline);
    histPass.setBindGroup(0, histGroup);
    histPass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
    histPass.end();

    encoder.copyBufferToBuffer(binsBuffer, 0, readBuffer, 0, 256 * 4);
    this.device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const rawBins = new Uint32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    binsBuffer.destroy();
    readBuffer.destroy();
    paramsBuf.destroy();
    // Scale like the CPU histogram: the tallest bin is 1.
    let maxCount = 0;
    for (let i = 0; i < 256; i++) if (rawBins[i] > maxCount) maxCount = rawBins[i];
    const result = new Array(256);
    if (maxCount > 0) {
      for (let i = 0; i < 256; i++) result[i] = rawBins[i] / maxCount;
    } else {
      for (let i = 0; i < 256; i++) result[i] = 0;
    }
    return result;
  }
}

/**
 * Create a colormap engine on the hardware adapter, or null when the browser
 * has no WebGPU; callers then draw with applyColormap/renderToOffscreen.
 */
export async function createGPUColormapEngine(): Promise<GPUColormapEngine | null> {
  const device = await getHardwareGPUDevice();
  if (!device) return null;
  // A display pipeline the adapter's shader compiler cannot build paints
  // black without an exception (FXC, below D3D feature level 12, rejected the
  // first level-edge shaders), so such a page draws with the CPU path instead.
  try {
    await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: COLORMAP_SHADER }), entryPoint: "main" },
    });
  } catch (error) {
    console.warn("WebGPU display shaders do not build on this adapter; drawing on the CPU.", error);
    return null;
  }
  return new GPUColormapEngine(device);
}

let gpuColormapEngine: GPUColormapEngine | null = null;
// An engine on a lost device can never paint again; the next caller rebuilds
// one, or gets null and draws with the CPU path once WebGPU is off for the page.
onGPULost(() => { gpuColormapEngine = null; });

/** Get or create the singleton colormap engine, or null without hardware WebGPU. */
export async function getGPUColormapEngine(): Promise<GPUColormapEngine | null> {
  if (gpuColormapEngine) return gpuColormapEngine;
  gpuColormapEngine = await createGPUColormapEngine();
  return gpuColormapEngine;
}
