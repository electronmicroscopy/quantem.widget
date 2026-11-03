// WGSL display shaders of the WebGPU colormap engine. A change to this text
// needs the Windows FXC check: Chrome's D3D12 compiler below feature level 12
// rejects shaders that build everywhere else.

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
export const AVERAGE_SHADER = ROUNDED_QUOTIENT_WGSL + /* wgsl */ `
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
export const DISPLAY_NORMALIZE_WGSL = ROUNDED_QUOTIENT_WGSL + /* wgsl */ `
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
export const RANGE_REDUCE_WGSL = /* wgsl */ `
struct RangeParams { count: u32, groups: u32, _p0: u32, _p1: u32 };
const RANGE_MAX_STEPS = 1024u;
fn range_stride() -> u32 { return max(params.groups, 1u) * 256u; }
fn range_steps(stride: u32) -> u32 { return min(params.count / stride + 1u, RANGE_MAX_STEPS); }
`;

// The full-screen triangle vertex stage and packed-RGB unpack of the direct colormap shaders.
const FULLSCREEN_VERTEX_WGSL = /* wgsl */ `struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

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
`;

export const COLORMAP_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
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

export const SCALED_COLORMAP_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
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
export const SCALED_UINT8_COLORMAP_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
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

export const DIRECT_GRID_COLORMAP_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
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

${FULLSCREEN_VERTEX_WGSL}
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

export const DIRECT_SLOT_COLORMAP_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
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

${FULLSCREEN_VERTEX_WGSL}
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

export const DIRECT_SLOT_GPU_RANGE_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
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
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read> range_in: RangeOut;

fn display_value(index: u32) -> f32 {
  return data[index];
}
${FULLSCREEN_VERTEX_WGSL}
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

export const PACKED_PANEL_TRANSFORM_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct Params {
  dims0: vec4u,  // src_width, src_height, source_panel_width, out_width
  dims1: vec4u,  // out_height, panel_count, cols, rows
  flags: vec4u,  // unused (per-panel log is ranges.z), bg_rgb, smooth, use_source_indices
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
  if (range_info.z > 0.5) {
    if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); }
  }
  let t = display_normalize(val, range_info.x, range_info.y);
  let lut_idx = min(u32(t * 255.0), 255u);
  rgba[out_idx] = lut[lut_idx] | 0xFF000000u;
}
`;

// Fullscreen-quad blit shader: reads RGBA u32 buffer, renders to canvas texture
export const BLIT_SHADER = /* wgsl */ `
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
export const RGB_PASSTHROUGH_SHADER = /* wgsl */ `
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
