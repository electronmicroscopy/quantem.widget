// @ts-nocheck  (node:fs and __dirname read the shader sources; no @types/node here)
// A value exactly on a colormap level edge must get the same LUT entry on the
// Canvas2D and WebGPU paths. The CPU path rounds every float32 step correctly
// (displayNormalize); a GPU `/` may not (WGSL allows 2.5 ULP and Metal uses
// it), and one ULP below an edge is the neighbouring level. The WGSL therefore
// divides through display_divide, an integer long division of the
// significands. These tests replay that WGSL in JavaScript, operation for
// operation in u32 arithmetic, and check it against the CPU path at every
// level edge of several display windows.
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { VOLUME_SLICE_MIN_ZOOM, displayNormalize } from "./display/colormaps";

const COLORMAPS_SOURCE = readFileSync(path.join(__dirname, "display", "colormaps.ts"), "utf8");
const f32 = Math.fround;
const bits = new DataView(new ArrayBuffer(4));
const floatBits = (value: number) => { bits.setFloat32(0, value); return bits.getUint32(0); };
const bitsFloat = (word: number) => { bits.setUint32(0, word >>> 0); return bits.getFloat32(0); };
const firstLeadingBit = (value: number) => 31 - Math.clz32(value);

/** rounded_quotient, replayed in u32 arithmetic. */
function roundedQuotient(numerator: number, denominator: number): number {
  if (numerator === 0) return 0;
  const numeratorShift = 23 - firstLeadingBit(numerator);
  const denominatorShift = 23 - firstLeadingBit(denominator);
  const dividend = (numerator << numeratorShift) >>> 0;
  const divisor = (denominator << denominatorShift) >>> 0;
  const belowOne = dividend < divisor;
  const exponent = denominatorShift - numeratorShift - (belowOne ? 1 : 0);
  let mantissa = belowOne ? 0 : 1;
  let remainder = belowOne ? dividend : dividend - divisor;
  for (const chunkBits of [8, 8, belowOne ? 8 : 7]) {
    remainder = (remainder << chunkBits) >>> 0;
    mantissa = ((mantissa << chunkBits) | Math.floor(remainder / divisor)) >>> 0;
    remainder %= divisor;
  }
  if (remainder * 2 > divisor || (remainder * 2 === divisor && (mantissa & 1) !== 0)) mantissa++;
  return bitsFloat(((exponent + 127) << 23) + mantissa - 0x800000);
}

/** display_divide, replayed: significands through roundedQuotient, then the exact power of two. */
function displayDivide(numerator: number, denominator: number): number {
  const numeratorBits = floatBits(numerator);
  const denominatorBits = floatBits(denominator);
  const numeratorField = (numeratorBits >>> 23) & 0xff;
  const denominatorField = (denominatorBits >>> 23) & 0xff;
  const numeratorSignificand = (numeratorBits & 0x7fffff) | (numeratorField !== 0 ? 0x800000 : 0);
  const denominatorSignificand = (denominatorBits & 0x7fffff) | (denominatorField !== 0 ? 0x800000 : 0);
  const exponent = Math.max(numeratorField, 1) - Math.max(denominatorField, 1);
  return f32(roundedQuotient(numeratorSignificand, denominatorSignificand) * 2 ** exponent);
}

/** display_normalize for a finite value, replayed (a window whose span overflows is halved first). */
function gpuNormalize(value: number, low: number, high: number): number {
  if (!(high > low)) return 0.5;
  const scale = Number.isFinite(f32(high - low)) ? 1 : 0.5;
  const span = f32(f32(high * scale) - f32(low * scale));
  const offset = f32(f32(value * scale) - f32(low * scale));
  if (!(offset > 0)) return 0;
  if (offset >= span) return 1;
  return displayDivide(offset, span);
}

const level = (t: number) => Math.min(Math.trunc(f32(t * 255)), 255);

describe("colormap level edges", () => {
  it("the WGSL normalization divides only through display_divide", () => {
    const shader = /const DISPLAY_NORMALIZE_WGSL = [^`]*`([^`]*)`/.exec(COLORMAPS_SOURCE)![1];
    const normalize = shader.slice(shader.indexOf("fn display_normalize"));
    expect(normalize).toContain("display_divide(offset, span)");
    expect(normalize.replace(/\/\/.*$/gm, "")).not.toMatch(/[^/]\/[^/]/);
  });

  it("display_normalize divides once, since FXC fails to build a shader that inlines more divisions", () => {
    const shader = /const DISPLAY_NORMALIZE_WGSL = [^`]*`([^`]*)`/.exec(COLORMAPS_SOURCE)![1];
    const normalize = shader.slice(shader.indexOf("fn display_normalize"));
    expect(normalize.match(/display_divide\(/g)).toHaveLength(1);
  });

  it("the integer quotient is correctly rounded for every count pair the shaders pass", () => {
    let seed = 777;
    const random = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    for (let trial = 0; trial < 200000; trial++) {
      const numerator = 1 + Math.floor(random() * 16777215);
      const denominator = 1 + Math.floor(random() * (trial % 2 ? 65535 : 16777215));
      expect(roundedQuotient(numerator, denominator)).toBe(f32(numerator / denominator));
    }
    for (const [numerator, denominator] of [[1, 1], [16777215, 1], [1, 16777215], [16777215, 16777214], [3, 2], [5, 10], [8388609, 8388608]]) {
      expect(roundedQuotient(numerator, denominator)).toBe(f32(numerator / denominator));
    }
  });

  it("an inexact quotient one ULP low lands a value on an edge in the neighbouring level", () => {
    const t = displayNormalize(2, 0, 6);
    expect(level(t)).toBe(85);
    expect(level(bitsFloat(floatBits(t) - 1))).toBe(84);
  });

  it("the WGSL division is the correctly rounded float32 quotient", () => {
    let seed = 12345;
    const random = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    for (let trial = 0; trial < 20000; trial++) {
      const denominator = f32(2 ** (random() * 60 - 30) * (1 + random()));
      const numerator = f32(denominator * random());
      if (!(numerator > 0)) continue;
      expect(displayDivide(numerator, denominator)).toBe(f32(numerator / denominator));
    }
    for (const [numerator, denominator] of [[1, 3], [2, 6], [1, 1.5], [7, 255], [1e-45, 1], [3e-39, 4e-39], [1, 16777215]]) {
      expect(displayDivide(f32(numerator), f32(denominator))).toBe(f32(f32(numerator) / f32(denominator)));
    }
  });

  it("every level edge maps to the same LUT entry on both paths", () => {
    const windows = [[0, 6], [0, 1], [-1, 1], [0, 255], [3, 7], [1e-3, 2e-3], [-5000, 65535], [0.1, 0.7],
      [-3e38, 3e38], [-3.4e38, 2e38], [-1e38, 3.3e38]];
    for (const [vmin, vmax] of windows) {
      const low = f32(vmin), high = f32(vmax);
      for (let edge = 0; edge <= 255; edge++) {
        // The float32 values nearest the exact edge, and one ULP either side.
        const exact = f32(low + (edge * (high - low)) / 255);
        for (const value of [bitsFloat(floatBits(exact) - 1), exact, bitsFloat(floatBits(exact) + 1)]) {
          expect(level(gpuNormalize(value, low, high))).toBe(level(displayNormalize(value, low, high)));
        }
      }
    }
  });
});

describe("volume slice area average", () => {
  /** Source columns the shader's zoomed block covers for output column `column` (no pan). */
  function blockWidth(column: number, canvasWidth: number, sliceWidth: number, zoom: number): number {
    const z = f32(Math.max(zoom, VOLUME_SLICE_MIN_ZOOM));
    const center = canvasWidth * 0.5;
    const edge = (x: number) => f32(f32(f32(f32(x - center) / z) + center) * sliceWidth) / canvasWidth;
    const low = Math.min(edge(column), edge(column + 1));
    const high = Math.max(edge(column), edge(column + 1));
    const first = Math.min(Math.max(Math.floor(low), 0), sliceWidth - 1);
    return Math.max(first + 1, Math.min(Math.max(Math.ceil(high), 1), sliceWidth)) - first;
  }

  it("clamps the zoom at the widget's minimum, so a zoom-out never averages more than that view", () => {
    const shader = COLORMAPS_SOURCE.slice(COLORMAPS_SOURCE.indexOf("function volumeSliceShader"));
    expect(shader).toContain("let z = max(p.zoom, ${VOLUME_SLICE_MIN_ZOOM});");
    expect(shader).toContain("for (var xx = x0; xx < x1; xx++)");
    for (const column of [0, 37, 199, 300]) {
      const atMinimum = blockWidth(column, 400, 4096, VOLUME_SLICE_MIN_ZOOM);
      expect(blockWidth(column, 400, 4096, 1e-6)).toBe(atMinimum);
      expect(atMinimum).toBeLessThanOrEqual(Math.ceil(4096 / 400 / VOLUME_SLICE_MIN_ZOOM) + 1);
      expect(blockWidth(column, 400, 4096, 1)).toBeLessThanOrEqual(Math.ceil(4096 / 400) + 1);
    }
  });
});
