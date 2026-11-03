// The display colormaps: control points shared with Python, expanded to the
// 256-entry RGB lookup tables the CPU and WebGPU paths paint with.
// One table for Python and the browser: the package file Plot2D and static renders read.
import colormapPoints from "../../src/quantem/widget/colormaps.json";

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
