const COLORMAP_POINTS: Record<string, number[][]> = {
  inferno: [
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [252, 193, 57], [252, 255, 164],
  ],
  viridis: [
    [68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141],
    [42, 120, 142], [33, 145, 140], [34, 168, 132], [68, 191, 112],
    [122, 209, 81], [189, 223, 38], [253, 231, 37],
  ],
  plasma: [
    [13, 8, 135], [75, 3, 161], [126, 3, 168], [168, 34, 150],
    [203, 70, 121], [229, 107, 93], [248, 148, 65], [253, 195, 40], [240, 249, 33],
  ],
  magma: [
    [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
    [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191],
  ],
  hot: [
    [0, 0, 0], [87, 0, 0], [173, 0, 0], [255, 0, 0],
    [255, 87, 0], [255, 173, 0], [255, 255, 0], [255, 255, 128], [255, 255, 255],
  ],
  gray: [[0, 0, 0], [255, 255, 255]],
  hsv: [
    [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255],
    [0, 0, 255], [255, 0, 255], [255, 0, 0],
  ],
  turbo: [
    [48, 18, 59], [69, 55, 161], [66, 107, 230], [30, 162, 230],
    [29, 212, 169], [79, 241, 89], [175, 240, 32], [244, 195, 12],
    [248, 118, 11], [207, 46, 3], [122, 4, 2],
  ],
  RdBu: [
    [103, 0, 31], [178, 24, 43], [214, 96, 77], [244, 165, 130],
    [253, 219, 199], [247, 247, 247], [209, 229, 240], [146, 197, 222],
    [67, 147, 195], [33, 102, 172], [5, 48, 97],
  ],
};

export const COLORMAP_NAMES = Object.keys(COLORMAP_POINTS);

function createColormapLUT(points: number[][]): Uint8Array {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (points.length - 1);
    const idx = Math.floor(t);
    const frac = t - idx;
    const p0 = points[Math.min(idx, points.length - 1)];
    const p1 = points[Math.min(idx + 1, points.length - 1)];
    lut[i * 3] = Math.round(p0[0] + frac * (p1[0] - p0[0]));
    lut[i * 3 + 1] = Math.round(p0[1] + frac * (p1[1] - p0[1]));
    lut[i * 3 + 2] = Math.round(p0[2] + frac * (p1[2] - p0[2]));
  }
  return lut;
}

export const COLORMAPS: Record<string, Uint8Array> = Object.fromEntries(
  Object.entries(COLORMAP_POINTS).map(([name, points]) => [name, createColormapLUT(points)])
);

/** Apply colormap LUT to float data, writing into an RGBA Uint8ClampedArray. */
export function applyColormap(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
): void {
  const range = vmax > vmin ? vmax - vmin : 1;
  const uniformData = !(vmax > vmin);
  for (let i = 0; i < data.length; i++) {
    const clipped = Math.max(vmin, Math.min(vmax, data[i]));
    const v = uniformData ? 128 : Math.min(255, Math.floor(((clipped - vmin) / range) * 255));
    const j = i * 4;
    const lutIdx = v * 3;
    rgba[j] = lut[lutIdx];
    rgba[j + 1] = lut[lutIdx + 1];
    rgba[j + 2] = lut[lutIdx + 2];
    rgba[j + 3] = 255;
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
