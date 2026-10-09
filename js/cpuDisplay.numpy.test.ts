// @ts-nocheck
// Parity: the Canvas2D/JS display path (the one a browser without WebGPU runs)
// against a numpy reference on the same input. The reference repeats the WGSL
// display arithmetic in float32, so equality here is what makes the CPU and
// WebGPU canvases agree pixel for pixel.
import { execFileSync } from "node:child_process";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { applyColormap, COLORMAPS } from "./display/colormaps";
import { histogramBins } from "./display/cpuColormap";
import { computeMagnitude, CPUFFT, fftshift } from "./display/fft";
import { percentileClip } from "./display/stats";

const WIDTH = 48;
const HEIGHT = 40;
// Smooth structure plus fine texture, with exact window edges and non-finite
// samples so the normalize branches and LUT end points are all exercised.
const IMAGE = Float32Array.from({ length: WIDTH * HEIGHT }, (_, index) => {
  const row = Math.floor(index / WIDTH);
  const column = index % WIDTH;
  return 120 * Math.exp(-((row - 18) ** 2 + (column - 26) ** 2) / 90) + 7 * Math.sin(column * 1.7) * Math.cos(row * 0.9) + 10;
});
IMAGE[0] = Number.NaN;
IMAGE[1] = Number.POSITIVE_INFINITY;
IMAGE[2] = Number.NEGATIVE_INFINITY;
IMAGE[3] = 20;
IMAGE[4] = 100;
const LOW = 20;
const HIGH = 100;

function numpyReference() {
  const python = process.env.PYTHON || "python";
  const code = String.raw`
import json
import sys

import numpy as np

payload = json.load(sys.stdin)
width, height = payload["width"], payload["height"]
image = np.array([np.nan if v is None else v for v in payload["image"]], dtype=np.float32)
image[1], image[2] = np.inf, -np.inf
low, high = np.float32(payload["low"]), np.float32(payload["high"])
points = np.array(json.load(open(payload["colormaps"]))["inferno"], dtype=np.float64)

# LUT: linear interpolation between control points, round half up.
t = np.arange(256) / 255 * (len(points) - 1)
base = np.floor(t).astype(int)
frac = (t - base)[:, None]
p0 = points[np.minimum(base, len(points) - 1)]
p1 = points[np.minimum(base + 1, len(points) - 1)]
lut = np.floor(p0 + frac * (p1 - p0) + 0.5).astype(np.uint8)


def normalize(values):
    with np.errstate(invalid="ignore"):
        out = np.clip((values - low) / np.float32(high - low), np.float32(0), np.float32(1))
    out[np.isnan(values) | (values == -np.inf)] = 0
    out[values == np.inf] = 1
    return out.astype(np.float32)


entries = np.minimum((normalize(image) * np.float32(255)).astype(np.float32).astype(np.int64), 255)
rgb = lut[entries]

finite = image[np.isfinite(image)]
bins = np.minimum((normalize(finite) * np.float32(256)).astype(np.float32).astype(np.int64), 255)
counts = np.bincount(bins, minlength=256).astype(np.float64)

padded_w, padded_h = 1 << (width - 1).bit_length(), 1 << (height - 1).bit_length()
plane = np.zeros((padded_h, padded_w))
clean = np.where(np.isfinite(image), image, 0).reshape(height, width)
plane[:height, :width] = clean
magnitude = np.abs(np.fft.fftshift(np.fft.fft2(plane)))

print(json.dumps({
    "rgb": rgb.ravel().tolist(),
    "bins": (counts / counts.max()).tolist(),
    "magnitude": magnitude.ravel().tolist(),
}))
`;
  const input = JSON.stringify({
    width: WIDTH, height: HEIGHT, low: LOW, high: HIGH,
    image: Array.from(IMAGE, value => Number.isFinite(value) ? value : null),
    colormaps: path.resolve(__dirname, "..", "src", "quantem", "widget", "colormaps.json"),
  });
  return JSON.parse(execFileSync(python, ["-c", code], { input, encoding: "utf8" }));
}

describe("Canvas2D display path matches numpy", () => {
  const reference = numpyReference();

  it("applies the colormap LUT with the float32 window and floor indexing", () => {
    const rgba = new Uint8ClampedArray(IMAGE.length * 4);
    applyColormap(IMAGE, rgba, COLORMAPS.inferno, LOW, HIGH);
    const rgb = Array.from(rgba).filter((_, index) => index % 4 !== 3);
    expect(rgb).toEqual(reference.rgb);
  });

  it("bins the histogram exactly as the float32 WGSL histogram pass", () => {
    expect(histogramBins(IMAGE, LOW, HIGH)).toEqual(reference.bins);
  });

  it("places the percentile contrast window within one histogram bin of the exact quantile", () => {
    // Dense samples of 100 * u^2 for uniform u: the q quantile is 100 * q^2.
    const count = 1_000_000;
    const values = Float32Array.from({ length: count }, (_, index) => 100 * (index / (count - 1)) ** 2);
    const clip = percentileClip(values, 2, 98);
    const step = 100 / 1024;
    expect(Math.abs(clip.vmin - 100 * 0.02 ** 2)).toBeLessThan(step);
    expect(Math.abs(clip.vmax - 100 * 0.98 ** 2)).toBeLessThan(step / 100);
  });

  it("computes the zero-padded shifted FFT magnitude to float32 precision", async () => {
    const paddedWidth = 64;
    const paddedHeight = 64;
    const real = new Float32Array(paddedWidth * paddedHeight);
    for (let row = 0; row < HEIGHT; row++) {
      for (let column = 0; column < WIDTH; column++) {
        const value = IMAGE[row * WIDTH + column];
        real[row * paddedWidth + column] = Number.isFinite(value) ? value : 0;
      }
    }
    const result = await new CPUFFT().fft2D(real, new Float32Array(real.length), paddedWidth, paddedHeight);
    fftshift(result.real, paddedWidth, paddedHeight);
    fftshift(result.imag, paddedWidth, paddedHeight);
    const magnitude = computeMagnitude(result.real, result.imag);
    const peak = Math.max(...reference.magnitude);
    let worst = 0;
    magnitude.forEach((value, index) => { worst = Math.max(worst, Math.abs(value - reference.magnitude[index])); });
    expect(worst / peak).toBeLessThan(1e-6);
  });
});
