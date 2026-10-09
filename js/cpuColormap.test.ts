// The Canvas2D colormap engine that Show2D runs without WebGPU. Show2D paints
// its image panels and its FFT panels from separate animation frames, in
// different colormaps; each paint must come out in the colormap it names,
// whichever paint ran or started in between.
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { applyColormap, COLORMAPS, displayLog } from "./display/colormaps";
import { colormapSampledPixels, createColormapEngine } from "./display/cpuColormap";

class TestImageData {
  constructor(readonly data: Uint8ClampedArray, readonly width: number, readonly height: number) {}
}

const SIZE = 4;
const RAMP = Float32Array.from({ length: SIZE * SIZE }, (_, index) => index);
const REVERSED_RAMP = Float32Array.from(RAMP).reverse();
const RANGE = { vmin: 0, vmax: SIZE * SIZE - 1 };

function expectedPixels(data: Float32Array, lut: Uint8Array): number[] {
  const rgba = new Uint8ClampedArray(data.length * 4);
  applyColormap(data, rgba, lut, RANGE.vmin, RANGE.vmax);
  return Array.from(rgba);
}

function pixels(bitmap: ImageBitmap | null | undefined): number[] {
  return Array.from((bitmap as unknown as TestImageData).data);
}

beforeEach(() => {
  // jsdom has no ImageData or createImageBitmap: the bitmap is a copy of the
  // ImageData, taken when createImageBitmap is called, as in a browser.
  vi.stubGlobal("ImageData", TestImageData);
  vi.stubGlobal("createImageBitmap", (image: TestImageData) => Promise.resolve(new TestImageData(Uint8ClampedArray.from(image.data), image.width, image.height)));
});
afterEach(() => vi.unstubAllGlobals());

it("paints the image panels in their colormap while the FFT panels paint in another", async () => {
  const engine = await createColormapEngine();
  expect(engine.path).toBe("CPU");
  engine.uploadData(0, RAMP, SIZE, SIZE);
  engine.uploadData(1, REVERSED_RAMP, SIZE, SIZE);
  engine.uploadData(2, RAMP, SIZE, SIZE);
  const images = engine.renderSlotsToImageBitmapAsync([0, 1], [RANGE, RANGE], false, "viridis", COLORMAPS.viridis);
  const fft = engine.renderSlotsToImageBitmapAsync([2], [RANGE], false, "inferno", COLORMAPS.inferno);
  const imagesAgain = engine.renderSlotsToImageBitmapAsync([0], [RANGE], false, "viridis", COLORMAPS.viridis);
  const [imageBitmaps, fftBitmaps, againBitmaps] = await Promise.all([images, fft, imagesAgain]);
  expect(pixels(imageBitmaps?.[0])).toEqual(expectedPixels(RAMP, COLORMAPS.viridis));
  expect(pixels(imageBitmaps?.[1])).toEqual(expectedPixels(REVERSED_RAMP, COLORMAPS.viridis));
  expect(pixels(fftBitmaps?.[0])).toEqual(expectedPixels(RAMP, COLORMAPS.inferno));
  expect(pixels(againBitmaps?.[0])).toEqual(expectedPixels(RAMP, COLORMAPS.viridis));
});

it("paints a scaled panel in the colormap that paint names", async () => {
  const engine = await createColormapEngine();
  engine.uploadData(0, RAMP, SIZE, SIZE);
  await engine.renderSlotsToImageBitmapAsync([0], [RANGE], false, "inferno", COLORMAPS.inferno);
  const scaled = await engine.renderSlotScaledToImageBitmapAsync(0, RANGE, false, SIZE, SIZE, "gray", COLORMAPS.gray);
  expect(pixels(scaled)).toEqual(expectedPixels(RAMP, COLORMAPS.gray));
});

// The per-pixel loop inlines displayNormalize for the common window; NaN,
// infinities, log scale, an empty window and a window whose float32 span
// overflows must still pick the reference LUT entry (applyColormap).
const SPECIAL = Float32Array.from([0, 1, -3.5, 2.25, NaN, Infinity, -Infinity, 1e-7, 3.4e38, -3.4e38, 7, 7.000001, 15, -0, 0.5, 9]);

it.each([
  ["plain window", { vmin: -4, vmax: 15 }, false],
  ["log scale", { vmin: 0, vmax: 3 }, true],
  ["empty window", { vmin: 7, vmax: 7 }, false],
  ["overflowing span", { vmin: -3.4e38, vmax: 3.4e38 }, false],
])("colors every value like the reference colormap: %s", async (_name, range, logScale) => {
  const engine = await createColormapEngine();
  engine.uploadData(0, SPECIAL, SIZE, SIZE);
  const [bitmap] = (await engine.renderSlotsToImageBitmapAsync([0], [range], logScale, "viridis", COLORMAPS.viridis))!;
  const values = logScale ? SPECIAL.map(displayLog) : SPECIAL;
  const reference = new Uint8ClampedArray(SPECIAL.length * 4);
  applyColormap(values, reference, COLORMAPS.viridis, range.vmin, range.vmax);
  expect(pixels(bitmap)).toEqual(Array.from(reference));
});

it("samples a scaled render from the nearest source pixel of each output pixel", async () => {
  const engine = await createColormapEngine();
  engine.uploadData(0, RAMP, SIZE, SIZE);
  const scaled = await engine.renderSlotScaledToImageBitmapAsync(0, RANGE, false, 3, 7, "gray", COLORMAPS.gray);
  const f32 = Math.fround;
  const source = (out: number, size: number, outSize: number) => Math.min(Math.trunc(f32(f32(f32(out + 0.5) * size) / outSize)), size - 1);
  const expected = Float32Array.from({ length: 3 * 7 }, (_, k) => RAMP[source(Math.floor(k / 3), SIZE, 7) * SIZE + source(k % 3, SIZE, 3)]);
  expect(pixels(scaled)).toEqual(expectedPixels(expected, COLORMAPS.gray));
});

it("paints only the sampled source pixels of a canvas, and leaves uncovered pixels transparent", () => {
  const sourceIndex = Int32Array.from([5, -1, 0, 15, 15, -1]);
  const target = new TestImageData(new Uint8ClampedArray(sourceIndex.length * 4).fill(9), 3, 2) as unknown as ImageData;
  colormapSampledPixels(REVERSED_RAMP, sourceIndex, RANGE, false, COLORMAPS.inferno, target);
  const full = expectedPixels(REVERSED_RAMP, COLORMAPS.inferno);
  const expected = Array.from(sourceIndex).flatMap(index => index < 0 ? [0, 0, 0, 0] : full.slice(4 * index, 4 * index + 4));
  expect(Array.from(target.data)).toEqual(expected);
});
