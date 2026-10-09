// Show2D mounted in jsdom, where there is no WebGPU, so its panels paint on the
// Canvas2D colormap engine. The user story: two images in gray, pick viridis,
// turn FFT on, then replace the images from Python. The image panels paint from
// one animation frame and the FFT panels (inferno) from another; a browser runs
// microtasks after every frame callback, so the FFT frame can run in between the
// image panels' LUT choice and their paint. The image panels must stay viridis.
// The live contrast preview of a histogram drag must paint what the release
// paints: each panel in its own colormap, and packed uint8 frames through the
// engine's uint8-aware scaled pass at the panel's canvas size.
import { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { COLORMAPS } from "../display/colormaps";

type Paint = { slot: number; pixels: Uint8ClampedArray; width: number; height: number; pass: "batched" | "scaled" };
const paints = vi.hoisted(() => [] as Paint[]);
const engineCalls = vi.hoisted(() => [] as string[]);
vi.mock("../display/cpuColormap", async (importOriginal) => {
  const original = await importOriginal<typeof import("../display/cpuColormap")>();
  return {
    ...original,
    createColormapEngine: async () => {
      const engine = await original.createColormapEngine();
      for (const name of ["uploadData", "computeRangeBatch", "computeHistogramBatch", "computeHistogramWithRange"] as const) {
        const call = (engine[name] as (...args: unknown[]) => unknown).bind(engine);
        (engine as unknown as Record<string, unknown>)[name] = (...args: unknown[]) => {
          engineCalls.push(name === "uploadData" ? `uploadData ${args[0]}` : name);
          return call(...args);
        };
      }
      const record = (slot: number, bitmap: unknown, pass: Paint["pass"]) => {
        const image = bitmap as TestImageData;
        paints.push({ slot, pixels: image.data, width: image.width, height: image.height, pass });
      };
      const render = engine.renderSlotsToImageBitmapAsync.bind(engine);
      engine.renderSlotsToImageBitmapAsync = async (indices, ...rest) => {
        const bitmaps = await render(indices, ...rest);
        bitmaps?.forEach((bitmap, k) => { if (bitmap) record(indices[k], bitmap, "batched"); });
        return bitmaps;
      };
      const renderScaled = engine.renderSlotScaledToImageBitmapAsync.bind(engine);
      engine.renderSlotScaledToImageBitmapAsync = async (slot, ...rest) => {
        const bitmap = await renderScaled(slot, ...rest);
        if (bitmap) record(slot, bitmap, "scaled");
        return bitmap;
      };
      return engine;
    },
  };
});

const { render: renderShow2D } = await import("./index");

class TestImageData {
  constructor(readonly data: Uint8ClampedArray, readonly width: number, readonly height: number) {}
}

const SIZE = 16;
const BF = Float32Array.from({ length: SIZE * SIZE }, (_, i) => 100 + (i % SIZE) * 3 + Math.floor(i / SIZE));
const ADF = Float32Array.from({ length: SIZE * SIZE }, (_, i) => 500 - (i % SIZE) * 7 + 2 * Math.floor(i / SIZE));

function frameBytes(...images: Float32Array[]): DataView {
  const out = new Float32Array(images.length * SIZE * SIZE);
  images.forEach((image, k) => out.set(image, k * SIZE * SIZE));
  return new DataView(out.buffer);
}

/** The colormaps whose LUT holds every pixel of a painted bitmap. */
function colormapsOf(pixels: Uint8ClampedArray): string[] {
  return ["gray", "viridis", "inferno"].filter(name => {
    const lut = COLORMAPS[name];
    const entries = new Set<string>();
    for (let i = 0; i < 256; i++) entries.add(`${lut[3 * i]},${lut[3 * i + 1]},${lut[3 * i + 2]}`);
    for (let i = 0; i < pixels.length; i += 4) {
      if (!entries.has(`${pixels[i]},${pixels[i + 1]},${pixels[i + 2]}`)) return false;
    }
    return true;
  });
}

class Model {
  values: Record<string, unknown> = {
    n_images: 2, width: SIZE, height: SIZE, frame_bytes: frameBytes(BF, ADF), cmap: "gray", labels: ["BF", "ADF"],
    is_rgb: [false, false], ncols: 2, show_fft: false, show_controls: true, show_stats: true, link_contrast: true,
    link_zoom: true, link_pan: true, panel_cmaps: [], hidden_panels: [], panel_order: [], log_scale: false,
    auto_contrast: false, image_rotations: [0, 0], image_flips_horizontal: [false, false],
    image_flips_vertical: [false, false], panel_frame_counts: [1, 1], panel_frame_indices: [0, 0],
    panel_stack_offsets: [-1, -1], selected_idx: 0, stats_mean: [0, 0], stats_min: [0, 0], stats_max: [0, 0],
    stats_std: [0, 0], fft_window: true, fft_metrics: true, pixel_size: 0, pixel_unit: "pixels",
    scale_bar_visible: true, show_panel_titles: true, title: "", n_pages: 1, page_idx: 0,
  };
  listeners = new Map<string, Set<() => void>>();
  get(key: string) { return this.values[key]; }
  set(key: string, value: unknown) {
    this.values[key] = value;
    this.listeners.get(`change:${key}`)?.forEach(fn => fn());
  }
  save_changes = vi.fn();
  send = vi.fn();
  on(key: string, fn: () => void) {
    if (!this.listeners.has(key)) this.listeners.set(key, new Set());
    this.listeners.get(key)!.add(fn);
  }
  off(key: string, fn: () => void) { this.listeners.get(key)?.delete(fn); }
}

let frames: Map<number, FrameRequestCallback>;
let nextFrame: number;
let el: HTMLDivElement;
let cleanup: (() => void) | undefined;

async function settle() { await act(async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); }); }

/** One browser frame: each callback runs, then its microtasks, before the next callback. */
async function paintFrame() {
  const pending = [...frames.values()];
  frames.clear();
  for (const callback of pending) {
    act(() => callback(0));
    await settle();
  }
}

beforeEach(() => {
  paints.length = 0;
  engineCalls.length = 0;
  frames = new Map();
  nextFrame = 0;
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("ImageData", TestImageData);
  // The bitmap is a copy taken when createImageBitmap is called, as in a browser.
  vi.stubGlobal("createImageBitmap", (image: TestImageData) => Promise.resolve(Object.assign(new TestImageData(Uint8ClampedArray.from(image.data), image.width, image.height), { close() {} })));
  vi.stubGlobal("requestAnimationFrame", (fn: FrameRequestCallback) => { frames.set(++nextFrame, fn); return nextFrame; });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => frames.delete(id));
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(new Proxy({}, {
    get: (target: Record<string, unknown>, prop: string) => prop in target ? target[prop]
      : prop === "measureText" ? (text: string) => ({ width: text.length * 6 })
        : prop === "getImageData" ? (_x: number, _y: number, w: number, h: number) => new TestImageData(new Uint8ClampedArray(w * h * 4), w, h)
          : prop === "createImageData" ? (w: number, h: number) => new TestImageData(new Uint8ClampedArray(w * h * 4), w, h)
            : () => undefined,
    set: (target: Record<string, unknown>, prop: string, value: unknown) => { target[prop] = value; return true; },
  }) as never);
  el = document.createElement("div");
  document.body.append(el);
});
afterEach(async () => {
  act(() => cleanup?.());
  cleanup = undefined;
  await settle();
  el.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("keeps the image panels in their colormap while FFT panels repaint in another", async () => {
  const model = new Model();
  act(() => { cleanup = renderShow2D({ model, el } as never) as (() => void) | undefined; });
  await settle();
  for (const [key, value] of [["cmap", "viridis"], ["show_fft", true]] as const) {
    act(() => model.set(key, value));
    await settle();
    for (let i = 0; i < 4; i++) await paintFrame();
  }
  act(() => model.set("frame_bytes", frameBytes(ADF, BF)));
  await settle();
  for (let i = 0; i < 6; i++) {
    await paintFrame();
    // A later display update (an async range or contrast result in the browser)
    // re-runs the image paint while the FFT colour frame is already queued.
    act(() => model.set("is_rgb", [false, false]));
    await settle();
  }
  for (let i = 0; i < 4; i++) await paintFrame();
  const last = (slot: number) => paints.filter(paint => paint.slot === slot).slice(-1)[0];
  expect(colormapsOf(last(0).pixels)).toEqual(["viridis"]);
  expect(colormapsOf(last(1).pixels)).toEqual(["viridis"]);
  expect(colormapsOf(last(2).pixels)).toEqual(["inferno"]);
  expect(colormapsOf(last(3).pixels)).toEqual(["inferno"]);
  expect(paints.filter(paint => paint.slot < 2 && colormapsOf(paint.pixels).includes("inferno"))).toEqual([]);
});

/** Move the max handle of the image histogram the way MUI's hidden range input reports a drag. */
function dragHistogramMax(value: number) {
  const inputs = el.querySelectorAll<HTMLInputElement>('input[aria-label="Histogram intensity clip range"]');
  const input = inputs[inputs.length - 1];
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, String(value));
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

it("previews a histogram drag on each panel in its own colormap", async () => {
  const model = new Model();
  model.values.panel_cmaps = ["gray", "viridis"];
  act(() => { cleanup = renderShow2D({ model, el } as never) as (() => void) | undefined; });
  await settle();
  for (let i = 0; i < 4; i++) await paintFrame();
  expect(colormapsOf(paints.filter(paint => paint.slot === 1).slice(-1)[0].pixels)).toEqual(["viridis"]);
  paints.length = 0;
  dragHistogramMax(60);
  await settle();
  for (let i = 0; i < 4; i++) await paintFrame();
  const dragged = (slot: number) => paints.filter(paint => paint.slot === slot);
  expect(dragged(0).length).toBeGreaterThan(0);
  expect(dragged(1).length).toBeGreaterThan(0);
  for (const paint of dragged(0)) expect(colormapsOf(paint.pixels)).toEqual(["gray"]);
  for (const paint of dragged(1)) expect(colormapsOf(paint.pixels)).toEqual(["viridis"]);
});

it("previews a histogram drag of a packed uint8 frame through the scaled pass at canvas size", async () => {
  const model = new Model();
  const frame = Uint8Array.from({ length: SIZE * SIZE }, (_, i) => Math.round((255 * i) / (SIZE * SIZE - 1)));
  Object.assign(model.values, {
    n_images: 1, labels: ["uint8"], is_rgb: [false], frame_bytes: new DataView(frame.buffer), offline: true,
    _offline_min: 0, _offline_max: 255, _offline_mins: [0], _offline_maxs: [255], image_rotations: [0],
    image_flips_horizontal: [false], image_flips_vertical: [false], panel_frame_counts: [1], panel_frame_indices: [0],
    panel_stack_offsets: [-1], stats_mean: [0], stats_min: [0], stats_max: [0], stats_std: [0], ncols: 1,
  });
  act(() => { cleanup = renderShow2D({ model, el } as never) as (() => void) | undefined; });
  await settle();
  for (let i = 0; i < 4; i++) await paintFrame();
  const settled = paints.filter(paint => paint.slot === 0).slice(-1)[0];
  expect(settled.pass).toBe("scaled");
  paints.length = 0;
  dragHistogramMax(60);
  await settle();
  for (let i = 0; i < 4; i++) await paintFrame();
  expect(paints.length).toBeGreaterThan(0);
  for (const paint of paints) {
    // The batched pass reads every slot as float32: on WebGPU a packed uint8 slot comes out as noise.
    expect(paint.pass).toBe("scaled");
    expect([paint.width, paint.height]).toEqual([settled.width, settled.height]);
  }
});

it.each([false, true])("uploads each frame once and runs each display pass once on first render (auto contrast %s)", async (autoContrast) => {
  const model = new Model();
  model.values.auto_contrast = autoContrast;
  act(() => { cleanup = renderShow2D({ model, el } as never) as (() => void) | undefined; });
  await settle();
  for (let i = 0; i < 6; i++) await paintFrame();
  const count = (name: string) => engineCalls.filter(call => call === name).length;
  expect(engineCalls.filter(call => call.startsWith("uploadData")).sort()).toEqual(["uploadData 0", "uploadData 1"]);
  expect(count("computeRangeBatch")).toBe(1);
  // the image histogram, plus Auto's percentile histograms when it is on
  expect(count("computeHistogramBatch") + count("computeHistogramWithRange")).toBe(autoContrast ? 2 : 1);
  expect(paints.map(paint => paint.slot).sort()).toEqual([0, 1]);
  // a new frame from Python repeats each pass once more
  engineCalls.length = 0;
  paints.length = 0;
  act(() => model.set("frame_bytes", frameBytes(ADF, BF)));
  await settle();
  for (let i = 0; i < 6; i++) await paintFrame();
  expect(engineCalls.filter(call => call.startsWith("uploadData")).sort()).toEqual(["uploadData 0", "uploadData 1"]);
  expect(count("computeRangeBatch")).toBe(1);
  expect(paints.map(paint => paint.slot).sort()).toEqual([0, 1]);
});
