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
const rgbPaints = vi.hoisted(() => [] as { pixels: Uint8ClampedArray; bitmap: unknown }[]);
const gates = vi.hoisted(() => ({ range: null as Promise<void> | null, bitmap: null as Promise<void> | null }));
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
          if (name === "computeRangeBatch" && gates.range) return gates.range.then(() => call(...args));
          return call(...args);
        };
      }
      const record = (slot: number, bitmap: unknown, pass: Paint["pass"]) => {
        const image = bitmap as TestImageData;
        paints.push({ slot, pixels: image.data, width: image.width, height: image.height, pass });
      };
      const render = engine.renderSlotsToImageBitmapAsync.bind(engine);
      engine.renderSlotsToImageBitmapAsync = async (indices, ...rest) => {
        if (gates.bitmap) await gates.bitmap;
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
      const renderRgb = engine.renderRgbToImageBitmapAsync.bind(engine);
      engine.renderRgbToImageBitmapAsync = async (...args) => {
        const bitmap = await renderRgb(...args);
        if (bitmap) rgbPaints.push({ pixels: (bitmap as unknown as TestImageData).data, bitmap });
        return bitmap;
      };
      return engine;
    },
  };
});

const { default: { render: renderShow2D } } = await import("./index");

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
    is_rgb: [false, false], columns: 2, show_fft: false, show_controls: true, show_stats: true, link_contrast: true,
    link_zoom: true, link_pan: true, panel_cmaps: [], hidden_panels: [], panel_order: [], log_scale: false,
    auto_contrast: false, image_rotations: [0, 0], image_flips_horizontal: [false, false],
    image_flips_vertical: [false, false], panel_frame_counts: [1, 1], panel_frame_indices: [0, 0],
    panel_stack_offsets: [-1, -1], selected_idx: 0, stats_mean: [0, 0], stats_min: [0, 0], stats_max: [0, 0],
    stats_std: [0, 0], fft_window: true, fft_metrics: true, pixel_size: 0, pixel_unit: "pixels",
    show_scale_bar: true, show_panel_titles: true, title: "", n_pages: 1, page_idx: 0,
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
let canvasDraws: { target: HTMLCanvasElement; source: unknown }[];

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
  rgbPaints.length = 0;
  engineCalls.length = 0;
  gates.range = gates.bitmap = null;
  canvasDraws = [];
  frames = new Map();
  nextFrame = 0;
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("ImageData", TestImageData);
  // The bitmap is a copy taken when createImageBitmap is called, as in a browser.
  vi.stubGlobal("createImageBitmap", (image: TestImageData) => Promise.resolve(Object.assign(new TestImageData(Uint8ClampedArray.from(image.data), image.width, image.height), { close() {} })));
  vi.stubGlobal("requestAnimationFrame", (fn: FrameRequestCallback) => { frames.set(++nextFrame, fn); return nextFrame; });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => frames.delete(id));
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function (this: HTMLCanvasElement) {
    const canvas = this;
    return new Proxy({}, {
    get: (target: Record<string, unknown>, prop: string) => prop in target ? target[prop]
      : prop === "drawImage" ? (source: unknown) => canvasDraws.push({ target: canvas, source })
      : prop === "measureText" ? (text: string) => ({ width: text.length * 6 })
        : prop === "getImageData" ? (_x: number, _y: number, w: number, h: number) => new TestImageData(new Uint8ClampedArray(w * h * 4), w, h)
          : prop === "createImageData" ? (w: number, h: number) => new TestImageData(new Uint8ClampedArray(w * h * 4), w, h)
            : () => undefined,
    set: (target: Record<string, unknown>, prop: string, value: unknown) => { target[prop] = value; return true; },
    }) as never;
  });
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
  // Without the control rows each jsdom render is about 5x cheaper; the panels
  // paint and the engine is called in exactly the same sequence.
  model.values.show_controls = false;
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
    _offline_mins: [0], _offline_maxs: [255], image_rotations: [0],
    image_flips_horizontal: [false], image_flips_vertical: [false], panel_frame_counts: [1], panel_frame_indices: [0],
    panel_stack_offsets: [-1], stats_mean: [0], stats_min: [0], stats_max: [0], stats_std: [0], columns: 1,
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

it("retains completed images and their labels while resized frames await range and bitmap work", async () => {
  const model = new Model();
  model.values.show_controls = false;
  model.values.auto_contrast = true;
  act(() => { cleanup = renderShow2D({ model, el } as never) as (() => void) | undefined; });
  await settle();
  for (let i = 0; i < 6; i++) await paintFrame();
  const panels = Array.from(el.querySelectorAll<HTMLCanvasElement>("canvas[data-show2d-main-canvas]"));
  expect(panels).toHaveLength(2);
  const displayedSource = (panel: HTMLCanvasElement) => canvasDraws.filter(draw => draw.target === panel).at(-1)!.source as HTMLCanvasElement;
  const completed = panels.map(displayedSource);
  expect(completed.map(canvas => [canvas.width, canvas.height])).toEqual([[SIZE, SIZE], [SIZE, SIZE]]);

  let finishRange!: () => void;
  let finishBitmap!: () => void;
  gates.range = new Promise(resolve => { finishRange = resolve; });
  gates.bitmap = new Promise(resolve => { finishBitmap = resolve; });
  const nextWidth = SIZE * 2;
  const next = Float32Array.from({ length: 2 * nextWidth * SIZE }, (_, i) => 1000 + i);
  act(() => {
    model.set("width", nextWidth);
    model.set("labels", ["new BF", "new ADF"]);
    model.set("frame_bytes", new DataView(next.buffer));
  });
  await settle();
  // Force the ordinary layout repaint while Auto's asynchronous range is pending.
  act(() => model.set("smooth", true));
  await settle();
  for (let i = 0; i < 3; i++) await paintFrame();
  expect(panels.map(displayedSource)).toEqual(completed);
  expect(el.textContent).not.toContain("new BF");
  expect(el.textContent).not.toContain("new ADF");
  expect(el.textContent).not.toContain("Mean");

  finishRange();
  gates.range = null;
  await settle();
  for (let i = 0; i < 3; i++) await paintFrame();
  expect(panels.map(displayedSource)).toEqual(completed);
  expect(el.textContent).not.toContain("new BF");

  finishBitmap();
  gates.bitmap = null;
  await settle();
  for (let i = 0; i < 6; i++) await paintFrame();
  const replacement = panels.map(displayedSource);
  expect(replacement.every((canvas, i) => canvas !== completed[i])).toBe(true);
  expect(replacement.map(canvas => [canvas.width, canvas.height])).toEqual([[nextWidth, SIZE], [nextWidth, SIZE]]);
  expect(el.textContent).toContain("new BF");
  expect(el.textContent).toContain("new ADF");
  expect(el.textContent).toContain("Mean");
  const last = (slot: number) => paints.filter(paint => paint.slot === slot).at(-1)!;
  expect([last(0).width, last(0).height, last(1).width, last(1).height]).toEqual([nextWidth, SIZE, nextWidth, SIZE]);
});

it("discards a superseded resized bitmap instead of publishing it over the latest frame", async () => {
  const model = new Model();
  model.values.show_controls = false;
  model.values.auto_contrast = true;
  act(() => { cleanup = renderShow2D({ model, el } as never) as (() => void) | undefined; });
  await settle();
  for (let i = 0; i < 6; i++) await paintFrame();
  const panels = Array.from(el.querySelectorAll<HTMLCanvasElement>("canvas[data-show2d-main-canvas]"));
  let finishBitmap!: () => void;
  gates.bitmap = new Promise(resolve => { finishBitmap = resolve; });
  const resize = async (width: number, label: string) => {
    const data = Float32Array.from({ length: 2 * width * SIZE }, (_, i) => i + width);
    act(() => {
      model.set("width", width);
      model.set("labels", [label, `${label} sum`]);
      model.set("frame_bytes", new DataView(data.buffer));
    });
    await settle();
    for (let i = 0; i < 3; i++) await paintFrame();
  };
  await resize(SIZE * 2, "superseded");
  await resize(SIZE * 3, "latest");
  canvasDraws.length = 0;
  finishBitmap();
  gates.bitmap = null;
  await settle();
  for (let i = 0; i < 6; i++) await paintFrame();
  const displayed = canvasDraws.filter(draw => panels.includes(draw.target));
  expect(displayed.length).toBeGreaterThan(0);
  expect(displayed.every(draw => (draw.source as HTMLCanvasElement).width === SIZE * 3)).toBe(true);
  expect(el.textContent).toContain("latest sum");
  expect(el.textContent).not.toContain("superseded");
});

it.each([[true, false], [true, true]])("commits resized true-color panels with RGB flags %j, %j", async (firstRgb, secondRgb) => {
  const model = new Model();
  const flags = [firstRgb, secondRgb];
  const pixels = (width: number) => new DataView(Float32Array.from(
    { length: flags.reduce((sum, rgb) => sum + (rgb ? 3 : 1), 0) * width * SIZE },
    (_, i) => (i % 100) / 100,
  ).buffer);
  Object.assign(model.values, { show_controls: false, is_rgb: flags, frame_bytes: pixels(SIZE) });
  act(() => { cleanup = renderShow2D({ model, el } as never) as (() => void) | undefined; });
  await settle();
  for (let i = 0; i < 6; i++) await paintFrame();
  const panels = Array.from(el.querySelectorAll<HTMLCanvasElement>("canvas[data-show2d-main-canvas]"));
  const displayedSource = (panel: HTMLCanvasElement) => canvasDraws.filter(draw => draw.target === panel).at(-1)!.source as HTMLCanvasElement;
  const previous = panels.map(displayedSource);
  let finishRange!: () => void;
  gates.range = new Promise(resolve => { finishRange = resolve; });
  act(() => {
    model.set("width", SIZE * 2);
    model.set("frame_bytes", pixels(SIZE * 2));
  });
  await settle();
  act(() => model.set("smooth", true));
  await settle();
  expect(panels.map(displayedSource)).toEqual(previous);
  finishRange();
  gates.range = null;
  await settle();
  for (let i = 0; i < 6; i++) await paintFrame();
  expect(panels.map(displayedSource).map(canvas => canvas.width)).toEqual([SIZE * 2, SIZE * 2]);
});

it("packs a color panel through the colormap engine's true-color pass, beside a gray panel", async () => {
  // The engine is WebGPU in a browser with an adapter and Canvas2D here; either way the color panel is
  // packed by the engine, so the panel is drawn by the path the title badge names.
  const model = new Model();
  const rgb = Float32Array.from({ length: 3 * SIZE * SIZE }, (_, i) => [0, 0.25, 0.5, 1, -0.5, 2, Number.NaN][i % 7]);
  const out = new Float32Array(3 * SIZE * SIZE + SIZE * SIZE);
  out.set(rgb);
  out.set(BF, 3 * SIZE * SIZE);
  Object.assign(model.values, { show_controls: false, is_rgb: [true, false], frame_bytes: new DataView(out.buffer) });
  act(() => { cleanup = renderShow2D({ model, el } as never) as (() => void) | undefined; });
  await settle();
  for (let i = 0; i < 6; i++) await paintFrame();
  expect(rgbPaints.length).toBeGreaterThan(0);
  const expected = Uint8ClampedArray.from({ length: 4 * SIZE * SIZE }, (_, k) => (
    k % 4 === 3 ? 255 : [0, 64, 128, 255, 0, 255, 0][(3 * Math.floor(k / 4) + (k % 4)) % 7]
  ));
  expect(Array.from(rgbPaints.at(-1)!.pixels)).toEqual(Array.from(expected));
  expect(canvasDraws.some(draw => draw.source === rgbPaints.at(-1)!.bitmap)).toBe(true);
});
