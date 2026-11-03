import { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import plot2d from "./index";

const gpu = vi.hoisted(() => ({
  uploadData: vi.fn(), destroy: vi.fn(), create: vi.fn(),
  renderSlotsToImageBitmapAsync: vi.fn(),
}));
vi.mock("../display/colormaps", () => ({
  COLORMAPS: { viridis: new Uint8Array(768), magma: new Uint8Array(768).fill(255) },
  createGPUColormapEngine: () => gpu.create(),
  renderToOffscreen: vi.fn(() => document.createElement("canvas")),
}));

class Bitmap {
  close = vi.fn();
}
class Model {
  values: Record<string, any> = {
    grid: { rows: 2, cols: 3, bounds: [0, 3, 0, 2] },
    data_bytes: new Uint8Array(new Float64Array([1, 2, 3, 4, 5, 6]).buffer),
    cmap: "viridis", vmin: 0, vmax: 6, view_bounds: [],
    max_width: 600, plot_height_px: 350,
    title: "Scalar map", x_label: "Distance", y_label: "Angle", colorbar_label: "Value",
  };
  listeners = new Map<string, Set<() => void>>();
  get(key: string) { return this.values[key]; }
  set(key: string, value: unknown) {
    this.values[key] = value;
    this.listeners.get(`change:${key}`)?.forEach(fn => fn());
  }
  save_changes = vi.fn();
  on(key: string, fn: () => void) {
    if (!this.listeners.has(key)) this.listeners.set(key, new Set());
    this.listeners.get(key)!.add(fn);
  }
  off(key: string, fn: () => void) { this.listeners.get(key)?.delete(fn); }
}
const CHAR_WIDTH = 7;
type DrawnText = { text: string; x: number; y: number; align?: string; baseline?: string; transform: { x: number; y: number; angle: number } };
let texts: DrawnText[];
let cleanup: (() => void) | undefined;
let frames: Map<number, FrameRequestCallback>;
let nextId: number;
let context: Record<string, any>;
let el: HTMLDivElement;
let model: Model;
async function settle() { await act(async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); }); }
function paintFrame() {
  act(() => { const pending = [...frames.values()]; frames.clear(); pending.forEach(fn => fn(0)); });
}
function deferred() {
  let resolve!: (value: Bitmap[]) => void;
  const promise = new Promise<Bitmap[]>(done => { resolve = done; });
  return { promise, resolve };
}
async function mount() {
  act(() => { cleanup = plot2d.render({ model, el }); });
  await settle(); paintFrame();
}
function hover() {
  const canvas = el.querySelector("canvas")!;
  canvas.onpointermove!({ clientX: 90, clientY: 190 } as PointerEvent);
  paintFrame();
}
beforeEach(() => {
  vi.clearAllMocks();
  frames = new Map(); nextId = 0;
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ImageBitmap", Bitmap);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("requestAnimationFrame", (fn: FrameRequestCallback) => { frames.set(++nextId, fn); return nextId; });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => frames.delete(id));
  context = Object.fromEntries(["scale", "fillRect", "save", "translate", "rotate", "drawImage", "restore", "strokeRect", "fillText", "setLineDash", "beginPath", "moveTo", "lineTo", "stroke"].map(key => [key, vi.fn()]));
  // 12px text, CHAR_WIDTH per character; the transform stack follows save/translate/rotate/restore.
  context.measureText = (text: string) => ({ width: text.length * CHAR_WIDTH });
  let transform = { x: 0, y: 0, angle: 0 };
  const stack: typeof transform[] = [];
  context.save.mockImplementation(() => stack.push({ ...transform }));
  context.restore.mockImplementation(() => { transform = stack.pop()!; });
  context.translate.mockImplementation((x: number, y: number) => { transform = { ...transform, x: transform.x + x, y: transform.y + y }; });
  context.rotate.mockImplementation((angle: number) => { transform = { ...transform, angle: transform.angle + angle }; });
  texts = [];
  context.fillText.mockImplementation((text: string, x: number, y: number) => {
    texts.push({ text, x, y, align: context.textAlign, baseline: context.textBaseline, transform: { ...transform } });
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context as any);
  document.body.dataset.jpThemeLight = "true";
  el = document.createElement("div"); document.body.append(el); model = new Model();
  gpu.create.mockResolvedValue(gpu);
  gpu.renderSlotsToImageBitmapAsync.mockResolvedValue([new Bitmap()]);
});
afterEach(async () => {
  act(() => cleanup?.()); cleanup = undefined;
  await settle(); el.remove(); delete document.body.dataset.jpThemeLight;
  vi.restoreAllMocks(); vi.unstubAllGlobals();
});

it("keeps visible pixels, hover and color limits together during rapid replacements", async () => {
  await mount(); hover();
  expect(el.textContent).toContain("value 1.00000");
  const pending = deferred(), stale = new Bitmap(), latest = new Bitmap();
  gpu.renderSlotsToImageBitmapAsync.mockReturnValueOnce(pending.promise).mockResolvedValueOnce([latest]);
  act(() => model.set("data_bytes", new Uint8Array(new Float64Array([11, 12, 13, 14, 15, 16]).buffer)));
  await settle(); hover();
  expect(el.textContent).toContain("value 1.00000");
  act(() => { model.set("cmap", "magma"); model.set("vmax", 20); });
  paintFrame();
  expect(context.fillText).not.toHaveBeenCalledWith("20", expect.anything(), expect.anything());
  pending.resolve([stale]); await settle();
  expect(stale.close).toHaveBeenCalledOnce();
  expect(context.drawImage.mock.calls.some((call: unknown[]) => call[0] === stale)).toBe(false);
  expect(context.drawImage.mock.lastCall?.[0]).toBe(latest);
  expect(context.fillText).toHaveBeenCalledWith("20", expect.anything(), expect.anything());
  expect(el.textContent).toContain("value 11.0000");
  expect(gpu.renderSlotsToImageBitmapAsync.mock.lastCall?.[3]).toBe("magma");
});

it("releases an in-flight bitmap and the engine when the view closes", async () => {
  const pending = deferred(), bitmap = new Bitmap();
  gpu.renderSlotsToImageBitmapAsync.mockReturnValueOnce(pending.promise);
  await mount();
  act(() => cleanup?.()); cleanup = undefined;
  pending.resolve([bitmap]); await settle(); paintFrame();
  expect(bitmap.close).toHaveBeenCalledOnce();
  expect(gpu.destroy).toHaveBeenCalledOnce();
  expect(context.drawImage).not.toHaveBeenCalled();
  expect(el.childElementCount).toBe(0);
  expect([...model.listeners.values()].every(set => set.size === 0)).toBe(true);
});

it("repaints notebook theme changes without changing scientific data", async () => {
  await mount(); const bytes = model.get("data_bytes");
  const uploads = gpu.uploadData.mock.calls.length;
  act(() => { document.body.dataset.jpThemeLight = "false"; });
  await settle(); paintFrame();
  expect(el.firstElementChild?.getAttribute("style")).toContain("rgb(30, 30, 30)");
  expect(context.fillStyle).toBe("#e0e0e0");
  expect(model.get("data_bytes")).toBe(bytes);
  expect(gpu.uploadData).toHaveBeenCalledTimes(uploads);
  expect(model.save_changes).not.toHaveBeenCalled();
});

it("keeps zoom previews local and commits the final viewport on reset", async () => {
  await mount();
  const bytes = model.get("data_bytes"), uploads = gpu.uploadData.mock.calls.length;
  const canvas = el.querySelector("canvas")!;
  canvas.dispatchEvent(new WheelEvent("wheel", { clientX: 150, clientY: 120, deltaY: -200, cancelable: true }));
  paintFrame();
  expect(model.save_changes).not.toHaveBeenCalled();
  expect(model.get("view_bounds")).toEqual([]);
  expect(context.drawImage.mock.lastCall?.[3]).toBeLessThan(3);
  const reset = [...el.querySelectorAll("button")].find(button => button.textContent === "Reset View")!;
  reset.click(); paintFrame();
  expect(model.get("view_bounds")).toEqual([0, 3, 0, 2]);
  expect(context.drawImage.mock.lastCall?.slice(1, 5)).toEqual([0, 0, 3, 2]);
  expect(model.get("data_bytes")).toBe(bytes);
  expect(gpu.uploadData).toHaveBeenCalledTimes(uploads);
});

it("reports canvas fallback while retaining original readout values", async () => {
  gpu.renderSlotsToImageBitmapAsync.mockResolvedValueOnce(null);
  await mount(); hover();
  expect(el.textContent).toContain("CPU display");
  expect(el.textContent).toContain("value 1.00000");
  expect(context.drawImage.mock.lastCall?.[0]).toBeInstanceOf(HTMLCanvasElement);
});

it("keeps a failed replacement from changing the displayed source", async () => {
  await mount(); hover();
  gpu.renderSlotsToImageBitmapAsync.mockRejectedValueOnce(new Error("render interrupted"));
  act(() => model.set("data_bytes", new Uint8Array(new Float64Array([21, 22, 23, 24, 25, 26]).buffer)));
  await settle(); hover();
  expect(el.textContent).toContain("Display error: Error: render interrupted");
  expect(el.textContent).toContain("value 1.00000");
  act(() => model.set("cmap", "magma")); await settle();
  expect(el.textContent).toContain("value 21.0000");
  expect(el.textContent).not.toContain("Display error");
});


it("shows a lightweight saved preview until live data arrives", async () => {
  model.values.data_bytes = new Uint8Array();
  model.values._static_fallback_image = "iVBORw0KGgo=";
  model.values._static_fallback_mime = "image/png";
  await mount();
  expect(el.querySelector("img")?.style.display).toBe("block");
  expect(el.querySelector("canvas")?.style.display).toBe("none");
  expect(gpu.uploadData).not.toHaveBeenCalled();
  expect(el.textContent).toContain("rerun the cell for interaction");
  act(() => model.set("data_bytes", new Uint8Array(new Float64Array([1, 2, 3, 4, 5, 6]).buffer)));
  await settle(); hover();
  expect(el.querySelector("img")?.style.display).toBe("none");
  expect(el.querySelector("canvas")?.style.display).toBe("block");
  expect(el.textContent).toContain("value 1.00000");
});


it("releases a late engine when a saved-preview view closes before initialization", async () => {
  let resolve!: (value: typeof gpu) => void;
  gpu.create.mockReturnValueOnce(new Promise<typeof gpu>(done => { resolve = done; }));
  model.values.data_bytes = new Uint8Array();
  await mount();
  act(() => cleanup?.()); cleanup = undefined;
  expect(gpu.destroy).not.toHaveBeenCalled();
  resolve(gpu); await settle();
  expect(gpu.destroy).toHaveBeenCalledOnce();
});


/** Canvas box of drawn 12px text (ascent 9, descent 3), under the transform at draw time. */
function textBox({ text, x, y, align, baseline, transform }: DrawnText) {
  const width = text.length * CHAR_WIDTH;
  const left = x + (align === "right" ? -width : align === "center" ? -width / 2 : 0);
  const [top, bottom] = baseline === "top" ? [y, y + 12] : baseline === "middle" ? [y - 6, y + 6] : [y - 9, y + 3];
  // The y-axis title is the only rotated text: a quarter turn counterclockwise.
  if (transform.angle !== 0) return { text, left: transform.x + top, right: transform.x + bottom, top: transform.y - left - width, bottom: transform.y - left };
  return { text, left: transform.x + left, right: transform.x + left + width, top: transform.y + top, bottom: transform.y + bottom };
}

it("keeps every label inside the canvas and clear of the others at any zoom", async () => {
  // Ticks sit at 1, 2, 2.5 or 5 times a power of ten with the decimals the step needs.
  const cases = [
    // the view a 6.8x wheel zoom left on a Plot2D whose corner labels collided and clipped
    { bounds: [-3, 3, -2, 2], view: [-1.251, -0.3609, -0.611, -0.01465],
      x: ["-1.2", "-1.0", "-0.8", "-0.6", "-0.4"], y: ["-0.6", "-0.5", "-0.4", "-0.3", "-0.2", "-0.1"] },
    { bounds: [-3, 3, -2, 2], view: [], x: ["-3", "-2", "-1", "0", "1", "2", "3"], y: ["-2", "-1", "0", "1", "2"] },
    { bounds: [-3, 3, -2, 2], view: [2.95, 3, -0.0003, 0.0001],
      x: ["2.96", "2.98", "3.00"], y: ["-0.0003", "-0.0002", "-0.0001", "0.0000", "0.0001"] },
    { bounds: [-5e5, 5e5, -3, 3], view: [], x: ["-5e+5", "0", "5e+5"], y: ["-3", "-2", "-1", "0", "1", "2", "3"] },
    // labels thirteen characters long leave room for one x label on this 260 px canvas
    { bounds: [1e6, 1e6 + 10, 0, 1], view: [1e6 + 1.23456, 1e6 + 1.3, 0.1, 0.10001],
      x: ["1.00000125e+6"], y: ["1.00000e-1", "1.00002e-1", "1.00004e-1", "1.00006e-1", "1.00008e-1", "1.00010e-1"] },
  ];
  for (const { bounds, view, x, y } of cases) {
    act(() => cleanup?.());
    model = new Model();
    Object.assign(model.values, {
      grid: { rows: 60, cols: 80, bounds }, data_bytes: new Uint8Array(new Float64Array(60 * 80).buffer), view_bounds: view,
      title: "Plot2D", x_label: "x (nm)", y_label: "y (nm)", colorbar_label: "intensity", vmin: -0.09946, vmax: 1.045,
    });
    await mount();
    texts = [];
    act(() => model.set("plot_height_px", 350));
    paintFrame();
    const canvas = el.querySelector("canvas")!;
    const boxes = texts.map(textBox);
    for (const box of boxes) {
      expect(box.left >= 0 && box.right <= canvas.width && box.top >= 0 && box.bottom <= canvas.height, `${box.text} leaves the canvas`).toBe(true);
    }
    boxes.forEach((box, index) => boxes.slice(index + 1).forEach(other => {
      const overlap = box.left < other.right && other.left < box.right && box.top < other.bottom && other.top < box.bottom;
      expect(overlap, `"${box.text}" overlaps "${other.text}"`).toBe(false);
    }));
    // Every tick label names the value at its tick (x labels may shift inward at the canvas edge).
    const [left, top, width, height] = context.strokeRect.mock.lastCall;
    const [xLow, xHigh, yLow, yHigh] = view.length ? view : bounds;
    const xTicks = texts.filter(text => text.baseline === "top");
    const yTicks = texts.filter(text => text.baseline === "middle");
    expect(xTicks.map(tick => tick.text)).toEqual(x);
    expect(yTicks.map(tick => tick.text)).toEqual(y);
    for (const tick of xTicks) {
      const value = Number(tick.text), half = (tick.text.length * CHAR_WIDTH) / 2;
      expect(value >= xLow && value <= xHigh, tick.text).toBe(true);
      const position = left + ((value - xLow) / (xHigh - xLow)) * width;
      expect(tick.x).toBeCloseTo(Math.min(Math.max(position, half + 1), canvas.width - half - 1), 3);
    }
    for (const tick of yTicks) {
      const value = Number(tick.text);
      expect(value >= yLow && value <= yHigh, tick.text).toBe(true);
      expect(tick.y).toBeCloseTo(top + height * (1 - (value - yLow) / (yHigh - yLow)), 3);
    }
  }
});
