// Show1D's plot canvas mounted in jsdom with a 2D context that keeps the
// current path the way a browser does: beginPath empties it, path calls append
// to it, and save/restore leave it alone. Clipping to the plot frame must clip
// to the frame rectangle only, on every redraw.
import { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import show1d from "./index";

const PATH_CALLS = new Set(["moveTo", "lineTo", "rect", "arc", "arcTo", "ellipse", "roundRect", "closePath", "quadraticCurveTo", "bezierCurveTo"]);

type Recorded = { name: string; args: unknown[]; path: string[] };

function recordingContext(log: Recorded[]) {
  let path: string[] = [];
  const state: Record<string, unknown> = {};
  return new Proxy(state, {
    get(target, prop) {
      if (prop in target) return target[prop as string];
      if (prop === "measureText") return (text: string) => ({ width: text.length * 6 });
      if (prop === "getImageData") return (_x: number, _y: number, w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h });
      return (...args: unknown[]) => {
        const name = String(prop);
        if (name === "beginPath") path = [];
        else if (PATH_CALLS.has(name)) path.push(name);
        log.push({ name, args, path: path.slice() });
      };
    },
    set(target, prop, value) {
      target[prop as string] = value;
      return true;
    },
  });
}

class Model {
  values: Record<string, unknown> = {
    y_bytes: new DataView(new Float32Array([1, 3, 2, 5, 2, 1, 4, 3]).buffer),
    x_bytes: new DataView(new ArrayBuffer(0)),
    n_traces: 2, n_points: 4, labels: ["particle", "support"], colors: ["#0072B2", "#D55E00"],
    plot_height_px: 320, plot_width_px: 400, show_grid: true, show_legend: true, show_controls: true,
    log_scale: false, line_width: 1.5, focused_trace: -1, x_range: [], y_range: [],
    x_label: "Scattering angle", x_unit: "mrad", y_label: "Intensity", y_unit: "counts",
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

let logs: Map<HTMLCanvasElement, Recorded[]>;
let frames: Map<number, FrameRequestCallback>;
let nextFrame: number;
let el: HTMLDivElement;
let model: Model;
let cleanup: (() => void) | undefined;

async function settle() { await act(async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); }); }
function paintFrames() {
  act(() => { const pending = [...frames.values()]; frames.clear(); pending.forEach(fn => fn(0)); });
}
function plotCanvas(): HTMLCanvasElement {
  return el.querySelector("canvas[data-quantem-scientific-output='show1d-plot']") as HTMLCanvasElement;
}
function hoverCanvas(): HTMLCanvasElement {
  return plotCanvas().nextElementSibling as HTMLCanvasElement;
}
function clipPaths(canvas: HTMLCanvasElement): string[][] {
  return (logs.get(canvas) ?? []).filter(call => call.name === "clip").map(call => call.path);
}

beforeEach(() => {
  logs = new Map();
  frames = new Map();
  nextFrame = 0;
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("requestAnimationFrame", (fn: FrameRequestCallback) => { frames.set(++nextFrame, fn); return nextFrame; });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => frames.delete(id));
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function (this: HTMLCanvasElement) {
    if (!logs.has(this)) {
      const log: Recorded[] = [];
      logs.set(this, log);
      (this as unknown as { recordingContext: unknown }).recordingContext = recordingContext(log);
    }
    return (this as unknown as { recordingContext: unknown }).recordingContext;
  } as never);
  el = document.createElement("div");
  document.body.append(el);
  model = new Model();
});
afterEach(async () => {
  act(() => cleanup?.());
  cleanup = undefined;
  await settle();
  el.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

async function mount() {
  act(() => { cleanup = show1d.render({ model, el } as never) as (() => void) | undefined; });
  await settle();
  paintFrames();
}

it("clips every redraw of the traces to the plot frame alone", async () => {
  await mount();
  act(() => model.set("log_scale", true));
  await settle();
  act(() => model.set("log_scale", false));
  await settle();
  const clips = clipPaths(plotCanvas());
  expect(clips.length).toBeGreaterThanOrEqual(3);
  for (const path of clips) expect(path).toEqual(["rect"]);
});

it("clips every hover marker to the plot frame alone", async () => {
  await mount();
  const points = (logs.get(plotCanvas()) ?? []).filter(call => call.name === "moveTo" || call.name === "lineTo");
  const tracePoints = points.slice(-4).map(call => call.args as number[]);
  for (const [clientX, clientY] of tracePoints.slice(0, 3)) {
    act(() => { plotCanvas().dispatchEvent(new MouseEvent("pointermove", { bubbles: true, clientX, clientY })); });
    paintFrames();
    await settle();
  }
  const clips = clipPaths(hoverCanvas());
  expect(clips.length).toBeGreaterThanOrEqual(2);
  for (const path of clips) expect(path).toEqual(["rect"]);
});

it("zooms on the wheel without scrolling the notebook", async () => {
  await mount();
  const wheel = new WheelEvent("wheel", { bubbles: true, cancelable: true, clientX: 200, clientY: 120, deltaY: -200 });
  act(() => { plotCanvas().dispatchEvent(wheel); });
  await settle();
  expect(wheel.defaultPrevented).toBe(true);
  const [low, high] = model.get("x_range") as number[];
  expect(high - low).toBeLessThan(3);
});

it("zooms y on Shift+wheel whichever axis Chrome puts the delta on", async () => {
  await mount();
  const ySpan = () => { const [low, high] = model.get("y_range") as number[]; return high - low; };
  // Firefox keeps a Shift+wheel notch on deltaY: zoom in.
  act(() => { plotCanvas().dispatchEvent(new WheelEvent("wheel", { bubbles: true, cancelable: true, clientX: 200, clientY: 120, deltaY: -200, shiftKey: true })); });
  await settle();
  const zoomedIn = ySpan();
  // Chrome moves it to deltaX with deltaY = 0: the same notch the other way zooms back out.
  const wheel = new WheelEvent("wheel", { bubbles: true, cancelable: true, clientX: 200, clientY: 120, deltaX: 200, deltaY: 0, shiftKey: true });
  act(() => { plotCanvas().dispatchEvent(wheel); });
  await settle();
  expect(wheel.defaultPrevented).toBe(true);
  expect(ySpan() / zoomedIn).toBeCloseTo(Math.exp(0.2), 9);
});

it("labels a log axis zoomed inside one decade with round values", async () => {
  await mount();
  act(() => { model.set("log_scale", true); model.set("y_range", [5600, 6400]); });
  await settle();
  const draws = logs.get(plotCanvas()) ?? [];
  const lastClear = draws.map(call => call.name).lastIndexOf("fillRect");
  const yLabels = draws.slice(lastClear).filter(call => call.name === "fillText" && call.args[1] === 56).map(call => String(call.args[0]));
  expect(yLabels).toEqual(["5600", "5700", "5800", "5900", "6000", "6100", "6200", "6300", "6400"]);
});
