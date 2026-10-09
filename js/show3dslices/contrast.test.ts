// Show3DSlices mounted in jsdom, where there is no WebGPU, so every slice is
// painted on the Canvas2D path. Both paths must paint with the window the
// WebGPU path uses: the image_vmin_pct to image_vmax_pct share of the
// stack-wide range, the same for every slice. A per-slice window would paint
// different pixels on the two paths and change contrast while scrubbing.
// Explicit vmin/vmax are that window as given (Auto never narrows them), and
// the colorbar labels the window the slices are painted with, Flip included.
// Every switch carries its accessible name, Align fits the drift on the
// volume of the panel on screen, and the depth slider shows each slice while
// dragged, not only on release.
import { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { percentileClip, signedLog1p, sliderRange } from "../display/stats";

const painted = vi.hoisted(() => [] as { length: number; vmin: number; vmax: number }[]);
const colorbars = vi.hoisted(() => [] as { vmin: number; vmax: number }[]);
vi.mock("../figure", async (importOriginal) => {
  const original = await importOriginal<typeof import("../figure")>();
  return {
    ...original,
    drawColorbar: (_ctx: unknown, _w: number, _h: number, _lut: Uint8Array, vmin: number, vmax: number) => {
      colorbars.push({ vmin, vmax });
    },
  };
});
vi.mock("../display/colormaps", async (importOriginal) => {
  const original = await importOriginal<typeof import("../display/colormaps")>();
  return {
    ...original,
    createGPUColormapEngine: async () => null,
    renderToOffscreen: (data: Float32Array, width: number, height: number, _lut: Uint8Array, vmin: number, vmax: number) => {
      painted.push({ length: data.length, vmin, vmax });
      return Object.assign(document.createElement("canvas"), { width, height });
    },
    renderToOffscreenReuse: (data: Float32Array, _lut: Uint8Array, vmin: number, vmax: number) => {
      painted.push({ length: data.length, vmin, vmax });
    },
  };
});

const alignmentVolumes = vi.hoisted(() => [] as Float32Array[]);
vi.mock("../sliceAlignment", () => ({
  estimateSliceAlignment: async (volume: Float32Array) => {
    alignmentVolumes.push(volume.slice());
    return { rowShiftPxPerSlice: 0, colShiftPxPerSlice: 0, backend: "cpu" };
  },
}));

const { default: show3dslices } = await import("./index");

const NZ = 4, NY = 12, NX = 16;
// Each depth sits in its own band (slice z spans 100 z to 100 z + 50), so a
// slice's own 2/98 percentiles are far from the stack's.
const VOLUME = Float32Array.from({ length: NZ * NY * NX }, (_, index) => {
  const z = Math.floor(index / (NY * NX));
  const within = index % (NY * NX);
  return 100 * z + (50 * within) / (NY * NX - 1);
});

class Model {
  values: Record<string, unknown> = {
    volume_bytes: new DataView(VOLUME.buffer), nx: NX, ny: NY, nz: NZ,
    slice_x: 8, slice_y: 6, slice_z: 2, cmap: "plasma", auto_contrast: true,
    image_vmin_pct: 0, image_vmax_pct: 100, vmin: null, vmax: null, log_scale: false, flip: false,
    dim_labels: ["slice", "row", "col"], panel_count: 1, panel_titles: [], plane_visibility: [true, true],
    pixel_size_axes: [0, 0, 0], slice_alignment: "off", show_controls: true, show_title: true,
    show_crosshair: true, view_state: {}, oblique_angle: 0, oblique_profile_line: [], page_labels: [],
    n_pages: 1, page_idx: 0, panels_per_page: 0, active_panel: 0, z_stretch: 30, fps: 30,
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

let el: HTMLDivElement;
let model: Model;
let cleanup: (() => void) | undefined;

async function settle() { await act(async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); }); }

beforeEach(() => {
  painted.length = 0;
  colorbars.length = 0;
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("ImageData", class { constructor(readonly width: number, readonly height: number) {} });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(new Proxy({}, {
    get: (target: Record<string, unknown>, prop: string) => prop in target ? target[prop]
      : prop === "measureText" ? (text: string) => ({ width: text.length * 6 }) : () => undefined,
    set: (target: Record<string, unknown>, prop: string, value: unknown) => { target[prop] = value; return true; },
  }) as never);
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

it("paints every slice with one stack-wide window, as the WebGPU path does", async () => {
  act(() => { cleanup = show3dslices.render({ model, el } as never) as (() => void) | undefined; });
  await settle();
  act(() => model.set("slice_z", 1));
  await settle();
  const vminPct = model.get("image_vmin_pct") as number;
  const vmaxPct = model.get("image_vmax_pct") as number;
  // Auto snapped the sliders inside the stack range: the window is not the full range.
  expect(vminPct).toBeGreaterThan(0);
  expect(vmaxPct).toBeLessThan(100);
  const stackWindow = sliderRange(0, 350, vminPct, vmaxPct);
  const top = painted.filter(paint => paint.length === NY * NX);
  const side = painted.filter(paint => paint.length !== NY * NX);
  expect(top.length).toBeGreaterThanOrEqual(2);
  expect(side.length).toBeGreaterThanOrEqual(1);
  const last = (paints: typeof painted) => paints[paints.length - 1];
  for (const paint of [last(top), top[top.length - 2], last(side)]) {
    expect({ vmin: paint.vmin, vmax: paint.vmax }).toEqual(stackWindow);
  }
});

it("keeps Auto on the displayed values after a log toggle", async () => {
  act(() => { cleanup = show3dslices.render({ model, el } as never) as (() => void) | undefined; });
  await settle();
  act(() => model.set("log_scale", true));
  await settle();
  // Auto's window is the 2/98 percentile of the log-scaled volume, on every slice.
  const logVolume = Float32Array.from(VOLUME, signedLog1p);
  const { vmin, vmax } = percentileClip(logVolume, 2, 98);
  const top = painted.filter(paint => paint.length === NY * NX);
  const side = painted.filter(paint => paint.length !== NY * NX);
  for (const paint of [top[top.length - 1], side[side.length - 1]]) {
    expect(paint.vmin).toBeCloseTo(vmin, 9);
    expect(paint.vmax).toBeCloseTo(vmax, 9);
  }
});

it("paints explicit vmin/vmax as given while Auto is on", async () => {
  // The volume spans 0 to 350, so Auto's 2/98 percentiles sit inside these limits.
  model.values.vmin = -100;
  model.values.vmax = 500;
  act(() => { cleanup = show3dslices.render({ model, el } as never) as (() => void) | undefined; });
  await settle();
  act(() => model.set("slice_z", 1));
  await settle();
  expect([model.get("image_vmin_pct"), model.get("image_vmax_pct")]).toEqual([0, 100]);
  const top = painted.filter(paint => paint.length === NY * NX);
  const side = painted.filter(paint => paint.length !== NY * NX);
  for (const paint of [top[top.length - 1], top[top.length - 2], side[side.length - 1]]) {
    expect({ vmin: paint.vmin, vmax: paint.vmax }).toEqual({ vmin: -100, vmax: 500 });
  }
});

it("labels the colorbar with the flipped window the slices are painted with", async () => {
  model.values.show_colorbar = true;
  model.values.flip = true;
  act(() => { cleanup = show3dslices.render({ model, el } as never) as (() => void) | undefined; });
  await settle();
  const top = painted.filter(paint => paint.length === NY * NX);
  const last = top[top.length - 1];
  // Flip negates the displayed values: the window runs from -vmax to -vmin.
  expect(last.vmax).toBeLessThan(0);
  expect(colorbars.length).toBeGreaterThanOrEqual(1);
  expect(colorbars[colorbars.length - 1]).toEqual({ vmin: last.vmin, vmax: last.vmax });
});

it("names every switch for assistive technology", async () => {
  act(() => { cleanup = show3dslices.render({ model, el } as never) as (() => void) | undefined; });
  await settle();
  const switches = [...el.querySelectorAll<HTMLInputElement>("input[type=checkbox]")];
  expect(switches.length).toBeGreaterThanOrEqual(5);
  for (const input of switches) {
    expect(input.getAttribute("role")).toBe("switch");
    expect(input.getAttribute("aria-label")).toBeTruthy();
  }
});

it("fits slice alignment on the volume of the active panel", async () => {
  // The second panel is the first plus 1000, so the fitted values name the panel.
  const panels = new Float32Array(2 * VOLUME.length);
  panels.set(VOLUME);
  panels.set(VOLUME.map(value => value + 1000), VOLUME.length);
  model.values.volume_bytes = new DataView(panels.buffer);
  model.values.panel_count = 2;
  model.values.active_panel = 1;
  alignmentVolumes.length = 0;
  act(() => { cleanup = show3dslices.render({ model, el } as never) as (() => void) | undefined; });
  await settle();
  const advanced = [...el.querySelectorAll("button")].find(button => button.textContent === "Advanced")!;
  act(() => advanced.click());
  await settle();
  const align = el.querySelector<HTMLInputElement>('input[aria-label="Align slices with automatic global slice alignment"]')!;
  act(() => align.click());
  await settle();
  expect(alignmentVolumes).toHaveLength(1);
  expect(Array.from(alignmentVolumes[0])).toEqual(Array.from(panels.subarray(VOLUME.length)));
});

it("paints the dragged depth slice before release without WebGPU", async () => {
  const frames: FrameRequestCallback[] = [];
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => frames.push(callback));
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({ left: 0, top: 0, width: 300, height: 10, right: 300, bottom: 10, x: 0, y: 0, toJSON: () => ({}) });
  model.values.slice_z = 1;
  act(() => { cleanup = show3dslices.render({ model, el } as never) as (() => void) | undefined; });
  await settle();
  const slider = el.querySelector("input[aria-label^='slice slice']")!.closest(".MuiSlider-root")!;
  painted.length = 0;
  act(() => {
    slider.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, button: 0, clientX: 100, clientY: 5 }));
    document.dispatchEvent(new MouseEvent("mousemove", { bubbles: true, buttons: 1, clientX: 200, clientY: 5 }));
  });
  act(() => frames.splice(0).forEach(callback => callback(0)));
  expect(model.get("slice_z")).toBe(1);
  const top = painted.filter(paint => paint.length === NY * NX);
  expect(top).toHaveLength(1);
  const vminPct = model.get("image_vmin_pct") as number;
  const vmaxPct = model.get("image_vmax_pct") as number;
  expect({ vmin: top[0].vmin, vmax: top[0].vmax }).toEqual(sliderRange(0, 350, vminPct, vmaxPct));
  act(() => { document.dispatchEvent(new MouseEvent("mouseup", { bubbles: true, clientX: 200, clientY: 5 })); });
  await settle();
  expect(model.get("slice_z")).toBe(2);
});
