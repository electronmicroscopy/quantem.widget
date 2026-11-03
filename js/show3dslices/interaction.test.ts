// Show3DSlices pointer and label behavior, mounted in jsdom (no WebGPU, so the
// slices paint on the Canvas2D path). A drag of the oblique cut's handle moves
// the cut once per pointer move, from one document listener bound once; a
// click on an FFT panel ends the pan even when the peak refinement after it
// fails; the Top slice counter and its slider name count slices from 1, every
// label spells the length unit px, and the FFT panels name the CPU FFT when
// they use it.
import { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

vi.mock("../display/colormaps", async (importOriginal) => {
  const original = await importOriginal<typeof import("../display/colormaps")>();
  return {
    ...original,
    createGPUColormapEngine: async () => null,
    renderToOffscreen: (_data: Float32Array, width: number, height: number) => Object.assign(document.createElement("canvas"), { width, height }),
    renderToOffscreenReuse: () => {},
  };
});
const peakRefinement = vi.hoisted(() => ({ findFFTPeakBrowser: vi.fn() }));
vi.mock("../display/geometry", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../display/geometry")>()),
  findFFTPeakBrowser: peakRefinement.findFFTPeakBrowser,
}));

const { default: show3dslices } = await import("./index");

const NUM_SLICES = 4, NUM_ROWS = 12, NUM_COLS = 16;
const VOLUME = Float32Array.from({ length: NUM_SLICES * NUM_ROWS * NUM_COLS }, (_, index) => Math.sin(index / 7) + (index % NUM_COLS) / NUM_COLS);

class Model {
  values: Record<string, unknown> = {
    volume_bytes: new DataView(VOLUME.buffer), volume_shape: [NUM_SLICES, NUM_ROWS, NUM_COLS],
    col_index: 8, row_index: 6, slice_index: 2, cmap: "plasma", auto_contrast: true,
    image_vmin_pct: 0, image_vmax_pct: 100, vmin: null, vmax: null, log_scale: false, flip: false,
    dim_labels: ["slice", "row", "col"], panel_count: 1, panel_titles: [], plane_visibility: [true, true],
    voxel_size_A: [0, 0, 0], slice_alignment: "off", show_controls: true, show_title: true,
    show_crosshair: true, view_state: {}, oblique_angle: 0, oblique_profile_line: [], page_labels: [],
    n_pages: 1, page_idx: 0, panels_per_page: 0, active_panel: 0, slice_stretch: 30, fps: 30, loop: false,
  };
  writes: string[] = [];
  listeners = new Map<string, Set<() => void>>();
  get(key: string) { return this.values[key]; }
  set(key: string, value: unknown) {
    this.writes.push(key);
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
function mount() {
  act(() => { cleanup = show3dslices.render({ model, el } as never) as (() => void) | undefined; });
}
function mouse(target: EventTarget, type: string, clientX: number, clientY: number, buttons = 1) {
  target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, button: 0, buttons, clientX, clientY }));
}
const slicePanel = () => el.querySelector<HTMLCanvasElement>('canvas[data-quantem-scientific-output="show3dslices-slice-0"]')!;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("ImageData", class { constructor(readonly width: number, readonly height: number) {} });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(new Proxy({}, {
    get: (target: Record<string, unknown>, prop: string) => prop in target ? target[prop]
      : prop === "measureText" ? (text: string) => ({ width: text.length * 6 }) : () => undefined,
    set: (target: Record<string, unknown>, prop: string, value: unknown) => { target[prop] = value; return true; },
  }) as never);
  // Every canvas sits at the page origin at its own pixel size, so client and canvas coordinates agree.
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
    const width = this instanceof HTMLCanvasElement ? this.width : 300;
    const height = this instanceof HTMLCanvasElement ? this.height : 10;
    return { left: 0, top: 0, width, height, right: width, bottom: height, x: 0, y: 0, toJSON: () => ({}) };
  });
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

it("moves the oblique cut once per pointer move of a handle drag", async () => {
  mount();
  await settle();
  // 16 x 12 voxels on a 480 x 360 canvas (30 px per voxel); the default cut runs
  // along row 6 from column 1 to 14, so its start handle sits at (30, 180).
  const canvas = slicePanel();
  act(() => mouse(canvas, "mousedown", 30, 180));
  model.writes.length = 0;
  act(() => mouse(canvas, "mousemove", 90, 120));
  expect(model.writes.filter(key => key === "oblique_profile_line")).toHaveLength(1);
  act(() => mouse(document, "mousemove", 120, 60));
  expect(model.writes.filter(key => key === "oblique_profile_line")).toHaveLength(2);
  act(() => mouse(document, "mouseup", 120, 60, 0));
  await settle();
  // The released handle sits where the pointer let go: column 4, row 2.
  expect(model.get("oblique_profile_line")).toEqual([{ row: 2, col: 4 }, { row: 6, col: 14 }]);
});

it("binds the oblique drag listeners once, not on every render", async () => {
  mount();
  await settle();
  const added = vi.spyOn(document, "addEventListener");
  for (const cmap of ["viridis", "magma", "inferno"]) {
    act(() => model.set("cmap", cmap));
    await settle();
  }
  expect(added.mock.calls.filter(([type]) => type === "mousemove")).toHaveLength(0);
});

it("ends the FFT pan on a click whose peak refinement fails", async () => {
  model.values.show_fft = true;
  peakRefinement.findFFTPeakBrowser.mockRejectedValue(new Error("device lost"));
  vi.spyOn(console, "error").mockImplementation(() => {});
  mount();
  // The FFT panels compute after an 80 ms debounce; the click snaps to a peak of the cached magnitude.
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 250)); });
  const fft = el.querySelector<HTMLCanvasElement>('canvas[data-quantem-scientific-output="show3dslices-fft-0"]')!;
  // Pressed and released before React commits, as automation and fast clicks deliver them.
  act(() => {
    mouse(fft, "mousedown", 100, 100);
    mouse(fft, "mouseup", 100, 100, 0);
  });
  await settle();
  expect(peakRefinement.findFFTPeakBrowser).toHaveBeenCalledOnce();
  // With no button held, moving over the FFT must not pan it.
  act(() => mouse(fft, "mousemove", 160, 100, 0));
  act(() => mouse(document, "mouseup", 160, 100, 0));
  await settle();
  const fftZooms = (model.get("view_state") as { fft_zooms?: { panX: number }[] }).fft_zooms;
  expect(fftZooms?.[0].panX ?? 0).toBe(0);
});

it("counts Top slices from 1 in the counter and the slider name", async () => {
  mount();
  await settle();
  const slider = el.querySelector<HTMLInputElement>("input[aria-label='Top slice 3 of 4']");
  expect(slider).not.toBeNull();
  const row = slider!.closest(".MuiSlider-root")!.parentElement!;
  expect(row.textContent).toBe("slice3/4");
  act(() => { model.set("loop", true); model.set("slice_index", 0); });
  await settle();
  expect(el.querySelector("input[aria-label='Loop range and current top slice (1 of 4, loop 1 to 4)']")).not.toBeNull();
});

it("spells the length unit px in every label", async () => {
  model.values.slice_alignment = "manual";
  model.values.show_fft = true;
  mount();
  await settle();
  const advanced = [...el.querySelectorAll("button")].find(button => button.textContent === "Advanced")!;
  act(() => advanced.click());
  await settle();
  const labels = [...el.querySelectorAll("[aria-label], [title]")].flatMap(node => [node.getAttribute("aria-label"), node.getAttribute("title")]);
  expect(labels.filter(label => label?.includes("Row shift per slice"))).toEqual(["Row shift per slice 0.000 px"]);
  expect(labels.filter(label => label && /pixels/.test(label))).toEqual([]);
});

it("names the CPU FFT on the FFT panels of a browser without WebGPU", async () => {
  model.values.show_fft = true;
  mount();
  await settle();
  const captions = [...el.querySelectorAll(".MuiTypography-caption")].map(node => node.textContent ?? "").filter(text => text.startsWith("FFT "));
  expect(captions).toEqual(["FFT Top (CPU fallback)", "FFT oblique 0.0° (CPU fallback)"]);
});
