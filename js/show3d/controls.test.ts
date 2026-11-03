// Show3D mounted in jsdom (no WebGPU, so the canvas paths run) against a model that records writes.
// The user stories: dragging a control writes its trait once, on release, not once per path or per step.
import { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

// jsdom has no WebGPU, so the colormap engine is null (the canvas path) unless a test supplies one.
const gpuEngine = vi.hoisted(() => ({ current: null as unknown }));
vi.mock("../display/colormaps", async (importOriginal) => ({
  ...await importOriginal<typeof import("../display/colormaps")>(),
  createGPUColormapEngine: async () => gpuEngine.current,
}));

const { default: { render: renderShow3D } } = await import("./index");

// Mounting the whole widget in jsdom takes seconds on a loaded machine.
// GitHub's two-core runner ran these mounts up to 17 times slower than a workstation core (248 ms here, 4.4 s there),
// and the two-widget reorder test (1.2 s on one core here) passed 30 s there
vi.setConfig({ testTimeout: 120_000 });

const SIZE = 16;
const FRAMES = 4;

class TestImageData {
  constructor(readonly data: Uint8ClampedArray, readonly width: number, readonly height: number) {}
}

class Model {
  values: Record<string, unknown>;
  writes: [string, unknown][] = [];
  listeners = new Map<string, Set<() => void>>();
  constructor(overrides: Record<string, unknown> = {}) {
    const stack = Float32Array.from({ length: FRAMES * SIZE * SIZE }, (_, i) => Math.sin(i / 7) + (i % SIZE) / SIZE);
    this.values = {
      n_slices: FRAMES, width: SIZE, height: SIZE, n_panels: 1, slice_idx: 0, labels: [], title: "", dim_label: "Frame",
      _offline_float_stack: new DataView(stack.buffer), data_min: -1, data_max: 2, auto_vmins: [], auto_vmaxs: [],
      cmap: "gray", show_controls: true, controls_collapsed: false, show_title: true, fps: 30, loop: false,
      loop_start: 0, loop_end: -1, percentile_low: 0.5, percentile_high: 99.5, auto_contrast: false, link_contrast: true,
      link_panels: true, panel_cmaps: [], hidden_panels: [], selected_panels: [], panel_order: [], panel_titles: [],
      image_vmin_pct: 0, image_vmax_pct: 100, vmin_per_panel: [], vmax_per_panel: [], roi_list: [], roi_selected_idx: -1,
      profile_line: [], profile_width: 1, frame_rotations: [], image_rotation: 0, bookmarked_frames: [], playback_path: [],
      diff_mode: "off", avg_window: 1, compare_mode: "off", compare_pair: [0, 1], blink_fps: 2, denoise: "none",
      denoise_sigma: 4, denoise_bin: 1, denoise_modes: [], denoise_sigmas: [], denoise_bins: [], denoise_scope: "all",
      denoise_enabled: false, show_denoise: false, frequency_filter: "none", frequency_filter_enabled: false,
      frequency_filter_cutoff: 0.15, frequency_filter_center: 0.3, frequency_filter_width: 0.12, frequency_filter_modes: [],
      frequency_filter_cutoffs: [], frequency_filter_centers: [], frequency_filter_widths: [], frequency_filter_scope: "all",
      show_frequency_filter: false, show_fft: false, fft_layout: "bottom", fft_overlay_size: 0.35, fft_overlay_zoom: 1,
      fft_window: true, pixel_size: 0, pixel_unit: "px", show_scale_bar: true, show_panel_titles: true, panel_width_px: 0,
      max_cols: 4, panel_gap_px: 0, n_pages: 1, page_idx: 0, panels_per_page: 0, view_state: {},
      ...overrides,
    };
  }
  get(key: string) { return this.values[key]; }
  set(key: string, value: unknown) {
    this.values[key] = value;
    this.writes.push([key, value]);
    this.listeners.get(`change:${key}`)?.forEach(fn => fn());
  }
  save_changes = vi.fn();
  send = vi.fn();
  on(key: string, fn: () => void) {
    if (!this.listeners.has(key)) this.listeners.set(key, new Set());
    this.listeners.get(key)!.add(fn);
  }
  off(key: string, fn: () => void) { this.listeners.get(key)?.delete(fn); }
  /** Writes of `key` since the last call. */
  written(key: string): unknown[] {
    const values = this.writes.filter(([name]) => name === key).map(([, value]) => value);
    this.writes = this.writes.filter(([name]) => name !== key);
    return values;
  }
}

let el: HTMLDivElement;
let cleanup: (() => void) | undefined;
let unmountOthers: (() => void)[] = [];
// jsdom has no pointer capture; the ring's pointer path needs it.
const captured = new Set<number>();
const pointerCapture = {
  setPointerCapture(id: number) { captured.add(id); },
  hasPointerCapture(id: number) { return captured.has(id); },
  releasePointerCapture(id: number) { captured.delete(id); },
};

async function settle() { await act(async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); }); }

async function mount(model: Model) {
  act(() => { cleanup = renderShow3D({ model, el } as never) as (() => void) | undefined; });
  await settle();
}

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("ImageData", TestImageData);
  vi.stubGlobal("createImageBitmap", (image: TestImageData) => Promise.resolve(Object.assign(image, { close() {} })));
  vi.stubGlobal("requestAnimationFrame", () => 0);
  vi.stubGlobal("cancelAnimationFrame", () => undefined);
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(new Proxy({}, {
    get: (target: Record<string, unknown>, prop: string) => prop in target ? target[prop]
      : prop === "measureText" ? (text: string) => ({ width: text.length * 6 })
        : prop === "getImageData" || prop === "createImageData" ? (...args: number[]) => new TestImageData(new Uint8ClampedArray(args[args.length - 2] * args[args.length - 1] * 4), args[args.length - 2], args[args.length - 1])
          : () => undefined,
    set: (target: Record<string, unknown>, prop: string, value: unknown) => { target[prop] = value; return true; },
  }) as never);
  Object.assign(Element.prototype, pointerCapture);
  captured.clear();
  el = document.createElement("div");
  document.body.append(el);
});
afterEach(async () => {
  gpuEngine.current = null;
  act(() => { for (const unmount of unmountOthers) unmount(); });
  unmountOthers = [];
  act(() => cleanup?.());
  cleanup = undefined;
  await settle();
  el.remove();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  for (const name of Object.keys(pointerCapture)) delete (Element.prototype as unknown as Record<string, unknown>)[name];
});

/**
 * A mouse drag as a browser delivers it: each pointer event, then its compatibility mouse event unless the
 * pointerdown was cancelled (the Pointer Events rule that keeps one gesture from running two handlers).
 */
function mouseDrag(target: Element, points: [number, number][]) {
  const fire = (type: string, [clientX, clientY]: [number, number]) => {
    const pointer = new PointerEvent(type, { bubbles: true, cancelable: true, clientX, clientY, pointerId: 1, button: 0, buttons: type === "pointerup" ? 0 : 1 });
    act(() => { target.dispatchEvent(pointer); });
    return pointer;
  };
  let compatible = true;
  const mouse = (type: string, [clientX, clientY]: [number, number], mouseTarget: EventTarget) => {
    if (compatible) act(() => { mouseTarget.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, clientX, clientY, button: 0 })); });
  };
  compatible = !fire("pointerdown", points[0]).defaultPrevented;
  mouse("mousedown", points[0], target);
  for (const point of points.slice(1, -1)) {
    fire("pointermove", point);
    mouse("mousemove", point, document);
  }
  fire("pointerup", points[points.length - 1]);
  mouse("mouseup", points[points.length - 1], document);
}

it("commits one frequency cutoff, where the ring is released, for a mouse drag of the FFT ring", async () => {
  const model = new Model({ show_fft: true, frequency_filter: "lowpass", frequency_filter_enabled: true, frequency_filter_cutoff: 0.3 });
  await mount(model);
  const ring = el.querySelector(".quantem-frequency-filter-ring")!;
  vi.spyOn(ring.parentElement!, "getBoundingClientRect").mockReturnValue({ left: 0, top: 0, width: 200, height: 200 } as DOMRect);
  model.writes = [];
  // the ring's radius is the cutoff as a fraction of Nyquist, the half side of the FFT (100 px)
  mouseDrag(ring, [[130, 100], [150, 100], [160, 100], [170, 100]]);
  await settle();
  expect(model.written("frequency_filter_cutoff")).toEqual([0.7]);
});

/** A mouse drag of the MUI slider whose hidden input has `label`, on a track 0 to 100 px wide. */
function dragSlider(label: string, positions: number[], betweenMoves: () => void = () => {}) {
  const root = el.querySelector(`input[aria-label="${label}"]`)!.closest(".MuiSlider-root")!;
  vi.spyOn(root, "getBoundingClientRect").mockReturnValue({ left: 0, width: 100, top: 0, height: 10, bottom: 10, right: 100 } as DOMRect);
  const mouse = (type: string, clientX: number, target: EventTarget) => act(() => {
    target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, clientX, clientY: 5, button: 0, buttons: type === "mouseup" ? 0 : 1 }));
  });
  mouse("mousedown", positions[0], root);
  for (const position of positions.slice(1)) {
    mouse("mousemove", position, document);
    betweenMoves();
  }
  mouse("mouseup", positions[positions.length - 1], document);
}

it("commits the band-pass width once, on release, not at every drag step", async () => {
  const model = new Model({ frequency_filter: "bandpass", frequency_filter_enabled: true, show_frequency_filter: true });
  await mount(model);
  model.writes = [];
  dragSlider("Band-pass width as fraction of Nyquist", [20, 30, 40, 60]);
  await settle();
  // 60 % of the 0.01 to 1 track, on the 0.005 step
  expect(model.written("frequency_filter_width")).toEqual([0.605]);
});

it("keeps the current frame while a loop end is dragged", async () => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  const model = new Model({ n_slices: FRAMES, loop: true, loop_start: 0, loop_end: 3, slice_idx: 2 });
  await mount(model);
  model.writes = [];
  // the start thumb (frame 0) moves to frame 1; the current frame (2) must not be re-sent mid-drag
  const root = el.querySelector('input[data-index="0"]')!.closest(".MuiSlider-root")!;
  vi.spyOn(root, "getBoundingClientRect").mockReturnValue({ left: 0, width: 90, top: 0, height: 10, bottom: 10, right: 90 } as DOMRect);
  const mouse = (type: string, clientX: number, target: EventTarget) => act(() => {
    target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, clientX, clientY: 5, button: 0, buttons: 1 }));
  });
  mouse("mousedown", 1, root);
  mouse("mousemove", 32, document);
  act(() => { vi.advanceTimersByTime(1000); });
  expect(model.written("loop_start")).toEqual([1]);
  expect(model.written("slice_idx")).toEqual([]);
});

it("marks More as active while denoise is on with its settings row closed", async () => {
  const model = new Model({ denoise: "gaussian", denoise_enabled: true, show_denoise: false });
  await mount(model);
  const more = el.querySelector('button[aria-label="More tools"]')!;
  const badge = more.parentElement!.querySelector(".MuiBadge-badge")!;
  expect(badge.textContent).toBe("1");
  expect(badge.classList.contains("MuiBadge-invisible")).toBe(false);
});

it("resizes a ROI by the distance the pointer moved from the press, not to the press point", async () => {
  const size = 256;
  const stack = new Float32Array(FRAMES * size * size).fill(1);
  const model = new Model({
    width: size, height: size, _offline_float_stack: new DataView(stack.buffer), roi_active: true, roi_selected_idx: 0,
    roi_list: [{ row: 128, col: 128, shape: "square", radius: 40, radius_inner: 5, width: 80, height: 80 }],
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLCanvasElement) {
    return { left: 0, top: 0, width: this.width, height: this.height, right: this.width, bottom: this.height } as DOMRect;
  });
  await mount(model);
  const canvas = el.querySelector('canvas[data-quantem-scientific-output="show3d-image"]') as HTMLCanvasElement;
  const pixel = canvas.width / size;  // canvas pixels per image pixel
  const mouse = (type: string, col: number) => act(() => {
    canvas.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, clientX: col * pixel, clientY: 128 * pixel, button: 0, buttons: 1 }));
  });
  model.writes = [];
  // press 3 px inside the right edge of the square (half side 40), inside the resize band
  mouse("mousedown", 165);
  mouse("mousemove", 165);
  mouse("mousemove", 168);
  mouse("mouseup", 168);
  await settle();
  const radii = model.written("roi_list").map((list) => (list as { radius: number }[])[0].radius);
  expect(radii).toEqual([40, 43]);
});

it("releases every GPU slot of a replaced stack, the moving-average slot included", async () => {
  // A recording engine: the resident frames and the moving average fill slots, a new stack must free them.
  const filled = new Set<number>();
  const released = new Set<number>();
  gpuEngine.current = new Proxy({
    uploadData: (slot: number) => { filled.add(slot); },
    averageResidentSlotsInto: (slot: number) => { filled.add(slot); return true; },
    releaseSlot: (slot: number) => { released.add(slot); },
    waitForSubmittedWork: async () => {},
  } as Record<string, unknown>, {
    // every other method is a no-op; "then" stays undefined so the engine is not taken for a promise
    get: (target, prop: string) => prop === "then" ? undefined : target[prop] ?? (() => undefined),
  });
  const model = new Model({ avg_window: 2 });
  await mount(model);
  // the upload yields to the browser between frames
  for (let i = 0; i < 4; i++) {
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 10)); });
    await settle();
  }
  expect([...filled].sort()).toEqual([0, 1, 2, 3, FRAMES]);
  const stack = Float32Array.from({ length: FRAMES * SIZE * SIZE }, (_, i) => Math.cos(i / 5));
  act(() => model.set("_offline_float_stack", new DataView(stack.buffer)));
  await settle();
  expect([...released].sort()).toEqual([...filled].sort());
});

it("lays out panels that share one source frame by num_panel_cols, not by width / n_panels", async () => {
  // Show3D(stack, stack, stack) sends one frame of `width` columns for three slots. width / n_panels, the
  // fallback for a concatenated frame, is a third of a panel there, so only num_panel_cols gives the layout.
  await mount(new Model({ n_panels: 3, shared_panel_source: true, num_panel_cols: SIZE, max_cols: 3 }));
  const image = el.querySelector<HTMLCanvasElement>('canvas[data-quantem-scientific-output="show3d-image"]')!;
  expect(image.width / image.height).toBeCloseTo(3, 1);
});

it("leaves a settled multi-panel histogram idle until the frame changes", async () => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  const schedule = vi.spyOn(window, "setTimeout");
  const model = new Model({
    n_panels: 3, width: 3 * SIZE, num_panel_cols: SIZE, max_cols: 3,
    _offline_float_stack: new DataView(new Float32Array(FRAMES * SIZE * 3 * SIZE).fill(1).buffer),
  });
  await mount(model);
  await act(async () => { await vi.advanceTimersByTimeAsync(64); });
  const histogramSchedules = () => schedule.mock.calls.filter(([, delay]) => delay === 32).length;
  const settledSchedules = histogramSchedules();
  expect(settledSchedules).toBeGreaterThan(0);
  await act(async () => { await vi.advanceTimersByTimeAsync(96); });
  expect(histogramSchedules()).toBe(settledSchedules);
  act(() => model.set("slice_idx", 1));
  await settle();
  expect(histogramSchedules()).toBeGreaterThan(settledSchedules);
});

it("reorders a panel among its own widget's panels when another Show3D on the page is also reordering", async () => {
  // Two three-panel widgets, both in reorder mode. Panel 1 of the first is dropped in a gap below its row,
  // nearest its own panel 3 but nearer still the second widget's panel 1.
  const packed = (frames: number) => new DataView(new Float32Array(frames * SIZE * 3 * SIZE).fill(1).buffer);
  const options = { n_panels: 3, width: 3 * SIZE, num_panel_cols: SIZE, _offline_float_stack: packed(FRAMES), max_cols: 3 };
  const first = new Model(options);
  const second = new Model(options);
  const secondEl = document.createElement("div");
  document.body.append(secondEl);
  await mount(first);
  // num_panel_cols is the image columns of one panel: three square panels in one row
  const image = el.querySelector<HTMLCanvasElement>('canvas[data-quantem-scientific-output="show3d-image"]')!;
  expect(image.width / image.height).toBeCloseTo(3, 1);
  act(() => {
    const unmount = renderShow3D({ model: second, el: secondEl } as never) as () => void;
    unmountOthers.push(() => { unmount(); secondEl.remove(); delete (document as unknown as Record<string, unknown>).elementsFromPoint; });
  });
  await settle();
  for (const root of [el, secondEl]) act(() => (root.querySelector('button[aria-label="Reorder panels"]') as HTMLElement).click());
  await settle();
  const panels = (root: Element) => [...root.querySelectorAll<HTMLElement>("[data-show3d-reorder-panel]")];
  const place = (panel: HTMLElement, left: number, top: number) => vi.spyOn(panel, "getBoundingClientRect")
    .mockReturnValue({ left, top, width: 100, height: 100, right: left + 100, bottom: top + 100 } as DOMRect);
  panels(el).forEach((panel, slot) => place(panel, 110 * slot, 0));
  panels(secondEl).forEach((panel, slot) => place(panel, 330 + 110 * slot, 300));
  // the pointer is over neither widget's panels (jsdom has no hit testing)
  Object.assign(document, { elementsFromPoint: () => [] });
  const dragged = panels(el)[0];
  const pointer = (type: string, clientX: number, clientY: number) => act(() => {
    dragged.dispatchEvent(new PointerEvent(type, { bubbles: true, cancelable: true, clientX, clientY, pointerId: 1, button: 0 }));
  });
  first.writes = [];
  pointer("pointerdown", 50, 50);
  pointer("pointermove", 330, 290);
  pointer("pointerup", 330, 290);
  await settle();
  expect(first.written("panel_order")).toEqual([[1, 2, 0]]);
});
