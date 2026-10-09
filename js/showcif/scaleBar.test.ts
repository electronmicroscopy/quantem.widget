// ShowCIF mounted in jsdom, which has no WebGPU: the panels show the widget
// background instead of the dark WebGPU scene, so the scale bars must be drawn
// in the theme text colour, not in white on a white page. With a WebGPU device
// (stubbed here) the potential maps paint vacuum in the colormap's first
// colour, near white for twilight, so their bars must contrast with that
// colour on every colormap the widget offers.
import { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { COLORMAP_POINTS } from "../display/colormaps";

// GPU renderers whose every method is a no-op: only the overlays are under test.
vi.mock("./render", async (importOriginal) => {
  const original = await importOriginal<typeof import("./render")>();
  class NoOp { constructor() { return new Proxy(this, { get: () => () => undefined }); } }
  return { ...original, AtomRenderer: NoOp };
});
vi.mock("./potential", async (importOriginal) => {
  const original = await importOriginal<typeof import("./potential")>();
  class NoOp { constructor() { return new Proxy(this, { get: () => () => undefined }); } }
  return { ...original, PotentialGPU: NoOp };
});

const { default: showcif } = await import("./index");

// The Python trait's colormap choices (showcif.py potential_colormap).
const POTENTIAL_COLORMAPS = ["inferno", "viridis", "plasma", "magma", "magenta", "hot", "gray", "hsv", "turbo",
  "cividis", "RdBu", "RdBu_r", "seismic", "twilight", "twilight_shifted"];

/** WCAG contrast ratio of two sRGB colours (0 to 255 per channel). */
function contrast(a: number[], b: number[]): number {
  const luminance = (rgb: number[]) => {
    const [r, g, bl] = rgb.map(c => (c / 255 <= 0.04045 ? c / 255 / 12.92 : ((c / 255 + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * bl;
  };
  const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (high + 0.05) / (low + 0.05);
}

const UNIT_ATOMS = new Float32Array([
  0, 0, 0, 2,
  2, 2, 2, 1,
  2, 2, 0, 0,
  2, 0, 2, 0,
  0, 2, 2, 0,
]);

class Model {
  values: Record<string, unknown> = {
    title: "BaTiO3 · Atomic structure", unit_atom_bytes: new DataView(UNIT_ATOMS.buffer),
    potential_bytes: new DataView(new ArrayBuffer(0)),
    unit_cell: [[4, 0, 0], [0, 4, 0], [0, 0, 4]], repeats: [2, 2, 2], zone_axis: [0, 0, 1],
    species: [
      { symbol: "O", number: 8, color: [1, 0.051, 0.051], count: 24 },
      { symbol: "Ti", number: 22, color: [0.749, 0.761, 0.78], count: 8 },
      { symbol: "Ba", number: 56, color: [0, 0.788, 0], count: 8 },
    ],
    visible_species: [true, true, true], specimen_tilt_mrad: [0, 0], view_mode: "unit_cells",
    field_of_view_A: 40, magnification_calibration: [], orthogonal_views: false, num_slices: 16,
    energy_keV: 300, potential_colormap: "viridis", potential_quantity: "average", potential_pixels: 256,
    potential_sigma_A: 0.08, source_summary: "ASE structure · 5 atoms/cell · lengths in Å",
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
let cleanup: (() => void) | undefined;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  document.body.dataset.jpThemeLight = "true";
  el = document.createElement("div");
  document.body.append(el);
});
afterEach(async () => {
  act(() => cleanup?.());
  cleanup = undefined;
  await act(async () => { for (let i = 0; i < 10; i++) await Promise.resolve(); });
  el.remove();
  delete document.body.dataset.jpThemeLight;
  vi.unstubAllGlobals();
});

it("draws the scale bars in the theme text colour when no WebGPU scene is behind them", async () => {
  await act(async () => { cleanup = showcif.render({ model: new Model(), el } as never) as (() => void) | undefined; });
  await act(async () => { for (let i = 0; i < 10; i++) await Promise.resolve(); });
  expect(el.querySelector('[data-render-path="none"]')).not.toBeNull();
  const bars = Array.from(el.querySelectorAll<HTMLElement>("[data-scale-bar]"));
  expect(bars.length).toBeGreaterThanOrEqual(2);
  for (const bar of bars) {
    expect(bar.style.color).toBe("var(--cif-text)");
    expect(bar.style.borderBottom).toContain("var(--cif-text)");
  }
  // The variable resolves to the light theme's text colour, which contrasts with its background.
  const root = el.querySelector<HTMLElement>(".qcif")!;
  expect(root.style.getPropertyValue("--cif-text")).not.toBe(root.style.getPropertyValue("--cif-bg"));
});

it("draws the potential-map scale bars in the colour that contrasts with vacuum on every colormap", async () => {
  const device = { addEventListener() {}, destroy() {} };
  Object.defineProperty(navigator, "gpu", {
    configurable: true,
    value: { requestAdapter: async () => ({ requestDevice: async () => device }), getPreferredCanvasFormat: () => "rgba8unorm" },
  });
  try {
    const model = new Model();
    model.values.potential_bytes = new DataView(new Float32Array(3 * 64).fill(1).buffer);
    model.values.potential_colormap = "twilight";
    await act(async () => { cleanup = showcif.render({ model, el } as never) as (() => void) | undefined; });
    await act(async () => { for (let i = 0; i < 10; i++) await Promise.resolve(); });
    const mapBars = () => Array.from(el.querySelectorAll<HTMLElement>(".potential-panel [data-scale-bar]"));
    expect(mapBars().length).toBeGreaterThanOrEqual(2);
    for (const bar of mapBars()) expect(bar.style.color).toBe("black");  // twilight's vacuum is near white
    for (const name of POTENTIAL_COLORMAPS) {
      await act(async () => { model.set("potential_colormap", name); });
      const vacuum = COLORMAP_POINTS[name][0];
      for (const bar of mapBars()) {
        const ink = bar.style.color === "black" ? [0, 0, 0] : [255, 255, 255];
        expect(contrast(ink, vacuum), `${name}: ${bar.style.color} on vacuum ${vacuum}`).toBeGreaterThanOrEqual(4.5);
        expect(bar.style.borderBottom).toContain(bar.style.color);
        // The halo is the other colour, so the bar also reads over atoms at the colormap's far end.
        expect(bar.style.filter).toContain(bar.style.color === "black" ? "white" : "black");
      }
    }
    const sceneBars = Array.from(el.querySelectorAll<HTMLElement>("[data-scale-bar]")).filter(bar => !bar.closest(".potential-panel"));
    expect(sceneBars.length).toBeGreaterThanOrEqual(2);
    for (const bar of sceneBars) {
      expect(bar.style.color).toBe("white");  // on the near-black atom scene
      expect(bar.style.filter).toContain("black");
    }
  } finally {
    delete (navigator as unknown as { gpu?: unknown }).gpu;
  }
});
