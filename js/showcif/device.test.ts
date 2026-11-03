// ShowCIF draws with the page's one hardware WebGPU device (display/device),
// which every other widget on the page shares: mounting uses it without asking
// navigator.gpu for an adapter of its own, and unmounting releases only what
// ShowCIF made, never the shared device itself.
import { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

const sharedDevice = vi.hoisted(() => ({
  destroy: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
}));
vi.mock("../display/device", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../display/device")>()),
  getHardwareGPUDevice: async () => sharedDevice,
}));
// GPU renderers whose every method is a no-op: only the device lifetime is under test.
vi.mock("./render", async (importOriginal) => {
  const original = await importOriginal<typeof import("./render")>();
  class NoOp { constructor() { return new Proxy(this, { get: () => () => undefined }); } }
  return { ...original, AtomRenderer: NoOp };
});

const { default: showcif } = await import("./index");

const UNIT_ATOMS = new Float32Array([0, 0, 0, 1, 2, 2, 2, 0]);

class Model {
  values: Record<string, unknown> = {
    title: "BaTiO3 · Atomic structure", unit_atom_bytes: new DataView(UNIT_ATOMS.buffer),
    potential_bytes: new DataView(new ArrayBuffer(0)),
    unit_cell: [[4, 0, 0], [0, 4, 0], [0, 0, 4]], repeats: [2, 2, 2], zone_axis: [0, 0, 1],
    species: [
      { symbol: "O", number: 8, color: [1, 0.051, 0.051], count: 8 },
      { symbol: "Ba", number: 56, color: [0, 0.788, 0], count: 8 },
    ],
    visible_species: [true, true], specimen_tilt_mrad: [0, 0], view_mode: "unit_cells",
    field_of_view_A: 40, magnification_calibration: [], orthogonal_views: false, num_slices: 16,
    energy_keV: 300, potential_colormap: "viridis", potential_quantity: "average", potential_pixels: 256,
    potential_sigma_A: 0.08, source_summary: "ASE structure · 2 atoms/cell · lengths in Å",
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

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  el = document.createElement("div");
  document.body.append(el);
});
afterEach(() => {
  el.remove();
  vi.unstubAllGlobals();
});

it("draws with the shared device and leaves it alive on unmount", async () => {
  let cleanup: (() => void) | undefined;
  await act(async () => { cleanup = showcif.render({ model: new Model(), el } as never) as (() => void) | undefined; });
  await act(async () => { for (let i = 0; i < 10; i++) await Promise.resolve(); });
  expect(el.querySelector('[data-render-path="none"]')).toBeNull();
  expect(el.textContent).toContain("WebGPU · exact atomic positions");
  const [event, handler] = sharedDevice.addEventListener.mock.calls[0];
  expect(event).toBe("uncapturederror");
  act(() => cleanup?.());
  await act(async () => { for (let i = 0; i < 10; i++) await Promise.resolve(); });
  expect(sharedDevice.destroy).not.toHaveBeenCalled();
  expect(sharedDevice.removeEventListener).toHaveBeenCalledWith("uncapturederror", handler);
});
