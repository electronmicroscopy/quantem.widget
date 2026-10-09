// ShowCIF without WebGPU (jsdom has no navigator.gpu): the notice states the
// actual reason once. On a secure page (localhost, HTTPS) WebGPU is off or
// unsupported, so the notice must not send the reader to HTTPS or localhost;
// only a page that is not a secure context gets that instruction.
import { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import showcif from "./index";

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
let cleanup: (() => void) | undefined;

async function notice(): Promise<string> {
  await act(async () => { cleanup = showcif.render({ model: new Model(), el } as never) as (() => void) | undefined; });
  await act(async () => { for (let i = 0; i < 10; i++) await Promise.resolve(); });
  return el.querySelector<HTMLElement>('[data-render-path="none"]')?.textContent ?? "";
}

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  el = document.createElement("div");
  document.body.append(el);
});
afterEach(async () => {
  act(() => cleanup?.());
  cleanup = undefined;
  await act(async () => { for (let i = 0; i < 10; i++) await Promise.resolve(); });
  el.remove();
  vi.unstubAllGlobals();
});

it("names the missing navigator.gpu on a secure page, without sending the reader to HTTPS", async () => {
  vi.stubGlobal("isSecureContext", true);
  const text = await notice();
  expect(text).toContain("Atoms and projections need WebGPU, which this browser does not provide");
  expect(text).toContain("navigator.gpu is missing");
  expect(text).not.toMatch(/HTTPS|localhost/);
  expect(text).not.toMatch(/Open /);  // one statement of the reason, no repeated instruction
});

it("tells a page that is not a secure context to open over HTTPS or localhost", async () => {
  vi.stubGlobal("isSecureContext", false);
  const text = await notice();
  expect(text).toContain("not a secure context");
  expect(text.match(/HTTPS or localhost/g)?.length).toBe(1);
});
