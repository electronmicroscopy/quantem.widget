// Show1D's Reset restores every interactive trait to its value when the widget
// first settled, and is enabled exactly while one of them differs.
import { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import show1d from "./index";

// The interactive traits at their Python defaults, plus a small plot.
const INITIAL: Record<string, unknown> = {
  y_bytes: new DataView(new Float32Array([1, 3, 2, 5, 2, 1, 4, 3]).buffer),
  x_bytes: new DataView(new ArrayBuffer(0)),
  n_traces: 2, n_points: 4, labels: ["particle", "support"], colors: ["#0072B2", "#D55E00"],
  show_grid: true, show_controls: true, line_width: 1.5, x_label: "angle", x_unit: "mrad", y_label: "I", y_unit: "",
  controls_collapsed: false, log_scale: false, show_stats: false, show_review: false, show_legend: true,
  starred_snapshot_image_labels: [], trial_notes: {}, trial_tags: {}, show_trial_notes: false, show_starred_only: false,
  trial_sort_key: "label", trial_sort_descending: false, trial_filter_text: "", top_trial_count: 0,
  plot_height_px: 320, plot_width_px: 400, side_panel_width_px: 360, snapshot_panel_width_px: 0, focused_trace: -1,
  x_range: [], y_range: [], selected_snapshot_idx: -1, selected_snapshot_group_idx: -1, bookmarked_snapshot_groups: [],
  hidden_snapshot_image_labels: [], show_snapshot_histogram: true, show_snapshot_fft: false, snapshot_fft_layout: "overlay",
  snapshot_fft_window: true, snapshot_fft_cmap: "magma", snapshot_contrast_preset: "full", snapshot_panel_contrast_ranges: {},
  snapshot_contrast_range: [], snapshot_thumbnail_size: 48, snapshot_overlay_position: "top-right", image_cmap: "viridis",
  snapshot_real_space_zoom: 1, snapshot_real_space_center: [], snapshot_fft_zoom: 1, snapshot_fft_center: [],
  show_snapshot_profile: false, snapshot_profile_line: [], snapshot_playing: false, snapshot_fps: 2, snapshot_loop: true,
  snapshot_bounce: false,
};

class Model {
  values: Record<string, unknown> = { ...INITIAL };
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

async function settle() {
  await act(async () => {
    for (let i = 0; i < 12; i++) await Promise.resolve();
    await new Promise(resolve => setTimeout(resolve, 5));
  });
}
function resetButton(): HTMLButtonElement {
  return Array.from(el.querySelectorAll("button")).find(button => button.textContent?.trim() === "Reset")!;
}

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  const context = new Proxy({}, { get: (_target, prop) => prop === "measureText" ? () => ({ width: 6 }) : () => undefined, set: () => true });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => context as never);
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

it("restores the settled values and is enabled only while one differs", async () => {
  act(() => { cleanup = show1d.render({ model, el } as never) as (() => void) | undefined; });
  await settle();
  expect(resetButton().disabled).toBe(true);
  act(() => {
    model.set("log_scale", true);
    model.set("x_range", [1, 2]);
    model.set("trial_notes", { particle: "drifted" });
    model.set("image_cmap", "magma");
    model.set("snapshot_profile_line", [{ row: 1, col: 2 }, { row: 3, col: 4 }]);
  });
  await settle();
  expect(resetButton().disabled).toBe(false);
  act(() => resetButton().click());
  await settle();
  for (const key of ["log_scale", "x_range", "trial_notes", "image_cmap", "snapshot_profile_line"]) {
    expect(model.get(key), key).toEqual(INITIAL[key]);
  }
  expect(model.save_changes).toHaveBeenCalled();
  expect(resetButton().disabled).toBe(true);
});
