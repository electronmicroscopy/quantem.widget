import { describe, expect, it } from "vitest";

import { COLORMAPS } from "../display/colormaps";
import { signedLog1p } from "../display/stats";
import { frameKeyTarget, renderFramePlayback, renderFrameScaledPlayback, renderPackedPanelPlayback, writeLiveFrameControls } from "./playback";

describe("playback keys", () => {
  it("a held ArrowRight steps one frame per press from the frame on screen", () => {
    // 30 key repeats in 1 s: each press starts where the last one drew, so the hold advances 30 frames
    let shown = 0;
    for (let press = 0; press < 30; press++) shown = frameKeyTarget("ArrowRight", shown, 100, null)!;
    expect(shown).toBe(30);
  });

  it("stops at the stack ends, or at the loop range while Loop is on", () => {
    expect(frameKeyTarget("ArrowRight", 99, 100, null)).toBe(99);
    expect(frameKeyTarget("ArrowLeft", 0, 100, null)).toBe(0);
    expect(frameKeyTarget("ArrowRight", 40, 100, [10, 40])).toBe(40);
    expect(frameKeyTarget("ArrowLeft", 10, 100, [10, 40])).toBe(10);
    expect(frameKeyTarget("Home", 25, 100, [10, 40])).toBe(10);
    expect(frameKeyTarget("End", 25, 100, [10, 40])).toBe(40);
    expect(frameKeyTarget("End", 25, 30, [10, 40])).toBe(29);
    expect(frameKeyTarget("Home", 25, 100, null)).toBe(0);
    expect(frameKeyTarget("End", 25, 100, null)).toBe(99);
  });

  it("leaves other keys to the rest of the shortcut handler", () => {
    expect(frameKeyTarget(" ", 5, 100, null)).toBeNull();
    expect(frameKeyTarget("r", 5, 100, null)).toBeNull();
  });
});

describe("live frame controls", () => {
  function widget(withCounter: boolean): { root: HTMLElement; slider: HTMLElement } {
    const root = document.createElement("div");
    root.innerHTML = `<span class="slider"><span class="MuiSlider-track"></span><span class="MuiSlider-thumb"><input type="range"></span></span>`
      + (withCounter ? `<span data-show3d-playback-count="true">5/20</span>` : "");
    document.body.appendChild(root);
    return { root, slider: root.querySelector(".slider") as HTMLElement };
  }

  it("write only their own widget, even when it shows no frame counter", () => {
    // two Show3D on one page: A's counter is not rendered, B's reads 5/20
    document.body.innerHTML = "";
    const a = widget(false);
    const b = widget(true);
    writeLiveFrameControls(a.root, a.slider, null, 3, 20, false);
    expect(b.root.querySelector("[data-show3d-playback-count]")!.textContent).toBe("5/20");
    expect(a.slider.querySelector(".MuiSlider-thumb")!.getAttribute("aria-valuenow")).toBe("3");
    writeLiveFrameControls(b.root, b.slider, null, 7, 20, false);
    expect(b.root.querySelector("[data-show3d-playback-count]")!.textContent).toBe("8/20");
  });
});

describe("CPU playback colormap", () => {
  // Values across and beyond the window, its exact ends, NaN and infinities, and a deterministic spread.
  const WIDTH = 64;
  const HEIGHT = 48;
  let state = 7;
  const random = () => (state = (state * 1103515245 + 12345) >>> 0) / 2 ** 32;
  const data = Float32Array.from({ length: WIDTH * HEIGHT }, (_, i) => [Number.NaN, Infinity, -Infinity, -2, 3, 9][i % 97] ?? -4 + 16 * random());
  const lut = COLORMAPS.inferno;

  /** The playback formula: signed log1p, then the float64 floor of (v - vmin) * 255 / span, clamped; NaN takes entry 0. */
  function reference(values: number[], vmin: number, vmax: number, logScale: boolean): Uint8ClampedArray {
    const rgba = new Uint8ClampedArray(values.length * 4);
    const invRange = vmax > vmin ? 255 / (vmax - vmin) : 0;
    values.forEach((raw, k) => {
      const value = logScale ? signedLog1p(raw) : raw;
      const entry = value <= vmin ? 0 : value >= vmax ? 255 : Number.isNaN(value) ? 0 : Math.floor((value - vmin) * invRange);
      rgba.set([lut[3 * entry], lut[3 * entry + 1], lut[3 * entry + 2], 255], 4 * k);
    });
    return rgba;
  }

  for (const [vmin, vmax, logScale] of [[-2, 9, false], [0.3, 2.1, true], [3, 3, false]] as const) {
    it(`draws full, packed and scaled frames with the playback formula (window ${vmin} to ${vmax}, log ${logScale})`, () => {
      const full = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
      renderFramePlayback(data, full, lut, vmin, vmax, logScale);
      expect(full).toEqual(reference(Array.from(data), vmin, vmax, logScale));
      // the third of four 16-column panels
      const panel = new Uint8ClampedArray(16 * HEIGHT * 4);
      renderPackedPanelPlayback(data, WIDTH, 32, 16, HEIGHT, panel, lut, vmin, vmax, logScale);
      const panelValues = Array.from({ length: 16 * HEIGHT }, (_, k) => data[Math.floor(k / 16) * WIDTH + 32 + (k % 16)]);
      expect(panel).toEqual(reference(panelValues, vmin, vmax, logScale));
      // every third column of every second row
      const xMap = Uint32Array.from({ length: WIDTH / 4 }, (_, x) => 3 * x);
      const yMap = Uint32Array.from({ length: HEIGHT / 2 }, (_, y) => 2 * y * WIDTH);
      const scaled = new Uint8ClampedArray(xMap.length * yMap.length * 4);
      renderFrameScaledPlayback(data, scaled, xMap, yMap, xMap.length, yMap.length, lut, vmin, vmax, logScale);
      const scaledValues = Array.from({ length: xMap.length * yMap.length }, (_, k) => data[yMap[Math.floor(k / xMap.length)] + xMap[k % xMap.length]]);
      expect(scaled).toEqual(reference(scaledValues, vmin, vmax, logScale));
    });
  }
});
