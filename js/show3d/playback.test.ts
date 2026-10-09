import { describe, expect, it } from "vitest";

import { frameKeyTarget, writeLiveFrameControls } from "./playback";

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
