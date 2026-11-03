import * as React from "react";
import {act} from "react";
import {createRoot, type Root} from "react-dom/client";
import {afterEach, beforeEach, expect, it, vi} from "vitest";
import {Histogram, type HistogramProps} from "./Histogram";

let mount: HTMLDivElement;
let root: Root;
const fillRect = vi.fn();
const values = new Float32Array([-8, -1, 0, 2]);

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    setTransform() {}, fillRect,
  } as unknown as ReturnType<HTMLCanvasElement["getContext"]>);
  mount = document.createElement("div");
  document.body.append(mount);
  root = createRoot(mount);
});

afterEach(() => {
  act(() => root.unmount());
  mount.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  fillRect.mockClear();
});

function render(fixedRange?: HistogramProps["fixedRange"]) {
  act(() => root.render(React.createElement(Histogram, {
    data: values,
    dataMin: 0, dataMax: 245, vminPct: 0, vmaxPct: 100,
    onRangeChange: vi.fn(), fixedRange,
  })));
}

it("shows the active signed limits instead of unrelated histogram sliders", () => {
  render([-10, 10]);
  const limits = mount.querySelector('[aria-label="Fixed color limits"]');
  expect(limits?.textContent).toContain("-10.000");
  expect(limits?.textContent).toContain("10.000");
  expect(limits?.textContent).toContain("Set in Python");
  expect(limits?.getAttribute("title")).toContain("percentile controls are inactive");
  expect(mount.querySelector('input[type="range"]')).toBeNull();
  expect(mount.textContent).not.toContain("245");

  render([-0.00125, 0.00125]);
  expect(limits?.textContent).toContain("-1.250e-3");
  expect(limits?.textContent).toContain("1.250e-3");
});

it("restores the normal histogram and percentile handles when absolute limits are cleared", () => {
  render([-10, 10]);
  fillRect.mockClear();
  render();
  expect(mount.querySelector('[aria-label="Fixed color limits"]')).toBeNull();
  expect(mount.querySelectorAll('input[type="range"]')).toHaveLength(2);
  expect(mount.textContent).toContain("245.0");
  expect(fillRect).toHaveBeenCalled();
});
