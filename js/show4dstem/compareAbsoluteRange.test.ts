// @ts-expect-error Vitest raw source import is not part of the widget bundle.
import source from "./index.tsx?raw";
import ts from "typescript";
import {describe, expect, it} from "vitest";
import {applyColormap, displayLog, displayNormalize} from "../display/displayMath";
import {COLORMAPS} from "../display/luts";
import {findDataRange, percentileClip, sliderRange} from "../display/stats";

// Run the actual mounted grid's rendering callback with a recording renderer.
// No physical GPU is needed to check which limits every signed tile receives.
const tree = ts.createSourceFile("index.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let callback = "";
let canvasCallback = "";
const props: string[] = [];
function visit(node: ts.Node) {
  if (ts.isVariableDeclaration(node) && node.name.getText(tree) === "renderGpuSlotsNow"
      && node.initializer && ts.isCallExpression(node.initializer)) {
    callback = node.initializer.arguments[0].getText(tree);
  }
  if (ts.isCallExpression(node) && node.expression.getText(tree) === "React.useEffect"
      && node.arguments[0]?.getText(tree).includes("const styleKey =")) {
    canvasCallback = node.arguments[0].getText(tree);
  }
  if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(tree) === "CompareVirtualGrid") {
    props.push(node.attributes.getText(tree));
  }
  ts.forEachChild(node, visit);
}
visit(tree);
const code = ts.transpileModule(`const render = ${callback};`, {
  compilerOptions: {target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.None},
}).outputText;
function renderedRanges(absoluteMin: number | null, absoluteMax: number | null, scaleMode = "linear", gpuBytesCurrent = false) {
  const ranges: {vmin: number; vmax: number}[] = [];
  const bindings = {
    absoluteMin, absoluteMax, displayLog, colormap: "RdBu_r", COLORMAPS, gpuBytesCurrent,
    gpuEngine: {uploadLUT(_name: string, lut: Uint8Array) {
      expect(lut).toBe(COLORMAPS.RdBu_r);
      expect(lut.length).toBe(256 * 3);
    }, configureCanvas() {
      return {unconfigure() {}};
    }, renderPanelSlotsDirectToCanvas(_slots: unknown, limits: {vmin: number; vmax: number}) {
      ranges.push(limits);
    }, renderSlotsDirectWithGpuRangeToCanvases(slots: number[], contexts: unknown[], low: number, high: number) {
      expect(slots).toEqual([0, 1]);
      expect(contexts).toHaveLength(2);
      // Represent the new dark-field images, far below the previous ranges.
      ranges.push(...[0.001, 0.002].map(max => ({vmin: max * low / 100, vmax: max * high / 100})));
      return slots.length;
    }},
    gpuSlots: new Map([[0, 0], [1, 1]]),
    gpuRanges: new Map([[0, {min: -8, max: 2}], [1, {min: -1, max: 3}]]),
    gpuCanvasRefs: {current: [{}, {}]}, gpuContextsRef: {current: new Map()},
    autoContrast: false,
    renderEntries: [{frame: 0}, {frame: 1}], scaleMode, vminPct: 0, vmaxPct: 100,
    shapeRows: 2, shapeCols: 2, smooth: false,
  };
  new Function(...Object.keys(bindings), `${code};return render();`)(...Object.values(bindings));
  return ranges;
}

describe("signed comparison limits", () => {
  it("uses the arriving image's GPU range during a bright-to-dark detector drag", () => {
    expect(renderedRanges(null, null, "linear", true)).toEqual([
      {vmin: 0, vmax: 0.001}, {vmin: 0, vmax: 0.002},
    ]);
    expect(renderedRanges(-10, 10, "linear", true)).toEqual([
      {vmin: -10, vmax: 10}, {vmin: -10, vmax: 10},
    ]);
  });
  it("sends one symmetric range to every tile and keeps zero at the midpoint", () => {
    const ranges = renderedRanges(-10, 10);
    expect(ranges).toEqual([{vmin: -10, vmax: 10}, {vmin: -10, vmax: 10}]);
    for (const range of ranges) {
      expect(displayNormalize(0, range.vmin, range.vmax)).toBe(0.5);
      expect(displayNormalize(-5, range.vmin, range.vmax)).toBe(0.25);
    }
  });
  it("keeps signed Canvas2D pixels on the same absolute scale even with auto contrast enabled", () => {
    let pixels: Uint8ClampedArray = new Uint8ClampedArray();
    const context = {
      createImageData() {return {data: new Uint8ClampedArray(16)};},
      putImageData(image: {data: Uint8ClampedArray}) {pixels = image.data;},
    };
    const lut = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) lut.fill(i, i * 3, i * 3 + 3);
    const bindings = {
      absoluteMin: -10, absoluteMax: 10, autoContrast: true, colormap: "gray", COLORMAPS: {gray: lut},
      gpuEngine: null, gpuSlots: null, scaleMode: "linear", vminPct: 0, vmaxPct: 100,
      smooth: false, shapeRows: 2, shapeCols: 2, canvasDrawCacheRef: {current: new Map()},
      renderEntries: [{frame: 0, panel: new Float32Array([-8, 0, 2, 10])}],
      canvasRefs: {current: [{width: 2, height: 2, getContext() {return context;}}]},
      findDataRange, percentileClip, sliderRange, applyColormap,
    };
    const canvasCode = ts.transpileModule(`const render = ${canvasCallback};`, {
      compilerOptions: {target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.None},
    }).outputText;
    new Function(...Object.keys(bindings), `${canvasCode};render();`)(...Object.values(bindings));
    expect([pixels[0], pixels[4], pixels[8], pixels[12]]).toEqual([25, 127, 153, 255]);
  });
  it("preserves the current signed-log transform for fixed bounds", () => {
    const ranges = renderedRanges(-10, 10, "log");
    expect(ranges).toEqual([
      {vmin: displayLog(-10), vmax: displayLog(10)},
      {vmin: displayLog(-10), vmax: displayLog(10)},
    ]);
    expect(displayNormalize(0, ranges[0].vmin, ranges[0].vmax)).toBe(0.5);
  });
  it("retains data-derived ranges when no complete absolute range is supplied", () => {
    expect(renderedRanges(null, null)).toEqual([{vmin: -8, vmax: 2}, {vmin: -1, vmax: 3}]);
  });
  it("wires the existing DP and VI traits into their corresponding grids", () => {
    expect(props).toHaveLength(2);
    expect(props[0]).toContain("absoluteMin={traitDpVmin}");
    expect(props[0]).toContain("absoluteMax={traitDpVmax}");
    expect(props[1]).toContain("absoluteMin={traitViVmin}");
    expect(props[1]).toContain("absoluteMax={traitViVmax}");
  });
});
