// @ts-expect-error Vitest raw source import is not part of the widget bundle.
import source from "./index.tsx?raw";
import ts from "typescript";
import {expect, it} from "vitest";

// Exercise the real offline preset handler. Saved snapshots represent the
// geometry read by a live kernel after selecting a detector in the browser.
const tree = ts.createSourceFile("index.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let callback = "";
function visit(node: ts.Node) {
  if (ts.isVariableDeclaration(node) && node.name.getText(tree) === "onPreset") {
    callback = node.initializer!.getText(tree);
  }
  ts.forEachChild(node, visit);
}
visit(tree);
const code = ts.transpileModule(`const onPreset = ${callback};`, {
  compilerOptions: {target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.None},
}).outputText;

it("saves the displayed ADF then BF detector geometry for Python readback", () => {
  const state: Record<string, unknown> = {
    det_rows: 32, det_cols: 32,
    bright_field_radius_px: 6, bright_field_center_row_px: 11.5, bright_field_center_col_px: 12.5, roi_mode: "circle",
    roi_radius: 6, roi_radius_inner: 0, vi_source: "roi",
  };
  const saved: Record<string, unknown>[] = [];
  const displayed: Record<string, unknown>[] = [];
  const model = {
    get(key: string) {return state[key];},
    set(key: string, value: unknown) {state[key] = value;},
    save_changes() {saved.push({...state});},
  };
  const recomputeVI = () => {displayed.push({...state});};
  const onPreset = new Function("model", "recomputeVI", "recomputeCompareVI",
    `let suppressViTraitRecompute = false; ${code}; return onPreset;`)(model, recomputeVI, () => {});

  for (const preset of ["adf", "bf", "bf"]) {
    state._preset_request = preset;
    onPreset();
  }
  expect(saved).toHaveLength(3);
  expect(saved).toEqual(displayed);
  expect(saved[0]).toMatchObject({
    roi_mode: "annular", roi_radius_inner: 6, roi_radius: 12,
    roi_center_row: 11.5, roi_center_col: 12.5, roi_active: true,
    vi_source: "roi", _preset_request: "",
  });
  expect(saved[1]).toMatchObject({roi_mode: "circle", roi_radius_inner: 0, roi_radius: 6});
  expect(saved[2]).toEqual(saved[1]);
});

it("stops an annular preset at the detector edge and refuses one past it, as Python does", () => {
  const state: Record<string, unknown> = {
    det_rows: 24, det_cols: 24, bright_field_radius_px: 7, bright_field_center_row_px: 11.5, bright_field_center_col_px: 11.5,
    roi_mode: "circle", roi_radius: 7, roi_radius_inner: 0, vi_source: "roi",
  };
  const model = {
    get(key: string) {return state[key];},
    set(key: string, value: unknown) {state[key] = value;},
    save_changes() {},
  };
  const warn = console.warn;
  const warnings: string[] = [];
  console.warn = (message: string) => {warnings.push(message);};
  try {
    const onPreset = new Function("model", "recomputeVI", "recomputeCompareVI",
      `let suppressViTraitRecompute = false; ${code}; return onPreset;`)(model, () => {}, () => {});
    state._preset_request = "adf";
    onPreset();
    expect(state).toMatchObject({roi_mode: "annular", roi_radius_inner: 7, roi_radius: 12});  // the edge, not 14
    state._preset_request = "haadf";
    onPreset();
    expect(state).toMatchObject({roi_mode: "annular", roi_radius_inner: 7, roi_radius: 12});  // unchanged
    expect(warnings.join(" ")).toContain("HAADF starts 14.0 px from the bright-field centre; this detector reaches 12.0 px");
  } finally {
    console.warn = warn;
  }
});
