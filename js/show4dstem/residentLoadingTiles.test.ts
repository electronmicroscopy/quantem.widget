// @ts-expect-error Vitest raw source import is not part of the widget bundle.
import source from "./index.tsx?raw";
import ts from "typescript";
import {describe, expect, it} from "vitest";

// Exercise the mounted grid's actual tile-selection callback with successive
// acquisition completions; missing data must keep its place during loading.
const tree = ts.createSourceFile("index.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let body = "";
function visit(node: ts.Node) {
  if (ts.isVariableDeclaration(node) && node.name.getText(tree) === "renderEntries"
      && node.initializer && ts.isCallExpression(node.initializer)) {
    body = node.initializer.arguments[0].getText(tree);
  }
  ts.forEachChild(node, visit);
}
visit(tree);
const code = ts.transpileModule(`const select = ${body};`, {
  compilerOptions: {target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.None},
}).outputText;
function tiles(ready: number, loading: boolean, residentSource = true, rangesReady = 0) {
  const bindings = {renderIndices: [0, 1, 2], panelByFrame: new Map(),
    gpuSlots: new Map(Array.from({length: ready}, (_, i) => [i, 60 + i])),
    residentSource, gpuRanges: new Map(Array.from({length: rangesReady}, (_, i) => [i, {min: 0, max: 1}])),
    gpuEngine: {}, integerCounts: false, batchEnabled: false, batchFailed: false,
    sourceLoading: loading, progressivePage: null};
  return new Function(...Object.keys(bindings), `${code};return select();`)(...Object.values(bindings));
}

describe("progressive resident tile positions", () => {
  it("keeps ordinary HDF5 tiles pending until their display range is ready", () => {
    expect(tiles(1, false, false, 0)).toHaveLength(0);
    expect(tiles(1, false, false, 1).map((entry: {frame: number}) => entry.frame)).toEqual([0]);
  });
  it("keeps all requested positions while the first and next images become ready", () => {
    for (const ready of [1, 2, 3]) {
      const entries = tiles(ready, ready < 3);
      expect(entries.map((entry: {frame: number}) => entry.frame)).toEqual([0, 1, 2]);
      expect(entries.filter((entry: {gpuLoaded: boolean}) => entry.gpuLoaded)).toHaveLength(ready);
    }
  });
  it("retains normal filtering when there is no progressive admission", () => {
    expect(tiles(1, false).map((entry: {frame: number}) => entry.frame)).toEqual([0]);
  });
});
