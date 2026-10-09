// @ts-expect-error Vitest raw source import is not part of the widget bundle.
import source from "./index.tsx?raw";
import ts from "typescript";
import {describe, expect, it} from "vitest";

// Exercise the mounted grid's actual tile-selection callback: a GPU tile is
// admitted only once its image can be displayed.
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
function tiles(ready: number, rangesReady = 0) {
  const bindings = {renderIndices: [0, 1, 2], panelByFrame: new Map(),
    gpuSlots: new Map(Array.from({length: ready}, (_, i) => [i, 60 + i])),
    gpuRanges: new Map(Array.from({length: rangesReady}, (_, i) => [i, {min: 0, max: 1}])),
    gpuEngine: {}, progressivePage: null};
  return new Function(...Object.keys(bindings), `${code};return select();`)(...Object.values(bindings));
}

describe("compare tile admission", () => {
  it("keeps ordinary HDF5 tiles pending until their display range is ready", () => {
    expect(tiles(1, 0)).toHaveLength(0);
    expect(tiles(1, 1).map((entry: {frame: number}) => entry.frame)).toEqual([0]);
  });
});
