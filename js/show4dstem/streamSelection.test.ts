import { expect, test } from "vitest";
import { createElement } from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import ts from "typescript";
import { comparisonImageView, comparisonPanelKey, interactionContext, selectStreamFrame } from "./streamSelection";
import { detectorChannel } from "./detectorChannel";
// @ts-expect-error Vitest raw source import is not part of the widget bundle.
import source from "./index.tsx?raw";

const tree = ts.createSourceFile("index.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

function findNode<T extends ts.Node>(match: (node: ts.Node) => node is T): T {
  let found: T | undefined;
  const visit = (node: ts.Node) => { if (match(node)) found = node; else ts.forEachChild(node, visit); };
  visit(tree);
  if (!found) throw new Error("Missing production interaction callback");
  return found;
}

test("a queued selection names the rendered source, not its reused index", () => {
  const state: Record<string, unknown> = {frame_idx: 0, _stream_source_ids: ["B", "C", "D"], _stream_selected_source: "B"};
  const saved: object[] = [];
  const model = {get: (key: string) => state[key], set: (key: string, value: unknown) => {state[key] = value;},
    save_changes: () => saved.push({...state})};
  let fallback = 0;
  selectStreamFrame(model, 2, ["A", "B", "C"], () => fallback++);
  expect(state._stream_selected_source).toBe("C");
  expect(state.frame_idx).toBe(0); // No stale numeric trait is sent.
  expect(interactionContext(model, "frame_idx")).toEqual(["C"]);
  expect(saved).toHaveLength(1);
  selectStreamFrame(model, 1, [], () => fallback++);
  expect(fallback).toBe(1); // Static widgets retain their existing selection API.
  state.offline = true;
  selectStreamFrame(model, 1, ["B", "C", "D"], () => fallback++);
  expect(fallback).toBe(2); // Exports of a live viewer have no kernel to resolve identities.
});

test("the production tile key preserves its drag-owning DOM node across prepend and eviction", () => {
  const attribute = findNode((node): node is ts.JsxAttribute => ts.isJsxAttribute(node)
    && node.name.getText(tree) === "key" && node.getText(tree).includes("comparisonPanelKey"));
  const expression = (attribute.initializer as ts.JsxExpression).expression!.getText(tree);
  const key = new Function("comparisonPanelKey", "frame", "sourceIds", `return ${expression};`);
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  const show = (ids: string[]) => flushSync(() => root.render(createElement("div", {},
    ids.map((id, frame) => ({id, frame})).reverse().map(({id, frame}) => createElement("input", {
      key: key(comparisonPanelKey, frame, ids), "data-source": id, defaultValue: id,
    })) )));
  try {
    show(["A", "B"]);
    const owner = host.querySelector<HTMLInputElement>('[data-source="B"]')!;
    owner.value = "held pointer owner";
    show(["A", "B", "C"]);
    expect(host.querySelector('[data-source="B"]')).toBe(owner);
    show(["B", "C", "D"]);
    expect(host.querySelector('[data-source="B"]')).toBe(owner);
    expect(owner.value).toBe("held pointer owner");
  } finally { flushSync(() => root.unmount()); host.remove(); }
});

test("a rejected position releases the optimistic cursor so the identical pixel can be retried", () => {
  const declaration = findNode((node): node is ts.VariableDeclaration => ts.isVariableDeclaration(node)
    && node.name.getText(tree) === "positionRequests");
  const memo = declaration.initializer as ts.CallExpression;
  const factory = memo.arguments[0] as ts.ArrowFunction;
  const callback = (factory.body as ts.CallExpression).arguments[2].getText(tree);
  const optimistic = {current: [1, 2] as number[] | null};
  const current = {current: [1, 2]};
  const local: number[] = [];
  const settled = new Function("model", "scanPositionOptimisticRef", "scanPositionCurrentRef", "setLocalPosRow", "setLocalPosCol",
    `return (${callback});`)({get: () => 0}, optimistic, current, (row: number) => local[0] = row, (col: number) => local[1] = col);
  const sent: any[] = [];
  const channel = detectorChannel("one", request => sent.push(request), settled, "show4dstem_position_receipt", 2);
  channel.request(1, 2);
  channel.receive({...sent[0], type: "show4dstem_position_receipt", status: "rejected"});
  expect(optimistic.current).toBeNull();
  expect(current.current).toEqual([0, 0]);
  expect(local).toEqual([0, 0]);
  channel.request(1, 2);
  expect(sent.map(request => request.center)).toEqual([[1, 2], [1, 2]]);
});


test("FFT reads the selected current comparison panel, not the single-image stream placeholder", () => {
  const declaration = findNode((node): node is ts.VariableDeclaration => ts.isVariableDeclaration(node)
    && node.name.getText(tree) === "displayedVirtualImageBytes");
  const memo = declaration.initializer as ts.CallExpression;
  const factory = memo.arguments[0] as ts.ArrowFunction;
  const select = new Function("comparisonImageView", "compareMode", "displayedCompareVirtualImageBytes",
    "comparePanelIndices", "frameIdx", "shapeRows", "shapeCols", "frontendVirtualImageBytes",
    "virtualImageBytes", "activeViSource", `return (${factory.getText(tree)})();`);
  const placeholder = new DataView(new Float32Array([NaN, NaN]).buffer);
  const original = new DataView(new Float32Array([99, 3, 4, 1, 2, 99]).buffer, 4, 16);
  const read = (stack: DataView, indices: number[], frame: number, multiple = true) => select(
    comparisonImageView, multiple, stack, indices, frame, 1, 2, null, placeholder, "roi",
  ) as DataView;
  const selected = read(original, [1, 0], 0);
  expect([...new Float32Array(selected.buffer, selected.byteOffset, 2)]).toEqual([1, 2]);
  expect(selected.buffer).toBe(original.buffer);
  const updated = new DataView(new Float32Array([5, 6, 7, 8]).buffer);
  const next = read(updated, [0, 1], 0);
  expect([...new Float32Array(next.buffer, next.byteOffset, 2)]).toEqual([5, 6]);
  expect(read(updated, [0, 1], 0, false)).toBe(placeholder);
  expect(comparisonImageView(updated, [0, 1], 2, 1, 2)).toBeNull();
  expect(comparisonImageView(new DataView(new ArrayBuffer(0)), [0], 0, 1, 2)).toBeNull();
});
