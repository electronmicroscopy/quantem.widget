import { expect, test, vi } from "vitest";
import { detectorChannel, detectorViewId } from "./detectorChannel";
import { interactionContext } from "./streamSelection";
// @ts-expect-error Vitest raw source import is not part of the widget bundle.
import source from "./index.tsx?raw";
import ts from "typescript";

const receipt = (request: object, status: string) => ({ ...request, type: "show4dstem_detector_receipt", status }) as Parameters<ReturnType<typeof detectorChannel>["receive"]>[0];

test("view identifiers also work where secure-context randomUUID is unavailable", () => {
  vi.stubGlobal("crypto", { getRandomValues: (words: Uint32Array) => words.fill(7) });
  try { expect(detectorViewId()).toBe("7-7-7-7"); }
  finally { vi.unstubAllGlobals(); }
});

test("a held drag sends the latest center after acceptance and retains it until completion", () => {
  const sent: { view: string; epoch: number; seq: number; center: number[] }[] = [];
  let settled = 0;
  const channel = detectorChannel("one", request => sent.push(request), () => settled++);
  channel.request(1, 2);
  for (let row = 2; row <= 60; row++) channel.request(row, 3);
  expect(sent.map(request => request.center)).toEqual([[1, 2]]);
  channel.receive(receipt(sent[0], "accepted"));
  expect(sent.map(request => request.center)).toEqual([[1, 2], [60, 3]]);
  channel.receive(receipt(sent[0], "complete"));
  expect(channel.isPending()).toBe(true);
  channel.receive(receipt(sent[1], "accepted"));
  expect(channel.isPending()).toBe(true);
  channel.receive(receipt(sent[1], "complete"));
  expect(channel.isPending()).toBe(false);
  expect(settled).toBe(1);
});

test("cancel discards stale unsent positions without granting an extra in-flight credit", () => {
  const sent: { view: string; epoch: number; seq: number; center: number[] }[] = [];
  const channel = detectorChannel("one", request => sent.push(request), () => {});
  channel.request(1, 2); channel.request(2, 3); channel.clear();
  expect(channel.isPending()).toBe(false);
  channel.request(5, 6);
  expect(sent).toHaveLength(1);
  channel.receive(receipt({ ...sent[0], view: "other" }, "accepted"));
  expect(sent).toHaveLength(1);
  channel.receive(receipt(sent[0], "accepted"));
  expect(sent[1].center).toEqual([5, 6]);
  channel.request(7, 8);
  channel.receive(receipt(sent[0], "accepted"));
  channel.receive(receipt(sent[0], "complete"));
  expect(sent).toHaveLength(2);
  channel.receive(receipt(sent[1], "accepted"));
  expect(sent[2].center).toEqual([7, 8]);
  expect(sent[2].epoch).toBe(1);
});

test("a failed computation permits a retry at the identical center", () => {
  const sent: { view: string; epoch: number; seq: number; center: number[] }[] = [];
  const channel = detectorChannel("one", request => sent.push(request), () => {});
  channel.request(4, 5);
  channel.receive(receipt(sent[0], "accepted"));
  channel.receive(receipt(sent[0], "rejected"));
  expect(channel.isPending()).toBe(false);
  channel.request(4, 5);
  expect(sent.map(request => request.center)).toEqual([[4, 5], [4, 5]]);
  expect(sent[1].seq).toBeGreaterThan(sent[0].seq);
});

test("point patterns pipeline two requests and retain only the newest unsent position", () => {
  const sent: Parameters<ReturnType<typeof detectorChannel>["receive"]>[0][] = [];
  const type = "show4dstem_position_receipt";
  const channel = detectorChannel("point", request => sent.push({...request, type, status: "accepted"}),
    () => {}, type, 2);
  for (let row = 0; row < 60; row++) channel.request(row, 4);
  expect(sent.map(request => request.center)).toEqual([[0, 4], [1, 4]]);
  channel.receive({...sent[0], type: "show4dstem_detector_receipt"});
  expect(sent).toHaveLength(2);
  channel.receive(sent[0]);
  expect(sent.map(request => request.center)).toEqual([[0, 4], [1, 4], [59, 4]]);
  channel.receive({...sent[0], status: "complete"});
  expect(sent).toHaveLength(3);
  channel.receive({...sent[2], status: "complete"});
  expect(channel.isPending()).toBe(false);
});

test("a disconnected view stops sending custom requests even after a delayed acceptance", () => {
  const sent: { view: string; epoch: number; seq: number; center: number[] }[] = [];
  const channel = detectorChannel("one", request => sent.push(request), () => {});
  expect(channel.request(1, 2)).toBe(true);
  channel.request(3, 4);
  channel.stop();
  channel.receive(receipt(sent[0], "accepted"));
  expect(channel.request(5, 6)).toBe(false);
  expect(channel.isPending()).toBe(false);
  expect(sent).toHaveLength(1);
});

test.each(["circle", "square", "annular"])("%s radius requests ignore only an unused inner radius", mode => {
  const tree = ts.createSourceFile("index.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let callback = "";
  const visit = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && node.name.getText(tree) === "radiusRequests") {
      const memo = node.initializer as ts.CallExpression;
      const factory = memo.arguments[0] as ts.ArrowFunction;
      callback = (factory.body as ts.CallExpression).arguments[1].getText(tree);
    }
    ts.forEachChild(node, visit);
  };
  visit(tree);
  expect(callback).not.toBe("");
  const state: Record<string, unknown> = {roi_mode: mode, roi_center_row: 5.5, roi_center_col: 7.5};
  const messages: Record<string, unknown>[] = [];
  const model = {get: (key: string) => state[key], send: (message: Record<string, unknown>) => messages.push(message)};
  const send = new Function("model", "radiusContext", "interactionContext", `return (${callback});`)(model, "roi_mode roi_center_row roi_center_col", interactionContext);
  // Switching from an annulus to a circle/square can leave an unused inner
  // radius larger than the newly dragged outer edge. Only the wire value changes.
  const request = {view: "radius", epoch: 2, seq: 3, center: [4, mode === "annular" ? 2 : 8]};
  send(request);
  expect(messages).toEqual([{type: "show4dstem_detector_radius", version: 1,
    view: "radius", epoch: 2, seq: 3, radii: [4, mode === "annular" ? 2 : 0],
    context: [mode, 5.5, 7.5]}]);
  expect(request.center).toEqual([4, mode === "annular" ? 2 : 8]);
});
