import { expect, test, vi } from "vitest";
import { detectorChannel, detectorViewId } from "./detectorChannel";

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
