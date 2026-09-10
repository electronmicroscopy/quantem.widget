import { expect, test } from "vitest";
import { latestRegion } from "./latestRegion";

test("dragging keeps the latest position without queuing obsolete reductions", () => {
  const sent: number[][] = [];
  const requests = latestRegion((row, col) => sent.push([row, col]), [0, 0]);
  requests.request(1, 2);
  for (let row = 2; row <= 60; row++) requests.request(row, 3);
  expect(sent).toEqual([[1, 2]]);
  requests.acknowledge();
  expect(sent).toEqual([[1, 2], [60, 3]]);
  requests.request(60, 3);
  requests.acknowledge();
  expect(sent).toHaveLength(2);
  requests.request(4, 5);
  expect(sent[sent.length - 1]).toEqual([4, 5]);
});
