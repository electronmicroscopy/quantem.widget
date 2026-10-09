import { describe, it, expect } from "vitest";
import { projectionBasis, dot, cellCorners } from "./geometry";
describe("ShowCIF directions", () => {
  it("uses the direct lattice for a nonorthogonal crystal", () => {
    const b = projectionBasis(
      [
        [4, 0, 0],
        [2, 3, 0],
        [0, 0, 5],
      ],
      [0, 1, 0],
    );
    expect(b.beam[0]).toBeCloseTo(2 / Math.sqrt(13));
    expect(b.beam[1]).toBeCloseTo(3 / Math.sqrt(13));
    expect(dot(b.up, b.beam)).toBeCloseTo(0);
    expect(dot(b.right, b.up)).toBeCloseTo(0);
  });
  it("keeps [001] right=x up=y and exact cell corners", () => {
    const c = [
        [4, 0, 0],
        [0, 4, 0],
        [0, 0, 5],
      ],
      b = projectionBasis(c, [0, 0, 1]);
    expect(b.right).toEqual([1, 0, 0]);
    expect(b.up).toEqual([0, 1, 0]);
    expect(cellCorners(c)[7]).toEqual([4, 4, 5]);
  });
});
it("orthogonal side views stay right-handed for an oblique unit cell", async () => {
  const { projectionBasis, orthogonalBases, dot, cross } =
    await import("./geometry");
  const b = projectionBasis(
    [
      [4, 0, 0],
      [1, 5, 0],
      [0.5, 1, 6],
    ],
    [1, 1, 1],
  );
  const views = orthogonalBases(b);
  for (const v of views) {
    expect(dot(v.right, v.up)).toBeCloseTo(0);
    expect(dot(v.right, v.beam)).toBeCloseTo(0);
    expect(dot(v.up, v.beam)).toBeCloseTo(0);
    expect(dot(cross(v.right, v.up), v.beam)).toBeCloseTo(1);
  }
  expect(views[1].up).toEqual(b.beam);
  expect(views[2].up).toEqual(b.beam);
  expect(dot(views[1].beam, views[2].beam)).toBeCloseTo(0);
});
