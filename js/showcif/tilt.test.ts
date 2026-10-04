import { describe, it, expect } from "vitest";
import { projectionBasis, specimenTiltBasis, dot, cross } from "./geometry";
const cell = [
  [4, 0, 0],
  [0, 4, 0],
  [0, 0, 4],
];
const b = projectionBasis(cell, [0, 0, 1]);
describe("rigid specimen tilt", () => {
  it("is exactly the existing geometry at zero", () => {
    expect(specimenTiltBasis(b, [0, 0])).toBe(b);
  });
  it("leans down for positive row and right for positive column", () => {
    const row = specimenTiltBasis(b, [15, 0]);
    const col = specimenTiltBasis(b, [0, 15]);
    const depth = [0, 0, 600];
    expect(-dot(depth, row.up)).toBeCloseTo(600 * Math.sin(0.015), 10);
    expect(dot(depth, row.right)).toBe(0);
    expect(dot(depth, col.right)).toBeCloseTo(600 * Math.sin(0.015), 10);
    expect(dot(depth, col.up)).toBe(0);
    expect(dot(depth, col.beam)).toBeCloseTo(600 * Math.cos(0.015), 10);
  });
  it("preserves lengths and handedness for compound tilts on an oblique cell", () => {
    const nominal = projectionBasis(
      [
        [4, 0, 0],
        [1, 5, 0],
        [0.5, 1, 6],
      ],
      [1, 1, 1],
    );
    const t = specimenTiltBasis(nominal, [-15, 12]);
    const p = [1.7, 3.2, 7.6];
    expect(
      Math.hypot(dot(p, t.right), dot(p, t.up), dot(p, t.beam)),
    ).toBeCloseTo(Math.hypot(...p), 12);
    expect(dot(cross(t.right, t.up), t.beam)).toBeCloseTo(1, 12);
    expect(dot(t.right, t.up)).toBeCloseTo(0, 12);
    const inverse = specimenTiltBasis(t, [15, -12]);
    for (const key of ["right", "up", "beam"] as const)
      inverse[key].forEach((v, i) =>
        expect(v).toBeCloseTo(nominal[key][i], 12),
      );
  });
  it("rejects malformed or out-of-range angles", () => {
    for (const v of [[16, 0], [NaN, 0], [0], [0, Infinity]])
      expect(() => specimenTiltBasis(b, v)).toThrow();
  });
});
