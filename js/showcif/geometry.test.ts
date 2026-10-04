import { describe, it, expect } from "vitest";
import { projectionBasis, dot, cellCorners } from "./geometry";
import { planGeometry } from "../planptycho/geometry";
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
it("double wave support changes only model width", () => {
  const p = {
    voltage_kV: 300,
    semiangle_mrad: 30,
    focus_depth_nm: -10,
    thickness_nm: 60,
    detector_px: 192,
    detector_mrad_per_px: 0.5570968023269496,
    scan_step_A: 0.99775,
    scan_size_px: 64,
  };
  const a = planGeometry(p),
    b = planGeometry({ ...p, wave_window_factor: 2 });
  expect(b.window_A).toBe(2 * a.window_A);
  expect(b.pixel_A).toBe(a.pixel_A);
  expect(b.theta_max_mrad).toBe(a.theta_max_mrad);
  expect(a.widest_A).toBeGreaterThan(a.window_A);
  expect(b.widest_A).toBeLessThan(b.window_A);
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
