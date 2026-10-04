import { describe, expect, it } from "vitest";
import { repeatedAtom, repeatCount } from "./geometry";

describe("unit-cell repeats", () => {
  it("maps every instance to an oblique-cell translation in ASE order", () => {
    const atoms = new Float32Array([0.25, 0.5, 0.75, 0, 1, 1, 1, 1]);
    const cell = [
      [3, 0, 0],
      [1, 4, 0],
      [0.5, 0.25, 5],
    ];
    let id = 0;
    for (let a = 0; a < 2; a++)
      for (let b = 0; b < 3; b++)
        for (let c = 0; c < 4; c++) {
          for (let n = 0; n < 2; n++) {
            const result = repeatedAtom(atoms, cell, [2, 3, 4], id++);
            expect(result).toEqual([
              atoms[4 * n] + 3 * a + b + 0.5 * c,
              atoms[4 * n + 1] + 4 * b + 0.25 * c,
              atoms[4 * n + 2] + 5 * c,
              n,
            ]);
          }
        }
    expect(id).toBe(repeatCount(2, [2, 3, 4]));
  });
  it("rejects invalid or excessive repeats while allowing the display limit", () => {
    expect(repeatCount(5, [50, 50, 20])).toBe(250000);
    for (const repeats of [
      [0, 1, 1],
      [1.5, 2, 2],
      [1, 2],
      [100, 100, 100],
    ]) {
      expect(() => repeatCount(5, repeats)).toThrow();
    }
  });
});
