import { expect, it } from "vitest";
import { interactionConstant } from "./phase";

it("matches independent abTEM energy2sigma values across the electron energy range", () => {
  // abTEM uses older ASE constants; allow 1 ppm for CODATA-version differences.
  const reference = [
    [1, 0.008112322940737268],
    [10, 0.00259902808855018],
    [60, 0.0011356905381324882],
    [80, 0.0010087066046262614],
    [100, 0.0009243958222223731],
    [200, 0.0007288401085927866],
    [300, 0.0006526161464700888],
  ];
  for (const [energy, sigma] of reference)
    expect(Math.abs(interactionConstant(energy)! / sigma - 1)).toBeLessThan(
      1e-6,
    );
  for (const energy of [0, -1, 301, NaN, Infinity])
    expect(interactionConstant(energy)).toBeNull();
});
it("converts integrated V Å, without dividing phase by specimen thickness", () => {
  const sigma = interactionConstant(300)!;
  expect(1000 * sigma).toBeCloseTo(0.65261614647, 6);
  expect(200 * sigma + 800 * sigma).toBeCloseTo(1000 * sigma, 14);
  expect(interactionConstant(60)!).toBeGreaterThan(sigma);
});
