/** Relativistic interaction constant in radians / (V Å), CODATA 2018 SI. */
export function interactionConstant(energyKeV: number): number | null {
  if (!Number.isFinite(energyKeV) || energyKeV <= 0 || energyKeV > 300)
    return null;
  const e = 1.602176634e-19,
    h = 6.62607015e-34;
  const m = 9.1093837015e-31,
    c = 299792458;
  const x = (energyKeV * 1000 * e) / (m * c * c);
  const speed = (c * Math.sqrt(x * (2 + x))) / (1 + x);
  return ((2 * Math.PI * e) / (h * speed)) * 1e-10;
}
