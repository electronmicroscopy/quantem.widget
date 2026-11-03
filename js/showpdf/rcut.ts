// r cut slider on the r grid of the computed G(r), n·r_step from r = 0: from the first grid point
// above 0 to the last one at or below min(5 Å, r max). `decimals` is the number r_step needs
// (0.02 → 2), for the readout ("0.80 Å").
export function rCutSlider(
  rStep: number, rMax: number,
): { min: number; max: number; step: number; decimals: number } {
  const step = rStep > 0 ? rStep : 0.1;
  const decimals = decimalsOf(step);
  const gridPoint = (n: number) => Number((n * step).toFixed(decimals));
  const min = gridPoint(1);
  const max = gridPoint(Math.floor(Math.min(5, rMax) / step + 1e-9));
  return { min, max: Math.max(min, max), step, decimals };
}

// Decimals that write step exactly, from 1 to at most 6.
function decimalsOf(step: number) {
  const offInteger = (x: number) => Math.abs(x - Math.round(x));
  let decimals = 1;
  while (decimals < 6 && offInteger(step * 10 ** decimals) > 1e-6) decimals++;
  return decimals;
}
