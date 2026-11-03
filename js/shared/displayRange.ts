import { signedLog1p } from "../display/stats";

/** Display bounds for the contrast slider: explicit vmin/vmax traits win over
 *  the data range, and both move into log space when log scale is on so the
 *  slider percentages map onto what is actually painted. */
export function resolveDisplayBounds(
  dataMin: number, dataMax: number,
  traitVmin: number | null | undefined, traitVmax: number | null | undefined,
  logScale: boolean,
): { min: number; max: number } {
  return {
    min: logScale ? signedLog1p(traitVmin ?? dataMin) : (traitVmin ?? dataMin),
    max: logScale ? signedLog1p(traitVmax ?? dataMax) : (traitVmax ?? dataMax),
  };
}
