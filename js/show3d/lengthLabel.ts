import { angstromsPerUnit, unitSymbol } from "../units";

/**
 * A length given in `unit` (the widget's `pixel_unit`) as the number and symbol to print.
 *
 * Lengths read in nm from 1 nm up and in Å below (µm and mm for larger ones), whatever unit the
 * data are calibrated in, so the profile axis and the FFT d-spacing never read nm data as
 * Angstrom. A unit that is not a length (px, mrad, ...) keeps its own symbol.
 */
export function readableLength(value: number, unit: string): { value: number; unit: string } {
  const unitAngstroms = angstromsPerUnit(unit);
  if (unitAngstroms === undefined) return { value, unit: unitSymbol(unit) };
  const angstroms = value * unitAngstroms;
  if (angstroms >= 1e7) return { value: angstroms / 1e7, unit: "mm" };
  if (angstroms >= 1e4) return { value: angstroms / 1e4, unit: "µm" };
  if (angstroms >= 10) return { value: angstroms / 10, unit: "nm" };
  return { value: angstroms, unit: "Å" };
}

/** `readableLength` as text with two decimals, e.g. "2.35 Å", "1.20 nm". */
export function formatLength(value: number, unit: string): string {
  const length = readableLength(value, unit);
  return `${length.value.toFixed(2)} ${length.unit}`;
}
