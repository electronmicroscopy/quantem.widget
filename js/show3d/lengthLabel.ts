import { unitSymbol } from "../figure";

// Size of one unit in Angstrom, keyed by the lower-case spellings `pixel_unit` accepts.
const ANGSTROMS_PER_UNIT: Record<string, number> = {
  "å": 1, a: 1, ang: 1, angstrom: 1, angstroms: 1,
  nm: 10, nanometer: 10, nanometers: 10,
  pm: 0.01, picometer: 0.01, picometers: 0.01,
  "µm": 1e4, "μm": 1e4, um: 1e4, micron: 1e4, microns: 1e4,
  mm: 1e7, millimeter: 1e7, millimeters: 1e7,
};

/**
 * A length given in `unit` (the widget's `pixel_unit`) as the number and symbol to print.
 *
 * Lengths read in nm from 1 nm up and in Å below (µm and mm for larger ones), whatever unit the
 * data are calibrated in; before, the profile axis and the FFT d-spacing treated every
 * calibrated length as Angstrom, so nm data read ten times too small. A unit that is not a
 * length (px, mrad, ...) keeps its own symbol.
 */
export function readableLength(value: number, unit: string): { value: number; unit: string } {
  const angstromsPerUnit = ANGSTROMS_PER_UNIT[(unit || "").trim().toLowerCase()];
  if (angstromsPerUnit === undefined) return { value, unit: unitSymbol(unit || "px") };
  const angstroms = value * angstromsPerUnit;
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
