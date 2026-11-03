/**
 * The one unit table of the frontends: what the unit string of a calibrated axis means.
 * Mirrors src/quantem/widget/units.py, which validates every unit before it reaches a trait;
 * keep the two in step. Reference units: angstrom (length), 1/angstrom (reciprocal_length),
 * mrad (angle), second (time).
 */

export type UnitKind = "length" | "reciprocal_length" | "angle" | "time";

interface UnitEntry { kind: UnitKind; size: number; symbol: string }

const UNIT_TABLE: Record<string, UnitEntry> = {
  angstrom: { kind: "length", size: 1, symbol: "Å" }, "Å": { kind: "length", size: 1, symbol: "Å" }, A: { kind: "length", size: 1, symbol: "Å" },
  nm: { kind: "length", size: 10, symbol: "nm" }, pm: { kind: "length", size: 0.01, symbol: "pm" },
  "µm": { kind: "length", size: 1e4, symbol: "µm" }, "μm": { kind: "length", size: 1e4, symbol: "µm" }, um: { kind: "length", size: 1e4, symbol: "µm" },
  mm: { kind: "length", size: 1e7, symbol: "mm" }, m: { kind: "length", size: 1e10, symbol: "m" },
  "1/angstrom": { kind: "reciprocal_length", size: 1, symbol: "Å⁻¹" }, "1/Å": { kind: "reciprocal_length", size: 1, symbol: "Å⁻¹" },
  "1/A": { kind: "reciprocal_length", size: 1, symbol: "Å⁻¹" }, "A^-1": { kind: "reciprocal_length", size: 1, symbol: "Å⁻¹" },
  "Å^-1": { kind: "reciprocal_length", size: 1, symbol: "Å⁻¹" },
  "1/nm": { kind: "reciprocal_length", size: 0.1, symbol: "nm⁻¹" }, "nm^-1": { kind: "reciprocal_length", size: 0.1, symbol: "nm⁻¹" },
  "1/µm": { kind: "reciprocal_length", size: 1e-4, symbol: "µm⁻¹" }, "1/μm": { kind: "reciprocal_length", size: 1e-4, symbol: "µm⁻¹" },
  "1/um": { kind: "reciprocal_length", size: 1e-4, symbol: "µm⁻¹" },
  mrad: { kind: "angle", size: 1, symbol: "mrad" }, rad: { kind: "angle", size: 1e3, symbol: "rad" },
  "µrad": { kind: "angle", size: 1e-3, symbol: "µrad" }, urad: { kind: "angle", size: 1e-3, symbol: "µrad" },
  deg: { kind: "angle", size: (1e3 * Math.PI) / 180, symbol: "°" },
  s: { kind: "time", size: 1, symbol: "s" }, ms: { kind: "time", size: 1e-3, symbol: "ms" },
  "µs": { kind: "time", size: 1e-6, symbol: "µs" }, us: { kind: "time", size: 1e-6, symbol: "µs" },
  ns: { kind: "time", size: 1e-9, symbol: "ns" }, ps: { kind: "time", size: 1e-12, symbol: "ps" }, fs: { kind: "time", size: 1e-15, symbol: "fs" },
};

// Length units a label may switch to, largest first, each as its size in Angstrom: a value
// below 1 in one unit (0.5 nm) reads as a whole number in a smaller one (5 Å).
export const LENGTH_LADDER: { symbol: string; angstroms: number }[] = [
  { symbol: "mm", angstroms: 1e7 }, { symbol: "µm", angstroms: 1e4 }, { symbol: "nm", angstroms: 10 },
  { symbol: "Å", angstroms: 1 }, { symbol: "pm", angstroms: 0.01 },
];

/**
 * The symbol a label prints for a unit string: the table's symbol (Å, nm, Å⁻¹, mrad), "px" for
 * an uncalibrated image axis, and any other text unchanged (the frame axis of a stack is labelled
 * with whatever its author wrote, such as "frame" or "K").
 */
export function unitSymbol(unit: string | null | undefined): string {
  const text = unit ?? "";
  if (text === "" || text === "pixels" || text === "pixel" || text === "px") return "px";
  return UNIT_TABLE[text]?.symbol ?? text;
}

/** One `unit` in Angstrom, or undefined when the unit is not a length (px, mrad, 1/angstrom, ps). */
export function angstromsPerUnit(unit: string | null | undefined): number | undefined {
  const entry = UNIT_TABLE[unit ?? ""];
  return entry?.kind === "length" ? entry.size : undefined;
}

/**
 * An angle in degrees as the files hold it: the same direction in [0, 360). A slider may run
 * below 0 or past 360 (-10 and 350 are one rotation); the Python writers wrap the same way
 * (quantem.widget.units.wrap_degrees), so a page that wrote the raw slider value put a negative
 * angle beside theirs. An angle already in the range is returned bit for bit: adding 360 before a
 * second remainder would write 0.1 as 0.10000000000002274.
 */
export function wrapDegrees(value: number): number {
  const remainder = value % 360;
  const wrapped = remainder < 0 ? remainder + 360 : remainder;
  // a tiny negative angle rounds up to 360 itself
  return wrapped === 360 ? 0 : wrapped;
}

/**
 * The turn of `value` (degrees) that lies in [min, max], or `value` when none does: a control
 * whose range starts below 0 shows a stored 350 at -10 instead of pinned at its upper end.
 */
export function degreesWithin(value: number, min: number, max: number): number {
  if (value >= min && value <= max) return value;
  const turned = value - 360 * Math.ceil((value - max) / 360);
  return turned >= min ? turned : value;
}
