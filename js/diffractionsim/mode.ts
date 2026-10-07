/** Pattern modes of DiffractionSim. */
export type PatternMode = "nanobeam" | "cbed" | "kikuchi";

export const PATTERN_MODES: readonly PatternMode[] = ["nanobeam", "cbed", "kikuchi"];

/**
 * Canonical pattern mode for a trait or directive value. "kossel" is the
 * earlier name of "kikuchi" and maps to it; anything unknown gives `fallback`.
 */
export function normalizeMode(value: unknown, fallback: PatternMode = "nanobeam"): PatternMode {
  const v = typeof value === "string" ? value.trim().toLowerCase() : "";
  if (v === "kossel") return "kikuchi";
  return (PATTERN_MODES as readonly string[]).includes(v) ? (v as PatternMode) : fallback;
}
