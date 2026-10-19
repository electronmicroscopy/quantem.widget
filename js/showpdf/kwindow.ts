// k-window slider commit: a handle left where it was keeps its setting (<= 0: follow the fit
// range), a moved one takes the committed value.
export function committedWindowEnd(
  committed: number, setting: number, fit: number, min: number, max: number,
): number {
  const shown = shownOnSlider(setting > 0 ? setting : fit, min, max);
  return committed === shown ? setting : committed;
}

// k-fit slider commit, by the same rule (Python accepts a fit end up to half a k step outside the
// float32 k range, which MUI reports clamped).
export function committedFitEnd(committed: number, value: number, min: number, max: number): number {
  return committed === shownOnSlider(value, min, max) ? value : committed;
}

// Where an untouched handle reports value v: MUI clamps it to the slider's [min, max].
function shownOnSlider(v: number, min: number, max: number) {
  return Math.max(min, Math.min(v, max));
}
