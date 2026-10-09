/** ShowDiffraction canvas geometry: data-space <-> screen-space mapping.
 *
 * Data coordinates use the pixel-center convention (integer index = sample
 * location), matching the Python analysis grid. The square canvas stretches
 * non-square detectors anisotropically, so each axis carries its own scale.
 */

interface ViewTransform {
  scX: number;
  scY: number;
  offX: number;
  offY: number;
}

/** Per-axis scales and pan/zoom offsets for the square display canvas. */
export function viewTransform(
  canvasSize: number,
  zoom: number,
  panX: number,
  panY: number,
  detRows: number,
  detCols: number,
): ViewTransform {
  return {
    scX: (canvasSize / Math.max(detCols, 1)) * zoom,
    scY: (canvasSize / Math.max(detRows, 1)) * zoom,
    offX: (canvasSize - canvasSize * zoom) / 2 + panX,
    offY: (canvasSize - canvasSize * zoom) / 2 + panY,
  };
}

/** Screen x of a data column (pixel-center). */
export const dataColToScreenX = (col: number, view: ViewTransform) => view.offX + (col + 0.5) * view.scX;

/** Screen y of a data row (pixel-center). */
export const dataRowToScreenY = (row: number, view: ViewTransform) => view.offY + (row + 0.5) * view.scY;

/** Data coordinates of a canvas point (pixel-center). */
export function screenToData(canvasX: number, canvasY: number, view: ViewTransform) {
  return { row: (canvasY - view.offY) / view.scY - 0.5, col: (canvasX - view.offX) / view.scX - 0.5 };
}

/** Data-space azimuth (deg, +col toward +row) to canvas arc angle (rad). */
export function dataAngleToScreen(angleDeg: number, view: ViewTransform): number {
  const angleRad = (angleDeg * Math.PI) / 180;
  return Math.atan2(Math.sin(angleRad) * view.scY, Math.cos(angleRad) * view.scX);
}

/** Mean, min, max, std of a frame (offline stats for baked stacks). */
export function frameStats(frame: Float32Array): [number, number, number, number] {
  const count = frame.length;
  if (count === 0) return [0, 0, 0, 0];
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < count; i++) {
    const value = frame[i];
    sum += value;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const mean = sum / count;
  let squaredSum = 0;
  for (let i = 0; i < count; i++) {
    const deviation = frame[i] - mean;
    squaredSum += deviation * deviation;
  }
  return [mean, min, max, Math.sqrt(squaredSum / count)];
}

/** Offline panes are baked at export time; label them once scrubbed away. */
export function staleFrameNote(
  offline: boolean,
  nFrames: number,
  frameIdx: number,
  bakedFrameIdx: number,
): string | null {
  if (!offline || nFrames <= 1 || frameIdx === bakedFrameIdx) return null;
  return `computed on frame ${bakedFrameIdx + 1}`;
}

/**
 * Rows for the ring labels on the radial profile, staggered like axis tick
 * labels: a label starts just right of its ring marker and takes the first
 * row (0 at the top) whose previous label ends at least ``gap`` px before it,
 * so neighbouring rings never print over each other ("2.962.53Å"). A label
 * that fits no row is left out (null) rather than drawn on top of another.
 */
export function staggerLabelRows(
  spans: { start: number; width: number }[],
  rowCount: number,
  gap = 3,
): (number | null)[] {
  const rowEnds = new Array<number>(Math.max(0, rowCount)).fill(-Infinity);
  const rows = new Array<number | null>(spans.length).fill(null);
  const leftToRight = spans.map((_, index) => index).sort((a, b) => spans[a].start - spans[b].start);
  for (const index of leftToRight) {
    const row = rowEnds.findIndex(end => end + gap <= spans[index].start);
    if (row < 0) continue;
    rows[index] = row;
    rowEnds[row] = spans[index].start + spans[index].width;
  }
  return rows;
}
