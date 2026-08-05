/** ShowDiffraction canvas geometry: data-space <-> screen-space mapping.
 *
 * Data coordinates use the pixel-center convention (integer index = sample
 * location), matching the Python analysis grid. The square canvas stretches
 * non-square detectors anisotropically, so each axis carries its own scale.
 */

export interface ViewTransform {
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
export const dataColToScreenX = (col: number, t: ViewTransform) => t.offX + (col + 0.5) * t.scX;

/** Screen y of a data row (pixel-center). */
export const dataRowToScreenY = (row: number, t: ViewTransform) => t.offY + (row + 0.5) * t.scY;

/** Data coordinates of a canvas point (pixel-center). */
export function screenToData(mx: number, my: number, t: ViewTransform) {
  return { row: (my - t.offY) / t.scY - 0.5, col: (mx - t.offX) / t.scX - 0.5 };
}

/** Data-space azimuth (deg, +col toward +row) to canvas arc angle (rad). */
export function dataAngleToScreen(angleDeg: number, t: ViewTransform): number {
  const a = (angleDeg * Math.PI) / 180;
  return Math.atan2(Math.sin(a) * t.scY, Math.cos(a) * t.scX);
}

/** Mean, min, max, std of a frame (offline stats for baked stacks). */
export function frameStats(frame: Float32Array): [number, number, number, number] {
  const n = frame.length;
  if (n === 0) return [0, 0, 0, 0];
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = frame[i];
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const mean = sum / n;
  let sq = 0;
  for (let i = 0; i < n; i++) {
    const d = frame[i] - mean;
    sq += d * d;
  }
  return [mean, min, max, Math.sqrt(sq / n)];
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
