// How an image panel's colormapped offscreen lands on its canvas: the draw
// transform (rotation of a kernel-less export, zoom about the centre, pan,
// flips) and, for a live contrast drag without WebGPU, which source pixel each
// canvas pixel shows.

export type PanelPlacement = {
  canvasW: number;
  canvasH: number;
  zoom: number;
  panX: number;
  panY: number;
  /** Quarter turns, counter-clockwise positive like `np.rot90(k=...)`. */
  rotationTurns: number;
  flipX: boolean;
  flipY: boolean;
};

/**
 * Set `ctx`'s transform for one panel and return the size the offscreen is
 * drawn at (canvas size, axes swapped by an odd rotation). Callers then draw
 * `drawImage(offscreen, 0, 0, offscreen.width, offscreen.height, 0, 0, drawW, drawH)`.
 */
export function applyPanelTransform(ctx: CanvasRenderingContext2D, placement: PanelPlacement): { drawW: number; drawH: number } {
  const { canvasW, canvasH, zoom, panX, panY, rotationTurns, flipX, flipY } = placement;
  const rotated = rotationTurns % 2 !== 0;
  const drawW = rotated ? canvasH : canvasW;
  const drawH = rotated ? canvasW : canvasH;
  if (rotationTurns !== 0) {
    ctx.translate(canvasW / 2, canvasH / 2);
    // Canvas rotates clockwise for positive angles on its y-down axis; the
    // turn count follows np.rot90, which is counter-clockwise.
    ctx.rotate(-rotationTurns * Math.PI / 2);
    ctx.translate(-drawW / 2, -drawH / 2);
  }
  if (zoom !== 1 || panX !== 0 || panY !== 0) {
    const cx = drawW / 2;
    const cy = drawH / 2;
    ctx.translate(cx + panX, cy + panY);
    ctx.scale(zoom, zoom);
    ctx.translate(-cx, -cy);
  }
  if (flipX || flipY) {
    ctx.translate(flipX ? drawW : 0, flipY ? drawH : 0);
    ctx.scale(flipX ? -1 : 1, flipY ? -1 : 1);
  }
  return { drawW, drawH };
}

/** A W x 1 (or 1 x H) strip whose pixel k holds k in its RGB bytes. */
function indexStrip(length: number, horizontal: boolean): HTMLCanvasElement {
  const strip = document.createElement("canvas");
  strip.width = horizontal ? length : 1;
  strip.height = horizontal ? 1 : length;
  const ctx = strip.getContext("2d")!;
  const image = ctx.createImageData(strip.width, strip.height);
  for (let k = 0; k < length; k++) {
    image.data[4 * k] = k & 255;
    image.data[4 * k + 1] = (k >> 8) & 255;
    image.data[4 * k + 2] = (k >> 16) & 255;
    image.data[4 * k + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return strip;
}

/**
 * The source pixel (row * width + col) each canvas pixel shows when a
 * width x height panel is drawn with nearest sampling, or -1 where the panel
 * does not cover the pixel; null when it covers none (or the canvas cannot be
 * read). Which source pixel a scaled drawImage picks is the browser's
 * rounding, so the browser is asked: a column-index strip and a row-index
 * strip are drawn through the same transform and read back. Nearest sampling
 * of one axis does not depend on the other, so a strip lands on the columns
 * (rows) the full panel lands on, and a paint from this map equals the
 * panel's drawImage pixel for pixel.
 */
export function sampledSourceIndex(width: number, height: number, placement: PanelPlacement): Int32Array | null {
  const probe = document.createElement("canvas");
  probe.width = placement.canvasW;
  probe.height = placement.canvasH;
  const ctx = probe.getContext("2d");
  if (!ctx) return null;
  const readStrip = (strip: HTMLCanvasElement): Uint8ClampedArray | null => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, probe.width, probe.height);
    ctx.imageSmoothingEnabled = false;
    const { drawW, drawH } = applyPanelTransform(ctx, placement);
    ctx.drawImage(strip, 0, 0, strip.width, strip.height, 0, 0, drawW, drawH);
    return ctx.getImageData(0, 0, probe.width, probe.height)?.data ?? null;
  };
  const columns = readStrip(indexStrip(width, true));
  const rows = readStrip(indexStrip(height, false));
  if (!columns || !rows) return null;
  const index = new Int32Array(probe.width * probe.height);
  let covered = 0;
  for (let k = 0; k < index.length; k++) {
    const p = 4 * k;
    if (columns[p + 3] !== 255 || rows[p + 3] !== 255) { index[k] = -1; continue; }
    const col = columns[p] | (columns[p + 1] << 8) | (columns[p + 2] << 16);
    const row = rows[p] | (rows[p + 1] << 8) | (rows[p + 2] << 16);
    index[k] = row * width + col;
    covered++;
  }
  return covered > 0 ? index : null;
}
