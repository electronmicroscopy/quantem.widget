export type CanvasRectangle = { x: number; y: number; width: number; height: number };
type Bounds = { left: number; top: number; width: number; height: number };

/** Native image pixels where texture limits permit; source arrays are never resized. */
export function sharedCanvasLayout(grid: Bounds, tiles: Bounds[], rows: number, cols: number, limit: number) {
  if (grid.width <= 0 || grid.height <= 0 || !tiles.length
    || tiles.some(tile => tile.width <= 0 || tile.height <= 0)) return null;
  const nativeScale = Math.max(...tiles.map(tile => Math.max(cols / tile.width, rows / tile.height)));
  const scale = Math.min(nativeScale, limit / grid.width, limit / grid.height);
  const width = Math.max(1, Math.min(limit, Math.round(grid.width * scale)));
  const height = Math.max(1, Math.min(limit, Math.round(grid.height * scale)));
  const rectangles = tiles.map(tile => {
    const x = Math.max(0, Math.min(width - 1, Math.round((tile.left - grid.left) * scale)));
    const y = Math.max(0, Math.min(height - 1, Math.round((tile.top - grid.top) * scale)));
    const right = Math.max(x + 1, Math.min(width, Math.round((tile.left - grid.left + tile.width) * scale)));
    const bottom = Math.max(y + 1, Math.min(height, Math.round((tile.top - grid.top + tile.height) * scale)));
    return { x, y, width: right - x, height: bottom - y };
  });
  return { width, height, rectangles };
}
