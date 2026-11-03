export type DetectorRoiMode = "point" | "circle" | "square" | "rect" | "annular" | "off";

type DetectorResizeGeometry = {
  radius?: number;
  radiusInner?: number;
  width?: number;
  height?: number;
};

/** What a press on a detector ROI grabs: the move zone, a resize band (and which axes it resizes), a ring's inner handle, or nothing. */
export type DetectorGrab =
  | { action: "move" }
  | { action: "resize"; rows: boolean; cols: boolean }
  | { action: "resizeInner" }
  | { action: "outside" };

/** cos(45 deg): circle and ring resize handles are drawn on the edge at 45 degrees. */
export const CIRCLE_HANDLE_ANGLE = 0.707;

/** The pointer and the ROI geometry at the press that starts a resize. */
export type DetectorResizeStart = {
  pointerRow: number;
  pointerCol: number;
  radius: number;
  radiusInner: number;
  width: number;
  height: number;
  rows: boolean;
  cols: boolean;
};

// Show4DSTEM's masks (Python show4dstem/detector.py and scan_indices) center
// pixel i at coordinate i, while a canvas draws pixel i over [i, i + 1). Overlays
// and pointers go through these two maps, so a drawn ROI sits on the pixels its
// mask selects and a ROI dropped on a pixel is centered on that pixel.
export function maskToCanvas(coordinate: number, zoom: number, pan: number): number {
  return (coordinate + 0.5) * zoom + pan;
}

export function canvasToMask(canvasCoordinate: number, zoom: number, pan: number): number {
  return (canvasCoordinate - pan) / zoom - 0.5;
}

export function clampDetectorCenter(
  row: number,
  col: number,
  detectorRows: number,
  detectorCols: number,
): { row: number; col: number } {
  return {
    row: Math.max(0, Math.min(detectorRows - 1, row)),
    col: Math.max(0, Math.min(detectorCols - 1, col)),
  };
}

/**
 * What a press at (pointerRow, pointerCol) grabs on a detector ROI.
 *
 * Circle, annular and square: the inner half of the radius moves the ROI (so
 * does the hole of a ring wider than that); from there out to `hitMargin` past
 * the edge the press resizes it, so the rim is easy to catch by hand. Distance
 * is Euclidean for circles and rings and Chebyshev for squares. Rect: a press
 * within `hitMargin` of an edge resizes along that edge's axis (both at a
 * corner), anywhere else inside moves. `hitMargin` is the pointer's hit area in
 * detector pixels.
 *
 * A ring's inner handle (drawn at 45 degrees on the inner edge) resizes the
 * hole when pressed within `hitMargin`. Near the center of a rect or ring a
 * press always moves the ROI, out to the smaller of `hitMargin` and 0.45 of
 * the distance to the nearest edge: on a detector only a few pixels wide the
 * hit margins otherwise cover its center and the ROI can only be resized.
 */
export function grabDetector({
  mode,
  centerRow,
  centerCol,
  pointerRow,
  pointerCol,
  radius,
  radiusInner,
  width,
  height,
  hitMargin,
}: {
  mode: DetectorRoiMode;
  centerRow: number;
  centerCol: number;
  pointerRow: number;
  pointerCol: number;
  radius: number;
  radiusInner: number;
  width: number;
  height: number;
  hitMargin: number;
}): DetectorGrab {
  const rowOffset = Math.abs(pointerRow - centerRow);
  const colOffset = Math.abs(pointerCol - centerCol);
  const centerGrab = (nearestEdge: number) => Math.hypot(rowOffset, colOffset) <= Math.min(hitMargin, 0.45 * nearestEdge);
  if (mode === "rect") {
    if (centerGrab(Math.min(width, height) / 2)) return { action: "move" };
    const cols = Math.abs(colOffset - width / 2) <= hitMargin && rowOffset <= height / 2 + hitMargin;
    const rows = Math.abs(rowOffset - height / 2) <= hitMargin && colOffset <= width / 2 + hitMargin;
    if (rows || cols) return { action: "resize", rows, cols };
    return colOffset <= width / 2 && rowOffset <= height / 2 ? { action: "move" } : { action: "outside" };
  }
  if ((mode !== "circle" && mode !== "annular" && mode !== "square") || !(radius > 0)) return { action: "outside" };
  if (mode === "annular" && radiusInner > 0 && !centerGrab(radiusInner)) {
    const handleOffset = radiusInner * CIRCLE_HANDLE_ANGLE;
    if (Math.hypot(pointerRow - centerRow - handleOffset, pointerCol - centerCol - handleOffset) < hitMargin) return { action: "resizeInner" };
  }
  const distance = mode === "square" ? Math.max(rowOffset, colOffset) : Math.hypot(rowOffset, colOffset);
  const moveRadius = Math.max(radius / 2, mode === "annular" ? radiusInner : 0);
  if (distance < moveRadius) return { action: "move" };
  if (distance <= radius + hitMargin) return { action: "resize", rows: true, cols: true };
  return { action: "outside" };
}

/**
 * The ROI size while a resize drags the pointer from `start` to (pointerRow, pointerCol).
 *
 * The edge follows the pointer by the distance the pointer moved, so a press
 * anywhere on the resize band never jumps the size to the press point: a
 * circle or ring radius changes as the pointer's distance from the center
 * changes, a square half-side as its Chebyshev distance, and a rect side by
 * twice the change along each axis the press grabbed. `preserveAspect` (Shift)
 * keeps the rect's starting aspect ratio.
 */
export function resizeDetectorFromPointer({
  mode,
  centerRow,
  centerCol,
  pointerRow,
  pointerCol,
  start,
  resizeInner = false,
  preserveAspect = false,
}: {
  mode: DetectorRoiMode;
  centerRow: number;
  centerCol: number;
  pointerRow: number;
  pointerCol: number;
  start: DetectorResizeStart;
  resizeInner?: boolean;
  preserveAspect?: boolean;
}): DetectorResizeGeometry | null {
  const rowOffset = Math.abs(pointerRow - centerRow);
  const colOffset = Math.abs(pointerCol - centerCol);
  const startRowOffset = Math.abs(start.pointerRow - centerRow);
  const startColOffset = Math.abs(start.pointerCol - centerCol);
  const radialChange = Math.hypot(rowOffset, colOffset) - Math.hypot(startRowOffset, startColOffset);

  if (resizeInner && mode === "annular") {
    return {
      radiusInner: Math.max(1, Math.min(start.radius - 1, start.radiusInner + radialChange)),
    };
  }

  if (mode === "rect") {
    let width = start.cols ? Math.max(2, start.width + 2 * (colOffset - startColOffset)) : start.width;
    let height = start.rows ? Math.max(2, start.height + 2 * (rowOffset - startRowOffset)) : start.height;
    if (preserveAspect && start.width > 0 && start.height > 0) {
      const aspectRatio = start.width / start.height;
      if (width / height > aspectRatio) height = Math.max(2, width / aspectRatio);
      else width = Math.max(2, height * aspectRatio);
    }
    return { width, height };
  }

  if (mode === "circle" || mode === "annular") {
    return { radius: Math.max(mode === "annular" ? start.radiusInner + 1 : 1, start.radius + radialChange) };
  }

  if (mode === "square") {
    const chebyshevChange = Math.max(rowOffset, colOffset) - Math.max(startRowOffset, startColOffset);
    return { radius: Math.max(1, start.radius + chebyshevChange) };
  }

  return null;
}
