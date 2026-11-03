/** Distance from (col, row) to the segment (col0, row0)-(col1, row1); used for
 *  line-profile and ROI hit testing in image pixel units. */
export function pointToSegmentDistance(col: number, row: number, col0: number, row0: number, col1: number, row1: number): number {
  const deltaCol = col1 - col0;
  const deltaRow = row1 - row0;
  const lengthSquared = deltaCol * deltaCol + deltaRow * deltaRow;
  if (lengthSquared <= 1e-12) return Math.sqrt((col - col0) ** 2 + (row - row0) ** 2);
  // Fraction along the segment of the point's projection, clamped to the segment.
  const fraction = Math.max(0, Math.min(1, ((col - col0) * deltaCol + (row - row0) * deltaRow) / lengthSquared));
  return Math.sqrt((col - (col0 + fraction * deltaCol)) ** 2 + (row - (row0 + fraction * deltaRow)) ** 2);
}

type TouchPoint = { clientX: number; clientY: number };

export const touchDistance = (a: TouchPoint, b: TouchPoint) => Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
export const touchMidpoint = (a: TouchPoint, b: TouchPoint) => ({ x: (a.clientX + b.clientX) / 2, y: (a.clientY + b.clientY) / 2 });

/** Start of a one-finger pan: where it began and the view it began from. */
export function panGestureStart<S>(touch: TouchPoint, startState: S) {
  return {
    mode: "pan" as const,
    startX: touch.clientX,
    startY: touch.clientY,
    startDistance: 0,
    startMidX: touch.clientX,
    startMidY: touch.clientY,
    startState,
  };
}

/** Start of a two-finger pinch: its midpoint, finger distance (at least 1 px) and the view it began from. */
export function pinchGestureStart<S>(first: TouchPoint, second: TouchPoint, startState: S) {
  const mid = touchMidpoint(first, second);
  return {
    mode: "pinch" as const,
    startX: mid.x,
    startY: mid.y,
    startDistance: Math.max(1, touchDistance(first, second)),
    startMidX: mid.x,
    startMidY: mid.y,
    startState,
  };
}
