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
