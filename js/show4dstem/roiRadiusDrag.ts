/** Circular handles retain subpixel geometry while preserving the one-pixel gap. */
export function circularDragRadius(distance: number, boundary: "inner" | "outer", oppositeRadius: number): number {
  return boundary === "inner" ? Math.max(1, Math.min(oppositeRadius - 1, distance)) : Math.max(oppositeRadius + 1, distance);
}

/** Live resident geometry reaches mask construction before any model/RAF flush. */
export function liveRoiGeometry(
  model: { get(name: string): any },
  center: [number, number] | null,
  outer: number | null,
  inner: number | null,
) {
  if (!center && outer === null && inner === null) return model;
  return { get(name: string) {
    if (name === "roi_center_row" && center) return center[0];
    if (name === "roi_center_col" && center) return center[1];
    if (name === "roi_radius" && outer !== null) return outer;
    if (name === "roi_radius_inner" && inner !== null) return inner;
    return model.get(name);
  } };
}
