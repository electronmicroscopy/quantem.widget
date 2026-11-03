/** Show3D regions of interest: their defaults and the statistics of the pixels inside one. */

export type ROIItem = {
  row: number;
  col: number;
  shape: string;
  radius: number;
  radius_inner: number;
  width: number;
  height: number;
  color: string;
  line_width: number;
  highlight: boolean;
};

export const ROI_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"];

/** A new ROI at (row, col) with a default radius of 5% of the shorter image side (at least 10 px). */
export function createROI(row: number, col: number, shape: string, index: number, imgW: number = 0, imgH: number = 0): ROIItem {
  const defaultRadius = imgW > 0 && imgH > 0 ? Math.max(10, Math.round(Math.min(imgW, imgH) * 0.05)) : 10;
  return {
    row,
    col,
    shape,
    radius: defaultRadius,
    radius_inner: Math.max(5, Math.round(defaultRadius * 0.5)),
    width: defaultRadius * 2,
    height: defaultRadius * 2,
    color: ROI_COLORS[index % ROI_COLORS.length],
    line_width: 2,
    highlight: false,
  };
}

/** Fill missing ROI fields with defaults; ROIs from Python or saved state may omit shape, color or sizes. */
export function normalizeROI(roi: ROIItem, index: number): ROIItem {
  return {
    ...roi,
    color: roi.color || ROI_COLORS[index % ROI_COLORS.length],
    shape: roi.shape || "circle",
    radius: roi.radius ?? 10,
    radius_inner: roi.radius_inner ?? 5,
    width: roi.width ?? 20,
    height: roi.height ?? 20,
    line_width: roi.line_width ?? 2,
    highlight: !!roi.highlight,
  };
}

/** Mean, min, max and std of the pixels inside one ROI, or null when it holds none. */
export function computeROIPixelStats(
  data: Float32Array, imgW: number, imgH: number,
  roi: ROIItem,
): { mean: number; min: number; max: number; std: number } | null {
  const shape = roi.shape || "circle";
  let col0: number, row0: number, col1: number, row1: number;

  if (shape === "rectangle") {
    const halfWidth = roi.width / 2;
    const halfHeight = roi.height / 2;
    col0 = Math.max(0, Math.floor(roi.col - halfWidth));
    row0 = Math.max(0, Math.floor(roi.row - halfHeight));
    col1 = Math.min(imgW, Math.ceil(roi.col + halfWidth));
    row1 = Math.min(imgH, Math.ceil(roi.row + halfHeight));
  } else {
    const radius = roi.radius;
    col0 = Math.max(0, Math.floor(roi.col - radius));
    row0 = Math.max(0, Math.floor(roi.row - radius));
    col1 = Math.min(imgW, Math.ceil(roi.col + radius));
    row1 = Math.min(imgH, Math.ceil(roi.row + radius));
  }

  const cropW = col1 - col0;
  const cropH = row1 - row0;
  if (cropW < 1 || cropH < 1) return null;

  let sum = 0, sumSq = 0, minVal = Infinity, maxVal = -Infinity, count = 0;

  if (shape === "circle") {
    const radiusSq = roi.radius * roi.radius;
    for (let rowOffset = 0; rowOffset < cropH; rowOffset++) {
      for (let colOffset = 0; colOffset < cropW; colOffset++) {
        const imgCol = col0 + colOffset, imgRow = row0 + rowOffset;
        const distSq = (imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2;
        if (distSq > radiusSq) continue;
        const value = data[imgRow * imgW + imgCol];
        sum += value; sumSq += value * value;
        if (value < minVal) minVal = value;
        if (value > maxVal) maxVal = value;
        count++;
      }
    }
  } else if (shape === "annular") {
    const radiusSq = roi.radius * roi.radius;
    const innerRadiusSq = (roi.radius_inner || 0) ** 2;
    for (let rowOffset = 0; rowOffset < cropH; rowOffset++) {
      for (let colOffset = 0; colOffset < cropW; colOffset++) {
        const imgCol = col0 + colOffset, imgRow = row0 + rowOffset;
        const distSq = (imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2;
        if (distSq > radiusSq || distSq < innerRadiusSq) continue;
        const value = data[imgRow * imgW + imgCol];
        sum += value; sumSq += value * value;
        if (value < minVal) minVal = value;
        if (value > maxVal) maxVal = value;
        count++;
      }
    }
  } else {
    // square or rectangle - all pixels in bounding box
    for (let rowOffset = 0; rowOffset < cropH; rowOffset++) {
      for (let colOffset = 0; colOffset < cropW; colOffset++) {
        const value = data[(row0 + rowOffset) * imgW + (col0 + colOffset)];
        sum += value; sumSq += value * value;
        if (value < minVal) minVal = value;
        if (value > maxVal) maxVal = value;
        count++;
      }
    }
  }

  if (count === 0) return null;
  const mean = sum / count;
  const std = Math.sqrt(Math.max(0, sumSq / count - mean * mean));
  return { mean, min: minVal, max: maxVal, std };
}
