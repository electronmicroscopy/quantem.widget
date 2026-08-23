/** Shared pan/zoom transform for the single-canvas image widgets. */

import { useEffect, useState } from "react";
import { extractBytes } from "./format";

export const MIN_ZOOM = 0.5;
export const MAX_ZOOM = 20;

export interface ImageViewport {
  /** Original image rows and columns. */
  height: number;
  width: number;
  /** Square canvas edge, in CSS px. */
  canvas: number;
  zoom: number;
  panX: number;
  panY: number;
}

export type ViewTransform = Pick<ImageViewport, "zoom" | "panX" | "panY">;

export const IDENTITY_VIEW: ViewTransform = { zoom: 1, panX: 0, panY: 0 };

export function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

/** Image pixels per CSS px at zoom 1, fitting the longest side to the canvas. */
export function displayScale(v: ImageViewport): number {
  const longest = Math.max(v.height, v.width);
  return longest > 0 ? v.canvas / longest : 1;
}

/** Canvas coordinates of a mouse event, corrected for CSS scaling. */
export function canvasPoint(
  canvas: HTMLCanvasElement,
  e: { clientX: number; clientY: number },
): [number, number] {
  const rect = canvas.getBoundingClientRect();
  return [
    (e.clientX - rect.left) * (canvas.width / rect.width),
    (e.clientY - rect.top) * (canvas.height / rect.height),
  ];
}

export function screenToImage(
  v: ImageViewport,
  canvasX: number,
  canvasY: number,
): [number, number] {
  const scale = displayScale(v) * v.zoom;
  const center = v.canvas / 2;
  const col = (canvasX - center - v.panX) / scale + v.width / 2;
  const row = (canvasY - center - v.panY) / scale + v.height / 2;
  return [row, col];
}

export function imageToScreen(
  v: ImageViewport,
  row: number,
  col: number,
): [number, number] {
  const scale = displayScale(v) * v.zoom;
  const center = v.canvas / 2;
  return [
    center + (col - v.width / 2) * scale + v.panX,
    center + (row - v.height / 2) * scale + v.panY,
  ];
}

/** Wheel zoom that keeps the image point under the cursor fixed. */
export function zoomAt(
  v: ImageViewport,
  canvasX: number,
  canvasY: number,
  deltaY: number,
): ViewTransform {
  const center = v.canvas / 2;
  const anchorX = (canvasX - center - v.panX) / v.zoom + center;
  const anchorY = (canvasY - center - v.panY) / v.zoom + center;
  const zoom = clamp(v.zoom * (deltaY > 0 ? 0.9 : 1.1), MIN_ZOOM, MAX_ZOOM);
  return {
    zoom,
    panX: canvasX - (anchorX - center) * zoom - center,
    panY: canvasY - (anchorY - center) * zoom - center,
  };
}

/** Paint the image into the canvas under the current view. */
export function drawImage(
  ctx: CanvasRenderingContext2D,
  image: CanvasImageSource,
  v: ImageViewport,
): void {
  const scale = displayScale(v) * v.zoom;
  const center = v.canvas / 2;
  const drawW = v.width * scale;
  const drawH = v.height * scale;
  ctx.drawImage(image, center - drawW / 2 + v.panX, center - drawH / 2 + v.panY, drawW, drawH);
}

/** Decode PNG bytes from a synced trait into a drawable bitmap. */
export function usePngBitmap(
  bytes: DataView | Uint8Array | null | undefined,
): ImageBitmap | HTMLImageElement | null {
  const [image, setImage] = useState<ImageBitmap | HTMLImageElement | null>(null);

  useEffect(() => {
    const raw = bytes ? extractBytes(bytes) : new Uint8Array(0);
    if (raw.length === 0) {
      setImage(null);
      return;
    }
    let cancelled = false;
    const blob = new Blob([raw as unknown as BlobPart], { type: "image/png" });
    if (typeof createImageBitmap === "function") {
      createImageBitmap(blob).then((bitmap) => {
        if (!cancelled) setImage(bitmap);
      });
    } else {
      const url = URL.createObjectURL(blob);
      const element = new Image();
      element.onload = () => {
        if (!cancelled) setImage(element);
        URL.revokeObjectURL(url);
      };
      element.src = url;
    }
    return () => {
      cancelled = true;
    };
  }, [bytes]);

  return image;
}
