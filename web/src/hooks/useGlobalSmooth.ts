/**
 * useGlobalSmooth — global CSS image-rendering toggle for every canvas/img
 * surface in the app.
 *
 * Operator presses `s` anywhere to flip `image-rendering` between
 * `pixelated` (default — microscopists read pixels, not marketing graphics)
 * and `auto` (browser bilinear interpolation — useful when comparing
 * coarse features across panels). State persists in localStorage and
 * broadcasts across the app via a `screen-smooth-changed` window event so
 * every viewer re-reads it without prop drilling.
 *
 * Usage:
 *   const smooth = useGlobalSmooth();
 *   <canvas style={{ imageRendering: smooth ? "auto" : "pixelated" }} />
 */
import { useEffect, useState } from "react";

const KEY = "screen.smooth";
const EVENT = "screen-smooth-changed";

export function readGlobalSmooth(): boolean {
  if (typeof window === "undefined") return false;
  return localStorage.getItem(KEY) === "1";
}

export function setGlobalSmooth(next: boolean): void {
  localStorage.setItem(KEY, next ? "1" : "0");
  window.dispatchEvent(new Event(EVENT));
}

export function toggleGlobalSmooth(): void {
  setGlobalSmooth(!readGlobalSmooth());
}

export function useGlobalSmooth(): boolean {
  const [smooth, setSmooth] = useState<boolean>(readGlobalSmooth);
  useEffect(() => {
    const handler = () => setSmooth(readGlobalSmooth());
    window.addEventListener(EVENT, handler);
    window.addEventListener("storage", handler);
    return () => {
      window.removeEventListener(EVENT, handler);
      window.removeEventListener("storage", handler);
    };
  }, []);
  return smooth;
}

/**
 * `imageRenderingFor(smooth)` — string suitable for CSS `imageRendering`.
 * Use when you have the boolean already and want one-line styling.
 */
export function imageRenderingFor(smooth: boolean): "auto" | "pixelated" {
  return smooth ? "auto" : "pixelated";
}
