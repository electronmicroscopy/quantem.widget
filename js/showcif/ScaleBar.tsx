import * as React from "react";
import { scaleBarWidth } from "./geometry";

/** WCAG relative luminance of an sRGB colour given as 0 to 255 per channel. */
function luminance(rgb: number[]): number {
  const [r, g, b] = rgb.map((channel) => {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * Physical length overlay, independent of numerical data and color mapping.
 * `backdrop` is the colour behind the bar on a WebGPU scene: the near-black
 * atom scene, or the low end of the potential colormap, where vacuum sits.
 * The bar is black or white, whichever has the higher WCAG contrast with it
 * (white on the atom scene, black on twilight's near-white vacuum), and the
 * line and label carry a halo of the other so the bar still reads where atoms
 * take the colormap's other end. Without a WebGPU scene (`backdrop` null) the
 * panel shows the widget background, so the bar takes the theme text colour
 * with a halo of the background.
 */
export function ScaleBar({ span, backdrop }: { span: number; backdrop: number[] | null }) {
  const length = scaleBarWidth(span);
  const backdropLuminance = backdrop ? luminance(backdrop) : 0;
  const blackInk = !!backdrop && (backdropLuminance + 0.05) / 0.05 > 1.05 / (backdropLuminance + 0.05);
  const ink = !backdrop ? "var(--cif-text)" : blackInk ? "black" : "white";
  const halo = !backdrop ? "var(--cif-bg)" : blackInk ? "white" : "black";
  return (
    <div
      data-scale-bar="true"
      style={{
        position: "absolute",
        left: "5%",
        bottom: "5%",
        width: `${(length / span) * 100}%`,
        borderBottom: `3px solid ${ink}`,
        color: ink,
        fontSize: 12,
        textAlign: "left",
        filter: `drop-shadow(0 0 1px ${halo}) drop-shadow(0 0 1px ${halo})`,
        pointerEvents: "none",
      }}
    >
      {length >= 10 ? `${length / 10} nm` : `${length} Å`}
    </div>
  );
}
