import * as React from "react";
import { scaleBarWidth } from "./geometry";
/** Physical length overlay, independent of numerical data and color mapping. */
export function ScaleBar({ span }: { span: number }) {
  const length = scaleBarWidth(span);
  return (
    <div
      data-scale-bar="true"
      style={{
        position: "absolute",
        left: "5%",
        bottom: "5%",
        width: `${(length / span) * 100}%`,
        borderBottom: "3px solid white",
        color: "white",
        fontSize: 12,
        textAlign: "left",
        textShadow: "0 1px 3px black, 0 0 2px black",
        pointerEvents: "none",
      }}
    >
      {length >= 10 ? `${length / 10} nm` : `${length} Å`}
    </div>
  );
}
