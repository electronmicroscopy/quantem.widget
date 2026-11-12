// @ts-nocheck  (node:fs and __dirname read the widget sources; no @types/node here)
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { instantSliderSx } from "./Slider";

const JS_ROOT = path.join(__dirname, "..");

describe("widget sliders", () => {
  it("have no position easing", () => {
    expect(instantSliderSx["& .MuiSlider-thumb, & .MuiSlider-track, & .MuiSlider-valueLabel"]).toEqual({ transition: "none" });
  });

  it("all come from shared/Slider, never straight from MUI", () => {
    const offenders = readdirSync(JS_ROOT, { recursive: true, encoding: "utf8" })
      .filter((file) => /\.(ts|tsx)$/.test(file) && !file.includes("node_modules") && !file.includes(".generated"))
      .filter((file) => path.join(JS_ROOT, file) !== path.join(__dirname, "Slider.tsx"))
      // ShowPDF is outside this pass; its sliders change separately.
      .filter((file) => !file.startsWith("showpdf" + path.sep))
      .filter((file) => {
        const source = readFileSync(path.join(JS_ROOT, file), "utf8");
        return /from\s+["']@mui\/material\/Slider["']/.test(source)
          || /import\s*\{[^}]*\bSlider\b[^}]*\}\s*from\s+["']@mui\/material["']/.test(source);
      });
    expect(offenders).toEqual([]);
  });
});
