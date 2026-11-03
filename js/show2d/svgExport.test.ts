import { describe, expect, it } from "vitest";

import { svgInsetPlotElement, svgPanelAnnotationElement } from "./svgExport";

describe("Show2D SVG export", () => {
  it("keeps a semi-transparent annotation background semi-transparent", () => {
    const svg = svgPanelAnnotationElement({ text: "scan 3" }, 0, 0, 200, 100);
    // The badge default is rgba(0,0,0,0.72) on screen; SVG 1.1 takes the alpha as an opacity.
    expect(svg).toContain('fill="rgb(0, 0, 0)" fill-opacity="0.72"');
    expect(svg).toContain('stroke="rgb(255, 255, 255)" stroke-opacity="0.5"');
    expect(svg).not.toContain("rgba(");
    // Top-left badge: the text sits inside its box, 6 px in from the left edge.
    const box = svg.match(/<rect x="([\d.]+)" y="([\d.]+)" width="[\d.]+" height="([\d.]+)"/)!.slice(1).map(Number);
    const text = svg.match(/<text x="([\d.]+)" y="([\d.]+)"/)!.slice(1).map(Number);
    expect(text[0]).toBe(box[0] + 6);
    expect(text[1]).toBeGreaterThan(box[1]);
    expect(text[1]).toBeLessThan(box[1] + box[2]);
  });

  it("draws an inset plot's ticks, labels, corners and panel index like the canvas", () => {
    const svg = svgInsetPlotElement({
      x: [0, 2, 4, 6, 8, 10],
      y: [0, 0.4, 0.9, 1, 0.6, 0.2],
      show_ticks: true,
      xticks: [0, 5, 10],
      yticks: [0, 1],
      xlabel: "q (1/nm)",
      ylabel: "I",
      legend: "data",
      legend_position: "bottom-right",
      annotation: "fit",
      show_panel_index: true,
      background: "rgba(20,20,20,0.5)",
    }, 0, 0, 400, 300, "#4fc3f7", false, 2, "show2d-svg-inset-0");
    const texts = Array.from(svg.matchAll(/<text [^>]*>([^<]*)<\/text>/g), (match) => match[1]);
    expect(texts).toEqual(["0", "5", "10", "0", "1", "data", "data", "fit", "fit", "q (1/nm)", "I", "3"]);
    expect(svg).toMatch(/<text [^>]*text-anchor="end"[^>]*fill="#4fc3f7"[^>]*>data<\/text>/);
    expect(svg).toContain("rotate(-90)");
    expect(svg).toContain('fill="rgb(20, 20, 20)" fill-opacity="0.5"');
    expect(svg).toContain('clip-path="url(#show2d-svg-inset-0)"');
  });
});
