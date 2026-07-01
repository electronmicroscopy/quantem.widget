import { describe, expect, it } from "vitest";
import { formatScaleLabel, unitSymbol } from "./figure";

describe("scale bar labels", () => {
  it("renders pixel units as integer px labels", () => {
    expect(unitSymbol("pixels")).toBe("px");
    expect(formatScaleLabel(0.2, "pixels")).toBe("1 px");
    expect(formatScaleLabel(2.1, "pixel")).toBe("2 px");
    expect(formatScaleLabel(50, "px")).toBe("50 px");
  });

  it("keeps calibrated length units on the clean unit ladder", () => {
    expect(formatScaleLabel(0.5, "nm")).toBe("5 Å");
    expect(formatScaleLabel(20, "nm")).toBe("20 nm");
  });
});
