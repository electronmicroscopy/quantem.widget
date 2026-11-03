import { describe, expect, it } from "vitest";

import { formatScaleLabel, formatZoomLabel } from "./figure";
import { unitSymbol } from "./units";

describe("formatZoomLabel", () => {
  it("uses one decimal and the multiplication sign", () => {
    expect(formatZoomLabel(1)).toBe("1.0×");
    expect(formatZoomLabel(2)).toBe("2.0×");
    expect(formatZoomLabel(5.06)).toBe("5.1×");
  });
});

describe("formatScaleLabel", () => {
  it("prints a length as a whole number in the unit that reads cleanest", () => {
    expect(formatScaleLabel(0.5, "nm")).toBe("5 Å");
    expect(formatScaleLabel(0.005, "nm")).toBe("5 pm");
    expect(formatScaleLabel(20, "Å")).toBe("2 nm");
    expect(formatScaleLabel(5, "angstrom")).toBe("5 Å");
    expect(formatScaleLabel(1500, "um")).toBe("2 mm");
    expect(formatScaleLabel(0.2, "μm")).toBe("200 nm");
  });

  it("rescales every length unit of the table, meters included", () => {
    // these units printed "0.00 m" or kept a decimal in the given unit before
    expect(formatScaleLabel(1e-9, "m")).toBe("1 nm");
    expect(formatScaleLabel(0.002, "mm")).toBe("2 µm");
    expect(formatScaleLabel(0.05, "µm")).toBe("50 nm");
    // the table is in Angstrom, so 1e-7 mm is exactly 1 Å (tests/infrastructure/test_shared_modules.py holds the same cases)
    expect(formatScaleLabel(1e-7, "mm")).toBe("1 Å");
    // the power of ten is exact: Math.pow(10, -4) gave 0.9999999999999999 Å and the label "100 pm"
    expect(formatScaleLabel(1e-4, "um")).toBe("1 Å");
    expect(formatScaleLabel(1.2e-5, "um")).toBe("10 pm");
  });

  it("keeps a unit that is not a length", () => {
    expect(formatScaleLabel(5, "mrad")).toBe("5 mrad");
    expect(formatScaleLabel(0.2, "ps")).toBe("0.20 ps");
    expect(formatScaleLabel(50, "px")).toBe("50 px");
  });
});

describe("unitSymbol", () => {
  it("labels uncalibrated axes px, as every widget does", () => {
    expect(unitSymbol("pixels")).toBe("px");
    expect(formatScaleLabel(20, "pixels")).toBe("20 px");
  });

  it("reads the unit strings a .qem file writes", () => {
    expect(formatScaleLabel(12, "angstrom")).toBe("1 nm");
    expect(formatScaleLabel(0.2, "1/angstrom")).toBe("0.20 Å⁻¹");
    expect(formatScaleLabel(5, "mrad")).toBe("5 mrad");
  });

  it("leaves the label of a frame axis as its author wrote it", () => {
    expect(unitSymbol("frame")).toBe("frame");
    expect(unitSymbol("K")).toBe("K");
  });
});
