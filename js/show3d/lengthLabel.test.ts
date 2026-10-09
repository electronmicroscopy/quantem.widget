import { describe, expect, it } from "vitest";

import { formatLength, readableLength } from "./lengthLabel";

describe("length labels follow the widget's pixel unit", () => {
  it("prints Angstrom data as before: Å below 1 nm, nm from 1 nm", () => {
    expect(formatLength(2.35, "A")).toBe("2.35 Å");
    expect(formatLength(2.35, "Å")).toBe("2.35 Å");
    expect(formatLength(12, "angstrom")).toBe("1.20 nm");
  });

  it("reads nm data in nm, not as Angstrom ten times too small", () => {
    // a 0.149 nm/px image (the widget-export tutorial): a 0.2 nm spacing is 2 Å, a 3 nm one 3 nm
    expect(formatLength(0.2, "nm")).toBe("2.00 Å");
    expect(formatLength(3, "nm")).toBe("3.00 nm");
    expect(readableLength(64 * 0.149, "nm")).toEqual({ value: 64 * 0.149, unit: "nm" });
  });

  it("moves to µm and mm for large lengths and keeps units that are not lengths", () => {
    expect(formatLength(2.5, "um")).toBe("2.50 µm");
    expect(formatLength(5000, "µm")).toBe("5.00 mm");
    expect(formatLength(0.5, "pm")).toBe("0.01 Å");
    expect(formatLength(12.5, "px")).toBe("12.50 px");
    expect(formatLength(3, "mrad")).toBe("3.00 mrad");
  });
});
