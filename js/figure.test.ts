import { describe, expect, it } from "vitest";

import { formatZoomLabel } from "./figure";

describe("formatZoomLabel", () => {
  it("uses one decimal and the multiplication sign", () => {
    expect(formatZoomLabel(1)).toBe("1.0×");
    expect(formatZoomLabel(2)).toBe("2.0×");
    expect(formatZoomLabel(5.06)).toBe("5.1×");
  });
});
