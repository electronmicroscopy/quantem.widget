import { describe, expect, it } from "vitest";
import { committedFitEnd, committedWindowEnd } from "./kwindow";

describe("ShowPDF k-window slider commit", () => {
  const kMinAvail = Math.fround(0.05), kMaxAvail = Math.fround(1.55);  // float32 data range

  it("keeps the follow-fit sentinel for an untouched end clamped by the slider", () => {
    // k_max_fit = 1.55 lies just above the float32 k_max_available: MUI reports
    // the untouched upper handle as kMaxAvail, which is still "left where it was".
    expect(committedWindowEnd(kMaxAvail, 0, 1.55, kMinAvail, kMaxAvail)).toBe(0);
    expect(committedWindowEnd(kMinAvail, 0, 0.05, kMinAvail, kMaxAvail)).toBe(0);
    expect(committedWindowEnd(1.2, 0, 1.2, kMinAvail, kMaxAvail)).toBe(0);
    expect(committedWindowEnd(0.4, 0.4, 0.3, kMinAvail, kMaxAvail)).toBe(0.4);
  });

  it("keeps a negative setting, which follows the fit range as 0 does", () => {
    // Python accepts any finite end and treats one <= 0 as unset: the slider shows the fit end.
    expect(committedWindowEnd(0.3, -0.5, 0.3, kMinAvail, kMaxAvail)).toBe(-0.5);
    expect(committedWindowEnd(1.2, -2, 1.2, kMinAvail, kMaxAvail)).toBe(-2);
    // a moved end is explicit, also at the slider's min (not the clamp of the setting)
    expect(committedWindowEnd(kMinAvail, -0.5, 0.3, kMinAvail, kMaxAvail)).toBe(kMinAvail);
    expect(committedWindowEnd(0.31, -0.5, 0.3, kMinAvail, kMaxAvail)).toBe(0.31);
  });

  it("keeps the setting of an untouched end off the slider's step grid", () => {
    // fit max after keyboard steps (1.2 - 20 * 0.01), or set from Python
    expect(committedWindowEnd(0.9999999999999998, 0, 0.9999999999999998, kMinAvail, kMaxAvail)).toBe(0);
    expect(committedWindowEnd(1.234, 0, 1.234, kMinAvail, kMaxAvail)).toBe(0);
    // an explicit off-grid end keeps its exact value
    expect(committedWindowEnd(1.234, 1.234, 1.2, kMinAvail, kMaxAvail)).toBe(1.234);
  });

  it("takes a moved end as an explicit bound", () => {
    expect(committedWindowEnd(0.06, 0, 0.3, kMinAvail, kMaxAvail)).toBe(0.06);
    expect(committedWindowEnd(1.3, 0, 1.55, kMinAvail, kMaxAvail)).toBe(1.3);
    // one step from an off-grid end: a keyboard step, and a drag to the next grid value
    expect(committedWindowEnd(1.224, 0, 1.234, kMinAvail, kMaxAvail)).toBe(1.224);
    expect(committedWindowEnd(1.23, 0, 1.234, kMinAvail, kMaxAvail)).toBe(1.23);
    expect(committedWindowEnd(1.24, 0, 1.234, kMinAvail, kMaxAvail)).toBe(1.24);
    expect(committedWindowEnd(1.0, 0, 1.01, kMinAvail, kMaxAvail)).toBe(1.0);
    // to the slider's max (End, or a drag past it) from an end less than a step below it
    expect(committedWindowEnd(kMaxAvail, 1.546, 1.2, kMinAvail, kMaxAvail)).toBe(kMaxAvail);
    expect(committedWindowEnd(kMaxAvail, 0, 1.546, kMinAvail, kMaxAvail)).toBe(kMaxAvail);
  });
});

describe("ShowPDF k-fit slider commit", () => {
  const kMinAvail = Math.fround(0.05), kMaxAvail = Math.fround(1.55);  // float32 data range

  it("keeps the value of an untouched end clamped by the slider", () => {
    // k_max_fit = 1.55 lies just above the float32 k_max_available, k_min_fit = 0.05 just below
    expect(committedFitEnd(kMaxAvail, 1.55, kMinAvail, kMaxAvail)).toBe(1.55);
    expect(committedFitEnd(kMinAvail, 0.05, kMinAvail, kMaxAvail)).toBe(0.05);
    expect(committedFitEnd(1.234, 1.234, kMinAvail, kMaxAvail)).toBe(1.234);
  });

  it("takes a moved end", () => {
    expect(committedFitEnd(1.54, 1.55, kMinAvail, kMaxAvail)).toBe(1.54);
    expect(committedFitEnd(0.31, 0.3, kMinAvail, kMaxAvail)).toBe(0.31);
    expect(committedFitEnd(kMaxAvail, 1.2, kMinAvail, kMaxAvail)).toBe(kMaxAvail);
  });
});
