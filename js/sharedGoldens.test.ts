import { describe, expect, it } from "vitest";

import goldens from "./display/goldens.json";
import { reciprocalCoordinatesFromShiftedOffset } from "./display/fft";
import { dequantizeUint8 } from "./display/quantization";

describe("display numerical goldens", () => {
  for (const testCase of goldens.quantized) {
    it(`decodes uint8 ${testCase.name}`, () => {
      const actual = dequantizeUint8(Uint8Array.from(testCase.bytes), testCase.low, testCase.high);
      testCase.expected.forEach((expected, index) => expect(actual[index]).toBeCloseTo(expected, 5));
    });
  }

  it("uses the same nonfinite and reversed quantization-range policy", () => {
    expect(Array.from(dequantizeUint8(Uint8Array.of(0, 255), Number.NaN, 7))).toEqual([0, 7]);
    expect(Array.from(dequantizeUint8(Uint8Array.of(0, 255), -2, Number.POSITIVE_INFINITY))).toEqual([-2, -2]);
    expect(Array.from(dequantizeUint8(Uint8Array.of(0, 255), 4, -4))).toEqual([4, 4]);
  });

  for (const testCase of goldens.reciprocal) {
    it(`converts reciprocal coordinates ${testCase.name}`, () => {
      const actual = reciprocalCoordinatesFromShiftedOffset(
        testCase.row_offset,
        testCase.column_offset,
        testCase.rows,
        testCase.columns,
        testCase.row_sampling,
        testCase.column_sampling,
      );
      expect(actual.rowFrequency).toBeCloseTo(testCase.expected[0]!, 12);
      expect(actual.columnFrequency).toBeCloseTo(testCase.expected[1]!, 12);
      expect(actual.spatialFrequency).toBeCloseTo(testCase.expected[2]!, 12);
      if (testCase.expected[3] == null) expect(actual.dSpacing).toBeNull();
      else expect(actual.dSpacing).toBeCloseTo(testCase.expected[3], 12);
    });
  }
});
