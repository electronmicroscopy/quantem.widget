import { describe, expect, it } from "vitest";

import { cropMaskedRegion, sampleLineProfile, sampleLineProfileUint8 } from "./display/geometry";

const image = Float32Array.from({ length: 30 }, (_, index) => index);

describe("shared masked crop reference", () => {
  it("uses floor/ceil half-extents and exclusive rectangle bounds", () => {
    const crop = cropMaskedRegion(image, 6, 5, {
      row: 2,
      col: 3,
      shape: "rectangle",
      radius: 0,
      width: 3,
      height: 3,
    });
    expect(crop && [crop.cropW, crop.cropH]).toEqual([4, 4]);
    expect(Array.from(crop!.cropped)).toEqual([
      1, 2, 3, 4,
      7, 8, 9, 10,
      13, 14, 15, 16,
      19, 20, 21, 22,
    ]);
  });

  it("zeros samples outside the inclusive outer disk and preserves selected nonfinite values", () => {
    const withNonfinite = image.slice();
    withNonfinite[14] = NaN;
    const crop = cropMaskedRegion(withNonfinite, 6, 5, {
      row: 2,
      col: 2,
      shape: "circle",
      radius: 2,
      width: 0,
      height: 0,
    });
    expect(crop && [crop.cropW, crop.cropH]).toEqual([4, 4]);
    const expected = [
      0, 0, 2, 0,
      0, 7, 8, 9,
      12, 13, NaN, 15,
      0, 19, 20, 21,
    ];
    const actual = Array.from(crop!.cropped);
    expected.forEach((value, index) => {
      if (Number.isNaN(value)) expect(actual[index]).toBeNaN();
      else expect(actual[index]).toBe(value);
    });
  });

  it("keeps annular FFT crops equivalent to their outer circle and clips image edges", () => {
    const circle = cropMaskedRegion(image, 6, 5, {
      row: 0.5, col: 0.5, shape: "circle", radius: 2, width: 0, height: 0,
    });
    const annular = cropMaskedRegion(image, 6, 5, {
      row: 0.5, col: 0.5, shape: "annular", radius: 2, width: 0, height: 0,
    });
    expect(annular).toEqual(circle);
    expect(circle && [circle.cropW, circle.cropH]).toEqual([3, 3]);
    expect(cropMaskedRegion(image, 6, 5, {
      row: 0, col: 0, shape: "circle", radius: 0.5, width: 0, height: 0,
    })).toBeNull();
  });
});

describe("shared bilinear line profile reference", () => {
  it("uses ceil(length) endpoint-inclusive samples on a nonsquare image", () => {
    const ramp = Float32Array.from({ length: 12 }, (_, index) => Math.floor(index / 4) * 10 + index % 4);
    const values = sampleLineProfile(ramp, 4, 3, 0, 0, 2, 3);
    expect(values).toHaveLength(4);
    [0, 23 / 3, 46 / 3, 23].forEach((value, index) => {
      expect(values[index]).toBeCloseTo(value, 5);
    });
  });

  it("averages perpendicular profiles and keeps nearest-edge sampling", () => {
    const rows = Float32Array.from({ length: 12 }, (_, index) => Math.floor(index / 4));
    for (const value of sampleLineProfile(rows, 4, 3, 0, -1, 0, 3, 3)) {
      expect(value).toBeCloseTo(1 / 3, 6);
    }
  });

  it("rounds a fractional width to whole lines, as the WebGPU kernel does", () => {
    const ramp = Float32Array.from({ length: 36 }, (_, index) => Math.floor(index / 6) ** 2);
    expect(Array.from(sampleLineProfile(ramp, 6, 6, 2, 0, 2, 5, 2.6))).toEqual(Array.from(sampleLineProfile(ramp, 6, 6, 2, 0, 2, 5, 3)));
    expect(Array.from(sampleLineProfile(ramp, 6, 6, 2, 0, 2, 5, 1.4))).toEqual(Array.from(sampleLineProfile(ramp, 6, 6, 2, 0, 2, 5, 1)));
  });

  it("preserves non-finite source evidence", () => {
    const values = new Float32Array(9).fill(1);
    values[0] = NaN;
    expect(sampleLineProfile(values, 3, 3, 0, 0, 2, 2)[0]).toBeNaN();
  });
});

describe("quantized profile reference", () => {
  it("samples uint8+range without materializing a float image", () => {
    const encoded = Uint8Array.from({ length: 12 }, (_, index) => index * 20);
    const decoded = Float32Array.from(encoded, value => value * 2 / 255 - 1);
    const expected = sampleLineProfile(decoded, 4, 3, -0.25, 0.5, 2.25, 4.5, 3);
    const actual = sampleLineProfileUint8(encoded, -1, 1, 4, 3, -0.25, 0.5, 2.25, 4.5, 3);
    expected.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, 6));
  });

  it("applies the finite collapsed-range policy while sampling uint8", () => {
    const encoded = Uint8Array.of(0, 255, 0, 255);
    expect(Array.from(sampleLineProfileUint8(encoded, Number.NaN, 7, 2, 2, 0, 0, 1, 1))).toEqual([0, 7]);
    expect(Array.from(sampleLineProfileUint8(encoded, 4, -4, 2, 2, 0, 0, 1, 1))).toEqual([4, 4]);
  });
});
