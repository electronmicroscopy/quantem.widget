import { describe, expect, it } from "vitest";

import { encodeIndexedGif, quantizeRgbaForBrowserGif } from "./gif";

function ascii(bytes: Uint8Array): string {
  return String.fromCharCode(...bytes);
}

function countSequence(bytes: Uint8Array, sequence: number[]): number {
  let count = 0;
  for (let offset = 0; offset <= bytes.length - sequence.length; offset++) {
    if (sequence.every((value, index) => bytes[offset + index] === value)) {
      count++;
    }
  }
  return count;
}

describe("browser GIF export", () => {
  it("quantizes opaque colors and composites transparency onto white", () => {
    const rgba = new Uint8ClampedArray([
      255, 0, 0, 255,
      0, 255, 0, 255,
      0, 0, 255, 255,
      0, 0, 0, 0,
    ]);

    expect(Array.from(quantizeRgbaForBrowserGif(rgba))).toEqual([
      180,
      30,
      5,
      215,
    ]);
  });

  it("writes a deterministic looping GIF with the requested dimensions", () => {
    const frames = [
      new Uint8Array([0, 1, 2, 3, 4, 5]),
      new Uint8Array([5, 4, 3, 2, 1, 0]),
    ];
    const first = encodeIndexedGif(3, 2, frames, 12.4);
    const second = encodeIndexedGif(3, 2, frames, 12.4);

    expect(first).toEqual(second);
    expect(ascii(first.subarray(0, 6))).toBe("GIF89a");
    expect(Array.from(first.subarray(6, 10))).toEqual([3, 0, 2, 0]);
    expect(ascii(first).includes("NETSCAPE2.0")).toBe(true);
    expect(first[first.length - 1]).toBe(0x3b);
    expect(countSequence(first, [0x21, 0xf9, 0x04, 0x04])).toBe(2);
  });

  it("rejects empty exports and incorrectly sized frames", () => {
    expect(() => encodeIndexedGif(0, 2, [new Uint8Array(0)], 10)).toThrow(
      "at least one non-empty frame",
    );
    expect(() => encodeIndexedGif(2, 2, [new Uint8Array(3)], 10)).toThrow(
      "GIF frame has 3 pixels; expected 4",
    );
  });
});
