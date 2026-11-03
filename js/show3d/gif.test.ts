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

/** The global color table and the palette indices of the first frame of a GIF from encodeIndexedGif. */
function decodeGif(bytes: Uint8Array): { width: number; height: number; palette: Uint8Array; indices: number[] } {
  const width = bytes[6] | (bytes[7] << 8);
  const height = bytes[8] | (bytes[9] << 8);
  const palette = bytes.slice(13, 13 + 768);
  let offset = 13 + 768;
  while (bytes[offset] === 0x21) {
    offset += 2;
    while (bytes[offset] !== 0) offset += bytes[offset] + 1;
    offset += 1;
  }
  expect(bytes[offset]).toBe(0x2c);
  offset += 10;
  const minimumCodeSize = bytes[offset++];
  const codes: number[] = [];
  while (bytes[offset] !== 0) {
    codes.push(...bytes.subarray(offset + 1, offset + 1 + bytes[offset]));
    offset += bytes[offset] + 1;
  }
  return { width, height, palette, indices: lzwDecode(Uint8Array.from(codes), minimumCodeSize) };
}

/** GIF LZW decoding: variable code size growing to 12 bits, clear and end codes. */
function lzwDecode(codes: Uint8Array, minimumCodeSize: number): number[] {
  const clearCode = 1 << minimumCodeSize;
  const endCode = clearCode + 1;
  let codeSize = minimumCodeSize + 1;
  let table: number[][] = [];
  const out: number[] = [];
  let previous: number[] | null = null;
  let bitPosition = 0;
  while (bitPosition + codeSize <= codes.length * 8) {
    let code = 0;
    for (let bit = 0; bit < codeSize; bit++, bitPosition++) code |= ((codes[bitPosition >> 3] >> (bitPosition & 7)) & 1) << bit;
    if (code === clearCode) {
      table = Array.from({ length: clearCode + 2 }, (_, index) => [index]);
      codeSize = minimumCodeSize + 1;
      previous = null;
      continue;
    }
    if (code === endCode) break;
    const entry: number[] = code < table.length ? table[code] : [...previous!, previous![0]];
    out.push(...entry);
    if (previous) table.push([...previous, entry[0]]);
    if (table.length === 1 << codeSize && codeSize < 12) codeSize++;
    previous = entry;
  }
  return out;
}

describe("browser GIF export of gray and large frames", () => {
  it("keeps a gray ramp within 3 levels, on the palette's gray entries", () => {
    const rgba = new Uint8ClampedArray(256 * 4);
    for (let level = 0; level < 256; level++) rgba.set([level, level, level, 255], 4 * level);
    const { palette, indices } = decodeGif(encodeIndexedGif(256, 1, [quantizeRgbaForBrowserGif(rgba)], 10));
    const shown = indices.map((index) => [palette[3 * index], palette[3 * index + 1], palette[3 * index + 2]]);
    expect(shown.every(([red, green, blue]) => red === green && green === blue)).toBe(true);
    expect(Math.max(...shown.map(([gray], level) => Math.abs(gray - level)))).toBeLessThanOrEqual(3);
    expect(new Set(shown.map(([gray]) => gray)).size).toBe(44);
  });

  it("round-trips frames longer than the 4096-entry code table, with growing codes", () => {
    let state = 12345;
    const noise = Uint8Array.from({ length: 300 * 200 }, () => (state = (state * 1103515245 + 12345) >>> 0) >>> 24);
    const flat = new Uint8Array(300 * 200).fill(216 + 20);
    for (const frame of [noise, flat]) {
      const decoded = decodeGif(encodeIndexedGif(300, 200, [frame], 10));
      expect([decoded.width, decoded.height]).toEqual([300, 200]);
      expect(Uint8Array.from(decoded.indices)).toEqual(frame);
    }
    // a uniform frame needs only a few hundred growing codes, not one 9-bit code per pixel (68 kB)
    expect(encodeIndexedGif(300, 200, [flat], 10).length).toBeLessThan(2000);
  });
});
