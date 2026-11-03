function asciiBytes(value: string): Uint8Array {
  const out = new Uint8Array(value.length);
  for (let i = 0; i < value.length; i++) {
    out[i] = value.charCodeAt(i) & 0xff;
  }
  return out;
}

function u16Bytes(value: number): Uint8Array {
  const clamped = Math.max(0, Math.min(65535, Math.round(value)));
  return new Uint8Array([clamped & 0xff, (clamped >> 8) & 0xff]);
}

function concatUint8(parts: Uint8Array[]): Uint8Array {
  const length = parts.reduce((total, part) => total + part.byteLength, 0);
  const out = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.byteLength;
  }
  return out;
}

/**
 * The fixed 256-color GIF palette: a 6x6x6 RGB cube (levels 0, 51, ..., 255)
 * at indices 0-215, then a 40-step gray ramp. Fixed so every frame shares one
 * global color table and quantizeRgbaForBrowserGif can index it arithmetically.
 */
const PALETTE = (() => {
  const colors = new Uint8Array(256 * 3);
  let index = 0;
  for (let red = 0; red < 6; red++) {
    for (let green = 0; green < 6; green++) {
      for (let blue = 0; blue < 6; blue++) {
        colors.set([red * 51, green * 51, blue * 51], 3 * index++);
      }
    }
  }
  const grayCount = 256 - index;
  for (let gray = 0; index < 256; index++, gray++) {
    colors.fill(Math.round((gray / (grayCount - 1)) * 255), 3 * index, 3 * index + 3);
  }
  return colors;
})();

/**
 * Palette index of the gray nearest each 8-bit level, among the ramp and the
 * cube's six grays (the lower index on a tie). Gray pixels, most microscopy,
 * take these 44 levels; the cube alone would band them to 6.
 */
const NEAREST_GRAY_INDEX = (() => {
  const grays: number[] = [];
  for (let index = 0; index < 256; index++) {
    if (PALETTE[3 * index] === PALETTE[3 * index + 1] && PALETTE[3 * index] === PALETTE[3 * index + 2]) grays.push(index);
  }
  return Uint8Array.from({ length: 256 }, (_, level) => grays.reduce(
    (best, index) => (Math.abs(PALETTE[3 * index] - level) < Math.abs(PALETTE[3 * best] - level) ? index : best),
  ));
})();

/** A channel composited onto white, (value * alpha + 255 * (255 - alpha)) / 255, since GIF has no alpha. */
function overWhite(value: number, alpha: number): number {
  return alpha === 255 ? value : Math.round((value * alpha + 255 * (255 - alpha)) / 255);
}

/**
 * Palette indices for RGBA pixels: each channel is composited onto white; gray
 * pixels take the nearest palette gray, other colors the nearest of the six
 * cube levels per channel, index = 36 red + 6 green + blue.
 */
export function quantizeRgbaForBrowserGif(rgba: Uint8ClampedArray): Uint8Array {
  const out = new Uint8Array(Math.floor(rgba.length / 4));
  for (let pixel = 0, offset = 0; pixel < out.length; pixel++, offset += 4) {
    const alpha = rgba[offset + 3];
    const red = overWhite(rgba[offset], alpha);
    const green = overWhite(rgba[offset + 1], alpha);
    const blue = overWhite(rgba[offset + 2], alpha);
    if (red === green && green === blue) {
      out[pixel] = NEAREST_GRAY_INDEX[red];
      continue;
    }
    out[pixel] = Math.round(red / 51) * 36 + Math.round(green / 51) * 6 + Math.round(blue / 51);
  }
  return out;
}

/**
 * GIF LZW with 8-bit pixel codes: the code size grows from 9 to 12 bits as the
 * string table fills, and a clear code restarts the table at 4096 entries. A
 * string table, not one code per pixel, keeps uniform and repeating regions small.
 */
function lzwEncode(indices: Uint8Array): Uint8Array {
  const clearCode = 256;
  const endCode = clearCode + 1;
  const bytes: number[] = [];
  let codeSize = 9;
  let nextCode = endCode + 1;
  let table = new Map<number, number>();
  let bitBuffer = 0;
  let bitCount = 0;
  const writeCode = (code: number) => {
    bitBuffer |= code << bitCount;
    bitCount += codeSize;
    while (bitCount >= 8) {
      bytes.push(bitBuffer & 0xff);
      bitBuffer >>= 8;
      bitCount -= 8;
    }
  };
  writeCode(clearCode);
  let prefix = indices[0];
  for (let position = 1; position < indices.length; position++) {
    const pixel = indices[position];
    const key = (prefix << 8) | pixel;
    const known = table.get(key);
    if (known !== undefined) {
      prefix = known;
      continue;
    }
    writeCode(prefix);
    if (nextCode === 4096) {
      writeCode(clearCode);
      table = new Map();
      nextCode = endCode + 1;
      codeSize = 9;
    } else {
      // the decoder widens its codes once its table reaches the current code range
      if (nextCode >= 1 << codeSize) codeSize++;
      table.set(key, nextCode++);
    }
    prefix = pixel;
  }
  writeCode(prefix);
  writeCode(endCode);
  if (bitCount > 0) bytes.push(bitBuffer & 0xff);
  return new Uint8Array(bytes);
}

function pushSubBlocks(parts: Uint8Array[], data: Uint8Array): void {
  for (let offset = 0; offset < data.length; offset += 255) {
    const chunk = data.subarray(offset, Math.min(offset + 255, data.length));
    parts.push(new Uint8Array([chunk.length]), chunk);
  }
  parts.push(new Uint8Array([0]));
}

/**
 * A looping GIF89a of palette-indexed frames (one global palette, NETSCAPE2.0
 * loop block, `delayCs` hundredths of a second per frame). Built in the browser
 * so a standalone HTML page can export a GIF without a kernel.
 */
export function encodeIndexedGif(
  width: number,
  height: number,
  frames: Uint8Array[],
  delayCs: number,
): Uint8Array {
  if (width <= 0 || height <= 0 || frames.length === 0) {
    throw new Error("GIF export needs at least one non-empty frame.");
  }

  const pixelCount = width * height;
  for (const frame of frames) {
    if (frame.length !== pixelCount) {
      throw new Error(`GIF frame has ${frame.length} pixels; expected ${pixelCount}.`);
    }
  }

  const parts: Uint8Array[] = [
    asciiBytes("GIF89a"),
    u16Bytes(width),
    u16Bytes(height),
    new Uint8Array([0xf7, 0, 0]),
    PALETTE,
    new Uint8Array([0x21, 0xff, 0x0b]),
    asciiBytes("NETSCAPE2.0"),
    new Uint8Array([0x03, 0x01, 0x00, 0x00, 0x00]),
  ];
  const delay = Math.max(1, Math.min(65535, Math.round(delayCs)));
  for (const frame of frames) {
    parts.push(new Uint8Array([0x21, 0xf9, 0x04, 0x04]));
    parts.push(u16Bytes(delay));
    parts.push(new Uint8Array([0, 0, 0x2c, 0, 0, 0, 0]));
    parts.push(u16Bytes(width), u16Bytes(height));
    parts.push(new Uint8Array([0, 8]));
    pushSubBlocks(parts, lzwEncode(frame));
  }
  parts.push(new Uint8Array([0x3b]));
  return concatUint8(parts);
}
