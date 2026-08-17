let show3dGifPalette: Uint8Array | null = null;

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

function palette(): Uint8Array {
  if (show3dGifPalette) return show3dGifPalette;

  const colors = new Uint8Array(256 * 3);
  let index = 0;
  for (let red = 0; red < 6; red++) {
    for (let green = 0; green < 6; green++) {
      for (let blue = 0; blue < 6; blue++) {
        const offset = index * 3;
        colors[offset] = red * 51;
        colors[offset + 1] = green * 51;
        colors[offset + 2] = blue * 51;
        index++;
      }
    }
  }

  const grayCount = 256 - index;
  for (let gray = 0; index < 256; index++, gray++) {
    const value = grayCount <= 1 ? 0 : Math.round((gray / (grayCount - 1)) * 255);
    const offset = index * 3;
    colors[offset] = value;
    colors[offset + 1] = value;
    colors[offset + 2] = value;
  }
  show3dGifPalette = colors;
  return colors;
}

export function quantizeRgbaForBrowserGif(rgba: Uint8ClampedArray): Uint8Array {
  const out = new Uint8Array(Math.floor(rgba.length / 4));
  for (let pixel = 0, offset = 0; pixel < out.length; pixel++, offset += 4) {
    const alpha = rgba[offset + 3];
    const red = alpha === 255
      ? rgba[offset]
      : Math.round((rgba[offset] * alpha + 255 * (255 - alpha)) / 255);
    const green = alpha === 255
      ? rgba[offset + 1]
      : Math.round((rgba[offset + 1] * alpha + 255 * (255 - alpha)) / 255);
    const blue = alpha === 255
      ? rgba[offset + 2]
      : Math.round((rgba[offset + 2] * alpha + 255 * (255 - alpha)) / 255);
    const redBin = Math.max(0, Math.min(5, Math.round(red / 51)));
    const greenBin = Math.max(0, Math.min(5, Math.round(green / 51)));
    const blueBin = Math.max(0, Math.min(5, Math.round(blue / 51)));
    out[pixel] = redBin * 36 + greenBin * 6 + blueBin;
  }
  return out;
}

function lzwEncode(indices: Uint8Array): Uint8Array {
  const minimumCodeSize = 8;
  const clearCode = 1 << minimumCodeSize;
  const endCode = clearCode + 1;
  const codeSize = minimumCodeSize + 1;
  const bytes: number[] = [];
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

  // Clear before the decoder grows beyond 9-bit codes. Browser exports favor
  // a simple, dependable stream over maximum compression.
  writeCode(clearCode);
  let sinceClear = 0;
  for (const index of indices) {
    if (sinceClear >= 250) {
      writeCode(clearCode);
      sinceClear = 0;
    }
    writeCode(index);
    sinceClear++;
  }
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
    palette(),
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
