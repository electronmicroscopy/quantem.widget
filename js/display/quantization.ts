/** Uint8 scientific-display decoding shared by browser widgets. */

/**
 * The affine map a uint8 display sample decodes with, value = byte * scale + low:
 * a non-finite low reads as 0, a non-finite high as low, and an empty or
 * reversed range decodes every byte to low.
 */
export function uint8Decoding(low: number, high: number): { low: number; scale: number } {
  const finiteLow = Number.isFinite(low) ? low : 0;
  const finiteHigh = Number.isFinite(high) ? high : finiteLow;
  return { low: finiteLow, scale: finiteHigh > finiteLow ? (finiteHigh - finiteLow) / 255 : 0 };
}

/** Decode uint8 display samples using ``value * (high - low) / 255 + low``. */
export function dequantizeUint8(
  values: Uint8Array,
  low: number,
  high: number,
  output: Float32Array = new Float32Array(values.length),
): Float32Array {
  if (output.length < values.length) throw new Error("dequantizeUint8 output is shorter than input");
  const decoding = uint8Decoding(low, high);
  for (let index = 0; index < values.length; index++) {
    output[index] = values[index] * decoding.scale + decoding.low;
  }
  return output;
}
