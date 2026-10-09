/** Uint8 scientific-display decoding shared by browser widgets. */

/** Decode uint8 display samples using ``value * (high - low) / 255 + low``. */
export function dequantizeUint8(
  values: Uint8Array,
  low: number,
  high: number,
  output: Float32Array = new Float32Array(values.length),
): Float32Array {
  if (output.length < values.length) throw new Error("dequantizeUint8 output is shorter than input");
  const finiteLow = Number.isFinite(low) ? low : 0;
  const finiteHigh = Number.isFinite(high) ? high : finiteLow;
  const scale = finiteHigh > finiteLow ? (finiteHigh - finiteLow) / 255 : 0;
  for (let index = 0; index < values.length; index++) {
    output[index] = values[index] * scale + finiteLow;
  }
  return output;
}
