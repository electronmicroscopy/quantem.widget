import { describe, expect, it } from "vitest";

import { fft2d } from "./fft";
import { estimateSliceAlignment, median } from "./sliceAlignment";

/**
 * Mirrors `_texture` / `_fractionally_shifted_stack` in
 * tests/test_show3dslices_slice_alignment.py: a fixed textured base image is
 * shifted by an exact sub-pixel amount per slice with an FFT phase ramp, so the
 * drift the estimator should recover is known analytically rather than measured.
 */
function texture(size: number): Float32Array {
  const image = new Float32Array(size * size);
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const blobA = 4.0 * Math.exp(-(((row - size * 0.35) ** 2) + ((col - size * 0.42) ** 2)) / 80.0);
      const blobB = 2.5 * Math.exp(-(((row - size * 0.68) ** 2) + ((col - size * 0.73) ** 2)) / 55.0);
      image[row * size + col] =
        Math.sin(row / 3.1)
        + 0.7 * Math.cos(col / 4.7)
        + 0.35 * Math.sin((row + col) / 5.3)
        + blobA
        + blobB;
    }
  }
  return image;
}

function shiftedStack(
  size: number, slices: number, rowDrift: number, colDrift: number,
): Float32Array {
  const base = texture(size);
  const real = Float32Array.from(base);
  const imag = new Float32Array(size * size);
  fft2d(real, imag, size, size, false);
  const stack = new Float32Array(slices * size * size);
  for (let z = 0; z < slices; z++) {
    const shiftedReal = new Float32Array(size * size);
    const shiftedImag = new Float32Array(size * size);
    for (let row = 0; row < size; row++) {
      const rowFreq = (row >= (size + 1) >> 1 ? row - size : row) / size;
      for (let col = 0; col < size; col++) {
        const colFreq = (col >= (size + 1) >> 1 ? col - size : col) / size;
        const angle = -2 * Math.PI * (rowFreq * z * rowDrift + colFreq * z * colDrift);
        const cosine = Math.cos(angle);
        const sine = Math.sin(angle);
        const i = row * size + col;
        shiftedReal[i] = real[i] * cosine - imag[i] * sine;
        shiftedImag[i] = real[i] * sine + imag[i] * cosine;
      }
    }
    fft2d(shiftedReal, shiftedImag, size, size, true);
    stack.set(shiftedReal, z * size * size);
  }
  return stack;
}

describe("estimateSliceAlignment", () => {
  // The expected numbers below are what `Show3DSlices.estimate_slice_alignment`
  // returns for the identical stack, not the nominal drift that built it. Both
  // implementations sit slightly inside the nominal shift because the Hann
  // window tapers the drifted content - which is why the Python test allows
  // +-0.4 on the col slope. Freezing the kernel's own numbers makes this a
  // parity test: the browser estimate must not diverge from the kernel estimate.
  it("matches the Python estimate for a whole-pixel drift", async () => {
    const size = 64;
    const slices = 7;
    const stack = shiftedStack(size, slices, 1, -2);

    const result = await estimateSliceAlignment(stack, size, size, slices, null, null);

    // Content drifts row +1 / col -2 per deeper slice, so the correction to
    // apply is the opposite shift - the same convention the Python test asserts.
    expect(result.rowShiftPxPerSlice).toBeCloseTo(-0.95, 3);
    expect(result.colShiftPxPerSlice).toBeCloseTo(1.6857, 3);
    expect(result.fitR2.row).toBeGreaterThan(0.9);
    expect(result.fitR2.col).toBeGreaterThan(0.9);
    expect(result.adjacentShiftPx).toHaveLength(slices - 1);
    expect(result.backend).toBe("cpu");
  });

  it("matches the Python estimate for a fractional drift", async () => {
    const size = 64;
    const slices = 7;
    const stack = shiftedStack(size, slices, 0.75, -0.4);

    const result = await estimateSliceAlignment(stack, size, size, slices, null, null);

    // A whole-pixel-only estimator would return 0 here, so these also prove the
    // upsampled-DFT refinement runs.
    expect(result.rowShiftPxPerSlice).toBeCloseTo(-0.7, 3);
    expect(result.colShiftPxPerSlice).toBeCloseTo(0.35, 3);
  });

  it("reports a zero slope for a stack with no drift", async () => {
    const size = 64;
    const slices = 5;
    const stack = shiftedStack(size, slices, 0, 0);

    const result = await estimateSliceAlignment(stack, size, size, slices, null, null);

    expect(Math.abs(result.rowShiftPxPerSlice)).toBeLessThan(0.05);
    expect(Math.abs(result.colShiftPxPerSlice)).toBeLessThan(0.05);
  });

  it("refuses a stack that has nothing to register", async () => {
    await expect(
      estimateSliceAlignment(new Float32Array(64), 8, 8, 1, null, null),
    ).rejects.toThrow(/at least 2 slices/);
  });
});

describe("median", () => {
  // The quickselect median must return exactly what a full sort would, or every
  // registration image is centered on the wrong value.
  function medianBySort(values: Float32Array): number {
    const sorted = Float32Array.from(values).sort();
    const mid = sorted.length >> 1;
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  it("matches a full sort across lengths and distributions", () => {
    let seed = 12345;
    const rand = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    for (const n of [1, 2, 3, 4, 5, 16, 17, 999, 1000, 4096, 100001]) {
      for (const kind of ["random", "sorted", "reverse", "constant", "twovalue", "negative"]) {
        const values = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          values[i] = kind === "random" ? rand() * 200 - 100
            : kind === "sorted" ? i
            : kind === "reverse" ? n - i
            : kind === "constant" ? 7
            : kind === "twovalue" ? (i % 2 ? -3 : 11)
            : -rand() * 50;
        }
        expect(median(values), `n=${n} ${kind}`).toBe(medianBySort(values));
      }
    }
  });

  it("does not modify its input", () => {
    const values = Float32Array.from([5, 1, 4, 2, 3]);
    median(values);
    expect(Array.from(values)).toEqual([5, 1, 4, 2, 3]);
  });
});
