// @ts-nocheck
// Parity: the browser slice-alignment estimator (CPU path, the one a browser
// without WebGPU runs) against src/quantem/widget/show3dslices/alignment.py,
// run on the same input from this test. The GPU-resident path is checked
// against this one on the hardware adapter in
// tests/browser/test_show3dslices_alignment.py.
import { execFileSync } from "node:child_process";
import * as path from "node:path";
import { describe, expect, it } from "vitest";

import { fft2d } from "../display/fft";
import { estimateSliceAlignment, median, registrationImage } from "./sliceAlignment";

/**
 * Textured image with a background ramp and deterministic noise. The ramp
 * makes the high-pass boundary handling matter: a blur that pads the edges
 * differently from alignment.py changes the registration image near the
 * borders and with it the correlation quality.
 */
function texture(height: number, width: number, seed: number): Float32Array {
  let state = seed;
  const noise = () => { state = (state * 1103515245 + 12345) & 0x7fffffff; return state / 0x7fffffff - 0.5; };
  const image = new Float32Array(height * width);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const blobA = 4.0 * Math.exp(-(((row - height * 0.35) ** 2) + ((col - width * 0.42) ** 2)) / 80.0);
      const blobB = 2.5 * Math.exp(-(((row - height * 0.68) ** 2) + ((col - width * 0.73) ** 2)) / 55.0);
      image[row * width + col] = Math.sin(row / 3.1) + 0.7 * Math.cos(col / 4.7) + 0.35 * Math.sin((row + col) / 5.3)
        + blobA + blobB + 0.08 * row + 0.05 * col + 0.8 * noise();
    }
  }
  return image;
}

/**
 * The texture shifted by an exact sub-pixel amount per slice with an FFT phase
 * ramp, so the drift the estimator should recover is known analytically.
 */
function shiftedStack(size: number, slices: number, rowDrift: number, colDrift: number, seed: number): Float32Array {
  const real = texture(size, size, seed);
  const imag = new Float32Array(size * size);
  fft2d(real, imag, size, size, false);
  const stack = new Float32Array(slices * size * size);
  for (let sliceIndex = 0; sliceIndex < slices; sliceIndex++) {
    const shiftedReal = new Float32Array(size * size);
    const shiftedImag = new Float32Array(size * size);
    for (let row = 0; row < size; row++) {
      const rowFreq = (row >= (size + 1) >> 1 ? row - size : row) / size;
      for (let col = 0; col < size; col++) {
        const colFreq = (col >= (size + 1) >> 1 ? col - size : col) / size;
        const angle = -2 * Math.PI * (rowFreq * sliceIndex * rowDrift + colFreq * sliceIndex * colDrift);
        const i = row * size + col;
        shiftedReal[i] = real[i] * Math.cos(angle) - imag[i] * Math.sin(angle);
        shiftedImag[i] = real[i] * Math.sin(angle) + imag[i] * Math.cos(angle);
      }
    }
    fft2d(shiftedReal, shiftedImag, size, size, true);
    stack.set(shiftedReal, sliceIndex * size * size);
  }
  return stack;
}

// Shapes cover square, non-square, a plane too small for the 12 px sigma (the
// kernel then wraps the whole padded axis) and the 12 px cap on a wide plane.
const IMAGES = [[64, 64, 1], [40, 48, 2], [4, 9, 3], [130, 200, 4]].map(([height, width, seed]) => ({
  height, width, values: texture(height, width, seed),
}));
const STACKS = [
  { size: 64, slices: 7, rowDrift: 1, colDrift: -2, seed: 5 },
  { size: 64, slices: 7, rowDrift: 0.75, colDrift: -0.4, seed: 6 },
  { size: 64, slices: 6, rowDrift: 1.3, colDrift: 0.55, seed: 7 },
  { size: 64, slices: 5, rowDrift: 0, colDrift: 0, seed: 8 },
].map(stack => ({ ...stack, values: shiftedStack(stack.size, stack.slices, stack.rowDrift, stack.colDrift, stack.seed) }));

type PythonEstimate = {
  row_shift_px_per_slice: number;
  col_shift_px_per_slice: number;
  adjacent_shift_apply_px: number[][];
  quality: number[];
};

function pythonReference(): { images: number[][]; estimates: PythonEstimate[] } {
  const python = process.env.PYTHON || "python";
  const code = String.raw`
import importlib.util
import json
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("alignment", sys.argv[1])
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)
payload = json.load(sys.stdin)
images = [alignment.registration_image(np.asarray(case["values"], dtype=np.float32).reshape(case["height"], case["width"])).ravel().tolist()
          for case in payload["images"]]
estimates = [alignment.estimate_global_slice_alignment(np.asarray(case["values"], dtype=np.float32).reshape(case["slices"], case["size"], case["size"]))
             for case in payload["stacks"]]
print(json.dumps({"images": images, "estimates": estimates}))
`;
  const reference = path.resolve(__dirname, "..", "..", "src", "quantem", "widget", "show3dslices", "alignment.py");
  const input = JSON.stringify({
    images: IMAGES.map(({ height, width, values }) => ({ height, width, values: Array.from(values) })),
    stacks: STACKS.map(({ size, slices, values }) => ({ size, slices, values: Array.from(values) })),
  });
  return JSON.parse(execFileSync(python, ["-c", code, reference], { input, encoding: "utf8", maxBuffer: 1 << 28 }));
}

describe("slice alignment matches alignment.py", () => {
  const reference = pythonReference();

  it("builds the same high-passed, windowed registration image", () => {
    IMAGES.forEach(({ height, width, values }, index) => {
      const image = registrationImage(values, width, height);
      let worst = 0;
      image.forEach((value, i) => { worst = Math.max(worst, Math.abs(value - reference.images[index][i])); });
      expect(worst, `${height}x${width}`).toBeLessThan(1e-5);
    });
  });

  it("estimates the same adjacent shifts, slopes and correlation quality", async () => {
    for (const [index, { size, slices, values }] of STACKS.entries()) {
      const result = await estimateSliceAlignment(values, size, size, slices, null);
      const expected = reference.estimates[index];
      expect(result.backend).toBe("cpu");
      // Shifts land on the 1/20 px refinement grid, so equal estimates agree exactly.
      expect(result.adjacentShiftPx).toEqual(expected.adjacent_shift_apply_px);
      expect(result.rowShiftPxPerSlice).toBeCloseTo(expected.row_shift_px_per_slice, 9);
      expect(result.colShiftPxPerSlice).toBeCloseTo(expected.col_shift_px_per_slice, 9);
      result.quality.forEach((value, pair) => expect(Math.abs(value - expected.quality[pair])).toBeLessThan(1e-6));
    }
  });

  it("recovers the applied drift as the correction slope", async () => {
    // Content drifts row +0.75 / col -0.4 per deeper slice, so the correction
    // is the opposite shift; the Hann window keeps the fit slightly inside it.
    const { size, slices, values } = STACKS[1];
    const result = await estimateSliceAlignment(values, size, size, slices, null);
    expect(Math.abs(result.rowShiftPxPerSlice + 0.75)).toBeLessThan(0.1);
    expect(Math.abs(result.colShiftPxPerSlice - 0.4)).toBeLessThan(0.1);
  });

  it("refuses a stack that has nothing to register", async () => {
    await expect(estimateSliceAlignment(new Float32Array(64), 8, 8, 1, null)).rejects.toThrow(/at least 2 slices/);
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
