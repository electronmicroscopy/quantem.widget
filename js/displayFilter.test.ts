// @ts-nocheck
// Parity: the browser display-filter port vs the Python/scipy reference
// (src/quantem/widget/utils/display_filter.py). The reference arrays are
// generated at test time by the actual apply_display_filter implementation,
// so a scientist scrubbing sigma in an exported HTML page sees the same
// pixels the kernel would have produced.
//
// The CPU path exercised here shares its gaussian kernel weights, zoom
// output-shape/coordinate math and percentile with the WGSL compute passes
// in GPUDisplayFilterEngine; GPU-vs-CPU agreement on a real adapter is
// checked in the browser (node has no WebGPU runtime).
import { execFileSync } from "node:child_process";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { applyDisplayFilterCPU, resolvePanelDenoiseKnobs } from "./displayFilter";

describe("per-panel denoise knobs", () => {
  it("returns the selected panel's independent mode, sigma, and bin", () => {
    const fallback = { mode: "none", sigma: 4, bin: 1 };
    expect(resolvePanelDenoiseKnobs(
      1,
      ["none", "gaussian", "anscombe"],
      [4, 2, 8],
      [1, 1, 2],
      fallback,
    )).toEqual({ mode: "gaussian", sigma: 2, bin: 1 });
    expect(resolvePanelDenoiseKnobs(
      2,
      ["none", "gaussian", "anscombe"],
      [4, 20, 8],
      [1, 1, 2],
      fallback,
    )).toEqual({ mode: "anscombe", sigma: 8, bin: 2 });
  });
});

type FixtureCase = {
  mode: string;
  sigma: number;
  spatial_bin: number;
  expected: number[];
};

type Fixture = {
  width: number;
  height: number;
  image: number[];
  odd_width: number;
  odd_height: number;
  odd_image: number[];
  cases: FixtureCase[];
  odd_cases: FixtureCase[];
};

function pythonReferenceFixture(): Fixture | null {
  const python = process.env.PYTHON || "python";
  const code = String.raw`
import json

import numpy as np

from quantem.widget.utils.display_filter import apply_display_filter

rng = np.random.default_rng(7)
image = rng.poisson(0.3, (128, 128)).astype(np.float32)
odd_image = rng.poisson(0.3, (67, 65)).astype(np.float32)  # (n_rows, n_cols)

cases = [
    ("gaussian", 4.0, 1),
    ("gaussian", 2.5, 1),
    ("bin2", 4.0, 1),
    ("anscombe", 4.0, 1),
    ("anscombe", 8.0, 2),
    ("bin2_anscombe", 8.0, 1),
    ("bin4_anscombe", 8.0, 1),
    ("none", 4.0, 4),
]
odd_cases = [("gaussian", 3.0, 1), ("bin2", 4.0, 1), ("bin2_anscombe", 6.0, 1)]


def run(img, mode, sigma, spatial_bin):
    out = apply_display_filter(img, mode=mode, sigma=sigma, spatial_bin=spatial_bin)
    return out.ravel().tolist()


print(json.dumps({
    "width": 128,
    "height": 128,
    "image": image.ravel().tolist(),
    "odd_width": 65,
    "odd_height": 67,
    "odd_image": odd_image.ravel().tolist(),
    "cases": [
        {"mode": m, "sigma": s, "spatial_bin": b, "expected": run(image, m, s, b)}
        for m, s, b in cases
    ],
    "odd_cases": [
        {"mode": m, "sigma": s, "spatial_bin": b, "expected": run(odd_image, m, s, b)}
        for m, s, b in odd_cases
    ],
}))
`;
  try {
    return JSON.parse(execFileSync(python, ["-c", code], {
      encoding: "utf8",
      maxBuffer: 128 * 1024 * 1024,
      env: {
        ...process.env,
        PYTHONPATH: [path.resolve(__dirname, "..", "src"), process.env.PYTHONPATH || ""].join(path.delimiter),
      },
    }));
  } catch {
    return null;
  }
}

function maxAbsDiff(actual: Float32Array, expected: number[]): number {
  let worst = 0;
  for (let i = 0; i < expected.length; i++) {
    worst = Math.max(worst, Math.abs(actual[i] - expected[i]));
  }
  return worst;
}

const fixture = pythonReferenceFixture();

(fixture ? describe : describe.skip)("display filter Python/scipy parity", () => {
  it("matches apply_display_filter on a 128x128 Poisson map for every browser mode", () => {
    const image = Float32Array.from(fixture!.image);
    for (const testCase of fixture!.cases) {
      const actual = applyDisplayFilterCPU(
        image, fixture!.width, fixture!.height,
        testCase.mode, testCase.sigma, testCase.spatial_bin,
      );
      expect(actual.length).toBe(testCase.expected.length);
      expect(
        maxAbsDiff(actual, testCase.expected),
        `${testCase.mode} sigma=${testCase.sigma} bin=${testCase.spatial_bin}`,
      ).toBeLessThan(1e-3);
    }
  });

  it("matches on an odd-sized map (exercises the ndimage.zoom shape rounding)", () => {
    const image = Float32Array.from(fixture!.odd_image);
    for (const testCase of fixture!.odd_cases) {
      const actual = applyDisplayFilterCPU(
        image, fixture!.odd_width, fixture!.odd_height,
        testCase.mode, testCase.sigma, testCase.spatial_bin,
      );
      expect(actual.length).toBe(testCase.expected.length);
      expect(
        maxAbsDiff(actual, testCase.expected),
        `${testCase.mode} sigma=${testCase.sigma} bin=${testCase.spatial_bin}`,
      ).toBeLessThan(1e-3);
    }
  });
});
