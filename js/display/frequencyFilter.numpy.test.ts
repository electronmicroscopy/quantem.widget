// @ts-nocheck
// Parity: the CPU frequency filter (the path a browser without WebGPU runs)
// against numpy on the same input: zero-pad to powers of two, fft2, the radial
// sigmoid mask on the padded grid, ifft2, crop. The WebGPU path does the same
// on the padded grid, so non-power-of-two images are where the two used to part.
import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { applyFrequencyFilterCPU } from "./frequencyFilter";

const CASES = [
  { width: 5, height: 3, image: new Array(15).fill(7), options: { mode: "highpass", cutoff: 0.15 } },
  {
    width: 37, height: 20,
    image: Array.from({ length: 740 }, (_, index) => 50 + 30 * Math.sin(index * 0.37) * Math.cos(Math.floor(index / 37) * 0.6) + (index % 11)),
    options: { mode: "bandpass", center: 0.35, width: 0.2 },
  },
  { width: 64, height: 48, image: Array.from({ length: 3072 }, (_, index) => (index * 7919) % 101), options: { mode: "lowpass", cutoff: 0.3, edge: 0.05 } },
];

function numpyReference(): number[][] {
  const python = process.env.PYTHON || "python";
  const code = String.raw`
import json
import sys

import numpy as np

results = []
for case in json.load(sys.stdin):
    width, height, options = case["width"], case["height"], case["options"]
    image = np.array(case["image"], dtype=np.float32).reshape(height, width)
    padded_w, padded_h = 1 << (width - 1).bit_length(), 1 << (height - 1).bit_length()
    plane = np.zeros((padded_h, padded_w))
    plane[:height, :width] = image
    rows, cols = np.indices((padded_h, padded_w))
    fy = np.minimum(rows, padded_h - rows) / max(1, padded_h / 2)
    fx = np.minimum(cols, padded_w - cols) / max(1, padded_w / 2)
    radius = np.minimum(1, np.hypot(fx, fy))
    edge = max(0.002, options.get("edge", 0.035))
    sigmoid = lambda x: 1 / (1 + np.exp(-x / edge))
    if options["mode"] == "lowpass":
        mask = 1 - sigmoid(radius - options["cutoff"])
    elif options["mode"] == "highpass":
        mask = sigmoid(radius - options["cutoff"])
    else:
        half = max(edge, options["width"] / 2)
        mask = sigmoid(radius - max(0, options["center"] - half)) * (1 - sigmoid(radius - min(1, options["center"] + half)))
    results.append(np.fft.ifft2(np.fft.fft2(plane) * mask).real[:height, :width].ravel().tolist())
print(json.dumps(results))
`;
  return JSON.parse(execFileSync(python, ["-c", code], { input: JSON.stringify(CASES), encoding: "utf8" }));
}

describe("CPU frequency filter matches numpy", () => {
  const reference = numpyReference();

  it.each(CASES.map((testCase, index) => [`${testCase.width}x${testCase.height} ${testCase.options.mode}`, index] as const))("%s", (_, index) => {
    const { width, height, image, options } = CASES[index];
    const filtered = applyFrequencyFilterCPU(Float32Array.from(image), width, height, options);
    const scale = Math.max(...image.map(Math.abs));
    let worst = 0;
    filtered.forEach((value, pixel) => { worst = Math.max(worst, Math.abs(value - reference[index][pixel])); });
    expect(worst / scale).toBeLessThan(2e-6);
  });
});
