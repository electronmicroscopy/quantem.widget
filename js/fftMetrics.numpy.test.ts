// @ts-nocheck
import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { computeFftQualityMetrics } from "./fftMetrics";

function numpyReferenceFixture() {
  const python = process.env.PYTHON || "python";
  const code = String.raw`
import json
import math

import numpy as np


def sampling_to_angstrom(sampling, unit):
    if sampling is None or not math.isfinite(sampling) or sampling <= 0:
        return None
    normalized = str(unit or "").strip().lower()
    if normalized in ("", "px", "pixel", "pixels"):
        return None
    if normalized in ("a", "å", "angstrom", "angstroms"):
        return sampling
    if normalized in ("nm", "nanometer", "nanometers"):
        return sampling * 10
    if normalized in ("pm", "picometer", "picometers"):
        return sampling * 0.01
    if normalized in ("um", "µm", "micrometer", "micrometers"):
        return sampling * 10000
    if normalized == "m":
        return sampling * 1e10
    return None


def band_bounds(width, height, sampling_a):
    r_max = min(width, height) / 2
    if r_max < 16:
        return None
    if sampling_a and sampling_a > 0:
        min_period_a, max_period_a = 1.0, 6.0
        n = min(width, height)
        lo = max(2, n * sampling_a / max_period_a)
        hi = min(r_max - 1, n * sampling_a / min_period_a)
        if hi > lo + 2:
            return lo, hi
    return 0.15 * r_max, 0.6 * r_max


def compute_reference_metrics(mag, width, height, sampling, unit):
    sampling_a = sampling_to_angstrom(sampling, unit)
    bounds = band_bounds(width, height, sampling_a)
    if bounds is None:
        return None
    lo, hi = bounds
    cx = width / 2
    cy = height / 2
    r_max = min(width, height) / 2
    dc_radius = max(2, r_max * 0.015)
    stride = max(1, math.ceil(max(width, height) / max(64, 768)))
    r_lo_sq = lo * lo
    r_hi_sq = hi * hi
    dc_sq = dc_radius * dc_radius
    r_max_sq = (r_max - 1) * (r_max - 1)

    total_power = 0.0
    band_power = 0.0
    band_count = 0
    band_sum = 0.0
    band_sum_sq = 0.0
    strongest = 0.0
    angle_bins = 16
    angle_sum = np.zeros(angle_bins, dtype=np.float64)
    angle_count = np.zeros(angle_bins, dtype=np.uint32)
    flat = mag.ravel()

    for y in range(0, height, stride):
        dy = y - cy
        row = y * width
        for x in range(0, width, stride):
            dx = x - cx
            r_sq = dx * dx + dy * dy
            if r_sq <= dc_sq or r_sq >= r_max_sq:
                continue
            v = max(0.0, float(flat[row + x]))
            power = v * v
            total_power += power
            if r_sq < r_lo_sq or r_sq > r_hi_sq:
                continue
            band_power += power
            band_count += 1
            band_sum += v
            band_sum_sq += v * v
            strongest = max(strongest, v)
            theta = math.atan2(dy, dx)
            bin_idx = max(0, min(angle_bins - 1, math.floor(((theta + math.pi) / (2 * math.pi)) * angle_bins)))
            angle_sum[bin_idx] += v
            angle_count[bin_idx] += 1

    mean = band_sum / band_count
    variance = max(0.0, band_sum_sq / band_count - mean * mean)
    std = math.sqrt(variance)
    threshold = mean + max(3, 5 * std if std > 0 else mean * 2)
    peaks = 0
    for y in range(stride, height - stride, stride):
        dy = y - cy
        row = y * width
        for x in range(stride, width - stride, stride):
            dx = x - cx
            r_sq = dx * dx + dy * dy
            if r_sq < r_lo_sq or r_sq > r_hi_sq:
                continue
            center = float(flat[row + x])
            if center < threshold:
                continue
            is_peak = True
            for oy in range(-stride, stride + 1, stride):
                if not is_peak:
                    break
                for ox in range(-stride, stride + 1, stride):
                    if ox == 0 and oy == 0:
                        continue
                    if float(flat[(y + oy) * width + x + ox]) >= center:
                        is_peak = False
                        break
            if is_peak:
                peaks += 1

    ring = None
    if int(np.count_nonzero(angle_count)) > 1:
        angle_means = np.zeros(angle_bins, dtype=np.float64)
        for i in range(angle_bins):
            if angle_count[i] > 0:
                angle_means[i] = angle_sum[i] / angle_count[i]
        valid = angle_means[angle_count > 0]
        angle_mean = float(valid.mean())
        if angle_mean > 1e-20:
            ring = max(0.0, 1 - float(valid.std()) / angle_mean)

    return {
        "sharp": 100 * band_power / total_power,
        "peaks": peaks,
        "snr": max(0.0, (strongest - mean) / std) if std > 1e-20 else (strongest / mean if mean > 1e-20 else None),
        "ring": ring,
    }


n = 128
sampling = 0.04
unit = "nm"
rng = np.random.default_rng(123)
y, x = np.mgrid[:n, :n].astype(np.float64)
xx = (x - n / 2) / n
yy = (y - n / 2) / n
image = np.zeros((n, n), dtype=np.float64)
for angle in np.deg2rad([0, 60, 120]):
    image += np.cos(2 * np.pi * 18 * (np.cos(angle) * xx + np.sin(angle) * yy) + 0.4)
image = (image / 3.0 + 1.0) * 0.5
image *= np.exp(-((xx * 1.15) ** 2 + (yy * 1.15) ** 2) * 1.6)
image += 0.025 * rng.standard_normal((n, n))
mag = np.abs(np.fft.fftshift(np.fft.fft2(image))).astype(np.float32)
print(json.dumps({
    "width": n,
    "height": n,
    "sampling": sampling,
    "unit": unit,
    "mag": mag.ravel().tolist(),
    "expected": compute_reference_metrics(mag, n, n, sampling, unit),
}))
`;
  try {
    return JSON.parse(execFileSync(python, ["-c", code], { encoding: "utf8", maxBuffer: 10 * 1024 * 1024 }));
  } catch {
    return null;
  }
}

const fixture = numpyReferenceFixture();

(fixture ? describe : describe.skip)("FFT quality metrics NumPy parity", () => {
  it("matches a NumPy FFT/reference implementation on a deterministic lattice image", () => {
    const actual = computeFftQualityMetrics(
      new Float32Array(fixture.mag),
      fixture.width,
      fixture.height,
      { sampling: fixture.sampling, unit: fixture.unit },
    );

    expect(actual).not.toBeNull();
    expect(actual.peaks).toBe(fixture.expected.peaks);
    expect(actual.sharp).toBeCloseTo(fixture.expected.sharp, 6);
    expect(actual.snr).toBeCloseTo(fixture.expected.snr, 6);
    expect(actual.ring).toBeCloseTo(fixture.expected.ring, 6);
  });
});
