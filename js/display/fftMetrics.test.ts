import { describe, expect, it } from "vitest";
import { computeFftQualityMetrics, summarizeFftQualityMetrics } from "./fftMetrics";

function addGaussian(mag: Float32Array, width: number, height: number, row: number, col: number, amp: number, sigma: number) {
  const r0 = Math.max(0, Math.floor(row - sigma * 3));
  const r1 = Math.min(height - 1, Math.ceil(row + sigma * 3));
  const c0 = Math.max(0, Math.floor(col - sigma * 3));
  const c1 = Math.min(width - 1, Math.ceil(col + sigma * 3));
  const sigma2 = sigma * sigma;
  for (let r = r0; r <= r1; r++) {
    for (let c = c0; c <= c1; c++) {
      const dr = r - row;
      const dc = c - col;
      mag[r * width + c] += amp * Math.exp(-(dr * dr + dc * dc) / (2 * sigma2));
    }
  }
}

function lowFrequencyMagnitude(width: number, height: number): Float32Array {
  const mag = new Float32Array(width * height);
  const cy = height / 2;
  const cx = width / 2;
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const dist = Math.hypot(r - cy, c - cx);
      mag[r * width + c] = 10 / (1 + dist * dist);
    }
  }
  return mag;
}

describe("FFT quality metrics", () => {
  it("scores lattice-like FFT peaks above a low-frequency-only FFT", () => {
    const width = 128;
    const height = 128;
    const low = lowFrequencyMagnitude(width, height);
    const lattice = lowFrequencyMagnitude(width, height);
    const cy = height / 2;
    const cx = width / 2;
    const radius = 30;
    for (let i = 0; i < 6; i++) {
      const theta = (i / 6) * Math.PI * 2;
      addGaussian(lattice, width, height, cy + Math.sin(theta) * radius, cx + Math.cos(theta) * radius, 80, 1.4);
    }

    const lowMetrics = computeFftQualityMetrics(low, width, height);
    const latticeMetrics = computeFftQualityMetrics(lattice, width, height);

    expect(lowMetrics?.sharp ?? 0).toBeLessThan(10);
    expect(latticeMetrics?.sharp ?? 0).toBeGreaterThan((lowMetrics?.sharp ?? 0) * 10);
    expect(lowMetrics?.peaks ?? 0).toBe(0);
    expect(latticeMetrics?.peaks ?? 0).toBeGreaterThanOrEqual(6);
    expect(latticeMetrics?.snr ?? 0).toBeGreaterThan(4);
  });

  it("summarizes Show3D-style tiled FFT panel metrics", () => {
    const panelW = 64;
    const panelH = 64;
    const width = panelW * 2;
    const height = panelH;
    const mag = new Float32Array(width * height);
    addGaussian(mag, width, height, panelH / 2, panelW / 2 + 18, 70, 1.2);
    addGaussian(mag, width, height, panelH / 2, panelW + panelW / 2 + 18, 35, 1.2);

    const left = computeFftQualityMetrics(mag, width, height, {
      region: { x: 0, y: 0, width: panelW, height: panelH },
    });
    const right = computeFftQualityMetrics(mag, width, height, {
      region: { x: panelW, y: 0, width: panelW, height: panelH },
    });
    const summary = summarizeFftQualityMetrics([left, right]);

    expect(summary?.peaks).toBe((left?.peaks ?? 0) + (right?.peaks ?? 0));
    expect(summary?.sharp ?? 0).toBeGreaterThan(0);
    expect(summary?.snr ?? 0).toBeGreaterThan(0);
  });
});
