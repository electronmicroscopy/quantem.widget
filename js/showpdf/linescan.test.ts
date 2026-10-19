import { describe, expect, it } from "vitest";
import {
  effectiveLineEnds, lineBandColumns, lineBandIncludes, linescanSampleCount, linescanSampleIndex,
  sampleFraction, sampledBinIndices, type LineEnds,
} from "./linescan";

describe("ShowPDF line-scan sampling", () => {
  it("snaps the sampled line's ends to scan positions", () => {
    expect(effectiveLineEnds({ r0: 1.3, c0: 0.4, r1: 15.8, c1: 12.5 }, false, 40, 40))
      .toEqual({ r0: 1, c0: 0, r1: 16, c1: 13 });
    // perpendicular bisector of (14, 12) -> (14, 23) has half-pixel ends (19.5, 17.5)
    // -> (8.5, 17.5); they snap halves up
    expect(effectiveLineEnds({ r0: 14, c0: 12, r1: 14, c1: 23 }, true, 40, 40))
      .toEqual({ r0: 20, c0: 18, r1: 9, c1: 18 });
    expect(effectiveLineEnds({ r0: 3, c0: 3, r1: 3, c1: 3 }, true, 40, 40))
      .toEqual({ r0: 3, c0: 3, r1: 3, c1: 3 });
  });

  it("clips the sampled line to the scan, so its tip has data", () => {
    // perpendicular bisector of (1, 4) -> (1, 20) runs from (9, 12) to (-7, 12)
    expect(effectiveLineEnds({ r0: 1, c0: 4, r1: 1, c1: 20 }, true, 24, 24))
      .toEqual({ r0: 9, c0: 12, r1: 0, c1: 12 });
    // oblique line leaving the right edge
    expect(effectiveLineEnds({ r0: 2, c0: 10, r1: 12, c1: 40 }, false, 24, 24))
      .toEqual({ r0: 2, c0: 10, r1: 6, c1: 23 });
    // wholly outside: collapses to a point
    const out = effectiveLineEnds({ r0: -5, c0: 2, r1: -2, c1: 9 }, false, 24, 24);
    expect(out.r0 === out.r1 && out.c0 === out.c1).toBe(true);
  });

  it("selects the band of _line_band_mask: flat ends at A and B, half width at least 0.5", () => {
    // The band as one string of 0/1 per scan row.
    const band = (ends: LineEnds, width: number, rows: number, cols: number) => {
      const inBand = lineBandIncludes(ends, width)!;
      const row = (r: number) => Array.from({ length: cols }, (_, c) => (inBand(r, c) ? 1 : 0)).join("");
      return Array.from({ length: rows }, (_, r) => row(r));
    };
    // horizontal, width 3: rows 1-3, columns from A to B only
    expect(band({ r0: 2, c0: 1, r1: 2, c1: 4 }, 3, 5, 6))
      .toEqual(["000000", "011110", "011110", "011110", "000000"]);
    // width 1: half width 0.5; the neighbours of a 45-degree line are sqrt(0.5) off it
    expect(band({ r0: 0, c0: 0, r1: 3, c1: 3 }, 1, 4, 4)).toEqual(["1000", "0100", "0010", "0001"]);
    expect(band({ r0: 0, c0: 0, r1: 2, c1: 4 }, 1, 3, 5)).toEqual(["11000", "01110", "00011"]);
    expect(lineBandIncludes({ r0: 3, c0: 3, r1: 3, c1: 3 }, 5)).toBeNull();
  });

  it("limits each row of the band to its columns, at every angle", () => {
    // Lines from near the centre of a 121 x 121 scan every 3 degrees, short to
    // clipped, with snapped (sampled) and with fractional ends.
    const rows = 121, cols = 121;
    let lines = 0, missed = 0, extra = 0;
    for (let deg = 0; deg < 360; deg += 3) {
      const th = (deg * Math.PI) / 180;
      for (const len of [1, 7, 50, 80]) {
        const drawn = { r0: 60.3, c0: 59.8, r1: 60.3 - len * Math.sin(th), c1: 59.8 + len * Math.cos(th) };
        for (const ends of [effectiveLineEnds(drawn, false, rows, cols), drawn]) {
          for (const width of [1, 3, 5, 9, 15]) {
            const inBand = lineBandIncludes(ends, width);
            if (!inBand) continue;
            lines++;
            for (let r = 0; r < rows; r++) {
              const [lo, hi] = lineBandColumns(ends, width, r, cols);
              let n = 0;
              for (let c = 0; c < cols; c++) {
                if (!inBand(r, c)) continue;
                n++;
                if (c < lo || c > hi) missed++;   // every band position is a candidate
              }
              if (hi - lo + 1 > n + 4) extra++;   // and at most 4 other columns are
            }
          }
        }
      }
    }
    expect(lines).toBeGreaterThan(4000);
    expect(missed).toBe(0);
    expect(extra).toBe(0);
  });

  it("takes one sample per scan step along the longer axis, capped", () => {
    expect(linescanSampleCount(0, 10, 256)).toBe(11);   // horizontal, 10 px
    expect(linescanSampleCount(10, 10, 256)).toBe(11);  // 45 degrees: 14.1 px, 11 samples
    expect(linescanSampleCount(-7, 3, 256)).toBe(8);
    expect(linescanSampleCount(7, 3, 4)).toBe(4);       // linescan_max_bins cap
    expect(linescanSampleCount(0, 0, 256)).toBe(1);
  });

  it("puts the first sample at A and the last at the tip B", () => {
    for (const N of [2, 3, 8, 11, 57]) {
      expect(sampleFraction(0, N)).toBe(0);
      expect(sampleFraction(N - 1, N)).toBe(1);
    }
    expect(sampleFraction(0, 1)).toBe(0.5);
  });

  it("assigns positions to the nearest sample", () => {
    expect(linescanSampleIndex(0, 11)).toBe(0);
    expect(linescanSampleIndex(1, 11)).toBe(10);
    expect(linescanSampleIndex(0.049, 11)).toBe(0);
    expect(linescanSampleIndex(0.05, 11)).toBe(1);             // tie goes up, as floor(x + 0.5)
    expect(linescanSampleIndex(0.75 - 1e-15, 3)).toBe(2);      // a tie off by rounding error still goes up
    expect(linescanSampleIndex(0.5, 1)).toBe(0);
  });

  it("picks exactly evenly spaced waterfall curves that end at the tip", () => {
    for (let N = 1; N <= 80; N++) {
      for (let count = 1; count <= 30; count++) {
        const s = sampledBinIndices(N, count);
        expect(s.length).toBeGreaterThan(0);
        expect(s[0]).toBeGreaterThanOrEqual(0);
        expect(s[s.length - 1]).toBe(N - 1);                // the tip is always drawn
        const steps = new Set(s.slice(1).map((v, i) => v - s[i]));
        expect(steps.size).toBeLessThanOrEqual(1);          // one step size only
        if (steps.size === 1) expect(s[0]).toBeLessThan([...steps][0]);  // gap at A below one step
        if (count >= N) expect(s.length).toBe(N);
        if (count <= 1) expect(s).toEqual([N - 1]);
        // the count is the closest achievable with exact spacing ending at the tip
        if (count > 1 && count < N) {
          let best = Infinity;
          for (let k = 1; k <= N - 1; k++) {
            best = Math.min(best, Math.abs(Math.floor((N - 1) / k) + 1 - count));
          }
          expect(Math.abs(s.length - count)).toBe(best);
        }
      }
    }
    expect(sampledBinIndices(10, 3)).toEqual([1, 5, 9]);
    expect(sampledBinIndices(15, 10)).toEqual([0, 2, 4, 6, 8, 10, 12, 14]);
    expect(sampledBinIndices(22, 10)).toEqual([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]);
  });
});
