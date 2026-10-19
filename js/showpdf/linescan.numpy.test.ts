// @ts-nocheck
import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import {
  effectiveLineEnds, lineBandColumns, lineBandIncludes, linescanSampleCount, linescanSampleIndex,
  sampleFraction,
} from "./linescan";

// Python reference: ShowPDF._effective_line and ShowPDF._linescan_samples
// themselves (bound to a light stub so no dataset is needed), for random lines
// with whole, half-pixel and fractional endpoints, perpendicular sampling,
// several widths and linescan_max_bins caps, for lines every 4 degrees, and
// for decimal perpendicular lines that snap differently with a hypot length.
// Skips when the interpreter in $PYTHON cannot import ShowPDF (it needs
// quantem PR 248).
function pythonSamples() {
  const python = process.env.PYTHON || "python";
  const code = String.raw`
import json
import math
import random
import sys

try:
    from quantem.widget.showpdf import ShowPDF
except Exception:
    sys.exit(1)


class Stub:
    _line_band_mask = ShowPDF._line_band_mask
    _effective_line = ShowPDF._effective_line
    _linescan_samples = ShowPDF._linescan_samples


rng = random.Random(7)
stubs = []
for i in range(150):
    st = Stub()
    st.scan_rows, st.scan_cols = rng.randint(6, 40), rng.randint(6, 40)
    st.line_active = True
    st.line_perpendicular = rng.random() < 0.4
    kind = rng.random()
    if kind < 0.4:
        pick = lambda hi: float(rng.randint(0, hi - 1))
    elif kind < 0.7:
        pick = lambda hi: rng.randint(0, 2 * hi - 3) / 2.0
    elif kind < 0.85:
        pick = lambda hi: rng.uniform(0, hi - 1)
    else:  # ends outside the scan (clipped)
        pick = lambda hi: rng.uniform(-8, hi + 7)
    st.line_row0, st.line_col0 = pick(st.scan_rows), pick(st.scan_cols)
    st.line_row1, st.line_col1 = pick(st.scan_rows), pick(st.scan_cols)
    st.line_width = rng.choice([1, 3, 5, 7, 9])
    st.linescan_max_bins = rng.choice([256, 256, 256, 7, 3, 1])
    stubs.append(st)
for deg in range(0, 360, 4):  # every angle class, from the centre of a 41 x 41 scan
    st = Stub()
    st.scan_rows = st.scan_cols = 41
    st.line_active, st.line_perpendicular, st.linescan_max_bins = True, False, 256
    th = math.radians(deg)
    st.line_row0, st.line_col0 = 20.0, 20.0
    st.line_row1, st.line_col1 = 20.0 - 17 * math.sin(th), 20.0 + 17 * math.cos(th)
    st.line_width = (1, 3, 5, 9)[deg // 4 % 4]
    stubs.append(st)
# Decimal ends whose bisector snaps differently when its length comes from a
# hypot (np.hypot, math.hypot, Math.hypot) instead of sqrt(dr * dr + dc * dc).
for ends in ([0, 26.2, 34.4, 4.8], [37.3, 28.7, 30.8, 17.8], [19.2, 3.5, 7.2, 10.9], [55.9, 57.1, 2.6, 9.2],
             [16.4, 34.0, 53.4, 7.2]):
    st = Stub()
    st.scan_rows = st.scan_cols = 64
    st.line_active, st.line_perpendicular, st.linescan_max_bins = True, True, 256
    st.line_row0, st.line_col0, st.line_row1, st.line_col1 = ends
    st.line_width = 3
    stubs.append(st)
cases = []
for st in stubs:
    eff = st._effective_line()
    res = st._linescan_samples()
    case = dict(drawn=[st.line_row0, st.line_col0, st.line_row1, st.line_col1], perp=st.line_perpendicular,
                eff=[float(v) for v in eff], rows=st.scan_rows, cols=st.scan_cols,
                width=st.line_width, maxBins=st.linescan_max_bins, n=None)
    if res is not None:
        band, idx, n, length, positions = res
        case.update(n=int(n), length=length, positions=[float(p) for p in positions],
                    band=band.astype(int).ravel().tolist(), idx=idx.astype(int).ravel().tolist())
    cases.append(case)
print(json.dumps(cases))
`;
  try {
    const out = execFileSync(python, ["-c", code], {
      encoding: "utf8", maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "pipe", "ignore"],
    });
    return JSON.parse(out);
  } catch {
    return null;
  }
}

const cases = pythonSamples();

// Python reference for every integer line on a 16 x 16 scan, sampled
// perpendicular. Many of their bisectors have an end exactly half way between
// scan positions, so the line's length must round exactly as in Python for
// that end to snap the same way (e.g. (0, 0) -> (12, 7) samples (9, 0) -> (2, 10)).
function pythonPerpendicularSweep() {
  const python = process.env.PYTHON || "python";
  const code = String.raw`
import json
import sys

try:
    from quantem.widget.showpdf import ShowPDF
except Exception:
    sys.exit(1)


class Stub:
    _effective_line = ShowPDF._effective_line


st = Stub()
st.scan_rows = st.scan_cols = 16
st.line_active = st.line_perpendicular = True
lines = []
for r0 in range(16):
    for c0 in range(16):
        for r1 in range(16):
            for c1 in range(16):
                st.line_row0, st.line_col0, st.line_row1, st.line_col1 = float(r0), float(c0), float(r1), float(c1)
                lines.append([r0, c0, r1, c1, *(float(v) for v in st._effective_line())])
print(json.dumps(lines))
`;
  try {
    const out = execFileSync(python, ["-c", code], {
      encoding: "utf8", maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "pipe", "ignore"],
    });
    return JSON.parse(out);
  } catch {
    return null;
  }
}

const sweep = pythonPerpendicularSweep();

(cases ? describe : describe.skip)("ShowPDF line-scan sampling matches Python", () => {
  it("has enough reference lines", () => {
    expect(cases.filter((c) => c.n !== null).length).toBeGreaterThan(80);
  });

  it("snaps the sampled line exactly as Python", () => {
    const hypotCase = { drawn: [0, 26.2, 34.4, 4.8], perp: true, eff: [8, 0, 28, 33] };
    expect(cases).toContainEqual(expect.objectContaining(hypotCase));
    for (const c of cases) {
      const [r0, c0, r1, c1] = c.drawn;
      const e = effectiveLineEnds({ r0, c0, r1, c1 }, c.perp, c.rows, c.cols);
      expect([e.r0, e.c0, e.r1, e.c1]).toEqual(c.eff);
    }
  });

  it("gives the same samples, positions and assignment for every scan position", () => {
    for (const c of cases) {
      const [r0, c0, r1, c1] = c.eff;
      // no band (a zero-length line): nothing to draw or select
      if (c.n === null) {
        expect(lineBandIncludes({ r0, c0, r1, c1 }, c.width)).toBeNull();
        continue;
      }
      const dr = r1 - r0;
      const dc = c1 - c0;
      const segSq = dr * dr + dc * dc;
      const len = Math.sqrt(segSq);
      const n = linescanSampleCount(dr, dc, c.maxBins);
      expect(n).toBe(c.n);
      for (let i = 0; i < n; i++) expect(sampleFraction(i, n) * c.length).toBeCloseTo(c.positions[i], 9);
      if (n > 1) expect(c.positions[n - 1]).toBeCloseTo(len, 9);   // the last position is the tip
      // the band the overlay draws and the offline recompute selects
      const inBand = lineBandIncludes({ r0, c0, r1, c1 }, c.width);
      expect(inBand).not.toBeNull();
      for (let r = 0; r < c.rows; r++) {
        for (let col = 0; col < c.cols; col++) {
          const t = ((r - r0) * dr + (col - c0) * dc) / segSq;
          expect(inBand(r, col) ? 1 : 0).toBe(c.band[r * c.cols + col]);
          if (inBand(r, col)) expect(linescanSampleIndex(t, n)).toBe(c.idx[r * c.cols + col]);
        }
      }
      // the overlay tests only lineBandColumns' columns and paints exactly the band
      const painted = new Array(c.rows * c.cols).fill(0);
      for (let r = 0; r < c.rows; r++) {
        const [lo, hi] = lineBandColumns({ r0, c0, r1, c1 }, c.width, r, c.cols);
        for (let col = lo; col <= hi; col++) if (inBand(r, col)) painted[r * c.cols + col] = 1;
      }
      expect(painted).toEqual(c.band);
    }
    // Thousands of checks: about 1.5 s alone, past the default 5 s when every test file runs at once.
  }, 20_000);
});

(sweep ? describe : describe.skip)("ShowPDF perpendicular line ends match Python", () => {
  it("snaps every integer line's bisector on a 16 x 16 scan as Python", () => {
    expect(sweep.length).toBe(16 ** 4);
    expect(sweep).toContainEqual([0, 0, 12, 7, 9, 0, 2, 10]);
    const differ = [];
    for (const [r0, c0, r1, c1, ...eff] of sweep) {
      const e = effectiveLineEnds({ r0, c0, r1, c1 }, true, 16, 16);
      if ([e.r0, e.c0, e.r1, e.c1].some((v, i) => v !== eff[i])) differ.push([r0, c0, r1, c1]);
    }
    expect(differ).toEqual([]);
  });
});
