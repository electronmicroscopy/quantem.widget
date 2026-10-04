/**
 * PlanPtycho arithmetic: beam geometry through the specimen, the model window the detector sampling gives, the grades
 * and text of each check, thickness recommendations and the detector calibration. Mirrors `plan_geometry`,
 * `check_statuses`, `check_rows`, `recommended_settings` and `detector_sampling_mrad` in
 * `src/quantem/widget/planptycho.py`; both are pinned to `goldens.json` (regenerate with
 * `python scripts/planptycho_goldens.py`).
 *
 * Lengths in Å, angles in mrad unless a name says otherwise.
 */

import { fft2d } from "../fft";

export interface PlanSettings {
  voltage_kV: number;
  semiangle_mrad: number;
  focus_depth_nm: number;
  thickness_nm: number;
  detector_px: number;
  wave_window_factor?: number;
  detector_mrad_per_px: number;
  scan_step_A: number;
  scan_size_px: number;
  tilt_mrad?: number[];
  holz_repeat_A?: number | null;
}

export interface PlanGeometry {
  wavelength_A: number;
  window_A: number;
  pixel_A: number;
  theta_max_mrad: number;
  airy_A: number;
  entrance_A: number;
  exit_A: number;
  widest_A: number;
  depth_of_field_A: number;
  scan_A: number;
  overlap: number;
  reach: number;
  tilt_mrad: number;
  lean_A: number;
  thickness_A: number;
  focus_A: number;
  holz_mrad: number | null;
}

export type Grade = "pass" | "caution" | "fail";
export type CheckId = "window" | "margin" | "overlap" | "reach" | "lean" | "split";
export interface CheckRow { id: string; label: string; status: Grade | "info"; value: string; rule: string; note: string }
export interface DetectorPresets {
  detectors: Record<string, { pixels: number; pixel_pitch_um: number; note: string }>;
  arina_mrad_per_px: Record<string, number>;
  arina_mrad_mm: number;
}

/** Relativistic electron wavelength: 12.26434 / sqrt(V (1 + 0.97848e-6 V)), V in volts. */
export function wavelengthA(voltage_kV: number): number {
  const volts = voltage_kV * 1e3;
  return 12.26434 / Math.sqrt(volts * (1 + 0.97848e-6 * volts));
}

/** Beam diameter at depth z below the entrance: 2 alpha |z - f| + 1.22 lambda / alpha (geometric cone plus the Airy disk). */
export function beamDiameterA(depth_A: number, focus_A: number, semiangle_mrad: number, wavelength_A: number): number {
  const alpha = semiangle_mrad * 1e-3;
  return 2 * alpha * Math.abs(depth_A - focus_A) + (1.22 * wavelength_A) / alpha;
}

export function planGeometry(s: PlanSettings): PlanGeometry {
  const lam = wavelengthA(s.voltage_kV);
  const alpha = s.semiangle_mrad * 1e-3;
  const dtheta = s.detector_mrad_per_px * 1e-3;
  const thickness = s.thickness_nm * 10;
  const focus = s.focus_depth_nm * 10;
  const airy = (1.22 * lam) / alpha;
  const entrance = 2 * alpha * Math.abs(focus) + airy;
  const exit = 2 * alpha * Math.abs(thickness - focus) + airy;
  const factor = s.wave_window_factor ?? 1;
  if (factor !== 1 && factor !== 2) throw new Error("Wave window factor must be 1 or 2.");
  const window = factor * lam / dtheta;
  const tilt = Math.hypot(...(s.tilt_mrad ?? [0, 0]));
  return {
    wavelength_A: lam,
    window_A: window,
    pixel_A: window / factor / s.detector_px,
    theta_max_mrad: (s.detector_px * s.detector_mrad_per_px) / 2,
    airy_A: airy,
    entrance_A: entrance,
    exit_A: exit,
    widest_A: Math.max(entrance, exit),
    depth_of_field_A: (2 * lam) / (alpha * alpha),
    scan_A: s.scan_size_px * s.scan_step_A,
    overlap: 1 - s.scan_step_A / entrance,
    reach: (s.detector_px * s.detector_mrad_per_px) / 2 / s.semiangle_mrad,
    tilt_mrad: tilt,
    lean_A: thickness * Math.tan(tilt * 1e-3),
    thickness_A: thickness,
    focus_A: focus,
    holz_mrad: s.holz_repeat_A ? 1e3 * Math.sqrt((2 * lam) / s.holz_repeat_A) : null,
  };
}

/** Window, margin and lean only caution: simulated SrTiO3 reconstructed past all three (see planptycho.py). */
export function checkStatuses(g: PlanGeometry): Record<CheckId, Grade> {
  return {
    window: g.widest_A <= g.window_A ? "pass" : "caution",
    margin: g.scan_A >= 2 * g.widest_A ? "pass" : "caution",
    overlap: g.overlap >= 0.6 ? "pass" : g.overlap >= 0.3 ? "caution" : "fail",
    reach: g.reach >= 1.5 ? "pass" : g.reach >= 1.0 ? "caution" : "fail",
    lean: g.lean_A <= g.pixel_A ? "pass" : "caution",
    split: g.focus_A >= 0 && g.focus_A <= g.thickness_A ? "pass" : "caution",
  };
}

/** Where the specimen lies relative to the focus (Python `_split_text`). */
export function splitText(g: PlanGeometry): string {
  const above = g.focus_A / 10, below = (g.thickness_A - g.focus_A) / 10;
  if (above < 0) return `specimen entirely below the focus (${(-above).toFixed(1)} nm above the entrance)`;
  if (below < 0) return `specimen entirely above the focus (${(-below).toFixed(1)} nm below the exit)`;
  return `${above.toFixed(1)} nm above, ${below.toFixed(1)} nm below the focus`;
}

/** The text of every check, identical to Python `check_rows` (pinned by the goldens). */
export function checkRows(g: PlanGeometry, detector_px: number, scan_step_A: number, column_phase_rad_per_A = 0): CheckRow[] {
  const status = checkStatuses(g);
  const radius = g.widest_A / 2;
  const wavePixels = Math.round(g.window_A / g.pixel_A);
  const wide = wavePixels !== detector_px;
  const f = (value: number, digits: number) => value.toFixed(digits);
  const graded: Omit<CheckRow, "status">[] = [
    { id: "window", label: "Beam fits the virtual window", value: `${f(g.widest_A, 0)} Å beam in a ${f(g.window_A, 0)} Å window`,
      rule: wide ? "widest beam (entrance or exit) ≤ window factor × wavelength / detector pixel angle" : "widest beam (entrance or exit) ≤ wavelength / detector pixel angle",
      note: "The beam wraps around the virtual window. Simulated SrTiO3 still reconstructed with it (130 nm: 68 Å beam in a 35 Å "
        + "window, picture 0.80). A longer camera length widens the window but lowers the detector reach; a 2x virtual window "
        + "in the reconstruction removes the wrap without changing the acquisition." },
    { id: "margin", label: "Scan margin for the spread beam", value: `${f(g.scan_A, 0)} Å scan, beam radius ${f(radius, 0)} Å`,
      rule: "scan side ≥ 4 × widest beam radius",
      note: `The outer ~${f(radius, 0)} Å of the scan is lit from one side only deep in the sample (110 nm SrTiO3, 24 Å scan: `
        + `centre 0.79, edge 0.60). Scan ${f(4 * radius, 0)} Å or more, or report only the interior.` },
    { id: "overlap", label: "Probe overlap at the entrance", value: `${f(100 * g.overlap, 0)} % (step ${f(scan_step_A, 3)} Å, beam ${f(g.entrance_A, 1)} Å)`,
      rule: "1 − step / entrance beam diameter ≥ 60 %",
      note: "Neighbouring probe positions barely share illuminated area; a smaller step or a focus deeper below the entrance increases the overlap." },
    { id: "reach", label: "Detector reach", value: `${f(g.theta_max_mrad, 0)} mrad = ${f(g.reach, 1)} × semiangle`,
      rule: "detector edge ≥ 1.5 × semiangle",
      note: status.reach === "fail" ? "The detector cuts into the bright-field disk." : "Little dark field is recorded; scattering to high angles carries the fine detail." },
    { id: "lean", label: "Column lean across the thickness", value: `${f(g.lean_A, 1)} Å at ${f(g.tilt_mrad, 1)} mrad tilt`,
      rule: "thickness × tan(tilt) ≤ object pixel",
      note: "The columns lean across more than an object pixel: measure the tilt and hold it in the propagator. 110 nm SrTiO3 at "
        + "4 mrad reached 0.74 with the measured tilt held, 0.81 when the probe was also frozen for the first 10 iterations." },
    { id: "split", label: "Focus splits the specimen", value: splitText(g), rule: "0 ≤ focus depth ≤ thickness",
      note: "The whole thickness is on one side of the focus, so the beam is widest at one surface. Thick-sample runs focused "
        + "inside the specimen (110 nm SrTiO3: 17 nm below the entrance)." },
  ];
  const rows: CheckRow[] = graded.map((row) => {
    const grade = status[row.id as CheckId];
    return { ...row, status: grade, note: grade === "pass" ? "" : row.note };
  });
  rows.push(
    { id: "pixel", label: "Object pixel", status: "info", value: `${f(g.pixel_A, 3)} Å (${wavePixels} px over ${f(g.window_A, 1)} Å)`, rule: wide ? "model window / wave pixels; measured detector unchanged" : "window / detector pixels", note: "" },
    { id: "depth", label: "Depth of field", status: "info", value: `${f(g.depth_of_field_A / 10, 1)} nm`,
      rule: "2 × wavelength / semiangle²: the depth the probe resolves", note: "" },
    { id: "beam", label: "Beam diameter", status: "info", value: `entrance ${f(g.entrance_A, 1)} Å, focus ${f(g.airy_A, 1)} Å, exit ${f(g.exit_A, 1)} Å`,
      rule: "2 × semiangle × |depth − focus| + 1.22 × wavelength / semiangle", note: "" },
  );
  if (column_phase_rad_per_A > 0) rows.push({ id: "phase", label: "Column phase per nm", status: "info",
    value: `${f(column_phase_rad_per_A * 10, 2)} rad (strongest column)`,
    rule: "interaction constant × projected potential of the strongest column per nm of thickness", note: "" });
  if (g.holz_mrad !== null) {
    const edge = g.theta_max_mrad;
    const where = g.holz_mrad <= edge ? "on" : g.holz_mrad <= Math.SQRT2 * edge ? "in the corners of" : "beyond";
    rows.push({ id: "holz", label: "First HOLZ ring", status: "info", value: `${f(g.holz_mrad, 0)} mrad, ${where} the detector`,
      rule: "√(2 × wavelength / lattice period along the beam)", note: "" });
  }
  return rows;
}

/** Starting settings for a thickness (see Python `recommended_settings`): focus at mid-thickness and a scan wide enough
 *  for the margin check (at least 128, multiples of 16). */
export function recommendedSettings(thickness_nm: number, voltage_kV: number, semiangle_mrad: number, scan_step_A: number) {
  const lam = wavelengthA(voltage_kV);
  const alpha = semiangle_mrad * 1e-3;
  const widest = 2 * alpha * thickness_nm * 10 / 2 + (1.22 * lam) / alpha;
  const scanPx = Math.max(128, 16 * Math.ceil((2 * widest) / scan_step_A / 16));
  return { thickness_nm, focus_depth_nm: thickness_nm / 2, scan_size_px: scanPx };
}

/** Angular sampling per native pixel (see Python `detector_sampling_mrad`), or null for an unknown detector. */
export function detectorSamplingMrad(presets: DetectorPresets | undefined, detector: string, camera_length_mm: number): number | null {
  const spec = presets?.detectors?.[detector];
  const arinaPitch = presets?.detectors?.Arina?.pixel_pitch_um;
  if (!presets || !spec || !arinaPitch) return null;
  const arina = presets.arina_mrad_per_px[String(camera_length_mm)] ?? presets.arina_mrad_mm / camera_length_mm;
  return (arina * spec.pixel_pitch_um) / arinaPitch;
}

function nextPow2(value: number): number {
  return 2 ** Math.ceil(Math.log2(Math.max(2, value)));
}

/**
 * Probe intensity at `defocus_A` from its focus, on the reconstruction's model window (lambda / detector pixel angle):
 * psi(k) = A(k) exp(-i pi lambda dz k^2), free-space propagation (no channelling). Fourier sampling 1 / window equals the
 * detector's, and only frequencies the detector records (the square of `detector_px` pixels) are kept, so an aperture the
 * detector cuts is cut here too. The grid is the next power of two that holds the detector and the aperture; a beam wider
 * than the window wraps around, as in the reconstruction. Returns `{intensity, n}`, centred, float32 row-major.
 */
export function probeIntensity(window_A: number, semiangle_mrad: number, wavelength_A: number, defocus_A: number, detector_px: number) {
  const dk = 1 / window_A;
  const kCut = (semiangle_mrad * 1e-3) / wavelength_A;
  const n = Math.min(1024, nextPow2(Math.max(detector_px, 2 * Math.ceil(kCut / dk) + 4)));
  const half = detector_px / 2;
  const real = new Float32Array(n * n);
  const imag = new Float32Array(n * n);
  const edge = dk * 0.5;                          // half-pixel soft edge avoids aliasing rings
  for (let row = 0; row < n; row++) {
    const ir = row < n / 2 ? row : row - n;
    if (Math.abs(ir) > half) continue;
    for (let col = 0; col < n; col++) {
      const ic = col < n / 2 ? col : col - n;
      if (Math.abs(ic) > half) continue;
      const k = Math.hypot(ir, ic) * dk;
      const aperture = Math.min(1, Math.max(0, (kCut - k) / edge + 0.5));
      if (aperture <= 0) continue;
      const chi = -Math.PI * wavelength_A * defocus_A * k * k;
      real[row * n + col] = aperture * Math.cos(chi);
      imag[row * n + col] = aperture * Math.sin(chi);
    }
  }
  fft2d(real, imag, n, n, true);
  const intensity = new Float32Array(n * n);
  for (let row = 0; row < n; row++) {
    const src = ((row + n / 2) % n) * n;
    for (let col = 0; col < n; col++) {
      const index = src + ((col + n / 2) % n);
      intensity[row * n + col] = real[index] * real[index] + imag[index] * imag[index];
    }
  }
  return { intensity, n };
}

/** Geometric simulation-cell planning only; not a multislice boundary-convergence test. */
export function simulationGeometry(settings: PlanSettings, cell: number[], repeats: number[], pixels: number, guard: number) {
  if (cell.length !== 3 || cell.some(a => !Number.isFinite(a) || a <= 0))
    throw Error("Use three positive oriented cell lengths in Å.");
  if (repeats.length !== 2 || repeats.some(n => !Number.isSafeInteger(n) || n < 1))
    throw Error("Use two positive whole-number repeats in (row, col) order.");
  if (!Number.isSafeInteger(pixels) || pixels < 1)
    throw Error("Pixels per cell must be a positive integer; try 96.");
  if (!Number.isFinite(guard) || guard < 0)
    throw Error("Guard must be finite and nonnegative, in Å.");
  const g = planGeometry(settings);
  const span = (settings.scan_size_px - 1) * settings.scan_step_A;
  const extent = cell.slice(0, 2).map((a, i) => a * repeats[i]);
  const margins = extent.map((a, i) => (a - span - g.widest_A) / 2 - settings.thickness_nm * 10 * Math.abs(Math.tan((settings.tilt_mrad?.[i] || 0) * 1e-3)));
  return { extent, span, margins, gpts: repeats.map(n => n * pixels), sampling: cell.slice(0, 2).map(a => a / pixels),
    zRepeats: Math.ceil(settings.thickness_nm * 10 / cell[2]), fits: margins.every(m => m >= guard) };
}
