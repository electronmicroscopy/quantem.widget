/** ShowDiffraction browser-side measurement export.
 *
 * Column schema kept in sync with the Python exporter so browser CSV/JSON
 * downloads match the notebook output.
 */

export const MEASUREMENT_COLUMNS = [
  "id",
  "kind",
  "raw_row",
  "raw_col",
  "row",
  "col",
  "row_err",
  "col_err",
  "r_pixels",
  "r_pixels_err",
  "g_inv_angstrom",
  "g_inv_angstrom_err",
  "d_angstrom",
  "d_angstrom_err",
  "angle_deg",
  "angle_deg_err",
  "intensity",
  "fit_quality",
  "fwhm_px",
  "fwhm_inv_angstrom",
  "intensity_integrated",
  "hkl",
  "hkl_candidates",
  "note",
] as const;

type MeasurementRecord = Record<
  (typeof MEASUREMENT_COLUMNS)[number],
  string | number | null
>;

/** One spot as the Python widget syncs it in ``spots``. */
export interface SpotDict {
  id: number;
  row: number;
  col: number;
  raw_row?: number | null;
  raw_col?: number | null;
  row_err?: number | null;
  col_err?: number | null;
  d_spacing: number | null;
  d_spacing_err?: number | null;
  g_magnitude: number | null;
  g_magnitude_err?: number | null;
  r_pixels: number;
  r_pixels_err?: number | null;
  angle_deg?: number | null;
  angle_deg_err?: number | null;
  fit_quality?: number | null;
  intensity: number;
  hkl?: string;
  hkl_candidates?: string[];
  d_ref?: number | null;
  d_error?: number | null;
  note?: string;
}

/** One ring as the Python widget syncs it in ``rings``. */
export interface RingDict {
  id: number;
  radius_px: number;
  g_magnitude: number | null;
  d_spacing: number | null;
  intensity: number;
  hkl?: string;
  hkl_candidates?: string[];
  d_ref?: number | null;
  d_error?: number | null;
  fwhm_px?: number | null;
  fwhm_inv_angstrom?: number | null;
  intensity_integrated?: number | null;
  fit_quality?: number | null;
  note?: string;
}

/** Export row for one spot record. */
function spotMeasurementRecord(spot: SpotDict): MeasurementRecord {
  return {
    id: spot.id,
    kind: "spot",
    raw_row: spot.raw_row ?? null,
    raw_col: spot.raw_col ?? null,
    row: spot.row,
    col: spot.col,
    row_err: spot.row_err ?? null,
    col_err: spot.col_err ?? null,
    r_pixels: spot.r_pixels,
    r_pixels_err: spot.r_pixels_err ?? null,
    g_inv_angstrom: spot.g_magnitude ?? null,
    g_inv_angstrom_err: spot.g_magnitude_err ?? null,
    d_angstrom: spot.d_spacing ?? null,
    d_angstrom_err: spot.d_spacing_err ?? null,
    angle_deg: spot.angle_deg ?? null,
    angle_deg_err: spot.angle_deg_err ?? null,
    intensity: spot.intensity,
    fit_quality: spot.fit_quality ?? null,
    fwhm_px: null,
    fwhm_inv_angstrom: null,
    intensity_integrated: null,
    hkl: spot.hkl ?? "",
    hkl_candidates: (spot.hkl_candidates ?? []).join("|"),
    note: spot.note ?? "",
  };
}

/** Export row for one ring record. */
function ringMeasurementRecord(ring: RingDict): MeasurementRecord {
  return {
    id: ring.id,
    kind: "ring",
    raw_row: null,
    raw_col: null,
    row: null,
    col: null,
    row_err: null,
    col_err: null,
    r_pixels: ring.radius_px,
    r_pixels_err: null,
    g_inv_angstrom: ring.g_magnitude ?? null,
    g_inv_angstrom_err: null,
    d_angstrom: ring.d_spacing ?? null,
    d_angstrom_err: null,
    angle_deg: null,
    angle_deg_err: null,
    intensity: ring.intensity,
    fit_quality: ring.fit_quality ?? null,
    fwhm_px: ring.fwhm_px ?? null,
    fwhm_inv_angstrom: ring.fwhm_inv_angstrom ?? null,
    intensity_integrated: ring.intensity_integrated ?? null,
    hkl: ring.hkl ?? "",
    hkl_candidates: (ring.hkl_candidates ?? []).join("|"),
    note: ring.note ?? "",
  };
}

/** Export rows for all spots and rings. */
export function buildMeasurementRecords(spots: SpotDict[], rings: RingDict[]): MeasurementRecord[] {
  return [...spots.map(spotMeasurementRecord), ...rings.map(ringMeasurementRecord)];
}

/** CSV text with the full column schema. */
export function measurementCsv(records: MeasurementRecord[]): string {
  const escapeCsv = (value: string | number | null) => {
    const text = value == null ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const rows = records.map((record) => MEASUREMENT_COLUMNS.map((column) => escapeCsv(record[column])).join(","));
  return [MEASUREMENT_COLUMNS.join(","), ...rows].join("\n");
}

interface MetadataState {
  brightFieldCenterRowPx: number;
  brightFieldCenterColPx: number;
  centerMethod: string;
  detectorSamplingInvA: number;
  kCalibrated: boolean;
  calibrationSource: string;
  calibrationRefD: number;
  calibrationRefRadius: number;
  maskRegions: object[];
  backgroundSubtracted: boolean;
}

/** Export metadata block for CSV/JSON downloads. */
export function measurementMetadata(state: MetadataState): object {
  return {
    widget_name: "ShowDiffraction",
    bright_field_center_row_px: state.brightFieldCenterRowPx,
    bright_field_center_col_px: state.brightFieldCenterColPx,
    center_method: state.centerMethod ?? "",
    detector_sampling_inv_A: state.detectorSamplingInvA,
    calibrated: Boolean(state.kCalibrated),
    calibration_source: state.calibrationSource ?? "none",
    calibration_ref_d_angstrom: state.calibrationRefD ?? 0,
    calibration_ref_radius_px: state.calibrationRefRadius ?? 0,
    mask_regions: state.maskRegions ?? [],
    background_subtracted: Boolean(state.backgroundSubtracted),
  };
}
