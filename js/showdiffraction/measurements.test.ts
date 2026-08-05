import { describe, expect, it } from "vitest";
import {
  MEASUREMENT_COLUMNS,
  buildMeasurementRecords,
  measurementCsv,
  measurementMetadata,
} from "./measurements";

const SPOT = {
  id: 1,
  row: 10.5,
  col: 20.25,
  raw_row: 10.0,
  raw_col: 20.0,
  row_err: 0.1,
  col_err: 0.2,
  r_pixels: 30.0,
  r_pixels_err: 0.15,
  g_magnitude: 0.3,
  g_magnitude_err: 0.001,
  d_spacing: 3.33,
  d_spacing_err: 0.02,
  angle_deg: 45.0,
  angle_deg_err: 0.5,
  intensity: 999.0,
  fit_quality: 0.98,
  hkl: "111",
  hkl_candidates: ["111", "200"],
  note: "a,b",
};

const RING = {
  id: 2,
  radius_px: 60.0,
  g_magnitude: 0.6,
  d_spacing: 1.67,
  intensity: 500.0,
  fit_quality: 0.95,
  fwhm_px: 2.5,
  fwhm_inv_angstrom: 0.025,
  intensity_integrated: 1234.0,
  hkl: "220",
  hkl_candidates: ["220"],
};

describe("MEASUREMENT_COLUMNS", () => {
  it("mirrors the Python export schema", () => {
    expect(MEASUREMENT_COLUMNS).toEqual([
      "id", "kind", "raw_row", "raw_col", "row", "col", "row_err", "col_err",
      "r_pixels", "r_pixels_err", "g_inv_angstrom", "g_inv_angstrom_err",
      "d_angstrom", "d_angstrom_err", "angle_deg", "angle_deg_err",
      "intensity", "fit_quality", "fwhm_px", "fwhm_inv_angstrom",
      "intensity_integrated", "hkl", "hkl_candidates", "note",
    ]);
  });
});

describe("buildMeasurementRecords", () => {
  it("keeps spots before rings and fills kind-specific nulls", () => {
    const [spot, ring] = buildMeasurementRecords([SPOT], [RING]);
    expect(spot.kind).toBe("spot");
    expect(spot.raw_row).toBe(10.0);
    expect(spot.hkl_candidates).toBe("111|200");
    expect(spot.fwhm_px).toBeNull();
    expect(spot.fwhm_inv_angstrom).toBeNull();
    expect(spot.intensity_integrated).toBeNull();
    expect(ring.kind).toBe("ring");
    expect(ring.r_pixels).toBe(60.0);
    expect(ring.fwhm_px).toBe(2.5);
    expect(ring.fwhm_inv_angstrom).toBe(0.025);
    expect(ring.intensity_integrated).toBe(1234.0);
    expect(ring.hkl_candidates).toBe("220");
    expect(ring.row).toBeNull();
    expect(ring.angle_deg).toBeNull();
    for (const record of [spot, ring]) {
      expect(Object.keys(record)).toEqual(MEASUREMENT_COLUMNS);
    }
  });

  it("tolerates missing optional fields", () => {
    const [spot] = buildMeasurementRecords([{ id: 3, row: 1, col: 2, r_pixels: 5, intensity: 7 }], []);
    expect(spot.hkl_candidates).toBe("");
    expect(spot.d_angstrom).toBeNull();
    expect(spot.note).toBe("");
  });
});

describe("measurementCsv", () => {
  it("writes every column and escapes commas", () => {
    const csv = measurementCsv(buildMeasurementRecords([SPOT], [RING]));
    const lines = csv.split("\n");
    expect(lines[0]).toBe(MEASUREMENT_COLUMNS.join(","));
    expect(lines).toHaveLength(3);
    expect(lines[1]).toContain("111|200");
    expect(lines[1]).toContain('"a,b"');
    expect(lines[2].split(",")[1]).toBe("ring");
  });
});

describe("measurementMetadata", () => {
  it("mirrors the Python metadata block", () => {
    const metadata = measurementMetadata({
      centerRow: 64.5,
      centerCol: 63.5,
      centerMethod: "symmetry",
      kPixelSize: 0.01,
      kCalibrated: true,
      calibrationSource: "from_phase",
      calibrationRefD: 2.355,
      calibrationRefRadius: 42.0,
      maskRegions: [{ kind: "wedge", start_deg: 0, end_deg: 90 }],
      backgroundSubtracted: false,
    });
    expect(metadata).toEqual({
      widget_name: "ShowDiffraction",
      center_row: 64.5,
      center_col: 63.5,
      center_method: "symmetry",
      k_pixel_size_inv_angstrom_per_px: 0.01,
      calibrated: true,
      calibration_source: "from_phase",
      calibration_ref_d_angstrom: 2.355,
      calibration_ref_radius_px: 42.0,
      mask_regions: [{ kind: "wedge", start_deg: 0, end_deg: 90 }],
      background_subtracted: false,
    });
  });
});
