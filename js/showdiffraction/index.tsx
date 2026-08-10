/** ShowDiffraction frontend. */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Switch from "@mui/material/Switch";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, drawColorbar } from "../figure";
import { extractBytes, extractFloat32, formatNumber, downloadBlob, preserveRestoredWidgetModelsOnSave } from "../format";
import { computeHistogramFromBytes, findDataRange, sliderRange, applyLogScaleInPlace } from "../stats";
import { COLORMAPS, COLORMAP_NAMES, applyColormap } from "../colormaps";
import { MetadataSection } from "../widgetInfo";
import {
  dataAngleToScreen,
  dataColToScreenX,
  dataRowToScreenY,
  frameStats,
  screenToData,
  staleFrameNote,
  viewTransform,
} from "./overlayGeometry";
import { buildMeasurementRecords, measurementCsv, measurementMetadata } from "./measurements";

// Style tokens

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DPR = window.devicePixelRatio || 1;
const CANVAS_MIN = 384;
const PROFILE_H = 140;
const PROFILE_PAD = { left: 54, right: 14, top: 16, bottom: 34 };
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;
const typography = {
  label: { fontSize: 11 },
  value: { fontSize: 10, fontFamily: "monospace" },
};
const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
};
const switchStyles = {
  small: { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
};
const sliderStyles = {
  small: { py: 0, "& .MuiSlider-thumb": { width: 10, height: 10 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 } },
};
const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  textTransform: "none" as const,
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};
const downwardMenuProps = {
  anchorOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "top" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

// Info tooltip

function InfoTooltip({ text, theme = "dark" }: { text: React.ReactNode; theme?: "light" | "dark" }) {
  const isDark = theme === "dark";
  const content = typeof text === "string"
    ? <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    : text;
  return (
    <Tooltip
      title={content}
      arrow
      placement="bottom"
      componentsProps={{
        tooltip: {
          sx: {
            bgcolor: isDark ? "#333" : "#fff",
            color: isDark ? "#ddd" : "#333",
            border: `1px solid ${isDark ? "#555" : "#ccc"}`,
            maxWidth: 280,
            p: 1,
          },
        },
        arrow: {
          sx: {
            color: isDark ? "#333" : "#fff",
            "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` },
          },
        },
      }}
    >
      <Typography
        component="span"
        sx={{
          fontSize: 12,
          color: isDark ? "#888" : "#666",
          cursor: "help",
          ml: 0.5,
          "&:hover": { color: isDark ? "#aaa" : "#444" },
        }}
      >
        ⓘ
      </Typography>
    </Tooltip>
  );
}

function KeyboardShortcuts({ items }: { items: [string, string][] }) {
  return (
    <Box component="table" sx={{ borderCollapse: "collapse", "& td": { py: 0.25, fontSize: 11, lineHeight: 1.3, verticalAlign: "top" }, "& td:first-of-type": { pr: 1.5, opacity: 0.7, fontFamily: "monospace", fontSize: 10, whiteSpace: "nowrap" } }}>
      <tbody>
        {items.map(([key, desc], i) => (
          <tr key={i}><td>{key}</td><td>{desc}</td></tr>
        ))}
      </tbody>
    </Box>
  );
}

function calibrationSourceLabel(source: string): string {
  const labels: Record<string, string> = {
    from_phase: "phase",
    from_ring: "ring",
    from_spot: "spot",
    manual: "manual",
    metadata: "metadata",
  };
  return labels[source] ?? source;
}

// Mobile viewport
function useMobileViewport(): boolean {
  const getIsMobile = React.useCallback(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return false;
    }
    return window.matchMedia("(pointer: coarse)").matches || window.matchMedia("(max-width: 768px)").matches;
  }, []);
  const [isMobile, setIsMobile] = React.useState(getIsMobile);

  React.useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return;
    }
    const coarsePointer = window.matchMedia("(pointer: coarse)");
    const narrowViewport = window.matchMedia("(max-width: 768px)");
    const update = () => setIsMobile(getIsMobile());
    const addQueryListener = (query: MediaQueryList) => {
      if (typeof query.addEventListener === "function") query.addEventListener("change", update);
      else query.addListener(update);
    };
    const removeQueryListener = (query: MediaQueryList) => {
      if (typeof query.removeEventListener === "function") query.removeEventListener("change", update);
      else query.removeListener(update);
    };
    update();
    addQueryListener(coarsePointer);
    addQueryListener(narrowViewport);
    window.addEventListener("resize", update);
    return () => {
      removeQueryListener(coarsePointer);
      removeQueryListener(narrowViewport);
      window.removeEventListener("resize", update);
    };
  }, [getIsMobile]);

  return isMobile;
}

// Contrast histogram

interface HistogramProps {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  onRangePreview?: (min: number, max: number) => void;
  onRangeCommit?: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
}

function Histogram({ data, vminPct, vmaxPct, onRangeChange, onRangePreview, onRangeCommit, width = 110, height = 50, theme = "dark", dataMin = 0, dataMax = 1 }: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const sliderRef = React.useRef<HTMLDivElement | null>(null);
  const minLabelRef = React.useRef<HTMLElement | null>(null);
  const maxLabelRef = React.useRef<HTMLElement | null>(null);
  const onRangeChangeRef = React.useRef(onRangeChange);
  const onRangePreviewRef = React.useRef(onRangePreview);
  const onRangeCommitRef = React.useRef(onRangeCommit);
  const pendingRangeRef = React.useRef<[number, number] | null>(null);
  const rangeRafRef = React.useRef<number | null>(null);
  const [liveRange, setLiveRange] = React.useState<[number, number]>([vminPct, vmaxPct]);
  React.useEffect(() => { setLiveRange([vminPct, vmaxPct]); }, [vminPct, vmaxPct]);
  const [liveVminPct, liveVmaxPct] = liveRange;
  const bins = React.useMemo(() => data ? computeHistogramFromBytes(data) : new Array(256).fill(0), [data]);
  const isDark = theme === "dark";
  const colors = isDark
    ? { bg: "#1a1a2e", barActive: "#888", barInactive: "#444", border: "#333" }
    : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" };

  const formatValue = React.useCallback((pct: number) => {
    const val = dataMin + (pct / 100) * (dataMax - dataMin);
    return val >= 1000 ? val.toExponential(1) : val.toFixed(1);
  }, [dataMax, dataMin]);

  const drawHistogram = React.useCallback((loPct: number, hiPct: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = width * DPR;
    canvas.height = height * DPR;
    ctx.scale(DPR, DPR);
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);
    const displayBins = 64;
    const binRatio = Math.floor(bins.length / displayBins);
    const reducedBins: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) sum += bins[i * binRatio + j] || 0;
      reducedBins.push(sum / binRatio);
    }
    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;
    const vminBin = Math.floor((loPct / 100) * displayBins);
    const vmaxBin = Math.floor((hiPct / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      ctx.fillStyle = (i >= vminBin && i <= vmaxBin) ? colors.barActive : colors.barInactive;
      ctx.fillRect(i * barWidth + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [bins, colors, height, width]);

  const applyRangePreview = React.useCallback((next: [number, number]) => {
    const [lo, hi] = next;
    const slider = sliderRef.current?.querySelector(".MuiSlider-root") as HTMLElement | null;
    const thumbs = slider?.querySelectorAll(".MuiSlider-thumb");
    const track = slider?.querySelector(".MuiSlider-track") as HTMLElement | null;
    if (thumbs && thumbs.length >= 2) {
      (thumbs[0] as HTMLElement).style.left = `${lo}%`;
      (thumbs[1] as HTMLElement).style.left = `${hi}%`;
    }
    if (track) {
      track.style.left = `${lo}%`;
      track.style.width = `${Math.max(0, hi - lo)}%`;
    }
    if (minLabelRef.current) minLabelRef.current.textContent = formatValue(lo);
    if (maxLabelRef.current) maxLabelRef.current.textContent = formatValue(hi);
    drawHistogram(lo, hi);
  }, [drawHistogram, formatValue]);

  React.useEffect(() => {
    drawHistogram(liveVminPct, liveVmaxPct);
  }, [drawHistogram, liveVmaxPct, liveVminPct]);

  React.useEffect(() => {
    onRangeChangeRef.current = onRangeChange;
    onRangePreviewRef.current = onRangePreview;
    onRangeCommitRef.current = onRangeCommit;
  }, [onRangeChange, onRangeCommit, onRangePreview]);
  const emitRangePreview = React.useCallback((min: number, max: number) => {
    (onRangePreviewRef.current || onRangeChangeRef.current)(min, max);
  }, []);
  const emitRangeCommit = React.useCallback((min: number, max: number) => {
    (onRangeCommitRef.current || onRangeChangeRef.current)(min, max);
  }, []);
  const applySliderValue = (v: number | number[], emit: (min: number, max: number) => void) => {
    const [newMin, newMax] = v as number[];
    const next: [number, number] = [Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1)];
    setLiveRange(next);
    emit(next[0], next[1]);
  };
  const flushRangePreview = React.useCallback(() => {
    if (rangeRafRef.current != null) {
      window.cancelAnimationFrame(rangeRafRef.current);
      rangeRafRef.current = null;
    }
    const pending = pendingRangeRef.current;
    pendingRangeRef.current = null;
    if (pending) {
      setLiveRange(pending);
      applyRangePreview(pending);
      emitRangeCommit(pending[0], pending[1]);
    }
  }, [applyRangePreview, emitRangeCommit]);
  React.useEffect(() => () => {
    if (rangeRafRef.current != null) window.cancelAnimationFrame(rangeRafRef.current);
  }, []);
  const beginRangeDrag = React.useCallback((event: React.PointerEvent, dragWidth: number, lo0: number, hi0: number) => {
    const startX = event.clientX;
    const span = Math.max(1, hi0 - lo0);
    const previousCursor = document.body.style.cursor;
    document.body.style.cursor = "grabbing";
    const onMove = (moveEvent: PointerEvent) => {
      moveEvent.preventDefault();
      const deltaPct = ((moveEvent.clientX - startX) / Math.max(1, dragWidth)) * 100;
      const lo = Math.max(0, Math.min(100 - span, lo0 + deltaPct));
      const next: [number, number] = [lo, lo + span];
      pendingRangeRef.current = next;
      if (rangeRafRef.current == null) {
        rangeRafRef.current = window.requestAnimationFrame(() => {
          rangeRafRef.current = null;
          const pending = pendingRangeRef.current;
          if (pending) {
            setLiveRange(pending);
            applyRangePreview(pending);
            emitRangePreview(pending[0], pending[1]);
          }
        });
      }
    };
    const onUp = () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      document.removeEventListener("pointercancel", onUp);
      document.body.style.cursor = previousCursor;
      flushRangePreview();
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
    document.addEventListener("pointercancel", onUp);
  }, [applyRangePreview, emitRangePreview, flushRangePreview]);

  const sliderInset = 4;
  const sliderWidth = Math.max(1, width - sliderInset * 2);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0, width, overflow: "visible" }}>
      <Box sx={{ position: "relative", width, height: height + 6, overflow: "visible" }}>
        <canvas ref={canvasRef} style={{ width, height, border: `1px solid ${colors.border}`, display: "block" }} />
        <Box
          ref={sliderRef}
          onPointerDownCapture={(e) => {
            if ((e.target as HTMLElement).closest(".MuiSlider-thumb")) return;
            const rect = sliderRef.current?.getBoundingClientRect();
            if (!rect) return;
            const lo = Math.max(0, Math.min(100, Math.min(liveVminPct, liveVmaxPct)));
            const hi = Math.max(0, Math.min(100, Math.max(liveVminPct, liveVmaxPct)));
            const pct = ((e.clientX - rect.left) / Math.max(1, rect.width)) * 100;
            if (pct < lo || pct > hi) return;
            const thumbGuardPct = Math.max(4, (10 / Math.max(1, rect.width)) * 100);
            if (Math.abs(pct - lo) <= thumbGuardPct || Math.abs(pct - hi) <= thumbGuardPct) return;
            beginRangeDrag(e, rect.width, lo, hi);
            e.preventDefault();
            e.stopPropagation();
            e.nativeEvent.stopImmediatePropagation();
          }}
          sx={{ position: "absolute", left: sliderInset, top: height - 1, width: sliderWidth, height: 8, display: "flex", alignItems: "flex-start", cursor: "grab", zIndex: 2, overflow: "visible", touchAction: "none" }}
        >
          <Slider
            value={liveRange}
            onChange={(_, v) => applySliderValue(v, emitRangePreview)}
            onChangeCommitted={(_, v) => applySliderValue(v, emitRangeCommit)}
            min={0} max={100} size="small" valueLabelDisplay="auto"
            valueLabelFormat={formatValue}
            sx={{
              width: sliderWidth,
              py: 0,
              position: "relative",
              zIndex: 3,
              overflow: "visible",
              "& .MuiSlider-rail": { height: 2, zIndex: 1 },
              "& .MuiSlider-track": { height: 2, cursor: "grab", zIndex: 2 },
              "& .MuiSlider-thumb": { width: 8, height: 8, zIndex: 4 },
              "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px", zIndex: 5 },
            }}
          />
        </Box>
      </Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}><Typography ref={minLabelRef} sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{formatValue(liveVminPct)}</Typography><Typography ref={maxLabelRef} sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{formatValue(liveVmaxPct)}</Typography></Box>
    </Box>
  );
}

// Stat values
function formatStat(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.01) return v.toExponential(2);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

// Model types

interface SpotDict {
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
  r_pixels_err?: number;
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

interface PhaseEntry {
  name: string;
  a: number;
  b?: number;
  c?: number;
  alpha?: number;
  beta?: number;
  gamma?: number;
  absences: string;
}

interface MaskRegion {
  kind: string;
  start_deg?: number;
  end_deg?: number;
  row?: number;
  col?: number;
  radius?: number;
}

interface RingDict {
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

interface IdentifyLine {
  obs_d: number | null;
  ref_d: number | null;
  hkl: string;
  err: number | null;
  i_rel: number | null;
}

interface IdentifyResult {
  phase_id: string;
  name: string;
  matched: number;
  n_obs: number;
  mean_err: number | null;
  n_missing_strong: number | null;
  lines: IdentifyLine[];
}

interface QualityDict {
  center?: { method: string };
  calibration?: { source: string; k_pixel_size: number; rms_px: number };
  ellipse?: { ratio: number; angle_deg: number; corrected: boolean };
  rings?: { id: number; fit_quality: number }[];
  n_unexplained_rings?: number;
  mask_coverage_pct?: number;
  ring_snr?: { cv: number; coverage: number; snr: number };
}

// Spot colors
const PICK_COLORS = [
  "#ff4d4f", "#40a9ff", "#73d13d", "#ffa940",
  "#9254de", "#13c2c2", "#f759ab", "#bae637",
];
const spotColorAt = (index: number) => PICK_COLORS[((index % PICK_COLORS.length) + PICK_COLORS.length) % PICK_COLORS.length];

// Main component

function ShowDiffraction() {
  // Offline theme
  const [offline] = useModelState<boolean>("offline");
  const { themeInfo, colors: themeColors } = useTheme(offline);
  const rootRef = React.useRef<HTMLDivElement>(null);

  const model = useModel();
  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);

  const themedSelect = {
    "& .MuiSelect-select": { py: 0.25, px: 1, fontSize: 10, color: themeColors.text },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
    bgcolor: themeColors.controlBg,
    minWidth: 80,
  };
  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}`, maxHeight: 200, overflowY: "auto" } },
  };
  const topToolbarMenuProps = {
    ...downwardMenuProps,
    PaperProps: themedMenuProps.PaperProps,
  };
  // Control group
  const controlBox = { ...controlRow, border: `1px solid ${themeColors.border}`, borderRadius: "4px", bgcolor: themeColors.controlBg };
  // Number input
  const numInput = (width: number) => ({ width, fontSize: 10, padding: "2px 4px", background: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` });
  // Status colors
  const statusColors = themeInfo.theme === "dark"
    ? { good: "#81c784", warn: "#ffb74d", bad: "#e57373" }
    : { good: "#2e7d32", warn: "#e65100", bad: "#d32f2f" };

  // Model state
  const [title] = useModelState<string>("title");
  const [showTitle] = useModelState<boolean>("show_title");
  const [detRows] = useModelState<number>("det_rows");
  const [detCols] = useModelState<number>("det_cols");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [offlineFrames] = useModelState<DataView>("offline_frames");
  const [frameIdx, setFrameIdx] = useModelState<number>("frame_idx");
  const [nFrames] = useModelState<number>("n_frames");
  const [centerRow, setCenterRow] = useModelState<number>("center_row");
  const [centerCol, setCenterCol] = useModelState<number>("center_col");
  const [bfRadius] = useModelState<number>("bf_radius");
  const [kPixelSize] = useModelState<number>("k_pixel_size");
  const [kCalibrated] = useModelState<boolean>("k_calibrated");
  const [spots] = useModelState<SpotDict[]>("spots");
  const [snapEnabled, setSnapEnabled] = useModelState<boolean>("snap_enabled");
  const [snapRadius] = useModelState<number>("snap_radius");
  const [spotRefine, setSpotRefine] = useModelState<boolean>("spot_refine");
  const [, setSpotAddRequest] = useModelState<number[]>("_spot_add_request");
  const [, setSpotUndoRequest] = useModelState<boolean>("_spot_undo_request");
  const [, setSpotClearRequest] = useModelState<boolean>("_spot_clear_request");
  const [, setDetectRequest] = useModelState<number>("_detect_spots_request");
  const [, setDetectRingsRequest] = useModelState<number>("_detect_rings_request");
  const [, setSpotRemoveRequest] = useModelState<number>("_spot_remove_request");
  const [, setSpotMoveRequest] = useModelState<number[]>("_spot_move_request");
  const [, setRingRemoveRequest] = useModelState<number>("_ring_remove_request");
  const [dpColormap, setDpColormap] = useModelState<string>("dp_colormap");
  const [dpScaleMode, setDpScaleMode] = useModelState<string>("dp_scale_mode");
  const [dpInvert, setDpInvert] = useModelState<boolean>("dp_invert");
  const [dpVminPct, setDpVminPct] = useModelState<number>("dp_vmin_pct");
  const [dpVmaxPct, setDpVmaxPct] = useModelState<number>("dp_vmax_pct");
  const [dpStats] = useModelState<number[]>("dp_stats");
  const [showStats, setShowStats] = useModelState<boolean>("show_stats");
  const [showControls, setShowControls] = useModelState<boolean>("show_controls");
  const [controlsCollapsed, setControlsCollapsed] = useModelState<boolean>("controls_collapsed");
  const controlsVisible = showControls && !controlsCollapsed;
  const [panelWidthPx] = useModelState<number>("panel_width_px");

  // HTML export
  const [, setExportRequest] = useModelState<string>("export_request");
  const [exportStatus] = useModelState<string>("export_status");
  const [exportEnabled] = useModelState<boolean>("export_enabled");
  const [exportPayload] = useModelState<DataView>("export_payload");
  const [exportPayloadId] = useModelState<string>("export_payload_id");
  const [exportPayloadFilename] = useModelState<string>("export_filename");
  const exportCounterRef = React.useRef(0);
  const pendingExportRef = React.useRef<string>("");

  // Geometry
  const [centerMode, setCenterMode] = useModelState<string>("center_mode");
  const [rings] = useModelState<RingDict[]>("rings");
  const [showHkl, setShowHkl] = useModelState<boolean>("show_hkl");
  const [zoneAxis] = useModelState<string>("zone_axis");
  const [phaseMatch] = useModelState<string>("phase_match");
  const [calibrationSource] = useModelState<string>("calibration_source");
  const [calibrationRefD] = useModelState<number>("calibration_ref_d");
  const [calibrationRefRadius] = useModelState<number>("calibration_ref_radius");
  const [, setRingUndoRequest] = useModelState<boolean>("_ring_undo_request");
  const [, setRingClearRequest] = useModelState<boolean>("_ring_clear_request");
  const [, setCalibrateFromRingRequest] = useModelState<number[]>("_calibrate_from_ring_request");
  const [, setCalibrateFromSpotRequest] = useModelState<number[]>("_calibrate_from_spot_request");
  const [ellipseRatio] = useModelState<number>("ellipse_ratio");
  const [ellipseAngle] = useModelState<number>("ellipse_angle");
  const [ellipseCorrected, setEllipseCorrected] = useModelState<boolean>("ellipse_corrected");
  const [analysisStatus] = useModelState<string>("analysis_status");
  const [showProfile, setShowProfile] = useModelState<boolean>("show_profile");
  const [profileLog, setProfileLog] = useModelState<boolean>("profile_log");
  const [profileSubtract, setProfileSubtract] = useModelState<boolean>("profile_subtract_background");
  const [profileData] = useModelState<DataView>("_profile_data");
  const [, setRingAddRequest] = useModelState<number[]>("_ring_add_request");
  const [, setRefineCenterRequest] = useModelState<boolean>("_refine_center_request");
  const [, setFitRingsRequest] = useModelState<boolean>("_fit_rings_request");
  const [, setFitEllipseRequest] = useModelState<boolean>("_fit_ellipse_request");
  const [, setCalibratePhaseRequest] = useModelState<boolean>("_calibrate_phase_request");
  const [, setIndexRingsRequest] = useModelState<boolean>("_index_rings_request");
  const [, setIndexSpotsRequest] = useModelState<boolean>("_index_spots_request");
  const [, setIdentifyRequest] = useModelState<boolean>("_identify_request");
  const [, setAutoRequest] = useModelState<boolean>("_auto_request");
  const [refineMethod, setRefineMethod] = useModelState<string>("refine_method");
  const [centerMethod] = useModelState<string>("center_method");
  const [, setMergeRequest] = useModelState<boolean>("_merge_request");
  const [, setQualityRequest] = useModelState<boolean>("_quality_request");
  const [identifyElements, setIdentifyElements] = useModelState<string>("identify_elements");
  const [identifyCustomOnly, setIdentifyCustomOnly] = useModelState<boolean>("identify_custom_only");
  const [identifyResults] = useModelState<IdentifyResult[]>("_identify_results");
  const [quality] = useModelState<QualityDict>("_quality");
  const [selectedRingId, setSelectedRingId] = useModelState<number>("selected_ring_id");
  const [phaseName, setPhaseName] = useModelState<string>("phase_name");
  const [customPhases, setCustomPhases] = useModelState<PhaseEntry[]>("custom_phases");
  const [phaseLibrary] = useModelState<PhaseEntry[]>("_phase_library");
  const [maskRegions, setMaskRegions] = useModelState<MaskRegion[]>("mask_regions");
  const [showMask, setShowMask] = useModelState<boolean>("show_mask");
  const [showAzimuthal, setShowAzimuthal] = useModelState<boolean>("show_azimuthal");
  const [azimuthalData] = useModelState<DataView>("_azimuthal_data");

  // CSV/JSON export; column schema kept in sync with the Python exporter
  const exportMeasurements = React.useCallback((format: "csv" | "json", kind: "spots" | "rings" | "all" = "all") => {
    const records = buildMeasurementRecords(
      kind === "rings" ? [] : spots || [],
      kind === "spots" ? [] : rings || [],
    );
    const basename = kind === "all" ? "measurements" : kind;
    if (format === "json") {
      const metadata = measurementMetadata({
        centerRow, centerCol, centerMethod, kPixelSize, kCalibrated,
        calibrationSource, calibrationRefD, calibrationRefRadius,
        maskRegions: maskRegions || [], backgroundSubtracted: profileSubtract,
      });
      const blob = new Blob([JSON.stringify({ metadata, measurements: records }, null, 2)], { type: "application/json" });
      downloadBlob(blob, `${basename}.json`);
    } else {
      downloadBlob(new Blob([measurementCsv(records)], { type: "text/csv" }), `${basename}.csv`);
    }
  }, [spots, rings, centerRow, centerCol, centerMethod, kPixelSize, kCalibrated,
    calibrationSource, calibrationRefD, calibrationRefRadius, maskRegions, profileSubtract]);

  // Local UI state
  const isMobile = useMobileViewport();
  const initialCanvasSize = React.useMemo(() => {
    const requested = Number(panelWidthPx);
    return Number.isFinite(requested) && requested > 0 ? Math.max(CANVAS_MIN, Math.round(requested)) : CANVAS_MIN;
  }, [panelWidthPx]);
  const hasResizedCanvasRef = React.useRef(false);
  const [canvasSize, setCanvasSize] = React.useState(initialCanvasSize);
  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [resizeCanvasStart, setResizeCanvasStart] = React.useState<{ x: number; y: number; size: number } | null>(null);
  const [dpZoom, setDpZoom] = React.useState(1);
  const [dpPanX, setDpPanX] = React.useState(0);
  const [dpPanY, setDpPanY] = React.useState(0);
  const [dpHistData, setDpHistData] = React.useState<Float32Array | null>(null);
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(null);
  const [dpExportAnchor, setDpExportAnchor] = React.useState<HTMLElement | null>(null);
  const [phaseMenuAnchor, setPhaseMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [maskMenuAnchor, setMaskMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [refineMenuAnchor, setRefineMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [showQc, setShowQc] = React.useState(false);
  const toggleQuality = React.useCallback(() => {
    const next = !showQc;
    setShowQc(next);
    if (next) setQualityRequest(true);
  }, [showQc, setQualityRequest]);
  const [identifyCollapsed, setIdentifyCollapsed] = React.useState(false);
  const [expandedPhaseId, setExpandedPhaseId] = React.useState<string | null>(null);
  const [customName, setCustomName] = React.useState("");
  const [customA, setCustomA] = React.useState("");
  const [customB, setCustomB] = React.useState("");
  const [customC, setCustomC] = React.useState("");
  const [customAlpha, setCustomAlpha] = React.useState("");
  const [customBeta, setCustomBeta] = React.useState("");
  const [customGamma, setCustomGamma] = React.useState("");
  const [customAbsences, setCustomAbsences] = React.useState("fcc");
  const [wedgeStart, setWedgeStart] = React.useState("");
  const [wedgeEnd, setWedgeEnd] = React.useState("");
  const [diskRadius, setDiskRadius] = React.useState("");
  const [dKnown, setDKnown] = React.useState("");

  React.useEffect(() => {
    if (!hasResizedCanvasRef.current) {
      setCanvasSize(initialCanvasSize);
    }
  }, [initialCanvasSize]);

  // Canvas drag modes
  const [moveSpots, setMoveSpots] = React.useState(false);
  const [drawMode, setDrawMode] = React.useState<"disk" | "wedge" | null>(null);
  type DragPreview =
    | { kind: "spot"; id: number; row: number; col: number }
    | { kind: "disk"; row: number; col: number; radius: number }
    | { kind: "wedge"; start_deg: number; end_deg: number };
  const [dragPreview, setDragPreview] = React.useState<DragPreview | null>(null);
  const dragTargetRef = React.useRef<
    | { kind: "spot"; id: number }
    | { kind: "disk"; row: number; col: number }
    | { kind: "wedge"; start_deg: number }
    | null
  >(null);
  const dragPosRef = React.useRef({ row: 0, col: 0 });
  const dragRafRef = React.useRef(0);

  // Smooth scrubbing
  const [localFrame, setLocalFrame] = React.useState(frameIdx);
  React.useEffect(() => { setLocalFrame(frameIdx); }, [frameIdx]);

  // Identify table
  React.useEffect(() => { setIdentifyCollapsed(false); }, [identifyResults]);

  // Center zoom (pixel-center convention)
  const zoomToCenter = React.useCallback(() => {
    const Z = 2.5;
    setDpZoom(Z);
    setDpPanX(canvasSize * Z * (0.5 - (centerCol + 0.5) / Math.max(detCols, 1)));
    setDpPanY(canvasSize * Z * (0.5 - (centerRow + 0.5) / Math.max(detRows, 1)));
  }, [canvasSize, centerRow, centerCol, detRows, detCols]);

  const dpCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const dpUiRef = React.useRef<HTMLCanvasElement>(null);
  const dpScaleRef = React.useRef<HTMLCanvasElement>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const azimuthalCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const dpOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const [dpVersion, setDpVersion] = React.useState(0);
  const dpVminRef = React.useRef(0);
  const dpVmaxRef = React.useRef(1);

  const decodeHalves = (data: DataView, on: boolean) => {
    if (!on) return null;
    const arr = extractFloat32(data);
    if (!arr || arr.length < 4 || arr.length % 2 !== 0) return null;
    const n = arr.length / 2;
    return { x: arr.subarray(0, n), y: arr.subarray(n) };
  };

  // Curve painter
  const drawCurvePanel = React.useCallback((
    canvas: HTMLCanvasElement | null,
    data: { x: Float32Array; y: Float32Array } | null,
    xLabel: string, yLabel: string, zeroLine: boolean,
    opts?: { logY?: boolean; rings?: RingDict[]; bfRadius?: number; selectedRingId?: number },
  ) => {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasSize * DPR;
    canvas.height = PROFILE_H * DPR;
    ctx.scale(DPR, DPR);
    const isDark = themeInfo.theme === "dark";
    ctx.fillStyle = isDark ? "#050505" : "#ffffff";
    ctx.fillRect(0, 0, canvasSize, PROFILE_H);
    if (!data) return;
    const plotW = canvasSize - PROFILE_PAD.left - PROFILE_PAD.right;
    const plotH = PROFILE_H - PROFILE_PAD.top - PROFILE_PAD.bottom;
    const n = data.x.length;
    const xMin = data.x[0], xMax = data.x[n - 1];
    let vMin = Infinity, vMax = -Infinity;
    for (let i = 0; i < n; i++) {
      if (data.y[i] < vMin) vMin = data.y[i];
      if (data.y[i] > vMax) vMax = data.y[i];
    }
    const xOf = (x: number) => PROFILE_PAD.left + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * plotW;
    const yOf = (v: number) => {
      const t = opts?.logY
        ? Math.log1p(Math.max(0, v - vMin)) / Math.max(1e-9, Math.log1p(vMax - vMin))
        : (v - vMin) / Math.max(1e-9, vMax - vMin);
      return PROFILE_PAD.top + (1 - t) * plotH;
    };
    ctx.strokeStyle = isDark ? "#333333" : "#d8d8d8";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = PROFILE_PAD.top + (g / 4) * plotH;
      ctx.beginPath(); ctx.moveTo(PROFILE_PAD.left, y); ctx.lineTo(canvasSize - PROFILE_PAD.right, y); ctx.stroke();
    }
    if (zeroLine && vMin < 0 && vMax > 0) {
      ctx.strokeStyle = isDark ? "#666666" : "#999999";
      ctx.beginPath(); ctx.moveTo(PROFILE_PAD.left, yOf(0)); ctx.lineTo(canvasSize - PROFILE_PAD.right, yOf(0)); ctx.stroke();
    }
    const ringColor = isDark ? "#ffb74d" : "#e65100";
    for (const ring of opts?.rings || []) {
      const x = xOf(ring.radius_px);
      const selected = opts?.selectedRingId != null && opts.selectedRingId !== 0 && ring.id === opts.selectedRingId;
      ctx.strokeStyle = selected ? themeColors.accent : ringColor;
      ctx.lineWidth = selected ? 2.5 : 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, PROFILE_PAD.top); ctx.lineTo(x, PROFILE_PAD.top + plotH); ctx.stroke();
      ctx.setLineDash([]);
      ctx.lineWidth = 1;
      ctx.fillStyle = selected ? themeColors.accent : ringColor;
      ctx.font = "9px -apple-system, sans-serif";
      ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(`${ring.id}`, x + 2, PROFILE_PAD.top);
      if (kCalibrated && kPixelSize > 0 && ring.radius_px > 0) {
        ctx.fillText(`${(1 / (ring.radius_px * kPixelSize)).toFixed(2)}Å`, x + 2, PROFILE_PAD.top + 10);
      }
    }
    if (opts?.bfRadius && opts.bfRadius > 0 && opts.bfRadius <= xMax) {
      ctx.strokeStyle = isDark ? "#333333" : "#d8d8d8";
      ctx.beginPath();
      ctx.moveTo(xOf(opts.bfRadius), PROFILE_PAD.top + plotH - 6);
      ctx.lineTo(xOf(opts.bfRadius), PROFILE_PAD.top + plotH);
      ctx.stroke();
    }
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const px = xOf(data.x[i]), py = yOf(data.y[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.fillStyle = isDark ? "#dddddd" : "#222222";
    ctx.font = "10px -apple-system, sans-serif";
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(yLabel, 4, 4);
    ctx.textBaseline = "bottom";
    ctx.fillText(xLabel, PROFILE_PAD.left, PROFILE_H - 4);
    ctx.textAlign = "center";
    for (const frac of [0.25, 0.5, 0.75, 1.0]) {
      const xv = xMin + (xMax - xMin) * frac;
      ctx.fillText(xv >= 100 ? `${Math.round(xv)}` : xv.toFixed(1), xOf(xv), PROFILE_H - 16);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }, [canvasSize, themeInfo.theme, themeColors.accent, kCalibrated, kPixelSize]);

  const azimuthalArrays = React.useMemo(() => decodeHalves(azimuthalData, showAzimuthal), [azimuthalData, showAzimuthal]);

  React.useLayoutEffect(() => {
    if (showAzimuthal) drawCurvePanel(azimuthalCanvasRef.current, azimuthalArrays, "θ (°)", "I", false);
  }, [azimuthalArrays, showAzimuthal, drawCurvePanel]);

  const profileArrays = React.useMemo(() => decodeHalves(profileData, showProfile), [profileData, showProfile]);

  const handleProfileClick = React.useCallback((e: React.PointerEvent) => {
    const canvas = profileCanvasRef.current;
    if (!canvas || !profileArrays) return;
    const rect = canvas.getBoundingClientRect();
    const scale = rect.width > 0 ? canvasSize / rect.width : 1;
    const plotW = canvasSize - PROFILE_PAD.left - PROFILE_PAD.right;
    const rMax = profileArrays.x[profileArrays.x.length - 1];
    const radius = (((e.clientX - rect.left) * scale - PROFILE_PAD.left) / Math.max(1, plotW)) * rMax;
    if (!(radius > 0 && radius <= rMax)) return;
    // Ring selection
    const hit = (rings || []).find((r) => Math.abs(r.radius_px - radius) <= 3);
    if (hit) {
      setSelectedRingId(selectedRingId === hit.id ? 0 : hit.id);
      return;
    }
    setRingAddRequest([radius]);
  }, [profileArrays, canvasSize, rings, selectedRingId, setSelectedRingId, setRingAddRequest]);

  React.useLayoutEffect(() => {
    if (showProfile) {
      drawCurvePanel(profileCanvasRef.current, profileArrays, "r (px)", profileLog ? "log I" : "I",
        false, { logY: profileLog, rings, bfRadius, selectedRingId });
    }
  }, [profileArrays, profileLog, rings, bfRadius, showProfile, selectedRingId, drawCurvePanel]);

  // Canvas resize
  const handleCanvasResizeStart = (e: React.PointerEvent) => {
    e.stopPropagation();
    e.preventDefault();
    hasResizedCanvasRef.current = true;
    setIsResizingCanvas(true);
    setResizeCanvasStart({ x: e.clientX, y: e.clientY, size: canvasSize });
  };

  React.useEffect(() => {
    if (!isResizingCanvas) return;
    let rafId = 0;
    let latestSize = resizeCanvasStart ? resizeCanvasStart.size : canvasSize;
    const handleMouseMove = (e: PointerEvent) => {
      if (!resizeCanvasStart) return;
      const delta = Math.max(e.clientX - resizeCanvasStart.x, e.clientY - resizeCanvasStart.y);
      latestSize = Math.max(CANVAS_MIN, resizeCanvasStart.size + delta);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setCanvasSize(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      cancelAnimationFrame(rafId);
      setCanvasSize(latestSize);
      setIsResizingCanvas(false);
      setResizeCanvasStart(null);
    };
    document.addEventListener("pointermove", handleMouseMove);
    document.addEventListener("pointerup", handleMouseUp);
    document.addEventListener("pointercancel", handleMouseUp);
    return () => {
      cancelAnimationFrame(rafId);
      document.removeEventListener("pointermove", handleMouseMove);
      document.removeEventListener("pointerup", handleMouseUp);
      document.removeEventListener("pointercancel", handleMouseUp);
    };
  }, [isResizingCanvas, resizeCanvasStart]);

  // Colormap LUT
  const dpLut = React.useMemo(() => {
    const base = COLORMAPS[dpColormap] || COLORMAPS.inferno;
    if (!dpInvert) return base;
    const n = base.length / 3;
    const inv = new Uint8Array(base.length);
    for (let k = 0; k < n; k++) {
      const s = (n - 1 - k) * 3, d = k * 3;
      inv[d] = base[s]; inv[d + 1] = base[s + 1]; inv[d + 2] = base[s + 2];
    }
    return inv;
  }, [dpColormap, dpInvert]);

  // Frame bytes
  const activeFrame = React.useMemo<Float32Array | null>(() => {
    const frameLen = detRows * detCols;
    if (offline && offlineFrames && frameLen > 0
        && offlineFrames.byteLength >= frameLen * 4 * nFrames) {
      const stack = extractFloat32(offlineFrames);
      const idx = Math.max(0, Math.min(frameIdx, nFrames - 1));
      if (stack && stack.length >= frameLen * (idx + 1)) {
        return stack.subarray(idx * frameLen, (idx + 1) * frameLen);
      }
    }
    return extractFloat32(frameBytes, frameLen);
  }, [offline, offlineFrames, frameBytes, frameIdx, nFrames, detRows, detCols]);

  // offline stacks: stats recompute in JS; panes flag the baked frame
  const bakedFrameRef = React.useRef(frameIdx);
  const offlineStats = React.useMemo(
    () => (offline && nFrames > 1 && activeFrame ? frameStats(activeFrame) : null),
    [offline, nFrames, activeFrame],
  );
  const displayStats = offlineStats ?? dpStats;
  const paneStaleNote = staleFrameNote(offline, nFrames, frameIdx, bakedFrameRef.current);

  // Frame scaling
  const scaledFrame = React.useMemo(() => {
    const raw = activeFrame;
    if (!raw || raw.length === 0) return null;
    let scaled: Float32Array;
    if (dpScaleMode === "log") {
      scaled = new Float32Array(raw.length);
      applyLogScaleInPlace(raw, scaled);
    } else if (dpScaleMode === "sqrt") {
      scaled = new Float32Array(raw.length);
      let mn = Infinity;
      for (let i = 0; i < raw.length; i++) if (raw[i] < mn) mn = raw[i];
      for (let i = 0; i < raw.length; i++) scaled[i] = Math.sqrt(Math.max(raw[i] - mn, 0));
    } else {
      scaled = raw;
    }
    const { min: dataMin, max: dataMax } = findDataRange(scaled);
    return { scaled, dataMin, dataMax };
  }, [activeFrame, dpScaleMode]);

  // Contrast painter
  const paintContrast = React.useCallback((loPct: number, hiPct: number) => {
    if (!scaledFrame) return;
    const { vmin, vmax } = sliderRange(scaledFrame.dataMin, scaledFrame.dataMax, loPct, hiPct);
    dpVminRef.current = vmin;
    dpVmaxRef.current = vmax;
    let offscreen = dpOffscreenRef.current;
    if (!offscreen) { offscreen = document.createElement("canvas"); dpOffscreenRef.current = offscreen; }
    offscreen.width = detCols;
    offscreen.height = detRows;
    const ctx = offscreen.getContext("2d");
    if (!ctx) return;
    const imgData = ctx.createImageData(detCols, detRows);
    applyColormap(scaledFrame.scaled, imgData.data, dpLut, vmin, vmax);
    ctx.putImageData(imgData, 0, 0);
    setDpVersion(v => v + 1);
  }, [scaledFrame, dpLut, detRows, detCols]);

  // Frame render
  React.useEffect(() => {
    if (!scaledFrame) return;
    paintContrast(dpVminPct, dpVmaxPct);
    setDpHistData(scaledFrame.scaled);
  }, [scaledFrame, paintContrast, dpVminPct, dpVmaxPct]);

  // Contrast preview: rAF-throttled paint, traits written once on commit
  const pendingContrastRef = React.useRef<[number, number] | null>(null);
  const contrastRafRef = React.useRef(0);
  const previewContrast = React.useCallback((lo: number, hi: number) => {
    pendingContrastRef.current = [lo, hi];
    if (!contrastRafRef.current) {
      contrastRafRef.current = window.requestAnimationFrame(() => {
        contrastRafRef.current = 0;
        const pending = pendingContrastRef.current;
        if (pending) paintContrast(pending[0], pending[1]);
      });
    }
  }, [paintContrast]);
  const commitContrast = React.useCallback((lo: number, hi: number) => {
    if (contrastRafRef.current) {
      window.cancelAnimationFrame(contrastRafRef.current);
      contrastRafRef.current = 0;
    }
    pendingContrastRef.current = null;
    paintContrast(lo, hi);
    setDpVminPct(lo);
    setDpVmaxPct(hi);
  }, [paintContrast, setDpVminPct, setDpVmaxPct]);
  React.useEffect(() => () => {
    if (contrastRafRef.current) window.cancelAnimationFrame(contrastRafRef.current);
  }, []);

  // Frame canvas
  React.useLayoutEffect(() => {
    const canvas = dpCanvasRef.current;
    const offscreen = dpOffscreenRef.current;
    if (!canvas || !offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasSize;
    canvas.height = canvasSize;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasSize, canvasSize);
    const offX = (canvasSize - canvasSize * dpZoom) / 2 + dpPanX;
    const offY = (canvasSize - canvasSize * dpZoom) / 2 + dpPanY;
    ctx.drawImage(offscreen, offX, offY, canvasSize * dpZoom, canvasSize * dpZoom);
  }, [dpVersion, dpZoom, dpPanX, dpPanY, canvasSize, detRows, detCols]);

  // Overlays
  React.useLayoutEffect(() => {
    const canvas = dpUiRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const cssW = canvasSize;
    canvas.width = cssW * DPR;
    canvas.height = cssW * DPR;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, cssW, cssW);

    // per-axis transform for non-square detectors
    const view = viewTransform(cssW, dpZoom, dpPanX, dpPanY, detRows, detCols);
    const { scX, scY } = view;

    // Center crosshair
    const cx = dataColToScreenX(centerCol, view);
    const cy = dataRowToScreenY(centerRow, view);
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx - 10, cy); ctx.lineTo(cx + 10, cy);
    ctx.moveTo(cx, cy - 10); ctx.lineTo(cx, cy + 10);
    ctx.stroke();
    // BF disk circle
    ctx.beginPath();
    ctx.ellipse(cx, cy, bfRadius * scX, bfRadius * scY, 0, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);

    // Spot markers
    if (spots && spots.length > 0) {
      spots.forEach((spot, i) => {
        const dragged = dragPreview?.kind === "spot" && dragPreview.id === spot.id ? dragPreview : null;
        const sx = dataColToScreenX(dragged ? dragged.col : spot.col, view);
        const sy = dataRowToScreenY(dragged ? dragged.row : spot.row, view);
        const color = spotColorAt(i);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.font = "bold 10px -apple-system, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(`${spot.id}`, sx + 8, sy - 2);
        if (showHkl && spot.hkl) {
          ctx.textBaseline = "top";
          ctx.fillText(spot.hkl, sx + 8, sy + 2);
          ctx.textBaseline = "bottom";
        }
      });
    }

    // Rings
    if (rings && rings.length > 0) {
      const ringColor = themeInfo.theme === "dark" ? "#ffb74d" : "#e65100";
      for (const ring of rings) {
        const selected = selectedRingId !== 0 && ring.id === selectedRingId;
        ctx.strokeStyle = selected ? themeColors.accent : ringColor;
        ctx.lineWidth = selected ? 2.5 : 1.2;
        ctx.beginPath();
        ctx.ellipse(cx, cy, ring.radius_px * scX, ring.radius_px * scY, 0, 0, 2 * Math.PI);
        ctx.stroke();
        if (showHkl && ring.hkl) {
          const rrX = ring.radius_px * scX * Math.SQRT1_2;
          const rrY = ring.radius_px * scY * Math.SQRT1_2;
          ctx.fillStyle = selected ? themeColors.accent : ringColor;
          ctx.font = "bold 10px -apple-system, sans-serif";
          ctx.textAlign = "left";
          ctx.textBaseline = "bottom";
          ctx.fillText(ring.hkl, cx + rrX + 4, cy - rrY - 4);
        }
      }
    }

    // Excluded mask regions
    if (showMask && maskRegions && maskRegions.length > 0) {
      ctx.save();
      ctx.fillStyle = themeInfo.theme === "dark" ? "rgba(244,67,54,0.18)" : "rgba(211,47,47,0.15)";
      for (const region of maskRegions) {
        if (region.kind === "disk" && region.radius != null) {
          ctx.beginPath();
          ctx.ellipse(
            dataColToScreenX(region.col ?? 0, view), dataRowToScreenY(region.row ?? 0, view),
            region.radius * scX, region.radius * scY, 0, 0, 2 * Math.PI,
          );
          ctx.fill();
        } else if (region.kind === "wedge" && region.start_deg != null && region.end_deg != null) {
          const rBig = Math.hypot(detRows * scY, detCols * scX);
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.arc(cx, cy, rBig, dataAngleToScreen(region.start_deg, view), dataAngleToScreen(region.end_deg, view));
          ctx.closePath();
          ctx.fill();
        }
      }
      ctx.restore();
    }

    // Draw-mode preview
    if (dragPreview && dragPreview.kind !== "spot") {
      ctx.save();
      ctx.strokeStyle = themeInfo.theme === "dark" ? "rgba(244,67,54,0.9)" : "rgba(211,47,47,0.9)";
      ctx.fillStyle = themeInfo.theme === "dark" ? "rgba(244,67,54,0.18)" : "rgba(211,47,47,0.15)";
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      if (dragPreview.kind === "disk") {
        ctx.ellipse(
          dataColToScreenX(dragPreview.col, view), dataRowToScreenY(dragPreview.row, view),
          dragPreview.radius * scX, dragPreview.radius * scY, 0, 0, 2 * Math.PI,
        );
      } else {
        const rBig = Math.hypot(detRows * scY, detCols * scX);
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, rBig, dataAngleToScreen(dragPreview.start_deg, view), dataAngleToScreen(dragPreview.end_deg, view));
        ctx.closePath();
      }
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // Fitted ellipse
    if (ellipseRatio > 1.002 && rings && rings.length > 0) {
      const rMax = Math.max(...rings.map((r) => r.radius_px));
      const s = Math.sqrt(ellipseRatio);
      ctx.save();
      ctx.strokeStyle = themeInfo.theme === "dark" ? "#4dd0e1" : "#00838f";
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      // build the path in data units, then stroke with a uniform screen pen
      ctx.save();
      ctx.translate(cx, cy);
      ctx.scale(scX, scY);
      ctx.ellipse(0, 0, rMax * s, rMax / s, (ellipseAngle * Math.PI) / 180, 0, 2 * Math.PI);
      ctx.restore();
      ctx.stroke();
      ctx.restore();
    }

    drawColorbar(ctx, cssW, cssW, dpLut, dpVminRef.current, dpVmaxRef.current, dpScaleMode === "log");

    if (dpZoom !== 1) {
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.font = "11px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${dpZoom.toFixed(1)}×`, 8, cssW - 8);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }, [dpVersion, dpZoom, dpPanX, dpPanY, canvasSize, detRows, detCols, centerRow, centerCol, bfRadius, spots, rings, showHkl, selectedRingId, ellipseRatio, ellipseAngle, maskRegions, showMask, dragPreview, dpLut, dpScaleMode, themeInfo.theme, themeColors.accent]);

  // Scale bar
  React.useLayoutEffect(() => {
    const canvas = dpScaleRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasSize * DPR;
    canvas.height = canvasSize * DPR;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (kCalibrated && kPixelSize > 0) {
      drawScaleBarHiDPI(canvas, DPR, dpZoom, kPixelSize, "1/Å", detCols);
    }
  }, [canvasSize, dpZoom, kCalibrated, kPixelSize, detCols]);

  // Pointer handlers
  const dpIsDragging = React.useRef(false);
  const dpDragStart = React.useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  // CSS px to internal canvas px (canvas can render narrower than canvasSize)
  const dpDisplayScale = () => {
    const canvas = dpCanvasRef.current;
    if (!canvas) return 1;
    const rect = canvas.getBoundingClientRect();
    return rect.width > 0 ? canvasSize / rect.width : 1;
  };

  // canvas to data (pixel-center)
  const dpToImage = (e: { clientX: number; clientY: number }) => {
    const canvas = dpCanvasRef.current;
    if (!canvas) return { row: 0, col: 0 };
    const rect = canvas.getBoundingClientRect();
    const scale = rect.width > 0 ? canvasSize / rect.width : 1;
    const mx = (e.clientX - rect.left) * scale;
    const my = (e.clientY - rect.top) * scale;
    return screenToData(mx, my, viewTransform(canvasSize, dpZoom, dpPanX, dpPanY, detRows, detCols));
  };

  const angleOf = (row: number, col: number) =>
    (Math.atan2(row - centerRow, col - centerCol) * 180 / Math.PI + 360) % 360;

  // Two-finger pinch (zoom + pan) via tracked pointers
  const activePointersRef = React.useRef(new Map<number, { x: number; y: number }>());
  const pinchStartRef = React.useRef<
    | { dist: number; midX: number; midY: number; zoom: number; offX: number; offY: number }
    | null
  >(null);
  const pinchRafRef = React.useRef(0);
  const multiTouchRef = React.useRef(false);
  const lastTapRef = React.useRef({ time: 0, x: 0, y: 0 });
  const pendingTapRef = React.useRef<{ x: number; y: number; row: number; col: number } | null>(null);

  const cancelCanvasDrag = () => {
    dragTargetRef.current = null;
    cancelAnimationFrame(dragRafRef.current);
    dragRafRef.current = 0;
    setDragPreview(null);
  };

  const applyPinch = () => {
    const start = pinchStartRef.current;
    const points = Array.from(activePointersRef.current.values());
    if (!start || points.length < 2) return;
    const scale = dpDisplayScale();
    const dist = Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y) * scale;
    const canvas = dpCanvasRef.current;
    if (!canvas || dist <= 0 || start.dist <= 0) return;
    const rect = canvas.getBoundingClientRect();
    const midX = ((points[0].x + points[1].x) / 2 - rect.left) * scale;
    const midY = ((points[0].y + points[1].y) / 2 - rect.top) * scale;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, start.zoom * (dist / start.dist)));
    const zoomRatio = newZoom / start.zoom;
    // keep the content point under the start midpoint anchored to the live midpoint
    const newOffX = midX - (start.midX - start.offX) * zoomRatio;
    const newOffY = midY - (start.midY - start.offY) * zoomRatio;
    setDpZoom(newZoom);
    setDpPanX(newOffX - (canvasSize - canvasSize * newZoom) / 2);
    setDpPanY(newOffY - (canvasSize - canvasSize * newZoom) / 2);
  };

  const beginPinch = () => {
    const points = Array.from(activePointersRef.current.values());
    const canvas = dpCanvasRef.current;
    if (!canvas || points.length < 2) return;
    const rect = canvas.getBoundingClientRect();
    const scale = rect.width > 0 ? canvasSize / rect.width : 1;
    pinchStartRef.current = {
      dist: Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y) * scale,
      midX: ((points[0].x + points[1].x) / 2 - rect.left) * scale,
      midY: ((points[0].y + points[1].y) / 2 - rect.top) * scale,
      zoom: dpZoom,
      offX: (canvasSize - canvasSize * dpZoom) / 2 + dpPanX,
      offY: (canvasSize - canvasSize * dpZoom) / 2 + dpPanY,
    };
  };

  const handleDpPointerDown = (e: React.PointerEvent) => {
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* unsupported host */ }
    activePointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (activePointersRef.current.size === 2) {
      // second finger: switch to pinch, cancel any single-pointer gesture
      multiTouchRef.current = true;
      dpIsDragging.current = false;
      pendingTapRef.current = null;
      cancelCanvasDrag();
      beginPinch();
      return;
    }
    if (pinchStartRef.current) return;
    if (e.button === 1 || e.button === 2 || e.shiftKey || (e.pointerType === "touch" && dpZoom !== 1 && drawMode === null && !moveSpots)) {
      dpIsDragging.current = true;
      dpDragStart.current = { x: e.clientX, y: e.clientY, panX: dpPanX, panY: dpPanY };
      return;
    }
    const { row, col } = dpToImage(e);
    // pixel-center coords: the detector spans [-0.5, det - 0.5)
    if (!(row >= -0.5 && row < detRows - 0.5 && col >= -0.5 && col < detCols - 0.5)) return;
    dragPosRef.current = { row, col };
    if (drawMode === "disk") {
      dragTargetRef.current = { kind: "disk", row, col };
      setDragPreview({ kind: "disk", row, col, radius: 0 });
      return;
    }
    if (drawMode === "wedge") {
      const start = angleOf(row, col);
      dragTargetRef.current = { kind: "wedge", start_deg: start };
      setDragPreview({ kind: "wedge", start_deg: start, end_deg: start });
      return;
    }
    if (moveSpots && spots && spots.length > 0) {
      // per-axis screen distance
      const { scX, scY } = viewTransform(canvasSize, dpZoom, 0, 0, detRows, detCols);
      let nearest = -1, nearestDist = Infinity;
      spots.forEach((s, i) => {
        const dist = Math.hypot((s.row - row) * scY, (s.col - col) * scX);
        if (dist < nearestDist) { nearestDist = dist; nearest = i; }
      });
      const hitPx = (e.pointerType === "touch" ? 20 : 12) * dpDisplayScale();
      if (nearest >= 0 && nearestDist <= hitPx) {
        dragTargetRef.current = { kind: "spot", id: spots[nearest].id };
        setDragPreview({ kind: "spot", id: spots[nearest].id, row, col });
        return;
      }
    }
    // touch: defer tap actions to pointerup so a pinch's first finger fires nothing
    if (e.pointerType === "touch") {
      pendingTapRef.current = { x: e.clientX, y: e.clientY, row, col };
      return;
    }
    commitTapAction(row, col);
  };

  const commitTapAction = (row: number, col: number) => {
    if (centerMode === "manual") {
      setCenterRow(row);
      setCenterCol(col);
      return;
    }
    setSpotAddRequest([row, col]);
  };

  const handleDpPointerMove = (e: React.PointerEvent) => {
    if (activePointersRef.current.has(e.pointerId)) {
      activePointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    }
    const pendingTap = pendingTapRef.current;
    if (pendingTap && Math.hypot(e.clientX - pendingTap.x, e.clientY - pendingTap.y) > 10) {
      pendingTapRef.current = null;  // finger wandered: not a tap
    }
    if (pinchStartRef.current) {
      e.preventDefault();
      if (!pinchRafRef.current) {
        pinchRafRef.current = requestAnimationFrame(() => {
          pinchRafRef.current = 0;
          applyPinch();
        });
      }
      return;
    }
    if (dpIsDragging.current) {
      const scale = dpDisplayScale();
      setDpPanX(dpDragStart.current.panX + (e.clientX - dpDragStart.current.x) * scale);
      setDpPanY(dpDragStart.current.panY + (e.clientY - dpDragStart.current.y) * scale);
      return;
    }
    if (dragTargetRef.current) {
      dragPosRef.current = dpToImage(e);
      if (!dragRafRef.current) {
        dragRafRef.current = requestAnimationFrame(() => {
          dragRafRef.current = 0;
          const target = dragTargetRef.current;
          if (!target) return;
          const { row, col } = dragPosRef.current;
          if (target.kind === "spot") {
            setDragPreview({ kind: "spot", id: target.id, row, col });
          } else if (target.kind === "disk") {
            setDragPreview({ kind: "disk", row: target.row, col: target.col, radius: Math.hypot(row - target.row, col - target.col) });
          } else {
            setDragPreview({ kind: "wedge", start_deg: target.start_deg, end_deg: angleOf(row, col) });
          }
        });
      }
      return;
    }
    if (!activeFrame) return;
    const { row, col } = dpToImage(e);
    const ri = Math.round(row), ci = Math.round(col);
    if (ri >= 0 && ri < detRows && ci >= 0 && ci < detCols) {
      const raw = activeFrame;
      setCursorInfo({ row: ri, col: ci, value: raw[ri * detCols + ci] });
    } else {
      setCursorInfo(null);
    }
  };

  const releasePointer = (e: React.PointerEvent) => {
    activePointersRef.current.delete(e.pointerId);
    if (activePointersRef.current.size === 0) multiTouchRef.current = false;
    if (pinchStartRef.current) {
      if (activePointersRef.current.size < 2) {
        pinchStartRef.current = null;
        cancelAnimationFrame(pinchRafRef.current);
        pinchRafRef.current = 0;
        dpIsDragging.current = false;
      } else {
        beginPinch();  // remaining fingers become the new baseline
      }
    }
  };

  const handleDpPointerUp = (e: React.PointerEvent) => {
    const wasMultiTouch = multiTouchRef.current;
    const pendingTap = pendingTapRef.current;
    pendingTapRef.current = null;
    releasePointer(e);
    if (wasMultiTouch) return;
    // double-tap resets the view (touch counterpart of double-click)
    if (e.pointerType === "touch" && !dragTargetRef.current && !dpIsDragging.current) {
      const now = performance.now();
      const last = lastTapRef.current;
      if (now - last.time < 320 && Math.hypot(e.clientX - last.x, e.clientY - last.y) < 24) {
        lastTapRef.current = { time: 0, x: 0, y: 0 };
        resetDpView();
        return;
      }
      lastTapRef.current = { time: now, x: e.clientX, y: e.clientY };
    }
    if (pendingTap) {
      commitTapAction(pendingTap.row, pendingTap.col);
      return;
    }
    dpIsDragging.current = false;
    const target = dragTargetRef.current;
    if (!target) return;
    const { row, col } = dragPosRef.current;
    if (target.kind === "spot") {
      setSpotMoveRequest([target.id, row, col]);
    } else if (target.kind === "disk") {
      const radius = Math.hypot(row - target.row, col - target.col);
      if (radius >= 2) {
        setMaskRegions([...(maskRegions || []), { kind: "disk", row: Math.round(target.row * 10) / 10, col: Math.round(target.col * 10) / 10, radius: Math.round(radius * 10) / 10 }]);
        setShowMask(true);
      }
      setDrawMode(null);
    } else {
      const end = angleOf(row, col);
      if (end !== target.start_deg) {
        setMaskRegions([...(maskRegions || []), { kind: "wedge", start_deg: Math.round(target.start_deg * 10) / 10, end_deg: Math.round(end * 10) / 10 }]);
        setShowMask(true);
      }
      setDrawMode(null);
    }
    cancelCanvasDrag();
  };
  const handleDpPointerCancel = (e: React.PointerEvent) => {
    releasePointer(e);
    pendingTapRef.current = null;
    dpIsDragging.current = false;
    cancelCanvasDrag();
    setCursorInfo(null);
  };
  // pointer capture keeps drags alive past the edge; leave only clears the readout
  const handleDpPointerLeave = () => { setCursorInfo(null); };

  // Scroll zoom
  const handleDpWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setDpZoom(z => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z * delta)));
  };

  const resetDpView = () => { setDpZoom(1); setDpPanX(0); setDpPanY(0); };

  // Wheel guard
  const dpContainerRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const prevent = (e: WheelEvent) => e.preventDefault();
    const dp = dpContainerRef.current;
    if (dp) dp.addEventListener("wheel", prevent, { passive: false });
    return () => {
      if (dp) dp.removeEventListener("wheel", prevent);
    };
  }, []);

  // Export handlers
  const handleCopyDP = () => {
    const offscreen = dpOffscreenRef.current;
    if (!offscreen) return;
    offscreen.toBlob((blob) => {
      if (blob) {
        try { navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]); }
        catch { downloadBlob(blob, "diffraction.png"); }
      }
    });
  };

  const handleExportPng = () => {
    setDpExportAnchor(null);
    if (!dpCanvasRef.current) return;
    dpCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "showdiffraction_dp.png"); }, "image/png");
  };

  // HTML request
  const handleExportHtml = () => {
    setDpExportAnchor(null);
    exportCounterRef.current += 1;
    const id = `html-${exportCounterRef.current}`;
    const slug = (title || "showdiffraction")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "") || "showdiffraction";
    const filename = `${slug}_${nFrames}x${detRows}x${detCols}.html`;
    pendingExportRef.current = id;
    setExportRequest(JSON.stringify({ mode: "single", download: true, id, filename }));
  };

  // HTML payload
  React.useEffect(() => {
    if (!exportPayloadId || exportPayloadId !== pendingExportRef.current) return;
    const bytes = extractBytes(exportPayload);
    if (bytes.length === 0) return;
    const payload = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
      ? bytes
      : bytes.slice();
    const filename = exportPayloadFilename || "showdiffraction.html";
    downloadBlob(new Blob([payload as BlobPart], { type: "text/html;charset=utf-8" }), filename);
    pendingExportRef.current = "";
    setExportRequest(JSON.stringify({ mode: "clear" }));
  }, [exportPayload, exportPayloadId, exportPayloadFilename, setExportRequest]);

  // Keyboard
  // Typing guard
  const isTypingTarget = React.useCallback((target: EventTarget | null): boolean => {
    if (!(target instanceof HTMLElement)) return false;
    if (target.isContentEditable) return true;
    return target.closest("input, textarea, select, [role='textbox'], [contenteditable='true']") !== null;
  }, []);

  const handleRootMouseDownCapture = React.useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement | null;
    if (target?.closest("canvas")) rootRef.current?.focus();
  }, []);

  const handleKeyDown = React.useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    if (isTypingTarget(e.target)) return;

    const step = e.shiftKey ? 10 : 1;
    let handled = false;

    switch (e.key) {
      case "ArrowLeft":
        if (nFrames > 1) { setFrameIdx(Math.max(0, frameIdx - step)); handled = true; }
        break;
      case "ArrowRight":
        if (nFrames > 1) { setFrameIdx(Math.min(nFrames - 1, frameIdx + step)); handled = true; }
        break;
      case "r":
      case "R":
        resetDpView();
        handled = true;
        break;
      case "z":
      case "Z":
        setSpotUndoRequest(true);
        handled = true;
        break;
      case "Escape":
        setDrawMode(null);
        cancelCanvasDrag();
        handled = true;
        break;
    }

    if (handled) {
      e.preventDefault();
      e.stopPropagation();
    }
  }, [isTypingTarget, frameIdx, nFrames, setFrameIdx, setSpotUndoRequest]);

  const canvasBox = {
    position: "relative" as const,
    border: `1px solid ${themeColors.border}`,
    overflow: "hidden",
    width: "100%",
    maxWidth: canvasSize,
    aspectRatio: "1 / 1",
    bgcolor: "#000",
    touchAction: "none" as const,
    boxSizing: "border-box" as const,
  };
  const sideMenuWidth = 76;
  const patternPanelWidth = canvasSize + sideMenuWidth + SPACING.XS;
  // fluid panel: full width on small hosts, capped at the canvas size
  const panelWidth = { width: "100%", maxWidth: canvasSize, boxSizing: "border-box" as const };

  return (
    <Box
      ref={rootRef}
      sx={{ p: { xs: `${SPACING.SM}px`, sm: `${SPACING.LG}px` }, bgcolor: themeColors.bg, color: themeColors.text, outline: "none", overflow: "visible", maxWidth: "100%", boxSizing: "border-box" }}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={handleRootMouseDownCapture}
    >
      {/* Header */}
      {(showTitle || showControls) && (
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.SM}px` }}>
          {showTitle && (
            <Stack direction="row" alignItems="center" spacing={`${SPACING.XS}px`}>
              <Typography sx={{ fontSize: 13, fontWeight: 600 }}>{title || "Diffraction"}</Typography>
              <InfoTooltip theme={themeInfo.theme} text={
                <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  <MetadataSection rows={[
                    ["Detector", `${detRows} x ${detCols}`],
                    ["Frames", nFrames > 1 ? `${nFrames}` : "single frame"],
                    ["Calibration", kCalibrated && kPixelSize > 0 ? `${formatNumber(kPixelSize)} 1/Å/px` : "pixel units"],
                    ["Center", `${formatNumber(centerRow)}, ${formatNumber(centerCol)}`],
                    ["Annotations", `${spots.length} spots, ${rings.length} rings`],
                  ]} />
                  <KeyboardShortcuts items={[
                    ["Click", "Add spot (or set center in Manual mode)"],
                    ["← →", "Previous / next frame"],
                    ["Shift+Arrow", "Move 10 frames"],
                    ["Scroll", "Zoom in/out"],
                    ["Shift+Drag", "Pan"],
                    ["R", "Reset zoom/pan"],
                    ["Z", "Undo last spot"],
                    ["Esc", "Cancel draw mode"],
                    ["Double-click", "Reset view"],
                  ]} />
                </Box>
              } />
            </Stack>
          )}
          {showControls && (
            <Stack direction="row" spacing={`${SPACING.XS}px`}>
              <Button
                size="small"
                sx={{ ...compactButton, color: themeColors.accent }}
                onClick={() => setControlsCollapsed(!controlsCollapsed)}
                aria-label={controlsCollapsed ? "Show controls" : "Hide controls"}
              >
                {controlsCollapsed ? "Controls" : "Hide"}
              </Button>
              {controlsVisible && (
                <>
                  <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={handleCopyDP}>
                    Copy
                  </Button>
                  <Button
                    size="small"
                    sx={{ ...compactButton, color: themeColors.accent }}
                    onClick={(e) => setDpExportAnchor(e.currentTarget)}
                    title={exportStatus || "Export a PNG or a standalone HTML viewer"}
                  >
                    Export
                  </Button>
                  <Menu anchorEl={dpExportAnchor} open={Boolean(dpExportAnchor)} onClose={() => setDpExportAnchor(null)} {...downwardMenuProps}>
                    <MenuItem onClick={handleExportPng} sx={{ fontSize: 12 }}>PNG</MenuItem>
                    {exportEnabled && <MenuItem onClick={handleExportHtml} sx={{ fontSize: 12 }}>HTML</MenuItem>}
                  </Menu>
                  {exportEnabled && exportStatus && (
                    <Typography
                      sx={{
                        ...typography.value,
                        maxWidth: 160,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                        color: exportStatus.startsWith("Export failed") ? "#d32f2f" : themeColors.textMuted,
                      }}
                      title={exportStatus}
                    >
                      {exportStatus}
                    </Typography>
                  )}
                </>
              )}
            </Stack>
          )}
        </Stack>
      )}

      {/* DP panel */}
      <Box sx={{ display: "flex", justifyContent: "flex-start" }}>
        <Box>
          {/* Toolbar */}
          {controlsVisible && (
            <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} useFlexGap sx={{ mb: `${SPACING.XS}px`, minHeight: 28, flexWrap: "wrap", rowGap: `${SPACING.XS}px`, maxWidth: isMobile ? "100%" : patternPanelWidth, boxSizing: "border-box", px: 1, py: 0.5, border: `1px solid ${themeColors.border}`, borderRadius: "4px", bgcolor: themeColors.controlBg }}>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent, fontWeight: 700 }} onClick={() => setAutoRequest(true)} title="Run full analysis">Auto</Button>
              {nFrames > 1 && (
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={() => setMergeRequest(true)} title="Align and merge frames">Merge</Button>
              )}
              <Button size="small" sx={{ ...compactButton, color: phaseName ? themeColors.accent : themeColors.textMuted }} onClick={(e) => setPhaseMenuAnchor(e.currentTarget)} title="Phase library">
                {phaseName ? `Phase ${phaseName}` : "Phase"}
              </Button>
              <Button size="small" sx={{ ...compactButton, color: maskRegions && maskRegions.length ? themeColors.accent : themeColors.textMuted }} onClick={(e) => setMaskMenuAnchor(e.currentTarget)} title="Edit excluded regions">
                Exclude{maskRegions && maskRegions.length ? ` (${maskRegions.length})` : ""}
              </Button>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>Center</Typography>
              <Select size="small" value={centerMode} onChange={(e) => setCenterMode(String(e.target.value))} sx={{ ...themedSelect, minWidth: 80 }} MenuProps={topToolbarMenuProps}>
                <MenuItem value="auto" sx={{ fontSize: 10 }}>Auto</MenuItem>
                <MenuItem value="manual" sx={{ fontSize: 10 }}>Manual</MenuItem>
              </Select>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={(e) => setRefineMenuAnchor(e.currentTarget)} title="Refine center">Refine</Button>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>Color</Typography>
              <Select size="small" value={dpColormap} onChange={(e) => setDpColormap(e.target.value)} sx={themedSelect} MenuProps={topToolbarMenuProps}>
                {COLORMAP_NAMES.map(n => <MenuItem key={n} value={n} sx={{ fontSize: 10 }}>{n}</MenuItem>)}
              </Select>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale</Typography>
              <Select size="small" value={dpScaleMode} onChange={(e) => setDpScaleMode(e.target.value)} sx={{ ...themedSelect, minWidth: 60 }} MenuProps={topToolbarMenuProps}>
                <MenuItem value="linear" sx={{ fontSize: 10 }}>Linear</MenuItem>
                <MenuItem value="log" sx={{ fontSize: 10 }}>Log</MenuItem>
                <MenuItem value="sqrt" sx={{ fontSize: 10 }}>Sqrt</MenuItem>
              </Select>
              {centerMode === "manual" && (
                <Typography sx={{ ...typography.value, color: themeColors.accent }}>click to set</Typography>
              )}
            </Stack>
          )}

          <Menu anchorEl={phaseMenuAnchor} open={Boolean(phaseMenuAnchor)} onClose={() => setPhaseMenuAnchor(null)} {...downwardMenuProps}>
            <Box sx={{ px: 1.5, py: 0.5, width: "min(300px, calc(100vw - 48px))", boxSizing: "border-box", bgcolor: themeColors.controlBg }}>
              <Typography sx={{ ...typography.label, mb: 0.5 }}>Phase library</Typography>
              <Box sx={{ maxHeight: 180, overflow: "auto", border: `1px solid ${themeColors.border}`, mb: 1 }}>
                <MenuItem selected={!phaseName} onClick={() => setPhaseName("")} sx={{ fontSize: 11, minHeight: 24 }}>None</MenuItem>
                {(phaseLibrary || []).concat(customPhases || []).map((p) => (
                  <MenuItem key={p.name} selected={p.name === phaseName} onClick={() => setPhaseName(p.name)} sx={{ fontSize: 11, minHeight: 24, display: "flex", justifyContent: "space-between" }}>
                    <span>{p.name}</span>
                    <span style={{ color: themeColors.textMuted }}>{[`a=${p.a}`, p.b != null ? `b=${p.b}` : "", p.c != null ? `c=${p.c}` : ""].filter(Boolean).join(" ")} · {p.absences}</span>
                  </MenuItem>
                ))}
              </Box>
              <Stack direction="row" spacing={`${SPACING.XS}px`} alignItems="center" sx={{ mb: 0.5 }}>
                <input value={customName} onChange={(e) => setCustomName(e.target.value)} placeholder="name" style={numInput(70)} />
                <input type="number" value={customA} onChange={(e) => setCustomA(e.target.value)} placeholder="a (Å)" style={numInput(56)} />
                <Select size="small" value={customAbsences} onChange={(e) => setCustomAbsences(String(e.target.value))} sx={{ ...themedSelect, minWidth: 78 }} MenuProps={themedMenuProps}>
                  {["none", "fcc", "bcc", "diamond", "hcp", "wurtzite", "rhombohedral", "rhombohedral-c"].map((r) => <MenuItem key={r} value={r} sx={{ fontSize: 10 }}>{r}</MenuItem>)}
                </Select>
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }}
                  disabled={!customName.trim() || !(parseFloat(customA) > 0)}
                  onClick={() => {
                    const entry: PhaseEntry = { name: customName.trim(), a: parseFloat(customA), absences: customAbsences };
                    if (parseFloat(customB) > 0) entry.b = parseFloat(customB);
                    if (parseFloat(customC) > 0) entry.c = parseFloat(customC);
                    if (parseFloat(customAlpha) > 0) entry.alpha = parseFloat(customAlpha);
                    if (parseFloat(customBeta) > 0) entry.beta = parseFloat(customBeta);
                    if (parseFloat(customGamma) > 0) entry.gamma = parseFloat(customGamma);
                    setCustomPhases([...(customPhases || []), entry]);
                    setPhaseName(customName.trim());
                    setCustomName(""); setCustomA(""); setCustomB(""); setCustomC("");
                    setCustomAlpha(""); setCustomBeta(""); setCustomGamma("");
                  }}>Add</Button>
              </Stack>
              <Stack direction="row" spacing={`${SPACING.XS}px`} alignItems="center" sx={{ mb: 1 }} title="Optional non-cubic lattice; blank = b,c=a and angles 90°">
                <input type="number" value={customB} onChange={(e) => setCustomB(e.target.value)} placeholder="b=a" style={numInput(46)} />
                <input type="number" value={customC} onChange={(e) => setCustomC(e.target.value)} placeholder="c=a" style={numInput(46)} />
                <input type="number" value={customAlpha} onChange={(e) => setCustomAlpha(e.target.value)} placeholder="α 90" style={numInput(46)} />
                <input type="number" value={customBeta} onChange={(e) => setCustomBeta(e.target.value)} placeholder="β 90" style={numInput(46)} />
                <input type="number" value={customGamma} onChange={(e) => setCustomGamma(e.target.value)} placeholder="γ 90" style={numInput(46)} />
              </Stack>
              <Stack direction="row" spacing={`${SPACING.XS}px`} alignItems="center" sx={{ mb: 0.5 }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }} title="Identify ranks only custom phases, not the library">Identify candidates only</Typography>
                <Switch size="small" checked={identifyCustomOnly} onChange={(_, v) => setIdentifyCustomOnly(v)} sx={switchStyles.small} />
              </Stack>
              <Stack direction="row" spacing={`${SPACING.XS}px`} alignItems="center" useFlexGap sx={{ flexWrap: "wrap" }}>
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={!phaseName || !rings || rings.length < 2} onClick={() => setCalibratePhaseRequest(true)} title="Calibrate from rings">Calibrate</Button>
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={!phaseName || !rings || rings.length === 0} onClick={() => setIndexRingsRequest(true)}>Index Rings</Button>
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={!phaseName || !spots || spots.length === 0} onClick={() => setIndexSpotsRequest(true)}>Index Spots</Button>
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={(!rings || rings.length === 0) && (!spots || spots.length === 0)} onClick={() => setIdentifyRequest(true)} title="Identify phase">Identify</Button>
              </Stack>
            </Box>
          </Menu>

          <Menu anchorEl={maskMenuAnchor} open={Boolean(maskMenuAnchor)} onClose={() => setMaskMenuAnchor(null)} {...downwardMenuProps}>
            <Box sx={{ px: 1.5, py: 0.5, width: "min(290px, calc(100vw - 48px))", boxSizing: "border-box", bgcolor: themeColors.controlBg }}>
              <Typography sx={{ ...typography.label, mb: 0.5 }}>Excluded regions</Typography>
              {(maskRegions || []).map((region, i) => (
                <Stack key={i} direction="row" justifyContent="space-between" alignItems="center">
                  <Typography sx={typography.value}>
                    {region.kind === "wedge"
                      ? `wedge ${region.start_deg}°–${region.end_deg}°`
                      : `disk (${Number(region.row).toFixed(0)}, ${Number(region.col).toFixed(0)}) r=${region.radius}`}
                  </Typography>
                  <span onClick={() => setMaskRegions(maskRegions.filter((_, j) => j !== i))} title="Remove" style={{ cursor: "pointer", color: themeColors.textMuted, fontWeight: "bold", padding: "0 4px" }}>×</span>
                </Stack>
              ))}
              <Stack direction="row" spacing={`${SPACING.XS}px`} alignItems="center" sx={{ mt: 1 }}>
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} title="Drag on the pattern: center out to the radius" onClick={() => { setDrawMode("disk"); setMaskMenuAnchor(null); }}>Draw Disk</Button>
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} title="Drag on the pattern: start angle to end angle" onClick={() => { setDrawMode("wedge"); setMaskMenuAnchor(null); }}>Draw Wedge</Button>
              </Stack>
              <Stack direction="row" spacing={`${SPACING.XS}px`} alignItems="center" sx={{ mt: 0.5 }}>
                <input type="number" value={wedgeStart} onChange={(e) => setWedgeStart(e.target.value)} placeholder="start°" style={numInput(52)} />
                <input type="number" value={wedgeEnd} onChange={(e) => setWedgeEnd(e.target.value)} placeholder="end°" style={numInput(52)} />
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }}
                  disabled={wedgeStart === "" || wedgeEnd === ""}
                  onClick={() => { setMaskRegions([...(maskRegions || []), { kind: "wedge", start_deg: parseFloat(wedgeStart), end_deg: parseFloat(wedgeEnd) }]); setWedgeStart(""); setWedgeEnd(""); }}>Add Wedge</Button>
              </Stack>
              <Stack direction="row" spacing={`${SPACING.XS}px`} alignItems="center" sx={{ mt: 0.5 }}>
                <input type="number" value={diskRadius} onChange={(e) => setDiskRadius(e.target.value)} placeholder="radius px" style={numInput(66)} />
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }}
                  disabled={!(parseFloat(diskRadius) > 0) || !spots || spots.length === 0}
                  title="Mask disk at last spot"
                  onClick={() => { const s = spots[spots.length - 1]; setMaskRegions([...(maskRegions || []), { kind: "disk", row: s.row, col: s.col, radius: parseFloat(diskRadius) }]); setDiskRadius(""); }}>Add Disk at Last Spot</Button>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Show</Typography>
                <Switch size="small" checked={showMask} onChange={(_, v) => setShowMask(v)} sx={switchStyles.small} />
              </Stack>
            </Box>
          </Menu>

          <Menu anchorEl={refineMenuAnchor} open={Boolean(refineMenuAnchor)} onClose={() => setRefineMenuAnchor(null)} {...downwardMenuProps}>
            <Box sx={{ px: 1.5, py: 0.5, width: "min(210px, calc(100vw - 48px))", boxSizing: "border-box", bgcolor: themeColors.controlBg }}>
              <Typography sx={{ ...typography.label, mb: 0.5 }}>Refine center</Typography>
              <Stack direction="row" spacing={`${SPACING.XS}px`} alignItems="center">
                <Select size="small" value={refineMethod || "auto"} onChange={(e) => setRefineMethod(String(e.target.value))} sx={{ ...themedSelect, minWidth: 100 }} MenuProps={themedMenuProps}>
                  <MenuItem value="auto" sx={{ fontSize: 10 }}>Auto</MenuItem>
                  <MenuItem value="symmetry" sx={{ fontSize: 10 }}>Symmetry</MenuItem>
                  <MenuItem value="phase_corr" sx={{ fontSize: 10 }}>Phase corr</MenuItem>
                </Select>
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={() => { setRefineCenterRequest(true); setRefineMenuAnchor(null); }}>Run</Button>
              </Stack>
              {centerMethod && (
                <Typography sx={{ ...typography.value, mt: 0.5, color: themeColors.textMuted }}>
                  {centerMethod}
                </Typography>
              )}
            </Box>
          </Menu>
          <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mb: `${SPACING.XS}px` }}>
            {nFrames > 1 ? `Frame ${localFrame + 1} / ${nFrames}` : "Diffraction"}
            {drawMode && <span style={{ marginLeft: 8, color: themeColors.accent }}>
              drag on the pattern to draw the excluded {drawMode} (Esc cancels)
            </span>}
            {moveSpots && !drawMode && <span style={{ marginLeft: 8, color: themeColors.accent }}>
              drag a spot to move it
            </span>}
            {cursorInfo && <span style={{ marginLeft: 8, color: themeColors.accent }}>
              ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
            </span>}
          </Typography>
          <Stack direction={isMobile ? "column" : "row"} spacing={`${SPACING.XS}px`} alignItems="flex-start" sx={{ width: isMobile ? "100%" : patternPanelWidth, maxWidth: "100%" }}>
          <Box ref={dpContainerRef} sx={{ ...canvasBox, flex: "0 1 auto", minWidth: 0 }}>
            <canvas ref={dpCanvasRef} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas ref={dpUiRef} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} />
            <canvas ref={dpScaleRef} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} />
            <canvas
              style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", cursor: "crosshair", opacity: 0, touchAction: "none" }}
              width={canvasSize} height={canvasSize}
              onPointerDown={handleDpPointerDown}
              onPointerMove={handleDpPointerMove}
              onPointerUp={handleDpPointerUp}
              onPointerCancel={handleDpPointerCancel}
              onPointerLeave={handleDpPointerLeave}
              onWheel={handleDpWheel}
              onDoubleClick={resetDpView}
            />
            {/* Resize handle */}
            <Box onPointerDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 }, touchAction: "none" }} />
          </Box>

          {/* Side menu */}
          <Stack
            direction={isMobile ? "row" : "column"}
            spacing="2px"
            sx={isMobile
              ? { width: "100%", flexWrap: "wrap", rowGap: "2px" }
              : { width: sideMenuWidth, flexShrink: 0 }}
            useFlexGap={isMobile}
          >
            {([
              ["Profile", "Radial profile", showProfile, () => setShowProfile(!showProfile)],
              ["Azim", "Azimuthal profile", showAzimuthal, () => setShowAzimuthal(!showAzimuthal)],
              ["HKL", "hkl labels", showHkl, () => setShowHkl(!showHkl)],
              ["Mask View", "Mask overlay", showMask, () => setShowMask(!showMask)],
              ["Invert", "Invert colormap", dpInvert, () => setDpInvert(!dpInvert)],
              ["Stats", "Statistics", showStats, () => setShowStats(!showStats)],
              ["Quality", "Analysis quality", showQc, toggleQuality],
              ["Controls", "Control bar", showControls, () => setShowControls(!showControls)],
            ] as [string, string, boolean, () => void][]).map(([label, hint, on, toggle]) => (
              <Button key={label} size="small" title={hint} onClick={toggle}
                sx={{ ...compactButton, width: isMobile ? "auto" : sideMenuWidth, justifyContent: "flex-start", color: on ? themeColors.accent : themeColors.textMuted }}>
                {label}
              </Button>
            ))}
          </Stack>
          </Stack>

          {/* Quality */}
          {showQc && (
            <Box sx={{ mt: `${SPACING.XS}px`, ...panelWidth, px: 1, py: 0.5, border: `1px solid ${themeColors.border}`, borderRadius: "4px", bgcolor: themeColors.controlBg }}>
              <Typography sx={{ ...typography.label, mb: 0.5 }}>Analysis quality</Typography>
              {quality?.center && (
                <Typography sx={typography.value}>
                  <span style={{ color: themeColors.textMuted }}>Center: </span>
                  {quality.center.method}
                </Typography>
              )}
              {quality?.calibration && (
                <Typography sx={typography.value}>
                  <span style={{ color: themeColors.textMuted }}>Calibration: </span>
                  {quality.calibration.source === "none" ? (
                    <span style={{ color: themeColors.textMuted }}>uncalibrated</span>
                  ) : (
                    <span style={{ color: quality.calibration.rms_px <= 0.5 ? statusColors.good : quality.calibration.rms_px <= 1.5 ? statusColors.warn : statusColors.bad }}>
                      {quality.calibration.source} · rms {quality.calibration.rms_px.toFixed(2)} px
                    </span>
                  )}
                </Typography>
              )}
              {quality?.ellipse && (
                <Typography sx={typography.value}>
                  <span style={{ color: themeColors.textMuted }}>Ellipse: </span>
                  {quality.ellipse.ratio.toFixed(3)} @ {quality.ellipse.angle_deg.toFixed(1)}° · {quality.ellipse.corrected ? "corrected" : "not corrected"}
                </Typography>
              )}
              {quality?.rings && quality.rings.length > 0 && (
                <Typography sx={typography.value}>
                  <span style={{ color: themeColors.textMuted }}>Ring fits: </span>
                  {quality.rings.filter((r) => r.fit_quality >= 0.9).length}/{quality.rings.length}
                </Typography>
              )}
              {quality?.n_unexplained_rings != null && (
                <Typography sx={typography.value}>
                  <span style={{ color: themeColors.textMuted }}>Unexplained rings: </span>
                  <span style={{ color: quality.n_unexplained_rings === 0 ? statusColors.good : statusColors.warn }}>
                    {quality.n_unexplained_rings}
                  </span>
                </Typography>
              )}
              {quality?.mask_coverage_pct != null && (
                <Typography sx={typography.value}>
                  <span style={{ color: themeColors.textMuted }}>Mask coverage: </span>
                  {quality.mask_coverage_pct.toFixed(1)}%
                </Typography>
              )}
              {quality?.ring_snr && (
                <Typography sx={typography.value}>
                  <span style={{ color: themeColors.textMuted }}>Ring SNR: </span>
                  {quality.ring_snr.snr.toFixed(1)} · cov {quality.ring_snr.coverage.toFixed(2)}
                </Typography>
              )}
            </Box>
          )}

          {/* Radial profile */}
          {showProfile && (
            <Box sx={{ mt: `${SPACING.XS}px`, ...panelWidth }}>
              <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} sx={{ px: 1, mb: `${SPACING.XS}px` }}>
                <Typography sx={typography.label}>Radial profile</Typography>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Log</Typography>
                <Switch size="small" checked={profileLog} onChange={(_, v) => setProfileLog(v)} sx={switchStyles.small} />
                <Typography sx={{ ...typography.label, fontSize: 10 }}>−bg</Typography>
                <Switch size="small" checked={profileSubtract} onChange={(_, v) => setProfileSubtract(v)} sx={switchStyles.small} />
                <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>click to add ring</Typography>
                {paneStaleNote && (
                  <Typography sx={{ ...typography.value, color: statusColors.warn }}>{paneStaleNote}</Typography>
                )}
              </Stack>
              <canvas
                ref={profileCanvasRef}
                style={{ display: "block", width: "100%", maxWidth: canvasSize, height: PROFILE_H, cursor: "crosshair", border: `1px solid ${themeColors.border}`, boxSizing: "border-box", opacity: paneStaleNote ? 0.4 : 1 }}
                onPointerDown={handleProfileClick}
              />
            </Box>
          )}

          {/* Azimuthal */}
          {showAzimuthal && (
            <Box sx={{ mt: `${SPACING.XS}px`, ...panelWidth }}>
              <Typography sx={{ ...typography.label, px: 1, mb: `${SPACING.XS}px` }}>
                Azimuthal profile (outermost ring)
                {paneStaleNote && (
                  <Box component="span" sx={{ ...typography.value, ml: 1, color: statusColors.warn }}>{paneStaleNote}</Box>
                )}
              </Typography>
              <canvas ref={azimuthalCanvasRef} style={{ display: "block", width: "100%", maxWidth: canvasSize, height: PROFILE_H, border: `1px solid ${themeColors.border}`, boxSizing: "border-box", opacity: paneStaleNote ? 0.4 : 1 }} />
            </Box>
          )}

          {/* Candidates */}
          {identifyResults && identifyResults.length > 0 && (
            <Box sx={{ mt: `${SPACING.XS}px`, ...panelWidth }}>
              <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} sx={{ px: 1, mb: `${SPACING.XS}px` }}>
                <Typography sx={typography.label}>Candidate phases</Typography>
                <input value={identifyElements} onChange={(e) => setIdentifyElements(e.target.value)} placeholder="elements e.g. Fe,O" style={numInput(110)} />
                <Typography sx={{ ...typography.label, fontSize: 10 }} title="Rank only custom phases, not the library">candidates only</Typography>
                <Switch size="small" checked={identifyCustomOnly} onChange={(_, v) => setIdentifyCustomOnly(v)} sx={switchStyles.small} />
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={() => setIdentifyRequest(true)} title="Identify phase">Identify</Button>
                <Button size="small" sx={{ ...compactButton, color: themeColors.textMuted }} onClick={() => setIdentifyCollapsed(true)} title="Hide results">Clear</Button>
              </Stack>
              {!identifyCollapsed && (
                <Box sx={{ maxHeight: 240, overflow: "auto", border: `1px solid ${themeColors.border}` }}>
                  <table style={{ width: "100%", fontSize: 10, fontFamily: "monospace", borderCollapse: "collapse", color: themeColors.text }}>
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${themeColors.border}`, textAlign: "left" }}>
                        <th style={{ padding: "2px 6px" }}>name</th>
                        <th style={{ padding: "2px 6px" }} title="matched lines / observed">matched</th>
                        <th style={{ padding: "2px 6px" }} title="mean |Δd| vs reference">Δd (%)</th>
                        <th style={{ padding: "2px 6px" }} title="strong reference lines not observed">missing</th>
                      </tr>
                    </thead>
                    <tbody>
                      {identifyResults.map((cand) => (
                        <React.Fragment key={cand.phase_id}>
                          <tr
                            style={{ borderBottom: `1px solid ${themeColors.border}22`, cursor: "pointer", background: expandedPhaseId === cand.phase_id ? `${themeColors.accent}18` : undefined }}
                            onClick={() => setExpandedPhaseId(expandedPhaseId === cand.phase_id ? null : cand.phase_id)}
                          >
                            <td style={{ padding: "2px 6px", color: themeColors.accent }}>{cand.name}</td>
                            <td style={{ padding: "2px 6px" }}>{cand.matched}/{cand.n_obs}</td>
                            <td style={{ padding: "2px 6px" }}>{cand.mean_err != null ? (cand.mean_err * 100).toFixed(2) : "—"}</td>
                            <td style={{ padding: "2px 6px" }}>{cand.n_missing_strong ?? "-"}</td>
                          </tr>
                          {expandedPhaseId === cand.phase_id && (
                            <tr style={{ borderBottom: `1px solid ${themeColors.border}22` }}>
                              <td colSpan={4} style={{ padding: "2px 6px 4px 14px" }}>
                                <table style={{ width: "100%", fontSize: 10, fontFamily: "monospace", borderCollapse: "collapse", color: themeColors.text }}>
                                  <thead>
                                    <tr style={{ borderBottom: `1px solid ${themeColors.border}`, textAlign: "left", color: themeColors.textMuted }}>
                                      <th style={{ padding: "1px 6px" }}>d_obs (Å)</th>
                                      <th style={{ padding: "1px 6px" }}>d_ref (Å)</th>
                                      <th style={{ padding: "1px 6px" }}>hkl</th>
                                      <th style={{ padding: "1px 6px" }}>Δd (%)</th>
                                      <th style={{ padding: "1px 6px" }}>I_rel</th>
                                      <th style={{ padding: "1px 6px" }}></th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {(cand.lines || []).map((line, li) => {
                                      const lineColor = line.ref_d == null ? statusColors.bad : line.obs_d == null ? themeColors.textMuted : undefined;
                                      return (
                                        <tr key={li} style={{ color: lineColor }}>
                                          <td style={{ padding: "1px 6px" }}>{line.obs_d != null ? line.obs_d.toFixed(3) : "—"}</td>
                                          <td style={{ padding: "1px 6px" }}>{line.ref_d != null ? line.ref_d.toFixed(3) : "—"}</td>
                                          <td style={{ padding: "1px 6px" }}>{line.hkl || "—"}</td>
                                          <td style={{ padding: "1px 6px" }}>{line.err != null ? (line.err * 100).toFixed(2) : "—"}</td>
                                          <td style={{ padding: "1px 6px" }}>{line.i_rel != null ? line.i_rel.toFixed(0) : "—"}</td>
                                          <td style={{ padding: "1px 6px" }}>{line.ref_d == null ? "unexplained" : line.obs_d == null ? "missing" : ""}</td>
                                        </tr>
                                      );
                                    })}
                                  </tbody>
                                </table>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      ))}
                    </tbody>
                  </table>
                </Box>
              )}
            </Box>
          )}

          {/* Frame slider */}
          {nFrames > 1 && (
            <Box sx={{ ...controlRow, ...panelWidth }}>
              <Typography sx={typography.label}>Frame</Typography>
              <Slider
                value={localFrame}
                min={0} max={nFrames - 1} step={1} size="small"
                valueLabelDisplay="auto" valueLabelFormat={(v) => `${v + 1}`}
                onChange={(_, v) => setLocalFrame(v as number)}
                onChangeCommitted={(_, v) => setFrameIdx(v as number)}
                aria-label={`Frame ${localFrame + 1} of ${nFrames}`}
                sx={{ ...sliderStyles.small, flex: 1, minWidth: 40, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }}
              />
              <Typography sx={typography.value}>{localFrame + 1}/{nFrames}</Typography>
            </Box>
          )}

          {/* Stats */}
          {showStats && displayStats && displayStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.25, display: "flex", gap: 2 }}>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(displayStats[0])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(displayStats[1])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(displayStats[2])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(displayStats[3])}</Box>
              </Typography>
            </Box>
          )}

          {/* Indexing */}
          {showStats && (zoneAxis || phaseMatch) && (
            <Box sx={{ px: 1, py: 0.25, display: "flex", gap: 2, flexWrap: "wrap" }}>
              {zoneAxis && (
                <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                  Zone <Box component="span" sx={{ color: themeColors.accent }}>{zoneAxis}</Box>
                </Typography>
              )}
              {phaseMatch && (
                <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                  {phaseMatch}
                </Typography>
              )}
            </Box>
          )}

          {/* Status */}
          {analysisStatus && (
            <Box sx={{ px: 1, py: 0.25 }}>
              <Typography
                sx={{ ...typography.value, maxWidth: canvasSize, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: analysisStatus.includes("failed") ? "#d32f2f" : themeColors.textMuted }}
                title={analysisStatus}
              >
                {analysisStatus}
              </Typography>
            </Box>
          )}
        </Box>
      </Box>

      {/* Spots */}
      <Box sx={{ mt: `${SPACING.MD}px`, maxWidth: canvasSize }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px` }}>
            <Stack direction="row" alignItems="center" spacing={`${SPACING.XS}px`}>
              <Typography sx={{ ...typography.label, color: themeColors.text }}>
                Spots ({spots ? spots.length : 0})
              </Typography>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={() => setDetectRequest(-1)} title="Detect spots">Auto</Button>
              <Button size="small" sx={{ ...compactButton, color: moveSpots ? themeColors.accent : themeColors.textMuted }} onClick={() => setMoveSpots(!moveSpots)} title="Drag spots on the pattern to move them">Move</Button>
            </Stack>
            <Stack direction="row" spacing={`${SPACING.XS}px`} sx={{ p: 0.25, border: `1px solid ${themeColors.border}`, borderRadius: "4px", bgcolor: themeColors.controlBg }}>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={!spots || spots.length === 0}
                onClick={() => exportMeasurements("csv", "spots")}
              >
                CSV
              </Button>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={!spots || spots.length === 0}
                onClick={() => exportMeasurements("json", "spots")}
              >
                JSON
              </Button>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={!spots || spots.length === 0}
                onClick={() => setSpotUndoRequest(true)}
              >
                Undo
              </Button>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={!spots || spots.length === 0}
                onClick={() => setSpotClearRequest(true)}
              >
                Clear
              </Button>
            </Stack>
          </Stack>
          {spots && spots.length > 0 && (
            <Box sx={{ maxHeight: 220, overflow: "auto", border: `1px solid ${themeColors.border}` }}>
              <table style={{ width: "100%", fontSize: 10, fontFamily: "monospace", borderCollapse: "collapse", color: themeColors.text }}>
                <thead>
                  <tr style={{ borderBottom: `1px solid ${themeColors.border}`, textAlign: "left" }}>
                    <th style={{ padding: "2px 4px" }}>#</th>
                    <th style={{ padding: "2px 6px" }}>d (Å)</th>
                    <th style={{ padding: "2px 6px" }} title="indexed reflection">hkl</th>
                    {spots.some((s) => s.d_ref != null) && (
                      <th style={{ padding: "2px 6px" }} title="measured vs reference d">Δd (%)</th>
                    )}
                    <th style={{ padding: "2px 6px" }} title="|g| = 1/d, shown in 1/Å / 1/nm">|g| (1/Å / 1/nm)</th>
                    <th style={{ padding: "2px 6px" }} title="angle vs reference spot">∠ (°)</th>
                    <th style={{ padding: "2px 6px" }} title="Gaussian fit R²">fit</th>
                    <th style={{ padding: "2px 6px" }}>I</th>
                    <th style={{ padding: "2px 4px" }}></th>
                  </tr>
                </thead>
                <tbody>
                  {spots.map((spot: SpotDict, i: number) => {
                    const color = spotColorAt(i);
                    const dStr = spot.d_spacing != null
                      ? (spot.d_spacing_err ? `${spot.d_spacing.toFixed(3)}±${spot.d_spacing_err.toFixed(3)}` : spot.d_spacing.toFixed(3))
                      : "—";
                    const gStr = spot.g_magnitude != null
                      ? `${spot.g_magnitude.toFixed(4)} / ${(spot.g_magnitude * 10).toFixed(3)}`
                      : `${spot.r_pixels.toFixed(1)} px`;
                    const aStr = spot.angle_deg != null
                      ? (spot.angle_deg_err ? `${spot.angle_deg.toFixed(1)}±${spot.angle_deg_err.toFixed(1)}` : spot.angle_deg.toFixed(1))
                      : "—";
                    return (
                      <tr key={spot.id} style={{ borderBottom: `1px solid ${themeColors.border}22` }}>
                        <td style={{ padding: "2px 4px", color, fontWeight: "bold" }}>{spot.id}</td>
                        <td style={{ padding: "2px 6px" }}>{dStr}</td>
                        <td style={{ padding: "2px 6px" }}>{spot.hkl || "—"}</td>
                        {spots.some((s) => s.d_ref != null) && (
                          <td style={{ padding: "2px 6px" }}>{spot.d_error != null ? (spot.d_error * 100).toFixed(2) : "—"}</td>
                        )}
                        <td style={{ padding: "2px 6px" }}>{gStr}</td>
                        <td style={{ padding: "2px 6px" }}>{aStr}</td>
                        <td style={{ padding: "2px 6px" }}>{spot.fit_quality != null ? spot.fit_quality.toFixed(2) : "—"}</td>
                        <td style={{ padding: "2px 6px" }}>{formatNumber(spot.intensity)}</td>
                        <td style={{ padding: "1px 4px", textAlign: "center" }}>
                          <span
                            onClick={() => setSpotRemoveRequest(spot.id)}
                            title="Delete this spot"
                            style={{ cursor: "pointer", color: themeColors.textMuted, fontWeight: "bold", padding: "0 3px" }}
                          >×</span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </Box>
          )}
        </Box>

      {/* Rings */}
      <Box sx={{ mt: `${SPACING.MD}px`, maxWidth: canvasSize }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px` }}>
            <Stack direction="row" alignItems="center" spacing={`${SPACING.XS}px`}>
              <Typography sx={{ ...typography.label, color: themeColors.text }}>
                Rings ({rings ? rings.length : 0})
              </Typography>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={() => setDetectRingsRequest(-1)} title="Detect rings">Auto</Button>
            </Stack>
            <Stack direction="row" spacing={`${SPACING.XS}px`} sx={{ p: 0.25, border: `1px solid ${themeColors.border}`, borderRadius: "4px", bgcolor: themeColors.controlBg }}>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={!rings || rings.length === 0} onClick={() => exportMeasurements("csv", "rings")}>CSV</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={!rings || rings.length === 0} onClick={() => exportMeasurements("json", "rings")}>JSON</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={!rings || rings.length === 0} onClick={() => setFitRingsRequest(true)} title="Fit ring profiles">Fit</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={!rings || rings.length === 0} onClick={() => setRingUndoRequest(true)}>Undo</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={!rings || rings.length === 0} onClick={() => setRingClearRequest(true)}>Clear</Button>
            </Stack>
          </Stack>
          {rings && rings.length > 0 && (
          <Box sx={{ maxHeight: 160, overflow: "auto", border: `1px solid ${themeColors.border}` }}>
            <table style={{ width: "100%", fontSize: 10, fontFamily: "monospace", borderCollapse: "collapse", color: themeColors.text }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${themeColors.border}`, textAlign: "left" }}>
                  <th style={{ padding: "2px 6px" }}>#</th>
                  <th style={{ padding: "2px 6px" }}>radius (px)</th>
                  <th style={{ padding: "2px 6px" }}>d (Å)</th>
                  <th style={{ padding: "2px 6px" }} title="indexed reflection">hkl</th>
                  {rings.some((r) => r.d_ref != null) && (
                    <th style={{ padding: "2px 6px" }} title="measured vs reference d">Δd (%)</th>
                  )}
                  <th style={{ padding: "2px 6px" }}>|g| (1/Å)</th>
                  {rings.some((r) => r.fwhm_px != null) && (
                    <th style={{ padding: "2px 6px" }} title="fitted peak width">fwhm (px)</th>
                  )}
                  <th style={{ padding: "2px 6px" }}>I</th>
                  <th style={{ padding: "2px 4px" }}></th>
                </tr>
              </thead>
              <tbody>
                {rings.map((ring: RingDict) => (
                  <tr
                    key={ring.id}
                    style={{ borderBottom: `1px solid ${themeColors.border}22`, cursor: "pointer", background: selectedRingId === ring.id ? `${themeColors.accent}18` : undefined }}
                    onClick={() => setSelectedRingId(selectedRingId === ring.id ? 0 : ring.id)}
                  >
                    <td style={{ padding: "2px 6px", color: themeColors.accent }}>{ring.id}</td>
                    <td style={{ padding: "2px 6px" }}>{ring.radius_px.toFixed(1)}</td>
                    <td style={{ padding: "2px 6px" }}>{ring.d_spacing != null ? ring.d_spacing.toFixed(3) : "—"}</td>
                    <td style={{ padding: "2px 6px" }}>{ring.hkl || "—"}</td>
                    {rings.some((r) => r.d_ref != null) && (
                      <td style={{ padding: "2px 6px" }}>{ring.d_error != null ? (ring.d_error * 100).toFixed(2) : "—"}</td>
                    )}
                    <td style={{ padding: "2px 6px" }}>{ring.g_magnitude != null ? ring.g_magnitude.toFixed(4) : "—"}</td>
                    {rings.some((r) => r.fwhm_px != null) && (
                      <td style={{ padding: "2px 6px" }}>{ring.fwhm_px != null ? ring.fwhm_px.toFixed(2) : "—"}</td>
                    )}
                    <td style={{ padding: "2px 6px" }}>{formatNumber(ring.intensity)}</td>
                    <td style={{ padding: "1px 4px", textAlign: "center" }}>
                      <span
                        onClick={(e) => { e.stopPropagation(); setRingRemoveRequest(ring.id); }}
                        title="Delete this ring"
                        style={{ cursor: "pointer", color: themeColors.textMuted, fontWeight: "bold", padding: "0 3px" }}
                      >×</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Box>
          )}
      </Box>

      {/* Fine controls below the display */}
      {controlsVisible && (
        <Box sx={{ mt: `${SPACING.MD}px`, maxWidth: canvasSize }}>
          <Stack direction="row" spacing={`${SPACING.LG}px`} sx={{ flexWrap: "wrap" }}>
            <Box sx={controlBox} title={`How clicked spots are placed (±${snapRadius} px search window)`}>
              <Typography sx={typography.label}>Spot pick</Typography>
              <Select
                size="small"
                value={spotRefine ? "fit" : snapEnabled ? "snap" : "exact"}
                onChange={(e) => { const v = String(e.target.value); setSpotRefine(v === "fit"); setSnapEnabled(v === "snap"); }}
                sx={{ ...themedSelect, minWidth: 96 }}
                MenuProps={themedMenuProps}
              >
                <MenuItem value="fit" sx={{ fontSize: 10 }}>Fit peak</MenuItem>
                <MenuItem value="snap" sx={{ fontSize: 10 }}>Snap to max</MenuItem>
                <MenuItem value="exact" sx={{ fontSize: 10 }}>Exact click</MenuItem>
              </Select>
            </Box>

            <Box sx={{ ...controlBox, overflow: "visible" }}>
              <Histogram
                data={dpHistData}
                vminPct={dpVminPct}
                vmaxPct={dpVmaxPct}
                dataMin={scaledFrame?.dataMin ?? 0}
                dataMax={scaledFrame?.dataMax ?? 1}
                onRangeChange={(min, max) => { setDpVminPct(min); setDpVmaxPct(max); }}
                onRangePreview={previewContrast}
                onRangeCommit={commitContrast}
                theme={themeInfo.theme}
              />
            </Box>
          </Stack>

          <Box sx={{ ...controlBox, mt: `${SPACING.XS}px` }}>
            <Typography sx={typography.label}>Calibrate d (Å)</Typography>
            <input
              type="number" value={dKnown}
              onChange={(e) => setDKnown(e.target.value)}
              placeholder="2.355"
              style={numInput(64)}
            />
            <Button
              size="small" sx={{ ...compactButton, color: themeColors.accent }}
              disabled={!spots || spots.length === 0 || !(parseFloat(dKnown) > 0)}
              onClick={() => { const d = parseFloat(dKnown); const s = spots[spots.length - 1]; if (d > 0 && s) setCalibrateFromSpotRequest([s.row, s.col, d]); }}
            >From Spot</Button>
            <Button
              size="small" sx={{ ...compactButton, color: themeColors.accent }}
              disabled={!rings || rings.length === 0 || !(parseFloat(dKnown) > 0)}
              onClick={() => { const d = parseFloat(dKnown); const r = rings[rings.length - 1]; if (d > 0 && r) setCalibrateFromRingRequest([r.radius_px, d]); }}
            >From Ring</Button>
            <Button
              size="small" sx={{ ...compactButton, color: themeColors.accent }}
              onClick={zoomToCenter}
              title="Zoom to the diffraction center"
            >Center View</Button>
          </Box>

          <Box sx={{ ...controlBox, mt: `${SPACING.XS}px` }}>
            <Typography sx={typography.label}>Distortion</Typography>
            <Button
              size="small" sx={{ ...compactButton, color: themeColors.accent }}
              disabled={!rings || rings.length === 0}
              onClick={() => setFitEllipseRequest(true)}
              title="Fit ellipse distortion"
            >Fit Ellipse</Button>
            <Typography sx={{ ...typography.label, fontSize: 10 }}>Use Correction</Typography>
            <Switch
              size="small" checked={ellipseCorrected}
              onChange={(_, v) => setEllipseCorrected(v)}
              sx={switchStyles.small}
              title="Use fitted ellipse correction"
              disabled={!(ellipseRatio > 1.0)}
            />
            {ellipseRatio > 1.0 && (
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                a/b {ellipseRatio.toFixed(3)} @ {ellipseAngle.toFixed(1)}°
              </Typography>
            )}
          </Box>

          <Box sx={{ ...controlRow, mt: `${SPACING.XS}px` }}>
            <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
              Center: ({centerRow.toFixed(1)}, {centerCol.toFixed(1)})  BF r={bfRadius.toFixed(1)}
              {kCalibrated && <span style={{ marginLeft: 8 }}>k={kPixelSize.toFixed(4)} 1/Å/px</span>}
            </Typography>
          </Box>
          {kCalibrated && (
            <Box sx={controlRow}>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Calibration: {calibrationSourceLabel(calibrationSource)}
                {calibrationRefD > 0 && ` (d=${calibrationRefD.toFixed(3)} Å @ r=${calibrationRefRadius.toFixed(1)} px)`}
              </Typography>
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(ShowDiffraction);
