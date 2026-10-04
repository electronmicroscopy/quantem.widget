/**
 * PlanPtycho: multislice ptychography settings checked against a known crystal.
 *
 * Top: the projected crystal (smeared by the tilt across the thickness) with the scan field, one probe position (drag to
 * move) and its beam at the view depth. Side: the specimen as a stationary slab with vacuum above and below, its columns
 * leaning with the tilt, the beam cone at the probe position; drag the dashed focus line to move the focus, drag
 * elsewhere to set the view depth. Probe: the probe on the reconstruction's model window at the view depth, optionally
 * over the crystal at that depth. Detector: bright-field disk, Bragg disks, the zone axis and HOLZ ring moved by the tilt,
 * the detector edge. Every panel zooms with the wheel; double-click resets. Sliders update everything locally and commit
 * to Python on release.
 */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Slider from "@mui/material/Slider";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Tooltip from "@mui/material/Tooltip";
import Button from "@mui/material/Button";
import Switch from "@mui/material/Switch";
import { useTheme, type ThemeColors } from "../theme";
import { drawScaleBarHiDPI } from "../figure";
import { extractFloat32 } from "../format";
import { COLORMAPS, applyColormap } from "../colormaps";
import {
  beamDiameterA, checkRows, detectorSamplingMrad, planGeometry, probeIntensity, recommendedSettings,
  simulationGeometry, type DetectorPresets, type Grade, type PlanSettings,
} from "./geometry";

// ============================================================================
// Style tokens (inlined - matches Show3DSlices single-file convention)
// ============================================================================
const PANEL = 224;
const SPACING = { XS: 4, SM: 8, MD: 12 } as const;
// overlays sit on black image canvases in both notebook themes, so they are fixed colours
const OVERLAY = { scan: "#4dd0e1", beam: "#ffd54f", window: "#90caf9", focus: "rgba(255,255,255,0.8)", edge: "#4fc3f7", zone: "#ff8a80", slab: "rgba(255,255,255,0.35)" } as const;
const GRADE_COLOR: Record<Grade | "info", string> = { pass: "#43a047", caution: "#fb8c00", fail: "#e53935", info: "#888" };
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
};
const sliderSx = {
  py: 0, width: 116, flexShrink: 0,
  "& .MuiSlider-thumb": { width: 10, height: 10 },
  "& .MuiSlider-rail": { height: 2 },
  "& .MuiSlider-track": { height: 2 },
};
const switchSx = { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } };
const monoOverlay = {
  position: "absolute" as const, top: 4, right: 4, px: 0.75, py: 0.25, bgcolor: "rgba(0,0,0,0.62)", color: "#eee",
  fontFamily: "monospace", fontSize: 10, lineHeight: 1.35, pointerEvents: "none" as const, whiteSpace: "pre" as const, textAlign: "right" as const,
};
const VOLTAGES = [60, 80, 120, 200, 300];
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 40;

type View = { zoom: number; cx: number; cy: number };
const HOME: View = { zoom: 1, cx: PANEL / 2, cy: PANEL / 2 };
type MicroscopePreset = { voltage_kV: number; semiangle_mrad: number; detector: string; camera_length_mm: number };

// ============================================================================
// Canvas helpers
// ============================================================================
function devicePixelRatio(): number {
  return typeof window === "undefined" ? 1 : Math.min(2, window.devicePixelRatio || 1);
}

function sizeCanvas(canvas: HTMLCanvasElement | null, dpr: number): CanvasRenderingContext2D | null {
  if (!canvas) return null;
  const size = Math.round(PANEL * dpr);
  if (canvas.width !== size) { canvas.width = size; canvas.height = size; }
  return canvas.getContext("2d");
}

/** Scene transform: panel coordinates (css px, unzoomed) -> device pixels for the current zoom and centre. */
function applyView(ctx: CanvasRenderingContext2D, dpr: number, view: View) {
  ctx.setTransform(dpr * view.zoom, 0, 0, dpr * view.zoom, dpr * (PANEL / 2 - view.cx * view.zoom), dpr * (PANEL / 2 - view.cy * view.zoom));
}

/** Pointer position in panel coordinates (css px, unzoomed). */
function toScene(e: { clientX: number; clientY: number }, canvas: HTMLCanvasElement, view: View): [number, number] {
  const box = canvas.getBoundingClientRect();
  const x = ((e.clientX - box.left) / box.width) * PANEL, y = ((e.clientY - box.top) / box.height) * PANEL;
  return [(x - PANEL / 2) / view.zoom + view.cx, (y - PANEL / 2) / view.zoom + view.cy];
}

/** Wheel zoom about the cursor; the listener is non-passive so the page does not scroll. */
function useWheelZoom(ref: React.RefObject<HTMLCanvasElement | null>, setView: React.Dispatch<React.SetStateAction<View>>) {
  React.useEffect(() => {
    const canvas = ref.current; if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const box = canvas.getBoundingClientRect();
      const sx = ((e.clientX - box.left) / box.width) * PANEL, sy = ((e.clientY - box.top) / box.height) * PANEL;
      setView((v) => {
        const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.zoom * Math.exp(-e.deltaY * 0.0015)));
        const px = (sx - PANEL / 2) / v.zoom + v.cx, py = (sy - PANEL / 2) / v.zoom + v.cy;         // scene point under the cursor stays put
        return { zoom, cx: px - (sx - PANEL / 2) / zoom, cy: py - (sy - PANEL / 2) / zoom };
      });
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [ref, setView]);
}

function drawBar(canvas: HTMLCanvasElement | null, dpr: number, view: View, unitsPerPx: number, unit: string) {
  if (!canvas) return;
  const size = Math.round(PANEL * dpr);
  if (canvas.width !== size) { canvas.width = size; canvas.height = size; }
  drawScaleBarHiDPI(canvas, dpr, view.zoom, unitsPerPx, unit, PANEL, { position: "bottom-left", showZoomIndicator: true });
}

/** Float image -> colormapped canvas (reused), auto range from the 1st / 99.5th percentile of a sample. */
function paintInto(target: HTMLCanvasElement, values: Float32Array, width: number, height: number, cmap: string, gamma = 1) {
  const shown = gamma === 1 ? values : values.map((v) => Math.pow(Math.max(v, 0), gamma));
  const stride = Math.max(1, Math.floor(shown.length / 4096));
  const sample = new Float32Array(Math.ceil(shown.length / stride));
  for (let i = 0, j = 0; i < shown.length; i += stride, j++) sample[j] = shown[i];
  sample.sort();
  const lo = sample[Math.floor(sample.length * 0.01)] ?? 0, hi = sample[Math.floor(sample.length * 0.995)] ?? 1;
  if (target.width !== width || target.height !== height) { target.width = width; target.height = height; }
  const ctx = target.getContext("2d")!;
  const image = ctx.createImageData(width, height);
  applyColormap(shown, image.data, COLORMAPS[cmap], lo, hi > lo ? hi : lo + 1);
  ctx.putImageData(image, 0, 0);
}

/** Projected crystal at physical (row, col) in Å, periodic in the cell. */
function sampleCell(cell: Float32Array, rows: number, cols: number, sizeRow: number, sizeCol: number, rowA: number, colA: number): number {
  const r = ((((rowA / sizeRow) % 1) + 1) % 1) * rows;
  const c = ((((colA / sizeCol) % 1) + 1) % 1) * cols;
  const r0 = Math.floor(r) % rows, c0 = Math.floor(c) % cols, fr = r - Math.floor(r), fc = c - Math.floor(c);
  const r1 = (r0 + 1) % rows, c1 = (c0 + 1) % cols;
  return (cell[r0 * cols + c0] * (1 - fr) + cell[r1 * cols + c0] * fr) * (1 - fc)
       + (cell[r0 * cols + c1] * (1 - fr) + cell[r1 * cols + c1] * fr) * fc;
}

/** Crystal pattern whose tile is one cell; origin (0 Å) at panel (originX, originY), `pxPerA` css px per Å. */
function crystalPattern(ctx: CanvasRenderingContext2D, tile: HTMLCanvasElement, cellSize: number[], pxPerA: number, originX: number, originY: number) {
  const pattern = ctx.createPattern(tile, "repeat");
  if (!pattern) return null;
  pattern.setTransform(new DOMMatrix([(cellSize[1] / tile.width) * pxPerA, 0, 0, (cellSize[0] / tile.height) * pxPerA, originX, originY]));
  return pattern;
}

// ============================================================================
// Controls
// ============================================================================
/** A value you can type: shows the current number, commits on Enter or when focus leaves, reverts on Escape. */
function NumberField({ value, digits, unit, onCommit, colors, integer = false }: {
  value: number; digits: number; unit: string; onCommit: (v: number) => void; colors: ThemeColors; integer?: boolean;
}) {
  const [text, setText] = React.useState(value.toFixed(digits));
  const [editing, setEditing] = React.useState(false);
  React.useEffect(() => { if (!editing) setText(value.toFixed(digits)); }, [value, digits, editing]);
  const finish = () => {
    setEditing(false);
    const parsed = integer ? Math.round(Number(text)) : Number(text);
    if (text.trim() !== "" && Number.isFinite(parsed)) onCommit(parsed); else setText(value.toFixed(digits));
  };
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
      <Box component="input" value={text} inputMode="decimal" aria-label={unit}
        onFocus={() => setEditing(true)} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setText(e.target.value)} onBlur={finish}
        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
          if (e.key === "Escape") { setText(value.toFixed(digits)); setEditing(false); (e.target as HTMLInputElement).blur(); }
        }}
        sx={{ ...typography.value, width: 52, px: 0.5, py: 0.1, color: colors.text, bgcolor: "transparent", border: `1px solid transparent`,
              "&:hover": { borderColor: colors.border }, "&:focus": { outline: "none", borderColor: colors.accent, bgcolor: colors.controlBg } }} />
      <Typography sx={{ ...typography.value, color: colors.textMuted }}>{unit}</Typography>
    </Box>
  );
}

function SliderRow({ label, value, min, max, step, unit, digits, onChange, onCommit, tip, colors }: {
  label: string; value: number; min: number; max: number; step: number; unit: string; digits: number;
  onChange: (v: number) => void; onCommit: (v: number) => void; tip: string; colors: ThemeColors;
}) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, minHeight: 22 }}>
      <Tooltip title={tip} placement="top" arrow><Typography sx={{ ...typography.labelSmall, width: 64, cursor: "help" }}>{label}</Typography></Tooltip>
      <Slider size="small" sx={sliderSx} value={Math.min(max, Math.max(min, value))} min={min} max={max} step={step}
        onChange={(_, v) => onChange(v as number)} onChangeCommitted={(_, v) => onCommit(v as number)} />
      <NumberField value={value} digits={digits} unit={unit} onCommit={onCommit} colors={colors} />
    </Box>
  );
}

function NumberRow({ label, value, digits, unit, onCommit, tip, colors, integer = false }: {
  label: string; value: number; digits: number; unit: string; onCommit: (v: number) => void; tip: string; colors: ThemeColors; integer?: boolean;
}) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, minHeight: 22 }}>
      <Tooltip title={tip} placement="top" arrow><Typography sx={{ ...typography.labelSmall, width: 64, cursor: "help" }}>{label}</Typography></Tooltip>
      <NumberField value={value} digits={digits} unit={unit} onCommit={onCommit} colors={colors} integer={integer} />
    </Box>
  );
}

function SelectRow({ label, value, options, format, onChange, tip, colors, width = 88 }: {
  label: string; value: string; options: string[]; format?: (v: string) => string; onChange: (v: string) => void; tip: string; colors: ThemeColors; width?: number;
}) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, minHeight: 22 }}>
      {label && <Tooltip title={tip} placement="top" arrow><Typography sx={{ ...typography.labelSmall, width: 64, cursor: "help" }}>{label}</Typography></Tooltip>}
      <Select size="small" variant="outlined" value={options.includes(value) ? value : ""} displayEmpty onChange={(e) => onChange(String(e.target.value))}
        renderValue={(v) => (v ? (format ? format(String(v)) : String(v)) : "custom")}
        sx={{ fontSize: 10, height: 20, minWidth: width, bgcolor: colors.controlBg, color: colors.text,
              "& .MuiSelect-select": { py: 0, px: 0.75 }, "& .MuiSelect-icon": { color: colors.textMuted },
              "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border }, "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: colors.accent } }}
        MenuProps={{ sx: { zIndex: 9999 }, PaperProps: { sx: { bgcolor: colors.controlBg, color: colors.text, border: `1px solid ${colors.border}` } } }}>
        {options.map((o) => <MenuItem key={o} value={o} sx={{ fontSize: 11, minHeight: 0, py: 0.5 }}>{format ? format(o) : o}</MenuItem>)}
      </Select>
    </Box>
  );
}

function PanelFrame({ title, actions, children, colors }: { title: string; actions?: React.ReactNode; children: React.ReactNode; colors: ThemeColors }) {
  return (
    <Box sx={{ width: PANEL, flexShrink: 0, border: `1px solid ${colors.border}`, bgcolor: colors.bgAlt }}>
      <Box sx={{ height: 22, display: "flex", alignItems: "center", px: 1, borderBottom: `1px solid ${colors.border}` }}>
        <Typography sx={{ ...typography.label, fontWeight: 600, color: colors.textMuted }}>{title}</Typography>
        <Box sx={{ flex: 1 }} />
        {actions}
      </Box>
      <Box sx={{ position: "relative", width: PANEL, height: PANEL, bgcolor: "#000" }}>{children}</Box>
    </Box>
  );
}

// ============================================================================
// Widget
// ============================================================================
function PlanPtycho() {
  const { colors } = useTheme();
  const model = useModel();
  const [title] = useModelState<string>("title");
  const [zoneAxis, setZoneAxis] = useModelState<number[]>("zone_axis");
  const [zoneAxes] = useModelState<number[][]>("zone_axes");
  const [cellBytes] = useModelState<DataView>("cell_bytes");
  const [cellShape] = useModelState<number[]>("cell_shape");
  const [cellSize] = useModelState<number[]>("cell_size_A");
  const [holzRepeat] = useModelState<number>("holz_repeat_A");
  const [bragg] = useModelState<number[][]>("bragg_inv_A");
  const [columnPhase] = useModelState<number>("column_phase_rad_per_A");
  const [presets] = useModelState<DetectorPresets>("detector_presets");
  const [microscopePresets] = useModelState<Record<string, MicroscopePreset>>("microscope_presets");
  const [thicknessPresets] = useModelState<number[]>("thickness_presets");
  const [detector] = useModelState<string>("detector");
  // committed settings; each is mirrored into `live`, which the sliders move on every event
  const [thicknessNm] = useModelState<number>("thickness_nm");
  const [voltageKV] = useModelState<number>("voltage_kV");
  const [semiangle] = useModelState<number>("semiangle_mrad");
  const [focusNm] = useModelState<number>("focus_depth_nm");
  const [tilt] = useModelState<number[]>("tilt_mrad");
  const [cameraLength] = useModelState<number>("camera_length_mm");
  const [waveFactor] = useModelState<number>("wave_window_factor");
  const [detectorPx] = useModelState<number>("detector_px");
  const [samplingMrad] = useModelState<number>("detector_mrad_per_px");
  const [stepA] = useModelState<number>("scan_step_A");
  const [scanPx] = useModelState<number>("scan_size_px");
  const [simRepeats, setSimRepeats] = useModelState<number[]>("simulation_repeats");
  const [simPixels, setSimPixels] = useModelState<number>("simulation_pixels_per_cell");
  const [simGuard, setSimGuard] = useModelState<number>("simulation_guard_A");
  const [viewDepthNm] = useModelState<number>("view_depth_nm");

  const committed = React.useMemo(() => ({
    thickness_nm: thicknessNm, voltage_kV: voltageKV, semiangle_mrad: semiangle, focus_depth_nm: focusNm,
    tilt_row: tilt?.[0] ?? 0, tilt_col: tilt?.[1] ?? 0, camera_length_mm: cameraLength, detector_px: detectorPx, wave_window_factor: waveFactor ?? 1,
    detector_mrad_per_px: samplingMrad, scan_step_A: stepA, scan_size_px: scanPx, view_depth_nm: viewDepthNm,
  }), [thicknessNm, voltageKV, semiangle, focusNm, tilt, cameraLength, detectorPx, waveFactor, samplingMrad, stepA, scanPx, viewDepthNm]);
  type Name = keyof typeof committed;
  const [live, setLive] = React.useState(committed);
  React.useEffect(() => setLive(committed), [committed]);                    // Python-side changes (observers, scripts)
  const move = (name: Name) => (value: number) => setLive((prev) => ({ ...prev, [name]: value }));
  /** Write several settings to Python in one message. */
  const commitMany = (values: Partial<Record<Name, number>> & { detector?: string }) => {
    const { detector: newDetector, ...numbers } = values;
    setLive((prev) => ({ ...prev, ...numbers }));
    for (const [name, value] of Object.entries(numbers)) {
      if (name === "tilt_row" || name === "tilt_col") continue;
      model.set(name, value);
    }
    if ("tilt_row" in numbers || "tilt_col" in numbers) model.set("tilt_mrad", [numbers.tilt_row ?? live.tilt_row, numbers.tilt_col ?? live.tilt_col]);
    if (newDetector) model.set("detector", newDetector);
    model.save_changes();
  };
  const commit = (name: Name) => (value: number) => {
    const extra: Partial<Record<Name, number>> = {};
    if (name === "thickness_nm" && live.view_depth_nm > value) extra.view_depth_nm = value;             // the view stays inside the specimen
    commitMany({ [name]: value, ...extra });
  };

  const settings: PlanSettings = {
    voltage_kV: live.voltage_kV, semiangle_mrad: live.semiangle_mrad, focus_depth_nm: live.focus_depth_nm, thickness_nm: live.thickness_nm,
    detector_px: live.detector_px, wave_window_factor: live.wave_window_factor, detector_mrad_per_px: live.detector_mrad_per_px,
    scan_step_A: live.scan_step_A, scan_size_px: live.scan_size_px, tilt_mrad: [live.tilt_row, live.tilt_col], holz_repeat_A: holzRepeat || null,
  };
  const g = planGeometry(settings);
  const sim = simulationGeometry(settings, cellSize || [1, 1, 1], simRepeats || [24, 24], simPixels || 96, simGuard ?? 5);
  const rows = checkRows(g, live.detector_px, live.scan_step_A, columnPhase || 0);
  const thicknessA = live.thickness_nm * 10;
  const viewDepthA = Math.min(Math.max(live.view_depth_nm, 0), live.thickness_nm) * 10;
  const focusA = live.focus_depth_nm * 10;
  const beamAtView = beamDiameterA(viewDepthA, focusA, live.semiangle_mrad, g.wavelength_A);
  const tiltRow = live.tilt_row * 1e-3, tiltCol = live.tilt_col * 1e-3;

  // probe position in Å from the scan centre (row, col); starts at the centre
  const [probe, setProbe] = React.useState<[number, number]>([0, 0]);
  const scanHalf = g.scan_A / 2;
  const fov = Math.max(g.scan_A + g.widest_A, g.window_A * 1.1);                        // lateral span of Top and Side (Å)
  const reach = fov / 2;
  const probeRow = Math.max(-reach, Math.min(reach, probe[0])), probeCol = Math.max(-reach, Math.min(reach, probe[1]));
  const cell = React.useMemo(() => (cellBytes ? extractFloat32(cellBytes) : null), [cellBytes]);
  const hasCell = !!cell && !!cellShape?.[0] && !!cellSize?.[0];

  const dpr = devicePixelRatio();
  const topRef = React.useRef<HTMLCanvasElement>(null), topBarRef = React.useRef<HTMLCanvasElement>(null);
  const sideRef = React.useRef<HTMLCanvasElement>(null);
  const probeRef = React.useRef<HTMLCanvasElement>(null), probeBarRef = React.useRef<HTMLCanvasElement>(null);
  const detRef = React.useRef<HTMLCanvasElement>(null), detBarRef = React.useRef<HTMLCanvasElement>(null);
  const [topView, setTopView] = React.useState<View>(HOME);
  const [sideView, setSideView] = React.useState<View>(HOME);
  const [probeView, setProbeView] = React.useState<View>(HOME);
  const [detView, setDetView] = React.useState<View>(HOME);
  useWheelZoom(topRef, setTopView); useWheelZoom(sideRef, setSideView); useWheelZoom(probeRef, setProbeView); useWheelZoom(detRef, setDetView);
  const [showSample, setShowSample] = React.useState(true);

  // offscreen tiles, reused: the cell as seen through the whole (tilted) thickness, the cell at one depth, the side slab, the probe
  const smearTile = React.useRef<HTMLCanvasElement | null>(null);
  const sharpTile = React.useRef<HTMLCanvasElement | null>(null);
  const sideTile = React.useRef<HTMLCanvasElement | null>(null);
  const probeTile = React.useRef<HTMLCanvasElement | null>(null);
  const tile = (ref: React.MutableRefObject<HTMLCanvasElement | null>) => (ref.current ??= document.createElement("canvas"));

  const smearVersion = React.useMemo(() => {
    if (!hasCell) return 0;
    const [rows_, cols_] = cellShape, values = new Float32Array(rows_ * cols_);
    const leanRow = thicknessA * Math.tan(tiltRow), leanCol = thicknessA * Math.tan(tiltCol);
    const samples = Math.hypot(leanRow, leanCol) > 0.05 ? 24 : 1;                        // leaning columns project to lines
    for (let s = 0; s < samples; s++) {
      const t = samples === 1 ? 0 : s / (samples - 1) - 0.5;
      for (let r = 0; r < rows_; r++) for (let c = 0; c < cols_; c++)
        values[r * cols_ + c] += sampleCell(cell!, rows_, cols_, cellSize[0], cellSize[1], (r / rows_) * cellSize[0] - t * leanRow, (c / cols_) * cellSize[1] - t * leanCol) / samples;
    }
    paintInto(tile(smearTile), values, cols_, rows_, "inferno", 0.7);
    return Math.random();
  }, [cell, cellShape, cellSize, thicknessA, tiltRow, tiltCol]);
  const sharpVersion = React.useMemo(() => {
    if (!hasCell) return 0;
    paintInto(tile(sharpTile), cell!, cellShape[1], cellShape[0], "inferno", 0.7);
    return Math.random();
  }, [cell, cellShape]);

  // ---- Top
  React.useEffect(() => {
    const canvas = topRef.current, ctx = sizeCanvas(canvas, dpr); if (!ctx || !canvas) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    applyView(ctx, dpr, topView);
    const pxPerA = PANEL / fov, at = (a: number) => a * pxPerA + PANEL / 2, lw = (px: number) => px / topView.zoom;
    if (smearVersion && smearTile.current) {
      const pattern = crystalPattern(ctx, smearTile.current, cellSize, pxPerA, PANEL / 2, PANEL / 2);
      if (pattern) { ctx.fillStyle = pattern; ctx.fillRect(-PANEL * 2, -PANEL * 2, PANEL * 5, PANEL * 5); }
    }
    ctx.lineWidth = lw(1.5); ctx.strokeStyle = OVERLAY.scan; ctx.strokeRect(at(-scanHalf), at(-scanHalf), 2 * scanHalf * pxPerA, 2 * scanHalf * pxPerA);
    const pr = at(probeRow), pc = at(probeCol), w = g.window_A * pxPerA;
    ctx.setLineDash([lw(4), lw(3)]); ctx.strokeStyle = OVERLAY.window; ctx.lineWidth = lw(1); ctx.strokeRect(pc - w / 2, pr - w / 2, w, w); ctx.setLineDash([]);
    ctx.strokeStyle = OVERLAY.beam; ctx.lineWidth = lw(1.8); ctx.beginPath(); ctx.arc(pc, pr, Math.max(lw(1.5), (beamAtView / 2) * pxPerA), 0, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = OVERLAY.beam; ctx.beginPath(); ctx.arc(pc, pr, lw(2.5), 0, 2 * Math.PI); ctx.fill();
    drawBar(topBarRef.current, dpr, topView, fov / PANEL, "Å");
  }, [smearVersion, cellSize, fov, scanHalf, probeRow, probeCol, beamAtView, g.window_A, topView, dpr]);

  // ---- Side: stationary slab between the entrance (0) and exit (thickness) surfaces, vacuum above and below
  const pad = 0.08 * Math.max(thicknessA, 1);
  const zTop = Math.min(0, focusA) - pad, zBottom = Math.max(thicknessA, focusA) + pad;
  const depthToY = (depth: number) => ((depth - zTop) / (zBottom - zTop)) * PANEL;
  const yToDepth = (y: number) => zTop + (y / PANEL) * (zBottom - zTop);
  const sideVersion = React.useMemo(() => {
    if (!hasCell) return 0;
    const width = Math.round(PANEL * dpr), depths = 96, values = new Float32Array(width * depths);
    for (let d = 0; d < depths; d++) {
      const depth = ((d + 0.5) / depths) * thicknessA;
      for (let x = 0; x < width; x++) {
        const colA = ((x + 0.5) / width - 0.5) * fov;
        values[d * width + x] = sampleCell(cell!, cellShape[0], cellShape[1], cellSize[0], cellSize[1], probeRow - depth * Math.tan(tiltRow), colA - depth * Math.tan(tiltCol));
      }
    }
    paintInto(tile(sideTile), values, width, depths, "inferno");
    return Math.random();
  }, [cell, cellShape, cellSize, fov, probeRow, thicknessA, tiltRow, tiltCol, dpr]);

  React.useEffect(() => {
    const canvas = sideRef.current, ctx = sizeCanvas(canvas, dpr); if (!ctx || !canvas) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    applyView(ctx, dpr, sideView);
    const lw = (px: number) => px / sideView.zoom, x = (a: number) => (a / fov + 0.5) * PANEL;
    const top = depthToY(0), bottom = depthToY(thicknessA);
    if (sideVersion && sideTile.current) { ctx.save(); ctx.globalAlpha = 0.45; ctx.imageSmoothingEnabled = true; ctx.drawImage(sideTile.current, 0, top, PANEL, bottom - top); ctx.restore(); }
    ctx.strokeStyle = OVERLAY.slab; ctx.lineWidth = lw(1);
    for (const y of [top, bottom]) { ctx.beginPath(); ctx.moveTo(-PANEL, y); ctx.lineTo(2 * PANEL, y); ctx.stroke(); }
    ctx.setLineDash([lw(3), lw(3)]); ctx.strokeStyle = OVERLAY.scan;
    for (const edge of [-scanHalf, scanHalf]) { ctx.beginPath(); ctx.moveTo(x(edge), top); ctx.lineTo(x(edge), bottom); ctx.stroke(); }
    ctx.strokeStyle = OVERLAY.window;
    for (const edge of [probeCol - g.window_A / 2, probeCol + g.window_A / 2]) { ctx.beginPath(); ctx.moveTo(x(edge), -PANEL); ctx.lineTo(x(edge), 2 * PANEL); ctx.stroke(); }
    ctx.setLineDash([]);
    const steps = 96, half = (d: number) => beamDiameterA(d, focusA, live.semiangle_mrad, g.wavelength_A) / 2;
    ctx.beginPath();
    for (let i = 0; i <= steps; i++) { const d = zTop + (i / steps) * (zBottom - zTop); ctx.lineTo(x(probeCol - half(d)), depthToY(d)); }
    for (let i = steps; i >= 0; i--) { const d = zTop + (i / steps) * (zBottom - zTop); ctx.lineTo(x(probeCol + half(d)), depthToY(d)); }
    ctx.closePath(); ctx.fillStyle = "rgba(255,213,79,0.2)"; ctx.fill(); ctx.strokeStyle = OVERLAY.beam; ctx.lineWidth = lw(1.4); ctx.stroke();
    const yf = depthToY(focusA);
    ctx.setLineDash([lw(4), lw(3)]); ctx.strokeStyle = OVERLAY.focus; ctx.lineWidth = lw(1.2);
    ctx.beginPath(); ctx.moveTo(-PANEL, yf); ctx.lineTo(2 * PANEL, yf); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = OVERLAY.focus; ctx.beginPath(); ctx.moveTo(PANEL, yf - lw(5)); ctx.lineTo(PANEL - lw(8), yf); ctx.lineTo(PANEL, yf + lw(5)); ctx.fill();
    // how much of the specimen sits above and below the focus
    ctx.font = `${lw(10)}px monospace`; ctx.textAlign = "left"; ctx.fillStyle = "rgba(255,255,255,0.85)";
    const label = (text: string, y0: number, y1: number) => { if (Math.abs(y1 - y0) * sideView.zoom > 14) { ctx.textBaseline = "middle"; ctx.fillText(text, lw(12), (y0 + y1) / 2); } };
    if (focusA > 0 && focusA < thicknessA) {
      label(`${(focusA / 10).toFixed(1)} nm above focus`, top, yf);
      label(`${((thicknessA - focusA) / 10).toFixed(1)} nm below focus`, yf, bottom);
    } else {
      label(focusA <= 0 ? "all below the focus" : "all above the focus", top, bottom);
    }
    const yv = depthToY(viewDepthA);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = lw(1.5); ctx.beginPath(); ctx.moveTo(-PANEL, yv); ctx.lineTo(2 * PANEL, yv); ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.moveTo(0, yv - lw(5)); ctx.lineTo(lw(8), yv); ctx.lineTo(0, yv + lw(5)); ctx.fill();
  }, [sideVersion, fov, scanHalf, probeCol, g.window_A, g.wavelength_A, thicknessA, live.semiangle_mrad, focusA, viewDepthA, zTop, zBottom, sideView, dpr]);

  // ---- Probe: the model window at the view depth, the crystal at that depth under it
  const probeImage = React.useMemo(() => probeIntensity(g.window_A, live.semiangle_mrad, g.wavelength_A, viewDepthA - focusA, live.detector_px * live.wave_window_factor),
    [g.window_A, live.semiangle_mrad, g.wavelength_A, viewDepthA, focusA, live.detector_px, live.wave_window_factor]);
  React.useEffect(() => {
    const canvas = probeRef.current, ctx = sizeCanvas(canvas, dpr); if (!ctx || !canvas) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    applyView(ctx, dpr, probeView);
    const pxPerA = PANEL / g.window_A;
    const withSample = showSample && !!sharpVersion && !!sharpTile.current;
    if (withSample) {
      // a column at x sits at x + depth tan(theta) at that depth; the window is centred on the probe
      const originX = PANEL / 2 + (-probeCol + viewDepthA * Math.tan(tiltCol)) * pxPerA;
      const originY = PANEL / 2 + (-probeRow + viewDepthA * Math.tan(tiltRow)) * pxPerA;
      const pattern = crystalPattern(ctx, sharpTile.current!, cellSize, pxPerA, originX, originY);
      if (pattern) { ctx.save(); ctx.globalAlpha = 0.75; ctx.fillStyle = pattern; ctx.fillRect(0, 0, PANEL, PANEL); ctx.restore(); }
    }
    paintInto(tile(probeTile), probeImage.intensity, probeImage.n, probeImage.n, withSample ? "gray" : "inferno", 0.5);
    ctx.save(); ctx.imageSmoothingEnabled = true;
    if (withSample) ctx.globalCompositeOperation = "screen";
    ctx.drawImage(probeTile.current!, 0, 0, PANEL, PANEL); ctx.restore();
    drawBar(probeBarRef.current, dpr, probeView, g.window_A / PANEL, "Å");
  }, [probeImage, showSample, sharpVersion, cellSize, g.window_A, probeRow, probeCol, viewDepthA, tiltRow, tiltCol, probeView, dpr]);

  // ---- Detector: recorded square, bright-field disk, Bragg disks (radius = semiangle), zone axis and HOLZ ring at the tilt
  React.useEffect(() => {
    const canvas = detRef.current, ctx = sizeCanvas(canvas, dpr); if (!ctx || !canvas) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    applyView(ctx, dpr, detView);
    const span = Math.max(g.theta_max_mrad * 1.15, live.semiangle_mrad * 2.4), lw = (px: number) => px / detView.zoom;
    const m = (mrad: number) => (mrad / (2 * span) + 0.5) * PANEL, rad = (mrad: number) => (mrad / (2 * span)) * PANEL;
    ctx.fillStyle = "rgba(255,255,255,0.05)"; ctx.fillRect(m(-g.theta_max_mrad), m(-g.theta_max_mrad), rad(2 * g.theta_max_mrad), rad(2 * g.theta_max_mrad));
    // at large semiangles the disks overlap (SrTiO3 {100} is 5 mrad apart at 300 kV): outline the strong ones, dot every centre
    const toMrad = g.wavelength_A * 1e3;
    for (const [gr, gc, amplitude] of bragg || []) {
      const r = gr * toMrad, c = gc * toMrad;
      if (Math.abs(r) > span * 1.5 || Math.abs(c) > span * 1.5) continue;
      if (amplitude >= 0.25) {
        ctx.strokeStyle = `rgba(255,183,77,${Math.min(0.9, 0.25 + 0.6 * Math.sqrt(amplitude))})`; ctx.lineWidth = lw(1);
        ctx.beginPath(); ctx.arc(m(c), m(r), rad(live.semiangle_mrad), 0, 2 * Math.PI); ctx.stroke();
      }
      ctx.fillStyle = `rgba(255,183,77,${Math.min(1, 0.3 + Math.sqrt(amplitude))})`;
      ctx.beginPath(); ctx.arc(m(c), m(r), lw(1 + 2.5 * Math.sqrt(amplitude)), 0, 2 * Math.PI); ctx.fill();
    }
    ctx.fillStyle = "rgba(255,255,255,0.28)"; ctx.beginPath(); ctx.arc(m(0), m(0), rad(live.semiangle_mrad), 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = lw(1.4); ctx.stroke();
    const zr = m(live.tilt_row), zc = m(live.tilt_col);
    if (g.holz_mrad !== null) {
      ctx.setLineDash([lw(4), lw(3)]); ctx.strokeStyle = OVERLAY.window; ctx.lineWidth = lw(1);
      ctx.beginPath(); ctx.arc(zc, zr, rad(g.holz_mrad), 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
    }
    if (g.tilt_mrad > 0) {
      ctx.strokeStyle = OVERLAY.zone; ctx.lineWidth = lw(1.6); const s = lw(5);
      ctx.beginPath(); ctx.moveTo(zc - s, zr - s); ctx.lineTo(zc + s, zr + s); ctx.moveTo(zc - s, zr + s); ctx.lineTo(zc + s, zr - s); ctx.stroke();
    }
    ctx.strokeStyle = OVERLAY.edge; ctx.lineWidth = lw(1.5); ctx.strokeRect(m(-g.theta_max_mrad), m(-g.theta_max_mrad), rad(2 * g.theta_max_mrad), rad(2 * g.theta_max_mrad));
    drawBar(detBarRef.current, dpr, detView, (2 * span) / PANEL, "mrad");
  }, [bragg, g.theta_max_mrad, g.wavelength_A, g.holz_mrad, g.tilt_mrad, live.semiangle_mrad, live.tilt_row, live.tilt_col, detView, dpr]);

  // ---- pointer: Top moves the probe; Side moves the focus (near the dashed line) or the view depth
  const drag = React.useRef<null | "probe" | "focus" | "depth">(null);
  const onTop = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.type === "pointerdown") { if (e.button !== 0) return; drag.current = "probe"; e.currentTarget.setPointerCapture(e.pointerId); }
    if (drag.current !== "probe") return;
    if (e.type === "pointerup" || e.type === "pointercancel") { drag.current = null; return; }
    const [sx, sy] = toScene(e, e.currentTarget, topView), pxPerA = PANEL / fov;
    setProbe([Math.max(-reach, Math.min(reach, (sy - PANEL / 2) / pxPerA)), Math.max(-reach, Math.min(reach, (sx - PANEL / 2) / pxPerA))]);
  };
  const onSide = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const [, sy] = toScene(e, e.currentTarget, sideView);
    if (e.type === "pointerdown") {
      if (e.button !== 0) return;
      drag.current = Math.abs(sy - depthToY(focusA)) * sideView.zoom < 7 ? "focus" : "depth";
      e.currentTarget.setPointerCapture(e.pointerId);
    }
    const mode = drag.current;
    if (mode !== "focus" && mode !== "depth") return;
    const depthNm = yToDepth(sy) / 10, done = e.type === "pointerup" || e.type === "pointercancel";
    const value = mode === "focus" ? Math.round(Math.max(-50, Math.min(live.thickness_nm + 50, depthNm)) * 10) / 10
                                   : Math.round(Math.max(0, Math.min(live.thickness_nm, depthNm)) * 10) / 10;
    const name: Name = mode === "focus" ? "focus_depth_nm" : "view_depth_nm";
    if (done) { drag.current = null; commit(name)(value); } else move(name)(value);
  };

  // ---- presets
  const applyMicroscope = (name: string) => {
    const p = microscopePresets?.[name]; if (!p) return;
    const native = presets?.detectors?.[p.detector]?.pixels, sampling = detectorSamplingMrad(presets, p.detector, p.camera_length_mm);
    if (!native || sampling === null) return;
    commitMany({ detector: p.detector, voltage_kV: p.voltage_kV, semiangle_mrad: p.semiangle_mrad, camera_length_mm: p.camera_length_mm,
                 detector_px: native, detector_mrad_per_px: Number(sampling.toFixed(4)) });
  };
  const applyCamera = (name: string, cameraLengthMm: number) => {
    const native = presets?.detectors?.[name]?.pixels, sampling = detectorSamplingMrad(presets, name, cameraLengthMm);
    if (!native || sampling === null) return;
    const px = name === detector ? live.detector_px : native;                           // same camera: keep the binning
    commitMany({ detector: name, camera_length_mm: cameraLengthMm, detector_px: px, detector_mrad_per_px: Number((sampling * native / px).toFixed(4)) });
  };
  const applyBinning = (bin: number) => {
    const sampling = detectorSamplingMrad(presets, detector, live.camera_length_mm), nativePx = presets?.detectors?.[detector]?.pixels;
    if (sampling === null || !nativePx) return;
    commitMany({ detector_px: Math.round(nativePx / bin), detector_mrad_per_px: Number((sampling * bin).toFixed(4)) });
  };
  const applyThickness = (thickness: number) => {
    const r = recommendedSettings(thickness, live.voltage_kV, live.semiangle_mrad, live.scan_step_A);
    commitMany({ thickness_nm: r.thickness_nm, focus_depth_nm: r.focus_depth_nm, scan_size_px: r.scan_size_px, view_depth_nm: r.focus_depth_nm });
  };
  const microscopeName = Object.entries(microscopePresets || {}).find(([, p]) =>
    p.detector === detector && p.voltage_kV === live.voltage_kV && p.semiangle_mrad === live.semiangle_mrad && p.camera_length_mm === live.camera_length_mm)?.[0] ?? "";
  const native = presets?.detectors?.[detector]?.pixels;
  const expected = native ? (detectorSamplingMrad(presets, detector, live.camera_length_mm) ?? 0) * native / live.detector_px : 0;
  const onCalibration = Math.abs(expected - live.detector_mrad_per_px) < 5e-4;
  const binning = native && onCalibration && Number.isInteger(native / live.detector_px) ? native / live.detector_px : 0;
  const cameraLengths = Array.from(new Set([...Object.keys(presets?.arina_mrad_per_px || {}).map(Number), live.camera_length_mm])).sort((a, b) => a - b);
  const recommendedMatch = (thicknessPresets || []).find((t) => {
    const r = recommendedSettings(t, live.voltage_kV, live.semiangle_mrad, live.scan_step_A);
    return r.thickness_nm === live.thickness_nm && r.focus_depth_nm === live.focus_depth_nm && r.scan_size_px === live.scan_size_px;
  });

  const zoneLabel = (z: number[]) => `[${z.join("")}]`;
  // knobs offer the values the microscope actually has; the current value is always selectable
  const semiangles = Array.from(new Set([...Object.values(microscopePresets || {}).map((p) => p.semiangle_mrad), live.semiangle_mrad])).sort((a, b) => a - b);
  const scanSizes = Array.from(new Set([64, 128, 256, 512, live.scan_size_px])).sort((a, b) => a - b);
  const focusText = live.focus_depth_nm < 0 ? `${(-live.focus_depth_nm).toFixed(1)} nm above the entrance`
    : live.focus_depth_nm > live.thickness_nm ? `${(live.focus_depth_nm - live.thickness_nm).toFixed(1)} nm below the exit`
    : `${live.focus_depth_nm.toFixed(1)} nm into the specimen`;
  const overlay = (text: string) => <Box sx={monoOverlay}>{text}</Box>;
  const barCanvas = (ref: React.RefObject<HTMLCanvasElement | null>) =>
    <canvas ref={ref} style={{ position: "absolute", inset: 0, width: PANEL, height: PANEL, pointerEvents: "none" }} />;
  const canvasStyle = (cursor: string) => ({ width: PANEL, height: PANEL, cursor, touchAction: "none" as const, display: "block" });
  const sectionLabel = (text: string) => <Typography sx={{ ...typography.labelSmall, color: colors.textMuted, mb: 0.5 }}>{text}</Typography>;

  return (
    <Box sx={{ p: 2, color: colors.text, fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", overflowX: "auto" }}>
      <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: `${SPACING.SM}px`, mb: 1 }}>
        <Typography sx={{ fontSize: 12, fontWeight: "bold", color: colors.accent }}>{title}</Typography>
        <Typography sx={{ ...typography.value, color: colors.textMuted }}>
          {live.thickness_nm.toFixed(0)} nm · {live.voltage_kV.toFixed(0)} kV · {live.semiangle_mrad.toFixed(1)} mrad · λ {g.wavelength_A.toFixed(4)} Å
        </Typography>
        <Box sx={{ flex: 1 }} />
        <SelectRow label="" colors={colors} width={196} value={microscopeName} options={Object.keys(microscopePresets || {})} tip="" onChange={applyMicroscope} />
        <SelectRow label="" colors={colors} width={120} value={recommendedMatch !== undefined ? String(recommendedMatch) : ""} options={(thicknessPresets || []).map(String)}
          format={(v) => `Recommended ${v} nm`} tip="" onChange={(v) => applyThickness(Number(v))} />
        <Button size="small" variant="outlined" sx={{ fontSize: 10, textTransform: "none", py: 0.1, px: 1, minWidth: 0 }} onClick={() => setProbe([0, 0])}>Reset Probe</Button>
      </Box>

      <Box sx={{ display: "flex", flexWrap: "wrap", gap: `${SPACING.SM}px` }}>
        <PanelFrame title="Top" colors={colors}>
          <canvas ref={topRef} style={canvasStyle("crosshair")} onPointerDown={onTop} onPointerMove={onTop} onPointerUp={onTop} onPointerCancel={onTop}
            onDoubleClick={() => setTopView(HOME)} />
          {barCanvas(topBarRef)}
          {overlay(`depth ${(viewDepthA / 10).toFixed(1)} nm\nbeam ${beamAtView.toFixed(1)} Å`)}
        </PanelFrame>
        <PanelFrame title="Side" colors={colors}>
          <canvas ref={sideRef} style={canvasStyle("ns-resize")} onPointerDown={onSide} onPointerMove={onSide} onPointerUp={onSide} onPointerCancel={onSide}
            onDoubleClick={() => setSideView(HOME)} />
          {overlay(`${live.thickness_nm.toFixed(0)} nm slab\nfocus ${live.focus_depth_nm.toFixed(1)} nm`)}
        </PanelFrame>
        <PanelFrame title="Probe" colors={colors} actions={
          <Box sx={{ display: "flex", alignItems: "center" }}>
            <Typography sx={{ ...typography.labelSmall, color: colors.textMuted }}>Sample</Typography>
            <Switch size="small" sx={switchSx} checked={showSample} onChange={(e) => setShowSample(e.target.checked)} />
          </Box>}>
          <canvas ref={probeRef} style={canvasStyle("default")} onDoubleClick={() => setProbeView(HOME)} />
          {barCanvas(probeBarRef)}
          {overlay(`${live.detector_px * live.wave_window_factor}² virtual · ${g.window_A.toFixed(1)} Å\n${((viewDepthA - focusA) / 10).toFixed(1)} nm from focus`)}
          {beamAtView > g.window_A && (
            <Box sx={{ ...monoOverlay, top: "auto", bottom: 4, right: 4, color: "#ff8a80" }}>{`beam ${beamAtView.toFixed(0)} Å > window: wraps`}</Box>
          )}
        </PanelFrame>
        <PanelFrame title="Detector (schematic)" colors={colors}>
          <canvas ref={detRef} style={canvasStyle("default")} onDoubleClick={() => setDetView(HOME)} />
          {barCanvas(detBarRef)}
          {overlay(`edge ${g.theta_max_mrad.toFixed(0)} mrad\n${live.detector_px} px`)}
        </PanelFrame>
      </Box>

      <Box sx={{ display: "flex", flexWrap: "wrap", gap: `${SPACING.MD * 2}px`, mt: 1.5 }}>
        <Box>
          {sectionLabel("Microscope")}
          <SelectRow label="Voltage" colors={colors} value={String(live.voltage_kV)} options={Array.from(new Set([...VOLTAGES, live.voltage_kV])).sort((a, b) => a - b).map(String)}
            format={(v) => `${v} kV`} tip="Accelerating voltage" onChange={(v) => commit("voltage_kV")(Number(v))} />
          <SliderRow colors={colors} label="Semiangle" value={live.semiangle_mrad} min={1} max={60} step={0.1} unit="mrad" digits={1}
            tip={`Convergence semiangle, set by the condenser aperture (in use here: ${semiangles.join(", ")} mrad); type a collaborator's value`}
            onChange={move("semiangle_mrad")} onCommit={commit("semiangle_mrad")} />
          <SliderRow colors={colors} label="C10" value={-live.focus_depth_nm} min={-Math.max(live.thickness_nm + 50, live.focus_depth_nm)} max={50} step={0.5} unit="nm" digits={1}
            tip="Defocus C10 (quantem sign: negative focuses below the entrance surface, into the specimen). Drag the dashed line in Side too."
            onChange={(v) => move("focus_depth_nm")(-v)} onCommit={(v) => commit("focus_depth_nm")(-v)} />
          <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, minHeight: 22 }}>
            <Typography sx={{ ...typography.labelSmall, width: 64 }}>Focus</Typography>
            <Typography sx={typography.value}>{focusText}</Typography>
          </Box>
        </Box>
        <Box>
          {sectionLabel("Detector")}
          <SelectRow label="Camera" colors={colors} value={detector} options={[...Object.keys(presets?.detectors || {}), "custom"]}
            tip={detector === "custom" ? "Pixels and sampling as reported for the acquisition" : presets?.detectors?.[detector]?.note || ""}
            format={(v) => (v === "custom" ? "custom" : `${v} ${presets?.detectors?.[v]?.pixels ?? ""} px`)}
            onChange={(v) => (v === "custom" ? commitMany({ detector: "custom" }) : applyCamera(v, live.camera_length_mm))} />
          {detector === "custom" ? <>
            <NumberRow colors={colors} label="Pixels" value={live.detector_px} digits={0} unit="px" integer tip="Detector pixels per side, as reported"
              onCommit={(v) => v > 0 && commit("detector_px")(v)} />
            <NumberRow colors={colors} label="Sampling" value={live.detector_mrad_per_px} digits={4} unit="mrad/px" tip="Angular sampling per detector pixel, as reported"
              onCommit={(v) => v > 0 && commit("detector_mrad_per_px")(v)} />
          </> : <>
            <SelectRow label="Length" colors={colors} value={String(live.camera_length_mm)} options={cameraLengths.map(String)} format={(v) => `${v} mm`}
              tip="Camera length; the Arina sampling is measured at 91, 115 and 185 mm (0.554, 0.461, 0.269 mrad per pixel)" onChange={(v) => applyCamera(detector, Number(v))} />
            <SelectRow label="Binning" colors={colors} value={binning ? String(binning) : ""} options={["1", "2", "4"]} format={(v) => `${v}x · ${native ? native / Number(v) : "?"} px`}
              tip="Detector binning: fewer pixels, the same total angle, a smaller model window" onChange={(v) => applyBinning(Number(v))} />
            <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, minHeight: 22 }}>
              <Tooltip title="Set by the camera, the camera length and the binning" placement="top" arrow>
                <Typography sx={{ ...typography.labelSmall, width: 64, cursor: "help" }}>Sampling</Typography>
              </Tooltip>
              <Typography sx={typography.value}>{live.detector_mrad_per_px.toFixed(3)} mrad/px{onCalibration ? "" : " (custom)"} · {live.detector_px} px</Typography>
            </Box>
          </>}
        </Box>
        <Box>
          {sectionLabel("Scan")}
          <SliderRow colors={colors} label="Step" value={live.scan_step_A} min={0.1} max={3} step={0.005} unit="Å" digits={3} tip="Probe step, set by the magnification"
            onChange={move("scan_step_A")} onCommit={commit("scan_step_A")} />
          <SelectRow label="Size" colors={colors} value={String(live.scan_size_px)} options={scanSizes.map(String)} format={(v) => `${v} × ${v}`}
            tip="Scan positions per side" onChange={(v) => commit("scan_size_px")(Number(v))} />
          <NumberRow colors={colors} label="" value={live.scan_size_px} digits={0} unit="positions per side" integer tip="Type any scan size"
            onCommit={(v) => v > 0 && commit("scan_size_px")(v)} />
          <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, minHeight: 22 }}>
            <Typography sx={{ ...typography.labelSmall, width: 64 }}>Field</Typography>
            <Typography sx={typography.value}>{g.scan_A.toFixed(1)} Å</Typography>
          </Box>
        </Box>
        <Box>
          {sectionLabel("Sample")}
          <SelectRow label="Zone" colors={colors} value={zoneLabel(zoneAxis || [0, 0, 1])} options={(zoneAxes || []).map(zoneLabel)} tip="Crystal direction along the beam"
            onChange={(v) => { const z = (zoneAxes || []).find((a) => zoneLabel(a) === v); if (z) setZoneAxis(z); }} />
          <SliderRow colors={colors} label="Thickness" value={live.thickness_nm} min={1} max={200} step={1} unit="nm" digits={0} tip="Specimen thickness"
            onChange={move("thickness_nm")} onCommit={commit("thickness_nm")} />
          <SliderRow colors={colors} label="Tilt row" value={live.tilt_row} min={-20} max={20} step={0.1} unit="mrad" digits={1} tip="Specimen tilt off the zone axis, along rows"
            onChange={move("tilt_row")} onCommit={commit("tilt_row")} />
          <SliderRow colors={colors} label="Tilt col" value={live.tilt_col} min={-20} max={20} step={0.1} unit="mrad" digits={1} tip="Specimen tilt off the zone axis, along columns"
            onChange={move("tilt_col")} onCommit={commit("tilt_col")} />
          <SliderRow colors={colors} label="View depth" value={Math.min(live.view_depth_nm, live.thickness_nm)} min={0} max={live.thickness_nm} step={0.1} unit="nm" digits={1}
            tip="Depth shown in Top and Probe; drag in Side" onChange={move("view_depth_nm")} onCommit={commit("view_depth_nm")} />
        </Box>
      </Box>

      <Box sx={{ mt: 1.5, p: 1, border: `1px solid ${colors.border}`, borderRadius: 1 }} data-section="simulation-cell">
        {sectionLabel("Simulation Cell · Geometry Only")}
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
          <Box>
            <NumberRow colors={colors} label="Cells row" value={simRepeats?.[0] || 24} digits={0} integer unit="cells" tip="Periodic simulation supercell; cover the entire scan and propagated probe"
              onCommit={v => v >= 1 && setSimRepeats([Math.round(v), simRepeats?.[1] || 24])} />
            <NumberRow colors={colors} label="Cells col" value={simRepeats?.[1] || 24} digits={0} integer unit="cells" tip="Lateral unit cells along columns"
              onCommit={v => v >= 1 && setSimRepeats([simRepeats?.[0] || 24, Math.round(v)])} />
          </Box>
          <Box>
            <NumberRow colors={colors} label="Grid / cell" value={simPixels || 96} digits={0} integer unit="px" tip="Potential pixels per unit cell; independent of detector pixel count"
              onCommit={v => v >= 1 && setSimPixels(Math.round(v))} />
            <NumberRow colors={colors} label="Guard" value={simGuard ?? 5} digits={1} unit="Å / side" tip="Extra geometric clearance; not proof that wave tails are negligible"
              onCommit={v => v >= 0 && setSimGuard(v)} />
          </Box>
          <Typography sx={{ ...typography.value, lineHeight: 1.8 }}>
            {sim.extent.map(v => v.toFixed(2)).join(" × ")} Å · {sim.gpts.join(" × ")} grid<br />
            {sim.sampling.map(v => v.toFixed(5)).join(" × ")} Å/px · {sim.zRepeats} cells to cover depth<br />
            Scan centers: {sim.span.toFixed(3)} Å · margins: {sim.margins.map(v => v.toFixed(2)).join(" / ")} Å<br />
            {sim.fits ? "Geometric clearance passes" : "Enlarge cell: geometric clearance fails"} · boundary convergence untested
          </Typography>
        </Box>
        <Typography sx={{ ...typography.labelSmall, color: colors.textMuted, mt: 0.5 }}>
          CIF / ASE input is supplied in Python. Planning controls do not run multislice or update stored simulated diffraction.
          Confirm wave tails and compare a larger cell before accepting the simulation.
        </Typography>
      </Box>

      <Box component="table" sx={{ mt: 1.5, borderCollapse: "collapse", width: "100%", maxWidth: 1100,
        "& td": { fontSize: 11, py: 0.4, pr: 1.5, verticalAlign: "top", borderTop: `1px solid ${colors.border}`, color: colors.text } }}>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} data-check={row.id} data-grade={row.status}>
              <td style={{ whiteSpace: "nowrap", width: 70 }}>
                <Box component="span" sx={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", bgcolor: GRADE_COLOR[row.status], mr: 0.75 }} />
                <Box component="span" sx={{ fontSize: 10, color: GRADE_COLOR[row.status], fontWeight: 600 }}>{row.status === "info" ? "" : row.status}</Box>
              </td>
              <td style={{ whiteSpace: "nowrap" }}>
                <Tooltip title={row.rule} placement="top" arrow><Box component="span" sx={{ cursor: "help" }}>{row.label}</Box></Tooltip>
              </td>
              <td style={{ whiteSpace: "nowrap" }}><Box component="span" sx={typography.value}>{row.value}</Box></td>
              <td><Box component="span" sx={{ color: colors.textMuted }}>{row.note}</Box></td>
            </tr>
          ))}
        </tbody>
      </Box>
      <Typography sx={{ ...typography.labelSmall, color: colors.textMuted, mt: 1 }}>
        Beam widths are the geometric cone plus the Airy disk in vacuum; channelling along atomic columns narrows the real beam.
        Recommended settings put the focus at mid-thickness and pass every check; they come from the checks, and reconstructions tested the checks only on 90-130 nm SrTiO3 (focus 17 nm). Values can be typed: click a number, enter, press Enter.
        Wheel zooms a panel, double-click resets it.
      </Typography>
      <Box sx={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 1, mt: 1.5, mb: 0.5 }} data-section="virtual-window">
        <Typography sx={typography.label}>Virtual Window</Typography>
        <Typography sx={{ ...typography.labelSmall, color: colors.textMuted }}>Optional</Typography>
        {[1, 2].map(factor => <Button key={factor} size="small" sx={{ textTransform: "none" }} variant={live.wave_window_factor === factor ? "contained" : "outlined"}
          aria-pressed={live.wave_window_factor === factor} onClick={() => commit("wave_window_factor")(factor)}>
          {factor === 1 ? "Native · Camera" : "Expanded · 2×"} · {live.detector_px * factor}²
        </Button>)}
        <Typography sx={typography.value}>{g.window_A.toFixed(2)} Å wide · {g.pixel_A.toFixed(5)} Å/px · measured {live.detector_px} × {live.detector_px} unchanged</Typography>
      </Box>
      <Typography sx={{ ...typography.labelSmall, mb: 1 }}>Camera sampling sets the native size: λ / Δθ = {(g.pixel_A * live.detector_px).toFixed(2)} Å. Optional expansion keeps pixel size and the measured detector unchanged. This adjusts the planning preview only.</Typography>
    </Box>
  );
}

export const render = createRender(PlanPtycho);
