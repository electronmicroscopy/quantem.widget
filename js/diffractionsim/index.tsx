/**
 * DiffractionSim: unit cell (left) and its diffraction pattern (right), both
 * following the same orientation. Drag the cell to tilt the crystal and the
 * pattern follows live. Every calculation (kinematical and Bloch wave
 * intensities, CBED disks, Kikuchi lines) runs here in the browser, so the
 * exported HTML page keeps working without Python.
 */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Tooltip from "@mui/material/Tooltip";
import { ThemeColors, useTheme } from "../theme";
import { COLORMAP_NAMES } from "../colormaps";
import { downloadBlob, extractBytes, preserveRestoredWidgetModelsOnSave } from "../format";
import { computeHistogramFromBytes, findDataRange, sliderRange } from "../stats";
import { sliderStyles } from "../controlStyles";
import { useCanvasRepaintSignal } from "../canvasLifecycle";
import { useHideStaticFallback } from "../staticFallback";
import { Quat, Vec3, directionIndices, matTVec, parseDirection, qmult, qnormalize, quatFromAxisAngle, quatFromZoneAxis, quatToMatrix, threeToFour } from "./math";
import {
  Reflection, blochIntensities, blochSolve, kinematicalTilted, kosselLines, kosselLookup, labReflections,
  nanobeamIntensities, nanobeamSolve, parseCrystal, parseKossel, hybridBeams, precessionTilts, slabIntensities,
} from "./physics";
import { cellGeometry, drawCell } from "./crystal3d";
import { normalizeMode } from "./mode";
import {
  Frame, cbedImage, drawDisks, drawEwaldPanel, drawImage, drawKikuchiOverlay, drawKosselLines, drawMarkers,
  hklText, nanobeamImage, setupCanvas, tiltGrid, toPx,
} from "./pattern";

const DIRECT: Reflection = { index: -1, hkl: [0, 0, 0], g: [0, 0, 0], gLen: 0, s: 0 };
const SPIN_FPS = 20; // orientation update rate while spinning
const CBED_PREC_NODES = 8; // precession ring nodes per incident direction of the cone
const QUALITY: Record<string, { grid: number; beams: number; nanobeam: number }> = {
  fast: { grid: 5, beams: 24, nanobeam: 40 },
  medium: { grid: 7, beams: 36, nanobeam: 64 },
  fine: { grid: 9, beams: 56, nanobeam: 96 },
};
const QUALITY_LABELS: Record<string, string> = { fast: "Fast", medium: "Medium", fine: "Fine" };
const ENERGIES_KEV = [60, 80, 100, 120, 200, 300];
const GAP = 8; // px between the cell and the Ewald panel
const MIN_PANEL = 220;
const PREVIEW_MAX_PX = 800; // longest side of the saved-notebook preview
const PREVIEW_DELAY_MS = 1000; // capture the preview after the view settles
const SPACING = { XS: 4, SM: 8 } as const;
const compactButton = { fontSize: 11, height: 26, minWidth: 0, px: 1, textTransform: "none" as const };
const downwardMenuProps = {
  anchorOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "top" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

// ---------------------------------------------------------------------------
function Histogram({ data, vminPct, vmaxPct, onRangeChange, lo, hi, colors, width = 130, height = 40 }: {
  data: Float32Array; vminPct: number; vmaxPct: number; onRangeChange: (a: number, b: number) => void;
  lo: number; hi: number; colors: ThemeColors; width?: number; height?: number;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = colors.bgAlt; ctx.fillRect(0, 0, width, height);
    const nb = 64, ratio = Math.floor(bins.length / nb);
    const red: number[] = [];
    for (let i = 0; i < nb; i++) { let s = 0; for (let j = 0; j < ratio; j++) s += bins[i * ratio + j] || 0; red.push(s); }
    const mx = Math.max(...red.map((v) => Math.log1p(v)), 1e-3);
    const bw = width / nb;
    const b0 = Math.floor((vminPct / 100) * nb), b1 = Math.floor((vmaxPct / 100) * nb);
    for (let i = 0; i < nb; i++) {
      const h = (Math.log1p(red[i]) / mx) * (height - 2);
      ctx.fillStyle = i >= b0 && i <= b1 ? colors.textMuted : colors.border;
      ctx.fillRect(i * bw + 0.5, height - h, Math.max(1, bw - 1), h);
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);
  const fmt = (pct: number) => {
    const v = lo + (pct / 100) * (hi - lo);
    return Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : v.toFixed(2);
  };
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas ref={canvasRef} style={{ width, height, border: `1px solid ${colors.border}` }} />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => { const [a, b] = v as number[]; onRangeChange(Math.min(a, b - 1), Math.max(b, a + 1)); }}
        min={0} max={100} size="small" valueLabelDisplay="auto" valueLabelFormat={fmt}
        aria-label="Contrast window"
        sx={{ ...sliderStyles.small, width, color: colors.accent, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}>
        <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: colors.textMuted, lineHeight: 1 }}>{fmt(vminPct)}</Typography>
        <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: colors.textMuted, lineHeight: 1 }}>{fmt(vmaxPct)}</Typography>
      </Box>
    </Box>
  );
}

function LabeledSlider({ label, value, onChange, min, max, step, fmt, colors, width = 180, disabled }: {
  label: string; value: number; onChange: (v: number) => void; min: number; max: number; step: number;
  fmt: (v: number) => string; colors: ThemeColors; width?: number; disabled?: boolean;
}) {
  const color = disabled ? colors.textMuted : colors.text;
  return (
    <Box sx={{ width }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", gap: 1 }}>
        <Typography sx={{ fontSize: 11, color }}>{label}</Typography>
        <Typography sx={{ fontSize: 11, fontFamily: "monospace", color }}>{fmt(value)}</Typography>
      </Box>
      <Slider value={value} min={min} max={max} step={step} size="small" disabled={disabled} aria-label={label}
        onChange={(_, v) => onChange(v as number)}
        sx={{ ...sliderStyles.small, py: 0.75, color: colors.accent }} />
    </Box>
  );
}

/** A compact label and the switch it names, kept on one line when the row wraps. */
function SwitchPair({ label, checked, onChange, colors, title }: {
  label: string; checked: boolean; onChange: (v: boolean) => void; colors: ThemeColors; title?: string;
}) {
  return (
    <Box component="label" title={title} sx={{ display: "inline-flex", alignItems: "center", gap: "2px", whiteSpace: "nowrap", cursor: "pointer" }}>
      <Typography sx={{ fontSize: 11, color: colors.text }}>{label}</Typography>
      <Switch size="small" checked={checked} onChange={(e) => onChange(e.target.checked)}
        sx={{ "& .MuiSwitch-track": { bgcolor: colors.textMuted } }} />
    </Box>
  );
}

/** A compact label and the control it names, kept on one line when the row wraps. */
function ControlPair({ label, colors, children }: { label: string; colors: ThemeColors; children: React.ReactNode }) {
  return (
    <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px`, whiteSpace: "nowrap" }}>
      <Typography sx={{ fontSize: 11, color: colors.text }}>{label}</Typography>
      {children}
    </Box>
  );
}

function fmtIndices(v: [number, number, number] | null, hexagonal = false): string {
  if (!v) return "—";
  const idx: number[] = hexagonal ? threeToFour(v) : v;
  return "[" + idx.map((h) => (h < 0 ? `${-h}̅` : `${h}`)).join("") + "]";
}

function slug(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "crystal";
}

/** Both panels side by side on one canvas, scaled by `scale` (1: device pixels). */
function composePanels(
  cell: HTMLCanvasElement | null, ewald: HTMLCanvasElement | null, pattern: HTMLCanvasElement | null,
  background: string, cellBackground: string, scale = 1,
): HTMLCanvasElement | null {
  if (!cell || !pattern) return null;
  const gap = Math.round(GAP * (window.devicePixelRatio || 1));
  const leftH = cell.height + (ewald ? gap + ewald.height : 0);
  const out = document.createElement("canvas");
  out.width = Math.max(1, Math.round((cell.width + gap + pattern.width) * scale));
  out.height = Math.max(1, Math.round(Math.max(leftH, pattern.height) * scale));
  const ctx = out.getContext("2d");
  if (!ctx) return null;
  ctx.fillStyle = background; ctx.fillRect(0, 0, out.width, out.height);
  ctx.scale(scale, scale);
  ctx.fillStyle = cellBackground; // the cell canvas is transparent over its CSS background
  ctx.fillRect(0, 0, cell.width, cell.height);
  ctx.drawImage(cell, 0, 0);
  if (ewald) ctx.drawImage(ewald, 0, cell.height + gap);
  ctx.drawImage(pattern, cell.width + gap, 0);
  return out;
}

// ---------------------------------------------------------------------------
function DiffSim() {
  const model = useModel();
  const rootRef = React.useRef<HTMLDivElement>(null);
  const [offline] = useModelState<boolean>("offline");
  const { themeInfo, colors } = useTheme(!!offline);
  const dark = themeInfo.theme === "dark";
  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);
  useHideStaticFallback(model, rootRef);
  const repaint = useCanvasRepaintSignal();

  const [title] = useModelState<string>("title");
  const [crystalJson, setCrystalJson] = useModelState<string>("crystal_json");
  const [kosselJson] = useModelState<string>("kossel_json");
  const [presets] = useModelState<string[]>("presets");
  const [offlinePresets] = useModelState<Record<string, unknown>>("_offline_presets");
  const [preset, setPreset] = useModelState<string>("preset");
  const [energy, setEnergy] = useModelState<number>("energy_keV");
  const [orientation, setOrientation] = useModelState<number[]>("orientation");
  const [modeTrait, setMode] = useModelState<string>("mode");
  const mode = normalizeMode(modeTrait); // "kossel" (earlier name) reads as "kikuchi"
  const [render, setRender] = useModelState<string>("render");
  const [dynamical, setDynamical] = useModelState<boolean>("dynamical");
  const [thickness, setThickness] = useModelState<number>("thickness_A");
  const [semiangle, setSemiangle] = useModelState<number>("semiangle_mrad");
  const [precession, setPrecession] = useModelState<number>("precession_deg");
  const [nPrecession] = useModelState<number>("n_precession");
  const [sigma, setSigma] = useModelState<number>("sigma_excitation_inv_A");
  const [stepDeg, setStepDeg] = useModelState<number>("rotation_step_deg");
  const [spinSpeed, setSpinSpeed] = useModelState<number>("rotation_speed_deg_per_s");
  const [scaling, setScaling] = useModelState<string>("scaling");
  const [power, setPower] = useModelState<number>("power");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [cmapLuts] = useModelState<Record<string, number[]>>("cmap_luts");
  const [markerPower, setMarkerPower] = useModelState<number>("marker_power");
  const [markerSize, setMarkerSize] = useModelState<number>("marker_size_px");
  const [vminPct, setVminPct] = useModelState<number>("vmin_pct");
  const [vmaxPct, setVmaxPct] = useModelState<number>("vmax_pct");
  const [showLabels, setShowLabels] = useModelState<boolean>("show_labels");
  const [showHkl, setShowHkl] = useModelState<boolean>("show_hkl");
  const [showCellAxes, setShowCellAxes] = useModelState<boolean>("show_cell_axes");
  const [nCells, setNCells] = useModelState<number[]>("n_cells");
  const [polyhedra, setPolyhedra] = useModelState<boolean>("polyhedra");
  const [showEwald, setShowEwald] = useModelState<boolean>("show_ewald");
  const [panelWidthPx] = useModelState<number>("panel_width_px");
  const [status] = useModelState<string>("status");
  const [patternRange, setPatternRange] = useModelState<number>("pattern_range_inv_A");
  const [fieldMrad, setFieldMrad] = useModelState<number>("field_mrad");
  const [SG_MAX] = useModelState<number>("sg_max_inv_A");
  const [quality, setQuality] = useModelState<string>("quality");
  const [kikuchi, setKikuchi] = useModelState<boolean>("show_kikuchi");
  const [viewFrom] = useModelState<string>("view_from");
  const [saveState] = useModelState<boolean>("_save_state");

  // HTML export bridge
  const [, setExportRequest] = useModelState<string>("export_request");
  const [exportStatus] = useModelState<string>("export_status");
  const [exportEnabled] = useModelState<boolean>("export_enabled");
  const [exportPayload] = useModelState<DataView>("export_payload");
  const [exportPayloadId] = useModelState<string>("export_payload_id");
  const [exportFilename] = useModelState<string>("export_filename");
  const exportCounterRef = React.useRef(0);
  const pendingExportRef = React.useRef<string>("");
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);

  const crystal = React.useMemo(() => parseCrystal(crystalJson), [crystalJson]);
  const kossel = React.useMemo(() => parseKossel(kosselJson), [kosselJson]);
  const nCellsSafe: [number, number, number] = [nCells?.[0] || 1, nCells?.[1] || 1, nCells?.[2] || 1];
  const geom = React.useMemo(() => (crystal ? cellGeometry(crystal, nCellsSafe, polyhedra) : null), [crystal, nCellsSafe.join(","), polyhedra]);
  const cmapNames = React.useMemo(() => [...COLORMAP_NAMES, ...Object.keys(cmapLuts || {})], [cmapLuts]);

  const viewX = viewFrom === "gun" ? 1 : -1;
  const qMaxDisp = Math.min(Math.max(patternRange || 0, 0.2), crystal?.k_max ?? 4);
  const [zoneText, setZoneText] = React.useState("");
  const [ptrDrag, setDragging] = React.useState(false);
  // continuous slow rotation about the screen axes (toggle buttons next to the step box)
  const [spin, setSpin] = React.useState<{ x: boolean; y: boolean }>({ x: false, y: false });
  const spinning = spin.x || spin.y;
  const dragging = ptrDrag || spinning; // reduced quality while the orientation is changing
  const draggingRef = React.useRef(dragging);
  draggingRef.current = dragging;

  // panel sizes from the container width: pattern square S on the right; on
  // the left the cell square Sc above the Ewald panel (Sc x Se), S = Sc + GAP + Se
  const [hostW, setHostW] = React.useState<number | null>(null);
  React.useEffect(() => {
    const el = rootRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => setHostW(Math.round(entry.contentRect.width)));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const requested = Number.isFinite(panelWidthPx) && panelWidthPx > 0 ? panelWidthPx : 420;
  const S = Math.round(Math.max(MIN_PANEL, hostW ? Math.min(requested, hostW - 24) : requested));
  const Sc = showEwald ? Math.round((S - GAP) / 1.5) : S;
  const Se = showEwald ? S - GAP - Sc : 0;

  // orientation: local quaternion for smooth dragging, pushed to the model with a throttle
  const [quat, setQuatLocal] = React.useState<Quat>(orientation as Quat);
  const quatRef = React.useRef<Quat>(quat);
  React.useEffect(() => { const q = orientation as Quat; quatRef.current = q; setQuatLocal(q); }, [orientation.join(",")]);
  const pushTimer = React.useRef<number | null>(null);
  const setQuat = React.useCallback((q: Quat, immediate = false) => {
    quatRef.current = q;
    setQuatLocal(q);
    const push = () => { pushTimer.current = null; setOrientation([...quatRef.current]); };
    if (immediate) { if (pushTimer.current) window.clearTimeout(pushTimer.current); push(); }
    else if (!pushTimer.current) pushTimer.current = window.setTimeout(push, 200);
  }, [setOrientation]);

  // axis given in SCREEN coordinates (x right, y up, z toward the viewer); mapped to the lab frame by the view
  const rotateLab = React.useCallback((axis: Vec3, deg: number, immediate = true) => {
    const dq = quatFromAxisAngle([viewX * axis[0], axis[1], viewX * axis[2]], (deg * Math.PI) / 180);
    setQuat(qnormalize(qmult(dq, quatRef.current)), immediate);
  }, [setQuat, viewX]);

  const spinRef = React.useRef({ ...spin, speed: spinSpeed || 6 });
  spinRef.current = { ...spin, speed: spinSpeed || 6 };
  React.useEffect(() => {
    if (!spinning) return;
    let raf = 0, last = 0, due = 0, pendX = 0, pendY = 0;
    const tick = (t: number) => {
      const dt = last ? Math.min(0.1, (t - last) / 1000) : 0;
      last = t;
      if (spinRef.current.y) pendY += spinRef.current.speed * dt;
      if (spinRef.current.x) pendX += spinRef.current.speed * dt;
      if (t >= due) {
        due = t + 1000 / SPIN_FPS;
        if (pendY) rotateLab([0, 1, 0], pendY, false);
        if (pendX) rotateLab([1, 0, 0], pendX, false);
        pendX = pendY = 0;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [spinning, rotateLab]);

  // ---- pointer handling on the cell canvas (mouse and touch) -------------
  const cellRef = React.useRef<HTMLCanvasElement>(null);
  const pointers = React.useRef<Map<number, [number, number]>>(new Map());
  const onPointerDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    pointers.current.set(e.pointerId, [e.clientX, e.clientY]);
    setDragging(true);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const prev = pointers.current.get(e.pointerId);
    if (!prev) return;
    const cur: [number, number] = [e.clientX, e.clientY];
    if (pointers.current.size >= 2) {
      // two fingers: twist about the beam axis
      const other = [...pointers.current.entries()].find(([id]) => id !== e.pointerId);
      if (other) {
        const [ox, oy] = other[1];
        const a0 = Math.atan2(prev[1] - oy, prev[0] - ox);
        const a1 = Math.atan2(cur[1] - oy, cur[0] - ox);
        let da = a1 - a0;
        if (da > Math.PI) da -= 2 * Math.PI;
        if (da < -Math.PI) da += 2 * Math.PI;
        rotateLab([0, 0, 1], (-da * 180) / Math.PI, false);
      }
    } else if (e.shiftKey) {
      // shift-drag: twist about the beam (the desktop version of the two-finger gesture)
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
      let da = Math.atan2(cur[1] - cy, cur[0] - cx) - Math.atan2(prev[1] - cy, prev[0] - cx);
      if (da > Math.PI) da -= 2 * Math.PI;
      if (da < -Math.PI) da += 2 * Math.PI;
      rotateLab([0, 0, 1], (-da * 180) / Math.PI, false);
    } else {
      const dx = cur[0] - prev[0], dy = cur[1] - prev[1];
      const degPerPx = 180 / Sc;
      const ang = Math.hypot(dx, dy) * degPerPx;
      if (ang > 0) rotateLab([dy, dx, 0], ang, false); // trackball: the face nearest the viewer follows the pointer
    }
    pointers.current.set(e.pointerId, cur);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size === 0) { setDragging(false); setQuat(quatRef.current, true); }
  };

  // ---- pointer handling on the pattern canvas ---------------------------------
  // Dragging the pattern by dq (1/A, or rad in Kikuchi mode) moves the zone
  // axis so the pattern follows: the Laue circle centre sits at q = -k0 delta
  // for a zone axis tilted by delta, so the crystal tilts by -dq/k0.
  const shiftPattern = React.useCallback((dqx: number, dqy: number, inverseAngstrom: boolean, immediate: boolean) => {
    const k0 = crystal ? 1 / crystal.wavelength : 1;
    const ax = inverseAngstrom ? dqx / k0 : dqx, ay = inverseAngstrom ? dqy / k0 : dqy;
    const ang = Math.hypot(ax, ay);
    if (ang <= 0) return;
    const dq = quatFromAxisAngle([ay, -ax, 0], ang);
    setQuat(qnormalize(qmult(dq, quatRef.current)), immediate);
  }, [crystal, setQuat]);
  const patPointers = React.useRef<Map<number, [number, number]>>(new Map());
  const patScale = React.useRef(1); // px per unit of the current frame
  const onPatDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    patPointers.current.set(e.pointerId, [e.clientX, e.clientY]);
    setDragging(true);
  };
  const onPatMove = (e: React.PointerEvent) => {
    const prev = patPointers.current.get(e.pointerId);
    if (!prev) return;
    const cur: [number, number] = [e.clientX, e.clientY];
    if (patPointers.current.size >= 2) {
      const other = [...patPointers.current.entries()].find(([id]) => id !== e.pointerId);
      if (other) {
        const [ox, oy] = other[1];
        let da = Math.atan2(cur[1] - oy, cur[0] - ox) - Math.atan2(prev[1] - oy, prev[0] - ox);
        if (da > Math.PI) da -= 2 * Math.PI;
        if (da < -Math.PI) da += 2 * Math.PI;
        rotateLab([0, 0, 1], (-da * 180) / Math.PI, false);
      }
    } else if (e.shiftKey) {
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
      let da = Math.atan2(cur[1] - cy, cur[0] - cx) - Math.atan2(prev[1] - cy, prev[0] - cx);
      if (da > Math.PI) da -= 2 * Math.PI;
      if (da < -Math.PI) da += 2 * Math.PI;
      rotateLab([0, 0, 1], (-da * 180) / Math.PI, false);
    } else {
      const dx = (viewX * (cur[0] - prev[0])) / patScale.current, dy = -(cur[1] - prev[1]) / patScale.current;
      shiftPattern(dx, dy, mode !== "kikuchi", false);
    }
    patPointers.current.set(e.pointerId, cur);
  };
  const onPatUp = (e: React.PointerEvent) => {
    patPointers.current.delete(e.pointerId);
    if (patPointers.current.size === 0) { setDragging(false); setQuat(quatRef.current, true); }
  };
  // Double-click on a visible disk: tilt the crystal to the exact Bragg
  // condition of that reflection (two-beam: the Laue circle through 000 and
  // g). A crystal tilt (wx, wy) about the lab axes changes s_g by
  // wx g_y - wy g_x, so w = s_g (-g_y, g_x) / |g_xy|^2 zeroes it. On empty
  // space the Laue-circle centre moves to the clicked point (same sense:
  // the zone axis tilts away from the click by q / k0); in Kikuchi mode the
  // clicked direction of the tilt map moves onto the axis.
  const onPatDoubleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    if ((mode === "nanobeam" || mode === "cbed") && crystal) {
      // a convergent beam draws the same reflections as wide disks, so both
      // modes snap to the same two-beam condition the same way
      const beamList = mode === "cbed" && cbed ? cbed.beams : nbBeams;
      const intenList = mode === "cbed" ? cbedMean : nbInten;
      const snapPx = mode === "cbed"
        ? Math.max(8, k0 * Math.sin(alpha) * patScale.current)
        : Math.max(8, 1.2 * (render === "disks" ? k0 * Math.sin(alpha) * patScale.current : markerSize * (S / 420)));
      // candidates under the click: several reflections of different g_z
      // share one spot (in hcp the first HOLZ layer is only 0.21 1/A up), so
      // take the one that needs the SMALLEST tilt to reach Bragg, and never
      // jump by more than 5 degrees
      let best: Reflection | null = null, bestTilt = (5 * Math.PI) / 180;
      let iMax = 0;
      for (let i = 1; i < beamList.length; i++) iMax = Math.max(iMax, intenList[i] || 0);
      for (let i = 0; i < beamList.length; i++) {
        const b = beamList[i];
        if (b.index < 0 || !(intenList[i] > 1e-4 * iMax)) continue;
        const [px, py] = toPx(frame, b.g[0], b.g[1]);
        if (Math.hypot(px - x, py - y) > snapPx) continue;
        const gxy = Math.hypot(b.g[0], b.g[1]);
        if (gxy < 1e-6) continue;
        const tilt = Math.abs(b.s) / gxy;
        if (tilt < bestTilt) { bestTilt = tilt; best = b; }
      }
      if (best) {
        const gxy2 = best.g[0] ** 2 + best.g[1] ** 2;
        const wx = (-best.s * best.g[1]) / gxy2, wy = (best.s * best.g[0]) / gxy2;
        setQuat(qnormalize(qmult(quatFromAxisAngle([wx, wy, 0], Math.hypot(wx, wy)), quatRef.current)), true);
        return;
      }
    }
    const qx = (viewX * (x - rect.width / 2)) / patScale.current, qy = -(y - rect.height / 2) / patScale.current;
    if (mode === "kikuchi") shiftPattern(-qx, -qy, false, true);
    else shiftPattern(qx, qy, true, true);
  };

  // ---- saved-notebook preview ----------------------------------------------
  // After the view settles, send a PNG of both panels to Python. It is the
  // static image a saved notebook shows without a kernel (save_state=False).
  // Not needed with save_state=True or in an exported page.
  const previewTimer = React.useRef<number | null>(null);
  const lastPreview = React.useRef<string>("");
  const previewEnabledRef = React.useRef(false);
  previewEnabledRef.current = !offline && !saveState;
  const schedulePreview = React.useCallback(() => {
    if (!previewEnabledRef.current) return;
    if (previewTimer.current) window.clearTimeout(previewTimer.current);
    const capture = () => {
      previewTimer.current = null;
      if (draggingRef.current) { previewTimer.current = window.setTimeout(capture, PREVIEW_DELAY_MS); return; }
      const cell = cellRef.current, pattern = patRef.current;
      if (!cell || !pattern) return;
      const ewald = ewaldRef.current;
      const fullW = cell.width + pattern.width;
      const fullH = Math.max(cell.height + (ewald ? ewald.height : 0), pattern.height);
      const scale = Math.min(1, PREVIEW_MAX_PX / Math.max(fullW, fullH, 1));
      const out = composePanels(cell, ewald, pattern, colorsRef.current.bg, colorsRef.current.bgAlt, scale);
      if (!out) return;
      const b64 = out.toDataURL("image/png").split(",")[1] || "";
      if (!b64 || b64 === lastPreview.current) return;
      lastPreview.current = b64;
      model.set("_static_fallback_mime", "image/png");
      model.set("_static_fallback_jpeg", b64);
      model.save_changes();
    };
    previewTimer.current = window.setTimeout(capture, PREVIEW_DELAY_MS);
  }, [model]);
  React.useEffect(() => () => { if (previewTimer.current) window.clearTimeout(previewTimer.current); }, []);
  const colorsRef = React.useRef(colors);
  colorsRef.current = colors;

// ---- derived geometry ---------------------------------------------------
  const R = React.useMemo(() => quatToMatrix(quat), [quat]);
  const zoneAxis = React.useMemo(() => {
    if (!crystal) return null;
    const dc = matTVec(R, [0, 0, 1]);
    return directionIndices(crystal.cell, dc);
  }, [crystal, R]);
  const k0 = crystal ? 1 / crystal.wavelength : 0;
  const qual = QUALITY[quality] || QUALITY.medium;

  // ---- nanobeam -----------------------------------------------------------
  // one Bloch solution per precession node (a single untilted node without precession)
  const precNodes = React.useMemo(() => {
    const n = dragging ? Math.max(6, Math.round((nPrecession || 24) / 2)) : nPrecession || 24;
    return precessionTilts(k0, precession || 0, n);
  }, [k0, precession, nPrecession, dragging]);
  const nbSolution = React.useMemo(() => {
    if (!crystal || mode !== "nanobeam" || !dynamical) return null;
    // the physics uses every reflection the crystal carries; the pattern range only crops the drawing
    return nanobeamSolve(crystal, quat, crystal.k_max, SG_MAX, dragging ? Math.min(qual.nanobeam, 40) : qual.nanobeam, precNodes);
  }, [crystal, quat, mode, dynamical, dragging, qual, SG_MAX, precNodes]);
  const nbBeams = React.useMemo<Reflection[]>(() => {
    if (!crystal || mode !== "nanobeam") return [];
    if (nbSolution) return nbSolution.beams;
    return [DIRECT, ...labReflections(crystal, quat, crystal.k_max)];
  }, [crystal, quat, mode, nbSolution]);
  const nb = { beams: nbBeams, nDyn: nbSolution ? Math.round(nbSolution.nDynMean) : 0 };
  const nbInten = React.useMemo(() => {
    if (!crystal || mode !== "nanobeam") return new Float64Array(0);
    if (nbSolution) return nanobeamIntensities(crystal, nbSolution, thickness);
    const out = new Float64Array(nbBeams.length);
    for (const t of precNodes) {
      const v = kinematicalTilted(crystal, nbBeams, t, sigma);
      for (let i = 0; i < out.length; i++) out[i] += v[i] / precNodes.length;
    }
    return out;
  }, [crystal, nbBeams, nbSolution, precNodes, mode, thickness, sigma]);

  // ---- CBED ---------------------------------------------------------------
  const alpha = semiangle * 1e-3;
  const cbed = React.useMemo(() => {
    if (!crystal || mode !== "cbed") return null;
    const Rk = k0 * Math.sin(alpha);
    const grid = tiltGrid(Rk, dragging ? 5 : qual.grid);
    // every incident direction of the cone is itself precessed, so the cost is
    // the grid times the ring: fewer ring nodes here than in nanobeam
    const nodes = precessionTilts(k0, precession || 0, dragging ? 4 : CBED_PREC_NODES);
    if (!dynamical) {
      return { grid, beams: [DIRECT, ...labReflections(crystal, quat, crystal.k_max)], nDyn: 0, nodes, sols: null };
    }
    const { beams, nDyn } = hybridBeams(crystal, quat, crystal.k_max, SG_MAX, dragging ? Math.min(qual.beams, 24) : qual.beams, Math.sin(alpha));
    const dyn = beams.slice(0, nDyn);
    const sols: ReturnType<typeof blochSolve>[] = []; // grid tilt major, ring node minor
    for (const t of grid.tilts) {
      for (const nd of nodes) sols.push(blochSolve(crystal, dyn, [t[0] + nd[0], t[1] + nd[1]]));
    }
    return { grid, beams, nDyn, nodes, sols };
  }, [crystal, quat, qMaxDisp, mode, dynamical, alpha, k0, dragging, qual, SG_MAX, precession]);
  const cbedInten = React.useMemo(() => {
    if (!crystal || !cbed) return null;
    const { grid, beams, nDyn, nodes, sols } = cbed;
    return grid.tilts.map((t, i) => {
      const out = new Float64Array(beams.length);
      for (let k = 0; k < nodes.length; k++) {
        const tilt: [number, number] = [t[0] + nodes[k][0], t[1] + nodes[k][1]];
        const acc = new Float64Array(beams.length);
        if (sols) {
          acc.set(blochIntensities(sols[i * nodes.length + k], thickness));
          slabIntensities(crystal, beams, nDyn, tilt, thickness, acc);
        } else {
          acc.set(kinematicalTilted(crystal, beams, tilt, sigma));
        }
        for (let b = 0; b < out.length; b++) out[b] += acc[b] / nodes.length;
      }
      return out;
    });
  }, [crystal, cbed, thickness, sigma]);
  // disk-averaged intensity of every beam, for the double-click snap
  const cbedMean = React.useMemo(() => {
    if (!cbed || !cbedInten) return new Float64Array(0);
    const out = new Float64Array(cbed.beams.length);
    for (const arr of cbedInten) for (let b = 0; b < out.length; b++) out[b] += arr[b] / cbedInten.length;
    return out;
  }, [cbed, cbedInten]);

  // ---- Kossel -------------------------------------------------------------
  const fieldRad = fieldMrad * 1e-3;
  const lines = React.useMemo(() => {
    if (!crystal || (mode !== "kikuchi" && !kikuchi)) return [];
    const fov = mode === "kikuchi" ? fieldRad : qMaxDisp / k0;
    return kosselLines(crystal, quat, Math.min(crystal.k_max, 2.5), fov);
  }, [crystal, quat, mode, fieldRad, kikuchi, qMaxDisp, k0]);

  // ---- pixel image of the current mode --------------------------------------
  const frame: Frame = React.useMemo(() => ({ size: S, qMax: mode === "kikuchi" ? fieldRad : qMaxDisp, viewX }), [S, mode, fieldRad, qMaxDisp, viewX]);
  patScale.current = (0.5 * S * 0.92) / frame.qMax;
  const pixelMode = (mode === "nanobeam" && render === "pixels") || mode === "cbed" || (mode === "kikuchi" && render === "pixels");
  const image = React.useMemo<Float32Array | null>(() => {
    if (!crystal || !pixelMode) return null;
    if (mode === "nanobeam") return nanobeamImage(frame, nbBeams, nbInten, Math.max(1.5, S / 200));
    if (mode === "cbed" && cbed && cbedInten) return cbedImage(frame, cbed.beams, cbed.grid, cbedInten);
    if (mode === "kikuchi" && kossel) return kosselLookup(kossel, quat, fieldRad, S, thickness, viewX);
    return null;
  }, [crystal, pixelMode, mode, frame, nbBeams, nbInten, cbed, cbedInten, kossel, quat, fieldRad, S, thickness, viewX]);
  const display = React.useMemo(() => {
    if (!image) return null;
    let data = image;
    if (scaling === "log") {
      let mx = 0;
      for (let i = 0; i < image.length; i++) if (isFinite(image[i])) mx = Math.max(mx, image[i]);
      const eps = 1e-4 * (mx || 1);
      data = new Float32Array(image.length);
      for (let i = 0; i < image.length; i++) data[i] = isFinite(image[i]) ? Math.log10(Math.max(image[i], 0) + eps) : NaN;
    } else if (scaling === "power") {
      const pw = Math.min(Math.max(power || 0.5, 0.05), 1);
      data = new Float32Array(image.length);
      for (let i = 0; i < image.length; i++) data[i] = isFinite(image[i]) ? Math.pow(Math.max(image[i], 0), pw) : NaN;
    }
    let { min: lo, max: hi } = findDataRange(data);
    if (!(hi > lo)) { lo = 0; hi = 1; }
    return { data, lo, hi };
  }, [image, scaling, power]);

  // ---- drawing --------------------------------------------------------------
  React.useEffect(() => {
    const canvas = cellRef.current;
    if (!canvas || !geom || !crystal) return;
    drawCell(canvas, geom, quat, Sc, { dark, showAxes: showCellAxes, showLabels: showLabels, atomScale: 0.45, viewX });
    schedulePreview();
  }, [geom, quat, Sc, dark, showCellAxes, showLabels, viewX, repaint]);

  const ewaldRef = React.useRef<HTMLCanvasElement>(null);
  React.useEffect(() => {
    const canvas = ewaldRef.current;
    if (!canvas || !crystal || !showEwald || Se <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Sc * dpr || canvas.height !== Se * dpr) { canvas.width = Sc * dpr; canvas.height = Se * dpr; }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const refl = mode === "nanobeam" && nbBeams.length ? nbBeams : labReflections(crystal, quat, crystal.k_max);
    drawEwaldPanel(ctx, Sc, Se, refl, k0, qMaxDisp, SG_MAX, viewX, dark, colors.bgAlt, mode !== "kikuchi" ? precession || 0 : 0);
  }, [quat, Sc, Se, dark, colors, viewX, showEwald, crystal, mode, nbBeams, qMaxDisp, k0, SG_MAX, precession, repaint]);

  const patRef = React.useRef<HTMLCanvasElement>(null);
  React.useEffect(() => {
    const canvas = patRef.current;
    if (!canvas || !crystal) return;
    const ctx = setupCanvas(canvas, S);
    if (!ctx) return;
    if (display) {
      const { vmin, vmax } = sliderRange(display.lo, display.hi, vminPct, vmaxPct);
      drawImage(ctx, frame, display.data, cmap, vmin, vmax, dark, mode === "kikuchi" ? "mrad" : "Å⁻¹", cmapLuts?.[cmap]);
      if (mode !== "kikuchi" && kikuchi) drawKikuchiOverlay(ctx, frame, lines, k0, true);
      if (mode === "nanobeam" && showHkl) labelBeams(ctx, frame, nbBeams, nbInten);
    } else if (mode === "nanobeam" && render === "disks") {
      drawDisks(ctx, frame, nbBeams, nbInten, dark, showHkl, k0 * Math.sin(alpha), markerPower);
      if (kikuchi) drawKikuchiOverlay(ctx, frame, lines, k0, dark);
    } else if (mode === "nanobeam") {
      drawMarkers(ctx, frame, nbBeams, nbInten, dark, showHkl, !dynamical, markerPower, markerSize);
      if (kikuchi) drawKikuchiOverlay(ctx, frame, lines, k0, dark);
    } else if (mode === "kikuchi") {
      if (render === "pixels" && !kossel) {
        ctx.fillStyle = dark ? "#000" : "#fff"; ctx.fillRect(0, 0, S, S);
        ctx.fillStyle = colors.text; ctx.font = "13px sans-serif"; ctx.textAlign = "center";
        ctx.fillText("No Kossel reference pattern loaded", S / 2, S / 2 - 10);
        ctx.fillText(offline ? "(export the page after compute_kossel_reference)" : "Press Compute Reference below", S / 2, S / 2 + 10);
      } else {
        drawKosselLines(ctx, frame, lines, dark, showHkl, 0.02);
      }
    } else if (mode === "cbed") {
      ctx.fillStyle = dark ? "#000" : "#fff"; ctx.fillRect(0, 0, S, S);
    }
    schedulePreview();
  }, [crystal, display, frame, mode, render, dark, colors, cmap, cmapLuts, vminPct, vmaxPct, nbBeams, nbInten, showHkl, dynamical, kikuchi, lines, k0, kossel, S, offline, markerPower, markerSize, alpha, repaint]);

  // ---- actions ---------------------------------------------------------------
  const goZoneAxis = () => {
    const uvw = parseDirection(zoneText);
    if (!uvw || !crystal) return;
    const c = crystal.cell;
    const d: Vec3 = [
      uvw[0] * c[0][0] + uvw[1] * c[1][0] + uvw[2] * c[2][0],
      uvw[0] * c[0][1] + uvw[1] * c[1][1] + uvw[2] * c[2][1],
      uvw[0] * c[0][2] + uvw[1] * c[1][2] + uvw[2] * c[2][2],
    ];
    setQuat(quatFromZoneAxis(d), true);
  };
  const choosePreset = (name: string) => {
    if (offline) {
      const data = offlinePresets?.[name];
      if (data) { setCrystalJson(JSON.stringify(data)); setPreset(name); }
    } else {
      setPreset(name);
    }
  };
  const baseName = `${slug(crystal?.name || "crystal")}_${mode}`;
  const panelsPng = (onBlob: (blob: Blob) => void) => {
    const out = composePanels(cellRef.current, showEwald ? ewaldRef.current : null, patRef.current, colors.bg, colors.bgAlt);
    out?.toBlob((blob) => { if (blob) onBlob(blob); }, "image/png");
  };
  const handleCopy = () => {
    panelsPng((blob) => {
      try { navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]); }
      catch { downloadBlob(blob, `${baseName}.png`); }
    });
  };
  const handleExportPng = () => {
    setExportAnchor(null);
    panelsPng((blob) => downloadBlob(blob, `${baseName}.png`));
  };
  const handleExportHtml = () => {
    setExportAnchor(null);
    exportCounterRef.current += 1;
    const id = `html-${exportCounterRef.current}`;
    pendingExportRef.current = id;
    setExportRequest(JSON.stringify({ mode: "single", encoding: "full", downsample: null, download: true, id, filename: `${baseName}_diffractionsim.html` }));
  };
  React.useEffect(() => {
    if (!exportPayloadId || exportPayloadId !== pendingExportRef.current) return;
    const bytes = extractBytes(exportPayload);
    if (bytes.length === 0) return;
    const payload = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength ? bytes : bytes.slice();
    downloadBlob(new Blob([payload as BlobPart], { type: "text/html;charset=utf-8" }), exportFilename || "diffractionsim.html");
    pendingExportRef.current = "";
    setExportRequest(JSON.stringify({ mode: "clear" }));
  }, [exportPayload, exportPayloadId, exportFilename, setExportRequest]);

  // ---- styles ----------------------------------------------------------------
  const presetNames = offline ? Object.keys(offlinePresets || {}) : presets;
  const themedSelect = {
    fontSize: 11, height: 26, bgcolor: colors.controlBg, color: colors.text,
    "& .MuiSelect-select": { py: 0.25, fontSize: 11 },
    "& .MuiSvgIcon-root": { color: colors.textMuted },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: colors.accent },
    "&.Mui-disabled": { color: colors.textMuted, "& .MuiSelect-select": { WebkitTextFillColor: colors.textMuted } },
  };
  const themedMenuProps = {
    ...downwardMenuProps,
    PaperProps: { sx: { bgcolor: colors.controlBg, color: colors.text, border: `1px solid ${colors.border}` } },
  };
  const toggleGroup = {
    "& .MuiToggleButton-root": {
      px: 1, py: 0.25, fontSize: 11, textTransform: "none", color: colors.text, borderColor: colors.border, bgcolor: colors.controlBg,
      "&.Mui-selected": { color: colors.accent, bgcolor: colors.bg, fontWeight: 600 },
      "&:hover": { bgcolor: colors.bgAlt },
    },
  };
  const textField = {
    "& input": { fontSize: 11, py: 0.5, color: colors.text },
    "& input::placeholder": { color: colors.textMuted, opacity: 1 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: colors.accent },
    bgcolor: colors.controlBg,
  };
  const button = { ...compactButton, color: colors.accent, borderColor: colors.border, "&:hover": { borderColor: colors.accent } };
  const sectionLabel = { fontSize: 11, fontWeight: 600, color: colors.text, mr: 0.5 };
  const panelCanvas = { borderRadius: 4, border: `1px solid ${colors.border}`, display: "block" } as const;
  const nDyn = mode === "nanobeam" ? nb.nDyn : mode === "cbed" && cbed ? cbed.nDyn : 0;
  const nBeams = mode === "nanobeam" ? nbBeams.length : mode === "cbed" && cbed ? cbed.beams.length : lines.length;

  return (
    <Box ref={rootRef} sx={{ width: "100%", maxWidth: "100%" }}>
      {!crystal || !geom ? (
        <Box sx={{ p: 2, color: colors.text, bgcolor: colors.bg }}>Loading crystal…</Box>
      ) : (
    <Box sx={{ bgcolor: colors.bg, color: colors.text, p: 1.25, borderRadius: 1, border: `1px solid ${colors.border}`, width: "fit-content", maxWidth: "100%", boxSizing: "border-box" }}>
      {/* toolbar: widget controls, Copy, Export, export status, Reset */}
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mb: 1 }}>
        <Typography sx={{ fontSize: 13, fontWeight: 600, mr: 0.5, color: colors.text }}>{title}</Typography>
        <Select size="small" value={presetNames.includes(preset) ? preset : ""} displayEmpty onChange={(e) => choosePreset(e.target.value as string)}
          sx={{ ...themedSelect, minWidth: 160 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Crystal" }}>
          {!presetNames.includes(preset) && <MenuItem value="" sx={{ fontSize: 11 }}>{crystal.name}</MenuItem>}
          {presetNames.map((p) => <MenuItem key={p} value={p} sx={{ fontSize: 11 }}>{p}</MenuItem>)}
        </Select>
        <Tooltip title={offline ? "The exported page carries one beam energy" : "Beam energy; the reflection list is recomputed in Python"}>
          <Select size="small" value={energy} disabled={!!offline} onChange={(e) => setEnergy(Number(e.target.value))}
            sx={{ ...themedSelect, minWidth: 84 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Beam energy" }}>
            {ENERGIES_KEV.filter((v) => v !== energy).concat([energy]).sort((a, b) => a - b).map((v) => (
              <MenuItem key={v} value={v} sx={{ fontSize: 11 }}>{`${+v.toFixed(1)} keV`}</MenuItem>
            ))}
          </Select>
        </Tooltip>
        <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px` }}>
          <TextField size="small" placeholder={crystal.hexagonal ? "zone axis, e.g. 0 0 0 1" : "zone axis, e.g. 1 1 0"} value={zoneText}
            onChange={(e) => setZoneText(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") goZoneAxis(); }}
            inputProps={{ "aria-label": "Zone axis" }} sx={{ width: 150, ...textField }} />
          <Button size="small" variant="outlined" onClick={goZoneAxis} sx={button}>Go</Button>
        </Box>
        <Box sx={{ flex: 1 }} />
        <Button size="small" onClick={handleCopy} sx={button} title="Copy both panels as a PNG">Copy</Button>
        <Button size="small" onClick={(e) => setExportAnchor(e.currentTarget)} sx={button}
          title={exportStatus || "Export a PNG of both panels or a standalone HTML page"}>Export</Button>
        <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} {...themedMenuProps}>
          <MenuItem onClick={handleExportPng} sx={{ fontSize: 11 }}>PNG</MenuItem>
          {exportEnabled && <MenuItem onClick={handleExportHtml} sx={{ fontSize: 11 }}>HTML</MenuItem>}
        </Menu>
        {exportEnabled && exportStatus && (
          <Typography title={exportStatus} sx={{
            fontSize: 10, fontFamily: "monospace", maxWidth: 120, minWidth: 0, flexShrink: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            color: exportStatus.startsWith("Export failed") ? "#d32f2f" : colors.textMuted,
          }}>{exportStatus}</Typography>
        )}
        <Button size="small" onClick={() => setQuat([1, 0, 0, 0], true)} sx={button}
          title="Reset the orientation (c axis along the beam)">Reset</Button>
      </Stack>

      {/* panels: cell (with the Ewald view below it) and the pattern */}
      <Stack direction="row" spacing={1.5} flexWrap="wrap" useFlexGap alignItems="flex-start">
        <Box sx={{ width: Sc }}>
          <canvas ref={cellRef} aria-label="Unit cell" style={{ ...panelCanvas, width: Sc, height: Sc, touchAction: "none", cursor: ptrDrag ? "grabbing" : "grab", background: colors.bgAlt }}
            onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp} onPointerLeave={onPointerUp} />
          {showEwald && Se > 0 && (
            <canvas ref={ewaldRef} aria-label="Ewald sphere side view" style={{ ...panelCanvas, width: Sc, height: Se, marginTop: GAP, background: colors.bgAlt }} />
          )}
        </Box>
        <Box sx={{ width: S }}>
          <canvas ref={patRef} aria-label="Diffraction pattern" style={{ ...panelCanvas, width: S, height: S, touchAction: "none", cursor: ptrDrag ? "grabbing" : "grab" }}
            onPointerDown={onPatDown} onPointerMove={onPatMove} onPointerUp={onPatUp} onPointerCancel={onPatUp} onPointerLeave={onPatUp} onDoubleClick={onPatDoubleClick} />
        </Box>
      </Stack>

      {/* controls, full width under the panels */}
      <Box sx={{ width: Sc + 12 + S, maxWidth: "100%", mt: 1 }}>
        {/* crystal row */}
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
          <Typography sx={sectionLabel}>Crystal</Typography>
          {(["x", "y", "z"] as const).map((ax, i) => (
            <ToggleButtonGroup key={ax} size="small" exclusive value={null} sx={toggleGroup}>
              <ToggleButton value="-" aria-label={`Rotate about ${ax}, negative`} onClick={() => rotateLab([+(i === 0), +(i === 1), +(i === 2)], -stepDeg)}>{ax} −</ToggleButton>
              <ToggleButton value="+" aria-label={`Rotate about ${ax}, positive`} onClick={() => rotateLab([+(i === 0), +(i === 1), +(i === 2)], stepDeg)}>{ax} +</ToggleButton>
            </ToggleButtonGroup>
          ))}
          <ControlPair label="Step" colors={colors}>
            <TextField size="small" type="number" value={stepDeg} onChange={(e) => setStepDeg(Math.min(180, Math.max(0.01, Number(e.target.value) || 0.01)))}
              inputProps={{ step: 1, min: 0.01, max: 180, "aria-label": "Rotation step in degrees", style: { fontSize: 11, padding: "4px 6px", width: 42 } }} sx={textField} />
            <Typography sx={{ fontSize: 11, color: colors.text }}>°</Typography>
          </ControlPair>
          <ControlPair label="Spin" colors={colors}>
            <ToggleButtonGroup size="small" value={[spin.y ? "y" : "", spin.x ? "x" : ""]} sx={toggleGroup}>
              <ToggleButton value="y" title="Rotate slowly, horizontally (click again to stop)" onClick={() => setSpin((s) => ({ ...s, y: !s.y }))}>↔</ToggleButton>
              <ToggleButton value="x" title="Rotate slowly, vertically (click again to stop)" onClick={() => setSpin((s) => ({ ...s, x: !s.x }))}>↕</ToggleButton>
            </ToggleButtonGroup>
            <Slider value={spinSpeed || 6} min={1} max={30} step={1} size="small" onChange={(_, v) => setSpinSpeed(v as number)}
              aria-label="Rotation speed" sx={{ ...sliderStyles.small, width: 56, mx: 0.5, color: colors.accent }} />
            <Typography sx={{ fontSize: 11, fontFamily: "monospace", color: colors.text }}>{Math.round(spinSpeed || 6)}°/s</Typography>
          </ControlPair>
          <Typography sx={{ fontSize: 11, fontFamily: "monospace", ml: 1, color: colors.text }}>Zone axis {fmtIndices(zoneAxis, crystal.hexagonal)}</Typography>
          <Typography sx={{ fontSize: 11, color: colors.textMuted }}>{crystal.name} · {crystal.spacegroup || crystal.pointgroup}</Typography>
        </Stack>
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mt: 0.25 }}>
          <SwitchPair label="Cell Axes" checked={showCellAxes} onChange={setShowCellAxes} colors={colors} />
          <SwitchPair label="Axis Labels" checked={showLabels} onChange={setShowLabels} colors={colors} />
          <SwitchPair label="Polyhedra" checked={polyhedra} onChange={setPolyhedra} colors={colors} />
          <SwitchPair label="Ewald Sphere" checked={showEwald} onChange={setShowEwald} colors={colors} />
          <ControlPair label="Cells" colors={colors}>
            {[0, 1, 2].map((i) => (
              <TextField key={i} size="small" type="number" value={nCellsSafe[i]}
                onChange={(e) => { const v = [...nCellsSafe]; v[i] = Math.max(1, Math.min(6, Math.round(Number(e.target.value) || 1))); setNCells(v); }}
                inputProps={{ min: 1, max: 6, step: 1, "aria-label": `Cells along ${"abc"[i]}`, style: { fontSize: 11, padding: "3px 4px", width: 26 } }} sx={textField} />
            ))}
            <Typography sx={{ fontSize: 11, color: colors.textMuted }}>along a, b, c</Typography>
          </ControlPair>
        </Stack>

        {/* pattern row */}
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mt: 1 }}>
          <Typography sx={sectionLabel}>Pattern</Typography>
          <ToggleButtonGroup size="small" exclusive value={mode} onChange={(_, v) => v && setMode(v)} sx={toggleGroup} aria-label="Pattern mode">
            <ToggleButton value="nanobeam">Nanobeam</ToggleButton>
            <ToggleButton value="cbed">CBED</ToggleButton>
            <ToggleButton value="kikuchi">Kikuchi</ToggleButton>
          </ToggleButtonGroup>
          {mode !== "cbed" && (
            <ToggleButtonGroup size="small" exclusive value={render} onChange={(_, v) => v && setRender(v)} sx={toggleGroup} aria-label="Rendering">
              <ToggleButton value="markers">{mode === "kikuchi" ? "Lines" : "Markers"}</ToggleButton>
              {mode === "nanobeam" && <ToggleButton value="disks">Disks</ToggleButton>}
              <ToggleButton value="pixels">Pixels</ToggleButton>
            </ToggleButtonGroup>
          )}
          {mode !== "kikuchi" && <SwitchPair label="Dynamical" checked={dynamical} onChange={setDynamical} colors={colors} title="Bloch wave intensities; off: kinematical" />}
          <SwitchPair label="hkl Labels" checked={showHkl} onChange={setShowHkl} colors={colors} />
          {mode !== "kikuchi" && <SwitchPair label="Kikuchi Lines" checked={kikuchi} onChange={setKikuchi} colors={colors} />}
          {mode !== "kikuchi" && dynamical && (
            <ControlPair label="Quality" colors={colors}>
              <Select size="small" value={quality} onChange={(e) => setQuality(e.target.value as string)} sx={{ ...themedSelect, minWidth: 84 }}
                MenuProps={themedMenuProps} inputProps={{ "aria-label": "Quality" }}>
                {Object.keys(QUALITY).map((n) => <MenuItem key={n} value={n} sx={{ fontSize: 11 }}>{QUALITY_LABELS[n]}</MenuItem>)}
              </Select>
            </ControlPair>
          )}
          {mode === "kikuchi" && render === "pixels" && !kossel && !offline && (
            <Button size="small" variant="outlined" onClick={() => model.send({ type: "kossel_reference" })} sx={button}>Compute Reference</Button>
          )}
        </Stack>

        {/* physics sliders */}
        <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap sx={{ mt: 0.75 }}>
          <LabeledSlider label="Thickness" value={thickness} onChange={setThickness} min={10} max={2000} step={5} fmt={(v) => `${v.toFixed(0)} Å`} colors={colors}
            disabled={mode !== "kikuchi" ? !dynamical : render !== "pixels"} />
          {(mode === "cbed" || (mode === "nanobeam" && render === "disks")) && <LabeledSlider label="Semiangle" value={semiangle} onChange={setSemiangle} min={0.2} max={30} step={0.1} fmt={(v) => `${v.toFixed(1)} mrad`} colors={colors} />}
          {mode !== "kikuchi" && <LabeledSlider label="Precession" value={precession} onChange={setPrecession} min={0} max={3} step={0.05} fmt={(v) => (v > 0 ? `${v.toFixed(2)}°` : "off")} colors={colors} />}
          {mode !== "kikuchi" && <LabeledSlider label="Pattern Range" value={qMaxDisp} onChange={setPatternRange} min={0.2} max={crystal.k_max} step={0.05} fmt={(v) => `${v.toFixed(2)} Å⁻¹`} colors={colors} />}
          {mode === "kikuchi" && <LabeledSlider label="Field Half-Angle" value={fieldMrad} onChange={setFieldMrad} min={10} max={250} step={5} fmt={(v) => `${v.toFixed(0)} mrad`} colors={colors} />}
          {mode !== "kikuchi" && !dynamical && <LabeledSlider label="Excitation Error σ" value={sigma} onChange={setSigma} min={0.002} max={0.1} step={0.001} fmt={(v) => `${v.toFixed(3)} Å⁻¹`} colors={colors} />}
        </Stack>

        {/* display row */}
        <Stack direction="row" spacing={2} alignItems="flex-start" flexWrap="wrap" useFlexGap sx={{ mt: 0.75 }}>
          {mode === "nanobeam" && render === "markers" && (
            <>
              <LabeledSlider label="Marker Area ∝ Intensity^p" value={markerPower} onChange={setMarkerPower} min={0.1} max={1} step={0.05} fmt={(v) => `p = ${v.toFixed(2)}`} colors={colors} />
              <LabeledSlider label="Max Marker Radius" value={markerSize} onChange={setMarkerSize} min={2} max={40} step={1} fmt={(v) => `${v.toFixed(0)} px`} colors={colors} width={140} />
            </>
          )}
          {mode === "nanobeam" && render === "disks" && (
            <LabeledSlider label="Brightness ∝ Intensity^p" value={markerPower} onChange={setMarkerPower} min={0.1} max={1} step={0.05} fmt={(v) => `p = ${v.toFixed(2)}`} colors={colors} />
          )}
          {pixelMode && (
            <Stack spacing={0.5}>
              <ControlPair label="Scale" colors={colors}>
                <Select size="small" value={["linear", "power", "log"].includes(scaling) ? scaling : "linear"} onChange={(e) => setScaling(e.target.value as string)}
                  sx={{ ...themedSelect, minWidth: 84 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Intensity scaling" }}>
                  <MenuItem value="linear" sx={{ fontSize: 11 }}>Linear</MenuItem>
                  <MenuItem value="power" sx={{ fontSize: 11 }}>Power</MenuItem>
                  <MenuItem value="log" sx={{ fontSize: 11 }}>Log</MenuItem>
                </Select>
              </ControlPair>
              <ControlPair label="Color" colors={colors}>
                <Select size="small" value={cmapNames.includes(cmap) ? cmap : cmapNames[0]} onChange={(e) => setCmap(e.target.value as string)}
                  sx={{ ...themedSelect, minWidth: 84 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Colormap" }}>
                  {cmapNames.map((n) => <MenuItem key={n} value={n} sx={{ fontSize: 11 }}>{n}</MenuItem>)}
                </Select>
              </ControlPair>
            </Stack>
          )}
          {pixelMode && scaling === "power" && (
            <LabeledSlider label="Exponent" value={power} onChange={setPower} min={0.1} max={1} step={0.05} fmt={(v) => v.toFixed(2)} colors={colors} width={110} />
          )}
          {display && (
            <Histogram data={display.data} vminPct={vminPct} vmaxPct={vmaxPct} onRangeChange={(a, b) => { setVminPct(a); setVmaxPct(b); }}
              lo={display.lo} hi={display.hi} colors={colors} />
          )}
        </Stack>

        <Typography sx={{ fontSize: 10.5, color: colors.textMuted, mt: 0.75 }}>
          {+energy.toFixed(1)} keV · λ = {(crystal.wavelength * 100).toFixed(3)} pm · {nBeams} {mode === "kikuchi" ? "lines" : "beams"}
          {mode !== "kikuchi" && dynamical ? ` · ${nDyn} Bloch beams (|s| < ${SG_MAX} Å⁻¹${crystal.absorptive ? ", absorptive" : ""}), thin-slab intensities for the rest` : ""}
          {mode !== "kikuchi" && precession > 0
            ? ` · precession ${precession.toFixed(2)}° over ${mode === "cbed" ? (cbed ? cbed.nodes.length : 0) : precNodes.length} ring nodes`
            : ""}
          {mode === "cbed" ? " · disks summed incoherently where they overlap" : ""}
          {status ? ` · ${status}` : ""}
        </Typography>
        <Typography sx={{ fontSize: 10.5, color: colors.textMuted }}>Drag the cell (near face follows) or the pattern (tilt map follows) · shift-drag or two fingers twist about the beam · double-click a disk for its two-beam condition, or empty space to put the Laue circle centre there · buttons rotate about the screen axes</Typography>
      </Box>
    </Box>
      )}
    </Box>
  );
}

function labelBeams(ctx: CanvasRenderingContext2D, f: Frame, beams: Reflection[], inten: Float64Array) {
  let iMax = 0;
  for (let i = 0; i < beams.length; i++) iMax = Math.max(iMax, inten[i]);
  if (iMax <= 0) return;
  ctx.font = `${Math.max(9, Math.round(f.size / 38))}px sans-serif`;
  ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  ctx.fillStyle = "#ffd54f"; // on the colormapped image
  let count = 0;
  for (let i = 0; i < beams.length && count < 40; i++) {
    if (inten[i] / iMax < 0.08) continue;
    const [x, y] = toPx(f, beams[i].g[0], beams[i].g[1]);
    ctx.fillText(hklText(beams[i].hkl), x, y - 6);
    count++;
  }
}

export default { render: createRender(DiffSim) };
