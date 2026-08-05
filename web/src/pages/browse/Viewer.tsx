import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { breakpoints, colors, fontSizes, radii } from "../../theme";
import { useGlobalSmooth, imageRenderingFor } from "../../hooks/useGlobalSmooth";
import {
  ALPHA_MRAD, COLORMAP_OPTIONS, DETECTOR_MODES, DET_SHAPES,
  fetchBfGeometry, fetchCBED, fetchCBEDRoi, fetchVirtualImage,
  fetchVirtualImageBufferGpu, fetchVirtualImageShape, fileKey, formatMrad,
  type BfGeometry, type BrowseDtype, type ColormapName, type DetBin, type DetShape,
  type DetectorMode, type MasterFile, type MasterLoadStatus, type RawData,
  type Session, type Set5D, type ShapeParams,
} from "./types";
import { allocateSlot, getGPUColormapEngine } from "../../utils/gpu-colormap";
import { applyColormap, COLORMAPS } from "../../utils/colormaps";
import { bucketize, percentileClip, percentileClipMasked } from "../../utils/stats";
import { pickScaleBarPx, scaleBarLabel as formatScaleBarLabel, SCALE_BAR_TARGET_PX } from "../../utils/scalebar";
import { fft2dMagnitudeGPU } from "../../utils/webgpu-fft";
import { fft2dMagnitude, prepareFftInput } from "../../utils/fft";
import { sampleLineProfile } from "../../utils/lineProfile";

type IntensityScale = "linear" | "log" | "sqrt" | "power";

const DETECTOR_MODE_LABELS: Record<DetectorMode, string> = {
  BF: "BF",
  ADF: "ADF",
  DF: "DF",
  CoMmag: "CoM mag",
  CoMy: "CoM row",
  CoMx: "CoM col",
  iCoM: "iCoM",
};

function detectorModeLabel(mode: DetectorMode): string {
  return DETECTOR_MODE_LABELS[mode] ?? mode;
}

const VIRTUAL_DETECTOR_FETCH_THROTTLE_MS = 12;
const SCAN_POINT_FETCH_THROTTLE_MS = 12;

// Native <select> on Mac Chrome can inherit OS dark-mode widget rendering
// unless the control opts into a light color scheme. Keep one style object so
// toolbar and panel dropdowns stay visually identical.
const SELECT_CONTROL_STYLE = {
  fontSize: fontSizes.sm,
  minHeight: 44,
  padding: "9px 10px",
  borderRadius: radii.md,
  border: `1px solid ${colors.border.default}`,
  backgroundColor: colors.bg.page,
  color: colors.text.secondary,
  colorScheme: "light" as const,
};

/** Apply the user-selected intensity transform in-place into a fresh
 *  Float32Array so the source buffer stays untouched (it feeds histograms
 *  and other panels). `linear` returns the input as-is. Mirrors
 *  `_apply_scale_mode` in show4dstem.py:1382. */
function applyScale(data: Float32Array, scale: IntensityScale, powerExp: number = 0.5): Float32Array {
  if (scale === "linear") return data;
  const out = new Float32Array(data.length);
  if (scale === "log") {
    for (let i = 0; i < data.length; i++) out[i] = Math.log1p(Math.max(0, data[i]));
  } else if (scale === "sqrt") {
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      out[i] = v > 0 ? Math.sqrt(v) : 0;
    }
  } else {
    // power: x^p for x >= 0, clamp negative to 0 (matches Show4DSTEM)
    const p = powerExp;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      out[i] = v > 0 ? Math.pow(v, p) : 0;
    }
  }
  return out;
}

// First-toggle-on lazy WebGPU FFT warm. The previous design ran a 16×16
// dummy FFT in a mount-time effect so the shader compile cost (50-500 ms)
// landed during page render. That spent GPU on users who never toggle FFT
// AND re-paid the cost on every Viewer remount (e.g. navigate away from
// /browse and back). Move it to the toggle path: first time the user
// actually opens FFT we await this helper, absorbing the compile inside
// the toggle action where a brief stall is tolerable; subsequent toggles
// resolve immediately from the memoized promise.
let _fftWarmPromise: Promise<void> | null = null;
function ensureFFTWarmed(): Promise<void> {
  if (_fftWarmPromise) return _fftWarmPromise;
  _fftWarmPromise = (async () => {
    const warm = new Float32Array(16 * 16);
    try { await fft2dMagnitudeGPU(warm, 16, 16); } catch { /* best-effort */ }
  })();
  return _fftWarmPromise;
}

interface Props {
  session: Session;
  file: MasterFile;
  browseDtype?: BrowseDtype;
  singleDetBin?: DetBin;
  mode: DetectorMode;
  setMode: (m: DetectorMode) => void;
  cmapImage: ColormapName;
  setCmapImage: (c: ColormapName) => void;
  cmapDp: ColormapName;
  setCmapDp: (c: ColormapName) => void;
  scanPos: { x: number; y: number };
  setScanPos: (p: { x: number; y: number }) => void;
  ringInner: number;
  setRingInner: (v: number) => void;
  ringOuter: number;
  setRingOuter: (v: number) => void;
  /** Active detector-mask shape on the DP. ``"circle"`` keeps the legacy
   *  α-unit ring behavior; the others (square / rect / annulus / point)
   *  drive the new /realspace-shape endpoint with detector-pixel params. */
  dpShape: DetShape;
  setDpShape: (s: DetShape) => void;
  /** Detector-pixel parameters for the active shape. Only the fields
   *  relevant to the current shape are read by the backend. Lifted from
   *  Browse.tsx so the shape persists across master switches. */
  shapeParams: ShapeParams;
  setShapeParams: (p: ShapeParams | ((prev: ShapeParams) => ShapeParams)) => void;
  showRing: boolean;
  logScale: boolean;
  clipLo: number;
  setClipLo: (v: number) => void;
  clipHi: number;
  setClipHi: (v: number) => void;
  dpClipLo: number;
  setDpClipLo: (v: number) => void;
  dpClipHi: number;
  setDpClipHi: (v: number) => void;
  /** Compact mode collapses histograms to a 16 px-tall colored gradient
   *  strip and shrinks toolbar padding by ~50 %. The lo/hi clip handles
   *  remain draggable on the strip so percentile control is preserved. */
  compact?: boolean;
  /** Active 5D-STEM set, or null when the viewer is in single-master mode.
   *  When non-null, the file prop above is the master at
   *  ``set5D.files[set5D.activeIdx]``. The scrubber strip + arrow-key
   *  navigation are wired through ``onSetActiveIdx``. */
  set5D?: Set5D | null;
  /** Step the 5D set's active index. Called from the scrubber slider, the
   *  thumbnail strip, and ``Alt+←/→`` while the viewer has focus. */
  onSetActiveIdx?: (idx: number) => void;
  /** Real-space rectangular ROI (in scan-pixel coords). Non-null →
   *  DP panel shows the SUM of CBEDs across the rectangle. The user
   *  drags with Alt held to set/extend; Esc clears. */
  realRoi?: { row0: number; col0: number; row1: number; col1: number } | null;
  setRealRoi?: (r: { row0: number; col0: number; row1: number; col1: number } | null) => void;
  /** FFT-of-virtual-image panel toggle. When true, the real-space column
   *  stacks an FFT magnitude panel below the image. */
  fftOn?: boolean;
  setFftOn?: (v: boolean) => void;
  /** Hann window toggle for FFT input. Show4DSTEM `fft_window` parity:
   *  default ON (Hann). Off = raw centered+detrended, no window. */
  fftWindow?: boolean;
  setFftWindow?: (v: boolean) => void;
  /** Intensity scaling per panel. Independent so the user can pick
   *  log for DP (default) and linear for the virtual image (default). */
  imageScale?: IntensityScale;
  setImageScale?: (s: IntensityScale) => void;
  dpScale?: IntensityScale;
  setDpScale?: (s: IntensityScale) => void;
  /** Power-exp for `power` scale mode. Show4DSTEM dp_power_exp/vi_power_exp
   *  default 0.5 (sqrt-like). Range 0.1-2.0. */
  imagePowerExp?: number;
  setImagePowerExp?: (v: number) => void;
  dpPowerExp?: number;
  setDpPowerExp?: (v: number) => void;
  /** Mask the central 3×3 of the DP from percentile-clip stats so the
   *  central beam doesn't crush the histogram. Show4DSTEM `mask_dc`. */
  maskDC?: boolean;
  setMaskDC?: (v: boolean) => void;
  /** Line profile on the virtual image. Show4DSTEM/Show2D port:
   *  bilinear sampling between two scan-pixel endpoints, optional width
   *  averaging. Null = no profile drawn. */
  profileLine?: { row0: number; col0: number; row1: number; col1: number } | null;
  setProfileLine?: (l: { row0: number; col0: number; row1: number; col1: number } | null) => void;
  profileWidth?: number;
  setProfileWidth?: (v: number) => void;
}

const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";

/** Two-significant-figure byte size for the error banner. Mirrors the
 *  server-side `_format_size` so a dataset reading "8.3 GB" in the file
 *  tree matches the same number in the missing-chunks readout. */
function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  const units: [string, number][] = [
    ["TB", 1 << 40], ["GB", 1 << 30], ["MB", 1 << 20], ["KB", 1 << 10],
  ];
  for (const [label, scale] of units) {
    if (n >= scale) return `${(n / scale).toFixed(1)} ${label}`;
  }
  return `${Math.round(n)} B`;
}

/** Detailed missing-chunks readout shown inside the Viewer error overlay
 *  when `file.load_status.missing_count > 0`. Renders the human reason,
 *  present-byte total, and the first 5 missing filenames inline (with a
 *  "show details" toggle for the rest). */
function MissingChunksDetail({ status }: { status: MasterLoadStatus }) {
  const [expanded, setExpanded] = useState(false);
  const present = formatBytes(status.present_bytes);
  // Note: status.expected_bytes_estimate is the UNCOMPRESSED extent
  // (~100-400× the on-disk LZ4+bitshuffle total), so we don't render
  // "present of expected" — that ratio is misleading. The present-bytes
  // number alone tells the user how much real data they have on disk.
  const allChunks = status.missing_chunks;
  const inline = allChunks.slice(0, 5);
  const rest = allChunks.slice(5);
  const baseTextSx = {
    fontSize: fontSizes.xs, color: colors.overlay.onImage,
    fontFamily: MONO, opacity: 0.92, textAlign: "center" as const,
    textShadow: "0 1px 2px rgba(0,0,0,0.6)",
  };
  return (
    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center",
               gap: 0.5, maxWidth: "85%" }}>
      <Box sx={{ ...baseTextSx, fontWeight: 700 }}>
        {status.reason}
      </Box>
      <Box sx={baseTextSx}>
        {present} present on disk
      </Box>
      {inline.length > 0 && (
        <Box sx={{ ...baseTextSx, mt: 0.5,
                   bgcolor: "rgba(0,0,0,0.45)", px: 1, py: 0.5,
                   borderRadius: radii.sm, opacity: 0.95 }}>
          {inline.map((c, i) => (
            <Box key={i} component="span" sx={{ display: "block" }}>{c}</Box>
          ))}
          {rest.length > 0 && (
            <Box
              component="button"
              onClick={() => setExpanded((v) => !v)}
              sx={{ mt: 0.5, fontSize: fontSizes.xs, fontFamily: MONO,
                    cursor: "pointer", border: "none", background: "transparent",
                    color: colors.overlay.onImage, textDecoration: "underline",
                    opacity: 0.9 }}
            >
              {expanded ? "hide details" : `show ${rest.length} more`}
            </Box>
          )}
          {expanded && rest.map((c, i) => (
            <Box key={`r${i}`} component="span" sx={{ display: "block" }}>{c}</Box>
          ))}
        </Box>
      )}
      <Box sx={{ ...baseTextSx, opacity: 0.75, mt: 0.5 }}>
        try a different master in the tree
      </Box>
    </Box>
  );
}

// Slot indices reserved for Browse — separate so PanelViewer/Trials slots
// don't get clobbered when the user navigates between pages.
const SLOT_REAL = allocateSlot();
const SLOT_DP = allocateSlot();
const SLOT_FFT = allocateSlot();

// Module-level CBED cache. Persists across re-renders so re-visiting a scan
// position we've already fetched paints synchronously (zero network latency,
// zero arraybuffer decode). LRU evicts at 256 entries — at typical 96×96
// float32, that's ~9 MB; at 192×192, ~38 MB. Either way comfortably under
// any browser's heap budget. Keys include the file identity so different
// masters get independent cache lines (no manual reset on file change).
const CBED_CACHE_MAX = 256;
const CBED_CACHE = new Map<string, RawData>();

function cbedKey(s: Session, f: MasterFile, ix: number, iy: number, detBin: DetBin, dtype: BrowseDtype): string {
  return `${fileKey(s, f)}|b${detBin}|${dtype}|${ix},${iy}`;
}

/** Cache-fronted CBED fetch. Re-insertion on hit moves the key to the end
 *  of the Map so the LRU eviction picks the truly oldest entry. */
async function fetchCBEDCached(
  s: Session, f: MasterFile, ix: number, iy: number, signal?: AbortSignal,
  detBin: DetBin = 1, dtype: BrowseDtype = "uint8",
): Promise<RawData | null> {
  const k = cbedKey(s, f, ix, iy, detBin, dtype);
  const hit = CBED_CACHE.get(k);
  if (hit) {
    CBED_CACHE.delete(k);
    CBED_CACHE.set(k, hit);
    return hit;
  }
  const r = await fetchCBED(s, f, ix, iy, signal, detBin, dtype);
  if (r) {
    CBED_CACHE.set(k, r);
    if (CBED_CACHE.size > CBED_CACHE_MAX) {
      const oldest = CBED_CACHE.keys().next().value;
      if (oldest !== undefined) CBED_CACHE.delete(oldest);
    }
  }
  return r;
}

/** Color a histogram strip's RGBA gradient from the active colormap LUT. */
function colormapGradient(name: string, stops = 8): string {
  const lut = COLORMAPS[name] || COLORMAPS["gray"];
  const parts: string[] = [];
  for (let i = 0; i <= stops; i++) {
    const idx = Math.min(255, Math.floor((i / stops) * 255)) * 3;
    const r = lut[idx], g = lut[idx + 1], b = lut[idx + 2];
    parts.push(`rgb(${r},${g},${b}) ${(i / stops) * 100}%`);
  }
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}

// Tiny canvas-based 1D profile plot. Mirrors Show2D's profile-canvas
// (white-on-dark line graph, no axes). Auto-scales y to data range.
function ProfilePlot({ values }: { values: Float32Array }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || values.length < 2) return;
    const W = canvas.clientWidth || 240;
    const H = 56;
    canvas.width = Math.round(W * (window.devicePixelRatio || 1));
    canvas.height = Math.round(H * (window.devicePixelRatio || 1));
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(canvas.width / W, canvas.height / H);
    ctx.clearRect(0, 0, W, H);
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      if (v < mn) mn = v; if (v > mx) mx = v;
    }
    if (!Number.isFinite(mn) || !Number.isFinite(mx) || mn === mx) return;
    ctx.fillStyle = colors.bg.subtle;
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "#2dd4bf";
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = (i / (values.length - 1)) * W;
      const y = H - ((values[i] - mn) / (mx - mn)) * (H - 4) - 2;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [values]);
  return (
    <canvas ref={ref}
            style={{ width: "100%", height: 56, display: "block",
                     borderRadius: radii.sm, border: `1px solid ${colors.border.default}` }}
            data-testid="profile-plot" />
  );
}

/** Per-panel sub-toolbar. Sits between the canvas and its histogram so all
 *  knobs that only affect ONE panel (cmap, scale, power-exp, extras like
 *  mask-DC or Hann) live with the panel they control. Show4DSTEM parity:
 *  every panel has its own little control row, no global colormap dropdown.
 *  The top toolbar keeps only truly global controls (modes, shapes, tools). */
function PanelToolbar({
  cmap, setCmap, scale, setScale, powerExp, setPowerExp,
  extras, compact,
}: {
  cmap: ColormapName;
  setCmap: (c: ColormapName) => void;
  scale: IntensityScale;
  setScale: (s: IntensityScale) => void;
  powerExp: number;
  setPowerExp?: (v: number) => void;
  /** Optional panel-specific extras (mask-DC for DP, Hann for FFT, …). */
  extras?: React.ReactNode;
  compact?: boolean;
}) {
  const smooth = useGlobalSmooth();
  return (
    <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap",
               gap: compact ? 0.5 : 0.75, mt: 0.25,
                 [`@media (max-width: ${breakpoints.tablet}px)`]: {
                   "& button, & [role='button']": {
                   minWidth: 44,
                   minHeight: 44,
                   display: "inline-flex",
                   alignItems: "center",
                   justifyContent: "center",
                 },
               } }}>
      <select
        value={cmap}
        onChange={(e) => setCmap(e.target.value as ColormapName)}
        title="Colormap"
        style={SELECT_CONTROL_STYLE}
      >
        {COLORMAP_OPTIONS.map((c) => <option key={c} value={c}>{c}</option>)}
      </select>
      <select
        value={scale}
        onChange={(e) => setScale(e.target.value as IntensityScale)}
        title="Intensity scale"
        style={SELECT_CONTROL_STYLE}
      >
        <option value="linear">linear</option>
        <option value="log">log</option>
        <option value="sqrt">sqrt</option>
        <option value="power">power</option>
      </select>
      {scale === "power" && setPowerExp && (
        <input
          type="number"
          min={0.1} max={2.0} step={0.05}
          value={powerExp}
          onChange={(e) => setPowerExp(parseFloat(e.target.value) || 0.5)}
          title="Power exponent (Show4DSTEM *_power_exp default 0.5)"
          style={{ ...SELECT_CONTROL_STYLE, width: 56 }}
        />
      )}
      {extras}
    </Box>
  );
}

function Histogram({
  data, cmap, lo, hi, setLo, setHi, label, compact, formatValue,
}: {
  data: Float32Array | null;
  cmap: ColormapName;
  lo: number; hi: number;
  setLo: (v: number) => void; setHi: (v: number) => void;
  label: string;
  compact?: boolean;
  /** Optional formatter: takes a 0..1 percentile and returns a human label
   *  for the vmin/vmax readout. The DP histogram passes a formatter that
   *  inverts log1p so the displayed numbers are RAW counts, not log values. */
  formatValue?: (pct: number) => string;
}) {
  const N = 96;
  // Keep the LAST GOOD bins so a momentary `data === null` (e.g. between
  // CBED fetches during scan drag) doesn't flash the histogram bars to
  // zero. We only overwrite when new data with a real range arrives.
  const lastBinsRef = useRef<number[] | null>(null);
  const bins = useMemo(() => {
    if (!data || data.length === 0) {
      return lastBinsRef.current || new Array<number>(N).fill(0);
    }
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!(max > min)) return lastBinsRef.current || new Array<number>(N).fill(0);
    const next = bucketize(data, min, max, N, true);
    lastBinsRef.current = Array.from(next);
    return lastBinsRef.current;
  }, [data]);
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<"lo" | "hi" | null>(null);

  const onPtMove = useCallback((e: MouseEvent) => {
    if (!dragging || !trackRef.current) return;
    const r = trackRef.current.getBoundingClientRect();
    const v = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    if (dragging === "lo") setLo(Math.min(v, hi - 0.02));
    if (dragging === "hi") setHi(Math.max(v, lo + 0.02));
  }, [dragging, lo, hi, setLo, setHi]);

  useEffect(() => {
    if (!dragging) return;
    const m = (e: MouseEvent) => onPtMove(e);
    const u = () => setDragging(null);
    window.addEventListener("mousemove", m);
    window.addEventListener("mouseup", u);
    return () => { window.removeEventListener("mousemove", m); window.removeEventListener("mouseup", u); };
  }, [dragging, onPtMove]);

  const grad = useMemo(() => colormapGradient(cmap), [cmap]);
  const presets: [number, number][] = [[0.005, 0.995], [0.02, 0.98], [0.05, 0.95], [0.10, 0.90]];

  // Compact strip: 16 px-tall colormap gradient with two thin draggable
  // thumbs. No bars, no preset chips, no min/max readout — those live
  // behind the "comfy" mode toggle. Drag handles remain functional so
  // percentile control is preserved at every screen size.
  if (compact) {
    return (
      <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25, mt: 0.5 }}>
        <Box ref={trackRef}
             sx={{ position: "relative", height: 16, userSelect: "none",
                   borderRadius: radii.sm, overflow: "hidden",
                   border: `1px solid ${colors.border.default}` }}>
          {/* Underlay: gray bin silhouette so the strip still hints at the
              data distribution. Painted at 35 % opacity so the gradient
              dominates the visual. */}
          <svg viewBox="0 0 100 16" preserveAspectRatio="none"
               style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
            {bins.map((y: number, i: number) => (
              <rect key={i} x={(i / N) * 100} y={16 - y * 14} width={100 / N} height={y * 14}
                    fill={colors.text.muted} opacity="0.30" />
            ))}
          </svg>
          {/* Active range: the colormap gradient itself, clipped to lo/hi. */}
          <Box sx={{ position: "absolute", top: 0, height: "100%", background: grad,
                     left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }} />
          {(["lo", "hi"] as const).map((side) => {
            const left = (side === "lo" ? lo : hi) * 100;
            return (
              <Box
                key={side}
                onMouseDown={(e) => { e.preventDefault(); setDragging(side); }}
                sx={{ position: "absolute", top: 0, left: `${left}%`, transform: "translateX(-50%)",
                      width: 8, height: "100%", cursor: "ew-resize",
                      "&::before": {
                        content: '""', position: "absolute", left: "50%", top: 0,
                        transform: "translateX(-50%)", width: 2, height: "100%",
                        background: colors.text.primary, opacity: 0.85,
                      } }}
              />
            );
          })}
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, fontSize: fontSizes.xs,
                   color: colors.text.muted, fontFamily: MONO }}>
          <span>{label}</span>
          <Box sx={{ flex: 1 }} />
          <span>{(lo * 100).toFixed(0)}%–{(hi * 100).toFixed(0)}%</span>
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, mt: 1 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Typography sx={{ fontSize: fontSizes.xs, color: colors.text.muted }}>
          {label}
        </Typography>
        <Box sx={{ display: "flex", gap: 0.5, ml: "auto" }}>
          {presets.map(([a, b]) => {
            const on = Math.abs(lo - a) < 0.001 && Math.abs(hi - b) < 0.001;
            return (
              <Box
                key={a}
                component="button"
                onClick={() => { setLo(a); setHi(b); }}
                sx={{ fontSize: fontSizes.xs, px: 0.75, py: 0.25, cursor: "pointer",
                      border: `1px solid ${on ? colors.interactive.active : colors.border.default}`,
                      bgcolor: on ? colors.interactive.active : colors.bg.page,
                      color: on ? colors.text.white : colors.text.secondary,
                      borderRadius: radii.sm, fontFamily: MONO,
                      fontWeight: on ? 700 : 500,
                      opacity: on ? 1 : 0.72,
                      [`@media (max-width: ${breakpoints.tablet}px)`]: {
                        minWidth: 44,
                        minHeight: 44,
                        display: "inline-flex",
                        alignItems: "center",
                        justifyContent: "center",
                      },
                      "&:hover": { bgcolor: on ? colors.interactive.hover : colors.bg.hover, opacity: 1 } }}
              >
                {(a * 100).toFixed(a < 0.01 ? 1 : 0)}–{(b * 100).toFixed(b > 0.99 ? 1 : 0)}%
              </Box>
            );
          })}
        </Box>
      </Box>
      <Box ref={trackRef} sx={{ position: "relative", height: 36, userSelect: "none" }}>
        <svg viewBox="0 0 100 32" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
          {bins.map((y: number, i: number) => (
            <rect key={i} x={(i / N) * 100} y={32 - y * 30} width={100 / N} height={y * 30}
                  fill={colors.text.muted} opacity="0.55" />
          ))}
          <rect x={lo * 100} y="0" width={(hi - lo) * 100} height="32" fill={colors.text.dark} opacity="0.06" />
        </svg>
        <Box sx={{ position: "absolute", top: 28, height: 4, background: grad,
                   left: `${lo * 100}%`, width: `${(hi - lo) * 100}%`, borderRadius: 1 }} />
        {(["lo", "hi"] as const).map((side) => {
          const left = (side === "lo" ? lo : hi) * 100;
          return (
            <Box
              key={side}
              onMouseDown={(e) => { e.preventDefault(); setDragging(side); }}
              sx={{ position: "absolute", top: 0, left: `${left}%`, transform: "translateX(-50%)",
                    width: 10, height: 36, cursor: "ew-resize",
                    "&::before": {
                      content: '""', position: "absolute", left: "50%", top: 0,
                      transform: "translateX(-50%)", width: 2, height: "100%",
                      background: colors.text.primary, opacity: 0.85,
                    } }}
            />
          );
        })}
      </Box>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, fontSize: fontSizes.xs,
                 color: colors.text.tertiary, fontFamily: MONO }}>
        <span>min <b>{(lo * 100).toFixed(1)}%</b>{formatValue ? ` (${formatValue(lo)})` : ""}</span>
        <span>max <b>{(hi * 100).toFixed(1)}%</b>{formatValue ? ` (${formatValue(hi)})` : ""}</span>
        <Box sx={{ flex: 1 }} />
        <Box component="button" onClick={() => { setLo(0.02); setHi(0.98); }}
             sx={{ fontSize: fontSizes.xs, px: 0.75, py: 0.25, cursor: "pointer",
                   border: `1px solid ${colors.border.default}`, bgcolor: colors.bg.page,
                   color: colors.text.secondary, borderRadius: radii.sm,
                   [`@media (max-width: ${breakpoints.tablet}px)`]: {
                     minWidth: 44,
                     minHeight: 44,
                     display: "inline-flex",
                     alignItems: "center",
                     justifyContent: "center",
                   },
                   "&:hover": { bgcolor: colors.bg.hover } }}>
          reset
        </Box>
      </Box>
    </Box>
  );
}

/** "Nice number" series for px-only scale bars when no physical
 *  calibration is plumbed through. Mirrors the nm series in
 *  `utils/scalebar.ts` but in raw pixel counts so we always get a
 *  legible bar (e.g. "32 px" for a 192-px detector at 2× zoom). */
const PX_NICE_SERIES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];

function pickPxNiceLength(imagePx: number, displayPx: number): { nicePx: number; barPx: number } | null {
  if (!imagePx || !displayPx) return null;
  const screenPxPerImagePx = displayPx / imagePx;
  if (!Number.isFinite(screenPxPerImagePx) || screenPxPerImagePx <= 0) return null;
  let bestNice = PX_NICE_SERIES[0];
  let bestDelta = Infinity;
  for (const v of PX_NICE_SERIES) {
    const d = Math.abs(v * screenPxPerImagePx - SCALE_BAR_TARGET_PX);
    if (d < bestDelta) { bestDelta = d; bestNice = v; }
  }
  return { nicePx: bestNice, barPx: bestNice * screenPxPerImagePx };
}

/** "Nice number" series for diffraction (mrad) scale bars. Picks a round
 *  mrad length whose on-screen pixel width lands near SCALE_BAR_TARGET_PX. */
const MRAD_NICE_SERIES = [
  0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500,
];

function pickMradNiceLength(
  imagePx: number, mradPerPx: number, displayPx: number,
): { mrad: number; barPx: number } | null {
  if (!imagePx || !mradPerPx || !displayPx) return null;
  const pxPerMrad = displayPx / (imagePx * mradPerPx);
  if (!Number.isFinite(pxPerMrad) || pxPerMrad <= 0) return null;
  let best = MRAD_NICE_SERIES[0];
  let bestDelta = Infinity;
  for (const v of MRAD_NICE_SERIES) {
    const d = Math.abs(v * pxPerMrad - SCALE_BAR_TARGET_PX);
    if (d < bestDelta) { bestDelta = d; best = v; }
  }
  return { mrad: best, barPx: best * pxPerMrad };
}

/** Pick a scale bar for a canvas given image dims, optional physical
 *  calibration (Å/px), zoom factor, and the on-screen pixel size of the
 *  canvas. Returns the on-screen bar pixel width plus a formatted label.
 *
 *  When `stepA` is null the bar falls back to "N px" — better than
 *  showing fake nm. When everything is known the bar tracks zoom: bar
 *  screen length stays in 60-120 px and the label updates as the user
 *  zooms in/out.
 *
 *  Returns null when calibration AND display size are both unavailable
 *  (e.g. canvas size 0 before first paint) so the caller can hide the bar. */
function pickScaleBarForCanvas(opts: {
  imagePx: number;
  /** Real-space pixel pitch in Å (only honored when `unit === "real"`). */
  stepA?: number | null;
  /** Detector pixel pitch in mrad (only honored when `unit === "k"`). */
  stepMrad?: number | null;
  unit?: "real" | "k";
  zoom: number;
  displayPx: number;
}): { barPx: number; label: string } | null {
  const { imagePx, stepA, stepMrad, zoom, displayPx } = opts;
  if (!imagePx || !displayPx) return null;
  const effectiveDisplayPx = displayPx * zoom;
  const unit = opts.unit ?? "real";
  if (unit === "real" && stepA && stepA > 0) {
    const sb = pickScaleBarPx(imagePx, stepA, effectiveDisplayPx);
    if (sb) return { barPx: sb.barPx, label: formatScaleBarLabel(sb.lengthNm) };
  }
  if (unit === "k" && stepMrad && stepMrad > 0) {
    const sb = pickMradNiceLength(imagePx, stepMrad, effectiveDisplayPx);
    if (sb) return { barPx: sb.barPx, label: formatMrad(sb.mrad) };
  }
  const px = pickPxNiceLength(imagePx, effectiveDisplayPx);
  if (!px) return null;
  return { barPx: px.barPx, label: `${px.nicePx} px` };
}

/** Pan/zoom transform state. `scale` is a multiplier; `tx`/`ty` are pixel
 *  offsets in the canvas-frame's screen space (0..frameSize). The transform
 *  is applied as `translate(tx, ty) scale(scale)` from the frame's top-left
 *  origin, so {tx:0, ty:0, scale:1} is fit-to-canvas. */
type Transform = { tx: number; ty: number; scale: number };
const IDENTITY: Transform = { tx: 0, ty: 0, scale: 1 };
const ZOOM_MIN = 1;
const ZOOM_MAX = 32;

/** Clamp pan offsets so the image edge can't be dragged past the canvas
 *  center. With scale s and frame size W, the image extends from tx to
 *  tx + s*W. Constraint: image's right edge >= W/2 → tx >= W/2 - s*W;
 *  image's left edge <= W/2 → tx <= W/2. Symmetric for ty. */
function clampTransform(t: Transform, frameW: number, frameH: number): Transform {
  const s = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, t.scale));
  if (s <= 1) return { tx: 0, ty: 0, scale: 1 };
  const txMin = frameW / 2 - s * frameW;
  const txMax = frameW / 2;
  const tyMin = frameH / 2 - s * frameH;
  const tyMax = frameH / 2;
  return {
    tx: Math.max(txMin, Math.min(txMax, t.tx)),
    ty: Math.max(tyMin, Math.min(tyMax, t.ty)),
    scale: s,
  };
}

function CanvasCard({
  innerRef, onMouseDown, onMouseMove, onMouseLeave, onDoubleClick, cursor, label, corner, hoverChip, scaleBarPx, scaleBarLabel,
  overlay, scaleBarTone = "light", busy, skeleton, error, errorDetail, transform, focused, children,
}: {
  innerRef: React.RefObject<HTMLDivElement | null>;
  onMouseDown?: (e: React.MouseEvent) => void;
  onMouseMove?: (e: React.MouseEvent) => void;
  onMouseLeave?: (e: React.MouseEvent) => void;
  onDoubleClick?: (e: React.MouseEvent) => void;
  cursor: string;
  label: string;
  corner: React.ReactNode;
  hoverChip: string;
  scaleBarPx: number;
  scaleBarLabel: string;
  overlay?: React.ReactNode;
  scaleBarTone?: "light" | "dark";
  busy?: boolean;
  skeleton?: boolean;
  error?: string | null;
  /** Optional rich-content block rendered under the headline pill when the
   *  error has structured detail (missing chunks, present-bytes, etc.).
   *  When absent we fall back to the generic "missing chunk files" sentence. */
  errorDetail?: React.ReactNode;
  /** Pan/zoom transform applied to the image + overlay layers (NOT the chrome
   *  layers). A single transform is shared so markers stay locked to image
   *  coordinates. Defaults to identity when not provided. */
  transform?: Transform;
  /** When true, draws a subtle focus ring around the frame so the user knows
   *  which canvas owns keyboard focus (arrow keys move the corresponding
   *  scan position vs aperture). */
  focused?: boolean;
  children: React.ReactNode;
}) {
  const smooth = useGlobalSmooth();
  const t = transform || IDENTITY;
  const tfStyle = {
    transform: `translate(${t.tx}px, ${t.ty}px) scale(${t.scale})`,
    transformOrigin: "top left" as const,
    willChange: "transform" as const,
  };
  return (
    <Box
      ref={innerRef as unknown as React.Ref<HTMLDivElement>}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      onDoubleClick={onDoubleClick}
      sx={{ position: "relative", aspectRatio: "1 / 1", width: "100%",
            bgcolor: colors.overlay.onImageBg,
            // Square corners on the canvas frame itself — surrounding chrome
            // (toolbar, histogram strips) keeps its own radii. The active
            // canvas gets a subtle focus ring so the user always knows which
            // canvas owns arrow keys.
            borderRadius: 0, overflow: "hidden",
            outline: focused ? `2px solid ${colors.interactive.border}` : "none",
            outlineOffset: focused ? "-2px" : 0,
            cursor, touchAction: "none" }}
    >
      <Box sx={{ position: "absolute", inset: 0, ...tfStyle,
                 "& canvas": { width: "100%", height: "100%", display: "block",
                               // preserve scan aspect: a non-square / partial dataset (e.g. only
                               // the top rows decoded) letterboxes instead of stretching to garbage.
                               objectFit: "contain",
                               imageRendering: imageRenderingFor(smooth) } }}>
        {children}
      </Box>
      {overlay && (
        <Box sx={{ position: "absolute", inset: 0, pointerEvents: "none", ...tfStyle }}>
          {overlay}
        </Box>
      )}
      {/* Chrome layer below: NOT transformed. Sits in screen space so the
       *  scale bar, label pills, and overlays stay legible regardless of zoom.
       *  Busy indicator is a single pulsing word in the corner — matches
       *  trialPlaceholderShimmer in Trials.tsx (opacity 0.4 → 0.85 → 0.4,
       *  1.4s). Only renders when `busy=true`; the previous frame stays
       *  visible during warm refreshes (3-6 ms CBED, 40-65 ms realspace)
       *  and never flashes. */}
      {busy && !error && (
        <>
          {skeleton && (
            <Box sx={{
              position: "absolute",
              inset: 0,
              pointerEvents: "none",
              background: `linear-gradient(90deg, ${colors.bg.darker} 0%, ${colors.bg.dark} 45%, ${colors.bg.black} 100%)`,
              backgroundSize: "220% 100%",
              animation: "browseCanvasSkeleton 1.15s ease-in-out infinite",
              "@keyframes browseCanvasSkeleton": {
                "0%": { backgroundPosition: "100% 0" },
                "100%": { backgroundPosition: "-100% 0" },
              },
            }} />
          )}
          <Box sx={{ position: "absolute", inset: 0,
                     display: "flex", alignItems: "center", justifyContent: "center",
                     pointerEvents: "none",
                     color: colors.overlay.onImage,
                     fontSize: fontSizes.xs, fontFamily: MONO,
                     textShadow: "0 1px 2px rgba(0,0,0,0.6)",
                     animation: "browseCanvasBusy 1.4s ease-in-out infinite",
                     "@keyframes browseCanvasBusy": {
                       "0%,100%": { opacity: 0.4 },
                       "50%": { opacity: 0.85 },
                     } }}>
            loading…
          </Box>
        </>
      )}
      {error && (
        <Box sx={{ position: "absolute", inset: 0, display: "flex",
                   flexDirection: "column", alignItems: "center", justifyContent: "center",
                   pointerEvents: "auto", gap: 1, p: 2,
                   background: "rgba(0,0,0,0.55)", overflow: "auto" }}>
          <Box sx={{ px: 1.5, py: 0.75, borderRadius: radii.md,
                     bgcolor: colors.warning.bg, color: colors.warning.text,
                     border: `1px solid ${colors.warning.border}`,
                     fontSize: fontSizes.sm, fontFamily: MONO, fontWeight: 600,
                     maxWidth: "85%", textAlign: "center" }}>
            ⚠ {error}
          </Box>
          {errorDetail ? (
            errorDetail
          ) : (
            <Box sx={{ fontSize: fontSizes.xs, color: colors.overlay.onImage,
                       fontFamily: MONO, opacity: 0.85, textAlign: "center", maxWidth: "75%",
                       textShadow: "0 1px 2px rgba(0,0,0,0.6)" }}>
              master likely missing one or more <code>_data_*.h5</code> chunk files —
              try a different master in the tree
            </Box>
          )}
        </Box>
      )}
      <Box sx={{ position: "absolute", left: 8, top: 8, color: colors.overlay.onImage,
                 fontSize: fontSizes.xs, fontFamily: MONO, textShadow: "0 1px 2px rgba(0,0,0,0.6)" }}>
        {label}
      </Box>
      <Box sx={{ position: "absolute", right: 8, top: 8, color: colors.overlay.onImage,
                 fontSize: fontSizes.xs, fontFamily: MONO, textAlign: "right",
                 textShadow: "0 1px 2px rgba(0,0,0,0.6)" }}>
        {corner}
      </Box>
      {scaleBarPx > 0 && scaleBarLabel && (
        <Box sx={{ position: "absolute", left: 8, bottom: 8, color: colors.overlay.onImage,
                   display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 0.25,
                   pointerEvents: "none" }}>
          {/* Bar: 3 px white fill on a 1 px black halo so it stays legible
           *  on any colormap. Matches utils/scalebar.ts publication spec. */}
          <Box sx={{ position: "relative", width: scaleBarPx + 2, height: 5 }}>
            <Box sx={{ position: "absolute", inset: 0, bgcolor: "#000", borderRadius: 1 }} />
            <Box sx={{ position: "absolute", left: 1, top: 1, width: scaleBarPx, height: 3,
                       bgcolor: scaleBarTone === "dark" ? colors.text.dark : "#fff", borderRadius: 1 }} />
          </Box>
          <span style={{ fontSize: fontSizes.xs, fontFamily: MONO,
                         textShadow: "0 1px 2px rgba(0,0,0,0.85), 0 0 2px rgba(0,0,0,0.85)",
                         fontWeight: 600 }}>
            {scaleBarLabel}
          </span>
        </Box>
      )}
      <Box sx={{ position: "absolute", right: 8, bottom: 8, color: colors.overlay.onImage,
                 fontSize: fontSizes.xs, fontFamily: MONO,
                 background: "rgba(0,0,0,0.45)", px: 0.75, py: 0.25, borderRadius: radii.sm,
                 transform: scaleBarPx > 0 && scaleBarLabel ? "translateY(-30px)" : "none",
                 maxWidth: "calc(100% - 16px)", overflow: "hidden", textOverflow: "ellipsis",
                 whiteSpace: "nowrap" }}>
        {hoverChip}
      </Box>
    </Box>
  );
}

/** GPU-RESIDENT realspace render: colormap a slot straight to a WebGPU canvas via
 *  `applyToCanvas` - NO rgba readback (the readback fence was ~half the per-frame cost). Source is
 *  either a CPU Float32Array (committed render) or an adopted GPU buffer (the drag fast-path, where
 *  the maskedSum result never leaves the GPU). vmin/vmax are passed in (computed once on commit,
 *  HELD during a drag) so no CPU percentile is needed per frame. Falls back to the 2D path when
 *  WebGPU is unavailable (headless). Returns true if it painted via WebGPU. */
async function renderRealspaceGpu(
  canvas: HTMLCanvasElement | null, slotIdx: number, cmap: string,
  src: { data: Float32Array } | { gpuBuffer: GPUBuffer }, width: number, height: number,
  vmin: number, vmax: number,
): Promise<boolean> {
  if (!canvas) return false;
  const engine = await getGPUColormapEngine();
  let ctx: GPUCanvasContext | null = null;
  try { ctx = canvas.getContext("webgpu") as GPUCanvasContext | null; } catch { ctx = null; }
  if (!engine || !ctx) {
    if ("data" in src) await renderToCanvas(canvas, src.data, width, height, cmap, slotIdx, 0, 1, { vmin, vmax });
    return false;
  }
  engine.uploadLUT(cmap);
  if ("data" in src) engine.uploadData(slotIdx, src.data, width, height);
  else engine.adoptBuffer(slotIdx, src.gpuBuffer, width, height);
  engine.applyToCanvas(slotIdx, vmin, vmax, ctx, canvas, false);
  return true;
}

/** Render Float32Array → 2D canvas via WebGPU colormap engine, with CPU
 *  fallback. Mirrors PanelViewer's WebGPU-first pattern but tuned for a
 *  single-slot use case (no FFT, no histogram-on-GPU). */
async function renderToCanvas(
  canvas: HTMLCanvasElement | null,
  data: Float32Array, width: number, height: number,
  cmap: string, slotIdx: number,
  pLo: number, pHi: number,
  /** Optional explicit (vmin, vmax) — when provided, skips the internal
   *  percentile clip. Used by the DP path to feed mask-DC'd percentiles. */
  clipOverride?: { vmin: number; vmax: number },
): Promise<void> {
  if (!canvas) return;
  // Resizing a canvas clears it to transparent black. Only resize when the
  // dimensions actually changed — otherwise dragging the histogram handles
  // (which only changes pLo/pHi) flashes the canvas to black between frames.
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const { vmin, vmax } = clipOverride ?? percentileClip(data, pLo * 100, pHi * 100);
  const engine = await getGPUColormapEngine();
  if (engine) {
    engine.uploadLUT(cmap);
    engine.uploadData(slotIdx, data, width, height);
    const rgba = await engine.apply(slotIdx, vmin, vmax, false);
    if (rgba) {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const buf = new ArrayBuffer(rgba.byteLength);
      new Uint8Array(buf).set(rgba);
      const img = new ImageData(new Uint8ClampedArray(buf), width, height);
      ctx.putImageData(img, 0, 0);
      return;
    }
  }
  // CPU fallback — keeps Browse functional in headless tests, but the
  // hot path on the user's GPU is always WebGPU.
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const lut = COLORMAPS[cmap] || COLORMAPS["gray"];
  const buf = new ArrayBuffer(width * height * 4);
  const rgba = new Uint8ClampedArray(buf);
  applyColormap(data, rgba, lut, vmin, vmax);
  const img = new ImageData(rgba, width, height);
  ctx.putImageData(img, 0, 0);
}

/** Per-master thumbnail used in the 5D scrubber strip. Lazily fetches the
 *  default-mode BF realspace image at the set's binning, paints it via
 *  WebGPU, and lights up the active row with the toolbar accent border.
 *  Designed to stay subtle: ~64 px tall, click selects, the strip itself
 *  occupies ~80 px including label so it never dominates the viewer. */
function ScrubberThumb({
  session, file, idx, active, detBin, onSelect,
}: {
  session: Session;
  file: MasterFile;
  idx: number;
  active: boolean;
  detBin: DetBin;
  onSelect: () => void;
}) {
  const smooth = useGlobalSmooth();
  const ref = useRef<HTMLCanvasElement>(null);
  const slotRef = useRef<number | null>(null);
  if (slotRef.current === null) slotRef.current = allocateSlot();
  const [errored, setErrored] = useState(false);
  useEffect(() => {
    let cancelled = false;
    fetchVirtualImage(session, file, "BF", 0, 1.0, null, null, undefined, detBin)
      .then((r) => {
        if (cancelled || !r) {
          if (!cancelled) setErrored(true);
          return;
        }
        // Paint at native resolution; CSS scales to the strip's height.
        void renderToCanvas(ref.current, r.data, r.width, r.height,
                            "viridis", slotRef.current!, 0.02, 0.98);
      })
      .catch(() => { if (!cancelled) setErrored(true); });
    return () => { cancelled = true; };
  }, [session.source, session.date, file.name, detBin]);
  // Release the GPU slot when this scrubber thumbnail unmounts. The strip
  // mounts one per master file; switching session/file would otherwise
  // accumulate slots forever (#438).
  useEffect(() => () => {
    const slot = slotRef.current;
    if (slot === null) return;
    void getGPUColormapEngine().then(e => e?.releaseSlot(slot));
  }, []);
  return (
    <Box
      onClick={onSelect}
      title={`#${idx + 1} · ${file.name}`}
      sx={{ position: "relative", flex: "0 0 auto",
            width: 64, height: 64, cursor: "pointer",
            border: `2px solid ${active ? colors.interactive.border : colors.border.default}`,
            borderRadius: radii.sm, overflow: "hidden",
            bgcolor: colors.overlay.onImageBg,
            boxShadow: active ? `0 0 0 1px ${colors.interactive.border}` : "none",
            "&:hover": { borderColor: colors.interactive.border } }}
    >
      <canvas ref={ref} style={{ width: "100%", height: "100%",
                                  display: "block", imageRendering: imageRenderingFor(smooth) }} />
      {errored && (
        <Box sx={{ position: "absolute", inset: 0,
                   display: "flex", alignItems: "center", justifyContent: "center",
                   fontSize: fontSizes.xs, fontFamily: MONO,
                   color: colors.warning.text, bgcolor: "rgba(0,0,0,0.5)" }}>
          err
        </Box>
      )}
      <Box sx={{ position: "absolute", left: 2, top: 2,
                 fontSize: fontSizes.xs, fontFamily: MONO,
                 fontWeight: 700,
                 color: colors.overlay.onImage,
                 textShadow: "0 1px 2px rgba(0,0,0,0.85)" }}>
        {idx + 1}
      </Box>
    </Box>
  );
}

/** 5D-STEM scrubber: thumbnail strip + slider beneath. Renders only when
 *  ``set5D != null``. Drag fires ``onSetActiveIdx`` per pointermove so the
 *  CBED + realspace fetches repaint in real time as the user scrubs.
 *
 *  Show4DSTEM frame-animation parity: Play/Pause/fps/Boomerang/Reverse
 *  controls drive the active index automatically through the loaded set.
 *  Boomerang ping-pongs between ends; Reverse plays single-direction
 *  backwards; default fps = 5 (matches show4dstem.py:271 frame_fps). */
function Scrubber5D({
  set5D, onSetActiveIdx,
}: {
  set5D: Set5D;
  onSetActiveIdx: (idx: number) => void;
}) {
  const N = set5D.files.length;
  const active = set5D.activeIdx;
  const sliderValue = N > 1 ? active / (N - 1) : 0;
  const onSliderInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    const idx = Math.round(v * (N - 1));
    if (idx !== active) onSetActiveIdx(idx);
  };
  // Playback state. Defaults match show4dstem.py:269-273: playing=False,
  // loop=True, fps=5.0, reverse=False, boomerang=False.
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(5);
  const [reverse, setReverse] = useState(false);
  const [boomerang, setBoomerang] = useState(false);
  // Direction sign drives the increment per tick. Starts +1, flipped on
  // boomerang turnaround. Held in a ref so the rAF callback closes over
  // the current value without rebuilding the timer.
  const dirRef = useRef(1);
  useEffect(() => { dirRef.current = reverse ? -1 : 1; }, [reverse]);
  // Spacebar toggles play/pause anywhere on the page (Show4DSTEM convention).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space") return;
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      e.preventDefault();
      setPlaying((p) => !p);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  // Tick loop: setInterval at 1000/fps, advances activeIdx by dirRef.
  // Boomerang flips direction at endpoints; non-boomerang loops to other end.
  // Mirrors _update_frame in show4dstem.py:3978.
  useEffect(() => {
    if (!playing || N < 2) return;
    const ms = Math.max(16, Math.round(1000 / Math.max(0.5, fps)));
    let idx = set5D.activeIdx;
    const t = setInterval(() => {
      const next = idx + dirRef.current;
      if (next >= N) {
        if (boomerang) { dirRef.current = -1; idx = N - 2; }
        else idx = 0;
      } else if (next < 0) {
        if (boomerang) { dirRef.current = 1; idx = 1; }
        else idx = N - 1;
      } else {
        idx = next;
      }
      onSetActiveIdx(idx);
    }, ms);
    return () => clearInterval(t);
    // active index is intentionally read once at start of each effect run;
    // re-running on every active change would reset the timer and stutter.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, fps, boomerang, N]);
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5,
               px: 1, py: 0.75, bgcolor: colors.bg.subtle,
               border: `1px solid ${colors.border.default}`,
               borderRadius: radii.md,
               [`@media (max-width: ${breakpoints.tablet}px)`]: {
                 "& button": {
                   minWidth: 44,
                   minHeight: 44,
                   display: "inline-flex",
                   alignItems: "center",
                   justifyContent: "center",
                 },
                 "& input": {
                   minHeight: 44,
                 },
               } }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1,
                 fontSize: fontSizes.xs, fontFamily: MONO,
                 color: colors.text.muted }}>
        <span style={{ fontWeight: 700, color: colors.text.secondary }}>
          5D {active + 1} / {N}
        </span>
        <span>{set5D.files[active]?.name}</span>
        <Box sx={{ flex: 1 }} />
        <Box
          component="button"
          onClick={() => setPlaying((p) => !p)}
          title="Play / Pause (Space)"
          data-testid="set5d-play"
          sx={{ fontSize: fontSizes.xs, fontWeight: 700,
                px: 0.75, py: 0.25, cursor: "pointer",
                border: `1px solid ${playing ? colors.interactive.border : colors.border.default}`,
                bgcolor: playing ? colors.interactive.bg : colors.bg.page,
                color: playing ? colors.interactive.selectedText : colors.text.secondary,
                borderRadius: radii.sm, lineHeight: 1.2,
                "&:hover": { bgcolor: playing ? colors.interactive.bg : colors.bg.hover } }}
        >
          {playing ? "❚❚" : "▶"}
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.25 }}>
          <span>fps</span>
          <input
            type="number" min={1} max={60} step={1} value={fps}
            onChange={(e) => setFps(Math.max(1, Math.min(60, parseFloat(e.target.value) || 5)))}
            style={{ width: 44, fontSize: fontSizes.xs, padding: "1px 4px",
                     borderRadius: radii.sm, border: `1px solid ${colors.border.default}` }}
          />
        </Box>
        <Box
          component="button"
          onClick={() => setBoomerang((b) => !b)}
          title="Boomerang: ping-pong between endpoints (Show4DSTEM frame_boomerang)"
          data-testid="set5d-boomerang"
          sx={{ fontSize: fontSizes.xs, fontWeight: 700,
                px: 0.5, py: 0.25, cursor: "pointer",
                border: `1px solid ${boomerang ? colors.interactive.border : colors.border.default}`,
                bgcolor: boomerang ? colors.interactive.bg : colors.bg.page,
                color: boomerang ? colors.interactive.selectedText : colors.text.secondary,
                borderRadius: radii.sm, lineHeight: 1.2,
                "&:hover": { bgcolor: boomerang ? colors.interactive.bg : colors.bg.hover } }}
        >↔</Box>
        <Box
          component="button"
          onClick={() => setReverse((r) => !r)}
          title="Reverse direction (Show4DSTEM frame_reverse)"
          data-testid="set5d-reverse"
          sx={{ fontSize: fontSizes.xs, fontWeight: 700,
                px: 0.5, py: 0.25, cursor: "pointer",
                border: `1px solid ${reverse ? colors.interactive.border : colors.border.default}`,
                bgcolor: reverse ? colors.interactive.bg : colors.bg.page,
                color: reverse ? colors.interactive.selectedText : colors.text.secondary,
                borderRadius: radii.sm, lineHeight: 1.2,
                "&:hover": { bgcolor: reverse ? colors.interactive.bg : colors.bg.hover } }}
        >◀</Box>
        <span>·</span>
        <span>bin {set5D.detBin}×</span>
        <span>·</span>
        <span>
          {set5D.warmMode === "window"
            ? `cached ${set5D.warmCount ?? 0}/${set5D.warmTotal ?? N}`
            : "cached all"}
        </span>
        <span>·</span>
        <span>Alt+←/→</span>
      </Box>
      <Box sx={{ display: "flex", gap: 0.5, overflowX: "auto", py: 0.25 }}>
        {set5D.files.map((f, i) => (
          <ScrubberThumb
            key={fileKey(set5D.session, f)}
            session={set5D.session}
            file={f}
            idx={i}
            active={i === active}
            detBin={set5D.detBin}
            onSelect={() => onSetActiveIdx(i)}
          />
        ))}
      </Box>
      <input
        type="range"
        min={0}
        max={1}
        step={N > 1 ? 1 / (N - 1) : 0.5}
        value={sliderValue}
        onChange={onSliderInput}
        style={{ width: "100%", margin: 0, cursor: "pointer",
                 accentColor: colors.interactive.border }}
      />
    </Box>
  );
}

export default function Viewer(props: Props) {
  const {
    session, file, browseDtype = "uint8", singleDetBin,
    mode, setMode, cmapImage, setCmapImage, cmapDp, setCmapDp,
    scanPos, setScanPos,
    ringInner, setRingInner, ringOuter, setRingOuter,
    dpShape, setDpShape, shapeParams, setShapeParams,
    showRing,
    clipLo, setClipLo, clipHi, setClipHi,
    dpClipLo, setDpClipLo, dpClipHi, setDpClipHi,
    compact, set5D, onSetActiveIdx,
    realRoi, setRealRoi,
    fftOn = false, setFftOn,
    fftWindow = true, setFftWindow,
    imageScale = "linear", setImageScale,
    dpScale = "log", setDpScale,
    imagePowerExp = 0.5, setImagePowerExp,
    dpPowerExp = 0.5, setDpPowerExp,
    maskDC = true, setMaskDC,
    profileLine = null, setProfileLine,
    profileWidth = 1, setProfileWidth,
  } = props;
  // Compute backend driving Browse: "cuda" (full-res no-bin is real-time) or
  // "mps" (MacBook; a full 192x192 masked-sum is multi-second cold, so default
  // to bin2 = 96x96 for fast browse). Read once from cache-status on mount.
  const [browseBackend, setBrowseBackend] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    fetch("/api/browse/cache-status")
      .then((r) => r.json())
      .then((d) => { if (alive) setBrowseBackend(d?.backend ?? null); })
      .catch(() => {});
    return () => { alive = false; };
  }, []);
  // Det binning honored on every realspace fetch so the cache lookup hits the
  // right LRU entry. 5D mode uses the chosen stack bin; single-master mode is
  // no-bin on CUDA, bin2 on MacBook (MPS) for fast browse. At no-bin on MPS the
  // server's bin2 sidecar keeps masked-sum real-time anyway.
  const detBin: DetBin = set5D?.detBin ?? singleDetBin ?? (browseBackend === "mps" ? 2 : 1);

  const realRef = useRef<HTMLCanvasElement>(null);
  const dpRef = useRef<HTMLCanvasElement>(null);
  // True while an aperture drag owns the realspace canvas via the GPU-resident fast path (so the
  // slow VI effect skips). viCommitTick is bumped on drag-release to fire the full path once.
  const draggingRef = useRef(false);
  const [viCommitTick, setViCommitTick] = useState(0);
  const realFrameRef = useRef<HTMLDivElement>(null);
  const dpFrameRef = useRef<HTMLDivElement>(null);

  // Track each frame's on-screen pixel width so the scale bar can size
  // itself relative to actual rendered layout (the canvases sit in a
  // responsive grid — width is a function of the viewport, not a fixed
  // number). Updated via ResizeObserver so the bar stays valid as the
  // user resizes the window.
  const [realDisplayPx, setRealDisplayPx] = useState(0);
  const [dpDisplayPx, setDpDisplayPx] = useState(0);
  useEffect(() => {
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        if (entry.target === realFrameRef.current) setRealDisplayPx(w);
        else if (entry.target === dpFrameRef.current) setDpDisplayPx(w);
      }
    });
    if (realFrameRef.current) ro.observe(realFrameRef.current);
    if (dpFrameRef.current) ro.observe(dpFrameRef.current);
    return () => ro.disconnect();
  }, []);

  const [realData, setRealData] = useState<{ data: Float32Array; w: number; h: number } | null>(null);
  const [dpData, setDpData] = useState<{ data: Float32Array; w: number; h: number } | null>(null);
  const [realBusy, setRealBusy] = useState(false);
  const [dpBusy, setDpBusy] = useState(false);
  // Master-ready gate: true once the first BF/realspace paint of the
  // active file lands. Suppresses CBED fetch until then so the user
  // never sees DP-without-BF (one transition: empty → both ready).
  const [masterReady, setMasterReady] = useState(false);
  const [realErr, setRealErr] = useState<string | null>(null);
  const [dpErr, setDpErr] = useState<string | null>(null);

  // Auto-fit BF disk geometry (cy, cx, r_bf in detector px). Seeded from the
  // backend on master change. Used to (a) position the aperture marker
  // overlay on the DP canvas, (b) seed the aperture-center state for arrow
  // keys, (c) reset the aperture on BF/ADF/DF mode click.
  const [bfGeom, setBfGeom] = useState<BfGeometry | null>(null);
  // Aperture center in DETECTOR pixels. Independent of bfGeom so the user
  // can move the aperture off-center via DP arrow keys without losing the
  // fit; clicking BF/ADF/DF in the toolbar resets it back to bfGeom.
  const [aperture, setAperture] = useState<{ cx: number; cy: number } | null>(null);
  // Which canvas owns keyboard focus — drives whether arrow keys move scan
  // position (real-space) or aperture (DP). Set on mousedown of either frame.
  const [activeCanvas, setActiveCanvas] = useState<"real" | "dp">("real");
  // Line-profile mode toggle. When ON, click-drag on the virtual image
  // sets profileLine endpoints (overrides scan-pos drag). Show2D parity.
  const [profileActive, setProfileActive] = useState(false);
  // ROI-active mode: when ON, plain click-drag on VI sets the rectangular
  // real-space ROI (without needing Alt). The summed CBED across the
  // rectangle replaces the per-position DP, via the existing /cbed-roi
  // endpoint. Show4DSTEM vi_roi_* parity (rect mode).
  const [roiActiveMode, setRoiActiveMode] = useState(false);
  // Mid-drag profile editing: which endpoint (or whole line) is being moved.
  const [profileDrag, setProfileDrag] = useState<"new" | "p0" | "p1" | null>(null);
  const [scanDragging, setScanDragging] = useState(false);

  // Rastering animation state. Steps scanPos through the scan grid in
  // row-major order at the chosen cadence. Speed corresponds to fps:
  // slow=10, med=30, fast=60. Loop wraps to (0,0) after the last position;
  // otherwise the animation stops at the end of the grid.
  type RasterSpeed = "slow" | "med" | "fast";
  type RasterPattern = "raster" | "snake" | "spiral" | "circle" | "random";
  const [rasterOn, setRasterOn] = useState(false);
  const [rasterSpeed, setRasterSpeed] = useState<RasterSpeed>("med");
  const [rasterPattern, setRasterPattern] = useState<RasterPattern>("raster");
  const [rasterLoop, setRasterLoop] = useState(true);
  // Mirror rasterOn in a ref so the real-space click handler (closed over an
  // older render) can pause the loop without going stale.
  const rasterOnRef = useRef(false);
  useEffect(() => { rasterOnRef.current = rasterOn; }, [rasterOn]);

  // Reset when the file changes — wipe stale frames so the user sees the
  // loading state on the new master rather than the previous file's pixels.
  const fileId = fileKey(session, file);
  useEffect(() => {
    setRealData(null);
    setDpData(null);
    setRealErr(null);
    setDpErr(null);
    setBfGeom(null);
    setAperture(null);
    setMasterReady(false);
  }, [fileId]);

  // Fetch BF geometry on master change. Cached server-side per (path, mtime)
  // so this is fast on revisit. Seeds the aperture state when it arrives.
  useEffect(() => {
    let cancelled = false;
    fetchBfGeometry(session, file, undefined, detBin, browseDtype).then((g) => {
      if (cancelled) return;
      setBfGeom(g);
      if (g) {
        setAperture({ cx: g.cx, cy: g.cy });
        // Seed every shape's defaults from the auto-fit BF disk so the
        // user can switch shape after master load and see the same area
        // covered by the sensible default.
        setShapeParams((prev) => ({
          ...prev,
          cx: g.cx, cy: g.cy,
          r: prev.r > 0 ? prev.r : g.r_bf,
          half: prev.half > 0 ? prev.half : g.r_bf,
          inner: prev.inner > 0 ? prev.inner : g.r_bf,
          outer: prev.outer > 0 ? prev.outer : 4 * g.r_bf,
          row0: prev.row0 || Math.max(0, Math.round(g.cy - g.r_bf)),
          col0: prev.col0 || Math.max(0, Math.round(g.cx - g.r_bf)),
          row1: prev.row1 || Math.round(g.cy + g.r_bf),
          col1: prev.col1 || Math.round(g.cx + g.r_bf),
          px: prev.px || Math.round(g.cx),
          py: prev.py || Math.round(g.cy),
        }));
      }
    });
    return () => { cancelled = true; };
  }, [fileId, detBin, browseDtype]); // eslint-disable-line react-hooks/exhaustive-deps

  // BF/ADF/DF mode click → snap shape + params to Show4DSTEM-style defaults
  // for that mode. Mirrors the ringInner/Outer reset in Browse.tsx but in
  // detector-pixel land.
  useEffect(() => {
    if (!bfGeom) return;
    const r_bf = bfGeom.r_bf;
    const cx = bfGeom.cx, cy = bfGeom.cy;
    if (mode === "BF") {
      setDpShape("circle");
      setShapeParams((prev) => ({ ...prev, cx, cy, r: r_bf }));
    } else if (mode === "ADF") {
      setDpShape("annulus");
      setShapeParams((prev) => ({ ...prev, cx, cy, inner: r_bf, outer: 3 * r_bf }));
    } else if (mode === "DF") {
      setDpShape("annulus");
      setShapeParams((prev) => ({ ...prev, cx, cy, inner: r_bf, outer: 2 * r_bf }));
    }
  }, [mode, bfGeom]); // eslint-disable-line react-hooks/exhaustive-deps

  // Shape dropdown change → seed sensible defaults for the new shape.
  // Triggered AFTER mode-change effect so the user's shape choice wins.
  const lastShapeRef = useRef<DetShape>(dpShape);
  useEffect(() => {
    if (lastShapeRef.current === dpShape) return;
    lastShapeRef.current = dpShape;
    if (!bfGeom) return;
    const { cx, cy, r_bf } = bfGeom;
    setShapeParams((prev) => {
      if (dpShape === "circle") return { ...prev, cx, cy, r: prev.r || r_bf };
      if (dpShape === "square") return { ...prev, cx, cy, half: prev.half || r_bf };
      if (dpShape === "annulus") return { ...prev, cx, cy,
        inner: prev.inner || r_bf, outer: prev.outer || 4 * r_bf };
      if (dpShape === "rect") return {
        ...prev,
        row0: Math.max(0, Math.round(cy - r_bf)),
        col0: Math.max(0, Math.round(cx - r_bf)),
        row1: Math.round(cy + r_bf),
        col1: Math.round(cx + r_bf),
      };
      if (dpShape === "point") return { ...prev,
        px: Math.round(cx), py: Math.round(cy) };
      return prev;
    });
  }, [dpShape, bfGeom]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch virtual image when (file, mode, inner, outer, aperture) change.
  // Virtual-detector manipulation targets the display cadence the user asked
  // for: ~30 fps end-to-end. The server path is ~6-10 ms warm, so the UI
  // throttle is a little tighter than 33 ms to leave room for browser
  // scheduling/fetch overhead. Point requests are latest-wins with at most
  // one server request in flight; intermediate pointer positions are coalesced.
  const apertureAtAutoCenter = !!(
    bfGeom &&
    aperture &&
    Math.abs(aperture.cx - bfGeom.cx) < 0.01 &&
    Math.abs(aperture.cy - bfGeom.cy) < 0.01
  );
  const apertureAffectsMode = mode === "BF" || mode === "ADF" || mode === "DF";
  const apCx = apertureAffectsMode && !apertureAtAutoCenter ? (aperture?.cx ?? null) : null;
  const apCy = apertureAffectsMode && !apertureAtAutoCenter ? (aperture?.cy ?? null) : null;
  const lastFireRef = useRef(0);
  const trailingTimerRef = useRef<number | null>(null);
  const seqVImgRef = useRef(0);
  const lastPaintedVImgRef = useRef(0);
  const vImgAbortRef = useRef<AbortController | null>(null);
  const vImgInFlightRef = useRef(false);
  const vImgPendingFireRef = useRef<(() => void) | null>(null);
  // Stable scalar deps for the shape effect — avoids re-firing on object
  // identity changes when the user moves the mouse without changing the
  // numeric values. Only matters when mode is BF/ADF/DF.
  const shapeUsesEndpoint = (mode === "BF" && dpShape !== "circle")
    || ((mode === "ADF" || mode === "DF") && dpShape !== "annulus");
  const sCx = shapeParams.cx, sCy = shapeParams.cy;
  const sR = shapeParams.r, sHalf = shapeParams.half;
  const sInner = shapeParams.inner, sOuter = shapeParams.outer;
  const sR0 = shapeParams.row0, sC0 = shapeParams.col0;
  const sR1 = shapeParams.row1, sC1 = shapeParams.col1;
  const sPx = shapeParams.px, sPy = shapeParams.py;
  const shapeParamsReady = !shapeUsesEndpoint
    || (dpShape === "circle" && sR > 0)
    || (dpShape === "square" && sHalf > 0)
    || (dpShape === "annulus" && sOuter > sInner && sOuter > 0)
    || (dpShape === "rect" && sR1 > sR0 && sC1 > sC0)
    || dpShape === "point";
  useEffect(() => {
    // While an aperture drag is active the GPU-resident fast path owns the realspace canvas;
    // skip the slow CPU-readback fetch (it would fight the fast path + tank the framerate).
    if (draggingRef.current) return;
    const liveDetectorActive = realData != null;
    if (shapeUsesEndpoint && !shapeParamsReady) {
      vImgPendingFireRef.current = null;
      vImgInFlightRef.current = false;
      if (vImgAbortRef.current) {
        vImgAbortRef.current.abort();
        vImgAbortRef.current = null;
      }
      if (realData == null) setRealBusy(true);
      return;
    }
    const throttleMs = VIRTUAL_DETECTOR_FETCH_THROTTLE_MS;
    const requestSeq = ++seqVImgRef.current;
    if (!liveDetectorActive) {
      vImgPendingFireRef.current = null;
      vImgInFlightRef.current = false;
    }
    if (!liveDetectorActive && vImgAbortRef.current) {
      vImgAbortRef.current.abort();
      vImgAbortRef.current = null;
    }
    // Only show loading on the initial image. Warm detector drags keep the
    // previous image visible and repaint at ~30 fps, which reads as realtime.
    if (realData == null) setRealBusy(true);
    const fire = () => {
      if (liveDetectorActive && vImgInFlightRef.current) {
        vImgPendingFireRef.current = fire;
        return;
      }
      lastFireRef.current = performance.now();
      if (liveDetectorActive) vImgInFlightRef.current = true;
      const controller = liveDetectorActive ? null : new AbortController();
      if (controller) vImgAbortRef.current = controller;
      const signal = controller?.signal;
      const releaseLiveFlight = () => {
        if (!liveDetectorActive) return;
        vImgInFlightRef.current = false;
        const pending = vImgPendingFireRef.current;
        if (!pending) return;
        vImgPendingFireRef.current = null;
        const nextDelay = Math.max(0, VIRTUAL_DETECTOR_FETCH_THROTTLE_MS - (performance.now() - lastFireRef.current));
        if (trailingTimerRef.current !== null) {
          window.clearTimeout(trailingTimerRef.current);
        }
        trailingTimerRef.current = window.setTimeout(() => {
          trailingTimerRef.current = null;
          pending();
        }, nextDelay);
      };
      const promise = shapeUsesEndpoint
        ? fetchVirtualImageShape(session, file, dpShape, shapeParams, signal, detBin, browseDtype)
        : fetchVirtualImage(session, file, mode, ringInner, ringOuter, apCx, apCy, signal, detBin, browseDtype);
      promise
        .then((r) => {
          if (signal?.aborted) { releaseLiveFlight(); return; }
          // Paint every completed warm detector frame in order. The next
          // pending cursor position fires immediately after release, so the
          // UI stays moving instead of dropping all mid-drag frames.
          if (requestSeq <= lastPaintedVImgRef.current) { releaseLiveFlight(); return; }
          lastPaintedVImgRef.current = requestSeq;
          if (controller && vImgAbortRef.current === controller) vImgAbortRef.current = null;
          setRealBusy(false);
          if (!r) {
            // When load_status reports loadable=true, the master is fine
            // and the fetch failure is almost always GPU OOM (another
            // process holding VRAM, or the cache hit its slot budget).
            // Surface a hint instead of the misleading generic message.
            const loadable = file.load_status?.loadable !== false;
            setRealErr(loadable
              ? "GPU memory unavailable — try Clear cache (top right) or pick a smaller master"
              : "failed to compute virtual image");
            // Even on failure, unblock CBED — user can still scrub.
            setMasterReady(true);
            releaseLiveFlight();
            return;
          }
          setRealErr(null);
          setRealData({ data: r.data, w: r.width, h: r.height });
          setMasterReady(true);
          releaseLiveFlight();
        })
        .catch(() => {
          if (signal?.aborted) { releaseLiveFlight(); return; }
          if (requestSeq === seqVImgRef.current) setRealBusy(false);
          releaseLiveFlight();
        });
    };
    const elapsed = performance.now() - lastFireRef.current;
    if (elapsed >= throttleMs) {
      // Leading-edge fire: outside the throttle window, go now.
      if (trailingTimerRef.current !== null) {
        window.clearTimeout(trailingTimerRef.current);
        trailingTimerRef.current = null;
      }
      fire();
    } else {
      // Inside the window: schedule a trailing fire so the LAST params
      // still get fetched once the window expires. Replacing any
      // already-pending trailing timer with a fresh one ensures the most
      // recent params are what the trailing call will read.
      if (trailingTimerRef.current !== null) {
        window.clearTimeout(trailingTimerRef.current);
      }
      trailingTimerRef.current = window.setTimeout(() => {
        trailingTimerRef.current = null;
        fire();
      }, throttleMs - elapsed);
    }
  }, [fileId, mode, ringInner, ringOuter, apCx, apCy, detBin, browseDtype, viCommitTick,
      shapeUsesEndpoint, dpShape,
      sCx, sCy, sR, sHalf, sInner, sOuter,
      sR0, sC0, sR1, sC1, sPx, sPy]); // eslint-disable-line react-hooks/exhaustive-deps

  // BF/ADF/DF mode click: also reset the aperture center to the auto-fit
  // BF disk center. This is the "auto-fit probe" behavior — the user clicks
  // a detector mode and the ring + aperture snap to standard defaults.
  // (The ring radii themselves are reset by Browse.tsx's mode effect.)
  useEffect(() => {
    if (mode === "BF" || mode === "ADF" || mode === "DF") {
      if (bfGeom) setAperture({ cx: bfGeom.cx, cy: bfGeom.cy });
    }
  }, [mode, bfGeom]);

  // CBED fetch — raster can still drive the scan position at its selected
  // cadence, but manual scan-point dragging is capped near display cadence and always
  // favors the newest cursor position. Stale requests are aborted instead of
  // serializing behind an older CBED, so the detector panel tracks the drag
  // rather than updating only when the point is placed.
  //
  // Re-visited scan positions are served by a synchronous module-level
  // cache peek BEFORE any await — so dragging the crosshair back over a
  // pixel we've already fetched paints with zero network round-trip and
  // zero busy state. Fresh positions go through fetchCBEDCached which
  // populates the cache for next time.
  const [wx, wy, kx, ky] = file.shape;
  const haveScan = wx > 0 && wy > 0;
  const ix = haveScan ? Math.round(scanPos.x * (wx - 1)) : 0;
  const iy = haveScan ? Math.round(scanPos.y * (wy - 1)) : 0;
  const seqRef = useRef(0);
  const lastPaintedSeqRef = useRef(0);
  const cbedAbortRef = useRef<AbortController | null>(null);
  const cbedTrailingTimerRef = useRef<number | null>(null);
  const lastCbedFireRef = useRef(0);
  const cbedInFlightRef = useRef(false);
  const cbedPendingFireRef = useRef<(() => void) | null>(null);
  // Stable scalar deps so the ROI param doesn't trigger a single-CBED fetch.
  const roiR0 = realRoi?.row0 ?? null;
  const roiC0 = realRoi?.col0 ?? null;
  const roiR1 = realRoi?.row1 ?? null;
  const roiC1 = realRoi?.col1 ?? null;
  useEffect(() => {
    if (!haveScan) return;
    // Gate CBED on master-ready: wait for first BF/realspace paint so the
    // user never sees DP-without-BF (mismatched timing made BF feel slow).
    // 2026-04-25 user UX call: "Until we've done the initial bright field,
    // I don't think we should show the diffraction patterns."
    if (!masterReady) return;
    const seq = ++seqRef.current;
    const scanPointDragActive = scanDragging && !rasterOn
      && roiR0 === null && roiC0 === null && roiR1 === null && roiC1 === null;
    // Rectangular ROI mode: fetch the SUMMED CBED across the rectangle
    // instead of a single-position frame. No client-side cache (the
    // server-side LRU + GPU master cache absorb the cost).
    if (roiR0 !== null && roiC0 !== null && roiR1 !== null && roiC1 !== null) {
      if (dpData == null) setDpBusy(true);
      fetchCBEDRoi(session, file, roiR0, roiC0, roiR1, roiC1, undefined, detBin, browseDtype)
        .then((r) => {
          if (seq <= lastPaintedSeqRef.current) return;
          lastPaintedSeqRef.current = seq;
          setDpBusy(false);
          if (!r) { setDpErr("failed to load summed CBED"); return; }
          setDpErr(null);
          setDpData({ data: r.data, w: r.width, h: r.height });
        })
        .catch(() => { if (seq === seqRef.current) setDpBusy(false); });
      return;
    }
    // Synchronous cache peek — if we've fetched (ix, iy) before for this
    // master, paint immediately. No await, no fetch, no busy flicker.
    const cached = CBED_CACHE.get(cbedKey(session, file, ix, iy, detBin, browseDtype));
    if (cached) {
      if (cbedTrailingTimerRef.current !== null) {
        window.clearTimeout(cbedTrailingTimerRef.current);
        cbedTrailingTimerRef.current = null;
      }
      lastPaintedSeqRef.current = seq;
      setDpData({ data: cached.data, w: cached.width, h: cached.height });
      setDpErr(null);
      setDpBusy(false);
      return;
    }
    const fire = () => {
      if (scanPointDragActive && cbedInFlightRef.current) {
        cbedPendingFireRef.current = fire;
        return;
      }
      lastCbedFireRef.current = performance.now();
      if (scanPointDragActive) {
        cbedInFlightRef.current = true;
      } else if (cbedAbortRef.current) {
        cbedAbortRef.current.abort();
        cbedAbortRef.current = null;
      }
      const controller = scanPointDragActive ? null : new AbortController();
      if (controller) cbedAbortRef.current = controller;
      const signal = controller?.signal;
      const releaseScanFlight = () => {
        if (!scanPointDragActive) return;
        cbedInFlightRef.current = false;
        const pending = cbedPendingFireRef.current;
        if (!pending) return;
        cbedPendingFireRef.current = null;
        const nextDelay = Math.max(0, SCAN_POINT_FETCH_THROTTLE_MS - (performance.now() - lastCbedFireRef.current));
        if (cbedTrailingTimerRef.current !== null) {
          window.clearTimeout(cbedTrailingTimerRef.current);
        }
        cbedTrailingTimerRef.current = window.setTimeout(() => {
          cbedTrailingTimerRef.current = null;
          pending();
        }, nextDelay);
      };
      // Never throw a centered loading overlay during scan scrubbing. A
      // GPU-hot CBED returns in ~1-5 ms, and preserving the current/blank
      // frame feels realtime; "loading..." makes the interaction read as a
      // blocking batch operation.
      if (dpData == null && !scanPointDragActive) setDpBusy(true);
      fetchCBEDCached(session, file, ix, iy, signal, detBin, browseDtype)
        .then((r) => {
          if (signal?.aborted) { releaseScanFlight(); return; }
          // Paint EVERY response that is newer than what is currently
          // painted (not strict-equality vs the latest issued seq).
          // Strict `seq !== seqRef.current` drops every mid-drag frame
          // and only the final-after-release ever paints. Paint the latest
          // completed response and drop race-losers. lastPaintedSeqRef is
          // shared across drag and non-drag branches so race-losers are
          // dropped uniformly.
          if (seq <= lastPaintedSeqRef.current) { releaseScanFlight(); return; }
          lastPaintedSeqRef.current = seq;
          if (controller && cbedAbortRef.current === controller) cbedAbortRef.current = null;
          setDpBusy(false);
          if (!r) {
            setDpErr("failed to load CBED");
            releaseScanFlight();
            return;
          }
          setDpErr(null);
          setDpData({ data: r.data, w: r.width, h: r.height });
          releaseScanFlight();
        })
        .catch(() => {
          if (signal?.aborted) { releaseScanFlight(); return; }
          // Same race-loser drop in the error path. We do NOT bump
          // lastPaintedSeqRef on errors so a later successful response
          // can still paint.
          if (seq <= lastPaintedSeqRef.current) { releaseScanFlight(); return; }
          if (controller && cbedAbortRef.current === controller) cbedAbortRef.current = null;
          setDpBusy(false);
          releaseScanFlight();
        });
    };
    const throttleMs = scanPointDragActive ? SCAN_POINT_FETCH_THROTTLE_MS : 0;
    const elapsed = performance.now() - lastCbedFireRef.current;
    if (throttleMs <= 0 || elapsed >= throttleMs) {
      if (cbedTrailingTimerRef.current !== null) {
        window.clearTimeout(cbedTrailingTimerRef.current);
        cbedTrailingTimerRef.current = null;
      }
      fire();
    } else {
      if (cbedTrailingTimerRef.current !== null) {
        window.clearTimeout(cbedTrailingTimerRef.current);
      }
      cbedTrailingTimerRef.current = window.setTimeout(() => {
        cbedTrailingTimerRef.current = null;
        fire();
      }, throttleMs - elapsed);
    }
  }, [fileId, ix, iy, haveScan, roiR0, roiC0, roiR1, roiC1, detBin, masterReady,
      scanDragging, rasterOn]); // eslint-disable-line react-hooks/exhaustive-deps

  // Real-space display buffer — apply the user-selected intensity scale.
  // `linear` returns the raw data; `log` / `sqrt` materialize a fresh buffer
  // so the histogram and canvas see the same transformed values.
  const realDisplayData = useMemo(() => {
    if (!realData) return null;
    return applyScale(realData.data, imageScale, imagePowerExp);
  }, [realData, imageScale, imagePowerExp]);

  // Last (vmin,vmax) used for the realspace VI - cached so the aperture-drag fast path can HOLD
  // contrast (no per-frame CPU percentile -> no readback). Refreshed on every committed render.
  const lastRealClipRef = useRef<{ vmin: number; vmax: number } | null>(null);
  // ---- 60fps aperture-drag fast path (mirrors Show4DSTEM's direct recompute) ----
  // During a BF/ADF/DF aperture drag we bypass React entirely: maskedSum stays a GPU buffer (no
  // readback), colormapped straight to the canvas (applyToCanvas, no readback), contrast HELD.
  // The slow path (setState -> CPU readback -> percentile -> apply readback -> putImageData, the
  // ~110ms/frame that capped the drag at ~9fps) is skipped while dragging. Stale-closure-proof:
  // the latest mode/aperture/cmap live in refs, read at render time.
  const apertureRef = useRef<{ cx: number; cy: number } | null>(null);
  const ringRef = useRef<{ inner: number; outer: number }>({ inner: 0, outer: 0 });
  const dragRafRef = useRef<number | null>(null);
  const dragAgainRef = useRef(false);
  apertureRef.current = aperture;
  ringRef.current = { inner: ringInner, outer: ringOuter };
  // Coalesced (single in-flight) GPU-resident recompute+paint of the realspace canvas from the
  // CURRENT aperture. Re-fires if the aperture moved again mid-frame. Read via fastViRef so the
  // window mousemove handler always calls the freshest closure (no stale session/mode/cmap).
  const scheduleFastVI = () => {
    if (dragRafRef.current != null) { dragAgainRef.current = true; return; }
    dragRafRef.current = requestAnimationFrame(async () => {
      dragRafRef.current = null;
      const ap = apertureRef.current;
      const res = await fetchVirtualImageBufferGpu(session, file, mode, ringRef.current.inner, ringRef.current.outer, ap?.cx ?? null, ap?.cy ?? null, detBin, browseDtype);
      if (res) {
        const clip = lastRealClipRef.current ?? { vmin: 0, vmax: 1 };
        await renderRealspaceGpu(realRef.current, SLOT_REAL, cmapImage, { gpuBuffer: res.buffer }, res.width, res.height, clip.vmin, clip.vmax);
      }
      if (dragAgainRef.current && draggingRef.current) { dragAgainRef.current = false; scheduleFastVI(); }
    });
  };
  const fastViRef = useRef(scheduleFastVI);
  fastViRef.current = scheduleFastVI;
  const endApertureDrag = () => {
    draggingRef.current = false;
    if (dragRafRef.current != null) { cancelAnimationFrame(dragRafRef.current); dragRafRef.current = null; }
    dragAgainRef.current = false;
    setViCommitTick((t) => t + 1);
  };
  // Paint real-space canvas whenever the data, colormap, scale, or clip range changes. Uses the
  // GPU-resident path (applyToCanvas, no rgba readback). Computes the percentile here so the held
  // value is available to the drag fast path.
  useEffect(() => {
    if (!realData || !realDisplayData) return;
    const { vmin, vmax } = percentileClip(realDisplayData, clipLo * 100, clipHi * 100);
    lastRealClipRef.current = { vmin, vmax };
    void renderRealspaceGpu(realRef.current, SLOT_REAL, cmapImage, { data: realDisplayData }, realData.w, realData.h, vmin, vmax);
  }, [realData, realDisplayData, cmapImage, clipLo, clipHi]);

  // FFT of the virtual image. Always log-scaled (Bragg-disc patterns span
  // many orders of magnitude). Recomputes on master swap and on every
  // realData refresh; WebGPU first, CPU fallback. The buffer is the FFT
  // magnitude already log1p-scaled inside fft2dMagnitude{,GPU}().
  const fftRef = useRef<HTMLCanvasElement>(null);
  const fftFrameRef = useRef<HTMLDivElement>(null);
  const [fftData, setFftData] = useState<{ data: Float32Array; w: number; h: number } | null>(null);
  const [fftClipLo, setFftClipLo] = useState(0.05);
  const [fftClipHi, setFftClipHi] = useState(0.99);
  // FFT panel gets its own cmap/scale/power-exp so the FFT colormap can
  // differ from the VI colormap. Show4DSTEM keeps fft_colormap separate
  // from vi_colormap (show4dstem.py:234). Defaults match Show4DSTEM:
  // fft_scale_mode="linear", fft_power_exp=0.5.
  const [cmapFft, setCmapFft] = useState<ColormapName>("inferno");
  const [fftScale, setFftScale] = useState<IntensityScale>("linear");
  const [fftPowerExp, setFftPowerExp] = useState(0.5);
  const fftSeqRef = useRef(0);
  useEffect(() => {
    if (!fftOn || !realData) {
      setFftData(null);
      return;
    }
    const seq = ++fftSeqRef.current;
    (async () => {
      // First-toggle-on: pay the WebGPU shader compile inside the user's
      // click action; resolves immediately on every subsequent toggle.
      await ensureFFTWarmed();
      if (seq !== fftSeqRef.current) return;
      const w = realData.w, h = realData.h;
      const windowed = prepareFftInput(realData.data, h, w, fftWindow);
      let mag = await fft2dMagnitudeGPU(windowed, h, w);
      if (!mag) mag = fft2dMagnitude(windowed, h, w);
      if (seq !== fftSeqRef.current) return;
      setFftData({ data: mag, w, h });
    })().catch(() => { /* fft is best-effort */ });
  }, [fftOn, realData, fftWindow]);

  // FFT scale transform applied per-pixel before colormap. Defaults to
  // linear (matches Show4DSTEM fft_scale_mode default).
  const fftDisplayData = useMemo(() => {
    if (!fftData) return null;
    return applyScale(fftData.data, fftScale, fftPowerExp);
  }, [fftData, fftScale, fftPowerExp]);
  useEffect(() => {
    if (!fftData || !fftDisplayData) return;
    void renderToCanvas(fftRef.current, fftDisplayData, fftData.w, fftData.h,
                        cmapFft, SLOT_FFT, fftClipLo, fftClipHi);
  }, [fftData, fftDisplayData, cmapFft, fftClipLo, fftClipHi]);

  // Paint DP canvas — user-selected scale (default log so diffuse Bragg
  // disks survive the central beam's dynamic range). Colormap and clip
  // range are independent from the real-space panel.
  const dpDisplayData = useMemo(() => {
    if (!dpData) return null;
    return applyScale(dpData.data, dpScale, dpPowerExp);
  }, [dpData, dpScale, dpPowerExp]);

  // Cache min/max of the DP display buffer so the histogram readout's
  // raw-counts conversion is one inverse-transform + linear-interp, no
  // extra scans across the buffer.
  const dpLogRange = useMemo(() => {
    if (!dpDisplayData || dpDisplayData.length === 0) return null;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < dpDisplayData.length; i++) {
      const v = dpDisplayData[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    if (!Number.isFinite(mn) || !Number.isFinite(mx) || mn === mx) return null;
    return { mn, mx };
  }, [dpDisplayData]);
  // Live profile values: bilinear sample of the (raw, pre-scale) virtual
  // image along the profileLine. Mirrors Show2D `profileData` derivation.
  const profileValues = useMemo(() => {
    if (!profileLine || !realData) return null;
    return sampleLineProfile(
      realData.data, realData.w, realData.h,
      profileLine.row0, profileLine.col0, profileLine.row1, profileLine.col1,
      Math.max(1, Math.round(profileWidth)),
    );
  }, [profileLine, profileWidth, realData]);

  const dpInverseScale = useCallback((v: number): number => {
    if (dpScale === "log") return Math.expm1(v);
    if (dpScale === "sqrt") return v * v;
    if (dpScale === "power") return v > 0 ? Math.pow(v, 1 / dpPowerExp) : 0;
    return v;
  }, [dpScale, dpPowerExp]);
  useEffect(() => {
    if (!dpData || !dpDisplayData) return;
    // Mask-DC: percentile clip ignores the central 3×3 around the detected
    // BF center so the central beam doesn't crush the histogram. Show4DSTEM
    // parity (show4dstem.py:3991).
    const cr = bfGeom?.cy ?? dpData.h / 2;
    const cc = bfGeom?.cx ?? dpData.w / 2;
    const clip = percentileClipMasked(
      dpDisplayData, dpData.w, dpData.h,
      dpClipLo * 100, dpClipHi * 100,
      maskDC, cr, cc,
    );
    void renderToCanvas(dpRef.current, dpDisplayData, dpData.w, dpData.h,
                        cmapDp, SLOT_DP, dpClipLo, dpClipHi, clip);
  }, [dpData, dpDisplayData, cmapDp, dpClipLo, dpClipHi, maskDC, bfGeom]);

  // Pan/zoom transforms for each canvas. State held here so zoom/pan
  // survives across renders (e.g. CBED frame swaps as the user drags
  // the scan position). Each canvas has its own transform — zooming the
  // real-space view does not zoom the diffraction pattern and vice-versa.
  const [realTf, setRealTf] = useState<Transform>(IDENTITY);
  const [dpTf, setDpTf] = useState<Transform>(IDENTITY);

  // Scan-side drag — converts a mouse position on the (possibly zoomed)
  // real-space frame into a normalized (0..1) scan coordinate. Inverts the
  // current transform so the crosshair lands on the correct image pixel
  // even when the view is zoomed in.
  const scanMovePendingRef = useRef<{ x: number; y: number } | null>(null);
  const scanMoveRafRef = useRef<number | null>(null);
  const flushScanMove = useCallback(() => {
    scanMoveRafRef.current = null;
    const pending = scanMovePendingRef.current;
    scanMovePendingRef.current = null;
    if (!pending) return;
    setScanPos(pending);
  }, [setScanPos]);
  const onScanMove = useCallback((e: MouseEvent) => {
    const frame = realFrameRef.current;
    if (!frame) return;
    const r = frame.getBoundingClientRect();
    const px = e.clientX - r.left;
    const py = e.clientY - r.top;
    // Invert: image_px = (screen_px - tx) / scale (in frame units).
    // Then normalize by frame size to get (0..1).
    const ix = (px - realTf.tx) / realTf.scale;
    const iy = (py - realTf.ty) / realTf.scale;
    const x = ix / r.width;
    const y = iy / r.height;
    scanMovePendingRef.current = {
      x: Math.max(0, Math.min(1, x)),
      y: Math.max(0, Math.min(1, y)),
    };
    if (scanMoveRafRef.current === null) {
      scanMoveRafRef.current = window.requestAnimationFrame(flushScanMove);
    }
  }, [realTf, flushScanMove]);
  useEffect(() => () => {
    if (scanMoveRafRef.current !== null) {
      window.cancelAnimationFrame(scanMoveRafRef.current);
      scanMoveRafRef.current = null;
    }
  }, []);

  // ROI rectangle drag state. The user holds Alt and drags on the real-space
  // canvas to define a (row0, col0) → (row1, col1) rectangle in scan-pixel
  // coordinates. Live updates fire setRealRoi every mousemove so the DP
  // panel repaints with the running summed CBED as the rect grows.
  const [roiDragging, setRoiDragging] = useState<
    | { startRow: number; startCol: number }
    | null
  >(null);
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (setRealRoi) setRealRoi(null);
        if (setProfileLine) setProfileLine(null);
        setProfileActive(false);
      }
    };
    window.addEventListener("keydown", down);
    return () => window.removeEventListener("keydown", down);
  }, [setRealRoi, setProfileLine]);
  // Pan state — which canvas is being panned and the drag origin in
  // screen coordinates plus the transform at drag start.
  const [panState, setPanState] = useState<
    | { which: "real" | "dp"; startX: number; startY: number; startTf: Transform }
    | null
  >(null);
  const [shiftHeld, setShiftHeld] = useState(false);

  // Ring-resize state. Show4DSTEM-style: ONE handle per ring at the SE
  // diagonal (cos45°). Hovering or grabbing that handle starts a drag that
  // sets the corresponding ring radius to sqrt(dx²+dy²) from the aperture
  // center. No compass-aware lockstep — Show4DSTEM treats inner/outer as
  // independent. `nearRing` reflects HOVER over the handle.
  const [nearRing, setNearRing] = useState<"inner" | "outer" | null>(null);
  const [ringDrag, setRingDrag] = useState<
    | { which: "inner" | "outer" }
    | null
  >(null);
  // Aperture-center drag: Show4DSTEM lets the user reposition the BF center
  // by grabbing the crosshair marker on the DP. Same paradigm — mousedown
  // within ~10 CSS-px of `aperture` starts a drag; mousemove updates
  // aperture.cx/cy in detector pixels, BF/ADF/DF re-render in real time.
  const [nearCenter, setNearCenter] = useState(false);
  const [centerDrag, setCenterDrag] = useState(false);

  // Generic shape drag — square/rect/point overlays. ``which`` identifies
  // the handle the user grabbed: ``"center"`` translates the whole shape,
  // ``"se"`` resizes from the SE corner, ``"corner-tl"`` etc. re-anchor a
  // rect corner. ``startEvent`` captures the initial detector-px state so
  // the move math is delta-based and doesn't drift.
  const [shapeDrag, setShapeDrag] = useState<
    | { kind: "square-center"; startCx: number; startCy: number; startMx: number; startMy: number }
    | { kind: "square-se"; cx: number; cy: number }
    | { kind: "rect-corner"; corner: "tl" | "tr" | "bl" | "br";
        startR0: number; startC0: number; startR1: number; startC1: number }
    | { kind: "rect-center"; startR0: number; startC0: number; startR1: number; startC1: number;
        startMx: number; startMy: number }
    | { kind: "point" }
    | null
  >(null);
  const pointDragPendingRef = useRef<{ px: number; py: number } | null>(null);
  const pointDragRafRef = useRef<number | null>(null);
  const flushPointDrag = useCallback(() => {
    pointDragRafRef.current = null;
    const pending = pointDragPendingRef.current;
    pointDragPendingRef.current = null;
    if (!pending) return;
    setShapeParams((prev) => {
      if (prev.px === pending.px && prev.py === pending.py) return prev;
      return { ...prev, px: pending.px, py: pending.py };
    });
  }, [setShapeParams]);

  /** Hit test for the aperture center marker. Returns true when the cursor
   *  is within ~10 CSS-px of the current aperture center, mapped to detector
   *  pixels via the active zoom. Center wins over ring-edge hit tests so the
   *  user always grabs the center first when they aim at it. */
  const apertureHitTest = useCallback((clientX: number, clientY: number): boolean => {
    if (!aperture || !dpData) return false;
    const frame = dpFrameRef.current;
    if (!frame) return false;
    const bbox = frame.getBoundingClientRect();
    if (bbox.width <= 0 || bbox.height <= 0) return false;
    const screenPxPerDetPx = (bbox.width / dpData.w) * dpTf.scale;
    if (!Number.isFinite(screenPxPerDetPx) || screenPxPerDetPx <= 0) return false;
    const detX = ((clientX - bbox.left - dpTf.tx) / dpTf.scale) * (dpData.w / bbox.width);
    const detY = ((clientY - bbox.top - dpTf.ty) / dpTf.scale) * (dpData.h / bbox.height);
    const tolDet = 10 / screenPxPerDetPx;
    const dx = detX - aperture.cx;
    const dy = detY - aperture.cy;
    return (dx * dx + dy * dy) <= tolDet * tolDet;
  }, [aperture, dpData, dpTf]);

  /** Convert a clientX/clientY pair to detector-pixel distance from the
   *  aperture center. Honors the DP canvas's pan/zoom transform AND the
   *  current bounding box so the math stays correct when the user has
   *  zoomed in. Returns null when geometry isn't ready. */
  const dpDistanceFromCenter = useCallback((clientX: number, clientY: number): {
    dist: number; screenPxPerDetPx: number;
  } | null => {
    const frame = dpFrameRef.current;
    if (!frame || !dpData || !aperture) return null;
    const bbox = frame.getBoundingClientRect();
    if (bbox.width <= 0 || bbox.height <= 0) return null;
    const screenPxPerDetPx = (bbox.width / dpData.w) * dpTf.scale;
    if (!Number.isFinite(screenPxPerDetPx) || screenPxPerDetPx <= 0) return null;
    // Inverse of `translate(tx,ty) scale(scale)` on the frame, then map
    // frame pixels (0..bbox.width) to detector pixels (0..dpData.w).
    const detX = ((clientX - bbox.left - dpTf.tx) / dpTf.scale) * (dpData.w / bbox.width);
    const detY = ((clientY - bbox.top - dpTf.ty) / dpTf.scale) * (dpData.h / bbox.height);
    const dx = detX - aperture.cx;
    const dy = detY - aperture.cy;
    return { dist: Math.sqrt(dx * dx + dy * dy), screenPxPerDetPx };
  }, [dpData, aperture, dpTf]);

  /** Convert a client (x,y) to detector pixel coords on the DP. Returns
   *  null when the DP frame isn't laid out yet. Same inverse-transform as
   *  `dpDistanceFromCenter` but without the aperture-relative distance —
   *  used by the shape-overlay drag handlers. */
  const clientToDetPx = useCallback((clientX: number, clientY: number): {
    detX: number; detY: number; screenPxPerDetPx: number;
  } | null => {
    const frame = dpFrameRef.current;
    if (!frame || !dpData) return null;
    const bbox = frame.getBoundingClientRect();
    if (bbox.width <= 0 || bbox.height <= 0) return null;
    const screenPxPerDetPx = (bbox.width / dpData.w) * dpTf.scale;
    if (!Number.isFinite(screenPxPerDetPx) || screenPxPerDetPx <= 0) return null;
    const detX = ((clientX - bbox.left - dpTf.tx) / dpTf.scale) * (dpData.w / bbox.width);
    const detY = ((clientY - bbox.top - dpTf.ty) / dpTf.scale) * (dpData.h / bbox.height);
    return { detX, detY, screenPxPerDetPx };
  }, [dpData, dpTf]);

  /** Show4DSTEM resize-handle hit test
   *  (~/repos/quantem.widget/js/show4dstem/index.tsx:2596-2621).
   *
   *  Each ring has ONE resize handle at the SE diagonal — offset
   *  (radius * cos45°, radius * cos45°) from the aperture center. We
   *  hit-test against the handle position with a 10 CSS-px radius
   *  (RESIZE_HIT_AREA_PX), translated to detector pixels via the current
   *  zoom. No compass / cardinal logic: the user grabs ONE handle and
   *  drags. */
  const ringHitTest = useCallback((clientX: number, clientY: number): {
    side: "inner" | "outer";
  } | null => {
    if (!bfGeom || !aperture) return null;
    const frame = dpFrameRef.current;
    if (!frame || !dpData) return null;
    const bbox = frame.getBoundingClientRect();
    if (bbox.width <= 0 || bbox.height <= 0) return null;
    const screenPxPerDetPx = (bbox.width / dpData.w) * dpTf.scale;
    if (!Number.isFinite(screenPxPerDetPx) || screenPxPerDetPx <= 0) return null;
    // Convert the cursor to detector pixels.
    const detX = ((clientX - bbox.left - dpTf.tx) / dpTf.scale) * (dpData.w / bbox.width);
    const detY = ((clientY - bbox.top - dpTf.ty) / dpTf.scale) * (dpData.h / bbox.height);
    // 10 CSS-px hit radius → detector px (Show4DSTEM RESIZE_HIT_AREA_PX).
    const tolDet = 10 / screenPxPerDetPx;
    const rInner = ringInner * bfGeom.r_bf;
    const rOuter = ringOuter * bfGeom.r_bf;
    // Handle positions: SE diagonal, offset = radius * cos(45°).
    const COS45 = Math.SQRT1_2; // 0.7071...
    const outerHx = aperture.cx + rOuter * COS45;
    const outerHy = aperture.cy + rOuter * COS45;
    const innerHx = aperture.cx + rInner * COS45;
    const innerHy = aperture.cy + rInner * COS45;
    const dOuter = Math.sqrt((detX - outerHx) ** 2 + (detY - outerHy) ** 2);
    const dInner = Math.sqrt((detX - innerHx) ** 2 + (detY - innerHy) ** 2);
    // Outer wins ties (Show4DSTEM convention: outer is the more common
    // interaction; inner is hidden when ringInner ≈ 0).
    if (dOuter < tolDet && dOuter <= dInner) return { side: "outer" };
    if (dInner < tolDet && ringInner > 0.01) return { side: "inner" };
    if (dOuter < tolDet) return { side: "outer" };
    return null;
  }, [bfGeom, ringInner, ringOuter, dpData, aperture, dpTf]);

  // DP hover — light up the center crosshair OR ring SE handle as the
  // cursor approaches. Center wins over ring (matches mousedown priority).
  const onDpHover = useCallback((e: React.MouseEvent) => {
    if (ringDrag || centerDrag || panState || !showRing) return;
    const overCenter = apertureHitTest(e.clientX, e.clientY);
    setNearCenter(overCenter);
    if (overCenter) { setNearRing(null); return; }
    const hit = ringHitTest(e.clientX, e.clientY);
    setNearRing((prev) => (prev === (hit?.side ?? null) ? prev : (hit?.side ?? null)));
  }, [ringDrag, centerDrag, panState, showRing, ringHitTest, apertureHitTest]);

  const onDpLeave = useCallback(() => {
    if (!ringDrag && !centerDrag) {
      setNearRing(null);
      setNearCenter(false);
    }
  }, [ringDrag]);

  // Track shift key globally so cursor feedback updates even when the
  // mouse is hovering and shift is pressed/released without movement.
  useEffect(() => {
    const down = (e: KeyboardEvent) => { if (e.key === "Shift") setShiftHeld(true); };
    const up = (e: KeyboardEvent) => { if (e.key === "Shift") setShiftHeld(false); };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  // Arrow-key navigation. Two modes:
  //   - activeCanvas === "real": step scan position by one scan pixel
  //     (Shift ×10). Real-time DP update follows.
  //   - activeCanvas === "dp": step the virtual-detector aperture center
  //     by one DETECTOR pixel (Shift ×10). Triggers a virtual-image
  //     re-render via the apCx/apCy effect.
  // R always resets both pan/zoom transforms.
  useEffect(() => {
    if (!haveScan) return;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        setRealTf(IDENTITY);
        setDpTf(IDENTITY);
        return;
      }
      let dxSign = 0, dySign = 0;
      if (e.key === "ArrowLeft") dxSign = -1;
      else if (e.key === "ArrowRight") dxSign = 1;
      else if (e.key === "ArrowUp") dySign = -1;
      else if (e.key === "ArrowDown") dySign = 1;
      else return;
      e.preventDefault();
      // Alt+ArrowLeft / Alt+ArrowRight scrubs the 5D-STEM index. Up/down
      // keep their existing scan-position-step behavior (no orthogonal
      // 5D action). Probe parameters (aperture / rings / scan / cmap)
      // stay locked across the index swap by design.
      if (set5D && onSetActiveIdx && e.altKey && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
        onSetActiveIdx(set5D.activeIdx + dxSign);
        return;
      }
      const step = e.shiftKey ? 10 : 1;
      if (activeCanvas === "dp") {
        // Move the aperture center on the detector. Clamp inside the
        // detector frame so the ring stays drawable.
        setAperture((cur) => {
          const seed = cur ?? (bfGeom ? { cx: bfGeom.cx, cy: bfGeom.cy }
                                      : { cx: kx / 2, cy: ky / 2 });
          const ncx = Math.max(0, Math.min(kx - 1, seed.cx + dxSign * step));
          const ncy = Math.max(0, Math.min(ky - 1, seed.cy + dySign * step));
          return { cx: ncx, cy: ncy };
        });
      } else {
        const stepX = step / Math.max(1, wx - 1);
        const stepY = step / Math.max(1, wy - 1);
        setScanPos({
          x: Math.max(0, Math.min(1, scanPos.x + dxSign * stepX)),
          y: Math.max(0, Math.min(1, scanPos.y + dySign * stepY)),
        });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [haveScan, wx, wy, kx, ky, scanPos.x, scanPos.y, setScanPos, activeCanvas, bfGeom, set5D, onSetActiveIdx]);

  /** Convert a client (x,y) on the real-space frame to integer scan pixel
   *  coords (row, col). Honors the active pan/zoom transform. */
  const clientToScanPixel = useCallback((clientX: number, clientY: number): { row: number; col: number } | null => {
    const frame = realFrameRef.current;
    if (!frame) return null;
    const r = frame.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return null;
    const px = clientX - r.left;
    const py = clientY - r.top;
    const ix = (px - realTf.tx) / realTf.scale;
    const iy = (py - realTf.ty) / realTf.scale;
    const fracX = Math.max(0, Math.min(1, ix / r.width));
    const fracY = Math.max(0, Math.min(1, iy / r.height));
    return {
      col: Math.max(0, Math.min(wx - 1, Math.round(fracX * (wx - 1)))),
      row: Math.max(0, Math.min(wy - 1, Math.round(fracY * (wy - 1)))),
    };
  }, [realTf, wx, wy]);

  // Real-space mousedown — disambiguate scan-position selection vs pan vs ROI.
  // Alt held → ROI rectangle drag. Shift held → pan (only meaningful when
  // zoomed). No modifier → scan select, preserving the existing
  // aperture-linking behavior. Also sets activeCanvas so subsequent arrow
  // keys move the scan position.
  const onRealMouseDown = (e: React.MouseEvent) => {
    setActiveCanvas("real");
    // Any user click on the real-space canvas pauses the rastering loop —
    // matches the spec "any user click on the real-space canvas pauses".
    if (rasterOnRef.current) setRasterOn(false);
    // Profile mode: drag from p0 to p1 to set a fresh profile, or grab an
    // existing endpoint to refine. Wins over scan-pos / ROI / pan when
    // toggled on. Show2D parity.
    if (profileActive && setProfileLine && haveScan) {
      const p = clientToScanPixel(e.clientX, e.clientY);
      if (p) {
        e.preventDefault();
        if (profileLine) {
          // Hit-test endpoints (within ~5 scan-px).
          const d0 = Math.hypot(p.row - profileLine.row0, p.col - profileLine.col0);
          const d1 = Math.hypot(p.row - profileLine.row1, p.col - profileLine.col1);
          if (d0 < 5 && d0 <= d1) { setProfileDrag("p0"); return; }
          if (d1 < 5) { setProfileDrag("p1"); return; }
        }
        // New profile: anchor at click, second endpoint follows mouse.
        setProfileLine({ row0: p.row, col0: p.col, row1: p.row, col1: p.col });
        setProfileDrag("new");
      }
      return;
    }
    if ((e.altKey || roiActiveMode) && setRealRoi && haveScan) {
      e.preventDefault();
      const p = clientToScanPixel(e.clientX, e.clientY);
      if (p) {
        setRoiDragging({ startRow: p.row, startCol: p.col });
        setRealRoi({ row0: p.row, col0: p.col, row1: p.row + 1, col1: p.col + 1 });
      }
      return;
    }
    if (e.shiftKey && realTf.scale > 1) {
      e.preventDefault();
      setPanState({
        which: "real",
        startX: e.clientX, startY: e.clientY,
        startTf: realTf,
      });
      return;
    }
    setScanDragging(true);
    onScanMove(e.nativeEvent);
  };

  // CBED mousedown — priority: aperture center > ring resize > pan.
  // Always sets activeCanvas to "dp" so arrow keys move the aperture center.
  const onDpMouseDown = (e: React.MouseEvent) => {
    setActiveCanvas("dp");
    // Aperture-center drag (Show4DSTEM): grab the crosshair marker to move
    // the BF/ADF/DF center. Wins over ring resize when the cursor is
    // simultaneously near both (rare — rings are at radius>0).
    if (showRing && aperture && apertureHitTest(e.clientX, e.clientY)) {
      e.preventDefault();
      draggingRef.current = true;   // GPU-resident fast path owns the canvas until mouseup
      setCenterDrag(true);
      return;
    }
    // Shape-driven modes (BF/ADF/DF) with non-circle/non-annulus shape:
    // hit-test the new shape handles. Square has SE-corner resize +
    // center translate; rect has 4 corner handles + center translate;
    // point uses its crosshair as the drag target.
    const shapeUsesEndpointMd = mode === "BF" || mode === "ADF" || mode === "DF";
    if (shapeUsesEndpointMd && dpShape !== "circle" && dpShape !== "annulus") {
      const probe = clientToDetPx(e.clientX, e.clientY);
      if (probe) {
        const tolDet = 10 / probe.screenPxPerDetPx;
        if (dpShape === "square") {
          const cx = shapeParams.cx, cy = shapeParams.cy, half = shapeParams.half;
          // SE-corner handle wins over center.
          const seX = cx + half, seY = cy + half;
          if (Math.hypot(probe.detX - seX, probe.detY - seY) < tolDet) {
            e.preventDefault();
            setShapeDrag({ kind: "square-se", cx, cy });
            return;
          }
          if (Math.abs(probe.detX - cx) <= half && Math.abs(probe.detY - cy) <= half) {
            e.preventDefault();
            setShapeDrag({ kind: "square-center", startCx: cx, startCy: cy,
                           startMx: probe.detX, startMy: probe.detY });
            return;
          }
        } else if (dpShape === "rect") {
          const r0 = shapeParams.row0, c0 = shapeParams.col0;
          const r1 = shapeParams.row1, c1 = shapeParams.col1;
          // 4 corner handles. tl=(c0,r0), tr=(c1,r0), bl=(c0,r1), br=(c1,r1).
          const corners: { name: "tl" | "tr" | "bl" | "br"; x: number; y: number }[] = [
            { name: "tl", x: c0, y: r0 }, { name: "tr", x: c1, y: r0 },
            { name: "bl", x: c0, y: r1 }, { name: "br", x: c1, y: r1 },
          ];
          for (const c of corners) {
            if (Math.hypot(probe.detX - c.x, probe.detY - c.y) < tolDet) {
              e.preventDefault();
              setShapeDrag({ kind: "rect-corner", corner: c.name,
                             startR0: r0, startC0: c0, startR1: r1, startC1: c1 });
              return;
            }
          }
          // Center-drag if inside the rect.
          if (probe.detX >= c0 && probe.detX <= c1
              && probe.detY >= r0 && probe.detY <= r1) {
            e.preventDefault();
            setShapeDrag({ kind: "rect-center",
                           startR0: r0, startC0: c0, startR1: r1, startC1: c1,
                           startMx: probe.detX, startMy: probe.detY });
            return;
          }
        } else if (dpShape === "point") {
          // Point's crosshair: any click on the DP repositions it
          // immediately + starts a translate-on-move drag.
          e.preventDefault();
          const detW = dpData?.w ?? 1, detH = dpData?.h ?? 1;
          const px = Math.max(0, Math.min(detW - 1, Math.round(probe.detX)));
          const py = Math.max(0, Math.min(detH - 1, Math.round(probe.detY)));
          pointDragPendingRef.current = null;
          if (pointDragRafRef.current !== null) {
            window.cancelAnimationFrame(pointDragRafRef.current);
            pointDragRafRef.current = null;
          }
          setShapeParams((prev) => (
            prev.px === px && prev.py === py ? prev : { ...prev, px, py }
          ));
          setShapeDrag({ kind: "point" });
          return;
        }
      }
    }
    // Ring resize takes precedence over pan/zoom interactions.
    if (showRing && bfGeom && (dpShape === "circle" || dpShape === "annulus")) {
      const hit = ringHitTest(e.clientX, e.clientY);
      if (hit) {
        e.preventDefault();
        setNearRing(hit.side);
        draggingRef.current = true;   // GPU-resident fast path owns the canvas until mouseup
        setRingDrag({ which: hit.side });
        return;
      }
    }
    if (dpTf.scale <= 1) return;
    e.preventDefault();
    setPanState({
      which: "dp",
      startX: e.clientX, startY: e.clientY,
      startTf: dpTf,
    });
  };

  useEffect(() => {
    if (!scanDragging) return;
    const m = (e: MouseEvent) => onScanMove(e);
    const u = () => setScanDragging(false);
    window.addEventListener("mousemove", m);
    window.addEventListener("mouseup", u);
    return () => { window.removeEventListener("mousemove", m); window.removeEventListener("mouseup", u); };
  }, [scanDragging, onScanMove]);

  // Profile-line drag: keep updating row1/col1 (or whichever endpoint is
  // grabbed) until mouseup. Show2D parity: live update during drag, no
  // commit-on-release.
  useEffect(() => {
    if (!profileDrag || !setProfileLine || !profileLine) return;
    const m = (e: MouseEvent) => {
      const p = clientToScanPixel(e.clientX, e.clientY);
      if (!p) return;
      if (profileDrag === "new" || profileDrag === "p1") {
        setProfileLine({ ...profileLine, row1: p.row, col1: p.col });
      } else if (profileDrag === "p0") {
        setProfileLine({ ...profileLine, row0: p.row, col0: p.col });
      }
    };
    const u = () => setProfileDrag(null);
    window.addEventListener("mousemove", m);
    window.addEventListener("mouseup", u);
    return () => { window.removeEventListener("mousemove", m); window.removeEventListener("mouseup", u); };
  }, [profileDrag, profileLine, clientToScanPixel, setProfileLine]);

  // ROI rectangle drag — every move updates row0/col0/row1/col1 in scan
  // pixels with the canonical ordering (row0 ≤ row1, col0 ≤ col1). The
  // CBED-roi fetch effect throttles itself naturally via the seq guard.
  useEffect(() => {
    if (!roiDragging || !setRealRoi) return;
    const m = (e: MouseEvent) => {
      const p = clientToScanPixel(e.clientX, e.clientY);
      if (!p) return;
      const r0 = Math.min(roiDragging.startRow, p.row);
      const r1 = Math.max(roiDragging.startRow, p.row) + 1;
      const c0 = Math.min(roiDragging.startCol, p.col);
      const c1 = Math.max(roiDragging.startCol, p.col) + 1;
      setRealRoi({ row0: r0, col0: c0, row1: r1, col1: c1 });
    };
    const u = () => setRoiDragging(null);
    window.addEventListener("mousemove", m);
    window.addEventListener("mouseup", u);
    return () => {
      window.removeEventListener("mousemove", m);
      window.removeEventListener("mouseup", u);
    };
  }, [roiDragging, clientToScanPixel, setRealRoi]);

  // Pan drag — translate the active transform by the delta from drag start,
  // clamped so the image edge can't be pulled past the canvas center.
  useEffect(() => {
    if (!panState) return;
    const frame = panState.which === "real" ? realFrameRef.current : dpFrameRef.current;
    const setter = panState.which === "real" ? setRealTf : setDpTf;
    const m = (e: MouseEvent) => {
      if (!frame) return;
      const r = frame.getBoundingClientRect();
      const dx = e.clientX - panState.startX;
      const dy = e.clientY - panState.startY;
      const next = {
        tx: panState.startTf.tx + dx,
        ty: panState.startTf.ty + dy,
        scale: panState.startTf.scale,
      };
      setter(clampTransform(next, r.width, r.height));
    };
    const u = () => setPanState(null);
    window.addEventListener("mousemove", m);
    window.addEventListener("mouseup", u);
    return () => {
      window.removeEventListener("mousemove", m);
      window.removeEventListener("mouseup", u);
    };
  }, [panState]);

  // Aperture-center drag — Show4DSTEM lets the user reposition the BF
  // center by dragging the crosshair marker on the DP. Mouse position →
  // detector-pixel coordinates via the same inverse-transform math used
  // by ringHitTest. Updates aperture.cx/cy on every frame; the realspace
  // fetch effect picks up the change and repaints.
  useEffect(() => {
    if (!centerDrag || !dpData) return;
    const m = (e: MouseEvent) => {
      const frame = dpFrameRef.current;
      if (!frame) return;
      const bbox = frame.getBoundingClientRect();
      if (bbox.width <= 0 || bbox.height <= 0) return;
      const detX = ((e.clientX - bbox.left - dpTf.tx) / dpTf.scale) * (dpData.w / bbox.width);
      const detY = ((e.clientY - bbox.top - dpTf.ty) / dpTf.scale) * (dpData.h / bbox.height);
      const cx = Math.max(0, Math.min(dpData.w - 1, detX));
      const cy = Math.max(0, Math.min(dpData.h - 1, detY));
      apertureRef.current = { cx, cy };   // latest center for the fast path (sync, pre-render)
      fastViRef.current();                // GPU-resident recompute+paint, no React, no readback
      setAperture({ cx, cy });
      // Keep the shape-endpoint params in sync so circle/annulus stay
      // centered on the same point the user is dragging.
      setShapeParams((prev) => ({ ...prev, cx, cy }));
    };
    const u = () => endApertureDrag();
    window.addEventListener("mousemove", m);
    window.addEventListener("mouseup", u);
    return () => {
      window.removeEventListener("mousemove", m);
      window.removeEventListener("mouseup", u);
    };
  }, [centerDrag, dpData, dpTf]);

  // Ring resize drag — Show4DSTEM convention
  // (~/repos/quantem.widget/js/show4dstem/index.tsx:2823-2855).
  //
  // The new radius is sqrt(dx² + dy²) from the aperture center, mapped to
  // α = dist / r_bf. Inner and outer drag independently — outer must stay
  // greater than inner + GAP, inner must stay less than outer - GAP. No
  // lockstep, no anchored delta math. Updates fire every frame; the
  // realspace fetch effect's 80 ms throttle absorbs the rate.
  useEffect(() => {
    if (!ringDrag || !bfGeom) return;
    const { which } = ringDrag;
    const MIN = 0, MAX = 6, GAP = 0.05;
    const m = (e: MouseEvent) => {
      const probe = dpDistanceFromCenter(e.clientX, e.clientY);
      if (!probe) return;
      const alpha = probe.dist / bfGeom.r_bf;
      // Mirror to shape-endpoint params (detector pixels) for the new
      // /realspace-shape path. The legacy /realspace path reads
      // ringInner/Outer (α-units); both stay in lockstep so toggling
      // shape between circle/annulus on the dropdown works seamlessly.
      const distPx = probe.dist;
      if (which === "inner") {
        setShapeParams((prev) => ({ ...prev, inner: distPx }));
      } else {
        // Outer drag handles BOTH circle.r AND annulus.outer — same handle.
        setShapeParams((prev) => ({ ...prev, r: distPx, outer: distPx }));
      }
      if (which === "inner") {
        const v = Math.max(MIN, Math.min(ringOuter - GAP, alpha));
        ringRef.current = { ...ringRef.current, inner: v }; setRingInner(v);
      } else {
        const v = Math.max(ringInner + GAP, Math.min(MAX, alpha));
        ringRef.current = { ...ringRef.current, outer: v }; setRingOuter(v);
      }
      fastViRef.current();   // GPU-resident recompute+paint from the new radius, no React/readback
    };
    const u = () => { endApertureDrag(); setRingDrag(null); };
    window.addEventListener("mousemove", m);
    window.addEventListener("mouseup", u);
    return () => {
      window.removeEventListener("mousemove", m);
      window.removeEventListener("mouseup", u);
    };
  }, [ringDrag, bfGeom, dpDistanceFromCenter, ringInner, ringOuter, setRingInner, setRingOuter]);

  // Shape drag — square / rect / point handlers. Heavy detector shapes still
  // update through the throttled fetch path; point movement is cheap enough to
  // coalesce to requestAnimationFrame for near-display-rate feedback.
  useEffect(() => {
    if (!shapeDrag) return;
    const m = (e: MouseEvent) => {
      const probe = clientToDetPx(e.clientX, e.clientY);
      if (!probe) return;
      const detW = dpData?.w ?? 1, detH = dpData?.h ?? 1;
      const cl = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
      if (shapeDrag.kind === "square-center") {
        const dx = probe.detX - shapeDrag.startMx;
        const dy = probe.detY - shapeDrag.startMy;
        setShapeParams((prev) => ({ ...prev,
          cx: cl(shapeDrag.startCx + dx, 0, detW - 1),
          cy: cl(shapeDrag.startCy + dy, 0, detH - 1) }));
      } else if (shapeDrag.kind === "square-se") {
        const half = Math.max(1, Math.hypot(probe.detX - shapeDrag.cx,
                                            probe.detY - shapeDrag.cy) * Math.SQRT1_2);
        setShapeParams((prev) => ({ ...prev, half }));
      } else if (shapeDrag.kind === "rect-corner") {
        // Corner drag: anchor the OPPOSITE corner; new corner follows mouse.
        let { startR0: r0, startC0: c0, startR1: r1, startC1: c1 } = shapeDrag;
        const dx = cl(probe.detX, 0, detW), dy = cl(probe.detY, 0, detH);
        if (shapeDrag.corner === "tl") { r0 = dy; c0 = dx; }
        if (shapeDrag.corner === "tr") { r0 = dy; c1 = dx; }
        if (shapeDrag.corner === "bl") { r1 = dy; c0 = dx; }
        if (shapeDrag.corner === "br") { r1 = dy; c1 = dx; }
        // Canonicalize.
        const nr0 = Math.min(r0, r1), nr1 = Math.max(r0, r1);
        const nc0 = Math.min(c0, c1), nc1 = Math.max(c0, c1);
        setShapeParams((prev) => ({ ...prev,
          row0: nr0, row1: nr1, col0: nc0, col1: nc1 }));
      } else if (shapeDrag.kind === "rect-center") {
        const dx = probe.detX - shapeDrag.startMx;
        const dy = probe.detY - shapeDrag.startMy;
        const w = shapeDrag.startC1 - shapeDrag.startC0;
        const h = shapeDrag.startR1 - shapeDrag.startR0;
        let nc0 = shapeDrag.startC0 + dx, nr0 = shapeDrag.startR0 + dy;
        nc0 = cl(nc0, 0, detW - w); nr0 = cl(nr0, 0, detH - h);
        setShapeParams((prev) => ({ ...prev,
          col0: nc0, row0: nr0, col1: nc0 + w, row1: nr0 + h }));
      } else if (shapeDrag.kind === "point") {
        pointDragPendingRef.current = {
          px: cl(Math.round(probe.detX), 0, detW - 1),
          py: cl(Math.round(probe.detY), 0, detH - 1),
        };
        if (pointDragRafRef.current === null) {
          pointDragRafRef.current = window.requestAnimationFrame(flushPointDrag);
        }
      }
    };
    const u = () => {
      if (pointDragRafRef.current !== null) {
        window.cancelAnimationFrame(pointDragRafRef.current);
        pointDragRafRef.current = null;
      }
      flushPointDrag();
      setShapeDrag(null);
    };
    window.addEventListener("mousemove", m);
    window.addEventListener("mouseup", u);
    return () => {
      window.removeEventListener("mousemove", m);
      window.removeEventListener("mouseup", u);
      if (pointDragRafRef.current !== null) {
        window.cancelAnimationFrame(pointDragRafRef.current);
        pointDragRafRef.current = null;
      }
      pointDragPendingRef.current = null;
    };
  }, [shapeDrag, clientToDetPx, dpData, flushPointDrag, setShapeParams]);

  // Wheel zoom — anchor zoom on the cursor so the pixel under the mouse
  // stays put across the zoom step. Passive listener has to be opted out
  // (`{ passive: false }`) so we can preventDefault on the page scroll.
  useEffect(() => {
    const wireWheel = (
      el: HTMLDivElement | null,
      tf: Transform,
      setTf: (t: Transform) => void,
    ) => {
      if (!el) return () => {};
      const handler = (e: WheelEvent) => {
        e.preventDefault();
        const r = el.getBoundingClientRect();
        const cx = e.clientX - r.left;
        const cy = e.clientY - r.top;
        // 1.1 multiplier per notch. deltaY > 0 = scroll down = zoom out.
        const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
        const newScale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, tf.scale * factor));
        const ratio = newScale / tf.scale;
        // Keep the pixel under the cursor anchored: solve for new tx, ty
        // such that (cx - tx) / scale stays constant after the zoom step.
        const nextTx = cx - (cx - tf.tx) * ratio;
        const nextTy = cy - (cy - tf.ty) * ratio;
        setTf(clampTransform({ tx: nextTx, ty: nextTy, scale: newScale }, r.width, r.height));
      };
      el.addEventListener("wheel", handler, { passive: false });
      return () => el.removeEventListener("wheel", handler);
    };
    const offReal = wireWheel(realFrameRef.current, realTf, setRealTf);
    const offDp = wireWheel(dpFrameRef.current, dpTf, setDpTf);
    return () => { offReal(); offDp(); };
  }, [realTf, dpTf]);

  const onRealDoubleClick = () => setRealTf(IDENTITY);
  const onDpDoubleClick = () => setDpTf(IDENTITY);

  // Rastering animation. setInterval at the chosen fps; each tick advances
  // the scan position one grid cell in row-major order. The CBED fetch
  // pipeline (cache-fronted, with previous-frame persistence) absorbs warm
  // hits at 30 fps; cold cells fall behind the cadence, surfacing as a
  // "buffering" badge below the toolbar button.
  useEffect(() => {
    if (!rasterOn || !haveScan) return;
    const fps = rasterSpeed === "slow" ? 10
              : rasterSpeed === "fast" ? 60
              : 30;
    const periodMs = Math.max(1, Math.round(1000 / fps));
    let step = 0;
    // Pattern generators: each returns the next (ix, iy) given a step
    // counter. `null` from `next` = pattern complete (used by raster
    // when not looping).
    const total = wx * wy;
    const cxScan = (wx - 1) / 2;
    const cyScan = (wy - 1) / 2;
    const next = (): { ix: number; iy: number } | null => {
      const i = step++;
      if (rasterPattern === "snake") {
        // Boustrophedon: row-major but flips column direction every row.
        const row = Math.floor(i / wx) % wy;
        const colInRow = i % wx;
        const col = row % 2 === 0 ? colInRow : (wx - 1 - colInRow);
        return { ix: col, iy: row };
      }
      if (rasterPattern === "circle") {
        // Concentric rings — 60 points per ring, 12 rings total. One full
        // sweep covers the field in 720 ticks (24 s at 30 fps, 12 s at 60).
        const maxR = Math.min(cxScan, cyScan) * 0.95;
        const N_PTS = 60;
        const N_RINGS = 12;
        const ringIdx = Math.floor(i / N_PTS);
        const ptInRing = i % N_PTS;
        if (ringIdx >= N_RINGS) {
          if (rasterLoop) { step = 1; return next(); }
          return null;
        }
        const r = maxR * (ringIdx + 1) / N_RINGS;
        const theta = (ptInRing / N_PTS) * 2 * Math.PI;
        const ix = Math.round(cxScan + r * Math.cos(theta));
        const iy = Math.round(cyScan + r * Math.sin(theta));
        return {
          ix: Math.max(0, Math.min(wx - 1, ix)),
          iy: Math.max(0, Math.min(wy - 1, iy)),
        };
      }
      if (rasterPattern === "spiral") {
        // Archimedean spiral from center outward with VISIBLE step. Each
        // tick advances ~5° in angle and r grows linearly with i. Reaches
        // the edge in ~360 ticks (12 s at 30 fps, 6 s at 60).
        const maxR = Math.min(cxScan, cyScan) * 0.95;
        const dTheta = (5 * Math.PI) / 180; // 5° per step
        const theta = i * dTheta;
        const r = (maxR / 360) * i;
        if (r > maxR) {
          if (rasterLoop) { step = 1; return next(); }
          return null;
        }
        const ix = Math.round(cxScan + r * Math.cos(theta));
        const iy = Math.round(cyScan + r * Math.sin(theta));
        return {
          ix: Math.max(0, Math.min(wx - 1, ix)),
          iy: Math.max(0, Math.min(wy - 1, iy)),
        };
      }
      if (rasterPattern === "random") {
        // Random walk over the grid. Lightweight, no repeat tracking.
        return {
          ix: Math.floor(Math.random() * wx),
          iy: Math.floor(Math.random() * wy),
        };
      }
      // Default raster: row-major.
      if (i >= total) {
        if (rasterLoop) { step = 1; return { ix: 0, iy: 0 }; }
        return null;
      }
      return { ix: i % wx, iy: Math.floor(i / wx) };
    };
    const tick = () => {
      const p = next();
      if (!p) {
        setRasterOn(false);
        return;
      }
      setScanPos({
        x: p.ix / Math.max(1, wx - 1),
        y: p.iy / Math.max(1, wy - 1),
      });
    };
    const realId = window.setInterval(tick, periodMs);
    return () => window.clearInterval(realId);
  }, [rasterOn, rasterSpeed, rasterPattern, rasterLoop, haveScan, wx, wy]); // eslint-disable-line react-hooks/exhaustive-deps

  // "Buffering" indicator: the raster cadence outpaces the warm CBED cache
  // when many cells are cold. We surface a visible flag whenever the DP is
  // busy AND the rastering loop is running.
  const rasterBuffering = rasterOn && dpBusy;

  // Cursor feedback. Real-space: crosshair (default scan-select), grab when
  // shift held + zoomed, grabbing while panning. DP: default when not zoomed,
  // grab when zoomed, grabbing while panning.
  const realCursor = (() => {
    if (panState?.which === "real") return "grabbing";
    if (shiftHeld && realTf.scale > 1) return "grab";
    return "crosshair";
  })();
  // Resize cursor — Show4DSTEM places the handle at the SE diagonal, so
  // the cursor is always `nwse-resize` while hovering or dragging it.
  const dpCursor = (() => {
    if (centerDrag) return "grabbing";
    if (ringDrag) return "nwse-resize";
    if (panState?.which === "dp") return "grabbing";
    if (nearCenter) return "move";
    if (nearRing) return "nwse-resize";
    if (dpTf.scale > 1) return "grab";
    return "default";
  })();

  const ringStroke = mode === "BF" ? colors.overlay.peakMarker : colors.text.white;
  const modeLabel = detectorModeLabel(mode);

  // Scale-bar sizing per canvas. `scan_sampling_A` (Å per scan pixel) and
  // `k_pixel_size_mrad` (mrad per detector pixel) come from the session's
  // `dataset.yaml` via `/api/browse/sessions`. When either is null the
  // canvas falls through to a "N px" label that still tracks zoom.
  const stepA = file.scan_sampling_A ?? null;
  const stepMrad = file.k_pixel_size_mrad ?? null;
  const realScaleBar = useMemo(
    () => pickScaleBarForCanvas({
      imagePx: wx, stepA, unit: "real",
      zoom: realTf.scale, displayPx: realDisplayPx,
    }),
    [wx, stepA, realTf.scale, realDisplayPx],
  );
  const dpScaleBar = useMemo(
    () => pickScaleBarForCanvas({
      imagePx: kx, stepMrad, unit: "k",
      zoom: dpTf.scale, displayPx: dpDisplayPx,
    }),
    [kx, stepMrad, dpTf.scale, dpDisplayPx],
  );

  // When the master.h5 references chunk files that don't exist on disk, the
  // server returns a non-loadable status with a structured missing-chunks
  // report. Surface that as a rich error block so the user can see WHICH
  // files are missing and how much data IS present (so they know whether
  // it's a broken transfer vs an inactive master). Falls through to the
  // generic banner when the request failed for any other reason.
  const missingStatus = file.load_status;
  const hasMissingChunks =
    !!missingStatus && missingStatus.missing_count > 0;
  const realErrHeadline = hasMissingChunks
    ? "Master not loadable"
    : realErr;
  const realErrorDetail = hasMissingChunks && missingStatus
    ? <MissingChunksDetail status={missingStatus} />
    : undefined;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5, minWidth: 0 }}>
      {set5D && set5D.files.length >= 2 && onSetActiveIdx && (
        <Scrubber5D set5D={set5D} onSetActiveIdx={onSetActiveIdx} />
      )}
      {/* Toolbar — borderless, sits naked on the page. Tool toggles live on
          the real-space side; detector-mode and detector-shape controls are
          grouped to the right next to DP/raster controls. Compact mode shrinks
          padding ~50 % so the canvas owns more vertical space on a small
          MacBook screen without losing the mode/cmap controls. */}
      <Box sx={{ display: "flex", flexWrap: "wrap", alignItems: "center",
                 gap: compact ? 0.75 : 1.5,
                 p: compact ? 0.5 : 1, bgcolor: colors.bg.subtle,
                 [`@media (max-width: ${breakpoints.tablet}px)`]: {
                   "& button, & [role='button']": {
                     minWidth: 44,
                     minHeight: 44,
                     display: "inline-flex",
                     alignItems: "center",
                     justifyContent: "center",
                   },
                   "& label": {
                     minHeight: 44,
                     display: "inline-flex",
                     alignItems: "center",
                   },
                   "& input[type='checkbox']": {
                     width: 24,
                     height: 24,
                   },
                 } }}>
        {/* Per-panel cmap/scale/power/extras now live above each panel's
            histogram (Show4DSTEM grouping). Top toolbar keeps only global
            controls: modes, shapes, FFT toggle, Profile, ROI, raster. */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <Box
            component="button"
            onClick={() => setFftOn && setFftOn(!fftOn)}
            title="Toggle FFT-of-virtual-image panel"
            sx={{ fontSize: fontSizes.sm, fontWeight: 600,
                  px: compact ? 0.75 : 1, py: compact ? 0.25 : 0.5, cursor: "pointer",
                  border: `1px solid ${fftOn ? colors.interactive.border : colors.border.default}`,
                  bgcolor: fftOn ? colors.interactive.bg : colors.bg.page,
                  color: fftOn ? colors.interactive.selectedText : colors.text.secondary,
                  borderRadius: radii.sm,
                  "&:hover": { bgcolor: fftOn ? colors.interactive.bg : colors.bg.hover } }}
          >
            FFT
          </Box>
          {setProfileLine && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
              <Box
                component="button"
                onClick={() => {
                  setProfileActive((on) => !on);
                  if (profileActive && setProfileLine) setProfileLine(null);
                }}
                title="Line profile on virtual image: drag to set endpoints, drag endpoint dots to refine, Esc to clear (Show2D parity)"
                data-testid="profile-toggle"
                sx={{ fontSize: fontSizes.sm, fontWeight: 600,
                      px: compact ? 0.75 : 1, py: compact ? 0.25 : 0.5, cursor: "pointer",
                      border: `1px solid ${profileActive ? colors.interactive.border : colors.border.default}`,
                      bgcolor: profileActive ? colors.interactive.bg : colors.bg.page,
                      color: profileActive ? colors.interactive.selectedText : colors.text.secondary,
                      borderRadius: radii.sm,
                      "&:hover": { bgcolor: profileActive ? colors.interactive.bg : colors.bg.hover } }}
              >
                Profile
              </Box>
              {profileLine && setProfileWidth && (
                <input
                  type="number"
                  min={1} max={21} step={1}
                  value={profileWidth}
                  onChange={(e) => setProfileWidth(Math.max(1, Math.min(21, parseInt(e.target.value, 10) || 1)))}
                  title="Width: average N parallel lines (Show4DSTEM profile_width, default 1)"
                  style={{ fontSize: fontSizes.sm, width: 56, padding: "3px 6px",
                           borderRadius: radii.md, border: `1px solid ${colors.border.default}` }}
                />
              )}
            </Box>
          )}
          {setRealRoi && (
            <Box
              component="button"
              onClick={() => {
                setRoiActiveMode((on) => !on);
                if (roiActiveMode && setRealRoi) setRealRoi(null);
              }}
              title="Real-space ROI: drag a rectangle on the virtual image. The DP panel updates to the SUMMED diffraction pattern across the rectangle (Show4DSTEM vi_roi rect parity). Esc clears."
              data-testid="roi-toggle"
              sx={{ fontSize: fontSizes.sm, fontWeight: 600,
                    px: compact ? 0.75 : 1, py: compact ? 0.25 : 0.5, cursor: "pointer",
                    border: `1px solid ${roiActiveMode || realRoi ? colors.interactive.border : colors.border.default}`,
                    bgcolor: roiActiveMode || realRoi ? colors.interactive.bg : colors.bg.page,
                    color: roiActiveMode || realRoi ? colors.interactive.selectedText : colors.text.secondary,
                    borderRadius: radii.sm,
                    "&:hover": { bgcolor: roiActiveMode || realRoi ? colors.interactive.bg : colors.bg.hover } }}
            >
              ROI
            </Box>
          )}
          {realRoi && setRealRoi && (
            <Box
              component="button"
              onClick={() => setRealRoi(null)}
              title="Clear rectangular ROI (Esc)"
              sx={{ fontSize: fontSizes.sm, fontWeight: 600,
                    px: compact ? 0.75 : 1, py: compact ? 0.25 : 0.5, cursor: "pointer",
                    border: `1px solid ${colors.warning.border}`,
                    bgcolor: colors.warning.bg, color: colors.warning.text,
                    borderRadius: radii.sm,
                    "&:hover": { opacity: 0.85 } }}
            >
              clear ROI
            </Box>
          )}
        </Box>

        <Box sx={{ flex: 1 }} />

        <Box sx={{ display: "flex", border: `1px solid ${colors.border.default}`, borderRadius: radii.md, overflow: "hidden" }}>
          {DETECTOR_MODES.map((m) => (
            <Box
              key={m}
              component="button"
              onClick={() => setMode(m)}
              title={
                m === "BF" ? "Bright field: central disk, 0-1 x BF radius"
                : m === "DF" ? "Dark field: annulus, 1-2 x BF radius"
                : m === "ADF" ? "Annular dark field: annulus, 1-3 x BF radius"
                : detectorModeLabel(m)
              }
              sx={{ px: compact ? 0.75 : 1, py: compact ? 0.25 : 0.5,
                    fontSize: fontSizes.sm, fontWeight: 600, cursor: "pointer", border: "none",
                    bgcolor: mode === m ? colors.text.primary : colors.bg.page,
                    color: mode === m ? colors.text.white : colors.text.secondary,
                    borderRight: `1px solid ${colors.border.default}`,
                    "&:last-child": { borderRight: "none" },
                    "&:hover": { bgcolor: mode === m ? colors.text.primary : colors.bg.hover } }}
            >{detectorModeLabel(m)}</Box>
          ))}
        </Box>

        {/* Detector-shape presets — Show4DSTEM-style live shape on the DP,
            rendered as the same segmented pill row as the detector modes
            so it's visually unmissable. Only meaningful for BF/ADF/DF
            (mask-based modes); CoM/iCoM/SSB modes ignore the shape and stay
            on the legacy /realspace path. */}
        {(mode === "BF" || mode === "ADF" || mode === "DF") && (
          <Box sx={{ display: "flex", border: `1px solid ${colors.border.default}`, borderRadius: radii.md, overflow: "hidden" }} data-testid="dp-shape-presets">
            {DET_SHAPES.map((s) => (
              <Box
                key={s}
                component="button"
                onClick={() => setDpShape(s)}
                title={`Detector shape: ${s}`}
                sx={{ px: compact ? 0.75 : 1, py: compact ? 0.25 : 0.5,
                      fontSize: fontSizes.sm, fontWeight: 600, cursor: "pointer", border: "none",
                      bgcolor: dpShape === s ? colors.text.primary : colors.bg.page,
                      color: dpShape === s ? colors.text.white : colors.text.secondary,
                      borderRight: `1px solid ${colors.border.default}`,
                      "&:last-child": { borderRight: "none" },
                      "&:hover": { bgcolor: dpShape === s ? colors.text.primary : colors.bg.hover } }}
              >{s}</Box>
            ))}
          </Box>
        )}

        {/* Rastering animation toggle — sweeps scanPos through the scan
            grid in row-major order. Wired through the same setScanPos
            path used by drag + arrow keys, so each tick fetches one CBED
            via the warm cache. */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <Box
            component="button"
            onClick={() => setRasterOn((on) => !on)}
            title={rasterOn ? "Pause rastering animation" : "Start rastering animation"}
            sx={{ fontSize: fontSizes.sm, fontWeight: 600,
                  px: compact ? 0.75 : 1, py: compact ? 0.25 : 0.5, cursor: "pointer",
                  border: `1px solid ${rasterOn ? colors.interactive.border : colors.border.default}`,
                  bgcolor: rasterOn ? colors.interactive.bg : colors.bg.page,
                  color: rasterOn ? colors.interactive.selectedText : colors.text.secondary,
                  borderRadius: radii.sm,
                  "&:hover": { bgcolor: rasterOn ? colors.interactive.bg : colors.bg.hover } }}
          >
            {rasterOn ? "❚❚ raster" : "▶ raster"}
          </Box>
          <select
            value={rasterSpeed}
            onChange={(e) => setRasterSpeed(e.target.value as RasterSpeed)}
            title="Rastering frame rate"
            style={SELECT_CONTROL_STYLE}
          >
            <option value="slow">10 fps</option>
            <option value="med">30 fps</option>
            <option value="fast">60 fps</option>
          </select>
          <select
            value={rasterPattern}
            onChange={(e) => setRasterPattern(e.target.value as RasterPattern)}
            title="Rastering pattern across scan grid"
            style={SELECT_CONTROL_STYLE}
          >
            <option value="raster">raster</option>
            <option value="snake">snake</option>
            <option value="spiral">spiral</option>
            <option value="circle">circle</option>
            <option value="random">random</option>
          </select>
          <Box component="label"
               sx={{ display: "flex", alignItems: "center", gap: 0.5,
                     fontSize: fontSizes.sm, color: colors.text.muted, cursor: "pointer" }}
               title="Wrap to (0,0) at end of scan grid">
            <input type="checkbox" checked={rasterLoop}
                   onChange={(e) => setRasterLoop(e.target.checked)}
                   style={{ margin: 0 }} />
            loop
          </Box>
        </Box>
      </Box>
      {rasterBuffering && (
        <Box sx={{ display: "flex", justifyContent: "flex-end", mt: -0.5 }}>
          <Box sx={{ fontSize: fontSizes.xs, fontFamily: MONO,
                     color: colors.text.muted, opacity: 0.85,
                     px: 0.75, py: 0.25, borderRadius: radii.sm,
                     bgcolor: colors.bg.subtle }}>
            buffering…
          </Box>
        </Box>
      )}

      {/* Stage: real-space | DP | FFT (3 columns when FFT on, matches
          Show4DSTEM dp_vi_fft template). Two columns when FFT off. */}
      <Box sx={{ display: "grid",
                 gridTemplateColumns: {
                   xs: "1fr",
                   md: fftOn ? "1fr 1fr 1fr" : "1fr 1fr",
                 },
                 gap: 1.5 }}>
        <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
          <CanvasCard
            innerRef={realFrameRef}
            onMouseDown={onRealMouseDown}
            onDoubleClick={onRealDoubleClick}
            cursor={realCursor}
            transform={realTf}
            focused={activeCanvas === "real"}
            label={`${modeLabel} · real-space${realTf.scale > 1 ? ` · ${realTf.scale.toFixed(1)}×` : ""}`}
            corner={haveScan ? <>scan {wx}×{wy}</> : <>scan ?</>}
            hoverChip={haveScan ? `scan (${ix},${iy})` : "no scan shape"}
            scaleBarPx={realScaleBar?.barPx ?? 0}
            scaleBarLabel={realScaleBar?.label ?? ""}
            busy={realBusy && !hasMissingChunks}
            error={realErrHeadline || (hasMissingChunks ? "Master not loadable" : null)}
            errorDetail={realErrorDetail}
            overlay={
              haveScan ? (
                <Box sx={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
                  <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
                    <line x1={scanPos.x * 100} y1="0" x2={scanPos.x * 100} y2="100"
                          stroke={colors.overlay.profileAccent} strokeWidth="0.25" strokeDasharray="1 1" opacity="0.85" />
                    <line x1="0" y1={scanPos.y * 100} x2="100" y2={scanPos.y * 100}
                          stroke={colors.overlay.profileAccent} strokeWidth="0.25" strokeDasharray="1 1" opacity="0.85" />
                    <circle cx={scanPos.x * 100} cy={scanPos.y * 100} r="1.2" fill="none"
                            stroke={colors.overlay.profileAccent} strokeWidth="0.4" />
                    {realRoi && wx > 0 && wy > 0 && (() => {
                      // ROI rectangle in 0..100 viewBox space. Yellow dashed,
                      // tinted fill so the ROI is visible at any zoom.
                      const x = (realRoi.col0 / Math.max(1, wx - 1)) * 100;
                      const y = (realRoi.row0 / Math.max(1, wy - 1)) * 100;
                      const w = ((realRoi.col1 - realRoi.col0) / Math.max(1, wx - 1)) * 100;
                      const h = ((realRoi.row1 - realRoi.row0) / Math.max(1, wy - 1)) * 100;
                      return (
                        <rect x={x} y={y} width={w} height={h}
                              fill="rgba(255,215,0,0.12)" stroke="rgba(255,215,0,0.95)"
                              strokeWidth="0.4" strokeDasharray="1.2 0.8" />
                      );
                    })()}
                    {profileLine && wx > 0 && wy > 0 && (() => {
                      // Profile line in 0..100 viewBox. Cyan stroke with
                      // endpoint dots; matches Show2D profile color.
                      const x0 = (profileLine.col0 / Math.max(1, wx - 1)) * 100;
                      const y0 = (profileLine.row0 / Math.max(1, wy - 1)) * 100;
                      const x1 = (profileLine.col1 / Math.max(1, wx - 1)) * 100;
                      const y1 = (profileLine.row1 / Math.max(1, wy - 1)) * 100;
                      return (
                        <>
                          <line x1={x0} y1={y0} x2={x1} y2={y1}
                                stroke="#2dd4bf" strokeWidth="0.4" opacity="0.95" />
                          <circle cx={x0} cy={y0} r="0.9" fill="#2dd4bf" />
                          <circle cx={x1} cy={y1} r="0.9" fill="#2dd4bf" />
                        </>
                      );
                    })()}
                  </svg>
                </Box>
              ) : undefined
            }
          >
            <canvas ref={realRef} />
          </CanvasCard>
          <PanelToolbar
            cmap={cmapImage}
            setCmap={setCmapImage}
            scale={imageScale}
            setScale={(s) => setImageScale && setImageScale(s)}
            powerExp={imagePowerExp}
            setPowerExp={setImagePowerExp}
            compact={compact}
          />
          <Histogram data={realDisplayData} cmap={cmapImage} lo={clipLo} hi={clipHi}
                     setLo={setClipLo} setHi={setClipHi} label={`intensity · ${modeLabel}`}
                     compact={compact} />
          {profileValues && profileLine && (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25, mt: 0.25 }}>
              <Box sx={{ display: "flex", justifyContent: "space-between",
                         fontSize: fontSizes.xs, color: colors.text.muted, fontFamily: MONO }}>
                <span>line profile · {profileValues.length} samples · width {profileWidth}px</span>
                <span>
                  ({profileLine.col0.toFixed(0)},{profileLine.row0.toFixed(0)}) →
                  ({profileLine.col1.toFixed(0)},{profileLine.row1.toFixed(0)}) ·
                  d={Math.hypot(profileLine.col1 - profileLine.col0, profileLine.row1 - profileLine.row0).toFixed(1)}px
                </span>
              </Box>
              <ProfilePlot values={profileValues} />
            </Box>
          )}
          {/* FFT moved to its own 3rd column (matches Show4DSTEM
              dp_vi_fft layout). */}
        </Box>

        <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
          {/* No alpha slider above the DP — the user resizes the rings by
              dragging directly on the DP canvas (#418). The alpha-mrad
              numeric readout was removed 2026-04-25 because the canvas-drag
              UX is intuitive on its own. */}
          <CanvasCard
            innerRef={dpFrameRef}
            onMouseDown={onDpMouseDown}
            onMouseMove={onDpHover}
            onMouseLeave={onDpLeave}
            onDoubleClick={onDpDoubleClick}
            cursor={dpCursor}
            transform={dpTf}
            focused={activeCanvas === "dp"}
            label={`${realRoi ? "Σ CBED · ROI" : "CBED"} · k-space${dpTf.scale > 1 ? ` · ${dpTf.scale.toFixed(1)}×` : ""}`}
            corner={kx ? <>det {kx}×{ky}</> : <>det ?</>}
            hoverChip={
              realRoi
                ? `Σ over ${realRoi.row1 - realRoi.row0}×${realRoi.col1 - realRoi.col0}`
                : (haveScan ? `CBED at (${ix},${iy})` : "(no scan shape)")
            }
            scaleBarPx={dpScaleBar?.barPx ?? 0}
            scaleBarLabel={dpScaleBar?.label ?? ""}
            busy={dpBusy}
            error={dpErr}
            overlay={
              showRing && bfGeom ? (
                <Box sx={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
                  {/* Image-coord viewBox: matches the detector pixel grid so
                      the aperture marker + rings track the actual aperture
                      position in detector pixels. The center is at the
                      user-controlled `aperture` (defaulting to bfGeom). */}
                  <svg viewBox={`0 0 ${kx} ${ky}`} preserveAspectRatio="none"
                       style={{ width: "100%", height: "100%" }}>
                    {(() => {
                      const acx = aperture?.cx ?? bfGeom.cx;
                      const acy = aperture?.cy ?? bfGeom.cy;
                      const rOuter = ringOuter * bfGeom.r_bf;
                      const rInner = ringInner * bfGeom.r_bf;
                      const sw = Math.max(0.5, kx / 250);
                      const dash = `${sw * 2.5} ${sw * 1.5}`;
                      // Highlight the ring the cursor is hovering OR currently
                      // dragging. ~2× stroke width keeps dashes recognizable.
                      const activeSide: "inner" | "outer" | null =
                        ringDrag ? ringDrag.which : nearRing;
                      const innerHi = activeSide === "inner";
                      const outerHi = activeSide === "outer";
                      const swOuter = outerHi ? sw * 2 : sw;
                      const swInner = innerHi ? sw * 2 : sw;
                      // Show4DSTEM resize handle: ONE filled circle per ring
                      // at the SE diagonal (cos45°). Outer = green
                      // rgba(0,255,0,0.8); inner = cyan rgba(0,220,255,0.8).
                      // Hover/drag changes the fill to red rgba(255,100,100,1)
                      // for outer, cyan-bright rgba(0,200,255,1) for inner.
                      // Handle radius ≈ 6 CSS px → scale to detector-px via
                      // viewBox ratio (≈ kx / 80 for typical 192-px detectors
                      // displayed at 480 CSS px).
                      const COS45 = Math.SQRT1_2;
                      const handleR = Math.max(1.5, kx / 80);
                      const outerHx = acx + rOuter * COS45;
                      const outerHy = acy + rOuter * COS45;
                      const innerHx = acx + rInner * COS45;
                      const innerHy = acy + rInner * COS45;
                      const outerFill = outerHi ? "rgba(255,100,100,1)" : "rgba(0,255,0,0.85)";
                      const innerFill = innerHi ? "rgba(0,200,255,1)" : "rgba(0,220,255,0.85)";
                      const annulusPath = ringInner > 0.01
                        ? [
                            `M ${acx - rOuter} ${acy}`,
                            `a ${rOuter} ${rOuter} 0 1 0 ${rOuter * 2} 0`,
                            `a ${rOuter} ${rOuter} 0 1 0 ${-rOuter * 2} 0`,
                            `M ${acx - rInner} ${acy}`,
                            `a ${rInner} ${rInner} 0 1 0 ${rInner * 2} 0`,
                            `a ${rInner} ${rInner} 0 1 0 ${-rInner * 2} 0`,
                          ].join(" ")
                        : "";
                      // Shape-specific overlay branches. When the user picks
                      // square / rect / point on the dropdown, render those
                      // primitives instead of the legacy ring. The aperture
                      // crosshair is shared across every shape so the user
                      // always sees where DP arrow keys will land.
                      const handleFill = "rgba(255,215,0,0.95)";
                      const handleStroke = "rgba(0,0,0,0.7)";
                      const square = (() => {
                        const sx = shapeParams.cx, sy = shapeParams.cy;
                        const half = Math.max(1, shapeParams.half);
                        const seX = sx + half, seY = sy + half;
                        return (
                          <>
                            <rect x={sx - half} y={sy - half}
                                  width={half * 2} height={half * 2}
                                  fill="rgba(255,215,0,0.10)"
                                  stroke={ringStroke} strokeWidth={sw}
                                  strokeDasharray={dash} />
                            <circle cx={seX} cy={seY} r={handleR}
                                    fill={handleFill} stroke={handleStroke}
                                    strokeWidth={Math.max(0.4, sw * 0.5)} />
                          </>
                        );
                      });
                      const rectShape = (() => {
                        const r0 = shapeParams.row0, c0 = shapeParams.col0;
                        const r1 = shapeParams.row1, c1 = shapeParams.col1;
                        const corners = [
                          [c0, r0], [c1, r0], [c0, r1], [c1, r1],
                        ] as const;
                        return (
                          <>
                            <rect x={c0} y={r0}
                                  width={Math.max(1, c1 - c0)}
                                  height={Math.max(1, r1 - r0)}
                                  fill="rgba(255,215,0,0.10)"
                                  stroke={ringStroke} strokeWidth={sw}
                                  strokeDasharray={dash} />
                            {corners.map(([x, y], i) => (
                              <circle key={i} cx={x} cy={y} r={handleR}
                                      fill={handleFill} stroke={handleStroke}
                                      strokeWidth={Math.max(0.4, sw * 0.5)} />
                            ))}
                          </>
                        );
                      });
                      const pointShape = (() => {
                        const px = shapeParams.px, py = shapeParams.py;
                        const armLen = Math.max(3, kx / 32);
                        return (
                          <>
                            <line x1={px - armLen} y1={py} x2={px + armLen} y2={py}
                                  stroke={ringStroke} strokeWidth={sw * 1.5} />
                            <line x1={px} y1={py - armLen} x2={px} y2={py + armLen}
                                  stroke={ringStroke} strokeWidth={sw * 1.5} />
                            <circle cx={px} cy={py} r={handleR}
                                    fill={handleFill} stroke={handleStroke}
                                    strokeWidth={Math.max(0.4, sw * 0.5)} />
                          </>
                        );
                      });
                      const ringShape = (
                        <>
                          {mode !== "BF" && ringInner > 0.01 && (
                            <path d={annulusPath}
                                  fill="rgba(0,220,255,0.10)"
                                  fillRule="evenodd" />
                          )}
                          <circle cx={acx} cy={acy} r={rOuter} fill="none"
                                  stroke={ringStroke} strokeWidth={swOuter} strokeDasharray={dash} />
                          {ringInner > 0.01 && (
                            <circle cx={acx} cy={acy} r={rInner} fill="none"
                                    stroke={ringStroke} strokeWidth={swInner} strokeDasharray={dash} />
                          )}
                          {mode === "BF" && (
                            <circle cx={acx} cy={acy} r={rOuter} fill="rgba(255,215,0,0.10)" />
                          )}
                          <circle cx={acx} cy={acy} r={bfGeom.r_bf} fill="none"
                                  stroke="rgba(255,255,255,0.25)" strokeWidth={sw * 0.5}
                                  strokeDasharray={dash} />
                          <circle cx={outerHx} cy={outerHy} r={handleR}
                                  fill={outerFill}
                                  stroke="rgba(255,255,255,0.9)"
                                  strokeWidth={Math.max(0.4, sw * 0.5)} />
                          {ringInner > 0.01 && (
                            <circle cx={innerHx} cy={innerHy} r={handleR}
                                    fill={innerFill}
                                    stroke="rgba(255,255,255,0.9)"
                                    strokeWidth={Math.max(0.4, sw * 0.5)} />
                          )}
                        </>
                      );
                      const shapeBody =
                        dpShape === "square" ? square()
                        : dpShape === "rect" ? rectShape()
                        : dpShape === "point" ? pointShape()
                        : ringShape;
                      return (
                        <>
                          {shapeBody}
                          {/* Aperture-center crosshair so the user can see
                              where DP arrow keys will move next. */}
                          <line x1={acx - sw * 4} y1={acy} x2={acx + sw * 4} y2={acy}
                                stroke={colors.overlay.profileAccent} strokeWidth={sw} />
                          <line x1={acx} y1={acy - sw * 4} x2={acx} y2={acy + sw * 4}
                                stroke={colors.overlay.profileAccent} strokeWidth={sw} />
                        </>
                      );
                    })()}
                  </svg>
                </Box>
              ) : undefined
            }
          >
            <canvas ref={dpRef} />
          </CanvasCard>
          <PanelToolbar
            cmap={cmapDp}
            setCmap={setCmapDp}
            scale={dpScale}
            setScale={(s) => setDpScale && setDpScale(s)}
            powerExp={dpPowerExp}
            setPowerExp={setDpPowerExp}
            compact={compact}
            extras={setMaskDC ? (
              <Box
                component="button"
                onClick={() => setMaskDC(!maskDC)}
                title="Mask central 3×3 from DP percentile-clip stats (Show4DSTEM mask_dc, default ON)"
                data-testid="dp-mask-dc"
                sx={{ fontSize: fontSizes.sm, fontWeight: 600,
                      px: compact ? 0.75 : 1, py: compact ? 0.25 : 0.5, cursor: "pointer",
                      border: `1px solid ${maskDC ? colors.interactive.border : colors.border.default}`,
                      bgcolor: maskDC ? colors.interactive.bg : colors.bg.page,
                      color: maskDC ? colors.interactive.selectedText : colors.text.secondary,
                      borderRadius: radii.sm,
                      "&:hover": { bgcolor: maskDC ? colors.interactive.bg : colors.bg.hover } }}
              >
                mask DC
              </Box>
            ) : undefined}
          />
          <Histogram data={dpDisplayData} cmap={cmapDp} lo={dpClipLo} hi={dpClipHi}
                     setLo={setDpClipLo} setHi={setDpClipHi} label="intensity · DP"
                     compact={compact}
                     formatValue={(pct) => {
                       // The DP histogram operates on a transformed buffer
                       // (log1p / sqrt / identity). Convert the percentile
                       // back to the original units so the user sees raw
                       // intensity counts, not display values.
                       if (!dpLogRange) return "—";
                       const v = dpLogRange.mn + pct * (dpLogRange.mx - dpLogRange.mn);
                       const raw = dpInverseScale(v);
                       if (!Number.isFinite(raw)) return "—";
                       if (raw >= 1e6) return `${(raw / 1e6).toFixed(1)}M`;
                       if (raw >= 1e3) return `${(raw / 1e3).toFixed(1)}k`;
                       return raw.toFixed(0);
                     }}
          />
        </Box>

        {/* 3rd column: FFT panel. Mounted only when FFT toggle is ON, in
            the same row as VI + DP — matches Show4DSTEM dp_vi_fft 3-column
            layout (show4dstem.py:list_figure_templates → "publication_dp_vi_fft"). */}
        {fftOn && (
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
            <CanvasCard
              innerRef={fftFrameRef}
              cursor="default"
              label={`FFT · ${modeLabel}`}
              corner={fftData ? <>{fftData.w}×{fftData.h}</> : <>—</>}
              hoverChip="log magnitude"
              scaleBarPx={0}
              scaleBarLabel=""
              busy={fftOn && !fftData}
              skeleton={fftOn && !fftData}
              error={null}
            >
              <canvas ref={fftRef} />
            </CanvasCard>
            <PanelToolbar
              cmap={cmapFft}
              setCmap={setCmapFft}
              scale={fftScale}
              setScale={setFftScale}
              powerExp={fftPowerExp}
              setPowerExp={setFftPowerExp}
              compact={compact}
              extras={setFftWindow ? (
                <Box
                  component="button"
                  onClick={() => setFftWindow(!fftWindow)}
                  title="Apply Hann window before FFT (Show4DSTEM fft_window, default ON)"
                  data-testid="fft-window-toggle"
                  sx={{ fontSize: fontSizes.sm, fontWeight: 600,
                        px: compact ? 0.75 : 1, py: compact ? 0.25 : 0.5, cursor: "pointer",
                        border: `1px solid ${fftWindow ? colors.interactive.border : colors.border.default}`,
                        bgcolor: fftWindow ? colors.interactive.bg : colors.bg.page,
                        color: fftWindow ? colors.interactive.selectedText : colors.text.secondary,
                        borderRadius: radii.sm,
                        "&:hover": { bgcolor: fftWindow ? colors.interactive.bg : colors.bg.hover } }}
                >
                  Hann
                </Box>
              ) : undefined}
            />
            <Histogram data={fftDisplayData} cmap={cmapFft}
                       lo={fftClipLo} hi={fftClipHi}
                       setLo={setFftClipLo} setHi={setFftClipHi}
                       label="intensity · FFT" compact={compact} />
          </Box>
        )}
      </Box>
    </Box>
  );
}
