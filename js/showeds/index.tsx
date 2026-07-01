/// <reference types="@webgpu/types" />
import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import Typography from "@mui/material/Typography";
import { drawScaleBarHiDPI } from "../figure";
import { downloadBlob, extractBytes, extractFloat32, formatNumber, markWidgetNotebookDirty, preserveRestoredWidgetModelsOnSave } from "../format";
import { computeHistogramFromBytes, percentileClip, sliderRange } from "../stats";
import { useTheme } from "../theme";

type RoiShape = "rect" | "circle" | "ellipse";
type Roi = { row: number; col: number; height: number; width: number; shape?: RoiShape };
type DragMode = "roi-move" | "roi-resize" | "map-pan" | "band-move" | "band-left" | "band-right" | null;
type EdsLineHint = { element: string; line: string; family?: string; energy_keV: number; intensity?: number };
type SavedRoi = Roi & { name?: string };
type SavedBand = { name?: string; start: number; end: number };
type ExportPreset = { label?: string; mode?: string; downsample?: number; binning?: number; description?: string };
type PeriodicElement = { z: number; symbol: string; name: string; group: number; period: number };
type PanelResize = { mode: "map" | "spectrum"; x: number; y: number; width: number; height: number } | null;
type BandSliderDrag = { x: number; width: number; bandStart: number; bandEnd: number } | null;
type NumericArray = Float32Array | Float64Array | Uint32Array;
type PerfKind = "mapMs" | "spectrumMs" | "mapDrawMs" | "spectrumDrawMs";
type PerfStat = { last: number; avg: number; max: number };
type SidecarMeta = {
  format?: string;
  rows: number;
  cols: number;
  n_energy: number;
  n_events?: number;
  energy_prefix?: string;
  spatial_prefix?: string;
  channel_offsets?: string;
  channel_pixels?: string;
  pixel_offsets?: string;
  pixel_channels?: string;
};
type SidecarState = {
  meta: SidecarMeta;
  worker: Worker;
};
type SidecarWorkerResponse = {
  id?: number;
  type?: "map" | "spectrum" | "fetch-range";
  fetchId?: number;
  name?: string;
  start?: number;
  end?: number;
  width?: number;
  height?: number;
  viewRow?: number;
  viewCol?: number;
  viewRows?: number;
  viewCols?: number;
  buffer?: ArrayBuffer;
  ms?: number;
  backend?: string;
  aborted?: boolean;
  message?: string;
};
type SidecarPendingRequest = {
  resolve: (response: SidecarWorkerResponse) => void;
  reject: (error: Error) => void;
};
type SaveableWidgetModel = {
  save_changes?: () => unknown;
};
type ShowEDSWritableFile = {
  write: (data: BlobPart) => Promise<void>;
  close: () => Promise<void>;
};
type ShowEDSFileHandle = {
  createWritable: () => Promise<ShowEDSWritableFile>;
};
type ShowEDSSavePickerOptions = {
  suggestedName?: string;
  types?: { description: string; accept: Record<string, string[]> }[];
};
type ShowEDSWindow = Window & typeof globalThis & {
  showSaveFilePicker?: (options?: ShowEDSSavePickerOptions) => Promise<ShowEDSFileHandle>;
};
type ShowEDSPerfWindow = Window & typeof globalThis & {
  __quantemShowEDSPerf?: Record<PerfKind, number[]>;
  __quantemShowEDSBackend?: { map?: string; spectrum?: string };
};
type PreviewMap = {
  data: NumericArray;
  width: number;
  height: number;
  viewRow: number;
  viewCol: number;
  viewRows: number;
  viewCols: number;
};

const WORKGROUP = 64;
const INTERACTION_COMPUTE_INTERVAL_MS = 16;
const MAP_RASTER_DPR = 1;
const MIN_MAP_ZOOM = 1;
const MAX_MAP_ZOOM = 32;
const MIN_SPECTRUM_SPAN = 8;
const UI_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
const HISTOGRAM_LIGHT_COLORS = { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc", label: "#666", slider: "#1976d2" };
const HISTOGRAM_DARK_COLORS = { bg: "#1b1b1b", barActive: "#aaa", barInactive: "#555", border: "#3a3a3a", label: "#aaa", slider: "#5af" };
const HTML_EXPORT_OVERHEAD_BYTES = 700_000;
const ROI_SHAPE_LABELS: Record<RoiShape, string> = { rect: "Rect", circle: "Circle", ellipse: "Ellipse" };
const PERIODIC_ELEMENTS: PeriodicElement[] = [
  { z: 1, symbol: "H", name: "Hydrogen", group: 1, period: 1 },
  { z: 2, symbol: "He", name: "Helium", group: 18, period: 1 },
  { z: 3, symbol: "Li", name: "Lithium", group: 1, period: 2 },
  { z: 4, symbol: "Be", name: "Beryllium", group: 2, period: 2 },
  { z: 5, symbol: "B", name: "Boron", group: 13, period: 2 },
  { z: 6, symbol: "C", name: "Carbon", group: 14, period: 2 },
  { z: 7, symbol: "N", name: "Nitrogen", group: 15, period: 2 },
  { z: 8, symbol: "O", name: "Oxygen", group: 16, period: 2 },
  { z: 9, symbol: "F", name: "Fluorine", group: 17, period: 2 },
  { z: 10, symbol: "Ne", name: "Neon", group: 18, period: 2 },
  { z: 11, symbol: "Na", name: "Sodium", group: 1, period: 3 },
  { z: 12, symbol: "Mg", name: "Magnesium", group: 2, period: 3 },
  { z: 13, symbol: "Al", name: "Aluminium", group: 13, period: 3 },
  { z: 14, symbol: "Si", name: "Silicon", group: 14, period: 3 },
  { z: 15, symbol: "P", name: "Phosphorus", group: 15, period: 3 },
  { z: 16, symbol: "S", name: "Sulfur", group: 16, period: 3 },
  { z: 17, symbol: "Cl", name: "Chlorine", group: 17, period: 3 },
  { z: 18, symbol: "Ar", name: "Argon", group: 18, period: 3 },
  { z: 19, symbol: "K", name: "Potassium", group: 1, period: 4 },
  { z: 20, symbol: "Ca", name: "Calcium", group: 2, period: 4 },
  { z: 21, symbol: "Sc", name: "Scandium", group: 3, period: 4 },
  { z: 22, symbol: "Ti", name: "Titanium", group: 4, period: 4 },
  { z: 23, symbol: "V", name: "Vanadium", group: 5, period: 4 },
  { z: 24, symbol: "Cr", name: "Chromium", group: 6, period: 4 },
  { z: 25, symbol: "Mn", name: "Manganese", group: 7, period: 4 },
  { z: 26, symbol: "Fe", name: "Iron", group: 8, period: 4 },
  { z: 27, symbol: "Co", name: "Cobalt", group: 9, period: 4 },
  { z: 28, symbol: "Ni", name: "Nickel", group: 10, period: 4 },
  { z: 29, symbol: "Cu", name: "Copper", group: 11, period: 4 },
  { z: 30, symbol: "Zn", name: "Zinc", group: 12, period: 4 },
  { z: 31, symbol: "Ga", name: "Gallium", group: 13, period: 4 },
  { z: 32, symbol: "Ge", name: "Germanium", group: 14, period: 4 },
  { z: 33, symbol: "As", name: "Arsenic", group: 15, period: 4 },
  { z: 34, symbol: "Se", name: "Selenium", group: 16, period: 4 },
  { z: 35, symbol: "Br", name: "Bromine", group: 17, period: 4 },
  { z: 36, symbol: "Kr", name: "Krypton", group: 18, period: 4 },
  { z: 37, symbol: "Rb", name: "Rubidium", group: 1, period: 5 },
  { z: 38, symbol: "Sr", name: "Strontium", group: 2, period: 5 },
  { z: 39, symbol: "Y", name: "Yttrium", group: 3, period: 5 },
  { z: 40, symbol: "Zr", name: "Zirconium", group: 4, period: 5 },
  { z: 41, symbol: "Nb", name: "Niobium", group: 5, period: 5 },
  { z: 42, symbol: "Mo", name: "Molybdenum", group: 6, period: 5 },
  { z: 43, symbol: "Tc", name: "Technetium", group: 7, period: 5 },
  { z: 44, symbol: "Ru", name: "Ruthenium", group: 8, period: 5 },
  { z: 45, symbol: "Rh", name: "Rhodium", group: 9, period: 5 },
  { z: 46, symbol: "Pd", name: "Palladium", group: 10, period: 5 },
  { z: 47, symbol: "Ag", name: "Silver", group: 11, period: 5 },
  { z: 48, symbol: "Cd", name: "Cadmium", group: 12, period: 5 },
  { z: 49, symbol: "In", name: "Indium", group: 13, period: 5 },
  { z: 50, symbol: "Sn", name: "Tin", group: 14, period: 5 },
  { z: 51, symbol: "Sb", name: "Antimony", group: 15, period: 5 },
  { z: 52, symbol: "Te", name: "Tellurium", group: 16, period: 5 },
  { z: 53, symbol: "I", name: "Iodine", group: 17, period: 5 },
  { z: 54, symbol: "Xe", name: "Xenon", group: 18, period: 5 },
  { z: 55, symbol: "Cs", name: "Caesium", group: 1, period: 6 },
  { z: 56, symbol: "Ba", name: "Barium", group: 2, period: 6 },
  { z: 57, symbol: "La", name: "Lanthanum", group: 3, period: 6 },
  { z: 72, symbol: "Hf", name: "Hafnium", group: 4, period: 6 },
  { z: 73, symbol: "Ta", name: "Tantalum", group: 5, period: 6 },
  { z: 74, symbol: "W", name: "Tungsten", group: 6, period: 6 },
  { z: 75, symbol: "Re", name: "Rhenium", group: 7, period: 6 },
  { z: 76, symbol: "Os", name: "Osmium", group: 8, period: 6 },
  { z: 77, symbol: "Ir", name: "Iridium", group: 9, period: 6 },
  { z: 78, symbol: "Pt", name: "Platinum", group: 10, period: 6 },
  { z: 79, symbol: "Au", name: "Gold", group: 11, period: 6 },
  { z: 80, symbol: "Hg", name: "Mercury", group: 12, period: 6 },
  { z: 81, symbol: "Tl", name: "Thallium", group: 13, period: 6 },
  { z: 82, symbol: "Pb", name: "Lead", group: 14, period: 6 },
  { z: 83, symbol: "Bi", name: "Bismuth", group: 15, period: 6 },
  { z: 84, symbol: "Po", name: "Polonium", group: 16, period: 6 },
  { z: 85, symbol: "At", name: "Astatine", group: 17, period: 6 },
  { z: 86, symbol: "Rn", name: "Radon", group: 18, period: 6 },
  { z: 87, symbol: "Fr", name: "Francium", group: 1, period: 7 },
  { z: 88, symbol: "Ra", name: "Radium", group: 2, period: 7 },
  { z: 89, symbol: "Ac", name: "Actinium", group: 3, period: 7 },
  { z: 104, symbol: "Rf", name: "Rutherfordium", group: 4, period: 7 },
  { z: 105, symbol: "Db", name: "Dubnium", group: 5, period: 7 },
  { z: 106, symbol: "Sg", name: "Seaborgium", group: 6, period: 7 },
  { z: 107, symbol: "Bh", name: "Bohrium", group: 7, period: 7 },
  { z: 108, symbol: "Hs", name: "Hassium", group: 8, period: 7 },
  { z: 109, symbol: "Mt", name: "Meitnerium", group: 9, period: 7 },
  { z: 110, symbol: "Ds", name: "Darmstadtium", group: 10, period: 7 },
  { z: 111, symbol: "Rg", name: "Roentgenium", group: 11, period: 7 },
  { z: 112, symbol: "Cn", name: "Copernicium", group: 12, period: 7 },
  { z: 113, symbol: "Nh", name: "Nihonium", group: 13, period: 7 },
  { z: 114, symbol: "Fl", name: "Flerovium", group: 14, period: 7 },
  { z: 115, symbol: "Mc", name: "Moscovium", group: 15, period: 7 },
  { z: 116, symbol: "Lv", name: "Livermorium", group: 16, period: 7 },
  { z: 117, symbol: "Ts", name: "Tennessine", group: 17, period: 7 },
  { z: 118, symbol: "Og", name: "Oganesson", group: 18, period: 7 },
  { z: 58, symbol: "Ce", name: "Cerium", group: 4, period: 8 },
  { z: 59, symbol: "Pr", name: "Praseodymium", group: 5, period: 8 },
  { z: 60, symbol: "Nd", name: "Neodymium", group: 6, period: 8 },
  { z: 61, symbol: "Pm", name: "Promethium", group: 7, period: 8 },
  { z: 62, symbol: "Sm", name: "Samarium", group: 8, period: 8 },
  { z: 63, symbol: "Eu", name: "Europium", group: 9, period: 8 },
  { z: 64, symbol: "Gd", name: "Gadolinium", group: 10, period: 8 },
  { z: 65, symbol: "Tb", name: "Terbium", group: 11, period: 8 },
  { z: 66, symbol: "Dy", name: "Dysprosium", group: 12, period: 8 },
  { z: 67, symbol: "Ho", name: "Holmium", group: 13, period: 8 },
  { z: 68, symbol: "Er", name: "Erbium", group: 14, period: 8 },
  { z: 69, symbol: "Tm", name: "Thulium", group: 15, period: 8 },
  { z: 70, symbol: "Yb", name: "Ytterbium", group: 16, period: 8 },
  { z: 71, symbol: "Lu", name: "Lutetium", group: 17, period: 8 },
  { z: 90, symbol: "Th", name: "Thorium", group: 4, period: 9 },
  { z: 91, symbol: "Pa", name: "Protactinium", group: 5, period: 9 },
  { z: 92, symbol: "U", name: "Uranium", group: 6, period: 9 },
];
const MAP_WGSL = `
@group(0) @binding(0) var<storage,read> cube: array<u32>;
@group(0) @binding(1) var<storage,read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>; // rows, cols, n_energy, start
@group(0) @binding(3) var<uniform> q: vec4<u32>; // end, mode: 0=f32 bits, 1=uint16 counts, 2=uint32 counts

fn sample(idx: u32, mode: u32) -> f32 {
  if (mode == 1u) {
    let word = cube[idx >> 1u];
    let bits = select(word >> 16u, word & 0xffffu, (idx & 1u) == 0u);
    return f32(bits);
  }
  if (mode == 2u) {
    return f32(cube[idx]);
  }
  return bitcast<f32>(cube[idx]);
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pix = gid.x;
  let n_pix = p.x * p.y;
  if (pix >= n_pix) { return; }
  var sum = 0.0;
  let base = pix * p.z;
  for (var e = p.w; e < q.x; e = e + 1u) {
    sum = sum + sample(base + e, q.y);
  }
  out[pix] = sum;
}`;
const SPEC_WGSL = `
@group(0) @binding(0) var<storage,read> cube: array<u32>;
@group(0) @binding(1) var<storage,read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>; // rows, cols, n_energy, r0
@group(0) @binding(3) var<uniform> q: vec4<u32>; // c0, r1, c1, mode | shape<<4; shape 0=rect, 1=round/ellipse

fn sample(idx: u32, mode: u32) -> f32 {
  if (mode == 1u) {
    let word = cube[idx >> 1u];
    let bits = select(word >> 16u, word & 0xffffu, (idx & 1u) == 0u);
    return f32(bits);
  }
  if (mode == 2u) {
    return f32(cube[idx]);
  }
  return bitcast<f32>(cube[idx]);
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x;
  if (e >= p.z) { return; }
  let packed = q.w;
  let mode = packed & 15u;
  let shape = packed >> 4u;
  let cy = (f32(p.w) + f32(q.y)) * 0.5;
  let cx = (f32(q.x) + f32(q.z)) * 0.5;
  let ry = max(0.5, f32(q.y - p.w) * 0.5);
  let rx = max(0.5, f32(q.z - q.x) * 0.5);
  var sum = 0.0;
  for (var r = p.w; r < q.y; r = r + 1u) {
    for (var c = q.x; c < q.z; c = c + 1u) {
      if (shape == 1u) {
        let dy = f32(r) + 0.5 - cy;
        let dx = f32(c) + 0.5 - cx;
        if ((dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) > 1.0) { continue; }
      }
      sum = sum + sample((r * p.y + c) * p.z + e, mode);
    }
  }
  out[e] = sum;
}`;

function finiteRange(values: NumericArray): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return [0, 1];
  return [lo, hi];
}

function MapHistogram({
  data,
  vminPct,
  vmaxPct,
  onRangeChange,
  onRangePreview,
  onRangeCommit,
  dataMin,
  dataMax,
  theme,
  width = 128,
  height = 58,
}: {
  data: NumericArray | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  onRangePreview?: (min: number, max: number) => void;
  onRangeCommit?: (min: number, max: number) => void;
  dataMin: number;
  dataMax: number;
  theme: "light" | "dark";
  width?: number;
  height?: number;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const sliderRef = React.useRef<HTMLDivElement | null>(null);
  const minLabelRef = React.useRef<HTMLElement | null>(null);
  const maxLabelRef = React.useRef<HTMLElement | null>(null);
  const onRangeChangeRef = React.useRef(onRangeChange);
  const onRangePreviewRef = React.useRef(onRangePreview);
  const onRangeCommitRef = React.useRef(onRangeCommit);
  const pendingRangeRef = React.useRef<[number, number] | null>(null);
  const rangeRafRef = React.useRef<number | null>(null);
  const [liveRange, setLiveRange] = React.useState<[number, number]>(clampPercentPair(vminPct, vmaxPct));
  React.useEffect(() => { setLiveRange(clampPercentPair(vminPct, vmaxPct)); }, [vminPct, vmaxPct]);
  const [liveVminPct, liveVmaxPct] = liveRange;
  const bins = React.useMemo(
    () => computeHistogramFromBytes(data, 256, dataMin, dataMax),
    [data, dataMin, dataMax],
  );
  const colors = theme === "dark" ? HISTOGRAM_DARK_COLORS : HISTOGRAM_LIGHT_COLORS;
  const valueLabel = React.useCallback((pct: number) => {
    const val = dataMin + (pct / 100) * (dataMax - dataMin);
    return Math.abs(val) >= 1000 ? val.toExponential(1) : val.toFixed(1);
  }, [dataMax, dataMin]);
  const drawHistogram = React.useCallback((loPct: number, hiPct: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);
    const displayBins = 64;
    const binRatio = Math.max(1, Math.floor(bins.length / displayBins));
    const reducedBins: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) sum += bins[i * binRatio + j] || 0;
      reducedBins.push(sum / binRatio);
    }
    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;
    const vminBin = Math.floor((Math.max(0, Math.min(100, loPct)) / 100) * displayBins);
    const vmaxBin = Math.floor((Math.max(0, Math.min(100, hiPct)) / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      ctx.fillStyle = i >= vminBin && i <= vmaxBin ? colors.barActive : colors.barInactive;
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
    if (minLabelRef.current) minLabelRef.current.textContent = valueLabel(lo);
    if (maxLabelRef.current) maxLabelRef.current.textContent = valueLabel(hi);
    drawHistogram(lo, hi);
  }, [drawHistogram, valueLabel]);
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
  const beginRangeDrag = React.useCallback((event: React.MouseEvent, dragWidth: number, lo0: number, hi0: number) => {
    const startX = event.clientX;
    const span = Math.max(1, hi0 - lo0);
    const previousCursor = document.body.style.cursor;
    document.body.style.cursor = "grabbing";
    const onMove = (moveEvent: MouseEvent) => {
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
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = previousCursor;
      flushRangePreview();
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [applyRangePreview, emitRangePreview, flushRangePreview]);
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0, width, flexShrink: 0 }}>
      <Box sx={{ position: "relative", width, height: height + 6 }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, border: `1px solid ${colors.border}`, display: "block" }}
        aria-label="EDS map histogram"
      />
      <Box
        ref={sliderRef}
        onMouseDownCapture={(e) => {
          if ((e.target as HTMLElement).closest(".MuiSlider-thumb")) return;
          const rect = sliderRef.current?.getBoundingClientRect();
          if (!rect) return;
          const [lo, hi] = clampPercentPair(liveVminPct, liveVmaxPct);
          const pct = ((e.clientX - rect.left) / Math.max(1, rect.width)) * 100;
          if (pct < lo || pct > hi) return;
          const thumbGuardPct = Math.max(4, (10 / Math.max(1, rect.width)) * 100);
          if (Math.abs(pct - lo) <= thumbGuardPct || Math.abs(pct - hi) <= thumbGuardPct) return;
          beginRangeDrag(e, rect.width, lo, hi);
          e.preventDefault();
          e.stopPropagation();
          e.nativeEvent.stopImmediatePropagation();
        }}
        sx={{ position: "absolute", left: 0, top: height - 1, width, height: 8, display: "flex", alignItems: "flex-start", cursor: "grab", zIndex: 2, overflow: "visible" }}
      >
        <Slider
          value={liveRange}
          min={0}
          max={100}
          step={0.5}
          size="small"
          valueLabelDisplay="auto"
          valueLabelFormat={valueLabel}
          onChange={(_, v) => {
            if (!Array.isArray(v)) return;
            const [newMin, newMax] = v.map(Number);
            const next: [number, number] = [Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1)];
            setLiveRange(next);
            emitRangePreview(next[0], next[1]);
          }}
          onChangeCommitted={(_, v) => {
            if (!Array.isArray(v)) return;
            const [newMin, newMax] = v.map(Number);
            const next: [number, number] = [Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1)];
            setLiveRange(next);
            emitRangeCommit(next[0], next[1]);
          }}
          sx={{
            width,
            py: 0,
            position: "relative",
            zIndex: 3,
            overflow: "visible",
            "& .MuiSlider-rail": { height: 2, bgcolor: colors.barInactive, zIndex: 1 },
            "& .MuiSlider-track": { height: 2, cursor: "grab", bgcolor: colors.slider, zIndex: 2 },
            "& .MuiSlider-thumb": { width: 8, height: 8, bgcolor: colors.slider, zIndex: 4 },
            "& .MuiSlider-valueLabel": { fontSize: 10, px: 0.5, py: 0.25, zIndex: 5 },
          }}
        />
      </Box>
      </Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}>
        <Typography ref={minLabelRef} sx={{ fontSize: 8, fontFamily: "monospace", color: colors.label, lineHeight: 1 }}>{valueLabel(liveVminPct)}</Typography>
        <Typography ref={maxLabelRef} sx={{ fontSize: 8, fontFamily: "monospace", color: colors.label, lineHeight: 1 }}>{valueLabel(liveVmaxPct)}</Typography>
      </Box>
    </Box>
  );
}

function clampPercentPair(lo: number, hi: number): [number, number] {
  const a = Math.max(0, Math.min(100, lo));
  const b = Math.max(0, Math.min(100, hi));
  return a <= b ? [a, b] : [b, a];
}

function recordPerf(kind: PerfKind, ms: number) {
  const w = window as ShowEDSPerfWindow;
  const perf = w.__quantemShowEDSPerf || (w.__quantemShowEDSPerf = { mapMs: [], spectrumMs: [], mapDrawMs: [], spectrumDrawMs: [] });
  perf[kind] ||= [];
  perf[kind].push(ms);
  if (perf[kind].length > 120) perf[kind].splice(0, perf[kind].length - 120);
}

function recordBackend(kind: "map" | "spectrum", backend: string | undefined) {
  const w = window as ShowEDSPerfWindow;
  const state = w.__quantemShowEDSBackend || (w.__quantemShowEDSBackend = {});
  state[kind] = backend || "unknown";
}

function colorize(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, Math.max(0, 1.8 * x - 0.25)));
  const g = Math.round(255 * Math.sin(Math.PI * Math.min(1, Math.max(0, x))) ** 0.65);
  const b = Math.round(255 * Math.min(1, Math.max(0, 1.4 * (1 - x))));
  return [r, g, b];
}

function formatEnergy(v: number): string {
  return Number.isFinite(v) ? `${v.toFixed(2)} keV` : "";
}

function clampNumber(value: number, lo: number, hi: number): number {
  if (!Number.isFinite(value)) return lo;
  return Math.max(lo, Math.min(hi, value));
}

function normalizeRoiShape(value: unknown): RoiShape {
  const shape = String(value || "").trim().toLowerCase();
  if (shape === "circle") return "circle";
  if (shape === "ellipse" || shape === "oval") return "ellipse";
  return "rect";
}

function normalizeRoi(input: Roi, rows: number, cols: number): Roi {
  const shape = normalizeRoiShape(input.shape);
  let row = Math.max(0, Math.min(rows - 1, Math.round(input.row)));
  let col = Math.max(0, Math.min(cols - 1, Math.round(input.col)));
  let height = Math.max(1, Math.round(input.height));
  let width = Math.max(1, Math.round(input.width));
  if (shape === "circle") {
    let diameter = Math.max(1, Math.max(height, width));
    diameter = Math.min(diameter, rows, cols);
    row = Math.max(0, Math.min(rows - diameter, row));
    col = Math.max(0, Math.min(cols - diameter, col));
    return { row, col, height: diameter, width: diameter, shape };
  }
  height = Math.max(1, Math.min(rows - row, height));
  width = Math.max(1, Math.min(cols - col, width));
  return { row, col, height, width, shape };
}

function energyToIndex(axis: number[], value: number): number {
  if (!axis.length || !Number.isFinite(value)) return 0;
  if (axis.length === 1 || value <= axis[0]) return 0;
  const last = axis.length - 1;
  if (value >= axis[last]) return last;
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (axis[mid] <= value) lo = mid;
    else hi = mid;
  }
  const span = axis[hi] - axis[lo];
  if (!Number.isFinite(span) || span === 0) return lo;
  return lo + (value - axis[lo]) / span;
}

function makeExportFilename(title: string, rows: number, cols: number, nEnergy: number, mode: string, downsample?: number): string {
  let slug = (title || "showeds")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  while (slug.includes("__")) slug = slug.replace(/__/g, "_");
  if (!slug) slug = "showeds";
  const suffix = downsample ? `sum_binned_${downsample}x` : mode === "folder" ? "folder" : "single";
  return `${slug}_${rows}x${cols}x${nEnergy}_${suffix}.html`;
}

function formatBytes(bytes: number): string {
  let value = Math.max(0, bytes);
  const units = ["B", "KB", "MB", "GB", "TB"];
  for (const unit of units) {
    if (value < 1024 || unit === "TB") {
      if (unit === "B") return `${Math.round(value)} B`;
      if (value >= 100) return `${Math.round(value)} ${unit}`;
      if (value >= 10) return `${value.toFixed(1)} ${unit}`;
      return `${value.toFixed(2)} ${unit}`;
    }
    value /= 1024;
  }
  return `${Math.round(bytes)} B`;
}

function formatEstimatedHtmlSize(payloadBytes: number): string {
  const htmlBytes = Math.max(0, payloadBytes) * 4 / 3 + HTML_EXPORT_OVERHEAD_BYTES;
  return `~${formatBytes(htmlBytes)}`;
}

function estimateBinnedPayloadBytes(rows: number, cols: number, nEnergy: number, factor: number): number {
  const safeFactor = Math.max(1, Math.round(factor));
  const outRows = Math.floor(rows / safeFactor);
  const outCols = Math.floor(cols / safeFactor);
  const outEnergy = Math.floor(nEnergy / safeFactor);
  if (outRows <= 0 || outCols <= 0 || outEnergy <= 0) return 0;
  const cubeBytes = outRows * outCols * outEnergy * 4;
  const baseBytes = outRows * outCols * 4;
  const startupBytes = outRows * outCols * 4 + outEnergy * 4;
  return cubeBytes + baseBytes + startupBytes;
}

function isAbortLikeError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function lineLabel(line: EdsLineHint): string {
  const symbol = line.line
    .replace(/^Ka/, "Kα")
    .replace(/^Kb/, "Kβ")
    .replace(/^La/, "Lα")
    .replace(/^Lb/, "Lβ")
    .replace(/^Lg/, "Lγ")
    .replace(/^Ll$/, "Lℓ")
    .replace(/^Ln$/, "Lη")
    .replace(/^Ma$/, "Mα")
    .replace(/^Mb$/, "Mβ")
    .replace(/^Mg$/, "Mγ");
  return `${line.element} ${symbol}`;
}

function normalizeElementSymbol(value: unknown): string {
  const raw = String(value ?? "").trim();
  if (!raw) return "";
  return `${raw.slice(0, 1).toUpperCase()}${raw.slice(1).toLowerCase()}`;
}

function uniqueSymbols(values: unknown[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const value of values) {
    const symbol = normalizeElementSymbol(value);
    if (!symbol || seen.has(symbol)) continue;
    seen.add(symbol);
    out.push(symbol);
  }
  return out;
}

function extractStorageBytes(dataView: DataView | ArrayBuffer | Uint8Array, logicalBytes: number): Uint8Array | null {
  const bytes = extractBytes(dataView);
  if (bytes.length === 0) return null;
  const usableBytes = Math.max(0, Math.min(bytes.byteLength, logicalBytes));
  if (usableBytes === 0) return null;
  const storageBytes = Math.ceil(usableBytes / 4) * 4;
  const out = new Uint8Array(storageBytes);
  out.set(bytes.subarray(0, usableBytes));
  return out;
}

function copyExtractedBuffer(dataView: DataView | ArrayBuffer | Uint8Array): ArrayBuffer | null {
  const bytes = extractBytes(dataView);
  if (bytes.length === 0) return null;
  const out = new Uint8Array(bytes.byteLength);
  out.set(bytes);
  return out.buffer as ArrayBuffer;
}

function createSidecarWorker(): Worker {
  const source = `
let meta = null;
let baseUrl = "";
let prefixCache = new Map();
let spectrumCache = new Map();
let streamIndexPromise = null;
let channelOffsets = null;
let channelPixels = null;
let pixelOffsets = null;
let pixelChannels = null;
let streamGpu = null;
let streamGpuPromise = null;
let fetchRequestId = 0;
let fetchPending = new Map();
let activeMapController = null;
let activeSpectrumController = null;
const MAX_SPECTRUM_CACHE = 8192;
const WORKGROUP = 64;
const CLEAR_WGSL = [
  "@group(0) @binding(0) var<storage,read_write> out: array<atomic<u32>>;",
  "@group(0) @binding(1) var<uniform> p: vec4<u32>;",
  "@compute @workgroup_size(64)",
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
  "  let i = gid.x;",
  "  if (i >= p.x) { return; }",
  "  atomicStore(&out[i], 0u);",
  "}",
].join("\\n");
const STREAM_MAP_WGSL = [
  "@group(0) @binding(0) var<storage,read> channelPixels: array<u32>;",
  "@group(0) @binding(1) var<storage,read_write> out: array<atomic<u32>>;",
  "@group(0) @binding(2) var<uniform> p: vec4<u32>;",
  "@compute @workgroup_size(64)",
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
  "  let i = gid.x;",
  "  if (i >= p.y) { return; }",
  "  let pixel = channelPixels[p.x + i];",
  "  atomicAdd(&out[pixel], 1u);",
  "}",
].join("\\n");
const STREAM_MAP_PREVIEW_WGSL = [
  "struct Params {",
  "  p0: u32, count: u32, rows: u32, cols: u32,",
  "  outW: u32, outH: u32, pad0: u32, pad1: u32,",
  "  viewRow: f32, viewCol: f32, viewRows: f32, viewCols: f32,",
  "};",
  "@group(0) @binding(0) var<storage,read> channelPixels: array<u32>;",
  "@group(0) @binding(1) var<storage,read_write> out: array<atomic<u32>>;",
  "@group(0) @binding(2) var<uniform> p: Params;",
  "@compute @workgroup_size(64)",
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
  "  let i = gid.x;",
  "  if (i >= p.count) { return; }",
  "  let pixel = channelPixels[p.p0 + i];",
  "  let row = pixel / p.cols;",
  "  let col = pixel - row * p.cols;",
  "  let yf = (f32(row) + 0.5 - p.viewRow) / max(1e-6, p.viewRows);",
  "  let xf = (f32(col) + 0.5 - p.viewCol) / max(1e-6, p.viewCols);",
  "  if (xf < 0.0 || xf >= 1.0 || yf < 0.0 || yf >= 1.0) { return; }",
  "  let x = min(p.outW - 1u, u32(floor(xf * f32(p.outW))));",
  "  let y = min(p.outH - 1u, u32(floor(yf * f32(p.outH))));",
  "  atomicAdd(&out[y * p.outW + x], 1u);",
  "}",
].join("\\n");
const STREAM_SPECTRUM_WGSL = [
  "struct Params { rows: u32, cols: u32, nEnergy: u32, r0: u32, c0: u32, r1: u32, c1: u32, shape: u32 };",
  "@group(0) @binding(0) var<storage,read> pixelOffsets: array<u32>;",
  "@group(0) @binding(1) var<storage,read> pixelChannelWords: array<u32>;",
  "@group(0) @binding(2) var<storage,read_write> out: array<atomic<u32>>;",
  "@group(0) @binding(3) var<uniform> p: Params;",
  "fn channelAt(index: u32) -> u32 {",
  "  let word = pixelChannelWords[index >> 1u];",
  "  return select(word >> 16u, word & 0xffffu, (index & 1u) == 0u);",
  "}",
  "@compute @workgroup_size(64)",
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
  "  let width = p.c1 - p.c0;",
  "  let height = p.r1 - p.r0;",
  "  let local = gid.x;",
  "  if (local >= width * height) { return; }",
  "  let r = p.r0 + local / width;",
  "  let c = p.c0 + local - (local / width) * width;",
  "  if (p.shape > 0u) {",
  "    let cx = (f32(p.c0) + f32(p.c1)) * 0.5;",
  "    let cy = (f32(p.r0) + f32(p.r1)) * 0.5;",
  "    let rx = max(0.5, f32(width) * 0.5);",
  "    let ry = max(0.5, f32(height) * 0.5);",
  "    let dx = f32(c) + 0.5 - cx;",
  "    let dy = f32(r) + 0.5 - cy;",
  "    if ((dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) > 1.0) { return; }",
  "  }",
  "  let pixel = r * p.cols + c;",
  "  let start = pixelOffsets[pixel];",
  "  let end = pixelOffsets[pixel + 1u];",
  "  for (var k = start; k < end; k = k + 1u) {",
  "    atomicAdd(&out[channelAt(k)], 1u);",
  "  }",
  "}",
].join("\\n");

function isAbortError(error) {
  return error && (error.name === "AbortError" || error.message === "aborted");
}

function abortError() {
  const error = new Error("aborted");
  error.name = "AbortError";
  return error;
}

function sidecarFileUrl(name) {
  const base = new URL(baseUrl);
  const url = new URL(name, base);
  if (base.search && !url.search) url.search = base.search;
  return url.href;
}

async function proxyFetchRange(name, start, end) {
  return await new Promise((resolve, reject) => {
    const fetchId = ++fetchRequestId;
    fetchPending.set(fetchId, { resolve, reject });
    self.postMessage({ type: "fetch-range", fetchId, name, start, end });
  });
}

async function fetchRange(name, start, end, signal) {
  if (signal && signal.aborted) throw abortError();
  const expectedByteLength = end - start + 1;
  try {
    const resp = await fetch(sidecarFileUrl(name), {
      credentials: "include",
      headers: { Range: "bytes=" + start + "-" + end },
      signal,
    });
    if (resp.status === 206) return await resp.arrayBuffer();
    if (resp.status === 200) {
      const buffer = await resp.arrayBuffer();
      if (buffer.byteLength === expectedByteLength && start === 0) return buffer;
      if (buffer.byteLength >= end + 1) return buffer.slice(start, end + 1);
      throw new Error("range fetch returned " + buffer.byteLength + " bytes for a " + expectedByteLength + " byte request");
    }
    throw new Error("range fetch failed: " + resp.status);
  } catch (error) {
    if (isAbortError(error)) throw error;
    return await proxyFetchRange(name, start, end);
  }
}

async function fetchPrefixPlane(energyIndex, signal) {
  const idx = Math.max(0, Math.min(meta.n_energy, Math.round(energyIndex)));
  const cached = prefixCache.get(idx);
  if (cached) return cached;
  const planeValues = meta.rows * meta.cols;
  const start = idx * planeValues * 4;
  const end = start + planeValues * 4 - 1;
  const arr = new Uint32Array(await fetchRange(meta.energy_prefix, start, end, signal));
  prefixCache.set(idx, arr);
  if (prefixCache.size > 64) prefixCache.delete(prefixCache.keys().next().value);
  return arr;
}

async function fetchSpatialPrefixSpectrum(row, col, signal) {
  const r = Math.max(0, Math.min(meta.rows, Math.round(row)));
  const c = Math.max(0, Math.min(meta.cols, Math.round(col)));
  const key = r * (meta.cols + 1) + c;
  const cached = spectrumCache.get(key);
  if (cached) return cached;
  const start = key * meta.n_energy * 4;
  const end = start + meta.n_energy * 4 - 1;
  const arr = new Uint32Array(await fetchRange(meta.spatial_prefix, start, end, signal));
  spectrumCache.set(key, arr);
  if (spectrumCache.size > MAX_SPECTRUM_CACHE) spectrumCache.delete(spectrumCache.keys().next().value);
  return arr;
}

function isStreamSidecar() {
  return meta && meta.format === "quantem.widget.showeds.stream-sidecar.v1";
}

function roundUp4(value) {
  return Math.ceil(value / 4) * 4;
}

function paddedBytes(view) {
  const raw = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  if (raw.byteLength % 4 === 0) return raw;
  const out = new Uint8Array(roundUp4(raw.byteLength));
  out.set(raw);
  return out;
}

function destroyStreamGpu() {
  if (!streamGpu || streamGpu.disabled) {
    streamGpu = null;
    return;
  }
  for (const key of [
    "channelPixels",
    "pixelOffsets",
    "pixelChannels",
    "mapOut",
    "mapPreviewOut",
    "spectrumOut",
    "clearParams",
    "mapParams",
    "mapPreviewParams",
    "spectrumParams",
  ]) {
    try { streamGpu[key]?.destroy?.(); } catch {}
  }
  streamGpu = null;
}

async function ensureStreamIndexes() {
  if (channelOffsets && channelPixels && pixelOffsets && pixelChannels) return;
  if (streamIndexPromise) return await streamIndexPromise;
  streamIndexPromise = (async () => {
    const nEvents = Math.max(0, Math.round(meta.n_events || 0));
    const [channelOffsetsBuffer, channelPixelsBuffer, pixelOffsetsBuffer, pixelChannelsBuffer] = await Promise.all([
      fetchRange(meta.channel_offsets, 0, (meta.n_energy + 1) * 4 - 1),
      fetchRange(meta.channel_pixels, 0, nEvents * 4 - 1),
      fetchRange(meta.pixel_offsets, 0, (meta.rows * meta.cols + 1) * 4 - 1),
      fetchRange(meta.pixel_channels, 0, nEvents * 2 - 1),
    ]);
    channelOffsets = new Uint32Array(channelOffsetsBuffer);
    channelPixels = new Uint32Array(channelPixelsBuffer);
    pixelOffsets = new Uint32Array(pixelOffsetsBuffer);
    pixelChannels = new Uint16Array(pixelChannelsBuffer);
  })();
  try {
    await streamIndexPromise;
  } finally {
    streamIndexPromise = null;
  }
}

async function makeStreamGpuBuffer(device, view, usage) {
  const bytes = paddedBytes(view);
  const buffer = device.createBuffer({ size: bytes.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

async function readGpuUint32(device, source, count) {
  const byteLength = count * 4;
  const read = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, read, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await read.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(read.getMappedRange().slice(0));
  read.unmap();
  read.destroy();
  return out;
}

async function ensureStreamGpu() {
  if (streamGpu) return streamGpu.disabled ? null : streamGpu;
  if (streamGpuPromise) return await streamGpuPromise;
  streamGpuPromise = (async () => {
    try {
      await ensureStreamIndexes();
      if (!self.navigator?.gpu) throw new Error("WebGPU is not available in this worker");
      const adapter = await self.navigator.gpu.requestAdapter();
      if (!adapter) throw new Error("No WebGPU adapter found for sparse EDS data");
      const device = await adapter.requestDevice();
      const rows = Math.max(1, Math.round(meta.rows));
      const cols = Math.max(1, Math.round(meta.cols));
      const nEnergy = Math.max(1, Math.round(meta.n_energy));
      const channelPixelsBuffer = await makeStreamGpuBuffer(device, channelPixels, GPUBufferUsage.STORAGE);
      const pixelOffsetsBuffer = await makeStreamGpuBuffer(device, pixelOffsets, GPUBufferUsage.STORAGE);
      const pixelChannelsBuffer = await makeStreamGpuBuffer(device, pixelChannels, GPUBufferUsage.STORAGE);
      const mapOut = device.createBuffer({ size: rows * cols * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const spectrumOut = device.createBuffer({ size: nEnergy * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const clearParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const mapParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const mapPreviewParams = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const spectrumParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const clearPipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: CLEAR_WGSL }), entryPoint: "main" } });
      const mapPipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: STREAM_MAP_WGSL }), entryPoint: "main" } });
      const mapPreviewPipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: STREAM_MAP_PREVIEW_WGSL }), entryPoint: "main" } });
      const spectrumPipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: STREAM_SPECTRUM_WGSL }), entryPoint: "main" } });
      streamGpu = {
        device,
        channelPixels: channelPixelsBuffer,
        pixelOffsets: pixelOffsetsBuffer,
        pixelChannels: pixelChannelsBuffer,
        mapOut,
        mapPreviewOut: null,
        mapPreviewCount: 0,
        spectrumOut,
        clearParams,
        mapParams,
        mapPreviewParams,
        spectrumParams,
        clearPipeline,
        mapPipeline,
        mapPreviewPipeline,
        spectrumPipeline,
        mapClearBindGroup: device.createBindGroup({
          layout: clearPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: mapOut } },
            { binding: 1, resource: { buffer: clearParams } },
          ],
        }),
        spectrumClearBindGroup: device.createBindGroup({
          layout: clearPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: spectrumOut } },
            { binding: 1, resource: { buffer: clearParams } },
          ],
        }),
        mapBindGroup: device.createBindGroup({
          layout: mapPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: channelPixelsBuffer } },
            { binding: 1, resource: { buffer: mapOut } },
            { binding: 2, resource: { buffer: mapParams } },
          ],
        }),
        mapPreviewBindGroup: null,
        mapPreviewClearBindGroup: null,
        spectrumBindGroup: device.createBindGroup({
          layout: spectrumPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: pixelOffsetsBuffer } },
            { binding: 1, resource: { buffer: pixelChannelsBuffer } },
            { binding: 2, resource: { buffer: spectrumOut } },
            { binding: 3, resource: { buffer: spectrumParams } },
          ],
        }),
      };
      return streamGpu;
    } catch (error) {
      streamGpu = { disabled: true, error: error instanceof Error ? error.message : String(error) };
      return null;
    } finally {
      streamGpuPromise = null;
    }
  })();
  return await streamGpuPromise;
}

async function computeStreamMapGpu(start, end) {
  const gpu = await ensureStreamGpu();
  if (!gpu) return null;
  const rows = Math.max(1, Math.round(meta.rows));
  const cols = Math.max(1, Math.round(meta.cols));
  const p0 = channelOffsets[start];
  const p1 = channelOffsets[end];
  const count = Math.max(0, p1 - p0);
  gpu.device.queue.writeBuffer(gpu.clearParams, 0, new Uint32Array([rows * cols, 0, 0, 0]));
  gpu.device.queue.writeBuffer(gpu.mapParams, 0, new Uint32Array([p0, count, 0, 0]));
  const encoder = gpu.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(gpu.clearPipeline);
  pass.setBindGroup(0, gpu.mapClearBindGroup);
  pass.dispatchWorkgroups(Math.ceil((rows * cols) / WORKGROUP));
  if (count > 0) {
    pass.setPipeline(gpu.mapPipeline);
    pass.setBindGroup(0, gpu.mapBindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP));
  }
  pass.end();
  gpu.device.queue.submit([encoder.finish()]);
  return await readGpuUint32(gpu.device, gpu.mapOut, rows * cols);
}

function ensureStreamMapPreviewBuffers(gpu, outCount) {
  const needed = Math.max(1, Math.round(outCount));
  if (gpu.mapPreviewOut && gpu.mapPreviewCount >= needed) return;
  try { gpu.mapPreviewOut?.destroy?.(); } catch {}
  gpu.mapPreviewOut = gpu.device.createBuffer({ size: needed * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  gpu.mapPreviewCount = needed;
  gpu.mapPreviewClearBindGroup = gpu.device.createBindGroup({
    layout: gpu.clearPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpu.mapPreviewOut } },
      { binding: 1, resource: { buffer: gpu.clearParams } },
    ],
  });
  gpu.mapPreviewBindGroup = gpu.device.createBindGroup({
    layout: gpu.mapPreviewPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpu.channelPixels } },
      { binding: 1, resource: { buffer: gpu.mapPreviewOut } },
      { binding: 2, resource: { buffer: gpu.mapPreviewParams } },
    ],
  });
}

async function computeStreamMapPreviewGpu(start, end, viewRow, viewCol, viewRows, viewCols, outW, outH) {
  const gpu = await ensureStreamGpu();
  if (!gpu) return null;
  const rows = Math.max(1, Math.round(meta.rows));
  const cols = Math.max(1, Math.round(meta.cols));
  const width = Math.max(1, Math.round(outW));
  const height = Math.max(1, Math.round(outH));
  const outCount = width * height;
  ensureStreamMapPreviewBuffers(gpu, outCount);
  const p0 = channelOffsets[start];
  const p1 = channelOffsets[end];
  const count = Math.max(0, p1 - p0);
  const params = new ArrayBuffer(48);
  const u32 = new Uint32Array(params);
  const f32 = new Float32Array(params);
  u32[0] = p0;
  u32[1] = count;
  u32[2] = rows;
  u32[3] = cols;
  u32[4] = width;
  u32[5] = height;
  f32[8] = Number.isFinite(viewRow) ? viewRow : 0;
  f32[9] = Number.isFinite(viewCol) ? viewCol : 0;
  f32[10] = Math.max(1e-6, Number.isFinite(viewRows) ? viewRows : rows);
  f32[11] = Math.max(1e-6, Number.isFinite(viewCols) ? viewCols : cols);
  gpu.device.queue.writeBuffer(gpu.clearParams, 0, new Uint32Array([outCount, 0, 0, 0]));
  gpu.device.queue.writeBuffer(gpu.mapPreviewParams, 0, params);
  const encoder = gpu.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(gpu.clearPipeline);
  pass.setBindGroup(0, gpu.mapPreviewClearBindGroup);
  pass.dispatchWorkgroups(Math.ceil(outCount / WORKGROUP));
  if (count > 0) {
    pass.setPipeline(gpu.mapPreviewPipeline);
    pass.setBindGroup(0, gpu.mapPreviewBindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP));
  }
  pass.end();
  gpu.device.queue.submit([encoder.finish()]);
  return await readGpuUint32(gpu.device, gpu.mapPreviewOut, outCount);
}

async function computeStreamSpectrumGpu(r0, c0, r1, c1, shape) {
  const gpu = await ensureStreamGpu();
  if (!gpu) return null;
  const rows = Math.max(1, Math.round(meta.rows));
  const cols = Math.max(1, Math.round(meta.cols));
  const nEnergy = Math.max(1, Math.round(meta.n_energy));
  const width = Math.max(1, c1 - c0);
  const height = Math.max(1, r1 - r0);
  const shapeFlag = shape === "rect" ? 0 : 1;
  gpu.device.queue.writeBuffer(gpu.clearParams, 0, new Uint32Array([nEnergy, 0, 0, 0]));
  gpu.device.queue.writeBuffer(gpu.spectrumParams, 0, new Uint32Array([rows, cols, nEnergy, r0, c0, r1, c1, shapeFlag]));
  const encoder = gpu.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(gpu.clearPipeline);
  pass.setBindGroup(0, gpu.spectrumClearBindGroup);
  pass.dispatchWorkgroups(Math.ceil(nEnergy / WORKGROUP));
  pass.setPipeline(gpu.spectrumPipeline);
  pass.setBindGroup(0, gpu.spectrumBindGroup);
  pass.dispatchWorkgroups(Math.ceil((width * height) / WORKGROUP));
  pass.end();
  gpu.device.queue.submit([encoder.finish()]);
  return await readGpuUint32(gpu.device, gpu.spectrumOut, nEnergy);
}

function normaliseRoiShape(shape) {
  if (shape === "circle") return "circle";
  if (shape === "ellipse" || shape === "oval") return "ellipse";
  return "rect";
}

function roundSegmentForRow(row, r0, c0, r1, c1, shape) {
  const cx = (c0 + c1) * 0.5;
  const cy = (r0 + r1) * 0.5;
  const dy = row + 0.5 - cy;
  let half = 0;
  if (shape === "circle") {
    const radius = Math.max(r1 - r0, c1 - c0) * 0.5;
    if (Math.abs(dy) > radius) return null;
    half = Math.sqrt(Math.max(0, radius * radius - dy * dy));
  } else {
    const ry = Math.max(0.5, (r1 - r0) * 0.5);
    const rx = Math.max(0.5, (c1 - c0) * 0.5);
    const normY = dy / ry;
    if (Math.abs(normY) > 1) return null;
    half = rx * Math.sqrt(Math.max(0, 1 - normY * normY));
  }
  const start = Math.max(c0, Math.ceil(cx - half - 0.5));
  const end = Math.min(c1, Math.floor(cx + half - 0.5) + 1);
  return end > start ? [start, end] : null;
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  try {
    if (msg.type === "range-result") {
      const pending = fetchPending.get(msg.fetchId);
      if (pending) {
        fetchPending.delete(msg.fetchId);
        pending.resolve(msg.buffer);
      }
      return;
    }
    if (msg.type === "range-error") {
      const pending = fetchPending.get(msg.fetchId);
      if (pending) {
        fetchPending.delete(msg.fetchId);
        pending.reject(new Error(msg.message || "range fetch failed"));
      }
      return;
    }
    if (msg.type === "init") {
      meta = msg.meta;
      baseUrl = msg.baseUrl || "";
      prefixCache = new Map();
      spectrumCache = new Map();
      streamIndexPromise = null;
      destroyStreamGpu();
      streamGpuPromise = null;
      channelOffsets = null;
      channelPixels = null;
      pixelOffsets = null;
      pixelChannels = null;
      fetchPending = new Map();
      activeMapController?.abort();
      activeSpectrumController?.abort();
      activeMapController = null;
      activeSpectrumController = null;
      if (msg.streamBuffers) {
        channelOffsets = new Uint32Array(msg.streamBuffers.channelOffsets);
        channelPixels = new Uint32Array(msg.streamBuffers.channelPixels);
        pixelOffsets = new Uint32Array(msg.streamBuffers.pixelOffsets);
        pixelChannels = new Uint16Array(msg.streamBuffers.pixelChannels);
      }
      return;
    }
    if (!meta) throw new Error("folder data worker is not initialized");
    const t0 = performance.now();
    if (msg.type === "map") {
      activeMapController?.abort();
      const controller = new AbortController();
      activeMapController = controller;
      try {
        const s = Math.max(0, Math.min(meta.n_energy - 1, Math.round(msg.start)));
        const e = Math.max(s + 1, Math.min(meta.n_energy, Math.round(msg.end)));
        if (isStreamSidecar()) {
          await ensureStreamIndexes();
          if (controller.signal.aborted) throw abortError();
          const gpuOut = await computeStreamMapGpu(s, e);
          if (gpuOut) {
            if (controller.signal.aborted) throw abortError();
            self.postMessage({ id: msg.id, type: "map", buffer: gpuOut.buffer, ms: performance.now() - t0, backend: "webgpu-sparse" }, [gpuOut.buffer]);
            return;
          }
          const out = new Uint32Array(meta.rows * meta.cols);
          const p0 = channelOffsets[s];
          const p1 = channelOffsets[e];
          for (let i = p0; i < p1; i++) out[channelPixels[i]]++;
          if (controller.signal.aborted) throw abortError();
          self.postMessage({ id: msg.id, type: "map", buffer: out.buffer, ms: performance.now() - t0, backend: "worker-sparse" }, [out.buffer]);
          return;
        }
        const [lo, hi] = await Promise.all([
          fetchPrefixPlane(s, controller.signal),
          fetchPrefixPlane(e, controller.signal),
        ]);
        if (controller.signal.aborted) throw abortError();
        const out = new Uint32Array(meta.rows * meta.cols);
        for (let i = 0; i < out.length; i++) out[i] = hi[i] - lo[i];
        if (controller.signal.aborted) throw abortError();
        self.postMessage({ id: msg.id, type: "map", buffer: out.buffer, ms: performance.now() - t0 }, [out.buffer]);
      } catch (error) {
        if (isAbortError(error)) {
          self.postMessage({ id: msg.id, type: "map", aborted: true });
          return;
        }
        throw error;
      } finally {
        if (activeMapController === controller) activeMapController = null;
      }
      return;
    }
    if (msg.type === "map-preview") {
      activeMapController?.abort();
      const controller = new AbortController();
      activeMapController = controller;
      try {
        const s = Math.max(0, Math.min(meta.n_energy - 1, Math.round(msg.start)));
        const e = Math.max(s + 1, Math.min(meta.n_energy, Math.round(msg.end)));
        if (!isStreamSidecar()) throw new Error("map preview requires sparse stream data");
        await ensureStreamIndexes();
        if (controller.signal.aborted) throw abortError();
        const width = Math.max(1, Math.round(msg.width || 1));
        const height = Math.max(1, Math.round(msg.height || 1));
        const gpuOut = await computeStreamMapPreviewGpu(
          s,
          e,
          Number(msg.viewRow || 0),
          Number(msg.viewCol || 0),
          Number(msg.viewRows || meta.rows),
          Number(msg.viewCols || meta.cols),
          width,
          height,
        );
        if (gpuOut) {
          if (controller.signal.aborted) throw abortError();
          self.postMessage({
            id: msg.id,
            type: "map",
            buffer: gpuOut.buffer,
            width,
            height,
            viewRow: Number(msg.viewRow || 0),
            viewCol: Number(msg.viewCol || 0),
            viewRows: Number(msg.viewRows || meta.rows),
            viewCols: Number(msg.viewCols || meta.cols),
            ms: performance.now() - t0,
            backend: "webgpu-sparse-preview",
          }, [gpuOut.buffer]);
          return;
        }
        const out = new Uint32Array(width * height);
        const viewRow = Number(msg.viewRow || 0);
        const viewCol = Number(msg.viewCol || 0);
        const viewRows = Math.max(1e-6, Number(msg.viewRows || meta.rows));
        const viewCols = Math.max(1e-6, Number(msg.viewCols || meta.cols));
        const p0 = channelOffsets[s];
        const p1 = channelOffsets[e];
        for (let i = p0; i < p1; i++) {
          const pixel = channelPixels[i];
          const row = Math.floor(pixel / meta.cols);
          const col = pixel - row * meta.cols;
          const xf = (col + 0.5 - viewCol) / viewCols;
          const yf = (row + 0.5 - viewRow) / viewRows;
          if (xf < 0 || xf >= 1 || yf < 0 || yf >= 1) continue;
          const x = Math.min(width - 1, Math.floor(xf * width));
          const y = Math.min(height - 1, Math.floor(yf * height));
          out[y * width + x]++;
        }
        if (controller.signal.aborted) throw abortError();
        self.postMessage({
          id: msg.id,
          type: "map",
          buffer: out.buffer,
          width,
          height,
          viewRow,
          viewCol,
          viewRows,
          viewCols,
          ms: performance.now() - t0,
          backend: "worker-sparse-preview",
        }, [out.buffer]);
      } catch (error) {
        if (isAbortError(error)) {
          self.postMessage({ id: msg.id, type: "map", aborted: true });
          return;
        }
        throw error;
      } finally {
        if (activeMapController === controller) activeMapController = null;
      }
      return;
    }
    if (msg.type === "spectrum") {
      activeSpectrumController?.abort();
      const controller = new AbortController();
      activeSpectrumController = controller;
      try {
        const r0 = Math.max(0, Math.min(meta.rows - 1, Math.round(msg.row)));
        const c0 = Math.max(0, Math.min(meta.cols - 1, Math.round(msg.col)));
        const r1 = Math.max(r0 + 1, Math.min(meta.rows, r0 + Math.round(msg.height)));
        const c1 = Math.max(c0 + 1, Math.min(meta.cols, c0 + Math.round(msg.width)));
        const shape = normaliseRoiShape(msg.shape);
        if (isStreamSidecar()) {
          await ensureStreamIndexes();
          if (controller.signal.aborted) throw abortError();
          const gpuOut = await computeStreamSpectrumGpu(r0, c0, r1, c1, shape);
          if (gpuOut) {
            if (controller.signal.aborted) throw abortError();
            self.postMessage({ id: msg.id, type: "spectrum", buffer: gpuOut.buffer, ms: performance.now() - t0, backend: "webgpu-sparse" }, [gpuOut.buffer]);
            return;
          }
          const out = new Uint32Array(meta.n_energy);
          for (let r = r0; r < r1; r++) {
            if (controller.signal.aborted) throw abortError();
            let segments;
            if (shape === "rect") {
              segments = [[c0, c1]];
            } else {
              const segment = roundSegmentForRow(r, r0, c0, r1, c1, shape);
              segments = segment ? [segment] : [];
            }
            for (const [s0, s1] of segments) {
              const pixelStart = r * meta.cols + s0;
              const pixelEnd = r * meta.cols + s1;
              for (let pixel = pixelStart; pixel < pixelEnd; pixel++) {
                const o0 = pixelOffsets[pixel];
                const o1 = pixelOffsets[pixel + 1];
                for (let k = o0; k < o1; k++) out[pixelChannels[k]]++;
              }
            }
          }
          if (controller.signal.aborted) throw abortError();
          self.postMessage({ id: msg.id, type: "spectrum", buffer: out.buffer, ms: performance.now() - t0, backend: "worker-sparse" }, [out.buffer]);
          return;
        }
        if (shape !== "rect") {
          const out = new Uint32Array(meta.n_energy);
          const segments = [];
          for (let r = r0; r < r1; r++) {
            if (controller.signal.aborted) throw abortError();
            const segment = roundSegmentForRow(r, r0, c0, r1, c1, shape);
            if (!segment) continue;
            const [s0, s1] = segment;
            segments.push({ r, s0, s1 });
          }
          const batchSize = 128;
          for (let i = 0; i < segments.length; i += batchSize) {
            if (controller.signal.aborted) throw abortError();
            const rows = await Promise.all(segments.slice(i, i + batchSize).map(async ({ r, s0, s1 }) => {
              const [br, tr, bl, tl] = await Promise.all([
                fetchSpatialPrefixSpectrum(r + 1, s1, controller.signal),
                fetchSpatialPrefixSpectrum(r, s1, controller.signal),
                fetchSpatialPrefixSpectrum(r + 1, s0, controller.signal),
                fetchSpatialPrefixSpectrum(r, s0, controller.signal),
              ]);
              return [br, tr, bl, tl];
            }));
            if (controller.signal.aborted) throw abortError();
            for (const [br, tr, bl, tl] of rows) {
              for (let e = 0; e < meta.n_energy; e++) out[e] += br[e] - tr[e] - bl[e] + tl[e];
            }
          }
          if (controller.signal.aborted) throw abortError();
          self.postMessage({ id: msg.id, type: "spectrum", buffer: out.buffer, ms: performance.now() - t0 }, [out.buffer]);
          return;
        }
        const [br, tr, bl, tl] = await Promise.all([
          fetchSpatialPrefixSpectrum(r1, c1, controller.signal),
          fetchSpatialPrefixSpectrum(r0, c1, controller.signal),
          fetchSpatialPrefixSpectrum(r1, c0, controller.signal),
          fetchSpatialPrefixSpectrum(r0, c0, controller.signal),
        ]);
        if (controller.signal.aborted) throw abortError();
        const out = new Uint32Array(meta.n_energy);
        for (let e = 0; e < meta.n_energy; e++) out[e] = br[e] - tr[e] - bl[e] + tl[e];
        if (controller.signal.aborted) throw abortError();
        self.postMessage({ id: msg.id, type: "spectrum", buffer: out.buffer, ms: performance.now() - t0 }, [out.buffer]);
      } catch (error) {
        if (isAbortError(error)) {
          self.postMessage({ id: msg.id, type: "spectrum", aborted: true });
          return;
        }
        throw error;
      } finally {
        if (activeSpectrumController === controller) activeSpectrumController = null;
      }
      return;
    }
    throw new Error("unknown folder data worker request: " + msg.type);
  } catch (error) {
    self.postMessage({ id: msg.id, message: error instanceof Error ? error.message : String(error) });
  }
};
`;
  const blob = new Blob([source], { type: "text/javascript" });
  const url = URL.createObjectURL(blob);
  const worker = new Worker(url);
  URL.revokeObjectURL(url);
  return worker;
}

async function readBuffer(device: GPUDevice, source: GPUBuffer, byteLength: number): Promise<Float32Array> {
  const read = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(source, 0, read, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  await read.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  read.destroy();
  return out;
}

function ShowEDS() {
  const model = useModel();
  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);

  const [offlineForTheme] = useModelState<boolean>("_export_light");
  const { themeInfo, colors: tc } = useTheme(offlineForTheme);
  const themeColors = React.useMemo(() => {
    const isDark = themeInfo.theme === "dark";
    return {
      ...tc,
      mapBg: "#000",
      plotBg: isDark ? "#050505" : "#ffffff",
      plotGrid: isDark ? "#333" : "#d8d8d8",
      plotText: isDark ? "#ddd" : "#222",
      spectrumLine: isDark ? "#ffd54f" : "#8a5a00",
      lineHint: isDark ? "rgba(129, 212, 250, 0.72)" : "rgba(0, 102, 204, 0.66)",
      lineHintMuted: isDark ? "rgba(129, 212, 250, 0.22)" : "rgba(0, 102, 204, 0.20)",
      lineHintText: isDark ? "rgba(129, 212, 250, 0.92)" : "rgba(0, 80, 160, 0.92)",
      bandFill: isDark ? "rgba(255, 215, 0, 0.22)" : "rgba(255, 193, 7, 0.26)",
      roi: isDark ? "#00ff7f" : "#00b866",
      resize: isDark ? "#5af" : "#0066cc",
      hudBg: isDark ? "rgba(0,0,0,0.78)" : "rgba(245,245,245,0.96)",
      hudText: isDark ? "#d8f6ff" : "#1e4a5f",
      error: "#d32f2f",
      sliderPreview: "#1976d2",
      buttonText: isDark ? "#001018" : "#ffffff",
      hoverBg: isDark ? "#303030" : "#e8f2ff",
    };
  }, [tc, themeInfo.theme]);

  const [title] = useModelState<string>("title");
  const [rows] = useModelState<number>("n_rows");
  const [cols] = useModelState<number>("n_cols");
  const [nEnergy] = useModelState<number>("n_energy");
  const [cubeBytes] = useModelState<DataView>("cube_bytes");
  const [cubeDtype] = useModelState<string>("cube_dtype");
  const [baseBytes] = useModelState<DataView>("base_image_bytes");
  const [initialMapBytes] = useModelState<DataView>("initial_map_bytes");
  const [initialSpectrumBytes] = useModelState<DataView>("initial_spectrum_bytes");
  const [computeBackend] = useModelState<string>("compute_backend");
  const [sidecarUrl] = useModelState<string>("sidecar_url");
  const [sidecarMetaJson] = useModelState<string>("sidecar_meta_json");
  const [streamChannelOffsetsBytes] = useModelState<DataView>("stream_channel_offsets_bytes");
  const [streamChannelPixelsBytes] = useModelState<DataView>("stream_channel_pixels_bytes");
  const [streamPixelOffsetsBytes] = useModelState<DataView>("stream_pixel_offsets_bytes");
  const [streamPixelChannelsBytes] = useModelState<DataView>("stream_pixel_channels_bytes");
  const [energy] = useModelState<number[]>("energy_keV");
  const [bandStart, setBandStart] = useModelState<number>("band_start");
  const [bandEnd, setBandEnd] = useModelState<number>("band_end");
  const [roiRow, setRoiRow] = useModelState<number>("roi_row");
  const [roiCol, setRoiCol] = useModelState<number>("roi_col");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiShape, setRoiShape] = useModelState<RoiShape>("roi_shape");
  const [panelWidth, setPanelWidth] = useModelState<number>("panel_width_px");
  const [spectrumWidth, setSpectrumWidth] = useModelState<number>("spectrum_width_px");
  const [spectrumHeight, setSpectrumHeight] = useModelState<number>("spectrum_height_px");
  const [showControls] = useModelState<boolean>("show_controls");
  const [logSpectrum, setLogSpectrum] = useModelState<boolean>("log_spectrum");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [scaleBarVisible, setScaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [mapVminPct, setMapVminPct] = useModelState<number>("map_vmin_pct");
  const [mapVmaxPct, setMapVmaxPct] = useModelState<number>("map_vmax_pct");
  const [overlayOpacity, setOverlayOpacity] = useModelState<number>("overlay_opacity");
  const [mapZoom, setMapZoom] = useModelState<number>("map_zoom");
  const [mapViewRow, setMapViewRow] = useModelState<number>("map_view_row");
  const [mapViewCol, setMapViewCol] = useModelState<number>("map_view_col");
  const [spectrumViewStart, setSpectrumViewStart] = useModelState<number>("spectrum_view_start");
  const [spectrumViewEnd, setSpectrumViewEnd] = useModelState<number>("spectrum_view_end");
  const [elementLabel] = useModelState<string>("element_label");
  const [showLineHints] = useModelState<boolean>("show_line_hints");
  const [lineHints] = useModelState<EdsLineHint[]>("line_hints");
  const [selectedElements, setSelectedElements] = useModelState<string[]>("selected_elements");
  const [autoIdentify, setAutoIdentify] = useModelState<boolean>("auto_identify");
  const [showDebug, setShowDebug] = useModelState<boolean>("show_debug");
  const [backendSummary, setBackendSummary] = React.useState<{ map?: string; spectrum?: string }>({});
  const [debugControlVisible] = useModelState<boolean>("debug_control_visible");
  const [savedRois, setSavedRois] = useModelState<SavedRoi[]>("saved_rois");
  const [savedBands, setSavedBands] = useModelState<SavedBand[]>("saved_bands");
  const [exportPresets] = useModelState<ExportPreset[]>("export_presets");
  const [, setExportRequest] = useModelState<string>("export_request");
  const [exportStatus] = useModelState<string>("export_status");
  const [exportEnabled] = useModelState<boolean>("export_enabled");
  const [exportPayload] = useModelState<DataView>("export_payload");
  const [exportPayloadId] = useModelState<string>("export_payload_id");
  const [exportPayloadFilename] = useModelState<string>("export_filename");
  const [exportSidecarBytes] = useModelState<number>("export_sidecar_bytes");

  const mapCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const mapViewportRef = React.useRef<HTMLDivElement | null>(null);
  const mapOverlayCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const mapUiOverlayCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const specCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const specBandOverlayRef = React.useRef<HTMLDivElement | null>(null);
  const bandSliderRef = React.useRef<HTMLDivElement | null>(null);
  const bandSliderPreviewRef = React.useRef<HTMLDivElement | null>(null);
  const bandStatusRef = React.useRef<HTMLParagraphElement | null>(null);
  const gpuRef = React.useRef<{
    device: GPUDevice;
    cube: GPUBuffer;
    map: GPUBuffer;
    spectrum: GPUBuffer;
    mapParamsA: GPUBuffer;
    mapParamsB: GPUBuffer;
    specParamsA: GPUBuffer;
    specParamsB: GPUBuffer;
    mapPipeline: GPUComputePipeline;
    specPipeline: GPUComputePipeline;
    mapBindGroup: GPUBindGroup;
    specBindGroup: GPUBindGroup;
  } | null>(null);
  const [gpuError, setGpuError] = React.useState("");
  const [elementMap, setElementMap] = React.useState<NumericArray | null>(null);
  const [elementMapPreview, setElementMapPreview] = React.useState<PreviewMap | null>(null);
  const [roiSpectrum, setRoiSpectrum] = React.useState<NumericArray | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [localBand, setLocalBand] = React.useState<[number, number] | null>(null);
  const [localRoi, setLocalRoi] = React.useState<Roi | null>(null);
  const [localOverlayOpacity, setLocalOverlayOpacity] = React.useState<number | null>(null);
  const [drag, setDrag] = React.useState<{
    mode: DragMode;
    x: number;
    y: number;
    roi: Roi;
    bandStart: number;
    bandEnd: number;
    mapViewRow?: number;
    mapViewCol?: number;
  } | null>(null);
  const [panelResize, setPanelResize] = React.useState<PanelResize>(null);
  const [bandSliderDrag, setBandSliderDrag] = React.useState<BandSliderDrag>(null);
  const [exportMenuAnchor, setExportMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [elementMenuAnchor, setElementMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [roiMenuAnchor, setRoiMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [roiShapeMenuAnchor, setRoiShapeMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [bandMenuAnchor, setBandMenuAnchor] = React.useState<HTMLElement | null>(null);
  const imageRenderingStyle: React.CSSProperties["imageRendering"] = smooth ? "auto" : "pixelated";
  const [exportBusy, setExportBusy] = React.useState(false);
  const [localExportStatus, setLocalExportStatus] = React.useState("");
  const [perfTick, setPerfTick] = React.useState(0);
  const [fps, setFps] = React.useState(0);
  const mapRequestRef = React.useRef<{ start: number; end: number; interactive: boolean } | null>(null);
  const scheduleMapRef = React.useRef<((start: number, end: number, interactive?: boolean) => void) | null>(null);
  const mapRafRef = React.useRef<number | null>(null);
  const mapThrottleTimerRef = React.useRef<number | null>(null);
  const lastMapFlushRef = React.useRef(0);
  const mapRunningRef = React.useRef(false);
  const mapRerunRef = React.useRef(false);
  const specRequestRef = React.useRef<Roi | null>(null);
  const specRafRef = React.useRef<number | null>(null);
  const specThrottleTimerRef = React.useRef<number | null>(null);
  const lastSpecFlushRef = React.useRef(0);
  const specRunningRef = React.useRef(false);
  const specRerunRef = React.useRef(false);
  const requestIdRef = React.useRef(0);
  const sidecarRef = React.useRef<SidecarState | null>(null);
  const sidecarWorkerRequestIdRef = React.useRef(0);
  const sidecarPendingRef = React.useRef<Map<number, SidecarPendingRequest>>(new Map());
  const mapComputeSeqRef = React.useRef(0);
  const spectrumComputeSeqRef = React.useRef(0);
  const bandPersistTimerRef = React.useRef<number | null>(null);
  const roiPersistTimerRef = React.useRef<number | null>(null);
  const viewPersistTimerRef = React.useRef<number | null>(null);
  const pendingLocalBandRef = React.useRef<[number, number] | null>(null);
  const localBandRafRef = React.useRef<number | null>(null);
  const pendingBandPersistRef = React.useRef<[number, number] | null>(null);
  const pendingRoiPersistRef = React.useRef<Roi | null>(null);
  const pendingExportRef = React.useRef<{
    id: string;
    filename: string;
    mode: string;
    downsample?: number;
    handle: ShowEDSFileHandle | null;
  } | null>(null);

  const saveWidgetChanges = React.useCallback(() => {
    const saveChanges = (model as SaveableWidgetModel | null)?.save_changes;
    markWidgetNotebookDirty(model);
    if (typeof saveChanges !== "function") return;
    window.setTimeout(() => {
      saveChanges.call(model);
      markWidgetNotebookDirty(model);
    }, 0);
  }, [model]);

  const queueViewPersist = React.useCallback(() => {
    markWidgetNotebookDirty(model);
    if (viewPersistTimerRef.current != null) window.clearTimeout(viewPersistTimerRef.current);
    viewPersistTimerRef.current = window.setTimeout(() => {
      viewPersistTimerRef.current = null;
      saveWidgetChanges();
    }, 180);
  }, [model, saveWidgetChanges]);

  const flushLocalBandPreview = React.useCallback(() => {
    if (localBandRafRef.current != null) {
      window.cancelAnimationFrame(localBandRafRef.current);
      localBandRafRef.current = null;
    }
    const pending = pendingLocalBandRef.current;
    pendingLocalBandRef.current = null;
    if (pending) setLocalBand(pending);
  }, [setLocalBand]);

  const setLocalBandPreview = React.useCallback((value: [number, number], immediate = true) => {
    if (immediate) {
      if (localBandRafRef.current != null) {
        window.cancelAnimationFrame(localBandRafRef.current);
        localBandRafRef.current = null;
      }
      pendingLocalBandRef.current = null;
      setLocalBand(value);
      return;
    }
    pendingLocalBandRef.current = value;
    if (localBandRafRef.current == null) {
      localBandRafRef.current = window.requestAnimationFrame(() => {
        localBandRafRef.current = null;
        const pending = pendingLocalBandRef.current;
        pendingLocalBandRef.current = null;
        if (pending) setLocalBand(pending);
      });
    }
  }, [setLocalBand]);

  const recordWidgetPerf = React.useCallback((kind: PerfKind, ms: number) => {
    recordPerf(kind, ms);
    if (showDebug) setPerfTick((value) => (value + 1) % 100000);
  }, [showDebug]);

  const recordComputeBackend = React.useCallback((kind: "map" | "spectrum", backend: string | undefined) => {
    recordBackend(kind, backend);
    if (!backend) return;
    setBackendSummary((current) => (
      current[kind] === backend ? current : { ...current, [kind]: backend }
    ));
  }, []);

  React.useEffect(() => {
    if (!showDebug) return;
    let raf = 0;
    let frames = 0;
    let last = performance.now();
    const tick = () => {
      frames += 1;
      const now = performance.now();
      if (now - last >= 500) {
        setFps((frames * 1000) / (now - last));
        frames = 0;
        last = now;
      }
      raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [showDebug]);

  const commitOverlayOpacity = React.useCallback((value: number) => {
    const opacity = Math.max(0, Math.min(1, value));
    setLocalOverlayOpacity(opacity);
    setOverlayOpacity(opacity);
    saveWidgetChanges();
  }, [saveWidgetChanges, setOverlayOpacity]);

  const commitDebug = React.useCallback((value: boolean) => {
    setShowDebug(value);
    saveWidgetChanges();
  }, [saveWidgetChanges, setShowDebug]);

  const commitScaleBarVisible = React.useCallback((value: boolean) => {
    setScaleBarVisible(value);
    saveWidgetChanges();
  }, [saveWidgetChanges, setScaleBarVisible]);

  const cube = React.useMemo(() => {
    const count = rows * cols * nEnergy;
    return extractStorageBytes(cubeBytes, count * (cubeDtype === "uint16" ? 2 : 4));
  }, [cubeBytes, cubeDtype, rows, cols, nEnergy]);
  const base = React.useMemo(() => extractFloat32(baseBytes, rows * cols), [baseBytes, rows, cols]);
  const initialMap = React.useMemo(() => extractFloat32(initialMapBytes, rows * cols), [initialMapBytes, rows, cols]);
  const initialSpectrum = React.useMemo(() => extractFloat32(initialSpectrumBytes, nEnergy), [initialSpectrumBytes, nEnergy]);
  const safeSavedRois = React.useMemo(() => (Array.isArray(savedRois) ? savedRois : [])
    .map((item, index) => ({
      name: String(item?.name || `ROI ${index + 1}`),
      ...normalizeRoi({
        row: Number(item?.row ?? 0),
        col: Number(item?.col ?? 0),
        height: Number(item?.height ?? 1),
        width: Number(item?.width ?? 1),
        shape: normalizeRoiShape(item?.shape),
      }, rows, cols),
    })), [cols, rows, savedRois]);
  const safeSavedBands = React.useMemo(() => (Array.isArray(savedBands) ? savedBands : [])
    .map((item, index) => {
      const start = Math.max(0, Math.min(nEnergy - 1, Math.round(Number(item?.start ?? 0))));
      const end = Math.max(start + 1, Math.min(nEnergy, Math.round(Number(item?.end ?? start + 1))));
      return { name: String(item?.name || `Band ${index + 1}`), start, end };
    }), [nEnergy, savedBands]);
  const isKernelBackend = computeBackend === "kernel";
  const isSidecarBackend = computeBackend === "sidecar";
  const isStreamBackend = computeBackend === "stream";
  const isSparseWorkerBackend = isSidecarBackend || isStreamBackend;
  const size = Math.max(180, Math.round(panelWidth || 420));
  const specW = Math.max(320, Math.round(spectrumWidth || size + 220));
  const specH = Math.max(140, Math.round(spectrumHeight || Math.round(size * 0.58)));
  const modelRoi: Roi = normalizeRoi({ row: roiRow, col: roiCol, height: roiHeight, width: roiWidth, shape: normalizeRoiShape(roiShape) }, rows, cols);
  const roi: Roi = localRoi ?? modelRoi;
  const rawBandStart = localBand?.[0] ?? bandStart;
  const rawBandEnd = localBand?.[1] ?? bandEnd;
  const bandLo = Math.max(0, Math.min(nEnergy - 1, Math.round(rawBandStart)));
  const bandHi = Math.max(bandLo + 1, Math.min(nEnergy, Math.round(rawBandEnd)));
  const displayOverlayOpacity = Math.max(0, Math.min(1, localOverlayOpacity ?? overlayOpacity));
  const bandEnergyLo = Math.min(energy[bandLo] ?? 0, energy[Math.max(bandLo, bandHi - 1)] ?? 0);
  const bandEnergyHi = Math.max(energy[bandLo] ?? 0, energy[Math.max(bandLo, bandHi - 1)] ?? 0);
  const mapView = React.useMemo(() => {
    const zoom = clampNumber(mapZoom || 1, MIN_MAP_ZOOM, MAX_MAP_ZOOM);
    const viewRows = rows / zoom;
    const viewCols = cols / zoom;
    const row = clampNumber(mapViewRow || 0, 0, Math.max(0, rows - viewRows));
    const col = clampNumber(mapViewCol || 0, 0, Math.max(0, cols - viewCols));
    return { zoom, row, col, rows: viewRows, cols: viewCols };
  }, [cols, mapViewCol, mapViewRow, mapZoom, rows]);
  const setMapView = React.useCallback((zoom: number, row: number, col: number, persist = true) => {
    const nextZoom = clampNumber(zoom, MIN_MAP_ZOOM, MAX_MAP_ZOOM);
    const nextRows = rows / nextZoom;
    const nextCols = cols / nextZoom;
    const nextRow = clampNumber(row, 0, Math.max(0, rows - nextRows));
    const nextCol = clampNumber(col, 0, Math.max(0, cols - nextCols));
    setMapZoom(nextZoom);
    setMapViewRow(nextRow);
    setMapViewCol(nextCol);
    if (persist) queueViewPersist();
  }, [cols, queueViewPersist, rows, setMapViewCol, setMapViewRow, setMapZoom]);
  const mapScreenToImage = React.useCallback((x: number, y: number) => ({
    row: mapView.row + (y / Math.max(1, size)) * mapView.rows,
    col: mapView.col + (x / Math.max(1, size)) * mapView.cols,
  }), [mapView, size]);
  const mapImageToScreen = React.useCallback((row: number, col: number) => ({
    x: ((col - mapView.col) / Math.max(1e-9, mapView.cols)) * size,
    y: ((row - mapView.row) / Math.max(1e-9, mapView.rows)) * size,
  }), [mapView, size]);
  const mapLayerStyle = React.useMemo<React.CSSProperties>(() => ({
    position: "absolute" as const,
    left: 0,
    top: 0,
    width: size,
    height: size,
    display: "block",
    imageRendering: imageRenderingStyle,
    willChange: "contents",
  }), [imageRenderingStyle, size]);
  const spectrumView = React.useMemo(() => {
    let start = Number.isFinite(spectrumViewStart) ? Number(spectrumViewStart) : 0;
    let end = Number.isFinite(spectrumViewEnd) && Number(spectrumViewEnd) > start
      ? Number(spectrumViewEnd)
      : nEnergy;
    const minSpan = Math.min(MIN_SPECTRUM_SPAN, Math.max(1, nEnergy));
    start = clampNumber(start, 0, Math.max(0, nEnergy - minSpan));
    end = clampNumber(end, start + minSpan, nEnergy);
    return { start, end, span: Math.max(minSpan, end - start) };
  }, [nEnergy, spectrumViewEnd, spectrumViewStart]);
  const setSpectrumView = React.useCallback((start: number, end: number, persist = true) => {
    const minSpan = Math.min(MIN_SPECTRUM_SPAN, Math.max(1, nEnergy));
    const span = clampNumber(end - start, minSpan, Math.max(minSpan, nEnergy));
    const nextStart = clampNumber(start, 0, Math.max(0, nEnergy - span));
    const nextEnd = clampNumber(nextStart + span, nextStart + minSpan, nEnergy);
    setSpectrumViewStart(nextStart);
    setSpectrumViewEnd(nextEnd);
    if (persist) queueViewPersist();
  }, [nEnergy, queueViewPersist, setSpectrumViewEnd, setSpectrumViewStart]);
  const indexToSpecX = React.useCallback((index: number, width = specW) => {
    const padL = 54;
    const padR = 14;
    const plotW = Math.max(1, width - padL - padR);
    return padL + ((index - spectrumView.start) / Math.max(1e-9, spectrumView.span)) * plotW;
  }, [specW, spectrumView]);
  const specXToIndex = React.useCallback((x: number, width = specW) => {
    const padL = 54;
    const padR = 14;
    const plotW = Math.max(1, width - padL - padR);
    return spectrumView.start + ((x - padL) / plotW) * spectrumView.span;
  }, [specW, spectrumView]);
  const positionBandPreview = React.useCallback((start: number, end: number, sliderWidth?: number): [number, number] => {
    const s = Math.max(0, Math.min(nEnergy - 1, Math.round(start)));
    const e = Math.max(s + 1, Math.min(nEnergy, Math.round(end)));
    const specEl = specBandOverlayRef.current;
    if (specEl) {
      const x0 = indexToSpecX(s);
      const x1 = indexToSpecX(e - 1);
      specEl.style.transform = `translateX(${x0}px)`;
      specEl.style.width = `${Math.max(2, x1 - x0)}px`;
    }
    const sliderEl = bandSliderPreviewRef.current;
    if (sliderEl) {
      const width = Math.max(1, sliderWidth ?? bandSliderRef.current?.clientWidth ?? 260);
      const x0 = (s / Math.max(1, nEnergy)) * width;
      const x1 = (e / Math.max(1, nEnergy)) * width;
      sliderEl.style.transform = `translate(${x0}px, -50%)`;
      sliderEl.style.width = `${Math.max(2, x1 - x0)}px`;
    }
    const statusEl = bandStatusRef.current;
    if (statusEl) {
      const e0 = energy[s] ?? 0;
      const e1 = energy[Math.max(s, e - 1)] ?? 0;
      let sum = 0;
      if (roiSpectrum) {
        for (let i = s; i < e; i++) sum += roiSpectrum[i] || 0;
      }
      let candidates = "";
      if (showLineHints && autoIdentify && Array.isArray(lineHints)) {
        const step = Math.abs((energy[Math.min(nEnergy - 1, s + 1)] ?? e1) - (energy[s] ?? e0)) || 0.02;
        const lo = Math.min(e0, e1) - step * 0.65;
        const hi = Math.max(e0, e1) + step * 0.65;
        candidates = lineHints
          .filter((line) => Number.isFinite(line.energy_keV) && line.energy_keV >= lo && line.energy_keV <= hi)
          .sort((a, b) => (b.intensity ?? 0) - (a.intensity ?? 0))
          .slice(0, 4)
          .map(lineLabel)
          .join(", ");
      }
      statusEl.textContent = `Band ${s}-${e - 1}: ${formatEnergy(e0)} - ${formatEnergy(e1)}; ROI band counts ${formatNumber(sum, 2)}${candidates ? `; candidates ${candidates}` : ""}`;
    }
    return [s, e];
  }, [autoIdentify, energy, indexToSpecX, lineHints, nEnergy, roiSpectrum, showLineHints]);
  const previewCenterBand = React.useCallback((start: number, end: number, sliderWidth?: number) => {
    const [s, e] = positionBandPreview(start, end, sliderWidth);
    pendingLocalBandRef.current = [s, e];
    pendingBandPersistRef.current = [s, e];
    mapRequestRef.current = { start: s, end: e, interactive: true };
    scheduleMapRef.current?.(s, e, true);
  }, [positionBandPreview]);
  React.useLayoutEffect(() => {
    positionBandPreview(bandLo, bandHi);
  }, [bandHi, bandLo, positionBandPreview]);
  const mapDataRange = React.useMemo(() => elementMap ? finiteRange(elementMap) : [0, 1] as [number, number], [elementMap]);
  const mapDisplayRange = React.useMemo(() => {
    const range = sliderRange(mapDataRange[0], mapDataRange[1], mapVminPct, mapVmaxPct);
    if (!Number.isFinite(range.vmin) || !Number.isFinite(range.vmax) || range.vmax <= range.vmin) return mapDataRange;
    return [range.vmin, range.vmax] as [number, number];
  }, [mapDataRange, mapVmaxPct, mapVminPct]);
  const mapPercentRange = React.useCallback((loPct: number, hiPct: number): [number, number] => {
    const range = sliderRange(mapDataRange[0], mapDataRange[1], loPct, hiPct);
    if (!Number.isFinite(range.vmin) || !Number.isFinite(range.vmax) || range.vmax <= range.vmin) return mapDataRange;
    return [range.vmin, range.vmax];
  }, [mapDataRange]);
  const candidateLines = React.useMemo(() => {
    if (!showLineHints || !autoIdentify || !Array.isArray(lineHints)) return [];
    const step = Math.abs((energy[Math.min(nEnergy - 1, bandLo + 1)] ?? bandEnergyHi) - (energy[bandLo] ?? bandEnergyLo)) || 0.02;
    const lo = bandEnergyLo - step * 0.65;
    const hi = bandEnergyHi + step * 0.65;
    return lineHints
      .filter((line) => Number.isFinite(line.energy_keV) && line.energy_keV >= lo && line.energy_keV <= hi)
      .sort((a, b) => (b.intensity ?? 0) - (a.intensity ?? 0))
      .slice(0, 4);
  }, [autoIdentify, bandEnergyHi, bandEnergyLo, bandLo, energy, lineHints, nEnergy, showLineHints]);
  const candidateText = candidateLines.map(lineLabel).join(", ");
  const safeSelectedElements = React.useMemo(() => uniqueSymbols(Array.isArray(selectedElements) ? selectedElements : []), [selectedElements]);
  const selectedElementSet = React.useMemo(() => new Set(safeSelectedElements), [safeSelectedElements]);
  const hasEmbeddedStream = Boolean(
    !sidecarUrl
    && sidecarMetaJson
    && extractBytes(streamChannelOffsetsBytes).length > 0
    && extractBytes(streamChannelPixelsBytes).length > 0
    && extractBytes(streamPixelOffsetsBytes).length > 0
    && extractBytes(streamPixelChannelsBytes).length > 0,
  );
  const elementsWithLines = React.useMemo(() => {
    const out = new Set<string>();
    if (Array.isArray(lineHints)) {
      for (const line of lineHints) out.add(normalizeElementSymbol(line.element));
    }
    return out;
  }, [lineHints]);
  const selectedLineHints = React.useMemo(() => {
    if (!Array.isArray(lineHints) || selectedElementSet.size === 0) return [];
    return lineHints
      .filter((line) => selectedElementSet.has(normalizeElementSymbol(line.element)))
      .sort((a, b) => (a.energy_keV - b.energy_keV) || ((b.intensity ?? 0) - (a.intensity ?? 0)));
  }, [lineHints, selectedElementSet]);
  const suggestedLines = React.useMemo(() => {
    const source = selectedLineHints.length > 0 ? selectedLineHints : candidateLines;
    const visibleLo = energy[Math.max(0, Math.floor(spectrumView.start))] ?? bandEnergyLo;
    const visibleHi = energy[Math.min(nEnergy - 1, Math.ceil(spectrumView.end) - 1)] ?? bandEnergyHi;
    return source
      .filter((line) => Number.isFinite(line.energy_keV) && line.energy_keV >= visibleLo && line.energy_keV <= visibleHi)
      .sort((a, b) => (b.intensity ?? 0) - (a.intensity ?? 0))
      .slice(0, 10);
  }, [bandEnergyHi, bandEnergyLo, candidateLines, energy, nEnergy, selectedLineHints, spectrumView.end, spectrumView.start]);
  const autoElementScores = React.useMemo(() => {
    if (!autoIdentify || !roiSpectrum || !Array.isArray(lineHints) || energy.length < 2) return [];
    const lo = Math.min(bandEnergyLo, bandEnergyHi);
    const hi = Math.max(bandEnergyLo, bandEnergyHi);
    const byElement = new Map<string, { symbol: string; score: number; lines: EdsLineHint[] }>();
    for (const line of lineHints) {
      if (!Number.isFinite(line.energy_keV) || line.energy_keV < lo || line.energy_keV > hi) continue;
      const idx = Math.max(0, Math.min(nEnergy - 1, Math.round(energyToIndex(energy, line.energy_keV))));
      const value = Math.max(0, Number(roiSpectrum[idx] ?? 0));
      const score = value * Math.max(0.05, line.intensity ?? 0.1);
      const symbol = normalizeElementSymbol(line.element);
      const prev = byElement.get(symbol) || { symbol, score: 0, lines: [] };
      prev.score += score;
      prev.lines.push(line);
      byElement.set(symbol, prev);
    }
    return [...byElement.values()]
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 6);
  }, [autoIdentify, bandEnergyHi, bandEnergyLo, energy, lineHints, nEnergy, roiSpectrum]);
  const toggleSelectedElement = React.useCallback((symbol: string) => {
    const clean = normalizeElementSymbol(symbol);
    if (!clean || !elementsWithLines.has(clean)) return;
    const next = selectedElementSet.has(clean)
      ? safeSelectedElements.filter((item) => item !== clean)
      : [...safeSelectedElements, clean];
    setSelectedElements(next);
  }, [elementsWithLines, safeSelectedElements, selectedElementSet, setSelectedElements]);
  const selectOnlyElement = React.useCallback((symbol: string) => {
    const clean = normalizeElementSymbol(symbol);
    if (!clean || !elementsWithLines.has(clean)) return;
    setSelectedElements([clean]);
  }, [elementsWithLines, setSelectedElements]);
  const isBandCenterPreviewing = Boolean(bandSliderDrag || drag?.mode === "band-move");
  const bandMoveHitWidthPx = Math.max(
    32,
    Math.min(96, ((bandHi - bandLo) / Math.max(1, nEnergy)) * 260 + 20),
  );
  const bandMoveCenterPct = ((bandLo + bandHi) / 2 / Math.max(1, nEnergy)) * 100;
  React.useEffect(() => {
    if (localOverlayOpacity === null) return;
    if (Math.abs(localOverlayOpacity - overlayOpacity) <= 1e-6) setLocalOverlayOpacity(null);
  }, [localOverlayOpacity, overlayOpacity]);
  const roiOverlayStyle = React.useMemo(() => {
    const topLeft = mapImageToScreen(roi.row, roi.col);
    const bottomRight = mapImageToScreen(roi.row + roi.height, roi.col + roi.width);
    const left = topLeft.x;
    const top = topLeft.y;
    const width = bottomRight.x - topLeft.x;
    const height = bottomRight.y - topLeft.y;
    return { left, top, width, height };
  }, [mapImageToScreen, roi.col, roi.height, roi.row, roi.width]);
  const backendLabel = isSparseWorkerBackend
    ? (hasEmbeddedStream || isStreamBackend ? "Sparse stream" : "Data folder")
    : isKernelBackend ? "Kernel exact" : "WebGPU";
  const embeddedPayloadBytes =
    (cube?.byteLength ?? 0)
    + (base?.byteLength ?? rows * cols * 4)
    + (initialMap?.byteLength ?? 0)
    + (initialSpectrum?.byteLength ?? 0)
    + (hasEmbeddedStream ? (
      extractBytes(streamChannelOffsetsBytes).length
      + extractBytes(streamChannelPixelsBytes).length
      + extractBytes(streamPixelOffsetsBytes).length
      + extractBytes(streamPixelChannelsBytes).length
    ) : 0)
    + nEnergy * 4;
  const embeddedExportSize = formatEstimatedHtmlSize(embeddedPayloadBytes);
  const prefersFolderExport = isSidecarBackend && !hasEmbeddedStream;
  const exactExportLabel = prefersFolderExport
    ? `Exact linked folder (${embeddedExportSize} HTML + ${exportSidecarBytes > 0 ? formatBytes(exportSidecarBytes) : "data folder"})`
    : isSparseWorkerBackend ? `Exact single sparse (${embeddedExportSize})`
    : `Exact single file (${embeddedExportSize})`;
  const exportOptions = React.useMemo(() => {
    const custom = Array.isArray(exportPresets)
      ? exportPresets
          .map((preset) => ({
            mode: String(preset.mode || (prefersFolderExport ? "folder" : "single")),
            downsample: preset.downsample === undefined
              ? (preset.binning === undefined ? undefined : Number(preset.binning))
              : Number(preset.downsample),
            label: String(preset.label || preset.description || ""),
          }))
          .filter((preset) => preset.label)
      : [];
    if (custom.length > 0) return custom;
    const options: { mode: string; downsample?: number; label: string }[] = [
      { mode: prefersFolderExport ? "folder" : "single", label: exactExportLabel },
    ];
    for (const factor of [2, 4]) {
      if (rows >= factor && cols >= factor && nEnergy >= factor) {
        const labelPrefix = factor === 2 ? "Portable sharing" : "Small tutorial";
        options.push({
          mode: "single",
          downsample: factor,
          label: `${labelPrefix} sum-binned ${factor}x uint32 (${formatEstimatedHtmlSize(estimateBinnedPayloadBytes(rows, cols, nEnergy, factor))})`,
        });
      }
    }
    return options;
  }, [cols, exactExportLabel, exportPresets, prefersFolderExport, nEnergy, rows]);

  React.useEffect(() => {
    if (!exportStatus) return;
    const preparing = exportStatus.startsWith("Preparing ") || exportStatus.startsWith("Exporting ");
    if (preparing) {
      setExportBusy(true);
    } else if (!pendingExportRef.current) {
      setExportBusy(false);
    }
  }, [exportStatus]);

  const handleExportMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setExportMenuAnchor(event.currentTarget);
  };

  const handleExportMenuClose = () => {
    setExportMenuAnchor(null);
  };

  const handleExportSelect = async (mode: string, downsample?: number) => {
    setExportMenuAnchor(null);
    const filename = makeExportFilename(title, rows, cols, nEnergy, mode, downsample);
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setExportBusy(true);
    setLocalExportStatus("Choose export location...");
    const picker = (window as ShowEDSWindow).showSaveFilePicker;
    let handle: ShowEDSFileHandle | null = null;
    if (picker) {
      try {
        handle = await picker({
          suggestedName: filename,
          types: [{ description: "Standalone HTML", accept: { "text/html": [".html"] } }],
        });
      } catch (err) {
        if (isAbortLikeError(err)) {
          setExportBusy(false);
          setLocalExportStatus("Export canceled");
          return;
        }
        setExportBusy(false);
        setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
        return;
      }
    }
    pendingExportRef.current = { id, filename, mode, downsample, handle };
    setLocalExportStatus(`Preparing ${filename}...`);
    setExportRequest(JSON.stringify({ mode, encoding: "full", downsample, id, filename, download: true }));
  };

  React.useEffect(() => {
    const pending = pendingExportRef.current;
    if (!pending || exportPayloadId !== pending.id) return;
    const bytes = extractBytes(exportPayload);
    if (bytes.length === 0) return;
    let canceled = false;
    const save = async () => {
      const payload = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
        ? bytes
        : bytes.slice();
      const filename = exportPayloadFilename || pending.filename;
      const blob = new Blob([payload as BlobPart], { type: "text/html;charset=utf-8" });
      try {
        if (pending.handle) {
          setLocalExportStatus(`Saving ${filename}...`);
          const writable = await pending.handle.createWritable();
          await writable.write(blob);
          await writable.close();
        } else {
          downloadBlob(blob, filename);
        }
        if (canceled) return;
        pendingExportRef.current = null;
        setExportBusy(false);
        setLocalExportStatus(`Saved ${filename} (${formatBytes(bytes.byteLength)})`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      } catch (err) {
        if (canceled) return;
        pendingExportRef.current = null;
        setExportBusy(false);
        setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      }
    };
    void save();
    return () => { canceled = true; };
  }, [exportPayload, exportPayloadFilename, exportPayloadId, setExportRequest]);

  React.useEffect(() => {
    setLocalBand([bandStart, bandEnd]);
  }, [bandStart, bandEnd]);

  React.useEffect(() => {
    setLocalRoi(modelRoi);
  }, [roiRow, roiCol, roiHeight, roiWidth]);

  React.useEffect(() => {
    if (initialMap && initialMap.length === rows * cols) setElementMap(initialMap);
  }, [initialMap, rows, cols]);

  React.useEffect(() => {
    if (initialSpectrum && initialSpectrum.length === nEnergy) setRoiSpectrum(initialSpectrum);
  }, [initialSpectrum, nEnergy]);

  React.useEffect(() => {
    if (!isSparseWorkerBackend) return;
    if (!sidecarUrl && !hasEmbeddedStream) return;
    let disposed = false;
    const rejectPending = (message: string) => {
      for (const pending of sidecarPendingRef.current.values()) pending.reject(new Error(message));
      sidecarPendingRef.current.clear();
    };
    (async () => {
      try {
        let absoluteSidecarUrl = "";
        let meta: SidecarMeta;
        let streamBuffers: {
          channelOffsets: ArrayBuffer;
          channelPixels: ArrayBuffer;
          pixelOffsets: ArrayBuffer;
          pixelChannels: ArrayBuffer;
        } | null = null;
        if (sidecarUrl) {
          absoluteSidecarUrl = new URL(sidecarUrl, window.location.href).href;
          meta = await (await fetch(new URL("meta.json", absoluteSidecarUrl), { credentials: "include" })).json() as SidecarMeta;
        } else {
          meta = JSON.parse(sidecarMetaJson) as SidecarMeta;
          const channelOffsets = copyExtractedBuffer(streamChannelOffsetsBytes);
          const channelPixels = copyExtractedBuffer(streamChannelPixelsBytes);
          const pixelOffsets = copyExtractedBuffer(streamPixelOffsetsBytes);
          const pixelChannels = copyExtractedBuffer(streamPixelChannelsBytes);
          if (!channelOffsets || !channelPixels || !pixelOffsets || !pixelChannels) {
            throw new Error("embedded EDS stream buffers are incomplete");
          }
          streamBuffers = { channelOffsets, channelPixels, pixelOffsets, pixelChannels };
        }
        if (disposed) return;
        sidecarRef.current?.worker.terminate();
        rejectPending("EDS sparse data worker was replaced");
        const worker = createSidecarWorker();
        const sidecarFileUrl = (name: string) => {
          const base = new URL(absoluteSidecarUrl);
          const url = new URL(name, base);
          if (base.search && !url.search) url.search = base.search;
          return url;
        };
        worker.onmessage = (event: MessageEvent<SidecarWorkerResponse>) => {
          const response = event.data;
          if (response.type === "fetch-range") {
            const fetchId = response.fetchId;
            const name = response.name || "";
            if (!sidecarUrl) {
              worker.postMessage({
                type: "range-error",
                fetchId,
                message: "embedded EDS stream has no external data folder",
              });
              return;
            }
            const start = Math.max(0, Math.floor(response.start ?? 0));
            const end = Math.max(start, Math.floor(response.end ?? start));
            const expectedByteLength = end - start + 1;
            (async () => {
              try {
                const resp = await fetch(sidecarFileUrl(name), {
                  credentials: "include",
                  headers: { Range: `bytes=${start}-${end}` },
                });
                let buffer: ArrayBuffer;
                if (resp.status === 206) {
                  buffer = await resp.arrayBuffer();
                } else if (resp.status === 200) {
                  buffer = await resp.arrayBuffer();
                  if (buffer.byteLength === expectedByteLength && start === 0) {
                    // Use the full response below.
                  } else if (buffer.byteLength >= end + 1) {
                    buffer = buffer.slice(start, end + 1);
                  } else {
                    throw new Error(`range fetch returned ${buffer.byteLength} bytes for a ${expectedByteLength} byte request`);
                  }
                } else {
                  throw new Error(`range fetch failed: ${resp.status}`);
                }
                worker.postMessage({ type: "range-result", fetchId, buffer }, [buffer]);
              } catch (err) {
                worker.postMessage({
                  type: "range-error",
                  fetchId,
                  message: err instanceof Error ? err.message : String(err),
                });
              }
            })();
            return;
          }
          if (response.id === undefined) return;
          const pending = sidecarPendingRef.current.get(response.id);
          if (!pending) return;
          sidecarPendingRef.current.delete(response.id);
          if (response.message) pending.reject(new Error(response.message));
          else pending.resolve(response);
        };
        worker.onerror = (event) => {
          rejectPending(event.message || "EDS sparse data worker failed");
        };
        if (streamBuffers) {
          worker.postMessage(
            { type: "init", meta, baseUrl: "", streamBuffers },
            [streamBuffers.channelOffsets, streamBuffers.channelPixels, streamBuffers.pixelOffsets, streamBuffers.pixelChannels],
          );
        } else {
          worker.postMessage({ type: "init", meta, baseUrl: absoluteSidecarUrl });
        }
        sidecarRef.current = { meta, worker };
        setGpuError("");
      } catch (err) {
        if (!disposed) setGpuError(`Could not load EDS sparse data: ${err instanceof Error ? err.message : String(err)}`);
      }
    })();
    return () => {
      disposed = true;
      sidecarRef.current?.worker.terminate();
      sidecarRef.current = null;
      rejectPending("EDS sparse data worker was disposed");
    };
  }, [
    isSparseWorkerBackend,
    sidecarUrl,
    sidecarMetaJson,
    streamChannelOffsetsBytes,
    streamChannelPixelsBytes,
    streamPixelOffsetsBytes,
    streamPixelChannelsBytes,
  ]);

  React.useEffect(() => {
    const handler = (content: { type?: string; message?: string }, buffers?: DataView[]) => {
      const first = buffers?.[0];
      if (content.type === "map" && first) {
        setElementMap(extractFloat32(first, rows * cols));
        setElementMapPreview(null);
        setBusy(false);
      } else if (content.type === "spectrum" && first) {
        setRoiSpectrum(extractFloat32(first, nEnergy));
        setBusy(false);
      } else if (content.type === "error") {
        setGpuError(content.message || "Kernel-backed EDS compute failed.");
        setBusy(false);
      }
    };
    model.on("msg:custom", handler);
    return () => model.off("msg:custom", handler);
  }, [model, nEnergy, rows, cols]);

  React.useEffect(() => {
    let disposed = false;
    const init = async () => {
      setGpuError("");
      if (isKernelBackend || isSparseWorkerBackend || !cube || cube.length === 0 || rows <= 0 || cols <= 0 || nEnergy <= 0) return;
      if (!navigator.gpu) {
        setGpuError("WebGPU is not available in this browser.");
        return;
      }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        setGpuError("No WebGPU adapter found.");
        return;
      }
      const device = await adapter.requestDevice();
      if (disposed) return;
      const cubeBuf = device.createBuffer({ size: cube.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(cubeBuf, 0, cube as Uint8Array<ArrayBuffer>);
      const mapBuf = device.createBuffer({ size: rows * cols * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const specBuf = device.createBuffer({ size: nEnergy * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const makeUniform = () => device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const mapPipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: MAP_WGSL }), entryPoint: "main" } });
      const specPipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: SPEC_WGSL }), entryPoint: "main" } });
      const mapParamsA = makeUniform();
      const mapParamsB = makeUniform();
      const specParamsA = makeUniform();
      const specParamsB = makeUniform();
      gpuRef.current?.cube.destroy();
      gpuRef.current?.map.destroy();
      gpuRef.current?.spectrum.destroy();
      gpuRef.current = {
        device,
        cube: cubeBuf,
        map: mapBuf,
        spectrum: specBuf,
        mapParamsA,
        mapParamsB,
        specParamsA,
        specParamsB,
        mapPipeline,
        specPipeline,
        mapBindGroup: device.createBindGroup({
          layout: mapPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: cubeBuf } },
            { binding: 1, resource: { buffer: mapBuf } },
            { binding: 2, resource: { buffer: mapParamsA } },
            { binding: 3, resource: { buffer: mapParamsB } },
          ],
        }),
        specBindGroup: device.createBindGroup({
          layout: specPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: cubeBuf } },
            { binding: 1, resource: { buffer: specBuf } },
            { binding: 2, resource: { buffer: specParamsA } },
            { binding: 3, resource: { buffer: specParamsB } },
          ],
        }),
      };
    };
    void init();
    return () => { disposed = true; };
  }, [cube, rows, cols, nEnergy, isKernelBackend, isSparseWorkerBackend]);

  React.useEffect(() => {
    return () => {
      if (mapRafRef.current != null) window.cancelAnimationFrame(mapRafRef.current);
      if (specRafRef.current != null) window.cancelAnimationFrame(specRafRef.current);
      if (mapThrottleTimerRef.current != null) window.clearTimeout(mapThrottleTimerRef.current);
      if (specThrottleTimerRef.current != null) window.clearTimeout(specThrottleTimerRef.current);
      if (bandPersistTimerRef.current != null) window.clearTimeout(bandPersistTimerRef.current);
      if (roiPersistTimerRef.current != null) window.clearTimeout(roiPersistTimerRef.current);
      if (viewPersistTimerRef.current != null) window.clearTimeout(viewPersistTimerRef.current);
      if (localBandRafRef.current != null) window.cancelAnimationFrame(localBandRafRef.current);
    };
  }, []);

  const requestSidecarWorker = React.useCallback((request: Record<string, number | string>) => {
    const worker = sidecarRef.current?.worker;
    if (!worker) return Promise.reject(new Error("EDS sparse data worker is not ready"));
    const id = ++sidecarWorkerRequestIdRef.current;
    return new Promise<SidecarWorkerResponse>((resolve, reject) => {
      sidecarPendingRef.current.set(id, { resolve, reject });
      worker.postMessage({ ...request, id });
    });
  }, []);

  const computeMap = React.useCallback(async (start: number, end: number, interactive = false) => {
    if (isSparseWorkerBackend) {
      if (!sidecarRef.current) return;
      const seq = ++mapComputeSeqRef.current;
      const t0 = performance.now();
      const s = Math.max(0, Math.min(nEnergy - 1, Math.round(start)));
      const e = Math.max(s + 1, Math.min(nEnergy, Math.round(end)));
      const previewWidth = Math.max(1, Math.round(size * MAP_RASTER_DPR));
      const previewHeight = Math.max(1, Math.round(size * MAP_RASTER_DPR));
      const response = await requestSidecarWorker(interactive && isStreamBackend ? {
        type: "map-preview",
        start: s,
        end: e,
        width: previewWidth,
        height: previewHeight,
        viewRow: mapView.row,
        viewCol: mapView.col,
        viewRows: mapView.rows,
        viewCols: mapView.cols,
      } : { type: "map", start: s, end: e });
      if (response.aborted) return;
      if (seq !== mapComputeSeqRef.current) return;
      if (!response.buffer) throw new Error("EDS sparse data worker returned an empty map");
      const buffer = response.buffer;
      setGpuError("");
      const previewResponseWidth = typeof response.width === "number" ? response.width : 0;
      const previewResponseHeight = typeof response.height === "number" ? response.height : 0;
      if (interactive && isStreamBackend && previewResponseWidth > 0 && previewResponseHeight > 0) {
        const viewRow = Number(response.viewRow ?? mapView.row);
        const viewCol = Number(response.viewCol ?? mapView.col);
        const viewRows = Number(response.viewRows ?? mapView.rows);
        const viewCols = Number(response.viewCols ?? mapView.cols);
        React.startTransition(() => setElementMapPreview({
          data: new Uint32Array(buffer),
          width: Math.max(1, Math.round(previewResponseWidth)),
          height: Math.max(1, Math.round(previewResponseHeight)),
          viewRow,
          viewCol,
          viewRows,
          viewCols,
        }));
      } else {
        React.startTransition(() => setElementMap(new Uint32Array(buffer)));
        setElementMapPreview(null);
      }
      recordComputeBackend("map", response.backend);
      recordWidgetPerf("mapMs", response.ms ?? performance.now() - t0);
      return;
    }
    if (isKernelBackend) {
      setBusy(true);
      model.send({ type: "compute_map", request_id: ++requestIdRef.current, start, end });
      return;
    }
    const gpu = gpuRef.current;
    if (!gpu) return;
    const t0 = performance.now();
    const s = Math.max(0, Math.min(nEnergy - 1, Math.round(start)));
    const e = Math.max(s + 1, Math.min(nEnergy, Math.round(end)));
    gpu.device.queue.writeBuffer(gpu.mapParamsA, 0, new Uint32Array([rows, cols, nEnergy, s]));
    gpu.device.queue.writeBuffer(gpu.mapParamsB, 0, new Uint32Array([e, cubeDtype === "uint16" ? 1 : cubeDtype === "uint32" ? 2 : 0, 0, 0]));
    const enc = gpu.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(gpu.mapPipeline);
    pass.setBindGroup(0, gpu.mapBindGroup);
    pass.dispatchWorkgroups(Math.ceil((rows * cols) / WORKGROUP));
    pass.end();
    gpu.device.queue.submit([enc.finish()]);
    const out = await readBuffer(gpu.device, gpu.map, rows * cols * 4);
    React.startTransition(() => setElementMap(out));
    setElementMapPreview(null);
    recordWidgetPerf("mapMs", performance.now() - t0);
  }, [cols, cubeDtype, isKernelBackend, isSparseWorkerBackend, isStreamBackend, mapView, model, nEnergy, recordComputeBackend, recordWidgetPerf, rows, requestSidecarWorker, sidecarUrl, size]);

  const computeSpectrum = React.useCallback(async (nextRoi: Roi) => {
    if (isSparseWorkerBackend) {
      if (!sidecarRef.current) return;
      const seq = ++spectrumComputeSeqRef.current;
      const t0 = performance.now();
      const r0 = Math.max(0, Math.min(rows - 1, Math.round(nextRoi.row)));
      const c0 = Math.max(0, Math.min(cols - 1, Math.round(nextRoi.col)));
      const r1 = Math.max(r0 + 1, Math.min(rows, r0 + Math.round(nextRoi.height)));
      const c1 = Math.max(c0 + 1, Math.min(cols, c0 + Math.round(nextRoi.width)));
      const response = await requestSidecarWorker({
        type: "spectrum",
        row: r0,
        col: c0,
        height: r1 - r0,
        width: c1 - c0,
        shape: normalizeRoiShape(nextRoi.shape),
      });
      if (response.aborted) return;
      if (seq !== spectrumComputeSeqRef.current) return;
      if (!response.buffer) throw new Error("EDS sparse data worker returned an empty spectrum");
      const buffer = response.buffer;
      setGpuError("");
      React.startTransition(() => setRoiSpectrum(new Uint32Array(buffer)));
      recordComputeBackend("spectrum", response.backend);
      recordWidgetPerf("spectrumMs", response.ms ?? performance.now() - t0);
      return;
    }
    if (isKernelBackend) {
      setBusy(true);
      model.send({ type: "compute_spectrum", request_id: ++requestIdRef.current, ...nextRoi });
      return;
    }
    const gpu = gpuRef.current;
    if (!gpu) return;
    const t0 = performance.now();
    const r0 = Math.max(0, Math.min(rows - 1, Math.round(nextRoi.row)));
    const c0 = Math.max(0, Math.min(cols - 1, Math.round(nextRoi.col)));
    const r1 = Math.max(r0 + 1, Math.min(rows, r0 + Math.round(nextRoi.height)));
    const c1 = Math.max(c0 + 1, Math.min(cols, c0 + Math.round(nextRoi.width)));
    const mode = cubeDtype === "uint16" ? 1 : cubeDtype === "uint32" ? 2 : 0;
    const shape = normalizeRoiShape(nextRoi.shape) === "rect" ? 0 : 1;
    gpu.device.queue.writeBuffer(gpu.specParamsA, 0, new Uint32Array([rows, cols, nEnergy, r0]));
    gpu.device.queue.writeBuffer(gpu.specParamsB, 0, new Uint32Array([c0, r1, c1, mode | (shape << 4)]));
    const enc = gpu.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(gpu.specPipeline);
    pass.setBindGroup(0, gpu.specBindGroup);
    pass.dispatchWorkgroups(Math.ceil(nEnergy / WORKGROUP));
    pass.end();
    gpu.device.queue.submit([enc.finish()]);
    const out = await readBuffer(gpu.device, gpu.spectrum, nEnergy * 4);
    React.startTransition(() => setRoiSpectrum(out));
    recordWidgetPerf("spectrumMs", performance.now() - t0);
  }, [cols, cubeDtype, isKernelBackend, isSparseWorkerBackend, model, nEnergy, recordComputeBackend, recordWidgetPerf, rows, requestSidecarWorker, sidecarUrl]);

  const flushMapRequest = React.useCallback(() => {
    mapRafRef.current = null;
    const allowConcurrent = false;
    if (mapRunningRef.current && !allowConcurrent) {
      mapRerunRef.current = true;
      return;
    }
    const req = mapRequestRef.current;
    if (!req) return;
    lastMapFlushRef.current = performance.now();
    if (!allowConcurrent) mapRunningRef.current = true;
    if (!isSparseWorkerBackend) setBusy(true);
    void computeMap(req.start, req.end, req.interactive).catch((err) => {
      setGpuError(err instanceof Error ? err.message : String(err));
    }).finally(() => {
      if (allowConcurrent) return;
      mapRunningRef.current = false;
      if (mapRerunRef.current) {
        mapRerunRef.current = false;
        const latest = mapRequestRef.current;
        if (latest) scheduleMap(latest.start, latest.end, latest.interactive);
      } else if (!isSparseWorkerBackend) {
        setBusy(false);
      }
    });
  }, [computeMap, isSparseWorkerBackend]);

  const scheduleMap = React.useCallback((start: number, end: number, interactive = false) => {
    mapRequestRef.current = { start, end, interactive };
    if (!interactive && mapThrottleTimerRef.current != null) {
      window.clearTimeout(mapThrottleTimerRef.current);
      mapThrottleTimerRef.current = null;
    }
    if (mapRafRef.current != null || mapThrottleTimerRef.current != null) return;
    const scheduleFrame = () => {
      mapThrottleTimerRef.current = null;
      mapRafRef.current = window.requestAnimationFrame(flushMapRequest);
    };
    if (interactive) {
      const elapsed = performance.now() - lastMapFlushRef.current;
      const waitMs = Math.max(0, INTERACTION_COMPUTE_INTERVAL_MS - elapsed);
      if (waitMs > 0) {
        mapThrottleTimerRef.current = window.setTimeout(scheduleFrame, waitMs);
        return;
      }
    }
    scheduleFrame();
  }, [flushMapRequest]);
  scheduleMapRef.current = scheduleMap;

  const flushSpectrumRequest = React.useCallback(() => {
    specRafRef.current = null;
    const allowConcurrent = false;
    if (specRunningRef.current && !allowConcurrent) {
      specRerunRef.current = true;
      return;
    }
    const req = specRequestRef.current;
    if (!req) return;
    lastSpecFlushRef.current = performance.now();
    if (!allowConcurrent) specRunningRef.current = true;
    void computeSpectrum(req).catch((err) => {
      setGpuError(err instanceof Error ? err.message : String(err));
    }).finally(() => {
      if (allowConcurrent) return;
      specRunningRef.current = false;
      if (specRerunRef.current) {
        specRerunRef.current = false;
        const latest = specRequestRef.current;
        if (latest) scheduleSpectrum(latest, true);
      }
    });
  }, [computeSpectrum, isSparseWorkerBackend]);

  const scheduleSpectrum = React.useCallback((nextRoi: Roi, interactive = false) => {
    specRequestRef.current = nextRoi;
    if (!interactive && specThrottleTimerRef.current != null) {
      window.clearTimeout(specThrottleTimerRef.current);
      specThrottleTimerRef.current = null;
    }
    if (specRafRef.current != null || specThrottleTimerRef.current != null) return;
    const scheduleFrame = () => {
      specThrottleTimerRef.current = null;
      specRafRef.current = window.requestAnimationFrame(flushSpectrumRequest);
    };
    if (interactive) {
      const elapsed = performance.now() - lastSpecFlushRef.current;
      const waitMs = Math.max(0, INTERACTION_COMPUTE_INTERVAL_MS - elapsed);
      if (waitMs > 0) {
        specThrottleTimerRef.current = window.setTimeout(scheduleFrame, waitMs);
        return;
      }
    }
    scheduleFrame();
  }, [flushSpectrumRequest]);

  React.useEffect(() => {
    if (!gpuRef.current && !isSparseWorkerBackend) return;
    scheduleMap(bandStart, bandEnd);
  }, [bandEnd, bandStart, isSparseWorkerBackend, scheduleMap, gpuRef.current]);
  React.useEffect(() => {
    if (!gpuRef.current && !isSparseWorkerBackend) return;
    scheduleSpectrum(modelRoi);
  }, [isSparseWorkerBackend, roiCol, roiHeight, roiRow, roiShape, roiWidth, scheduleSpectrum, gpuRef.current]);
  React.useEffect(() => {
    const id = window.setTimeout(() => {
      if (gpuRef.current || isSparseWorkerBackend) {
        scheduleMap(bandStart, bandEnd);
        scheduleSpectrum(modelRoi);
      }
    }, 100);
    return () => window.clearTimeout(id);
  }, [bandEnd, bandStart, isSparseWorkerBackend, roiCol, roiHeight, roiRow, roiShape, roiWidth, scheduleMap, scheduleSpectrum, gpuRef.current]);

  React.useEffect(() => {
    const canvas = mapCanvasRef.current;
    if (!canvas || !base) return;
    const t0 = performance.now();
    const dpr = MAP_RASTER_DPR;
    canvas.width = Math.max(1, Math.round(size * dpr));
    canvas.height = Math.max(1, Math.round(size * dpr));
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const image = ctx.createImageData(canvas.width, canvas.height);
    const [baseLo, baseHi] = finiteRange(base);
    const baseSpan = Math.max(1e-12, baseHi - baseLo);
    const rowStep = mapView.rows / Math.max(1, canvas.height);
    const colStep = mapView.cols / Math.max(1, canvas.width);
    for (let py = 0; py < canvas.height; py++) {
      const r = Math.max(0, Math.min(rows - 1, Math.floor(mapView.row + (py + 0.5) * rowStep)));
      for (let px = 0; px < canvas.width; px++) {
        const c = Math.max(0, Math.min(cols - 1, Math.floor(mapView.col + (px + 0.5) * colStep)));
        const idx = r * cols + c;
        const off = (py * canvas.width + px) * 4;
        const g = Math.round(255 * Math.max(0, Math.min(1, (base[idx] - baseLo) / baseSpan)));
        image.data[off] = g; image.data[off + 1] = g; image.data[off + 2] = g; image.data[off + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
    recordWidgetPerf("mapDrawMs", performance.now() - t0);
  }, [base, cols, mapView, recordWidgetPerf, rows, size]);

  const drawMapOverlay = React.useCallback((displayRange: [number, number]) => {
    const canvas = mapOverlayCanvasRef.current;
    if (!canvas) return;
    const dpr = MAP_RASTER_DPR;
    canvas.width = Math.max(1, Math.round(size * dpr));
    canvas.height = Math.max(1, Math.round(size * dpr));
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const preview = elementMapPreview;
    const previewMatches = preview
      && preview.width === canvas.width
      && preview.height === canvas.height
      && Math.abs(preview.viewRow - mapView.row) < 1e-3
      && Math.abs(preview.viewCol - mapView.col) < 1e-3
      && Math.abs(preview.viewRows - mapView.rows) < 1e-3
      && Math.abs(preview.viewCols - mapView.cols) < 1e-3;
    if (!elementMap && !previewMatches) return;
    const t0 = performance.now();
    const image = ctx.createImageData(canvas.width, canvas.height);
    const [mapLo, mapHi] = previewMatches
      ? (() => {
        const previewRange = finiteRange(preview.data);
        const range = sliderRange(previewRange[0], previewRange[1], mapVminPct, mapVmaxPct);
        return Number.isFinite(range.vmin) && Number.isFinite(range.vmax) && range.vmax > range.vmin
          ? [range.vmin, range.vmax] as [number, number]
          : previewRange;
      })()
      : displayRange;
    const mapSpan = Math.max(1e-12, mapHi - mapLo);
    if (previewMatches) {
      const data = preview.data;
      for (let py = 0; py < canvas.height; py++) {
        for (let px = 0; px < canvas.width; px++) {
          const idx = py * canvas.width + px;
          const off = idx * 4;
          const value = data[idx];
          if (!Number.isFinite(value)) continue;
          const [cr, cg, cb] = colorize((value - mapLo) / mapSpan);
          image.data[off] = cr; image.data[off + 1] = cg; image.data[off + 2] = cb; image.data[off + 3] = 255;
        }
      }
    } else if (elementMap) {
      const rowStep = mapView.rows / Math.max(1, canvas.height);
      const colStep = mapView.cols / Math.max(1, canvas.width);
      for (let py = 0; py < canvas.height; py++) {
        const r = Math.max(0, Math.min(rows - 1, Math.floor(mapView.row + (py + 0.5) * rowStep)));
        for (let px = 0; px < canvas.width; px++) {
          const c = Math.max(0, Math.min(cols - 1, Math.floor(mapView.col + (px + 0.5) * colStep)));
          const idx = r * cols + c;
          const off = (py * canvas.width + px) * 4;
          const value = elementMap[idx];
          if (!Number.isFinite(value)) continue;
          const [cr, cg, cb] = colorize((value - mapLo) / mapSpan);
          image.data[off] = cr; image.data[off + 1] = cg; image.data[off + 2] = cb; image.data[off + 3] = 255;
        }
      }
    }
    ctx.putImageData(image, 0, 0);
    recordWidgetPerf("mapDrawMs", performance.now() - t0);
  }, [cols, elementMap, elementMapPreview, mapView, mapVmaxPct, mapVminPct, recordWidgetPerf, rows, size]);

  React.useEffect(() => {
    if (!elementMapPreview) return;
    if (
      Math.abs(elementMapPreview.viewRow - mapView.row) > 1e-3
      || Math.abs(elementMapPreview.viewCol - mapView.col) > 1e-3
      || Math.abs(elementMapPreview.viewRows - mapView.rows) > 1e-3
      || Math.abs(elementMapPreview.viewCols - mapView.cols) > 1e-3
    ) {
      setElementMapPreview(null);
    }
  }, [elementMapPreview, mapView]);

  React.useEffect(() => {
    drawMapOverlay(mapDisplayRange);
  }, [drawMapOverlay, mapDisplayRange]);

  React.useEffect(() => {
    const canvas = mapUiOverlayCanvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(size * dpr));
    canvas.height = Math.max(1, Math.round(size * dpr));
    if (scaleBarVisible) {
      const pxSize = pixelSize > 0 ? pixelSize : 1;
      const unit = pixelSize > 0 ? pixelUnit : "px";
      drawScaleBarHiDPI(canvas, dpr, mapView.zoom, pxSize, unit, cols);
      return;
    }
    const ctx = canvas.getContext("2d");
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
  }, [cols, mapView.zoom, pixelSize, pixelUnit, scaleBarVisible, size]);

  React.useEffect(() => {
    const canvas = specCanvasRef.current;
    if (!canvas || !roiSpectrum) return;
    const t0 = performance.now();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(specW * dpr);
    canvas.height = Math.round(specH * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = themeColors.plotBg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const padL = 54 * dpr, padR = 14 * dpr, padT = 16 * dpr, padB = 34 * dpr;
    const plotW = canvas.width - padL - padR;
    const plotH = canvas.height - padT - padB;
    const values = roiSpectrum;
    const viewStart = Math.max(0, Math.min(values.length - 1, Math.floor(spectrumView.start)));
    const viewEnd = Math.max(viewStart + 1, Math.min(values.length, Math.ceil(spectrumView.end)));
    const transformed = new Float32Array(viewEnd - viewStart);
    for (let i = viewStart; i < viewEnd; i++) transformed[i - viewStart] = logSpectrum ? Math.log10(Math.max(1, values[i])) : values[i];
    const [lo, hi] = finiteRange(transformed);
    const ySpan = Math.max(1e-12, hi - lo);
    ctx.strokeStyle = themeColors.plotGrid;
    ctx.lineWidth = dpr;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (i / 4) * plotH;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
    }
    if (showLineHints && Array.isArray(lineHints) && energy.length > 1) {
      const visibleLines = lineHints
        .filter((line) => {
          const selected = selectedElementSet.has(normalizeElementSymbol(line.element));
          return selected || line.intensity === undefined || line.intensity >= 0.04 || (line.energy_keV >= bandEnergyLo && line.energy_keV <= bandEnergyHi);
        });
      ctx.save();
      ctx.font = `${10 * dpr}px ${UI_FONT}`;
      for (const line of visibleLines) {
        const lineIndex = energyToIndex(energy, line.energy_keV);
        const x = padL + ((lineIndex - spectrumView.start) / Math.max(1e-9, spectrumView.span)) * plotW;
        if (x < padL || x > padL + plotW) continue;
        const inBand = line.energy_keV >= bandEnergyLo && line.energy_keV <= bandEnergyHi;
        const selected = selectedElementSet.has(normalizeElementSymbol(line.element));
        ctx.strokeStyle = selected || inBand ? themeColors.lineHint : themeColors.lineHintMuted;
        ctx.lineWidth = selected ? 2.25 * dpr : inBand ? 1.5 * dpr : dpr;
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
      }
      const labelLines = (selectedLineHints.length > 0 ? selectedLineHints : candidateLines).slice(0, 5);
      labelLines.forEach((line, index) => {
        const lineIndex = energyToIndex(energy, line.energy_keV);
        const x = padL + ((lineIndex - spectrumView.start) / Math.max(1e-9, spectrumView.span)) * plotW;
        if (x < padL || x > padL + plotW) return;
        ctx.fillStyle = themeColors.lineHintText;
        ctx.fillText(lineLabel(line), Math.min(x + 3 * dpr, padL + plotW - 48 * dpr), padT + (13 + index * 12) * dpr);
      });
      ctx.restore();
    }
    ctx.strokeStyle = themeColors.spectrumLine;
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    for (let i = viewStart; i < viewEnd; i++) {
      const x = padL + ((i - spectrumView.start) / Math.max(1e-9, spectrumView.span)) * plotW;
      const y = padT + plotH - ((transformed[i - viewStart] - lo) / ySpan) * plotH;
      if (i === viewStart) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = themeColors.plotText;
    ctx.font = `${11 * dpr}px ${UI_FONT}`;
    ctx.fillText(`${formatEnergy(energy[bandLo])} - ${formatEnergy(energy[Math.max(bandLo, bandHi - 1)])}`, padL, canvas.height - 10 * dpr);
    ctx.fillText(logSpectrum ? "log counts" : "counts", 8 * dpr, 16 * dpr);
    recordWidgetPerf("spectrumDrawMs", performance.now() - t0);
  }, [
    bandEnergyHi,
    bandEnergyLo,
    bandHi,
    bandLo,
    candidateLines,
    energy,
    lineHints,
    logSpectrum,
    recordWidgetPerf,
    roiSpectrum,
    selectedElementSet,
    selectedLineHints,
    showLineHints,
    specH,
    specW,
    spectrumView,
    themeColors.lineHint,
    themeColors.lineHintMuted,
    themeColors.lineHintText,
    themeColors.plotBg,
    themeColors.plotGrid,
    themeColors.plotText,
    themeColors.spectrumLine,
  ]);

  const flushBandPersist = React.useCallback(() => {
    if (bandPersistTimerRef.current != null) {
      window.clearTimeout(bandPersistTimerRef.current);
      bandPersistTimerRef.current = null;
    }
    const pending = pendingBandPersistRef.current;
    pendingBandPersistRef.current = null;
    if (!pending) return;
    setBandStart(pending[0]);
    setBandEnd(pending[1]);
    saveWidgetChanges();
  }, [saveWidgetChanges, setBandEnd, setBandStart]);

  const queueBandPersist = React.useCallback((start: number, end: number, immediate = false, defer = false) => {
    pendingBandPersistRef.current = [start, end];
    if (immediate) {
      flushBandPersist();
      return;
    }
    if (bandPersistTimerRef.current != null) window.clearTimeout(bandPersistTimerRef.current);
    if (defer) {
      bandPersistTimerRef.current = null;
      return;
    }
    bandPersistTimerRef.current = window.setTimeout(flushBandPersist, 250);
  }, [flushBandPersist]);

  const flushRoiPersist = React.useCallback(() => {
    if (roiPersistTimerRef.current != null) {
      window.clearTimeout(roiPersistTimerRef.current);
      roiPersistTimerRef.current = null;
    }
    const pending = pendingRoiPersistRef.current;
    pendingRoiPersistRef.current = null;
    if (!pending) return;
    setRoiRow(pending.row);
    setRoiCol(pending.col);
    setRoiHeight(pending.height);
    setRoiWidth(pending.width);
    setRoiShape(normalizeRoiShape(pending.shape));
    saveWidgetChanges();
  }, [saveWidgetChanges, setRoiCol, setRoiHeight, setRoiRow, setRoiShape, setRoiWidth]);

  const queueRoiPersist = React.useCallback((next: Roi, immediate = false, defer = false) => {
    pendingRoiPersistRef.current = next;
    if (immediate) {
      flushRoiPersist();
      return;
    }
    if (roiPersistTimerRef.current != null) window.clearTimeout(roiPersistTimerRef.current);
    if (defer) {
      roiPersistTimerRef.current = null;
      return;
    }
    roiPersistTimerRef.current = window.setTimeout(flushRoiPersist, 250);
  }, [flushRoiPersist]);

  const updateRoi = (next: Roi, sync = true, interactive = false) => {
    const normalized = normalizeRoi({ ...next, shape: normalizeRoiShape(next.shape ?? roi.shape) }, rows, cols);
    const shapeChanged = normalizeRoiShape(normalized.shape) !== normalizeRoiShape(roi.shape);
    const deferCurvedSidecarSpectrum = interactive && isSparseWorkerBackend && normalizeRoiShape(normalized.shape) !== "rect";
    setLocalRoi(normalized);
    if (sync) {
      queueRoiPersist(normalized, shapeChanged, interactive && !shapeChanged);
    }
    if (deferCurvedSidecarSpectrum) {
      specRequestRef.current = normalized;
    } else {
      scheduleSpectrum(normalized, interactive);
    }
  };

  const updateBand = (start: number, end: number, sync = true, interactive = false, compute = true) => {
    const s = Math.max(0, Math.min(nEnergy - 1, Math.round(start)));
    const e = Math.max(s + 1, Math.min(nEnergy, Math.round(end)));
    setLocalBandPreview([s, e], compute);
    if (sync) {
      queueBandPersist(s, e, false, interactive);
    }
    if (compute) {
      scheduleMap(s, e, interactive);
    } else {
      mapRequestRef.current = { start: s, end: e, interactive };
    }
  };

  const snapBandToLine = (line: EdsLineHint) => {
    const center = energyToIndex(energy, line.energy_keV);
    const span = Math.max(3, bandHi - bandLo);
    const start = Math.round(center - span / 2);
    updateBand(start, start + span, true);
    setElementMenuAnchor(null);
  };

  const saveCurrentRoi = () => {
    const name = `ROI ${safeSavedRois.length + 1}`;
    const next = [
      ...safeSavedRois.filter((item) => item.name !== name),
      { name, row: roi.row, col: roi.col, height: roi.height, width: roi.width, shape: normalizeRoiShape(roi.shape) },
    ];
    setSavedRois(next);
    saveWidgetChanges();
  };

  const saveCurrentBand = () => {
    const name = `Band ${safeSavedBands.length + 1}`;
    const next = [
      ...safeSavedBands.filter((item) => item.name !== name),
      { name, start: bandLo, end: bandHi },
    ];
    setSavedBands(next);
    saveWidgetChanges();
  };

  const applySavedRoi = (saved: SavedRoi) => {
    updateRoi(saved, true);
    setRoiMenuAnchor(null);
  };

  const applyRoiShape = (shape: RoiShape) => {
    updateRoi({ ...roi, shape }, true);
    setRoiShapeMenuAnchor(null);
  };

  const applySavedBand = (saved: SavedBand) => {
    updateBand(saved.start, saved.end, true);
    setBandMenuAnchor(null);
  };

  const clearSavedRois = () => {
    setSavedRois([]);
    setRoiMenuAnchor(null);
    saveWidgetChanges();
  };

  const clearSavedBands = () => {
    setSavedBands([]);
    setBandMenuAnchor(null);
    saveWidgetChanges();
  };

  const summarizePerf = React.useCallback((values?: number[]): PerfStat => {
    const arr = (values || []).slice(-30);
    if (arr.length === 0) return { last: 0, avg: 0, max: 0 };
    const sum = arr.reduce((acc, value) => acc + value, 0);
    return {
      last: arr[arr.length - 1],
      avg: sum / arr.length,
      max: Math.max(...arr),
    };
  }, []);

  const perfSummary = React.useMemo(() => {
    const perf = (window as ShowEDSPerfWindow).__quantemShowEDSPerf;
    return {
      map: summarizePerf(perf?.mapMs),
      spectrum: summarizePerf(perf?.spectrumMs),
      mapDraw: summarizePerf(perf?.mapDrawMs),
      spectrumDraw: summarizePerf(perf?.spectrumDrawMs),
    };
  }, [fps, perfTick, summarizePerf]);

  const mapPoint = (e: React.MouseEvent) => {
    const r = mapViewportRef.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const onMapDown = (e: React.MouseEvent) => {
    const p = mapPoint(e);
    const img = mapScreenToImage(p.x, p.y);
    const hitCol = (14 / Math.max(1, size)) * mapView.cols;
    const hitRow = (14 / Math.max(1, size)) * mapView.rows;
    const roiShapeNow = normalizeRoiShape(roi.shape);
    const roiIsRound = roiShapeNow !== "rect";
    const handleCol = roiIsRound ? roi.col + roi.width / 2 + roi.width / (2 * Math.SQRT2) : roi.col + roi.width;
    const handleRow = roiIsRound ? roi.row + roi.height / 2 + roi.height / (2 * Math.SQRT2) : roi.row + roi.height;
    const nearCorner = Math.abs(img.col - handleCol) < hitCol && Math.abs(img.row - handleRow) < hitRow;
    const insideBox = img.col >= roi.col && img.col <= roi.col + roi.width && img.row >= roi.row && img.row <= roi.row + roi.height;
    const inside = roiIsRound
      ? insideBox && (((img.col - (roi.col + roi.width / 2)) / Math.max(1e-9, roi.width / 2)) ** 2
        + ((img.row - (roi.row + roi.height / 2)) / Math.max(1e-9, roi.height / 2)) ** 2 <= 1)
      : insideBox;
    if (nearCorner || inside) {
      setDrag({ mode: nearCorner ? "roi-resize" : "roi-move", x: p.x, y: p.y, roi, bandStart: bandLo, bandEnd: bandHi });
      e.preventDefault();
    } else if (mapView.zoom > 1.001) {
      setDrag({ mode: "map-pan", x: p.x, y: p.y, roi, bandStart: bandLo, bandEnd: bandHi, mapViewRow: mapView.row, mapViewCol: mapView.col });
      e.preventDefault();
    }
  };

  const zoomMapAt = React.useCallback((clientX: number, clientY: number, deltaY: number) => {
    const rect = mapViewportRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const cursor = mapScreenToImage(x, y);
    const nextZoom = clampNumber(mapView.zoom * (deltaY > 0 ? 0.9 : 1.1), MIN_MAP_ZOOM, MAX_MAP_ZOOM);
    const nextRows = rows / nextZoom;
    const nextCols = cols / nextZoom;
    setMapView(
      nextZoom,
      cursor.row - (y / Math.max(1, size)) * nextRows,
      cursor.col - (x / Math.max(1, size)) * nextCols,
    );
  }, [cols, mapScreenToImage, mapView.zoom, rows, setMapView, size]);

  const onSpecDown = (e: React.MouseEvent) => {
    const rect = specCanvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const x0 = indexToSpecX(bandLo);
    const x1 = indexToSpecX(bandHi - 1);
    let mode: DragMode = null;
    const insideBand = x >= x0 && x <= x1;
    const bandPx = Math.abs(x1 - x0);
    if (insideBand && bandPx < 28) mode = "band-move";
    else if (Math.abs(x - x0) < 12) mode = "band-left";
    else if (Math.abs(x - x1) < 12) mode = "band-right";
    else if (insideBand) mode = "band-move";
    if (mode) {
      setDrag({ mode, x, y: e.clientY - rect.top, roi, bandStart: bandLo, bandEnd: bandHi });
      e.preventDefault();
    }
  };

  const zoomSpectrumAt = React.useCallback((clientX: number, deltaY: number) => {
    const rect = specCanvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = clientX - rect.left;
    const cursor = specXToIndex(x);
    const factor = deltaY > 0 ? 1.12 : 0.88;
    const nextSpan = clampNumber(spectrumView.span * factor, Math.min(MIN_SPECTRUM_SPAN, nEnergy), Math.max(1, nEnergy));
    const padL = 54;
    const padR = 14;
    const plotW = Math.max(1, specW - padL - padR);
    const frac = clampNumber((x - padL) / plotW, 0, 1);
    setSpectrumView(cursor - frac * nextSpan, cursor + (1 - frac) * nextSpan);
  }, [nEnergy, setSpectrumView, specW, specXToIndex, spectrumView.span]);

  React.useEffect(() => {
    const el = mapViewportRef.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      zoomMapAt(event.clientX, event.clientY, event.deltaY);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomMapAt]);

  React.useEffect(() => {
    const el = specCanvasRef.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      zoomSpectrumAt(event.clientX, event.deltaY);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomSpectrumAt]);

  React.useEffect(() => {
    if (!panelResize) return;
    const onMove = (e: MouseEvent) => {
      if (panelResize.mode === "map") {
        const delta = Math.max(e.clientX - panelResize.x, e.clientY - panelResize.y);
        setPanelWidth(Math.max(220, Math.min(900, Math.round(panelResize.width + delta))));
      } else {
        const dx = e.clientX - panelResize.x;
        const dy = e.clientY - panelResize.y;
        setSpectrumWidth(Math.max(320, Math.min(1100, Math.round(panelResize.width + dx))));
        setSpectrumHeight(Math.max(140, Math.min(620, Math.round(panelResize.height + dy))));
      }
    };
    const onUp = () => {
      setPanelResize(null);
      saveWidgetChanges();
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => { document.removeEventListener("mousemove", onMove); document.removeEventListener("mouseup", onUp); };
  }, [panelResize, saveWidgetChanges, setPanelWidth, setSpectrumHeight, setSpectrumWidth]);

  React.useEffect(() => {
    if (!drag) return;
    const onMove = (e: MouseEvent) => {
      if (drag.mode?.startsWith("roi")) {
        const rect = mapViewportRef.current?.getBoundingClientRect();
        if (!rect) return;
        const dx = (e.clientX - rect.left - drag.x) / size * mapView.cols;
        const dy = (e.clientY - rect.top - drag.y) / size * mapView.rows;
        if (drag.mode === "roi-move") updateRoi({ ...drag.roi, row: drag.roi.row + dy, col: drag.roi.col + dx }, true, true);
        else if (normalizeRoiShape(drag.roi.shape) === "circle") {
          const delta = Math.max(dx, dy);
          updateRoi({ ...drag.roi, width: drag.roi.width + delta, height: drag.roi.height + delta }, true, true);
        } else if (normalizeRoiShape(drag.roi.shape) === "ellipse") {
          updateRoi({ ...drag.roi, width: drag.roi.width + dx, height: drag.roi.height + dy }, true, true);
        } else {
          updateRoi({ ...drag.roi, width: drag.roi.width + dx, height: drag.roi.height + dy }, true, true);
        }
      } else if (drag.mode === "map-pan") {
        const rect = mapViewportRef.current?.getBoundingClientRect();
        if (!rect) return;
        const dx = (e.clientX - rect.left - drag.x) / size * mapView.cols;
        const dy = (e.clientY - rect.top - drag.y) / size * mapView.rows;
        setMapView(mapView.zoom, (drag.mapViewRow ?? mapView.row) - dy, (drag.mapViewCol ?? mapView.col) - dx);
      } else if (drag.mode?.startsWith("band")) {
        const rect = specCanvasRef.current?.getBoundingClientRect();
        if (!rect) return;
        const di = specXToIndex(e.clientX - rect.left) - specXToIndex(drag.x);
        if (drag.mode === "band-left") updateBand(drag.bandStart + di, drag.bandEnd, true, true);
        else if (drag.mode === "band-right") updateBand(drag.bandStart, drag.bandEnd + di, true, true);
        else previewCenterBand(drag.bandStart + di, drag.bandEnd + di);
      }
    };
    const onUp = () => {
      if (drag.mode?.startsWith("roi")) {
        flushRoiPersist();
        const latest = specRequestRef.current;
        if (latest) scheduleSpectrum(latest);
      }
      if (drag.mode?.startsWith("band")) {
        flushLocalBandPreview();
        flushBandPersist();
        const latest = mapRequestRef.current;
        if (latest) scheduleMap(latest.start, latest.end);
      }
      setDrag(null);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => { document.removeEventListener("mousemove", onMove); document.removeEventListener("mouseup", onUp); };
  }, [drag, flushBandPersist, flushLocalBandPreview, flushRoiPersist, mapView, previewCenterBand, setMapView, size, specXToIndex]);

  React.useEffect(() => {
    if (!bandSliderDrag) return;
    const onMove = (e: MouseEvent) => {
      const di = ((e.clientX - bandSliderDrag.x) / Math.max(1, bandSliderDrag.width)) * Math.max(1, nEnergy);
      previewCenterBand(bandSliderDrag.bandStart + di, bandSliderDrag.bandEnd + di, bandSliderDrag.width);
    };
    const onUp = () => {
      flushLocalBandPreview();
      flushBandPersist();
      const latest = mapRequestRef.current;
      if (latest) scheduleMap(latest.start, latest.end);
      setBandSliderDrag(null);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => { document.removeEventListener("mousemove", onMove); document.removeEventListener("mouseup", onUp); };
  }, [bandSliderDrag, flushBandPersist, flushLocalBandPreview, nEnergy, previewCenterBand]);

  const bandCounts = React.useMemo(() => {
    if (!roiSpectrum) return 0;
    let sum = 0;
    for (let i = bandLo; i < bandHi; i++) sum += roiSpectrum[i] || 0;
    return sum;
  }, [bandHi, bandLo, roiSpectrum]);
  const [autoMapContrastOn, setAutoMapContrastOn] = React.useState(false);

  const resetView = () => {
    setPanelWidth(420);
    setSpectrumWidth(640);
    setSpectrumHeight(250);
    setMapVminPct(2);
    setMapVmaxPct(98);
    setMapView(1, 0, 0);
    setSpectrumView(0, nEnergy);
    updateBand(bandLo, bandHi);
    updateRoi({
      row: Math.max(0, rows / 2 - rows / 8),
      col: Math.max(0, cols / 2 - cols / 8),
      height: Math.max(8, rows / 4),
      width: Math.max(8, cols / 4),
      shape: "rect",
    });
  };

  const autoMapContrast = () => {
    if (!elementMap) return;
    const span = mapDataRange[1] - mapDataRange[0];
    if (!Number.isFinite(span) || span <= 0) {
      setMapVminPct(0);
      setMapVmaxPct(100);
      return;
    }
    const clipped = percentileClip(elementMap, 2, 98);
    let lo = clipped.vmin;
    let hi = clipped.vmax;
    const eps = Math.max(1e-12, Math.abs(span) * 1e-6);
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi - lo <= eps) {
      lo = mapDataRange[0];
      hi = mapDataRange[1];
    }
    const loPct = Math.max(0, Math.min(99, ((lo - mapDataRange[0]) / span) * 100));
    const hiPct = Math.max(loPct + 1, Math.min(100, ((hi - mapDataRange[0]) / span) * 100));
    setMapVminPct(loPct);
    setMapVmaxPct(hiPct);
  };

  React.useEffect(() => {
    if (autoMapContrastOn) autoMapContrast();
  }, [autoMapContrastOn, elementMap]);

  const controlRowSx = {
    display: "flex",
    alignItems: "center",
    gap: 1,
    border: `1px solid ${themeColors.border}`,
    bgcolor: themeColors.controlBg,
    px: 1,
    py: 0.5,
    width: "fit-content",
    maxWidth: "100%",
    boxSizing: "border-box",
  } as const;
  const controlLabelSx = { fontSize: 10, color: themeColors.textMuted, flexShrink: 0, lineHeight: "20px" } as const;
  const compactSliderSx = {
    py: 0,
    height: 20,
    display: "flex",
    alignItems: "center",
    "& .MuiSlider-root": { height: 20 },
    "& .MuiSlider-thumb": { width: 12, height: 12 },
    "& .MuiSlider-rail": { height: 2 },
    "& .MuiSlider-track": { height: 2 },
  } as const;
  const compactButtonSx = {
    fontSize: 10,
    py: 0.25,
    px: 1,
    minWidth: 0,
    textTransform: "none",
    color: themeColors.accent,
    borderColor: themeColors.border,
    "&.Mui-disabled": {
      color: themeColors.textMuted,
      borderColor: themeColors.border,
      opacity: 0.55,
    },
  } as const;
  const themedMenuProps = {
    PaperProps: {
      sx: {
        bgcolor: themeColors.controlBg,
        color: themeColors.text,
        border: `1px solid ${themeColors.border}`,
        "& .MuiMenuItem-root": { fontSize: 12 },
      },
    },
  };

  return (
    <Box sx={{ p: 2, fontFamily: UI_FONT, bgcolor: themeColors.bg, color: themeColors.text, overflowX: "auto" }}>
      <Stack spacing={1.2}>
        <Stack direction="row" spacing={1} alignItems="center" sx={{ width: specW + size + 16 }}>
          <Stack direction="row" spacing={1} alignItems="center" sx={{ flex: 1, minWidth: 0 }}>
            <Typography sx={{ fontWeight: 700, fontSize: 14 }}>{title || "EDS spectrum image"}</Typography>
            <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
              {rows}x{cols}x{nEnergy} | {backendLabel} {elementLabel ? `| ${elementLabel}` : ""} {busy ? "| computing" : ""}
            </Typography>
          </Stack>
          {showControls && (
            <Stack direction="row" spacing={1} alignItems="center">
              <Button
                size="small"
                variant={safeSelectedElements.length > 0 ? "contained" : "outlined"}
                sx={{
                  ...compactButtonSx,
                  bgcolor: safeSelectedElements.length > 0 ? themeColors.accent : "transparent",
                  color: safeSelectedElements.length > 0 ? themeColors.buttonText : themeColors.accent,
                  "&:hover": {
                    bgcolor: safeSelectedElements.length > 0 ? themeColors.accent : themeColors.hoverBg,
                  },
                }}
                onClick={(e) => setElementMenuAnchor(e.currentTarget)}
                aria-label="Open EDS periodic table"
                aria-controls={elementMenuAnchor ? "showeds-elements-menu" : undefined}
                aria-expanded={elementMenuAnchor ? "true" : undefined}
                aria-haspopup="menu"
                title="Pick elements and characteristic lines"
              >
                Elements {safeSelectedElements.length || ""}
              </Button>
              <Menu
                id="showeds-elements-menu"
                anchorEl={elementMenuAnchor}
                open={Boolean(elementMenuAnchor)}
                onClose={() => setElementMenuAnchor(null)}
                MenuListProps={{ "aria-label": "ShowEDS periodic table and line picker" }}
                anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
                transformOrigin={{ vertical: "top", horizontal: "left" }}
                {...themedMenuProps}
                sx={{ zIndex: 9999 }}
              >
                <Box sx={{ p: 1.25, width: 680, maxWidth: "calc(100vw - 48px)" }}>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                    <Typography sx={{ fontWeight: 700, fontSize: 13, color: themeColors.text, flex: 1 }}>
                      Periodic table
                    </Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Auto ID</Typography>
                    <Switch checked={autoIdentify} onChange={(e) => setAutoIdentify(e.target.checked)} size="small" />
                    <Button
                      size="small"
                      variant="outlined"
                      sx={compactButtonSx}
                      onClick={() => setSelectedElements([])}
                      disabled={safeSelectedElements.length === 0}
                    >
                      Clear
                    </Button>
                  </Stack>
                  <Box
                    sx={{
                      display: "grid",
                      gridTemplateColumns: "repeat(18, 30px)",
                      gridTemplateRows: "repeat(9, 30px)",
                      gap: 0.35,
                      alignItems: "stretch",
                    }}
                  >
                    {PERIODIC_ELEMENTS.map((el) => {
                      const enabled = elementsWithLines.has(el.symbol);
                      const selected = selectedElementSet.has(el.symbol);
                      const scored = autoElementScores.some((item) => item.symbol === el.symbol);
                      return (
                        <Button
                          key={el.symbol}
                          size="small"
                          disabled={!enabled}
                          onClick={() => toggleSelectedElement(el.symbol)}
                          onDoubleClick={() => selectOnlyElement(el.symbol)}
                          title={`${el.name}${enabled ? ": click to toggle, double-click to isolate" : ": no line in current table"}`}
                          sx={{
                            gridColumn: el.group,
                            gridRow: el.period,
                            minWidth: 0,
                            width: 30,
                            height: 30,
                            p: 0,
                            borderRadius: "4px",
                            border: `1px solid ${selected ? themeColors.accent : scored ? themeColors.lineHintText : themeColors.border}`,
                            bgcolor: selected
                              ? themeColors.accent
                              : scored
                                ? themeColors.bandFill
                                : themeColors.controlBg,
                            color: selected ? themeColors.buttonText : themeColors.text,
                            fontSize: 10,
                            fontWeight: selected || scored ? 800 : 600,
                            opacity: enabled ? 1 : 0.28,
                            textTransform: "none",
                            "&.Mui-disabled": {
                              color: themeColors.textMuted,
                              borderColor: themeColors.border,
                            },
                            "&:hover": {
                              bgcolor: selected ? themeColors.accent : themeColors.hoverBg,
                              borderColor: themeColors.accent,
                            },
                          }}
                        >
                          {el.symbol}
                        </Button>
                      );
                    })}
                  </Box>
                  <Stack direction="row" spacing={1} sx={{ mt: 1 }} alignItems="flex-start">
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Typography sx={{ fontSize: 11, fontWeight: 700, color: themeColors.text, mb: 0.5 }}>
                        Auto-ID candidates
                      </Typography>
                      <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap">
                        {autoElementScores.length === 0 ? (
                          <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                            Move the band over peaks to rank likely elements.
                          </Typography>
                        ) : autoElementScores.map((item) => (
                          <Button
                            key={item.symbol}
                            size="small"
                            variant={selectedElementSet.has(item.symbol) ? "contained" : "outlined"}
                            sx={compactButtonSx}
                            onClick={() => toggleSelectedElement(item.symbol)}
                            title={`${item.symbol}: ${item.lines.slice(0, 3).map(lineLabel).join(", ")}`}
                          >
                            {item.symbol}
                          </Button>
                        ))}
                      </Stack>
                    </Box>
                    <Box sx={{ flex: 1.2, minWidth: 0 }}>
                      <Typography sx={{ fontSize: 11, fontWeight: 700, color: themeColors.text, mb: 0.5 }}>
                        Lines
                      </Typography>
                      <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap">
                        {suggestedLines.length === 0 ? (
                          <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                            Select elements to show their lines.
                          </Typography>
                        ) : suggestedLines.map((line) => (
                          <Button
                            key={`${line.element}-${line.line}-${line.energy_keV}`}
                            size="small"
                            variant="outlined"
                            sx={compactButtonSx}
                            onClick={() => snapBandToLine(line)}
                            title={`Center band on ${lineLabel(line)} at ${formatEnergy(line.energy_keV)}`}
                          >
                            {lineLabel(line)} {formatEnergy(line.energy_keV)}
                          </Button>
                        ))}
                      </Stack>
                    </Box>
                  </Stack>
                </Box>
              </Menu>
              <Typography sx={{ fontSize: 11, color: themeColors.text }}>Log</Typography>
              <Switch checked={logSpectrum} onChange={(e) => setLogSpectrum(e.target.checked)} size="small" />
              <Typography sx={{ fontSize: 11, color: themeColors.text }}>Scale</Typography>
              <Switch checked={scaleBarVisible} onChange={(e) => commitScaleBarVisible(e.target.checked)} size="small" />
              {debugControlVisible && (
                <>
                  <Typography sx={{ fontSize: 11, color: themeColors.text }}>Debug</Typography>
                  <Switch checked={showDebug} onChange={(e) => commitDebug(e.target.checked)} size="small" />
                </>
              )}
              {exportEnabled && (
                <>
                  <Button
                    size="small"
                    variant="outlined"
                    sx={compactButtonSx}
                    disabled={exportBusy}
                    onClick={handleExportMenuOpen}
                    aria-label="Export ShowEDS HTML"
                    aria-controls={exportMenuAnchor ? "showeds-export-menu" : undefined}
                    aria-expanded={exportMenuAnchor ? "true" : undefined}
                    aria-haspopup="menu"
                    title={localExportStatus || exportStatus || "Export current ShowEDS state as HTML"}
                  >
                    {exportBusy ? "Exporting" : "Export"}
                  </Button>
                  <Menu
                    id="showeds-export-menu"
                    anchorEl={exportMenuAnchor}
                    open={Boolean(exportMenuAnchor)}
                    onClose={handleExportMenuClose}
                    MenuListProps={{ "aria-label": "ShowEDS HTML export options" }}
                    anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
                    transformOrigin={{ vertical: "top", horizontal: "right" }}
                    {...themedMenuProps}
                    sx={{ zIndex: 9999 }}
                  >
                    {exportOptions.map((option, index) => (
                      <MenuItem key={`${option.mode}-${option.downsample ?? 0}-${index}`} onClick={() => handleExportSelect(option.mode, option.downsample)} sx={{ fontSize: 12 }}>
                        {option.label}
                      </MenuItem>
                    ))}
                  </Menu>
                </>
              )}
              <Button size="small" sx={compactButtonSx} variant="outlined" onClick={resetView}>Reset</Button>
              {exportEnabled && (localExportStatus || exportStatus) && (
                <Typography
                  sx={{
                    fontSize: 10,
                    maxWidth: 140,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    color: (localExportStatus || exportStatus).startsWith("Export failed") ? themeColors.error : themeColors.textMuted,
                  }}
                  title={localExportStatus || exportStatus}
                >
                  {localExportStatus || exportStatus}
                </Typography>
              )}
            </Stack>
          )}
        </Stack>
        {gpuError && <Typography sx={{ color: themeColors.error, fontSize: 12 }}>{gpuError}</Typography>}
        {showDebug && (
          <Box
            data-testid="showeds-perf-hud"
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1.25,
              width: specW + size + 16,
              px: 1,
              py: 0.5,
              bgcolor: themeColors.hudBg,
              color: themeColors.hudText,
              border: `1px solid ${themeColors.border}`,
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
              fontSize: 10,
              lineHeight: 1.2,
              boxSizing: "border-box",
              overflowX: "auto",
            }}
            aria-label="ShowEDS performance HUD"
          >
            <span>FPS {fps.toFixed(0)}</span>
            <span>map {perfSummary.map.last.toFixed(1)} / {perfSummary.map.avg.toFixed(1)} ms</span>
            <span>spectrum {perfSummary.spectrum.last.toFixed(1)} / {perfSummary.spectrum.avg.toFixed(1)} ms</span>
            <span>draw map {perfSummary.mapDraw.last.toFixed(1)} ms</span>
            <span>draw spec {perfSummary.spectrumDraw.last.toFixed(1)} ms</span>
            <span>backend map {backendSummary.map ?? "pending"}</span>
            <span>backend spectrum {backendSummary.spectrum ?? "pending"}</span>
          </Box>
        )}
        <Stack direction="row" spacing={2} alignItems="flex-start">
          <Box>
            <Box
              ref={mapViewportRef}
              onDoubleClick={() => setMapView(1, 0, 0)}
              sx={{ position: "relative", width: size, height: size, bgcolor: themeColors.mapBg, border: `1px solid ${themeColors.border}`, overflow: "hidden" }}
            >
              <canvas
                ref={mapCanvasRef}
                onMouseDown={onMapDown}
                style={{
                  ...mapLayerStyle,
                  cursor: drag?.mode === "map-pan" || drag?.mode?.startsWith("roi") ? "grabbing" : mapView.zoom > 1.001 ? "grab" : "crosshair",
                }}
                aria-label={`EDS real-space map${title ? `: ${title}` : ""}`}
              />
              <canvas
                ref={mapOverlayCanvasRef}
                style={{
                  ...mapLayerStyle,
                  pointerEvents: "none",
                  opacity: displayOverlayOpacity,
                }}
                aria-hidden="true"
              />
              <canvas
                ref={mapUiOverlayCanvasRef}
                style={{
                  position: "absolute",
                  left: 0,
                  top: 0,
                  width: size,
                  height: size,
                  pointerEvents: "none",
                  zIndex: 2,
                }}
                aria-hidden="true"
              />
              <Box
                sx={{
                  position: "absolute",
                  pointerEvents: "none",
                  boxSizing: "border-box",
                  left: `${roiOverlayStyle.left}px`,
                  top: `${roiOverlayStyle.top}px`,
                  width: `${roiOverlayStyle.width}px`,
                  height: `${roiOverlayStyle.height}px`,
                  border: `2px dashed ${themeColors.roi}`,
                  borderRadius: normalizeRoiShape(roi.shape) === "rect" ? 0 : "50%",
                  zIndex: 3,
                }}
              >
                <Box
                  sx={{
                    position: "absolute",
                    ...(normalizeRoiShape(roi.shape) !== "rect"
                      ? {
                        left: `${50 + 50 / Math.SQRT2}%`,
                        top: `${50 + 50 / Math.SQRT2}%`,
                        transform: "translate(-50%, -50%)",
                      }
                      : { right: -9, bottom: -9 }),
                    width: 18,
                    height: 18,
                    borderRadius: normalizeRoiShape(roi.shape) === "rect" ? 0 : "50%",
                    bgcolor: themeColors.roi,
                  }}
                />
              </Box>
              <Box
                onMouseDown={(e) => {
                  setPanelResize({ mode: "map", x: e.clientX, y: e.clientY, width: size, height: size });
                  e.preventDefault();
                  e.stopPropagation();
                }}
                sx={{
                  position: "absolute",
                  bottom: 0,
                  right: 0,
                  width: 16,
                  height: 16,
                  cursor: "nwse-resize",
                  opacity: 0.65,
                  zIndex: 4,
                  background: `linear-gradient(135deg, transparent 50%, ${themeColors.resize} 50%)`,
                  "&:hover": { opacity: 1 },
                }}
              />
            </Box>
            <Typography sx={{ mt: 0.5, fontSize: 11, color: themeColors.text }}>
              Overlay ROI: {normalizeRoiShape(roi.shape) === "circle"
                ? `circle row ${roi.row}, col ${roi.col}, d ${roi.width}`
                : normalizeRoiShape(roi.shape) === "ellipse"
                  ? `ellipse row ${roi.row}, col ${roi.col}, ${roi.height}x${roi.width}`
                  : `row ${roi.row}, col ${roi.col}, ${roi.height}x${roi.width}`}
            </Typography>
          </Box>
          <Box>
            <Box
              onDoubleClick={() => setSpectrumView(0, nEnergy)}
              sx={{ position: "relative", width: specW, overflow: "hidden" }}
            >
              <canvas
                ref={specCanvasRef}
                onMouseDown={onSpecDown}
                style={{ width: specW, height: specH, display: "block", cursor: drag?.mode?.startsWith("band") ? "grabbing" : "ew-resize", background: themeColors.plotBg, border: `1px solid ${themeColors.border}` }}
                aria-label={`EDS spectrum${title ? `: ${title}` : ""}`}
              />
              <Box
                ref={specBandOverlayRef}
                sx={{
                  position: "absolute",
                  top: 16,
                  left: 0,
                  height: Math.max(1, specH - 50),
                  width: 2,
                  transform: "translateX(54px)",
                  bgcolor: themeColors.bandFill,
                  pointerEvents: "none",
                  willChange: "transform, width",
                }}
              />
              <Box
                onMouseDown={(e) => {
                  setPanelResize({ mode: "spectrum", x: e.clientX, y: e.clientY, width: specW, height: specH });
                  e.preventDefault();
                  e.stopPropagation();
                }}
                sx={{
                  position: "absolute",
                  bottom: 0,
                  right: 0,
                  width: 16,
                  height: 16,
                  cursor: "nwse-resize",
                  opacity: 0.65,
                  background: `linear-gradient(135deg, transparent 50%, ${themeColors.resize} 50%)`,
                  "&:hover": { opacity: 1 },
                }}
              />
            </Box>
            <Typography ref={bandStatusRef} sx={{ mt: 0.5, fontSize: 11, color: themeColors.text }}>
              Band {bandLo}-{bandHi - 1}: {formatEnergy(energy[bandLo])} - {formatEnergy(energy[Math.max(bandLo, bandHi - 1)])}; ROI band counts {formatNumber(bandCounts, 2)}
              {candidateText ? `; candidates ${candidateText}` : ""}
            </Typography>
          </Box>
        </Stack>
        {showControls && (
          <Box sx={{ mt: 0.25, display: "flex", gap: 1, width: "fit-content", maxWidth: "100%", boxSizing: "border-box", alignItems: "flex-start", overflowX: "auto", pb: 0.5 }}>
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 0.5, flex: "0 0 auto", minWidth: 0 }}>
              <Box sx={controlRowSx}>
                <Typography sx={controlLabelSx}>Band:</Typography>
                <Box
                  ref={bandSliderRef}
                  sx={{ width: 260, flexShrink: 0, display: "flex", alignItems: "center", height: 20, position: "relative" }}
                >
                  <Box
                    ref={bandSliderPreviewRef}
                    sx={{
                      position: "absolute",
                      left: 0,
                      top: "50%",
                      width: 2,
                      height: 2,
                      transform: "translate(0, -50%)",
                      bgcolor: themeColors.sliderPreview,
                      borderRadius: 1,
                      opacity: isBandCenterPreviewing ? 1 : 0,
                      pointerEvents: "none",
                      zIndex: 2,
                      willChange: "transform, width",
                      "&::before, &::after": {
                        content: '""',
                        position: "absolute",
                        top: -5,
                        width: 12,
                        height: 12,
                        borderRadius: "50%",
                        bgcolor: themeColors.sliderPreview,
                        boxShadow: `0 0 0 2px ${themeColors.bg}`,
                      },
                      "&::before": { left: -6 },
                      "&::after": { right: -6 },
                    }}
                  />
                  <Box
                    onMouseDown={(e) => {
                      const rect = bandSliderRef.current?.getBoundingClientRect();
                      if (!rect) return;
                      previewCenterBand(bandLo, bandHi, rect.width);
                      setBandSliderDrag({ x: e.clientX, width: rect.width, bandStart: bandLo, bandEnd: bandHi });
                      e.preventDefault();
                      e.stopPropagation();
                      e.nativeEvent.stopImmediatePropagation();
                    }}
                    sx={{
                      position: "absolute",
                      top: 0,
                      bottom: 0,
                      left: `clamp(0px, calc(${bandMoveCenterPct}% - ${bandMoveHitWidthPx / 2}px), calc(100% - ${bandMoveHitWidthPx}px))`,
                      width: `${bandMoveHitWidthPx}px`,
                      cursor: isBandCenterPreviewing ? "grabbing" : "grab",
                      zIndex: 3,
                      pointerEvents: "auto",
                    }}
                    aria-label="Drag selected energy band"
                  />
                  <Slider
                    value={[bandLo, bandHi]}
                    min={0}
                    max={Math.max(1, nEnergy)}
                    step={1}
                    disableSwap
                    onChange={(_, v) => Array.isArray(v) && updateBand(v[0], v[1], true, true)}
                    onChangeCommitted={(_, v) => {
                      if (!Array.isArray(v)) return;
                      const s = Math.max(0, Math.min(nEnergy - 1, Math.round(v[0])));
                      const e = Math.max(s + 1, Math.min(nEnergy, Math.round(v[1])));
                      queueBandPersist(s, e, true);
                      scheduleMap(s, e);
                    }}
                    size="small"
                    sx={{
                      width: "100%",
                      ...compactSliderSx,
                      ...(isBandCenterPreviewing ? {
                        "& .MuiSlider-track, & .MuiSlider-thumb": { opacity: 0 },
                      } : {}),
                      "& .MuiSlider-track": { height: 2, cursor: isBandCenterPreviewing ? "grabbing" : "grab" },
                    }}
                  />
                </Box>
                <Typography sx={controlLabelSx}>Overlay:</Typography>
                <Slider
                  value={displayOverlayOpacity}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={(_, v) => setLocalOverlayOpacity(Number(v))}
                  onChangeCommitted={(_, v) => commitOverlayOpacity(Number(v))}
                  size="small"
                  sx={{ width: 100, flexShrink: 0, ...compactSliderSx }}
                />
                <Button size="small" sx={compactButtonSx} variant="outlined" onClick={saveCurrentBand}>Save Band</Button>
                <Button
                  size="small"
                  sx={compactButtonSx}
                  variant="outlined"
                  disabled={safeSavedBands.length === 0}
                  onClick={(e) => setBandMenuAnchor(e.currentTarget)}
                  aria-label="Saved energy band presets"
                  aria-controls={bandMenuAnchor ? "showeds-band-menu" : undefined}
                  aria-expanded={bandMenuAnchor ? "true" : undefined}
                  aria-haspopup="menu"
                >
                  Bands {safeSavedBands.length}
                </Button>
                <Menu
                  id="showeds-band-menu"
                  anchorEl={bandMenuAnchor}
                  open={Boolean(bandMenuAnchor)}
                  onClose={() => setBandMenuAnchor(null)}
                  MenuListProps={{ "aria-label": "ShowEDS saved energy band presets" }}
                  anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
                  transformOrigin={{ vertical: "top", horizontal: "right" }}
                  {...themedMenuProps}
                  sx={{ zIndex: 9999 }}
                >
                  {safeSavedBands.map((saved, index) => (
                    <MenuItem key={`${saved.name}-${index}`} onClick={() => applySavedBand(saved)} sx={{ fontSize: 12 }}>
                      {saved.name}: {formatEnergy(energy[saved.start])} - {formatEnergy(energy[Math.max(saved.start, saved.end - 1)])}
                    </MenuItem>
                  ))}
                  <MenuItem onClick={clearSavedBands} sx={{ fontSize: 12, color: themeColors.textMuted }}>Clear saved bands</MenuItem>
                </Menu>
              </Box>
              <Box sx={controlRowSx}>
                <Typography sx={controlLabelSx}>ROI:</Typography>
                <Button
                  size="small"
                  sx={compactButtonSx}
                  variant="outlined"
                  onClick={(e) => setRoiShapeMenuAnchor(e.currentTarget)}
                  aria-label="ROI shape"
                  aria-controls={roiShapeMenuAnchor ? "showeds-roi-shape-menu" : undefined}
                  aria-expanded={roiShapeMenuAnchor ? "true" : undefined}
                  aria-haspopup="menu"
                  title="Choose ROI shape"
                >
                  {ROI_SHAPE_LABELS[normalizeRoiShape(roi.shape)]}
                </Button>
                <Typography sx={controlLabelSx}>Auto:</Typography>
                <Switch
                  checked={autoMapContrastOn}
                  onChange={(e) => {
                    setAutoMapContrastOn(e.target.checked);
                    if (e.target.checked) autoMapContrast();
                  }}
                  size="small"
                  sx={{ flexShrink: 0, my: 0 }}
                />
                <Typography
                  sx={controlLabelSx}
                  title="CSS bilinear interpolation. Same data, browser smooths visually when upscaling."
                >
                  Smooth:
                </Typography>
                <Switch
                  checked={Boolean(smooth)}
                  onChange={(e) => setSmooth(e.target.checked)}
                  size="small"
                  sx={{ flexShrink: 0, my: 0 }}
                />
                <Menu
                  id="showeds-roi-shape-menu"
                  anchorEl={roiShapeMenuAnchor}
                  open={Boolean(roiShapeMenuAnchor)}
                  onClose={() => setRoiShapeMenuAnchor(null)}
                  MenuListProps={{ "aria-label": "ShowEDS ROI shape options" }}
                  anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
                  transformOrigin={{ vertical: "top", horizontal: "right" }}
                  {...themedMenuProps}
                  sx={{ zIndex: 9999 }}
                >
                  <MenuItem onClick={() => applyRoiShape("rect")} sx={{ fontSize: 12 }}>Rect</MenuItem>
                  <MenuItem onClick={() => applyRoiShape("circle")} sx={{ fontSize: 12 }}>Circle</MenuItem>
                  <MenuItem onClick={() => applyRoiShape("ellipse")} sx={{ fontSize: 12 }}>Ellipse</MenuItem>
                </Menu>
                <Button size="small" sx={compactButtonSx} variant="outlined" onClick={saveCurrentRoi}>Save ROI</Button>
                <Button
                  size="small"
                  sx={compactButtonSx}
                  variant="outlined"
                  disabled={safeSavedRois.length === 0}
                  onClick={(e) => setRoiMenuAnchor(e.currentTarget)}
                  aria-label="Saved ROI presets"
                  aria-controls={roiMenuAnchor ? "showeds-roi-menu" : undefined}
                  aria-expanded={roiMenuAnchor ? "true" : undefined}
                  aria-haspopup="menu"
                >
                  ROIs {safeSavedRois.length}
                </Button>
                <Menu
                  id="showeds-roi-menu"
                  anchorEl={roiMenuAnchor}
                  open={Boolean(roiMenuAnchor)}
                  onClose={() => setRoiMenuAnchor(null)}
                  MenuListProps={{ "aria-label": "ShowEDS saved ROI presets" }}
                  anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
                  transformOrigin={{ vertical: "top", horizontal: "right" }}
                  {...themedMenuProps}
                  sx={{ zIndex: 9999 }}
                >
                  {safeSavedRois.map((saved, index) => (
                    <MenuItem key={`${saved.name}-${index}`} onClick={() => applySavedRoi(saved)} sx={{ fontSize: 12 }}>
                      {saved.name}: {ROI_SHAPE_LABELS[normalizeRoiShape(saved.shape)]} row {saved.row}, col {saved.col}, {saved.height}x{saved.width}
                    </MenuItem>
                  ))}
                  <MenuItem onClick={clearSavedRois} sx={{ fontSize: 12, color: themeColors.textMuted }}>Clear saved ROIs</MenuItem>
                </Menu>
              </Box>
            </Box>
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "flex-start", gap: 0.5, flexShrink: 0 }}>
              <MapHistogram
                data={elementMap}
                vminPct={mapVminPct}
                vmaxPct={mapVmaxPct}
                dataMin={mapDataRange[0]}
                dataMax={mapDataRange[1]}
                theme={themeInfo.theme === "dark" ? "dark" : "light"}
                width={110}
                height={58}
                onRangeChange={(lo, hi) => {
                  if (autoMapContrastOn) setAutoMapContrastOn(false);
                  setMapVminPct(lo);
                  setMapVmaxPct(hi);
                }}
                onRangePreview={(lo, hi) => {
                  drawMapOverlay(mapPercentRange(lo, hi));
                }}
                onRangeCommit={(lo, hi) => {
                  if (autoMapContrastOn) setAutoMapContrastOn(false);
                  drawMapOverlay(mapPercentRange(lo, hi));
                  setMapVminPct(lo);
                  setMapVmaxPct(hi);
                }}
              />
            </Box>
          </Box>
        )}
      </Stack>
    </Box>
  );
}

const render = createRender(ShowEDS);
export default { render };
