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
import { downloadBlob, extractBytes, extractFloat32, formatNumber, markWidgetNotebookDirty, preserveRestoredWidgetModelsOnSave } from "../format";
import { computeHistogramFromBytes, percentileClip, sliderRange } from "../stats";

type Roi = { row: number; col: number; height: number; width: number };
type DragMode = "roi-move" | "roi-resize" | "band-move" | "band-left" | "band-right" | null;
type EdsLineHint = { element: string; line: string; family?: string; energy_keV: number; intensity?: number };
type SavedRoi = Roi & { name?: string };
type SavedBand = { name?: string; start: number; end: number };
type ExportPreset = { label?: string; mode?: string; downsample?: number; binning?: number; description?: string };
type PanelResize = { mode: "map" | "spectrum"; x: number; y: number; width: number; height: number } | null;
type BandSliderDrag = { x: number; width: number; bandStart: number; bandEnd: number } | null;
type NumericArray = Float32Array | Float64Array | Uint32Array;
type PerfKind = "mapMs" | "spectrumMs" | "mapDrawMs" | "spectrumDrawMs";
type PerfStat = { last: number; avg: number; max: number };
type SidecarMeta = {
  rows: number;
  cols: number;
  n_energy: number;
  energy_prefix: string;
  spatial_prefix: string;
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
  buffer?: ArrayBuffer;
  ms?: number;
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
};

const WORKGROUP = 64;
const INTERACTION_COMPUTE_INTERVAL_MS = 32;
const UI_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
const HISTOGRAM_LIGHT_COLORS = { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" };
const HTML_EXPORT_OVERHEAD_BYTES = 700_000;
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
@group(0) @binding(3) var<uniform> q: vec4<u32>; // c0, r1, c1, mode: 0=f32 bits, 1=uint16 counts, 2=uint32 counts

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
  var sum = 0.0;
  for (var r = p.w; r < q.y; r = r + 1u) {
    for (var c = q.x; c < q.z; c = c + 1u) {
      sum = sum + sample((r * p.y + c) * p.z + e, q.w);
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
  dataMin,
  dataMax,
  width = 128,
  height = 58,
}: {
  data: NumericArray | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  dataMin: number;
  dataMax: number;
  width?: number;
  height?: number;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const sliderRef = React.useRef<HTMLDivElement | null>(null);
  const onRangeChangeRef = React.useRef(onRangeChange);
  const [rangeDrag, setRangeDrag] = React.useState<{ x: number; width: number; lo: number; hi: number } | null>(null);
  const pendingRangeRef = React.useRef<[number, number] | null>(null);
  const rangeRafRef = React.useRef<number | null>(null);
  const bins = React.useMemo(
    () => computeHistogramFromBytes(data, 256, dataMin, dataMax),
    [data, dataMin, dataMax],
  );
  const colors = HISTOGRAM_LIGHT_COLORS;
  React.useEffect(() => {
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
    const vminBin = Math.floor((Math.max(0, Math.min(100, vminPct)) / 100) * displayBins);
    const vmaxBin = Math.floor((Math.max(0, Math.min(100, vmaxPct)) / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      ctx.fillStyle = i >= vminBin && i <= vmaxBin ? colors.barActive : colors.barInactive;
      ctx.fillRect(i * barWidth + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [bins, colors, height, vmaxPct, vminPct, width]);
  const valueLabel = (pct: number) => {
    const val = dataMin + (pct / 100) * (dataMax - dataMin);
    return Math.abs(val) >= 1000 ? val.toExponential(1) : val.toFixed(1);
  };
  React.useEffect(() => {
    onRangeChangeRef.current = onRangeChange;
  }, [onRangeChange]);
  const flushRangePreview = React.useCallback(() => {
    if (rangeRafRef.current != null) {
      window.cancelAnimationFrame(rangeRafRef.current);
      rangeRafRef.current = null;
    }
    const pending = pendingRangeRef.current;
    pendingRangeRef.current = null;
    if (pending) onRangeChangeRef.current(pending[0], pending[1]);
  }, []);
  React.useEffect(() => () => {
    if (rangeRafRef.current != null) window.cancelAnimationFrame(rangeRafRef.current);
  }, []);
  React.useEffect(() => {
    if (!rangeDrag) return;
    const onMove = (e: MouseEvent) => {
      const span = Math.max(1, rangeDrag.hi - rangeDrag.lo);
      const deltaPct = ((e.clientX - rangeDrag.x) / Math.max(1, rangeDrag.width)) * 100;
      const lo = Math.max(0, Math.min(100 - span, rangeDrag.lo + deltaPct));
      pendingRangeRef.current = [lo, lo + span];
      if (rangeRafRef.current == null) {
        rangeRafRef.current = window.requestAnimationFrame(() => {
          rangeRafRef.current = null;
          const pending = pendingRangeRef.current;
          pendingRangeRef.current = null;
          if (pending) onRangeChangeRef.current(pending[0], pending[1]);
        });
      }
    };
    const onUp = () => {
      flushRangePreview();
      setRangeDrag(null);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }, [flushRangePreview, rangeDrag]);
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25, width, flexShrink: 0 }}>
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
          const [lo, hi] = clampPercentPair(vminPct, vmaxPct);
          const pct = ((e.clientX - rect.left) / Math.max(1, rect.width)) * 100;
          if (pct < lo || pct > hi) return;
          setRangeDrag({ x: e.clientX, width: rect.width, lo, hi });
          e.preventDefault();
          e.stopPropagation();
          e.nativeEvent.stopImmediatePropagation();
        }}
        sx={{ width, height: 14, display: "flex", alignItems: "center", cursor: rangeDrag ? "grabbing" : "grab" }}
      >
        <Slider
          value={clampPercentPair(vminPct, vmaxPct)}
          min={0}
          max={100}
          step={0.5}
          size="small"
          valueLabelDisplay="auto"
          valueLabelFormat={valueLabel}
          onChange={(_, v) => {
            if (!Array.isArray(v)) return;
            const [newMin, newMax] = v.map(Number);
            onRangeChange(Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1));
          }}
          sx={{
            width,
            py: 0,
            "& .MuiSlider-thumb": { width: 8, height: 8 },
            "& .MuiSlider-rail": { height: 2 },
            "& .MuiSlider-track": { height: 2, cursor: rangeDrag ? "grabbing" : "grab" },
            "& .MuiSlider-valueLabel": { fontSize: 10, px: 0.5, py: 0.25 },
          }}
        />
      </Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{valueLabel(vminPct)}</Typography>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{valueLabel(vmaxPct)}</Typography>
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

function createSidecarWorker(): Worker {
  const source = `
let meta = null;
let baseUrl = "";
let prefixCache = new Map();
let spectrumCache = new Map();
let fetchRequestId = 0;
let fetchPending = new Map();
let activeMapController = null;
let activeSpectrumController = null;

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
  try {
    const resp = await fetch(sidecarFileUrl(name), {
      credentials: "include",
      headers: { Range: "bytes=" + start + "-" + end },
      signal,
    });
    if (resp.status !== 206) throw new Error("range fetch failed: " + resp.status);
    return await resp.arrayBuffer();
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
  if (spectrumCache.size > 512) spectrumCache.delete(spectrumCache.keys().next().value);
  return arr;
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
      fetchPending = new Map();
      activeMapController?.abort();
      activeSpectrumController?.abort();
      activeMapController = null;
      activeSpectrumController = null;
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
    if (msg.type === "spectrum") {
      activeSpectrumController?.abort();
      const controller = new AbortController();
      activeSpectrumController = controller;
      try {
        const r0 = Math.max(0, Math.min(meta.rows - 1, Math.round(msg.row)));
        const c0 = Math.max(0, Math.min(meta.cols - 1, Math.round(msg.col)));
        const r1 = Math.max(r0 + 1, Math.min(meta.rows, r0 + Math.round(msg.height)));
        const c1 = Math.max(c0 + 1, Math.min(meta.cols, c0 + Math.round(msg.width)));
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
  const [energy] = useModelState<number[]>("energy_keV");
  const [bandStart, setBandStart] = useModelState<number>("band_start");
  const [bandEnd, setBandEnd] = useModelState<number>("band_end");
  const [roiRow, setRoiRow] = useModelState<number>("roi_row");
  const [roiCol, setRoiCol] = useModelState<number>("roi_col");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [panelWidth, setPanelWidth] = useModelState<number>("panel_width_px");
  const [spectrumWidth, setSpectrumWidth] = useModelState<number>("spectrum_width_px");
  const [spectrumHeight, setSpectrumHeight] = useModelState<number>("spectrum_height_px");
  const [showControls] = useModelState<boolean>("show_controls");
  const [logSpectrum, setLogSpectrum] = useModelState<boolean>("log_spectrum");
  const [mapVminPct, setMapVminPct] = useModelState<number>("map_vmin_pct");
  const [mapVmaxPct, setMapVmaxPct] = useModelState<number>("map_vmax_pct");
  const [overlayOpacity, setOverlayOpacity] = useModelState<number>("overlay_opacity");
  const [elementLabel] = useModelState<string>("element_label");
  const [showLineHints] = useModelState<boolean>("show_line_hints");
  const [lineHints] = useModelState<EdsLineHint[]>("line_hints");
  const [showDebug, setShowDebug] = useModelState<boolean>("show_debug");
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
  const mapOverlayCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const specCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const specBandOverlayRef = React.useRef<HTMLDivElement | null>(null);
  const bandSliderRef = React.useRef<HTMLDivElement | null>(null);
  const bandSliderPreviewRef = React.useRef<HTMLDivElement | null>(null);
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
  const [roiSpectrum, setRoiSpectrum] = React.useState<NumericArray | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [localBand, setLocalBand] = React.useState<[number, number] | null>(null);
  const [localRoi, setLocalRoi] = React.useState<Roi | null>(null);
  const [localOverlayOpacity, setLocalOverlayOpacity] = React.useState<number | null>(null);
  const [drag, setDrag] = React.useState<{ mode: DragMode; x: number; y: number; roi: Roi; bandStart: number; bandEnd: number } | null>(null);
  const [panelResize, setPanelResize] = React.useState<PanelResize>(null);
  const [bandSliderDrag, setBandSliderDrag] = React.useState<BandSliderDrag>(null);
  const [exportMenuAnchor, setExportMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [roiMenuAnchor, setRoiMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [bandMenuAnchor, setBandMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [exportBusy, setExportBusy] = React.useState(false);
  const [localExportStatus, setLocalExportStatus] = React.useState("");
  const [perfTick, setPerfTick] = React.useState(0);
  const [fps, setFps] = React.useState(0);
  const mapRequestRef = React.useRef<{ start: number; end: number; interactive: boolean } | null>(null);
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
      row: Math.max(0, Math.min(rows - 1, Math.round(Number(item?.row ?? 0)))),
      col: Math.max(0, Math.min(cols - 1, Math.round(Number(item?.col ?? 0)))),
      height: Math.max(1, Math.min(rows, Math.round(Number(item?.height ?? 1)))),
      width: Math.max(1, Math.min(cols, Math.round(Number(item?.width ?? 1)))),
    })), [cols, rows, savedRois]);
  const safeSavedBands = React.useMemo(() => (Array.isArray(savedBands) ? savedBands : [])
    .map((item, index) => {
      const start = Math.max(0, Math.min(nEnergy - 1, Math.round(Number(item?.start ?? 0))));
      const end = Math.max(start + 1, Math.min(nEnergy, Math.round(Number(item?.end ?? start + 1))));
      return { name: String(item?.name || `Band ${index + 1}`), start, end };
    }), [nEnergy, savedBands]);
  const isKernelBackend = computeBackend === "kernel";
  const isSidecarBackend = computeBackend === "sidecar";
  const size = Math.max(180, Math.round(panelWidth || 420));
  const specW = Math.max(320, Math.round(spectrumWidth || size + 220));
  const specH = Math.max(140, Math.round(spectrumHeight || Math.round(size * 0.58)));
  const modelRoi: Roi = { row: roiRow, col: roiCol, height: roiHeight, width: roiWidth };
  const roi: Roi = localRoi ?? modelRoi;
  const rawBandStart = localBand?.[0] ?? bandStart;
  const rawBandEnd = localBand?.[1] ?? bandEnd;
  const bandLo = Math.max(0, Math.min(nEnergy - 1, Math.round(rawBandStart)));
  const bandHi = Math.max(bandLo + 1, Math.min(nEnergy, Math.round(rawBandEnd)));
  const displayOverlayOpacity = Math.max(0, Math.min(1, localOverlayOpacity ?? overlayOpacity));
  const bandEnergyLo = Math.min(energy[bandLo] ?? 0, energy[Math.max(bandLo, bandHi - 1)] ?? 0);
  const bandEnergyHi = Math.max(energy[bandLo] ?? 0, energy[Math.max(bandLo, bandHi - 1)] ?? 0);
  const positionBandPreview = React.useCallback((start: number, end: number, sliderWidth?: number): [number, number] => {
    const s = Math.max(0, Math.min(nEnergy - 1, Math.round(start)));
    const e = Math.max(s + 1, Math.min(nEnergy, Math.round(end)));
    const specEl = specBandOverlayRef.current;
    if (specEl) {
      const padL = 54;
      const padR = 14;
      const plotW = Math.max(1, specW - padL - padR);
      const x0 = padL + (s / Math.max(1, nEnergy - 1)) * plotW;
      const x1 = padL + ((e - 1) / Math.max(1, nEnergy - 1)) * plotW;
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
    return [s, e];
  }, [nEnergy, specW]);
  const previewCenterBand = React.useCallback((start: number, end: number, sliderWidth?: number) => {
    const [s, e] = positionBandPreview(start, end, sliderWidth);
    pendingLocalBandRef.current = [s, e];
    pendingBandPersistRef.current = [s, e];
    mapRequestRef.current = { start: s, end: e, interactive: true };
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
  const candidateLines = React.useMemo(() => {
    if (!showLineHints || !Array.isArray(lineHints)) return [];
    const step = Math.abs((energy[Math.min(nEnergy - 1, bandLo + 1)] ?? bandEnergyHi) - (energy[bandLo] ?? bandEnergyLo)) || 0.02;
    const lo = bandEnergyLo - step * 0.65;
    const hi = bandEnergyHi + step * 0.65;
    return lineHints
      .filter((line) => Number.isFinite(line.energy_keV) && line.energy_keV >= lo && line.energy_keV <= hi)
      .sort((a, b) => (b.intensity ?? 0) - (a.intensity ?? 0))
      .slice(0, 4);
  }, [bandEnergyHi, bandEnergyLo, bandLo, energy, lineHints, nEnergy, showLineHints]);
  const candidateText = candidateLines.map(lineLabel).join(", ");
  React.useEffect(() => {
    if (localOverlayOpacity === null) return;
    if (Math.abs(localOverlayOpacity - overlayOpacity) <= 1e-6) setLocalOverlayOpacity(null);
  }, [localOverlayOpacity, overlayOpacity]);
  const roiOverlayStyle = React.useMemo(() => {
    const left = (roi.col / Math.max(1, cols)) * size;
    const top = (roi.row / Math.max(1, rows)) * size;
    const width = (roi.width / Math.max(1, cols)) * size;
    const height = (roi.height / Math.max(1, rows)) * size;
    return { left, top, width, height };
  }, [cols, roi.col, roi.height, roi.row, roi.width, rows, size]);
  const backendLabel = isSidecarBackend ? "Data folder" : isKernelBackend ? "Kernel exact" : "WebGPU";
  const embeddedPayloadBytes =
    (cube?.byteLength ?? 0)
    + (base?.byteLength ?? rows * cols * 4)
    + (initialMap?.byteLength ?? 0)
    + (initialSpectrum?.byteLength ?? 0)
    + nEnergy * 4;
  const embeddedExportSize = formatEstimatedHtmlSize(embeddedPayloadBytes);
  const exactExportLabel = isSidecarBackend
    ? `Exact linked folder (${embeddedExportSize} HTML + ${exportSidecarBytes > 0 ? formatBytes(exportSidecarBytes) : "data folder"})`
    : `Exact single file (${embeddedExportSize})`;
  const exportOptions = React.useMemo(() => {
    const custom = Array.isArray(exportPresets)
      ? exportPresets
          .map((preset) => ({
            mode: String(preset.mode || (isSidecarBackend ? "folder" : "single")),
            downsample: preset.downsample === undefined
              ? (preset.binning === undefined ? undefined : Number(preset.binning))
              : Number(preset.downsample),
            label: String(preset.label || preset.description || ""),
          }))
          .filter((preset) => preset.label)
      : [];
    if (custom.length > 0) return custom;
    const options: { mode: string; downsample?: number; label: string }[] = [
      { mode: isSidecarBackend ? "folder" : "single", label: exactExportLabel },
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
  }, [cols, exactExportLabel, exportPresets, isSidecarBackend, nEnergy, rows]);

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
    if (!isSidecarBackend || !sidecarUrl) return;
    let disposed = false;
    const rejectPending = (message: string) => {
      for (const pending of sidecarPendingRef.current.values()) pending.reject(new Error(message));
      sidecarPendingRef.current.clear();
    };
    (async () => {
      try {
        const absoluteSidecarUrl = new URL(sidecarUrl, window.location.href).href;
        const meta = await (await fetch(new URL("meta.json", absoluteSidecarUrl), { credentials: "include" })).json() as SidecarMeta;
        if (disposed) return;
        sidecarRef.current?.worker.terminate();
        rejectPending("EDS folder data worker was replaced");
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
            const start = Math.max(0, Math.floor(response.start ?? 0));
            const end = Math.max(start, Math.floor(response.end ?? start));
            (async () => {
              try {
                const resp = await fetch(sidecarFileUrl(name), {
                  credentials: "include",
                  headers: { Range: `bytes=${start}-${end}` },
                });
                if (resp.status !== 206) throw new Error(`range fetch failed: ${resp.status}`);
                const buffer = await resp.arrayBuffer();
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
          rejectPending(event.message || "EDS folder data worker failed");
        };
        worker.postMessage({ type: "init", meta, baseUrl: absoluteSidecarUrl });
        sidecarRef.current = { meta, worker };
        setGpuError("");
      } catch (err) {
        if (!disposed) setGpuError(`Could not load EDS data folder: ${err instanceof Error ? err.message : String(err)}`);
      }
    })();
    return () => {
      disposed = true;
      sidecarRef.current?.worker.terminate();
      sidecarRef.current = null;
      rejectPending("EDS folder data worker was disposed");
    };
  }, [isSidecarBackend, sidecarUrl]);

  React.useEffect(() => {
    const handler = (content: { type?: string; message?: string }, buffers?: DataView[]) => {
      const first = buffers?.[0];
      if (content.type === "map" && first) {
        setElementMap(extractFloat32(first, rows * cols));
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
      if (isKernelBackend || isSidecarBackend || !cube || cube.length === 0 || rows <= 0 || cols <= 0 || nEnergy <= 0) return;
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
  }, [cube, rows, cols, nEnergy, isKernelBackend, isSidecarBackend]);

  React.useEffect(() => {
    return () => {
      if (mapRafRef.current != null) window.cancelAnimationFrame(mapRafRef.current);
      if (specRafRef.current != null) window.cancelAnimationFrame(specRafRef.current);
      if (mapThrottleTimerRef.current != null) window.clearTimeout(mapThrottleTimerRef.current);
      if (specThrottleTimerRef.current != null) window.clearTimeout(specThrottleTimerRef.current);
      if (bandPersistTimerRef.current != null) window.clearTimeout(bandPersistTimerRef.current);
      if (roiPersistTimerRef.current != null) window.clearTimeout(roiPersistTimerRef.current);
      if (localBandRafRef.current != null) window.cancelAnimationFrame(localBandRafRef.current);
    };
  }, []);

  const requestSidecarWorker = React.useCallback((request: Record<string, number | string>) => {
    const worker = sidecarRef.current?.worker;
    if (!worker) return Promise.reject(new Error("EDS folder data worker is not ready"));
    const id = ++sidecarWorkerRequestIdRef.current;
    return new Promise<SidecarWorkerResponse>((resolve, reject) => {
      sidecarPendingRef.current.set(id, { resolve, reject });
      worker.postMessage({ ...request, id });
    });
  }, []);

  const computeMap = React.useCallback(async (start: number, end: number) => {
    if (isSidecarBackend) {
      if (!sidecarRef.current || !sidecarUrl) return;
      const seq = ++mapComputeSeqRef.current;
      const t0 = performance.now();
      const s = Math.max(0, Math.min(nEnergy - 1, Math.round(start)));
      const e = Math.max(s + 1, Math.min(nEnergy, Math.round(end)));
      const response = await requestSidecarWorker({ type: "map", start: s, end: e });
      if (response.aborted) return;
      if (seq !== mapComputeSeqRef.current) return;
      if (!response.buffer) throw new Error("EDS folder data worker returned an empty map");
      const buffer = response.buffer;
      React.startTransition(() => setElementMap(new Uint32Array(buffer)));
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
    recordWidgetPerf("mapMs", performance.now() - t0);
  }, [cols, cubeDtype, isKernelBackend, isSidecarBackend, model, nEnergy, recordWidgetPerf, rows, requestSidecarWorker, sidecarUrl]);

  const computeSpectrum = React.useCallback(async (nextRoi: Roi) => {
    if (isSidecarBackend) {
      if (!sidecarRef.current || !sidecarUrl) return;
      const seq = ++spectrumComputeSeqRef.current;
      const t0 = performance.now();
      const r0 = Math.max(0, Math.min(rows - 1, Math.round(nextRoi.row)));
      const c0 = Math.max(0, Math.min(cols - 1, Math.round(nextRoi.col)));
      const r1 = Math.max(r0 + 1, Math.min(rows, r0 + Math.round(nextRoi.height)));
      const c1 = Math.max(c0 + 1, Math.min(cols, c0 + Math.round(nextRoi.width)));
      const response = await requestSidecarWorker({ type: "spectrum", row: r0, col: c0, height: r1 - r0, width: c1 - c0 });
      if (response.aborted) return;
      if (seq !== spectrumComputeSeqRef.current) return;
      if (!response.buffer) throw new Error("EDS folder data worker returned an empty spectrum");
      const buffer = response.buffer;
      React.startTransition(() => setRoiSpectrum(new Uint32Array(buffer)));
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
    gpu.device.queue.writeBuffer(gpu.specParamsA, 0, new Uint32Array([rows, cols, nEnergy, r0]));
    gpu.device.queue.writeBuffer(gpu.specParamsB, 0, new Uint32Array([c0, r1, c1, cubeDtype === "uint16" ? 1 : cubeDtype === "uint32" ? 2 : 0]));
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
  }, [cols, cubeDtype, isKernelBackend, isSidecarBackend, model, nEnergy, recordWidgetPerf, rows, requestSidecarWorker, sidecarUrl]);

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
    if (!isSidecarBackend) setBusy(true);
    void computeMap(req.start, req.end).catch((err) => {
      setGpuError(err instanceof Error ? err.message : String(err));
    }).finally(() => {
      if (allowConcurrent) return;
      mapRunningRef.current = false;
      if (mapRerunRef.current) {
        mapRerunRef.current = false;
        const latest = mapRequestRef.current;
        if (latest) scheduleMap(latest.start, latest.end, latest.interactive);
      } else if (!isSidecarBackend) {
        setBusy(false);
      }
    });
  }, [computeMap, isSidecarBackend]);

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
  }, [computeSpectrum, isSidecarBackend]);

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
    if (!gpuRef.current && !isSidecarBackend) return;
    scheduleMap(bandStart, bandEnd);
  }, [bandEnd, bandStart, isSidecarBackend, scheduleMap, gpuRef.current]);
  React.useEffect(() => {
    if (!gpuRef.current && !isSidecarBackend) return;
    scheduleSpectrum(modelRoi);
  }, [isSidecarBackend, roiCol, roiHeight, roiRow, roiWidth, scheduleSpectrum, gpuRef.current]);
  React.useEffect(() => {
    const id = window.setTimeout(() => {
      if (gpuRef.current || isSidecarBackend) {
        scheduleMap(bandStart, bandEnd);
        scheduleSpectrum(modelRoi);
      }
    }, 100);
    return () => window.clearTimeout(id);
  }, [bandEnd, bandStart, isSidecarBackend, roiCol, roiHeight, roiRow, roiWidth, scheduleMap, scheduleSpectrum, gpuRef.current]);

  React.useEffect(() => {
    const canvas = mapCanvasRef.current;
    if (!canvas || !base) return;
    const t0 = performance.now();
    canvas.width = Math.max(1, cols);
    canvas.height = Math.max(1, rows);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const image = ctx.createImageData(canvas.width, canvas.height);
    const [baseLo, baseHi] = finiteRange(base);
    const baseSpan = Math.max(1e-12, baseHi - baseLo);
    for (let py = 0; py < canvas.height; py++) {
      const r = Math.min(rows - 1, py);
      for (let px = 0; px < canvas.width; px++) {
        const c = Math.min(cols - 1, px);
        const idx = r * cols + c;
        const off = (py * canvas.width + px) * 4;
        const g = Math.round(255 * Math.max(0, Math.min(1, (base[idx] - baseLo) / baseSpan)));
        image.data[off] = g; image.data[off + 1] = g; image.data[off + 2] = g; image.data[off + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
    recordWidgetPerf("mapDrawMs", performance.now() - t0);
  }, [base, cols, recordWidgetPerf, rows]);

  React.useEffect(() => {
    const canvas = mapOverlayCanvasRef.current;
    if (!canvas) return;
    canvas.width = Math.max(1, cols);
    canvas.height = Math.max(1, rows);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!elementMap) return;
    const t0 = performance.now();
    const image = ctx.createImageData(canvas.width, canvas.height);
    const [mapLo, mapHi] = mapDisplayRange;
    const mapSpan = Math.max(1e-12, mapHi - mapLo);
    for (let py = 0; py < canvas.height; py++) {
      const r = Math.min(rows - 1, py);
      for (let px = 0; px < canvas.width; px++) {
        const c = Math.min(cols - 1, px);
        const idx = r * cols + c;
        const off = (py * canvas.width + px) * 4;
        const value = elementMap[idx];
        if (!Number.isFinite(value)) continue;
        const [cr, cg, cb] = colorize((value - mapLo) / mapSpan);
        image.data[off] = cr; image.data[off + 1] = cg; image.data[off + 2] = cb; image.data[off + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
    recordWidgetPerf("mapDrawMs", performance.now() - t0);
  }, [cols, elementMap, mapDisplayRange, recordWidgetPerf, rows]);

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
    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const padL = 54 * dpr, padR = 14 * dpr, padT = 16 * dpr, padB = 34 * dpr;
    const plotW = canvas.width - padL - padR;
    const plotH = canvas.height - padT - padB;
    const values = roiSpectrum;
    const transformed = new Float32Array(values.length);
    for (let i = 0; i < values.length; i++) transformed[i] = logSpectrum ? Math.log10(Math.max(1, values[i])) : values[i];
    const [lo, hi] = finiteRange(transformed);
    ctx.strokeStyle = "#333";
    ctx.lineWidth = dpr;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (i / 4) * plotH;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
    }
    if (showLineHints && Array.isArray(lineHints) && energy.length > 1) {
      const e0 = energy[0];
      const e1 = energy[energy.length - 1];
      const span = e1 - e0;
      if (Number.isFinite(span) && span !== 0) {
        const visibleLines = lineHints
          .filter((line) => line.intensity === undefined || line.intensity >= 0.04 || (line.energy_keV >= bandEnergyLo && line.energy_keV <= bandEnergyHi));
        ctx.save();
        ctx.font = `${10 * dpr}px ${UI_FONT}`;
        for (const line of visibleLines) {
          const x = padL + ((line.energy_keV - e0) / span) * plotW;
          if (x < padL || x > padL + plotW) continue;
          const inBand = line.energy_keV >= bandEnergyLo && line.energy_keV <= bandEnergyHi;
          ctx.strokeStyle = inBand ? "rgba(129, 212, 250, 0.72)" : "rgba(129, 212, 250, 0.22)";
          ctx.lineWidth = inBand ? 1.5 * dpr : dpr;
          ctx.beginPath();
          ctx.moveTo(x, padT);
          ctx.lineTo(x, padT + plotH);
          ctx.stroke();
        }
        const labelLines = candidateLines.slice(0, 3);
        labelLines.forEach((line, index) => {
          const x = padL + ((line.energy_keV - e0) / span) * plotW;
          if (x < padL || x > padL + plotW) return;
          ctx.fillStyle = "rgba(129, 212, 250, 0.92)";
          ctx.fillText(lineLabel(line), Math.min(x + 3 * dpr, padL + plotW - 48 * dpr), padT + (13 + index * 12) * dpr);
        });
        ctx.restore();
      }
    }
    ctx.strokeStyle = "#ffd54f";
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = padL + (i / Math.max(1, values.length - 1)) * plotW;
      const y = padT + plotH - ((transformed[i] - lo) / (hi - lo)) * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = "#ddd";
    ctx.font = `${11 * dpr}px ${UI_FONT}`;
    ctx.fillText(`${formatEnergy(energy[bandLo])} - ${formatEnergy(energy[Math.max(bandLo, bandHi - 1)])}`, padL, canvas.height - 10 * dpr);
    ctx.fillText(logSpectrum ? "log counts" : "counts", 8 * dpr, 16 * dpr);
    recordWidgetPerf("spectrumDrawMs", performance.now() - t0);
  }, [bandEnergyHi, bandEnergyLo, bandHi, bandLo, candidateLines, energy, lineHints, logSpectrum, nEnergy, recordWidgetPerf, roiSpectrum, showLineHints, specH, specW]);

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
    saveWidgetChanges();
  }, [saveWidgetChanges, setRoiCol, setRoiHeight, setRoiRow, setRoiWidth]);

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
    const r = Math.max(0, Math.min(rows - 1, Math.round(next.row)));
    const c = Math.max(0, Math.min(cols - 1, Math.round(next.col)));
    const h = Math.max(1, Math.min(rows - r, Math.round(next.height)));
    const w = Math.max(1, Math.min(cols - c, Math.round(next.width)));
    const normalized = { row: r, col: c, height: h, width: w };
    setLocalRoi(normalized);
    if (sync) {
      queueRoiPersist(normalized, false, interactive);
    }
    scheduleSpectrum(normalized, interactive);
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

  const saveCurrentRoi = () => {
    const name = `ROI ${safeSavedRois.length + 1}`;
    const next = [
      ...safeSavedRois.filter((item) => item.name !== name),
      { name, row: roi.row, col: roi.col, height: roi.height, width: roi.width },
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
    const r = mapCanvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const onMapDown = (e: React.MouseEvent) => {
    const p = mapPoint(e);
    const sx = size / cols, sy = size / rows;
    const x = roi.col * sx, y = roi.row * sy, w = roi.width * sx, h = roi.height * sy;
    const nearCorner = Math.abs(p.x - (x + w)) < 14 && Math.abs(p.y - (y + h)) < 14;
    const inside = p.x >= x && p.x <= x + w && p.y >= y && p.y <= y + h;
    if (nearCorner || inside) {
      setDrag({ mode: nearCorner ? "roi-resize" : "roi-move", x: p.x, y: p.y, roi, bandStart: bandLo, bandEnd: bandHi });
      e.preventDefault();
    }
  };

  const onSpecDown = (e: React.MouseEvent) => {
    const rect = specCanvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const padL = 54, padR = 14;
    const plotW = specW - padL - padR;
    const x0 = padL + (bandLo / Math.max(1, nEnergy - 1)) * plotW;
    const x1 = padL + ((bandHi - 1) / Math.max(1, nEnergy - 1)) * plotW;
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
        const rect = mapCanvasRef.current?.getBoundingClientRect();
        if (!rect) return;
        const dx = (e.clientX - rect.left - drag.x) / size * cols;
        const dy = (e.clientY - rect.top - drag.y) / size * rows;
        if (drag.mode === "roi-move") updateRoi({ ...drag.roi, row: drag.roi.row + dy, col: drag.roi.col + dx }, true, true);
        else updateRoi({ ...drag.roi, width: drag.roi.width + dx, height: drag.roi.height + dy }, true, true);
      } else if (drag.mode?.startsWith("band")) {
        const rect = specCanvasRef.current?.getBoundingClientRect();
        if (!rect) return;
        const padL = 54, padR = 14;
        const plotW = specW - padL - padR;
        const di = ((e.clientX - rect.left - drag.x) / plotW) * Math.max(1, nEnergy - 1);
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
  }, [cols, drag, flushBandPersist, flushLocalBandPreview, flushRoiPersist, nEnergy, previewCenterBand, rows, size, specW]);

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
    updateBand(bandLo, bandHi);
    updateRoi({
      row: Math.max(0, rows / 2 - rows / 8),
      col: Math.max(0, cols / 2 - cols / 8),
      height: Math.max(8, rows / 4),
      width: Math.max(8, cols / 4),
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
    minHeight: 34,
    border: "1px solid rgba(128,128,128,0.35)",
    bgcolor: "rgba(128,128,128,0.08)",
    px: 1,
    py: 0,
    boxSizing: "border-box",
  } as const;
  const controlLabelSx = { fontSize: 10, color: "text.secondary", flexShrink: 0, lineHeight: "20px" } as const;
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
    "&.Mui-disabled": {
      color: "text.disabled",
      borderColor: "divider",
    },
  } as const;

  return (
    <Box sx={{ p: 2, fontFamily: UI_FONT, color: "inherit", overflowX: "auto" }}>
      <Stack spacing={1.2}>
        <Stack direction="row" spacing={1} alignItems="center" sx={{ width: specW + size + 16 }}>
          <Stack direction="row" spacing={1} alignItems="center" sx={{ flex: 1, minWidth: 0 }}>
            <Typography sx={{ fontWeight: 700, fontSize: 14 }}>{title || "EDS spectrum image"}</Typography>
            <Typography sx={{ fontSize: 11, color: "text.secondary" }}>
              {rows}x{cols}x{nEnergy} | {backendLabel} {elementLabel ? `| ${elementLabel}` : ""} {busy ? "| computing" : ""}
            </Typography>
          </Stack>
          {showControls && (
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography sx={{ fontSize: 11 }}>Log</Typography>
              <Switch checked={logSpectrum} onChange={(e) => setLogSpectrum(e.target.checked)} size="small" />
              <Typography sx={{ fontSize: 11 }}>Debug</Typography>
              <Switch checked={showDebug} onChange={(e) => commitDebug(e.target.checked)} size="small" />
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
                    color: (localExportStatus || exportStatus).startsWith("Export failed") ? "#d32f2f" : "text.secondary",
                  }}
                  title={localExportStatus || exportStatus}
                >
                  {localExportStatus || exportStatus}
                </Typography>
              )}
            </Stack>
          )}
        </Stack>
        {gpuError && <Typography sx={{ color: "#d32f2f", fontSize: 12 }}>{gpuError}</Typography>}
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
              bgcolor: "rgba(0,0,0,0.78)",
              color: "#d8f6ff",
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
          </Box>
        )}
        <Stack direction="row" spacing={2} alignItems="flex-start">
          <Box>
            <Box sx={{ position: "relative", width: size, height: size, bgcolor: "#000", border: "1px solid #444" }}>
              <canvas
                ref={mapCanvasRef}
                onMouseDown={onMapDown}
                style={{ width: size, height: size, display: "block", cursor: drag?.mode?.startsWith("roi") ? "grabbing" : "grab" }}
                aria-label={`EDS real-space map${title ? `: ${title}` : ""}`}
              />
              <canvas
                ref={mapOverlayCanvasRef}
                style={{
                  position: "absolute",
                  inset: 0,
                  width: size,
                  height: size,
                  pointerEvents: "none",
                  opacity: displayOverlayOpacity,
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
                  border: "2px dashed #00ff7f",
                }}
              >
                <Box
                  sx={{
                    position: "absolute",
                    right: -9,
                    bottom: -9,
                    width: 18,
                    height: 18,
                    bgcolor: "#00ff7f",
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
                  background: "linear-gradient(135deg, transparent 50%, #5af 50%)",
                  "&:hover": { opacity: 1 },
                }}
              />
            </Box>
            <Typography sx={{ mt: 0.5, fontSize: 11 }}>
              Overlay ROI: row {roi.row}, col {roi.col}, {roi.height}x{roi.width}
            </Typography>
          </Box>
          <Box>
            <Box sx={{ position: "relative", width: specW }}>
              <canvas
                ref={specCanvasRef}
                onMouseDown={onSpecDown}
                style={{ width: specW, height: specH, display: "block", cursor: drag?.mode?.startsWith("band") ? "grabbing" : "ew-resize", background: "#050505", border: "1px solid #444" }}
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
                  bgcolor: "rgba(255, 215, 0, 0.22)",
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
                  background: "linear-gradient(135deg, transparent 50%, #5af 50%)",
                  "&:hover": { opacity: 1 },
                }}
              />
            </Box>
            <Typography sx={{ mt: 0.5, fontSize: 11 }}>
              Band {bandLo}-{bandHi - 1}: {formatEnergy(energy[bandLo])} - {formatEnergy(energy[Math.max(bandLo, bandHi - 1)])}; ROI band counts {formatNumber(bandCounts, 2)}
              {candidateText ? `; candidates ${candidateText}` : ""}
            </Typography>
          </Box>
        </Stack>
        {showControls && (
          <Box sx={{ mt: 0.25, display: "flex", gap: 1, maxWidth: specW + size, boxSizing: "border-box", alignItems: "flex-start", overflowX: "auto", pb: 0.5 }}>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, flex: "0 0 auto", minWidth: 0 }}>
              <Box sx={controlRowSx}>
                <Typography sx={controlLabelSx}>Band:</Typography>
                <Box
                  ref={bandSliderRef}
                  onMouseDownCapture={(e) => {
                    if ((e.target as HTMLElement).closest(".MuiSlider-thumb")) return;
                    const rect = bandSliderRef.current?.getBoundingClientRect();
                    if (!rect) return;
                    const x = e.clientX - rect.left;
                    const x0 = (bandLo / Math.max(1, nEnergy)) * rect.width;
                    const x1 = (bandHi / Math.max(1, nEnergy)) * rect.width;
                    if (x < x0 || x > x1) return;
                    previewCenterBand(bandLo, bandHi, rect.width);
                    setBandSliderDrag({ x: e.clientX, width: rect.width, bandStart: bandLo, bandEnd: bandHi });
                    e.preventDefault();
                    e.stopPropagation();
                    e.nativeEvent.stopImmediatePropagation();
                  }}
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
                      bgcolor: "#1976d2",
                      borderRadius: 1,
                      opacity: bandSliderDrag ? 1 : 0,
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
                        bgcolor: "#1976d2",
                        boxShadow: "0 0 0 2px #fff",
                      },
                      "&::before": { left: -6 },
                      "&::after": { right: -6 },
                    }}
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
                      ...(bandSliderDrag ? {
                        "& .MuiSlider-track, & .MuiSlider-thumb": { opacity: 0 },
                      } : {}),
                      "& .MuiSlider-track": { height: 2, cursor: bandSliderDrag ? "grabbing" : "grab" },
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
                  sx={{ zIndex: 9999 }}
                >
                  {safeSavedRois.map((saved, index) => (
                    <MenuItem key={`${saved.name}-${index}`} onClick={() => applySavedRoi(saved)} sx={{ fontSize: 12 }}>
                      {saved.name}: row {saved.row}, col {saved.col}, {saved.height}x{saved.width}
                    </MenuItem>
                  ))}
                  <MenuItem onClick={clearSavedRois} sx={{ fontSize: 12, color: "text.secondary" }}>Clear saved ROIs</MenuItem>
                </Menu>
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
                  sx={{ zIndex: 9999 }}
                >
                  {safeSavedBands.map((saved, index) => (
                    <MenuItem key={`${saved.name}-${index}`} onClick={() => applySavedBand(saved)} sx={{ fontSize: 12 }}>
                      {saved.name}: {formatEnergy(energy[saved.start])} - {formatEnergy(energy[Math.max(saved.start, saved.end - 1)])}
                    </MenuItem>
                  ))}
                  <MenuItem onClick={clearSavedBands} sx={{ fontSize: 12, color: "text.secondary" }}>Clear saved bands</MenuItem>
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
                width={110}
                height={58}
                onRangeChange={(lo, hi) => {
                  if (autoMapContrastOn) setAutoMapContrastOn(false);
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
