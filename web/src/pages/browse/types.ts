// Shared types + data helpers for the Browse page. In the standalone WebGPU app the
// fetch* bodies are reimplemented against a locally-picked folder + the WGSL engine
// (see local/store.ts); every type + pure helper below is unchanged from quantem.live.

import * as store from "../../local/store";

export type CalStatus = "ok" | "warn" | "un";

/** Structured probe of "can this master.h5 actually open?" Mirrors the
 *  server-side `_master_load_status` shape. All fields optional on the
 *  client because older server builds may omit the block entirely. */
export interface MasterLoadStatus {
  loadable: boolean;
  /** Filenames of missing _data_*.h5 chunks, capped to ~10 with a final
   *  "...and N more" sentinel string when truncated. */
  missing_chunks: string[];
  missing_count: number;
  expected_count: number;
  present_bytes: number;
  /** Uncompressed extent × dtype.itemsize. None when the virtual-source
   *  extent could not be read. */
  expected_bytes_estimate: number | null;
  /** One short human sentence for the error banner headline. */
  reason: string;
}

export interface MasterFile {
  name: string;
  shape: [number, number, number, number];
  cal: CalStatus;
  size: string;
  size_bytes?: number;
  /** Server-side check: all external _data_*.h5 chunks resolve. False for
   *  half-copied masters (e.g. operator/sample_028 with 35/105 chunks).
   *  Used by defaultSelection to avoid showing a broken cold-load. */
  loadable?: boolean;
  /** Rich probe of the missing-chunks state for the error banner. */
  load_status?: MasterLoadStatus;
  /** Real-space pixel pitch (Å per scan pixel) resolved from `dataset.yaml`
   *  (per-file override, then session default). Null when no calibration
   *  is plumbed through, in which case the scale bar shows raw pixels. */
  scan_sampling_A?: number | null;
  /** Detector mrad-per-pixel (post-bin) resolved from `dataset.yaml` or
   *  derived from the standard 58 mrad Arina geometry. Null when unknown. */
  k_pixel_size_mrad?: number | null;
  /** Existing Screen sidecar has an SSB phase that Browse can display without recomputing. */
  has_ssb?: boolean;
}

/** Format a length in Å as the most readable unit. */
export function formatLength(angstrom: number): string {
  if (!isFinite(angstrom) || angstrom <= 0) return "—";
  if (angstrom >= 10000) return `${(angstrom / 10000).toFixed(0)} \u00B5m`;
  if (angstrom >= 10) return `${(angstrom / 10).toFixed(angstrom < 100 ? 1 : 0)} nm`;
  return `${angstrom.toFixed(1)} \u00C5`;
}

/** Format a convergence/scattering angle in mrad. */
export function formatMrad(mrad: number): string {
  if (!isFinite(mrad) || mrad <= 0) return "—";
  return mrad >= 10 ? `${mrad.toFixed(0)} mrad` : `${mrad.toFixed(1)} mrad`;
}

export interface Session {
  source: string;
  date: string;
  files: MasterFile[];
}

// No SSB for the local browser app: SSB is an iterative phase reconstruction, not a per-frame
// virtual image, so it only ever showed a precomputed Screen sidecar (has_ssb) which loose local
// .h5 files never have. Omitted here; it lives in the Screening surface, not local browse.
export const DETECTOR_MODES = ["BF", "ADF", "DF", "CoMmag", "CoMy", "CoMx", "iCoM"] as const;
export type DetectorMode = typeof DETECTOR_MODES[number];

export const COLORMAP_OPTIONS = ["viridis", "inferno", "magma", "plasma", "gray", "hot", "cividis"] as const;
export type ColormapName = typeof COLORMAP_OPTIONS[number];

/** Probe convergence semiangle in mrad. The dashboard currently doesn't
 *  expose per-master α from dataset.yaml on the wire, so we use the most
 *  common 4D-STEM session value as the display constant. The ring radii
 *  the user types in α-units are forwarded server-side, where they're
 *  multiplied by the auto-fit BF disk radius in pixels — so the correct
 *  ring is drawn regardless of what α we display in the toolbar. */
export const ALPHA_MRAD = 23.0;

/** Stable identity for a file inside its session. */
export function fileKey(s: Session, f: MasterFile): string {
  return `${s.source}/${s.date}/${f.name}`;
}

export function findSession(sessions: Session[], source: string, date: string): Session | undefined {
  return sessions.find((s) => s.source === source && s.date === date);
}

export function findFile(
  sessions: Session[], source: string, date: string, name: string,
): { session: Session; file: MasterFile } | undefined {
  const session = findSession(sessions, source, date);
  if (!session) return undefined;
  const file = session.files.find((f) => f.name === name);
  if (!file) return undefined;
  return { session, file };
}

export function defaultSelection(sessions: Session[]): { session: Session; file: MasterFile } | null {
  if (sessions.length === 0) return null;
  // Pick the first FULLY LOADABLE + calibrated master. `loadable: false`
  // = master.h5 has external _data_*.h5 chunks the server can't resolve
  // (half-copied collaborator dir). `cal: "ok"` = dataset.yaml present.
  // Both filters apply so the cold-load lands on a master that actually
  // renders BF/CBED end-to-end.
  for (const s of sessions) {
    const f = s.files.find((f) => f.loadable && f.cal === "ok");
    if (f) return { session: s, file: f };
  }
  // Loosen: loadable but uncalibrated.
  for (const s of sessions) {
    const f = s.files.find((f) => f.loadable);
    if (f) return { session: s, file: f };
  }
  // Fallback (everything is partial — pick first; user sees the empty state).
  for (const s of sessions) {
    if (s.files.length) return { session: s, file: s.files[0] };
  }
  return null;
}

// --- API ------------------------------------------------------------------

export async function fetchSessions(): Promise<Session[]> {
  return store.getSessions();
}

/** # of files the last folder scan skipped (corrupt / truncated / junk). */
export function lastScanSkipped(): number { return store.lastScanSkipped(); }

export interface RawData {
  data: Float32Array;
  width: number;
  height: number;
}

/** Fetch a single CBED frame at scan position (sx, sy) for one master.
 *  Width/height come from response headers; pass-through Float32Array
 *  feeds straight into the WebGPU colormap engine. */
export function fetchCBED(
  s: Session, f: MasterFile, sx: number, sy: number, _signal?: AbortSignal,
  detBin: DetBin = 1, dtype: BrowseDtype = "uint8",
): Promise<RawData | null> {
  return store.cbedFrame(s.source, s.date, f.name, sx, sy, detBin, dtype);
}

/** Fetch a SUMMED CBED frame across the rectangular scan ROI
 *  (row0:row1, col0:col1). Result is one float32 detector frame
 *  representing the integrated diffraction pattern over all scan positions
 *  inside the rectangle. Uses the GPU master cache, so this is fast as
 *  long as the master is hot (it always is, since /realspace warms it). */
export function fetchCBEDRoi(
  s: Session, f: MasterFile,
  row0: number, col0: number, row1: number, col1: number,
  _signal?: AbortSignal,
  detBin: DetBin = 1, dtype: BrowseDtype = "uint8",
): Promise<RawData | null> {
  return store.cbedRoi(s.source, s.date, f.name, row0, col0, row1, col1, detBin, dtype);
}

/** Fetch one virtual image (BF / ADF / DF / CoM mag / CoMx / CoMy / iCoM / SSB)
 *  computed across all scan positions with the given α-unit ring radii.
 *  Optional ``cx`` / ``cy`` (detector pixels) move the aperture center off
 *  the auto-fit BF disk — null/undefined means "use the server's auto-fit".
 *  ``detBin`` selects the binned-master GPU cache entry (1 = full resolution,
 *  2 = 2× detector binning); only honored when the master has been preloaded
 *  with the matching det_bin via ``preloadSet5D``. */
export function fetchVirtualImage(
  s: Session, f: MasterFile, mode: DetectorMode,
  inner: number, outer: number,
  cx?: number | null, cy?: number | null,
  _signal?: AbortSignal,
  detBin: DetBin = 1,
  dtype: BrowseDtype = "uint8",
): Promise<RawData | null> {
  return store.virtualImage(s.source, s.date, f.name, mode, inner, outer, cx ?? null, cy ?? null, detBin, dtype);
}

/** GPU-resident virtual image for the aperture-drag fast path: returns the maskedSum GPU buffer
 *  (no readback) for BF/ADF/DF, or null for CoM modes (caller falls back to fetchVirtualImage). */
export function fetchVirtualImageBufferGpu(
  s: Session, f: MasterFile, mode: DetectorMode, inner: number, outer: number,
  cx: number | null, cy: number | null, detBin: DetBin = 1, dtype: BrowseDtype = "uint8",
): Promise<{ buffer: GPUBuffer; width: number; height: number } | null> {
  return store.virtualImageBufferGpu(s.source, s.date, f.name, mode, inner, outer, cx ?? null, cy ?? null, detBin, dtype);
}

// --- Detector-shape selector (Show4DSTEM-style live shape on the DP) ----

/** The shape the user picked on the DP overlay. Drives the detector mask
 *  the server builds before reducing over scan. ``"circle"`` is the
 *  standard BF aperture; ``"annulus"`` covers ADF/DF; ``"square"`` /
 *  ``"rect"`` / ``"point"`` are the new free-form options. */
export const DET_SHAPES = ["circle", "square", "rect", "annulus", "point"] as const;
export type DetShape = typeof DET_SHAPES[number];

/** Parameters for the active shape, all in detector pixels.
 *
 *  Only the fields relevant to the active shape are read by the backend;
 *  the rest are ignored. We carry every field on every request anyway so
 *  the shape change feels instantaneous (no second round-trip when the
 *  user toggles between circle and rect). */
export interface ShapeParams {
  /** Center for circle / square / annulus / point. */
  cx: number;
  cy: number;
  /** Circle radius. */
  r: number;
  /** Square half-side (full edge = 2 × half). */
  half: number;
  /** Annulus inner / outer radii. */
  inner: number;
  outer: number;
  /** Rect corners, canonical (row0 ≤ row1, col0 ≤ col1). */
  row0: number;
  col0: number;
  row1: number;
  col1: number;
  /** Point-mode pixel coords. */
  px: number;
  py: number;
}

/** Fetch a virtual image with an arbitrary detector mask shape. Server
 *  builds the mask on GPU 0 + reduces over scan with integer accumulation.
 *  All shape params travel on every request so the cache key is stable. */
export function fetchVirtualImageShape(
  s: Session, f: MasterFile, shape: DetShape, p: ShapeParams,
  _signal?: AbortSignal,
  detBin: DetBin = 1, dtype: BrowseDtype = "uint8",
): Promise<RawData | null> {
  return store.virtualImageShape(s.source, s.date, f.name, shape, p, detBin, dtype);
}

/** Auto-fit BF disk geometry for a master (cy, cx, r_bf in detector px).
 *  Used by the Viewer to seed the aperture-center state on master change.
 *  Cached server-side per (path, mtime); cold ~1-3 s, warm <1 ms. */
export interface BfGeometry { cy: number; cx: number; r_bf: number }

export async function fetchBfGeometry(
  s: Session, f: MasterFile, _signal?: AbortSignal, detBin: DetBin = 1, dtype: BrowseDtype = "uint8",
): Promise<BfGeometry | null> {
  try {
    return await store.bfGeometry(s.source, s.date, f.name, detBin, dtype);
  } catch {
    return null;
  }
}

// --- Master metadata (hover popover on FileTree) -------------------------

/** One key/value entry in the master metadata payload. `value` is a
 *  human-formatted string with units; `raw` is the underlying scalar/array
 *  for downstream tooling that wants the unformatted number. */
export interface MasterMetadataField {
  key: string;
  value: string;
  raw: unknown;
}

export interface MasterMetadata {
  session: string;
  name: string;
  master_path: string;
  file_size: string;
  file_size_bytes: number;
  shape_summary: string;
  fields: MasterMetadataField[];
  errors: string[];
}

/** In-memory cache so hovering the same row twice doesn't refetch.
 *  Keyed by `<source>/<date>/<file>` — the same key as `fileKey()`. */
const METADATA_CACHE = new Map<string, Promise<MasterMetadata>>();

export function fetchMasterMetadata(
  s: Session, f: MasterFile, _signal?: AbortSignal,
): Promise<MasterMetadata> {
  const key = fileKey(s, f);
  const cached = METADATA_CACHE.get(key);
  if (cached) return cached;
  const [wx, wy, kx, ky] = f.shape;
  const meta: MasterMetadata = {
    session: `${s.source}/${s.date}`, name: f.name, master_path: f.name,
    file_size: f.size, file_size_bytes: f.size_bytes ?? 0,
    shape_summary: `${wx}×${wy}×${kx}×${ky}`,
    fields: [
      { key: "scan", value: `${wx} × ${wy}`, raw: [wx, wy] },
      { key: "detector", value: `${kx} × ${ky}`, raw: [kx, ky] },
      { key: "format", value: "HDF5 · bitshuffle+lz4", raw: null },
    ],
    errors: [],
  };
  const p = Promise.resolve(meta);
  METADATA_CACHE.set(key, p);
  return p;
}

// --- 5D-STEM scrubber ----------------------------------------------------

/** Detector binning levels the backend supports. 1 = full resolution,
 *  higher values divide each detector dimension, cutting per-master VRAM
 *  by ``bin²``. ``dtype`` chooses the estimated browser cache precision. */
export type DetBin = 1 | 2 | 4 | 8;
/** Browse-block precision. "uint8" (default) halves VRAM so ~2× more masters
 *  cache; lossless when raw counts ≤ 255 (the common Arina case). "uint16"
 *  keeps raw counts for exact CBED brightness. */
export type BrowseDtype = "uint8" | "uint16";

/** UI-side binning setting. ``"auto"`` means "pick the smallest bin that
 *  fits the selected set in available VRAM"; resolved to a concrete
 *  ``DetBin`` at preload time via ``pickAutoBin``. */
export type DetBinSetting = DetBin | "auto";

/** Estimated on-GPU bytes for one master at a given binning. */
export function masterBytesAtBin(f: MasterFile, bin: DetBin, dtype: BrowseDtype = "uint16"): number {
  const [wx, wy, kx, ky] = f.shape;
  if (!wx || !wy || !kx || !ky) return 0;
  const det_w = Math.floor(kx / bin);
  const det_h = Math.floor(ky / bin);
  return wx * wy * det_w * det_h * (dtype === "uint8" ? 1 : 2);
}

/** Pick the best (smallest) detector bin so the selected set fits in
 *  ``freeBytes * safety`` of GPU 0. Walks 1 → 2 → 4 → 8 and returns the
 *  first that fits. Falls through to 8 when nothing fits — caller can
 *  warn the user that the set is just too big. Empty file list → 1. */
export function pickAutoBin(files: MasterFile[], freeBytes: number, safety = 0.45, dtype: BrowseDtype = "uint8"): DetBin {
  if (!files.length) return 1;
  const budget = freeBytes * safety;
  const bins: DetBin[] = [1, 2, 4, 8];
  for (const bin of bins) {
    const total = files.reduce((acc, f) => acc + masterBytesAtBin(f, bin, dtype), 0);
    if (total <= budget) return bin;
  }
  return 8;
}

export interface WarmPlan {
  files: MasterFile[];
  activeIdx: number;
  budgetBytes: number;
  estimatedBytes: number;
  totalFiles: number;
  mode: "all" | "window";
}

function cacheKeyAtBin(s: Session, f: MasterFile, bin: DetBin, dtype: BrowseDtype): string {
  return `${fileKey(s, f)}|b${bin}|${dtype}`;
}

function htmlResidentBytes(f: MasterFile, bin: DetBin, dtype: BrowseDtype): number {
  return masterBytesAtBin(f, bin, dtype);
}

export function planWarmSet5D(
  files: MasterFile[], activeIdx: number, freeBytes: number, bin: DetBin = 1, dtype: BrowseDtype = "uint8",
): WarmPlan {
  const totalFiles = files.length;
  const clamped = Math.max(0, Math.min(totalFiles - 1, activeIdx));
  const reportedBudget = Math.floor(Math.max(0, freeBytes) * 0.4);
  const hardBudget = 3 * 1024 * 1024 * 1024;
  const budgetBytes = Math.max(256 * 1024 * 1024, Math.min(reportedBudget || hardBudget, hardBudget));
  const allBytes = files.reduce((acc, f) => acc + htmlResidentBytes(f, bin, dtype), 0);
  if (allBytes <= budgetBytes) {
    return { files, activeIdx: clamped, budgetBytes, estimatedBytes: allBytes, totalFiles, mode: "all" };
  }

  const chosen = new Set<number>();
  let estimatedBytes = 0;
  const tryAdd = (idx: number): boolean => {
    if (idx < 0 || idx >= totalFiles || chosen.has(idx)) return true;
    const bytes = htmlResidentBytes(files[idx], bin, dtype);
    if (estimatedBytes + bytes > budgetBytes) return false;
    chosen.add(idx);
    estimatedBytes += bytes;
    return true;
  };
  tryAdd(clamped);
  for (let radius = 1; radius < totalFiles; radius++) {
    const leftOk = tryAdd(clamped - radius);
    const rightOk = tryAdd(clamped + radius);
    if (!leftOk && !rightOk) break;
  }
  const idxs = Array.from(chosen).sort((a, b) => a - b);
  return {
    files: idxs.map((idx) => files[idx]),
    activeIdx: clamped,
    budgetBytes,
    estimatedBytes,
    totalFiles,
    mode: "window",
  };
}

/** Snapshot of which masters are currently in the server's GPU LRU
 *  cache. Used by the file-tree UI to show a ⚡ icon next to each cached
 *  row and a "N/M slots" summary in the Datasets header. */
export interface CacheStatus {
  slots_used: number;
  slots_total: number;
  bytes_used?: number;
  size_used?: string;
  bytes_total?: number;
  size_total?: string;
  cached: {
    session: string;
    file: string;
    det_bin: number;
    cache_key_det_bin?: number;
    dtype?: string | null;
    shape?: number[];
    bytes?: number;
    size?: string;
  }[];
  active: { session: string; file: string; det_bin: number } | null;
}

export async function fetchCacheStatus(): Promise<CacheStatus | null> {
  return null;  // no server-side GPU cache in the standalone app
}

/** Drop ALL cached masters from GPU 0 — used by the "clear" UI button.
 *  We keep Browse caches warm across page changes so returning to Browse is
 *  not forced through a cold GPU/file-handle load. Always succeeds (server
 *  returns 200 even on no-op). */
export async function clearMasterCache(): Promise<{ cleared: number } | null> {
  return { cleared: 0 };  // handled by the engine's own LRU + dispose
}

/** Fetch current free GPU 0 bytes from the global ``/api/gpu`` endpoint.
 *  Used by ``pickAutoBin`` + the per-folder VRAM chip. The endpoint returns
 *  per-GPU dicts with ``mem_used_mb`` / ``mem_total_mb`` — we read GPU 0
 *  (the dashboard's pinned device per CLAUDE.md). */
export async function fetchGpuFreeBytes(): Promise<number> {
  return store.freeVramBytes();
}

/** A user-curated 5D-STEM set: ordered list of masters from ONE session
 *  that share the same probe parameters (aperture, rings, scan, mode,
 *  cmap, clip) across the scrub. ``activeIdx`` is the file currently
 *  shown in the viewer; ``detBin`` is the per-master binning toggle —
 *  ``2`` cuts VRAM 4× per master at the cost of detector pixel density,
 *  letting many more masters fit on one GPU for long focal/tilt series. */
export interface Set5D {
  /** Source session (so the parent can clear the set on session change). */
  session: Session;
  /** Ordered list, in selection order. UI shows numbered chips matching
   *  ``files.indexOf(file) + 1`` so the user sees their own ordering. */
  files: MasterFile[];
  /** Index into ``files`` of the master currently shown. */
  activeIdx: number;
  /** Detector binning applied at GPU load time. Determines which cache
   *  entry the realspace endpoint queries via the ``det_bin`` param. */
  detBin: DetBin;
  /** Number of masters intentionally warmed/pinned under the browser VRAM budget. */
  warmCount?: number;
  warmTotal?: number;
  warmMode?: "all" | "window";
}

/** Stable identity for a Set5D (used as a React effect dependency). */
export function set5DKey(set: Set5D): string {
  return `${set.session.source}/${set.session.date}|b${set.detBin}|${set.files.map((f) => f.name).join(",")}`;
}

interface PreloadSetResponse {
  queued: number;
  det_bin: number;
  session?: string;
}

/** Warm a budgeted subset of a 5D set in standalone browser mode. The active
 *  file plus as many neighbors as fit are pinned; the rest stay cold and load
 *  on demand when the scrubber reaches them. */
export async function preloadSet5D(
  s: Session, files: MasterFile[], detBin: DetBin = 1, _dtype: BrowseDtype = "uint8",
  activeIdx = 0, freeBytes = fetchGpuFreeBytes(),
): Promise<PreloadSetResponse> {
  const free = typeof freeBytes === "number" ? freeBytes : await freeBytes;
  const plan = planWarmSet5D(files, activeIdx, free, detBin, _dtype);
  store.setPinned5DKeys(plan.files.map((f) => cacheKeyAtBin(s, f, detBin, _dtype)));
  void store.warmSet5D(plan.files.map((f) => ({ source: s.source, date: s.date, name: f.name, detBin, dtype: _dtype })));
  return { queued: plan.files.length, det_bin: detBin };
}

/** Fire-and-forget warm-up. Tells the backend to open the master file
 *  handle + prime the GPU decompressor cache so a subsequent `/cbed` or
 *  `/realspace` call hits warm state. Errors (404 if endpoint missing,
 *  500 if backend chokes) are swallowed silently — hover prefetch must
 *  never surface a console error to the user. */
export async function prefetchMaster(
  _s: Session, _f: MasterFile, _opts: { gpu?: boolean } = {},
): Promise<void> {
  // no-op: the engine opens + decodes on first view
}
