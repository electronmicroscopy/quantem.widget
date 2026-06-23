import { useEffect, useMemo, useRef, useState } from "react";
import Box from "@mui/material/Box";
import Fade from "@mui/material/Fade";
import Popper from "@mui/material/Popper";
import Typography from "@mui/material/Typography";
import { colors, fontSizes, radii } from "../../theme";
import {
  clearMasterCache, fetchMasterMetadata, fileKey, masterBytesAtBin,
  pickAutoBin, prefetchMaster,
  type CacheStatus, type DetBin, type DetBinSetting, type MasterFile, type MasterMetadata, type Session,
} from "./types";

const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";

// Module-level dedup: every (session, file) gets prefetched at most once
// per page lifetime. Hover-spamming the same row is a no-op after the first
// hit. Holding the promise (vs deleting on resolve) means re-hovering after
// the warm-up landed still skips the network call — the backend's open file
// handle stays warm for the rest of the session.
const PREFETCH_INFLIGHT = new Map<string, Promise<void>>();
const GPU_PREFETCH_FIRED = new Set<string>();
function fireOnce(s: Session, f: MasterFile) {
  const k = fileKey(s, f);
  if (PREFETCH_INFLIGHT.has(k)) return;
  PREFETCH_INFLIGHT.set(k, prefetchMaster(s, f));
}
/** Sustained-hover GPU warmup: tells the backend to background-load the
 *  master onto GPU 0 so the first realspace click hits a warm cache. Single-
 *  slot GPU LRU server-side means hovering many masters doesn't spam VRAM —
 *  only the most recently sustained-hover wins. */
function fireGpu(s: Session, f: MasterFile) {
  const k = fileKey(s, f);
  if (GPU_PREFETCH_FIRED.has(k)) return;
  GPU_PREFETCH_FIRED.add(k);
  void prefetchMaster(s, f, { gpu: true });
}

interface Props {
  sessions: Session[];
  activeFile: string;
  onSelect: (s: Session, f: MasterFile) => void;
  query: string;
  setQuery: (q: string) => void;
  loading?: boolean;
  error?: string | null;
  /** ORDERED list of selected (session, file) keys for the 5D-STEM set.
   *  Order is the chip number the user sees; entries from sessions other
   *  than the active multi-select session are pruned by ``onToggleSelect``.
   *  Empty array means "no 5D set in progress". */
  selected5DKeys?: string[];
  /** Cmd/Ctrl-click toggles inclusion. Shift-click extends a contiguous
   *  range from the last clicked row to the current one (within the same
   *  session — multi-select scope is one session per the spec). */
  onToggleSelect?: (s: Session, f: MasterFile, mode: "toggle" | "range") => void;
  /** User pressed "load 5D" — fires the backend preload + lifts state to
   *  the parent so Browse swaps to scrubber mode. */
  onLoad5D?: () => void;
  /** Det-bin setting for the binning select. ``"auto"`` picks the
   *  smallest bin that fits in available VRAM; concrete values are 1
   *  (full), 2 (4× cut), 4 (16× cut), 8 (64× cut). */
  detBin?: DetBinSetting;
  setDetBin?: (b: DetBinSetting) => void;
  /** Live free GPU bytes from /api/browse/gpu-stats. Used by the per-folder
   *  indicator chip to show the auto-bin level + total VRAM the folder
   *  would occupy if loaded as a 5D set. ``0`` = math hidden. */
  gpuFreeBytes?: number;
  /** "Load entire folder as 5D" handler. Called with all loadable files
   *  in the folder + the resolved auto-bin level. */
  onLoadFolder5D?: (s: Session, files: MasterFile[]) => void;
}

/** Three-way status dot for the file tree:
 *
 *   green  — calibrated AND every chunk file resolves (master will open)
 *   amber  — calibration partial (warn) AND chunks resolve (master opens, prep needed)
 *   red    — chunks missing (master CANNOT open) regardless of cal status
 *   gray   — uncalibrated and chunks resolve (master opens, no metadata)
 *
 * The red state is the most important: it tells the user the master is
 * physically broken and a click will hit a black canvas. Previously every
 * `cal === "ok"` row showed green even when chunks were missing — user
 * clicked the green dot and got the error banner. */
const dotColor = (f: MasterFile): string => {
  const loadable = f.loadable !== false;  // undefined treated as "unknown OK"
  if (!loadable) return colors.warning.dot;       // red — broken master
  if (f.cal === "warn") return colors.warning.medium;  // amber — partial yaml
  if (f.cal === "un") return colors.text.muted;        // gray — no yaml
  return colors.success.dot;                            // green — fully ready
};

const dotTitle = (f: MasterFile, active = false, cached = false): string => {
  const loadable = f.loadable !== false;
  const reason = f.load_status?.reason;
  const prefix = [
    active ? "Currently selected" : null,
    cached ? "GPU cached" : null,
  ].filter(Boolean).join(" · ");
  const withPrefix = (label: string) => prefix ? `${prefix} · ${label}` : label;
  if (!loadable) {
    return withPrefix(reason
      ? `Missing chunks — ${reason}`
      : "Missing chunks — master will not open");
  }
  if (f.cal === "warn") return withPrefix("Calibration partial (dataset.yaml incomplete)");
  if (f.cal === "un") return withPrefix("Uncalibrated — no dataset.yaml");
  return withPrefix("Calibrated and loadable");
};

const fmtBytes = (n: number): string => {
  const GB = 1 << 30, MB = 1 << 20;
  if (n >= GB) return `${(n / GB).toFixed(1)} GB`;
  if (n >= MB) return `${(n / MB).toFixed(0)} MB`;
  return `${(n / 1024).toFixed(0)} KB`;
};

/** Per-folder VRAM dot — tiny colored 14px badge showing the auto-bin
 *  level (1× / 2× / 4× / 8×). Hover tooltip spells out the math; click
 *  loads the whole folder as a 5D scrubber. Hidden when only one file
 *  in the folder (5D needs ≥2). */
function FolderVramChip({
  files, freeBytes, onLoad,
}: {
  files: MasterFile[];
  freeBytes: number;
  onLoad?: () => void;
}) {
  if (files.length < 2 || !freeBytes) return null;
  const bin: DetBin = pickAutoBin(files, freeBytes);
  const total = files.reduce((s, f) => s + masterBytesAtBin(f, bin, "uint8"), 0);
  const budget = freeBytes * 0.45;
  let color: string;
  if (bin === 1) color = colors.success.dot;
  else if (total <= budget) color = colors.warning.medium;
  else color = colors.warning.dot;
  const binLabel = bin === 1 ? "1×" : `${bin}×`;
  // uint8 browse packs ~2× more masters than uint16, so the load fills VRAM with
  // many of this folder's masters; cached masters are instant to browse.
  const tip = (
    `Load this folder: ${files.length} masters into VRAM (uint8, as many as fit), ` +
    `so each is instant to browse + scrub. Auto-bin ${binLabel} · ${fmtBytes(total)} ` +
    `(${fmtBytes(freeBytes)} free).`
  );
  return (
    <Box
      component="button"
      onClick={(e: React.MouseEvent) => { e.stopPropagation(); onLoad?.(); }}
      title={tip}
      sx={{ display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 0.4,
            ml: 0.5, minWidth: { xs: 56, sm: 52 }, height: { xs: 30, sm: 18 }, px: 0.6, py: 0,
            fontSize: { xs: 11, sm: 10 }, fontFamily: MONO, fontWeight: 700,
            border: "none", color: colors.text.white, bgcolor: color,
            borderRadius: radii.sm, whiteSpace: "nowrap",
            cursor: onLoad ? "pointer" : "default", opacity: 0.9,
            "&:hover": onLoad ? { opacity: 1 } : {} }}
    >
      {`⤓ load ${files.length}`}
    </Box>
  );
}

function LegendDot({ color, label, title }: { color: string; label: string; title: string }) {
  return (
    <Box component="span" title={title}
         sx={{ display: "inline-flex", alignItems: "center", gap: 0.35, whiteSpace: "nowrap" }}>
      <Box component="span"
           sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: color, flex: "0 0 auto" }} />
      <span>{label}</span>
    </Box>
  );
}

export default function FileTree({
  sessions, activeFile, onSelect, query, setQuery, loading, error,
  selected5DKeys, onToggleSelect, onLoad5D, detBin = 1, setDetBin,
  gpuFreeBytes = 0, onLoadFolder5D,
}: Props) {
  // Lookup map for fast chip-number rendering (selection order index + 1).
  // Module-shape `Map<key, position>` so an O(1) chip lookup happens once
  // per row instead of an array scan inside the render loop.
  const selectionMap = useMemo(() => {
    const map = new Map<string, number>();
    (selected5DKeys || []).forEach((k, i) => map.set(k, i + 1));
    return map;
  }, [selected5DKeys]);
  const selectedCount = selectionMap.size;
  // Default-open: first session that contains the active file, OR the first
  // session period. Recomputed when sessions arrive so the tree never starts
  // collapsed when only one session matters.
  const [open, setOpen] = useState<Record<string, boolean>>({});
  useEffect(() => {
    if (sessions.length === 0) return;
    setOpen((prev) => {
      // Don't override user toggles once they exist.
      if (Object.keys(prev).length > 0) return prev;
      const next: Record<string, boolean> = {};
      const owning = sessions.find((s) => s.files.some((f) => f.name === activeFile));
      const target = owning || sessions[0];
      next[`${target.source}/${target.date}`] = true;
      return next;
    });
  }, [sessions, activeFile]);

  // Two-stage hover prefetch — see issue #475:
  //   100 ms → file-handle warmup (cheap, ~25 ms backend cost)
  //   1500 ms → GPU master upload (expensive, ~1-30 s backend cost, evicts
  //             single-slot server cache). Gated on sustained hover so
  //             dragging the cursor across 50 file rows fires AT MOST 1
  //             GPU upload (the row the cursor came to rest on), not 50.
  // Both timers are cleared on mouseleave so a fly-by scroll never fires.
  const HOVER_FILE_HANDLE_MS = 100;
  const HOVER_GPU_MS = 1500;
  const hoverTimerRef = useRef<number | null>(null);
  const gpuTimerRef = useRef<number | null>(null);
  const cancelHover = () => {
    if (hoverTimerRef.current !== null) {
      window.clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
    if (gpuTimerRef.current !== null) {
      window.clearTimeout(gpuTimerRef.current);
      gpuTimerRef.current = null;
    }
  };
  const scheduleHover = (s: Session, f: MasterFile) => {
    cancelHover();
    hoverTimerRef.current = window.setTimeout(() => {
      hoverTimerRef.current = null;
      fireOnce(s, f);
    }, HOVER_FILE_HANDLE_MS);
    gpuTimerRef.current = window.setTimeout(() => {
      gpuTimerRef.current = null;
      fireGpu(s, f);
    }, HOVER_GPU_MS);
  };
  useEffect(() => () => cancelHover(), []);

  // Subscribe to /api/browse/cache-stream. The server pushes a fresh
  // snapshot on connect and one event per cache mutation (load / evict /
  // clear / active-master pin), so the file tree updates within a few ms
  // of the server-side state change — no polling. 15 s keepalive comments
  // keep idle connections open through reverse proxies.
  const [cacheStatus, setCacheStatus] = useState<CacheStatus | null>(null);
  useEffect(() => {
    const es = new EventSource(`/api/browse/cache-stream`);
    es.onmessage = (ev) => {
      try {
        const snap = JSON.parse(ev.data) as CacheStatus;
        setCacheStatus(snap);
      } catch {
        // ignore malformed payload — next event will overwrite
      }
    };
    es.onerror = () => {
      // EventSource auto-reconnects on transient drops (uses the server's
      // `retry: 2000` hint). Nothing to do here — we just don't want the
      // browser console to flag the close.
    };
    return () => { es.close(); };
  }, []);
  const cachedSet = useMemo(() => {
    const set = new Set<string>();
    cacheStatus?.cached.forEach((c) => set.add(`${c.session}/${c.file}`));
    return set;
  }, [cacheStatus]);

  // Hover popover: 250 ms grace before showing so a mouse fly-by doesn't
  // pop up the whole metadata block. The cache lives in types.ts so
  // re-hovers for the same row are network-free. Shape:
  //   { anchor, key, file, metadata }
  // `metadata === null` means "loading"; an empty `errors` array on
  // success means the master parsed cleanly.
  const [popover, setPopover] = useState<{
    anchor: HTMLElement;
    fileId: string;
    session: Session;
    file: MasterFile;
    metadata: MasterMetadata | null;
    error: string | null;
  } | null>(null);
  const popoverShowTimerRef = useRef<number | null>(null);
  const popoverHideTimerRef = useRef<number | null>(null);

  const cancelPopoverShow = () => {
    if (popoverShowTimerRef.current !== null) {
      window.clearTimeout(popoverShowTimerRef.current);
      popoverShowTimerRef.current = null;
    }
  };
  const cancelPopoverHide = () => {
    if (popoverHideTimerRef.current !== null) {
      window.clearTimeout(popoverHideTimerRef.current);
      popoverHideTimerRef.current = null;
    }
  };
  const schedulePopover = (
    anchor: HTMLElement, s: Session, f: MasterFile,
  ) => {
    cancelPopoverHide();
    cancelPopoverShow();
    const fileId = fileKey(s, f);
    popoverShowTimerRef.current = window.setTimeout(() => {
      popoverShowTimerRef.current = null;
      // Only fetch if the master is loadable. For broken masters we still
      // open the popover to show load_status.reason.
      if (f.loadable !== false) {
        setPopover({ anchor, fileId, session: s, file: f, metadata: null, error: null });
        fetchMasterMetadata(s, f).then(
          (md) => setPopover((cur) => (cur && cur.fileId === fileId ? { ...cur, metadata: md } : cur)),
          (err) => setPopover((cur) => (
            cur && cur.fileId === fileId
              ? { ...cur, error: err instanceof Error ? err.message : String(err) }
              : cur
          )),
        );
      } else {
        setPopover({ anchor, fileId, session: s, file: f, metadata: null, error: null });
      }
    }, 250);
  };
  const dismissPopover = () => {
    cancelPopoverShow();
    cancelPopoverHide();
    popoverHideTimerRef.current = window.setTimeout(() => {
      popoverHideTimerRef.current = null;
      setPopover(null);
    }, 120);
  };
  useEffect(() => () => { cancelPopoverShow(); cancelPopoverHide(); }, []);

  const total = sessions.reduce((a, s) => a + s.files.length, 0);
  return (
    <Box sx={{ display: "flex", flexDirection: "column", border: `1px solid ${colors.border.default}`,
              borderRadius: `${radii.lg}px`, bgcolor: colors.bg.page, overflow: "hidden", minHeight: 0 }}>
      <Box sx={{ px: 1.5, py: 1, borderBottom: `1px solid ${colors.border.default}` }}>
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
          <Typography sx={{ fontSize: fontSizes.lg, fontWeight: 700 }}>Datasets</Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.75,
                     fontSize: fontSizes.xs, color: colors.text.muted }}>
            <span>{total} files · {sessions.length} sessions</span>
            {cacheStatus && (
              <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.5 }}>
                <Box component="span"
                     title={[
                       `${cacheStatus.slots_used} of ${cacheStatus.slots_total} GPU cache slots used`,
                       cacheStatus.size_used && cacheStatus.size_total
                         ? `${cacheStatus.size_used} of ${cacheStatus.size_total} Browse VRAM budget`
                         : null,
                       "masters listed below with ⚡ are warm in VRAM",
                     ].filter(Boolean).join(" — ")}
                     sx={{
                       fontFamily: MONO,
                       color: cacheStatus.slots_used > 0 ? colors.success.dot : colors.text.muted,
                       fontWeight: 600,
                       whiteSpace: "nowrap",
                     }}>
                  ⚡ {cacheStatus.size_used && cacheStatus.size_total
                    ? `${cacheStatus.size_used}/${cacheStatus.size_total}`
                    : `${cacheStatus.slots_used}/${cacheStatus.slots_total}`}
                </Box>
                {cacheStatus.slots_used > 0 && (
                  <Box component="button"
                       onClick={async (e: React.MouseEvent) => {
                         e.stopPropagation();
                         await clearMasterCache();
                       }}
                       title="Clear all cached masters from GPU 0 (free VRAM)"
                       sx={{ fontSize: 10, fontFamily: MONO, lineHeight: 1,
                             border: `1px solid ${colors.interactive.border}`,
                             bgcolor: colors.interactive.bg,
                             color: colors.interactive.selectedText,
                             borderRadius: radii.sm, px: 0.65, py: 0.25,
                             cursor: "pointer",
                             "&:hover": { bgcolor: colors.bg.hover } }}>
                    clear
                  </Box>
                )}
              </Box>
            )}
          </Box>
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, flexWrap: "wrap",
                   mt: 0.45, fontSize: fontSizes.xs, color: colors.text.muted, lineHeight: 1.2 }}>
          <LegendDot color={colors.success.dot} label="ready" title="Calibrated and loadable" />
          <LegendDot color={colors.warning.medium} label="draft" title="Calibration partial" />
          <LegendDot color={colors.warning.dot} label="missing" title="Missing chunks; master will not open" />
          <LegendDot color={colors.text.muted} label="uncal" title="Uncalibrated; no dataset.yaml" />
          <Box component="span" title="Warm in GPU VRAM cache; click should be instant"
               sx={{ fontFamily: MONO, color: colors.success.dot, fontWeight: 600, whiteSpace: "nowrap" }}>
            ⚡ warm
          </Box>
        </Box>
      </Box>
      <Box sx={{ p: 1, borderBottom: `1px solid ${colors.border.default}` }}>
        <input
          placeholder="Search masters, sessions…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{
            width: "100%", boxSizing: "border-box",
            padding: "6px 8px", fontSize: fontSizes.md,
            border: `1px solid ${colors.border.default}`, borderRadius: radii.md,
            outline: "none", background: colors.bg.page,
          }}
        />
        {selectedCount >= 1 && onLoad5D && (
          // 5D toolbar — appears as soon as ANY row is in the set so the
          // user can adjust binning + see the count grow before clicking
          // load. Hint text spells out the keyboard convention.
          <Box sx={{ mt: 1, display: "flex", flexDirection: "column", gap: 0.5 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
              <Box
                component="button"
                onClick={selectedCount >= 2 ? onLoad5D : undefined}
                disabled={selectedCount < 2}
                title={selectedCount < 2
                  ? "Select 2+ masters with Cmd/Ctrl-click to load a 5D set"
                  : `Load all ${selectedCount} masters onto GPU 0 for scrubbing`}
                sx={{ flex: 1, fontSize: fontSizes.sm, fontWeight: 600,
                      px: 0.75, py: 0.5,
                      cursor: selectedCount >= 2 ? "pointer" : "default",
                      border: `1px solid ${selectedCount >= 2 ? colors.interactive.border : colors.border.default}`,
                      bgcolor: selectedCount >= 2 ? colors.interactive.bg : colors.bg.subtle,
                      color: selectedCount >= 2 ? colors.interactive.selectedText : colors.text.muted,
                      borderRadius: radii.sm,
                      "&:hover": selectedCount >= 2
                        ? { bgcolor: colors.interactive.bg }
                        : undefined }}
              >
                load 5D ({selectedCount})
              </Box>
              <Box component="label"
                   title="Detector binning at GPU load. 2× cuts VRAM 4× per master so many more fit on GPU 0 — at the cost of detector pixel density."
                   sx={{ display: "flex", alignItems: "center", gap: 0.25,
                         fontSize: fontSizes.xs, color: colors.text.muted,
                         fontFamily: MONO }}>
                <span>bin:</span>
                <select
                  value={String(detBin)}
                  onChange={(e) => {
                    const v = e.target.value;
                    if (v === "auto") setDetBin?.("auto");
                    else {
                      const n = parseInt(v, 10) as 1 | 2 | 4 | 8;
                      if ([1, 2, 4, 8].includes(n)) setDetBin?.(n);
                    }
                  }}
                  style={{ fontSize: fontSizes.xs, padding: "2px 4px",
                           borderRadius: radii.sm,
                           border: `1px solid ${colors.border.default}`,
                           background: colors.bg.page,
                           color: colors.text.secondary }}
                >
                  <option value="auto">auto (fit VRAM)</option>
                  <option value="1">full</option>
                  <option value="2">2× (¼ VRAM)</option>
                  <option value="4">4× (1/16 VRAM)</option>
                  <option value="8">8× (1/64 VRAM)</option>
                </select>
              </Box>
            </Box>
            <Typography sx={{ fontSize: fontSizes.xs, color: colors.text.muted, lineHeight: 1.3 }}>
              Cmd-click to add · Shift-click to extend · click row to clear
            </Typography>
          </Box>
        )}
      </Box>
      <Box sx={{ flex: 1, overflowY: "auto", pt: 0.5, pb: 18 }}>
        {loading && (
          <Typography sx={{ p: 1.5, fontSize: fontSizes.sm, color: colors.text.muted }}>
            loading sessions…
          </Typography>
        )}
        {error && (
          <Typography sx={{ p: 1.5, fontSize: fontSizes.sm, color: colors.warning.text }}>
            {error}
          </Typography>
        )}
        {!loading && !error && sessions.length === 0 && (
          <Typography sx={{ p: 1.5, fontSize: fontSizes.sm, color: colors.text.muted }}>
            Choose a folder of Arina .h5 datasets (button, top right) to begin. Everything decodes on
            your GPU - nothing leaves this machine.
          </Typography>
        )}
        <MetadataPopover
          state={popover}
          onPointerEnter={cancelPopoverHide}
          onPointerLeave={dismissPopover}
        />
        {sessions.map((s) => {
          const key = `${s.source}/${s.date}`;
          const isOpen = open[key];
          const filtered = query
            ? s.files.filter((f) => (f.name + " " + key).toLowerCase().includes(query.toLowerCase()))
            : s.files;
          if (query && filtered.length === 0) return null;
          return (
            <Box key={key} sx={{ mb: 0.5 }}>
              <Typography sx={{ px: 1.5, mt: 0.5, fontSize: fontSizes.xs, color: colors.text.muted,
                                textTransform: "uppercase", letterSpacing: 0.5 }}>
                {s.source}
              </Typography>
              <Box
                onClick={() => setOpen((o) => ({ ...o, [key]: !o[key] }))}
                sx={{ display: "flex", alignItems: "center", gap: 0.75, px: 1.5, py: 0.5,
                      cursor: "pointer", fontSize: fontSizes.md, fontWeight: 600,
                      color: colors.text.secondary,
                      "&:hover": { bgcolor: colors.bg.hover },
                      "&:hover .browse-vram-chip": { opacity: 1, pointerEvents: "auto" } }}
              >
                <Box component="span" sx={{ display: "inline-block", width: 10, transform: isOpen ? "rotate(90deg)" : "rotate(0deg)",
                                            transition: "transform 120ms", color: colors.text.tertiary }}>▶</Box>
                <span>{s.date}</span>
                <span style={{ marginLeft: "auto", fontSize: fontSizes.xs, color: colors.text.muted }}>
                  {filtered.length}
                </span>
                <Box className="browse-vram-chip"
                     sx={{ opacity: 0, pointerEvents: "none", transition: "opacity 120ms" }}>
                  <FolderVramChip
                    files={s.files.filter((f) => f.loadable !== false)}
                    freeBytes={gpuFreeBytes}
                    onLoad={onLoadFolder5D ? () => onLoadFolder5D(s, s.files.filter((f) => f.loadable !== false)) : undefined}
                  />
                </Box>
              </Box>
              {isOpen && (
                <Box sx={{ pl: 2.5 }}>
                  {filtered.map((f) => {
                    const isActive = activeFile === f.name;
                    const fkey = fileKey(s, f);
                    const isCached = cachedSet.has(fkey);
                    const chipNum = selectionMap.get(fkey);
                    const isSelected = chipNum !== undefined;
                    return (
                      <Box
                        key={f.name}
                        onClick={(e) => {
                          // Cmd/Ctrl-click toggles inclusion in the 5D set.
                          // Shift-click extends from the last clicked row.
                          // Plain click selects (the existing single-file
                          // behavior preserved). Broken masters (loadable=false)
                          // can't be added to a 5D set — they crash the loader.
                          if (onToggleSelect && (e.metaKey || e.ctrlKey) && f.loadable !== false) {
                            e.preventDefault();
                            onToggleSelect(s, f, "toggle");
                            return;
                          }
                          if (onToggleSelect && e.shiftKey && f.loadable !== false) {
                            e.preventDefault();
                            onToggleSelect(s, f, "range");
                            return;
                          }
                          onSelect(s, f);
                        }}
                        onMouseEnter={(e) => {
                          scheduleHover(s, f);
                          schedulePopover(e.currentTarget as HTMLElement, s, f);
                        }}
                        onMouseLeave={() => { cancelHover(); dismissPopover(); }}
                        sx={{ display: "flex", alignItems: "center", gap: 0.75,
                              px: 1, py: 0.5, cursor: "pointer", fontSize: fontSizes.sm,
                              borderRadius: radii.md,
                              bgcolor: isActive ? colors.status.activeBg : "transparent",
                              color: isActive ? colors.interactive.selectedText : colors.text.primary,
                              "&:hover": { bgcolor: isActive ? colors.status.activeBg : colors.bg.hover } }}
                      >
                        <Box component="span"
                             title={dotTitle(f, isActive, isCached)}
                             sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: dotColor(f),
                                   flex: "0 0 auto" }} />
                        {isCached && (
                          <Box component="span"
                               title="In GPU VRAM cache — instant click"
                               sx={{ flex: "0 0 auto", fontSize: fontSizes.xs,
                                     color: colors.success.dot, lineHeight: 1,
                                     fontWeight: 700 }}>
                            ⚡
                          </Box>
                        )}
                        <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {f.name}
                        </span>
                        {isSelected && (
                          <Box component="span"
                               title={`#${chipNum} in 5D set`}
                               sx={{ display: "inline-flex", alignItems: "center", justifyContent: "center",
                                     width: 18, height: 18, borderRadius: "50%",
                                     bgcolor: colors.interactive.bg,
                                     color: colors.interactive.selectedText,
                                     border: `1px solid ${colors.interactive.border}`,
                                     fontSize: fontSizes.xs, fontWeight: 700,
                                     fontFamily: "ui-monospace,monospace",
                                     flex: "0 0 auto" }}>
                            {chipNum}
                          </Box>
                        )}
                        <span style={{ fontSize: fontSizes.xs, color: colors.text.muted, fontFamily: "ui-monospace,monospace" }}>
                          {f.shape[0] && f.shape[2]
                            ? f.shape.join("×")
                            : f.size}
                        </span>
                      </Box>
                    );
                  })}
                </Box>
              )}
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}

// --- Hover popover ---------------------------------------------------------

interface PopoverState {
  anchor: HTMLElement;
  fileId: string;
  session: Session;
  file: MasterFile;
  metadata: MasterMetadata | null;
  error: string | null;
}

function MetadataPopover({
  state, onPointerEnter, onPointerLeave,
}: {
  state: PopoverState | null;
  onPointerEnter: () => void;
  onPointerLeave: () => void;
}) {
  const open = !!state;
  return (
    <Popper
      open={open}
      anchorEl={state?.anchor ?? null}
      placement="right-start"
      transition
      modifiers={[
        { name: "offset", options: { offset: [0, 12] } },
        { name: "preventOverflow", options: { padding: 8 } },
        { name: "flip", options: { fallbackPlacements: ["right", "left-start", "bottom-start"] } },
      ]}
      sx={{ zIndex: 1400 }}
    >
      {({ TransitionProps }) => (
        <Fade {...TransitionProps} timeout={120}>
          <Box
            onMouseEnter={onPointerEnter}
            onMouseLeave={onPointerLeave}
            sx={{
              minWidth: 320, maxWidth: 420, maxHeight: 520, overflow: "auto",
              bgcolor: colors.dark.bg.elevated,
              color: colors.dark.text.primary,
              border: `1px solid ${colors.border.default}`,
              borderRadius: `${radii.md}px`,
              boxShadow: "0 12px 40px rgba(0,0,0,0.55)",
              p: 1.25,
            }}
          >
            {state && <PopoverContents state={state} />}
          </Box>
        </Fade>
      )}
    </Popper>
  );
}

function PopoverContents({ state }: { state: PopoverState }) {
  const { file, metadata, error } = state;
  const broken = file.loadable === false;
  const status = file.load_status;
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
      <Box sx={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 1 }}>
        <Typography sx={{ fontSize: fontSizes.md, fontWeight: 700, fontFamily: MONO,
                          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {file.name}
        </Typography>
        <Typography sx={{ fontSize: fontSizes.xs, color: colors.dark.text.muted, fontFamily: MONO,
                          flex: "0 0 auto" }}>
          {metadata?.file_size ?? file.size}
        </Typography>
      </Box>
      <Typography sx={{ fontSize: fontSizes.xs, color: colors.dark.text.secondary, fontFamily: MONO }}>
        {metadata?.shape_summary ?? (
          file.shape[0] && file.shape[2]
            ? `${file.shape[0]}×${file.shape[1]} scan × ${file.shape[2]}×${file.shape[3]} detector`
            : "shape unknown"
        )}
      </Typography>

      {broken && status && (
        <Box sx={{ mt: 0.5, p: 0.75, borderRadius: `${radii.sm}px`,
                   bgcolor: "rgba(252,129,129,0.10)",
                   border: `1px solid ${colors.dark.status.error}` }}>
          <Typography sx={{ fontSize: fontSizes.xs, color: colors.dark.status.error,
                            fontFamily: MONO, fontWeight: 600 }}>
            {status.reason}
          </Typography>
          {status.missing_chunks.length > 0 && (
            <Box sx={{ mt: 0.5, fontSize: fontSizes.xs, color: colors.dark.text.secondary,
                       fontFamily: MONO, lineHeight: 1.4 }}>
              {status.missing_chunks.slice(0, 6).map((c) => (
                <div key={c} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c}</div>
              ))}
              {status.missing_chunks.length > 6 && (
                <div style={{ color: colors.dark.text.muted }}>
                  …and {status.missing_chunks.length - 6} more
                </div>
              )}
            </Box>
          )}
        </Box>
      )}

      {!broken && !metadata && !error && (
        <Typography sx={{ fontSize: fontSizes.xs, color: colors.dark.text.muted, fontFamily: MONO,
                          mt: 0.5 }}>
          loading metadata…
        </Typography>
      )}

      {error && (
        <Typography sx={{ fontSize: fontSizes.xs, color: colors.dark.status.error,
                          fontFamily: MONO, mt: 0.5 }}>
          {error}
        </Typography>
      )}

      {metadata && metadata.fields.length > 0 && (
        <Box sx={{ mt: 0.5, display: "grid",
                   gridTemplateColumns: "minmax(120px, auto) 1fr",
                   columnGap: 1, rowGap: 0.25 }}>
          {metadata.fields.map((f, i) => (
            <FieldRow key={`${f.key}-${i}`} k={f.key} v={f.value} />
          ))}
        </Box>
      )}
    </Box>
  );
}

function FieldRow({ k, v }: { k: string; v: string }) {
  return (
    <>
      <Box sx={{ fontSize: fontSizes.xs, color: colors.dark.text.muted, fontFamily: MONO,
                 whiteSpace: "nowrap" }}>
        {k}
      </Box>
      <Box sx={{ fontSize: fontSizes.xs, color: colors.dark.text.primary, fontFamily: MONO,
                 overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
           title={v}>
        {v}
      </Box>
    </>
  );
}
