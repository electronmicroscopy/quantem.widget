import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import useMediaQuery from "@mui/material/useMediaQuery";
import IconButton from "@mui/material/IconButton";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Slider from "@mui/material/Slider";
import Tooltip from "@mui/material/Tooltip";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import SkipNextIcon from "@mui/icons-material/SkipNext";
import SkipPreviousIcon from "@mui/icons-material/SkipPrevious";
import { colors, fontSizes, radii, breakpoints } from "../../theme";
import { useRegisterShortcuts, type Shortcut } from "../../hooks/useKeyboardShortcuts";
import FileTree from "./FileTree";
import Viewer from "./Viewer";
import MetaRail from "./MetaRail";
import { canRefreshWatchedFolders, pickFolderAndScan, refreshWatchedFolders } from "../../local/folderPicker";
import {
  defaultSelection, fetchSessions, fetchGpuFreeBytes, lastScanSkipped,
  fileKey, findFile, pickAutoBin, planWarmSet5D, preloadSet5D,
  type BrowseDtype, type ColormapName, type DetBin, type DetBinSetting, type DetectorMode,
  type DetShape, type MasterFile, type Session, type Set5D, type ShapeParams,
} from "./types";

/** /browse/<source>/<date>/<file> → resolve against `sessions`. Returns
 *  null until sessions arrive (the viewer then renders an empty stage). */
function resolveBrowsePath(
  pathname: string, sessions: Session[],
): { session: Session; file: MasterFile } | null {
  const tail = pathname.replace(/^\/browse\/?/, "");
  if (!tail) return defaultSelection(sessions);
  const parts = tail.split("/").filter(Boolean);
  if (parts.length >= 3) {
    const [source, date, ...rest] = parts;
    const name = rest.join("/");
    const found = findFile(sessions, source, date, name);
    if (found) return found;
  }
  return defaultSelection(sessions);
}

function buildBrowsePath(s: Session, f: MasterFile): string {
  return `/browse/${s.source}/${s.date}/${f.name}`;
}

const NARROW_MAX_PX = 1280;
const RAIL_COLLAPSED_PX = 44;
const LEFT_RAIL_DEFAULT_PX = 260;
const LEFT_RAIL_MIN_PX = 220;
const LEFT_RAIL_MAX_PX = 560;
const BROWSE_STACK_FPS_OPTIONS = [1, 2, 5, 10, 15];
const FOLDER_WATCH_INTERVAL_MS = 1000;
const BROWSE_SHORTCUTS: Shortcut[] = [
  { key: "drag", label: "Move scan crosshair", group: "Browse viewer", handler: () => {} },
  { key: "↑/↓/←/→", label: "Step scan or detector position", group: "Browse viewer", handler: () => {} },
  { key: "Shift+↑/↓/←/→", label: "Step 10 pixels", group: "Browse viewer", handler: () => {} },
  { key: "R", label: "Reset pan and zoom", group: "Browse viewer", handler: () => {} },
  { key: "scroll", label: "Zoom the hovered panel", group: "Browse viewer", handler: () => {} },
];

function clampLeftRailWidth(width: number): number {
  if (!Number.isFinite(width)) return LEFT_RAIL_DEFAULT_PX;
  return Math.round(Math.max(LEFT_RAIL_MIN_PX, Math.min(LEFT_RAIL_MAX_PX, width)));
}

/** Slim 44 px sidebar shown when a rail is collapsed. Click anywhere on the
 *  rail to expand it. The icon + vertical label tells the user which rail
 *  they're toggling without taking real-estate from the canvas. */
/** Thin clickable strip on a rail's INNER edge — invisible by default,
 *  shows a chevron on hover, click collapses the rail. Mirrors the slim-
 *  sidebar expand pattern so collapse is just as discoverable as expand. */
function EdgeCollapser({
  side, onClick, onResizeStart, onResizeReset, widthValue,
}: {
  /** Which rail's edge: "left" rail collapses with ◀ on its right edge,
   *  "right" rail collapses with ▶ on its left edge. */
  side: "left" | "right";
  onClick: () => void;
  onResizeStart?: (e: React.PointerEvent<HTMLElement>) => void;
  onResizeReset?: () => void;
  widthValue?: number;
}) {
  const arrow = side === "left" ? "◀" : "▶";
  const edgeStyle = side === "left" ? { right: -7 } : { left: 0 };
  const resizable = !!onResizeStart;
  return (
    <Box
      onPointerDown={(e: React.PointerEvent<HTMLElement>) => {
        if (!onResizeStart) return;
        e.stopPropagation();
        onResizeStart(e);
      }}
      onDoubleClick={(e: React.MouseEvent) => {
        if (!resizable) return;
        e.stopPropagation();
        onResizeReset?.();
      }}
      role={resizable ? "separator" : "button"}
      aria-label={resizable ? `${side} rail resize handle` : `collapse ${side} rail`}
      aria-orientation={resizable ? "vertical" : undefined}
      aria-valuemin={resizable ? LEFT_RAIL_MIN_PX : undefined}
      aria-valuemax={resizable ? LEFT_RAIL_MAX_PX : undefined}
      aria-valuenow={resizable ? widthValue : undefined}
      sx={{ position: "absolute", top: 0, bottom: 0, width: 14, ...edgeStyle,
            display: "flex", alignItems: "center", justifyContent: "center",
            cursor: resizable ? "col-resize" : "pointer", userSelect: "none", zIndex: 2,
            color: colors.text.muted, fontSize: fontSizes.xs,
            opacity: 0, transition: "opacity 120ms",
            "&:hover": { opacity: 1, bgcolor: colors.bg.hover } }}
    >
      <Box
        component="button"
        type="button"
        title={`Collapse ${side} rail`}
        aria-label={`collapse ${side} rail`}
        onPointerDown={(e: React.PointerEvent) => e.stopPropagation()}
        onClick={(e: React.MouseEvent) => { e.stopPropagation(); onClick(); }}
        sx={{ width: 18, height: 26, p: 0, appearance: "none",
              display: "flex", alignItems: "center", justifyContent: "center",
              cursor: "pointer", border: `1px solid ${colors.border.default}`,
              borderRadius: radii.sm, bgcolor: colors.bg.page, color: colors.text.secondary,
              fontSize: fontSizes.xs, lineHeight: 1,
              "&:hover": { bgcolor: colors.bg.hover, color: colors.text.primary } }}
      >
        {arrow}
      </Box>
    </Box>
  );
}

function CollapsedRail({
  side, label, icon, onClick,
}: {
  side: "left" | "right";
  label: string;
  icon: string;
  onClick: () => void;
}) {
  return (
    <Box
      onClick={onClick}
      role="button"
      aria-label={`expand ${label} rail`}
      sx={{ width: RAIL_COLLAPSED_PX, display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "flex-start", gap: 1,
            py: 1.5, cursor: "pointer", userSelect: "none",
            border: `1px solid ${colors.border.default}`,
            borderRadius: radii.lg, bgcolor: colors.bg.page,
            "&:hover": { bgcolor: colors.bg.hover } }}
    >
      <Box sx={{ fontSize: fontSizes.lg, lineHeight: 1, color: colors.text.secondary }}>{icon}</Box>
      <Box sx={{ writingMode: "vertical-rl", transform: side === "left" ? "rotate(180deg)" : "none",
                 fontSize: fontSizes.xs, color: colors.text.muted, fontWeight: 600,
                 letterSpacing: 1, textTransform: "uppercase" }}>
        {label}
      </Box>
    </Box>
  );
}

function BrowseStackViewer({
  session,
  activeFile,
  activeSet,
  onScrubIndex,
  onActivateStack,
}: {
  session: Session;
  activeFile: MasterFile;
  activeSet: Set5D | null;
  onScrubIndex: (idx: number) => void;
  onActivateStack: (idx: number) => void;
}) {
  const [open, setOpen] = useState(() => {
    try { return localStorage.getItem("browse.stackViewerOpen") === "1"; }
    catch { return false; }
  });
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(5);
  const files = useMemo(
    () => session.files.filter((f) => f.loadable !== false),
    [session],
  );
  const activeKey = fileKey(session, activeFile);
  const activeIdx = Math.max(0, files.findIndex((f) => fileKey(session, f) === activeKey));
  const activeStack = !!activeSet
    && activeSet.session.source === session.source
    && activeSet.session.date === session.date;

  useEffect(() => {
    try { localStorage.setItem("browse.stackViewerOpen", open ? "1" : "0"); }
    catch { /* ignore */ }
  }, [open]);

  useEffect(() => {
    if (!playing || files.length < 2) return;
    const id = window.setInterval(() => {
      onScrubIndex((activeIdx + 1) % files.length);
    }, Math.max(80, 1000 / fps));
    return () => window.clearInterval(id);
  }, [playing, fps, files.length, activeIdx, onScrubIndex]);

  if (files.length < 2) return null;
  const go = (delta: number) => {
    setPlaying(false);
    onScrubIndex((activeIdx + delta + files.length) % files.length);
  };

  return (
    <Box data-testid="browse-stack-viewer"
         sx={{ bgcolor: colors.text.white, border: `1px solid ${colors.border.default}`,
               borderRadius: radii.sm, overflow: "hidden", flexShrink: 0 }}>
      <Box
        onClick={() => setOpen(v => !v)}
        sx={{ display: "flex", alignItems: "center", gap: 1,
              px: 1, py: 0.5, cursor: "pointer",
              bgcolor: open ? colors.bg.subtle : undefined,
              "&:hover": { bgcolor: colors.bg.hover } }}
      >
        <Typography sx={{ flexShrink: 0, fontSize: fontSizes.sm, fontWeight: 700,
                          color: colors.text.secondary, whiteSpace: "nowrap" }}>
          {open ? "▾" : "▸"} Stack viewer
        </Typography>
        <Typography sx={{ fontSize: fontSizes.sm, color: colors.text.subtle,
                          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {activeIdx + 1}/{files.length} · {activeFile.name}
        </Typography>
      </Box>
      {open && (
        <Box sx={{ p: 0.75, display: "flex", alignItems: "center",
                   gap: 0.75, flexWrap: "wrap" }}>
          <Tooltip title="Previous master">
            <span>
              <IconButton size="small" onClick={() => go(-1)}
                          sx={{ minWidth: { xs: 44, sm: 28 }, minHeight: { xs: 44, sm: 28 } }}>
                <SkipPreviousIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title={playing ? "Pause session stack" : "Play session stack"}>
            <span>
              <IconButton size="small" onClick={() => setPlaying(v => !v)}
                          sx={{ minWidth: { xs: 44, sm: 28 }, minHeight: { xs: 44, sm: 28 } }}>
                {playing ? <PauseIcon fontSize="small" /> : <PlayArrowIcon fontSize="small" />}
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Next master">
            <span>
              <IconButton size="small" onClick={() => go(1)}
                          sx={{ minWidth: { xs: 44, sm: 28 }, minHeight: { xs: 44, sm: 28 } }}>
                <SkipNextIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Select
            size="small"
            value={fps}
            onChange={(e) => setFps(Number(e.target.value))}
            sx={{ height: { xs: 44, sm: 28 }, minWidth: 72, fontSize: fontSizes.sm,
                  "& .MuiSelect-select": { py: 0.25 } }}
          >
            {BROWSE_STACK_FPS_OPTIONS.map(v => <MenuItem key={v} value={v}>{v} fps</MenuItem>)}
          </Select>
          <ToolbarButton
            active={activeStack}
            onClick={() => onActivateStack(activeIdx)}
            title={activeStack ? "Session stack is preloaded" : "Preload current session as a 5D stack"}
          >
            5D
          </ToolbarButton>
          <Box sx={{ flex: 1, minWidth: { xs: "100%", sm: 220 }, px: { xs: 0, sm: 1 } }}>
            <Slider
              size="small"
              min={0}
              max={files.length - 1}
              value={activeIdx}
              onChange={(_, v) => {
                setPlaying(false);
                onScrubIndex(Array.isArray(v) ? v[0] : v);
              }}
              aria-label="Browse session stack index"
            />
          </Box>
          <Typography sx={{ fontSize: fontSizes.xs, color: colors.text.faint,
                            fontFamily: "monospace", maxWidth: { xs: "100%", md: 360 },
                            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {files[activeIdx]?.name}
          </Typography>
        </Box>
      )}
    </Box>
  );
}

export default function Browse() {
  const location = useLocation();
  const navigate = useNavigate();
  useRegisterShortcuts(BROWSE_SHORTCUTS);

  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [activeFile, setActiveFile] = useState<MasterFile | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [scanNotice, setScanNotice] = useState("");   // transient "skipped N bad files" banner
  const [folderScanBusy, setFolderScanBusy] = useState(false);
  const [folderWatchEnabled, setFolderWatchEnabled] = useState(false);
  const [folderCanRefresh, setFolderCanRefresh] = useState(false);
  const [folderWatchBusy, setFolderWatchBusy] = useState(false);
  const folderWatchInFlightRef = useRef(false);
  const activeSessionRef = useRef<Session | null>(null);
  useEffect(() => { activeSessionRef.current = activeSession; }, [activeSession]);

  // Cold-load sessions once.
  useEffect(() => {
    let alive = true;
    fetchSessions()
      .then((rows) => {
        if (!alive) return;
        setSessions(rows);
        const sel = resolveBrowsePath(location.pathname, rows);
        if (sel) {
          setActiveSession(sel.session);
          setActiveFile(sel.file);
        }
      })
      .catch((e: Error) => {
        if (!alive) return;
        setLoadError(e.message || "failed to load sessions");
      });
    return () => { alive = false; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-read the dataset tree whenever a folder is picked (in-page "Choose folder" or a dev hook).
  // Auto-select the first dataset if nothing is active yet, so picking immediately shows content.
  useEffect(() => {
    const onLoaded = () => void fetchSessions().then((rows) => {
      setSessions(rows);
      const n = lastScanSkipped();
      if (n > 0) { setScanNotice(`Skipped ${n} unreadable file${n > 1 ? "s" : ""} (corrupt / not 4D-STEM).`); window.setTimeout(() => setScanNotice(""), 7000); }
      setActiveFile((cur) => {
        if (cur) {
          const curSession = activeSessionRef.current;
          const curKey = curSession ? fileKey(curSession, cur) : cur.name;
          const sel = rows.flatMap((session) => session.files.map((file) => ({ session, file })))
            .find(({ session, file }) => fileKey(session, file) === curKey || file.name === curKey);
          if (sel) { setActiveSession(sel.session); return sel.file; }
        }
        const sel = defaultSelection(rows);
        if (sel) { setActiveSession(sel.session); return sel.file; }
        setActiveSession(null);
        return null;
      });
    });
    window.addEventListener("quantem-folder-loaded", onLoaded);
    return () => window.removeEventListener("quantem-folder-loaded", onLoaded);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // URL → selection (Back/Forward / deep links).
  useEffect(() => {
    if (sessions.length === 0) return;
    const sel = resolveBrowsePath(location.pathname, sessions);
    if (!sel) return;
    if (sel.session !== activeSession) setActiveSession(sel.session);
    if (sel.file !== activeFile) setActiveFile(sel.file);
  }, [location.pathname, sessions]); // eslint-disable-line react-hooks/exhaustive-deps

  // Selection → URL.
  useEffect(() => {
    if (!activeSession || !activeFile) return;
    const target = buildBrowsePath(activeSession, activeFile);
    if (location.pathname !== target) {
      navigate(target, { replace: true });
    }
  }, [activeSession, activeFile]); // eslint-disable-line react-hooks/exhaustive-deps

  // 5D-STEM state. ``selectedKeys`` is the per-session multi-select buffer
  // (cmd/shift-click). Once the user clicks "load 5D" we promote that into
  // ``set5D`` which actually drives the viewer's scrubber strip. The two are
  // separate so the buffer can grow before triggering a backend preload.
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const [selectionSession, setSelectionSession] = useState<Session | null>(null);
  const [selectionAnchorKey, setSelectionAnchorKey] = useState<string | null>(null);
  const [pendingDetBin, setPendingDetBin] = useState<DetBinSetting>("auto");
  // Browse-block precision. uint8 default halves VRAM (~2× more masters cache);
  // lossless when raw counts ≤ 255. Persisted so the choice survives reloads.
  const [browseDtype, setBrowseDtypeState] = useState<BrowseDtype>(
    () => (typeof localStorage !== "undefined"
      && localStorage.getItem("browse.dtype") === "uint16" ? "uint16" : "uint8"));
  const toggleBrowseDtype = () => setBrowseDtypeState((prev) => {
    const next: BrowseDtype = prev === "uint8" ? "uint16" : "uint8";
    try { localStorage.setItem("browse.dtype", next); } catch { /* ignore */ }
    return next;
  });
  const [set5D, setSet5D] = useState<Set5D | null>(null);
  // Live GPU free-bytes for the per-folder VRAM indicator chip. Subscribes
  // to /api/gpu/stream — same SSE channel the dashboard's status chip uses
  // (2 s cadence). Lets the chip's auto-bin estimate stay current as other
  // tabs evict masters, without yet another HTTP poll.
  const [gpuFreeBytes, setGpuFreeBytes] = useState(0);
  useEffect(() => {
    void fetchGpuFreeBytes().then(setGpuFreeBytes).catch(() => {});
    const es = new EventSource(`/api/gpu/stream`);
    es.onmessage = (ev) => {
      try {
        const snap = JSON.parse(ev.data);
        const gpu0 = (snap?.gpus ?? []).find((g: { index: number }) => g.index === 0);
        if (!gpu0) return;
        const usedMb = Number(gpu0.mem_used_mb ?? 0);
        const totalMb = Number(gpu0.mem_total_mb ?? 0);
        const free = Math.max(0, (totalMb - usedMb) * 1024 * 1024);
        setGpuFreeBytes(free);
      } catch {
        // ignore malformed snapshot — next event will overwrite
      }
    };
    es.onerror = () => {
      // EventSource auto-reconnects via the server's `retry` hint.
    };
    return () => { es.close(); };
  }, []);

  const singleDetBin = useMemo<DetBin>(() => (
    activeFile ? pickAutoBin([activeFile], gpuFreeBytes || 4 * 1024 * 1024 * 1024, 0.45, browseDtype) : 1
  ), [activeFile, gpuFreeBytes, browseDtype]);

  // Multi-select toggle handler. Cmd-click toggles inclusion; shift-click
  // extends a contiguous range from the last toggle anchor to the current
  // row using the FileTree's order. Both modes are scoped to one session —
  // selecting in a different session clears the buffer.
  const onToggleSelect = (s: Session, f: MasterFile, mode: "toggle" | "range") => {
    const targetKey = fileKey(s, f);
    setSelectedKeys((prev) => {
      const sameSession = selectionSession?.source === s.source && selectionSession?.date === s.date;
      const base = sameSession ? prev : [];
      if (mode === "toggle") {
        const idx = base.indexOf(targetKey);
        if (idx >= 0) {
          const next = base.slice();
          next.splice(idx, 1);
          return next;
        }
        return [...base, targetKey];
      }
      // range: from anchor (or first selected) to target, in session order.
      if (base.length === 0) {
        // First click in shift mode behaves like toggle.
        return [targetKey];
      }
      const anchorKey = selectionAnchorKey ?? base[base.length - 1];
      const order = s.files.map((ff) => fileKey(s, ff));
      const a = order.indexOf(anchorKey);
      const b = order.indexOf(targetKey);
      if (a < 0 || b < 0) return [...base, targetKey];
      const [lo, hi] = a <= b ? [a, b] : [b, a];
      const range = order.slice(lo, hi + 1).filter((k) => {
        const ff = s.files[order.indexOf(k)];
        return ff && ff.loadable !== false;
      });
      // Merge: existing selection + range, dedup, preserve buffer order
      // for old entries and append new ones.
      const seen = new Set(base);
      const merged = base.slice();
      for (const k of range) {
        if (!seen.has(k)) { merged.push(k); seen.add(k); }
      }
      return merged;
    });
    setSelectionSession(s);
    setSelectionAnchorKey(targetKey);
  };

  /** Click the per-folder "5D · bin X · Y GB" chip → multi-select every
   *  loadable file in that session, set the binning the chip showed, and
   *  fire the same preload path as the manual button. Skips broken masters. */
  const onLoadFolder5D = (s: Session, files: MasterFile[]) => {
    const loadable = files.filter((f) => f.loadable !== false);
    if (loadable.length < 2) return;
    const keys = loadable.map((f) => fileKey(s, f));
    setSelectedKeys(keys);
    setSelectionSession(s);
    setSelectionAnchorKey(keys[keys.length - 1]);
    // Pretend the user clicked "load 5D" with auto binning.
    setPendingDetBin("auto");
    // Run the load on the next tick so the state updates flush first.
    setTimeout(() => { void onLoad5DInternal(s, loadable, "auto"); }, 0);
  };

  const onLoad5DInternal = async (
    sess: Session, orderedFiles: MasterFile[], setting: DetBinSetting, initialIdx = 0,
  ) => {
    const free = await fetchGpuFreeBytes();
    let resolvedBin: DetBin;
    if (setting === "auto") {
      resolvedBin = pickAutoBin(orderedFiles, free, 0.45, browseDtype);
      // eslint-disable-next-line no-console
      console.log(`5D auto-bin: ${orderedFiles.length} files, ${(free / 1e9).toFixed(1)} GB free → bin=${resolvedBin}`);
    } else {
      resolvedBin = setting;
    }
    const activeIdx = Math.max(0, Math.min(orderedFiles.length - 1, initialIdx));
    const warmPlan = planWarmSet5D(orderedFiles, activeIdx, free, resolvedBin, browseDtype);
    const next: Set5D = {
      session: sess, files: orderedFiles, activeIdx, detBin: resolvedBin,
      warmCount: warmPlan.files.length, warmTotal: warmPlan.totalFiles, warmMode: warmPlan.mode,
    };
    setSet5D(next);
    setActiveSession(sess);
    setActiveFile(orderedFiles[activeIdx]);
    void preloadSet5D(sess, orderedFiles, resolvedBin, browseDtype, activeIdx, free).catch((err) => {
      // eslint-disable-next-line no-console
      console.warn("preloadSet5D failed:", err);
    });
  };

  const onLoad5D = async () => {
    if (!selectionSession || selectedKeys.length < 2) return;
    // Resolve ordered keys -> MasterFiles in the session.
    const fmap = new Map<string, MasterFile>();
    for (const ff of selectionSession.files) fmap.set(fileKey(selectionSession, ff), ff);
    const orderedFiles = selectedKeys
      .map((k) => fmap.get(k))
      .filter((f): f is MasterFile => !!f);
    if (orderedFiles.length < 2) return;
    // Resolve "auto" to a concrete DetBin by sizing against the conservative
    // browser memory proxy. The warm planner may still choose a sliding window
    // rather than pinning the whole set.
    const free = await fetchGpuFreeBytes();
    let resolvedBin: DetBin;
    if (pendingDetBin === "auto") {
      resolvedBin = pickAutoBin(orderedFiles, free, 0.45, browseDtype);
      // eslint-disable-next-line no-console
      console.log(`5D auto-bin: ${orderedFiles.length} files, ${(free / 1e9).toFixed(1)} GB free → bin=${resolvedBin}`);
    } else {
      resolvedBin = pendingDetBin;
    }
    const warmPlan = planWarmSet5D(orderedFiles, 0, free, resolvedBin, browseDtype);
    const next: Set5D = {
      session: selectionSession,
      files: orderedFiles,
      activeIdx: 0,
      detBin: resolvedBin,
      warmCount: warmPlan.files.length,
      warmTotal: warmPlan.totalFiles,
      warmMode: warmPlan.mode,
    };
    setSet5D(next);
    setActiveSession(selectionSession);
    setActiveFile(orderedFiles[0]);
    void preloadSet5D(selectionSession, orderedFiles, resolvedBin, browseDtype, 0, free).catch((err) => {
      // Surface as console warning — the user's URL still works, fetches
      // for the active master will trigger a single-master load on demand.
      // eslint-disable-next-line no-console
      console.warn("preloadSet5D failed:", err);
    });
  };

  // Clear set5D when user navigates to a file in a different session, or
  // when the multi-select buffer becomes empty.
  const setActiveIdx = (idx: number) => {
    if (!set5D) return;
    const clamped = Math.max(0, Math.min(set5D.files.length - 1, idx));
    const warmPlan = planWarmSet5D(set5D.files, clamped, gpuFreeBytes, set5D.detBin, browseDtype);
    setSet5D({
      ...set5D,
      activeIdx: clamped,
      warmCount: warmPlan.files.length,
      warmTotal: warmPlan.totalFiles,
      warmMode: warmPlan.mode,
    });
    setActiveFile(set5D.files[clamped]);
    void preloadSet5D(set5D.session, set5D.files, set5D.detBin, browseDtype, clamped, gpuFreeBytes).catch((err) => {
      // eslint-disable-next-line no-console
      console.warn("preloadSet5D failed:", err);
    });
  };

  const sessionStackFiles = useMemo(
    () => activeSession?.files.filter((f) => f.loadable !== false) ?? [],
    [activeSession],
  );
  const scrubSessionIndex = useCallback((idx: number) => {
    if (!activeSession || sessionStackFiles.length === 0) return;
    const clamped = Math.max(0, Math.min(sessionStackFiles.length - 1, idx));
    const activeStack = !!set5D
      && set5D.session.source === activeSession.source
      && set5D.session.date === activeSession.date;
    if (activeStack) {
      setActiveIdx(clamped);
      return;
    }
    setActiveFile(sessionStackFiles[clamped]);
  }, [activeSession, sessionStackFiles, set5D]);
  const activateSessionStack = useCallback((idx: number) => {
    if (!activeSession || sessionStackFiles.length < 2) return;
    void onLoad5DInternal(activeSession, sessionStackFiles, pendingDetBin, idx);
  }, [activeSession, sessionStackFiles, pendingDetBin]);

  // When set5D is active, navigating to a different file via the file tree
  // (single-click) should clear the 5D set so the user can browse normally.
  const handleTreeSelect = (s: Session, f: MasterFile) => {
    if (set5D) {
      const inSet = set5D.files.some((ff) => fileKey(set5D.session, ff) === fileKey(s, f));
      if (!inSet) setSet5D(null);
      else {
        // Clicking a file already in the set just navigates within the set.
        const idx = set5D.files.findIndex((ff) => fileKey(set5D.session, ff) === fileKey(s, f));
        if (idx >= 0) setActiveIdx(idx);
      }
    }
    setActiveSession(s);
    setActiveFile(f);
    // Clear multi-select buffer when user clicks (vs cmd/shift-clicks).
    setSelectedKeys([]);
    setSelectionAnchorKey(null);
  };

  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<DetectorMode>("BF");
  const [cmapImage, setCmapImage] = useState<ColormapName>("viridis");
  const [cmapDp, setCmapDp] = useState<ColormapName>("inferno");
  const [scanPos, setScanPos] = useState({ x: 0.5, y: 0.5 });
  const [ringInner, setRingInner] = useState(0);
  const [ringOuter, setRingOuter] = useState(1.0);
  // Detector-shape selector (Show4DSTEM-style). When dpShape !== "circle"
  // OR mode is BF/ADF/DF, the Viewer routes through the new
  // /realspace-shape endpoint instead of the legacy /realspace one. The
  // CoM/iCoM/SSB modes (CoMmag/CoMx/CoMy/iCoM/SSB) ignore this and stay on
  // /realspace.
  const [dpShape, setDpShape] = useState<DetShape>("circle");
  const [shapeParams, setShapeParams] = useState<ShapeParams>({
    cx: 0, cy: 0, r: 0, half: 0, inner: 0, outer: 0,
    row0: 0, col0: 0, row1: 0, col1: 0, px: 0, py: 0,
  });
  const [clipLo, setClipLo] = useState(0.02);
  const [clipHi, setClipHi] = useState(0.98);
  const [dpClipLo, setDpClipLo] = useState(0.02);
  const [dpClipHi, setDpClipHi] = useState(0.98);
  // Real-space rectangular ROI (in normalized 0..1 scan coords). When
  // non-null, the DP panel shows the SUM of CBEDs over the rectangle
  // instead of a single-position frame. Esc / clear-button → null.
  const [realRoi, setRealRoi] = useState<
    { row0: number; col0: number; row1: number; col1: number } | null
  >(null);
  // FFT-of-virtual-image panel toggle. When ON, the real-space column
  // splits into image (top) + FFT magnitude (bottom).
  const [fftOn, setFftOn] = useState(false);
  // Hann window before FFT. Show4DSTEM `fft_window` parity (default ON).
  const [fftWindow, setFftWindow] = useState(true);
  // Per-panel intensity scale: linear / log / sqrt. Defaults match the
  // historical behavior (DP was hardcoded log; image was implicit linear).
  const [imageScale, setImageScale] = useState<"linear" | "log" | "sqrt" | "power">("linear");
  const [dpScale, setDpScale] = useState<"linear" | "log" | "sqrt" | "power">("log");
  // Show4DSTEM dp_power_exp / vi_power_exp parity, default 0.5.
  const [imagePowerExp, setImagePowerExp] = useState(0.5);
  const [dpPowerExp, setDpPowerExp] = useState(0.5);
  // Show4DSTEM mask_dc parity, default ON.
  const [maskDC, setMaskDC] = useState(true);
  // Show2D / Show4DSTEM line profile on the virtual image. Null when no
  // profile is set; profileWidth defaults to 1 (single bilinear line).
  const [profileLine, setProfileLine] = useState<{ row0: number; col0: number; row1: number; col1: number } | null>(null);
  const [profileWidth, setProfileWidth] = useState(1);

  // Show4DSTEM-style detector presets in BF-radius units:
  //   BF  = central disk, 0→1×r_BF
  //   DF  = inner annular dark-field band, 1→2×r_BF
  //   ADF = broader annular dark-field band, 1→3×r_BF
  // CoM-family modes do not use a ring, so leave radii unchanged.
  useEffect(() => {
    if (mode === "BF")       { setRingInner(0);   setRingOuter(1.0); }
    else if (mode === "ADF") { setRingInner(1.0); setRingOuter(3.0); }
    else if (mode === "DF")  { setRingInner(1.0); setRingOuter(2.0); }
    // CoMmag / DPC / CoMx / CoMy / iCoM / SSB: no ring change on mode click.
  }, [mode]);


  const totalFiles = useMemo(
    () => sessions.reduce((a, s) => a + s.files.length, 0),
    [sessions],
  );

  // Responsive: < 1280 px viewport (typical small MacBook Air, side-by-side
  // browser windows on a larger MacBook) collapses both rails to slim
  // finger-sized sidebars by default. Wider viewports keep both rails open.
  const isNarrow = useMediaQuery(`(max-width: ${NARROW_MAX_PX}px)`);
  const isPhone = useMediaQuery(`(max-width: ${breakpoints.phone}px)`);

  // User overrides. null = follow auto/responsive default; true/false = force.
  const [leftOverride, setLeftOverride] = useState<boolean | null>(() => {
    try {
      const v = localStorage.getItem("browse.leftCollapsed");
      return v === null ? null : v === "1";
    } catch { return null; }
  });
  const [rightOverride, setRightOverride] = useState<boolean | null>(() => {
    try {
      const v = localStorage.getItem("browse.rightCollapsed");
      return v === null ? null : v === "1";
    } catch { return null; }
  });
  const [leftRailWidth, setLeftRailWidth] = useState<number>(() => {
    try {
      const raw = localStorage.getItem("browse.leftRailWidth");
      return raw === null ? LEFT_RAIL_DEFAULT_PX : clampLeftRailWidth(Number(raw));
    } catch { return LEFT_RAIL_DEFAULT_PX; }
  });
  const [leftRailResizing, setLeftRailResizing] = useState(false);

  // Compact mode: shrinks histograms + toolbar. Persisted across reloads.
  const [compact, setCompact] = useState<boolean>(() => {
    try { return localStorage.getItem("browse.compact") === "1"; } catch { return false; }
  });
  const toggleCompact = () => {
    setCompact((c) => {
      const next = !c;
      try { localStorage.setItem("browse.compact", next ? "1" : "0"); } catch { /* ignore */ }
      return next;
    });
  };
  const chooseFolder = useCallback(async () => {
    if (folderScanBusy) return;
    setFolderScanBusy(true);
    try {
      await pickFolderAndScan();
      const canRefresh = canRefreshWatchedFolders();
      setFolderCanRefresh(canRefresh);
      if (canRefresh) setFolderWatchEnabled(true);
    } finally {
      setFolderScanBusy(false);
    }
  }, [folderScanBusy]);

  const refreshFolder = useCallback(async () => {
    if (!canRefreshWatchedFolders() || folderWatchInFlightRef.current) return;
    folderWatchInFlightRef.current = true;
    setFolderWatchBusy(true);
    try {
      await refreshWatchedFolders();
      setFolderCanRefresh(canRefreshWatchedFolders());
    } finally {
      folderWatchInFlightRef.current = false;
      setFolderWatchBusy(false);
    }
  }, []);

  useEffect(() => {
    if (!folderWatchEnabled || !folderCanRefresh) return;
    const id = window.setInterval(() => { void refreshFolder(); }, FOLDER_WATCH_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [folderWatchEnabled, folderCanRefresh, refreshFolder]);

  const leftCollapsed = leftOverride !== null ? leftOverride : isNarrow;
  const rightCollapsed = rightOverride !== null ? rightOverride : isNarrow;
  const effectiveCompact = compact || isPhone;

  const setLeftCollapsed = (v: boolean) => {
    setLeftOverride(v);
    try { localStorage.setItem("browse.leftCollapsed", v ? "1" : "0"); } catch { /* ignore */ }
  };
  const setRightCollapsed = (v: boolean) => {
    setRightOverride(v);
    try { localStorage.setItem("browse.rightCollapsed", v ? "1" : "0"); } catch { /* ignore */ }
  };
  const resetLeftRailWidth = () => {
    setLeftRailWidth(LEFT_RAIL_DEFAULT_PX);
    try { localStorage.setItem("browse.leftRailWidth", String(LEFT_RAIL_DEFAULT_PX)); } catch { /* ignore */ }
  };
  const startLeftRailResize = (e: React.PointerEvent<HTMLElement>) => {
    if (leftCollapsed || e.button !== 0) return;
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = leftRailWidth;
    let latestWidth = startWidth;
    const prevCursor = document.body.style.cursor;
    const prevUserSelect = document.body.style.userSelect;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    setLeftRailResizing(true);
    const onMove = (ev: PointerEvent) => {
      latestWidth = clampLeftRailWidth(startWidth + ev.clientX - startX);
      setLeftRailWidth(latestWidth);
    };
    const finish = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      document.body.style.cursor = prevCursor;
      document.body.style.userSelect = prevUserSelect;
      setLeftRailResizing(false);
      try { localStorage.setItem("browse.leftRailWidth", String(latestWidth)); } catch { /* ignore */ }
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
  };

  // Three-column template — slim finger-sized sidebars when collapsed, full panel
  // widths otherwise. Center column always 1fr so the canvas keeps the
  // dominant share of the page no matter the viewport.
  const leftCol = leftCollapsed ? `${RAIL_COLLAPSED_PX}px` : `${leftRailWidth}px`;
  const rightCol = rightCollapsed ? `${RAIL_COLLAPSED_PX}px` : (effectiveCompact ? "260px" : "320px");
  const gridTemplate = `${leftCol} 1fr ${rightCol}`;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: { xs: 1, sm: 1.5 }, minHeight: 0,
              height: { md: "calc(100vh - 110px)" }, overflow: { md: "hidden" } }}>
      {/* Page head + page-level toolbar (compact toggle, rail toggles) */}
      <Box sx={{ display: "flex", alignItems: "flex-start", gap: 2, flexWrap: "wrap" }}>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, fontSize: fontSizes.xs, color: colors.text.muted,
                     overflow: "hidden", whiteSpace: "nowrap" }}>
            <span>browse</span>
            {activeSession && activeFile && (
              <>
                <span style={{ opacity: 0.5 }}>/</span>
                <span>{activeSession.source}</span>
                <span style={{ opacity: 0.5 }}>/</span>
                <span>{activeSession.date}</span>
                <span style={{ opacity: 0.5 }}>/</span>
                <Typography component="span" sx={{ fontSize: fontSizes.xs, color: colors.text.primary, fontWeight: 600 }}>
                  {activeFile.name}
                </Typography>
              </>
            )}
          </Box>
          <Typography sx={{ fontSize: { xs: fontSizes.xl, sm: fontSizes["2xl"] }, fontWeight: 700, mt: 0.5 }}>
            Browse 4D-STEM
          </Typography>
          <Typography sx={{ display: { xs: "none", sm: "block" }, fontSize: fontSizes.sm, color: colors.text.tertiary, mt: 0.25 }}>
            Open any master.h5, scrub the scan, generate virtual images on the fly. Hand off to
            Screening, SSB, or Ptychography in one click.
          </Typography>
          {scanNotice && (
            <Typography sx={{ mt: 0.5, fontSize: fontSizes.xs, fontWeight: 600, color: colors.warning.text }}>
              ⚠ {scanNotice}
            </Typography>
          )}
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mt: 0.5, flexWrap: "wrap" }}>
          <Box
            component="button"
            onClick={() => { void chooseFolder(); }}
            disabled={folderScanBusy}
            title="Open a folder of Arina .h5 datasets - decodes on your GPU, nothing leaves this machine"
            sx={{ px: 1.25, py: 0.5, fontSize: fontSizes.sm, fontWeight: 600, cursor: "pointer",
                  border: "none", borderRadius: radii.md, color: colors.text.white,
                  bgcolor: colors.text.primary, "&:hover": { opacity: 0.88 },
                  "&:disabled": { cursor: "wait", opacity: 0.66 } }}
          >
            {folderScanBusy ? "Scanning..." : "📂 Choose folder"}
          </Box>
          {folderCanRefresh && (
            <ToolbarButton
              active={folderWatchEnabled}
              onClick={() => setFolderWatchEnabled((v) => !v)}
              title={folderWatchEnabled
                ? "Stop polling the picked folder for new .h5 files"
                : "Poll the picked folder for new .h5 files"}
            >
              {folderWatchEnabled ? "watching" : "watch"}
            </ToolbarButton>
          )}
          {folderCanRefresh && !folderWatchEnabled && (
            <ToolbarButton
              active={folderWatchBusy}
              onClick={() => { void refreshFolder(); }}
              title="Refresh the picked folder now"
            >
              {folderWatchBusy ? "refreshing" : "refresh"}
            </ToolbarButton>
          )}
          <ToolbarButton
            active={!leftCollapsed}
            onClick={() => setLeftCollapsed(!leftCollapsed)}
            title={leftCollapsed ? "Show file tree" : "Hide file tree"}
          >
            ☰ files
          </ToolbarButton>
          <ToolbarButton
            active={effectiveCompact}
            onClick={toggleCompact}
            title="Toggle compact layout (smaller histograms, tighter toolbar)"
          >
            compact ⇄ comfy
          </ToolbarButton>
          <ToolbarButton
            active={!rightCollapsed}
            onClick={() => setRightCollapsed(!rightCollapsed)}
            title={rightCollapsed ? "Show metadata panels" : "Hide metadata panels"}
          >
            metadata ▾
          </ToolbarButton>
        </Box>
      </Box>

      {/* Three-column responsive layout. `flex: 1, minHeight: 0` lets the
          grid consume whatever vertical space the page-head doesn't, so each
          column scrolls internally instead of pushing the page taller. */}
      <Box sx={{ display: "grid",
                 gridTemplateColumns: { xs: "1fr", md: gridTemplate },
                 gap: { xs: 1, sm: 1.5 }, minHeight: 0, flex: 1,
                 transition: leftRailResizing ? "none" : "grid-template-columns 200ms ease" }}>
        {leftCollapsed ? (
          <CollapsedRail side="left" label="datasets" icon="☰"
                         onClick={() => setLeftCollapsed(false)} />
        ) : (
          <Box sx={{ position: "relative", display: "flex", flexDirection: "column", minHeight: 0 }}>
            <FileTree
              sessions={sessions}
              activeFile={activeFile?.name || ""}
              onSelect={handleTreeSelect}
              query={query}
              setQuery={setQuery}
              loading={sessions.length === 0 && !loadError}
              error={loadError}
              selected5DKeys={selectedKeys}
              onToggleSelect={onToggleSelect}
              onLoad5D={onLoad5D}
              detBin={pendingDetBin}
              setDetBin={setPendingDetBin}
              gpuFreeBytes={gpuFreeBytes}
              onLoadFolder5D={onLoadFolder5D}
            />
            <EdgeCollapser
              side="left"
              onClick={() => setLeftCollapsed(true)}
              onResizeStart={startLeftRailResize}
              onResizeReset={resetLeftRailWidth}
              widthValue={leftRailWidth}
            />
          </Box>
        )}
        {activeSession && activeFile ? (
          <Box sx={{ minWidth: 0, minHeight: 0, display: "flex", flexDirection: "column", gap: 1 }}>
            <BrowseStackViewer
              session={activeSession}
              activeFile={activeFile}
              activeSet={set5D}
              onScrubIndex={scrubSessionIndex}
              onActivateStack={activateSessionStack}
            />
            <Viewer
              session={activeSession}
              file={activeFile}
              browseDtype={browseDtype}
              singleDetBin={singleDetBin}
              mode={mode}
              setMode={setMode}
              cmapImage={cmapImage}
              setCmapImage={setCmapImage}
              cmapDp={cmapDp}
              setCmapDp={setCmapDp}
              scanPos={scanPos}
              setScanPos={setScanPos}
              ringInner={ringInner}
              setRingInner={setRingInner}
              ringOuter={ringOuter}
              setRingOuter={setRingOuter}
              dpShape={dpShape}
              setDpShape={setDpShape}
              shapeParams={shapeParams}
              setShapeParams={setShapeParams}
              showRing
              logScale
              clipLo={clipLo}
              setClipLo={setClipLo}
              clipHi={clipHi}
              setClipHi={setClipHi}
              dpClipLo={dpClipLo}
              setDpClipLo={setDpClipLo}
              dpClipHi={dpClipHi}
              setDpClipHi={setDpClipHi}
              compact={effectiveCompact}
              set5D={set5D}
              onSetActiveIdx={setActiveIdx}
              realRoi={realRoi}
              setRealRoi={setRealRoi}
              fftOn={fftOn}
              setFftOn={setFftOn}
              fftWindow={fftWindow}
              setFftWindow={setFftWindow}
              imageScale={imageScale}
              setImageScale={setImageScale}
              dpScale={dpScale}
              setDpScale={setDpScale}
              imagePowerExp={imagePowerExp}
              setImagePowerExp={setImagePowerExp}
              dpPowerExp={dpPowerExp}
              setDpPowerExp={setDpPowerExp}
              maskDC={maskDC}
              setMaskDC={setMaskDC}
              profileLine={profileLine}
              setProfileLine={setProfileLine}
              profileWidth={profileWidth}
              setProfileWidth={setProfileWidth}
            />
          </Box>
        ) : (
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center",
                     border: `1px solid ${colors.border.default}`, borderRadius: radii.lg,
                     bgcolor: colors.bg.page, minHeight: 360,
                     fontSize: fontSizes.sm, color: colors.text.muted }}>
            {loadError
              ? `failed to load sessions: ${loadError}`
              : sessions.length === 0
                ? "📂 Choose a folder (top right) to open your 4D-STEM datasets"
                : "select a master file from the tree"}
          </Box>
        )}
        {rightCollapsed ? (
          <CollapsedRail side="right" label="metadata" icon="ⓘ"
                         onClick={() => setRightCollapsed(false)} />
        ) : activeSession && activeFile ? (
          <Box sx={{ position: "relative", display: "flex", flexDirection: "column", minHeight: 0 }}>
            <MetaRail session={activeSession} file={activeFile}
              browseDtype={browseDtype} onToggleDtype={toggleBrowseDtype} />
            <EdgeCollapser side="right" onClick={() => setRightCollapsed(true)} />
          </Box>
        ) : (
          <Box />
        )}
      </Box>

      {/* Hint footer */}
      <Box sx={{ display: { xs: "none", sm: "flex" }, alignItems: "center", justifyContent: "space-between",
                 fontSize: fontSizes.xs, color: colors.text.muted, mt: 0.5, gap: 1, flexWrap: "wrap" }}>
        <span>{totalFiles} files across {sessions.length} sessions</span>
        <span>
          <Kbd>drag</Kbd> scan crosshair&nbsp;
          <Kbd>↑</Kbd><Kbd>↓</Kbd><Kbd>←</Kbd><Kbd>→</Kbd> step (<Kbd>Shift</Kbd> ×10)&nbsp;
          <Kbd>R</Kbd> reset zoom&nbsp;
          <Kbd>scroll</Kbd> zoom&nbsp;
          <Kbd>?</Kbd> shortcuts
        </span>
      </Box>
    </Box>
  );
}

function ToolbarButton({
  active, onClick, title, children,
}: {
  active?: boolean;
  onClick: () => void;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <Box
      component="button"
      onClick={onClick}
      title={title}
      sx={{
        fontSize: fontSizes.xs,
        fontWeight: 600,
        px: 0.75,
        py: 0.5,
        cursor: "pointer",
        [`@media (max-width: ${breakpoints.tablet}px)`]: {
          minWidth: 44,
          minHeight: 44,
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
        },
        border: `1px solid ${active ? colors.interactive.border : colors.border.default}`,
        bgcolor: active ? colors.interactive.bg : colors.bg.page,
        color: active ? colors.interactive.selectedText : colors.text.secondary,
        borderRadius: radii.sm,
        "&:hover": { bgcolor: active ? colors.interactive.bg : colors.bg.hover },
      }}
    >
      {children}
    </Box>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <Box component="kbd"
         sx={{ display: "inline-block", px: 0.5, mx: 0.25, fontSize: fontSizes.xs,
               fontFamily: "ui-monospace, Menlo, monospace",
               border: `1px solid ${colors.border.default}`, borderRadius: radii.sm,
               bgcolor: colors.bg.subtle, color: colors.text.secondary }}>
      {children}
    </Box>
  );
}
