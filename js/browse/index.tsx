/**
 * Browse - File/folder browser widget for microscope data workflows.
 *
 * Features:
 * - Directory navigation with breadcrumb
 * - File/folder selection (single and multi-select)
 * - Extension filtering
 * - Sort by name/size/modified
 * - Image preview panel for .tif/.png/.emd files
 * - Keyboard navigation (ArrowUp/Down, Enter, Space, Ctrl+A)
 * - Name search filter
 * - File type icons (colored by extension)
 * - Resizable file list
 * - Light/dark theme support
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import { useTheme, type ThemeColors } from "../theme";
import { extractFloat32 } from "../format";
import { findDataRange, sliderRange } from "../stats";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen } from "../colormaps";
import { computeToolVisibility } from "../tool-parity";
import { ControlCustomizer } from "../control-customizer";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const DEFAULT_LIST_HEIGHT = 400;
const MIN_LIST_HEIGHT = 150;
const MAX_LIST_HEIGHT = 800;
const PREVIEW_CANVAS_W = 200;
const DPR = window.devicePixelRatio || 1;

const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  section: { fontSize: 11, fontWeight: "bold" as const },
};

const controlRow: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: SPACING.SM,
  flexWrap: "wrap",
  padding: `${SPACING.XS}px ${SPACING.SM}px`,
};

const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatDate(ts: number): string {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d ago`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

// File icon: returns unicode emoji for compact rendering (no MUI icon imports needed)
function getFileIcon(ext: string, isFolder: boolean): { char: string; color: string } {
  if (isFolder) return { char: "📁", color: "#ffa726" };
  switch (ext) {
    case ".h5":
    case ".hdf5":
      return { char: "🔬", color: "#ab47bc" };
    case ".npy":
    case ".npz":
      return { char: "📊", color: "#42a5f5" };
    case ".tif":
    case ".tiff":
    case ".png":
    case ".jpg":
    case ".jpeg":
    case ".bmp":
    case ".emd":
      return { char: "🖼", color: "#66bb6a" };
    case ".txt":
    case ".csv":
    case ".log":
    case ".md":
      return { char: "📄", color: "" };
    case ".py":
    case ".json":
    case ".yaml":
    case ".yml":
      return { char: "📝", color: "#ffb74d" };
    case ".dm3":
    case ".dm4":
      return { char: "🔬", color: "#66bb6a" };
    default:
      return { char: "📄", color: "" };
  }
}

// ---------------------------------------------------------------------------
// InfoTooltip & KeyboardShortcuts (matching Show3D)
// ---------------------------------------------------------------------------
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
        &#9432;
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

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface FileEntry {
  name: string;
  path: string;
  kind: "file" | "folder";
  size: number;
  size_str: string;
  ext: string;
  modified: number;
  is_previewable: boolean;
}

// ---------------------------------------------------------------------------
// Preview component
// ---------------------------------------------------------------------------
interface PreviewPanelProps {
  previewBytes: DataView | null;
  previewWidth: number;
  previewHeight: number;
  previewTitle: string;
  previewInfo: string;
  previewError: string;
  previewCmap: string;
  colors: ThemeColors;
}

function PreviewPanel({ previewBytes, previewWidth, previewHeight, previewTitle, previewInfo, previewError, previewCmap, colors }: PreviewPanelProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !previewBytes || previewWidth === 0 || previewHeight === 0) return;
    const data = extractFloat32(previewBytes);
    if (!data || data.length === 0) return;
    if (data.length !== previewWidth * previewHeight) return;
    const { min: dMin, max: dMax } = findDataRange(data);
    const { vmin, vmax } = sliderRange(dMin, dMax, 0, 100);
    const lut = COLORMAPS[previewCmap] || COLORMAPS["inferno"];
    const offscreen = renderToOffscreen(data, previewWidth, previewHeight, lut, vmin, vmax);
    const cssW = PREVIEW_CANVAS_W;
    const aspect = previewHeight / previewWidth;
    const cssH = Math.round(cssW * aspect);
    canvas.width = cssW * DPR;
    canvas.height = cssH * DPR;
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    const ctx = canvas.getContext("2d")!;
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
  }, [previewBytes, previewWidth, previewHeight, previewCmap]);

  if (previewError) {
    return (
      <Box sx={{ width: PREVIEW_CANVAS_W, p: 1 }}>
        <Typography sx={{ ...typography.labelSmall, color: "#f44" }}>{previewError}</Typography>
      </Box>
    );
  }

  if (!previewBytes || previewWidth === 0) {
    return (
      <Box sx={{ width: PREVIEW_CANVAS_W, p: 1, display: "flex", alignItems: "center", justifyContent: "center", height: 80 }}>
        <Typography sx={{ ...typography.labelSmall, color: colors.textMuted }}>Click a file to preview</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: PREVIEW_CANVAS_W, p: 1 }}>
      <canvas ref={canvasRef} style={{ display: "block", border: `1px solid ${colors.border}` }} />
      {previewTitle && (
        <Typography sx={{ ...typography.labelSmall, color: colors.text, mt: 0.5, wordBreak: "break-all" }}>
          {previewTitle}
        </Typography>
      )}
      {previewInfo && (
        <Typography sx={{ ...typography.value, color: colors.textMuted }}>
          {previewInfo}
        </Typography>
      )}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// File row component (plain div, no MUI List)
// ---------------------------------------------------------------------------
interface FileRowProps {
  entry: FileEntry;
  isSelected: boolean;
  isFocused: boolean;
  colors: ThemeColors;
  selectedBg: string;
  selectedHoverBg: string;
  focusBorderColor: string;
  onClick: (e: React.MouseEvent) => void;
  onDoubleClick: () => void;
  onCheckbox: (e: React.MouseEvent) => void;
}

const FileRow = React.forwardRef<HTMLDivElement, FileRowProps>(
  ({ entry, isSelected, isFocused, colors, selectedBg, selectedHoverBg, focusBorderColor, onClick, onDoubleClick, onCheckbox }, ref) => {
    const isFolder = entry.kind === "folder";
    const icon = getFileIcon(entry.ext, isFolder);
    return (
      <div
        ref={ref}
        onClick={onClick}
        onDoubleClick={onDoubleClick}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 4,
          padding: "1px 6px",
          cursor: "pointer",
          fontSize: 10,
          fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
          backgroundColor: isSelected ? selectedBg : "transparent",
          outline: isFocused ? `1px solid ${focusBorderColor}` : "none",
          outlineOffset: -1,
          userSelect: "none",
        }}
        onMouseEnter={(e) => { if (isSelected) e.currentTarget.style.backgroundColor = selectedHoverBg; }}
        onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = isSelected ? selectedBg : "transparent"; }}
      >
        {/* Checkbox */}
        <input
          type="checkbox"
          checked={isSelected}
          onClick={onCheckbox}
          onChange={() => {}}
          style={{ width: 12, height: 12, margin: 0, cursor: "pointer", accentColor: colors.accent, flexShrink: 0 }}
        />
        {/* Icon */}
        <span style={{ fontSize: 12, lineHeight: 1, flexShrink: 0, width: 16, textAlign: "center" }}>{icon.char}</span>
        {/* Filename */}
        <span style={{
          flex: 1,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          color: isFolder ? colors.accent : colors.text,
          fontWeight: isFolder ? 600 : 400,
        }}>
          {entry.name}
        </span>
        {/* Size */}
        {!isFolder && entry.size_str && (
          <span style={{ color: colors.textMuted, flexShrink: 0, minWidth: 50, textAlign: "right", fontFamily: "monospace", fontSize: 9 }}>
            {entry.size_str}
          </span>
        )}
        {/* Date */}
        {entry.modified > 0 && (
          <span style={{ color: colors.textMuted, opacity: 0.7, flexShrink: 0, minWidth: 50, textAlign: "right", fontFamily: "monospace", fontSize: 9 }}>
            {formatDate(entry.modified)}
          </span>
        )}
      </div>
    );
  }
);

// ---------------------------------------------------------------------------
// Main widget
// ---------------------------------------------------------------------------
function BrowseWidget() {
  const { themeInfo, colors } = useTheme();
  const [title] = useModelState<string>("title");
  const [root] = useModelState<string>("root");
  const [currentPath, setCurrentPath] = useModelState<string>("current_path");
  const [entriesJson] = useModelState<string>("entries_json");
  const [selected, setSelected] = useModelState<string[]>("selected");
  const [selectionMode] = useModelState<string>("selection_mode");
  const [filterExts, setFilterExts] = useModelState<string[]>("filter_exts");
  const [sortKey, setSortKey] = useModelState<string>("sort_key");
  const [sortAsc, setSortAsc] = useModelState<boolean>("sort_asc");
  const [showHidden, setShowHidden] = useModelState<boolean>("show_hidden");
  const [showControls] = useModelState<boolean>("show_controls");
  const [, setPreviewRequest] = useModelState<string>("preview_request");
  const [previewBytes] = useModelState<DataView | null>("preview_bytes");
  const [previewWidth] = useModelState<number>("preview_width");
  const [previewHeight] = useModelState<number>("preview_height");
  const [previewTitle] = useModelState<string>("preview_title");
  const [previewInfo] = useModelState<string>("preview_info");
  const [previewError] = useModelState<string>("preview_error");
  const [availableExts] = useModelState<string[]>("available_exts");
  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");
  const [previewCmap, setPreviewCmap] = useModelState<string>("preview_cmap");
  const [isScanning] = useModelState<boolean>("is_scanning");

  const entries: FileEntry[] = React.useMemo(() => {
    try { return JSON.parse(entriesJson); } catch { return []; }
  }, [entriesJson]);

  const vis = React.useMemo(
    () => computeToolVisibility("Browse", disabledTools, hiddenTools),
    [disabledTools, hiddenTools]
  );

  const navLocked = vis.isLocked("navigation");
  const previewHidden = vis.isHidden("preview");
  const filterHidden = vis.isHidden("filter");

  // Search state
  const [searchQuery, setSearchQuery] = React.useState("");
  const searchRef = React.useRef<HTMLInputElement>(null);

  // Keyboard focus state
  const [focusedIdx, setFocusedIdx] = React.useState(-1);
  const listRef = React.useRef<HTMLDivElement>(null);
  const itemRefs = React.useRef<Map<number, HTMLDivElement>>(new Map());

  // Resize state
  const [listHeight, setListHeight] = React.useState(DEFAULT_LIST_HEIGHT);
  const resizeDragRef = React.useRef<{ startY: number; startH: number } | null>(null);

  // Last click index for Shift+Click range selection
  const lastClickIdxRef = React.useRef<number>(-1);

  // Reset search and focus on directory change
  React.useEffect(() => {
    setSearchQuery("");
    setFocusedIdx(-1);
    lastClickIdxRef.current = -1;
  }, [currentPath]);

  // Filter entries by search query
  const filteredEntries = React.useMemo(() => {
    if (!searchQuery) return entries;
    const q = searchQuery.toLowerCase();
    return entries.filter(e => e.name.toLowerCase().includes(q));
  }, [entries, searchQuery]);

  // Reset focusedIdx when filtered entries change
  React.useEffect(() => { setFocusedIdx(-1); }, [filteredEntries.length]);

  // Scroll focused item into view
  React.useEffect(() => {
    if (focusedIdx >= 0) {
      const el = itemRefs.current.get(focusedIdx);
      if (el) el.scrollIntoView({ block: "nearest" });
    }
  }, [focusedIdx]);

  // Selection colors
  const selectedBg = themeInfo.theme === "dark" ? "rgba(90, 170, 255, 0.12)" : "rgba(0, 102, 204, 0.08)";
  const selectedHoverBg = themeInfo.theme === "dark" ? "rgba(90, 170, 255, 0.18)" : "rgba(0, 102, 204, 0.14)";
  const focusBorderColor = colors.accent;

  // Compute total selected size
  const selectedTotalSize = React.useMemo(() => {
    const selectedSet = new Set(selected);
    let total = 0;
    for (const e of entries) {
      if (selectedSet.has(e.path) && e.kind === "file") total += e.size;
    }
    return total;
  }, [selected, entries]);

  // Themed styles
  const themedSelect = {
    sx: {
      fontSize: 10,
      height: 24,
      bgcolor: colors.controlBg,
      color: colors.text,
      "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border },
      "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: colors.accent },
      "& .MuiSelect-icon": { color: colors.text },
    },
  };

  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: colors.controlBg, color: colors.text, border: `1px solid ${colors.border}` } },
  };

  const themeColors = {
    controlBg: colors.controlBg,
    text: colors.text,
    border: colors.border,
    textMuted: colors.textMuted,
    accent: colors.accent,
  };

  const nFolders = filteredEntries.filter(e => e.kind === "folder").length;
  const nFiles = filteredEntries.filter(e => e.kind === "file").length;

  // Breadcrumb segments
  const breadcrumbs = React.useMemo(() => {
    const rootPath = root.replace(/\\/g, "/");
    const curr = currentPath.replace(/\\/g, "/");
    if (!curr.startsWith(rootPath)) return [{ label: root, path: root }];
    const rel = curr.slice(rootPath.length).replace(/^\//, "");
    const segments: { label: string; path: string }[] = [{ label: rootPath.split("/").pop() || root, path: root }];
    if (rel) {
      const parts = rel.split("/");
      let accum = rootPath;
      for (const part of parts) {
        accum += "/" + part;
        segments.push({ label: part, path: accum });
      }
    }
    return segments;
  }, [root, currentPath]);

  const handleNavigate = React.useCallback((path: string) => {
    if (navLocked) return;
    setCurrentPath(path);
    setPreviewRequest("");
  }, [setCurrentPath, setPreviewRequest, navLocked]);

  const handleGoUp = React.useCallback(() => {
    if (navLocked) return;
    const curr = currentPath.replace(/\\/g, "/");
    const rootNorm = root.replace(/\\/g, "/");
    if (curr === rootNorm) return;
    const parent = curr.split("/").slice(0, -1).join("/") || rootNorm;
    if (!parent.startsWith(rootNorm)) return;
    setCurrentPath(parent);
    setPreviewRequest("");
  }, [currentPath, root, setCurrentPath, setPreviewRequest, navLocked]);

  // macOS Finder-style click handling:
  // - Click = focus + preview (no selection change)
  // - Cmd+Click = toggle individual selection
  // - Shift+Click = range selection from last click
  // - Checkbox = toggle individual selection
  const handleRowClick = React.useCallback((idx: number, entry: FileEntry, e: React.MouseEvent) => {
    setFocusedIdx(idx);
    // Preview on any click (if previewable)
    if (entry.kind === "file" && entry.is_previewable && !previewHidden) {
      setPreviewRequest(entry.path);
    }
    if (e.metaKey || e.ctrlKey) {
      // Cmd+Click: toggle selection
      const existing = selected.indexOf(entry.path);
      if (existing >= 0) {
        setSelected(selected.filter((_, i) => i !== existing));
      } else {
        setSelected([...selected, entry.path]);
      }
      lastClickIdxRef.current = idx;
    } else if (e.shiftKey && lastClickIdxRef.current >= 0) {
      // Shift+Click: range selection
      const start = Math.min(lastClickIdxRef.current, idx);
      const end = Math.max(lastClickIdxRef.current, idx);
      const rangePaths = filteredEntries.slice(start, end + 1).map(e => e.path);
      const newSelected = new Set(selected);
      for (const p of rangePaths) newSelected.add(p);
      setSelected(Array.from(newSelected));
    } else {
      // Plain click: just focus, no selection change
      lastClickIdxRef.current = idx;
    }
  }, [selected, setSelected, filteredEntries, setPreviewRequest, previewHidden]);

  // Checkbox always toggles selection (regardless of modifier keys)
  const handleCheckboxClick = React.useCallback((idx: number, entry: FileEntry, e: React.MouseEvent) => {
    e.stopPropagation();
    const existing = selected.indexOf(entry.path);
    if (existing >= 0) {
      setSelected(selected.filter((_, i) => i !== existing));
    } else {
      setSelected([...selected, entry.path]);
    }
    lastClickIdxRef.current = idx;
  }, [selected, setSelected]);

  const handleDoubleClick = React.useCallback((entry: FileEntry) => {
    if (entry.kind === "folder") handleNavigate(entry.path);
  }, [handleNavigate]);

  const handleSelectAll = React.useCallback(() => {
    const allPaths = filteredEntries.map(e => e.path);
    const allSelected = allPaths.length > 0 && allPaths.every(p => selected.includes(p));
    if (allSelected) {
      setSelected([]);
      setPreviewRequest("");
    } else {
      setSelected(allPaths);
    }
  }, [filteredEntries, selected, setSelected, setPreviewRequest]);

  const handleKeyDown = React.useCallback((e: React.KeyboardEvent) => {
    const tag = (e.target as HTMLElement).tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setFocusedIdx(prev => Math.min(prev + 1, filteredEntries.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setFocusedIdx(prev => Math.max(prev - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (focusedIdx >= 0 && focusedIdx < filteredEntries.length) {
        const entry = filteredEntries[focusedIdx];
        if (entry.kind === "folder") handleNavigate(entry.path);
        else {
          // Toggle selection on Enter
          const existing = selected.indexOf(entry.path);
          if (existing >= 0) setSelected(selected.filter((_, i) => i !== existing));
          else setSelected([...selected, entry.path]);
          lastClickIdxRef.current = focusedIdx;
        }
      }
    } else if (e.key === " ") {
      e.preventDefault();
      if (focusedIdx >= 0 && focusedIdx < filteredEntries.length) {
        const entry = filteredEntries[focusedIdx];
        const existing = selected.indexOf(entry.path);
        if (existing >= 0) setSelected(selected.filter((_, i) => i !== existing));
        else setSelected([...selected, entry.path]);
        lastClickIdxRef.current = focusedIdx;
      }
    } else if (e.key === "Backspace") {
      e.preventDefault();
      handleGoUp();
    } else if (e.key === "Escape") {
      if (searchQuery) setSearchQuery("");
      else { setSelected([]); setPreviewRequest(""); }
    } else if (e.key === "a" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSelectAll();
    } else if (e.key === "/" || (e.key === "f" && (e.metaKey || e.ctrlKey))) {
      e.preventDefault();
      searchRef.current?.focus();
    }
  }, [filteredEntries, focusedIdx, handleGoUp, handleNavigate, handleSelectAll, searchQuery, selected, setSelected, setPreviewRequest]);

  // Resize handlers
  const handleResizeMouseDown = React.useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    resizeDragRef.current = { startY: e.clientY, startH: listHeight };
    const handleMouseMove = (ev: MouseEvent) => {
      if (!resizeDragRef.current) return;
      const delta = ev.clientY - resizeDragRef.current.startY;
      const newH = Math.max(MIN_LIST_HEIGHT, Math.min(MAX_LIST_HEIGHT, resizeDragRef.current.startH + delta));
      setListHeight(newH);
    };
    const handleMouseUp = () => {
      resizeDragRef.current = null;
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
  }, [listHeight]);

  return (
    <Box
      className="browse-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      sx={{
        bgcolor: colors.bg,
        color: colors.text,
        border: `1px solid ${colors.border}`,
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        outline: "none",
        maxWidth: 700,
      }}
    >
      {/* Header row — matching Show4DSTEM pattern */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, px: 1, py: 0.25, borderBottom: `1px solid ${colors.border}` }}>
        <Typography sx={{ ...typography.section, color: colors.text, flexShrink: 0 }}>{title}</Typography>
        <InfoTooltip
          theme={themeInfo.theme}
          text={
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Keyboard shortcuts</Typography>
              <KeyboardShortcuts items={[
                ["\u2191 / \u2193", "Navigate files"],
                ["Enter", "Open folder / Toggle select"],
                ["Space", "Toggle select"],
                ["Backspace", "Go up one level"],
                ["Escape", "Clear search / selection"],
                ["\u2318/Ctrl+A", "Select all"],
                ["/", "Focus search"],
              ]} />
            </Box>
          }
        />
        <Tooltip title="Go to root">
          <IconButton size="small" onClick={() => handleNavigate(root)} sx={{ color: colors.text, p: 0.25 }}>
            <span style={{ fontSize: 12 }}>🏠</span>
          </IconButton>
        </Tooltip>
        <Tooltip title="Go up">
          <IconButton size="small" onClick={handleGoUp} sx={{ color: colors.text, p: 0.25 }}>
            <span style={{ fontSize: 12 }}>⬆</span>
          </IconButton>
        </Tooltip>
        {/* Breadcrumb */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.25, overflow: "hidden", flex: 1 }}>
          {breadcrumbs.map((seg, i) => (
            <React.Fragment key={seg.path}>
              {i > 0 && <span style={{ fontSize: 10, color: colors.textMuted }}>/</span>}
              <span
                onClick={() => handleNavigate(seg.path)}
                style={{
                  fontSize: 10,
                  color: i === breadcrumbs.length - 1 ? colors.text : colors.accent,
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                }}
              >
                {seg.label}
              </span>
            </React.Fragment>
          ))}
        </Box>
        {/* Search */}
        <input
          ref={searchRef}
          type="text"
          placeholder="Search..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => {
            e.stopPropagation();
            if (e.key === "Escape") {
              if (searchQuery) setSearchQuery("");
              else (e.target as HTMLInputElement).blur();
            }
          }}
          style={{
            fontSize: 10,
            width: 100,
            padding: "1px 4px",
            border: `1px solid ${colors.border}`,
            background: colors.controlBg,
            color: colors.text,
            outline: "none",
            fontFamily: "inherit",
          }}
        />
        <ControlCustomizer
          widgetName="Browse"
          hiddenTools={hiddenTools}
          setHiddenTools={setHiddenTools}
          disabledTools={disabledTools}
          setDisabledTools={setDisabledTools}
          themeColors={themeColors}
        />
      </Box>

      {/* Main content: file list + optional preview */}
      <Box sx={{ display: "flex" }}>
        {/* File list */}
        <div
          ref={listRef}
          style={{
            flex: 1,
            minWidth: 0,
            height: listHeight,
            overflow: "auto",
          }}
        >
          {isScanning && (
            <div style={{ padding: 8, textAlign: "center", fontSize: 10, color: colors.accent }}>Scanning...</div>
          )}
          {/* Go up entry */}
          {currentPath !== root && !isScanning && (
            <div
              onClick={handleGoUp}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 4,
                padding: "1px 6px",
                cursor: "pointer",
                fontSize: 10,
                fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                color: colors.accent,
              }}
            >
              <span style={{ width: 12, flexShrink: 0 }} />
              <span style={{ fontSize: 12, width: 16, textAlign: "center" }}>📁</span>
              <span style={{ fontWeight: 600 }}>..</span>
            </div>
          )}
          {!isScanning && filteredEntries.map((entry, idx) => {
            const isSelected = selected.includes(entry.path);
            const isFocused = idx === focusedIdx;
            return (
              <FileRow
                key={entry.path}
                ref={(el: HTMLDivElement | null) => { if (el) itemRefs.current.set(idx, el); else itemRefs.current.delete(idx); }}
                entry={entry}
                isSelected={isSelected}
                isFocused={isFocused}
                colors={colors}
                selectedBg={selectedBg}
                selectedHoverBg={selectedHoverBg}
                focusBorderColor={focusBorderColor}
                onClick={(e) => handleRowClick(idx, entry, e)}
                onDoubleClick={() => handleDoubleClick(entry)}
                onCheckbox={(e) => handleCheckboxClick(idx, entry, e)}
              />
            );
          })}
          {!isScanning && filteredEntries.length === 0 && (
            <div style={{ padding: 16, textAlign: "center", fontSize: 10, color: colors.textMuted }}>
              {searchQuery ? "No matches" : "Empty directory"}
            </div>
          )}
        </div>

        {/* Preview panel */}
        {!previewHidden && (
          <Box sx={{ borderLeft: `1px solid ${colors.border}`, flexShrink: 0 }}>
            <PreviewPanel
              previewBytes={previewBytes}
              previewWidth={previewWidth}
              previewHeight={previewHeight}
              previewTitle={previewTitle}
              previewInfo={previewInfo}
              previewError={previewError}
              previewCmap={previewCmap}
              colors={colors}
            />
            {previewBytes && previewWidth > 0 && (
              <Box sx={{ px: 1, pb: 0.5 }}>
                <Select
                  value={previewCmap}
                  onChange={(e) => setPreviewCmap(e.target.value)}
                  size="small"
                  {...themedSelect}
                  sx={{ ...themedSelect.sx, width: "100%", fontSize: 10 }}
                  MenuProps={themedMenuProps}
                >
                  {COLORMAP_NAMES.map((name) => (
                    <MenuItem key={name} value={name} sx={{ fontSize: 10 }}>{name}</MenuItem>
                  ))}
                </Select>
              </Box>
            )}
          </Box>
        )}
      </Box>

      {/* Resize handle */}
      <div
        onMouseDown={handleResizeMouseDown}
        style={{
          height: 3,
          cursor: "ns-resize",
          borderTop: `1px solid ${colors.border}`,
        }}
      />

      {/* Controls row — tight, matching Show4DSTEM */}
      {showControls && !filterHidden && (
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: "2px 8px",
          borderTop: `1px solid ${colors.border}`,
          backgroundColor: colors.bgAlt,
          flexWrap: "wrap",
        }}>
          <span style={{ fontSize: 10, color: colors.text }}>Sort:</span>
          <Select
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value)}
            size="small"
            {...themedSelect}
            MenuProps={themedMenuProps}
          >
            <MenuItem value="name" sx={{ fontSize: 10 }}>Name</MenuItem>
            <MenuItem value="size" sx={{ fontSize: 10 }}>Size</MenuItem>
            <MenuItem value="modified" sx={{ fontSize: 10 }}>Modified</MenuItem>
          </Select>
          <Button
            size="small"
            onClick={() => setSortAsc(!sortAsc)}
            sx={{ minWidth: 0, px: 0.5, py: 0, fontSize: 10, color: colors.text }}
          >
            {sortAsc ? "\u25B2" : "\u25BC"}
          </Button>

          <span style={{ fontSize: 10, color: colors.text }}>Hidden:</span>
          <Switch
            checked={showHidden}
            onChange={(e) => setShowHidden(e.target.checked)}
            size="small"
            sx={switchStyles.small}
          />

          {/* Active filter chips — inline, compact */}
          {filterExts.length > 0 && filterExts.map((ext) => (
            <span
              key={ext}
              style={{
                fontSize: 9,
                padding: "0px 4px",
                border: `1px solid ${colors.border}`,
                color: colors.text,
                display: "inline-flex",
                alignItems: "center",
                gap: 2,
              }}
            >
              {ext}
              <span
                onClick={() => setFilterExts(filterExts.filter(e => e !== ext))}
                style={{ cursor: "pointer", fontSize: 10, color: colors.textMuted, lineHeight: 1 }}
              >
                ✕
              </span>
            </span>
          ))}

          {/* Add filter dropdown */}
          {availableExts.length > 0 && (
            <Select
              value=""
              displayEmpty
              onChange={(e) => {
                const ext = e.target.value;
                if (ext && !filterExts.includes(ext)) setFilterExts([...filterExts, ext]);
              }}
              size="small"
              renderValue={() => "+Filter"}
              {...themedSelect}
              sx={{ ...themedSelect.sx, minWidth: 60 }}
              MenuProps={themedMenuProps}
            >
              {availableExts.map((ext) => (
                <MenuItem key={ext} value={ext} sx={{ fontSize: 10 }}>
                  {ext} {filterExts.includes(ext) ? "\u2713" : ""}
                </MenuItem>
              ))}
            </Select>
          )}

          <span style={{ flex: 1 }} />
          <Button
            size="small"
            onClick={handleSelectAll}
            sx={{ minWidth: 0, px: 0.5, py: 0, fontSize: 10, color: colors.text }}
          >
            {filteredEntries.length > 0 && filteredEntries.every(e => selected.includes(e.path)) ? "DESELECT" : "SELECT ALL"}
          </Button>
          {selected.length > 0 && (
            <Button
              size="small"
              onClick={() => { setSelected([]); setPreviewRequest(""); }}
              sx={{ minWidth: 0, px: 0.5, py: 0, fontSize: 10, color: colors.text }}
            >
              CLEAR
            </Button>
          )}
        </div>
      )}

      {/* Status bar — tight */}
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", px: 1, py: 0.25, borderTop: `1px solid ${colors.border}` }}>
        <Typography sx={{ ...typography.value, color: colors.textMuted }}>
          {isScanning ? "Scanning..." : `${nFolders} folders, ${nFiles} files`}
        </Typography>
        <Typography sx={{ ...typography.value, color: selected.length > 0 ? colors.accent : colors.text }}>
          {selected.length > 0
            ? `Selected: ${selected.length} item${selected.length === 1 ? "" : "s"}${selectedTotalSize > 0 ? ` (${formatBytes(selectedTotalSize)})` : ""}`
            : `Mode: ${selectionMode}`}
        </Typography>
      </Box>
    </Box>
  );
}

export const render = createRender(BrowseWidget);
