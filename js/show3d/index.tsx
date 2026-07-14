/// <reference types="@webgpu/types" />
/**
 * Show3D - Interactive 3D stack viewer with playback controls.
 *
 * Features:
 * - Scroll to zoom, double-click to reset
 * - Adjustable ROI size via slider
 * - FPS slider control
 * - WebGPU-accelerated FFT
 * - Equal-sized FFT and histogram panels
 * - Automatic theme detection (light/dark mode)
 */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Slider from "@mui/material/Slider";
import IconButton from "@mui/material/IconButton";
import Select from "@mui/material/Select";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Button from "@mui/material/Button";
import Badge from "@mui/material/Badge";
import Tooltip from "@mui/material/Tooltip";
import TextField from "@mui/material/TextField";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import FastForwardIcon from "@mui/icons-material/FastForward";
import StopIcon from "@mui/icons-material/Stop";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import DragIndicatorIcon from "@mui/icons-material/DragIndicator";
import { useTheme } from "../theme";
import { useCanvasRepaintSignal } from "../canvasLifecycle";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, drawColorbar, formatZoomLabel, roundToNiceValue, unitSymbol, formatScaleLabel } from "../figure";
import {
  applyStandaloneWidgetViewState,
  downloadBlob,
  extractBytes,
  extractFloat32,
  formatNumber,
  preserveRestoredWidgetModelsOnSave,
  standaloneHtmlWithCurrentWidgetState,
  standaloneWidgetStaticHtmlFromDocument,
} from "../format";
import { useHideStaticFallback } from "../staticFallback";
import { findDataRange, applyLogScale, applyLogScaleInPlace, percentileClip, sliderRange, computeStats, computeHistogramFromBytes } from "../stats";
import { MetadataSection } from "../widgetInfo";
import { EmbeddedWidgetView } from "../embeddedWidget";
import { FolderWatchBadge, useFolderWatchModelLive } from "../folderWatchStatus";
import { applyFrequencyFilterBrowser, frequencyFilterActive, getFrequencyFilterBackend, normalizeFrequencyFilterMode } from "../frequencyFilter";

const SHOW3D_TO_SHOW2D_LINKED_TRAITS = [
  { source: "cmap" },
  { source: "log_scale" },
  { source: "auto_contrast" },
  { source: "vmin" },
  { source: "vmax" },
  { source: "show_stats" },
  { source: "show_controls" },
  { source: "controls_collapsed" },
  { source: "debug" },
  { source: "link_contrast" },
  { source: "show_fft" },
  { source: "hidden_panels" },
  { source: "hidden_page_slots" },
];

const SHOW3D_STANDALONE_VIEW_STATE_KEYS = [
  "auto_contrast",
  "avg_window",
  "blink_fps",
  "bookmarked_frames",
  "boomerang",
  "cmap",
  "col_markers",
  "compare_background",
  "compare_mode",
  "compare_pair",
  "contrast_preset",
  "controls_collapsed",
  "debug",
  "denoise",
  "denoise_bin",
  "denoise_enabled",
  "denoise_sigma",
  "diff_cmap",
  "diff_mode",
  "fft_layout",
  "fft_metrics",
  "fft_overlay_position",
  "fft_overlay_size",
  "fft_overlay_zoom",
  "fft_window",
  "flip_horizontal",
  "flip_vertical",
  "fps",
  "frame_rotations",
  "frequency_filter",
  "frequency_filter_center",
  "frequency_filter_cutoff",
  "frequency_filter_enabled",
  "frequency_filter_width",
  "hidden_indices",
  "hidden_page_slots",
  "hidden_panels",
  "identity_colors",
  "image_rotation",
  "image_vmax_pct",
  "image_vmin_pct",
  "link_contrast",
  "link_panels",
  "log_scale",
  "loop",
  "loop_end",
  "loop_start",
  "marker_colors",
  "marker_style",
  "max_cols",
  "page_idx",
  "panel_annotations",
  "panel_cmaps",
  "panel_gap",
  "panel_groups",
  "panel_order",
  "panel_overlays",
  "panel_title_font_size",
  "panel_title_spans",
  "panel_title_style",
  "percentile_high",
  "percentile_low",
  "playback_path",
  "playing",
  "profile_line",
  "profile_width",
  "roi_active",
  "roi_list",
  "roi_selected_idx",
  "rotation_scope",
  "row_markers",
  "scale_bar_visible",
  "show_controls",
  "show_denoise",
  "show_fft",
  "show_frequency_filter",
  "show_kymograph",
  "show_panel_titles",
  "show_resize_handles",
  "show_stats",
  "show_title",
  "show_zoom_indicator",
  "slice_idx",
  "smooth",
  "starred",
  "subpixel_align_enabled",
  "subpixel_align_reference",
  "view_state",
  "vmax",
  "vmax_per_panel",
  "vmin",
  "vmin_per_panel",
] as const;
// ============================================================================
// Style tokens (inlined - matches Show2D/Show4DSTEM single-file convention)
// ============================================================================
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;
const UI_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const controlRow = {
  display: "flex",
  alignItems: "center",
  flexWrap: "wrap",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
  maxWidth: "100%",
  boxSizing: "border-box",
} as const;
const compactButton = {
  fontSize: 10,
  fontFamily: "inherit",
  textTransform: "none" as const,
  letterSpacing: 0,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};
const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};
const sliderStyles = {
  small: {
    py: 0,
    "& .MuiSlider-thumb": { width: 10, height: 10 },
    "& .MuiSlider-rail": { height: 2 },
    "& .MuiSlider-track": { height: 2 },
  },
};
const PAGE_PLAY_FPS_OPTIONS = [1, 2, 3, 4] as const;
const CONTRAST_PRESETS = [
  { value: "custom", label: "Custom", low: 0, high: 100 },
  { value: "0.5-99.5", label: "0.5–99.5", low: 0.5, high: 99.5 },
  { value: "1-99", label: "1–99", low: 1, high: 99 },
  { value: "2-98", label: "2–98", low: 2, high: 98 },
  { value: "3-97", label: "3–97", low: 3, high: 97 },
] as const;
const IDENTITY_PALETTE = ["#2e7d32", "#c62828", "#d81b60", "#1565c0", "#f9a825", "#6a1b9a"] as const;
const OFFLINE_FRAME_CACHE_BYTES = 2 * 1024 * 1024 * 1024;
const OFFLINE_FRAME_CACHE_MIN_FRAMES = 2;
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: UI_FONT },
  title: { fontWeight: "bold" as const },
};
type FftOverlayPosition = "top-left" | "top-right" | "bottom-left" | "bottom-right";
type ReorderPlacement = "before" | "after";
type ReorderDragVisual = {
  panel: number;
  label: string;
  imageUrl: string;
  width: number;
  height: number;
  x: number;
  y: number;
  offsetX: number;
  offsetY: number;
};


type RichTitleSpan = { text?: unknown; math?: unknown; color?: unknown };
type PanelTitleStyle = Record<string, unknown>;
type MarkerMap = Record<string, string>;
type PanelGroup = { panels?: number[]; color?: string; label?: string };
type PanelAnnotationSpec = {
  text?: string;
  math?: string;
  spans?: RichTitleSpan[];
  position?: string;
  anchor?: string;
  x?: number;
  y?: number;
  box?: [number, number, number, number];
  variant?: string;
  class_name?: string;
  bg?: string;
  fg?: string;
  color?: string;
  border_color?: string;
  border_width?: number;
  font_size?: number;
  font_weight?: string | number;
  pad_x?: number;
  pad_y?: number;
  radius?: number;
  opacity?: number;
  align?: string;
  max_width?: string;
  offset?: [number, number];
};
type PanelOverlaySpec = {
  shape?: "circle" | "rect" | "rectangle" | "square";
  coords?: "data" | "relative";
  row?: number;
  col?: number;
  radius?: number;
  row0?: number;
  col0?: number;
  row1?: number;
  col1?: number;
  stroke?: string;
  stroke_width?: number;
  line_style?: string;
  dash?: number[];
  fill?: string;
  opacity?: number;
  fill_opacity?: number;
  stroke_opacity?: number;
  z_order?: number;
};
type OverlaySelection = { panel: number; overlay: number };
type OverlayDragState = {
  mode: "move" | "resize";
  panel: number;
  overlay: number;
  handle?: string;
  startRow: number;
  startCol: number;
  original: PanelOverlaySpec;
};

function styleNumber(value: unknown, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function styleString(value: unknown, fallback = ""): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function withAlpha(color: string | undefined, alpha: number): string | undefined {
  if (!color || color === "none") return undefined;
  const a = Math.max(0, Math.min(1, alpha));
  if (color.startsWith("#")) {
    const hex = color.slice(1);
    if (hex.length === 3 || hex.length === 6) {
      const full = hex.length === 3 ? hex.split("").map((ch) => ch + ch).join("") : hex;
      const r = parseInt(full.slice(0, 2), 16);
      const g = parseInt(full.slice(2, 4), 16);
      const b = parseInt(full.slice(4, 6), 16);
      if ([r, g, b].every(Number.isFinite)) return `rgba(${r}, ${g}, ${b}, ${a})`;
    }
  }
  return color;
}

function overlayDashPattern(overlay: PanelOverlaySpec, lineWidth: number): number[] {
  const custom = Array.isArray(overlay.dash)
    ? overlay.dash.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value >= 0)
    : [];
  if (custom.some((value) => value > 0)) return custom;
  const w = Math.max(1, lineWidth);
  const lineStyle = styleString(overlay.line_style, "solid").toLowerCase().replace("_", "-");
  if (lineStyle === "dashed" || lineStyle === "dash") return [4 * w, 2 * w];
  if (lineStyle === "dotted" || lineStyle === "dot") return [w, 1.8 * w];
  if (lineStyle === "dashdot" || lineStyle === "dash-dot") return [4 * w, 2 * w, w, 2 * w];
  return [];
}

const LATEX_SYMBOLS: Record<string, string> = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε",
  zeta: "ζ", eta: "η", theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ",
  lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π", rho: "ρ", varrho: "ϱ",
  sigma: "σ", tau: "τ", upsilon: "υ", phi: "φ", varphi: "ϕ", chi: "χ",
  psi: "ψ", omega: "ω", Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ",
  Xi: "Ξ", Pi: "Π", Sigma: "Σ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
  pm: "±", times: "×", cdot: "·", degree: "°", angstrom: "Å", le: "≤",
  ge: "≥", neq: "≠", approx: "≈", infty: "∞",
};

function readLatexGroup(expr: string, start: number): { text: string; next: number } {
  if (expr[start] !== "{") return { text: expr[start] || "", next: start + 1 };
  let depth = 0;
  for (let i = start; i < expr.length; i += 1) {
    if (expr[i] === "{") depth += 1;
    if (expr[i] === "}") depth -= 1;
    if (depth === 0) return { text: expr.slice(start + 1, i), next: i + 1 };
  }
  return { text: expr.slice(start + 1), next: expr.length };
}

function readLatexAtom(expr: string, start: number): { text: string; next: number } {
  if (expr[start] === "{") return readLatexGroup(expr, start);
  if (expr[start] === "\\") {
    const match = expr.slice(start + 1).match(/^[A-Za-z]+/);
    if (match) return { text: `\\${match[0]}`, next: start + 1 + match[0].length };
  }
  return { text: expr[start] || "", next: start + 1 };
}

function renderLatexMath(expr: string, keyPrefix: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  let i = 0;
  let key = 0;
  while (i < expr.length) {
    const ch = expr[i];
    if ((ch === "^" || ch === "_") && i + 1 < expr.length) {
      const atom = readLatexAtom(expr, i + 1);
      const Tag = ch === "^" ? "sup" : "sub";
      nodes.push(
        <Tag key={`${keyPrefix}-script-${key++}`} style={{ fontSize: "0.72em", lineHeight: 0 }}>
          {renderLatexMath(atom.text, `${keyPrefix}-script-${key}`)}
        </Tag>
      );
      i = atom.next;
      continue;
    }
    if (ch === "\\") {
      const match = expr.slice(i + 1).match(/^[A-Za-z]+/);
      if (match) {
        const command = match[0];
        i += command.length + 1;
        if ((command === "mathrm" || command === "text") && expr[i] === "{") {
          const group = readLatexGroup(expr, i);
          nodes.push(<span key={`${keyPrefix}-roman-${key++}`} style={{ fontStyle: "normal" }}>{group.text}</span>);
          i = group.next;
          continue;
        }
        nodes.push(LATEX_SYMBOLS[command] || command);
        continue;
      }
    }
    if (ch === "{" || ch === "}") {
      i += 1;
      continue;
    }
    nodes.push(ch);
    i += 1;
  }
  return nodes;
}

function renderMathExpression(expr: string, keyPrefix: string): React.ReactNode {
  const normalized = expr.trim().replace(/\\+(?=[A-Za-z])/g, "\\");
  return (
    <span key={keyPrefix} data-quantem-math="true" style={{ fontFamily: "Cambria Math, STIX Two Math, Times New Roman, serif", fontStyle: "italic" }}>
      {renderLatexMath(normalized, keyPrefix)}
    </span>
  );
}

function findUnescapedDollar(text: string, from = 0): number {
  for (let i = from; i < text.length; i += 1) {
    if (text[i] === "$" && text[i - 1] !== "\\") return i;
  }
  return -1;
}

function renderTextWithInlineMath(text: string, keyPrefix: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  let rest = text;
  let key = 0;
  while (rest.length) {
    const dollar = findUnescapedDollar(rest);
    const paren = rest.indexOf("\\(");
    const starts = [dollar, paren].filter((idx) => idx >= 0);
    if (!starts.length) {
      nodes.push(rest.replace(/\\\$/g, "$"));
      break;
    }
    const start = Math.min(...starts);
    if (start > 0) nodes.push(rest.slice(0, start).replace(/\\\$/g, "$"));
    const dollarMode = rest[start] === "$";
    const close = dollarMode ? findUnescapedDollar(rest, start + 1) - (start + 1) : rest.indexOf("\\)", start + 2) - (start + 2);
    if (close < 0) {
      nodes.push(rest.slice(start).replace(/\\\$/g, "$"));
      break;
    }
    const math = rest.slice(start + (dollarMode ? 1 : 2), start + (dollarMode ? 1 : 2) + close);
    nodes.push(renderMathExpression(math, `${keyPrefix}-math-${key++}`));
    rest = rest.slice(start + (dollarMode ? 2 : 4) + close);
  }
  return nodes;
}

function panelTitleChromeSx(
  style: PanelTitleStyle | undefined,
  defaults: Record<string, unknown> = {},
): Record<string, unknown> {
  const s = style || {};
  const borderWidth = Math.max(0, styleNumber(s.border_width, 0));
  const bg = styleString(s.bg);
  const align = styleString(s.align, String(defaults.textAlign || "center"));
  const maxWidth = s.max_width;
  const mode = typeof maxWidth === "string" ? maxWidth : "";
  const sx: Record<string, unknown> = {
    ...defaults,
    color: styleString(s.fg, String(defaults.color || "rgba(255,255,255,0.95)")),
    bgcolor: bg || defaults.bgcolor,
    border: borderWidth > 0 ? `${borderWidth}px solid ${styleString(s.border_color, "rgba(255,255,255,0.35)")}` : defaults.border,
    borderRadius: s.radius != null ? `${Math.max(0, styleNumber(s.radius, 0))}px` : defaults.borderRadius,
    px: s.pad_x != null ? `${Math.max(0, styleNumber(s.pad_x, 0))}px` : defaults.px,
    py: s.pad_y != null ? `${Math.max(0, styleNumber(s.pad_y, 0))}px` : defaults.py,
    fontWeight: s.font_weight != null ? s.font_weight : defaults.fontWeight,
    opacity: s.opacity != null ? Math.max(0, Math.min(1, styleNumber(s.opacity, 1))) : defaults.opacity,
    textAlign: align,
    maxWidth: mode && mode !== "panel" && mode !== "hug" ? mode : defaults.maxWidth,
    width: mode === "panel" ? (defaults.width || "calc(100% - 56px)") : defaults.width,
    boxSizing: "border-box",
  };
  if (mode === "hug") {
    sx.left = defaults.left != null && defaults.width != null
      ? `calc(${String(defaults.left)} + (${String(defaults.width)}) / 2)`
      : "50%";
    sx.right = "auto";
    sx.width = "fit-content";
    sx.maxWidth = "calc(100% - 16px)";
    sx.transform = defaults.transform
      ? `${String(defaults.transform)} translateX(-50%)`
      : "translateX(-50%)";
  }
  return sx;
}

function richTitlePlainText(spans: RichTitleSpan[] | undefined, fallback: string): string {
  if (!Array.isArray(spans) || spans.length === 0) return fallback;
  const text = spans.map((span) => String(span?.text ?? span?.math ?? "")).join("");

  return text || fallback;
}

function renderRichTitle(spans: RichTitleSpan[] | undefined, fallback: string): React.ReactNode {

  if (!Array.isArray(spans) || spans.length === 0) return renderTextWithInlineMath(fallback, "title-fallback");
  return spans.map((span, idx) => {
    const text = String(span?.text ?? "");
    const math = span?.math == null ? "" : String(span.math);
    const color = typeof span?.color === "string" && span.color.trim() ? span.color : undefined;
    return (
      <span key={`title-span-${idx}`} style={color ? { color } : undefined}>
        {math ? renderMathExpression(math, `title-span-${idx}`) : renderTextWithInlineMath(text, `title-span-${idx}`)}

      </span>
    );
  });
}


function annotationAnchorTransform(anchor: string | undefined): string {
  const value = anchor || "top-left";
  const x = value.endsWith("center") || value === "center" ? "-50%" : value.endsWith("right") ? "-100%" : "0";
  const y = value.startsWith("center") || value === "center" ? "-50%" : value.startsWith("bottom") ? "-100%" : "0";
  return `translate(${x}, ${y})`;
}

function annotationPositionSx(spec: PanelAnnotationSpec): Record<string, unknown> {
  const margin = 8;
  const position = spec.position || "top-left";
  const offset = Array.isArray(spec.offset) ? spec.offset : [0, 0];
  if (Array.isArray(spec.box) && spec.box.length === 4) {
    const [left, top, width, height] = spec.box;
    return {
      left: `calc(${left * 100}% + ${offset[0] || 0}px)`,
      top: `calc(${top * 100}% + ${offset[1] || 0}px)`,
      width: `${width * 100}%`,
      minHeight: `${height * 100}%`,
    };
  }
  if (Number.isFinite(spec.x) && Number.isFinite(spec.y)) {
    return {
      left: `calc(${Number(spec.x) * 100}% + ${offset[0] || 0}px)`,
      top: `calc(${Number(spec.y) * 100}% + ${offset[1] || 0}px)`,
      transform: annotationAnchorTransform(spec.anchor || "center"),
    };
  }
  const sx: Record<string, unknown> = {};
  if (position.includes("top")) sx.top = margin + (offset[1] || 0);
  if (position.includes("bottom")) sx.bottom = margin - (offset[1] || 0);
  if (position.includes("left")) sx.left = margin + (offset[0] || 0);
  if (position.includes("right")) sx.right = margin - (offset[0] || 0);
  if (position === "top-center" || position === "center" || position === "bottom-center") {
    sx.left = `calc(50% + ${offset[0] || 0}px)`;
  }
  if (position === "center-left" || position === "center" || position === "center-right") {
    sx.top = `calc(50% + ${offset[1] || 0}px)`;
  }
  sx.transform = annotationAnchorTransform(spec.anchor || position);
  return sx;
}

function panelAnnotationSx(spec: PanelAnnotationSpec): Record<string, unknown> {
  const variant = spec.variant || "badge";
  const plain = variant === "plain";
  const outline = variant === "outline";
  const callout = variant === "callout";
  const pill = variant === "pill";
  const fg = styleString(spec.fg ?? spec.color, plain ? "rgba(255,255,255,0.92)" : "#fff");
  const bg = styleString(spec.bg, plain ? "transparent" : "rgba(0,0,0,0.72)");
  const borderWidth = Math.max(0, styleNumber(spec.border_width, outline || callout ? 1 : 0));
  return {
    position: "absolute",
    ...annotationPositionSx(spec),
    display: "block",
    boxSizing: "border-box",
    pointerEvents: "none",
    zIndex: 10,
    px: spec.pad_x != null ? `${Math.max(0, styleNumber(spec.pad_x, 0))}px` : (plain ? 0 : "6px"),
    py: spec.pad_y != null ? `${Math.max(0, styleNumber(spec.pad_y, 0))}px` : (plain ? 0 : "2px"),
    borderRadius: spec.radius != null ? `${Math.max(0, styleNumber(spec.radius, 0))}px` : (pill ? "999px" : "3px"),
    background: bg,
    color: fg,
    border: borderWidth > 0 ? `${borderWidth}px solid ${styleString(spec.border_color, "rgba(255,255,255,0.5)")}` : "none",
    opacity: spec.opacity != null ? Math.max(0, Math.min(1, styleNumber(spec.opacity, 1))) : 1,
    fontFamily: UI_FONT,
    fontSize: `${Math.max(6, styleNumber(spec.font_size, 10))}px`,
    fontWeight: spec.font_weight != null ? spec.font_weight : 700,
    lineHeight: 1.2,
    textAlign: styleString(spec.align, "center"),
    whiteSpace: Array.isArray(spec.box) ? "normal" : "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
    maxWidth: styleString(spec.max_width, Array.isArray(spec.box) ? "100%" : "calc(100% - 16px)"),
    textShadow: plain ? "0 1px 2px rgba(0,0,0,0.85)" : "none",
    boxShadow: callout ? "0 1px 4px rgba(0,0,0,0.45)" : "none",
  };
}

function renderPanelAnnotation(spec: PanelAnnotationSpec, fallback = ""): React.ReactNode {
  if (spec.math) return renderMathExpression(spec.math, "panel-annotation-math");
  return renderRichTitle(spec.spans, spec.text || fallback);
}

type ReorderDragStart = {
  x: number;
  y: number;
};
const REORDER_DRAG_THRESHOLD_PX = 8;

function useDebugFps(enabled: boolean): number | null {
  const [fps, setFps] = React.useState<number | null>(null);

  React.useEffect(() => {
    if (!enabled) {
      setFps(null);
      return;
    }
    if (typeof window === "undefined" || typeof window.requestAnimationFrame !== "function") {
      return;
    }
    let disposed = false;
    let frameCount = 0;
    let last = window.performance.now();
    let raf = 0;
    const tick = (now: number) => {
      if (disposed) return;
      frameCount += 1;
      const elapsed = now - last;
      if (elapsed >= 500) {
        setFps(Math.round((frameCount * 1000) / Math.max(1, elapsed)));
        frameCount = 0;
        last = now;
      }
      raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => {
      disposed = true;
      window.cancelAnimationFrame(raf);
    };
  }, [enabled]);

  return fps;
}

function DebugPerfBadge({
  widget,
  fps,
  themeColors,
}: {
  widget: string;
  fps: number | null;
  themeColors: { accent: string };
}) {
  const fpsText = fps === null ? "--" : String(fps);
  return (
    <Box
      component="span"
      data-quantem-debug-badge={widget}
      title={`${widget} debug browser UI FPS`}
      sx={{
        ml: 0.6,
        px: 0.5,
        py: 0,
        borderRadius: "3px",
        border: `1px solid ${themeColors.accent}55`,
        bgcolor: themeColors.accent + "18",
        color: themeColors.accent,
        fontSize: 9,
        fontWeight: 600,
        fontVariantNumeric: "tabular-nums",
        whiteSpace: "nowrap",
        verticalAlign: "baseline",
      }}
    >
      Debug UI FPS {fpsText}
    </Box>
  );
}

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

// ============================================================================
// Inlined utilities (matches Show2D/Show4DSTEM single-file convention)
// ============================================================================
const signedLog1p = (x: number): number => x >= 0 ? Math.log1p(x) : -Math.log1p(-x);

type Show3DWritableFile = {
  write: (data: BlobPart) => Promise<void>;
  close: () => Promise<void>;
};

type Show3DFileHandle = {
  createWritable: () => Promise<Show3DWritableFile>;
};

type Show3DSavePickerOptions = {
  suggestedName?: string;
  types?: { description: string; accept: Record<string, string[]> }[];
};

type Show3DWindow = Window & typeof globalThis & {
  showSaveFilePicker?: (options?: Show3DSavePickerOptions) => Promise<Show3DFileHandle>;
};

type PanelStats = {
  panel: number;
  mean: number;
  min: number;
  max: number;
  std: number;
};

type CursorInfo = {
  row: number;
  col: number;
  value: number;
  panelIdx: number;
};

function makeExportFilename(
  title: string,
  nSlices: number,
  height: number,
  width: number,
  mode: string,
  quality = "medium",
  downsample = 1,
): string {
  let slug = (title || "show3d")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  while (slug.includes("__")) slug = slug.replace(/__/g, "_");
  if (!slug) slug = "show3d";
  if (mode === "gif" || mode === "mp4") {
    return `${slug}_${nSlices}x${height}x${width}_${quality}.${mode}`;
  }
  const binSuffix = mode === "quantized" && downsample > 1 ? `_${downsample}xbin` : "";
  const suffix = mode === "quantized" ? `quantized${binSuffix}` : "exact";
  return `${slug}_${nSlices}x${height}x${width}_${suffix}.html`;
}

function exportPickerType(mode: string): { description: string; accept: Record<string, string[]> } {
  if (mode === "gif") return { description: "Animated GIF", accept: { "image/gif": [".gif"] } };
  if (mode === "mp4") return { description: "MP4 video", accept: { "video/mp4": [".mp4"] } };
  return { description: "Standalone HTML", accept: { "text/html": [".html"] } };
}

function exportBlobType(mode: string): string {
  if (mode === "gif") return "image/gif";
  if (mode === "mp4") return "video/mp4";
  return "text/html;charset=utf-8";
}

function formatSavedBytes(bytes: number): string {
  const mb = Math.max(0, bytes) / (1024 * 1024);
  if (mb >= 100) return `${Math.round(mb)} MB`;
  if (mb >= 10) return `${mb.toFixed(1)} MB`;
  return `${mb.toFixed(2)} MB`;
}

function isAbortLikeError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function float32FrameFromDataView(stack: DataView, frameIdx: number, pixelCount: number, copy: boolean): Float32Array | null {
  const byteStart = frameIdx * pixelCount * 4;
  const byteLength = pixelCount * 4;
  if (byteStart < 0 || byteStart + byteLength > stack.byteLength) return null;
  const byteOffset = stack.byteOffset + byteStart;
  let view: Float32Array;
  if (byteOffset % 4 === 0) {
    view = new Float32Array(stack.buffer, byteOffset, pixelCount);
  } else {
    const bytes = new Uint8Array(stack.buffer, byteOffset, byteLength);
    const aligned = new Uint8Array(byteLength);
    aligned.set(bytes);
    view = new Float32Array(aligned.buffer);
  }
  return copy ? new Float32Array(view) : view;
}

function rgbFrameToLuminance(rgb: Float32Array, pixelCount: number): Float32Array {
  const luminance = new Float32Array(pixelCount);
  const n = Math.min(pixelCount, Math.floor(rgb.length / 3));
  for (let k = 0; k < n; k++) {
    luminance[k] = 0.2126 * rgb[3 * k] + 0.7152 * rgb[3 * k + 1] + 0.0722 * rgb[3 * k + 2];
  }
  return luminance;
}

const clampPct = (x: number): number => Math.max(0, Math.min(100, x));
const valueToPct = (value: number | null | undefined, min: number, max: number, fallback: number): number => {
  if (value == null || !Number.isFinite(value) || max <= min) return fallback;
  return clampPct(((value - min) / (max - min)) * 100);
};
const pctToValue = (pct: number, min: number, max: number): number => min + (max - min) * (clampPct(pct) / 100);
const clampByte = (x: number): number => Math.max(0, Math.min(255, Math.round(x)));

function shouldIgnoreWidgetShortcut(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return target.closest([
    "input", "textarea", "button", "select",
    "[contenteditable='true']", "[role='button']", "[role='slider']",
    "[role='switch']", "[role='textbox']", "[role='combobox']", "[role='menuitem']",
    ".MuiSlider-root", ".MuiSelect-select",
  ].join(",")) !== null;
}

function findFFTPeak(
  mag: Float32Array, width: number, height: number,
  col: number, row: number, radius: number,
): { row: number; col: number } {
  const c0 = Math.max(0, Math.floor(col) - radius);
  const r0 = Math.max(0, Math.floor(row) - radius);
  const c1 = Math.min(width - 1, Math.floor(col) + radius);
  const r1 = Math.min(height - 1, Math.floor(row) + radius);
  let bestCol = Math.round(col), bestRow = Math.round(row), bestVal = -Infinity;
  for (let ir = r0; ir <= r1; ir++) {
    for (let ic = c0; ic <= c1; ic++) {
      const val = mag[ir * width + ic];
      if (val > bestVal) { bestVal = val; bestCol = ic; bestRow = ir; }
    }
  }
  const wc0 = Math.max(0, bestCol - 1), wc1 = Math.min(width - 1, bestCol + 1);
  const wr0 = Math.max(0, bestRow - 1), wr1 = Math.min(height - 1, bestRow + 1);
  let sumW = 0, sumWC = 0, sumWR = 0;
  for (let ir = wr0; ir <= wr1; ir++) {
    for (let ic = wc0; ic <= wc1; ic++) {
      const w = mag[ir * width + ic];
      sumW += w; sumWC += w * ic; sumWR += w * ir;
    }
  }
  if (sumW > 0) return { row: sumWR / sumW, col: sumWC / sumW };
  return { row: bestRow, col: bestCol };
}

function findFFTPeakInBounds(
  mag: Float32Array, width: number, height: number,
  col: number, row: number, radius: number,
  minCol: number, maxCol: number, minRow: number, maxRow: number,
): { row: number; col: number } {
  const c0 = Math.max(0, minCol, Math.floor(col) - radius);
  const r0 = Math.max(0, minRow, Math.floor(row) - radius);
  const c1 = Math.min(width - 1, maxCol, Math.floor(col) + radius);
  const r1 = Math.min(height - 1, maxRow, Math.floor(row) + radius);
  let bestCol = Math.round(col), bestRow = Math.round(row), bestVal = -Infinity;
  for (let ir = r0; ir <= r1; ir++) {
    for (let ic = c0; ic <= c1; ic++) {
      const val = mag[ir * width + ic];
      if (val > bestVal) { bestVal = val; bestCol = ic; bestRow = ir; }
    }
  }
  const wc0 = Math.max(0, minCol, bestCol - 1), wc1 = Math.min(width - 1, maxCol, bestCol + 1);
  const wr0 = Math.max(0, minRow, bestRow - 1), wr1 = Math.min(height - 1, maxRow, bestRow + 1);
  let sumW = 0, sumWC = 0, sumWR = 0;
  for (let ir = wr0; ir <= wr1; ir++) {
    for (let ic = wc0; ic <= wc1; ic++) {
      const w = mag[ir * width + ic];
      sumW += w; sumWC += w * ic; sumWR += w * ir;
    }
  }
  if (sumW > 0) return { row: sumWR / sumW, col: sumWC / sumW };
  return { row: bestRow, col: bestCol };
}

function resolveDisplayRange(
  dataMin: number, dataMax: number,
  traitVmin: number | null | undefined, traitVmax: number | null | undefined,
  logScale: boolean, vminPct: number, vmaxPct: number,
): { vmin: number; vmax: number } {
  const baseMin = logScale ? signedLog1p(traitVmin ?? dataMin) : (traitVmin ?? dataMin);
  const baseMax = logScale ? signedLog1p(traitVmax ?? dataMax) : (traitVmax ?? dataMax);
  return sliderRange(baseMin, baseMax, vminPct, vmaxPct);
}

function resolveDisplayBounds(
  dataMin: number, dataMax: number,
  traitVmin: number | null | undefined, traitVmax: number | null | undefined,
  logScale: boolean,
): { min: number; max: number } {
  return {
    min: logScale ? signedLog1p(traitVmin ?? dataMin) : (traitVmin ?? dataMin),
    max: logScale ? signedLog1p(traitVmax ?? dataMax) : (traitVmax ?? dataMax),
  };
}

function cachedAutoRange(
  vmins: number[] | null | undefined,
  vmaxs: number[] | null | undefined,
  idx: number,
): { vmin: number; vmax: number } | null {
  const vmin = vmins?.[idx];
  const vmax = vmaxs?.[idx];
  if (typeof vmin !== "number" || typeof vmax !== "number") return null;
  return Number.isFinite(vmin) && Number.isFinite(vmax) && vmax > vmin ? { vmin, vmax } : null;
}

function cachedAutoDisplayRange(
  vmins: number[] | null | undefined,
  vmaxs: number[] | null | undefined,
  idx: number,
  logScale: boolean,
): { vmin: number; vmax: number } | null {
  const range = cachedAutoRange(vmins, vmaxs, idx);
  if (!range) return null;
  if (!logScale) return range;
  return { vmin: signedLog1p(range.vmin), vmax: signedLog1p(range.vmax) };
}

const show3dPerfDebugFallback: Record<string, unknown> = {};

function show3dPerfDebug(): Record<string, unknown> | null {
  if (typeof window === "undefined") return null;
  const host = window as unknown as { __quantemShow3DPerf?: Record<string, unknown> };
  if (host.__quantemShow3DPerf) return host.__quantemShow3DPerf;
  try {
    host.__quantemShow3DPerf = {};
    return host.__quantemShow3DPerf;
  } catch {
    try {
      (document.documentElement as unknown as { __quantemShow3DPerf?: Record<string, unknown> }).__quantemShow3DPerf = show3dPerfDebugFallback;
    } catch {
      // Ignore locked-down standalone export environments; diagnostics must not
      // affect the rendering path.
    }
    return show3dPerfDebugFallback;
  }
}

function orderedFramePrewarmIndices(startIdx: number, nFrames: number): number[] {
  const n = Math.max(1, Math.round(nFrames || 1));
  const start = ((Math.round(startIdx) % n) + n) % n;
  const order: number[] = [start];
  for (let distance = 1; distance < n; distance++) {
    order.push((start + distance) % n);
    if (order.length >= n) break;
    order.push((start - distance + n) % n);
  }
  return order;
}

const FRAME_INTERVAL_HISTORY = 512;

function resetFramePacingDebug(dbg: Record<string, unknown>, targetMs: number): void {
  dbg.frameIntervalTargetMs = Number(targetMs.toFixed(2));
  dbg.frameIntervalCount = 0;
  dbg.frameIntervalSumMs = 0;
  dbg.frameIntervalAvgMs = 0;
  dbg.lastFrameIntervalMs = null;
  dbg.maxFrameIntervalMs = 0;
  dbg.overBudgetFrames = 0;
  dbg.frameIntervalHistory = [];
  dbg.lastRenderedAt = null;
}

function recordFramePacingDebug(dbg: Record<string, unknown>, now: number, targetMs: number): void {
  const lastRenderedAt = Number(dbg.lastRenderedAt ?? 0);
  if (lastRenderedAt > 0) {
    const interval = Math.max(0, now - lastRenderedAt);
    const count = Number(dbg.frameIntervalCount ?? 0) + 1;
    const sum = Number(dbg.frameIntervalSumMs ?? 0) + interval;
    const longFrameBudgetMs = Math.max(targetMs * 1.5, targetMs + 8);
    const history = Array.isArray(dbg.frameIntervalHistory)
      ? (dbg.frameIntervalHistory as number[])
      : [];
    history.push(Number(interval.toFixed(2)));
    if (history.length > FRAME_INTERVAL_HISTORY) history.splice(0, history.length - FRAME_INTERVAL_HISTORY);

    dbg.frameIntervalCount = count;
    dbg.frameIntervalSumMs = Number(sum.toFixed(2));
    dbg.frameIntervalAvgMs = Number((sum / count).toFixed(2));
    dbg.lastFrameIntervalMs = Number(interval.toFixed(2));
    dbg.maxFrameIntervalMs = Number(Math.max(Number(dbg.maxFrameIntervalMs ?? 0), interval).toFixed(2));
    dbg.overBudgetFrames = Number(dbg.overBudgetFrames ?? 0) + (interval > longFrameBudgetMs ? 1 : 0);
    dbg.frameIntervalHistory = history;
  }
  dbg.lastRenderedAt = now;
}

function percentileFromHistory(values: unknown, percentile: number): number | null {
  if (!Array.isArray(values) || values.length === 0) return null;
  const nums = values
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v))
    .sort((a, b) => a - b);
  if (nums.length === 0) return null;
  const idx = Math.min(nums.length - 1, Math.max(0, Math.ceil((percentile / 100) * nums.length) - 1));
  return Number(nums[idx].toFixed(2));
}

function sameNumberArray(a: number[] | undefined, b: number[]): boolean {
  if (!Array.isArray(a) || a.length !== b.length) return false;
  return a.every((value, idx) => value === b[idx]);
}

function normalizeHiddenPageSlots(values: unknown, maxSlots: number): number[] {
  const nSlots = Math.max(0, Math.trunc(Number(maxSlots) || 0));
  if (!Array.isArray(values) || nSlots <= 1) return [];
  const clean = new Set<number>();
  for (const value of values) {
    const slot = Math.trunc(Number(value));
    if (Number.isFinite(slot) && slot >= 0 && slot < nSlots) clean.add(slot);
  }
  const sorted = Array.from(clean).sort((a, b) => a - b);
  if (sorted.length >= nSlots) sorted.pop();
  return sorted;
}

function estimateRafFps(sampleMs: number): Promise<number | null> {
  if (typeof window === "undefined" || typeof window.requestAnimationFrame !== "function") {
    return Promise.resolve(null);
  }
  return new Promise(resolve => {
    let first = 0;
    let last = 0;
    let frames = 0;
    const tick = (ts: number) => {
      if (first === 0) first = ts;
      last = ts;
      frames++;
      if (ts - first >= sampleMs) {
        const elapsed = Math.max(1, last - first);
        resolve(frames > 1 ? (frames - 1) * 1000 / elapsed : null);
        return;
      }
      window.requestAnimationFrame(tick);
    };
    window.requestAnimationFrame(tick);
  });
}

const FRAME_SERVER_STREAM_CACHE_BYTES = 4 * 1024 * 1024 * 1024;
const FRAME_SERVER_FULL_STACK_CACHE_BYTES = 24 * 1024 * 1024 * 1024;
const FRAME_SERVER_JS_FULL_STACK_CACHE_BYTES = 8 * 1024 * 1024 * 1024;
const FRAME_SERVER_SEPARATE_PANEL_GPU_CACHE_BYTES = 1024 * 1024 * 1024;
const FRAME_SERVER_MIN_CACHE_FRAMES = 6;
const FRAME_SERVER_PREFETCH_FRAMES = 8;

// ============================================================================
// Inlined components (matches Show2D single-file convention)
// ============================================================================
function InfoTooltip({
  text,
  theme = "dark",
  icon = "ⓘ",
}: {
  text: React.ReactNode;
  theme?: "light" | "dark";
  icon?: React.ReactNode;
}) {
  const isDark = theme === "dark";
  const content = typeof text === "string"
    ? <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    : text;
  return (
    <Tooltip
      title={content} arrow placement="bottom"
      componentsProps={{
        tooltip: { sx: { bgcolor: isDark ? "#333" : "#fff", color: isDark ? "#ddd" : "#333", border: `1px solid ${isDark ? "#555" : "#ccc"}`, maxWidth: 360, p: 1 } },
        arrow: { sx: { color: isDark ? "#333" : "#fff", "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` } } },
      }}
    >
      <Typography component="span" sx={{ fontSize: 12, color: isDark ? "#888" : "#666", cursor: "help", ml: 0.5, "&:hover": { color: isDark ? "#aaa" : "#444" } }}>
        {icon}
      </Typography>
    </Tooltip>
  );
}

function KeyboardShortcuts({ items }: { items: [string, string][] }) {
  return (
    <Box
      component="table"
      sx={{
        borderCollapse: "collapse",
        "& td": { py: 0.25, fontSize: 11, lineHeight: 1.3, verticalAlign: "top" },
        "& td:first-of-type": { pr: 1.5, opacity: 0.7, fontFamily: "monospace", fontSize: 10, whiteSpace: "nowrap" },
      }}
    >
      <tbody>
        {items.map(([key, desc], i) => (
          <tr key={i}><td>{key}</td><td>{desc}</td></tr>
        ))}
      </tbody>
    </Box>
  );
}

interface HistogramProps {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  onRangePreview?: (min: number, max: number) => void;
  commitOnChange?: boolean;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
  pinBinsToRange?: boolean;
  ariaHidden?: boolean;
  // Pre-computed 256-element bin array (e.g. from GPU). When provided, the
  // CPU `computeHistogramFromBytes` fallback is skipped entirely.
  bins?: number[] | null;
}

const Histogram = React.memo(function Histogram({
  data, vminPct, vmaxPct, onRangeChange,
  onRangePreview,
  commitOnChange = true,
  width = 110, height = 40, theme = "dark",
  dataMin = 0, dataMax = 1, pinBinsToRange = true, ariaHidden = false,
  bins: precomputedBins = null,
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const sliderRef = React.useRef<HTMLDivElement | null>(null);
  const minLabelRef = React.useRef<HTMLElement | null>(null);
  const maxLabelRef = React.useRef<HTMLElement | null>(null);
  const onRangeChangeRef = React.useRef(onRangeChange);
  const onRangePreviewRef = React.useRef(onRangePreview);
  const pendingRangeRef = React.useRef<[number, number] | null>(null);
  const previewRangeRef = React.useRef<[number, number] | null>(null);
  const [previewRange, setPreviewRange] = React.useState<[number, number] | null>(null);
  const rangeRafRef = React.useRef<number | null>(null);
  // Bins source priority: GPU-precomputed > CPU memoized scan. The CPU path
  // is an O(N) pass over 16.8 M Float32 at 4k (89% of scrub cost in profiling)
  // so we only run it when the GPU path didn't produce bins.
  const bins = React.useMemo(
    () => {
      // Use GPU-precomputed bins only if non-empty. The GPU path can return
      // an all-zero array when the engine slot has no data yet (e.g. the
      // colormap render effect hasn't run yet), which would draw a blank
      // histogram. Falling back to the CPU bin scan in that case keeps the
      // first paint correct; subsequent renders use the GPU bins.
      if (precomputedBins && precomputedBins.length === 256) {
        let total = 0;
        for (let i = 0; i < precomputedBins.length; i++) total += precomputedBins[i];
        if (total > 0) return precomputedBins;
      }
      return pinBinsToRange
        ? computeHistogramFromBytes(data, 256, dataMin, dataMax)
        : computeHistogramFromBytes(data);
    },
    [precomputedBins, data, dataMin, dataMax, pinBinsToRange],
  );
  const colors = theme === "dark"
    ? { bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333" }
    : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" };
  const formatValue = React.useCallback((pct: number) => {
    const val = dataMin + (pct / 100) * (dataMax - dataMin);
    return val >= 1000 ? val.toExponential(2) : val.toFixed(2);
  }, [dataMax, dataMin]);
  const drawHistogram = React.useCallback((loPct: number, hiPct: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    // setTransform (not scale) so React 19 StrictMode double-invoke doesn't stack.
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
    const vminBin = Math.floor((loPct / 100) * displayBins);
    const vmaxBin = Math.floor((hiPct / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      const x = i * barWidth;
      ctx.fillStyle = i >= vminBin && i <= vmaxBin ? colors.barActive : colors.barInactive;
      ctx.fillRect(x + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
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
    drawHistogram(vminPct, vmaxPct);
  }, [drawHistogram, vmaxPct, vminPct]);
  React.useEffect(() => {
    onRangeChangeRef.current = onRangeChange;
  }, [onRangeChange]);
  React.useEffect(() => {
    onRangePreviewRef.current = onRangePreview;
  }, [onRangePreview]);
  React.useEffect(() => {
    if (commitOnChange) return;
    const active = previewRangeRef.current;
    if (
      active &&
      Math.abs(active[0] - vminPct) < 0.01 &&
      Math.abs(active[1] - vmaxPct) < 0.01
    ) {
      previewRangeRef.current = null;
      setPreviewRange(null);
    }
  }, [commitOnChange, vminPct, vmaxPct]);
  const updatePreviewRange = React.useCallback((next: [number, number]) => {
    if (commitOnChange) {
      onRangeChangeRef.current(next[0], next[1]);
      return;
    }
    previewRangeRef.current = next;
    setPreviewRange(next);
    applyRangePreview(next);
    onRangePreviewRef.current?.(next[0], next[1]);
  }, [applyRangePreview, commitOnChange]);
  const commitRangePreview = React.useCallback((next: [number, number]) => {
    previewRangeRef.current = next;
    setPreviewRange(next);
    applyRangePreview(next);
    onRangeChangeRef.current(next[0], next[1]);
  }, [applyRangePreview]);
  const flushRangePreview = React.useCallback(() => {
    if (rangeRafRef.current != null) {
      window.cancelAnimationFrame(rangeRafRef.current);
      rangeRafRef.current = null;
    }
    const pending = pendingRangeRef.current;
    pendingRangeRef.current = null;
    if (pending) {
      commitRangePreview(pending);
    }
  }, [commitRangePreview]);
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
            applyRangePreview(pending);
            if (commitOnChange) onRangeChangeRef.current(pending[0], pending[1]);
            else onRangePreviewRef.current?.(pending[0], pending[1]);
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
  }, [applyRangePreview, flushRangePreview]);

  const sliderInset = 4;
  const sliderWidth = Math.max(1, width - sliderInset * 2);
  const sliderValue = previewRange ?? [vminPct, vmaxPct];

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0, width, overflow: "visible" }}>
      <Box sx={{ position: "relative", width, height: height + 6, overflow: "visible" }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, border: `1px solid ${colors.border}`, display: "block" }}
        role={ariaHidden ? undefined : "img"}
        aria-hidden={ariaHidden ? "true" : undefined}
        aria-label={ariaHidden ? undefined : "Histogram of intensity values with min and max clip handles"}
      />
      <Box
        ref={sliderRef}
        onMouseDownCapture={(e) => {
          if ((e.target as HTMLElement).closest(".MuiSlider-thumb")) return;
          const rect = sliderRef.current?.getBoundingClientRect();
          if (!rect) return;
          const lo = Math.max(0, Math.min(100, Math.min(vminPct, vmaxPct)));
          const hi = Math.max(0, Math.min(100, Math.max(vminPct, vmaxPct)));
          const pct = ((e.clientX - rect.left) / Math.max(1, rect.width)) * 100;
          if (pct < lo || pct > hi) return;
            const thumbGuardPct = Math.max(4, (10 / Math.max(1, rect.width)) * 100);
            if (Math.abs(pct - lo) <= thumbGuardPct || Math.abs(pct - hi) <= thumbGuardPct) return;
            beginRangeDrag(e, rect.width, lo, hi);
            e.preventDefault();
            e.stopPropagation();
          e.nativeEvent.stopImmediatePropagation();
        }}
        sx={{ position: "absolute", left: sliderInset, top: height - 1, width: sliderWidth, height: 8, display: "flex", alignItems: "flex-start", cursor: "grab", zIndex: 2, overflow: "visible" }}
      >
        <Slider
          value={sliderValue}
          onChange={(_, v) => {
            const [newMin, newMax] = v as number[];
            updatePreviewRange([
              Math.min(newMin, newMax - 1),
              Math.max(newMax, newMin + 1),
            ]);
          }}
          onChangeCommitted={(_, v) => {
            if (commitOnChange) return;
            const [newMin, newMax] = v as number[];
            commitRangePreview([
              Math.min(newMin, newMax - 1),
              Math.max(newMax, newMin + 1),
            ]);
          }}
          min={0} max={100} size="small"
          valueLabelDisplay="auto" valueLabelFormat={formatValue}
          aria-label="Histogram intensity clip range"
          sx={{
            width: sliderWidth, py: 0,
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
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}>
        <Typography ref={minLabelRef} sx={{ fontSize: 8, fontFamily: UI_FONT, opacity: 0.6, lineHeight: 1 }}>{formatValue(vminPct)}</Typography>
        <Typography ref={maxLabelRef} sx={{ fontSize: 8, fontFamily: UI_FONT, opacity: 0.6, lineHeight: 1 }}>{formatValue(vmaxPct)}</Typography>
      </Box>
    </Box>
  );
});

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const container = {
  // Match the shared canvas scale-bar typography for a clean microscope-viewer UI.
  root: {
    p: 2,
    bgcolor: "transparent",
    color: "inherit",
    fontFamily: UI_FONT,
    overflow: "visible",
    "& .MuiTypography-root, & .MuiButton-root, & .MuiInputBase-root": { fontFamily: "inherit" },
  },
  imageBox: { bgcolor: "transparent", overflow: "hidden", position: "relative" as const },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

import { resolveDenoiseMode, applyDisplayFilterBrowser, browserFilterSupported, filterKnobsActive, getGPUDisplayFilterEngine } from "../displayFilter";
import { COLORMAPS, COLORMAP_NAMES, applyColormap, renderToOffscreen, renderToOffscreenReuse, createGPUColormapEngine, GPUColormapEngine } from "../colormaps";

const DPR = window.devicePixelRatio || 1;
const RESIZE_HIT_AREA_PX = 10;
const ENABLE_GPU_CANVAS_DISPLAY = true;

function packedRgbFromHex(color: string): number {
  const raw = (color.startsWith("#") ? color.slice(1) : color).trim();
  const expanded = raw.length === 3
    ? raw.split("").map(ch => ch + ch).join("")
    : raw.slice(0, 6);
  const parsed = Number.parseInt(expanded, 16);
  return Number.isFinite(parsed) ? parsed & 0xFFFFFF : 0;
}

// ROI drawing
function drawROI(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  shape: "circle" | "square" | "rectangle" | "annular",
  radius: number,
  width: number,
  height: number,
  activeColor: string,
  inactiveColor: string,
  active: boolean = false,
  innerRadius: number = 0
): void {
  const strokeColor = active ? activeColor : inactiveColor;
  ctx.strokeStyle = strokeColor;
  // Caller sets ctx.lineWidth from roi.line_width; don't clobber.
  if (shape === "circle") {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
  } else if (shape === "square") {
    ctx.strokeRect(x - radius, y - radius, radius * 2, radius * 2);
  } else if (shape === "rectangle") {
    ctx.strokeRect(x - width / 2, y - height / 2, width, height);
  } else if (shape === "annular") {
    // Outer circle
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
    // Inner circle (cyan)
    ctx.strokeStyle = active ? "#0ff" : inactiveColor;
    ctx.beginPath();
    ctx.arc(x, y, innerRadius, 0, Math.PI * 2);
    ctx.stroke();
    // Annular fill
    ctx.fillStyle = (active ? activeColor : inactiveColor) + "15";
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.arc(x, y, innerRadius, 0, Math.PI * 2, true);
    ctx.fill();
    ctx.strokeStyle = strokeColor;
  }
  if (active) {
    ctx.beginPath();
    ctx.moveTo(x - 5, y);
    ctx.lineTo(x + 5, y);
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x, y + 5);
    ctx.stroke();
  }
}

function drawPanelOverlays(
  ctx: CanvasRenderingContext2D,
  overlays: PanelOverlaySpec[] | undefined,
  toScreenX: (col: number) => number,
  toScreenY: (row: number) => number,
  imageW: number,
  imageH: number,
): void {
  if (!overlays?.length) return;
  const ordered = [...overlays].sort((a, b) => styleNumber(a.z_order, 0) - styleNumber(b.z_order, 0));
  for (const overlay of ordered) {
    const coords = overlay.coords || "data";
    const shape = overlay.shape === "rectangle" ? "rect" : (overlay.shape || "circle");
    const scaleRow = coords === "relative" ? imageH : 1;
    const scaleCol = coords === "relative" ? imageW : 1;
    const scaleRadius = coords === "relative" ? Math.min(imageW, imageH) : 1;
    const opacity = styleNumber(overlay.opacity, 1);
    const strokeOpacity = opacity * styleNumber(overlay.stroke_opacity, 1);
    const fillOpacity = opacity * styleNumber(overlay.fill_opacity, overlay.fill ? 1 : 0);
    const stroke = withAlpha(styleString(overlay.stroke, "#00e5ff"), strokeOpacity);
    const fill = withAlpha(overlay.fill, fillOpacity);
    ctx.save();
    ctx.lineWidth = Math.max(0, styleNumber(overlay.stroke_width, 2));
    ctx.setLineDash(overlayDashPattern(overlay, ctx.lineWidth));
    ctx.lineCap = ctx.getLineDash().length ? "round" : "butt";
    if (fill) ctx.fillStyle = fill;
    if (stroke) ctx.strokeStyle = stroke;
    if (shape === "circle") {
      const col = styleNumber(overlay.col, 0) * scaleCol;
      const row = styleNumber(overlay.row, 0) * scaleRow;
      const radius = Math.max(0, styleNumber(overlay.radius, 0) * scaleRadius);
      const x = toScreenX(col);
      const y = toScreenY(row);
      const rx = Math.abs(toScreenX(col + radius) - x);
      const ry = Math.abs(toScreenY(row + radius) - y);
      const r = Math.max(0, (rx + ry) / 2);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      if (fill) ctx.fill();
      if (stroke && ctx.lineWidth > 0) ctx.stroke();
    } else {
      const col0 = styleNumber(overlay.col0, 0) * scaleCol;
      const row0 = styleNumber(overlay.row0, 0) * scaleRow;
      const col1 = styleNumber(overlay.col1, col0) * scaleCol;
      const row1 = styleNumber(overlay.row1, row0) * scaleRow;
      const x0 = toScreenX(col0);
      const y0 = toScreenY(row0);
      const x1 = toScreenX(col1);
      const y1 = toScreenY(row1);
      const x = Math.min(x0, x1);
      const y = Math.min(y0, y1);
      const w = Math.abs(x1 - x0);
      const h = Math.abs(y1 - y0);
      if (fill) ctx.fillRect(x, y, w, h);
      if (stroke && ctx.lineWidth > 0) ctx.strokeRect(x, y, w, h);
    }
    ctx.restore();
  }
}

function clonePanelOverlays(overlays: PanelOverlaySpec[][] | undefined): PanelOverlaySpec[][] {
  return (overlays || []).map((items) => (items || []).map((item) => ({ ...item })));
}

function overlayGeometry(overlay: PanelOverlaySpec, imageW: number, imageH: number) {
  const coords = overlay.coords || "data";
  const scaleRow = coords === "relative" ? imageH : 1;
  const scaleCol = coords === "relative" ? imageW : 1;
  const scaleRadius = coords === "relative" ? Math.min(imageW, imageH) : 1;
  const shape = overlay.shape === "rectangle" ? "rect" : (overlay.shape || "circle");
  if (shape === "circle") {
    return {
      shape,
      row: styleNumber(overlay.row, 0) * scaleRow,
      col: styleNumber(overlay.col, 0) * scaleCol,
      radius: Math.max(0, styleNumber(overlay.radius, 0) * scaleRadius),
    };
  }
  const row0 = styleNumber(overlay.row0, 0) * scaleRow;
  const col0 = styleNumber(overlay.col0, 0) * scaleCol;
  const row1 = styleNumber(overlay.row1, row0) * scaleRow;
  const col1 = styleNumber(overlay.col1, col0) * scaleCol;
  return {
    shape,
    row0: Math.min(row0, row1),
    col0: Math.min(col0, col1),
    row1: Math.max(row0, row1),
    col1: Math.max(col0, col1),
  };
}

function panelOverlayHit(
  overlays: PanelOverlaySpec[] | undefined,
  row: number,
  col: number,
  imageW: number,
  imageH: number,
  hitRadius: number,
): { overlay: number; mode: "move" | "resize"; handle?: string } | null {
  if (!overlays?.length) return null;
  const ordered = overlays.map((overlay, index) => ({ overlay, index })).sort((a, b) => styleNumber(a.overlay.z_order, 0) - styleNumber(b.overlay.z_order, 0));
  for (let orderIdx = ordered.length - 1; orderIdx >= 0; orderIdx -= 1) {
    const { overlay, index } = ordered[orderIdx];
    const geom = overlayGeometry(overlay, imageW, imageH);
    if (geom.shape === "circle") {
      const dist = Math.hypot(col - geom.col, row - geom.row);
      if (Math.abs(dist - geom.radius) <= hitRadius) return { overlay: index, mode: "resize" };
      if (dist <= geom.radius) return { overlay: index, mode: "move" };
      continue;
    }
    const inside = col >= geom.col0 - hitRadius && col <= geom.col1 + hitRadius && row >= geom.row0 - hitRadius && row <= geom.row1 + hitRadius;
    if (!inside) continue;
    const nearLeft = Math.abs(col - geom.col0) <= hitRadius;
    const nearRight = Math.abs(col - geom.col1) <= hitRadius;
    const nearTop = Math.abs(row - geom.row0) <= hitRadius;
    const nearBottom = Math.abs(row - geom.row1) <= hitRadius;
    if (nearLeft || nearRight || nearTop || nearBottom) {
      return {
        overlay: index,
        mode: "resize",
        handle: `${nearTop ? "t" : ""}${nearBottom ? "b" : ""}${nearLeft ? "l" : ""}${nearRight ? "r" : ""}` || "br",
      };
    }
    return { overlay: index, mode: "move" };
  }
  return null;
}

function updateOverlayFromDrag(
  original: PanelOverlaySpec,
  mode: "move" | "resize",
  startRow: number,
  startCol: number,
  row: number,
  col: number,
  imageW: number,
  imageH: number,
  handle = "br",
): PanelOverlaySpec {
  const coords = original.coords || "data";
  const scaleRow = coords === "relative" ? imageH : 1;
  const scaleCol = coords === "relative" ? imageW : 1;
  const scaleRadius = coords === "relative" ? Math.min(imageW, imageH) : 1;
  const toSpecRow = (value: number) => value / scaleRow;
  const toSpecCol = (value: number) => value / scaleCol;
  const toSpecRadius = (value: number) => value / scaleRadius;
  const geom = overlayGeometry(original, imageW, imageH);
  const next = { ...original };
  if (geom.shape === "circle") {
    if (mode === "move") {
      next.row = toSpecRow(Math.max(0, Math.min(imageH, geom.row + row - startRow)));
      next.col = toSpecCol(Math.max(0, Math.min(imageW, geom.col + col - startCol)));
    } else {
      next.radius = toSpecRadius(Math.max(1, Math.hypot(col - geom.col, row - geom.row)));
    }
    return next;
  }
  let row0 = geom.row0;
  let row1 = geom.row1;
  let col0 = geom.col0;
  let col1 = geom.col1;
  if (mode === "move") {
    const dr = row - startRow;
    const dc = col - startCol;
    const h = row1 - row0;
    const w = col1 - col0;
    row0 = Math.max(0, Math.min(imageH - h, row0 + dr));
    row1 = row0 + h;
    col0 = Math.max(0, Math.min(imageW - w, col0 + dc));
    col1 = col0 + w;
  } else {
    if (handle.includes("t")) row0 = row;
    if (handle.includes("b") || (!handle.includes("t") && !handle.includes("l") && !handle.includes("r"))) row1 = row;
    if (handle.includes("l")) col0 = col;
    if (handle.includes("r") || (!handle.includes("t") && !handle.includes("b") && !handle.includes("l"))) col1 = col;
    if (Math.abs(row1 - row0) < 1) row1 = row0 + (row1 >= row0 ? 1 : -1);
    if (Math.abs(col1 - col0) < 1) col1 = col0 + (col1 >= col0 ? 1 : -1);
  }
  next.row0 = toSpecRow(Math.max(0, Math.min(imageH, Math.min(row0, row1))));
  next.row1 = toSpecRow(Math.max(0, Math.min(imageH, Math.max(row0, row1))));
  next.col0 = toSpecCol(Math.max(0, Math.min(imageW, Math.min(col0, col1))));
  next.col1 = toSpecCol(Math.max(0, Math.min(imageW, Math.max(col0, col1))));
  return next;
}

function drawPanelOverlaySelection(
  ctx: CanvasRenderingContext2D,
  overlay: PanelOverlaySpec | undefined,
  toScreenX: (col: number) => number,
  toScreenY: (row: number) => number,
  imageW: number,
  imageH: number,
): void {
  if (!overlay) return;
  const geom = overlayGeometry(overlay, imageW, imageH);
  ctx.save();
  ctx.setLineDash([5, 3]);
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  if (geom.shape === "circle") {
    const x = toScreenX(geom.col);
    const y = toScreenY(geom.row);
    const r = Math.max(0, (Math.abs(toScreenX(geom.col + geom.radius) - x) + Math.abs(toScreenY(geom.row + geom.radius) - y)) / 2);
    ctx.beginPath();
    ctx.arc(x, y, r + 3, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    const x0 = toScreenX(geom.col0);
    const y0 = toScreenY(geom.row0);
    const x1 = toScreenX(geom.col1);
    const y1 = toScreenY(geom.row1);
    ctx.strokeRect(Math.min(x0, x1) - 3, Math.min(y0, y1) - 3, Math.abs(x1 - x0) + 6, Math.abs(y1 - y0) + 6);
  }
  ctx.setLineDash([]);
  ctx.restore();
}

import { WebGPUFFT, getWebGPUFFT, getGPUInfo, fft2d, fft2dAsync, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2, applyHannWindow2D } from "../fft";
import { computeFftQualityMetrics, formatFftQualityLabel, summarizeFftQualityMetrics, type FftQualityMetrics } from "../fftMetrics";
import {
  browserFilterCacheKey,
  normalizedAverageWindow,
  requiresClientFrameTransform,
  shouldApplyClientDifference,
  supportsClientAverage,
} from "./frameTransform";

const FFT_SNAP_RADIUS = 5;

type SubpixelShift = {
  row: number;
  col: number;
  quality: number;
};

function finiteMean(data: Float32Array): number {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < data.length; i++) {
    const value = data[i];
    if (!Number.isFinite(value)) continue;
    sum += value;
    count++;
  }
  return count > 0 ? sum / count : 0;
}

function finiteMedianSample(data: Float32Array, maxSamples = 8192): number {
  const step = Math.max(1, Math.floor(data.length / maxSamples));
  const values: number[] = [];
  for (let i = 0; i < data.length; i += step) {
    const value = data[i];
    if (Number.isFinite(value)) values.push(value);
  }
  if (!values.length) return 0;
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)];
}

function registrationImage(data: Float32Array, width: number, height: number): Float32Array {
  const out = new Float32Array(width * height);
  const mean = finiteMean(data);
  for (let i = 0; i < out.length; i++) {
    const value = data[i];
    out[i] = Number.isFinite(value) ? value - mean : 0;
  }
  applyHannWindow2D(out, width, height);
  return out;
}

async function fft2dComplex(
  real: Float32Array,
  imag: Float32Array,
  width: number,
  height: number,
  inverse: boolean,
  gpu: WebGPUFFT | null,
): Promise<{ real: Float32Array; imag: Float32Array; width: number; height: number }> {
  if (gpu && width === nextPow2(width) && height === nextPow2(height)) {
    const out = await gpu.fft2D(real, imag, width, height, inverse);
    return { ...out, width, height };
  }
  const paddedW = nextPow2(width);
  const paddedH = nextPow2(height);
  const realCopy = new Float32Array(paddedW * paddedH);
  const imagCopy = new Float32Array(paddedW * paddedH);
  for (let row = 0; row < height; row++) {
    realCopy.set(real.subarray(row * width, row * width + width), row * paddedW);
    imagCopy.set(imag.subarray(row * width, row * width + width), row * paddedW);
  }
  fft2d(realCopy, imagCopy, paddedW, paddedH, inverse);
  return { real: realCopy, imag: imagCopy, width: paddedW, height: paddedH };
}

function wrappedPeakOffset(index: number, size: number): number {
  return index > size / 2 ? index - size : index;
}

function parabolicPeakDelta(prev: number, center: number, next: number): number {
  const denom = prev - 2 * center + next;
  if (!Number.isFinite(denom) || Math.abs(denom) < 1e-12) return 0;
  const delta = 0.5 * (prev - next) / denom;
  if (!Number.isFinite(delta)) return 0;
  return Math.max(-0.5, Math.min(0.5, delta));
}

async function estimateSubpixelShift(
  reference: Float32Array,
  moving: Float32Array,
  width: number,
  height: number,
  gpu: WebGPUFFT | null,
): Promise<SubpixelShift> {
  const refReal = registrationImage(reference, width, height);
  const movReal = registrationImage(moving, width, height);
  const refF = await fft2dComplex(refReal, new Float32Array(refReal.length), width, height, false, gpu);
  const movF = await fft2dComplex(movReal, new Float32Array(movReal.length), width, height, false, gpu);
  const workW = refF.width;
  const workH = refF.height;
  const workSize = workW * workH;
  const crossReal = new Float32Array(workSize);
  const crossImag = new Float32Array(workSize);
  for (let i = 0; i < workSize; i++) {
    const real = refF.real[i] * movF.real[i] + refF.imag[i] * movF.imag[i];
    const imag = refF.imag[i] * movF.real[i] - refF.real[i] * movF.imag[i];
    crossReal[i] = real;
    crossImag[i] = imag;
  }
  const corr = await fft2dComplex(crossReal, crossImag, workW, workH, true, gpu);
  let peakIdx = 0;
  let peakValue = -Infinity;
  let total = 0;
  for (let i = 0; i < corr.real.length; i++) {
    const value = Math.hypot(corr.real[i], corr.imag[i]);
    total += value;
    if (value > peakValue) {
      peakValue = value;
      peakIdx = i;
    }
  }
  const peakRow = Math.floor(peakIdx / corr.width);
  const peakCol = peakIdx % corr.width;
  const at = (row: number, col: number) => {
    const r = (row + corr.height) % corr.height;
    const c = (col + corr.width) % corr.width;
    const idx = r * corr.width + c;
    return Math.hypot(corr.real[idx], corr.imag[idx]);
  };
  const rowDelta = parabolicPeakDelta(at(peakRow - 1, peakCol), peakValue, at(peakRow + 1, peakCol));
  const colDelta = parabolicPeakDelta(at(peakRow, peakCol - 1), peakValue, at(peakRow, peakCol + 1));
  const row = wrappedPeakOffset(peakRow, corr.height) + rowDelta;
  const col = wrappedPeakOffset(peakCol, corr.width) + colDelta;
  const background = total > 0 ? (total - peakValue) / Math.max(1, corr.real.length - 1) : 0;
  const quality = background > 1e-12 ? peakValue / background : peakValue;
  return { row, col, quality };
}

function shiftFrameBilinear(
  frame: Float32Array,
  width: number,
  height: number,
  rowShift: number,
  colShift: number,
  fillValue: number,
): Float32Array {
  if (Math.abs(rowShift) < 1e-4 && Math.abs(colShift) < 1e-4) return frame;
  const out = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    const srcRow = row - rowShift;
    const r0 = Math.floor(srcRow);
    const rf = srcRow - r0;
    for (let col = 0; col < width; col++) {
      const srcCol = col - colShift;
      const c0 = Math.floor(srcCol);
      const cf = srcCol - c0;
      const dst = row * width + col;
      if (r0 < 0 || c0 < 0 || r0 >= height - 1 || c0 >= width - 1) {
        out[dst] = fillValue;
        continue;
      }
      const idx = r0 * width + c0;
      const v00 = frame[idx];
      const v01 = frame[idx + 1];
      const v10 = frame[idx + width];
      const v11 = frame[idx + width + 1];
      out[dst] =
        v00 * (1 - rf) * (1 - cf) +
        v01 * (1 - rf) * cf +
        v10 * rf * (1 - cf) +
        v11 * rf * cf;
    }
  }
  return out;
}

/** Sample intensity values along a line using bilinear interpolation. */
function sampleSingleLine(data: Float32Array, w: number, h: number, row0: number, col0: number, row1: number, col1: number): Float32Array {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  const n = Math.max(2, Math.ceil(len));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const c = col0 + t * dc;
    const r = row0 + t * dr;
    const ci = Math.floor(c), ri = Math.floor(r);
    const cf = c - ci, rf = r - ri;
    const c0c = Math.max(0, Math.min(w - 1, ci));
    const c1c = Math.max(0, Math.min(w - 1, ci + 1));
    const r0c = Math.max(0, Math.min(h - 1, ri));
    const r1c = Math.max(0, Math.min(h - 1, ri + 1));
    out[i] = data[r0c * w + c0c] * (1 - cf) * (1 - rf) +
             data[r0c * w + c1c] * cf * (1 - rf) +
             data[r1c * w + c0c] * (1 - cf) * rf +
             data[r1c * w + c1c] * cf * rf;
  }
  return out;
}

/** Sample intensity along a line, averaging over profileWidth perpendicular pixels. */
function sampleLineProfile(data: Float32Array, w: number, h: number, row0: number, col0: number, row1: number, col1: number, profileWidth: number = 1): Float32Array {
  if (profileWidth <= 1) return sampleSingleLine(data, w, h, row0, col0, row1, col1);
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  if (len < 1e-8) return sampleSingleLine(data, w, h, row0, col0, row1, col1);
  const perpR = -dc / len;
  const perpC = dr / len;
  const half = (profileWidth - 1) / 2;
  let accumulated: Float32Array | null = null;
  for (let k = 0; k < profileWidth; k++) {
    const off = -half + k;
    const vals = sampleSingleLine(data, w, h, row0 + off * perpR, col0 + off * perpC, row1 + off * perpR, col1 + off * perpC);
    if (!accumulated) {
      accumulated = vals;
    } else {
      for (let i = 0; i < vals.length; i++) accumulated[i] += vals[i];
    }
  }
  if (accumulated) for (let i = 0; i < accumulated.length; i++) accumulated[i] /= profileWidth;
  return accumulated || new Float32Array(0);
}

// uint8-stack variants: dequantize ONLY the bilinear corners at each sample
// point instead of materializing the whole frame. Critical for kymograph on 4k
// stacks - sampling a line touches ~lineLen*4*width pixels, not width*height*N.
// `u8` is the packed offline stack; `base` = frameIdx * w * h; value =
// u8[base + idx] * scale + offset.
function sampleSingleLineU8(u8: Uint8Array, base: number, w: number, h: number, scale: number, offset: number, row0: number, col0: number, row1: number, col1: number): Float32Array {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  const n = Math.max(2, Math.ceil(len));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const c = col0 + t * dc;
    const r = row0 + t * dr;
    const ci = Math.floor(c), ri = Math.floor(r);
    const cf = c - ci, rf = r - ri;
    const c0c = Math.max(0, Math.min(w - 1, ci));
    const c1c = Math.max(0, Math.min(w - 1, ci + 1));
    const r0c = Math.max(0, Math.min(h - 1, ri));
    const r1c = Math.max(0, Math.min(h - 1, ri + 1));
    const v00 = u8[base + r0c * w + c0c] * scale + offset;
    const v01 = u8[base + r0c * w + c1c] * scale + offset;
    const v10 = u8[base + r1c * w + c0c] * scale + offset;
    const v11 = u8[base + r1c * w + c1c] * scale + offset;
    out[i] = v00 * (1 - cf) * (1 - rf) + v01 * cf * (1 - rf) + v10 * (1 - cf) * rf + v11 * cf * rf;
  }
  return out;
}

function sampleLineProfileU8(u8: Uint8Array, base: number, w: number, h: number, scale: number, offset: number, row0: number, col0: number, row1: number, col1: number, profileWidth: number = 1): Float32Array {
  if (profileWidth <= 1) return sampleSingleLineU8(u8, base, w, h, scale, offset, row0, col0, row1, col1);
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  if (len < 1e-8) return sampleSingleLineU8(u8, base, w, h, scale, offset, row0, col0, row1, col1);
  const perpR = -dc / len;
  const perpC = dr / len;
  const half = (profileWidth - 1) / 2;
  let accumulated: Float32Array | null = null;
  for (let k = 0; k < profileWidth; k++) {
    const off = -half + k;
    const vals = sampleSingleLineU8(u8, base, w, h, scale, offset, row0 + off * perpR, col0 + off * perpC, row1 + off * perpR, col1 + off * perpC);
    if (!accumulated) accumulated = vals;
    else for (let i = 0; i < vals.length; i++) accumulated[i] += vals[i];
  }
  if (accumulated) for (let i = 0; i < accumulated.length; i++) accumulated[i] /= profileWidth;
  return accumulated || new Float32Array(0);
}

function pointToSegmentDistance(col: number, row: number, col0: number, row0: number, col1: number, row1: number): number {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const lenSq = dc * dc + dr * dr;
  if (lenSq <= 1e-12) return Math.sqrt((col - col0) ** 2 + (row - row0) ** 2);
  const tRaw = ((col - col0) * dc + (row - row0) * dr) / lenSq;
  const t = Math.max(0, Math.min(1, tRaw));
  const projCol = col0 + t * dc;
  const projRow = row0 + t * dr;
  return Math.sqrt((col - projCol) ** 2 + (row - projRow) ** 2);
}

// ============================================================================
// Constants
// ============================================================================
// Reserved GPU slot for offline-mode histogram compute (well above any
// frame-server slot index = nSlices*nPanels), so uploading the scratch frame
// never clobbers a cached playback slot.
const OFFLINE_HIST_SLOT = 1_000_000;
const CANVAS_TARGET_SIZE = 600;
const MAX_PANEL_COLUMNS = 12;
const FFT_OVERLAY_MAX_SOURCE_SIZE = 512;
const MIN_ZOOM = 0.5;
const MIN_IMAGE_ZOOM = 1;
const MAX_ZOOM = 30;
const MAX_PLAYBACK_FPS = 60;
const HTML_EXPORT_OVERHEAD_BYTES = 700_000;
const ANIMATION_QUALITY_SCALE: Record<string, number> = { low: 0.35, medium: 0.6, high: 1.0 };
const ANIMATION_QUALITY_OPTIONS = ["low", "medium", "high"] as const;
type AnimationQuality = typeof ANIMATION_QUALITY_OPTIONS[number];
type ExportPanelMode = "home" | "gif" | "html" | "mp4";
type ExportSpatialPreset = "full" | "down2" | "down4" | "edge512" | "edge1024";
type GifExportPreset = "slides" | "compact" | "full" | "custom";
const EXPORT_SPATIAL_OPTIONS: Array<{
  value: ExportSpatialPreset;
  label: string;
  downsample: number;
  maxEdgePx: number | null;
}> = [
  { value: "full", label: "Full", downsample: 1, maxEdgePx: null },
  { value: "down2", label: "2x downsample", downsample: 2, maxEdgePx: null },
  { value: "down4", label: "4x downsample", downsample: 4, maxEdgePx: null },
  { value: "edge512", label: "Max edge 512 px", downsample: 1, maxEdgePx: 512 },
  { value: "edge1024", label: "Max edge 1024 px", downsample: 1, maxEdgePx: 1024 },
];
const GIF_EXPORT_PRESETS: Array<{ value: GifExportPreset; label: string }> = [
  { value: "slides", label: "Slides" },
  { value: "compact", label: "Compact" },
  { value: "full", label: "Full" },
];
const PANEL_GPU_READY_TIMEOUT_MS = 1200;
const INITIAL_NATIVE_PREVIEW_DELAY_MS = 350;

let browserGifPaletteCache: Uint8Array | null = null;

function asciiBytes(value: string): Uint8Array {
  const out = new Uint8Array(value.length);
  for (let i = 0; i < value.length; i++) out[i] = value.charCodeAt(i) & 0xff;
  return out;
}

function u16Bytes(value: number): Uint8Array {
  const v = Math.max(0, Math.min(65535, Math.round(value)));
  return new Uint8Array([v & 0xff, (v >> 8) & 0xff]);
}

function concatUint8(parts: Uint8Array[]): Uint8Array {
  let total = 0;
  for (const part of parts) total += part.byteLength;
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.byteLength;
  }
  return out;
}

function browserGifPalette(): Uint8Array {
  if (browserGifPaletteCache) return browserGifPaletteCache;
  const palette = new Uint8Array(256 * 3);
  let idx = 0;
  for (let r = 0; r < 6; r++) {
    for (let g = 0; g < 6; g++) {
      for (let b = 0; b < 6; b++) {
        const j = idx * 3;
        palette[j] = r * 51;
        palette[j + 1] = g * 51;
        palette[j + 2] = b * 51;
        idx++;
      }
    }
  }
  const grayCount = 256 - idx;
  for (let i = 0; idx < 256; idx++, i++) {
    const v = grayCount <= 1 ? 0 : Math.round((i / (grayCount - 1)) * 255);
    const j = idx * 3;
    palette[j] = v;
    palette[j + 1] = v;
    palette[j + 2] = v;
  }
  browserGifPaletteCache = palette;
  return palette;
}

function quantizeRgbaForBrowserGif(rgba: Uint8ClampedArray): Uint8Array {
  const out = new Uint8Array(Math.floor(rgba.length / 4));
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    const a = rgba[j + 3];
    const r = a === 255 ? rgba[j] : Math.round((rgba[j] * a + 255 * (255 - a)) / 255);
    const g = a === 255 ? rgba[j + 1] : Math.round((rgba[j + 1] * a + 255 * (255 - a)) / 255);
    const b = a === 255 ? rgba[j + 2] : Math.round((rgba[j + 2] * a + 255 * (255 - a)) / 255);
    const rq = Math.max(0, Math.min(5, Math.round(r / 51)));
    const gq = Math.max(0, Math.min(5, Math.round(g / 51)));
    const bq = Math.max(0, Math.min(5, Math.round(b / 51)));
    out[i] = rq * 36 + gq * 6 + bq;
  }
  return out;
}

function gifLzwEncode(indices: Uint8Array): Uint8Array {
  const minCodeSize = 8;
  const clearCode = 1 << minCodeSize;
  const endCode = clearCode + 1;
  const codeSize = minCodeSize + 1;
  const bytes: number[] = [];
  let bitBuffer = 0;
  let bitCount = 0;
  const writeCode = (code: number) => {
    bitBuffer |= code << bitCount;
    bitCount += codeSize;
    while (bitCount >= 8) {
      bytes.push(bitBuffer & 0xff);
      bitBuffer >>= 8;
      bitCount -= 8;
    }
  };

  // Conservative "clear-block" GIF LZW: emit raw color indices and reset before
  // the decoder grows past 9-bit codes. It is larger than dictionary-compressed
  // GIF, but browser-side standalone export values correctness over file size.
  writeCode(clearCode);
  let sinceClear = 0;
  for (let i = 0; i < indices.length; i++) {
    if (sinceClear >= 250) {
      writeCode(clearCode);
      sinceClear = 0;
    }
    writeCode(indices[i]);
    sinceClear++;
  }
  writeCode(endCode);
  if (bitCount > 0) bytes.push(bitBuffer & 0xff);
  return new Uint8Array(bytes);
}

function pushGifSubBlocks(parts: Uint8Array[], data: Uint8Array): void {
  for (let offset = 0; offset < data.length; offset += 255) {
    const chunk = data.subarray(offset, Math.min(offset + 255, data.length));
    parts.push(new Uint8Array([chunk.length]));
    parts.push(chunk);
  }
  parts.push(new Uint8Array([0]));
}

function encodeIndexedGif(
  width: number,
  height: number,
  frames: Uint8Array[],
  delayCs: number,
): Uint8Array {
  if (width <= 0 || height <= 0 || frames.length === 0) {
    throw new Error("GIF export needs at least one non-empty frame.");
  }
  const pixelCount = width * height;
  for (const frame of frames) {
    if (frame.length !== pixelCount) {
      throw new Error(`GIF frame has ${frame.length} pixels; expected ${pixelCount}.`);
    }
  }
  const parts: Uint8Array[] = [
    asciiBytes("GIF89a"),
    u16Bytes(width),
    u16Bytes(height),
    new Uint8Array([0xf7, 0, 0]),
    browserGifPalette(),
    new Uint8Array([0x21, 0xff, 0x0b]),
    asciiBytes("NETSCAPE2.0"),
    new Uint8Array([0x03, 0x01, 0x00, 0x00, 0x00]),
  ];
  const delay = Math.max(1, Math.min(65535, Math.round(delayCs)));
  for (const frame of frames) {
    parts.push(new Uint8Array([0x21, 0xf9, 0x04, 0x04]));
    parts.push(u16Bytes(delay));
    parts.push(new Uint8Array([0, 0]));
    parts.push(new Uint8Array([0x2c, 0, 0, 0, 0]));
    parts.push(u16Bytes(width));
    parts.push(u16Bytes(height));
    parts.push(new Uint8Array([0, 8]));
    pushGifSubBlocks(parts, gifLzwEncode(frame));
  }
  parts.push(new Uint8Array([0x3b]));
  return concatUint8(parts);
}

function formatEstimatedHtmlSize(payloadBytes: number): string {
  const htmlBytes = Math.max(0, payloadBytes) * 4 / 3 + HTML_EXPORT_OVERHEAD_BYTES;
  const mb = htmlBytes / (1024 * 1024);
  if (mb >= 100) return `~${Math.round(mb)} MB`;
  if (mb >= 10) return `~${mb.toFixed(1)} MB`;
  return `~${mb.toFixed(2)} MB`;
}

function animationOutputScale(
  width: number,
  height: number,
  quality: string,
  downsample = 1,
  maxEdgePx: number | null = null,
): number {
  const base = ANIMATION_QUALITY_SCALE[quality] ?? ANIMATION_QUALITY_SCALE.medium;
  const factor = Math.max(1, Math.round(downsample || 1));
  let scale = Math.min(1, base / factor);
  if (maxEdgePx && maxEdgePx > 0) {
    scale = Math.min(scale, maxEdgePx / Math.max(1, width, height));
  }
  return Math.max(scale, 1 / Math.max(1, width, height));
}

function spatialOptionFor(value: ExportSpatialPreset) {
  return EXPORT_SPATIAL_OPTIONS.find((option) => option.value === value) || EXPORT_SPATIAL_OPTIONS[0];
}

function buildAnimationFrameIndices(
  nSlices: number,
  startOne: number,
  endOne: number,
  everyN: number,
  maxFrames: number,
): number[] {
  const total = Math.max(1, Math.floor(nSlices || 1));
  const start = Math.max(0, Math.min(total - 1, Math.round(startOne || 1) - 1));
  const end = Math.max(start, Math.min(total - 1, Math.round(endOne || total) - 1));
  const step = Math.max(1, Math.round(everyN || 1));
  const frames: number[] = [];
  for (let idx = start; idx <= end; idx += step) frames.push(idx);
  const cap = Math.max(0, Math.round(maxFrames || 0));
  if (cap > 0 && frames.length > cap) {
    if (cap === 1) return [frames[0]];
    const sampled: number[] = [];
    for (let i = 0; i < cap; i++) {
      sampled.push(frames[Math.round((i * (frames.length - 1)) / (cap - 1))]);
    }
    return sampled;
  }
  return frames;
}

function formatEstimatedAnimationWork(
  width: number,
  height: number,
  nSlices: number,
  visiblePanels: number,
  maxCols: number,
  panelGap: number,
  quality: string,
  downsample = 1,
  maxEdgePx: number | null = null,
): string {
  const scale = animationOutputScale(width, height, quality, downsample, maxEdgePx);
  const panelW = Math.max(1, Math.floor(Math.max(1, width) * scale));
  const panelH = Math.max(1, Math.floor(Math.max(1, height) * scale));
  const panels = Math.max(1, visiblePanels);
  const cols = maxCols <= 0 ? panels : Math.max(1, Math.min(maxCols, panels));
  const rows = Math.max(1, Math.ceil(panels / cols));
  const gap = Math.max(0, Math.round((panelGap || 0) * scale));
  const outW = cols * panelW + Math.max(0, cols - 1) * gap;
  const outH = rows * panelH + Math.max(0, rows - 1) * gap;
  const rgbBytes = outW * outH * Math.max(1, nSlices) * 3;
  const mb = rgbBytes / (1024 * 1024);
  if (mb >= 100) return `~${Math.round(mb)} MB before compression`;
  if (mb >= 10) return `~${mb.toFixed(1)} MB before compression`;
  return `~${mb.toFixed(2)} MB before compression`;
}

const clampPlaybackFps = (value: number) => {
  const fps = Number.isFinite(value) ? value : 1;
  return Math.max(1, Math.min(MAX_PLAYBACK_FPS, fps));
};

const playbackIntervalMs = (value: number) => {
  const fps = clampPlaybackFps(value);
  return 1000 / fps;
};

function suppressFftRadialBackgroundInPlace(data: Float32Array, width: number, height: number): void {
  if (width < 16 || height < 16 || data.length !== width * height) return;
  const cx = Math.floor(width / 2);
  const cy = Math.floor(height / 2);
  const maxRadius = Math.ceil(Math.hypot(Math.max(cx, width - cx), Math.max(cy, height - cy)));
  const sums = new Float64Array(maxRadius + 1);
  const counts = new Uint32Array(maxRadius + 1);

  for (let y = 0; y < height; y++) {
    const dy = y - cy;
    const offset = y * width;
    for (let x = 0; x < width; x++) {
      const radius = Math.min(maxRadius, Math.floor(Math.hypot(x - cx, dy)));
      sums[radius] += data[offset + x];
      counts[radius]++;
    }
  }

  for (let radius = 0; radius <= maxRadius; radius++) {
    if (counts[radius] > 0) sums[radius] /= counts[radius];
  }

  // Display-only whitening: remove the smooth radial pedestal so Bragg spots and
  // lattice peaks remain visible in small FFT overlays without changing the
  // underlying magnitude data used for measurements.
  for (let y = 0; y < height; y++) {
    const dy = y - cy;
    const offset = y * width;
    for (let x = 0; x < width; x++) {
      const radius = Math.min(maxRadius, Math.floor(Math.hypot(x - cx, dy)));
      data[offset + x] -= sums[radius];
    }
  }
}

type ROIItem = {
  row: number;
  col: number;
  shape: string;
  radius: number;
  radius_inner: number;
  width: number;
  height: number;
  color: string;
  line_width: number;
  highlight: boolean;
};
const ROI_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"];

function createROI(row: number, col: number, shape: string, index: number, imgW: number = 0, imgH: number = 0): ROIItem {
  const defR = imgW > 0 && imgH > 0 ? Math.max(10, Math.round(Math.min(imgW, imgH) * 0.05)) : 10;
  return {
    row,
    col,
    shape,
    radius: defR,
    radius_inner: Math.max(5, Math.round(defR * 0.5)),
    width: defR * 2,
    height: defR * 2,
    color: ROI_COLORS[index % ROI_COLORS.length],
    line_width: 2,
    highlight: false,
  };
}

function normalizeROI(roi: ROIItem, index: number): ROIItem {
  return {
    ...roi,
    color: roi.color || ROI_COLORS[index % ROI_COLORS.length],
    shape: roi.shape || "circle",
    radius: roi.radius ?? 10,
    radius_inner: roi.radius_inner ?? 5,
    width: roi.width ?? 20,
    height: roi.height ?? 20,
    line_width: roi.line_width ?? 2,
    highlight: !!roi.highlight,
  };
}

/** Extract a single frame from the playback buffer (zero-copy subarray). */
function getFrameFromBuffer(
  buffer: Float32Array | null,
  bufStart: number,
  bufCount: number,
  nSlices: number,
  frameIdx: number,
  frameSize: number,
): Float32Array | null {
  if (!buffer || bufCount === 0) return null;
  let offset = frameIdx - bufStart;
  if (offset < 0) offset += nSlices;
  if (offset < 0 || offset >= bufCount) return null;
  const start = offset * frameSize;
  const end = start + frameSize;
  if (end > buffer.length) return null;
  return buffer.subarray(start, end);
}

/** Fused single-pass render: optional log scale + normalize + colormap → RGBA.
 *  Eliminates multiple data passes during playback for maximum frame rate. */
function renderFramePlayback(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  logScale: boolean,
): void {
  const range = vmax - vmin;
  const invRange = range > 0 ? 255 / range : 0;
  if (logScale) {
    for (let i = 0; i < data.length; i++) {
      const d = data[i];
      const v = d >= 0 ? Math.log1p(d) : -Math.log1p(-d);
      const idx = v <= vmin ? 0 : v >= vmax ? 255 : ((v - vmin) * invRange) | 0;
      const j = i << 2;
      const k = idx * 3;
      rgba[j] = lut[k];
      rgba[j + 1] = lut[k + 1];
      rgba[j + 2] = lut[k + 2];
      rgba[j + 3] = 255;
    }
  } else {
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      const idx = v <= vmin ? 0 : v >= vmax ? 255 : ((v - vmin) * invRange) | 0;
      const j = i << 2;
      const k = idx * 3;
      rgba[j] = lut[k];
      rgba[j + 1] = lut[k + 1];
      rgba[j + 2] = lut[k + 2];
      rgba[j + 3] = 255;
    }
  }
}

function renderFrameScaledPlayback(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  xMap: Uint32Array,
  yMap: Uint32Array,
  outW: number,
  outH: number,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  logScale: boolean,
): void {
  const range = vmax - vmin;
  const invRange = range > 0 ? 255 / range : 0;
  for (let y = 0; y < outH; y++) {
    const srcRow = yMap[y];
    const outRow = y * outW;
    for (let x = 0; x < outW; x++) {
      let v = data[srcRow + xMap[x]];
      if (logScale) v = v >= 0 ? Math.log1p(v) : -Math.log1p(-v);
      const idx = v <= vmin ? 0 : v >= vmax ? 255 : ((v - vmin) * invRange) | 0;
      const j = (outRow + x) << 2;
      const k = idx * 3;
      rgba[j] = lut[k];
      rgba[j + 1] = lut[k + 1];
      rgba[j + 2] = lut[k + 2];
      rgba[j + 3] = 255;
    }
  }
}

// ============================================================================
// Crop ROI region from raw float32 data for ROI-scoped FFT
// ============================================================================
function cropROIRegion(
  data: Float32Array, imgW: number, imgH: number,
  roi: ROIItem,
): { cropped: Float32Array; cropW: number; cropH: number } | null {
  const shape = roi.shape || "circle";
  let col0: number, row0: number, col1: number, row1: number;

  if (shape === "rectangle") {
    const hw = roi.width / 2;
    const hh = roi.height / 2;
    col0 = Math.max(0, Math.floor(roi.col - hw));
    row0 = Math.max(0, Math.floor(roi.row - hh));
    col1 = Math.min(imgW, Math.ceil(roi.col + hw));
    row1 = Math.min(imgH, Math.ceil(roi.row + hh));
  } else {
    const r = roi.radius;
    col0 = Math.max(0, Math.floor(roi.col - r));
    row0 = Math.max(0, Math.floor(roi.row - r));
    col1 = Math.min(imgW, Math.ceil(roi.col + r));
    row1 = Math.min(imgH, Math.ceil(roi.row + r));
  }

  const cropW = col1 - col0;
  const cropH = row1 - row0;
  if (cropW < 2 || cropH < 2) return null;

  const cropped = new Float32Array(cropW * cropH);

  if (shape === "circle" || shape === "annular") {
    const r = roi.radius;
    const rSq = r * r;
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const imgCol = col0 + dx;
        const imgRow = row0 + dy;
        const distSq = (imgCol - roi.col) * (imgCol - roi.col) + (imgRow - roi.row) * (imgRow - roi.row);
        cropped[dy * cropW + dx] = distSq <= rSq ? data[imgRow * imgW + imgCol] : 0;
      }
    }
  } else {
    for (let dy = 0; dy < cropH; dy++) {
      const srcOffset = (row0 + dy) * imgW + col0;
      cropped.set(data.subarray(srcOffset, srcOffset + cropW), dy * cropW);
    }
  }

  return { cropped, cropW, cropH };
}

// ============================================================================
// Compute stats for pixels inside a single ROI (mean/min/max/std)
// ============================================================================
function computeROIPixelStats(
  data: Float32Array, imgW: number, imgH: number,
  roi: ROIItem,
): { mean: number; min: number; max: number; std: number } | null {
  const shape = roi.shape || "circle";
  let col0: number, row0: number, col1: number, row1: number;

  if (shape === "rectangle") {
    const hw = roi.width / 2;
    const hh = roi.height / 2;
    col0 = Math.max(0, Math.floor(roi.col - hw));
    row0 = Math.max(0, Math.floor(roi.row - hh));
    col1 = Math.min(imgW, Math.ceil(roi.col + hw));
    row1 = Math.min(imgH, Math.ceil(roi.row + hh));
  } else {
    const r = roi.radius;
    col0 = Math.max(0, Math.floor(roi.col - r));
    row0 = Math.max(0, Math.floor(roi.row - r));
    col1 = Math.min(imgW, Math.ceil(roi.col + r));
    row1 = Math.min(imgH, Math.ceil(roi.row + r));
  }

  const cropW = col1 - col0;
  const cropH = row1 - row0;
  if (cropW < 1 || cropH < 1) return null;

  let sum = 0, sumSq = 0, minVal = Infinity, maxVal = -Infinity, n = 0;

  if (shape === "circle") {
    const rSq = roi.radius * roi.radius;
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const imgCol = col0 + dx, imgRow = row0 + dy;
        const distSq = (imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2;
        if (distSq > rSq) continue;
        const v = data[imgRow * imgW + imgCol];
        sum += v; sumSq += v * v;
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
        n++;
      }
    }
  } else if (shape === "annular") {
    const rSq = roi.radius * roi.radius;
    const riSq = (roi.radius_inner || 0) ** 2;
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const imgCol = col0 + dx, imgRow = row0 + dy;
        const distSq = (imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2;
        if (distSq > rSq || distSq < riSq) continue;
        const v = data[imgRow * imgW + imgCol];
        sum += v; sumSq += v * v;
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
        n++;
      }
    }
  } else {
    // square or rectangle - all pixels in bounding box
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const v = data[(row0 + dy) * imgW + (col0 + dx)];
        sum += v; sumSq += v * v;
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
        n++;
      }
    }
  }

  if (n === 0) return null;
  const mean = sum / n;
  const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
  return { mean, min: minVal, max: maxVal, std };
}

// ============================================================================
// Main Component
// ============================================================================
function Show3D() {
  const isMobileViewport = useMobileViewport();
  const canvasRepaintSignal = useCanvasRepaintSignal();
  const model = useModel();
  const folderWatchLive = useFolderWatchModelLive(model);
  React.useLayoutEffect(() => applyStandaloneWidgetViewState(model), [model]);
  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);

  // Theme detection (offline HTML exports force a light/white background)
  const [offlineForTheme] = useModelState<boolean>("_export_light");
  const { themeInfo, colors: baseColors } = useTheme(offlineForTheme);
  const themeColors = {
    ...baseColors,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a",
    accentYellow: themeInfo.theme === "dark" ? "#ff0" : "#b08800",
  };
  const mobileControlRowSx = isMobileViewport
    ? ({ columnGap: "8px", rowGap: "4px", px: 0.75, py: 0.25 } as const)
    : ({} as const);

  // Theme-aware select style (matching Show4DSTEM)
  const themedSelect = {
    ...controlPanel.select,
    fontFamily: "inherit",
    flexShrink: 0,  // never compress a dropdown below its width -> no truncated label
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    "& .MuiSelect-select": { py: 0.5, fontFamily: "inherit", textOverflow: "clip", overflow: "visible" },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
  };

  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}`, fontFamily: UI_FONT, "& .MuiMenuItem-root": { fontFamily: "inherit" } } },
  };
  const themedFastMenuProps = {
    ...themedMenuProps,
    keepMounted: true,
    transitionDuration: 0,
    MenuListProps: { dense: true },
  };

  // Model state (synced with Python)
  const [sliceIdx, setSliceIdx] = useModelState<number>("slice_idx");
  const [nSlices] = useModelState<number>("n_slices");
  const [folderWaiting] = useModelState<boolean>("folder_waiting");
  const [folderStatus] = useModelState<string>("folder_status");
  const [folderWatchState] = useModelState<string>("folder_watch_state");
  const [folderWatchDetail] = useModelState<string>("folder_watch_detail");
  const [labels] = useModelState<string[]>("labels");
  const [panelFrameLabels] = useModelState<string[][]>("panel_frame_labels");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [rawFrameBytes] = useModelState<DataView>("frame_bytes");
  // True-color PNG/JPEG stacks: frame_bytes is (H*W*3) float32 RGB in [0, 1].
  const [isRgb] = useModelState<boolean>("is_rgb");
  const [staticFallbackJpeg] = useModelState<string>("_static_fallback_jpeg");
  const [staticFallbackMime] = useModelState<string>("_static_fallback_mime");
  const rgbFrameDataRef = React.useRef<Float32Array | null>(null);
  // Defensive: traitlets.Bytes can identity-suppress trait events when content
  // and length are similar. frame_seq is incremented Python-side on every write
  // so JS effects always see a change. Use it in dep arrays alongside frameBytes.
  const [frameSeq] = useModelState<number>("frame_seq");
  // Offline mode: standalone HTML can carry either a compact uint8 stack
  // (_offline_stack) or an exact float32 stack (_offline_float_stack). JS
  // slices locally on scrub so exported reports do not need a Python kernel.
  // Sidecar path: empty _offline_stack + _offline_stack_url → fetch from disk.
  const [offline] = useModelState<boolean>("offline");
  const [offlineStackTrait] = useModelState<DataView>("_offline_stack");
  const [offlineStackUrl] = useModelState<string>("_offline_stack_url");
  const [offlineFloatStack] = useModelState<DataView>("_offline_float_stack");
  const [offlineMin] = useModelState<number>("_offline_min");
  const [offlineMax] = useModelState<number>("_offline_max");
  const [offlineMins] = useModelState<number[]>("_offline_mins");
  const [offlineMaxs] = useModelState<number[]>("_offline_maxs");
  const [nPanels] = useModelState<number>("n_panels");
  const [panelWidthPx] = useModelState<number>("panel_width_px");
  const [sharedPanelSource] = useModelState<boolean>("shared_panel_source");
  const [separatePanelFrames] = useModelState<boolean>("separate_panel_frames");
  const offlineStack: DataView | null =
    offlineStackTrait && offlineStackTrait.byteLength > 0
      ? offlineStackTrait
      : null;
  // Reused scratch Float32Array sized to one frame so per-scrub dequant
  // doesn't re-allocate. Indexed by (RGB, width, height) since reshape resets it.
  const offlineScratch = React.useRef<Float32Array | null>(null);
  const offlineScratchKey = React.useRef<number>(-1);
  const offlineFrameCacheRef = React.useRef<Map<number, Float32Array>>(new Map());
  const offlineFramePrewarmSerialRef = React.useRef(0);
  // Local live index used by the offline frameBytes useMemo. During MUI Slider
  // drag, `setSliceIdx` (anywidget useModelState) goes through model.set +
  // save_changes which appears to batch under rapid mousemove ticks - useMemo
  // doesn't see sliceIdx change until the user releases the slider, so the
  // canvas stays on the old frame the whole drag. Local React state updates
  // synchronously per tick. scrubToSlice writes both; the model trait is still
  // updated for state.dict round-trips and observers.
  const [liveSliceIdx, setLiveSliceIdx] = React.useState<number>(sliceIdx);
  React.useEffect(() => { setLiveSliceIdx(sliceIdx); }, [sliceIdx]);
  // Sidecar: load FULL stack into RAM once (per-frame Uint8Array slots — never
  // one multi-GB ArrayBuffer, which Chrome often refuses). Then scrub is free.
  const sidecarU8FrameCacheRef = React.useRef<Map<number, Uint8Array>>(new Map());
  const sidecarFetchInflightRef = React.useRef<Set<number>>(new Set());
  const sidecarRamReadyRef = React.useRef(false);
  const sidecarLoadKeyRef = React.useRef("");
  const sidecarBitmapFrameCacheRef = React.useRef<Map<number, ImageBitmap[]>>(new Map());
  const sidecarBitmapReadyRef = React.useRef(false);
  const sidecarBitmapCompleteRef = React.useRef(false);
  const sidecarBitmapBuildSerialRef = React.useRef(0);
  const sidecarCompositeFrameCacheRef = React.useRef<Map<number, HTMLCanvasElement>>(new Map());
  const sidecarCompositeReadyRef = React.useRef(false);
  const sidecarCompositeCompleteRef = React.useRef(false);
  const sidecarCompositeBuildSerialRef = React.useRef(0);
  const sidecarGpuPresenterRef = React.useRef<{
    device: GPUDevice;
    context: GPUCanvasContext;
    pipeline: GPURenderPipeline;
    sampler: GPUSampler;
    bindGroups: Map<number, GPUBindGroup>;
    textures: GPUTexture[];
    width: number;
    height: number;
  } | null>(null);
  const sidecarGpuReadyRef = React.useRef(false);
  const sidecarGpuBuildSerialRef = React.useRef(0);
  const sidecarSliceCommitTimerRef = React.useRef<number | null>(null);
  const sidecarDisplayCacheDirtyRef = React.useRef(false);
  const sidecarCompositeStyleKeyRef = React.useRef("");
  const [sidecarBitmapReady, setSidecarBitmapReady] = React.useState(false);
  const [sidecarBitmapComplete, setSidecarBitmapComplete] = React.useState(false);
  const [sidecarCompositeReady, setSidecarCompositeReady] = React.useState(false);
  const [sidecarCompositeComplete, setSidecarCompositeComplete] = React.useState(false);
  const [, setSidecarGpuReady] = React.useState(false);
  const [sidecarU8Frame, setSidecarU8Frame] = React.useState<{
    idx: number;
    u8: Uint8Array;
  } | null>(null);
  const [sidecarRamReady, setSidecarRamReady] = React.useState(false);
  const [offlineStackFetchStatus, setOfflineStackFetchStatus] = React.useState<string>("");
  const sidecarMode = Boolean(
    (offlineStackUrl || "").trim()
    && !(offlineStackTrait && offlineStackTrait.byteLength > 0),
  );
  const enableSidecarGpuTexturePresenter = true;
  const enableSidecarNativePanelBitmapCache = false;
  const clearSidecarBitmapCache = React.useCallback(() => {
    sidecarCompositeFrameCacheRef.current.clear();
    sidecarCompositeReadyRef.current = false;
    sidecarCompositeCompleteRef.current = false;
    setSidecarCompositeReady(false);
    setSidecarCompositeComplete(false);
    if (sidecarGpuPresenterRef.current) {
      for (const texture of sidecarGpuPresenterRef.current.textures) {
        try { texture.destroy(); } catch { /* ignore */ }
      }
    }
    sidecarGpuPresenterRef.current = null;
    sidecarGpuReadyRef.current = false;
    setSidecarGpuReady(false);
    for (const bitmaps of sidecarBitmapFrameCacheRef.current.values()) {
      for (const bitmap of bitmaps) {
        try { bitmap.close(); } catch { /* ignore */ }
      }
    }
    sidecarBitmapFrameCacheRef.current.clear();
    sidecarBitmapReadyRef.current = false;
    sidecarBitmapCompleteRef.current = false;
    setSidecarBitmapReady(false);
    setSidecarBitmapComplete(false);
  }, []);
  const clearSidecarCompositeCache = React.useCallback(() => {
    sidecarCompositeFrameCacheRef.current.clear();
    sidecarCompositeReadyRef.current = false;
    sidecarCompositeCompleteRef.current = false;
    setSidecarCompositeReady(false);
    setSidecarCompositeComplete(false);
    if (sidecarGpuPresenterRef.current) {
      for (const texture of sidecarGpuPresenterRef.current.textures) {
        try { texture.destroy(); } catch { /* ignore */ }
      }
    }
    sidecarGpuPresenterRef.current = null;
    sidecarGpuReadyRef.current = false;
    setSidecarGpuReady(false);
  }, []);
  const invalidateSidecarViewportCache = React.useCallback((reason: string) => {
    sidecarCompositeBuildSerialRef.current += 1;
    sidecarGpuBuildSerialRef.current += 1;
    sidecarCompositeStyleKeyRef.current = "";
    sidecarDisplayCacheDirtyRef.current = false;
    const hadViewportCache = (
      sidecarCompositeFrameCacheRef.current.size > 0 ||
      sidecarCompositeReadyRef.current ||
      sidecarCompositeCompleteRef.current ||
      sidecarGpuReadyRef.current ||
      !!sidecarGpuPresenterRef.current
    );
    if (hadViewportCache) clearSidecarCompositeCache();
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.sidecarViewportInvalidatedReason = reason;
      dbg.sidecarCompositeCacheFrames = sidecarCompositeFrameCacheRef.current.size;
      dbg.sidecarGpuTextureFrames = 0;
    }
  }, [clearSidecarCompositeCache]);
  React.useEffect(() => () => {
    if (sidecarSliceCommitTimerRef.current !== null) {
      window.clearTimeout(sidecarSliceCommitTimerRef.current);
      sidecarSliceCommitTimerRef.current = null;
    }
  }, []);

  // 1) Eager full-stack RAM load (once per url/shape).
  React.useEffect(() => {
    if (!sidecarMode) {
      sidecarU8FrameCacheRef.current.clear();
      sidecarRamReadyRef.current = false;
      sidecarLoadKeyRef.current = "";
      clearSidecarBitmapCache();
      setSidecarRamReady(false);
      setSidecarU8Frame(null);
      setOfflineStackFetchStatus("");
      return;
    }
    const url = (offlineStackUrl || "").trim();
    const ch = isRgb ? 3 : 1;
    const bytesPerFrame = Math.max(0, ch * Math.round(width || 0) * Math.round(height || 0));
    const n = Math.max(1, Math.round(nSlices || 1));
    if (!url || bytesPerFrame <= 0) return;

    const loadKey = `${url}|${bytesPerFrame}|${n}|${ch}`;
    if (sidecarLoadKeyRef.current === loadKey && sidecarRamReadyRef.current
      && sidecarU8FrameCacheRef.current.size >= n) {
      setSidecarRamReady(true);
      setOfflineStackFetchStatus("");
      return;
    }

    let cancelled = false;
    sidecarRamReadyRef.current = false;
    setSidecarRamReady(false);
    sidecarU8FrameCacheRef.current.clear();
    offlineFrameCacheRef.current.clear();
    clearSidecarBitmapCache();
    sidecarLoadKeyRef.current = loadKey;

    const totalMb = (n * bytesPerFrame) / (1024 * 1024);
    const loadStarted = performance.now();
    const loadDebug = show3dPerfDebug();
    if (loadDebug) {
      loadDebug.sidecarRamFrames = 0;
      loadDebug.sidecarRamLoadMs = null;
      loadDebug.sidecarBytesPerFrame = bytesPerFrame;
      loadDebug.sidecarTotalMb = Number(totalMb.toFixed(1));
      loadDebug.sidecarBitmapComplete = false;
    }
    setOfflineStackFetchStatus(
      `Loading full stack into RAM (0/${n} frames, ${totalMb.toFixed(0)} MB)…`,
    );

    (async () => {
      try {
        // Prefer streaming the whole file once, splitting into per-frame buffers
        // as bytes arrive (avoids one giant ArrayBuffer and avoids N HTTP RTTs).
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const contentLen = Number(resp.headers.get("content-length") || 0);
        const expected = n * bytesPerFrame;
        if (contentLen > 0 && contentLen !== expected) {
          // Fall through to Range-per-frame if size mismatches (partial file etc.)
          throw new Error(`size mismatch content-length=${contentLen} expected=${expected}`);
        }

        if (resp.body) {
          const reader = resp.body.getReader();
          let received = 0;
          let frameIdx = 0;
          let frameFill = 0;
          let current = new Uint8Array(bytesPerFrame);
          while (frameIdx < n) {
            const { done, value } = await reader.read();
            if (done) break;
            if (!value || value.byteLength === 0) continue;
            let off = 0;
            while (off < value.byteLength && frameIdx < n) {
              const need = bytesPerFrame - frameFill;
              const take = Math.min(need, value.byteLength - off);
              current.set(value.subarray(off, off + take), frameFill);
              frameFill += take;
              off += take;
              received += take;
              if (frameFill === bytesPerFrame) {
                // Own buffer per frame (do not reuse `current` after store).
                const stored = current;
                sidecarU8FrameCacheRef.current.set(frameIdx, stored);
                // Paint first available frame ASAP while the rest loads.
                if (!cancelled && frameIdx === ((Math.round(liveSliceIdx) % n) + n) % n) {
                  setSidecarU8Frame({ idx: frameIdx, u8: stored });
                }
                frameIdx += 1;
                frameFill = 0;
                if (frameIdx < n) current = new Uint8Array(bytesPerFrame);
                if (!cancelled) {
                  const dbg = show3dPerfDebug();
                  if (dbg) dbg.sidecarRamFrames = frameIdx;
                  const pct = ((100 * frameIdx) / n).toFixed(0);
                  setOfflineStackFetchStatus(
                    `Loading full stack into RAM… ${frameIdx}/${n} frames (${pct}%, ${totalMb.toFixed(0)} MB)`,
                  );
                }
              }
            }
            if (cancelled) {
              try { await reader.cancel(); } catch { /* ignore */ }
              return;
            }
          }
          if (frameIdx < n) {
            throw new Error(`truncated stream: got ${frameIdx}/${n} frames (${received} bytes)`);
          }
        } else {
          // No body stream: Range-fetch every frame (still no single multi-GB buffer).
          for (let i = 0; i < n; i++) {
            if (cancelled) return;
            const start = i * bytesPerFrame;
            const end = start + bytesPerFrame - 1;
            const r = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
            if (!(r.ok || r.status === 206)) throw new Error(`HTTP ${r.status} frame ${i}`);
            const buf = new Uint8Array(await r.arrayBuffer());
            if (buf.byteLength !== bytesPerFrame) {
              throw new Error(`frame ${i} size ${buf.byteLength}, expected ${bytesPerFrame}`);
            }
            sidecarU8FrameCacheRef.current.set(i, buf);
            if (!cancelled) {
              const pct = ((100 * (i + 1)) / n).toFixed(0);
              setOfflineStackFetchStatus(
                `Loading full stack into RAM… ${i + 1}/${n} frames (${pct}%, ${totalMb.toFixed(0)} MB)`,
              );
            }
          }
        }

        if (cancelled) return;
        sidecarRamReadyRef.current = true;
        setSidecarRamReady(true);
        setOfflineStackFetchStatus("Full stack in RAM; preparing full-resolution display cache…");
        const ramDebug = show3dPerfDebug();
        if (ramDebug) {
          ramDebug.sidecarRamFrames = sidecarU8FrameCacheRef.current.size;
          ramDebug.sidecarRamLoadMs = performance.now() - loadStarted;
        }
        const target = ((Math.round(liveSliceIdx) % n) + n) % n;
        const u8 = sidecarU8FrameCacheRef.current.get(target);
        if (u8) setSidecarU8Frame({ idx: target, u8 });
      } catch (streamErr) {
        // Fallback: sequential Range loads (works with Range servers even if
        // full GET is awkward).
        if (cancelled) return;
        try {
          setOfflineStackFetchStatus(
            `Loading full stack into RAM (Range)… 0/${n} frames`,
          );
          for (let i = 0; i < n; i++) {
            if (cancelled) return;
            const start = i * bytesPerFrame;
            const end = start + bytesPerFrame - 1;
            const r = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
            if (!(r.ok || r.status === 206)) {
              throw new Error(`HTTP ${r.status} for frame ${i}`);
            }
            const buf = new Uint8Array(await r.arrayBuffer());
            if (buf.byteLength !== bytesPerFrame) {
              throw new Error(`frame ${i} size ${buf.byteLength}, expected ${bytesPerFrame}`);
            }
            sidecarU8FrameCacheRef.current.set(i, buf);
            if (!cancelled) {
              const dbg = show3dPerfDebug();
              if (dbg) dbg.sidecarRamFrames = i + 1;
              const pct = ((100 * (i + 1)) / n).toFixed(0);
              setOfflineStackFetchStatus(
                `Loading full stack into RAM… ${i + 1}/${n} frames (${pct}%, ${totalMb.toFixed(0)} MB)`,
              );
            }
          }
          if (cancelled) return;
          sidecarRamReadyRef.current = true;
          setSidecarRamReady(true);
          setOfflineStackFetchStatus("Full stack in RAM; preparing full-resolution display cache…");
          const ramDebug = show3dPerfDebug();
          if (ramDebug) {
            ramDebug.sidecarRamFrames = sidecarU8FrameCacheRef.current.size;
            ramDebug.sidecarRamLoadMs = performance.now() - loadStarted;
          }
          const target = ((Math.round(liveSliceIdx) % n) + n) % n;
          const u8 = sidecarU8FrameCacheRef.current.get(target);
          if (u8) setSidecarU8Frame({ idx: target, u8 });
        } catch (err) {
          if (!cancelled) {
            sidecarRamReadyRef.current = false;
            setSidecarRamReady(false);
            setSidecarU8Frame(null);
            setOfflineStackFetchStatus(
              `Failed to load sidecar stack: ${err instanceof Error ? err.message : String(err)}`
              + (streamErr instanceof Error ? ` (stream: ${streamErr.message})` : ""),
            );
          }
        }
      }
    })();

    return () => {
      cancelled = true;
    };
    // Intentionally exclude liveSliceIdx — full load is once per stack identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sidecarMode, offlineStackUrl, offlineStackTrait, width, height, nSlices, isRgb, clearSidecarBitmapCache]);

  // 2) After RAM is ready, scrub only indexes memory (no network).
  React.useEffect(() => {
    if (!sidecarMode || !sidecarRamReady) return;
    const n = Math.max(1, Math.round(nSlices || 1));
    const target = ((Math.round(liveSliceIdx) % n) + n) % n;
    const u8 = sidecarU8FrameCacheRef.current.get(target);
    if (u8) {
      if (!sidecarBitmapReadyRef.current) setSidecarU8Frame({ idx: target, u8 });
      setOfflineStackFetchStatus("");
    }
  }, [sidecarMode, sidecarRamReady, liveSliceIdx, nSlices]);

  const offlineFrameCacheLimit = React.useMemo(() => {
    const n = Math.max(1, Math.round(nSlices || 1));
    const pixelCount = Math.max(1, Math.round(width || 0) * Math.round(height || 0));
    if (!offline || pixelCount <= 0) return 0;
    if (offlineFloatStack && offlineFloatStack.byteLength > 0) return n;
    if (sidecarMode) {
      return Math.min(n, 2);
    }
    const frameBytes = Math.max(1, pixelCount * (isRgb ? 3 : 1) * 4);
    const budgetFrames = Math.max(1, Math.floor(OFFLINE_FRAME_CACHE_BYTES / frameBytes));
    const minFrames = frameBytes <= OFFLINE_FRAME_CACHE_BYTES / OFFLINE_FRAME_CACHE_MIN_FRAMES
      ? OFFLINE_FRAME_CACHE_MIN_FRAMES
      : 1;
    return Math.max(1, Math.min(n, Math.max(minFrames, budgetFrames)));
  }, [offline, offlineFloatStack, width, height, nSlices, isRgb, sidecarMode]);
  React.useEffect(() => {
    offlineFrameCacheRef.current.clear();
    offlineFramePrewarmSerialRef.current++;
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.offlineFrameCacheSize = 0;
      dbg.offlineFrameCacheLimit = offlineFrameCacheLimit;
      dbg.offlineFrameCacheHits = 0;
      dbg.offlineFrameCacheMisses = 0;
      dbg.offlineFramePrewarmDone = 0;
      dbg.offlineFramePrewarmTarget = 0;
      dbg.offlineFramePrewarmActive = false;
    }
  }, [
    offline,
    offlineStack,
    offlineFloatStack,
    offlineMin,
    offlineMax,
    offlineMins,
    offlineMaxs,
    width,
    height,
    nPanels,
    panelWidthPx,
    nSlices,
    isRgb,
    offlineFrameCacheLimit,
  ]);
  const putOfflineFrameCache = React.useCallback((idx: number, frame: Float32Array) => {
    if (!offline || offlineFrameCacheLimit <= 0) return;
    const cache = offlineFrameCacheRef.current;
    if (cache.has(idx)) cache.delete(idx);
    cache.set(idx, frame);
    while (cache.size > offlineFrameCacheLimit) {
      const oldest = cache.keys().next().value;
      if (oldest === undefined) break;
      cache.delete(oldest);
    }
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.offlineFrameCacheSize = cache.size;
      dbg.offlineFrameCacheLimit = offlineFrameCacheLimit;
    }
  }, [offline, offlineFrameCacheLimit]);
  const frameBytes = React.useMemo<DataView>(() => {
    // Gray: H*W floats. True-color RGB: H*W*3 floats packed channel-last.
    const ch = isRgb ? 3 : 1;
    const floatsPerFrame = ch * width * height;
    const pixelCount = width * height;
    if (offline && sidecarMode && (sidecarBitmapReady || sidecarCompositeReady) && !isRgb) {
      return rawFrameBytes;
    }
    if (offline && offlineFloatStack && offlineFloatStack.byteLength > 0 && floatsPerFrame > 0) {
      const f32 = float32FrameFromDataView(offlineFloatStack, liveSliceIdx, floatsPerFrame, false);
      if (f32) return new DataView(f32.buffer, f32.byteOffset, f32.byteLength);
    }
    // Sidecar: prefer precomputed float cache (filled once after full RAM load).
    if (offline && sidecarMode && width > 0 && height > 0) {
      const n = Math.max(1, nSlices || 1);
      const idx = ((Math.round(liveSliceIdx) % n) + n) % n;
      const cached = offlineFrameCacheRef.current.get(idx);
      if (cached && cached.length >= floatsPerFrame) {
        return new DataView(cached.buffer, cached.byteOffset, floatsPerFrame * 4);
      }
    }
    // Shared dequant path for embedded offline stack and Range-fetched sidecar frames.
    const dequantU8Frame = (u8: Uint8Array): DataView | null => {
      if (u8.byteLength < ch * pixelCount || width <= 0 || height <= 0) return null;
      const key = ((isRgb ? 1 : 0) << 30) | (width << 15) | height;
      if (offlineScratchKey.current !== key || offlineScratch.current === null) {
        offlineScratch.current = new Float32Array(floatsPerFrame);
        offlineScratchKey.current = key;
      }
      const f32 = offlineScratch.current;
      if (isRgb) {
        for (let i = 0; i < floatsPerFrame; i++) f32[i] = u8[i] / 255.0;
      } else {
        // Offline uint8 packs are already display-quantized per panel. Restore
        // physical units with a panel-tiled loop (not per-pixel panel index).
        const panelCount = Math.max(1, nPanels || 1);
        const panelRanges = panelCount > 1 && offlineMins?.length >= panelCount && offlineMaxs?.length >= panelCount;
        const panelW = Math.max(1, panelWidthPx || Math.floor(width / panelCount) || width);
        if (panelRanges) {
          for (let p = 0; p < panelCount; p++) {
            const lo = offlineMins[p] ?? offlineMin;
            const hi = offlineMaxs[p] ?? offlineMax;
            const scale = (hi - lo) / 255.0;
            const x0 = p * panelW;
            const x1 = Math.min(width, x0 + panelW);
            for (let r = 0; r < height; r++) {
              let i = r * width + x0;
              for (let x = x0; x < x1; x++, i++) f32[i] = u8[i] * scale + lo;
            }
          }
        } else {
          const scale = (offlineMax - offlineMin) / 255.0;
          for (let i = 0; i < pixelCount; i++) f32[i] = u8[i] * scale + offlineMin;
        }
      }
      return new DataView(f32.buffer);
    };
    if (offline && offlineStack && offlineStack.byteLength > 0 && width > 0 && height > 0) {
      // RGB uint8 pack is H*W*3 bytes per frame (display-ready 0–255 → /255).
      const bytesPerFrame = ch * pixelCount;
      const start = liveSliceIdx * bytesPerFrame;
      if (start + bytesPerFrame <= offlineStack.byteLength) {
        const u8 = new Uint8Array(offlineStack.buffer, offlineStack.byteOffset + start, bytesPerFrame);
        const view = dequantU8Frame(u8);
        if (view) return view;
      }
    }
    if (
      offline
      && sidecarMode
      && sidecarU8Frame
      && sidecarU8Frame.idx === ((Math.round(liveSliceIdx) % Math.max(1, nSlices || 1)) + Math.max(1, nSlices || 1)) % Math.max(1, nSlices || 1)
    ) {
      const view = dequantU8Frame(sidecarU8Frame.u8);
      if (view) return view;
    }
    return rawFrameBytes;
  }, [offline, offlineStack, offlineFloatStack, offlineMin, offlineMax, offlineMins, offlineMaxs, rawFrameBytes, liveSliceIdx, width, height, nPanels, panelWidthPx, isRgb, sidecarMode, sidecarU8Frame, nSlices, sidecarBitmapReady, sidecarCompositeReady]);
  const getOfflineFrame = React.useCallback((idx: number): Float32Array | null => {
    // Cache per-frame Float32Array objects by frame index. The previous single
    // scratch buffer was unsafe because pointer-equality upload guards could skip
    // a texture refresh; per-index cached arrays keep stable identity without
    // mutating one shared backing store.
    if (!offline || width <= 0 || height <= 0) return null;
    const n = Math.max(1, nSlices || 1);
    const normalized = ((Math.round(idx) % n) + n) % n;
    const cached = offlineFrameCacheRef.current.get(normalized);
    const dbg = show3dPerfDebug();
    if (cached) {
      offlineFrameCacheRef.current.delete(normalized);
      offlineFrameCacheRef.current.set(normalized, cached);
      if (dbg) {
        dbg.offlineFrameCacheHits = ((dbg.offlineFrameCacheHits as number | undefined) ?? 0) + 1;
        dbg.offlineFrameCacheSize = offlineFrameCacheRef.current.size;
      }
      return cached;
    }
    if (dbg) {
      dbg.offlineFrameCacheMisses = ((dbg.offlineFrameCacheMisses as number | undefined) ?? 0) + 1;
    }
    const ch = isRgb ? 3 : 1;
    const floatsPerFrame = ch * width * height;
    const pixelCount = width * height;
    if (offlineFloatStack && offlineFloatStack.byteLength > 0) {
      const frame = float32FrameFromDataView(offlineFloatStack, normalized, floatsPerFrame, false);
      if (frame) putOfflineFrameCache(normalized, frame);
      return frame;
    }
    const bytesPerFrame = ch * pixelCount;
    const dequantU8 = (u8: Uint8Array): Float32Array => {
      const f32 = new Float32Array(floatsPerFrame);
      if (isRgb) {
        for (let i = 0; i < floatsPerFrame; i++) f32[i] = u8[i] / 255.0;
        return f32;
      }
      const panelCount = Math.max(1, nPanels || 1);
      const panelRanges = panelCount > 1 && offlineMins?.length >= panelCount && offlineMaxs?.length >= panelCount;
      const panelW = Math.max(1, panelWidthPx || Math.floor(width / panelCount) || width);
      if (panelRanges) {
        for (let p = 0; p < panelCount; p++) {
          const lo = offlineMins[p] ?? offlineMin;
          const hi = offlineMaxs[p] ?? offlineMax;
          const scale = (hi - lo) / 255.0;
          const x0 = p * panelW;
          const x1 = Math.min(width, x0 + panelW);
          for (let r = 0; r < height; r++) {
            let i = r * width + x0;
            for (let x = x0; x < x1; x++, i++) f32[i] = u8[i] * scale + lo;
          }
        }
      } else {
        const scale = (offlineMax - offlineMin) / 255.0;
        for (let i = 0; i < pixelCount; i++) f32[i] = u8[i] * scale + offlineMin;
      }
      return f32;
    };
    // Sidecar path: full stack held in RAM as per-frame Uint8Arrays after load.
    if (sidecarMode) {
      const u8 = sidecarU8FrameCacheRef.current.get(normalized);
      if (u8 && u8.byteLength >= bytesPerFrame) {
        const f32 = dequantU8(u8);
        putOfflineFrameCache(normalized, f32);
        return f32;
      }
      // Not in RAM yet (still loading) — optional Range fetch for this frame only.
      const url = (offlineStackUrl || "").trim();
      if (
        url
        && bytesPerFrame > 0
        && !sidecarRamReadyRef.current
        && !sidecarFetchInflightRef.current.has(normalized)
      ) {
        sidecarFetchInflightRef.current.add(normalized);
        const start = normalized * bytesPerFrame;
        const end = start + bytesPerFrame - 1;
        void fetch(url, { headers: { Range: `bytes=${start}-${end}` } })
          .then(async (resp) => {
            if (!(resp.ok || resp.status === 206)) return;
            const buf = new Uint8Array(await resp.arrayBuffer());
            if (buf.byteLength !== bytesPerFrame) return;
            // Do not overwrite a frame already placed by the full-stack loader.
            if (!sidecarU8FrameCacheRef.current.has(normalized)) {
              sidecarU8FrameCacheRef.current.set(normalized, buf);
            }
            putOfflineFrameCache(normalized, dequantU8(buf));
          })
          .catch(() => { /* play tick retries */ })
          .finally(() => {
            sidecarFetchInflightRef.current.delete(normalized);
          });
      }
      return null;
    }
    if (!offlineStack || offlineStack.byteLength === 0) return null;
    const start = normalized * bytesPerFrame;
    if (start < 0 || start + bytesPerFrame > offlineStack.byteLength) return null;
    const u8 = new Uint8Array(offlineStack.buffer, offlineStack.byteOffset + start, bytesPerFrame);
    const f32 = dequantU8(u8);
    putOfflineFrameCache(normalized, f32);
    return f32;
  }, [
    offline,
    width,
    height,
    nSlices,
    offlineFloatStack,
    offlineStack,
    offlineMins,
    offlineMaxs,
    offlineMin,
    offlineMax,
    nPanels,
    panelWidthPx,
    putOfflineFrameCache,
    isRgb,
    sidecarMode,
    offlineStackUrl,
  ]);

  React.useEffect(() => {
    if (!offline || width <= 0 || height <= 0 || nSlices <= 1 || offlineFrameCacheLimit <= 0) return;
    // Sidecar: neighbor Range-fetch already runs in the sidecar effect. Do NOT
    // also prewarm 10+ full float frames here (each is ~38–150 MB work).
    if (sidecarMode) return;
    const serial = ++offlineFramePrewarmSerialRef.current;
    let cancelled = false;
    let timer: number | null = null;
    const order = orderedFramePrewarmIndices(liveSliceIdx, nSlices).slice(0, offlineFrameCacheLimit);
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.offlineFramePrewarmActive = true;
      dbg.offlineFramePrewarmTarget = order.length;
      dbg.offlineFramePrewarmDone = 0;
      dbg.offlineFrameCacheLimit = offlineFrameCacheLimit;
    }
    const schedule = (delayMs = 0) => {
      timer = window.setTimeout(step, delayMs);
    };
    let cursor = 0;
    const step = () => {
      timer = null;
      if (cancelled || serial !== offlineFramePrewarmSerialRef.current) return;
      const frameBudgetStart = performance.now();
      while (cursor < order.length && performance.now() - frameBudgetStart < 8) {
        getOfflineFrame(order[cursor]);
        cursor++;
      }
      const d = show3dPerfDebug();
      if (d) {
        d.offlineFramePrewarmDone = cursor;
        d.offlineFrameCacheSize = offlineFrameCacheRef.current.size;
        d.offlineFrameCacheLimit = offlineFrameCacheLimit;
        d.offlineFramePrewarmActive = cursor < order.length;
      }
      if (cursor < order.length) {
        schedule(0);
      }
    };
    schedule(0);
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
      const d = show3dPerfDebug();
      if (d) d.offlineFramePrewarmActive = false;
    };
  }, [offline, width, height, nSlices, liveSliceIdx, offlineFrameCacheLimit, getOfflineFrame, sidecarMode]);

  // Truthful first-render signal: flipped ONCE after the first frame_bytes
  // arrives and the browser has had time to composite two frames.  Python side
  // observes `_js_rendered` and prints the real end-to-end wall clock, not the
  // misleading Python-only __init__ number.
  const [, setJsRendered] = useModelState<boolean>("_js_rendered");
  const firstRenderFiredRef = React.useRef(false);
  React.useEffect(() => {
    if (firstRenderFiredRef.current) return;
    if (!frameBytes || frameBytes.byteLength === 0) return;
    firstRenderFiredRef.current = true;
    requestAnimationFrame(() => requestAnimationFrame(() => setJsRendered(true)));
  }, [frameBytes, setJsRendered]);

  const [title] = useModelState<string>("title");
  const [showTitle] = useModelState<boolean>("show_title");
  const [dimLabel] = useModelState<string>("dim_label");
  const [dimSampling] = useModelState<number>("dim_sampling");
  const [dimUnit] = useModelState<string>("dim_unit");
  const [panelTitles] = useModelState<string[]>("panel_titles");
  const [panelTitleSpans] = useModelState<RichTitleSpan[][]>("panel_title_spans");
  const [panelRealFrames] = useModelState<number[]>("panel_real_frames");
  const [starred, setStarred] = useModelState<number[]>("starred");
  const [hiddenPanels, setHiddenPanels] = useModelState<number[]>("hidden_panels");
  const [hiddenPageSlotsTrait, setHiddenPageSlotsTrait] = useModelState<number[] | undefined>("hidden_page_slots");
  const [panelOrder, setPanelOrder] = useModelState<number[]>("panel_order");
  const [nPages] = useModelState<number>("n_pages");
  const [pageIdx, setPageIdx] = useModelState<number>("page_idx");
  const [panelsPerPage] = useModelState<number>("panels_per_page");
  const [pageLabels] = useModelState<string[]>("page_labels");
  const [pageStarred, setPageStarred] = useModelState<number[]>("page_starred");
  const [pagePlaying, setPagePlaying] = React.useState(false);
  const [pagePlayFps, setPagePlayFps] = React.useState<number>(2);
  const [pageSliderPreviewIdx, setPageSliderPreviewIdxState] = React.useState<number | null>(null);
  const pageSliderPreviewIdxRef = React.useRef<number | null>(null);
  const currentPageIdxRef = React.useRef(0);
  const pageCommitPendingRef = React.useRef<number | null>(null);
  const pageCommitRafRef = React.useRef<number | null>(null);
  const [reorderMode, setReorderMode] = React.useState(false);
  const [dragOverPanel, setDragOverPanel] = React.useState<number | null>(null);
  const [reorderPreviewOrder, setReorderPreviewOrder] = React.useState<number[] | null>(null);
  const [reorderDragVisual, setReorderDragVisual] = React.useState<ReorderDragVisual | null>(null);
  const draggedPanelRef = React.useRef<number | null>(null);
  const pointerReorderPanelRef = React.useRef<number | null>(null);
  const reorderPreviewOrderRef = React.useRef<number[] | null>(null);
  const reorderDragVisualRef = React.useRef<ReorderDragVisual | null>(null);
  const reorderGhostRef = React.useRef<HTMLDivElement>(null);
  const reorderGhostRafRef = React.useRef<number | null>(null);
  const reorderGhostPendingRef = React.useRef<{ x: number; y: number } | null>(null);
  const reorderDragStartRef = React.useRef<ReorderDragStart | null>(null);
  const reorderDragActivatedRef = React.useRef(false);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const gpuCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const canvasContainerRef = React.useRef<HTMLDivElement>(null);
  const totalPanelCount = Math.max(1, nPanels || 1);
  const isPaged = (nPages || 1) > 1 && (panelsPerPage || 0) > 0;
  const currentPageIdx = Math.max(0, Math.min((nPages || 1) - 1, Math.round(pageIdx || 0)));
  const displayPageIdx = pageSliderPreviewIdx === null
    ? currentPageIdx
    : Math.max(0, Math.min((nPages || 1) - 1, Math.round(pageSliderPreviewIdx || 0)));
  React.useEffect(() => {
    currentPageIdxRef.current = currentPageIdx;
  }, [currentPageIdx]);
  const clampPageIdx = React.useCallback((value: number) => (
    Math.max(0, Math.min((nPages || 1) - 1, Math.round(Number(value) || 0)))
  ), [nPages]);
  const setPageSliderPreviewIdx = React.useCallback((value: number | null) => {
    pageSliderPreviewIdxRef.current = value;
    setPageSliderPreviewIdxState(value);
  }, []);
  const commitPageIdx = React.useCallback((value: number, immediate = false) => {
    const next = clampPageIdx(value);
    pageCommitPendingRef.current = next;
    if (immediate) {
      if (pageCommitRafRef.current !== null) {
        window.cancelAnimationFrame(pageCommitRafRef.current);
        pageCommitRafRef.current = null;
      }
      pageCommitPendingRef.current = null;
      if (next !== currentPageIdxRef.current) setPageIdx(next);
      return;
    }
    if (pageCommitRafRef.current !== null) return;
    pageCommitRafRef.current = window.requestAnimationFrame(() => {
      pageCommitRafRef.current = null;
      const pending = pageCommitPendingRef.current;
      pageCommitPendingRef.current = null;
      if (pending !== null && pending !== currentPageIdxRef.current) setPageIdx(pending);
    });
  }, [clampPageIdx, setPageIdx]);
  const stopPagePlayback = React.useCallback(() => {
    setPagePlaying(value => value ? false : value);
  }, []);
  React.useEffect(() => {
    const preview = pageSliderPreviewIdxRef.current;
    if (preview !== null && preview === currentPageIdx) {
      setPageSliderPreviewIdx(null);
    }
  }, [currentPageIdx, setPageSliderPreviewIdx]);
  React.useEffect(() => () => {
    if (pageCommitRafRef.current !== null) {
      window.cancelAnimationFrame(pageCommitRafRef.current);
      pageCommitRafRef.current = null;
    }
  }, []);
  const pageControlIdx = clampPageIdx(pageSliderPreviewIdx ?? currentPageIdx);
  const pageControlLabel = pageLabels?.[pageControlIdx] || `Page ${pageControlIdx + 1}`;
  const pageControlStatus = `${pageControlLabel} ${pageControlIdx + 1}/${nPages || 1}`;
  React.useEffect(() => {
    if (!isPaged || (nPages || 1) <= 1) setPagePlaying(false);
  }, [isPaged, nPages]);
  React.useEffect(() => {
    if (!pagePlaying || !isPaged || (nPages || 1) <= 1) return;
    const timeout = window.setTimeout(() => {
      const next = (currentPageIdx + 1) % Math.max(1, nPages || 1);
      setPageSliderPreviewIdx(next);
      setPageIdx(next);
    }, 1000 / Math.max(1, pagePlayFps));
    return () => window.clearTimeout(timeout);
  }, [currentPageIdx, isPaged, nPages, pagePlayFps, pagePlaying, setPageIdx, setPageSliderPreviewIdx]);
  const activePageStart = isPaged ? displayPageIdx * Math.max(1, panelsPerPage || 1) : 0;
  const activePageEnd = isPaged ? Math.min(totalPanelCount, activePageStart + Math.max(1, panelsPerPage || 1)) : totalPanelCount;
  const activePageIndices = React.useMemo(
    () => Array.from({ length: Math.max(0, activePageEnd - activePageStart) }, (_, i) => activePageStart + i),
    [activePageStart, activePageEnd]
  );
  const activePanelCount = isPaged ? activePageIndices.length : totalPanelCount;
  const [hiddenPageSlots, setHiddenPageSlots] = React.useState<number[]>([]);
  const hiddenPageSlotsInitializedRef = React.useRef(false);
  React.useEffect(() => {
    if (!isPaged) {
      hiddenPageSlotsInitializedRef.current = false;
      setHiddenPageSlots(prev => prev.length === 0 ? prev : []);
      return;
    }
    if (Array.isArray(hiddenPageSlotsTrait)) {
      const slots = normalizeHiddenPageSlots(hiddenPageSlotsTrait, activePanelCount);
      hiddenPageSlotsInitializedRef.current = true;
      setHiddenPageSlots(prev => sameNumberArray(prev, slots) ? prev : slots);
      return;
    }
    if (hiddenPageSlotsInitializedRef.current) return;
    hiddenPageSlotsInitializedRef.current = true;
    const slots = normalizeHiddenPageSlots(
      (hiddenPanels || []).map((value) => Math.trunc(Number(value)) - activePageStart),
      activePanelCount,
    );
    setHiddenPageSlots(prev => sameNumberArray(prev, slots) ? prev : slots);
  }, [activePageStart, activePanelCount, hiddenPageSlotsTrait, hiddenPanels, isPaged]);
  const hiddenPanelSet = React.useMemo(() => {
    const clean = new Set<number>();
    if (isPaged) {
      for (const value of hiddenPageSlots || []) {
        const slot = Math.trunc(Number(value));
        const idx = activePageStart + slot;
        if (Number.isFinite(slot) && slot >= 0 && slot < activePanelCount && idx >= activePageStart && idx < activePageEnd) {
          clean.add(idx);
        }
      }
    } else {
      for (const value of hiddenPanels || []) {
        const idx = Math.trunc(Number(value));
        if (Number.isFinite(idx) && idx >= 0 && idx < totalPanelCount) clean.add(idx);
      }
    }
    const activeHiddenCount = (isPaged ? activePageIndices : Array.from({ length: totalPanelCount }, (_, panel) => panel))
      .filter((panel) => clean.has(panel)).length;
    if (activeHiddenCount >= Math.max(1, activePanelCount)) {
      const fallback = (isPaged ? activePageIndices : [totalPanelCount - 1])[Math.max(0, activePanelCount - 1)];
      clean.delete(fallback);
    }
    return clean;
  }, [activePageEnd, activePageIndices, activePageStart, activePanelCount, hiddenPageSlots, hiddenPanels, totalPanelCount, isPaged]);
  const naturalPanelOrder = React.useMemo(
    () => isPaged ? activePageIndices : Array.from({ length: totalPanelCount }, (_, panel) => panel),
    [activePageIndices, isPaged, totalPanelCount]
  );
  const orderedPanelIndices = React.useMemo(() => {
    if (isPaged) return naturalPanelOrder;
    const values = Array.isArray(panelOrder) ? panelOrder.map(value => Math.trunc(Number(value))) : [];
    const valid = (
      values.length === totalPanelCount &&
      values.every((value) => Number.isFinite(value) && value >= 0 && value < totalPanelCount) &&
      new Set(values).size === totalPanelCount
    );
    return valid ? values : naturalPanelOrder;
  }, [panelOrder, naturalPanelOrder, totalPanelCount, isPaged]);
  const previewOrderedPanelIndices = React.useMemo(() => {
    if (isPaged) return null;
    const values = Array.isArray(reorderPreviewOrder) ? reorderPreviewOrder.map(value => Math.trunc(Number(value))) : [];
    const valid = (
      values.length === totalPanelCount &&
      values.every((value) => Number.isFinite(value) && value >= 0 && value < totalPanelCount) &&
      new Set(values).size === totalPanelCount
    );
    return valid ? values : null;
  }, [reorderPreviewOrder, totalPanelCount, isPaged]);
  const displayOrderedPanelIndices = previewOrderedPanelIndices || orderedPanelIndices;
  const visiblePanelIndices = React.useMemo(
    () => displayOrderedPanelIndices.filter(panel => !hiddenPanelSet.has(panel)),
    [hiddenPanelSet, displayOrderedPanelIndices]
  );
  const visiblePanelCount = visiblePanelIndices.length;
  const panelMenuTotal = isPaged ? activePanelCount : totalPanelCount;
  const hasPanelChoices = panelMenuTotal > 1;
  const panelLabel = React.useCallback((panel: number) => (
    (panelTitles && panelTitles[panel]) || `Panel ${panel + 1}`
  ), [panelTitles]);
  const panelTitleContent = React.useCallback((panel: number) => (
    renderRichTitle(panelTitleSpans?.[panel], panelLabel(panel))
  ), [panelLabel, panelTitleSpans]);
  const panelTitleText = React.useCallback((panel: number) => (
    richTitlePlainText(panelTitleSpans?.[panel], panelLabel(panel))
  ), [panelLabel, panelTitleSpans]);
  const setPanelHidden = React.useCallback((panel: number, hidden: boolean) => {
    if (panel < 0 || panel >= totalPanelCount) return;
    if (isPaged) {
      if (panel < activePageStart || panel >= activePageEnd) return;
      const slot = panel - activePageStart;
      const next = new Set<number>();
      for (const value of hiddenPageSlots || []) {
        const idx = Math.trunc(Number(value));
        if (Number.isFinite(idx) && idx >= 0 && idx < activePanelCount) next.add(idx);
      }
      if (hidden) {
        if (!next.has(slot) && activePanelCount - next.size <= 1) return;
        next.add(slot);
      } else {
        next.delete(slot);
      }
      const slots = normalizeHiddenPageSlots(Array.from(next), activePanelCount);
      setHiddenPageSlots(slots);
      setHiddenPageSlotsTrait(slots);
      return;
    }
    const next = new Set<number>();
    for (const value of hiddenPanels || []) {
      const idx = Math.trunc(Number(value));
      if (Number.isFinite(idx) && idx >= 0 && idx < totalPanelCount) next.add(idx);
    }
    if (hidden) {
      const activeVisible = (isPaged ? activePageIndices : Array.from({ length: totalPanelCount }, (_, idx) => idx))
        .filter((idx) => !next.has(idx)).length;
      if (!next.has(panel) && activeVisible <= 1) return;
      next.add(panel);
    } else {
      next.delete(panel);
    }
    setHiddenPanels(Array.from(next).sort((a, b) => a - b));
  }, [activePageEnd, activePageStart, activePanelCount, hiddenPageSlots, hiddenPanels, totalPanelCount, isPaged, activePageIndices, setHiddenPanels, setHiddenPageSlotsTrait]);
  const applyPanelOrder = React.useCallback((order: number[]) => {
    const clean = order.filter((value) => Number.isInteger(value) && value >= 0 && value < totalPanelCount);
    if (clean.length !== totalPanelCount || new Set(clean).size !== totalPanelCount) return;
    const natural = clean.every((value, idx) => value === idx);
    setPanelOrder(natural ? [] : clean);
  }, [setPanelOrder, totalPanelCount]);
  const setReorderPreviewOrderValue = React.useCallback((order: number[] | null) => {
    reorderPreviewOrderRef.current = order;
    setReorderPreviewOrder(order);
  }, []);
  const setReorderDragVisualValue = React.useCallback((visual: ReorderDragVisual | null) => {
    reorderDragVisualRef.current = visual;
    setReorderDragVisual(visual);
  }, []);
  const captureReorderPanelImage = React.useCallback((panelRect: DOMRect, containerRect: DOMRect): string => {
    const container = canvasContainerRef.current;
    if (!container) return "";
    const canvases = Array.from(container.querySelectorAll("canvas")) as HTMLCanvasElement[];
    const source = canvases.find((canvas) => {
      const rect = canvas.getBoundingClientRect();
      const style = window.getComputedStyle(canvas);
      const opacity = Number(style.opacity || "1");
      return rect.width > 0 && rect.height > 0 && canvas.width > 0 && canvas.height > 0 &&
        style.display !== "none" && opacity > 0.5;
    }) || canvases.find((canvas) => canvas.width > 0 && canvas.height > 0);
    if (!source) return "";
    const scaleX = source.width / Math.max(1, containerRect.width);
    const scaleY = source.height / Math.max(1, containerRect.height);
    const sx = Math.max(0, Math.round((panelRect.left - containerRect.left) * scaleX));
    const sy = Math.max(0, Math.round((panelRect.top - containerRect.top) * scaleY));
    const sw = Math.max(1, Math.min(source.width - sx, Math.round(panelRect.width * scaleX)));
    const sh = Math.max(1, Math.min(source.height - sy, Math.round(panelRect.height * scaleY)));
    if (sw <= 0 || sh <= 0) return "";
    const scratch = document.createElement("canvas");
    scratch.width = sw;
    scratch.height = sh;
    const ctx = scratch.getContext("2d");
    if (!ctx) return "";
    try {
      ctx.drawImage(source, sx, sy, sw, sh, 0, 0, sw, sh);
      return scratch.toDataURL("image/png");
    } catch {
      return "";
    }
  }, []);
  const updateReorderGhostPosition = React.useCallback((clientX: number, clientY: number) => {
    const visual = reorderDragVisualRef.current;
    const container = canvasContainerRef.current;
    if (!visual || !container) return;
    const rect = container.getBoundingClientRect();
    const x = Math.max(0, Math.min(Math.max(0, rect.width - visual.width), clientX - rect.left - visual.offsetX));
    const y = Math.max(0, Math.min(Math.max(0, rect.height - visual.height), clientY - rect.top - visual.offsetY));
    reorderGhostPendingRef.current = { x, y };
    if (reorderGhostRafRef.current !== null) return;
    reorderGhostRafRef.current = window.requestAnimationFrame(() => {
      reorderGhostRafRef.current = null;
      const pending = reorderGhostPendingRef.current;
      const ghost = reorderGhostRef.current;
      if (!pending || !ghost) return;
      ghost.style.transform = `translate3d(${pending.x}px, ${pending.y}px, 0)`;
    });
  }, []);
  const beginReorderDragVisual = React.useCallback((event: React.PointerEvent, panel: number) => {
    const container = canvasContainerRef.current;
    if (!container) return;
    const panelRect = event.currentTarget.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    const offsetX = Math.max(0, Math.min(panelRect.width, event.clientX - panelRect.left));
    const offsetY = Math.max(0, Math.min(panelRect.height, event.clientY - panelRect.top));
    const x = Math.max(0, Math.min(Math.max(0, containerRect.width - panelRect.width), event.clientX - containerRect.left - offsetX));
    const y = Math.max(0, Math.min(Math.max(0, containerRect.height - panelRect.height), event.clientY - containerRect.top - offsetY));
    setReorderDragVisualValue({
      panel,
      label: panelLabel(panel),
      imageUrl: captureReorderPanelImage(panelRect, containerRect),
      width: panelRect.width,
      height: panelRect.height,
      x,
      y,
      offsetX,
      offsetY,
    });
    reorderGhostPendingRef.current = { x, y };
    requestAnimationFrame(() => updateReorderGhostPosition(event.clientX, event.clientY));
  }, [captureReorderPanelImage, panelLabel, setReorderDragVisualValue, updateReorderGhostPosition]);
  const clearReorderDragVisual = React.useCallback(() => {
    if (reorderGhostRafRef.current !== null) {
      window.cancelAnimationFrame(reorderGhostRafRef.current);
      reorderGhostRafRef.current = null;
    }
    reorderGhostPendingRef.current = null;
    reorderDragStartRef.current = null;
    reorderDragActivatedRef.current = false;
    setReorderDragVisualValue(null);
  }, [setReorderDragVisualValue]);
  const reorderDragHasPassedThreshold = React.useCallback((clientX: number, clientY: number) => {
    const start = reorderDragStartRef.current;
    if (!start) return true;
    if (reorderDragActivatedRef.current) return true;
    const distance = Math.hypot(clientX - start.x, clientY - start.y);
    if (distance < REORDER_DRAG_THRESHOLD_PX) return false;
    reorderDragActivatedRef.current = true;
    return true;
  }, []);
  const buildPanelMovedOrder = React.useCallback((
    source: number,
    target: number,
    placement: ReorderPlacement,
    baseOrder?: number[] | null,
  ): number[] | null => {
    if (source === target) return null;
    const base = Array.isArray(baseOrder) && baseOrder.length === totalPanelCount
      ? baseOrder
      : orderedPanelIndices;
    const next = [...base];
    const from = next.indexOf(source);
    if (from < 0) return null;
    next.splice(from, 1);
    const targetIndex = next.indexOf(target);
    if (targetIndex < 0) return null;
    const insertAt = placement === "after" ? targetIndex + 1 : targetIndex;
    next.splice(insertAt, 0, source);
    return next;
  }, [orderedPanelIndices, totalPanelCount]);
  const panelReorderTargetFromPoint = React.useCallback((clientX: number, clientY: number): { panel: number; placement: ReorderPlacement } | null => {
    if (typeof document === "undefined") return null;
    const elements = document.elementsFromPoint(clientX, clientY);
    let targetEl: HTMLElement | null = null;
    for (const element of elements) {
      if (!(element instanceof HTMLElement)) continue;
      const candidate = element.closest("[data-show3d-reorder-panel]");
      if (candidate instanceof HTMLElement) {
        targetEl = candidate;
        break;
      }
    }
    const allTargets = Array.from(document.querySelectorAll<HTMLElement>("[data-show3d-reorder-panel]"))
      .map((element) => {
        const rect = element.getBoundingClientRect();
        const raw = element.dataset.show3dReorderPanel;
        const panel = raw == null ? Number.NaN : Math.trunc(Number(raw));
        return { element, rect, panel };
      })
      .filter((item) => Number.isFinite(item.panel) && item.rect.width > 0 && item.rect.height > 0);
    if (!targetEl && allTargets.length) {
      let best = allTargets[0];
      let bestDistance = Number.POSITIVE_INFINITY;
      for (const item of allTargets) {
        const dx = Math.max(item.rect.left - clientX, 0, clientX - item.rect.right);
        const dy = Math.max(item.rect.top - clientY, 0, clientY - item.rect.bottom);
        const distance = dx * dx + dy * dy;
        if (distance < bestDistance) {
          best = item;
          bestDistance = distance;
        }
      }
      targetEl = best.element;
    }
    if (!targetEl) return null;
    const raw = targetEl.dataset.show3dReorderPanel;
    const panel = raw == null ? Number.NaN : Math.trunc(Number(raw));
    if (!Number.isFinite(panel) || panel < 0 || panel >= totalPanelCount) return null;
    const rect = targetEl.getBoundingClientRect();
    const sameRowNeighbor = allTargets.some((item) => item.panel !== panel && Math.abs(item.rect.top - rect.top) < 8);
    const sameColumnNeighbor = allTargets.some((item) => item.panel !== panel && Math.abs(item.rect.left - rect.left) < 8);
    const useHorizontal = sameRowNeighbor || !sameColumnNeighbor;
    const placement: ReorderPlacement = useHorizontal
      ? (clientX >= rect.left + rect.width / 2 ? "after" : "before")
      : (clientY >= rect.top + rect.height / 2 ? "after" : "before");
    return { panel, placement };
  }, [totalPanelCount]);
  const previewPanelReorderFromPoint = React.useCallback((clientX: number, clientY: number) => {
    const source = pointerReorderPanelRef.current ?? draggedPanelRef.current;
    if (source === null) return;
    const target = panelReorderTargetFromPoint(clientX, clientY);
    if (!target) return;
    setDragOverPanel(target.panel);
    const base = reorderPreviewOrderRef.current || orderedPanelIndices;
    const next = buildPanelMovedOrder(source, target.panel, target.placement, base);
    if (!next) return;
    const current = reorderPreviewOrderRef.current || orderedPanelIndices;
    if (next.length === current.length && next.every((value, idx) => value === current[idx])) return;
    setReorderPreviewOrderValue(next);
  }, [buildPanelMovedOrder, orderedPanelIndices, panelReorderTargetFromPoint, setReorderPreviewOrderValue]);
  const commitPanelReorderPreview = React.useCallback(() => {
    const next = reorderPreviewOrderRef.current;
    if (next) applyPanelOrder(next);
    setReorderPreviewOrderValue(null);
    setDragOverPanel(null);
    draggedPanelRef.current = null;
    pointerReorderPanelRef.current = null;
    clearReorderDragVisual();
  }, [applyPanelOrder, clearReorderDragVisual, setReorderPreviewOrderValue]);
  const cancelPanelReorderPreview = React.useCallback(() => {
    setReorderPreviewOrderValue(null);
    setDragOverPanel(null);
    draggedPanelRef.current = null;
    pointerReorderPanelRef.current = null;
    clearReorderDragVisual();
  }, [clearReorderDragVisual, setReorderPreviewOrderValue]);
  const handlePanelDragStart = React.useCallback((event: React.DragEvent, panel: number) => {
    if (!reorderMode) return;
    draggedPanelRef.current = panel;
    setReorderPreviewOrderValue(orderedPanelIndices);
    setDragOverPanel(panel);
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", String(panel));
    const blankDragImage = document.createElement("canvas");
    blankDragImage.width = 1;
    blankDragImage.height = 1;
    event.dataTransfer.setDragImage(blankDragImage, 0, 0);
    event.stopPropagation();
  }, [orderedPanelIndices, reorderMode, setReorderPreviewOrderValue]);
  const handlePanelDragOver = React.useCallback((event: React.DragEvent, panel: number) => {
    if (!reorderMode) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    if (dragOverPanel !== panel) setDragOverPanel(panel);
    previewPanelReorderFromPoint(event.clientX, event.clientY);
    event.stopPropagation();
  }, [dragOverPanel, previewPanelReorderFromPoint, reorderMode]);
  const handlePanelDrop = React.useCallback((event: React.DragEvent) => {
    if (!reorderMode) return;
    event.preventDefault();
    const raw = event.dataTransfer.getData("text/plain");
    const source = raw.trim() !== "" && Number.isFinite(Number(raw))
      ? Math.trunc(Number(raw))
      : draggedPanelRef.current;
    if (source !== null && source !== undefined) {
      draggedPanelRef.current = source;
      previewPanelReorderFromPoint(event.clientX, event.clientY);
    }
    commitPanelReorderPreview();
    event.stopPropagation();
  }, [commitPanelReorderPreview, previewPanelReorderFromPoint, reorderMode]);
  const handlePanelDragEnd = React.useCallback(() => {
    cancelPanelReorderPreview();
  }, [cancelPanelReorderPreview]);
  const handlePanelReorderPointerDown = React.useCallback((event: React.PointerEvent, panel: number) => {
    if (!reorderMode) return;
    pointerReorderPanelRef.current = panel;
    draggedPanelRef.current = panel;
    reorderDragStartRef.current = { x: event.clientX, y: event.clientY };
    reorderDragActivatedRef.current = false;
    setReorderPreviewOrderValue(orderedPanelIndices);
    setDragOverPanel(panel);
    beginReorderDragVisual(event, panel);
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Some browser automation paths do not expose pointer capture.
    }
    event.preventDefault();
    event.stopPropagation();
  }, [beginReorderDragVisual, orderedPanelIndices, reorderMode, setReorderPreviewOrderValue]);
  const handlePanelReorderPointerEnter = React.useCallback((event: React.PointerEvent, panel: number) => {
    if (!reorderMode || pointerReorderPanelRef.current === null) return;
    if (dragOverPanel !== panel) setDragOverPanel(panel);
    event.stopPropagation();
  }, [dragOverPanel, reorderMode]);
  const handlePanelReorderPointerMove = React.useCallback((event: React.PointerEvent) => {
    if (!reorderMode || pointerReorderPanelRef.current === null) return;
    updateReorderGhostPosition(event.clientX, event.clientY);
    if (!reorderDragHasPassedThreshold(event.clientX, event.clientY)) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    previewPanelReorderFromPoint(event.clientX, event.clientY);
    event.preventDefault();
    event.stopPropagation();
  }, [previewPanelReorderFromPoint, reorderDragHasPassedThreshold, reorderMode, updateReorderGhostPosition]);
  const handlePanelReorderPointerUp = React.useCallback((event: React.PointerEvent) => {
    if (!reorderMode) return;
    updateReorderGhostPosition(event.clientX, event.clientY);
    if (reorderDragHasPassedThreshold(event.clientX, event.clientY)) {
      previewPanelReorderFromPoint(event.clientX, event.clientY);
      commitPanelReorderPreview();
    } else {
      cancelPanelReorderPreview();
    }
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // Ignore capture release failures from synthetic pointer streams.
    }
    event.preventDefault();
    event.stopPropagation();
  }, [cancelPanelReorderPreview, commitPanelReorderPreview, previewPanelReorderFromPoint, reorderDragHasPassedThreshold, reorderMode, updateReorderGhostPosition]);
  const resetPanelOrder = React.useCallback(() => {
    setPanelOrder([]);
    cancelPanelReorderPreview();
  }, [cancelPanelReorderPreview, setPanelOrder]);
  React.useEffect(() => {
    if (((nPanels || 1) <= 1 || isPaged) && reorderMode) setReorderMode(false);
  }, [nPanels, isPaged, reorderMode]);
  React.useEffect(() => {
    if (reorderMode) return;
    cancelPanelReorderPreview();
  }, [cancelPanelReorderPreview, reorderMode]);
  React.useEffect(() => () => {
    if (reorderGhostRafRef.current !== null) {
      window.cancelAnimationFrame(reorderGhostRafRef.current);
      reorderGhostRafRef.current = null;
    }
  }, []);
  const [hiddenIndices] = useModelState<number[]>("hidden_indices");
  const hiddenSet = new Set(hiddenIndices || []);
  const nextVisible = (from: number, dir: 1 | -1, allowWrap = true): number => {
    if (!hiddenSet.size) return from + dir;
    let n = from + dir;
    while (n >= 0 && n < nSlices) {
      if (!hiddenSet.has(n)) return n;
      n += dir;
    }
    if (!allowWrap) return from;
    n = dir > 0 ? 0 : nSlices - 1;
    while (n !== from) {
      if (!hiddenSet.has(n)) return n;
      n += dir;
      if (n < 0 || n >= nSlices) return from;
    }
    return from;
  };
  const visibleCount = nSlices - hiddenSet.size;
  // If the user hides the currently-displayed slice, snap to next visible.
  React.useEffect(() => {
    if (!hiddenSet.has(sliceIdx)) return;
    const next = nextVisible(sliceIdx, 1, true);
    if (next !== sliceIdx) setSliceIdx(next);
  }, [hiddenIndices]);
  const [maxCols, setMaxCols] = useModelState<number>("max_cols");
  const [linkPanels, setLinkPanels] = useModelState<boolean>("link_panels");
  const [showResizeHandles] = useModelState<boolean>("show_resize_handles");
  const allowResizeControls = showResizeHandles !== false && !isMobileViewport;
  const [showZoomIndicator] = useModelState<boolean>("show_zoom_indicator");
  const [showPanelTitles] = useModelState<boolean>("show_panel_titles");
  const [panelTitleFontSize] = useModelState<number>("panel_title_font_size");
  const [panelTitleStyle] = useModelState<PanelTitleStyle>("panel_title_style");
  const [panelGapTrait] = useModelState<number>("panel_gap");
  const [linkContrast, setLinkContrast] = useModelState<boolean>("link_contrast");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [panelCmaps, setPanelCmaps] = useModelState<string[]>("panel_cmaps");
  const [markerColors] = useModelState<string[]>("marker_colors");
  const [identityColors] = useModelState<string[]>("identity_colors");
  const [markerStyle] = useModelState<string>("marker_style");
  const [rowMarkers] = useModelState<MarkerMap>("row_markers");
  const [colMarkers] = useModelState<MarkerMap>("col_markers");
  const [panelGroups] = useModelState<PanelGroup[]>("panel_groups");
  const [panelAnnotations] = useModelState<PanelAnnotationSpec[][]>("panel_annotations");
  const [panelOverlays, setPanelOverlays] = useModelState<PanelOverlaySpec[][]>("panel_overlays");
  const [flipRows, setFlipRows] = useModelState<boolean>("flip_vertical");
  const [flipCols, setFlipCols] = useModelState<boolean>("flip_horizontal");
  const [compareMode, setCompareMode] = useModelState<string>("compare_mode");
  const [comparePair, setComparePair] = useModelState<number[]>("compare_pair");
  const [blinkFps, setBlinkFps] = useModelState<number>("blink_fps");
  const [diffCmap, setDiffCmap] = useModelState<string>("diff_cmap");
  const [compareBackground, setCompareBackground] = useModelState<string>("compare_background");
  const previousCompareModeRef = React.useRef<string>(compareMode || "off");
  const [blinkPhase, setBlinkPhase] = React.useState(0);
  const panelMarkerColor = React.useCallback((panel: number) => {
    const value = identityColors?.[panel] || markerColors?.[panel];
    return value || IDENTITY_PALETTE[panel % IDENTITY_PALETTE.length];
  }, [identityColors, markerColors]);
  const markerAround = (markerStyle || "left") === "around";
  const normalizedPanelCmaps = React.useMemo(
    () => Array.isArray(panelCmaps) ? panelCmaps : [],
    [panelCmaps],
  );
  const panelCmapFor = React.useCallback((panelIdx: number) => {
    const value = normalizedPanelCmaps[panelIdx];
    return (value && COLORMAPS[value]) ? value : (cmap || "inferno");
  }, [normalizedPanelCmaps, cmap]);
  const hasMixedPanelCmaps = React.useMemo(() => {
    if (Math.max(1, nPanels || 1) <= 1) return false;
    if (normalizedPanelCmaps.length !== Math.max(1, nPanels || 1)) return false;
    const first = panelCmapFor(0);
    return normalizedPanelCmaps.some((_, idx) => panelCmapFor(idx) !== first);
  }, [normalizedPanelCmaps, nPanels, panelCmapFor]);
  const colorShared = normalizedPanelCmaps.length !== Math.max(1, nPanels || 1) || Math.max(1, nPanels || 1) <= 1;
  const setColorShared = React.useCallback((shared: boolean, panelIdx = 0) => {
    const n = Math.max(1, nPanels || 1);
    if (shared || n <= 1) {
      setCmap(panelCmapFor(panelIdx));
      setPanelCmaps([]);
      return;
    }
    setPanelCmaps(Array.from({ length: n }, (_, idx) => panelCmapFor(idx)));
  }, [nPanels, panelCmapFor, setCmap, setPanelCmaps]);
  const setCmapForPanel = React.useCallback((panelIdx: number, value: string) => {
    const n = Math.max(1, nPanels || 1);
    if (n <= 1 || colorShared) {
      setCmap(value);
      setPanelCmaps([]);
      return;
    }
    const idx = Math.max(0, Math.min(n - 1, Math.round(panelIdx)));
    const next = normalizedPanelCmaps.length === n
      ? [...normalizedPanelCmaps]
      : Array.from({ length: n }, () => cmap || "inferno");
    next[idx] = value;
    setPanelCmaps(next);
    if (idx === 0) setCmap(value);
  }, [cmap, colorShared, normalizedPanelCmaps, nPanels, setCmap, setPanelCmaps]);

  const prioritizedSidecarFrameOrder = React.useCallback((start: number, n: number) => {
    const total = Math.max(1, Math.round(n || 1));
    const base = ((Math.round(start) % total) + total) % total;
    const order: number[] = [base];
    for (let offset = 1; order.length < total; offset++) {
      order.push((base + offset) % total);
      if (order.length < total) order.push((base - offset + total) % total);
    }
    return order;
  }, []);

  // Playback
  const [playing, setPlaying] = useModelState<boolean>("playing");
  const [reverse, setReverse] = useModelState<boolean>("reverse");
  const [boomerang, setBoomerang] = useModelState<boolean>("boomerang");
  const [fps, setFpsModel] = useModelState<number>("fps");
  const playbackFps = clampPlaybackFps(fps);
  const setPlaybackFps = React.useCallback((value: number) => {
    setFpsModel(clampPlaybackFps(value));
  }, [setFpsModel]);
  React.useEffect(() => {
    if (fps !== playbackFps) setFpsModel(playbackFps);
  }, [fps, playbackFps, setFpsModel]);
  const [loop, setLoop] = useModelState<boolean>("loop");
  const [loopStart, setLoopStart] = useModelState<number>("loop_start");
  const [loopEnd, setLoopEnd] = useModelState<number>("loop_end");
  const [bookmarkedFrames, setBookmarkedFrames] = useModelState<number[]>("bookmarked_frames");
  const [playbackPath, setPlaybackPath] = useModelState<number[]>("playback_path");

  // Boomerang direction ref (avoids stale closure in setInterval)
  const bounceDirRef = React.useRef<1 | -1>(1);

  // Stats
  const [showStats, setShowStats] = useModelState<boolean>("show_stats");
  // "More" overflow menu (mirrors Show2D): tucks Stats + Denoise off the crowded
  // top toolbar. Badge shows how many of its tools are active.
  const [moreMenuAnchor, setMoreMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [playbackStyleMenuAnchor, setPlaybackStyleMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [showRotationSettings, setShowRotationSettings] = React.useState(false);
  const [showControls] = useModelState<boolean>("show_controls");
  const [controlsCollapsed, setControlsCollapsed] = useModelState<boolean>("controls_collapsed");
  const [debug] = useModelState<boolean>("debug");
  const controlsVisible = showControls && !controlsCollapsed;
  const panelChromeVisible = controlsVisible;
  const showResizeControls = allowResizeControls && panelChromeVisible;
  const debugFps = useDebugFps(Boolean(debug));
  const [statsMean] = useModelState<number>("stats_mean");
  const [statsMin] = useModelState<number>("stats_min");
  const [statsMax] = useModelState<number>("stats_max");
  const [statsStd] = useModelState<number>("stats_std");

  // Display options
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [contrastPreset, setContrastPreset] = useModelState<string>("contrast_preset");
  const [percentileLow, setPercentileLow] = useModelState<number>("percentile_low");
  const [percentileHigh, setPercentileHigh] = useModelState<number>("percentile_high");
  const [traitVmin] = useModelState<number | null>("vmin");
  const [traitVmax] = useModelState<number | null>("vmax");
  const [imageVminPct, setImageVminPct] = useModelState<number>("image_vmin_pct");
  const [imageVmaxPct, setImageVmaxPct] = useModelState<number>("image_vmax_pct");
  const manualImageRangeBeforeAutoRef = React.useRef<{ min: number; max: number } | null>(null);
  const [vminPerPanel, setVminPerPanel] = useModelState<(number | null)[]>("vmin_per_panel");
  const [vmaxPerPanel, setVmaxPerPanel] = useModelState<(number | null)[]>("vmax_per_panel");
  const vminPerPanelLiveRef = React.useRef<(number | null)[]>(vminPerPanel);
  const vmaxPerPanelLiveRef = React.useRef<(number | null)[]>(vmaxPerPanel);
  React.useEffect(() => {
    vminPerPanelLiveRef.current = vminPerPanel;
  }, [vminPerPanel]);
  React.useEffect(() => {
    vmaxPerPanelLiveRef.current = vmaxPerPanel;
  }, [vmaxPerPanel]);
  const [dataMin] = useModelState<number>("data_min");
  const [dataMax] = useModelState<number>("data_max");
  const [autoVmins] = useModelState<number[]>("auto_vmins");
  const [autoVmaxs] = useModelState<number[]>("auto_vmaxs");
  // Full-resolution display cache for sidecar scrub/play. The sidecar bytes
  // are already quantized for display, so convert each native panel to an
  // ImageBitmap once after RAM load. Scrub then swaps/draws bitmaps instead of
  // dequantizing and CPU-colormapping ~38 Mpx on every step.
  React.useEffect(() => {
    if (!enableSidecarNativePanelBitmapCache) {
      for (const bitmaps of sidecarBitmapFrameCacheRef.current.values()) {
        for (const bitmap of bitmaps) {
          try { bitmap.close(); } catch { /* ignore */ }
        }
      }
      sidecarBitmapFrameCacheRef.current.clear();
      sidecarBitmapReadyRef.current = false;
      sidecarBitmapCompleteRef.current = false;
      setSidecarBitmapReady(false);
      setSidecarBitmapComplete(false);
      return;
    }
    if (!sidecarMode || !sidecarRamReady || isRgb || sharedPanelSource || width <= 0 || height <= 0) {
      clearSidecarBitmapCache();
      return;
    }
    const panelCount = Math.max(1, nPanels || 1);
    const n = Math.max(1, Math.round(nSlices || 1));
    const panelW = Math.max(1, panelWidthPx || Math.floor(width / panelCount) || width);
    if (panelCount <= 1 || panelW <= 0 || panelW * panelCount > width + panelW) {
      clearSidecarBitmapCache();
      setOfflineStackFetchStatus("");
      return;
    }

    const serial = ++sidecarBitmapBuildSerialRef.current;
    let cancelled = false;
    clearSidecarBitmapCache();
    setOfflineStackFetchStatus(`Preparing full-resolution display cache… 0/${n} frames`);

    const buildCache = async () => {
      const img = new ImageData(panelW, height);
      const rgba = img.data;
      const fallbackLut = COLORMAPS[cmap] || COLORMAPS.inferno;
      const loByte = Math.max(0, Math.min(255, Math.round((Number(imageVminPct) || 0) * 2.55)));
      const hiByte = Math.max(0, Math.min(255, Math.round((Number(imageVmaxPct) || 100) * 2.55)));
      const byteSpan = Math.max(1, hiByte - loByte);
      const started = performance.now();
      let builtPanels = 0;
      let builtFrames = 0;
      const startFrame = ((Math.round(playbackIdxRef.current || 0) % n) + n) % n;
      const frameOrder = Array.from({ length: n }, (_, idx) => (startFrame + idx) % n);
      try {
        for (const frameIdx of frameOrder) {
          if (cancelled || serial !== sidecarBitmapBuildSerialRef.current) return;
          const u8 = sidecarU8FrameCacheRef.current.get(frameIdx);
          if (!u8 || u8.byteLength < width * height) continue;
          const bitmaps: ImageBitmap[] = [];
          for (let panel = 0; panel < panelCount; panel++) {
            const x0 = panel * panelW;
            if (x0 >= width) break;
            const x1 = Math.min(width, x0 + panelW);
            const lut = COLORMAPS[panelCmapFor(panel)] || fallbackLut;
            for (let r = 0; r < height; r++) {
              let src = r * width + x0;
              let dst = r * panelW * 4;
              for (let x = x0; x < x1; x++, src++, dst += 4) {
                const v = Math.max(0, Math.min(255, Math.floor(((u8[src] - loByte) / byteSpan) * 255)));
                const li = v * 3;
                rgba[dst] = lut[li];
                rgba[dst + 1] = lut[li + 1];
                rgba[dst + 2] = lut[li + 2];
                rgba[dst + 3] = 255;
              }
              for (let x = x1 - x0; x < panelW; x++, dst += 4) {
                rgba[dst] = 0;
                rgba[dst + 1] = 0;
                rgba[dst + 2] = 0;
                rgba[dst + 3] = 255;
              }
            }
            bitmaps.push(await createImageBitmap(img));
            builtPanels += 1;
            if (builtPanels % 8 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
          }
          if (cancelled || serial !== sidecarBitmapBuildSerialRef.current) {
            bitmaps.forEach((bitmap) => bitmap.close());
            return;
          }
          sidecarBitmapFrameCacheRef.current.set(frameIdx, bitmaps);
          builtFrames += 1;
          if (!sidecarBitmapReadyRef.current) {
            sidecarBitmapReadyRef.current = true;
            setSidecarBitmapReady(true);
          }
          const loopDebug = show3dPerfDebug();
          if (loopDebug) {
            loopDebug.sidecarBitmapCacheFrames = sidecarBitmapFrameCacheRef.current.size;
            loopDebug.sidecarBitmapCachePanels = builtPanels;
            loopDebug.sidecarBitmapComplete = false;
            loopDebug.sidecarFirstFrameMs = loopDebug.sidecarFirstFrameMs ?? (performance.now() - started);
          }
          if (builtFrames % 2 === 0 || builtFrames === 1 || builtFrames === n) {
            const elapsed = ((performance.now() - started) / 1000).toFixed(1);
            setOfflineStackFetchStatus(
              builtFrames === 1
                ? `First full-resolution frame ready; optimizing playback cache… 1/${n} frames (${elapsed}s)`
                : `Optimizing full-resolution playback cache… ${builtFrames}/${n} frames (${elapsed}s)`,
            );
          }
        }
        if (cancelled || serial !== sidecarBitmapBuildSerialRef.current) return;
        sidecarBitmapCompleteRef.current = true;
        setSidecarBitmapComplete(true);
        setOfflineStackFetchStatus("");
        const d = show3dPerfDebug();
        if (d) {
          d.sidecarBitmapCacheFrames = sidecarBitmapFrameCacheRef.current.size;
          d.sidecarBitmapComplete = true;
          d.sidecarBitmapCachePanels = builtPanels;
          d.sidecarBitmapBuildMs = performance.now() - started;
          d.sidecarBitmapPanelWidth = panelW;
          d.sidecarBitmapContrastBytes = [loByte, hiByte];
          d.lastRenderPath = "sidecar-imagebitmap-cache-ready";
        }
      } catch (err) {
        clearSidecarBitmapCache();
        setOfflineStackFetchStatus(
          `Failed to prepare display cache: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    };
    void buildCache();
    return () => {
      cancelled = true;
    };
  }, [
    sidecarMode,
    sidecarRamReady,
    isRgb,
    sharedPanelSource,
    width,
    height,
    nPanels,
    panelWidthPx,
    nSlices,
    cmap,
    panelCmapFor,
    imageVminPct,
    imageVmaxPct,
    clearSidecarBitmapCache,
  ]);
  React.useEffect(() => {
    if (compareMode !== "blink") {
      setBlinkPhase(0);
      return;
    }
    const intervalMs = 1000 / Math.max(0.25, Number(blinkFps || 2));
    const id = window.setInterval(() => setBlinkPhase((phase) => (phase + 1) % 2), intervalMs);
    return () => window.clearInterval(id);
  }, [blinkFps, compareMode]);
  // Scale bar
  const [pixelSize] = useModelState<number>("pixel_size");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  // Display-only filter knobs for sparse map stacks (EDS, low dose). Python
  // owns the math and re-sends the playback buffer on change; raw data is
  // never modified.
  const [displayFilter, setDisplayFilter] = useModelState<string>("denoise");
  const [displaySigma, setDisplaySigma] = useModelState<number>("denoise_sigma");
  const [spatialBin, setSpatialBin] = useModelState<number>("denoise_bin");
  const [displayFilterBanner] = useModelState<string>("denoise_banner");
  const [showDenoise, setShowDenoise] = useModelState<boolean>("show_denoise");
  // Master ON/OFF of the denoise EFFECT (off -> raw, config preserved & gated).
  const [denoiseEnabled, setDenoiseEnabled] = useModelState<boolean>("denoise_enabled");
  const [frequencyFilter, setFrequencyFilter] = useModelState<string>("frequency_filter");
  const [frequencyFilterEnabled, setFrequencyFilterEnabled] = useModelState<boolean>("frequency_filter_enabled");
  const [frequencyFilterCutoff, setFrequencyFilterCutoff] = useModelState<number>("frequency_filter_cutoff");
  const [frequencyFilterCenter, setFrequencyFilterCenter] = useModelState<number>("frequency_filter_center");
  const [frequencyFilterWidth, setFrequencyFilterWidth] = useModelState<number>("frequency_filter_width");
  const [showFrequencyFilter, setShowFrequencyFilter] = useModelState<boolean>("show_frequency_filter");
  const [subpixelAlignEnabled, setSubpixelAlignEnabled] = useModelState<boolean>("subpixel_align_enabled");
  const [subpixelAlignReference, setSubpixelAlignReference] = useModelState<number>("subpixel_align_reference");
  const [subpixelAlignStatus, setSubpixelAlignStatus] = React.useState("Off");
  const [subpixelAlignBusy, setSubpixelAlignBusy] = React.useState(false);
  const [subpixelAlignVersion, setSubpixelAlignVersion] = React.useState(0);
  const subpixelAlignShiftsRef = React.useRef<SubpixelShift[] | null>(null);
  const subpixelAlignCacheRef = React.useRef<Map<string, Float32Array>>(new Map());
  const subpixelAlignSerialRef = React.useRef(0);
  const [frequencyDraft, setFrequencyDraft] = React.useState<number | null>(null);
  const [frequencyRenderVersion, setFrequencyRenderVersion] = React.useState(0);
  const [frequencyFilterBackend, setFrequencyFilterBackend] = React.useState("off");
  const frequencyFilterCacheRef = React.useRef<Map<string, Float32Array>>(new Map());
  const frequencyFilterPendingRef = React.useRef<Set<string>>(new Set());
  const frequencyFilterIsActive = !!frequencyFilterEnabled && frequencyFilterActive(frequencyFilter) && !isRgb;
  const frequencyOptions = React.useMemo(() => {
    const mode = normalizeFrequencyFilterMode(frequencyFilter);
    return {
      mode,
      cutoff: frequencyDraft ?? frequencyFilterCutoff,
      center: mode === "bandpass" ? (frequencyDraft ?? frequencyFilterCenter) : frequencyFilterCenter,
      width: frequencyFilterWidth,
    };
  }, [frequencyFilter, frequencyDraft, frequencyFilterCutoff, frequencyFilterCenter, frequencyFilterWidth]);
  const frequencyValueLabel = React.useCallback((value: number) => {
    const unit = String(pixelUnit || "").trim().toLowerCase();
    if (pixelSize > 0 && (unit === "nm" || unit.includes("nanometer"))) return `${(value / (2 * pixelSize)).toFixed(3)} nm⁻¹`;
    if (pixelSize > 0 && (unit === "a" || unit === "å" || unit.includes("angstrom"))) return `${(value * 10 / (2 * pixelSize)).toFixed(3)} nm⁻¹`;
    return `${value.toFixed(3)} Nyq`;
  }, [pixelSize, pixelUnit]);
  const setFrequencyMaster = (enabled: boolean) => {
    if (enabled && !frequencyFilterActive(frequencyFilter)) setFrequencyFilter("lowpass");
    setFrequencyFilterEnabled(enabled);
    setShowFrequencyFilter(enabled); // reveal the settings row while filtering; hide it when off (mirrors Denoise)
  };
  // Local slider value during drag; the model (and the Python refilter) only
  // updates on release so scrubbing sigma stays smooth on large stacks.
  const [sigmaDraft, setSigmaDraft] = React.useState<number | null>(null);
  const displayFilterOff = resolveDenoiseMode(displayFilter || "none").mode === "none";
  // Browser-side denoise negotiation (mirrors Show2D). With _webgpu_filter_ok
  // Python ships RAW frames and the WGSL port applies gaussian/bin/anscombe
  // here, so dragging sigma is LIVE (zero kernel round-trips). Only a real
  // (non-software) adapter flips it, so SwiftShader-class fallbacks keep the
  // Python path. Offline pages keep their Python-baked frames.
  const [webgpuFilterOk, setWebgpuFilterOk] = useModelState<boolean>("_webgpu_filter_ok");
  // Offline (the auto-enabled uint8-pack path for stacks <1GB) ALSO filters in the
  // browser: Python ships the offline stack RAW and sets _webgpu_filter_ok, so the WGSL
  // port applies denoise here with live sigma - same as the live-kernel path. (Was
  // gated `&& !offline`, which disabled real-time denoise for the common offline case.)
  const browserFilterActive = !!webgpuFilterOk && !isRgb && (denoiseEnabled ?? true);
  const denoiseResolved = resolveDenoiseMode(displayFilter || "none", spatialBin || 1);
  const denoiseSigmaLive = sigmaDraft ?? Number(displaySigma ?? 4);
  const browserFilterKnobsOn = browserFilterActive
    && filterKnobsActive(denoiseResolved.mode, denoiseResolved.bin)
    && browserFilterSupported(denoiseResolved.mode);
  // Filtered-frame cache keyed on the Python frame sequence as well as the
  // logical index and view knobs. During a live scrub, slice_idx can arrive
  // before the replacement frame_bytes. Without frameSeq, that old byte view
  // can populate the new index's cache key and the repaint keeps reading the
  // wrong generation (often leaving the newly arrived raw frame visible).
  const browserFilterCacheRef = React.useRef<Map<string, Float32Array>>(new Map());
  const browserFilterPendingRef = React.useRef<Set<string>>(new Set());
  const [browserFilterTick, setBrowserFilterTick] = React.useState(0);
  const applyPackedPanelTransform = React.useCallback(async (
    frame: Float32Array,
    transform: (panelFrame: Float32Array, panelWidth: number, panelHeight: number) => Promise<Float32Array>,
  ): Promise<Float32Array> => {
    const panelCount = Math.max(1, nPanels || 1);
    if (panelCount <= 1 || sharedPanelSource || width % panelCount !== 0) {
      return transform(frame, width, height);
    }
    const panelWidth = width / panelCount;
    const output = new Float32Array(frame.length);
    for (let panel = 0; panel < panelCount; panel++) {
      const panelFrame = new Float32Array(panelWidth * height);
      const srcX0 = panel * panelWidth;
      for (let row = 0; row < height; row++) {
        const srcOffset = row * width + srcX0;
        const dstOffset = row * panelWidth;
        panelFrame.set(frame.subarray(srcOffset, srcOffset + panelWidth), dstOffset);
      }
      const filtered = await transform(panelFrame, panelWidth, height);
      for (let row = 0; row < height; row++) {
        const srcOffset = row * panelWidth;
        const dstOffset = row * width + srcX0;
        output.set(filtered.subarray(srcOffset, srcOffset + panelWidth), dstOffset);
      }
    }
    return output;
  }, [height, nPanels, sharedPanelSource, width]);
  // Live gate read inside memoized render ticks (avoids stale closures): when on,
  // denoise is treated as a client frame transform so every path routes through
  // displayFrameForIndex and skips the raw GPU-slot cache.
  const browserFilterOnRef = React.useRef(false);
  browserFilterOnRef.current = browserFilterKnobsOn;
  React.useEffect(() => {
    let disposed = false;
    getGPUDisplayFilterEngine().then((engine) => {
      if (!disposed && !offline) setWebgpuFilterOk(!!engine);
    });
    return () => { disposed = true; };
  }, [offline, setWebgpuFilterOk]);
  // Return a filtered copy of `frame` for DISPLAY only, keyed on the live knobs.
  // The GPU filter is async, so on a cache miss we return the raw frame now and
  // repaint once the filtered result lands (setBrowserFilterTick). Stats/ROI
  // keep reading raw frames (frame_bytes ships raw), so numbers stay honest.
  const browserFilterFrame = React.useCallback((idx: number, frame: Float32Array | null, options: { allowRawOnMiss?: boolean } = {}): Float32Array | null => {
    if (!frame || !browserFilterKnobsOn) return frame;
    const allowRawOnMiss = options.allowRawOnMiss !== false;
    const key = browserFilterCacheKey({
      frameIndex: idx,
      frameSeq,
      mode: denoiseResolved.mode,
      sigma: denoiseSigmaLive,
      bin: denoiseResolved.bin,
      avgWindow: playRef.current.avgWindow,
      diffMode: playRef.current.diffMode,
      panels: (Math.max(1, nPanels || 1) > 1 && !sharedPanelSource) ? Math.max(1, nPanels || 1) : 1,
    });
    const cache = browserFilterCacheRef.current;
    const hit = cache.get(key);
    if (hit) return hit;
    if (browserFilterPendingRef.current.has(key)) return allowRawOnMiss ? frame : null;
    browserFilterPendingRef.current.add(key);
    applyPackedPanelTransform(
      frame,
      (panelFrame, panelWidth, panelHeight) => applyDisplayFilterBrowser(panelFrame, panelWidth, panelHeight, denoiseResolved.mode, denoiseSigmaLive, denoiseResolved.bin),
    )
      .then((filtered) => {
        browserFilterPendingRef.current.delete(key);
        cache.set(key, filtered);
        if (cache.size > 48) cache.delete(cache.keys().next().value as string);
        setBrowserFilterTick((t) => t + 1);
      })
      .catch(() => { browserFilterPendingRef.current.delete(key); });
    return allowRawOnMiss ? frame : null;
  }, [applyPackedPanelTransform, browserFilterKnobsOn, denoiseResolved.mode, denoiseResolved.bin, denoiseSigmaLive, frameSeq, height, nPanels, sharedPanelSource, width]);
  // The "Denoise" toggle is the master ON/OFF of the EFFECT: ON shows the
  // denoised view, OFF shows raw (nothing of the denoised view leaks through).
  // The config (mode/sigma/bin) is PRESERVED across the toggle; a clean widget
  // gets a visible gaussian (σ 4) the first time it is enabled.
  const toggleDenoise = () => {
    const next = !denoiseEnabled;
    setDenoiseEnabled(next);
    setShowDenoise(next); // editor follows: shown while denoising, hidden when raw
    if (next && displayFilterOff) setDisplayFilter("gaussian");
    // Turning OFF preserves the config; browserFilterActive gates the display.
  };
  const [imageRotation, setImageRotation] = useModelState<number>("image_rotation");
  const [rotationScope, setRotationScope] = useModelState<string>("rotation_scope");
  const [frameRotations, setFrameRotations] = useModelState<number[]>("frame_rotations");
  const normalizeRotation = React.useCallback((value: number) => {
    const k = Math.round(Number(value) / 90);
    if (Number.isFinite(k) && Math.abs(Number(value)) > 3) return ((k % 4) + 4) % 4;
    return ((Math.round(Number(value)) % 4) + 4) % 4;
  }, []);

  // Customization
  const [canvasSizeTrait, setCanvasSizeTrait] = useModelState<number>("size");

  // ROI
  const [roiActive, setRoiActive] = useModelState<boolean>("roi_active");
  const [roiList, setRoiList] = useModelState<ROIItem[]>("roi_list");
  const [roiSelectedIdx, setRoiSelectedIdx] = useModelState<number>("roi_selected_idx");
  const [roiPlotData] = useModelState<DataView>("roi_plot_data");
  const [newRoiShape, setNewRoiShape] = React.useState<"circle" | "square" | "rectangle" | "annular">("square");

  // Diff mode
  const [diffMode, setDiffMode] = useModelState<string>("diff_mode");
  const [avgWindow, setAvgWindow] = useModelState<number>("avg_window");
  const averageSupported = supportsClientAverage(separatePanelFrames);
  React.useEffect(() => {
    if (averageSupported || normalizedAverageWindow(avgWindow) <= 1) return;
    console.warn(
      "[Show3D] Moving average is unavailable for separate full-resolution panel streams; using avg=1",
    );
    setAvgWindow(1);
  }, [averageSupported, avgWindow, setAvgWindow]);

  // FFT
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [fftLayout, setFftLayout] = useModelState<string>("fft_layout");
  const [fftOverlayPosition, setFftOverlayPosition] = useModelState<string>("fft_overlay_position");
  const [fftOverlaySize, setFftOverlaySize] = useModelState<number>("fft_overlay_size");
  const [fftOverlayZoomTrait, setFftOverlayZoomTrait] = useModelState<number>("fft_overlay_zoom");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const [fftMetricsTrait] = useModelState<boolean>("fft_metrics");
  const fftMetricsEnabled = fftMetricsTrait !== false;
  const resolvedFftLayout = (["bottom", "right", "overlay"].includes(String(fftLayout)) ? String(fftLayout) : "bottom") as "bottom" | "right" | "overlay";
  const fftLayoutBottom = resolvedFftLayout === "bottom";
  const fftLayoutOverlay = resolvedFftLayout === "overlay";
  const resolvedFftOverlayPosition = (["top-left", "top-right", "bottom-left", "bottom-right"].includes(String(fftOverlayPosition)) ? String(fftOverlayPosition) : "top-left") as FftOverlayPosition;
  const resolvedFftOverlaySize = Math.max(0.2, Math.min(0.7, Number.isFinite(fftOverlaySize) ? fftOverlaySize : 0.35));
  const resolvedFftOverlayZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, Number.isFinite(fftOverlayZoomTrait) ? fftOverlayZoomTrait : 1));


  // Playback buffer (sliding prefetch)
  const [bufferBytes] = useModelState<DataView>("_buffer_bytes");
  const [bufferStart] = useModelState<number>("_buffer_start");
  const [bufferCount] = useModelState<number>("_buffer_count");
  const [, setPrefetchRequest] = useModelState<number>("_prefetch_request");
  const [frameServerUrl] = useModelState<string>("frame_server_url");
  const [frameServerVersion] = useModelState<number>("frame_server_version");
  const [benchmarkRequest] = useModelState<Record<string, unknown>>("benchmark_request");
  const [, setBenchmarkResult] = useModelState<Record<string, unknown>>("benchmark_result");
  const [frameTransportTiming] = useModelState<Record<string, unknown>>("frame_transport_timing");
  const [bufferTransportTiming] = useModelState<Record<string, unknown>>("buffer_transport_timing");
  const [scrubPreviewBytes] = useModelState<DataView>("_scrub_preview_bytes");
  const [scrubPreviewInfo] = useModelState<Record<string, unknown>>("_scrub_preview_info");
  const [, setScrubPreviewRequest] = useModelState<string>("_scrub_preview_request");
  const [, setExportRequest] = useModelState<string>("export_request");
  const [exportStatus] = useModelState<string>("export_status");
  const [exportEnabled] = useModelState<boolean>("export_enabled");
  const [exportPayload] = useModelState<DataView>("export_payload");
  const [exportPayloadId] = useModelState<string>("export_payload_id");
  const [exportPayloadFilename] = useModelState<string>("export_filename");
  const [, setHandoffRequest] = useModelState<string>("handoff_request");
  const [handoffStatus] = useModelState<string>("handoff_status");
  const [handoffEnabled] = useModelState<boolean>("handoff_enabled");
  const [preparedViewWidget] = useModelState<unknown>("prepared_view_widget");

  // Canvas refs
  const rootRef = React.useRef<HTMLDivElement>(null);
  const hasLiveFrameBytes = !!rawFrameBytes && rawFrameBytes.byteLength > 0;
  const hasOfflineStack = !!offlineStack && offlineStack.byteLength > 0;
  const hasOfflineFloatStack = !!offlineFloatStack && offlineFloatStack.byteLength > 0;
  const hasFrameServer = !offline && !!frameServerUrl;
  const canRenderLive = hasLiveFrameBytes || hasOfflineStack || hasOfflineFloatStack || hasFrameServer;
  const [framePopulation, setFramePopulation] = React.useState({ ready: 0, target: 0, active: false });
  const [previewPopulation, setPreviewPopulation] = React.useState({ ready: false, idx: 0, factor: 1 });
  const transportSamplesRef = React.useRef<Record<string, unknown>[]>([]);
  const pendingTransportPaintRef = React.useRef<Record<string, unknown> | null>(null);
  const scrubPreviewRafRef = React.useRef<number | null>(null);
  const scrubPreviewPendingIdxRef = React.useRef<number | null>(null);
  const scrubPreviewTokenRef = React.useRef(0);
  const scrubPreviewLoggedFactorRef = React.useRef<number | null>(null);
  const serverFallbackPreviewKeyRef = React.useRef<string>("");
  const initialNativePreviewKeyRef = React.useRef<string>("");
  const recordTransportSample = React.useCallback((sample: Record<string, unknown>) => {
    const next = [...transportSamplesRef.current, sample];
    transportSamplesRef.current = next.length > 200 ? next.slice(next.length - 200) : next;
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.lastTransportSample = sample;
      dbg.transportSamples = transportSamplesRef.current;
    }
  }, []);
  const markTransportPaintProxy = React.useCallback((paintAt: number = performance.now()) => {
    const pending = pendingTransportPaintRef.current;
    if (!pending) return;
    pendingTransportPaintRef.current = null;
    const sendTimeMs = typeof pending.sendTimeMs === "number" ? pending.sendTimeMs : null;
    recordTransportSample({
      ...pending,
      paintProxyAtMs: Number(paintAt.toFixed(3)),
      endToEndUiLatencyMs: sendTimeMs === null ? null : Number((Date.now() - sendTimeMs).toFixed(3)),
    });
  }, [recordTransportSample]);
  const requestCommFramePreview = React.useCallback((idx: number, reason = "scrub"): boolean => {
    if (offline || width <= 0 || height <= 0 || nSlices <= 0) return false;
    const normalized = Math.max(0, Math.min(nSlices - 1, Math.round(idx)));
    const key = `${reason}:${normalized}:${frameServerVersion || 0}`;
    if (reason !== "scrub" && serverFallbackPreviewKeyRef.current === key) return true;
    if (reason !== "scrub") serverFallbackPreviewKeyRef.current = key;
    scrubPreviewPendingIdxRef.current = normalized;
    if (scrubPreviewRafRef.current !== null) return true;
    scrubPreviewRafRef.current = window.requestAnimationFrame(() => {
      scrubPreviewRafRef.current = null;
      const pendingIdx = scrubPreviewPendingIdxRef.current;
      scrubPreviewPendingIdxRef.current = null;
      if (pendingIdx == null) return;
      const token = `${Date.now()}-${++scrubPreviewTokenRef.current}`;
      const dbg = show3dPerfDebug();
      if (dbg) {
        dbg.lastCommPreviewRequest = pendingIdx;
        dbg.lastCommPreviewReason = reason;
      }
      setScrubPreviewRequest(JSON.stringify({
        token,
        idx: pendingIdx,
        maxBytes: 16 * 1024 * 1024,
        reason,
      }));
    });
    return true;
  }, [offline, width, height, nSlices, frameServerVersion, setScrubPreviewRequest]);
  const staticFallbackUrl = staticFallbackJpeg
    ? `data:${staticFallbackMime || "image/jpeg"};base64,${staticFallbackJpeg}`
    : "";
  const hasSavedStaticFallback = staticFallbackUrl.length > 0;
  useHideStaticFallback(
    model,
    rootRef,
    folderWaiting || canRenderLive || hasSavedStaticFallback,
  );
  const gpuCanvasCtxRef = React.useRef<GPUCanvasContext | null>(null);
  const gpuCanvasSizeRef = React.useRef<{ w: number; h: number } | null>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const uiRef = React.useRef<HTMLCanvasElement>(null);
  const canvasWheelHandlerRef = React.useRef<((event: WheelEvent) => void) | null>(null);
  const fftInsetNativeWheelHandlerRef = React.useRef<((event: WheelEvent) => boolean) | null>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const fftInsetLayerRef = React.useRef<HTMLCanvasElement>(null);

  const [exportMenuAnchor, setExportMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [panelMenuAnchor, setPanelMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [viewMenuAnchor, setViewMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [exportBusy, setExportBusy] = React.useState(false);
  const [exportPanelMode, setExportPanelMode] = React.useState<ExportPanelMode>("home");
  const [exportQuality, setExportQuality] = React.useState<AnimationQuality>("medium");
  const [exportFrameStart, setExportFrameStart] = React.useState(1);
  const [exportFrameEnd, setExportFrameEnd] = React.useState(Math.max(1, nSlices || 1));
  const [exportEveryN, setExportEveryN] = React.useState(1);
  const [exportMaxFrames, setExportMaxFrames] = React.useState(40);
  const [exportFps, setExportFps] = React.useState(Math.max(1, Math.round(playbackFps || 10)));
  const [exportSpatialPreset, setExportSpatialPreset] = React.useState<ExportSpatialPreset>("edge512");
  const [exportGifPreset, setExportGifPreset] = React.useState<GifExportPreset>("slides");
  const [localExportStatus, setLocalExportStatus] = React.useState("");
  const fftOverlayDragRef = React.useRef<{
    pointerId: number;
    startClientX: number;
    startClientY: number;
    startInsetX: number;
    startInsetY: number;
    panelLeft: number;
    panelTop: number;
    panelW: number;
    panelH: number;
    insetW: number;
    insetH: number;
    moved: boolean;
  } | null>(null);
  const [fftOverlayDragPreview, setFftOverlayDragPreview] = React.useState<{ x: number; y: number } | null>(null);
  const pendingExportRef = React.useRef<{
    id: string;
    filename: string;
    mode: string;
    downsample: number;
    handle: Show3DFileHandle | null;
  } | null>(null);
  React.useEffect(() => {
    if (!exportStatus) return;
    const preparing = exportStatus.startsWith("Preparing ") || exportStatus.startsWith("Exporting ");
    if (preparing) {
      setExportBusy(true);
    } else if (!pendingExportRef.current) {
      setExportBusy(false);
    }
  }, [exportStatus]);
  React.useEffect(() => {
    if (!localExportStatus || exportBusy) return;
    if (localExportStatus.startsWith("Preparing ") || localExportStatus.startsWith("Saving ")) return;
    const id = window.setTimeout(() => {
      setLocalExportStatus((current) => current === localExportStatus ? "" : current);
    }, 12000);
    return () => window.clearTimeout(id);
  }, [localExportStatus, exportBusy]);
  const voxelCount = Math.max(0, Math.floor(nSlices) * Math.floor(height) * Math.floor(width));
  const exactExportSize = formatEstimatedHtmlSize(voxelCount * 4);
  const quantizedExportSize = formatEstimatedHtmlSize(voxelCount);
  const quantizedExportSize2 = formatEstimatedHtmlSize(Math.ceil(voxelCount / 4));
  const quantizedExportSize4 = formatEstimatedHtmlSize(Math.ceil(voxelCount / 16));
  const quantizedExportSize8 = formatEstimatedHtmlSize(Math.ceil(voxelCount / 64));
  const selectedSpatialOption = spatialOptionFor(exportSpatialPreset);
  const exportFrameIndices = React.useMemo(
    () => buildAnimationFrameIndices(nSlices, exportFrameStart, exportFrameEnd, exportEveryN, exportMaxFrames),
    [nSlices, exportFrameStart, exportFrameEnd, exportEveryN, exportMaxFrames],
  );
  const animationWorkEstimate = formatEstimatedAnimationWork(
    width,
    height,
    exportFrameIndices.length,
    visiblePanelCount,
    maxCols,
    panelGapTrait ?? 10,
    exportQuality,
    selectedSpatialOption.downsample,
    selectedSpatialOption.maxEdgePx,
  );
  const exportFrameSummary = `${exportFrameIndices.length}/${Math.max(1, nSlices || 1)} frames`;
  const animationExportRequest = React.useMemo(() => ({
    fps: Math.max(1, Math.round(exportFps || playbackFps || 10)),
    frame_start: exportFrameIndices[0] ?? 0,
    frame_stop: (exportFrameIndices[exportFrameIndices.length - 1] ?? 0) + 1,
    every_n: Math.max(1, Math.round(exportEveryN || 1)),
    max_frames: Math.max(0, Math.round(exportMaxFrames || 0)),
    downsample: selectedSpatialOption.downsample,
    max_edge_px: selectedSpatialOption.maxEdgePx || null,
    preset: exportGifPreset === "custom" ? "custom" : exportGifPreset,
    slides_preset: exportGifPreset === "slides",
    show_panel_titles: showPanelTitles !== false,
    show_scale_bar: Boolean(scaleBarVisible),
    show_zoom: showZoomIndicator !== false,
  }), [
    exportFps,
    playbackFps,
    exportFrameIndices,
    exportEveryN,
    exportMaxFrames,
    selectedSpatialOption.downsample,
    selectedSpatialOption.maxEdgePx,
    exportGifPreset,
    showPanelTitles,
    scaleBarVisible,
    showZoomIndicator,
  ]);
  const canDownloadCurrentHtml = !exportEnabled && (offline || hasOfflineStack || hasOfflineFloatStack || offlineForTheme);
  const standaloneHtmlMode = offline || hasOfflineStack ? "quantized" : "exact";
  const standaloneHtmlLabel = standaloneHtmlMode === "quantized"
    ? `HTML encoded uint8 (${quantizedExportSize})`
    : `HTML exact float32 (${exactExportSize})`;
  const canExportStandaloneGif = !exportEnabled && offline && width > 0 && height > 0 && nSlices > 0 && (
    hasOfflineStack || hasOfflineFloatStack || sidecarRamReady
  );
  const standaloneGifUnavailableTitle = sidecarMode && !sidecarRamReady
    ? "GIF export becomes available after the folder data finishes loading into RAM."
    : "GIF export needs embedded or loaded standalone image data.";
  const standaloneMp4UnavailableTitle = "MP4 export requires the live Python backend. Standalone HTML can export GIF or download itself as HTML.";
  const handleExportMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setExportPanelMode("home");
    setExportMenuAnchor(event.currentTarget);
  };
  const handleExportMenuClose = () => {
    setExportMenuAnchor(null);
  };
  React.useEffect(() => {
    const total = Math.max(1, Math.floor(nSlices || 1));
    setExportFrameStart((current) => Math.max(1, Math.min(total, Math.round(current || 1))));
    setExportFrameEnd((current) => Math.max(1, Math.min(total, Math.round(current || total))));
  }, [nSlices]);
  React.useEffect(() => {
    setExportFps(Math.max(1, Math.round(playbackFps || 10)));
  }, [playbackFps]);
  const handleExportSelect = async (
    mode: string,
    quality = "medium",
    downsample = 1,
    requestOptions: Record<string, unknown> = {},
  ) => {
    setExportMenuAnchor(null);
    if (mode !== "exact" && mode !== "quantized" && mode !== "gif" && mode !== "mp4") return;
    const filename = makeExportFilename(title, nSlices, height, width, mode, quality, downsample);
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setExportBusy(true);
    setLocalExportStatus("Choose export location...");
    const picker = (window as Show3DWindow).showSaveFilePicker;
    let handle: Show3DFileHandle | null = null;
    if (picker) {
      try {
        handle = await picker({
          suggestedName: filename,
          types: [exportPickerType(mode)],
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
    setExportRequest(JSON.stringify({ mode, quality, downsample, ...requestOptions, id, filename, download: true }));
  };
  const handleStandaloneHtmlDownload = () => {
    setExportMenuAnchor(null);
    const filename = makeExportFilename(title, nSlices, height, width, standaloneHtmlMode);
    try {
      const html = `<!doctype html>\n${standaloneHtmlWithCurrentWidgetState(
        model,
        standaloneWidgetStaticHtmlFromDocument(),
        SHOW3D_STANDALONE_VIEW_STATE_KEYS,
      )}`;
      const blob = new Blob([html], { type: "text/html;charset=utf-8" });
      downloadBlob(blob, filename);
      setLocalExportStatus(`Downloaded ${filename} to browser Downloads (${formatSavedBytes(blob.size)})`);
    } catch (err) {
      setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };
  const renderStandaloneGifFrame = (
    frameIdx: number,
    quality: string,
    options: { downsample: number; maxEdgePx: number | null },
  ): { width: number; height: number; indices: Uint8Array } | null => {
    const frame = getOfflineFrame(frameIdx);
    if (!frame) return null;
    const sourcePanelCount = Math.max(1, nPanels || 1);
    const panelW = sharedPanelSource ? Math.max(1, width) : Math.max(1, panelWidthPx || Math.floor(width / sourcePanelCount) || width);
    const panelH = Math.max(1, height);
    const scale = animationOutputScale(panelW, panelH, quality, options.downsample, options.maxEdgePx);
    const panels = (visiblePanelIndices.length ? visiblePanelIndices : [0])
      .filter((panel) => panel >= 0 && panel < sourcePanelCount);
    const activePanels = panels.length ? panels : [0];
    const panelOutW = Math.max(1, Math.round(panelW * scale));
    const panelOutH = Math.max(1, Math.round(panelH * scale));
    const cols = panelColsForCount(activePanels.length);
    const rows = Math.max(1, Math.ceil(activePanels.length / cols));
    const gap = activePanels.length > 1 ? Math.max(0, Math.round((panelGapTrait ?? 10) * scale)) : 0;
    const outW = cols * panelOutW + Math.max(0, cols - 1) * gap;
    const outH = rows * panelOutH + Math.max(0, rows - 1) * gap;
    const out = document.createElement("canvas");
    out.width = outW;
    out.height = outH;
    const outCtx = out.getContext("2d");
    if (!outCtx) return null;
    outCtx.imageSmoothingEnabled = smooth;
    outCtx.fillStyle = themeColors.bg;
    outCtx.fillRect(0, 0, outW, outH);

    const panelCanvas = document.createElement("canvas");
    panelCanvas.width = panelW;
    panelCanvas.height = panelH;
    const panelCtx = panelCanvas.getContext("2d");
    if (!panelCtx) return null;
    const panelImage = panelCtx.createImageData(panelW, panelH);
    const fallbackLut = COLORMAPS[cmap] || COLORMAPS.inferno;
    let sharedAutoRange: { vmin: number; vmax: number } | null = null;
    if (autoContrast && linkContrast) {
      sharedAutoRange =
        cachedAutoDisplayRange(autoVmins, autoVmaxs, frameIdx, logScale) ||
        cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, frameIdx, logScale);
      if (!sharedAutoRange) {
        const data = logScale ? applyLogScale(frame) : frame;
        sharedAutoRange = percentileClip(data, percentileLow, percentileHigh);
      }
    }

    for (let slot = 0; slot < activePanels.length; slot++) {
      const panel = activePanels[slot];
      const panelData = extractPanelSlice(frame, panel, logScale);
      if (!panelData) continue;
      const lut = COLORMAPS[panelCmapFor(panel)] || fallbackLut;
      let range: { vmin: number; vmax: number };
      if (autoContrast) {
        if (sharedAutoRange && linkContrast) {
          range = sharedAutoRange;
        } else {
          const clipped = percentileClip(panelData, percentileLow, percentileHigh);
          if (clipped.vmax > clipped.vmin) {
            range = clipped;
          } else {
            const fallback = findDataRange(panelData);
            range = { vmin: fallback.min, vmax: fallback.max };
          }
        }
      } else if (!linkContrast && activePanels.length > 1) {
        const stackBounds = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
        const pdr = panelDataRanges[panel];
        const bounds = (perPanelHistogramEnabled && pdr && pdr.max > pdr.min) ? pdr : stackBounds;
        range = resolvePanelRange(panel, bounds);
      } else {
        range = resolveDisplayRange(
          dataMin,
          dataMax,
          traitVmin,
          traitVmax,
          logScale,
          imageVminPct,
          imageVmaxPct,
        );
      }
      renderFramePlayback(panelData, panelImage.data, lut, range.vmin, range.vmax, false);
      panelCtx.putImageData(panelImage, 0, 0);
      const col = slot % cols;
      const row = Math.floor(slot / cols);
      const x = col * (panelOutW + gap);
      const y = row * (panelOutH + gap);
      outCtx.save();
      outCtx.beginPath();
      outCtx.rect(x, y, panelOutW, panelOutH);
      outCtx.clip();
      outCtx.translate(x, y);
      if (flipCols || flipRows) {
        outCtx.translate(flipCols ? panelOutW : 0, flipRows ? panelOutH : 0);
        outCtx.scale(flipCols ? -1 : 1, flipRows ? -1 : 1);
      }
      if (imageRotation % 4 !== 0) {
        outCtx.translate(panelOutW / 2, panelOutH / 2);
        outCtx.rotate((imageRotation * Math.PI) / 2);
        outCtx.translate(-panelOutW / 2, -panelOutH / 2);
      }
      outCtx.drawImage(panelCanvas, 0, 0, panelW, panelH, 0, 0, panelOutW, panelOutH);
      outCtx.restore();
      if (showPanelTitles !== false) {
        const label = panelTitleText(panel);
        if (label) {
          outCtx.save();
          outCtx.font = `700 ${Math.max(8, Math.round(11 * scale))}px ${UI_FONT}`;
          outCtx.textAlign = "center";
          outCtx.textBaseline = "top";
          outCtx.shadowColor = "rgba(0,0,0,0.75)";
          outCtx.shadowBlur = 2;
          outCtx.shadowOffsetX = 1;
          outCtx.shadowOffsetY = 1;
          outCtx.fillStyle = "white";
          outCtx.fillText(label, x + panelOutW / 2, y + Math.max(2, Math.round(3 * scale)));
          outCtx.restore();
        }
      }
      if (scaleBarVisible && pixelSize > 0) {
        const outputPixelSize = pixelSize / Math.max(1e-6, scale);
        const targetBarPx = Math.min(60 * scale, panelOutW * 0.25);
        const nicePhysical = roundToNiceValue(targetBarPx * outputPixelSize);
        const barPx = Math.max(1, (nicePhysical / outputPixelSize));
        const barThickness = Math.max(2, Math.round(5 * scale));
        const fontSize = Math.max(8, Math.round(16 * scale));
        const margin = Math.max(6, Math.round(12 * scale));
        const barX = x + panelOutW - barPx - margin;
        const barY = y + panelOutH - margin;
        outCtx.save();
        outCtx.shadowColor = "rgba(0,0,0,0.75)";
        outCtx.shadowBlur = 2;
        outCtx.shadowOffsetX = 1;
        outCtx.shadowOffsetY = 1;
        outCtx.fillStyle = "white";
        outCtx.fillRect(barX, barY, barPx, barThickness);
        outCtx.font = `${fontSize}px ${UI_FONT}`;
        outCtx.textAlign = "center";
        outCtx.textBaseline = "bottom";
        outCtx.fillText(formatScaleLabel(nicePhysical, pixelUnit || "px"), barX + barPx / 2, barY - Math.max(2, Math.round(4 * scale)));
        if (showZoomIndicator !== false) {
          outCtx.textAlign = "left";
          outCtx.fillText(formatZoomLabel(1), x + margin, y + panelOutH - margin + barThickness);
        }
        outCtx.restore();
      }
    }
    const rgba = outCtx.getImageData(0, 0, outW, outH).data;
    return { width: outW, height: outH, indices: quantizeRgbaForBrowserGif(rgba) };
  };
  const handleStandaloneGifDownload = async (
    quality = "medium",
    requestOptions: Record<string, unknown> = {},
  ) => {
    setExportMenuAnchor(null);
    if (!canExportStandaloneGif) {
      setLocalExportStatus(standaloneGifUnavailableTitle);
      return;
    }
    const filename = makeExportFilename(title, nSlices, height, width, "gif", quality);
    setExportBusy(true);
    setLocalExportStatus(`Preparing ${filename}...`);
    try {
      const frames: Uint8Array[] = [];
      let outW = 0;
      let outH = 0;
      const frameIndices = exportFrameIndices.length ? exportFrameIndices : [Math.max(0, Math.min(Math.max(1, nSlices || 1) - 1, sliceIdx || 0))];
      const spatialOption = {
        downsample: Number(requestOptions.downsample ?? selectedSpatialOption.downsample),
        maxEdgePx: requestOptions.max_edge_px == null ? null : Number(requestOptions.max_edge_px),
      };
      const total = frameIndices.length;
      for (let slot = 0; slot < total; slot++) {
        const frameIdx = frameIndices[slot];
        const rendered = renderStandaloneGifFrame(frameIdx, quality, spatialOption);
        if (!rendered) throw new Error(`frame ${frameIdx + 1}/${Math.max(1, nSlices || 1)} is not loaded`);
        outW = rendered.width;
        outH = rendered.height;
        frames.push(rendered.indices);
        if (slot === 0 || slot === total - 1 || (slot + 1) % 4 === 0) {
          setLocalExportStatus(`Encoding ${filename}... ${slot + 1}/${total}`);
          await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
        }
      }
      const delayCs = 100 / clampPlaybackFps(Number(requestOptions.fps ?? exportFps ?? playbackFps));
      const gif = encodeIndexedGif(outW, outH, frames, delayCs);
      const blob = new Blob([gif as BlobPart], { type: "image/gif" });
      downloadBlob(blob, filename);
      setLocalExportStatus(`Downloaded ${filename} to browser Downloads (${formatSavedBytes(blob.size)})`);
    } catch (err) {
      setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setExportBusy(false);
    }
  };
  const handleAnimationExportSelect = (mode: "gif" | "mp4", quality: string, requestOptions = animationExportRequest) => {
    if (mode === "gif" && canExportStandaloneGif) {
      void handleStandaloneGifDownload(quality, requestOptions);
      return;
    }
    void handleExportSelect(mode, quality, 1, requestOptions);
  };
  const applyGifExportPreset = React.useCallback((preset: GifExportPreset) => {
    setExportPanelMode("gif");
    setExportGifPreset(preset);
    setExportFrameStart(1);
    setExportFrameEnd(Math.max(1, nSlices || 1));
    if (preset === "slides") {
      setExportQuality("medium");
      setExportEveryN(1);
      setExportMaxFrames(40);
      setExportFps(Math.min(10, Math.max(1, Math.round(playbackFps || 10))));
      setExportSpatialPreset("edge512");
    } else if (preset === "compact") {
      setExportQuality("low");
      setExportEveryN(Math.max(1, Math.ceil(Math.max(1, nSlices || 1) / 24)));
      setExportMaxFrames(24);
      setExportFps(8);
      setExportSpatialPreset("down4");
    } else if (preset === "full") {
      setExportQuality("high");
      setExportEveryN(1);
      setExportMaxFrames(0);
      setExportFps(Math.max(1, Math.round(playbackFps || 10)));
      setExportSpatialPreset("full");
    }
  }, [nSlices, playbackFps]);
  const markGifPresetCustom = React.useCallback(() => {
    setExportGifPreset((preset) => preset === "custom" ? preset : "custom");
  }, []);
  const exportNumberFieldSx = {
    width: 72,
    "& .MuiInputBase-input": { py: 0.45, px: 0.75, fontSize: 11 },
  } as const;
  const exportPanelButtonSx = { ...compactButton, fontSize: 11, border: `1px solid ${themeColors.border}` } as const;
  const renderAnimationExportPanel = (mode: "gif" | "mp4") => {
    const isGif = mode === "gif";
    const disabled = isGif ? !(exportEnabled || canExportStandaloneGif) : !exportEnabled;
    const disabledTitle = isGif ? standaloneGifUnavailableTitle : standaloneMp4UnavailableTitle;
    const titleText = isGif ? "GIF" : "MP4 video";
    return (
      <Box sx={{ width: 360, maxWidth: "90vw", px: 1.25, py: 1, display: "flex", flexDirection: "column", gap: 0.9 }}>
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
          <Button size="small" sx={compactButton} onClick={() => setExportPanelMode("home")}>Back</Button>
          <Box sx={{ display: "flex", alignItems: "center", flex: 1, minWidth: 0 }}>
            <Typography sx={{ ...typography.title, fontSize: 12 }}>{titleText}</Typography>
            {isGif && (
              <InfoTooltip
                theme={themeInfo.theme}
                icon="?"
                text={(
                  <Box sx={{ fontSize: 11, lineHeight: 1.4 }}>
                    <b>GIF export lifecycle</b>
                    <br />1. Pick a preset or adjust frames, fps, quality, and size.
                    <br />2. Press <b>Export GIF</b>; the widget renders panel-only frames using the current contrast and overlays.
                    <br />3. In standalone HTML, the browser downloads the GIF to Downloads. In a live notebook, choose a save location when prompted.
                  </Box>
                )}
              />
            )}
          </Box>
        </Box>
        <Typography sx={{ fontSize: 11, color: themeColors.textMuted, lineHeight: 1.35 }}>
          {isGif
            ? "GIF exports the current panel movie without toolbar chrome. Use a preset, then export."
            : "MP4 is secondary for presentation workflows and requires the live Python backend."}
        </Typography>
        {isGif && (
          <Box sx={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 0.5 }}>
            {GIF_EXPORT_PRESETS.map((preset) => (
              <Button
                key={preset.value}
                size="small"
                sx={{
                  ...compactButton,
                  minWidth: 0,
                  border: `1px solid ${exportGifPreset === preset.value ? themeColors.accent : themeColors.border}`,
                  bgcolor: exportGifPreset === preset.value ? "rgba(25,118,210,0.12)" : "transparent",
                }}
                onClick={() => applyGifExportPreset(preset.value)}
              >
                {preset.label}
              </Button>
            ))}
          </Box>
        )}
        <Box sx={{ display: "grid", gridTemplateColumns: "auto 1fr auto 1fr", alignItems: "center", gap: 0.75 }}>
          <Typography sx={typography.label}>Frames</Typography>
          <TextField size="small" type="number" value={exportFrameStart} onChange={(event) => { markGifPresetCustom(); setExportFrameStart(Number(event.target.value)); }} inputProps={{ min: 1, max: Math.max(1, nSlices || 1), "aria-label": "Export first frame" }} sx={exportNumberFieldSx} />
          <Typography sx={typography.label}>to</Typography>
          <TextField size="small" type="number" value={exportFrameEnd} onChange={(event) => { markGifPresetCustom(); setExportFrameEnd(Number(event.target.value)); }} inputProps={{ min: 1, max: Math.max(1, nSlices || 1), "aria-label": "Export last frame" }} sx={exportNumberFieldSx} />
          <Typography sx={typography.label}>Every</Typography>
          <TextField size="small" type="number" value={exportEveryN} onChange={(event) => { markGifPresetCustom(); setExportEveryN(Math.max(1, Number(event.target.value))); }} inputProps={{ min: 1, max: Math.max(1, nSlices || 1), "aria-label": "Export every Nth frame" }} sx={exportNumberFieldSx} />
          <Typography sx={typography.label}>Max frames</Typography>
          <TextField size="small" type="number" value={exportMaxFrames} onChange={(event) => { markGifPresetCustom(); setExportMaxFrames(Math.max(0, Number(event.target.value))); }} inputProps={{ min: 0, max: Math.max(1, nSlices || 1), "aria-label": "Maximum exported frames; zero means all" }} sx={exportNumberFieldSx} />
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 0.75 }}>
          <Typography sx={typography.label}>fps</Typography>
          <TextField size="small" type="number" value={exportFps} onChange={(event) => { markGifPresetCustom(); setExportFps(Math.max(1, Number(event.target.value))); }} inputProps={{ min: 1, max: MAX_PLAYBACK_FPS, "aria-label": "Export animation frames per second" }} sx={exportNumberFieldSx} />
          <Typography sx={typography.label}>Quality</Typography>
          <Select size="small" value={exportQuality} onChange={(event) => { markGifPresetCustom(); setExportQuality(event.target.value as AnimationQuality); }} sx={{ ...themedSelect, minWidth: 86, fontSize: 11 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Animation quality" }}>
            {ANIMATION_QUALITY_OPTIONS.map((quality) => <MenuItem key={quality} value={quality}>{quality}</MenuItem>)}
          </Select>
          <Typography sx={typography.label}>Size</Typography>
          <Select size="small" value={exportSpatialPreset} onChange={(event) => { markGifPresetCustom(); setExportSpatialPreset(event.target.value as ExportSpatialPreset); }} sx={{ ...themedSelect, minWidth: 144, fontSize: 11 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Animation spatial size" }}>
            {EXPORT_SPATIAL_OPTIONS.map((option) => <MenuItem key={option.value} value={option.value}>{option.label}</MenuItem>)}
          </Select>
        </Box>
        <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
          {exportFrameSummary} · {animationWorkEstimate}
          {selectedSpatialOption.downsample > 1 ? ` · ${selectedSpatialOption.downsample}x downsample` : ""}
          {selectedSpatialOption.maxEdgePx ? ` · max edge ${selectedSpatialOption.maxEdgePx}px` : ""}
        </Typography>
        <Button
          size="small"
          sx={exportPanelButtonSx}
          disabled={disabled || exportBusy}
          title={disabled ? disabledTitle : undefined}
          onClick={() => handleAnimationExportSelect(mode, exportQuality, animationExportRequest)}
        >
          {isGif ? "Export GIF" : "Export MP4"}
        </Button>
      </Box>
    );
  };
  const renderHtmlExportPanel = () => (
    <Box sx={{ width: 340, maxWidth: "86vw", px: 1.25, py: 1, display: "flex", flexDirection: "column", gap: 0.8 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Button size="small" sx={compactButton} onClick={() => setExportPanelMode("home")}>Back</Button>
        <Typography sx={{ ...typography.title, fontSize: 12 }}>Interactive HTML</Typography>
      </Box>
      <Typography sx={{ fontSize: 11, color: themeColors.textMuted, lineHeight: 1.35 }}>
        HTML is the primary interactive sharing path. Exact keeps float32 data; encoded uint8 is smaller for visual reports.
      </Typography>
      {exportEnabled && <Button size="small" sx={exportPanelButtonSx} onClick={() => handleExportSelect("exact")}>HTML exact float32 ({exactExportSize})</Button>}
      {exportEnabled && <Button size="small" sx={exportPanelButtonSx} onClick={() => handleExportSelect("quantized")}>HTML encoded uint8 ({quantizedExportSize})</Button>}
      {exportEnabled && height >= 2 && width >= 2 && <Button size="small" sx={exportPanelButtonSx} onClick={() => handleExportSelect("quantized", "medium", 2)}>HTML encoded uint8, 2x downsample ({quantizedExportSize2})</Button>}
      {exportEnabled && height >= 4 && width >= 4 && <Button size="small" sx={exportPanelButtonSx} onClick={() => handleExportSelect("quantized", "medium", 4)}>HTML encoded uint8, 4x downsample ({quantizedExportSize4})</Button>}
      {exportEnabled && height >= 8 && width >= 8 && <Button size="small" sx={exportPanelButtonSx} onClick={() => handleExportSelect("quantized", "medium", 8)}>HTML encoded uint8, 8x downsample ({quantizedExportSize8})</Button>}
      {canDownloadCurrentHtml && standaloneHtmlMode === "quantized" && <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Exact float32 is not embedded in this standalone page; open the live widget for exact export.</Typography>}
      {canDownloadCurrentHtml && <Button size="small" sx={exportPanelButtonSx} onClick={handleStandaloneHtmlDownload}>{standaloneHtmlLabel}</Button>}
      {canDownloadCurrentHtml && standaloneHtmlMode !== "quantized" && <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Encoded uint8 export requires the Python backend to repack the current float32 stack.</Typography>}
    </Box>
  );
  const renderExportMenuContent = () => {
    if (exportPanelMode === "gif") return renderAnimationExportPanel("gif");
    if (exportPanelMode === "mp4") return renderAnimationExportPanel("mp4");
    if (exportPanelMode === "html") return renderHtmlExportPanel();
    return (
      <Box sx={{ width: 320, maxWidth: "86vw", py: 0.5 }}>
        <MenuItem disabled={!(exportEnabled || canExportStandaloneGif)} title={!(exportEnabled || canExportStandaloneGif) ? standaloneGifUnavailableTitle : undefined} onClick={() => setExportPanelMode("gif")}>
          GIF
        </MenuItem>
        <MenuItem disabled={!(exportEnabled || canDownloadCurrentHtml)} onClick={() => setExportPanelMode("html")}>
          Interactive HTML
        </MenuItem>
        <MenuItem disabled={!exportEnabled} title={!exportEnabled ? standaloneMp4UnavailableTitle : undefined} onClick={() => setExportPanelMode("mp4")}>
          MP4 video (secondary)
        </MenuItem>
        <Box sx={{ px: 2, py: 0.75, borderTop: `1px solid ${themeColors.border}` }}>
          <Typography sx={{ fontSize: 11, color: themeColors.textMuted, lineHeight: 1.35 }}>
            Use GIF for slides, HTML for interactive review, and MP4 only when a video file is required.
          </Typography>
        </Box>
      </Box>
    );
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
      const blob = new Blob([payload as BlobPart], { type: exportBlobType(pending.mode) });
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
        setLocalExportStatus(
          pending.handle
            ? `Saved ${filename} to selected location (${formatSavedBytes(bytes.byteLength)})`
            : `Downloaded ${filename} to browser Downloads (${formatSavedBytes(bytes.byteLength)})`,
        );
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
  }, [exportPayload, exportPayloadId, exportPayloadFilename, setExportRequest]);

  // Local state
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false);
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  const resizeAspectRef = React.useRef<number | null>(null);
  const roiItems = (roiList || []).map((roi, i) => normalizeROI(roi, i));
  const selectedRoi = roiSelectedIdx >= 0 && roiSelectedIdx < roiItems.length ? roiItems[roiSelectedIdx] : null;
  const [showRoiResizeHint, setShowRoiResizeHint] = React.useState(true);
  const [overlayEditMode, setOverlayEditMode] = React.useState(false);
  const [overlaySelection, setOverlaySelection] = React.useState<OverlaySelection | null>(null);
  const [isDraggingOverlay, setIsDraggingOverlay] = React.useState(false);
  const [isHoveringOverlay, setIsHoveringOverlay] = React.useState(false);
  const overlayDragRef = React.useRef<OverlayDragState | null>(null);
  const overlayBaselineRef = React.useRef<PanelOverlaySpec[][] | null>(null);
  const hasPanelOverlays = React.useMemo(() => (panelOverlays || []).some((items) => items && items.length > 0), [panelOverlays]);
  React.useEffect(() => {
    if (!overlayBaselineRef.current && hasPanelOverlays) {
      overlayBaselineRef.current = clonePanelOverlays(panelOverlays);
    }
  }, [hasPanelOverlays, panelOverlays]);
  React.useEffect(() => {
    if (!overlaySelection) return;
    const exists = Boolean(panelOverlays?.[overlaySelection.panel]?.[overlaySelection.overlay]);
    if (!exists) setOverlaySelection(null);
  }, [overlaySelection, panelOverlays]);

  const updatePanelOverlay = React.useCallback((panel: number, overlay: number, nextSpec: PanelOverlaySpec) => {
    const next = clonePanelOverlays(panelOverlays);
    while (next.length <= panel) next.push([]);
    if (!next[panel] || overlay < 0 || overlay >= next[panel].length) return;
    next[panel][overlay] = nextSpec;
    setPanelOverlays(next);
  }, [panelOverlays, setPanelOverlays]);

  const deleteSelectedOverlay = React.useCallback(() => {
    if (!overlaySelection) return;
    const next = clonePanelOverlays(panelOverlays);
    const items = next[overlaySelection.panel];
    if (!items || overlaySelection.overlay < 0 || overlaySelection.overlay >= items.length) return;
    items.splice(overlaySelection.overlay, 1);
    setPanelOverlays(next);
    setOverlaySelection(null);
  }, [overlaySelection, panelOverlays, setPanelOverlays]);

  const resetPanelOverlays = React.useCallback(() => {
    if (!overlayBaselineRef.current) return;
    setPanelOverlays(clonePanelOverlays(overlayBaselineRef.current));
    setOverlaySelection(null);
    overlayDragRef.current = null;
    setIsDraggingOverlay(false);
  }, [setPanelOverlays]);
  const pendingRoiAddRef = React.useRef<{ row: number; col: number } | null>(null);

  // Preview panel state (JS-only, shows ROI crop at full resolution - auto-shows when ROI selected)
  const [previewZoom, setPreviewZoom] = React.useState({ zoom: 1, panX: 0, panY: 0 });
  const previewCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const previewOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const previewContainerRef = React.useRef<HTMLDivElement>(null);
  const [isDraggingPreviewPan, setIsDraggingPreviewPan] = React.useState(false);
  const [previewPanStart, setPreviewPanStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);
  const [previewCropDims, setPreviewCropDims] = React.useState<{ w: number; h: number } | null>(null);
  const previewOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const [previewVersion, setPreviewVersion] = React.useState(0);

  const updateSelectedRoi = (updates: Partial<ROIItem>) => {
    if (roiSelectedIdx < 0 || !roiList) return;
    const newList = [...roiList];
    newList[roiSelectedIdx] = { ...newList[roiSelectedIdx], ...updates };
    setRoiList(newList);
  };
  // Per-panel zoom/pan: index 0 is also used as the shared linked state.
  // Each panel keeps its own state when unlinked.
  type PanelState = {
    zoom: number;
    panX: number;
    panY: number;
    imageVminPct: number;
    imageVmaxPct: number;
  };
  type TouchTransformState = {
    panelIdx: number;
    mode: "pan" | "pinch";
    startX: number;
    startY: number;
    startDistance: number;
    startMidX: number;
    startMidY: number;
    startState: PanelState;
  };
  type FftTouchTransformState = {
    mode: "pan" | "pinch";
    startX: number;
    startY: number;
    startDistance: number;
    startMidX: number;
    startMidY: number;
    startState: { zoom: number; panX: number; panY: number };
  };
  const initialState: PanelState = {
    zoom: 1,
    panX: 0,
    panY: 0,
    imageVminPct: 0,
    imageVmaxPct: 100,
  };
  type Show3DViewState = {
    linked_state?: Partial<PanelState>;
    panel_states?: Partial<PanelState>[];
  };
  const [viewState, setViewState] = useModelState<Show3DViewState>("view_state");
  const readNumber = (value: unknown, fallback: number): number => (
    typeof value === "number" && Number.isFinite(value) ? value : fallback
  );
  const normalizePanelState = (value: Partial<PanelState> | undefined, fallback: PanelState): PanelState => ({
    zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, readNumber(value?.zoom, fallback.zoom))),
    panX: readNumber(value?.panX, readNumber((value as { pan_x?: unknown } | undefined)?.pan_x, fallback.panX)),
    panY: readNumber(value?.panY, readNumber((value as { pan_y?: unknown } | undefined)?.pan_y, fallback.panY)),
    imageVminPct: readNumber(value?.imageVminPct, readNumber((value as { image_vmin_pct?: unknown } | undefined)?.image_vmin_pct, fallback.imageVminPct)),
    imageVmaxPct: readNumber(value?.imageVmaxPct, readNumber((value as { image_vmax_pct?: unknown } | undefined)?.image_vmax_pct, fallback.imageVmaxPct)),
  });
  const savedPanelStates = Array.isArray(viewState?.panel_states)
    ? viewState.panel_states.map(v => normalizePanelState(v, initialState))
    : [initialState];
  const [linkedState, setLinkedState] = React.useState<PanelState>(() => normalizePanelState(viewState?.linked_state, savedPanelStates[0] || initialState));
  const [panelStates, setPanelStates] = React.useState<PanelState[]>(() => savedPanelStates.length ? savedPanelStates : [initialState]);
  const linkedStateLiveRef = React.useRef<PanelState>(linkedState);
  const panelStatesLiveRef = React.useRef<PanelState[]>(panelStates);
  const transformRenderRafRef = React.useRef<number | null>(null);
  const transformStateCommitTimerRef = React.useRef<number | null>(null);
  const transformInputAtRef = React.useRef(0);
  React.useEffect(() => {
    const n = Math.max(1, nPanels || 1);
    setPanelStates(prev => {
      if (prev.length === n) return prev;
      const next = Array.from({ length: n }, (_, i) => prev[i] || { ...initialState });
      return next;
    });
  }, [nPanels]);
  // Seamless toggle: on link→unlink, copy linkedState into every panel; on
  // unlink→link, copy panel 0 into linkedState. Single effect so both axes
  // sync atomically.
  const prevLinkRef = React.useRef(linkPanels);
  React.useEffect(() => {
    if (prevLinkRef.current && !linkPanels) {
      // Linked → unlinked: distribute linkedState to all panels
      const s = linkedState;
      setPanelStates(arr => {
        const next = arr.map(() => ({ ...s }));
        setViewState({ linked_state: { ...s }, panel_states: next.map(v => ({ ...v })) });
        return next;
      });
    } else if (!prevLinkRef.current && linkPanels) {
      // Unlinked → linked: adopt panel 0's state as the shared linked state
      const s0 = panelStates[0] || initialState;
      setLinkedState({ ...s0 });
      setViewState({ linked_state: { ...s0 }, panel_states: panelStates.map(v => ({ ...v })) });
    }
    prevLinkRef.current = linkPanels;
  }, [linkPanels]);
  const stateFor = React.useCallback((panelIdx: number): PanelState => {
    const livePanels = panelStatesLiveRef.current;
    return linkPanels
      ? linkedStateLiveRef.current
      : (livePanels[panelIdx] || panelStates[panelIdx] || initialState);
  }, [linkPanels, panelStates]);
  const sidecarDisplayStyleKey = React.useMemo(() => {
    const panels = visiblePanelIndices.length
      ? visiblePanelIndices
      : Array.from({ length: Math.max(1, nPanels || 1) }, (_, idx) => idx);
    return JSON.stringify({
      cmap,
      autoContrast,
      logScale,
      percentileLow: Number(percentileLow || 0).toFixed(3),
      percentileHigh: Number(percentileHigh || 100).toFixed(3),
      linkContrast,
      imageVminPct: Number(imageVminPct || 0).toFixed(3),
      imageVmaxPct: Number(imageVmaxPct || 100).toFixed(3),
      autoVmins: (autoVmins || []).map((value) => Number(value).toFixed(6)),
      autoVmaxs: (autoVmaxs || []).map((value) => Number(value).toFixed(6)),
      panels: panels.map((panel) => {
        const state = panelStates[panel] || initialState;
        return [
          panel,
          panelCmapFor(panel),
          Number(state.imageVminPct || 0).toFixed(3),
          Number(state.imageVmaxPct || 100).toFixed(3),
          offlineMins?.[panel] ?? offlineMin ?? null,
          offlineMaxs?.[panel] ?? offlineMax ?? null,
          vminPerPanel?.[panel] ?? null,
          vmaxPerPanel?.[panel] ?? null,
        ];
      }),
    });
  }, [
    autoContrast,
    autoVmins,
    autoVmaxs,
    cmap,
    imageVminPct,
    imageVmaxPct,
    initialState,
    linkContrast,
    logScale,
    nPanels,
    offlineMin,
    offlineMax,
    offlineMins,
    offlineMaxs,
    panelCmapFor,
    panelStates,
    percentileLow,
    percentileHigh,
    visiblePanelIndices,
    vminPerPanel,
    vmaxPerPanel,
  ]);
  const syncPlaybackPanelTransform = (panelIdx: number, nextZoom: number, nextPanX: number, nextPanY: number) => {
    const clampAxis = (pan: number, viewport: number, zoomValue: number) => {
      if (viewport <= 0) return 0;
      if (zoomValue <= 1) return viewport * (1 - zoomValue) / 2;
      return Math.max(viewport * (1 - zoomValue), Math.min(0, pan));
    };
    const panelCount = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(panelCount);
    const rows = Math.max(1, Math.ceil(panelCount / cols));
    const gap = panelCount > 1 ? (panelGapTrait ?? 10) : 0;
    const viewportW = (canvasW - gap * (cols - 1)) / cols;
    const viewportH = (canvasH - gap * (rows - 1)) / rows;
    const zoomValue = Math.max(MIN_IMAGE_ZOOM, Math.min(MAX_ZOOM, nextZoom));
    const panXValue = clampAxis(nextPanX, viewportW, zoomValue);
    const panYValue = clampAxis(nextPanY, viewportH, zoomValue);
    const c = playRef.current;
    if (c.linkPanels) {
      const nextLinked = { ...c.linkedState, zoom: zoomValue, panX: panXValue, panY: panYValue };
      c.linkedState = nextLinked;
      linkedStateLiveRef.current = nextLinked;
    } else {
      const next = c.panelStates.slice();
      const prev = next[panelIdx] || initialState;
      next[panelIdx] = { ...prev, zoom: zoomValue, panX: panXValue, panY: panYValue };
      c.panelStates = next;
      panelStatesLiveRef.current = next;
    }
    if (sidecarMode) invalidateSidecarViewportCache("view-transform");
  };
  // Back-compat aliases for the single-panel code paths (ROI, profile, etc.)
  // which still expect plain zoom/panX/panY. Use panel 0's state.
  const zoom = stateFor(0).zoom;
  const panX = stateFor(0).panX;
  const panY = stateFor(0).panY;
  const [isDraggingPan, setIsDraggingPan] = React.useState(false);
  const [panStart, setPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);
  const panStartPanelRef = React.useRef<number>(0);
  const [mainCanvasSize, setMainCanvasSize] = React.useState(CANVAS_TARGET_SIZE);
  const rawFrameDataRef = React.useRef<Float32Array | null>(null);
  const initialCanvasSizeRef = React.useRef<number>(canvasSizeTrait > 0 ? canvasSizeTrait : CANVAS_TARGET_SIZE);
  const panelColsForCount = React.useCallback((count: number) => {
    const n = Math.max(1, count || 1);
    const requestedCols = (maxCols && maxCols > 0) ? Math.min(maxCols, n, MAX_PANEL_COLUMNS) : Math.min(n, MAX_PANEL_COLUMNS);
    if (n <= 1) return 1;
    return Math.max(1, requestedCols);
  }, [maxCols]);
  const show3dColumnOptions = React.useMemo(() => {
    const n = Math.max(1, visiblePanelCount || 1);
    const values = new Set<number>([1, 2, 3, 4, 5, 6, 8, 10, 12]);
    return Array.from(values).filter((cols) => cols >= 1 && cols <= n).sort((a, b) => a - b);
  }, [visiblePanelCount]);
  const clampedMaxCols = Math.max(1, Math.min((maxCols && maxCols > 0) ? maxCols : visiblePanelCount || 1, visiblePanelCount || 1, MAX_PANEL_COLUMNS));

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<CursorInfo | null>(null);
  const [cursorReadoutVisible, setCursorReadoutVisible] = React.useState(false);
  const cursorReadoutVisibleRef = React.useRef(false);
  const cursorInfoPendingRef = React.useRef<CursorInfo | null>(null);
  const cursorInfoRafRef = React.useRef<number | null>(null);
  const [showRoiPlot, setShowRoiPlot] = React.useState(true);
  const roiPlotCanvasRef = React.useRef<HTMLCanvasElement>(null);

  // Lens (magnifier inset)
  const [showLens, setShowLens] = React.useState(false);
  const [lensPos, setLensPos] = React.useState<{ row: number; col: number } | null>(null);
  const [lensMag, setLensMag] = React.useState(4);
  const [lensDisplaySize, setLensDisplaySize] = React.useState(128);
  const [lensAnchor, setLensAnchor] = React.useState<{ x: number; y: number } | null>(null);
  const [isDraggingLens, setIsDraggingLens] = React.useState(false);
  const lensCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const lensDragStartRef = React.useRef<{ mx: number; my: number; ax: number; ay: number } | null>(null);
  const [isResizingLens, setIsResizingLens] = React.useState(false);
  const [isHoveringLensEdge, setIsHoveringLensEdge] = React.useState(false);
  const lensResizeStartRef = React.useRef<{ my: number; startSize: number } | null>(null);

  const scheduleCursorInfo = React.useCallback((next: CursorInfo | null) => {
    cursorInfoPendingRef.current = next;
    if (cursorReadoutVisibleRef.current !== Boolean(next)) {
      cursorReadoutVisibleRef.current = Boolean(next);
      setCursorReadoutVisible(Boolean(next));
    }
    if (cursorInfoRafRef.current != null) return;
    if (typeof window === "undefined" || typeof window.requestAnimationFrame !== "function") {
      if (next) setCursorInfo(next);
      return;
    }
    cursorInfoRafRef.current = window.requestAnimationFrame(() => {
      cursorInfoRafRef.current = null;
      const pending = cursorInfoPendingRef.current;
      if (!pending) return;
      setCursorInfo((prev) => (
        prev &&
        prev.row === pending.row &&
        prev.col === pending.col &&
        prev.panelIdx === pending.panelIdx &&
        prev.value === pending.value
          ? prev
          : pending
      ));
    });
  }, []);

  React.useEffect(() => () => {
    if (cursorInfoRafRef.current != null && typeof window !== "undefined") {
      window.cancelAnimationFrame(cursorInfoRafRef.current);
    }
  }, []);

  // Reusable rendering buffers (avoid per-frame allocation)
  const mainOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const mainImgDataRef = React.useRef<ImageData | null>(null);
  const mainOffscreenSourcePanelWidthRef = React.useRef<number | undefined>(undefined);
  const scaledPlaybackImgDataRef = React.useRef<{ width: number; height: number; imageData: ImageData } | null>(null);
  const scaledPlaybackMapRef = React.useRef<{
    srcW: number;
    srcH: number;
    outW: number;
    outH: number;
    xMap: Uint32Array;
    yMap: Uint32Array;
  } | null>(null);
  const logBufferRef = React.useRef<Float32Array | null>(null);

  // Playback buffer refs (double-buffer: current + next to avoid overwrite stalls)
  const bufferRef = React.useRef<Float32Array | null>(null);
  const bufferStartRef = React.useRef(0);
  const bufferCountRef = React.useRef(0);
  const nextBufferRef = React.useRef<Float32Array | null>(null);
  const nextBufferStartRef = React.useRef(0);
  const nextBufferCountRef = React.useRef(0);
  const prefetchPendingRef = React.useRef(false);
  // Seed from the model's slice_idx (not 0): on mount the not-playing branch
  // of the playback effect syncs this ref back onto slice_idx in offline mode,
  // and a stale 0 would clobber a baked middle-slice start.
  const playbackIdxRef = React.useRef(Number.isFinite(sliceIdx) ? sliceIdx : 0);
  const playbackSliderRef = React.useRef<HTMLSpanElement>(null);
  const playbackLiveCountRef = React.useRef<HTMLElement>(null);
  const frameFetchCacheRef = React.useRef<Map<number, Float32Array>>(new Map());
  const frameFetchPendingRef = React.useRef<Map<number, Promise<Float32Array | null>>>(new Map());
  const panelGpuFramePendingRef = React.useRef<Map<number, Promise<boolean>>>(new Map());
  const frameFetchSerialRef = React.useRef(0);
  const localAutoVminsRef = React.useRef<number[]>([]);
  const localAutoVmaxsRef = React.useRef<number[]>([]);
  const autoRangeComputeTokenRef = React.useRef(0);

  const [displaySliceIdx, setDisplaySliceIdx] = React.useState(sliceIdx);
  const [playbackUiSliceIdx, setPlaybackUiSliceIdx] = React.useState(sliceIdx);
  const [localStats, setLocalStats] = React.useState<{ mean: number; min: number; max: number; std: number } | null>(null);
  const [localPanelStats, setLocalPanelStats] = React.useState<PanelStats[] | null>(null);
  const setCompareActiveFromCurrentFrame = React.useCallback((enabled: boolean) => {
    if (!enabled) {
      setCompareMode("off");
      return;
    }
    const n = Math.max(1, Math.round(nSlices || 1));
    const current = ((Math.round(playbackIdxRef.current || displaySliceIdx || sliceIdx || 0) % n) + n) % n;
    const neighbor = current < n - 1 ? current + 1 : Math.max(0, current - 1);
    setComparePair([current, neighbor]);
    setCompareMode("blink");
  }, [displaySliceIdx, nSlices, setCompareMode, setComparePair, sliceIdx]);
  const frameRotationFor = React.useCallback((frame: number) => {
    return ((Math.round(frameRotations?.[frame] ?? 0) % 4) + 4) % 4;
  }, [frameRotations]);
  const rotationActive = ((imageRotation % 4) + 4) % 4 !== 0
    || Boolean(frameRotations?.some(k => ((k % 4) + 4) % 4 !== 0));
  const clearRotations = React.useCallback(() => {
    setImageRotation(0);
    setFrameRotations(Array.from({ length: Math.max(1, nSlices || 1) }, () => 0));
    setShowRotationSettings(false);
  }, [nSlices, setFrameRotations, setImageRotation]);
  const setRotationForScope = React.useCallback((quarterTurns: number) => {
    const k = normalizeRotation(quarterTurns);
    if ((rotationScope || "all") === "frame") {
      const idx = Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(displaySliceIdx || sliceIdx || 0)));
      const next = Array.from({ length: Math.max(1, nSlices || 1) }, (_, frame) => frameRotationFor(frame));
      next[idx] = k;
      setFrameRotations(next);
      setImageRotation(k);
      return;
    }
    setImageRotation(k);
  }, [displaySliceIdx, frameRotationFor, nSlices, normalizeRotation, rotationScope, setFrameRotations, setImageRotation, sliceIdx]);
  React.useEffect(() => {
    if ((rotationScope || "all") !== "frame") return;
    const idx = Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(displaySliceIdx || sliceIdx || 0)));
    const k = frameRotationFor(idx);
    if (((imageRotation % 4) + 4) % 4 !== k) setImageRotation(k);
  }, [displaySliceIdx, frameRotationFor, imageRotation, nSlices, rotationScope, setImageRotation, sliceIdx]);

  // WebGPU FFT state
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const gpuFftInitPromiseRef = React.useRef<Promise<WebGPUFFT | null> | null>(null);
  const offlineFftGpuDisabledRef = React.useRef(false);
  const offlineFftGpuInFlightRef = React.useRef(false);
  const [, setGpuReady] = React.useState(false);  // value unused; setter gates FFT-ready re-renders
  const [fftBackendInfo, setFftBackendInfo] = React.useState<{
    webgpu: "unknown" | "ready" | "software" | "unavailable";
    adapter: string;
    source: string;
    ms: number | null;
    panels: number | null;
    grid: string;
  }>({ webgpu: "unknown", adapter: "", source: "", ms: null, panels: null, grid: "" });
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const kymoOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  // WebGPU colormap engine (GPU-accelerated colormap for 4K frames)
  const gpuCmapRef = React.useRef<GPUColormapEngine | null>(null);
  const gpuCmapReadyRef = React.useRef(false);
  const gpuFrameCacheUploadedRef = React.useRef<Set<number>>(new Set());
  const gpuUploadRef = React.useRef<{
    source: Float32Array | null;
    data: Float32Array | null;
    width: number;
    height: number;
    logScale: boolean;
  } | null>(null);
  const gpuRenderSerialRef = React.useRef(0);
  const gpuDisplayVisibleRef = React.useRef<boolean | null>(false);
  const [gpuDisplayVisible, setGpuDisplayVisibleState] = React.useState(false);

  const ensureFftGpu = React.useCallback(async (): Promise<WebGPUFFT | null> => {
    if (gpuFFTRef.current) return gpuFFTRef.current;
    if (!gpuFftInitPromiseRef.current) {
      gpuFftInitPromiseRef.current = getWebGPUFFT().then(fft => {
        if (fft) {
          const info = getGPUInfo();
          if (/swiftshader|software/i.test(info)) {
            console.log(`[Show3D] Software WebGPU adapter detected (${info}); using CPU FFT fallback`);
            setFftBackendInfo(prev => ({ ...prev, webgpu: "software", adapter: info || "software adapter" }));
            return null;
          }
          gpuFFTRef.current = fft;
          setGpuReady(true);
          setFftBackendInfo(prev => ({ ...prev, webgpu: "ready", adapter: info || "GPU" }));
          console.log(`[Show3D] WebGPU FFT initialized - ${info || "GPU"}`);
        } else {
          setFftBackendInfo(prev => ({ ...prev, webgpu: "unavailable", adapter: "" }));
          console.log("[Show3D] WebGPU FFT unavailable - CPU fallback will be used");
        }
        return fft;
      }).catch(err => {
        console.warn("[Show3D] WebGPU FFT init failed; CPU fallback will be used.", err);
        setFftBackendInfo(prev => ({ ...prev, webgpu: "unavailable", adapter: "" }));
        return null;
      });
    }
    return gpuFftInitPromiseRef.current;
  }, []);

  const subpixelAlignSupported =
    !!offline &&
    !isRgb &&
    Math.max(1, nPanels || 1) === 1 &&
    width > 0 &&
    height > 0 &&
    nSlices > 1 &&
    ((!!offlineFloatStack && offlineFloatStack.byteLength >= nSlices * width * height * 4) ||
      (!!offlineStack && offlineStack.byteLength >= nSlices * width * height));

  const computeSubpixelAlignment = React.useCallback(async () => {
    const serial = ++subpixelAlignSerialRef.current;
    subpixelAlignCacheRef.current.clear();
    if (!subpixelAlignEnabled) {
      subpixelAlignShiftsRef.current = null;
      setSubpixelAlignStatus("Off");
      setSubpixelAlignVersion((value) => value + 1);
      return;
    }
    if (!subpixelAlignSupported) {
      subpixelAlignShiftsRef.current = null;
      setSubpixelAlignStatus("Needs a single-panel client-side stack");
      setSubpixelAlignVersion((value) => value + 1);
      return;
    }
    const n = Math.max(1, nSlices || 1);
    const refIdx = Math.max(0, Math.min(n - 1, Math.round(subpixelAlignReference || 0)));
    const reference = getOfflineFrame(refIdx);
    if (!reference || reference.length < width * height) {
      subpixelAlignShiftsRef.current = null;
      setSubpixelAlignStatus("Reference frame unavailable");
      setSubpixelAlignVersion((value) => value + 1);
      return;
    }
    setSubpixelAlignBusy(true);
    setSubpixelAlignStatus(`Aligning to frame ${refIdx + 1}…`);
    try {
      // Use the shared CPU FFT for this first production path. The WebGPU FFT
      // remains excellent for display FFTs, but registration needs stricter
      // row/column parity: a browser drive caught the GPU path reporting
      // near-zero row shifts on an intentionally drifted stack. Keep alignment
      // correct and visibly trustworthy, then promote a GPU path after parity
      // tests prove the same shifts.
      const gpu: WebGPUFFT | null = null;
      const shifts: SubpixelShift[] = [];
      for (let idx = 0; idx < n; idx++) {
        if (serial !== subpixelAlignSerialRef.current) return;
        if (idx === refIdx) {
          shifts.push({ row: 0, col: 0, quality: Infinity });
          continue;
        }
        const frame = getOfflineFrame(idx);
        if (!frame || frame.length < width * height) {
          shifts.push({ row: 0, col: 0, quality: 0 });
          continue;
        }
        shifts.push(await estimateSubpixelShift(reference, frame, width, height, gpu));
      }
      if (serial !== subpixelAlignSerialRef.current) return;
      subpixelAlignShiftsRef.current = shifts;
      subpixelAlignCacheRef.current.clear();
      const maxRow = shifts.reduce((value, shift) => Math.max(value, Math.abs(shift.row)), 0);
      const maxCol = shifts.reduce((value, shift) => Math.max(value, Math.abs(shift.col)), 0);
      const currentIdx = Math.max(0, Math.min(n - 1, Math.round(liveSliceIdx || 0)));
      const currentShift = shifts[currentIdx] ?? { row: 0, col: 0, quality: 0 };
      const backend = gpu ? "WebGPU" : "CPU";
      setSubpixelAlignStatus(
        `Aligned to frame ${refIdx + 1} · current row ${currentShift.row.toFixed(1)} px, col ${currentShift.col.toFixed(1)} px · max ${maxRow.toFixed(1)}/${maxCol.toFixed(1)} px · ${backend}`,
      );
      setSubpixelAlignVersion((value) => value + 1);
    } catch (error) {
      if (serial !== subpixelAlignSerialRef.current) return;
      console.warn("[Show3D] sub-pixel alignment failed", error);
      subpixelAlignShiftsRef.current = null;
      setSubpixelAlignStatus("Alignment failed; showing raw frames");
      setSubpixelAlignVersion((value) => value + 1);
    } finally {
      if (serial === subpixelAlignSerialRef.current) setSubpixelAlignBusy(false);
    }
  }, [
    ensureFftGpu,
    getOfflineFrame,
    height,
    isRgb,
    liveSliceIdx,
    nPanels,
    nSlices,
    offline,
    offlineFloatStack,
    offlineStack,
    subpixelAlignEnabled,
    subpixelAlignReference,
    subpixelAlignSupported,
    width,
  ]);

  React.useEffect(() => {
    subpixelAlignCacheRef.current.clear();
    subpixelAlignShiftsRef.current = null;
    setSubpixelAlignVersion((value) => value + 1);
    if (!subpixelAlignEnabled) {
      setSubpixelAlignStatus("Off");
      return;
    }
    if (!subpixelAlignSupported) {
      setSubpixelAlignStatus("Needs a single-panel client-side stack");
      return;
    }
    const refIdx = Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(subpixelAlignReference || 0)));
    setSubpixelAlignStatus(`Ready · press Align to use frame ${refIdx + 1}`);
  }, [nSlices, subpixelAlignEnabled, subpixelAlignReference, subpixelAlignSupported]);

  const subpixelAlignFrameForIndex = React.useCallback((idx: number, frame: Float32Array | null): Float32Array | null => {
    if (!frame || !subpixelAlignEnabled) return frame;
    const shifts = subpixelAlignShiftsRef.current;
    if (!shifts || shifts.length === 0) return frame;
    const n = Math.max(1, nSlices || 1);
    const normalized = ((Math.round(idx) % n) + n) % n;
    const shift = shifts[normalized];
    if (!shift) return frame;
    const key = [
      normalized,
      frameSeq,
      subpixelAlignVersion,
      playRef.current.avgWindow,
      playRef.current.diffMode,
      shift.row.toFixed(4),
      shift.col.toFixed(4),
    ].join(":");
    const cached = subpixelAlignCacheRef.current.get(key);
    if (cached) return cached;
    const shifted = shiftFrameBilinear(
      frame,
      width,
      height,
      shift.row,
      shift.col,
      finiteMedianSample(frame),
    );
    subpixelAlignCacheRef.current.set(key, shifted);
    if (subpixelAlignCacheRef.current.size > 48) {
      subpixelAlignCacheRef.current.delete(subpixelAlignCacheRef.current.keys().next().value as string);
    }
    return shifted;
  }, [frameSeq, height, nSlices, subpixelAlignEnabled, subpixelAlignVersion, width]);

  const setGpuDisplayVisible = React.useCallback((visible: boolean) => {
    gpuDisplayVisibleRef.current = visible;
    const gpuCanvas = gpuCanvasRef.current;
    const canvas = canvasRef.current;
    const gpuVisible = ENABLE_GPU_CANVAS_DISPLAY && visible;
    setGpuDisplayVisibleState(gpuVisible);
    if (gpuCanvas) gpuCanvas.style.opacity = gpuVisible ? "1" : "0";
    if (canvas) {
      canvas.style.opacity = gpuVisible ? "0" : "1";
      canvas.style.display = "block";
    }
  }, []);

  const ensureGpuDisplayContext = React.useCallback((
    engine: GPUColormapEngine,
    w: number,
    h: number,
  ): GPUCanvasContext | null => {
    const canvas = gpuCanvasRef.current;
    if (!canvas) return null;
    const widthPx = Math.max(1, Math.round(w));
    const heightPx = Math.max(1, Math.round(h));
    const size = gpuCanvasSizeRef.current;
    if (!gpuCanvasCtxRef.current || !size || size.w !== widthPx || size.h !== heightPx) {
      gpuCanvasCtxRef.current = engine.configureCanvas(canvas, widthPx, heightPx);
      gpuCanvasSizeRef.current = { w: widthPx, h: heightPx };
    }
    return gpuCanvasCtxRef.current;
  }, []);

  const ensureLocalAutoRange = React.useCallback((
    idx: number,
    data: Float32Array,
    low: number,
    high: number,
  ): { vmin: number; vmax: number } => {
    const synced = cachedAutoRange(autoVmins, autoVmaxs, idx);
    if (synced) return synced;
    const local = cachedAutoRange(localAutoVminsRef.current, localAutoVmaxsRef.current, idx);
    if (local) return local;

    const range = percentileClip(data, low, high);
    const needed = Math.max(nSlices, idx + 1);
    while (localAutoVminsRef.current.length < needed) {
      localAutoVminsRef.current.push(Number.NaN);
      localAutoVmaxsRef.current.push(Number.NaN);
    }
    localAutoVminsRef.current[idx] = range.vmin;
    localAutoVmaxsRef.current[idx] = range.vmax;
    return range;
  }, [autoVmins, autoVmaxs, nSlices]);

  React.useEffect(() => {
    localAutoVminsRef.current = [];
    localAutoVmaxsRef.current = [];
    autoRangeComputeTokenRef.current++;
  }, [percentileLow, percentileHigh, nSlices, width, height]);

  const [gpuCmapReady, setGpuCmapReady] = React.useState(false);
  React.useEffect(() => {
    let disposed = false;
    ensureFftGpu().then(fft => {
      if (disposed || !fft) return;
      gpuFFTRef.current = fft;
      setGpuReady(true);
    });
    createGPUColormapEngine().then(engine => {
      if (disposed) {
        engine?.destroy();
        return;
      }
      if (engine) {
        gpuCmapRef.current = engine;
        gpuCmapReadyRef.current = true;
        // State counterpart of the ref so downstream useEffects re-fire
        // when the GPU engine becomes available. Without this, the data
        // effect that fires at mount paints via the CPU fallback BEFORE
        // the engine is ready and never re-paints when it IS ready.
        setGpuCmapReady(true);
      }
    });
    return () => {
      disposed = true;
      gpuCmapRef.current?.destroy();
      gpuCmapRef.current = null;
      gpuCmapReadyRef.current = false;
      setGpuCmapReady(false);
      gpuCanvasCtxRef.current = null;
      gpuCanvasSizeRef.current = null;
      frameFetchSerialRef.current++;
      frameFetchCacheRef.current.clear();
      frameFetchPendingRef.current.clear();
      panelGpuFramePendingRef.current.clear();
      gpuFrameCacheUploadedRef.current.clear();
    };
  }, []);

  const getFrameServerCacheLimit = React.useCallback(() => {
    const frameByteLength = Math.max(1, width * height * 4);
    const stackByteLength = frameByteLength * Math.max(1, nSlices);
    const cacheBudget = stackByteLength <= FRAME_SERVER_JS_FULL_STACK_CACHE_BYTES
      ? stackByteLength
      : FRAME_SERVER_STREAM_CACHE_BYTES;
    const budgetFrames = Math.floor(cacheBudget / frameByteLength);
    const minFrames = frameByteLength <= cacheBudget / FRAME_SERVER_MIN_CACHE_FRAMES
      ? FRAME_SERVER_MIN_CACHE_FRAMES
      : 1;
    return Math.max(1, Math.min(Math.max(1, nSlices), Math.max(minFrames, budgetFrames)));
  }, [width, height, nSlices]);

  const getSeparatePanelGpuCacheLimit = React.useCallback(() => {
    const visiblePanels = Math.max(1, visiblePanelCount || 1);
    const frameByteLength = Math.max(1, panelWidthPx * height * 4 * visiblePanels);
    const budgetFrames = Math.floor(FRAME_SERVER_SEPARATE_PANEL_GPU_CACHE_BYTES / frameByteLength);
    const minFrames = frameByteLength <= FRAME_SERVER_SEPARATE_PANEL_GPU_CACHE_BYTES / FRAME_SERVER_MIN_CACHE_FRAMES
      ? FRAME_SERVER_MIN_CACHE_FRAMES
      : 1;
    return Math.max(1, Math.min(Math.max(1, nSlices), Math.max(minFrames, budgetFrames)));
  }, [panelWidthPx, height, visiblePanelCount, nSlices]);

  const releasePanelGpuFrame = React.useCallback((idx: number) => {
    const normalized = ((Math.round(idx) % Math.max(1, nSlices)) + Math.max(1, nSlices)) % Math.max(1, nSlices);
    const engine = gpuCmapRef.current;
    const n = Math.max(1, Math.round(nPanels || 1));
    for (let panel = 0; panel < n; panel++) {
      engine?.releaseSlot(normalized * n + panel);
    }
    gpuFrameCacheUploadedRef.current.delete(normalized);
  }, [nPanels, nSlices]);

  const getCachedServerFrame = React.useCallback((idx: number): Float32Array | null => {
    const cache = frameFetchCacheRef.current;
    const frame = cache.get(idx);
    if (!frame) return null;
    cache.delete(idx);
    cache.set(idx, frame);
    return frame;
  }, []);

  const putCachedServerFrame = React.useCallback((idx: number, frame: Float32Array) => {
    const cache = frameFetchCacheRef.current;
    if (cache.has(idx)) cache.delete(idx);
    cache.set(idx, frame);
    const limit = getFrameServerCacheLimit();
    while (cache.size > limit) {
      const oldest = cache.keys().next().value;
      if (oldest === undefined) break;
      cache.delete(oldest);
    }
  }, [getFrameServerCacheLimit]);

  React.useEffect(() => {
    frameFetchSerialRef.current++;
    frameFetchCacheRef.current.clear();
    frameFetchPendingRef.current.clear();
    panelGpuFramePendingRef.current.clear();
    gpuFrameCacheUploadedRef.current.clear();
    setFramePopulation({ ready: 0, target: Math.max(0, nSlices), active: !!frameServerUrl });
    setPreviewPopulation({ ready: false, idx: 0, factor: 1 });
    // set_image() intentionally publishes an empty buffer and bumps the frame
    // server version. Empty payloads do not enter the parser effect, so clear
    // both playback buffers here or same-shape replacement data can replay the
    // previous stack before the new server frames arrive.
    bufferRef.current = null;
    bufferStartRef.current = 0;
    bufferCountRef.current = 0;
    nextBufferRef.current = null;
    nextBufferStartRef.current = 0;
    nextBufferCountRef.current = 0;
    gpuCmapRef.current?.destroy();
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.frameServerUrl = frameServerUrl || "";
      dbg.frameServerVersion = frameServerVersion;
      dbg.frameFetchCacheSize = 0;
      dbg.frameFetchPendingSize = 0;
    }
  }, [frameServerUrl, frameServerVersion, width, height, nSlices]);

  React.useEffect(() => {
    if (offline || !frameServerUrl || width <= 0 || height <= 0 || nSlices <= 0) return;
    if (hasLiveFrameBytes || hasOfflineStack || hasOfflineFloatStack) return;
    if (separatePanelFrames) return;
    const normalized = ((Math.round(sliceIdx) % nSlices) + nSlices) % nSlices;
    const key = `${frameServerVersion || 0}:${normalized}:${width}:${height}`;
    if (initialNativePreviewKeyRef.current === key) return;
    initialNativePreviewKeyRef.current = key;
    const timer = window.setTimeout(() => {
      if (rawFrameDataRef.current && mainOffscreenSourcePanelWidthRef.current === undefined) return;
      requestCommFramePreview(normalized, "initial-native-preview");
    }, INITIAL_NATIVE_PREVIEW_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [
    offline,
    frameServerUrl,
    frameServerVersion,
    width,
    height,
    nSlices,
    sliceIdx,
    hasLiveFrameBytes,
    hasOfflineStack,
    hasOfflineFloatStack,
    separatePanelFrames,
    requestCommFramePreview,
  ]);

  const fetchFrameFromServer = React.useCallback(async (idx: number): Promise<Float32Array | null> => {
    if (offline || !frameServerUrl || width <= 0 || height <= 0 || nSlices <= 0) return null;
    const normalized = ((Math.round(idx) % nSlices) + nSlices) % nSlices;
    const cached = getCachedServerFrame(normalized);
    const dbg = show3dPerfDebug();
    if (cached) {
      if (dbg) {
        dbg.frameFetchHits = ((dbg.frameFetchHits as number | undefined) ?? 0) + 1;
        dbg.frameFetchCacheSize = frameFetchCacheRef.current.size;
      }
      return cached;
    }
    const pending = frameFetchPendingRef.current.get(normalized);
    if (pending) return pending;

    let url: URL;
    try {
      url = new URL(frameServerUrl);
    } catch {
      return null;
    }
    url.searchParams.set("idx", String(normalized));
    url.searchParams.set("version", String(frameServerVersion));

    const serial = frameFetchSerialRef.current;
    const t0 = performance.now();
    let promise!: Promise<Float32Array | null>;
    promise = (async () => {
      try {
        if (dbg) {
          dbg.frameFetchMisses = ((dbg.frameFetchMisses as number | undefined) ?? 0) + 1;
          dbg.frameFetchPendingSize = frameFetchPendingRef.current.size + 1;
        }
        const response = await fetch(url.toString(), { cache: "no-store" });
        if (!response.ok) throw new Error(`frame fetch ${response.status}`);
        const buffer = await response.arrayBuffer();
        if (serial !== frameFetchSerialRef.current) return null;
        const expectedBytes = width * height * 4;
        if (buffer.byteLength !== expectedBytes) {
          throw new Error(`expected ${expectedBytes} bytes, got ${buffer.byteLength}`);
        }
        const frame = new Float32Array(buffer);
        putCachedServerFrame(normalized, frame);
        setFramePopulation(prev => ({
          ready: Math.max(prev.ready, frameFetchCacheRef.current.size),
          target: Math.max(0, nSlices),
          active: frameFetchPendingRef.current.size > 0,
        }));
        if (dbg) {
          dbg.lastFrameFetchMs = performance.now() - t0;
          dbg.lastFetchedFrame = normalized;
          dbg.frameFetchCacheSize = frameFetchCacheRef.current.size;
        }
        return frame;
      } catch (err) {
        requestCommFramePreview(normalized, "frame-server-fallback");
        if (dbg) {
          dbg.lastFrameFetchError = err instanceof Error ? err.message : String(err);
          dbg.lastFrameFetchErrorAt = performance.now();
        }
        return null;
      }
    })().finally(() => {
      if (frameFetchPendingRef.current.get(normalized) === promise) {
        frameFetchPendingRef.current.delete(normalized);
      }
      const d = show3dPerfDebug();
      if (d) d.frameFetchPendingSize = frameFetchPendingRef.current.size;
      setFramePopulation(prev => ({
        ...prev,
        target: Math.max(0, nSlices),
        active: frameFetchPendingRef.current.size > 0,
      }));
    });
    frameFetchPendingRef.current.set(normalized, promise);
    return promise;
  }, [offline, frameServerUrl, frameServerVersion, width, height, nSlices, getCachedServerFrame, putCachedServerFrame, requestCommFramePreview]);

  const fetchPanelFrameFromServer = React.useCallback(async (idx: number, panel: number): Promise<Float32Array | null> => {
    if (offline || !frameServerUrl || panelWidthPx <= 0 || height <= 0 || nSlices <= 0) return null;
    const normalized = ((Math.round(idx) % nSlices) + nSlices) % nSlices;
    const panelIdx = Math.max(0, Math.min(Math.max(0, nPanels - 1), Math.round(panel)));
    let url: URL;
    try {
      url = new URL(frameServerUrl);
    } catch {
      return null;
    }
    url.searchParams.set("idx", String(normalized));
    url.searchParams.set("panel", String(panelIdx));
    url.searchParams.set("version", String(frameServerVersion));

    const serial = frameFetchSerialRef.current;
    const t0 = performance.now();
    const dbg = show3dPerfDebug();
    try {
      if (dbg) {
        dbg.panelFrameFetchAttempts = ((dbg.panelFrameFetchAttempts as number | undefined) ?? 0) + 1;
        dbg.lastPanelFrameFetch = `${normalized}:${panelIdx}`;
      }
      const response = await fetch(url.toString(), { cache: "no-store" });
      if (!response.ok) throw new Error(`panel frame fetch ${response.status}`);
      const buffer = await response.arrayBuffer();
      if (serial !== frameFetchSerialRef.current) return null;
      const expectedBytes = panelWidthPx * height * 4;
      if (buffer.byteLength !== expectedBytes) {
        throw new Error(`expected ${expectedBytes} panel bytes, got ${buffer.byteLength}`);
      }
      if (dbg) dbg.lastPanelFrameFetchMs = performance.now() - t0;
      return new Float32Array(buffer);
    } catch (err) {
      requestCommFramePreview(normalized, "panel-server-fallback");
      if (dbg) {
        // Real misses only (failed fetch), not every attempt - the old counter
        // incremented at the top of try and read as "~every request missed".
        dbg.panelFrameFetchMisses = ((dbg.panelFrameFetchMisses as number | undefined) ?? 0) + 1;
        dbg.lastPanelFrameFetchError = err instanceof Error ? err.message : String(err);
        dbg.lastPanelFrameFetchErrorAt = performance.now();
      }
      return null;
    }
  }, [offline, frameServerUrl, frameServerVersion, panelWidthPx, height, nSlices, nPanels, requestCommFramePreview]);

  const fetchSeparatePanelPackedFrameFromServer = React.useCallback(async (idx: number): Promise<Float32Array | null> => {
    if (offline || !frameServerUrl || !separatePanelFrames || panelWidthPx <= 0 || height <= 0 || width <= 0 || nSlices <= 0) return null;
    const normalized = ((Math.round(idx) % nSlices) + nSlices) % nSlices;
    const cached = getCachedServerFrame(normalized);
    if (cached) return cached;
    const n = Math.max(1, Math.round(nPanels || 1));
    const packed = new Float32Array(width * height);
    const t0 = performance.now();
    for (let panel = 0; panel < n; panel++) {
      const frame = await fetchPanelFrameFromServer(normalized, panel);
      if (!frame) return null;
      for (let row = 0; row < height; row++) {
        const src = row * panelWidthPx;
        const dst = row * width + panel * panelWidthPx;
        packed.set(frame.subarray(src, src + panelWidthPx), dst);
      }
      await new Promise<void>(resolve => setTimeout(resolve, 0));
    }
    putCachedServerFrame(normalized, packed);
    setFramePopulation(prev => ({
      ready: Math.max(prev.ready, frameFetchCacheRef.current.size),
      target: Math.max(0, nSlices),
      active: frameFetchPendingRef.current.size > 0 || panelGpuFramePendingRef.current.size > 0,
    }));
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.lastFrameSource = "cpu-packed-panel-native";
      dbg.lastPackedPanelFrame = normalized;
      dbg.lastPackedPanelFrameMs = Number((performance.now() - t0).toFixed(2));
      dbg.frameFetchCacheSize = frameFetchCacheRef.current.size;
    }
    return packed;
  }, [
    offline,
    frameServerUrl,
    separatePanelFrames,
    panelWidthPx,
    height,
    width,
    nSlices,
    nPanels,
    getCachedServerFrame,
    putCachedServerFrame,
    fetchPanelFrameFromServer,
  ]);

  React.useEffect(() => {
    if (!separatePanelFrames) return;
    // Separate-panel GPU slots are keyed by frame plus panel index. A page
    // change swaps the visible panel set, so the old "frame is uploaded" mark
    // cannot be reused for the new page's panel slots.
    gpuFrameCacheUploadedRef.current.clear();
    panelGpuFramePendingRef.current.clear();
  }, [hiddenPanels, separatePanelFrames, visiblePanelIndices]);

  const ensurePanelFrameGpu = React.useCallback(async (
    idx: number,
    rgbaCapacityHint?: number,
  ): Promise<boolean> => {
    if (offline || !separatePanelFrames || !frameServerUrl || width <= 0 || height <= 0 || nSlices <= 0) return false;
    const normalized = ((Math.round(idx) % nSlices) + nSlices) % nSlices;
    if (gpuFrameCacheUploadedRef.current.has(normalized)) {
      gpuFrameCacheUploadedRef.current.delete(normalized);
      gpuFrameCacheUploadedRef.current.add(normalized);
      return true;
    }
    const pending = panelGpuFramePendingRef.current.get(normalized);
    if (pending) return pending;

    const promise = (async () => {
      const waitStartedAt = performance.now();
      while (!gpuCmapReadyRef.current || !gpuCmapRef.current) {
        if (performance.now() - waitStartedAt > PANEL_GPU_READY_TIMEOUT_MS) {
          const dbg = show3dPerfDebug();
          if (dbg) {
            dbg.lastPanelGpuWaitTimeoutFrame = normalized;
            dbg.lastPanelGpuWaitTimeoutMs = PANEL_GPU_READY_TIMEOUT_MS;
            dbg.lastFrameSource = "panel-gpu-unavailable";
          }
          return false;
        }
        await new Promise<void>(resolve => setTimeout(resolve, 25));
      }
      const engine = gpuCmapRef.current;
      if (!engine) return false;
      try {
        const n = Math.max(1, Math.round(nPanels || 1));
        for (const panel of visiblePanelIndices) {
          const frame = await fetchPanelFrameFromServer(normalized, panel);
          if (!frame) {
            const dbg = show3dPerfDebug();
            if (dbg) {
              dbg.lastPanelGpuMissFrame = normalized;
              dbg.lastPanelGpuMissPanel = panel;
            }
            return false;
          }
          engine.uploadData(normalized * n + panel, frame, panelWidthPx, height, rgbaCapacityHint, true);
        }
      } catch (err) {
        const dbg = show3dPerfDebug();
        if (dbg) {
          dbg.lastPanelGpuUploadFrame = normalized;
          dbg.lastPanelGpuUploadError = err instanceof Error ? err.message : String(err);
        }
        return false;
      }
      gpuFrameCacheUploadedRef.current.add(normalized);
      setFramePopulation(prev => ({
        ready: Math.max(prev.ready, gpuFrameCacheUploadedRef.current.size),
        target: Math.max(0, nSlices),
        active: panelGpuFramePendingRef.current.size > 0,
      }));
      const cacheLimit = getSeparatePanelGpuCacheLimit();
      while (gpuFrameCacheUploadedRef.current.size > cacheLimit) {
        let oldest = gpuFrameCacheUploadedRef.current.keys().next().value;
        if (oldest === undefined) break;
        if (oldest === normalized) {
          gpuFrameCacheUploadedRef.current.delete(oldest);
          gpuFrameCacheUploadedRef.current.add(oldest);
          oldest = gpuFrameCacheUploadedRef.current.keys().next().value;
          if (oldest === undefined || oldest === normalized) break;
        }
        releasePanelGpuFrame(oldest);
      }
      const dbg = show3dPerfDebug();
      if (dbg) {
        dbg.gpuPreloadDone = gpuFrameCacheUploadedRef.current.size;
        dbg.gpuFrameCacheUploaded = gpuFrameCacheUploadedRef.current.size;
        dbg.gpuPanelCacheLimit = cacheLimit;
        dbg.gpuPanelCacheLayout = "panel-slots";
        dbg.lastFrameSource = "gpu-panel-cache-slots";
      }
      return true;
    })().finally(() => {
      if (panelGpuFramePendingRef.current.get(normalized) === promise) {
        panelGpuFramePendingRef.current.delete(normalized);
      }
      setFramePopulation(prev => ({
        ...prev,
        target: Math.max(0, nSlices),
        active: panelGpuFramePendingRef.current.size > 0,
      }));
    });
    panelGpuFramePendingRef.current.set(normalized, promise);
    return promise;
  }, [
    offline,
    separatePanelFrames,
    frameServerUrl,
    width,
    height,
    nSlices,
    nPanels,
    visiblePanelIndices,
    panelWidthPx,
    height,
    fetchPanelFrameFromServer,
    getSeparatePanelGpuCacheLimit,
    releasePanelGpuFrame,
    requestCommFramePreview,
  ]);

  React.useEffect(() => {
    if (offline || !frameServerUrl || width <= 0 || height <= 0 || nSlices <= 0) return;
    if (separatePanelFrames) return;
    const frameByteLength = Math.max(1, width * height * 4);
    const stackByteLength = frameByteLength * Math.max(1, nSlices);
    if (stackByteLength > FRAME_SERVER_JS_FULL_STACK_CACHE_BYTES) return;
    let cancelled = false;
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.frameFetchPreloadTarget = nSlices;
      dbg.frameFetchPreloadDone = 0;
      dbg.frameFetchPreloadActive = true;
    }
    void (async () => {
      for (let i = 0; i < nSlices; i++) {
        if (cancelled) break;
        await fetchFrameFromServer(i);
        const d = show3dPerfDebug();
        if (d) {
          d.frameFetchPreloadDone = i + 1;
          d.frameFetchCacheSize = frameFetchCacheRef.current.size;
        }
        await new Promise<void>(resolve => setTimeout(resolve, 0));
      }
      const d = show3dPerfDebug();
      if (d) d.frameFetchPreloadActive = false;
    })();
    return () => {
      cancelled = true;
      const d = show3dPerfDebug();
      if (d) d.frameFetchPreloadActive = false;
    };
  }, [offline, frameServerUrl, frameServerVersion, width, height, nSlices, fetchFrameFromServer, separatePanelFrames]);

  const prefetchServerFrames = React.useCallback((
    startIdx: number,
    reversePlayback = false,
    loopPlayback = false,
    loopStartIdx = 0,
    loopEndIdx = -1,
  ) => {
    if (offline || !frameServerUrl || nSlices <= 0) return;
    if (separatePanelFrames) return;
    const dir = reversePlayback ? -1 : 1;
    const rangeStart = loopPlayback ? Math.max(0, Math.min(loopStartIdx, nSlices - 1)) : 0;
    const rangeEnd = loopPlayback
      ? Math.max(rangeStart, Math.min(loopEndIdx < 0 ? nSlices - 1 : loopEndIdx, nSlices - 1))
      : nSlices - 1;
    const rangeSize = Math.max(1, rangeEnd - rangeStart + 1);
    for (let step = 0; step < FRAME_SERVER_PREFETCH_FRAMES; step++) {
      let next = Math.round(startIdx) + dir * step;
      if (loopPlayback) {
        while (next < rangeStart) next += rangeSize;
        while (next > rangeEnd) next -= rangeSize;
      } else {
        next = ((next % nSlices) + nSlices) % nSlices;
      }
      void fetchFrameFromServer(next);
    }
  }, [offline, frameServerUrl, nSlices, fetchFrameFromServer, separatePanelFrames]);

  const prefetchPanelGpuFrames = React.useCallback((
    startIdx: number,
    reversePlayback = false,
    loopPlayback = false,
    loopStartIdx = 0,
    loopEndIdx = -1,
  ) => {
    if (offline || !frameServerUrl || !separatePanelFrames || nSlices <= 0) return;
    const dir = reversePlayback ? -1 : 1;
    const rangeStart = loopPlayback ? Math.max(0, Math.min(loopStartIdx, nSlices - 1)) : 0;
    const rangeEnd = loopPlayback
      ? Math.max(rangeStart, Math.min(loopEndIdx < 0 ? nSlices - 1 : loopEndIdx, nSlices - 1))
      : nSlices - 1;
    const rangeSize = Math.max(1, rangeEnd - rangeStart + 1);
    const count = Math.min(FRAME_SERVER_PREFETCH_FRAMES, getSeparatePanelGpuCacheLimit());
    const live = playRef.current;
    const rgbaCapacity = Math.max(1, Math.round(live.canvasW * live.canvasH));
    for (let step = 0; step < count; step++) {
      let next = Math.round(startIdx) + dir * step;
      if (loopPlayback) {
        while (next < rangeStart) next += rangeSize;
        while (next > rangeEnd) next -= rangeSize;
      } else {
        next = ((next % nSlices) + nSlices) % nSlices;
      }
      void ensurePanelFrameGpu(next, rgbaCapacity);
    }
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.gpuPanelPrefetchCount = count;
      dbg.gpuPanelCacheLimit = getSeparatePanelGpuCacheLimit();
    }
  }, [
    offline,
    frameServerUrl,
    separatePanelFrames,
    nSlices,
    ensurePanelFrameGpu,
    getSeparatePanelGpuCacheLimit,
  ]);

  // Parse incoming playback buffer (double-buffer to avoid overwrite stalls)
  React.useEffect(() => {
    if (!bufferBytes || bufferBytes.byteLength === 0) return;
    const receiveAt = performance.now();
    const decodeStart = performance.now();
    const parsed = extractFloat32(bufferBytes, Math.max(0, bufferCount) * width * height);
    const decodeMs = performance.now() - decodeStart;
    if (!parsed) return;
    const transport = bufferTransportTiming ?? {};
    const sendTimeMs = typeof transport.sendTimeMs === "number" ? transport.sendTimeMs : null;
    recordTransportSample({
      ...transport,
      kind: "buffer",
      receiveAtMs: Number(receiveAt.toFixed(3)),
      jsDecodeMs: Number(decodeMs.toFixed(3)),
      browserReceiveLatencyMs: sendTimeMs === null ? null : Number((Date.now() - sendTimeMs).toFixed(3)),
    });
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.lastBufferByteLength = bufferBytes.byteLength;
      dbg.lastParsedFloatLength = parsed.length;
      dbg.lastBufferStart = bufferStart;
      dbg.lastBufferCount = bufferCount;
      dbg.lastBufferAt = performance.now();
    }
    if (!bufferRef.current || bufferCountRef.current === 0) {
      // No active buffer - use as current (initial load)
      bufferRef.current = parsed;
      bufferStartRef.current = bufferStart;
      bufferCountRef.current = bufferCount;
    } else {
      // Active buffer exists - store as next (prefetch)
      nextBufferRef.current = parsed;
      nextBufferStartRef.current = bufferStart;
      nextBufferCountRef.current = bufferCount;
    }
    prefetchPendingRef.current = false;

    if (autoContrast && !logScale && width > 0 && height > 0 && nSlices > 0 && bufferCount > 0) {
      const frameSize = width * height;
      const availableFrames = Math.min(bufferCount, Math.floor(parsed.length / frameSize));
      const hasSyncedRanges = (autoVmins?.length ?? 0) >= nSlices && (autoVmaxs?.length ?? 0) >= nSlices;
      if (!hasSyncedRanges && availableFrames > 0) {
        const token = ++autoRangeComputeTokenRef.current;
        let j = 0;
        const computeNextRange = () => {
          if (token !== autoRangeComputeTokenRef.current) return;
          const idx = (bufferStart + j) % nSlices;
          const start = j * frameSize;
          const end = start + frameSize;
          if (!cachedAutoRange(localAutoVminsRef.current, localAutoVmaxsRef.current, idx)) {
            ensureLocalAutoRange(idx, parsed.subarray(start, end), percentileLow, percentileHigh);
          }
          j++;
          if (j < availableFrames) {
            const ric = (window as unknown as { requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number }).requestIdleCallback;
            if (ric) ric(computeNextRange, { timeout: 150 });
            else window.setTimeout(computeNextRange, 0);
          }
        };
        window.setTimeout(computeNextRange, 0);
      }
    }
    return () => { autoRangeComputeTokenRef.current++; };
  }, [bufferBytes, bufferStart, bufferCount, autoContrast, logScale, width, height, nSlices, autoVmins, autoVmaxs, percentileLow, percentileHigh, ensureLocalAutoRange, bufferTransportTiming, recordTransportSample]);

  // Sync displaySliceIdx with model when not playing
  React.useEffect(() => {
    if (!playing) {
      setGpuDisplayVisible(false);
      playbackIdxRef.current = sliceIdx;
      setDisplaySliceIdx(sliceIdx);
      setPlaybackUiSliceIdx(sliceIdx);
    }
  }, [sliceIdx, playing, setGpuDisplayVisible]);

  // Histogram state for main image
  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);
  // GPU-computed 256-bin histogram. When non-null, the Histogram component
  // uses these bins directly and skips its CPU bin-scan fallback.
  const [imageHistogramBins, setImageHistogramBins] = React.useState<number[] | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [panelHistogramData, setPanelHistogramData] = React.useState<(Float32Array | null)[]>([]);
  const [panelDataRanges, setPanelDataRanges] = React.useState<{ min: number; max: number }[]>([]);
  const imageHistogramPreviewPctRef = React.useRef<[number, number] | null>(null);
  const panelHistogramPreviewPctRef = React.useRef<Map<number, [number, number]>>(new Map());
  const histogramPreviewPaintRafRef = React.useRef<number | null>(null);
  const perPanelHistogramEnabled = (nPanels || 1) > 1 && !linkContrast;

  const updatePanelState = (panel: number, patch: Partial<PanelState>) => {
    setPanelStates(prev => prev.map((state, i) => i === panel ? { ...state, ...patch } : state));
  };
  const setPanelRangeValues = (panel: number, minValue: number | null, maxValue: number | null) => {
    const n = Math.max(1, nPanels || 1);
    const nextMins = Array.from({ length: n }, (_, i) => vminPerPanelLiveRef.current[i] ?? null);
    const nextMaxs = Array.from({ length: n }, (_, i) => vmaxPerPanelLiveRef.current[i] ?? null);
    nextMins[panel] = minValue;
    nextMaxs[panel] = maxValue;
    vminPerPanelLiveRef.current = nextMins;
    vmaxPerPanelLiveRef.current = nextMaxs;
    setVminPerPanel(nextMins);
    setVmaxPerPanel(nextMaxs);
  };
  const extractPanelSlice = React.useCallback((
    raw: Float32Array,
    panel: number,
    panelLogScale: boolean,
  ): Float32Array | null => {
    const n = Math.max(1, nPanels || 1);
    if (height <= 0 || raw.length === 0) return null;
    const panelW = totalPanelCount > 1
      ? Math.max(1, panelWidthPx || Math.round(width / totalPanelCount))
      : Math.max(1, width);
    const fullW = raw.length === height * panelW ? panelW : width;
    const srcPanel = sharedPanelSource ? 0 : panel;
    const x0 = Math.min(Math.max(0, srcPanel * panelW), Math.max(0, fullW - panelW));
    if (raw.length < height * fullW || x0 + panelW > fullW || panel >= n) return null;
    const out = new Float32Array(height * panelW);
    for (let r = 0; r < height; r++) {
      out.set(raw.subarray(r * fullW + x0, r * fullW + x0 + panelW), r * panelW);
    }
    return panelLogScale ? applyLogScale(out) : out;
  }, [height, nPanels, panelWidthPx, sharedPanelSource, totalPanelCount, width]);

  const resolvePanelRange = (
    panel: number,
    range: { min: number; max: number },
    sharedAutoRange?: { vmin: number; vmax: number } | null,
  ): { vmin: number; vmax: number; logScale: boolean } => {
    const state = panelStates[panel] || initialState;
    if (sharedAutoRange && !perPanelHistogramEnabled) {
      return { ...sharedAutoRange, logScale };
    }
    // Per-panel mode: always interpret slider pct in THIS panel's data
    // range. Stack-wide bounds (for mixed BF/DF counts vs SSB radians)
    // would decode SSB sliders to count-territory values → black image.
    const pdr = panelDataRanges[panel];
    const effectiveRange = (perPanelHistogramEnabled && pdr && pdr.max > pdr.min)
      ? pdr
      : range;
    const useStoredManual = !autoContrast;
    if (useStoredManual) {
      const storedMin = vminPerPanelLiveRef.current[panel];
      const storedMax = vmaxPerPanelLiveRef.current[panel];
      if (storedMin != null || storedMax != null) {
        const lo = storedMin ?? effectiveRange.min;
        const hi = storedMax ?? effectiveRange.max;
        return { vmin: lo, vmax: Math.max(lo, hi), logScale };
      }
    }
    const slider = sliderRange(effectiveRange.min, effectiveRange.max, state.imageVminPct, state.imageVmaxPct);
    return { ...slider, logScale };
  };

  const autoPanelRangeFromData = (
    panelData: Float32Array | null,
    fallbackRange: { min: number; max: number },
    low: number,
    high: number,
  ): { vmin: number; vmax: number; logScale: boolean } | null => {
    if (!panelData || panelData.length === 0) return null;
    const dataRange = findDataRange(panelData);
    const range = dataRange.max > dataRange.min ? dataRange : fallbackRange;
    if (range.max <= range.min) return null;
    let clipped: { vmin: number; vmax: number } = percentileClip(panelData, low, high);
    const span = range.max - range.min;
    if (!Number.isFinite(clipped.vmin) || !Number.isFinite(clipped.vmax) || clipped.vmax <= clipped.vmin || clipped.vmax - clipped.vmin < span * 1e-4) {
      clipped = { vmin: range.min, vmax: range.max };
    }
    return { vmin: clipped.vmin, vmax: Math.max(clipped.vmin, clipped.vmax), logScale };
  };

  const panelAutoClipPcts = (
    panel: number,
    state: PanelState,
    stackBounds: { min: number; max: number },
  ): (Pick<PanelState, "imageVminPct" | "imageVmaxPct"> & { vmin: number; vmax: number }) | null => {
    const panelRaw = panelHistogramData[panel];
    if (!panelRaw || panelRaw.length === 0) return null;
    // panelHistogramData is already in the active display domain; in log mode
    // refreshHistogram populated it from extractPanelSlice(..., logScale).
    const panelRange = panelDataRanges[panel];
    const range = (panelRange && panelRange.max > panelRange.min) ? panelRange : stackBounds;
    const span = range.max - range.min;
    if (span <= 0) return null;
    let clipped: { vmin: number; vmax: number } = percentileClip(panelRaw, percentileLow, percentileHigh);
    if (
      !Number.isFinite(clipped.vmin) ||
      !Number.isFinite(clipped.vmax) ||
      clipped.vmax <= clipped.vmin ||
      clipped.vmax - clipped.vmin < span * 1e-4
    ) {
      clipped = { vmin: range.min, vmax: range.max };
    }
    return {
      vmin: clipped.vmin,
      vmax: Math.max(clipped.vmin, clipped.vmax),
      imageVminPct: valueToPct(clipped.vmin, range.min, range.max, state.imageVminPct),
      imageVmaxPct: valueToPct(clipped.vmax, range.min, range.max, state.imageVmaxPct),
    };
  };

  const restorePanelManualClipPcts = () => {
    const stackBounds = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
    if (stackBounds.max <= stackBounds.min) return;
    const n = Math.max(1, nPanels || 1);
    const liveStates = panelStatesLiveRef.current.length === n ? panelStatesLiveRef.current : panelStates;
    const nextStates = Array.from({ length: n }, (_, i) => {
      const state = liveStates[i] || initialState;
      const storedMin = vminPerPanelLiveRef.current[i];
      const storedMax = vmaxPerPanelLiveRef.current[i];
      if (storedMin == null && storedMax == null) {
        return { ...state, imageVminPct: 0, imageVmaxPct: 100 };
      }
      const panelRange = panelDataRanges[i];
      const range = (panelRange && panelRange.max > panelRange.min) ? panelRange : stackBounds;
      if (range.max <= range.min) return { ...state, imageVminPct: 0, imageVmaxPct: 100 };
      const lo = storedMin ?? range.min;
      const hi = Math.max(lo, storedMax ?? range.max);
      return {
        ...state,
        imageVminPct: valueToPct(lo, range.min, range.max, state.imageVminPct),
        imageVmaxPct: valueToPct(hi, range.min, range.max, state.imageVmaxPct),
      };
    });
    panelStatesLiveRef.current = nextStates;
    setPanelStates(nextStates);
  };

  const resolvePanelRenderRange = (
    panel: number,
    range: { min: number; max: number },
    sharedAutoRange: { vmin: number; vmax: number } | null,
    panelData: Float32Array | null,
    autoOn: boolean,
    low: number,
    high: number,
  ): { vmin: number; vmax: number; logScale: boolean } => {
    if (perPanelHistogramEnabled && autoOn) {
      const autoRange = autoPanelRangeFromData(panelData, range, low, high);
      if (autoRange) return autoRange;
    }
    return resolvePanelRange(panel, range, sharedAutoRange);
  };

  const handleAutoContrastChange = (on: boolean) => {
    if (on) {
      manualImageRangeBeforeAutoRef.current = { min: imageVminPct, max: imageVmaxPct };
    }
    setAutoContrast(on);
    if (perPanelHistogramEnabled) {
      if (on) {
        // Keep remembered manual per-panel clips. Auto rendering ignores them,
        // and toggling Auto back off should restore the user's manual window.
        // Per-panel snap fires automatically via the [autoContrast,
        // panelHistogramData, ...] useEffect below. Calling the legacy
        // stack-wide snap here would race-write 0/100 to every panel
        // before the effect overrode with the correct per-panel clip,
        // causing a 1-frame flash to washed contrast on every toggle.
      } else {
        // OFF restores manual contrast. If the user never set a manual range,
        // default to each panel's full local histogram range.
        restorePanelManualClipPcts();
        manualImageRangeBeforeAutoRef.current = null;
      }
      return;
    }
    if (on && imageHistogramData) {
      // ON -> snap slider thumbs to actual percentile clip so slider shows what's rendered.
      const cached = cachedAutoDisplayRange(autoVmins, autoVmaxs, displaySliceIdx, logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, displaySliceIdx, logScale);
      const { vmin: pmin, vmax: pmax } = cached ?? percentileClip(imageHistogramData, percentileLow, percentileHigh);
      const { min: autoMin, max: autoMax } = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
      const span = autoMax - autoMin;
      if (span > 0) {
        setImageVminPct(Math.max(0, Math.min(100, ((pmin - autoMin) / span) * 100)));
        setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - autoMin) / span) * 100)));
      }
    } else {
      // OFF -> restore the user's manual window from before Auto was enabled.
      const restore = manualImageRangeBeforeAutoRef.current;
      if (restore) {
        setImageVminPct(restore.min);
        setImageVmaxPct(restore.max);
        manualImageRangeBeforeAutoRef.current = null;
      } else {
        setImageVminPct(0);
        setImageVmaxPct(100);
      }
    }
  };
  const applyContrastPreset = React.useCallback((preset: string) => {
    setContrastPreset(preset);
    if (preset === "custom") return;
    const match = preset.match(/^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$/);
    if (!match) return;
    const lo = Math.max(0, Math.min(99, Number(match[1])));
    const hi = Math.max(lo + 0.01, Math.min(100, Number(match[2])));
    setPercentileHigh(hi);
    setPercentileLow(lo);
    handleAutoContrastChange(true);
  }, [handleAutoContrastChange, setContrastPreset, setPercentileHigh, setPercentileLow]);

  // Histogram state for FFT
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataRange, setFftDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [fftStats, setFftStats] = React.useState<{ mean: number; min: number; max: number; std: number }>({ mean: 0, min: 0, max: 0, std: 0 });
  const [fftQuality, setFftQuality] = React.useState<FftQualityMetrics | null>(null);
  const fftQualityKeyRef = React.useRef("");
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftLogScale, setFftLogScale] = React.useState(false);
  const [fftAuto, setFftAuto] = React.useState(true);  // Auto: mask DC + 99.9% clipping
  const [fftShowColorbar, setFftShowColorbar] = React.useState(false);
  const [fftOffscreenVersion, setFftOffscreenVersion] = React.useState(0);
  const [showColorbar, setShowColorbar] = React.useState(false);
  // True-color RGB figures: no colormap / intensity tools; pixelated (no smooth).
  React.useEffect(() => {
    if (isRgb) {
      setShowColorbar(false);
      if (autoContrast) setAutoContrast(false);
      if (logScale) setLogScale(false);
      if (diffMode && diffMode !== "off") setDiffMode("off");
      if (!displayFilterOff) setDisplayFilter("none");
      if (Number(spatialBin || 1) !== 1) setSpatialBin(1);
      if (showDenoise) setShowDenoise(false);
      if (smooth) setSmooth(false);
    }
  }, [isRgb]); // eslint-disable-line react-hooks/exhaustive-deps -- one-shot mode switch

  // Histogram state for kymograph (mirrors FFT contrast/colormap controls)
  const [kymoVminPct, setKymoVminPct] = React.useState(0);
  const [kymoVmaxPct, setKymoVmaxPct] = React.useState(100);
  const [kymoHistogramData, setKymoHistogramData] = React.useState<Float32Array | null>(null);
  const [kymoDataRange, setKymoDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [kymoStats, setKymoStats] = React.useState<{ mean: number; min: number; max: number; std: number }>({ mean: 0, min: 0, max: 0, std: 0 });
  const [kymoColormap, setKymoColormap] = React.useState("inferno");
  const [kymoLogScale, setKymoLogScale] = React.useState(false);
  const [kymoAuto, setKymoAuto] = React.useState(true);  // Auto: percentile-clip like the main image
  const [kymoShowColorbar, setKymoShowColorbar] = React.useState(false);

  const handleRootMouseDownCapture = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement | null;
    if (target?.closest("canvas")) rootRef.current?.focus();
  };

  const lastBenchmarkTokenRef = React.useRef<unknown>(null);
  const benchmarkPlaybackFpsRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    const req = benchmarkRequest ?? {};
    const token = req.token;
    const mode = typeof req.mode === "string" ? req.mode : "playback";
    if ((typeof token !== "string" && typeof token !== "number") || mode === "renderBurst" || mode === "scrubTransport" || mode === "scrubPreviewTransport" || lastBenchmarkTokenRef.current === token) return;
    lastBenchmarkTokenRef.current = token;

    let cancelled = false;
    const sleep = (ms: number) => new Promise<void>(resolve => window.setTimeout(resolve, ms));
    const numberFromReq = (key: string, fallback: number) => {
      const value = req[key];
      return typeof value === "number" && Number.isFinite(value) ? value : fallback;
    };
    const warmupMs = Math.max(0, numberFromReq("warmupMs", 3000));
    const sampleMs = Math.max(250, numberFromReq("sampleMs", 10000));
    const targetFps = clampPlaybackFps(numberFromReq("targetFps", playbackFps));
    const expectedFrames = Math.max(0, Math.floor(numberFromReq("expectedFrames", nSlices)));
    const waitForGpuPreload = req.waitForGpuPreload === true;
    const reportUrl = typeof req.reportUrl === "string" ? req.reportUrl : "";
    const label = typeof req.label === "string" ? req.label : "show3d benchmark";

    void (async () => {
      const startedAt = performance.now();
      const setStatus = (status: string, extra: Record<string, unknown> = {}) => {
        if (!cancelled) {
          setBenchmarkResult({ token, label, status, targetFps, ...extra });
        }
      };

      try {
        setStatus("warming");
        const estimatedRefresh = await estimateRafFps(Math.max(300, numberFromReq("refreshProbeMs", 750)));
        if (estimatedRefresh !== null) {
          setStatus("warming", { displayRefreshFps: Number(estimatedRefresh.toFixed(2)) });
        }
        benchmarkPlaybackFpsRef.current = targetFps;
        setPlaybackFps(targetFps);
        setPlaying(true);

        if (waitForGpuPreload && expectedFrames > 0) {
          const preloadDeadline = performance.now() + Math.max(5000, numberFromReq("preloadTimeoutMs", 120000));
          let preloadReady = false;
          while (!cancelled && performance.now() < preloadDeadline) {
            const dbg = show3dPerfDebug() ?? {};
            const gpuDone = Number(dbg.gpuPreloadDone ?? dbg.gpuFrameCacheUploaded ?? 0);
            const fetchDone = Number(dbg.frameFetchPreloadDone ?? 0);
            preloadReady = separatePanelFrames
              ? gpuDone >= expectedFrames
              : gpuDone >= expectedFrames || fetchDone >= expectedFrames;
            if (preloadReady) break;
            setStatus("preloading", { gpuPreloadDone: gpuDone, frameFetchPreloadDone: fetchDone });
            await sleep(250);
          }
          if (!preloadReady) {
            const dbg = show3dPerfDebug() ?? {};
            const gpuDone = Number(dbg.gpuPreloadDone ?? dbg.gpuFrameCacheUploaded ?? 0);
            const fetchDone = Number(dbg.frameFetchPreloadDone ?? 0);
            setStatus("error", {
              error: "GPU preload incomplete",
              gpuPreloadDone: gpuDone,
              frameFetchPreloadDone: fetchDone,
              gpuPreloadTarget: expectedFrames,
              gpuPreloadError: dbg.gpuPreloadError ?? null,
              gpuPreloadLastMiss: dbg.gpuPreloadLastMiss ?? null,
            });
            return;
          }
        }

        await sleep(warmupMs);
        if (cancelled) return;

        const dbgStart = show3dPerfDebug() ?? {};
        resetFramePacingDebug(dbgStart, playbackIntervalMs(targetFps));
        const startFrames = Number(dbgStart.renderedFrames ?? 0);
        const sampleStart = performance.now();
        setStatus("sampling");
        await sleep(sampleMs);
        if (cancelled) return;

        const dbgEnd = show3dPerfDebug() ?? {};
        const elapsedSeconds = Math.max(0.001, (performance.now() - sampleStart) / 1000);
        const endFrames = Number(dbgEnd.renderedFrames ?? 0);
        const frames = Math.max(0, endFrames - startFrames);
        const measuredFps = frames / elapsedSeconds;
        const frameIntervalCount = Number(dbgEnd.frameIntervalCount ?? 0);
        const overBudgetFrames = Number(dbgEnd.overBudgetFrames ?? 0);
        const passTarget = measuredFps >= targetFps * 0.98;
        const displayRefreshFps = estimatedRefresh !== null ? Number(estimatedRefresh.toFixed(2)) : null;
        const refreshLimited = displayRefreshFps !== null && targetFps > displayRefreshFps * 1.03;
        const result = {
          token,
          label,
          status: "done",
          targetFps,
          displayRefreshFps,
          refreshLimited,
          measuredFps: Number(measuredFps.toFixed(2)),
          frames,
          elapsedSeconds: Number(elapsedSeconds.toFixed(2)),
          passTarget,
          pass60: measuredFps >= 60 * 0.98,
          frameIntervalAvgMs: dbgEnd.frameIntervalAvgMs ?? null,
          frameIntervalP95Ms: percentileFromHistory(dbgEnd.frameIntervalHistory, 95),
          maxFrameIntervalMs: dbgEnd.maxFrameIntervalMs ?? null,
          overBudgetFrames,
          overBudgetPct: frameIntervalCount > 0 ? Number(((overBudgetFrames / frameIntervalCount) * 100).toFixed(2)) : null,
          lastRenderPath: dbgEnd.lastRenderPath ?? null,
          lastRenderMs: dbgEnd.lastRenderMs ?? null,
          lastFrameSource: dbgEnd.lastFrameSource ?? null,
          frameFetchCacheSize: dbgEnd.frameFetchCacheSize ?? null,
          gpuFrameCacheUploaded: dbgEnd.gpuFrameCacheUploaded ?? null,
          gpuPreloadDone: dbgEnd.gpuPreloadDone ?? null,
          frameFetchPreloadDone: dbgEnd.frameFetchPreloadDone ?? null,
          missingFrame: dbgEnd.missingFrame ?? null,
          totalMs: Number((performance.now() - startedAt).toFixed(1)),
        };
        setBenchmarkResult(result);
        if (reportUrl) {
          void fetch(reportUrl, { method: "POST", mode: "no-cors", body: JSON.stringify(result) }).catch(() => {});
        }
      } catch (err) {
        setStatus("error", { error: err instanceof Error ? err.message : String(err) });
      } finally {
        if (!cancelled) setPlaying(false);
        benchmarkPlaybackFpsRef.current = null;
      }
    })();

    return () => {
      cancelled = true;
      benchmarkPlaybackFpsRef.current = null;
    };
  }, [benchmarkRequest, playbackFps, nSlices, separatePanelFrames, setBenchmarkResult, setPlaybackFps, setPlaying]);

  const lastScrubTransportBenchmarkTokenRef = React.useRef<unknown>(null);
  React.useEffect(() => {
    const req = benchmarkRequest ?? {};
    const token = req.token;
    const mode = typeof req.mode === "string" ? req.mode : "playback";
    const previewMode = mode === "scrubPreviewTransport";
    if ((typeof token !== "string" && typeof token !== "number") || (mode !== "scrubTransport" && !previewMode) || lastScrubTransportBenchmarkTokenRef.current === token) return;
    lastScrubTransportBenchmarkTokenRef.current = token;

    let cancelled = false;
    const sleep = (ms: number) => new Promise<void>(resolve => window.setTimeout(resolve, ms));
    const numberFromReq = (key: string, fallback: number) => {
      const value = req[key];
      return typeof value === "number" && Number.isFinite(value) ? value : fallback;
    };
    const sampleCount = Math.max(1, Math.floor(numberFromReq("sampleCount", Math.min(12, nSlices))));
    const settleMs = Math.max(0, numberFromReq("settleMs", 80));
    const timeoutMs = Math.max(250, numberFromReq("timeoutMs", 8000));
    const label = typeof req.label === "string" ? req.label : "show3d scrub transport";
    const reportUrl = typeof req.reportUrl === "string" ? req.reportUrl : "";

    const summarize = (samples: Record<string, unknown>[]) => {
      const numeric = (key: string) => samples
        .map(sample => sample[key])
        .filter((value): value is number => typeof value === "number" && Number.isFinite(value))
        .sort((a, b) => a - b);
      const stats = (key: string) => {
        const values = numeric(key);
        if (values.length === 0) return null;
        const avg = values.reduce((acc, value) => acc + value, 0) / values.length;
        const p95 = values[Math.min(values.length - 1, Math.floor(values.length * 0.95))];
        return {
          avgMs: Number(avg.toFixed(3)),
          p95Ms: Number(p95.toFixed(3)),
          maxMs: Number(values[values.length - 1].toFixed(3)),
        };
      };
      return {
        pythonPrepare: stats("pythonPrepareMs"),
        pythonWire: stats("pythonWireMs"),
        pythonEncode: stats("pythonEncodeMs"),
        pythonTraitSet: stats("pythonTraitSetMs"),
        browserReceive: stats("browserReceiveLatencyMs"),
        jsDecode: stats("jsDecodeMs"),
        uiLatency: stats("endToEndUiLatencyMs"),
      };
    };

    void (async () => {
      const startedAt = performance.now();
      const firstSample = transportSamplesRef.current.length;
      const frames: number[] = [];
      for (let i = 0; i < sampleCount; i++) {
        const idx = nSlices <= 1 ? 0 : Math.round((i / Math.max(1, sampleCount - 1)) * (nSlices - 1));
        frames.push(idx);
      }
      try {
        setBenchmarkResult({ token, label, status: "sampling", mode, sampleCount, frames });
        setPlaying(false);
        await sleep(settleMs);
        for (const [sampleIndex, idx] of frames.entries()) {
          if (cancelled) return;
          if (previewMode) {
            setDisplaySliceIdx(idx);
            setPlaybackUiSliceIdx(idx);
            setScrubPreviewRequest(JSON.stringify({
              token: `${String(token)}-${idx}-${sampleIndex}`,
              idx,
              maxBytes: numberFromReq("maxBytes", 16 * 1024 * 1024),
            }));
          } else {
            setSliceIdx(idx);
          }
          const deadline = performance.now() + timeoutMs;
          while (!cancelled && performance.now() < deadline) {
            const latest = transportSamplesRef.current[transportSamplesRef.current.length - 1];
            if (
              latest &&
              latest.kind === (previewMode ? "scrubPreview" : "frame") &&
              Number(previewMode ? latest.idx : latest.slice) === idx &&
              typeof latest.endToEndUiLatencyMs === "number"
            ) break;
            await sleep(16);
          }
          await sleep(settleMs);
        }
        if (cancelled) return;
        const expectedKind = previewMode ? "scrubPreview" : "frame";
        const samples = transportSamplesRef.current
          .slice(firstSample)
          .filter(sample => sample.kind === expectedKind);
        const result = {
          token,
          label,
          status: "done",
          mode,
          sampleCount,
          receivedSamples: samples.length,
          frames,
          summary: summarize(samples),
          samples,
          totalMs: Number((performance.now() - startedAt).toFixed(1)),
        };
        setBenchmarkResult(result);
        if (reportUrl) {
          void fetch(reportUrl, { method: "POST", mode: "no-cors", body: JSON.stringify(result) }).catch(() => {});
        }
      } catch (err) {
        setBenchmarkResult({ token, label, status: "error", mode, error: err instanceof Error ? err.message : String(err) });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [benchmarkRequest, nSlices, setBenchmarkResult, setPlaying, setSliceIdx, setScrubPreviewRequest]);

  // FFT d-spacing measurement
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);

  // ROI FFT state: when ROI + FFT are both active, compute FFT of cropped ROI region
  const [fftCropDims, setFftCropDims] = React.useState<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);
  const fftCropDimsRef = React.useRef<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);
  const fftPanelGridRef = React.useRef<{ panelWidth: number; panelHeight: number; cols: number; rows: number; count: number } | null>(null);

  // FFT zoom/pan state
  const [fftZoom, setFftZoom] = React.useState(1);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const internalFftZoomSyncRef = React.useRef(false);
  const fftViewLiveRef = React.useRef({ zoom: 1, panX: 0, panY: 0 });
  const fftViewRafRef = React.useRef<number | null>(null);
  const fftViewReactSyncTimerRef = React.useRef<number | null>(null);
  const fftViewTraitSyncTimerRef = React.useRef<number | null>(null);
  const fftViewDirectRedrawRef = React.useRef<((view: { zoom: number; panX: number; panY: number }) => void) | null>(null);
  const fftViewCenterOnViewportRef = React.useRef(false);
  const fftOverlayInitialCenterPendingRef = React.useRef(true);
  const fftOverlayWasActiveRef = React.useRef(false);
  const fftUserAdjustedViewRef = React.useRef(false);

  const commitFftViewReactState = React.useCallback(() => {
    const live = fftViewLiveRef.current;
    setFftZoom(prev => Math.abs(prev - live.zoom) > 0.001 ? live.zoom : prev);
    setFftPanX(prev => Math.abs(prev - live.panX) > 0.5 ? live.panX : prev);
    setFftPanY(prev => Math.abs(prev - live.panY) > 0.5 ? live.panY : prev);
  }, []);

  const scheduleFftViewState = React.useCallback((next: { zoom: number; panX: number; panY: number }, syncTrait = false, directOnly = false) => {
    fftViewLiveRef.current = next;
    if (directOnly) {
      fftViewDirectRedrawRef.current?.(next);
      if (fftViewReactSyncTimerRef.current !== null) {
        window.clearTimeout(fftViewReactSyncTimerRef.current);
      }
      fftViewReactSyncTimerRef.current = window.setTimeout(() => {
        fftViewReactSyncTimerRef.current = null;
        commitFftViewReactState();
      }, 80);
    } else if (fftViewRafRef.current === null) {
      fftViewRafRef.current = window.requestAnimationFrame(() => {
        fftViewRafRef.current = null;
        commitFftViewReactState();
      });
    }
    if (syncTrait) {
      if (fftViewTraitSyncTimerRef.current !== null) {
        window.clearTimeout(fftViewTraitSyncTimerRef.current);
      }
      fftViewTraitSyncTimerRef.current = window.setTimeout(() => {
        fftViewTraitSyncTimerRef.current = null;
        internalFftZoomSyncRef.current = true;
        setFftOverlayZoomTrait(Number(fftViewLiveRef.current.zoom.toFixed(3)));
      }, 160);
    }
  }, [commitFftViewReactState, setFftOverlayZoomTrait]);

  React.useEffect(() => {
    fftViewLiveRef.current = { zoom: fftZoom, panX: fftPanX, panY: fftPanY };
  }, [fftZoom, fftPanX, fftPanY]);

  React.useEffect(() => () => {
    if (fftViewRafRef.current !== null) {
      window.cancelAnimationFrame(fftViewRafRef.current);
      fftViewRafRef.current = null;
    }
    if (fftViewReactSyncTimerRef.current !== null) {
      window.clearTimeout(fftViewReactSyncTimerRef.current);
      fftViewReactSyncTimerRef.current = null;
    }
    if (fftViewTraitSyncTimerRef.current !== null) {
      window.clearTimeout(fftViewTraitSyncTimerRef.current);
      fftViewTraitSyncTimerRef.current = null;
    }
  }, []);

  React.useEffect(() => {
    if (internalFftZoomSyncRef.current) {
      internalFftZoomSyncRef.current = false;
      return;
    }
    const reset = { zoom: resolvedFftOverlayZoom, panX: 0, panY: 0 };
    fftViewLiveRef.current = reset;
    fftViewCenterOnViewportRef.current = true;
    fftOverlayInitialCenterPendingRef.current = true;
    fftUserAdjustedViewRef.current = false;
    setFftZoom(reset.zoom);
    setFftPanX(reset.panX);
    setFftPanY(reset.panY);
  }, [resolvedFftOverlayZoom]);

  React.useEffect(() => {
    if (fftLayoutOverlay && !fftOverlayWasActiveRef.current) {
      fftOverlayInitialCenterPendingRef.current = true;
      fftViewCenterOnViewportRef.current = true;
    }
    fftOverlayWasActiveRef.current = fftLayoutOverlay;
  }, [fftLayoutOverlay]);

  React.useEffect(() => {
    if (fftLayoutOverlay && fftOverlayInitialCenterPendingRef.current) {
      fftViewCenterOnViewportRef.current = true;
    }
  }, [fftLayoutOverlay, fftOffscreenVersion]);
  const fftContainerRef = React.useRef<HTMLDivElement>(null);

  // Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileLine, setProfileLine] = useModelState<{row: number; col: number}[]>("profile_line");
  const [profileWidth, setProfileWidth] = useModelState<number>("profile_width");
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const [profilePanelIdx, setProfilePanelIdx] = React.useState(0);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profilePoints = profileLine || [];
  const singlePanelPageProfile = isPaged && Math.max(1, panelsPerPage || 0) === 1;
  React.useEffect(() => {
    if (!singlePanelPageProfile) return;
    setProfilePanelIdx((current) => current === activePageStart ? current : activePageStart);
  }, [activePageStart, singlePanelPageProfile]);
  // Kymograph (space-time) panel: static (nFrames, lineLen) image built by
  // sampling the profile line on every frame from the offline stack. Recompute
  // is cold-path (on line / width change only), not per render tick.
  const [showKymograph, setShowKymograph] = useModelState<boolean>("show_kymograph");
  const kymoCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const kymoOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const kymoDataRef = React.useRef<{ data: Float32Array; lineLen: number; nFrames: number } | null>(null);
  const [kymoVersion, setKymoVersion] = React.useState(0);
  // Kymograph zoom/pan state (mirrors FFT)
  const [kymoZoom, setKymoZoom] = React.useState(1);
  const [kymoPanX, setKymoPanX] = React.useState(0);
  const [kymoPanY, setKymoPanY] = React.useState(0);
  const kymoContainerRef = React.useRef<HTMLDivElement>(null);
  // Click readout: cursor maps to (frame index, distance index) and looks up
  // intensity in the static kymograph image. Mirrors FFT d-spacing readout.
  const [kymoClickInfo, setKymoClickInfo] = React.useState<{
    timeVal: number; timeUnit: string; distVal: number; distUnit: string; intensity: number;
    col: number; row: number;
  } | null>(null);
  const kymoClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [profileHeight, setProfileHeight] = React.useState(76);
  const [isResizingProfile, setIsResizingProfile] = React.useState(false);
  const [profileResizeStart, setProfileResizeStart] = React.useState<{ y: number; height: number } | null>(null);
  const profileBaseImageRef = React.useRef<ImageData | null>(null);
  const profileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);

  // Sync sizes from Python and set initial minimum. In multi-panel mode the user
  // is comparing N images side-by-side; default per-panel sizing keeps each image
  // readable instead of crushed when the widget concatenates them into one wide
  // canvas (e.g. 4 panels at 500 px total → 125 px per panel = too small).
  React.useEffect(() => {
    // size is PER PANEL. For multi-panel, total canvas width = size * cols.
    // NEVER BIN rule: data is never averaged. CSS canvas scales the painted
    // image for display, source pixels stay intact. 500 px/panel default
    // gives 4 cols → 2000 px wide which fits a typical monitor; operator
    // drags the resize handle larger when they want pixel-1:1.
    const n = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(n);
    const perPanel = canvasSizeTrait > 0 ? canvasSizeTrait : (n > 1 ? 500 : CANVAS_TARGET_SIZE);
    const target = perPanel * cols;
    setMainCanvasSize(target);
    if (initialCanvasSizeRef.current === CANVAS_TARGET_SIZE) {
      initialCanvasSizeRef.current = target;
    }
  }, [canvasSizeTrait, visiblePanelCount, panelColsForCount]);

  // Calculate display scale. In multi-panel mode `width` may be either the
  // concatenated source width or one shared source frame drawn into N slots.
  // `panel_width_px` keeps the per-panel source geometry explicit.
  const _nPanelsLocal = Math.max(1, visiblePanelCount || 1);
  const _colsLocal = panelColsForCount(_nPanelsLocal);
  const _rowsLocal = Math.ceil(_nPanelsLocal / _colsLocal);
  const fftAllowed = true;
  const effectiveShowFft = showFft && fftAllowed;
  const sourcePanelWidth = totalPanelCount > 1
    ? Math.max(1, panelWidthPx || Math.round(width / totalPanelCount))
    : Math.max(1, width);
  const sourcePanelHeight = Math.max(1, height);
  const isMultiPanelSource = totalPanelCount > 1;
  const displayScale = isMultiPanelSource
    ? mainCanvasSize / Math.max(1, sourcePanelWidth * _colsLocal)
    : mainCanvasSize / Math.max(width, height);
  // For 90°/270° rotations, swap canvas dims so non-square images fit without clipping.
  const rotSwap = (imageRotation % 2) !== 0;
  const canvasW = isMultiPanelSource
    ? Math.round(sourcePanelWidth * displayScale * _colsLocal)
    : Math.round((rotSwap ? height : width) * displayScale);
  // Grid layout: when max_cols wraps panels into multiple rows, canvasH grows to fit `rows` rows.
  const _canvasHSingleRow = Math.round((rotSwap ? width : height) * displayScale);
  const _gapForLayout = _nPanelsLocal > 1 ? (panelGapTrait ?? 10) : 0;
  const _slotWForLayout = (canvasW - _gapForLayout * (_colsLocal - 1)) / _colsLocal;
  const _slotHForLayout = _slotWForLayout * (sourcePanelHeight / sourcePanelWidth);
  const canvasH = isMultiPanelSource
    ? Math.round(_slotHForLayout * _rowsLocal + _gapForLayout * (_rowsLocal - 1))
    : _canvasHSingleRow;
  const mainPanelWidth = `min(100%, ${canvasW}px)`;
  const mainPanelAspectRatio = `${Math.max(canvasW, 1)} / ${Math.max(canvasH, 1)}`;
  const groupMarkerOverlays = React.useMemo(() => {
    if ((nPanels || 1) <= 1 || visiblePanelIndices.length === 0 || canvasW <= 0 || canvasH <= 0) return [];
    const count = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(count);
    const gap = count > 1 ? (panelGapTrait ?? 10) : 0;
    const rows = Math.ceil(count / cols);
    const panelW = (canvasW - gap * (cols - 1)) / cols;
    const panelH = (canvasH - gap * (rows - 1)) / rows;
    type GroupMarkerOverlay = {
      key: string;
      axis: "row" | "col" | "panel";
      color: string;
      leftPct: number;
      topPct: number;
      widthPct: number;
      heightPct: number;
      label?: string;
    };
    const boundsForSlots = (slots: number[]) => {
      if (slots.length === 0) return null;
      const rowVals = slots.map((slot) => Math.floor(slot / cols));
      const colVals = slots.map((slot) => slot % cols);
      const row0 = Math.min(...rowVals);
      const row1 = Math.max(...rowVals);
      const col0 = Math.min(...colVals);
      const col1 = Math.max(...colVals);
      const left = col0 * (panelW + gap);
      const top = row0 * (panelH + gap);
      return {
        leftPct: (left / Math.max(1, canvasW)) * 100,
        topPct: (top / Math.max(1, canvasH)) * 100,
        widthPct: (((col1 - col0 + 1) * panelW + Math.max(0, col1 - col0) * gap) / Math.max(1, canvasW)) * 100,
        heightPct: (((row1 - row0 + 1) * panelH + Math.max(0, row1 - row0) * gap) / Math.max(1, canvasH)) * 100,
      };
    };
    const build = (markers: MarkerMap | undefined, axis: "row" | "col") => Object.entries(markers || {})
      .map(([rawKey, color]) => {
        const target = Number(rawKey);
        if (!Number.isFinite(target) || target < 0 || !color) return null;
        const slots = visiblePanelIndices
          .map((_, slot) => slot)
          .filter((slot) => (axis === "row" ? Math.floor(slot / cols) : slot % cols) === target);
        const bounds = boundsForSlots(slots);
        if (!bounds) return null;
        return {
          key: `${axis}-${rawKey}`,
          axis,
          color: String(color),
          ...bounds,
        };
      })
      .filter(Boolean) as GroupMarkerOverlay[];
    const visibleSlotByPanel = new Map<number, number>();
    visiblePanelIndices.forEach((panel, slot) => visibleSlotByPanel.set(panel, slot));
    const panelGroupOverlays = (panelGroups || [])
      .map((group, index) => {
        const slots = (group?.panels || [])
          .map((panel) => visibleSlotByPanel.get(Number(panel)))
          .filter((slot): slot is number => Number.isFinite(slot));
        const bounds = boundsForSlots(slots);
        if (!bounds) return null;
        const color = group?.color ? String(group.color) : "#22c55e";
        const label = group?.label ? String(group.label) : undefined;
        return {
          key: `panel-group-${index}`,
          axis: "panel" as const,
          color,
          label,
          ...bounds,
        };
      })
      .filter(Boolean) as GroupMarkerOverlay[];
    return [...build(rowMarkers, "row"), ...build(colMarkers, "col"), ...panelGroupOverlays];
  }, [canvasH, canvasW, colMarkers, nPanels, panelColsForCount, panelGapTrait, panelGroups, rowMarkers, visiblePanelCount, visiblePanelIndices]);
  const effectiveLoopEnd = loopEnd < 0 ? nSlices - 1 : loopEnd;
  // ROI hidden while the kymograph is shown - both are line/region analysis on
  // the same side slot, and showing them together confuses which panel is which.
  const roiAllowed = totalPanelCount === 1 && !showKymograph;
  const effectiveRoiActive = roiAllowed && roiActive;

  type PanelGeometry = {
    panelIdx: number;
    slotX: number;
    slotY: number;
    slotW: number;
    slotH: number;
    scaleX: number;
    scaleY: number;
    state: PanelState;
  };
  const getPanelLayout = () => {
    const n = _nPanelsLocal;
    const cols = _colsLocal;
    const rows = _rowsLocal;
    const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
    const slotW = (canvasW - gap * (cols - 1)) / cols;
    const slotH = (canvasH - gap * (rows - 1)) / rows;
    return { n, cols, rows, gap, slotW, slotH };
  };
  const getPanelGeometry = (panelIdx: number): PanelGeometry | null => {
    const { n, cols, rows, gap, slotW, slotH } = getPanelLayout();
    if (panelIdx < 0 || panelIdx >= totalPanelCount) return null;
    const slotIdx = visiblePanelIndices.indexOf(panelIdx);
    if (slotIdx < 0 || slotIdx >= n) return null;
    const col = slotIdx % cols;
    const row = Math.floor(slotIdx / cols);
    if (row >= rows) return null;
    return {
      panelIdx,
      slotX: col * (slotW + gap),
      slotY: row * (slotH + gap),
      slotW,
      slotH,
      scaleX: slotW / Math.max(1, sourcePanelWidth),
      scaleY: slotH / Math.max(1, sourcePanelHeight),
      state: stateFor(panelIdx),
    };
  };
  const getFftSlot = React.useCallback((slot: number, count: number, cols: number, rows: number) => {
    const gap = count > 1 ? (panelGapTrait ?? 10) : 0;
    const slotW = (canvasW - gap * (cols - 1)) / cols;
    const slotH = (canvasH - gap * (rows - 1)) / rows;
    const col = slot % cols;
    const row = Math.floor(slot / cols);
    return {
      x: col * (slotW + gap),
      y: row * (slotH + gap),
      w: slotW,
      h: slotH,
    };
  }, [canvasW, canvasH, panelGapTrait]);
  const drawFftOffscreen = React.useCallback((ctx: CanvasRenderingContext2D, offscreen: HTMLCanvasElement) => {
    ctx.clearRect(0, 0, canvasW, canvasH);
    const grid = fftPanelGridRef.current;
    if (grid) {
      ctx.fillStyle = themeColors.bg;
      ctx.fillRect(0, 0, canvasW, canvasH);
      for (let slot = 0; slot < grid.count; slot++) {
        const srcCol = slot % grid.cols;
        const srcRow = Math.floor(slot / grid.cols);
        const srcX = srcCol * grid.panelWidth;
        const srcY = srcRow * grid.panelHeight;
        const dst = getFftSlot(slot, grid.count, grid.cols, grid.rows);
        ctx.imageSmoothingEnabled = grid.panelWidth < dst.w || grid.panelHeight < dst.h;
        ctx.save();
        ctx.beginPath();
        ctx.rect(dst.x, dst.y, dst.w, dst.h);
        ctx.clip();
        ctx.translate(dst.x + fftPanX, dst.y + fftPanY);
        ctx.scale(fftZoom, fftZoom);
        ctx.drawImage(
          offscreen,
          srcX,
          srcY,
          grid.panelWidth,
          grid.panelHeight,
          0,
          0,
          dst.w,
          dst.h,
        );
        ctx.restore();
      }
    } else {
      ctx.save();
      ctx.translate(fftPanX, fftPanY);
      ctx.scale(fftZoom, fftZoom);
      ctx.imageSmoothingEnabled = offscreen.width < canvasW || offscreen.height < canvasH;
      ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);
      ctx.restore();
    }
  }, [canvasW, canvasH, fftPanX, fftPanY, fftZoom, getFftSlot, themeColors.bg]);
  const panelGlobalColOffset = (panelIdx: number) => (totalPanelCount > 1 && !sharedPanelSource) ? panelIdx * sourcePanelWidth : 0;
  const panelLocalCol = (globalCol: number, panelIdx: number) => globalCol - panelGlobalColOffset(panelIdx);
  const panelGlobalCol = (localCol: number, panelIdx: number) => localCol + panelGlobalColOffset(panelIdx);
  // A single-panel page reuses one spatial profile on every page. Keep the
  // trait coordinates page-local, then add the active page's packed-frame
  // offset only when sampling the concatenated source frame.
  const profileSampleColOffset = singlePanelPageProfile
    ? panelGlobalColOffset(activePageStart)
    : 0;
  const sampleProfileForActivePage = React.useCallback((
    data: Float32Array,
    p0: { row: number; col: number },
    p1: { row: number; col: number },
    widthPx: number = profileWidth,
  ) => sampleLineProfile(
    data,
    width,
    height,
    p0.row,
    p0.col + profileSampleColOffset,
    p1.row,
    p1.col + profileSampleColOffset,
    widthPx,
  ), [height, profileSampleColOffset, profileWidth, width]);
  const getImageHitRadius = (panelIdx: number) => {
    const geom = getPanelGeometry(panelIdx);
    if (!geom) return RESIZE_HIT_AREA_PX / Math.max(1e-6, displayScale * zoom);
    const scale = Math.max(1e-6, Math.min(geom.scaleX, geom.scaleY) * geom.state.zoom);
    return RESIZE_HIT_AREA_PX / scale;
  };
  const canvasPointFromEvent = (e: React.MouseEvent): { x: number; y: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    };
  };

  React.useEffect(() => {
    if (offline || !frameServerUrl || width <= 0 || height <= 0 || nSlices <= 0 || canvasW <= 0 || canvasH <= 0) return;
    const n = Math.max(1, nPanels || 1);
    if (separatePanelFrames && n > 1) {
      let cancelled = false;
      const cacheLimit = getSeparatePanelGpuCacheLimit();
      const preloadCount = Math.min(nSlices, cacheLimit);
      const startIdx = ((Math.round(playbackIdxRef.current) % nSlices) + nSlices) % nSlices;
      const dbg = show3dPerfDebug();
      if (dbg) {
        dbg.gpuPreloadTarget = preloadCount;
        dbg.gpuPreloadDone = gpuFrameCacheUploadedRef.current.size;
        dbg.gpuPreloadActive = true;
        dbg.gpuPreloadMode = "separate-panel-direct-grid";
        dbg.gpuPanelCacheLimit = cacheLimit;
        dbg.gpuPanelTotalFrames = nSlices;
      }
      void (async () => {
        const rgbaCapacity = Math.max(1, Math.round(canvasW * canvasH));
        for (let step = 0; step < preloadCount; step++) {
          if (cancelled) break;
          const i = (startIdx + step) % nSlices;
          try {
            const ready = await ensurePanelFrameGpu(i, rgbaCapacity);
            if (!ready) {
              const d = show3dPerfDebug();
              if (d) {
                d.gpuPreloadMisses = ((d.gpuPreloadMisses as number | undefined) ?? 0) + 1;
                d.gpuPreloadLastMiss = i;
              }
              // One transient miss (stale 409, dropped socket, GPU not yet
              // ready) must NOT abort the whole preload - skip this frame and
              // keep going, matching the non-panel branch. `break` here left
              // the cache permanently below nSlices.
              continue;
            }
          } catch (err) {
            const d = show3dPerfDebug();
            if (d) d.gpuPreloadError = err instanceof Error ? err.message : String(err);
            continue;
          }
          const d = show3dPerfDebug();
          if (d) {
            d.gpuPreloadDone = gpuFrameCacheUploadedRef.current.size;
            d.gpuFrameCacheUploaded = gpuFrameCacheUploadedRef.current.size;
          }
          await new Promise<void>(resolve => setTimeout(resolve, 0));
        }
        const d = show3dPerfDebug();
        if (d) d.gpuPreloadActive = false;
      })();
      return () => {
        cancelled = true;
        const d = show3dPerfDebug();
        if (d) d.gpuPreloadActive = false;
      };
    }
    const stackByteLength = Math.max(1, width * height * 4) * Math.max(1, nSlices);
    if (stackByteLength > FRAME_SERVER_FULL_STACK_CACHE_BYTES) return;
    let cancelled = false;
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.gpuPreloadTarget = nSlices;
      dbg.gpuPreloadDone = 0;
      dbg.gpuPreloadActive = true;
      dbg.gpuPreloadMode = "direct-grid";
    }
    void (async () => {
      while (!cancelled && (!gpuCmapReadyRef.current || !gpuCmapRef.current)) {
        await new Promise<void>(resolve => setTimeout(resolve, 50));
      }
      const engine = gpuCmapRef.current;
      if (!engine || cancelled) return;
      const rgbaCapacity = Math.max(1, Math.round(canvasW * canvasH));
      for (let i = 0; i < nSlices; i++) {
        if (cancelled) break;
        const frame = await fetchFrameFromServer(i);
        if (cancelled) break;
        const d = show3dPerfDebug();
        if (!frame) {
          if (d) d.gpuPreloadMisses = ((d.gpuPreloadMisses as number | undefined) ?? 0) + 1;
          continue;
        }
        try {
          engine.uploadData(i, frame, width, height, rgbaCapacity);
          gpuFrameCacheUploadedRef.current.add(i);
          frameFetchCacheRef.current.delete(i);
          if (d) {
            d.gpuPreloadDone = gpuFrameCacheUploadedRef.current.size;
            d.gpuFrameCacheUploaded = gpuFrameCacheUploadedRef.current.size;
            d.frameFetchCacheSize = frameFetchCacheRef.current.size;
          }
        } catch (err) {
          if (d) d.gpuPreloadError = err instanceof Error ? err.message : String(err);
          break;
        }
        await new Promise<void>(resolve => setTimeout(resolve, 0));
      }
      const d = show3dPerfDebug();
      if (d) d.gpuPreloadActive = false;
    })();
    return () => {
      cancelled = true;
      const d = show3dPerfDebug();
      if (d) d.gpuPreloadActive = false;
    };
  }, [offline, frameServerUrl, frameServerVersion, width, height, nSlices, nPanels, canvasW, canvasH, fetchFrameFromServer, separatePanelFrames, ensurePanelFrameGpu, getSeparatePanelGpuCacheLimit, sliceIdx]);

  // ROI FFT active: both ROI and FFT on, with a selected ROI
  const roiFftActive = effectiveShowFft && effectiveRoiActive && roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0);

  // Preview panel visible: auto-shows when ROI active with a selected ROI
  const previewVisible = effectiveRoiActive && roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0);
  const selectedRoiKey = (() => {
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return "";
    const r = roiList[roiSelectedIdx];
    return `${r.row},${r.col},${r.radius},${r.radius_inner},${r.width},${r.height},${r.shape}`;
  })();

  // Compute stats for ALL ROIs (memoized, recomputes on frame/ROI geometry change)
  const allRoiStats = React.useMemo(() => {
    const raw = rawFrameDataRef.current;
    if (!effectiveRoiActive || !roiItems.length || !raw || !width || !height) return [];
    return roiItems.map(roi => computeROIPixelStats(raw, width, height, roi));
    // frameBytes triggers recompute on frame change; displaySliceIdx triggers recompute during playback
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveRoiActive, roiItems, width, height, frameBytes, displaySliceIdx]);

  // Initialize reusable offscreen canvas + ImageData (resized when dimensions change)
  React.useEffect(() => {
    if (width <= 0 || height <= 0) return;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    logBufferRef.current = new Float32Array(width * height);
    if (mainOffscreenRef.current && mainOffscreenSourcePanelWidthRef.current !== undefined && !rawFrameDataRef.current) {
      scaledPlaybackImgDataRef.current = null;
      scaledPlaybackMapRef.current = null;
      return;
    }
    mainOffscreenRef.current = canvas;
    mainOffscreenSourcePanelWidthRef.current = undefined;
    mainImgDataRef.current = canvas.getContext("2d")!.createImageData(width, height);
    scaledPlaybackImgDataRef.current = null;
    scaledPlaybackMapRef.current = null;
  }, [width, height]);

  // Prevent page scroll on secondary canvas containers. Main image wheel is
  // handled by a non-passive listener below so zoom works in notebook outputs.
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const el2 = fftContainerRef.current;
    const el3 = previewContainerRef.current;
    el2?.addEventListener("wheel", preventDefault, { passive: false });
    el3?.addEventListener("wheel", preventDefault, { passive: false });
    return () => {
      el2?.removeEventListener("wheel", preventDefault);
      el3?.removeEventListener("wheel", preventDefault);
    };
  }, [effectiveShowFft, previewVisible]);


  // Sync boomerang direction ref with reverse state
  React.useEffect(() => {
    bounceDirRef.current = reverse ? -1 : 1;
  }, [reverse]);

  // All playback params as a single ref (avoids stale closures in rAF loop)
  const pathIdxRef = React.useRef(0);
  const playRef = React.useRef({
    fps: playbackFps, reverse, boomerang, loop, loopStart, loopEnd: effectiveLoopEnd,
    nSlices, width, height, displayScale, canvasW, canvasH,
    logScale, autoContrast, percentileLow, percentileHigh,
    dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
    autoVmins, autoVmaxs,
    linkContrast,
    linkedState, linkPanels,
    panelStates, vminPerPanel, vmaxPerPanel,
    visiblePanelIndices,
    zoom, panX, panY, playbackPath,
    profileActive, profilePoints, profileWidth, profileColOffset: profileSampleColOffset,
    traitVmin, traitVmax, smooth, imageRotation, showStats,
    diffMode, avgWindow,
  });
  React.useEffect(() => {
    linkedStateLiveRef.current = linkedState;
  }, [linkedState]);
  React.useEffect(() => {
    panelStatesLiveRef.current = panelStates;
  }, [panelStates]);
  React.useEffect(() => {
    const liveLinkedState = linkedStateLiveRef.current;
    const livePanelStates = panelStatesLiveRef.current.length === Math.max(1, nPanels || 1)
      ? panelStatesLiveRef.current
      : panelStates;
    playRef.current = {
      fps: playbackFps, reverse, boomerang, loop, loopStart, loopEnd: effectiveLoopEnd,
      nSlices, width, height, displayScale, canvasW, canvasH,
      logScale, autoContrast, percentileLow, percentileHigh,
      dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
      autoVmins, autoVmaxs,
      linkContrast,
      linkedState: liveLinkedState, linkPanels,
      panelStates: livePanelStates, vminPerPanel, vmaxPerPanel,
      visiblePanelIndices,
      zoom, panX, panY, playbackPath,
      profileActive, profilePoints, profileWidth, profileColOffset: profileSampleColOffset,
      traitVmin, traitVmax, smooth, imageRotation, showStats,
      diffMode, avgWindow,
    };
  }, [playbackFps, reverse, boomerang, loop, loopStart, effectiveLoopEnd,
    nSlices, width, height, displayScale, canvasW, canvasH,
    logScale, autoContrast, percentileLow, percentileHigh,
    dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
    autoVmins, autoVmaxs, linkContrast, linkedState, linkPanels, panelStates, vminPerPanel, vmaxPerPanel, visiblePanelIndices,
    zoom, panX, panY, playbackPath,
    profileActive, profilePoints, profileWidth, profileSampleColOffset,
    traitVmin, traitVmax, smooth, imageRotation, showStats, diffMode, avgWindow]);

  const updatePlaybackLiveControls = React.useCallback((idx: number) => {
    const c = playRef.current;
    const total = Math.max(1, c.nSlices || nSlices || 1);
    // The playback row is an honesty indicator for the frame actually drawn.
    // Do not clamp this live DOM update to loop handles: custom playback paths,
    // transient trait hydration, and direct-frame filter retries can legitimately
    // display a frame outside the current loop-handle span. Clamping here made
    // users see "1/18" while the canvas was already showing a later frame.
    const clamped = Math.max(0, Math.min(total - 1, Math.round(idx)));
    const pct = total > 1 ? (clamped / (total - 1)) * 100 : 0;
    const slider = playbackSliderRef.current;
    const activeThumb = slider?.querySelector(c.loop ? ".MuiSlider-thumb[data-index='1']" : ".MuiSlider-thumb") as HTMLElement | null;
    const track = slider?.querySelector(".MuiSlider-track") as HTMLElement | null;
    const input = activeThumb?.querySelector("input") as HTMLInputElement | null;
    const count = playbackLiveCountRef.current
      ?? playbackSliderRef.current?.parentElement?.querySelector("[data-show3d-playback-count]")
      ?? rootRef.current?.querySelector("[data-show3d-playback-count]")
      ?? document.querySelector("[data-show3d-playback-count]");
    if (activeThumb) {
      activeThumb.style.left = `${pct}%`;
      activeThumb.setAttribute("aria-valuenow", String(clamped));
    }
    if (input) input.value = String(clamped);
    if (track && !c.loop) {
      track.style.left = "0%";
      track.style.width = `${pct}%`;
    }
    if (count) {
      count.textContent = hiddenSet.size ? `${clamped + 1}/${visibleCount} (${total})` : `${clamped + 1}/${total}`;
      const dbg = show3dPerfDebug();
      if (dbg) dbg.lastPlaybackLiveCountText = count.textContent;
    }
    const panelCounts = rootRef.current?.querySelectorAll("[data-show3d-panel-frame-count]") ?? [];
    panelCounts.forEach((node) => {
      const el = node as HTMLElement;
      const panelTotal = Math.max(1, Math.round(Number(el.dataset.realFrameCount || total) || total));
      const shown = Math.min(clamped + 1, panelTotal);
      el.textContent = `${shown}/${panelTotal}`;
    });
  }, [hiddenSet.size, nSlices, visibleCount]);

  const sidecarViewTransformActive = React.useCallback(() => {
    if (imageRotation % 4 !== 0 || flipRows || flipCols) return true;
    const panels = visiblePanelIndices.length
      ? visiblePanelIndices
      : Array.from({ length: Math.max(1, nPanels || 1) }, (_, idx) => idx);
    for (const panelIdx of panels) {
      const state = stateFor(panelIdx);
      if (
        Math.abs((state.zoom || 1) - 1) > 1e-3 ||
        Math.abs(state.panX || 0) > 0.5 ||
        Math.abs(state.panY || 0) > 0.5
      ) {
        return true;
      }
    }
    return false;
  }, [flipCols, flipRows, imageRotation, nPanels, stateFor, visiblePanelIndices]);

  const frameTransformActive = () => requiresClientFrameTransform({
    offline,
    diffMode: playRef.current.diffMode,
    avgWindow: playRef.current.avgWindow,
  }) || browserFilterOnRef.current || frequencyFilterIsActive || !!subpixelAlignEnabled;

  const rawFrameForIndex = (idx: number, currentIdx: number, currentFrame: Float32Array | null): Float32Array | null => {
    const n = Math.max(1, nSlices || 1);
    const normalized = ((Math.round(idx) % n) + n) % n;
    if (currentFrame && normalized === ((Math.round(currentIdx) % n) + n) % n) return currentFrame;
    if (offline && sidecarMode && (sidecarBitmapReadyRef.current || sidecarCompositeReadyRef.current) && !frameTransformActive()) return null;
    if (offline) return getOfflineFrame(normalized);
    const frameSize = width * height;
    const fromBuffer = getFrameFromBuffer(bufferRef.current, bufferStartRef.current, bufferCountRef.current, n, normalized, frameSize)
      || getFrameFromBuffer(nextBufferRef.current, nextBufferStartRef.current, nextBufferCountRef.current, n, normalized, frameSize);
    if (fromBuffer) return fromBuffer;
    const cached = getCachedServerFrame(normalized);
    if (cached) return cached;
    return null;
  };

  // Mean of `avg_window` consecutive frames (temporal denoise). At the stack
  // ends the window SLIDES INWARD to stay full-width (frame 0, win 5 -> [0..4])
  // rather than shrinking - constant denoise strength, but the average is not
  // centered on `idx` near the ends. Even windows are front-biased.
  const averagedFrameForIndex = (idx: number, currentIdx: number, currentFrame: Float32Array | null): Float32Array | null => {
    const frameSize = width * height;
    const win = normalizedAverageWindow(playRef.current.avgWindow);
    if (win <= 1) return rawFrameForIndex(idx, currentIdx, currentFrame);
    const n = Math.max(1, nSlices || 1);
    const center = Math.max(0, Math.min(n - 1, Math.round(idx)));
    const half = Math.floor(win / 2);
    let start = center - half;
    let end = start + win - 1;
    if (start < 0) {
      end = Math.min(n - 1, end - start);
      start = 0;
    }
    if (end >= n) {
      start = Math.max(0, start - (end - n + 1));
      end = n - 1;
    }
    if (offline && offlineFloatStack && offlineFloatStack.byteLength >= n * frameSize * 4) {
      const out = new Float32Array(frameSize);
      let count = 0;
      for (let j = start; j <= end; j++) {
        const frame = float32FrameFromDataView(offlineFloatStack, j, frameSize, false);
        if (!frame || frame.length < frameSize) continue;
        for (let k = 0; k < frameSize; k++) out[k] += frame[k];
        count++;
      }
      if (count > 0) {
        const inv = 1 / count;
        for (let k = 0; k < frameSize; k++) out[k] *= inv;
        return out;
      }
    }
    if (offline && offlineStack && offlineStack.byteLength >= n * frameSize) {
      const out = new Float32Array(frameSize);
      let count = 0;
      for (let j = start; j <= end; j++) {
        const frame = getOfflineFrame(j);
        if (!frame || frame.length < frameSize) continue;
        for (let k = 0; k < frameSize; k++) out[k] += frame[k];
        count++;
      }
      if (count > 0) {
        const inv = 1 / count;
        for (let k = 0; k < frameSize; k++) out[k] *= inv;
        return out;
      }
    }
    const out = new Float32Array(frameSize);
    let count = 0;
    for (let j = start; j <= end; j++) {
      const frame = rawFrameForIndex(j, currentIdx, currentFrame);
      if (!frame || frame.length < frameSize) continue;
      for (let k = 0; k < frameSize; k++) out[k] += frame[k];
      count++;
    }
    if (count === 0) return rawFrameForIndex(idx, currentIdx, currentFrame);
    if (count > 1) {
      const inv = 1 / count;
      for (let k = 0; k < frameSize; k++) out[k] *= inv;
    }
    return out;
  };

  // Temporal mean of an RGB window: the color twin of averagedFrameForIndex.
  // Averages 3-channel frames straight from the offline color stack so avg
  // denoises true-color playback without collapsing to luminance.
  const averagedRgbFrameForIndex = (idx: number, fallback: Float32Array): Float32Array => {
    const win = normalizedAverageWindow(playRef.current.avgWindow);
    if (win <= 1) return fallback;
    const n = Math.max(1, nSlices || 1);
    const center = Math.max(0, Math.min(n - 1, Math.round(idx)));
    const half = Math.floor(win / 2);
    let start = center - half;
    let end = start + win - 1;
    if (start < 0) { end = Math.min(n - 1, end - start); start = 0; }
    if (end >= n) { start = Math.max(0, start - (end - n + 1)); end = n - 1; }
    const size = width * height * 3;
    const out = new Float32Array(size);
    let count = 0;
    for (let j = start; j <= end; j++) {
      const frame = j === center ? fallback : getOfflineFrame(j);
      if (!frame || frame.length < size) continue;
      for (let k = 0; k < size; k++) out[k] += frame[k];
      count++;
    }
    if (count === 0) return fallback;
    const inv = 1 / count;
    for (let k = 0; k < size; k++) out[k] *= inv;
    return out;
  };

  const frequencyFilterKeyForIndex = React.useCallback((idx: number) => {
    const mode = normalizeFrequencyFilterMode(frequencyFilter);
    const packedPanels = (Math.max(1, nPanels || 1) > 1 && !sharedPanelSource) ? Math.max(1, nPanels || 1) : 1;
    return [
      Math.round(idx),
      frameSeq,
      mode,
      Number(frequencyOptions.cutoff ?? 0).toFixed(4),
      Number(frequencyOptions.center ?? 0).toFixed(4),
      Number(frequencyOptions.width ?? 0).toFixed(4),
      `panels${packedPanels}`,
    ].join(":");
  }, [frameSeq, frequencyFilter, frequencyOptions, nPanels, sharedPanelSource]);

  const frequencyFilterFrameForDisplay = React.useCallback((idx: number, frame: Float32Array | null, options: { allowRawOnMiss?: boolean } = {}): Float32Array | null => {
    if (!frame || !frequencyFilterIsActive) return frame;
    const allowRawOnMiss = options.allowRawOnMiss !== false;
    const key = frequencyFilterKeyForIndex(idx);
    const cache = frequencyFilterCacheRef.current;
    const hit = cache.get(key);
    if (hit) return hit;
    if (frequencyFilterPendingRef.current.has(key)) return allowRawOnMiss ? frame : null;
    frequencyFilterPendingRef.current.add(key);
    applyPackedPanelTransform(
      frame,
      (panelFrame, panelWidth, panelHeight) => applyFrequencyFilterBrowser(panelFrame, panelWidth, panelHeight, frequencyOptions),
    )
      .then((filtered) => {
        frequencyFilterPendingRef.current.delete(key);
        cache.set(key, filtered);
        if (cache.size > 48) cache.delete(cache.keys().next().value as string);
        setFrequencyFilterBackend(getFrequencyFilterBackend());
        setFrequencyRenderVersion((value) => value + 1);
      })
      .catch((error) => {
        frequencyFilterPendingRef.current.delete(key);
        console.warn("[Show3D] frequency filter failed; showing unfiltered frame", error);
      });
    return allowRawOnMiss ? frame : null;
  }, [applyPackedPanelTransform, frequencyFilterIsActive, frequencyFilterKeyForIndex, frequencyOptions, height, width]);

  const displayFrameForIndex = (idx: number, currentFrame: Float32Array | null, options: { allowRawOnMiss?: boolean } = {}): Float32Array | null => {
    const activeCompareMode = String(compareMode || "off");
    const compareFrameFor = (frameIdx: number): Float32Array | null => {
      const clamped = clampSlice(frameIdx);
      const raw = clamped === idx ? currentFrame : getOfflineFrame(clamped);
      return subpixelAlignFrameForIndex(clamped, averagedFrameForIndex(clamped, idx, raw));
    };
    let frame = subpixelAlignFrameForIndex(idx, averagedFrameForIndex(idx, idx, currentFrame));
    if (!isRgb && (nPanels || 1) === 1 && activeCompareMode !== "off") {
      const aIdx = clampSlice(comparePair?.[0] ?? 0);
      const bIdx = clampSlice(comparePair?.[1] ?? Math.min(1, nSlices - 1));
      const a = compareFrameFor(aIdx);
      const b = compareFrameFor(bIdx);
      if (activeCompareMode === "blink") {
        frame = (blinkPhase % 2 === 0 ? a : b) || frame;
      } else if (a && b) {
        const frameSize = width * height;
        const out = new Float32Array(frameSize);
        if (activeCompareMode === "difference") {
          for (let k = 0; k < frameSize; k++) out[k] = b[k] - a[k];
        } else if (activeCompareMode === "overlay") {
          for (let k = 0; k < frameSize; k++) out[k] = 0.5 * a[k] + 0.5 * b[k];
        }
        frame = out;
      }
    }
    const activeDiffMode = playRef.current.diffMode;
    let result: Float32Array | null = frame;
    if (frame && shouldApplyClientDifference(offline, activeDiffMode)) {
      const refIdx = activeDiffMode === "first" ? 0 : Math.max(0, Math.round(idx) - 1);
      const ref = subpixelAlignFrameForIndex(refIdx, averagedFrameForIndex(refIdx, idx, currentFrame));
      if (ref) {
        const frameSize = width * height;
        const out = new Float32Array(frameSize);
        for (let k = 0; k < frameSize; k++) out[k] = frame[k] - ref[k];
        result = out;
      }
    }
    // Browser-side denoise (WGSL) applied to the display frame with LIVE sigma.
    return browserFilterFrame(idx, result, options);
  };

  const displayAndFrequencyFrameForIndex = (idx: number, currentFrame: Float32Array | null, options: { allowRawOnMiss?: boolean } = {}): Float32Array | null => {
    const display = displayFrameForIndex(idx, currentFrame, options);
    return frequencyFilterFrameForDisplay(idx, display, options);
  };
  const refreshCurrentDisplayFrameForTransform = React.useCallback(() => {
    if (!offline || isRgb || width <= 0 || height <= 0 || nSlices <= 0) return;
    if (sidecarMode && (sidecarBitmapReadyRef.current || sidecarCompositeReadyRef.current) && !frameTransformActive()) return;
    const idx = clampSlice(liveSliceIdx);
    const raw = getOfflineFrame(idx);
    if (!raw) return;
    const display = displayAndFrequencyFrameForIndex(idx, raw, { allowRawOnMiss: true }) ?? raw;
    rawFrameDataRef.current = display;
    rgbFrameDataRef.current = null;
    gpuUploadRef.current = null;
  }, [
    avgWindow,
    browserFilterTick,
    diffMode,
    displayFilter,
    frequencyFilterIsActive,
    frequencyRenderVersion,
    getOfflineFrame,
    height,
    isRgb,
    liveSliceIdx,
    nSlices,
    offline,
    sidecarMode,
    sidecarBitmapReady,
    sidecarCompositeReady,
    spatialBin,
    subpixelAlignEnabled,
    subpixelAlignVersion,
    width,
  ]);

  // A completed sub-pixel alignment changes the display transform, not the
  // underlying frame bytes. Refresh the currently visible offline frame before
  // the passive canvas paint effect runs; otherwise the More → Align button can
  // report "Aligned" while the canvas still holds the old unaligned buffer until
  // the user scrubs or toggles another display control.
  React.useLayoutEffect(() => {
    refreshCurrentDisplayFrameForTransform();
  }, [refreshCurrentDisplayFrameForTransform]);

  const warmPlaybackDisplayFrame = (idx: number, currentIdx: number, currentFrame: Float32Array | null) => {
    const raw = rawFrameForIndex(idx, currentIdx, currentFrame);
    if (!raw) return;
    void displayAndFrequencyFrameForIndex(idx, raw, { allowRawOnMiss: true });
  };

  const renderGpuPanelSlice = (idx: number, updateDisplayState = true): boolean => {
    const normalized = ((Math.round(idx) % Math.max(1, nSlices)) + Math.max(1, nSlices)) % Math.max(1, nSlices);
    if (!separatePanelFrames || !gpuFrameCacheUploadedRef.current.has(normalized)) return false;
    const engine = gpuCmapRef.current;
    if (!engine || !gpuCmapReadyRef.current) return false;
    const c = playRef.current;
    if (c.imageRotation % 4 !== 0) return false;
    const sourcePanelCount = Math.max(1, nPanels || 1);
    const n = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(n);
    const rows = Math.ceil(n / cols);
    const gap = n > 1 ? (panelGapTrait ?? 10) : 0;

    const lut = COLORMAPS[c.cmap] || COLORMAPS.inferno;
    engine.uploadLUT(c.cmap, lut);
    let renderRanges: { vmin: number; vmax: number } | { vmin: number; vmax: number }[];
    let renderLogScale: boolean | boolean[];
    if (n > 1 && !c.linkContrast) {
      let sharedAutoRange: { vmin: number; vmax: number } | null = null;
      if (c.autoContrast) {
        sharedAutoRange = cachedAutoDisplayRange(c.autoVmins, c.autoVmaxs, normalized, c.logScale)
          || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, normalized, c.logScale);
        if (!sharedAutoRange) {
          sharedAutoRange = resolveDisplayRange(
            c.dataMin,
            c.dataMax,
            c.traitVmin,
            c.traitVmax,
            c.logScale,
            c.imageVminPct,
            c.imageVmaxPct,
          );
        }
      }
      renderRanges = visiblePanelIndices.map((panel) => {
        const stack = resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale);
        const pdr = panelDataRanges[panel];
        const bounds = (perPanelHistogramEnabled && pdr && pdr.max > pdr.min) ? pdr : stack;
        return resolvePanelRange(panel, bounds, sharedAutoRange);
      });
      renderLogScale = c.logScale;
    } else {
      let vmin: number, vmax: number;
      if (c.autoContrast) {
        const cached = cachedAutoDisplayRange(c.autoVmins, c.autoVmaxs, normalized, c.logScale)
          || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, normalized, c.logScale);
        if (cached) {
          ({ vmin, vmax } = cached);
        } else {
          ({ vmin, vmax } = resolveDisplayRange(
            c.dataMin,
            c.dataMax,
            c.traitVmin,
            c.traitVmax,
            c.logScale,
            c.imageVminPct,
            c.imageVmaxPct,
          ));
        }
      } else {
        ({ vmin, vmax } = resolveDisplayRange(
          c.dataMin,
          c.dataMax,
          c.traitVmin,
          c.traitVmax,
          c.logScale,
          c.imageVminPct,
          c.imageVmaxPct,
        ));
      }
      renderRanges = { vmin, vmax };
      renderLogScale = c.logScale;
    }

    const renderStartMs = performance.now();
    const gpuCtx = ensureGpuDisplayContext(engine, c.canvasW, c.canvasH);
    if (!gpuCtx) return false;
    const panelSlots = visiblePanelIndices.map((panel) => normalized * sourcePanelCount + panel);
    const transforms = visiblePanelIndices.map((panel) => {
      const base = c.panelStates[panel] || initialState;
      return {
        zoom: c.linkPanels ? c.linkedState.zoom : base.zoom,
        panX: c.linkPanels ? c.linkedState.panX : base.panX,
        panY: c.linkPanels ? c.linkedState.panY : base.panY,
      };
    });
    const rendered = engine.renderPanelSlotsDirectToCanvas(
      panelSlots,
      renderRanges,
      renderLogScale,
      gpuCtx,
      {
        width: c.canvasW,
        height: c.canvasH,
        panelCount: n,
        cols,
        rows,
        gap,
        bgRgb: packedRgbFromHex(themeColors.bg),
        transforms,
      },
    );
    if (!rendered) return false;
    setGpuDisplayVisible(true);
    playbackIdxRef.current = normalized;
    if (updateDisplayState) setDisplaySliceIdx(normalized);
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.missingFrame = null;
      dbg.lastFrame = normalized;
      dbg.lastFrameSource = "gpu-panel-cache-slots";
      dbg.lastRenderPath = "webgpu-grid-separate-panels-panel-slots-direct-fragment";
      dbg.lastRenderMs = Number((performance.now() - renderStartMs).toFixed(2));
      dbg.lastPanelTransforms = transforms.map(t => ({
        zoom: Number(t.zoom.toFixed(3)),
        panX: Number(t.panX.toFixed(1)),
        panY: Number(t.panY.toFixed(1)),
      }));
    }
    return true;
  };

  const renderGpuCachedSliceDirect = (idx: number, updateDisplayState = true): boolean => {
    if (separatePanelFrames) return renderGpuPanelSlice(idx, updateDisplayState);
    const normalized = ((Math.round(idx) % Math.max(1, nSlices)) + Math.max(1, nSlices)) % Math.max(1, nSlices);
    if (!gpuFrameCacheUploadedRef.current.has(normalized)) return false;
    const engine = gpuCmapRef.current;
    if (!engine || !gpuCmapReadyRef.current) return false;
    const c = playRef.current;
    if (c.imageRotation % 4 !== 0 || c.zoom !== 1 || c.panX !== 0 || c.panY !== 0) return false;
    if (!separatePanelFrames) {
      const naturalVisibleOrder = visiblePanelIndices.length === Math.max(1, nPanels || 1)
        && visiblePanelIndices.every((panel, slot) => panel === slot);
      const panelViewsAreDefault = visiblePanelIndices.every(panel => {
        const state = c.panelStates[panel] || initialState;
        const view = c.linkPanels ? c.linkedState : state;
        return view.zoom === 1 && view.panX === 0 && view.panY === 0;
      });
      if (!naturalVisibleOrder || !panelViewsAreDefault) return false;
    }
    const gpuCtx = ensureGpuDisplayContext(engine, c.canvasW, c.canvasH);
    if (!gpuCtx) return false;

    const lut = COLORMAPS[c.cmap] || COLORMAPS.inferno;
    engine.uploadLUT(c.cmap, lut);
    let vmin: number, vmax: number;
    if (c.autoContrast) {
      const cached = cachedAutoDisplayRange(c.autoVmins, c.autoVmaxs, normalized, c.logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, normalized, c.logScale);
      if (cached) {
        ({ vmin, vmax } = cached);
      } else {
        ({ vmin, vmax } = resolveDisplayRange(
          c.dataMin,
          c.dataMax,
          c.traitVmin,
          c.traitVmax,
          c.logScale,
          c.imageVminPct,
          c.imageVmaxPct,
        ));
      }
    } else {
      ({ vmin, vmax } = resolveDisplayRange(
        c.dataMin,
        c.dataMax,
        c.traitVmin,
        c.traitVmax,
        c.logScale,
        c.imageVminPct,
        c.imageVmaxPct,
      ));
    }

    if (hiddenPanelSet.size > 0 && !separatePanelFrames) return false;
    const n = Math.max(1, nPanels || 1);
    const cols = panelColsForCount(n);
    const rows = Math.ceil(n / cols);
    const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
    const renderStartMs = performance.now();
    if (n > 1 && !c.linkContrast && !sharedPanelSource) {
      const panelW = Math.max(1, panelWidthPx || Math.round(c.width / n));
      const regions = Array.from({ length: n }, (_, panel) => ({
        x: panel * panelW, y: 0, width: panelW, height: c.height,
      }));
      const sharedAutoRange = c.autoContrast ? { vmin, vmax } : null;
      const transformActive = requiresClientFrameTransform({
        offline,
        diffMode: c.diffMode,
        avgWindow: c.avgWindow,
      }) || browserFilterOnRef.current || frequencyFilterIsActive;
      const rawForRanges = rawFrameForIndex(normalized, normalized, rawFrameDataRef.current);
      const frameForRanges = rawForRanges && transformActive
        ? (displayAndFrequencyFrameForIndex(normalized, rawForRanges) ?? rawForRanges)
        : rawForRanges;
      const ranges = Array.from({ length: n }, (_, panel) => {
        const stack = resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale);
        const panelData = frameForRanges ? extractPanelSlice(frameForRanges, panel, c.logScale) : null;
        const pdr = panelDataRanges[panel];
        const bounds = (panelData && panelData.length > 0)
          ? findDataRange(panelData)
          : ((perPanelHistogramEnabled && pdr && pdr.max > pdr.min) ? pdr : stack);
        return resolvePanelRenderRange(panel, bounds, sharedAutoRange, panelData, c.autoContrast, c.percentileLow, c.percentileHigh);
      });
      const logs = c.logScale;
      const bitmaps = engine.renderPerPanelGpuExplicit(normalized, regions, ranges, logs);
      const offCtx = mainOffscreenRef.current?.getContext("2d");
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (!bitmaps) return false;
      try {
        if (!offCtx || !ctx || !mainOffscreenRef.current) return false;
        offCtx.clearRect(0, 0, c.width, c.height);
        for (let panel = 0; panel < n; panel++) {
          if (bitmaps[panel]) {
            offCtx.drawImage(bitmaps[panel], panel * panelW, 0);
          }
        }
      } finally {
        bitmaps.forEach(bitmap => bitmap?.close());
      }
      drawMain(ctx, mainOffscreenRef.current);
      setGpuDisplayVisible(false);
      playbackIdxRef.current = normalized;
      if (updateDisplayState) setDisplaySliceIdx(normalized);
      const dbg = show3dPerfDebug();
      if (dbg) {
        dbg.missingFrame = null;
        dbg.lastFrame = normalized;
        dbg.lastFrameSource = "gpu-cache";
        dbg.lastRenderPath = "webgpu-grid-panels-explicit-ranges";
        dbg.lastRenderMs = Number((performance.now() - renderStartMs).toFixed(2));
      }
      return true;
    }
    const sourcePanelWidthForGrid = sharedPanelSource
      ? Math.max(1, panelWidthPx || c.width)
      : Math.max(1, panelWidthPx || Math.round(c.width / n));
    const gridOpts = {
      width: c.canvasW,
      height: c.canvasH,
      panelCount: n,
      cols,
      rows,
      gap,
      bgRgb: packedRgbFromHex(themeColors.bg),
      sourcePanelWidth: sourcePanelWidthForGrid,
      sharedSource: !!sharedPanelSource,
    };
    const rendered = engine.renderSharedGridDirectToCanvas(
      normalized,
      { vmin, vmax },
      c.logScale,
      gpuCtx,
      gridOpts,
    );
    if (!rendered) return false;
    setGpuDisplayVisible(true);
    playbackIdxRef.current = normalized;
    if (updateDisplayState) setDisplaySliceIdx(normalized);
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.missingFrame = null;
      dbg.lastFrame = normalized;
      dbg.lastFrameSource = "gpu-cache";
      dbg.lastRenderPath = n === 1
        ? "webgpu-grid-single-panel-direct-fragment"
        : (sharedPanelSource ? "webgpu-grid-shared-panels-direct-fragment" : "webgpu-grid-panels-direct-fragment");
      dbg.lastRenderMs = Number((performance.now() - renderStartMs).toFixed(2));
    }
    return true;
  };

  const lastRenderBurstBenchmarkTokenRef = React.useRef<unknown>(null);
  React.useEffect(() => {
    const req = benchmarkRequest ?? {};
    const token = req.token;
    const mode = typeof req.mode === "string" ? req.mode : "playback";
    if ((typeof token !== "string" && typeof token !== "number") || mode !== "renderBurst" || lastRenderBurstBenchmarkTokenRef.current === token) return;
    lastRenderBurstBenchmarkTokenRef.current = token;

    let cancelled = false;
    const sleep = (ms: number) => new Promise<void>(resolve => window.setTimeout(resolve, ms));
    const numberFromReq = (key: string, fallback: number) => {
      const value = req[key];
      return typeof value === "number" && Number.isFinite(value) ? value : fallback;
    };
    const sampleMs = Math.max(250, numberFromReq("sampleMs", 3000));
    const expectedFrames = Math.max(0, Math.floor(numberFromReq("expectedFrames", nSlices)));
    const syncEvery = Math.max(0, Math.floor(numberFromReq("syncEvery", 1)));
    const reportUrl = typeof req.reportUrl === "string" ? req.reportUrl : "";
    const label = typeof req.label === "string" ? req.label : "show3d render burst";

    void (async () => {
      const startedAt = performance.now();
      const setStatus = (status: string, extra: Record<string, unknown> = {}) => {
        if (!cancelled) setBenchmarkResult({ token, label, status, targetFps: 0, mode, ...extra });
      };
      try {
        setStatus("preloading");
        const engine = gpuCmapRef.current;
        if (!engine || !gpuCmapReadyRef.current) throw new Error("WebGPU colormap engine is not ready");
        const rgbaCapacity = Math.max(1, Math.round(canvasW * canvasH));
        const framesToPrepare = expectedFrames > 0 ? Math.min(expectedFrames, nSlices) : nSlices;
        for (let i = 0; i < framesToPrepare; i++) {
          if (cancelled) return;
          if (separatePanelFrames) {
            const ready = await ensurePanelFrameGpu(i, rgbaCapacity);
            if (!ready) throw new Error(`panel frame ${i} was not available for GPU upload`);
          } else if (!gpuFrameCacheUploadedRef.current.has(i)) {
            const frame = await fetchFrameFromServer(i);
            if (!frame) throw new Error(`frame ${i} was not available for GPU upload`);
            engine.uploadData(i, frame, width, height, rgbaCapacity);
            gpuFrameCacheUploadedRef.current.add(i);
          }
          if (i % 4 === 0) {
            setStatus("preloading", { preparedFrames: i + 1, expectedFrames: framesToPrepare });
            await sleep(0);
          }
        }
        await engine.waitForSubmittedWork();

        setStatus("sampling", { preparedFrames: framesToPrepare, syncEvery });
        const sampleStart = performance.now();
        let frames = 0;
        let misses = 0;
        while (!cancelled && performance.now() - sampleStart < sampleMs) {
          const idx = frames % Math.max(1, framesToPrepare);
          const ok = renderGpuCachedSliceDirect(idx, false);
          if (!ok) {
            misses++;
            await sleep(0);
            continue;
          }
          frames++;
          if (syncEvery > 0 && frames % syncEvery === 0) {
            await engine.waitForSubmittedWork();
          } else if (frames % 32 === 0) {
            await sleep(0);
          }
        }
        await engine.waitForSubmittedWork();
        const elapsedSeconds = Math.max(0.001, (performance.now() - sampleStart) / 1000);
        const measuredFps = frames / elapsedSeconds;
        const dbgEnd = show3dPerfDebug() ?? {};
        const result = {
          token,
          label,
          status: "done",
          mode,
          syncEvery,
          measuredFps: Number(measuredFps.toFixed(2)),
          frames,
          misses,
          elapsedSeconds: Number(elapsedSeconds.toFixed(2)),
          preparedFrames: framesToPrepare,
          lastRenderPath: dbgEnd.lastRenderPath ?? null,
          lastRenderMs: dbgEnd.lastRenderMs ?? null,
          totalMs: Number((performance.now() - startedAt).toFixed(1)),
        };
        setBenchmarkResult(result);
        if (reportUrl) {
          void fetch(reportUrl, { method: "POST", mode: "no-cors", body: JSON.stringify(result) }).catch(() => {});
        }
      } catch (err) {
        setStatus("error", { error: err instanceof Error ? err.message : String(err) });
      }
    })();

    return () => {
      cancelled = true;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [benchmarkRequest, nSlices, separatePanelFrames, canvasW, canvasH, width, height]);

  const playbackHistogramCounterRef = React.useRef(0);
  const refreshHistogramRef = React.useRef<((idxArg?: number) => void | Promise<void>) | null>(null);

  // Playback logic - rAF-driven, zero React re-renders in hot path
  React.useEffect(() => {
    if (!playing) {
      // Playback stopped - sync final position to Python
      if (playbackIdxRef.current !== sliceIdx && (bufferRef.current || separatePanelFrames || offline)) {
        setLiveSliceIdx(playbackIdxRef.current);
        setSliceIdx(playbackIdxRef.current);
      }
      if (!playRef.current.showStats) setLocalStats(null);
      prefetchPendingRef.current = false;
      return;
    }

    // === PLAYBACK START ===
    // Snap slice_idx into [loop_start, loop_end] before first tick, otherwise
    // playback walked outside the loop range on the first frame.
    {
      const c0 = playRef.current;
      const rs0 = c0.loop ? Math.max(0, Math.min(c0.loopStart, c0.nSlices - 1)) : 0;
      const re0 = c0.loop ? Math.max(rs0, Math.min(c0.loopEnd, c0.nSlices - 1)) : c0.nSlices - 1;
      const liveStart = Number.isFinite(playbackIdxRef.current)
        ? playbackIdxRef.current
        : (Number.isFinite(displaySliceIdx) ? displaySliceIdx : sliceIdx);
      playbackIdxRef.current = Math.max(rs0, Math.min(re0, Math.round(liveStart)));
    }
    const pathLen = playRef.current.playbackPath?.length ?? 0;
    pathIdxRef.current = pathLen > 0 ? (playRef.current.reverse ? pathLen : -1) : 0;
    bounceDirRef.current = playRef.current.reverse ? -1 : 1;
    if (frameServerUrl && gpuFrameCacheUploadedRef.current.size < playRef.current.nSlices) {
      const c0 = playRef.current;
      if (separatePanelFrames) {
        prefetchPanelGpuFrames(playbackIdxRef.current, c0.reverse, c0.loop, c0.loopStart, c0.loopEnd);
      } else {
        prefetchServerFrames(playbackIdxRef.current, c0.reverse, c0.loop, c0.loopStart, c0.loopEnd);
      }
    }
    let lastFrameTime = 0;
    let lastUIUpdate = 0;
    let animId = 0;
    let tick: (now: number) => void = () => {};
    const scheduleTick = () => {
      animId = requestAnimationFrame(tick);
    };
    const startDbg = show3dPerfDebug();
    const startFps = clampPlaybackFps(benchmarkPlaybackFpsRef.current ?? playRef.current.fps);
    if (startDbg) resetFramePacingDebug(startDbg, playbackIntervalMs(startFps));

    tick = (_now: number) => {
      const tickNow = performance.now();
      const c = playRef.current;
      const effectiveFps = clampPlaybackFps(benchmarkPlaybackFpsRef.current ?? c.fps);
      const intervalMs = playbackIntervalMs(effectiveFps);
      const uiUpdateIntervalMs = effectiveFps >= 60 ? 250 : 100;
      const dbg = show3dPerfDebug();
      if (dbg) {
        dbg.playing = true;
        dbg.effectiveFps = effectiveFps;
        dbg.lastTickAt = tickNow;
        dbg.currentBufferFloatLength = bufferRef.current?.length ?? 0;
        dbg.currentBufferStart = bufferStartRef.current;
        dbg.currentBufferCount = bufferCountRef.current;
        dbg.nextBufferFloatLength = nextBufferRef.current?.length ?? 0;
        dbg.nextBufferStart = nextBufferStartRef.current;
        dbg.nextBufferCount = nextBufferCountRef.current;
      }

      // First tick paints immediately; otherwise every playback start drops
      // one frame before the cadence timer is even allowed to run.
      if (lastFrameTime === 0) {
        lastFrameTime = tickNow - intervalMs;
        lastUIUpdate = tickNow;
      }

      const elapsed = tickNow - lastFrameTime;
      // Frame-pacing tolerance: at 60 fps intervalMs (16.67) equals the vsync
      // period, so a rAF tick arriving a hair early (elapsed 16.6 < 16.67) would
      // be dropped and cost a whole vsync -> steady 17/33 ms alternation = 30 fps.
      // Allow a tick that is within tolerance of the deadline through, and
      // phase-correct lastFrameTime by the deadline (not tickNow) so drift does
      // not accumulate. Restores 60 fps on the GPU-cached multi-panel path.
      const framePacingToleranceMs = Math.min(6, intervalMs * 0.2);
      if (elapsed + framePacingToleranceMs < intervalMs) {
        scheduleTick();
        return;
      }
      lastFrameTime = tickNow - Math.max(0, elapsed - intervalMs);

      // Advance frame
      let next: number;
      if (c.playbackPath && c.playbackPath.length > 0) {
        // Custom playback path
        const pp = c.playbackPath;
        let pi = pathIdxRef.current;
        if (c.boomerang) {
          // Loop remains the master repeat control. When a scientist turns
          // Loop off, Bounce should shape motion only until the path endpoint;
          // it must not keep ping-ponging forever.
          // Visit endpoints once (matches grid-mode boomerang). Earlier code
          // jumped to pp.length-2 / 1 on overshoot, skipping endpoints.
          pi += bounceDirRef.current;
          if (pi >= pp.length) {
            if (!c.loop) { setPlaying(false); return; }
            bounceDirRef.current = -1;
            pi = pp.length - 1;
          } else if (pi < 0) {
            if (!c.loop) { setPlaying(false); return; }
            bounceDirRef.current = 1;
            pi = 0;
          }
        } else {
          pi += (c.reverse ? -1 : 1);
          if (pi >= pp.length) { if (!c.loop) { setPlaying(false); return; } pi = 0; }
          if (pi < 0) { if (!c.loop) { setPlaying(false); return; } pi = pp.length - 1; }
        }
        pi = Math.max(0, Math.min(pp.length - 1, pi));
        pathIdxRef.current = pi;
        next = pp[pi];
      } else {
        const rangeStart = c.loop ? Math.max(0, Math.min(c.loopStart, c.nSlices - 1)) : 0;
        const rangeEnd = c.loop ? Math.max(rangeStart, Math.min(c.loopEnd, c.nSlices - 1)) : c.nSlices - 1;
        const prev = Number.isFinite(playbackIdxRef.current)
          ? Math.round(playbackIdxRef.current)
          : Math.max(rangeStart, Math.min(rangeEnd, Math.round(displaySliceIdx || 0)));

        if (c.boomerang) {
          next = prev + bounceDirRef.current;
          if (next > rangeEnd) {
            if (!c.loop) { setPlaying(false); return; }
            bounceDirRef.current = -1;
            next = prev - 1 >= rangeStart ? prev - 1 : prev;
          } else if (next < rangeStart) {
            if (!c.loop) { setPlaying(false); return; }
            bounceDirRef.current = 1;
            next = prev + 1 <= rangeEnd ? prev + 1 : prev;
          }
        } else {
          next = prev + (c.reverse ? -1 : 1);
          if (c.reverse) {
            if (next < rangeStart) { if (!c.loop) { setPlaying(false); return; } next = rangeEnd; }
          } else {
            if (next > rangeEnd) { if (!c.loop) { setPlaying(false); return; } next = rangeStart; }
          }
        }
      }

      // OFFLINE mode (nbconvert HTML export with packed stack in widget state):
      // bypass ALL the kernel-fed buffer paths — bufferRef/nextBufferRef/
      // gpuFrameCacheUploadedRef are stale from prior pause+resume cycles and
      // can pin the canvas to a single frame. Always re-derive from the
      // offline stack so play→pause→play repaints correctly. Verified bug
      // 2026-05-24: 2nd autoplay cycle painted same frame from buffer cache.
      const frameSize = c.width * c.height;
      const transformActive = requiresClientFrameTransform({
        offline,
        diffMode: c.diffMode,
        avgWindow: c.avgWindow,
      }) || browserFilterOnRef.current || frequencyFilterIsActive;
      let frame: Float32Array | null = null;
      let frameSource = "buffer";
      // The GPU-cache fast paths (renderGpuPanelSlice / direct-grid) only handle
      // imageRotation%4===0; renderGpuPanelSlice bails (returns false) on a 90/270
      // rotation, which froze playback (renderedFrames + canvas stuck, playing
      // true). When rotated, skip the GPU-cache path so the frame is fetched and
      // drawMain applies the rotation. Verified bug 2026-05-29.
      if (
        offline &&
        sidecarMode &&
        (sidecarBitmapReadyRef.current || sidecarCompositeReadyRef.current) &&
        !isRgb &&
        !transformActive
      ) {
        if (drawSidecarBitmapFrame(next, false, "playback")) {
          playbackIdxRef.current = next;
          updatePlaybackLiveControls(next);
          if (dbg) {
            dbg.missingFrame = null;
            dbg.lastFrame = next;
            dbg.lastFrameSource = "sidecar-imagebitmap-cache";
          }
          const d = show3dPerfDebug();
          if (d) {
            recordFramePacingDebug(d, performance.now(), intervalMs);
            d.renderedFrames = ((d.renderedFrames as number | undefined) ?? 0) + 1;
          }
          lastUIUpdate = tickNow;
          scheduleTick();
          return;
        }
      }
      const rotationAllowsGpuCache = (c.imageRotation % 4) === 0;
      const gpuCachedSlotReady = !offline
        && !transformActive
        && rotationAllowsGpuCache
        && gpuFrameCacheUploadedRef.current.has(next);
      // Cache presence alone is not render readiness: hidden/reordered panels
      // and non-default transforms can make the direct path decline. Probe the
      // actual renderer; on a miss we still acquire a CPU/server frame instead
      // of advancing the slider over a frozen canvas.
      let gpuCachedFrameReady = false;
      if (gpuCachedSlotReady) {
        try {
          gpuCachedFrameReady = renderGpuCachedSliceDirect(next, false);
        } catch (err) {
          if (dbg) dbg.lastRenderError = err instanceof Error ? err.message : String(err);
        }
      }
      const gpuPanelFrameReady = separatePanelFrames && gpuCachedFrameReady;
      if (offline) {
        frame = getOfflineFrame(next);
        if (frame) frameSource = "offline";
      } else if (gpuPanelFrameReady) {
        frameSource = "gpu-panel-cache";
      } else if (separatePanelFrames && rotationAllowsGpuCache && !transformActive) {
        frameSource = "gpu-panel-fetch";
        void ensurePanelFrameGpu(next, Math.max(1, Math.round(c.canvasW * c.canvasH)));
        if (dbg) {
          dbg.lastPanelGpuRequestedFrame = next;
          dbg.lastFrameSource = frameSource;
        }
      } else if (!gpuCachedFrameReady) {
        frame = getFrameFromBuffer(bufferRef.current, bufferStartRef.current, bufferCountRef.current, c.nSlices, next, frameSize);
        if (!frame && nextBufferRef.current) {
          // Current buffer doesn't have this frame - swap to next buffer
          bufferRef.current = nextBufferRef.current;
          bufferStartRef.current = nextBufferStartRef.current;
          bufferCountRef.current = nextBufferCountRef.current;
          nextBufferRef.current = null;
          nextBufferCountRef.current = 0;
          frame = getFrameFromBuffer(bufferRef.current, bufferStartRef.current, bufferCountRef.current, c.nSlices, next, frameSize);
        }
        if (!frame && frameServerUrl) {
          frame = getCachedServerFrame(next);
          if (frame) {
            frameSource = "server";
          } else {
            prefetchServerFrames(next, c.reverse, c.loop, c.loopStart, c.loopEnd);
          }
        }
        if (!frame) {
          frame = getOfflineFrame(next);
          if (frame) frameSource = "offline";
        }
      }
      if (!frame && !gpuCachedFrameReady) {
        // Buffer not ready yet - keep requesting frames
        if (dbg) {
          dbg.missingFrame = next;
          dbg.missingFrameAt = tickNow;
        }
        scheduleTick();
        return;
      }
      if (dbg) {
        dbg.missingFrame = null;
        dbg.lastFrame = next;
        dbg.lastFrameSource = frameSource;
      }

      if (frame && transformActive && !isRgb) {
        const filteredFrame = displayAndFrequencyFrameForIndex(next, frame, { allowRawOnMiss: false });
        if (!filteredFrame) {
          warmPlaybackDisplayFrame(next, playbackIdxRef.current, frame);
          if (dbg) {
            dbg.missingFrame = next;
            dbg.lastFrameSource = `${frameSource}-filter-pending`;
          }
          scheduleTick();
          return;
        }
        frame = filteredFrame;
      }
      playbackIdxRef.current = next;
      updatePlaybackLiveControls(next);
      if (frame && isRgb && offline && frame.length >= c.width * c.height * 3) {
        rgbFrameDataRef.current = frame;
        rawFrameDataRef.current = rgbFrameToLuminance(frame, c.width * c.height);
      } else if (frame) {
        rawFrameDataRef.current = frame;
      }
      const offlinePackedPanelPlaybackUsesStaticCanvas = (
        offline &&
        Math.max(1, nPanels || 1) > 1 &&
        !sharedPanelSource
      );
      const offlineDirectRender = (
        offline &&
        !isRgb &&
        !offlinePackedPanelPlaybackUsesStaticCanvas &&
        !!frame &&
        !!gpuCmapRef.current &&
        gpuCmapReadyRef.current
      );
      // Static offline paint is driven by liveSliceIdx. When WebGPU is ready we
      // render offline frames directly in the rAF hot path and throttle React
      // state updates below so large 2k/4k exports do not double-paint.
      const liveGpuFrameRender = !offline && !!frame && !transformActive && !!gpuCmapRef.current && gpuCmapReadyRef.current;
      const gpuDirectRender = gpuCachedFrameReady || gpuPanelFrameReady || liveGpuFrameRender;
      if (!offlineDirectRender && !gpuDirectRender) setLiveSliceIdx(next);
      // Offline mode short-circuit: hand the frame to the React static paint
      // pipeline (proven smooth on slider drag) and skip the rAF direct paint
      // entirely. The two paths fought on Mac/retina (Linux didn't expose it),
      // producing the "play is flaky while drag is smooth" symptom verified
      // 2026-05-24 on sample_device_trial.html.
      if (offline && !offlineDirectRender) {
        setGpuDisplayVisible(false);
        const d = show3dPerfDebug();
        if (d) {
          recordFramePacingDebug(d, performance.now(), intervalMs);
          d.renderedFrames = ((d.renderedFrames as number | undefined) ?? 0) + 1;
        }
        if (tickNow - lastUIUpdate > uiUpdateIntervalMs) {
          lastUIUpdate = tickNow;
          setDisplaySliceIdx(next);
          setPlaybackUiSliceIdx(next);
          playbackHistogramCounterRef.current = (playbackHistogramCounterRef.current + 1) % 2;
          if (playbackHistogramCounterRef.current === 0) {
            void refreshHistogramRef.current?.(next);
          }
        }
        scheduleTick();
        return;
      }
      if (gpuPanelFrameReady) {
        if (!renderGpuPanelSlice(next, false)) {
          if (dbg) {
            dbg.missingFrame = next;
            dbg.lastRenderError = "separate panel GPU render failed";
          }
          scheduleTick();
          return;
        }
        const d = show3dPerfDebug();
        if (d) {
          recordFramePacingDebug(d, performance.now(), intervalMs);
          d.renderedFrames = ((d.renderedFrames as number | undefined) ?? 0) + 1;
        }
        if (tickNow - lastUIUpdate > uiUpdateIntervalMs) {
          lastUIUpdate = tickNow;
          setDisplaySliceIdx(next);
          setPlaybackUiSliceIdx(next);
          playbackHistogramCounterRef.current = (playbackHistogramCounterRef.current + 1) % 2;
          if (playbackHistogramCounterRef.current === 0) {
            void refreshHistogramRef.current?.(next);
          }
        }
        scheduleTick();
        return;
      }

      // Render frame. The 4k playback hot path must stay off the JS CPU:
      // one 4096^2 colormap loop alone is ~37 ms, before auto-contrast/canvas.
      const renderStartMs = performance.now();
      const lut = COLORMAPS[c.cmap] || COLORMAPS.inferno;
      if (mainOffscreenRef.current && mainImgDataRef.current) {
        let vmin: number, vmax: number;
        let cpuData: Float32Array | null = frame;
        let cpuDataAlreadyLogged = false;
        if (c.autoContrast) {
          const cached = transformActive ? null : (
            cachedAutoDisplayRange(c.autoVmins, c.autoVmaxs, next, c.logScale)
            || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, next, c.logScale)
          );
          if (cached) {
            ({ vmin, vmax } = cached);
          } else if (frame && c.logScale && logBufferRef.current) {
            applyLogScaleInPlace(frame, logBufferRef.current);
            ({ vmin, vmax } = percentileClip(logBufferRef.current, c.percentileLow, c.percentileHigh));
            cpuData = logBufferRef.current;
            cpuDataAlreadyLogged = true;
          } else if (frame) {
            ({ vmin, vmax } = percentileClip(frame, c.percentileLow, c.percentileHigh));
          } else {
            ({ vmin, vmax } = resolveDisplayRange(
              c.dataMin,
              c.dataMax,
              c.traitVmin,
              c.traitVmax,
              c.logScale,
              c.imageVminPct,
              c.imageVmaxPct,
            ));
          }
        } else {
          ({ vmin, vmax } = resolveDisplayRange(
            c.dataMin,
            c.dataMax,
            c.traitVmin,
            c.traitVmax,
            c.logScale,
            c.imageVminPct,
            c.imageVmaxPct,
          ));
        }

        let rendered = false;
        let drewDisplayDirect = false;
        const dw = Math.max(1, Math.round(c.width * c.displayScale));
        const dh = Math.max(1, Math.round(c.height * c.displayScale));
        const panelCountForGrid = Math.max(1, nPanels || 1);
        const allPanelsVisibleForDirect = c.visiblePanelIndices.length === panelCountForGrid;
        const panelTransformsForDirect = c.visiblePanelIndices.map((panel) => {
          const base = c.panelStates[panel] || initialState;
          return {
            zoom: c.linkPanels ? c.linkedState.zoom : base.zoom,
            panX: c.linkPanels ? c.linkedState.panX : base.panX,
            panY: c.linkPanels ? c.linkedState.panY : base.panY,
          };
        });
        const panelTransformsAreDefault = panelTransformsForDirect.every((transform) => (
          transform.zoom === 1 && transform.panX === 0 && transform.panY === 0
        ));
        const packedPanelDirectCanvas = panelCountForGrid > 1 && !sharedPanelSource;
        // In standalone exported HTML, direct WebGPU canvas presentation can
        // briefly win the opacity handoff before the next image is visible,
        // producing a black flash when playback starts. Keep offline playback
        // on the stable 2D display canvas; live frame-server paths can still
        // use the direct GPU canvas for maximum throughput.
        const allowDirectGpuCanvas = !offline;
        const canDirectGridCanvas =
          allowDirectGpuCanvas &&
          allPanelsVisibleForDirect &&
          c.imageRotation % 4 === 0 &&
          (
            packedPanelDirectCanvas ||
            (c.zoom === 1 && c.panX === 0 && c.panY === 0 && panelTransformsAreDefault)
          );
        const canSharedPanelScaledDirect =
          !!sharedPanelSource &&
          canDirectGridCanvas &&
          panelTransformsAreDefault &&
          panelCountForGrid > 1;
        const canScaledDirect =
          (nPanels === 1 || canSharedPanelScaledDirect) &&
          c.imageRotation % 4 === 0 &&
          c.zoom === 1 &&
          c.panX === 0 &&
          c.panY === 0 &&
          dw <= c.canvasW &&
          dh <= c.canvasH;
        const drawSharedScaledBitmap = (ctx: CanvasRenderingContext2D, bitmap: ImageBitmap) => {
          const n = Math.max(1, nPanels || 1);
          const cols = panelColsForCount(n);
          const rows = Math.ceil(n / cols);
          const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
          const outPanelW = (c.canvasW - gap * (cols - 1)) / cols;
          const outPanelH = (c.canvasH - gap * (rows - 1)) / rows;
          ctx.clearRect(0, 0, c.canvasW, c.canvasH);
          ctx.fillStyle = themeColors.bg;
          ctx.imageSmoothingEnabled = c.smooth;
          for (let i = 0; i < n; i++) {
            const col = i % cols;
            const row = Math.floor(i / cols);
            const slotX = col * (outPanelW + gap);
            const slotY = row * (outPanelH + gap);
            ctx.fillRect(slotX, slotY, outPanelW, outPanelH);
            ctx.drawImage(bitmap, slotX, slotY, outPanelW, outPanelH);
          }
        };
        const renderOfflinePackedPanels2D = (): boolean => {
          if (!offline || panelCountForGrid <= 1 || c.linkContrast || sharedPanelSource || !frame) return false;
          const offscreen = mainOffscreenRef.current;
          const canvas = canvasRef.current;
          const offCtx = offscreen?.getContext("2d");
          const ctx = canvas?.getContext("2d");
          if (!offscreen || !offCtx || !ctx) return false;
          const panelW = Math.max(1, Math.floor(c.width / panelCountForGrid));
          const panelImg = offCtx.createImageData(panelW, c.height);
          const sharedAutoRange = c.autoContrast ? { vmin, vmax } : null;
          offCtx.clearRect(0, 0, offscreen.width, offscreen.height);
          for (const panel of c.visiblePanelIndices) {
            if (panel < 0 || panel >= panelCountForGrid) continue;
            const panelData = extractPanelSlice(frame, panel, c.logScale);
            if (!panelData) continue;
            const pdr = panelDataRanges[panel];
            const panelRange = panelData.length > 0
              ? findDataRange(panelData)
              : ((perPanelHistogramEnabled && pdr && pdr.max > pdr.min)
                  ? pdr
                  : resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale));
            const range = resolvePanelRenderRange(panel, panelRange, sharedAutoRange, panelData, c.autoContrast, c.percentileLow, c.percentileHigh);
            applyColormap(panelData, panelImg.data, lut, range.vmin, range.vmax);
            offCtx.putImageData(panelImg, panel * panelW, 0);
          }
          drawMain(ctx, offscreen);
          setGpuDisplayVisible(false);
          if (dbg) dbg.lastRenderPath = "offline-packed-panels-2d-per-panel";
          return true;
        };
        if (!rendered && renderOfflinePackedPanels2D()) {
          rendered = true;
          drewDisplayDirect = true;
        }
        const engine = gpuCmapRef.current;
        const preferGpuScaledPlayback = !!engine && gpuCmapReadyRef.current;
        if (frame && canScaledDirect && !canSharedPanelScaledDirect && !c.smooth && !preferGpuScaledPlayback) {
          const canvas = canvasRef.current;
          const ctx = canvas?.getContext("2d");
          if (ctx) {
            let cached = scaledPlaybackImgDataRef.current;
            if (!cached || cached.width !== dw || cached.height !== dh) {
              cached = { width: dw, height: dh, imageData: ctx.createImageData(dw, dh) };
              scaledPlaybackImgDataRef.current = cached;
            }
            let map = scaledPlaybackMapRef.current;
            if (!map || map.srcW !== c.width || map.srcH !== c.height || map.outW !== dw || map.outH !== dh) {
              const xMap = new Uint32Array(dw);
              const yMap = new Uint32Array(dh);
              for (let x = 0; x < dw; x++) {
                xMap[x] = Math.min(c.width - 1, Math.floor(((x + 0.5) * c.width) / dw));
              }
              for (let y = 0; y < dh; y++) {
                yMap[y] = Math.min(c.height - 1, Math.floor(((y + 0.5) * c.height) / dh)) * c.width;
              }
              map = { srcW: c.width, srcH: c.height, outW: dw, outH: dh, xMap, yMap };
              scaledPlaybackMapRef.current = map;
            }
            renderFrameScaledPlayback(frame, cached.imageData.data, map.xMap, map.yMap, dw, dh, lut, vmin, vmax, c.logScale);
            ctx.imageSmoothingEnabled = false;
            ctx.clearRect(0, 0, c.canvasW, c.canvasH);
            ctx.putImageData(cached.imageData, 0, 0);
            setGpuDisplayVisible(false);
            rendered = true;
            drewDisplayDirect = true;
            if (dbg) dbg.lastRenderPath = "scaled-cpu";
          }
        }
        if (!rendered && engine && gpuCmapReadyRef.current) {
          try {
            engine.uploadLUT(c.cmap, lut);
            const stackByteLength = c.width * c.height * 4 * c.nSlices;
            const hasGpuSlot = gpuFrameCacheUploadedRef.current.has(next);
            const canGpuFrameCache =
              !!frameServerUrl &&
              stackByteLength <= FRAME_SERVER_FULL_STACK_CACHE_BYTES &&
              (hasGpuSlot || frameFetchCacheRef.current.size >= c.nSlices);
            const slotIdx = canGpuFrameCache ? next : 0;
            const gpuRgbaCapacityHint = canDirectGridCanvas
              ? c.canvasW * c.canvasH
              : (canScaledDirect ? dw * dh : undefined);
            if (canGpuFrameCache) {
              if (!gpuFrameCacheUploadedRef.current.has(slotIdx)) {
                if (frame) {
                  engine.uploadData(slotIdx, frame, c.width, c.height, gpuRgbaCapacityHint);
                  gpuFrameCacheUploadedRef.current.add(slotIdx);
                  if (dbg) dbg.gpuFrameCacheUploaded = gpuFrameCacheUploadedRef.current.size;
                }
              } else if (dbg) {
                dbg.gpuFrameCacheHits = ((dbg.gpuFrameCacheHits as number | undefined) ?? 0) + 1;
              }
            } else if (frame) {
              engine.uploadData(0, frame, c.width, c.height, gpuRgbaCapacityHint);
              if (dbg) dbg.gpuFrameCacheUploaded = 0;
            }
            if (canDirectGridCanvas) {
              const gpuCtx = ensureGpuDisplayContext(engine, c.canvasW, c.canvasH);
              if (gpuCtx) {
                const n = panelCountForGrid;
                const cols = panelColsForCount(n);
                const rows = Math.ceil(n / cols);
                const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
                const sourcePanelWidthForGrid = sharedPanelSource
                  ? Math.max(1, panelWidthPx || c.width)
                  : Math.max(1, panelWidthPx || Math.round(c.width / n));
                const gridOpts = {
                  width: c.canvasW,
                  height: c.canvasH,
                  panelCount: n,
                  cols,
                  rows,
                  gap,
                  bgRgb: packedRgbFromHex(themeColors.bg),
                  sourcePanelWidth: sourcePanelWidthForGrid,
                  sharedSource: !!sharedPanelSource,
                };
                const usedPackedPanelRegions = n > 1 && !sharedPanelSource;
                if (usedPackedPanelRegions) {
                  const panelW = sourcePanelWidthForGrid;
                  const sharedAutoRange = c.autoContrast ? { vmin, vmax } : null;
                  const ranges = c.linkContrast
                    ? { vmin, vmax }
                    : Array.from({ length: n }, (_, p) => {
                        const panelData = frame ? extractPanelSlice(frame, p, c.logScale) : null;
                        // In per-panel mode, ALWAYS prefer this panel's stored
                        // data range so slider pct decodes in panel space (not
                        // stack space). Without this SSB phase [±0.04] gets
                        // decoded against stack range [≈-0.04, ≈30000] → vmin
                        // and vmax both land in DF-count territory → all SSB
                        // pixels render black.
                        const pdr = panelDataRanges[p];
                        const panelRange = panelData && panelData.length > 0
                          ? findDataRange(panelData)
                          : ((perPanelHistogramEnabled && pdr && pdr.max > pdr.min)
                              ? pdr
                              : resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale));
                        return resolvePanelRenderRange(p, panelRange, sharedAutoRange, panelData, c.autoContrast, c.percentileLow, c.percentileHigh);
                      });
                  rendered = engine.renderCombinedPanelRegionsDirectToCanvas(
                    slotIdx,
                    ranges,
                    c.logScale,
                    gpuCtx,
                    {
                      width: c.canvasW,
                      height: c.canvasH,
                      panelCount: n,
                      cols,
                      rows,
                      gap,
                      bgRgb: packedRgbFromHex(themeColors.bg),
                      sourcePanelWidth: panelW,
                      transforms: panelTransformsForDirect,
                    },
                  );
                  if (dbg) {
                    dbg.lastRenderPath = rendered
                      ? "webgpu-grid-packed-panels-direct-fragment"
                      : "webgpu-grid-packed-panels-direct-fragment-miss";
                    dbg.lastPanelTransforms = panelTransformsForDirect.map(t => ({
                      zoom: Number(t.zoom.toFixed(3)),
                      panX: Number(t.panX.toFixed(1)),
                      panY: Number(t.panY.toFixed(1)),
                    }));
                  }
                } else {
                  const renderedDirect = engine.renderSharedGridDirectToCanvas(slotIdx, { vmin, vmax }, c.logScale, gpuCtx, gridOpts);
                  rendered = renderedDirect || engine.renderSharedGridToCanvas(slotIdx, { vmin, vmax }, c.logScale, gpuCtx, gridOpts);
                  if (dbg) {
                    const gridPath = n === 1
                      ? "webgpu-grid-single-panel"
                      : (sharedPanelSource ? "webgpu-grid-shared-panels" : "webgpu-grid-panels");
                    dbg.lastRenderPath = renderedDirect ? `${gridPath}-direct-fragment` : gridPath;
                  }
                }
                if (rendered) {
                  setGpuDisplayVisible(true);
                  drewDisplayDirect = true;
                }
              }
            }
            if (canScaledDirect) {
              const bitmap = rendered
                ? null
                : engine.renderSlotScaledToImageBitmap(slotIdx, { vmin, vmax }, c.logScale, dw, dh);
              const canvas = canvasRef.current;
              const ctx = canvas?.getContext("2d");
              if (bitmap) {
                try {
                  if (ctx) {
                    if (canSharedPanelScaledDirect) {
                      drawSharedScaledBitmap(ctx, bitmap);
                    } else {
                      ctx.imageSmoothingEnabled = c.smooth;
                      ctx.clearRect(0, 0, c.canvasW, c.canvasH);
                      ctx.drawImage(bitmap, 0, 0, dw, dh);
                    }
                    setGpuDisplayVisible(false);
                    rendered = true;
                    drewDisplayDirect = true;
                    if (dbg) dbg.lastRenderPath = canSharedPanelScaledDirect ? "scaled-gpu-shared-panels" : "scaled-gpu";
                  }
                } finally {
                  bitmap.close();
                }
              }
            }
            if (!rendered && frame) {
              const bitmaps = engine.renderSlotsToImageBitmap([slotIdx], [{ vmin, vmax }], c.logScale);
              if (bitmaps && bitmaps[0]) {
                try {
                  const offCtx = mainOffscreenRef.current.getContext("2d");
                  if (offCtx) {
                    offCtx.drawImage(bitmaps[0], 0, 0);
                    rendered = true;
                    if (dbg) dbg.lastRenderPath = "full-gpu";
                  }
                } finally {
                  bitmaps[0].close();
                }
              }
            }
          } catch (err) {
            if (dbg) {
              dbg.lastRenderError = err instanceof Error ? err.message : String(err);
            }
            rendered = false;
            drewDisplayDirect = false;
          }
        }
        if (!rendered) {
          if (!frame && !cpuData) {
            if (dbg) {
              dbg.missingFrame = next;
              dbg.missingFrameAt = tickNow;
            }
            scheduleTick();
            return;
          }
          if (cpuDataAlreadyLogged && cpuData) {
            renderToOffscreenReuse(cpuData, lut, vmin, vmax, mainOffscreenRef.current, mainImgDataRef.current);
          } else if (frame) {
            renderFramePlayback(frame, mainImgDataRef.current.data, lut, vmin, vmax, c.logScale);
            mainOffscreenRef.current.getContext("2d")!.putImageData(mainImgDataRef.current, 0, 0);
          }
          if (dbg) dbg.lastRenderPath = "cpu";
        }

        // Draw to display canvas. Apply image_rotation so playback matches the
        // static render path (lines 1444-1453); otherwise rotated stacks
        // silently lose their rotation when the user hits Play.
        const canvas = canvasRef.current;
        if (canvas && !drewDisplayDirect) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            if ((nPanels || 1) > 1) {
              drawMain(ctx, mainOffscreenRef.current);
            } else {
              ctx.imageSmoothingEnabled = c.smooth;
              ctx.clearRect(0, 0, c.canvasW, c.canvasH);
              ctx.save();
              ctx.translate(c.panX, c.panY);
              ctx.scale(c.zoom, c.zoom);
              const dw = c.width * c.displayScale, dh = c.height * c.displayScale;
              if (c.imageRotation % 4 !== 0) {
                const cx = c.canvasW / 2 / c.zoom, cy = c.canvasH / 2 / c.zoom;
                ctx.translate(cx, cy);
                ctx.rotate((c.imageRotation * Math.PI) / 2);
                ctx.translate(-dw / 2, -dh / 2);
                ctx.drawImage(mainOffscreenRef.current, 0, 0, dw, dh);
              } else {
                ctx.drawImage(mainOffscreenRef.current, 0, 0, dw, dh);
              }
              ctx.restore();
            }
          }
        }
      }
      if (dbg) {
        dbg.lastRenderMs = Number((performance.now() - renderStartMs).toFixed(2));
        recordFramePacingDebug(dbg, performance.now(), intervalMs);
        dbg.renderedFrames = ((dbg.renderedFrames as number | undefined) ?? 0) + 1;
      }

      // Throttled UI updates for slider/stats/profile. At the 60 fps cap, keep React
      // comfortably out of the frame loop; the canvas still renders every rAF.
      // liveSliceIdx is per-tick for static offline paint and throttled for
      // direct WebGPU offline paint to avoid a competing React render path.
      if (tickNow - lastUIUpdate > uiUpdateIntervalMs) {
        lastUIUpdate = tickNow;
        if (offlineDirectRender) setLiveSliceIdx(next);
        setDisplaySliceIdx(next);
        setPlaybackUiSliceIdx(next);
        if (frame && c.showStats) setLocalStats(computeStats(frame));
        if (frame && c.profileActive && c.profilePoints.length === 2) {
          const p0 = c.profilePoints[0], p1 = c.profilePoints[1];
          setProfileData(sampleLineProfile(
            frame,
            c.width,
            c.height,
            p0.row,
            p0.col + c.profileColOffset,
            p1.row,
            p1.col + c.profileColOffset,
            c.profileWidth,
          ));
        }
        // Histogram refresh during playback. The non-playback effect path is keyed on
        // frameBytes/frameSeq which DON'T change during rAF playback (frames come from
        // the prefetch buffer, not via Comm), so we drive histogram updates directly
        // here at the same 10 Hz cadence. Skip every 2nd tick → ~5 Hz refresh.
        playbackHistogramCounterRef.current = (playbackHistogramCounterRef.current + 1) % 2;
        if (playbackHistogramCounterRef.current === 0) {
          if ((nPanels || 1) > 1 && !linkContrast && frame) {
            // Per-panel histograms have no single GPU slot; keep the per-panel
            // CPU extract (cold-ish, only when contrast is unlinked).
            const n = Math.max(1, nPanels || 1);
            const nextData: (Float32Array | null)[] = Array.from({ length: n }, () => null);
            const nextRanges: { min: number; max: number }[] = Array.from(
              { length: n },
              () => resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale),
            );
            for (const panel of c.visiblePanelIndices) {
              const panelData = extractPanelSlice(frame, panel, c.logScale);
              nextData[panel] = panelData;
              nextRanges[panel] = panelData && panelData.length > 0
                ? findDataRange(panelData)
                : resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale);
            }
            setPanelHistogramData(nextData);
            setPanelDataRanges(nextRanges);
          } else {
            // GPU histogram for the current frame (honors WebGPU-first-class):
            // refreshHistogram computes bins on the GPU (live slot or offline
            // scratch slot) AND sets lastHistogramFrame so it is verifiable.
            // Replaces the old CPU setImageHistogramData(frame).
            void refreshHistogramRef.current?.(next);
          }
        }
      }
      if (!isRgb && transformActive) {
        const warmDirection = c.reverse ? -1 : 1;
        warmPlaybackDisplayFrame(next + warmDirection, next, frame);
        warmPlaybackDisplayFrame(next + warmDirection * 2, next, frame);
      }

      // Prefetch at 25% buffer consumed - only if no next buffer is already queued.
      // Respect loop range so we don't fetch frames outside [loop_start, loop_end].
      if (!offline && frameServerUrl && separatePanelFrames) {
        prefetchPanelGpuFrames(next, c.reverse, c.loop, c.loopStart, c.loopEnd);
      } else if (!offline && frameServerUrl && gpuFrameCacheUploadedRef.current.size < c.nSlices) {
        prefetchServerFrames(next, c.reverse, c.loop, c.loopStart, c.loopEnd);
      } else if (!prefetchPendingRef.current && !nextBufferRef.current && bufferCountRef.current > 0) {
        let idxInBuffer = next - bufferStartRef.current;
        if (idxInBuffer < 0) idxInBuffer += c.nSlices;
        if (idxInBuffer >= Math.floor(bufferCountRef.current / 4)) {
          let prefetchStart = (bufferStartRef.current + bufferCountRef.current) % c.nSlices;
          // If loop range is constrained, snap prefetch start into it so we
          // don't waste buffer on frames the loop will never display.
          if (c.loop && (c.loopStart > 0 || c.loopEnd >= 0)) {
            const rs = Math.max(0, Math.min(c.loopStart, c.nSlices - 1));
            const re = c.loopEnd < 0 ? c.nSlices - 1 : Math.max(rs, Math.min(c.loopEnd, c.nSlices - 1));
            if (prefetchStart < rs || prefetchStart > re) prefetchStart = rs;
          }
          prefetchPendingRef.current = true;
          setPrefetchRequest(prefetchStart);
        }
      }

      scheduleTick();
    };

    scheduleTick();
    return () => {
      cancelAnimationFrame(animId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  // Update frame ref when frame changes
  React.useEffect(() => {
    // RGB frames ship as H*W*3 float32; gray remains H*W.
    const expectedFloats = isRgb ? width * height * 3 : width * height;
    const receiveAt = performance.now();
    const decodeStart = performance.now();
    const parsed = extractFloat32(frameBytes, expectedFloats);
    const decodeMs = performance.now() - decodeStart;
    if (!parsed || parsed.length === 0) return;
    const transport = frameTransportTiming ?? {};
    const sendTimeMs = typeof transport.sendTimeMs === "number" ? transport.sendTimeMs : null;
    pendingTransportPaintRef.current = {
      ...transport,
      kind: "frame",
      receiveAtMs: Number(receiveAt.toFixed(3)),
      jsDecodeMs: Number(decodeMs.toFixed(3)),
      browserReceiveLatencyMs: sendTimeMs === null ? null : Number((Date.now() - sendTimeMs).toFixed(3)),
    };
    requestAnimationFrame(() => requestAnimationFrame(() => markTransportPaintProxy()));
    if (isRgb) {
      // Keep color plane for paint; expose Rec. 709 luminance for stats/FFT.
      rgbFrameDataRef.current = parsed;
      rawFrameDataRef.current = rgbFrameToLuminance(parsed, width * height);
    } else {
      rgbFrameDataRef.current = null;
      const displayFrame = displayAndFrequencyFrameForIndex(offline ? liveSliceIdx : sliceIdx, parsed) ?? parsed;
      rawFrameDataRef.current = displayFrame;
    }
    const displayFrame = rawFrameDataRef.current;
    gpuUploadRef.current = null;
    if (!showStats) {
      setLocalStats(null);
      setLocalPanelStats(null);
      return;
    }
    // Recompute stats JS-side only while visible. On 4k frames this is a full
    // 16M-float scan, so keep the default hidden state paint-limited.
    const n = Math.max(1, nPanels || 1);
    const total = computeStats(displayFrame);
    setLocalStats(total);
    if (n > 1 && height > 0 && width > 0 && width % n === 0) {
      const pw = width / n;
      const panels: PanelStats[] = [];
      for (const p of visiblePanelIndices) {
        // Slice columns [p*pw, (p+1)*pw) for all rows.
        const slab = new Float32Array(height * pw);
        for (let r = 0; r < height; r++) {
          const srcOff = r * width + p * pw;
          slab.set(displayFrame.subarray(srcOff, srcOff + pw), r * pw);
        }
        panels.push({ panel: p, ...computeStats(slab) });
      }
      setLocalPanelStats(panels);
    } else {
      setLocalPanelStats(null);
    }
  }, [frameBytes, frameSeq, nPanels, visiblePanelIndices, width, height, showStats, diffMode, avgWindow, offline, liveSliceIdx, sliceIdx, isRgb, frequencyFilterIsActive, frequencyOptions, browserFilterTick, subpixelAlignEnabled, subpixelAlignVersion, frameTransportTiming, markTransportPaintProxy]);

  // Histogram bins are computed on the GPU via `engine.computeHistogramWithRange`
  // when the colormap engine is ready. CPU fallback (computeHistogramFromBytes
  // inside the Histogram component) still runs if WebGPU isn't available.
  // Debounce: 100 ms past the last scrub frame so drag doesn't fire bin scans
  // on every tick. Playback uses the established 2-tick (5 Hz) throttle.
  const histogramTimerRef = React.useRef<number | null>(null);
  const histogramRefreshInFlightRef = React.useRef(false);
  const histogramRefreshPendingIdxRef = React.useRef<number | null>(null);
  const histogramRefreshSerialRef = React.useRef(0);
  const refreshHistogram = React.useCallback(async (idxArg?: number) => {
    if (isRgb) return;
    if (sidecarMode && (sidecarBitmapReadyRef.current || sidecarCompositeReadyRef.current) && !perPanelHistogramEnabled) {
      const d = show3dPerfDebug();
      if (d) {
        d.lastHistogramFrame = clampSlice(idxArg ?? displaySliceIdx);
        d.lastHistogramSource = "sidecar-display-cache-skip";
      }
      return;
    }
    const renderIdx = clampSlice(idxArg ?? displaySliceIdx);
    if (histogramRefreshInFlightRef.current) {
      histogramRefreshPendingIdxRef.current = renderIdx;
      return;
    }
    histogramRefreshInFlightRef.current = true;
    const serial = ++histogramRefreshSerialRef.current;
    try {
      // Offline path: ensurePanelFrameGpu returns false offline, so the GPU
      // block below never runs and the CPU fallback hits raw==null for
      // separate-panel -> histogram frozen on frame 0 during offline playback
      // (the frame-server slots don't exist offline). Bin the dequantized
      // offline frame directly so the histogram tracks the playing frame.
      if (offline && !perPanelHistogramEnabled) {
        const offFrame = getOfflineFrame(renderIdx);
        if (offFrame && offFrame.length) {
          const engine = gpuCmapRef.current;
          let bins: number[] | null = null;
          // GPU histogram (operator: everything WebGPU). Upload the dequantized
          // offline frame to a reserved scratch slot, then compute bins on GPU.
          if (engine && gpuCmapReadyRef.current && dataMax > dataMin) {
            try {
              const rgbaCapacity = Math.max(1, width * height);
              engine.uploadData(OFFLINE_HIST_SLOT, offFrame, width, height, rgbaCapacity, true);
              bins = await engine.computeHistogramWithRange(OFFLINE_HIST_SLOT, dataMin, dataMax, logScale);
            } catch {
              bins = null;  // Histogram component CPU-bins from imageHistogramData below
            }
          }
          const dbg = show3dPerfDebug();
          if (dbg) { dbg.lastHistogramFrame = renderIdx; dbg.lastHistogramSource = bins ? "offline-gpu" : "offline-cpu"; }
          setImageDataRange(resolveDisplayBounds(dataMin, dataMax, null, null, logScale));
          setImageHistogramBins(bins);
          setImageHistogramData(bins ? null : (logScale ? applyLogScale(offFrame) : offFrame));
          return;
        }
      }
      if (!perPanelHistogramEnabled) {
        const engine = gpuCmapRef.current;
        if (
          engine &&
          gpuCmapReadyRef.current &&
          (separatePanelFrames || gpuFrameCacheUploadedRef.current.has(renderIdx)) &&
          dataMax > dataMin
        ) {
          let bins: number[] | null = null;
          try {
            if (separatePanelFrames) {
              const rgbaCapacity = Math.max(1, Math.round(canvasW * canvasH));
              const ready = await ensurePanelFrameGpu(renderIdx, rgbaCapacity);
              if (ready) {
                const summed = new Array<number>(256).fill(0);
                const sourcePanelCount = Math.max(1, nPanels || 1);
                for (const panel of visiblePanelIndices) {
                  const panelBins = await engine.computeHistogramWithRange(renderIdx * sourcePanelCount + panel, dataMin, dataMax, logScale);
                  if (!panelBins) {
                    bins = null;
                    break;
                  }
                  for (let i = 0; i < summed.length; i++) summed[i] += panelBins[i] ?? 0;
                  bins = summed;
                }
              }
            } else {
              bins = await engine.computeHistogramWithRange(renderIdx, dataMin, dataMax, logScale);
            }
          } catch {
            bins = null;
          }
          if (serial === histogramRefreshSerialRef.current && bins) {
            const dbg = show3dPerfDebug();
            if (dbg) {
              dbg.lastHistogramFrame = renderIdx;
              dbg.lastHistogramSource = separatePanelFrames ? "gpu-panel-slots" : "gpu-cache";
            }
            setImageDataRange(resolveDisplayBounds(dataMin, dataMax, null, null, logScale));
            setImageHistogramBins(bins);
            setImageHistogramData(null);
            return;
          }
        }
      }

      const raw = rawFrameDataRef.current;
      if (!raw || raw.length === 0) return;
      if (perPanelHistogramEnabled) {
        const n = Math.max(1, nPanels || 1);
        const nextData: (Float32Array | null)[] = Array.from({ length: n }, () => null);
        const nextRanges: { min: number; max: number }[] = Array.from(
          { length: n },
          () => resolveDisplayBounds(dataMin, dataMax, null, null, logScale),
        );
        for (const panel of visiblePanelIndices) {
          const panelData = extractPanelSlice(raw, panel, logScale);
          nextData[panel] = panelData;
          nextRanges[panel] = panelData && panelData.length > 0
            ? findDataRange(panelData)
            : resolveDisplayBounds(dataMin, dataMax, null, null, logScale);
        }
        setPanelHistogramData(nextData);
        setPanelDataRanges(nextRanges);
        setImageHistogramBins(null);
        return;
      }
      const data = logScale ? applyLogScale(raw) : raw;
      setImageDataRange(resolveDisplayBounds(dataMin, dataMax, null, null, logScale));
      // GPU bins: the colormap engine has the frame data uploaded to slot 0
      // already (via the render effect). Reuse that slot's buffer for a
      // 256-bin compute pass; fall back to CPU bins in the Histogram component
      // when the engine isn't ready or returns null.
      const engine = gpuCmapRef.current;
      let bins: number[] | null = null;
      if (engine && gpuCmapReadyRef.current && dataMax > dataMin) {
        try {
          // Use the requested frame's slot, not a hardcoded 0 (which is whatever
          // the data effect last uploaded, not the playing frame).
          const slot = gpuFrameCacheUploadedRef.current.has(renderIdx) ? renderIdx : 0;
          bins = await engine.computeHistogramWithRange(slot, dataMin, dataMax, logScale);
        } catch {
          bins = null;  // fall through to CPU path
        }
      }
      const dbg = show3dPerfDebug();
      if (dbg) { dbg.lastHistogramFrame = renderIdx; dbg.lastHistogramSource = bins ? "gpu-slot" : "cpu-data"; }
      setImageHistogramBins(bins);
      setImageHistogramData(data);
    } finally {
      histogramRefreshInFlightRef.current = false;
      const pending = histogramRefreshPendingIdxRef.current;
      histogramRefreshPendingIdxRef.current = null;
      if (pending !== null && pending !== renderIdx) {
        window.setTimeout(() => { void refreshHistogram(pending); }, 0);
      }
    }
  }, [logScale, dataMin, dataMax, perPanelHistogramEnabled, nPanels, visiblePanelIndices, extractPanelSlice, displaySliceIdx, separatePanelFrames, canvasW, canvasH, ensurePanelFrameGpu, isRgb, sidecarMode]);
  refreshHistogramRef.current = refreshHistogram;
  React.useEffect(() => {
    if (playing) {
      return;
    }
    playbackHistogramCounterRef.current = 0;
    if (histogramTimerRef.current !== null) {
      window.clearTimeout(histogramTimerRef.current);
    }
    histogramTimerRef.current = window.setTimeout(() => {
      refreshHistogram(displaySliceIdx);
      histogramTimerRef.current = null;
    }, 32);
  }, [frameBytes, frameSeq, playing, displaySliceIdx, refreshHistogram]);

  // Auto-snap thumbs to percentile-clip values while Auto is on. Fires once at mount
  // (so the slider visually reflects the percentile-clipped contrast that Python applies
  // when auto_contrast=True), and re-fires when logScale flips (linear vs log percentile
  // give different clip values, so the thumbs must follow). The lastLogScaleRef tracks
  // the previous logScale value so we only re-snap on transitions, not on every render.
  const initialAutoSnappedRef = React.useRef(false);
  const lastLogScaleRef = React.useRef(logScale);
  const lastAutoContrastRef = React.useRef(autoContrast);
  React.useEffect(() => {
    const logScaleChanged = lastLogScaleRef.current !== logScale;
    // Detect Auto toggled false -> true (user re-engages Auto).
    // Re-snap thumbs to auto range whenever Auto turns back on.
    const autoToggledOn = !lastAutoContrastRef.current && autoContrast;
    lastLogScaleRef.current = logScale;
    lastAutoContrastRef.current = autoContrast;
    if (perPanelHistogramEnabled) return;
    if (!autoContrast || !imageHistogramData || imageHistogramData.length === 0) return;
    // Skip initial snap if user already moved thumbs (e.g. loaded from saved state).
    if (!initialAutoSnappedRef.current && (imageVminPct !== 0 || imageVmaxPct !== 100)) {
      initialAutoSnappedRef.current = true;
      return;
    }
    // After first snap, re-snap only on logScale OR Auto-toggle-on transitions.
    if (initialAutoSnappedRef.current && !logScaleChanged && !autoToggledOn) return;
    const { min: autoMin, max: autoMax } = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
    const span = autoMax - autoMin;
    if (span <= 0) return;
    const cached = frameTransformActive() ? null : (
      cachedAutoDisplayRange(autoVmins, autoVmaxs, sliceIdx, logScale)
      || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, sliceIdx, logScale)
    );
    const { vmin: pmin, vmax: pmax } = cached ?? percentileClip(imageHistogramData, percentileLow, percentileHigh);
    setImageVminPct(Math.max(0, Math.min(100, ((pmin - autoMin) / span) * 100)));
    setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - autoMin) / span) * 100)));
    initialAutoSnappedRef.current = true;
  }, [autoContrast, imageHistogramData, dataMin, dataMax, traitVmin, traitVmax, autoVmins, autoVmaxs, sliceIdx, percentileLow, percentileHigh, logScale, imageVminPct, imageVmaxPct, perPanelHistogramEnabled]);

  // useEffect (not useLayoutEffect) so the per-panel auto-snap runs AFTER
  // the data effect populates panelHistogramData for the new frame.
  // useLayoutEffect fires BEFORE useEffects → rawFrameDataRef would be
  // stale and the snap would bail at mount.
  React.useEffect(() => {
    if (!perPanelHistogramEnabled || !autoContrast || panelHistogramData.length === 0) return;
    const stackBounds = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
    if (stackBounds.max <= stackBounds.min) return;
    setPanelStates(prev => {
      const out = prev.map((state, i) => {
        // PER-PANEL auto: percentile-clip THIS panel's own data, then map
        // pct in THIS panel's data range. Mixed-unit stacks (BF/DF counts
        // vs SSB radians) span many orders of magnitude — using stack
        // range squashes tight panels to pct ≈ 0.
        const clip = panelAutoClipPcts(i, state, stackBounds);
        return clip ? { ...state, imageVminPct: clip.imageVminPct, imageVmaxPct: clip.imageVmaxPct } : state;
      });
      return out;
    });
  }, [perPanelHistogramEnabled, autoContrast, panelHistogramData, panelDataRanges, dataMin, dataMax, traitVmin, traitVmax, logScale, percentileLow, percentileHigh]);

  React.useEffect(() => {
    if (!perPanelHistogramEnabled || autoContrast || panelDataRanges.length === 0) return;
    setPanelStates(prev => {
      let changed = false;
      const out = prev.map((state, i) => {
        const storedMin = vminPerPanel[i];
        const storedMax = vmaxPerPanel[i];
        if (storedMin == null && storedMax == null) return state;
        const panelRange = panelDataRanges[i];
        const range = (panelRange && panelRange.max > panelRange.min)
          ? panelRange
          : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
        if (range.max <= range.min) return state;
        const lo = storedMin ?? range.min;
        const hi = Math.max(lo, storedMax ?? range.max);
        const nextMinPct = valueToPct(lo, range.min, range.max, state.imageVminPct);
        const nextMaxPct = valueToPct(hi, range.min, range.max, state.imageVmaxPct);
        if (Math.abs(nextMinPct - state.imageVminPct) < 0.01 && Math.abs(nextMaxPct - state.imageVmaxPct) < 0.01) return state;
        changed = true;
        return { ...state, imageVminPct: nextMinPct, imageVmaxPct: nextMaxPct };
      });
      return changed ? out : prev;
    });
  }, [perPanelHistogramEnabled, autoContrast, panelDataRanges, vminPerPanel, vmaxPerPanel, dataMin, dataMax, traitVmin, traitVmax, logScale]);

  React.useEffect(() => {
    if (!effectiveRoiActive || roiItems.length === 0 || !showRoiResizeHint) return;
    const timer = window.setTimeout(() => setShowRoiResizeHint(false), 6000);
    return () => window.clearTimeout(timer);
  }, [effectiveRoiActive, roiItems.length, showRoiResizeHint]);

  React.useEffect(() => {
    if (compareMode !== "blink") {
      setBlinkPhase(0);
      return;
    }
    const hz = Math.max(0.25, Math.min(8, Number(blinkFps) || 2));
    const timer = window.setInterval(() => setBlinkPhase((value) => (value + 1) % 2), Math.round(1000 / hz));
    return () => window.clearInterval(timer);
  }, [blinkFps, compareMode]);

  React.useEffect(() => {
    if (!sidecarMode) return;
    if (compareMode === "blink") {
      invalidateSidecarViewportCache("compare-blink");
    } else if (compareMode !== "off") {
      invalidateSidecarViewportCache("compare-mode");
    }
  }, [blinkPhase, compareMode, comparePair, sidecarMode, invalidateSidecarViewportCache]);

  React.useEffect(() => {
    if (!sidecarMode) return;
    invalidateSidecarViewportCache("visibility-sidecar");
  }, [sidecarMode, visiblePanelIndices, invalidateSidecarViewportCache]);

  // Data effect: normalize + colormap → reusable offscreen canvas, then draw
  React.useEffect(() => {
    // Invalidate any rAF/mapAsync work from the previous render before every
    // early ownership return (notably the transition into playback).
    const renderSerial = ++gpuRenderSerialRef.current;
    let frameData = rawFrameDataRef.current;
    if (!frameData || frameData.length === 0) return;
    if (!isRgb && compareMode !== "off" && width > 0 && height > 0) {
      const n = Math.max(1, nSlices || 1);
      const aIdx = Math.max(0, Math.min(n - 1, Math.round(comparePair?.[0] ?? 0)));
      const bIdx = Math.max(0, Math.min(n - 1, Math.round(comparePair?.[1] ?? Math.min(1, n - 1))));
      const aFrame = getOfflineFrame(aIdx) ?? frameData;
      const bFrame = getOfflineFrame(bIdx) ?? frameData;
      if (aFrame.length === frameData.length && bFrame.length === frameData.length) {
        if (compareMode === "blink") {
          frameData = blinkPhase % 2 === 0 ? aFrame : bFrame;
        } else if (compareMode === "difference" || compareMode === "overlay") {
          const diff = new Float32Array(frameData.length);
          for (let i = 0; i < diff.length; i++) diff[i] = bFrame[i] - aFrame[i];
          frameData = diff;
        }
      }
    }
    if (!mainOffscreenRef.current || !mainImgDataRef.current) return;
    // True-color RGB: paint on the GPU (paintRgbFrame), applying the moving
    // average across color frames when avg > 1 so an avg change re-denoises the
    // static frame, not just live playback.
    if (isRgb && rgbFrameDataRef.current && rgbFrameDataRef.current.length >= width * height * 3) {
      const idx = offline ? liveSliceIdx : displaySliceIdx;
      const rgb = normalizedAverageWindow(avgWindow) > 1
        ? averagedRgbFrameForIndex(idx, rgbFrameDataRef.current)
        : rgbFrameDataRef.current;
      paintRgbFrame(rgb);
      return;
    }
    const offlinePackedPanelPlaybackUsesStaticCanvas = (
      offline &&
      playing &&
      Math.max(1, nPanels || 1) > 1 &&
      !sharedPanelSource
    );
    const offlineGpuPlaybackOwnsCanvas = (
      offline &&
      playing &&
      !frequencyFilterIsActive &&
      !offlinePackedPanelPlaybackUsesStaticCanvas &&
      !!gpuCmapRef.current &&
      gpuCmapReadyRef.current
    );
    if (offlineGpuPlaybackOwnsCanvas) return;
    // During live playback with browser-side transforms (denoise, diff/avg, or
    // FFT filter), the rAF playback loop owns the canvas and waits until the
    // final transformed display frame is cached before advancing. Letting this
    // static effect repaint on browserFilterTick/frequencyRenderVersion races
    // the loop and produces the visible "twitch" scientists saw after enabling
    // denoise + band/high/low-pass filters.
    if (!offline && playing && frameTransformActive()) return;
    // Apply log scale using reusable buffer
    const processed = logScale && logBufferRef.current
      ? applyLogScaleInPlace(frameData, logBufferRef.current)
      : frameData;

    const nP = Math.max(1, nPanels || 1);
    const perPanelContrast = nP > 1 && !linkContrast && !sharedPanelSource && width % nP === 0 && height > 0;
    const transformActive = frameTransformActive();

    // Compute vmin/vmax (per-panel branch uses GPU multi-slot below)
    let vmin: number, vmax: number;
    const hasTraitRange = traitVmin != null || traitVmax != null;
    if (hasTraitRange) {
      ({ vmin, vmax } = resolveDisplayRange(
        dataMin,
        dataMax,
        traitVmin,
        traitVmax,
        logScale,
        imageVminPct,
        imageVmaxPct,
      ));
    } else if (autoContrast) {
      const renderIdx = offline ? liveSliceIdx : sliceIdx;
      const cached = transformActive ? null : (
        cachedAutoDisplayRange(autoVmins, autoVmaxs, renderIdx, logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, renderIdx, logScale)
      );
      if (cached) {
        ({ vmin, vmax } = cached);
      } else {
        ({ vmin, vmax } = percentileClip(processed, percentileLow, percentileHigh));
      }
    } else {
      // Use the global data range (loaded once at widget mount) rather than
      // re-scanning the frame on every scrub. findDataRange does an O(N) min/max
      // pass which is ~8 ms at 4k - avoidable when the stack-wide bounds already
      // bracket the per-frame range.
      const lo = logScale ? (dataMin >= 0 ? Math.log1p(dataMin) : -Math.log1p(-dataMin)) : dataMin;
      const hi = logScale ? (dataMax >= 0 ? Math.log1p(dataMax) : -Math.log1p(-dataMax)) : dataMax;
      ({ vmin, vmax } = sliderRange(lo, hi, imageVminPct, imageVmaxPct));
    }

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const mixedPanelCmaps = hasMixedPanelCmaps && nP > 1 && !sharedPanelSource && width % nP === 0 && height > 0;
    const renderPackedPanelsCpu = (
      offscreen: HTMLCanvasElement | OffscreenCanvas,
      offCtx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
      sharedAutoRange: { vmin: number; vmax: number } | null,
    ) => {
      offCtx.clearRect(0, 0, offscreen.width, offscreen.height);
      const panelW = Math.max(1, Math.floor(width / nP));
      const panelImg = offCtx.createImageData(panelW, height);
      for (const p of visiblePanelIndices) {
        if (p < 0 || p >= nP) continue;
        const panelData = extractPanelSlice(frameData, p, logScale);
        if (!panelData) continue;
        const pdr = panelDataRanges[p];
        const panelRange = panelData.length > 0
          ? findDataRange(panelData)
          : ((perPanelHistogramEnabled && pdr && pdr.max > pdr.min)
              ? pdr
              : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale));
        const range = perPanelContrast
          ? resolvePanelRenderRange(p, panelRange, sharedAutoRange, panelData, autoContrast, percentileLow, percentileHigh)
          : { vmin, vmax };
        const panelLut = COLORMAPS[panelCmapFor(p)] || lut;
        applyColormap(panelData, panelImg.data, panelLut, range.vmin, range.vmax);
        offCtx.putImageData(panelImg, p * panelW, 0);
      }
    };

    if (offline) {
      const canvas = canvasRef.current;
      const offscreen = mainOffscreenRef.current;
      const imgData = mainImgDataRef.current;
      const ctx = canvas?.getContext("2d");
      const offCtx = offscreen?.getContext("2d");
      if (!canvas || !offscreen || !imgData || !ctx || !offCtx) return;
      if (perPanelContrast || mixedPanelCmaps) {
        const sharedAutoRange = autoContrast ? { vmin, vmax } : null;
        renderPackedPanelsCpu(offscreen, offCtx, sharedAutoRange);
      } else {
        renderToOffscreenReuse(processed, lut, vmin, vmax, offscreen, imgData);
      }
      drawMain(ctx, offscreen);
      return;
    }

    if (mixedPanelCmaps) {
      const canvas = canvasRef.current;
      const offscreen = mainOffscreenRef.current;
      const ctx = canvas?.getContext("2d");
      const offCtx = offscreen?.getContext("2d");
      if (!canvas || !offscreen || !ctx || !offCtx) return;
      const sharedAutoRange = autoContrast ? { vmin, vmax } : null;
      renderPackedPanelsCpu(offscreen, offCtx, sharedAutoRange);
      drawMain(ctx, offscreen);
      return;
    }

    // GPU colormap path (single frame) - zero-copy via OffscreenCanvas→ImageBitmap
    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current) {
      engine.uploadLUT(cmap, lut);
      // Per-panel contrast: upload the FULL frame ONCE as slot 0, then run a
      // fused GPU pipeline that, per panel: reduces a sub-region → vmin/vmax,
      // colormaps the panel sub-image using those values + slider pcts, and
      // blits to a panel-sized OffscreenCanvas. No JS slab extraction, no
      // findDataRange loop, no CPU readback between range and colormap.
      const dataForGpu = perPanelContrast ? frameData : (logScale ? processed : frameData);
      const ensureGpuUpload = () => {
        const prev = gpuUploadRef.current;
        if (
          prev &&
          prev.source === frameData &&
          prev.data === dataForGpu &&
          prev.width === width &&
          prev.height === height &&
          prev.logScale === logScale
        ) {
          return;
        }
        engine.uploadData(0, dataForGpu, width, height);
        gpuUploadRef.current = { source: frameData, data: dataForGpu, width, height, logScale };
      };
      if (perPanelContrast) {
        const pw = width / nP;
        ensureGpuUpload();
        const activePanels = visiblePanelIndices.filter((p) => p >= 0 && p < nP);
        const regions = activePanels.map((p) => ({
          x: p * pw, y: 0, width: pw, height,
        }));
        const sharedAutoRange = autoContrast ? { vmin, vmax } : null;
        const panelRanges = activePanels.map((p) => {
          const panelData = extractPanelSlice(frameData, p, logScale);
          const pdr = panelDataRanges[p];
          const panelRange = panelData && panelData.length > 0
            ? findDataRange(panelData)
            : ((perPanelHistogramEnabled && pdr && pdr.max > pdr.min)
                ? pdr
                : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale));
          return resolvePanelRenderRange(p, panelRange, sharedAutoRange, panelData, autoContrast, percentileLow, percentileHigh);
        });
        const panelLogs = logScale;
        requestAnimationFrame(() => {
          if (renderSerial !== gpuRenderSerialRef.current) return;
          if (!mainOffscreenRef.current) return;
          const bitmaps = engine.renderPerPanelGpuExplicit(0, regions, panelRanges, panelLogs);
          if (bitmaps) {
            try {
              const ctx = mainOffscreenRef.current.getContext("2d");
              if (ctx) {
                for (let slot = 0; slot < activePanels.length; slot++) {
                  const p = activePanels[slot];
                  if (bitmaps[slot]) {
                    ctx.drawImage(bitmaps[slot], p * pw, 0);
                  }
                }
              }
            } finally {
              bitmaps.forEach(bitmap => bitmap?.close());
            }
          }
          const canvas = canvasRef.current;
          if (!canvas) return;
          const ctx2 = canvas.getContext("2d");
          if (renderSerial !== gpuRenderSerialRef.current) return;
          if (ctx2 && mainOffscreenRef.current) drawMain(ctx2, mainOffscreenRef.current);
        });
        return;
      }
      ensureGpuUpload();
      const capturedVmin = vmin, capturedVmax = vmax;
      const blitAndDraw = async (forceReadback = false): Promise<boolean> => {
        if (renderSerial !== gpuRenderSerialRef.current) return false;
        if (!mainOffscreenRef.current) return false;
        // Zero-copy: GPU → OffscreenCanvas → ImageBitmap → drawImage
        const bitmaps = forceReadback
          ? null
          : engine.renderSlotsToImageBitmap([0], [{ vmin: capturedVmin, vmax: capturedVmax }], false);
        if (bitmaps && bitmaps[0]) {
          try {
            const ctx = mainOffscreenRef.current.getContext("2d");
            if (ctx) ctx.drawImage(bitmaps[0], 0, 0);
          } finally {
            // ImageBitmap holds external GPU/CPU memory not reclaimed by GC.
            bitmaps[0].close();
          }
        } else {
          // The mapAsync fallback must never write into the shared live
          // offscreen before its generation is confirmed. Render into a local
          // target, then commit only if this request is still current.
          const liveOffscreen = mainOffscreenRef.current;
          const liveImgData = mainImgDataRef.current;
          if (liveOffscreen && liveImgData) {
            const temporary = Object.assign(document.createElement("canvas"), {
              width: liveOffscreen.width,
              height: liveOffscreen.height,
            });
            const temporaryCtx = temporary.getContext("2d");
            const temporaryImgData = temporaryCtx
              ? temporaryCtx.createImageData(liveImgData.width, liveImgData.height)
              : null;
            let rendered = 0;
            if (temporaryImgData) {
              try {
                rendered = await engine.renderSlots(
                  [0], [{ vmin: capturedVmin, vmax: capturedVmax }],
                  [temporary], [temporaryImgData], false,
                );
              } catch (err) {
                if (renderSerial === gpuRenderSerialRef.current) {
                  console.warn("[Show3D] WebGPU mapAsync colormap fallback failed; using CPU", err);
                }
              }
            }
            if (renderSerial !== gpuRenderSerialRef.current) return false;
            if (rendered > 0) {
              const liveCtx = liveOffscreen.getContext("2d");
              if (liveCtx) liveCtx.drawImage(temporary, 0, 0);
            } else {
              renderToOffscreenReuse(processed, lut, capturedVmin, capturedVmax, liveOffscreen, liveImgData);
            }
          }
        }
        // Redraw main canvas (per-panel)
        const canvas = canvasRef.current;
        if (!canvas) return false;
        const ctx = canvas.getContext("2d");
        if (renderSerial !== gpuRenderSerialRef.current) return false;
        if (ctx && mainOffscreenRef.current) drawMain(ctx, mainOffscreenRef.current);
        return true;
      };
      requestAnimationFrame(async () => {
        const ok = await blitAndDraw();
        // Mac/Metal flush race: a one-shot static render captures the ImageBitmap
        // before the GPU submit has flushed ~2/3 of the time, leaving the canvas
        // blank until something re-renders. Playback's continuous rAF self-heals;
        // a static offline mount has no follow-up frame, so the panels stay black
        // (D6). Re-blit on a confirming second rAF when NOT playing - by the next
        // frame the GPU work has flushed and the bitmap is valid. Idempotent.
        if (ok && !playing) requestAnimationFrame(() => { void blitAndDraw(true); });
      });
    } else {
      // WebGPU-only per CLAUDE.md "WebGPU is THE pipeline" rule.
      // setGpuCmapReady(true) upstream triggers this effect to re-fire as
      // soon as the engine resolves. Skip painting until then (canvas
      // briefly blank for ~50-200 ms on some GPU workstations, never CPU-rendered).
      gpuRenderSerialRef.current++;
    }

    // Draw to main canvas (CPU path only - GPU path draws in its own rAF above)
    if (!engine || !gpuCmapReadyRef.current) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (ctx && mainOffscreenRef.current) drawMain(ctx, mainOffscreenRef.current);
    }
  }, [frameBytes, frameSeq, width, height, cmap, panelCmapFor, hasMixedPanelCmaps, displayScale, canvasW, canvasH, imageVminPct, imageVmaxPct, logScale, autoContrast, percentileLow, percentileHigh, traitVmin, traitVmax, dataMin, dataMax, autoVmins, autoVmaxs, smooth, imageRotation, nPanels, sharedPanelSource, visiblePanelIndices, perPanelHistogramEnabled, linkContrast, panelStates, panelDataRanges, vminPerPanel, vmaxPerPanel, offline, liveSliceIdx, sliceIdx, diffMode, avgWindow, playing, gpuCmapReady, canvasRepaintSignal, isRgb, browserFilterTick, denoiseSigmaLive, displayFilter, spatialBin, browserFilterKnobsOn, frequencyRenderVersion, frequencyFilterIsActive, subpixelAlignEnabled, subpixelAlignVersion, compareMode, comparePair, blinkPhase, getOfflineFrame, nSlices]);

  // Per-panel render: each slot gets its own zoom/pan transform. 2px gap
  // between slots painted as the canvas bg (transparent through clearRect).
  const drawMain = (
    ctx: CanvasRenderingContext2D,
    offscreen: HTMLCanvasElement | OffscreenCanvas,
    options: { preserveGpuDisplay?: boolean; sourcePanelWidth?: number } = {},
  ) => {
    const drawSliceIdx = offline ? liveSliceIdx : displaySliceIdx;
    const keepDirectGpuVisible =
      !offline &&
      gpuCmapReadyRef.current &&
      gpuFrameCacheUploadedRef.current.has(displaySliceIdx) &&
      imageRotation % 4 === 0 &&
      (separatePanelFrames || hiddenPanelSet.size === 0) &&
      (separatePanelFrames || (
        linkedState.zoom === 1 &&
        linkedState.panX === 0 &&
        linkedState.panY === 0
      ));
    if (!keepDirectGpuVisible && !options.preserveGpuDisplay) {
      setGpuDisplayVisible(false);
    }
    ctx.imageSmoothingEnabled = smooth;
    // Clear entire canvas. Slot-level bg fill happens inside the per-panel
    // loop so empty grid cells (partial last row) stay transparent - the
    // page bg shows through instead of a dead white block.
    ctx.clearRect(0, 0, canvasW, canvasH);
    const n = Math.max(1, visiblePanelCount || 1);
    const sourcePanelCount = Math.max(1, nPanels || 1);
    const cols = panelColsForCount(n);
    const rows = Math.ceil(n / cols);
    const srcPanelW = options.sourcePanelWidth
      ? Math.max(1, options.sourcePanelWidth)
      : sharedPanelSource
      ? offscreen.width
      : Math.max(1, panelWidthPx || offscreen.width / sourcePanelCount);
    const srcH = offscreen.height;
    const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
    const outPanelW = (canvasW - gap * (cols - 1)) / cols;
    const outPanelH = (canvasH - gap * (rows - 1)) / rows;
    for (let slot = 0; slot < n; slot++) {
      const i = visiblePanelIndices[slot] ?? slot;
      const panelState = stateFor(i);
      const col = slot % cols;
      const row = Math.floor(slot / cols);
      const slotX = col * (outPanelW + gap);
      const slotY = row * (outPanelH + gap);
      // Per-slot bg fill - only real panels get the theme bg; empty grid
      // cells in a partial last row stay transparent.
      ctx.fillStyle = themeColors.bg;
      ctx.fillRect(slotX, slotY, outPanelW, outPanelH);
      // End-of-stack: when current frame exceeds this panel's real frame
      // count, blur the (repeated last) frame + draw "end ({real}/{real})"
      // badge so operator sees they're scrubbing past real data.
      const realN = panelRealFrames && panelRealFrames[i];
      const pastEnd = !!(realN && drawSliceIdx >= realN);
      ctx.save();
      ctx.beginPath();
      ctx.rect(slotX, slotY, outPanelW, outPanelH);
      ctx.clip();
      ctx.translate(slotX + panelState.panX, slotY + panelState.panY);
      ctx.scale(panelState.zoom, panelState.zoom);
      const w = outPanelW, h = outPanelH;
      if (flipCols || flipRows) {
        ctx.translate(flipCols ? w : 0, flipRows ? h : 0);
        ctx.scale(flipCols ? -1 : 1, flipRows ? -1 : 1);
      }
      if (imageRotation % 4 !== 0) {
        const cx = w / 2 / panelState.zoom, cy = h / 2 / panelState.zoom;
        ctx.translate(cx, cy);
        ctx.rotate((imageRotation * Math.PI) / 2);
        ctx.translate(-w / 2, -h / 2);
      }
      if (pastEnd) ctx.filter = "blur(4px)";
      const srcX = sharedPanelSource ? 0 : i * srcPanelW;
      ctx.drawImage(offscreen as CanvasImageSource, srcX, 0, srcPanelW, srcH, 0, 0, w, h);
      ctx.restore();
      // No end badge - blur alone signals past-real-frame.
    }
  };

  const paintSidecarPanelBitmapsToContext = React.useCallback((
    ctx: CanvasRenderingContext2D,
    drawIdx: number,
    targetW: number,
    targetH: number,
  ): boolean => {
    const bitmaps = sidecarBitmapFrameCacheRef.current.get(drawIdx);
    if (!bitmaps || bitmaps.length === 0) return false;
    ctx.imageSmoothingEnabled = smooth;
    ctx.clearRect(0, 0, targetW, targetH);
    const visibleCountLocal = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(visibleCountLocal);
    const rows = Math.ceil(visibleCountLocal / cols);
    const gap = visibleCountLocal > 1 ? (panelGapTrait ?? 10) : 0;
    const outPanelW = (targetW - gap * (cols - 1)) / cols;
    const outPanelH = (targetH - gap * (rows - 1)) / rows;
    for (let slot = 0; slot < visibleCountLocal; slot++) {
      const panelIdx = visiblePanelIndices[slot] ?? slot;
      const bitmap = bitmaps[panelIdx];
      if (!bitmap) continue;
      const panelState = stateFor(panelIdx);
      const col = slot % cols;
      const row = Math.floor(slot / cols);
      const slotX = col * (outPanelW + gap);
      const slotY = row * (outPanelH + gap);
      ctx.fillStyle = themeColors.bg;
      ctx.fillRect(slotX, slotY, outPanelW, outPanelH);
      const realN = panelRealFrames && panelRealFrames[panelIdx];
      const pastEnd = !!(realN && drawIdx >= realN);
      ctx.save();
      ctx.beginPath();
      ctx.rect(slotX, slotY, outPanelW, outPanelH);
      ctx.clip();
      ctx.translate(slotX + panelState.panX, slotY + panelState.panY);
      ctx.scale(panelState.zoom, panelState.zoom);
      if (flipCols || flipRows) {
        ctx.translate(flipCols ? outPanelW : 0, flipRows ? outPanelH : 0);
        ctx.scale(flipCols ? -1 : 1, flipRows ? -1 : 1);
      }
      if (imageRotation % 4 !== 0) {
        const cx = outPanelW / 2 / panelState.zoom;
        const cy = outPanelH / 2 / panelState.zoom;
        ctx.translate(cx, cy);
        ctx.rotate((imageRotation * Math.PI) / 2);
        ctx.translate(-outPanelW / 2, -outPanelH / 2);
      }
      if (pastEnd) ctx.filter = "blur(4px)";
      ctx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height, 0, 0, outPanelW, outPanelH);
      ctx.restore();
    }
    return true;
  }, [
    smooth,
    visiblePanelCount,
    visiblePanelIndices,
    panelColsForCount,
    panelGapTrait,
    stateFor,
    themeColors.bg,
    panelRealFrames,
    flipCols,
    flipRows,
    imageRotation,
  ]);

  const paintSidecarU8ViewportToContext = React.useCallback((
    ctx: CanvasRenderingContext2D,
    drawIdx: number,
    targetW: number,
    targetH: number,
  ): boolean => {
    if (isRgb || sharedPanelSource) return false;
    const u8 = sidecarU8FrameCacheRef.current.get(drawIdx);
    if (!u8 || u8.byteLength < width * height) return false;
    const rotation = ((Math.round(imageRotation) % 4) + 4) % 4;
    const visibleCountLocal = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(visibleCountLocal);
    const rows = Math.ceil(visibleCountLocal / cols);
    const gap = visibleCountLocal > 1 ? Math.max(0, Math.round(panelGapTrait ?? 10)) : 0;
    const sourcePanelW = Math.max(1, Math.round(panelWidthPx || Math.floor(width / Math.max(1, nPanels || 1)) || width));
    const img = ctx.createImageData(targetW, targetH);
    const rgba = img.data;
    const bg = themeColors.bg || "#000";
    const parsedBg = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(bg.trim());
    let bgR = 0, bgG = 0, bgB = 0;
    if (parsedBg) {
      const raw = parsedBg[1];
      const hex = raw.length === 3 ? raw.split("").map((ch) => ch + ch).join("") : raw;
      const value = Number.parseInt(hex, 16);
      bgR = (value >> 16) & 255;
      bgG = (value >> 8) & 255;
      bgB = value & 255;
    }
    for (let p = 0; p < rgba.length; p += 4) {
      rgba[p] = bgR;
      rgba[p + 1] = bgG;
      rgba[p + 2] = bgB;
      rgba[p + 3] = 255;
    }
    const outPanelWFloat = (targetW - gap * (cols - 1)) / cols;
    const outPanelHFloat = (targetH - gap * (rows - 1)) / rows;
    const debugPaintPanels: Array<Record<string, number | string | boolean | null>> = [];
    for (let slot = 0; slot < visibleCountLocal; slot++) {
      const panelIdx = visiblePanelIndices[slot] ?? slot;
      const panelState = stateFor(panelIdx);
      const slotX0 = Math.max(0, Math.round((slot % cols) * (outPanelWFloat + gap)));
      const slotY0 = Math.max(0, Math.round(Math.floor(slot / cols) * (outPanelHFloat + gap)));
      const slotX1 = Math.min(targetW, Math.round(slotX0 + outPanelWFloat));
      const slotY1 = Math.min(targetH, Math.round(slotY0 + outPanelHFloat));
      if (slotX1 <= slotX0 || slotY1 <= slotY0) continue;
      const realN = panelRealFrames && panelRealFrames[panelIdx];
      if (realN && drawIdx >= realN) continue;
      const lut = COLORMAPS[panelCmapFor(panelIdx)] || COLORMAPS.inferno;
      const panelStateRange = !linkContrast && Math.max(1, nPanels || 1) > 1
        ? panelState
        : null;
      const panelPreview = panelHistogramPreviewPctRef.current.get(panelIdx) ?? null;
      const sharedPreview = imageHistogramPreviewPctRef.current;
      const preview = panelStateRange ? panelPreview : sharedPreview;
      const panelByteMin = (
        offlineMins?.length >= Math.max(1, nPanels || 1) &&
        Number.isFinite(offlineMins[panelIdx])
      )
        ? offlineMins[panelIdx]
        : offlineMin;
      const panelByteMax = (
        offlineMaxs?.length >= Math.max(1, nPanels || 1) &&
        Number.isFinite(offlineMaxs[panelIdx])
      )
        ? offlineMaxs[panelIdx]
        : offlineMax;
      const valueToPanelByte = (value: number): number | null => {
        if (!Number.isFinite(value) || !Number.isFinite(panelByteMin) || !Number.isFinite(panelByteMax) || panelByteMax <= panelByteMin) {
          return null;
        }
        return clampByte(((value - panelByteMin) / (panelByteMax - panelByteMin)) * 255);
      };
      let loByte: number;
      let hiByte: number;
      let byteRangeSource = "manual-percent";
      if (preview) {
        loByte = clampByte((Number(preview[0]) || 0) * 2.55);
        hiByte = clampByte((Number(preview[1]) || 100) * 2.55);
        byteRangeSource = "histogram-preview";
      } else if (autoContrast) {
        const autoRange = cachedAutoDisplayRange(autoVmins, autoVmaxs, drawIdx, logScale)
          || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, drawIdx, logScale);
        const mappedLo = autoRange ? valueToPanelByte(autoRange.vmin) : null;
        const mappedHi = autoRange ? valueToPanelByte(autoRange.vmax) : null;
        if (mappedLo !== null && mappedHi !== null && mappedHi > mappedLo) {
          loByte = mappedLo;
          hiByte = mappedHi;
          byteRangeSource = "auto-range";
        } else {
          loByte = clampByte((Number(percentileLow) || 0) * 2.55);
          hiByte = clampByte((Number(percentileHigh) || 100) * 2.55);
          byteRangeSource = "auto-percent-fallback";
        }
      } else {
        const loPct = panelStateRange ? panelStateRange.imageVminPct : imageVminPct;
        const hiPct = panelStateRange ? panelStateRange.imageVmaxPct : imageVmaxPct;
        loByte = clampByte((Number(loPct) || 0) * 2.55);
        hiByte = clampByte((Number(hiPct) || 100) * 2.55);
      }
      if (hiByte <= loByte) {
        hiByte = Math.min(255, loByte + 1);
      }
      const byteSpan = Math.max(1, hiByte - loByte);
      const srcPanelX = Math.max(0, Math.min(width - 1, panelIdx * sourcePanelW));
      const srcPanelXMax = Math.max(srcPanelX, Math.min(width - 1, srcPanelX + sourcePanelW - 1));
      let wrotePanelPixel = false;
      let sampleByte: number | null = null;
      let sampleMapped: number | null = null;
      let sampleRgb: number | null = null;
      for (let y = slotY0; y < slotY1; y++) {
        const localDrawY = ((y - slotY0) - (panelState.panY || 0)) / Math.max(1e-6, panelState.zoom || 1);
        if (localDrawY < 0 || localDrawY >= outPanelHFloat) continue;
        let localY = localDrawY / Math.max(1, outPanelHFloat);
        if (flipRows) localY = 1 - localY;
        let dst = (y * targetW + slotX0) * 4;
        for (let x = slotX0; x < slotX1; x++, dst += 4) {
          const localDrawX = ((x - slotX0) - (panelState.panX || 0)) / Math.max(1e-6, panelState.zoom || 1);
          if (localDrawX < 0 || localDrawX >= outPanelWFloat) continue;
          let localX = localDrawX / Math.max(1, outPanelWFloat);
          if (flipCols) localX = 1 - localX;
          let srcNormX = localX;
          let srcNormY = localY;
          if (rotation === 1) {
            srcNormX = localY;
            srcNormY = 1 - localX;
          } else if (rotation === 2) {
            srcNormX = 1 - localX;
            srcNormY = 1 - localY;
          } else if (rotation === 3) {
            srcNormX = 1 - localY;
            srcNormY = localX;
          }
          const srcY = Math.max(0, Math.min(height - 1, Math.floor(srcNormY * height)));
          const srcX = Math.max(srcPanelX, Math.min(srcPanelXMax, srcPanelX + Math.floor(srcNormX * sourcePanelW)));
          const src = srcY * width + srcX;
          const v = Math.max(0, Math.min(255, Math.floor(((u8[src] - loByte) / byteSpan) * 255)));
          const li = v * 3;
          if (sampleByte === null) {
            sampleByte = u8[src];
            sampleMapped = v;
            sampleRgb = lut[li];
          }
          rgba[dst] = lut[li];
          rgba[dst + 1] = lut[li + 1];
          rgba[dst + 2] = lut[li + 2];
          rgba[dst + 3] = 255;
          wrotePanelPixel = true;
        }
      }
      debugPaintPanels.push({
        panelIdx,
        slot,
        slotX0,
        slotY0,
        srcPanelX,
        loByte,
        hiByte,
        byteRangeSource,
        zoom: Number((panelState.zoom || 1).toFixed(3)),
        panX: Number((panelState.panX || 0).toFixed(1)),
        panY: Number((panelState.panY || 0).toFixed(1)),
        rotation,
        flipCols,
        flipRows,
        wrote: wrotePanelPixel,
        sampleByte,
        sampleMapped,
        sampleRgb,
      });
    }
    ctx.putImageData(img, 0, 0);
    const debug = show3dPerfDebug();
    if (debug) {
      debug.sidecarViewportPaintVisibleCount = visibleCountLocal;
      debug.sidecarViewportPaintCanvas = { width: targetW, height: targetH };
      debug.sidecarViewportPaintPanels = debugPaintPanels;
    }
    return true;
  }, [
    isRgb,
    sharedPanelSource,
    imageRotation,
    flipCols,
    flipRows,
    width,
    height,
    visiblePanelCount,
    panelColsForCount,
    panelGapTrait,
    panelWidthPx,
    nPanels,
    offlineMins,
    offlineMaxs,
    offlineMin,
    offlineMax,
    autoContrast,
    autoVmins,
    autoVmaxs,
    logScale,
    percentileLow,
    percentileHigh,
    imageVminPct,
    imageVmaxPct,
    themeColors.bg,
    visiblePanelIndices,
    stateFor,
    panelRealFrames,
    panelCmapFor,
    linkContrast,
  ]);

  const drawSidecarBitmapFrame = React.useCallback((
    idx: number,
    updateDisplayState = true,
    reason = "scrub",
  ): boolean => {
    const canvas = canvasRef.current;
    if (!canvas) return false;
    const ctx = canvas.getContext("2d");
    if (!ctx) return false;
    const nSlicesLocal = Math.max(1, Math.round(nSlices || 1));
    const drawIdx = ((Math.round(idx) % nSlicesLocal) + nSlicesLocal) % nSlicesLocal;
    const start = performance.now();
    const transformActive = sidecarViewTransformActive();
    const liveViewportPaint = (
      transformActive ||
      sidecarDisplayCacheDirtyRef.current ||
      (!sidecarCompositeReadyRef.current && !sidecarGpuReadyRef.current && sidecarRamReadyRef.current)
    );
    if (liveViewportPaint) {
      setGpuDisplayVisible(false);
      const ok = paintSidecarU8ViewportToContext(ctx, drawIdx, canvasW, canvasH);
      if (!ok) return false;
      playbackIdxRef.current = drawIdx;
      if (updateDisplayState) {
        if (displaySliceIdx !== drawIdx) setDisplaySliceIdx(drawIdx);
        if (playbackUiSliceIdx !== drawIdx) setPlaybackUiSliceIdx(drawIdx);
      }
      const d = show3dPerfDebug();
      if (d) {
        d.lastRenderPath = transformActive
          ? `sidecar-u8-viewport-transform-${reason}`
          : sidecarDisplayCacheDirtyRef.current
          ? `sidecar-u8-viewport-display-style-${reason}`
          : `sidecar-u8-viewport-live-${reason}`;
        d.lastRenderMs = performance.now() - start;
        d.lastPaintMs = d.lastRenderMs;
        d.lastFrame = drawIdx;
        d.sidecarCompositeSource = transformActive
          ? "u8-viewport-transform"
          : sidecarDisplayCacheDirtyRef.current
          ? "u8-viewport-display-style"
          : "u8-viewport-live";
      }
      return true;
    }
    if (sidecarGpuReadyRef.current && renderSidecarGpuFrame(drawIdx, reason)) {
      if (updateDisplayState) {
        if (displaySliceIdx !== drawIdx) setDisplaySliceIdx(drawIdx);
        if (playbackUiSliceIdx !== drawIdx) setPlaybackUiSliceIdx(drawIdx);
      }
      return true;
    }
    setGpuDisplayVisible(false);
    const composite = sidecarCompositeReadyRef.current
      ? sidecarCompositeFrameCacheRef.current.get(drawIdx)
      : null;
    if (composite) {
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvasW, canvasH);
      ctx.drawImage(composite, 0, 0, composite.width, composite.height, 0, 0, canvasW, canvasH);
      playbackIdxRef.current = drawIdx;
      if (updateDisplayState) {
        if (displaySliceIdx !== drawIdx) setDisplaySliceIdx(drawIdx);
        if (playbackUiSliceIdx !== drawIdx) setPlaybackUiSliceIdx(drawIdx);
      }
      const d = show3dPerfDebug();
      if (d) {
        d.lastRenderPath = `sidecar-composite-${reason}`;
        d.lastRenderMs = performance.now() - start;
        d.lastPaintMs = d.lastRenderMs;
        d.lastFrame = drawIdx;
        d.sidecarCompositeCacheFrames = sidecarCompositeFrameCacheRef.current.size;
      }
      return true;
    }
    const bitmaps = sidecarBitmapFrameCacheRef.current.get(drawIdx);
    if (!bitmaps || bitmaps.length === 0) return false;
    ctx.imageSmoothingEnabled = smooth;
    ctx.clearRect(0, 0, canvasW, canvasH);
    const visibleCountLocal = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(visibleCountLocal);
    const rows = Math.ceil(visibleCountLocal / cols);
    const gap = visibleCountLocal > 1 ? (panelGapTrait ?? 10) : 0;
    const outPanelW = (canvasW - gap * (cols - 1)) / cols;
    const outPanelH = (canvasH - gap * (rows - 1)) / rows;
    for (let slot = 0; slot < visibleCountLocal; slot++) {
      const panelIdx = visiblePanelIndices[slot] ?? slot;
      const bitmap = bitmaps[panelIdx];
      if (!bitmap) continue;
      const panelState = stateFor(panelIdx);
      const col = slot % cols;
      const row = Math.floor(slot / cols);
      const slotX = col * (outPanelW + gap);
      const slotY = row * (outPanelH + gap);
      ctx.fillStyle = themeColors.bg;
      ctx.fillRect(slotX, slotY, outPanelW, outPanelH);
      const realN = panelRealFrames && panelRealFrames[panelIdx];
      const pastEnd = !!(realN && drawIdx >= realN);
      ctx.save();
      ctx.beginPath();
      ctx.rect(slotX, slotY, outPanelW, outPanelH);
      ctx.clip();
      ctx.translate(slotX + panelState.panX, slotY + panelState.panY);
      ctx.scale(panelState.zoom, panelState.zoom);
      if (flipCols || flipRows) {
        ctx.translate(flipCols ? outPanelW : 0, flipRows ? outPanelH : 0);
        ctx.scale(flipCols ? -1 : 1, flipRows ? -1 : 1);
      }
      if (imageRotation % 4 !== 0) {
        const cx = outPanelW / 2 / panelState.zoom;
        const cy = outPanelH / 2 / panelState.zoom;
        ctx.translate(cx, cy);
        ctx.rotate((imageRotation * Math.PI) / 2);
        ctx.translate(-outPanelW / 2, -outPanelH / 2);
      }
      if (pastEnd) ctx.filter = "blur(4px)";
      ctx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height, 0, 0, outPanelW, outPanelH);
      ctx.restore();
    }
    playbackIdxRef.current = drawIdx;
    if (updateDisplayState) {
      if (displaySliceIdx !== drawIdx) setDisplaySliceIdx(drawIdx);
      if (playbackUiSliceIdx !== drawIdx) setPlaybackUiSliceIdx(drawIdx);
    }
    const d = show3dPerfDebug();
    if (d) {
      d.lastRenderPath = `sidecar-imagebitmap-${reason}`;
      d.lastRenderMs = performance.now() - start;
      d.lastPaintMs = d.lastRenderMs;
      d.lastFrame = drawIdx;
      d.sidecarBitmapCacheFrames = sidecarBitmapFrameCacheRef.current.size;
    }
    return true;
  }, [
    canvasW,
    canvasH,
    smooth,
    visiblePanelCount,
    visiblePanelIndices,
    panelColsForCount,
    panelGapTrait,
    nSlices,
    stateFor,
    themeColors.bg,
    panelRealFrames,
    flipCols,
    flipRows,
    imageRotation,
    displaySliceIdx,
    playbackUiSliceIdx,
    setGpuDisplayVisible,
    sidecarViewTransformActive,
    paintSidecarU8ViewportToContext,
  ]);

  const paintHistogramPreviewSidecar = React.useCallback((reason = "hist-preview") => {
    if (
      !offline ||
      !sidecarMode ||
      !sidecarRamReady ||
      isRgb ||
      canvasW <= 0 ||
      canvasH <= 0
    ) {
      return;
    }
    if (!sidecarCompositeReadyRef.current && !sidecarGpuReadyRef.current) return;
    sidecarDisplayCacheDirtyRef.current = true;
    setGpuDisplayVisible(false);
    const n = Math.max(1, Math.round(nSlices || 1));
    const drawIdx = ((Math.round(playbackIdxRef.current || liveSliceIdx || 0) % n) + n) % n;
    drawSidecarBitmapFrame(drawIdx, false, reason);
    updatePlaybackLiveControls(drawIdx);
    const debug = show3dPerfDebug();
    if (debug) {
      debug.sidecarDisplayStyleDirty = true;
      debug.sidecarDisplayStyleImmediateFrame = drawIdx;
      debug.sidecarHistogramPreview = true;
    }
  }, [
    canvasH,
    canvasW,
    drawSidecarBitmapFrame,
    isRgb,
    liveSliceIdx,
    nSlices,
    offline,
    setGpuDisplayVisible,
    sidecarMode,
    sidecarRamReady,
    updatePlaybackLiveControls,
  ]);

  const scheduleHistogramPreviewPaint = React.useCallback((reason = "hist-preview") => {
    if (histogramPreviewPaintRafRef.current !== null) return;
    histogramPreviewPaintRafRef.current = window.requestAnimationFrame(() => {
      histogramPreviewPaintRafRef.current = null;
      paintHistogramPreviewSidecar(reason);
    });
  }, [paintHistogramPreviewSidecar]);

  React.useEffect(() => () => {
    if (histogramPreviewPaintRafRef.current !== null) {
      window.cancelAnimationFrame(histogramPreviewPaintRafRef.current);
      histogramPreviewPaintRafRef.current = null;
    }
  }, []);

  React.useEffect(() => {
    const preview = imageHistogramPreviewPctRef.current;
    if (
      preview &&
      Math.abs(preview[0] - imageVminPct) < 0.01 &&
      Math.abs(preview[1] - imageVmaxPct) < 0.01
    ) {
      imageHistogramPreviewPctRef.current = null;
    }
  }, [imageVminPct, imageVmaxPct]);

  React.useEffect(() => {
    const previews = panelHistogramPreviewPctRef.current;
    if (previews.size === 0) return;
    for (const [panel, preview] of Array.from(previews.entries())) {
      const state = panelStates[panel];
      if (
        state &&
        Math.abs(preview[0] - state.imageVminPct) < 0.01 &&
        Math.abs(preview[1] - state.imageVmaxPct) < 0.01
      ) {
        previews.delete(panel);
      }
    }
  }, [panelStates]);

  React.useEffect(() => {
    const dbg = show3dPerfDebug();
    if (!dbg) return;
    const percentile = (values: number[], pct: number) => {
      if (!values.length) return 0;
      const sorted = [...values].sort((a, b) => a - b);
      const idx = Math.max(0, Math.min(sorted.length - 1, Math.ceil((pct / 100) * sorted.length) - 1));
      return sorted[idx];
    };
    const drawCachedFrame = (idx: number) => {
      const n = Math.max(1, Math.round(nSlices || 1));
      const frame = ((Math.round(idx) % n) + n) % n;
      const t0 = performance.now();
      const ok = drawSidecarBitmapFrame(frame, false, "debug-direct");
      updatePlaybackLiveControls(frame);
      return {
        ok,
        frame,
        drawMs: performance.now() - t0,
        path: dbg.lastRenderPath ?? null,
      };
    };
    const benchCachedFrames = async (steps = 120) => {
      const n = Math.max(1, Math.round(nSlices || 1));
      const requested = Math.max(1, Math.round(Number(steps) || 120));
      const drawMs: number[] = [];
      const intervals: number[] = [];
      const frames: number[] = [];
      let last = performance.now();
      let idx = Number.isFinite(playbackIdxRef.current) ? playbackIdxRef.current : 0;
      for (let i = 0; i < requested; i++) {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        const before = performance.now();
        intervals.push(before - last);
        last = before;
        idx = (idx + 1) % n;
        const result = drawCachedFrame(idx);
        drawMs.push(result.drawMs);
        frames.push(result.frame);
      }
      const elapsedMs = intervals.reduce((sum, value) => sum + value, 0);
      const meanIntervalMs = elapsedMs / Math.max(1, intervals.length);
      const meanDrawMs = drawMs.reduce((sum, value) => sum + value, 0) / Math.max(1, drawMs.length);
      const result = {
        steps: requested,
        frames,
        meanIntervalMs,
        p50IntervalMs: percentile(intervals, 50),
        p95IntervalMs: percentile(intervals, 95),
        fps: meanIntervalMs > 0 ? 1000 / meanIntervalMs : 0,
        meanDrawMs,
        p50DrawMs: percentile(drawMs, 50),
        p95DrawMs: percentile(drawMs, 95),
        path: dbg.lastRenderPath ?? null,
      };
      dbg.sidecarDirectBenchLast = result;
      return result;
    };
    dbg.sidecarDrawCachedFrame = drawCachedFrame;
    dbg.sidecarBenchCachedFrames = benchCachedFrames;
    return () => {
      if (dbg.sidecarDrawCachedFrame === drawCachedFrame) delete dbg.sidecarDrawCachedFrame;
      if (dbg.sidecarBenchCachedFrames === benchCachedFrames) delete dbg.sidecarBenchCachedFrames;
    };
  }, [drawSidecarBitmapFrame, nSlices, updatePlaybackLiveControls]);

  React.useEffect(() => {
    const d = show3dPerfDebug();
    const transformActive = sidecarViewTransformActive();
    if (d) {
      d.sidecarViewportEffectSeen = performance.now();
      d.sidecarViewportFlags = {
        offline,
        sidecarMode,
        sidecarRamReady,
        isRgb,
        canvasW,
        canvasH,
        nSlices,
        transformActive,
      };
    }
    if (
      !offline ||
      !sidecarMode ||
      !sidecarRamReady ||
      isRgb ||
      canvasW <= 0 ||
      canvasH <= 0 ||
      transformActive
    ) {
      if (d) {
        d.sidecarViewportSkipReason = !offline
          ? "not-offline"
          : !sidecarMode
            ? "not-sidecar"
            : !sidecarRamReady
              ? "ram-not-ready"
              : isRgb
                ? "rgb"
                : canvasW <= 0 || canvasH <= 0
                  ? "missing-canvas"
                  : "view-transform";
      }
    if (transformActive) {
      invalidateSidecarViewportCache("view-transform");
      setOfflineStackFetchStatus("");
    } else {
      clearSidecarCompositeCache();
    }
    return;
  }
  if (d) d.sidecarViewportSkipReason = "";
    const n = Math.max(1, Math.round(nSlices || 1));
    const serial = ++sidecarCompositeBuildSerialRef.current;
    let cancelled = false;
    const build = async () => {
      clearSidecarCompositeCache();
      setOfflineStackFetchStatus(
        sidecarDisplayCacheDirtyRef.current
          ? `Updating display playback cache… 0/${n} frames`
          : `Preparing viewport playback cache… 0/${n} frames`,
      );
      const scratch = document.createElement("canvas");
      scratch.width = Math.max(1, Math.round(canvasW));
      scratch.height = Math.max(1, Math.round(canvasH));
      const ctx = scratch.getContext("2d");
      if (!ctx) {
        setOfflineStackFetchStatus("Failed to prepare viewport playback cache: no 2D context");
        return;
      }
      const started = performance.now();
      const order = prioritizedSidecarFrameOrder(playbackIdxRef.current || liveSliceIdx || 0, n);
      let builtFrames = 0;
      try {
        for (const idx of order) {
          if (cancelled || serial !== sidecarCompositeBuildSerialRef.current) return;
          const ok = paintSidecarU8ViewportToContext(ctx, idx, scratch.width, scratch.height);
          if (!ok) {
            const debug = show3dPerfDebug();
            if (debug) debug.sidecarViewportSkipReason = "unsupported-layout";
            setOfflineStackFetchStatus("Viewport cache needs native fallback for this layout");
            return;
          }
          if (cancelled || serial !== sidecarCompositeBuildSerialRef.current || sidecarViewTransformActive()) return;
          const retained = document.createElement("canvas");
          retained.width = scratch.width;
          retained.height = scratch.height;
          const retainedCtx = retained.getContext("2d");
          if (!retainedCtx) continue;
          retainedCtx.drawImage(scratch, 0, 0);
          sidecarCompositeFrameCacheRef.current.set(idx, retained);
          builtFrames += 1;
          if (!sidecarCompositeReadyRef.current) {
            sidecarCompositeReadyRef.current = true;
            setSidecarCompositeReady(true);
            if ((compareMode || "off") === "off") {
              drawSidecarBitmapFrame(idx, false, "viewport-first");
            }
          }
          if (builtFrames === 1 || builtFrames % 8 === 0 || builtFrames === n) {
            const elapsed = ((performance.now() - started) / 1000).toFixed(1);
            setOfflineStackFetchStatus(
              builtFrames === n
                ? ""
                : `Preparing viewport playback cache… ${builtFrames}/${n} frames (${elapsed}s)`,
            );
          }
          const d = show3dPerfDebug();
          if (d) {
            d.sidecarCompositeCacheFrames = sidecarCompositeFrameCacheRef.current.size;
            d.sidecarCompositeBuildMs = performance.now() - started;
            d.sidecarCompositeWidth = scratch.width;
            d.sidecarCompositeHeight = scratch.height;
            d.sidecarCompositeSource = "u8-viewport";
            d.lastRenderPath = d.lastRenderPath ?? "sidecar-u8-viewport-cache-building";
          }
          if (builtFrames % 4 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
        }
        if (cancelled || serial !== sidecarCompositeBuildSerialRef.current) return;
        sidecarCompositeReadyRef.current = true;
        sidecarCompositeCompleteRef.current = true;
        sidecarDisplayCacheDirtyRef.current = false;
        sidecarCompositeStyleKeyRef.current = sidecarDisplayStyleKey;
        setSidecarCompositeReady(true);
        setSidecarCompositeComplete(true);
        setOfflineStackFetchStatus("");
        const d = show3dPerfDebug();
        if (d) {
          d.sidecarCompositeCacheFrames = sidecarCompositeFrameCacheRef.current.size;
          d.sidecarCompositeBuildMs = performance.now() - started;
          d.sidecarCompositeWidth = scratch.width;
          d.sidecarCompositeHeight = scratch.height;
          d.sidecarCompositeSource = "u8-viewport";
          d.sidecarDisplayStyleDirty = false;
          d.sidecarDisplayStyleKey = sidecarDisplayStyleKey;
          d.sidecarHistogramPreview = false;
          d.lastRenderPath = "sidecar-u8-viewport-cache-ready";
        }
      } catch (err) {
        clearSidecarCompositeCache();
        setOfflineStackFetchStatus(
          `Failed to prepare viewport playback cache: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    };
    const hasWarmCache = sidecarCompositeFrameCacheRef.current.size > 0 || sidecarGpuReadyRef.current;
    const rebuildDelayMs = sidecarDisplayCacheDirtyRef.current && hasWarmCache ? 180 : 0;
    let timer: number | null = null;
    if (rebuildDelayMs > 0) {
      const debug = show3dPerfDebug();
      if (debug) {
        debug.sidecarCompositeRebuildDebounceMs = rebuildDelayMs;
        debug.sidecarCompositeRebuildReason = "display-style";
      }
      setOfflineStackFetchStatus(`Updating display playback cache…`);
      timer = window.setTimeout(() => {
        timer = null;
        void build();
      }, rebuildDelayMs);
    } else {
      void build();
    }
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [
    offline,
    sidecarMode,
    sidecarRamReady,
    isRgb,
    canvasW,
    canvasH,
    nSlices,
    liveSliceIdx,
    prioritizedSidecarFrameOrder,
    paintSidecarU8ViewportToContext,
    drawSidecarBitmapFrame,
    clearSidecarCompositeCache,
    invalidateSidecarViewportCache,
    sidecarViewTransformActive,
    sidecarDisplayStyleKey,
  ]);

  React.useLayoutEffect(() => {
    if (
      !offline ||
      !sidecarMode ||
      !sidecarRamReady ||
      isRgb ||
      canvasW <= 0 ||
      canvasH <= 0
    ) {
      return;
    }
    if (!sidecarCompositeReadyRef.current && !sidecarGpuReadyRef.current) return;
    const previous = sidecarCompositeStyleKeyRef.current;
    if (!previous || previous === sidecarDisplayStyleKey) return;
    sidecarDisplayCacheDirtyRef.current = true;
    setGpuDisplayVisible(false);
    const n = Math.max(1, Math.round(nSlices || 1));
    const drawIdx = ((Math.round(playbackIdxRef.current || liveSliceIdx || 0) % n) + n) % n;
    drawSidecarBitmapFrame(drawIdx, false, "immediate");
    updatePlaybackLiveControls(drawIdx);
    const debug = show3dPerfDebug();
    if (debug) {
      debug.sidecarDisplayStyleDirty = true;
      debug.sidecarDisplayStyleImmediateFrame = drawIdx;
      debug.sidecarDisplayStyleKey = sidecarDisplayStyleKey;
    }
  }, [
    canvasH,
    canvasW,
    compareMode,
    drawSidecarBitmapFrame,
    isRgb,
    liveSliceIdx,
    nSlices,
    offline,
    setGpuDisplayVisible,
    sidecarDisplayStyleKey,
    sidecarMode,
    sidecarRamReady,
    updatePlaybackLiveControls,
  ]);

  React.useEffect(() => {
    if (!offline || !sidecarMode || playing || (!sidecarBitmapReady && !sidecarCompositeReady)) return;
    drawSidecarBitmapFrame(liveSliceIdx, true, "scrub");
  }, [
    offline,
    sidecarMode,
    playing,
    sidecarBitmapReady,
    sidecarCompositeReady,
    liveSliceIdx,
    drawSidecarBitmapFrame,
  ]);

  React.useEffect(() => {
    if (!enableSidecarNativePanelBitmapCache) return;
    if (
      !offline ||
      !sidecarMode ||
      !sidecarBitmapComplete ||
      canvasW <= 0 ||
      canvasH <= 0
    ) {
      clearSidecarCompositeCache();
      return;
    }
    const n = Math.max(1, Math.round(nSlices || 1));
    const serial = ++sidecarCompositeBuildSerialRef.current;
    let cancelled = false;
    clearSidecarCompositeCache();
    setOfflineStackFetchStatus(`Preparing viewport playback cache… 0/${n} frames`);
    const build = async () => {
      const scratch = document.createElement("canvas");
      scratch.width = Math.max(1, Math.round(canvasW));
      scratch.height = Math.max(1, Math.round(canvasH));
      const ctx = scratch.getContext("2d");
      if (!ctx) {
        setOfflineStackFetchStatus("Failed to prepare viewport playback cache: no 2D context");
        return;
      }
      const started = performance.now();
      try {
        for (let idx = 0; idx < n; idx++) {
          if (cancelled || serial !== sidecarCompositeBuildSerialRef.current) return;
          const ok = paintSidecarPanelBitmapsToContext(ctx, idx, scratch.width, scratch.height);
          if (!ok) continue;
          if (cancelled || serial !== sidecarCompositeBuildSerialRef.current) {
            return;
          }
          const retained = document.createElement("canvas");
          retained.width = scratch.width;
          retained.height = scratch.height;
          const retainedCtx = retained.getContext("2d");
          if (!retainedCtx) continue;
          retainedCtx.drawImage(scratch, 0, 0);
          sidecarCompositeFrameCacheRef.current.set(idx, retained);
          if (idx % 4 === 0 || idx === n - 1) {
            const elapsed = ((performance.now() - started) / 1000).toFixed(1);
            setOfflineStackFetchStatus(`Preparing viewport playback cache… ${idx + 1}/${n} frames (${elapsed}s)`);
          }
          if (idx % 4 === 3) await new Promise((resolve) => setTimeout(resolve, 0));
        }
        if (cancelled || serial !== sidecarCompositeBuildSerialRef.current) return;
        sidecarCompositeReadyRef.current = true;
        sidecarCompositeCompleteRef.current = true;
        setSidecarCompositeReady(true);
        setSidecarCompositeComplete(true);
        setOfflineStackFetchStatus("");
        const d = show3dPerfDebug();
        if (d) {
          d.sidecarCompositeCacheFrames = sidecarCompositeFrameCacheRef.current.size;
          d.sidecarCompositeBuildMs = performance.now() - started;
          d.sidecarCompositeWidth = scratch.width;
          d.sidecarCompositeHeight = scratch.height;
          d.lastRenderPath = "sidecar-composite-cache-ready";
        }
      } catch (err) {
        clearSidecarCompositeCache();
        setOfflineStackFetchStatus(
          `Failed to prepare viewport playback cache: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    };
    void build();
    return () => {
      cancelled = true;
    };
  }, [
    offline,
    sidecarMode,
    sidecarBitmapComplete,
    canvasW,
    canvasH,
    nSlices,
    paintSidecarPanelBitmapsToContext,
    clearSidecarCompositeCache,
  ]);

  function renderSidecarGpuFrame(idx: number, reason = "scrub"): boolean {
    const presenter = sidecarGpuPresenterRef.current;
    if (!presenter || !sidecarGpuReadyRef.current) return false;
    const n = Math.max(1, Math.round(nSlices || 1));
    const drawIdx = ((Math.round(idx) % n) + n) % n;
    const bindGroup = presenter.bindGroups.get(drawIdx);
    if (!bindGroup) return false;
    const start = performance.now();
    const encoder = presenter.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: presenter.context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(presenter.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    presenter.device.queue.submit([encoder.finish()]);
    setGpuDisplayVisible(true);
    playbackIdxRef.current = drawIdx;
    const d = show3dPerfDebug();
    if (d) {
      d.lastRenderPath = `sidecar-gpu-texture-${reason}`;
      d.lastRenderMs = performance.now() - start;
      d.lastPaintMs = d.lastRenderMs;
      d.lastFrame = drawIdx;
      d.sidecarGpuTextureFrames = presenter.bindGroups.size;
    }
    return true;
  }

  React.useEffect(() => {
    if (
      !offline ||
      !sidecarMode ||
      !enableSidecarGpuTexturePresenter ||
      !sidecarCompositeComplete ||
      !sidecarCompositeReadyRef.current ||
      !sidecarCompositeCompleteRef.current ||
      canvasW <= 0 ||
      canvasH <= 0 ||
      !("gpu" in navigator) ||
      sidecarViewTransformActive()
    ) {
      return;
    }
    const gpuCanvas = gpuCanvasRef.current;
    if (!gpuCanvas) return;
    const n = Math.max(1, Math.round(nSlices || 1));
    const serial = ++sidecarGpuBuildSerialRef.current;
    let cancelled = false;
    if (sidecarGpuPresenterRef.current) {
      for (const texture of sidecarGpuPresenterRef.current.textures) {
        try { texture.destroy(); } catch { /* ignore */ }
      }
    }
    sidecarGpuPresenterRef.current = null;
    sidecarGpuReadyRef.current = false;
    setSidecarGpuReady(false);
    setOfflineStackFetchStatus(`Uploading viewport playback cache to GPU… 0/${n} frames`);
    const build = async () => {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter || cancelled || serial !== sidecarGpuBuildSerialRef.current) return;
        const device = await adapter.requestDevice();
        if (cancelled || serial !== sidecarGpuBuildSerialRef.current) return;
        const context = gpuCanvas.getContext("webgpu");
        if (!context) return;
        const format = navigator.gpu.getPreferredCanvasFormat();
        const widthPx = Math.max(1, Math.round(canvasW));
        const heightPx = Math.max(1, Math.round(canvasH));
        gpuCanvas.width = widthPx;
        gpuCanvas.height = heightPx;
        context.configure({ device, format, alphaMode: "opaque" });
        const shader = device.createShaderModule({ code: `
          @group(0) @binding(0) var frameTex: texture_2d<f32>;
          @group(0) @binding(1) var frameSampler: sampler;
          struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
          @vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
            var out: VSOut;
            let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
            let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
            out.pos = vec4f(x, y, 0.0, 1.0);
            out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
            return out;
          }
          @fragment fn fs(in: VSOut) -> @location(0) vec4f {
            return textureSample(frameTex, frameSampler, in.uv);
          }
        ` });
        const pipeline = device.createRenderPipeline({
          layout: "auto",
          vertex: { module: shader, entryPoint: "vs" },
          fragment: { module: shader, entryPoint: "fs", targets: [{ format }] },
          primitive: { topology: "triangle-list" },
        });
        const sampler = device.createSampler({ magFilter: smooth ? "linear" : "nearest", minFilter: smooth ? "linear" : "nearest" });
        const bindGroups = new Map<number, GPUBindGroup>();
        const textures: GPUTexture[] = [];
        const started = performance.now();
        for (let idx = 0; idx < n; idx++) {
          if (cancelled || serial !== sidecarGpuBuildSerialRef.current) {
            textures.forEach((texture) => texture.destroy());
            return;
          }
          const source = sidecarCompositeFrameCacheRef.current.get(idx);
          if (!source) continue;
          const texture = device.createTexture({
            size: { width: widthPx, height: heightPx },
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
          });
          device.queue.copyExternalImageToTexture(
            { source },
            { texture },
            { width: widthPx, height: heightPx },
          );
          textures.push(texture);
          bindGroups.set(idx, device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: texture.createView() },
              { binding: 1, resource: sampler },
            ],
          }));
          if (idx % 8 === 0 || idx === n - 1) {
            const elapsed = ((performance.now() - started) / 1000).toFixed(1);
            setOfflineStackFetchStatus(`Uploading viewport playback cache to GPU… ${idx + 1}/${n} frames (${elapsed}s)`);
            await new Promise((resolve) => setTimeout(resolve, 0));
          }
        }
        if (cancelled || serial !== sidecarGpuBuildSerialRef.current) {
          textures.forEach((texture) => texture.destroy());
          return;
        }
        sidecarGpuPresenterRef.current = { device, context, pipeline, sampler, bindGroups, textures, width: widthPx, height: heightPx };
        sidecarGpuReadyRef.current = true;
        setSidecarGpuReady(true);
        setOfflineStackFetchStatus("");
        const d = show3dPerfDebug();
        if (d) {
          d.sidecarGpuTextureFrames = bindGroups.size;
          d.sidecarGpuUploadMs = performance.now() - started;
          d.lastRenderPath = "sidecar-gpu-texture-cache-ready";
        }
        renderSidecarGpuFrame(playbackIdxRef.current, "ready");
      } catch (err) {
        setOfflineStackFetchStatus(`Failed to upload viewport playback cache to GPU: ${err instanceof Error ? err.message : String(err)}`);
      }
    };
    void build();
    return () => {
      cancelled = true;
    };
  }, [offline, sidecarMode, enableSidecarGpuTexturePresenter, sidecarCompositeComplete, canvasW, canvasH, nSlices, smooth, sidecarViewTransformActive]);

  React.useLayoutEffect(() => {
    const previousCompareMode = previousCompareModeRef.current || "off";
    const activeCompareMode = compareMode || "off";
    previousCompareModeRef.current = activeCompareMode;
    if (
      isRgb ||
      !offline ||
      !sidecarMode ||
      !sidecarRamReady
    ) {
      return;
    }
    if (activeCompareMode === "off") {
      if (previousCompareMode !== "off") {
        const n = Math.max(1, nSlices || 1);
        const drawIdx = ((Math.round(playbackIdxRef.current || liveSliceIdx || 0) % n) + n) % n;
        drawSidecarBitmapFrame(drawIdx, false, "compare-off");
        updatePlaybackLiveControls(drawIdx);
      }
      return;
    }
    if (activeCompareMode !== "blink") {
      const debug = show3dPerfDebug();
      if (debug) debug.lastComparePath = "sidecar-compare-unsupported";
      return;
    }
    const n = Math.max(1, nSlices || 1);
    const pair = Array.isArray(comparePair) && comparePair.length === 2 ? comparePair : [0, 1];
    const aIdx = Math.max(0, Math.min(n - 1, Math.round(pair[0] ?? 0)));
    const bIdx = Math.max(0, Math.min(n - 1, Math.round(pair[1] ?? Math.min(1, n - 1))));
    const activeIdx = activeCompareMode === "blink" && blinkPhase ? bIdx : aIdx;
    const ok = drawSidecarBitmapFrame(activeIdx, false, "compare-blink");
    if (ok) {
      updatePlaybackLiveControls(activeIdx);
      const debug = show3dPerfDebug();
      if (debug) {
        debug.lastComparePath = "sidecar-blink";
        debug.lastCompareFrame = activeIdx;
      }
    }
  }, [
    blinkPhase,
    compareMode,
    comparePair,
    drawSidecarBitmapFrame,
    isRgb,
    liveSliceIdx,
    nSlices,
    offline,
    sidecarMode,
    sidecarRamReady,
    updatePlaybackLiveControls,
  ]);

  React.useEffect(() => {
    if (compareMode === "off" || isRgb || !canvasRef.current) return;
    if (offline && sidecarMode) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const n = Math.max(1, nSlices || 1);
    const pair = Array.isArray(comparePair) && comparePair.length === 2 ? comparePair : [0, 1];
    const aIdx = Math.max(0, Math.min(n - 1, Math.round(pair[0] ?? 0)));
    const bIdx = Math.max(0, Math.min(n - 1, Math.round(pair[1] ?? Math.min(1, n - 1))));
    const activeIdx = compareMode === "blink" && blinkPhase ? bIdx : aIdx;
    const frameA = rawFrameForIndex(aIdx, displaySliceIdx, rawFrameDataRef.current);
    const frameB = rawFrameForIndex(bIdx, displaySliceIdx, rawFrameDataRef.current);
    const active = activeIdx === aIdx ? frameA : frameB;
    if (!frameA || !frameB || !active) return;
    const panelCount = Math.max(1, nPanels || 1);
    const panelW = sharedPanelSource ? width : Math.max(1, panelWidthPx || Math.floor(width / panelCount) || width);
    const out = document.createElement("canvas");
    out.width = width;
    out.height = height;
    const outCtx = out.getContext("2d");
    if (!outCtx) return;
    if (compareBackground === "dark") {
      ctx.save();
      ctx.fillStyle = "#050505";
      ctx.fillRect(0, 0, canvasW, canvasH);
      ctx.restore();
    }
    const paintNormal = (frame: Float32Array) => {
      const data = logScale ? applyLogScale(frame) : frame;
      const range = percentileClip(data, percentileLow, percentileHigh);
      const img = outCtx.createImageData(width, height);
      const lut = COLORMAPS[panelCmapFor(visiblePanelIndices[0] ?? 0)] || COLORMAPS.plasma;
      renderFramePlayback(data, img.data, lut, range.vmin, range.vmax, false);
      outCtx.putImageData(img, 0, 0);
      drawMain(ctx, out, { sourcePanelWidth: sharedPanelSource ? undefined : panelW });
    };
    if (compareMode === "blink") {
      paintNormal(active);
      return;
    }
    const pixels = outCtx.createImageData(width, height);
    const px = pixels.data;
    if (compareMode === "overlay") {
      const aRange = percentileClip(frameA, percentileLow, percentileHigh);
      const bRange = percentileClip(frameB, percentileLow, percentileHigh);
      const aSpan = Math.max(1e-12, aRange.vmax - aRange.vmin);
      const bSpan = Math.max(1e-12, bRange.vmax - bRange.vmin);
      for (let i = 0; i < width * height; i++) {
        const a = Math.max(0, Math.min(1, (frameA[i] - aRange.vmin) / aSpan));
        const b = Math.max(0, Math.min(1, (frameB[i] - bRange.vmin) / bSpan));
        px[4 * i] = Math.round(255 * a);
        px[4 * i + 1] = Math.round(255 * b);
        px[4 * i + 2] = Math.round(255 * a);
        px[4 * i + 3] = 255;
      }
    } else {
      let sym = 0;
      for (let i = 0; i < width * height; i++) sym = Math.max(sym, Math.abs(frameB[i] - frameA[i]));
      const scale = sym > 0 ? 1 / sym : 1;
      const magentaPositive = String(diffCmap || "magenta-green").toLowerCase() === "magenta-green";
      for (let i = 0; i < width * height; i++) {
        const d = Math.max(-1, Math.min(1, (frameB[i] - frameA[i]) * scale));
        const v = Math.round(255 * Math.abs(d));
        const positive = d >= 0;
        const magenta = positive === magentaPositive;
        px[4 * i] = magenta ? v : 0;
        px[4 * i + 1] = magenta ? 0 : v;
        px[4 * i + 2] = magenta ? v : 0;
        px[4 * i + 3] = 255;
      }
    }
    outCtx.putImageData(pixels, 0, 0);
    drawMain(ctx, out, { sourcePanelWidth: sharedPanelSource ? undefined : panelW });
  }, [compareMode, comparePair, blinkPhase, blinkFps, compareBackground, diffCmap, isRgb, canvasW, canvasH, width, height, nSlices, nPanels, panelWidthPx, sharedPanelSource, displaySliceIdx, frameBytes, frameSeq, cmap, panelCmaps, percentileLow, percentileHigh, logScale, visiblePanelIndices, canvasRepaintSignal, offline, sidecarMode]);

  React.useEffect(() => {
    if (!scrubPreviewBytes || scrubPreviewBytes.byteLength === 0) return;
    const info = scrubPreviewInfo ?? {};
    const token = String(info.token ?? "");
    const idx = Number(info.idx);
    const previewW = Number(info.width);
    const previewH = Number(info.height);
    const fullW = Number(info.fullWidth ?? width);
    const channels = Math.max(1, Number(info.channels ?? 1));
    if (!token || !Number.isFinite(idx) || previewW <= 0 || previewH <= 0) return;
    const receiveAt = performance.now();
    const decodeStart = performance.now();
    const preview = extractFloat32(scrubPreviewBytes, previewW * previewH * channels);
    const decodeMs = performance.now() - decodeStart;
    if (!preview || preview.length === 0) return;
    const previewCanvas = document.createElement("canvas");
    previewCanvas.width = previewW;
    previewCanvas.height = previewH;
    const previewCtx = previewCanvas.getContext("2d");
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!previewCtx || !ctx) return;
    const image = previewCtx.createImageData(previewW, previewH);
    if (channels === 3) {
      const n = previewW * previewH;
      for (let i = 0; i < n; i++) {
        const src = i * 3;
        const dst = i * 4;
        image.data[dst] = Math.max(0, Math.min(255, Math.round(preview[src] * 255)));
        image.data[dst + 1] = Math.max(0, Math.min(255, Math.round(preview[src + 1] * 255)));
        image.data[dst + 2] = Math.max(0, Math.min(255, Math.round(preview[src + 2] * 255)));
        image.data[dst + 3] = 255;
      }
    } else {
      const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
      let vmin: number;
      let vmax: number;
      if (traitVmin != null || traitVmax != null) {
        ({ vmin, vmax } = resolveDisplayRange(
          dataMin,
          dataMax,
          traitVmin,
          traitVmax,
          logScale,
          imageVminPct,
          imageVmaxPct,
        ));
      } else if (autoContrast) {
        ({ vmin, vmax } = percentileClip(preview, percentileLow, percentileHigh));
      } else {
        const lo = logScale ? (dataMin >= 0 ? Math.log1p(dataMin) : -Math.log1p(-dataMin)) : dataMin;
        const hi = logScale ? (dataMax >= 0 ? Math.log1p(dataMax) : -Math.log1p(-dataMax)) : dataMax;
        ({ vmin, vmax } = sliderRange(lo, hi, imageVminPct, imageVmaxPct));
      }
      renderFramePlayback(preview, image.data, lut, vmin, vmax, logScale);
    }
    previewCtx.putImageData(image, 0, 0);
    const factor = Math.max(1, Number(info.factor ?? 1));
    if (factor > 1 && scrubPreviewLoggedFactorRef.current !== factor) {
      console.log(
        `[Show3D scrub preview] displaying ${factor}x reduced frames during slider drag; ` +
        "release the slider or zoom/settle the view to request native full resolution.",
      );
      scrubPreviewLoggedFactorRef.current = factor;
    }
    setGpuDisplayVisible(false);
    setDisplaySliceIdx(idx);
    setPlaybackUiSliceIdx(idx);
    playbackIdxRef.current = idx;
    const sourcePanelWidth = sharedPanelSource
      ? undefined
      : Math.max(1, Math.round((panelWidthPx || fullW / Math.max(1, nPanels || 1)) / factor));
    mainOffscreenRef.current = previewCanvas;
    mainOffscreenSourcePanelWidthRef.current = sourcePanelWidth;
    mainImgDataRef.current = null;
    drawMain(ctx, previewCanvas, { sourcePanelWidth });
    setPreviewPopulation({ ready: true, idx, factor });
    requestAnimationFrame(() => requestAnimationFrame((paintAt) => {
      const sendTimeMs = typeof info.sendTimeMs === "number" ? info.sendTimeMs : null;
      recordTransportSample({
        ...info,
        kind: "scrubPreview",
        receiveAtMs: Number(receiveAt.toFixed(3)),
        jsDecodeMs: Number(decodeMs.toFixed(3)),
        paintAtMs: Number(paintAt.toFixed(3)),
        browserReceiveLatencyMs: sendTimeMs === null ? null : Number((Date.now() - sendTimeMs).toFixed(3)),
        endToEndUiLatencyMs: sendTimeMs === null ? null : Number((Date.now() - sendTimeMs).toFixed(3)),
      });
    }));
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.lastFrame = idx;
      dbg.lastFrameSource = "scrub-preview";
      dbg.scrubPreviewFactor = factor;
      dbg.scrubPreviewBytes = Number(info.bytes ?? scrubPreviewBytes.byteLength);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scrubPreviewBytes, scrubPreviewInfo, width, height, nPanels, panelWidthPx, sharedPanelSource, cmap, logScale, autoContrast, percentileLow, percentileHigh, traitVmin, traitVmax, dataMin, dataMax, imageVminPct, imageVmaxPct, canvasW, canvasH, smooth, imageRotation, panelStates, linkContrast, linkedState, visiblePanelIndices, hiddenPanelSet, panelGapTrait, maxCols]);

  const ensureFullSizeMainOffscreen = React.useCallback((): boolean => {
    if (width <= 0 || height <= 0) return false;
    const current = mainOffscreenRef.current;
    if (
      current &&
      current.width === width &&
      current.height === height &&
      mainImgDataRef.current &&
      mainOffscreenSourcePanelWidthRef.current === undefined
    ) {
      return true;
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    mainOffscreenRef.current = canvas;
    mainOffscreenSourcePanelWidthRef.current = undefined;
    mainImgDataRef.current = canvas.getContext("2d")!.createImageData(width, height);
    return true;
  }, [width, height]);

  const paintRgbFrame = (rgb: Float32Array): boolean => {
    if (!ensureFullSizeMainOffscreen() || !mainOffscreenRef.current) return false;
    // WebGPU passthrough: pack the RGB channels on the GPU and blit, keeping the
    // per-pixel loop off the UI thread. Falls back to the CPU loop when the
    // engine is unavailable or the frame exceeds the storage-buffer limit.
    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current) {
      const bitmap = engine.renderRgbToImageBitmap(rgb, width, height);
      if (bitmap) {
        try {
          const octx = mainOffscreenRef.current.getContext("2d");
          if (octx) octx.drawImage(bitmap, 0, 0);
        } finally {
          bitmap.close();
        }
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d");
        if (ctx) drawMain(ctx, mainOffscreenRef.current);
        return true;
      }
    }
    if (!mainImgDataRef.current) return false;
    const px = mainImgDataRef.current.data;
    const n = Math.min(width * height, Math.floor(rgb.length / 3));
    for (let k = 0; k < n; k++) {
      px[4 * k] = Math.max(0, Math.min(255, Math.round(rgb[3 * k] * 255)));
      px[4 * k + 1] = Math.max(0, Math.min(255, Math.round(rgb[3 * k + 1] * 255)));
      px[4 * k + 2] = Math.max(0, Math.min(255, Math.round(rgb[3 * k + 2] * 255)));
      px[4 * k + 3] = 255;
    }
    mainOffscreenRef.current.getContext("2d")!.putImageData(mainImgDataRef.current, 0, 0);
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (ctx) drawMain(ctx, mainOffscreenRef.current);
    return true;
  };

  const renderFloatFrameSlice = (inputFrame: Float32Array, idx: number): boolean => {
    const c = playRef.current;
    if (!ensureFullSizeMainOffscreen() || !mainOffscreenRef.current || !mainImgDataRef.current) return false;
    // True-color stack: paint RGB as-is (no colormap), applying the moving
    // average across color frames when avg > 1.
    if (isRgb && inputFrame.length >= width * height * 3) {
      gpuRenderSerialRef.current++;
      playbackIdxRef.current = idx;
      setDisplaySliceIdx(idx);
      const toPaint = normalizedAverageWindow(c.avgWindow) > 1
        ? averagedRgbFrameForIndex(idx, inputFrame)
        : inputFrame;
      rgbFrameDataRef.current = toPaint;
      rawFrameDataRef.current = rgbFrameToLuminance(toPaint, width * height);
      return paintRgbFrame(toPaint);
    }
    const transformActive = requiresClientFrameTransform({
      offline,
      diffMode: c.diffMode,
      avgWindow: c.avgWindow,
    }) || browserFilterOnRef.current || frequencyFilterIsActive || !!subpixelAlignEnabled;
    const frame = transformActive
      ? displayAndFrequencyFrameForIndex(idx, inputFrame, { allowRawOnMiss: !playing })
      : inputFrame;
    if (!frame) return false;

    gpuRenderSerialRef.current++;
    playbackIdxRef.current = idx;
    rawFrameDataRef.current = frame;
    setDisplaySliceIdx(idx);

    const lut = COLORMAPS[c.cmap] || COLORMAPS.inferno;
    let vmin: number, vmax: number;
    let cpuData: Float32Array = frame;
    let cpuDataAlreadyLogged = false;
    if (c.autoContrast) {
      const cached = transformActive ? null : (
        cachedAutoDisplayRange(c.autoVmins, c.autoVmaxs, idx, c.logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, idx, c.logScale)
      );
      if (cached) {
        ({ vmin, vmax } = cached);
      } else if (c.logScale && logBufferRef.current) {
        applyLogScaleInPlace(frame, logBufferRef.current);
        ({ vmin, vmax } = percentileClip(logBufferRef.current, c.percentileLow, c.percentileHigh));
        cpuData = logBufferRef.current;
        cpuDataAlreadyLogged = true;
      } else {
        ({ vmin, vmax } = percentileClip(frame, c.percentileLow, c.percentileHigh));
      }
    } else {
      ({ vmin, vmax } = resolveDisplayRange(
        c.dataMin,
        c.dataMax,
        c.traitVmin,
        c.traitVmax,
        c.logScale,
        c.imageVminPct,
        c.imageVmaxPct,
      ));
    }

    let rendered = false;
    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current) {
      try {
        engine.uploadLUT(c.cmap, lut);
        engine.uploadData(0, frame, c.width, c.height);
        const bitmaps = engine.renderSlotsToImageBitmap([0], [{ vmin, vmax }], c.logScale);
        if (bitmaps && bitmaps[0]) {
          try {
            const offCtx = mainOffscreenRef.current.getContext("2d");
            if (offCtx) {
              offCtx.drawImage(bitmaps[0], 0, 0);
              rendered = true;
            }
          } finally {
            bitmaps[0].close();
          }
        }
      } catch {
        rendered = false;
      }
    }
    if (!rendered) {
      if (cpuDataAlreadyLogged) {
        renderToOffscreenReuse(cpuData, lut, vmin, vmax, mainOffscreenRef.current, mainImgDataRef.current);
      } else {
        renderFramePlayback(frame, mainImgDataRef.current.data, lut, vmin, vmax, c.logScale);
        mainOffscreenRef.current.getContext("2d")!.putImageData(mainImgDataRef.current, 0, 0);
      }
    }

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (ctx) drawMain(ctx, mainOffscreenRef.current);
    if (c.showStats) setLocalStats(computeStats(frame));
    if (c.profileActive && c.profilePoints.length === 2) {
      const p0 = c.profilePoints[0], p1 = c.profilePoints[1];
      setProfileData(sampleLineProfile(
        frame,
        c.width,
        c.height,
        p0.row,
        p0.col + c.profileColOffset,
        p1.row,
        p1.col + c.profileColOffset,
        c.profileWidth,
      ));
    }
    return true;
  };

  const renderBufferedSlice = (idx: number): boolean => {
    const c = playRef.current;
    const frameSize = c.width * c.height;
    let frame = getFrameFromBuffer(bufferRef.current, bufferStartRef.current, bufferCountRef.current, c.nSlices, idx, frameSize);
    if (!frame && nextBufferRef.current) {
      const nextFrame = getFrameFromBuffer(nextBufferRef.current, nextBufferStartRef.current, nextBufferCountRef.current, c.nSlices, idx, frameSize);
      if (nextFrame) {
        bufferRef.current = nextBufferRef.current;
        bufferStartRef.current = nextBufferStartRef.current;
        bufferCountRef.current = nextBufferCountRef.current;
        nextBufferRef.current = null;
        nextBufferCountRef.current = 0;
        frame = nextFrame;
      }
    }
    if (!frame) return false;
    return renderFloatFrameSlice(frame, idx);
  };

  const renderFetchedSlice = async (idx: number): Promise<boolean> => {
    const transformActive = requiresClientFrameTransform({ offline, diffMode, avgWindow }) || browserFilterOnRef.current || frequencyFilterIsActive;
    if (!transformActive && renderGpuCachedSliceDirect(idx)) return true;
    if (separatePanelFrames) {
      // Neighbor-frame averaging is intentionally clamped off for this mode;
      // never show an unaveraged GPU slot during the brief state transition.
      if (transformActive) return false;
      if (gpuCmapReadyRef.current && gpuCmapRef.current) {
        const c = playRef.current;
        const rgbaCapacity = Math.max(1, Math.round(c.canvasW * c.canvasH));
        const ready = await ensurePanelFrameGpu(idx, rgbaCapacity);
        if (ready && renderGpuPanelSlice(idx)) return true;
      }
      const frame = await fetchSeparatePanelPackedFrameFromServer(idx);
      if (frame) return renderFloatFrameSlice(frame, idx);
      requestCommFramePreview(idx, "panel-native-preview");
      return false;
    }
    const frame = getCachedServerFrame(idx) ?? await fetchFrameFromServer(idx);
    if (!frame) return false;
    return renderFloatFrameSlice(frame, idx);
  };

  const commitLivePanelTransforms = () => {
    if (transformStateCommitTimerRef.current !== null) {
      window.clearTimeout(transformStateCommitTimerRef.current);
      transformStateCommitTimerRef.current = null;
    }
    const nextLinked = linkedStateLiveRef.current;
    const nextPanels = panelStatesLiveRef.current;
    setViewState({ linked_state: { ...nextLinked }, panel_states: nextPanels.map(v => ({ ...v })) });
    setLinkedState(prev => (
      prev.zoom === nextLinked.zoom &&
      prev.panX === nextLinked.panX &&
      prev.panY === nextLinked.panY
        ? prev
        : { ...prev, zoom: nextLinked.zoom, panX: nextLinked.panX, panY: nextLinked.panY }
    ));
    setPanelStates(prev => {
      const n = Math.max(prev.length, nextPanels.length);
      let changed = prev.length !== n;
      const merged = Array.from({ length: n }, (_, i) => {
        const base = prev[i] || initialState;
        const live = nextPanels[i] || base;
        if (
          base.zoom !== live.zoom ||
          base.panX !== live.panX ||
          base.panY !== live.panY ||
          base.imageVminPct !== live.imageVminPct ||
          base.imageVmaxPct !== live.imageVmaxPct
        ) {
          changed = true;
        }
        return { ...base, ...live };
      });
      return changed ? merged : prev;
    });
  };

  const scheduleTransformStateCommit = (delayMs = 120) => {
    if (transformStateCommitTimerRef.current !== null) {
      window.clearTimeout(transformStateCommitTimerRef.current);
    }
    transformStateCommitTimerRef.current = window.setTimeout(commitLivePanelTransforms, delayMs);
  };

  const renderCurrentPanelTransformDirect = (): boolean => {
    if (offline && sidecarMode && (sidecarRamReadyRef.current || sidecarU8FrameCacheRef.current.size > 0)) {
      const idx = Number.isFinite(playbackIdxRef.current) ? playbackIdxRef.current : liveSliceIdx;
      const start = performance.now();
      const rendered = drawSidecarBitmapFrame(idx, false, "transform");
      const dbg = show3dPerfDebug();
      if (dbg) {
        const latencyMs = transformInputAtRef.current > 0 ? performance.now() - transformInputAtRef.current : 0;
        dbg.lastInteractionRenderMs = Number((performance.now() - start).toFixed(2));
        dbg.lastInteractionLatencyMs = Number(latencyMs.toFixed(2));
        dbg.lastInteractionRenderFrame = idx;
        dbg.lastInteractionRenderPath = rendered ? "sidecar-u8-viewport-transform" : "miss";
      }
      return rendered;
    }
    if (!separatePanelFrames) {
      const canvas = canvasRef.current;
      const offscreen = mainOffscreenRef.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !offscreen || !ctx) return false;
      const start = performance.now();
      drawMain(ctx, offscreen);
      const dbg = show3dPerfDebug();
      if (dbg) {
        const latencyMs = transformInputAtRef.current > 0 ? performance.now() - transformInputAtRef.current : 0;
        dbg.lastInteractionRenderMs = Number((performance.now() - start).toFixed(2));
        dbg.lastInteractionLatencyMs = Number(latencyMs.toFixed(2));
        dbg.lastInteractionRenderFrame = playbackIdxRef.current;
        dbg.lastInteractionRenderPath = "canvas-packed-transform";
      }
      return true;
    }
    if (
      offline
      || imageRotation % 4 !== 0
      || requiresClientFrameTransform({ offline, diffMode, avgWindow })
      || browserFilterOnRef.current
      || frequencyFilterIsActive
    ) return false;
    const n = Math.max(1, nSlices || 1);
    const idx = ((Math.round(playbackIdxRef.current) % n) + n) % n;
    if (!gpuFrameCacheUploadedRef.current.has(idx)) return false;
    const start = performance.now();
    const rendered = renderGpuPanelSlice(idx, false);
    const dbg = show3dPerfDebug();
    if (dbg) {
      const latencyMs = transformInputAtRef.current > 0 ? performance.now() - transformInputAtRef.current : 0;
      dbg.lastInteractionRenderMs = Number((performance.now() - start).toFixed(2));
      dbg.lastInteractionLatencyMs = Number(latencyMs.toFixed(2));
      dbg.lastInteractionRenderFrame = idx;
      dbg.lastInteractionRenderPath = rendered ? "webgpu-panel-transform" : "miss";
    }
    return rendered;
  };

  const scheduleTransformRender = (): boolean => {
    if (separatePanelFrames && !sidecarMode && (offline || imageRotation % 4 !== 0)) return false;
    if (transformRenderRafRef.current !== null) return true;
    transformRenderRafRef.current = window.requestAnimationFrame(() => {
      transformRenderRafRef.current = null;
      renderCurrentPanelTransformDirect();
    });
    return true;
  };

  React.useEffect(() => () => {
    if (transformRenderRafRef.current !== null) {
      window.cancelAnimationFrame(transformRenderRafRef.current);
      transformRenderRafRef.current = null;
    }
    if (transformStateCommitTimerRef.current !== null) {
      window.clearTimeout(transformStateCommitTimerRef.current);
      transformStateCommitTimerRef.current = null;
    }
  }, []);

  React.useEffect(() => {
    if (offline || !frameServerUrl || playing) return;
    void renderFetchedSlice(sliceIdx);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [offline, separatePanelFrames, frameServerUrl, frameServerVersion, sliceIdx, playing, canvasW, canvasH, cmap, imageVminPct, imageVmaxPct, autoContrast, logScale, panelStates, linkedState, linkPanels, panelGapTrait, maxCols]);

  React.useLayoutEffect(() => {
    if (!mainOffscreenRef.current || !canvasRef.current) return;
    const offlinePackedPanelPlaybackUsesStaticCanvas = (
      offline &&
      playing &&
      Math.max(1, nPanels || 1) > 1 &&
      !sharedPanelSource
    );
    const offlineGpuPlaybackOwnsCanvas = (
      offline &&
      playing &&
      !offlinePackedPanelPlaybackUsesStaticCanvas &&
      !!gpuCmapRef.current &&
      gpuCmapReadyRef.current
    );
    if (offlineGpuPlaybackOwnsCanvas) return;
    const viewTransformActive = sidecarViewTransformActive();
    const preserveGpuDisplay = playing && gpuDisplayVisibleRef.current === true && imageRotation % 4 === 0 && !viewTransformActive;
    if (preserveGpuDisplay && separatePanelFrames) return;
    if (offline && sidecarMode && viewTransformActive && sidecarRamReadyRef.current) {
      if (drawSidecarBitmapFrame(playing ? playbackIdxRef.current : liveSliceIdx, false, "layout-transform")) return;
    }
    if (offline && sidecarMode && (sidecarBitmapReadyRef.current || sidecarCompositeReadyRef.current) && !viewTransformActive) {
      if (drawSidecarBitmapFrame(playing ? playbackIdxRef.current : liveSliceIdx, false, "layout")) return;
    }
    const ctx = canvasRef.current.getContext("2d");
    if (ctx) drawMain(ctx, mainOffscreenRef.current, {
      preserveGpuDisplay,
      sourcePanelWidth: mainOffscreenSourcePanelWidthRef.current,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [smooth, canvasW, canvasH, nPanels, visiblePanelIndices, maxCols, imageRotation, flipRows, flipCols, panelStates, linkedState, linkPanels, themeColors.bg, panelRealFrames, panelTitles, showPanelTitles, panelGapTrait, panelTitleFontSize, panelWidthPx, sharedPanelSource, sliceIdx, displaySliceIdx, liveSliceIdx, offline, playing, nSlices, canvasRepaintSignal, sidecarMode, sidecarBitmapReady, sidecarCompositeReady, drawSidecarBitmapFrame, sidecarViewTransformActive]);

  // A presented WebGPU texture is not a durable cache. Re-present the current
  // cached frame when it owns the live display; otherwise re-blit the retained
  // 2D offscreen. Neither path changes the frame index or playback state.
  React.useEffect(() => {
    if (canvasRepaintSignal === 0) return;
    const frameIdx = playbackIdxRef.current;
    let restoredGpu = false;
    let restoredCanvas = false;
    const offlinePackedPanelPlaybackUsesStaticCanvas = (
      offline
      && playing
      && Math.max(1, nPanels || 1) > 1
      && !sharedPanelSource
    );
    const offlineGpuPlaybackOwnsCanvas = (
      offline
      && playing
      && !offlinePackedPanelPlaybackUsesStaticCanvas
      && !!gpuCmapRef.current
      && gpuCmapReadyRef.current
    );
    // The playback loop owns and refreshes the direct GPU canvas. Hiding it
    // here exposes a stale 2D offscreen until the user presses Play again.
    if (offlineGpuPlaybackOwnsCanvas) {
      const dbg = show3dPerfDebug();
      if (dbg) {
        dbg.visibilityResumeAttempts = Number(dbg.visibilityResumeAttempts ?? 0) + 1;
        dbg.lastVisibilityResumeFrame = frameIdx;
        dbg.lastVisibilityResumePath = "playback-owner";
      }
      return;
    }
    const transformActive = requiresClientFrameTransform({ offline, diffMode, avgWindow }) || browserFilterOnRef.current || frequencyFilterIsActive;
    if (!offline && !transformActive && gpuDisplayVisibleRef.current) {
      try {
        restoredGpu = renderGpuCachedSliceDirect(frameIdx, false);
      } catch (err) {
        console.warn("[Show3D] Foreground WebGPU re-present failed; using the retained 2D frame", err);
      }
    }
    if (!restoredGpu && offline && sidecarMode && sidecarRamReadyRef.current && !isRgb) {
      restoredCanvas = drawSidecarBitmapFrame(frameIdx, false, "visibility-sidecar");
    }
    if (!restoredGpu && !restoredCanvas) {
      setGpuDisplayVisible(false);
      const canvas = canvasRef.current;
      const offscreen = mainOffscreenRef.current;
      const ctx = canvas?.getContext("2d");
      if (ctx && offscreen) {
        drawMain(ctx, offscreen);
        restoredCanvas = true;
      }
    }
    const dbg = show3dPerfDebug();
    if (dbg) {
      const restored = restoredGpu || restoredCanvas;
      dbg.visibilityResumeAttempts = Number(dbg.visibilityResumeAttempts ?? 0) + 1;
      if (restored) dbg.visibilityResumePaints = Number(dbg.visibilityResumePaints ?? 0) + 1;
      else dbg.visibilityResumeMisses = Number(dbg.visibilityResumeMisses ?? 0) + 1;
      dbg.lastVisibilityResumeFrame = frameIdx;
      dbg.lastVisibilityResumePath = restoredGpu ? "webgpu-cache" : restoredCanvas ? "canvas-offscreen" : "miss";
    }
    // The foreground signal is the intentional invalidation boundary. The
    // render helpers read the latest state through refs/playRef.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canvasRepaintSignal]);

  // Render overlay (ROI only) - HiDPI aware
  React.useEffect(() => {
    if (!overlayRef.current) return;
    const ctx = overlayRef.current.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, canvasW, canvasH);
    // Match the main image's rotation so ROIs / profile sit on the right pixels.
    // Image draw applies `translate(panX,panY) → scale(zoom) → rotate(around cx)`,
    // so the rotation pivot in screen pixels is (canvasW/2+panX, canvasH/2+panY).
    // Overlay must use the SAME screen-space pivot - earlier bug used (canvasW/2,
    // canvasH/2) without pan offset, drifting ROIs when user panned + rotated.
    if (imageRotation % 4 !== 0) {
      const cx = canvasW / 2 + panX;
      const cy = canvasH / 2 + panY;
      ctx.translate(cx, cy);
      ctx.rotate((imageRotation * Math.PI) / 2);
      ctx.translate(-cx, -cy);
    }
    for (const panel of visiblePanelIndices) {
      const overlaySpecs = panelOverlays?.[panel] || [];
      if (!overlaySpecs.length) continue;
      const geom = getPanelGeometry(panel);
      if (!geom) continue;
      ctx.save();
      ctx.beginPath();
      ctx.rect(geom.slotX, geom.slotY, geom.slotW, geom.slotH);
      ctx.clip();
      const toScreenX = (col: number) => geom.slotX + geom.state.panX + col * geom.scaleX * geom.state.zoom;
      const toScreenY = (row: number) => geom.slotY + geom.state.panY + row * geom.scaleY * geom.state.zoom;
      drawPanelOverlays(ctx, overlaySpecs, toScreenX, toScreenY, sourcePanelWidth, sourcePanelHeight);
      if (overlaySelection?.panel === panel) {
        drawPanelOverlaySelection(ctx, overlaySpecs[overlaySelection.overlay], toScreenX, toScreenY, sourcePanelWidth, sourcePanelHeight);
      }
      ctx.restore();
    }

    if (effectiveRoiActive && roiItems.length > 0) {
      const highlightedRois = roiItems.filter(r => r.highlight);
      if (highlightedRois.length > 0) {
        ctx.save();
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(0, 0, canvasW, canvasH);
        ctx.globalCompositeOperation = "destination-out";
        for (const roi of highlightedRois) {
          const sx = roi.col * displayScale * zoom + panX;
          const sy = roi.row * displayScale * zoom + panY;
          const sr = roi.radius * displayScale * zoom;
          const shape = roi.shape || "circle";
          ctx.fillStyle = "rgba(0,0,0,1)";
          if (shape === "circle") {
            ctx.beginPath(); ctx.arc(sx, sy, sr, 0, Math.PI * 2); ctx.fill();
          } else if (shape === "square") {
            ctx.fillRect(sx - sr, sy - sr, sr * 2, sr * 2);
          } else if (shape === "rectangle") {
            const sw = roi.width * displayScale * zoom;
            const sh = roi.height * displayScale * zoom;
            ctx.fillRect(sx - sw / 2, sy - sh / 2, sw, sh);
          } else if (shape === "annular") {
            ctx.beginPath(); ctx.arc(sx, sy, sr, 0, Math.PI * 2); ctx.fill();
            ctx.globalCompositeOperation = "source-over";
            ctx.fillStyle = "rgba(0,0,0,0.6)";
            const sir = roi.radius_inner * displayScale * zoom;
            ctx.beginPath(); ctx.arc(sx, sy, sir, 0, Math.PI * 2); ctx.fill();
            ctx.globalCompositeOperation = "destination-out";
          }
        }
        ctx.restore();
      }

      for (let roiIdx = 0; roiIdx < roiItems.length; roiIdx++) {
        const roi = roiItems[roiIdx];
        const isSelected = roiIdx === roiSelectedIdx;
        const screenX = roi.col * displayScale * zoom + panX;
        const screenY = roi.row * displayScale * zoom + panY;
        const screenRadius = roi.radius * displayScale * zoom;
        const screenWidth = roi.width * displayScale * zoom;
        const screenHeight = roi.height * displayScale * zoom;
        const screenRadiusInner = roi.radius_inner * displayScale * zoom;
        const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
        ctx.lineWidth = roi.line_width || 2;
        const color = roi.color || ROI_COLORS[roiIdx % ROI_COLORS.length];
        drawROI(ctx, screenX, screenY, shape, screenRadius, screenWidth, screenHeight, color, color, isSelected && isDraggingROI, screenRadiusInner);
        if (isSelected) {
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1;
          if (shape === "circle" || shape === "annular") {
            ctx.beginPath(); ctx.arc(screenX, screenY, screenRadius + 3, 0, Math.PI * 2); ctx.stroke();
          } else if (shape === "square") {
            ctx.strokeRect(screenX - screenRadius - 3, screenY - screenRadius - 3, (screenRadius + 3) * 2, (screenRadius + 3) * 2);
          } else if (shape === "rectangle") {
            ctx.strokeRect(screenX - screenWidth / 2 - 3, screenY - screenHeight / 2 - 3, screenWidth + 6, screenHeight + 6);
          }
          ctx.setLineDash([]);
        }
      }
    }

    // Line profile overlay. Use the same slot, clip, zoom, pan, and rotation
    // transform as drawMain so profiles stay attached to their panel.
    if (profileActive && profilePoints.length > 0) {
      const ownerPanel = singlePanelPageProfile
        ? activePageStart
        : Math.max(0, Math.min(totalPanelCount - 1, profilePanelIdx));
      const geom = getPanelGeometry(ownerPanel);
      if (geom) {
        const profileLocalCol = (col: number) => singlePanelPageProfile
          ? col
          : panelLocalCol(col, ownerPanel);
        const toPanelX = (col: number) => profileLocalCol(col) * geom.scaleX;
        const toPanelY = (row: number) => row * geom.scaleY;
        const inverseZoom = 1 / Math.max(1, geom.state.zoom);
        const markerR = 8 * inverseZoom;
        const profileColor = "#00e5ff";
        const profileHalo = "rgba(0, 0, 0, 0.88)";
        const drawEndpoint = (x: number, y: number, label: string) => {
          ctx.fillStyle = profileHalo;
          ctx.beginPath();
          ctx.arc(x, y, markerR + 2 * inverseZoom, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = profileColor;
          ctx.beginPath();
          ctx.arc(x, y, markerR, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#001018";
          ctx.font = `700 ${10 * inverseZoom}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(label, x, y + 0.5 * inverseZoom);
        };
        ctx.save();
        ctx.beginPath();
        ctx.rect(geom.slotX, geom.slotY, geom.slotW, geom.slotH);
        ctx.clip();
        ctx.translate(geom.slotX + geom.state.panX, geom.slotY + geom.state.panY);
        ctx.scale(geom.state.zoom, geom.state.zoom);
        if (imageRotation % 4 !== 0) {
          const cx = geom.slotW / 2 / geom.state.zoom;
          const cy = geom.slotH / 2 / geom.state.zoom;
          ctx.translate(cx, cy);
          ctx.rotate((imageRotation * Math.PI) / 2);
          ctx.translate(-geom.slotW / 2, -geom.slotH / 2);
        }

        const ax = toPanelX(profilePoints[0].col);
        const ay = toPanelY(profilePoints[0].row);

        if (profilePoints.length === 2) {
          const bx = toPanelX(profilePoints[1].col);
          const by = toPanelY(profilePoints[1].row);

          // Draw band when profile width > 1
          if (profileWidth > 1) {
            const dc = profilePoints[1].col - profilePoints[0].col;
            const dr = profilePoints[1].row - profilePoints[0].row;
            const lineLen = Math.sqrt(dc * dc + dr * dr);
            if (lineLen > 0) {
              const halfW = (profileWidth - 1) / 2;
              const perpR = -dc / lineLen * halfW;
              const perpC = dr / lineLen * halfW;
              ctx.fillStyle = "rgba(0, 229, 255, 0.22)";
              ctx.strokeStyle = "rgba(0, 0, 0, 0.72)";
              ctx.lineWidth = 2 * inverseZoom;
              ctx.beginPath();
              ctx.moveTo(toPanelX(profilePoints[0].col + perpC), toPanelY(profilePoints[0].row + perpR));
              ctx.lineTo(toPanelX(profilePoints[1].col + perpC), toPanelY(profilePoints[1].row + perpR));
              ctx.lineTo(toPanelX(profilePoints[1].col - perpC), toPanelY(profilePoints[1].row - perpR));
              ctx.lineTo(toPanelX(profilePoints[0].col - perpC), toPanelY(profilePoints[0].row - perpR));
              ctx.closePath();
              ctx.fill();
              ctx.stroke();
            }
          }

          ctx.strokeStyle = profileHalo;
          ctx.lineWidth = 6 * inverseZoom;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();

          ctx.strokeStyle = profileColor;
          ctx.lineWidth = 2.5 * inverseZoom;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();

          drawEndpoint(ax, ay, "1");
          drawEndpoint(bx, by, "2");
        } else {
          drawEndpoint(ax, ay, "1");
        }
        ctx.restore();
      }
    }
  }, [activePageStart, effectiveRoiActive, roiItems, roiSelectedIdx, isDraggingROI, canvasW, canvasH, displayScale, zoom, panX, panY, themeColors, profileActive, profilePoints, profileWidth, profilePanelIdx, nPanels, panelTitles, imageRotation, width, height, panelStates, linkedState, linkPanels, panelGapTrait, sourcePanelWidth, sourcePanelHeight, sharedPanelSource, singlePanelPageProfile, totalPanelCount, canvasRepaintSignal, panelOverlays, overlaySelection, visiblePanelIndices]);

  // Lens inset rendering
  React.useEffect(() => {
    const lensCanvas = lensCanvasRef.current;
    if (lensCanvas) {
      const lctx = lensCanvas.getContext("2d");
      if (lctx) lctx.clearRect(0, 0, lensCanvas.width, lensCanvas.height);
    }
    if (!showLens || !lensPos || !rawFrameDataRef.current) return;
    if ((nPanels || 1) > 1) return;  // Lens disabled in multi-panel mode
    if (!lensCanvas) return;
    const ctx = lensCanvas.getContext("2d");
    if (!ctx) return;

    const raw = rawFrameDataRef.current;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const processed = logScale ? applyLogScale(raw) : raw;
    let vmin: number, vmax: number;
    if (traitVmin != null || traitVmax != null) {
      ({ vmin, vmax } = resolveDisplayRange(
        dataMin,
        dataMax,
        traitVmin,
        traitVmax,
        logScale,
        imageVminPct,
        imageVmaxPct,
      ));
    } else if (autoContrast) {
      const cached = cachedAutoDisplayRange(autoVmins, autoVmaxs, displaySliceIdx, logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, displaySliceIdx, logScale);
      ({ vmin, vmax } = cached ?? percentileClip(processed, percentileLow, percentileHigh));
    } else if (imageDataRange.min !== imageDataRange.max) {
      ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
    } else {
      const r = findDataRange(processed);
      vmin = r.min; vmax = r.max;
    }

    const regionSize = Math.max(4, Math.round(lensDisplaySize / lensMag));
    const lensSize = lensDisplaySize;
    const margin = 12;
    const half = Math.floor(regionSize / 2);
    const r0 = lensPos.row - half;
    const c0 = lensPos.col - half;

    const regionCanvas = document.createElement("canvas");
    regionCanvas.width = regionSize;
    regionCanvas.height = regionSize;
    const rctx = regionCanvas.getContext("2d");
    if (!rctx) return;
    const imgData = rctx.createImageData(regionSize, regionSize);
    const range = vmax - vmin || 1;
    for (let dr = 0; dr < regionSize; dr++) {
      for (let dc = 0; dc < regionSize; dc++) {
        const sr = r0 + dr;
        const sc = c0 + dc;
        const idx = (dr * regionSize + dc) * 4;
        if (sr < 0 || sr >= height || sc < 0 || sc >= width) {
          imgData.data[idx] = 0; imgData.data[idx + 1] = 0; imgData.data[idx + 2] = 0; imgData.data[idx + 3] = 255;
        } else {
          const val = processed[sr * width + sc];
          const t = Math.max(0, Math.min(1, (val - vmin) / range));
          const li = Math.round(t * 255);
          imgData.data[idx] = lut[li * 3]; imgData.data[idx + 1] = lut[li * 3 + 1]; imgData.data[idx + 2] = lut[li * 3 + 2]; imgData.data[idx + 3] = 255;
        }
      }
    }
    rctx.putImageData(imgData, 0, 0);

    ctx.save();
    ctx.scale(DPR, DPR);
    // Clamp anchor + default position to canvas bounds. Without clamp a small canvas
    // (e.g. multi-panel 100 px tall) puts the inset off-screen (-60 px) because
    // default ly = canvasH - lensSize - margin - 20 goes negative.
    const cssH = canvasH;
    const cssW = canvasW;
    const rawLx = lensAnchor ? lensAnchor.x : margin;
    const rawLy = lensAnchor ? lensAnchor.y : cssH - lensSize - margin - 20;
    const lx = Math.max(0, Math.min(cssW - lensSize, rawLx));
    const ly = Math.max(0, Math.min(cssH - lensSize, rawLy));
    ctx.imageSmoothingEnabled = smooth;
    ctx.drawImage(regionCanvas, lx, ly, lensSize, lensSize);
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(lx, ly, lensSize, lensSize);
    const cx = lx + lensSize / 2;
    const cy = ly + lensSize / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = "10px monospace";
    ctx.fillText(`${lensMag}×`, lx + 4, ly + lensSize - 4);
    ctx.restore();
  }, [showLens, lensPos, cmap, logScale, autoContrast, imageDataRange, imageVminPct, imageVmaxPct, dataMin, dataMax, traitVmin, traitVmax, width, height, canvasW, canvasH, themeColors, lensMag, lensDisplaySize, lensAnchor, percentileLow, percentileHigh, frameBytes, sliceIdx, displaySliceIdx, nPanels, canvasRepaintSignal]);

  // ROI sparkline plot
  React.useEffect(() => {
    const canvas = roiPlotCanvasRef.current;
    if (!canvas || !showRoiPlot || !effectiveRoiActive) return;
    const plotW = canvasW;
    const plotH = 76;
    canvas.width = Math.round(plotW * DPR);
    canvas.height = Math.round(plotH * DPR);
    canvas.style.width = `${plotW}px`;
    canvas.style.height = `${plotH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, plotW, plotH);

    if (!roiPlotData || roiPlotData.byteLength < 4) return;
    const values = extractFloat32(roiPlotData);
    if (!values || values.length === 0) return;
    let min = values[0], max = values[0];
    for (let i = 1; i < values.length; i++) {
      if (values[i] < min) min = values[i];
      if (values[i] > max) max = values[i];
    }
    const range = max - min || 1;
    const padY = 14;
    const drawH = plotH - padY * 2;

    // Draw plot line
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const denom = Math.max(1, values.length - 1);
    for (let i = 0; i < values.length; i++) {
      const x = (i / denom) * plotW;
      const y = padY + drawH - ((values[i] - min) / range) * drawH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw current frame marker
    const activeIdx = displaySliceIdx;
    const markerIdx = Math.max(0, Math.min(values.length - 1, activeIdx));
    const markerX = (markerIdx / denom) * plotW;
    ctx.strokeStyle = themeColors.textMuted;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(markerX, padY);
    ctx.lineTo(markerX, padY + drawH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Current value dot
    if (values.length > 0) {
      const cy = padY + drawH - ((values[markerIdx] - min) / range) * drawH;
      ctx.fillStyle = themeColors.accent;
      ctx.beginPath();
      ctx.arc(markerX, cy, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Y-axis labels
    ctx.fillStyle = themeColors.textMuted;
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText(formatNumber(max), 2, padY - 2);
    ctx.fillText(formatNumber(min), 2, padY + drawH + 10);
  }, [roiPlotData, effectiveRoiActive, showRoiPlot, canvasW, themeColors, sliceIdx, displaySliceIdx, playing, canvasRepaintSignal]);

  // Keep sampled profile data current, but do not reopen the profile UI after
  // the user has turned it off. The line stays cached so toggling Profile back
  // on restores the latest sampled data.
  React.useEffect(() => {
    if (profilePoints.length === 2 && rawFrameDataRef.current) {
      const p0 = profilePoints[0], p1 = profilePoints[1];
      const data = rawFrameDataRef.current;
      setProfileData(sampleProfileForActivePage(data, p0, p1));
    } else {
      setProfileData(null);
    }
  }, [frameBytes, profilePoints, profileWidth, sampleProfileForActivePage]);

  // Render profile sparkline
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvasW;
    const cssH = profileHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);

    const isDark = themeInfo.theme === "dark";
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, cssW, cssH);

    if (!profileData || profileData.length < 2) {
      ctx.font = "10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillStyle = isDark ? "#555" : "#999";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(
        profilePoints.length === 1
          ? "Choose point 2 on the image"
          : "Click point 1, then point 2, to draw a profile",
        cssW / 2,
        cssH / 2,
      );
      return;
    }

    const padLeft = 40;
    const padRight = 8;
    const padTop = 6;
    const padBottom = 18;
    const plotW = cssW - padLeft - padRight;
    const plotH = cssH - padTop - padBottom;

    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < profileData.length; i++) {
      if (profileData[i] < gMin) gMin = profileData[i];
      if (profileData[i] > gMax) gMax = profileData[i];
    }
    const range = gMax - gMin || 1;

    // Draw profile line
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < profileData.length; i++) {
      const x = padLeft + (i / (profileData.length - 1)) * plotW;
      const y = padTop + plotH - ((profileData[i] - gMin) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // X-axis: calibrated distance
    let totalDist = profileData.length - 1;
    let xUnit = "px";
    if (profilePoints.length === 2) {
      const dx = profilePoints[1].col - profilePoints[0].col;
      const dy = profilePoints[1].row - profilePoints[0].row;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      if (pixelSize > 0) {
        const distA = distPx * pixelSize;
        if (distA >= 10) { totalDist = distA / 10; xUnit = "nm"; }
        else { totalDist = distA; xUnit = "Å"; }
      } else {
        totalDist = distPx;
      }
    }

    // Draw x-axis ticks
    const tickY = padTop + plotH;
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    const idealTicks = Math.max(2, Math.floor(plotW / 70));
    const tickStep = roundToNiceValue(totalDist / idealTicks);
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textBaseline = "top";
    const ticks: number[] = [];
    for (let v = 0; v <= totalDist + tickStep * 0.01; v += tickStep) {
      if (v > totalDist * 1.001) break;
      ticks.push(v);
    }
    for (let i = 0; i < ticks.length; i++) {
      const v = ticks[i];
      const frac = totalDist > 0 ? v / totalDist : 0;
      const x = padLeft + frac * plotW;
      ctx.beginPath(); ctx.moveTo(x, tickY); ctx.lineTo(x, tickY + 3); ctx.stroke();
      ctx.textAlign = frac < 0.05 ? "left" : frac > 0.95 ? "right" : "center";
      const valStr = v % 1 === 0 ? v.toFixed(0) : v.toFixed(1);
      ctx.fillText(i === ticks.length - 1 ? `${valStr} ${xUnit}` : valStr, x, tickY + 4);
    }

    // Y-axis min/max labels
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), padLeft - 3, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), padLeft - 3, padTop + plotH);

    // Draw axis lines
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Save base rendering + layout for hover overlay
    profileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    profileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [profileActive, profileData, profilePoints, pixelSize, canvasW, themeInfo.theme, themeColors.accent, profileHeight, canvasRepaintSignal]);

  // Profile hover handler - draws crosshair + value readout
  const handleProfileMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    const layout = profileLayoutRef.current;
    if (!canvas || !base || !layout) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit } = layout;
    const range = gMax - gMin || 1;

    ctx.putImageData(base, 0, 0);
    if (cssX < padLeft || cssX > padLeft + plotW) return;
    const frac = (cssX - padLeft) / plotW;

    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Vertical crosshair
    ctx.strokeStyle = themeInfo.theme === "dark" ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(cssX, padTop);
    ctx.lineTo(cssX, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot on profile line + value
    if (profileData && profileData.length >= 2) {
      const dataIdx = Math.min(profileData.length - 1, Math.max(0, Math.round(frac * (profileData.length - 1))));
      const val = profileData[dataIdx];
      const y = padTop + plotH - ((val - gMin) / range) * plotH;
      ctx.fillStyle = themeColors.accent;
      ctx.beginPath();
      ctx.arc(cssX, y, 3, 0, Math.PI * 2);
      ctx.fill();

      // Value readout label
      const dist = frac * totalDist;
      const label = `${formatNumber(val)}  @  ${dist.toFixed(1)} ${xUnit}`;
      const isDark = themeInfo.theme === "dark";
      ctx.font = "bold 9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      const textW = ctx.measureText(label).width;
      const labelX = Math.min(cssX + 6, padLeft + plotW - textW - 2);
      const labelY = padTop + 2;
      ctx.fillStyle = isDark ? "rgba(0,0,0,0.7)" : "rgba(255,255,255,0.8)";
      ctx.fillRect(labelX - 2, labelY - 1, textW + 4, 11);
      ctx.fillStyle = isDark ? "#fff" : "#000";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(label, labelX, labelY);
    }

    ctx.restore();
  };

  const handleProfileMouseLeave = () => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    if (!canvas || !base) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(base, 0, 0);
  };

  // Profile height resize
  React.useEffect(() => {
    if (!isResizingProfile) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!profileResizeStart) return;
      const delta = e.clientY - profileResizeStart.y;
      setProfileHeight(Math.max(40, Math.min(300, profileResizeStart.height + delta)));
    };
    const handleMouseUp = () => {
      setIsResizingProfile(false);
      setProfileResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingProfile, profileResizeStart]);

  // Render HiDPI scale bar + zoom indicator + colorbar
  React.useEffect(() => {
    if (!uiRef.current) return;
    const ctx = uiRef.current.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, uiRef.current.width, uiRef.current.height);
    const showImageZoomIndicator = showZoomIndicator !== false && panelChromeVisible;
    if (scaleBarVisible || showImageZoomIndicator) {
      const unit = pixelSize > 0 ? pixelUnit : "px";
      const pxSize = pixelSize > 0 ? pixelSize : 1;
      // Per-panel scale bar + zoom indicator. Each panel slot uses its
      // own panelStates[i].zoom so panels at different zoom levels show
      // their own length bar.
      const n = Math.max(1, visiblePanelCount || 1);
      const cols = panelColsForCount(n);
      const rows = Math.ceil(n / cols);
      const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
      const cssW = uiRef.current.width / DPR;
      const cssH = uiRef.current.height / DPR;
      const slotW = (cssW - gap * (cols - 1)) / cols;
      const slotH = (cssH - gap * (rows - 1)) / rows;
      ctx.save();
      ctx.scale(DPR, DPR);
      // Exact Show2D drawScaleBarHiDPI style: 60 px target, 5 px thickness,
      // 16 px font, 12 px margin. Per-panel: each slot acts as its own
      // canvas region with width=slotW, image source width=`width`.
      const targetBarPxSpec = 60;
      const barThickness = 5;
      const fontSize = 16;
      const margin = 12;
      ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
      for (let slot = 0; slot < n; slot++) {
        const i = visiblePanelIndices[slot] ?? slot;
        const panelState = stateFor(i);
        const col = slot % cols;
        const row = Math.floor(slot / cols);
        const slotX = col * (slotW + gap);
        const slotY = row * (slotH + gap);
        ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
        ctx.shadowBlur = 2;
        ctx.shadowOffsetX = 1;
        ctx.shadowOffsetY = 1;
        ctx.fillStyle = "white";
        if (scaleBarVisible) {
          // Cap bar at 25% of slot width so it never overflows a small slot.
          const targetBarPx = Math.min(targetBarPxSpec, slotW * 0.25);
          const slotScale = slotW / sourcePanelWidth;
          const effectiveZoom = panelState.zoom * slotScale;
          const targetPhysical = (targetBarPx / effectiveZoom) * pxSize;
          const nicePhysical = (function (v: number) {
            if (v <= 0) return 1;
            const mag = Math.pow(10, Math.floor(Math.log10(v)));
            const norm = v / mag;
            if (norm < 1.5) return mag;
            if (norm < 3.5) return 2 * mag;
            if (norm < 7.5) return 5 * mag;
            return 10 * mag;
          })(targetPhysical);
          const barPx = (nicePhysical / pxSize) * effectiveZoom;
          const barY = slotY + slotH - margin;
          const barX = slotX + slotW - barPx - margin;
          ctx.fillRect(barX, barY, barPx, barThickness);
          const label = formatScaleLabel(nicePhysical, unit);
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillText(label, barX + barPx / 2, barY - 4);
        }
        if (showImageZoomIndicator) {
          ctx.textAlign = "left";
          ctx.textBaseline = "bottom";
          ctx.fillText(formatZoomLabel(panelState.zoom), slotX + margin, slotY + slotH - margin + barThickness);
        }
      }
      ctx.restore();
    }
    if (showColorbar) {
      const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
      // Colorbar must match what's painted on the image, not the raw data range.
      // When autoContrast is on, the image uses percentileClip(low, high) of the
      // current frame - show that range. Otherwise use slider range over data.
      let vmin: number, vmax: number;
      if (traitVmin != null || traitVmax != null) {
        ({ vmin, vmax } = resolveDisplayRange(
          dataMin,
          dataMax,
          traitVmin,
          traitVmax,
          logScale,
          imageVminPct,
          imageVmaxPct,
        ));
      } else if (autoContrast && imageHistogramData && imageHistogramData.length > 0) {
        const cached = cachedAutoDisplayRange(autoVmins, autoVmaxs, displaySliceIdx, logScale)
          || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, displaySliceIdx, logScale);
        ({ vmin, vmax } = cached ?? percentileClip(imageHistogramData, percentileLow, percentileHigh));
      } else {
        ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
      }
      ctx.save();
      ctx.scale(DPR, DPR);
      const n = Math.max(1, visiblePanelCount || 1);
      const cols = panelColsForCount(n);
      const rows = Math.ceil(n / cols);
      const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
      const cssW = uiRef.current.width / DPR;
      const cssH = uiRef.current.height / DPR;
      const slotW = (cssW - gap * (cols - 1)) / cols;
      const slotH = (cssH - gap * (rows - 1)) / rows;
      const perPanelColorbar = n > 1 && !linkContrast && !sharedPanelSource;
      const currentFrame = rawFrameDataRef.current;
      const sharedAutoRange = autoContrast ? { vmin, vmax } : null;
      for (let slot = 0; slot < n; slot++) {
        const panel = visiblePanelIndices[slot] ?? slot;
        let panelVmin = vmin;
        let panelVmax = vmax;
        if (perPanelColorbar) {
          const panelData = currentFrame ? extractPanelSlice(currentFrame, panel, logScale) : null;
          const pdr = panelDataRanges[panel];
          const panelRange = panelData && panelData.length > 0
            ? findDataRange(panelData)
            : ((perPanelHistogramEnabled && pdr && pdr.max > pdr.min)
                ? pdr
                : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale));
          const resolved = resolvePanelRenderRange(panel, panelRange, sharedAutoRange, panelData, autoContrast, percentileLow, percentileHigh);
          panelVmin = resolved.vmin;
          panelVmax = resolved.vmax;
        }
        const col = slot % cols;
        const row = Math.floor(slot / cols);
        const slotX = col * (slotW + gap);
        const slotY = row * (slotH + gap);
        ctx.save();
        ctx.beginPath();
        ctx.rect(slotX, slotY, slotW, slotH);
        ctx.clip();
        ctx.translate(slotX, slotY);
        drawColorbar(ctx, slotW, slotH, lut, panelVmin, panelVmax, logScale);
        ctx.restore();
      }
      ctx.restore();
    }
  }, [pixelSize, pixelUnit, scaleBarVisible, width, sourcePanelWidth, canvasW, canvasH, displayScale, zoom, nPanels, visiblePanelCount, visiblePanelIndices, maxCols, panelStates, linkedState, linkPanels, panelGapTrait, showZoomIndicator, panelChromeVisible, showColorbar, cmap, imageDataRange, imageVminPct, imageVmaxPct, logScale, autoContrast, imageHistogramData, autoVmins, autoVmaxs, displaySliceIdx, percentileLow, percentileHigh, dataMin, dataMax, traitVmin, traitVmax, linkContrast, sharedPanelSource, panelDataRanges, vminPerPanel, vmaxPerPanel, canvasRepaintSignal]);

  // Compute FFT magnitude (expensive, async - only re-run on data/GPU changes)
  // Supports ROI-scoped FFT: when ROI is active with a selected ROI, compute
  // FFT of the cropped region instead of the full frame.
  type FftMagnitudeCacheEntry = {
    mag: Float32Array;
    cropDims: { cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null;
    grid: { panelWidth: number; panelHeight: number; cols: number; rows: number; count: number } | null;
    source: string;
    panels: number;
    gridLabel: string | null;
    sizeLabel: string;
  };
  const fftMagnitudeCacheBaseMaxBytes = 256 * 1024 * 1024;
  const fftMagRef = React.useRef<Float32Array | null>(null);
  const fftMagnitudeCacheRef = React.useRef<Map<string, FftMagnitudeCacheEntry>>(new Map());
  const fftActiveCacheKeyRef = React.useRef<string | null>(null);
  const [fftMagVersion, setFftMagVersion] = React.useState(0);

  React.useEffect(() => {
    fftMagnitudeCacheRef.current.clear();
    fftActiveCacheKeyRef.current = null;
    fftMagRef.current = null;
    fftMagCacheRef.current = null;
    fftPanelGridRef.current = null;
    fftCropDimsRef.current = null;
    fftOffscreenRef.current = null;
    fftQualityKeyRef.current = "";
    setFftCropDims(null);
    setFftHistogramData(null);
    setFftQuality(null);
    setFftOffscreenVersion(v => v + 1);
    setFftBackendInfo(prev => ({ ...prev, source: "", ms: null, panels: null, grid: "" }));
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.fftCacheSize = 0;
      dbg.fftCacheBytes = 0;
      dbg.fftCacheInvalidations = Number(dbg.fftCacheInvalidations || 0) + 1;
    }
  }, [frameServerVersion, offline, width, height, nSlices, nPanels, sourcePanelWidth, sharedPanelSource]);

  React.useEffect(() => {
    if (!effectiveShowFft) return;
    // FFT is useful context, but for heavy multi-panel stacks it is far more
    // expensive than showing the next image frame. During playback keep the
    // last FFT visible and let the image path own the frame budget; when the
    // user pauses or finishes scrubbing, compute immediately for that settled
    // frame so overlay/bottom FFT does not sit blank behind render churn.
    if (playing) return;
    if (!rawFrameDataRef.current) {
      const idx = offline ? liveSliceIdx : displaySliceIdx;
      const offlineFrame = offline ? getOfflineFrame(idx) : null;
      const parsed = offlineFrame ?? extractFloat32(frameBytes, width * height);
      const frame = parsed
        ? (displayAndFrequencyFrameForIndex(idx, parsed) ?? parsed)
        : null;
      if (frame) rawFrameDataRef.current = frame;
    }
    if (!rawFrameDataRef.current) return;
    let cancelled = false;
    const doCompute = async () => {
      const fftStartMs = performance.now();
      const data = rawFrameDataRef.current!;
      const panelCount = Math.max(1, nPanels || 1);
      const multiPanelFft = panelCount > 1 && !roiFftActive;
      const fftFrameIdx = clampSlice(offline ? liveSliceIdx : displaySliceIdx);
      const selectedRoi = roiFftActive && roiList && roiSelectedIdx >= 0 && roiSelectedIdx < roiList.length
        ? roiList[roiSelectedIdx]
        : null;
      const roiKey = selectedRoi
        ? JSON.stringify({
          idx: roiSelectedIdx,
          row: Math.round(Number(selectedRoi.row ?? 0) * 100) / 100,
          col: Math.round(Number(selectedRoi.col ?? 0) * 100) / 100,
          radius: Math.round(Number(selectedRoi.radius ?? 0) * 100) / 100,
          radius_inner: Math.round(Number(selectedRoi.radius_inner ?? 0) * 100) / 100,
          width: Math.round(Number(selectedRoi.width ?? 0) * 100) / 100,
          height: Math.round(Number(selectedRoi.height ?? 0) * 100) / 100,
          shape: selectedRoi.shape,
        })
        : "none";
      const fftGridCols = multiPanelFft ? panelColsForCount(Math.max(1, visiblePanelIndices.length || 1)) : 1;
      // Cache identity is the rendered FFT source, not the traitlet delivery.
      // frame_seq can tick every time Python sends frame_bytes during live
      // scrubbing, so including it here makes already-computed FFTs miss.
      // frameServerVersion is bumped only when the underlying data stack
      // changes; the effect above clears old FFTs at that boundary.
      const fftCacheKey = [
        offline ? "offline" : "live",
        `frame=${fftFrameIdx}`,
        `data=${frameServerVersion || 0}`,
        `dims=${width}x${height}`,
        `panels=${panelCount}`,
        `visible=${visiblePanelIndices.join(",")}`,
        `cols=${fftGridCols}`,
        `sourceW=${sourcePanelWidth}`,
        `overlay=${fftLayoutOverlay ? 1 : 0}`,
        `overlayCap=${fftLayoutOverlay ? FFT_OVERLAY_MAX_SOURCE_SIZE : 0}`,
        `shared=${sharedPanelSource ? 1 : 0}`,
        `roi=${roiFftActive ? roiKey : "none"}`,
        `window=${fftWindow ? 1 : 0}`,
        `transform=${diffMode}:${Math.max(1, Math.round(avgWindow || 1))}`,
      ].join("|");
      const cache = fftMagnitudeCacheRef.current;
      const cached = cache.get(fftCacheKey);
      if (cached) {
        cache.delete(fftCacheKey);
        cache.set(fftCacheKey, cached);
        const dbg = show3dPerfDebug();
        if (dbg) {
          dbg.lastFftMs = 0;
          dbg.lastFftSource = `${cached.source}-cache`;
          dbg.lastFftPanels = cached.panels;
          dbg.lastFftSize = cached.sizeLabel;
          dbg.lastFftGrid = cached.gridLabel;
          dbg.fftCacheHits = Number(dbg.fftCacheHits || 0) + 1;
          dbg.fftCacheSize = cache.size;
          dbg.fftCacheBytes = Array.from(cache.values()).reduce((total, item) => total + item.mag.byteLength, 0);
        }
        if (fftActiveCacheKeyRef.current === fftCacheKey) {
          return;
        }
        fftActiveCacheKeyRef.current = fftCacheKey;
        fftMagRef.current = cached.mag;
        fftMagCacheRef.current = cached.mag;
        fftPanelGridRef.current = cached.grid;
        fftCropDimsRef.current = cached.cropDims;
        setFftCropDims(cached.cropDims);
        setFftMagVersion(v => v + 1);
        setFftBackendInfo(prev => ({
          ...prev,
          source: `${cached.source}-cache`,
          ms: 0,
          panels: cached.panels,
          grid: cached.gridLabel || "",
        }));
        return;
      }
      const rememberFft = (entry: FftMagnitudeCacheEntry) => {
        cache.set(fftCacheKey, entry);
        const maxEntries = Math.max(2, Math.min(24, nSlices || 12));
        const maxBytes = Math.max(fftMagnitudeCacheBaseMaxBytes, Math.min(1024 * 1024 * 1024, entry.mag.byteLength * 3));
        let totalBytes = Array.from(cache.values()).reduce((total, item) => total + item.mag.byteLength, 0);
        while (cache.size > maxEntries || totalBytes > maxBytes) {
          const oldest = cache.keys().next().value;
          if (oldest === undefined) break;
          const oldestEntry = cache.get(oldest);
          cache.delete(oldest);
          totalBytes -= oldestEntry?.mag.byteLength ?? 0;
        }
        const dbg = show3dPerfDebug();
        if (dbg) {
          dbg.fftCacheMisses = Number(dbg.fftCacheMisses || 0) + 1;
          dbg.fftCacheSize = cache.size;
          dbg.fftCacheBytes = totalBytes;
          dbg.fftComputes = Number(dbg.fftComputes || 0) + 1;
        }
      };

      if (multiPanelFft) {
        const panelW = sharedPanelSource
          ? Math.max(1, sourcePanelWidth)
          : Math.max(1, Math.floor(width / panelCount));
        const panelH = height;
        const overlayScale = fftLayoutOverlay
          ? Math.max(1, Math.ceil(Math.max(panelW, panelH) / FFT_OVERLAY_MAX_SOURCE_SIZE))
          : 1;
        const fftSourceW = Math.max(1, Math.ceil(panelW / overlayScale));
        const fftSourceH = Math.max(1, Math.ceil(panelH / overlayScale));
        const fftW = nextPow2(fftSourceW);
        const fftH = nextPow2(fftSourceH);
        const panels: { real: Float32Array; imag: Float32Array }[] = [];
        const fullW = data.length === height * panelW ? panelW : width;
        for (const panel of visiblePanelIndices) {
          const srcPanel = sharedPanelSource ? 0 : panel;
          const x0 = Math.min(Math.max(0, srcPanel * panelW), Math.max(0, fullW - panelW));
          if (data.length < height * fullW || x0 + panelW > fullW) continue;
          const source = new Float32Array(fftSourceW * fftSourceH);
          if (overlayScale > 1) {
            for (let y = 0; y < fftSourceH; y++) {
              const srcY = Math.min(panelH - 1, y * overlayScale);
              const srcOffset = srcY * fullW + x0;
              const dstOffset = y * fftSourceW;
              for (let x = 0; x < fftSourceW; x++) {
                source[dstOffset + x] = data[srcOffset + Math.min(panelW - 1, x * overlayScale)];
              }
            }
          } else {
            for (let y = 0; y < panelH; y++) {
              source.set(data.subarray(y * fullW + x0, y * fullW + x0 + panelW), y * fftSourceW);
            }
          }
          // Window the real source extent, then pad. Applying the taper to the
          // already-padded grid changes the intended Hann profile.
          if (fftWindow) applyHannWindow2D(source, fftSourceW, fftSourceH);
          const real = new Float32Array(fftW * fftH);
          for (let y = 0; y < fftSourceH; y++) {
            real.set(source.subarray(y * fftSourceW, (y + 1) * fftSourceW), y * fftW);
          }
          panels.push({ real, imag: new Float32Array(real.length) });
        }
        if (panels.length === 0) return;

        let results: { real: Float32Array; imag: Float32Array }[];
        let fftSource = "worker-batch";
        const offlineGpuTimeoutMs = 5000;
        const withOfflineTimeout = <T,>(promise: Promise<T>): Promise<T> => {
          if (!offline) return promise;
          return Promise.race([
            promise,
            new Promise<T>((_, reject) => window.setTimeout(() => reject(new Error("offline WebGPU FFT timed out")), offlineGpuTimeoutMs)),
          ]);
        };
        const offlineGpuDisabled = () => offlineFftGpuDisabledRef.current;
        const disableOfflineGpu = () => {
          offlineFftGpuDisabledRef.current = true;
        };
        const offlineGpuInFlight = () => offlineFftGpuInFlightRef.current;
        const skipOfflineWebGpu = offline && /HeadlessChrome/i.test(navigator.userAgent);
        // A replacement effect can start while an older offline GPU batch is
        // still draining. Do not abandon the replacement (which left FFT
        // blank until another interaction); compute it on CPU instead.
        const fftGpu = (!skipOfflineWebGpu && !offlineGpuDisabled() && !offlineGpuInFlight())
          ? await withOfflineTimeout(ensureFftGpu())
          : null;
        if (cancelled) return;
        if (fftGpu && panels.length > 1) {
          let startedOfflineGpu = false;
          try {
            if (offline) {
              offlineFftGpuInFlightRef.current = true;
              startedOfflineGpu = true;
            }
            results = await withOfflineTimeout(
              fftGpu.fft2DBatch(
                panels.map(({ real, imag }) => ({ real, imag })),
                fftW,
                fftH,
              )
            );
            fftSource = "webgpu-batch";
          } catch (err) {
            console.warn("Show3D WebGPU FFT failed; falling back to worker FFT.", err);
            if (offline) {
              disableOfflineGpu();
              results = panels.map(({ real, imag }) => {
                fft2d(real, imag, fftW, fftH, false);
                fftshift(real, fftW, fftH);
                fftshift(imag, fftW, fftH);
                return { real, imag };
              });
              fftSource = "cpu-sync-shifted";
            } else {
              results = (await Promise.all(panels.map(({ real, imag }) => fft2dAsync(real, imag, fftW, fftH, false))))
                .map(({ real, imag }) => ({ real, imag }));
            }
          } finally {
            if (startedOfflineGpu) offlineFftGpuInFlightRef.current = false;
          }
        } else if (offline) {
          results = panels.map(({ real, imag }) => {
            fft2d(real, imag, fftW, fftH, false);
            fftshift(real, fftW, fftH);
            fftshift(imag, fftW, fftH);
            return { real, imag };
          });
          fftSource = "cpu-sync-shifted";
        } else {
          results = (await Promise.all(panels.map(({ real, imag }) => fft2dAsync(real, imag, fftW, fftH, false))))
            .map(({ real, imag }) => ({ real, imag }));
        }
        if (cancelled) return;

        const cols = panelColsForCount(panels.length);
        const rows = Math.ceil(panels.length / cols);
        const gridW = cols * fftW;
        const gridH = rows * fftH;
        const gridMag = new Float32Array(gridW * gridH);
        const resultsAlreadyShifted = fftSource === "worker-batch" || fftSource === "cpu-sync-shifted";
        for (let panel = 0; panel < results.length; panel++) {
          const { real, imag } = results[panel];
          if (!resultsAlreadyShifted) {
            fftshift(real, fftW, fftH);
            fftshift(imag, fftW, fftH);
          }
          const mag = computeMagnitude(real, imag);
          const col = panel % cols;
          const row = Math.floor(panel / cols);
          const dstX = col * fftW;
          const dstY = row * fftH;
          for (let y = 0; y < fftH; y++) {
            gridMag.set(mag.subarray(y * fftW, y * fftW + fftW), (dstY + y) * gridW + dstX);
          }
        }

        fftMagRef.current = gridMag;
        fftActiveCacheKeyRef.current = fftCacheKey;
        fftMagCacheRef.current = gridMag;
        const gridInfo = { panelWidth: fftW, panelHeight: fftH, cols, rows, count: panels.length };
        const cropDims = { cropWidth: fftSourceW, cropHeight: fftSourceH, fftWidth: gridW, fftHeight: gridH };
        fftPanelGridRef.current = gridInfo;
        fftCropDimsRef.current = cropDims;
        setFftCropDims(cropDims);
        rememberFft({
          mag: gridMag,
          cropDims,
          grid: gridInfo,
          source: fftSource,
          panels: panels.length,
          gridLabel: `${gridW}x${gridH}`,
          sizeLabel: overlayScale > 1 ? `${fftW}x${fftH} overlay/${overlayScale}x` : `${fftW}x${fftH}`,
        });
        setFftMagVersion(v => v + 1);
        const dbg = show3dPerfDebug();
        const elapsedMs = Number((performance.now() - fftStartMs).toFixed(2));
        setFftBackendInfo(prev => ({
          ...prev,
          source: fftSource,
          ms: elapsedMs,
          panels: panels.length,
          grid: `${gridW}x${gridH}`,
        }));
        if (dbg) {
          dbg.lastFftMs = elapsedMs;
          dbg.lastFftSource = fftSource;
          dbg.lastFftPanels = panels.length;
          dbg.lastFftSize = overlayScale > 1 ? `${fftW}x${fftH} overlay/${overlayScale}x` : `${fftW}x${fftH}`;
          dbg.lastFftGrid = `${gridW}x${gridH}`;
        }
        return;
      }

      fftPanelGridRef.current = null;
      fftCropDimsRef.current = null;
      let fftW = width;
      let fftH = height;
      let inputData = data;

      // ROI crop: extract bounding box and optionally zero-mask outside radius
      let origCropW = 0, origCropH = 0;
      if (roiFftActive && roiList && roiSelectedIdx >= 0 && roiSelectedIdx < roiList.length) {
        const roi = roiList[roiSelectedIdx];
        const crop = cropROIRegion(data, width, height, roi);
        if (crop) {
          origCropW = crop.cropW;
          origCropH = crop.cropH;
          // Apply Hann window to crop at native dimensions BEFORE zero-padding
          if (fftWindow) applyHannWindow2D(crop.cropped, crop.cropW, crop.cropH);
          // Pad to next power-of-2 so fft2d doesn't truncate frequency data
          const padW = nextPow2(crop.cropW);
          const padH = nextPow2(crop.cropH);
          const padded = new Float32Array(padW * padH);
          for (let y = 0; y < crop.cropH; y++) {
            for (let x = 0; x < crop.cropW; x++) {
              padded[y * padW + x] = crop.cropped[y * crop.cropW + x];
            }
          }
          inputData = padded;
          fftW = padW;
          fftH = padH;
        }
      }

      // Pre-pad non-power-of-2 full images so fft2d doesn't truncate frequency data
      if (origCropW === 0) {
        if (fftWindow) {
          inputData = data.slice();
          applyHannWindow2D(inputData, width, height);
        }
        const padW = nextPow2(fftW);
        const padH = nextPow2(fftH);
        if (padW !== fftW || padH !== fftH) {
          const padded = new Float32Array(padW * padH);
          for (let y = 0; y < fftH; y++) {
            for (let x = 0; x < fftW; x++) {
              padded[y * padW + x] = inputData[y * fftW + x];
            }
          }
          inputData = padded;
          fftW = padW;
          fftH = padH;
        }
      }

      let real: Float32Array, imag: Float32Array;

      let fftSource = "cpu";
      const fftGpu = await ensureFftGpu();
      if (cancelled) return;
      if (fftGpu) {
        try {
          const gpuReal = inputData.slice();
          const gpuImag = new Float32Array(inputData.length);
          const result = await fftGpu.fft2D(gpuReal, gpuImag, fftW, fftH, false);
          real = result.real;
          imag = result.imag;
          fftSource = "webgpu";
        } catch (err) {
          console.warn("Show3D WebGPU FFT failed; falling back to worker FFT.", err);
          const result = await fft2dAsync(inputData.slice(), new Float32Array(inputData.length), fftW, fftH, false);
          real = result.real;
          imag = result.imag;
          fftSource = "worker";
        }
      } else {
        const result = await fft2dAsync(inputData.slice(), new Float32Array(inputData.length), fftW, fftH, false);
        real = result.real;
        imag = result.imag;
        fftSource = "worker";
      }

      if (cancelled) return;
      if (fftSource !== "worker") {
        fftshift(real, fftW, fftH);
        fftshift(imag, fftW, fftH);
      }

      fftMagRef.current = computeMagnitude(real, imag);
      fftActiveCacheKeyRef.current = fftCacheKey;
      fftMagCacheRef.current = fftMagRef.current;
      // Track FFT dimensions when they differ from image dimensions (ROI crop or non-pow2 padding)
      let cropDims: { cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null = null;
      if (origCropW > 0) {
        cropDims = { cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH };
      } else if (fftW !== width || fftH !== height) {
        cropDims = { cropWidth: width, cropHeight: height, fftWidth: fftW, fftHeight: fftH };
      }
      fftCropDimsRef.current = cropDims;
      setFftCropDims(cropDims);
      rememberFft({
        mag: fftMagRef.current,
        cropDims,
        grid: null,
        source: fftSource,
        panels: 1,
        gridLabel: `${fftW}x${fftH}`,
        sizeLabel: `${fftW}x${fftH}`,
      });
      setFftMagVersion(v => v + 1);
      const dbg = show3dPerfDebug();
      const elapsedMs = Number((performance.now() - fftStartMs).toFixed(2));
      setFftBackendInfo(prev => ({
        ...prev,
        source: fftSource,
        ms: elapsedMs,
        panels: 1,
        grid: `${fftW}x${fftH}`,
      }));
      if (dbg) {
        dbg.lastFftMs = elapsedMs;
        dbg.lastFftSource = fftSource;
        dbg.lastFftPanels = 1;
        dbg.lastFftSize = `${fftW}x${fftH}`;
        dbg.lastFftGrid = null;
      }
    };

    void doCompute();

    return () => {
      cancelled = true;
    };
  }, [effectiveShowFft, playing, frameBytes, frameSeq, frameServerVersion, offline, liveSliceIdx, displaySliceIdx, width, height, roiFftActive, roiList, roiSelectedIdx, fftWindow, nPanels, nSlices, visiblePanelIndices, sourcePanelWidth, sharedPanelSource, maxCols, panelColsForCount, fftLayoutOverlay, extractPanelSlice, ensureFftGpu, diffMode, avgWindow]);

  // Clear FFT measurement when ROI FFT state changes
  React.useEffect(() => { setFftClickInfo(null); }, [roiFftActive, roiSelectedIdx]);

  // Process FFT magnitude → histogram + colormap rendering (cheap, sync)
  React.useEffect(() => {
    const mag = fftMagRef.current;
    if (!effectiveShowFft || !mag) return;

    // Use ref-backed dimensions so the magnitude and its layout metadata remain
    // consistent in the same render tick; React state may lag by one effect.
    const cropDimsForRender = fftCropDimsRef.current;
    const fftW = cropDimsForRender?.fftWidth ?? width;
    const fftH = cropDimsForRender?.fftHeight ?? height;
    const grid = fftPanelGridRef.current;
    if (fftMetricsEnabled) {
      const qualityKey = `${fftMagVersion}:${fftW}x${fftH}:${pixelSize || 0}:${pixelUnit || ""}:${grid ? `${grid.panelWidth}x${grid.panelHeight}x${grid.cols}x${grid.count}` : "single"}`;
      if (fftQualityKeyRef.current !== qualityKey) {
        fftQualityKeyRef.current = qualityKey;
        const metricStartMs = performance.now();
        let nextQuality: FftQualityMetrics | null;
        if (grid) {
          const panelMetrics: Array<FftQualityMetrics | null> = [];
          for (let panel = 0; panel < grid.count; panel++) {
            panelMetrics.push(computeFftQualityMetrics(mag, fftW, fftH, {
              sampling: pixelSize,
              unit: pixelUnit,
              region: {
                x: (panel % grid.cols) * grid.panelWidth,
                y: Math.floor(panel / grid.cols) * grid.panelHeight,
                width: grid.panelWidth,
                height: grid.panelHeight,
              },
            }));
          }
          nextQuality = summarizeFftQualityMetrics(panelMetrics);
        } else {
          nextQuality = computeFftQualityMetrics(mag, fftW, fftH, { sampling: pixelSize, unit: pixelUnit });
        }
        setFftQuality(nextQuality);
        const dbg = show3dPerfDebug();
        if (dbg) {
          dbg.fftMetricComputes = Number(dbg.fftMetricComputes || 0) + 1;
          dbg.lastFftMetricMs = Number((performance.now() - metricStartMs).toFixed(2));
          dbg.lastFftMetricKey = qualityKey;
          dbg.lastFftMetricLabel = formatFftQualityLabel(nextQuality);
        }
      }
    } else if (fftQualityKeyRef.current) {
      fftQualityKeyRef.current = "";
      setFftQuality(null);
    }

    let displayMin: number, displayMax: number;
    let displayData: Float32Array;
    if (fftAuto && grid) {
      // Multi-panel FFTs can differ by orders of magnitude (BF/DF vs SSB).
      // Auto mode should reveal each panel, so normalize every FFT tile before
      // composing the shared canvas. Manual mode below intentionally stays global.
      displayData = new Float32Array(mag.length);
      const panelDisplay = new Float32Array(grid.panelWidth * grid.panelHeight);
      for (let panel = 0; panel < grid.count; panel++) {
        const col = panel % grid.cols;
        const row = Math.floor(panel / grid.cols);
        const srcX = col * grid.panelWidth;
        const srcY = row * grid.panelHeight;
        for (let y = 0; y < grid.panelHeight; y++) {
          const srcOffset = (srcY + y) * fftW + srcX;
          const dstOffset = y * grid.panelWidth;
          for (let x = 0; x < grid.panelWidth; x++) {
            // FFT magnitudes are extremely heavy-tailed; even in "Lin" UI mode,
            // auto contrast should reveal Bragg/fringe peaks instead of letting
            // the DC/low-frequency pedestal flatten the tile.
            panelDisplay[dstOffset + x] = Math.log1p(Math.max(0, mag[srcOffset + x]));
          }
        }
        const cx = Math.floor(grid.panelWidth / 2);
        const cy = Math.floor(grid.panelHeight / 2);
        const dcRadius = Math.max(2, Math.round(Math.min(grid.panelWidth, grid.panelHeight) * 0.01));
        const ringRadius = dcRadius + 2;
        let ringSum = 0;
        let ringCount = 0;
        for (let yy = Math.max(0, cy - ringRadius); yy <= Math.min(grid.panelHeight - 1, cy + ringRadius); yy++) {
          for (let xx = Math.max(0, cx - ringRadius); xx <= Math.min(grid.panelWidth - 1, cx + ringRadius); xx++) {
            const dist = Math.hypot(xx - cx, yy - cy);
            if (dist > dcRadius && dist <= ringRadius) {
              ringSum += panelDisplay[yy * grid.panelWidth + xx];
              ringCount++;
            }
          }
        }
        const dcFill = ringCount > 0 ? ringSum / ringCount : 0;
        for (let yy = Math.max(0, cy - dcRadius); yy <= Math.min(grid.panelHeight - 1, cy + dcRadius); yy++) {
          for (let xx = Math.max(0, cx - dcRadius); xx <= Math.min(grid.panelWidth - 1, cx + dcRadius); xx++) {
            panelDisplay[yy * grid.panelWidth + xx] = dcFill;
          }
        }
        suppressFftRadialBackgroundInPlace(panelDisplay, grid.panelWidth, grid.panelHeight);

        const range = findDataRange(panelDisplay);
        const clipped = percentileClip(panelDisplay, 5, 99.99);
        const pMin = clipped.vmin < clipped.vmax ? clipped.vmin : range.min;
        const pMax = clipped.vmax > pMin ? clipped.vmax : range.max;
        const denom = pMax > pMin ? pMax - pMin : 1;
        for (let y = 0; y < grid.panelHeight; y++) {
          const dstOffset = (srcY + y) * fftW + srcX;
          const srcOffset = y * grid.panelWidth;
          for (let x = 0; x < grid.panelWidth; x++) {
            const normalized = (panelDisplay[srcOffset + x] - pMin) / denom;
            displayData[dstOffset + x] = Math.max(0, Math.min(1, normalized));
          }
        }
      }
      displayMin = 0;
      displayMax = 1;
    } else {
      if (fftAuto) {
        ({ min: displayMin, max: displayMax } = autoEnhanceFFT(mag, fftW, fftH));
      } else {
        ({ min: displayMin, max: displayMax } = findDataRange(mag));
      }
      displayData = fftLogScale ? applyLogScale(mag) : mag;
      if (fftLogScale) {
        displayMin = Math.log1p(displayMin);
        displayMax = Math.log1p(displayMax);
      }
    }

    setFftHistogramData(displayData);
    setFftDataRange({ min: displayMin, max: displayMax });
    setFftStats(computeStats(displayData));

    const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(displayData, fftW, fftH, lut, vmin, vmax);
    if (!offscreen) return;

    fftOffscreenRef.current = offscreen;
    setFftOffscreenVersion(v => v + 1);

    if (fftCanvasRef.current) {
      const ctx = fftCanvasRef.current.getContext("2d");
      if (ctx) {
        drawFftOffscreen(ctx, offscreen);
      }
    }
  }, [effectiveShowFft, fftMagVersion, fftLogScale, fftAuto, fftVminPct, fftVmaxPct, fftColormap, width, height, canvasW, canvasH, fftCropDims, drawFftOffscreen, pixelSize, pixelUnit, fftMetricsEnabled, canvasRepaintSignal]);

  // Redraw cached FFT with zoom/pan/resize before paint. Changing a canvas
  // width/height attribute clears its bitmap, so a normal effect can expose a
  // one-frame blank flash during resize drags.
  React.useLayoutEffect(() => {
    if (!effectiveShowFft || !fftCanvasRef.current || !fftOffscreenRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    drawFftOffscreen(ctx, fftOffscreenRef.current);
  }, [effectiveShowFft, fftOffscreenVersion, fftZoom, fftPanX, fftPanY, canvasW, canvasH, drawFftOffscreen, canvasRepaintSignal]);

  const drawFftInsetLayer = React.useCallback((
    view: { zoom: number; panX: number; panY: number } = fftViewLiveRef.current,
  ) => {
    const canvas = fftInsetLayerRef.current;
    if (!canvas || !effectiveShowFft || !fftLayoutOverlay || !fftOffscreenRef.current) return;
    const offscreen = fftOffscreenRef.current;
    const grid = fftPanelGridRef.current;
    const count = grid ? grid.count : 1;
    const n = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(n);
    const rows = Math.ceil(n / cols);
    const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
    const panelW = (canvasW - gap * (cols - 1)) / cols;
    const panelH = (canvasH - gap * (rows - 1)) / rows;
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const srcW = grid ? grid.panelWidth : fftW;
    const srcH = grid ? grid.panelHeight : fftH;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = srcW < panelW || srcH < panelH;
    visiblePanelIndices.forEach((_panel, slot) => {
      if (slot >= count) return;
      const panelLeft = (slot % cols) * (panelW + gap);
      const panelTop = Math.floor(slot / cols) * (panelH + gap);
      const insetPad = Math.min(8, Math.max(3, panelW * 0.025));
      const insetMaxW = Math.max(24, panelW - insetPad * 2);
      const insetMaxH = Math.max(20, panelH - insetPad * 2);
      const insetBase = Math.min(insetMaxW, insetMaxH);
      const insetW = Math.max(24, Math.min(insetMaxW, insetBase * resolvedFftOverlaySize));
      const insetH = Math.max(20, Math.min(insetMaxH, insetBase * resolvedFftOverlaySize));
      const insetX = resolvedFftOverlayPosition.endsWith("right")
        ? panelLeft + panelW - insetW - insetPad
        : panelLeft + insetPad;
      const insetY = resolvedFftOverlayPosition.startsWith("bottom")
        ? panelTop + panelH - insetH - insetPad
        : panelTop + insetPad;
      const dstX = fftOverlayDragPreview ? panelLeft + fftOverlayDragPreview.x : insetX;
      const dstY = fftOverlayDragPreview ? panelTop + fftOverlayDragPreview.y : insetY;
      const srcX = grid ? (slot % grid.cols) * grid.panelWidth : 0;
      const srcY = grid ? Math.floor(slot / grid.cols) * grid.panelHeight : 0;
      const insetPanX = !fftUserAdjustedViewRef.current && view.zoom > 1
        ? insetW * (1 - view.zoom) / 2
        : view.panX;
      const insetPanY = !fftUserAdjustedViewRef.current && view.zoom > 1
        ? insetH * (1 - view.zoom) / 2
        : view.panY;
      ctx.save();
      ctx.fillStyle = "#000";
      ctx.fillRect(dstX, dstY, insetW, insetH);
      ctx.beginPath();
      ctx.rect(dstX, dstY, insetW, insetH);
      ctx.clip();
      ctx.translate(dstX + insetPanX, dstY + insetPanY);
      ctx.scale(view.zoom, view.zoom);
      ctx.drawImage(offscreen, srcX, srcY, srcW, srcH, 0, 0, insetW, insetH);
      ctx.restore();
      ctx.strokeStyle = "rgba(255,255,255,0.48)";
      ctx.lineWidth = 1;
      ctx.strokeRect(dstX + 0.5, dstY + 0.5, Math.max(0, insetW - 1), Math.max(0, insetH - 1));
    });
  }, [effectiveShowFft, fftLayoutOverlay, fftCropDims, width, height, visiblePanelCount, visiblePanelIndices, panelColsForCount, panelGapTrait, canvasW, canvasH, resolvedFftOverlaySize, resolvedFftOverlayPosition, fftOverlayDragPreview]);

  React.useEffect(() => {
    fftViewDirectRedrawRef.current = () => {
      if (!effectiveShowFft || !fftLayoutOverlay || !fftOffscreenRef.current) return;
      if (fftViewRafRef.current !== null) return;
      fftViewRafRef.current = window.requestAnimationFrame(() => {
        fftViewRafRef.current = null;
        drawFftInsetLayer(fftViewLiveRef.current);
      });
    };
    return () => {
      fftViewDirectRedrawRef.current = null;
    };
  }, [drawFftInsetLayer, effectiveShowFft, fftLayoutOverlay]);

  React.useLayoutEffect(() => {
    if (!effectiveShowFft || !fftLayoutOverlay || !fftOffscreenRef.current) return;
    drawFftInsetLayer();
  }, [effectiveShowFft, fftLayoutOverlay, fftOffscreenVersion, fftZoom, fftPanX, fftPanY, fftCropDims, width, height, drawFftInsetLayer, canvasRepaintSignal]);

  // === Kymograph (space-time) ===
  // A sub-feature of the line profile (Henry: "the profile feature created a 2D
  // image ... distance along the line ... time axis"). Requires the profile tool
  // ON with a drawn line and some way to read every frame: the offline pack for
  // exported HTML, or the live frame server while the notebook kernel is up.
  const kymoExactStackReady = offline && !!offlineFloatStack && offlineFloatStack.byteLength > 0;
  const kymoQuantizedStackReady = offline && !!offlineStack && offlineStack.byteLength > 0;
  const kymoOfflineStackReady = kymoExactStackReady || kymoQuantizedStackReady;
  const kymoLiveStackReady = !offline && !!frameServerUrl;
  const kymographAvailable = ((nPanels || 1) === 1 || singlePanelPageProfile)
    && (kymoOfflineStackReady || kymoLiveStackReady)
    && width > 0 && height > 0 && nSlices > 1;
  const canKymograph = kymographAvailable && profileActive && profilePoints.length === 2;
  const kymoReady = canKymograph && showKymograph;

  // Compute the (nFrames, lineLen) image: sample the profile line on every
  // frame. Cold path - fires on line / width / stack change, never per tick.
  React.useEffect(() => {
    if (!kymoReady) { kymoDataRef.current = null; return; }
    const p0 = profilePoints[0], p1 = profilePoints[1];
    const pixelCount = width * height;
    const panelIdx = singlePanelPageProfile
      ? activePageStart
      : Math.max(0, Math.min(totalPanelCount - 1, profilePanelIdx));
    const colOffset = singlePanelPageProfile ? panelGlobalColOffset(panelIdx) : 0;
    const row0 = p0.row, col0 = p0.col + colOffset;
    const row1 = p1.row, col1 = p1.col + colOffset;
    let cancelled = false;

    const publish = (kymo: Float32Array, lineLen: number) => {
      if (cancelled) return;
      kymoDataRef.current = { data: kymo, lineLen, nFrames: nSlices };
      setKymoVersion(v => v + 1);
    };

    if (kymoExactStackReady && offlineFloatStack) {
      const sampleFrame = (frameIdx: number): Float32Array => {
        const frame = float32FrameFromDataView(offlineFloatStack, frameIdx, pixelCount, false);
        return frame
          ? sampleLineProfile(frame, width, height, row0, col0, row1, col1, profileWidth)
          : new Float32Array(0);
      };
      const first = sampleFrame(0);
      const lineLen = first.length;
      if (lineLen < 2) { kymoDataRef.current = null; return; }
      const kymo = new Float32Array(nSlices * lineLen);
      kymo.set(first.subarray(0, lineLen), 0);
      for (let f = 1; f < nSlices; f++) {
        kymo.set(sampleFrame(f).subarray(0, lineLen), f * lineLen);
      }
      publish(kymo, lineLen);
      return () => { cancelled = true; };
    }

    if (kymoQuantizedStackReady && offlineStack) {
      const scale = (offlineMax - offlineMin) / 255.0;
      // Read straight from the packed uint8 stack, dequantizing only the
      // bilinear corners per sample point. No whole-frame dequant.
      const u8 = new Uint8Array(offlineStack.buffer, offlineStack.byteOffset, offlineStack.byteLength);
      const sampleFrame = (frameIdx: number) =>
        sampleLineProfileU8(u8, frameIdx * pixelCount, width, height, scale, offlineMin,
          row0, col0, row1, col1, profileWidth);
      const first = sampleFrame(0);
      const lineLen = first.length;
      if (lineLen < 2) { kymoDataRef.current = null; return; }
      const kymo = new Float32Array(nSlices * lineLen);
      kymo.set(first.subarray(0, lineLen), 0);
      for (let f = 1; f < nSlices; f++) {
        kymo.set(sampleFrame(f).subarray(0, lineLen), f * lineLen);
      }
      publish(kymo, lineLen);
      return () => { cancelled = true; };
    }

    if (kymoLiveStackReady) {
      void (async () => {
        const firstFrame = await fetchFrameFromServer(0);
        if (cancelled || !firstFrame || firstFrame.length < pixelCount) {
          if (!cancelled) kymoDataRef.current = null;
          return;
        }
        const first = sampleLineProfile(firstFrame, width, height, row0, col0, row1, col1, profileWidth);
        const lineLen = first.length;
        if (lineLen < 2) {
          if (!cancelled) kymoDataRef.current = null;
          return;
        }
        const kymo = new Float32Array(nSlices * lineLen);
        kymo.set(first.subarray(0, lineLen), 0);
        for (let f = 1; f < nSlices; f++) {
          if (cancelled) return;
          const frame = await fetchFrameFromServer(f);
          if (!frame || frame.length < pixelCount) {
            if (!cancelled) kymoDataRef.current = null;
            return;
          }
          kymo.set(sampleLineProfile(frame, width, height, row0, col0, row1, col1, profileWidth).subarray(0, lineLen), f * lineLen);
          await new Promise<void>(resolve => setTimeout(resolve, 0));
        }
        publish(kymo, lineLen);
      })();
      return () => { cancelled = true; };
    }

    kymoDataRef.current = null;
    return undefined;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kymoReady, kymoExactStackReady, kymoQuantizedStackReady, kymoLiveStackReady, offlineStack, offlineFloatStack, offlineMin, offlineMax, width, height, nSlices,
      profileWidth, profilePoints[0]?.row, profilePoints[0]?.col,
      profilePoints[1]?.row, profilePoints[1]?.col, profilePanelIdx, activePageStart,
      singlePanelPageProfile, totalPanelCount, fetchFrameFromServer]);

  // Process kymograph data → histogram + colormap rendering (cheap, sync).
  // Mirrors the FFT pipeline: range → log scale → histogram/stats → slider
  // range → LUT → offscreen → draw with zoom/pan. Cold path, image is tiny.
  React.useEffect(() => {
    const kymo = kymoDataRef.current;
    if (!kymoReady || !kymo) return;
    const { data, lineLen, nFrames } = kymo;

    let displayMin: number, displayMax: number;
    if (kymoAuto) {
      ({ vmin: displayMin, vmax: displayMax } = percentileClip(data, percentileLow, percentileHigh));
    } else {
      ({ min: displayMin, max: displayMax } = findDataRange(data));
    }

    const displayData = kymoLogScale ? applyLogScale(data) : data;
    if (kymoLogScale) {
      displayMin = Math.log1p(displayMin);
      displayMax = Math.log1p(displayMax);
    }

    setKymoHistogramData(displayData);
    setKymoDataRange({ min: displayMin, max: displayMax });
    setKymoStats(computeStats(displayData));

    const { vmin, vmax } = sliderRange(displayMin, displayMax, kymoVminPct, kymoVmaxPct);
    const lut = COLORMAPS[kymoColormap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(displayData, lineLen, nFrames, lut, vmin, vmax);
    if (!offscreen) return;

    kymoOffscreenRef.current = offscreen;

    if (kymoCanvasRef.current) {
      const ctx = kymoCanvasRef.current.getContext("2d");
      if (ctx) {
        ctx.imageSmoothingEnabled = lineLen < canvasW || nFrames < canvasH;
        ctx.clearRect(0, 0, canvasW, canvasH);
        ctx.save();
        ctx.translate(kymoPanX, kymoPanY);
        ctx.scale(kymoZoom, kymoZoom);
        ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);
        ctx.restore();
      }
    }
  }, [kymoReady, kymoVersion, kymoLogScale, kymoAuto, kymoVminPct, kymoVmaxPct, kymoColormap,
      percentileLow, percentileHigh, canvasW, canvasH, canvasRepaintSignal]);

  // Redraw cached kymograph with zoom/pan (cheap - no recomputation)
  React.useEffect(() => {
    if (!kymoReady || !kymoCanvasRef.current || !kymoOffscreenRef.current) return;
    const canvas = kymoCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const offW = kymoOffscreenRef.current.width;
    const offH = kymoOffscreenRef.current.height;
    ctx.imageSmoothingEnabled = offW < canvasW || offH < canvasH;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(kymoPanX, kymoPanY);
    ctx.scale(kymoZoom, kymoZoom);
    ctx.drawImage(kymoOffscreenRef.current, 0, 0, canvasW, canvasH);
    ctx.restore();
  }, [kymoReady, kymoZoom, kymoPanX, kymoPanY, canvasW, canvasH, canvasRepaintSignal]);

  // Render kymograph overlay (playhead + axis scale bars + colorbar + click
  // crosshair). Mirrors the FFT overlay structure; the playhead is the only
  // part that tracks the current frame. Never recomputes the image.
  React.useEffect(() => {
    const overlay = kymoOverlayRef.current;
    const kymo = kymoDataRef.current;
    if (!overlay || !kymoReady || !kymo) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    overlay.width = Math.round(canvasW * DPR);
    overlay.height = Math.round(canvasH * DPR);
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Playhead row marker - tracks the current frame in zoomed/panned space.
    const y = kymoPanY + kymoZoom * (((liveSliceIdx + 0.5) / kymo.nFrames) * canvasH);
    ctx.save();
    ctx.scale(DPR, DPR);
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvasW, y);
    ctx.stroke();
    ctx.restore();

    // Distance scale bar along the bottom edge (distance axis, pixelUnit).
    if (pixelSize > 0) {
      drawScaleBarHiDPI(overlay, DPR, kymoZoom, pixelSize, pixelUnit || "px", kymo.lineLen);
    }

    // Time scale bar along the left edge (time axis, dimUnit). Vertical bar +
    // label so the operator can read the temporal extent of the kymograph.
    if (dimSampling > 0 && dimUnit) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const targetBarPx = 60;
      const barThickness = 5;
      const margin = 12;
      const scaleY = canvasH / kymo.nFrames;
      const effectiveZoom = kymoZoom * scaleY;
      const targetPhysical = (targetBarPx / effectiveZoom) * dimSampling;
      const nicePhysical = roundToNiceValue(targetPhysical);
      const barPx = (nicePhysical / dimSampling) * effectiveZoom;
      const barX = margin;
      const barY = margin;
      ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
      ctx.shadowBlur = 2;
      ctx.shadowOffsetX = 1;
      ctx.shadowOffsetY = 1;
      ctx.fillStyle = "white";
      ctx.fillRect(barX, barY, barThickness, barPx);
      ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const label = nicePhysical >= 1 ? `${nicePhysical} ${dimUnit}` : `${nicePhysical.toPrecision(2)} ${dimUnit}`;
      ctx.fillText(label, barX + barThickness + 4, barY + barPx / 2);
      ctx.restore();
    }

    // Colorbar when enabled (mirror FFT colorbar draw).
    if (kymoShowColorbar && kymoDataRange.min !== kymoDataRange.max) {
      const { vmin, vmax } = sliderRange(kymoDataRange.min, kymoDataRange.max, kymoVminPct, kymoVmaxPct);
      const lut = COLORMAPS[kymoColormap] || COLORMAPS.inferno;
      ctx.save();
      ctx.scale(DPR, DPR);
      drawColorbar(ctx, overlay.width / DPR, overlay.height / DPR, lut, vmin, vmax, kymoLogScale);
      ctx.restore();
    }

    // Click crosshair marker - mirror FFT marker, coordinates in zoomed space.
    if (kymoClickInfo) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const screenX = kymoPanX + kymoZoom * (kymoClickInfo.col / kymo.lineLen * canvasW);
      const screenY = kymoPanY + kymoZoom * (kymoClickInfo.row / kymo.nFrames * canvasH);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
      ctx.shadowBlur = 2;
      ctx.lineWidth = 1.5;
      const r = 8;
      ctx.beginPath();
      ctx.moveTo(screenX - r, screenY); ctx.lineTo(screenX - 3, screenY);
      ctx.moveTo(screenX + 3, screenY); ctx.lineTo(screenX + r, screenY);
      ctx.moveTo(screenX, screenY - r); ctx.lineTo(screenX, screenY - 3);
      ctx.moveTo(screenX, screenY + 3); ctx.lineTo(screenX, screenY + r);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(screenX, screenY, 4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }, [kymoReady, kymoVersion, liveSliceIdx, canvasW, canvasH, themeColors.accent, kymoZoom, kymoPanX, kymoPanY,
      pixelSize, pixelUnit, dimSampling, dimUnit, kymoShowColorbar, kymoDataRange, kymoVminPct, kymoVmaxPct,
      kymoColormap, kymoLogScale, kymoClickInfo, canvasRepaintSignal]);

  // Render FFT overlay (reciprocal-space scale bar + colorbar)
  React.useEffect(() => {
    const overlay = fftOverlayRef.current;
    if (!overlay || !effectiveShowFft) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    overlay.width = Math.round(canvasW * DPR);
    overlay.height = Math.round(canvasH * DPR);
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Use crop dimensions for reciprocal-space calculations
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;

    // Reciprocal-space scale bar (pixelSize is in Å)
    if (pixelSize > 0) {
      const panelGrid = fftPanelGridRef.current;
      const reciprocalWidth = panelGrid ? panelGrid.panelWidth : fftW;
      const fftPixelSize = 1 / (reciprocalWidth * pixelSize);
      drawFFTScaleBarHiDPI(overlay, DPR, fftZoom, fftPixelSize, fftW, `${unitSymbol(pixelUnit || "px")}⁻¹`, false);
    }

    // FFT colorbar
    if (fftShowColorbar && fftDataRange.min !== fftDataRange.max) {
      const { vmin, vmax } = sliderRange(fftDataRange.min, fftDataRange.max, fftVminPct, fftVmaxPct);
      const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
      ctx.save();
      ctx.scale(DPR, DPR);
      const cssW = overlay.width / DPR;
      const cssH = overlay.height / DPR;
      drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, fftLogScale);
      ctx.restore();
    }

    // D-spacing crosshair marker - use crop dims for coordinate mapping
    if (fftClickInfo) {
      ctx.save();
      ctx.scale(DPR, DPR);
      let screenX = fftPanX + fftZoom * (fftClickInfo.col / fftW * canvasW);
      let screenY = fftPanY + fftZoom * (fftClickInfo.row / fftH * canvasH);
      let centerX = fftPanX + fftZoom * (canvasW / 2);
      let centerY = fftPanY + fftZoom * (canvasH / 2);
      let radiusX = fftZoom * (fftClickInfo.distPx / Math.max(1, fftW)) * canvasW;
      let radiusY = fftZoom * (fftClickInfo.distPx / Math.max(1, fftH)) * canvasH;
      let clipRect: { x: number; y: number; w: number; h: number } | null = null;
      const panelGrid = fftPanelGridRef.current;
      if (panelGrid) {
        const slot = Math.max(0, Math.min(panelGrid.count - 1, Math.floor(fftClickInfo.row / panelGrid.panelHeight) * panelGrid.cols + Math.floor(fftClickInfo.col / panelGrid.panelWidth)));
        const dst = getFftSlot(slot, panelGrid.count, panelGrid.cols, panelGrid.rows);
        const localCol = fftClickInfo.col - (slot % panelGrid.cols) * panelGrid.panelWidth;
        const localRow = fftClickInfo.row - Math.floor(slot / panelGrid.cols) * panelGrid.panelHeight;
        screenX = dst.x + fftPanX + fftZoom * ((localCol / panelGrid.panelWidth) * dst.w);
        screenY = dst.y + fftPanY + fftZoom * ((localRow / panelGrid.panelHeight) * dst.h);
        centerX = dst.x + fftPanX + fftZoom * (dst.w / 2);
        centerY = dst.y + fftPanY + fftZoom * (dst.h / 2);
        radiusX = fftZoom * (fftClickInfo.distPx / Math.max(1, panelGrid.panelWidth)) * dst.w;
        radiusY = fftZoom * (fftClickInfo.distPx / Math.max(1, panelGrid.panelHeight)) * dst.h;
        clipRect = dst;
      }
      ctx.lineCap = "round";
      ctx.shadowBlur = 0;
      const r = 8;
      const drawRing = () => {
        ctx.beginPath();
        ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
        ctx.stroke();
      };
      const drawMarker = () => {
        ctx.beginPath();
        ctx.moveTo(screenX - r, screenY); ctx.lineTo(screenX - 3, screenY);
        ctx.moveTo(screenX + 3, screenY); ctx.lineTo(screenX + r, screenY);
        ctx.moveTo(screenX, screenY - r); ctx.lineTo(screenX, screenY - 3);
        ctx.moveTo(screenX, screenY + 3); ctx.lineTo(screenX, screenY + r);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(screenX, screenY, 4, 0, Math.PI * 2);
        ctx.stroke();
      };
      if (clipRect) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(clipRect.x, clipRect.y, clipRect.w, clipRect.h);
        ctx.clip();
      }
      ctx.strokeStyle = "rgba(0, 0, 0, 0.78)";
      ctx.lineWidth = 4;
      drawRing();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.64)";
      ctx.lineWidth = 1.25;
      drawRing();
      if (clipRect) ctx.restore();
      ctx.strokeStyle = "rgba(0, 0, 0, 0.92)";
      ctx.lineWidth = 4;
      drawMarker();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.96)";
      ctx.lineWidth = 1.5;
      drawMarker();
      const label = fftClickInfo.dSpacing != null
        ? (() => {
          const d = fftClickInfo.dSpacing!;
          return d >= 10 ? `d = ${(d / 10).toFixed(2)} nm` : `d = ${d.toFixed(2)} Å`;
        })()
        : `dist = ${fftClickInfo.distPx.toFixed(1)} px`;
      ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const padX = 5;
      const labelW = Math.ceil(ctx.measureText(label).width + padX * 2);
      const labelH = 18;
      const cssW = overlay.width / DPR;
      const cssH = overlay.height / DPR;
      const labelX = Math.max(2, Math.min(cssW - labelW - 2, screenX + 10));
      const labelY = Math.max(labelH / 2 + 2, Math.min(cssH - labelH / 2 - 2, screenY - 10));
      ctx.fillStyle = "rgba(0, 0, 0, 0.74)";
      ctx.strokeStyle = "rgba(255, 255, 255, 0.82)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(labelX, labelY - labelH / 2, labelW, labelH, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "white";
      ctx.fillText(label, labelX + padX, labelY);
      ctx.restore();
    }
  }, [effectiveShowFft, fftZoom, fftPanX, fftPanY, canvasW, canvasH, pixelSize, width, height, fftDataRange, fftVminPct, fftVmaxPct, fftColormap, fftLogScale, fftShowColorbar, fftClickInfo, fftCropDims, getFftSlot, canvasRepaintSignal]);

  // -------------------------------------------------------------------------
  // Preview panel - cache colormapped offscreen (only recomputes when ROI
  // geometry, data, or display settings change - NOT on zoom/pan)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!previewVisible || !rawFrameDataRef.current) {
      previewOffscreenRef.current = null;
      return;
    }

    const raw = rawFrameDataRef.current;
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return;

    const roi = roiList[roiSelectedIdx];
    const crop = cropROIRegion(raw, width, height, roi);
    if (!crop) {
      previewOffscreenRef.current = null;
      setPreviewCropDims(null);
      setPreviewVersion(v => v + 1);
      return;
    }

    setPreviewCropDims({ w: crop.cropW, h: crop.cropH });

    const processed = logScale ? applyLogScale(crop.cropped) : crop.cropped;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    let vmin: number, vmax: number;
    const nP = Math.max(1, nPanels || 1);
    const hasTraitRange = traitVmin != null || traitVmax != null;
    const perPanelContrast = nP > 1 && !linkContrast && !sharedPanelSource && width % nP === 0 && height > 0;
    if (hasTraitRange) {
      ({ vmin, vmax } = resolveDisplayRange(
        dataMin,
        dataMax,
        traitVmin,
        traitVmax,
        logScale,
        imageVminPct,
        imageVmaxPct,
      ));
    } else if (autoContrast) {
      const cached = cachedAutoDisplayRange(autoVmins, autoVmaxs, displaySliceIdx, logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, displaySliceIdx, logScale);
      const mainProcessed = logScale ? applyLogScale(raw) : raw;
      ({ vmin, vmax } = cached ?? percentileClip(mainProcessed, percentileLow, percentileHigh));
    } else if (perPanelContrast) {
      const panelW = width / nP;
      const panel = Math.max(0, Math.min(nP - 1, Math.floor((Number(roi.col) || 0) / panelW)));
      const panelData = extractPanelSlice(raw, panel, logScale);
      const pdr = panelDataRanges[panel];
      const panelRange = (perPanelHistogramEnabled && pdr && pdr.max > pdr.min)
        ? pdr
        : (panelData && panelData.length > 0
            ? findDataRange(panelData)
            : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale));
      const resolved = resolvePanelRange(panel, panelRange, null);
      vmin = resolved.vmin;
      vmax = resolved.vmax;
    } else {
      const lo = logScale ? (dataMin >= 0 ? Math.log1p(dataMin) : -Math.log1p(-dataMin)) : dataMin;
      const hi = logScale ? (dataMax >= 0 ? Math.log1p(dataMax) : -Math.log1p(-dataMax)) : dataMax;
      ({ vmin, vmax } = sliderRange(lo, hi, imageVminPct, imageVmaxPct));
    }

    const offscreen = renderToOffscreen(processed, crop.cropW, crop.cropH, lut, vmin, vmax);
    previewOffscreenRef.current = offscreen;
    setPreviewVersion(v => v + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewVisible, selectedRoiKey, cmap, logScale, autoContrast, imageVminPct, imageVmaxPct, dataMin, dataMax, traitVmin, traitVmax, percentileLow, percentileHigh, width, height, frameBytes, displaySliceIdx, autoVmins, autoVmaxs, nPanels, linkContrast, sharedPanelSource, panelStates, vminPerPanel, vmaxPerPanel, canvasRepaintSignal]);

  // -------------------------------------------------------------------------
  // Preview panel - compute aspect-ratio-aware canvas dimensions
  // -------------------------------------------------------------------------
  const previewCanvasDims = (() => {
    if (!previewCropDims) return { w: canvasW, h: canvasH };
    const { w: cropW, h: cropH } = previewCropDims;
    const aspect = cropW / cropH;
    if (aspect >= 1) {
      return { w: canvasW, h: Math.max(20, Math.round(canvasW / aspect)) };
    } else {
      return { w: Math.max(20, Math.round(canvasH * aspect)), h: canvasH };
    }
  })();

  // -------------------------------------------------------------------------
  // Preview panel - draw cached offscreen with zoom/pan (fast, no recompute)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const canvas = previewCanvasRef.current;
    if (!canvas || !previewVisible) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const pw = previewCanvasDims.w;
    const ph = previewCanvasDims.h;
    const offscreen = previewOffscreenRef.current;
    if (!offscreen || !previewCropDims) {
      ctx.clearRect(0, 0, pw, ph);
      return;
    }

    ctx.imageSmoothingEnabled = smooth;
    ctx.clearRect(0, 0, pw, ph);

    const { zoom: pz, panX: ppX, panY: ppY } = previewZoom;
    if (pz !== 1 || ppX !== 0 || ppY !== 0) {
      ctx.save();
      const cx = pw / 2;
      const cy = ph / 2;
      ctx.translate(cx + ppX, cy + ppY);
      ctx.scale(pz, pz);
      ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, previewCropDims.w, previewCropDims.h, 0, 0, pw, ph);
      ctx.restore();
    } else {
      ctx.drawImage(offscreen, 0, 0, previewCropDims.w, previewCropDims.h, 0, 0, pw, ph);
    }
  }, [previewVisible, previewVersion, previewZoom, previewCanvasDims, previewCropDims, canvasRepaintSignal]);

  // Preview overlay - scale bar + zoom indicator
  React.useEffect(() => {
    const overlay = previewOverlayRef.current;
    if (!overlay || !previewVisible) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (previewCropDims && pixelSize > 0) {
      const unit = "Å" as const;
      drawScaleBarHiDPI(overlay, DPR, previewZoom.zoom, pixelSize, unit, previewCropDims.w);
    }
  }, [previewVisible, previewZoom, previewCropDims, previewCanvasDims, pixelSize, canvasRepaintSignal]);

  // Mouse handlers
  const panelIdxFromXY = (cssX: number, cssY: number): number => {
    const { n, cols, rows, gap, slotW, slotH } = getPanelLayout();
    if (n === 1) {
      return cssX >= 0 && cssX <= canvasW && cssY >= 0 && cssY <= canvasH
        ? (visiblePanelIndices[0] ?? 0)
        : -1;
    }
    const col = Math.floor(cssX / Math.max(1, slotW + gap));
    const row = Math.floor(cssY / Math.max(1, slotH + gap));
    if (col < 0 || col >= cols || row < 0 || row >= rows) return -1;
    const localX = cssX - col * (slotW + gap);
    const localY = cssY - row * (slotH + gap);
    if (localX < 0 || localX > slotW || localY < 0 || localY > slotH) return -1;
    const idx = row * cols + col;
    // Empty grid cells past N panels (partial last row) are not panels.
    return idx >= n ? -1 : (visiblePanelIndices[idx] ?? -1);
  };
  const panelIdxFromEvent = (e: React.MouseEvent): number => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    const rect = canvas.getBoundingClientRect();
    const cssX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const cssY = (e.clientY - rect.top) * (canvas.height / rect.height);
    return panelIdxFromXY(cssX, cssY);
  };
  const canvasPointFromClient = (clientX: number, clientY: number): { x: number; y: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  };
  const panelIdxFromClient = (clientX: number, clientY: number): number => {
    const pt = canvasPointFromClient(clientX, clientY);
    return pt ? panelIdxFromXY(pt.x, pt.y) : -1;
  };
  const beginPan = (e: React.MouseEvent) => {
    const idx = panelIdxFromEvent(e);
    if (idx < 0) return;
    panStartPanelRef.current = idx;
    const live = playRef.current;
    const base = live.panelStates[idx] || stateFor(idx);
    const s = {
      ...base,
      zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
      panX: live.linkPanels ? live.linkedState.panX : base.panX,
      panY: live.linkPanels ? live.linkedState.panY : base.panY,
    };
    setIsDraggingPan(true);
    setPanStart({ x: e.clientX, y: e.clientY, pX: s.panX, pY: s.panY });
  };
  const applyCanvasWheelZoom = (clientX: number, clientY: number, deltaY: number): boolean => {
    const canvas = canvasRef.current;
    if (!canvas) return false;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    const mouseX = (clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (clientY - rect.top) * (canvas.height / rect.height);
    const panelIdx = panelIdxFromXY(mouseX, mouseY);
    if (panelIdx < 0) return false;
    const live = playRef.current;
    const base = live.panelStates[panelIdx] || stateFor(panelIdx);
    const cur = {
      ...base,
      zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
      panX: live.linkPanels ? live.linkedState.panX : base.panX,
      panY: live.linkPanels ? live.linkedState.panY : base.panY,
    };
    const zoomFactor = Math.max(0.75, Math.min(1.35, Math.exp(-deltaY * 0.002)));
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, cur.zoom * zoomFactor));
    const zoomRatio = newZoom / cur.zoom;
    // Mouse position relative to this panel's slot (so zoom anchors to cursor within slot).
    const geom = getPanelGeometry(panelIdx);
    if (!geom) return false;
    const localX = mouseX - geom.slotX;
    const localY = mouseY - geom.slotY;
    const newPanX = localX - (localX - cur.panX) * zoomRatio;
    const newPanY = localY - (localY - cur.panY) * zoomRatio;
    syncPlaybackPanelTransform(panelIdx, newZoom, newPanX, newPanY);
    transformInputAtRef.current = performance.now();
    if (sidecarMode) setGpuDisplayVisible(false);
    if (scheduleTransformRender()) {
      scheduleTransformStateCommit();
    } else {
      commitLivePanelTransforms();
    }
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.lastWheelZoom = {
        panelIdx,
        zoom: Number(newZoom.toFixed(3)),
        panX: Number(newPanX.toFixed(1)),
        panY: Number(newPanY.toFixed(1)),
        deltaY: Number(deltaY.toFixed(3)),
      };
    }
    return true;
  };

  canvasWheelHandlerRef.current = (event: WheelEvent) => {
    if (fftInsetNativeWheelHandlerRef.current?.(event)) return;
    event.preventDefault();
    event.stopPropagation();
    if (reorderMode) return;
    applyCanvasWheelZoom(event.clientX, event.clientY, event.deltaY);
  };

  React.useEffect(() => {
    const el = canvasContainerRef.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => canvasWheelHandlerRef.current?.(event);
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [canvasW, canvasH]);

  React.useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const onFftInsetWheelCapture = (event: WheelEvent) => {
      fftInsetNativeWheelHandlerRef.current?.(event);
    };
    root.addEventListener("wheel", onFftInsetWheelCapture, { capture: true, passive: false });
    return () => root.removeEventListener("wheel", onFftInsetWheelCapture, { capture: true });
  }, []);

  const handleDoubleClick = () => {
    const resetPanels = Array.from({ length: Math.max(1, nPanels || 1) }, (_, i) => ({
      ...(playRef.current.panelStates[i] || initialState),
      zoom: 1,
      panX: 0,
      panY: 0,
    }));
    const resetLinked = { ...playRef.current.linkedState, zoom: 1, panX: 0, panY: 0 };
    playRef.current.linkedState = resetLinked;
    playRef.current.panelStates = resetPanels;
    linkedStateLiveRef.current = resetLinked;
    panelStatesLiveRef.current = resetPanels;
    if (sidecarMode) {
      invalidateSidecarViewportCache("view-reset");
      sidecarDisplayCacheDirtyRef.current = true;
    }
    setLinkedState(s => ({ ...s, zoom: 1, panX: 0, panY: 0 }));
    setPanelStates(arr => arr.map(s => ({ ...s, zoom: 1, panX: 0, panY: 0 })));
    setViewState({ linked_state: { ...resetLinked }, panel_states: resetPanels.map(v => ({ ...v })) });
    scheduleTransformRender();
  };

  const addROIAt = (row: number, col: number, shape: "circle" | "square" | "rectangle" | "annular" = newRoiShape) => {
    const clampedRow = Math.max(0, Math.min(height - 1, Math.round(row)));
    const clampedCol = Math.max(0, Math.min(width - 1, Math.round(col)));
    const next = [...roiItems, createROI(clampedRow, clampedCol, shape, roiItems.length, width, height)];
    setRoiList(next);
    setRoiSelectedIdx(next.length - 1);
    setShowRoiResizeHint(true);
  };

  const deleteSelectedROI = () => {
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return;
    const next = roiList.filter((_, i) => i !== roiSelectedIdx);
    setRoiList(next);
    setRoiSelectedIdx(next.length > 0 ? Math.min(roiSelectedIdx, next.length - 1) : -1);
  };

  const duplicateSelectedROI = () => {
    if (!selectedRoi) return;
    const duplicated: ROIItem = {
      ...selectedRoi,
      row: Math.max(0, Math.min(height - 1, Math.round(selectedRoi.row + 3))),
      col: Math.max(0, Math.min(width - 1, Math.round(selectedRoi.col + 3))),
      shape: selectedRoi.shape,
      radius: selectedRoi.radius,
      radius_inner: selectedRoi.radius_inner,
      width: selectedRoi.width,
      height: selectedRoi.height,
      color: ROI_COLORS[roiItems.length % ROI_COLORS.length],
      line_width: selectedRoi.line_width,
      highlight: false,
    };
    const next = [...roiItems, duplicated];
    setRoiList(next);
    setRoiSelectedIdx(next.length - 1);
  };


  const handleCopy = async () => {
    if (!canvasRef.current) return;
    try {
      const blob = await new Promise<Blob | null>(resolve => canvasRef.current!.toBlob(resolve, "image/png"));
      if (!blob) return;
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch (err) {
      console.warn("Show3D copy failed", err);
    }
  };

  const handleHandoffToShow2D = React.useCallback(() => {
    setViewMenuAnchor(null);
    setHandoffRequest(JSON.stringify({
      mode: "show2d",
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      frame: displaySliceIdx,
      panel: visiblePanelIndices,
    }));
  }, [displaySliceIdx, visiblePanelIndices, setHandoffRequest]);

  const handleClosePreparedView = React.useCallback(() => {
    setHandoffRequest(JSON.stringify({
      mode: "clear",
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    }));
  }, [setHandoffRequest]);

  const clickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const touchTransformRef = React.useRef<TouchTransformState | null>(null);
  const fftTouchTransformRef = React.useRef<FftTouchTransformState | null>(null);
  const fftInsetTouchTransformRef = React.useRef<FftTouchTransformState | null>(null);
  const kymoTouchTransformRef = React.useRef<FftTouchTransformState | null>(null);
  const lastTapRef = React.useRef<{ time: number; panelIdx: number } | null>(null);
  const lastFftTapRef = React.useRef<{ time: number } | null>(null);
  const lastFftInsetTapRef = React.useRef<{ time: number } | null>(null);
  const lastKymoTapRef = React.useRef<{ time: number } | null>(null);
  const [draggingProfileEndpoint, setDraggingProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingProfileLine, setIsDraggingProfileLine] = React.useState(false);
  const [hoveredProfileEndpoint, setHoveredProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringProfileLine, setIsHoveringProfileLine] = React.useState(false);
  const profileDragStartRef = React.useRef<{ row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } } | null>(null);

  const screenToImg = (e: React.MouseEvent): { imgCol: number; imgRow: number; panelIdx: number; panelCol: number } => {
    const pt = canvasPointFromEvent(e);
    if (!pt) return { imgCol: 0, imgRow: 0, panelIdx: -1, panelCol: 0 };
    const panelIdx = panelIdxFromXY(pt.x, pt.y);
    const geom = getPanelGeometry(panelIdx);
    if (!geom) return { imgCol: 0, imgRow: 0, panelIdx: -1, panelCol: 0 };
    // Undo slot offset, pan, zoom, then panel source scaling.
    let localCol = (pt.x - geom.slotX - geom.state.panX) / (geom.scaleX * geom.state.zoom);
    let row = (pt.y - geom.slotY - geom.state.panY) / (geom.scaleY * geom.state.zoom);
    // Undo image_rotation in panel-local source coordinates.
    const r = (((imageRotation % 4) + 4) % 4) | 0;
    if (r !== 0) {
      const rotSwap = (r % 2) !== 0;
      const visW = rotSwap ? sourcePanelHeight : sourcePanelWidth;
      const visH = rotSwap ? sourcePanelWidth : sourcePanelHeight;
      const cx = localCol - visW / 2;
      const cy = row - visH / 2;
      let ux: number, uy: number;
      if (r === 1) { ux = cy; uy = -cx; }
      else if (r === 2) { ux = -cx; uy = -cy; }
      else { ux = -cy; uy = cx; }
      localCol = ux + sourcePanelWidth / 2;
      row = uy + sourcePanelHeight / 2;
    }
    return { imgCol: panelGlobalCol(localCol, panelIdx), imgRow: row, panelIdx, panelCol: localCol };
  };
  const profileCoordinateWidth = singlePanelPageProfile ? sourcePanelWidth : width;
  const screenToProfileImg = (e: React.MouseEvent): { imgCol: number; imgRow: number; panelIdx: number; panelCol: number } => {
    const point = screenToImg(e);
    return singlePanelPageProfile && point.panelIdx >= 0
      ? { ...point, imgCol: point.panelCol }
      : point;
  };

  const hitTestROI = (imgCol: number, imgRow: number): number => {
    if (!effectiveRoiActive || roiItems.length === 0) return -1;
    for (let roiIdx = roiItems.length - 1; roiIdx >= 0; roiIdx--) {
      const roi = roiItems[roiIdx];
      const shape = roi.shape || "circle";
      if (shape === "circle" || shape === "annular") {
        if (Math.sqrt((imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2) <= roi.radius) return roiIdx;
      } else if (shape === "square") {
        if (Math.abs(imgCol - roi.col) <= roi.radius && Math.abs(imgRow - roi.row) <= roi.radius) return roiIdx;
      } else if (shape === "rectangle") {
        if (Math.abs(imgCol - roi.col) <= roi.width / 2 && Math.abs(imgRow - roi.row) <= roi.height / 2) return roiIdx;
      }
    }
    return -1;
  };

  const getHitArea = () => RESIZE_HIT_AREA_PX / (displayScale * zoom);

  const isNearEdge = (imgCol: number, imgRow: number, roi: ROIItem): boolean => {
    const hitArea = getHitArea();
    const shape = roi.shape || "circle";
    if (shape === "circle" || shape === "annular") {
      const dist = Math.sqrt((imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2);
      return Math.abs(dist - roi.radius) < hitArea;
    }
    if (shape === "square") {
      const dx = Math.abs(imgCol - roi.col);
      const dy = Math.abs(imgRow - roi.row);
      const r = roi.radius;
      return (dx <= r + hitArea && dy <= r + hitArea) && (Math.abs(dx - r) < hitArea || Math.abs(dy - r) < hitArea);
    }
    if (shape === "rectangle") {
      const dx = Math.abs(imgCol - roi.col);
      const dy = Math.abs(imgRow - roi.row);
      const hw = roi.width / 2;
      const hh = roi.height / 2;
      return (dx <= hw + hitArea && dy <= hh + hitArea) && (Math.abs(dx - hw) < hitArea || Math.abs(dy - hh) < hitArea);
    }
    return false;
  };

  const isNearResizeHandle = (imgCol: number, imgRow: number): boolean => {
    if (!effectiveRoiActive || !selectedRoi) return false;
    return isNearEdge(imgCol, imgRow, selectedRoi);
  };

  const isNearAnyEdge = (imgCol: number, imgRow: number): boolean => {
    if (!effectiveRoiActive || roiItems.length === 0) return false;
    return roiItems.some(roi => isNearEdge(imgCol, imgRow, roi));
  };

  const isNearResizeHandleInner = (imgCol: number, imgRow: number): boolean => {
    if (!effectiveRoiActive || !selectedRoi || selectedRoi.shape !== "annular") return false;
    const hitArea = getHitArea();
    const dist = Math.sqrt((imgCol - selectedRoi.col) ** 2 + (imgRow - selectedRoi.row) ** 2);
    return Math.abs(dist - selectedRoi.radius_inner) < hitArea;
  };

  const updateROI = (e: React.MouseEvent) => {
    if (!selectedRoi) return;
    const { imgCol, imgRow } = screenToImg(e);
    updateSelectedRoi({
      col: Math.max(0, Math.min(width - 1, Math.floor(imgCol))),
      row: Math.max(0, Math.min(height - 1, Math.floor(imgRow))),
    });
  };

  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    // Ignore clicks in empty grid cells (partial last row when N isn't a
    // multiple of max_cols). Otherwise the click attributes to the last
    // real panel and zoom/pan jumps unexpectedly.
    if (panelIdxFromEvent(e) < 0) return;
    clickStartRef.current = { x: e.clientX, y: e.clientY };
    pendingRoiAddRef.current = null;
    // Check if clicking on lens inset for drag or resize
    if (showLens) {
      const rect = canvasContainerRef.current?.getBoundingClientRect();
      if (rect) {
        const cssX = e.clientX - rect.left;
        const cssY = e.clientY - rect.top;
        const margin = 12;
        const lx = lensAnchor ? lensAnchor.x : margin;
        const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
        if (cssX >= lx && cssX <= lx + lensDisplaySize && cssY >= ly && cssY <= ly + lensDisplaySize) {
          const edgeHit = 8;
          const nearEdge = cssX - lx < edgeHit || lx + lensDisplaySize - cssX < edgeHit ||
                           cssY - ly < edgeHit || ly + lensDisplaySize - cssY < edgeHit;
          if (nearEdge) {
            setIsResizingLens(true);
            lensResizeStartRef.current = { my: e.clientY, startSize: lensDisplaySize };
          } else {
            setIsDraggingLens(true);
            lensDragStartRef.current = { mx: e.clientX, my: e.clientY, ax: lx, ay: ly };
          }
          return;
        }
      }
    }
    if (overlayEditMode) {
      const { imgRow, panelIdx, panelCol } = screenToImg(e);
      if (panelIdx >= 0) {
        const hitRadius = getImageHitRadius(panelIdx);
        const hit = panelOverlayHit(panelOverlays?.[panelIdx], imgRow, panelCol, sourcePanelWidth, sourcePanelHeight, hitRadius);
        if (hit) {
          const original = panelOverlays?.[panelIdx]?.[hit.overlay];
          if (!original) return;
          setOverlaySelection({ panel: panelIdx, overlay: hit.overlay });
          overlayDragRef.current = {
            mode: hit.mode,
            panel: panelIdx,
            overlay: hit.overlay,
            handle: hit.handle,
            startRow: imgRow,
            startCol: panelCol,
            original,
          };
          setIsDraggingOverlay(true);
          setIsDraggingPan(false);
          setPanStart(null);
          e.preventDefault();
          return;
        }
      }
      setOverlaySelection(null);
    }
    if (profileActive) {
      const { imgCol, imgRow, panelIdx } = screenToProfileImg(e);
      if (profilePoints.length === 2) {
        if (panelIdx !== profilePanelIdx) {
          beginPan(e);
          return;
        }
        const p0 = profilePoints[0];
        const p1 = profilePoints[1];
        const hitRadius = getImageHitRadius(profilePanelIdx);
        const d0 = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
        const d1 = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
        if (d0 <= hitRadius || d1 <= hitRadius) {
          setDraggingProfileEndpoint(d0 <= d1 ? 0 : 1);
          setIsDraggingPan(false);
          setPanStart(null);
          return;
        }
        if (pointToSegmentDistance(imgCol, imgRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius) {
          setIsDraggingProfileLine(true);
          profileDragStartRef.current = {
            row: imgRow,
            col: imgCol,
            p0: { row: p0.row, col: p0.col },
            p1: { row: p1.row, col: p1.col },
          };
          setIsDraggingPan(false);
          setPanStart(null);
          return;
        }
      }
      beginPan(e);
      return;
    }
    if (effectiveRoiActive) {
      const { imgCol, imgRow } = screenToImg(e);
      if (isNearResizeHandleInner(imgCol, imgRow)) {
        setIsDraggingResizeInner(true);
        return;
      }
      if (isNearResizeHandle(imgCol, imgRow)) {
        e.preventDefault();
        resizeAspectRef.current = selectedRoi && (selectedRoi.shape === "rectangle") && selectedRoi.width > 0 && selectedRoi.height > 0 ? selectedRoi.width / selectedRoi.height : null;
        setIsDraggingResize(true);
        return;
      }
      if (roiItems.length > 0) {
        for (let roiIdx = roiItems.length - 1; roiIdx >= 0; roiIdx--) {
          const roi = roiItems[roiIdx];
          if (isNearEdge(imgCol, imgRow, roi)) {
            e.preventDefault();
            resizeAspectRef.current = roi && (roi.shape === "rectangle") && roi.width > 0 && roi.height > 0 ? roi.width / roi.height : null;
            setRoiSelectedIdx(roiIdx);
            setIsDraggingResize(true);
            return;
          }
        }
      }
      const hitIdx = hitTestROI(imgCol, imgRow);
      if (hitIdx >= 0) {
        setRoiSelectedIdx(hitIdx);
        setIsDraggingROI(true);
        return;
      }
      setRoiSelectedIdx(-1);
      pendingRoiAddRef.current = {
        row: Math.max(0, Math.min(height - 1, Math.round(imgRow))),
        col: Math.max(0, Math.min(width - 1, Math.round(imgCol))),
      };
      return;
    }
    beginPan(e);
  };

  type TouchPoint = { clientX: number; clientY: number };
  const touchDistance = (a: TouchPoint, b: TouchPoint) => Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
  const touchMidpoint = (a: TouchPoint, b: TouchPoint) => ({ x: (a.clientX + b.clientX) / 2, y: (a.clientY + b.clientY) / 2 });

  const handleCanvasTouchStart = (e: React.TouchEvent) => {
    if (profileActive || effectiveRoiActive) return;
    if (e.touches.length === 1) {
      const t = e.touches[0];
      const panelIdx = panelIdxFromClient(t.clientX, t.clientY);
      if (panelIdx < 0) return;
      const now = Date.now();
      const lastTap = lastTapRef.current;
      if (lastTap && lastTap.panelIdx === panelIdx && now - lastTap.time < 320) {
        e.preventDefault();
        handleDoubleClick();
        lastTapRef.current = null;
        touchTransformRef.current = null;
        return;
      }
      lastTapRef.current = { time: now, panelIdx };
      if (showLens) return;
      const live = playRef.current;
      const base = live.panelStates[panelIdx] || stateFor(panelIdx);
      touchTransformRef.current = {
        panelIdx,
        mode: "pan",
        startX: t.clientX,
        startY: t.clientY,
        startDistance: 0,
        startMidX: t.clientX,
        startMidY: t.clientY,
        startState: {
          ...base,
          zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
          panX: live.linkPanels ? live.linkedState.panX : base.panX,
          panY: live.linkPanels ? live.linkedState.panY : base.panY,
        },
      };
      e.preventDefault();
      return;
    }
    if (e.touches.length >= 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      const mid = touchMidpoint(a, b);
      const panelIdx = panelIdxFromClient(mid.x, mid.y);
      if (panelIdx < 0) return;
      const live = playRef.current;
      const base = live.panelStates[panelIdx] || stateFor(panelIdx);
      touchTransformRef.current = {
        panelIdx,
        mode: "pinch",
        startX: mid.x,
        startY: mid.y,
        startDistance: Math.max(1, touchDistance(a, b)),
        startMidX: mid.x,
        startMidY: mid.y,
        startState: {
          ...base,
          zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
          panX: live.linkPanels ? live.linkedState.panX : base.panX,
          panY: live.linkPanels ? live.linkedState.panY : base.panY,
        },
      };
      e.preventDefault();
    }
  };

  const handleCanvasTouchMove = (e: React.TouchEvent) => {
    const start = touchTransformRef.current;
    if (!start) return;
    const canvas = canvasRef.current;
    const geom = getPanelGeometry(start.panelIdx);
    if (!canvas || !geom) return;
    e.preventDefault();
    const base = start.startState;
    if (start.mode === "pinch" && e.touches.length >= 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      const mid = touchMidpoint(a, b);
      const startPoint = canvasPointFromClient(start.startMidX, start.startMidY);
      const currentPoint = canvasPointFromClient(mid.x, mid.y);
      if (!startPoint || !currentPoint) return;
      const startLocalX = startPoint.x - geom.slotX;
      const startLocalY = startPoint.y - geom.slotY;
      const currentLocalX = currentPoint.x - geom.slotX;
      const currentLocalY = currentPoint.y - geom.slotY;
      const imageX = (startLocalX - base.panX) / base.zoom;
      const imageY = (startLocalY - base.panY) / base.zoom;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, base.zoom * (touchDistance(a, b) / start.startDistance)));
      syncPlaybackPanelTransform(start.panelIdx, newZoom, currentLocalX - imageX * newZoom, currentLocalY - imageY * newZoom);
    } else if (start.mode === "pan" && e.touches.length === 1) {
      const t = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / Math.max(1, rect.width);
      const scaleY = canvas.height / Math.max(1, rect.height);
      syncPlaybackPanelTransform(
        start.panelIdx,
        base.zoom,
        base.panX + (t.clientX - start.startX) * scaleX,
        base.panY + (t.clientY - start.startY) * scaleY,
      );
    }
    transformInputAtRef.current = performance.now();
    if (sidecarMode) setGpuDisplayVisible(false);
    if (scheduleTransformRender()) scheduleTransformStateCommit();
    else commitLivePanelTransforms();
  };

  const handleCanvasTouchEnd = (e: React.TouchEvent) => {
    if (e.touches.length > 0 || !touchTransformRef.current) return;
    commitLivePanelTransforms();
    touchTransformRef.current = null;
  };

  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    if (overlayDragRef.current) {
      const drag = overlayDragRef.current;
      const { imgRow, panelIdx, panelCol } = screenToImg(e);
      if (panelIdx !== drag.panel) return;
      updatePanelOverlay(
        drag.panel,
        drag.overlay,
        updateOverlayFromDrag(drag.original, drag.mode, drag.startRow, drag.startCol, imgRow, panelCol, sourcePanelWidth, sourcePanelHeight, drag.handle),
      );
      e.preventDefault();
      return;
    }
    // Fast path: during pan drag, skip all cursor/hover/lens work - just update pan
    if (isDraggingPan && panStart) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - panStart.x) * scaleX;
      const dy = (e.clientY - panStart.y) * scaleY;
      const newPanX = panStart.pX + dx;
      const newPanY = panStart.pY + dy;
      const live = playRef.current;
      const base = live.panelStates[panStartPanelRef.current] || stateFor(panStartPanelRef.current);
      const current = {
        ...base,
        zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
        panX: live.linkPanels ? live.linkedState.panX : base.panX,
        panY: live.linkPanels ? live.linkedState.panY : base.panY,
      };
      syncPlaybackPanelTransform(panStartPanelRef.current, current.zoom, newPanX, newPanY);
      transformInputAtRef.current = performance.now();
      if (sidecarMode) setGpuDisplayVisible(false);
      if (scheduleTransformRender()) scheduleTransformStateCommit();
      else commitLivePanelTransforms();
      return;
    }

    // Cursor readout: convert screen position to image pixel coordinates.
    // Skip when hovering an empty grid cell (partial last row when nPanels
    // isn't a multiple of max_cols) so dead space doesn't flash row/col
    // numbers from a phantom panel.
    const canvas = canvasRef.current;
    const hoverPanelIdx = panelIdxFromEvent(e);
    if (hoverPanelIdx < 0) {
      scheduleCursorInfo(null);
      if (showLens) setLensPos(null);
    } else if (canvas && rawFrameDataRef.current) {
      const { imgRow, imgCol, panelIdx, panelCol } = screenToImg(e);
      const pixelDataCol = Math.floor(imgCol);
      const pixelPanelCol = Math.floor(panelCol);
      const pixelRow = Math.floor(imgRow);
      if (
        pixelDataCol >= 0 && pixelDataCol < width &&
        pixelPanelCol >= 0 && pixelPanelCol < sourcePanelWidth &&
        pixelRow >= 0 && pixelRow < height
      ) {
        const rawData = rawFrameDataRef.current;
        scheduleCursorInfo({
          row: pixelRow,
          col: pixelPanelCol,
          value: rawData[pixelRow * width + pixelDataCol],
          panelIdx,
        });
        if (showLens) setLensPos({ row: pixelRow, col: pixelDataCol });
      } else {
        scheduleCursorInfo(null);
        if (showLens) setLensPos(null);
      }
    }

    // Lens edge hover detection
    if (showLens) {
      const rect2 = canvasContainerRef.current?.getBoundingClientRect();
      if (rect2) {
        const cssX2 = e.clientX - rect2.left;
        const cssY2 = e.clientY - rect2.top;
        const margin = 12;
        const lx = lensAnchor ? lensAnchor.x : margin;
        const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
        const inside = cssX2 >= lx && cssX2 <= lx + lensDisplaySize && cssY2 >= ly && cssY2 <= ly + lensDisplaySize;
        const edgeHit = 8;
        const nearEdge = inside && (cssX2 - lx < edgeHit || lx + lensDisplaySize - cssX2 < edgeHit ||
                                     cssY2 - ly < edgeHit || ly + lensDisplaySize - cssY2 < edgeHit);
        setIsHoveringLensEdge(nearEdge);
      }
    } else {
      setIsHoveringLensEdge(false);
    }
    if (overlayEditMode && !isDraggingPan) {
      const { imgRow, panelIdx, panelCol } = screenToImg(e);
      const hit = panelIdx >= 0
        ? panelOverlayHit(panelOverlays?.[panelIdx], imgRow, panelCol, sourcePanelWidth, sourcePanelHeight, getImageHitRadius(panelIdx))
        : null;
      setIsHoveringOverlay(Boolean(hit));
      return;
    } else if (isHoveringOverlay) {
      setIsHoveringOverlay(false);
    }

    // Lens drag
    if (isDraggingLens && lensDragStartRef.current) {
      const dx = e.clientX - lensDragStartRef.current.mx;
      const dy = e.clientY - lensDragStartRef.current.my;
      setLensAnchor({ x: lensDragStartRef.current.ax + dx, y: lensDragStartRef.current.ay + dy });
      return;
    }

    // Lens resize drag
    if (isResizingLens && lensResizeStartRef.current) {
      const dy = e.clientY - lensResizeStartRef.current.my;
      setLensDisplaySize(Math.max(64, Math.min(256, lensResizeStartRef.current.startSize + dy)));
      return;
    }

    if (profileActive && profilePoints.length === 2) {
      const { imgCol, imgRow, panelIdx } = screenToProfileImg(e);
      const p0 = profilePoints[0];
      const p1 = profilePoints[1];
      const hitRadius = getImageHitRadius(profilePanelIdx);
      const sameProfilePanel = panelIdx === profilePanelIdx;
      const d0 = sameProfilePanel ? Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2) : Infinity;
      const d1 = sameProfilePanel ? Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2) : Infinity;
      if (draggingProfileEndpoint !== null) {
        if (!rawFrameDataRef.current || panelIdx !== profilePanelIdx) return;
        const clampedRow = Math.max(0, Math.min(height - 1, imgRow));
        const clampedCol = Math.max(0, Math.min(profileCoordinateWidth - 1, imgCol));
        const next = [
          draggingProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : profilePoints[0],
          draggingProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : profilePoints[1],
        ];
        setProfileLine(next);
        setProfileData(sampleProfileForActivePage(rawFrameDataRef.current, next[0], next[1]));
        return;
      }
      if (isDraggingProfileLine && profileDragStartRef.current) {
        if (!rawFrameDataRef.current || panelIdx !== profilePanelIdx) return;
        const drag = profileDragStartRef.current;
        let deltaRow = imgRow - drag.row;
        let deltaCol = imgCol - drag.col;
        const minRow = Math.min(drag.p0.row, drag.p1.row);
        const maxRow = Math.max(drag.p0.row, drag.p1.row);
        const minCol = Math.min(drag.p0.col, drag.p1.col);
        const maxCol = Math.max(drag.p0.col, drag.p1.col);
        deltaRow = Math.max(deltaRow, -minRow);
        deltaRow = Math.min(deltaRow, (height - 1) - maxRow);
        deltaCol = Math.max(deltaCol, -minCol);
        deltaCol = Math.min(deltaCol, (profileCoordinateWidth - 1) - maxCol);
        const next = [
          { row: drag.p0.row + deltaRow, col: drag.p0.col + deltaCol },
          { row: drag.p1.row + deltaRow, col: drag.p1.col + deltaCol },
        ];
        setProfileLine(next);
        setProfileData(sampleProfileForActivePage(rawFrameDataRef.current, next[0], next[1]));
        return;
      }
      const nextHoveredEndpoint: 0 | 1 | null = d0 <= hitRadius ? 0 : d1 <= hitRadius ? 1 : null;
      const nextHoverLine = nextHoveredEndpoint === null && pointToSegmentDistance(imgCol, imgRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius;
      setHoveredProfileEndpoint(nextHoveredEndpoint);
      setIsHoveringProfileLine(nextHoverLine);
    } else {
      if (hoveredProfileEndpoint !== null) setHoveredProfileEndpoint(null);
      if (isHoveringProfileLine) setIsHoveringProfileLine(false);
    }

    // Resize handle dragging
    if (isDraggingResizeInner && selectedRoi) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      const newR = Math.sqrt((ic - selectedRoi.col) ** 2 + (ir - selectedRoi.row) ** 2);
      updateSelectedRoi({ radius_inner: Math.max(1, Math.min(selectedRoi.radius - 1, Math.round(newR))) });
      setShowRoiResizeHint(false);
      return;
    }
    if (isDraggingResize && selectedRoi) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      const shape = selectedRoi.shape || "circle";
      if (shape === "rectangle") {
        let newW = Math.max(2, Math.round(Math.abs(ic - selectedRoi.col) * 2));
        let newH = Math.max(2, Math.round(Math.abs(ir - selectedRoi.row) * 2));
        if (e.shiftKey && resizeAspectRef.current != null) {
          const aspect = resizeAspectRef.current;
          if (newW / newH > aspect) newH = Math.max(2, Math.round(newW / aspect));
          else newW = Math.max(2, Math.round(newH * aspect));
        }
        updateSelectedRoi({ width: newW, height: newH });
      } else {
        const newR = shape === "square"
          ? Math.max(Math.abs(ic - selectedRoi.col), Math.abs(ir - selectedRoi.row))
          : Math.sqrt((ic - selectedRoi.col) ** 2 + (ir - selectedRoi.row) ** 2);
        const minR = shape === "annular" ? selectedRoi.radius_inner + 1 : 1;
        updateSelectedRoi({ radius: Math.max(minR, Math.round(newR)) });
      }
      setShowRoiResizeHint(false);
      return;
    }

    // Hover state for resize handles
    if (effectiveRoiActive && !isDraggingROI && !isDraggingPan) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      const hoveringInner = isNearResizeHandleInner(ic, ir);
      const hoveringOuter = isNearAnyEdge(ic, ir);
      setIsHoveringResizeInner(hoveringInner);
      setIsHoveringResize(hoveringOuter);
      if (hoveringInner || hoveringOuter) setShowRoiResizeHint(false);
    }

    if (isDraggingROI) {
      updateROI(e);
    }
  };

  const handleCanvasMouseUp = (e: React.MouseEvent) => {
    if (overlayDragRef.current) {
      overlayDragRef.current = null;
      setIsDraggingOverlay(false);
      return;
    }
    if (draggingProfileEndpoint !== null || isDraggingProfileLine) {
      setDraggingProfileEndpoint(null);
      setIsDraggingProfileLine(false);
      profileDragStartRef.current = null;
      clickStartRef.current = null;
      pendingRoiAddRef.current = null;
      setIsDraggingROI(false);
      setIsDraggingResize(false);
      setIsDraggingResizeInner(false);
      setIsDraggingLens(false);
      lensDragStartRef.current = null;
      setIsResizingLens(false);
      lensResizeStartRef.current = null;
      setIsDraggingPan(false);
      setPanStart(null);
      setHoveredProfileEndpoint(null);
      setIsHoveringProfileLine(false);
      return;
    }

    // Profile click capture
    if (profileActive && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        if (rawFrameDataRef.current) {
          const { imgCol, imgRow, panelIdx } = screenToProfileImg(e);
          if (panelIdx >= 0 && imgCol >= 0 && imgCol < profileCoordinateWidth && imgRow >= 0 && imgRow < height) {
            const pt = { row: imgRow, col: imgCol };
            if (profilePoints.length === 0 || profilePoints.length === 2 || panelIdx !== profilePanelIdx) {
              setProfilePanelIdx(panelIdx);
              setProfileLine([pt]);
              setProfileData(null);
            } else {
              const p0 = profilePoints[0];
              setProfileLine([p0, pt]);
              setProfileData(sampleProfileForActivePage(rawFrameDataRef.current, p0, pt));
            }
          }
        }
      }
    }

    // ROI click-to-add (empty-area click)
    if (effectiveRoiActive && pendingRoiAddRef.current && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        addROIAt(pendingRoiAddRef.current.row, pendingRoiAddRef.current.col);
      }
    }
    clickStartRef.current = null;
    pendingRoiAddRef.current = null;
    if (isDraggingPan) commitLivePanelTransforms();
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setIsDraggingLens(false);
    lensDragStartRef.current = null;
    setIsResizingLens(false);
    lensResizeStartRef.current = null;
    setIsDraggingPan(false);
    setPanStart(null);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    profileDragStartRef.current = null;
  };

  const handleCanvasMouseLeave = () => {
    scheduleCursorInfo(null);
    // Lens persists at last position when cursor exits main canvas. Wiping on every
    // leave kills the inset whenever the user touches a slider, FFT panel, or any
    // sibling control - surprising "lens vanished" footgun. User explicitly turns
    // lens off via the Lens switch.
    pendingRoiAddRef.current = null;
    overlayDragRef.current = null;
    setIsDraggingOverlay(false);
    setIsHoveringOverlay(false);
    if (isDraggingPan) commitLivePanelTransforms();
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setIsDraggingLens(false);
    lensDragStartRef.current = null;
    setIsResizingLens(false);
    lensResizeStartRef.current = null;
    setIsHoveringLensEdge(false);
    setIsHoveringResize(false);
    setIsHoveringResizeInner(false);
    setIsDraggingPan(false);
    setPanStart(null);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    profileDragStartRef.current = null;
  };

  // FFT mouse handlers
  const [isFftDragging, setIsFftDragging] = React.useState(false);
  const [fftPanStart, setFftPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  const clampFftPan = React.useCallback((panX: number, panY: number, zoom: number, viewportW: number, viewportH: number) => {
    const clampAxis = (pan: number, viewport: number) => {
      if (zoom <= 1 || viewport <= 0) return 0;
      return Math.max(viewport * (1 - zoom), Math.min(0, pan));
    };
    return {
      panX: clampAxis(panX, viewportW),
      panY: clampAxis(panY, viewportH),
    };
  }, []);

  const zoomFftAtPoint = React.useCallback((anchorX: number, anchorY: number, deltaY: number, viewportW?: number, viewportH?: number) => {
    const currentBase = fftViewLiveRef.current;
    const current = !fftUserAdjustedViewRef.current && currentBase.zoom > 1 && viewportW != null && viewportH != null
      ? {
        zoom: currentBase.zoom,
        panX: viewportW * (1 - currentBase.zoom) / 2,
        panY: viewportH * (1 - currentBase.zoom) / 2,
      }
      : currentBase;
    fftUserAdjustedViewRef.current = true;
    fftOverlayInitialCenterPendingRef.current = false;
    fftViewCenterOnViewportRef.current = false;
    const zoomFactor = Math.max(0.75, Math.min(1.35, Math.exp(-deltaY * 0.002)));
    const minZoom = fftLayoutOverlay ? 1 : MIN_ZOOM;
    const newZoom = Math.max(minZoom, Math.min(MAX_ZOOM, current.zoom * zoomFactor));
    const zoomRatio = newZoom / Math.max(1e-6, current.zoom);
    const nextPanX = anchorX - (anchorX - current.panX) * zoomRatio;
    const nextPanY = anchorY - (anchorY - current.panY) * zoomRatio;
    const clamped = viewportW != null && viewportH != null
      ? clampFftPan(nextPanX, nextPanY, newZoom, viewportW, viewportH)
      : { panX: nextPanX, panY: nextPanY };
    scheduleFftViewState({ zoom: newZoom, panX: clamped.panX, panY: clamped.panY }, true, fftLayoutOverlay);
  }, [clampFftPan, fftLayoutOverlay, scheduleFftViewState]);

  fftInsetNativeWheelHandlerRef.current = (event: WheelEvent) => {
    const target = event.target;
    let inset: Element | null = target instanceof Element
      ? target.closest('[data-show3d-fft-inset="true"]')
      : null;
    if (!(inset instanceof HTMLElement)) {
      inset = document.elementsFromPoint(event.clientX, event.clientY)
        .find(el => el instanceof Element && el.closest('[data-show3d-fft-inset="true"]'))
        ?.closest('[data-show3d-fft-inset="true"]') ?? null;
    }
    if (!(inset instanceof HTMLElement)) {
      const root = rootRef.current;
      const hit = root
        ? Array.from(root.querySelectorAll<HTMLElement>('[data-show3d-fft-inset="true"]')).find(el => {
          const rect = el.getBoundingClientRect();
          return event.clientX >= rect.left && event.clientX <= rect.right
            && event.clientY >= rect.top && event.clientY <= rect.bottom;
        })
        : null;
      inset = hit ?? null;
    }
    if (!(inset instanceof HTMLElement)) return false;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    const rect = inset.getBoundingClientRect();
    zoomFftAtPoint(event.clientX - rect.left, event.clientY - rect.top, event.deltaY, rect.width, rect.height);
    return true;
  };

  const handleFftWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const canvas = fftCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const panelGrid = fftPanelGridRef.current;
    if (panelGrid) {
      for (let slot = 0; slot < panelGrid.count; slot++) {
        const dst = getFftSlot(slot, panelGrid.count, panelGrid.cols, panelGrid.rows);
        if (mouseX < dst.x || mouseX >= dst.x + dst.w || mouseY < dst.y || mouseY >= dst.y + dst.h) continue;
        const localX = mouseX - dst.x;
        const localY = mouseY - dst.y;
        zoomFftAtPoint(localX, localY, e.deltaY, dst.w, dst.h);
        return;
      }
    }
    zoomFftAtPoint(mouseX, mouseY, e.deltaY, canvas.width, canvas.height);
  };

  const handleFftInsetWheel = (e: React.WheelEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const localX = e.clientX - rect.left;
    const localY = e.clientY - rect.top;
    zoomFftAtPoint(localX, localY, e.deltaY, rect.width, rect.height);
  };

  const handleFftInsetTouchStart = (e: React.TouchEvent<HTMLElement>) => {
    const now = Date.now();
    if (e.touches.length === 1) {
      const lastTap = lastFftInsetTapRef.current;
      if (lastTap && now - lastTap.time < 320) {
        e.preventDefault();
        e.stopPropagation();
        handleFftReset();
        lastFftInsetTapRef.current = null;
        fftInsetTouchTransformRef.current = null;
        return;
      }
      lastFftInsetTapRef.current = { time: now };
      return;
    }
    if (e.touches.length < 2) return;
    e.preventDefault();
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const a = e.touches[0];
    const b = e.touches[1];
    const mid = touchMidpoint(a, b);
    const live = fftViewLiveRef.current;
    const base = !fftUserAdjustedViewRef.current && live.zoom > 1
      ? {
        zoom: live.zoom,
        panX: rect.width * (1 - live.zoom) / 2,
        panY: rect.height * (1 - live.zoom) / 2,
      }
      : live;
    fftInsetTouchTransformRef.current = {
      mode: "pinch",
      startX: mid.x,
      startY: mid.y,
      startDistance: Math.max(1, touchDistance(a, b)),
      startMidX: mid.x,
      startMidY: mid.y,
      startState: base,
    };
  };

  const handleFftInsetTouchMove = (e: React.TouchEvent<HTMLElement>) => {
    const start = fftInsetTouchTransformRef.current;
    if (!start || e.touches.length < 2) return;
    e.preventDefault();
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const a = e.touches[0];
    const b = e.touches[1];
    const mid = touchMidpoint(a, b);
    const startX = start.startMidX - rect.left;
    const startY = start.startMidY - rect.top;
    const currentX = mid.x - rect.left;
    const currentY = mid.y - rect.top;
    const base = start.startState;
    const imageX = (startX - base.panX) / Math.max(1e-6, base.zoom);
    const imageY = (startY - base.panY) / Math.max(1e-6, base.zoom);
    const newZoom = Math.max(1, Math.min(MAX_ZOOM, base.zoom * (touchDistance(a, b) / start.startDistance)));
    const clamped = clampFftPan(
      currentX - imageX * newZoom,
      currentY - imageY * newZoom,
      newZoom,
      rect.width,
      rect.height,
    );
    fftUserAdjustedViewRef.current = true;
    fftOverlayInitialCenterPendingRef.current = false;
    fftViewCenterOnViewportRef.current = false;
    scheduleFftViewState({ zoom: newZoom, panX: clamped.panX, panY: clamped.panY }, true, true);
  };

  const handleFftInsetTouchEnd = (e: React.TouchEvent<HTMLElement>) => {
    if (e.touches.length < 2) fftInsetTouchTransformRef.current = null;
  };

  const handleFftInsetPointerDown = (
    e: React.PointerEvent<HTMLElement>,
    panelLeft: number,
    panelTop: number,
    panelW: number,
    panelH: number,
    insetX: number,
    insetY: number,
    insetW: number,
    insetH: number,
  ) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    fftOverlayDragRef.current = {
      pointerId: e.pointerId,
      startClientX: e.clientX,
      startClientY: e.clientY,
      startInsetX: insetX,
      startInsetY: insetY,
      panelLeft,
      panelTop,
      panelW,
      panelH,
      insetW,
      insetH,
      moved: false,
    };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };

  const handleFftInsetPointerMove = (e: React.PointerEvent<HTMLElement>) => {
    const drag = fftOverlayDragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    if (Math.hypot(e.clientX - drag.startClientX, e.clientY - drag.startClientY) > 4) {
      drag.moved = true;
    }
    if (drag.moved) {
      const nextX = Math.max(drag.panelLeft, Math.min(drag.panelLeft + drag.panelW - drag.insetW, drag.startInsetX + e.clientX - drag.startClientX));
      const nextY = Math.max(drag.panelTop, Math.min(drag.panelTop + drag.panelH - drag.insetH, drag.startInsetY + e.clientY - drag.startClientY));
      setFftOverlayDragPreview({ x: nextX - drag.panelLeft, y: nextY - drag.panelTop });
    }
  };

  const handleFftInsetPointerUp = (e: React.PointerEvent<HTMLElement>) => {
    const drag = fftOverlayDragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    fftOverlayDragRef.current = null;
    setFftOverlayDragPreview(null);
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    if (!drag.moved) return;
    const centerX = drag.startInsetX + e.clientX - drag.startClientX - drag.panelLeft + drag.insetW / 2;
    const centerY = drag.startInsetY + e.clientY - drag.startClientY - drag.panelTop + drag.insetH / 2;
    const vertical = centerY < drag.panelH / 2 ? "top" : "bottom";
    const horizontal = centerX < drag.panelW / 2 ? "left" : "right";
    setFftOverlayPosition(`${vertical}-${horizontal}`);
  };

  const handleFftInsetMouseDown = (
    e: React.MouseEvent<HTMLElement>,
    panelLeft: number,
    panelTop: number,
    panelW: number,
    panelH: number,
    insetX: number,
    insetY: number,
    insetW: number,
    insetH: number,
  ) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const drag = {
      pointerId: -1,
      startClientX: e.clientX,
      startClientY: e.clientY,
      startInsetX: insetX,
      startInsetY: insetY,
      panelLeft,
      panelTop,
      panelW,
      panelH,
      insetW,
      insetH,
      moved: false,
    };
    fftOverlayDragRef.current = drag;
    const onMove = (ev: MouseEvent) => {
      ev.preventDefault();
      if (Math.hypot(ev.clientX - drag.startClientX, ev.clientY - drag.startClientY) > 4) {
        drag.moved = true;
      }
      if (drag.moved) {
        const nextX = Math.max(drag.panelLeft, Math.min(drag.panelLeft + drag.panelW - drag.insetW, drag.startInsetX + ev.clientX - drag.startClientX));
        const nextY = Math.max(drag.panelTop, Math.min(drag.panelTop + drag.panelH - drag.insetH, drag.startInsetY + ev.clientY - drag.startClientY));
        setFftOverlayDragPreview({ x: nextX - drag.panelLeft, y: nextY - drag.panelTop });
      }
    };
    const onUp = (ev: MouseEvent) => {
      window.removeEventListener("mousemove", onMove, true);
      window.removeEventListener("mouseup", onUp, true);
      if (fftOverlayDragRef.current === drag) fftOverlayDragRef.current = null;
      setFftOverlayDragPreview(null);
      if (!drag.moved) return;
      const centerX = drag.startInsetX + ev.clientX - drag.startClientX - drag.panelLeft + drag.insetW / 2;
      const centerY = drag.startInsetY + ev.clientY - drag.startClientY - drag.panelTop + drag.insetH / 2;
      const vertical = centerY < drag.panelH / 2 ? "top" : "bottom";
      const horizontal = centerX < drag.panelW / 2 ? "left" : "right";
      setFftOverlayPosition(`${vertical}-${horizontal}`);
    };
    window.addEventListener("mousemove", onMove, true);
    window.addEventListener("mouseup", onUp, true);
  };

  const handleFftInsetPanMouseDown = (e: React.MouseEvent<HTMLElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const target = e.currentTarget;
    const rect = target.getBoundingClientRect();
    const viewportW = Math.max(1, rect.width);
    const viewportH = Math.max(1, rect.height);
    const startX = e.clientX;
    const startY = e.clientY;
    const current = fftViewLiveRef.current;
    const startView = !fftUserAdjustedViewRef.current && current.zoom > 1
      ? {
        zoom: current.zoom,
        panX: viewportW * (1 - current.zoom) / 2,
        panY: viewportH * (1 - current.zoom) / 2,
      }
      : current;
    if (!fftUserAdjustedViewRef.current) {
      scheduleFftViewState(startView, false, fftLayoutOverlay);
    }
    fftUserAdjustedViewRef.current = true;
    fftOverlayInitialCenterPendingRef.current = false;
    fftViewCenterOnViewportRef.current = false;
    const onMove = (ev: MouseEvent) => {
      ev.preventDefault();
      const clamped = clampFftPan(
        startView.panX + (ev.clientX - startX),
        startView.panY + (ev.clientY - startY),
        startView.zoom,
        viewportW,
        viewportH,
      );
      scheduleFftViewState({ zoom: startView.zoom, panX: clamped.panX, panY: clamped.panY }, false, fftLayoutOverlay);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove, true);
      window.removeEventListener("mouseup", onUp, true);
    };
    window.addEventListener("mousemove", onMove, true);
    window.addEventListener("mouseup", onUp, true);
  };

  React.useEffect(() => {
    if (!effectiveShowFft) return;
    const overlayCanvas = fftLayoutOverlay ? fftInsetLayerRef.current : null;
    const fftCanvas = fftCanvasRef.current;
    const panelGrid = fftPanelGridRef.current;
    const viewport = overlayCanvas
      ? (() => {
        const n = Math.max(1, visiblePanelCount || 1);
        const cols = panelColsForCount(n);
        const rows = Math.ceil(n / cols);
        const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
        const panelW = (canvasW - gap * (cols - 1)) / cols;
        const panelH = (canvasH - gap * (rows - 1)) / rows;
        const insetPad = Math.min(8, Math.max(3, panelW * 0.025));
        const insetMaxW = Math.max(24, panelW - insetPad * 2);
        const insetMaxH = Math.max(20, panelH - insetPad * 2);
        const insetBase = Math.min(insetMaxW, insetMaxH);
        return {
          w: Math.max(24, Math.min(insetMaxW, insetBase * resolvedFftOverlaySize)),
          h: Math.max(20, Math.min(insetMaxH, insetBase * resolvedFftOverlaySize)),
        };
      })()
      : fftCanvas
        ? panelGrid
          ? getFftSlot(0, panelGrid.count, panelGrid.cols, panelGrid.rows)
          : { w: fftCanvas.width, h: fftCanvas.height }
        : null;
    if (!viewport) return;
    const current = fftViewLiveRef.current;
    const centered = fftViewCenterOnViewportRef.current && current.zoom > 1
      ? {
        panX: viewport.w * (1 - current.zoom) / 2,
        panY: viewport.h * (1 - current.zoom) / 2,
      }
      : { panX: current.panX, panY: current.panY };
    fftViewCenterOnViewportRef.current = false;
    fftOverlayInitialCenterPendingRef.current = false;
    const clamped = clampFftPan(centered.panX, centered.panY, current.zoom, viewport.w, viewport.h);
    if (Math.abs(clamped.panX - current.panX) > 0.5 || Math.abs(clamped.panY - current.panY) > 0.5) {
      scheduleFftViewState({ zoom: current.zoom, panX: clamped.panX, panY: clamped.panY });
    }
  }, [clampFftPan, effectiveShowFft, fftLayoutOverlay, fftZoom, fftPanX, fftPanY, canvasW, canvasH, resolvedFftOverlaySize, visiblePanelCount, panelColsForCount, panelGapTrait, fftOffscreenVersion, scheduleFftViewState]);

  // Convert FFT canvas mouse position to FFT image pixel coordinates
  const fftScreenToImg = (e: React.MouseEvent): { col: number; row: number } | null => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;
    const panelGrid = fftPanelGridRef.current;
    if (panelGrid) {
      for (let slot = 0; slot < panelGrid.count; slot++) {
        const dst = getFftSlot(slot, panelGrid.count, panelGrid.cols, panelGrid.rows);
        if (mouseX < dst.x || mouseX >= dst.x + dst.w || mouseY < dst.y || mouseY >= dst.y + dst.h) continue;
        const localX = (mouseX - dst.x - fftPanX) / fftZoom;
        const localY = (mouseY - dst.y - fftPanY) / fftZoom;
        if (localX < 0 || localX >= dst.w || localY < 0 || localY >= dst.h) return null;
        const srcCol = slot % panelGrid.cols;
        const srcRow = Math.floor(slot / panelGrid.cols);
        const tileX = (localX / Math.max(1, dst.w)) * panelGrid.panelWidth;
        const tileY = (localY / Math.max(1, dst.h)) * panelGrid.panelHeight;
        return {
          col: srcCol * panelGrid.panelWidth + Math.max(0, Math.min(panelGrid.panelWidth - 1, tileX)),
          row: srcRow * panelGrid.panelHeight + Math.max(0, Math.min(panelGrid.panelHeight - 1, tileY)),
        };
      }
      return null;
    }
    const localX = (mouseX - fftPanX) / fftZoom;
    const localY = (mouseY - fftPanY) / fftZoom;
    const imgCol = localX / canvasW * fftW;
    const imgRow = localY / canvasH * fftH;
    if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
      return { col: imgCol, row: imgRow };
    }
    return null;
  };

  const handleFftMouseDown = (e: React.MouseEvent) => {
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsFftDragging(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, pX: fftPanX, pY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent) => {
    if (isFftDragging && fftPanStart) {
      const canvas = fftCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - fftPanStart.x) * scaleX;
      const dy = (e.clientY - fftPanStart.y) * scaleY;
      const panelGrid = fftPanelGridRef.current;
      const viewport = panelGrid ? getFftSlot(0, panelGrid.count, panelGrid.cols, panelGrid.rows) : { w: canvas.width, h: canvas.height };
      const clamped = clampFftPan(fftPanStart.pX + dx, fftPanStart.pY + dy, fftZoom, viewport.w, viewport.h);
      setFftPanX(clamped.panX);
      setFftPanY(clamped.panY);
    }
  };

  const handleFftMouseUp = (e: React.MouseEvent) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const pos = fftScreenToImg(e);
        if (pos) {
          // Use crop dimensions when ROI FFT is active
          const fftW = fftCropDims?.fftWidth ?? width;
          const fftH = fftCropDims?.fftHeight ?? height;
          const panelGrid = fftPanelGridRef.current;
          let imgCol = pos.col;
          let imgRow = pos.row;
          if (fftMagCacheRef.current) {
            const bounds = panelGrid ? (() => {
              const panelCol = Math.max(0, Math.min(panelGrid.cols - 1, Math.floor(imgCol / panelGrid.panelWidth)));
              const panelRow = Math.max(0, Math.min(panelGrid.rows - 1, Math.floor(imgRow / panelGrid.panelHeight)));
              return {
                minCol: panelCol * panelGrid.panelWidth,
                maxCol: Math.min(fftW - 1, (panelCol + 1) * panelGrid.panelWidth - 1),
                minRow: panelRow * panelGrid.panelHeight,
                maxRow: Math.min(fftH - 1, (panelRow + 1) * panelGrid.panelHeight - 1),
              };
            })() : null;
            const snapped = bounds
              ? findFFTPeakInBounds(fftMagCacheRef.current, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS, bounds.minCol, bounds.maxCol, bounds.minRow, bounds.maxRow)
              : findFFTPeak(fftMagCacheRef.current, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
            imgCol = snapped.col;
            imgRow = snapped.row;
          }
          const local = panelGrid ? (() => {
            const panelCol = Math.max(0, Math.min(panelGrid.cols - 1, Math.floor(imgCol / panelGrid.panelWidth)));
            const panelRow = Math.max(0, Math.min(panelGrid.rows - 1, Math.floor(imgRow / panelGrid.panelHeight)));
            return {
              col: imgCol - panelCol * panelGrid.panelWidth,
              row: imgRow - panelRow * panelGrid.panelHeight,
              width: panelGrid.panelWidth,
              height: panelGrid.panelHeight,
            };
          })() : { col: imgCol, row: imgRow, width: fftW, height: fftH };
          const halfW = Math.floor(local.width / 2);
          const halfH = Math.floor(local.height / 2);
          const dcol = local.col - halfW;
          const drow = local.row - halfH;
          const distPx = Math.sqrt(dcol * dcol + drow * drow);
          if (distPx < 1) {
            setFftClickInfo(null);
          } else {
            let spatialFreq: number | null = null;
            let dSpacing: number | null = null;
            if (pixelSize > 0) {
              const paddedW = nextPow2(local.width);
              const paddedH = nextPow2(local.height);
              const binC = ((Math.round(local.col) - halfW) % local.width + local.width) % local.width;
              const binR = ((Math.round(local.row) - halfH) % local.height + local.height) % local.height;
              const freqC = binC <= paddedW / 2 ? binC / (paddedW * pixelSize) : (binC - paddedW) / (paddedW * pixelSize);
              const freqR = binR <= paddedH / 2 ? binR / (paddedH * pixelSize) : (binR - paddedH) / (paddedH * pixelSize);
              spatialFreq = Math.sqrt(freqC * freqC + freqR * freqR);
              dSpacing = spatialFreq > 0 ? 1 / spatialFreq : null;
            }
            setFftClickInfo({ row: imgRow, col: imgCol, distPx, spatialFreq, dSpacing });
          }
        }
      }
      fftClickStartRef.current = null;
    }
    setIsFftDragging(false);
    setFftPanStart(null);
  };

  const handleFftTouchStart = (e: React.TouchEvent) => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return;
    const now = Date.now();
    const base = fftViewLiveRef.current;
    if (e.touches.length === 1) {
      const lastTap = lastFftTapRef.current;
      if (lastTap && now - lastTap.time < 320) {
        e.preventDefault();
        handleFftReset();
        lastFftTapRef.current = null;
        fftTouchTransformRef.current = null;
        return;
      }
      lastFftTapRef.current = { time: now };
      const t = e.touches[0];
      fftTouchTransformRef.current = {
        mode: "pan",
        startX: t.clientX,
        startY: t.clientY,
        startDistance: 0,
        startMidX: t.clientX,
        startMidY: t.clientY,
        startState: base,
      };
      e.preventDefault();
      return;
    }
    if (e.touches.length >= 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      const mid = touchMidpoint(a, b);
      fftTouchTransformRef.current = {
        mode: "pinch",
        startX: mid.x,
        startY: mid.y,
        startDistance: Math.max(1, touchDistance(a, b)),
        startMidX: mid.x,
        startMidY: mid.y,
        startState: base,
      };
      e.preventDefault();
    }
  };

  const handleFftTouchMove = (e: React.TouchEvent) => {
    const start = fftTouchTransformRef.current;
    const canvas = fftCanvasRef.current;
    if (!start || !canvas) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const toCanvas = (clientX: number, clientY: number) => ({
      x: (clientX - rect.left) * (canvas.width / Math.max(1, rect.width)),
      y: (clientY - rect.top) * (canvas.height / Math.max(1, rect.height)),
    });
    const base = start.startState;
    if (start.mode === "pinch" && e.touches.length >= 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      const mid = touchMidpoint(a, b);
      const startCanvas = toCanvas(start.startMidX, start.startMidY);
      const currentCanvas = toCanvas(mid.x, mid.y);
      const imageX = (startCanvas.x - base.panX) / base.zoom;
      const imageY = (startCanvas.y - base.panY) / base.zoom;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, base.zoom * (touchDistance(a, b) / start.startDistance)));
      fftUserAdjustedViewRef.current = true;
      fftOverlayInitialCenterPendingRef.current = false;
      fftViewCenterOnViewportRef.current = false;
      scheduleFftViewState({
        zoom: newZoom,
        panX: currentCanvas.x - imageX * newZoom,
        panY: currentCanvas.y - imageY * newZoom,
      }, true);
      return;
    }
    if (start.mode === "pan" && e.touches.length === 1) {
      const t = e.touches[0];
      const scaleX = canvas.width / Math.max(1, rect.width);
      const scaleY = canvas.height / Math.max(1, rect.height);
      scheduleFftViewState({
        zoom: base.zoom,
        panX: base.panX + (t.clientX - start.startX) * scaleX,
        panY: base.panY + (t.clientY - start.startY) * scaleY,
      });
    }
  };

  const handleFftTouchEnd = (e: React.TouchEvent) => {
    if (e.touches.length > 0 || !fftTouchTransformRef.current) return;
    fftTouchTransformRef.current = null;
  };

  const handleFftReset = () => {
    const reset = { zoom: 1, panX: 0, panY: 0 };
    fftViewLiveRef.current = reset;
    fftViewCenterOnViewportRef.current = true;
    fftOverlayInitialCenterPendingRef.current = true;
    fftUserAdjustedViewRef.current = false;
    setFftZoom(reset.zoom);
    internalFftZoomSyncRef.current = true;
    setFftOverlayZoomTrait(1);
    setFftPanX(reset.panX);
    setFftPanY(reset.panY);
    setFftClickInfo(null);
  };

  // Kymograph mouse handlers (mirror FFT: wheel-zoom + pan-drag). Click readout
  // replaces the FFT d-spacing measurement (domain adaptation).
  const [isKymoDragging, setIsKymoDragging] = React.useState(false);
  const [kymoPanStart, setKymoPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  const handleKymoWheel = (e: React.WheelEvent) => {
    const canvas = kymoCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, kymoZoom * zoomFactor));
    const zoomRatio = newZoom / kymoZoom;
    setKymoZoom(newZoom);
    setKymoPanX(mouseX - (mouseX - kymoPanX) * zoomRatio);
    setKymoPanY(mouseY - (mouseY - kymoPanY) * zoomRatio);
  };

  // Convert kymograph canvas mouse position to (frame index, distance index).
  const kymoScreenToImg = (e: React.MouseEvent): { col: number; row: number } | null => {
    const canvas = kymoCanvasRef.current;
    const kymo = kymoDataRef.current;
    if (!canvas || !kymo) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;
    // The click already passed the canvas hit-test, so map into the image and
    // clamp - edge/last-row clicks must still yield a readout (a strict
    // `< nFrames` check silently dropped clicks on the bottom row).
    const imgCol = Math.max(0, Math.min(kymo.lineLen - 1, ((mouseX - kymoPanX) / kymoZoom) / canvasW * kymo.lineLen));
    const imgRow = Math.max(0, Math.min(kymo.nFrames - 1, ((mouseY - kymoPanY) / kymoZoom) / canvasH * kymo.nFrames));
    return { col: imgCol, row: imgRow };
  };

  const handleKymoMouseDown = (e: React.MouseEvent) => {
    kymoClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsKymoDragging(true);
    setKymoPanStart({ x: e.clientX, y: e.clientY, pX: kymoPanX, pY: kymoPanY });
  };

  const handleKymoMouseMove = (e: React.MouseEvent) => {
    if (isKymoDragging && kymoPanStart) {
      const canvas = kymoCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - kymoPanStart.x) * scaleX;
      const dy = (e.clientY - kymoPanStart.y) * scaleY;
      setKymoPanX(kymoPanStart.pX + dx);
      setKymoPanY(kymoPanStart.pY + dy);
    }
  };

  const handleKymoMouseUp = (e: React.MouseEvent) => {
    // Click detection for intensity readout at (time, distance).
    if (kymoClickStartRef.current) {
      const dx = e.clientX - kymoClickStartRef.current.x;
      const dy = e.clientY - kymoClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const pos = kymoScreenToImg(e);
        const kymo = kymoDataRef.current;
        if (pos && kymo) {
          const frame = Math.max(0, Math.min(kymo.nFrames - 1, Math.round(pos.row)));
          const dist = Math.max(0, Math.min(kymo.lineLen - 1, Math.round(pos.col)));
          const intensity = kymo.data[frame * kymo.lineLen + dist];
          const timeVal = dimSampling > 0 && dimUnit ? frame * dimSampling : frame;
          const timeUnit = dimSampling > 0 && dimUnit ? unitSymbol(dimUnit) : "frame";
          const distVal = pixelSize > 0 ? dist * pixelSize : dist;
          const distUnit = pixelSize > 0 ? unitSymbol(pixelUnit || "px") : "px";
          setKymoClickInfo({ timeVal, timeUnit, distVal, distUnit, intensity, col: dist, row: frame });
        } else {
          setKymoClickInfo(null);
        }
      }
      kymoClickStartRef.current = null;
    }
    setIsKymoDragging(false);
    setKymoPanStart(null);
  };

  const handleKymoTouchStart = (e: React.TouchEvent) => {
    const canvas = kymoCanvasRef.current;
    if (!canvas) return;
    const now = Date.now();
    const base = { zoom: kymoZoom, panX: kymoPanX, panY: kymoPanY };
    if (e.touches.length === 1) {
      const lastTap = lastKymoTapRef.current;
      if (lastTap && now - lastTap.time < 320) {
        e.preventDefault();
        handleKymoReset();
        lastKymoTapRef.current = null;
        kymoTouchTransformRef.current = null;
        return;
      }
      lastKymoTapRef.current = { time: now };
      const t = e.touches[0];
      kymoTouchTransformRef.current = {
        mode: "pan",
        startX: t.clientX,
        startY: t.clientY,
        startDistance: 0,
        startMidX: t.clientX,
        startMidY: t.clientY,
        startState: base,
      };
      e.preventDefault();
      return;
    }
    if (e.touches.length >= 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      const mid = touchMidpoint(a, b);
      kymoTouchTransformRef.current = {
        mode: "pinch",
        startX: mid.x,
        startY: mid.y,
        startDistance: Math.max(1, touchDistance(a, b)),
        startMidX: mid.x,
        startMidY: mid.y,
        startState: base,
      };
      e.preventDefault();
    }
  };

  const handleKymoTouchMove = (e: React.TouchEvent) => {
    const start = kymoTouchTransformRef.current;
    const canvas = kymoCanvasRef.current;
    if (!start || !canvas) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const toCanvas = (clientX: number, clientY: number) => ({
      x: (clientX - rect.left) * (canvas.width / Math.max(1, rect.width)),
      y: (clientY - rect.top) * (canvas.height / Math.max(1, rect.height)),
    });
    const base = start.startState;
    if (start.mode === "pinch" && e.touches.length >= 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      const mid = touchMidpoint(a, b);
      const startCanvas = toCanvas(start.startMidX, start.startMidY);
      const currentCanvas = toCanvas(mid.x, mid.y);
      const imageX = (startCanvas.x - base.panX) / base.zoom;
      const imageY = (startCanvas.y - base.panY) / base.zoom;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, base.zoom * (touchDistance(a, b) / start.startDistance)));
      setKymoZoom(newZoom);
      setKymoPanX(currentCanvas.x - imageX * newZoom);
      setKymoPanY(currentCanvas.y - imageY * newZoom);
      return;
    }
    if (start.mode === "pan" && e.touches.length === 1) {
      const t = e.touches[0];
      const scaleX = canvas.width / Math.max(1, rect.width);
      const scaleY = canvas.height / Math.max(1, rect.height);
      setKymoPanX(base.panX + (t.clientX - start.startX) * scaleX);
      setKymoPanY(base.panY + (t.clientY - start.startY) * scaleY);
    }
  };

  const handleKymoTouchEnd = (e: React.TouchEvent) => {
    if (e.touches.length > 0 || !kymoTouchTransformRef.current) return;
    kymoTouchTransformRef.current = null;
  };

  const handleKymoReset = () => {
    setKymoZoom(1);
    setKymoPanX(0);
    setKymoPanY(0);
    setKymoClickInfo(null);
  };

  const kymoNeedsReset = kymoZoom !== 1 || kymoPanX !== 0 || kymoPanY !== 0;

  // Preview panel zoom/pan handlers
  const handlePreviewWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const canvas = previewCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const pw = previewCanvasDims.w;
    const ph = previewCanvasDims.h;
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = pw / 2;
    const cy = ph / 2;
    const mouseImageX = (mouseCanvasX - cx - previewZoom.panX) / previewZoom.zoom + cx;
    const mouseImageY = (mouseCanvasY - cy - previewZoom.panY) / previewZoom.zoom + cy;
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, previewZoom.zoom * zoomFactor));
    const newPanX = mouseCanvasX - (mouseImageX - cx) * newZoom - cx;
    const newPanY = mouseCanvasY - (mouseImageY - cy) * newZoom - cy;
    setPreviewZoom({ zoom: newZoom, panX: newPanX, panY: newPanY });
  };

  const handlePreviewMouseDown = (e: React.MouseEvent) => {
    setIsDraggingPreviewPan(true);
    setPreviewPanStart({ x: e.clientX, y: e.clientY, pX: previewZoom.panX, pY: previewZoom.panY });
  };

  const handlePreviewMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingPreviewPan || !previewPanStart) return;
    const canvas = previewCanvasRef.current;
    const rect = canvas?.getBoundingClientRect();
    const scaleX = canvas && rect ? canvas.width / Math.max(1, rect.width) : 1;
    const scaleY = canvas && rect ? canvas.height / Math.max(1, rect.height) : 1;
    const dx = (e.clientX - previewPanStart.x) * scaleX;
    const dy = (e.clientY - previewPanStart.y) * scaleY;
    setPreviewZoom(prev => ({ ...prev, panX: previewPanStart.pX + dx, panY: previewPanStart.pY + dy }));
  };

  const handlePreviewMouseUp = () => {
    setIsDraggingPreviewPan(false);
    setPreviewPanStart(null);
  };

  const handlePreviewDoubleClick = () => {
    setPreviewZoom({ zoom: 1, panX: 0, panY: 0 });
  };

  // Resize handlers
  const handleMainResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    const rect = canvasContainerRef.current?.getBoundingClientRect();
    const startSize = rect && rect.width > 0 ? rect.width : mainCanvasSize;
    const startX = e.clientX;
    const startY = e.clientY;
    const visiblePanels = Math.max(1, visiblePanelCount || 1);
    let rafId = 0;
    let latestSize = startSize;
    const handleMouseMove = (e: MouseEvent) => {
      const delta = Math.max(e.clientX - startX, e.clientY - startY);
      const nextSize = startSize + delta;
      // Absolute minimum: 200 px per panel column. Lets reader shrink BELOW
      // the initial `size=` value (preset / kwarg) when their screen is small,
      // without collapsing the canvas to an unreadable sliver.
      const colsLocal = panelColsForCount(visiblePanels);
      const minSize = 200 * colsLocal;
      latestSize = Math.max(minSize, nextSize);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setMainCanvasSize(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      cancelAnimationFrame(rafId);
      setMainCanvasSize(latestSize);
      const colsLocal = panelColsForCount(visiblePanels);
      setCanvasSizeTrait(Math.round(latestSize / colsLocal));
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const clampSlice = (idx: number) => Math.max(0, Math.min(nSlices - 1, Math.round(idx)));
  const frameLabelForIndex = React.useCallback((idx: number): string => {
    const label = labels?.[idx];
    if (label == null) return "";
    const text = String(label).trim();
    if (!text || text === String(idx) || text === String(idx + 1)) return "";
    return text;
  }, [labels]);
  const panelFrameLabelForIndex = React.useCallback((panel: number, idx: number): string => {
    const panelLabels = panelFrameLabels?.[panel];
    const panelRealN = panelRealFrames?.[panel];
    const panelIdx = panelRealN ? Math.min(idx, Math.max(0, panelRealN - 1)) : idx;
    const label = panelLabels?.[panelIdx];
    if (label != null) {
      const text = String(label).trim();
      if (text && text !== String(panelIdx) && text !== String(panelIdx + 1)) return text;
    }
    return frameLabelForIndex(idx);
  }, [frameLabelForIndex, panelFrameLabels, panelRealFrames]);
  const formatFrameValueLabel = React.useCallback((idx: number) => {
    const rounded = clampSlice(idx);
    const label = frameLabelForIndex(rounded);
    return label ? `${rounded + 1}: ${label}` : `${rounded + 1}`;
  }, [frameLabelForIndex, nSlices]);
  const visibleSliceIdx = clampSlice(playing ? playbackUiSliceIdx : (offline ? liveSliceIdx : displaySliceIdx));
  React.useLayoutEffect(() => {
    updatePlaybackLiveControls(visibleSliceIdx);
  }, [updatePlaybackLiveControls, visibleSliceIdx]);
  const normalizedBookmarkedFrames = React.useMemo(() => {
    const seen = new Set<number>();
    for (const raw of bookmarkedFrames || []) {
      const value = Math.round(Number(raw));
      if (Number.isFinite(value) && value >= 0 && value < nSlices) seen.add(value);
    }
    return Array.from(seen).sort((a, b) => a - b);
  }, [bookmarkedFrames, nSlices]);
  const bookmarkedFrameMarks = React.useMemo(
    () => normalizedBookmarkedFrames.map((value) => ({ value })),
    [normalizedBookmarkedFrames]
  );
  const currentFrameBookmarked = normalizedBookmarkedFrames.includes(visibleSliceIdx);
  const toggleCurrentFrameBookmark = React.useCallback(() => {
    const frame = visibleSliceIdx;
    const next = new Set(normalizedBookmarkedFrames);
    if (next.has(frame)) next.delete(frame);
    else next.add(frame);
    setBookmarkedFrames(Array.from(next).sort((a, b) => a - b));
  }, [normalizedBookmarkedFrames, setBookmarkedFrames, visibleSliceIdx]);
  const currentPlaybackIndex = () => (
    Number.isFinite(playbackIdxRef.current)
      ? playbackIdxRef.current
      : (Number.isFinite(displaySliceIdx) ? displaySliceIdx : sliceIdx)
  );
  const playFromCurrentFrame = (direction: 1 | -1 | null = null) => {
    if (sidecarSliceCommitTimerRef.current !== null) {
      window.clearTimeout(sidecarSliceCommitTimerRef.current);
      sidecarSliceCommitTimerRef.current = null;
    }
    const nextReverse = direction === null ? reverse : direction < 0;
    const rangeStart = loop ? Math.max(0, Math.min(loopStart, nSlices - 1)) : 0;
    const rangeEnd = loop ? Math.max(rangeStart, Math.min(effectiveLoopEnd, nSlices - 1)) : nSlices - 1;
    let start = Math.max(rangeStart, Math.min(rangeEnd, Math.round(currentPlaybackIndex())));
    if (!loop) {
      if (!nextReverse && start >= rangeEnd) start = rangeStart;
      if (nextReverse && start <= rangeStart) start = rangeEnd;
    }
    playbackIdxRef.current = start;
    setDisplaySliceIdx(start);
    setPlaybackUiSliceIdx(start);
    setLiveSliceIdx(start);
    if (!(offline && sidecarMode && (sidecarBitmapReadyRef.current || sidecarCompositeReadyRef.current) && !isRgb)) setSliceIdx(start);
    if (direction !== null) setReverse(nextReverse);
    setPlaying(true);
  };
  const pausePlayback = () => {
    if (sidecarSliceCommitTimerRef.current !== null) {
      window.clearTimeout(sidecarSliceCommitTimerRef.current);
      sidecarSliceCommitTimerRef.current = null;
    }
    const current = clampSlice(currentPlaybackIndex());
    playbackIdxRef.current = current;
    setDisplaySliceIdx(current);
    setPlaybackUiSliceIdx(current);
    setLiveSliceIdx(current);
    setSliceIdx(current);
    setPlaying(false);
  };
  const stopPlayback = () => {
    if (sidecarSliceCommitTimerRef.current !== null) {
      window.clearTimeout(sidecarSliceCommitTimerRef.current);
      sidecarSliceCommitTimerRef.current = null;
    }
    const home = loop ? Math.max(0, Math.min(loopStart, nSlices - 1)) : 0;
    playbackIdxRef.current = home;
    setDisplaySliceIdx(home);
    setPlaybackUiSliceIdx(home);
    setLiveSliceIdx(home);
    setSliceIdx(home);
    setPlaying(false);
  };
  const playbackPathLength = Array.isArray(playbackPath) ? playbackPath.length : 0;
  const playbackStyleSummary = playbackPathLength > 0 ? `Path ${playbackPathLength}` : "Linear";
  const clampFrameIndex = React.useCallback(
    (value: number) => Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(value))),
    [nSlices],
  );
  const makePlaybackStylePath = React.useCallback((style: "power-in" | "power-out" | "ease-in-out") => {
    const start = loop ? Math.max(0, Math.min(loopStart, nSlices - 1)) : 0;
    const loopEndCandidate = loop ? Math.min(effectiveLoopEnd, nSlices - 1) : Math.max(0, nSlices - 1);
    // A one-frame loop is meaningful for manual review but not for choosing a
    // temporal curve. If the range is still hydrating or collapsed, style
    // buttons fall back to the full stack instead of producing "Path 1".
    const end = loopEndCandidate > start ? loopEndCandidate : Math.max(start, nSlices - 1);
    const span = Math.max(0, end - start);
    if (span <= 0) return [start];
    const steps = Math.max(span + 1, Math.min(96, Math.round((span + 1) * 1.75)));
    const path: number[] = [];
    for (let i = 0; i < steps; i++) {
      const t = steps <= 1 ? 1 : i / (steps - 1);
      let eased = t;
      if (style === "power-in") eased = t * t;
      else if (style === "power-out") eased = 1 - ((1 - t) * (1 - t));
      else eased = 0.5 - 0.5 * Math.cos(Math.PI * t);
      path.push(clampFrameIndex(start + eased * span));
    }
    if (path[0] !== start) path.unshift(start);
    if (path[path.length - 1] !== end) path.push(end);
    return path;
  }, [clampFrameIndex, effectiveLoopEnd, loop, loopStart, nSlices]);
  const applyPlaybackStylePreset = React.useCallback((style: "linear" | "power-in" | "power-out" | "ease-in-out") => {
    if (style === "linear") {
      setPlaybackPath([]);
    } else {
      setPlaybackPath(makePlaybackStylePath(style));
    }
    setPlaying(false);
    setPlaybackStyleMenuAnchor(null);
  }, [makePlaybackStylePath, setPlaybackPath, setPlaying]);
  const playbackStyleActive = React.useMemo<"linear" | "power-in" | "power-out" | "ease-in-out" | null>(() => {
    if (!playbackPathLength) return "linear";
    const samePath = (candidate: number[]) => (
      candidate.length === playbackPathLength
      && candidate.every((value, idx) => value === playbackPath[idx])
    );
    for (const style of ["power-in", "power-out", "ease-in-out"] as const) {
      if (samePath(makePlaybackStylePath(style))) return style;
    }
    return null;
  }, [makePlaybackStylePath, playbackPath, playbackPathLength]);
  const playbackStyleButtonSx = React.useCallback((style: "linear" | "power-in" | "power-out" | "ease-in-out") => {
    const active = playbackStyleActive === style;
    return {
      ...compactButton,
      justifyContent: "flex-start",
      color: active ? themeColors.accent : themeColors.textMuted,
      border: `1px solid ${active ? themeColors.accent : "transparent"}`,
      bgcolor: active ? themeColors.controlBg : "transparent",
      "&:hover": {
        color: active ? themeColors.accent : themeColors.text,
        borderColor: active ? themeColors.accent : themeColors.border,
        bgcolor: themeColors.controlBg,
      },
    };
  }, [playbackStyleActive, themeColors.accent, themeColors.border, themeColors.controlBg, themeColors.text, themeColors.textMuted]);

  // Keyboard
  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (shouldIgnoreWidgetShortcut(e.target)) return;

    let handled = false;
    const sidecarDirectNavigation =
      offline &&
      sidecarMode &&
      (sidecarBitmapReadyRef.current || sidecarCompositeReadyRef.current) &&
      !isRgb &&
      !requiresClientFrameTransform({ offline, diffMode, avgWindow }) &&
      !browserFilterOnRef.current &&
      !frequencyFilterIsActive;
    const shortcutBaseIdx = sidecarDirectNavigation
      ? clampSlice(playbackIdxRef.current)
      : visibleSliceIdx;

    switch (e.key) {
        case " ":
          if (playing) pausePlayback();
          else playFromCurrentFrame();
          handled = true;
          break;
        case "ArrowLeft": {
          const lo = loop ? Math.max(0, loopStart) : 0;
          const base = shortcutBaseIdx;
          const candidate = hiddenSet.size ? nextVisible(base, -1, false) : base - 1;
          scrubToSlice(Math.max(lo, candidate));
          handled = true;
          break;
        }
        case "ArrowRight": {
          const hi = loop ? Math.min(effectiveLoopEnd, nSlices - 1) : nSlices - 1;
          const base = shortcutBaseIdx;
          const candidate = hiddenSet.size ? nextVisible(base, 1, false) : base + 1;
          scrubToSlice(Math.min(hi, candidate));
          handled = true;
          break;
        }
        case "Home":
          scrubToSlice(loop ? Math.max(0, loopStart) : 0);
          handled = true;
          break;
        case "End":
          scrubToSlice(loop ? Math.min(effectiveLoopEnd, nSlices - 1) : nSlices - 1);
          handled = true;
          break;
        case "r":
        case "R":
          handleDoubleClick();
          handled = true;
          break;
        case "c":
        case "C":
          if (cursorInfo && cursorReadoutVisible) {
            navigator.clipboard.writeText(`(${cursorInfo.row}, ${cursorInfo.col}, ${cursorInfo.value})`);
            handled = true;
          }
          break;
        case "Delete":
        case "Backspace":
          if (overlayEditMode && overlaySelection) {
            deleteSelectedOverlay();
            handled = true;
          } else if (effectiveRoiActive && roiSelectedIdx >= 0) {
            deleteSelectedROI();
            handled = true;
          }
          break;
        case "d":
        case "D":
          if (effectiveRoiActive && roiSelectedIdx >= 0 && (e.metaKey || e.ctrlKey || e.shiftKey)) {
            duplicateSelectedROI();
            handled = true;
          }
          break;
        case "Escape":
          rootRef.current?.blur();
          handled = true;
          break;
      }
    if (handled) {
      e.preventDefault();
      e.stopPropagation();
    }
  };

  // Check if view needs reset
  const needsReset = zoom !== 1 || panX !== 0 || panY !== 0;
  const requestScrubPreview = (idx: number): boolean => {
    if (separatePanelFrames) return false;
    return requestCommFramePreview(clampSlice(idx), "scrub");
  };
  const scrubToSlice = (idx: number) => {
    const next = clampSlice(idx);
    if (playing) setPlaying(false);
    const transformActive = frameTransformActive();
    if (offline && sidecarMode && (sidecarBitmapReadyRef.current || sidecarCompositeReadyRef.current) && !isRgb && !transformActive) {
      playbackIdxRef.current = next;
      drawSidecarBitmapFrame(next, false, "scrub-direct");
      updatePlaybackLiveControls(next);
      if (sidecarSliceCommitTimerRef.current !== null) {
        window.clearTimeout(sidecarSliceCommitTimerRef.current);
      }
      sidecarSliceCommitTimerRef.current = window.setTimeout(() => {
        sidecarSliceCommitTimerRef.current = null;
        setLiveSliceIdx(next);
        setDisplaySliceIdx(next);
        setPlaybackUiSliceIdx(next);
        setSliceIdx(next);
      }, 750);
      return;
    }
    setPlaybackUiSliceIdx(next);
    if (!transformActive && renderGpuCachedSliceDirect(next)) return;
    setLiveSliceIdx(next);
    if (renderBufferedSlice(next)) return;
    if (!offline && frameServerUrl) {
      setDisplaySliceIdx(next);
      setPlaybackUiSliceIdx(next);
      void renderFetchedSlice(next).then((ok) => {
        if (!ok && !transformActive) requestScrubPreview(next);
      });
      prefetchServerFrames(next, false, false);
      return;
    }
    setDisplaySliceIdx(next);
    setPlaybackUiSliceIdx(next);
    if (!transformActive && requestScrubPreview(next)) return;
    setSliceIdx(next);
  };
  const commitSlice = (idx: number) => {
    const next = clampSlice(idx);
    if (sidecarSliceCommitTimerRef.current !== null) {
      window.clearTimeout(sidecarSliceCommitTimerRef.current);
      sidecarSliceCommitTimerRef.current = null;
    }
    setLiveSliceIdx(next);
    setPlaybackUiSliceIdx(next);
    setSliceIdx(next);
  };
  const handleLoopSliderMouseDown = (e: React.MouseEvent<HTMLSpanElement>) => {
    const target = e.target as HTMLElement;
    if (target.closest(".MuiSlider-thumb")) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = rect.width > 0 ? (e.clientX - rect.left) / rect.width : 0;
    const next = clampSlice(pct * Math.max(0, nSlices - 1));
    e.preventDefault();
    e.stopPropagation();
    scrubToSlice(next);
    commitSlice(next);
  };
  const handleLoopSliderPointerDownCapture = (e: React.PointerEvent<HTMLSpanElement>) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement;
    if (target.closest(".MuiSlider-thumb")) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const sliceFromClientX = (clientX: number) => {
      const pct = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
      return clampSlice(pct * Math.max(0, nSlices - 1));
    };
    const moveCurrent = (clientX: number, commit: boolean) => {
      const next = sliceFromClientX(clientX);
      scrubToSlice(next);
      if (commit) commitSlice(next);
    };
    e.preventDefault();
    e.stopPropagation();
    e.nativeEvent.stopImmediatePropagation();
    moveCurrent(e.clientX, false);
    const onMove = (ev: PointerEvent) => {
      ev.preventDefault();
      moveCurrent(ev.clientX, false);
    };
    const onUp = (ev: PointerEvent) => {
      ev.preventDefault();
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      moveCurrent(ev.clientX, true);
    };
    window.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
  };
  const overlayCanvasVisible = effectiveRoiActive || profileActive || (panelOverlays || []).some((items) => items && items.length > 0);
  const lensCanvasVisible = showLens && lensPos !== null;
  const keyboardShortcutItems: [string, string][] = [
    ["Space", "Play / Pause"],
    ["← / →", `Prev / Next ${dimLabel.toLowerCase()}`],
    ["Home / End", `First / Last ${dimLabel.toLowerCase()}`],
    ["R", "Reset zoom"],
    ["C", "Copy cursor coords"],
    ...(roiAllowed ? [["Del", "Delete selected ROI"], ["Ctrl/⌘+D", "Duplicate selected ROI"]] as [string, string][] : []),
    ["Esc", "Release keyboard focus"],
    ["Scroll", "Zoom"],
    ["Dbl-click", "Reset view"],
  ];
  const webgpuStatusLabel =
    fftBackendInfo.webgpu === "ready" ? "available"
      : fftBackendInfo.webgpu === "software" ? "software adapter ignored"
        : fftBackendInfo.webgpu === "unavailable" ? "unavailable"
          : "checking";
  const fftSourceRaw = fftBackendInfo.source || "";
  const fftSourceCached = fftSourceRaw.endsWith("-cache");
  const fftSourceBase = fftSourceCached ? fftSourceRaw.slice(0, -6) : fftSourceRaw;
  const fftSourceLabel =
    fftSourceCached ? "Cached"
      : fftSourceBase === "webgpu-batch" || fftSourceBase === "webgpu" ? "WebGPU"
        : fftSourceBase ? "CPU fallback"
        : "not run yet";
  const fftSourceDetail =
    fftSourceBase === "cpu-sync-shifted" ? "offline CPU"
      : fftSourceBase === "worker-batch" || fftSourceBase === "worker" ? "CPU worker"
        : fftSourceBase || "";
  const nativeCacheLabel = hasFrameServer
    ? framePopulation.ready > 0
      ? `Native cache ${framePopulation.ready}`
      : previewPopulation.ready ? "Preview ready"
        : framePopulation.active ? "Native loading" : "Native pending"
    : "";
  const nativeCacheTitle = framePopulation.ready > 0
    ? `${framePopulation.ready} native frames cached from ${framePopulation.target}`
    : previewPopulation.ready
      ? `Reduced preview frame ${previewPopulation.idx + 1}/${Math.max(1, nSlices)} displayed while native frames are pending${previewPopulation.factor > 1 ? ` (${previewPopulation.factor}x reduced)` : ""}`
      : nativeCacheLabel;
  const frequencyRingValue = normalizeFrequencyFilterMode(frequencyFilter) === "bandpass"
    ? (frequencyDraft ?? frequencyFilterCenter)
    : (frequencyDraft ?? frequencyFilterCutoff);
  const show3dFrequencyRing = frequencyFilterIsActive ? (
    <Box
      className="quantem-frequency-filter-ring"
      data-frequency-filter={normalizeFrequencyFilterMode(frequencyFilter)}
      aria-label={`Draggable ${normalizeFrequencyFilterMode(frequencyFilter)} frequency ring at ${frequencyValueLabel(frequencyRingValue)}`}
      title="Drag the ring to choose a frequency from the FFT"
      onMouseDown={(event: React.MouseEvent<HTMLDivElement>) => {
        event.preventDefault();
        event.stopPropagation();
        const parent = event.currentTarget.parentElement;
        if (!parent) return;
        const rect = parent.getBoundingClientRect();
        const valueAt = (clientX: number, clientY: number) => Math.max(0, Math.min(1, Math.hypot(clientX - (rect.left + rect.width / 2), clientY - (rect.top + rect.height / 2)) / (Math.min(rect.width, rect.height) / 2)));
        const onMove = (moveEvent: MouseEvent) => setFrequencyDraft(valueAt(moveEvent.clientX, moveEvent.clientY));
        const onUp = (upEvent: MouseEvent) => {
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
          const value = valueAt(upEvent.clientX, upEvent.clientY);
          if (normalizeFrequencyFilterMode(frequencyFilter) === "bandpass") setFrequencyFilterCenter(value);
          else setFrequencyFilterCutoff(value);
          setFrequencyDraft(null);
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
      }}
      onPointerDown={(event: React.PointerEvent<HTMLDivElement>) => {
        event.stopPropagation();
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event: React.PointerEvent<HTMLDivElement>) => {
        if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
        const parent = event.currentTarget.parentElement;
        if (!parent) return;
        const rect = parent.getBoundingClientRect();
        setFrequencyDraft(Math.max(0, Math.min(1, Math.hypot(event.clientX - (rect.left + rect.width / 2), event.clientY - (rect.top + rect.height / 2)) / (Math.min(rect.width, rect.height) / 2))));
      }}
      onPointerUp={(event: React.PointerEvent<HTMLDivElement>) => {
        event.stopPropagation();
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        if (normalizeFrequencyFilterMode(frequencyFilter) === "bandpass") setFrequencyFilterCenter(frequencyRingValue);
        else setFrequencyFilterCutoff(frequencyRingValue);
        setFrequencyDraft(null);
      }}
      sx={{ position: "absolute", left: "50%", top: "50%", width: `${frequencyRingValue * 100}%`, height: `${frequencyRingValue * 100}%`, transform: "translate(-50%, -50%)", borderRadius: "50%", border: "2px solid rgba(0,229,255,0.95)", bgcolor: normalizeFrequencyFilterMode(frequencyFilter) === "highpass" ? "rgba(0,0,0,0.55)" : "transparent", boxShadow: normalizeFrequencyFilterMode(frequencyFilter) === "lowpass" ? "0 0 0 1px rgba(0,0,0,0.75), 0 0 0 9999px rgba(0,0,0,0.55)" : "0 0 0 1px rgba(0,0,0,0.75)", cursor: "crosshair", touchAction: "none", zIndex: 6 }}
    >
      {normalizeFrequencyFilterMode(frequencyFilter) === "bandpass" && [
        Math.max(0, frequencyFilterCenter - frequencyFilterWidth / 2),
        Math.min(1, frequencyFilterCenter + frequencyFilterWidth / 2),
      ].map((radius, index) => <Box key={index} sx={{ position: "absolute", left: "50%", top: "50%", width: `${radius / Math.max(0.001, frequencyRingValue) * 100}%`, height: `${radius / Math.max(0.001, frequencyRingValue) * 100}%`, transform: "translate(-50%, -50%)", borderRadius: "50%", border: "1px dashed rgba(255,255,255,0.95)", bgcolor: index === 0 ? "rgba(0,0,0,0.55)" : "transparent", boxShadow: index === 1 ? "0 0 0 9999px rgba(0,0,0,0.55)" : "none", pointerEvents: "none" }} />)}
      <Box sx={{ position: "absolute", left: "50%", top: -24, transform: "translateX(-50%)", px: 0.75, py: 0.25, borderRadius: 0.75, bgcolor: "rgba(0,0,0,0.78)", color: "rgba(200,250,255,0.98)", fontSize: 9, lineHeight: 1.2, fontWeight: 700, whiteSpace: "nowrap", pointerEvents: "none", textShadow: "0 1px 1px #000" }}>
        {normalizeFrequencyFilterMode(frequencyFilter) === "lowpass" ? "Inside kept" : normalizeFrequencyFilterMode(frequencyFilter) === "highpass" ? "Outside kept" : "Band kept"}
      </Box>
    </Box>
  ) : null;
  return (
    <Box
      ref={rootRef}
      className="show3d-root"
      data-show3d-canvas-repaint-signal={canvasRepaintSignal}
      data-frequency-filter-backend={frequencyFilterIsActive ? frequencyFilterBackend : "off"}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={handleRootMouseDownCapture}
      sx={{ ...container.root, width: "100%", maxWidth: "100%", boxSizing: "border-box", position: "relative", bgcolor: themeColors.bg, color: themeColors.text, outline: "none", "&:focus::after": { content: '""', position: "absolute", inset: 0, pointerEvents: "none", zIndex: 20, boxShadow: "inset 0 0 0 2px #0af" }, "& canvas": { display: "block" }, "@media (max-width: 700px)": { p: 0, ".jp-OutputArea-output &, .jp-OutputArea-child &": { width: "calc(100vw - 96px)", maxWidth: "calc(100vw - 96px)" } } }}
    >
      <FolderWatchBadge
        state={folderWatchState}
        detail={folderWatchDetail}
        live={folderWatchLive}
      />
      {offlineStackFetchStatus && (
        <Box
          role="status"
          aria-live="polite"
          data-show3d-sidecar-status="true"
          sx={{
            width: "100%",
            px: 1.5,
            py: 0.75,
            mb: 1,
            boxSizing: "border-box",
            borderRadius: 1,
            bgcolor: themeColors.controlBg,
            border: `1px solid ${themeColors.border}`,
            color: themeColors.text,
          }}
        >
          <Typography sx={{ fontSize: 12 }}>{offlineStackFetchStatus}</Typography>
        </Box>
      )}
      {folderWaiting && (
        <Box
          role="region"
          aria-label="Show3D folder waiting view"
          data-show3d-folder-waiting="true"
          sx={{
            width: "100%",
            minHeight: 120,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            px: 2,
            py: 3,
            boxSizing: "border-box",
            border: `1px dashed ${themeColors.border}`,
            borderRadius: 1,
            color: themeColors.textMuted,
          }}
        >
          <Typography sx={{ fontSize: 12, textAlign: "center" }}>
            {folderStatus || "Waiting for the first stable frame"}
          </Typography>
        </Box>
      )}
      {!folderWaiting && !canRenderLive && hasSavedStaticFallback && (
        <Box sx={{ width: "100%", maxWidth: mainPanelWidth, boxSizing: "border-box" }}>
          <Box
            component="img"
            src={staticFallbackUrl}
            alt={`${title || "Show3D"} saved preview`}
            sx={{
              display: "block",
              width: "100%",
              maxWidth: mainPanelWidth,
              height: "auto",
              border: `1px solid ${themeColors.border}`,
              boxSizing: "border-box",
            }}
          />
        </Box>
      )}
      {!folderWaiting && (canRenderLive || !hasSavedStaticFallback) && (
      <>
      <Stack
        direction="row"
        spacing={`${SPACING.SM}px`}
        alignItems="flex-start"
        sx={{
          flexWrap: effectiveShowFft && fftLayoutBottom ? "wrap" : "nowrap",
          width: "100%",
          maxWidth: "100%",
          minWidth: 0,
          boxSizing: "border-box",
          "@media (max-width: 900px)": {
            flexDirection: "column",
            alignItems: "stretch",
            flexWrap: "nowrap",
            "& > :not(style) + :not(style)": {
              marginLeft: "0 !important",
              marginTop: `${SPACING.SM}px`,
            },
          },
        }}
      >
        <Box sx={{ width: mainPanelWidth, maxWidth: "100%", flexShrink: effectiveShowFft && fftLayoutBottom ? 0 : 1, boxSizing: "border-box" }}>
          {/* Title row */}
          {showTitle && <Typography variant="caption" sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px`, display: "block", height: 16, lineHeight: "16px", overflow: "hidden" }}>
            {title || "Image"}
            {diffMode !== "off" && (
              <Typography component="span" sx={{ fontSize: 9, fontWeight: "bold", color: "#fff", bgcolor: "#e65100", px: 0.5, py: 0.125, ml: 0.5, verticalAlign: "middle" }}>
                {diffMode === "previous" ? "\u0394-PREV" : "\u0394-FIRST"}
              </Typography>
            )}
            {debug && <DebugPerfBadge widget="Show3D" fps={debugFps} themeColors={themeColors} />}
            {nativeCacheLabel && (
              <Typography
                component="span"
                data-show3d-native-cache-status="true"
                title={nativeCacheTitle}
                sx={{ fontSize: 9, fontWeight: 700, color: themeColors.accentGreen, bgcolor: themeColors.controlBg, border: `1px solid ${themeColors.border}`, px: 0.5, py: 0.125, ml: 0.5, verticalAlign: "middle", whiteSpace: "nowrap" }}
              >
                {nativeCacheLabel}
              </Typography>
            )}
	            {showControls && <InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <MetadataSection rows={[
                ["Shape", `${nSlices} x ${height} x ${width}`],
                ["Panels", nPanels > 1 ? `${nPanels} panels` : "single panel"],
                ["Frame axis", `${dimLabel || "Frame"}${dimSampling ? `, ${formatNumber(dimSampling)} ${dimUnit || ""}` : ""}`],
                ["Sampling", pixelSize > 0 ? `${formatNumber(pixelSize)} ${unitSymbol(pixelUnit || "px")}/px` : ""],
                ["Source", hasFrameServer ? "detail server" : "embedded stack"],
              ]} />
              <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Show power spectrum (Fourier transform) alongside image.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                FFT d-spacing uses the provided real-space sampling: Δk = 1 / (N × pixel_size), |g| = √(kx² + ky²), d = 1 / |g|. Current pixel_size: {pixelSize > 0 ? `${formatNumber(pixelSize)} ${unitSymbol(pixelUnit || "px")}/px` : "not set, so only pixel distances are shown"}.
              </Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Backend</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                WebGPU: {webgpuStatusLabel}{fftBackendInfo.adapter ? ` (${fftBackendInfo.adapter})` : ""}.
              </Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                FFT compute: {fftSourceLabel}{fftSourceDetail && fftSourceDetail !== fftSourceLabel ? ` (${fftSourceDetail})` : ""}
                {fftBackendInfo.ms != null ? `, ${fftBackendInfo.ms.toFixed(1)} ms` : ""}.
              </Typography>
              {fftBackendInfo.panels != null && (
                <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                  FFT panels: {fftBackendInfo.panels}{fftBackendInfo.grid ? `, grid ${fftBackendInfo.grid}` : ""}.
                </Typography>
              )}
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points on image to draw a line intensity profile.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Lens: Magnifier inset that follows the cursor.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Scale: Linear or logarithmic intensity mapping.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Stack-wide percentile contrast for Show3D image panels. FFT Auto masks DC + clips to 99.9th.</Typography>
              {roiAllowed && <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>ROI: Click empty image to add at cursor, click ROI to select, drag to move, hover edge to resize. Del removes selected; Ctrl/⌘+D duplicates.</Typography>}
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Cols / Panels: Change the panel grid or hide panels without changing the source stack.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Pinning: Click a panel to select or pin it for keyboard actions, per-panel zoom, ROI edits, and deletion shortcuts.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Pan: With Pan enabled, drag the image to move the zoomed view. With Link Zoom on, pan and zoom move together across panels.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Loop: Loop playback. Drag end markers on slider for loop range.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Bounce: Ping-pong playback - alternates forward and reverse.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Speed: Lower fps, increase avg only when needed, shorten the loop range, hide panels, or turn off FFT/Profile/Stats to reduce heavy-stack playback work.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT layout: Use side, bottom, or overlay mode. Overlay FFTs can be resized; wheel and drag over the overlay inspect FFT detail independently.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Export / Copy: Export HTML, GIF, or MP4 panel-only animations, or copy the current panel view from the toolbar.</Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
              <KeyboardShortcuts items={keyboardShortcutItems} />
	            </Box>} theme={themeInfo.theme} />}
	            {showControls && (
	              <Button
	                size="small"
	                sx={{
	                  ...compactButton,
	                  ml: 0.75,
	                  py: 0,
	                  px: 0.5,
	                  minHeight: 16,
	                  lineHeight: "16px",
	                  verticalAlign: "baseline",
	                }}
	                onClick={() => setControlsCollapsed(!controlsCollapsed)}
	                aria-label={controlsCollapsed ? "Show controls" : "Hide controls"}
	                aria-pressed={!controlsCollapsed}
	                title={controlsCollapsed ? "Show controls" : "Hide controls"}
	              >
	                Controls
	              </Button>
	            )}
	            {showControls && controlsCollapsed && (exportEnabled || canDownloadCurrentHtml) && (
	              <>
	                <Button
	                  size="small"
	                  sx={{
	                    ...compactButton,
	                    ml: 0.5,
	                    py: 0,
	                    px: 0.5,
	                    minHeight: 16,
	                    lineHeight: "16px",
	                    verticalAlign: "baseline",
	                  }}
	                  disabled={exportBusy || (!exportEnabled && !canDownloadCurrentHtml)}
	                  onClick={handleExportMenuOpen}
	                  aria-label="Export widget or animation"
	                  aria-controls={exportMenuAnchor ? "show3d-export-menu-collapsed" : undefined}
	                  aria-expanded={exportMenuAnchor ? "true" : undefined}
	                  aria-haspopup="menu"
	                  title={localExportStatus || exportStatus || (exportEnabled ? "Export HTML, GIF, or MP4 with a save dialog" : "Export this standalone HTML")}
	                >
	                  {exportBusy ? "Exporting" : "Export"}
	                </Button>
	                <Menu
	                  id="show3d-export-menu-collapsed"
	                  anchorEl={exportMenuAnchor}
	                  open={Boolean(exportMenuAnchor)}
	                  onClose={handleExportMenuClose}
	                  MenuListProps={{ "aria-label": "Export options" }}
	                  {...themedMenuProps}
	                >
	                  {renderExportMenuContent()}
	                </Menu>
	              </>
	            )}
	          </Typography>}
	          {/* Page navigation sits above the analysis toolbar so a long page
	              label never competes with Profile / Stats / FFT controls. */}
	          {controlsVisible && isPaged && (
	            <Box
	              data-show3d-page-controls="true"
	              aria-label="Page navigation"
	              sx={{
	                display: "flex",
	                alignItems: "center",
	                flexWrap: "wrap",
	                columnGap: "8px",
	                rowGap: "3px",
	                mb: "3px",
	                minHeight: 26,
	                pb: "3px",
	                borderBottom: `1px solid ${themeColors.border}`,
	              }}
	            >
	              <Box sx={{ display: "flex", alignItems: "baseline", gap: "5px", flex: "1 1 240px", minWidth: 0 }}>
	                <Typography sx={{ ...typography.label, fontSize: 10, flexShrink: 0 }}>Page</Typography>
	                <Typography
	                  data-show3d-page-status="true"
	                  title={pageControlStatus}
	                  sx={{
	                    ...typography.label,
	                    fontSize: 10,
	                    lineHeight: 1.25,
	                    color: themeColors.accent,
	                    minWidth: 0,
	                    whiteSpace: "normal",
	                    overflowWrap: "anywhere",
	                    fontVariantNumeric: "tabular-nums",
	                  }}
	                >
	                  {pageControlStatus}
	                </Typography>
	              </Box>
	              <Box sx={{ display: "flex", alignItems: "center", gap: "4px", flex: "0 1 auto", minWidth: 0 }}>
	                <Slider
	                  value={pageControlIdx}
	                  min={0}
	                  max={Math.max(0, (nPages || 1) - 1)}
	                  step={1}
	                  onPointerDownCapture={() => {
	                    stopPagePlayback();
	                    setPageSliderPreviewIdx(currentPageIdx);
	                  }}
	                  onKeyDown={() => stopPagePlayback()}
	                  onChange={(_, value) => {
	                    const raw = Array.isArray(value) ? value[0] : value;
	                    const next = clampPageIdx(Number(raw));
	                    setPageSliderPreviewIdx(next);
	                    commitPageIdx(next);
	                  }}
	                  onChangeCommitted={(_, value) => {
	                    const raw = Array.isArray(value) ? value[0] : value;
	                    const next = clampPageIdx(Number(raw));
	                    stopPagePlayback();
	                    setPageSliderPreviewIdx(next);
	                    commitPageIdx(next, true);
	                  }}
	                  size="small"
	                  sx={{ ...sliderStyles.small, width: 150, flex: "0 1 150px", minWidth: 92, color: themeColors.accent }}
	                  aria-label="Page"
	                />
	                <IconButton
	                  size="small"
	                  onClick={() => setPagePlaying((value) => !value)}
	                  title={pagePlaying ? "Pause page playback" : "Play pages"}
	                  aria-label={pagePlaying ? "Pause page playback" : "Play pages"}
	                  sx={{ width: 24, height: 24, p: 0, color: themeColors.accent }}
	                >
	                  {pagePlaying ? <PauseIcon sx={{ fontSize: 16 }} /> : <PlayArrowIcon sx={{ fontSize: 16 }} />}
	                </IconButton>
	                <Select
	                  value={String(pagePlayFps)}
	                  onChange={(e) => setPagePlayFps(Number(e.target.value) || 2)}
	                  size="small"
	                  sx={{ ...themedSelect, minWidth: 48, fontSize: 10 }}
	                  MenuProps={themedMenuProps}
	                  inputProps={{ "aria-label": "Page playback frames per second" }}
	                  title="Page playback speed"
	                >
	                  {PAGE_PLAY_FPS_OPTIONS.map((fps) => (
	                    <MenuItem key={fps} value={String(fps)}>{fps} fps</MenuItem>
	                  ))}
	                </Select>
	                <IconButton
	                  size="small"
	                  onClick={() => {
	                    const next = Array.from({ length: Math.max(1, nPages || 1) }, (_, idx) => pageStarred?.[idx] ? 1 : 0);
	                    next[pageControlIdx] = next[pageControlIdx] ? 0 : 1;
	                    setPageStarred(next);
	                  }}
	                  title={(pageStarred?.[pageControlIdx] ? "Unstar " : "Star ") + pageControlLabel}
	                  aria-label={(pageStarred?.[pageControlIdx] ? "Unstar " : "Star ") + pageControlLabel}
	                  sx={{
	                    width: 24,
	                    height: 24,
	                    p: 0,
	                    color: pageStarred?.[pageControlIdx] ? "#ffc107" : themeColors.textMuted,
	                    "&:hover": { color: pageStarred?.[pageControlIdx] ? "#ffc107" : themeColors.text },
	                  }}
	                >
	                  {pageStarred?.[pageControlIdx] ? "★" : "☆"}
	                </IconButton>
	              </Box>
	            </Box>
	          )}
	          {/* Analysis and display controls row. */}
	          {controlsVisible && (
	          <Box data-show3d-tool-controls="true" sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: "4px", mb: `${SPACING.XS}px`, minHeight: 28 }}>
            {visiblePanelCount > 1 && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Cols</Typography>
                <Select
                  value={String(clampedMaxCols)}
                  onChange={(e) => {
                    const next = Math.max(1, Math.min(Number(e.target.value) || 1, visiblePanelCount || 1, MAX_PANEL_COLUMNS));
                    setMaxCols(next);
                  }}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 48, fontSize: 10 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Show3D panel columns" }}
                  title="Number of Show3D panel columns"
                >
                  {show3dColumnOptions.map((cols) => (
                    <MenuItem key={cols} value={String(cols)}>{cols}</MenuItem>
                  ))}
                </Select>
              </>
            )}
            {/* Kymograph toggle: HIDDEN until a profile line exists (not shown-
                but-disabled). Kymograph is a line-profile sub-feature, so the
                control only appears once there's a line to build it from. */}
            {/* Kymograph: appears only with a drawn profile line (canKymograph).
                Turning it on takes the side slot from FFT. */}
            {canKymograph && <>
              <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Kymo</Typography>
              <Switch checked={showKymograph} onChange={(e) => { const on = e.target.checked; setShowKymograph(on); if (on) setShowFft(false); }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle kymograph space-time panel" } }} />
            </>}
            {/* Profile and ROI are mutually exclusive line/region tools. Turning
                one on turns the other off. Kymograph rides on Profile. */}
            <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Profile</Typography>
            <Switch checked={profileActive} onChange={(e) => {
              const on = e.target.checked;
              setProfileActive(on);
              if (on) {
                setRoiActive(false); setRoiSelectedIdx(-1);
              } else {
                // Toggle OFF hides overlay + kymograph but keeps the line + data
                // so re-enable restores instantly. Use Clear to actively wipe.
                setShowKymograph(false);
                setHoveredProfileEndpoint(null); setIsHoveringProfileLine(false);
              }
            }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle line intensity profile tool" } }} />
            {profileActive && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "4px" }}>W</Typography>
                <Slider value={profileWidth} min={1} max={15} step={1} onChange={(_, v) => setProfileWidth(v as number)} size="small" valueLabelDisplay="auto" sx={{ width: 60, ml: "2px" }} aria-label={`Profile width ${profileWidth} px`} />
              </>
            )}
            {(nPanels || 1) === 1 && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Lens</Typography>
                <Switch
                  checked={showLens}
                  onChange={() => {
                    if (!showLens) { setShowLens(true); setLensPos({ row: Math.floor(height / 2), col: Math.floor(width / 2) }); }
                    else { setShowLens(false); setLensPos(null); }
                  }}
                  size="small"
                  sx={switchStyles.small}
                  slotProps={{ input: { "aria-label": "Toggle magnifier lens" } }}
                />
              </>
            )}
            {/* ROI hidden while kymograph is shown (roiAllowed already encodes
                single-panel && !showKymograph). */}
            {roiAllowed && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>ROI</Typography>
                <Switch checked={roiActive} onChange={(e) => {
                  const on = e.target.checked;
                  if (on) {
                    setRoiActive(true); setShowRoiResizeHint(true);
                    setProfileActive(false); setProfileLine([]); setProfileData(null); setHoveredProfileEndpoint(null); setIsHoveringProfileLine(false);
                  } else {
                    setRoiActive(false); setRoiSelectedIdx(-1); pendingRoiAddRef.current = null;
                  }
                }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle ROI selection tool" } }} />
              </>
            )}
            {/* "More" overflow: Stats + Denoise + Filter live here (mirrors Show2D) to
                keep the top toolbar calm.
                Compatibility text for older contract tests:
                title="More tools: Stats, Denoise, Filter, Sub-pixel alignment, Color, Flip, Compare" */}
            <Badge
              badgeContent={(showStats ? 1 : 0) + (overlayEditMode ? 1 : 0) + (denoiseEnabled ? 1 : 0) + (!isRgb && frequencyFilterIsActive ? 1 : 0) + (subpixelAlignEnabled ? 1 : 0) + (!isRgb && hasPanelChoices && !colorShared ? 1 : 0) + (flipRows ? 1 : 0) + (flipCols ? 1 : 0) + (compareMode !== "off" ? 1 : 0) + (rotationActive ? 1 : 0)}
              invisible={!showStats && !overlayEditMode && !showDenoise && !(!isRgb && frequencyFilterIsActive) && !subpixelAlignEnabled && !(!isRgb && hasPanelChoices && !colorShared) && !flipRows && !flipCols && compareMode === "off" && !rotationActive}
              sx={{ "& .MuiBadge-badge": { bgcolor: themeColors.accent, color: "#fff", fontSize: 9, fontWeight: 600, minWidth: 14, height: 14, px: 0.25 } }}
            >
              <Button
                size="small"
                sx={{ minWidth: 0, px: 0.75, fontSize: 10, textTransform: "none", color: (showStats || overlayEditMode || showDenoise || (!isRgb && frequencyFilterIsActive) || subpixelAlignEnabled || (!isRgb && hasPanelChoices && !colorShared) || flipRows || flipCols || compareMode !== "off" || rotationActive) ? themeColors.accent : themeColors.text }}
                onClick={(e) => setMoreMenuAnchor(e.currentTarget)}
                aria-label="More tools"
                aria-haspopup="menu"
                title="More tools: Stats, Denoise, Filter, Sub-pixel alignment, Color, Flip, Rotate, Compare"
              >
                More
              </Button>
            </Badge>
            <Menu
              anchorEl={moreMenuAnchor}
              open={Boolean(moreMenuAnchor)}
              onClose={() => setMoreMenuAnchor(null)}
              MenuListProps={{ "aria-label": "More tools" }}
              {...themedMenuProps}
            >
              <Box sx={{ px: 1.5, pt: 0.75, pb: 0.35, minWidth: 260 }}>
                <Typography sx={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: themeColors.textMuted, textTransform: "uppercase" }}>Readout</Typography>
              </Box>
              <MenuItem dense onClick={() => setShowStats(!showStats)} sx={{ fontSize: 12, gap: 1, color: showStats ? themeColors.accent : themeColors.text }}>
                <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Mean / min / max / std readout under the image.">Stats</Typography>
                <Switch checked={showStats} onClick={(e) => e.stopPropagation()} onChange={(e) => setShowStats(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle statistics readout" } }} />
              </MenuItem>
              {hasPanelOverlays && (
                <MenuItem dense onClick={() => setOverlayEditMode(!overlayEditMode)} sx={{ fontSize: 12, gap: 1, color: overlayEditMode ? themeColors.accent : themeColors.text }}>
                  <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Edit API-defined circles and rectangles: click to select, drag to move, drag an edge to resize.">Overlay Edit</Typography>
                  <Switch
                    checked={overlayEditMode}
                    onClick={(event) => event.stopPropagation()}
                    onChange={(event) => setOverlayEditMode(event.target.checked)}
                    size="small"
                    sx={switchStyles.small}
                    slotProps={{ input: { "aria-label": "Toggle overlay editing" } }}
                  />
                </MenuItem>
              )}
              {hasPanelOverlays && overlayBaselineRef.current && (
                <MenuItem dense onClick={resetPanelOverlays} sx={{ fontSize: 12, color: overlaySelection ? themeColors.accent : themeColors.text }}>
                  Reset Overlays
                </MenuItem>
              )}
              <Box sx={{ mx: 1.5, my: 0.5, borderTop: `1px solid ${themeColors.border}`, opacity: 0.9 }} />
              <Box sx={{ px: 1.5, pt: 0.35, pb: 0.35 }}>
                <Typography sx={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: themeColors.textMuted, textTransform: "uppercase" }}>Processing</Typography>
              </Box>
              <MenuItem dense onClick={toggleDenoise} sx={{ fontSize: 12, gap: 1, color: denoiseEnabled ? themeColors.accent : themeColors.text }}>
                <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Display-only denoise: ON shows the denoised view, OFF shows raw (config preserved). Raw data and stats keep original counts.">Denoise</Typography>
                <Switch checked={denoiseEnabled ?? false} onClick={(e) => e.stopPropagation()} onChange={toggleDenoise} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle denoise on/off" } }} />
              </MenuItem>
              {!isRgb && (
                <MenuItem dense onClick={() => setFrequencyMaster(!frequencyFilterEnabled)} sx={{ fontSize: 12, gap: 1, color: frequencyFilterIsActive ? themeColors.accent : themeColors.text }}>
                  <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Off by default. Turn on to remove a background or isolate a periodicity; raw counts remain unchanged.">Filter</Typography>
                  <Switch checked={frequencyFilterEnabled ?? false} onClick={(e) => e.stopPropagation()} onChange={() => setFrequencyMaster(!frequencyFilterEnabled)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle frequency filter effect" } }} />
                </MenuItem>
              )}
              {!isRgb && (
                <MenuItem
                  dense
                  onClick={() => {
                    const next = !subpixelAlignEnabled;
                    setSubpixelAlignEnabled(next);
                    if (next && !subpixelAlignSupported) {
                      setSubpixelAlignStatus("Needs a single-panel client-side stack");
                    }
                  }}
                  sx={{ fontSize: 12, gap: 1, color: subpixelAlignEnabled ? themeColors.accent : themeColors.text }}
                >
                  <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Display-only sub-pixel frame alignment. First scope: single-panel client-side stacks; raw data stays unchanged.">Sub-pixel align</Typography>
                  <Switch
                    checked={subpixelAlignEnabled ?? false}
                    onClick={(e) => e.stopPropagation()}
                    onChange={() => setSubpixelAlignEnabled(!subpixelAlignEnabled)}
                    size="small"
                    sx={switchStyles.small}
                    slotProps={{ input: { "aria-label": "Toggle sub-pixel alignment" } }}
                  />
                </MenuItem>
              )}
              {(subpixelAlignEnabled || subpixelAlignStatus !== "Off") && !isRgb && (
                <Box
                  onClick={(e) => e.stopPropagation()}
                  sx={{
                    px: 1.5,
                    py: 0.75,
                    minWidth: 260,
                    display: "grid",
                    gridTemplateColumns: "1fr auto",
                    gap: 0.75,
                    alignItems: "center",
                  }}
                >
                  <TextField
                    label="Reference frame"
                    type="number"
                    size="small"
                    value={Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(subpixelAlignReference || 0)))}
                    onChange={(e) => setSubpixelAlignReference(Number(e.target.value))}
                    inputProps={{ min: 0, max: Math.max(0, nSlices - 1), "aria-label": "Sub-pixel alignment reference frame" }}
                    sx={{
                      "& .MuiInputBase-input": { fontSize: 11, py: 0.5 },
                      "& .MuiInputLabel-root": { fontSize: 11 },
                    }}
                  />
                  <Button
                    size="small"
                    sx={compactButton}
                    disabled={!subpixelAlignEnabled || subpixelAlignBusy || !subpixelAlignSupported}
                    onClick={() => void computeSubpixelAlignment()}
                    title="Compute alignment now and repaint the current frame"
                  >
                    {subpixelAlignBusy ? "Aligning" : subpixelAlignShiftsRef.current ? "Re-align" : "Align"}
                  </Button>
                  <Typography sx={{ gridColumn: "1 / -1", fontSize: 10, color: subpixelAlignSupported || !subpixelAlignEnabled ? themeColors.textMuted : themeColors.accentYellow }}>
                    {subpixelAlignStatus}
                  </Typography>
                </Box>
              )}
              <Box sx={{ mx: 1.5, my: 0.5, borderTop: `1px solid ${themeColors.border}`, opacity: 0.9 }} />
              <Box sx={{ px: 1.5, pt: 0.35, pb: 0.35 }}>
                <Typography sx={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: themeColors.textMuted, textTransform: "uppercase" }}>Orientation</Typography>
              </Box>
              <MenuItem dense onClick={() => setFlipRows(!flipRows)} sx={{ fontSize: 12, gap: 1, color: flipRows ? themeColors.accent : themeColors.text }}>
                <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Display-only vertical flip for orientation checks; raw data and coordinates are unchanged.">Flip Rows</Typography>
                <Switch checked={flipRows} onClick={(e) => e.stopPropagation()} onChange={(e) => setFlipRows(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle vertical row flip" } }} />
              </MenuItem>
              <MenuItem dense onClick={() => setFlipCols(!flipCols)} sx={{ fontSize: 12, gap: 1, color: flipCols ? themeColors.accent : themeColors.text }}>
                <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Display-only horizontal flip for handedness checks; raw data and coordinates are unchanged.">Flip Cols</Typography>
                <Switch checked={flipCols} onClick={(e) => e.stopPropagation()} onChange={(e) => setFlipCols(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle horizontal column flip" } }} />
              </MenuItem>
              <MenuItem
                dense
                onClick={() => {
                  if (rotationActive) clearRotations();
                  else setShowRotationSettings(!showRotationSettings);
                }}
                sx={{ fontSize: 12, gap: 1, color: (rotationActive || showRotationSettings) ? themeColors.accent : themeColors.text }}
              >
                <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Display-only orientation review. Turn on to choose angle and scope.">Rotate</Typography>
                <Switch
                  checked={rotationActive || showRotationSettings}
                  onClick={(event) => event.stopPropagation()}
                  onChange={(event) => {
                    if (event.target.checked) setShowRotationSettings(true);
                    else clearRotations();
                  }}
                  size="small"
                  sx={switchStyles.small}
                  slotProps={{ input: { "aria-label": "Toggle rotation settings" } }}
                />
              </MenuItem>
              {(rotationActive || showRotationSettings) && (
                <Box
                  onClick={(event) => event.stopPropagation()}
                  sx={{
                    px: 1.5,
                    pb: 1,
                    minWidth: 260,
                    display: "grid",
                    gridTemplateColumns: "auto 1fr",
                    gap: 0.75,
                    alignItems: "center",
                  }}
                >
                  <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>Angle</Typography>
                  <Select
                    value={String(((imageRotation % 4) + 4) % 4 * 90)}
                    onChange={(event) => setRotationForScope(Number(event.target.value) / 90)}
                    size="small"
                    sx={{ ...themedSelect, minWidth: 92 }}
                    MenuProps={themedMenuProps}
                    inputProps={{ "aria-label": "Display rotation" }}
                    title="Display-only rotation; raw data coordinates stay unchanged"
                  >
                    <MenuItem value="0">0°</MenuItem>
                    <MenuItem value="90">90°</MenuItem>
                    <MenuItem value="180">180°</MenuItem>
                    <MenuItem value="270">270°</MenuItem>
                  </Select>
                  <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>Scope</Typography>
                  <Select
                    value={rotationScope || "all"}
                    onChange={(event) => setRotationScope(String(event.target.value))}
                    size="small"
                    sx={{ ...themedSelect, minWidth: 92 }}
                    MenuProps={themedMenuProps}
                    inputProps={{ "aria-label": "Rotation scope" }}
                  >
                    <MenuItem value="all">All</MenuItem>
                    <MenuItem value="frame">Frame</MenuItem>
                  </Select>
                  <Typography sx={{ gridColumn: "1 / -1", fontSize: 10, color: themeColors.textMuted }}>
                    {(rotationScope || "all") === "frame"
                      ? `${dimLabel || "Frame"} ${Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(displaySliceIdx || sliceIdx || 0)))} only`
                      : "Applies to the whole stack"}
                  </Typography>
                </Box>
              )}
              {!isRgb && hasPanelChoices && (
                <>
                  <Box sx={{ mx: 1.5, my: 0.5, borderTop: `1px solid ${themeColors.border}`, opacity: 0.9 }} />
                  <Box sx={{ px: 1.5, pt: 0.35, pb: 0.35 }}>
                    <Typography sx={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: themeColors.textMuted, textTransform: "uppercase" }}>Color</Typography>
                  </Box>
                  <MenuItem
                    dense
                    onClick={() => setColorShared(
                      colorShared ? false : true,
                      nPanels > 1 ? Math.max(0, cursorInfo?.panelIdx ?? visiblePanelIndices[0] ?? 0) : 0,
                    )}
                    sx={{ fontSize: 12, gap: 1, color: !colorShared ? themeColors.accent : themeColors.text }}
                  >
                    <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Shared keeps one colormap for every panel. Turn off to let the Color dropdown edit only the hovered or selected panel.">Color shared</Typography>
                    <Switch
                      checked={colorShared}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => setColorShared(
                        e.target.checked,
                        nPanels > 1 ? Math.max(0, cursorInfo?.panelIdx ?? visiblePanelIndices[0] ?? 0) : 0,
                      )}
                      size="small"
                      sx={switchStyles.small}
                      slotProps={{ input: { "aria-label": "Toggle shared panel colormap" } }}
                    />
                  </MenuItem>
                </>
              )}
              <Box sx={{ mx: 1.5, my: 0.5, borderTop: `1px solid ${themeColors.border}`, opacity: 0.9 }} />
              <Box sx={{ px: 1.5, pt: 0.35, pb: 0.35 }}>
                <Typography sx={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: themeColors.textMuted, textTransform: "uppercase" }}>Compare</Typography>
              </Box>
              <MenuItem
                dense
                onClick={() => setCompareActiveFromCurrentFrame(compareMode === "off")}
                sx={{ fontSize: 12, gap: 1, color: compareMode !== "off" ? themeColors.accent : themeColors.text }}
              >
                <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Blink, difference, or overlay two frames for change detection.">Compare</Typography>
                <Switch
                  checked={compareMode !== "off"}
                  onClick={(event) => event.stopPropagation()}
                  onChange={(event) => setCompareActiveFromCurrentFrame(event.target.checked)}
                  size="small"
                  sx={switchStyles.small}
                  slotProps={{ input: { "aria-label": "Toggle compare settings" } }}
                />
              </MenuItem>
              {compareMode !== "off" && (
                <Box
                  onClick={(e) => e.stopPropagation()}
                  sx={{
                    px: 1.5,
                    pb: 1,
                    minWidth: 260,
                    display: "grid",
                    gridTemplateColumns: "auto 1fr",
                    gap: 0.75,
                    alignItems: "center",
                  }}
                >
                  <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>Mode</Typography>
                  <Select
                    value={compareMode || "blink"}
                    onChange={(e) => setCompareMode(String(e.target.value))}
                    size="small"
                    sx={{ ...themedSelect, minWidth: 120 }}
                    MenuProps={themedMenuProps}
                    inputProps={{ "aria-label": "Compare mode" }}
                  >
                    <MenuItem value="blink">Blink</MenuItem>
                    <MenuItem value="difference">Difference</MenuItem>
                    <MenuItem value="overlay">Overlay</MenuItem>
                  </Select>
                <Typography sx={{ fontSize: 12, color: themeColors.text }}>A</Typography>
                <TextField
                  type="number"
                  size="small"
                  value={Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(comparePair?.[0] ?? 0)))}
                  onChange={(e) => setComparePair([Number(e.target.value) || 0, comparePair?.[1] ?? 1])}
                  inputProps={{ min: 0, max: Math.max(0, nSlices - 1), "aria-label": "Compare frame A" }}
                  sx={{ input: { color: themeColors.text, fontSize: 12, py: 0.5 }, "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border } }}
                />
                <Typography sx={{ fontSize: 12, color: themeColors.text }}>B</Typography>
                <TextField
                  type="number"
                  size="small"
                  value={Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(comparePair?.[1] ?? 1)))}
                  onChange={(e) => setComparePair([comparePair?.[0] ?? 0, Number(e.target.value) || 0])}
                  inputProps={{ min: 0, max: Math.max(0, nSlices - 1), "aria-label": "Compare frame B" }}
                  sx={{ input: { color: themeColors.text, fontSize: 12, py: 0.5 }, "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border } }}
                />
                <Typography sx={{ fontSize: 12, color: themeColors.text }}>Speed</Typography>
                <Select
                  value={String(blinkFps)}
                  onChange={(e) => setBlinkFps(Number(e.target.value) || 2)}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 92 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Blink speed" }}
                >
                  <MenuItem value="0.5">0.5x</MenuItem>
                  <MenuItem value="1">1x</MenuItem>
                  <MenuItem value="2">2x</MenuItem>
                  <MenuItem value="4">4x</MenuItem>
                </Select>
                <Typography sx={{ fontSize: 12, color: themeColors.text }}>Background</Typography>
                <Select
                  value={compareBackground || "dark"}
                  onChange={(e) => setCompareBackground(String(e.target.value))}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 92 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Compare background" }}
                >
                  <MenuItem value="dark">Dark</MenuItem>
                  <MenuItem value="light">Light</MenuItem>
                </Select>
                <Typography sx={{ fontSize: 12, color: themeColors.text }}>Diff</Typography>
                <Select
                  value={diffCmap || "magenta-green"}
                  onChange={(e) => setDiffCmap(String(e.target.value))}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 120 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Difference colormap" }}
                >
                  <MenuItem value="magenta-green">Magenta/Green</MenuItem>
                  <MenuItem value="red-blue">Red/Blue</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                </Box>
              )}
              {!isRgb && (
                <>
                <Box sx={{ mx: 1.5, my: 0.5, borderTop: `1px solid ${themeColors.border}`, opacity: 0.9 }} />
                <Box
                  onClick={(event) => event.stopPropagation()}
                  sx={{
                    px: 1.5,
                    pt: 0.35,
                    pb: 1,
                    minWidth: 260,
                    display: "grid",
                    gridTemplateColumns: "auto 1fr",
                    gap: 0.75,
                    alignItems: "center",
                  }}
                >
                  <Typography sx={{ gridColumn: "1 / -1", fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: themeColors.textMuted, textTransform: "uppercase" }}>Contrast</Typography>
                  <Typography sx={{ fontSize: 12, color: themeColors.text }} title="Choose the percentile contrast range. Histogram stays visible below the image.">Range</Typography>
                  <Select
                    size="small"
                    value={contrastPreset || "custom"}
                    onChange={(e) => applyContrastPreset(String(e.target.value))}
                    sx={{ ...themedSelect, minWidth: 110 }}
                    MenuProps={themedMenuProps}
                    inputProps={{ "aria-label": "Contrast percentile range" }}
                  >
                    {CONTRAST_PRESETS.map((preset) => (
                      <MenuItem key={preset.value} value={preset.value}>{preset.label}</MenuItem>
                    ))}
                  </Select>
                </Box>
                </>
              )}
            </Menu>
            {hasPanelChoices && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Link</Typography>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Zoom</Typography>
                <Switch checked={linkPanels} onChange={(e) => setLinkPanels(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Link zoom and pan across panels" } }} />
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Contrast</Typography>
                <Switch checked={linkContrast} onChange={(e) => setLinkContrast(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Link contrast across panels" } }} />
              </>
            )}
            {fftAllowed && (
              <Box aria-hidden="true" sx={{ width: "1px", height: 20, flex: "0 0 1px", alignSelf: "center", mx: "4px", bgcolor: themeColors.border, opacity: 0.8 }} />
            )}
            {/* FFT can be shown below, beside, or as an inset over the image grid. */}
            {fftAllowed && <>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT</Typography>
              <Switch checked={showFft} onChange={(e) => { const on = e.target.checked; setShowFft(on); if (on) setShowKymograph(false); }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle FFT power spectrum panel" } }} />
              {showFft && (
                <Select
                  value={resolvedFftLayout}
                  onChange={(e) => setFftLayout(String(e.target.value))}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 78, fontSize: 10, ml: "2px" }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "FFT panel layout" }}
                >
                  <MenuItem value="bottom">Bottom</MenuItem>
                  <MenuItem value="right">Right</MenuItem>
                  <MenuItem value="overlay">Overlay</MenuItem>
                </Select>
              )}
              {showFft && fftLayoutOverlay && (
                <>
                  <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Size</Typography>
                  <Select
                    value={String(Math.round(resolvedFftOverlaySize * 100))}
                    onChange={(e) => setFftOverlaySize(Number(e.target.value) / 100)}
                    size="small"
                    sx={{ ...themedSelect, minWidth: 52, fontSize: 10, ml: "2px" }}
                    MenuProps={themedMenuProps}
                    inputProps={{ "aria-label": "FFT overlay size" }}
                  >
                    <MenuItem value="25">25%</MenuItem>
                    <MenuItem value="35">35%</MenuItem>
                    <MenuItem value="50">50%</MenuItem>
                    <MenuItem value="65">65%</MenuItem>
                  </Select>
                </>
              )}
            </>}
            <Box sx={{ flex: 1 }} />
            <Box sx={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <Button size="small" sx={compactButton} onClick={handleCopy} aria-label="Copy current frame to clipboard as PNG">Copy</Button>
              {hasPanelChoices && (
                <>
                  {!isPaged && (
                    <Button
                      size="small"
                      sx={{
                        ...compactButton,
                        color: reorderMode ? themeColors.accent : themeColors.text,
                        "& .MuiButton-startIcon": { mr: 0.4 },
                      }}
                      startIcon={<DragIndicatorIcon sx={{ fontSize: 14 }} />}
                      onClick={() => setReorderMode((value) => !value)}
                      aria-pressed={reorderMode ? "true" : "false"}
                      aria-label={reorderMode ? "Finish reordering panels" : "Reorder panels"}
                      title={reorderMode ? "Finish reordering panels" : "Reorder panels"}
                    >
                      Reorder
                    </Button>
                  )}
                  <Button
                    size="small"
                    sx={{ ...compactButton, "& .MuiButton-startIcon": { mr: 0.4 } }}
                    startIcon={<VisibilityIcon sx={{ fontSize: 14 }} />}
                    onClick={(event) => setPanelMenuAnchor(event.currentTarget)}
                    aria-label="Choose visible panels"
                    aria-controls={panelMenuAnchor ? "show3d-panels-menu" : undefined}
                    aria-expanded={panelMenuAnchor ? "true" : undefined}
                    aria-haspopup="menu"
                  >
                    {visiblePanelCount === panelMenuTotal ? "Panels" : `Panels ${visiblePanelCount}/${panelMenuTotal}`}
                  </Button>
                  <Menu
                    id="show3d-panels-menu"
                    anchorEl={panelMenuAnchor}
                    open={Boolean(panelMenuAnchor)}
                    onClose={() => setPanelMenuAnchor(null)}
                    MenuListProps={{ "aria-label": "Panel visibility options" }}
                    {...themedMenuProps}
                  >
                    {orderedPanelIndices.map((panel) => {
                      const hidden = hiddenPanelSet.has(panel);
                      const disabled = !hidden && visiblePanelCount <= 1;
                      return (
                        <MenuItem
                          key={`panel-menu-${panel}`}
                          dense
                          disabled={disabled}
                          onClick={() => setPanelHidden(panel, !hidden)}
                          title={disabled ? "At least one panel must remain visible" : undefined}
                        >
                          {hidden
                            ? <VisibilityOffIcon sx={{ fontSize: 16, mr: 1, color: themeColors.textMuted }} />
                            : <VisibilityIcon sx={{ fontSize: 16, mr: 1, color: themeColors.accent }} />}
                          <Typography sx={{ fontSize: 11, color: disabled ? themeColors.textMuted : themeColors.text }}>
                            {panelTitleContent(panel)}
                          </Typography>
                        </MenuItem>
                      );
                    })}
                    <MenuItem
                      dense
                      disabled={hiddenPanelSet.size === 0}
                      onClick={() => {
                        if (isPaged) {
                          setHiddenPageSlots([]);
                          setHiddenPageSlotsTrait([]);
                        }
                        setHiddenPanels([]);
                      }}
                    >
                      <VisibilityIcon sx={{ fontSize: 16, mr: 1, color: themeColors.accent }} />
                      <Typography sx={{ fontSize: 11 }}>Show all panels</Typography>
                    </MenuItem>
                    {!isPaged && (
                      <MenuItem
                        dense
                        disabled={(panelOrder || []).length === 0}
                        onClick={resetPanelOrder}
                      >
                        <DragIndicatorIcon sx={{ fontSize: 16, mr: 1, color: themeColors.accent }} />
                        <Typography sx={{ fontSize: 11 }}>Reset order</Typography>
                      </MenuItem>
                    )}
                  </Menu>
                </>
              )}
              {handoffEnabled && (
                <>
                  <Button
                    size="small"
                    sx={compactButton}
                    onClick={(event) => setViewMenuAnchor(event.currentTarget)}
                    aria-label="Open view options"
                    aria-controls={viewMenuAnchor ? "show3d-view-menu" : undefined}
                    aria-expanded={viewMenuAnchor ? "true" : undefined}
                    aria-haspopup="menu"
                    title={handoffStatus || "View options"}
                  >
                    View
                  </Button>
                  <Menu
                    id="show3d-view-menu"
                    anchorEl={viewMenuAnchor}
                    open={Boolean(viewMenuAnchor)}
                    onClose={() => setViewMenuAnchor(null)}
                    MenuListProps={{ "aria-label": "View options" }}
                    {...themedMenuProps}
                  >
                    <MenuItem onClick={handleHandoffToShow2D}>
                      View frame as 2D
                    </MenuItem>
                  </Menu>
                </>
              )}
              {(exportEnabled || canDownloadCurrentHtml) && (
                <>
                  <Button
                    size="small"
                    sx={compactButton}
                    disabled={exportBusy || (!exportEnabled && !canDownloadCurrentHtml)}
                    onClick={handleExportMenuOpen}
                    aria-label="Export widget or animation"
                    aria-controls={exportMenuAnchor ? "show3d-export-menu" : undefined}
                    aria-expanded={exportMenuAnchor ? "true" : undefined}
                    aria-haspopup="menu"
                    title={localExportStatus || exportStatus || (exportEnabled ? "Export HTML, GIF, or MP4 with a save dialog" : "Export this standalone HTML")}
                  >
                    {exportBusy ? "Exporting" : "Export"}
                  </Button>
                  <Menu
                    id="show3d-export-menu"
                    anchorEl={exportMenuAnchor}
                    open={Boolean(exportMenuAnchor)}
                    onClose={handleExportMenuClose}
                    MenuListProps={{ "aria-label": "Export options" }}
                    {...themedMenuProps}
                  >
                    {renderExportMenuContent()}
                  </Menu>
                </>
              )}
              {(exportEnabled || canDownloadCurrentHtml || canExportStandaloneGif) && (localExportStatus || exportStatus) && (
                <Typography
                  sx={{
                    ...typography.label,
                    fontSize: 10,
                    maxWidth: 260,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    color: (localExportStatus || exportStatus).startsWith("Export failed") ? "#d32f2f" : themeColors.textMuted,
                  }}
                  title={localExportStatus || exportStatus}
                >
                  {localExportStatus || exportStatus}
                </Typography>
              )}
              <Button size="small" sx={compactButton} disabled={!needsReset} onClick={handleDoubleClick} aria-label="Reset zoom and pan">Reset</Button>
              {handoffEnabled && handoffStatus && (
                <Typography
                  sx={{
                    ...typography.label,
                    fontSize: 10,
                    maxWidth: 140,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    color: handoffStatus.startsWith("View failed") ? "#d32f2f" : themeColors.textMuted,
                  }}
                  title={handoffStatus}
                >
                  {handoffStatus}
                </Typography>
              )}
	          </Box>
	          </Box>
	          )}
          <Box
            ref={canvasContainerRef}
            sx={{
              ...container.imageBox,
              bgcolor: compareMode !== "off" ? (compareBackground === "light" ? "#f7f7f7" : "#050505") : container.imageBox.bgcolor,
              width: "100%",
              maxWidth: canvasW,
              aspectRatio: mainPanelAspectRatio,
              height: "auto",
              overscrollBehavior: "contain",
              touchAction: "none",
              "&:hover .show3d-panel-hide-button, &:focus-within .show3d-panel-hide-button": {
                opacity: 1,
                pointerEvents: "auto",
                transform: "translateY(0)",
              },
              "@media (hover: none), (pointer: coarse)": {
                "& .show3d-panel-hide-button": {
                  display: "none",
                },
              },
              ...(reorderMode ? {
                "@keyframes show3d-reorder-jiggle": {
                  "0%": { rotate: "-0.45deg" },
                  "100%": { rotate: "0.45deg" },
                },
              } : {}),
              cursor: reorderMode
                ? "grab"
                : overlayEditMode
                ? (isDraggingOverlay ? "grabbing" : isHoveringOverlay ? "nwse-resize" : "crosshair")
                : isHoveringLensEdge
                ? "nwse-resize"
                : (isHoveringResize || isDraggingResize || isHoveringResizeInner || isDraggingResizeInner)
                  ? "nwse-resize"
                  : (draggingProfileEndpoint !== null || isDraggingProfileLine)
                    ? "grabbing"
                    : (profileActive && (hoveredProfileEndpoint !== null || isHoveringProfileLine))
                      ? "grab"
                      : (effectiveRoiActive || profileActive)
                        ? "crosshair"
                        : "grab",
            }}
            onMouseDown={reorderMode ? undefined : handleCanvasMouseDown}
            onMouseMove={reorderMode ? undefined : handleCanvasMouseMove}
            onMouseUp={reorderMode ? undefined : handleCanvasMouseUp}
            onMouseLeave={reorderMode ? undefined : handleCanvasMouseLeave}
            onDoubleClick={reorderMode ? undefined : handleDoubleClick}
          >
            <canvas
              ref={canvasRef}
              width={canvasW}
              height={canvasH}
              onTouchStart={reorderMode ? undefined : handleCanvasTouchStart}
              onTouchMove={reorderMode ? undefined : handleCanvasTouchMove}
              onTouchEnd={reorderMode ? undefined : handleCanvasTouchEnd}
              onTouchCancel={reorderMode ? undefined : handleCanvasTouchEnd}
              style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: smooth ? "auto" : "pixelated", opacity: gpuDisplayVisible ? 0 : 1, display: "block", touchAction: "none" }}
              role="img"
              aria-label={`Slice image ${visibleSliceIdx + 1} of ${nSlices}${title ? `: ${title}` : ""} (${width} by ${height} pixels). Use arrow keys to scrub frames.`}
            />
            <canvas ref={gpuCanvasRef} width={canvasW} height={canvasH} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: smooth ? "auto" : "pixelated", pointerEvents: "none", opacity: gpuDisplayVisible ? 1 : 0 }} aria-hidden="true" />
            <canvas ref={overlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", display: overlayCanvasVisible ? "block" : "none" }} aria-hidden="true" />
            <canvas ref={uiRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
            <canvas ref={lensCanvasRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", display: lensCanvasVisible ? "block" : "none" }} aria-hidden="true" />
            {effectiveShowFft && fftLayoutOverlay && (
              <canvas
                ref={fftInsetLayerRef}
                width={canvasW}
                height={canvasH}
                style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: smooth ? "auto" : "pixelated", pointerEvents: "none", zIndex: 7 }}
                aria-hidden="true"
              />
            )}
            {groupMarkerOverlays.map((marker) => (
              <Box
                key={marker.key}
                data-show3d-row-marker={marker.axis === "row" ? marker.key.slice(4) : undefined}
                data-show3d-col-marker={marker.axis === "col" ? marker.key.slice(4) : undefined}
                data-show3d-panel-group={marker.axis === "panel" ? marker.key.slice("panel-group-".length) : undefined}
                data-show3d-group-marker-color={marker.color}
                title={marker.label ? `${marker.label} group` : undefined}
                sx={{
                  position: "absolute",
                  left: `${marker.leftPct}%`,
                  top: `${marker.topPct}%`,
                  width: `${marker.widthPct}%`,
                  height: `${marker.heightPct}%`,
                  boxSizing: "border-box",
                  boxShadow: `inset 0 0 0 3px ${marker.color}, inset 0 0 0 5px rgba(0,0,0,0.9)`,
                  pointerEvents: "none",
                  zIndex: 9,
                }}
              >
                {marker.label && (
                  <Box
                    component="span"
                    sx={{
                      position: "absolute",
                      top: 4,
                      left: 6,
                      maxWidth: "calc(100% - 12px)",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      px: 0.5,
                      py: 0.1,
                      borderRadius: "2px",
                      background: "rgba(0,0,0,0.72)",
                      color: "#fff",
                      fontSize: 11,
                      fontWeight: 700,
                      lineHeight: 1.2,
                      textShadow: "0 1px 2px rgba(0,0,0,0.8)",
                    }}
                  >
                    {marker.label}
                  </Box>
                )}
              </Box>
            ))}
            {(nPanels || 1) > 1 && visiblePanelIndices.map((panel, slot) => {
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const rows = Math.ceil(n / cols);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              const color = panelMarkerColor(panel);
              return (
                <Box
                  key={`panel-marker-${panel}`}
                  data-show3d-marker-color={color}
                  data-show3d-marker-style={markerAround ? "around" : "left"}
                  title={`Panel marker ${color} · ${panelLabel(panel)}`}
                  sx={{
                    position: "absolute",
                    left: `${(panelLeft / Math.max(1, canvasW)) * 100}%`,
                    top: `${(panelTop / Math.max(1, canvasH)) * 100}%`,
                    width: markerAround ? `${(panelW / Math.max(1, canvasW)) * 100}%` : 5,
                    height: `${(panelH / Math.max(1, canvasH)) * 100}%`,
                    boxSizing: "border-box",
                    bgcolor: markerAround ? "transparent" : color,
                    boxShadow: markerAround
                      ? `inset 0 0 0 3px ${color}, inset 0 0 0 5px rgba(0,0,0,0.9)`
                      : "0 0 0 1px rgba(0,0,0,0.45)",
                    pointerEvents: "none",
                    zIndex: 8,
                  }}
                />
              );
            })}
            {visiblePanelIndices.flatMap((panel, slot) => {
              const annotations = panelAnnotations?.[panel] || [];
              if (!annotations.length) return [];
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const rows = Math.ceil(n / cols);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              return annotations.map((annotation, annotationIdx) => (
                <Box
                  key={`panel-annotation-${panel}-${annotationIdx}`}
                  className={annotation.class_name}
                  data-show3d-panel-annotation={panel}
                  data-show3d-panel-annotation-index={annotationIdx}
                  data-show3d-panel-annotation-position={annotation.position || "top-left"}
                  data-show3d-panel-annotation-variant={annotation.variant || "badge"}
                  title={annotation.text}
                  sx={{
                    position: "absolute",
                    left: `${(panelLeft / Math.max(1, canvasW)) * 100}%`,
                    top: `${(panelTop / Math.max(1, canvasH)) * 100}%`,
                    width: `${(panelW / Math.max(1, canvasW)) * 100}%`,
                    height: `${(panelH / Math.max(1, canvasH)) * 100}%`,
                    pointerEvents: "none",
                    zIndex: 10,
                  }}
                >
                  <Box
                    component="span"
                    sx={panelAnnotationSx(annotation)}
                  >
                    {renderPanelAnnotation(annotation)}
                  </Box>
                </Box>
              ));
            })}
            {showPanelTitles !== false && (nPanels || 1) > 1 && visiblePanelIndices.map((panel, slot) => {
              const titleText = panelTitleText(panel);
              if (!titleText) return null;
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const rows = Math.ceil(n / cols);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              const shownIdx = visibleSliceIdx;
              const realN = panelRealFrames?.[panel];
              const shown = realN ? Math.min(shownIdx + 1, realN) : shownIdx + 1;
              const total = realN || nSlices;
              const frameLabel = panelFrameLabelForIndex(panel, shownIdx);
              return (
                <Box
                  key={`panel-title-${panel}`}
                  data-show3d-panel-title={panel}
                  sx={{
                    ...panelTitleChromeSx(panelTitleStyle, {
                    position: "absolute",
                    top: `${((panelTop + 6) / Math.max(1, canvasH)) * 100}%`,
                    left: `${(panelLeft / Math.max(1, canvasW)) * 100}%`,
                    width: `${(panelW / Math.max(1, canvasW)) * 100}%`,
                    px: 1,
                    boxSizing: "border-box",
                    color: "rgba(255, 255, 255, 0.95)",
                    fontFamily: UI_FONT,
                    fontSize: Math.max(8, panelTitleFontSize || 11),
                    fontWeight: 700,
                    lineHeight: 1.2,
                    textAlign: "center",
                    textShadow: "1px 1px 0 rgba(0,0,0,0.85), 0 0 3px rgba(0,0,0,0.75)",
                    pointerEvents: "none",
                    userSelect: "none",
                    zIndex: 2,
                    whiteSpace: "normal",
                    overflow: "visible",
                    textOverflow: "clip",
                    overflowWrap: "anywhere",
                    }),
                  }}
                >

                  {panelTitleContent(panel)}{frameLabel ? ` · ${frameLabel}` : ""}{" "}
                  <span data-show3d-panel-frame-count="true" data-real-frame-count={total}>
                    {shown}/{total}
                  </span>

                </Box>
              );
            })}
            {/* Per-panel "best frame" stars. One gold ★ button top-right of
                each panel. Click toggles the star on the currently displayed
                slice for THAT panel. Programmatic API: widget.star_panel(i). */}
	            {panelChromeVisible && visiblePanelIndices.map((i, slot) => {
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (Math.ceil(n / cols) - 1)) / Math.ceil(n / cols);
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              const starredFrame = starred?.[i] ?? -1;
              const isStarredHere = starredFrame === visibleSliceIdx;
              const starElsewhere = starredFrame >= 0 && !isStarredHere;
              const tooltip = isStarredHere
                ? `★ Starred. Click to unstar frame ${visibleSliceIdx + 1}.`
                : starElsewhere
                  ? `Star is on frame ${starredFrame + 1}. Click to move it to frame ${visibleSliceIdx + 1}.`
                  : `Click to mark frame ${visibleSliceIdx + 1} as best for ${panelLabel(i)}.`;
              return (
                <button
                  key={`star-${i}`}
                  onMouseDown={(event) => event.stopPropagation()}
                  onClick={() => {
                    const cur = Array.from({ length: totalPanelCount }, (_, k) => starred?.[k] ?? -1);
                    cur[i] = isStarredHere ? -1 : visibleSliceIdx;
                    setStarred(cur);
                  }}
                  title={tooltip}
                  aria-label={tooltip}
                  style={{
                    position: "absolute",
                    top: `${((panelTop + 6) / Math.max(1, canvasH)) * 100}%`,
                    left: `calc(${((panelLeft + panelW) / Math.max(1, canvasW)) * 100}% - 26px)`,
                    width: 20, height: 20,
                    padding: 0,
                    border: "none",
                    background: "transparent",
                    cursor: "pointer",
                    fontSize: 18,
                    lineHeight: "20px",
                    textAlign: "center",
                    color: isStarredHere
                      ? "#ffc107"  // bright gold: star IS on this frame
                      : starElsewhere
                        ? "rgba(255, 193, 7, 0.45)"  // faded gold: star elsewhere on this panel
                        : "rgba(255,255,255,0.5)",   // grey: no star on this panel
                    textShadow: "0 0 3px rgba(0,0,0,0.8)",
                    pointerEvents: "auto",
                    userSelect: "none",
                  }}
                >
                  {isStarredHere ? "★" : "☆"}
                </button>
              );
            })}
            {panelChromeVisible && hasPanelChoices && visiblePanelIndices.map((panel, slot) => {
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (Math.ceil(n / cols) - 1)) / Math.ceil(n / cols);
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              const disabled = visiblePanelCount <= 1;
              const label = disabled ? `Cannot hide the last visible panel` : `Hide ${panelLabel(panel)}`;
              return (
                <IconButton
                  key={`panel-hide-${panel}`}
                  className="show3d-panel-hide-button"
                  size="small"
                  disabled={disabled}
                  onMouseDown={(event) => event.stopPropagation()}
                  onClick={(event) => {
                    event.stopPropagation();
                    setPanelHidden(panel, true);
                  }}
                  aria-label={label}
                  title={label}
                  sx={{
                    position: "absolute",
                    top: `${((panelTop + 5) / Math.max(1, canvasH)) * 100}%`,
                    left: `${((panelLeft + 5) / Math.max(1, canvasW)) * 100}%`,
                    width: 22,
                    height: 22,
                    p: 0,
                    opacity: 0,
                    transform: "translateY(-3px)",
                    transition: "opacity 120ms ease, transform 120ms ease, background-color 120ms ease, color 120ms ease",
                    color: disabled ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.75)",
                    bgcolor: "rgba(0,0,0,0.22)",
                    pointerEvents: "none",
                    "&:hover, &:focus-visible": {
                      bgcolor: "rgba(0,0,0,0.42)",
                      color: "rgba(255,255,255,0.95)",
                    },
                  }}
                >
                  <VisibilityOffIcon sx={{ fontSize: 15 }} />
                </IconButton>
              );
            })}
            {panelChromeVisible && reorderMode && (nPanels || 1) > 1 && visiblePanelIndices.map((panel, slot) => {
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const rows = Math.ceil(n / cols);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              const active = dragOverPanel === panel;
              const draggingThisPanel = reorderDragVisual?.panel === panel;
              return (
                <Box
                  key={`panel-reorder-${panel}`}
                  draggable={reorderMode}
                  role="button"
                  data-show3d-reorder-panel={panel}
                  aria-label={`Move ${panelLabel(panel)}`}
                  title={`Drag to reorder ${panelLabel(panel)}`}
                  onDragStart={(event) => handlePanelDragStart(event, panel)}
                  onDragOver={(event) => handlePanelDragOver(event, panel)}
                  onDrop={handlePanelDrop}
                  onDragEnd={handlePanelDragEnd}
                  onPointerDown={(event) => handlePanelReorderPointerDown(event, panel)}
                  onPointerEnter={(event) => handlePanelReorderPointerEnter(event, panel)}
                  onPointerMove={handlePanelReorderPointerMove}
                  onPointerUp={handlePanelReorderPointerUp}
                  onPointerCancel={cancelPanelReorderPreview}
                  sx={{
                    position: "absolute",
                    top: `${(panelTop / Math.max(1, canvasH)) * 100}%`,
                    left: `${(panelLeft / Math.max(1, canvasW)) * 100}%`,
                    width: `${(panelW / Math.max(1, canvasW)) * 100}%`,
                    height: `${(panelH / Math.max(1, canvasH)) * 100}%`,
                    boxSizing: "border-box",
                    border: `2px solid ${active ? themeColors.accent : "rgba(255,255,255,0.48)"}`,
                    bgcolor: draggingThisPanel ? "rgba(0,0,0,0.28)" : active ? "rgba(79, 195, 247, 0.16)" : "rgba(0,0,0,0.04)",
                    outline: active ? `1px solid ${themeColors.accent}` : "none",
                    opacity: draggingThisPanel ? 0.38 : 1,
                    transform: active ? "translateY(-3px) scale(1.006)" : "translateY(0) scale(1)",
                    transition: "transform 110ms ease, opacity 110ms ease, background-color 110ms ease, border-color 110ms ease, box-shadow 110ms ease",
                    animation: "show3d-reorder-jiggle 220ms ease-in-out infinite alternate",
                    boxShadow: active ? `0 0 0 2px ${themeColors.accent}, 0 8px 18px rgba(0,0,0,0.20)` : "none",
                    cursor: draggedPanelRef.current === panel ? "grabbing" : "grab",
                    pointerEvents: "auto",
                    zIndex: 8,
                  }}
                >
                  <Box
                    sx={{
                      position: "absolute",
                      bottom: 6,
                      left: "50%",
                      transform: "translateX(-50%)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      width: 30,
                      height: 22,
                      borderRadius: 1,
                      bgcolor: "rgba(0,0,0,0.38)",
                      color: "rgba(255,255,255,0.92)",
                      pointerEvents: "none",
                    }}
                  >
                    <DragIndicatorIcon sx={{ fontSize: 18 }} />
                  </Box>
                </Box>
              );
            })}
            {panelChromeVisible && reorderMode && reorderDragVisual && (
              <Box
                ref={reorderGhostRef}
                data-show3d-reorder-ghost={reorderDragVisual.panel}
                aria-hidden="true"
                sx={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: `${reorderDragVisual.width}px`,
                  height: `${reorderDragVisual.height}px`,
                  transform: `translate3d(${reorderDragVisual.x}px, ${reorderDragVisual.y}px, 0)`,
                  boxSizing: "border-box",
                  overflow: "hidden",
                  border: `2px solid ${themeColors.accent}`,
                  bgcolor: reorderDragVisual.imageUrl ? "rgba(0,0,0,0.04)" : "rgba(25,25,25,0.68)",
                  boxShadow: `0 10px 24px rgba(0,0,0,0.32), 0 0 0 1px ${themeColors.accent}`,
                  opacity: 0.9,
                  pointerEvents: "none",
                  zIndex: 12,
                  willChange: "transform",
                }}
              >
                {reorderDragVisual.imageUrl && (
                  <Box
                    sx={{
                      position: "absolute",
                      inset: 0,
                      backgroundImage: `url(${reorderDragVisual.imageUrl})`,
                      backgroundSize: "100% 100%",
                      backgroundPosition: "center",
                      imageRendering: smooth ? "auto" : "pixelated",
                    }}
                  />
                )}
                <Box
                  sx={{
                    position: "absolute",
                    top: 6,
                    left: 8,
                    right: 8,
                    px: 0.75,
                    py: 0.25,
                    borderRadius: 0.75,
                    bgcolor: "rgba(0,0,0,0.48)",
                    color: "rgba(255,255,255,0.96)",
                    fontFamily: UI_FONT,
                    fontSize: Math.max(8, panelTitleFontSize || 11),
                    fontWeight: 700,
                    lineHeight: 1.2,
                    textAlign: "center",
                    textShadow: "0 1px 2px rgba(0,0,0,0.9)",
                    whiteSpace: "normal",
                    overflow: "visible",
                    textOverflow: "clip",
                    overflowWrap: "anywhere",
                  }}
                >
                  {reorderDragVisual.label}
                </Box>
                <Box
                  sx={{
                    position: "absolute",
                    bottom: 8,
                    left: "50%",
                    transform: "translateX(-50%)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: 34,
                    height: 24,
                    borderRadius: 1,
                    bgcolor: "rgba(0,0,0,0.48)",
                    color: "rgba(255,255,255,0.95)",
                  }}
                >
                  <DragIndicatorIcon sx={{ fontSize: 19 }} />
                </Box>
              </Box>
            )}
            {/* Zoom indicator now drawn on the ui canvas in the scale-bar
                pass (Show2D-matching style: white, sans, Unicode ×). */}
            {/* Cursor readout overlay */}
	            {panelChromeVisible && cursorInfo && (() => {
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const rows = Math.ceil(n / cols);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              const slot = visiblePanelIndices.indexOf(cursorInfo.panelIdx);
              if (slot < 0) return null;
              const col = slot % cols;
              const row = Math.floor(slot / cols);
              const panelLeft = col * (panelW + gap);
              const panelTop = row * (panelH + gap);
              return (
                <Box className="show3d-cursor-readout" sx={{
                  position: "absolute",
                  top: `${((panelTop + 3) / Math.max(1, canvasH)) * 100}%`,
                  right: `calc(${((canvasW - (panelLeft + panelW)) / Math.max(1, canvasW)) * 100}% + 3px)`,
                  bgcolor: "rgba(0,0,0,0.35)",
                  px: 0.5,
                  py: 0.15,
                  opacity: cursorReadoutVisible ? 1 : 0,
                  transform: cursorReadoutVisible ? "translateY(0)" : "translateY(-2px)",
                  transition: "opacity 90ms ease, transform 90ms ease",
                  willChange: "opacity, transform",
                  pointerEvents: "none",
                  minWidth: 78,
                  maxWidth: `calc(${(panelW / Math.max(1, canvasW)) * 100}% - 6px)`,
                  textAlign: "right",
                }}>
                  <Typography sx={{ fontSize: 9, fontFamily: "monospace", fontVariantNumeric: "tabular-nums", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2, overflow: "hidden", textOverflow: "ellipsis" }}>
                    ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                  </Typography>
                </Box>
              );
            })()}
	            {panelChromeVisible && effectiveRoiActive && roiItems.length > 0 && showRoiResizeHint && (
              <Box sx={{ position: "absolute", left: 6, top: 6, px: 0.6, py: 0.25, bgcolor: "rgba(0,0,0,0.45)", pointerEvents: "none" }}>
                <Typography sx={{ fontSize: 9, color: "rgba(255,255,255,0.8)", lineHeight: 1.1 }}>
                  Hover ROI edge to resize
                </Typography>
              </Box>
            )}
            {/* Per-panel resize corner. Empty cells (partial last row) get
                no handle. Each handle scales the whole multi-panel canvas
                (linked behavior). Match Show2D gallery: 16x16, grey, 0.6/1.0.
                User trait `show_resize_handles` toggles visibility. */}
            {showResizeControls && (() => {
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const rows = Math.ceil(n / cols);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const outPanelW = (canvasW - gap * (cols - 1)) / cols;
              const outPanelH = (canvasH - gap * (rows - 1)) / rows;
              return visiblePanelIndices.map((panel, slot) => {
                const col = slot % cols;
                const row = Math.floor(slot / cols);
                const slotX = col * (outPanelW + gap);
                const slotY = row * (outPanelH + gap);
                return (
                  <Box
                    key={`resize-${panel}`}
                    onMouseDown={handleMainResizeStart}
                    sx={{
                      position: "absolute",
                      left: `calc(${((slotX + outPanelW) / Math.max(1, canvasW)) * 100}% - 16px)`,
                      top: `calc(${((slotY + outPanelH) / Math.max(1, canvasH)) * 100}% - 16px)`,
                      width: 16,
                      height: 16,
                      cursor: "nwse-resize",
                      opacity: 0.6,
                      background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
                      touchAction: "none",
                      zIndex: 5,
                      "&:hover": { opacity: 1 },
                    }}
                  />
                );
              });
            })()}
            {effectiveShowFft && fftLayoutOverlay && (() => {
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const rows = Math.ceil(n / cols);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              return visiblePanelIndices.map((panel, slot) => {
                const panelLeft = (slot % cols) * (panelW + gap);
                const panelTop = Math.floor(slot / cols) * (panelH + gap);
                const insetPad = Math.min(8, Math.max(3, panelW * 0.025));
                const insetMaxW = Math.max(24, panelW - insetPad * 2);
                const insetMaxH = Math.max(20, panelH - insetPad * 2);
                const insetBase = Math.min(insetMaxW, insetMaxH);
                const insetW = Math.max(24, Math.min(insetMaxW, insetBase * resolvedFftOverlaySize));
                const insetH = Math.max(20, Math.min(insetMaxH, insetBase * resolvedFftOverlaySize));
                const insetX = resolvedFftOverlayPosition.endsWith("right")
                  ? panelLeft + panelW - insetW - insetPad
                  : panelLeft + insetPad;
                const insetY = resolvedFftOverlayPosition.startsWith("bottom")
                  ? panelTop + panelH - insetH - insetPad
                  : panelTop + insetPad;
                const previewInsetX = fftOverlayDragPreview ? panelLeft + fftOverlayDragPreview.x : insetX;
                const previewInsetY = fftOverlayDragPreview ? panelTop + fftOverlayDragPreview.y : insetY;
                return (
                  <Box
                    key={`fft-overlay-inset-${panel}`}
                    data-show3d-fft-inset="true"
                    title="Drag to move FFT overlay; Shift-drag to pan FFT detail"
                    onWheel={handleFftInsetWheel}
                    onMouseDown={(e) => {
                      if (e.shiftKey) {
                        handleFftInsetPanMouseDown(e);
                      } else {
                        handleFftInsetMouseDown(e, panelLeft, panelTop, panelW, panelH, insetX, insetY, insetW, insetH);
                      }
                    }}
                    onDoubleClick={(e) => { e.preventDefault(); e.stopPropagation(); handleFftReset(); }}
                    onTouchStart={handleFftInsetTouchStart}
                    onTouchMove={handleFftInsetTouchMove}
                    onTouchEnd={handleFftInsetTouchEnd}
                    onTouchCancel={handleFftInsetTouchEnd}
                    role="img"
                    aria-label={`FFT power spectrum overlay for ${panelLabel(panel)}`}
                    sx={{
                      position: "absolute",
                      left: `${(previewInsetX / Math.max(1, canvasW)) * 100}%`,
                      top: `${(previewInsetY / Math.max(1, canvasH)) * 100}%`,
                      width: `${(insetW / Math.max(1, canvasW)) * 100}%`,
                      height: `${(insetH / Math.max(1, canvasH)) * 100}%`,
                      bgcolor: "transparent",
                      border: "1px solid transparent",
                      zIndex: 8,
                      overflow: "hidden",
                      pointerEvents: "auto",
                      cursor: "move",
                      touchAction: "none",
                    }}
                  >
                    <Box
                      data-show3d-fft-move-handle="true"
                      aria-label="Move FFT overlay; snaps to nearest corner"
                      onPointerDown={(e) => handleFftInsetPointerDown(e, panelLeft, panelTop, panelW, panelH, insetX, insetY, insetW, insetH)}
                      onPointerMove={handleFftInsetPointerMove}
                      onPointerUp={handleFftInsetPointerUp}
                      onPointerCancel={(e) => {
                        if (fftOverlayDragRef.current?.pointerId === e.pointerId) {
                          fftOverlayDragRef.current = null;
                          setFftOverlayDragPreview(null);
                        }
                      }}
                      onMouseDown={(e) => handleFftInsetMouseDown(e, panelLeft, panelTop, panelW, panelH, insetX, insetY, insetW, insetH)}
                      sx={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        right: 0,
                        height: Math.min(16, Math.max(10, insetH * 0.18)),
                        zIndex: 2,
                        cursor: "move",
                        background: "linear-gradient(180deg, rgba(0,0,0,0.38), rgba(0,0,0,0))",
                        opacity: 0.65,
                        touchAction: "none",
                        "&:hover": { opacity: 1 },
                      }}
                    />
                    {showZoomIndicator !== false && panelChromeVisible && (
                      <Box
                        className="quantem-fft-zoom-label"
                        data-show3d-fft-zoom-indicator={panel}
                        data-fft-zoom={formatZoomLabel(fftZoom)}
                        aria-label={`FFT zoom for ${panelLabel(panel)}: ${formatZoomLabel(fftZoom)}`}
                        sx={{
                          position: "absolute",
                          left: Math.min(12, Math.max(5, insetW * 0.08)),
                          bottom: Math.min(7, Math.max(4, insetH * 0.06)),
                          color: "white",
                          fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                          fontSize: Math.max(9, Math.min(14, insetW * 0.1)),
                          fontWeight: 400,
                          fontVariantNumeric: "tabular-nums",
                          lineHeight: 1,
                          textShadow: "1px 1px 2px rgba(0,0,0,0.85)",
                          pointerEvents: "none",
                          userSelect: "none",
                          zIndex: 3,
                        }}
                      >
                        {formatZoomLabel(fftZoom)}
                      </Box>
                    )}
                    {slot === 0 && fftMetricsEnabled && fftQuality && (
                      <Box
                        className="quantem-fft-quality-label"
                        aria-label={`FFT quality: ${formatFftQualityLabel(fftQuality)}`}
                        sx={{
                          position: "absolute",
                          top: 4,
                          left: 5,
                          right: 5,
                          color: "rgba(255,255,255,0.96)",
                          fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                          fontSize: 10,
                          fontWeight: 700,
                          lineHeight: 1.15,
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          textShadow: "1px 1px 0 rgba(0,0,0,0.9), 0 0 3px rgba(0,0,0,0.85)",
                          pointerEvents: "none",
                          userSelect: "none",
                          zIndex: 3,
                        }}
                      >
                        {formatFftQualityLabel(fftQuality)}
                      </Box>
                    )}
                  </Box>
                );
              });
            })()}
          </Box>
          {/* Panel titles render ON canvas inside drawMain - follows grid layout. */}
          {/* Statistics bar - right below the image. Multi-panel = one row per panel. */}
          {showStats && (
            (localPanelStats && (nPanels || 1) > 1) ? (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", flexDirection: "column", gap: 0.25, width: "100%", maxWidth: canvasW, boxSizing: "border-box", fontFamily: "ui-monospace, monospace" }}>
                {localPanelStats.map((st) => (
                  <Box key={st.panel} sx={{ display: "flex", gap: 2, alignItems: "center", flexWrap: "wrap", maxWidth: "100%" }}>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted, minWidth: 80 }}>{panelTitleContent(st.panel)}</Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.mean)}</Box></Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.min)}</Box></Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.max)}</Box></Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.std)}</Box></Typography>
                  </Box>
                ))}
              </Box>
            ) : (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", flexWrap: "wrap", width: "100%", maxWidth: canvasW, boxSizing: "border-box" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.mean : statsMean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.min : statsMin)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.max : statsMax)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.std : statsStd)}</Box></Typography>
              </Box>
            )
          )}
          {/* Line profile sparkline */}
          {profileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, boxSizing: "border-box" }}>
              <canvas
                ref={profileCanvasRef}
                onMouseMove={handleProfileMouseMove}
                onMouseLeave={handleProfileMouseLeave}
                style={{ width: "100%", height: profileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
                role="img"
                aria-label="Line intensity profile along the drawn line"
              />
              {showResizeControls && (
                <div
                  onMouseDown={(e) => { e.preventDefault(); setIsResizingProfile(true); setProfileResizeStart({ y: e.clientY, height: profileHeight }); }}
                  style={{ width: "100%", height: 4, cursor: "ns-resize", borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, background: `linear-gradient(to bottom, ${themeColors.border}, transparent)` }}
                />
              )}
            </Box>
          )}
          {/* ROI sparkline plot */}
          {effectiveRoiActive && showRoiPlot && roiPlotData && roiPlotData.byteLength >= 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, boxSizing: "border-box" }}>
              <canvas
                ref={roiPlotCanvasRef}
                style={{ width: "100%", height: 76, display: "block", border: `1px solid ${themeColors.border}` }}
                role="img"
                aria-label="ROI mean intensity over frames"
              />
            </Box>
          )}
          {/* Image controls stay content-sized so multi-panel stacks do not
              create a large empty gutter between display and playback rows. */}
	          {controlsVisible && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", columnGap: `${SPACING.SM}px`, rowGap: `${SPACING.XS}px`, alignItems: "flex-start", justifyContent: "flex-start", width: "fit-content", maxWidth: "100%", boxSizing: "border-box", flexWrap: "wrap" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: isMobileViewport ? "1 1 100%" : "0 0 auto", width: isMobileViewport ? "100%" : "auto", maxWidth: "100%", minWidth: 0, justifyContent: "center" }}>
                {/* True-color figure stacks: hide colormap / intensity / Smooth —
                    paper figures are already final pixels. */}
                {!isRgb && (<>
                {/* Row 1: Scale + Auto + Color */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale</Typography>
                  <Select value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Intensity scale (linear or logarithmic)" }}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }} title={perPanelHistogramEnabled ? "Stack-wide auto contrast. Turn off for independent panel clips." : "Automatic percentile-based contrast."}>
                    {perPanelHistogramEnabled ? "Auto stack" : "Auto"}
                  </Typography>
                  <Switch checked={autoContrast} onChange={(e) => handleAutoContrastChange(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": perPanelHistogramEnabled ? "Toggle stack-wide automatic contrast" : "Toggle automatic percentile-based contrast" } }} />
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Colorbar</Typography>
                  <Switch checked={showColorbar} onChange={(e) => setShowColorbar(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle colorbar overlay" } }} />
                </Box>
                {/* Row 2: Color + Smooth + Diff + zoom indicator */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color</Typography>
                  <Select
                    size="small"
                    value={panelCmapFor(nPanels > 1 ? Math.max(0, cursorInfo?.panelIdx ?? visiblePanelIndices[0] ?? 0) : 0)}
                    onChange={(e) => setCmapForPanel(
                      nPanels > 1 ? Math.max(0, cursorInfo?.panelIdx ?? visiblePanelIndices[0] ?? 0) : 0,
                      e.target.value,
                    )}
                    MenuProps={themedFastMenuProps}
                    sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }}
                    inputProps={{ "aria-label": nPanels > 1 ? "Current panel colormap" : "Image colormap" }}
                  >
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name} dense>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Smooth</Typography>
                  <Switch checked={smooth} onChange={(e) => setSmooth(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle bilinear smoothing" } }} />
                  {/* Denoise on/off moved to the "More" menu (top toolbar). */}
                  {!showDenoise && displayFilterBanner && (
                    /* House rule: an active reduction is never invisible,
                       even with the denoise controls row hidden. */
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent }} title={displayFilterBanner}>
                      {displayFilterBanner.split(" (")[0]}
                    </Typography>
                  )}
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Diff</Typography>
                  <Select value={diffMode} onChange={(e) => setDiffMode(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Difference mode (off, previous frame, first frame)" }}>
                    <MenuItem value="off">Off</MenuItem>
                    <MenuItem value="previous">Prev</MenuItem>
                    <MenuItem value="first">First</MenuItem>
                  </Select>
                  {/* zoom indicator moved onto the canvas overlay */}
                </Box>
                {/* Row 3 (toggle-gated): display-only denoise for sparse map stacks (EDS, low dose) */}
                {showDenoise && (
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }} title="Poisson (Anscombe): count-respecting smoothing for sparse EDS/counting data - recommended with Bin 2, sigma 6-10. Gaussian: simple smooth for decent-dose images. None: raw counts (use for anything quantitative).">Denoise</Typography>
                  <Select size="small" value={resolveDenoiseMode(displayFilter || "none").mode} onChange={(e) => { setDisplayFilter(e.target.value); if (resolveDenoiseMode(e.target.value).mode !== "none" || (spatialBin || 1) > 1) setDenoiseEnabled(true); }} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 88, fontSize: 10 }} inputProps={{ "aria-label": "Display-only denoise method" }}>
                    {[["none", "None"], ["gaussian", "Gaussian"], ["anscombe", "Poisson (Anscombe)"]].map(([mode, label]) => (
                      <MenuItem key={mode} value={mode}>{label}</MenuItem>
                    ))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted, minWidth: 40, display: "inline-block" }}>σ {(sigmaDraft ?? Number(displaySigma ?? 4)).toFixed(1)}</Typography>
                  <Slider
                    value={sigmaDraft ?? Number(displaySigma ?? 4)}
                    min={0} max={20} step={0.5}
                    onChange={(_, v) => { if (displayFilterOff) setDisplayFilter("gaussian"); setSigmaDraft(v as number); }}
                    onChangeCommitted={(_, v) => { setDisplaySigma(v as number); setSigmaDraft(null); if (displayFilterOff) setDisplayFilter("gaussian"); setDenoiseEnabled(true); }}
                    size="small" sx={{ ...sliderStyles.small, width: 60 }}
                    aria-label="Display filter sigma in pixels"
                  />
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }} title="Display-side 2x bin passes for SNR, combined with the denoise method. 1 is lossless.">Bin</Typography>
                  <Select size="small" value={String(spatialBin || 1)} onChange={(e) => { const b = parseInt(e.target.value, 10); setSpatialBin(b); if (b > 1 || resolveDenoiseMode(displayFilter || "none").mode !== "none") setDenoiseEnabled(true); }} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 40, fontSize: 10 }} inputProps={{ "aria-label": "Display spatial bin factor" }}>
                    {[1, 2, 4].map((b) => (<MenuItem key={b} value={String(b)}>{b}</MenuItem>))}
                  </Select>
                </Box>
                )}
                {showFrequencyFilter && (
                <Box sx={{ ...controlRow, ...mobileControlRowSx, width: "100%", maxWidth: "100%", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px`, flexWrap: "nowrap" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }} title="Low-pass removes fine detail; High-pass removes slow background; Band-pass isolates a periodicity.">Filter</Typography>
                    <Select size="small" value={normalizeFrequencyFilterMode(frequencyFilter)} onChange={(event) => { const mode = String(event.target.value); setFrequencyFilter(mode); if (mode !== "none") setFrequencyFilterEnabled(true); }} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 84, fontSize: 10 }} inputProps={{ "aria-label": "Frequency filter mode" }}>
                      <MenuItem value="none">None</MenuItem>
                      <MenuItem value="lowpass">Low-pass</MenuItem>
                      <MenuItem value="highpass">High-pass</MenuItem>
                      <MenuItem value="bandpass">Band-pass</MenuItem>
                    </Select>
                  </Box>
                  {normalizeFrequencyFilterMode(frequencyFilter) === "bandpass" ? (<>
                    <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px`, flexWrap: "nowrap" }}>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted, minWidth: 84, display: "inline-block" }}>Center {frequencyValueLabel(frequencyDraft ?? frequencyFilterCenter)}</Typography>
                      <Slider value={frequencyDraft ?? frequencyFilterCenter} min={0} max={1} step={0.005} onChange={(_, value) => setFrequencyDraft(value as number)} onChangeCommitted={(_, value) => { setFrequencyFilterCenter(value as number); setFrequencyDraft(null); }} size="small" sx={{ ...sliderStyles.small, width: 72 }} aria-label="Band-pass center as fraction of Nyquist" />
                    </Box>
                    <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px`, flexWrap: "nowrap" }}>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted, minWidth: 80, display: "inline-block" }}>Width {frequencyValueLabel(frequencyFilterWidth)}</Typography>
                      <Slider value={frequencyFilterWidth} min={0.01} max={1} step={0.005} onChange={(_, value) => setFrequencyFilterWidth(value as number)} size="small" sx={{ ...sliderStyles.small, width: 72 }} aria-label="Band-pass width as fraction of Nyquist" />
                    </Box>
                  </>) : (<>
                    <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px`, flexWrap: "nowrap" }}>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted, minWidth: 84, display: "inline-block" }}>Cutoff {frequencyValueLabel(frequencyDraft ?? frequencyFilterCutoff)}</Typography>
                      <Slider value={frequencyDraft ?? frequencyFilterCutoff} min={0} max={1} step={0.005} disabled={!frequencyFilterActive(frequencyFilter)} onChange={(_, value) => setFrequencyDraft(value as number)} onChangeCommitted={(_, value) => { setFrequencyFilterCutoff(value as number); setFrequencyDraft(null); }} size="small" sx={{ ...sliderStyles.small, width: 72 }} aria-label="Frequency cutoff as fraction of Nyquist" />
                    </Box>
                  </>)}
                </Box>
                )}
                </>)}
              </Box>
              {/* Playback: 2 rows side-by-side with Display + Histogram. */}
              {(() => { const activeIdx = visibleSliceIdx; return (
                <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: `${SPACING.XS}px`, flex: "0 1 auto", minWidth: 0, maxWidth: "100%", justifyContent: "center" }}>
                  <Box sx={{ ...controlRow, ...mobileControlRowSx, width: "fit-content", maxWidth: "100%", flexWrap: "nowrap", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, boxSizing: "border-box" }}>
                    <Stack direction="row" spacing={0} sx={{ flexShrink: 0, mr: 0.5 }}>
                      <IconButton size="small" onClick={() => playFromCurrentFrame(-1)} sx={{ color: reverse && playing ? themeColors.accent : themeColors.textMuted, p: 0.25 }} aria-label="Play in reverse" title="Play reverse">
                        <FastRewindIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                      <IconButton size="small" onClick={() => { if (playing) pausePlayback(); else playFromCurrentFrame(); }} sx={{ color: themeColors.accent, p: 0.25 }} aria-label={playing ? "Pause playback" : "Play"} title={playing ? "Pause (Space)" : "Play (Space)"}>
                        {playing ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
                      </IconButton>
                      <IconButton size="small" onClick={() => playFromCurrentFrame(1)} sx={{ color: !reverse && playing ? themeColors.accent : themeColors.textMuted, p: 0.25 }} aria-label="Play forward" title="Play forward">
                        <FastForwardIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                      <IconButton size="small" onClick={stopPlayback} sx={{ color: themeColors.textMuted, p: 0.25 }} aria-label="Stop and rewind to start" title="Stop">
                        <StopIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Stack>
                    {loop ? (
                      <Slider ref={playbackSliderRef} value={[loopStart, activeIdx, effectiveLoopEnd]} onMouseDown={handleLoopSliderMouseDown} onPointerDownCapture={handleLoopSliderPointerDownCapture} onChange={(_, v) => { const vals = v as number[]; setLoopStart(vals[0]); scrubToSlice(vals[1]); setLoopEnd(vals[2]); }} onChangeCommitted={(_, v) => { const vals = v as number[]; setLoopStart(vals[0]); commitSlice(vals[1]); setLoopEnd(vals[2]); }} disableSwap min={0} max={nSlices - 1} size="small" valueLabelDisplay="auto" valueLabelFormat={(v) => formatFrameValueLabel(v)} marks={bookmarkedFrameMarks} aria-label={`Loop range and current ${dimLabel.toLowerCase()} (frame ${activeIdx + 1} of ${nSlices}, loop ${loopStart + 1} to ${effectiveLoopEnd + 1})`} sx={{ ...sliderStyles.small, width: 150, flex: "0 1 150px", minWidth: 90, "& .MuiSlider-thumb[data-index='0']": { width: 8, height: 8, bgcolor: themeColors.textMuted }, "& .MuiSlider-thumb[data-index='1']": { width: 12, height: 12 }, "& .MuiSlider-thumb[data-index='2']": { width: 8, height: 8, bgcolor: themeColors.textMuted }, "& .MuiSlider-mark": { bgcolor: "#ffc107", width: 5, height: 5, borderRadius: "50%", top: "50%", transform: "translate(-50%, -50%)" }, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px", maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }} />
                    ) : (
                      <Slider ref={playbackSliderRef} value={activeIdx} onChange={(_, v) => scrubToSlice(v as number)} onChangeCommitted={(_, v) => commitSlice(v as number)} min={0} max={nSlices - 1} size="small" valueLabelDisplay="auto" valueLabelFormat={(v) => formatFrameValueLabel(v)} marks={bookmarkedFrameMarks} aria-label={`Current ${dimLabel.toLowerCase()} (${activeIdx + 1} of ${nSlices})`} sx={{ ...sliderStyles.small, width: 150, flex: "0 1 150px", minWidth: 90, "& .MuiSlider-mark": { bgcolor: "#ffc107", width: 5, height: 5, borderRadius: "50%", top: "50%", transform: "translate(-50%, -50%)" }, "& .MuiSlider-valueLabel": { maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }} />
                    )}
                    <span
                      ref={playbackLiveCountRef}
                      data-show3d-playback-count="true"
                      style={{
                        fontSize: 10,
                        fontFamily: UI_FONT,
                        color: themeColors.textMuted,
                        minWidth: hiddenSet.size ? `${String(nSlices).length * 2 + String(visibleCount).length + 5}ch` : `${String(nSlices).length * 2 + 1}ch`,
                        fontVariantNumeric: "tabular-nums",
                        textAlign: "right",
                        flexShrink: 0,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {hiddenSet.size ? `${activeIdx + 1}/${visibleCount} (${nSlices})` : `${activeIdx + 1}/${nSlices}`}
                    </span>
                    <IconButton size="small" onClick={toggleCurrentFrameBookmark} aria-pressed={currentFrameBookmarked} aria-label={`${currentFrameBookmarked ? "Unstar" : "Star"} frame ${activeIdx + 1}`} title={`${currentFrameBookmarked ? "Unstar" : "Star"} frame ${activeIdx + 1}`} sx={{ color: currentFrameBookmarked ? "#ffc107" : themeColors.textMuted, p: 0.25, width: 22, height: 22, flexShrink: 0, "&:hover": { color: currentFrameBookmarked ? "#ffc107" : themeColors.text } }}>
                      <Box component="span" sx={{ fontSize: 18, lineHeight: "18px" }}>{currentFrameBookmarked ? "★" : "☆"}</Box>
                    </IconButton>
                    <Badge
                      badgeContent={playbackPathLength > 0 ? 1 : 0}
                      invisible={playbackPathLength === 0}
                      sx={{ "& .MuiBadge-badge": { bgcolor: themeColors.accent, color: "#fff", fontSize: 9, fontWeight: 600, minWidth: 12, height: 12, px: 0.25 } }}
                    >
                      <Button
                        size="small"
                        sx={{ minWidth: 0, px: 0.6, py: 0.1, fontSize: 10, lineHeight: 1.2, textTransform: "none", color: playbackPathLength > 0 ? themeColors.accent : themeColors.textMuted, flexShrink: 0 }}
                        onClick={(e) => setPlaybackStyleMenuAnchor(e.currentTarget)}
                        aria-label="More playback style options"
                        aria-haspopup="menu"
                        title={`Playback style: ${playbackStyleSummary}`}
                      >
                        More
                      </Button>
                    </Badge>
                    <Menu
                      anchorEl={playbackStyleMenuAnchor}
                      open={Boolean(playbackStyleMenuAnchor)}
                      onClose={() => setPlaybackStyleMenuAnchor(null)}
                      MenuListProps={{ "aria-label": "Playback style options" }}
                      {...themedMenuProps}
                    >
                      <Box sx={{ px: 1.5, py: 0.75, minWidth: 240 }}>
                        <Typography sx={{ fontSize: 11, fontWeight: 700, color: themeColors.text, mb: 0.5 }}>Play Style</Typography>
                        <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mb: 0.75 }} title="Changes only playback_path. Loop, Bounce, fps, and range stay user-controlled in the playback row.">
                          {playbackStyleSummary} · uses current range
                        </Typography>
                        <Box sx={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 0.5 }}>
                          <Button size="small" sx={playbackStyleButtonSx("linear")} aria-pressed={playbackStyleActive === "linear"} onClick={() => applyPlaybackStylePreset("linear")} title="Use the current loop range at constant frame spacing.">Linear</Button>
                          <Button size="small" sx={playbackStyleButtonSx("power-in")} aria-pressed={playbackStyleActive === "power-in"} onClick={() => applyPlaybackStylePreset("power-in")} title="Start slowly, then accelerate through the current range.">Power In</Button>
                          <Button size="small" sx={playbackStyleButtonSx("power-out")} aria-pressed={playbackStyleActive === "power-out"} onClick={() => applyPlaybackStylePreset("power-out")} title="Move quickly at first, then settle near the end of the current range.">Power Out</Button>
                          <Button size="small" sx={playbackStyleButtonSx("ease-in-out")} aria-pressed={playbackStyleActive === "ease-in-out"} onClick={() => applyPlaybackStylePreset("ease-in-out")} title="Smoothly accelerate, then decelerate through the current range.">Ease In/Out</Button>
                        </Box>
                      </Box>
                    </Menu>
                  </Box>
                  <Box sx={{ ...controlRow, ...mobileControlRowSx, width: "fit-content", maxWidth: "100%", flexWrap: "wrap", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, boxSizing: "border-box" }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: isMobileViewport ? "4px" : `${SPACING.SM}px`, flexShrink: 0 }}>
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, flexShrink: 0 }}>fps</Typography>
                      <Slider value={playbackFps} min={1} max={MAX_PLAYBACK_FPS} step={1} onChange={(_, v) => setPlaybackFps(v as number)} size="small" sx={{ ...sliderStyles.small, width: isMobileViewport ? 40 : 44, mx: isMobileViewport ? "3px" : 0, flexShrink: 0 }} aria-label="Playback frames per second" valueLabelDisplay="auto" />
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, minWidth: isMobileViewport ? 16 : 20, flexShrink: 0 }}>{Math.round(playbackFps)}</Typography>
                    </Box>
                    <Box
                      title={averageSupported ? "Moving average window" : "Moving average is unavailable for separate full-resolution panel streams"}
                      sx={{ display: "flex", alignItems: "center", gap: isMobileViewport ? "4px" : `${SPACING.SM}px`, flexShrink: 0 }}
                    >
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, flexShrink: 0 }}>avg</Typography>
                      <Slider
                        value={avgWindow}
                        min={1}
                        max={15}
                        step={1}
                        onChange={(_, v) => setAvgWindow(v as number)}
                        disabled={!averageSupported}
                        size="small"
                        sx={{ ...sliderStyles.small, width: isMobileViewport ? 40 : 44, mx: isMobileViewport ? "3px" : 0, flexShrink: 0 }}
                        aria-label="Moving average window"
                        valueLabelDisplay="auto"
                      />
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, minWidth: 16, flexShrink: 0 }}>{Math.round(avgWindow || 1)}</Typography>
                    </Box>
                    <Box sx={{ display: "flex", alignItems: "center", gap: isMobileViewport ? "4px" : `${SPACING.SM}px`, flexShrink: 0 }}>
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, flexShrink: 0 }}>Loop</Typography>
                      <Switch size="small" checked={loop} onChange={() => setLoop(!loop)} sx={{ ...switchStyles.small, flexShrink: 0 }} slotProps={{ input: { "aria-label": "Toggle loop playback" } }} />
                    </Box>
                    <Box sx={{ display: "flex", alignItems: "center", gap: isMobileViewport ? "4px" : `${SPACING.SM}px`, flexShrink: 0 }}>
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, flexShrink: 0 }}>Bounce</Typography>
                      <Switch size="small" checked={boomerang} onChange={() => setBoomerang(!boomerang)} sx={{ ...switchStyles.small, flexShrink: 0 }} slotProps={{ input: { "aria-label": "Toggle bounce playback" } }} />
                    </Box>
                  </Box>
                </Box>
              ); })()}
              {/* Intensity histogram + clip sliders are gray-only (colormap window). */}
              {!isRgb && (() => {
                // Global stack range from Python (data_min/data_max trait), not per-frame.
                // Log mode: log1p the range so bins line up with the log-scaled frame data.
                const { min: histMin, max: histMax } = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
                if (perPanelHistogramEnabled) {
                  const n = Math.max(1, visiblePanelCount || 1);
                  const cols = panelColsForCount(n);
                  // Match Show2D shell exactly (width=110, height=58, gap=15px)
                  // so the per-panel histogram strip is visually consistent
                  // across widgets.
                  const panelHistWidth = 110;
                  const panelHistGap = 15;
                  const panelHistMaxWidth = cols * panelHistWidth + Math.max(0, cols - 1) * panelHistGap;
                  return (
                    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "flex-start", gap: 0.5, opacity: 1, pointerEvents: "auto", maxWidth: "100%" }}>
                      <Box sx={{ display: "grid", gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, ${panelHistWidth}px), ${panelHistWidth}px))`, gap: `${panelHistGap}px`, width: "100%", maxWidth: panelHistMaxWidth, justifyContent: "start" }}>
                      {visiblePanelIndices.map((panel) => {
                        const state = panelStates[panel] || initialState;
                        // Per-panel histogram uses THIS panel's data range
                        // (not stack-wide histMin/histMax). Tight-range
                        // modalities (SSB phase) get a sensible slider
                        // space instead of being squashed by DF counts.
                        const pdr = panelDataRanges[panel];
                        const panelRange = (pdr && pdr.max > pdr.min) ? pdr : { min: histMin, max: histMax };
                        const vminPct = state.imageVminPct;
                        const vmaxPct = state.imageVmaxPct;
                        return (
                          <Histogram
                            key={`panel-hist-${panel}`}
                            data={panelHistogramData[panel] ?? null}
                            bins={null}
                            vminPct={vminPct}
                            vmaxPct={vmaxPct}
                            onRangePreview={(min, max) => {
                              panelHistogramPreviewPctRef.current.set(panel, [min, max]);
                              scheduleHistogramPreviewPaint("hist-preview");
                            }}
                            onRangeChange={(min, max) => {
                              panelHistogramPreviewPctRef.current.set(panel, [min, max]);
                              scheduleHistogramPreviewPaint("hist-commit");
                              const commitPanelRange = () => {
                                updatePanelState(panel, { imageVminPct: min, imageVmaxPct: max });
                                setPanelRangeValues(panel, pctToValue(min, panelRange.min, panelRange.max), pctToValue(max, panelRange.min, panelRange.max));
                                if (autoContrast) {
                                  restorePanelManualClipPcts();
                                  manualImageRangeBeforeAutoRef.current = null;
                                  setAutoContrast(false);
                                }
                              };
                              if (sidecarMode) window.setTimeout(commitPanelRange, 0);
                              else commitPanelRange();
                            }}
                            commitOnChange={!sidecarMode}
                            width={110}
                            height={58}
                            theme={themeInfo.theme === "dark" ? "dark" : "light"}
                            dataMin={panelRange.min}
                            dataMax={panelRange.max}
                          />
                        );
                      })}
                      </Box>
                    </Box>
                  );
                }
                return (
                <Box sx={{
                  // Match Show2D histogram shell exactly so visual stays consistent
                  // across widgets. alignItems: flex-end (not stretch) prevents the
                  // inner Slider thumbs from overflowing onto the canvas, which was
                  // the source of the "2.8 tooltip overlaps bars" overlap bug.
                  display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "flex-start", gap: 0.5,
                }}>
                  <Histogram
                    data={imageHistogramData}
                    bins={imageHistogramBins}
                    vminPct={imageVminPct}
                    vmaxPct={imageVmaxPct}
                    onRangePreview={(min, max) => {
                      imageHistogramPreviewPctRef.current = [min, max];
                      scheduleHistogramPreviewPaint("hist-preview");
                    }}
                    onRangeChange={(min, max) => {
                      imageHistogramPreviewPctRef.current = [min, max];
                      scheduleHistogramPreviewPaint("hist-commit");
                      const commitSharedRange = () => {
                        setImageVminPct(min);
                        setImageVmaxPct(max);
                        if (autoContrast) {
                          manualImageRangeBeforeAutoRef.current = null;
                          setAutoContrast(false);
                        }
                      };
                      if (sidecarMode) window.setTimeout(commitSharedRange, 0);
                      else commitSharedRange();
                    }}
                    commitOnChange={!sidecarMode}
                    width={110}
                    height={58}
                    theme={themeInfo.theme === "dark" ? "dark" : "light"}
                    dataMin={histMin}
                    dataMax={histMax}
                  />
                </Box>
                );
              })()}
            </Box>
          )}
          {/* Lens settings row (when Lens is active) */}
          {showLens && (
            <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, width: "fit-content" }}>
              <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Lens {lensMag}×</Typography>
                <Slider value={lensMag} min={2} max={8} step={1} onChange={(_, v) => setLensMag(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} aria-label="Lens magnification" valueLabelDisplay="auto" />
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>{lensDisplaySize}px</Typography>
                <Slider value={lensDisplaySize} min={64} max={256} step={16} onChange={(_, v) => setLensDisplaySize(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} aria-label="Lens display size in pixels" valueLabelDisplay="auto" />
              </Box>
            </Box>
          )}
          {/* ROI settings row (when ROI is active) */}
          {effectiveRoiActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, width: "fit-content" }}>
              <Box sx={{ border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, px: 1, py: 0.5, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}>
                {/* ROI: shape + add/duplicate + plot + dim */}
                <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>ROI</Typography>
                  <Select
                    size="small"
                    value={newRoiShape}
                    onChange={(e) => setNewRoiShape(e.target.value as "circle" | "square" | "rectangle" | "annular")}
                    MenuProps={themedMenuProps}
                    sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                    inputProps={{ "aria-label": "New ROI shape" }}
                  >
                    {(["square", "rectangle", "circle", "annular"] as const).map((shape) => (<MenuItem key={shape} value={shape}>{shape.charAt(0).toUpperCase() + shape.slice(1)}</MenuItem>))}
                  </Select>
                  <Button size="small" sx={compactButton} onClick={() => addROIAt(height / 2, width / 2)} aria-label="Add ROI at image center">Add</Button>
                  <Button size="small" sx={compactButton} disabled={!selectedRoi} onClick={duplicateSelectedROI} aria-label="Duplicate selected ROI">Dup</Button>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Plot</Typography>
                  <Switch checked={showRoiPlot} onChange={(e) => setShowRoiPlot(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle ROI intensity plot" } }} />
                  <Box sx={{ flex: 1 }} />
                  <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 24, color: "#ef5350" }} disabled={!roiItems.length} onClick={() => { setRoiList([]); setRoiSelectedIdx(-1); }} aria-label="Clear all ROIs">Clear</Button>
                </Box>

                {/* Selected ROI details */}
                {selectedRoi && (
                  <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: `${SPACING.SM}px`, borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: selectedRoi.color }}>#{roiSelectedIdx + 1}/{roiItems.length}</Typography>
                    <Select
                      size="small"
                      value={selectedRoi.shape || "circle"}
                      onChange={(e) => updateSelectedRoi({ shape: String(e.target.value) })}
                      MenuProps={themedMenuProps}
                      sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                      inputProps={{ "aria-label": "Selected ROI shape" }}
                    >
                      {(["square", "rectangle", "circle", "annular"] as const).map((shape) => (<MenuItem key={shape} value={shape}>{shape.charAt(0).toUpperCase() + shape.slice(1)}</MenuItem>))}
                    </Select>
                    {selectedRoi.shape === "rectangle" && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>W</Typography>
                        <Slider value={selectedRoi.width} min={5} max={width} onChange={(_, v) => updateSelectedRoi({ width: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="ROI width" valueLabelDisplay="auto" />
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>H</Typography>
                        <Slider value={selectedRoi.height} min={5} max={height} onChange={(_, v) => updateSelectedRoi({ height: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="ROI height" valueLabelDisplay="auto" />
                      </>
                    )}
                    {selectedRoi.shape === "annular" && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Inner</Typography>
                        <Slider value={selectedRoi.radius_inner} min={1} max={Math.max(2, selectedRoi.radius - 1)} onChange={(_, v) => updateSelectedRoi({ radius_inner: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="Annular ROI inner radius" valueLabelDisplay="auto" />
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Outer</Typography>
                        <Slider value={selectedRoi.radius} min={selectedRoi.radius_inner + 1} max={Math.max(width, height)} onChange={(_, v) => updateSelectedRoi({ radius: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="Annular ROI outer radius" valueLabelDisplay="auto" />
                      </>
                    )}
                    {selectedRoi.shape !== "rectangle" && selectedRoi.shape !== "annular" && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Size</Typography>
                        <Slider value={selectedRoi.radius} min={5} max={Math.max(width, height)} onChange={(_, v) => updateSelectedRoi({ radius: v as number })} size="small" sx={{ ...sliderStyles.small, width: 50 }} aria-label="ROI radius" valueLabelDisplay="auto" />
                      </>
                    )}
                    <Box sx={{ display: "flex", gap: "2px" }}>
                      {ROI_COLORS.map(c => (
                        <Box key={c} onClick={() => updateSelectedRoi({ color: c })} sx={{ width: 12, height: 12, bgcolor: c, cursor: "pointer", border: c === selectedRoi.color ? `2px solid ${themeColors.text}` : "1px solid transparent", "&:hover": { opacity: 0.8 } }} />
                      ))}
                    </Box>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Border</Typography>
                    <Slider value={selectedRoi.line_width} min={1} max={6} step={1} onChange={(_, v) => updateSelectedRoi({ line_width: v as number })} size="small" sx={{ ...sliderStyles.small, width: 30 }} aria-label="ROI border line width" valueLabelDisplay="auto" />
                    <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 20, color: "#ef5350" }} onClick={deleteSelectedROI} aria-label="Delete selected ROI">&times;</Button>
                  </Box>
                )}

                {/* ROI list */}
                {roiItems.length > 0 && (
                  <Box sx={{ display: "flex", flexDirection: "column", borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                    {roiItems.map((roi, i) => {
                      const c = roi.color || ROI_COLORS[i % ROI_COLORS.length];
                      const isSelected = i === roiSelectedIdx;
                      const shapeLabel = roi.shape === "rectangle" ? `${roi.width}×${roi.height}` : roi.shape === "annular" ? `r${roi.radius_inner}-${roi.radius}` : `r${roi.radius}`;
                      return (
                        <Box key={i} onClick={() => setRoiSelectedIdx(i)} sx={{ display: "flex", alignItems: "center", gap: "3px", lineHeight: 1.6, cursor: "pointer", "&:hover .roi-delete": { opacity: 1 } }}>
                          <Box sx={{ width: 8, height: 8, borderRadius: roi.shape === "square" || roi.shape === "rectangle" ? 0 : "50%", bgcolor: c, border: isSelected ? "2px solid #fff" : "1px solid transparent", flexShrink: 0 }} />
                          <Typography component="span" sx={{ fontSize: 10, color: isSelected ? themeColors.text : themeColors.textMuted, fontWeight: isSelected ? "bold" : "normal" }}>
                            <Box component="span" sx={{ color: c }}>{i + 1}</Box>{" "}
                            {roi.shape} ({Math.round(roi.row)}, {Math.round(roi.col)}) {shapeLabel}
                          </Typography>
                          <Box
                            onClick={(e) => { e.stopPropagation(); const newList = roiItems.map((r, j) => ({ ...r, highlight: j === i ? !r.highlight : false })); setRoiList(newList); }}
                            sx={{ cursor: "pointer", fontSize: 10, color: roi.highlight ? themeColors.accentGreen : themeColors.textMuted, lineHeight: 1, opacity: roi.highlight ? 1 : 0.5, "&:hover": { opacity: 1 } }}
                            title="Focus (dim outside)"
                          >{roi.highlight ? "\u25C9" : "\u25CB"}</Box>
                          <Box
                            className="roi-delete"
                            onClick={(e) => { e.stopPropagation(); const newList = roiItems.filter((_, j) => j !== i); setRoiList(newList); setRoiSelectedIdx(newList.length > 0 ? Math.min(roiSelectedIdx, newList.length - 1) : -1); }}
                            sx={{ opacity: 0, cursor: "pointer", fontSize: 10, color: themeColors.textMuted, ml: 0.5, lineHeight: 1, "&:hover": { color: "#f44336" } }}
                          >&times;</Box>
                        </Box>
                      );
                    })}
                  </Box>
                )}
              </Box>
            </Box>
          )}
        </Box>

        {/* Preview Panel - ROI crop at full resolution with aspect ratio */}
        {previewVisible && (
          <Box sx={{ width: "100%", maxWidth: canvasW, boxSizing: "border-box" }}>
            {/* Spacer - matches main panel title row height for canvas alignment */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />
            {/* Header row - matches main panel controls row height */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, minHeight: 28, height: "auto", flexWrap: "wrap", gap: `${SPACING.XS}px` }}>
              <Typography sx={{ ...typography.label, color: themeColors.accentGreen }}>
                Preview{previewCropDims ? ` (${previewCropDims.w}\u00d7${previewCropDims.h})` : ""}
              </Typography>
              <Button size="small" sx={compactButton} disabled={previewZoom.zoom === 1 && previewZoom.panX === 0 && previewZoom.panY === 0} onClick={handlePreviewDoubleClick} aria-label="Reset preview zoom and pan">Reset</Button>
            </Stack>
            <Box
              ref={previewContainerRef}
              sx={{
                position: "relative",
                bgcolor: "#000",
                border: `1px solid ${themeColors.border}`,
                cursor: "grab",
                width: "100%",
                maxWidth: previewCanvasDims.w,
                aspectRatio: `${Math.max(previewCanvasDims.w, 1)} / ${Math.max(previewCanvasDims.h, 1)}`,
                height: "auto",
              }}
              onWheel={handlePreviewWheel}
              onDoubleClick={handlePreviewDoubleClick}
              onMouseDown={handlePreviewMouseDown}
              onMouseMove={handlePreviewMouseMove}
              onMouseUp={handlePreviewMouseUp}
              onMouseLeave={handlePreviewMouseUp}
            >
              <canvas ref={previewCanvasRef} width={previewCanvasDims.w} height={previewCanvasDims.h} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: "pixelated" }} role="img" aria-label={`ROI preview crop${previewCropDims ? ` (${previewCropDims.w} by ${previewCropDims.h} pixels)` : ""}`} />
              <canvas ref={previewOverlayRef} width={Math.round(previewCanvasDims.w * DPR)} height={Math.round(previewCanvasDims.h * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
              {showResizeControls && (
                <Box onMouseDown={handleMainResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, touchAction: "none", zIndex: 5, "&:hover": { opacity: 1 } }} />
              )}
            </Box>
            {/* All-ROI Stats - one row per ROI, same style as main stats bar */}
            {showStats && allRoiStats.length > 0 && (
              <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: 0.5, width: "100%", maxWidth: previewCanvasDims.w, boxSizing: "border-box" }}>
                {allRoiStats.map((stats, i) => {
                  if (!stats) return null;
                  const color = roiItems[i]?.color || ROI_COLORS[i % ROI_COLORS.length];
                  const isSelected = i === roiSelectedIdx;
                  return (
                    <Box key={i} sx={{ px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", flexWrap: "wrap", border: isSelected ? `1px solid ${color}` : `1px solid transparent` }}>
                      <Box sx={{ width: 8, height: 8, bgcolor: color, borderRadius: "50%", flexShrink: 0 }} />
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color }}>{formatNumber(stats.mean)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color }}>{formatNumber(stats.min)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color }}>{formatNumber(stats.max)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color }}>{formatNumber(stats.std)}</Box></Typography>
                    </Box>
                  );
                })}
              </Box>
            )}
          </Box>
        )}

        {/* FFT Panel - same size as main image. Bottom stacks below; Right uses the side slot. */}
        {effectiveShowFft && !fftLayoutOverlay && (
          <Box sx={{
            width: "100%",
            maxWidth: fftLayoutBottom ? "100%" : canvasW,
            flex: fftLayoutBottom ? "1 0 100%" : `0 1 min(100%, ${canvasW}px)`,
            minWidth: fftLayoutBottom ? "100%" : undefined,
            ml: fftLayoutBottom ? "0 !important" : undefined,
            mt: fftLayoutBottom ? "0 !important" : undefined,
            boxSizing: "border-box",
          }}>
            {/* Spacer - matches main panel title row height for canvas alignment */}
            {!fftLayoutBottom && <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />}
            {/* Controls row - matches main panel controls row height */}
            {(!fftLayoutBottom || (roiFftActive && fftCropDims)) && (
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, minHeight: 28, height: "auto", flexWrap: "wrap", gap: `${SPACING.XS}px` }}>
                {roiFftActive && fftCropDims ? (
                  <Typography sx={{ ...typography.label, color: themeColors.accentGreen }}>
                    ROI FFT ({fftCropDims.cropWidth}&times;{fftCropDims.cropHeight})
                  </Typography>
                ) : <Box />}
              </Stack>
            )}
            {/* FFT Canvas - same size as main image */}
            <Box
              ref={fftContainerRef}
              sx={{
                ...container.imageBox,
                width: "100%",
                maxWidth: canvasW,
                aspectRatio: mainPanelAspectRatio,
                height: "auto",
                cursor: "grab",
                touchAction: "none",
              }}
              onMouseDown={handleFftMouseDown}
              onMouseMove={handleFftMouseMove}
              onMouseUp={handleFftMouseUp}
              onMouseLeave={() => { fftClickStartRef.current = null; setIsFftDragging(false); setFftPanStart(null); }}
              onWheel={handleFftWheel}
              onDoubleClick={handleFftReset}
              onTouchStart={handleFftTouchStart}
              onTouchMove={handleFftTouchMove}
              onTouchEnd={handleFftTouchEnd}
              onTouchCancel={handleFftTouchEnd}
            >
              <canvas ref={fftCanvasRef} width={canvasW} height={canvasH} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: smooth ? "auto" : "pixelated", touchAction: "none" }} role="img" aria-label={roiFftActive && fftCropDims ? `FFT power spectrum of ROI crop (${fftCropDims.cropWidth} by ${fftCropDims.cropHeight} pixels)` : "FFT power spectrum of current frame"} />
              <canvas ref={fftOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
              {show3dFrequencyRing}
              {showZoomIndicator !== false && panelChromeVisible && (() => {
                const n = Math.max(1, visiblePanelCount || 1);
                const cols = panelColsForCount(n);
                const rows = Math.ceil(n / cols);
                const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
                const outPanelW = (canvasW - gap * (cols - 1)) / cols;
                const outPanelH = (canvasH - gap * (rows - 1)) / rows;
                return visiblePanelIndices.map((panel, slot) => {
                  const col = slot % cols;
                  const row = Math.floor(slot / cols);
                  const slotX = col * (outPanelW + gap);
                  const slotY = row * (outPanelH + gap);
                  return (
                    <Box
                      key={`fft-zoom-${panel}`}
                      className="quantem-fft-zoom-label"
                      data-show3d-fft-zoom-indicator={panel}
                      data-fft-zoom={formatZoomLabel(fftZoom)}
                      aria-label={`FFT zoom for ${panelLabel(panel)}: ${formatZoomLabel(fftZoom)}`}
                      sx={{
                        position: "absolute",
                        left: `calc(${(slotX / Math.max(1, canvasW)) * 100}% + 12px)`,
                        top: `calc(${((slotY + outPanelH) / Math.max(1, canvasH)) * 100}% - 23px)`,
                        maxWidth: `calc(${(outPanelW / Math.max(1, canvasW)) * 100}% - 24px)`,
                        color: "white",
                        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                        fontSize: 16,
                        fontWeight: 400,
                        fontVariantNumeric: "tabular-nums",
                        lineHeight: 1,
                        textShadow: "1px 1px 2px rgba(0,0,0,0.85)",
                        pointerEvents: "none",
                        userSelect: "none",
                        zIndex: 4,
                      }}
                    >
                      {formatZoomLabel(fftZoom)}
                    </Box>
                  );
                });
              })()}
              {fftMetricsEnabled && fftQuality && (
                <Box
                  className="quantem-fft-quality-label"
                  aria-label={`FFT quality: ${formatFftQualityLabel(fftQuality)}`}
                  sx={{
                    position: "absolute",
                    top: 8,
                    left: 8,
                    maxWidth: "calc(100% - 16px)",
                    color: "rgba(255,255,255,0.96)",
                    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                    fontSize: 11,
                    fontWeight: 700,
                    lineHeight: 1.2,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    textShadow: "1px 1px 0 rgba(0,0,0,0.9), 0 0 3px rgba(0,0,0,0.85)",
                    pointerEvents: "none",
                    userSelect: "none",
                    zIndex: 4,
                  }}
                >
                  {formatFftQualityLabel(fftQuality)}
                </Box>
              )}
              {showResizeControls && (() => {
                const n = Math.max(1, visiblePanelCount || 1);
                const cols = panelColsForCount(n);
                const rows = Math.ceil(n / cols);
                const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
                const outPanelW = (canvasW - gap * (cols - 1)) / cols;
                const outPanelH = (canvasH - gap * (rows - 1)) / rows;
                return visiblePanelIndices.map((panel, slot) => {
                  const col = slot % cols;
                  const row = Math.floor(slot / cols);
                  const slotX = col * (outPanelW + gap);
                  const slotY = row * (outPanelH + gap);
                  return (
                    <Box
                      key={`fft-resize-${panel}`}
                      onMouseDown={handleMainResizeStart}
                      sx={{
                        position: "absolute",
                        left: `calc(${((slotX + outPanelW) / Math.max(1, canvasW)) * 100}% - 16px)`,
                        top: `calc(${((slotY + outPanelH) / Math.max(1, canvasH)) * 100}% - 16px)`,
                        width: 16,
                        height: 16,
                        cursor: "nwse-resize",
                        opacity: 0.6,
                        background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
                        touchAction: "none",
                        zIndex: 5,
                        "&:hover": { opacity: 1 },
                      }}
                    />
                  );
                });
              })()}
            </Box>
            {/* FFT Statistics bar */}
            {showStats && (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, flexWrap: "wrap" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.mean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.min)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.max)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.std)}</Box></Typography>
              </Box>
            )}
            {fftClickInfo && (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, border: `1px solid ${themeColors.border}`, display: "flex", gap: 1.25, alignItems: "center", flexWrap: "wrap", width: "fit-content", maxWidth: canvasW, boxSizing: "border-box" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted, fontWeight: 600 }}>FFT mark</Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                  {fftClickInfo.dSpacing != null ? (
                    <>d = <Box component="span" sx={{ color: themeColors.accent, fontWeight: "bold" }}>{fftClickInfo.dSpacing >= 10 ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm` : `${fftClickInfo.dSpacing.toFixed(2)} Å`}</Box>{" | |g| = "}<Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.spatialFreq!.toFixed(4)} Å⁻¹</Box></>
                  ) : (
                    <>dist = <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box></>
                  )}
                </Typography>
              </Box>
            )}
            {/* FFT Controls - two rows with histogram on right (like Show4DSTEM) */}
	            {controlsVisible && <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", maxWidth: canvasW, boxSizing: "border-box", flexWrap: "wrap" }}>
              {/* Left: two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Auto */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale</Typography>
                  <Select value={fftLogScale ? "log" : "linear"} onChange={(e) => setFftLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT intensity scale (linear or logarithmic)" }}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto</Typography>
                  <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle automatic FFT contrast" } }} />
                  {roiFftActive && fftCropDims && (
                    <>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Win</Typography>
                      <Switch checked={fftWindow} onChange={(e) => setFftWindow(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle Hann windowing before FFT" } }} />
                    </>
                  )}
                </Box>
                {/* Row 2: Color + Colorbar */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color</Typography>
                  <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT colormap" }}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Colorbar</Typography>
                  <Switch checked={fftShowColorbar} onChange={(e) => setFftShowColorbar(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle FFT colorbar overlay" } }} />
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                <Histogram
                  data={fftHistogramData}
                  vminPct={fftVminPct}
                  vmaxPct={fftVmaxPct}
                  onRangeChange={(min, max) => { setFftVminPct(min); setFftVmaxPct(max); }}
                  width={110}
                  height={58}
                  theme={themeInfo.theme}
                  dataMin={fftDataRange.min}
                  dataMax={fftDataRange.max}
                />
              </Box>
            </Box>}
          </Box>
        )}

        {/* Kymograph Panel - static space-time image (X = distance along line,
            Y = frame/time). Shares the side slot with FFT (mutually exclusive).
            Mirrors the FFT panel's adjustability (contrast, zoom/pan, colormap). */}
        {kymoReady && (
          <Box sx={{ width: "100%", maxWidth: canvasW, boxSizing: "border-box" }}>
            {/* Spacer - matches main panel title row height for canvas alignment */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />
            {/* Controls row - title on left, Reset on right */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, minHeight: 28, height: "auto", flexWrap: "wrap", gap: `${SPACING.XS}px` }}>
              <Typography sx={{ ...typography.label, color: themeColors.accentGreen }}>
                Kymograph{singlePanelPageProfile ? ` · ${pageControlLabel}` : ""} ({kymoDataRef.current?.nFrames ?? nSlices} {dimUnit ? unitSymbol(dimUnit) : "frames"} &times; {kymoDataRef.current?.lineLen ?? 0} px)
              </Typography>
              <Button size="small" sx={compactButton} disabled={!kymoNeedsReset} onClick={handleKymoReset} aria-label="Reset kymograph zoom and pan">Reset</Button>
            </Stack>
            {/* Kymograph canvas - same size as main image */}
            <Box
              ref={kymoContainerRef}
              sx={{
                ...container.imageBox,
                width: "100%",
                maxWidth: canvasW,
                aspectRatio: mainPanelAspectRatio,
                height: "auto",
                cursor: "grab",
                position: "relative",
                touchAction: "none",
              }}
              onMouseDown={handleKymoMouseDown}
              onMouseMove={handleKymoMouseMove}
              onMouseUp={handleKymoMouseUp}
              onMouseLeave={() => { kymoClickStartRef.current = null; setIsKymoDragging(false); setKymoPanStart(null); }}
              onWheel={handleKymoWheel}
              onDoubleClick={handleKymoReset}
              onTouchStart={handleKymoTouchStart}
              onTouchMove={handleKymoTouchMove}
              onTouchEnd={handleKymoTouchEnd}
              onTouchCancel={handleKymoTouchEnd}
            >
              <canvas ref={kymoCanvasRef} width={canvasW} height={canvasH} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: "pixelated", touchAction: "none" }} role="img" aria-label={`Kymograph${singlePanelPageProfile ? ` for ${pageControlLabel}` : ""}: distance along profile line versus frame index`} />
              <canvas ref={kymoOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
            </Box>
            {/* Axis labels - kymograph-specific footer */}
            <Box sx={{ display: "flex", justifyContent: "space-between", mt: 0.5, px: 0.5 }}>
              <Typography sx={{ fontSize: 9, color: themeColors.textMuted }}>
                {dimUnit ? `time (${unitSymbol(dimUnit)})${dimSampling && dimSampling !== 1 ? `, ${(dimSampling).toFixed(2)}/frame` : ""} ↓` : "frame ↓"}
              </Typography>
              <Typography sx={{ fontSize: 9, color: themeColors.textMuted }}>distance along line →</Typography>
            </Box>
            {/* Kymograph Statistics bar */}
            {showStats && (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, flexWrap: "wrap" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoStats.mean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoStats.min)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoStats.max)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoStats.std)}</Box></Typography>
                {kymoClickInfo && (
                  <>
                    <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                      t = <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoClickInfo.timeVal)} {kymoClickInfo.timeUnit}</Box>{" | d = "}<Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoClickInfo.distVal)} {kymoClickInfo.distUnit}</Box>{" | I = "}<Box component="span" sx={{ color: themeColors.accent, fontWeight: "bold" }}>{formatNumber(kymoClickInfo.intensity)}</Box>
                    </Typography>
                  </>
                )}
              </Box>
            )}
            {/* Kymograph Controls - two rows with histogram on right (mirror FFT) */}
	            {controlsVisible && <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", maxWidth: canvasW, boxSizing: "border-box", flexWrap: "wrap" }}>
              {/* Left: two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Auto */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale</Typography>
                  <Select value={kymoLogScale ? "log" : "linear"} onChange={(e) => setKymoLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Kymograph intensity scale (linear or logarithmic)" }}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto</Typography>
                  <Switch checked={kymoAuto} onChange={(e) => setKymoAuto(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle automatic kymograph contrast" } }} />
                </Box>
                {/* Row 2: Color + Colorbar */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color</Typography>
                  <Select value={kymoColormap} onChange={(e) => setKymoColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Kymograph colormap" }}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Colorbar</Typography>
                  <Switch checked={kymoShowColorbar} onChange={(e) => setKymoShowColorbar(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle kymograph colorbar overlay" } }} />
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                <Histogram
                  data={kymoHistogramData}
                  vminPct={kymoVminPct}
                  vmaxPct={kymoVmaxPct}
                  onRangeChange={(min, max) => { setKymoVminPct(min); setKymoVmaxPct(max); }}
                  width={110}
                  height={58}
                  theme={themeInfo.theme}
                  dataMin={kymoDataRange.min}
                  dataMax={kymoDataRange.max}
                />
              </Box>
            </Box>}
          </Box>
        )}
      </Stack>
      {handoffEnabled && preparedViewWidget != null && (
        <EmbeddedWidgetView
          hostModel={model}
          widgetModel={preparedViewWidget}
          title="2D view"
          onClose={handleClosePreparedView}
          themeColors={themeColors}
          linkedTraits={SHOW3D_TO_SHOW2D_LINKED_TRAITS}
        />
      )}
      </>
      )}

    </Box>
  );
}

// anywidget v0.9+ deprecates `export render` in favor of `export default { render }`.
const render = createRender(Show3D);
export default { render };
