/**
 * Shared design tokens for quantem-live dashboard.
 *
 * Color system: light tokens for trials/screen pages,
 * dark tokens for processing pages (denoise, drift).
 * Every color in the UI should come from here.
 */

export const colors = {
  /* ── Light page tokens (trials, screen) ── */

  text: {
    primary: "#111",
    secondary: "#374151",
    tertiary: "#6b7280",
    muted: "#9ca3af",
    disabled: "#d1d5db",
    placeholder: "#ccc",
    faint: "#888",
    subtle: "#666",
    dim: "#555",
    hint: "#999",
    dark: "#333",
    black: "#000",
    white: "#fff",
  },

  bg: {
    page: "#fff",
    subtle: "#fafafa",
    muted: "#f3f4f6",
    hover: "#f9fafb",
    soft: "#f5f5f5",
    canvas: "#f0f0f0",
    row: "#eee",
    dark: "#1e293b",
    darker: "#1f2937",
    black: "#111827",
  },

  border: {
    light: "#f3f4f6",
    default: "#e5e7eb",
    medium: "#d1d5db",
    dark: "#374151",
    soft: "#e5e5e5",
  },

  shadow: {
    subtle: "rgba(0,0,0,0.08)",
    medium: "rgba(0,0,0,0.16)",
  },

  warning: {
    text: "#991b1b",
    bg: "#fef2f2",
    border: "#fca5a5",
    dot: "#dc2626",
    light: "#ef4444",
    medium: "#ca8a04",
  },

  success: {
    text: "#166534",
    dot: "#16a34a",
    bg: "#dcfce7",
  },

  interactive: {
    text: "#2563eb",
    bg: "#eff6ff",
    border: "#93c5fd",
    active: "#2563eb",
    hover: "#1d4ed8",
    focus: "#3b82f6",
    selectedText: "#1e40af",
  },

  star: {
    filled: "#f59e0b",
    hover: "#d97706",
  },

  status: {
    goodBg: "#dcfce7",
    badBg: "#fee2e2",
    reconstructBg: "#dbeafe",
    starredBg: "#fffbeb",
    compareBg: "#fef3c7",
    selectedBg: "#dbeafe",
    activeBg: "#e3f2fd",
    filterBg: "#fffde7",
    /* Drag-toggle multi-select row tint (#517). Soft Ba-blue at 18% so the
       selection reads at a glance without overpowering the row's own status
       coloring (starred, hidden, running). */
    selectedTint: "rgba(99, 162, 255, 0.18)",
  },

  overlay: {
    dark: "rgba(0,0,0,0.55)",
    darker: "rgba(0,0,0,0.70)",
    iconIdle: "rgba(255,255,255,0.5)",
    iconHover: "rgba(255,255,255,0.8)",
    textShadow: "#000",
    panelShadow: "rgba(0,0,0,0.5)",
    plotBg: "#0b0b0e",
    peakMarker: "#ffd700",
    freqRing: "rgba(100,200,255,0.35)",
    freqLabel: "rgba(100,200,255,0.7)",
    driftLine: "rgba(255,220,100,0.6)",
    histPhase: "#60a5fa",
    histFft: "#a78bfa",
    profileLine: "#60a5fa",
    hoverAccent: "#00ffee",    // cyan accent on hover readouts (HoverPixelChip, Gallery)
    profileAccent: "#00ff00",  // bright green for line-profile endpoints + ROI stroke
    roiAlt: "#4488ff",         // blue for alt ROI strokes (Shift+drag etc.)
    roiWarm: "#f97316",        // orange guide line for stack profiles / ROI crosshair
    roiFocusDim: "rgba(0,0,0,0.46)", // dim non-ROI image area while an ROI is focused
    roiSelectedHalo: "rgba(255,255,255,0.96)",
    roiSelectedGlow: "rgba(37,99,235,0.72)",
    roiSelectedFill: "rgba(37,99,235,0.10)",
    roiHoverHalo: "rgba(255,255,255,0.72)",
    roiHoverGlow: "rgba(37,99,235,0.42)",
    roiRowHoverBg: "rgba(37,99,235,0.06)",
    roiRowSelectedBg: "rgba(37,99,235,0.12)",
    roiRowSelectedBorder: "rgba(37,99,235,0.55)",
    roiRowSelectedShadow: "rgba(37,99,235,0.22)",
    onImage: "#ddd",           // label text color over canvas (near-white, soft)
    onImageBg: "#000",         // background color behind canvases / image tiles
  },

  /* ── Dark page tokens (denoise, drift, processing) ── */

  dark: {
    bg: {
      page: "#0f0f1a",
      panel: "#1a1a2e",
      card: "#2a2a4a",
      cardHover: "#3a3a5a",
      elevated: "#353560",
      input: "#2a3a4a",
      deep: "#0a0a14",
      sidebar: "#1e2a3a",
      deepest: "#0a0a1a",
      muted: "#1e1e3a",
      hover: "#222244",
    },
    text: {
      primary: "#e2e8f0",
      secondary: "#9ca3af",
      muted: "#6b7280",
      dim: "#4a5568",
    },
    accent: {
      primary: "#7c9ef7",
      hover: "#5a82e0",
      secondary: "#a0c4ff",
      blue: "#60a5fa",
    },
    status: {
      success: "#4ade80",
      warning: "#f59e0b",
      error: "#fc8181",
      errorBg: "#5a2a2a",
      errorBgDeep: "#3a1a1a",
      successBg: "#1e3a2a",
      cacheBg: "#1e3a5f",
      herbBg: "#064e3b",
      herbHover: "#065f46",
      done: "#d1fae5",
      active: "#bfdbfe",
      rowShift: "#f87171",
      purple: "#a78bfa",
      modeActive: "#92400e",
      star: "#fbbf24",
    },
    runStatus: {
      pending: { bg: "#3a3a4a", fg: "#cbd5e1", border: "#4a4a5a" },
      done: { bg: "#1e3a2a", fg: "#4ade80", border: "#2a4a3a" },
      failed: { bg: "#3a1a1a", fg: "#fc8181", border: "#5a2a2a" },
      canceled: { bg: "#2a2a3a", fg: "#9ca3af", border: "#3a3a4a" },
      stuck: { bg: "#3a2a14", fg: "#fbbf24", border: "#92400e", rowBg: "rgba(146, 64, 14, 0.08)", rowHoverBg: "rgba(146, 64, 14, 0.18)" },
      stale: { bg: "#2a2014", fg: "#fbbf24", border: "#7a5a14" },
    },
    planKind: {
      explore: { bg: "#1a2a3a", fg: "#7dd3fc", border: "#0369a1" },
      refine: { bg: "#2a1a3a", fg: "#c4b5fd", border: "#6d28d9" },
      series: { bg: "#1a3a2a", fg: "#86efac", border: "#15803d" },
    },
    border: {
      default: "#2a3a6a",
      subtle: "#2a3a4a",
      input: "#2a3a5a",
      success: "#2a4a3a",
      hover: "#5a7ec7",
      error: "#f56565",
    },
  },

  /* ── Chart / canvas palette ── */

  chart: [
    "#2563eb", "#16a34a", "#f59e0b", "#dc2626",
    "#8b5cf6", "#0891b2", "#ea580c", "#d946ef",
  ],

  /* FFT / d-spacing overlay annotation colors (Gallery spot markers).
   * Bright gold for contrast against the inferno-colormapped FFT power. */
  annotation: {
    goldOverlay: "#ffcc66",
  },

  /* GPU pill palette (issue #303). Deterministic per-GPU colors shared
   * between the Queue page and the Trials table running-row chip. Each
   * slot has bg + fg + border so the pill passes WCAG contrast on both
   * light and dark surfaces. Unknown/unassigned GPU uses `gpuUnknown`. */
  gpuPalette: [
    { bg: "#dcfce7", fg: "#166534", border: "#16a34a" }, // 0 green
    { bg: "#dbeafe", fg: "#1e40af", border: "#3b82f6" }, // 1 blue
    { bg: "#fef3c7", fg: "#92400e", border: "#f59e0b" }, // 2 amber
    { bg: "#fce7f3", fg: "#9f1239", border: "#ec4899" }, // 3 pink
    { bg: "#e0e7ff", fg: "#4338ca", border: "#6366f1" }, // 4 indigo
    { bg: "#ecfccb", fg: "#3f6212", border: "#84cc16" }, // 5 lime
    { bg: "#cffafe", fg: "#155e75", border: "#06b6d4" }, // 6 cyan
    { bg: "#fee2e2", fg: "#991b1b", border: "#ef4444" }, // 7 red
  ],
  gpuUnknown: { bg: "#f3f4f6", fg: "#6b7280", border: "#d1d5db" },

  /* Floating GPU status chip + popover palette. Kept here so process
   * activity badges stay aligned with the rest of the dashboard chrome. */
  gpuStatus: {
    surface: "rgba(20, 20, 30, 0.92)",
    surfaceStrong: "rgba(20, 20, 30, 0.96)",
    shadow: "0 2px 12px rgba(0,0,0,0.35)",
    shadowStrong: "0 8px 32px rgba(0,0,0,0.5)",
    selectedBg: "rgba(234, 179, 8, 0.12)",
    hoverBg: "rgba(255,255,255,0.08)",
    subActivity: {
      browse: "#26C6DA",
      trials: "#FFB74D",
      screen: "#BA68C8",
      denoise: "#81C784",
      drift: "#FF8A65",
      queue: "#90A4AE",
    },
    activity: {
      dashboard: "#4FC3F7",
      ptycho: "#FFB74D",
      screen: "#BA68C8",
      denoise: "#81C784",
      jupyter: "#F06292",
      node: "#9CCC65",
      python: "#90A4AE",
    },
  },

  /* Queue-page accent tokens. running-row state + pending/done badges. */
  queue: {
    runningBg: "#dbeafe",
    runningFg: "#1e40af",
    runningBorder: "#3b82f6",
    errorBg: "#fee2e2",
    errorFg: "#991b1b",
    errorBorder: "#dc2626",
    rowHover: "#f9fafb",
    divider: "#e5e7eb",

    /* Status badges on the Queue page row (pending/running/done/failed/canceled). */
    statusBadge: {
      pending:  { bg: "#e5e7eb", fg: "#374151" },
      running:  { bg: "#dbeafe", fg: "#1e40af" },
      done:     { bg: "#dcfce7", fg: "#166534" },
      failed:   { bg: "#fee2e2", fg: "#991b1b" },
      canceled: { bg: "#f3e8ff", fg: "#6b21a8" },
    },

    /* Empty-slot placeholder (no thumb yet) + muted text. */
    slotPlaceholderBg: "#f3f4f6",
    slotPlaceholderBorder: "#d1d5db",
    mutedText: "#6b7280",
    mutedTextLight: "#9ca3af",
  },

  /* Sparkline / sweep strip dot colors. Selection ring must be an
   * orthogonal color (blue) so a green "good" dot stays legible under
   * the ring. See docs/user-guide/ui-publication-quality.md §Sparkline. */
  sparkline: {
    good: "#16a34a",       // green: loss at/below the "good" threshold
    medium: "#f59e0b",     // amber: loss between "good" and "bad"
    bad: "#dc2626",        // red: loss above the quality gate
    best: "#059669",       // darker green: minimum-loss dot
    selected: "#2563eb",   // blue ring around selected/focused dot
    axis: "#4b5563",       // axis line + tick color
    tick: "#374151",       // tick label color
    baseline: "#d1d5db",   // dashed threshold baseline
  },

  /* Active nav link color. Split by page theme so the nav label stays
   * readable on both light (#fff) and dark (#0f0f1a) page backgrounds.
   * Both clear WCAG AA (>= 4.5:1) against their respective page bg. */
  activeNav: "#111",         // on light pages (trials, screening)
  activeNavDark: "#f3f4f6",  // on dark pages (denoise, drift)
} as const;

export const radii = {
  sm: 2,    // 0.25 in MUI units
  md: 4,    // 0.5 in MUI units
  lg: 8,    // 1 in MUI units
} as const;

export const fontSizes = {
  xs: 9,
  sm: 10,
  md: 11,
  base: 12,
  lg: 13,
  xl: 14,
  section: 15,   // processing-page section labels (Denoise/Drift)
  "2xl": 16,
  "3xl": 18,
} as const;

/**
 * Responsive breakpoints (issue #116). Widths in px, max-width semantics.
 * - phone:   < 600  single-column, burger nav, full-screen popup
 * - tablet:  < 1024 two-column where sensible, nav visible
 * - desktop: >= 1024 full layout
 * Usage:
 *   @media (max-width: ${breakpoints.phone}px) { ... }
 *   sx={{ flexDirection: { xs: "column", md: "row" } }}  // MUI key equiv
 */
export const breakpoints = {
  phone: 600,
  tablet: 1024,
  desktop: 1440,
} as const;

/**
 * Shared dark theme for processing pages (Denoise, Drift, Denoise3D, DenoiseLive).
 * Import and pass to <ThemeProvider> instead of copy-pasting createTheme().
 */
import { createTheme, type Theme } from "@mui/material/styles";

export const processingDarkTheme: Theme = createTheme({
  palette: {
    mode: "dark",
    background: { default: colors.dark.bg.page, paper: colors.dark.bg.panel },
    primary: { main: colors.dark.accent.primary },
    secondary: { main: colors.dark.status.purple },
  },
  typography: {
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    fontSize: 13,
  },
  components: {
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: radii.md,
          "&.Mui-selected": { backgroundColor: colors.dark.bg.card },
          "&.Mui-selected:hover": { backgroundColor: colors.dark.bg.cardHover },
        },
      },
    },
  },
});
