import * as React from "react";
import Box from "@mui/material/Box";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";
import { computeHistogramFromBytes } from "../display/stats";

export interface HistogramProps {
  /** Pre-computed 256-element bin array (GPU histogram). Wins over `data`. */
  bins?: ArrayLike<number> | null;
  /** Raw values to bin on the CPU when no `bins` are supplied. Widgets whose
   *  histogram work is GPU-only pass neither and get an empty histogram. */
  data?: Float32Array | null;
  /** Bin `data` over [dataMin, dataMax] instead of its own min/max so scrubbing
   *  through a stack does not rescale the bars per frame. */
  pinBinsToRange?: boolean;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  /** Fired while dragging; falls back to onRangeChange when omitted. */
  onRangePreview?: (min: number, max: number) => void;
  /** Fired on release; falls back to onRangeChange when omitted. */
  onRangeCommit?: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
  ariaHidden?: boolean;
  /** Decimal places for the min/max labels and slider tooltips. */
  valueDecimals?: number;
  labelFontFamily?: string;
  /** Horizontal inset of the slider track inside the canvas width. */
  sliderInset?: number;
  /** Canvas background in the dark theme; widgets with a bluer palette override it. */
  darkBackground?: string;
}

/**
 * Intensity histogram with an overlaid two-thumb clip slider. The bars inside
 * the clip range are drawn brighter so the user sees which part of the data
 * maps onto the colormap. Dragging the track between the thumbs moves the
 * whole window; drags paint through refs and a single rAF so the image can
 * follow at frame rate without re-rendering the widget tree.
 */
export const Histogram = React.memo(function Histogram({
  bins: precomputedBins = null,
  data = null,
  pinBinsToRange = false,
  vminPct, vmaxPct, onRangeChange, onRangePreview, onRangeCommit,
  width = 110, height = 40, theme = "dark",
  dataMin = 0, dataMax = 1, ariaHidden = false,
  valueDecimals = 1,
  labelFontFamily = "monospace",
  sliderInset = 4,
  darkBackground = "#1a1a1a",
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const sliderRef = React.useRef<HTMLDivElement | null>(null);
  const minLabelRef = React.useRef<HTMLElement | null>(null);
  const maxLabelRef = React.useRef<HTMLElement | null>(null);
  const onRangeChangeRef = React.useRef(onRangeChange);
  const onRangePreviewRef = React.useRef(onRangePreview);
  const onRangeCommitRef = React.useRef(onRangeCommit);
  const pendingRangeRef = React.useRef<[number, number] | null>(null);
  const rangeRafRef = React.useRef<number | null>(null);
  const [liveRange, setLiveRange] = React.useState<[number, number]>([vminPct, vmaxPct]);
  React.useEffect(() => { setLiveRange([vminPct, vmaxPct]); }, [vminPct, vmaxPct]);
  const [liveVminPct, liveVmaxPct] = liveRange;
  const bins = React.useMemo(
    () => precomputedBins && precomputedBins.length > 0
      ? Array.from(precomputedBins)
      : data
        ? (pinBinsToRange ? computeHistogramFromBytes(data, 256, dataMin, dataMax) : computeHistogramFromBytes(data))
        : new Array<number>(256).fill(0),
    [precomputedBins, data, pinBinsToRange, dataMin, dataMax],
  );
  const colors = React.useMemo(() => theme === "dark"
    ? { bg: darkBackground, barActive: "#888", barInactive: "#444", border: "#333" }
    : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" },
  [theme, darkBackground]);
  const formatValue = React.useCallback((pct: number) => {
    const value = dataMin + (pct / 100) * (dataMax - dataMin);
    return value >= 1000 ? value.toExponential(valueDecimals) : value.toFixed(valueDecimals);
  }, [dataMax, dataMin, valueDecimals]);
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
    if (minLabelRef.current) minLabelRef.current.textContent = formatValue(lo);
    if (maxLabelRef.current) maxLabelRef.current.textContent = formatValue(hi);
    drawHistogram(lo, hi);
  }, [drawHistogram, formatValue]);
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
  const applySliderValue = (value: number | number[], emit: (min: number, max: number) => void) => {
    const [newMin, newMax] = value as number[];
    const next: [number, number] = [Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1)];
    setLiveRange(next);
    emit(next[0], next[1]);
  };
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
  const beginRangeDrag = React.useCallback((event: React.PointerEvent, dragWidth: number, lo0: number, hi0: number) => {
    const startX = event.clientX;
    const span = Math.max(1, hi0 - lo0);
    const previousCursor = document.body.style.cursor;
    document.body.style.cursor = "grabbing";
    const onMove = (moveEvent: PointerEvent) => {
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
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      document.removeEventListener("pointercancel", onUp);
      document.body.style.cursor = previousCursor;
      flushRangePreview();
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
    document.addEventListener("pointercancel", onUp);
  }, [applyRangePreview, emitRangePreview, flushRangePreview]);

  const sliderWidth = Math.max(1, width - sliderInset * 2);

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
          onPointerDownCapture={(e) => {
            if ((e.target as HTMLElement).closest(".MuiSlider-thumb")) return;
            const rect = sliderRef.current?.getBoundingClientRect();
            if (!rect) return;
            const lo = Math.max(0, Math.min(100, Math.min(liveVminPct, liveVmaxPct)));
            const hi = Math.max(0, Math.min(100, Math.max(liveVminPct, liveVmaxPct)));
            const pct = ((e.clientX - rect.left) / Math.max(1, rect.width)) * 100;
            if (pct < lo || pct > hi) return;
            // Leave a thumb-sized guard so a press near a handle still goes to MUI.
            const thumbGuardPct = Math.max(4, (10 / Math.max(1, rect.width)) * 100);
            if (Math.abs(pct - lo) <= thumbGuardPct || Math.abs(pct - hi) <= thumbGuardPct) return;
            beginRangeDrag(e, rect.width, lo, hi);
            e.preventDefault();
            e.stopPropagation();
            e.nativeEvent.stopImmediatePropagation();
          }}
          sx={{ position: "absolute", left: sliderInset, top: height - 1, width: sliderWidth, height: 8, display: "flex", alignItems: "flex-start", cursor: "grab", zIndex: 2, overflow: "visible", touchAction: "none" }}
        >
          <Slider
            value={liveRange}
            onChange={(_, value) => applySliderValue(value, emitRangePreview)}
            onChangeCommitted={(_, value) => applySliderValue(value, emitRangeCommit)}
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
        <Typography ref={minLabelRef} sx={{ fontSize: 8, fontFamily: labelFontFamily, opacity: 0.6, lineHeight: 1 }}>{formatValue(liveVminPct)}</Typography>
        <Typography ref={maxLabelRef} sx={{ fontSize: 8, fontFamily: labelFontFamily, opacity: 0.6, lineHeight: 1 }}>{formatValue(liveVmaxPct)}</Typography>
      </Box>
    </Box>
  );
});
