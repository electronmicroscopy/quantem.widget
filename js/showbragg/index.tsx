/** ShowBragg - panel stack driving the quantem BraggVectors workflow. */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import FormControlLabel from "@mui/material/FormControlLabel";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import { useTheme, type ThemeColors } from "../theme";
import { preserveRestoredWidgetModelsOnSave } from "../format";
import { useHideStaticFallback } from "../staticFallback";
import {
  canvasPoint,
  clamp,
  drawImage,
  imageToScreen,
  screenToImage,
  usePngBitmap,
  zoomAt,
  type ImageViewport,
  type ViewTransform,
} from "../imageView";

const CANVAS_SIZE = 340;
const HIT_PX = 10;
const CLICK_MOVE_THRESHOLD_PX = 4;
const BASIS_COLORS = ["#ff4d4f", "#40a9ff", "#73d13d"];
const LABEL_CLEARANCE_PX = 26;
const BASIS_LABELS = ["origin", "g1", "g2"] as const;
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;
const CMAPS = ["inferno", "magma", "viridis", "plasma", "gray", "turbo"];

const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  textTransform: "none" as const,
};

type Role = (typeof BASIS_LABELS)[number];
type Marker = { row: number; col: number; color: string; label: string };
type PreviewPeaks = { positions: number[][]; counts: number[]; peaks: number[][] };

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function Panel({
  title,
  colors,
  blocked,
  children,
}: {
  title: string;
  colors: ThemeColors;
  blocked?: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = React.useState(true);

  return (
    <Box sx={{ border: `1px solid ${colors.border}`, borderRadius: 1, mb: `${SPACING.SM}px` }}>
      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems="center"
        onClick={() => setOpen(!open)}
        sx={{ px: `${SPACING.MD}px`, py: `${SPACING.SM}px`, cursor: "pointer", bgcolor: colors.bgAlt }}
      >
        <Typography sx={{ fontSize: 12, fontWeight: 600 }}>{title}</Typography>
        <Typography sx={{ fontSize: 10, color: colors.textMuted }}>{open ? "hide" : "show"}</Typography>
      </Stack>

      {open && (
        <Box sx={{ p: `${SPACING.MD}px` }}>
          {blocked ? (
            <Typography sx={{ fontSize: 11, color: colors.textMuted }}>{blocked}</Typography>
          ) : (
            children
          )}
        </Box>
      )}
    </Box>
  );
}

function NumberField({
  label,
  value,
  onChange,
  step = 1,
  disabled,
  colors,
  hint,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  disabled?: boolean;
  colors: ThemeColors;
  hint?: string;
}) {
  return (
    <TextField
      label={label}
      title={hint}
      type="number"
      size="small"
      disabled={disabled}
      value={value}
      onChange={(e) => {
        const parsed = Number(e.target.value);
        if (Number.isFinite(parsed)) onChange(parsed);
      }}
      slotProps={{ htmlInput: { step }, inputLabel: { shrink: true } }}
      sx={{
        width: 120,
        "& .MuiInputBase-input": { fontSize: 11, py: 0.5, color: colors.text },
        "& .MuiInputLabel-root": { fontSize: 11, color: colors.textMuted },
      }}
    />
  );
}

function ImageCanvas({
  bytes,
  shape,
  colors,
  overlay,
  markers,
  onMoveMarker,
  onPick,
  caption,
}: {
  bytes: DataView | null;
  shape: number[];
  colors: ThemeColors;
  overlay?: (ctx: CanvasRenderingContext2D, viewport: ImageViewport) => void;
  markers?: Marker[];
  onMoveMarker?: (index: number, row: number, col: number) => void;
  onPick?: (row: number, col: number) => void;
  caption?: string;
}) {
  const image = usePngBitmap(bytes);
  const [view, setView] = React.useState<ViewTransform>({ zoom: 1, panX: 0, panY: 0 });
  const baseRef = React.useRef<HTMLCanvasElement>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);

  const [height, width] = shape.length === 2 ? shape : [1, 1];
  const viewport: ImageViewport = React.useMemo(
    () => ({ height, width, canvas: CANVAS_SIZE, ...view }),
    [height, width, view],
  );

  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const prevent = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", prevent, { passive: false });
    return () => el.removeEventListener("wheel", prevent);
  }, []);

  React.useLayoutEffect(() => {
    const canvas = baseRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    if (image) drawImage(ctx, image, viewport);
  }, [image, viewport, colors.bg]);

  React.useLayoutEffect(() => {
    const canvas = overlayRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    overlay?.(ctx, viewport);

    (markers ?? []).forEach(({ row, col, color, label }) => {
      const [x, y] = imageToScreen(viewport, row, col);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.font = "bold 11px -apple-system, sans-serif";
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = "rgba(0,0,0,0.85)";
      ctx.lineWidth = 3;
      ctx.strokeText(label, x + 8, y - 6);
      ctx.fillText(label, x + 8, y - 6);
    });
  }, [overlay, markers, viewport]);

  const dragRef = React.useRef<{
    moved: boolean;
    startX: number;
    startY: number;
    panX: number;
    panY: number;
    marker: number;
  } | null>(null);

  const toImage = (e: React.MouseEvent): [number, number] => {
    const canvas = baseRef.current;
    if (!canvas) return [0, 0];
    const [x, y] = canvasPoint(canvas, e);
    return screenToImage(viewport, x, y);
  };

  const hitMarker = (row: number, col: number): number => {
    if (!onMoveMarker) return -1;
    const scale = (CANVAS_SIZE / Math.max(height, width)) * view.zoom;
    const radius = HIT_PX / scale;
    const list = markers ?? [];
    for (let i = list.length - 1; i >= 0; i--) {
      if (Math.hypot(row - list[i].row, col - list[i].col) <= radius) return i;
    }
    return -1;
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const [row, col] = toImage(e);
    dragRef.current = {
      moved: false,
      startX: e.clientX,
      startY: e.clientY,
      panX: view.panX,
      panY: view.panY,
      marker: hitMarker(row, col),
    };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const drag = dragRef.current;
    if (!drag) return;
    if (Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) > CLICK_MOVE_THRESHOLD_PX) {
      drag.moved = true;
    }
    if (drag.marker >= 0) {
      const [row, col] = toImage(e);
      onMoveMarker?.(
        drag.marker,
        clamp(row, 0, height - 1),
        clamp(col, 0, width - 1),
      );
      return;
    }
    if (drag.moved) {
      setView((v) => ({
        ...v,
        panX: drag.panX + (e.clientX - drag.startX),
        panY: drag.panY + (e.clientY - drag.startY),
      }));
    }
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag || drag.moved || drag.marker >= 0) return;
    const [row, col] = toImage(e);
    onPick?.(row, col);
  };

  return (
    <Box>
      <Box
        ref={containerRef}
        sx={{
          position: "relative",
          width: CANVAS_SIZE,
          height: CANVAS_SIZE,
          border: `1px solid ${colors.border}`,
          overflow: "hidden",
        }}
      >
        <canvas
          ref={baseRef}
          style={{ position: "absolute", top: 0, left: 0, width: CANVAS_SIZE, height: CANVAS_SIZE, imageRendering: "pixelated" }}
        />
        <canvas
          ref={overlayRef}
          style={{ position: "absolute", top: 0, left: 0, width: CANVAS_SIZE, height: CANVAS_SIZE, pointerEvents: "none" }}
        />
        <canvas
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: CANVAS_SIZE,
            height: CANVAS_SIZE,
            opacity: 0,
            cursor: onPick || onMoveMarker ? "crosshair" : "grab",
          }}
          onWheel={(e) => {
            const canvas = baseRef.current;
            if (!canvas) return;
            const [x, y] = canvasPoint(canvas, e);
            setView(zoomAt(viewport, x, y, e.deltaY));
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => { dragRef.current = null; }}
          onDoubleClick={() => setView({ zoom: 1, panX: 0, panY: 0 })}
        />
      </Box>

      {caption && (
        <Typography sx={{ fontSize: 10, color: colors.textMuted, mt: `${SPACING.XS}px` }}>
          {caption}
        </Typography>
      )}
    </Box>
  );
}

function ShowBragg() {
  const model = useModel();
  const rootRef = React.useRef<HTMLDivElement>(null);
  const { colors } = useTheme();

  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);
  useHideStaticFallback(model, rootRef);

  const [title] = useModelState<string>("title");
  const [showTitle] = useModelState<boolean>("show_title");
  const [showControls] = useModelState<boolean>("show_controls");
  const [controlsCollapsed, setControlsCollapsed] = useModelState<boolean>("controls_collapsed");
  const controlsVisible = showControls && !controlsCollapsed;
  const [scanShape] = useModelState<number[]>("scan_shape");
  const [qShape] = useModelState<number[]>("q_shape");
  const [status] = useModelState<string>("status");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");

  const [templateSource, setTemplateSource] = useModelState<string>("template_source");
  const [templateRadius, setTemplateRadius] = useModelState<number>("template_radius");
  const [templateEdge, setTemplateEdge] = useModelState<number>("template_edge");
  const [templateSubtractMean, setTemplateSubtractMean] = useModelState<boolean>("template_subtract_mean");
  const [templatePng] = useModelState<DataView>("template_png");
  const [templateShape] = useModelState<number[]>("template_shape");
  const [hasProbe] = useModelState<boolean>("has_probe");

  const [probePosition, setProbePosition] = useModelState<number[]>("probe_position");
  const [probeDiffractionPng] = useModelState<DataView>("probe_diffraction_png");
  const [probeCorrelationPng] = useModelState<DataView>("probe_correlation_png");

  const [minAbsIntensity, setMinAbsIntensity] = useModelState<number>("min_abs_intensity");
  const [minSpacing, setMinSpacing] = useModelState<number>("min_spacing");
  const [edgeBoundary, setEdgeBoundary] = useModelState<number>("edge_boundary");
  const [subpixel, setSubpixel] = useModelState<string>("subpixel");
  const [upsampleFactor, setUpsampleFactor] = useModelState<number>("upsample_factor");
  const [maxNumPeaks, setMaxNumPeaks] = useModelState<number>("max_num_peaks");
  const [previewGrid, setPreviewGrid] = useModelState<number>("preview_grid");
  const [previewPeaksJson] = useModelState<string>("preview_peaks");
  const [detectionState] = useModelState<string>("detection_state");

  const [bvmSampling, setBvmSampling] = useModelState<number>("bvm_sampling");
  const [bvmPng] = useModelState<DataView>("bvm_png");
  const [numCandidates, setNumCandidates] = useModelState<number>("num_candidates");
  const [candidateMinSpacing, setCandidateMinSpacing] = useModelState<number>("candidate_min_spacing");
  const [candidateMinAbsIntensity, setCandidateMinAbsIntensity] = useModelState<number>("candidate_min_abs_intensity");
  const [candidates] = useModelState<number[][]>("candidates");
  const [originIndex, setOriginIndex] = useModelState<number>("origin_index");
  const [g1Index, setG1Index] = useModelState<number>("g1_index");
  const [g2Index, setG2Index] = useModelState<number>("g2_index");
  const [originRc, setOriginRc] = useModelState<number[]>("origin_rc");
  const [g1Rc, setG1Rc] = useModelState<number[]>("g1_rc");
  const [g2Rc, setG2Rc] = useModelState<number[]>("g2_rc");

  const [minNumPeaks, setMinNumPeaks] = useModelState<number>("min_num_peaks");
  const [maxPeakShift, setMaxPeakShift] = useModelState<number>("max_peak_shift");
  const [fitState] = useModelState<string>("fit_state");
  const [maskWeightPng] = useModelState<DataView>("mask_weight_png");
  const [fitErrorPng] = useModelState<DataView>("fit_error_png");

  const send = React.useCallback(
    (content: Record<string, unknown>) => model.send(content),
    [model],
  );

  const detecting = detectionState === "running" || detectionState === "preview";
  const fitting = fitState === "running";
  const busy = detecting || fitting;
  const hasPeaks = candidates.length > 0 || bvmPng?.byteLength > 0;

  const preview: PreviewPeaks | null = React.useMemo(() => {
    if (!previewPeaksJson) return null;
    try {
      return JSON.parse(previewPeaksJson) as PreviewPeaks;
    } catch {
      return null;
    }
  }, [previewPeaksJson]);

  const drawPreviewPeaks = React.useCallback(
    (ctx: CanvasRenderingContext2D, viewport: ImageViewport) => {
      if (!preview) return;
      ctx.strokeStyle = "#40a9ff";
      ctx.lineWidth = 1.5;
      preview.peaks.forEach(([row, col]) => {
        const [x, y] = imageToScreen(viewport, row, col);
        ctx.beginPath();
        ctx.moveTo(x - 5, y);
        ctx.lineTo(x + 5, y);
        ctx.moveTo(x, y - 5);
        ctx.lineTo(x, y + 5);
        ctx.stroke();
      });
    },
    [preview],
  );

  const drawCandidates = React.useCallback(
    (ctx: CanvasRenderingContext2D, viewport: ImageViewport) => {
      const placed: Array<[number, number]> = [];
      const byBrightness = candidates
        .map((c, i) => ({ row: c[0], col: c[1], intensity: c[2] ?? 0, index: i }))
        .sort((a, b) => b.intensity - a.intensity);

      ctx.font = "10px -apple-system, sans-serif";
      for (const { row, col, index } of byBrightness) {
        const [x, y] = imageToScreen(viewport, row, col);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.strokeStyle = "rgba(255,255,255,0.8)";
        ctx.lineWidth = 1;
        ctx.stroke();

        const clear = placed.every(
          ([px, py]) => Math.hypot(x - px, y - py) >= LABEL_CLEARANCE_PX,
        );
        if (!clear) continue;
        placed.push([x, y]);

        ctx.fillStyle = "#fff";
        ctx.strokeStyle = "rgba(0,0,0,0.85)";
        ctx.lineWidth = 2.5;
        ctx.strokeText(String(index), x + 4, y - 4);
        ctx.fillText(String(index), x + 4, y - 4);
      }
    },
    [candidates],
  );

  const basisMarkers: Marker[] = React.useMemo(() => {
    if (originRc.length !== 2) return [];
    const tips = [originRc, g1Rc, g2Rc];
    return tips.flatMap((rc, i) => {
      if (rc.length !== 2) return [];
      const row = i === 0 ? rc[0] : originRc[0] + rc[0];
      const col = i === 0 ? rc[1] : originRc[1] + rc[1];
      return [{ row, col, color: BASIS_COLORS[i], label: BASIS_LABELS[i] }];
    });
  }, [originRc, g1Rc, g2Rc]);

  const setRoleVector = (index: number, row: number, col: number) => {
    if (index === 0) {
      setOriginIndex(-1);
      setOriginRc([row, col]);
    } else if (index === 1) {
      setG1Index(-1);
      setG1Rc([row - originRc[0], col - originRc[1]]);
    } else {
      setG2Index(-1);
      setG2Rc([row - originRc[0], col - originRc[1]]);
    }
  };

  const [activeRole, setActiveRole] = React.useState<Role>("origin");

  const pickCandidate = (row: number, col: number) => {
    if (!candidates.length) return;
    let nearest = 0;
    let best = Infinity;
    candidates.forEach(([cr, cc], i) => {
      const d = Math.hypot(row - cr, col - cc);
      if (d < best) {
        best = d;
        nearest = i;
      }
    });
    if (activeRole === "origin") setOriginIndex(nearest);
    else if (activeRole === "g1") setG1Index(nearest);
    else setG2Index(nearest);
  };

  const vectorText = (rc: number[]) =>
    rc.length === 2 ? `(${rc[0].toFixed(2)}, ${rc[1].toFixed(2)})` : "not set";

  return (
    <Box
      ref={rootRef}
      sx={{ p: `${SPACING.LG}px`, bgcolor: colors.bg, color: colors.text, width: "100%", boxSizing: "border-box" }}
    >
      {(showTitle || showControls) && (
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.SM}px` }}>
          {showTitle ? (
            <Typography sx={{ fontSize: 13, fontWeight: 600 }}>{title || "Show Bragg"}</Typography>
          ) : (
            <span />
          )}
          <Stack direction="row" spacing={1.5} alignItems="center">
            {showTitle && (
              <Typography sx={{ fontSize: 10, color: colors.textMuted }}>
                scan {scanShape.join(" x ")} · detector {qShape.join(" x ")}
              </Typography>
            )}
            {showControls && (
              <Button
                size="small"
                sx={{ ...compactButton, color: colors.accent }}
                onClick={() => setControlsCollapsed(!controlsCollapsed)}
                aria-label={controlsCollapsed ? "Show controls" : "Hide controls"}
              >
                {controlsCollapsed ? "Controls" : "Hide"}
              </Button>
            )}
          </Stack>
        </Stack>
      )}

      {status && (
        <Typography sx={{ fontSize: 11, color: "#ff7875", mb: `${SPACING.SM}px` }}>{status}</Typography>
      )}

      {controlsVisible && (
        <Stack
          direction="row"
          spacing={2}
          alignItems="center"
          flexWrap="wrap"
          useFlexGap
          sx={{
            mb: `${SPACING.SM}px`,
            px: `${SPACING.MD}px`,
            py: `${SPACING.XS}px`,
            border: `1px solid ${colors.border}`,
            borderRadius: 1,
            bgcolor: colors.bgAlt,
          }}
        >
          <Typography sx={{ fontSize: 10, color: colors.textMuted, letterSpacing: "0.06em" }}>
            DISPLAY
          </Typography>
          <Select
            size="small"
            value={cmap}
            onChange={(e) => setCmap(e.target.value)}
            sx={{ fontSize: 11, "& .MuiSelect-select": { py: 0.25, color: colors.text } }}
          >
            {CMAPS.map((name) => (
              <MenuItem key={name} sx={{ fontSize: 11 }} value={name}>
                {name}
              </MenuItem>
            ))}
          </Select>
          <FormControlLabel
            control={
              <Switch
                size="small"
                checked={Boolean(logScale)}
                onChange={(e) => setLogScale(e.target.checked)}
              />
            }
            label="log scale"
            slotProps={{ typography: { sx: { fontSize: 11, color: colors.text } } }}
          />
          <Typography sx={{ fontSize: 10, color: colors.textMuted }}>
            scroll to zoom · drag to pan · double-click to reset
          </Typography>
        </Stack>
      )}

      <Panel title="1. Template" colors={colors}>
        <Stack direction="row" spacing={2} alignItems="flex-start" flexWrap="wrap">
          {controlsVisible && (
          <Stack spacing={1.5} sx={{ minWidth: 200 }}>
            <Select
              size="small"
              value={templateSource}
              onChange={(e) => setTemplateSource(e.target.value)}
              sx={{ fontSize: 11, "& .MuiSelect-select": { py: 0.5, color: colors.text } }}
            >
              <MenuItem sx={{ fontSize: 11 }} value="synthetic">synthetic disk</MenuItem>
              <MenuItem sx={{ fontSize: 11 }} value="data">mean of data</MenuItem>
              <MenuItem sx={{ fontSize: 11 }} value="probe" disabled={!hasProbe}>measured probe</MenuItem>
            </Select>

            <NumberField
              label="radius"
              hint="disk radius in pixels; 0 estimates it"
              value={templateRadius}
              step={0.5}
              disabled={templateSource !== "synthetic"}
              onChange={setTemplateRadius}
              colors={colors}
            />
            <NumberField
              label="edge"
              hint="tanh edge falloff width"
              value={templateEdge}
              step={0.25}
              disabled={templateSource !== "synthetic"}
              onChange={setTemplateEdge}
              colors={colors}
            />
            <Button
              size="small"
              sx={{ ...compactButton, color: colors.accent, alignSelf: "flex-start" }}
              onClick={() => setTemplateSubtractMean(!templateSubtractMean)}
            >
              subtract mean: {templateSubtractMean ? "on" : "off"}
            </Button>
          </Stack>
          )}

          <ImageCanvas
            bytes={templatePng}
            shape={templateShape.length === 2 && templateShape[0] > 0 ? templateShape : qShape}
            colors={colors}
            caption="correlation template, cropped to the disk"
          />
        </Stack>
      </Panel>

      <Panel title="2. Probe" colors={colors}>
        <Stack direction="row" spacing={2} alignItems="flex-start" flexWrap="wrap">
          {controlsVisible && (
          <Stack spacing={1.5} sx={{ minWidth: 200 }}>
            <NumberField
              label="scan row"
              value={probePosition[0]}
              onChange={(v) => setProbePosition([v, probePosition[1]])}
              colors={colors}
            />
            <NumberField
              label="scan col"
              value={probePosition[1]}
              onChange={(v) => setProbePosition([probePosition[0], v])}
              colors={colors}
            />
          </Stack>
          )}

          <ImageCanvas
            bytes={probeDiffractionPng}
            shape={qShape}
            colors={colors}
            overlay={drawPreviewPeaks}
            caption="diffraction pattern"
          />
          <ImageCanvas
            bytes={probeCorrelationPng}
            shape={qShape}
            colors={colors}
            overlay={drawPreviewPeaks}
            caption="correlation map"
          />
        </Stack>
      </Panel>

      <Panel title="3. Detection" colors={colors}>
        {controlsVisible && (
        <Stack direction="row" spacing={1.5} flexWrap="wrap" useFlexGap>
          <NumberField label="min_abs_intensity" value={minAbsIntensity} step={0.1} onChange={setMinAbsIntensity} colors={colors} />
          <NumberField label="min_spacing" value={minSpacing} step={0.5} onChange={setMinSpacing} colors={colors} />
          <NumberField label="edge_boundary" value={edgeBoundary} onChange={setEdgeBoundary} colors={colors} />
          <NumberField label="upsample_factor" value={upsampleFactor} onChange={setUpsampleFactor} colors={colors} />
          <NumberField label="max_num_peaks" value={maxNumPeaks} onChange={setMaxNumPeaks} colors={colors} />
          <NumberField label="preview_grid" hint="preview samples an N x N grid" value={previewGrid} onChange={setPreviewGrid} colors={colors} />
          <Select
            size="small"
            value={subpixel}
            onChange={(e) => setSubpixel(e.target.value)}
            sx={{ width: 120, fontSize: 11, "& .MuiSelect-select": { py: 0.5, color: colors.text } }}
          >
            <MenuItem sx={{ fontSize: 11 }} value="none">none</MenuItem>
            <MenuItem sx={{ fontSize: 11 }} value="parabolic">parabolic</MenuItem>
            <MenuItem sx={{ fontSize: 11 }} value="upsample">upsample</MenuItem>
          </Select>
        </Stack>
        )}

        {controlsVisible && (
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: `${SPACING.MD}px` }}>
          <Button
            size="small"
            sx={{ ...compactButton, color: colors.accent }}
            disabled={busy}
            onClick={() => send({ type: "preview_detect" })}
          >
            Preview on grid
          </Button>
          <Button
            size="small"
            sx={{ ...compactButton, color: colors.accent }}
            disabled={busy}
            onClick={() => send({ type: "run_detect" })}
          >
            {detectionState === "running" ? "Detecting..." : "Run full detection"}
          </Button>
        </Stack>
        )}

        {preview && (
          <Typography sx={{ fontSize: 10, color: colors.textMuted, mt: `${SPACING.SM}px`, fontFamily: "monospace" }}>
            {preview.peaks.length} peaks at the probed position · across {preview.counts.length} sampled
            positions min {Math.min(...preview.counts)}, median {median(preview.counts)}, max{" "}
            {Math.max(...preview.counts)}. Marks are drawn on the probe panel.
          </Typography>
        )}
      </Panel>

      <Panel
        title="4. Bragg vector map"
        colors={colors}
        blocked={hasPeaks ? undefined : "Run full detection to accumulate the Bragg vector map."}
      >
        <Stack direction="row" spacing={2} alignItems="flex-start" flexWrap="wrap">
          {controlsVisible && (
          <Stack spacing={1.5} sx={{ minWidth: 200 }}>
            <NumberField label="bvm_sampling" value={bvmSampling} step={0.1} onChange={setBvmSampling} colors={colors} />
            <Button
              size="small"
              sx={{ ...compactButton, color: colors.accent, alignSelf: "flex-start" }}
              disabled={busy}
              onClick={() => send({ type: "compute_bvm" })}
            >
              Recompute map
            </Button>
          </Stack>
          )}

          <ImageCanvas
            bytes={bvmPng}
            shape={qShape}
            colors={colors}
            overlay={drawCandidates}
            caption={`${candidates.length} candidate peaks${numCandidates > 0 ? "" : " (auto)"}, brightest numbered; zoom in for more`}
          />
        </Stack>
      </Panel>

      <Panel
        title="5. Basis"
        colors={colors}
        blocked={hasPeaks ? undefined : "Run full detection before choosing basis vectors."}
      >
        <Stack direction="row" spacing={2} alignItems="flex-start" flexWrap="wrap">
          {controlsVisible && (
          <Stack spacing={1.5} sx={{ minWidth: 220 }}>
            <NumberField
              label="num_candidates"
              hint="0 picks the count from the busiest scan position"
              value={numCandidates}
              onChange={setNumCandidates}
              colors={colors}
            />
            <NumberField label="min_spacing" value={candidateMinSpacing} step={0.5} onChange={setCandidateMinSpacing} colors={colors} />
            <NumberField label="min_abs_intensity" value={candidateMinAbsIntensity} step={0.1} onChange={setCandidateMinAbsIntensity} colors={colors} />

            <Stack direction="row" spacing={0.5}>
              {BASIS_LABELS.map((role, i) => (
                <Button
                  key={role}
                  size="small"
                  sx={{
                    ...compactButton,
                    color: activeRole === role ? BASIS_COLORS[i] : colors.textMuted,
                    fontWeight: activeRole === role ? 700 : 400,
                  }}
                  onClick={() => setActiveRole(role)}
                >
                  {role}
                </Button>
              ))}
            </Stack>

            <Button
              size="small"
              sx={{ ...compactButton, color: colors.textMuted, alignSelf: "flex-start" }}
              onClick={() => {
                setOriginIndex(-1);
                setG1Index(-1);
                setG2Index(-1);
                setOriginRc([]);
                setG1Rc([]);
                setG2Rc([]);
              }}
            >
              Reset to automatic
            </Button>

            <Box sx={{ fontFamily: "monospace", fontSize: 11 }}>
              <Typography sx={{ fontSize: 11, fontFamily: "monospace", color: BASIS_COLORS[0] }}>
                origin {vectorText(originRc)}{originIndex >= 0 ? ` · candidate ${originIndex}` : ""}
              </Typography>
              <Typography sx={{ fontSize: 11, fontFamily: "monospace", color: BASIS_COLORS[1] }}>
                g1 {vectorText(g1Rc)}{g1Index >= 0 ? ` · candidate ${g1Index}` : ""}
              </Typography>
              <Typography sx={{ fontSize: 11, fontFamily: "monospace", color: BASIS_COLORS[2] }}>
                g2 {vectorText(g2Rc)}{g2Index >= 0 ? ` · candidate ${g2Index}` : ""}
              </Typography>
            </Box>
          </Stack>
          )}

          <ImageCanvas
            bytes={bvmPng}
            shape={qShape}
            colors={colors}
            overlay={drawCandidates}
            markers={basisMarkers}
            onMoveMarker={setRoleVector}
            onPick={pickCandidate}
            caption={`click a numbered candidate to set ${activeRole}, or drag a marker`}
          />
        </Stack>
      </Panel>

      <Panel
        title="6. Lattice fit"
        colors={colors}
        blocked={basisMarkers.length === 3 ? undefined : "Choose the basis vectors before fitting."}
      >
        {controlsVisible && (
        <Stack direction="row" spacing={1.5} alignItems="center" flexWrap="wrap" useFlexGap>
          <NumberField label="min_num_peaks" value={minNumPeaks} onChange={setMinNumPeaks} colors={colors} />
          <NumberField
            label="max_peak_shift"
            hint="inclusion radius in pixels; 0 uses half the shorter spacing"
            value={maxPeakShift}
            step={0.5}
            onChange={setMaxPeakShift}
            colors={colors}
          />
          <Button
            size="small"
            sx={{ ...compactButton, color: colors.accent }}
            disabled={busy}
            onClick={() => send({ type: "run_fit" })}
          >
            {fitting ? "Fitting..." : "Run fit"}
          </Button>
        </Stack>
        )}

        {fitState === "done" && (
          <Stack direction="row" spacing={2} sx={{ mt: `${SPACING.MD}px` }} flexWrap="wrap">
            <ImageCanvas bytes={maskWeightPng} shape={scanShape} colors={colors} caption="mask weight" />
            <ImageCanvas bytes={fitErrorPng} shape={scanShape} colors={colors} caption="fit error (px)" />
          </Stack>
        )}
      </Panel>
    </Box>
  );
}

export const render = createRender(ShowBragg);
