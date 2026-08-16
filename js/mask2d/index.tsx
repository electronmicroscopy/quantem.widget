import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";

import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen } from "../colormaps";
import { extractBytes, extractFloat32 } from "../format";
import { findDataRange, percentileClip } from "../stats";
import { useTheme } from "../theme";

type Shape = "rectangle" | "square" | "circle";
type Point = { row: number; col: number };
type Region = {
  row: number;
  col: number;
  shape: Shape;
  radius: number;
  radius_inner: number;
  width: number;
  height: number;
  color: string;
  line_width: number;
  highlight: boolean;
  visible: boolean;
};

const clamp = (value: number, low: number, high: number) => (
  Math.max(low, Math.min(high, value))
);

function regionFromDrag(start: Point, end: Point, shape: Shape): Region {
  const deltaRow = Math.abs(end.row - start.row);
  const deltaCol = Math.abs(end.col - start.col);
  const radius = shape === "circle"
    ? Math.max(0.5, Math.hypot(deltaRow, deltaCol) / 2)
    : Math.max(0.5, Math.max(deltaRow, deltaCol) / 2);
  return {
    row: (start.row + end.row) / 2,
    col: (start.col + end.col) / 2,
    shape,
    radius,
    radius_inner: radius / 2,
    width: Math.max(1, deltaCol),
    height: Math.max(1, deltaRow),
    color: "#00b8ff",
    line_width: 2,
    highlight: true,
    visible: true,
  };
}

function Mask2D() {
  const { colors } = useTheme(false);
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [offline] = useModelState<boolean>("offline");
  const [offlineMin] = useModelState<number>("_offline_min");
  const [offlineMax] = useModelState<number>("_offline_max");
  const [title] = useModelState<string>("title");
  const [showTitle] = useModelState<boolean>("show_title");
  const [size] = useModelState<number>("size");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [vmin, setVmin] = useModelState<number | null>("vmin");
  const [vmax, setVmax] = useModelState<number | null>("vmax");
  const [shape, setShape] = useModelState<Shape>("mask_shape");
  const [regions, setRegions] = useModelState<Region[]>("roi_list");
  const [, setSelectedRegion] = useModelState<number>("roi_selected_idx");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [pixelUnit] = useModelState<string>("pixel_unit");

  const expectedLength = Math.max(0, width * height);
  const data = React.useMemo(() => {
    if (!frameBytes || expectedLength === 0) return new Float32Array(0);
    if (!offline) {
      return extractFloat32(frameBytes, expectedLength) ?? new Float32Array(0);
    }
    const encoded = extractBytes(frameBytes).subarray(0, expectedLength);
    const decoded = new Float32Array(encoded.length);
    const scale = (offlineMax - offlineMin) / 255;
    for (let index = 0; index < encoded.length; index++) {
      decoded[index] = offlineMin + encoded[index] * scale;
    }
    return decoded;
  }, [expectedLength, frameBytes, offline, offlineMax, offlineMin]);

  const dataRange = React.useMemo(() => findDataRange(data), [data]);
  const automaticRange = React.useMemo(
    () => percentileClip(data, 1, 99),
    [data],
  );
  const range = React.useMemo(() => {
    if (autoContrast) {
      return { low: automaticRange.vmin, high: automaticRange.vmax };
    }
    const low = Number.isFinite(vmin) ? Number(vmin) : dataRange.min;
    const high = Number.isFinite(vmax) ? Number(vmax) : dataRange.max;
    return { low, high: high > low ? high : low + 1 };
  }, [autoContrast, automaticRange, dataRange, vmax, vmin]);
  const toPercent = React.useCallback((value: number) => {
    const span = dataRange.max - dataRange.min;
    return span > 0 ? 100 * (value - dataRange.min) / span : 0;
  }, [dataRange]);
  const [contrast, setContrast] = React.useState<[number, number]>([0, 100]);
  React.useEffect(() => {
    setContrast([
      clamp(toPercent(range.low), 0, 100),
      clamp(toPercent(range.high), 0, 100),
    ]);
  }, [range.high, range.low, toPercent]);
  const visibleRange = React.useMemo(() => {
    const span = dataRange.max - dataRange.min;
    const low = dataRange.min + span * contrast[0] / 100;
    const high = dataRange.min + span * contrast[1] / 100;
    return { low, high: high > low ? high : low + 1 };
  }, [contrast, dataRange]);

  const maxSide = Math.max(280, size || 640);
  const displayWidth = width >= height
    ? maxSide
    : Math.max(240, Math.round(maxSide * width / Math.max(1, height)));
  const displayHeight = height >= width
    ? maxSide
    : Math.max(180, Math.round(maxSide * height / Math.max(1, width)));
  const baseCanvas = React.useRef<HTMLCanvasElement | null>(null);
  const overlayCanvas = React.useRef<HTMLCanvasElement | null>(null);
  const [zoom, setZoom] = React.useState(1);
  const [pan, setPan] = React.useState({ x: 0, y: 0 });
  const [draft, setDraft] = React.useState<Region | null>(null);
  const drag = React.useRef<{
    kind: "select" | "pan";
    start?: Point;
    clientX?: number;
    clientY?: number;
    panX?: number;
    panY?: number;
  } | null>(null);
  const pendingDraft = React.useRef<Region | null>(null);
  const draftFrame = React.useRef(0);

  const offscreen = React.useMemo(() => {
    if (data.length !== expectedLength || expectedLength === 0) return null;
    return renderToOffscreen(
      data,
      width,
      height,
      COLORMAPS[cmap] ?? COLORMAPS.gray,
      visibleRange.low,
      visibleRange.high,
    );
  }, [cmap, data, expectedLength, height, visibleRange.high, visibleRange.low, width]);

  const imageToDisplay = React.useCallback((point: Point) => ({
    x: ((point.col / Math.max(1, width)) * displayWidth - displayWidth / 2) * zoom
      + displayWidth / 2 + pan.x,
    y: ((point.row / Math.max(1, height)) * displayHeight - displayHeight / 2) * zoom
      + displayHeight / 2 + pan.y,
  }), [displayHeight, displayWidth, height, pan, width, zoom]);

  const pointerToImage = React.useCallback((clientX: number, clientY: number): Point => {
    const rect = overlayCanvas.current?.getBoundingClientRect();
    if (!rect) return { row: 0, col: 0 };
    const x = (clientX - rect.left) * displayWidth / Math.max(1, rect.width);
    const y = (clientY - rect.top) * displayHeight / Math.max(1, rect.height);
    const unscaledX = (x - displayWidth / 2 - pan.x) / zoom + displayWidth / 2;
    const unscaledY = (y - displayHeight / 2 - pan.y) / zoom + displayHeight / 2;
    return {
      row: clamp(unscaledY * height / displayHeight, 0, Math.max(0, height - 1)),
      col: clamp(unscaledX * width / displayWidth, 0, Math.max(0, width - 1)),
    };
  }, [displayHeight, displayWidth, height, pan, width, zoom]);

  React.useEffect(() => {
    const canvas = baseCanvas.current;
    if (!canvas || !offscreen) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(displayWidth * dpr);
    canvas.height = Math.round(displayHeight * dpr);
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, displayWidth, displayHeight);
    context.save();
    context.translate(displayWidth / 2 + pan.x, displayHeight / 2 + pan.y);
    context.scale(zoom, zoom);
    context.translate(-displayWidth / 2, -displayHeight / 2);
    context.imageSmoothingEnabled = true;
    context.drawImage(offscreen, 0, 0, displayWidth, displayHeight);
    context.restore();
  }, [displayHeight, displayWidth, offscreen, pan, zoom]);

  React.useEffect(() => {
    const canvas = overlayCanvas.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(displayWidth * dpr);
    canvas.height = Math.round(displayHeight * dpr);
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, displayWidth, displayHeight);
    const region = draft ?? regions?.[0];
    if (!region || region.visible === false) return;
    const center = imageToDisplay({ row: region.row, col: region.col });
    const scaleX = displayWidth * zoom / Math.max(1, width);
    const scaleY = displayHeight * zoom / Math.max(1, height);
    const radius = region.radius * (scaleX + scaleY) / 2;
    const regionWidth = region.width * scaleX;
    const regionHeight = region.height * scaleY;
    context.beginPath();
    context.rect(0, 0, displayWidth, displayHeight);
    if (region.shape === "circle") {
      context.moveTo(center.x + radius, center.y);
      context.arc(center.x, center.y, radius, 0, 2 * Math.PI);
    } else {
      const side = 2 * radius;
      const drawWidth = region.shape === "square" ? side : regionWidth;
      const drawHeight = region.shape === "square" ? side : regionHeight;
      context.rect(center.x - drawWidth / 2, center.y - drawHeight / 2, drawWidth, drawHeight);
    }
    context.fillStyle = "rgba(0, 0, 0, 0.38)";
    context.fill("evenodd");
    context.beginPath();
    if (region.shape === "circle") {
      context.arc(center.x, center.y, radius, 0, 2 * Math.PI);
    } else {
      const side = 2 * radius;
      const drawWidth = region.shape === "square" ? side : regionWidth;
      const drawHeight = region.shape === "square" ? side : regionHeight;
      context.rect(center.x - drawWidth / 2, center.y - drawHeight / 2, drawWidth, drawHeight);
    }
    context.strokeStyle = region.color || colors.accent;
    context.lineWidth = 2;
    context.setLineDash(draft ? [6, 4] : []);
    context.stroke();
  }, [colors.accent, displayHeight, displayWidth, draft, height, imageToDisplay, regions, width, zoom]);

  React.useEffect(() => () => {
    if (draftFrame.current) cancelAnimationFrame(draftFrame.current);
  }, []);

  const scheduleDraft = React.useCallback((region: Region) => {
    pendingDraft.current = region;
    if (draftFrame.current) return;
    draftFrame.current = requestAnimationFrame(() => {
      draftFrame.current = 0;
      setDraft(pendingDraft.current);
    });
  }, []);

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    if (event.shiftKey || event.button === 1) {
      drag.current = {
        kind: "pan",
        clientX: event.clientX,
        clientY: event.clientY,
        panX: pan.x,
        panY: pan.y,
      };
    } else {
      const start = pointerToImage(event.clientX, event.clientY);
      drag.current = { kind: "select", start };
      scheduleDraft(regionFromDrag(start, start, shape));
    }
    event.preventDefault();
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const active = drag.current;
    if (!active) return;
    if (active.kind === "pan") {
      setPan({
        x: Number(active.panX) + event.clientX - Number(active.clientX),
        y: Number(active.panY) + event.clientY - Number(active.clientY),
      });
    } else if (active.start) {
      scheduleDraft(regionFromDrag(
        active.start,
        pointerToImage(event.clientX, event.clientY),
        shape,
      ));
    }
    event.preventDefault();
  };

  const finishPointer = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (drag.current?.kind === "select") {
      const committed = pendingDraft.current ?? draft;
      if (committed) {
        setRegions([committed]);
        setSelectedRegion(0);
      }
      pendingDraft.current = null;
      setDraft(null);
    }
    drag.current = null;
    event.preventDefault();
  };

  const handleWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const nextZoom = clamp(zoom * Math.exp(-event.deltaY * 0.0015), 1, 20);
    if (nextZoom === zoom) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) * displayWidth / Math.max(1, rect.width);
    const y = (event.clientY - rect.top) * displayHeight / Math.max(1, rect.height);
    const ratio = nextZoom / zoom;
    setPan({
      x: x - displayWidth / 2 - ratio * (x - displayWidth / 2 - pan.x),
      y: y - displayHeight / 2 - ratio * (y - displayHeight / 2 - pan.y),
    });
    setZoom(nextZoom);
  };

  const commitContrast = (_event: Event | React.SyntheticEvent, value: number | number[]) => {
    const values = value as number[];
    const span = dataRange.max - dataRange.min;
    setAutoContrast(false);
    setVmin(dataRange.min + span * values[0] / 100);
    setVmax(dataRange.min + span * values[1] / 100);
  };

  const selected = regions?.[0];
  return (
    <Box sx={{ display: "inline-flex", flexDirection: "column", gap: 0.75, color: colors.text }}>
      {showTitle && title && (
        <Typography sx={{ fontSize: 13, fontWeight: 600 }}>{title}</Typography>
      )}
      <Box sx={{
        display: "flex", alignItems: "center", flexWrap: "wrap", gap: 0.75,
        px: 1, py: 0.65, border: `1px solid ${colors.border}`, bgcolor: colors.controlBg,
      }}>
        <Typography sx={{ fontSize: 11 }}>Shape</Typography>
        <Select
          size="small"
          value={shape}
          onChange={(event) => setShape(event.target.value as Shape)}
          inputProps={{ "aria-label": "Region shape" }}
          sx={{ minWidth: 100, height: 28, fontSize: 11, color: colors.text }}
        >
          <MenuItem value="rectangle">Rectangle</MenuItem>
          <MenuItem value="square">Square</MenuItem>
          <MenuItem value="circle">Circle</MenuItem>
        </Select>
        <Typography sx={{ fontSize: 11 }}>Color</Typography>
        <Select
          size="small"
          value={cmap}
          onChange={(event) => setCmap(String(event.target.value))}
          inputProps={{ "aria-label": "Image colormap" }}
          sx={{ minWidth: 92, height: 28, fontSize: 11, color: colors.text }}
        >
          {COLORMAP_NAMES.map((name) => <MenuItem key={name} value={name}>{name}</MenuItem>)}
        </Select>
        <Button
          size="small"
          variant={autoContrast ? "contained" : "outlined"}
          onClick={() => setAutoContrast(true)}
          sx={{ minWidth: 48, height: 28, fontSize: 10, textTransform: "none" }}
        >
          Auto
        </Button>
        <Button
          size="small"
          onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
          sx={{ height: 28, fontSize: 10, textTransform: "none" }}
        >
          Reset View
        </Button>
        <Button
          size="small"
          disabled={!regions?.length}
          onClick={() => { setRegions([]); setSelectedRegion(-1); }}
          sx={{ height: 28, fontSize: 10, color: "#ef5350", textTransform: "none" }}
        >
          Clear
        </Button>
      </Box>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 0.5 }}>
        <Typography sx={{ fontSize: 10, color: colors.textMuted }}>Contrast</Typography>
        <Slider
          aria-label="Display contrast"
          value={contrast}
          min={0}
          max={100}
          size="small"
          onChange={(_event, value) => { setContrast(value as [number, number]); setAutoContrast(false); }}
          onChangeCommitted={commitContrast}
          sx={{ width: Math.min(260, displayWidth * 0.45) }}
        />
        <Typography sx={{ fontSize: 10, color: colors.textMuted }}>
          {visibleRange.low.toPrecision(3)}–{visibleRange.high.toPrecision(3)}
        </Typography>
      </Box>
      <Box sx={{
        position: "relative", width: displayWidth, height: displayHeight,
        maxWidth: "100%", border: `1px solid ${colors.border}`, overflow: "hidden",
        bgcolor: colors.bgAlt,
      }}>
        <canvas
          ref={baseCanvas}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
        />
        <canvas
          ref={overlayCanvas}
          aria-label="Mask2D image"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={finishPointer}
          onPointerCancel={finishPointer}
          onWheel={handleWheel}
          onDoubleClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
          style={{
            position: "absolute", inset: 0, width: "100%", height: "100%",
            cursor: drag.current?.kind === "pan" ? "grabbing" : "crosshair",
            touchAction: "none",
          }}
        />
      </Box>
      <Typography sx={{ fontSize: 10.5, color: colors.textMuted }}>
        Drag to replace the selection · scroll to zoom · Shift-drag to pan
        {selected ? ` · ${selected.shape} center (${selected.row.toFixed(1)}, ${selected.col.toFixed(1)})` : ""}
        {pixelSize > 0 ? ` · ${pixelSize} ${pixelUnit}/px` : ""}
      </Typography>
    </Box>
  );
}

export default { render: createRender(Mask2D) };
