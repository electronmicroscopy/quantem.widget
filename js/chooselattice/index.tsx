/**
 * ChooseLattice — pick an ordered origin + two lattice-vector points on an image.
 *
 * Lean single-panel viewer: one canvas showing a pre-rendered (server-side
 * colormapped) image. Scroll to zoom, drag to pan, click to place up to 3
 * ordered points, drag an existing point to adjust it. Points are reported
 * in ORIGINAL image pixel coordinates regardless of the current zoom/pan.
 */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Button from "@mui/material/Button";
import { useTheme } from "../theme";
import { extractBytes, preserveRestoredWidgetModelsOnSave } from "../format";
import { useHideStaticFallback } from "../staticFallback";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 20;
const CANVAS_SIZE = 512;
const CANVAS_BORDER_PX = 1;
const HIT_PX = 10;
const CLICK_MOVE_THRESHOLD_PX = 4;
const POINT_COLORS = ["#ff4d4f", "#40a9ff", "#73d13d"];

const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;
const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  textTransform: "none" as const,
};

type Point = [number, number]; // [row, col] in original image pixels
type DragMode = "none" | "pan" | "point";

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

function ChooseLattice() {
  const model = useModel();
  const rootRef = React.useRef<HTMLDivElement>(null);
  const { colors: themeColors } = useTheme();

  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);
  useHideStaticFallback(model, rootRef);

  const [height] = useModelState<number>("height");
  const [width] = useModelState<number>("width");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [title] = useModelState<string>("title");
  const [pointLabels] = useModelState<string[]>("point_labels");
  const [points, setPoints] = useModelState<Point[]>("points");

  // Decode the PNG payload once per change into a drawable bitmap.
  const [image, setImage] = React.useState<ImageBitmap | HTMLImageElement | null>(null);
  React.useEffect(() => {
    const bytes = extractBytes(frameBytes);
    if (bytes.length === 0) {
      setImage(null);
      return;
    }
    let cancelled = false;
    const blob = new Blob([bytes as unknown as BlobPart], { type: "image/png" });
    if (typeof createImageBitmap === "function") {
      createImageBitmap(blob).then((bmp) => { if (!cancelled) setImage(bmp); });
    } else {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => { if (!cancelled) setImage(img); URL.revokeObjectURL(url); };
      img.src = url;
    }
    return () => { cancelled = true; };
  }, [frameBytes]);

  // View state: zoom + pan (CSS px, canvas-centered).
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);

  // displayScale maps original image pixels -> CSS px at zoom=1.
  const displayScale = height > 0 && width > 0
    ? CANVAS_SIZE / Math.max(height, width)
    : 1;
  const canvasW = CANVAS_SIZE;
  const canvasH = CANVAS_SIZE;

  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const uiRef = React.useRef<HTMLCanvasElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Block page scroll on wheel-zoom with a native non-passive listener;
  // React's synthetic onWheel is passive and would only warn if it called
  // preventDefault itself.
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const prevent = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", prevent, { passive: false });
    return () => el.removeEventListener("wheel", prevent);
  }, []);

  // Draw the base image with pan/zoom applied.
  React.useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasW;
    canvas.height = canvasH;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = themeColors.bg;
    ctx.fillRect(0, 0, canvasW, canvasH);
    if (!image || !width || !height) return;
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const drawW = width * displayScale * zoom;
    const drawH = height * displayScale * zoom;
    const x = cx - drawW / 2 + panX;
    const y = cy - drawH / 2 + panY;
    ctx.drawImage(image, x, y, drawW, drawH);
  }, [image, width, height, displayScale, zoom, panX, panY, canvasW, canvasH, themeColors.bg]);

  // Convert a mouse event to original-image (row, col) coordinates.
  const screenToImg = React.useCallback((e: { clientX: number; clientY: number }): Point => {
    const canvas = canvasRef.current;
    if (!canvas) return [0, 0];
    const rect = canvas.getBoundingClientRect();
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const col = (mouseCanvasX - cx - panX) / (displayScale * zoom) + width / 2;
    const row = (mouseCanvasY - cy - panY) / (displayScale * zoom) + height / 2;
    return [row, col];
  }, [canvasW, canvasH, panX, panY, displayScale, zoom, width, height]);

  const imgToScreen = React.useCallback((row: number, col: number): [number, number] => {
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const x = cx + (col - width / 2) * displayScale * zoom + panX;
    const y = cy + (row - height / 2) * displayScale * zoom + panY;
    return [x, y];
  }, [canvasW, canvasH, panX, panY, displayScale, zoom, width, height]);

  const hitTestPoint = React.useCallback((row: number, col: number): number => {
    const hitArea = HIT_PX / (displayScale * zoom);
    const list = points || [];
    for (let i = list.length - 1; i >= 0; i--) {
      const [pr, pc] = list[i];
      if (Math.hypot(row - pr, col - pc) <= hitArea) return i;
    }
    return -1;
  }, [points, displayScale, zoom]);

  // Wheel: cursor-anchored zoom. Page-scroll prevention is handled by a
  // native non-passive listener below (React's synthetic onWheel is passive,
  // so calling preventDefault directly here would only log a console warning).
  const handleWheel = (e: React.WheelEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const mouseImageX = (mouseCanvasX - cx - panX) / zoom + cx;
    const mouseImageY = (mouseCanvasY - cy - panY) / zoom + cy;
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = clamp(zoom * zoomFactor, MIN_ZOOM, MAX_ZOOM);
    setPanX(mouseCanvasX - (mouseImageX - cx) * newZoom - cx);
    setPanY(mouseCanvasY - (mouseImageY - cy) * newZoom - cy);
    setZoom(newZoom);
  };

  const resetView = React.useCallback(() => {
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, []);

  // A click's mousedown/mouseup fire BEFORE the browser knows whether a
  // second click will follow (making it a double-click). Placing a point
  // immediately on every plain click would mean double-clicking to reset the
  // view always drops a spurious point first. So a plain click's point is
  // held for this long — if a double-click follows, it is cancelled instead.
  const DOUBLE_CLICK_GRACE_MS = 300;
  const pointsRef = React.useRef(points);
  React.useEffect(() => { pointsRef.current = points; }, [points]);
  const pendingPointTimeoutRef = React.useRef<number | null>(null);
  const cancelPendingPoint = React.useCallback(() => {
    if (pendingPointTimeoutRef.current !== null) {
      window.clearTimeout(pendingPointTimeoutRef.current);
      pendingPointTimeoutRef.current = null;
    }
  }, []);
  React.useEffect(() => cancelPendingPoint, [cancelPendingPoint]);

  // Mouse drag: pan the view, drag an existing point, or (on a plain click)
  // place the next point in order.
  const dragRef = React.useRef<{
    mode: DragMode;
    startClientX: number;
    startClientY: number;
    startPanX: number;
    startPanY: number;
    pointIndex: number;
  } | null>(null);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.detail >= 2) {
      // Second (and later) click of a double-click: cancel any point the
      // first click was about to place, and let onDoubleClick reset the view.
      cancelPendingPoint();
      dragRef.current = null;
      return;
    }
    const [row, col] = screenToImg(e);
    const hitIdx = hitTestPoint(row, col);
    if (hitIdx !== -1) {
      dragRef.current = {
        mode: "point", startClientX: e.clientX, startClientY: e.clientY,
        startPanX: panX, startPanY: panY, pointIndex: hitIdx,
      };
      return;
    }
    dragRef.current = {
      mode: "none", startClientX: e.clientX, startClientY: e.clientY,
      startPanX: panX, startPanY: panY, pointIndex: -1,
    };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const drag = dragRef.current;
    if (!drag) return;
    if (drag.mode === "point") {
      const [row, col] = screenToImg(e);
      const next = (points || []).slice();
      next[drag.pointIndex] = [clamp(row, 0, Math.max(0, height - 1)), clamp(col, 0, Math.max(0, width - 1))];
      setPoints(next);
      return;
    }
    const moved = Math.hypot(e.clientX - drag.startClientX, e.clientY - drag.startClientY);
    if (drag.mode === "none" && moved > CLICK_MOVE_THRESHOLD_PX) {
      dragRef.current = { ...drag, mode: "pan" };
    }
    if (dragRef.current?.mode === "pan") {
      setPanX(drag.startPanX + (e.clientX - drag.startClientX));
      setPanY(drag.startPanY + (e.clientY - drag.startClientY));
    }
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag) return;
    if (drag.mode === "none") {
      // Plain click (no drag past the threshold): place the next point,
      // after a short grace period a following double-click can cancel.
      const list = pointsRef.current || [];
      if (list.length < 3) {
        const [row, col] = screenToImg(e);
        const clamped: Point = [
          clamp(row, 0, Math.max(0, height - 1)),
          clamp(col, 0, Math.max(0, width - 1)),
        ];
        cancelPendingPoint();
        pendingPointTimeoutRef.current = window.setTimeout(() => {
          pendingPointTimeoutRef.current = null;
          setPoints([...(pointsRef.current || []), clamped]);
        }, DOUBLE_CLICK_GRACE_MS);
      }
    }
  };

  const handleDoubleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    cancelPendingPoint();
    resetView();
  };

  // Cursor readout while hovering (not dragging).
  const [cursorPos, setCursorPos] = React.useState<Point | null>(null);
  const handleMouseMoveReadout = (e: React.MouseEvent) => {
    handleMouseMove(e);
    if (!dragRef.current || dragRef.current.mode === "none") {
      setCursorPos(screenToImg(e));
    }
  };

  // Overlay: point markers + guide lines from Origin -> u and Origin -> v.
  React.useLayoutEffect(() => {
    const canvas = uiRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasW;
    canvas.height = canvasH;
    ctx.clearRect(0, 0, canvasW, canvasH);
    const list = points || [];
    if (list.length > 1) {
      const [ox, oy] = imgToScreen(list[0][0], list[0][1]);
      ctx.strokeStyle = "rgba(255,255,255,0.6)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      for (let i = 1; i < list.length; i++) {
        const [x, y] = imgToScreen(list[i][0], list[i][1]);
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    list.forEach(([row, col], i) => {
      const [x, y] = imgToScreen(row, col);
      const color = POINT_COLORS[i % POINT_COLORS.length];
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.stroke();
      const label = pointLabels && pointLabels[i] ? pointLabels[i] : String(i + 1);
      ctx.font = "bold 11px -apple-system, sans-serif";
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = "rgba(0,0,0,0.85)";
      ctx.lineWidth = 3;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.strokeText(label, x + 9, y - 6);
      ctx.fillText(label, x + 9, y - 6);
    });
  }, [points, pointLabels, imgToScreen, canvasW, canvasH]);

  const canvasBox = {
    position: "relative" as const,
    border: `${CANVAS_BORDER_PX}px solid ${themeColors.border}`,
    overflow: "hidden",
    width: canvasW,
    height: canvasH,
  };
  // The bordered canvas is a fixed pixel width, so the CONTENT column (title
  // row, canvas, footer, readout) is capped to that width and everything
  // aligns to the canvas's own edges. The background is a separate, always
  // full-width wrapper: capping IT to the content width would leave the
  // notebook/page background exposed beside the widget whenever the output
  // cell is wider than the canvas.
  const contentMaxWidth = canvasW + 2 * CANVAS_BORDER_PX;

  return (
    <Box
      ref={rootRef}
      sx={{
        p: `${SPACING.LG}px`,
        bgcolor: themeColors.bg,
        color: themeColors.text,
        width: "100%",
        boxSizing: "border-box",
      }}
    >
      <Box sx={{ maxWidth: contentMaxWidth }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.SM}px` }}>
          <Typography sx={{ fontSize: 13, fontWeight: 600 }}>{title || "Choose Lattice"}</Typography>
          <Stack direction="row" spacing={1}>
            <Button
              size="small"
              sx={{ ...compactButton, color: themeColors.accent }}
              disabled={!points || points.length === 0}
              onClick={() => setPoints([])}
            >
              Clear Points
            </Button>
          </Stack>
        </Stack>

        <Box ref={containerRef} sx={canvasBox}>
          <canvas
            ref={canvasRef}
            style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, imageRendering: "pixelated" }}
          />
          <canvas
            ref={uiRef}
            style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
          />
          <canvas
            width={canvasW}
            height={canvasH}
            style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, cursor: "crosshair", opacity: 0 }}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMoveReadout}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => { dragRef.current = null; setCursorPos(null); }}
            onDoubleClick={handleDoubleClick}
          />
        </Box>

        <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mt: `${SPACING.XS}px` }}>
          {(points || []).length < 3
            ? "Click to place the next point. Scroll to zoom, drag to pan."
            : "Drag a point to adjust it. Scroll to zoom, drag to pan."}
          {cursorPos && (
            <span style={{ marginLeft: 8, color: themeColors.accent }}>
              ({cursorPos[0].toFixed(1)}, {cursorPos[1].toFixed(1)})
            </span>
          )}
        </Typography>

        <Box sx={{ mt: `${SPACING.SM}px` }}>
          {(pointLabels || []).map((label, i) => {
            const p = (points || [])[i];
            const origin = (points || [])[0];
            // Origin is reported as its raw pixel position; the other two
            // points are reported as lattice vectors relative to the origin
            // (u = a1 - origin, v = a2 - origin), not raw pixel positions.
            const isVector = i > 0;
            const value = isVector && p && origin
              ? [p[0] - origin[0], p[1] - origin[1]]
              : (!isVector ? p : null);
            return (
              <Typography key={label + i} sx={{ fontSize: 11, fontFamily: "monospace", color: value ? POINT_COLORS[i % POINT_COLORS.length] : themeColors.textMuted }}>
                {label}: {value ? `(${value[0].toFixed(1)}, ${value[1].toFixed(1)})` : "not placed"}
              </Typography>
            );
          })}
        </Box>
      </Box>
    </Box>
  );
}

export const render = createRender(ChooseLattice);