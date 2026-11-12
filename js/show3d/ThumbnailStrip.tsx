import * as React from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { DriftChart, type DriftReadout } from "./DriftChart";

/** CSS width of the strip. The thumbnails are square and share it: 1, 2 or 3 across. */
export const THUMBNAIL_STRIP_WIDTH = 264;
// thumbnails touch: no lines, no white padding, every pixel shows data
const GAP = 0;
const PAD = 0;
const INNER_WIDTH = THUMBNAIL_STRIP_WIDTH - 2; // inside the 1 px strip border
const HEADER_HEIGHT = 30;
// Thumbnails stay small on the GPU: at most this many device pixels across, whatever the screen density.
const MAX_DEVICE_PX = 320;

export type ThumbnailStripColors = { text: string; textMuted: string; border: string; accent: string; controlBg: string };

export type ThumbnailStripProps = {
  frames: number;
  currentFrame: number;
  labels?: string[];
  columns: number;
  onColumnsChange: (columns: number) => void;
  onHide: () => void;
  /** Show `frame` in the viewer while the pointer rests on it; null restores the selected frame. */
  onHover: (frame: number | null) => void;
  /** Move the frame slider to `frame`. */
  onSelect: (frame: number) => void;
  /** The viewer's height in CSS px: the strip is exactly as tall and scrolls inside. */
  height: number;
  /** Width / height of one thumbnail image (one panel); a non-square image is centred in its square cell. */
  imageAspect: number;
  /** Draw `frame` into `canvas` (device-pixel sized); false when it cannot yet (frame not resident). */
  draw: (canvas: HTMLCanvasElement, frame: number) => boolean;
  /** Free the GPU textures of a thumbnail scrolled far out of view; it is drawn again on return. */
  release: (canvas: HTMLCanvasElement) => void;
  /** Changes whenever every thumbnail must be drawn again (a new stack, colormap, scale). */
  drawKey: string;
  /** Changes when more frames become drawable (resident on the GPU): thumbnails not yet drawn try again. */
  retryKey: number | string;
  /** Why no thumbnail can be drawn (no WebGPU, a color stack); null when they can. */
  unavailableReason: string | null;
  colors: ThumbnailStripColors;
  /** Frames picked for Find drift (stack indices); ⌘/Ctrl-click toggles one, Shift-click picks a range. */
  chosen?: number[];
  onChoose?: (frames: number[]) => void;
  /** Runs Find drift on the picked frames (every frame when fewer than two are picked). */
  onFindDrift?: () => void;
  /** The button's words while it runs, or the last result's status. */
  driftStatus?: string;
  driftBusy?: boolean;
  /** The last Find drift result, drawn as a chart under the thumbnails. */
  drift?: DriftReadout | null;
};

/**
 * A scrollable strip of square frame thumbnails beside the viewer. Only thumbnails in or near view are
 * drawn, once per `drawKey`. Hover previews a frame, click selects it; the header sets 1, 2 or 3 across.
 */
export function ThumbnailStrip(props: ThumbnailStripProps) {
  const { frames, currentFrame, labels, columns, height, imageAspect, draw, drawKey, retryKey, colors } = props;
  const chosen = React.useMemo(() => new Set(props.chosen ?? []), [props.chosen]);
  const lastChosen = React.useRef<number | null>(null);
  /** ⌘/Ctrl-click toggles a frame for Find drift; Shift-click adds the range from the last one; a plain click selects. */
  const handleClick = (event: React.MouseEvent, frame: number) => {
    if (props.onChoose && (event.metaKey || event.ctrlKey)) {
      const next = new Set(chosen);
      if (next.has(frame)) next.delete(frame); else next.add(frame);
      lastChosen.current = frame;
      props.onChoose([...next].sort((a, b) => a - b));
      return;
    }
    if (props.onChoose && event.shiftKey) {
      const from = lastChosen.current ?? currentFrame;
      const next = new Set(chosen);
      for (let f = Math.min(from, frame); f <= Math.max(from, frame); f++) next.add(f);
      lastChosen.current = frame;
      props.onChoose([...next].sort((a, b) => a - b));
      return;
    }
    props.onSelect(frame);
  };
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const canvasRefs = React.useRef(new Map<number, HTMLCanvasElement>());
  const drawnKeys = React.useRef(new Map<number, string>());
  const [visible, setVisible] = React.useState<Set<number>>(() => new Set());
  const pointerInside = React.useRef(false);
  const cols = Math.max(1, Math.min(3, Math.round(columns) || 2));
  const cell = Math.floor((INNER_WIDTH - GAP * (cols - 1)) / cols);
  const aspect = Number.isFinite(imageAspect) && imageAspect > 0 ? imageAspect : 1;
  const imageWidth = aspect >= 1 ? cell : Math.round(cell * aspect);
  const imageHeight = aspect >= 1 ? Math.round(cell / aspect) : cell;
  const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
  const deviceScale = Math.min(dpr, MAX_DEVICE_PX / Math.max(1, imageWidth, imageHeight));
  const deviceWidth = Math.max(1, Math.round(imageWidth * deviceScale));
  const deviceHeight = Math.max(1, Math.round(imageHeight * deviceScale));
  const totalHeight = Math.max(160, Math.round(height));

  // A new size or draw key invalidates every drawn thumbnail.
  React.useEffect(() => { drawnKeys.current.clear(); }, [drawKey, deviceWidth, deviceHeight]);

  // Track which cells are in or near view; only those are drawn.
  React.useEffect(() => {
    const root = scrollRef.current;
    if (!root || typeof IntersectionObserver === "undefined") {
      setVisible(new Set(Array.from({ length: Math.min(frames, 60) }, (_, i) => i)));
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      setVisible(prev => {
        const next = new Set(prev);
        for (const entry of entries) {
          const frame = Number((entry.target as HTMLElement).dataset.frame);
          if (entry.isIntersecting) { next.add(frame); continue; }
          next.delete(frame);
          // far out of view: give its textures back to the GPU, draw it again when it returns
          const canvas = canvasRefs.current.get(frame);
          if (canvas && drawnKeys.current.has(frame)) { props.release(canvas); drawnKeys.current.delete(frame); }
        }
        return next;
      });
    }, { root, rootMargin: "240px 0px" });
    root.querySelectorAll<HTMLElement>("[data-frame]").forEach(el => observer.observe(el));
    return () => observer.disconnect();
  }, [frames, cols, props.unavailableReason]);

  // Draw the visible thumbnails that are missing or stale.
  React.useEffect(() => {
    if (props.unavailableReason) return;
    for (const frame of visible) {
      const canvas = canvasRefs.current.get(frame);
      if (!canvas || drawnKeys.current.get(frame) === drawKey) continue;
      if (canvas.width !== deviceWidth) canvas.width = deviceWidth;
      if (canvas.height !== deviceHeight) canvas.height = deviceHeight;
      if (draw(canvas, frame)) drawnKeys.current.set(frame, drawKey);
    }
  }, [visible, drawKey, retryKey, frames, deviceWidth, deviceHeight, draw, props.unavailableReason]);

  // Keep the selected frame in view (playback, slider) unless the pointer is browsing the strip.
  React.useEffect(() => {
    const root = scrollRef.current;
    if (pointerInside.current || !root) return;
    const el = root.querySelector<HTMLElement>(`[data-frame="${currentFrame}"]`);
    if (!el) return;
    const top = el.offsetTop - PAD;
    if (top < root.scrollTop || top + el.offsetHeight > root.scrollTop + root.clientHeight) {
      root.scrollTop = Math.max(0, top - (root.clientHeight - el.offsetHeight) / 2);
    }
  }, [currentFrame, cols]);

  const segment = (n: number) => ({
    width: 24, height: 22, display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 11, cursor: "pointer", userSelect: "none" as const,
    color: cols === n ? "#fff" : colors.text, bgcolor: cols === n ? colors.accent : "transparent",
    borderLeft: n > 1 ? `1px solid ${colors.border}` : "none",
  });

  return (
    <Box
      data-testid="show3d-thumbnail-strip"
      sx={{ width: THUMBNAIL_STRIP_WIDTH, height: totalHeight, flex: "0 0 auto", display: "flex", flexDirection: "column", border: `1px solid ${colors.border}`, boxSizing: "border-box" }}
      onPointerEnter={() => { pointerInside.current = true; }}
      onPointerLeave={() => { pointerInside.current = false; props.onHover(null); }}
    >
      <Box sx={{ height: HEADER_HEIGHT, flex: "0 0 auto", display: "flex", alignItems: "center", gap: 0.75, px: "6px", borderBottom: `1px solid ${colors.border}` }}>
        <Typography sx={{ fontSize: 11, fontWeight: 600, color: colors.text }}>Frames</Typography>
        <Typography sx={{ fontSize: 11, color: colors.textMuted, flex: 1 }}>{chosen.size ? `${chosen.size} of ${frames}` : frames}</Typography>
        {props.onFindDrift && (
          <Box role="button" aria-label="Find drift" data-testid="show3d-find-drift"
            title={"Find drift: register the picked frames (⌘/Ctrl-click, Shift-click; all when none) on the GPU, " +
              "to 1/100 px, and fit the drift velocity"}
            onClick={props.driftBusy ? undefined : props.onFindDrift}
            sx={{ px: "6px", height: 20, display: "flex", alignItems: "center", fontSize: 11, fontWeight: 600, whiteSpace: "nowrap",
              border: `1px solid ${colors.accent}`, color: props.driftBusy ? colors.textMuted : colors.accent,
              cursor: props.driftBusy ? "progress" : "pointer", "&:hover": { bgcolor: props.driftBusy ? "transparent" : colors.controlBg } }}>
            {props.driftBusy ? "Finding…" : "Find drift"}
          </Box>
        )}
        <Box role="group" aria-label="Thumbnails across" sx={{ display: "flex", border: `1px solid ${colors.border}` }}>
          {[1, 2, 3].map(n => (
            <Box key={n} role="button" aria-label={`${n} thumbnail column${n > 1 ? "s" : ""}`} aria-pressed={cols === n}
              title={`${n} across`} sx={segment(n)} onClick={() => props.onColumnsChange(n)}>{n}</Box>
          ))}
        </Box>
        <Box role="button" aria-label="Hide thumbnails" title="Hide thumbnails" onClick={props.onHide}
          sx={{ width: 22, height: 22, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, color: colors.textMuted, cursor: "pointer", "&:hover": { color: colors.text } }}>×</Box>
      </Box>
      {props.unavailableReason ? (
        <Typography sx={{ fontSize: 11, color: colors.textMuted, p: 1.25, lineHeight: 1.4 }}>{props.unavailableReason}</Typography>
      ) : (
        <Box ref={scrollRef} sx={{ flex: "1 1 auto", minHeight: 0, overflowY: "auto", overscrollBehavior: "contain", scrollbarWidth: "thin", position: "relative" }}>
          <Box sx={{
            // columns share the strip's last pixels, so no white sliver is left at the right edge
            display: "grid", gridTemplateColumns: `repeat(${cols}, 1fr)`, gridAutoRows: `${cell}px`, gap: `${GAP}px`,
            width: "100%",
          }}>
          {Array.from({ length: frames }, (_, frame) => {
            const selected = frame === currentFrame;
            // frames count from 1 as the slider does; a label that is only the 0-based index adds nothing
            const custom = labels?.[frame] && labels[frame] !== String(frame) ? labels[frame] : "";
            return (
              <Box key={frame} data-frame={frame} title={custom ? `Frame ${frame + 1}: ${custom}` : `Frame ${frame + 1}`}
                onPointerEnter={() => props.onHover(frame)}
                onClick={(event) => handleClick(event, frame)}
                sx={{
                  position: "relative", height: cell, cursor: "pointer", bgcolor: "#000",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  outline: selected ? `2px solid ${colors.accent}` : "none", outlineOffset: -2,
                  "&:hover": selected ? {} : { outline: "1px solid rgba(255,255,255,0.7)", outlineOffset: -1 },
                }}>
                {/* A 1fr column is a fraction of a pixel wider than the whole-pixel row: "contain" would letterbox that
                    fraction and show the cell background as a seam, so an image that fills its cell covers it. */}
                <canvas
                  ref={(el) => { if (el) canvasRefs.current.set(frame, el); else canvasRefs.current.delete(frame); }}
                  aria-label={`Frame ${frame + 1} thumbnail`}
                  style={{ display: "block", width: imageWidth >= cell ? "100%" : imageWidth, height: imageHeight >= cell ? "100%" : imageHeight, objectFit: imageWidth >= cell && imageHeight >= cell ? "cover" : "contain" }}
                />
                <Box sx={{
                  position: "absolute", left: 3, bottom: 3, px: 0.5, maxWidth: cell - 6,
                  fontSize: 10, lineHeight: "14px", color: "#fff", bgcolor: selected ? colors.accent : "rgba(0,0,0,0.55)",
                  overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", pointerEvents: "none",
                }}>
                  {cols < 3 && custom ? `${frame + 1} · ${custom}` : frame + 1}
                </Box>
                {chosen.has(frame) && (
                  <Box aria-label={`Frame ${frame + 1} picked for Find drift`} sx={{
                    position: "absolute", right: 3, top: 3, width: 16, height: 16, borderRadius: "50%", fontSize: 11,
                    lineHeight: "16px", textAlign: "center", color: "#fff", bgcolor: colors.accent, pointerEvents: "none",
                  }}>✓</Box>
                )}
              </Box>
            );
          })}
          </Box>
        </Box>
      )}
      {(props.drift || props.driftStatus) && (
        <DriftChart drift={props.drift ?? null} status={props.driftStatus ?? ""} width={INNER_WIDTH} colors={colors} />
      )}
    </Box>
  );
}
