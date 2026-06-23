/** Scale bar utilities shared across pages.
 *
 * Mimics `quantem.widget` show2d scale bar algorithm
 * (`js/scalebar.ts:roundToNiceValue` + `formatScaleLabel`) so dashboard
 * matches the widget output users see in Jupyter notebooks: nice 1 / 2 / 5
 * × 10^N values, automatic Å→nm demotion when bar >= 10 Å, and a
 * resolution-agnostic decade picker that does NOT rely on a hardcoded
 * series.
 *
 * Visual spec (widget show2d):
 *   - 5 px white bar with drop shadow (rgba 0,0,0,0.5; blur 2; +1,+1).
 *   - Single centered label above the bar.
 *   - 16 px bold sans, label "{nice} Å" or "{nice/10} nm" only.
 *   - 12 px inset from bottom-right corner.
 *   - Optional zoom indicator "{zoom.toFixed(1)}×" at bottom-left.
 *   - Resolution-agnostic: the physical label is recomputed from the
 *     current effective zoom, keeping the on-screen bar near 60 px.
 */

/**
 * Target on-screen bar length in CSS pixels. Matches quantem.widget
 * `drawScaleBarHiDPI` (`targetBarPx = 60`) so zooming keeps the bar near
 * this display length while the nice physical label changes dynamically.
 */
export const SCALE_BAR_TARGET_PX = 60;

/** Pick a "nice" length close to `targetUnits` using the quantem.widget
 * show2d decade rule:
 *   normalized in [1, 1.5)   -> 1   * 10^exp
 *   normalized in [1.5, 3.5) -> 2   * 10^exp
 *   normalized in [3.5, 7.5) -> 5   * 10^exp
 *   normalized in [7.5, 10)  -> 10  * 10^exp
 *
 * `targetUnits` and the return value are in the same arbitrary unit.
 */
export function pickNiceLength(targetUnits: number): number {
  if (!Number.isFinite(targetUnits) || targetUnits <= 0) return 0;
  const exp = Math.floor(Math.log10(targetUnits));
  const magnitude = Math.pow(10, exp);
  const normalized = targetUnits / magnitude;
  if (normalized < 1.5) return magnitude;
  if (normalized < 3.5) return 2 * magnitude;
  if (normalized < 7.5) return 5 * magnitude;
  return 10 * magnitude;
}

/** Pick a round scale bar length (nm) that renders close to
 * SCALE_BAR_TARGET_PX on screen at the given zoom. Returns null when
 * calibration is missing.
 *
 * `scanPx` = width of the image in pixels (at zoom=1).
 * `stepA` = real-space size of one pixel in Angstroms.
 * `zoom` = the current on-screen zoom factor.
 */
export function pickScaleBar(
  scanPx: number | undefined,
  stepA: number | null,
  zoom: number = 1,
): {
  lengthNm: number;
  fractionOfFov: number;
  fractionOfFovZoomed: number;
} | null {
  if (!scanPx || !stepA) return null;
  const fovNm = (scanPx * stepA) / 10;
  const visibleFovNm = fovNm / zoom;
  const targetNm = visibleFovNm * (SCALE_BAR_TARGET_PX / 800);
  const lengthNm = pickNiceLength(targetNm);
  if (lengthNm <= 0) return null;
  const fractionOfFov = lengthNm / fovNm;
  return {
    lengthNm,
    fractionOfFov,
    fractionOfFovZoomed: fractionOfFov * zoom,
  };
}

/** Pick a scale bar sized to an explicit display width in pixels.
 * Prefer this in canvas draw paths so the on-screen bar follows
 * quantem.widget Show2D: compute the nice physical length from the current
 * effective zoom and keep the displayed bar near SCALE_BAR_TARGET_PX.
 */
export function pickScaleBarPx(
  scanPx: number | undefined,
  stepA: number | null,
  displayPx: number,
  _opts: { targetDisplayPx?: number | null } = {},
): { lengthNm: number; barPx: number } | null {
  if (!scanPx || !stepA || !displayPx) return null;
  const fovNm = (scanPx * stepA) / 10;
  const pxPerNm = displayPx / fovNm;
  if (!Number.isFinite(pxPerNm) || pxPerNm <= 0) return null;
  const targetNm = SCALE_BAR_TARGET_PX / pxPerNm;
  const lengthNm = pickNiceLength(targetNm);
  if (lengthNm <= 0) return null;
  return { lengthNm, barPx: lengthNm * pxPerNm };
}

/** Format a label using the quantem.widget show2d rule for the "Å" unit:
 *   length_Å >= 10 -> "{round(Å/10)} nm"
 *   length_Å >= 1  -> "{round(Å)} Å"
 *   else           -> "{Å.toFixed(2)} Å"
 *
 * Input is in nm (dashboard's internal convention); converted internally.
 * Sub-Å lengths surface as 2-decimal Å (matches widget) instead of pm,
 * unlike quantem.core which auto-demotes to pm < 0.1 Å.
 */
export function scaleBarLabel(lengthNm: number): string {
  const lengthA = lengthNm * 10;
  if (lengthA >= 10) return `${Math.round(lengthA / 10)} nm`;
  if (lengthA >= 1) return `${Math.round(lengthA)} Å`;
  return `${lengthA.toFixed(2)} Å`;
}

/** Draw a publication-quality scale bar matching quantem.widget show2d.
 *
 * Spec: 5 px white bar with rgba(0,0,0,0.5) drop shadow (blur 2, +1+1),
 * single label centered above the bar in 16 px bold sans, 12 px inset
 * from the bottom-right corner.
 *
 * `displayPx` is the on-screen width (in CSS pixels) currently occupied
 * by the image's `scanPx` pixels. It can include viewer zoom and is used
 * for physical scale length.
 *
 * `styleDisplayPx` is the on-screen CSS width represented by `canvasW`.
 * Keep it independent from zoomed `displayPx` so bar thickness, label font,
 * margins, and shadows stay fixed while the image zoom changes.
 */
export function drawScaleBarCanvas(
  ctx: CanvasRenderingContext2D,
  opts: {
    /** Canvas internal pixel width (the coordinate system fillRect draws in). */
    canvasW: number;
    /** Canvas internal pixel height. */
    canvasH: number;
    /** Image dimension in source pixels. */
    scanPx: number | undefined;
    /** Real-space size of one source pixel, in Å. */
    stepA: number | null;
    /** Width on screen in CSS pixels (after CSS scaling, zoom, transform).
     * Used to pick a physical length whose on-screen width is near
     * SCALE_BAR_TARGET_PX. Pass `getBoundingClientRect().width`. */
    displayPx: number;
    /** Deprecated: retained for source compatibility. The picker now matches
     * Show2D and always derives the physical label from displayPx. */
    targetDisplayPx?: number | null;
    /** CSS width corresponding to canvasW. Defaults to displayPx for legacy
     * callers where the image fills the canvas. */
    styleDisplayPx?: number | null;
    /** Right anchor in canvas coords. Defaults to `canvasW - inset`. */
    anchorRight?: number;
    /** Bottom anchor in canvas coords. Defaults to `canvasH - inset`. */
    anchorBottom?: number;
    /** Optional zoom factor — when provided, paints "{z.toFixed(1)}×"
     * at the bottom-left to match widget show2d behavior. */
    zoom?: number;
  },
): void {
  const sb = pickScaleBarPx(opts.scanPx, opts.stepA, opts.displayPx);
  if (!sb) return;
  // sb.barPx is in CSS (on-screen) pixels. Convert those CSS pixels into
  // the current canvas coordinate space using the canvas display footprint,
  // not the zoomed image footprint. Otherwise zooming an image would shrink
  // the label font and stroke thickness.
  const styleDisplayPx =
    opts.styleDisplayPx && opts.styleDisplayPx > 0
      ? opts.styleDisplayPx
      : opts.displayPx;
  const cssToCanvas = opts.canvasW / Math.max(1, styleDisplayPx);
  const barPx = sb.barPx * cssToCanvas;
  if (barPx < 4 || barPx > opts.canvasW * 0.9) return;
  const label = scaleBarLabel(sb.lengthNm);
  // Inset, bar thickness, and font scale with canvas-vs-CSS only, so they
  // look the same on screen across canvas resolutions and viewer zooms.
  const margin = 12 * cssToCanvas;
  const barH = 5 * cssToCanvas;
  const fontPx = 16 * cssToCanvas;
  const right = opts.anchorRight ?? opts.canvasW - margin;
  const bottom = opts.anchorBottom ?? opts.canvasH - margin;

  ctx.save();
  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2 * cssToCanvas;
  ctx.shadowOffsetX = cssToCanvas;
  ctx.shadowOffsetY = cssToCanvas;

  ctx.fillStyle = "#fff";
  ctx.fillRect(right - barPx, bottom, barPx, barH);

  ctx.font = `bold ${fontPx}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(label, right - barPx / 2, bottom - 4 * cssToCanvas);

  if (opts.zoom != null) {
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(`${opts.zoom.toFixed(1)}×`, margin, bottom + barH);
  }

  ctx.restore();
}
