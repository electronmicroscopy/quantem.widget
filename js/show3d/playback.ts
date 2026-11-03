/** Show3D playback: the frame a key moves to, the prewarm order, the fps clamp and the CPU colormap fills of a frame. */
import { packedLut } from "../display/displayMath";
import { signedLog1p } from "../display/stats";

/**
 * The frame a playback key moves to, or null for a key that does not move the frame.
 *
 * Steps start from `current`, the frame on screen. The fast scrub path draws a frame and
 * commits the React state only after the input settles (350 ms), so stepping from React state
 * made every press of a held arrow key compute the same next frame: the image stayed one frame
 * ahead for the whole hold. `loopRange` [first, last] bounds the steps while Loop is on.
 */
export function frameKeyTarget(
  key: string,
  current: number,
  frameCount: number,
  loopRange: [number, number] | null,
): number | null {
  const first = loopRange ? Math.max(0, loopRange[0]) : 0;
  const last = loopRange ? Math.min(loopRange[1], frameCount - 1) : frameCount - 1;
  if (key === "ArrowLeft") return Math.max(first, current - 1);
  if (key === "ArrowRight") return Math.min(last, current + 1);
  if (key === "Home") return first;
  if (key === "End") return last;
  return null;
}

/**
 * Moves the frame slider thumb, its track and the frame counters of one widget to frame `idx` without a React
 * render, so a fast scrub or playback shows the frame actually drawn at once.
 *
 * The playback row is an honesty indicator for the frame actually drawn, so it is not clamped to the loop
 * handles: custom playback paths, transient trait hydration and direct-frame filter retries can legitimately
 * display a frame outside the loop span (clamping made users see "1/18" while the canvas showed a later frame).
 * Every element is looked up inside `root`: a page-wide lookup let one Show3D write another's counter when its
 * own counter was not rendered.
 */
export function writeLiveFrameControls(
  root: HTMLElement | null,
  slider: HTMLElement | null,
  count: HTMLElement | null,
  idx: number,
  total: number,
  loop: boolean,
): void {
  const clamped = Math.max(0, Math.min(total - 1, Math.round(idx)));
  const pct = total > 1 ? (clamped / (total - 1)) * 100 : 0;
  const activeThumb = slider?.querySelector(loop ? ".MuiSlider-thumb[data-index='1']" : ".MuiSlider-thumb") as HTMLElement | null;
  const track = slider?.querySelector(".MuiSlider-track") as HTMLElement | null;
  const input = activeThumb?.querySelector("input") as HTMLInputElement | null;
  const counter = count ?? root?.querySelector("[data-show3d-playback-count]");
  if (activeThumb) {
    activeThumb.style.left = `${pct}%`;
    activeThumb.setAttribute("aria-valuenow", String(clamped));
  }
  if (input) input.value = String(clamped);
  if (track && !loop) {
    track.style.left = "0%";
    track.style.width = `${pct}%`;
  }
  if (counter) counter.textContent = `${clamped + 1}/${total}`;
  root?.querySelectorAll<HTMLElement>("[data-show3d-panel-frame-count]").forEach((el) => {
    const panelTotal = Math.max(1, Math.round(Number(el.dataset.realFrameCount || total) || total));
    el.textContent = `${Math.min(clamped + 1, panelTotal)}/${panelTotal}`;
  });
}

/**
 * Frame indices ordered by distance from startIdx, alternating forward and
 * backward with wrap-around: [s, s+1, s-1, s+2, s-2, ...]. Prewarming in this
 * order caches first the frames a scrub in either direction reaches first.
 */
export function orderedFramePrewarmIndices(startIdx: number, nFrames: number): number[] {
  const frameCount = Math.max(1, Math.round(nFrames || 1));
  const start = ((Math.round(startIdx) % frameCount) + frameCount) % frameCount;
  const order: number[] = [start];
  for (let distance = 1; distance < frameCount; distance++) {
    order.push((start + distance) % frameCount);
    if (order.length >= frameCount) break;
    order.push((start - distance + frameCount) % frameCount);
  }
  return order;
}

export const MAX_PLAYBACK_FPS = 60;

export const clampPlaybackFps = (value: number) => {
  const fps = Number.isFinite(value) ? value : 1;
  return Math.max(1, Math.min(MAX_PLAYBACK_FPS, fps));
};

export const playbackIntervalMs = (value: number) => {
  const fps = clampPlaybackFps(value);
  return 1000 / fps;
};

/** The display window of a CPU playback colormap pass, with the LUT packed one RGBA word per entry. */
type PlaybackWindow = { lut: Uint32Array; vmin: number; vmax: number; logScale: boolean };

/**
 * One row of the CPU playback colormap: output pixel k (0 to count - 1, written from pixel `targetStart`
 * of `pixels`) shows `data[sourceStart + (columnMap ? columnMap[k] : k)]`. Log scale is the signed log1p,
 * and the LUT entry the float64 floor of (v - vmin) * 255 / (vmax - vmin), clamped at both ends. Every
 * CPU playback renderer draws with this one loop, so a scaled, packed or full frame colors alike; a
 * pixel is one 32-bit store of the packed LUT entry.
 */
function colormapPlaybackRow(
  data: Float32Array,
  sourceStart: number,
  columnMap: Uint32Array | null,
  count: number,
  pixels: Uint32Array,
  targetStart: number,
  { lut, vmin, vmax, logScale }: PlaybackWindow,
): void {
  const range = vmax - vmin;
  const invRange = range > 0 ? 255 / range : 0;
  const entry = (raw: number) => {
    const value = logScale ? signedLog1p(raw) : raw;
    return value <= vmin ? 0 : value >= vmax ? 255 : ((value - vmin) * invRange) | 0;
  };
  // Separate loops keep the contiguous rows free of a per-pixel column-map branch.
  if (columnMap) {
    for (let k = 0; k < count; k++) pixels[targetStart + k] = lut[entry(data[sourceStart + columnMap[k]])];
  } else {
    for (let k = 0; k < count; k++) pixels[targetStart + k] = lut[entry(data[sourceStart + k])];
  }
}

/** The RGBA bytes of an ImageData as one 32-bit word per pixel. */
const pixelWords = (rgba: Uint8ClampedArray) => new Uint32Array(rgba.buffer, rgba.byteOffset, rgba.length >> 2);

/** Colormap a whole frame into RGBA in one pass: the CPU playback path, used without WebGPU. */
export function renderFramePlayback(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  logScale: boolean,
): void {
  colormapPlaybackRow(data, 0, null, data.length, pixelWords(rgba), 0, { lut: packedLut(lut), vmin, vmax, logScale });
}

/** Render one packed multi-panel slice into RGBA without allocating a panel copy.
 *
 * Large standalone reports can pack many panels side-by-side in one frame
 * (for example 8 × 1366 × 1366). Copying each panel into a fresh Float32Array
 * during playback allocates tens of MB per frame and can crash Chromium with
 * `Array buffer allocation failed`. This renderer walks the packed source rows
 * directly and writes into a reusable per-panel ImageData buffer.
 */
export function renderPackedPanelPlayback(
  source: Float32Array,
  sourceWidth: number,
  sourceX0: number,
  panelWidth: number,
  panelHeight: number,
  rgba: Uint8ClampedArray,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  logScale: boolean,
): void {
  const pixels = pixelWords(rgba);
  const displayWindow = { lut: packedLut(lut), vmin, vmax, logScale };
  for (let row = 0; row < panelHeight; row++) {
    colormapPlaybackRow(source, row * sourceWidth + sourceX0, null, panelWidth, pixels, row * panelWidth, displayWindow);
  }
}

/**
 * Colormap a frame straight into an outW x outH canvas image, nearest neighbour: output pixel (x, y)
 * shows source pixel yMap[y] + xMap[x] (yMap holds row offsets). Skips the full-size pass a
 * downscaled view would otherwise colormap and then shrink.
 */
export function renderFrameScaledPlayback(
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
  const pixels = pixelWords(rgba);
  const displayWindow = { lut: packedLut(lut), vmin, vmax, logScale };
  for (let y = 0; y < outH; y++) colormapPlaybackRow(data, yMap[y], xMap, outW, pixels, y * outW, displayWindow);
}
