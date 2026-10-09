/** Show3D playback: the frame a key moves to, the prewarm order, the fps clamp and the CPU colormap fills of a frame. */
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

/** Fused single-pass render: optional log scale + normalize + colormap → RGBA.
 *  Eliminates multiple data passes during playback for maximum frame rate. */
export function renderFramePlayback(
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
      const v = signedLog1p(d);
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
  const range = vmax - vmin;
  const invRange = range > 0 ? 255 / range : 0;
  let dst = 0;
  for (let row = 0; row < panelHeight; row++) {
    let src = row * sourceWidth + sourceX0;
    const end = src + panelWidth;
    for (; src < end; src++) {
      const raw = source[src];
      const value = logScale
        ? signedLog1p(raw)
        : raw;
      const idx = value <= vmin ? 0 : value >= vmax ? 255 : ((value - vmin) * invRange) | 0;
      const lutIdx = idx * 3;
      rgba[dst] = lut[lutIdx];
      rgba[dst + 1] = lut[lutIdx + 1];
      rgba[dst + 2] = lut[lutIdx + 2];
      rgba[dst + 3] = 255;
      dst += 4;
    }
  }
}

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
  const range = vmax - vmin;
  const invRange = range > 0 ? 255 / range : 0;
  for (let y = 0; y < outH; y++) {
    const srcRow = yMap[y];
    const outRow = y * outW;
    for (let x = 0; x < outW; x++) {
      let v = data[srcRow + xMap[x]];
      if (logScale) v = signedLog1p(v);
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
