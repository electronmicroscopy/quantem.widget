/**
 * Pattern renderers for the simulator: nanobeam markers or pixels, CBED
 * disks, Kikuchi (Kossel) lines and the Kossel reference lookup. Every renderer works
 * in canvas coordinates with q_x to the right and q_y up.
 */

import { COLORMAPS, applyColormap } from "../colormaps";
import { formatScaleLabel, roundToNiceValue } from "../figure";
import type { Reflection } from "./physics";
import type { KosselLine } from "./physics";

const MAX_LABELS = 20; // strongest reflections labelled

export interface Frame {
  size: number; // canvas CSS px (square)
  qMax: number; // 1/A at the edge (nanobeam / CBED) or rad (Kossel)
  viewX: number; // +1: seen from the gun side (lab x to the right); -1: from the detector side (mirrored)
}

export function scaleOf(f: Frame): number {
  return (0.5 * f.size * 0.92) / f.qMax; // px per unit
}

/** Lab (x, y) in pattern units to canvas px. */
export function toPx(f: Frame, x: number, y: number): [number, number] {
  const s = scaleOf(f);
  return [f.size / 2 + f.viewX * x * s, f.size / 2 - y * s];
}

/** Canvas px to lab (x, y) in pattern units. */
export function fromPx(f: Frame, px: number, py: number): [number, number] {
  const s = scaleOf(f);
  return [(f.viewX * (px - f.size / 2)) / s, -(py - f.size / 2) / s];
}

export function setupCanvas(canvas: HTMLCanvasElement, size: number): CanvasRenderingContext2D | null {
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== size * dpr || canvas.height !== size * dpr) {
    canvas.width = size * dpr;
    canvas.height = size * dpr;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/**
 * Nanobeam pattern as markers. Marker AREA scales as intensity^markerPower
 * (0.5 = sqrt intensity, the default); markerSize is the radius (px) of
 * the strongest beam, capped so the densest net does not merge.
 */
export function drawMarkers(
  ctx: CanvasRenderingContext2D, f: Frame, beams: Reflection[], inten: Float64Array, dark: boolean,
  labels: boolean, kinematic: boolean, markerPower = 0.5, markerSize = 20,
) {
  const s = scaleOf(f);
  ctx.fillStyle = dark ? "#000" : "#fff";
  ctx.fillRect(0, 0, f.size, f.size);
  // normalise to the strongest diffracted beam: the direct beam saturates and the weak spots stay visible
  let iMax = 0;
  for (let i = 0; i < beams.length; i++) if (beams[i].index >= 0) iMax = Math.max(iMax, inten[i]);
  if (iMax <= 0) iMax = 1;
  // marker radius capped so neighbouring spots of the densest net do not merge
  let gMin = Infinity;
  for (const b of beams) if (b.index >= 0 && b.gLen > 1e-6) gMin = Math.min(gMin, b.gLen);
  const rMax = Math.min(markerSize * (f.size / 420), isFinite(gMin) ? 0.42 * gMin * s : Infinity);
  const fg = dark ? "#fff" : "#000";
  const strong: { x: number; y: number; r: number; hkl: number[]; rel: number }[] = [];
  for (let i = 0; i < beams.length; i++) {
    const b = beams[i];
    const rel = Math.min(1, inten[i] / iMax);
    const [x, y] = toPx(f, b.g[0], b.g[1]);
    if (b.index < 0 && kinematic) {
      ctx.strokeStyle = fg; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, rMax * 0.9, 0, 2 * Math.PI); ctx.stroke();
      strong.push({ x, y, r: rMax * 0.9, hkl: b.hkl, rel: 2 });
      continue;
    }
    if (rel < 1e-6) continue;
    const r = rMax * Math.pow(rel, 0.5 * markerPower); // area ~ I^markerPower
    ctx.fillStyle = fg;
    ctx.globalAlpha = 0.9;
    ctx.beginPath(); ctx.arc(x, y, Math.max(r, 0.6), 0, 2 * Math.PI); ctx.fill();
    ctx.globalAlpha = 1;
    if (rel > 0.08) strong.push({ x, y, r, hkl: b.hkl, rel: b.index < 0 ? 2 : rel });
  }
  if (labels) {
    ctx.font = `${Math.max(9, Math.round(f.size / 38))}px sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillStyle = dark ? "#ffd54f" : "#c62828";
    for (const p of strong.sort((a, b) => b.rel - a.rel).slice(0, MAX_LABELS)) {
      ctx.fillText(hklText(p.hkl), p.x, p.y - p.r - 2);
    }
  }
  drawScaleBar(ctx, f, 1 / s, "Å⁻¹", dark ? "#fff" : "#000", dark);
}

/**
 * Nanobeam pattern as disks of the physical convergence angle: filled
 * circles of radius k0 sin(alpha) at every beam, brightness (I/Imax)^power.
 * Overlapping disks add; the direct beam is drawn like the others.
 */
export function drawDisks(
  ctx: CanvasRenderingContext2D, f: Frame, beams: Reflection[], inten: Float64Array, dark: boolean,
  labels: boolean, radiusQ: number, power = 0.5, vmin = 0, vmax = 1,
) {
  const s = scaleOf(f);
  // dark theme: bright disks on black, adding where they overlap; light
  // theme: the inverted greyscale, dark disks on white, multiplying
  ctx.fillStyle = dark ? "#000" : "#fff";
  ctx.fillRect(0, 0, f.size, f.size);
  // normalise to the strongest diffracted beam (the direct beam saturates)
  let iMax = 0;
  for (let i = 0; i < beams.length; i++) if (beams[i].index >= 0) iMax = Math.max(iMax, inten[i]);
  if (iMax <= 0) iMax = 1;
  const r = Math.max(1.2, radiusQ * s);
  const strong: { x: number; y: number; hkl: number[]; rel: number }[] = [];
  ctx.globalCompositeOperation = dark ? "lighter" : "multiply";
  for (let i = 0; i < beams.length; i++) {
    const rel = Math.min(1, inten[i] / iMax);
    if (rel < 1e-6) continue;
    const [x, y] = toPx(f, beams[i].g[0], beams[i].g[1]);
    if (x < -r || y < -r || x > f.size + r || y > f.size + r) continue;
    const v = Math.min(1, Math.max(0, (Math.pow(rel, power) - vmin) / Math.max(vmax - vmin, 1e-6))); // contrast window on I^power
    const c = Math.round(255 * (dark ? v : 1 - v));
    ctx.fillStyle = `rgb(${c},${c},${c})`;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
    if (rel > 0.08) strong.push({ x, y, hkl: beams[i].hkl, rel: beams[i].index < 0 ? 2 : rel });
  }
  ctx.globalCompositeOperation = "source-over";
  if (labels) {
    ctx.font = `${Math.max(9, Math.round(f.size / 38))}px sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillStyle = dark ? "#ffd54f" : "#c62828";
    for (const p of strong.sort((a, b) => b.rel - a.rel).slice(0, MAX_LABELS)) ctx.fillText(hklText(p.hkl), p.x, p.y - r - 2);
  }
  drawScaleBar(ctx, f, 1 / s, "Å⁻¹", dark ? "#fff" : "#000", dark);
}

export function hklText(hkl: number[]): string {
  return hkl.map((h) => (h < 0 ? `${-h}̅` : `${h}`)).join("");
}

/**
 * Side view of the Ewald sphere, filling a W x H canvas: the lab x-z
 * plane (x mirrored by viewX like the pattern, +z = upstream at the top),
 * the reciprocal lattice points with |g_y| below a slab width as dots, the
 * sphere z = k0 - sqrt(k0^2 - x^2) through the origin, and the excited
 * reflections (|s| < sgMax) highlighted. The z axis is stretched so the
 * sphere's sagitta over the pattern range fills the inset. `background` is
 * the panel color, a theme token of the host page.
 */
export function drawEwaldPanel(
  ctx: CanvasRenderingContext2D, W: number, H: number, refl: Reflection[], k0: number, qMax: number, sgMax: number,
  viewX: number, dark: boolean, background: string, precDeg = 0,
) {
  const size = Math.max(W, 2 * H); // reference for the font size
  const x0 = 0, y0 = 0;
  const fg = dark ? "#d8d8d8" : "#222";
  ctx.save();
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, W, H);
  ctx.beginPath(); ctx.rect(x0, y0, W, H); ctx.clip();
  const sinP = Math.sin((precDeg * Math.PI) / 180);
  const sag = (qMax * qMax) / (2 * k0) + qMax * sinP; // sphere height over the pattern range, incl. the precession tilt
  const zHalf = Math.max(sag * 1.15, 4 * sgMax);
  const sx = (W * 0.46) / qMax; // px per 1/A along x
  const sz = (H * 0.4) / zHalf; // px per 1/A along z (stretched)
  const cx = x0 + W / 2, cy = y0 + H * 0.64; // origin: lower middle
  const X = (x: number) => cx + viewX * x * sx;
  const Z = (z: number) => cy - z * sz;
  // sphere for an incident beam with in-plane wavevector tx: centre (-tx, kz), through the origin
  const sphereZ = (x: number, tx: number) => {
    const kz = Math.sqrt(Math.max(k0 * k0 - tx * tx, 0));
    return kz - Math.sqrt(Math.max(k0 * k0 - (x + tx) * (x + tx), 0));
  };
  const xs: number[] = [];
  for (let i = 0; i <= 60; i++) xs.push(-qMax * 1.08 + (2.16 * qMax * i) / 60);
  const orange = dark ? "#e0b060" : "#c07a00";
  if (sinP > 0) {
    // precession: the sphere sweeps the band between its two extreme tilts in this plane
    const tA = k0 * sinP, tB = -k0 * sinP;
    ctx.fillStyle = dark ? "rgba(224,176,96,0.22)" : "rgba(192,122,0,0.18)";
    ctx.beginPath();
    xs.forEach((x, i) => { const z = sphereZ(x, tA); if (i === 0) ctx.moveTo(X(x), Z(z)); else ctx.lineTo(X(x), Z(z)); });
    for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(X(xs[i]), Z(sphereZ(xs[i], tB)));
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = orange; ctx.lineWidth = 0.9;
    for (const t of [tA, tB]) {
      ctx.beginPath();
      xs.forEach((x, i) => { const z = sphereZ(x, t); if (i === 0) ctx.moveTo(X(x), Z(z)); else ctx.lineTo(X(x), Z(z)); });
      ctx.stroke();
    }
    // the beam cone: the two extreme incident directions, from the top to the origin
    ctx.strokeStyle = dark ? "#9ad" : "#37c"; ctx.lineWidth = 1;
    const zTop = (cy - y0 - 4) / sz;
    for (const t of [tA, tB]) {
      const kz = Math.sqrt(Math.max(k0 * k0 - t * t, 0));
      ctx.beginPath(); ctx.moveTo(X((-t / kz) * zTop), Z(zTop)); ctx.lineTo(cx, cy); ctx.stroke();
    }
  }
  // untilted beam arrow (travels -z: from the top toward the origin) and sphere
  ctx.strokeStyle = dark ? "#9ad" : "#37c"; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 1.3;
  ctx.beginPath(); ctx.moveTo(cx, y0 + 4); ctx.lineTo(cx, cy - 3); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx - 3.5, cy - 7); ctx.lineTo(cx + 3.5, cy - 7); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = orange; ctx.lineWidth = 1.4;
  ctx.beginPath();
  xs.forEach((x, i) => { const z = sphereZ(x, 0); if (i === 0) ctx.moveTo(X(x), Z(z)); else ctx.lineTo(X(x), Z(z)); });
  ctx.stroke();
  // reciprocal lattice points in a slab about the x-z plane
  const slab = 0.12 * qMax;
  for (const r of refl) {
    if (Math.abs(r.g[1]) > slab || Math.abs(r.g[0]) > qMax * 1.08 || Math.abs(r.g[2]) > zHalf * 1.3) continue;
    const excited = Math.abs(r.s) < sgMax + Math.abs(r.g[0]) * sinP; // within the swept band
    ctx.fillStyle = excited ? "#00cc66" : (dark ? "#8a8a8a" : "#777");
    ctx.beginPath(); ctx.arc(X(r.g[0]), Z(r.g[2]), excited ? 2.6 : 1.6, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.fillStyle = dark ? "#eee" : "#111";
  ctx.beginPath(); ctx.arc(cx, cy, 2.4, 0, 2 * Math.PI); ctx.fill();
  // labels on an opaque strip so the cone lines do not run through them
  const fontPx = Math.max(10, Math.round(size / 36));
  ctx.font = `${fontPx}px sans-serif`; ctx.textAlign = "left"; ctx.textBaseline = "top";
  const line1 = `Ewald sphere, side view, z ×${Math.round(sz / sx)}`;
  const line2 = sinP > 0 ? `precession ±${precDeg.toFixed(2)}°` : "";
  const tw = Math.max(ctx.measureText(line1).width, line2 ? ctx.measureText(line2).width : 0);
  ctx.save();
  ctx.globalAlpha = 0.9;
  ctx.fillStyle = background;
  ctx.fillRect(x0 + 1, y0 + 1, tw + 9, (line2 ? 2.5 : 1.2) * fontPx + 6);
  ctx.restore();
  ctx.fillStyle = fg;
  ctx.fillText(line1, x0 + 5, y0 + 4);
  if (line2) ctx.fillText(line2, x0 + 5, y0 + 5 + 1.4 * fontPx);
  ctx.restore();
}

// Scale bar with the geometry and label rules of js/figure.ts
// (drawScaleBarHiDPI): a 60 px target rounded to 1, 2 or 5 x 10^n, 5 px
// thick, 12 px from the bottom right corner. Drawn into the pattern canvas
// itself, in `color`; `shadow` adds the drop shadow used on images.
const SCALE_BAR_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
function drawScaleBar(ctx: CanvasRenderingContext2D, f: Frame, unitsPerPx: number, unit: string, color: string, shadow: boolean) {
  const nice = roundToNiceValue(60 * unitsPerPx);
  const barPx = nice / unitsPerPx;
  const margin = 12, thickness = 5;
  const x = f.size - barPx - margin, y = f.size - margin;
  ctx.save();
  ctx.fillStyle = color;
  ctx.fillRect(x, y, barPx, thickness);
  if (shadow) {
    ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
    ctx.shadowBlur = 2; ctx.shadowOffsetX = 1; ctx.shadowOffsetY = 1;
  }
  ctx.font = `14px ${SCALE_BAR_FONT}`;
  ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  ctx.fillText(formatScaleLabel(nice, unit), x + barPx / 2, y - 4);
  ctx.restore();
}

/** Splat beams as Gaussian spots into a float image (canvas order). */
export function nanobeamImage(f: Frame, beams: Reflection[], inten: Float64Array, spotPx: number): Float32Array {
  const n = f.size;
  const img = new Float32Array(n * n);
  const sig = spotPx;
  const w = Math.ceil(3.5 * sig);
  for (let i = 0; i < beams.length; i++) {
    if (inten[i] <= 0) continue;
    const [x, y] = toPx(f, beams[i].g[0], beams[i].g[1]);
    const amp = inten[i] / (2 * Math.PI * sig * sig);
    const x0 = Math.max(0, Math.floor(x - w)), x1 = Math.min(n - 1, Math.ceil(x + w));
    const y0 = Math.max(0, Math.floor(y - w)), y1 = Math.min(n - 1, Math.ceil(y + w));
    for (let py = y0; py <= y1; py++) {
      const dy = py + 0.5 - y;
      for (let px = x0; px <= x1; px++) {
        const dx = px + 0.5 - x;
        img[py * n + px] += amp * Math.exp(-(dx * dx + dy * dy) / (2 * sig * sig));
      }
    }
  }
  return img;
}

export interface TiltGrid {
  n: number; // grid points per side
  h: number; // spacing, 1/A
  R: number; // disk radius, 1/A
  tilts: [number, number][]; // grid points within R + h (row-major over the n x n grid, NaN-free)
  index: Int32Array; // n*n -> position in tilts or -1
}

export function tiltGrid(R: number, n: number): TiltGrid {
  const h = (2 * R) / (n - 1);
  const tilts: [number, number][] = [];
  const index = new Int32Array(n * n).fill(-1);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const tx = -R + i * h, ty = -R + j * h;
      if (Math.hypot(tx, ty) <= R + 1.01 * h) {
        index[j * n + i] = tilts.length;
        tilts.push([tx, ty]);
      }
    }
  }
  return { n, h, R, tilts, index };
}

/**
 * CBED image: for every beam a disk of radius R centered on g, with the
 * intensity at each pixel interpolated bilinearly from the tilt grid.
 * intensities[t][b] is the intensity of beam b at tilt t.
 */
export function cbedImage(f: Frame, beams: Reflection[], grid: TiltGrid, intensities: Float64Array[]): Float32Array {
  const n = f.size;
  const img = new Float32Array(n * n);
  const s = scaleOf(f);
  const Rpx = grid.R * s;
  const lookup = (b: number, tx: number, ty: number): number => {
    const fx = (tx + grid.R) / grid.h, fy = (ty + grid.R) / grid.h;
    const i0 = Math.min(Math.max(Math.floor(fx), 0), grid.n - 2);
    const j0 = Math.min(Math.max(Math.floor(fy), 0), grid.n - 2);
    const wx = Math.min(Math.max(fx - i0, 0), 1), wy = Math.min(Math.max(fy - j0, 0), 1);
    const v = (i: number, j: number) => {
      const k = grid.index[j * grid.n + i];
      return k < 0 ? 0 : intensities[k][b];
    };
    return v(i0, j0) * (1 - wx) * (1 - wy) + v(i0 + 1, j0) * wx * (1 - wy) + v(i0, j0 + 1) * (1 - wx) * wy + v(i0 + 1, j0 + 1) * wx * wy;
  };
  for (let b = 0; b < beams.length; b++) {
    const [cx, cy] = toPx(f, beams[b].g[0], beams[b].g[1]);
    if (cx < -Rpx || cy < -Rpx || cx > n + Rpx || cy > n + Rpx) continue;
    let anyInt = 0;
    for (const arr of intensities) if (arr[b] > 1e-7) { anyInt = 1; break; }
    if (!anyInt) continue;
    const x0 = Math.max(0, Math.floor(cx - Rpx - 1)), x1 = Math.min(n - 1, Math.ceil(cx + Rpx + 1));
    const y0 = Math.max(0, Math.floor(cy - Rpx - 1)), y1 = Math.min(n - 1, Math.ceil(cy + Rpx + 1));
    for (let py = y0; py <= y1; py++) {
      const dy = (py + 0.5 - cy);
      for (let px = x0; px <= x1; px++) {
        const dx = (px + 0.5 - cx);
        const rr = Math.hypot(dx, dy);
        if (rr > Rpx + 0.5) continue;
        const edge = Math.min(1, Rpx + 0.5 - rr); // anti-aliased rim
        const tx = (f.viewX * dx) / s, ty = -dy / s;
        img[py * n + px] += edge * lookup(b, tx, ty);
      }
    }
  }
  return img;
}

/** Colormaps whose lookup table comes from quantem through Python (`cmap_lut`). */
export const QUANTEM_COLORMAP_NAMES = ["turbo_black", "turbo_black_r"];

/**
 * 256-entry RGB lookup table of a shared colormap. A 768-entry `custom` table
 * (sent from Python for colormaps imported from quantem) takes precedence. A
 * name ending in "_r" with no table of its own gives the reversed base map
 * (e.g. "gray_r"); unknown names fall back to the first map.
 */
export function colormapLut(name: string, custom?: ArrayLike<number>): Uint8Array {
  if (custom && custom.length === 768) return Uint8Array.from(custom);
  if (COLORMAPS[name]) return COLORMAPS[name];
  const base = name.endsWith("_r") ? COLORMAPS[name.slice(0, -2)] : undefined;
  if (!base) return COLORMAPS[Object.keys(COLORMAPS)[0]];
  const out = new Uint8Array(base.length);
  for (let i = 0; i < 256; i++) out.set(base.subarray(3 * (255 - i), 3 * (255 - i) + 3), 3 * i);
  return out;
}

/**
 * Colormapped float image onto the canvas, with a contrast window. `unit` is
 * "Å⁻¹" (frame in 1/A) or "mrad" (Kikuchi frame in rad).
 */
export function drawImage(
  ctx: CanvasRenderingContext2D, f: Frame, img: Float32Array, cmap: string, vmin: number, vmax: number,
  dark: boolean, unit: string, customLut?: ArrayLike<number>,
) {
  const n = f.size;
  const lut = colormapLut(cmap, customLut);
  const rgba = new Uint8ClampedArray(n * n * 4);
  const clean = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) clean[i] = isFinite(img[i]) ? img[i] : vmin;
  applyColormap(clean, rgba, lut, vmin, vmax);
  for (let i = 0; i < n * n; i++) if (!isFinite(img[i])) rgba[4 * i + 3] = 0;
  const off = document.createElement("canvas");
  off.width = n; off.height = n;
  const octx = off.getContext("2d");
  if (!octx) return;
  octx.putImageData(new ImageData(rgba, n, n), 0, 0);
  ctx.fillStyle = dark ? "#000" : "#fff";
  ctx.fillRect(0, 0, n, n);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, n, n);
  const perPx = 1 / scaleOf(f);
  // the corner outside the aperture is the theme background, so the bar follows the theme
  drawScaleBar(ctx, f, unit === "mrad" ? 1000 * perPx : perPx, unit, dark ? "#fff" : "#000", dark);
}

/** Bright field Kossel pattern as vector lines (deficient lines dark). */
export function drawKosselLines(
  ctx: CanvasRenderingContext2D, f: Frame, lines: KosselLine[], dark: boolean, labels: boolean, minStrength: number,
) {
  const n = f.size, s = scaleOf(f);
  const cx = n / 2, cy = n / 2, Rpx = f.qMax * s;
  ctx.fillStyle = dark ? "#000" : "#fff";
  ctx.fillRect(0, 0, n, n);
  ctx.save();
  ctx.beginPath(); ctx.arc(cx, cy, Rpx, 0, 2 * Math.PI); ctx.clip();
  ctx.fillStyle = dark ? "#bdbdbd" : "#e0e0e0";
  ctx.fill();
  const sorted = [...lines].filter((l) => l.strength >= minStrength).sort((a, b) => a.strength - b.strength);
  const L = 2 * f.qMax;
  for (const l of sorted) {
    const [nx, ny] = l.normal;
    // line p . n = distance, direction t = (-ny, nx); drawn between two lab points
    const [x0, y0] = toPx(f, nx * l.distance - ny * L, ny * l.distance + nx * L);
    const [x1, y1] = toPx(f, nx * l.distance + ny * L, ny * l.distance - nx * L);
    ctx.strokeStyle = dark ? `rgba(20,20,20,${Math.min(1, 0.1 + 0.9 * l.strength)})` : `rgba(30,30,30,${Math.min(1, 0.1 + 0.9 * l.strength)})`;
    ctx.lineWidth = Math.max(0.6, l.width * s);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }
  if (labels) {
    ctx.font = `${Math.max(9, Math.round(n / 40))}px sans-serif`;
    ctx.fillStyle = dark ? "#ffd54f" : "#c62828";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const strong = sorted.filter((l) => l.strength > 0.3 && Math.abs(l.distance) < f.qMax * 0.95).slice(-24);
    for (const l of strong) {
      const [nx, ny] = l.normal;
      // label near the rim along the line
      const t = Math.sqrt(Math.max(f.qMax * f.qMax * 0.8 - l.distance * l.distance, 0));
      const [x, y] = toPx(f, nx * l.distance - ny * t, ny * l.distance + nx * t);
      ctx.fillText(hklText(l.hkl), x, y);
    }
  }
  ctx.restore();
  drawScaleBar(ctx, f, 1000 / s, "mrad", dark ? "#fff" : "#000", dark);
}

/**
 * Kikuchi line pairs overlaid on a nanobeam pattern (deficient blue, excess
 * red). Opacity and width grow with the cube of the strength relative to the
 * strongest line of the cell, so only the few major lines read as heavy and
 * the rest fade into the background; at most the MAX_KIKUCHI strongest pairs
 * are drawn, weakest first.
 */
const MAX_KIKUCHI = 40;
export function drawKikuchiOverlay(ctx: CanvasRenderingContext2D, f: Frame, lines: KosselLine[], k0: number, dark: boolean) {
  const L = 3 * f.qMax;
  const shown = lines.filter((l) => l.strength > 0).sort((a, b) => b.strength - a.strength).slice(0, MAX_KIKUCHI).reverse();
  const sMax = shown.length ? shown[shown.length - 1].strength : 1;
  for (const l of shown) {
    const d = l.distance * k0; // 1/A
    const [nx, ny] = l.normal;
    const r3 = (l.strength / sMax) ** 3;
    const alpha = (0.05 + 0.5 * r3).toFixed(3);
    const width = 0.5 + 1.3 * r3;
    const draw = (dist: number, color: string) => {
      const [x0, y0] = toPx(f, nx * dist - ny * L, ny * dist + nx * L);
      const [x1, y1] = toPx(f, nx * dist + ny * L, ny * dist - nx * L);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    };
    draw(d, dark ? `rgba(120,170,255,${alpha})` : `rgba(30,90,200,${alpha})`);
    draw(d + l.gxy, dark ? `rgba(255,140,120,${alpha})` : `rgba(200,60,40,${alpha})`);
  }
}
