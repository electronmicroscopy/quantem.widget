/**
 * Orthographic renderer of the unit cell (or a block of cells) on a 2D
 * canvas. The view looks along the lab z axis (the beam), so the drawing
 * shares its axes with the diffraction pattern; viewX = -1 mirrors x for the
 * view from the detector side. Optional coordination polyhedra are convex
 * hulls of the nearest neighbours of each centre atom.
 */

import { Quat, Vec3, matVec, quatToMatrix } from "./math";
import type { CrystalData } from "./physics";

export interface CellStyle {
  dark: boolean;
  showAxes: boolean;
  showLabels: boolean;
  atomScale: number; // covalent radius multiplier
  viewX: number; // +1 gun side (beam into the screen), -1 detector side (beam toward the viewer)
}

interface Atom {
  pos: Vec3; // Cartesian, crystal frame, relative to the block centre
  color: string;
  radius: number;
  symbol: string;
}

interface Face {
  verts: Vec3[]; // polygon, crystal frame relative to the block centre
  color: string;
}

function rgb(c: number[]): string {
  return `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;
}

export interface CellGeometry {
  atoms: Atom[];
  corners: Vec3[]; // 8 corners of the block relative to the centre
  edges: [number, number][];
  innerEdges: [Vec3, Vec3][]; // cell boundaries inside the block
  faces: Face[];
  axes: Vec3[]; // a, b, c (one cell) from the origin corner
  origin: Vec3;
  radius: number; // bounding radius, A
}

/**
 * Geometry of an na x nb x nc block of cells. Atoms on the block boundary
 * are repeated (the corner atoms of fcc all appear). Polyhedra are drawn
 * around every species except the most numerous one (cations in an oxide;
 * every atom of an elemental crystal), using neighbours within
 * 1.2 (r_i + r_j) of the covalent radii, including atoms outside the block.
 */
export function cellGeometry(c: CrystalData, nCells: [number, number, number] = [1, 1, 1], polyhedra = false): CellGeometry {
  const cell = c.cell;
  const [na, nb, nc] = nCells.map((n) => Math.max(1, Math.min(6, Math.round(n)))) as [number, number, number];
  const cart = (f: number[]): Vec3 => [
    f[0] * cell[0][0] + f[1] * cell[1][0] + f[2] * cell[2][0],
    f[0] * cell[0][1] + f[1] * cell[1][1] + f[2] * cell[2][1],
    f[0] * cell[0][2] + f[1] * cell[1][2] + f[2] * cell[2][2],
  ];
  const center = cart([na / 2, nb / 2, nc / 2]);
  const rel = (v: Vec3): Vec3 => [v[0] - center[0], v[1] - center[1], v[2] - center[2]];
  const corners: Vec3[] = [];
  for (let i = 0; i < 8; i++) corners.push(rel(cart([(i & 1) * na, ((i >> 1) & 1) * nb, ((i >> 2) & 1) * nc])));
  const edges: [number, number][] = [];
  for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
    const d = i ^ j;
    if (d === 1 || d === 2 || d === 4) edges.push([i, j]);
  }
  const innerEdges: [Vec3, Vec3][] = [];
  const n3 = [na, nb, nc];
  for (let ax = 0; ax < 3; ax++) {
    const [u, v] = [(ax + 1) % 3, (ax + 2) % 3];
    for (let iu = 0; iu <= n3[u]; iu++) for (let iv = 0; iv <= n3[v]; iv++) {
      const onBoundary = (iu === 0 || iu === n3[u]) && (iv === 0 || iv === n3[v]);
      if (onBoundary) continue;
      const f0 = [0, 0, 0], f1 = [0, 0, 0];
      f0[u] = iu; f0[v] = iv; f1[u] = iu; f1[v] = iv; f1[ax] = n3[ax];
      innerEdges.push([rel(cart(f0)), rel(cart(f1))]);
    }
  }

  // atoms inside the block (boundary included) and a halo of images for neighbour search
  const eps = 1e-4;
  const atoms: Atom[] = [];
  const halo: { pos: Vec3; species: number }[] = [];
  const inBlock: boolean[] = [];
  const nSpec = c.positions_frac.length;
  for (let n = 0; n < nSpec; n++) {
    const f = c.positions_frac[n].map((x) => x - Math.floor(x + eps));
    for (let sx = -1; sx <= na + 1; sx++) for (let sy = -1; sy <= nb + 1; sy++) for (let sz = -1; sz <= nc + 1; sz++) {
      const g = [f[0] + sx, f[1] + sy, f[2] + sz];
      const inside = g[0] <= na + eps && g[1] <= nb + eps && g[2] <= nc + eps && g[0] >= -eps && g[1] >= -eps && g[2] >= -eps;
      const p = rel(cart(g));
      halo.push({ pos: p, species: n });
      inBlock.push(inside);
      if (inside) atoms.push({ pos: p, color: rgb(c.colors[n]), radius: c.radii[n], symbol: c.symbols[n] });
    }
  }

  const faces: Face[] = [];
  if (polyhedra) {
    const counts = new Map<string, number>();
    for (const s of c.symbols) counts.set(s, (counts.get(s) || 0) + 1);
    const species = [...counts.keys()];
    let centres = new Set(species);
    if (species.length > 1) {
      const most = species.reduce((a, b) => ((counts.get(a) || 0) >= (counts.get(b) || 0) ? a : b));
      centres = new Set(species.filter((s) => s !== most));
    }
    const cutoffMax = 1.2 * 2 * Math.max(...c.radii);
    for (let i = 0; i < halo.length; i++) {
      if (!inBlock[i]) continue;
      const si = halo[i].species;
      if (!centres.has(c.symbols[si])) continue;
      const pi = halo[i].pos;
      const nbr: Vec3[] = [];
      for (let j = 0; j < halo.length; j++) {
        if (j === i) continue;
        const sj = halo[j].species;
        if (species.length > 1 && c.symbols[sj] === c.symbols[si]) continue;
        const pj = halo[j].pos;
        const d = Math.hypot(pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]);
        if (d > cutoffMax) continue;
        if (d <= 1.2 * (c.radii[si] + c.radii[sj])) nbr.push(pj);
      }
      if (nbr.length >= 4 && nbr.length <= 14) {
        for (const poly of convexHullFaces(nbr, pi)) faces.push({ verts: poly, color: rgb(c.colors[si]) });
      }
    }
  }

  let radius = 0;
  for (const p of corners) radius = Math.max(radius, Math.hypot(p[0], p[1], p[2]));
  return {
    atoms, corners, edges, innerEdges, faces,
    axes: [cell[0] as Vec3, cell[1] as Vec3, cell[2] as Vec3], origin: corners[0], radius: radius + 0.5,
  };
}

/** Faces of the convex hull of a few points (brute force, n <= 14), as outward-ordered polygons. */
function convexHullFaces(pts: Vec3[], inside: Vec3): Vec3[][] {
  const n = pts.length;
  const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const planes = new Map<string, { normal: Vec3; verts: Set<number> }>();
  let scale = 0;
  for (const p of pts) scale = Math.max(scale, Math.hypot(...sub(p, inside)));
  const tol = 1e-3 * scale;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) {
    let nrm = cross(sub(pts[j], pts[i]), sub(pts[k], pts[i]));
    const L = Math.hypot(...nrm);
    if (L < 1e-9) continue;
    nrm = [nrm[0] / L, nrm[1] / L, nrm[2] / L];
    if (dot(nrm, sub(pts[i], inside)) < 0) nrm = [-nrm[0], -nrm[1], -nrm[2]];
    const off = dot(nrm, pts[i]);
    let ok = true;
    const on: number[] = [];
    for (let m = 0; m < n; m++) {
      const d = dot(nrm, pts[m]) - off;
      if (d > tol) { ok = false; break; }
      if (Math.abs(d) <= tol) on.push(m);
    }
    if (!ok) continue;
    const key = `${nrm.map((v) => v.toFixed(2)).join(",")}|${off.toFixed(2)}`;
    const entry = planes.get(key) || { normal: nrm, verts: new Set<number>() };
    for (const m of on) entry.verts.add(m);
    planes.set(key, entry);
  }
  const out: Vec3[][] = [];
  for (const { normal, verts } of planes.values()) {
    const vs = [...verts].map((m) => pts[m]);
    if (vs.length < 3) continue;
    const cen: Vec3 = [0, 0, 0];
    for (const v of vs) { cen[0] += v[0] / vs.length; cen[1] += v[1] / vs.length; cen[2] += v[2] / vs.length; }
    const e1 = sub(vs[0], cen);
    const e2 = cross(normal, e1);
    vs.sort((a, b) => Math.atan2(dot(sub(a, cen), e2), dot(sub(a, cen), e1)) - Math.atan2(dot(sub(b, cen), e2), dot(sub(b, cen), e1)));
    out.push(vs);
  }
  return out;
}

export function drawCell(
  canvas: HTMLCanvasElement, geom: CellGeometry, q: Quat, size: number, style: CellStyle,
) {
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== size * dpr || canvas.height !== size * dpr) {
    canvas.width = size * dpr;
    canvas.height = size * dpr;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);
  const R = quatToMatrix(q);
  const scale = (0.5 * size * 0.82) / geom.radius; // px per A
  const cx = size / 2, cy = size / 2;
  const proj = (v: Vec3): [number, number, number] => {
    const w = matVec(R, v);
    return [cx + style.viewX * w[0] * scale, cy - w[1] * scale, style.viewX * w[2]];
  };
  const edgeColor = style.dark ? "rgba(220,220,220," : "rgba(40,40,40,";
  const depthFrac = (z: number) => Math.min(1, Math.max(0, 0.5 + z / (2 * geom.radius)));
  const pc = geom.corners.map(proj);
  type Prim = { z: number; draw: () => void };
  const prims: Prim[] = [];
  for (const [i, j] of geom.edges) {
    const a = pc[i], b = pc[j];
    const z = 0.5 * (a[2] + b[2]);
    prims.push({
      z,
      draw: () => {
        ctx.strokeStyle = edgeColor + (0.35 + 0.5 * depthFrac(z)) + ")";
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      },
    });
  }
  for (const [p0, p1] of geom.innerEdges) {
    const a = proj(p0), b = proj(p1);
    const z = 0.5 * (a[2] + b[2]);
    prims.push({
      z,
      draw: () => {
        ctx.strokeStyle = edgeColor + (0.12 + 0.2 * depthFrac(z)) + ")";
        ctx.lineWidth = 0.8;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      },
    });
  }
  for (const face of geom.faces) {
    const pv = face.verts.map(proj);
    let z = 0;
    for (const p of pv) z += p[2] / pv.length;
    prims.push({
      z: z - 1e-3,
      draw: () => {
        ctx.beginPath();
        ctx.moveTo(pv[0][0], pv[0][1]);
        for (let i = 1; i < pv.length; i++) ctx.lineTo(pv[i][0], pv[i][1]);
        ctx.closePath();
        ctx.fillStyle = face.color;
        ctx.globalAlpha = 0.18 + 0.17 * depthFrac(z);
        ctx.fill();
        ctx.globalAlpha = 0.6;
        ctx.strokeStyle = face.color;
        ctx.lineWidth = 0.8;
        ctx.stroke();
        ctx.globalAlpha = 1;
      },
    });
  }
  const atomScale = style.atomScale * (geom.faces.length ? 0.6 : 1);
  for (const at of geom.atoms) {
    const p = proj(at.pos);
    const r = Math.max(1.5, at.radius * atomScale * scale);
    prims.push({
      z: p[2],
      draw: () => {
        const grad = ctx.createRadialGradient(p[0] - 0.35 * r, p[1] - 0.35 * r, 0.1 * r, p[0], p[1], r);
        grad.addColorStop(0, lighten(at.color, 0.55));
        grad.addColorStop(0.7, at.color);
        grad.addColorStop(1, lighten(at.color, -0.45));
        ctx.fillStyle = grad;
        ctx.globalAlpha = 0.55 + 0.45 * depthFrac(p[2]);
        ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI); ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = style.dark ? "rgba(0,0,0,0.6)" : "rgba(0,0,0,0.35)";
        ctx.lineWidth = 0.8;
        ctx.stroke();
      },
    });
  }
  prims.sort((a, b) => a.z - b.z);
  for (const p of prims) p.draw();

  if (style.showAxes) {
    const o = proj(geom.origin);
    const names = ["a", "b", "c"];
    const cols = ["#e53935", "#43a047", "#1e88e5"];
    for (let i = 0; i < 3; i++) {
      const ax = geom.axes[i];
      const tip = proj([geom.origin[0] + ax[0], geom.origin[1] + ax[1], geom.origin[2] + ax[2]]);
      ctx.strokeStyle = cols[i];
      ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(tip[0], tip[1]); ctx.stroke();
      if (style.showLabels) {
        const dx = tip[0] - o[0], dy = tip[1] - o[1];
        const L = Math.hypot(dx, dy) || 1;
        ctx.fillStyle = cols[i];
        ctx.font = "bold 14px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(names[i], tip[0] + (dx / L) * 11, tip[1] + (dy / L) * 11);
      }
    }
  }
  // lab axes in the corner
  const ox = 22, oy = size - 22, L = 18;
  ctx.strokeStyle = style.dark ? "#aaa" : "#555";
  ctx.fillStyle = style.dark ? "#aaa" : "#555";
  ctx.lineWidth = 1.2;
  ctx.font = "10px sans-serif";
  ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + L, oy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox, oy - L); ctx.stroke();
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  ctx.fillText("x", ox + L + 3, oy);
  ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  ctx.fillText("y", ox, oy - L - 2);
  ctx.beginPath(); ctx.arc(ox, oy, 3.5, 0, 2 * Math.PI); ctx.stroke();
  ctx.textAlign = "left"; ctx.textBaseline = "top";
  if (style.viewX > 0) {
    ctx.beginPath(); ctx.moveTo(ox - 2.5, oy - 2.5); ctx.lineTo(ox + 2.5, oy + 2.5); ctx.moveTo(ox - 2.5, oy + 2.5); ctx.lineTo(ox + 2.5, oy - 2.5); ctx.stroke();
    ctx.fillText("beam into screen (view from gun)", ox + 6, oy + 4);
  } else {
    ctx.beginPath(); ctx.arc(ox, oy, 1.2, 0, 2 * Math.PI); ctx.fill();
    ctx.fillText("beam toward you (view from detector)", ox + 6, oy + 4);
  }
}

function lighten(color: string, amount: number): string {
  const m = color.match(/rgb\((\d+),(\d+),(\d+)\)/);
  if (!m) return color;
  const f = (v: number) => Math.max(0, Math.min(255, Math.round(amount >= 0 ? v + (255 - v) * amount : v * (1 + amount))));
  return `rgb(${f(+m[1])},${f(+m[2])},${f(+m[3])})`;
}
