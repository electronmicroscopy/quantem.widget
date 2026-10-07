/**
 * CIF reader for the browser simulator: cell, symmetry operations and atom
 * sites from a CIF text, expanded to the full cell, with kinematical
 * structure factors (Lobato & Van Dyck electron scattering factors, see
 * lobato.ts) and Bloch couplings on the reciprocal lattice out to k_max.
 *
 * Approximation: the absorptive part of the potential is not computed here.
 * It is a fixed fraction of the elastic coupling (ABSORPTION_FRACTION =
 * 0.08), which damps the thickness fringes at a plausible rate but is not a
 * calculated absorption. The preset structures carry the absorptive
 * Weickenmeier-Kohl factors computed in Python by quantem (Weickenmeier &
 * Kohl, Acta Cryst. A 47, 590-597 (1991),
 * https://doi.org/10.1107/S0108767391004774).
 */

import type { CrystalData } from "../diffractionsim/physics";
import { relativisticGamma, electronWavelength } from "../diffractionsim/physics";
import { LOBATO } from "./lobato";

const ABSORPTION_FRACTION = 0.08;

const SYMBOLS = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"];

// jmol colors (fraction of 255) and covalent radii (A) for the drawn atoms
const JMOL: Record<string, [number, number, number]> = {
  H: [1, 1, 1], He: [0.85, 1, 1], Li: [0.8, 0.5, 1], Be: [0.76, 1, 0], B: [1, 0.71, 0.71], C: [0.56, 0.56, 0.56], N: [0.19, 0.31, 0.97], O: [1, 0.05, 0.05],
  F: [0.56, 0.88, 0.31], Ne: [0.7, 0.89, 0.96], Na: [0.67, 0.36, 0.95], Mg: [0.54, 1, 0], Al: [0.75, 0.65, 0.65], Si: [0.94, 0.78, 0.63], P: [1, 0.5, 0], S: [1, 1, 0.19],
  Cl: [0.12, 0.94, 0.12], Ar: [0.5, 0.82, 0.89], K: [0.56, 0.25, 0.83], Ca: [0.24, 1, 0], Sc: [0.9, 0.9, 0.9], Ti: [0.75, 0.76, 0.78], V: [0.65, 0.65, 0.67], Cr: [0.54, 0.6, 0.78],
  Mn: [0.61, 0.48, 0.78], Fe: [0.88, 0.4, 0.2], Co: [0.94, 0.56, 0.63], Ni: [0.31, 0.82, 0.31], Cu: [0.78, 0.5, 0.2], Zn: [0.49, 0.5, 0.69], Ga: [0.76, 0.56, 0.56], Ge: [0.4, 0.56, 0.56],
  As: [0.74, 0.5, 0.89], Se: [1, 0.63, 0], Br: [0.65, 0.16, 0.16], Kr: [0.36, 0.72, 0.82], Rb: [0.44, 0.18, 0.69], Sr: [0, 1, 0], Y: [0.58, 1, 1], Zr: [0.58, 0.88, 0.88],
  Nb: [0.45, 0.76, 0.79], Mo: [0.33, 0.71, 0.71], Tc: [0.23, 0.62, 0.62], Ru: [0.14, 0.56, 0.56], Rh: [0.04, 0.49, 0.55], Pd: [0, 0.41, 0.52], Ag: [0.75, 0.75, 0.75], Cd: [1, 0.85, 0.56],
  In: [0.65, 0.46, 0.45], Sn: [0.4, 0.5, 0.5], Sb: [0.62, 0.39, 0.71], Te: [0.83, 0.48, 0], I: [0.58, 0, 0.58], Xe: [0.26, 0.62, 0.69], Cs: [0.34, 0.09, 0.56], Ba: [0, 0.79, 0],
  La: [0.44, 0.83, 1], Ce: [1, 1, 0.78], Pr: [0.85, 1, 0.78], Nd: [0.78, 1, 0.78], Sm: [0.56, 1, 0.78], Eu: [0.38, 1, 0.78], Gd: [0.27, 1, 0.78], Tb: [0.19, 1, 0.78], Dy: [0.12, 1, 0.78],
  Ho: [0, 1, 0.61], Er: [0, 0.9, 0.46], Tm: [0, 0.83, 0.32], Yb: [0, 0.75, 0.22], Lu: [0, 0.67, 0.14], Hf: [0.3, 0.76, 1], Ta: [0.3, 0.65, 1], W: [0.13, 0.58, 0.84], Re: [0.15, 0.49, 0.67],
  Os: [0.15, 0.4, 0.59], Ir: [0.09, 0.33, 0.53], Pt: [0.82, 0.82, 0.88], Au: [1, 0.82, 0.14], Hg: [0.72, 0.72, 0.82], Tl: [0.65, 0.33, 0.3], Pb: [0.34, 0.35, 0.38], Bi: [0.62, 0.31, 0.71],
  Th: [0, 0.73, 1], U: [0, 0.56, 1],
};
const RADII: Record<string, number> = {
  H: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71, O: 0.66, F: 0.57, Ne: 0.58, Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02, Ar: 1.06,
  K: 2.03, Ca: 1.76, Sc: 1.7, Ti: 1.6, V: 1.53, Cr: 1.39, Mn: 1.39, Fe: 1.32, Co: 1.26, Ni: 1.24, Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.2, As: 1.19, Se: 1.2, Br: 1.2, Kr: 1.16,
  Rb: 2.2, Sr: 1.95, Y: 1.9, Zr: 1.75, Nb: 1.64, Mo: 1.54, Tc: 1.47, Ru: 1.46, Rh: 1.42, Pd: 1.39, Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39, Sb: 1.39, Te: 1.38, I: 1.39, Xe: 1.4,
  Cs: 2.44, Ba: 2.15, La: 2.07, Ce: 2.04, Pr: 2.03, Nd: 2.01, Sm: 1.98, Eu: 1.98, Gd: 1.96, Tb: 1.94, Dy: 1.92, Ho: 1.92, Er: 1.89, Tm: 1.9, Yb: 1.87, Lu: 1.87, Hf: 1.75,
  Ta: 1.7, W: 1.62, Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36, Au: 1.36, Hg: 1.32, Tl: 1.45, Pb: 1.46, Bi: 1.48, Th: 2.06, U: 1.96,
};

// ---------------------------------------------------------------------------
// CIF text -> tokens, data items, loops
// ---------------------------------------------------------------------------
function tokenize(text: string): string[] {
  const out: string[] = [];
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  let i = 0;
  while (i < lines.length) {
    let line = lines[i];
    if (line.startsWith(";")) {
      // multi-line text field
      const parts = [line.slice(1)];
      i++;
      while (i < lines.length && !lines[i].startsWith(";")) { parts.push(lines[i]); i++; }
      out.push(parts.join("\n"));
      i++;
      continue;
    }
    const hash = line.indexOf("#");
    if (hash >= 0 && !/['"]/.test(line.slice(0, hash))) line = line.slice(0, hash);
    const re = /'([^']*)'|"([^"]*)"|(\S+)/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(line))) out.push(m[1] ?? m[2] ?? m[3]);
    i++;
  }
  return out;
}

interface CifBlock { items: Record<string, string>; loops: Record<string, string[]>[] }

function parseBlocks(text: string): CifBlock {
  const tok = tokenize(text);
  const block: CifBlock = { items: {}, loops: [] };
  let i = 0;
  // use the first data block that has atom sites
  while (i < tok.length) {
    const t = tok[i];
    if (t.toLowerCase().startsWith("data_") || t.toLowerCase() === "stop_" || t.toLowerCase().startsWith("save_")) { i++; continue; }
    if (t.toLowerCase() === "loop_") {
      i++;
      const names: string[] = [];
      while (i < tok.length && tok[i].startsWith("_")) { names.push(tok[i].toLowerCase()); i++; }
      const loop: Record<string, string[]> = {};
      for (const n of names) loop[n] = [];
      let k = 0;
      while (i < tok.length && !tok[i].startsWith("_") && !/^(loop_|data_|stop_|save_)/i.test(tok[i])) {
        loop[names[k % names.length]].push(tok[i]);
        k++; i++;
      }
      block.loops.push(loop);
      continue;
    }
    if (t.startsWith("_")) {
      block.items[t.toLowerCase()] = tok[i + 1] ?? "";
      i += 2;
      continue;
    }
    i++;
  }
  return block;
}

const num = (s: string | undefined, fallback = NaN): number => {
  if (s === undefined) return fallback;
  const v = parseFloat(s.replace(/\(.*\)/, ""));
  return isFinite(v) ? v : fallback;
};

// ---------------------------------------------------------------------------
// symmetry operations "x, y+1/2, -z" -> rotation matrix + translation
// ---------------------------------------------------------------------------
interface SymOp { R: number[][]; t: number[] }

function parseSymop(s: string): SymOp | null {
  const parts = s.toLowerCase().replace(/\s+/g, "").split(",");
  if (parts.length !== 3) return null;
  const R: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const t = [0, 0, 0];
  for (let r = 0; r < 3; r++) {
    // split into signed terms
    const terms = parts[r].replace(/-/g, "+-").split("+").filter(Boolean);
    for (const term of terms) {
      const m = term.match(/^([+-]?\d*\.?\d*(?:\/\d+)?)\*?([xyz])?$/);
      if (!m) return null;
      let coef = 1;
      const cs = m[1];
      if (cs && cs !== "+" && cs !== "-") {
        coef = cs.includes("/") ? parseFloat(cs.split("/")[0]) / parseFloat(cs.split("/")[1]) : parseFloat(cs);
      } else if (cs === "-") coef = -1;
      if (m[2]) R[r]["xyz".indexOf(m[2])] += coef;
      else t[r] += coef;
    }
  }
  return { R, t };
}

// ---------------------------------------------------------------------------
export interface ParsedCif {
  name: string;
  cellpar: [number, number, number, number, number, number];
  cell: number[][]; // rows a, b, c (A), a along x, b in the xy plane
  symbols: string[];
  positions: number[][]; // fractional, full cell
  occupancy: number[];
  spacegroup: string;
  centering: number[][]; // pure translations (centering vectors)
}

export function parseCif(text: string, name = "CIF"): ParsedCif {
  const blk = parseBlocks(text);
  const it = blk.items;
  const cellpar = [it["_cell_length_a"], it["_cell_length_b"], it["_cell_length_c"], it["_cell_angle_alpha"], it["_cell_angle_beta"], it["_cell_angle_gamma"]].map((v) => num(v));
  if (cellpar.some((v) => !isFinite(v))) throw new Error("CIF: missing cell parameters");
  const [a, b, c, al, be, ga] = cellpar;
  const d2r = Math.PI / 180;
  const cosA = Math.cos(al * d2r), cosB = Math.cos(be * d2r), cosG = Math.cos(ga * d2r), sinG = Math.sin(ga * d2r);
  const cx = c * cosB, cy = (c * (cosA - cosB * cosG)) / sinG;
  const cz = Math.sqrt(Math.max(c * c - cx * cx - cy * cy, 0));
  const cell = [[a, 0, 0], [b * cosG, b * sinG, 0], [cx, cy, cz]];

  // atom sites
  const atomLoop = blk.loops.find((l) => "_atom_site_fract_x" in l);
  if (!atomLoop) throw new Error("CIF: no _atom_site_fract_x loop");
  const n = atomLoop["_atom_site_fract_x"].length;
  const typeCol = atomLoop["_atom_site_type_symbol"] || atomLoop["_atom_site_label"];
  const occCol = atomLoop["_atom_site_occupancy"];
  const base: { sym: string; p: number[]; occ: number }[] = [];
  for (let i = 0; i < n; i++) {
    const raw = (typeCol?.[i] || "").replace(/[^A-Za-z]/g, "");
    let sym = raw.slice(0, 2);
    if (!(sym in LOBATO)) sym = raw.slice(0, 1);
    if (!(sym in LOBATO)) throw new Error(`CIF: unknown element "${raw}"`);
    const p = [num(atomLoop["_atom_site_fract_x"][i]), num(atomLoop["_atom_site_fract_y"][i]), num(atomLoop["_atom_site_fract_z"][i])];
    if (p.some((v) => !isFinite(v))) continue;
    base.push({ sym, p, occ: occCol ? num(occCol[i], 1) : 1 });
  }

  // symmetry operations
  const symLoop = blk.loops.find((l) => "_symmetry_equiv_pos_as_xyz" in l || "_space_group_symop_operation_xyz" in l);
  const opStrings = symLoop ? symLoop["_symmetry_equiv_pos_as_xyz"] || symLoop["_space_group_symop_operation_xyz"] : ["x,y,z"];
  const ops = opStrings.map(parseSymop).filter((o): o is SymOp => !!o);
  if (!ops.length) ops.push({ R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], t: [0, 0, 0] });
  const spacegroup = it["_symmetry_space_group_name_h-m"] || it["_space_group_name_h-m_alt"] || (it["_symmetry_int_tables_number"] ? `#${it["_symmetry_int_tables_number"]}` : symLoop ? "" : "P1 (no symmetry operations in file)");

  // expand to the full cell
  const symbols: string[] = [];
  const positions: number[][] = [];
  const occupancy: number[] = [];
  const wrap = (v: number) => ((v % 1) + 1) % 1;
  for (const at of base) {
    for (const op of ops) {
      const q = [0, 1, 2].map((r) => wrap(op.R[r][0] * at.p[0] + op.R[r][1] * at.p[1] + op.R[r][2] * at.p[2] + op.t[r]));
      let dup = false;
      for (let j = 0; j < positions.length; j++) {
        if (symbols[j] !== at.sym) continue;
        const d = positions[j].map((v, k) => Math.abs(wrap(v - q[k] + 0.5) - 0.5));
        if (d.every((v) => v < 1e-3)) { dup = true; break; }
      }
      if (!dup) { symbols.push(at.sym); positions.push(q); occupancy.push(at.occ); }
    }
  }
  // centering translations: from the symmetry list, and also any of the
  // standard centerings that map the expanded atom list onto itself (a P1
  // listing of a conventional cell carries them only implicitly); the
  // reflections they extinguish never carry intensity and are dropped
  const centering = ops.filter((o) => o.R.every((row, r) => row.every((v, cc) => v === (r === cc ? 1 : 0))) && o.t.some((v) => Math.abs(wrap(v)) > 1e-6)).map((o) => o.t.map(wrap));
  const mapsOntoItself = (t: number[]) => positions.every((pos, j) => {
    const q = pos.map((v, k) => wrap(v + t[k]));
    return positions.some((p2, j2) => symbols[j2] === symbols[j] && p2.every((v, k) => Math.abs(wrap(v - q[k] + 0.5) - 0.5) < 1e-3));
  });
  for (const t of [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0], [0.5, 0.5, 0.5], [2 / 3, 1 / 3, 1 / 3], [1 / 3, 2 / 3, 2 / 3]]) {
    if (!centering.some((c) => c.every((v, k) => Math.abs(v - t[k]) < 1e-6)) && mapsOntoItself(t)) centering.push(t);
  }
  return { name, cellpar: cellpar as ParsedCif["cellpar"], cell, symbols, positions, occupancy, spacegroup, centering };
}

// ---------------------------------------------------------------------------
// structure factors -> CrystalData
// ---------------------------------------------------------------------------
function scatteringFactor(sym: string, g: number): number {
  const [a, b] = LOBATO[sym];
  const g2 = g * g;
  let f = 0;
  for (let i = 0; i < 5; i++) {
    const d = 1 + b[i] * g2;
    f += (a[i] * (2 + b[i] * g2)) / (d * d);
  }
  return f;
}

function inv3(m: number[][]): number[][] {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}

const MAX_REFLECTIONS = 30000; // keeps the live solve responsive for large cells

/** k_max (1/A) at which a cell of this volume (primitive lattice points per
 *  centering) reaches the reflection budget, capped at the requested value. */
export function kMaxForCell(p: ParsedCif, kMaxRequested: number): number {
  const cell = p.cell;
  const volume = Math.abs(cell[0][0] * (cell[1][1] * cell[2][2] - cell[1][2] * cell[2][1]) - cell[0][1] * (cell[1][0] * cell[2][2] - cell[1][2] * cell[2][0]) + cell[0][2] * (cell[1][0] * cell[2][1] - cell[1][1] * cell[2][0]));
  const vPrim = volume / (p.centering.length + 1);
  // N(k) = 4/3 pi k^3 V_prim
  const kBudget = Math.cbrt((3 * MAX_REFLECTIONS) / (4 * Math.PI * vPrim));
  return Math.min(kMaxRequested, Math.floor(kBudget * 20) / 20);
}

/** Build the simulator's crystal data from a parsed CIF at an energy and k_max (1/A). */
export function crystalFromCif(p: ParsedCif, energyEv: number, kMaxRequested: number): CrystalData {
  const kMax = kMaxForCell(p, kMaxRequested);
  const cell = p.cell;
  const inv = inv3(cell); // columns of inv = reciprocal vectors; recip rows b_i = inv^T rows
  const recip = [[inv[0][0], inv[1][0], inv[2][0]], [inv[0][1], inv[1][1], inv[2][1]], [inv[0][2], inv[1][2], inv[2][2]]];
  const volume = Math.abs(cell[0][0] * (cell[1][1] * cell[2][2] - cell[1][2] * cell[2][1]) - cell[0][1] * (cell[1][0] * cell[2][2] - cell[1][2] * cell[2][0]) + cell[0][2] * (cell[1][0] * cell[2][1] - cell[1][1] * cell[2][0]));
  const gamma = relativisticGamma(energyEv);
  const species = [...new Set(p.symbols)];
  const nmax = recip.map((b) => Math.ceil(kMax / Math.hypot(b[0], b[1], b[2])) + 1);
  const hkl: number[][] = [];
  const g = new Float32Array(0);
  const gList: number[] = [];
  const F2: number[] = [];
  const Ure: number[] = [];
  const Uim: number[] = [];
  const fCache = new Map<string, number>();
  // extinguished by a centering translation when g . t is not an integer
  const isCentered = (h: number, k: number, l: number) => p.centering.some((t) => { const x = h * t[0] + k * t[1] + l * t[2]; return Math.abs(x - Math.round(x)) > 1e-6; });
  for (let h = -nmax[0]; h <= nmax[0]; h++) for (let k = -nmax[1]; k <= nmax[1]; k++) for (let l = -nmax[2]; l <= nmax[2]; l++) {
    if (h === 0 && k === 0 && l === 0) continue;
    const gx = h * recip[0][0] + k * recip[1][0] + l * recip[2][0];
    const gy = h * recip[0][1] + k * recip[1][1] + l * recip[2][1];
    const gz = h * recip[0][2] + k * recip[1][2] + l * recip[2][2];
    const gl = Math.hypot(gx, gy, gz);
    if (gl > kMax || isCentered(h, k, l)) continue;
    // structure factor F = sum f_j occ_j exp(-2 pi i g.r_j) / V  (quantem convention)
    let re = 0, im = 0;
    const gk = gl.toFixed(5);
    for (const s of species) {
      const key = s + gk;
      let f = fCache.get(key);
      if (f === undefined) { f = scatteringFactor(s, gl); fCache.set(key, f); }
      for (let j = 0; j < p.symbols.length; j++) {
        if (p.symbols[j] !== s) continue;
        const ph = -2 * Math.PI * (h * p.positions[j][0] + k * p.positions[j][1] + l * p.positions[j][2]);
        re += f * p.occupancy[j] * Math.cos(ph);
        im += f * p.occupancy[j] * Math.sin(ph);
      }
    }
    re /= volume; im /= volume;
    hkl.push([h, k, l]);
    gList.push(gx, gy, gz);
    F2.push(re * re + im * im);
    // coupling U = gamma F / pi, plus an absorptive part i * fraction * U (same phase)
    const ur = (gamma * re) / Math.PI, ui = (gamma * im) / Math.PI;
    Ure.push(ur - ABSORPTION_FRACTION * ui);
    Uim.push(ui + ABSORPTION_FRACTION * ur);
  }
  void g;
  let f0 = 0;
  for (let j = 0; j < p.symbols.length; j++) f0 += scatteringFactor(p.symbols[j], 0) * p.occupancy[j];
  const u0 = (gamma * f0) / (Math.PI * volume);
  const couplingRe = new Map<string, number>();
  const couplingIm = new Map<string, number>();
  for (let i = 0; i < hkl.length; i++) {
    couplingRe.set(`${hkl[i][0]},${hkl[i][1]},${hkl[i][2]}`, Ure[i]);
    couplingIm.set(`${hkl[i][0]},${hkl[i][1]},${hkl[i][2]}`, Uim[i]);
  }
  const [a, b, , , , ga] = p.cellpar;
  return {
    name: p.name,
    spacegroup: p.spacegroup,
    pointgroup: "",
    cell,
    recip,
    positions_frac: p.positions,
    numbers: p.symbols.map((s) => Math.max(1, SYMBOLS.indexOf(s))),
    symbols: p.symbols,
    colors: p.symbols.map((s) => JMOL[s] || [0.7, 0.7, 0.7]),
    radii: p.symbols.map((s) => RADII[s] || 1.4),
    hkl,
    g: Float32Array.from(gList),
    F2: Float32Array.from(F2),
    U_re: Float32Array.from(Ure),
    U_im: Float32Array.from(Uim),
    couplingRe,
    couplingIm,
    u0_imag: ABSORPTION_FRACTION * u0,
    absorptive: true,
    energy_ev: energyEv,
    wavelength: electronWavelength(energyEv),
    k_max: kMax,
    hexagonal: Math.abs(a - b) < 1e-3 * a && Math.abs(ga - 120) < 0.05,
  };
}
