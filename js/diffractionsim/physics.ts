/**
 * Diffraction physics for the simulator, all in the browser: reciprocal
 * lattice geometry, kinematical intensities, the Bloch wave calculation
 * (structure matrix, eigendecomposition, thickness dependence, first-order
 * absorption), CBED tilt sampling, Kossel line geometry and the reference
 * pattern lookup.
 */

import { Quat, Vec3, decodeF32, eighComplex, matVec, quatToMatrix } from "./math";

export interface CrystalData {
  name: string;
  spacegroup: string;
  pointgroup: string;
  cell: number[][];
  recip: number[][];
  positions_frac: number[][];
  numbers: number[];
  symbols: string[];
  colors: number[][];
  radii: number[];
  hkl: number[][];
  g: Float32Array; // (N, 3) crystal frame
  F2: Float32Array;
  U_re: Float32Array;
  U_im: Float32Array;
  couplingRe: Map<string, number>;
  couplingIm: Map<string, number>;
  u0_imag: number;
  absorptive: boolean;
  energy_ev: number;
  wavelength: number;
  k_max: number;
  hexagonal: boolean;
}

/** Relativistic electron wavelength (A) for a beam energy in eV. */
export function electronWavelength(energyEv: number): number {
  return 12.2643 / Math.sqrt(energyEv * (1 + 0.97845e-6 * energyEv));
}

/** Relativistic mass factor 1 + E / (m0 c^2). */
export function relativisticGamma(energyEv: number): number {
  return 1 + energyEv / 510998.95;
}

/**
 * Parse the crystal data. With energyEv the stored couplings (computed at
 * the data's energy) are rescaled by the ratio of relativistic mass factors
 * and the wavelength is recomputed: exact for the elastic potential, an
 * approximation for the absorptive part. Kinematical |F|^2 is unchanged.
 */
export function parseCrystal(json: string, energyEv?: number): CrystalData | null {
  if (!json || json === "{}") return null;
  const o = JSON.parse(json);
  const scaleU = energyEv && Math.abs(energyEv - o.energy_ev) > 1 ? relativisticGamma(energyEv) / relativisticGamma(o.energy_ev) : 1;
  if (scaleU !== 1) {
    o.energy_ev = energyEv;
    o.wavelength = electronWavelength(energyEv!);
    o.u0_imag = o.u0_imag * scaleU;
  }
  // reflection indices packed as int16 triplets; g rebuilt from the reciprocal cell
  const bin = atob(o.hkl_i16 || "");
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const hi16 = new Int16Array(bytes.buffer);
  const n = hi16.length / 3;
  const hkl: number[][] = new Array(n);
  const g = new Float32Array(3 * n);
  const B: number[][] = o.recip;
  for (let i = 0; i < n; i++) {
    const h = hi16[3 * i], k = hi16[3 * i + 1], l = hi16[3 * i + 2];
    hkl[i] = [h, k, l];
    g[3 * i] = h * B[0][0] + k * B[1][0] + l * B[2][0];
    g[3 * i + 1] = h * B[0][1] + k * B[1][1] + l * B[2][1];
    g[3 * i + 2] = h * B[0][2] + k * B[1][2] + l * B[2][2];
  }
  const U_re = decodeF32(o.U_re), U_im = decodeF32(o.U_im);
  if (scaleU !== 1) for (let i = 0; i < U_re.length; i++) { U_re[i] *= scaleU; U_im[i] *= scaleU; }
  const couplingRe = new Map<string, number>();
  const couplingIm = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    const key = `${hkl[i][0]},${hkl[i][1]},${hkl[i][2]}`;
    couplingRe.set(key, U_re[i]);
    couplingIm.set(key, U_im[i]);
  }
  return {
    name: o.name, spacegroup: o.spacegroup, pointgroup: o.pointgroup,
    cell: o.cell, recip: o.recip, positions_frac: o.positions_frac, numbers: o.numbers,
    symbols: o.symbols, colors: o.colors, radii: o.radii, hkl,
    g, F2: decodeF32(o.F2), U_re, U_im,
    couplingRe, couplingIm, u0_imag: o.u0_imag, absorptive: o.absorptive,
    energy_ev: o.energy_ev, wavelength: o.wavelength, k_max: o.k_max, hexagonal: o.hexagonal,
  };
}

export interface Reflection {
  index: number;
  hkl: number[];
  g: Vec3; // lab frame
  gLen: number;
  s: number; // excitation error at zero tilt
}

/** Lab-frame reflections within kMax, with excitation errors for the beam along -z. */
export function labReflections(c: CrystalData, q: Quat, kMax: number): Reflection[] {
  const R = quatToMatrix(q);
  const lam = c.wavelength;
  const out: Reflection[] = [];
  const n = c.hkl.length;
  for (let i = 0; i < n; i++) {
    const gc: Vec3 = [c.g[3 * i], c.g[3 * i + 1], c.g[3 * i + 2]];
    const gLen = Math.hypot(gc[0], gc[1], gc[2]);
    if (gLen > kMax) continue;
    const g = matVec(R, gc);
    const g2 = gLen * gLen;
    const s = (2 * g[2] - lam * g2) / (2 - 2 * lam * g[2]);
    out.push({ index: i, hkl: c.hkl[i], g, gLen, s });
  }
  return out;
}

/** Kinematical intensities |F|^2 exp(-s^2 / 2 sigma^2). */
export function kinematicalIntensities(c: CrystalData, refl: Reflection[], sigma: number): Float64Array {
  const out = new Float64Array(refl.length);
  for (let i = 0; i < refl.length; i++) {
    const r = refl[i];
    out[i] = c.F2[r.index] * Math.exp(-(r.s * r.s) / (2 * sigma * sigma));
  }
  return out;
}

export interface BlochSolution {
  beams: Reflection[]; // beam 0 is the direct beam (hkl 000)
  n: number;
  gammaRe: Float64Array; // 1/A
  gammaIm: Float64Array;
  vecRe: Float64Array; // columns = eigenvectors
  vecIm: Float64Array;
  psi0Re: Float64Array; // conj(C_0j)
  psi0Im: Float64Array;
}

const DIRECT: Reflection = { index: -1, hkl: [0, 0, 0], g: [0, 0, 0], gLen: 0, s: 0 };

function tiltedExcitation(r: Reflection, kz: number, tilt: [number, number]): number {
  const g2 = r.gLen * r.gLen;
  const num = 2 * kz * r.g[2] - 2 * (tilt[0] * r.g[0] + tilt[1] * r.g[1]) - g2;
  return num / (2 * (kz - r.g[2]));
}

/** |U_h| for an index difference, zero when the crystal carries no such factor. */
function couplingMag(c: CrystalData, h: number, k: number, l: number): number {
  const key = `${h},${k},${l}`;
  return Math.hypot(c.couplingRe.get(key) ?? 0, c.couplingIm.get(key) ?? 0);
}

const N_STRONG = 24; // candidates treated as the intermediate beams of a two-step path

/**
 * Rank candidates for the Bloch set, strongest first.
 *
 * Ranking by |U_g| alone drops exactly the reflections a dynamical
 * calculation exists to show: silicon 002 and 222 have no structure factor
 * of their own, so they score zero, yet they are the textbook example of a
 * beam that fills by double diffraction and grows with thickness. A beam is
 * worth keeping when it is strongly coupled to something that is itself
 * strongly excited, so each candidate is scored by the largest coupling
 * joining it either to the transmitted beam (|U_g|) or to one of the
 * strongest candidates (|U_(g-h)|), divided by its excitation error.
 */
function rankCandidates(c: CrystalData, cand: { ref: Reflection; s: number }[]): { ref: Reflection; s: number }[] {
  const direct = cand.map((x) => ({
    x,
    v: Math.hypot(c.U_re[x.ref.index], c.U_im[x.ref.index]) / (Math.abs(x.s) + 1e-4),
  }));
  direct.sort((a, b) => b.v - a.v);
  const strong = direct.slice(0, Math.min(N_STRONG, direct.length)).map((d) => d.x.ref.hkl);
  const scored = cand.map((x) => {
    let best = Math.hypot(c.U_re[x.ref.index], c.U_im[x.ref.index]);
    for (const h of strong) {
      const u = couplingMag(c, x.ref.hkl[0] - h[0], x.ref.hkl[1] - h[1], x.ref.hkl[2] - h[2]);
      if (u > best) best = u;
    }
    return { x, v: best / (Math.abs(x.s) + 1e-4) };
  });
  scored.sort((a, b) => b.v - a.v);
  return scored.map((sc) => sc.x);
}

/**
 * Beams entering the Bloch calculation: the direct beam plus every
 * reflection within sgMax of the Ewald sphere at zero tilt (sgMax widened
 * by the tilt range when a convergent beam is sampled), capped at maxBeams
 * by the ranking of rankCandidates.
 */
export function selectBeams(c: CrystalData, q: Quat, kMax: number, sgMax: number, maxBeams: number, tiltRange = 0): Reflection[] {
  const all = labReflections(c, q, kMax);
  const widen = tiltRange * kMax; // max |delta s| = sin(alpha) |g|
  let beams = all.filter((r) => Math.abs(r.s) < sgMax + widen);
  if (beams.length > maxBeams) {
    beams = rankCandidates(c, beams.map((r) => ({ ref: r, s: r.s })))
      .slice(0, maxBeams)
      .map((x) => x.ref);
  }
  return [DIRECT, ...beams];
}

/**
 * Bloch wave eigenproblem for a fixed beam list and an incident beam with
 * in-plane wavevector tilt (1/A). The structure matrix has 2 k0 s_g on the
 * diagonal and the couplings U_(g-h) off it; the Hermitian part is
 * diagonalized and the absorption enters to first order (the
 * fast_absorption path of quantem).
 */
export function blochSolve(c: CrystalData, beams: Reflection[], tilt: [number, number] = [0, 0]): BlochSolution {
  const k0 = 1 / c.wavelength;
  const kz = Math.sqrt(Math.max(k0 * k0 - tilt[0] ** 2 - tilt[1] ** 2, 1e-12));
  const n = beams.length;
  const sg = beams.map((r) => (r.index < 0 ? 0 : tiltedExcitation(r, kz, tilt)));
  const hRe = new Float64Array(n * n);
  const hIm = new Float64Array(n * n);
  const wRe = new Float64Array(n * n);
  const wIm = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    hRe[i * n + i] = 2 * k0 * sg[i];
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const d0 = beams[i].hkl[0] - beams[j].hkl[0], d1 = beams[i].hkl[1] - beams[j].hkl[1], d2 = beams[i].hkl[2] - beams[j].hkl[2];
      const ur = c.couplingRe.get(`${d0},${d1},${d2}`) ?? 0;
      const ui = c.couplingIm.get(`${d0},${d1},${d2}`) ?? 0;
      const tr = c.couplingRe.get(`${-d0},${-d1},${-d2}`) ?? 0;
      const ti = c.couplingIm.get(`${-d0},${-d1},${-d2}`) ?? 0;
      // A_ij = U_(gi - gj) = (ur, ui); conj(A_ji) = (tr, -ti)
      // H = (A + A^dagger) / 2, W = (A - H) / i
      const hr = 0.5 * (ur + tr), hi = 0.5 * (ui - ti);
      hRe[i * n + j] = hr;
      hIm[i * n + j] = hi;
      wRe[i * n + j] = ui - hi;
      wIm[i * n + j] = -(ur - hr);
    }
  }
  const { vals, vecRe, vecIm } = eighComplex(hRe, hIm, n);
  const gammaRe = new Float64Array(n);
  const gammaIm = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    gammaRe[j] = vals[j] / (2 * k0);
    let acc = 0;
    for (let i = 0; i < n; i++) {
      let sr = 0, si = 0;
      for (let k = 0; k < n; k++) {
        const wr = wRe[i * n + k], wi = wIm[i * n + k];
        if (wr === 0 && wi === 0) continue;
        const cr = vecRe[k * n + j], ci = vecIm[k * n + j];
        sr += wr * cr - wi * ci;
        si += wr * ci + wi * cr;
      }
      acc += vecRe[i * n + j] * sr + vecIm[i * n + j] * si;
    }
    gammaIm[j] = (acc + c.u0_imag) / (2 * k0);
  }
  const psi0Re = new Float64Array(n);
  const psi0Im = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    psi0Re[j] = vecRe[j];
    psi0Im[j] = -vecIm[j];
  }
  return { beams, n, gammaRe, gammaIm, vecRe, vecIm, psi0Re, psi0Im };
}

/**
 * Beam list for the hybrid pattern: the direct beam, then the Bloch set
 * (within sgMax of the Ewald sphere, capped), then every other reflection
 * within kMax. The first nDyn beams enter the Bloch calculation; the rest
 * take thin-slab intensities (slabIntensities).
 */
export function hybridBeams(c: CrystalData, q: Quat, kMax: number, sgMax: number, maxBeams: number, tiltRange = 0): { beams: Reflection[]; nDyn: number } {
  const dyn = selectBeams(c, q, kMax, sgMax, maxBeams, tiltRange);
  const inDyn = new Set<number>();
  for (const b of dyn) inDyn.add(b.index);
  const rest = labReflections(c, q, kMax).filter((r) => !inDyn.has(r.index));
  return { beams: [...dyn, ...rest], nDyn: dyn.length };
}

/**
 * Thin-slab (first Born) intensities I_g = (pi |U_g| t / k0)^2 sinc^2(pi s_g t)
 * for beams[from..] at an incident tilt: the weak-beam limit of the Bloch
 * result, in the same normalization (fraction of the incident intensity).
 */
export function slabIntensities(c: CrystalData, beams: Reflection[], from: number, tilt: [number, number], thickness: number, out: Float64Array) {
  const k0 = 1 / c.wavelength;
  const kz = Math.sqrt(Math.max(k0 * k0 - tilt[0] ** 2 - tilt[1] ** 2, 1e-12));
  for (let i = from; i < beams.length; i++) {
    const r = beams[i];
    const s = tilt[0] === 0 && tilt[1] === 0 ? r.s : tiltedExcitation(r, kz, tilt);
    const u = Math.hypot(c.U_re[r.index], c.U_im[r.index]);
    const x = Math.PI * s * thickness;
    const sinc = Math.abs(x) < 1e-8 ? 1 : Math.sin(x) / x;
    const amp = (Math.PI * u * thickness) / k0;
    out[i] = amp * amp * sinc * sinc;
  }
}

/** Kinematical intensities of a beam list at an incident tilt. */
export function kinematicalTilted(c: CrystalData, beams: Reflection[], tilt: [number, number], sigma: number): Float64Array {
  const k0 = 1 / c.wavelength;
  const kz = Math.sqrt(Math.max(k0 * k0 - tilt[0] ** 2 - tilt[1] ** 2, 1e-12));
  const out = new Float64Array(beams.length);
  for (let i = 0; i < beams.length; i++) {
    const r = beams[i];
    if (r.index < 0) { out[i] = 1; continue; }
    const s = tiltedExcitation(r, kz, tilt);
    out[i] = c.F2[r.index] * Math.exp(-(s * s) / (2 * sigma * sigma));
  }
  return out;
}

/**
 * Incident-beam tilts (1/A) sampling a precession cone of half angle
 * precDeg: n points evenly spaced on the ring of radius k0 sin(phi), the
 * Gauss-Chebyshev quadrature of the azimuthal average. No precession
 * returns the single untilted beam.
 */
export function precessionTilts(k0: number, precDeg: number, n: number): [number, number][] {
  if (!(precDeg > 0) || n < 1) return [[0, 0]];
  const r = k0 * Math.sin((precDeg * Math.PI) / 180);
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const th = (2 * Math.PI * (i + 0.5)) / n;
    out.push([r * Math.cos(th), r * Math.sin(th)]);
  }
  return out;
}

/**
 * Hybrid nanobeam intensities averaged over incident tilts: Bloch
 * intensities of the first nDyn beams from one solution per tilt, thin-slab
 * intensities for the rest, mean over the tilts.
 */
export function averagedIntensities(
  c: CrystalData, beams: Reflection[], nDyn: number, sols: (BlochSolution | null)[], tilts: [number, number][], thickness: number,
): Float64Array {
  const out = new Float64Array(beams.length);
  const tmp = new Float64Array(beams.length);
  for (let t = 0; t < tilts.length; t++) {
    tmp.fill(0);
    const sol = sols[t];
    if (sol) tmp.set(blochIntensities(sol, thickness));
    slabIntensities(c, beams, sol ? nDyn : 0, tilts[t], thickness, tmp);
    for (let i = 0; i < beams.length; i++) out[i] += tmp[i] / tilts.length;
  }
  return out;
}

export interface NanobeamSolution {
  beams: Reflection[]; // DIRECT + every reflection within kMax (draw list)
  nodes: { tilt: [number, number]; sol: BlochSolution; pos: Int32Array }[]; // pos: position of each Bloch beam in beams
  nDynMean: number;
}

/**
 * Nanobeam pattern averaged over incident tilts (precession ring, or the
 * single untilted beam). Each node selects its own Bloch set from the
 * reflections within sgMax of ITS Ewald sphere (capped at maxBeams by
 * |U_g| / |s_g|), so a 3 degree precession cone excites the right beams at
 * every azimuth; every other reflection takes the thin-slab intensity.
 */
export function nanobeamSolve(c: CrystalData, q: Quat, kMax: number, sgMax: number, maxBeams: number, tilts: [number, number][]): NanobeamSolution {
  const k0 = 1 / c.wavelength;
  const all = labReflections(c, q, kMax);
  const beams = [DIRECT, ...all];
  const nodes: NanobeamSolution["nodes"] = [];
  let nSum = 0;
  for (const tilt of tilts) {
    const kz = Math.sqrt(Math.max(k0 * k0 - tilt[0] ** 2 - tilt[1] ** 2, 1e-12));
    let cand: { ref: Reflection; s: number; i: number }[] = [];
    for (let i = 0; i < all.length; i++) {
      const st = tiltedExcitation(all[i], kz, tilt);
      if (Math.abs(st) < sgMax) cand.push({ ref: all[i], s: st, i });
    }
    if (cand.length > maxBeams) {
      cand = rankCandidates(c, cand).slice(0, maxBeams) as typeof cand;
    }
    const dyn = [DIRECT, ...cand.map((x) => x.ref)];
    const pos = new Int32Array(dyn.length);
    pos[0] = 0;
    cand.forEach((x, j) => { pos[j + 1] = x.i + 1; });
    nodes.push({ tilt, sol: blochSolve(c, dyn, tilt), pos });
    nSum += cand.length;
  }
  return { beams, nodes, nDynMean: nSum / Math.max(1, tilts.length) };
}

/** Intensities of a NanobeamSolution at a thickness (A): mean over the nodes. */
export function nanobeamIntensities(c: CrystalData, ns: NanobeamSolution, thickness: number): Float64Array {
  const out = new Float64Array(ns.beams.length);
  const tmp = new Float64Array(ns.beams.length);
  for (const node of ns.nodes) {
    slabIntensities(c, ns.beams, 1, node.tilt, thickness, tmp); // every diffracted beam, then overwrite the Bloch ones
    tmp[0] = 0;
    const bi = blochIntensities(node.sol, thickness);
    for (let j = 0; j < node.pos.length; j++) tmp[node.pos[j]] = bi[j];
    for (let i = 0; i < out.length; i++) out[i] += tmp[i] / ns.nodes.length;
  }
  return out;
}

/** Beam intensities at a thickness (A) from a Bloch solution. */
export function blochIntensities(sol: BlochSolution, thickness: number): Float64Array {
  const { n, gammaRe, gammaIm, vecRe, vecIm, psi0Re, psi0Im } = sol;
  const phRe = new Float64Array(n);
  const phIm = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const amp = Math.exp(-2 * Math.PI * gammaIm[j] * thickness);
    const ph = 2 * Math.PI * gammaRe[j] * thickness;
    const er = amp * Math.cos(ph), ei = amp * Math.sin(ph);
    // e^{2 pi i gamma t} * conj(C_0j)
    phRe[j] = er * psi0Re[j] - ei * psi0Im[j];
    phIm[j] = er * psi0Im[j] + ei * psi0Re[j];
  }
  const out = new Float64Array(n);
  for (let g = 0; g < n; g++) {
    let sr = 0, si = 0;
    for (let j = 0; j < n; j++) {
      const cr = vecRe[g * n + j], ci = vecIm[g * n + j];
      sr += cr * phRe[j] - ci * phIm[j];
      si += cr * phIm[j] + ci * phRe[j];
    }
    out[g] = sr * sr + si * si;
  }
  return out;
}

export interface KosselLine {
  hkl: number[];
  normal: [number, number]; // unit, in the tilt plane (rad)
  distance: number; // rad, line is {theta : theta . normal = distance}
  width: number; // rad
  strength: number; // relative 0..1
  gxy: number; // |g_xy|, 1/A (excess Kikuchi line offset)
}

/**
 * Kossel (deficiency) lines of every reflection in the tilt plane:
 * g_xy . theta = g_z - lambda |g|^2 / 2, a straight line at small angles.
 * The width is the two-beam rocking width |U_g| / (k0 |g_xy|).
 */
/** Coupling of a strong reflection (Si 111 at 200 keV is 0.05 1/A^2): line darkness is measured against it. */
export const U_REF_LINES = 0.04;
const MAX_LINES = 200; // strongest lines kept: enough for the rosettes of small cells, readable for large ones

export function kosselLines(c: CrystalData, q: Quat, kMax: number, fieldRad: number): KosselLine[] {
  const R = quatToMatrix(q);
  const lam = c.wavelength;
  const k0 = 1 / lam;
  const lines: KosselLine[] = [];
  // strength is ABSOLUTE (|U_g| / U_REF_LINES, capped at 1): a weakly
  // scattering cell gives faint lines, and only the strongest MAX_LINES are
  // kept so a large cell does not draw a hundred thousand of them
  const uMin = 0.02 * U_REF_LINES;
  for (let i = 0; i < c.hkl.length; i++) {
    const gc: Vec3 = [c.g[3 * i], c.g[3 * i + 1], c.g[3 * i + 2]];
    const gLen = Math.hypot(gc[0], gc[1], gc[2]);
    if (gLen > kMax) continue;
    const g = matVec(R, gc);
    const gxy = Math.hypot(g[0], g[1]);
    if (gxy < 1e-6) continue;
    const dist = (g[2] - (lam * gLen * gLen) / 2) / gxy;
    if (Math.abs(dist) > fieldRad * 1.5) continue;
    const u = Math.hypot(c.U_re[i], c.U_im[i]);
    if (u < uMin) continue;
    lines.push({ hkl: c.hkl[i], normal: [g[0] / gxy, g[1] / gxy], distance: dist, width: u / (k0 * gxy), strength: Math.min(1, u / U_REF_LINES), gxy });
  }
  if (lines.length > MAX_LINES) {
    lines.sort((a, b) => b.strength - a.strength);
    lines.length = MAX_LINES;
  }
  return lines;
}

export interface KosselReference {
  shape: number[]; // (T, n, n)
  step: number;
  thicknesses: number[];
  data: Float32Array;
}

export function parseKossel(json: string): KosselReference | null {
  if (!json || json === "{}") return null;
  const o = JSON.parse(json);
  return { shape: o.shape, step: o.step, thicknesses: o.thicknesses, data: decodeF32(o.data) };
}

/** Bright field Kossel image (size x size) over +-fieldRad by lookup in the reference. */
export function kosselLookup(ref: KosselReference, q: Quat, fieldRad: number, size: number, thickness: number, viewX = 1): Float32Array {
  const R = quatToMatrix(q);
  const [T, n] = ref.shape;
  const half = (n - 1) / 2;
  // nearest thickness with linear blend
  let ti = 0;
  for (let k = 0; k < T; k++) if (Math.abs(ref.thicknesses[k] - thickness) < Math.abs(ref.thicknesses[ti] - thickness)) ti = k;
  const plane = ti * n * n;
  const out = new Float32Array(size * size);
  for (let r = 0; r < size; r++) {
    const ty = -((r + 0.5) / size - 0.5) * 2 * fieldRad; // canvas rows run downward
    for (let col = 0; col < size; col++) {
      const tx = viewX * ((col + 0.5) / size - 0.5) * 2 * fieldRad;
      if (tx * tx + ty * ty > fieldRad * fieldRad) { out[r * size + col] = NaN; continue; }
      // incident wavevector is (t, -kz) in quantem (beam travels along -z); the
      // reference is indexed by the anti-propagation direction (-t, +kz)/k0
      const dl: Vec3 = [-tx, -ty, Math.sqrt(Math.max(1 - tx * tx - ty * ty, 0))];
      let dc: Vec3 = [R[0] * dl[0] + R[3] * dl[1] + R[6] * dl[2], R[1] * dl[0] + R[4] * dl[1] + R[7] * dl[2], R[2] * dl[0] + R[5] * dl[1] + R[8] * dl[2]];
      if (dc[2] < 0) dc = [-dc[0], -dc[1], -dc[2]];
      const rho = Math.sqrt(Math.max(2 * (1 - dc[2]), 0));
      const dxy = Math.max(Math.hypot(dc[0], dc[1]), 1e-12);
      const fx = (dc[0] / dxy) * rho / ref.step + half;
      const fy = (dc[1] / dxy) * rho / ref.step + half;
      const ix = Math.min(Math.max(Math.floor(fx), 0), n - 2), iy = Math.min(Math.max(Math.floor(fy), 0), n - 2);
      const wx = Math.min(Math.max(fx - ix, 0), 1), wy = Math.min(Math.max(fy - iy, 0), 1);
      const v = ref.data[plane + ix * n + iy] * (1 - wx) * (1 - wy) + ref.data[plane + (ix + 1) * n + iy] * wx * (1 - wy)
        + ref.data[plane + ix * n + iy + 1] * (1 - wx) * wy + ref.data[plane + (ix + 1) * n + iy + 1] * wx * wy;
      out[r * size + col] = v;
    }
  }
  return out;
}

/** Percentile-based display range of finite values. */
export function percentiles(data: Float32Array | Float64Array, pLow: number, pHigh: number): [number, number] {
  const vals: number[] = [];
  for (let i = 0; i < data.length; i++) if (isFinite(data[i])) vals.push(data[i]);
  if (!vals.length) return [0, 1];
  vals.sort((a, b) => a - b);
  const lo = vals[Math.min(vals.length - 1, Math.floor((pLow / 100) * (vals.length - 1)))];
  const hi = vals[Math.min(vals.length - 1, Math.floor((pHigh / 100) * (vals.length - 1)))];
  return hi > lo ? [lo, hi] : [lo, lo + 1e-12];
}
