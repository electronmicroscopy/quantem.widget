// diffraction-sim.js: interactive electron diffraction simulator for web
// pages (MyST anywidget directive, no framework). Built from the
// quantem.widget sources in js/diffractionsim-web/ and js/diffractionsim/;
// `npm run build:web` writes dist/web/diffraction-sim.js, a single ESM file
// with `export default { render }`. `npm run build` does not build it.
//
// Left panel: the unit cell, drawn as seen from the detector side (the beam
// comes toward you). Drag to tilt, two fingers to twist, buttons for 15 deg
// steps. Right panel: the diffraction pattern of the same orientation,
// computed live in the browser: nanobeam disks (Bloch wave intensities for
// the beams near the Ewald sphere, thin-slab intensities for the rest),
// CBED disks, or the Kikuchi (Kossel) line pattern. Drag the pattern to move
// the tilt map with the pointer; double-click a point to tilt the crystal by
// that angle (the clicked direction moves onto the optic axis).
//
// Directive options (all optional), named like the DiffractionSim traits:
//   preset (structure name), zone_axis [u,v,w] or [u,v,t,w], thickness_A, semiangle_mrad, precession_deg,
//   pattern_range_inv_A, mode ("nanobeam" | "cbed" | "kikuchi", default "cbed"; "kossel" is accepted
//   as the earlier name of "kikuchi"), dynamical, panel_width_px (px per panel), show_labels (cell
//   axes), show_hkl, show_kikuchi (default true), polyhedra, n_cells, power (brightness exponent,
//   default 0.5 = square root of the intensity), vmax (upper end of the contrast window, default 0.5
//   of the strongest diffracted beam), energy_keV, field_mrad, rotation_speed_deg_per_s, show_ewald.
// Earlier option names are still read: semiconv_mrad, pattern_range, size, energy_ev,
// rotation_speed_deg.
//
// The structure menu is js/diffractionsim-web/presets.ts, written by
// scripts/diffraction_sim_presets.py.

import { Quat, Vec3, directionIndices, fourToThree, matTVec, parseDirection, qmult, qnormalize, quatFromAxisAngle, quatFromZoneAxis, quatToMatrix, threeToFour } from "../diffractionsim/math";
import {
  CrystalData, NanobeamSolution, Reflection, blochIntensities, blochSolve, hybridBeams, kinematicalTilted, kosselLines,
  labReflections, nanobeamIntensities, nanobeamSolve, parseCrystal, precessionTilts, slabIntensities,
} from "../diffractionsim/physics";
import { cellGeometry, drawCell } from "../diffractionsim/crystal3d";
import {
  Frame, cbedImage, drawDisks, drawEwaldPanel, drawImage, drawKikuchiOverlay, drawKosselLines, setupCanvas, tiltGrid, toPx,
} from "../diffractionsim/pattern";
import { normalizeMode } from "../diffractionsim/mode";
import { ENERGY_EV, PRESETS } from "./presets";
import { ParsedCif, crystalFromCif, parseCif } from "./cif";

const DIRECT: Reflection = { index: -1, hkl: [0, 0, 0], g: [0, 0, 0], gLen: 0, s: 0 };
const SG_MAX = 0.05;
const VIEW_X = -1; // detector-side view: cell and pattern move together
const MAX_BEAMS = 48;
const MAX_BEAMS_DRAG = 28;
const SPIN_FPS = 20; // recompute rate while spinning (battery)
const CBED_GRID = 7;
const CBED_GRID_DRAG = 5;
const CBED_PREC_NODES = 8; // ring nodes per incident direction of the cone

interface Model { get(key: string): unknown }

function fmtIndices(v: [number, number, number] | null, hexagonal = false): string {
  if (!v) return "—";
  const idx: number[] = hexagonal ? threeToFour(v) : v;
  return "[" + idx.map((h) => (h < 0 ? `${-h}̅` : `${h}`)).join("") + "]";
}

function detectDark(): boolean {
  const de = document.documentElement;
  if (de.classList.contains("dark")) return true;
  if (de.classList.contains("light")) return false;
  try {
    const m = getComputedStyle(document.body).backgroundColor.match(/\d+/g);
    if (m && m.length >= 3) return (0.299 * +m[0] + 0.587 * +m[1] + 0.114 * +m[2]) / 255 < 0.5;
  } catch (e) { /* ignore */ }
  return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
}

export function render({ model, el }: { model: Model; el: HTMLElement }) {
  const opt = <T,>(key: string, fallback: T): T => {
    const v = model && typeof model.get === "function" ? (model.get(key) as T | undefined) : undefined;
    return v === undefined || v === null ? fallback : v;
  };
  const id = "dsim-" + Math.random().toString(36).slice(2, 8);
  const names = Object.keys(PRESETS);

  // ---------------------------------------------------------------- state
  const state = {
    preset: names.includes(opt("preset", "")) ? opt("preset", "") : names[0],
    mode: normalizeMode(opt("mode", "cbed"), "cbed") as string,
    dynamical: opt("dynamical", true) as boolean,
    thickness: opt("thickness_A", 400) as number,
    semiconv: opt("semiangle_mrad", opt("semiconv_mrad", 3)) as number,
    precession: opt("precession_deg", 0) as number,
    fieldMrad: opt("field_mrad", 50) as number,
    patternRange: opt("pattern_range_inv_A", opt("pattern_range", 3.0)) as number,
    stepDeg: 15,
    spinSpeed: opt("rotation_speed_deg_per_s", opt("rotation_speed_deg", 6)) as number, // deg/s for the continuous rotation buttons
    spinX: false, // continuous slow rotation about the screen x axis (vertical motion)
    spinY: false, // ... about the screen y axis (horizontal motion)
    showLabels: opt("show_labels", true) as boolean,
    showHkl: opt("show_hkl", true) as boolean,
    showAppearance: false,
    kikuchi: opt("show_kikuchi", true) as boolean,
    polyhedra: opt("polyhedra", false) as boolean,
    nCells: (opt("n_cells", [1, 1, 1]) as number[]).slice(0, 3) as [number, number, number],
    sizePref: opt("panel_width_px", opt("size", 400)) as number,
    power: opt("power", 0.5) as number, // brightness ~ intensity^power
    vmin: 0, // contrast window on the scaled intensities (1 = strongest diffracted beam)
    vmax: opt("vmax", 0.5) as number,
    energy: opt("energy_keV", NaN) > 0 ? opt("energy_keV", 0) * 1e3 : (opt("energy_ev", ENERGY_EV) as number),
    showEwald: opt("show_ewald", true) as boolean,
    quat: [1, 0, 0, 0] as Quat,
    dragging: false,
    spinning: false,
  };
  const ENERGIES = [60e3, 80e3, 100e3, 120e3, 200e3, 300e3];
  if (!ENERGIES.includes(state.energy)) ENERGIES.push(state.energy);
  ENERGIES.sort((a, b) => a - b);
  const cifs = new Map<string, ParsedCif>(); // structures loaded from CIF files, by menu name
  const loadCrystal = (name: string, energy: number): CrystalData => {
    const cif = cifs.get(name);
    if (cif) return crystalFromCif(cif, energy, 3.0);
    return parseCrystal(PRESETS[name], energy)!;
  };
  let crystal: CrystalData = loadCrystal(state.preset, state.energy);
  let geom = cellGeometry(crystal, state.nCells, state.polyhedra);
  const k0 = () => 1 / crystal.wavelength;

  const setZoneAxis = (uvw: Vec3) => {
    const c = crystal.cell;
    const d: Vec3 = [
      uvw[0] * c[0][0] + uvw[1] * c[1][0] + uvw[2] * c[2][0],
      uvw[0] * c[0][1] + uvw[1] * c[1][1] + uvw[2] * c[2][1],
      uvw[0] * c[0][2] + uvw[1] * c[1][2] + uvw[2] * c[2][2],
    ];
    state.quat = quatFromZoneAxis(d);
  };
  const za = opt("zone_axis", null) as number[] | null;
  if (za && za.length === 4) setZoneAxis(fourToThree(za)); // Miller-Bravais [u v t w]
  else if (za && za.length === 3) setZoneAxis(za as Vec3);
  else setZoneAxis(crystal.hexagonal ? [0, 0, 1] : [1, 1, 0]);

  // ---------------------------------------------------------------- DOM
  const style = document.createElement("style");
  style.textContent = `
    .${id}-wrap { background: var(--${id}-bg, #111); color: var(--${id}-fg, #ccc); border: 1px solid var(--${id}-border, #444);
      border-radius: 8px; padding: 10px 14px 8px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 13px; max-width: 100%; box-sizing: border-box; }
    .${id}-titlerow { position: relative; display: flex; align-items: center; justify-content: center; margin-bottom: 6px; min-height: 28px; }
    .${id}-title { font-size: 16px; font-weight: 600; color: var(--${id}-title, #ddd); text-align: center; }
    .${id}-titlerow .${id}-btn { position: absolute; right: 0; top: 50%; transform: translateY(-50%); }
    .${id}-file { display: none; }
    .${id}-top { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; justify-content: center; margin-bottom: 8px; }
    .${id}-panels { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; }
    .${id}-panel { display: flex; flex-direction: column; gap: 6px; min-width: 0; flex: 0 0 auto; }
    .${id}-ctl { display: flex; flex-direction: column; gap: 6px; }
    .${id}-panels.stacked { flex-direction: column; flex-wrap: nowrap; align-items: center; }
    .${id}-panels.stacked .${id}-panel { display: contents; }
    .${id}-panels.stacked #${id}-cell { order: 1; }
    .${id}-panels.stacked #${id}-pat { order: 2; }
    .${id}-panels.stacked #${id}-patctl { order: 3; }
    .${id}-panels.stacked #${id}-cellctl { order: 4; }
    .${id}-panels.stacked #${id}-ewaldc { order: 5; }
    .${id}-canvas { display: block; border-radius: 4px; touch-action: none; cursor: grab; background: #000; }
    .${id}-canvas.cell { background: var(--${id}-cellbg, #161616); border: 1px solid var(--${id}-border, #444); }
    .${id}-row { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; }
    .${id}-hint { font-size: 11px; color: var(--${id}-dim, #777); line-height: 1.35; }
    .${id}-mono { font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
    .${id}-btn { padding: 4px 9px; border: 1px solid var(--${id}-border, #444); background: var(--${id}-btnbg, #222);
      color: var(--${id}-fg, #ccc); border-radius: 4px; cursor: pointer; font-size: 12px; line-height: 1.2; }
    .${id}-btn:hover { background: var(--${id}-btnhover, #333); }
    .${id}-btn.active { background: #1a4d2e; border-color: #00cc66; color: #00ff88; }
    .${id}-group { display: inline-flex; }
    .${id}-group .${id}-btn { border-radius: 0; margin-left: -1px; }
    .${id}-group .${id}-btn:first-child { border-radius: 4px 0 0 4px; margin-left: 0; }
    .${id}-group .${id}-btn:last-child { border-radius: 0 4px 4px 0; }
    .${id}-wrap select, .${id}-wrap input[type=text], .${id}-wrap input[type=number] { font-size: 12px; padding: 3px 6px; border-radius: 4px;
      border: 1px solid var(--${id}-border, #444); background: var(--${id}-btnbg, #222); color: var(--${id}-fg, #ccc); }
    .${id}-slider { flex: 1; min-width: 130px; max-width: 200px; }
    .${id}-slider label { display: flex; justify-content: space-between; font-size: 11px; color: var(--${id}-label, #aaa); }
    .${id}-slider input[type=range] { width: 100%; accent-color: #00cc66; margin: 2px 0 0; }
    .${id}-check { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; }
    .${id}-check input { accent-color: #00cc66; }
    .${id}-appearance { padding: 6px 8px; border: 1px solid var(--${id}-border, #444); border-radius: 6px; }
    .${id}-hist { display: block; border: 1px solid var(--${id}-border, #444); border-radius: 3px; cursor: ew-resize; touch-action: none; }
    .${id}-histwrap { display: flex; flex-direction: column; gap: 2px; }
    .${id}-histwrap .${id}-hint { display: flex; justify-content: space-between; font-family: ui-monospace, Menlo, monospace; }
  `;
  el.appendChild(style);

  const wrap = document.createElement("div");
  wrap.className = `${id}-wrap`;
  const presetOptions = names.map((n) => `<option value="${n}"${n === state.preset ? " selected" : ""}>${n}</option>`).join("");
  wrap.innerHTML = `
    <div class="${id}-titlerow">
      <div class="${id}-title">Electron diffraction simulator</div>
      <div class="${id}-btn" id="${id}-cifbtn" title="Load a crystal structure from a CIF file">load CIF…</div>
      <input type="file" class="${id}-file" id="${id}-ciffile" accept=".cif,.CIF,text/plain">
    </div>
    <div class="${id}-top">
      <label>structure <select id="${id}-preset">${presetOptions}</select></label>
      <label>zone axis <input type="text" id="${id}-zone" size="8" placeholder="1 1 0"></label>
      <div class="${id}-btn" id="${id}-go">go</div>
      <div class="${id}-btn" id="${id}-reset">reset</div>
      <label>energy <select id="${id}-energy">${ENERGIES.map((e) => `<option value="${e}"${e === state.energy ? " selected" : ""}>${(e / 1e3).toFixed(0)} keV</option>`).join("")}</select></label>
    </div>
    <div class="${id}-panels">
      <div class="${id}-panel">
        <canvas class="${id}-canvas cell" id="${id}-cell"></canvas>
        <canvas class="${id}-canvas cell" id="${id}-ewaldc" style="cursor:default"></canvas>
        <div class="${id}-ctl" id="${id}-cellctl">
        <div class="${id}-row">
          <div class="${id}-group"><div class="${id}-btn" data-rot="x-">x −</div><div class="${id}-btn" data-rot="x+">x +</div></div>
          <div class="${id}-group"><div class="${id}-btn" data-rot="y-">y −</div><div class="${id}-btn" data-rot="y+">y +</div></div>
          <div class="${id}-group"><div class="${id}-btn" data-rot="z-">z −</div><div class="${id}-btn" data-rot="z+">z +</div></div>
          <input type="number" id="${id}-step" value="15" min="0.1" max="180" step="1" style="width:46px"> <span class="${id}-hint">°</span>
          <div class="${id}-group"><div class="${id}-btn" data-spin="y" title="rotate slowly, horizontally (click again to stop)">↔</div><div class="${id}-btn" data-spin="x" title="rotate slowly, vertically (click again to stop)">↕</div></div>
          <input type="range" id="${id}-spin" min="1" max="30" step="1" value="${state.spinSpeed}" style="width:64px" title="rotation speed of the ↔ ↕ buttons"> <span class="${id}-hint ${id}-mono" id="${id}-spinv">${state.spinSpeed}°/s</span>
        </div>
        <div class="${id}-row">
          <span id="${id}-info" class="${id}-hint"></span>
          <span id="${id}-za" class="${id}-mono"></span>
        </div>
        <div class="${id}-row">
          <label class="${id}-check"><input type="checkbox" id="${id}-labels" ${state.showLabels ? "checked" : ""}> labels</label>
          <label class="${id}-check"><input type="checkbox" id="${id}-poly" ${state.polyhedra ? "checked" : ""}> polyhedra</label>
          <label class="${id}-check"><input type="checkbox" id="${id}-ewald" ${state.showEwald ? "checked" : ""}> Ewald sphere</label>
          <label class="${id}-check">cells <input type="number" id="${id}-ncell" value="${state.nCells[0]}" min="1" max="3" step="1" style="width:40px"></label>
        </div>
        <div class="${id}-hint">Drag to tilt the crystal (the near face follows the pointer). Shift-drag, or two fingers, twist about the beam. The beam comes toward you.</div>
        </div>
      </div>
      <div class="${id}-panel">
        <canvas class="${id}-canvas" id="${id}-pat"></canvas>
        <div class="${id}-ctl" id="${id}-patctl">
        <div class="${id}-row">
          <div class="${id}-group" id="${id}-modes">
            <div class="${id}-btn" data-mode="nanobeam">nanobeam</div>
            <div class="${id}-btn" data-mode="cbed">CBED</div>
            <div class="${id}-btn" data-mode="kikuchi">Kikuchi pattern</div>
          </div>
          <label class="${id}-check" id="${id}-dynwrap"><input type="checkbox" id="${id}-dyn" ${state.dynamical ? "checked" : ""}> dynamical</label>
          <label class="${id}-check"><input type="checkbox" id="${id}-hkl" ${state.showHkl ? "checked" : ""}> hkl labels</label>
          <label class="${id}-check" id="${id}-kikwrap"><input type="checkbox" id="${id}-kik" ${state.kikuchi ? "checked" : ""}> Kikuchi lines</label>
        </div>
        <div class="${id}-row">
          <div class="${id}-slider" id="${id}-thickwrap"><label><span>thickness</span><span id="${id}-thick-val"></span></label>
            <input type="range" id="${id}-thick" min="20" max="1500" step="5" value="${state.thickness}"></div>
          <div class="${id}-slider" id="${id}-convwrap"><label><span>convergence semiangle</span><span id="${id}-conv-val"></span></label>
            <input type="range" id="${id}-conv" min="0.2" max="20" step="0.1" value="${state.semiconv}"></div>
          <div class="${id}-slider" id="${id}-precwrap"><label><span>precession angle</span><span id="${id}-prec-val"></span></label>
            <input type="range" id="${id}-prec" min="0" max="3" step="0.05" value="${state.precession}"></div>
          <div class="${id}-slider" id="${id}-rangewrap"><label><span>pattern range</span><span id="${id}-range-val"></span></label>
            <input type="range" id="${id}-range" min="0.3" max="${crystal.k_max}" step="0.05" value="${state.patternRange}"></div>
          <div class="${id}-slider" id="${id}-fieldwrap"><label><span>field of view</span><span id="${id}-field-val"></span></label>
            <input type="range" id="${id}-field" min="10" max="200" step="5" value="${state.fieldMrad}"></div>
        </div>
        <div class="${id}-row" id="${id}-approw">
          <div class="${id}-btn" id="${id}-apptoggle">appearance ▾</div>
        </div>
        <div class="${id}-row ${id}-appearance" id="${id}-apppanel" style="display:none">
          <div class="${id}-slider" id="${id}-powwrap"><label><span>brightness ∝ intensity^p</span><span id="${id}-pow-val"></span></label>
            <input type="range" id="${id}-pow" min="0.1" max="1" step="0.05" value="${state.power}"></div>
          <div class="${id}-histwrap" id="${id}-histwrapper">
            <span class="${id}-hint" style="justify-content:flex-start">contrast (drag the handles)</span>
            <canvas class="${id}-hist" id="${id}-hist" width="160" height="40"></canvas>
            <span class="${id}-hint"><span id="${id}-hist-lo">0.00</span><span id="${id}-hist-hi">1.00</span></span>
          </div>
        </div>
        <div class="${id}-hint" id="${id}-status"></div>
        <div class="${id}-hint">Drag the pattern to move the tilt map. Double-click a disk to tilt to its two-beam condition, or empty space to put the Laue circle centre there.</div>
        </div>
      </div>
    </div>
  `;
  el.appendChild(wrap);
  const $ = <T extends HTMLElement>(sel: string) => wrap.querySelector(sel) as T;
  const cellCanvas = $<HTMLCanvasElement>(`#${id}-cell`);
  const ewaldCanvas = $<HTMLCanvasElement>(`#${id}-ewaldc`);
  const patCanvas = $<HTMLCanvasElement>(`#${id}-pat`);

  // ---------------------------------------------------------------- theme
  const palettes = {
    dark: { bg: "#111", fg: "#ccc", title: "#ddd", label: "#aaa", dim: "#777", btnbg: "#222", btnhover: "#333", border: "#444", cellbg: "#161616" },
    light: { bg: "#f4f6f3", fg: "#333", title: "#1f1f1f", label: "#556", dim: "#777", btnbg: "#ffffff", btnhover: "#e9ebe6", border: "#d3d6d0", cellbg: "#fbfbfa" },
  };
  let dark = detectDark();
  const applyTheme = () => {
    dark = detectDark();
    const p = palettes[dark ? "dark" : "light"];
    for (const k in p) wrap.style.setProperty(`--${id}-${k}`, (p as Record<string, string>)[k]);
    drawAll();
  };
  const mo = new MutationObserver(() => applyTheme());
  mo.observe(document.documentElement, { attributes: true });
  if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);

  // ---------------------------------------------------------------- size
  // Layout: left column = cell (Sc square) above the Ewald panel (Sc x Se);
  // right column = pattern square whose side S matches the left column's
  // height, so S = Sc + gap + Se with Se = Sc / 2. On narrow (portrait)
  // screens everything stacks in one column with the cell and the pattern
  // adjacent, then the controls, then the Ewald panel. Both layouts are
  // capped so the canvases fit in the visible height below a fixed page
  // header (the MyST top bar).
  let S = 400, Sc = 260, Se = 130;
  const GAP = 8;
  const panelsEl = $<HTMLElement>(`.${id}-panels`);
  // small viewport height (browser toolbars shown): stable while the mobile toolbars collapse on scroll
  const probe = document.createElement("div");
  probe.style.cssText = "position:fixed;top:0;left:0;width:0;height:100vh;height:100svh;visibility:hidden;pointer-events:none";
  el.appendChild(probe);
  const availHeight = () => {
    const h = probe.offsetHeight || window.innerHeight;
    let nav = 0;
    for (const e of document.querySelectorAll<HTMLElement>(".myst-top-nav, header")) {
      const p = getComputedStyle(e).position;
      const r = e.getBoundingClientRect();
      if ((p === "fixed" || p === "sticky") && r.top <= 0 && r.height < h / 3) nav = Math.max(nav, r.bottom);
    }
    return h - nav - 16;
  };
  const computeSize = () => {
    const w = wrap.clientWidth - 30;
    const h = availHeight();
    const sideBySide = w >= 2.5 * 180 + 2 * GAP + 12;
    if (sideBySide) {
      const hCap = state.showEwald ? (h - GAP) / 1.5 : h;
      Sc = Math.floor(Math.max(140, Math.min((w - 12 - GAP) / 2.5, state.sizePref / 1.5, hCap)));
      Se = state.showEwald ? Math.round(Sc / 2) : 0;
      S = Sc + (Se ? GAP + Se : 0);
    } else {
      // cell and pattern stacked: keep both on screen at once
      Sc = Math.floor(Math.max(180, Math.min(state.sizePref, w, (h - 12) / 2)));
      Se = state.showEwald ? Math.round(Sc / 2) : 0;
      S = Sc;
    }
    panelsEl.classList.toggle("stacked", !sideBySide);
    cellCanvas.style.width = `${Sc}px`; cellCanvas.style.height = `${Sc}px`;
    ewaldCanvas.style.display = Se ? "block" : "none";
    ewaldCanvas.style.width = `${Sc}px`; ewaldCanvas.style.height = `${Se}px`;
    patCanvas.style.width = `${S}px`; patCanvas.style.height = `${S}px`;
    const panels = wrap.querySelectorAll<HTMLElement>(`.${id}-panel`);
    if (panels[0]) panels[0].style.width = `${Sc}px`;
    if (panels[1]) panels[1].style.width = `${S}px`;
    $<HTMLElement>(`#${id}-cellctl`).style.width = sideBySide ? "" : `${Math.max(Sc, Math.min(w, 420))}px`;
    $<HTMLElement>(`#${id}-patctl`).style.width = sideBySide ? "" : `${Math.max(Sc, Math.min(w, 420))}px`;
  };

  // ---------------------------------------------------------------- physics cache
  let nb: { beams: Reflection[]; nDyn: number } = { beams: [], nDyn: 0 };
  let nbSolution: NanobeamSolution | null = null;
  let nbTilts: [number, number][] = [[0, 0]];
  let cbed: {
    grid: ReturnType<typeof tiltGrid>;
    beams: Reflection[];
    nDyn: number;
    nodes: [number, number][]; // precession ring, one entry at the origin without it
    sols: ReturnType<typeof blochSolve>[] | null; // grid tilt major, ring node minor
  } | null = null;
  let lines: ReturnType<typeof kosselLines> = [];
  let geomKey = "";

  const frame = (): Frame => ({ size: S, qMax: state.mode === "kikuchi" ? state.fieldMrad * 1e-3 : state.patternRange, viewX: VIEW_X });

  // quality while dragging adapts to the device: the beam cap and the
  // precession node count shrink when a recompute takes too long
  let dragBeams = MAX_BEAMS_DRAG;
  let dragNodes = 8;
  const solveOrientation = () => {
    // everything that depends on the orientation (not on thickness)
    const q = state.quat;
    const alpha = state.semiconv * 1e-3;
    const maxBeams = (state.dragging || state.spinning) ? dragBeams : MAX_BEAMS;
    if (state.mode === "nanobeam") {
      nbTilts = precessionTilts(k0(), state.precession, (state.dragging || state.spinning) ? dragNodes : 16);
      if (state.dynamical) {
        // the physics always uses every reflection the crystal carries; the pattern range only crops the drawing
        nbSolution = nanobeamSolve(crystal, q, crystal.k_max, SG_MAX, maxBeams, nbTilts);
        nb = { beams: nbSolution.beams, nDyn: Math.round(nbSolution.nDynMean) };
      } else {
        nb = { beams: [DIRECT, ...labReflections(crystal, q, crystal.k_max)], nDyn: 0 };
        nbSolution = null;
      }
    } else if (state.mode === "cbed") {
      const Rk = k0() * Math.sin(alpha);
      const grid = tiltGrid(Rk, (state.dragging || state.spinning) ? CBED_GRID_DRAG : CBED_GRID);
      // every incident direction of the cone is itself precessed, so the cost
      // is the grid times the ring: fewer ring nodes here than in nanobeam
      const nRing = (state.dragging || state.spinning) ? Math.min(4, dragNodes) : CBED_PREC_NODES;
      const nodes = precessionTilts(k0(), state.precession, nRing);
      if (state.dynamical) {
        const { beams, nDyn } = hybridBeams(crystal, q, crystal.k_max, SG_MAX, (state.dragging || state.spinning) ? 20 : 32, Math.sin(alpha));
        const dyn = beams.slice(0, nDyn);
        const sols: ReturnType<typeof blochSolve>[] = [];
        for (const t of grid.tilts) {
          for (const nd of nodes) sols.push(blochSolve(crystal, dyn, [t[0] + nd[0], t[1] + nd[1]]));
        }
        cbed = { grid, beams, nDyn, nodes, sols };
      } else {
        cbed = { grid, beams: [DIRECT, ...labReflections(crystal, q, crystal.k_max)], nDyn: 0, nodes, sols: null };
      }
    }
    if (state.mode === "kikuchi" || (state.mode !== "kikuchi" && state.kikuchi)) {
      const fov = state.mode === "kikuchi" ? state.fieldMrad * 1e-3 : state.patternRange / k0();
      lines = kosselLines(crystal, q, Math.min(crystal.k_max, 2.5), fov);
    }
  };

  const intensitiesNanobeam = (): Float64Array => {
    if (state.dynamical && nbSolution) return nanobeamIntensities(crystal, nbSolution, state.thickness);
    const out = new Float64Array(nb.beams.length);
    for (const t of nbTilts) {
      const v = kinematicalTilted(crystal, nb.beams, t, 0.02);
      for (let i = 0; i < out.length; i++) out[i] += v[i] / nbTilts.length;
    }
    return out;
  };

  /** Intensity of every beam at each incident direction of the cone, averaged
   * over the precession ring. */
  const intensitiesCbed = (): Float64Array[] => {
    if (!cbed) return [];
    const { grid, beams, nDyn, nodes, sols } = cbed;
    return grid.tilts.map((t, i) => {
      const out = new Float64Array(beams.length);
      for (let k = 0; k < nodes.length; k++) {
        const tilt: [number, number] = [t[0] + nodes[k][0], t[1] + nodes[k][1]];
        const acc = new Float64Array(beams.length);
        if (sols) {
          acc.set(blochIntensities(sols[i * nodes.length + k], state.thickness));
          slabIntensities(crystal, beams, nDyn, tilt, state.thickness, acc);
        } else {
          acc.set(kinematicalTilted(crystal, beams, tilt, 0.02));
        }
        for (let b = 0; b < out.length; b++) out[b] += acc[b] / nodes.length;
      }
      return out;
    });
  };

  /** Disk-averaged intensity of every beam, for the double-click snap. */
  const intensitiesCbedMean = (): Float64Array => {
    const per = intensitiesCbed();
    const out = new Float64Array(cbed ? cbed.beams.length : 0);
    for (const arr of per) for (let b = 0; b < out.length; b++) out[b] += arr[b] / Math.max(per.length, 1);
    return out;
  };

  // ---------------------------------------------------------------- drawing
  const drawCellPanel = () => {
    const key = `${state.preset}|${state.nCells.join(",")}|${state.polyhedra}`;
    if (key !== geomKey) { geom = cellGeometry(crystal, state.nCells, state.polyhedra); geomKey = key; }
    drawCell(cellCanvas, geom, state.quat, Sc, { dark, showAxes: true, showLabels: state.showLabels, atomScale: 0.45, viewX: VIEW_X });
    if (state.showEwald && Se > 0) {
      const dpr = window.devicePixelRatio || 1;
      if (ewaldCanvas.width !== Sc * dpr || ewaldCanvas.height !== Se * dpr) { ewaldCanvas.width = Sc * dpr; ewaldCanvas.height = Se * dpr; }
      const ctx = ewaldCanvas.getContext("2d");
      if (ctx) {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const refl = state.mode === "nanobeam" && nb.beams.length ? nb.beams : labReflections(crystal, state.quat, crystal.k_max);
        drawEwaldPanel(ctx, Sc, Se, refl, k0(), state.patternRange, SG_MAX, VIEW_X, dark, palettes[dark ? "dark" : "light"].cellbg, state.mode !== "kikuchi" ? state.precession : 0);
      }
    }
    const R = quatToMatrix(state.quat);
    $(`#${id}-za`).textContent = "zone axis " + fmtIndices(directionIndices(crystal.cell, matTVec(R, [0, 0, 1])), crystal.hexagonal);
    $<HTMLInputElement>(`#${id}-zone`).placeholder = crystal.hexagonal ? "0 0 0 1" : "1 1 0";
    $(`#${id}-info`).textContent = `${crystal.name} · ${crystal.spacegroup || crystal.pointgroup}`;
  };

  // ---- histogram of the scaled intensities with a draggable contrast window
  const histCanvas = $<HTMLCanvasElement>(`#${id}-hist`);
  let histBins = new Array(64).fill(0);
  const histogramOf = (vals: ArrayLike<number>) => {
    const bins = new Array(64).fill(0);
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (!(v >= 0)) continue;
      bins[Math.min(63, Math.floor(v * 63.999))]++;
    }
    histBins = bins;
    drawHistogram();
  };
  const drawHistogram = () => {
    const dpr = window.devicePixelRatio || 1;
    const W = 160, H = 40;
    if (histCanvas.width !== W * dpr) { histCanvas.width = W * dpr; histCanvas.height = H * dpr; }
    histCanvas.style.width = `${W}px`; histCanvas.style.height = `${H}px`;
    const c = histCanvas.getContext("2d");
    if (!c) return;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = dark ? "#1a1a1a" : "#f0f0f0";
    c.fillRect(0, 0, W, H);
    const mx = Math.max(1e-3, ...histBins.map((v) => Math.log1p(v)));
    const bw = W / 64;
    for (let i = 0; i < 64; i++) {
      const h = (Math.log1p(histBins[i]) / mx) * (H - 2);
      const x = (i + 0.5) / 64;
      c.fillStyle = x >= state.vmin && x <= state.vmax ? (dark ? "#9a9a9a" : "#666") : (dark ? "#444" : "#c4c4c4");
      c.fillRect(i * bw + 0.5, H - h, Math.max(1, bw - 1), h);
    }
    c.strokeStyle = "#00cc66"; c.lineWidth = 2;
    for (const v of [state.vmin, state.vmax]) { c.beginPath(); c.moveTo(v * W, 0); c.lineTo(v * W, H); c.stroke(); }
    $(`#${id}-hist-lo`).textContent = state.vmin.toFixed(2);
    $(`#${id}-hist-hi`).textContent = state.vmax.toFixed(2);
  };
  let histHandle: "lo" | "hi" | null = null;
  histCanvas.addEventListener("pointerdown", (e) => {
    const rect = histCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    histHandle = Math.abs(x - state.vmin) <= Math.abs(x - state.vmax) ? "lo" : "hi";
    histCanvas.setPointerCapture?.(e.pointerId);
    e.preventDefault();
  });
  histCanvas.addEventListener("pointermove", (e) => {
    if (!histHandle) return;
    const rect = histCanvas.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    if (histHandle === "lo") state.vmin = Math.min(x, state.vmax - 0.02);
    else state.vmax = Math.max(x, state.vmin + 0.02);
    drawPattern();
  });
  const histUp = () => { histHandle = null; };
  histCanvas.addEventListener("pointerup", histUp);
  histCanvas.addEventListener("pointercancel", histUp);

  const drawPattern = () => {
    const ctx = setupCanvas(patCanvas, S);
    if (!ctx) return;
    const f = frame();
    let status = "";
    if (state.mode === "nanobeam") {
      const inten = intensitiesNanobeam();
      drawDisks(ctx, f, nb.beams, inten, dark, state.showHkl, k0() * Math.sin(state.semiconv * 1e-3), state.power, state.vmin, state.vmax);
      if (state.kikuchi) drawKikuchiOverlay(ctx, f, lines, k0(), dark);
      let iMax = 0;
      for (let i = 0; i < inten.length; i++) if (nb.beams[i].index >= 0) iMax = Math.max(iMax, inten[i]);
      const scaled = new Float32Array(inten.length);
      for (let i = 0; i < inten.length; i++) scaled[i] = inten[i] > 1e-6 * iMax ? Math.pow(inten[i] / (iMax || 1), state.power) : -1;
      histogramOf(scaled);
      const prec = state.precession > 0 ? `; precession ${state.precession.toFixed(2)}° averaged over ${nbTilts.length} ring nodes` : "";
      status = state.dynamical
        ? `${nb.nDyn} Bloch beams (|s| < ${SG_MAX} Å⁻¹, absorptive), thin-slab intensities for the other ${nb.beams.length - nb.nDyn}${prec}`
        : `kinematical: |F|² with a Gaussian excitation envelope (σ = 0.02 Å⁻¹)`;
    } else if (state.mode === "cbed" && cbed) {
      const inten = intensitiesCbed();
      const img = cbedImage(f, cbed.beams, cbed.grid, inten);
      // normalise to the brightest pixel outside the direct disk
      const Rpx = cbed.grid.R * (0.5 * S * 0.92) / f.qMax + 1.5;
      let hi = 0;
      for (let py = 0; py < S; py++) for (let px = 0; px < S; px++) {
        if (Math.hypot(px + 0.5 - S / 2, py + 0.5 - S / 2) <= Rpx) continue;
        const v = img[py * S + px];
        if (v > hi) hi = v;
      }
      if (!(hi > 0)) hi = 1;
      // power-law scaling and the contrast window, like the disks
      const disp = new Float32Array(img.length);
      for (let i = 0; i < img.length; i++) disp[i] = Math.min(1, Math.pow(Math.max(img[i], 0) / hi, state.power));
      histogramOf(disp);
      drawImage(ctx, f, disp, dark ? "gray" : "gray_r", state.vmin, state.vmax, dark, "Å⁻¹");
      if (state.kikuchi) drawKikuchiOverlay(ctx, f, lines, k0(), dark);
      const prec = state.precession > 0
        ? `; precession ${state.precession.toFixed(2)}° over ${cbed.nodes.length} ring nodes`
        : "";
      status = `${cbed.nDyn} Bloch beams × ${cbed.grid.tilts.length} incident tilts per disk; disks summed where they overlap${prec}`;
    } else if (state.mode === "kikuchi") {
      drawKosselLines(ctx, f, lines, dark, state.showHkl, 0.02);
      status = `deficient line of every reflection: line width = two-beam rocking width |U_g| / (k₀|g|), darkness ∝ |U_g|`;
    }
    $(`#${id}-status`).textContent = status;
  };

  const drawAll = () => { drawCellPanel(); drawPattern(); };
  const recompute = () => {
    const t0 = performance.now();
    solveOrientation();
    drawAll();
    if (state.dragging || state.spinning) {
      // keep dragging responsive: aim for well under 100 ms per recompute
      const dt = performance.now() - t0;
      if (dt > 90) { dragBeams = Math.max(12, Math.round(dragBeams * 0.7)); dragNodes = Math.max(4, dragNodes - 2); }
      else if (dt < 30) { dragBeams = Math.min(MAX_BEAMS_DRAG, dragBeams + 4); dragNodes = Math.min(8, dragNodes + 1); }
    }
  };

  // ---------------------------------------------------------------- interaction
  const rotateScreen = (axis: Vec3, deg: number) => {
    // axis in screen coordinates (x right, y up, z toward the viewer) -> lab
    const dq = quatFromAxisAngle([VIEW_X * axis[0], axis[1], VIEW_X * axis[2]], (deg * Math.PI) / 180);
    state.quat = qnormalize(qmult(dq, state.quat));
  };
  const shiftPattern = (dqx: number, dqy: number, inverseAngstrom: boolean) => {
    const ax = inverseAngstrom ? dqx / k0() : dqx, ay = inverseAngstrom ? dqy / k0() : dqy;
    const ang = Math.hypot(ax, ay);
    if (ang <= 0) return;
    state.quat = qnormalize(qmult(quatFromAxisAngle([ay, -ax, 0], ang), state.quat));
  };
  let raf = 0;
  const scheduleRecompute = () => {
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = 0; recompute(); });
  };
  const endDrag = () => {
    if (!state.dragging) return;
    state.dragging = false;
    cellCanvas.style.cursor = patCanvas.style.cursor = "grab";
    recompute(); // full quality
  };

  const attachDrag = (canvas: HTMLCanvasElement, onMove: (dx: number, dy: number) => void) => {
    const pointers = new Map<number, [number, number]>();
    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture?.(e.pointerId);
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      state.dragging = true;
      canvas.style.cursor = "grabbing";
      e.preventDefault();
    });
    canvas.addEventListener("pointermove", (e) => {
      const prev = pointers.get(e.pointerId);
      if (!prev) return;
      const cur: [number, number] = [e.clientX, e.clientY];
      if (pointers.size >= 2) {
        const other = [...pointers.entries()].find(([pid]) => pid !== e.pointerId);
        if (other) {
          const [ox, oy] = other[1];
          let da = Math.atan2(cur[1] - oy, cur[0] - ox) - Math.atan2(prev[1] - oy, prev[0] - ox);
          if (da > Math.PI) da -= 2 * Math.PI;
          if (da < -Math.PI) da += 2 * Math.PI;
          rotateScreen([0, 0, 1], (-da * 180) / Math.PI);
        }
      } else if (e.shiftKey) {
        // shift-drag twists about the beam: the desktop version of the two-finger gesture
        const rect = canvas.getBoundingClientRect();
        const cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
        let da = Math.atan2(cur[1] - cy, cur[0] - cx) - Math.atan2(prev[1] - cy, prev[0] - cx);
        if (da > Math.PI) da -= 2 * Math.PI;
        if (da < -Math.PI) da += 2 * Math.PI;
        rotateScreen([0, 0, 1], (-da * 180) / Math.PI);
      } else {
        onMove(cur[0] - prev[0], cur[1] - prev[1]);
      }
      pointers.set(e.pointerId, cur);
      scheduleRecompute();
    });
    const up = (e: PointerEvent) => { pointers.delete(e.pointerId); if (pointers.size === 0) endDrag(); };
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", up);
    canvas.addEventListener("pointerleave", up);
  };
  attachDrag(cellCanvas, (dx, dy) => {
    const ang = Math.hypot(dx, dy) * (180 / Sc);
    if (ang > 0) rotateScreen([dy, dx, 0], ang); // trackball: the near face follows the pointer
  });
  attachDrag(patCanvas, (dx, dy) => {
    const f = frame();
    const sc = (0.5 * S * 0.92) / f.qMax;
    shiftPattern((VIEW_X * dx) / sc, -dy / sc, state.mode !== "kikuchi");
  });
  // Double-click on a visible disk: tilt the crystal to the exact Bragg
  // condition of that reflection (two-beam: Laue circle through 000 and g;
  // w = s_g (-g_y, g_x) / |g_xy|^2 zeroes s_g). On empty space the
  // Laue-circle centre moves to the clicked point (same sense); in Kossel
  // mode the clicked direction of the tilt map moves onto the axis.
  patCanvas.addEventListener("dblclick", (e) => {
    const rect = patCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const f = frame();
    const sc = (0.5 * S * 0.92) / f.qMax;
    if (state.mode === "nanobeam" || state.mode === "cbed") {
      // a convergent beam draws the same reflections as wide disks, so both
      // modes snap to the same two-beam condition the same way
      const beams = state.mode === "cbed" && cbed ? cbed.beams : nb.beams;
      const inten = state.mode === "cbed" ? intensitiesCbedMean() : intensitiesNanobeam();
      let iMax = 0;
      for (let i = 1; i < beams.length; i++) iMax = Math.max(iMax, inten[i]);
      const disk = k0() * Math.sin(state.semiconv * 1e-3) * sc;
      const snap = Math.max(8, (state.mode === "cbed" ? 1.0 : 1.5) * disk);
      // several reflections of different g_z share one spot (in hcp the first
      // HOLZ layer is only 0.21 1/A up): take the candidate under the click
      // that needs the smallest tilt to reach Bragg, never more than 5 degrees
      let best: Reflection | null = null, bestTilt = (5 * Math.PI) / 180;
      for (let i = 0; i < beams.length; i++) {
        const b = beams[i];
        if (b.index < 0 || !(inten[i] > 1e-4 * iMax)) continue;
        const [px, py] = toPx(f, b.g[0], b.g[1]);
        if (Math.hypot(px - x, py - y) > snap) continue;
        const gxy = Math.hypot(b.g[0], b.g[1]);
        if (gxy < 1e-6) continue;
        const tilt = Math.abs(b.s) / gxy;
        if (tilt < bestTilt) { bestTilt = tilt; best = b; }
      }
      if (best) {
        const gxy2 = best.g[0] ** 2 + best.g[1] ** 2;
        const wx = (-best.s * best.g[1]) / gxy2, wy = (best.s * best.g[0]) / gxy2;
        state.quat = qnormalize(qmult(quatFromAxisAngle([wx, wy, 0], Math.hypot(wx, wy)), state.quat));
        recompute();
        return;
      }
      // empty space: the Laue circle centre moves to the click
      shiftPattern((VIEW_X * (x - rect.width / 2)) / sc, -(y - rect.height / 2) / sc, true);
    } else {
      // the Kikuchi map is a map of beam directions, so the clicked direction
      // moves onto the axis, which is the opposite sense
      shiftPattern(-(VIEW_X * (x - rect.width / 2)) / sc, (y - rect.height / 2) / sc, false);
    }
    recompute();
  });

  // ---------------------------------------------------------------- controls
  wrap.querySelectorAll<HTMLElement>(`[data-rot]`).forEach((b) => {
    b.addEventListener("click", () => {
      const key = b.dataset.rot!;
      const axis: Vec3 = [+(key[0] === "x"), +(key[0] === "y"), +(key[0] === "z")];
      rotateScreen(axis, key[1] === "+" ? state.stepDeg : -state.stepDeg);
      recompute();
    });
  });
  $<HTMLInputElement>(`#${id}-step`).addEventListener("change", (e) => { state.stepDeg = Math.max(0.1, +(e.target as HTMLInputElement).value || 15); });
  // continuous slow rotation (toggle buttons): state.spinSpeed deg/s about the screen
  // axes, recomputed at drag quality no more than SPIN_FPS times a second
  let spinRaf = 0, spinLast = 0, spinDue = 0, spinPendX = 0, spinPendY = 0;
  const spinTick = (t: number) => {
    spinRaf = 0;
    if (!state.spinX && !state.spinY) {
      if (state.spinning) { state.spinning = false; recompute(); } // back to full quality
      return;
    }
    const dt = spinLast ? Math.min(0.1, (t - spinLast) / 1000) : 0;
    spinLast = t;
    if (state.spinY) spinPendY += state.spinSpeed * dt;
    if (state.spinX) spinPendX += state.spinSpeed * dt;
    if (t >= spinDue) {
      spinDue = t + 1000 / SPIN_FPS;
      if (spinPendY) rotateScreen([0, 1, 0], spinPendY);
      if (spinPendX) rotateScreen([1, 0, 0], spinPendX);
      spinPendX = spinPendY = 0;
      recompute();
    }
    spinRaf = requestAnimationFrame(spinTick);
  };
  $<HTMLInputElement>(`#${id}-spin`).addEventListener("input", (e) => {
    state.spinSpeed = Math.max(1, +(e.target as HTMLInputElement).value || 6);
    $(`#${id}-spinv`).textContent = `${state.spinSpeed}°/s`;
  });
  wrap.querySelectorAll<HTMLElement>(`[data-spin]`).forEach((b) => {
    b.addEventListener("click", () => {
      if (b.dataset.spin === "x") state.spinX = !state.spinX; else state.spinY = !state.spinY;
      b.classList.toggle("active", b.dataset.spin === "x" ? state.spinX : state.spinY);
      if ((state.spinX || state.spinY) && !spinRaf) { state.spinning = true; spinLast = 0; spinRaf = requestAnimationFrame(spinTick); }
    });
  });
  const goZone = () => {
    const v = parseDirection($<HTMLInputElement>(`#${id}-zone`).value); // 3 or 4 (Miller-Bravais) indices
    if (!v) return;
    setZoneAxis(v);
    recompute();
  };
  $(`#${id}-go`).addEventListener("click", goZone);
  $<HTMLInputElement>(`#${id}-zone`).addEventListener("keydown", (e) => { if (e.key === "Enter") goZone(); });
  $(`#${id}-reset`).addEventListener("click", () => { setZoneAxis(crystal.hexagonal ? [0, 0, 1] : [1, 1, 0]); recompute(); });
  // CIF upload: parse in the browser, add to the structure menu, select it
  const fileInput = $<HTMLInputElement>(`#${id}-ciffile`);
  $(`#${id}-cifbtn`).addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = parseCif(text, file.name.replace(/\.cif$/i, ""));
      const menuName = `${parsed.name} (CIF)`;
      cifs.set(menuName, parsed);
      const sel = $<HTMLSelectElement>(`#${id}-preset`);
      if (![...sel.options].some((o) => o.value === menuName)) {
        const opt = document.createElement("option");
        opt.value = menuName; opt.textContent = menuName;
        sel.appendChild(opt);
      }
      sel.value = menuName;
      state.preset = menuName;
      crystal = loadCrystal(menuName, state.energy);
      const range = $<HTMLInputElement>(`#${id}-range`);
      range.max = String(crystal.k_max);
      if (state.patternRange > crystal.k_max) { state.patternRange = crystal.k_max; range.value = String(crystal.k_max); }
      range.dispatchEvent(new Event("input")); // refresh the slider label
      geomKey = "";
      setZoneAxis(crystal.hexagonal ? [0, 0, 1] : [1, 1, 0]);
      recompute();
      $(`#${id}-status`).textContent = `${menuName}: ${parsed.symbols.length} atoms in the cell, ${parsed.spacegroup || "symmetry from the file"}, ${crystal.hkl.length} reflections out to ${crystal.k_max.toFixed(2)} Å⁻¹${crystal.k_max < 3 ? " (reduced for this cell size)" : ""}; absorption approximated as 8 % of the potential`;
    } catch (err) {
      $(`#${id}-status`).textContent = `could not read ${file.name}: ${(err as Error).message}`;
    }
    fileInput.value = "";
  });
  $<HTMLSelectElement>(`#${id}-preset`).addEventListener("change", (e) => {
    state.preset = (e.target as HTMLSelectElement).value;
    crystal = loadCrystal(state.preset, state.energy);
    const range = $<HTMLInputElement>(`#${id}-range`);
    range.max = String(crystal.k_max);
    if (state.patternRange > crystal.k_max) { state.patternRange = crystal.k_max; range.value = String(crystal.k_max); }
    setZoneAxis(crystal.hexagonal ? [0, 0, 1] : [1, 1, 0]);
    recompute();
  });
  const modeButtons = wrap.querySelectorAll<HTMLElement>(`[data-mode]`);
  const updateModeUI = () => {
    modeButtons.forEach((b) => b.classList.toggle("active", b.dataset.mode === state.mode));
    const show = (sel: string, on: boolean) => { $(sel).style.display = on ? "" : "none"; };
    // nanobeam and CBED share every control; the Kikuchi pattern is a map of
    // beam directions and takes the field of view instead
    const pattern = state.mode !== "kikuchi";
    show(`#${id}-thickwrap`, pattern && state.dynamical);
    show(`#${id}-convwrap`, pattern);
    show(`#${id}-rangewrap`, pattern);
    show(`#${id}-fieldwrap`, !pattern);
    show(`#${id}-dynwrap`, pattern);
    show(`#${id}-kikwrap`, pattern);
    show(`#${id}-precwrap`, pattern);
    show(`#${id}-approw`, state.mode !== "kikuchi");
    show(`#${id}-apppanel`, state.mode !== "kikuchi" && state.showAppearance);
  };
  modeButtons.forEach((b) => b.addEventListener("click", () => { state.mode = b.dataset.mode!; updateModeUI(); recompute(); }));
  $<HTMLInputElement>(`#${id}-dyn`).addEventListener("change", (e) => { state.dynamical = (e.target as HTMLInputElement).checked; updateModeUI(); recompute(); });
  $<HTMLInputElement>(`#${id}-kik`).addEventListener("change", (e) => { state.kikuchi = (e.target as HTMLInputElement).checked; recompute(); });
  $<HTMLInputElement>(`#${id}-hkl`).addEventListener("change", (e) => { state.showHkl = (e.target as HTMLInputElement).checked; drawPattern(); });
  $(`#${id}-apptoggle`).addEventListener("click", () => {
    state.showAppearance = !state.showAppearance;
    $(`#${id}-apptoggle`).textContent = state.showAppearance ? "appearance ▴" : "appearance ▾";
    updateModeUI();
    if (state.showAppearance) drawPattern();
  });
  $<HTMLInputElement>(`#${id}-labels`).addEventListener("change", (e) => { state.showLabels = (e.target as HTMLInputElement).checked; drawAll(); });
  $<HTMLInputElement>(`#${id}-poly`).addEventListener("change", (e) => { state.polyhedra = (e.target as HTMLInputElement).checked; drawAll(); });
  $<HTMLInputElement>(`#${id}-ewald`).addEventListener("change", (e) => { state.showEwald = (e.target as HTMLInputElement).checked; computeSize(); drawAll(); });
  $<HTMLSelectElement>(`#${id}-energy`).addEventListener("change", (e) => {
    state.energy = +(e.target as HTMLSelectElement).value;
    crystal = loadCrystal(state.preset, state.energy);
    recompute();
  });
  $<HTMLInputElement>(`#${id}-ncell`).addEventListener("change", (e) => {
    const n = Math.max(1, Math.min(3, Math.round(+(e.target as HTMLInputElement).value || 1)));
    state.nCells = [n, n, n]; drawAll();
  });
  const slider = (sel: string, valSel: string, fmt: (v: number) => string, apply: (v: number) => void, heavy: boolean) => {
    const inp = $<HTMLInputElement>(sel);
    const out = $(valSel);
    const update = () => { const v = +inp.value; out.textContent = fmt(v); apply(v); };
    inp.addEventListener("input", () => { update(); if (heavy) scheduleRecompute(); else drawPattern(); });
    update();
  };
  slider(`#${id}-thick`, `#${id}-thick-val`, (v) => `${v.toFixed(0)} Å`, (v) => { state.thickness = v; }, false);
  slider(`#${id}-pow`, `#${id}-pow-val`, (v) => `p = ${v.toFixed(2)}`, (v) => { state.power = v; }, false);
  slider(`#${id}-conv`, `#${id}-conv-val`, (v) => `${v.toFixed(1)} mrad`, (v) => { state.semiconv = v; }, true);
  slider(`#${id}-prec`, `#${id}-prec-val`, (v) => (v > 0 ? `${v.toFixed(2)}°` : "off"), (v) => { state.precession = v; }, true);
  slider(`#${id}-range`, `#${id}-range-val`, (v) => `${v.toFixed(2)} Å⁻¹`, (v) => { state.patternRange = v; }, true);
  slider(`#${id}-field`, `#${id}-field-val`, (v) => `${v.toFixed(0)} mrad`, (v) => { state.fieldMrad = v; }, true);
  // the convergence slider only changes the disk radius in nanobeam mode: no re-solve needed there
  $<HTMLInputElement>(`#${id}-conv`).addEventListener("input", () => { if (state.mode === "nanobeam") drawPattern(); });

  // ---------------------------------------------------------------- go
  updateModeUI();
  computeSize();
  applyTheme();
  recompute();
  const relayout = () => { const old = S + Sc; computeSize(); if (S + Sc !== old) drawAll(); };
  const ro = new ResizeObserver(relayout);
  ro.observe(wrap);
  ro.observe(probe); // viewport height: rotation, window resize
  return () => { mo.disconnect(); ro.disconnect(); probe.remove(); };
}

export default { render };
