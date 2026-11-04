import { describe, expect, it } from "vitest";
import { parseCrystal } from "../diffractionsim/physics";
import { crystalFromCif, parseCif } from "./cif";
import { PRESETS } from "./presets";
import { expi, wkFactor } from "./wk";

// quantem 0.1.10 compute_WK_factor at g = 0, 0.1, 0.5, 1, 2 and 4 1/Å: Z, energy (eV), thermal sigma (Å), real and
// imaginary parts (Å). Ba is evaluated without thermal motion (thermal_sigma=None).
const G = [0, 0.1, 0.5, 1, 2, 4];
const QUANTEM: [number, number, number, number[], number[]][] = [
  [8, 200e3, 0.05, [0.878842827, 0.8569955867, 0.5350432374, 0.2287918166, 0.05804389514, 0.008763085257], [0.07381676796, 0.02921108114, 0.007140258995, 0.003502613072, 0.001662580703, 0.0003639388325]],
  [14, 200e3, 0.05, [2.573242388, 2.400648068, 0.8584158641, 0.3129411002, 0.09872395307, 0.01530708127], [0.08731621303, 0.04434035067, 0.01540323684, 0.008929541906, 0.004824238505, 0.001067781699]],
  [29, 80e3, 0.08, [2.039508722, 1.944264181, 0.985944491, 0.479651793, 0.1146950689, 0.007371980509], [0.162097214, 0.1294121581, 0.07831580287, 0.05346970825, 0.02179115835, 0.0005167931711]],
  [79, 300e3, 0.05, [5.322768626, 5.150244333, 2.995159809, 1.484446398, 0.4979596292, 0.08858975466], [0.3107062301, 0.2885104641, 0.2232146917, 0.1813617199, 0.116554577, 0.03159529951]],
  [56, 200e3, 0, [8.072943284, 7.08037479, 2.451600523, 1.108530842, 0.4019062488, 0.1224114238], [0.09606916902, 0.07095366241, 0.02153503785, 0.006892579868, 0.001490825535, 0.0002744802185]],
];

// The silicon preset (ASE diamond cubic, a = 5.431 Å) as a P1 CIF.
const SI_CIF = `data_Si
_cell_length_a 5.431
_cell_length_b 5.431
_cell_length_c 5.431
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_space_group_symop_operation_xyz
'x, y, z'
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si 0 0 0
Si 0.25 0.25 0.25
Si 0 0.5 0.5
Si 0.25 0.75 0.75
Si 0.5 0 0.5
Si 0.75 0.25 0.75
Si 0.5 0.5 0
Si 0.75 0.75 0.25
`;

describe("Weickenmeier-Kohl scattering factors", () => {
  it("match quantem's compute_WK_factor", () => {
    expect(expi(1)).toBeCloseTo(1.8951178163559368, 13);
    expect(expi(-1)).toBeCloseTo(-0.21938393439552062, 13);
    expect(expi(-10) / -4.156968929685324e-6).toBeCloseTo(1, 12);
    expect(expi(15) / 234955.85249076830).toBeCloseTo(1, 12);
    for (const [Z, energy, sigma, re, im] of QUANTEM) {
      G.forEach((g, i) => {
        const [fr, fi] = wkFactor(g, Z, energy, sigma);
        expect(Math.abs(fr - re[i]) / re[0]).toBeLessThan(1e-8);
        expect(Math.abs(fi - im[i]) / im[0]).toBeLessThan(1e-8);
      });
    }
  });

  it("give a CIF of a preset structure the couplings quantem computed for the preset", () => {
    const preset = parseCrystal(PRESETS["Si (diamond cubic)"])!;
    const cif = crystalFromCif(parseCif(SI_CIF, "Si"), preset.energy_ev, 3.0);
    let uMax = 0;
    preset.couplingRe.forEach((v, key) => { uMax = Math.max(uMax, Math.hypot(v, preset.couplingIm.get(key)!)); });
    let compared = 0;
    cif.hkl.forEach((h, i) => {
      const key = h.join(",");
      if (!preset.couplingRe.has(key)) return;
      expect(Math.abs(cif.U_re[i] - preset.couplingRe.get(key)!)).toBeLessThan(1e-5 * uMax);
      expect(Math.abs(cif.U_im[i] - preset.couplingIm.get(key)!)).toBeLessThan(1e-5 * uMax);
      compared++;
    });
    expect(compared).toBeGreaterThan(1000);
    expect(cif.u0_imag / preset.u0_imag).toBeCloseTo(1, 5);
  });
});
