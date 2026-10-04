import { describe, it, expect } from 'vitest';
import { simulationGeometry } from './geometry';
const settings = {voltage_kV:300,semiangle_mrad:30,focus_depth_nm:20,thickness_nm:60,detector_px:192,detector_mrad_per_px:.5570968023269496,scan_step_A:.373,scan_size_px:128,tilt_mrad:[0,0]};
describe('Simulation cell planning', () => {
 it('uses scan-center span and separate potential pixels', () => {
  const g=simulationGeometry(settings,[3.991,3.991,4.0352],[48,48],96,5);
  expect(g.span).toBeCloseTo(47.371,10);
  expect(g.extent[0]).toBeCloseTo(191.568,10);
  expect(g.sampling[0]).toBeCloseTo(.0415729166667,10);
  expect(g.gpts).toEqual([4608,4608]);
  expect(g.zRepeats).toBe(149);expect(g.fits).toBe(true);
  expect(simulationGeometry(settings,[3.991,3.991,4.0352],[12,12],96,5).fits).toBe(false);
 });
 it('includes directional tilt and explicit guard', () => {
  const base=simulationGeometry(settings,[3.991,3.991,4.0352],[24,24],96,5);
  const tilted=simulationGeometry({...settings,tilt_mrad:[20,0]},[3.991,3.991,4.0352],[24,24],96,5);
  expect(tilted.margins[0]).toBeLessThan(base.margins[0]);
  expect(tilted.margins[1]).toBe(base.margins[1]);
  expect(simulationGeometry(settings,[3.991,3.991,4.0352],[24,24],96,100).fits).toBe(false);
 });
});

it("rejects invalid live planning inputs before computing coverage", () => {
  for (const repeats of [[0, 4], [2.5, 4], [4]]) {
    expect(() => simulationGeometry(settings, [4, 4, 4], repeats, 96, 5)).toThrow(/repeats/);
  }
  expect(() => simulationGeometry(settings, [4, 4, 4], [4, 4], 0, 5)).toThrow(/Pixels/);
  expect(() => simulationGeometry(settings, [4, 4, 4], [4, 4], 96, NaN)).toThrow(/Guard/);
});
