import { act } from 'react';
import { afterEach, expect, test, vi } from 'vitest';
import widget from './index';
import * as colormaps from '../display/colormaps';

/** The Python trait defaults (``ShowPtycho.class_traits(sync=True)``) a live session starts from. */
const DEFAULTS: Record<string, unknown> = {
  C10_max_nm: 40, C10_min_nm: -40, auto_sampling_note: '', sampling_limit_note: '', auto_upsample: 1,
  calibration_path: '', calibration_saved_at: '', calibration_status: '',
  crop_refit_available: false, crop_refit_request_json: '', crop_refit_status: '',
  fitted_C10_nm: 0, fitted_C12_nm: 0, fitted_loss: null, fitted_phi12_deg: 0,
  fitted_sample_agreement: null, fitted_tilt_fit_gain: null,
  fitted_higher_order_aberrations_json: '{}',
  fitted_sample_json: '{}', fitted_scan_detector_rotation_deg: 0, display_negate_phase: false,
  higher_order_aberrations_json: '{}', initial_fft_on: false, local_tilt_active_region: '',
  local_tilt_can_fit: false, local_tilt_can_reconstruct: false, local_tilt_enabled: false,
  local_tilt_images_json: '[]', local_tilt_json: '', local_tilt_max_columns: 3,
  local_tilt_preview_json: '', local_tilt_reply_json: '', local_tilt_request_json: '',
  local_tilt_selection_json: '', local_tilt_stale: false, notes: '', num_bright_field_pixels: 0,
  object_sampling_A: [], phase_height: 0, phase_width: 0,
  pin_json: '', request_json: '', result_json: '', sample_available: false, sample_fit_json: '',
  sample_fit_request: 0, sample_fit_status: '', sample_json: '{}', save_trigger: 0, scan_cols: 0,
  scan_detector_rotation_deg: 0, scan_detector_rotation_max_deg: 180,
  scan_detector_rotation_min_deg: -180, scan_rows: 0, scan_sampling_A: [], session_note: '',
  ptycho_session: true, stars_path: '', tilt_correction: true, trials_json: '',
  upsample_available: false, upsample: 'auto', webgpu_cal_json: '',
  webgpu_h5_source_json: '', webgpu_preview_enabled: false, webgpu_preview_status: '',
  webgpu_standalone: false,
  amplitude_bytes: new DataView(new ArrayBuffer(0)), local_tilt_bytes: new DataView(new ArrayBuffer(0)),
};

class Model {
  values: Record<string, unknown>;
  sent: any[] = [];
  listeners = new Map<string, Set<() => void>>();
  constructor(values: Record<string, unknown>) { this.values = { ...DEFAULTS, ...values }; }
  get(key: string) { return this.values[key]; }
  set(key: string, value: unknown) {
    this.values[key] = value;
    this.listeners.get(`change:${key}`)?.forEach(listener => listener());
  }
  save_changes() {}
  send(content: unknown) { this.sent.push(content); }
  on(key: string, listener: () => void) {
    if (!this.listeners.has(key)) this.listeners.set(key, new Set());
    this.listeners.get(key)!.add(listener);
  }
  off(key: string, listener: () => void) { this.listeners.get(key)?.delete(listener); }
}

const START = { C10_nm: 3, C12_nm: 1, phi12_deg: 10 };
const REGION = { id: 'a', label: 'Upper', bounds: [0, 64, 0, 64] };
const FIT = { ...REGION, aberrations: START, sample_tilt_mrad: [4, -2], depth_spread_nm: 12, seed_sd_mrad: 0.1 };

function session(values: Record<string, unknown> = {}) {
  return new Model({
    phase_width: 4, phase_height: 4, phase_bytes: new DataView(new Float32Array(16).fill(0.25).buffer),
    scan_rows: 128, scan_cols: 128, scan_sampling_A: [0.5, 0.5], sample_available: true,
    fitted_C10_nm: 3, fitted_C12_nm: 1, fitted_phi12_deg: 10,
    result_json: JSON.stringify({ id: 0, ...START, loss: 0.5 }),
    local_tilt_json: JSON.stringify({ scan_shape: [128, 128], regions: [REGION], fits: [FIT], map: [FIT] }),
    local_tilt_selection_json: JSON.stringify([REGION]),
    local_tilt_enabled: true, local_tilt_active_region: 'a', local_tilt_can_fit: true,
    ...values,
  });
}

let cleanup: (() => void) | undefined;
let el: HTMLDivElement;
async function mount(model: Model) {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  el = document.createElement('div'); document.body.append(el);
  await act(async () => { cleanup = widget.render({ model, el } as never) as unknown as () => void; });
}
afterEach(() => { act(() => cleanup?.()); cleanup = undefined; el?.remove(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

/** The C10 value thumb's hidden range input, as a keyboard or drag moves it. */
function dragC10(value: number) {
  const slider = Array.from(el.querySelectorAll('.MuiSlider-root'))
    .find(root => root.parentElement?.textContent?.includes('C10'))!;
  const input = slider.querySelector('input[data-index="1"]') as HTMLInputElement;
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, String(value));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

test('a running regional fit disables every aberration and sample slider', async () => {
  const model = session({ local_tilt_reply_json: JSON.stringify({ status: 'running' }) });
  await mount(model);
  // Every slider in the controls except the two-thumb display-contrast histogram.
  const sliders = () => Array.from(el.querySelectorAll('fieldset .MuiSlider-root'))
    .filter(root => root.querySelectorAll('input').length !== 2);
  const sampleToggle = Array.from(el.querySelectorAll('fieldset p'))
    .find(node => node.textContent?.includes('Sample tilt')) as HTMLElement;
  act(() => { sampleToggle.parentElement!.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
  const locked = sliders();
  // C10, C12 and phi12 (rotation is shared, so hidden) plus tilt row, tilt col and depth spread.
  expect(locked.map(root => root.querySelector('input')!.getAttribute('aria-label') ?? '').filter(Boolean))
    .toEqual(['tilt row', 'tilt col', 'depth spread']);
  expect(locked).toHaveLength(6);
  // MUI ignores pointer and keyboard input only through its own disabled prop.
  expect(locked.every(root => root.classList.contains('Mui-disabled'))).toBe(true);
  expect(locked.flatMap(root => Array.from(root.querySelectorAll('input'))).every(input => input.disabled)).toBe(true);
  act(() => model.set('local_tilt_reply_json', JSON.stringify({ status: 'complete' })));
  expect(sliders().some(root => root.classList.contains('Mui-disabled'))).toBe(false);
});

test('a failed regional preview drops queued drags and says why', async () => {
  const model = session();
  await mount(model);
  dragC10(4);
  dragC10(4.5);  // queued behind the first request
  expect(model.sent).toHaveLength(1);
  const first = model.sent[0].request;
  expect(first).toMatchObject({ region_id: 'a', C10_nm: 4 });
  act(() => model.set('local_tilt_preview_json', JSON.stringify({ id: first.id, region_id: 'a', error: 'Region moved. Refit it before adjusting its optics.' })));
  expect(el.querySelector('[role="alert"]')?.textContent).toContain('Region moved');
  expect(model.sent).toHaveLength(1);  // the queued drag belonged to the failed state
  dragC10(5);
  expect(model.sent).toHaveLength(2);
  act(() => model.set('local_tilt_preview_json', JSON.stringify({ id: model.sent[1].request.id, region_id: 'a', loss: null })));
  expect(model.sent).toHaveLength(2);  // nothing stale is replayed afterwards
  expect(el.querySelector('[role="alert"]')).toBeNull();
});

test('a late regional reply does not replace the overview loss', async () => {
  const model = session();
  await mount(model);
  dragC10(4);
  const request = model.sent[0].request;
  act(() => model.set('local_tilt_active_region', ''));  // back to Overview before the reply
  expect(el.textContent).toContain('0.50000000');
  act(() => model.set('local_tilt_preview_json', JSON.stringify({ id: request.id, region_id: 'a', loss: 9.75 })));
  expect(el.textContent).toContain('0.50000000');
  expect(el.textContent).not.toContain('9.75000000');
  // The reply still released the queue: the next overview drag is sent at once.
  dragC10(6);
  expect(model.sent).toHaveLength(2);
  expect(model.sent[1].request.region_id).toBeUndefined();
});

test('the fit readout shows the model agreement and tilt gain, never the phase variance', async () => {
  const model = session({
    local_tilt_enabled: false, local_tilt_active_region: '', fitted_loss: 0.0046,
    result_json: JSON.stringify({ id: 0, ...START, loss: null }),
    trials_json: JSON.stringify([{ rank: 0, ...START, loss: -4.0e17, objective: 'agreement' }]),
  });
  await mount(model);
  expect(el.textContent).toContain('Fit agreement not measured');
  expect(el.textContent).not.toContain('variance');
  expect(el.textContent).not.toContain('undefined');
  act(() => { model.set('fitted_sample_agreement', 0.8123); model.set('fitted_tilt_fit_gain', 1.25); });
  expect(el.textContent).toContain('Fit agreement 0.8123');
  expect(el.textContent).toContain('tilt gain 1.25×');
  expect(el.textContent).not.toContain('0.0046');
});

test('trial preview and restore use each complete fitted model, including tilt and rotation', async () => {
  const higher = { C30_nm: 1500, C50_nm: 20000 };
  const sample = { sample_tilt_mrad: [1, -2], depth_spread_nm: 10 };
  const model = session({
    local_tilt_enabled: false, local_tilt_active_region: '', tilt_correction: false,
    fitted_scan_detector_rotation_deg: 20,
    fitted_sample_json: JSON.stringify(sample), sample_json: JSON.stringify(sample),
    fitted_higher_order_aberrations_json: JSON.stringify(higher),
    higher_order_aberrations_json: JSON.stringify({ C30_nm: 2500 }),
    trials_json: JSON.stringify([{
      rank: 0, ...START, loss: -0.6, objective: 'agreement',
      sample_tilt_mrad: [0.5, 1.5], depth_spread_nm: 6,
      higher_order: {}, scan_detector_rotation_deg: 200,
    }]),
  });
  await mount(model);
  const stripToggle = Array.from(el.querySelectorAll('div')).find(node => node.childElementCount === 0 && node.textContent?.startsWith('▶ Trial previews'))!;
  act(() => stripToggle.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  const trial = Array.from(el.querySelectorAll('p')).find(node => node.textContent === '#1')!;
  act(() => trial.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  const first = model.sent.at(-1).request;
  expect(first).toMatchObject({ higher_order: {}, tilt_correction: true, scan_detector_rotation_deg: 200,
    sample: { sample_tilt_mrad: [0.5, 1.5], depth_spread_nm: 6 } });
  act(() => model.set('result_json', JSON.stringify({ id: first.id, ...START, loss: null })));
  act(() => model.set('tilt_correction', false));
  act(() => (el.querySelector('[aria-label="Restore joint fit"]') as HTMLButtonElement).click());
  expect(model.sent.at(-1).request).toMatchObject({ ...START, higher_order: higher, sample,
    tilt_correction: true, scan_detector_rotation_deg: 20 });
  expect(model.get('tilt_correction')).toBe(true);
});

test('joint fit becomes available again after fifth-order completion', async () => {
  const model = session({ local_tilt_enabled: false, local_tilt_active_region: '' });
  await mount(model);
  const toggle = Array.from(el.querySelectorAll('p')).find(node => node.textContent?.includes('Sample tilt'))!;
  act(() => toggle.parentElement!.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  const button = Array.from(el.querySelectorAll('button')).find(node => node.textContent === 'Joint fit')!;
  act(() => button.click());
  expect(button.disabled).toBe(true);
  act(() => model.set('sample_fit_status', 'Fifth-order fit: tilt (1.0, -2.0) mrad · depth spread 10.0 nm.'));
  expect(button.disabled).toBe(false);
  expect(button.textContent).toBe('Joint fit');
});

test('wheel zoom works through region overlays and stays independent of other panels and fits', async () => {
  // jsdom has no canvas renderer; keep real ImagePanel interaction and state.
  const context = new Proxy({
    createImageData: (width: number, height: number) => ({ data: new Uint8ClampedArray(width * height * 4) }),
    measureText: () => ({ width: 10 }),
  }, { get: (target, key) => Reflect.get(target, key) ?? (() => {}) });
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as never);
  vi.stubGlobal('ImageBitmap', class {});
  vi.spyOn(colormaps, 'renderToOffscreen').mockImplementation((_values, width, height) => {
    const canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    return canvas;
  });
  const lower = { id: 'b', label: 'Lower', bounds: [64, 128, 64, 128] };
  const regions = [REGION, lower];
  const model = session({
    initial_fft_on: true,
    local_tilt_json: JSON.stringify({ scan_shape: [128, 128], regions, fits: [FIT, { ...FIT, ...lower }] }),
    local_tilt_selection_json: JSON.stringify(regions),
    local_tilt_images_json: JSON.stringify(regions.map((region, index) => ({
      label: `${region.label} · Direct`, shape: [4, 4], offset: index * 16, sampling_A: 0.5,
    }))),
    local_tilt_bytes: new DataView(new Float32Array(32).fill(0.25).buffer),
  });
  await mount(model);
  const science = { ...model.values };
  const panel = (label: string) => el.querySelector(`canvas[aria-label="${label}"]`)!.parentElement!;
  const overview = panel('Overview');
  const upper = panel('Upper · Direct');
  const bottom = panel('Lower · Direct');
  const wheel = async (target: Element, deltaY = -100) => {
    const event = new WheelEvent('wheel', { deltaY, bubbles: true, cancelable: true });
    act(() => { target.dispatchEvent(event); });
    await act(async () => { await new Promise(requestAnimationFrame); });
    return event;
  };
  for (const mode of ['Draw', 'Move', 'Pan']) {
    act(() => { Array.from(el.querySelectorAll('button')).find(button => button.textContent === mode)!.click(); });
    const target = overview.querySelector(mode === 'Pan' ? 'canvas' : 'svg')!;
    expect((await wheel(target)).defaultPrevented).toBe(true);
    expect(overview.textContent).toContain('1.1x');
    expect(upper.textContent).not.toMatch(/\dx/);
    expect(bottom.textContent).not.toMatch(/\dx/);
    await wheel(target, 100);
    expect(overview.textContent).not.toMatch(/\dx/);
  }
  await wheel(upper.querySelector('canvas')!);
  expect(upper.textContent).toContain('1.1x');
  expect(bottom.textContent).not.toMatch(/\dx/);
  // A burst accumulates every wheel delta but publishes only on the next frame.
  act(() => {
    for (let i = 0; i < 2; i++) bottom.querySelector('canvas')!.dispatchEvent(
      new WheelEvent('wheel', { deltaY: -100, bubbles: true, cancelable: true }));
  });
  expect(bottom.textContent).not.toMatch(/\dx/);
  await act(async () => { await new Promise(requestAnimationFrame); });
  expect(bottom.textContent).toContain('1.3x');
  expect(upper.textContent).toContain('1.1x');
  expect(overview.textContent).not.toMatch(/\dx/);
  const inset = overview.querySelector('[data-showptycho-fft-inset]')!;
  expect(inset).not.toBeNull();
  const initialFFTZoom = Number(inset.getAttribute('data-view-zoom'));
  expect((await wheel(inset)).defaultPrevented).toBe(true);
  expect(Number(inset.getAttribute('data-view-zoom'))).toBeGreaterThan(initialFFTZoom);
  act(() => inset.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));
  expect(Number(inset.getAttribute('data-view-zoom'))).toBeCloseTo(initialFFTZoom);
  expect(overview.textContent).not.toMatch(/\dx/);
  // Zooming never queues a refit, changes a selection, or alters fit/calibration traits.
  expect(model.values).toEqual(science);
  expect(model.sent).toEqual([]);
});

test('pinning an unscored reconstruction preserves a missing diagnostic', async () => {
  const context = new Proxy({
    createImageData: (width: number, height: number) => ({ data: new Uint8ClampedArray(width * height * 4) }),
    measureText: () => ({ width: 10 }),
  }, { get: (target, key) => Reflect.get(target, key) ?? (() => {}) });
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as never);
  vi.stubGlobal('ImageBitmap', class {});
  vi.spyOn(colormaps, 'renderToOffscreen').mockImplementation((_values, width, height) => {
    const canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    return canvas;
  });
  const model = session({
    local_tilt_enabled: false, local_tilt_active_region: '', compute_loss: false,
    result_json: JSON.stringify({ id: 0, ...START, loss: null }),
  });
  await mount(model);
  await act(async () => { await new Promise(requestAnimationFrame); });
  const pin = el.querySelector('button[aria-label="Pin current snapshot"]') as HTMLButtonElement;
  expect(pin.disabled).toBe(false);
  act(() => pin.click());
  expect(JSON.parse(model.get('pin_json') as string)).toMatchObject({ action: 'pin', loss: null });
});

test('sampling settings explain the Auto cap and let the user explicitly choose 8×', async () => {
  const note = 'Auto 4× (sampling supports 8×; nominal 5.08×)';
  const model = session({ upsample_available: true, auto_upsample: 4,
    sampling_limit_note: note, local_tilt_enabled: false });
  await mount(model);
  act(() => { (el.querySelector('[aria-label="View settings"]') as HTMLElement).click(); });
  expect(document.body.textContent).toContain(note);
  expect(document.body.textContent).toContain('Auto (4×)');
  act(() => { document.querySelector('[role="combobox"][aria-label="Upsampling"]')!
    .dispatchEvent(new MouseEvent('mousedown', { bubbles: true, button: 0 })); });
  const eight = Array.from(document.querySelectorAll('[role="option"]'))
    .find(option => option.textContent === '8×') as HTMLElement;
  expect(eight).toBeDefined();
  act(() => eight.click());
  expect(model.get('upsample')).toBe(8);
  expect(document.body.textContent).not.toContain(note);
});

test('the stats row follows a held C10 slider before release', async () => {
  const model = session({ local_tilt_enabled: false, local_tilt_active_region: '' });
  await mount(model);
  const chip = () => Array.from(el.querySelectorAll('p'))
    .find(node => node.textContent?.startsWith('C10 ') && node.textContent.endsWith(' nm'))!;
  expect(chip()).toBeDefined();
  dragC10(6.3);
  expect(chip().textContent).toBe('C10 6.3 nm');
});
