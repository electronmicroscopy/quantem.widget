import { act } from 'react';
import { afterEach, expect, test, vi } from 'vitest';
import widget from './index';

/** The Python trait defaults (``ShowPtycho.class_traits(sync=True)``) a live session starts from. */
const DEFAULTS: Record<string, unknown> = {
  C10_max_nm: 40, C10_min_nm: -40, auto_sampling_note: '', auto_upsampling_factor: 1,
  calibration_path: '', calibration_saved_at: '', calibration_status: '',
  crop_refit_available: false, crop_refit_request_json: '', crop_refit_status: '',
  fitted_C10_nm: 0, fitted_C12_nm: 0, fitted_loss: 0, fitted_phi12_deg: 0,
  fitted_sample_json: '{}', fitted_scan_detector_rotation_deg: 0, display_negate_phase: false,
  higher_order_aberrations_json: '{}', initial_fft_on: false, local_tilt_active_region: '',
  local_tilt_can_fit: false, local_tilt_can_reconstruct: false, local_tilt_enabled: false,
  local_tilt_images_json: '[]', local_tilt_json: '', local_tilt_max_columns: 3,
  local_tilt_preview_json: '', local_tilt_reply_json: '', local_tilt_request_json: '',
  local_tilt_selection_json: '', local_tilt_stale: false, notes: '', num_bright_field_pixels: 0,
  num_drag_bright_field_pixels: 0, object_sampling_A: [], phase_height: 0, phase_width: 0,
  pin_json: '', request_json: '', result_json: '', sample_available: false, sample_fit_json: '',
  sample_fit_request: 0, sample_fit_status: '', sample_json: '{}', save_trigger: 0, scan_cols: 0,
  scan_detector_rotation_deg: 0, scan_detector_rotation_max_deg: 180,
  scan_detector_rotation_min_deg: -180, scan_rows: 0, scan_sampling_A: [], session_note: '',
  ssb_session: true, stars_path: '', tilt_correction: true, trials_json: '',
  upsampling_available: false, upsampling_factor: 'auto', webgpu_cal_json: '',
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
afterEach(() => { act(() => cleanup?.()); cleanup = undefined; el?.remove(); vi.unstubAllGlobals(); });

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
    .find(node => node.textContent?.includes('Sample tilt (thick SSB)')) as HTMLElement;
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
