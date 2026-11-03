import * as React from 'react';
import { act } from 'react';
import { createRender } from '@anywidget/react';
import { expect, test, vi } from 'vitest';
import { useLocalTiltWorkspace } from './localTilt';
import { WIDGET_TYPOGRAPHY } from '../theme';

vi.mock('../display/colormaps', () => ({
  COLORMAPS: { magma: new Uint8Array(768) },
  createGPUColormapEngine: async () => null,
  renderToOffscreen: () => document.createElement('canvas'),
}));

class Model {
  constructor(public values: Record<string, unknown>) {}
  listeners = new Map<string, Set<() => void>>();
  get(key: string) { return this.values[key]; }
  set(key: string, value: unknown) {
    this.values[key] = value;
    this.listeners.get(`change:${key}`)?.forEach(listener => listener());
  }
  save_changes() {}
  on(key: string, listener: () => void) {
    if (!this.listeners.has(key)) this.listeners.set(key, new Set());
    this.listeners.get(key)!.add(listener);
  }
  off(key: string, listener: () => void) { this.listeners.get(key)?.delete(listener); }
}

function Workspace() {
  const workspace = useLocalTiltWorkspace({
    Panel: ({ label }: { label: string }) => React.createElement('div', null, label),
    tc: {}, ui: { typography: WIDGET_TYPOGRAPHY }, overviewSize: 512,
    zoom: { zoom: 1, panX: 0, panY: 0 }, setZoom() {},
    cmap: 'magma', smooth: false, contrastRange: [1, 99],
  });
  return React.createElement(React.Fragment, null,
    workspace?.toolbar, workspace?.overlay, workspace?.comparison, workspace?.settings);
}

test('clicking a result targets its own rectangle, bounds and refit', async () => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  const regions = [
    { id: 'region-2', label: 'Region 2', bounds: [32, 96, 16, 80] },
    { id: 'region-3', label: 'Region 3', bounds: [160, 288, 160, 288] },
  ];
  const model = new Model({
    local_tilt_enabled: true, local_tilt_active_region: 'region-3',
    local_tilt_can_fit: true, local_tilt_stale: false,
    local_tilt_selection_json: JSON.stringify(regions), local_tilt_reply_json: '',
    local_tilt_json: JSON.stringify({ regions, scan_shape: [512, 512], fits: regions.map(region => ({
      ...region, sample_tilt_mrad: [1, 2], seed_sd_mrad: 0.1,
    })) }),
    local_tilt_images_json: JSON.stringify(regions.map((region, index) => ({
      label: `${region.label} · SSB`, shape: [2, 2], offset: index * 4, sampling_A: 0.5,
    }))),
    local_tilt_bytes: new DataView(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]).buffer),
  });
  const el = document.createElement('div'); document.body.append(el);
  let cleanup: (() => void) | undefined;
  try {
    await act(async () => { cleanup = createRender(Workspace)({ model, el } as never) as () => void; });
    for (const region of regions) {
      const panel = Array.from(el.querySelectorAll('[data-local-tilt-result]'))
        .find(element => element.textContent?.includes(`${region.label} · SSB`))!;
      act(() => panel.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true })));
      expect(model.get('local_tilt_active_region')).toBe(region.id);
      const selected = el.querySelector('svg rect[stroke-width="2"]')!;
      expect(selected.parentElement?.querySelector('text')?.textContent).toBe(region.label);
      expect((el.querySelector('[aria-label="Row start"]') as HTMLInputElement).value)
        .toBe(String(region.bounds[0]));
    }
    const fit = Array.from(el.querySelectorAll('button')).find(button => button.textContent === 'Fit Region')!;
    act(() => fit.click());
    expect(JSON.parse(model.get('local_tilt_request_json') as string)).toMatchObject({
      action: 'fit', only_changed: true, force_region: 'region-3', regions,
    });
    expect((el.querySelector('[aria-label="Delete Region 2"]') as HTMLButtonElement).disabled).toBe(true);
    act(()=>model.set('local_tilt_reply_json',JSON.stringify({status:'complete',signature:JSON.stringify(regions)})));
    const previousRequest=model.get('local_tilt_request_json');
    // The hover action deletes its own panel without changing another region's controls.
    const hoverDelete=el.querySelector('[aria-label="Delete Region 2"]') as HTMLButtonElement;
    act(()=>hoverDelete.dispatchEvent(new MouseEvent('pointerdown',{bubbles:true})));
    expect(model.get('local_tilt_active_region')).toBe('region-3');
    act(()=>hoverDelete.click());
    expect(JSON.parse(model.get('local_tilt_selection_json') as string)).toEqual([regions[1]]);
    expect(model.get('local_tilt_active_region')).toBe('region-3');
    expect(el.querySelectorAll('[data-local-tilt-result]')).toHaveLength(1);
    expect(model.get('local_tilt_request_json')).toBe(previousRequest);
    act(()=>model.set('local_tilt_selection_json',JSON.stringify(regions)));
    const deleteButton=el.querySelector('[aria-label="Delete selected region"]') as HTMLButtonElement;
    act(()=>deleteButton.click());
    expect(JSON.parse(model.get('local_tilt_selection_json') as string)).toEqual([regions[0]]);
    expect(el.querySelectorAll('[data-local-tilt-result]')).toHaveLength(1);
    act(()=>el.querySelector('[data-local-tilt-result]')!.dispatchEvent(new MouseEvent('pointerdown',{bubbles:true})));
    const editor=el.querySelector('svg[aria-label="Local tilt region editor"]') as SVGSVGElement;
    act(()=>editor.dispatchEvent(new KeyboardEvent('keydown',{key:'Delete',bubbles:true})));
    expect(JSON.parse(model.get('local_tilt_selection_json') as string)).toEqual([]);
    expect(el.querySelectorAll('[data-local-tilt-result]')).toHaveLength(0);
    expect(model.get('local_tilt_request_json')).toBe(previousRequest);

    // Draw back toward the upper left after deleting the last rectangle.
    act(()=>Array.from(el.querySelectorAll('button')).find(button=>button.textContent==='Draw')!.click());
    editor.setPointerCapture=vi.fn();
    act(()=>editor.dispatchEvent(new MouseEvent('pointerdown',{clientX:480,clientY:480,bubbles:true,button:0})));
    act(()=>editor.dispatchEvent(new MouseEvent('pointermove',{clientX:320,clientY:352,bubbles:true})));
    act(()=>editor.dispatchEvent(new MouseEvent('pointerup',{clientX:320,clientY:352,bubbles:true})));
    const drawn=JSON.parse(model.get('local_tilt_selection_json') as string);
    expect(drawn).toHaveLength(1);
    expect(drawn[0].bounds).toEqual([352,480,320,480]);
    expect(model.get('local_tilt_active_region')).toBe(drawn[0].id);
    expect(Array.from(el.querySelectorAll('button')).find(button=>button.textContent==='Move')!.getAttribute('aria-pressed')).toBe('true');
  } finally {
    act(() => cleanup?.()); el.remove(); vi.unstubAllGlobals();
  }
});

test('from Overview, clicking beside the rectangles moves none of them', async () => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  const regions = [
    { id: 'region-1', label: 'Region 1', bounds: [0, 64, 0, 64] },
    { id: 'region-2', label: 'Region 2', bounds: [160, 288, 160, 288] },
  ];
  const selection = JSON.stringify(regions);
  const model = new Model({
    local_tilt_enabled: true, local_tilt_active_region: '',
    local_tilt_can_fit: true, local_tilt_stale: false,
    local_tilt_selection_json: selection, local_tilt_reply_json: '',
    local_tilt_json: JSON.stringify({ regions, scan_shape: [512, 512], fits: [] }),
    local_tilt_images_json: '[]', local_tilt_bytes: new DataView(new ArrayBuffer(0)),
  });
  const el = document.createElement('div'); document.body.append(el);
  let cleanup: (() => void) | undefined;
  try {
    await act(async () => { cleanup = createRender(Workspace)({ model, el } as never) as () => void; });
    // Overview: no rectangle is highlighted, has corner handles or bound inputs.
    expect(el.querySelector('svg rect[stroke-width="2"]')).toBeNull();
    expect(el.querySelector('[aria-label="Row start"]')).toBeNull();
    const editor = el.querySelector('svg[aria-label="Local tilt region editor"]') as SVGSVGElement;
    editor.setPointerCapture = vi.fn();
    expect(Array.from(el.querySelectorAll('button')).find(button => button.textContent === 'Move')!
      .getAttribute('aria-pressed')).toBe('true');
    act(() => editor.dispatchEvent(new MouseEvent('pointerdown', { clientX: 400, clientY: 400, bubbles: true, button: 0 })));
    act(() => editor.dispatchEvent(new MouseEvent('pointerup', { clientX: 400, clientY: 400, bubbles: true })));
    expect(model.get('local_tilt_selection_json')).toBe(selection);
    expect(model.get('local_tilt_active_region')).toBe('');
    // Clicking a rectangle itself still selects it, with its own bounds.
    const second = el.querySelectorAll('svg g rect')[1] as SVGRectElement;
    act(() => second.dispatchEvent(new MouseEvent('pointerdown', { clientX: 200, clientY: 200, bubbles: true, button: 0 })));
    act(() => editor.dispatchEvent(new MouseEvent('pointerup', { clientX: 200, clientY: 200, bubbles: true })));
    expect(model.get('local_tilt_active_region')).toBe('region-2');
    expect((el.querySelector('[aria-label="Row start"]') as HTMLInputElement).value).toBe('160');
    expect(JSON.parse(model.get('local_tilt_selection_json') as string)[0]).toEqual(regions[0]);
  } finally {
    act(() => cleanup?.()); el.remove(); vi.unstubAllGlobals();
  }
});
