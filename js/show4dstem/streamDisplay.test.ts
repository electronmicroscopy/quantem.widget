import { afterEach, expect, test, vi } from "vitest";
import { connectStreamDisplay, initialStreamPanels, streamPanelChanges } from "./streamDisplay";
import { createElement, useEffect, useLayoutEffect, useState, useSyncExternalStore } from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";

afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

test("saved full state renders without a comm; duplicate live views wait only after transport deltas", async () => {
  for (const pending of [false, true]) {
    const canonical = new DataView(new Float32Array([7, 9]).buffer);
    const state: Record<string, unknown> = { compare_virtual_image_bytes: canonical, compare_image_revision: 3,
      _stream_source_ids: ["A", "B"], compare_panel_indices: [0, 1], offline: false,
      _stream_panel_update: pending ? {snapshot_pending: true} : {} };
    const listeners = new Map<string, (...args: any[]) => void>();
    const model = {get: (key: string) => state[key], send: vi.fn(), // A restored dead comm sends no response.
      on: (event: string, fn: (...args: any[]) => void) => listeners.set(event, fn),
      off: (event: string) => listeners.delete(event)};
    function View() {
      const [image, setImage] = useState<DataView | null | undefined>(() => initialStreamPanels(model));
      useEffect(() => connectStreamDisplay(model, setImage), []);
      return createElement("div", null, image === undefined ? "pending" : [...new Float32Array((image ?? canonical).buffer)].join(","));
    }
    const host = document.createElement("div");
    const root = createRoot(host);
    try {
      flushSync(() => root.render(createElement(View)));
      await new Promise(resolve => setTimeout(resolve, 0));
      expect(host.textContent).toBe(pending ? "pending" : "7,9");
      expect(model.send.mock.calls[0][0].type).toBe("show4dstem_frontend_ready");
      if (pending) {
        state._stream_panel_update = {};
        state.compare_image_revision = 4;
        flushSync(() => listeners.get("change:compare_image_revision")!());
        expect(host.textContent).toBe("7,9");
      }
    } finally {flushSync(() => root.unmount());}
  }
});

test("structural deltas install exact retained pixels with their new identities in one render", async () => {
  const original = new DataView(new Uint32Array([0x80000000, 0x7fc00001]).buffer);
  const state: Record<string, any> = { compare_virtual_image_bytes: original, compare_image_revision: 1,
    shape_rows: 1, shape_cols: 1, compare_panel_count: 2, compare_panel_indices: [1, 0], _stream_source_ids: ["A", "B"] };
  const listeners = new Map<string, Set<(...args: any[]) => void>>();
  const model = {get: (key: string) => state[key], send: vi.fn(),
    on: (event: string, fn: (...args: any[]) => void) => { if (!listeners.has(event)) listeners.set(event, new Set()); listeners.get(event)!.add(fn); },
    off: (event: string, fn: (...args: any[]) => void) => listeners.get(event)?.delete(fn)};
  const emit = (event: string, ...args: any[]) => listeners.get(event)?.forEach(fn => fn(...args));
  const observed: {ids: string[]; pixels: number[]}[] = [];
  function View() {
    const [image, setImage] = useState<DataView | null | undefined>(null);
    const ids = useSyncExternalStore(fn => {model.on("change:_stream_source_ids", fn); return () => model.off("change:_stream_source_ids", fn);},
      () => model.get("_stream_source_ids"));
    useEffect(() => connectStreamDisplay(model, setImage), []);
    useLayoutEffect(() => {
      if (image === undefined) { observed.push({ids: [], pixels: []}); return; }
      observed.push({ids: state.compare_panel_indices.map((index: number) => ids[index]),
        pixels: [...new Uint32Array((image ?? state.compare_virtual_image_bytes).buffer)]});
    });
    return createElement("div");
  }
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  flushSync(() => root.render(createElement(View)));
  try {
    const ready = model.send.mock.calls[0][0];
    expect(ready.stream_layout).toBe(true);
    const apply = (attributes: Record<string, unknown>) => {
      Object.assign(state, attributes); // WidgetModel installs every attribute before emitting changes.
      Object.keys(attributes).forEach(key => emit(`change:${key}`));
    };
    apply({_stream_source_ids: ["A", "B", "C"], compare_panel_count: 3, compare_panel_indices: [2, 1, 0],
      _stream_panel_update: {base: 1, revision: 2, previous_ids: ["B", "A"], source_ids: ["C", "B", "A"],
        reuse: [-1, 0, 1], slots: [0], buffers: [new DataView(new Uint32Array([3]).buffer)], rows: 1, cols: 1}});
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(observed.at(-1)).toEqual({ids: ["C", "B", "A"], pixels: [3, 0x80000000, 0x7fc00001]});
    apply({_stream_source_ids: ["B", "C"], compare_panel_count: 2, compare_panel_indices: [1, 0],
      _stream_panel_update: {base: 2, revision: 3, previous_ids: ["C", "B", "A"], source_ids: ["C", "B"],
        reuse: [0, 1], slots: [], buffers: [], rows: 1, cols: 1}});
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(observed.at(-1)).toEqual({ids: ["C", "B"], pixels: [3, 0x80000000]});
    expect(observed.every(value => value.ids.length === value.pixels.length)).toBe(true);
    expect(state.compare_virtual_image_bytes).toBe(original); // No rebuilt stack enters model.set/save_changes.
    apply({_stream_panel_update: {...state._stream_panel_update, base: 0, revision: 4}});
    expect(model.send.mock.lastCall![0].type).toBe("show4dstem_stream_resync");
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(observed.at(-1)).toEqual({ids: [], pixels: []});
    const restored = new DataView(new Uint32Array([3, 0x80000000]).buffer);
    // Full snapshots clear the transport patch. Both change-event orders are safe.
    for (const attributes of [
      {_stream_panel_update: {}, compare_virtual_image_bytes: restored, compare_image_revision: 4},
      {compare_image_revision: 5, compare_virtual_image_bytes: restored, _stream_panel_update: {}},
    ]) {
      apply(attributes);
      await new Promise(resolve => setTimeout(resolve, 0));
      expect(observed.at(-1)).toEqual({ids: ["C", "B"], pixels: [3, 0x80000000]});
    }
  } finally {flushSync(() => root.unmount()); host.remove();}
});

test("stream receipts follow model delivery, ignore stale views, and survive hidden tabs", () => {
  vi.useFakeTimers();
  vi.stubGlobal("window", new EventTarget());
  const callbacks = new Map<number, FrameRequestCallback>();
  let id = 0;
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => { callbacks.set(++id, callback); return id; });
  vi.stubGlobal("cancelAnimationFrame", (key: number) => callbacks.delete(key));
  const model = { get: vi.fn(), send: vi.fn(), on: vi.fn(), off: vi.fn() };
  const disconnect = connectStreamDisplay(model);
  const ready = model.send.mock.calls[0][0];
  const receive = model.on.mock.calls[0][1];
  const update = { type: "show4dstem_stream_update", version: 1, view: ready.view, seq: 1 };
  receive({ ...update, view: "stale" });
  expect(callbacks.size).toBe(0);
  receive(update);
  expect(model.send).toHaveBeenCalledTimes(1);
  callbacks.get(id)!(16);
  expect(model.send.mock.lastCall![0]).toEqual({ ...update, type: "show4dstem_stream_ack" });
  vi.advanceTimersByTime(100);
  expect(model.send).toHaveBeenCalledTimes(2);
  receive(update);
  expect(callbacks.size).toBe(0);
  receive({ ...update, seq: 2 });
  vi.advanceTimersByTime(100); // No animation frames in a hidden notebook tab.
  expect(model.send.mock.lastCall![0].seq).toBe(2);
  receive({ ...update, seq: 3 });
  disconnect();
  vi.advanceTimersByTime(100);
  expect(model.send).toHaveBeenCalledTimes(4);
  expect(model.send.mock.lastCall![0].type).toBe("show4dstem_frontend_closed");
  expect(callbacks.size).toBe(0);
  expect(model.off).toHaveBeenCalledWith("msg:custom", receive);
});

test("a surviving view answers control-only probes and reconnects after browser history restores it", () => {
  const window = new EventTarget();
  vi.stubGlobal("window", window);
  const model = { get: vi.fn(), send: vi.fn(), on: vi.fn(), off: vi.fn() };
  const disconnect = connectStreamDisplay(model);
  const ready = model.send.mock.calls[0][0];
  const receive = model.on.mock.calls[0][1];
  receive({ type: "show4dstem_stream_probe", version: 1, seq: 9 });
  expect(model.send.mock.lastCall![0]).toEqual({ type: "show4dstem_stream_alive", version: 1, view: ready.view, seq: 9 });
  window.dispatchEvent(new Event("pagehide"));
  expect(model.send.mock.lastCall![0]).toEqual({ type: "show4dstem_frontend_closed", version: 1, view: ready.view });
  window.dispatchEvent(Object.assign(new Event("pageshow"), { persisted: true }));
  expect(model.send.mock.lastCall![0]).toEqual(ready);
  disconnect();
  const calls = model.send.mock.calls.length;
  window.dispatchEvent(new Event("pagehide"));
  expect(model.send).toHaveBeenCalledTimes(calls);
});

test("panel deltas preserve every float32 bit, skip unchanged panels, and recover a stale snapshot", () => {
  vi.useFakeTimers();
  vi.stubGlobal("window", new EventTarget());
  vi.stubGlobal("requestAnimationFrame", () => 1);
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  const original = new DataView(new Uint32Array([0x7fc00001, 0x80000000, 2, 3, 4, 5]).buffer);
  const state: Record<string, unknown> = { compare_virtual_image_bytes: original, compare_image_revision: 4,
    shape_rows: 1, shape_cols: 2, compare_panel_count: 3 };
  const model = { get: (name: string) => state[name], send: vi.fn(), on: vi.fn(), off: vi.fn() };
  const display = vi.fn();
  const disconnect = connectStreamDisplay(model, display);
  const ready = model.send.mock.calls[0][0];
  expect(ready.stream_delta).toBe(true);
  const receive = model.on.mock.calls.find(([name]) => name === "msg:custom")![1];
  const full = model.on.mock.calls.find(([name]) => name === "change:compare_virtual_image_bytes")![1];
  const delta = { type: "show4dstem_stream_panels", version: 1, base: 4, revision: 5,
    panels: 3, rows: 1, cols: 2, slots: [1] };
  const changed = new DataView(new Uint32Array([0x7fc00002, 0x80000000]).buffer);
  receive(delta, [changed]);
  const next = display.mock.lastCall![0] as DataView;
  expect([...new Uint32Array(next.buffer)]).toEqual([0x7fc00001, 0x80000000, 0x7fc00002, 0x80000000, 4, 5]);
  expect(streamPanelChanges(next)?.base.deref()).toBe(original);
  expect([...streamPanelChanges(next)!.slots]).toEqual([1]);
  expect(state.compare_virtual_image_bytes).toBe(original); // Never echoes a reconstructed stack to Python.
  receive({ ...delta, base: 5, revision: 6, slots: [] }, []);
  expect(display).toHaveBeenCalledTimes(1);
  receive({ ...delta, base: 6, revision: 7 }, [changed]);
  expect(display).toHaveBeenCalledTimes(2);
  receive({ ...delta, base: 1, revision: 8 }, [changed]);
  receive({ ...delta, base: 1, revision: 8 }, [changed]);
  expect(model.send.mock.calls.filter(([message]) => message.type === "show4dstem_stream_resync")).toHaveLength(1);
  receive({ type: "show4dstem_stream_update", version: 1, view: ready.view, seq: 1 });
  vi.advanceTimersByTime(100);
  expect(model.send.mock.calls.some(([message]) => message.type === "show4dstem_stream_ack")).toBe(false);
  state.compare_image_revision = 8;
  state.compare_virtual_image_bytes = next;
  receive({ type: "show4dstem_stream_update", version: 1, view: ready.view, seq: 2, revision: 8 });
  vi.advanceTimersByTime(100);
  expect(model.send.mock.calls.some(([message]) => message.type === "show4dstem_stream_ack")).toBe(false);
  full();
  expect(display.mock.lastCall![0]).toBeNull();
  vi.advanceTimersByTime(100); // A marker arriving before async full-state delivery is not lost.
  expect(model.send.mock.lastCall![0]).toEqual({ type: "show4dstem_stream_ack", version: 1, view: ready.view, seq: 2 });
  receive({ ...delta, base: 8, revision: 9 }, [changed]);
  expect(display.mock.lastCall![0]).toBeInstanceOf(DataView);
  receive({ type: "show4dstem_stream_update", version: 1, view: ready.view, seq: 3, revision: 10 });
  vi.advanceTimersByTime(100);
  expect(model.send.mock.lastCall![0].seq).toBe(2);
  receive({ ...delta, base: 9, revision: 10, slots: [] }, []);
  vi.advanceTimersByTime(100);
  expect(model.send.mock.lastCall![0]).toEqual({ type: "show4dstem_stream_ack", version: 1, view: ready.view, seq: 3 });
  disconnect();
});

test("two views reconstruct the same deltas and only the receipt owner admits the next update", () => {
  vi.useFakeTimers();
  vi.stubGlobal("window", new EventTarget());
  vi.stubGlobal("requestAnimationFrame", () => 1);
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  const state: Record<string, unknown> = { compare_virtual_image_bytes: new DataView(new Float32Array([1, 2]).buffer),
    compare_image_revision: 1, shape_rows: 1, shape_cols: 1, compare_panel_count: 2 };
  const model = { get: (name: string) => state[name], send: vi.fn(), on: vi.fn(), off: vi.fn() };
  const displays = [vi.fn(), vi.fn()];
  const disconnects = displays.map(display => connectStreamDisplay(model, display));
  const views = model.send.mock.calls.map(([message]) => message.view);
  const receivers = model.on.mock.calls.filter(([name]) => name === "msg:custom").map(([, callback]) => callback);
  receivers.forEach(receive => receive({ type: "show4dstem_stream_panels", version: 1,
    base: 1, revision: 2, panels: 2, rows: 1, cols: 1, slots: [1] }, [new DataView(new Float32Array([9]).buffer)]));
  displays.forEach(display => expect([...new Float32Array(display.mock.lastCall![0].buffer)]).toEqual([1, 9]));
  let marker = { type: "show4dstem_stream_update", version: 1, view: views[1], seq: 1, revision: 2 };
  receivers.forEach(receive => receive(marker));
  receivers.forEach(receive => receive({ type: "show4dstem_stream_panels", version: 1,
    base: 2, revision: 3, panels: 2, rows: 1, cols: 1, slots: [0] }, [new DataView(new Float32Array([11]).buffer)]));
  displays.forEach(display => expect([...new Float32Array(display.mock.lastCall![0].buffer)]).toEqual([11, 9]));
  marker = {...marker, seq: 2, revision: 3};
  receivers.forEach(receive => receive(marker));
  vi.advanceTimersByTime(100);
  expect(model.send.mock.calls.filter(([message]) => message.type === "show4dstem_stream_ack")
    .map(([message]) => [message.view, message.seq])).toEqual([[views[1], 2]]);
  disconnects[1]();
  receivers[0]({ ...marker, view: views[0] });
  vi.advanceTimersByTime(100);
  expect(model.send.mock.lastCall![0].view).toBe(views[0]);
  disconnects[0]();
});

test("identical full snapshots advance receipts and replace a private ROI snapshot after a product view", () => {
  vi.useFakeTimers();
  vi.stubGlobal("window", new EventTarget());
  vi.stubGlobal("requestAnimationFrame", () => 1);
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  const original = new DataView(new Float32Array([1, 2]).buffer);
  const state: Record<string, unknown> = { compare_virtual_image_bytes: original, compare_image_revision: 1,
    shape_rows: 1, shape_cols: 1, compare_panel_count: 2, vi_source: "roi" };
  const model = { get: (name: string) => state[name], send: vi.fn(), on: vi.fn(), off: vi.fn() };
  const display = vi.fn();
  const disconnect = connectStreamDisplay(model, display);
  const ready = model.send.mock.calls[0][0];
  const callback = (name: string) => model.on.mock.calls.find(([event]) => event === name)![1];
  const receive = callback("msg:custom");
  const marker = (seq: number, revision: number) => receive({ type: "show4dstem_stream_update", version: 1,
    view: ready.view, seq, revision });
  receive({ type: "show4dstem_stream_panels", version: 1, base: 1, revision: 2,
    rows: 1, cols: 1, panels: 2, slots: [1] }, [new DataView(new Float32Array([9]).buffer)]);
  expect([...new Float32Array(display.mock.lastCall![0].buffer)]).toEqual([1, 9]);

  state.vi_source = "DPC_row"; // Product views ignore the retained ROI snapshot.
  state.vi_source = "roi";
  // Jupyter installs every attribute before emitting change events. Its
  // DataView deep equality suppresses the bytes event for this identical ROI.
  state.compare_virtual_image_bytes = new DataView(new Float32Array([1, 2]).buffer);
  state.compare_image_revision = 3;
  callback("change:compare_image_revision")();
  expect(display.mock.lastCall![0]).toBeNull(); // The fresh full image now wins over [1, 9].
  marker(1, 3);
  vi.advanceTimersByTime(100);
  expect(model.send.mock.lastCall![0].type).toBe("show4dstem_stream_ack");

  receive({ type: "show4dstem_stream_panels", version: 1, base: 0, revision: 4,
    rows: 1, cols: 1, panels: 2, slots: [] }, []);
  expect(model.send.mock.lastCall![0].type).toBe("show4dstem_stream_resync");
  // The marker precedes async application of an otherwise identical resync.
  // Full resends advance the revision, guaranteeing one scalar change event.
  marker(2, 4);
  vi.advanceTimersByTime(100);
  expect(model.send.mock.lastCall![0].type).toBe("show4dstem_stream_resync");
  state.compare_image_revision = 4;
  callback("change:compare_image_revision")();
  vi.advanceTimersByTime(100);
  expect(model.send.mock.lastCall![0]).toEqual({ type: "show4dstem_stream_ack", version: 1, view: ready.view, seq: 2 });

  state.compare_virtual_image_bytes = new DataView(new ArrayBuffer(0));
  state.compare_image_revision = 5;
  callback("change:compare_virtual_image_bytes")();
  // Keeping the comparison grid hidden advances only its revision thereafter.
  state.compare_image_revision = 6;
  callback("change:compare_image_revision")();
  marker(3, 6);
  vi.advanceTimersByTime(100);
  expect(model.send.mock.lastCall![0].seq).toBe(3);
  disconnect();
  for (const [name, listener] of model.on.mock.calls) expect(model.off).toHaveBeenCalledWith(name, listener);
});

test.each([
  {slots: null}, {slots: {}}, {buffers: [null]}, {buffers: [{}]},
  {rows: 0}, {cols: 0.5}, {panels: -1}, {slots: [0, 0]},
])("malformed panel delta requests a fresh snapshot and then recovers: %j", invalid => {
  const original = new DataView(new Float32Array([1, 2]).buffer);
  const state: Record<string, unknown> = {compare_virtual_image_bytes: original, compare_image_revision: 1,
    shape_rows: 1, shape_cols: 1, compare_panel_count: 2};
  const model = {get: (name: string) => state[name], send: vi.fn(), on: vi.fn(), off: vi.fn()};
  const display = vi.fn();
  const disconnect = connectStreamDisplay(model, display);
  try {
    const receive = model.on.mock.calls.find(([name]) => name === "msg:custom")![1];
    const full = model.on.mock.calls.find(([name]) => name === "change:compare_image_revision")![1];
    const {buffers = [new DataView(new Float32Array([9]).buffer)], ...fields} = invalid as Record<string, any>;
    const delta = {type: "show4dstem_stream_panels", version: 1, base: 1, revision: 2,
      rows: 1, cols: 1, panels: 2, slots: [1]};
    expect(() => receive({...delta, ...fields}, buffers)).not.toThrow();
    expect(model.send.mock.lastCall![0].type).toBe("show4dstem_stream_resync");
    expect(display.mock.lastCall![0]).toBeUndefined();
    state.compare_image_revision = 2;
    full();
    receive({...delta, base: 2, revision: 3}, [new DataView(new Float32Array([9]).buffer)]);
    expect([...new Float32Array(display.mock.lastCall![0].buffer)]).toEqual([1, 9]);
  } finally {disconnect();}
});

test("unmounted views discard queued messages and dispose once", () => {
  vi.useFakeTimers();
  vi.stubGlobal("window", new EventTarget());
  const model = {get: vi.fn(), send: vi.fn(), on: vi.fn(), off: vi.fn()};
  const disconnect = connectStreamDisplay(model);
  const receive = model.on.mock.calls[0][1];
  const view = model.send.mock.calls[0][0].view;
  disconnect();
  disconnect();
  receive({type: "show4dstem_stream_update", version: 1, seq: 1, view});
  receive({type: "show4dstem_stream_probe", version: 1, seq: 1});
  vi.advanceTimersByTime(100);
  expect(model.send.mock.calls.map(([message]) => message.type)).toEqual([
    "show4dstem_frontend_ready", "show4dstem_frontend_closed",
  ]);
});
