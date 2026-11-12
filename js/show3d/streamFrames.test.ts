import { afterEach, expect, test, vi } from "vitest";
import { StreamFrameStore, streamedStackExtends } from "./streamFrames";

afterEach(() => { vi.useRealTimers(); });

function fakeModel(state: Record<string, unknown>) {
  const listeners = new Map<string, Set<(...args: any[]) => void>>();
  return {
    state,
    send: vi.fn(),
    get: (key: string) => state[key],
    on: (event: string, fn: (...args: any[]) => void) => {
      if (!listeners.has(event)) listeners.set(event, new Set());
      listeners.get(event)!.add(fn);
    },
    off: (event: string, fn: (...args: any[]) => void) => listeners.get(event)?.delete(fn),
    emit: (event: string, ...args: any[]) => listeners.get(event)?.forEach(fn => fn(...args)),
  };
}

const frame = (value: number, pixels = 4) => new DataView(new Float32Array(pixels).fill(value).buffer);
const floats = (view: DataView | null) => (view ? [...new Float32Array(view.buffer, view.byteOffset, view.byteLength / 4)] : []);

test("appended frames extend the stack and are acknowledged once rendered", () => {
  vi.useFakeTimers();
  const model = fakeModel({ _offline_float_stack: frame(1), height: 2, width: 2, frame_seq: 3, is_rgb: false });
  const store = new StreamFrameStore(model);
  expect(model.send.mock.calls[0][0]).toEqual({ type: "show3d_frontend_ready", version: 1, frames: 1, revision: 3 });
  const base = model.state._offline_float_stack as DataView;
  model.emit("msg:custom", { type: "show3d_stream_frames", version: 1, seq: 1, start: 1, count: 2, rows: 2, cols: 2, revision: 3 },
    [frame(2), frame(3)]);
  expect(floats(store.view)).toEqual([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
  expect(streamedStackExtends(base, store.view)).toBe(1);
  vi.advanceTimersByTime(150);
  expect(model.send.mock.lastCall![0]).toEqual({ type: "show3d_stream_ack", version: 1, seq: 1 });
  store.dispose();
});

test("a missed transfer asks for the whole stack once, then adopts it", () => {
  const model = fakeModel({ _offline_float_stack: new DataView(new ArrayBuffer(0)), height: 0, width: 0, frame_seq: 0, is_rgb: false });
  const store = new StreamFrameStore(model);
  const message = { type: "show3d_stream_frames", version: 1, seq: 2, start: 3, count: 1, rows: 2, cols: 2, revision: 0 };
  model.emit("msg:custom", message, [frame(5)]);
  model.emit("msg:custom", { ...message, seq: 3, start: 4 }, [frame(6)]);
  expect(model.send.mock.calls.filter(([sent]) => sent.type === "show3d_stream_resync")).toHaveLength(1);
  Object.assign(model.state, { _offline_float_stack: frame(7, 16), height: 2, width: 2, frame_seq: 1 });
  model.emit("change:frame_seq");
  expect(store.view).toBeNull();
  model.emit("msg:custom", { ...message, seq: 4, start: 4, revision: 1 }, [frame(8)]);
  expect(floats(store.view).slice(-4)).toEqual([8, 8, 8, 8]);
  store.dispose();
});
