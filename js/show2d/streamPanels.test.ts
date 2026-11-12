import { afterEach, expect, test, vi } from "vitest";
import { StreamPanelStore } from "./streamPanels";

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

const bytes = (...values: number[]) => new DataView(new Uint8Array(values).buffer);

test("changed panels patch the images and are acknowledged", () => {
  vi.useFakeTimers();
  const model = fakeModel({ frame_bytes: bytes(1, 1, 2, 2, 3, 3), _stream_revision: 4, _stream_seq: 0 });
  const store = new StreamPanelStore(model);
  expect(model.send.mock.calls[0][0]).toEqual({ type: "show2d_frontend_ready", version: 1, revision: 4, bytes: 6 });
  model.emit("msg:custom", { type: "show2d_stream_panels", version: 1, seq: 1, revision: 4, slots: [1], block: 2, bytes: 6 },
    [bytes(9, 9)]);
  expect([...new Uint8Array(store.view!.buffer)]).toEqual([1, 1, 9, 9, 3, 3]);
  vi.advanceTimersByTime(150);
  expect(model.send.mock.lastCall![0]).toEqual({ type: "show2d_stream_ack", version: 1, seq: 1 });
  // A full update replaces the patched copy and acknowledges its own number.
  Object.assign(model.state, { frame_bytes: bytes(5, 5), _stream_revision: 5, _stream_seq: 2 });
  model.emit("change:_stream_revision");
  expect(store.view).toBeNull();
  vi.advanceTimersByTime(150);
  expect(model.send.mock.lastCall![0]).toEqual({ type: "show2d_stream_ack", version: 1, seq: 2 });
  store.dispose();
});

test("a patch for another revision asks for the full images once", () => {
  const model = fakeModel({ frame_bytes: bytes(1, 1), _stream_revision: 1, _stream_seq: 0 });
  const store = new StreamPanelStore(model);
  const patch = { type: "show2d_stream_panels", version: 1, seq: 1, revision: 0, slots: [0], block: 2, bytes: 2 };
  model.emit("msg:custom", patch, [bytes(7, 7)]);
  model.emit("msg:custom", { ...patch, seq: 2 }, [bytes(8, 8)]);
  expect(model.send.mock.calls.filter(([sent]) => sent.type === "show2d_stream_resync")).toHaveLength(1);
  expect(store.view).toBeNull();
  store.dispose();
});
