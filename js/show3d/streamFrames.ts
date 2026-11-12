import * as React from "react";
import { liveAcknowledger, sharedPerModel, type LiveModel } from "../shared/liveDisplay";

/**
 * Show3D.stream frames: each `show3d_stream_frames` message carries only the
 * appended frames. They are copied into a growing float32 stack that stands in
 * for the `_offline_float_stack` trait, which Python leaves unchanged while
 * streaming. A full trait update (frame_seq changes) replaces the local stack.
 */
type FramesMessage = {
  type?: string; version?: number; seq?: number; start?: number; count?: number;
  rows?: number; cols?: number; revision?: number;
};

/** Which stack a streamed DataView extends: same generation means the earlier frames are unchanged. */
const stackOrigins = new WeakMap<DataView, { generation: number; frames: number }>();

/**
 * Frames `previous` already held when `next` only appended to it, else null.
 * Lets the GPU residency keep uploaded frames and upload just the new ones.
 */
export function streamedStackExtends(previous: DataView | null | undefined, next: DataView | null | undefined): number | null {
  if (!previous || !next || previous === next) return null;
  const before = stackOrigins.get(previous);
  const after = stackOrigins.get(next);
  if (!before || !after || before.generation !== after.generation || after.frames <= before.frames) return null;
  return before.frames;
}

export class StreamFrameStore {
  view: DataView | null = null;
  private bytes: Uint8Array | null = null;
  private frames = 0;
  private rows = 0;
  private cols = 0;
  private generation = 0;
  private resyncing = false;
  private readonly listeners = new Set<() => void>();
  private readonly acks: ReturnType<typeof liveAcknowledger>;

  constructor(private readonly model: LiveModel) {
    this.acks = liveAcknowledger(seq => model.send({ type: "show3d_stream_ack", version: 1, seq }));
    model.on("msg:custom", this.receive);
    model.on("change:_offline_float_stack", this.adopt);
    model.on("change:frame_seq", this.adopt);
    this.adopt();
    model.send({ type: "show3d_frontend_ready", version: 1, frames: this.frames, revision: Number(model.get("frame_seq") || 0) });
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }

  dispose(): void {
    this.acks.dispose();
    this.model.off("msg:custom", this.receive);
    this.model.off("change:_offline_float_stack", this.adopt);
    this.model.off("change:frame_seq", this.adopt);
  }

  /** Start again from the model's full stack. */
  private adopt = (): void => {
    const stack = this.model.get("_offline_float_stack") as DataView | null;
    this.rows = Number(this.model.get("height") || 0);
    this.cols = Number(this.model.get("width") || 0);
    const frameBytes = this.rows * this.cols * 4;
    this.frames = stack && frameBytes > 0 && !this.model.get("is_rgb") ? Math.floor(stack.byteLength / frameBytes) : 0;
    this.generation += 1;
    this.bytes = null;
    this.resyncing = false;
    if (stack) stackOrigins.set(stack, { generation: this.generation, frames: this.frames });
    this.publish(null);
  };

  private resync(): void {
    if (!this.resyncing) this.model.send({ type: "show3d_stream_resync", version: 1 });
    this.resyncing = true;
  }

  private publish(view: DataView | null): void {
    this.view = view;
    this.listeners.forEach(listener => listener());
  }

  private receive = (message: FramesMessage, buffers: DataView[] = []): void => {
    if (!message || message.type !== "show3d_stream_frames" || message.version !== 1 || this.resyncing) return;
    const { start, count, rows, cols, seq, revision } = message;
    if (![start, count, rows, cols].every(value => Number.isSafeInteger(value) && (value as number) >= 0)
        || !count || !rows || !cols || buffers.length !== count) { this.resync(); return; }
    const frameBytes = rows! * cols! * 4;
    if (revision !== Number(this.model.get("frame_seq") || 0)
        || buffers.some(buffer => !(buffer instanceof DataView) || buffer.byteLength !== frameBytes)) { this.resync(); return; }
    if (start === 0 && this.frames === 0) {
      this.rows = rows!;
      this.cols = cols!;
      this.bytes = null;
    } else if (start !== this.frames || rows !== this.rows || cols !== this.cols) {
      this.resync(); // a transfer was missed: ask for the whole stack once
      return;
    }
    const used = this.frames * frameBytes;
    const needed = used + count! * frameBytes;
    if (!this.bytes || this.bytes.byteLength < needed) {
      // Double the capacity so appending N frames copies O(N) frames in total.
      const grown = new Uint8Array(Math.max(needed, 2 * (this.bytes?.byteLength ?? 0)));
      if (this.bytes) grown.set(this.bytes.subarray(0, used));
      else if (used) {
        const base = this.model.get("_offline_float_stack") as DataView;
        grown.set(new Uint8Array(base.buffer, base.byteOffset, used));
      }
      this.bytes = grown;
    }
    buffers.forEach((buffer, index) => {
      this.bytes!.set(new Uint8Array(buffer.buffer, buffer.byteOffset, frameBytes), used + index * frameBytes);
    });
    this.frames += count!;
    const view = new DataView(this.bytes.buffer, 0, this.frames * frameBytes);
    stackOrigins.set(view, { generation: this.generation, frames: this.frames });
    this.publish(view);
    this.acks.received(seq);
  };
}

const stores = new WeakMap<object, { store: StreamFrameStore; users: number }>();

/** The streamed display stack, or null while the `_offline_float_stack` trait is current. */
export function useStreamFrames(model: LiveModel): DataView | null {
  const [view, setView] = React.useState<DataView | null>(null);
  React.useEffect(() => {
    const { store, release } = sharedPerModel(stores, model, () => new StreamFrameStore(model));
    const update = () => setView(store.view);
    const unsubscribe = store.subscribe(update);
    update();
    return () => { unsubscribe(); release(); };
  }, [model]);
  return view;
}
