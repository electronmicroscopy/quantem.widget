import * as React from "react";
import { liveAcknowledger, sharedPerModel, type LiveModel } from "../shared/liveDisplay";

/**
 * Show2D.stream panel patches: a `show2d_stream_panels` message carries only
 * the panels whose bytes changed. They are written over a copy of the current
 * images, which stands in for the `frame_bytes` trait; Python keeps the full
 * images in that trait without resending it. Any full update (a new
 * `_stream_revision`) replaces the local copy.
 */
type PanelsMessage = {
  type?: string; version?: number; seq?: number; revision?: number;
  slots?: number[]; block?: number; bytes?: number;
};

export class StreamPanelStore {
  view: DataView | null = null;
  private resyncing = false;
  private readonly listeners = new Set<() => void>();
  private readonly acks: ReturnType<typeof liveAcknowledger>;

  constructor(private readonly model: LiveModel) {
    this.acks = liveAcknowledger(seq => model.send({ type: "show2d_stream_ack", version: 1, seq }));
    model.on("msg:custom", this.receive);
    model.on("change:frame_bytes", this.adopt);
    model.on("change:_stream_revision", this.adopt);
    const bytes = (model.get("frame_bytes") as DataView | null)?.byteLength ?? 0;
    model.send({ type: "show2d_frontend_ready", version: 1, revision: Number(model.get("_stream_revision") || 0), bytes });
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }

  dispose(): void {
    this.acks.dispose();
    this.model.off("msg:custom", this.receive);
    this.model.off("change:frame_bytes", this.adopt);
    this.model.off("change:_stream_revision", this.adopt);
  }

  private publish(view: DataView | null): void {
    this.view = view;
    this.listeners.forEach(listener => listener());
  }

  /** A full image update: draw the trait again and acknowledge it when it was a streamed publication. */
  private adopt = (): void => {
    this.resyncing = false;
    this.publish(null);
    this.acks.received(this.model.get("_stream_seq"));
  };

  private resync(): void {
    if (!this.resyncing) this.model.send({ type: "show2d_stream_resync", version: 1 });
    this.resyncing = true;
  }

  private receive = (message: PanelsMessage, buffers: DataView[] = []): void => {
    if (!message || message.type !== "show2d_stream_panels" || message.version !== 1 || this.resyncing) return;
    const { slots, block, bytes, revision, seq } = message;
    const current = this.view ?? (this.model.get("frame_bytes") as DataView | null);
    if (!Array.isArray(slots) || !Number.isSafeInteger(block) || !Number.isSafeInteger(bytes) || block! <= 0
        || revision !== Number(this.model.get("_stream_revision") || 0)
        || !current || current.byteLength !== bytes || slots.length !== buffers.length
        || slots.some((slot, index) => !Number.isSafeInteger(slot) || slot < 0 || (slot + 1) * block! > bytes!
          || !(buffers[index] instanceof DataView) || buffers[index].byteLength !== block)) {
      this.resync(); // a patch that does not fit the images held here: ask for them whole once
      return;
    }
    const next = new Uint8Array(bytes!);
    next.set(new Uint8Array(current.buffer, current.byteOffset, current.byteLength));
    slots.forEach((slot, index) => {
      next.set(new Uint8Array(buffers[index].buffer, buffers[index].byteOffset, block!), slot * block!);
    });
    // Kept out of model.set(): a later save_changes() would echo every image back to Python.
    this.publish(new DataView(next.buffer));
    this.acks.received(seq);
  };
}

const stores = new WeakMap<object, { store: StreamPanelStore; users: number }>();

/** The patched images, or null while the `frame_bytes` trait is current. */
export function useStreamPanels(model: LiveModel): DataView | null {
  const [view, setView] = React.useState<DataView | null>(null);
  React.useEffect(() => {
    const { store, release } = sharedPerModel(stores, model, () => new StreamPanelStore(model));
    const update = () => setView(store.view);
    const unsubscribe = store.subscribe(update);
    update();
    return () => { unsubscribe(); release(); };
  }, [model]);
  return view;
}
