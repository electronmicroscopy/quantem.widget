import { detectorViewId } from "./detectorChannel";

type Message = { type?: string; version?: number; view?: string; seq?: number;
  base?: number; revision?: number; slots?: number[]; panels?: number; rows?: number; cols?: number };
type Model = {
  get(name: string): unknown;
  send(message: Record<string, unknown>): void;
  on(event: string, callback: (message: Message, buffers?: DataView[]) => void): void;
  off(event: string, callback: (message: Message, buffers?: DataView[]) => void): void;
};

const panelChanges = new WeakMap<DataView, { base: WeakRef<DataView>; slots: Set<number> }>();
export const streamPanelChanges = (image: DataView) => panelChanges.get(image);

export function initialStreamPanels(model: Pick<Model, "get">): null | undefined {
  // Saved/full state omits transport patches and needs no live comm to render.
  return (model.get("_stream_panel_update") as { snapshot_pending?: boolean })?.snapshot_pending
    ? undefined : null;
}

/** Bound streamed image traffic; acknowledgment means model receipt, not GPU presentation. */
export function connectStreamDisplay(model: Model, showPanels?: (image: DataView | null | undefined) => void): () => void {
  const view = detectorViewId();
  let active = true;
  let disposed = false;
  let latest = 0;
  let acknowledged = 0;
  let expectedRevision = 0;
  let frame: number | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let current = model.get("compare_virtual_image_bytes") as DataView;
  const panelIdentities = () => {
    const ids = (model.get("_stream_source_ids") || []) as string[];
    return ((model.get("compare_panel_indices") || []) as number[]).map(index => ids[index]);
  };
  let identities = panelIdentities();
  let revision = Number(model.get("compare_image_revision") || 0);
  let resyncing = false;
  const fullSnapshot = () => {
    if (disposed) return;
    current = model.get("compare_virtual_image_bytes") as DataView;
    identities = panelIdentities();
    revision = Number(model.get("compare_image_revision") || 0);
    resyncing = false;
    showPanels?.(null);
    if (latest > acknowledged) scheduleAcknowledgment();
  };
  const resync = () => {
    showPanels?.(undefined); // Hide outdated slots until a coherent full state arrives.
    if (!resyncing) model.send({ type: "show4dstem_stream_resync", version: 1, view });
    resyncing = true;
  };
  const cancel = () => {
    if (frame !== null) cancelAnimationFrame(frame);
    if (timer !== null) clearTimeout(timer);
    frame = timer = null;
  };
  const acknowledge = () => {
    cancel();
    if (!active || resyncing || revision < expectedRevision) return;
    acknowledged = latest;
    model.send({ type: "show4dstem_stream_ack", version: 1, view, seq: latest });
  };
  const scheduleAcknowledgment = () => {
    cancel();
    if (!active || resyncing) return;
    // Let React consume the preceding image buffers before admitting another.
    frame = requestAnimationFrame(acknowledge);
    timer = setTimeout(acknowledge, 100); // Hidden tabs pause rAF.
  };
  const layoutSnapshot = () => {
    if (disposed) return;
    const update = model.get("_stream_panel_update") as {
      base: number; revision: number; previous_ids: string[]; source_ids: string[];
      reuse: number[]; slots: number[]; buffers: DataView[]; rows: number; cols: number;
    } | undefined;
    if (!update?.revision || resyncing || !showPanels) return;
    const { base, revision: nextRevision, previous_ids, source_ids, reuse, slots, buffers, rows, cols } = update;
    if (![previous_ids, source_ids, reuse, slots, buffers].every(Array.isArray)
        || ![rows, cols].every(value => Number.isSafeInteger(value) && value > 0)
        || source_ids.some(identity => typeof identity !== "string" || !identity)
        || buffers.some(buffer => !(buffer instanceof DataView))) { resync(); return; }
    const panelBytes = rows * cols * 4;
    const target = panelIdentities();
    if (base !== revision || !Number.isSafeInteger(nextRevision) || nextRevision <= revision
        || !current || current.byteLength !== identities.length * panelBytes
        || rows !== model.get("shape_rows") || cols !== model.get("shape_cols")
        || JSON.stringify(previous_ids) !== JSON.stringify(identities)
        || JSON.stringify(source_ids) !== JSON.stringify(target)
        || new Set(source_ids).size !== source_ids.length
        || source_ids.length !== model.get("compare_panel_count") || reuse.length !== target.length
        || slots.length !== buffers.length || new Set(slots).size !== slots.length
        || reuse.some((source, index) => !Number.isSafeInteger(source) || source < -1
          || (source >= 0 && identities[source] !== target[index])
          || (source === -1) !== slots.includes(index))
        || slots.some((slot, index) => !Number.isSafeInteger(slot) || slot < 0 || slot >= target.length
          || buffers[index].byteLength !== panelBytes)) { resync(); return; }
    const bytes = new Uint8Array(target.length * panelBytes);
    reuse.forEach((source, index) => {
      if (source >= 0) bytes.set(new Uint8Array(current.buffer, current.byteOffset + source * panelBytes, panelBytes), index * panelBytes);
    });
    slots.forEach((slot, index) => bytes.set(new Uint8Array(
      buffers[index].buffer, buffers[index].byteOffset, panelBytes), slot * panelBytes));
    const next = new DataView(bytes.buffer);
    // All metadata is already installed by WidgetModel's state transaction.
    // Moved panels must upload to their new GPU slots even when counts match.
    panelChanges.set(next, { base: new WeakRef(current), slots: new Set(target.map((_, index) => index)) });
    current = next;
    identities = target;
    revision = nextRevision;
    showPanels(next);
    if (latest > acknowledged) scheduleAcknowledgment();
  };
  const receive = (message: Message, buffers: DataView[] = []) => {
    if (!active || !message || typeof message !== "object") return;
    if (message.type === "show4dstem_stream_panels" && message.version === 1 && showPanels) {
      const { base, revision: nextRevision, slots = [], panels = 0, rows = 0, cols = 0 } = message;
      if (resyncing) return;
      const panelBytes = rows * cols * 4;
      if (!Array.isArray(slots) || !Array.isArray(buffers)
          || buffers.some(buffer => !(buffer instanceof DataView))
          || ![rows, cols, panels].every(value => Number.isSafeInteger(value) && value > 0)
          || base !== revision || !Number.isSafeInteger(nextRevision) || nextRevision! <= revision
          || !current || current.byteLength !== panels * panelBytes
          || rows !== model.get("shape_rows") || cols !== model.get("shape_cols")
          || panels !== model.get("compare_panel_count") || slots.length !== buffers.length
          || new Set(slots).size !== slots.length
          || slots.some((slot, index) => !Number.isSafeInteger(slot) || slot < 0 || slot >= panels
            || buffers[index].byteLength !== panelBytes)) { resync(); return; }
      revision = nextRevision!;
      if (latest > acknowledged) scheduleAcknowledgment();
      if (!slots.length) return;
      const bytes = new Uint8Array(current.byteLength);
      bytes.set(new Uint8Array(current.buffer, current.byteOffset, current.byteLength));
      slots.forEach((slot, index) => bytes.set(new Uint8Array(
        buffers[index].buffer, buffers[index].byteOffset, panelBytes), slot * panelBytes));
      const next = new DataView(bytes.buffer);
      panelChanges.set(next, { base: new WeakRef(current), slots: new Set(slots) });
      current = next;
      // Keep remote image data out of model.set(): a later save_changes()
      // would otherwise echo the entire reconstructed stack back to Python.
      showPanels(next);
      return;
    }
    if (message.type === "show4dstem_stream_probe" && message.version === 1 && Number.isSafeInteger(message.seq)) {
      model.send({ type: "show4dstem_stream_alive", version: 1, view, seq: message.seq });
      return;
    }
    if (message.type !== "show4dstem_stream_update" || message.version !== 1
        || message.view !== view || !Number.isSafeInteger(message.seq) || message.seq! <= latest) return;
    latest = message.seq!;
    expectedRevision = Number(message.revision ?? revision);
    scheduleAcknowledgment();
  };
  const ready = () => {
    if (disposed) return;
    active = true;
    model.send({ type: "show4dstem_frontend_ready", version: 1,
      stream_ack: true, stream_delta: Boolean(showPanels), stream_layout: Boolean(showPanels), view });
  };
  const closed = () => {
    if (!active) return;
    active = false;
    cancel();
    model.send({ type: "show4dstem_frontend_closed", version: 1, view });
  };
  const resume = (event: PageTransitionEvent) => { if (event.persisted) ready(); };
  model.on("msg:custom", receive);
  model.on("change:compare_virtual_image_bytes", fullSnapshot);
  model.on("change:compare_image_revision", fullSnapshot);
  model.on("change:_stream_panel_update", layoutSnapshot);
  window.addEventListener("pagehide", closed);
  window.addEventListener("pageshow", resume);
  ready();
  return () => {
    if (disposed) return;
    disposed = true;
    closed();
    model.off("msg:custom", receive);
    model.off("change:compare_virtual_image_bytes", fullSnapshot);
    model.off("change:compare_image_revision", fullSnapshot);
    model.off("change:_stream_panel_update", layoutSnapshot);
    window.removeEventListener("pagehide", closed);
    window.removeEventListener("pageshow", resume);
  };
}
