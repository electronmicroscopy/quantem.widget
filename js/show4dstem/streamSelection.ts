type Model = { get(name: string): any; set(name: string, value: unknown): void; save_changes(): unknown };

/** Live selection uses acquisition identity; a retained slot can be reused. */
export function selectStreamFrame(model: Model, index: number, identities: string[], otherwise: () => void) {
  if (!identities.length || model.get("offline")) { otherwise(); return; }
  const source = identities[index];
  if (source === undefined) return;
  model.set("_stream_selected_source", source);
  model.save_changes();
}

export function interactionContext(model: Pick<Model, "get">, keys: string) {
  return keys.split(" ").map(key => key === "frame_idx"
    ? model.get("_stream_selected_source") || model.get(key) : model.get(key));
}

/** Keep a captured tile mounted when a scan is prepended or another is retired. */
export function comparisonPanelKey(frame: number, identities: string[]) {
  return identities[frame] ?? frame;
}

/** View the selected displayed image without duplicating a multi-scan buffer. */
export function comparisonImageView(
  images: DataView | null | undefined, indices: number[], frame: number, rows: number, cols: number,
): DataView | null {
  const slot = indices.indexOf(frame);
  const bytes = rows * cols * Float32Array.BYTES_PER_ELEMENT;
  if (!images || slot < 0 || bytes <= 0 || (slot + 1) * bytes > images.byteLength) return null;
  return new DataView(images.buffer, images.byteOffset + slot * bytes, bytes);
}
