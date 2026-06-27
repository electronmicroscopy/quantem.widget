/** Convert anywidget DataView/ArrayBuffer to Uint8Array. */
export function extractBytes(dataView: DataView | ArrayBuffer | Uint8Array): Uint8Array {
  if (dataView instanceof Uint8Array) return dataView;
  if (dataView instanceof ArrayBuffer) return new Uint8Array(dataView);
  if (dataView && "buffer" in dataView) {
    return new Uint8Array(dataView.buffer, dataView.byteOffset, dataView.byteLength);
  }
  return new Uint8Array(0);
}

/** Extract Float32Array from anywidget DataView. Returns null if empty.
 *
 * `expectedFloats` lets callers ignore trailing pad bytes. Some widget buffers
 * are padded to a multiple of three bytes so notebook/html base64 embeds do not
 * need `=` padding; the logical float payload still has a known element count.
 */
export function extractFloat32(dataView: DataView | ArrayBuffer | Uint8Array, expectedFloats?: number): Float32Array | null {
  const bytes = extractBytes(dataView);
  if (bytes.length === 0) return null;
  const usableBytes = expectedFloats !== undefined
    ? Math.max(0, Math.min(bytes.byteLength, Math.floor(expectedFloats) * 4))
    : bytes.byteLength;
  if (usableBytes === 0 || usableBytes % 4 !== 0) return null;
  if (bytes.byteOffset % 4 !== 0 || usableBytes !== bytes.byteLength) {
    const aligned = new Uint8Array(usableBytes);
    aligned.set(bytes.subarray(0, usableBytes));
    return new Float32Array(aligned.buffer);
  }
  return new Float32Array(bytes.buffer, bytes.byteOffset, usableBytes / 4);
}

/** Download a Blob as a file. */
export function downloadBlob(blob: Blob, filename: string): void {
  const link = document.createElement("a");
  link.download = filename;
  const url = URL.createObjectURL(blob);
  link.href = url;
  link.click();
  // Defer revocation to ensure browser has time to start the download
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

/** Download a DataView as a file (e.g. GIF/ZIP from Python). */
export function downloadDataView(dataView: DataView, filename: string, mimeType: string): void {
  const buf = new Uint8Array(dataView.buffer as ArrayBuffer, dataView.byteOffset, dataView.byteLength);
  downloadBlob(new Blob([buf as BlobPart], { type: mimeType }), filename);
}

/** Format number with exponential notation for large/small values. */
export function formatNumber(val: number, decimals: number = 2): string {
  if (val === 0) return "0";
  if (Math.abs(val) >= 1000 || Math.abs(val) < 0.01) return val.toExponential(decimals);
  return val.toFixed(decimals);
}

type AnyWidgetModelWithManager = {
  model_id?: string;
  id?: string;
  save_changes?: (...args: unknown[]) => unknown;
  __quantemMarkDirtyPatch?: boolean;
  widget_manager?: {
    get_state_sync?: (options?: Record<string, unknown>) => unknown;
    _modelsSync?: Map<string, { _comm_live?: boolean; comm_live?: boolean }>;
    __quantemSaveRestoredModelsPatch?: boolean;
    setDirty?: () => void;
    context?: NotebookContextLike;
    _context?: NotebookContextLike;
  };
};

type NotebookContextLike = {
  model?: {
    dirty?: boolean;
  };
};

function markNotebookDirty(manager: AnyWidgetModelWithManager["widget_manager"]): void {
  if (typeof manager?.setDirty === "function") {
    manager.setDirty();
    return;
  }
  for (const context of [manager?.context, manager?._context]) {
    const model = context?.model;
    if (model) {
      model.dirty = true;
      return;
    }
  }
}

export function markWidgetNotebookDirty(model: unknown): void {
  const widgetModel = model as AnyWidgetModelWithManager | null;
  markNotebookDirty(widgetModel?.widget_manager);
}

/**
 * JupyterLab's widget manager can restore notebook-embedded widget models without
 * a kernel, but its Cmd+S path only serializes models whose comm is live. That
 * drops restored no-kernel models from metadata.widgets. Temporarily treating
 * those restored models as live during serialization preserves their current
 * frontend state when the notebook is saved.
 */
export function preserveRestoredWidgetModelsOnSave(model: unknown): void {
  const widgetModel = model as AnyWidgetModelWithManager | null;
  const manager = widgetModel?.widget_manager;
  if (!manager || typeof manager.get_state_sync !== "function") return;
  const modelId = widgetModel?.model_id ?? widgetModel?.id;
  if (modelId && manager._modelsSync) {
    manager._modelsSync.set(modelId, widgetModel as { _comm_live?: boolean; comm_live?: boolean });
  }
  if (!widgetModel.__quantemMarkDirtyPatch && typeof widgetModel.save_changes === "function") {
    const originalSaveChanges = widgetModel.save_changes.bind(widgetModel);
    widgetModel.save_changes = (...args: unknown[]) => {
      const result = originalSaveChanges(...args);
      markNotebookDirty(manager);
      return result;
    };
    widgetModel.__quantemMarkDirtyPatch = true;
  }
  if (manager.__quantemSaveRestoredModelsPatch) return;

  const originalGetStateSync = manager.get_state_sync.bind(manager);
  manager.get_state_sync = (options?: Record<string, unknown>) => {
    const models = Array.from(manager._modelsSync?.values?.() ?? []);
    const restoredModels = models.filter((widgetModel) => !widgetModel.comm_live);
    for (const widgetModel of restoredModels) widgetModel._comm_live = true;
    try {
      return originalGetStateSync(options);
    } finally {
      for (const widgetModel of restoredModels) widgetModel._comm_live = false;
    }
  };
  manager.__quantemSaveRestoredModelsPatch = true;
}
