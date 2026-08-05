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
  cid?: string;
  get?: (key: string) => unknown;
  set?: (values: Record<string, unknown>) => unknown;
  attributes?: Record<string, unknown>;
  save_changes?: (...args: unknown[]) => unknown;
  __quantemCurrentViewStateApplied?: boolean;
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

const CURRENT_WIDGET_VIEW_STATE_SCRIPT_ID = "quantem-current-widget-view-state";

type CurrentWidgetViewStatePayload = {
  version: 1;
  models: Record<string, Record<string, unknown>>;
};

function safeInlineJson(value: unknown): string {
  return JSON.stringify(value)
    .replace(/</g, "\\u003c")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
}

function stripCurrentWidgetViewState(html: string): string {
  return html.replace(
    /<script\b[^>]*\bid=["']quantem-current-widget-view-state["'][^>]*>[\s\S]*?<\/script>\s*/gi,
    "",
  );
}

function escapeHtmlAttr(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function elementAttributes(element: Element | null): string {
  if (!element) return "";
  const attrs = Array.from(element.attributes)
    .map((attr) => `${attr.name}="${escapeHtmlAttr(attr.value)}"`)
    .join(" ");
  return attrs ? ` ${attrs}` : "";
}

export function standaloneWidgetStaticHtmlFromDocument(): string {
  if (typeof document === "undefined") return "";
  const head = document.head?.cloneNode(true) as HTMLHeadElement | null;
  head?.querySelectorAll("script").forEach((script) => script.remove());
  head?.querySelectorAll("style:not(#quantem-widget-export-layout)").forEach((style) => style.remove());
  const scripts = Array.from(document.body?.querySelectorAll("script") ?? [])
    .filter((script) => {
      const type = script.getAttribute("type") || "";
      return Boolean(script.src)
        || script.id === "quantem-widget-anywidget-requirejs"
        || type === "application/vnd.jupyter.widget-state+json"
        || type === "application/vnd.jupyter.widget-view+json";
    })
    .map((script) => script.outerHTML)
    .join("\n");
  if (!scripts) return document.documentElement.outerHTML;
  return `<html${elementAttributes(document.documentElement)}><head>${head?.innerHTML ?? ""}</head><body${elementAttributes(document.body)}>\n${scripts}\n</body></html>`;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object") return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function isSerializableViewValue(value: unknown, depth = 0): boolean {
  if (depth > 6) return false;
  if (value === null) return true;
  const kind = typeof value;
  if (kind === "string" || kind === "boolean") return true;
  if (kind === "number") return Number.isFinite(value as number);
  if (Array.isArray(value)) {
    if (value.length > 20000) return false;
    return value.every((item) => isSerializableViewValue(item, depth + 1));
  }
  if (typeof ArrayBuffer !== "undefined") {
    if (value instanceof ArrayBuffer) return false;
    if (ArrayBuffer.isView(value as ArrayBufferView)) return false;
  }
  if (!isPlainObject(value)) return false;
  const entries = Object.entries(value);
  if (entries.length > 2000) return false;
  return entries.every(([, item]) => isSerializableViewValue(item, depth + 1));
}

function widgetModelId(model: AnyWidgetModelWithManager | null): string | null {
  const id = model?.model_id ?? model?.id ?? model?.cid;
  return typeof id === "string" && id.trim() ? id : null;
}

function singleWidgetViewModelId(html: string): string | null {
  const matches = Array.from(html.matchAll(
    /<script\b[^>]*type=["']application\/vnd\.jupyter\.widget-view\+json["'][^>]*>([\s\S]*?)<\/script>/gi,
  ));
  if (matches.length !== 1) return null;
  try {
    const payload = JSON.parse(matches[0][1]) as { model_id?: unknown };
    return typeof payload.model_id === "string" && payload.model_id.trim()
      ? payload.model_id
      : null;
  } catch {
    return null;
  }
}

function updateEmbeddedWidgetStateScript(
  html: string,
  modelId: string,
  viewState: Record<string, unknown>,
): string | null {
  if (typeof DOMParser === "undefined") return null;
  const doc = new DOMParser().parseFromString(html, "text/html");
  const script = doc.querySelector<HTMLScriptElement>(
    'script[type="application/vnd.jupyter.widget-state+json"]',
  );
  if (!script?.textContent) return null;
  let payload: { state?: Record<string, { state?: Record<string, unknown> }> };
  try {
    payload = JSON.parse(script.textContent) as { state?: Record<string, { state?: Record<string, unknown> }> };
  } catch {
    return null;
  }
  const modelState = payload.state?.[modelId]?.state;
  if (!modelState || !isPlainObject(modelState)) return null;
  for (const [key, value] of Object.entries(viewState)) {
    modelState[key] = value;
  }
  script.textContent = safeInlineJson(payload);
  return doc.documentElement.outerHTML;
}

export function currentWidgetViewState(model: unknown, keys: readonly string[]): Record<string, unknown> {
  const widgetModel = model as AnyWidgetModelWithManager | null;
  const next: Record<string, unknown> = {};
  const attrs = widgetModel?.attributes ?? {};
  for (const key of keys) {
    const value = typeof widgetModel?.get === "function"
      ? widgetModel.get(key)
      : attrs[key];
    if (isSerializableViewValue(value)) next[key] = value;
  }
  return next;
}

export function standaloneHtmlWithCurrentWidgetState(
  model: unknown,
  html: string,
  keys: readonly string[],
): string {
  const widgetModel = model as AnyWidgetModelWithManager | null;
  const stripped = stripCurrentWidgetViewState(html);
  const id = widgetModelId(widgetModel) ?? singleWidgetViewModelId(stripped);
  if (!id) return html;
  const viewState = currentWidgetViewState(widgetModel, keys);
  const updatedEmbeddedState = updateEmbeddedWidgetStateScript(stripped, id, viewState);
  if (updatedEmbeddedState) return updatedEmbeddedState;
  const payload: CurrentWidgetViewStatePayload = {
    version: 1,
    models: { [id]: viewState },
  };
  const script = `<script id="${CURRENT_WIDGET_VIEW_STATE_SCRIPT_ID}" type="application/json">${safeInlineJson(payload)}</script>`;
  if (stripped.includes("</body>")) {
    return stripped.replace("</body>", `${script}\n</body>`);
  }
  return `${stripped}\n${script}`;
}

export function applyStandaloneWidgetViewState(model: unknown): void {
  const widgetModel = model as AnyWidgetModelWithManager | null;
  if (!widgetModel || widgetModel.__quantemCurrentViewStateApplied) return;
  const script = typeof document !== "undefined"
    ? document.getElementById(CURRENT_WIDGET_VIEW_STATE_SCRIPT_ID)
    : null;
  if (!script?.textContent) return;
  let payload: CurrentWidgetViewStatePayload | null = null;
  try {
    payload = JSON.parse(script.textContent) as CurrentWidgetViewStatePayload;
  } catch {
    return;
  }
  const id = widgetModelId(widgetModel);
  const models = payload?.models ?? {};
  const state = id && models[id]
    ? models[id]
    : Object.keys(models).length === 1
      ? models[Object.keys(models)[0]]
      : null;
  if (!state || !isPlainObject(state)) return;
  const clean: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(state)) {
    if (isSerializableViewValue(value)) clean[key] = value;
  }
  if (!Object.keys(clean).length || typeof widgetModel.set !== "function") return;
  widgetModel.__quantemCurrentViewStateApplied = true;
  widgetModel.set(clean);
}
