import { extractFloat32 } from "../format";

export type ComparePageGeneration = string;

export type ComparePageCacheState =
  | "off"
  | "miss"
  | "partial"
  | "cached"
  | "fresh"
  | "warning";

export type ProgressiveComparePage = {
  generation: ComparePageGeneration;
  page: number;
  expectedIndices: number[];
  panels: Map<number, Float32Array>;
  cachedIndices: Set<number>;
  cacheState: ComparePageCacheState;
  loading: boolean;
  complete: boolean;
};

export type ComparePageMessage = {
  type?: string;
  generation?: number | string;
  page?: number;
  page_idx?: number;
  index?: number;
  frame_idx?: number;
  position?: number;
  slot?: number;
  expected_indices?: number[];
  indices?: number[];
  loaded_count?: number;
  first_panel_ms?: number;
  total_ms?: number;
  cached?: boolean;
  cached_indices?: number[];
  cache_state?: string;
};

export type ProgressiveCompareCacheBadge = {
  label: string;
  tone: "loading" | "cached" | "fresh" | "warning";
};

export type ProgressiveComparePanelPresentation = {
  cached: boolean;
  busy: boolean;
  disabled: boolean;
  labelSuffix: string;
};

export type Show4DSTEMComparePagePerf = {
  pageClicks: number;
  pageStarts: number;
  firstPanels: number;
  firstCachedPanelReceipts: number;
  firstFreshPanelReceipts: number;
  firstCachedPanelPaints: number;
  firstFreshPanelPaints: number;
  cachedVisiblePaints: number;
  freshVisiblePaints: number;
  pageCompletes: number;
  generation: ComparePageGeneration | null;
  page: number | null;
  clickAtMs: number | null;
  startAtMs: number | null;
  clickToStartMs: number | null;
  firstPanelAtMs: number | null;
  startToFirstPanelMs: number | null;
  clickToFirstPanelMs: number | null;
  firstCachedPanelReceiptAtMs: number | null;
  startToFirstCachedPanelReceiptMs: number | null;
  clickToFirstCachedPanelReceiptMs: number | null;
  firstFreshPanelReceiptAtMs: number | null;
  startToFirstFreshPanelReceiptMs: number | null;
  clickToFirstFreshPanelReceiptMs: number | null;
  firstCachedPanelPaintAtMs: number | null;
  startToFirstCachedPanelPaintMs: number | null;
  clickToFirstCachedPanelPaintMs: number | null;
  firstFreshPanelPaintAtMs: number | null;
  startToFirstFreshPanelPaintMs: number | null;
  clickToFirstFreshPanelPaintMs: number | null;
  cachedVisiblePaintAtMs: number | null;
  startToCachedVisiblePaintMs: number | null;
  clickToCachedVisiblePaintMs: number | null;
  freshVisiblePaintAtMs: number | null;
  startToFreshVisiblePaintMs: number | null;
  clickToFreshVisiblePaintMs: number | null;
  completeAtMs: number | null;
  startToCompleteMs: number | null;
  clickToCompleteMs: number | null;
  loadedCount: number;
  staleDrops: number;
};

export type Show4DSTEMPerf = {
  comparePage: Show4DSTEMComparePagePerf;
};

export function pendingComparePageIndices(
  order: number[],
  hiddenValues: number[],
  page: number,
  pageSize: number,
): number[] {
  const size = Math.max(1, Math.round(Number(pageSize) || 1));
  const current = Math.max(0, Math.round(Number(page) || 0));
  const hidden = new Set(hiddenValues.filter((idx) => Number.isInteger(idx) && idx >= 0));
  return order
    .slice(current * size, (current + 1) * size)
    .filter((idx) => !hidden.has(idx));
}

export function beginPendingProgressiveComparePage(
  enabled: boolean,
  generation: ComparePageGeneration,
  page: number,
  order: number[],
  hiddenValues: number[],
  pageSize: number,
): ProgressiveComparePage | null {
  if (!enabled) return null;
  return {
    generation,
    page: Math.max(0, Math.round(Number(page) || 0)),
    expectedIndices: pendingComparePageIndices(order, hiddenValues, page, pageSize),
    panels: new Map(),
    cachedIndices: new Set(),
    cacheState: "off",
    loading: true,
    complete: false,
  };
}

export function shouldClearProgressiveComparePage(
  loading: boolean,
  expectedIndices: number[],
  durableIndices: number[],
): boolean {
  return !loading && expectedIndices.length === 0 && durableIndices.length === 0;
}

export function reconcileCompletedCompareIndices(
  expectedIndices: number[],
  durableIndices: number[],
): number[] {
  if (durableIndices.length === 0) return [...expectedIndices];
  const expected = new Set(expectedIndices);
  if (!durableIndices.every((frame) => expected.has(frame))) {
    return [...expectedIndices];
  }
  return [...durableIndices];
}

export function reconcileProgressiveComparePanels(
  state: ProgressiveComparePage | null,
  durableIndices: number[],
  durablePanels: Float32Array[],
): Map<number, Float32Array> {
  const durable = new Map<number, Float32Array>();
  durableIndices.forEach((frame, index) => {
    const panel = durablePanels[index];
    if (panel) durable.set(frame, panel);
  });
  if (!state) return durable;

  const panels = new Map(state.panels);
  state.expectedIndices.forEach((frame) => {
    if (
      !panels.has(frame)
      && durable.has(frame)
      && (state.complete || state.cachedIndices.has(frame))
    ) {
      panels.set(frame, durable.get(frame)!);
    }
  });
  return panels;
}

export function mergeProgressiveCompareCacheMetadata(
  state: ProgressiveComparePage,
  cachedValues: unknown,
  cacheStateValue: unknown,
): ProgressiveComparePage {
  const expected = new Set(state.expectedIndices);
  const cachedIndices = cachedValues === undefined
    ? new Set(state.cachedIndices)
    : new Set(uniqueIndices(cachedValues).filter((frame) => expected.has(frame)));
  const cacheState = normaliseCacheState(cacheStateValue) ?? state.cacheState;
  return { ...state, cachedIndices, cacheState };
}

export function retainCachedProgressiveComparePanels(
  state: ProgressiveComparePage,
  previous: ProgressiveComparePage | null,
): ProgressiveComparePage {
  if (!previous || previous.page !== state.page || state.cachedIndices.size === 0) return state;
  const panels = new Map(state.panels);
  state.cachedIndices.forEach((frame) => {
    const panel = previous.panels.get(frame);
    if (panel && !panels.has(frame)) panels.set(frame, panel);
  });
  return panels.size === state.panels.size ? state : { ...state, panels };
}

export function progressiveComparePanelPresentation(
  state: ProgressiveComparePage | null,
  frame: number,
  loaded: boolean,
): ProgressiveComparePanelPresentation {
  const cached = Boolean(loaded && state?.cachedIndices.has(frame));
  const loading = Boolean(state?.loading);
  return {
    cached,
    busy: loading && (!loaded || cached),
    disabled: !loaded,
    labelSuffix: !loaded
      ? loading ? ", loading" : ", unavailable"
      : cached && loading ? ", cached preview, refreshing" : cached ? ", cached preview" : "",
  };
}

export function progressiveCompareCacheBadge(
  state: ProgressiveComparePage | null,
  visiblePanels: Map<number, Float32Array>,
): ProgressiveCompareCacheBadge | null {
  if (!state) return null;
  const expectedCount = Math.max(1, state.expectedIndices.length);
  const visibleIndices = state.expectedIndices.filter((frame) => visiblePanels.has(frame));
  if (visibleIndices.length === 0) return null;
  const cachedCount = visibleIndices.filter((frame) => state.cachedIndices.has(frame)).length;
  const freshCount = visibleIndices.length - cachedCount;

  if (state.cacheState === "warning" && cachedCount > 0) {
    return {
      label: "Cached preview · refresh failed; retry page",
      tone: "warning",
    };
  }
  if (state.loading && cachedCount > 0) {
    return {
      label: `Cached preview · refreshing ${freshCount}/${expectedCount}`,
      tone: "cached",
    };
  }
  if (state.loading) {
    return { label: `Loading ${visibleIndices.length}/${expectedCount}`, tone: "loading" };
  }
  if (cachedCount > 0) {
    return {
      label: freshCount > 0
        ? `Cached preview · ${freshCount}/${expectedCount} refreshed`
        : "Cached preview",
      tone: state.cacheState === "warning" ? "warning" : "cached",
    };
  }
  if (state.cacheState === "fresh") {
    return { label: "Fresh", tone: "fresh" };
  }
  return null;
}

declare global {
  interface Window {
    __quantemShow4DSTEMPerf?: Show4DSTEMPerf;
  }
}

const emptyComparePagePerf = (): Show4DSTEMComparePagePerf => ({
  pageClicks: 0,
  pageStarts: 0,
  firstPanels: 0,
  firstCachedPanelReceipts: 0,
  firstFreshPanelReceipts: 0,
  firstCachedPanelPaints: 0,
  firstFreshPanelPaints: 0,
  cachedVisiblePaints: 0,
  freshVisiblePaints: 0,
  pageCompletes: 0,
  generation: null,
  page: null,
  clickAtMs: null,
  startAtMs: null,
  clickToStartMs: null,
  firstPanelAtMs: null,
  startToFirstPanelMs: null,
  clickToFirstPanelMs: null,
  firstCachedPanelReceiptAtMs: null,
  startToFirstCachedPanelReceiptMs: null,
  clickToFirstCachedPanelReceiptMs: null,
  firstFreshPanelReceiptAtMs: null,
  startToFirstFreshPanelReceiptMs: null,
  clickToFirstFreshPanelReceiptMs: null,
  firstCachedPanelPaintAtMs: null,
  startToFirstCachedPanelPaintMs: null,
  clickToFirstCachedPanelPaintMs: null,
  firstFreshPanelPaintAtMs: null,
  startToFirstFreshPanelPaintMs: null,
  clickToFirstFreshPanelPaintMs: null,
  cachedVisiblePaintAtMs: null,
  startToCachedVisiblePaintMs: null,
  clickToCachedVisiblePaintMs: null,
  freshVisiblePaintAtMs: null,
  startToFreshVisiblePaintMs: null,
  clickToFreshVisiblePaintMs: null,
  completeAtMs: null,
  startToCompleteMs: null,
  clickToCompleteMs: null,
  loadedCount: 0,
  staleDrops: 0,
});

let pendingPageClick: { page: number; atMs: number } | null = null;

export function show4DSTEMPerf(): Show4DSTEMPerf | null {
  if (typeof window === "undefined") return null;
  if (!window.__quantemShow4DSTEMPerf) {
    window.__quantemShow4DSTEMPerf = { comparePage: emptyComparePagePerf() };
  }
  return window.__quantemShow4DSTEMPerf;
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function finitePage(value: unknown): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.max(0, Math.round(numeric)) : 0;
}

function normaliseGeneration(value: unknown): ComparePageGeneration | null {
  if (typeof value !== "number" && typeof value !== "string") return null;
  const generation = String(value);
  return generation.length > 0 ? generation : null;
}

function normaliseCacheState(value: unknown): ComparePageCacheState | null {
  return value === "off"
    || value === "miss"
    || value === "partial"
    || value === "cached"
    || value === "fresh"
    || value === "warning"
    ? value
    : null;
}

function uniqueIndices(values: unknown): number[] {
  if (!Array.isArray(values)) return [];
  const seen = new Set<number>();
  const indices: number[] = [];
  values.forEach((value) => {
    const index = Number(value);
    if (!Number.isInteger(index) || index < 0 || seen.has(index)) return;
    seen.add(index);
    indices.push(index);
  });
  return indices;
}

export function compareMessageGeneration(message: ComparePageMessage): ComparePageGeneration | null {
  return normaliseGeneration(message.generation);
}

export function recordComparePageClick(page: number, atMs = nowMs()): void {
  pendingPageClick = { page: finitePage(page), atMs };
  const perf = show4DSTEMPerf()?.comparePage;
  if (!perf) return;
  perf.pageClicks += 1;
  perf.page = finitePage(page);
  perf.clickAtMs = atMs;
  perf.startAtMs = null;
  perf.clickToStartMs = null;
  perf.firstPanelAtMs = null;
  perf.startToFirstPanelMs = null;
  perf.clickToFirstPanelMs = null;
  perf.firstCachedPanelReceiptAtMs = null;
  perf.startToFirstCachedPanelReceiptMs = null;
  perf.clickToFirstCachedPanelReceiptMs = null;
  perf.firstFreshPanelReceiptAtMs = null;
  perf.startToFirstFreshPanelReceiptMs = null;
  perf.clickToFirstFreshPanelReceiptMs = null;
  perf.firstCachedPanelPaintAtMs = null;
  perf.startToFirstCachedPanelPaintMs = null;
  perf.clickToFirstCachedPanelPaintMs = null;
  perf.firstFreshPanelPaintAtMs = null;
  perf.startToFirstFreshPanelPaintMs = null;
  perf.clickToFirstFreshPanelPaintMs = null;
  perf.cachedVisiblePaintAtMs = null;
  perf.startToCachedVisiblePaintMs = null;
  perf.clickToCachedVisiblePaintMs = null;
  perf.freshVisiblePaintAtMs = null;
  perf.startToFreshVisiblePaintMs = null;
  perf.clickToFreshVisiblePaintMs = null;
  perf.completeAtMs = null;
  perf.startToCompleteMs = null;
  perf.clickToCompleteMs = null;
  perf.loadedCount = 0;
}

export function beginProgressiveComparePage(
  message: ComparePageMessage,
  atMs = nowMs(),
): ProgressiveComparePage | null {
  const generation = compareMessageGeneration(message);
  if (generation === null) return null;
  const expectedIndices = uniqueIndices(message.expected_indices ?? message.indices);
  const page = finitePage(message.page ?? message.page_idx);
  const expected = new Set(expectedIndices);
  const cachedIndices = new Set(
    uniqueIndices(message.cached_indices).filter((frame) => expected.has(frame)),
  );
  const pageClick = pendingPageClick?.page === page ? pendingPageClick : null;
  if (pageClick) pendingPageClick = null;
  const perf = show4DSTEMPerf()?.comparePage;
  if (perf) {
    perf.pageStarts += 1;
    perf.generation = generation;
    perf.page = page;
    perf.clickAtMs = pageClick?.atMs ?? null;
    perf.startAtMs = atMs;
    perf.clickToStartMs = perf.clickAtMs === null ? null : Math.max(0, atMs - perf.clickAtMs);
    perf.firstPanelAtMs = null;
    perf.startToFirstPanelMs = null;
    perf.clickToFirstPanelMs = null;
    perf.firstCachedPanelReceiptAtMs = null;
    perf.startToFirstCachedPanelReceiptMs = null;
    perf.clickToFirstCachedPanelReceiptMs = null;
    perf.firstFreshPanelReceiptAtMs = null;
    perf.startToFirstFreshPanelReceiptMs = null;
    perf.clickToFirstFreshPanelReceiptMs = null;
    perf.firstCachedPanelPaintAtMs = null;
    perf.startToFirstCachedPanelPaintMs = null;
    perf.clickToFirstCachedPanelPaintMs = null;
    perf.firstFreshPanelPaintAtMs = null;
    perf.startToFirstFreshPanelPaintMs = null;
    perf.clickToFirstFreshPanelPaintMs = null;
    perf.cachedVisiblePaintAtMs = null;
    perf.startToCachedVisiblePaintMs = null;
    perf.clickToCachedVisiblePaintMs = null;
    perf.freshVisiblePaintAtMs = null;
    perf.startToFreshVisiblePaintMs = null;
    perf.clickToFreshVisiblePaintMs = null;
    perf.completeAtMs = null;
    perf.startToCompleteMs = null;
    perf.clickToCompleteMs = null;
    perf.loadedCount = 0;
  }
  return {
    generation,
    page,
    expectedIndices,
    panels: new Map(),
    cachedIndices,
    cacheState: normaliseCacheState(message.cache_state) ?? "off",
    loading: true,
    complete: false,
  };
}

export function mergeProgressiveComparePanel(
  state: ProgressiveComparePage,
  message: ComparePageMessage,
  buffers: Array<DataView | ArrayBuffer | Uint8Array> | undefined,
  expectedFloats: number,
  atMs = nowMs(),
): ProgressiveComparePage | null {
  if (compareMessageGeneration(message) !== state.generation) return null;
  const position = Number(message.position ?? message.slot);
  const explicitIndex = Number(message.index ?? message.frame_idx);
  const frame = Number.isInteger(explicitIndex) && explicitIndex >= 0
    ? explicitIndex
    : Number.isInteger(position) && position >= 0
      ? state.expectedIndices[position]
      : undefined;
  if (frame === undefined || !state.expectedIndices.includes(frame)) return null;
  const panel = buffers?.[0] ? extractFloat32(buffers[0], expectedFloats) : null;
  if (!panel || panel.length !== expectedFloats) return null;

  const panels = new Map(state.panels);
  panels.set(frame, panel.slice());
  const cachedIndices = new Set(state.cachedIndices);
  if (message.cached === true) cachedIndices.add(frame);
  if (message.cached === false) cachedIndices.delete(frame);
  const cachedReceipt = message.cached === true
    || (message.cached !== false && state.cachedIndices.has(frame));
  const perf = show4DSTEMPerf()?.comparePage;
  if (perf) {
    if (perf.firstPanelAtMs === null) {
      perf.firstPanels += 1;
      perf.firstPanelAtMs = atMs;
      perf.startToFirstPanelMs = perf.startAtMs === null ? null : Math.max(0, atMs - perf.startAtMs);
      perf.clickToFirstPanelMs = perf.clickAtMs === null ? null : Math.max(0, atMs - perf.clickAtMs);
    }
    if (cachedReceipt && perf.firstCachedPanelReceiptAtMs === null) {
      perf.firstCachedPanelReceipts += 1;
      perf.firstCachedPanelReceiptAtMs = atMs;
      perf.startToFirstCachedPanelReceiptMs = perf.startAtMs === null
        ? null
        : Math.max(0, atMs - perf.startAtMs);
      perf.clickToFirstCachedPanelReceiptMs = perf.clickAtMs === null
        ? null
        : Math.max(0, atMs - perf.clickAtMs);
    }
    if (!cachedReceipt && perf.firstFreshPanelReceiptAtMs === null) {
      perf.firstFreshPanelReceipts += 1;
      perf.firstFreshPanelReceiptAtMs = atMs;
      perf.startToFirstFreshPanelReceiptMs = perf.startAtMs === null
        ? null
        : Math.max(0, atMs - perf.startAtMs);
      perf.clickToFirstFreshPanelReceiptMs = perf.clickAtMs === null
        ? null
        : Math.max(0, atMs - perf.clickAtMs);
    }
    perf.loadedCount = panels.size;
  }
  return { ...state, panels, cachedIndices };
}

/** Record the first cached and fresh panel after-paint milestones. */
export function recordComparePageFirstPanelPaint(
  state: ProgressiveComparePage,
  paintedIndices: Iterable<number>,
  atMs = nowMs(),
): "cached" | "fresh" | "mixed" | null {
  const perf = show4DSTEMPerf()?.comparePage;
  if (!perf || perf.generation !== state.generation || perf.page !== state.page) return null;
  const expected = new Set(state.expectedIndices);
  const painted = Array.from(paintedIndices).filter((frame) => expected.has(frame));
  if (painted.length === 0) return null;

  const hasCached = painted.some((frame) => state.cachedIndices.has(frame));
  const hasFresh = painted.some((frame) => !state.cachedIndices.has(frame));
  if (hasCached && perf.firstCachedPanelPaintAtMs === null) {
    perf.firstCachedPanelPaints += 1;
    perf.firstCachedPanelPaintAtMs = atMs;
    perf.startToFirstCachedPanelPaintMs = perf.startAtMs === null
      ? null
      : Math.max(0, atMs - perf.startAtMs);
    perf.clickToFirstCachedPanelPaintMs = perf.clickAtMs === null
      ? null
      : Math.max(0, atMs - perf.clickAtMs);
    markComparePagePaint("cached-first-panel");
  }
  if (hasFresh && perf.firstFreshPanelPaintAtMs === null) {
    perf.firstFreshPanelPaints += 1;
    perf.firstFreshPanelPaintAtMs = atMs;
    perf.startToFirstFreshPanelPaintMs = perf.startAtMs === null
      ? null
      : Math.max(0, atMs - perf.startAtMs);
    perf.clickToFirstFreshPanelPaintMs = perf.clickAtMs === null
      ? null
      : Math.max(0, atMs - perf.clickAtMs);
    markComparePagePaint("fresh-first-panel");
  }
  return hasCached && hasFresh ? "mixed" : hasCached ? "cached" : "fresh";
}

/**
 * Record the first full-page cached or fresh canvas paint milestone.
 *
 * Callers supply the canvases that still contain the page's current panel
 * arrays. The UI invokes this from the second requestAnimationFrame callback,
 * which is an after-paint proxy rather than an exact compositor timestamp.
 */
export function recordComparePageVisiblePaint(
  state: ProgressiveComparePage,
  paintedIndices: Iterable<number>,
  atMs = nowMs(),
): "cached" | "fresh" | null {
  const perf = show4DSTEMPerf()?.comparePage;
  if (!perf || perf.generation !== state.generation || perf.page !== state.page) return null;
  if (state.expectedIndices.length === 0) return null;
  const painted = new Set(paintedIndices);
  if (!state.expectedIndices.every((frame) => painted.has(frame))) return null;

  const kind = state.expectedIndices.every((frame) => state.cachedIndices.has(frame))
    ? "cached"
    : state.expectedIndices.every((frame) => !state.cachedIndices.has(frame))
      ? "fresh"
      : null;
  if (kind === null) return null;

  if (kind === "cached" && perf.cachedVisiblePaintAtMs === null) {
    perf.cachedVisiblePaints += 1;
    perf.cachedVisiblePaintAtMs = atMs;
    perf.startToCachedVisiblePaintMs = perf.startAtMs === null
      ? null
      : Math.max(0, atMs - perf.startAtMs);
    perf.clickToCachedVisiblePaintMs = perf.clickAtMs === null
      ? null
      : Math.max(0, atMs - perf.clickAtMs);
    markComparePagePaint(`${kind}-visible`);
  }
  if (kind === "fresh" && perf.freshVisiblePaintAtMs === null) {
    perf.freshVisiblePaints += 1;
    perf.freshVisiblePaintAtMs = atMs;
    perf.startToFreshVisiblePaintMs = perf.startAtMs === null
      ? null
      : Math.max(0, atMs - perf.startAtMs);
    perf.clickToFreshVisiblePaintMs = perf.clickAtMs === null
      ? null
      : Math.max(0, atMs - perf.clickAtMs);
    markComparePagePaint(`${kind}-visible`);
  }
  return kind;
}

function markComparePagePaint(kind: string): void {
  if (typeof performance === "undefined" || typeof performance.mark !== "function") return;
  performance.mark(`quantem:show4dstem:compare:${kind}-paint`);
}

export function completeProgressiveComparePage(
  state: ProgressiveComparePage,
  message: ComparePageMessage,
  atMs = nowMs(),
): ProgressiveComparePage | null {
  if (compareMessageGeneration(message) !== state.generation) return null;
  const finalIndices = uniqueIndices(message.indices);
  const expectedIndices = finalIndices.length > 0 ? finalIndices : state.expectedIndices;
  const completed = mergeProgressiveCompareCacheMetadata(
    { ...state, expectedIndices },
    message.cached_indices,
    message.cache_state,
  );
  const perf = show4DSTEMPerf()?.comparePage;
  if (perf) {
    perf.pageCompletes += 1;
    perf.completeAtMs = atMs;
    perf.startToCompleteMs = perf.startAtMs === null ? null : Math.max(0, atMs - perf.startAtMs);
    perf.clickToCompleteMs = perf.clickAtMs === null ? null : Math.max(0, atMs - perf.clickAtMs);
    perf.loadedCount = Number.isFinite(Number(message.loaded_count))
      ? Math.max(0, Math.round(Number(message.loaded_count)))
      : state.panels.size;
  }
  return { ...completed, loading: false, complete: true };
}

export function recordComparePageStaleDrop(): void {
  const perf = show4DSTEMPerf()?.comparePage;
  if (perf) perf.staleDrops += 1;
}
