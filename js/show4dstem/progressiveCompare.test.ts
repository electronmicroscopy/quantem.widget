import { beforeEach, describe, expect, it } from "vitest";
import {
  beginPendingProgressiveComparePage,
  beginProgressiveComparePage,
  completeProgressiveComparePage,
  freshVisibleComparePagePaintAck,
  mergeProgressiveCompareCacheMetadata,
  mergeProgressiveComparePanel,
  progressiveCompareCacheBadge,
  progressiveComparePanelPresentation,
  pendingComparePageIndices,
  recordComparePageClick,
  recordComparePageFirstPanelPaint,
  recordComparePageStaleDrop,
  recordComparePageVisiblePaint,
  reconcileCompletedCompareIndices,
  reconcileProgressiveComparePanels,
  retainCachedProgressiveComparePanels,
  shouldClearProgressiveComparePage,
  show4DSTEMPerf,
} from "./progressiveCompare";

describe("progressive Show4DSTEM compare pages", () => {
  beforeEach(() => {
    delete window.__quantemShow4DSTEMPerf;
  });

  it("reserves ordered slots and fills panels without changing the page layout", () => {
    recordComparePageClick(2, 100);
    const started = beginProgressiveComparePage({
      type: "compare_page_start",
      generation: 7,
      page_idx: 2,
      indices: [8, 9, 10],
    }, 125);
    expect(started?.expectedIndices).toEqual([8, 9, 10]);
    expect(started?.panels.size).toBe(0);

    const values = new Float32Array([1, 2, 3, 4]);
    const merged = mergeProgressiveComparePanel(
      started!,
      { type: "compare_panel", generation: 7, page_idx: 2, frame_idx: 9, slot: 1 },
      [new DataView(values.buffer)],
      values.length,
      140,
    );
    expect(merged?.expectedIndices).toEqual([8, 9, 10]);
    expect(Array.from(merged?.panels.get(9) ?? [])).toEqual([1, 2, 3, 4]);

    const complete = completeProgressiveComparePage(
      merged!,
      { type: "compare_page_complete", generation: 7, page_idx: 2, loaded_count: 1 },
      180,
    );
    expect(complete?.loading).toBe(false);
    expect(complete?.complete).toBe(true);

    expect(show4DSTEMPerf()?.comparePage).toMatchObject({
      generation: "7",
      page: 2,
      pageClicks: 1,
      pageStarts: 1,
      firstPanels: 1,
      pageCompletes: 1,
      clickToStartMs: 25,
      startToFirstPanelMs: 15,
      clickToFirstPanelMs: 40,
      startToCompleteMs: 55,
      clickToCompleteMs: 80,
      loadedCount: 1,
      staleDrops: 0,
    });
  });

  it("separates cached and fresh receipt latency while preserving first-panel metrics", () => {
    recordComparePageClick(0, 10);
    const started = beginProgressiveComparePage({
      generation: 20,
      page_idx: 0,
      expected_indices: [0, 1],
      cached_indices: [0, 1],
    }, 20)!;
    const cachedValues = new Float32Array([1, 2]);
    const freshValues = new Float32Array([3, 4]);
    const firstCached = mergeProgressiveComparePanel(
      started,
      { generation: 20, frame_idx: 0, cached: true },
      [new DataView(cachedValues.buffer)],
      2,
      30,
    )!;
    const secondCached = mergeProgressiveComparePanel(
      firstCached,
      { generation: 20, frame_idx: 1, cached: true },
      [new DataView(cachedValues.buffer)],
      2,
      35,
    )!;
    const firstFresh = mergeProgressiveComparePanel(
      secondCached,
      { generation: 20, frame_idx: 0, cached: false },
      [new DataView(freshValues.buffer)],
      2,
      50,
    )!;

    expect(firstFresh.cachedIndices).toEqual(new Set([1]));
    expect(show4DSTEMPerf()?.comparePage).toMatchObject({
      firstPanels: 1,
      firstPanelAtMs: 30,
      firstCachedPanelReceipts: 1,
      firstCachedPanelReceiptAtMs: 30,
      startToFirstCachedPanelReceiptMs: 10,
      clickToFirstCachedPanelReceiptMs: 20,
      firstFreshPanelReceipts: 1,
      firstFreshPanelReceiptAtMs: 50,
      startToFirstFreshPanelReceiptMs: 30,
      clickToFirstFreshPanelReceiptMs: 40,
    });
  });

  it("records cached-visible and fresh-visible paint only for complete canvas sets", () => {
    recordComparePageClick(1, 100);
    let state = beginProgressiveComparePage({
      generation: "paint-21",
      page_idx: 1,
      expected_indices: [4, 5],
      cached_indices: [4, 5],
    }, 110)!;
    const values = new Float32Array([5, 6]);
    state = mergeProgressiveComparePanel(
      state,
      { generation: "paint-21", frame_idx: 4, cached: true },
      [new DataView(values.buffer)],
      2,
      115,
    )!;
    state = mergeProgressiveComparePanel(
      state,
      { generation: "paint-21", frame_idx: 5, cached: true },
      [new DataView(values.buffer)],
      2,
      120,
    )!;

    expect(recordComparePageFirstPanelPaint(state, [4], 125)).toBe("cached");
    expect(recordComparePageFirstPanelPaint(state, [4], 127)).toBe("cached");
    expect(recordComparePageVisiblePaint(state, [4], 125)).toBeNull();
    expect(recordComparePageVisiblePaint(state, [4, 5], 130)).toBe("cached");
    expect(recordComparePageVisiblePaint(state, [4, 5], 135)).toBe("cached");

    state = mergeProgressiveComparePanel(
      state,
      { generation: "paint-21", frame_idx: 4, cached: false },
      [new DataView(values.buffer)],
      2,
      140,
    )!;
    expect(recordComparePageFirstPanelPaint(state, [4, 5], 145)).toBe("mixed");
    expect(recordComparePageVisiblePaint(state, [4, 5], 145)).toBeNull();
    state = mergeProgressiveComparePanel(
      state,
      { generation: "paint-21", frame_idx: 5, cached: false },
      [new DataView(values.buffer)],
      2,
      150,
    )!;
    expect(recordComparePageVisiblePaint(state, [4, 5], 160)).toBe("fresh");

    expect(show4DSTEMPerf()?.comparePage).toMatchObject({
      firstCachedPanelPaints: 1,
      firstCachedPanelPaintAtMs: 125,
      startToFirstCachedPanelPaintMs: 15,
      clickToFirstCachedPanelPaintMs: 25,
      firstFreshPanelPaints: 1,
      firstFreshPanelPaintAtMs: 145,
      startToFirstFreshPanelPaintMs: 35,
      clickToFirstFreshPanelPaintMs: 45,
      cachedVisiblePaints: 1,
      cachedVisiblePaintAtMs: 130,
      startToCachedVisiblePaintMs: 20,
      clickToCachedVisiblePaintMs: 30,
      freshVisiblePaints: 1,
      freshVisiblePaintAtMs: 160,
      startToFreshVisiblePaintMs: 50,
      clickToFreshVisiblePaintMs: 60,
    });
  });

  it("builds exactly one acknowledgement for each fully fresh visible page", () => {
    const fresh = beginProgressiveComparePage({
      generation: 31,
      page_idx: 2,
      expected_indices: [8, 9],
    }, 10)!;

    expect(freshVisibleComparePagePaintAck(fresh, [8], null)).toBeNull();
    const ack = freshVisibleComparePagePaintAck(fresh, [8, 9], null);
    expect(ack).toEqual({
      key: "31:2",
      message: {
        type: "compare_page_paint_ack",
        version: 1,
        generation: "31",
        page_idx: 2,
        painted_indices: [8, 9],
        paint_kind: "fresh",
      },
    });
    expect(freshVisibleComparePagePaintAck(fresh, [8, 9], ack!.key)).toBeNull();

    const cached = beginProgressiveComparePage({
      generation: 32,
      page_idx: 2,
      expected_indices: [8, 9],
      cached_indices: [8],
    }, 20)!;
    expect(freshVisibleComparePagePaintAck(cached, [8, 9], ack!.key)).toBeNull();
  });

  it("rejects stale generations and malformed panel buffers", () => {
    const started = beginProgressiveComparePage({ generation: "fresh", page: 0, expected_indices: [0] }, 0)!;
    const values = new Float32Array([1, 2]);

    expect(mergeProgressiveComparePanel(
      started,
      { generation: "old", index: 0 },
      [new DataView(values.buffer)],
      2,
      1,
    )).toBeNull();
    expect(mergeProgressiveComparePanel(
      started,
      { generation: "fresh", index: 0 },
      [new DataView(values.buffer)],
      3,
      1,
    )).toBeNull();
    expect(completeProgressiveComparePage(started, { generation: "old" }, 1)).toBeNull();

    recordComparePageStaleDrop();
    recordComparePageStaleDrop();
    expect(show4DSTEMPerf()?.comparePage.staleDrops).toBe(2);
  });

  it("uses the announced position when a panel index is omitted", () => {
    const started = beginProgressiveComparePage({ generation: 3, expected_indices: [12, 14] }, 0)!;
    const values = new Float32Array([5, 6]);
    const merged = mergeProgressiveComparePanel(
      started,
      { generation: 3, position: 1 },
      [new DataView(values.buffer)],
      2,
      1,
    );
    expect(Array.from(merged?.panels.get(14) ?? [])).toEqual([5, 6]);
  });

  it("matches Python page slicing before hidden panels are removed", () => {
    expect(pendingComparePageIndices(
      [0, 1, 2, 3, 4, 5],
      [1],
      1,
      3,
    )).toEqual([3, 4, 5]);
    expect(pendingComparePageIndices(
      [0, 1, 2, 3, 4, 5],
      [4],
      1,
      3,
    )).toEqual([3, 5]);
  });

  it("only reserves loading slots for a progressive folder-backed viewer", () => {
    expect(beginPendingProgressiveComparePage(
      false,
      "pending:1",
      1,
      [0, 1, 2, 3],
      [],
      2,
    )).toBeNull();
    expect(beginPendingProgressiveComparePage(
      true,
      "pending:2",
      1,
      [0, 1, 2, 3],
      [2],
      2,
    )).toMatchObject({
      generation: "pending:2",
      page: 1,
      expectedIndices: [3],
      loading: true,
      complete: false,
    });
  });

  it("clears stale slots after an empty page generation settles", () => {
    expect(shouldClearProgressiveComparePage(false, [], [])).toBe(true);
    expect(shouldClearProgressiveComparePage(true, [], [])).toBe(false);
    expect(shouldClearProgressiveComparePage(false, [3], [])).toBe(false);
    expect(shouldClearProgressiveComparePage(false, [], [3])).toBe(false);
  });

  it("never relabels the prior durable page as a failed requested page", () => {
    expect(reconcileCompletedCompareIndices(
      [3, 4, 5],
      [0, 1, 2],
    )).toEqual([3, 4, 5]);
    expect(reconcileCompletedCompareIndices(
      [3, 4, 5],
      [3, 4],
    )).toEqual([3, 4]);
  });

  it("uses the durable stack directly when progressive delivery is disabled", () => {
    const durable = new Float32Array([9, 10]);
    const reconciled = reconcileProgressiveComparePanels(null, [4], [durable]);
    expect(reconciled.get(4)).toBe(durable);
  });

  it("fills missing completed panels from the durable synced stack", () => {
    const streamed = new Float32Array([1, 2]);
    const durableA = new Float32Array([3, 4]);
    const durableB = new Float32Array([5, 6]);
    const state = {
      generation: "8",
      page: 1,
      expectedIndices: [3, 4, 5],
      panels: new Map([[3, streamed]]),
      cachedIndices: new Set<number>(),
      cacheState: "off" as const,
      loading: false,
      complete: true,
    };
    const reconciled = reconcileProgressiveComparePanels(
      state,
      [3, 4, 5],
      [durableA, durableB, new Float32Array([7, 8])],
    );
    expect(reconciled.get(3)).toBe(streamed);
    expect(Array.from(reconciled.get(4) ?? [])).toEqual([5, 6]);
    expect(Array.from(reconciled.get(5) ?? [])).toEqual([7, 8]);
  });

  it("keeps cached durable panels visible while raw data refreshes", () => {
    const cached = new Float32Array([11, 12]);
    const unrelated = new Float32Array([21, 22]);
    const state = beginProgressiveComparePage({
      generation: 10,
      page_idx: 0,
      expected_indices: [0, 1],
      cached_indices: [0],
      cache_state: "partial",
    }, 0)!;

    const reconciled = reconcileProgressiveComparePanels(
      state,
      [0, 1],
      [cached, unrelated],
    );
    expect(reconciled.get(0)).toBe(cached);
    expect(reconciled.has(1)).toBe(false);
    expect(progressiveCompareCacheBadge(state, reconciled)).toEqual({
      label: "Cached preview · refreshing 0/2",
      tone: "cached",
    });
  });

  it("marks a cached loaded tile busy without disabling its interactions", () => {
    const state = beginProgressiveComparePage({
      generation: 11,
      expected_indices: [4],
      cached_indices: [4],
      cache_state: "cached",
    }, 0)!;

    expect(progressiveComparePanelPresentation(state, 4, true)).toEqual({
      cached: true,
      busy: true,
      disabled: false,
      labelSuffix: ", cached preview, refreshing",
    });
  });

  it("applies trait fallback cache metadata only to the requested page slots", () => {
    const state = beginProgressiveComparePage({
      generation: 111,
      expected_indices: [4, 5],
    }, 0)!;
    const withTraitMetadata = mergeProgressiveCompareCacheMetadata(
      state,
      [5, 99, 5],
      "partial",
    );

    expect([...withTraitMetadata.cachedIndices]).toEqual([5]);
    expect(withTraitMetadata.cacheState).toBe("partial");
  });

  it("replaces cached panel state with fresh state as refresh messages arrive", () => {
    const cachedValues = new Float32Array([1, 2]);
    const freshValues = new Float32Array([3, 4]);
    const started = beginProgressiveComparePage({
      generation: 12,
      expected_indices: [7],
      cached_indices: [7],
      cache_state: "cached",
    }, 0)!;
    const cached = mergeProgressiveComparePanel(
      started,
      { generation: 12, frame_idx: 7, cached: true },
      [new DataView(cachedValues.buffer)],
      2,
      1,
    )!;
    const fresh = mergeProgressiveComparePanel(
      cached,
      { generation: 12, frame_idx: 7, cached: false },
      [new DataView(freshValues.buffer)],
      2,
      2,
    )!;
    const complete = completeProgressiveComparePage(
      fresh,
      {
        generation: 12,
        indices: [7],
        cached_indices: [],
        cache_state: "fresh",
      },
      3,
    )!;

    expect(Array.from(complete.panels.get(7) ?? [])).toEqual([3, 4]);
    expect(complete.cachedIndices.size).toBe(0);
    expect(progressiveComparePanelPresentation(complete, 7, true)).toMatchObject({
      cached: false,
      busy: false,
      disabled: false,
    });
    expect(progressiveCompareCacheBadge(complete, complete.panels)).toEqual({
      label: "Fresh",
      tone: "fresh",
    });
  });

  it("retains an already displayed cache panel across a new refresh generation", () => {
    const values = new Float32Array([31, 32]);
    const previous = mergeProgressiveComparePanel(
      beginProgressiveComparePage({
        generation: 13,
        page_idx: 2,
        expected_indices: [8],
        cached_indices: [8],
      }, 0)!,
      { generation: 13, frame_idx: 8, cached: true },
      [new DataView(values.buffer)],
      2,
      1,
    )!;
    const restarted = beginProgressiveComparePage({
      generation: 14,
      page_idx: 2,
      expected_indices: [8],
      cached_indices: [8],
      cache_state: "cached",
    }, 2)!;

    const retained = retainCachedProgressiveComparePanels(restarted, previous);
    expect(retained.panels.get(8)).toBe(previous.panels.get(8));
    expect(retained.loading).toBe(true);
  });

  it("keeps cached previews visible when refresh completion reports a warning", () => {
    const state = beginProgressiveComparePage({
      generation: 15,
      expected_indices: [2],
      cached_indices: [2],
      cache_state: "cached",
    }, 0)!;
    state.panels.set(2, new Float32Array([5, 6]));
    const complete = completeProgressiveComparePage(
      state,
      {
        generation: 15,
        indices: [2],
        cached_indices: [2],
        cache_state: "warning",
      },
      3,
    )!;

    expect(progressiveCompareCacheBadge(complete, complete.panels)).toEqual({
      label: "Cached preview · refresh failed; retry page",
      tone: "warning",
    });
  });
});
