import { describe, expect, it, vi } from "vitest";

import {
  cancelContrastPreview,
  contrastPreviewIndices,
  enqueueContrastPreview,
  linkedAutoContrastPercentRange,
  seedAutoContrastMirror,
  transitionContrastLinkState,
  type ContrastMirror,
} from "./contrastPreview";

function mirror(): ContrastMirror {
  return {
    linked: { vminPct: 5, vmaxPct: 95 },
    perImage: new Map([
      [0, { vminPct: 10, vmaxPct: 80 }],
      [1, { vminPct: 20, vmaxPct: 70 }],
    ]),
  };
}

describe("Show2D contrast drag preview", () => {
  it("repaints only the active grayscale panel when contrast is unlinked", () => {
    expect(contrastPreviewIndices(false, [0, 1, 2], 1, 3)).toEqual([1]);
  });

  it("repaints every visible grayscale panel when contrast is linked", () => {
    expect(contrastPreviewIndices(true, [0, 1, 2], 0, 3, [false, true, false])).toEqual([0, 2]);
  });

  it("falls back to all panels only when no visibility list exists", () => {
    expect(contrastPreviewIndices(true, [], 1, 4)).toEqual([0, 1, 2, 3]);
  });

  it("never sends an RGB panel through the grayscale preview path", () => {
    expect(contrastPreviewIndices(false, [0, 1, 2], 1, 3, [false, true, false])).toEqual([]);
    expect(contrastPreviewIndices(true, [], 0, 3, [false, true, false])).toEqual([0, 2]);
  });

  it("keeps one pending paint and renders the latest pointer range", async () => {
    const callbacks: FrameRequestCallback[] = [];
    const queue = { frame: 0, inFlight: false, pendingPanel: null as number | null, epoch: 0 };
    const rendered: number[] = [];
    const requestFrame = (callback: FrameRequestCallback) => {
      callbacks.push(callback);
      return callbacks.length;
    };

    const render = (value: number) => { rendered.push(value); };
    enqueueContrastPreview(queue, 10, requestFrame, render);
    enqueueContrastPreview(queue, 20, requestFrame, render);
    enqueueContrastPreview(queue, 30, requestFrame, render);

    expect(callbacks).toHaveLength(1);
    callbacks[0](0);
    await Promise.resolve();
    await Promise.resolve();
    expect(rendered).toEqual([30]);
    expect(queue).toEqual({ frame: 0, inFlight: false, pendingPanel: null, epoch: 0 });
  });

  it("cancels a pending paint during teardown", () => {
    const queue = { frame: 7, inFlight: false, pendingPanel: 2 as number | null, epoch: 0 };
    const cancelled: number[] = [];
    cancelContrastPreview(queue, (frame) => cancelled.push(frame));
    expect(cancelled).toEqual([7]);
    expect(queue).toEqual({ frame: 0, inFlight: false, pendingPanel: null, epoch: 1 });
  });

  it("coalesces pointer changes that arrive during an asynchronous GPU paint", async () => {
    const callbacks: FrameRequestCallback[] = [];
    const queue = { frame: 0, inFlight: false, pendingPanel: null as number | null, epoch: 0 };
    const rendered: number[] = [];
    const resolvers: Array<() => void> = [];
    const requestFrame = (callback: FrameRequestCallback) => {
      callbacks.push(callback);
      return callbacks.length;
    };
    const render = (value: number) => new Promise<void>((resolve) => {
      rendered.push(value);
      resolvers.push(resolve);
    });

    enqueueContrastPreview(queue, 10, requestFrame, render);
    callbacks.shift()!(0);
    enqueueContrastPreview(queue, 20, requestFrame, render);
    enqueueContrastPreview(queue, 30, requestFrame, render);
    expect(callbacks).toHaveLength(0);
    expect(rendered).toEqual([10]);

    resolvers.shift()!();
    await Promise.resolve();
    await Promise.resolve();
    expect(callbacks).toHaveLength(1);
    callbacks.shift()!(16);
    expect(rendered).toEqual([10, 30]);
    resolvers.shift()!();
    await Promise.resolve();
  });

  it("keeps painting sustained pointer input without waiting for release", async () => {
    const callbacks: FrameRequestCallback[] = [];
    const queue = { frame: 0, inFlight: false, pendingPanel: null as number | null, epoch: 0 };
    const rendered: number[] = [];
    const requestFrame = (callback: FrameRequestCallback) => {
      callbacks.push(callback);
      return callbacks.length;
    };

    for (const value of [92, 84, 76, 68]) {
      enqueueContrastPreview(queue, value, requestFrame, (panel) => {
        rendered.push(panel);
      });
      expect(callbacks).toHaveLength(1);
      callbacks.shift()!(value);
      await Promise.resolve();
      await Promise.resolve();
    }

    expect(rendered).toEqual([92, 84, 76, 68]);
    expect(queue).toEqual({ frame: 0, inFlight: false, pendingPanel: null, epoch: 0 });
  });

  it("drops queued work after teardown even when a GPU paint is still resolving", async () => {
    const callbacks: FrameRequestCallback[] = [];
    const queue = { frame: 0, inFlight: false, pendingPanel: null as number | null, epoch: 0 };
    const rendered: number[] = [];
    let finishPaint!: () => void;
    const requestFrame = (callback: FrameRequestCallback) => {
      callbacks.push(callback);
      return callbacks.length;
    };

    enqueueContrastPreview(queue, 10, requestFrame, (panel) => new Promise<void>((resolve) => {
      rendered.push(panel);
      finishPaint = resolve;
    }));
    callbacks.shift()!(0);
    enqueueContrastPreview(queue, 20, requestFrame, (panel) => {
      rendered.push(panel);
    });
    cancelContrastPreview(queue, () => undefined);
    finishPaint();
    await Promise.resolve();
    await Promise.resolve();

    expect(rendered).toEqual([10]);
    expect(callbacks).toHaveLength(0);
    expect(queue).toEqual({ frame: 0, inFlight: false, pendingPanel: null, epoch: 1 });
  });

  it("cannot let a cancelled GPU completion corrupt a replacement preview", async () => {
    const callbacks: FrameRequestCallback[] = [];
    const queue = { frame: 0, inFlight: false, pendingPanel: null as number | null, epoch: 0 };
    const rendered: number[] = [];
    const resolvers = new Map<number, () => void>();
    const requestFrame = (callback: FrameRequestCallback) => {
      callbacks.push(callback);
      return callbacks.length;
    };
    const render = (panel: number) => new Promise<void>((resolve) => {
      rendered.push(panel);
      resolvers.set(panel, resolve);
    });

    enqueueContrastPreview(queue, 10, requestFrame, render);
    callbacks.shift()!(0);
    cancelContrastPreview(queue, () => undefined);
    enqueueContrastPreview(queue, 20, requestFrame, render);
    callbacks.shift()!(16);
    expect(queue.inFlight).toBe(true);

    resolvers.get(10)!();
    await Promise.resolve();
    await Promise.resolve();
    expect(queue.inFlight).toBe(true);

    enqueueContrastPreview(queue, 30, requestFrame, render);
    expect(callbacks).toHaveLength(0);
    resolvers.get(20)!();
    await Promise.resolve();
    await Promise.resolve();
    expect(callbacks).toHaveLength(1);
    callbacks.shift()!(32);
    expect(rendered).toEqual([10, 20, 30]);
    resolvers.get(30)!();
    await Promise.resolve();
    await Promise.resolve();
    expect(queue).toEqual({ frame: 0, inFlight: false, pendingPanel: null, epoch: 1 });
  });

  it("reports a failed paint and continues with the latest pointer position", async () => {
    const callbacks: FrameRequestCallback[] = [];
    const queue = { frame: 0, inFlight: false, pendingPanel: null as number | null, epoch: 0 };
    const rendered: number[] = [];
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const requestFrame = (callback: FrameRequestCallback) => {
      callbacks.push(callback);
      return callbacks.length;
    };
    const render = (panel: number) => {
      rendered.push(panel);
      if (panel === 10) throw new Error("GPU device lost");
    };

    enqueueContrastPreview(queue, 10, requestFrame, render);
    callbacks.shift()!(0);
    enqueueContrastPreview(queue, 20, requestFrame, render);
    expect(callbacks).toHaveLength(1);
    callbacks.shift()!(16);
    await Promise.resolve();
    await Promise.resolve();

    expect(rendered).toEqual([10, 20]);
    expect(errorSpy).toHaveBeenCalledOnce();
    expect(errorSpy.mock.calls[0][0]).toBe("[Show2D] live contrast preview failed");
    expect(errorSpy.mock.calls[0][1]).toEqual(new Error("GPU device lost"));
    expect(queue).toEqual({ frame: 0, inFlight: false, pendingPanel: null, epoch: 0 });
    errorSpy.mockRestore();
  });
});

describe("Show2D auto/manual contrast transitions", () => {
  it("seeds every panel preview range and the linked envelope", () => {
    const seeded = seedAutoContrastMirror(mirror(), [
      { i: 0, vminPct: 2, vmaxPct: 88 },
      { i: 1, vminPct: 7, vmaxPct: 93 },
    ], true);

    expect(seeded.perImage.get(0)).toEqual({ vminPct: 2, vmaxPct: 88 });
    expect(seeded.perImage.get(1)).toEqual({ vminPct: 7, vmaxPct: 93 });
    expect(seeded.linked).toEqual({ vminPct: 2, vmaxPct: 93 });
  });

  it("keeps independent auto ranges independent", () => {
    const seeded = seedAutoContrastMirror(mirror(), [
      { i: 0, vminPct: 2, vmaxPct: 88 },
      { i: 1, vminPct: 7, vmaxPct: 93 },
    ], false);

    expect(seeded.linked).toEqual({ vminPct: 5, vmaxPct: 95 });
    expect(seeded.perImage.get(0)).toEqual({ vminPct: 2, vmaxPct: 88 });
    expect(seeded.perImage.get(1)).toEqual({ vminPct: 7, vmaxPct: 93 });
  });

  it("uses a shared-axis range for a linked histogram when provided", () => {
    const seeded = seedAutoContrastMirror(mirror(), [
      { i: 0, vminPct: 2, vmaxPct: 88 },
      { i: 1, vminPct: 7, vmaxPct: 93 },
    ], true, { vminPct: 4, vmaxPct: 61 });
    expect(seeded.linked).toEqual({ vminPct: 4, vmaxPct: 61 });
  });
});

describe("Show2D linked Auto contrast axis", () => {
  it("converts heterogeneous panels on one shared physical axis", () => {
    const range = linkedAutoContrastPercentRange(
      [{ min: 100, max: 200 }, { min: -10, max: 10 }],
      [{ vmin: 110, vmax: 120 }, { vmin: -2, vmax: 3 }],
      [0, 1],
    );

    expect(range?.vminPct).toBeCloseTo((8 / 210) * 100);
    expect(range?.vmaxPct).toBeCloseTo((130 / 210) * 100);
  });

  it("excludes RGB outliers from the linked grayscale axis", () => {
    expect(linkedAutoContrastPercentRange(
      [{ min: 0, max: 100 }, { min: -1e9, max: 1e9 }, { min: 50, max: 150 }],
      [{ vmin: 10, vmax: 90 }, { vmin: -1e8, vmax: 1e8 }, { vmin: 60, vmax: 120 }],
      [0, 1, 2],
      [false, true, false],
    )).toEqual({ vminPct: 20 / 3, vmaxPct: 80 });
  });

  it("returns no linked axis for RGB-only or flat panel sets", () => {
    expect(linkedAutoContrastPercentRange(
      [{ min: 0, max: 1 }],
      [{ vmin: 0, vmax: 1 }],
      [0],
      [true],
    )).toBeUndefined();
    expect(linkedAutoContrastPercentRange(
      [{ min: 7, max: 7 }],
      [{ vmin: 7, vmax: 7 }],
      [0],
    )).toBeUndefined();
  });
});

describe("Show2D contrast link toggle", () => {
  it("adopts the selected panel when linking", () => {
    const linked = transitionContrastLinkState(mirror(), true, 1, 2);
    expect(linked.linked).toEqual({ vminPct: 20, vmaxPct: 70 });
  });

  it("copies the shared range to every panel when unlinking", () => {
    const unlinked = transitionContrastLinkState(mirror(), false, 0, 3);
    expect([...unlinked.perImage.values()]).toEqual([
      { vminPct: 5, vmaxPct: 95 },
      { vminPct: 5, vmaxPct: 95 },
      { vminPct: 5, vmaxPct: 95 },
    ]);
  });
});
