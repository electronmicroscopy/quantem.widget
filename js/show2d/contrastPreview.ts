export type ContrastPercentRange = {
  vminPct: number;
  vmaxPct: number;
};

export type AutoContrastPercentRange = ContrastPercentRange & {
  i: number;
};

export type ContrastMirror = {
  linked: ContrastPercentRange;
  perImage: Map<number, ContrastPercentRange>;
};

export type ContrastPreviewQueue = {
  frame: number;
  inFlight: boolean;
  pendingPanel: number | null;
  epoch: number;
};

const FULL_CONTRAST: ContrastPercentRange = { vminPct: 0, vmaxPct: 100 };

export function contrastPreviewIndices(
  linked: boolean,
  visibleIndices: number[],
  activeIndex: number,
  imageCount: number,
  rgbPanels: boolean[] = [],
): number[] {
  const candidates = linked
    ? (visibleIndices.length > 0
      ? visibleIndices
      : Array.from({ length: imageCount }, (_, index) => index))
    : [activeIndex];
  return candidates.filter((index) => (
    index >= 0
    && index < imageCount
    && !rgbPanels[index]
  ));
}

export function seedAutoContrastMirror(
  mirror: ContrastMirror,
  ranges: AutoContrastPercentRange[],
  linked: boolean,
  linkedRange?: ContrastPercentRange,
): ContrastMirror {
  const perImage = new Map(mirror.perImage);
  for (const range of ranges) {
    perImage.set(range.i, {
      vminPct: range.vminPct,
      vmaxPct: range.vmaxPct,
    });
  }
  if (!linked || ranges.length === 0) {
    return { linked: mirror.linked, perImage };
  }
  return {
    linked: linkedRange || {
      vminPct: Math.min(...ranges.map((range) => range.vminPct)),
      vmaxPct: Math.max(...ranges.map((range) => range.vmaxPct)),
    },
    perImage,
  };
}

export function transitionContrastLinkState(
  mirror: ContrastMirror,
  nextLinked: boolean,
  selectedIndex: number,
  imageCount: number,
): ContrastMirror {
  if (nextLinked) {
    return {
      linked: mirror.perImage.get(selectedIndex) || mirror.linked || FULL_CONTRAST,
      perImage: new Map(mirror.perImage),
    };
  }
  const perImage = new Map(mirror.perImage);
  for (let index = 0; index < imageCount; index++) {
    perImage.set(index, mirror.linked);
  }
  return { linked: mirror.linked, perImage };
}

export function linkedAutoContrastPercentRange(
  baseRanges: { min: number; max: number }[],
  autoRanges: { vmin: number; vmax: number }[],
  panelIndices: number[],
  rgbPanels: boolean[] = [],
): ContrastPercentRange | undefined {
  const grayscaleIndices = [...new Set(panelIndices)].filter((index) => {
    if (index < 0 || index >= baseRanges.length || index >= autoRanges.length) return false;
    if (rgbPanels[index]) return false;
    const base = baseRanges[index];
    const auto = autoRanges[index];
    return Number.isFinite(base.min)
      && Number.isFinite(base.max)
      && Number.isFinite(auto.vmin)
      && Number.isFinite(auto.vmax);
  });
  if (grayscaleIndices.length === 0) return undefined;

  const baseMin = Math.min(...grayscaleIndices.map((index) => baseRanges[index].min));
  const baseMax = Math.max(...grayscaleIndices.map((index) => baseRanges[index].max));
  const span = baseMax - baseMin;
  if (!(span > 0)) return undefined;

  const autoMin = Math.min(...grayscaleIndices.map((index) => autoRanges[index].vmin));
  const autoMax = Math.max(...grayscaleIndices.map((index) => autoRanges[index].vmax));
  const clampPercent = (value: number) => Math.max(0, Math.min(100, value));
  return {
    vminPct: clampPercent(((autoMin - baseMin) / span) * 100),
    vmaxPct: clampPercent(((autoMax - baseMin) / span) * 100),
  };
}

export function enqueueContrastPreview(
  queue: ContrastPreviewQueue,
  panel: number,
  requestFrame: (callback: FrameRequestCallback) => number,
  render: (panel: number) => void | Promise<void>,
): void {
  queue.pendingPanel = panel;
  if (queue.frame !== 0 || queue.inFlight) return;
  const epoch = queue.epoch;
  queue.frame = requestFrame(() => {
    if (queue.epoch !== epoch) return;
    queue.frame = 0;
    const pendingPanel = queue.pendingPanel;
    queue.pendingPanel = null;
    if (pendingPanel === null) return;
    queue.inFlight = true;
    const finish = () => {
      if (queue.epoch !== epoch) return;
      queue.inFlight = false;
      if (queue.pendingPanel !== null) {
        enqueueContrastPreview(queue, queue.pendingPanel, requestFrame, render);
      }
    };
    let result: void | Promise<void>;
    try {
      result = render(pendingPanel);
    } catch (error) {
      console.error("[Show2D] live contrast preview failed", error);
      finish();
      return;
    }
    void Promise.resolve(result)
      .catch((error: unknown) => console.error("[Show2D] live contrast preview failed", error))
      .finally(finish);
  });
}

export function cancelContrastPreview(
  queue: ContrastPreviewQueue,
  cancelFrame: (frame: number) => void,
): void {
  if (queue.frame !== 0) cancelFrame(queue.frame);
  queue.epoch += 1;
  queue.frame = 0;
  queue.inFlight = false;
  queue.pendingPanel = null;
}
