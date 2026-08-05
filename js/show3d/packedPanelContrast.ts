export type PackedPanelByteRange = { lo: number; hi: number };

const packedPanelAutoByteRangeCache = new WeakMap<
  object,
  Map<string, PackedPanelByteRange>
>();

function finitePercent(value: number, fallback: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : fallback;
}

export function packedPanelAutoByteRange(
  data: Uint8Array,
  frameWidth: number,
  frameHeight: number,
  panelX: number,
  panelWidth: number,
  lowPct: number,
  highPct: number,
): PackedPanelByteRange | null {
  if (
    !Number.isInteger(frameWidth)
    || !Number.isInteger(frameHeight)
    || frameWidth <= 0
    || frameHeight <= 0
    || data.length < frameWidth * frameHeight
  ) {
    return null;
  }
  const x0 = Math.max(0, Math.min(frameWidth - 1, Math.round(panelX)));
  const x1 = Math.max(x0 + 1, Math.min(frameWidth, Math.round(panelX + panelWidth)));
  const low = finitePercent(lowPct, 0);
  const high = Math.max(low, finitePercent(highPct, 100));
  const key = `${data.byteOffset}:${data.byteLength}:${frameWidth}:${frameHeight}:${x0}:${x1}:${low}:${high}`;
  const bufferKey = data.buffer as object;
  let frameCache = packedPanelAutoByteRangeCache.get(bufferKey);
  const cached = frameCache?.get(key);
  if (cached) return cached;

  const counts = new Uint32Array(256);
  for (let y = 0; y < frameHeight; y++) {
    const row = y * frameWidth;
    for (let x = x0; x < x1; x++) counts[data[row + x]]++;
  }
  const total = (x1 - x0) * frameHeight;
  if (total <= 0) return null;

  const byteAt = (pct: number): number => {
    const target = Math.max(0, Math.min(total - 1, Math.floor((pct / 100) * (total - 1))));
    let cumulative = 0;
    for (let value = 0; value < counts.length; value++) {
      cumulative += counts[value];
      if (cumulative > target) return value;
    }
    return 255;
  };

  let lo = byteAt(low);
  let hi = byteAt(high);
  if (hi <= lo) {
    if (lo > 0) lo--;
    else hi = Math.min(255, hi + 1);
  }
  const range = { lo, hi };
  if (!frameCache) {
    frameCache = new Map();
    packedPanelAutoByteRangeCache.set(bufferKey, frameCache);
  }
  frameCache.set(key, range);
  return range;
}
