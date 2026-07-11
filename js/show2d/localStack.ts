export const DEFAULT_PANEL_PLAYBACK_FPS = 10;
export const MAX_PANEL_PLAYBACK_FPS = 30;
export const GALLERY_FFT_CACHE_MAX_ENTRIES = 64;
export const GALLERY_FFT_CACHE_MAX_BYTES = 256 * 1024 * 1024;

export type GalleryFftCacheEntry = {
  mag: Float32Array;
  fftWidth: number;
  fftHeight: number;
};

export type GalleryFftCacheStats = {
  entries: number;
  bytes: number;
  evictions: number;
};

export type VisibleDiffPlan = {
  visibleGrayscale: number[];
  reference: number;
  others: number[];
};

export function resolveVisibleDiffPlan(
  visiblePanels: readonly number[],
  isRgb: readonly boolean[] | null | undefined,
  configuredReference: number,
): VisibleDiffPlan {
  const visibleGrayscale = visiblePanels.filter(panel => !isRgb?.[panel]);
  const reference = visibleGrayscale.includes(configuredReference)
    ? configuredReference
    : (visibleGrayscale[0] ?? 0);
  return {
    visibleGrayscale,
    reference,
    others: visibleGrayscale.filter(panel => panel !== reference),
  };
}

export function clampPanelPlaybackFps(value: unknown): number {
  const fps = Number(value);
  if (!Number.isFinite(fps) || fps <= 0) return DEFAULT_PANEL_PLAYBACK_FPS;
  // Python accepts every finite value > 0 so deliberate slow comparison
  // cadences (for example 0.25 fps tomography slices) must survive unchanged.
  return Math.min(MAX_PANEL_PLAYBACK_FPS, fps);
}

export function panelPlaybackIntervalMs(value: unknown): number {
  return 1000 / clampPanelPlaybackFps(value);
}

export function makeGalleryFftCacheKey({
  dataEpoch,
  panel,
  frame,
  width,
  height,
  roiKey,
  fftWindow,
  overviewDownsample,
}: {
  dataEpoch: number;
  panel: number;
  frame: number;
  width: number;
  height: number;
  roiKey: string;
  fftWindow: boolean;
  overviewDownsample: number;
}): string {
  return [
    `data=${dataEpoch}`,
    `panel=${panel}`,
    `frame=${frame}`,
    `dims=${width}x${height}`,
    `roi=${roiKey || "none"}`,
    `window=${fftWindow ? 1 : 0}`,
    `overview=${Math.max(1, Math.round(overviewDownsample || 1))}`,
  ].join("|");
}

export function galleryFftCacheStats(
  cache: Map<string, GalleryFftCacheEntry>,
  evictions = 0,
): GalleryFftCacheStats {
  let bytes = 0;
  for (const entry of cache.values()) bytes += entry.mag.byteLength;
  return { entries: cache.size, bytes, evictions };
}

export function readGalleryFftCache(
  cache: Map<string, GalleryFftCacheEntry>,
  key: string,
): GalleryFftCacheEntry | null {
  const entry = cache.get(key) ?? null;
  if (!entry) return null;
  cache.delete(key);
  cache.set(key, entry);
  return entry;
}

export function rememberGalleryFftCache(
  cache: Map<string, GalleryFftCacheEntry>,
  key: string,
  entry: GalleryFftCacheEntry,
  {
    maxEntries = GALLERY_FFT_CACHE_MAX_ENTRIES,
    maxBytes = GALLERY_FFT_CACHE_MAX_BYTES,
    protectedKeys = new Set<string>(),
  }: {
    maxEntries?: number;
    maxBytes?: number;
    protectedKeys?: ReadonlySet<string>;
  } = {},
): GalleryFftCacheStats {
  cache.delete(key);
  cache.set(key, entry);
  let stats = galleryFftCacheStats(cache);
  let evictions = 0;
  const entryLimit = Math.max(1, Math.round(maxEntries));
  const byteLimit = Math.max(entry.mag.byteLength, Math.round(maxBytes));

  while (cache.size > entryLimit || stats.bytes > byteLimit) {
    let evictKey: string | undefined;
    for (const candidate of cache.keys()) {
      if (candidate !== key && !protectedKeys.has(candidate)) {
        evictKey = candidate;
        break;
      }
    }
    if (evictKey === undefined) break;
    cache.delete(evictKey);
    evictions += 1;
    stats = galleryFftCacheStats(cache);
  }
  return galleryFftCacheStats(cache, evictions);
}
