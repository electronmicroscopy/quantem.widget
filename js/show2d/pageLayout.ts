/** Pure layout helpers for paged Show2D galleries. */

export function pageShortcutTarget(
  visiblePanels: number[],
  shortcutIndex: number,
  isPaged: boolean,
  nImages: number,
): number | null {
  const slot = Math.trunc(Number(shortcutIndex));
  if (slot < 0) return null;
  if (isPaged) return visiblePanels[slot] ?? null;
  return slot < Math.max(0, Math.trunc(Number(nImages) || 0)) ? slot : null;
}

export function usesGalleryLayout(nImages: number, isPaged: boolean): boolean {
  return Boolean(isPaged) || Math.max(0, Math.trunc(Number(nImages) || 0)) > 1;
}
