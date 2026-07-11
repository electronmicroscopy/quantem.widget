/** Pure layout helpers for sequential Show2D folder pages. */

export function normalizedItemOrder(
  nImages: number,
  panelOrder: number[] | undefined,
): number[] {
  const count = Math.max(0, Math.trunc(Number(nImages) || 0));
  const natural = Array.from({ length: count }, (_, index) => index);
  if (!Array.isArray(panelOrder) || panelOrder.length !== count) return natural;
  const order = panelOrder.map(value => Math.trunc(Number(value)));
  const valid = order.every(value => (
    Number.isFinite(value) && value >= 0 && value < count
  )) && new Set(order).size === count;
  return valid ? order : natural;
}

export function itemPageIndices(
  nImages: number,
  pageIdx: number,
  pageSize: number,
  panelOrder: number[] | undefined,
): number[] {
  const order = normalizedItemOrder(nImages, panelOrder);
  const size = Math.max(1, Math.trunc(Number(pageSize) || 1));
  const pageCount = Math.max(1, Math.ceil(order.length / size));
  const page = Math.max(
    0,
    Math.min(pageCount - 1, Math.trunc(Number(pageIdx) || 0)),
  );
  const start = page * size;
  return order.slice(start, start + size);
}

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
