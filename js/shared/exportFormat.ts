/** Size and filename helpers shared by the widget export menus. */

// Bundle, fonts and markup that every standalone HTML export carries on top of
// the base64 payload; the estimate shown before export adds it to payload*4/3.
export const HTML_EXPORT_OVERHEAD_BYTES = 700_000;

/** Megabytes rounded for display: "123 MB", "12.3 MB", "1.23 MB". */
function formatMegabytes(mb: number): string {
  if (mb >= 100) return `${Math.round(mb)} MB`;
  if (mb >= 10) return `${mb.toFixed(1)} MB`;
  return `${mb.toFixed(2)} MB`;
}

export function formatSavedBytes(bytes: number): string {
  return formatMegabytes(Math.max(0, bytes) / (1024 * 1024));
}

export function formatEstimatedHtmlSize(payloadBytes: number, overheadBytes = HTML_EXPORT_OVERHEAD_BYTES): string {
  const htmlBytes = Math.max(0, payloadBytes) * 4 / 3 + overheadBytes;
  return `~${formatMegabytes(htmlBytes / (1024 * 1024))}`;
}

// A cancelled export (user closed the save dialog) is not an error worth showing.
export function isAbortLikeError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

/** Lowercase, underscore-joined form of a widget title for use in a filename. */
export function exportTitleSlug(title: string, fallback: string): string {
  // Each run of other characters (underscores included) becomes one underscore.
  const slug = (title || fallback).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return slug || fallback;
}
