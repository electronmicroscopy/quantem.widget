import { downloadBlob } from "../format";

/** Copy a PNG to the clipboard, or download it when the page has no clipboard (an insecure
 * origin) or the browser refuses the write. The write is awaited because a refusal is an
 * asynchronous rejection, which a synchronous try/catch never sees. */
export async function copyPngOrDownload(blob: Blob, filename: string): Promise<void> {
  try { await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]); }
  catch { downloadBlob(blob, filename); }
}
