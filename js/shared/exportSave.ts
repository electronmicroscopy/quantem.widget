/** Save-dialog and file-writing steps shared by the widget export menus. */
import * as React from "react";
import { downloadBlob, extractBytes } from "../format";
import { formatSavedBytes, isAbortLikeError } from "./exportFormat";

type WritableFile = {
  write: (data: BlobPart) => Promise<void>;
  close: () => Promise<void>;
};

/** A file picked in the browser's save dialog. */
export type SaveFileHandle = {
  createWritable: () => Promise<WritableFile>;
};

export type SavePickerType = { description: string; accept: Record<string, string[]> };

type SavePickerOptions = {
  suggestedName?: string;
  types?: SavePickerType[];
};

/** ``window`` with the File System Access save dialog, which some browsers lack. */
export type SavePickerWindow = Window & typeof globalThis & {
  showSaveFilePicker?: (options?: SavePickerOptions) => Promise<SaveFileHandle>;
};

/** An export the widget asked Python for and has not saved yet. */
export type PendingSave = { id: string; filename: string; handle: SaveFileHandle | null };

export const HTML_PICKER_TYPE: SavePickerType = { description: "Standalone HTML", accept: { "text/html": [".html"] } };

/**
 * Ask where to save ``filename``: the picked file, or null where the browser has no save dialog
 * (the export then downloads). A cancel or failure is reported through ``setBusy``/``setStatus``
 * and resolves to undefined.
 */
export async function pickSaveHandle(
  filename: string,
  type: SavePickerType,
  setBusy: (busy: boolean) => void,
  setStatus: (status: string) => void,
): Promise<SaveFileHandle | null | undefined> {
  const picker = (window as SavePickerWindow).showSaveFilePicker;
  if (!picker) return null;
  try {
    return await picker({ suggestedName: filename, types: [type] });
  } catch (err) {
    setBusy(false);
    setStatus(isAbortLikeError(err) ? "Export canceled" : `Export failed: ${err instanceof Error ? err.message : String(err)}`);
    return undefined;
  }
}

/**
 * Write the export bytes Python returns for ``pendingRef``'s request to the picked file (or the
 * browser downloads), report the result and clear the request. ``blobType`` and ``savedStatus``
 * default to a standalone HTML file and "Saved <name> (<size>)".
 */
export function useSaveExportPayload<P extends PendingSave>(
  pendingRef: React.MutableRefObject<P | null>,
  exportPayload: DataView,
  exportPayloadId: string,
  exportPayloadFilename: string,
  setExportRequest: (request: string) => void,
  setBusy: (busy: boolean) => void,
  setStatus: (status: string) => void,
  blobType: (pending: P) => string = () => "text/html;charset=utf-8",
  savedStatus: (pending: P, filename: string, size: string) => string = (_pending, filename, size) => `Saved ${filename} (${size})`,
): void {
  React.useEffect(() => {
    const pending = pendingRef.current;
    if (!pending || exportPayloadId !== pending.id) return;
    const bytes = extractBytes(exportPayload);
    if (bytes.length === 0) return;
    let canceled = false;
    const save = async () => {
      const payload = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
        ? bytes
        : bytes.slice();
      const filename = exportPayloadFilename || pending.filename;
      const blob = new Blob([payload as BlobPart], { type: blobType(pending) });
      try {
        if (pending.handle) {
          setStatus(`Saving ${filename}...`);
          const writable = await pending.handle.createWritable();
          await writable.write(blob);
          await writable.close();
        } else {
          downloadBlob(blob, filename);
        }
        if (canceled) return;
        pendingRef.current = null;
        setBusy(false);
        setStatus(savedStatus(pending, filename, formatSavedBytes(bytes.byteLength)));
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      } catch (err) {
        if (canceled) return;
        pendingRef.current = null;
        setBusy(false);
        setStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      }
    };
    void save();
    return () => { canceled = true; };
  }, [exportPayload, exportPayloadId, exportPayloadFilename, setExportRequest]);
}
