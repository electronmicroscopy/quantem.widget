/**
 * Files of a folder dropped on a served-less Show4DSTEM viewer: the .qem
 * acquisitions, or the HDF5 masters, data files and index sidecars the "Open
 * data folder" picker grants. Only File handles are collected; no detector
 * bytes are read here.
 */
interface DroppedEntry {
  isFile: boolean;
  isDirectory: boolean;
  file(success: (file: File) => void, failure: (error: DOMException) => void): void;
  createReader(): { readEntries(success: (entries: DroppedEntry[]) => void, failure: (error: DOMException) => void): void };
}

// "._" names are macOS AppleDouble metadata twins of real files, not acquisitions.
const isAcquisitionFile = (file: File) => !file.name.startsWith("._") && /\.(qem|h5|hdf5|qh5idx|qbslz4)$/i.test(file.name);

export async function collectDroppedFiles(data: DataTransfer, maxFiles = 10000): Promise<File[]> {
  // Items must be read during the drop event: the browser protects the drag
  // data store once the handler returns. A dropped file is taken as is; only a
  // dropped folder is walked through its entry.
  const dropped = Array.from(data.items, item => ({ entry: item.webkitGetAsEntry() as DroppedEntry | null, file: item.getAsFile() }));
  const files: File[] = [];
  const visit = async (entry: DroppedEntry): Promise<void> => {
    if (files.length >= maxFiles) throw new Error("Too many files. Drop only the acquisition folder.");
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) => entry.file(resolve, reject));
      if (isAcquisitionFile(file)) files.push(file);
    } else if (entry.isDirectory) {
      // readEntries returns a folder in batches until an empty one.
      const reader = entry.createReader();
      for (;;) {
        const children = await new Promise<DroppedEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
        if (!children.length) break;
        for (const child of children) await visit(child);
      }
    }
  };
  for (const { entry, file } of dropped) {
    if (entry?.isDirectory) await visit(entry);
    else if (file && isAcquisitionFile(file)) files.push(file);
  }
  return files;
}

/**
 * The granted files of a .qem viewer in the viewer's acquisition order.
 *
 * A viewer compares exactly the acquisitions it was exported with, so every
 * expected name must be granted once: a missing file would shift every later
 * panel's label, and two files of one name cannot be told apart.
 */
export function matchQemFiles(expected: string[], files: File[]): { files: File[] } | { missing: string[] } {
  const missing = expected.filter(name => files.filter(file => file.name === name).length !== 1);
  if (missing.length) return { missing };
  return { files: expected.map(name => files.find(file => file.name === name)!) };
}
