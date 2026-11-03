import { expect, test } from "vitest";
import { collectDroppedFiles, matchQemFiles } from "./localDrop";

/** A dropped file or folder as Chrome's webkitGetAsEntry exposes it; folders list children in batches. */
function fileEntry(name: string) {
  const file = new File([name], name);
  return { isFile: true, isDirectory: false, file: (resolve: (file: File) => void) => resolve(file) };
}

function folderEntry(children: object[], batch: number) {
  return {
    isFile: false,
    isDirectory: true,
    createReader() {
      let next = 0;
      return {
        readEntries(resolve: (entries: object[]) => void) {
          resolve(children.slice(next, next + batch));
          next += batch;
        },
      };
    },
  };
}

const names = (files: File[]) => files.map(file => file.name);

test("a dropped acquisition folder grants every HDF5 file in it, nested folders included", async () => {
  const tilt = "scan_17.0x_0.0y";
  const folder = folderEntry([
    fileEntry(`${tilt}_master.h5`),
    fileEntry(`._${tilt}_master.h5`),
    fileEntry("notes.txt"),
    folderEntry([fileEntry(`${tilt}_data_000001.h5`), fileEntry(`${tilt}_data_000002.h5`)], 1),
    fileEntry(`${tilt}.qh5idx`),
  ], 2);
  const drop = { items: [{ webkitGetAsEntry: () => folder, getAsFile: () => null }] } as unknown as DataTransfer;
  expect(names(await collectDroppedFiles(drop))).toEqual([
    `${tilt}_master.h5`,
    `${tilt}_data_000001.h5`,
    `${tilt}_data_000002.h5`,
    `${tilt}.qh5idx`,
  ]);
});

test("dropped files are taken as they are and filtered the same way", async () => {
  const files = ["a_master.h5", "a_data_000001.H5", "._a_master.h5", "notes.txt", "t0-counts.qem", "b.hdf5"].map(name => new File([name], name));
  // A dropped file's entry may not resolve (a synthesized drag has none on disk); its File is used directly.
  const unreadable = { isFile: true, isDirectory: false, file: (_: unknown, reject: (error: Error) => void) => reject(new Error("NotFoundError")) };
  const drop = { items: files.map(file => ({ webkitGetAsEntry: () => unreadable, getAsFile: () => file })) } as unknown as DataTransfer;
  expect(names(await collectDroppedFiles(drop))).toEqual(["a_master.h5", "a_data_000001.H5", "t0-counts.qem", "b.hdf5"]);
});

test("a .qem viewer takes its granted files in export order and names any it lacks", () => {
  const granted = ["t2-counts.qem", "t0-counts.qem", "notes.qem", "t1-counts.qem"].map(name => new File([name], name));
  const expected = ["t0-counts.qem", "t1-counts.qem", "t2-counts.qem"];
  const matched = matchQemFiles(expected, granted);
  expect("files" in matched && names(matched.files)).toEqual(expected);
  expect(matchQemFiles(expected, granted.slice(1))).toEqual({ missing: ["t2-counts.qem"] });
  expect(matchQemFiles(expected, [...granted, new File(["copy"], "t1-counts.qem")])).toEqual({ missing: ["t1-counts.qem"] });
});
