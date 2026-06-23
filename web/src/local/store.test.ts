import { beforeEach, describe, expect, it, vi } from "vitest";
import type { LocalFile } from "./store";

vi.mock("jsfive", () => ({
  File: class {
    name: string;

    constructor(_ab: ArrayBuffer, name: string) {
      this.name = name;
      if (name.includes("corrupt")) throw new Error("bad master");
    }

    get(path: string) {
      if (path === "entry/instrument/detector/detectorSpecific/ntrigger") {
        return { value: [16], shape: [1] };
      }
      if (path === "entry/instrument/detector/detectorSpecific/pixel_mask") {
        return { value: new Uint8Array(16), shape: [4, 4] };
      }
      throw new Error(`unexpected HDF5 path ${path}`);
    }
  },
}));

function h5(name: string, relPath = `root/session/${name}`, bytes = vi.fn(async () => new ArrayBuffer(8))): LocalFile {
  return {
    name,
    relPath,
    bytes,
    size: 8,
    lastModified: 1,
  };
}

describe("standalone folder scan race handling", () => {
  beforeEach(async () => {
    const { scanFolder } = await import("./store");
    await scanFolder([]);
  });

  it("ignores orphan data shards until the master is present", async () => {
    const { getSessions, lastScanSkipped, scanFolder } = await import("./store");
    const dataBytes = vi.fn(async () => {
      throw new Error("data shard should not be read during scan");
    });

    await scanFolder([h5("gold_data_000001.h5", "root/session/gold_data_000001.h5", dataBytes)]);

    expect(getSessions()).toEqual([]);
    expect(lastScanSkipped()).toBe(0);
    expect(dataBytes).not.toHaveBeenCalled();
  });

  it("lists a master without sidecar data as present but not loadable", async () => {
    const { getSessions, scanFolder } = await import("./store");

    await scanFolder([h5("gold_master.h5")]);

    expect(getSessions()).toHaveLength(1);
    expect(getSessions()[0].files).toEqual([
      expect.objectContaining({
        name: "gold_master.h5",
        shape: [4, 4, 4, 4],
        loadable: false,
      }),
    ]);
  });

  it("uses the master to claim sidecar data without reading data bytes during scan", async () => {
    const { getSessions, scanFolder } = await import("./store");
    const masterBytes = vi.fn(async () => new ArrayBuffer(8));
    const dataBytes = vi.fn(async () => {
      throw new Error("data shard should only be read when loading, not scanning");
    });

    await scanFolder([
      h5("gold_master.h5", "root/session/gold_master.h5", masterBytes),
      h5("gold_data_000001.h5", "root/session/gold_data_000001.h5", dataBytes),
    ]);

    expect(masterBytes).toHaveBeenCalledTimes(1);
    expect(dataBytes).not.toHaveBeenCalled();
    expect(getSessions()[0].files).toEqual([
      expect.objectContaining({
        name: "gold_master.h5",
        loadable: true,
      }),
    ]);
  });

  it("does not surface a dataset when a corrupt master is observed mid-copy", async () => {
    const { getSessions, lastScanSkipped, scanFolder } = await import("./store");
    const dataBytes = vi.fn(async () => new ArrayBuffer(8));

    await scanFolder([
      h5("corrupt_master.h5"),
      h5("corrupt_data_000001.h5", "root/session/corrupt_data_000001.h5", dataBytes),
    ]);

    expect(getSessions()).toEqual([]);
    expect(lastScanSkipped()).toBe(1);
    expect(dataBytes).not.toHaveBeenCalled();
  });
});
