import { describe, expect, it, vi } from "vitest";
import type { MasterFile, Session } from "./types";

const storeCalls = vi.hoisted(() => ({
  pinned: [] as string[][],
  warmed: [] as unknown[][],
}));

vi.mock("../../local/store", () => ({
  setPinned5DKeys: (keys: string[]) => { storeCalls.pinned.push(keys); },
  warmSet5D: (frames: unknown[]) => { storeCalls.warmed.push(frames); return Promise.resolve(); },
}));

import { masterBytesAtBin, pickAutoBin, planWarmSet5D, preloadSet5D } from "./types";

function file(name: string, shape: [number, number, number, number] = [120, 120, 192, 192]): MasterFile {
  const [scanW, scanH, detW, detH] = shape;
  return {
    name,
    path: `/gold/${name}`,
    shape,
    size_bytes: scanW * scanH * detW * detH * 2,
  };
}

const session: Session = {
  source: "picked",
  date: "gold",
  label: "gold",
  files: [],
};

describe("standalone 5D WebGPU resident planning", () => {
  it("budgets binned uint8 resident bytes instead of full uint16 bytes", () => {
    const files = Array.from({ length: 20 }, (_, i) => file(`f${i}_master.h5`));
    const freeBytes = 20 * 1024 ** 3;

    expect(masterBytesAtBin(files[0], 1, "uint16")).toBe(120 * 120 * 192 * 192 * 2);
    expect(masterBytesAtBin(files[0], 2, "uint8")).toBe(120 * 120 * 96 * 96);
    expect(pickAutoBin(files, freeBytes, 0.45, "uint8")).toBe(2);

    const full = planWarmSet5D(files, 10, freeBytes, 1, "uint8");
    const binned = planWarmSet5D(files, 10, freeBytes, 2, "uint8");

    expect(full.mode).toBe("window");
    expect(full.files.length).toBeLessThan(files.length);
    expect(binned.mode).toBe("all");
    expect(binned.files.length).toBe(files.length);
  });

  it("pins and warms with bin/dtype-specific cache keys", async () => {
    storeCalls.pinned.length = 0;
    storeCalls.warmed.length = 0;
    const files = [file("a_master.h5"), file("b_master.h5")];

    await preloadSet5D(session, files, 2, "uint8", 0, 20 * 1024 ** 3);

    expect(storeCalls.pinned.at(-1)).toEqual([
      "picked/gold/a_master.h5|b2|uint8",
      "picked/gold/b_master.h5|b2|uint8",
    ]);
    expect(storeCalls.warmed.at(-1)).toEqual([
      { source: "picked", date: "gold", name: "a_master.h5", detBin: 2, dtype: "uint8" },
      { source: "picked", date: "gold", name: "b_master.h5", detBin: 2, dtype: "uint8" },
    ]);
  });
});
