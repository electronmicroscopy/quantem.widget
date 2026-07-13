import { describe, expect, it } from "vitest";

import {
  itemPageIndices,
  normalizedItemOrder,
  pageShortcutTarget,
  usesGalleryLayout,
} from "./itemPages";

describe("Show2D sequential folder pages", () => {
  it("keeps a partial final page without fake panels", () => {
    expect(itemPageIndices(45, 0, 20, [])).toEqual(
      Array.from({ length: 20 }, (_, index) => index),
    );
    expect(itemPageIndices(45, 2, 20, [])).toEqual([40, 41, 42, 43, 44]);
  });

  it("slices a complete path-stable panel order", () => {
    expect(normalizedItemOrder(5, [4, 2, 0, 3, 1])).toEqual([4, 2, 0, 3, 1]);
    expect(itemPageIndices(5, 1, 2, [4, 2, 0, 3, 1])).toEqual([0, 3]);
    expect(normalizedItemOrder(3, [0, 0, 2])).toEqual([0, 1, 2]);
  });

  it("maps number shortcuts to the active page instead of global indices", () => {
    expect(pageShortcutTarget([40, 41, 44], 1, true, 45)).toBe(41);
    expect(pageShortcutTarget([40, 41, 44], 4, true, 45)).toBeNull();
    expect(pageShortcutTarget([40, 41, 44], 1, false, 45)).toBe(1);
  });

  it("keeps a one-item final page on the indexed gallery render path", () => {
    expect(usesGalleryLayout(21, true)).toBe(true);
    expect(usesGalleryLayout(1, false)).toBe(false);
  });
});
