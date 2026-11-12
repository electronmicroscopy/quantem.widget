import { describe, expect, it } from "vitest";

import { pageShortcutTarget, usesGalleryLayout } from "./pageLayout";

describe("Show2D page layout", () => {
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
