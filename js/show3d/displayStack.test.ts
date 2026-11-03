import { describe, expect, it } from "vitest";

import { BROWSER_MESSAGE_LIMIT_BYTES, missingDisplayStackStatus } from "./displayStack";

describe("missing display stack status", () => {
  it("names the cause and the display_bin the notebook printed for a 50 x 4096 x 4096 float32 stack", () => {
    // Python printed: "Show3D display payload is 3200 MB; ... Pass display_bin=2 (800 MB) for an interactive view."
    const status = missingDisplayStackStatus(50 * 4096 * 4096 * 4);
    expect(status).toBe(
      "The 3200 MB display stack did not reach the browser: one browser message holds at most 2048 MB. "
      + "Pass display_bin=2 (800 MB) for an interactive view, as printed in the notebook",
    );
    expect(status).not.toMatch(/WebGPU/);
  });

  it("picks the smallest bin that fits, as announce_browser_limit does", () => {
    // 9 x the limit needs a 3x bin (9 / 9 = 1), 9 x the limit + 1 byte needs 4x.
    expect(missingDisplayStackStatus(9 * BROWSER_MESSAGE_LIMIT_BYTES)).toContain("display_bin=3 (2048 MB)");
    expect(missingDisplayStackStatus(9 * BROWSER_MESSAGE_LIMIT_BYTES + 1)).toContain("display_bin=4 (1152 MB)");
  });

  it("suggests a factor above the display_bin already passed", () => {
    // display_bin=2 of a 9 x limit native stack leaves 2.25 x limit, so 3x is the next that fits.
    const binned = (9 * BROWSER_MESSAGE_LIMIT_BYTES) / 4;
    expect(missingDisplayStackStatus(binned, 2)).toContain("display_bin=3 (2048 MB)");
    expect(missingDisplayStackStatus(binned, 2)).not.toContain("display_bin=2");
  });

  it("states only the fact when the stack was small enough to arrive", () => {
    expect(missingDisplayStackStatus(64 * 1024 * 1024)).toBe("The 64.0 MB display stack did not reach the browser");
  });
});
