import { describe, expect, it } from "vitest";
import { extractFloat32 } from "./format";

describe("extractFloat32", () => {
  it("ignores trailing base64-safe pad bytes when the expected float count is known", () => {
    const values = new Float32Array([0, 0.5, 1]);
    const padded = new Uint8Array(values.byteLength + 2);
    padded.set(new Uint8Array(values.buffer));

    const parsed = extractFloat32(new DataView(padded.buffer), values.length);

    expect(Array.from(parsed ?? [])).toEqual([0, 0.5, 1]);
  });

  it("decodes expected floats from an offset DataView with trailing pad bytes", () => {
    const values = new Float32Array([3, 4, 5]);
    const padded = new Uint8Array(3 + values.byteLength + 2);
    padded.set(new Uint8Array(values.buffer), 3);

    const parsed = extractFloat32(new DataView(padded.buffer, 3), values.length);

    expect(Array.from(parsed ?? [])).toEqual([3, 4, 5]);
  });

  it("rejects unaligned byte counts when no expected float count is provided", () => {
    const values = new Float32Array([1, 2]);
    const padded = new Uint8Array(values.byteLength + 1);
    padded.set(new Uint8Array(values.buffer));

    expect(extractFloat32(padded)).toBeNull();
  });
});
