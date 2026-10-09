import { afterEach, describe, expect, it, vi } from "vitest";
import { downloadBlob } from "../format";
import { copyPngOrDownload } from "./copyImage";

vi.mock("../format", () => ({ downloadBlob: vi.fn() }));

afterEach(() => { vi.unstubAllGlobals(); vi.mocked(downloadBlob).mockClear(); });

describe("copy the diffraction image", () => {
  const blob = new Blob(["png"], { type: "image/png" });

  it("writes the image to the clipboard", async () => {
    const write = vi.fn(async () => {});
    vi.stubGlobal("navigator", { clipboard: { write } });
    vi.stubGlobal("ClipboardItem", class { constructor(public items: object) {} });
    await copyPngOrDownload(blob, "diffraction.png");
    expect(write).toHaveBeenCalledOnce();
    expect(downloadBlob).not.toHaveBeenCalled();
  });

  it("downloads the image when the browser refuses the write", async () => {
    vi.stubGlobal("navigator", { clipboard: { write: vi.fn(async () => { throw new DOMException("denied", "NotAllowedError"); }) } });
    vi.stubGlobal("ClipboardItem", class { constructor(public items: object) {} });
    await copyPngOrDownload(blob, "diffraction.png");
    expect(downloadBlob).toHaveBeenCalledWith(blob, "diffraction.png");
  });

  it("downloads the image when the page has no clipboard", async () => {
    vi.stubGlobal("navigator", {});
    await copyPngOrDownload(blob, "diffraction.png");
    expect(downloadBlob).toHaveBeenCalledWith(blob, "diffraction.png");
  });
});
