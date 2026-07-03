import * as React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";
import { useHideStaticFallback } from "./staticFallback";

function Harness({ shouldHide }: { shouldHide: boolean }) {
  const ref = React.useRef<HTMLDivElement | null>(null);
  useHideStaticFallback({ model_id: "model-1" }, ref, shouldHide);
  return React.createElement("div", { ref, className: "show2d-root" }, "live widget");
}

let root: Root | null = null;

afterEach(() => {
  if (root) {
    act(() => root?.unmount());
    root = null;
  }
  document.body.innerHTML = "";
});

function setupSavedNotebookCell() {
  document.body.innerHTML = `
    <div class="jp-Cell">
      <div id="mount"></div>
      <div id="fallback-output" class="jp-OutputArea-child">
        <img
          class="quantem-static-fallback"
          data-quantem-model-id="model-1"
          src="data:image/jpeg;base64,abc"
          alt="Show2D static render"
        >
      </div>
    </div>
  `;
  const mount = document.getElementById("mount");
  if (!mount) throw new Error("missing mount node");
  root = createRoot(mount);
}

describe("useHideStaticFallback", () => {
  it("keeps saved static output visible until the live widget has pixels", () => {
    setupSavedNotebookCell();

    act(() => root?.render(React.createElement(Harness, { shouldHide: false })));

    expect(document.getElementById("fallback-output")?.style.display).toBe("");
  });

  it("hides the saved static output when the live widget has pixels", () => {
    setupSavedNotebookCell();

    act(() => root?.render(React.createElement(Harness, { shouldHide: true })));

    expect(document.getElementById("fallback-output")?.style.display).toBe("none");
  });
});
