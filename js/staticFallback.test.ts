import * as React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useHideStaticFallback } from "./staticFallback";

function Harness({ shouldHide }: { shouldHide: boolean }) {
  const ref = React.useRef<HTMLDivElement | null>(null);
  useHideStaticFallback({ model_id: "model-1" }, ref, shouldHide);
  return React.createElement("div", { ref, className: "show2d-root" }, "live widget");
}

function HarnessNoModelId({ shouldHide }: { shouldHide: boolean }) {
  const ref = React.useRef<HTMLDivElement | null>(null);
  useHideStaticFallback({}, ref, shouldHide);
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

  it("hides the static output on docs pages (.cell scope, no model_id)", () => {
    // jupyter-book / myst-nb pages wrap outputs in .cell (no .jp-Cell), and
    // the html-manager model exposes no model_id; only the structural scope
    // can find the sibling there.
    document.body.innerHTML = `
      <div class="cell">
        <div id="mount"></div>
        <div class="cell_output">
          <img
            id="fallback-img"
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

    act(() => root?.render(React.createElement(HarnessNoModelId, { shouldHide: true })));

    expect(document.getElementById("fallback-img")?.style.display).toBe("none");
  });

  it("hides a fallback that arrives later in the cell and watches only that cell", async () => {
    setupSavedNotebookCell();
    document.getElementById("fallback-output")!.remove();
    const observe = vi.spyOn(MutationObserver.prototype, "observe");
    act(() => root?.render(React.createElement(Harness, { shouldHide: true })));
    expect(observe.mock.calls.map(([target]) => target)).toEqual([document.querySelector(".jp-Cell")]);
    const late = document.createElement("div");
    late.className = "jp-OutputArea-child";
    late.innerHTML = '<img class="quantem-static-fallback" data-quantem-model-id="model-1">';
    document.querySelector(".jp-Cell")!.appendChild(late);
    await Promise.resolve();
    expect(late.style.display).toBe("none");
    observe.mockRestore();
  });
});
