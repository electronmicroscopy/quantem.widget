import * as React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import { modelCommIsLive, useModelCommLive } from "./commLive";

let root: Root | null = null;

afterEach(() => {
  if (root) {
    act(() => root?.unmount());
    root = null;
  }
  document.body.innerHTML = "";
});

describe("modelCommIsLive", () => {
  it("trusts Jupyter's comm marker and assumes live without one", () => {
    // C1: Jupyter's restored model explicitly reports a dead comm.
    expect(modelCommIsLive({ comm_live: false })).toBe(false);
    expect(modelCommIsLive({ _comm_live: false })).toBe(false);
    expect(modelCommIsLive({ comm_live: true })).toBe(true);
    // C2: a host without Jupyter's private comm marker keeps the synchronized state.
    expect(modelCommIsLive({})).toBe(true);
  });
});

describe("useModelCommLive", () => {
  it("turns false immediately when the live comm closes", () => {
    // Jupyter emits comm:close before its private liveness getter is
    // necessarily updated, expect the hook to report a dead comm anyway.
    const handlers = new Map<string, Set<() => void>>();
    const model = {
      comm_live: true,
      on(event: string, callback: () => void) {
        const callbacks = handlers.get(event) ?? new Set<() => void>();
        callbacks.add(callback);
        handlers.set(event, callbacks);
      },
      off(event: string, callback: () => void) {
        handlers.get(event)?.delete(callback);
      },
    };
    function Harness() {
      const live = useModelCommLive(model);
      return React.createElement("span", { "data-live": String(live) });
    }
    const mount = document.createElement("div");
    document.body.appendChild(mount);
    root = createRoot(mount);
    act(() => root?.render(React.createElement(Harness)));
    expect(document.querySelector<HTMLElement>("[data-live]")?.dataset.live).toBe("true");
    act(() => {
      for (const callback of handlers.get("comm:close") ?? []) callback();
    });
    expect(document.querySelector<HTMLElement>("[data-live]")?.dataset.live).toBe("false");
  });
});
