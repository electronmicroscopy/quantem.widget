import * as React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import {
  FolderWatchBadge,
  effectiveFolderWatchState,
  folderWatchAnnouncement,
  folderWatchModelIsLive,
  folderWatchPresentation,
  useFolderWatchModelLive,
} from "./folderWatchStatus";


describe("folderWatchPresentation", () => {
  it("maps the live protocol to stable compact labels", () => {
    // C1: every public state is presented consistently across all viewers.
    expect(folderWatchPresentation("watching")?.label).toBe("Watching");
    expect(folderWatchPresentation("updating")?.label).toBe("Updating");
    expect(folderWatchPresentation("waiting")?.label).toBe("Waiting for file completion");
    expect(folderWatchPresentation("error")?.label).toBe("Watch error");
    expect(folderWatchPresentation("stopped")?.label).toBe("Stopped");
    expect(folderWatchPresentation("not_watching")?.label).toBe("Not watching");
  });

  it("hides fixed snapshots and treats an unknown backend state as an error", () => {
    // C1: watch=False emits no badge; an invalid state must never look green.
    expect(folderWatchPresentation("hidden")).toBeNull();
    expect(folderWatchPresentation("unexpected")).toMatchObject({
      state: "error",
      label: "Watch error",
    });
  });

  it("announces only states that need a user's attention", () => {
    // C1: routine scan cadence remains visible without a live-region message.
    expect(folderWatchAnnouncement("watching")).toBeNull();
    expect(folderWatchAnnouncement("updating")).toBeNull();

    // C2: waiting, failure, and stop transitions remain accessible.
    expect(folderWatchAnnouncement("waiting")).toEqual({
      role: "status",
      live: "polite",
    });
    expect(folderWatchAnnouncement("unexpected")).toEqual({
      role: "alert",
      live: "assertive",
    });
    expect(folderWatchAnnouncement("stopped")).toEqual({
      role: "status",
      live: "polite",
    });
  });

  it("does not trust a restored watcher state without a live comm", () => {
    // C1: Jupyter's restored model explicitly reports a dead comm, expect all
    // active/transient states to become a truthful neutral snapshot state.
    expect(folderWatchModelIsLive({ comm_live: false })).toBe(false);
    expect(folderWatchModelIsLive({ _comm_live: false })).toBe(false);
    expect(effectiveFolderWatchState("watching", false)).toBe("not_watching");
    expect(effectiveFolderWatchState("waiting", false)).toBe("not_watching");
    expect(effectiveFolderWatchState("error", false)).toBe("not_watching");
    expect(effectiveFolderWatchState("stopped", false)).toBe("stopped");

    // C2: a host without Jupyter's private comm marker keeps the synchronized
    // state instead of hiding a genuinely live watcher.
    expect(folderWatchModelIsLive({})).toBe(true);
  });
});

let root: Root | null = null;

function renderBadge(state: string, detail = ""): HTMLElement | null {
  let mount = document.getElementById("folder-watch-test-mount");
  if (!mount) {
    mount = document.createElement("div");
    mount.id = "folder-watch-test-mount";
    mount.style.color = "rgb(224, 224, 224)";
    mount.style.backgroundColor = "rgb(30, 30, 30)";
    document.body.appendChild(mount);
    root = createRoot(mount);
  }
  act(() => root?.render(React.createElement(FolderWatchBadge, { state, detail })));
  return document.querySelector<HTMLElement>("[data-folder-watch-state]");
}

afterEach(() => {
  if (root) {
    act(() => root?.unmount());
    root = null;
  }
  document.body.innerHTML = "";
});

describe("FolderWatchBadge", () => {
  it("hides watch=False and suppresses routine polling announcements", () => {
    // C1: a fixed snapshot has no badge in the rendered DOM.
    expect(renderBadge("hidden")).toBeNull();

    // C2: live routine states are readable but are not live regions.
    const watching = renderBadge("watching");
    expect(watching?.dataset.folderWatchState).toBe("watching");
    expect(watching?.getAttribute("role")).toBeNull();
    expect(watching?.getAttribute("aria-live")).toBeNull();
    expect(watching?.getAttribute("aria-label")).toBe("Watching");
  });

  it("normalizes unknown states and exposes corrective detail accessibly", () => {
    // C1: malformed backend state must have canonical red-error semantics.
    const badge = renderBadge("unexpected", "  storage unavailable  ");
    expect(badge?.dataset.folderWatchState).toBe("error");
    expect(badge?.getAttribute("role")).toBe("alert");
    expect(badge?.getAttribute("aria-live")).toBe("assertive");
    expect(badge?.getAttribute("aria-atomic")).toBe("true");
    expect(badge?.getAttribute("aria-label")).toBe(
      "Watch error: storage unavailable",
    );
    expect(badge?.getAttribute("title")).toBe(
      "Watch error: storage unavailable",
    );
  });

  it("keeps badge and detail nodes stable while states change", () => {
    // C1: polling transitions update attributes without replacing badge DOM.
    const watching = renderBadge("watching");
    const hiddenDetail = watching?.querySelector<HTMLElement>(
      "[data-folder-watch-detail]",
    );
    expect(hiddenDetail?.dataset.folderWatchDetailVisible).toBe("false");

    const waiting = renderBadge("waiting", "scan_01_master.h5 is incomplete");
    const visibleDetail = waiting?.querySelector<HTMLElement>(
      "[data-folder-watch-detail]",
    );
    expect(waiting).toBe(watching);
    expect(visibleDetail).toBe(hiddenDetail);
    expect(visibleDetail?.dataset.folderWatchDetailVisible).toBe("true");
    expect(visibleDetail?.textContent).toContain("scan_01_master.h5 is incomplete");
    expect(waiting?.getAttribute("role")).toBe("status");
    expect(waiting?.getAttribute("aria-live")).toBe("polite");
  });

  it("renders restored live state as Not watching without stale detail", () => {
    // C1: a notebook saved during acquisition reopens without a worker, expect
    // no false green and no stale transient error/detail text.
    let mount = document.getElementById("folder-watch-test-mount");
    if (!mount) {
      mount = document.createElement("div");
      mount.id = "folder-watch-test-mount";
      document.body.appendChild(mount);
      root = createRoot(mount);
    }
    act(() => root?.render(React.createElement(FolderWatchBadge, {
      state: "watching",
      detail: "old acquisition detail",
      live: false,
    })));
    const badge = document.querySelector<HTMLElement>("[data-folder-watch-state]");
    expect(badge?.dataset.folderWatchState).toBe("not_watching");
    expect(badge?.getAttribute("aria-label")).toBe("Not watching");
    expect(badge?.textContent).not.toContain("old acquisition detail");
  });

  it("drops false green immediately when the live comm closes", () => {
    // C1: Jupyter emits comm:close before its private liveness getter is
    // necessarily updated, expect the mounted badge to become neutral anyway.
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
      const live = useFolderWatchModelLive(model);
      return React.createElement(FolderWatchBadge, { state: "watching", live });
    }
    const mount = document.createElement("div");
    document.body.appendChild(mount);
    root = createRoot(mount);
    act(() => root?.render(React.createElement(Harness)));
    expect(
      document.querySelector<HTMLElement>("[data-folder-watch-state]")?.dataset
        .folderWatchState,
    ).toBe("watching");

    act(() => {
      for (const callback of handlers.get("comm:close") ?? []) callback();
    });
    expect(
      document.querySelector<HTMLElement>("[data-folder-watch-state]")?.dataset
        .folderWatchState,
    ).toBe("not_watching");
  });

  it("inherits high-contrast host text and clips narrow content", () => {
    // C1: viewer themes own text/background; the dot alone carries state color.
    const badge = renderBadge("waiting", "a very long acquisition detail");
    const darkStyles = window.getComputedStyle(badge as HTMLElement);
    const dot = badge?.querySelector<HTMLElement>("[data-folder-watch-dot]");
    expect(darkStyles.color).toBe("rgb(224, 224, 224)");
    expect(darkStyles.backgroundColor).toBe("rgba(0, 0, 0, 0)");
    expect(darkStyles.maxWidth).toBe("100%");
    expect(darkStyles.overflow).toBe("hidden");
    expect(window.getComputedStyle(dot as HTMLElement).backgroundColor).toBe(
      "rgb(237, 108, 2)",
    );

    // C2: the same neutral badge inherits a light viewer's contrasting text.
    const mount = document.getElementById("folder-watch-test-mount");
    if (!mount) throw new Error("missing folder watch test mount");
    mount.style.color = "rgb(30, 30, 30)";
    mount.style.backgroundColor = "rgb(255, 255, 255)";
    expect(window.getComputedStyle(badge as HTMLElement).color).toBe(
      "rgb(30, 30, 30)",
    );
  });
});
