import * as React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import { useCanvasRepaintSignal } from "./canvasLifecycle";

function Harness() {
  const signal = useCanvasRepaintSignal();
  return React.createElement("output", { "data-testid": "signal" }, signal);
}

let root: Root | null = null;
let hidden = false;
let nextFrameId = 1;
let pendingFrames = new Map<number, FrameRequestCallback>();
let originalHiddenDescriptor: PropertyDescriptor | undefined;
let originalRequestAnimationFrameDescriptor: PropertyDescriptor | undefined;
let originalCancelAnimationFrameDescriptor: PropertyDescriptor | undefined;

const requestAnimationFrameMock = vi.fn((callback: FrameRequestCallback) => {
  const frameId = nextFrameId;
  nextFrameId += 1;
  pendingFrames.set(frameId, callback);
  return frameId;
});
const cancelAnimationFrameMock = vi.fn((frameId: number) => {
  pendingFrames.delete(frameId);
});

function mountHarness(): void {
  const mount = document.createElement("div");
  document.body.appendChild(mount);
  root = createRoot(mount);
  act(() => root?.render(React.createElement(Harness)));
}

function signalValue(): string | null {
  return document.querySelector('[data-testid="signal"]')?.textContent ?? null;
}

function flushAnimationFrame(): void {
  const callbacks = [...pendingFrames.values()];
  pendingFrames.clear();
  act(() => callbacks.forEach((callback) => callback(0)));
}

function dispatch(target: Document | Window, type: string): void {
  act(() => target.dispatchEvent(new Event(type)));
}

beforeEach(() => {
  hidden = false;
  nextFrameId = 1;
  pendingFrames = new Map();
  requestAnimationFrameMock.mockClear();
  cancelAnimationFrameMock.mockClear();

  originalHiddenDescriptor = Object.getOwnPropertyDescriptor(document, "hidden");
  originalRequestAnimationFrameDescriptor = Object.getOwnPropertyDescriptor(
    window,
    "requestAnimationFrame",
  );
  originalCancelAnimationFrameDescriptor = Object.getOwnPropertyDescriptor(
    window,
    "cancelAnimationFrame",
  );
  Object.defineProperty(document, "hidden", {
    configurable: true,
    get: () => hidden,
  });
  Object.defineProperty(window, "requestAnimationFrame", {
    configurable: true,
    writable: true,
    value: requestAnimationFrameMock,
  });
  Object.defineProperty(window, "cancelAnimationFrame", {
    configurable: true,
    writable: true,
    value: cancelAnimationFrameMock,
  });
});

afterEach(() => {
  if (root) {
    act(() => root?.unmount());
    root = null;
  }
  document.body.innerHTML = "";

  if (originalHiddenDescriptor) {
    Object.defineProperty(document, "hidden", originalHiddenDescriptor);
  } else {
    Reflect.deleteProperty(document, "hidden");
  }
  if (originalRequestAnimationFrameDescriptor) {
    Object.defineProperty(
      window,
      "requestAnimationFrame",
      originalRequestAnimationFrameDescriptor,
    );
  } else {
    Reflect.deleteProperty(window, "requestAnimationFrame");
  }
  if (originalCancelAnimationFrameDescriptor) {
    Object.defineProperty(
      window,
      "cancelAnimationFrame",
      originalCancelAnimationFrameDescriptor,
    );
  } else {
    Reflect.deleteProperty(window, "cancelAnimationFrame");
  }
});

describe("useCanvasRepaintSignal", () => {
  it("ignores hidden visibility changes and schedules visible ones", () => {
    mountHarness();

    // C1: a background transition occurs, expect no repaint work to start.
    hidden = true;
    dispatch(document, "visibilitychange");
    expect(requestAnimationFrameMock).not.toHaveBeenCalled();
    expect(signalValue()).toBe("0");

    // C2: the document becomes visible, expect the settle sequence to start.
    hidden = false;
    dispatch(document, "visibilitychange");
    expect(requestAnimationFrameMock).toHaveBeenCalledTimes(1);
    expect(pendingFrames.size).toBe(1);
  });

  it("handles visibility, pageshow, and focus after two animation frames", () => {
    mountHarness();

    // C1: each supported foreground event occurs, expect one signal per event.
    const events = [
      [document, "visibilitychange"],
      [window, "pageshow"],
      [window, "focus"],
    ] as const;
    events.forEach(([target, type], eventIndex) => {
      dispatch(target, type);
      expect(signalValue()).toBe(String(eventIndex));
      flushAnimationFrame();
      expect(signalValue()).toBe(String(eventIndex));
      flushAnimationFrame();
      expect(signalValue()).toBe(String(eventIndex + 1));
    });

    expect(signalValue()).toBe("3");
  });

  it("coalesces foreground event bursts into one delayed signal", () => {
    mountHarness();

    // C1: foreground events arrive together, expect one two-frame repaint.
    act(() => {
      document.dispatchEvent(new Event("visibilitychange"));
      window.dispatchEvent(new Event("pageshow"));
      window.dispatchEvent(new Event("focus"));
    });
    expect(requestAnimationFrameMock).toHaveBeenCalledTimes(1);
    expect(signalValue()).toBe("0");

    flushAnimationFrame();
    expect(requestAnimationFrameMock).toHaveBeenCalledTimes(2);
    expect(signalValue()).toBe("0");

    flushAnimationFrame();
    expect(signalValue()).toBe("1");
    expect(pendingFrames.size).toBe(0);
  });

  it("removes listeners and cancels either pending animation frame", () => {
    mountHarness();

    // C1: unmount before the first frame, expect it and all listeners removed.
    dispatch(window, "focus");
    expect(pendingFrames.has(1)).toBe(true);
    act(() => root?.unmount());
    root = null;
    expect(cancelAnimationFrameMock).toHaveBeenCalledWith(1);
    expect(pendingFrames.size).toBe(0);
    const callsAfterUnmount = requestAnimationFrameMock.mock.calls.length;
    dispatch(document, "visibilitychange");
    dispatch(window, "pageshow");
    dispatch(window, "focus");
    expect(requestAnimationFrameMock).toHaveBeenCalledTimes(callsAfterUnmount);

    // C2: unmount after the first frame, expect the second frame cancelled.
    mountHarness();
    dispatch(window, "pageshow");
    expect(pendingFrames.has(2)).toBe(true);
    flushAnimationFrame();
    expect(pendingFrames.has(3)).toBe(true);
    act(() => root?.unmount());
    root = null;
    expect(cancelAnimationFrameMock).toHaveBeenCalledWith(3);
    expect(pendingFrames.size).toBe(0);
  });
});
