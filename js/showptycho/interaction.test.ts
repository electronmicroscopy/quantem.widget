import { afterEach, describe, expect, it, vi } from 'vitest';
import { idleDiagnostic, latestFrame } from './interaction';

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

it('waits for release, replaces queued diagnostics and aborts one when input resumes', () => {
  vi.useFakeTimers();
  const diagnostic = idleDiagnostic(200);
  const started: AbortSignal[] = [];
  diagnostic.schedule(signal => started.push(signal));
  vi.advanceTimersByTime(100);
  diagnostic.schedule(signal => started.push(signal));
  vi.advanceTimersByTime(199);
  expect(started).toHaveLength(0);
  vi.advanceTimersByTime(1);
  expect(started).toHaveLength(1);
  expect(started[0].aborted).toBe(false);
  diagnostic.cancel();
  expect(started[0].aborted).toBe(true);
  diagnostic.schedule(signal => started.push(signal));
  diagnostic.cancel();
  vi.runAllTimers();
  expect(started).toHaveLength(1);
});

describe('continuous tilt interaction', () => {
  it('updates during dragging, coalesces each frame and keeps the final value', () => {
    let frame: FrameRequestCallback | undefined;
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { frame = callback; return 1; });
    vi.stubGlobal('cancelAnimationFrame', () => { frame = undefined; });
    let tilt = 0;
    const shown: number[] = [];
    const scheduler = latestFrame(() => shown.push(tilt));
    for (let step = 0; step < 60; step++) {
      tilt = step; scheduler.schedule();
      tilt = step + 0.5; scheduler.schedule();
      const next = frame!; frame = undefined; next(step * 16.67);
    }
    expect(shown).toHaveLength(60);
    expect(shown[0]).toBe(0.5);
    expect(shown.at(-1)).toBe(59.5);
    scheduler.schedule(); scheduler.cancel();
    expect(frame).toBeUndefined();
  });
});
