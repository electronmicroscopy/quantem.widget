/** Coalesce pointer updates without postponing work until dragging stops. */
export function latestFrame(run: () => void) {
  let frame: number | null = null;
  return {
    schedule() {
      if (frame !== null) return;
      frame = requestAnimationFrame(() => { frame = null; run(); });
    },
    cancel() {
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
    },
  };
}

/** Run diagnostics after input settles; renewed input cancels queued GPU work. */
export function idleDiagnostic(delayMs = 200) {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let controller: AbortController | null = null;
  const cancel = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    controller?.abort();
    controller = null;
  };
  return {
    cancel,
    schedule(run: (signal: AbortSignal) => void) {
      cancel();
      const current = new AbortController();
      controller = current;
      timer = setTimeout(() => { timer = null; run(current.signal); }, delayMs);
    },
  };
}
