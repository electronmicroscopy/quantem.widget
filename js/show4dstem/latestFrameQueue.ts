/** Serialize point-pattern queries while retaining the latest requested position. */
export function createLatestFrameQueue(
  run: (isCurrent: () => boolean) => Promise<void>,
  onError: (error: unknown) => void,
) {
  let version = 0, generation = 0, busy = false, scheduled = false, closed = false;
  async function drain() {
    if (closed) return;
    busy = true;
    try {
      while (!closed) {
        const active = version;
        const source = generation;
        // Pending pointer positions must not starve visible progress. Queries are
        // serialized, so a completed frame cannot overtake a newer completion.
        const current = () => !closed && source === generation;
        try { await run(current); }
        catch (error) { if (current() && active === version) onError(error); }
        if (active === version) break;
      }
    } finally { busy = false; }
  }
  return {
    request() {
      if (closed) return;
      version++;
      if (busy || scheduled) return;
      scheduled = true;
      // Row, column, and selection notifications in one event share one query.
      queueMicrotask(() => { scheduled = false; void drain(); });
    },
    // Source/selection changes retire old results; pointer movement does not.
    invalidate() { generation++; },
    close() { closed = true; version++; },
  };
}
