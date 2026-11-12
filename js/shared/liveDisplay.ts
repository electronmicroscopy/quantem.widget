/**
 * Browser side of quantem.widget.live_display: the acknowledgment half of the
 * two-credit window Show4DSTEM.stream uses. A transfer is acknowledged after
 * React has had a frame to consume it (requestAnimationFrame, or 100 ms when a
 * hidden tab pauses animation frames); acknowledgments are cumulative.
 */
export type LiveModel = {
  get(name: string): unknown;
  send(message: Record<string, unknown>): void;
  on(event: string, callback: (...args: any[]) => void): void;
  off(event: string, callback: (...args: any[]) => void): void;
};

export function liveAcknowledger(send: (seq: number) => void) {
  let latest = 0;
  let acknowledged = 0;
  let frame: number | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  const cancel = () => {
    if (frame !== null && typeof cancelAnimationFrame === "function") cancelAnimationFrame(frame);
    if (timer !== null) clearTimeout(timer);
    frame = timer = null;
  };
  const flush = () => {
    cancel();
    if (latest > acknowledged) {
      acknowledged = latest;
      send(latest);
    }
  };
  return {
    /** Schedule an acknowledgment up to `seq` once the current update has rendered. */
    received(seq: unknown) {
      if (!Number.isSafeInteger(seq) || (seq as number) <= latest) return;
      latest = seq as number;
      cancel();
      if (typeof requestAnimationFrame === "function") frame = requestAnimationFrame(flush);
      timer = setTimeout(flush, 100);
    },
    dispose: cancel,
  };
}

/** One shared store per model, however many views of it are mounted. */
export function sharedPerModel<S extends { dispose(): void }>(
  stores: WeakMap<object, { store: S; users: number }>,
  model: object,
  create: () => S,
): { store: S; release(): void } {
  let entry = stores.get(model);
  if (!entry) {
    entry = { store: create(), users: 0 };
    stores.set(model, entry);
  }
  entry.users += 1;
  const held = entry;
  return {
    store: held.store,
    release() {
      held.users -= 1;
      if (held.users <= 0 && stores.get(model) === held) {
        held.store.dispose();
        stores.delete(model);
      }
    },
  };
}
