import * as React from "react";

/**
 * A value outside React state. Per-frame display values (histogram bins, statistics, the
 * painted frame) live here, so a new frame re-renders only the components that show them
 * (useStoreValue, Subscribe) instead of the whole widget.
 */
export class ValueStore<T> {
  private value: T;
  private listeners = new Set<() => void>();

  constructor(initial: T) {
    this.value = initial;
  }

  get = (): T => this.value;

  set(value: T): void {
    if (Object.is(value, this.value)) return;
    this.value = value;
    for (const listener of this.listeners) listener();
  }

  /** Merge `patch` into an object value; unchanged fields notify nobody. */
  update(patch: Partial<T>): void {
    const current = this.value as Record<string, unknown>;
    if (Object.entries(patch).every(([key, value]) => Object.is(current[key], value))) return;
    this.set({ ...this.value, ...patch });
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };
}

/** The store's value; the component re-renders when it changes. */
export function useStoreValue<T>(store: ValueStore<T>): T {
  return React.useSyncExternalStore(store.subscribe, store.get);
}

/** Render `children(value)` as its own component, so a store change re-renders only this part. */
export function Subscribe<T>({ store, children }: { store: ValueStore<T>; children: (value: T) => React.ReactNode }) {
  return <>{children(useStoreValue(store))}</>;
}

/**
 * useState whose setter skips a value equal to the last one it set. Setting an unchanged value can
 * still run the whole component once before React bails out, which per frame is a full widget
 * render. The setter takes values only, not updater functions.
 */
export function useChangedState<T>(initial: T): [T, (value: T) => void] {
  const [value, setValue] = React.useState(initial);
  const last = React.useRef(initial);
  const set = React.useCallback((next: T) => {
    if (Object.is(next, last.current)) return;
    last.current = next;
    setValue(next);
  }, []);
  return [value, set];
}
