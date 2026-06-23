/**
 * ShortcutRegistry — app-level registry of keybindings declared by
 * whichever page is currently mounted.  Lets the shared `?`-help
 * dialog render the bindings for the active page without the page
 * owning a dialog itself.
 *
 * The registry is deliberately simple: each call to `register` gets a
 * token it can later hand back to `unregister`.  `getAll()` returns
 * the flat list, which the help dialog groups by `Shortcut.group`.
 */
import React, { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";
import type { Shortcut } from "../hooks/useKeyboardShortcuts";

interface RegistryAPI {
  register: (bindings: Shortcut[]) => number;
  unregister: (token: number) => void;
  getAll: () => Shortcut[];
  version: number;
}

export const ShortcutRegistryContext = createContext<RegistryAPI | null>(null);
// Separate context so a version bump only re-renders consumers that actually
// need it (the `?`-help dialog). Consumers that just call register/unregister
// read the main context, whose api object has a STABLE identity. Without
// this split, every register() bump invalidated the api object, which
// re-ran every consumer's mount-effect → unregister → register → infinite
// loop ("Maximum update depth exceeded" on Trials popup open).
const ShortcutRegistryVersionContext = createContext<number>(0);

export function ShortcutRegistryProvider({ children }: { children: React.ReactNode }) {
  const bindingsRef = useRef<Map<number, Shortcut[]>>(new Map());
  const nextToken = useRef(1);
  // Version bump makes the help dialog re-read the registry when pages
  // mount/unmount. It's stored separately from the api object so the api
  // object's identity is stable across version bumps.
  const [version, setVersion] = useState(0);

  const register = useCallback((bindings: Shortcut[]) => {
    const token = nextToken.current++;
    bindingsRef.current.set(token, bindings);
    setVersion(v => v + 1);
    return token;
  }, []);

  const unregister = useCallback((token: number) => {
    bindingsRef.current.delete(token);
    setVersion(v => v + 1);
  }, []);

  const getAll = useCallback(() => {
    const out: Shortcut[] = [];
    for (const list of bindingsRef.current.values()) out.push(...list);
    return out;
  }, []);

  // Stable api — identity never changes. `version` is exposed via a
  // separate context. Effects that depend on `shortcutRegistry` as a whole
  // no longer re-run on every register/unregister.
  const api = useMemo<RegistryAPI>(
    () => ({ register, unregister, getAll, version: 0 }),
    [register, unregister, getAll],
  );

  return (
    <ShortcutRegistryContext.Provider value={api}>
      <ShortcutRegistryVersionContext.Provider value={version}>
        {children}
      </ShortcutRegistryVersionContext.Provider>
    </ShortcutRegistryContext.Provider>
  );
}

/** Subscribe to registry version bumps (use in the help dialog only). */
export function useShortcutRegistryVersion(): number {
  return useContext(ShortcutRegistryVersionContext);
}
