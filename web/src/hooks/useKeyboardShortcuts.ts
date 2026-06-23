import { useContext, useEffect } from "react";
import { ShortcutRegistryContext } from "../components/ShortcutRegistry";

/**
 * A single keyboard-shortcut binding.
 *
 * Matching rules for Shift:
 *   - `shift: undefined` (default) is FORGIVING: the binding fires regardless
 *     of the user's Shift state. Most common case - e.g. a binding for "h"
 *     fires on both `h` and `Shift+h`. This matches Gmail / Linear / most
 *     keyboard-driven web apps and avoids the "capital letter silently
 *     breaks my shortcut" footgun.
 *   - `shift: true` requires Shift to be held.
 *   - `shift: false` requires Shift to NOT be held.
 *
 * The `key` is matched case-insensitively via `toLowerCase()` so "h" and "H"
 * are equivalent.
 *
 * Ctrl / Meta / Alt are matched strictly (undefined = must NOT be held). A
 * binding MUST opt in to modifier keys to fire with them held - this prevents
 * bare-letter bindings from stealing browser shortcuts like Ctrl+R.
 */
export interface Shortcut {
  key: string;
  shift?: boolean;
  ctrl?: boolean;
  meta?: boolean;
  alt?: boolean;
  handler: (e: KeyboardEvent) => void;
  /** If true, skip this binding when target is an <input> or <textarea>. Default true. */
  ignoreInInputs?: boolean;
  /** Platform-aware modifier (Cmd on Mac, Ctrl elsewhere). Used by ShortcutHelp display only. */
  mod?: boolean;
  /** Group label for ShortcutHelp dialog display. */
  group?: string;
  /** Human-readable description for ShortcutHelp dialog display. */
  label?: string;
}

function matches(e: KeyboardEvent, s: Shortcut): boolean {
  // Case-insensitive key match. Letter keys implicitly accept both cases.
  if (e.key.toLowerCase() !== s.key.toLowerCase()) return false;
  // Shift is forgiving unless explicitly constrained.
  if (s.shift !== undefined && e.shiftKey !== s.shift) return false;
  // Ctrl / Meta / Alt must be explicitly opted into.
  if ((s.ctrl ?? false) !== e.ctrlKey) return false;
  if ((s.meta ?? false) !== e.metaKey) return false;
  if ((s.alt ?? false) !== e.altKey) return false;
  return true;
}

function isInEditableTarget(e: KeyboardEvent): boolean {
  const target = e.target as HTMLElement | null;
  if (!target) return false;
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if ((target as HTMLElement).isContentEditable) return true;
  return false;
}

/**
 * Register a list of keyboard shortcuts on window keydown for the lifetime
 * of the caller's component. Pass `enabled=false` to temporarily disable all
 * bindings without unmounting.
 */
export function useKeyboardShortcuts(shortcuts: Shortcut[], enabled: boolean = true): void {
  useEffect(() => {
    if (!enabled) return;
    const handler = (e: KeyboardEvent) => {
      for (const s of shortcuts) {
        if ((s.ignoreInInputs ?? true) && isInEditableTarget(e)) continue;
        if (matches(e, s)) {
          s.handler(e);
          return;
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [shortcuts, enabled]);
}

/**
 * Register shortcuts AND publish them to the ShortcutRegistry so the global
 * `?`-help dialog can show them. Use this instead of `useKeyboardShortcuts`
 * when you want the bindings to appear in the help menu (#113).
 */
export function useRegisterShortcuts(shortcuts: Shortcut[], enabled: boolean = true): void {
  useKeyboardShortcuts(shortcuts, enabled);
  const reg = useContext(ShortcutRegistryContext);
  // Destructure stable functions rather than depending on the full reg object.
  // reg.register/unregister are useCallback([]) in the provider — stable forever.
  // Including reg itself causes a loop: register() bumps version → new reg object
  // → effect re-runs → register() again → "Maximum update depth exceeded".
  const register = reg?.register;
  const unregister = reg?.unregister;
  useEffect(() => {
    if (!enabled || !register || !unregister) return;
    const token = register(shortcuts);
    return () => unregister(token);
  }, [shortcuts, enabled, register, unregister]);
}
