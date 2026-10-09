/** Keyboard shortcut gating shared by the canvas widgets. */

// Controls that consume arrows themselves. Buttons are left out so frame
// navigation keys still work while a toolbar button has focus.
export const WIDGET_TEXT_OR_VALUE_CONTROL_SELECTOR = [
  "input", "textarea", "select",
  "[contenteditable='true']", "[role='slider']",
  "[role='switch']", "[role='textbox']", "[role='combobox']", "[role='menuitem']",
  ".MuiSlider-root", ".MuiSelect-select",
].join(",");
// Anything that owns its own keyboard handling; widget shortcuts stay out of it.
const WIDGET_SHORTCUT_IGNORE_SELECTOR = `${WIDGET_TEXT_OR_VALUE_CONTROL_SELECTOR},button,[role='button']`;
const FRAME_NAVIGATION_KEYS = new Set(["ArrowLeft", "ArrowRight", "Home", "End"]);

export function shouldIgnoreWidgetShortcut(target: EventTarget | null, key = ""): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  if (FRAME_NAVIGATION_KEYS.has(key)) {
    return target.closest(WIDGET_TEXT_OR_VALUE_CONTROL_SELECTOR) !== null;
  }
  return target.closest(WIDGET_SHORTCUT_IGNORE_SELECTOR) !== null;
}
