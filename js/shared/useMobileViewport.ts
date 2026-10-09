import * as React from "react";

const COARSE_POINTER = "(pointer: coarse)";
const NARROW_VIEWPORT = "(max-width: 768px)";

/** Whether the viewport is touch-first or narrow; false where matchMedia is missing (jsdom in tests). */
function isMobileViewport(): boolean {
  if (typeof window.matchMedia !== "function") return false;
  return window.matchMedia(COARSE_POINTER).matches || window.matchMedia(NARROW_VIEWPORT).matches;
}

/** True on coarse-pointer or narrow viewports, so widgets can swap hover-only
 *  affordances for touch-sized ones. Tracks media-query changes and resizes. */
export function useMobileViewport(): boolean {
  const [isMobile, setIsMobile] = React.useState(isMobileViewport);
  React.useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const queries = [window.matchMedia(COARSE_POINTER), window.matchMedia(NARROW_VIEWPORT)];
    const update = () => setIsMobile(isMobileViewport());
    update();
    for (const query of queries) query.addEventListener("change", update);
    window.addEventListener("resize", update);
    return () => {
      for (const query of queries) query.removeEventListener("change", update);
      window.removeEventListener("resize", update);
    };
  }, []);
  return isMobile;
}
