import * as React from "react";

/**
 * Return a signal that changes after the page comes back to the foreground.
 *
 * Browsers may discard a canvas backing store while a tab is backgrounded.
 * Consumers can include this signal in their draw-effect dependencies to
 * repaint from their existing data or cache after the page is visible again.
 * Two animation frames let foreground compositing and layout settle first.
 */
export function useCanvasRepaintSignal(): number {
  const [signal, setSignal] = React.useState(0);
  const firstFrameRef = React.useRef<number | null>(null);
  const secondFrameRef = React.useRef<number | null>(null);

  React.useEffect(() => {
    if (typeof document === "undefined" || typeof window === "undefined") {
      return;
    }

    const scheduleRepaint = () => {
      if (document.hidden) return;
      if (firstFrameRef.current !== null || secondFrameRef.current !== null) {
        return;
      }

      firstFrameRef.current = window.requestAnimationFrame(() => {
        firstFrameRef.current = null;
        secondFrameRef.current = window.requestAnimationFrame(() => {
          secondFrameRef.current = null;
          setSignal((value) => value + 1);
        });
      });
    };
    const handleVisibilityChange = () => {
      if (!document.hidden) scheduleRepaint();
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    window.addEventListener("pageshow", scheduleRepaint);
    window.addEventListener("focus", scheduleRepaint);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("pageshow", scheduleRepaint);
      window.removeEventListener("focus", scheduleRepaint);
      if (firstFrameRef.current !== null) {
        window.cancelAnimationFrame(firstFrameRef.current);
        firstFrameRef.current = null;
      }
      if (secondFrameRef.current !== null) {
        window.cancelAnimationFrame(secondFrameRef.current);
        secondFrameRef.current = null;
      }
    };
  }, []);

  return signal;
}
