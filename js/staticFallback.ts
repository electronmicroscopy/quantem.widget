import * as React from "react";

/**
 * Hide the static-PNG sibling output while the live widget is mounted.
 *
 * Python's StaticFallbackMixin publishes a fallback <img> as a SEPARATE
 * display_data output because JupyterLab never falls back to an in-bundle
 * image/png when the widget renderer claims the output ("model not found"
 * on cold reopen). Live, both would show; the img carries the widget's
 * model id, so hide exactly ours. On a cold reopen this code never runs
 * (no kernel, no widget mount) and the fallback stays visible.
 *
 * Pass the widget's root-element ref so the primary scope is STRUCTURAL:
 * any fallback img inside this widget's own cell belongs to this cell's
 * display and must be hidden while live. The model-id match is only the
 * fallback for when the root ref hasn't attached yet (model_id can be
 * undefined on some anywidget versions).
 */
export function useHideStaticFallback(
  model: unknown,
  rootRef: React.RefObject<HTMLElement | null>,
  shouldHide = true,
): void {
  React.useEffect(() => {
    if (!shouldHide) return;
    if (typeof document === "undefined") return;
    const modelId = (model as { model_id?: string } | undefined)?.model_id;
    let cancelled = false;
    const hideFallback = () => {
      if (cancelled) return;
      const cell = rootRef.current?.closest(".jp-Cell") ?? null;
      const scope: ParentNode = cell ?? document;
      scope.querySelectorAll("img.quantem-static-fallback").forEach((img) => {
        if (!cell) {
          const imgId = img.getAttribute("data-quantem-model-id");
          if (!modelId || imgId !== modelId) return;
        }
        const output = img.closest(".jp-OutputArea-child") ?? img;
        (output as HTMLElement).style.display = "none";
      });
    };
    hideFallback();
    // The sibling output is filled DEFERRED (post_execute update_display_data)
    // and Lab may recreate its DOM node on the update, so fixed-delay retries
    // can miss it and the PNG would show under the live widget. Observe the
    // document for the widget's lifetime and hide the marker whenever it
    // (re)appears; disconnected on unmount, so a cold reopen (no widget JS)
    // still shows the fallback.
    const observer = new MutationObserver(hideFallback);
    observer.observe(document.body, { childList: true, subtree: true });
    return () => {
      cancelled = true;
      observer.disconnect();
    };
  }, [model, shouldHide]);
}
