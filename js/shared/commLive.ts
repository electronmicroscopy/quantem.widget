import * as React from "react";

/**
 * Whether the widget model still has a kernel behind it. A notebook reopened
 * without a kernel restores the model with a dead comm, and Show4DSTEM must
 * then apply a detector edit locally instead of waiting for a kernel reply.
 */
export function modelCommIsLive(model: unknown): boolean {
  const candidate = model as {
    comm_live?: unknown;
    _comm_live?: unknown;
  } | null;
  if (typeof candidate?.comm_live === "boolean") return candidate.comm_live;
  if (typeof candidate?._comm_live === "boolean") return candidate._comm_live;
  // Non-Jupyter anywidget hosts do not promise either private property. In
  // those hosts the synchronized backend state remains the best evidence.
  return true;
}

/**
 * ``modelCommIsLive`` as React state. Jupyter emits ``comm:close`` before its
 * private liveness getter is necessarily updated, so a close or destroy event
 * sets the state false at once.
 */
export function useModelCommLive(model: unknown): boolean {
  const candidate = model as {
    on?: (event: string, callback: () => void) => void;
    off?: (event: string, callback: () => void) => void;
  } | null;
  const [live, setLive] = React.useState(() => modelCommIsLive(model));

  React.useEffect(() => {
    setLive(modelCommIsLive(model));
    if (!candidate?.on || !candidate?.off) return undefined;
    const update = () => setLive(modelCommIsLive(model));
    const close = () => setLive(false);
    candidate.on("comm_live_update", update);
    candidate.on("comm:close", close);
    candidate.on("destroy", close);
    return () => {
      candidate.off?.("comm_live_update", update);
      candidate.off?.("comm:close", close);
      candidate.off?.("destroy", close);
    };
  }, [candidate, model]);

  return live;
}
