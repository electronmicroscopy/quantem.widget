import * as React from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";


export type FolderWatchState =
  | "hidden"
  | "watching"
  | "updating"
  | "waiting"
  | "error"
  | "stopped"
  | "not_watching";

type FolderWatchPresentation = {
  state: Exclude<FolderWatchState, "hidden">;
  label: string;
  color: string;
};

type VisibleFolderWatchState = Exclude<FolderWatchState, "hidden">;

const PRESENTATIONS: Record<VisibleFolderWatchState, Omit<FolderWatchPresentation, "state">> = {
  watching: { label: "Watching", color: "#2e7d32" },
  updating: { label: "Updating", color: "#1976d2" },
  waiting: { label: "Waiting for file completion", color: "#ed6c02" },
  error: { label: "Watch error", color: "#d32f2f" },
  stopped: { label: "Stopped", color: "#757575" },
  not_watching: { label: "Not watching", color: "#757575" },
};

export function folderWatchPresentation(state: string): FolderWatchPresentation | null {
  if (state === "hidden") return null;
  const canonicalState = Object.prototype.hasOwnProperty.call(PRESENTATIONS, state)
    ? state as VisibleFolderWatchState
    : "error";
  return { state: canonicalState, ...PRESENTATIONS[canonicalState] };
}

export function folderWatchAnnouncement(state: string): {
  role: "alert" | "status";
  live: "assertive" | "polite";
} | null {
  const presentation = folderWatchPresentation(state);
  if (!presentation) return null;
  if (presentation.state === "error") {
    return { role: "alert", live: "assertive" };
  }
  if (
    presentation.state === "waiting"
    || presentation.state === "stopped"
    || presentation.state === "not_watching"
  ) {
    return { role: "status", live: "polite" };
  }
  // Watching <-> Updating is the normal polling cadence. Keep both states
  // visible without continuously queueing screen-reader announcements.
  return null;
}

export function folderWatchModelIsLive(model: unknown): boolean {
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

export function useFolderWatchModelLive(model: unknown): boolean {
  const candidate = model as {
    on?: (event: string, callback: () => void) => void;
    off?: (event: string, callback: () => void) => void;
  } | null;
  const [live, setLive] = React.useState(() => folderWatchModelIsLive(model));

  React.useEffect(() => {
    setLive(folderWatchModelIsLive(model));
    if (!candidate?.on || !candidate?.off) return undefined;
    const update = () => setLive(folderWatchModelIsLive(model));
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

export function effectiveFolderWatchState(state: string, live: boolean): string {
  if (
    !live
    && state !== "hidden"
    && state !== "stopped"
    && state !== "not_watching"
  ) {
    return "not_watching";
  }
  return state;
}

export function FolderWatchBadge({
  state,
  detail,
  live = true,
}: {
  state: string;
  detail?: string;
  live?: boolean;
}) {
  const effectiveState = effectiveFolderWatchState(state, live);
  const presentation = folderWatchPresentation(effectiveState);
  if (!presentation) return null;
  const announcement = folderWatchAnnouncement(presentation.state);
  const normalizedDetail = live ? detail?.trim() ?? "" : "";
  const visibleDetail = Boolean(normalizedDetail)
    && presentation.state !== "watching"
    && presentation.state !== "stopped";
  const accessibleLabel = visibleDetail
    ? `${presentation.label}: ${normalizedDetail}`
    : presentation.label;
  return (
    <Box
      component="span"
      role={announcement?.role}
      aria-live={announcement?.live}
      aria-atomic={announcement ? "true" : undefined}
      aria-label={accessibleLabel}
      data-folder-watch-state={presentation.state}
      title={accessibleLabel}
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: "5px",
        minHeight: 18,
        maxWidth: "100%",
        overflow: "hidden",
        mb: "4px",
        px: "6px",
        py: "1px",
        border: "1px solid currentColor",
        borderRadius: "9px",
        bgcolor: "transparent",
        color: "inherit",
        boxSizing: "border-box",
        verticalAlign: "middle",
      }}
    >
      <Box
        component="span"
        aria-hidden="true"
        data-folder-watch-dot="true"
        sx={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          bgcolor: presentation.color,
          flex: "0 0 auto",
        }}
      />
      <Typography
        component="span"
        sx={{
          minWidth: 0,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          fontSize: 10,
          lineHeight: "14px",
          fontWeight: 600,
        }}
      >
        {presentation.label}
      </Typography>
      <Typography
        component="span"
        aria-hidden={!visibleDetail}
        data-folder-watch-detail="true"
        data-folder-watch-detail-visible={visibleDetail ? "true" : "false"}
        sx={{
          flex: visibleDetail ? "1 1 auto" : "0 0 0",
          minWidth: 0,
          maxWidth: visibleDetail ? 420 : 0,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          visibility: visibleDetail ? "visible" : "hidden",
          fontSize: 10,
          lineHeight: "14px",
          color: "inherit",
          opacity: visibleDetail ? 0.9 : 0,
        }}
      >
        {visibleDetail ? `— ${normalizedDetail}` : ""}
      </Typography>
    </Box>
  );
}
