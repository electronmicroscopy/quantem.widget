import * as React from "react";
import { IconButton } from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";

/** Shared Show3D playback button, including keyboard and accessible labels. */
export function PlayPauseButton({
  playing,
  onToggle,
  color,
  disabled = false,
  label,
}: {
  playing: boolean;
  onToggle: () => void;
  color: string;
  disabled?: boolean;
  label?: string;
}) {
  return (
    <IconButton
      size="small"
      onClick={onToggle}
      disabled={disabled}
      sx={{ color, p: 0.25, borderRadius: 0 }}
      aria-label={label ?? (playing ? "Pause playback" : "Play")}
      title={playing ? "Pause (Space)" : "Play (Space)"}
    >
      {playing ? (
        <PauseIcon sx={{ fontSize: 18 }} />
      ) : (
        <PlayArrowIcon sx={{ fontSize: 18 }} />
      )}
    </IconButton>
  );
}
