import * as React from "react";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";

/**
 * Small help glyph with a themed tooltip. `toggleOnClick` keeps the tooltip
 * open after a click or Enter/Space so touch and keyboard users can read it;
 * hover-only is the default because most widgets place it next to the title.
 */
export function InfoTooltip({
  text,
  theme = "dark",
  icon = "ⓘ",
  maxWidth = 280,
  toggleOnClick = false,
}: {
  text: React.ReactNode;
  theme?: "light" | "dark";
  icon?: React.ReactNode;
  maxWidth?: number;
  toggleOnClick?: boolean;
}) {
  const isDark = theme === "dark";
  const [open, setOpen] = React.useState(false);
  const content = typeof text === "string"
    ? <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    : text;
  const openProps = toggleOnClick
    ? { open, onOpen: () => setOpen(true), onClose: () => setOpen(false) }
    : {};
  const toggleProps = toggleOnClick
    ? {
      role: "button",
      tabIndex: 0,
      "aria-label": "Show controls help",
      "aria-expanded": open ? "true" as const : "false" as const,
      onClick: (event: React.MouseEvent) => {
        event.stopPropagation();
        setOpen((value) => !value);
      },
      onKeyDown: (event: React.KeyboardEvent) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          event.stopPropagation();
          setOpen((value) => !value);
        }
      },
    }
    : {};
  return (
    <Tooltip
      title={content}
      {...openProps}
      arrow placement="bottom"
      componentsProps={{
        tooltip: { sx: { bgcolor: isDark ? "#333" : "#fff", color: isDark ? "#ddd" : "#333", border: `1px solid ${isDark ? "#555" : "#ccc"}`, maxWidth, p: 1 } },
        arrow: { sx: { color: isDark ? "#333" : "#fff", "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` } } },
      }}
    >
      <Typography
        component="span"
        {...toggleProps}
        sx={{
          fontSize: 12, color: isDark ? "#888" : "#666", cursor: "help", ml: 0.5,
          "&:hover": { color: isDark ? "#aaa" : "#444" },
          ...(toggleOnClick ? { "&:focus-visible": { outline: `1px solid ${isDark ? "#aaa" : "#444"}`, outlineOffset: 1 } } : {}),
        }}
      >
        {icon}
      </Typography>
    </Tooltip>
  );
}
