/** Shared compact control styles of the widget toolbars. */
export const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;

/** One bordered row of controls. */
export const controlRow = {
  display: "flex",
  alignItems: "center",
  flexWrap: "wrap",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
  maxWidth: "100%",
  boxSizing: "border-box",
} as const;

/** Small text button (Reset, Export). */
export const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  textTransform: "none" as const,
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
};

export const switchStyles = {
  small: { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
  medium: { "& .MuiSwitch-thumb": { width: 14, height: 14 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
};

/** Menus that open above their control. */
export const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

export const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

/** The drag handle drawn on a panel in reorder mode. */
export const reorderBadgeSx = {
  position: "absolute",
  bottom: 6,
  left: "50%",
  transform: "translateX(-50%)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  width: 28,
  height: 20,
  borderRadius: 1,
  bgcolor: "rgba(0,0,0,0.35)",
  color: "rgba(255,255,255,0.9)",
  pointerEvents: "none",
  zIndex: 3,
} as const;

/** A panel's star button: hidden until the panel is hovered; starred panels stay visible. */
export function panelStarButtonSx(starred: boolean) {
  return {
    position: "absolute",
    top: 5,
    right: 5,
    width: 22,
    height: 22,
    p: 0,
    border: "none",
    bgcolor: "transparent",
    cursor: "pointer",
    fontSize: 18,
    lineHeight: "20px",
    textAlign: "center",
    color: starred ? "#ffc107" : "rgba(255,255,255,0.58)",
    textShadow: "0 0 3px rgba(0,0,0,0.8)",
    opacity: starred ? 1 : 0,
    pointerEvents: "auto",
    transform: starred ? "translateY(0)" : "translateY(-3px)",
    transition: "opacity 120ms ease, transform 120ms ease, background-color 120ms ease, color 120ms ease",
    userSelect: "none",
    zIndex: 3,
    "&:hover, &:focus-visible": {
      bgcolor: "rgba(0,0,0,0.22)",
      color: starred ? "#ffc107" : "rgba(255,255,255,0.9)",
    },
  } as const;
}

/** Shared compact sliders: circular handles, including their focus shadow. */
export const sliderStyles = {
  small: {
    py: 0,
    borderRadius: 0,
    "& .MuiSlider-thumb": { width: 10, height: 10, borderRadius: "50%" },
    "& .MuiSlider-thumb::before": { borderRadius: "50%" },
    "& .MuiSlider-rail": { height: 2, borderRadius: 0 },
    "& .MuiSlider-track": { height: 2, borderRadius: 0 },
    "& .MuiSlider-valueLabel": { borderRadius: 0 },
  },
};
