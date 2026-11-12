import * as React from "react";
import MuiSlider, { type SliderProps } from "@mui/material/Slider";

/**
 * Every widget slider sits under the pointer: no position easing on the thumb, track or label.
 *
 * MUI eases a slider thumb to its new position over 150 ms and only turns that off while MUI
 * itself runs the drag. Widgets that own their drags for speed (the Show3D frame slider) never
 * tell MUI a drag is active, so the thumb trailed the cursor by ~37 px while the image was already
 * on the new frame. Import this, never `@mui/material/Slider` (Slider.test.ts enforces it).
 */
export const instantSliderSx = {
  "& .MuiSlider-thumb, & .MuiSlider-track, & .MuiSlider-valueLabel": { transition: "none" },
} as const;

const Slider = React.forwardRef<HTMLSpanElement, SliderProps>(function Slider({ sx, ...props }, ref) {
  const own = sx ? (Array.isArray(sx) ? sx : [sx]) : [];
  return <MuiSlider ref={ref} {...props} sx={[instantSliderSx, ...own]} />;
});

export default Slider;
export type { SliderProps };

interface LiveNumberSliderProps {
  value: number;
  min: number;
  max: number;
  step: number;
  /** Every pointer move: paint from refs on the GPU, no widget state. */
  onLiveChange: (value: number) => void;
  /** Release: commit the value to widget state and model traits once. */
  onCommit: (value: number) => void;
  size?: "small" | "medium";
  valueLabelDisplay?: "auto" | "on" | "off";
  sx?: SliderProps["sx"];
  ariaLabel: string;
}

/**
 * A slider that keeps its own position while it is dragged, so a move re-renders only the slider
 * and the widget commits once on release.
 */
export const LiveNumberSlider = React.memo(function LiveNumberSlider({
  value, min, max, step, onLiveChange, onCommit, size = "small", valueLabelDisplay = "auto", sx, ariaLabel,
}: LiveNumberSliderProps) {
  const [liveValue, setLiveValue] = React.useState(value);
  React.useEffect(() => { setLiveValue(value); }, [value]);
  return (
    <Slider
      value={liveValue}
      min={min}
      max={max}
      step={step}
      onChange={(_, sliderValue) => {
        const next = sliderValue as number;
        setLiveValue(next);
        onLiveChange(next);
      }}
      onChangeCommitted={(_, sliderValue) => {
        const next = sliderValue as number;
        setLiveValue(next);
        onCommit(next);
      }}
      size={size}
      valueLabelDisplay={valueLabelDisplay}
      sx={sx}
      aria-label={ariaLabel}
    />
  );
});
