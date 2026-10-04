/** Shared compact slider geometry for Show3D, Show3DSlices and ShowCIF. */
export const sliderStyles = {
  small: {
    py: 0,
    borderRadius: 0,
    "& .MuiSlider-thumb": { width: 10, height: 10, borderRadius: 0 },
    "& .MuiSlider-thumb::before": { borderRadius: 0 },
    "& .MuiSlider-rail": { height: 2, borderRadius: 0 },
    "& .MuiSlider-track": { height: 2, borderRadius: 0 },
    "& .MuiSlider-valueLabel": { borderRadius: 0 },
  },
};
