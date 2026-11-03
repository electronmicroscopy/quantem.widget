// Display colormaps, ranges and canvas rendering owned by quantem.widget.
// The CPU path (applyColormap) is the reference; the WebGPU engine must match it.
// The pieces live in luts, displayMath, displayShaders, colormapEngine,
// directRender, gpuRange and volumeSlice; widgets import them from here.
export type { Uint32ImageView } from "./borrowedImage";
export { COLORMAP_NAMES, COLORMAP_POINTS, COLORMAPS } from "./luts";
export { applyColormap, displayLog, displayNormalize, renderToOffscreen, renderToOffscreenReuse } from "./displayMath";
export { VOLUME_SLICE_MIN_ZOOM } from "./volumeSlice";
export { createGPUColormapEngine, GPUColormapEngine } from "./colormapEngine";
