# Changelog

One line per release candidate: the main user-facing thing that changed. Newest
first. Add an entry under **Unreleased** as you land a change; move it under the
new `rcN` heading when that rc is published to TestPyPI.

## Unreleased

## rc23 - 2026-06-24

- New `quantem` command line: `quantem show <path>` (auto-detect) plus `show2d` / `show3d` / `show4dstem`. Render an image, a folder of images, or 4D-STEM master(s) straight to a standalone HTML (images) or a live notebook (4D-STEM, or `--html`); saves to `~/Downloads` and opens automatically. Runs on CUDA / Apple Silicon (MPS) / CPU.
- `quantem show4dstem A B ...` (or a folder) stacks several masters into one 5D viewer with a Dataset slider to flip between scans; `--combined --html` writes that as one offline-WebGPU file.
- Widget HTML export is now a size-labeled dropdown showing the resulting detector resolution (e.g. "uint8 96x96"); PNG / PDF / ZIP figure export removed (HTML only, plus Copy).
- Show4DSTEM HTML export detector binning is now MEAN, not sum, so the bright field no longer clips at uint8 on real-count detectors; binning happens at load so the full stack never has to fit in memory.
- The export button is hidden in the already-exported HTML, and the exported 4D viewer paints its virtual image on mount (no longer blank until you nudge a control).
- Show3D temporal averaging (avg_window) moved to a GPU compute shader - avg=15 scrubs as fast as avg=1 (was a CPU per-pixel loop on the UI thread).
- `io.read_image` loads any common format (tif/png/jpg/bmp/npy/dm3/dm4 + non-Velox emd) into a `Dataset2d`.
- `io.read_images(folder)` loads a whole folder of mixed-format images into a `list[Dataset2d]`.

## rc22

- Scale bar auto-picks nm/Å/pm so labels are clean integers (no more 0.5 nm decimals).

## rc16

- Show4DSTEM browser folder-GUI workflow fix.
