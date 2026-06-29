# Performance

These notes capture interaction bugs that were easy to misread while building
the widgets. Keep this page short and practical: it should explain what went
wrong, how to recognize the pattern, and what to do instead.

## Mistake log: ShowEDS band center drag

Date: 2026-06-27

Symptom: the ShowEDS real-data widget could compute maps quickly, but dragging
the center of the energy band still felt slightly delayed. The debug HUD showed
acceptable map, spectrum, and draw times, so the lag was initially missed.

What was wrong:

- The center-drag preview used the same React state path as normal committed
  widget state.
- Every mousemove could trigger widget rerender work and spectrum canvas work,
  even though the user only needed the visible band rectangle to translate.
- The performance HUD measured compute and draw durations, not the full
  pointer-to-preview latency that the user feels.
- The bottom MUI range slider is a poor target for very narrow energy windows
  because the two thumbs overlap. The spectrum band body is the reliable center
  drag target for narrow windows.

Fix:

- During center drag, move lightweight DOM preview overlays with imperative
  `transform` and `width` updates.
- Store the pending band in refs while dragging.
- Feed the pending band into the throttled map scheduler during drag, because
  the element-map overlay is part of the expected live feedback.
- Commit `band_start`, `band_end`, and notebook state once on mouseup.
- Keep endpoint drags on the normal precise state path.

Rule for future high-FPS widget selectors:

- Separate preview interaction from committed state.
- Use refs and CSS transforms for per-pointer-frame visual feedback.
- Do not call the interaction real-time based only on compute timings. Drive it
  in the in-app browser and judge pointer-to-preview response.
- Avoid Python/kernel round trips and notebook model saves during drag.
- Recompute expensive data on a throttle or on commit unless the computation is
  genuinely required for the next visual frame.
- If the user expects a derived overlay, map, or spectrum to move while dragging,
  that derived view is part of the preview and must be updated live through the
  fastest available scheduler.
- Keep all redundant views of the same selection synchronized during preview:
  the plot band, bottom slider handles, text readout, and derived overlay should
  move as one interaction.

This applies to ShowEDS energy bands and ROI drags, Show4DSTEM detector masks,
Show2D contrast controls, and any future draggable selector that needs to feel
attached to the pointer.

## Mistake log: EDS is a query source, not a spreadsheet

Date: 2026-06-28

Symptom: a real Velox EDS EMD file opened quickly in vendor tools, but the
prototype treated the spectrum image like a dense ``(row, col, energy)`` table
that had to be expanded before interaction. That was the wrong model. The user
usually asks for a current energy window, an ROI spectrum, or a visible preview,
not every empty channel in every pixel.

What was wrong:

- Native EDS files should be treated as query backends. Keep the file/chunks as
  the source and ask for only the data needed by the current view.
- A ShowEDS data folder is a prefix-cache export format. It is useful for small
  or deliberately spatial-binned portable demos, but it is not the default model
  for native no-bin analysis.
- Calling ``cube.compute()`` before a targeted query or explicit spatial binning
  defeats lazy I/O.
- Browser widget state is for small embedded demos, not native EMD storage.

Rule for future EDS work:

- Never expand a native EDS file just to prove a widget can open it.
- Default no-bin EMD loading to native/lazy queries.
- Build prefix-cache data folders only for existing caches, explicit sidecar
  requests, or intentional binned sharing/export workflows.
- Guard prefix-cache and widget-state sizes before reading data.
- Use lazy chunked sum-binning only for explicit portable demos and exports.
- Treat spatial binning as count-preserving; make energy binning explicit.
- The best long-term path is a sparse/tiled frontend backend: energy-window
  queries produce maps, spatial-window queries produce spectra, and WebGPU does
  the visible accumulation/drawing without Python round trips during drag.

Current ShowEDS policy:

- Small embedded cubes stay browser/WebGPU backed.
- ``ShowEDS.from_emd(..., backend="auto")`` uses an existing data folder when
  present; otherwise exact no-bin EMD uses the native lazy query path.
- Portable real-data demos can use an explicitly spatial-binned data folder.
- Exact one-file HTML export is not available for native lazy EMD because the
  exported page has no local query backend; use binned single-file export or a
  data-folder export when sharing outside Jupyter.

Update from the 0016 Velox stream test:

- Velox EDS ``SpectrumStream`` data is sparse event data. The logical dense
  shape can be tens of GB, but the actual useful stream can be a few hundred MB.
- Do not materialize zeros. Index the stream directly by channel and by pixel.
- A sparse stream data folder for the 2048 x 2048 x 4096 0016 file stores about
  26.9 million events in about 186 MB and keeps the full field of view exact.
- Full-field interaction should be validated with no crop and no binning before
  offering binned/export presets.
- If Jupyter ignores HTTP ``Range`` and returns ``200 OK`` with a whole file,
  slice the returned buffer when it contains the requested byte window instead
  of failing the sidecar worker.
