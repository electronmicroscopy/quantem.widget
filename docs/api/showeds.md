# ShowEDS

Public import:

```python
from quantem.widget import ShowEDS, load_eds
```

`ShowEDS` explores EDS/EELS spectrum images with shape
`(row, col, energy)`. The widget links a real-space map and a spectrum: dragging
the energy band updates the element map, and dragging the rectangular real-space
ROI updates the summed spectrum.

## Viewer UI

`ShowEDS` supports the shared `ui_mode`, `show_title`, `show_controls`,
`controls_collapsed`, and `show_scale_bar` names. The saved-state trait remains
`scale_bar_visible` for compatibility. See [Viewer UI controls](viewer-ui).

`ShowEDS` is still experimental. The merged baseline supports synthetic/small
cubes and the documented data-folder workflow. Direct native EMD sparse-stream
loading is under active real-data testing and should not be treated as finalized
release behavior yet.

Canonical forms:

```python
# Native EDS/EELS file: parse metadata and keep exact no-bin EMD data lazy.
# The returned SpectrumImage uses (row, col, energy) and feeds ShowEDS directly.
eds = load_eds("0031-CaSIO3_...EDS_HAADF_Diffraction_Nano.emd")
w = ShowEDS(
    eds,
    energy=8.04,
    width=0.24,
    pixel_size=0.138,
    pixel_unit="nm",
    element_label="Cu K",
    candidate_elements=["O", "Si", "Ca", "Cu", "Au"],
)

# Small or already-reduced cube: embeds the exact uint16/uint32/float32 cube in
# widget state, so notebook save/reopen and HTML export are self-contained.
w = ShowEDS(cube, energy_keV, base_image=haadf)

# Large EMD as a reusable data folder. The notebook stores startup state and a
# data-folder URL, not the multi-GB cube.
w = ShowEDS.from_emd(
    "0031-CaSIO3_...EDS_HAADF_Diffraction_Nano.emd",
    sidecar_dir="sidecars/eds_real_0031",
    energy=8.04,
    width=0.24,
    pixel_size=0.138,
    pixel_unit="nm",
    element_label="Cu K",
    candidate_elements=["O", "Si", "Ca", "Cu", "Au"],
    selected_elements=["Cu", "Au"],
)

# Shareable HTML. Small cubes can use single HTML with exact data. Large folder-backed
# cubes can use exact folder HTML, or count-preserving downsampled single HTML.
w.export_html("showeds_exact.html")
w.export_html("showeds_exact_folder.html", mode="folder")
w.export_html("showeds_downsample4.html", downsample=4)
```

For the real EDS 0031 dataset, the folder path keeps the live notebook small
because the saved `.ipynb` contains widget state plus startup arrays, not the
entire raw cube. The exact folder HTML mode is similarly small but must be
served next to its companion data folder. `downsample=2` or `downsample=4`
embeds sum-downsampled data into one HTML file, which is the better public
tutorial and Hugging Face demo path.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.showeds.ShowEDS
   :members:
   :show-inheritance:

.. autoclass:: quantem.widget.showeds.SpectrumImage
   :members:

.. autofunction:: quantem.widget.showeds.load_eds

.. autofunction:: quantem.widget.showeds.bin_spectrum_image

.. autofunction:: quantem.widget.showeds.prepare_spectrum_image_sidecar

.. autofunction:: quantem.widget.showeds.load_emd_spectrum_image

.. autofunction:: quantem.widget.showeds.eds_line_hints
```

## Interactive controls

With a small cube in single mode with exact data, all interaction runs in the browser. With a
folder-backed EMD widget, startup state is embedded and the browser fetches
exact prefix arrays from the data-folder URL, so dragging the band or ROI does not
need a Python round trip.

| Control | Trait | Expected effect |
|---|---|---|
| Energy band on spectrum | `band_start`, `band_end` | Real-space element map recomputes from the selected energy window |
| Band center drag | `band_start`, `band_end` | The selected energy window translates without changing width |
| Elements periodic table | `selected_elements`, `auto_identify`, `line_hints` | Select candidate elements, emphasize their line markers, and snap the band to characteristic X-ray lines |
| Detect button | `element_candidates`, `detect_status` | Runs `detect_elements()` in the kernel on the full sum spectrum and fills the Auto-ID candidate chips (advisory, click a chip to select) |
| Real-space ROI | `roi_row`, `roi_col`, `roi_height`, `roi_width`, `roi_shape` | Rectangular or circular ROI recomputes the summed spectrum from the selected real-space region |
| Spectrum log toggle | `log_spectrum` | Spectrum switches between linear and log display |
| Scroll zoom / pan | `map_zoom`, `map_view_row`, `map_view_col`, `spectrum_view_start`, `spectrum_view_end` | Real-space and spectrum views zoom around the cursor and reopen at the saved view |
| Real-space scale bar | `pixel_size`, `pixel_unit`, `scale_bar_visible` | Shows/hides a Show2D-style scale bar and zoom indicator without changing count data |
| Overlay slider | `overlay_opacity` | Element-map overlay fades against the base image |
| Auto / contrast range | `map_vmin_pct`, `map_vmax_pct` | Element-map display range updates without changing counts |
| Panel resize handles | `panel_width_px`, `spectrum_width_px`, `spectrum_height_px` | Real-space and spectrum panels resize independently |
| Export menu | `export_request`, `export_payload`, `export_status` | Generates exact folder or downsampled single HTML |
| Reset | multiple state traits | Restores default band, ROI, contrast, and panel sizing |

## State and sharing

Saving a notebook stores the interactive widget state: current band, ROI, scroll
zoom, scale-bar calibration, log toggle, overlay opacity, contrast limits, panel sizes, and export metadata. For
small cubes constructed with `save_state=True`, the saved notebook also contains the exact cube bytes. For
large `from_emd(...)` widgets, the saved notebook contains only startup arrays
and the data folder URL, so reopening stays lightweight and the data folder must
remain available.

For collaborators in JupyterLab, share the full-state notebook plus any data
folder it references. For GitHub's native notebook preview, use the GitHub
snapshot workflow instead of expecting live widgets. For a web page or tutorial,
export HTML: exact folder HTML for full-resolution internal use, or downsampled
single HTML for compact public demos.
