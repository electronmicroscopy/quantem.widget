# ShowBragg

Drive `quantem.diffraction.BraggVectors` Bragg-disk workflow. Probe the correlation map at a scan
position, tune disk detection on sampled subset, and from there accumulate a Bragg vector
map with fitting the lattice at every scan position.

```python
from quantem.widget import ShowBragg

widget = ShowBragg(dataset)          # Dataset4dstem, or an existing BraggVectors
```

Template generation and correlation probing are updated interactively as their controls change. Operations that require processing the full scan, like Bragg-disk detection and lattice fitting, are run explicitly using buttons or Python methods.

```python
widget.detect()                      # full detect_disks
widget.fit()                         # fit_lattice
widget.basis                         # (origin, g1, g2)
strain = widget.strain_map()         # Construct StrainMap 
```

Scientific results remain owned by the underlying BraggVectors workflow. The complete wrapped BraggVectors object is available through widget.bragg for functionality that is not exposed directly in the interface.

For template_radius and max_peak_shift, a value of 0.0 means that the corresponding value should be determined automatically by quantem.

Strain visualization is handled outside ShowBragg, the widget returns a StrainMap, which can then be visualized using the standard StrainMap.plot_strain interface.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.showbragg.ShowBragg
   :members:
   :show-inheritance:
```

## Display configuration

`ShowBragg` follows the shared viewer presets described in the
[UI Guide](../developer/ui-guide): 

Each UI preset configures show_title, show_controls, and controls_collapsed. 

The controls can also be managed programmatically:

* widget.collapse_controls()
* widget.expand_controls()
* widget.toggle_controls()

When the controls are collapsed, parameter columns and execution buttons are hidden while the primary visualization remains visible.

## Interactive controls

| Control | Trait | Behavior |
|---|---|---|
| Template source selector | `template_source` | Rebuilds the template from a synthetic disk, the data mean, or a supplied probe |
| Template radius / edge | `template_radius`, `template_edge` | Re-renders `template_png` and the correlation map |
| Subtract mean toggle | `template_subtract_mean` | Rebuilds the template as a zero-sum band-pass kernel |
| Scan row / col fields | `probe_position` | Re-renders the diffraction pattern and its correlation map |
| Six detection fields | `min_abs_intensity`, `min_spacing`, `edge_boundary`, `subpixel`, `upsample_factor`, `max_num_peaks` | Stored for the next preview or full run |
| Preview on grid button | `preview_peaks`, `detection_state` | Detects on `preview_grid` x `preview_grid` sampled positions and marks them on the probe panel |
| Run full detection button | `detection_state`, `bvm_png`, `candidates` | Detects at every scan position, then accumulates the Bragg vector map |
| Recompute map button | `bvm_sampling`, `bvm_png` | Re-accumulates the Bragg vector map at the given sampling |
| Candidate fields | `num_candidates`, `candidate_min_spacing`, `candidate_min_abs_intensity` | Re-derives the numbered candidate set live |
| Click a numbered candidate | `origin_index`, `g1_index`, `g2_index` | Assigns that candidate to the active basis role |
| Drag a basis marker | `origin_rc`, `g1_rc`, `g2_rc` | Places the vector freely and clears that role's candidate index |
| Reset to automatic button | all six basis traits | Lets quantem pick the basis again |
| Run fit button | `fit_state`, `mask_weight_png`, `fit_error_png` | Fits the lattice at every position and shows the two diagnostics |
| Controls / Hide button | `controls_collapsed` | Hides the parameter columns and run buttons, keeping the images |
| Pan (drag) / zoom (wheel) | view transform | Any image panel translates / zooms about the cursor |
| Double-click | view transform | Resets that panel's zoom and pan |
