# Show3D

A 3D volume scrubbed slice by slice, with playback and an interactive-HTML
export. See the [Show3D tutorial](../tutorials/show3d).

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.show3d.Show3D
   :members:
   :show-inheritance:
```

## Interactive controls

| Control | Trait | Expected effect |
|---|---|---|
| Slice slider | `slice_idx` | Canvas shows that depth slice |
| Arrow keys | `slice_idx` | Step one slice per press |
| Play / pause | `playing` | Auto-advances slices at `fps` |
| Reverse | `reverse` | Playback direction flips |
| Boomerang | `boomerang` | Ping-pongs at the ends instead of looping |
| FPS field | `fps` | Playback rate changes |
| Loop range | `loop_start`, `loop_end` | Playback confined to the sub-range |
| Colormap dropdown | `cmap` | Canvas recolors |
| Export button | `export_request`, `export_status` | Writes a standalone HTML viewer |
| Panel layout (multi-panel) | `n_panels`, `link_panels`, `max_cols` | Panels arrange; linked scrub moves all |

```{note}
`export_html(quantized=True)` writes the smaller uint8 pack; the default writes
exact float32. See the [widget export tutorial](../tutorials/widget_export).
```
