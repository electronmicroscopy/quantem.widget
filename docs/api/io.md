# I/O

The `quantem.widget.io` module loads, saves, discovers, and inspects 4D-STEM and
image data. `load` (the primary 4D-STEM loader) has its own [page](load); this
page covers the rest of the public I/O surface.

```python
from quantem.widget.io import survey, read_image, get_metadata, bin, download
```

## Discover & inspect

```{autodoc2-object} quantem.widget.io.survey.survey
render_plugin = "myst"
```
```{autodoc2-object} quantem.widget.io.hdf5.discover_masters
render_plugin = "myst"
```
```{autodoc2-object} quantem.widget.io.hdf5.get_metadata
render_plugin = "myst"
```

## Images (2D / 3D)

```{autodoc2-object} quantem.widget.io.image.read_image
render_plugin = "myst"
```
```{autodoc2-object} quantem.widget.io.image.read_image_stack
render_plugin = "myst"
```

## Detector binning

```{autodoc2-object} quantem.widget.io.hdf5.bin
render_plugin = "myst"
```

## Hugging Face datasets

```{autodoc2-object} quantem.widget.io.hub.list_datasets
render_plugin = "myst"
```
```{autodoc2-object} quantem.widget.io.hub.download
render_plugin = "myst"
```

## Save

```{autodoc2-object} quantem.widget.io.save.save
render_plugin = "myst"
```
