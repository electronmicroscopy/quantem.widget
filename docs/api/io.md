# I/O

The `quantem.widget.io` module loads, saves, discovers, and inspects 4D-STEM and
image data. `load` (the primary 4D-STEM loader) has its own [page](load); this
page covers the rest of the public I/O surface.

```python
from quantem.widget.io import survey, read_image, get_metadata, bin, download
```

## Discover & inspect

```{eval-rst}
.. autofunction:: quantem.widget.io.survey.survey
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hdf5.discover_masters
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hdf5.get_metadata
```

## Images (2D / 3D)

```{eval-rst}
.. autofunction:: quantem.widget.io.image.read_image
```
```{eval-rst}
.. autofunction:: quantem.widget.io.image.read_image_stack
```

## Detector binning

```{eval-rst}
.. autofunction:: quantem.widget.io.hdf5.bin
```

## Hugging Face datasets

```{eval-rst}
.. autofunction:: quantem.widget.io.hub.list_datasets
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hub.download
```

## Save

```{eval-rst}
.. autofunction:: quantem.widget.io.save.save
```
