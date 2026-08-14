# Mask2D

`Mask2D` turns one region drawn on a 2D image into a Boolean NumPy mask. The
ordinary workflow has only two steps:

```python
from quantem.widget import Mask2D

selector = Mask2D(image)
selector
```

After drawing or replacing the region, use the mask directly:

```python
mask = selector.mask
selected_values = image[mask]
```

There is no result dictionary to unpack. `selector.mask` has the same
`(row, col)` shape as the input and Boolean dtype. `selector.geometry` is
optional metadata for workflows that also need the selected center, bounds, or
radius.

## Configuration

Choose the initial shape and the display settings that matter for the image:

```python
selector = Mask2D(
    image,
    shape="circle",
    title="Gold particle region",
    cmap="magma",
    auto_contrast=True,
    sampling=0.02,
    units="nm",
    show_scale_bar=True,
)
```

The toolbar can switch between rectangle, square, and circle without rebuilding
the widget. A new drag replaces the previous region; **Clear** returns an
all-false mask. The selection is synchronized to Python when the pointer is
released, so dragging remains browser-local and responsive.

`Mask2D` accepts the normal [Show2D](show2d) display options. It defaults to a
focused selection surface with the general controls, statistics, and FFT hidden.
Set `show_controls=True`, `show_stats=True`, or `show_fft=True` when those
Show2D tools are useful for choosing the region. Display binning remains fixed
at 1 so `selector.mask` always matches the input image shape.

Standalone HTML export likewise keeps `downsample=1`; choose either full
float32 or uint8 encoding without changing the selection coordinates.

For a native-resolution image, pass the full array or dataset directly. This
real gold HAADF example keeps all 4096 by 4096 pixels:

```python
from quantem.widget.datasets import show2d_gold

gold = show2d_gold(size="full")
selector = Mask2D(
    gold,
    title="Select a gold region",
    cmap="inferno",
    show_scale_bar=True,
)
selector

mask = selector.mask  # Boolean shape: (4096, 4096)
```

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.mask2d.Mask2D
   :members:
   :show-inheritance:
```

## Interactive controls

| Control | Trait | Expected effect |
|---|---|---|
| Shape | `mask_shape` | Chooses rectangle, square, or circle for the next drag |
| Image drag | `roi_list`, `roi_selected_idx` | Shows a live preview, then replaces and synchronizes the selected region on release |
| Clear | `roi_list` | Removes the region; `selector.mask` becomes all false |
