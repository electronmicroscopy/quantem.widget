"""Focused interactive region selection for two-dimensional arrays."""

from pathlib import Path
from typing import Self

import numpy as np
import traitlets

from .show2d import Colormap, Show2D
from .utils.roi_geometry import roi_masks


class Mask2D(Show2D):
    """Draw one reusable Boolean mask over a two-dimensional image.

    ``Mask2D`` has its own focused browser interface: choose a shape and drag
    the desired region. The committed region is synchronized to Python only on
    release. Its display does not alter :class:`Show2D`.

    Parameters
    ----------
    data : array-like or Dataset2d
        Numerical image with shape ``(row, col)``. Dataset sampling and units
        are preserved for calibrated display.
    shape : {"rectangle", "square", "circle"}, default="rectangle"
        Initial drawing shape.
    title : str, default="Select region"
        Compact title shown above the image.
    cmap : str or Colormap, default="gray"
        Image colormap.
    **kwargs
        Image display options accepted by ``Show2D``. ``display_bin`` must
        remain 1 so the returned mask matches the input.

    Examples
    --------
    >>> region = Mask2D(image)
    >>> region
    >>> mask = region.mask
    >>> selected_values = image[mask]
    """

    _VALID_SHAPES = {"rectangle", "square", "circle"}
    _esm = Path(__file__).parent / "static" / "mask2d.js"

    mask_shape = traitlets.Enum(
        ["rectangle", "square", "circle"], default_value="rectangle"
    ).tag(sync=True)

    def __init__(
        self,
        data,
        *,
        shape: str = "rectangle",
        title: str = "Select region",
        cmap: str | Colormap = "gray",
        **kwargs,
    ) -> None:
        values = np.asarray(data.array if hasattr(data, "array") else data)
        if values.ndim == 3 and values.shape[0] == 1:
            data = values[0]
            values = data
        if values.ndim != 2:
            raise ValueError(
                "Mask2D accepts one 2-D image with shape (row, col); "
                f"got shape {values.shape}. Use one selected frame or projection."
            )
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("Mask2D data must contain numerical image intensities.")
        if shape not in self._VALID_SHAPES:
            choices = ", ".join(sorted(self._VALID_SHAPES))
            raise ValueError(f"shape must be one of {choices}; got {shape!r}.")
        if kwargs.get("display_bin", 1) != 1:
            raise ValueError(
                "Mask2D requires display_bin=1 so selector.mask keeps the "
                "input image shape."
            )
        kwargs.setdefault("show_controls", False)
        kwargs.setdefault("show_stats", False)
        kwargs.setdefault("show_fft", False)
        kwargs["display_bin"] = 1
        kwargs.setdefault("verbose", False)
        super().__init__(data, title=title, cmap=cmap, **kwargs)
        with self.hold_sync():
            self.mask_shape = shape
            self.roi_active = True

    @property
    def mask(self) -> np.ndarray:
        """Union of selected regions as one ``(row, col)`` Boolean mask."""
        masks = roi_masks(
            list(self.roi_list),
            height=int(self.height),
            width=int(self.width),
        )
        if not len(masks):
            return np.zeros((int(self.height), int(self.width)), dtype=bool)
        if len(masks) == 1:
            return masks[0]
        return np.any(masks, axis=0)

    @property
    def geometry(self) -> dict[str, object] | None:
        """Selected JSON-friendly geometry in ``(row, col)`` coordinates.

        Most workflows only need :attr:`mask`. Geometry is available when a
        downstream calculation also needs the selected center, bounds, or
        radius.
        """
        geometries = self.get_roi_geometries()
        if not geometries:
            return None
        selected = int(self.roi_selected_idx)
        for geometry in geometries:
            if int(geometry["index"]) == selected:
                return geometry
        return geometries[-1]

    def clear(self) -> Self:
        """Clear the selected region and return this widget."""
        self.clear_rois()
        self.roi_active = True
        return self

    def state_dict(self) -> dict:
        """Return display state including the active selection shape."""
        return {**super().state_dict(), "mask_shape": self.mask_shape}

    def load_state_dict(self, state) -> None:
        """Restore display and selection state."""
        state = dict(state)
        shape = state.pop("mask_shape", self.mask_shape)
        super().load_state_dict(state)
        if shape in self._VALID_SHAPES:
            self.mask_shape = shape

    def _normalise_html_export_options(self, **kwargs) -> tuple[str, bool, int]:
        options = super()._normalise_html_export_options(**kwargs)
        if options[2] != 1:
            raise ValueError(
                "Mask2D HTML export requires downsample=1 so selection "
                "coordinates remain full resolution."
            )
        return options
