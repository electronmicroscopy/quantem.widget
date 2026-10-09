"""quantem core datasets in and out of the widgets.

quantem core (``Dataset2d``, ``Dataset3d``, ``Dataset4dstem``) installs with
the widget everywhere except an Intel Mac (it needs torch 2.7 and NumPy 2) and
Windows on ARM (its hdf5plugin has no ARM64 wheel). This is the only module
that imports it. Readers build datasets with ``make_dataset``: a quantem core
dataset where it is installed, else an ``ArrayDataset`` with the part of the
core surface the widgets and their readers use, so ``read_image(path).array``
and ``Show2D(read_image(path))`` behave the same on every machine.
"""

from dataclasses import dataclass, field

import numpy as np

try:
    from quantem.core.datastructures import Dataset, Dataset2d, Dataset3d, Dataset4dstem
except ImportError:
    Dataset = Dataset2d = Dataset3d = Dataset4dstem = None

# Keys quantem core's Dataset4dstem puts in every new dataset's metadata.
DATASET4DSTEM_METADATA_KEYS = ("r_to_q_rotation_cw_deg", "ellipticity")


@dataclass(eq=False)
class ArrayDataset:
    """quantem core's dataset surface where quantem core cannot install.

    ``array``, ``name``, ``sampling`` (float64 array), ``units`` (list of
    str), ``metadata``, ``shape``, ``ndim`` and ``dtype`` carry the values and
    types a core ``Dataset2d``/``3d``/``4dstem`` built from the same array
    carries (``tests/infrastructure/test_platform.py`` checks this).
    """

    array: np.ndarray
    name: str
    sampling: np.ndarray
    units: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of ``array``, as a core dataset reports it."""
        return tuple(self.array.shape)

    @property
    def ndim(self) -> int:
        """Number of axes of ``array``."""
        return self.array.ndim

    @property
    def dtype(self) -> np.dtype:
        """dtype of ``array``."""
        return self.array.dtype


def available() -> bool:
    """Whether quantem core imported."""
    return Dataset is not None


def is_dataset(obj, ndim: int | None = None) -> bool:
    """Whether ``obj`` is a quantem core dataset or an ``ArrayDataset``, of ``ndim`` dimensions when given."""
    if not isinstance(obj, ArrayDataset) and (Dataset is None or not isinstance(obj, Dataset)):
        return False
    return ndim is None or len(obj.shape) == ndim


def as_array(dataset):
    """The values of a dataset: its NumPy array, or its torch tensor when it was built from one.

    A core dataset built with ``from_tensor`` keeps the tensor on its device
    and has no ``array``; returning the tensor lets a widget compute there.
    """
    return dataset.tensor if dataset.array is None else dataset.array


def make_dataset(array, *, sampling=None, units=None, name: str = ""):
    """A calibrated dataset of a NumPy ``array``: quantem core's ``Dataset2d``/``3d``/``4dstem``
    (``Dataset`` for another number of axes), else ``ArrayDataset``.

    ``sampling`` and ``units`` default to one pixel per axis (``"frame"`` for
    the first axis of a stack). The array is kept, not copied.
    """
    values = np.asarray(array)
    default_units = {3: ("frame", "pixels", "pixels")}.get(values.ndim, ("pixels",) * values.ndim)
    sampling = np.ones(values.ndim) if sampling is None else np.array([float(value) for value in sampling])
    units = list(default_units if units is None else (str(value) for value in units))
    if available():
        core_type = {2: Dataset2d, 3: Dataset3d, 4: Dataset4dstem}.get(values.ndim, Dataset)
        return core_type.from_array(values, sampling=sampling, units=units, name=str(name))
    metadata = dict.fromkeys(DATASET4DSTEM_METADATA_KEYS) if values.ndim == 4 else {}
    return ArrayDataset(values, name=str(name), sampling=sampling, units=units, metadata=metadata)
