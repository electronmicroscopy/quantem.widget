"""Widget readers and the tutorial-data download helper.

Every reader returns the dataset type of this machine: ``read_4dstem`` a
``quantem.gpu.io.Dataset4dstemGPU`` when quantem.gpu is installed and a GPU is
present (it loads through ``quantem.gpu.io.load``), else a quantem core
``Dataset4dstem``; ``read_image`` and ``read_images`` quantem core
``Dataset2d``; ``read_image_stack`` a quantem core ``Dataset3d``. Where quantem
core cannot install (an Intel Mac, Windows on ARM) the widget's stand-in
carries the same ``array``, ``name``, ``sampling``, ``units`` and ``metadata``.
Saving, discovery and inspection of 4D-STEM acquisitions belong to
``quantem.gpu.io``; this namespace does not forward them.

>>> from quantem.widget import io  # doctest: +SKIP
>>> ds = io.read_image(path)  # doctest: +SKIP
>>> arr = ds.array  # doctest: +SKIP
"""

from importlib import import_module

_EXPORTS = {
    "download": "quantem.widget.io.hub",
    "read_4dstem": "quantem.widget.show4dstem.reader",
    "read_image": "quantem.widget.io.image",
    "read_image_stack": "quantem.widget.io.image",
    "read_images": "quantem.widget.io.image",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Import a reader on first use, so ``from quantem.widget import io`` stays light."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include the lazy readers in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
