"""
quantem.widget: Interactive Jupyter widgets using anywidget + React.
"""

import importlib.metadata
from typing import TYPE_CHECKING, Any

try:
    __version__ = importlib.metadata.version("quantem-widget")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

if TYPE_CHECKING:  # pragma: no cover
    from quantem.widget.align2d import Align2D
    from quantem.widget.bin import Bin
    from quantem.widget.edit2d import Edit2D
    from quantem.widget.io import IO, IOResult
    from quantem.widget.mark2d import Mark2D
    from quantem.widget.show1d import Show1D
    from quantem.widget.show2d import Show2D
    from quantem.widget.show3d import Show3D
    from quantem.widget.show3dvolume import Show3DVolume
    from quantem.widget.show4d import Show4D
    from quantem.widget.show4dstem import Show4DSTEM
    from quantem.widget.showcomplex import ShowComplex2D


_EXPORTS = {
    "Align2D": ("quantem.widget.align2d", "Align2D"),
    "Bin": ("quantem.widget.bin", "Bin"),
    "Edit2D": ("quantem.widget.edit2d", "Edit2D"),
    "IO": ("quantem.widget.io", "IO"),
    "IOResult": ("quantem.widget.io", "IOResult"),
    "Mark2D": ("quantem.widget.mark2d", "Mark2D"),
    "Show1D": ("quantem.widget.show1d", "Show1D"),
    "Show2D": ("quantem.widget.show2d", "Show2D"),
    "Show3D": ("quantem.widget.show3d", "Show3D"),
    "Show3DVolume": ("quantem.widget.show3dvolume", "Show3DVolume"),
    "Show4D": ("quantem.widget.show4d", "Show4D"),
    "Show4DSTEM": ("quantem.widget.show4dstem", "Show4DSTEM"),
    "ShowComplex2D": ("quantem.widget.showcomplex", "ShowComplex2D"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = __import__(module_name, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'quantem.widget' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))

__all__ = [
    "Align2D",
    "Bin",
    "Edit2D",
    "IO",
    "IOResult",
    "Mark2D",
    "Show1D",
    "Show2D",
    "Show3D",
    "Show3DVolume",
    "Show4D",
    "Show4DSTEM",
    "ShowComplex2D",
]
