"""Temporary widget data adapters.

Core dataset containers belong in ``quantem.core``. Modules here are compatibility
homes for structures that have not moved to core yet.
"""

from quantem.widget.data.dataset5dstem import Dataset5dstem

_TUTORIAL_EXPORTS = {
    "load_tutorial_showfolder_folder",
    "load_tutorial_show2d",
    "load_tutorial_show3d",
    "load_tutorial_show4dstem",
    "show1d_ducky",
    "show2d_gold",
    "show3d_gold",
    "show4dstem_gold",
    "showfolder_gold",
}

__all__ = [
    "Dataset5dstem",
    "load_tutorial_showfolder_folder",
    "load_tutorial_show2d",
    "load_tutorial_show3d",
    "load_tutorial_show4dstem",
    "show1d_ducky",
    "show2d_gold",
    "show3d_gold",
    "show4dstem_gold",
    "showfolder_gold",
]


def __getattr__(name: str):
    if name in _TUTORIAL_EXPORTS:
        from quantem.widget.data import tutorials

        value = getattr(tutorials, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
