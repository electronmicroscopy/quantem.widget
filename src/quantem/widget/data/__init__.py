"""Widget-owned viewer residency and tutorial-data helpers.

Scientific dataset types belong in ``quantem.core``. ``Dataset5dstem`` remains
here because it implements Show4DSTEM paging and multi-device UI residency.
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
    "showdiffraction_fe3o4",
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
    "showdiffraction_fe3o4",
    "showfolder_gold",
]


def __getattr__(name: str):
    if name in _TUTORIAL_EXPORTS:
        from quantem.widget.data import tutorials

        value = getattr(tutorials, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
