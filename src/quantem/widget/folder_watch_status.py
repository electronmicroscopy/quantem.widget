"""Shared live-folder watcher state protocol for widget backends."""

from typing import Literal, get_args

FolderWatchState = Literal[
    "hidden",
    "watching",
    "updating",
    "waiting",
    "error",
    "stopped",
    "not_watching",  # set by the browser only (js/folderWatchStatus.tsx)
]

FOLDER_WATCH_STATE_VALUES = get_args(FolderWatchState)


def set_folder_watch_status(widget, state: FolderWatchState, detail: str = "") -> None:
    """Publish one watcher state to the widget's ``folder_watch_state`` and ``folder_watch_detail`` traits.

    The Enum trait refuses a state outside ``FolderWatchState``.
    """
    widget.folder_watch_state = state
    widget.folder_watch_detail = str(detail)
