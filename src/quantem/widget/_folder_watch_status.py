"""Shared live-folder watcher state protocol for widget backends."""

from __future__ import annotations

from typing import Any, Literal


FolderWatchState = Literal[
    "hidden",
    "watching",
    "updating",
    "waiting",
    "error",
    "stopped",
    "not_watching",
]

FOLDER_WATCH_STATE_VALUES = (
    "hidden",
    "watching",
    "updating",
    "waiting",
    "error",
    "stopped",
    "not_watching",
)
FOLDER_WATCH_STATES = frozenset(FOLDER_WATCH_STATE_VALUES)


def set_folder_watch_status(
    widget: Any,
    state: FolderWatchState,
    detail: str = "",
) -> None:
    """Publish one validated watcher state to optional synced traits."""
    if state not in FOLDER_WATCH_STATES:
        raise ValueError(
            f"Unknown folder watch state {state!r}; expected one of "
            f"{sorted(FOLDER_WATCH_STATES)}"
        )
    for name, value in (
        ("folder_watch_state", state),
        ("folder_watch_detail", str(detail)),
    ):
        try:
            setattr(widget, name, value)
        except Exception:
            # The helper is also used by strict/slotted protocol tests and
            # downstream widgets that may not expose these advisory traits.
            pass
