"""Widget state plumbing shared by every display widget.

Small mixins, each owning one copy of behaviour that used to be pasted
into each widget class:

``LiveMountMixin``
    Marks the first display handshake so saved-state trimming can tell a live
    mount apart from a notebook save.
``StateFileMixin``
    ``save`` writes the versioned JSON envelope around ``state_dict``.
``SavedStateMixin``
    Adds the ``get_state`` trim that keeps heavy pixel buffers and a running
    folder watcher out of the saved notebook (widgets with a static preview).

``announce_browser_limit`` says when a display payload is too large to reach
the browser in one message.
"""

import pathlib

from quantem.widget.utils.state_io import save_state_file

FOLDER_WATCH_LIVE_STATES = frozenset({"watching", "updating", "waiting", "error"})
# One kernel message above this never reaches the browser: the websocket drops
# and the widget keeps only its static preview. Measured 2026-10-07 with Show3D
# in JupyterLab and Chrome: a 1920 MB stack arrives, a 2176 MB stack does not.
BROWSER_MESSAGE_LIMIT_BYTES = 2 * 1024**3


def announce_browser_limit(widget: str, nbytes: int, display_bin: int = 1) -> None:
    """Print the smallest ``display_bin`` that fits when ``nbytes`` cannot reach the browser.

    The widget does not bin on its own (defaults stay lossless); without this
    line the viewer would silently stay a static picture. ``nbytes`` is the
    payload already binned by ``display_bin``, so the native size is
    ``nbytes * display_bin**2`` and the suggestion is the smallest factor above
    ``display_bin`` whose payload fits: repeating the factor the user already
    passed would not help.
    """
    if nbytes <= BROWSER_MESSAGE_LIMIT_BYTES:
        return
    native = nbytes * display_bin**2
    factor = display_bin + 1
    while native / factor**2 > BROWSER_MESSAGE_LIMIT_BYTES:
        factor += 1
    current = f" with display_bin={display_bin}" if display_bin > 1 else ""
    print(
        f"{widget} display payload is {nbytes / 2**20:.0f} MB{current}; one browser message holds at most "
        f"{BROWSER_MESSAGE_LIMIT_BYTES / 2**20:.0f} MB, so the viewer can only show its static preview. "
        f"Pass display_bin={factor} ({native / factor**2 / 2**20:.0f} MB) for an interactive view."
    )


class LiveMountMixin:
    """Flag the first display handshake.

    ``get_state`` is called both when a notebook is saved and while the live
    frontend mounts. Folder-watcher badges and the first live frame must stay
    truthful during the mount, so the display hooks set
    ``_initial_live_mount_state`` for their duration and the trimming in
    ``SavedStateMixin.get_state`` leaves that call alone.
    """

    _initial_live_mount_state = False

    def _with_initial_live_mount_state(self, display_hook, *args, **kwargs):
        """Run one display hook with the live-mount flag set, so its ``get_state`` calls are not trimmed as a save."""
        self._initial_live_mount_state = True
        try:
            return display_hook(*args, **kwargs)
        finally:
            self._initial_live_mount_state = False

    def _repr_mimebundle_(self, **kwargs):
        """The display bundle, built while the live-mount flag is set."""
        return self._with_initial_live_mount_state(super()._repr_mimebundle_, **kwargs)

    def _ipython_display_(self):
        """The notebook display, published while the live-mount flag is set."""
        return self._with_initial_live_mount_state(super()._ipython_display_)


class StateFileMixin:
    """``save`` for every widget with a ``state_dict``."""

    def save(self, path: str | pathlib.Path) -> None:
        """Write the display state (not the pixels) to a versioned JSON file.

        The envelope records the widget class and a schema version so
        ``load_state_dict`` can refuse a file written by another widget.
        """
        save_state_file(path, type(self).__name__, self.state_dict())


class SavedStateMixin(StateFileMixin, LiveMountMixin):
    """Saved-notebook trim of ``get_state`` for widgets with a static preview.

    Widgets list the synced traits that hold pixels in ``_UNSAVED_HEAVY_KEYS``
    and mix in ``StaticFallbackMixin`` for the preview image.
    """

    _UNSAVED_HEAVY_KEYS: tuple[str, ...] = ()
    _save_state = False

    def get_state(self, key=None, drop_defaults=False):
        """Trait state for comm sync and notebook embedding.

        ipywidgets calls this with ``key=None`` to snapshot the FULL state that
        gets written into the saved notebook's ``metadata.widgets``. When
        ``save_state`` is False the heavy pixel buffers are dropped from that
        snapshot, so a plain widget does not bake hundreds of MB into the
        .ipynb, and the static preview is filled in so a kernel-less reopen
        still shows the image. Targeted syncs (``key`` is a name or set, used
        by hold_sync / send_state during live rendering) are untouched, so the
        frontend still receives every buffer. ``save_state=True`` embeds
        everything so a reopened notebook restores the interactive widget
        without a kernel. A saved snapshot also cannot carry the Python folder
        watcher thread, so a live watcher badge is written as stopped.
        """
        state = super().get_state(key=key, drop_defaults=drop_defaults)
        if key is None and not self._save_state and not self._initial_live_mount_state:
            if not self._static_fallback_enabled():
                state.pop("_static_fallback_jpeg", None)
                state.pop("_static_fallback_mime", None)
            elif not self._static_fallback_jpeg:
                png = self._static_fallback_png_b64()
                if png:
                    self._store_static_fallback_preview(png)
                    state["_static_fallback_jpeg"] = self._static_fallback_jpeg
                    state["_static_fallback_mime"] = self._static_fallback_mime
            for heavy_key in self._UNSAVED_HEAVY_KEYS:
                state.pop(heavy_key, None)
        if (
            key is None
            and not self._initial_live_mount_state
            and state.get("folder_watch_state") in FOLDER_WATCH_LIVE_STATES
        ):
            state["folder_watch_state"] = "stopped"
            state["folder_watch_detail"] = (
                "Folder watcher is not running in saved widget state. "
                "Re-run the cell to resume live folder updates."
            )
            if state.get("folder_waiting"):
                state["folder_status"] = (
                    "No completed image was captured in this saved widget. "
                    "Re-run the cell to resume folder watching."
                )
        return state
