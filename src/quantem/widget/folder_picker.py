"""Persistent folder picker for notebooks.

Wraps ``ipyfilechooser.FileChooser`` with a JSON cache under
``~/.cache/qt/folders.json`` so users only have to click
once per dataset - kernel restarts reuse the last pick automatically.
The widget also shows the ``*_master.h5`` files inside the currently
selected folder so you immediately see what you're about to work with.

Example
-------
>>> from quantem.widget import pick_folder
>>> picker = pick_folder("browse_256", default="data/")
>>> picker            # displays the widget: chooser on top, master list below
>>> DATA_DIR = picker.value   # returns the cached path immediately; after a
                              # new click it returns the freshly selected path
"""

import html
import json
from pathlib import Path

_CACHE_DIR = Path.home() / ".cache" / "qt"
_CACHE_FILE = _CACHE_DIR / "folders.json"


def _load_cache() -> dict[str, str]:
    """Last pick per key; a missing or unreadable cache is no history, not an error."""
    if not _CACHE_FILE.exists():
        return {}
    try:
        return json.loads(_CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(data: dict[str, str]) -> None:
    """Write the cache through a temporary file and a rename.

    A crash mid-write then never leaves a half-written cache that the next
    launch treats as "no history".
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = _CACHE_FILE.with_suffix(_CACHE_FILE.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(_CACHE_FILE)


def _list_masters(path: str | None) -> list[str]:
    """Names of the ``*_master.h5`` files in ``path``; none for no folder or a path that is not one."""
    if not path:
        return []
    folder = Path(path)
    if not folder.is_dir():
        return []
    return sorted(master.name for master in folder.glob("*_master.h5"))


def _render_master_list(path: str | None) -> str:
    """HTML list of the folder's masters, so the user sees what a pick contains before loading it."""
    if not path:
        return '<div style="color:#888; font-size:12px;">No folder selected.</div>'
    masters = _list_masters(path)
    folder_name = html.escape(Path(path).name or path)
    if not masters:
        return (
            f'<div style="color:#888; font-size:12px;">'
            f'No <code>*_master.h5</code> files in <b>{folder_name}/</b>'
            f'</div>'
        )
    rows = "".join(
        f'<div style="font-family:monospace; font-size:12px; padding:1px 0;">'
        f'[{i:3d}] {html.escape(name)}</div>'
        for i, name in enumerate(masters)
    )
    return (
        f'<div style="margin-top:6px; padding:8px 10px; border:1px solid #ddd; '
        f'border-radius:4px; max-height:240px; overflow-y:auto;">'
        f'<div style="font-weight:600; font-size:13px; margin-bottom:4px;">'
        f'{len(masters)} master file{"s" if len(masters) != 1 else ""} in {folder_name}/'
        f'</div>{rows}</div>'
    )


class FolderPicker:
    """Persistent folder chooser with live master-file list.

    Pairs an ``ipyfilechooser.FileChooser`` with an HTML widget that shows
    every ``*_master.h5`` file in the currently selected folder. The cached
    path for ``key`` is read from ``~/.cache/qt/folders.json`` on
    construction and written on every selection, so kernel restarts keep
    the last pick and notebook cells can always do ``DATA_DIR = fc.value``.

    Parameters
    ----------
    key : str
        Cache key for this picker - one per notebook or dataset tag
        (e.g. ``"browse_256"``, ``"calibrate_512"``).
    default : str, optional
        Starting path when no cache entry exists. Falls back to ``$HOME``.
    """

    def __init__(self, key: str, default: str | None = None) -> None:
        try:
            from ipyfilechooser import FileChooser  # only session tooling (quantem.live) opens a FolderPicker
        except ImportError as exc:
            raise ImportError("FolderPicker needs ipyfilechooser: pip install ipyfilechooser") from exc
        from ipywidgets import HTML, VBox

        self._key = key
        self._default = default
        cache = _load_cache()
        cached = cache.get(key)
        if cached and Path(cached).is_dir():
            start = cached
        elif default and Path(default).is_dir():
            start = default
        else:
            start = str(Path.home())
        self._chooser = FileChooser(
            start,
            show_only_dirs=True,
            title=f"Select folder ({key})",
        )
        self._chooser.register_callback(self._on_pick)
        self._masters_html = HTML(value=_render_master_list(self.value))
        self._box = VBox([self._chooser, self._masters_html])

    def _on_pick(self, chooser) -> None:
        """Remember a new pick under this key and refresh the master list."""
        if chooser.selected:
            cache = _load_cache()
            cache[self._key] = chooser.selected
            _save_cache(cache)
        self._masters_html.value = _render_master_list(self.value)

    @property
    def value(self) -> str | None:
        """Current path. Resolution order: fresh click → cached → default."""
        if self._chooser.selected:
            return self._chooser.selected
        cached = _load_cache().get(self._key)
        if cached:
            return cached
        return self._default

    @property
    def masters(self) -> list[str]:
        """Filenames of ``*_master.h5`` files in the currently selected folder."""
        return _list_masters(self.value)

    def _ipython_display_(self) -> None:
        """Show the chooser above the master list."""
        from IPython.display import display
        display(self._box)


def pick_folder(key: str, default: str | None = None) -> FolderPicker:
    """Persistent folder chooser - see :class:`FolderPicker`."""
    return FolderPicker(key, default)
