"""User-facing microscopy folder browser widget."""
from __future__ import annotations

import html
from pathlib import Path
import threading
import time
from typing import Any


class ShowFolder:
    """Inspect and curate a microscopy folder in a notebook.

    ``ShowFolder`` is the public folder-level API. It scans microscopy image
    files, builds thumbnails, groups related acquisitions, and stores lightweight
    image-selection state without embedding large raw arrays in the notebook.

    Parameters
    ----------
    path
        Folder to browse. If omitted, display a folder chooser and run the
        browser after the user selects a folder.
    key
        Cache key for chooser mode. Reuses the last folder for that key.
    default
        Starting folder for chooser mode when no cached folder exists.
    **browser_kwargs
        Passed through to the internal ShowFolder builder.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        key: str = "show_folder",
        default: str | Path | None = None,
        **browser_kwargs: Any,
    ) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        self.key = key
        self.default = Path(default).expanduser() if default is not None else None
        self.browser_kwargs = dict(browser_kwargs)
        self.browser = None
        self.widget = None
        self._chooser = None
        self._output = None
        self._status = None
        self._watch_stop = None
        self._watch_thread = None
        self._watch_signature = None
        self._watch_status = None

        if self.path is not None:
            self.refresh(self.path)
        else:
            self._init_chooser()

    @property
    def folder(self) -> Path | None:
        """Current browsed folder, if one has been selected."""
        if self.browser is not None:
            return self.browser.folder
        return self.path

    @property
    def items(self) -> list[Any]:
        """Browsed file items."""
        return [] if self.browser is None else self.browser.items

    @property
    def inventory_rows(self) -> list[dict[str, Any]]:
        """JSON-serializable inventory rows."""
        return [] if self.browser is None else self.browser.inventory_rows

    @property
    def cache_info(self) -> dict[str, Any]:
        """Thumbnail/index cache status for the latest folder-browser run."""
        return {"enabled": False} if self.browser is None else dict(self.browser.cache_info)

    @property
    def cache_path(self) -> Path | None:
        """Folder containing the latest ShowFolder cache manifest and thumbnails."""
        return None if self.browser is None else self.browser.cache_path

    def selected(self, kind: str | None = None) -> list[Any]:
        """Return selected folder-browser items."""
        return [] if self.browser is None else self.browser.selected(kind)

    def paths(self, kind: str | None = None) -> list[Path]:
        """Return selected file paths."""
        return [] if self.browser is None else self.browser.paths(kind)

    def selected_paths(self, kind: str | None = None) -> list[Path]:
        """Compatibility alias for :meth:`paths`."""
        return self.paths(kind)

    def selected_folders(self) -> list[Path]:
        """Return parent folders containing selected files."""
        return sorted({path.parent for path in self.paths()})

    def selection(self) -> dict[str, Any]:
        """Return the current JSON-serializable selection snapshot."""
        if self.browser is None:
            return {
                "folder": None if self.folder is None else str(self.folder),
                "selected_files": [],
                "selected_folders": [],
            }
        data = dict(self.browser.selection())
        data["selected_folders"] = [str(path) for path in self.selected_folders()]
        return data

    def save(self, path: str | Path | None = None) -> Path:
        """Write the current selection JSON."""
        if self.browser is None:
            raise ValueError("No folder has been browsed yet.")
        return self.browser.save(path)

    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        """Load selection JSON into the current folder browser."""
        if self.browser is None:
            raise ValueError("No folder has been browsed yet.")
        return self.browser.load(path)

    def clear_cache(self) -> None:
        """Delete the current ShowFolder thumbnail/index cache, if present."""
        if self.browser is not None:
            self.browser.clear_cache()

    def export_html(
        self,
        path: str | Path | None = None,
        *,
        title: str | None = None,
    ) -> Path:
        """Write a standalone HTML folder browser and return its path."""
        if self.browser is None:
            raise ValueError("No folder has been browsed yet.")
        return self.browser.export_html(path, title=title)

    def show_selected(self):
        """Return a Show2D gallery containing the currently starred images."""
        if self.browser is None:
            raise ValueError("No folder has been browsed yet.")
        return self.browser.show_selected()

    def show_selected_stack(self):
        """Return currently starred images as a Show3D frame stack."""
        if self.browser is None:
            raise ValueError("No folder has been browsed yet.")
        return self.browser.show_selected_stack()

    def watch(
        self,
        interval: float = 2.0,
        *,
        start: bool = True,
    ) -> "ShowFolder":
        """Watch the folder and update the displayed browser when files change.

        The watcher uses the same thumbnail/index cache as normal ShowFolder
        construction. Unchanged files stay cached, new or modified files are
        read, and files removed from disk disappear from the in-memory browser
        and from the next cache manifest.
        """
        if self.path is None:
            raise ValueError("No folder path is selected.")
        if self.browser is None:
            self.refresh(self.path)
        if start:
            self.stop_watch()
        self._ensure_watch_status()
        self._watch_signature = self._folder_signature()
        self._set_watch_status("watching", interval=float(interval))
        if not start:
            return self
        stop = threading.Event()
        self._watch_stop = stop

        def _loop() -> None:
            while not stop.wait(float(interval)):
                try:
                    self.watch_once()
                except Exception as exc:  # pragma: no cover - defensive UI path
                    self._set_watch_status(f"watch error: {exc}")

        thread = threading.Thread(target=_loop, name="ShowFolderWatch", daemon=True)
        self._watch_thread = thread
        thread.start()
        return self

    def stop_watch(self) -> "ShowFolder":
        """Stop a running ShowFolder folder watcher."""
        stop = self._watch_stop
        if stop is not None:
            stop.set()
        thread = self._watch_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._watch_stop = None
        self._watch_thread = None
        if self._watch_status is not None:
            self._set_watch_status("watch stopped")
        return self

    def watch_once(self) -> bool:
        """Poll once and update the displayed browser if the folder changed."""
        if self.path is None:
            raise ValueError("No folder path is selected.")
        before = self._watch_signature or self._folder_signature()
        after = self._folder_signature()
        if after == before:
            self._set_watch_status("watching", interval=None)
            return False
        old_ids = {item.file_id for item in self.items}
        old_selection = self.selection()
        selected_ids = {
            str(row.get("id"))
            for row in old_selection.get("selected_files", [])
            if row.get("kind") == "image"
        }
        hidden_ids = set(old_selection.get("hidden_image_ids", []))
        started = time.perf_counter()
        self._set_watch_status("reading changed folder...")
        self._replace_browser(selected_image_ids=selected_ids, hidden_image_ids=hidden_ids)
        self._watch_signature = self._folder_signature()
        new_ids = {item.file_id for item in self.items}
        added = sorted(new_ids - old_ids)
        removed = sorted(old_ids - new_ids)
        cache = self.cache_info
        self._set_watch_status(
            "updated",
            added=added,
            removed=removed,
            hits=int(cache.get("hits", 0)),
            misses=int(cache.get("misses", 0)),
            seconds=time.perf_counter() - started,
        )
        return True

    def refresh(self, path: str | Path | None = None) -> "ShowFolder":
        """Run or rerun the microscopy folder browser."""
        from quantem.widget.showfolder_core import build_showfolder

        if path is not None:
            self.path = Path(path).expanduser()
        if self.path is None:
            raise ValueError("No folder path is selected.")
        if self._chooser is not None:
            from quantem.widget.folder_picker import _load_cache, _save_cache

            cache = _load_cache()
            cache[self.key] = str(self.path)
            _save_cache(cache)
        new_browser = build_showfolder(self.path, **self.browser_kwargs)
        if self.widget is not None and self.browser is not None:
            self._install_browser(new_browser)
        else:
            self.browser = new_browser
            self.widget = new_browser.widget
        self._watch_signature = self._folder_signature()
        if self._status is not None:
            self._status.value = (
                "<b>Selected folder:</b> "
                f"<code>{html.escape(str(self.browser.folder))}</code>"
            )
        return self

    def _replace_browser(
        self,
        *,
        selected_image_ids: set[str],
        hidden_image_ids: set[str],
    ) -> None:
        from quantem.widget.showfolder_core import build_showfolder

        if self.path is None:
            raise ValueError("No folder path is selected.")
        previous = self.browser
        new_browser = build_showfolder(self.path, **self.browser_kwargs)
        new_browser.apply_selection(
            selected_image_ids=selected_image_ids,
            hidden_image_ids=hidden_image_ids,
        )
        if previous is not None:
            new_browser.inherit_selected_viewers_from(previous)
        self._install_browser(new_browser)

    def _install_browser(self, new_browser: Any) -> None:
        if self.widget is not None:
            self.widget.children = tuple(new_browser.widget.children)
            new_browser.widget = self.widget
        else:
            self.widget = new_browser.widget
        self.browser = new_browser
        if self._watch_status is not None:
            self._insert_watch_status()

    def _folder_signature(self) -> tuple[tuple[str, int, int], ...]:
        if self.path is None:
            return ()
        root = Path(self.path).expanduser().resolve()
        glob = str(self.browser_kwargs.get("glob", "*.emd"))
        rows = []
        if not root.is_dir():
            return ()
        for path in sorted(p for p in root.glob(glob) if p.is_file() and not p.name.startswith(".")):
            stat = path.stat()
            rows.append((path.relative_to(root).as_posix(), int(stat.st_size), int(stat.st_mtime_ns)))
        return tuple(rows)

    def _ensure_watch_status(self) -> None:
        if self._watch_status is None:
            from ipywidgets import HTML

            self._watch_status = HTML()
        self._insert_watch_status()

    def _insert_watch_status(self) -> None:
        if self.widget is None or self._watch_status is None:
            return
        children = [child for child in self.widget.children if child is not self._watch_status]
        insert_at = 1 if children else 0
        self.widget.children = tuple(children[:insert_at] + [self._watch_status] + children[insert_at:])

    def _set_watch_status(
        self,
        message: str,
        *,
        interval: float | None = None,
        added: list[str] | None = None,
        removed: list[str] | None = None,
        hits: int | None = None,
        misses: int | None = None,
        seconds: float | None = None,
    ) -> None:
        if self._watch_status is None:
            return
        details = [html.escape(message)]
        if interval is not None:
            details.append(f"poll {interval:g}s")
        if hits is not None and misses is not None:
            details.append(f"{hits} cached")
            details.append(f"{misses} read")
        if added:
            details.append(f"{len(added)} new")
        if removed:
            details.append(f"{len(removed)} removed")
        if seconds is not None:
            details.append(f"{seconds:.2f}s")
        self._watch_status.value = (
            "<div style='font-size:12px;color:#555;margin:2px 0 8px 0;"
            "padding:4px 6px;background:#f6f8fa;border:1px solid #d0d7de;"
            "border-radius:4px;display:inline-block'>"
            f"ShowFolder watch: {' · '.join(details)}</div>"
        )

    def _init_chooser(self) -> None:
        from quantem.widget.folder_picker import _load_cache
        from ipywidgets import Button, HTML, Output, Text, VBox

        cache = _load_cache()
        cached = cache.get(self.key)
        if cached and Path(cached).is_dir():
            start = Path(cached)
        elif self.default is not None and self.default.is_dir():
            start = self.default
        else:
            start = Path.home()
        status = HTML("<b>Select a folder to browse.</b>")
        output = Output()
        refresh = Button(description="Open folder", tooltip="Run ShowFolder on the selected folder")
        children: list[Any]

        try:
            from ipyfilechooser import FileChooser
        except ModuleNotFoundError:
            path_preview = HTML(
                "<div style='font-size:13px;color:#444;margin:6px 0 2px 0;'>"
                "Folder path to browse</div>"
                f"<div style='font-family:monospace;font-size:13px;line-height:1.35;"
                f"padding:8px;border:1px solid #bbb;border-radius:6px;"
                f"background:#f7f7f7;overflow-wrap:anywhere;'>{html.escape(str(start))}</div>"
            )
            chooser = Text(
                value=str(start),
                description="Path:",
                placeholder="/path/to/microscopy/session",
                layout={"width": "100%", "max_width": "900px"},
                style={"description_width": "44px"},
            )
            help_text = HTML(
                "<p style='margin:4px 0;color:#666'>"
                "Install <code>ipyfilechooser</code> for a click-to-browse folder tree. "
                "This fallback still runs the same ShowFolder browser from the typed path."
                "</p>"
            )
            refresh.button_style = "primary"
            refresh.layout.width = "180px"
            children = [status, path_preview, chooser, help_text, refresh, output]
        else:
            chooser = FileChooser(
                str(start),
                show_only_dirs=True,
                title=f"Select folder ({self.key})",
            )
            children = [status, chooser, refresh, output]

        def _selected_path() -> Path | None:
            selected = getattr(chooser, "selected", None)
            if selected:
                return Path(selected).expanduser()
            value = getattr(chooser, "value", None)
            return Path(value).expanduser() if value else None

        def _run(_=None) -> None:
            selected = _selected_path()
            if selected is None:
                status.value = "<b>Select a folder first.</b>"
                return
            if "path_preview" in locals():
                path_preview.value = (
                    "<div style='font-size:13px;color:#444;margin:6px 0 2px 0;'>"
                    "Folder path to browse</div>"
                    f"<div style='font-family:monospace;font-size:13px;line-height:1.35;"
                    f"padding:8px;border:1px solid #bbb;border-radius:6px;"
                    f"background:#f7f7f7;overflow-wrap:anywhere;'>{html.escape(str(selected))}</div>"
                )
            output.clear_output(wait=True)
            with output:
                from IPython.display import display

                self.refresh(selected)
                display(self.browser.widget)

        if hasattr(chooser, "register_callback"):
            chooser.register_callback(lambda _: _run())
        refresh.on_click(_run)
        self._chooser = chooser
        self._output = output
        self._status = status
        self.widget = VBox(children)

    def _repr_mimebundle_(self, **kwargs):
        if self.widget is None:
            return None
        return self.widget._repr_mimebundle_(**kwargs)

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.widget)


def show_folder(
    path: str | Path | None = None,
    *,
    key: str = "show_folder",
    default: str | Path | None = None,
    **browser_kwargs: Any,
) -> ShowFolder:
    """Open a microscopy folder browser widget."""
    return ShowFolder(path, key=key, default=default, **browser_kwargs)


def prebuild_showfolder_cache(
    path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    thumb: int = 256,
    glob: str = "*.emd",
    rebuild_cache: bool = False,
    **browser_kwargs: Any,
) -> dict[str, Any]:
    """Build or refresh a ShowFolder thumbnail/index cache without displaying UI.

    This is useful on a workstation or SSD-backed scratch path before opening a
    large microscopy session in a notebook. It runs the same ShowFolder pipeline as
    :class:`ShowFolder`, then returns ``cache_info``.
    """
    widget = ShowFolder(
        path,
        thumb=thumb,
        glob=glob,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
        **browser_kwargs,
    )
    return widget.cache_info
