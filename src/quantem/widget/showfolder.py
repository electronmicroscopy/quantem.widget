"""User-facing folder survey widget."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any


class ShowFolder:
    """Inspect and curate a microscopy folder in a notebook.

    ``ShowFolder`` is the user-facing wrapper around :func:`survey`. It keeps
    the public API aligned with ``Show2D``/``Show3D`` while reusing the existing
    microscopy folder scanner for thumbnails, metadata, EDS detection, and
    selection state.

    Parameters
    ----------
    path
        Folder to survey. If omitted, display a folder chooser and run the
        survey after the user selects a folder.
    key
        Cache key for chooser mode. Reuses the last folder for that key.
    default
        Starting folder for chooser mode when no cached folder exists.
    **survey_kwargs
        Passed through to :func:`quantem.widget.survey.survey`.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        key: str = "show_folder",
        default: str | Path | None = None,
        **survey_kwargs: Any,
    ) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        self.key = key
        self.default = Path(default).expanduser() if default is not None else None
        self.survey_kwargs = dict(survey_kwargs)
        self.survey = None
        self.widget = None
        self._chooser = None
        self._output = None
        self._status = None

        if self.path is not None:
            self.refresh(self.path)
        else:
            self._init_chooser()

    @property
    def folder(self) -> Path | None:
        """Current surveyed folder, if one has been selected."""
        if self.survey is not None:
            return self.survey.folder
        return self.path

    @property
    def items(self) -> list[Any]:
        """Surveyed file items."""
        return [] if self.survey is None else self.survey.items

    @property
    def inventory_rows(self) -> list[dict[str, Any]]:
        """JSON-serializable inventory rows."""
        return [] if self.survey is None else self.survey.inventory_rows

    def selected(self, kind: str | None = None) -> list[Any]:
        """Return selected survey items."""
        return [] if self.survey is None else self.survey.selected(kind)

    def paths(self, kind: str | None = None) -> list[Path]:
        """Return selected file paths."""
        return [] if self.survey is None else self.survey.paths(kind)

    def selected_paths(self, kind: str | None = None) -> list[Path]:
        """Compatibility alias for :meth:`paths`."""
        return self.paths(kind)

    def selected_folders(self) -> list[Path]:
        """Return parent folders containing selected files."""
        return sorted({path.parent for path in self.paths()})

    def selection(self) -> dict[str, Any]:
        """Return the current JSON-serializable selection snapshot."""
        if self.survey is None:
            return {
                "folder": None if self.folder is None else str(self.folder),
                "selected_files": [],
                "selected_folders": [],
            }
        data = dict(self.survey.selection())
        data["selected_folders"] = [str(path) for path in self.selected_folders()]
        return data

    def save(self, path: str | Path | None = None) -> Path:
        """Write the current selection JSON."""
        if self.survey is None:
            raise ValueError("No folder has been surveyed yet.")
        return self.survey.save(path)

    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        """Load selection JSON into the current survey."""
        if self.survey is None:
            raise ValueError("No folder has been surveyed yet.")
        return self.survey.load(path)

    def refresh(self, path: str | Path | None = None) -> "ShowFolder":
        """Run or rerun the microscopy folder survey."""
        from quantem.widget.survey import survey

        if path is not None:
            self.path = Path(path).expanduser()
        if self.path is None:
            raise ValueError("No folder path is selected.")
        if self._chooser is not None:
            from quantem.widget.folder_picker import _load_cache, _save_cache

            cache = _load_cache()
            cache[self.key] = str(self.path)
            _save_cache(cache)
        self.survey = survey(self.path, **self.survey_kwargs)
        self.widget = self.survey.widget
        if self._status is not None:
            self._status.value = (
                "<b>Selected folder:</b> "
                f"<code>{html.escape(str(self.survey.folder))}</code>"
            )
        return self

    def _init_chooser(self) -> None:
        from quantem.widget.folder_picker import _load_cache
        from ipyfilechooser import FileChooser
        from ipywidgets import Button, HTML, Output, VBox

        cache = _load_cache()
        cached = cache.get(self.key)
        if cached and Path(cached).is_dir():
            start = Path(cached)
        elif self.default is not None and self.default.is_dir():
            start = self.default
        else:
            start = Path.home()
        chooser = FileChooser(
            str(start),
            show_only_dirs=True,
            title=f"Select folder ({self.key})",
        )
        status = HTML("<b>Select a folder to survey.</b>")
        output = Output()
        refresh = Button(description="Survey folder", tooltip="Run ShowFolder on the selected folder")

        def _selected_path() -> Path | None:
            selected = getattr(chooser, "selected", None)
            return Path(selected).expanduser() if selected else None

        def _run(_=None) -> None:
            selected = _selected_path()
            if selected is None:
                status.value = "<b>Select a folder first.</b>"
                return
            output.clear_output(wait=True)
            with output:
                from IPython.display import display

                self.refresh(selected)
                display(self.survey.widget)

        chooser.register_callback(lambda _: _run())
        refresh.on_click(_run)
        self._chooser = chooser
        self._output = output
        self._status = status
        self.widget = VBox([status, chooser, refresh, output])

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
    **survey_kwargs: Any,
) -> ShowFolder:
    """Open a microscopy folder survey widget."""
    return ShowFolder(path, key=key, default=default, **survey_kwargs)
