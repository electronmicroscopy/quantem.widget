"""Browse: File/folder browser widget for microscope data workflows."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import _resize_image
from quantem.widget.json_state import save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import bind_tool_runtime_api, normalize_tool_groups

_BROWSE_ESM = pathlib.Path(__file__).parent / "static" / "browse.js"
_BROWSE_CSS = pathlib.Path(__file__).parent / "static" / "browse.css"

_PREVIEWABLE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".emd", ".dm3", ".dm4"}
_MAX_ENTRIES = 2000
_PREVIEW_MAX_PX = 256


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / (1024 * 1024 * 1024):.1f} GB"


class Browse(anywidget.AnyWidget):
    """
    File/folder browser for selecting microscope data in Jupyter.

    Parameters
    ----------
    root : str or Path, default "."
        Starting directory.
    title : str, default "Browse"
        Widget header title.
    selection_mode : {"file", "files", "folder", "folders"}, default "files"
        What can be selected. "file"/"folder" = single, "files"/"folders" = multi.
    filter_exts : list of str, optional
        Extension filter (e.g. [".h5", ".tif"]).
    show_hidden : bool, default False
        Show dot-files.
    sort_key : {"name", "size", "modified"}, default "name"
        Sort criterion.
    sort_asc : bool, default True
        Sort ascending.
    state : dict or str or Path, optional
        Restore state from dict or JSON file.
    """

    _esm = _BROWSE_ESM if _BROWSE_ESM.exists() else "export function render() {}"
    _css = _BROWSE_CSS if _BROWSE_CSS.exists() else ""

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    root = traitlets.Unicode("").tag(sync=True)
    current_path = traitlets.Unicode("").tag(sync=True)
    entries_json = traitlets.Unicode("[]").tag(sync=True)
    selected = traitlets.List(traitlets.Unicode()).tag(sync=True)
    selection_mode = traitlets.Unicode("file").tag(sync=True)
    filter_exts = traitlets.List(traitlets.Unicode()).tag(sync=True)
    sort_key = traitlets.Unicode("name").tag(sync=True)
    sort_asc = traitlets.Bool(True).tag(sync=True)
    show_hidden = traitlets.Bool(False).tag(sync=True)
    title = traitlets.Unicode("Browse").tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)

    preview_request = traitlets.Unicode("").tag(sync=True)
    preview_bytes = traitlets.Bytes(b"").tag(sync=True)
    preview_width = traitlets.Int(0).tag(sync=True)
    preview_height = traitlets.Int(0).tag(sync=True)
    preview_title = traitlets.Unicode("").tag(sync=True)
    preview_info = traitlets.Unicode("").tag(sync=True)
    preview_error = traitlets.Unicode("").tag(sync=True)

    available_exts = traitlets.List(traitlets.Unicode()).tag(sync=True)

    preview_cmap = traitlets.Unicode("inferno").tag(sync=True)
    is_scanning = traitlets.Bool(False).tag(sync=True)

    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    def __init__(
        self,
        root: str | pathlib.Path = ".",
        *,
        title: str = "Browse",
        selection_mode: str = "files",
        filter_exts: list[str] | None = None,
        show_hidden: bool = False,
        sort_key: str = "name",
        sort_asc: bool = True,
        state: dict | str | pathlib.Path | None = None,
        **kwargs,
    ):
        from quantem.widget.json_state import resolve_widget_version
        super().__init__(**kwargs)
        resolved = pathlib.Path(root).expanduser().resolve()
        self.root = str(resolved)
        self.current_path = str(resolved)
        self.title = title
        self.selection_mode = selection_mode
        self.filter_exts = list(filter_exts) if filter_exts else []
        self.show_hidden = show_hidden
        self.sort_key = sort_key
        self.sort_asc = sort_asc
        self.widget_version = resolve_widget_version()
        self._scan_directory()
        bind_tool_runtime_api(Browse, "Browse")
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = json.loads(pathlib.Path(state).read_text())
            state = unwrap_state_payload(state)
            self.load_state_dict(state)

    @traitlets.observe("current_path")
    def _on_current_path_change(self, change):
        root = pathlib.Path(self.root).resolve()
        try:
            target = pathlib.Path(change["new"]).resolve()
        except (OSError, ValueError):
            self.current_path = str(root)
            return
        if not str(target).startswith(str(root)):
            self.current_path = str(root)
            return
        if not target.is_dir():
            self.current_path = change["old"] if change["old"] else str(root)
            return
        self._scan_directory()

    @traitlets.observe("filter_exts", "show_hidden", "sort_key", "sort_asc")
    def _on_filter_change(self, change):
        self._scan_directory()

    @traitlets.observe("preview_request")
    def _on_preview_request(self, change):
        path = change["new"]
        if not path:
            self.preview_bytes = b""
            self.preview_width = 0
            self.preview_height = 0
            self.preview_title = ""
            self.preview_info = ""
            self.preview_error = ""
            return
        self._load_preview(path)

    def _scan_directory(self):
        self.is_scanning = True
        try:
            current = pathlib.Path(self.current_path)
            if not current.is_dir():
                self.entries_json = "[]"
                self.available_exts = []
                return
            entries = []
            all_exts: set[str] = set()
            filter_set = {e.lower() for e in self.filter_exts} if self.filter_exts else set()
            try:
                items = list(current.iterdir())
            except PermissionError:
                self.entries_json = "[]"
                self.available_exts = []
                return
            for item in items:
                name = item.name
                if not self.show_hidden and name.startswith("."):
                    continue
                is_dir = item.is_dir()
                ext = item.suffix.lower() if not is_dir else ""
                if ext:
                    all_exts.add(ext)
                if filter_set and not is_dir and ext not in filter_set:
                    continue
                try:
                    stat = item.stat()
                    size = stat.st_size if not is_dir else 0
                    modified = stat.st_mtime
                except (OSError, PermissionError):
                    size = 0
                    modified = 0.0
                entries.append({
                    "name": name,
                    "path": str(item),
                    "kind": "folder" if is_dir else "file",
                    "size": size,
                    "size_str": _format_size(size) if not is_dir else "",
                    "ext": ext,
                    "modified": modified,
                    "is_previewable": ext in _PREVIEWABLE_EXTS,
                })
            if self.sort_key == "size":
                entries.sort(key=lambda e: (e["kind"] != "folder", e["size"] if self.sort_asc else -e["size"]))
            elif self.sort_key == "modified":
                entries.sort(key=lambda e: (e["kind"] != "folder", e["modified"] if self.sort_asc else -e["modified"]))
            else:
                entries.sort(key=lambda e: (e["kind"] != "folder", e["name"].lower()))
            if not self.sort_asc and self.sort_key == "name":
                folders = [e for e in entries if e["kind"] == "folder"]
                files = [e for e in entries if e["kind"] == "file"]
                folders.sort(key=lambda e: e["name"].lower(), reverse=True)
                files.sort(key=lambda e: e["name"].lower(), reverse=True)
                entries = folders + files
            entries = entries[:_MAX_ENTRIES]
            self.entries_json = json.dumps(entries)
            self.available_exts = sorted(all_exts)
        finally:
            self.is_scanning = False

    def _load_preview(self, path_str: str):
        try:
            path = pathlib.Path(path_str).resolve()
            root = pathlib.Path(self.root).resolve()
            if not str(path).startswith(str(root)):
                self.preview_error = "Path outside root"
                self.preview_bytes = b""
                return
            if not path.is_file():
                self.preview_error = "Not a file"
                self.preview_bytes = b""
                return
            ext = path.suffix.lower()
            if ext == ".npy":
                arr = np.load(str(path))
                if arr.ndim > 2:
                    arr = arr[0] if arr.ndim == 3 else arr[0, 0]
                arr = arr.astype(np.float32)
            elif ext in {".h5", ".hdf5"}:
                arr = self._load_h5_preview(path)
                if arr is None:
                    return
            elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                from PIL import Image
                img = Image.open(str(path))
                if img.mode not in ("L", "F"):
                    img = img.convert("L")
                arr = np.array(img, dtype=np.float32)
            else:
                self.preview_error = f"Unsupported format: {ext}"
                self.preview_bytes = b""
                return
            h, w = arr.shape
            if max(h, w) > _PREVIEW_MAX_PX:
                scale = _PREVIEW_MAX_PX / max(h, w)
                new_h = max(1, int(h * scale))
                new_w = max(1, int(w * scale))
                arr = _resize_image(arr, new_h, new_w)
            file_size = path.stat().st_size
            self.preview_bytes = arr.astype(np.float32).tobytes()
            self.preview_width = arr.shape[1]
            self.preview_height = arr.shape[0]
            self.preview_title = path.name
            self.preview_info = f"{h}×{w} · {_format_size(file_size)}"
            self.preview_error = ""
        except Exception as exc:
            self.preview_error = str(exc)
            self.preview_bytes = b""
            self.preview_width = 0
            self.preview_height = 0

    def _load_h5_preview(self, path: pathlib.Path) -> np.ndarray | None:
        try:
            import h5py
        except ImportError:
            self.preview_error = "h5py not installed"
            self.preview_bytes = b""
            return None
        try:
            import hdf5plugin  # noqa: F401 — enables bitshuffle+LZ4 (arina)
        except ImportError:
            pass
        try:
            with h5py.File(str(path), "r") as f:
                # Find first 2D+ dataset by walking the file
                arr = None
                def _find_dataset(name, obj):
                    nonlocal arr
                    if arr is not None:
                        return
                    if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                        # Take a 2D slice from the dataset
                        if obj.ndim == 2:
                            arr = obj[()]
                        elif obj.ndim == 3:
                            arr = obj[0]
                        elif obj.ndim >= 4:
                            idx = tuple(0 for _ in range(obj.ndim - 2))
                            arr = obj[idx]
                f.visititems(_find_dataset)
                if arr is None:
                    self.preview_error = "No 2D dataset found"
                    self.preview_bytes = b""
                    return None
                return arr.astype(np.float32)
        except Exception as exc:
            self.preview_error = f"HDF5 error: {exc}"
            self.preview_bytes = b""
            return None

    @property
    def path(self) -> str | None:
        return self.selected[0] if self.selected else None

    def cd(self, path: str | pathlib.Path) -> Self:
        self.current_path = str(pathlib.Path(path).expanduser().resolve())
        return self

    def filter(self, *exts: str) -> Self:
        self.filter_exts = list(exts)
        return self

    def clear_selection(self) -> Self:
        self.selected = []
        return self

    def load(self, **kwargs):
        """Load selected files/folders via IO.

        Returns IOResult for the selected path(s).
        Single .h5 with _master.h5 pattern → IO.arina_file()
        Single file → IO.file()
        Multiple files → IO.file(list)
        Single folder → IO.folder()
        """
        from quantem.widget.io import IO
        paths = list(self.selected)
        if not paths:
            raise ValueError("No files selected")
        if len(paths) == 1:
            p = pathlib.Path(paths[0])
            if p.is_dir():
                return IO.folder(str(p), **kwargs)
            if p.name.endswith("_master.h5"):
                return IO.arina_file(str(p), **kwargs)
            return IO.file(str(p), **kwargs)
        # Multiple files
        masters = [p for p in paths if p.endswith("_master.h5")]
        if masters:
            return IO.arina_file(masters, **kwargs)
        return IO.file(paths, **kwargs)

    def home(self) -> Self:
        self.current_path = self.root
        return self

    def state_dict(self) -> dict:
        return {
            "selection_mode": self.selection_mode,
            "filter_exts": list(self.filter_exts),
            "show_hidden": self.show_hidden,
            "sort_key": self.sort_key,
            "sort_asc": self.sort_asc,
            "title": self.title,
            "show_controls": self.show_controls,
            "preview_cmap": self.preview_cmap,
            "disabled_tools": list(self.disabled_tools),
            "hidden_tools": list(self.hidden_tools),
        }

    def save(self, path: str) -> None:
        save_state_file(path, "Browse", self.state_dict())

    def load_state_dict(self, state: dict) -> None:
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self) -> None:
        entries = json.loads(self.entries_json)
        n_folders = sum(1 for e in entries if e["kind"] == "folder")
        n_files = sum(1 for e in entries if e["kind"] == "file")
        print(f"Browse: {self.root}")
        print(f"  Current: {self.current_path}")
        print(f"  Mode: {self.selection_mode}")
        print(f"  Entries: {n_folders} folders, {n_files} files")
        print(f"  Selected: {len(self.selected)}")
        if self.filter_exts:
            print(f"  Filter: {', '.join(self.filter_exts)}")

    def __repr__(self) -> str:
        entries = json.loads(self.entries_json)
        return f"Browse({self.root!r}, mode={self.selection_mode}, {len(entries)} entries)"
