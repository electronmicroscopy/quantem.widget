"""Folder survey workflow for microscopy images and EDS spectrum images.

The public entry point is :func:`survey`: point it at a folder of Velox EMD
files and get a compact inventory, a Show2D HAADF/STEM thumbnail gallery, and
one lazy ShowEDS explorer per EDS spectrum-image file.
"""
from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


_DEFAULT_ELEMENTS = ("Si K", "O K", "In L", "As K", "Al K", "Au M", "C K", "Cu K")
_MAG_RE = re.compile(r"HAADF\s*(.*?)\s*Nano", re.IGNORECASE)
_FALLBACK_MAG_RE = re.compile(r"(\d+(?:\.\d+)?\s*[kmg]?x)", re.IGNORECASE)


@dataclass(frozen=True)
class SurveyItem:
    """One file in a microscopy folder survey."""

    path: Path
    file_id: str
    magnification: str | None
    shape: tuple[int, int] | None
    sampling: tuple[float, float] | None
    units: tuple[str, str] | None
    thumbnail_downsample: int | None
    scan_rotation_deg: float | None
    is_eds: bool
    stage_position_m: tuple[float, float, float] | None = None
    field_of_view_nm: tuple[float, float] | None = None
    fov_group: str | None = None
    error: str | None = None

    @property
    def label(self) -> str:
        parts = [self.file_id]
        if self.magnification:
            parts.append(self.magnification)
        if self.scan_rotation_deg is not None:
            parts.append(f"{self.scan_rotation_deg:.1f} deg")
        if self.thumbnail_downsample and self.thumbnail_downsample > 1:
            parts.append(f"downsample {self.thumbnail_downsample}x")
        if self.is_eds:
            parts.append("EDS")
        return " | ".join(parts)


class FolderSurvey:
    """Rendered folder survey.

    ``FolderSurvey`` behaves like a widget in Jupyter by delegating its rich
    display to an ``ipywidgets.VBox``. The component widgets are also exposed for
    generated notebooks: ``inventory``, ``gallery``, and ``eds_widgets``.
    """

    def __init__(
        self,
        *,
        folder: Path,
        items: list[SurveyItem],
        inventory: Any,
        gallery: Any | None,
        image_galleries: list[tuple[Any, list[SurveyItem]]],
        fov_groups: list[list[SurveyItem]],
        eds_widgets: list[Any],
        eds_selection_controls: dict[str, Any],
        widget: Any,
        thumb: int,
        glob: str,
        group_view: str,
    ) -> None:
        self.folder = folder
        self.items = items
        self.inventory = inventory
        self.gallery = gallery
        self.image_galleries = image_galleries
        self.fov_groups = fov_groups
        self.eds_widgets = eds_widgets
        self.eds_selection_controls = eds_selection_controls
        self._eds_selected_ids: set[str] = {
            file_id for file_id, control in eds_selection_controls.items()
            if bool(getattr(control, "value", False))
        }
        self.widget = widget
        self.thumb = thumb
        self.glob = glob
        self.group_view = group_view
        self.selection_panel = None

        for file_id, control in eds_selection_controls.items():
            control.observe(self._make_eds_selection_observer(file_id), names="value")

    @property
    def image_items(self) -> list[SurveyItem]:
        return [item for item in self.items if not item.is_eds and item.error is None]

    @property
    def eds_items(self) -> list[SurveyItem]:
        return [item for item in self.items if item.is_eds and item.error is None]

    @property
    def selection_file(self) -> Path:
        """Default JSON path for reversible survey curation state."""
        return self.folder / ".quantem-survey.json"

    @property
    def selection_path(self) -> Path:
        """Compatibility alias for :attr:`selection_file`."""
        return self.selection_file

    @property
    def inventory_rows(self) -> list[dict[str, Any]]:
        rows = []
        for item in self.items:
            rows.append({
                "id": item.file_id,
                "file": item.path.name,
                "kind": "EDS" if item.is_eds else "image",
                "magnification": item.magnification or "",
                "shape": "" if item.shape is None else f"{item.shape[0]}x{item.shape[1]}",
                "downsample": item.thumbnail_downsample,
                "pixel_size": _format_sampling(item),
                "scan_rotation_deg": item.scan_rotation_deg,
                "fov_group": item.fov_group or "",
                "status": item.error or "ok",
            })
        return rows

    @property
    def inventory_df(self):
        import pandas as pd

        return pd.DataFrame(self.inventory_rows)

    def selected_image_items(self) -> list[SurveyItem]:
        """Return image items starred in the survey image widgets."""
        selected: dict[str, SurveyItem] = {}
        galleries = self.image_galleries or ([(self.gallery, self.image_items)] if self.gallery is not None else [])
        for gallery, image_items in galleries:
            starred = list(getattr(gallery, "starred", []) or [])
            if _is_show3d_stack(gallery):
                frame = int(starred[0]) if starred else -1
                if 0 <= frame < len(image_items):
                    selected[image_items[frame].file_id] = image_items[frame]
            else:
                for i, item in enumerate(image_items):
                    if i < len(starred) and bool(starred[i]):
                        selected[item.file_id] = item
        return [item for item in self.items if item.file_id in selected and not item.is_eds and item.error is None]

    def selected_eds_items(self) -> list[SurveyItem]:
        """Return EDS items checked for downstream use."""
        selected = set(self._eds_selected_ids)
        return [item for item in self.eds_items if item.file_id in selected]

    def selected(self, kind: str | None = None) -> list[SurveyItem]:
        """Return selected items, optionally filtered by ``"image"`` or ``"eds"``."""
        if kind is None:
            selected_ids = {item.file_id for item in self.selected_image_items()}
            selected_ids.update(item.file_id for item in self.selected_eds_items())
            return [item for item in self.items if item.file_id in selected_ids and item.error is None]
        elif kind.lower() == "image":
            return self.selected_image_items()
        elif kind.lower() == "eds":
            return self.selected_eds_items()
        else:
            raise ValueError("kind must be None, 'image', or 'eds'")

    def selected_items(self) -> list[SurveyItem]:
        """Compatibility alias for ``selected()``."""
        return self.selected()

    def paths(self, kind: str | None = None) -> list[Path]:
        """Return selected file paths, optionally filtered by ``"image"`` or ``"eds"``."""
        items = self.selected(kind)
        return [item.path for item in items]

    def selected_paths(self, kind: str | None = None) -> list[Path]:
        """Compatibility alias for ``paths(kind)``."""
        return self.paths(kind)

    def selection(self) -> dict[str, Any]:
        """Return a JSON-serializable selection snapshot for pipeline handoff."""
        selected_images = self.selected_image_items()
        selected_eds = self.selected_eds_items()
        hidden_ids = []
        galleries = self.image_galleries or ([(self.gallery, self.image_items)] if self.gallery is not None else [])
        for gallery, image_items in galleries:
            for idx in getattr(gallery, "hidden_panels", []) or []:
                if 0 <= int(idx) < len(image_items):
                    hidden_ids.append(image_items[int(idx)].file_id)
        return {
            "folder": str(self.folder),
            "glob": self.glob,
            "thumb": self.thumb,
            "selected_image_ids": [item.file_id for item in selected_images],
            "selected_eds_ids": [item.file_id for item in selected_eds],
            "hidden_image_ids": hidden_ids,
            "selected_files": [
                {"id": item.file_id, "kind": "EDS" if item.is_eds else "image", "path": str(item.path.relative_to(self.folder))}
                for item in self.selected()
            ],
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Write selected files to JSON without modifying raw microscopy files."""
        out = Path(path).expanduser() if path is not None else self.selection_file
        out.write_text(json.dumps(self.selection(), indent=2) + "\n")
        return out

    def save_selection(self, path: str | Path | None = None) -> Path:
        """Compatibility alias for ``save(path)``."""
        return self.save(path)

    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        """Restore selected files from a saved survey JSON file."""
        source = Path(path).expanduser() if path is not None else self.selection_file
        data = json.loads(source.read_text())
        image_ids = set(data.get("selected_image_ids", []))
        for gallery, image_items in self.image_galleries:
            panels = [idx for idx, item in enumerate(image_items) if item.file_id in image_ids]
            if _is_show3d_stack(gallery):
                if panels:
                    gallery.star_panel(0, frame=panels[0])
                else:
                    gallery.unstar_panel(0)
            else:
                gallery.set_starred_panels(panels)
        self._eds_selected_ids = set(data.get("selected_eds_ids", []))
        for file_id, control in self.eds_selection_controls.items():
            control.value = file_id in self._eds_selected_ids
        self._refresh_selection_panel()
        return data

    def load_selection(self, path: str | Path | None = None) -> dict[str, Any]:
        """Compatibility alias for ``load(path)``."""
        return self.load(path)

    def hide_unselected(self) -> "FolderSurvey":
        """Collapse unstarred image panels in the current gallery."""
        if not self.image_galleries:
            return self
        if not self.selected_image_items():
            raise ValueError("No image panels are starred; star at least one panel first.")
        for gallery, image_items in self.image_galleries:
            if _is_show3d_stack(gallery):
                continue
            selected_ids = {item.file_id for item in self.selected_image_items()}
            to_hide = [idx for idx, item in enumerate(image_items) if item.file_id not in selected_ids]
            if len(to_hide) < len(image_items):
                gallery.set_hidden_panels(to_hide)
        self._refresh_selection_panel()
        return self

    def show_all_images(self) -> "FolderSurvey":
        """Restore every image panel in the survey gallery."""
        for gallery, _ in self.image_galleries:
            if hasattr(gallery, "show_all_panels"):
                gallery.show_all_panels()
        self._refresh_selection_panel()
        return self

    def show_selected(self):
        """Return a compact Show2D gallery containing only starred image panels."""
        from ipywidgets import HTML
        from quantem.widget import Show2D

        selected = self.selected_image_items()
        if not selected:
            return HTML("<p><b>No starred image panels yet.</b></p>")
        frames = []
        labels = []
        pixel_sizes_out = []
        pixel_unit = "pixels"
        for item in selected:
            for gallery, image_items in self.image_galleries:
                if item in image_items:
                    idx = image_items.index(item)
                    frames.append(gallery._data[idx])
                    labels.append(gallery.labels[idx])
                    if _is_show3d_stack(gallery):
                        pixel_size = float(getattr(gallery, "pixel_size", 0.0) or 0.0)
                        if pixel_size > 0:
                            pixel_sizes_out.append(pixel_size)
                            pixel_unit = getattr(gallery, "pixel_unit", "pixels")
                    else:
                        pixel_sizes = list(getattr(gallery, "pixel_sizes", []) or [])
                        if len(pixel_sizes) == len(image_items):
                            pixel_sizes_out.append(pixel_sizes[idx])
                            pixel_unit = getattr(gallery, "pixel_unit", "pixels")
                    break
        data = np.stack(frames).astype(np.float32, copy=False)
        source_gallery = self.gallery
        widget = Show2D(
            data,
            labels=labels,
            title="Selected survey images",
            gallery_gap_px=getattr(source_gallery, "gallery_gap_px", 2),
            panel_title_font_size=getattr(source_gallery, "panel_title_font_size", 9),
            save_state=False,
        )
        if len(pixel_sizes_out) == len(frames):
            widget.pixel_sizes = pixel_sizes_out
            widget.pixel_unit = pixel_unit
        return widget

    def attach_selection_panel(self) -> Any:
        """Append the live curation panel below the survey widget."""
        from ipywidgets import Button, HBox, HTML, Output, VBox

        summary = HTML()
        output = Output()
        refresh = Button(description="Refresh selection", tooltip="Read current stars/checks")
        save = Button(description="Save selection", tooltip=f"Write {self.selection_file.name}")
        hide = Button(description="Show starred only", tooltip="Hide unstarred image panels")
        show_all = Button(description="Show all images", tooltip="Restore hidden image panels")

        def _refresh(_=None):
            self._refresh_selection_panel(summary=summary, output=output)

        def _save(_=None):
            path = self.save()
            _refresh()
            with output:
                print(f"saved {path}")

        def _hide(_=None):
            try:
                self.hide_unselected()
            except Exception as exc:
                with output:
                    print(exc)
            _refresh()

        def _show_all(_=None):
            self.show_all_images()
            _refresh()

        refresh.on_click(_refresh)
        save.on_click(_save)
        hide.on_click(_hide)
        show_all.on_click(_show_all)
        panel = VBox([
            HTML("<h3 style=\"margin:12px 0 6px 0\">Selected for downstream analysis</h3>"),
            summary,
            HBox([refresh, save, hide, show_all]),
            output,
        ])
        self.selection_panel = panel
        self._selection_summary = summary
        self._selection_output = output
        for gallery, _ in self.image_galleries:
            gallery.observe(lambda _: self._refresh_selection_panel(), names="starred")
            gallery.observe(lambda _: self._refresh_selection_panel(), names="hidden_panels")
        self._refresh_selection_panel(summary=summary, output=output)
        self.widget.children = tuple(self.widget.children) + (panel,)
        return panel

    def _repr_mimebundle_(self, **kwargs):
        return self.widget._repr_mimebundle_(**kwargs)

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.widget)

    def _make_eds_selection_observer(self, file_id: str):
        def _observe(change):
            if bool(change["new"]):
                self._eds_selected_ids.add(file_id)
            else:
                self._eds_selected_ids.discard(file_id)
            self._refresh_selection_panel()
        return _observe

    def _refresh_selection_panel(self, summary=None, output=None) -> None:
        summary = summary or getattr(self, "_selection_summary", None)
        output = output or getattr(self, "_selection_output", None)
        if summary is None:
            return
        selected_images = self.selected_image_items()
        selected_eds = self.selected_eds_items()
        summary.value = (
            f"<div style=\"color:#555;margin-bottom:6px\">"
            f"{len(selected_images)} image · {len(selected_eds)} EDS selected · "
            f"default file <code>{html.escape(str(self.selection_file))}</code></div>"
        )
        if output is not None:
            output.clear_output(wait=True)
            with output:
                for item in self.selected():
                    kind = "EDS" if item.is_eds else "image"
                    print(f"{item.file_id}\t{kind}\t{item.path.name}")


def survey(
    folder: str | Path,
    *,
    glob: str = "*.emd",
    thumb: int = 512,
    eds_backend: str = "auto",
    load_eds: bool = False,
    candidate_elements: list[str] | tuple[str, ...] | None = None,
    title: str | None = None,
    group_by: str | None = "fov",
    group_view: str = "stack",
) -> FolderSurvey:
    """Survey a folder of microscopy files.

    Parameters
    ----------
    folder
        Folder containing Velox ``.emd`` files.
    glob
        File glob within ``folder``. v1 is designed around ``*.emd``.
    thumb
        Common thumbnail size for the Show2D gallery.
    eds_backend
        Backend passed to ``ShowEDS.from_emd``. Use ``"auto"`` for live Jupyter;
        use ``"stream"`` for a portable single-file HTML when the sparse stream
        fits the ShowEDS safety limit.
    load_eds
        If True, instantiate all ShowEDS widgets immediately. The default keeps
        the survey instant by rendering per-file launch buttons that create
        ShowEDS only when clicked.
    candidate_elements
        Element list passed to ShowEDS.
    title
        Optional survey title.
    group_by
        Layout grouping mode. Use ``"fov"`` to place files from the same
        microscope field of view in one row, or ``None``/``"none"`` for a flat
        gallery.
    group_view
        How grouped image frames are displayed. ``"stack"`` uses ``Show3D`` to
        flip through frames from the same field of view. ``"gallery"`` uses a
        compact ``Show2D`` row gallery.
    """
    from ipywidgets import Button, Checkbox, HBox, HTML, Output, VBox

    from quantem.widget import Show2D, Show3D, ShowEDS
    from quantem.widget.io import read_image

    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"not a folder: {root}")
    files = sorted(p for p in root.glob(glob) if p.is_file() and not p.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"no files matching {glob!r} in {root}")

    thumb = int(thumb)
    if thumb <= 0:
        raise ValueError(f"thumb must be positive, got {thumb}")
    if group_by is not None and group_by.lower() not in {"fov", "none"}:
        raise ValueError("group_by must be 'fov', 'none', or None")
    use_fov_groups = group_by is not None and group_by.lower() == "fov"
    group_view = str(group_view).lower()
    if group_view not in {"stack", "gallery"}:
        raise ValueError("group_view must be 'stack' or 'gallery'")

    items: list[SurveyItem] = []
    image_thumbnails: dict[str, np.ndarray] = {}
    image_pixel_sizes: dict[str, float] = {}
    image_pixel_units: dict[str, str] = {}
    for path in files:
        is_eds = has_eds(path)
        metadata = _file_metadata(path)
        shape = None
        sampling = None
        units = None
        thumbnail_downsample = None
        err = None
        arr = None
        try:
            if not is_eds:
                ds = read_image(path)
                arr = np.asarray(ds.array)
                shape = (int(arr.shape[-2]), int(arr.shape[-1]))
                sampling = _sampling_tuple(getattr(ds, "sampling", None))
                units = _units_tuple(getattr(ds, "units", None))
            else:
                shape = _spectrum_image_shape(path)
        except Exception as exc:
            err = str(exc)
        item = SurveyItem(
            path=path,
            file_id=_file_id(path),
            magnification=_magnification(path),
            shape=shape,
            sampling=sampling,
            units=units,
            thumbnail_downsample=thumbnail_downsample,
            scan_rotation_deg=_scan_rotation_from_metadata(metadata),
            is_eds=is_eds,
            stage_position_m=_stage_position_m(metadata),
            field_of_view_nm=_field_of_view_nm(metadata),
            error=err,
        )
        if arr is not None and err is None and not is_eds:
            thumbnail, downsample = _thumbnail(arr, thumb)
            item = replace(item, thumbnail_downsample=downsample)
            image_thumbnails[item.file_id] = thumbnail
            if item.sampling is not None:
                image_pixel_sizes[item.file_id] = float(item.sampling[-1]) * float(downsample)
            if item.units is not None:
                image_pixel_units[item.file_id] = str(item.units[-1])
        items.append(item)
    if use_fov_groups:
        items, fov_groups = _assign_fov_groups(items)
    else:
        fov_groups = []

    heading = title or f"{root.name} survey"
    inventory = HTML(_inventory_html(items))
    children: list[Any] = [
        HTML(
            f"<h2 style=\"margin:0 0 4px 0\">{html.escape(heading)}</h2>"
            f"<div style=\"color:#555;margin-bottom:8px\">"
            f"{html.escape(str(root))} · {len(items)} files matching {html.escape(glob)!r} · "
            f"{len([item for item in items if not item.is_eds and item.error is None])} image · "
            f"{len([item for item in items if item.is_eds and item.error is None])} EDS · "
            f"thumbnail {thumb}px · scale bars use downsampled pixel size</div>"
        )
    ]
    gallery = None
    eds_widgets = []
    eds_widgets_by_id = {}
    eds_selection_controls = {}
    elements = list(candidate_elements or _DEFAULT_ELEMENTS)
    for item in items:
        if not item.is_eds or item.error is not None:
            continue
        eds_title = f"{item.file_id}"
        if item.magnification:
            eds_title += f" {item.magnification}"
        eds_title += " EDS"
        if load_eds:
            try:
                eds = ShowEDS.from_emd(
                    item.path,
                    backend=eds_backend,
                    candidate_elements=elements,
                    title=eds_title,
                )
                eds_widgets.append(eds)
                eds_widgets_by_id[item.file_id] = eds
            except Exception as exc:
                failed = HTML(
                    f"<p><b>{html.escape(item.path.name)}</b>: ShowEDS load failed: "
                    f"{html.escape(str(exc))}</p>"
                )
                eds_widgets.append(failed)
                eds_widgets_by_id[item.file_id] = failed
        else:
            launcher, control = _eds_launcher(
                item,
                backend=eds_backend,
                candidate_elements=elements,
                title=eds_title,
                ShowEDS=ShowEDS,
                Button=Button,
                Checkbox=Checkbox,
                HTML=HTML,
                Output=Output,
                VBox=VBox,
            )
            eds_widgets.append(launcher)
            eds_widgets_by_id[item.file_id] = launcher
            eds_selection_controls[item.file_id] = control

    image_items = [item for item in items if not item.is_eds and item.error is None and item.file_id in image_thumbnails]
    image_galleries: list[tuple[Any, list[SurveyItem]]] = []

    def make_gallery(image_subset: list[SurveyItem], *, title: str, ncols: int | None = None):
        return _make_show2d_gallery(
            image_subset,
            thumbnails=image_thumbnails,
            pixel_sizes=image_pixel_sizes,
            pixel_units=image_pixel_units,
            title=title,
            Show2D=Show2D,
            ncols=ncols,
        )

    def make_stack(image_subset: list[SurveyItem], *, title: str):
        return _make_show3d_stack(
            image_subset,
            thumbnails=image_thumbnails,
            pixel_sizes=image_pixel_sizes,
            pixel_units=image_pixel_units,
            title=title,
            Show3D=Show3D,
        )

    grouped_ids: set[str] = set()
    if image_items and fov_groups:
        children.append(HTML(
            "<h3 style=\"margin:8px 0 4px 0\">Field-of-view rows</h3>"
            "<div style=\"color:#555;margin-bottom:6px\">"
            "Rows combine files with matching magnification, microscope field of view, "
            "and nearby stage position.</div>"
        ))
        for group in fov_groups:
            group_images = [item for item in group if not item.is_eds and item.file_id in image_thumbnails]
            group_eds = [item for item in group if item.is_eds and item.file_id in eds_widgets_by_id]
            if not group_images and not group_eds:
                continue
            grouped_ids.update(item.file_id for item in group)
            row_children = []
            if group_images:
                if group_view == "stack":
                    row_gallery = make_stack(group_images, title=_fov_group_title(group))
                else:
                    row_gallery = make_gallery(
                        group_images,
                        title=_fov_group_title(group),
                        ncols=max(1, len(group_images)),
                    )
                gallery = gallery or row_gallery
                image_galleries.append((row_gallery, group_images))
                row_children.append(row_gallery)
            if group_eds:
                row_children.append(VBox([eds_widgets_by_id[item.file_id] for item in group_eds]))
            children.append(VBox([
                HTML(_fov_group_html(group)),
                HBox(row_children, layout={"align_items": "flex-start"}),
            ]))

    single_images = [item for item in image_items if item.file_id not in grouped_ids]
    if single_images:
        single_title = "Single image frames" if fov_groups else "HAADF/STEM gallery"
        single_gallery = make_gallery(single_images, title=single_title)
        gallery = gallery or single_gallery
        image_galleries.append((single_gallery, single_images))
        children.append(single_gallery)
    elif not image_items:
        children.append(HTML("<p><b>HAADF/STEM gallery:</b> no non-EDS image files found.</p>"))

    children.append(inventory)

    ungrouped_eds_widgets = [
        eds_widgets_by_id[item.file_id]
        for item in items
        if item.is_eds and item.error is None and item.file_id not in grouped_ids and item.file_id in eds_widgets_by_id
    ]
    if ungrouped_eds_widgets:
        children.append(HTML(
            f"<h3>EDS spectrum images ({len(ungrouped_eds_widgets)})</h3>"
            "<p>Click a file to open the lazy ShowEDS explorer. This keeps the folder survey instant.</p>"
        ))
        children.extend(ungrouped_eds_widgets)
    elif not eds_widgets:
        children.append(HTML("<p><b>EDS spectrum images:</b> none found.</p>"))

    widget = VBox(children)
    result = FolderSurvey(
        folder=root,
        items=items,
        inventory=inventory,
        gallery=gallery,
        image_galleries=image_galleries,
        fov_groups=fov_groups,
        eds_widgets=eds_widgets,
        eds_selection_controls=eds_selection_controls,
        widget=widget,
        thumb=thumb,
        glob=glob,
        group_view=group_view,
    )
    result.attach_selection_panel()
    return result


def _eds_launcher(
    item: SurveyItem,
    *,
    backend: str,
    candidate_elements: list[str],
    title: str,
    ShowEDS,
    Button,
    Checkbox,
    HTML,
    Output,
    VBox,
):
    button = Button(
        description=f"Open EDS {item.file_id}",
        tooltip=str(item.path),
        button_style="",
        layout={"width": "180px"},
    )
    select = Checkbox(
        value=False,
        description=f"Use EDS {item.file_id}",
        indent=False,
        layout={"width": "180px"},
    )
    details = HTML(
        f"<span style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif\">"
        f"<b>{html.escape(item.file_id)}</b> "
        f"{html.escape(item.magnification or '')} "
        f"{'' if item.scan_rotation_deg is None else f'{item.scan_rotation_deg:.1f} deg'} "
        f"<span style=\"color:#666\">{html.escape(item.path.name)}</span></span>"
    )
    out = Output()
    state = {"loaded": False}

    def _open(_):
        if state["loaded"]:
            return
        state["loaded"] = True
        button.description = f"EDS {item.file_id} loaded"
        button.disabled = True
        with out:
            from IPython.display import display

            try:
                display(
                    ShowEDS.from_emd(
                        item.path,
                        backend=backend,
                        candidate_elements=candidate_elements,
                        title=title,
                    )
                )
            except Exception as exc:
                display(HTML(
                    f"<p><b>{html.escape(item.path.name)}</b>: ShowEDS load failed: "
                    f"{html.escape(str(exc))}</p>"
                ))

    button.on_click(_open)
    return VBox([details, select, button, out]), select


def _make_show2d_gallery(
    image_items: list[SurveyItem],
    *,
    thumbnails: dict[str, np.ndarray],
    pixel_sizes: dict[str, float],
    pixel_units: dict[str, str],
    title: str,
    Show2D,
    ncols: int | None = None,
):
    stack = np.stack([thumbnails[item.file_id] for item in image_items]).astype(np.float32, copy=False)
    kwargs: dict[str, Any] = {}
    if ncols is not None:
        kwargs["ncols"] = int(ncols)
    gallery = Show2D(
        stack,
        labels=[item.label for item in image_items],
        title=title,
        gallery_gap_px=2,
        panel_title_font_size=9,
        save_state=False,
        **kwargs,
    )
    sizes = [pixel_sizes.get(item.file_id) for item in image_items]
    units = [pixel_units.get(item.file_id) for item in image_items]
    unique_units = {unit for unit in units if unit}
    if all(size is not None for size in sizes) and len(unique_units) == 1:
        gallery.pixel_sizes = [float(size) for size in sizes if size is not None]
        gallery.pixel_unit = next(iter(unique_units))
    elif any(size is not None for size in sizes) or unique_units:
        gallery.scale_bar_visible = False
    return gallery


def _make_show3d_stack(
    image_items: list[SurveyItem],
    *,
    thumbnails: dict[str, np.ndarray],
    pixel_sizes: dict[str, float],
    pixel_units: dict[str, str],
    title: str,
    Show3D,
):
    stack = np.stack([thumbnails[item.file_id] for item in image_items]).astype(np.float32, copy=False)
    widget = Show3D(
        stack,
        labels=[item.label for item in image_items],
        title=title,
        cmap="inferno",
        smooth=False,
        show_controls=True,
        show_stats=True,
        show_fft=False,
        dim_label="Frame",
        panel_width_px=360,
        panel_title_font_size=10,
        panel_gap=4,
        show_panel_titles=True,
        save_state=False,
    )
    sizes = [pixel_sizes.get(item.file_id) for item in image_items]
    units = [pixel_units.get(item.file_id) for item in image_items]
    unique_units = {unit for unit in units if unit}
    if sizes and all(size is not None for size in sizes) and len(unique_units) == 1:
        first = float(sizes[0])
        if all(math.isclose(first, float(size), rel_tol=1e-6, abs_tol=1e-12) for size in sizes):
            widget.pixel_size = first
            widget.pixel_unit = next(iter(unique_units))
        else:
            widget.scale_bar_visible = False
    elif any(size is not None for size in sizes) or unique_units:
        widget.scale_bar_visible = False
    return widget


def _is_show3d_stack(widget: Any) -> bool:
    return hasattr(widget, "n_slices") and hasattr(widget, "starred_frames")


def _assign_fov_groups(items: list[SurveyItem]) -> tuple[list[SurveyItem], list[list[SurveyItem]]]:
    """Assign field-of-view group labels from mag, FOV, and stage position."""
    clusters: list[list[SurveyItem]] = []
    cluster_refs: list[SurveyItem] = []
    for item in items:
        if item.error is not None or item.stage_position_m is None or item.field_of_view_nm is None:
            continue
        placed = False
        for idx, ref in enumerate(cluster_refs):
            if _same_field_of_view(ref, item):
                clusters[idx].append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
            cluster_refs.append(item)

    label_by_id: dict[str, str] = {}
    fov_groups: list[list[SurveyItem]] = []
    group_index = 1
    for cluster in clusters:
        if len(cluster) <= 1:
            continue
        label = f"FOV {group_index}"
        group_index += 1
        for item in cluster:
            label_by_id[item.file_id] = label
        fov_groups.append([replace(item, fov_group=label) for item in cluster])

    if not label_by_id:
        return items, []
    updated = [
        replace(item, fov_group=label_by_id[item.file_id])
        if item.file_id in label_by_id else item
        for item in items
    ]
    group_lookup = {item.file_id: item for item in updated}
    updated_groups = [
        [group_lookup[item.file_id] for item in group if item.file_id in group_lookup]
        for group in fov_groups
    ]
    return updated, updated_groups


def _same_field_of_view(a: SurveyItem, b: SurveyItem) -> bool:
    if a.stage_position_m is None or b.stage_position_m is None:
        return False
    if a.field_of_view_nm is None or b.field_of_view_nm is None:
        return False
    if a.magnification and b.magnification and a.magnification != b.magnification:
        return False
    if not _close_pair(a.field_of_view_nm, b.field_of_view_nm, rel_tol=0.02):
        return False
    dx = a.stage_position_m[0] - b.stage_position_m[0]
    dy = a.stage_position_m[1] - b.stage_position_m[1]
    distance = math.hypot(dx, dy)
    fov_m = min(min(a.field_of_view_nm), min(b.field_of_view_nm)) * 1e-9
    tolerance = max(5e-9, 0.6 * fov_m)
    return distance <= tolerance


def _close_pair(
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    rel_tol: float,
    abs_tol: float = 1e-12,
) -> bool:
    return (
        math.isclose(a[0], b[0], rel_tol=rel_tol, abs_tol=abs_tol)
        and math.isclose(a[1], b[1], rel_tol=rel_tol, abs_tol=abs_tol)
    )


def _fov_group_title(group: list[SurveyItem]) -> str:
    label = group[0].fov_group or "Same field of view"
    mags = sorted({item.magnification for item in group if item.magnification})
    suffix = f" · {mags[0]}" if len(mags) == 1 else ""
    return f"{label}{suffix}"


def _fov_group_html(group: list[SurveyItem]) -> str:
    first = group[0]
    fov = ""
    if first.field_of_view_nm is not None:
        fov = f" · FOV {first.field_of_view_nm[0]:.4g} x {first.field_of_view_nm[1]:.4g} nm"
    ids = ", ".join(item.file_id for item in group)
    return (
        "<div style=\"margin:8px 0 2px 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif\">"
        f"<b>{html.escape(first.fov_group or 'Same field of view')}</b>"
        f"<span style=\"color:#555\"> · files {html.escape(ids)}{html.escape(fov)}</span>"
        "</div>"
    )


def has_eds(path: str | Path) -> bool:
    """Return True when a Velox EMD file has a SpectrumStream group."""
    import h5py

    try:
        with h5py.File(path, "r") as h:
            return "Data/SpectrumStream" in h
    except OSError:
        return False


def scan_rotation_deg(path: str | Path) -> float | None:
    """Read Velox ScanRotation from Image or SpectrumImage metadata."""
    return _scan_rotation_from_metadata(_file_metadata(path))


def _file_metadata(path: str | Path) -> dict[str, Any]:
    """Read the first Velox Image or SpectrumImage metadata JSON block."""
    import h5py

    try:
        with h5py.File(path, "r") as h:
            for kind in ("Image", "SpectrumImage"):
                group = h.get(f"Data/{kind}")
                if group is None:
                    continue
                for uid in group:
                    dataset = h.get(f"Data/{kind}/{uid}/Metadata")
                    if dataset is None:
                        continue
                    return _decode_metadata_dataset(dataset[()])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return {}


def _scan_rotation_from_metadata(meta: dict[str, Any]) -> float | None:
    value = _metadata_number(meta, "Scan", "ScanRotation")
    if value is None:
        return None
    return value * 180.0 / math.pi


def _stage_position_m(meta: dict[str, Any]) -> tuple[float, float, float] | None:
    pos = _metadata_value(meta, "Stage", "Position")
    if isinstance(pos, dict):
        values = [_metadata_number(pos, key) for key in ("x", "y", "z")]
        if all(value is not None for value in values):
            return (float(values[0]), float(values[1]), float(values[2]))
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        try:
            return (float(pos[0]), float(pos[1]), float(pos[2]))
        except (TypeError, ValueError):
            return None
    values = [
        _metadata_number(meta, "Stage", key)
        for key in ("PositionX", "PositionY", "PositionZ")
    ]
    if all(value is not None for value in values):
        return (float(values[0]), float(values[1]), float(values[2]))
    return None


def _field_of_view_nm(meta: dict[str, Any]) -> tuple[float, float] | None:
    full = _metadata_value(meta, "Optics", "FullScanFieldOfView")
    if isinstance(full, dict):
        width = _metadata_number(full, "width") or _metadata_number(full, "x")
        height = _metadata_number(full, "height") or _metadata_number(full, "y")
        scalar = _metadata_number(full, "value")
        if width is not None and height is not None:
            return (float(height) * 1e9, float(width) * 1e9)
        if scalar is not None:
            value = float(scalar) * 1e9
            return (value, value)
    if isinstance(full, (list, tuple)) and len(full) >= 2:
        try:
            return (float(full[0]) * 1e9, float(full[1]) * 1e9)
        except (TypeError, ValueError):
            pass
    scalar = _metadata_number(meta, "Optics", "FullScanFieldOfView")
    if scalar is not None:
        value = float(scalar) * 1e9
        return (value, value)

    pixel = _metadata_value(meta, "BinaryResult", "PixelSize")
    scan_size = _metadata_value(meta, "Scan", "ScanSize")
    if isinstance(pixel, dict) and isinstance(scan_size, dict):
        height = _metadata_number(pixel, "height")
        width = _metadata_number(pixel, "width")
        rows = _metadata_number(scan_size, "height") or _metadata_number(scan_size, "rows")
        cols = _metadata_number(scan_size, "width") or _metadata_number(scan_size, "columns")
        if height is not None and width is not None and rows is not None and cols is not None:
            return (float(height) * float(rows) * 1e9, float(width) * float(cols) * 1e9)
    return None


def _metadata_value(meta: dict[str, Any], *path: str) -> Any:
    value: Any = meta
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, dict) and "value" in value and len(value) <= 2:
        return value["value"]
    return value


def _metadata_number(meta: dict[str, Any], *path: str) -> float | None:
    value = _metadata_value(meta, *path)
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def write_survey_notebook(
    folder: str | Path,
    out: str | Path,
    *,
    glob: str = "*.emd",
    thumb: int = 512,
    eds_backend: str = "auto",
    title: str | None = None,
    group_by: str | None = "fov",
    group_view: str = "stack",
) -> Path:
    """Write a simple survey notebook for the public survey workflow."""
    out_path = Path(out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(folder).expanduser().resolve()
    heading = title or f"{root.name} survey"
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "title",
                "metadata": {},
                "source": [
                    f"# {heading}\n",
                    "\n",
                    f"Folder: `{root}`\n",
                    "\n",
                    "Quick folder survey with **quantem.widget**: inventory table, "
                    "HAADF/STEM thumbnail gallery, and lazy EDS explorers.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "load-survey",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "from quantem.widget import survey\n",
                    "\n",
                    "survey(\n",
                    f"    {str(root)!r},\n",
                    f"    glob={glob!r},\n",
                    f"    thumb={int(thumb)!r},\n",
                    f"    eds_backend={eds_backend!r},\n",
                    f"    title={heading!r},\n",
                    f"    group_by={group_by!r},\n",
                    f"    group_view={group_view!r},\n",
                    ")\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.write_text(json.dumps(nb, indent=1))
    return out_path


def _file_id(path: Path) -> str:
    name = path.name
    if " - " in name:
        return name.split(" - ", 1)[0].strip()
    return path.stem


def _magnification(path: Path) -> str | None:
    name = path.name
    match = _MAG_RE.search(name)
    if match:
        value = " ".join(match.group(1).split())
        return value or None
    fallback = _FALLBACK_MAG_RE.search(name.replace("_", " "))
    return fallback.group(1).replace(" ", "") if fallback else None


def _decode_metadata_dataset(raw: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(raw)
    if arr.ndim > 1:
        arr = arr[:, 0]
    payload = bytes(arr).split(b"\x00", 1)[0]
    if not payload:
        return {}
    return json.loads(payload.decode("utf-8", "ignore"))


def _spectrum_image_shape(path: Path) -> tuple[int, int] | None:
    import h5py

    try:
        with h5py.File(path, "r") as h:
            group = h.get("Data/SpectrumImage")
            if group is None:
                return None
            for uid in group:
                data = h.get(f"Data/SpectrumImage/{uid}/Data")
                if data is not None and data.ndim >= 2:
                    return (int(data.shape[0]), int(data.shape[1]))
    except OSError:
        return None
    return None


def _thumbnail(arr: np.ndarray, size: int) -> tuple[np.ndarray, int]:
    frame = np.asarray(arr)
    if frame.ndim > 2:
        frame = frame[..., 0]
    frame = frame.astype(np.float32, copy=False)
    step = max(1, int(math.floor(max(frame.shape[-2:]) / size)))
    sampled = frame[::step, ::step]
    out = np.zeros((size, size), dtype=np.float32)
    h = min(size, sampled.shape[0])
    w = min(size, sampled.shape[1])
    patch = sampled[:h, :w]
    if patch.size:
        lo, hi = np.percentile(patch, (1, 99))
        if hi > lo:
            patch = np.clip((patch - lo) / (hi - lo), 0.0, 1.0)
    out[:h, :w] = patch
    return out, step


def _sampling_tuple(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            sampling = (float(value), float(value))
        else:
            values = list(value)
            if len(values) < 2:
                return None
            sampling = (float(values[-2]), float(values[-1]))
    except (TypeError, ValueError):
        return None
    if sampling[0] <= 0 or sampling[1] <= 0:
        return None
    return sampling


def _units_tuple(value: Any) -> tuple[str, str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value, value)
    try:
        values = list(value)
    except TypeError:
        return None
    if len(values) < 2:
        return None
    return (str(values[-2]), str(values[-1]))


def _format_sampling(item: SurveyItem) -> str:
    if item.sampling is None or item.units is None:
        return ""
    factor = item.thumbnail_downsample or 1
    row = item.sampling[0] * factor
    col = item.sampling[1] * factor
    row_unit, col_unit = item.units
    if math.isclose(row, col, rel_tol=1e-6, abs_tol=1e-12) and row_unit == col_unit:
        return f"{row:.4g} {row_unit}/px"
    return f"({row:.4g}, {col:.4g}) ({row_unit}, {col_unit})/px"


def _inventory_html(items: list[SurveyItem]) -> str:
    rows = []
    for item in items:
        status = item.error or "ok"
        status_color = "#b00020" if item.error else "#1b5e20"
        rot = "" if item.scan_rotation_deg is None else f"{item.scan_rotation_deg:.1f}"
        shape = "" if item.shape is None else f"{item.shape[0]}x{item.shape[1]}"
        downsample = "" if item.thumbnail_downsample in (None, 1) else f"{item.thumbnail_downsample}x"
        sampling = _format_sampling(item)
        fov_group = item.fov_group or ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.file_id)}</td>"
            f"<td>{html.escape(item.path.name)}</td>"
            f"<td>{'EDS' if item.is_eds else 'image'}</td>"
            f"<td>{html.escape(item.magnification or '')}</td>"
            f"<td>{html.escape(fov_group)}</td>"
            f"<td>{shape}</td>"
            f"<td>{downsample}</td>"
            f"<td>{html.escape(sampling)}</td>"
            f"<td>{rot}</td>"
            f"<td style=\"color:{status_color};font-weight:600\">{html.escape(status)}</td>"
            "</tr>"
        )
    return f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
  <h3 style="margin:12px 0 6px 0">Inventory</h3>
  <table style="border-collapse:collapse;font-size:13px">
    <thead>
      <tr>
        <th style="text-align:left;padding:4px 8px;border-bottom:2px solid #444">id</th>
        <th style="text-align:left;padding:4px 8px;border-bottom:2px solid #444">file</th>
        <th style="text-align:left;padding:4px 8px;border-bottom:2px solid #444">kind</th>
        <th style="text-align:left;padding:4px 8px;border-bottom:2px solid #444">mag</th>
        <th style="text-align:left;padding:4px 8px;border-bottom:2px solid #444">FOV</th>
        <th style="text-align:left;padding:4px 8px;border-bottom:2px solid #444">shape</th>
        <th style="text-align:left;padding:4px 8px;border-bottom:2px solid #444">downsample</th>
        <th style="text-align:left;padding:4px 8px;border-bottom:2px solid #444">pixel size</th>
        <th style="text-align:left;padding:4px 8px;border-bottom:2px solid #444">rot deg</th>
        <th style="text-align:left;padding:4px 8px;border-bottom:2px solid #444">status</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</div>
"""
