"""
Show3DVolume: Orthogonal slice viewer for 3D volumetric data.
Displays XY, XZ, YZ planes with interactive sliders.
All slicing happens in JavaScript for instant response.
"""
import json
import pathlib
from typing import Optional, Union
import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.json_state import build_json_header, resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)


class Show3DVolume(anywidget.AnyWidget):
    """
    3D volume viewer with three orthogonal slice planes.

    Parameters
    ----------
    data : array_like
        3D array of shape (nz, ny, nx).
    title : str, optional
        Title displayed above the viewer.
    cmap : str, default "inferno"
        Colormap name.
    pixel_size : float, optional
        Pixel size in angstroms for scale bar.
    show_stats : bool, default True
        Show per-slice statistics.
    log_scale : bool, default False
        Use log scale for intensity mapping.
    auto_contrast : bool, default False
        Use percentile-based contrast.
    disabled_tools : list of str, optional
        Tool groups to lock while still showing controls. Supported:
        ``"display"``, ``"histogram"``, ``"playback"``, ``"fft"``,
        ``"navigation"``, ``"stats"``, ``"export"``, ``"view"``,
        ``"volume"``, ``"all"``
    disable_* : bool, optional
        Convenience flags mirroring ``disabled_tools``.
    hidden_tools : list of str, optional
        Tool groups to hide from the UI. Uses the same keys as
        ``disabled_tools``.
    hide_* : bool, optional
        Convenience flags mirroring ``disable_*`` for ``hidden_tools``.
    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show3DVolume
    >>> volume = np.random.rand(64, 64, 64).astype(np.float32)
    >>> Show3DVolume(volume, title="My Volume", cmap="viridis")
    """
    _esm = pathlib.Path(__file__).parent / "static" / "show3dvolume.js"
    _css = pathlib.Path(__file__).parent / "static" / "show3dvolume.css"
    # Volume dimensions
    nx = traitlets.Int(1).tag(sync=True)
    ny = traitlets.Int(1).tag(sync=True)
    nz = traitlets.Int(1).tag(sync=True)
    # Slice positions
    slice_x = traitlets.Int(0).tag(sync=True)
    slice_y = traitlets.Int(0).tag(sync=True)
    slice_z = traitlets.Int(0).tag(sync=True)
    # Raw volume data (sent once)
    volume_bytes = traitlets.Bytes(b"").tag(sync=True)
    # Display
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    # Scale bar
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    # UI
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_crosshair = traitlets.Bool(True).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    # Axis labels (dim 0, 1, 2 → default "Z", "Y", "X")
    dim_labels = traitlets.List(traitlets.Unicode(), default_value=["Z", "Y", "X"]).tag(sync=True)
    # Stats (3 values: xy, xz, yz)
    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)
    # Playback
    playing = traitlets.Bool(False).tag(sync=True)
    reverse = traitlets.Bool(False).tag(sync=True)
    boomerang = traitlets.Bool(False).tag(sync=True)
    fps = traitlets.Float(5.0).tag(sync=True)
    loop = traitlets.Bool(True).tag(sync=True)
    play_axis = traitlets.Int(0).tag(sync=True)  # 0=Z, 1=Y, 2=X, 3=All
    # Export
    _export_axis = traitlets.Int(0).tag(sync=True)  # 0=Z, 1=Y, 2=X
    _gif_export_requested = traitlets.Bool(False).tag(sync=True)
    _gif_data = traitlets.Bytes(b"").tag(sync=True)
    _gif_metadata_json = traitlets.Unicode("").tag(sync=True)
    _zip_export_requested = traitlets.Bool(False).tag(sync=True)
    _zip_data = traitlets.Bytes(b"").tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups):
        return normalize_tool_groups("Show3DVolume", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_playback: bool = False,
        disable_fft: bool = False,
        disable_navigation: bool = False,
        disable_stats: bool = False,
        disable_export: bool = False,
        disable_view: bool = False,
        disable_volume: bool = False,
        disable_all: bool = False,
    ):
        return build_tool_groups(
            "Show3DVolume",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "histogram": disable_histogram,
                "playback": disable_playback,
                "fft": disable_fft,
                "navigation": disable_navigation,
                "stats": disable_stats,
                "export": disable_export,
                "view": disable_view,
                "volume": disable_volume,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_playback: bool = False,
        hide_fft: bool = False,
        hide_navigation: bool = False,
        hide_stats: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_volume: bool = False,
        hide_all: bool = False,
    ):
        return build_tool_groups(
            "Show3DVolume",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "histogram": hide_histogram,
                "playback": hide_playback,
                "fft": hide_fft,
                "navigation": hide_navigation,
                "stats": hide_stats,
                "export": hide_export,
                "view": hide_view,
                "volume": hide_volume,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    def __init__(
        self,
        data: Union[np.ndarray, "torch.Tensor"],
        title: str = "",
        cmap: str = "inferno",
        pixel_size: float = 0.0,
        scale_bar_visible: bool = True,
        show_controls: bool = True,
        show_stats: bool = True,
        show_crosshair: bool = True,
        show_fft: bool = False,
        log_scale: bool = False,
        auto_contrast: bool = False,
        disabled_tools: list[str] | None = None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_playback: bool = False,
        disable_fft: bool = False,
        disable_navigation: bool = False,
        disable_stats: bool = False,
        disable_export: bool = False,
        disable_view: bool = False,
        disable_volume: bool = False,
        disable_all: bool = False,
        hidden_tools: list[str] | None = None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_playback: bool = False,
        hide_fft: bool = False,
        hide_navigation: bool = False,
        hide_stats: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_volume: bool = False,
        hide_all: bool = False,
        fps: float = 5.0,
        dim_labels: Optional[list] = None,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()
        self.fps = fps
        if dim_labels is not None:
            self.dim_labels = dim_labels

        # Check if data is a Dataset3d and extract metadata
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            if not title and data.name:
                title = data.name
            if pixel_size == 0.0 and hasattr(data, "units"):
                units = list(data.units)
                sampling_val = float(data.sampling[-1])
                if units[-1] in ("nm",):
                    pixel_size = sampling_val * 10  # nm → Å
                elif units[-1] in ("Å", "angstrom", "A"):
                    pixel_size = sampling_val
            data = data.array

        data = to_numpy(data)
        if data.ndim != 3:
            raise ValueError(f"Show3DVolume requires 3D data, got {data.ndim}D")
        self._data = data.astype(np.float32)
        self.nz, self.ny, self.nx = self._data.shape
        # Default to middle slices
        self.slice_z = self.nz // 2
        self.slice_y = self.ny // 2
        self.slice_x = self.nx // 2
        self.title = title
        self.cmap = cmap
        self.pixel_size = pixel_size
        self.scale_bar_visible = scale_bar_visible
        self.show_controls = show_controls
        self.show_stats = show_stats
        self.show_crosshair = show_crosshair
        self.show_fft = show_fft
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_histogram=disable_histogram,
            disable_playback=disable_playback,
            disable_fft=disable_fft,
            disable_navigation=disable_navigation,
            disable_stats=disable_stats,
            disable_export=disable_export,
            disable_view=disable_view,
            disable_volume=disable_volume,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_histogram=hide_histogram,
            hide_playback=hide_playback,
            hide_fft=hide_fft,
            hide_navigation=hide_navigation,
            hide_stats=hide_stats,
            hide_view=hide_view,
            hide_export=hide_export,
            hide_volume=hide_volume,
            hide_all=hide_all,
        )
        self._compute_stats()
        self.volume_bytes = self._data.tobytes()
        self.observe(self._on_slice_change, names=["slice_x", "slice_y", "slice_z"])
        self.observe(self._on_gif_export, names=["_gif_export_requested"])
        self.observe(self._on_zip_export, names=["_zip_export_requested"])

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    def set_image(self, data):
        """Replace the volume data. Preserves all display settings."""
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            data = data.array
        data = to_numpy(data)
        if data.ndim != 3:
            raise ValueError(f"Show3DVolume requires 3D data, got {data.ndim}D")
        self._data = data.astype(np.float32)
        self.nz, self.ny, self.nx = self._data.shape
        self.slice_z = min(self.slice_z, self.nz - 1)
        self.slice_y = min(self.slice_y, self.ny - 1)
        self.slice_x = min(self.slice_x, self.nx - 1)
        self._compute_stats()
        self.volume_bytes = self._data.tobytes()

    def __repr__(self) -> str:
        return f"Show3DVolume({self.nz}×{self.ny}×{self.nx}, slices=({self.slice_z},{self.slice_y},{self.slice_x}), cmap={self.cmap})"

    def state_dict(self):
        return {
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "show_crosshair": self.show_crosshair,
            "show_fft": self.show_fft,
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
            "pixel_size": self.pixel_size,
            "scale_bar_visible": self.scale_bar_visible,
            "slice_x": self.slice_x,
            "slice_y": self.slice_y,
            "slice_z": self.slice_z,
            "fps": self.fps,
            "loop": self.loop,
            "reverse": self.reverse,
            "boomerang": self.boomerang,
            "play_axis": self.play_axis,
            "dim_labels": self.dim_labels,
        }

    def save(self, path: str):
        save_state_file(path, "Show3DVolume", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            if key == "pixel_size_angstrom":
                key = "pixel_size"
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        lines = [self.title or "Show3DVolume", "═" * 32]
        lines.append(f"Volume:   {self.nz}×{self.ny}×{self.nx}")
        if self.pixel_size > 0:
            ps = self.pixel_size
            if ps >= 10:
                lines[-1] += f" ({ps / 10:.2f} nm/px)"
            else:
                lines[-1] += f" ({ps:.2f} Å/px)"
        labels = self.dim_labels
        lines.append(f"Slices:   {labels[0]}={self.slice_z}  {labels[1]}={self.slice_y}  {labels[2]}={self.slice_x}")
        if hasattr(self, "_data") and self._data is not None:
            arr = self._data
            lines.append(f"Data:     min={float(arr.min()):.4g}  max={float(arr.max()):.4g}  mean={float(arr.mean()):.4g}")
        cmap = self.cmap
        scale = "log" if self.log_scale else "linear"
        contrast = "auto contrast" if self.auto_contrast else "manual contrast"
        display = f"{cmap} | {contrast} | {scale}"
        if self.show_fft:
            display += " | FFT"
        lines.append(f"Display:  {display}")
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")
        print("\n".join(lines))

    def _compute_stats(self):
        """Compute statistics for the 3 current slices."""
        slices = [
            self._data[self.slice_z, :, :],
            self._data[:, self.slice_y, :],
            self._data[:, :, self.slice_x],
        ]
        with self.hold_sync():
            self.stats_mean = [float(np.mean(s)) for s in slices]
            self.stats_min = [float(np.min(s)) for s in slices]
            self.stats_max = [float(np.max(s)) for s in slices]
            self.stats_std = [float(np.std(s)) for s in slices]

    def _on_slice_change(self, change):
        self._compute_stats()

    def play(self):
        self.playing = True

    def pause(self):
        self.playing = False

    def stop(self):
        self.playing = False
        self.slice_z = self.nz // 2
        self.slice_y = self.ny // 2
        self.slice_x = self.nx // 2

    def _on_gif_export(self, change=None):
        if not self._gif_export_requested:
            return
        self._gif_export_requested = False
        self._generate_gif()

    def _on_zip_export(self, change=None):
        if not self._zip_export_requested:
            return
        self._zip_export_requested = False
        self._generate_zip()

    def _get_export_slices(self):
        axis = self._export_axis
        if axis == 0:
            return [self._data[z, :, :] for z in range(self.nz)]
        elif axis == 1:
            return [self._data[:, y, :] for y in range(self.ny)]
        else:
            return [self._data[:, :, x] for x in range(self.nx)]

    def _normalize_slice(self, slc: np.ndarray) -> np.ndarray:
        if self.log_scale:
            slc = np.log1p(np.maximum(slc, 0))
        if self.auto_contrast:
            vmin = float(np.percentile(slc, 2))
            vmax = float(np.percentile(slc, 98))
        else:
            vmin = float(slc.min())
            vmax = float(slc.max())
        if vmax > vmin:
            return np.clip((slc - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        return np.zeros(slc.shape, dtype=np.uint8)

    def _generate_gif(self):
        import io
        from matplotlib import colormaps
        from PIL import Image

        slices = self._get_export_slices()
        cmap_fn = colormaps.get_cmap(self.cmap)
        pil_frames = []
        for slc in slices:
            normalized = self._normalize_slice(slc)
            rgba = cmap_fn(normalized / 255.0)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(rgb))
        if not pil_frames:
            with self.hold_sync():
                self._gif_data = b""
                self._gif_metadata_json = ""
            return
        buf = io.BytesIO()
        duration_ms = int(1000 / max(0.1, self.fps))
        pil_frames[0].save(buf, format="GIF", save_all=True, append_images=pil_frames[1:], duration=duration_ms, loop=0)
        metadata = {
            **build_json_header("Show3DVolume"),
            "format": "gif",
            "export_kind": "animated_slices",
            "export_axis": int(self._export_axis),
            "n_slices": int(len(pil_frames)),
            "duration_ms": int(duration_ms),
            "display": {
                "cmap": self.cmap,
                "log_scale": bool(self.log_scale),
                "auto_contrast": bool(self.auto_contrast),
            },
        }
        with self.hold_sync():
            self._gif_metadata_json = json.dumps(metadata, indent=2)
            self._gif_data = buf.getvalue()

    def _generate_zip(self):
        import io
        import zipfile
        from matplotlib import colormaps
        from PIL import Image

        slices = self._get_export_slices()
        cmap_fn = colormaps.get_cmap(self.cmap)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            metadata = {
                **build_json_header("Show3DVolume"),
                "format": "zip",
                "export_kind": "png_slices",
                "n_slices": int(len(slices)),
                "display": {"cmap": self.cmap, "log_scale": bool(self.log_scale)},
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
            for i, slc in enumerate(slices):
                normalized = self._normalize_slice(slc)
                rgba = cmap_fn(normalized / 255.0)
                rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                img = Image.fromarray(rgb)
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                zf.writestr(f"slice_{i:04d}.png", img_buf.getvalue())
        self._zip_data = buf.getvalue()


    def save_image(self, path: str | pathlib.Path, *, plane: str | None = None,
                   slice_idx: int | None = None, format: str | None = None,
                   dpi: int = 150) -> pathlib.Path:
        """Save a volume slice as PNG, PDF, or TIFF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        plane : str, optional
            One of 'xy', 'xz', 'yz'. Defaults to 'xy'.
        slice_idx : int, optional
            Slice index along the chosen axis. Defaults to current position.
        format : str, optional
            'png', 'pdf', or 'tiff'. If omitted, inferred from file extension.
        dpi : int, default 150
            Output DPI metadata.

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        from matplotlib import colormaps
        from PIL import Image

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        plane = (plane or "xy").lower()
        if plane == "xy":
            idx = slice_idx if slice_idx is not None else self.slice_z
            max_idx = self.nz
        elif plane == "xz":
            idx = slice_idx if slice_idx is not None else self.slice_y
            max_idx = self.ny
        elif plane == "yz":
            idx = slice_idx if slice_idx is not None else self.slice_x
            max_idx = self.nx
        else:
            raise ValueError(f"Unknown plane: {plane!r}. Use 'xy', 'xz', or 'yz'.")

        if idx < 0 or idx >= max_idx:
            raise IndexError(f"Slice index {idx} out of range [0, {max_idx}) for plane '{plane}'")

        if plane == "xy":
            slc = self._data[idx]
        elif plane == "xz":
            slc = self._data[:, idx, :]
        else:
            slc = self._data[:, :, idx]

        normalized = self._normalize_slice(slc)
        cmap_fn = colormaps.get_cmap(self.cmap)
        rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)

        img = Image.fromarray(rgba)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), dpi=(dpi, dpi))
        return path


bind_tool_runtime_api(Show3DVolume, "Show3DVolume")
