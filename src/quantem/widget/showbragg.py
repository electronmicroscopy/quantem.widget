"""showbragg: front end for the BraggVectors disk-detection workflow"""

import base64
import json
import pathlib
from typing import Any

import anywidget
import numpy as np
import traitlets
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.vector import Vector
from quantem.diffraction.bragg_vectors import BraggVectors
from quantem.diffraction.disk_detection import SUBPIXEL_MODES
from quantem.diffraction.strain import StrainMap

from quantem.widget.render import frame_to_rgb, rgb_to_png_bytes
from quantem.widget.utils.array import to_numpy
from quantem.widget.utils.static_fallback import StaticFallbackMixin
from quantem.widget.utils.traits import reject_unknown_kwargs
from quantem.widget.utils.ui import UiMode, resolve_ui_mode

TEMPLATE_SOURCES = ("synthetic", "data", "probe")
STAGE_STATES = ("idle", "preview", "running", "done", "error")


class ShowBragg(StaticFallbackMixin, anywidget.AnyWidget):
    """Interactive driver for the BraggVectors disk-detection and lattice-fitting workflow.

    Parameters
    ----------
    data : Dataset4dstem or BraggVectors
        A 4D-STEM dataset, or an existing BraggVectors to reopen with its
        detection kept.
    probe : array_like, optional
        Measured vacuum probe, required for template_source="probe".
    device : str, default="cpu"
        Torch device passed through to BraggVectors.
    cmap : str, default="inferno"
        Matplotlib colormap for every image panel.
    title : str, default=""
        Title above the panels. Falls back to the dataset name.
    log_scale : bool, default=False
        Log-stretch every image panel.
    ui_mode : {"interactive", "presentation", "report", "minimal"}, default="interactive"
        Display-chrome preset; explicit keywords below override it.
    show_title : bool, default=True
        Show the title row.
    show_controls : bool, default=True
        Expose the parameter controls at all.
    controls_collapsed : bool, default=False
        Start with the parameter controls hidden.
    save_state : bool, default=False
        Embed the full interactive state in the notebook.
    notebook_preview_format : {"jpeg", "webp", "png"} or None, default=None
        Static preview format used when save_state=False. None matches
        ChooseLattice, whose live widget does not hide the fallback sibling.
    notebook_preview_quality : int, default=88
        Lossy preview quality for JPEG/WebP, from 1 to 100.
    notebook_preview_max_px : int, default=512
        Longest image side for the saved-notebook preview.

    Notes
    -----
    template_radius, max_peak_shift and num_candidates use 0 as the sentinel for
    "let the data choose"; none of the three is meaningful at zero. num_candidates
    resolves to however many peaks the busiest scan position detected.

    Examples
    --------
    >>> widget = ShowBragg(dataset)
    >>> widget
    >>> widget.detect()
    >>> widget.fit()
    >>> strain = widget.strain_map()
    """

    _esm = pathlib.Path(__file__).parent / "static" / "showbragg.js"

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    scan_shape = traitlets.List(traitlets.Int(), default_value=[0, 0]).tag(sync=True)
    q_shape = traitlets.List(traitlets.Int(), default_value=[0, 0]).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)

    # Stage 1, template
    template_source = traitlets.Enum(TEMPLATE_SOURCES, default_value="synthetic").tag(sync=True)
    template_radius = traitlets.Float(0.0).tag(sync=True)
    template_edge = traitlets.Float(1.0).tag(sync=True)
    template_center = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    template_subtract_mean = traitlets.Bool(True).tag(sync=True)
    template_roi = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    template_png = traitlets.Bytes(b"").tag(sync=True)
    template_shape = traitlets.List(traitlets.Int(), default_value=[0, 0]).tag(sync=True)
    has_probe = traitlets.Bool(False).tag(sync=True)

    # Stage 2, correlation probe
    probe_position = traitlets.List(traitlets.Int(), default_value=[0, 0]).tag(sync=True)
    probe_diffraction_png = traitlets.Bytes(b"").tag(sync=True)
    probe_correlation_png = traitlets.Bytes(b"").tag(sync=True)

    # Stages 3 and 4, detection
    min_abs_intensity = traitlets.Float(0.0).tag(sync=True)
    min_spacing = traitlets.Float(0.0).tag(sync=True)
    edge_boundary = traitlets.Int(1).tag(sync=True)
    subpixel = traitlets.Enum(SUBPIXEL_MODES, default_value="upsample").tag(sync=True)
    upsample_factor = traitlets.Int(16).tag(sync=True)
    max_num_peaks = traitlets.Int(1000).tag(sync=True)
    preview_grid = traitlets.Int(8).tag(sync=True)
    preview_peaks = traitlets.Unicode("").tag(sync=True)
    detection_state = traitlets.Enum(STAGE_STATES, default_value="idle").tag(sync=True)

    # Stages 5 and 6, Bragg vector map and basis
    bvm_sampling = traitlets.Float(1.0).tag(sync=True)
    bvm_png = traitlets.Bytes(b"").tag(sync=True)
    num_candidates = traitlets.Int(0).tag(sync=True)
    candidate_min_spacing = traitlets.Float(2.0).tag(sync=True)
    candidate_min_abs_intensity = traitlets.Float(0.0).tag(sync=True)
    candidates = traitlets.List(default_value=[]).tag(sync=True)
    origin_index = traitlets.Int(-1).tag(sync=True)
    g1_index = traitlets.Int(-1).tag(sync=True)
    g2_index = traitlets.Int(-1).tag(sync=True)
    origin_rc = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    g1_rc = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    g2_rc = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)

    # Stage 7, lattice fit
    min_num_peaks = traitlets.Int(5).tag(sync=True)
    max_peak_shift = traitlets.Float(0.0).tag(sync=True)
    fit_state = traitlets.Enum(STAGE_STATES, default_value="idle").tag(sync=True)
    mask_weight_png = traitlets.Bytes(b"").tag(sync=True)
    fit_error_png = traitlets.Bytes(b"").tag(sync=True)

    _TEMPLATE_TRAITS = (
        "template_source",
        "template_radius",
        "template_edge",
        "template_center",
        "template_subtract_mean",
        "template_roi",
    )
    _CANDIDATE_TRAITS = (
        "num_candidates",
        "candidate_min_spacing",
        "candidate_min_abs_intensity",
        "origin_index",
        "g1_index",
        "g2_index",
        "origin_rc",
        "g1_rc",
        "g2_rc",
    )

    def __init__(
        self,
        data,
        *,
        probe=None,
        device: str = "cpu",
        cmap: str = "inferno",
        log_scale: bool = False,
        title: str = "",
        ui_mode: UiMode = "interactive",
        show_title: bool | None = None,
        show_controls: bool | None = None,
        controls_collapsed: bool | None = None,
        save_state: bool = False,
        notebook_preview_format: str | None = None,
        notebook_preview_quality: int = 88,
        notebook_preview_max_px: int = 512,
        **kwargs,
    ) -> None:
        reject_unknown_kwargs(type(self), kwargs)
        super().__init__(**kwargs)

        if isinstance(data, BraggVectors):
            self._bragg = data
        elif isinstance(data, Dataset4dstem):
            self._bragg = BraggVectors.from_dataset(data, device=device)
        else:
            raise TypeError(
                "ShowBragg expects a Dataset4dstem or a BraggVectors instance, got "
                f"{type(data).__name__}."
            )

        dataset = self._bragg.dataset
        self._probe = None if probe is None else to_numpy(probe, dtype=np.float32)

        if self._bragg.template is None:
            self._build_template()

        self._configure_static_fallback(
            notebook_preview_format=notebook_preview_format,
            notebook_preview_quality=notebook_preview_quality,
            notebook_preview_max_px=notebook_preview_max_px,
        )
        self._save_state = bool(save_state)

        ui = resolve_ui_mode(
            ui_mode,
            defaults={
                "show_title": True,
                "show_controls": True,
                "controls_collapsed": False,
            },
            overrides={
                "show_title": show_title,
                "show_controls": show_controls,
                "controls_collapsed": controls_collapsed,
            },
        )

        with self.hold_sync():
            self.show_title = bool(ui["show_title"])
            self.show_controls = bool(ui["show_controls"])
            self.controls_collapsed = bool(ui["controls_collapsed"])
            self.scan_shape = [int(dataset.shape[0]), int(dataset.shape[1])]
            self.q_shape = [int(dataset.shape[-2]), int(dataset.shape[-1])]
            self.title = str(title or getattr(dataset, "name", "") or "")
            self.cmap = str(cmap)
            self.log_scale = bool(log_scale)
            self.has_probe = self._probe is not None
            self._render_template()
            self._render_probe()
            if self._bragg.peaks is not None:
                self.detection_state = "done"
                self._update_bvm()
                self._choose_basis()
            if self._bragg.u_array is not None:
                self.fit_state = "done"
                self._render_fit()

        self.observe(self._on_template_change, names=self._TEMPLATE_TRAITS)
        self.observe(self._on_probe_change, names="probe_position")
        self.observe(self._on_basis_change, names=self._CANDIDATE_TRAITS)
        self.observe(self._on_display_change, names=["cmap", "log_scale"])
        self.on_msg(self._handle_msg)

        try:
            from importlib.metadata import version

            self.widget_version = version("quantem-widget")
        except Exception:
            pass

    # Validators

    @traitlets.validate("probe_position")
    def _validate_probe_position(self, proposal):
        row, col = proposal["value"]
        rows, cols = self.scan_shape if self.scan_shape else (1, 1)
        return [
            int(np.clip(row, 0, max(0, rows - 1))),
            int(np.clip(col, 0, max(0, cols - 1))),
        ]

    @traitlets.validate("template_center", "origin_rc", "g1_rc", "g2_rc")
    def _validate_pair(self, proposal):
        value = [float(v) for v in proposal["value"]]
        if value and len(value) != 2:
            raise traitlets.TraitError(
                f"{proposal['trait'].name} must be empty or a (row, col) pair, "
                f"got {len(value)} values."
            )
        return value

    @traitlets.validate("template_roi")
    def _validate_roi(self, proposal):
        value = [float(v) for v in proposal["value"]]
        if value and len(value) != 4:
            raise traitlets.TraitError(
                f"template_roi must be empty or [r0, c0, r1, c1], got {len(value)} values."
            )
        return value

    # Public API

    @property
    def bragg(self) -> BraggVectors:
        """The wrapped workflow object, for anything the panels do not expose."""
        return self._bragg

    @property
    def peaks(self) -> Vector | None:
        """Detected peaks, or None before a full detection run."""
        return self._bragg.peaks

    @property
    def bvm(self) -> Dataset2d | None:
        """Bragg vector map, or None before a full detection run."""
        return self._bragg.bvm

    @property
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """(origin, g1, g2), with origin absolute and g1/g2 offsets from it."""
        if self._bragg.origin is None:
            return None
        return (self._bragg.origin, self._bragg.g1, self._bragg.g2)

    def detect(self, positions: list[tuple[int, int]] | None = None, **kwargs) -> Vector:
        """Run disk detection with the current detection traits.

        Parameters
        ----------
        positions : list of tuple of int, optional
            (row, col) positions to test on. None runs the full scan and
            populates peaks; a subset leaves the workflow state untouched.
        **kwargs
            Overrides for the detection traits, forwarded to detect_disks.

        Returns
        -------
        Vector
            Detected peaks.
        """
        detect_kwargs = {
            "min_abs_intensity": float(self.min_abs_intensity),
            "min_spacing": float(self.min_spacing),
            "edge_boundary": int(self.edge_boundary),
            "subpixel": str(self.subpixel),
            "upsample_factor": int(self.upsample_factor),
            "max_num_peaks": int(self.max_num_peaks),
            **kwargs,
        }
        preview = positions is not None

        self.detection_state = "preview" if preview else "running"
        self.status = ""
        try:
            found = self._bragg.detect_disks(
                positions=positions, progressbar=False, **detect_kwargs
            )
        except Exception as exc:
            self.detection_state = "error"
            self.status = f"Detection failed: {exc}"
            raise

        if preview:
            self.preview_peaks = json.dumps(
                {
                    "positions": [[int(r), int(c)] for r, c in positions],
                    "counts": [int(found[i].array.shape[0]) for i in range(len(positions))],
                    "peaks": [[float(v) for v in row] for row in found[0].array],
                }
            )
            self.detection_state = "done" if self._bragg.peaks is not None else "idle"
            return found

        with self.hold_sync():
            self.detection_state = "done"
            self._update_bvm()
            self._choose_basis()
        return found

    def fit(self, **kwargs) -> "ShowBragg":
        """Fit the lattice at every scan position with the current fit traits.

        Parameters
        ----------
        **kwargs
            Overrides forwarded to fit_lattice.

        Returns
        -------
        ShowBragg
            self, for method chaining.
        """
        fit_kwargs = {
            "min_num_peaks": int(self.min_num_peaks),
            "max_peak_shift": float(self.max_peak_shift) or None,
            **kwargs,
        }

        self.fit_state = "running"
        self.status = ""
        try:
            self._bragg.fit_lattice(progressbar=False, plot=False, **fit_kwargs)
        except Exception as exc:
            self.fit_state = "error"
            self.status = f"Lattice fit failed: {exc}"
            raise

        with self.hold_sync():
            self.fit_state = "done"
            self._render_fit()
        return self

    def collapse_controls(self) -> "ShowBragg":
        """Collapse the parameter controls."""
        self.controls_collapsed = True
        return self

    def expand_controls(self) -> "ShowBragg":
        """Expand the parameter controls."""
        self.controls_collapsed = False
        return self

    def toggle_controls(self) -> "ShowBragg":
        """Toggle whether the parameter controls are collapsed."""
        self.controls_collapsed = not bool(self.controls_collapsed)
        return self

    def strain_map(self, **kwargs) -> StrainMap:
        """Hand the fitted lattice vectors to a StrainMap.

        Parameters
        ----------
        **kwargs
            u_ref, v_ref and mask, forwarded to calculate_strain_map.

        Returns
        -------
        StrainMap
            Strain map built from the per-position lattice vectors.
        """
        return self._bragg.calculate_strain_map(**kwargs)

    # Stage handlers

    def _build_template(self) -> None:
        """Rebuild the correlation template from the current template traits."""
        center = tuple(self.template_center) if self.template_center else None
        subtract_mean = bool(self.template_subtract_mean)

        if self.template_source == "synthetic":
            self._bragg.make_template_synthetic(
                radius=float(self.template_radius) or None,
                edge=float(self.template_edge),
                center=center,
                subtract_mean=subtract_mean,
            )
        elif self.template_source == "data":
            self._bragg.make_template_from_data(
                roi=self._roi_mask(),
                subtract_mean=subtract_mean,
                center=center,
            )
        else:
            if self._probe is None:
                raise ValueError(
                    "template_source='probe' needs a probe image; pass "
                    "ShowBragg(..., probe=vacuum_probe)."
                )
            self._bragg.make_template_from_probe(
                self._probe, center=center, subtract_mean=subtract_mean
            )

    def _roi_mask(self) -> np.ndarray | None:
        """Boolean scan mask from template_roi, or None for the whole scan."""
        if not self.template_roi:
            return None
        rows, cols = self.scan_shape
        r0, c0, r1, c1 = self.template_roi
        mask = np.zeros((rows, cols), dtype=bool)
        mask[
            int(np.clip(min(r0, r1), 0, rows)) : int(np.clip(max(r0, r1), 0, rows)) + 1,
            int(np.clip(min(c0, c1), 0, cols)) : int(np.clip(max(c0, c1), 0, cols)) + 1,
        ] = True
        return mask

    def _preview_positions(self) -> list[tuple[int, int]]:
        """Probe position first, then an evenly spaced grid across the scan."""
        rows, cols = self.scan_shape
        n = max(1, int(self.preview_grid))
        grid_rows = np.unique(np.linspace(0, rows - 1, min(n, rows)).astype(int))
        grid_cols = np.unique(np.linspace(0, cols - 1, min(n, cols)).astype(int))

        probe = (int(self.probe_position[0]), int(self.probe_position[1]))
        positions = [probe]
        positions += [
            (int(r), int(c)) for r in grid_rows for c in grid_cols if (int(r), int(c)) != probe
        ]
        return positions

    def _candidate_count(self) -> int:
        """How many candidate peaks to number, from the data unless set explicitly."""
        if self.num_candidates > 0:
            return int(self.num_candidates)

        peaks = self._bragg.peaks
        counts = peaks.row_counts() if peaks is not None else []
        return max(4, int(max(counts))) if len(counts) else 100

    def _choose_basis(self) -> None:
        """Re-derive the candidate peaks and resolve the basis from the picks."""
        if self._bragg.bvm is None:
            return

        try:
            self._bragg.choose_basis_vectors(
                origin=self._basis_pick("origin"),
                g1=self._basis_pick("g1"),
                g2=self._basis_pick("g2"),
                num_candidates=self._candidate_count(),
                min_spacing=float(self.candidate_min_spacing),
                min_abs_intensity=float(self.candidate_min_abs_intensity),
                plot=False,
            )
            self._bragg.index_peaks(plot=False)
        except Exception as exc:
            self.status = f"Basis selection failed: {exc}"
            return

        cand_rc = self._bragg.candidates_rc
        cand_int = self._bragg.candidates_intensity

        self._resolving_basis = True
        try:
            with self.hold_sync():
                self.status = ""
                self.candidates = [
                    [float(rc[0]), float(rc[1]), float(i)]
                    for rc, i in zip(cand_rc, cand_int)
                ]
                self.origin_rc = [float(v) for v in self._bragg.origin]
                self.g1_rc = [float(v) for v in self._bragg.g1]
                self.g2_rc = [float(v) for v in self._bragg.g2]
        finally:
            self._resolving_basis = False

    def _basis_pick(self, name: str) -> int | list[float] | None:
        """One basis argument: candidate index, explicit vector, or None for auto."""
        index = int(getattr(self, f"{name}_index"))
        if index >= 0:
            return index
        vector = getattr(self, f"{name}_rc")
        return list(vector) if vector else None

    # Rendering

    def _render(self, frame) -> bytes:
        """Colormap a 2D array into PNG bytes, or empty bytes when absent."""
        if frame is None:
            return b""
        return rgb_to_png_bytes(
            frame_to_rgb(
                np.asarray(frame, dtype=float),
                cmap=self.cmap,
                log_scale=bool(self.log_scale),
            )
        )

    def _render_template(self) -> None:
        """Render the template cropped to the disk, which is tiny on a large detector."""
        template = self._bragg.template
        if template is None:
            self.template_png = b""
            return

        frame = np.asarray(template, dtype=float)
        rows, cols = frame.shape
        centre = (rows // 2, cols // 2)

        strong = np.argwhere(frame > 0.25 * frame.max())
        reach = 8.0 if strong.size == 0 else np.abs(strong - np.array(centre)).max()
        half = int(max(12, min(min(rows, cols) // 2, round(3 * reach))))

        crop = frame[
            centre[0] - half : centre[0] + half + 1,
            centre[1] - half : centre[1] + half + 1,
        ]
        with self.hold_sync():
            self.template_shape = [int(crop.shape[0]), int(crop.shape[1])]
            self.template_png = self._render(crop)

    def _render_probe(self) -> None:
        row, col = self.probe_position
        pattern = np.asarray(self._bragg.dataset.array[row, col])
        with self.hold_sync():
            self.probe_diffraction_png = self._render(pattern)
            self.probe_correlation_png = self._render(self._bragg.correlation_map(row, col))

    def _update_bvm(self) -> None:
        bvm = self._bragg.compute_bvm(float(self.bvm_sampling))
        self.bvm_png = self._render(bvm.array)

    def _render_fit(self) -> None:
        with self.hold_sync():
            self.mask_weight_png = self._render(self._bragg.mask_weight)
            self.fit_error_png = self._render(self._bragg.fit_error)

    # Observers and messages

    def _on_template_change(self, _change) -> None:
        try:
            self._build_template()
        except Exception as exc:
            self.status = str(exc)
            return
        with self.hold_sync():
            self.status = ""
            self._render_template()
            self._render_probe()

    def _on_probe_change(self, _change) -> None:
        self._render_probe()

    def _on_basis_change(self, _change) -> None:
        if getattr(self, "_resolving_basis", False):
            return
        self._choose_basis()

    def _on_display_change(self, _change) -> None:
        with self.hold_sync():
            self._render_template()
            self._render_probe()
            if self._bragg.bvm is not None:
                self.bvm_png = self._render(self._bragg.bvm.array)
            if self._bragg.u_array is not None:
                self._render_fit()

    def _handle_msg(self, _widget: Any, content: dict[str, Any], _buffers: list[Any]) -> None:
        """Panel actions. Errors land in status instead of the kernel."""
        action = content.get("type")
        try:
            if action == "preview_detect":
                self.detect(positions=self._preview_positions())
            elif action == "run_detect":
                self.detect()
            elif action == "compute_bvm":
                self._update_bvm()
                self._choose_basis()
            elif action == "run_fit":
                self.fit()
        except Exception as exc:
            if not self.status:
                self.status = str(exc)

    def _static_png_b64(self, max_px: int = 512) -> str | None:
        preview = self.bvm_png or self.probe_correlation_png or self.template_png
        if not preview:
            return None
        return base64.b64encode(bytes(preview)).decode("ascii")
