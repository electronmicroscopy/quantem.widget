"""DiffractionSim: interactive crystal and diffraction simulation for teaching.

Left panel: the unit cell in 3D, rotated by dragging (mouse or touch) or by
buttons about the screen axes. Right panel: the diffraction pattern of the
same orientation, updated live: nanobeam (kinematical markers, or Bloch wave
intensities that follow the thickness slider), CBED disks, or the wide-angle
Kikuchi pattern. Every simulation runs in the browser, so an exported HTML
page keeps rotating the crystal without Python.

The crystal data (reflection list, structure factors and Bloch couplings) is
computed in Python by :mod:`quantem.widget._diffraction_sim_crystal`, which
needs ``quantem.diffraction.Crystal`` and ``bloch``.
"""

from __future__ import annotations

import json
import math
import pathlib
import tempfile
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import anywidget
import traitlets

from quantem.widget._diffraction_sim_crystal import (
    PRESETS,
    prepare_crystal,
    prepare_kossel_reference,
    preset_crystal,
    require_crystal_tools,
)
from quantem.widget.export import ensure_mobile_viewport
from quantem.widget.utils.state_io import (
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)
from quantem.widget.utils.static_fallback import StaticFallbackMixin

if TYPE_CHECKING:
    from ase import Atoms
    from quantem.diffraction.crystal import Crystal

_STATIC = pathlib.Path(__file__).parent / "static" / "diffractionsim.js"
_WIDGET_NAME = "DiffractionSim"

MODES = ("nanobeam", "cbed", "kikuchi")
# Earlier mode names, accepted and stored as the canonical name.
_MODE_ALIASES = {"kossel": "kikuchi"}
RENDERS = ("markers", "disks", "pixels")
QUALITIES = ("fast", "medium", "fine")
SCALINGS = ("linear", "power", "log")
VIEWS = ("detector", "gun")

#: Colormaps that come from quantem rather than the shared frontend set.
QUANTEM_COLORMAPS = ("turbo_black", "turbo_black_r")

# Settings saved by state_dict(): everything that restores the view. The
# crystal is restored from ``preset``; a CIF or ASE crystal is passed to the
# constructor again. Derived data (crystal_json, kossel_json, cmap_luts,
# presets) and transient traits (status, export bridge) are left out.
_STATE_KEYS = (
    "title",
    "preset",
    "energy_keV",
    "k_max_inv_A",
    "orientation",
    "mode",
    "render",
    "dynamical",
    "thickness_A",
    "semiangle_mrad",
    "precession_deg",
    "n_precession",
    "sigma_excitation_inv_A",
    "sg_max_inv_A",
    "quality",
    "rotation_step_deg",
    "rotation_speed_deg_per_s",
    "pattern_range_inv_A",
    "field_mrad",
    "show_kikuchi",
    "view_from",
    "scaling",
    "power",
    "marker_power",
    "marker_size_px",
    "cmap",
    "vmin_pct",
    "vmax_pct",
    "show_labels",
    "show_hkl",
    "show_cell_axes",
    "n_cells",
    "polyhedra",
    "show_ewald",
    "panel_width_px",
)

# (low, high, unit, low inclusive) of the scalar range validators.
_RANGES: dict[str, tuple[float, float, str, bool]] = {
    "energy_keV": (0.0, 1000.0, "keV", False),
    "k_max_inv_A": (0.2, 8.0, "1/Å", True),
    "thickness_A": (0.0, 1e4, "Å", False),
    "semiangle_mrad": (0.0, 50.0, "mrad", False),
    "precession_deg": (0.0, 10.0, "degrees", True),
    "sigma_excitation_inv_A": (0.0, 1.0, "1/Å", False),
    "sg_max_inv_A": (0.0, 0.5, "1/Å", False),
    "rotation_step_deg": (0.0, 180.0, "degrees", False),
    "rotation_speed_deg_per_s": (0.0, 90.0, "degrees per second", False),
    "field_mrad": (1.0, 500.0, "mrad", True),
    "power": (0.05, 1.0, "", True),
    "marker_power": (0.05, 1.0, "", True),
    "marker_size_px": (0.0, 100.0, "px", False),
    "vmin_pct": (0.0, 100.0, "percent", True),
    "vmax_pct": (0.0, 100.0, "percent", True),
}
_MAX_CELLS = 6


#: Colormaps compiled into the frontend bundle (synced from quantem.gpu at build time), used
#: when quantem.gpu is not installed; a test checks it against quantem.gpu when it is.
_FRONTEND_COLORMAPS = (
    "gray",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "magenta",
    "hot",
    "hsv",
    "turbo",
    "RdBu",
    "cividis",
    "seismic",
    "RdBu_r",
    "twilight",
    "twilight_shifted",
)


def _colormap_names() -> tuple[str, ...]:
    """Colormaps of the shared frontend set, plus their ``_r`` reversals. Does not need
    quantem.gpu at runtime."""
    try:
        from quantem.gpu.display import colormap_names
    except ImportError:
        names = _FRONTEND_COLORMAPS
    else:
        names = tuple(colormap_names())
    reversed_names = tuple(
        f"{n}_r" for n in names if not n.endswith("_r") and f"{n}_r" not in names
    )
    return names + reversed_names


def quantem_colormap_luts() -> dict[str, list[int]]:
    """Lookup tables of the colormaps imported from quantem.

    Returns
    -------
    dict of str to list of int
        For each name in ``QUANTEM_COLORMAPS``, 768 ints (256 RGB entries,
        0-255). Empty when the installed quantem has no
        ``quantem.core.visualization.turbo_black``.
    """
    try:
        import numpy as np
        from quantem.core.visualization import turbo_black
    except ImportError:
        return {}
    base = turbo_black(256)
    out = {}
    for name in QUANTEM_COLORMAPS:
        cmap = base.reversed() if name.endswith("_r") else base
        rgb = cmap(np.linspace(0.0, 1.0, 256))[:, :3]
        out[name] = np.round(255 * rgb).astype(int).ravel().tolist()
    return out


def _slug(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


class DiffractionSim(StaticFallbackMixin, anywidget.AnyWidget):
    """Interactive unit cell and diffraction pattern simulator.

    The left panel draws the unit cell, the right panel the diffraction
    pattern of the same orientation. Dragging either panel tilts the crystal.
    Kinematical and Bloch wave intensities, CBED disks and Kikuchi lines are
    computed in the browser from a reflection list prepared in Python.

    Requires quantem with ``quantem.diffraction.Crystal`` and ``bloch``
    (electronmicroscopy/quantem PR #297, first release after 0.1.9), which
    brings ASE and spglib.

    Parameters
    ----------
    crystal : Crystal, ase.Atoms, str, pathlib.Path or None, optional
        A quantem ``Crystal``, an ASE ``Atoms`` object, a CIF path, or the
        name of a preset (see :meth:`presets_available`). ``None`` starts with
        silicon, or with the preset named in ``state``.
    zone_axis : sequence of 3 float, optional
        Initial zone axis [uvw] along the beam. ``None`` keeps the identity
        orientation (c axis along the beam).
    energy_keV : float, default 200
        Beam energy in keV, above 0 and up to 1000. Changing it recomputes the
        reflection list.
    k_max_inv_A : float, default 4
        Largest scattering vector of the reflection list, in 1/Å, from 0.2
        to 8. Changing it recomputes the reflection list.
    pattern_range_inv_A : float, optional
        Scattering vector at the edge of the nanobeam and CBED panel, in 1/Å,
        from 0.2 to ``k_max_inv_A``. ``None`` uses 3, or ``k_max_inv_A`` when
        that is smaller.
    mode : {"nanobeam", "cbed", "kikuchi"}, default "nanobeam"
        Nanobeam spots, convergent-beam disks, or the wide-angle Kikuchi
        pattern. ``"kossel"``, the earlier name of ``"kikuchi"``, is accepted
        and stored as ``"kikuchi"``.
    render : {"markers", "disks", "pixels"}, default "markers"
        Nanobeam: markers sized by intensity, disks of the convergence
        semiangle with brightness by intensity, or a pixelated pattern.
        Kikuchi: vector lines (``"markers"``), or the pixel lookup of the
        reference pattern (``"pixels"``, see :meth:`compute_kossel_reference`).
    dynamical : bool, default True
        Bloch wave intensities; ``False`` uses kinematical ``|F|^2`` with a
        Gaussian excitation envelope of width ``sigma_excitation_inv_A``.
    thickness_A : float, default 500
        Specimen thickness in Å, above 0 and up to 10,000.
    semiangle_mrad : float, default 2
        Convergence semiangle in mrad (CBED disks and nanobeam disk radius),
        above 0 and up to 50.
    precession_deg : float, default 0
        Precession half angle in degrees, 0 to 10. Intensities are averaged
        over ``n_precession`` incident tilts on the precession ring.
    n_precession : int, default 24
        Incident tilts on the precession ring (half as many while dragging),
        3 to 360.
    sigma_excitation_inv_A : float, default 0.02
        Width of the kinematical excitation envelope, in 1/Å.
    sg_max_inv_A : float, default 0.05
        Excitation error cutoff selecting the Bloch beams, in 1/Å.
        Reflections outside it take thin-slab intensities.
    quality : {"fast", "medium", "fine"}, default "medium"
        Bloch beam cap and CBED tilt sampling.
    field_mrad : float, default 50
        Half angle of the Kikuchi pattern field of view, in mrad.
    show_kikuchi : bool, default False
        Overlay the Kikuchi line pairs on the nanobeam and CBED patterns.
    view_from : {"detector", "gun"}, default "detector"
        Viewpoint shared by both panels (no UI control). ``"detector"`` looks
        up the column from the detector side: the exit face of the cell is
        nearest you and tilts together with the Laue circle and the Kikuchi
        pattern. ``"gun"`` is the operator's view down the column; the
        entrance face is nearest and tilts opposite to the pattern.
    scaling : {"linear", "power", "log"}, default "linear"
        Intensity scaling of the pixel renderings.
    power : float, default 0.5
        Exponent of the ``"power"`` scaling, 0.05 to 1.
    marker_power : float, default 0.5
        Marker area scales as ``intensity**marker_power`` (0.5: square root
        of the intensity), 0.05 to 1. For nanobeam disks, the brightness
        exponent.
    marker_size_px : float, default 20
        Radius in CSS pixels of the strongest marker at a panel width of 420.
    cmap : str, default "inferno"
        Colormap of the pixel renderings: a name of the shared widget set
        (``"inferno"``, ``"turbo"``, ``"gray"``, ...), the same name with a
        ``"_r"`` suffix to reverse it, or ``"turbo_black"`` and
        ``"turbo_black_r"``, imported from quantem.
    vmin_pct, vmax_pct : float, default 0, 100
        Contrast window of the pixel renderings, in percent of the displayed
        intensity range.
    show_labels : bool, default True
        Label the cell axes a, b, c.
    show_hkl : bool, default True
        Label the strongest reflections (or Kikuchi lines) with their hkl.
    show_cell_axes : bool, default True
        Draw the cell axes.
    n_cells : sequence of 3 int, default (1, 1, 1)
        Block of cells drawn in the left panel, 1 to 6 along each of a, b, c.
    polyhedra : bool, default False
        Draw coordination polyhedra around every species except the most
        numerous one; around every atom of an elemental crystal.
    show_ewald : bool, default True
        Side view of the Ewald sphere below the cell: the reciprocal lattice
        points near the x-z plane, the sphere through the origin (z
        stretched), and the excited reflections in green.
    rotation_step_deg : float, default 15
        Step of the x, y, z rotation buttons, in degrees.
    rotation_speed_deg_per_s : float, default 6
        Speed of the continuous rotation buttons, in degrees per second.
    panel_width_px : int, default 420
        Width of the pattern panel in CSS pixels, 220 to 2000. The frontend
        shrinks it to fit a narrower container.
    title : str, default "Diffraction Simulator"
        Heading above the panels.
    save_state : bool, default False
        Embed the full interactive state, including the Kikuchi reference
        pattern, in a saved notebook. By default the reference pattern is
        left out and a PNG of both panels, captured by the frontend, is saved
        for viewing the notebook without a kernel.
    state : dict, str or pathlib.Path, optional
        Settings from :meth:`state_dict` or a file written by :meth:`save`,
        applied after the other arguments. Its ``preset`` loads that crystal
        unless ``crystal`` is given.

    Attributes
    ----------
    orientation : list of 4 float
        Crystal orientation as a unit quaternion (w, x, y, z).
    preset : str
        Name of the loaded preset, or ``""`` for other crystals. Setting it
        loads that preset.
    cmap_luts : dict
        Read-only. Lookup tables of the colormaps imported from quantem.
    status : str
        Progress message shown under the panels.
    export_status : str
        Result of the last HTML export.

    Notes
    -----
    Bloch couplings are absorptive Weickenmeier-Kohl factors when every
    element has tabulated factors, and the elastic couplings
    ``gamma F_g / pi`` otherwise; the info line under the panels says
    "absorptive" only in the first case.

    Examples
    --------
    >>> from quantem.widget import DiffractionSim
    >>> sim = DiffractionSim("Si (diamond cubic)", zone_axis=[1, 1, 0], mode="cbed")
    >>> sim.thickness_A = 800
    >>> sim.save("si_110.json")
    >>> sim.export_html("si_110.html")
    """

    _esm = _STATIC
    _save_state = traitlets.Bool(False).tag(sync=True)
    # PNG of both panels (base64), captured by the frontend after each settled
    # redraw: the static preview of a saved notebook. Same trait names as the
    # saved previews of Show2D, Show3D and Plot2D.
    _static_fallback_jpeg = traitlets.Unicode("").tag(sync=True)
    _static_fallback_mime = traitlets.Unicode("image/png").tag(sync=True)
    _UNSAVED_HEAVY_KEYS = ("kossel_json", "export_payload")

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("Diffraction Simulator").tag(sync=True)
    crystal_json = traitlets.Unicode("{}").tag(sync=True)
    kossel_json = traitlets.Unicode("{}").tag(sync=True)
    presets = traitlets.List(trait=traitlets.Unicode(), default_value=list(PRESETS)).tag(
        sync=True
    )
    preset = traitlets.Unicode("").tag(sync=True)
    energy_keV = traitlets.Float(200.0).tag(sync=True)
    k_max_inv_A = traitlets.Float(4.0).tag(sync=True)
    orientation = traitlets.List(
        trait=traitlets.Float(), default_value=[1.0, 0.0, 0.0, 0.0]
    ).tag(sync=True)
    mode = traitlets.Unicode("nanobeam").tag(sync=True)
    render = traitlets.Enum(RENDERS, default_value="markers").tag(sync=True)
    dynamical = traitlets.Bool(True).tag(sync=True)
    thickness_A = traitlets.Float(500.0).tag(sync=True)
    semiangle_mrad = traitlets.Float(2.0).tag(sync=True)
    precession_deg = traitlets.Float(0.0).tag(sync=True)
    n_precession = traitlets.Int(24).tag(sync=True)
    sigma_excitation_inv_A = traitlets.Float(0.02).tag(sync=True)
    sg_max_inv_A = traitlets.Float(0.05).tag(sync=True)
    quality = traitlets.Enum(QUALITIES, default_value="medium").tag(sync=True)
    rotation_step_deg = traitlets.Float(15.0).tag(sync=True)
    rotation_speed_deg_per_s = traitlets.Float(6.0).tag(sync=True)
    pattern_range_inv_A = traitlets.Float(3.0).tag(sync=True)
    field_mrad = traitlets.Float(50.0).tag(sync=True)
    show_kikuchi = traitlets.Bool(False).tag(sync=True)
    view_from = traitlets.Enum(VIEWS, default_value="detector").tag(sync=True)
    scaling = traitlets.Enum(SCALINGS, default_value="linear").tag(sync=True)
    power = traitlets.Float(0.5).tag(sync=True)
    marker_power = traitlets.Float(0.5).tag(sync=True)
    marker_size_px = traitlets.Float(20.0).tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    cmap_luts = traitlets.Dict(
        value_trait=traitlets.List(traitlets.Int()), default_value={}
    ).tag(sync=True)
    vmin_pct = traitlets.Float(0.0).tag(sync=True)
    vmax_pct = traitlets.Float(100.0).tag(sync=True)
    show_labels = traitlets.Bool(True).tag(sync=True)
    show_hkl = traitlets.Bool(True).tag(sync=True)
    show_cell_axes = traitlets.Bool(True).tag(sync=True)
    n_cells = traitlets.List(trait=traitlets.Int(), default_value=[1, 1, 1]).tag(sync=True)
    polyhedra = traitlets.Bool(False).tag(sync=True)
    show_ewald = traitlets.Bool(True).tag(sync=True)
    panel_width_px = traitlets.Int(420).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)

    # Exported HTML: no kernel; the crystal menu lists the embedded presets.
    offline = traitlets.Bool(False).tag(sync=True)
    _offline_presets = traitlets.Dict(default_value={}).tag(sync=True)

    # HTML export bridge
    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        crystal: Crystal | Atoms | str | pathlib.Path | None = None,
        *,
        zone_axis: Sequence[float] | None = None,
        energy_keV: float = 200.0,
        k_max_inv_A: float = 4.0,
        pattern_range_inv_A: float | None = None,
        mode: str = "nanobeam",
        render: str = "markers",
        dynamical: bool = True,
        thickness_A: float = 500.0,
        semiangle_mrad: float = 2.0,
        precession_deg: float = 0.0,
        n_precession: int = 24,
        sigma_excitation_inv_A: float = 0.02,
        sg_max_inv_A: float = 0.05,
        quality: str = "medium",
        field_mrad: float = 50.0,
        show_kikuchi: bool = False,
        view_from: str = "detector",
        scaling: str = "linear",
        power: float = 0.5,
        marker_power: float = 0.5,
        marker_size_px: float = 20.0,
        cmap: str = "inferno",
        vmin_pct: float = 0.0,
        vmax_pct: float = 100.0,
        show_labels: bool = True,
        show_hkl: bool = True,
        show_cell_axes: bool = True,
        n_cells: Sequence[int] = (1, 1, 1),
        polyhedra: bool = False,
        show_ewald: bool = True,
        rotation_step_deg: float = 15.0,
        rotation_speed_deg_per_s: float = 6.0,
        panel_width_px: int = 420,
        title: str = "Diffraction Simulator",
        save_state: bool = False,
        state: dict[str, Any] | str | pathlib.Path | None = None,
    ) -> None:
        require_crystal_tools()
        saved = self._resolve_state(state) if state is not None else None
        if saved is not None:
            if crystal is None and saved.get("preset") in PRESETS:
                crystal = saved["preset"]
            saved = {k: v for k, v in saved.items() if k != "preset"}
        self._save_state = bool(save_state)
        self._configure_static_fallback(notebook_preview_format="png")
        super().__init__()
        self._crystal = None
        self._kossel_cache: dict = {}
        self._recompute_paused = False
        with self.hold_sync():
            self.widget_version = resolve_widget_version()
            self.cmap_luts = quantem_colormap_luts()
            self.title = title
            self.energy_keV = energy_keV
            self.k_max_inv_A = k_max_inv_A
            self.pattern_range_inv_A = (
                min(3.0, self.k_max_inv_A) if pattern_range_inv_A is None else pattern_range_inv_A
            )
            self.mode = mode
            self.render = render
            self.dynamical = dynamical
            self.thickness_A = thickness_A
            self.semiangle_mrad = semiangle_mrad
            self.precession_deg = precession_deg
            self.n_precession = n_precession
            self.sigma_excitation_inv_A = sigma_excitation_inv_A
            self.sg_max_inv_A = sg_max_inv_A
            self.quality = quality
            self.field_mrad = field_mrad
            self.show_kikuchi = show_kikuchi
            self.view_from = view_from
            self.scaling = scaling
            self.power = power
            self.marker_power = marker_power
            self.marker_size_px = marker_size_px
            self.cmap = cmap
            self.vmin_pct = vmin_pct
            self.vmax_pct = vmax_pct
            self.show_labels = show_labels
            self.show_hkl = show_hkl
            self.show_cell_axes = show_cell_axes
            self.n_cells = list(n_cells)
            self.polyhedra = polyhedra
            self.show_ewald = show_ewald
            self.rotation_step_deg = rotation_step_deg
            self.rotation_speed_deg_per_s = rotation_speed_deg_per_s
            self.panel_width_px = panel_width_px
            self.set_crystal("Si (diamond cubic)" if crystal is None else crystal)
            if zone_axis is not None:
                self.set_zone_axis(zone_axis)
            if saved is not None:
                self.load_state_dict(saved)
        self.observe(self._on_preset, names="preset")
        self.observe(self._on_physics, names=["energy_keV", "k_max_inv_A"])
        self.observe(self._on_export_request_change, names=["export_request"])
        self.on_msg(self._on_message)

    # ------------------------------------------------------------------
    # validation
    @traitlets.validate(*_RANGES)
    def _validate_range(self, proposal):
        name = proposal["trait"].name
        lo, hi, unit, lo_inclusive = _RANGES[name]
        value = proposal["value"]
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        ok = ok and math.isfinite(value)
        ok = ok and (value >= lo if lo_inclusive else value > lo) and value <= hi
        if not ok:
            low = f"from {lo:g}" if lo_inclusive else f"above {lo:g}"
            unit_text = f" {unit}" if unit else ""
            raise traitlets.TraitError(
                f"{name} must be finite, {low} and up to {hi:g}{unit_text}; got {value!r}."
            )
        return float(value)

    @traitlets.validate("pattern_range_inv_A")
    def _validate_pattern_range(self, proposal):
        value = proposal["value"]
        if not math.isfinite(value) or not 0.2 <= value <= self.k_max_inv_A:
            raise traitlets.TraitError(
                "pattern_range_inv_A must be from 0.2 to k_max_inv_A "
                f"({self.k_max_inv_A:g} 1/Å); got {value!r}. Raise k_max_inv_A first "
                "to show a wider range."
            )
        return float(value)

    @traitlets.validate("n_precession")
    def _validate_n_precession(self, proposal):
        value = proposal["value"]
        if isinstance(value, bool) or not 3 <= value <= 360:
            raise traitlets.TraitError(
                f"n_precession must be an integer from 3 to 360; got {value!r}."
            )
        return value

    @traitlets.validate("n_cells")
    def _validate_n_cells(self, proposal):
        value = list(proposal["value"])
        if len(value) != 3 or any(
            isinstance(v, bool) or int(v) != v or not 1 <= v <= _MAX_CELLS for v in value
        ):
            raise traitlets.TraitError(
                f"n_cells must be three integers from 1 to {_MAX_CELLS} along a, b, c; "
                f"got {value!r}."
            )
        return [int(v) for v in value]

    @traitlets.validate("panel_width_px")
    def _validate_panel_width(self, proposal):
        value = proposal["value"]
        if isinstance(value, bool) or not 220 <= value <= 2000:
            raise traitlets.TraitError(
                f"panel_width_px must be from 220 to 2000 CSS pixels; got {value!r}."
            )
        return value

    @traitlets.validate("orientation")
    def _validate_orientation(self, proposal):
        value = [float(v) for v in proposal["value"]]
        norm = math.sqrt(sum(v * v for v in value)) if len(value) == 4 else 0.0
        if len(value) != 4 or not all(math.isfinite(v) for v in value) or norm < 1e-12:
            raise traitlets.TraitError(
                "orientation must be a nonzero quaternion (w, x, y, z) of four finite "
                f"numbers; got {proposal['value']!r}. Use set_zone_axis() to orient by [uvw]."
            )
        return [v / norm for v in value]

    @traitlets.validate("mode")
    def _validate_mode(self, proposal):
        value = str(proposal["value"]).strip().lower()
        value = _MODE_ALIASES.get(value, value)
        if value not in MODES:
            raise traitlets.TraitError(
                f"mode must be one of {MODES} (or 'kossel', the earlier name of 'kikuchi'); "
                f"got {proposal['value']!r}"
            )
        return value

    @traitlets.validate("cmap")
    def _validate_cmap(self, proposal):
        value = proposal["value"]
        luts = self.cmap_luts or {}
        if value in QUANTEM_COLORMAPS and value not in luts:
            raise traitlets.TraitError(
                f"cmap {value!r} needs quantem.core.visualization.turbo_black, which the "
                "installed quantem does not provide; choose a shared colormap such as 'inferno'."
            )
        allowed = _colormap_names() + tuple(luts)
        if value not in allowed:
            raise traitlets.TraitError(
                f"Unknown colormap {value!r}; choose from {sorted(allowed)}."
            )
        return value

    @traitlets.validate("preset")
    def _validate_preset(self, proposal):
        value = proposal["value"]
        if value and value not in PRESETS:
            raise traitlets.TraitError(
                f"Unknown preset {value!r}; choose from DiffractionSim.presets_available()."
            )
        return value

    # ------------------------------------------------------------------
    # crystal and orientation
    @staticmethod
    def presets_available() -> list[str]:
        """Names of the built-in crystal structures.

        Returns
        -------
        list of str
            Preset names accepted by the constructor, :meth:`set_crystal`,
            the ``preset`` trait and the ``presets`` option of
            :meth:`export_html`.
        """
        return list(PRESETS)

    @property
    def crystal(self) -> Crystal:
        """The loaded ``quantem.diffraction.crystal.Crystal``."""
        return self._crystal

    def set_crystal(self, crystal: Crystal | Atoms | str | pathlib.Path) -> DiffractionSim:
        """Load a crystal and recompute the reflection list.

        Parameters
        ----------
        crystal : Crystal, ase.Atoms, str or pathlib.Path
            A quantem ``Crystal``, an ASE ``Atoms`` object, a CIF path, or a
            preset name.

        Returns
        -------
        DiffractionSim
            The widget, for chaining.

        Raises
        ------
        ValueError
            If a string is neither a preset name nor an existing CIF file.
        TypeError
            If ``crystal`` is none of the supported types.
        """
        from ase import Atoms
        from quantem.diffraction.crystal import Crystal

        preset_name = ""
        if isinstance(crystal, str) and crystal in PRESETS:
            preset_name = crystal
            xtl = preset_crystal(crystal)
        elif isinstance(crystal, (str, pathlib.Path)):
            path = pathlib.Path(crystal).expanduser()
            if not path.is_file():
                raise ValueError(
                    f"{str(crystal)!r} is neither a preset nor an existing CIF file; use one of "
                    "DiffractionSim.presets_available() or the path of a CIF."
                )
            xtl = Crystal.from_cif(str(path), verbose=False)
        elif isinstance(crystal, Atoms):
            xtl = Crystal.from_ase(crystal, verbose=False)
        elif isinstance(crystal, Crystal):
            xtl = crystal
        else:
            raise TypeError(
                "crystal must be a quantem Crystal, an ase.Atoms, a preset name or a CIF "
                f"path; got {type(crystal).__name__}."
            )
        self._crystal = xtl
        with self.hold_sync():
            self._update_crystal_json()
            if self.preset != preset_name:
                self.preset = preset_name
        return self

    def set_zone_axis(
        self, zone_axis: Sequence[float], in_plane_deg: float = 0.0
    ) -> DiffractionSim:
        """Put a crystal direction [uvw] along the beam.

        Parameters
        ----------
        zone_axis : sequence of 3 float
            Direct-lattice direction ``u*a + v*b + w*c``, not all zero.
        in_plane_deg : float, default 0
            Rotation about the beam, in degrees.

        Returns
        -------
        DiffractionSim
            The widget, for chaining.

        Raises
        ------
        ValueError
            If ``zone_axis`` is not three finite numbers, or is zero.
        """
        import torch
        from quantem.diffraction.rotations import quat_from_zone_axis

        uvw = [float(v) for v in zone_axis]
        if len(uvw) != 3 or not all(math.isfinite(v) for v in uvw) or not any(uvw):
            raise ValueError(
                "zone_axis must be three finite numbers [u, v, w], not all zero; "
                f"got {zone_axis!r}."
            )
        if not math.isfinite(in_plane_deg):
            raise ValueError(f"in_plane_deg must be finite; got {in_plane_deg!r}.")
        d = torch.as_tensor(uvw, dtype=torch.float64) @ self._crystal.lat_real
        q = quat_from_zone_axis(d[None], float(in_plane_deg))[0]
        self.orientation = [float(v) for v in q]
        return self

    def compute_kossel_reference(
        self,
        thicknesses_A: Sequence[float] = (300.0, 600.0, 1000.0),
        angle_step_mrad: float = 3.0,
        k_max_inv_A: float = 1.0,
    ) -> DiffractionSim:
        """Compute the reference pattern for the pixel rendering of the Kikuchi mode.

        Takes about a minute for silicon at 3 mrad. Results are cached per
        energy, thicknesses, step and ``k_max_inv_A``.

        Parameters
        ----------
        thicknesses_A : sequence of float, default (300, 600, 1000)
            Specimen thicknesses in Å; the browser interpolates between them.
        angle_step_mrad : float, default 3
            Angular step of the reference grid, in mrad.
        k_max_inv_A : float, default 1
            Largest scattering vector of the Bloch calculation, in 1/Å.

        Returns
        -------
        DiffractionSim
            The widget, for chaining.
        """
        thicknesses = [float(t) for t in thicknesses_A]
        if not thicknesses or not all(math.isfinite(t) and t > 0 for t in thicknesses):
            raise ValueError(
                f"thicknesses_A must be positive thicknesses in Å; got {thicknesses_A!r}."
            )
        if not (math.isfinite(angle_step_mrad) and angle_step_mrad > 0):
            raise ValueError(f"angle_step_mrad must be positive; got {angle_step_mrad!r}.")
        if not (math.isfinite(k_max_inv_A) and k_max_inv_A > 0):
            raise ValueError(f"k_max_inv_A must be positive; got {k_max_inv_A!r}.")
        key = (round(self.energy_keV * 1e3), tuple(thicknesses), angle_step_mrad, k_max_inv_A)
        if key not in self._kossel_cache:
            self.status = "computing Kikuchi reference pattern..."
            self._kossel_cache[key] = prepare_kossel_reference(
                self._crystal, self.energy_keV * 1e3, thicknesses, angle_step_mrad, k_max_inv_A
            )
            self.status = ""
        self.kossel_json = json.dumps(self._kossel_cache[key])
        return self

    def _update_crystal_json(self) -> None:
        self.crystal_json = json.dumps(
            prepare_crystal(self._crystal, self.energy_keV * 1e3, self.k_max_inv_A)
        )
        self.kossel_json = "{}"

    # ------------------------------------------------------------------
    # observers
    def _on_preset(self, change):
        name = change["new"]
        if (
            name in PRESETS
            and self._crystal is not None
            and self._crystal.name != name.split(" (")[0]
        ):
            self.set_crystal(name)

    @traitlets.observe("_static_fallback_jpeg")
    def _on_frontend_preview(self, change):
        # refresh the saved-notebook sibling with the latest frontend capture
        fill = getattr(self, "_static_fallback_fill", None)
        if callable(fill) and change["new"]:
            try:
                fill()
            except Exception:  # noqa: BLE001
                return

    def _on_physics(self, change):
        if change["name"] == "k_max_inv_A" and self.pattern_range_inv_A > self.k_max_inv_A:
            self.pattern_range_inv_A = self.k_max_inv_A
        if self._crystal is not None and not self._recompute_paused:
            self._update_crystal_json()

    def _on_message(self, widget, content, buffers):
        if isinstance(content, dict) and content.get("type") == "kossel_reference":
            self.compute_kossel_reference()

    # ------------------------------------------------------------------
    # state
    @staticmethod
    def _resolve_state(state: dict[str, Any] | str | pathlib.Path) -> dict[str, Any]:
        if isinstance(state, (str, pathlib.Path)):
            return unwrap_state_payload(
                json.loads(pathlib.Path(state).read_text()),
                require_envelope=True,
                expected_widget=_WIDGET_NAME,
            )
        return unwrap_state_payload(state, expected_widget=_WIDGET_NAME)

    def state_dict(self) -> dict[str, Any]:
        """Return the settings that restore the view, as a plain dict.

        Returns
        -------
        dict
            Trait name to value for the crystal preset, beam energy,
            orientation, mode and every display setting. The reflection list,
            Kikuchi reference pattern and colormap tables are derived data and
            are not included.
        """
        state = {}
        for key in _STATE_KEYS:
            value = getattr(self, key)
            state[key] = list(value) if isinstance(value, list) else value
        return state

    def load_state_dict(self, state: dict[str, Any]) -> DiffractionSim:
        """Apply settings from :meth:`state_dict`; unknown keys are ignored.

        A non-empty ``preset`` loads that crystal. Otherwise the current
        crystal is kept, so a CIF or ASE crystal is passed to the constructor
        again together with ``state``.

        Parameters
        ----------
        state : dict
            Settings, as returned by :meth:`state_dict`, or the versioned
            envelope written by :meth:`save`.

        Returns
        -------
        DiffractionSim
            The widget, for chaining.
        """
        state = unwrap_state_payload(state, expected_widget=_WIDGET_NAME)
        values = {k: v for k, v in state.items() if k in _STATE_KEYS}
        physics_changed = any(
            key in values and values[key] != getattr(self, key)
            for key in ("energy_keV", "k_max_inv_A")
        )
        with self.hold_sync():
            self._recompute_paused = True
            try:
                for key in ("energy_keV", "k_max_inv_A"):
                    if key in values:
                        setattr(self, key, values[key])
                preset = values.get("preset", "")
                if preset and preset != self.preset:
                    self.set_crystal(preset)
                elif physics_changed and self._crystal is not None:
                    self._update_crystal_json()
            finally:
                self._recompute_paused = False
            for key, value in values.items():
                if key in ("energy_keV", "k_max_inv_A", "preset"):
                    continue
                setattr(self, key, list(value) if isinstance(value, tuple) else value)
        return self

    def save(self, path: str | pathlib.Path) -> None:
        """Write the settings of :meth:`state_dict` to a versioned JSON file.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file. Missing parent directories are created.
        """
        save_state_file(path, _WIDGET_NAME, self.state_dict())

    # ------------------------------------------------------------------
    # saved-notebook state
    def get_state(self, key=None, drop_defaults=False):
        """Return full saved state, or an untrimmed targeted live update."""
        state = super().get_state(key=key, drop_defaults=drop_defaults)
        if key is None and not self._save_state:
            for heavy_key in self._UNSAVED_HEAVY_KEYS:
                state.pop(heavy_key, None)
        return state

    def _static_png_b64(self, max_px: int = 512) -> str | None:
        """Return the frontend PNG of both panels, or ``None`` before the first capture."""
        return self._static_fallback_jpeg or None

    # ------------------------------------------------------------------
    # HTML export
    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        presets: Sequence[str] | None = None,
        **options: Any,
    ) -> pathlib.Path:
        """Write a standalone HTML viewer with the current state.

        The page hydrates with the ipywidgets HTML manager, which loads from
        a CDN, so opening it needs network access. Every simulation then runs
        in the browser: rotating the cell and changing thickness, mode or
        display settings need no Python. The beam energy and ``k_max_inv_A``
        are fixed to the exported values, and the Kikuchi reference pattern
        is included when it was computed.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination HTML file. Missing parent directories are created.
            Defaults to ``<crystal>_<mode>_diffractionsim.html`` in the working
            directory.
        title : str, optional
            Browser page title; defaults to the ``title`` trait.
        presets : sequence of str, optional
            Additional preset structures to embed so the crystal menu works in
            the page. Each adds its reflection list to the file.
        **options
            Standard export options. Only ``mode="single"``,
            ``encoding="full"`` and ``downsample=None`` (or 1) apply: the page
            holds no image data to pack or reduce.

        Returns
        -------
        pathlib.Path
            The written file.
        """
        self._check_export_options(options)
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path()
        self._write_html_export(export_path, title=title, presets=presets)
        ensure_mobile_viewport(export_path)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, single file)"
        return export_path

    @staticmethod
    def _check_export_options(options: dict[str, Any]) -> None:
        unknown = sorted(set(options) - {"mode", "encoding", "downsample"})
        if unknown:
            raise TypeError(f"export_html() got an unexpected keyword argument {unknown[0]!r}.")
        if options.get("mode", "single") != "single":
            raise ValueError("DiffractionSim exports one HTML file; use mode='single'.")
        if options.get("encoding", "full") != "full":
            raise ValueError("DiffractionSim has no image data to quantize; use encoding='full'.")
        if options.get("downsample") not in (None, 1):
            raise ValueError(
                "DiffractionSim has no image data to downsample; use downsample=None."
            )

    def _on_export_request_change(self, change: dict) -> None:
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            if str(payload.get("mode", "single")) == "clear":
                self.export_payload = b""
                self.export_payload_id = ""
                self.export_filename = ""
                return
            presets = payload.get("presets") or None
            if payload.get("download"):
                filename = str(payload.get("filename") or self._default_html_export_path().name)
                request_id = str(payload.get("id") or "")
                self.export_status = f"Preparing {filename}..."
                html = self._html_export_bytes(presets=presets)
                self.export_filename = filename
                self.export_payload = html
                self.export_payload_id = request_id
                size_mb = len(html) / (1024 * 1024)
                self.export_status = f"Ready {filename} ({size_mb:.1f} MB, single file)"
            else:
                self.export_status = "Exporting HTML..."
                self.export_html(presets=presets)
        except Exception as exc:  # noqa: BLE001
            self.export_status = f"Export failed: {exc}"

    def _default_html_export_path(self) -> pathlib.Path:
        name = self._crystal.name if self._crystal is not None else "crystal"
        slug = _slug(f"{name} {self.mode}") or "crystal"
        return pathlib.Path.cwd() / f"{slug}_diffractionsim.html"

    def _write_html_export(
        self,
        path: str | pathlib.Path,
        *,
        title: str | None = None,
        presets: Sequence[str] | None = None,
    ) -> pathlib.Path:
        from ipywidgets.embed import dependency_state, embed_minimal_html

        export_path = pathlib.Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_widget = self._clone_for_html_export(presets)
        try:
            state = dependency_state([export_widget], drop_defaults=False)
            embed_minimal_html(
                str(export_path),
                views=[export_widget],
                title=title or self.title or _WIDGET_NAME,
                drop_defaults=False,
                state=state,
            )
        finally:
            export_widget.close()
        return export_path

    def _html_export_bytes(self, presets: Sequence[str] | None = None) -> bytes:
        with tempfile.TemporaryDirectory(prefix="diffractionsim-export-") as tmp:
            path = pathlib.Path(tmp) / self._default_html_export_path().name
            self._write_html_export(path, presets=presets)
            ensure_mobile_viewport(path)
            return path.read_bytes()

    def _clone_for_html_export(self, presets: Sequence[str] | None = None) -> DiffractionSim:
        unknown = [name for name in presets or () if name not in PRESETS]
        if unknown:
            raise ValueError(
                f"Unknown presets {unknown}; choose from DiffractionSim.presets_available()."
            )
        state = self.state_dict()
        state.pop("preset")
        # save_state=True keeps the Kikuchi reference in the embedded state
        clone = type(self)(self._crystal, state=state, save_state=True)
        energy_ev = self.energy_keV * 1e3
        embedded = {
            name: prepare_crystal(preset_crystal(name), energy_ev, self.k_max_inv_A)
            for name in presets or ()
            if name != self.preset
        }
        if self.preset:
            embedded[self.preset] = json.loads(self.crystal_json)
        with clone.hold_sync():
            clone.preset = self.preset
            clone.crystal_json = self.crystal_json
            clone.kossel_json = self.kossel_json
            clone.presets = list(embedded)
            clone._offline_presets = embedded
            clone.offline = True
            clone.status = ""
            clone.export_enabled = False
            clone.export_status = ""
            clone.export_payload = b""
            clone.export_payload_id = ""
            clone.export_filename = ""
        return clone

    def __repr__(self) -> str:
        name = self._crystal.name if self._crystal is not None else "?"
        return (
            f"DiffractionSim(crystal={name!r}, mode={self.mode!r}, "
            f"energy_keV={self.energy_keV:g}, thickness_A={self.thickness_A:g})"
        )


__all__ = [
    "MODES",
    "PRESETS",
    "QUANTEM_COLORMAPS",
    "DiffractionSim",
    "quantem_colormap_luts",
]
