"""Inspect crystallographic atoms and projected columns without changing the model."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms

import anywidget
import numpy as np
import traitlets


class ShowCIF(anywidget.AnyWidget):
    """Show linked 3D atoms and an orthographic column projection.

    Parameters
    ----------
    structure : str, pathlib.Path or ase.Atoms
        CIF filename or ASE structure. CIF symmetry expansion is performed by
        ASE. Partially occupied or mixed sites require an explicit ordered model.
    repeats : sequence of int, default (3, 3, 3)
        Number of unit cells along lattice vectors a, b, c. At most 250,000
        atoms are displayed; choose a smaller inspection region for large cells.
    zone_axis : sequence of int, default (0, 0, 1)
        Direct-lattice direction [uvw] for the projection, not a reciprocal
        plane normal. Rotating the 3D camera does not change this direction.
    specimen_tilt_mrad : sequence of float, default (0, 0)
        Rigid specimen tilt (row, column), each from -15 to +15 mrad, about
        the inspection cell center relative to zone_axis. Positive row/column
        leans down/right with increasing beam depth. Components form a rotation
        vector in the nominal beam frame; the CIF and camera are unchanged.
    title : str, optional
        Heading above the two views.

    view_mode : {"unit_cells", "microscope"}, default "unit_cells"
        Fit the inspection cell or show an exact square physical field of view.
    orthogonal_views : bool, default False
        Show additional column-depth and row-depth atom projections, linked to
        the same beam frame, species visibility, and slab selection.
    field_of_view_A : float, default 40
        Microscope projection width and height in Å. Does not add atoms.
    magnification_calibration : sequence of float, optional
        Reference magnification and its field of view in Å, for a fixed camera
        and acquisition geometry. For example, (1_000_000, 100). The viewer
        uses inverse magnification scaling; no calibration is assumed.
    energy_keV : float, default 300
        Electron kinetic energy from 0 to 300 keV. Zero disables phase display.
    potential_colormap : str, default "viridis"
        Shared Show3D colormap for every potential/phase panel, such as
        "viridis", "magma", "gray", or "RdBu".
    potential_quantity : {"integrated", "average", "phase"}, default "average"
        Display V Å, thickness-average V, or unwrapped projected-potential phase.
        Phase uses relativistic sigma(E) times the integrated potential; it is
        not a multislice exit-wave calculation.
    num_slices : int, default 16
        Equal-width preview slabs along the selected beam, from 1 to 64.
    potential : bool, default False
        Enable WebGPU independent-atom potential preview (requires abTEM).
        At most 8,192 atoms can contribute to this preview; the atom viewer
        supports up to 250,000. Reduce repeats if the preview is unavailable.
    potential_sigma_A : float, default 0.08
        Explicit transverse Gaussian preview filter width, in Å. Not thermal
        displacement or fitted experimental resolution.
    potential_pixels : int, default 256
        Preview grid width and height, one of 128, 256 or 512.

    Notes
    -----
    Species visibility, clipping and sphere radii are display settings only.
    The atom projection shows positions, not intensity or phase. Optional
    potential maps use neutral Lobato infinite atomic projections assigned by
    site depth; they are not finite-z potential integrals or reconstruction data.
    WebGPU is required for rendering. No structure editing is performed.

    Examples
    --------
    >>> viewer = ShowCIF("crystal.cif", repeats=(4, 4, 8))
    >>> viewer.specimen_tilt_mrad = [5.0, -2.0]
    >>> original = viewer.atoms()
    """

    _esm = Path(__file__).parent / "static" / "showcif.js"
    title = traitlets.Unicode("").tag(sync=True)
    atom_bytes = traitlets.Bytes(b"").tag(sync=True)
    unit_atom_bytes = traitlets.Bytes(b"").tag(sync=True)
    atom_count = traitlets.Int(0).tag(sync=True)
    cell = traitlets.List(default_value=[]).tag(sync=True)
    unit_cell = traitlets.List(default_value=[]).tag(sync=True)
    species = traitlets.List(default_value=[]).tag(sync=True)
    repeats = traitlets.List(traitlets.Int(), default_value=[3, 3, 3]).tag(sync=True)
    zone_axis = traitlets.List(traitlets.Int(), default_value=[0, 0, 1]).tag(sync=True)
    specimen_tilt_mrad = traitlets.List(
        traitlets.Float(), default_value=[0.0, 0.0]
    ).tag(sync=True)
    visible_species = traitlets.List(traitlets.Bool(), default_value=[]).tag(sync=True)
    source_summary = traitlets.Unicode("").tag(sync=True)
    potential_bytes = traitlets.Bytes(b"").tag(sync=True)
    potential_sigma_A = traitlets.Float(0.08).tag(sync=True)
    potential_pixels = traitlets.Int(256).tag(sync=True)
    energy_keV = traitlets.Float(300).tag(sync=True)
    potential_colormap = traitlets.Enum(
        [
            "inferno",
            "viridis",
            "plasma",
            "magma",
            "magenta",
            "hot",
            "gray",
            "hsv",
            "turbo",
            "cividis",
            "RdBu",
            "RdBu_r",
            "seismic",
            "twilight",
            "twilight_shifted",
        ],
        default_value="viridis",
    ).tag(sync=True)
    potential_quantity = traitlets.Enum(
        ["integrated", "average", "phase"], default_value="average"
    ).tag(sync=True)
    num_slices = traitlets.Int(16).tag(sync=True)
    view_mode = traitlets.Enum(
        ["unit_cells", "microscope"], default_value="unit_cells"
    ).tag(sync=True)
    orthogonal_views = traitlets.Bool(False).tag(sync=True)
    field_of_view_A = traitlets.Float(40).tag(sync=True)
    magnification_calibration = traitlets.List(traitlets.Float(), default_value=[]).tag(
        sync=True
    )

    def __init__(
        self,
        structure: str | Path | Atoms,
        *,
        repeats: Sequence[int] = (3, 3, 3),
        zone_axis: Sequence[int] = (0, 0, 1),
        specimen_tilt_mrad: Sequence[float] = (0, 0),
        title: str = "",
        view_mode: str = "unit_cells",
        field_of_view_A: float = 40,
        orthogonal_views: bool = False,
        magnification_calibration: Sequence[float] | None = None,
        energy_keV: float = 300,
        potential_colormap: str = "viridis",
        potential_quantity: str = "average",
        num_slices: int = 16,
        potential: bool = False,
        potential_sigma_A: float = 0.08,
        potential_pixels: int = 256,
    ) -> None:
        try:
            from ase import Atoms
            from ase.data.colors import jmol_colors
            from ase.io import read
        except ImportError as exc:
            raise ImportError(
                "ShowCIF requires ASE. Install quantem.widget[crystal] "
                "for CIF inspection and optional potential previews."
            ) from exc

        if (
            isinstance(num_slices, bool)
            or not isinstance(num_slices, int)
            or not 1 <= num_slices <= 64
        ):
            raise ValueError("num_slices must be an integer from 1 to 64.")
        if potential_pixels not in (128, 256, 512):
            raise ValueError(
                "potential_pixels must be 128, 256 or 512; this is an explicit preview grid."
            )
        r = list(repeats)
        if len(r) != 3 or any(
            isinstance(v, bool) or not np.isfinite(v) or int(v) != v or v < 1 for v in r
        ):
            raise ValueError(
                "repeats must contain three positive integers along a, b, c."
            )
        zone = list(zone_axis)
        if (
            len(zone) != 3
            or not any(zone)
            or any(
                isinstance(v, bool) or not np.isfinite(v) or int(v) != v for v in zone
            )
        ):
            raise ValueError(
                "zone_axis must be three integers [u, v, w], not all zero."
            )
        if isinstance(structure, Atoms):
            atoms = structure.copy()
            source = "ASE structure"
        else:
            path = Path(structure).expanduser()
            if not path.is_file():
                raise ValueError(
                    f"CIF not found: {path}. Supply an existing CIF path or ase.Atoms."
                )
            atoms = read(path)
            source = path.name
        if (
            not len(atoms)
            or atoms.cell.rank != 3
            or not np.all(np.isfinite(atoms.positions))
            or not np.all(np.isfinite(atoms.cell.array))
            or abs(np.linalg.det(atoms.cell.array)) < 1e-12
        ):
            raise ValueError(
                "Supply a nonempty crystal with a finite 3D unit cell and atomic positions."
            )
        occupancy = atoms.info.get("occupancy", {})
        if any(
            len(site) != 1
            or any(
                not np.isfinite(float(v)) or abs(float(v) - 1) > 1e-6
                for v in site.values()
            )
            for site in occupancy.values()
        ):
            raise ValueError(
                "Partial or mixed occupancies need an explicit ordered structure; they are not silently filled."
            )
        count = len(atoms) * prod(int(v) for v in r)
        if count > 250_000:
            raise ValueError(
                f"Requested {count:,} atoms; use a smaller inspection region (at most 250,000)."
            )
        super().__init__()
        self._atoms = atoms.copy()
        expanded = atoms.repeat(tuple(int(v) for v in r))
        numbers = sorted(set(atoms.numbers.tolist()))
        symbols = {
            int(z): symbol
            for z, symbol in zip(atoms.numbers, atoms.get_chemical_symbols())
        }
        packed = np.empty((len(expanded), 4), dtype=np.float32)
        packed[:, :3] = expanded.positions
        packed[:, 3] = [numbers.index(int(z)) for z in expanded.numbers]
        with self.hold_sync():
            self.view_mode = view_mode
            self.orthogonal_views = orthogonal_views
            self.field_of_view_A = field_of_view_A
            self.magnification_calibration = (
                list(magnification_calibration)
                if magnification_calibration is not None
                else []
            )
            self.energy_keV = energy_keV
            self.potential_colormap = potential_colormap
            self.potential_quantity = potential_quantity
            self.num_slices = num_slices
            self.potential_pixels = potential_pixels
            self.potential_sigma_A = potential_sigma_A
            if potential:
                from ._showcif_potential import projected_atom_table

                self.potential_bytes = np.stack(
                    [
                        projected_atom_table(symbols[z], potential_sigma_A)
                        for z in numbers
                    ]
                ).tobytes()
            self.title = title or atoms.get_chemical_formula() + " · Atomic structure"
            self.atom_bytes = packed.tobytes()
            unit_packed = np.empty((len(atoms), 4), dtype=np.float32)
            unit_packed[:, :3] = atoms.positions
            unit_packed[:, 3] = [numbers.index(int(z)) for z in atoms.numbers]
            self.unit_atom_bytes = unit_packed.tobytes()
            self.atom_count = len(expanded)
            self.unit_cell = atoms.cell.array.tolist()
            self.cell = expanded.cell.array.tolist()
            self.species = [
                {
                    "symbol": symbols[z],
                    "number": z,
                    "color": jmol_colors[z].tolist(),
                    "count": int(np.count_nonzero(expanded.numbers == z)),
                }
                for z in numbers
            ]
            self.repeats = [int(v) for v in r]
            self.zone_axis = [int(v) for v in zone]
            self.specimen_tilt_mrad = list(specimen_tilt_mrad)
            self.visible_species = [True] * len(numbers)
            self.source_summary = f"{source} · {len(atoms)} atoms/cell · lengths in Å"
        self._potential_enabled = potential
        self._ready = True

    @traitlets.validate("specimen_tilt_mrad")
    def _validate_specimen_tilt(self, proposal):
        """Keep live and exported specimen tilts inside the preview range."""
        value = proposal["value"]
        if len(value) != 2 or any(not np.isfinite(v) or abs(v) > 15 for v in value):
            raise traitlets.TraitError(
                "specimen_tilt_mrad must contain (row, column), each finite "
                "and between -15 and +15 mrad."
            )
        return value

    @traitlets.validate("energy_keV")
    def _validate_energy(self, proposal):
        value = proposal["value"]
        if not np.isfinite(value) or not 0 <= value <= 300:
            raise traitlets.TraitError(
                "energy_keV must be finite and between 0 and 300; phase is undefined at zero."
            )
        return value

    @traitlets.validate("field_of_view_A")
    def _validate_fov(self, proposal):
        value = proposal["value"]
        if not np.isfinite(value) or value <= 0:
            raise traitlets.TraitError(
                "field_of_view_A must be finite and positive, in Å."
            )
        return value

    @traitlets.validate("magnification_calibration")
    def _validate_magnification(self, proposal):
        value = proposal["value"]
        if value and (
            len(value) != 2 or any(not np.isfinite(v) or v <= 0 for v in value)
        ):
            raise traitlets.TraitError(
                "Supply (reference magnification, reference field of view in Å), "
                "both positive and finite; use [] to clear calibration."
            )
        return value

    @traitlets.validate("potential_pixels")
    def _validate_potential_pixels(self, proposal):
        if proposal["value"] not in (128, 256, 512):
            raise traitlets.TraitError("potential_pixels must be 128, 256 or 512.")
        return proposal["value"]

    @traitlets.validate("potential_sigma_A")
    def _validate_potential_sigma(self, proposal):
        value = proposal["value"]
        if not np.isfinite(value) or not 0.04 <= value <= 0.5:
            raise ValueError("potential_sigma_A must be between 0.04 and 0.5 Å.")
        return value

    @traitlets.observe("potential_sigma_A")
    def _sync_potential_sigma(self, change):
        if not getattr(self, "_ready", False) or not self._potential_enabled:
            return
        from ._showcif_potential import projected_atom_table

        self.potential_bytes = np.stack(
            [projected_atom_table(s["symbol"], change["new"]) for s in self.species]
        ).tobytes()

    @traitlets.validate("num_slices")
    def _validate_slices(self, proposal):
        if isinstance(proposal["value"], bool) or not 1 <= proposal["value"] <= 64:
            raise traitlets.TraitError("num_slices must be from 1 to 64.")
        return proposal["value"]

    @traitlets.validate("repeats")
    def _validate_repeats(self, proposal):
        value = proposal["value"]
        if len(value) != 3 or any(isinstance(v, bool) or v < 1 for v in value):
            raise traitlets.TraitError(
                "repeats must contain three positive integers along a, b, c."
            )
        if (
            hasattr(self, "_atoms")
            and len(self._atoms) * value[0] * value[1] * value[2] > 250_000
        ):
            raise traitlets.TraitError(
                "Use a smaller inspection region (at most 250,000 atoms)."
            )
        return value

    @traitlets.observe("repeats")
    def _sync_repeats(self, change):
        """Keep Python exports consistent; the browser instances the unit cell on GPU."""
        if not getattr(self, "_ready", False):
            return
        expanded = self._atoms.repeat(tuple(change["new"]))
        numbers = sorted(set(self._atoms.numbers.tolist()))
        packed = np.empty((len(expanded), 4), dtype=np.float32)
        packed[:, :3] = expanded.positions
        packed[:, 3] = [numbers.index(int(z)) for z in expanded.numbers]
        with self.hold_sync():
            self.atom_bytes = packed.tobytes()
            self.atom_count = len(expanded)
            self.cell = expanded.cell.array.tolist()
            self.species = [
                dict(s, count=int(np.count_nonzero(expanded.numbers == s["number"])))
                for s in self.species
            ]

    @traitlets.validate("zone_axis")
    def _validate_zone(self, proposal):
        value = proposal["value"]
        if len(value) != 3 or not any(value) or any(isinstance(v, bool) for v in value):
            raise traitlets.TraitError(
                "zone_axis must contain three integers [u, v, w], not all zero."
            )
        return value

    @traitlets.validate("visible_species")
    def _validate_visibility(self, proposal):
        value = proposal["value"]
        if self.species and len(value) != len(self.species):
            raise traitlets.TraitError(
                "Provide one visibility flag per chemical species."
            )
        return value

    def atoms(self) -> Atoms:
        """Return the original unit cell without display filters or specimen tilt.

        Returns
        -------
        ase.Atoms
            Independent copy of the supplied structure, before repeats.
        """
        return self._atoms.copy()

    def export_html(self, path: str | Path) -> Path:
        """Save an interactive structure viewer with its current synced settings.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination HTML file. Missing parent directories are created.

        Returns
        -------
        pathlib.Path
            The saved HTML path.

        Notes
        -----
        Atomic coordinates and potential tables are embedded. The widget
        manager loads from a CDN, so opening the file requires network access
        and a WebGPU-capable secure browser context. Transient camera, playback,
        and panel-visibility choices are not serialized.
        """
        from ipywidgets.embed import dependency_state, embed_minimal_html

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        embed_minimal_html(
            str(path),
            views=[self],
            state=dependency_state([self], drop_defaults=False),
            drop_defaults=False,
            title=self.title,
        )
        return path
