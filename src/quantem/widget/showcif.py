"""Inspect crystallographic atoms and projected columns without changing the model."""

import re
from collections.abc import Sequence
from functools import lru_cache
from math import prod
from pathlib import Path

import anywidget
import numpy as np
import traitlets
from ase import Atoms
from ase.data.colors import jmol_colors
from ase.io import read
from scipy.integrate import simpson
from scipy.special import j0


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
    title : str, optional
        Heading above the two views. Default: the CIF's own formula
        (``_chemical_formula_structural``, else ``_chemical_formula_sum``), else
        the Hill formula (C, H, then alphabetical), then " · Atomic structure".
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

    Notes
    -----
    Specimen tilt (``specimen_tilt_mrad``, (row, column) from -15 to +15 mrad
    about the inspection cell center), the microscope field of view
    (``view_mode``, ``field_of_view_A``, ``magnification_calibration``),
    orthogonal depth views (``orthogonal_views``) and the potential preview
    filter (``potential_sigma_A``, ``potential_pixels``) are synced traits set
    after construction, as the browser controls do.

    Species visibility, clipping and sphere radii are display settings only.
    The atom projection shows positions, not intensity or phase. Optional
    potential maps use neutral Lobato infinite atomic projections assigned by
    site depth; they are not finite-z potential integrals or reconstruction data.
    WebGPU is required for rendering. No structure editing is performed.

    Examples
    --------
    >>> viewer = ShowCIF("crystal.cif", repeats=(4, 4, 8))
    >>> viewer.specimen_tilt_mrad = [5.0, -2.0]
    """

    _esm = Path(__file__).parent / "static" / "showcif.js"
    title = traitlets.Unicode("").tag(sync=True)
    unit_atom_bytes = traitlets.Bytes(b"").tag(sync=True)
    unit_cell = traitlets.List(default_value=[]).tag(sync=True)
    species = traitlets.List(default_value=[]).tag(sync=True)
    repeats = traitlets.List(traitlets.Int(), default_value=[3, 3, 3]).tag(sync=True)
    zone_axis = traitlets.List(traitlets.Int(), default_value=[0, 0, 1]).tag(sync=True)
    specimen_tilt_mrad = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    visible_species = traitlets.List(traitlets.Bool(), default_value=[]).tag(sync=True)
    source_summary = traitlets.Unicode("").tag(sync=True)
    potential_bytes = traitlets.Bytes(b"").tag(sync=True)
    potential_sigma_A = traitlets.Float(0.08).tag(sync=True)
    potential_pixels = traitlets.Int(256).tag(sync=True)
    energy_keV = traitlets.Float(300).tag(sync=True)
    potential_colormap = traitlets.Enum(
        ["inferno", "viridis", "plasma", "magma", "magenta", "hot", "gray", "hsv", "turbo",
         "cividis", "RdBu", "RdBu_r", "seismic", "twilight", "twilight_shifted"],
        default_value="viridis",
    ).tag(sync=True)
    potential_quantity = traitlets.Enum(["integrated", "average", "phase"], default_value="average").tag(sync=True)
    num_slices = traitlets.Int(16).tag(sync=True)
    view_mode = traitlets.Enum(["unit_cells", "microscope"], default_value="unit_cells").tag(sync=True)
    orthogonal_views = traitlets.Bool(False).tag(sync=True)
    field_of_view_A = traitlets.Float(40).tag(sync=True)
    magnification_calibration = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)

    _atoms: Atoms | None = None
    _potential_enabled = False

    def __init__(
        self,
        structure: str | Path | Atoms,
        *,
        repeats: Sequence[int] = (3, 3, 3),
        zone_axis: Sequence[int] = (0, 0, 1),
        title: str = "",
        energy_keV: float = 300,
        potential_colormap: str = "viridis",
        potential_quantity: str = "average",
        num_slices: int = 16,
        potential: bool = False,
    ) -> None:
        if isinstance(num_slices, bool) or not isinstance(num_slices, int) or not 1 <= num_slices <= 64:
            raise ValueError("num_slices must be an integer from 1 to 64.")
        repeat_counts = list(repeats)
        if len(repeat_counts) != 3 or any(
            isinstance(count, bool) or not np.isfinite(count) or int(count) != count or count < 1 for count in repeat_counts
        ):
            raise ValueError("repeats must contain three positive integers along a, b, c.")
        zone = list(zone_axis)
        if len(zone) != 3 or not any(zone) or any(isinstance(index, bool) or not np.isfinite(index) or int(index) != index for index in zone):
            raise ValueError("zone_axis must be three integers [u, v, w], not all zero.")
        if isinstance(structure, Atoms):
            atoms = structure.copy()
            source = "ASE structure"
        else:
            path = Path(structure).expanduser()
            if not path.is_file():
                raise ValueError(f"CIF not found: {path}. Supply an existing CIF path or ase.Atoms.")
            # CIF tags carry the formula the file's authors wrote (BaTiO3), used for the title.
            atoms = read(path, store_tags=True) if path.suffix.lower() == ".cif" else read(path)
            source = path.name
        if (
            not len(atoms)
            or atoms.cell.rank != 3
            or not np.all(np.isfinite(atoms.positions))
            or not np.all(np.isfinite(atoms.cell.array))
            or abs(np.linalg.det(atoms.cell.array)) < 1e-12
        ):
            raise ValueError("Supply a nonempty crystal with a finite 3D unit cell and atomic positions.")
        occupancy = atoms.info.get("occupancy", {})
        if any(
            len(site) != 1 or any(not np.isfinite(float(fraction)) or abs(float(fraction) - 1) > 1e-6 for fraction in site.values())
            for site in occupancy.values()
        ):
            raise ValueError("Partial or mixed occupancies need an explicit ordered structure; they are not silently filled.")
        n_cells = prod(int(count) for count in repeat_counts)
        count = len(atoms) * n_cells
        if count > 250_000:
            raise ValueError(f"Requested {count:,} atoms; use a smaller inspection region (at most 250,000).")
        super().__init__()
        self._atoms = atoms.copy()
        numbers = sorted(set(atoms.numbers.tolist()))
        # (n, 4) float32 rows of x, y, z and the species index the browser instances.
        unit_atoms = np.empty((len(atoms), 4), dtype=np.float32)
        unit_atoms[:, :3] = atoms.positions
        unit_atoms[:, 3] = [numbers.index(int(z)) for z in atoms.numbers]
        symbols = {int(z): symbol for z, symbol in zip(atoms.numbers, atoms.get_chemical_symbols())}
        # Title formula as crystallographers write it: the CIF's own formula in its order, else the
        # Hill system (C, H, then alphabetical). Spaces and counts of 1 go: "Ba1 Ti1 O3" reads BaTiO3.
        # A tag written inside a loop_ comes back from ASE as a list of rows.
        tags = [atoms.info.get(key, "") for key in ("_chemical_formula_structural", "_chemical_formula_sum")]
        written = [str(tag[0] if isinstance(tag, list) else tag).strip() for tag in tags]
        written = [value for value in written if value not in ("", "?", ".")]
        formula = re.sub(r"(?<=[A-Za-z])1(?![\d.])", "", "".join(written[0].split())) if written else atoms.get_chemical_formula("hill")
        with self.hold_sync():
            self.energy_keV = energy_keV
            self.potential_colormap = potential_colormap
            self.potential_quantity = potential_quantity
            self.num_slices = num_slices
            if potential:
                self.potential_bytes = np.stack([projected_atom_table(symbols[z], self.potential_sigma_A) for z in numbers]).tobytes()
            self.title = title or formula + " · Atomic structure"
            self.unit_atom_bytes = unit_atoms.tobytes()
            self.unit_cell = atoms.cell.array.tolist()
            self.species = [
                {
                    "symbol": symbols[z],
                    "number": z,
                    "color": jmol_colors[z].tolist(),
                    "count": int(np.count_nonzero(atoms.numbers == z)) * n_cells,
                }
                for z in numbers
            ]
            self.repeats = [int(count) for count in repeat_counts]
            self.zone_axis = [int(index) for index in zone]
            self.visible_species = [True] * len(numbers)
            self.source_summary = f"{source} · {len(atoms)} atoms/cell · lengths in Å"
        self._potential_enabled = potential

    @traitlets.validate("specimen_tilt_mrad")
    def _validate_specimen_tilt(self, proposal):
        """Keep live and exported specimen tilts inside the preview range."""
        value = proposal["value"]
        if len(value) != 2 or any(not np.isfinite(v) or abs(v) > 15 for v in value):
            raise traitlets.TraitError("specimen_tilt_mrad must contain (row, column), each finite and between -15 and +15 mrad.")
        return value

    @traitlets.validate("energy_keV")
    def _validate_energy(self, proposal):
        """Keep the electron energy in the 0 to 300 keV range the phase preview is defined for."""
        value = proposal["value"]
        if not np.isfinite(value) or not 0 <= value <= 300:
            raise traitlets.TraitError("energy_keV must be finite and between 0 and 300; phase is undefined at zero.")
        return value

    @traitlets.validate("field_of_view_A")
    def _validate_fov(self, proposal):
        """Refuse a non-positive microscope field of view, which has no projection."""
        value = proposal["value"]
        if not np.isfinite(value) or value <= 0:
            raise traitlets.TraitError("field_of_view_A must be finite and positive, in Å.")
        return value

    @traitlets.validate("magnification_calibration")
    def _validate_magnification(self, proposal):
        """Accept [] (no calibration) or one positive (magnification, field of view) pair."""
        value = proposal["value"]
        if value and (len(value) != 2 or any(not np.isfinite(v) or v <= 0 for v in value)):
            raise traitlets.TraitError(
                "Supply (reference magnification, reference field of view in Å), both positive and finite; use [] to clear calibration."
            )
        return value

    @traitlets.validate("potential_pixels")
    def _validate_potential_pixels(self, proposal):
        """Only the 128, 256 and 512 px potential preview grids."""
        if proposal["value"] not in (128, 256, 512):
            raise traitlets.TraitError("potential_pixels must be 128, 256 or 512.")
        return proposal["value"]

    @traitlets.validate("potential_sigma_A")
    def _validate_potential_sigma(self, proposal):
        """Keep the preview filter width in the range :func:`projected_atom_table` tabulates."""
        value = proposal["value"]
        if not np.isfinite(value) or not 0.04 <= value <= 0.5:
            raise traitlets.TraitError("potential_sigma_A must be between 0.04 and 0.5 Å.")
        return value

    @traitlets.observe("potential_sigma_A")
    def _sync_potential_sigma(self, change):
        """The browser's Blur control re-requests the atomic tables at the new width."""
        if self._potential_enabled:
            self.potential_bytes = np.stack([projected_atom_table(entry["symbol"], change["new"]) for entry in self.species]).tobytes()

    @traitlets.validate("num_slices")
    def _validate_slices(self, proposal):
        """Keep the preview slab count from 1 to 64."""
        if isinstance(proposal["value"], bool) or not 1 <= proposal["value"] <= 64:
            raise traitlets.TraitError("num_slices must be from 1 to 64.")
        return proposal["value"]

    @traitlets.validate("repeats")
    def _validate_repeats(self, proposal):
        """Three positive repeats whose supercell stays within the 250,000-atom display limit."""
        value = proposal["value"]
        if len(value) != 3 or any(isinstance(count, bool) or count < 1 for count in value):
            raise traitlets.TraitError("repeats must contain three positive integers along a, b, c.")
        if self._atoms is not None and len(self._atoms) * value[0] * value[1] * value[2] > 250_000:
            raise traitlets.TraitError("Use a smaller inspection region (at most 250,000 atoms).")
        return value

    @traitlets.observe("repeats")
    def _sync_repeats(self, change):
        """Species counts follow the supercell; the browser instances the unit cell on GPU."""
        if self._atoms is None:
            return
        n_cells = prod(int(count) for count in change["new"])
        self.species = [
            dict(entry, count=int(np.count_nonzero(self._atoms.numbers == entry["number"])) * n_cells) for entry in self.species
        ]

    @traitlets.validate("zone_axis")
    def _validate_zone(self, proposal):
        """A [uvw] direction: three integers, not all zero."""
        value = proposal["value"]
        if len(value) != 3 or not any(value) or any(isinstance(index, bool) for index in value):
            raise traitlets.TraitError("zone_axis must contain three integers [u, v, w], not all zero.")
        return value

    @traitlets.validate("visible_species")
    def _validate_visibility(self, proposal):
        """One visibility flag per species, so a flag never hides the wrong element."""
        value = proposal["value"]
        if self.species and len(value) != len(self.species):
            raise traitlets.TraitError("Provide one visibility flag per chemical species.")
        return value


# --- projected potential tables ----------------------------------------------------


@lru_cache(maxsize=32)
def projected_atom_table(symbol: str, sigma: float = 0.08) -> np.ndarray:
    """Return a Lobato projected potential table in V Å.

    The radial table is the inverse Hankel transform of abTEM's Lobato
    projected scattering factor times a Gaussian ``exp(-2 pi² sigma² k²)``:
    ``V(r) = ∫ f(k) exp(-2 pi² sigma² k²) J0(2 pi k r) 2 pi k dk``. The
    Gaussian is an explicit transverse preview filter (``sigma`` in Å, 0.04 to
    0.5), not thermal displacement or a fitted resolution. Samples run every
    0.005 Å from 0 to 8 Å inclusive (1601 float32 values).

    These are infinite atomic projections assigned to one depth slab per site:
    not finite-z integration, bonding charge density or an electron-wave
    simulation. Cached per (symbol, sigma) because every sigma slider move
    re-requests the full species set.
    """
    if not np.isfinite(sigma) or not 0.04 <= sigma <= 0.5:
        raise ValueError("potential_sigma_A must be between 0.04 and 0.5 Å.")
    try:
        from abtem.parametrizations import LobatoParametrization
    except ImportError as exc:
        raise ImportError(
            'Potential previews require abTEM: pip install "quantem.widget[crystal]", '
            "or use ShowCIF(..., potential=False) for atoms only."
        ) from exc
    # k_max gives exp(-32) Gaussian attenuation. dk resolves the full 8 Å table.
    k = np.linspace(0, 4 / (np.pi * sigma), 2049)
    radius = np.arange(1601, dtype=np.float64) * 0.005
    spectrum = LobatoParametrization().projected_scattering_factor(symbol)(k * k)
    spectrum = spectrum * np.exp(-2 * np.pi**2 * sigma**2 * k**2) * (2 * np.pi * k)
    table = simpson(j0(2 * np.pi * radius[:, None] * k) * spectrum, x=k, axis=1)
    if not np.all(np.isfinite(table)) or np.min(table) < -1e-3:
        raise ValueError(f"Invalid projected potential for {symbol}; check the abTEM parameters.")
    result = np.maximum(table, 0).astype(np.float32)
    result.flags.writeable = False
    return result
