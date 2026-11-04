"""Interactive electron diffraction simulator for web pages and Dash apps.

The simulator is one ES module (``static/diffraction-sim.js``) with
``export default { render }``, built from the quantem.widget sources by
``npm run build:web``. Every calculation runs in the browser: the page passes
a structure as CIF text and the settings, and the bundle computes the
reflection list, the Lobato structure factors and the absorptive
Weickenmeier-Kohl Bloch couplings itself. This module converts structures to
CIF, checks settings and writes standalone pages; :mod:`quantem.diffsim.dash`
mounts the simulator in a Dash app.
"""

from __future__ import annotations

import base64
import html as _html
import io
import json
import pathlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("quantem.diffsim")
except PackageNotFoundError:  # a source checkout on sys.path
    __version__ = "0.0.0+local"

BUNDLE_PATH = pathlib.Path(__file__).parent / "static" / "diffraction-sim.js"

#: Settings the browser bundle reads (js/diffractionsim-web/index.ts).
SETTINGS = (
    "preset",
    "zone_axis",
    "mode",
    "dynamical",
    "energy_keV",
    "thickness_A",
    "semiangle_mrad",
    "precession_deg",
    "pattern_range_inv_A",
    "field_mrad",
    "thermal_sigma_A",
    "power",
    "vmax",
    "show_kikuchi",
    "show_hkl",
    "show_labels",
    "show_ewald",
    "polyhedra",
    "n_cells",
    "panel_width_px",
    "rotation_speed_deg_per_s",
    "title",
    "structure_name",
    "show_logo",
)
MODES = ("nanobeam", "cbed", "kikuchi")


def bundle_path() -> pathlib.Path:
    """Path of the browser bundle, ``diffraction-sim.js``.

    Raises
    ------
    FileNotFoundError
        In a source checkout where ``npm run build:web`` has not been run.
    """
    if not BUNDLE_PATH.is_file():
        raise FileNotFoundError(
            f"{BUNDLE_PATH} is missing; run `npm run build:web` in the quantem.widget repository."
        )
    return BUNDLE_PATH


def structure_to_cif(structure: Any, conventional: bool = True) -> str:
    """CIF text of a crystal structure.

    Parameters
    ----------
    structure : pymatgen Structure, ase.Atoms, str or pathlib.Path
        A pymatgen ``Structure``, an ASE ``Atoms`` object, CIF text, or the
        path of a CIF file.
    conventional : bool, default True
        For a pymatgen ``Structure``, write its conventional standard cell
        (``SpacegroupAnalyzer.get_conventional_standard_structure``), so that
        zone axes refer to the conventional lattice, with its space group and
        symmetry operations. ``False`` writes the cell as given. Other inputs
        are written as given.

    Returns
    -------
    str
        CIF text: the given text, the conventional cell with its symmetry
        operations, or every atom of the cell (space group P1).
    """
    if isinstance(structure, pathlib.Path):
        return structure.read_text()
    if isinstance(structure, str):
        if "_cell_length_a" in structure:
            return structure
        path = pathlib.Path(structure)
        if path.is_file():
            return path.read_text()
        raise ValueError("structure is a string but neither CIF text nor the path of a CIF file")
    module = type(structure).__module__
    if module.startswith("pymatgen"):
        from pymatgen.io.cif import CifWriter

        if conventional:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
            try:
                return str(CifWriter(structure, symprec=0.01))
            except Exception:  # noqa: BLE001  (a structure spglib cannot symmetrize is written in P1)
                pass
        return str(CifWriter(structure))
    if module.startswith("ase"):
        from ase.io import write

        buffer = io.BytesIO()
        write(buffer, structure, format="cif")
        return buffer.getvalue().decode()
    raise TypeError(
        f"structure must be a pymatgen Structure, an ase.Atoms, CIF text or a CIF path, got {type(structure).__name__}"
    )


def settings(structure: Any = None, *, conventional: bool = True, **values: Any) -> dict[str, Any]:
    """The settings dictionary the browser bundle reads.

    Parameters
    ----------
    structure : pymatgen Structure, ase.Atoms, str or pathlib.Path, optional
        The crystal, converted by :func:`structure_to_cif`. It replaces the
        structure menu of the simulator. ``None`` shows the preset structures.
    conventional : bool, default True
        Passed to :func:`structure_to_cif`.
    **values
        Any of :data:`SETTINGS`, for example ``zone_axis=[0, 0, 1]``,
        ``mode="nanobeam"``, ``energy_keV=200`` or ``thickness_A=500``.

    Returns
    -------
    dict
        JSON-serializable settings, with the CIF text under ``"cif"``.

    Raises
    ------
    ValueError
        For an unknown setting name or mode.
    """
    unknown = sorted(set(values) - set(SETTINGS))
    if unknown:
        raise ValueError(f"unknown settings {unknown}; the simulator reads {list(SETTINGS)}")
    if "mode" in values and values["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {values['mode']!r}")
    out = {key: (list(value) if isinstance(value, tuple) else value) for key, value in values.items()}
    if structure is not None:
        out["cif"] = structure_to_cif(structure, conventional=conventional)
        if "structure_name" not in out and _formula(structure):
            out["structure_name"] = _formula(structure)
    json.dumps(out)  # the browser receives JSON: fail here, not in the page
    return out


def html(
    structure: Any = None,
    *,
    page_title: str | None = None,
    bundle_url: str | None = None,
    **values: Any,
) -> str:
    """A standalone HTML page with one simulator.

    Parameters
    ----------
    structure : pymatgen Structure, ase.Atoms, str or pathlib.Path, optional
        The crystal, as for :func:`settings`.
    page_title : str, optional
        Browser tab title; defaults to the ``title`` setting.
    bundle_url : str, optional
        URL of ``diffraction-sim.js`` to load. By default the bundle is
        inlined, so the page works offline (about 1.2 MB).
    **values
        Settings, as for :func:`settings`.

    Returns
    -------
    str
        The HTML page.
    """
    data = settings(structure, **values)
    title = page_title or data.get("title") or "Electron diffraction simulator"
    if bundle_url is None:
        # base64, so no sequence in the bundle can end the script element early
        packed = base64.b64encode(bundle_path().read_bytes()).decode("ascii")
        loader = (
            "const packed = atob(document.getElementById('quantem-diffsim-bundle').textContent);\n"
            "const url = URL.createObjectURL(new Blob([Uint8Array.from(packed, (c) => c.charCodeAt(0))],"
            " { type: 'text/javascript' }));"
        )
        bundle_tag = f'<script type="text/plain" id="quantem-diffsim-bundle">{packed}</script>\n'
    else:
        loader = f"const url = {json.dumps(bundle_url)};"
        bundle_tag = ""
    settings_json = json.dumps(data).replace("</", "<\\/")
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_html.escape(title)}</title>\n</head>\n<body>\n"
        '<div id="quantem-diffsim"></div>\n'
        f"{bundle_tag}"
        f'<script type="application/json" id="quantem-diffsim-settings">{settings_json}</script>\n'
        '<script type="module">\n'
        f"{loader}\n"
        "const settings = JSON.parse(document.getElementById('quantem-diffsim-settings').textContent);\n"
        "const sim = (await import(url)).default;\n"
        "sim.render({ model: { get: (key) => settings[key] }, el: document.getElementById('quantem-diffsim') });\n"
        "</script>\n</body>\n</html>\n"
    )


def save_html(path: str | pathlib.Path, structure: Any = None, **kwargs: Any) -> pathlib.Path:
    """Write :func:`html` to a file and return its path.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file. Missing parent directories are created.
    structure : pymatgen Structure, ase.Atoms, str or pathlib.Path, optional
        The crystal, as for :func:`settings`.
    **kwargs
        ``page_title``, ``bundle_url`` and settings, as for :func:`html`.
    """
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html(structure, **kwargs), encoding="utf-8")
    return out


def _formula(structure: Any) -> str:
    """Reduced formula of a pymatgen or ASE structure, empty for CIF text."""
    composition = getattr(structure, "composition", None)
    if composition is not None and hasattr(composition, "reduced_formula"):
        return str(composition.reduced_formula)
    if hasattr(structure, "get_chemical_formula"):
        return str(structure.get_chemical_formula(empirical=True))
    return ""


__all__ = ["BUNDLE_PATH", "MODES", "SETTINGS", "bundle_path", "html", "save_html", "settings", "structure_to_cif"]
