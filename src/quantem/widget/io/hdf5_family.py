"""Resolve the files that form one compressed HDF5 detector acquisition."""

from __future__ import annotations

from pathlib import Path


def _external_data_files(master: Path) -> list[Path]:
    """Return data files referenced by external links in a wrapper master."""

    try:
        import h5py
    except ImportError:
        return []

    files: list[Path] = []
    try:
        with h5py.File(master, "r") as handle:
            group = handle.get("entry/data")
            if group is None:
                return []
            for name in group:
                link = group.get(name, getlink=True)
                if not isinstance(link, h5py.ExternalLink) or not link.filename:
                    continue
                source = Path(link.filename)
                if not source.is_absolute():
                    source = master.parent / source
                files.append(source.expanduser().resolve())
    except OSError:
        return []

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(
                f"HDF5 wrapper {master} points at missing data file {path}"
            )
        unique.append(path)
        seen.add(path)
    return unique


def collect_hdf5_family(master: str | Path) -> list[Path]:
    """Return a master followed by every compressed detector-data file.

    Parameters
    ----------
    master
        Native ``*_master.h5`` file or a wrapper master containing HDF5
        external links.

    Returns
    -------
    list[pathlib.Path]
        Resolved master path followed by its detector-data files.
    """

    master = Path(master).expanduser().resolve()
    external = _external_data_files(master)
    if external:
        return [master, *external]
    if not master.name.endswith("_master.h5"):
        raise ValueError(
            "HDF5 source must be a *_master.h5 file or a wrapper with external "
            f"data links; got {master.name!r}"
        )
    prefix = master.name[: -len("_master.h5")]
    data_files = sorted(master.parent.glob(f"{prefix}_data_*.h5"))
    if not data_files:
        raise FileNotFoundError(
            f"no HDF5 data files found next to {master}: "
            f"expected {prefix}_data_*.h5"
        )
    return [master, *data_files]
