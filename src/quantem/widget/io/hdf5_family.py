"""Open HDF5 files with hdf5plugin's filters; resolve the files of one compressed detector acquisition."""

import contextlib
from pathlib import Path


def open_hdf5(path):
    """Open ``path`` read-only with h5py, after hdf5plugin has registered its compression filters.

    Every reader that decodes HDF5 data opens files here, so a bitshuffle-LZ4
    or other hdf5plugin-compressed dataset reads whichever reader runs first.
    hdf5plugin is imported only now, and before h5py, which hdf5plugin 1.x
    requires. It is optional: Windows on ARM has no hdf5plugin wheel, and
    files without its filters open without it.
    """
    with contextlib.suppress(ImportError):
        import hdf5plugin  # noqa: F401  registers bitshuffle-LZ4, LZ4, Zstd and other filters with h5py
    import h5py

    return h5py.File(Path(path), "r")


def _external_data_files(master: Path) -> list[Path]:
    """Return data files referenced by external links in a wrapper master.

    A wrapper master names its detector files through HDF5 external links
    instead of the ``<prefix>_data_*.h5`` convention, so the family cannot be
    found by file name. A file that is not HDF5 has no links and returns [].
    """
    import h5py

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
    unique = list(dict.fromkeys(files))
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(
                f"HDF5 wrapper {master} points at missing data file {path}"
            )
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
