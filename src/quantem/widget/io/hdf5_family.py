"""Open HDF5 files with hdf5plugin's filters."""

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
