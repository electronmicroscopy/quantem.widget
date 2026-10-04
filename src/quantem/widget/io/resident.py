"""Native-count residency for interactive acquisition viewers."""

from pathlib import Path
from typing import TYPE_CHECKING

from quantem.gpu import io

if TYPE_CHECKING:
    from quantem.gpu.detector import DetectorSession


def load_resident(
    source: str | Path,
    *,
    backend: str = "cuda",
    representation: str = "encoded",
    device: int | None = None,
) -> io.Dataset4dstem:
    """Keep a complete acquisition available for repeated detector queries.

    Stored detector-mask pixels are replaced by their valid 3x3-neighbor
    median on the GPU before resident encoding. The source HDF5 remains
    unchanged, and correction provenance is retained in the load metadata.
    Memory pressure raises an error instead of binning or clipping counts.

    Parameters
    ----------
    source : str or Path
        Original acquisition master on the compute host.
    backend : str
        Compute backend, normally ``cuda`` for remote Browse.
    representation : str
        ``encoded`` keeps exact native counts for bounded detector queries.
    device : int, optional
        Process-visible device index.

    Returns
    -------
    quantem.core.datastructures.Dataset4dstem
        Owned resident source. Close it after closing its detector sessions.

    Examples
    --------
    >>> with load_resident("scan_master.h5") as loaded:
    ...     session = prepare_resident(loaded)
    ...     pattern = session.frame(0)
    ...     session.close()
    """
    if representation not in {"packed", "encoded"}:
        raise ValueError(
            f"Unknown resident representation {representation!r}; "
            "choose 'packed' or 'encoded'."
        )
    return io.load(
        source, backend=backend, representation=representation,
        dtype="native", device=device, apply_mask=False,
        hot_pixel_correction="median", verbose=False,
    )


def prepare_resident(loaded: io.Dataset4dstem) -> "DetectorSession":
    """Prepare exact indexed detector queries without expanding the acquisition.

    Parameters
    ----------
    loaded : quantem.core.datastructures.Dataset4dstem
        Owned source returned by :func:`load_resident`.

    Returns
    -------
    quantem.gpu.detector.DetectorSession
        Borrowing session. CUDA uses a one-acquisition series to prepare the
        shared spatial index; result arrays retain that leading axis.

    Examples
    --------
    >>> with load_resident("scan_master.h5") as loaded:
    ...     session = prepare_resident(loaded)
    ...     mean = session.mean_dp().reshape(loaded.shape[2:])
    ...     session.close()
    """
    from quantem.gpu import detector

    return detector.prepare([loaded] if loaded.metadata["backend"] == "cuda" else loaded)
