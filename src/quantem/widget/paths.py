"""Path helpers shared across notebooks."""
from __future__ import annotations

from pathlib import Path


def first_existing(*paths: "str | Path") -> Path:
    """Return the first path from ``paths`` that exists on disk.

    A notebook that runs on multiple hosts often has the same session
    dataset mounted at different absolute paths (shared NFS on one box,
    a local SSD mirror on another). Instead of writing a
    ``next(p for p in [...] if p.exists())`` inline in every notebook,
    pass every candidate to ``first_existing`` and let it pick the one
    that resolves on this host.

    Raises
    ------
    FileNotFoundError
        If none of the candidates exist. The error message lists all
        candidates so the operator knows which mount points were tried.

    Example
    -------
    >>> from quantem.widget import first_existing
    >>> SESSION = first_existing(
    ...     "/data/shared/microscopy-session",
    ...     "data/microscopy-session",
    ... )
    """
    if not paths:
        raise ValueError("first_existing() requires at least one candidate path")
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    tried = "\n".join(f"  - {p}" for p in paths)
    raise FileNotFoundError(
        f"None of the candidate paths exist on this host:\n{tried}"
    )
