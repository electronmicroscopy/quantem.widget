"""Crash-safe + concurrency-safe file IO helpers.

Calibration libraries (`_calibrations.json` and `~/quantem/calibrations.json`)
get hammered concurrently by the dashboard's "Save calibration" button and
the screener's auto-save thread, plus they're long-lived files that must
survive a Ctrl-C or SIGKILL mid-write.

Two primitives:

- :func:`atomic_write_json` writes to ``<path>.tmp`` (created via
  :func:`tempfile.mkstemp` in the same directory so ``os.replace`` can be
  truly atomic), ``fsync`` 's the data to disk, then ``os.replace``\\ s the
  tmp into the final path. A SIGKILL leaves either the previous-valid file
  or the new-valid file, never a half-written one.
- :func:`locked_json_rmw` wraps a read-modify-write on a JSON file under
  an OS-level :func:`fcntl.flock` ``LOCK_EX`` taken on a sidecar
  ``<path>.lock`` file. Two processes calling ``save_calibrations`` at the
  same time serialize through the lock, so neither loses the other's
  appended entry.

Why a sidecar lock and not the file itself: ``os.replace`` re-creates the
inode. A flock held on the original inode does not transfer to the
replacement, so concurrent writers each end up locking different inodes
and the lock is a no-op. Locking a stable sidecar (``<path>.lock``) keeps
the lock identity independent of the data file.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def atomic_write_json(path: str | Path, data: object, *, indent: int = 2) -> Path:
    """Atomically write ``data`` to ``path`` as JSON.

    Crash-safe: the final file is either the previous valid content or the
    newly-written content, never a truncation. Implemented via
    ``mkstemp`` + ``write`` + ``fsync`` + ``os.replace`` in the same
    directory as the target so the rename is atomic on POSIX.

    Returns the final path so callers can chain.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def locked_json_rmw(
    path: str | Path,
    update: Callable[[object | None], object],
    *,
    indent: int = 2,
) -> object:
    """Read-modify-write a JSON file under an exclusive OS-level lock.

    ``update`` receives the parsed JSON value (or ``None`` if the file
    does not exist or is empty) and must return the new value to write.
    The whole sequence runs while a sidecar ``<path>.lock`` file is held
    under ``fcntl.flock(LOCK_EX)``, so a second process attempting the
    same call blocks until the first releases.

    Returns the value ``update`` returned (post-write).

    Implementation note: dot-prefixed sidecars are forbidden by the
    project's no-hidden-files rule, so the lock file is named
    ``<path>.lock`` (visible). It is created the first time and reused
    forever; never deleted, since deleting it would race with another
    process trying to flock it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    # Open the lock file (create-if-missing) and hold an exclusive flock
    # for the whole read-modify-write. Closing the fd releases the lock.
    lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        current: object | None
        if path.exists():
            raw = path.read_text().strip()
            current = json.loads(raw) if raw else None
        else:
            current = None
        new_value = update(current)
        atomic_write_json(path, new_value, indent=indent)
        return new_value
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(lock_fd)
