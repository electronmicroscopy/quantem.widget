"""Compatibility shim for the migrated HDF5 GPU IO path.

The accelerated HDF5 load/decompress implementation now lives in
``quantem.gpu.io.hdf5``. This module preserves the historical
``quantem.widget.io.hdf5`` import path for one release while widget callers
move to the new package.
"""
from __future__ import annotations

from quantem.gpu.io.hdf5 import *  # noqa: F401,F403
from quantem.gpu.io.hdf5 import (  # noqa: F401
    __all__,
    __version__,
    _clip_to_uint8,
    _clip_to_uint8_count,
)


def _disk_interleaved_indices(paths) -> list[int]:
    """Return indices ordered to touch different physical disks early."""
    buckets: dict[str, list[int]] = {}
    disk_order: list[str] = []
    for idx, path in enumerate(paths):
        disk = disk_of(path)
        if disk not in buckets:
            disk_order.append(disk)
            buckets[disk] = []
        buckets[disk].append(idx)

    queues = [list(buckets[disk]) for disk in disk_order]
    order: list[int] = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop(0))
    return order


def _assign_indices_to_devices(filepaths, devices) -> dict[int, list[int]]:
    """Assign file indices to devices using disk-interleaved round robin."""
    devices = [int(device) for device in devices]
    if not devices:
        raise ValueError("devices must contain at least one CUDA device")

    assign: dict[int, list[int]] = {device: [] for device in devices}
    for offset, idx in enumerate(_disk_interleaved_indices(filepaths)):
        assign[devices[offset % len(devices)]].append(idx)
    return assign
