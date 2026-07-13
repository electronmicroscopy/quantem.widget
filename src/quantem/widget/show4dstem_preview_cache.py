"""Persistent reduced-preview cache for folder-backed Show4DSTEM viewers.

The cache deliberately stores only derived float32 virtual images.  Raw 4D
detector data remain owned by the lazy Dataset5dstem CUDA residency layer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time
from typing import Any

import numpy as np


_CACHE_SCHEMA = 1
_REDUCTION_ALGORITHM = "detector-mean-v1"
_STOP = object()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _stat_signature(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "missing": True}
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _source_signature(master: str | Path) -> dict[str, Any]:
    """Fingerprint a master and every external HDF5 data file it references."""
    master_path = Path(master).expanduser().resolve()
    paths: set[Path] = {master_path}
    datasets: list[dict[str, Any]] = []
    try:
        import h5py

        with h5py.File(master_path, "r") as handle:
            group = handle.get("entry/data")
            if group is not None:
                for name in sorted(group.keys()):
                    link = group.get(name, getlink=True)
                    if isinstance(link, h5py.ExternalLink):
                        linked = (master_path.parent / link.filename).resolve()
                        paths.add(linked)
                        # File stat identity below detects changed detector
                        # bytes. Do not open every linked chunk merely to hash a
                        # preview: real masters may contain 100+ links, and the
                        # processed output shape/dtype already live in the load
                        # and preview contracts.
                        datasets.append(
                            {
                                "name": str(name),
                                "file": str(linked),
                                "dataset": str(link.path),
                            }
                        )
                    else:
                        try:
                            dataset = group[name]
                            datasets.append(
                                {
                                    "name": str(name),
                                    "file": str(master_path),
                                    "dataset": str(dataset.name),
                                    "shape": [int(value) for value in dataset.shape],
                                    "dtype": str(dataset.dtype),
                                }
                            )
                        except (OSError, KeyError, ValueError):
                            continue
    except (ImportError, OSError, KeyError, ValueError):
        # Self-contained/non-HDF5 test inputs still receive a strong filesystem
        # signature.  A missing master is allowed to raise from stat below.
        pass
    return {
        "files": [_stat_signature(path) for path in sorted(paths)],
        "datasets": datasets,
    }


def _signature_is_complete(signature: dict[str, Any]) -> bool:
    files = signature.get("files", [])
    return bool(files) and all(not item.get("missing", False) for item in files)


def _cache_path(
    folder: Path,
    *,
    cache: bool | str,
    cache_dir: str | Path | None,
) -> tuple[Path | None, str]:
    token = "auto" if cache is True else str(cache).strip().lower()
    if cache is False or token in {"false", "off", "none", "0"}:
        return None, "off"
    folder_key = hashlib.sha256(str(folder).encode()).hexdigest()[:24]
    if cache_dir is not None:
        root = Path(cache_dir).expanduser()
        return root / "show4dstem" / "v1" / folder_key, "explicit"
    if token in {"folder", "local", "project"}:
        return folder / ".quantem" / "show4dstem-cache" / "v1", "folder"
    if token in {"auto", "true", "user"}:
        root = Path(
            os.environ.get(
                "QUANTEM_WIDGET_CACHE",
                Path.home() / ".cache" / "quantem.widget",
            )
        ).expanduser()
        return root / "show4dstem" / "v1" / folder_key, "user"
    raise ValueError(
        "preview_cache must be True, False, 'auto', 'user', or 'folder'"
    )


class Show4DSTEMPreviewCache:
    """Content-addressed, asynchronous cache of standard detector previews."""

    def __init__(
        self,
        folder: str | Path,
        *,
        cache: bool | str = "auto",
        cache_dir: str | Path | None = None,
        max_bytes: int | None = 4 << 30,
        rebuild: bool = False,
        load_contract: dict[str, Any] | None = None,
    ) -> None:
        self.folder = Path(folder).expanduser().resolve()
        self.path, self.mode = _cache_path(
            self.folder,
            cache=cache,
            cache_dir=cache_dir,
        )
        self.max_bytes = None if max_bytes is None else max(0, int(max_bytes))
        self.load_contract = dict(load_contract or {})
        self._lock = threading.RLock()
        # Bound retained float32 panels so a four-preset 82-master warm cannot
        # queue hundreds of MiB when the cache disk is slower than CUDA.
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=16)
        self._write_slots = threading.BoundedSemaphore(16)
        self._writer: threading.Thread | None = None
        self._closed = False
        self._clearing = False
        self._clear_lock = threading.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "corruptions": 0,
            "writes": 0,
            "write_errors": 0,
            "evictions": 0,
            "bytes_read": 0,
            "bytes_written": 0,
            "lookup_ms": 0.0,
            "read_ms": 0.0,
            "write_ms": 0.0,
        }
        if self.path is not None and rebuild:
            shutil.rmtree(self.path, ignore_errors=True)
        if self.enabled and self.path is not None and self.path.is_dir():
            self._prune()

    @property
    def enabled(self) -> bool:
        return self.path is not None and self.max_bytes != 0

    @staticmethod
    def source_signature(master: str | Path) -> dict[str, Any]:
        return _source_signature(master)

    def _entry_identity(
        self,
        master: str | Path,
        preset: str,
        preview_contract: dict[str, Any],
        signature: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any]]:
        master_path = Path(master).expanduser().resolve()
        try:
            relative = master_path.relative_to(self.folder).as_posix()
        except ValueError:
            relative = str(master_path)
        identity = {
            "schema": _CACHE_SCHEMA,
            "algorithm": _REDUCTION_ALGORITHM,
            "folder": str(self.folder),
            "master": relative,
            "preset": str(preset).lower(),
            "source": signature,
            "load": self.load_contract,
            "preview": preview_contract,
        }
        # Canonical JSON also normalizes tuples/NumPy-scalar-compatible values
        # to the exact list/primitive representation read back from NPZ.
        identity = json.loads(_canonical_json(identity))
        content_key = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
        slot_identity = {
            key: value for key, value in identity.items() if key != "source"
        }
        slot_key = hashlib.sha256(_canonical_json(slot_identity).encode()).hexdigest()
        return slot_key, content_key, identity

    def _entry_path(self, key: str) -> Path:
        if self.path is None:
            raise RuntimeError("preview cache is disabled")
        return self.path / f"{key}.npz"

    def load(
        self,
        master: str | Path,
        preset: str,
        preview_contract: dict[str, Any],
        *,
        source_signature: dict[str, Any] | None = None,
    ) -> np.ndarray | None:
        """Return one validated cached preview, or ``None`` on any miss."""
        if not self.enabled:
            return None
        lookup_started = time.perf_counter()
        try:
            signature_before = source_signature or _source_signature(master)
            if not _signature_is_complete(signature_before):
                with self._lock:
                    self._stats["misses"] += 1
                    self._stats["lookup_ms"] += (
                        time.perf_counter() - lookup_started
                    ) * 1000.0
                return None
            slot_key, content_key, identity = self._entry_identity(
                master,
                preset,
                preview_contract,
                signature_before,
            )
            path = self._entry_path(slot_key)
            if not path.is_file():
                with self._lock:
                    self._stats["misses"] += 1
                    self._stats["lookup_ms"] += (
                        time.perf_counter() - lookup_started
                    ) * 1000.0
                return None
            read_started = time.perf_counter()
            with np.load(path, allow_pickle=False) as archive:
                image = np.asarray(archive["image"])
                metadata_raw = archive["metadata"]
                metadata = json.loads(str(metadata_raw.item()))
            read_ms = (time.perf_counter() - read_started) * 1000.0
            signature_after = _source_signature(master)
            if signature_after != signature_before:
                with self._lock:
                    self._stats["invalidations"] += 1
                    self._stats["misses"] += 1
                    self._stats["lookup_ms"] += (
                        time.perf_counter() - lookup_started
                    ) * 1000.0
                    self._stats["read_ms"] += read_ms
                return None
            expected_shape = tuple(int(value) for value in preview_contract["shape"])
            expected_checksum = hashlib.sha256(image.tobytes(order="C")).hexdigest()
            valid = bool(
                metadata.get("key") == content_key
                and metadata.get("identity") == identity
                and metadata.get("checksum") == expected_checksum
                and image.dtype == np.dtype("<f4")
                and tuple(image.shape) == expected_shape
                and image.flags.c_contiguous
            )
            if not valid:
                with self._lock:
                    self._stats["invalidations"] += 1
                    self._stats["misses"] += 1
                    self._stats["lookup_ms"] += (
                        time.perf_counter() - lookup_started
                    ) * 1000.0
                    self._stats["read_ms"] += read_ms
                return None
            try:
                os.utime(path, None)
            except OSError:
                pass
            stored_bytes = int(path.stat().st_size)
            with self._lock:
                self._stats["hits"] += 1
                self._stats["bytes_read"] += stored_bytes
                self._stats["lookup_ms"] += (
                    time.perf_counter() - lookup_started
                ) * 1000.0
                self._stats["read_ms"] += read_ms
            return np.array(image, dtype=np.float32, copy=True, order="C")
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            with self._lock:
                self._stats["corruptions"] += 1
                self._stats["misses"] += 1
                self._stats["lookup_ms"] += (
                    time.perf_counter() - lookup_started
                ) * 1000.0
            return None

    def store(
        self,
        master: str | Path,
        preset: str,
        preview_contract: dict[str, Any],
        image: np.ndarray,
        *,
        source_signature: dict[str, Any] | None = None,
    ) -> bool:
        """Queue one small preview for atomic publication.

        ``source_signature`` should be captured before raw loading. The writer
        validates it before and after its atomic temporary-file write; the
        foreground panel path never waits on another HDF5 signature scan.
        """
        if not self.enabled:
            return False
        with self._lock:
            if self._closed or self._clearing:
                return False
        # Wait for bounded retained-image capacity without owning ``_lock``.
        # The writer updates telemetry under that lock before it can consume
        # its next queue item, so a blocking Queue.put() inside the lock can
        # deadlock permanently when the queue is saturated.
        self._write_slots.acquire()
        enqueued = False
        try:
            expected = source_signature or _source_signature(master)
            if not _signature_is_complete(expected):
                return False
            panel = np.ascontiguousarray(image, dtype="<f4")
            expected_shape = tuple(int(value) for value in preview_contract["shape"])
            if tuple(panel.shape) != expected_shape:
                raise ValueError(
                    f"preview shape {panel.shape} does not match {expected_shape}"
                )
            slot_key, content_key, identity = self._entry_identity(
                master,
                preset,
                preview_contract,
                expected,
            )
            with self._lock:
                if self._closed or self._clearing:
                    return False
                self._ensure_writer()
                self._queue.put_nowait(
                    (
                        Path(master).expanduser().resolve(),
                        slot_key,
                        content_key,
                        identity,
                        expected,
                        panel.copy(),
                    )
                )
                enqueued = True
            return True
        except (OSError, ValueError, TypeError, queue.Full):
            with self._lock:
                self._stats["write_errors"] += 1
            return False
        finally:
            if not enqueued:
                self._write_slots.release()

    def _ensure_writer(self) -> None:
        with self._lock:
            if self._writer is not None and self._writer.is_alive():
                return
            self._writer = threading.Thread(
                target=self._writer_loop,
                name="Show4DSTEM-preview-cache",
                daemon=True,
            )
            self._writer.start()

    def _writer_loop(self) -> None:
        writes_since_prune = 0
        while True:
            job = self._queue.get()
            try:
                if job is _STOP:
                    return
                if self._write_job(*job):
                    writes_since_prune += 1
                if writes_since_prune >= 32 or (
                    writes_since_prune > 0 and self._queue.empty()
                ):
                    self._prune()
                    writes_since_prune = 0
            finally:
                self._queue.task_done()
                if job is not _STOP:
                    self._write_slots.release()

    def _write_job(
        self,
        master: Path,
        slot_key: str,
        content_key: str,
        identity: dict[str, Any],
        expected_signature: dict[str, Any],
        image: np.ndarray,
    ) -> bool:
        write_started = time.perf_counter()
        try:
            if _source_signature(master) != expected_signature:
                with self._lock:
                    self._stats["invalidations"] += 1
                return False
            path = self._entry_path(slot_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "key": content_key,
                "identity": identity,
                "checksum": hashlib.sha256(image.tobytes(order="C")).hexdigest(),
                "created": time.time(),
            }
            fd, temporary_name = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=f".{slot_key}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, "w+b") as stream:
                    np.savez(
                        stream,
                        image=image,
                        metadata=np.asarray(_canonical_json(metadata)),
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                if _source_signature(master) != expected_signature:
                    with self._lock:
                        self._stats["invalidations"] += 1
                    return False
                os.replace(temporary, path)
                stored_bytes = int(path.stat().st_size)
                with self._lock:
                    self._stats["writes"] += 1
                    self._stats["bytes_written"] += stored_bytes
                    self._stats["write_ms"] += (
                        time.perf_counter() - write_started
                    ) * 1000.0
                return True
            finally:
                try:
                    temporary.unlink()
                except OSError:
                    pass
        except (OSError, ValueError, TypeError):
            with self._lock:
                self._stats["write_errors"] += 1
                self._stats["write_ms"] += (
                    time.perf_counter() - write_started
                ) * 1000.0
            return False

    def _prune(self) -> None:
        if self.path is None or self.max_bytes is None:
            return
        try:
            entries = sorted(
                (path for path in self.path.glob("*.npz") if path.is_file()),
                key=lambda path: (path.stat().st_mtime_ns, path.name),
            )
            total = sum(path.stat().st_size for path in entries)
            for path in entries:
                if total <= self.max_bytes:
                    break
                size = path.stat().st_size
                path.unlink()
                total -= size
                with self._lock:
                    self._stats["evictions"] += 1
        except OSError:
            return

    def flush(self) -> None:
        """Wait until all queued small preview writes have completed."""
        # Synchronize with the enqueue critical section. Stores accepted after
        # this point are outside this flush; close() first marks the cache closed
        # so no such store can exist during shutdown.
        with self._lock:
            writer = self._writer
        if writer is not None:
            self._queue.join()

    def clear(self) -> None:
        """Delete this scope's derived previews, never raw data or other caches."""
        # A foreground page may finish reducing while a user clears the cache.
        # Reject those late stores until the drained namespace and its counters
        # have been reset together; otherwise a new NPZ can survive with zeroed
        # accounting.
        with self._clear_lock:
            with self._lock:
                self._clearing = True
            try:
                self.flush()
                if self.path is not None:
                    try:
                        shutil.rmtree(self.path)
                    except FileNotFoundError:
                        pass
                with self._lock:
                    for name in self._stats:
                        self._stats[name] = 0
            finally:
                with self._lock:
                    self._clearing = False

    def close(self) -> None:
        """Flush and stop the cache writer."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            writer = self._writer
        self.flush()
        if writer is not None and writer.is_alive():
            self._queue.put(_STOP)
            self._queue.join()
            writer.join()

    @property
    def info(self) -> dict[str, Any]:
        entries = 0
        current_bytes = 0
        if self.path is not None and self.path.is_dir():
            try:
                paths = [path for path in self.path.glob("*.npz") if path.is_file()]
                entries = len(paths)
                current_bytes = sum(path.stat().st_size for path in paths)
            except OSError:
                pass
        with self._lock:
            stats = dict(self._stats)
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "path": None if self.path is None else str(self.path),
            "max_bytes": self.max_bytes,
            "entries": entries,
            "current_bytes": current_bytes,
            "pending_writes": self._queue.qsize(),
            "closed": self._closed,
            "errors": int(stats["corruptions"]) + int(stats["write_errors"]),
            **stats,
        }


__all__ = ["Show4DSTEMPreviewCache"]
