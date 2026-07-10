"""Shared full-resolution image-folder watching for Show2D and Show3D."""

from __future__ import annotations

import re
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np


_SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".dm3",
    ".dm4",
    ".emd",
    ".gif",
    ".jpeg",
    ".jpg",
    ".npy",
    ".png",
    ".tif",
    ".tiff",
}
_NATURAL_PART_RE = re.compile(r"(\d+)")


def _natural_path_key(path: Path, root: Path) -> tuple[tuple[tuple[int, object], ...], ...]:
    """Return a deterministic, case-insensitive natural key for a path."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    parts: list[tuple[tuple[int, object], ...]] = []
    for component in relative.parts:
        tokens: list[tuple[int, object]] = []
        for token in _NATURAL_PART_RE.split(component.casefold()):
            if not token:
                continue
            tokens.append((0, int(token)) if token.isdigit() else (1, token))
        parts.append(tuple(tokens))
    return tuple(parts)


def _canonical_path(path: Path) -> Path:
    """Resolve a path without requiring the target to continue to exist."""
    return path.expanduser().resolve(strict=False)


@dataclass(frozen=True)
class _FileFingerprint:
    """Cheap stability marker for one watched file."""

    size: int
    mtime_ns: int


@dataclass(frozen=True)
class ImageFolderRecord:
    """Metadata retained for one successfully read image file."""

    path: Path
    fingerprint: _FileFingerprint
    sampling: tuple[float, float] | None
    units: tuple[str, str] | None


@dataclass(frozen=True)
class _ReadImage:
    """A stable, full-resolution read ready to apply to a widget."""

    record: ImageFolderRecord
    array: np.ndarray


def _fingerprint(path: Path) -> _FileFingerprint:
    stat = path.stat()
    return _FileFingerprint(size=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns))


def _spatial_metadata(dataset: Any) -> tuple[tuple[float, float] | None, tuple[str, str] | None]:
    sampling = getattr(dataset, "sampling", None)
    units = getattr(dataset, "units", None)
    resolved_sampling = None
    resolved_units = None
    if sampling is not None and len(sampling) >= 2:
        resolved_sampling = (float(sampling[-2]), float(sampling[-1]))
    if units is not None and len(units) >= 2:
        resolved_units = (str(units[-2]), str(units[-1]))
    return resolved_sampling, resolved_units


def _calibration_matches(
    first: ImageFolderRecord,
    other: ImageFolderRecord,
) -> bool:
    if first.units != other.units:
        return False
    if first.sampling is None or other.sampling is None:
        return first.sampling is other.sampling
    return bool(np.allclose(first.sampling, other.sampling, rtol=1e-7, atol=0.0))


class WatchedImageFolder:
    """Append-only folder source that retries unstable files in place."""

    def __init__(
        self,
        folder: str | Path,
        *,
        pattern: str = "*",
        recursive: bool = False,
        interval: float = 1.0,
        mode: Literal["panels", "frames"],
    ) -> None:
        self.folder = _canonical_path(Path(folder))
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Image folder does not exist or is not a directory: {self.folder}")
        self.pattern = str(pattern)
        if not self.pattern:
            raise ValueError("pattern must be a non-empty glob, for example '*.tif' or '*'")
        self.recursive = bool(recursive)
        self.interval = self._validate_interval(interval)
        self.mode = mode
        self.expected_shape: tuple[int, int] | None = None
        self.records: list[ImageFolderRecord] = []
        self.errors: dict[Path, str] = {}
        self.calibration_status = ""
        self._explicit_calibration = False
        self._scale_bar_requested = True
        self._poll_lock = threading.Lock()
        self._watch_stop: threading.Event | None = None
        self._watch_thread: threading.Thread | None = None
        self._widget_ref: weakref.ReferenceType[Any] | None = None

    @staticmethod
    def _validate_interval(interval: float) -> float:
        value = float(interval)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"watch interval must be a finite value > 0 seconds, got {interval!r}")
        return value

    def discover(self) -> list[Path]:
        """Return supported files in canonical natural path order."""
        candidates = self.folder.rglob(self.pattern) if self.recursive else self.folder.glob(self.pattern)
        unique: dict[Path, None] = {}
        for candidate in candidates:
            if not candidate.is_file() or candidate.name.startswith("."):
                continue
            if candidate.suffix.casefold() not in _SUPPORTED_IMAGE_SUFFIXES:
                continue
            unique.setdefault(_canonical_path(candidate), None)
        return sorted(unique, key=lambda path: _natural_path_key(path, self.folder))

    def _read_stable(self, path: Path) -> _ReadImage | None:
        """Read one unchanged file through the canonical image reader."""
        try:
            before = _fingerprint(path)
            from quantem.widget import io as widget_io  # noqa: PLC0415

            dataset = widget_io.read_image(path)
            array = np.asarray(dataset.array)
            after = _fingerprint(path)
        except Exception as exc:
            self.errors[path] = f"{type(exc).__name__}: {exc}"
            return None
        if before != after:
            self.errors[path] = "file changed while it was being read; retrying on the next poll"
            return None
        if array.ndim != 2:
            self.errors[path] = f"expected a 2D image, got shape {tuple(int(v) for v in array.shape)}"
            return None
        actual_shape = (int(array.shape[0]), int(array.shape[1]))
        if self.expected_shape is not None and actual_shape != self.expected_shape:
            message = (
                f"Incompatible image shape for {path}: expected {self.expected_shape}, "
                f"got {actual_shape}. Show2D.from_folder and Show3D.from_folder "
                "keep every file at full resolution and do not resize mismatched images."
            )
            self.errors[path] = message
            raise ValueError(message)
        sampling, units = _spatial_metadata(dataset)
        self.errors.pop(path, None)
        return _ReadImage(
            ImageFolderRecord(path, after, sampling, units),
            array,
        )

    def read_initial(self) -> tuple[list[np.ndarray], list[ImageFolderRecord]]:
        """Read the initial stable image set, leaving failed files retryable."""
        arrays: list[np.ndarray] = []
        records: list[ImageFolderRecord] = []
        for path in self.discover():
            read = self._read_stable(path)
            if read is None:
                continue
            if self.expected_shape is None:
                self.expected_shape = tuple(int(v) for v in read.array.shape)
            records.append(read.record)
            arrays.append(read.array)
        if not arrays:
            detail = ""
            if self.errors:
                path, error = next(iter(self.errors.items()))
                detail = f" First unreadable candidate: {path} ({error})."
            raise FileNotFoundError(
                f"No readable 2D images matching {self.pattern!r} in {self.folder}."
                f"{detail} Partially written files are retried by a running watcher "
                "after at least one valid image is available."
            )
        self.records = records
        return arrays, records

    def attach(
        self,
        widget: Any,
        *,
        explicit_calibration: bool,
    ) -> Self:
        """Attach this source to an already constructed widget."""
        self._widget_ref = weakref.ref(widget)
        self._explicit_calibration = bool(explicit_calibration)
        self._scale_bar_requested = bool(getattr(widget, "scale_bar_visible", True))
        widget._folder_source = self
        self._apply_calibration(widget, self.records)
        return self

    def label(self, path: Path) -> str:
        """Return a concise path-derived label that remains unique recursively."""
        try:
            relative = path.relative_to(self.folder)
        except ValueError:
            relative = path
        return relative.with_suffix("").as_posix()

    @property
    def paths(self) -> list[Path]:
        return [record.path for record in self.records]

    def poll(self, widget: Any) -> list[int]:
        """Append stable new files and return their zero-based widget indices."""
        with self._poll_lock:
            old_records = list(self.records)
            old_by_path = {record.path: record for record in old_records}
            changed: dict[Path, _ReadImage] = {}
            for path in self.discover():
                previous = old_by_path.get(path)
                if previous is not None:
                    # Folder-backed viewers are append-only. Rewriting a source
                    # path must not silently replace scientific data that the
                    # user may already have curated, starred, or measured.
                    continue
                try:
                    _fingerprint(path)
                except OSError as exc:
                    self.errors[path] = f"{type(exc).__name__}: {exc}"
                    continue
                read = self._read_stable(path)
                if read is not None:
                    changed[path] = read
            if not changed:
                return []

            merged = dict(old_by_path)
            merged.update({path: read.record for path, read in changed.items()})
            new_records = sorted(
                merged.values(),
                key=lambda record: _natural_path_key(record.path, self.folder),
            )
            widget._apply_folder_image_records(
                old_records,
                new_records,
                {path: read.array for path, read in changed.items()},
            )
            self.records = new_records
            self._apply_calibration(widget, new_records)
            return [
                index
                for index, record in enumerate(new_records)
                if record.path in changed
            ]

    def _apply_calibration(self, widget: Any, records: list[ImageFolderRecord]) -> None:
        if self._explicit_calibration:
            self.calibration_status = "explicit sampling/units override"
            widget._folder_calibration_status = self.calibration_status
            return
        first = records[0]
        uniform = all(_calibration_matches(first, record) for record in records[1:])
        if not uniform:
            widget.pixel_size = 0.0
            if hasattr(widget, "pixel_sizes"):
                widget.pixel_sizes = []
            widget.scale_bar_visible = False
            self.calibration_status = (
                "Scale bar disabled because watched files have different sampling or units."
            )
            widget._folder_calibration_status = self.calibration_status
            return
        if first.sampling is not None and first.units is not None:
            # The scale bar is horizontal, so it uses the final (column) axis.
            widget.pixel_size = float(first.sampling[-1])
            widget.pixel_unit = str(first.units[-1])
            if hasattr(widget, "pixel_sizes"):
                widget.pixel_sizes = [float(record.sampling[-1]) for record in records]
            widget.scale_bar_visible = self._scale_bar_requested
            self.calibration_status = (
                f"uniform: {first.sampling[-1]:g} {first.units[-1]}/pixel"
            )
        else:
            widget.pixel_size = 0.0
            if hasattr(widget, "pixel_sizes"):
                widget.pixel_sizes = []
            self.calibration_status = "files do not provide spatial calibration"
        widget._folder_calibration_status = self.calibration_status

    def start(self, widget: Any, *, interval: float | None = None) -> Self:
        """Start an idempotent daemon watcher for this source."""
        self.stop()
        if interval is not None:
            self.interval = self._validate_interval(interval)
        stop = threading.Event()
        self._watch_stop = stop
        widget_ref = weakref.ref(widget)

        def worker() -> None:
            while not stop.wait(self.interval):
                current_widget = widget_ref()
                if current_widget is None:
                    break
                try:
                    self.poll(current_widget)
                except Exception as exc:
                    # Direct poll_folder() raises actionable errors. A background
                    # microscope watcher stays alive and retries the same file.
                    current_widget._folder_watch_error = f"{type(exc).__name__}: {exc}"
                else:
                    current_widget._folder_watch_error = ""

        thread = threading.Thread(
            target=worker,
            name=f"{type(widget).__name__}-image-folder-watch",
            daemon=True,
        )
        self._watch_thread = thread
        thread.start()
        return self

    def stop(self) -> None:
        """Signal and join the watcher thread; safe to call repeatedly."""
        stop = self._watch_stop
        thread = self._watch_thread
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        if self._watch_stop is stop:
            self._watch_stop = None
        if self._watch_thread is thread:
            self._watch_thread = None


class WatchedImageFolderMixin:
    """Public lifecycle shared by folder-backed Show2D and Show3D widgets."""

    def _require_folder_source(self) -> WatchedImageFolder:
        source = getattr(self, "_folder_source", None)
        if not isinstance(source, WatchedImageFolder):
            raise RuntimeError(
                f"{type(self).__name__}.poll_folder() is available only on widgets "
                f"created by {type(self).__name__}.from_folder(...)."
            )
        return source

    @property
    def folder_paths(self) -> list[Path]:
        """Canonical paths currently represented by this widget."""
        return list(self._require_folder_source().paths)

    @property
    def folder_errors(self) -> dict[Path, str]:
        """Files waiting for a successful later poll and their latest errors."""
        return dict(self._require_folder_source().errors)

    def poll_folder(self) -> list[int]:
        """Append stable new files and return their zero-based indices."""
        return self._require_folder_source().poll(self)

    def watch_folder(self, *, interval: float | None = None) -> Self:
        """Start or restart background folder watching."""
        self._require_folder_source().start(self, interval=interval)
        return self

    def stop_folder_watch(self) -> None:
        """Stop and join background folder watching, if active."""
        source = getattr(self, "_folder_source", None)
        if isinstance(source, WatchedImageFolder):
            source.stop()

    def close(self) -> None:
        """Stop folder work before closing the widget communication channel."""
        self.stop_folder_watch()
        super().close()
