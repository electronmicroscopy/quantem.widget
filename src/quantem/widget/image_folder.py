"""Shared full-resolution image-folder watching for Show2D and Show3D."""

import re
import threading
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np

from quantem.widget.folder_watch_status import set_folder_watch_status

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


def _normalize_file_types(
    file_types: str | Sequence[str] | None,
) -> frozenset[str] | None:
    """Normalize an optional image-extension filter."""
    if file_types is None:
        return None
    values = [file_types] if isinstance(file_types, str) else list(file_types)
    if not values:
        raise ValueError(
            "file_types must name at least one image type, for example "
            "'tif' or ['png', 'tif']"
        )
    suffixes: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError(
                "file_types entries must be strings such as 'emd', 'png', or 'tif'"
            )
        suffix = value.strip().casefold()
        if not suffix:
            raise ValueError("file_types entries must not be empty")
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        if suffix not in _SUPPORTED_IMAGE_SUFFIXES:
            supported = ", ".join(sorted(_SUPPORTED_IMAGE_SUFFIXES))
            raise ValueError(
                f"Unsupported image file type {value!r}. Supported types: {supported}"
            )
        suffixes.add(suffix)
    return frozenset(suffixes)


def _relative_to_folder(path: Path, root: Path) -> Path:
    """``path`` relative to ``root``, or ``path`` itself when it lies outside ``root``."""
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _total_natural_path_key(
    path: Path,
    root: Path,
) -> tuple[tuple[tuple[tuple[int, object], ...], ...], str]:
    """Return a total, case-insensitive natural-order key with an exact-path tie breaker.

    Digit runs compare as numbers (``frame_2`` before ``frame_10``); the exact
    relative path then orders names that read as the same number
    (``frame_01``, ``frame_1``) the same way on every poll.
    """
    relative = _relative_to_folder(path, root)
    parts: list[tuple[tuple[int, object], ...]] = []
    for component in relative.parts:
        tokens: list[tuple[int, object]] = []
        for token in _NATURAL_PART_RE.split(component.casefold()):
            if not token:
                continue
            tokens.append((0, int(token)) if token.isdigit() else (1, token))
        parts.append(tuple(tokens))
    return tuple(parts), relative.as_posix()


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
    """Size and modification time of ``path``; a file still being written changes it between two reads."""
    stat = path.stat()
    return _FileFingerprint(size=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns))


def _calibration_matches(
    first: ImageFolderRecord,
    other: ImageFolderRecord,
) -> bool:
    """Whether two files share units and sampling, so one scale bar is true for both."""
    if first.units != other.units:
        return False
    if first.sampling is None or other.sampling is None:
        return first.sampling is other.sampling
    return bool(np.allclose(first.sampling, other.sampling, rtol=1e-7, atol=0.0))


def _safe_set_widget_status(widget, name: str, value: object) -> None:
    """Publish optional folder status without requiring a widget trait."""
    try:
        setattr(widget, name, value)
    except (AttributeError, TypeError, ValueError):
        # Status is advisory. Strict/slotted test or downstream widgets may not
        # expose these optional attributes, which must not break data updates.
        pass


def _display_bin_factor(widget) -> int:
    """Return the active display-space bin factor for either image viewer.

    Show2D syncs it as ``_display_bin_factor``, Show3D keeps ``_display_bin``;
    the scale bar is drawn on binned display pixels, so the pixel size scales by it.
    """
    for name in ("_display_bin_factor", "_display_bin"):
        try:
            return max(1, int(getattr(widget, name)))
        except (AttributeError, TypeError, ValueError):
            continue
    return 1


def _clear_pixel_size(widget, *, scale_bar_visible: bool) -> None:
    """Drop the widget's pixel size so no scale bar claims a calibration the files do not give.

    Show2D also carries per-panel ``pixel_sizes``; Show3D has one ``pixel_size``.
    """
    widget.pixel_size = 0.0
    if hasattr(widget, "pixel_sizes"):
        widget.pixel_sizes = []
    widget.scale_bar_visible = scale_bar_visible


class WatchedImageFolder:
    """Append-only folder source that retries unstable files in place."""

    def __init__(
        self,
        folder: str | Path,
        *,
        pattern: str = "*",
        file_types: str | Sequence[str] | None = None,
        recursive: bool = False,
        interval: float = 1.0,
        mode: Literal["panels", "frames"],
        create: bool = False,
    ) -> None:
        self.folder = _canonical_path(Path(folder))
        if create:
            self.folder.mkdir(parents=True, exist_ok=True)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Image folder does not exist or is not a directory: {self.folder}")
        self.pattern = str(pattern)
        if not self.pattern:
            raise ValueError("pattern must be a non-empty glob, for example '*.tif' or '*'")
        self.file_types = _normalize_file_types(file_types)
        self.recursive = bool(recursive)
        self.interval = self._validate_interval(interval)
        self.mode = mode
        self.expected_shape: tuple[int, int] | None = None
        # 1 = grayscale (H, W); 3 = RGB (H, W, 3). Used for Show3D frame stacks.
        self.expected_channels: int | None = None
        self.records: list[ImageFolderRecord] = []
        self.errors: dict[Path, str] = {}
        self._ready_probation: dict[Path, _FileFingerprint] = {}
        self.calibration_status = ""
        self._explicit_calibration = False
        self._scale_bar_requested = True
        self._poll_lock = threading.Lock()
        self._watch_stop: threading.Event | None = None
        self._watch_thread: threading.Thread | None = None
        self._widget_ref: weakref.ReferenceType | None = None
        self._watch_enabled = False
        self._watch_started = False

    @staticmethod
    def _validate_interval(interval: float) -> float:
        """The polling interval in seconds; a zero, negative or non-finite one would spin or never poll."""
        value = float(interval)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"watch interval must be a finite value > 0 seconds, got {interval!r}")
        return value

    def discover(self) -> list[Path]:
        """Return supported folder-visible paths in total natural order."""
        candidates = self.folder.rglob(self.pattern) if self.recursive else self.folder.glob(self.pattern)
        unique: dict[Path, None] = {}
        for candidate in candidates:
            if not candidate.is_file() or candidate.name.startswith("."):
                continue
            suffix = candidate.suffix.casefold()
            if suffix not in _SUPPORTED_IMAGE_SUFFIXES:
                continue
            if self.file_types is not None and suffix not in self.file_types:
                continue
            # Keep the path as named in the folder: resolving through a final
            # symlink can land on an extension-less target (Hugging Face hub
            # cache blobs), which the format readers cannot dispatch on.
            unique.setdefault(candidate, None)
        return sorted(unique, key=lambda path: _total_natural_path_key(path, self.folder))

    def _read_stable(self, path: Path) -> _ReadImage | None:
        """Read one unchanged file through the canonical image reader."""
        try:
            before = _fingerprint(path)
            from quantem.widget import io as widget_io  # noqa: PLC0415

            dataset = widget_io.read_image(path)
            # a Dataset2d, its stand-in or an RgbImage: each holds the pixels in .array
            array = np.asarray(dataset.array)
            after = _fingerprint(path)
        except (OSError, ValueError, RuntimeError, KeyError, IndexError, EOFError) as exc:
            self.errors[path] = f"{type(exc).__name__}: {exc}"
            return None
        if before != after:
            self.errors[path] = "file changed while it was being read; retrying on the next poll"
            return None
        # Accept grayscale (H, W) and true-color RGB(A) (H, W, 3/4).
        if array.ndim == 3 and array.shape[-1] in (3, 4):
            array = array[..., :3]
        elif array.ndim != 2:
            self.errors[path] = (
                f"expected a 2D gray or RGB image, got shape "
                f"{tuple(int(size) for size in array.shape)}"
            )
            return None
        actual_shape = (int(array.shape[0]), int(array.shape[1]))
        if self.expected_shape is not None and actual_shape != self.expected_shape:
            message = (
                f"Incompatible image shape for {self.label(path)}: "
                f"expected {self.expected_shape}, "
                f"got {actual_shape}. Show2D.from_folder and Show3D.from_folder "
                "keep every file at full resolution and do not resize mismatched images."
            )
            self.errors[path] = message
            return None
        # Color vs gray mix is allowed for Show2D panels, but Show3D stacks must
        # share channel layout. Store spatial shape only in expected_shape;
        # channel mismatch is checked when stacking frames.
        if self.expected_channels is None:
            self.expected_channels = int(array.shape[2]) if array.ndim == 3 else 1
        else:
            channels = int(array.shape[2]) if array.ndim == 3 else 1
            if channels != int(self.expected_channels) and self.mode == "frames":
                self.errors[path] = (
                    f"Incompatible channel layout for {self.label(path)}: "
                    f"expected {self.expected_channels}-channel frames, got {channels}. "
                    "Show3D.from_folder cannot mix gray and RGB frames in one stack."
                )
                return None
        self.errors.pop(path, None)
        sampling = (float(dataset.sampling[-2]), float(dataset.sampling[-1]))
        units = (str(dataset.units[-2]), str(dataset.units[-1]))
        return _ReadImage(ImageFolderRecord(path, after, sampling, units), array)

    def read_initial(
        self,
        *,
        allow_empty: bool = False,
        require_unchanged_followup: bool = False,
    ) -> tuple[list[np.ndarray], list[ImageFolderRecord]]:
        """Read the initial stable image set, leaving failed files retryable."""
        arrays: list[np.ndarray] = []
        records: list[ImageFolderRecord] = []
        for path in self.discover():
            read = self._read_stable(path)
            if read is None:
                continue
            if require_unchanged_followup:
                # A readable file can still be paused mid-write. A live mount
                # starts immediately, but its initial candidates use the same
                # two-poll fingerprint probation as later arrivals.
                self._ready_probation[path] = read.record.fingerprint
                continue
            if self.expected_shape is None:
                self.expected_shape = (int(read.array.shape[0]), int(read.array.shape[1]))
            if self.expected_channels is None:
                self.expected_channels = (
                    int(read.array.shape[2]) if read.array.ndim == 3 else 1
                )
            records.append(read.record)
            arrays.append(read.array)
        if not arrays:
            self.records = []
            if allow_empty or require_unchanged_followup:
                return [], []
            detail = ""
            if self.errors:
                path, error = next(iter(self.errors.items()))
                detail = f" First unreadable candidate: {path} ({error})."
            raise FileNotFoundError(
                f"No readable 2D gray/RGB images matching {self.pattern!r} in {self.folder}."
                f"{detail} Partially written files are retried by a running watcher "
                "after at least one valid image is available."
            )
        self.records = records
        return arrays, records

    def attach(
        self,
        widget,
        *,
        explicit_calibration: bool,
    ) -> Self:
        """Attach this source to an already constructed widget."""
        self._widget_ref = weakref.ref(widget)
        self._explicit_calibration = bool(explicit_calibration)
        self._scale_bar_requested = bool(getattr(widget, "scale_bar_visible", True))
        widget._folder_source = self
        self._apply_calibration(widget, self.records)
        self._sync_widget_status(widget)
        return self

    def label(self, path: Path) -> str:
        """Return a concise path-derived label that remains unique recursively."""
        return _relative_to_folder(path, self.folder).with_suffix("").as_posix()

    @property
    def paths(self) -> list[Path]:
        """Paths of the files the widget shows, in display order."""
        return [record.path for record in self.records]

    def poll(self, widget) -> list[int]:
        """Append stable new files and return their zero-based widget indices.

        A concurrent manual/background poll returns immediately. New decoded
        paths remain probationary until a later poll sees the same file
        fingerprint; no polling thread sleeps while holding this lock.
        """
        if not self._poll_lock.acquire(blocking=False):
            return []
        try:
            return self._poll_once(widget)
        finally:
            self._poll_lock.release()

    def _poll_once(self, widget) -> list[int]:
        """Run one caller-owned folder scan while the poll lock is held."""
        if self._watch_enabled:
            set_folder_watch_status(
                widget,
                "updating",
                f"Scanning {self.folder.name or 'watched folder'}.",
            )
        old_records = list(self.records)
        old_by_path = {record.path: record for record in old_records}
        changed: dict[Path, _ReadImage] = {}
        discovered = self.discover()
        discovered_set = set(discovered)
        self.errors = {
            path: message
            for path, message in self.errors.items()
            if path in discovered_set
        }
        self._ready_probation = {
            path: fingerprint
            for path, fingerprint in self._ready_probation.items()
            if path in discovered_set and path not in old_by_path
        }
        if not old_records and not self._ready_probation:
            # A provisional first candidate may disappear before its confirming
            # poll. Let the next readable candidate establish the real shape.
            self.expected_shape = None

        for path in discovered:
            previous = old_by_path.get(path)
            if previous is not None:
                # Folder-backed viewers are append-only. Rewriting a source
                # path must not silently replace scientific data that the user
                # may already have curated, starred, or measured.
                self._ready_probation.pop(path, None)
                continue
            read = self._read_stable(path)
            if read is None:
                self._ready_probation.pop(path, None)
                continue
            if self.expected_shape is None:
                self.expected_shape = (int(read.array.shape[0]), int(read.array.shape[1]))

            fingerprint = read.record.fingerprint
            if self._ready_probation.get(path) != fingerprint:
                # The decoded file is readable, but acquisition software may
                # still append or rewrite it. A later caller-owned poll must
                # decode the same fingerprint again before it becomes visible.
                self._ready_probation[path] = fingerprint
                continue
            self._ready_probation.pop(path, None)
            changed[path] = read

        if not changed:
            self._sync_widget_status(widget)
            return []

        merged = dict(old_by_path)
        merged.update({path: read.record for path, read in changed.items()})
        new_records = sorted(
            merged.values(),
            key=lambda record: _total_natural_path_key(record.path, self.folder),
        )
        if self._watch_enabled:
            set_folder_watch_status(
                widget,
                "updating",
                f"Applying {len(changed)} stable image file"
                f"{'s' if len(changed) != 1 else ''}.",
            )
        # Publish the first-frame transition before replacing the resident
        # array. ``set_image`` exposes its new frame count partway through the
        # update, so a concurrent notebook/UI reader must never observe
        # ``n_slices > 0`` while ``folder_waiting`` is still true.
        _safe_set_widget_status(widget, "_folder_waiting", False)
        _safe_set_widget_status(widget, "folder_waiting", False)
        widget._apply_folder_image_records(
            old_records,
            new_records,
            {path: read.array for path, read in changed.items()},
        )
        self.records = new_records
        self._apply_calibration(widget, new_records)
        self._sync_widget_status(widget)
        return [
            index
            for index, record in enumerate(new_records)
            if record.path in changed
        ]

    def _apply_calibration(self, widget, records: list[ImageFolderRecord]) -> None:
        """Set the widget's scale bar from the files' shared calibration, or turn it off when they disagree.

        A scale bar is only drawn when every shown file has the same sampling
        and units; one mismatched file would make the bar wrong for some panels.
        """
        if self._explicit_calibration:
            self.calibration_status = "explicit sampling/units override"
        elif not records:
            _clear_pixel_size(widget, scale_bar_visible=False)
            self.calibration_status = "waiting for the first readable image"
        elif not all(_calibration_matches(records[0], record) for record in records[1:]):
            _clear_pixel_size(widget, scale_bar_visible=False)
            self.calibration_status = (
                "Scale bar disabled because watched files have different sampling or units."
            )
        elif records[0].sampling is not None and records[0].units is not None:
            first = records[0]
            # The scale bar is horizontal, so it uses the final (column) axis.
            display_bin = _display_bin_factor(widget)
            widget.pixel_size = float(first.sampling[-1]) * display_bin
            widget.pixel_unit = str(first.units[-1])
            if hasattr(widget, "pixel_sizes"):
                widget.pixel_sizes = [
                    float(record.sampling[-1]) * display_bin
                    for record in records
                ]
            widget.scale_bar_visible = self._scale_bar_requested
            self.calibration_status = (
                f"uniform: {first.sampling[-1]:g} {first.units[-1]}/pixel"
            )
        else:
            _clear_pixel_size(widget, scale_bar_visible=self._scale_bar_requested)
            self.calibration_status = "files do not provide spatial calibration"
        _safe_set_widget_status(
            widget,
            "_folder_calibration_status",
            self.calibration_status,
        )

    def _sync_widget_status(
        self,
        widget,
        *,
        watch_error: str | None = None,
        worker_failed: bool = False,
    ) -> None:
        """Publish advisory source state without requiring synced traits."""
        waiting = not self.records
        unexpected_error = bool(watch_error)
        if watch_error is None:
            pending_paths = set(self.errors) | set(self._ready_probation)
            if pending_paths:
                path = min(
                    pending_paths,
                    key=lambda item: _total_natural_path_key(item, self.folder),
                )
                count = len(pending_paths)
                prefix = f"{count} pending file{'s' if count != 1 else ''}"
                issue = self.errors.get(
                    path,
                    "decoded successfully; waiting for one unchanged follow-up poll",
                )
                watch_error = (
                    f"{prefix}; first: {self.label(path)} ({issue})"
                )
            else:
                watch_error = ""
        watch_error = self._compact_watch_detail(watch_error)
        ready_count = len(self.records)
        item_label = "panel" if self.mode == "panels" else "frame"
        if waiting:
            folder_status = (
                f"Waiting for the first stable {item_label} matching "
                f"{self.pattern!r} in {self.folder.name or self.folder}."
            )
            if watch_error:
                folder_status = f"{folder_status} {watch_error}"
        else:
            folder_status = (
                f"{ready_count} ready {item_label}"
                f"{'s' if ready_count != 1 else ''}"
            )
            if watch_error:
                folder_status = f"{folder_status}; {watch_error}"
        _safe_set_widget_status(widget, "_folder_waiting", waiting)
        _safe_set_widget_status(widget, "folder_waiting", waiting)
        _safe_set_widget_status(widget, "folder_status", folder_status)
        _safe_set_widget_status(widget, "_folder_watch_error", watch_error)
        _safe_set_widget_status(
            widget,
            "_folder_calibration_status",
            self.calibration_status,
        )
        if not self._watch_enabled:
            state = "stopped" if self._watch_started else "hidden"
            detail = "Folder watcher stopped" if self._watch_started else ""
        elif worker_failed:
            state = "error"
            detail = (
                f"{watch_error}. The watch worker stopped unexpectedly; "
                "call watch_folder() to restart it."
            )
        elif unexpected_error:
            state = "error"
            detail = (
                f"{watch_error}. The watcher is still alive and will retry; "
                "call stop_folder_watch() if the error persists."
            )
        elif self.errors or self._ready_probation:
            error_text = " ".join(self.errors.values()).casefold()
            incompatible = "incompatible image shape" in error_text
            if incompatible:
                state = "error"
                detail = watch_error
            else:
                state = "waiting"
                detail = watch_error or "Waiting for file completion"
        else:
            thread = self._watch_thread
            if thread is not None and thread.is_alive():
                state = "watching"
                detail = ""
            else:
                state = "error"
                detail = (
                    "Watch worker is not running. Call watch_folder() to restart it."
                )
        set_folder_watch_status(widget, state, detail)

    def _compact_watch_detail(self, detail: str, *, limit: int = 480) -> str:
        """Bound synced detail and remove host-specific absolute paths."""
        text = " ".join(str(detail).split())
        root = str(self.folder)
        variants = {root, str(self.folder.resolve(strict=False))}
        variants.update(
            value[len("/private") :]
            for value in tuple(variants)
            if value.startswith("/private/")
        )
        for value in sorted(variants, key=len, reverse=True):
            text = text.replace(f"{value}/", "")
            text = text.replace(value, self.folder.name or "watched folder")
        compact: list[str] = []
        for token in text.split(" "):
            stripped = token.strip("()[]{}<>,.;:")
            if stripped.startswith("/") and "/" in stripped[1:]:
                token = token.replace(
                    stripped,
                    Path(stripped).name or "source file",
                )
            compact.append(token)
        text = " ".join(compact)
        if len(text) > int(limit):
            text = f"{text[: max(0, int(limit) - 1)].rstrip()}…"
        return text

    def start(self, widget, *, interval: float | None = None) -> Self:
        """Start an idempotent daemon watcher for this source."""
        next_interval = (
            self.interval
            if interval is None
            else self._validate_interval(interval)
        )
        self.stop()
        self.interval = next_interval
        stop = threading.Event()
        self._watch_stop = stop
        self._watch_enabled = True
        self._watch_started = True
        widget_ref = weakref.ref(widget)

        def worker() -> None:
            """Poll every ``interval`` until stopped or the widget is gone; a fatal error turns the badge red."""
            fatal_error: str | None = None
            try:
                while not stop.wait(self.interval):
                    current_widget = widget_ref()
                    if current_widget is None:
                        break
                    try:
                        self.poll(current_widget)
                    except (OSError, ValueError, RuntimeError, KeyError, IndexError, TypeError, MemoryError) as exc:
                        # Unexpected per-poll failures must not stop a microscope
                        # watcher; ordinary unreadable/mismatched files are
                        # reported non-fatally in the folder status.
                        self._sync_widget_status(
                            current_widget,
                            watch_error=f"{type(exc).__name__}: {exc}",
                        )
            except BaseException as exc:
                fatal_error = f"{type(exc).__name__}: {exc}"
            finally:
                current = threading.current_thread()
                if self._watch_stop is stop:
                    self._watch_stop = None
                if self._watch_thread is current:
                    self._watch_thread = None
                if fatal_error is not None and not stop.is_set():
                    current_widget = widget_ref()
                    if current_widget is not None:
                        self._sync_widget_status(
                            current_widget,
                            watch_error=fatal_error,
                            worker_failed=True,
                        )

        thread = threading.Thread(
            target=worker,
            name=f"{type(widget).__name__}-image-folder-watch",
            daemon=True,
        )
        self._watch_thread = thread
        try:
            thread.start()
        except RuntimeError:
            self._watch_enabled = False
            if self._watch_stop is stop:
                self._watch_stop = None
            if self._watch_thread is thread:
                self._watch_thread = None
            set_folder_watch_status(
                widget,
                "error",
                "Watch worker could not start. Check the notebook kernel and retry.",
            )
            raise
        self._sync_widget_status(widget)
        return self

    def stop(self) -> None:
        """Signal and join the watcher thread; safe to call repeatedly."""
        stop = self._watch_stop
        thread = self._watch_thread
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self._watch_enabled = False
        if self._watch_stop is stop:
            self._watch_stop = None
        if self._watch_thread is thread:
            self._watch_thread = None
        widget = self._widget_ref() if self._widget_ref is not None else None
        if widget is not None:
            self._sync_widget_status(widget)


class WatchedImageFolderMixin:
    """Public lifecycle shared by folder-backed Show2D and Show3D widgets."""

    _folder_source: WatchedImageFolder | None = None  # set by WatchedImageFolder.attach in from_folder

    def _require_folder_source(self) -> WatchedImageFolder:
        """The folder source of a ``from_folder`` widget; other widgets have nothing to poll."""
        source = self._folder_source
        if source is None:
            raise RuntimeError(
                f"{type(self).__name__}.poll_folder() is available only on widgets "
                f"created by {type(self).__name__}.from_folder(...)."
            )
        return source

    @property
    def folder_paths(self) -> list[Path]:
        """Canonical paths currently represented by this widget."""
        return list(self._require_folder_source().paths)

    def poll_folder(self) -> list[int]:
        """Append stable new files and return their zero-based indices."""
        return self._require_folder_source().poll(self)

    def watch_folder(self, *, interval: float | None = None) -> Self:
        """Start or restart background folder watching."""
        self._require_folder_source().start(self, interval=interval)
        return self

    def stop_folder_watch(self) -> None:
        """Stop and join background folder watching, if active."""
        if self._folder_source is not None:
            self._folder_source.stop()

    def close(self) -> None:
        """Stop folder work before closing the widget communication channel."""
        self.stop_folder_watch()
        super().close()
