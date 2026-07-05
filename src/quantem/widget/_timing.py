"""Small timing helpers for widget notebooks and signoff reports."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable


def format_bytes(nbytes: int | float | None) -> str:
    """Return a compact binary-size string."""
    if nbytes is None:
        return "unknown"
    value = float(nbytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} B"
    if abs(value) >= 100:
        return f"{value:.0f} {unit}"
    if abs(value) >= 10:
        return f"{value:.1f} {unit}"
    return f"{value:.2f} {unit}"


def format_ms(ms: int | float | None) -> str:
    """Return a human-readable duration from milliseconds."""
    if ms is None:
        return "pending"
    value = float(ms)
    if value < 10:
        return f"{value:.1f} ms"
    if value < 1000:
        return f"{value:.0f} ms"
    return f"{value / 1000.0:.2f} s"


def _shape_of(data: Any) -> tuple[int, ...] | None:
    shape = getattr(data, "shape", None)
    if shape is None and hasattr(data, "array"):
        shape = getattr(data.array, "shape", None)
    if shape is None:
        return None
    return tuple(int(v) for v in shape)


def _dtype_of(data: Any) -> str | None:
    dtype = getattr(data, "dtype", None)
    if dtype is None and hasattr(data, "array"):
        dtype = getattr(data.array, "dtype", None)
    return None if dtype is None else str(dtype)


def _nbytes_of(data: Any) -> int | None:
    nbytes = getattr(data, "nbytes", None)
    if nbytes is None and hasattr(data, "array"):
        nbytes = getattr(data.array, "nbytes", None)
    return None if nbytes is None else int(nbytes)


def format_shape(shape: tuple[int, ...] | list[int] | str | None) -> str:
    """Format an array shape for terminal/notebook output."""
    if shape is None:
        return "unknown"
    if isinstance(shape, str):
        return shape
    return "x".join(str(int(v)) for v in shape)


def format_timing_table(
    title: str,
    rows: list[tuple[str, str | int | float | None]],
) -> str:
    """Format short timing rows as a copyable fixed-width table."""
    clean_rows = [
        (str(label), "" if value is None else str(value))
        for label, value in rows
    ]
    width = max([len(label) for label, _ in clean_rows] + [0])
    lines = [title, "-" * len(title)]
    lines.extend(f"{label:<{width}}  {value}" for label, value in clean_rows)
    return "\n".join(lines)


def format_widget_render_timing(
    widget_name: str,
    *,
    shape: tuple[int, ...] | list[int] | str | None = None,
    dtype: str | None = None,
    raw_bytes: int | None = None,
    total_ms: int | float | None = None,
    python_ms: int | float | None = None,
    wire_js_ms: int | float | None = None,
    extra_rows: list[tuple[str, str | int | float | None]] | None = None,
) -> str:
    """Format first-render widget timing as a stable notebook report."""
    data_parts = [format_shape(shape)]
    if dtype:
        data_parts.append(dtype)
    if raw_bytes is not None:
        data_parts.append(format_bytes(raw_bytes))
    rows: list[tuple[str, str | int | float | None]] = [
        ("Data", " | ".join(data_parts)),
        ("Render total", format_ms(total_ms)),
        ("Python build", format_ms(python_ms)),
        ("Wire + JS paint", format_ms(wire_js_ms)),
    ]
    if extra_rows:
        rows.extend(extra_rows)
    return format_timing_table(f"{widget_name} timing", rows)


@dataclass(slots=True)
class WidgetProfile:
    """Notebook-side timing record for widget load/build workflows."""

    label: str
    build_ms: float
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    raw_bytes: int | None = None
    load_ms: float | None = None
    pack_ms: float | None = None
    backend: str | None = None
    notes: list[str] = field(default_factory=list)

    def format(self) -> str:
        """Return a copyable timing table."""
        rows: list[tuple[str, str | int | float | None]] = []
        if self.backend:
            rows.append(("Backend", self.backend))
        data_parts = []
        if self.shape is not None:
            data_parts.append(format_shape(self.shape))
        if self.dtype:
            data_parts.append(self.dtype)
        if self.raw_bytes is not None:
            data_parts.append(format_bytes(self.raw_bytes))
        if data_parts:
            rows.append(("Data", " | ".join(data_parts)))
        rows.extend([
            ("Load", format_ms(self.load_ms)),
            ("Pack/prep", format_ms(self.pack_ms)),
            ("Widget build", format_ms(self.build_ms)),
        ])
        for i, note in enumerate(self.notes, start=1):
            rows.append((f"Note {i}", note))
        return format_timing_table(f"{self.label} profile", rows)


def profile_widget(
    label: str,
    factory: Callable[[], Any],
    *,
    data: Any | None = None,
    shape: tuple[int, ...] | list[int] | None = None,
    dtype: str | None = None,
    raw_bytes: int | None = None,
    load_ms: int | float | None = None,
    pack_ms: int | float | None = None,
    backend: str | None = None,
    notes: list[str] | None = None,
    verbose: bool = True,
) -> tuple[Any, WidgetProfile]:
    """Build a widget and print a standard notebook timing table.

    Use this around the final widget construction in profiling notebooks. It
    measures Python-side construction; widgets with first-paint telemetry still
    print their browser render table after the frontend paints.
    """
    if shape is None and data is not None:
        shape = _shape_of(data)
    if dtype is None and data is not None:
        dtype = _dtype_of(data)
    if raw_bytes is None and data is not None:
        raw_bytes = _nbytes_of(data)

    start = time.perf_counter()
    widget = factory()
    build_ms = (time.perf_counter() - start) * 1000.0
    profile = WidgetProfile(
        label=label,
        build_ms=build_ms,
        shape=None if shape is None else tuple(int(v) for v in shape),
        dtype=dtype,
        raw_bytes=raw_bytes,
        load_ms=None if load_ms is None else float(load_ms),
        pack_ms=None if pack_ms is None else float(pack_ms),
        backend=backend,
        notes=[] if notes is None else list(notes),
    )
    if verbose:
        print(profile.format(), flush=True)
    return widget, profile


def widget_timing_report(widget: Any, *, label: str | None = None) -> str:
    """Return the latest first-paint timing table for a widget instance."""
    name = label or type(widget).__name__
    shape = None
    if all(hasattr(widget, attr) for attr in ("n_images", "height", "width")):
        n_images = int(getattr(widget, "n_images"))
        shape = (
            (n_images, int(widget.height), int(widget.width))
            if n_images > 1
            else (int(widget.height), int(widget.width))
        )
    elif all(hasattr(widget, attr) for attr in ("n_slices", "height", "width")):
        shape = (int(widget.n_slices), int(widget.height), int(widget.width))
    raw_bytes = _nbytes_of(getattr(widget, "_data", None))
    return format_widget_render_timing(
        name,
        shape=shape,
        raw_bytes=raw_bytes,
        total_ms=getattr(widget, "render_total_ms", None),
        python_ms=getattr(widget, "render_python_build_ms", None),
        wire_js_ms=getattr(widget, "render_wire_js_ms", None),
    )
