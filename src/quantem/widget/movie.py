"""Compatibility exports for STEM movie helpers.

The implementation now lives in :mod:`quantem.gpu.movie` so accelerated export
backends are owned by the backend package.  This module preserves the existing
``quantem.widget.movie`` public import path for one migration window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quantem.gpu import movie as _gpu_movie
from quantem.gpu.movie import MovieData


def _is_pil_frame_sequence(data: Any) -> bool:
    if not isinstance(data, (list, tuple)) or not data:
        return False
    try:
        from PIL import Image
    except ImportError:
        return False
    return all(isinstance(frame, Image.Image) for frame in data)


def save_gif(data: MovieData, path: str | Path, *, fps: float = 12.0, **kwargs: Any) -> Path:
    if _is_pil_frame_sequence(data) and not kwargs:
        from quantem.widget.render import gif as gif_utils

        return gif_utils.write_gif(list(data), path, fps=fps)
    return _gpu_movie.save_gif(data, path, fps=fps, **kwargs)


def save_mp4(
    data: MovieData,
    path: str | Path,
    *,
    fps: float = 12.0,
    crf: int = 18,
    **kwargs: Any,
) -> Path:
    if _is_pil_frame_sequence(data) and not kwargs:
        gpu_writer = getattr(_gpu_movie, "_write_mp4", None)
        gpu_writer_module = str(getattr(gpu_writer, "__module__", ""))
        if callable(gpu_writer) and not gpu_writer_module.startswith("quantem.gpu."):
            return gpu_writer(list(data), path, fps=fps, crf=crf)
        from quantem.widget.render import gif as gif_utils

        return gif_utils.write_mp4(list(data), path, fps=fps, crf=crf)
    return _gpu_movie.save_mp4(data, path, fps=fps, crf=crf, **kwargs)


def save_movie(
    data: MovieData,
    path: str | Path,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> Path:
    suffix = (format or Path(path).suffix.lstrip(".")).lower()
    if suffix == "gif":
        return save_gif(data, path, **kwargs)
    if suffix == "mp4":
        return save_mp4(data, path, **kwargs)
    return _gpu_movie.save_movie(data, path, format=format, **kwargs)

__all__ = ["MovieData", "save_gif", "save_movie", "save_mp4"]
