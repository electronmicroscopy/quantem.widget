"""Simple GIF and MP4 export helpers for widget-style movies.

The public API is intentionally small:

``save_gif(data, path, ...)`` and ``save_mp4(data, path, ...)``.

``data`` may be one stack with shape ``(frame, row, col)``, several stacks as
``(movie, frame, row, col)``, a list of stacks, or a list of pre-rendered PIL
frames from a widget method. ``backend="auto"`` uses CUDA MP4 when available
and otherwise falls back to the portable CPU writer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from quantem.widget.render import gif as gif_utils

MovieData = np.ndarray | Sequence[np.ndarray] | Sequence[Image.Image]


def _is_pil_frame_sequence(data: object) -> bool:
    return (
        isinstance(data, Sequence)
        and len(data) > 0
        and all(isinstance(item, Image.Image) for item in data)
    )


def _as_stack_list(data: MovieData) -> list[np.ndarray]:
    """Normalize public movie data to a list of 3D stacks."""

    if isinstance(data, np.ndarray):
        arr = np.asarray(data)
        if arr.ndim == 3:
            return [arr]
        if arr.ndim == 4:
            return [arr[i] for i in range(arr.shape[0])]
        raise ValueError(
            "movie data must have shape (frame, row, col) or "
            f"(movie, frame, row, col), got {arr.shape}"
        )

    stacks = [np.asarray(item) for item in data]
    if not stacks:
        raise ValueError("movie data must contain at least one stack")
    for index, stack in enumerate(stacks):
        if stack.ndim != 3:
            raise ValueError(
                "each movie stack must have shape (frame, row, col), "
                f"item {index} has shape {stack.shape}"
            )
    return stacks


def _validate_stacks(stacks: list[np.ndarray]) -> tuple[int, int, int]:
    frames, height, width = stacks[0].shape
    for index, stack in enumerate(stacks[1:], start=1):
        if stack.shape[0] != frames:
            raise ValueError(
                "all movie stacks must have the same number of frames; "
                f"stack 0 has {frames}, stack {index} has {stack.shape[0]}"
            )
        if stack.shape[1:] != (height, width):
            raise ValueError(
                "all movie stacks must have the same spatial dimensions; "
                f"stack 0 has {(height, width)}, stack {index} has {stack.shape[1:]}"
            )
    return int(frames), int(height), int(width)


def _contrast_limits(
    stacks: list[np.ndarray],
    *,
    percentile: tuple[float, float],
    shared: bool,
    ref_stacks: Sequence[np.ndarray] | None,
) -> list[tuple[float, float]]:
    pmin, pmax = percentile
    if shared:
        refs = (
            [np.asarray(item) for item in ref_stacks]
            if ref_stacks is not None
            else stacks
        )
        values = np.concatenate([np.asarray(stack).ravel() for stack in refs])
        lo, hi = np.percentile(values, [pmin, pmax])
        if hi <= lo:
            hi = lo + 1.0
        return [(float(lo), float(hi))] * len(stacks)

    limits = []
    for stack in stacks:
        lo, hi = np.percentile(np.asarray(stack).ravel(), [pmin, pmax])
        if hi <= lo:
            hi = lo + 1.0
        limits.append((float(lo), float(hi)))
    return limits


def _to_uint8(stack: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = np.clip(
        (stack.astype(np.float64) - lo) / (hi - lo) * 255.0,
        0.0,
        255.0,
    )
    return scaled.astype(np.uint8)


def _font(label_height: int) -> ImageFont.ImageFont:
    size = max(10, int(label_height) - 8)
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _movie_frames(
    data: MovieData,
    *,
    labels: Sequence[str] | None,
    gap: int,
    label_height: int,
    max_width: int | None,
    cols: int | None,
    shared_contrast: bool,
    ref_stacks: Sequence[np.ndarray] | None,
    percentile: tuple[float, float],
) -> list[Image.Image]:
    if _is_pil_frame_sequence(data):
        return [
            frame.convert("RGB")
            for frame in data
            if isinstance(frame, Image.Image)
        ]

    stacks = _as_stack_list(data)
    n_frames, height, width = _validate_stacks(stacks)
    names = (
        [str(item) for item in labels]
        if labels is not None
        else [f"Movie {i + 1}" for i in range(len(stacks))]
    )
    limits = _contrast_limits(
        stacks,
        percentile=percentile,
        shared=shared_contrast,
        ref_stacks=ref_stacks,
    )
    uint8_stacks = [
        _to_uint8(stack, lo, hi)
        for stack, (lo, hi) in zip(stacks, limits)
    ]

    n_movies = len(stacks)
    n_cols = min(n_movies, 3) if cols is None else max(1, int(cols))
    n_rows = int(math.ceil(n_movies / n_cols))
    gap = max(0, int(gap))
    label_height = max(0, int(label_height))

    total_width = n_cols * width + (n_cols - 1) * gap
    cell_height = height + label_height
    total_height = n_rows * cell_height + (n_rows - 1) * gap
    scale = 1.0
    if max_width is not None and total_width > int(max_width):
        scale = int(max_width) / total_width
    out_width = max(1, int(round(total_width * scale)))
    out_height = max(1, int(round(total_height * scale)))
    frame_width = max(1, int(round(width * scale)))
    frame_height = max(1, int(round(height * scale)))
    gap_scaled = max(0, int(round(gap * scale)))
    label_height_scaled = max(0, int(round(label_height * scale)))
    cell_height_scaled = frame_height + label_height_scaled
    font = _font(label_height_scaled or label_height or 18)

    frames: list[Image.Image] = []
    for frame_idx in range(n_frames):
        canvas = Image.new("RGB", (out_width, out_height), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        for movie_idx, stack_u8 in enumerate(uint8_stacks):
            row, col = divmod(movie_idx, n_cols)
            tile = Image.fromarray(stack_u8[frame_idx], mode="L").convert("RGB")
            if scale != 1.0:
                tile = tile.resize((frame_width, frame_height), Image.LANCZOS)
            x = col * (frame_width + gap_scaled)
            y = row * (cell_height_scaled + gap_scaled)
            canvas.paste(tile, (x, y + label_height_scaled))
            if movie_idx < len(names) and label_height_scaled > 0:
                draw.text(
                    (x + 4, y + 2),
                    f"{names[movie_idx]} [{frame_idx + 1}/{n_frames}]",
                    fill=(255, 255, 255),
                    font=font,
                )
        frames.append(canvas)
    return frames


def save_gif(
    data: MovieData,
    path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    fps: float = 10,
    gap: int = 12,
    label_height: int = 28,
    max_width: int | None = 1200,
    cols: int | None = None,
    shared_contrast: bool = True,
    ref_stacks: Sequence[np.ndarray] | None = None,
    percentile: tuple[float, float] = (1.0, 99.0),
) -> Path:
    """Save one or more grayscale movie stacks as an animated GIF."""

    frames = _movie_frames(
        data,
        labels=labels,
        gap=gap,
        label_height=label_height,
        max_width=max_width,
        cols=cols,
        shared_contrast=shared_contrast,
        ref_stacks=ref_stacks,
        percentile=percentile,
    )
    return gif_utils.write_gif(frames, path, float(fps))


def save_mp4(
    data: MovieData,
    path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    fps: float = 10,
    gap: int = 12,
    label_height: int = 28,
    max_width: int | None = 1200,
    cols: int | None = None,
    shared_contrast: bool = True,
    ref_stacks: Sequence[np.ndarray] | None = None,
    percentile: tuple[float, float] = (1.0, 99.0),
    crf: int = 18,
    backend: str = "auto",
    **backend_options,
) -> Path:
    """Save one or more grayscale movie stacks as an H.264 MP4.

    Parameters
    ----------
    backend : {"auto", "cuda", "cpu"}
        ``"auto"`` uses the NVIDIA CUDA/NVENC backend when available for array
        inputs, otherwise it falls back to the portable CPU writer.
        ``"cuda"`` requires an NVIDIA CUDA environment.
    """

    backend = str(backend).lower()
    if backend not in {"auto", "cuda", "cpu"}:
        raise ValueError(
            f"unknown movie backend {backend!r}; use 'auto', 'cuda', or 'cpu'"
        )
    if backend == "cuda" and _is_pil_frame_sequence(data):
        raise ValueError(
            "backend='cuda' requires array movie data; rendered PIL frames "
            "must use backend='cpu'"
        )
    if backend in {"auto", "cuda"} and not _is_pil_frame_sequence(data):
        try_cuda = backend == "cuda"
        if backend == "auto":
            try:
                from quantem.widget.kernels import cuda_mp4
            except ImportError:
                cuda_mp4 = None
            try_cuda = bool(cuda_mp4 is not None and cuda_mp4.is_available())
        if try_cuda:
            from quantem.widget.kernels import cuda_mp4

            stacks = _as_stack_list(data)
            _validate_stacks(stacks)
            limits = _contrast_limits(
                stacks,
                percentile=percentile,
                shared=shared_contrast,
                ref_stacks=ref_stacks,
            )
            return cuda_mp4.save_mp4(
                stacks,
                path,
                labels=labels,
                fps=float(fps),
                gap=gap,
                label_height=label_height,
                max_width=max_width,
                cols=cols,
                limits=limits,
                qp=int(backend_options.pop("qp", crf)),
                preset=str(backend_options.pop("preset", "P3")),
                tuning_info=str(backend_options.pop("tuning_info", "high_quality")),
                gpu_id=int(backend_options.pop("gpu_id", 0)),
                faststart=bool(backend_options.pop("faststart", True)),
            )
    frames = _movie_frames(
        data,
        labels=labels,
        gap=gap,
        label_height=label_height,
        max_width=max_width,
        cols=cols,
        shared_contrast=shared_contrast,
        ref_stacks=ref_stacks,
        percentile=percentile,
    )
    return gif_utils.write_mp4(frames, path, float(fps), crf=int(crf))


def save_movie(
    data: MovieData,
    path: str | Path,
    *,
    format: str | None = None,
    **kwargs,
) -> Path:
    """Save a GIF or MP4, using ``format`` or the output suffix."""

    suffix = (format or Path(path).suffix.lstrip(".")).lower()
    if suffix == "gif":
        return save_gif(data, path, **kwargs)
    if suffix == "mp4":
        return save_mp4(data, path, **kwargs)
    raise ValueError(
        "movie format must be 'gif' or 'mp4', or path must end with .gif or .mp4"
    )
