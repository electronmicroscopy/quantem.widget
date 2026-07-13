"""CUDA/NVENC MP4 export kernels for :mod:`quantem.widget.movie`.

This module is intentionally optional. It imports CUDA/NVENC dependencies only
when the caller selects ``backend="cuda"`` or when ``backend="auto"`` probes
for support.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class LabelMask:
    x: int
    y: int
    white: object
    black: object


class CudaArrayView:
    """Expose a CuPy array through CUDA Array Interface for PyNvVideoCodec."""

    def __init__(self, array: object) -> None:
        self._array = array
        self.__cuda_array_interface__ = array.__cuda_array_interface__


class Nv12Frame:
    """PyNvVideoCodec-compatible NV12 frame backed by one CuPy allocation."""

    def __init__(self, nv12: object) -> None:
        height = nv12.shape[0] * 2 // 3
        width = nv12.shape[1]
        self._nv12 = nv12
        self._planes = [
            CudaArrayView(nv12[:height, :, None]),
            CudaArrayView(nv12[height:, :].reshape(height // 2, width // 2, 2)),
        ]

    def cuda(self) -> list[CudaArrayView]:
        return self._planes


def _imports() -> tuple[object, object, object]:
    try:
        import cupy as cp
        import imageio_ffmpeg
        import PyNvVideoCodec as nvc
    except ImportError as exc:
        raise RuntimeError(
            "CUDA movie export requires cupy, imageio-ffmpeg, and "
            "PyNvVideoCodec. Use backend='cpu' or install the NVIDIA "
            "movie dependencies in a CUDA environment."
        ) from exc
    if cp.cuda.runtime.getDeviceCount() <= 0:
        raise RuntimeError("CUDA movie export requires an NVIDIA CUDA device.")
    return cp, imageio_ffmpeg, nvc


def is_available() -> bool:
    """Return whether the CUDA/NVENC backend can be used in this process."""

    try:
        _imports()
    except RuntimeError:
        return False
    return True


def _kernels(cp: object) -> tuple[object, object]:
    scale_grid = cp.RawKernel(
        r'''
        extern "C" __global__
        void scale_grid_nv12(
            const float* const* __restrict__ stacks,
            const float* __restrict__ vmin,
            const float* __restrict__ scale,
            unsigned char* __restrict__ dst,
            const int n_panels,
            const int src_width,
            const int src_height,
            const int frame_stride,
            const int out_width,
            const int out_height,
            const int frame_width,
            const int frame_height,
            const int label_height,
            const int gap,
            const int cols
        ) {
            const int idx = blockDim.x * blockIdx.x + threadIdx.x;
            const int luma_size = out_width * out_height;
            const int total_size = luma_size + (luma_size >> 1);
            if (idx >= total_size) {
                return;
            }
            if (idx >= luma_size) {
                dst[idx] = 128;
                return;
            }

            const int y = idx / out_width;
            const int x = idx - y * out_width;
            unsigned char out = 0;
            const int cell_w = frame_width + gap;
            const int cell_h = label_height + frame_height + gap;
            const int col = cell_w > 0 ? x / cell_w : 0;
            const int row = cell_h > 0 ? y / cell_h : 0;
            const int local_x = x - col * cell_w;
            const int local_y = y - row * cell_h;
            const int panel = row * cols + col;

            if (
                panel >= 0 && panel < n_panels &&
                local_x >= 0 && local_x < frame_width &&
                local_y >= label_height &&
                local_y < label_height + frame_height
            ) {
                int sx = (int)(((long long)local_x * src_width) / frame_width);
                int sy = (int)(((long long)(local_y - label_height) * src_height) / frame_height);
                sx = min(max(sx, 0), src_width - 1);
                sy = min(max(sy, 0), src_height - 1);
                const float value = (stacks[panel][sy * src_width + sx] - vmin[panel]) * scale[panel];
                const float clipped = fminf(fmaxf(value, 0.0f), 255.0f);
                out = (unsigned char)(clipped);
            }
            dst[idx] = out;
        }
        ''',
        "scale_grid_nv12",
    )
    stamp_label = cp.RawKernel(
        r'''
        extern "C" __global__
        void stamp_label(
            unsigned char* __restrict__ dst,
            const int width,
            const int height,
            const unsigned char* __restrict__ white,
            const unsigned char* __restrict__ black,
            const int mask_width,
            const int mask_height,
            const int label_x,
            const int label_y
        ) {
            const int midx = blockDim.x * blockIdx.x + threadIdx.x;
            const int total = mask_width * mask_height;
            if (midx >= total) {
                return;
            }
            const int my = midx / mask_width;
            const int mx = midx - my * mask_width;
            const int x = label_x + mx;
            const int y = label_y + my;
            if (x < 0 || x >= width || y < 0 || y >= height) {
                return;
            }
            const int out_idx = y * width + x;
            if (black[midx] != 0) {
                dst[out_idx] = 0;
            }
            if (white[midx] != 0) {
                dst[out_idx] = 255;
            }
        }
        ''',
        "stamp_label",
    )
    return scale_grid, stamp_label


def _render_label_mask(cp: object, text: str, x: int, y: int, font_size: int) -> LabelMask:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    probe = Image.new("L", (1, 1), 0)
    left, top, right, bottom = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    pad = 4
    mask_width = max(1, right - left + 2 * pad)
    mask_height = max(1, bottom - top + 2 * pad)
    white = Image.new("L", (mask_width, mask_height), 0)
    black = Image.new("L", (mask_width, mask_height), 0)
    white_draw = ImageDraw.Draw(white)
    black_draw = ImageDraw.Draw(black)
    text_x = pad - left
    text_y = pad - top
    for dx in (-1, 1):
        for dy in (-1, 1):
            black_draw.text((text_x + dx, text_y + dy), text, font=font, fill=255)
    white_draw.text((text_x, text_y), text, font=font, fill=255)
    return LabelMask(
        x=x,
        y=y,
        white=cp.asarray(np.asarray(white, dtype=np.uint8)),
        black=cp.asarray(np.asarray(black, dtype=np.uint8)),
    )


def _layout(
    n_panels: int,
    height: int,
    width: int,
    *,
    cols: int | None,
    gap: int,
    label_height: int,
    max_width: int | None,
) -> tuple[int, int, int, int, int, int]:
    n_cols = min(n_panels, 3) if cols is None else max(1, int(cols))
    n_rows = int(math.ceil(n_panels / n_cols))
    gap = max(0, int(gap))
    label_height = max(0, int(label_height))
    total_width = n_cols * width + (n_cols - 1) * gap
    cell_height = height + label_height
    total_height = n_rows * cell_height + (n_rows - 1) * gap
    scale = 1.0
    if max_width is not None and total_width > int(max_width):
        scale = int(max_width) / total_width
    frame_width = max(1, int(round(width * scale)))
    frame_height = max(1, int(round(height * scale)))
    gap_scaled = max(0, int(round(gap * scale)))
    label_height_scaled = max(0, int(round(label_height * scale)))
    out_width = n_cols * frame_width + (n_cols - 1) * gap_scaled
    out_height = n_rows * (frame_height + label_height_scaled) + (n_rows - 1) * gap_scaled
    out_width += out_width % 2
    out_height += out_height % 2
    return out_width, out_height, frame_width, frame_height, gap_scaled, label_height_scaled


def _ffmpeg_mux_command(
    imageio_ffmpeg: object,
    elementary_path: Path,
    mp4_path: Path,
    fps: float,
    *,
    faststart: bool,
) -> list[str]:
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+genpts",
        "-f",
        "h264",
        "-r",
        str(max(0.1, float(fps))),
        "-i",
        str(elementary_path),
        "-c:v",
        "copy",
        "-an",
    ]
    if faststart:
        cmd.extend(["-movflags", "+faststart"])
    cmd.append(str(mp4_path))
    return cmd


def _packet_bytes(packets: list[dict[str, object]]) -> bytes:
    return b"".join(bytes(packet["data"]) for packet in packets)


def save_mp4(
    stacks: Sequence[np.ndarray],
    path: str | Path,
    *,
    labels: Sequence[str] | None,
    fps: float,
    gap: int,
    label_height: int,
    max_width: int | None,
    cols: int | None,
    limits: Sequence[tuple[float, float]],
    qp: int = 18,
    preset: str = "P3",
    tuning_info: str = "high_quality",
    gpu_id: int = 0,
    faststart: bool = True,
) -> Path:
    """Save a grayscale movie grid as H.264 MP4 using NVIDIA NVENC."""

    cp, imageio_ffmpeg, nvc = _imports()
    cp.cuda.Device(int(gpu_id)).use()

    if not stacks:
        raise ValueError("cuda_mp4.save_mp4 requires at least one stack")
    frames, height, width = stacks[0].shape
    n_panels = len(stacks)
    names = (
        [str(item) for item in labels]
        if labels is not None
        else [f"Movie {i + 1}" for i in range(n_panels)]
    )
    n_cols = min(n_panels, 3) if cols is None else max(1, int(cols))
    out_width, out_height, frame_width, frame_height, gap_scaled, label_height_scaled = _layout(
        n_panels,
        height,
        width,
        cols=cols,
        gap=gap,
        label_height=label_height,
        max_width=max_width,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stacks_gpu = [cp.asarray(np.asarray(stack, dtype=np.float32)) for stack in stacks]
    ptrs = cp.asarray([stack.data.ptr for stack in stacks_gpu], dtype=cp.uintp)
    vmin = cp.asarray([lo for lo, _hi in limits], dtype=cp.float32)
    scale = cp.asarray([255.0 / max(hi - lo, 1e-6) for lo, hi in limits], dtype=cp.float32)
    label_masks: list[list[LabelMask]] = [[] for _ in range(frames)]
    if label_height_scaled > 0:
        font_size = max(8, int(label_height_scaled) - 8)
        for frame_idx in range(frames):
            for panel_idx in range(n_panels):
                row, col = divmod(panel_idx, n_cols)
                x = col * (frame_width + gap_scaled) + 4
                y = row * (frame_height + label_height_scaled + gap_scaled) + 2
                text = f"{names[panel_idx]} [{frame_idx + 1}/{frames}]"
                label_masks[frame_idx].append(_render_label_mask(cp, text, x, y, font_size))

    scale_grid, stamp_label = _kernels(cp)
    nv12_buffers = [
        cp.empty((out_height + out_height // 2, out_width), dtype=cp.uint8)
        for _ in range(4)
    ]
    nv12_frames = [Nv12Frame(buf) for buf in nv12_buffers]
    config = {
        "codec": "h264",
        "gpu_id": int(gpu_id),
        "preset": str(preset).upper(),
        "tuning_info": str(tuning_info),
        "rc": "constqp",
        "qp": str(int(qp)),
        "fps": max(0.1, float(fps)),
    }
    encoder = nvc.CreateEncoder(out_width, out_height, "NV12", False, **config)
    block = 256
    total = out_width * out_height * 3 // 2
    grid = ((total + block - 1) // block,)

    with tempfile.NamedTemporaryFile(suffix=".h264", delete=False) as tmp:
        elementary_path = Path(tmp.name)
        try:
            for frame_idx in range(frames):
                ring_idx = frame_idx % len(nv12_buffers)
                frame_ptrs = cp.asarray(
                    [stack[frame_idx].data.ptr for stack in stacks_gpu],
                    dtype=cp.uintp,
                )
                scale_grid(
                    grid,
                    (block,),
                    (
                        frame_ptrs,
                        vmin,
                        scale,
                        nv12_buffers[ring_idx],
                        np.int32(n_panels),
                        np.int32(width),
                        np.int32(height),
                        np.int32(width * height),
                        np.int32(out_width),
                        np.int32(out_height),
                        np.int32(frame_width),
                        np.int32(frame_height),
                        np.int32(label_height_scaled),
                        np.int32(gap_scaled),
                        np.int32(n_cols),
                    ),
                )
                for mask in label_masks[frame_idx]:
                    mask_h, mask_w = mask.white.shape
                    mask_total = int(mask_h * mask_w)
                    mask_grid = ((mask_total + block - 1) // block,)
                    stamp_label(
                        mask_grid,
                        (block,),
                        (
                            nv12_buffers[ring_idx],
                            np.int32(out_width),
                            np.int32(out_height),
                            mask.white,
                            mask.black,
                            np.int32(mask_w),
                            np.int32(mask_h),
                            np.int32(mask.x),
                            np.int32(mask.y),
                        ),
                    )
                pic_params = nvc.NV_ENC_PIC_PARAMS()
                pic_params.inputTimeStamp = frame_idx
                tmp.write(_packet_bytes(encoder.Encode(nv12_frames[ring_idx], pic_params)))
            tmp.write(_packet_bytes(encoder.EndEncode()))
        finally:
            tmp.close()
    try:
        cmd = _ffmpeg_mux_command(imageio_ffmpeg, elementary_path, path, fps, faststart=faststart)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed while muxing NVENC MP4: {exc}") from exc
    finally:
        elementary_path.unlink(missing_ok=True)
    cp.cuda.Stream.null.synchronize()
    return path
