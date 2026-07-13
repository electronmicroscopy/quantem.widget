#!/usr/bin/env python3
"""Generate full-resolution denoising comparison MP4s with NVENC.

This script is intended for mjgoat's ``cuda-env``:

    CUDA_VISIBLE_DEVICES=0 /home/owner/miniforge3/envs/cuda-env/bin/python \
        scripts/make_movies_gpu.py 400C_5.1Mx

It keeps contrast scaling, label compositing, optional tiling, and NVENC input
surfaces on the GPU.  FFmpeg is used only for stream-copy muxing of the NVENC
elementary stream into MP4, never for video encoding.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import av
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyAV is required for MP4 validation: python -m pip install av") from exc

try:
    import imageio_ffmpeg
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "imageio-ffmpeg is required for MP4 stream-copy muxing: "
        "python -m pip install imageio-ffmpeg"
    ) from exc

try:
    import PyNvVideoCodec as nvc
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "PyNvVideoCodec is required for NVENC: python -m pip install PyNvVideoCodec"
    ) from exc


CACHE_DIR = Path("/home/owner/publications/denoise-paper/build/datasets/fig2_3x3_fullres")
OUT_ROOT = Path("/home/owner/publications/denoise-paper/figures/movies")
FPS = 12
PANELS = [0, 3, 4, 5, 6, 7, 8]
CPU_BASELINE = {
    # Full raw-panel CPU reference measured on mjgoat with the original
    # libx264/PIL path for 800C_1.3Mx, 69 native 2048x2048 frames.
    # Override with --cpu-baseline-fps when a slug-specific run is available.
    "400C_5.1Mx": 18.224934092100852,
    "800C_1.3Mx": 18.224934092100852,
    "800C_3.6Mx": 18.224934092100852,
}


class CudaArrayView:
    """Small wrapper exposing CuPy arrays through CUDA Array Interface."""

    def __init__(self, array: cp.ndarray) -> None:
        self._array = array
        self.__cuda_array_interface__ = array.__cuda_array_interface__


class Nv12Frame:
    """PyNvVideoCodec-compatible NV12 frame backed by one CuPy allocation."""

    def __init__(self, nv12: cp.ndarray) -> None:
        height = nv12.shape[0] * 2 // 3
        width = nv12.shape[1]
        self._nv12 = nv12
        self._planes = [
            CudaArrayView(nv12[:height, :, None]),
            CudaArrayView(nv12[height:, :].reshape(height // 2, width // 2, 2)),
        ]

    def cuda(self) -> list[CudaArrayView]:
        return self._planes


FUSED_NV12_KERNEL = cp.RawKernel(
    r'''
    extern "C" __global__
    void scale_label_nv12(
        const float* __restrict__ src,
        unsigned char* __restrict__ dst,
        const int width,
        const int height,
        const float vmin,
        const float scale,
        const unsigned char* __restrict__ white,
        const unsigned char* __restrict__ black,
        const int mask_width,
        const int mask_height,
        const int label_x,
        const int label_y,
        const int use_label
    ) {
        const int idx = blockDim.x * blockIdx.x + threadIdx.x;
        const int luma_size = width * height;
        const int total_size = luma_size + (luma_size >> 1);
        if (idx >= total_size) {
            return;
        }
        if (idx >= luma_size) {
            dst[idx] = 128;
            return;
        }

        float value = (src[idx] - vmin) * scale;
        value = fminf(fmaxf(value, 0.0f), 255.0f);
        unsigned char out = (unsigned char)(value);

        if (use_label) {
            const int y = idx / width;
            const int x = idx - y * width;
            const int mx = x - label_x;
            const int my = y - label_y;
            if (mx >= 0 && mx < mask_width && my >= 0 && my < mask_height) {
                const int midx = my * mask_width + mx;
                if (black[midx] != 0) {
                    out = 0;
                }
                if (white[midx] != 0) {
                    out = 255;
                }
            }
        }
        dst[idx] = out;
    }
    ''',
    "scale_label_nv12",
)


@dataclass
class EncodeStats:
    frames: int
    seconds: float
    fps: float
    output_bytes: int
    nvenc_session_seen: bool
    max_nvenc_sessions: int
    gpu_name: str
    transfer_seconds: float = 0.0
    mux_seconds: float = 0.0
    validate_seconds: float = 0.0


@dataclass
class LabelMask:
    x: int
    y: int
    white: cp.ndarray
    black: cp.ndarray


class PacketSink:
    """Write encoded packets either to a temp elementary file or ffmpeg stdin."""

    def __init__(self, out_path: Path, codec: str, fps: int, mode: str) -> None:
        self.out_path = out_path
        self.codec = codec
        self.fps = fps
        self.mode = mode
        self.elementary_path: Path | None = None
        self.handle: Any | None = None
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "PacketSink":
        if self.mode == "temp":
            with tempfile.NamedTemporaryFile(suffix=f".{codec_to_extension(self.codec)}", delete=False) as tmp:
                self.elementary_path = Path(tmp.name)
            self.handle = self.elementary_path.open("wb")
            return self
        if self.mode == "pipe":
            cmd = ffmpeg_mux_command("-", self.out_path, self.codec, self.fps)
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=False,
            )
            if self.process.stdin is None:
                raise RuntimeError("ffmpeg stdin pipe was not created")
            self.handle = self.process.stdin
            return self
        raise ValueError(f"unknown mux mode {self.mode!r}")

    def write(self, data: bytes) -> None:
        if self.handle is None:
            raise RuntimeError("packet sink is not open")
        self.handle.write(data)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self.handle is not None:
            self.handle.close()
        if self.process is not None:
            _stdout, stderr = self.process.communicate()
            if exc_type is None and self.process.returncode != 0:
                message = stderr.decode("utf-8", errors="replace") if stderr else ""
                raise RuntimeError(f"ffmpeg pipe mux failed for {self.out_path}: {message}")
        return False

    def finish_temp_mux(self) -> float:
        if self.mode != "temp":
            return 0.0
        if self.elementary_path is None:
            raise RuntimeError("temp mux requested without an elementary path")
        mux_start = time.perf_counter()
        mux_elementary_to_mp4(self.elementary_path, self.out_path, self.codec, self.fps)
        mux_seconds = time.perf_counter() - mux_start
        self.elementary_path.unlink(missing_ok=True)
        return mux_seconds


def panel_name(title: str, idx: int) -> str:
    """Return the reference-compatible panel filename slug."""

    cleaned = re.sub(r"\s+chi2.*|\s+r=.*", "", title)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return cleaned or f"panel{idx}"


def codec_to_extension(codec: str) -> str:
    return {"h264": "h264", "hevc": "hevc", "av1": "ivf"}[codec]


def codec_to_container_probe_name(codec: str) -> str:
    return {"h264": "h264", "hevc": "hevc", "av1": "av1"}[codec]


def physical_gpu_id(gpu_id: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return gpu_id
    ids = [item.strip() for item in visible.split(",") if item.strip()]
    if gpu_id >= len(ids):
        return gpu_id
    try:
        return int(ids[gpu_id])
    except ValueError:
        return gpu_id


def read_gpu_info(gpu_id: int) -> tuple[str, int]:
    query_gpu_id = physical_gpu_id(gpu_id)
    cmd = [
        "nvidia-smi",
        f"--id={query_gpu_id}",
        "--query-gpu=name,encoder.stats.sessionCount",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown GPU", 0
    name, count = [part.strip() for part in out.split(",", maxsplit=1)]
    return name, int(count)


def monitor_nvenc(gpu_id: int, stop: threading.Event, samples: list[int]) -> None:
    while not stop.is_set():
        _, count = read_gpu_info(gpu_id)
        samples.append(count)
        time.sleep(0.05)


def render_label_mask(
    text: str,
    font_size: int = 34,
) -> LabelMask:
    """Render one small white text mask and black outline mask, then upload to GPU."""

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    x, y = 20, 16
    probe = Image.new("L", (1, 1), 0)
    left, top, right, bottom = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    pad = 6
    mask_width = right - left + 2 * pad
    mask_height = bottom - top + 2 * pad
    white = Image.new("L", (mask_width, mask_height), 0)
    black = Image.new("L", (mask_width, mask_height), 0)
    white_draw = ImageDraw.Draw(white)
    black_draw = ImageDraw.Draw(black)
    text_x = pad - left
    text_y = pad - top
    for dx in (-2, 2):
        for dy in (-2, 2):
            black_draw.text((text_x + dx, text_y + dy), text, font=font, fill=255)
    white_draw.text((text_x, text_y), text, font=font, fill=255)
    return LabelMask(
        x=x - pad,
        y=y - pad,
        white=cp.asarray(np.asarray(white, dtype=np.uint8)),
        black=cp.asarray(np.asarray(black, dtype=np.uint8)),
    )


def memmap_npy_member_from_stored_npz(npz_path: Path, member: str) -> np.memmap:
    """Memory-map an uncompressed ``.npy`` member stored inside an ``.npz`` file.

    ``np.load`` cannot memory-map arrays inside ``.npz`` archives, even when the
    ZIP member is stored without compression.  These denoise-paper caches are
    ZIP_STORED, so the embedded ``panels.npy`` payload is contiguous on disk.
    We parse the ZIP local header and the NPY header, then create a memmap that
    starts directly at the array bytes.
    """

    with zipfile.ZipFile(npz_path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(
                f"{npz_path}:{member} is compressed; direct memmap requires ZIP_STORED. "
                "Convert/extract the cache to .npy, Zarr, or another chunked format."
            )

    with npz_path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        if len(header) != 30:
            raise RuntimeError(f"{npz_path}:{member} has a truncated ZIP local header")
        (
            signature,
            _version,
            _flags,
            compression,
            _mod_time,
            _mod_date,
            _crc,
            _compressed_size,
            _uncompressed_size,
            filename_length,
            extra_length,
        ) = struct.unpack("<IHHHHHIIIHH", header)
        if signature != 0x04034B50 or compression != zipfile.ZIP_STORED:
            raise RuntimeError(f"{npz_path}:{member} is not a stored ZIP member")

        npy_start = info.header_offset + 30 + filename_length + extra_length
        handle.seek(npy_start)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise RuntimeError(f"{npz_path}:{member} uses unsupported NPY version {version}")
        data_start = handle.tell()

    order = "F" if fortran_order else "C"
    return np.memmap(npz_path, dtype=dtype, mode="r", offset=data_start, shape=shape, order=order)


def load_cache_arrays(cache_path: Path, load_mode: str) -> tuple[np.ndarray, list[str], str]:
    """Load titles and expose panels through either direct memmap or np.load."""

    loaded = np.load(cache_path, allow_pickle=True)
    titles = [str(title) for title in loaded["titles"]]
    if load_mode == "np-load":
        panels = loaded["panels"]
        return panels, titles, "np.load full materialization"
    if load_mode == "npz-memmap":
        panels = memmap_npy_member_from_stored_npz(cache_path, "panels.npy")
        return panels, titles, "direct memmap of ZIP_STORED panels.npy inside .npz"
    raise ValueError(f"unknown load mode {load_mode!r}")


def copy_panels_to_gpu(panels: np.ndarray, indices: list[int]) -> cp.ndarray:
    """Copy selected panels to GPU one at a time to avoid a large CPU gather copy."""

    first = np.asarray(panels[indices[0]])
    out = cp.empty((len(indices),) + first.shape, dtype=first.dtype)
    out[0] = cp.asarray(first)
    for local_idx, panel_idx in enumerate(indices[1:], start=1):
        out[local_idx] = cp.asarray(np.asarray(panels[panel_idx]))
    cp.cuda.Stream.null.synchronize()
    return out


def copy_panel_to_gpu(panels: np.ndarray, panel_idx: int) -> tuple[cp.ndarray, float]:
    """Copy one CPU/memmap panel stack to the active GPU and time the transfer."""

    transfer_start = time.perf_counter()
    panel_gpu = cp.asarray(np.asarray(panels[panel_idx]))
    cp.cuda.Stream.null.synchronize()
    return panel_gpu, time.perf_counter() - transfer_start


def stamp_on_gpu(frame: cp.ndarray, label: LabelMask, x_offset: int = 0, y_offset: int = 0) -> cp.ndarray:
    """Composite one pre-rendered label mask into a grayscale GPU frame."""

    x0 = label.x + x_offset
    y0 = label.y + y_offset
    h, w = label.white.shape
    region = frame[y0 : y0 + h, x0 : x0 + w]
    region = cp.where(label.black > 0, cp.uint8(0), region)
    region = cp.where(label.white > 0, cp.uint8(255), region)
    frame[y0 : y0 + h, x0 : x0 + w] = region
    return frame


def scaled_u8(frame: cp.ndarray, vmin: float, scale: float) -> cp.ndarray:
    return cp.clip((frame - vmin) * scale, 0, 255).astype(cp.uint8)


def make_nv12_from_gray(gray: cp.ndarray, chroma: cp.ndarray) -> cp.ndarray:
    return cp.concatenate((gray, chroma), axis=0)


def fused_frame_to_nv12(
    frame: cp.ndarray,
    nv12: cp.ndarray,
    label: LabelMask | None,
    vmin: float,
    scale: float,
    use_label: bool,
) -> cp.ndarray:
    """Scale one float32 frame, stamp an optional label, and pack NV12 in one CUDA kernel."""

    height = frame.shape[0]
    width = frame.shape[1]
    total = width * height * 3 // 2
    if label is None or not use_label:
        white = cp.zeros((1, 1), dtype=cp.uint8)
        black = white
        mask_height, mask_width = 1, 1
        label_x, label_y = 0, 0
        use_label_int = 0
    else:
        white = label.white
        black = label.black
        mask_height, mask_width = white.shape
        label_x, label_y = label.x, label.y
        use_label_int = 1
    block = 256
    grid = ((total + block - 1) // block,)
    FUSED_NV12_KERNEL(
        grid,
        (block,),
        (
            frame,
            nv12,
            np.int32(width),
            np.int32(height),
            np.float32(vmin),
            np.float32(scale),
            white,
            black,
            np.int32(mask_width),
            np.int32(mask_height),
            np.int32(label_x),
            np.int32(label_y),
            np.int32(use_label_int),
        ),
    )
    return nv12


def concat_packets(packets: list[dict[str, Any]]) -> bytes:
    return b"".join(bytes(packet["data"]) for packet in packets)


def encoder_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "codec": args.codec,
        "gpu_id": args.gpu_id,
        "preset": args.preset.upper(),
        "tuning_info": "high_quality",
        "rc": "constqp",
        "qp": str(args.qp),
        "fps": args.fps,
    }
    if args.codec == "av1":
        config["use_ivf_container"] = "1"
    return config


def mux_elementary_to_mp4(
    elementary_path: Path,
    mp4_path: Path,
    codec: str,
    fps: int,
) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    input_format = {"h264": "h264", "hevc": "hevc", "av1": "ivf"}[codec]
    cmd = ffmpeg_mux_command(elementary_path, mp4_path, codec, fps)
    subprocess.run(cmd, check=True)


def ffmpeg_mux_command(
    elementary_path: Path | str,
    mp4_path: Path,
    codec: str,
    fps: int,
) -> list[str]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    input_format = {"h264": "h264", "hevc": "hevc", "av1": "ivf"}[codec]
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+genpts",
        "-f",
        input_format,
        "-r",
        str(fps),
        "-i",
        str(elementary_path),
        "-c:v",
        "copy",
        "-an",
        "-movflags",
        "+faststart",
        str(mp4_path),
    ]


def validate_mp4(path: Path, codec: str, width: int, height: int, frames: int) -> dict[str, Any]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    actual = {
        "codec": stream.codec_context.name,
        "width": stream.width,
        "height": stream.height,
        "frames": stream.frames,
    }
    if actual["frames"] == 0:
        actual["frames"] = sum(1 for _ in container.decode(stream))
    expected_codec = codec_to_container_probe_name(codec)
    if actual["codec"] != expected_codec:
        raise RuntimeError(f"{path}: codec {actual['codec']} != {expected_codec}")
    if (actual["width"], actual["height"]) != (width, height):
        raise RuntimeError(f"{path}: dimensions {(actual['width'], actual['height'])} != {(width, height)}")
    if actual["frames"] != frames:
        raise RuntimeError(f"{path}: frame count {actual['frames']} != {frames}")
    return actual


def maybe_validate_mp4(
    path: Path,
    codec: str,
    width: int,
    height: int,
    frames: int,
    enabled: bool,
) -> tuple[dict[str, Any] | None, float]:
    if not enabled:
        return None, 0.0
    validate_start = time.perf_counter()
    validated = validate_mp4(path, codec, width, height, frames)
    return validated, time.perf_counter() - validate_start


def encode_gray_stack(
    stack_gpu: cp.ndarray,
    title: str,
    out_path: Path,
    vmin: float,
    scale: float,
    args: argparse.Namespace,
) -> EncodeStats:
    frames, height, width = stack_gpu.shape
    chroma = cp.full((height // 2, width), 128, dtype=cp.uint8)
    labels: list[LabelMask] = []
    for frame_idx in range(frames):
        text = f"{title}   {frame_idx + 1}/{frames}   full-res (no bin/crop)"
        labels.append(render_label_mask(text))

    cfg = encoder_config(args)
    encoder = nvc.CreateEncoder(width, height, "NV12", False, **cfg)
    gpu_name, _ = read_gpu_info(args.gpu_id)
    samples: list[int] = []
    stop_monitor = threading.Event()
    monitor = threading.Thread(target=monitor_nvenc, args=(args.gpu_id, stop_monitor, samples), daemon=True)

    start = time.perf_counter()
    monitor.start()
    try:
        with PacketSink(out_path, args.codec, args.fps, args.mux_mode) as encoded:
            for frame_idx in range(frames):
                gray = scaled_u8(stack_gpu[frame_idx], vmin, scale)
                if args.labels:
                    gray = stamp_on_gpu(gray, labels[frame_idx])
                nv12 = make_nv12_from_gray(gray, chroma)
                pic_params = nvc.NV_ENC_PIC_PARAMS()
                pic_params.inputTimeStamp = frame_idx
                encoded.write(concat_packets(encoder.Encode(Nv12Frame(nv12), pic_params)))
            encoded.write(concat_packets(encoder.EndEncode()))
        mux_seconds = encoded.finish_temp_mux()
    finally:
        stop_monitor.set()
        monitor.join(timeout=2)
    cp.cuda.Stream.null.synchronize()
    seconds = time.perf_counter() - start
    validated, validate_seconds = maybe_validate_mp4(out_path, args.codec, width, height, frames, args.validate)
    if validated is not None and (validated["width"] != 2048 or validated["height"] != 2048):
        raise RuntimeError(f"{out_path}: full-resolution rule violated")

    max_sessions = max(samples, default=0)
    if not args.no_assert_nvenc and max_sessions <= 0:
        raise RuntimeError(
            f"{out_path}: nvidia-smi did not report an active NVENC session on GPU {args.gpu_id}"
        )

    return EncodeStats(
        frames=frames,
        seconds=seconds,
        fps=frames / seconds,
        output_bytes=out_path.stat().st_size,
        nvenc_session_seen=max_sessions > 0,
        max_nvenc_sessions=max_sessions,
        gpu_name=gpu_name,
        mux_seconds=mux_seconds,
        validate_seconds=validate_seconds,
    )


def encode_gray_stack_fused(
    stack_gpu: cp.ndarray,
    title: str,
    out_path: Path,
    vmin: float,
    scale: float,
    args: argparse.Namespace,
) -> EncodeStats:
    """Encode one panel using one fused CUDA kernel per frame."""

    cp.cuda.Device(args.gpu_id).use()
    frames, height, width = stack_gpu.shape
    labels: list[LabelMask] = []
    if args.labels:
        for frame_idx in range(frames):
            text = f"{title}   {frame_idx + 1}/{frames}   full-res (no bin/crop)"
            labels.append(render_label_mask(text))

    cfg = encoder_config(args)
    encoder = nvc.CreateEncoder(width, height, "NV12", False, **cfg)
    gpu_name, _ = read_gpu_info(args.gpu_id)
    samples: list[int] = []
    stop_monitor = threading.Event()
    monitor = threading.Thread(target=monitor_nvenc, args=(args.gpu_id, stop_monitor, samples), daemon=True)

    ring_size = max(1, args.nv12_ring)
    nv12_buffers = [cp.empty((height + height // 2, width), dtype=cp.uint8) for _ in range(ring_size)]
    nv12_frames = [Nv12Frame(buf) for buf in nv12_buffers]
    start = time.perf_counter()
    monitor.start()
    try:
        with PacketSink(out_path, args.codec, args.fps, args.mux_mode) as encoded:
            for frame_idx in range(frames):
                ring_idx = frame_idx % ring_size
                fused_frame_to_nv12(
                    stack_gpu[frame_idx],
                    nv12_buffers[ring_idx],
                    labels[frame_idx] if args.labels else None,
                    vmin,
                    scale,
                    args.labels,
                )
                pic_params = nvc.NV_ENC_PIC_PARAMS()
                pic_params.inputTimeStamp = frame_idx
                encoded.write(concat_packets(encoder.Encode(nv12_frames[ring_idx], pic_params)))
            encoded.write(concat_packets(encoder.EndEncode()))
        mux_seconds = encoded.finish_temp_mux()
    finally:
        stop_monitor.set()
        monitor.join(timeout=2)
    cp.cuda.Stream.null.synchronize()
    seconds = time.perf_counter() - start
    validated, validate_seconds = maybe_validate_mp4(out_path, args.codec, width, height, frames, args.validate)
    if validated is not None and (validated["width"] != 2048 or validated["height"] != 2048):
        raise RuntimeError(f"{out_path}: full-resolution rule violated")

    max_sessions = max(samples, default=0)
    if not args.no_assert_nvenc and max_sessions <= 0:
        raise RuntimeError(
            f"{out_path}: nvidia-smi did not report an active NVENC session on GPU {args.gpu_id}"
        )

    return EncodeStats(
        frames=frames,
        seconds=seconds,
        fps=frames / seconds,
        output_bytes=out_path.stat().st_size,
        nvenc_session_seen=max_sessions > 0,
        max_nvenc_sessions=max_sessions,
        gpu_name=gpu_name,
        mux_seconds=mux_seconds,
        validate_seconds=validate_seconds,
    )


def encode_panel_worker(
    panels_gpu: cp.ndarray,
    local_idx: int,
    panel_idx: int,
    titles: list[str],
    out_dir: Path,
    slug: str,
    vmin: float,
    scale: float,
    args: argparse.Namespace,
) -> tuple[str, EncodeStats]:
    title = titles[panel_idx]
    name = panel_name(title, panel_idx)
    out_path = out_dir / f"fig2_{slug}_{name}.mp4"
    stats = encode_gray_stack_fused(panels_gpu[local_idx], title, out_path, vmin, scale, args)
    return out_path.name, stats


def encode_panel_worker_from_cpu(
    panels: np.ndarray,
    panel_idx: int,
    titles: list[str],
    out_dir: Path,
    slug: str,
    vmin: float,
    scale: float,
    args: argparse.Namespace,
) -> tuple[str, EncodeStats]:
    """Copy one panel stack inside the worker, then encode it."""

    cp.cuda.Device(args.gpu_id).use()
    title = titles[panel_idx]
    name = panel_name(title, panel_idx)
    out_path = out_dir / f"fig2_{slug}_{name}.mp4"
    panel_gpu, transfer_seconds = copy_panel_to_gpu(panels, panel_idx)
    try:
        stats = encode_gray_stack_fused(panel_gpu, title, out_path, vmin, scale, args)
    finally:
        del panel_gpu
    stats.transfer_seconds = transfer_seconds
    return out_path.name, stats


def contrast_cache_path(out_dir: Path, slug: str) -> Path:
    return out_dir / f"fig2_{slug}_contrast_window.json"


def read_contrast_cache(path: Path, frames: int, height: int, width: int) -> tuple[float, float] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("method") != "raw_percentile_1_99":
        return None
    if data.get("shape") != [frames, height, width]:
        return None
    try:
        return float(data["vmin"]), float(data["vmax"])
    except (KeyError, TypeError, ValueError):
        return None


def write_contrast_cache(path: Path, frames: int, height: int, width: int, vmin: float, vmax: float) -> None:
    payload = {
        "method": "raw_percentile_1_99",
        "shape": [frames, height, width],
        "vmin": vmin,
        "vmax": vmax,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def encode_tiled(
    panels_gpu: cp.ndarray,
    titles: list[str],
    out_path: Path,
    vmin: float,
    scale: float,
    args: argparse.Namespace,
) -> EncodeStats:
    frames = panels_gpu.shape[1]
    tile_h, tile_w = panels_gpu.shape[2], panels_gpu.shape[3]
    height, width = tile_h * 3, tile_w * 3
    chroma = cp.full((height // 2, width), 128, dtype=cp.uint8)
    labels: list[list[LabelMask]] = []
    for frame_idx in range(frames):
        frame_labels = []
        for panel_idx, title in enumerate(titles):
            text = f"{title}   {frame_idx + 1}/{frames}"
            frame_labels.append(render_label_mask(text, font_size=34))
        labels.append(frame_labels)

    cfg = encoder_config(args)
    encoder = nvc.CreateEncoder(width, height, "NV12", False, **cfg)
    gpu_name, _ = read_gpu_info(args.gpu_id)
    samples: list[int] = []
    stop_monitor = threading.Event()
    monitor = threading.Thread(target=monitor_nvenc, args=(args.gpu_id, stop_monitor, samples), daemon=True)

    start = time.perf_counter()
    monitor.start()
    try:
        with PacketSink(out_path, args.codec, args.fps, args.mux_mode) as encoded:
            for frame_idx in range(frames):
                rows = []
                for row in range(3):
                    row_frames = []
                    for col in range(3):
                        panel_idx = row * 3 + col
                        gray = scaled_u8(panels_gpu[panel_idx, frame_idx], vmin, scale)
                        row_frames.append(gray)
                    rows.append(cp.concatenate(row_frames, axis=1))
                gray_tiled = cp.concatenate(rows, axis=0)
                if args.labels:
                    for panel_idx, label in enumerate(labels[frame_idx]):
                        row, col = divmod(panel_idx, 3)
                        gray_tiled = stamp_on_gpu(
                            gray_tiled,
                            label,
                            x_offset=col * tile_w,
                            y_offset=row * tile_h,
                        )
                nv12 = make_nv12_from_gray(gray_tiled, chroma)
                pic_params = nvc.NV_ENC_PIC_PARAMS()
                pic_params.inputTimeStamp = frame_idx
                encoded.write(concat_packets(encoder.Encode(Nv12Frame(nv12), pic_params)))
            encoded.write(concat_packets(encoder.EndEncode()))
        mux_seconds = encoded.finish_temp_mux()
    finally:
        stop_monitor.set()
        monitor.join(timeout=2)
    cp.cuda.Stream.null.synchronize()
    seconds = time.perf_counter() - start
    _, validate_seconds = maybe_validate_mp4(out_path, args.codec, width, height, frames, args.validate)
    max_sessions = max(samples, default=0)
    if not args.no_assert_nvenc and max_sessions <= 0:
        raise RuntimeError(
            f"{out_path}: nvidia-smi did not report an active NVENC session on GPU {args.gpu_id}"
        )

    return EncodeStats(
        frames=frames,
        seconds=seconds,
        fps=frames / seconds,
        output_bytes=out_path.stat().st_size,
        nvenc_session_seen=max_sessions > 0,
        max_nvenc_sessions=max_sessions,
        gpu_name=gpu_name,
        mux_seconds=mux_seconds,
        validate_seconds=validate_seconds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--codec", choices=["h264", "hevc", "av1"], default="h264")
    parser.add_argument(
        "--load-mode",
        choices=["npz-memmap", "np-load"],
        default="npz-memmap",
        help="Default maps uncompressed panels.npy directly inside the .npz instead of materializing all arrays.",
    )
    parser.add_argument("--qp", type=int, default=18, help="NVENC constqp value; lower is higher quality.")
    parser.add_argument("--preset", default="P3", help="NVENC preset, for example P3 or P5.")
    parser.add_argument(
        "--mux-mode",
        choices=["temp", "pipe"],
        default="temp",
        help="Mux via a temporary elementary stream or pipe packets directly to ffmpeg.",
    )
    parser.add_argument("--labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tiled", action="store_true", help="Also emit a 3x3 tiled movie.")
    parser.add_argument(
        "--contrast-cache",
        choices=["off", "read", "write", "read-write"],
        default="off",
        help="Reuse/store the raw 1/99 percentile window in the output directory.",
    )
    parser.add_argument(
        "--panel-transfer",
        choices=["preload", "worker"],
        default="preload",
        help="Copy all panels before encoding, or copy each panel inside its encoder worker.",
    )
    parser.add_argument("--cpu-baseline-fps", type=float, default=None)
    parser.add_argument("--no-assert-nvenc", action="store_true")
    parser.add_argument(
        "--legacy-cupy-pipeline",
        action="store_true",
        help="Use the original multi-operation CuPy path instead of the fused CUDA kernel.",
    )
    parser.add_argument(
        "--parallel-panels",
        type=int,
        default=1,
        help="Encode this many per-panel movies concurrently. Use 7 to run all panels at once.",
    )
    parser.add_argument(
        "--nv12-ring",
        type=int,
        default=4,
        help="Number of reusable GPU NV12 buffers per encoder.",
    )
    return parser.parse_args()


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    cp.cuda.Device(args.gpu_id).use()
    cache_path = args.cache_dir / f"{args.slug}.npz"
    out_dir = args.out_root / args.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_name, session_count = read_gpu_info(args.gpu_id)
    print(f"[{args.slug}] GPU {args.gpu_id}: {gpu_name}; active NVENC sessions before encode: {session_count}")
    print(f"[{args.slug}] loading {cache_path}")
    profile: dict[str, Any] = {}
    load_start = time.perf_counter()
    panels, titles, load_description = load_cache_arrays(cache_path, args.load_mode)
    load_seconds = time.perf_counter() - load_start
    profile["cache_open_seconds"] = load_seconds
    profile["cache_loader"] = load_description
    print(f"[{args.slug}] cache loader: {load_description} ({load_seconds:.3f} s)")
    _, frames, height, width = panels.shape
    if (height, width) != (2048, 2048):
        raise RuntimeError(f"Input cache is not native 2048x2048: {(height, width)}")

    percentile_start = time.perf_counter()
    cache_path_for_contrast = contrast_cache_path(out_dir, args.slug)
    cached_window = None
    if args.contrast_cache in {"read", "read-write"}:
        cached_window = read_contrast_cache(cache_path_for_contrast, frames, height, width)
    if cached_window is None:
        raw_gpu = cp.asarray(panels[0])
        vmin_gpu, vmax_gpu = cp.percentile(raw_gpu, cp.asarray([1, 99], dtype=cp.float32))
        vmin = float(vmin_gpu.get())
        vmax = float(vmax_gpu.get())
        del raw_gpu
        cp.get_default_memory_pool().free_all_blocks()
        profile["raw_percentile_cache_hit"] = False
        if args.contrast_cache in {"write", "read-write"}:
            write_contrast_cache(cache_path_for_contrast, frames, height, width, vmin, vmax)
    else:
        vmin, vmax = cached_window
        profile["raw_percentile_cache_hit"] = True
    scale = 255.0 / max(vmax - vmin, 1e-6)
    profile["raw_percentile_seconds"] = time.perf_counter() - percentile_start
    print(f"[{args.slug}] raw percentile window: [{vmin:.6g}, {vmax:.6g}]")

    baseline = args.cpu_baseline_fps or CPU_BASELINE.get(args.slug, CPU_BASELINE["800C_1.3Mx"])
    all_stats: dict[str, dict[str, Any]] = {}
    if args.parallel_panels > 1 and args.legacy_cupy_pipeline:
        raise RuntimeError("--parallel-panels requires the fused CUDA path")
    if args.parallel_panels > 1:
        worker_count = min(args.parallel_panels, len(PANELS))
        profile["panel_transfer_mode"] = args.panel_transfer
        if args.panel_transfer == "preload":
            print(f"[{args.slug}] preloading {len(PANELS)} output panels to GPU for {worker_count} encoders")
            preload_start = time.perf_counter()
            selected_panels_gpu = copy_panels_to_gpu(panels, PANELS)
            profile["panel_preload_to_gpu_seconds"] = time.perf_counter() - preload_start
            cp.cuda.Stream.null.synchronize()
        print(f"[{args.slug}] encoding {len(PANELS)} panels with {worker_count} concurrent NVENC sessions")
        encode_wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            if args.panel_transfer == "preload":
                futures = [
                    executor.submit(
                        encode_panel_worker,
                        selected_panels_gpu,
                        local_idx,
                        panel_idx,
                        titles,
                        out_dir,
                        args.slug,
                        vmin,
                        scale,
                        args,
                    )
                    for local_idx, panel_idx in enumerate(PANELS)
                ]
            else:
                futures = [
                    executor.submit(
                        encode_panel_worker_from_cpu,
                        panels,
                        panel_idx,
                        titles,
                        out_dir,
                        args.slug,
                        vmin,
                        scale,
                        args,
                    )
                    for panel_idx in PANELS
                ]
            for future in as_completed(futures):
                filename, stats = future.result()
                speedup = stats.fps / baseline if baseline > 0 else None
                speedup_text = f", {speedup:.1f}x CPU baseline" if speedup else ""
                print(
                    f"[{args.slug}] wrote {filename}: {stats.frames} frames, "
                    f"{height}x{width}, {stats.fps:.2f} fps NVENC{speedup_text}, "
                    f"{stats.output_bytes / 1e6:.1f} MB, max sessions={stats.max_nvenc_sessions}"
                )
                all_stats[filename] = stats.__dict__
        profile["parallel_encode_wall_seconds"] = time.perf_counter() - encode_wall_start
        if args.panel_transfer == "preload":
            del selected_panels_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        sequential_preload_total = 0.0
        sequential_wall_start = time.perf_counter()
        for panel_idx in PANELS:
            title = titles[panel_idx]
            name = panel_name(title, panel_idx)
            out_path = out_dir / f"fig2_{args.slug}_{name}.mp4"
            print(f"[{args.slug}] encoding {out_path.name}: {title}")
            panel_preload_start = time.perf_counter()
            panel_gpu = cp.asarray(panels[panel_idx])
            sequential_preload_total += time.perf_counter() - panel_preload_start
            if args.legacy_cupy_pipeline:
                stats = encode_gray_stack(panel_gpu, title, out_path, vmin, scale, args)
            else:
                stats = encode_gray_stack_fused(panel_gpu, title, out_path, vmin, scale, args)
            del panel_gpu
            cp.get_default_memory_pool().free_all_blocks()
            speedup = stats.fps / baseline if baseline > 0 else None
            speedup_text = f", {speedup:.1f}x CPU baseline" if speedup else ""
            print(
                f"[{args.slug}] wrote {out_path.name}: {stats.frames} frames, "
                f"{height}x{width}, {stats.fps:.2f} fps NVENC{speedup_text}, "
                f"{stats.output_bytes / 1e6:.1f} MB, max sessions={stats.max_nvenc_sessions}"
            )
            all_stats[out_path.name] = stats.__dict__
        profile["panel_preload_to_gpu_seconds"] = sequential_preload_total
        profile["sequential_encode_wall_seconds"] = time.perf_counter() - sequential_wall_start

    if args.tiled:
        out_path = out_dir / f"fig2_{args.slug}_tiled_3x3.mp4"
        print(f"[{args.slug}] encoding tiled layout {out_path.name}")
        panels_gpu = cp.asarray(panels)
        stats = encode_tiled(panels_gpu, titles, out_path, vmin, scale, args)
        del panels_gpu
        cp.get_default_memory_pool().free_all_blocks()
        print(
            f"[{args.slug}] wrote {out_path.name}: {stats.frames} frames, "
            f"{height * 3}x{width * 3}, {stats.fps:.2f} fps NVENC, "
            f"{stats.output_bytes / 1e6:.1f} MB, max sessions={stats.max_nvenc_sessions}"
        )
        all_stats[out_path.name] = stats.__dict__

    profile["total_wall_seconds"] = time.perf_counter() - total_start
    profile["sum_panel_gpu_encode_seconds"] = sum(stat["seconds"] for stat in all_stats.values())
    profile["sum_panel_transfer_seconds"] = sum(stat["transfer_seconds"] for stat in all_stats.values())
    profile["sum_mux_seconds"] = sum(stat["mux_seconds"] for stat in all_stats.values())
    profile["sum_validate_seconds"] = sum(stat["validate_seconds"] for stat in all_stats.values())
    stats_path = out_dir / f"fig2_{args.slug}_gpu_encode_stats.json"
    stats_path.write_text(
        json.dumps({"profile": profile, "outputs": all_stats}, indent=2),
        encoding="utf-8",
    )
    print(f"[{args.slug}] DONE - {len(all_stats)} movie(s), stats: {stats_path}")


if __name__ == "__main__":
    main()
