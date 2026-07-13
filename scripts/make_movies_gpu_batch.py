#!/usr/bin/env python3
"""Batch full-resolution denoise movie encodes across multiple GPUs.

The batch runner keeps one dataset job active per physical GPU.  Each child job
uses ``CUDA_VISIBLE_DEVICES=<gpu>`` and calls ``make_movies_gpu.py`` with
``--gpu-id 0`` so the single-dataset script sees the selected physical GPU as
its local device 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import subprocess
import sys
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIR = Path("/home/owner/publications/denoise-paper/build/datasets/fig2_3x3_fullres")
OUT_ROOT = Path("/home/owner/publications/denoise-paper/figures/movies")
CUDA_PYTHON = Path("/home/owner/miniforge3/envs/cuda-env/bin/python")


@dataclass
class BatchJobResult:
    slug: str
    gpu: str
    parallel_panels: int
    returncode: int
    wall_seconds: float
    log_path: Path
    stats_path: Path
    stats: dict[str, Any] | None
    error: str | None = None


@dataclass(frozen=True)
class GpuSlot:
    gpu: str
    parallel_panels: int


def parse_gpu_list(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise argparse.ArgumentTypeError("at least one GPU id is required")
    return gpus


def parse_gpu_slots(value: str) -> list[GpuSlot]:
    slots: list[GpuSlot] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            gpu, parallel_panels_text = item.split(":", maxsplit=1)
            try:
                parallel_panels = int(parallel_panels_text)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid slot {item!r}") from exc
        else:
            gpu = item
            parallel_panels = 0
        if not gpu:
            raise argparse.ArgumentTypeError(f"invalid slot {item!r}")
        slots.append(GpuSlot(gpu=gpu, parallel_panels=parallel_panels))
    if not slots:
        raise argparse.ArgumentTypeError("at least one GPU slot is required")
    return slots


def discover_slugs(cache_dir: Path) -> list[str]:
    slugs = sorted(path.stem for path in cache_dir.glob("*.npz"))
    if not slugs:
        raise SystemExit(f"no .npz cache files found in {cache_dir}")
    return slugs


def frame_count_for_slug(cache_dir: Path, slug: str) -> int:
    cache_path = cache_dir / f"{slug}.npz"
    with zipfile.ZipFile(cache_path) as archive:
        info = archive.getinfo("panels.npy")
    with cache_path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        if len(header) != 30:
            raise RuntimeError(f"{cache_path}:panels.npy has a truncated ZIP local header")
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
            raise RuntimeError(f"{cache_path}:panels.npy is not an uncompressed ZIP member")
        handle.seek(info.header_offset + 30 + filename_length + extra_length)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _fortran_order, _dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _fortran_order, _dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise RuntimeError(f"{cache_path}:panels.npy uses unsupported NPY version {version}")
    return int(shape[1])


def order_slugs(args: argparse.Namespace, slugs: list[str]) -> list[str]:
    if args.schedule == "input":
        return slugs
    if args.schedule == "largest-first":
        return sorted(slugs, key=lambda slug: frame_count_for_slug(args.cache_dir, slug), reverse=True)
    raise ValueError(f"unknown schedule {args.schedule!r}")


def bool_child_args(name: str, enabled: bool) -> list[str]:
    return [f"--{name}" if enabled else f"--no-{name}"]


def build_child_command(args: argparse.Namespace, slug: str, parallel_panels: int) -> list[str]:
    cmd = [
        str(args.python),
        "-u",
        str(args.script),
        slug,
        "--cache-dir",
        str(args.cache_dir),
        "--out-root",
        str(args.out_root),
        "--gpu-id",
        "0",
        "--fps",
        str(args.fps),
        "--codec",
        args.codec,
        "--load-mode",
        args.load_mode,
        "--qp",
        str(args.qp),
        "--preset",
        args.preset,
        "--mux-mode",
        args.mux_mode,
        "--parallel-panels",
        str(parallel_panels),
        "--nv12-ring",
        str(args.nv12_ring),
        "--panel-transfer",
        args.panel_transfer,
        "--contrast-cache",
        args.contrast_cache,
    ]
    cmd.extend(bool_child_args("labels", args.labels))
    cmd.extend(bool_child_args("validate", args.validate))
    if args.tiled:
        cmd.append("--tiled")
    if args.no_assert_nvenc:
        cmd.append("--no-assert-nvenc")
    if args.legacy_cupy_pipeline:
        cmd.append("--legacy-cupy-pipeline")
    for extra_arg in args.extra_arg:
        cmd.append(extra_arg)
    return cmd


def load_stats(stats_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not stats_path.exists():
        return None, f"stats file not found: {stats_path}"
    try:
        return json.loads(stats_path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"could not parse stats JSON {stats_path}: {exc}"


def run_job(
    args: argparse.Namespace,
    slug: str,
    slot: GpuSlot,
    log_dir: Path,
) -> BatchJobResult:
    gpu = slot.gpu
    parallel_panels = slot.parallel_panels or args.parallel_panels
    stats_path = args.out_root / slug / f"fig2_{slug}_gpu_encode_stats.json"
    log_path = log_dir / f"{slug}_gpu{gpu}_p{parallel_panels}.log"
    cmd = build_child_command(args, slug, parallel_panels)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    print(f"[batch] start slug={slug} gpu={gpu} parallel_panels={parallel_panels}: {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    error: str | None = None
    returncode = 1
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}\n")
        log.flush()
        if args.dry_run:
            returncode = 0
        else:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[gpu {gpu} {slug}] {line}", end="", flush=True)
            returncode = process.wait()
            if returncode != 0:
                error = f"child process exited with {returncode}"

    wall_seconds = time.perf_counter() - start
    stats, stats_error = load_stats(stats_path) if returncode == 0 and not args.dry_run else (None, None)
    if stats_error is not None:
        error = stats_error
    print(
        f"[batch] done slug={slug} gpu={gpu} rc={returncode} wall={wall_seconds:.3f}s "
        f"log={log_path}",
        flush=True,
    )
    return BatchJobResult(
        slug=slug,
        gpu=gpu,
        parallel_panels=parallel_panels,
        returncode=returncode,
        wall_seconds=wall_seconds,
        log_path=log_path,
        stats_path=stats_path,
        stats=stats,
        error=error,
    )


def summarize_result(result: BatchJobResult) -> dict[str, Any]:
    profile = (result.stats or {}).get("profile", {})
    outputs = (result.stats or {}).get("outputs", {})
    output_bytes = sum(int(item.get("output_bytes", 0)) for item in outputs.values())
    max_nvenc_sessions = max(
        (int(item.get("max_nvenc_sessions", 0)) for item in outputs.values()),
        default=0,
    )
    return {
        "slug": result.slug,
        "gpu": result.gpu,
        "parallel_panels": result.parallel_panels,
        "returncode": result.returncode,
        "batch_wall_seconds": result.wall_seconds,
        "child_total_wall_seconds": profile.get("total_wall_seconds"),
        "cache_open_seconds": profile.get("cache_open_seconds"),
        "raw_percentile_seconds": profile.get("raw_percentile_seconds"),
        "panel_preload_to_gpu_seconds": profile.get("panel_preload_to_gpu_seconds"),
        "sum_panel_transfer_seconds": profile.get("sum_panel_transfer_seconds"),
        "panel_transfer_mode": profile.get("panel_transfer_mode"),
        "parallel_encode_wall_seconds": profile.get("parallel_encode_wall_seconds"),
        "sum_panel_gpu_encode_seconds": profile.get("sum_panel_gpu_encode_seconds"),
        "sum_mux_seconds": profile.get("sum_mux_seconds"),
        "sum_validate_seconds": profile.get("sum_validate_seconds"),
        "outputs": len(outputs),
        "output_mb": output_bytes / 1e6,
        "max_nvenc_sessions": max_nvenc_sessions,
        "stats_path": str(result.stats_path),
        "log_path": str(result.log_path),
        "error": result.error,
    }


def write_summary(summary_dir: Path, results: list[BatchJobResult]) -> tuple[Path, Path]:
    rows = [summarize_result(result) for result in results]
    json_path = summary_dir / "batch_summary.json"
    csv_path = summary_dir / "batch_summary.csv"
    json_path.write_text(
        json.dumps({"results": rows}, indent=2),
        encoding="utf-8",
    )
    fieldnames = list(rows[0]) if rows else ["slug"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slugs", nargs="*", help="Dataset slugs. Defaults to all .npz files in --cache-dir.")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=CUDA_PYTHON)
    parser.add_argument("--script", type=Path, default=Path(__file__).with_name("make_movies_gpu.py"))
    parser.add_argument("--gpus", type=parse_gpu_list, default=parse_gpu_list("0,1"))
    parser.add_argument(
        "--gpu-slots",
        type=parse_gpu_slots,
        default=None,
        help=(
            "Concurrent GPU slots, optionally with per-slot panel parallelism, "
            "for example 0:7,1:4,1:4. Defaults to one slot per --gpus entry."
        ),
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--codec", choices=["h264", "hevc", "av1"], default="h264")
    parser.add_argument("--load-mode", choices=["npz-memmap", "np-load"], default="npz-memmap")
    parser.add_argument("--qp", type=int, default=18)
    parser.add_argument("--preset", default="P3")
    parser.add_argument("--mux-mode", choices=["temp", "pipe"], default="temp")
    parser.add_argument("--parallel-panels", type=int, default=7)
    parser.add_argument("--nv12-ring", type=int, default=4)
    parser.add_argument("--panel-transfer", choices=["preload", "worker"], default="preload")
    parser.add_argument(
        "--contrast-cache",
        choices=["off", "read", "write", "read-write"],
        default="off",
    )
    parser.add_argument("--labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--no-assert-nvenc", action="store_true")
    parser.add_argument("--legacy-cupy-pipeline", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra argument passed to each child job.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--schedule",
        choices=["input", "largest-first"],
        default="input",
        help="Order jobs as provided/discovered, or sort by frame count descending using metadata only.",
    )
    return parser.parse_args()


def main() -> None:
    batch_start = time.perf_counter()
    args = parse_args()
    slugs = order_slugs(args, args.slugs or discover_slugs(args.cache_dir))
    slots = args.gpu_slots or [GpuSlot(gpu=gpu, parallel_panels=args.parallel_panels) for gpu in args.gpus]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_dir = args.summary_dir or args.out_root / "_batch_profiles" / timestamp
    summary_dir.mkdir(parents=True, exist_ok=True)

    print(f"[batch] slugs={', '.join(slugs)}", flush=True)
    slot_text = ",".join(f"{slot.gpu}:p{slot.parallel_panels or args.parallel_panels}" for slot in slots)
    print(f"[batch] slots={slot_text} summary_dir={summary_dir}", flush=True)

    results: list[BatchJobResult] = []
    slug_iter = iter(slugs)
    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        future_to_slot = {}
        for slot in slots:
            try:
                slug = next(slug_iter)
            except StopIteration:
                break
            future_to_slot[executor.submit(run_job, args, slug, slot, summary_dir)] = slot

        while future_to_slot:
            done, _ = wait(future_to_slot, return_when=FIRST_COMPLETED)
            for future in done:
                slot = future_to_slot.pop(future)
                results.append(future.result())
                try:
                    slug = next(slug_iter)
                except StopIteration:
                    continue
                future_to_slot[executor.submit(run_job, args, slug, slot, summary_dir)] = slot

    results.sort(key=lambda item: slugs.index(item.slug))
    json_path, csv_path = write_summary(summary_dir, results)
    failed = [result for result in results if result.returncode != 0 or result.error]
    print(f"[batch] total wall={time.perf_counter() - batch_start:.3f}s", flush=True)
    print(f"[batch] summary json={json_path}", flush=True)
    print(f"[batch] summary csv={csv_path}", flush=True)
    if failed:
        for result in failed:
            print(f"[batch] FAILED slug={result.slug} gpu={result.gpu}: {result.error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
