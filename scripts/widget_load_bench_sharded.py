#!/usr/bin/env python3
"""Local-only sharded multi-GPU ``quantem.gpu.io.load`` benchmark.

Use this when a scientist wants to browse many real 4D-STEM masters from one
folder and keep them split across multiple NVIDIA GPUs. The benchmark reports
disk layout, GPU placement, cold/warm load time, and resident memory. It does
not belong in normal CI because it requires private lab data and CUDA hardware.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

GiB = 1 << 30


def _default_glob() -> str:
    return os.environ.get(
        "QUANTEM_WIDGET_BENCH_MASTERS_GLOB",
        os.environ.get("QUANTEM_BENCH_MASTER_GLOB", "data/**/*_master.h5"),
    )


def _parse_devices(raw: str) -> list[int]:
    devices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("expected at least one device, for example 0,1")
    return devices


def _data_files(masters: list[str]) -> list[str]:
    out: list[str] = []
    for master in masters:
        stem = os.path.basename(master).replace("_master.h5", "")
        out.extend(glob.glob(os.path.join(os.path.dirname(master), f"{stem}_data_*.h5")))
    return out


def _expand_master_globs(raw: str) -> list[str]:
    masters: list[str] = []
    for pattern in raw.split(os.pathsep):
        pattern = pattern.strip()
        if pattern:
            masters.extend(glob.glob(pattern))
    return list(dict.fromkeys(masters))


def _evict(masters: list[str]) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    for path in _data_files(masters):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _masters(args: argparse.Namespace) -> list[str]:
    masters = _expand_master_globs(args.masters_glob)
    if args.max_masters:
        masters = masters[: args.max_masters]
    if not masters:
        raise SystemExit(
            "No real 4D-STEM master files found. Pass --masters-glob or set "
            "QUANTEM_WIDGET_BENCH_MASTERS_GLOB."
        )
    return masters


def _build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    masters = _masters(args)
    counts = [int(item.strip()) for item in args.counts.split(",") if item.strip()]
    cases = []
    for count in counts:
        if len(masters) < count:
            continue
        for det_bin in args.det_bins:
            cases.append(
                {
                    "label": f"{count} masters det_bin={det_bin} across GPUs {args.devices}",
                    "masters": masters[:count],
                    "det_bin": det_bin,
                }
            )
    if not cases:
        raise SystemExit("No benchmark cases matched the discovered master count.")
    return cases[: args.max_cases] if args.max_cases else cases


def _free(devices: list[int]) -> None:
    import cupy as cp

    gc.collect()
    for device in devices:
        with cp.cuda.Device(device):
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()


def _sync(payload: dict[int, Any]) -> None:
    import cupy as cp

    for device in payload:
        with cp.cuda.Device(int(device)):
            cp.cuda.Device().synchronize()


def _hash(payload: dict[int, Any]) -> int:
    import cupy as cp

    total = 0
    for device, array in payload.items():
        with cp.cuda.Device(int(device)):
            total += int(array.sum(dtype=cp.uint64).get())
    return total


def _describe(payload: dict[int, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    shards = []
    total = 0
    for device, array in sorted(payload.items()):
        nbytes = int(array.nbytes)
        total += nbytes
        shards.append(
            {
                "device": int(device),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "gib": nbytes / GiB,
                "file_indices": list(metadata.get("shard_order", {}).get(device, [])),
            }
        )
    return {
        "total_gib": total / GiB,
        "shards": shards,
        "device_map": metadata.get("device_map", {}),
        "shard_order": metadata.get("shard_order", {}),
    }


def _timed_load(case: dict[str, Any], args: argparse.Namespace) -> tuple[float, Any, int | None]:
    from quantem.gpu.io import load

    kwargs: dict[str, Any] = {
        "det_bin": int(case["det_bin"]),
        "devices": args.devices,
        "verbose": False,
    }
    if args.dtype == "u8":
        kwargs["dtype"] = "u8"
    t0 = time.perf_counter()
    result = load(case["masters"], **kwargs)
    payload = result.data
    _sync(payload)
    wall = time.perf_counter() - t0
    digest = None if args.skip_parity else _hash(payload)
    return wall, result, digest


def _run_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import cupy as cp

    _free(args.devices)
    try:
        _, result, digest = _timed_load(case, args)
        description = _describe(result.data, result.metadata)
        del result
        _free(args.devices)
    except cp.cuda.memory.OutOfMemoryError as exc:
        _free(args.devices)
        return {"oom": True, "error": str(exc)[:160]}

    _evict(case["masters"])
    _free(args.devices)
    cold, result, cold_digest = _timed_load(case, args)
    if digest is not None and cold_digest != digest:
        raise AssertionError("PARITY DRIFT (cold)")
    del result
    _free(args.devices)

    warm_runs = []
    for _ in range(args.warm_runs):
        warm, result, warm_digest = _timed_load(case, args)
        if digest is not None and warm_digest != digest:
            raise AssertionError("PARITY DRIFT (warm)")
        warm_runs.append(warm)
        del result
        _free(args.devices)

    return {
        "oom": False,
        "cold_s": cold,
        "warm_s": min(warm_runs) if warm_runs else None,
        "warm_mean_s": sum(warm_runs) / len(warm_runs) if warm_runs else None,
        **description,
    }


def _disk_layout(masters: list[str], devices: list[int]) -> dict[str, Any]:
    from quantem.gpu.io.load import _assign_indices_to_devices

    groups: dict[str, list[str]] = {}
    for master in masters:
        try:
            disk = str(os.stat(master).st_dev)
        except OSError:
            disk = "unavailable"
        groups.setdefault(disk, []).append(master)
    return {
        "disk_count": len(groups),
        "disks": {disk: len(paths) for disk, paths in groups.items()},
        "assigned_indices": _assign_indices_to_devices(masters, devices),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--masters-glob",
        default=_default_glob(),
        help="Glob or os.pathsep-separated globs for private real *_master.h5 files.",
    )
    parser.add_argument("--devices", type=_parse_devices, default=[0, 1], help="Comma-separated CUDA devices.")
    parser.add_argument("--counts", default="4,8,16", help="Comma-separated master counts to benchmark.")
    parser.add_argument("--det-bin", dest="det_bins", type=int, action="append", default=None)
    parser.add_argument("--dtype", choices=["u16", "u8"], default="u16")
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--max-masters", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--skip-parity", action="store_true", help="Skip post-load uint64 hashes.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/quantem-widget-load-bench/load_bench_sharded.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--case", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.det_bins is None:
        args.det_bins = [1, 2, 4]
    return args


def _format_table(
    rows: list[tuple[str, dict[str, Any] | None]],
    layout: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# quantem.gpu.io.load sharded multi-GPU benchmark",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Real private data; artifacts stay local.",
        f"Devices: `{args.devices}`. Dtype: `{args.dtype}`. Disk groups: `{layout['disks']}`.",
        f"Cold = backing chunk files advised out of page cache. Warm = min of {args.warm_runs} hot run(s).",
        "",
        "| case | cold (s) | warm (s) | total GiB | shard summary |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for label, result in rows:
        if result is None:
            lines.append(f"| {label} | ERR | - | - | - |")
        elif result.get("oom"):
            lines.append(f"| {label} | OOM | - | - | {result.get('error', '')} |")
        else:
            warm = result.get("warm_s")
            warm_text = "-" if warm is None else f"{warm:.3f}"
            shards = ", ".join(
                f"gpu{row['device']}:{len(row['file_indices'])} files/{row['gib']:.1f} GiB"
                for row in result["shards"]
            )
            lines.append(
                f"| {label} | {result['cold_s']:.3f} | {warm_text} | "
                f"{result['total_gib']:.1f} | {shards} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    cases = _build_cases(args)
    if args.case is not None:
        result = _run_case(cases[args.case], args)
        print("RESULT " + json.dumps({"label": cases[args.case]["label"], "result": result}), flush=True)
        return 0

    masters = _masters(args)
    layout = _disk_layout(masters, args.devices)
    rows: list[tuple[str, dict[str, Any] | None]] = []
    for idx, case in enumerate(cases):
        print(f"running: {case['label']}", flush=True)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), *sys.argv[1:], "--case", str(idx)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        line = next((item for item in proc.stdout.splitlines() if item.startswith("RESULT ")), None)
        if line is None:
            print(proc.stdout[-1200:])
            rows.append((case["label"], None))
            continue
        payload = json.loads(line[len("RESULT "):])
        result = payload["result"]
        if result.get("oom"):
            print("  OOM")
        else:
            print(f"  cold={result['cold_s']:.3f}s warm={result.get('warm_s')} total={result['total_gib']:.1f}GiB")
        rows.append((case["label"], result))

    table = _format_table(rows, layout, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
