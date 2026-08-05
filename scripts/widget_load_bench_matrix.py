#!/usr/bin/env python3
"""Local-only real-data ``quantem.gpu.io.load`` benchmark matrix.

This widget-workflow signoff measures the canonical ``quantem.gpu.io`` loader.
It is not a normal CI test: it expects private ``*_master.h5`` files on the
workstation and writes report artifacts outside the repo by default.
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


def _data_files(masters: str | list[str]) -> list[str]:
    paths = masters if isinstance(masters, list) else [masters]
    out: list[str] = []
    for master in paths:
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


def _evict(masters: str | list[str]) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    for path in _data_files(masters):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    masters = _expand_master_globs(args.masters_glob)
    if args.one_master:
        one = args.one_master
    elif masters:
        one = masters[0]
    else:
        raise SystemExit(
            "No real 4D-STEM master files found. Pass --masters-glob or set "
            "QUANTEM_WIDGET_BENCH_MASTERS_GLOB."
        )
    cases: list[dict[str, Any]] = [
        {"label": "single no-bin exact uint16", "masters": one, "det_bin": 1, "dtype": "u16"},
        {"label": "single no-bin browse uint8", "masters": one, "det_bin": 1, "dtype": "u8"},
        {"label": "single det_bin=2 exact uint16", "masters": one, "det_bin": 2, "dtype": "u16"},
        {"label": "single det_bin=4 exact uint16", "masters": one, "det_bin": 4, "dtype": "u16"},
    ]
    for count, det_bin in [(4, 2), (8, 4), (16, 4)]:
        if len(masters) >= count:
            cases.append(
                {
                    "label": f"{count} masters det_bin={det_bin} stacked on one GPU",
                    "masters": masters[:count],
                    "det_bin": det_bin,
                    "dtype": "u16",
                }
            )
    return cases[: args.max_cases] if args.max_cases else cases


def _free() -> None:
    import cupy as cp

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _sync_payload(payload: Any) -> None:
    import cupy as cp

    if isinstance(payload, dict):
        for device in payload:
            with cp.cuda.Device(int(device)):
                cp.cuda.Device().synchronize()
    else:
        cp.cuda.Device().synchronize()


def _payload(result: Any) -> Any:
    return result.data if hasattr(result, "data") else result


def _nbytes(payload: Any) -> int:
    if isinstance(payload, dict):
        return sum(int(getattr(array, "nbytes", 0)) for array in payload.values())
    return int(getattr(payload, "nbytes", 0))


def _shape(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(device): list(getattr(array, "shape", ())) for device, array in payload.items()}
    return list(getattr(payload, "shape", ()))


def _hash_payload(payload: Any) -> int:
    import cupy as cp

    if isinstance(payload, dict):
        total = 0
        for device, array in payload.items():
            with cp.cuda.Device(int(device)):
                total += int(array.sum(dtype=cp.uint64).get())
        return total
    return int(payload.sum(dtype=cp.uint64).get())


def _timed_load(case: dict[str, Any], *, skip_parity: bool) -> tuple[float, Any, int | None]:
    from quantem.gpu.io import load

    kwargs: dict[str, Any] = {"det_bin": int(case["det_bin"]), "verbose": False}
    dtype = str(case.get("dtype") or "")
    if dtype == "u8":
        kwargs["dtype"] = "u8"
    t0 = time.perf_counter()
    result = load(case["masters"], **kwargs)
    payload = _payload(result)
    _sync_payload(payload)
    wall = time.perf_counter() - t0
    digest = None if skip_parity else _hash_payload(payload)
    return wall, payload, digest


def _run_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import cupy as cp

    _free()
    try:
        _, payload, digest = _timed_load(case, skip_parity=args.skip_parity)
        shape = _shape(payload)
        final_gib = _nbytes(payload) / GiB
        del payload
        _free()
    except cp.cuda.memory.OutOfMemoryError as exc:
        _free()
        return {"oom": True, "error": str(exc)[:160]}

    _evict(case["masters"])
    _free()
    cold, payload, cold_digest = _timed_load(case, skip_parity=args.skip_parity)
    if digest is not None and cold_digest != digest:
        raise AssertionError("PARITY DRIFT (cold)")
    del payload
    _free()

    warm_runs = []
    for _ in range(args.warm_runs):
        warm, payload, warm_digest = _timed_load(case, skip_parity=args.skip_parity)
        if digest is not None and warm_digest != digest:
            raise AssertionError("PARITY DRIFT (warm)")
        warm_runs.append(warm)
        del payload
        _free()

    return {
        "oom": False,
        "cold_s": cold,
        "warm_s": min(warm_runs) if warm_runs else None,
        "warm_mean_s": sum(warm_runs) / len(warm_runs) if warm_runs else None,
        "final_gib": final_gib,
        "shape": shape,
        "hash": digest,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--masters-glob",
        default=_default_glob(),
        help="Glob or os.pathsep-separated globs for private real *_master.h5 files.",
    )
    parser.add_argument("--one-master", default=os.environ.get("QUANTEM_WIDGET_BENCH_4DSTEM_MASTER"))
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=0, help="Limit cases for quick profiling.")
    parser.add_argument("--skip-parity", action="store_true", help="Skip post-load uint64 hashes.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/quantem-widget-load-bench/load_bench_matrix.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--case", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _format_table(rows: list[tuple[str, dict[str, Any] | None]], args: argparse.Namespace) -> str:
    lines = [
        "# quantem.gpu.io.load benchmark matrix",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Real private data; artifacts stay local.",
        f"Cold = backing chunk files advised out of page cache. Warm = min of {args.warm_runs} hot run(s).",
        "Parity hash is computed after the timer unless `--skip-parity` is used.",
        "",
        "| case | cold (s) | warm (s) | final GiB | shape |",
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
            lines.append(
                f"| {label} | {result['cold_s']:.3f} | {warm_text} | "
                f"{result['final_gib']:.1f} | `{result['shape']}` |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    cases = _build_cases(args)
    if args.case is not None:
        result = _run_case(cases[args.case], args)
        print("RESULT " + json.dumps({"label": cases[args.case]["label"], "result": result}), flush=True)
        return 0

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
            print(f"  cold={result['cold_s']:.3f}s warm={result.get('warm_s')} final={result['final_gib']:.1f}GiB")
        rows.append((case["label"], result))

    table = _format_table(rows, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
