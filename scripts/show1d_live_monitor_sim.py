#!/usr/bin/env python3
"""Write a ptychography-style Show1D monitor and optional review export.

The script simulates the file contract used by an overnight joint iterative
ptychography run: every JSONL event contains optimizer losses, optional metrics,
and object/probe checkpoint images.  It is intentionally lightweight so it can
be run in CI, a notebook terminal, or during widget design review.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import shutil
import time

import numpy as np

from quantem.widget import Show1D


def _ducky_object(size: int, *, phase: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    body = np.exp(-((x + 0.08 * np.sin(phase)) ** 2 + (y + 0.05) ** 2) * 4.5)
    head = 0.72 * np.exp(-((x - 0.32) ** 2 + (y + 0.18) ** 2) * 22)
    beak = 0.42 * np.exp(-((x - 0.62) ** 2 + (y + 0.18) ** 2) * 90)
    wing = -0.28 * np.exp(-((x + 0.18) ** 2 + (y - 0.05) ** 2) * 28)
    obj = body + head + beak + wing
    obj += 0.12 * np.sin(16 * x + phase) * np.cos(15 * y - phase)
    obj += rng.normal(0.0, noise, size=(size, size))
    return obj.astype(np.float32)


def _probe(size: int, *, width: float, astig: float, phase: float) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    probe = np.exp(-((x * (1 + astig)) ** 2 + (y * (1 - astig)) ** 2) / max(width, 1e-6))
    probe *= 1.0 + 0.08 * np.cos(10 * x + phase) * np.sin(9 * y)
    return probe.astype(np.float32)


def write_monitor(
    run_dir: pathlib.Path,
    *,
    iterations: int = 16,
    size: int = 96,
    snapshot_stride: int = 2,
    delay_s: float = 0.0,
) -> pathlib.Path:
    """Write a deterministic joint-ptychography monitor under ``run_dir``."""

    if run_dir.exists():
        shutil.rmtree(run_dir)
    snapshots = run_dir / "snapshots"
    snapshots.mkdir(parents=True)
    monitor = run_dir / "show1d_monitor.jsonl"
    rng = np.random.default_rng(7)
    lambdas = [0.0, 0.3, 1.0, 3.0, 10.0, 30.0]

    for iteration in range(iterations):
        t = iteration / max(1, iterations - 1)
        losses: dict[str, float | None] = {}
        metrics: dict[str, dict[str, float]] = {}
        snapshots_for_event: dict[str, str] = {}
        warnings: list[str] = []

        for lam in lambdas:
            label = f"lambda {lam:g}"
            base = 4.2 - 2.5 * (1 - math.exp(-3.0 * t))
            regularizer_bias = 0.35 * abs(math.log10(lam + 1.0) - math.log10(2.0))
            loss = base + regularizer_bias + 0.04 * math.sin(iteration * 0.9 + lam)
            rmse = 0.7 - 0.45 * t + 0.05 * abs(math.log10(lam + 1.0) - 0.45)
            flicker = 0.12 + 0.05 * abs(math.log10(lam + 1.0) - 0.45)

            if lam == 10.0 and iteration > iterations // 2:
                loss += 0.55 * (iteration - iterations // 2)
                flicker = 0.9
                if iteration == iterations // 2 + 1:
                    warnings.append("lambda 10 loss spike after probe update")
            if lam == 30.0 and iteration >= iterations - 2:
                loss = None
                rmse = 1.25
                warnings.append("lambda 30 produced non-finite loss")

            losses[label] = loss
            metrics[label] = {
                "rmse": float(rmse),
                "flicker": float(flicker),
                "object_quality": float(0.6 - abs(lam - 1.0) * 0.006),
                "probe_quality": float(0.5 - abs(lam - 1.0) * 0.004),
            }

            if iteration % snapshot_stride == 0 or iteration == iterations - 1:
                suffix = str(lam).replace(".", "p")
                object_path = snapshots / f"lambda_{suffix}_i{iteration:03d}_object.npy"
                probe_path = snapshots / f"lambda_{suffix}_i{iteration:03d}_probe.npy"
                noise = 0.22 * (1 - t) + 0.04 + 0.012 * abs(lam - 1.0)
                obj = _ducky_object(size, phase=0.45 * iteration + 0.1 * lam, noise=noise, rng=rng)
                if lam == 30.0 and iteration >= iterations - 2:
                    obj = np.zeros_like(obj)
                np.save(object_path, obj)
                np.save(probe_path, _probe(size, width=0.16 + 0.01 * lam, astig=0.015 * lam, phase=iteration))
                key = f"lambda_{lam:g}"
                snapshots_for_event[key] = str(object_path.relative_to(run_dir))
                snapshots_for_event[f"{key}_probe"] = str(probe_path.relative_to(run_dir))

        event = {
            "iteration": iteration,
            "losses": losses,
            "metrics": metrics,
            "snapshots": snapshots_for_event,
            "warnings": warnings,
        }
        if iteration == iterations - 1:
            event.update(
                {
                    "starred": ["lambda 1"],
                    "hidden": ["lambda 30"],
                    "notes": {"lambda 1": "best overnight candidate", "lambda 30": "bad start / collapse"},
                    "tags": {"lambda 1": ["best lambda"], "lambda 30": ["bad start"]},
                }
            )
        Show1D.append_monitor_event(monitor, event)
        if delay_s > 0:
            time.sleep(delay_s)
    return monitor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=pathlib.Path, default=pathlib.Path("/tmp/quantem-show1d-live-monitor"))
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--snapshot-stride", type=int, default=2)
    parser.add_argument("--delay-s", type=float, default=0.0)
    parser.add_argument("--export-html", type=pathlib.Path, default=None)
    parser.add_argument("--export-summary", type=pathlib.Path, default=None)
    args = parser.parse_args()

    monitor = write_monitor(
        args.run_dir,
        iterations=args.iterations,
        size=args.size,
        snapshot_stride=max(1, args.snapshot_stride),
        delay_s=max(0.0, args.delay_s),
    )
    print(monitor)

    if args.export_html or args.export_summary:
        widget = Show1D.from_monitor_file(
            monitor,
            title="Live joint iterative ptychography monitor",
            show_review=True,
            show_stats=False,
            snapshot_columns=4,
            snapshot_thumbnail_size=44,
            side_panel_width_px=760,
            image_cmap="viridis",
            snapshot_contrast_preset="1-99",
            trial_sort_key="final_loss",
        )
        widget.goto_snapshot(max(0, widget.n_snapshot_groups - 1))
        if args.export_summary:
            widget.export_run_summary(args.export_summary)
            print(args.export_summary)
        if args.export_html:
            widget.export_html(args.export_html)
            print(args.export_html)


if __name__ == "__main__":
    main()
