#!/usr/bin/env python3
"""Run resumable real-data Show4DSTEM folder paging/cache endurance.

This is an opt-in, local-only signoff runner.  The controller waits for the
selected physical NVIDIA devices to become idle, then starts every case in a
fresh child process so ``CUDA_VISIBLE_DEVICES`` is fixed before Torch imports.
It writes an atomic live report throughout the run and never mutates source
data or terminates processes it did not create.

Actual Jupyter/browser evidence is a separate required gate.  This runner owns
the backend paging/cache/endurance matrix and records that browser gate as
pending until the companion live-Jupyter drive attaches its artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Sequence


SCHEMA_VERSION = 1
DEFAULT_BLOCK_PATTERNS = (
    "overnight_ml_calibration_campaign.py",
    "overnight_zoo_campaign.py",
    "run_noiseless_block.py",
    "run_framewise_block.py",
    "live ptycho",
    "quantem.live.cli.ptycho",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_ready(value.tolist())
        except Exception:
            pass
    return str(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_ready(value), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_text(command: Sequence[str]) -> str:
    return subprocess.check_output(
        list(command),
        text=True,
        stderr=subprocess.STDOUT,
        timeout=30,
    ).strip()


def _git_snapshot(repo: Path) -> dict[str, Any]:
    try:
        commit = _run_text(["git", "-C", str(repo), "rev-parse", "HEAD"])
        status = _run_text(["git", "-C", str(repo), "status", "--short"])
        diff = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--binary"],
            stderr=subprocess.STDOUT,
        )
        return {
            "commit": commit,
            "dirty": bool(status),
            "status": status.splitlines(),
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _filesystem_snapshot(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    result: dict[str, Any] = {"path": str(resolved)}
    try:
        stat = resolved.stat()
        usage = shutil.disk_usage(resolved)
        result.update(
            {
                "device": int(stat.st_dev),
                "free_bytes": int(usage.free),
                "total_bytes": int(usage.total),
            }
        )
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    try:
        result["findmnt"] = _run_text(
            ["findmnt", "-T", str(resolved), "-no", "SOURCE,FSTYPE,TARGET"]
        )
    except Exception as exc:
        result["findmnt_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _gpu_snapshot() -> dict[str, Any]:
    fields = (
        "index,uuid,pci.bus_id,name,driver_version,memory.total,memory.used,"
        "memory.free,utilization.gpu"
    )
    result: dict[str, Any] = {"captured_at": _utc_now(), "gpus": [], "apps": []}
    try:
        raw = _run_text(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ]
        )
        for line in raw.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 9:
                continue
            result["gpus"].append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "pci_bus_id": parts[2],
                    "name": parts[3],
                    "driver_version": parts[4],
                    "total_mib": int(parts[5]),
                    "used_mib": int(parts[6]),
                    "free_mib": int(parts[7]),
                    "utilization_pct": int(parts[8]),
                }
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        raw = _run_text(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except Exception:
        raw = ""
    apps: list[dict[str, Any]] = []
    pids: list[int] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        pids.append(pid)
        apps.append(
            {
                "gpu_uuid": parts[0],
                "pid": pid,
                "process_name": parts[2],
                "used_mib": None if parts[3] == "[N/A]" else int(parts[3]),
            }
        )
    commands: dict[int, str] = {}
    if pids:
        try:
            raw_ps = _run_text(
                ["ps", "-ww", "-o", "pid=,command=", "-p", ",".join(map(str, pids))]
            )
            for line in raw_ps.splitlines():
                pieces = line.strip().split(maxsplit=1)
                if not pieces:
                    continue
                commands[int(pieces[0])] = pieces[1] if len(pieces) > 1 else ""
        except Exception:
            pass
    for app in apps:
        app["command"] = commands.get(int(app["pid"]), "")
    result["apps"] = apps
    return result


def _idle_decision(
    snapshot: dict[str, Any],
    devices: Sequence[int],
    *,
    max_utilization: int,
    min_free_mib: int,
    block_patterns: Sequence[str],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    rows = {int(row["index"]): row for row in snapshot.get("gpus", [])}
    selected_uuids: set[str] = set()
    for device in devices:
        row = rows.get(int(device))
        if row is None:
            reasons.append(f"physical GPU {device} is not visible to nvidia-smi")
            continue
        selected_uuids.add(str(row["uuid"]))
        if int(row["utilization_pct"]) > int(max_utilization):
            reasons.append(
                f"GPU {device} utilization {row['utilization_pct']}% > {max_utilization}%"
            )
        if int(row["free_mib"]) < int(min_free_mib):
            reasons.append(
                f"GPU {device} free {row['free_mib']} MiB < {min_free_mib} MiB"
            )
    lowered_patterns = [token.lower() for token in block_patterns if token]
    for app in snapshot.get("apps", []):
        if str(app.get("gpu_uuid")) not in selected_uuids:
            continue
        command = str(app.get("command", ""))
        lowered = command.lower()
        matched = next((token for token in lowered_patterns if token in lowered), None)
        if matched:
            reasons.append(
                f"GPU campaign PID {app.get('pid')} matches {matched!r}: {command[:260]}"
            )
    return not reasons, reasons


class LiveReport:
    def __init__(self, artifact_dir: Path, initial: dict[str, Any]) -> None:
        self.artifact_dir = artifact_dir.resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.artifact_dir / "report.json"
        self.status_path = self.artifact_dir / "status.json"
        self.events_path = self.artifact_dir / "events.jsonl"
        self.gpu_path = self.artifact_dir / "gpu-telemetry.jsonl"
        self._lock = threading.RLock()
        previous: dict[str, Any] = {}
        if self.path.is_file():
            try:
                previous = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        self.data = {**initial}
        if previous.get("schema_version") == SCHEMA_VERSION:
            self.data["cases"] = list(previous.get("cases", []))
            self.data["errors"] = list(previous.get("errors", []))
            self.data["restart_count"] = int(previous.get("restart_count", 0)) + 1
        self.flush()

    def event(self, event: str, **fields: Any) -> None:
        record = {"time": _utc_now(), "event": event, **fields}
        _append_jsonl(self.events_path, record)

    def gpu(self, snapshot: dict[str, Any], *, phase: str) -> None:
        _append_jsonl(self.gpu_path, {"phase": phase, **snapshot})

    def update(self, **fields: Any) -> None:
        with self._lock:
            self.data.update(fields)
            self.data["heartbeat_at"] = _utc_now()
            self.flush()

    def append_case(self, case: dict[str, Any]) -> None:
        with self._lock:
            cases = [
                item for item in self.data.setdefault("cases", [])
                if item.get("id") != case.get("id")
            ]
            cases.append(case)
            self.data["cases"] = cases
            self.data["completed_cases"] = sum(
                item.get("status") == "pass" for item in cases
            )
            self.data["heartbeat_at"] = _utc_now()
            self.flush()

    def add_error(self, error: dict[str, Any]) -> None:
        with self._lock:
            self.data.setdefault("errors", []).append(error)
            self.flush()

    def completed(self, case_id: str) -> dict[str, Any] | None:
        for case in self.data.get("cases", []):
            if case.get("id") == case_id and case.get("status") == "pass":
                return case
        return None

    def flush(self) -> None:
        with self._lock:
            _atomic_json(self.path, self.data)
            _atomic_json(
                self.artifact_dir / "environment.json",
                self.data.get("environment", {}),
            )
            _atomic_json(
                self.artifact_dir / "gates.json",
                self.data.get("gates", []),
            )
            _atomic_json(
                self.artifact_dir / "manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.data.get("run_id"),
                    "status": self.data.get("status"),
                    "current_phase": self.data.get("current_phase"),
                    "heartbeat_at": self.data.get("heartbeat_at"),
                    "completed_cases": self.data.get("completed_cases", 0),
                    "active_pid": self.data.get("active_pid"),
                    "active_command": self.data.get("active_command"),
                    "report": str(self.path),
                },
            )
            _atomic_json(
                self.status_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": self.data.get("status"),
                    "current_phase": self.data.get("current_phase"),
                    "heartbeat_at": self.data.get("heartbeat_at"),
                    "completed_cases": self.data.get("completed_cases", 0),
                    "last_error": (
                        self.data.get("errors", [])[-1]
                        if self.data.get("errors")
                        else None
                    ),
                },
            )
            self._write_html()

    def _write_html(self) -> None:
        status = html.escape(str(self.data.get("status", "unknown")))
        phase = html.escape(str(self.data.get("current_phase", "")))
        heartbeat = html.escape(str(self.data.get("heartbeat_at", "")))
        rows = []
        for case in self.data.get("cases", []):
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(case.get('id', '')))}</td>"
                f"<td>{html.escape(str(case.get('status', '')))}</td>"
                f"<td>{html.escape(str(case.get('elapsed_seconds', '')))}</td>"
                f"<td><code>{html.escape(str(case.get('error', '')))}</code></td>"
                "</tr>"
            )
        raw = html.escape(json.dumps(_json_ready(self.data), indent=2))
        page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="refresh" content="30"><title>Show4DSTEM overnight signoff</title>
<style>body{{font:14px system-ui;margin:24px;color:#17212b}}.status{{font-weight:700}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3da;padding:6px;text-align:left}}
pre{{white-space:pre-wrap;background:#f5f7f9;padding:12px;border-radius:6px}}code{{font-size:12px}}</style></head>
<body><h1>Show4DSTEM folder paging/cache overnight</h1>
<p class="status">{status} · {phase}</p><p>Heartbeat: {heartbeat}</p>
<p>This live backend report refreshes every 30 seconds. Browser/Jupyter evidence is a separate gate.</p>
<table><thead><tr><th>Case</th><th>Status</th><th>Seconds</th><th>Error</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table><h2>Machine-readable report</h2>
<p><a href="report.json">report.json</a> · <a href="status.json">status.json</a> ·
<a href="events.jsonl">events.jsonl</a> · <a href="gpu-telemetry.jsonl">gpu-telemetry.jsonl</a></p>
<details><summary>Current report</summary><pre>{raw}</pre></details></body></html>"""
        temporary = self.artifact_dir / f".index.{os.getpid()}.tmp"
        temporary.write_text(page, encoding="utf-8")
        os.replace(temporary, self.artifact_dir / "index.html")


def _parse_devices(value: str) -> list[int]:
    devices = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("at least one GPU index is required")
    if len(set(devices)) != len(devices) or any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError(f"invalid unique GPU list: {value!r}")
    return devices


def _child_state(widget: Any) -> dict[str, Any]:
    data = getattr(widget, "_data", None)
    if data is None:
        return {"freed": True}
    plan = data.residency_plan() if callable(getattr(data, "residency_plan", None)) else {}
    page_devices = [str(item) for item in getattr(data, "_page_devices", [])]
    return _json_ready(
        {
            "n_frames": int(widget.n_frames),
            "page_idx": int(widget.compare_page_idx),
            "page_count": int(widget.compare_page_count),
            "page_generation": int(widget.compare_page_generation),
            "expected_indices": list(widget.compare_page_expected_indices),
            "panel_indices": list(widget.compare_panel_indices),
            "loaded_count": int(widget.compare_page_loaded_count),
            "cache_state": str(widget.compare_page_cache_state),
            "first_panel_ms": float(widget.compare_page_first_panel_ms),
            "first_fresh_ms": float(widget.compare_page_first_fresh_ms),
            "total_ms": float(widget.compare_page_total_ms),
            "watch_state": str(widget.folder_watch_state),
            "watch_detail": str(widget.folder_watch_detail),
            "page_error": str(getattr(widget, "_compare_page_last_error", "")),
            "loaded_indices": data.loaded_indices(),
            "vram_resident": data.vram_resident(),
            "resident_nbytes": int(data.resident_nbytes),
            "logical_nbytes": int(data.nbytes),
            "residency_plan": plan,
            "page_devices": page_devices,
            "preview_cache": widget.preview_cache_info,
            "warm_status": str(getattr(widget, "_compare_cache_warm_status", "")),
        }
    )


def _touch_page(widget: Any, page: int, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    widget.set_compare_page(int(page))
    widget.wait_for_compare_page(timeout=timeout)
    error = str(getattr(widget, "_compare_page_last_error", ""))
    if error:
        raise RuntimeError(error)
    state = _child_state(widget)
    state["requested_page"] = int(page)
    state["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return state


def _run_child(args: argparse.Namespace) -> int:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    result_path = args.result_path.resolve()
    result: dict[str, Any] = {
        "id": args.case_id,
        "status": "running",
        "case_kind": args.case_kind,
        "started_at": _utc_now(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "page_actions": [],
        "cycles_completed": 0,
        "errors": [],
    }
    _atomic_json(result_path, result)
    widget = None
    started = time.perf_counter()
    try:
        import torch

        from quantem.widget import Show4DSTEM

        logical_gpus = list(range(torch.cuda.device_count()))
        if not logical_gpus:
            raise RuntimeError(
                "No CUDA GPU is visible in the child; check CUDA_VISIBLE_DEVICES."
            )
        result["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "logical_devices": [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_bytes": int(
                        torch.cuda.get_device_properties(idx).total_memory
                    ),
                }
                for idx in logical_gpus
            ],
        }
        preview_enabled = args.case_kind != "cold"
        rebuild = args.case_kind == "populate"
        warm_cache = args.case_kind == "populate"
        build_started = time.perf_counter()
        widget = Show4DSTEM.from_folder(
            args.source,
            pattern=args.pattern,
            recursive=True,
            ready_only=True,
            gpus=logical_gpus,
            backend="cuda",
            page_budget="auto",
            page_max_vram_fraction=args.max_vram_fraction,
            det_bin=args.det_bin,
            dtype=args.dtype,
            view_mode="multiple",
            columns=args.columns,
            page_size=args.page_size,
            preload_all_if_fits=False,
            warm_cache=warm_cache,
            preview_cache=preview_enabled,
            preview_cache_dir=args.cache_dir,
            preview_cache_max_bytes=args.cache_max_bytes,
            rebuild_preview_cache=rebuild,
            watch=True,
            watch_interval=args.watch_interval,
            preload_initial_page=False,
            precompute_virtual_images=False,
            compare_dp_mode="selected",
            debug=True,
            verbose=False,
            title=f"Show4DSTEM overnight · {args.case_id}",
        )
        result["build_seconds"] = round(time.perf_counter() - build_started, 6)
        if int(widget.n_frames) < int(args.min_ready):
            raise RuntimeError(
                f"Only {widget.n_frames} ready masters; require {args.min_ready}."
            )
        result["initial_state"] = _child_state(widget)
        widget.wait_for_compare_page(timeout=args.page_timeout)
        page_count = max(1, int(widget.compare_page_count))
        canonical = [0, min(1, page_count - 1), page_count - 1, 0]

        while (
            result["cycles_completed"] < args.cycles
            or time.time() < args.deadline_epoch
        ):
            for page in canonical:
                result["page_actions"].append(
                    _touch_page(widget, page, args.page_timeout)
                )
                _atomic_json(result_path, result)

            if page_count >= 3:
                rapid_started = time.perf_counter()
                widget.set_compare_page(0)
                widget.set_compare_page(1)
                widget.set_compare_page(2)
                widget.wait_for_compare_page(timeout=args.page_timeout)
                if int(widget.compare_page_idx) != 2:
                    raise RuntimeError(
                        "Rapid 1→2→3 navigation finished on the wrong page."
                    )
                result["rapid_navigation"] = {
                    **_child_state(widget),
                    "elapsed_seconds": round(
                        time.perf_counter() - rapid_started, 6
                    ),
                }

            if int(widget.n_frames) >= 3:
                widget.star_compare_panel(2)
                widget.hide_compare_panel(1)
                hidden = list(widget.compare_hidden_panels)
                starred = list(widget.compare_starred_panels)
                widget.show_compare_panel(1)
                result["curation"] = {
                    "hidden_during": hidden,
                    "starred": starred,
                    "hidden_after_restore": list(widget.compare_hidden_panels),
                    "passed": 1 in hidden and 2 in starred,
                }
                if not result["curation"]["passed"]:
                    raise RuntimeError("Hide/star state did not persist through the cycle.")

            hashes: dict[str, str] = {}
            for mode in ("selected", "average", "selected"):
                widget.compare_dp_mode = mode
                hashes[mode] = hashlib.sha256(bytes(widget.frame_bytes)).hexdigest()
            result["diffraction_modes"] = hashes
            result["cycles_completed"] += 1
            result["last_state"] = _child_state(widget)
            _atomic_json(result_path, result)

        if warm_cache:
            warm_deadline = time.monotonic() + args.maintenance_timeout
            while (
                str(getattr(widget, "_compare_cache_warm_status", ""))
                not in {"ready", "failed", "stopped"}
                and time.monotonic() < warm_deadline
            ):
                time.sleep(0.25)
            result["warm_status"] = str(
                getattr(widget, "_compare_cache_warm_status", "")
            )
            if result["warm_status"] == "failed":
                raise RuntimeError("Persistent preview cache warming failed.")

        cache = getattr(widget, "_compare_preview_cache", None)
        if cache is not None:
            cache.flush()
        result["final_state"] = _child_state(widget)
        plan = result["final_state"].get("residency_plan", {})
        budget_values = [
            int(value) for value in plan.get("budget_bytes", {}).values()
        ]
        resident_nbytes = int(result["final_state"].get("resident_nbytes", 0))
        cache_info = result["final_state"].get("preview_cache", {})
        result["correctness"] = {
            "bounded_residency": (
                not budget_values or resident_nbytes <= sum(budget_values)
            ),
            "requested_dtype": args.dtype,
            "det_bin": args.det_bin,
            "watch_worker_alive": bool(
                getattr(widget, "_folder_watch_thread", None)
                and widget._folder_watch_thread.is_alive()
            ),
            "persistent_cache_hits": int(cache_info.get("hits", 0)),
            "persistent_cache_entries": int(cache_info.get("entries", 0)),
            "persistent_cache_writes": int(cache_info.get("writes", 0)),
        }
        if not result["correctness"]["bounded_residency"]:
            raise RuntimeError("Managed raw residency exceeded its CUDA byte budget.")
        if args.case_kind == "populate" and (
            result["correctness"]["persistent_cache_entries"] <= 0
            or result["correctness"]["persistent_cache_writes"] <= 0
        ):
            raise RuntimeError("Cache population produced no persistent entries.")
        if args.case_kind == "reopen" and (
            result["correctness"]["persistent_cache_hits"] <= 0
        ):
            raise RuntimeError("Fresh-process cache reopen recorded no hits.")
        fatal_text = json.dumps(result).lower()
        fatal_tokens = (
            "cudaerrorillegaladdress",
            "illegal address",
            "out of memory",
            "host-register",
        )
        if any(token in fatal_text for token in fatal_tokens):
            raise RuntimeError("A fatal CUDA allocation/runtime token was recorded.")
        result["status"] = "pass"
    except BaseException as exc:
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result.setdefault("errors", []).append(result["error"])
    finally:
        if widget is not None:
            try:
                widget.close()
                result["workers_alive_after_close"] = {
                    name: bool(
                        getattr(widget, name, None)
                        and getattr(widget, name).is_alive()
                    )
                    for name in (
                        "_folder_watch_thread",
                        "_compare_page_thread",
                        "_compare_cache_warm_thread",
                        "_dataset_preload_thread",
                    )
                }
                if any(result["workers_alive_after_close"].values()):
                    result.setdefault("cleanup_errors", []).append(
                        "one or more Show4DSTEM workers remained alive after close"
                    )
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append(
                    f"close: {type(exc).__name__}: {exc}"
                )
        try:
            import torch

            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    with torch.cuda.device(idx):
                        torch.cuda.empty_cache()
        except Exception as exc:
            result.setdefault("cleanup_errors", []).append(
                f"torch cleanup: {type(exc).__name__}: {exc}"
            )
        result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        result["ended_at"] = _utc_now()
        if result.get("cleanup_errors") and result.get("status") == "pass":
            result["status"] = "fail"
            result["error"] = "; ".join(result["cleanup_errors"])
        _atomic_json(result_path, result)
    return 0 if result.get("status") == "pass" else 1


def _wait_for_idle(
    report: LiveReport,
    devices: Sequence[int],
    args: argparse.Namespace,
    *,
    phase: str,
) -> bool:
    deadline = time.monotonic() + args.wait_hours * 3600.0
    consecutive = 0
    while time.monotonic() < deadline:
        snapshot = _gpu_snapshot()
        idle, reasons = _idle_decision(
            snapshot,
            devices,
            max_utilization=args.max_idle_utilization,
            min_free_mib=args.min_free_mib,
            block_patterns=args.block_pattern,
        )
        consecutive = consecutive + 1 if idle else 0
        report.gpu(snapshot, phase=phase)
        report.update(
            status="waiting_for_gpu",
            current_phase=phase,
            selected_physical_gpus=list(devices),
            idle_consecutive_samples=consecutive,
            idle_required_samples=args.idle_samples,
            idle_block_reasons=reasons,
            latest_gpu=snapshot,
        )
        report.event(
            "gpu_idle_sample",
            phase=phase,
            idle=idle,
            consecutive=consecutive,
            reasons=reasons,
        )
        if consecutive >= args.idle_samples:
            return True
        time.sleep(args.idle_sample_seconds)
    report.add_error(
        {
            "time": _utc_now(),
            "phase": phase,
            "error": f"GPU idle wait exceeded {args.wait_hours} hours",
        }
    )
    return False


def _run_case(
    report: LiveReport,
    args: argparse.Namespace,
    *,
    case_id: str,
    case_kind: str,
    devices: Sequence[int],
    cycles: int,
    deadline_epoch: float,
) -> bool:
    if report.completed(case_id):
        report.event("case_resume_skip", case_id=case_id)
        return True
    if not _wait_for_idle(
        report,
        devices,
        args,
        phase=f"wait:{case_id}",
    ):
        return False
    case_dir = report.artifact_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    command = [
        str(args.python),
        str(Path(__file__).resolve()),
        "--child",
        "--case-id",
        case_id,
        "--case-kind",
        case_kind,
        "--source",
        str(args.source),
        "--cache-dir",
        str(args.cache_dir),
        "--result-path",
        str(result_path),
        "--pattern",
        args.pattern,
        "--det-bin",
        str(args.det_bin),
        "--dtype",
        args.dtype,
        "--columns",
        str(args.columns),
        "--page-size",
        str(args.page_size),
        "--min-ready",
        str(args.min_ready),
        "--cycles",
        str(cycles),
        "--deadline-epoch",
        str(deadline_epoch),
        "--page-timeout",
        str(args.page_timeout),
        "--maintenance-timeout",
        str(args.maintenance_timeout),
        "--watch-interval",
        str(args.watch_interval),
        "--cache-max-bytes",
        str(args.cache_max_bytes),
        "--max-vram-fraction",
        str(args.max_vram_fraction),
    ]
    env = dict(os.environ)
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices)),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(args.repo / "src"),
            "TMPDIR": str(report.artifact_dir / "tmp"),
            "XDG_CACHE_HOME": str(args.cache_dir / "xdg"),
            "CUPY_CACHE_DIR": str(args.cache_dir / "cupy"),
            "MPLCONFIGDIR": str(args.cache_dir / "matplotlib"),
        }
    )
    for key in ("TMPDIR", "XDG_CACHE_HOME", "CUPY_CACHE_DIR", "MPLCONFIGDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    stdout_path = case_dir / "stdout.log"
    stderr_path = case_dir / "stderr.log"
    report.update(status="running", current_phase=case_id, active_command=command)
    report.event(
        "case_start",
        case_id=case_id,
        kind=case_kind,
        physical_gpus=list(devices),
        command=command,
    )
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=args.repo,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        timed_out = False
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > args.case_timeout_hours * 3600.0:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                break
            snapshot = _gpu_snapshot()
            report.gpu(snapshot, phase=case_id)
            report.update(
                status="running",
                current_phase=case_id,
                active_pid=process.pid,
                active_elapsed_seconds=round(elapsed, 3),
                latest_gpu=snapshot,
            )
            time.sleep(args.heartbeat_seconds)
    try:
        case = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        case = {
            "id": case_id,
            "status": "fail",
            "error": f"missing/invalid child result: {type(exc).__name__}: {exc}",
        }
    case.update(
        {
            "id": case_id,
            "kind": case_kind,
            "physical_gpus": list(devices),
            "child_exit_code": process.returncode,
            "timed_out": timed_out,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
    )
    if timed_out or process.returncode != 0:
        case["status"] = "fail"
    report.append_case(case)
    report.event("case_end", case_id=case_id, status=case.get("status"))
    if case.get("status") != "pass":
        report.add_error(
            {
                "time": _utc_now(),
                "phase": case_id,
                "error": case.get("error", f"child exited {process.returncode}"),
            }
        )
        return False
    return True


def _run_controller(args: argparse.Namespace) -> int:
    args.source = args.source.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.artifact_dir = args.artifact_dir.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    for output, label in (
        (args.cache_dir, "cache"),
        (args.artifact_dir, "artifact"),
    ):
        try:
            output.relative_to(args.source)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"{label} directory must be outside the real-data source: {output}"
            )
    if args.cache_dir == args.artifact_dir:
        raise ValueError("cache and artifact directories must be distinct")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if not args.source.is_dir():
        raise FileNotFoundError(f"source folder not found: {args.source}")
    if args.artifact_dir.stat().st_dev != args.cache_dir.stat().st_dev:
        locality = "separate_filesystems"
    else:
        locality = "same_filesystem"
    initial = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "starting",
        "started_at": _utc_now(),
        "heartbeat_at": _utc_now(),
        "current_phase": "preflight",
        "completed_cases": 0,
        "restart_count": 0,
        "provenance": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "command": sys.argv,
            "git": _git_snapshot(args.repo),
        },
        "data": {
            "source": str(args.source),
            "pattern": args.pattern,
            "masters_discovered": len(list(args.source.rglob(args.pattern))),
            "det_bin": args.det_bin,
            "dtype": args.dtype,
            "min_ready": args.min_ready,
        },
        "environment": {
            "source_filesystem": _filesystem_snapshot(args.source),
            "cache_filesystem": _filesystem_snapshot(args.cache_dir),
            "report_filesystem": _filesystem_snapshot(args.artifact_dir),
            "cache_report_locality": locality,
            "initial_gpu": _gpu_snapshot(),
        },
        "cases": [],
        "errors": [],
        "gates": [
            {"id": "S4D-17-backend", "status": "pending"},
            {"id": "S4D-18-backend", "status": "pending"},
            {"id": "S4D-19-cache", "status": "pending"},
            {
                "id": "S4D-14-live-arrival-browser",
                "status": "pending",
                "reason": "companion staged live-Jupyter/browser drive required",
            },
            {
                "id": "S4D-20-browser",
                "status": "pending",
                "reason": "actual browser paint/FPS evidence required",
            },
            {
                "id": "S4D-20-no-bin",
                "status": "pending",
                "reason": "separate det_bin=1 capacity leg required",
            },
        ],
    }
    report = LiveReport(args.artifact_dir, initial)
    report.event("controller_start", run_id=args.run_id)
    if args.dry_run:
        report.update(status="planned", current_phase="dry_run", ended_at=_utc_now())
        return 0

    topologies = [
        ("one_gpu", args.one_gpu),
        ("two_gpu", args.two_gpus),
    ]
    all_passed = True
    for topology, devices in topologies:
        topology_started = time.time()
        fixed_cases = [(f"{topology}-cold-disabled", "cold")]
        if topology == "one_gpu":
            fixed_cases.append((f"{topology}-cache-populate", "populate"))
        else:
            # Reuse the one-GPU entries without rebuilding them. This is the
            # backend-independence check required before any two-GPU refresh.
            fixed_cases.append((f"{topology}-cache-cross-topology", "reopen"))
        fixed_cases.extend(
            (f"{topology}-cache-reopen-{idx}", "reopen")
            for idx in range(1, args.reopens + 1)
        )
        for case_id, kind in fixed_cases:
            passed = _run_case(
                report,
                args,
                case_id=case_id,
                case_kind=kind,
                devices=devices,
                cycles=1,
                deadline_epoch=0.0,
            )
            if not passed:
                all_passed = False
                if not args.continue_on_failure:
                    break
        if not all_passed and not args.continue_on_failure:
            break
        endurance_deadline = topology_started + args.topology_hours * 3600.0
        passed = _run_case(
            report,
            args,
            case_id=f"{topology}-endurance",
            case_kind="endurance",
            devices=devices,
            cycles=args.min_cycles,
            deadline_epoch=endurance_deadline,
        )
        all_passed = all_passed and passed
        if not passed and not args.continue_on_failure:
            break

    gates = list(report.data.get("gates", []))
    cases_by_id = {item.get("id"): item for item in report.data.get("cases", [])}
    aggregates: dict[str, Any] = {}
    for topology, _ in topologies:
        cold = cases_by_id.get(f"{topology}-cold-disabled", {})
        reopens = [
            cases_by_id.get(f"{topology}-cache-reopen-{idx}", {})
            for idx in range(1, args.reopens + 1)
        ]
        cold_times = [
            float(item.get("elapsed_seconds", 0.0))
            for item in cold.get("page_actions", [])
            if item.get("elapsed_seconds") is not None
        ]
        reopen_times = [
            float(action.get("elapsed_seconds", 0.0))
            for case in reopens
            for action in case.get("page_actions", [])
            if action.get("elapsed_seconds") is not None
        ]
        cache_hits = sum(
            int(case.get("final_state", {}).get("preview_cache", {}).get("hits", 0))
            for case in reopens
        )
        aggregates[topology] = {
            "cold_page_seconds": cold_times,
            "reopen_page_seconds": reopen_times,
            "reopen_cache_hits": cache_hits,
            "note": (
                "backend completion timings only; browser cached/fresh paint "
                "thresholds remain a separate pending gate"
            ),
        }
    for gate in gates:
        if gate["id"] == "S4D-17-backend":
            endurance = cases_by_id.get("one_gpu-endurance", {})
            plan = endurance.get("final_state", {}).get("residency_plan", {})
            gate["status"] = (
                "pass"
                if report.completed("one_gpu-endurance")
                and plan.get("fits") is False
                and endurance.get("correctness", {}).get("bounded_residency") is True
                else "fail"
            )
            gate["observed"] = {
                "full_series_fits_one_gpu": plan.get("fits"),
                "resident_nbytes": endurance.get("final_state", {}).get(
                    "resident_nbytes"
                ),
                "budget_bytes": plan.get("budget_bytes"),
            }
        elif gate["id"] == "S4D-18-backend":
            endurance = cases_by_id.get("two_gpu-endurance", {})
            page_devices = endurance.get("final_state", {}).get("page_devices", [])
            used = {str(device) for device in page_devices}
            gate["status"] = (
                "pass"
                if report.completed("two_gpu-endurance")
                and any("cuda:0" in device for device in used)
                and any("cuda:1" in device for device in used)
                else "fail"
            )
            gate["observed_page_devices"] = sorted(used)
        elif gate["id"] == "S4D-19-cache":
            required = [
                f"{topology}-cache-reopen-{idx}"
                for topology, _ in topologies
                for idx in range(1, args.reopens + 1)
            ]
            backend_reopens_pass = all(report.completed(item) for item in required)
            cache_hits = sum(
                int(aggregates[topology]["reopen_cache_hits"])
                for topology, _ in topologies
            )
            gate["status"] = (
                "limited_backend_pass"
                if backend_reopens_pass and cache_hits > 0
                else "fail"
            )
            gate["reason"] = (
                "Fresh-process disk-cache reuse passed in Python, but S4D-19 "
                "is not fully passed until cached/fresh browser paint gates pass."
            )
            gate["observed_cache_hits"] = cache_hits
    final_status = "backend_pass_browser_pending" if all_passed else "fail"
    report.update(
        status=final_status,
        current_phase="complete",
        gates=gates,
        aggregates=aggregates,
        ended_at=_utc_now(),
        active_pid=None,
    )
    report.event("controller_end", status=final_status)
    return 0 if all_passed else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "local-only real-data Show4DSTEM from_folder one-/two-GPU "
            "paging and persistent-cache overnight signoff"
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--run-id", default=f"show4dstem-{datetime.now():%Y%m%d-%H%M%S}")
    parser.add_argument("--pattern", default="*_master.h5")
    parser.add_argument("--one-gpu", type=_parse_devices, default=[0])
    parser.add_argument("--two-gpus", type=_parse_devices, default=[0, 1])
    parser.add_argument("--det-bin", type=int, default=4)
    parser.add_argument("--dtype", default="u16")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--min-ready", type=int, default=82)
    parser.add_argument("--reopens", type=int, default=5)
    parser.add_argument("--topology-hours", type=float, default=4.0)
    parser.add_argument("--min-cycles", type=int, default=100)
    parser.add_argument("--case-timeout-hours", type=float, default=8.0)
    parser.add_argument("--wait-hours", type=float, default=24.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--idle-sample-seconds", type=float, default=60.0)
    parser.add_argument("--idle-samples", type=int, default=5)
    parser.add_argument("--max-idle-utilization", type=int, default=15)
    parser.add_argument("--min-free-mib", type=int, default=70000)
    parser.add_argument(
        "--block-pattern",
        action="append",
        default=list(DEFAULT_BLOCK_PATTERNS),
        help="case-insensitive foreign command substring that blocks launch",
    )
    parser.add_argument("--page-timeout", type=float, default=900.0)
    parser.add_argument("--maintenance-timeout", type=float, default=14400.0)
    parser.add_argument("--watch-interval", type=float, default=2.0)
    parser.add_argument("--cache-max-bytes", type=int, default=4 << 30)
    parser.add_argument("--max-vram-fraction", type=float, default=0.92)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--case-kind", default="", help=argparse.SUPPRESS)
    parser.add_argument("--result-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cycles", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--deadline-epoch", type=float, default=0.0, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.det_bin < 1 or args.page_size < 1 or args.columns < 1:
        raise ValueError("det-bin, page-size, and columns must be positive")
    if args.child:
        if args.result_path is None or not args.case_id or not args.case_kind:
            raise ValueError("child mode requires case id/kind and result path")
        return _run_child(args)
    if args.artifact_dir is None:
        raise ValueError("controller mode requires --artifact-dir")
    return _run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
