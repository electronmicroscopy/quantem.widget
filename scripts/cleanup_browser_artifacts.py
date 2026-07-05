#!/usr/bin/env python3
"""Clean stale browser-test artifacts without touching normal Chrome sessions."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


ARTIFACT_PATTERNS = [
    "com.google.Chrome*",
    ".com.google.Chrome*",
    ".org.chromium.Chromium*",
    "playwright_chromiumdev_profile-*",
    "playwright-artifacts-*",
    "quantem-widget-browser-*",
    "quantem-widget-browser-profile-*",
    "quantem_chrome*.log",
    "quantem_headless_chrome.log",
    "quantem_survey_chrome.log",
    "quantem-survey-chrome-profile",
    "quantem_docs_browser",
    "quantem_visual_chrome*",
    "drift-paper-chrome*.log",
    "chrome*.log",
    "lhchrome.log",
]

SAFE_PROCESS_RE = re.compile(r"(chrome|chromium|playwright)", re.IGNORECASE)


def _age_seconds(path: Path, now: float) -> float | None:
    try:
        return now - path.stat().st_mtime
    except FileNotFoundError:
        return None


def _remove_path(path: Path, *, dry_run: bool) -> str:
    if dry_run:
        return "would_remove"
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return "removed"


def _process_command(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip()


def _clean_pid_files(
    pid_dir: Path,
    *,
    min_age_seconds: float,
    now: float,
    dry_run: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not pid_dir.exists():
        return records
    for pid_file in sorted(pid_dir.glob("*.pid")):
        age = _age_seconds(pid_file, now)
        if age is None:
            continue
        record: dict[str, Any] = {
            "path": str(pid_file),
            "age_seconds": round(age, 1),
            "kind": "pid_file",
        }
        if age < min_age_seconds:
            record["status"] = "skipped_recent"
            records.append(record)
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except ValueError:
            record["status"] = _remove_path(pid_file, dry_run=dry_run)
            record["reason"] = "invalid_pid_file"
            records.append(record)
            continue
        command = _process_command(pid)
        record["pid"] = pid
        record["command"] = command
        if command and SAFE_PROCESS_RE.search(command):
            if dry_run:
                record["status"] = "would_kill"
            else:
                try:
                    os.kill(pid, signal.SIGTERM)
                    record["status"] = "terminated"
                except ProcessLookupError:
                    record["status"] = "already_exited"
            if not dry_run:
                try:
                    pid_file.unlink()
                except FileNotFoundError:
                    pass
        else:
            record["status"] = _remove_path(pid_file, dry_run=dry_run)
            record["reason"] = "pid_not_running_or_not_browser"
        records.append(record)
    return records


def clean_browser_artifacts(
    tmp_root: Path,
    *,
    older_than_hours: float,
    pid_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    now = time.time()
    min_age_seconds = older_than_hours * 3600
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for pattern in ARTIFACT_PATTERNS:
        for path in sorted(tmp_root.glob(pattern)):
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            age = _age_seconds(path, now)
            if age is None:
                continue
            record: dict[str, Any] = {
                "path": str(path),
                "kind": "artifact",
                "age_seconds": round(age, 1),
            }
            if age < min_age_seconds:
                record["status"] = "skipped_recent"
            else:
                try:
                    record["status"] = _remove_path(path, dry_run=dry_run)
                except Exception as exc:  # pragma: no cover - depends on OS permissions
                    record["status"] = "failed"
                    record["error"] = str(exc)
            records.append(record)

    records.extend(
        _clean_pid_files(
            pid_dir,
            min_age_seconds=min_age_seconds,
            now=now,
            dry_run=dry_run,
        )
    )
    return {
        "tmp_root": str(tmp_root),
        "pid_dir": str(pid_dir),
        "older_than_hours": older_than_hours,
        "dry_run": dry_run,
        "removed": sum(1 for item in records if item["status"] in {"removed", "terminated", "already_exited"}),
        "skipped_recent": sum(1 for item in records if item["status"] == "skipped_recent"),
        "failed": sum(1 for item in records if item["status"] == "failed"),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean stale QuantEM browser-test temp files and tracked browser PID files."
    )
    parser.add_argument("--tmp-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--older-than-hours", type=float, default=6.0)
    parser.add_argument("--pid-dir", type=Path, default=Path("/tmp/quantem-widget-browser-pids"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the full cleanup report as JSON.")
    args = parser.parse_args()

    if args.older_than_hours < 0:
        raise SystemExit("--older-than-hours must be non-negative")
    report = clean_browser_artifacts(
        args.tmp_root.resolve(),
        older_than_hours=args.older_than_hours,
        pid_dir=args.pid_dir.resolve(),
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        action = "would remove" if args.dry_run else "removed"
        print(
            "Browser artifact cleanup: "
            f"{action} {report['removed']} stale item(s), "
            f"skipped {report['skipped_recent']} recent item(s), "
            f"failed {report['failed']}."
        )
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
