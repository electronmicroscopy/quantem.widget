from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes([size % 251]) * size)
    return path


def _master(root: Path, stem: str, master_size: int, sidecar_size: int = 0) -> Path:
    master = _write(root / f"{stem}_master.h5", master_size)
    if sidecar_size:
        _write(root / f"{stem}_data_000001.h5", sidecar_size)
    return master


def test_collect_data_transfer_groups_keeps_master_and_sidecars_together(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from quantem.widget.io import hdf5
    from quantem.widget.io.data_transfer import collect_data_transfer_groups

    monkeypatch.setattr(hdf5, "is_master_ready", lambda path: True)
    monkeypatch.setattr(hdf5, "disk_of", lambda path: "nvme0n1")

    _master(tmp_path, "scan_000", 10, 3)

    groups = collect_data_transfer_groups(tmp_path)

    assert len(groups) == 1
    assert groups[0].logical_id == "scan_000"
    assert groups[0].ready is True
    assert groups[0].size_bytes == 13
    assert [file.role for file in groups[0].files] == ["master", "sidecar"]
    assert [Path(file.source).name for file in groups[0].files] == [
        "scan_000_master.h5",
        "scan_000_data_000001.h5",
    ]
    assert [file.relative_path for file in groups[0].files] == [
        "scan_000_master.h5",
        "scan_000_data_000001.h5",
    ]


def test_plan_data_transfer_balances_groups_by_size(monkeypatch, tmp_path: Path) -> None:
    from quantem.widget.io import hdf5
    from quantem.widget.io.data_transfer import plan_data_transfer

    source = tmp_path / "source"
    target_a = tmp_path / "nvme_a"
    target_b = tmp_path / "nvme_b"
    target_a.mkdir()
    target_b.mkdir()
    _master(source, "big", 100, 10)
    _master(source, "medium", 60, 6)
    _master(source, "small", 50, 5)

    def fake_disk(path: str) -> str:
        if "nvme_a" in path:
            return "nvme0n1"
        if "nvme_b" in path:
            return "nvme1n1"
        return "source_disk"

    monkeypatch.setattr(hdf5, "is_master_ready", lambda path: True)
    monkeypatch.setattr(hdf5, "disk_of", fake_disk)

    plan = plan_data_transfer(source, [target_a, target_b], logical_name="timeseries")

    assert plan.logical_name == "timeseries"
    assert plan.total_bytes == 231
    assert plan.totals_by_target[str(target_a)] == 110
    assert plan.totals_by_target[str(target_b)] == 121
    assert [Path(entry.master).stem for entry in plan.entries] == [
        "big_master",
        "medium_master",
        "small_master",
    ]
    assert [entry.target_disk for entry in plan.entries] == [
        "nvme0n1",
        "nvme1n1",
        "nvme1n1",
    ]
    assert [Path(entry.target_master).name for entry in plan.entries] == [
        "big_master.h5",
        "medium_master.h5",
        "small_master.h5",
    ]


def test_plan_data_transfer_can_skip_not_ready_masters(monkeypatch, tmp_path: Path) -> None:
    from quantem.widget.io import hdf5
    from quantem.widget.io.data_transfer import plan_data_transfer

    good = _master(tmp_path / "source", "good", 10)
    bad = _master(tmp_path / "source", "bad", 20)
    target = tmp_path / "target"
    target.mkdir()

    monkeypatch.setattr(hdf5, "disk_of", lambda path: "disk")
    monkeypatch.setattr(hdf5, "is_master_ready", lambda path: str(path) == str(good))

    plan = plan_data_transfer(tmp_path / "source", [target], require_ready=True)

    assert [entry.master for entry in plan.entries] == [str(good)]
    assert [group.master for group in plan.skipped] == [str(bad)]


def test_data_transfer_manifest_and_copy_are_safe_by_default(monkeypatch, tmp_path: Path) -> None:
    from quantem.widget.io import hdf5
    from quantem.widget.io.data_transfer import (
        copy_data_transfer,
        filter_data_transfer_plan,
        inspect_data_transfer,
        plan_data_transfer,
        read_data_transfer_manifest,
        summarize_data_transfer,
        write_data_transfer_manifest,
    )

    source = tmp_path / "source"
    target = tmp_path / "target"
    _master(source, "scan_000", 10, 5)

    monkeypatch.setattr(hdf5, "disk_of", lambda path: "disk")
    monkeypatch.setattr(hdf5, "is_master_ready", lambda path: True)

    plan = plan_data_transfer(source, [target])
    dry = copy_data_transfer(plan)

    assert {result.status for result in dry} == {"planned"}
    assert {result.logical_id for result in dry} == {"scan_000"}
    assert not (target / "scan_000_master.h5").exists()
    assert {state.status for state in inspect_data_transfer(plan)} == {"not-started"}

    manifest = write_data_transfer_manifest(plan, tmp_path / "data-transfer.json")
    manifest_plan = read_data_transfer_manifest(manifest)
    assert manifest_plan.manifest_version == 1
    assert manifest_plan.logical_name == "source"
    assert len(manifest_plan.entries) == 1
    assert json.loads(manifest.read_text())["total_bytes"] == 15
    assert {state.status for state in inspect_data_transfer(manifest_plan)} == {"not-started"}
    assert filter_data_transfer_plan(manifest_plan).total_bytes == 15

    copied = copy_data_transfer(plan, dry_run=False)
    assert {result.status for result in copied} == {"copied"}
    states = inspect_data_transfer(plan)
    assert {state.status for state in states} == {"exists"}
    assert summarize_data_transfer(states).pending_bytes == 0
    assert (target / "scan_000_master.h5").read_bytes() == (
        source / "scan_000_master.h5"
    ).read_bytes()
    assert (target / "scan_000_data_000001.h5").read_bytes() == (
        source / "scan_000_data_000001.h5"
    ).read_bytes()
    assert not list(target.glob("*.partial"))


def test_copy_data_transfer_refuses_different_existing_target(monkeypatch, tmp_path: Path) -> None:
    from quantem.widget.io import hdf5
    from quantem.widget.io.data_transfer import copy_data_transfer, plan_data_transfer

    source = tmp_path / "source"
    target = tmp_path / "target"
    _master(source, "scan_000", 10)
    _write(target / "scan_000_master.h5", 3)

    monkeypatch.setattr(hdf5, "disk_of", lambda path: "disk")
    monkeypatch.setattr(hdf5, "is_master_ready", lambda path: True)

    plan = plan_data_transfer(source, [target])

    with pytest.raises(FileExistsError, match="different content"):
        copy_data_transfer(plan, dry_run=False)


def test_hash_verification_catches_same_size_target(monkeypatch, tmp_path: Path) -> None:
    from quantem.widget.io import hdf5
    from quantem.widget.io.data_transfer import copy_data_transfer, inspect_data_transfer, plan_data_transfer

    source = tmp_path / "source"
    target = tmp_path / "target"
    _write(source / "scan_000_master.h5", 4)
    _write(target / "scan_000_master.h5", 4)
    target_file = target / "scan_000_master.h5"
    target_file.write_bytes(b"abcd")
    source_file = source / "scan_000_master.h5"
    source_file.write_bytes(b"wxyz")

    monkeypatch.setattr(hdf5, "disk_of", lambda path: "disk")
    monkeypatch.setattr(hdf5, "is_master_ready", lambda path: True)

    plan = plan_data_transfer(source, [target], hash_algorithm="sha256")

    assert {state.status for state in inspect_data_transfer(plan, verify="size")} == {"exists"}
    assert {state.status for state in inspect_data_transfer(plan, verify="hash")} == {"mismatch"}
    with pytest.raises(FileExistsError, match="different content"):
        copy_data_transfer(plan, dry_run=False, verify="hash")


def test_inspect_data_transfer_reports_partial_and_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from quantem.widget.io import hdf5
    from quantem.widget.io.data_transfer import inspect_data_transfer, plan_data_transfer

    source = tmp_path / "source"
    target = tmp_path / "target"
    _master(source, "scan_000", 10, 5)
    _write(target / "scan_000_master.h5.partial", 2)
    _write(target / "scan_000_data_000001.h5", 1)

    monkeypatch.setattr(hdf5, "disk_of", lambda path: "disk")
    monkeypatch.setattr(hdf5, "is_master_ready", lambda path: True)

    plan = plan_data_transfer(source, [target])
    states = {
        Path(state.target).name: state.status
        for state in inspect_data_transfer(plan)
    }

    assert states["scan_000_master.h5"] == "partial"
    assert states["scan_000_data_000001.h5"] == "mismatch"
