from __future__ import annotations

import pytest

from quantem.widget.io import hdf5


def _patch_disks(monkeypatch: pytest.MonkeyPatch, disk_map: dict[str, str]) -> None:
    monkeypatch.setattr(hdf5, "disk_of", lambda path: disk_map[str(path)])


def test_disk_interleaved_indices_preserve_single_disk_order(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = ["a", "b", "c", "d"]
    _patch_disks(monkeypatch, {path: "nvme0n1" for path in paths})

    assert hdf5._disk_interleaved_indices(paths) == [0, 1, 2, 3]


def test_disk_interleaved_indices_mix_grouped_disks(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = ["d0_a", "d0_b", "d1_a", "d1_b", "d2_a"]
    _patch_disks(
        monkeypatch,
        {
            "d0_a": "nvme0n1",
            "d0_b": "nvme0n1",
            "d1_a": "nvme1n1",
            "d1_b": "nvme1n1",
            "d2_a": "nvme2n1",
        },
    )

    assert hdf5._disk_interleaved_indices(paths) == [0, 2, 4, 1, 3]


def test_sharded_assignment_keeps_two_grouped_disks_on_separate_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ["d0_a", "d0_b", "d1_a", "d1_b"]
    _patch_disks(
        monkeypatch,
        {
            "d0_a": "nvme0n1",
            "d0_b": "nvme0n1",
            "d1_a": "nvme1n1",
            "d1_b": "nvme1n1",
        },
    )

    assert hdf5._assign_indices_to_devices(paths, [0, 1]) == {
        0: [0, 1],
        1: [2, 3],
    }


def test_sharded_assignment_falls_back_to_round_robin_for_one_disk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ["a", "b", "c", "d", "e"]
    _patch_disks(monkeypatch, {path: "nvme0n1" for path in paths})

    assert hdf5._assign_indices_to_devices(paths, [0, 1]) == {
        0: [0, 2, 4],
        1: [1, 3],
    }


def test_sharded_assignment_requires_at_least_one_device() -> None:
    with pytest.raises(ValueError, match="at least one"):
        hdf5._assign_indices_to_devices(["a"], [])
