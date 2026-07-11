"""Header-only readiness checks for inline and external 4D-STEM masters."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from quantem.widget.io import (
    MasterReadiness,
    inspect_master_readiness,
    is_master_ready,
)


def _write_detector_file(
    path: Path,
    data: np.ndarray,
    *,
    maxshape: tuple[int | None, ...] | None = None,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=data,
            maxshape=maxshape,
            chunks=True if maxshape is not None else None,
        )


def _write_external_master(
    path: Path,
    sources: list[Path],
    *,
    ntrigger: int | None = None,
    nimages: int | None = None,
) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.require_group("entry/data")
        for index, source in enumerate(sources, start=1):
            group[f"data_{index:06d}"] = h5py.ExternalLink(
                source.name,
                "entry/data/data",
            )
        detector = handle.require_group(
            "entry/instrument/detector/detectorSpecific"
        )
        if ntrigger is not None:
            detector.create_dataset("ntrigger", data=int(ntrigger))
        if nimages is not None:
            detector.create_dataset("nimages", data=int(nimages))


def test_inline_master_reports_complete_header_contract(tmp_path: Path) -> None:
    # C1: a self-contained 4D dataset matches the explicit scan shape, expect a
    # ready report and a repeatable signature without reading detector pixels.
    master = tmp_path / "inline_master.h5"
    data = np.arange(2 * 2 * 4 * 5, dtype=np.uint16).reshape(2, 2, 4, 5)
    with h5py.File(master, "w") as handle:
        handle.create_dataset("entry/data/data", data=data)
        handle.create_dataset(
            "entry/instrument/detector/detectorSpecific/ntrigger",
            data=4,
        )

    report = inspect_master_readiness(master, scan_shape=(2, 2))
    repeated = inspect_master_readiness(master, scan_shape=(2, 2))

    assert isinstance(report, MasterReadiness)
    assert report.ready is True
    assert report.source_kind == "inline"
    assert report.actual_frames == 4
    assert report.expected_frames == 4
    assert report.detector_shape == (4, 5)
    assert report.dtype == np.dtype(np.uint16).str
    assert report.source_signature == repeated.source_signature
    assert json.loads(json.dumps(report.source_signature)) == report.source_signature
    assert report.source_signature["datasets"][0]["shape"] == [2, 2, 4, 5]
    assert is_master_ready(master, scan_shape=(2, 2)) is True


def test_external_master_totals_all_linked_frames(tmp_path: Path) -> None:
    # C1: two readable external chunks jointly satisfy ntrigger*nimages, expect
    # one complete external signature with both dataset headers.
    first = tmp_path / "scan_data_000001.h5"
    second = tmp_path / "scan_data_000002.h5"
    master = tmp_path / "scan_master.h5"
    _write_detector_file(first, np.ones((2, 4, 5), dtype=np.uint16))
    _write_detector_file(second, np.full((2, 4, 5), 2, dtype=np.uint16))
    _write_external_master(
        master,
        [first, second],
        ntrigger=2,
        nimages=2,
    )

    report = inspect_master_readiness(master)

    assert report.ready is True
    assert report.source_kind == "external"
    assert report.actual_frames == 4
    assert report.expected_frames == 4
    assert report.detector_shape == (4, 5)
    assert report.dtype == np.dtype(np.uint16).str
    assert len(report.source_signature["files"]) == 3
    assert [item["frames"] for item in report.source_signature["datasets"]] == [
        2,
        2,
    ]
    assert is_master_ready(master) is True


def test_missing_external_source_reports_corrective_action(tmp_path: Path) -> None:
    # C1: a master points at a detector file that has not arrived, expect a
    # non-ready result that identifies the path and tells the caller to retry.
    missing = tmp_path / "scan_data_000001.h5"
    master = tmp_path / "scan_master.h5"
    _write_external_master(master, [missing], ntrigger=4)

    report = inspect_master_readiness(master)

    assert report.ready is False
    assert "linked detector file is missing" in report.reason
    assert str(missing) in report.reason
    assert "then poll again" in report.action
    missing_sources = [
        item
        for item in report.source_signature["files"]
        if item.get("missing", False)
    ]
    assert missing_sources == [{"path": str(missing), "missing": True}]
    assert is_master_ready(master) is False


def test_explicit_scan_shape_overrides_incorrect_master_expectation(
    tmp_path: Path,
) -> None:
    # C1: stored data disagree with ntrigger but match an explicit operator scan
    # shape, expect metadata-only inspection to fail and the explicit contract
    # to pass without changing the source.
    master = tmp_path / "inline_master.h5"
    with h5py.File(master, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.ones((4, 3, 3), dtype=np.uint16),
        )
        handle.create_dataset(
            "entry/instrument/detector/detectorSpecific/ntrigger",
            data=9,
        )

    metadata_report = inspect_master_readiness(master)
    explicit_report = inspect_master_readiness(master, scan_shape=(2, 2))

    assert metadata_report.ready is False
    assert metadata_report.actual_frames == 4
    assert metadata_report.expected_frames == 9
    assert "stored frame count is 4; expected 9" in metadata_report.reason
    assert "remaining detector frames" in metadata_report.action
    assert explicit_report.ready is True
    assert explicit_report.expected_frames == 4


@pytest.mark.parametrize(
    ("second_shape", "second_dtype", "reason"),
    [
        ((2, 6, 5), np.uint16, "inconsistent detector shapes"),
        ((2, 4, 5), np.uint32, "inconsistent dtypes"),
    ],
)
def test_external_sources_require_consistent_detector_contract(
    tmp_path: Path,
    second_shape: tuple[int, int, int],
    second_dtype: type[np.generic],
    reason: str,
) -> None:
    # C1: one external chunk changes detector shape or stored dtype, expect a
    # precise non-ready contract instead of deferring failure to GPU decode.
    first = tmp_path / "scan_data_000001.h5"
    second = tmp_path / "scan_data_000002.h5"
    master = tmp_path / "scan_master.h5"
    _write_detector_file(first, np.ones((2, 4, 5), dtype=np.uint16))
    _write_detector_file(second, np.ones(second_shape, dtype=second_dtype))
    _write_external_master(master, [first, second], ntrigger=4)

    report = inspect_master_readiness(master)

    assert report.ready is False
    assert reason in report.reason
    assert report.action.startswith("Use a narrower") or report.action.startswith(
        "Repair or reacquire"
    )


def test_source_signature_changes_when_external_dataset_grows(
    tmp_path: Path,
) -> None:
    # C1: an extendable detector source grows between caller-owned polls, expect
    # both reports to remain readable but expose different complete signatures.
    source = tmp_path / "scan_data_000001.h5"
    master = tmp_path / "scan_master.h5"
    _write_detector_file(
        source,
        np.ones((2, 4, 5), dtype=np.uint16),
        maxshape=(None, 4, 5),
    )
    _write_external_master(master, [source])

    before = inspect_master_readiness(master)
    with h5py.File(source, "r+") as handle:
        dataset = handle["entry/data/data"]
        dataset.resize((3, 4, 5))
        dataset[2] = 3
        handle.flush()
    after = inspect_master_readiness(master)

    assert before.ready is True
    assert after.ready is True
    assert before.actual_frames == 2
    assert after.actual_frames == 3
    assert before.source_signature != after.source_signature
    assert after.source_signature["datasets"][0]["shape"] == [3, 4, 5]


@pytest.mark.parametrize("scan_shape", [(2,), (2, 0), (2, 1.5), (True, 2)])
def test_invalid_scan_shape_has_corrective_error(
    tmp_path: Path,
    scan_shape: tuple[object, ...],
) -> None:
    # C1: an invalid explicit scan contract cannot define an expected frame
    # count, expect an actionable ValueError before any filesystem inspection.
    with pytest.raises(ValueError, match="scan_shape.*positive integers"):
        inspect_master_readiness(
            tmp_path / "unused_master.h5",
            scan_shape=scan_shape,  # type: ignore[arg-type]
        )
