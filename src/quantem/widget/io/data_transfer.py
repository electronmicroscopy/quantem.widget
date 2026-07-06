"""Data-transfer utilities for large microscopy sessions.

The functions in this module plan and record where master files and their
sibling ``*_data_*.h5`` files should live before heavy browsing or
reconstruction. The first backend is local filesystem transfer across target
folders/disks, but the manifest model is intentionally general enough for
future HPC-to-HPC transfer and split-session workflows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Literal


Action = Literal["copy"]
Strategy = Literal["balance-by-size", "round-robin"]
HashAlgorithm = Literal["sha256"]
StateStatus = Literal[
    "not-started",
    "exists",
    "mismatch",
    "partial",
    "missing-source",
]
VerifyMode = Literal["size", "none", "hash", "sha256"]


__all__ = [
    "DataTransferEntry",
    "DataTransferFile",
    "DataTransferGroup",
    "DataTransferPlan",
    "DataTransferResult",
    "DataTransferState",
    "DataTransferSummary",
    "collect_data_transfer_groups",
    "copy_data_transfer",
    "data_transfer_plan_from_dict",
    "filter_data_transfer_plan",
    "inspect_data_transfer",
    "plan_data_transfer",
    "read_data_transfer_manifest",
    "summarize_data_transfer",
    "update_data_transfer_plan",
    "write_data_transfer_manifest",
]


@dataclass(frozen=True)
class DataTransferFile:
    """One physical file participating in a logical acquisition group."""

    source: str
    relative_path: str
    size_bytes: int
    role: Literal["master", "sidecar"] = "sidecar"
    checksum: str | None = None
    checksum_algorithm: HashAlgorithm | None = None


@dataclass(frozen=True)
class DataTransferGroup:
    """A logical acquisition: one master plus sidecars that move together."""

    logical_id: str
    master: str
    files: tuple[DataTransferFile, ...]
    size_bytes: int
    source_disk: str
    ready: bool
    missing_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataTransferEntry:
    """Placement decision for one :class:`DataTransferGroup`."""

    logical_id: str
    master: str
    target_master: str
    target_root: str
    target_disk: str
    source_disk: str
    size_bytes: int
    files: tuple[DataTransferFile, ...]
    target_files: tuple[str, ...]


@dataclass(frozen=True)
class DataTransferPlan:
    """Reloadable data-transfer plan for a microscopy session."""

    logical_name: str
    source_root: str
    target_roots: tuple[str, ...]
    action: Action
    strategy: Strategy
    entries: tuple[DataTransferEntry, ...]
    total_bytes: int
    totals_by_target: dict[str, int]
    skipped: tuple[DataTransferGroup, ...] = ()
    manifest_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the plan."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataTransferPlan":
        """Rebuild a typed plan from a JSON manifest dictionary."""
        return data_transfer_plan_from_dict(data)


@dataclass(frozen=True)
class DataTransferResult:
    """Result row from :func:`copy_data_transfer`."""

    logical_id: str
    source: str
    target: str
    size_bytes: int
    status: Literal["planned", "copied", "exists"]


@dataclass(frozen=True)
class DataTransferState:
    """Current on-disk state for one planned file transfer."""

    logical_id: str
    source: str
    target: str
    size_bytes: int
    status: StateStatus


@dataclass(frozen=True)
class DataTransferSummary:
    """Aggregated state useful for CLI output and widget status panels."""

    total_files: int
    total_bytes: int
    status_counts: dict[str, int]
    bytes_by_status: dict[str, int]
    complete_bytes: int
    pending_bytes: int
    problem_files: int


def _disk_of_existing(path: Path) -> str:
    """Return the physical disk for *path* or its nearest existing parent."""
    from quantem.widget.io.hdf5 import disk_of

    probe = path.expanduser()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return disk_of(str(probe))


def _master_sidecars(master: Path) -> list[Path]:
    """Return sidecar data files matching an Arina-style master file."""
    name = master.name
    if name.endswith("_master.h5"):
        prefix = name[: -len("_master.h5")]
        return sorted(master.parent.glob(f"{prefix}_data_*.h5"))
    return []


def _hash_file(path: Path, algorithm: HashAlgorithm = "sha256") -> str:
    """Return a hexadecimal file digest."""
    if algorithm != "sha256":
        raise ValueError("Only hash_algorithm='sha256' is supported.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_hash_algorithm(value: HashAlgorithm | str | None) -> HashAlgorithm | None:
    if value is None or value == "none":
        return None
    if value == "sha256":
        return "sha256"
    raise ValueError("hash_algorithm must be None, 'none', or 'sha256'.")


def _verify_mode_uses_hash(verify: VerifyMode) -> bool:
    return verify in ("hash", "sha256")


def _same_content(
    source: Path,
    target: Path,
    file: DataTransferFile,
    verify: VerifyMode,
) -> bool:
    if verify == "none":
        return True
    if target.stat().st_size != file.size_bytes:
        return False
    if not _verify_mode_uses_hash(verify):
        return True
    algorithm: HashAlgorithm = file.checksum_algorithm or "sha256"
    expected = file.checksum or _hash_file(source, algorithm)
    return _hash_file(target, algorithm) == expected


def _logical_id_from_master(master: Path, source_root: Path) -> str:
    """Return a readable acquisition ID for a master file."""
    try:
        relative = master.relative_to(source_root)
    except ValueError:
        relative = Path(master.name)
    name = relative.as_posix()
    if name.endswith("_master.h5"):
        return name[: -len("_master.h5")]
    return relative.with_suffix("").as_posix()


def _transfer_file(
    path: Path,
    source_root: Path,
    *,
    role: Literal["master", "sidecar"],
    hash_algorithm: HashAlgorithm | None = None,
) -> DataTransferFile:
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        relative = Path(path.name)
    return DataTransferFile(
        source=str(path),
        relative_path=relative.as_posix(),
        size_bytes=path.stat().st_size,
        role=role,
        checksum=_hash_file(path, hash_algorithm) if hash_algorithm else None,
        checksum_algorithm=hash_algorithm,
    )


def collect_data_transfer_groups(
    source: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    pattern: str = "*_master.h5",
    recursive: bool = True,
    require_ready: bool = False,
    hash_algorithm: HashAlgorithm | str | None = None,
) -> list[DataTransferGroup]:
    """Collect logical master-file groups for data transfer.

    Parameters
    ----------
    source
        Folder containing ``*_master.h5`` files, or an explicit sequence of
        master paths.
    pattern
        Glob used when *source* is a folder.
    recursive
        Search subfolders recursively when *source* is a folder.
    require_ready
        If ``True``, groups whose master does not pass ``is_master_ready`` can
        be returned in the plan's skipped list by :func:`plan_data_transfer`.
    hash_algorithm
        Optional digest to include in the manifest. Use ``"sha256"`` only when
        stronger verification is worth the extra full-file read.

    Returns
    -------
    list[DataTransferGroup]
        One group per master, including matching ``*_data_*.h5`` sidecars.
    """
    from quantem.widget.io.hdf5 import is_master_ready

    hash_algorithm = _normalize_hash_algorithm(hash_algorithm)
    if isinstance(source, (list, tuple)):
        masters = [Path(item).expanduser() for item in source]
        source_root = (
            Path(os.path.commonpath([str(p.parent) for p in masters]))
            if masters else Path.cwd()
        )
    else:
        source_root = Path(source).expanduser()
        glob = source_root.rglob if recursive else source_root.glob
        masters = sorted(path for path in glob(pattern) if path.is_file())

    groups: list[DataTransferGroup] = []
    for master in sorted(masters):
        files: list[DataTransferFile] = []
        if master.exists():
            files.append(_transfer_file(
                master,
                source_root,
                role="master",
                hash_algorithm=hash_algorithm,
            ))
        for sidecar in _master_sidecars(master):
            if sidecar.exists():
                files.append(_transfer_file(
                    sidecar,
                    source_root,
                    role="sidecar",
                    hash_algorithm=hash_algorithm,
                ))
        transfer_files = tuple(files)
        missing_files: tuple[str, ...] = ()
        ready = False
        try:
            ready = bool(is_master_ready(str(master)))
        except Exception:
            ready = False
        if require_ready and not ready:
            missing_files = tuple(
                file.relative_path for file in transfer_files if not Path(file.source).exists()
            )
        logical_id = _logical_id_from_master(master, source_root)
        groups.append(
            DataTransferGroup(
                logical_id=logical_id,
                master=str(master),
                files=transfer_files,
                size_bytes=sum(file.size_bytes for file in transfer_files),
                source_disk=_disk_of_existing(master),
                ready=ready,
                missing_files=missing_files,
            )
        )
    return groups


def plan_data_transfer(
    source: str | Path | list[str | Path] | tuple[str | Path, ...],
    targets: list[str | Path] | tuple[str | Path, ...],
    *,
    pattern: str = "*_master.h5",
    recursive: bool = True,
    strategy: Strategy = "balance-by-size",
    action: Action = "copy",
    require_ready: bool = False,
    logical_name: str | None = None,
    hash_algorithm: HashAlgorithm | str | None = None,
) -> DataTransferPlan:
    """Plan a safe data-transfer layout.

    This is a dry-run planner: it does not copy or move files. Use
    :func:`copy_data_transfer` after reviewing the plan. Set
    ``hash_algorithm="sha256"`` only when stronger verification is worth the
    extra full-file read.
    """
    if action != "copy":
        raise ValueError(
            "Only action='copy' is implemented. Use copy before adding "
            "destructive move workflows."
        )
    if strategy not in ("balance-by-size", "round-robin"):
        raise ValueError("strategy must be 'balance-by-size' or 'round-robin'.")
    if not targets:
        raise ValueError("At least one target folder is required.")

    target_roots = tuple(str(Path(target).expanduser()) for target in targets)
    totals = {target: 0 for target in target_roots}
    target_disks = {target: _disk_of_existing(Path(target)) for target in target_roots}
    groups = collect_data_transfer_groups(
        source,
        pattern=pattern,
        recursive=recursive,
        require_ready=require_ready,
        hash_algorithm=hash_algorithm,
    )
    active_groups: list[DataTransferGroup] = []
    skipped: list[DataTransferGroup] = []
    for group in groups:
        if require_ready and not group.ready:
            skipped.append(group)
        else:
            active_groups.append(group)

    entries: list[DataTransferEntry] = []
    ordered_groups = (
        sorted(active_groups, key=lambda group: group.size_bytes, reverse=True)
        if strategy == "balance-by-size"
        else list(active_groups)
    )
    for idx, group in enumerate(ordered_groups):
        if strategy == "balance-by-size":
            target_root = min(target_roots, key=lambda target: (totals[target], target))
        else:
            target_root = target_roots[idx % len(target_roots)]
        target_files = tuple(
            str(Path(target_root) / file.relative_path)
            for file in group.files
        )
        target_master = next(
            str(Path(target_root) / file.relative_path)
            for file in group.files
            if file.role == "master"
        )
        entries.append(
            DataTransferEntry(
                logical_id=group.logical_id,
                master=group.master,
                target_master=target_master,
                target_root=target_root,
                target_disk=target_disks[target_root],
                source_disk=group.source_disk,
                size_bytes=group.size_bytes,
                files=group.files,
                target_files=target_files,
            )
        )
        totals[target_root] += group.size_bytes

    if isinstance(source, (list, tuple)):
        source_root = (
            Path(os.path.commonpath([str(Path(item).expanduser().parent) for item in source]))
            if source else Path.cwd()
        )
    else:
        source_root = Path(source).expanduser()
    return DataTransferPlan(
        logical_name=logical_name or source_root.name,
        source_root=str(source_root),
        target_roots=target_roots,
        action=action,
        strategy=strategy,
        entries=tuple(entries),
        total_bytes=sum(entry.size_bytes for entry in entries),
        totals_by_target=totals,
        skipped=tuple(skipped),
    )


def _entry_for_group(
    group: DataTransferGroup,
    target_root: str,
    target_disk: str,
) -> DataTransferEntry:
    """Build a placement entry for a collected group."""
    target_files = tuple(
        str(Path(target_root) / file.relative_path)
        for file in group.files
    )
    target_master = next(
        str(Path(target_root) / file.relative_path)
        for file in group.files
        if file.role == "master"
    )
    return DataTransferEntry(
        logical_id=group.logical_id,
        master=group.master,
        target_master=target_master,
        target_root=target_root,
        target_disk=target_disk,
        source_disk=group.source_disk,
        size_bytes=group.size_bytes,
        files=group.files,
        target_files=target_files,
    )


def update_data_transfer_plan(
    plan: DataTransferPlan,
    *,
    source: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    pattern: str = "*_master.h5",
    recursive: bool = True,
    require_ready: bool = False,
    hash_algorithm: HashAlgorithm | str | None = None,
) -> DataTransferPlan:
    """Return a plan with newly discovered groups appended.

    Existing entries keep their target assignments. Newly discovered groups are
    assigned using the original strategy and the current assigned totals, which
    lets a watched folder grow without silently moving older datasets to a
    different disk.
    """
    if plan.action != "copy":
        raise ValueError("update_data_transfer_plan only supports copy plans.")
    target_roots = tuple(plan.target_roots)
    if not target_roots:
        raise ValueError("Cannot update a plan with no target roots.")

    source_to_scan = source if source is not None else Path(plan.source_root)
    groups = collect_data_transfer_groups(
        source_to_scan,
        pattern=pattern,
        recursive=recursive,
        require_ready=require_ready,
        hash_algorithm=hash_algorithm,
    )
    existing_ids = {entry.logical_id for entry in plan.entries}
    entries = list(plan.entries)
    totals = {target: 0 for target in target_roots}
    for entry in entries:
        totals[entry.target_root] = totals.get(entry.target_root, 0) + entry.size_bytes
    target_disks = {target: _disk_of_existing(Path(target)) for target in target_roots}
    skipped = list(plan.skipped)
    skipped_ids = {group.logical_id for group in skipped}

    new_groups = [group for group in groups if group.logical_id not in existing_ids]
    if require_ready:
        ready_groups = []
        for group in new_groups:
            if group.ready:
                ready_groups.append(group)
            elif group.logical_id not in skipped_ids:
                skipped.append(group)
                skipped_ids.add(group.logical_id)
        new_groups = ready_groups
    if plan.strategy == "balance-by-size":
        new_groups = sorted(new_groups, key=lambda group: group.size_bytes, reverse=True)

    for idx, group in enumerate(new_groups):
        if plan.strategy == "balance-by-size":
            target_root = min(target_roots, key=lambda target: (totals[target], target))
        else:
            target_root = target_roots[(len(entries) + idx) % len(target_roots)]
        entry = _entry_for_group(group, target_root, target_disks[target_root])
        entries.append(entry)
        totals[target_root] = totals.get(target_root, 0) + entry.size_bytes

    return DataTransferPlan(
        logical_name=plan.logical_name,
        source_root=plan.source_root,
        target_roots=target_roots,
        action=plan.action,
        strategy=plan.strategy,
        entries=tuple(entries),
        total_bytes=sum(entry.size_bytes for entry in entries),
        totals_by_target=totals,
        skipped=tuple(skipped),
        manifest_version=plan.manifest_version,
    )


def data_transfer_plan_from_dict(data: dict[str, Any]) -> DataTransferPlan:
    """Convert a manifest dictionary into a typed :class:`DataTransferPlan`."""
    version = int(data.get("manifest_version", 1))
    if version != 1:
        raise ValueError(f"Unsupported data-transfer manifest version: {version}")
    action = data.get("action", "copy")
    if action != "copy":
        raise ValueError(f"Unsupported data-transfer action: {action!r}")
    strategy = data.get("strategy", "balance-by-size")
    if strategy not in ("balance-by-size", "round-robin"):
        raise ValueError(f"Unsupported data-transfer strategy: {strategy!r}")

    def _file(row: dict[str, Any]) -> DataTransferFile:
        return DataTransferFile(
            source=str(row["source"]),
            relative_path=str(row["relative_path"]),
            size_bytes=int(row["size_bytes"]),
            role=row.get("role", "sidecar"),
            checksum=row.get("checksum"),
            checksum_algorithm=row.get("checksum_algorithm"),
        )

    def _group(row: dict[str, Any]) -> DataTransferGroup:
        return DataTransferGroup(
            logical_id=str(row["logical_id"]),
            master=str(row["master"]),
            files=tuple(_file(file) for file in row.get("files", [])),
            size_bytes=int(row.get("size_bytes", 0)),
            source_disk=str(row.get("source_disk", "")),
            ready=bool(row.get("ready", False)),
            missing_files=tuple(str(item) for item in row.get("missing_files", [])),
        )

    entries: list[DataTransferEntry] = []
    for row in data.get("entries", []):
        files = tuple(_file(file) for file in row.get("files", []))
        target_files = tuple(str(item) for item in row.get("target_files", []))
        if len(files) != len(target_files):
            raise ValueError(
                f"Manifest entry {row.get('logical_id', '<unknown>')!r} has "
                f"{len(files)} files but {len(target_files)} target files."
            )
        entries.append(
            DataTransferEntry(
                logical_id=str(row["logical_id"]),
                master=str(row["master"]),
                target_master=str(row["target_master"]),
                target_root=str(row["target_root"]),
                target_disk=str(row.get("target_disk", "")),
                source_disk=str(row.get("source_disk", "")),
                size_bytes=int(row.get("size_bytes", sum(file.size_bytes for file in files))),
                files=files,
                target_files=target_files,
            )
        )

    return DataTransferPlan(
        logical_name=str(data.get("logical_name", "")),
        source_root=str(data.get("source_root", "")),
        target_roots=tuple(str(item) for item in data.get("target_roots", [])),
        action="copy",
        strategy=strategy,
        entries=tuple(entries),
        total_bytes=int(data.get("total_bytes", sum(entry.size_bytes for entry in entries))),
        totals_by_target={
            str(target): int(size)
            for target, size in dict(data.get("totals_by_target", {})).items()
        },
        skipped=tuple(_group(row) for row in data.get("skipped", [])),
        manifest_version=version,
    )


def write_data_transfer_manifest(plan: DataTransferPlan, path: str | Path) -> Path:
    """Write a data-transfer manifest and return the path."""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    return output


def read_data_transfer_manifest(path: str | Path) -> DataTransferPlan:
    """Read a data-transfer manifest as a typed plan."""
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return data_transfer_plan_from_dict(data)


def inspect_data_transfer(
    plan: DataTransferPlan,
    *,
    verify: VerifyMode = "size",
) -> list[DataTransferState]:
    """Inspect current target/source state for a transfer plan."""
    if verify not in ("size", "none", "hash", "sha256"):
        raise ValueError("verify must be 'size', 'none', 'hash', or 'sha256'.")
    states: list[DataTransferState] = []
    for entry in plan.entries:
        for file, target_name in zip(entry.files, entry.target_files, strict=True):
            source = Path(file.source)
            target = Path(target_name)
            partial = target.with_name(f"{target.name}.partial")
            if not source.exists():
                status = "missing-source"
            elif target.exists():
                status = "exists" if _same_content(source, target, file, verify) else "mismatch"
            elif partial.exists():
                status = "partial"
            else:
                status = "not-started"
            states.append(
                DataTransferState(
                    logical_id=entry.logical_id,
                    source=str(source),
                    target=str(target),
                    size_bytes=file.size_bytes,
                    status=status,
                )
            )
    return states


def summarize_data_transfer(states: list[DataTransferState]) -> DataTransferSummary:
    """Aggregate per-file transfer states for reports and widgets."""
    status_counts: dict[str, int] = {}
    bytes_by_status: dict[str, int] = {}
    total_bytes = 0
    for state in states:
        status_counts[state.status] = status_counts.get(state.status, 0) + 1
        bytes_by_status[state.status] = bytes_by_status.get(state.status, 0) + state.size_bytes
        total_bytes += state.size_bytes
    complete_bytes = bytes_by_status.get("exists", 0)
    problem_files = status_counts.get("mismatch", 0) + status_counts.get("missing-source", 0)
    return DataTransferSummary(
        total_files=len(states),
        total_bytes=total_bytes,
        status_counts=status_counts,
        bytes_by_status=bytes_by_status,
        complete_bytes=complete_bytes,
        pending_bytes=max(0, total_bytes - complete_bytes),
        problem_files=problem_files,
    )


def filter_data_transfer_plan(
    plan: DataTransferPlan,
    *,
    statuses: tuple[StateStatus, ...] = ("not-started", "partial"),
    verify: VerifyMode = "size",
) -> DataTransferPlan:
    """Return a plan containing only files whose current status matches."""
    wanted = set(statuses)
    states = {
        (state.source, state.target): state.status
        for state in inspect_data_transfer(plan, verify=verify)
    }
    entries: list[DataTransferEntry] = []
    totals = {target: 0 for target in plan.target_roots}
    for entry in plan.entries:
        files: list[DataTransferFile] = []
        target_files: list[str] = []
        for file, target in zip(entry.files, entry.target_files, strict=True):
            if states.get((file.source, target)) in wanted:
                files.append(file)
                target_files.append(target)
        if not files:
            continue
        size_bytes = sum(file.size_bytes for file in files)
        totals[entry.target_root] = totals.get(entry.target_root, 0) + size_bytes
        entries.append(
            DataTransferEntry(
                logical_id=entry.logical_id,
                master=entry.master,
                target_master=entry.target_master,
                target_root=entry.target_root,
                target_disk=entry.target_disk,
                source_disk=entry.source_disk,
                size_bytes=size_bytes,
                files=tuple(files),
                target_files=tuple(target_files),
            )
        )
    return DataTransferPlan(
        logical_name=plan.logical_name,
        source_root=plan.source_root,
        target_roots=plan.target_roots,
        action=plan.action,
        strategy=plan.strategy,
        entries=tuple(entries),
        total_bytes=sum(entry.size_bytes for entry in entries),
        totals_by_target=totals,
        skipped=plan.skipped,
        manifest_version=plan.manifest_version,
    )


def copy_data_transfer(
    plan: DataTransferPlan,
    *,
    dry_run: bool = True,
    verify: VerifyMode = "size",
    overwrite: bool = False,
) -> list[DataTransferResult]:
    """Copy files from a reviewed data-transfer plan.

    Files are copied through ``*.partial`` siblings and then atomically replaced
    into their final path. ``dry_run=True`` is the default so notebooks and
    widgets can show the exact operation without mutating data.
    """
    if plan.action != "copy":
        raise ValueError("copy_data_transfer only supports copy plans.")
    if verify not in ("size", "none", "hash", "sha256"):
        raise ValueError("verify must be 'size', 'none', 'hash', or 'sha256'.")

    results: list[DataTransferResult] = []
    for entry in plan.entries:
        for file, target_name in zip(entry.files, entry.target_files, strict=True):
            source = Path(file.source)
            target = Path(target_name)
            if not source.exists():
                raise FileNotFoundError(f"Source file is missing: {source}")
            if target.exists() and not overwrite:
                if not _same_content(source, target, file, verify):
                    raise FileExistsError(
                        f"Target exists with different content: {target}. "
                        "Pass overwrite=True only after checking the destination."
                    )
                results.append(
                    DataTransferResult(
                        entry.logical_id,
                        str(source),
                        str(target),
                        file.size_bytes,
                        "exists",
                    )
                )
                continue
            if dry_run:
                results.append(
                    DataTransferResult(
                        entry.logical_id,
                        str(source),
                        str(target),
                        file.size_bytes,
                        "planned",
                    )
                )
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(f"{target.name}.partial")
            shutil.copy2(source, partial)
            if not _same_content(source, partial, file, verify):
                partial.unlink(missing_ok=True)
                raise IOError(f"Copied verification failed for {source} -> {target}")
            partial.replace(target)
            results.append(
                DataTransferResult(
                    entry.logical_id,
                    str(source),
                    str(target),
                    file.size_bytes,
                    "copied",
                )
            )
    return results
