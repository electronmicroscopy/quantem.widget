# Data Transfer Storyboard

Data transfer prepares large microscopy sessions for fast loading by copying or
splitting files across target folders and disks. It is intentionally not a
viewer widget right now. The supported interface is the set of Python
utilities under `quantem.widget.io` (the old `quantem data-transfer` CLI
subcommand was removed).

Keep this workflow separate from ShowFolder, Show2D, Show3D, and Show4DSTEM.
Those widgets inspect scientific data after it is already placed. Data transfer
only plans, verifies, copies, resumes, and records where files live.

## DT-1: Plan a Multi-Disk Session

**User story**: A microscopist has 60-80 time-resolved 4D-STEM masters for
future ptychography or browsing and wants to split the session across fast NVMe
disks before loading it into one or more NVIDIA GPUs.

**Primary tools**: `plan_data_transfer`, `write_data_transfer_manifest`.

**Acceptance checks**:

- Discovery finds every `*_master.h5` without loading detector frames or using
  GPU memory.
- Each master stays grouped with its matching `*_data_*.h5` sidecars.
- The plan reports logical ID, source path, target path, source disk, target
  disk, file count, and bytes.
- The default strategy balances by acquisition size, not by file count.
- Planning writes a JSON manifest by default.
- Dry-run planning never creates target files.
- Real-data signoff uses private workstation data; raw files are never added to
  the repository.

## DT-2: Copy Safely

**User story**: A user reviewed a plan and wants to copy files without corrupting
existing targets or partially written outputs.

**Primary tools**: `copy_data_transfer`, `filter_data_transfer_plan`.

**Acceptance checks**:

- Copy defaults to dry-run unless the CLI receives `--execute`.
- Copy writes to `*.partial` files first, then atomically replaces the final
  path.
- Existing targets with matching verification are skipped as already present.
- Existing targets with different content fail with a corrective error unless
  overwrite is explicit.
- Size verification is the fast default; SHA-256 is available when stronger
  verification is worth the extra reads.
- Result rows report `planned`, `copied`, or `exists` per physical file.

## DT-3: Resume and Verify

**User story**: A long transfer is interrupted by a notebook restart, network
hiccup, or disk quota problem. The user wants to resume without guessing which
files are safe.

**Primary tools**: `read_data_transfer_manifest`, `inspect_data_transfer`,
`summarize_data_transfer`.

**Acceptance checks**:

- Manifest reload is enough to inspect state; discovery does not need to rerun.
- State reports `not-started`, `partial`, `exists`, `mismatch`, or
  `missing-source`.
- Stale `*.partial` files are visible and retryable.
- A filtered plan can copy only pending or partial files.
- Inspect output is concise enough to paste into a notebook, issue, or lab log.

## DT-4: Append New Completed Masters

**User story**: A microscope or processing pipeline writes new masters while the
session is being prepared. The user wants to append complete groups without
changing existing target assignments.

**Primary tools**: `update_data_transfer_plan`, then manifest write/inspect/copy.

**Acceptance checks**:

- Newly discovered masters are ignored until readiness checks pass.
- Existing logical IDs keep their target assignments.
- New groups use the original target strategy.
- Repeated updates do not duplicate skipped or not-ready groups.
- Removed or failed files do not become valid acquisitions silently.
- The CLI rewrites the same manifest only after producing a valid updated plan.

## DT-5: Open Viewers After Transfer

**User story**: After files are copied, the user wants to browse them with
ShowFolder or load ready target masters with Show4DSTEM without remembering the
exact target paths.

**Primary tools**: `target_masters`, `ShowFolder(target_folder)`,
and `Show4DSTEM(load(...))`.

**Acceptance checks**:

- Viewer opening is a separate explicit step, not a data-transfer control.
- Transfer timing and viewer timing are measured separately.
- Show4DSTEM still relies on its own lazy paging and GPU memory controls.
- Data-transfer code does not import or construct viewer widgets.
- `target_masters(plan)` returns only complete target master groups by default.
- `target_masters(plan, existing_only=False, require_complete=False)` can list
  planned target paths before copy completion for lab-log review.
- The Show4DSTEM handoff notebook generated from a transfer plan explicitly
  uses `load(masters, devices=[0, 1], dtype="u8", det_bin=1)` and
  `Show4DSTEM(...)`.
- The generated notebook reloads the manifest and prints warnings so saved
  outputs record whether the handoff was complete and multi-disk.

## DT-6: Guard Disk-to-GPU Assumptions

**User story**: A user expects two GPUs to make browsing fast. They need the
tooling to say when a transfer layout still feeds both GPUs from one disk.

**Primary tools**: `data_transfer_load_warnings`.

**Acceptance checks**:

- Warnings appear when not all acquisitions are complete.
- Warnings appear when multiple target roots resolve to one physical disk.
- Warnings appear when multiple GPUs are requested but ready targets do not span
  multiple physical disks.
- Warnings are advisory and do not block `masters --all-masters` or dry-run
  inspection.
- Real-data signoff records source disk, target disks, requested GPUs, browse
  dtype, detector binning, cold-load time, warm-load time, and GPU memory before
  and after.

## Required Test Commands

Run these before committing data-transfer changes:

```bash
PYTHONPATH=src pytest -q tests/test_io_data_transfer.py
PYTHONPATH=src pytest -q tests/test_cli.py::test_data_transfer_cli_plan_inspect_copy_update_and_show4dstem
PYTHONPATH=src pytest -q tests/test_hdf5_disk_scheduling.py
```

For real workstation signoff, run the CLI against private NVIDIA workstation data and
record source disks, target disks, bytes, elapsed copy time, and verification
mode. Keep those raw files and generated benchmark artifacts out of Git.
