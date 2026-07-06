# DataTransfer Storyboard

DataTransfer is the data-layout workflow that prepares large microscopy
sessions for fast browsing, multi-GPU loading, and future joint iterative
ptychography. It is separate from ShowFolder: ShowFolder inspects files;
DataTransfer plans, verifies, and records where files actually live.

The name is intentionally broader than disk movement. The first backend is local
filesystem copy/split across target folders and disks, but the same manifest and
state model should later cover HPC-to-HPC transfer, split sessions, remote
scratch placement, and ptychography-ready logical datasets.

The primary user experience should be visual. A CLI is necessary for overnight
and batch ptychography preparation, but the widget must maintain the durable
state: logical acquisition IDs, source paths, target paths, target disks,
current file status, copy progress, and the manifest that lets a notebook or
future ptycho pipeline reproduce exactly what happened.

The implementation is a shared planner utility with a conservative CLI and a
notebook session-control widget. It inventories real data, groups master files
with their sidecars, chooses target disks, writes a typed manifest, inspects
current file state, watches for newly completed groups, and proves what would
happen without mutating the source folder.

## DT-1: Plan a Multi-Disk Ptycho Session

**User story**: A microscopist has 60-80 time-resolved 4D-STEM masters for
future joint iterative ptychography and wants to split the session across fast
NVMe disks before loading it into two or more NVIDIA GPUs.

**Primary tools**: `plan_data_transfer`, `write_data_transfer_manifest`,
`DataTransfer` widget, `quantem data-transfer plan`, and
`ShowFolder(...).data_transfer(...)` launcher.

**Data to use**: Real private 4D-STEM master files on a workstation. Prefer a
folder with 512 x 512 scan and 192 x 192 detector files. Keep raw data outside
the repository. Synthetic files are only acceptable for unit tests of planner
logic.

**Acceptance checks**:

- The planner discovers every `*_master.h5` file without decompressing detector
  frames or allocating GPU memory.
- Each master stays grouped with matching `*_data_*.h5` sidecar files.
- The dry-run output reports source disk, target disk, bytes per acquisition,
  total bytes, and bytes per target folder.
- The visual widget presents the same plan as a state table, not only terminal
  text: logical acquisition, source path, target path, source location, target
  location, bytes, and status.
- The widget presents a top-level session summary, target balance cards,
  backend/disk/GPU status, loader controls (`det_bin`, dtype, GPU list, page
  budget), safe action buttons, dataset readiness rows, and file detail rows.
- The default strategy balances by group size, not by file count.
- The planner writes a JSON manifest that the widget, CLI, or ptycho pipeline
  can read without rerunning discovery.
- Every planned acquisition has a logical ID, original master path, target
  master path, source disk, target disk, and grouped physical files.
- The dry-run does not create target files.
- If all current files live on one disk, the report says that data transfer is
  needed to prove multi-disk load bandwidth.

## DT-2: Copy Safely After Review

**User story**: A user has reviewed the data-transfer plan and wants the files copied
without corrupting partially written targets.

**Primary tools**: `copy_data_transfer`, `inspect_data_transfer`,
`DataTransfer` widget, `quantem data-transfer copy --execute`.

**Data to use**: A small real-derived folder for local tests and real private
master files for workstation signoff.

**Acceptance checks**:

- Copy is the default action. Move is not available until resume, verification,
  and user confirmation are implemented.
- Files copy to `*.partial` siblings first, then atomically replace the final
  path after verification.
- Existing targets with matching size are reported as already present.
- Existing targets with a different size fail with a corrective error unless
  the user explicitly allows overwrite.
- Size verification is available immediately; opt-in SHA-256 verification catches
  same-size corruption when the extra full-file reads are acceptable.
- The result table reports `planned`, `copied`, or `exists` for each physical
  file.
- `inspect_data_transfer(plan)` reports reloadable state for each physical file:
  `not-started`, `partial`, `exists`, `mismatch`, or `missing-source`.
- The widget state table is derived from the manifest plus filesystem state, not
  from terminal output.

## DT-3: Resume and Verify a Long Transfer Run

**User story**: A 60-80 frame data-transfer job is interrupted by a notebook restart,
network hiccup, or disk quota problem. The user wants to resume without
guessing which masters are safe.

**Primary tools**: typed manifest reload, `filter_data_transfer_plan`,
`summarize_data_transfer`, CLI inspect, and widget status panel.

**Data to use**: Real private multi-master session, transferred into at least two
target folders.

**Acceptance checks**:

- The manifest records logical session name, logical acquisition IDs, source
  paths, target paths, file sizes, target disks, action, and strategy.
- Rerunning the plan detects existing completed files and skips them when they
  match verification.
- Stale `*.partial` files are reported separately from completed files.
- The resume report names missing, mismatched, and completed groups.
- A filtered plan can select only `not-started` and `partial` files for retry.
- The report is concise enough to paste into a notebook or issue.

## DT-4: Open the Transferred Session for Browsing

**User story**: After data transfer, the user wants to immediately browse the
session without remembering another API.

**Primary tools**: DataTransfer widget, ShowFolder, Show4DSTEM.

**Data to use**: Transferred real private 4D-STEM masters spread across two target
folders.

**Acceptance checks**:

- The DataTransfer UI exposes an "Open in ShowFolder" path after a successful
  copy.
- ShowFolder can browse the transferred target roots or a manifest-backed virtual
  session.
- `open_show4dstem(gpus=[0, 1], det_bin=1, dtype="u8")` loads lazily and keeps
  GPU memory bounded while the user flips through datasets.
- Unit tests assert that DataTransfer -> Show4DSTEM loads only the first
  transferred target master initially; later datasets load on demand.
- The report distinguishes data-transfer time from widget load time and browser
  interaction FPS.

## DT-5: Live Acquisition Append

**User story**: A microscope or processing pipeline writes new masters while the
user is transferring and browsing. The user wants complete files to be copied
and visible without preloading everything.

**Primary tools**: `update_data_transfer_plan`, DataTransfer `rescan()`,
notebook watch controls, ShowFolder watch, DataTransfer widget.

**Data to use**: A folder where ready and incomplete `*_master.h5` groups can be
added during the test.

**Acceptance checks**:

- Newly discovered masters are ignored until they pass readiness checks.
- Complete new groups are appended to the existing manifest. Existing logical
  IDs keep their target assignments; new groups are assigned by the same
  strategy as the original plan.
- Active ShowFolder and Show4DSTEM handoffs can refresh after transfer completes.
- Removed or failed files do not remain in memory or manifest state as valid
  acquisitions.
- Repeated rescans do not duplicate skipped/not-ready groups.
- A live session can show "new groups", last scan time, watch state, and last
  copy/open timing without opening every dataset hot.

## CLI and Widget Split

The core data-transfer planner should be the source of truth.

- **Core utilities** own discovery, grouping, disk layout, planning, state
  inspection, manifest serialization, copy verification, and later resume/hash
  logic.
- **CLI** owns reproducible batch execution for overnight or ptycho-preparation
  workflows. It should print the same plan table the widget shows and write the
  manifest by default.
- **Widget** owns human review and durable state display: disk balance bars,
  backend/GPU status, logical acquisition table, source and target paths,
  incomplete file warnings, copy progress, error rows, loader choices, watch
  controls, and buttons to open the transferred session in ShowFolder or
  Show4DSTEM.
- **ShowFolder integration** should be a launch point, not core file movement:
  `ShowFolder(path).data_transfer(...)` can create the DataTransfer widget, but
  ShowFolder itself should remain a browser and not silently mutate lab data.

## Required Initial Test Commands

Run these before committing data-transfer planner changes:

```bash
PYTHONPATH=src pytest -q tests/test_io_data_transfer.py
PYTHONPATH=src pytest -q tests/test_cli.py::test_data_transfer_cli_plan_inspect_copy
PYTHONPATH=src pytest -q tests/test_showfolder.py::test_show_folder_launches_data_transfer_without_copying
PYTHONPATH=src pytest -q tests/test_show4dstem_paging.py::test_data_transfer_open_show4dstem_is_lazy_after_initial_frame
PYTHONPATH=src pytest -q tests/test_hdf5_disk_scheduling.py
```

When the CLI and widget wrappers exist, add a real-data local-only report that
records source disks, target disks, copied bytes, elapsed copy time, verification
mode, and the first Show4DSTEM load/flip timing from the transferred layout.
