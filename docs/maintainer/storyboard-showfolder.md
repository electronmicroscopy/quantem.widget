# ShowFolder Storyboard

ShowFolder is the folder-level entry point for electron microscopy sessions. It
should help a user look before loading: choose a folder, scan the files, inspect
calibrated thumbnails and metadata, select the useful files, and save a compact
selection without embedding large raw data in the notebook.

This file is the canonical protocol for ShowFolder behavior, testing, and agent
signoff. Keep scratch notebooks, screenshots, and phone-test pages outside the
repository unless a release explicitly needs them as committed examples.

## SF-1: Choose a Folder

**User story**: A microscopist opens a notebook and wants to point ShowFolder at
a session folder without writing boilerplate path code.

**Primary widgets**: ShowFolder folder chooser or fallback path input.

**Data to use**: A small local microscopy-like folder with at least two image
files and one nested subfolder. Use real or real-derived files when available.

**Acceptance checks**:

- `ShowFolder()` renders a visible folder selection UI in JupyterLab.
- If `ipyfilechooser` is installed, the chooser can select a directory and run
  the folder browser.
- If `ipyfilechooser` is not installed, the fallback path field clearly shows
  the selected path and the `Open folder` button runs the same browser.
- Invalid paths show a useful error with the path that failed.
- The selected folder is remembered by key for the next widget creation.

## SF-2: Survey Real Session Files

**User story**: A user opens a real microscopy session and needs fast thumbnails,
file names, calibrated scale bars, and enough metadata to decide what to load.

**Primary widgets**: ShowFolder and image gallery rows.

**Data to use**: A real or real-derived session folder containing calibrated 2D
images. Include EDS files only to confirm ShowFolder inventories them without
opening the EDS viewer.

**Acceptance checks**:

- `ShowFolder(path, thumb=256, group_by="fov")` scans the folder and shows
  image thumbnails without loading full-resolution arrays into notebook state.
- Scale bars use calibrated sampling when available.
- File names, shapes, and errors are readable.
- EDS files are listed in the inventory when present, but ShowFolder does not
  create ShowEDS launchers or EDS selection controls.
- `*_master.h5` files are summarized in the 4D-STEM master QC table with
  status, scan shape, detector shape, dtype, chunk count, and a corrective next
  step. This is a header-only check; it must not decompress detector frames or
  allocate GPU memory.
- The UI remains usable when the folder contains many files.

## SF-3: Cache and Rerun

**User story**: The first scan may spend time reading files, but the second scan
should reuse thumbnails and metadata when files have not changed.

**Primary widgets**: ShowFolder cache.

**Data to use**: A folder with multiple image files and a temporary cache
directory.

**Acceptance checks**:

- First run reports cache misses and writes `manifest.json` plus
  `thumbnails.npz`.
- Second run reports cache hits and does not reread unchanged image files.
- Editing, replacing, or touching a file invalidates only that file's cached
  entry.
- `cache=False` runs without writing cache files.
- `cache="folder"` writes under `.quantem/showfolder-cache`.
- `clear_cache()` removes the active manifest and thumbnail bundle.

## SF-4: Watch a Live Folder

**User story**: A user is collecting or generating microscopy files while a
notebook is open and wants ShowFolder to show new results without rerunning the
cell or losing their current selection. When every new image matters, the user
also wants one normal control to keep both the all-image Show2D gallery and
Show3D stack visibly synchronized.

**Primary widgets**: ShowFolder watcher, cache status banner, image gallery,
and the ShowFolder-managed Show2D/Show3D preview handoffs. Direct
``Show2D.from_folder(...)``, ``Show3D.from_folder(...)``, and
``Show4DSTEM.from_folder(...)`` are separate full-resolution scientific-data
watchers covered by S2D-18, S3D-17, and S4D-14.

**Data to use**: A real or real-derived folder where files can be copied in and
removed during the test. Prefer lab data from an HPC/workstation or microscope
backend; keep the raw files outside the widget repository.

**Acceptance checks**:

- `ShowFolder(path).watch(interval=...)` inserts a visible accessible status
  line. Green ``Watching`` means the actual ShowFolder worker is alive;
  ``Updating`` appears only after a real arrival or removal is detected, not on
  idle polls. Use amber ``Waiting for file completion`` for an incomplete file,
  red ``Watch error`` with corrective detail for an unreadable source or worker
  failure, and gray ``Stopped`` or ``Not watching`` after stop/close or when
  liveness cannot be established.
- A direct Show2D, Show3D, or Show4DSTEM snapshot handoff created with
  ``watch=False`` has no watch badge. Restored notebook state, a static fallback,
  exported HTML, or another non-live ShowFolder snapshot must never show green
  ``Watching`` without an actually alive worker.
- Adding a file updates the already displayed browser in place.
- The status line reports cached versus newly read files.
- Existing stars/hidden panels are preserved when those files still exist.
- Removing a file removes it from the browser, selection, and next cache
  manifest so stale files are not retained in memory or cache state.
- New `*_master.h5` 4D-STEM files trigger the same watch path and refresh the
  active lazy Show4DSTEM handoff without preloading every master. This
  ShowFolder handoff may rebuild its owned viewer; it does not replace the
  same-widget guarantee required of direct ``Show4DSTEM.from_folder(...)``.
- When the all-image viewers are open (`open_show2d(all_images=True)` or
  `open_show3d(all_images=True)`), new readable image files append to the same
  Show2D/Show3D widget instance on the next poll.
- `Open All Both` and ``open_both(all_images=True)`` activate both all-image
  viewers through public paths. With both visible, background ``watch()`` and a
  genuine EMD arrival update the same two widget model IDs without private
  ``_active_selected_modes`` or ``_selected_*`` assignments.
- The browser test clicks the real control, copies an EMD after watching starts,
  and verifies both canvases repaint. Calling ``watch_once()`` and inspecting
  Python counts remains useful automation, but it is not the full user story.
- `watch_once()` performs one deterministic poll for tests and automation.
- `stop_watch()` stops the background watcher before kernel shutdown.

## SF-5: Select and Save

**User story**: A user reviews a session, marks useful files, and saves the
selection so analysis code can use the exact file list later.

**Primary widgets**: ShowFolder selection controls.

**Data to use**: A mixed folder with at least three images and one file that
should not be selected.

**Acceptance checks**:

- Users can select individual image files and folders from the rendered browser.
- `selected()`, `paths()`, `selected_paths()`, and `selected_folders()` return
  stable path information.
- `save(path)` writes a compact JSON selection.
- `load(path)` restores the selected files without rerunning expensive image
  loading.
- Saved selections use file paths and metadata, not large image buffers.

## SF-6: Open Selected Images Immediately

**User story**: A microscopist stars interesting images while browsing a folder
and wants to inspect them immediately without remembering a second API or
rewriting notebook boilerplate.

**Primary widgets**: ShowFolder selection panel, Show2D, Show3D.

**Data to use**: A real or real-derived microscopy folder with at least three
calibrated image files. Prefer files from an HPC/workstation or lab backend; do
not commit those raw files to the widget repository.

**Acceptance checks**:

- Starred images appear in the selection summary in acquisition order.
- `Open Show2D` renders a compact Show2D gallery below ShowFolder.
- `Open Show3D` renders the same starred images as a frame stack below
  ShowFolder.
- `Open Both` renders the starred images in both viewers, and `Open All Both`
  renders every readable image in both viewers through one normal user action.
- `Open All Show2D` and `Open All Show3D` render every readable image in the
  folder and live-append new files while the watcher is running.
- ``open_both(all_images=False)`` and ``open_both(all_images=True)`` expose the
  same selected/all behavior for notebook authors without requiring private
  mode manipulation.
- The explicit Python methods `show_selected()` and `show_selected_stack()`
  return the same widgets for notebook authors who prefer code.
- Scale bars and labels are preserved when all selected previews share a
  compatible calibration.
- The embedded viewers use cached thumbnail/preview data for immediate
  inspection and do not silently embed full-resolution raw acquisitions.
- ShowFolder keeps its maintained thumbnail cache as numeric arrays
  (`thumbnails.npz`) because selected Show2D/Show3D handoff needs array data.
  WebP thumbnails are only for human review pages and dashboards.
- ShowFolder owns watching for viewers opened from its cached preview and
  selection workflow. Direct ``Show2D.from_folder(...)`` and
  ``Show3D.from_folder(...)`` own independent full-resolution watchers and do
  not use ShowFolder thumbnails.

## SF-7: Notebook and Documentation Behavior

**User story**: A user can trust the docs and notebooks without accidentally
creating giant notebook files.

**Primary widgets**: ShowFolder, documentation examples.

**Data to use**: Documentation sample data and one real local session folder.

**Acceptance checks**:

- API docs explain the folder chooser, fallback path input, cache modes, and
  selection save/load workflow.
- Tutorial notebooks do not embed huge raw file buffers in saved widget state.
- Documentation examples use real or real-derived microscopy data whenever
  possible.
- Generated GitHub-preview notebooks are copied outside `docs/tutorials/`
  before static conversion.
- Agent signoff says whether browser testing used JupyterLab, exported HTML, or
  a static report.
- API docs point folder browsing to `ShowFolder`, not a separate `survey()`
  function.

## SF-8: Export a Shareable Folder Browser

**User story**: A user finishes browsing a folder and wants to send a lightweight
interactive browser to a collaborator without asking them to run a live kernel.

**Primary widgets**: ShowFolder, nested Show2D rows, HTML export.

**Data to use**: A real or real-derived microscopy folder with multiple grouped
rows and calibrated scale bars.

**Acceptance checks**:

- `ShowFolder(path).export_html("session-showfolder.html")` writes a standalone
  HTML file.
- Opening the exported file restores the nested Show2D rows, not only the outer
  container.
- The exported header identifies the folder, file count, thumbnail size, and
  cache status.
- The exported browser keeps thumbnails and interaction state compact; it does
  not embed full raw acquisition arrays.
- Exported HTML uses the same public export convention as Show2D and Show3D,
  with `export_html(path=None, title=None)` returning the written path.

## SF-9: Prebuild for High-Throughput Sessions

**User story**: A user has a large session on a workstation or SSD-backed
scratch path and wants the folder browser to open quickly during live analysis.

**Primary widgets**: `prebuild_showfolder_cache`, ShowFolder cache.

**Data to use**: A real or real-derived high-throughput session folder with many
microscopy files. Prefer 4K-class files when available; use smaller public data
only when the heavy data is unavailable.

**Acceptance checks**:

- `prebuild_showfolder_cache(path, cache_dir=...)` builds the same cache that
  `ShowFolder(path, cache_dir=...)` later reuses.
- A warm ShowFolder run reports cache hits and avoids re-reading unchanged image
  files.
- Cache validation uses relative path, file size, and modification time so
  edited or replaced files are refreshed.
- The notebook can point ShowFolder at a fast cache directory without moving the
  raw microscope data.
- The signoff reports cold and warm timing, file count, cache path, and whether
  the data was real, real-derived, or synthetic.

## Required Test Commands

Run these before committing ShowFolder changes:

```bash
PYTHONPATH=src pytest -q tests/test_showfolder.py tests/test_showfolder_core.py
PYTHONPATH=src:. python scripts/widget_showfolder_live_smoke.py --artifact-dir /tmp/quantem-widget-showfolder-live
```

The live smoke report must include WebP thumbnail previews and 4D-STEM master
QC rows. Those report images are visual evidence for humans; they are not the
cache used by ShowFolder widgets.

When a change affects docs or examples, also run the docs build used by the
current branch and open the rendered ShowFolder page in a browser.

## Signoff Report

Use this short report in commit notes, release notes, or agent handoff:

```text
ShowFolder signoff:
- Story IDs driven:
- Backend host:
- Browser/frontend:
- Folder path and file count:
- Cache mode/result:
- Selection save/load result:
- Tests run:
- Not verified:
```
