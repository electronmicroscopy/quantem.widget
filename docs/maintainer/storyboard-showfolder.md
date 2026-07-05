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
cell or losing their current selection.

**Primary widgets**: ShowFolder watcher, cache status banner, image gallery.

**Data to use**: A real or real-derived folder where files can be copied in and
removed during the test. Prefer lab data from MJ Go or another backend; keep
the raw files outside the widget repository.

**Acceptance checks**:

- `ShowFolder(path).watch(interval=...)` inserts a visible status line.
- Adding a file updates the already displayed browser in place.
- The status line reports cached versus newly read files.
- Existing stars/hidden panels are preserved when those files still exist.
- Removing a file removes it from the browser, selection, and next cache
  manifest so stale files are not retained in memory or cache state.
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
calibrated image files. Prefer files from MJ Go or another lab backend; do not
commit those raw files to the widget repository.

**Acceptance checks**:

- Starred images appear in the selection summary in acquisition order.
- `Open Show2D` renders a compact Show2D gallery below ShowFolder.
- `Open Show3D` renders the same starred images as a frame stack below
  ShowFolder.
- The explicit Python methods `show_selected()` and `show_selected_stack()`
  return the same widgets for notebook authors who prefer code.
- Scale bars and labels are preserved when all selected previews share a
  compatible calibration.
- The embedded viewers use cached thumbnail/preview data for immediate
  inspection and do not silently embed full-resolution raw acquisitions.

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
```

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
