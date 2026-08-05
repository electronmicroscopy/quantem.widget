# ShowFolder

Use `ShowFolder` when you want a notebook widget for browsing an electron
microscopy session folder before loading the heavy data. `ShowFolder` is the
public folder-level API, matching the `Show2D` / `Show3D` naming style.

```python
from quantem.widget import ShowFolder, prebuild_showfolder_cache, show_folder

w = ShowFolder("/data/session", group_by="fov")
w
```

`ShowFolder` is designed as a look-before-you-load browser. The first run scans
the folder, builds calibrated thumbnails, and writes a small thumbnail/index
cache in the user cache directory by default. The second run validates file
path, size, and modification time, then reuses the cached thumbnails so users
can browse large microscopy sessions quickly without loading full arrays again.
The inventory includes per-pixel dwell time in microseconds when Velox records
``Scan.DwellTime``. AutoExport TIFFs inherit it only from an exact matching EMD
stem in the same folder or its parent; unmatched TIFFs remain blank.

```python
w = ShowFolder("/data/session", thumb=256, group_by="fov")
w.cache_info
# {'enabled': True, 'hits': ..., 'misses': ..., 'path': ...}
```

Use `cache="folder"` when you want a project-local cache under
`.quantem/showfolder-cache`, `cache_dir="showfolder-cache"` for a shared cache
cache, `rebuild_cache=True` to force regeneration, or `cache=False` for a
read-only/no-cache browser.

For large folders, warm the cache before opening the browser UI:

```python
prebuild_showfolder_cache("session", thumb=256, cache_dir="showfolder-cache")
w = ShowFolder("session", thumb=256, cache_dir="showfolder-cache")
```

For folders that are still being written, start the watcher. It polls for new,
changed, or removed files; reuses cached thumbnails for unchanged files; updates
the displayed browser in place; and shows a small status line such as
`2 cached · 1 read · 1 new`.

```python
w = ShowFolder("live-session", thumb=256, cache_dir="showfolder-cache")
w.watch(interval=2.0)
```

Use `watch_once()` in scripts or tests when you want one deterministic poll,
and `stop_watch()` before shutting down a long-running notebook kernel.

`ShowFolder` owns watching for its thumbnail browser and selection workflow.
When a viewer should follow scientific source data directly, use
`Show2D.from_folder(...)`, `Show3D.from_folder(...)`, or
`Show4DSTEM.from_folder(...)`; those APIs have their own default watcher and do
not depend on ShowFolder. Image selections opened through ShowFolder still
update its active Show2D/Show3D viewer in place. For 4D-STEM selections,
ShowFolder tracks `*_master.h5` files and refreshes the active lazy Show4DSTEM
view with the same paging options. The first dataset paints immediately; when
the complete known shape/dtype footprint fits the selected GPUs, the remaining
unhidden datasets preload in the background. Larger folders remain
full-resolution and lazy.

```python
stem = w.open_show4dstem(
    gpus=[0, 1],
    page_budget="auto",
    preload_all_if_fits=True,
)
stem.wait_for_dataset_preload(timeout=120)  # optional deterministic wait
```

Set `preload_all_if_fits=False` to force page-on-demand loading. The automatic
fit check does not silently bin or narrow data; `det_bin=` and `dtype=` remain
explicit scientific choices.

For acquisition folders where every new image should appear immediately, open
both all-image viewers before starting the watcher:

```python
w = ShowFolder("/data/live-session", thumb=256, group_by="none")
w.open_both(all_images=True)  # one live Show2D gallery and one live Show3D stack
w.watch(interval=2.0)
```

Use `open_show2d(all_images=True)` or `open_show3d(all_images=True)` when only
one view is useful. `Open All Both` in the selection panel performs the same
dual-view operation without Python calls.

The equivalent CLI shortcuts write that live notebook for you:

```bash
quantem show2d /data/live-session --watch
quantem show3d /data/live-session --watch
```

Folders with `*_master.h5` files also show a **4D-STEM master QC** table. This
is a header-only readiness check: it opens the HDF5 headers, verifies inline or
linked data chunks, and reports scan shape, detector shape, dtype, missing
chunks, and the next step. It does not decompress detector frames or allocate
GPU memory.

```python
w.master_qc_rows
# [{'file': 'scan_000_master.h5', 'status': 'ready', ...}, ...]
```

Use these rows to decide whether a master is ready for `Show4DSTEM` or whether
you should wait for sibling data files to finish copying. The status values are
`"ready"`, `"incomplete"`, and `"bad"`.

If you do not pass a path, `ShowFolder()` displays a folder chooser first:

```python
w = show_folder()
w
```

After users star image panels, the Python object exposes the selected files and
containing folders:

```python
w.paths()
w.selected_folders()
w.save("session-selection.quantem-showfolder.json")
w.clear_cache()
```

To share the folder browser without a live kernel, export standalone HTML:

```python
w.export_html("session-showfolder.html")
```

For the full visual workflow, see the [ShowFolder tutorial](../tutorials/showfolder).

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.showfolder.ShowFolder
   :members:

.. autofunction:: quantem.widget.showfolder.show_folder

.. autofunction:: quantem.widget.showfolder.prebuild_showfolder_cache
```
