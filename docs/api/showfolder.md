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

```python
w = ShowFolder("/data/session", thumb=256, group_by="fov")
w.cache_info
# {'enabled': True, 'hits': ..., 'misses': ..., 'path': ...}
```

Use `cache="folder"` when you want a project-local cache under
`.quantem/showfolder-cache`, `cache_dir="/fast/ssd/cache"` for a shared SSD
cache, `rebuild_cache=True` to force regeneration, or `cache=False` for a
read-only/no-cache browser.

For large folders, warm the cache before opening the browser UI:

```python
prebuild_showfolder_cache("/data/session", thumb=256, cache_dir="/fast/ssd/cache")
w = ShowFolder("/data/session", thumb=256, cache_dir="/fast/ssd/cache")
```

For folders that are still being written, start the watcher. It polls for new,
changed, or removed files; reuses cached thumbnails for unchanged files; updates
the displayed browser in place; and shows a small status line such as
`2 cached · 1 read · 1 new`.

```python
w = ShowFolder("/data/live-session", thumb=256, cache_dir="/fast/ssd/cache")
w.watch(interval=2.0)
```

Use `watch_once()` in scripts or tests when you want one deterministic poll,
and `stop_watch()` before shutting down a long-running notebook kernel.

`ShowFolder` owns folder watching. `Show2D`, `Show3D`, and `Show4DSTEM` stay as
display widgets: open them from the selection panel, then let `ShowFolder`
refresh the active view as files arrive. Image selections update the existing
Show2D/Show3D viewer in place. For 4D-STEM folders, the watcher also tracks
`*_master.h5` files and rebuilds the active lazy Show4DSTEM view with the same
paging options, so new Dataset4DSTEM masters appear without preloading the whole
folder.

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
