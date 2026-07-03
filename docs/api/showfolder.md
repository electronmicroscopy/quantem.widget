# ShowFolder

Use `ShowFolder` when you want a notebook widget for browsing an electron
microscopy session folder before loading the heavy data. It wraps the existing
folder survey workflow with a `Show2D` / `Show3D`-style public name.

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
read-only/no-cache survey.

For large folders, warm the cache before opening the browser UI:

```python
prebuild_showfolder_cache("/data/session", thumb=256, cache_dir="/fast/ssd/cache")
w = ShowFolder("/data/session", thumb=256, cache_dir="/fast/ssd/cache")
```

If you do not pass a path, `ShowFolder()` displays a folder chooser first:

```python
w = show_folder()
w
```

After users star image panels or select EDS entries, the Python object exposes
the selected files and containing folders:

```python
w.paths()
w.selected_folders()
w.save("session-selection.quantem-survey.json")
w.clear_cache()
```

To share the folder browser without a live kernel, export standalone HTML:

```python
w.export_html("session-showfolder.html")
```

The older `survey(path, ...)` function remains available for compatibility, but
new notebooks should prefer `ShowFolder(path, ...)`.

For the full visual workflow, see the [session survey tutorial](../tutorials/survey).

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.showfolder.ShowFolder
   :members:

.. autofunction:: quantem.widget.showfolder.show_folder

.. autofunction:: quantem.widget.showfolder.prebuild_showfolder_cache
```
