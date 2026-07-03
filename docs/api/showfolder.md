# ShowFolder

Use `ShowFolder` when you want a notebook widget for browsing an electron
microscopy session folder before loading the heavy data. It wraps the existing
folder survey workflow with a `Show2D` / `Show3D`-style public name.

```python
from quantem.widget import ShowFolder, show_folder

w = ShowFolder("/data/session", group_by="fov")
w
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
```

For the full visual workflow, see the [session survey tutorial](../tutorials/survey).

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.showfolder.ShowFolder
   :members:

.. autofunction:: quantem.widget.showfolder.show_folder
```
