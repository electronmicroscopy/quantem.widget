# Tutorial Datasets

The `quantem.widget.datasets` module provides small, named examples for
tutorials, smoke tests, and Colab notebooks. Users choose the widget/example and
an explicit size; the loader handles Hugging Face paths, cache layout, and
calibration details.

```python
from quantem.widget.datasets import (
    show1d_ducky,
    show2d_gold,
    show3d_gold,
    show4dstem_gold,
    showdiffraction_fe3o4,
    showfolder_gold,
)
```

All size selectors use the same language:

| Size | Intended use |
|---|---|
| `small` | Documentation, Colab, CI smoke tests, quick first view |
| `medium` | Better visual detail for local notebooks |
| `large` | Local workstation review |
| `full` | Full available tutorial source where practical |

For Show1D, the ducky ptychography example is a file-backed monitor run:

```python
from quantem.widget import Show1D
from quantem.widget.datasets import show1d_ducky

run = show1d_ducky(size="small")
widget = Show1D.from_monitor_file(
    run / "show1d_monitor.jsonl",
    title="Real ducky joint iterative ptychography",
    x_label="frame",
    y_label="final loss",
    log_scale=False,
)
widget
```

Use the one-line example API when you only need the viewer:

```python
from quantem.widget import Show1D

widget = Show1D.from_example("ducky", size="small")
widget
```

The public Hugging Face dataset
([bobleesj/quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data))
is organized under `widget-tutorials/`. Reused sources live once under
`shared/`; widget-specific monitor runs or session folders live under the
widget name. Upload and download commands are on that dataset card. The
GitHub loader pull request is in
[Contribute tutorial data](../tutorials/contribute_data.md).

```text
widget-tutorials/{widget-or-shared}/{example}/{size}/...
```

Current tutorial payloads:

```text
widget-tutorials/show1d/ducky/small/show1d_monitor.jsonl
widget-tutorials/show1d/ducky/small/snapshots/*.npy
widget-tutorials/shared/gold-haadf/full/data.npy
widget-tutorials/show4dstem/gold-128-bin8/full/data.npy
widget-tutorials/show4dstem/gold-512-bin4/full/data.npy
widget-tutorials/show4dstem/gold-512-bin8/full/data.npy
widget-tutorials/showfolder/gold-haadf-session/small/*.emd
```

`showdiffraction_fe3o4()` expects
`widget-tutorials/showdiffraction/fe3o4-saed/small/data.npy`. That folder is
not in the current hub snapshot.

This keeps widget tutorial payloads grouped together instead of placing many
example files at the top level of the shared dataset repository. `show2d_gold`
and `show3d_gold` intentionally share `widget-tutorials/shared/gold-haadf/full`
so the same HAADF source image is not duplicated per widget.

## Reference

```{eval-rst}
.. automodule:: quantem.widget.datasets
   :members:
```
