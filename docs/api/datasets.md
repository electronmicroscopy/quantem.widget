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

The public Hugging Face dataset is organized under:

```text
widget-tutorials/{widget}/{example}/{size}/...
```

For the current ducky tutorial:

```text
widget-tutorials/show1d/ducky/small/show1d_monitor.jsonl
widget-tutorials/show1d/ducky/small/snapshots/*.npy
```

This keeps widget tutorial payloads grouped together instead of placing many
example files at the top level of the shared dataset repository.

## Reference

```{eval-rst}
.. automodule:: quantem.widget.datasets
   :members:
```
