# Tutorial Datasets

The `quantem.widget.datasets` module provides small, named examples for
tutorials, smoke tests, and Colab notebooks. Choose a named example; image and
scan previews also take an explicit size. The loader handles Hugging Face
paths, cache layout, and calibration details.

```python
from quantem.widget.datasets import (
    gold_session,
    show1d_ducky,
    show2d_gold,
    show3d_gold,
    show4dstem_gold,
    showpdf_ta,
    showpdf_ta_ground_truth,
)
```

Images and stacks come back as quantem core `Dataset2d` and `Dataset3d`
(`show2d_gold`, `show3d_gold`), and the 4D-STEM scan
as every 4D-STEM file reads (`read_4dstem`): a `quantem.gpu.io.Dataset4dstemGPU`
when quantem.gpu is installed and a GPU is present, else a quantem core
`Dataset4dstem` (`show4dstem_gold`). `showpdf_ta` returns a quantem core
`Dataset4dstem` with its detector calibrated in 1/Å, and
`showpdf_ta_ground_truth` the `(r, g)` arrays of its atomic model's g(r). On
an Intel Mac or Windows on ARM, which cannot install quantem core, the
widget's stand-in carries the same `array`, `name`, `sampling`, `units`,
`metadata`, `shape`, `ndim` and `dtype`.
`show1d_ducky` and `gold_session` return folders.

`show1d_ducky()` selects one collection and has no `size` argument. For helpers that expose a size selector:

| Size | Intended use |
|---|---|
| `small` | Documentation, Colab, CI smoke tests, quick first view |
| `medium` | Better visual detail for local notebooks |
| `large` | Local workstation review |
| `full` | Full available tutorial source where practical |

For Show1D, the synthetic Ducky example is a file-backed monitor run.

Ducky simulation tutorial: Georgios Varnavides, [*Trademark ducky dataset* (October 23, 2025)](https://github.com/electronmicroscopy/quantem-tutorials/blob/dbadc111734641226ab0454ed988feaf018e65c4/simulation_notebooks/ducky-dataset.ipynb). The viewer files contain subsequent reconstruction results for the synthetic example.

```python
from quantem.widget import Show1D
from quantem.widget.datasets import show1d_ducky

run = show1d_ducky()
widget = Show1D.from_monitor_file(
    run / "show1d_monitor.jsonl",
    title="Synthetic Ducky reconstructions (tutorial: Georgios Varnavides)",
    x_label="frame",
    y_label="final loss",
    log_scale=False,
)
widget
```

## Shared inputs and revisions

Each helper downloads one input family from
[bobleesj/quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data),
then makes its documented preview locally. The default is immutable revision
`5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20`, so later folder changes cannot silently alter an example.

| Tutorial / helper | Canonical input | Local operation |
| --- | --- | --- |
| Show2D, widget export / `show2d_gold` | `haadf/gold_haadf_npy` | Explicit image stride; sampling multiplied by that stride |
| Show3D / `show3d_gold` | Same HAADF input | Moving crops for demonstration, not measured time/depth frames |
| Show4DSTEM / `show4dstem_gold` | `4dstem/gold_128_npy_bin8` | Scan stride only; detector sampling and uint16 counts preserved |
| Image readers, folder watching / `gold_session` | `widget/gold_session` | Existing 26 EMDs; original acquisition names |
| Show1D, reconstruction comparison / `show1d_ducky` | `widget/ducky` | Monitor JSONL and every relative snapshot path retained |
| ShowPDF / `showpdf_ta`, `showpdf_ta_ground_truth` | `4dstem/ta_amorphous_sim` | None; simulated scan and its model's g(r), pinned at the revision that adds them |

The helpers use the canonical paths above. An explicit revision must contain
those paths; loaders do not search retired folders or another revision. To
reproduce an older folder layout, use its matching software and data revision.
The Ducky helper returns its one monitor collection. The gold session has one
published size (`small`); no additional measured data is generated.

`small` describes the returned preview, not necessarily the network transfer:
the shared HAADF source is about 33.55 MB and the reduced diffraction source is
about 18.88 MB. Show2D and Show3D download the same HAADF file once into the
Hub cache. `force_download=True` explicitly refreshes it; `cache_dir` selects
a cache location.

For general retrieval, `quantem.data` owns qualified IDs and revisions:

```python
import quantem.data as qdata

folder = qdata.download(
    "4dstem/gold_128_npy_bin8",
    revision="5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20",
)
```

This qualified/revision API requires the matching updated `quantem.data`
implementation. It is prepared locally; no compatible wheel release or Colab
installation is claimed here. Existing friendly helpers use the pinned canonical inputs
without that new resolver. Native detector acquisitions are opened with
`quantem.gpu.io.load`; the Ducky monitor and EMD image session are different
input types, not raw 4D-STEM acquisitions.

Show3DSlices, ShowCIF and Plot2D generate their small
teaching inputs locally. Keep those recipes in the software repository.
Use Hugging Face for reusable input data, numeric metadata, checksums, credit
and minimal data cards. Keep notebook code, helpers, training overrides,
checkpoints and rendered reports in their owning software or private results
repository. This migration reuses existing public inputs; it adds no new
scientific data or private files.
See [Contribute tutorial data](../tutorials/contribute_data.md).

## Diffraction example

The ShowDiffraction notebook uses a labeled deterministic synthetic magnetite
control. No measured Fe3O4 pattern is published, so there is no download helper
for that missing input. Open your own measured image with `ShowDiffraction`.

## Pair distribution function example

`showpdf_ta()` and `showpdf_ta_ground_truth()` read `4dstem/ta_amorphous_sim/`
(`data.npy`, `meta.json` and `ground_truth_gr.npz`) at revision
`fe907499a3a2c1f827c8827c0b3472c5b08f47bf`, the dataset commit that added that
folder (the next commit after the default revision above); pass `revision` to read
another. The scan is a simulation of an atomic model, not a measured
acquisition.

## Reference

```{eval-rst}
.. automodule:: quantem.widget.datasets
   :members:
```
