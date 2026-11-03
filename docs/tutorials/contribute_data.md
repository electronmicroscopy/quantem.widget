# Contribute tutorial data

First check the existing inputs in
[bobleesj/quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data).
Viewer collections use `widget/`; reusable images and diffraction scans stay
under `haadf/` and `4dstem/`. Reusing a dataset for another tutorial does not
require a new upload. The current canonical-path migration adds no new data.

Hugging Face stores scientific inputs, minimal cards and numerical metadata.
Keep notebooks, Python helpers, models and result reports in their owning source
repositories. New data contributions require permission and accurate provenance;
follow the dataset card and review the exact file list. Contribution validation
belongs in the `quantem.data` source checkout that provides it, not in a Python
module downloaded from the dataset. Do not assume an unpublished validator or
qualified-download API is available in an installed release.

When a notebook should load an input by name, the widget source change includes:

- helper in `src/quantem/widget/datasets.py`
- list it on [Tutorial Datasets](../api/datasets.md)
- use it in the tutorial
- record its canonical folder and immutable Hugging Face revision
- link any associated data change

See [Tutorial Datasets](../api/datasets.md) for the current pin. An explicit
older revision must resolve all files within that revision, without fetching
missing files from a newer snapshot.

A new widget is discuss-first. See [Pull requests](../maintainer/pull-requests.md).
Do not commit `.npy` or `.emd` here.
