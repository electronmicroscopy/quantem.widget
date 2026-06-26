# Installation

`quantem.widget` is currently published on **TestPyPI** (pre-release). Install it from
there, with PyPI as the extra index so its dependencies (numpy, torch, ...) resolve
normally:

```bash
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    quantem.widget
```

That works on every backend; the widget picks the fastest path it finds at runtime.

## Backends

- **NVIDIA CUDA** - the universal Torch viewer runs on GPU. The integer-reduction
  detector path uses CuPy. We do not pin a CuPy wheel (a fixed `cuda12x`/`cuda13x`
  would collide with one your environment already ships); a real CUDA workflow
  already has the matching CuPy installed.
- **Apple Silicon (Metal / MPS)** - a dedicated raw-Metal viewer powers
  `Show4DSTEM` on the MacBook, with full-resolution CBED and a fast virtual-image
  path. The tiny `pyobjc-framework-Metal` wheel installs automatically on macOS.
- **CPU** - everything still runs, just slower. This is the path used to build
  these docs.

## Verify

```python
import quantem.widget as qw
print(qw.__version__)
print(qw.__all__)   # public widgets, load(), DPC helpers, detector helpers
```

## JupyterLab saved widgets

JupyterLab can reopen a saved notebook with interactive widget outputs without
rerunning the cell, as long as the JupyterLab environment has
`anywidget>=0.11.0` and `jupyterlab_widgets>=3.0.10`. The Lab-side `anywidget`
frontend must match the Python-side saved state version, and `jupyterlab_widgets`
3.0.10 bundles `@jupyter-widgets/jupyterlab-manager` 5.0.10, which fixed restoring
widget models from the notebook's saved `metadata.widgets` state.

If JupyterLab and the Python kernel live in different environments, install
`jupyterlab_widgets` in the environment that launches JupyterLab, and install
`quantem.widget`/`ipywidgets`/`anywidget` in the kernel environment:

```bash
python -m pip install "jupyterlab_widgets>=3.0.10"
python -m pip install "anywidget>=0.11.0"
```

`quantem jupyter` enables **Save Widget State Automatically** for the launched
JupyterLab session, so Cmd+S writes the widget state. If you start JupyterLab yourself,
enable that setting manually, save the notebook, close it, and reopen it. A saved
`Show2D` view should hydrate from the notebook file with no cell execution. Older Lab
widget-manager builds show `Error displaying widget: model not found` even when the
widget state is present in the notebook.

Make the final view while the notebook has a live kernel, then press Cmd+S. Model-backed
traits such as the visible frame, FFT/profile toggles, contrast limits, scale, colormap,
and saved view state are written to `metadata.widgets` by the widget manager. A
kernel-less reopened notebook preserves that saved state on later saves, but browser-only
changes made after reopening without a kernel are not a substitute for a live-kernel
save.
