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

## Google Colab

Each tutorial notebook can open directly in Colab from the badge at the top of
the notebook. Colab uses the same files that build these docs, so there is no
separate Colab copy to maintain.

If the package is not already available in the Colab runtime, run this once near
the top of the notebook:

```bash
%pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    quantem.widget
```

Common entry points:

| Tutorial | Colab | Source notebook |
|---|---|---|
| Example Data | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/download_data.ipynb) | [GitHub](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/download_data.ipynb) |
| Folder survey | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/survey.ipynb) | [GitHub](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/survey.ipynb) |
| FOV survey | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/survey_fov.ipynb) | [GitHub](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/survey_fov.ipynb) |
| Show2D | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/show2d.ipynb) | [GitHub](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/show2d.ipynb) |
| Show3D | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/show3d.ipynb) | [GitHub](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/show3d.ipynb) |
| Show3DSlices | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/show3dslices.ipynb) | [GitHub](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/show3dslices.ipynb) |
| Show4DSTEM | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/show4dstem.ipynb) | [GitHub](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/show4dstem.ipynb) |
| ShowEDS | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/showeds.ipynb) | [GitHub](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/showeds.ipynb) |
| ShowDiffraction | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/showdiffraction.ipynb) | [GitHub](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/showdiffraction.ipynb) |
| Saving and sharing | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/widget_export.ipynb) | [GitHub](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/widget_export.ipynb) |

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
