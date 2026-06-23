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
