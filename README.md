# quantem.widget

[![TestPyPI](https://img.shields.io/pypi/v/quantem-widget?pypiBaseUrl=https://test.pypi.org&label=TestPyPI)](https://test.pypi.org/project/quantem-widget/)

Interactive WebGPU visualization widgets for 4D-STEM and electron microscopy, in
Jupyter. Works with NumPy, PyTorch, or CuPy arrays.

> Prototype on [TestPyPI](https://test.pypi.org/project/quantem-widget/). Built on
> [`quantem`](https://github.com/electronmicroscopy/quantem) core.

## Install

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantem.widget
```

Verify:

```bash
python -c "import quantem.widget; print(quantem.widget.__version__)"
```

## Widgets

| Widget | Input | Shows |
|---|---|---|
| `Show2D` | 2D image or stack | image + contrast, FFT, line profiles, scale bar |
| `Show3D` | 3D stack | scrub / play through frames |
| `Show3DSlices` | 3D volume | orthogonal-slice viewer |
| `Show4DSTEM` | 4D-STEM array | live virtual detectors (BF / ABF / ADF), CoM / iCoM / DPC |

```python
import numpy as np
from quantem.widget import Show2D, Show3D, Show3DSlices, Show4DSTEM

Show2D(np.random.rand(512, 512))
Show4DSTEM(np.random.rand(64, 64, 128, 128))
```

## Load data

```python
from quantem.widget import load

data = load("scan_master.h5")   # Arina 4D-STEM .h5 -> GPU
Show4DSTEM(data)
```

`quantem.widget.io` also provides `survey`, `read_image`, `bin`, `download`, and
more - see the docs.

## Docs

https://bobleesj.github.io/quantem.widget/

## Issues

https://github.com/bobleesj/quantem.widget/issues
