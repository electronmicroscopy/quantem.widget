# quantem.widget

Interactive Jupyter widgets for electron microscopy visualization.
Works with NumPy, CuPy, and PyTorch arrays.

## Install (TestPyPI)

Currently published on [TestPyPI](https://test.pypi.org/project/quantem-widget/) (not yet on PyPI).

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantem-widget
```

## IO

Load any electron microscopy file with one line:

```python
from quantem.widget import IO

result = IO.file("image.dm4")         # single file (DM3/DM4/MRC/EMD/TIFF/PNG/NPY/...)
result = IO.folder("scans/")          # folder → stack
data   = IO.arina_file("master.h5", det_bin=2)  # GPU-accelerated arina 4D-STEM
```

See the [IO API reference](api/io) for full documentation and examples.

## widgets

| widget | description |
|--------|-------------|
| [Show1D](examples/show1d/show1d_simple) | 1D viewer for spectra, profiles, and time series |
| [Show2D](examples/show2d/show2d_simple) | 2D image viewer with gallery, FFT, histogram |
| [Show3D](examples/show3d/show3d_simple) | 3D stack viewer with playback, ROI, FFT, export |
| [Show3DVolume](examples/show3dvolume/show3dvolume_simple) | orthogonal slice viewer (XY, XZ, YZ) |
| [Show4D](examples/show4d/show4d_simple) | general 4D data viewer with dual navigation/signal panels |
| [Show4DSTEM](examples/show4dstem/show4dstem_simple) | 4D-STEM diffraction pattern viewer with virtual imaging |
| [ShowComplex2D](examples/showcomplex2d/showcomplex2d_simple) | complex-valued 2D viewer (amplitude/phase/HSV) |
| [Mark2D](examples/mark2d/mark2d_simple) | interactive point picker for 2D images |
| [Edit2D](examples/edit2d/edit2d_simple) | interactive crop/pad/mask editor |
| [Align2D](examples/align2d/align2d_simple) | image alignment overlay with phase correlation |
| [Bin](examples/bin/bin_simple) | calibration-aware binning + BF/ADF quality control |

```{toctree}
:maxdepth: 2
:hidden:

widgets/index
examples/index
api/index
dev/index
optimization
changelog
```
