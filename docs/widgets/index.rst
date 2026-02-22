Widgets
=======

quantem.widget provides twelve interactive widgets for electron microscopy
visualization and workflow operations.

Visualization widgets support:

- **NumPy**, **CuPy**, and **PyTorch** arrays as input
- Automatic **light/dark theme** detection
- **Zoom and pan** with mouse scroll and drag
- **Colormaps**: inferno, viridis, plasma, magma, hot, gray
- **State persistence** — ``summary()``, ``state_dict()``, ``save(path)``, ``state=`` constructor param

Workflow widgets:

- ``Bin`` for calibration-aware 4D-STEM binning
- ``Merge4DSTEM`` for stacking multiple 4D-STEM datasets along a time axis

.. toctree::
   :maxdepth: 1

   gallery
   show1d
   show2d
   show3d
   show3dvolume
   show4d
   show4dstem
   showcomplex
   mark2d
   edit2d
   align2d
   bin
   merge4dstem
   parity_matrix
