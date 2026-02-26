Show2D
======

Static 2D image viewer with optional FFT, histogram analysis, and gallery mode.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Show2D

   # Single image
   image = np.random.rand(256, 256)
   Show2D(image, cmap="inferno")

   # Gallery of images
   images = [np.random.rand(256, 256) for _ in range(6)]
   Show2D(images, labels=["A", "B", "C", "D", "E", "F"], ncols=3)

Features
--------

- **Gallery mode** — Display multiple images side-by-side with configurable columns
- **FFT** — Toggle Fourier transform display with ``show_fft=True``
- **Histogram** — Intensity histogram with adjustable contrast
- **Scale bar** — Calibrated scale bar when ``pixel_size`` is set
- **Log scale** — Logarithmic intensity scaling with ``log_scale=True``
- **Auto contrast** — Percentile-based contrast with ``auto_contrast=True``
- **File/folder loading** — Use ``IO.file()`` / ``IO.folder()`` for any EM format
- **Tool lock/hide** — ``disable_*`` / ``hide_*`` API for shared read-only workflows

File Loading
------------

.. code-block:: python

   from quantem.widget import IO, Show2D

   # Single file (any format: PNG, TIFF, EMD, DM3/DM4, MRC, SER, NPY)
   Show2D(IO.file("data/frame.dm4"))

   # Folder of images
   Show2D(IO.folder("data/png_stack", file_type="png"))

   # Multiple files
   Show2D(IO.file(["frame1.tiff", "frame2.tiff"]))

Control Groups
--------------

.. code-block:: python

   # Lock interactions but keep controls visible
   w_locked = Show2D(
       image,
       disable_view=True,
       disable_navigation=True,
       disable_export=True,
       disable_roi=True,
   )

   # Hide selected control groups completely
   w_clean = Show2D(
       image,
       hide_histogram=True,
       hide_stats=True,
       hide_export=True,
   )

State Persistence
-----------------

.. code-block:: python

   w = Show2D(image, cmap="viridis", log_scale=True)

   w.summary()          # Print human-readable state
   w.state_dict()       # Get all settings as a dict
   w.save("state.json") # Save versioned envelope JSON file

   # Restore from file or dict
   w2 = Show2D(image, state="state.json")

Loader Troubleshooting
----------------------

- **Error: ``h5py is required to read .emd files``**
  Install `h5py` in your environment, then retry.
- **Mixed folder with different formats**
  Use ``IO.folder(..., file_type="png")`` or ``file_type="tiff"`` to force one file family and avoid accidental mixing.

Examples
--------

- :doc:`Simple demo </examples/show2d/show2d_simple>`
- :doc:`All features </examples/show2d/show2d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Show2D` for full documentation.
