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
- **File/folder loading** — Explicit loaders for PNG/TIFF/EMD files and folders
- **Tool lock/hide** — ``disable_*`` / ``hide_*`` API for shared read-only workflows

Methods
-------

.. code-block:: python

   # Explicit file loaders
   w = Show2D.from_png("data/frame.png")
   w = Show2D.from_tiff("data/stack.tiff")                       # gallery for multi-page TIFF
   w = Show2D.from_emd("data/scan.emd", dataset_path="/data/signal")

   # Folder loaders (explicit file_type required)
   w = Show2D.from_folder("data/png_stack", file_type="png")
   w = Show2D.from_folder("data/tiff_stack", file_type="tiff")
   w = Show2D.from_folder("data/emd_stack", file_type="emd", dataset_path="/data/signal")

   # Optional stack reduction to single 2D image
   w = Show2D.from_tiff("data/stack.tiff", mode="mean")
   w = Show2D.from_emd("data/scan.emd", dataset_path="/data/signal", mode="index", index=3)

Loader Decision Table
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 32 46

   * - If your input is...
     - Use this API
     - Why this is the safest choice
   * - One PNG image
     - ``Show2D.from_png("frame.png")``
     - Fastest explicit path for a single 2D frame.
   * - Folder of PNG slices
     - ``Show2D.from_folder("png_stack/", file_type="png")``
     - Keeps slice order explicit and avoids format ambiguity.
   * - One TIFF / multi-page TIFF
     - ``Show2D.from_tiff("stack.tiff")``
     - Handles both single-page and stacked TIFF inputs directly.
   * - EMD file where dataset location is known
     - ``Show2D.from_emd("scan.emd", dataset_path="/data/signal")``
     - Prevents loading the wrong dataset in complex EMD containers.
   * - Mixed-type folder where you want one type only
     - ``Show2D.from_folder("mixed_stack/", file_type="png")``
     - Forces one file family and prevents accidental cross-format merges.
   * - Stack source but you want one 2D view
     - ``Show2D.from_tiff(..., mode="mean")`` or ``mode="index"``
     - Creates a deterministic 2D reduction for quick QC snapshots.

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

- **Error: ``file_type is required for folder loading``**
  Use ``from_folder(path, file_type="png" | "tiff" | "emd")``.
- **Error: ``h5py is required to read .emd files``**
  Install `h5py` in your environment, then retry the same `from_emd(...)` call.
- **Error: ``dataset_path ... not found in EMD file``**
  Verify the exact HDF dataset path and pass it to ``dataset_path=...``; this is recommended for complex EMD files.
- **Mixed folder with different formats**
  Use ``from_folder(..., file_type="png")`` or ``file_type="tiff"`` to force one file family and avoid accidental mixing.

Examples
--------

- :doc:`Simple demo </examples/show2d/show2d_simple>`
- :doc:`All features </examples/show2d/show2d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Show2D` for full documentation.
