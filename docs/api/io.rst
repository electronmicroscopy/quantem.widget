IO
==

.. autoclass:: quantem.widget.io.IOResult
   :members:
   :undoc-members:

.. autoclass:: quantem.widget.io.IO
   :no-members:

.. rubric:: File Loading

.. automethod:: quantem.widget.io.IO.file
.. automethod:: quantem.widget.io.IO.folder
.. automethod:: quantem.widget.io.IO.supported_formats

.. rubric:: GPU-Accelerated Loading

.. automethod:: quantem.widget.io.IO.arina_file
.. automethod:: quantem.widget.io.IO.arina_folder

Examples
--------

Read a single file (any format):

.. code-block:: python

   from quantem.widget import IO, Show2D

   result = IO.file("gold_nanoparticles.dm4")
   print(result.pixel_size, result.units)  # 1.43 Å
   Show2D(result, show_fft=True, log_scale=True)

Read a folder of images as a stack:

.. code-block:: python

   from quantem.widget import IO, Show3D

   result = IO.folder("/path/to/focal_series/", file_type="dm4")
   print(result.data.shape)   # (20, 4096, 4096)
   print(result.labels)       # ['image_001', 'image_002', ...]
   Show3D(result, title="Focal Series")

Auto-detect file type (omit ``file_type``):

.. code-block:: python

   # Auto-detects from folder contents (raises if mixed types)
   result = IO.folder("/path/to/tiff_scans/")

Read multiple files into a stack:

.. code-block:: python

   result = IO.file([
       "sample_region_A.dm4",
       "sample_region_B.dm4",
       "sample_region_C.dm4",
   ])
   Show3D(result)

Merge multiple folders into one stack:

.. code-block:: python

   result = IO.folder([
       "/path/to/session_1/",
       "/path/to/session_2/",
   ], file_type="dm3")
   # All images across both folders stacked into one (N, H, W) array

Read 4D-STEM data:

.. code-block:: python

   result = IO.file("4dstem_binned.h5")
   print(result.data.shape)  # (256, 256, 128, 128)

   from quantem.widget import Show4DSTEM
   Show4DSTEM(result, title="4D-STEM")

IOResult duck typing
--------------------

``IOResult`` forwards NumPy array methods to the underlying ``.data`` array,
so you can use it directly in expressions:

.. code-block:: python

   result = IO.file("image.dm4")
   result.shape        # (1024, 1024)
   result.dtype        # float32
   result.mean()       # 0.42

   # Reduce a 4D-STEM dataset to a virtual bright-field image
   result = IO.arina_file("master.h5", det_bin=2)
   vbf = result.sum(axis=(2, 3))
   Show2D(vbf, title="Virtual Bright Field")

``print(result)`` gives a human-readable summary:

.. code-block:: text

   IOResult
     shape:      512 x 512 x 96 x 96
     dtype:      float32
     title:      SnMoS2s_001
     pixel_size: 1.4298 Å
     labels:     ['frame_001', 'frame_002', 'frame_003', ...] (20 total)
     metadata:   ['General', 'Signal']

IO.arina_file — GPU-accelerated 4D-STEM loading
------------------------------------------------

``IO.arina_file()`` decompresses bitshuffle+LZ4 data on the GPU via Apple Metal,
and optionally bins detector and/or scan axes on the fly.
(``IO.arina()`` still works as an alias for backward compatibility.)

.. code-block:: python

   from quantem.widget import IO

   # 2x2 detector binning (most common)
   data = IO.arina_file("master.h5", det_bin=2)

   # Auto-select bin factor based on available RAM
   data = IO.arina_file("master.h5", det_bin="auto")

   # Bin both detector and scan axes
   data = IO.arina_file("master.h5", det_bin=2, scan_bin=2)

   # Disable hot pixel filtering (on by default)
   data = IO.arina_file("master.h5", det_bin=2, hot_pixel_filter=False)

Performance benchmarks
^^^^^^^^^^^^^^^^^^^^^^

Benchmarked on SnMoS2 dataset (262,144 frames, 192×192 detector, Apple M5).
Steady-state times (second+ call — first call adds ~0.5s for JIT/Metal warmup):

.. list-table::
   :header-rows: 1
   :widths: 30 25 15 15

   * - Configuration
     - Output shape
     - Memory
     - Time
   * - ``det_bin=2``
     - 512 × 512 × 96 × 96
     - 9.0 GB
     - 1.8 s
   * - ``det_bin=4``
     - 512 × 512 × 48 × 48
     - 2.3 GB
     - 1.7 s
   * - ``det_bin=8``
     - 512 × 512 × 24 × 24
     - 0.6 GB
     - 1.8 s
   * - ``det_bin=2, scan_bin=2``
     - 256 × 256 × 96 × 96
     - 2.3 GB
     - 2.0 s
   * - ``det_bin=2, scan_bin=4``
     - 128 × 128 × 96 × 96
     - 0.6 GB
     - 2.0 s

The pipeline is double-buffered (CPU reads chunk N+1 while GPU decompresses chunk N).
The bottleneck is GPU decompression: 262k frames of bitshuffle+LZ4 takes ~1.5s on M5
regardless of bin factor. The 1.7 GB disk read (8.2 GB/s SSD) is fully hidden.

.. note::

   ``det_bin=1`` (no binning) for this dataset requires ~18 GB of contiguous GPU
   memory. Use ``det_bin="auto"`` to let IO pick the smallest bin factor that
   fits in available RAM.

IO.arina_folder — batch 5D-STEM loading
----------------------------------------

``IO.arina_folder()`` finds all ``*_master.h5`` files in a folder, loads each
with ``IO.arina_file()``, and stacks them into a 5D dataset (time/tilt series).

.. code-block:: python

   from quantem.widget import IO

   # Load all scans in a folder → 5D (n_files, scan_r, scan_c, det_r, det_c)
   result = IO.arina_folder("/path/to/session/", det_bin=8)
   print(result.data.shape)  # (10, 256, 256, 24, 24)

   # Incomplete files are auto-skipped with a warning
   # "SKIPPED: [Errno 2] ... data_000003.h5 ... No such file or directory"

   # View as 5D-STEM time series with frame slider
   from quantem.widget import Show4DSTEM
   Show4DSTEM(result, frame_dim_label="Scan")

Benchmarked on 12 Arina scans (65,536 frames each, 192×192 uint32 detector, Apple M5).
2 incomplete files auto-skipped, 10 loaded:

.. list-table::
   :header-rows: 1
   :widths: 30 25 15 15 15

   * - Configuration
     - Output shape
     - Memory
     - Load
     - + Show4DSTEM
   * - ``det_bin=8`` (10 files)
     - 10 × 256 × 256 × 24 × 24
     - 1.5 GB
     - 9.5 s
     - 11.0 s
   * - ``det_bin=4`` (10 files)
     - 10 × 256 × 256 × 48 × 48
     - 6.0 GB
     - 10.8 s
     - 16.3 s

Standard file loading performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Single files load in under 200 ms on any machine — no GPU required:

.. list-table::
   :header-rows: 1
   :widths: 30 25 15

   * - Format
     - Size
     - Time
   * - NPY
     - 1024 × 1024
     - 1 ms
   * - DM3
     - 4096 × 4096
     - 14 ms
   * - DM4
     - 4096 × 4096
     - 14 ms
   * - TIFF
     - 2049 × 2040
     - 41 ms
   * - PNG
     - 2048 × 2048
     - 45 ms
   * - EMD (Velox)
     - 2048 × 2048
     - 105 ms

Folder loading scales linearly:

.. list-table::
   :header-rows: 1
   :widths: 30 25 15

   * - Folder
     - Stack shape
     - Time
   * - 40 TIFFs (256×256)
     - 40 × 256 × 256
     - 43 ms
   * - 6 EMDs (2048×2048)
     - 6 × 2048 × 2048
     - 65 ms
   * - 3 PNGs (2048×2048)
     - 3 × 2048 × 2048
     - 117 ms
   * - 5 DM3s (4096×4096)
     - 5 × 4096 × 4096
     - 150 ms

Supported formats
-----------------

**Native** (no extra dependencies): PNG, JPEG, BMP, TIFF, EMD, HDF5, NPY, NPZ

**Via rosettasciio** (``pip install rosettasciio``): DM3, DM4, MRC, SER, and
`60+ more formats <https://hyperspy.org/rosettasciio/supported_formats/index.html>`_.

**GPU-accelerated**: Arina 4D-STEM master files (``IO.arina_file()``) — requires
``pyobjc-framework-Metal`` on macOS. CUDA and Intel GPU backends coming soon.
