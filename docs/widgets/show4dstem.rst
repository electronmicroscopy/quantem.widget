Show4DSTEM
==========

Interactive 4D-STEM viewer with virtual imaging, ROI modes, and path animation.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Show4DSTEM

   data = np.random.rand(32, 32, 128, 128).astype(np.float32)
   w = Show4DSTEM(data, pixel_size=2.39, k_pixel_size=0.46)

Features
--------

- **Dual-panel viewer** — Diffraction pattern (left) and virtual image (right)
- **ROI modes** — Circle, square, rectangle, annular, and point ROIs
- **Auto-calibration** — Automatic BF disk center and radius detection
- **Path animation** — Animate scan position along custom paths or raster patterns
- **Virtual imaging** — Real-time BF, ABF, ADF virtual images
- **Scale bars** — Calibrated in angstroms (real-space) and mrad (k-space)
- **PyTorch acceleration** — GPU-accelerated virtual image computation
- **Quick view presets** — 3 save/load slots (UI buttons or keyboard ``1/2/3``, ``Shift+1/2/3``)

Control Groups
--------------

.. code-block:: python

   # Lock groups (visible but non-interactive)
   w = Show4DSTEM(
       data,
       disable_roi=True,
       disable_virtual=True,
       disable_frame=True,
       disable_playback=True,
   )

   # Hide groups entirely
   w = Show4DSTEM(
       data,
       hide_histogram=True,
       hide_stats=True,
       hide_fft=True,
       hidden_tools=["export"],
   )

Methods
-------

.. code-block:: python

   w = Show4DSTEM(data)

   # ROI modes
   w.roi_circle(radius=10)
   w.roi_annular(inner_radius=5, outer_radius=15)
   w.roi_square(half_size=8)
   w.roi_rect(width=20, height=10)
   w.roi_point()

   # Calibration
   w.auto_detect_center()

   # Path animation
   w.raster(step=2, interval_ms=50)
   w.set_path([(0, 0), (1, 0), (2, 0)], interval_ms=100)
   w.play()
   w.pause()
   w.stop()

State Persistence
-----------------

.. code-block:: python

   w = Show4DSTEM(data, center=(32, 32), bf_radius=9)
   w.dp_scale_mode = "log"

   w.summary()          # Print human-readable state
   state = w.state_dict()  # Snapshot full state
   w.save("state.json") # Save versioned envelope JSON file

   # Detector presets (built-in)
   w.apply_preset("bf")    # bright field
   w.apply_preset("abf")   # annular bright field
   w.apply_preset("adf")   # annular dark field
   w.apply_preset("haadf") # high-angle annular dark field

   # Save/load named presets (detector + display + export settings)
   w.save_preset("my_workflow", path="my_workflow.json")
   w.load_preset("my_workflow", path="my_workflow.json")

   # Restore in three ways
   w.load_state_dict(state)                  # 1) apply dict in-place
   w2 = Show4DSTEM(data, state="state.json") # 2) from saved file
   w3 = Show4DSTEM(data, state=state)        # 3) from dict at init

Programmatic Export
-------------------

Use ``save_image`` to export the current visualization state (including colormap,
scale mode, percentile range, ROI mode, frame index, and scan position):

.. code-block:: python

   w = Show4DSTEM(data)

   # Diffraction pattern at a specific scan position
   w.save_image(
       "dp.png",
       view="diffraction",
       position=(12, 8),
       include_overlays=True,
       include_scalebar=True,
   )

   # Virtual panel (PDF)
   w.save_image("virtual.pdf", view="virtual", frame_idx=3)

   # All visible analysis panels (DP + virtual + FFT when FFT is enabled)
   w.show_fft = True
   w.save_image("all_panels.png", view="all")

   # Sequence export (path / raster / frame sweeps) + one manifest JSON
   w.set_path([(4, 4), (8, 8), (12, 12)], autoplay=False)
   w.save_sequence("exports/path_run", mode="path", view="diffraction")

   # Figure template export (dp+vi, dp+vi+fft, publication presets)
   w.save_figure("figure.png", template="publication_dp_vi_fft")

   # Adaptive sparse path suggestion (coarse-to-fine utility planning)
   plan = w.suggest_adaptive_path(
       coarse_step=4,
       target_fraction=0.30,
       min_spacing=2,
       dose_lambda=0.25,
   )
   w.save_sequence("exports/adaptive", mode="path", path_points=plan["path_points"])

   # Session reproducibility report with exact export calls + output hashes
   w.save_reproducibility_report("exports/repro_report.json")

Accepted ``view`` values:

- ``"diffraction"``
- ``"virtual"``
- ``"fft"``
- ``"all"``

Built-in figure templates:

- ``"dp_vi"``
- ``"dp_vi_fft"``
- ``"publication_dp_vi"``
- ``"publication_dp_vi_fft"``

Sparse / Streaming Workflow APIs
--------------------------------

Show4DSTEM also exposes direct APIs for adaptive and streaming-style acquisition:

.. code-block:: python

   # Start with an empty scan buffer (same shape as your real dataset)
   w = Show4DSTEM(np.zeros_like(full_data), pixel_size=0.78, k_pixel_size=0.52)

   # Ingest one measured DP
   w.ingest_scan_point(row=12, col=9, dp=measured_dp, dose=1.0)

   # Ingest a block of measured DPs
   w.ingest_scan_block(rows=[1, 2, 3], cols=[4, 5, 6], dp_block=dp_stack)

   # Sparse checkpoint / restore
   sparse = w.get_sparse_state()
   w2 = Show4DSTEM(np.zeros_like(full_data))
   w2.set_sparse_state(sparse["mask"], sparse["sampled_data"])

   # Propose next adaptive points with budget constraints
   next_pts = w.propose_next_points(
       32,
       strategy="adaptive",
       budget={"max_total_fraction": 0.30, "min_spacing": 2, "dose_lambda": 0.25},
   )

   # Evaluate sparse reconstruction quality against full-raster reference
   report = w.evaluate_against_reference(reference="full_raster")
   print(report["metrics"])

   # Export one reproducibility bundle
   w.export_session_bundle("exports/session_bundle")

Batch CLI
---------

The non-interactive batch runner wraps the same export APIs:

.. code-block:: bash

   quantem-show4dstem-export \
     --input data_folder \
     --output-dir exports \
     --pattern "*.npy" \
     --mode frames \
     --frame-range 0:9 \
     --view all \
     --format png

   # Adaptive sparse planner + export
   quantem-show4dstem-export \
     --input sample.npy \
     --output-dir exports_adaptive \
     --mode adaptive \
     --adaptive-coarse-step 4 \
     --adaptive-target-fraction 0.30 \
     --adaptive-min-spacing 2 \
     --view all \
     --format png

Optional interactive prompts are available via ``--interactive``.

Examples
--------

- :doc:`Simple demo </examples/show4dstem/show4dstem_simple>`
- :doc:`All features </examples/show4dstem/show4dstem_all_features>`

API Reference
-------------

See :class:`quantem.widget.Show4DSTEM` for full documentation.
