Show3D
======

Interactive 3D stack viewer with playback, multi-ROI analysis, FFT, and export.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Show3D

   stack = np.random.rand(50, 256, 256)
   Show3D(stack, title="My Stack", fps=10, cmap="gray")

   # With timestamps
   defocus = np.linspace(-60, 60, 50)
   Show3D(
       stack,
       labels=[f"C10={d:.0f} nm" for d in defocus],
       pixel_size=2.5,
       timestamps=defocus.tolist(),
       timestamp_unit="nm",
   )

Features
--------

- **Playback** — Play/pause/stop with configurable FPS, loop, boomerang modes
- **ROI analysis** — Multiple ROIs (circle, square, rectangle, annular) with live stats
- **ROI interaction** — Click empty image to add ROI at cursor, Delete selected, duplicate selected, hover edge to resize
- **ROI legend** — Per-ROI visibility, lock state, focus toggle, and live stats for quick comparison
- **Focus mode** — Adjustable dim strength outside focused ROIs
- **FFT** — Toggle Fourier transform display
- **Histogram** — Live intensity histogram
- **Scale bar** — Calibrated scale bar when ``pixel_size`` is set
- **Export** — Figure/frame PNG, GIF, PNG ZIP, and one-click bundle export (PNG + ROI CSV + state JSON)
- **File/folder loading** — Build stacks from EMD/PNG/TIFF files with explicit loader functions
- **Tool customization** — Disable or hide control groups (including playback)

Methods
-------

.. code-block:: python

   w = Show3D(stack)

   # Build from explicit loaders (recommended)
   w = Show3D.from_emd("data/scan.emd", dataset_path="/data/signal")
   w = Show3D.from_tiff("data/focal_series.tiff")
   w = Show3D.from_png("data/frame_0000.png")
   w = Show3D.from_folder("data/png_stack", file_type="png")
   w = Show3D.from_folder("data/tiff_stack", file_type="tiff")
   w = Show3D.from_folder("data/emd_stack", file_type="emd", dataset_path="/data/signal")

   # Playback
   w.play()
   w.pause()
   w.stop()
   w.goto(25)  # Jump to frame 25
   w.set_playback_path([0, 10, 20, 10, 0])  # Custom frame order

   # ROI analysis (multi-ROI)
   w.add_roi(shape="circle")                 # add at center
   w.add_roi(row=96, col=128, shape="annular")
   w.roi_selected_idx = 0                    # edit selected ROI
   w.set_roi(row=128, col=128, radius=20)    # helper for selected ROI
   w.roi_circle(radius=15)
   w.roi_square(half_size=10)
   w.roi_rectangle(width=20, height=10)
   w.roi_annular(inner=5, outer=15)
   w.duplicate_selected_roi()
   w.delete_selected_roi()
   w.clear_rois()
   w.roi_list
   w.roi_stats

   # Line profile
   w.set_profile((10, 10), (200, 200))
   w.clear_profile()

   # Export from toolbar: Figure/PNG/GIF/ZIP/Bundle

Control Groups
--------------

.. code-block:: python

   # Lock groups (visible but non-interactive)
   w = Show3D(
       stack,
       disable_display=True,
       disable_playback=True,   # disable_navigation also works
       disable_roi=True,
   )

   # Hide groups entirely
   w = Show3D(
       stack,
       hide_histogram=True,
       hide_stats=True,
       hidden_tools=["profile"],
   )

State Persistence
-----------------

.. code-block:: python

   w = Show3D(stack, cmap="gray", fps=12, boomerang=True)
   w.bookmarked_frames = [0, 15, 29]

   w.summary()          # Print human-readable state
   state = w.state_dict()  # Snapshot full state
   w.save("state.json") # Save versioned envelope JSON file

   # Restore in three ways
   w.load_state_dict(state)               # 1) apply dict in-place
   w2 = Show3D(stack, state="state.json") # 2) from saved file
   w3 = Show3D(stack, state=state)        # 3) from dict at init

Examples
--------

- :doc:`Simple demo </examples/show3d/show3d_simple>`
- :doc:`All features </examples/show3d/show3d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Show3D` for full documentation.
