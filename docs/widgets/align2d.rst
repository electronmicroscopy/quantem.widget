Align2D
=======

Interactive image alignment widget with phase correlation and manual adjustment.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Align2D

   img_a = np.random.rand(256, 256).astype(np.float32)
   img_b = np.random.rand(256, 256).astype(np.float32)
   Align2D(img_a, img_b, title="Alignment")

   # With calibration
   Align2D(img_a, img_b, pixel_size=2.4, cmap="viridis")

Features
--------

- **Auto-alignment** -- Phase correlation with Tukey windowing and DFT refinement
- **Manual mode** -- Drag or joystick to fine-tune translation
- **Overlay blend** -- Adjustable opacity for visual comparison
- **Side panels** -- Show both images side-by-side
- **FFT toggle** -- Compare Fourier transforms
- **Scale bar** -- Calibrated when ``pixel_size`` is set

Methods
-------

.. code-block:: python

   w = Align2D(img_a, img_b)

   # Replace both images (re-runs auto-alignment)
   w.set_images(new_a, new_b, auto_align=True)

   # Access alignment result
   dx, dy = w.offset
   print(f"NCC: {w.ncc_aligned:.4f}")

   # Reset alignment to zero offset
   w.reset_alignment()

Control Groups
--------------

.. code-block:: python

   # Lock groups (visible but non-interactive)
   w = Align2D(
       img_a, img_b,
       disable_alignment=True,
       disable_display=True,
       disable_export=True,
   )

   # Hide groups entirely
   w = Align2D(
       img_a, img_b,
       hide_stats=True,
       hidden_tools=["export"],
   )

State Persistence
-----------------

.. code-block:: python

   w = Align2D(img_a, img_b, cmap="viridis", pixel_size=2.4)

   w.summary()          # Print human-readable state
   w.state_dict()       # Get all settings as a dict
   w.save("state.json") # Save versioned envelope JSON file

   # Restore from file or dict
   w2 = Align2D(img_a, img_b, state="state.json")

Examples
--------

- :doc:`Simple demo </examples/align2d/align2d_simple>`
- :doc:`All features </examples/align2d/align2d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Align2D` for full documentation.
