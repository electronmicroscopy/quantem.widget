Mark2D
=======

Interactive point picker for 2D images. Click to select atom positions, features, or lattice vectors.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Mark2D

   image = np.random.rand(256, 256)
   w = Mark2D(image, max_points=10)
   w  # display widget, click to pick points

   # Access selected points
   print(w.selected_points)

Features
--------

- **Point picking** — Click to place markers, click again to remove
- **Gallery mode** — Pick points on multiple images independently
- **Marker customization** — Shapes: circle, triangle, square, diamond, star
- **Undo/redo** — Undo and redo point selections
- **Max points** — Configurable maximum number of points per image
- **ROI center guides** — Active ROI shows dotted horizontal/vertical center lines
- **Tool locking** — Disable selected tools for shared/read-only workflows
- **Tool hiding** — Hide selected controls completely with ``hide_*`` flags

Methods
-------

.. code-block:: python

   w = Mark2D(image)
   w.set_image(new_image)
   w.set_image([img1, img2], labels=["A", "B"])

   # Disable selected tools for shared notebooks
   w = Mark2D(
       image,
       disable_points=True,
       disable_roi=True,
       disable_display=True,
   )

   # Fully lock everything
   w_read_only = Mark2D(image, disable_all=True)

   # Hide selected controls completely (also locks interaction)
   w_hidden = Mark2D(
       image,
       hide_display=True,
       hide_export=True,
   )

   # ROI geometry helpers (defaults to most recently added ROI)
   w.add_roi(128, 128, shape="circle", radius=20)
   center = w.roi_center()  # (row, col)
   radius = w.roi_radius()  # int for circle/square, None for rectangle
   size = w.roi_size()      # (width, height)
   w.clear_rois()

   # Line profile
   w.set_profile((10, 10), (200, 200))
   w.profile_values    # Float32Array of intensity along line
   w.profile_distance  # calibrated length
   w.clear_profile()

   # Point data access
   w.points_as_array()  # NumPy array of (row, col)
   w.points_as_dict()   # list of dicts
   w.clear_points()
   w.points_enabled = False  # disable point picking

   # Export
   w.save_image("annotated.png", include_markers=True)

   # Equivalent explicit list form:
   # Mark2D(image, disabled_tools=["points", "roi", "display"])
   # Mark2D(image, hidden_tools=["display", "export"])

State Persistence
-----------------

.. code-block:: python

   w = Mark2D(image, snap_enabled=True, cmap="viridis")

   w.summary()          # Print human-readable state
   w.state_dict()       # Get all settings as a dict
   w.save("state.json") # Save versioned envelope JSON file

   # Restore from file or dict — points, ROIs, and settings come back
   w2 = Mark2D(image, state="state.json")

Examples
--------

- :doc:`Simple demo </examples/mark2d/mark2d_simple>`
- :doc:`All features </examples/mark2d/mark2d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Mark2D` for full documentation.
