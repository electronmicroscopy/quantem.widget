Edit2D
======

Interactive crop, pad, and mask tool for 2D images.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Edit2D

   image = np.random.rand(256, 256).astype(np.float32)

   # Crop mode (default)
   w = Edit2D(image)
   w  # display, drag crop rectangle interactively
   cropped = w.result  # get cropped NumPy array

   # Mask mode — paint a binary mask
   w = Edit2D(image, mode="mask")
   masked = w.result  # masked pixels set to fill_value

   # Padding — extend bounds beyond image
   w = Edit2D(image, bounds=(-10, -10, 266, 266), fill_value=0.0)

Features
--------

- **Crop mode** -- Drag a rectangle to crop; extends beyond image bounds for padding
- **Mask mode** -- Paint a binary mask with brush tool
- **Multi-image** -- Apply the same crop/mask to multiple images
- **Fill value** -- Configurable padding value for out-of-bounds regions
- **Scale bar** -- Calibrated when ``pixel_size`` is set
- **Control customizer** -- Hover the widget header to reveal a menu for hiding control groups
- **Tool lock/hide** -- ``disable_*`` / ``hide_*`` API for shared read-only workflows

Methods
-------

.. code-block:: python

   w = Edit2D(image)

   # Replace data (preserves crop/mask settings)
   w.set_image(new_image)

   # Export
   w.save_image("cropped.png")

Properties
----------

.. code-block:: python

   w = Edit2D(image, bounds=(10, 20, 200, 220))

   w.result       # Cropped/masked NumPy array
   w.crop_bounds  # (top, left, bottom, right) tuple
   w.crop_size    # (height, width) of output
   w.mask         # Boolean mask array (mask mode)

Control Groups
--------------

.. code-block:: python

   # Initial visible control groups
   w = Edit2D(
       image,
       show_display_controls=True,
       show_edit_controls=True,
       show_histogram=False,
   )

   # Lock selected interactions (controls remain visible but disabled)
   w_locked = Edit2D(
       image,
       disable_edit=True,
       disable_navigation=True,
       disable_export=True,
   )

   # Hide selected groups (also interaction-locked)
   w_hidden = Edit2D(
       image,
       hide_histogram=True,
       hide_stats=True,
   )

State Persistence
-----------------

.. code-block:: python

   w = Edit2D(image, cmap="viridis", bounds=(10, 20, 200, 220))

   w.summary()          # Print human-readable state
   w.state_dict()       # Get all settings as a dict
   w.save("state.json") # Save versioned envelope JSON file

   # Restore from file or dict
   w2 = Edit2D(image, state="state.json")

Examples
--------

- :doc:`Simple demo </examples/edit2d/edit2d_simple>`
- :doc:`All features </examples/edit2d/edit2d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Edit2D` for full documentation.
