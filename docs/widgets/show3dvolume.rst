Show3DVolume
============

Orthogonal slice viewer for 3D volumes with XY, XZ, and YZ planes.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Show3DVolume

   volume = np.random.rand(64, 64, 64).astype(np.float32)
   Show3DVolume(volume, title="My Volume", cmap="viridis")

Features
--------

- **Three orthogonal views** — XY, XZ, and YZ slice planes
- **Interactive slicing** — Click or drag to navigate through the volume
- **Crosshair overlay** — Shows intersection of slice planes
- **3D volume rendering** — WebGL ray-casting for 3D visualization
- **Per-axis playback** — Animate through slices along any axis
- **FFT** — Toggle Fourier transform for each slice plane
- **Dual-volume comparison** — Side-by-side comparison with shared navigation
- **Tool customization** — Disable or hide control groups (including playback and 3D volume controls)

Dual-Volume Comparison
----------------------

Compare two volumes side by side with shared slice positions, zoom/pan, camera,
and colormap. Stats and cursor readout are independent per volume.

.. code-block:: python

   Show3DVolume(
       reconstruction,
       data_b=ground_truth,
       title="Reconstruction",
       title_b="Ground Truth",
       cmap="inferno",
   )

Both volumes must have the same shape. Pass any supported input type (NumPy,
PyTorch, IOResult, Dataset3d) for either volume.

.. code-block:: python

   # Replace both volumes (preserves display settings)
   w.set_image(new_recon, data_b=new_gt)

Methods
-------

.. code-block:: python

   w = Show3DVolume(volume)

   # Replace data (preserves display settings)
   w.set_image(new_volume)

   # Dual mode: replace both volumes
   w.set_image(volume_a, data_b=volume_b)

   # Playback (animate through slices)
   w.play()
   w.pause()
   w.stop()

   # Export
   w.save_image("xy_slice.png")                      # current XY slice
   w.save_image("xz.png", plane="xz", slice_idx=32)  # specific plane + slice

Control Groups
--------------

.. code-block:: python

   # Lock groups (visible but non-interactive)
   w = Show3DVolume(
       volume,
       disable_display=True,
       disable_playback=True,
       disable_volume=True,
   )

   # Hide groups entirely
   w = Show3DVolume(
       volume,
       hide_histogram=True,
       hide_stats=True,
       hide_volume=True,
       hidden_tools=["export"],
   )

State Persistence
-----------------

.. code-block:: python

   w = Show3DVolume(volume, cmap="viridis", log_scale=True)
   w.slice_z = 48

   w.summary()          # Print human-readable state
   w.state_dict()       # Get all settings as a dict
   w.save("state.json") # Save versioned envelope JSON file

   # Restore from file or dict
   w2 = Show3DVolume(volume, state="state.json")

Examples
--------

- :doc:`Simple demo </examples/show3dvolume/show3dvolume_simple>`
- :doc:`All features </examples/show3dvolume/show3dvolume_all_features>`

API
---

See :class:`quantem.widget.Show3DVolume` for full documentation.
