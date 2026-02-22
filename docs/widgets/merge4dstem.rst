Merge4DSTEM
===========

Stack multiple 4D-STEM datasets along a time axis to produce a 5D array.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Merge4DSTEM

   sources = [
       np.random.rand(64, 64, 128, 128).astype(np.float32),
       np.random.rand(64, 64, 128, 128).astype(np.float32),
       np.random.rand(64, 64, 128, 128).astype(np.float32),
   ]
   w = Merge4DSTEM(sources, pixel_size=2.39, k_pixel_size=0.46)
   w

Features
--------

- **Torch-backed merge** with ``torch.stack`` on GPU (MPS/CUDA/CPU auto-detect)
- **Shape validation** — hard fail on mismatched ``(scan_r, scan_c, det_r, det_c)``
- **Calibration checking** — warns on calibration mismatch between sources
- **Preview canvas** — mean diffraction pattern from the first source
- **Dataset4dstem output** via ``.result`` property
- **Save merged result** to Zarr zip archive with ``.save_result(path)``
- **Open in Show4DSTEM** with ``.to_show4dstem()``

Methods
-------

.. code-block:: python

   w = Merge4DSTEM(sources)

   # Run the merge
   w.merge()                          # stack on GPU

   # Access results
   w.result                           # -> Dataset4dstem (5D)
   w.result_array                     # -> numpy (5D)

   # Save to disk
   w.save_result("merged.zarr.zip")

   # Open result in Show4DSTEM viewer
   w.to_show4dstem()

State Persistence
-----------------

.. code-block:: python

   w.summary()                        # human-readable summary
   w.state_dict()                     # serializable state dictionary
   w.save("merge_preset.json")        # save preset JSON

   w2 = Merge4DSTEM(sources, state="merge_preset.json")

API Reference
-------------

See :class:`quantem.widget.Merge4DSTEM` for full documentation.
