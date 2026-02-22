Bin
===

Calibration-aware 4D-STEM binning widget with BF/ADF quality checks.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Bin

   data = np.random.rand(64, 64, 128, 128).astype(np.float32)
   w = Bin(data, pixel_size=2.39, k_pixel_size=0.46, device="cpu")
   w

Features
--------

- **Torch-only compute** with configurable device (``cpu``/``cuda``/``mps``)
- **Independent scan + detector bin factors** with ``mean``/``sum`` reduction
- **Edge handling** modes: ``crop``, ``pad``, ``error``
- **Calibration propagation** for real-space and detector sampling after binning
- **BF/ADF QC previews** before and after binning with compact stats
- **Export helpers**: ``save_image``, ``save_zip``, ``save_gif``

Methods
-------

.. code-block:: python

   w = Bin(data)

   # Replace data (preserves bin settings)
   w.set_data(new_data)

   # Access results
   w.result              # binned tensor (on device)
   w.get_binned_data()   # copy as tensor
   w.get_binned_data(as_numpy=True)  # copy as NumPy

State Persistence
-----------------

.. code-block:: python

   w = Bin(data, pixel_size=2.39, k_pixel_size=0.46)

   w.summary()                # human-readable summary
   w.state_dict()             # serializable state dictionary
   w.save("bin_preset.json")  # save preset JSON

   w2 = Bin(data, state="bin_preset.json")

Programmatic Export
-------------------

.. code-block:: python

   w.save_image("bin_grid.png", view="grid")
   w.save_zip("bin_bundle.zip", include_arrays=True)
   w.save_gif("bin_bf_compare.gif", channel="bf")

Batch Runner
------------

Use the same preset over folders (one file at a time):

.. code-block:: bash

   python -m quantem.widget.bin_batch \
     --input-dir /path/to/raw \
     --output-dir /path/to/binned \
     --preset bin_preset.json \
     --pattern '*.npy' \
     --recursive \
     --device cpu

Examples
--------

- :doc:`Simple demo </examples/bin/bin_simple>`
- :doc:`All features </examples/bin/bin_all_features>`

API Reference
-------------

See :class:`quantem.widget.Bin` for full documentation.
