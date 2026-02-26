Align2DBulk
===========

GPU-accelerated bulk alignment of N frames to a reference via FFT phase correlation.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Align2DBulk

   stack = np.random.rand(50, 256, 256).astype(np.float32)
   bulk = Align2DBulk(stack, reference=0, max_shift=15)

   # Access aligned stack
   aligned = bulk.stack        # (N, H', W') cropped + aligned
   offsets = bulk.offsets       # [(dx, dy), ...]
   ncc_values = bulk.ncc        # [float, ...]

Features
--------

- **GPU-accelerated** — batched FFT + phase correlation on MPS/CUDA
- **Sub-pixel accuracy** — DFT upsampling refinement (Guizar-Sicairos et al. 2008)
- **Auto crop** — crop to common overlap region with ``crop=True``
- **Spatial binning** — on-GPU ``avg_pool2d`` with ``bin=`` parameter
- **Interactive overlay** — verify alignment frame-by-frame with opacity blending
- **NCC quality** — per-frame normalized cross-correlation bar chart
- **CPU fallback** — works without torch (slower, numpy FFT)

Parameters
----------

.. code-block:: python

   Align2DBulk(
       images,          # 3D (N,H,W), list of 2D, or IOResult
       reference=0,     # reference frame index
       padding=0.2,     # fractional padding for shift search
       max_shift=0,     # max shift px (overrides padding when > 0)
       bin=1,           # spatial bin factor
       crop=True,       # crop to common overlap
       cmap="gray",     # colormap
       device=None,     # "cpu", "mps", "cuda", or None (auto)
   )

Control groups
--------------

.. code-block:: python

   # Lock groups (visible but non-interactive)
   bulk = Align2DBulk(
       stack,
       disable_display=True,
       disable_navigation=True,
       disable_export=True,
   )

   # Hide groups entirely
   bulk = Align2DBulk(
       stack,
       hide_stats=True,
       hidden_tools=["export"],
   )

Examples
--------

- :doc:`Simple demo </examples/align2d_bulk/align2d_bulk_simple>`

API
---

See :class:`quantem.widget.Align2DBulk` for full documentation.
