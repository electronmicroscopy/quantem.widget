"""CUDA compute kernel — placeholder.

The CUDA BF/DF/DPC/virtual-image reductions are NOT migrated here yet. They live,
working, in ``quantem.live.engine.preprocess`` (``bf_df_dpc``, ``dp_mean``,
``virtual_image``) on cupy, used by the browse / acquisitions / screen dashboard
surfaces. Extracting them is a SEPARATE gated effort. Until then, those callers
use ``engine.preprocess`` directly. See
docs/dev-notes/2026-06-01-kernels-backend-architecture.md.
"""
