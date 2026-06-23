"""Back-compat shim. The MPS io backend moved to ``quantem.widget.kernels.io.mps``
(see docs/dev-notes/2026-06-01-kernels-backend-architecture.md). Re-export so
existing import paths keep working during the kernels/ migration.
"""
from quantem.widget.kernels.io.mps import *  # noqa: F401,F403
from quantem.widget.kernels.io.mps import (  # noqa: F401  explicit non-public names
    MPSChunked4DSTEM,
    MPSDecompressor,
    _MtlArray,
    _metal_buffer_alloc,
    _numpy_view,
    _parse_headers,
    _read_pixel_mask,
    clear_mps_cache,
    load_master,
    load_master_chunked,
)
