"""Back-compat shim for the MPS IO backend.

The active Apple-GPU decompression implementation now lives in
``quantem.gpu.io.backends.mps``. Re-export it here so existing widget import
paths keep working during the migration.
"""
from quantem.gpu.io.backends.mps import *  # noqa: F401,F403
from quantem.gpu.io.backends.mps import (  # noqa: F401  explicit non-public names
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
