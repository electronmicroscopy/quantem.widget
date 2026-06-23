"""CUDA io kernel — placeholder.

The CUDA decode path is NOT migrated here yet. It lives, working and validated
across hundreds of datasets, in ``quantem.widget.io.hdf5`` (``GPUDecompressor``,
``_load_master_pipelined`` / ``_load_master_optimized``) using cupy RawKernels +
``io.bitshuffle``. It is woven into ``load()`` and the production reconstruction
pipeline; extracting it is a SEPARATE gated effort (dashboard smoke + ptycho
parity must pass), deliberately NOT bundled with the MPS refactor.

When migrated, this module exposes ``decode(path, det_bin, pixel_mask)`` as a thin
delegator to the hdf5 functions. Until then, the CUDA path is reached via
``io.load(backend="cuda")`` directly. See
docs/dev-notes/2026-06-01-kernels-backend-architecture.md.
"""
