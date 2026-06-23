"""compute kernels: masked-sum / prefix / bin / reduce (BF/DF/DPC callers).

Pick via ``quantem.widget.kernels.compute_backend()``. CHURNS — new virtual
detectors / derived properties land here. The MPS reductions live in
``compute.mps`` (MetalVirtualImage / ChunkedFrames); the cuda equivalent is
pending (cuda compute still uses engine/preprocess for now).
"""
