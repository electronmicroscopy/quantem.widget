"""io kernels: decode + bin + mask + scan-shape. Backend submodules: cuda, mps.

Pick via ``quantem.widget.kernels.io_backend()``. Same function names across
``cuda.py`` / ``mps.py`` (decode, ...). FROZEN-ish — touched only on detector
format changes. Migration target; logic still lives in
``quantem.widget.io.backends`` until moved here (parity-gated).
"""
