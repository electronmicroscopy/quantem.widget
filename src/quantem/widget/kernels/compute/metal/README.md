# compute Metal kernels (MPS)

`.msl` GPU source for the MPS compute path. Compiled + dispatched by `../mps.py`.

- `reductions.msl` — masked_sum, detector_sum, prefix-sum, bin2.
- new compute kernels (DPC, etc.) land here as new `.msl` files.

Kept as `.msl` files (not Python strings) for syntax highlighting, real compiler
errors, and isolated git history. See the architecture doc in `docs/dev-notes/`.
