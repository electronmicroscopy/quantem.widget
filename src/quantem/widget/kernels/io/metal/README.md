# io Metal kernels (MPS)

`.msl` GPU source for the MPS io path. Compiled + dispatched by `../mps.py`.

- `bslz4.msl` — bitshuffle+LZ4 decompress (the frozen decode kernel).

Kept as `.msl` files (not Python strings) for syntax highlighting, real compiler
errors, and isolated git history. See the architecture doc in `docs/dev-notes/`.
