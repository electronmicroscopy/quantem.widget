# Working in this repository

This repository is public. Everything committed here, and everything in its
history, can be read by anyone. These rules apply to coding agents and humans.

## Never commit

- Raw experiment or benchmark logs, manifests, JSONL, profiler dumps, browser
  signoff dumps or run folders. Write the dated summary under `docs/`
  (question, setup, numbers, conclusion) and keep the raw evidence out of git.
- Datasets, detector or acquisition files, other than small synthetic fixtures
  under `tests/data/`. Tutorial data lives in the public `bobleesj/quantem-data`
  Hugging Face dataset and is downloaded at run time.
- Images, figures or notebook outputs of any data other than the public gold
  nanoparticle dataset. Gold is the only dataset that may appear in this
  repository: in `docs/_static/`, in the docs and in any committed image. The
  one credited exception is the liquid-cell gold demo gif
  (`docs/_static/show4dstem-serin-gold.gif`) and its acknowledgement in the
  README and docs intro. A figure of any other acquisition, however anonymous,
  is not committed; describe the result in words and numbers.
- Notebook state. Tracked `.ipynb` files carry no saved widget state
  (`metadata.widgets`) and no `image/*` outputs: the docs build re-executes
  every tutorial and renders the widgets fresh, so committed state only grows
  the history. `scripts/check_notebook_sizes.py` bounds the remaining text
  outputs.
- Local paths (home directories), machine host names, or tailnet addresses.
- Names of collaborators, companies or people, other than the gold demo credit
  above. Name data by its public dataset id.
- Credentials or tokens.
- Files over 5 MB, other than `docs/_static/show4dstem-serin-gold.gif`
  (`scripts/check_large_files.py` holds the same exception).

`tests/test_public_repository.py` enforces these rules. It is fast and
offline, and it must pass before every push:

```bash
PYTHONPATH=src python -m pytest -q tests/test_public_repository.py
```

If unsure whether something is private, leave it out and ask.

## Tests

`PYTHONPATH=src python -m pytest -q` runs the test suite and must pass before
every push. Browser tests need the built JS bundle (`npm ci && npm run build`)
and a Chrome or Chromium executable; they skip when neither is found.
`python scripts/check_large_files.py` and
`python scripts/check_notebook_sizes.py` are the size guards the release
workflow runs.
