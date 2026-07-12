# Agent Commit Guidelines

This file records repository conventions that coding agents should follow when
making commits.

These conventions follow the scikit-package standards for reproducible
scientific software (https://doi.org/10.1039/d6dd00121a). See the
"Coding Standards (scikit-package)" section in [AGENTS.md](AGENTS.md) for the
full list: issue-first PRs, one themed PR per issue, NumPy-style docstrings,
and no force-pushing a branch under active review.

## Commit Messages

Use a short Conventional Commit-style first line:

```text
type: short imperative summary
```

Use these common types:

- `feat:` for a user-facing feature.
- `fix:` for a bug fix.
- `docs:` for documentation-only changes.
- `test:` for test-only changes.
- `refactor:` for internal restructuring without behavior changes.
- `perf:` for performance improvements.
- `build:` for packaging, dependencies, or build tooling.
- `ci:` for GitHub Actions or CI workflow changes.
- `chore:` for maintenance that does not fit the above.

Examples:

```text
feat: add ShowFolder thumbnail cache
fix: stabilize Show2D resize handle
docs: update HTML export protocol
```

Keep commit messages single-line unless the user or reviewer asks for a body.
Preserve the user's configured Git author and committer identity. Do not add
`Co-authored-by` trailers unless explicitly requested.

## Notebooks

Committed tutorial notebooks must carry NO baked widget state
(`metadata.widgets`) and must pass `scripts/check_notebook_sizes.py`. The docs
CI executes tutorials at build time (`execute_notebooks: force` in
`docs/_config.yml`) and bakes widget state into the published HTML only. Never
commit a re-executed notebook with stored widget state, and never switch the
docs build to `cache` mode — jupyter-cache silently drops widget state and
every widget on the docs site goes blank.

Colab bootstrap cells (the `try: import google.colab` / pip-install block) are
plumbing for Colab users only: keep the install strictly inside the
`google.colab` success branch, put the bootstrap in its OWN cell tagged
`remove-cell`, and never mix it with tutorial code. The tag strips it from
built docs pages while the Colab badge notebook keeps it.

Built docs pages must show exactly ONE output per widget cell: the interactive
widget. Every docs build sets `QUANTEM_WIDGET_STATIC_FALLBACK=0` (see
widget-docs.yml and widget_local_signoff.sh) so the saved-notebook static
preview sibling is never emitted — it would render as a duplicate image under
the live widget. Set the same variable in any new docs/CI build path, and
after changing widget display or save-state code, check built pages for zero
`img.quantem-static-fallback` occurrences.

## Widget Testing

When testing widgets, prefer organic user paths over heavily preconfigured
constructor calls. Start from a minimal call such as `Show2D(data)` or
`Show3D(stack)` and use the widget UI to turn on controls, change modes, scrub,
play, zoom, pan, save, and export like a scientist exploring a notebook. Use
many constructor parameters only when the behavior under test is specifically
that API-start state or when reproducing a precise historical bug that requires
those initial conditions. A kwargs-heavy repro notebook is useful evidence, but
it does not replace at least one organic UI-drive trial for interaction bugs.
