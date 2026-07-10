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
