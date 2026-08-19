# Contributing to quantem.widget

Thank you for contributing. Changes may come from a scientist working directly,
from a scientist collaborating with a coding agent, or from an agent preparing
work for human review. Everyone follows the same scientific and review
standards.

`quantem.widget` combines Python APIs, React frontends, saved notebook state,
standalone HTML, and GPU-backed scientific workflows. A useful contribution
keeps the parts it touches consistent and gives reviewers enough evidence to
understand what changed.

## Start with a focused change

Use one small branch for one issue or documentation goal. Describe the user
problem before the implementation, and keep unrelated experiments or cleanup
out of the branch.

Adding a new widget, or a refactor that other widgets will import, starts with
an issue or a maintainer discussion. A bug fix or a feature that stays inside
one existing widget can go straight to a pull request. The two classes are in
the [pull request guide](docs/maintainer/pull-requests.md).

Tutorial fixtures for notebooks belong under `widget-tutorials/` in the public
[bobleesj/quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data)
dataset. Upload and download commands are on that dataset card. A coding
agent can open the Hugging Face Community pull request with the prompt in
[Agent-assisted development](docs/agent-prompts.md#add-tutorial-data-hugging-face-pull-request).
The GitHub loader pull request is in
[Contribute tutorial data](docs/tutorials/contribute_data.md).

Once review begins, do not force-push. Add follow-up commits so GitHub preserves
inline comments and the reviewer can inspect only the changes since their last
visit. The project squash-merges completed work.

For Git and GitHub introductions, see
[ophusgroup/dev](https://github.com/ophusgroup/dev).

## Set up the project

Create a dedicated Conda environment, then install the Python package and
frontend dependencies:

```bash
conda create --name live-env --channel conda-forge python=3.14 nodejs=22 pip
conda activate live-env
pip install -e .
npm ci
```

For documentation work:

```bash
pip install -r docs/requirements.txt
```

## Check your change

Run the smallest checks that cover what you changed. For Python changes, run:

```bash
pytest -q
```

For frontend work, use the development build while editing and make a final
production build before review:

```bash
npm run dev
npm run build
```

If you changed frontend logic, also run the focused checks:

```bash
npm run typecheck
npm test
```

You do not need to memorize a separate command for every widget. The
[pull-request template](.github/PULL_REQUEST_TEMPLATE.md) routes each type of
change to the relevant checks, and the
[performance and UI testing guide](docs/maintainer/performance-ui-testing.md)
documents specialized and release-level gates.

## Use the pull-request template

The PR template is intentionally thorough. It is a shared review contract for
humans and coding agents, not a requirement to keep every checkbox in every PR.
Keep the core checklist and only the expandable sections relevant to your
change; delete the rest.

The structured verification fields—tests, user workflow, data shape and dtype,
backend, timing, and evidence artifacts—make scientific claims reproducible.
They also give agents and automation enough metadata to select checks, detect
missing evidence, and prepare a review without guessing about the experiment.
A human reviewer should still be able to understand the completed template
without reading agent logs.

## Preserve scientific behavior

- Use NumPy-style docstrings and modern Python type hints for public APIs.
- Present positions to users in `(row, col)` order and name physical units.
- Add helpful errors with a corrective next step when invalid input could
  produce a scientifically wrong result.
- Do not silently crop, bin, downsample, quantize, or substitute a backend.
- Test the workflow a scientist actually uses, not only an internal helper.
- For interactive changes, drive the UI and report the data shape, dtype,
  backend, and observed responsiveness.

See the [widget UI protocol](docs/maintainer/widget-ui-protocol.md),
[performance guide](docs/maintainer/widget-performance.md), and
[HTML export reference](docs/api/html-export.md) for the detailed contracts.

## Commit durable files only

Commit source, tests, and public documentation. Do not commit generated HTML,
documentation builds, `node_modules`, local screenshots, private notebooks or
data, machine paths, browser profiles, or AppleDouble `._*` metadata.

Before committing, inspect the exact scope:

```bash
git status --short
git diff --stat
```

Use a short, single-line Conventional Commit-style message such as:

```text
feat: add ShowFolder thumbnail cache
fix: stabilize Show2D resize handle
docs: clarify HTML export guidance
```

## Releases

Release candidates have additional packaging, browser, and TestPyPI gates.
Follow the [release guide](docs/maintainer/widget-release.md) before creating a
`widget-v*` tag.
