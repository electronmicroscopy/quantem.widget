# Pull requests

This package has two pull-request classes. The first needs a discussion before
code. The second can open a pull request directly.

Delivery style (one theme per PR, self-review as a draft, quote replies, keep
follow-up ideas in a new issue) follows the
[Billinge group pull-request practices](https://scikit-package.github.io/scikit-package/programming-guides/billinge-group-standards.html#pull-request-practices).
The class rule below is the extra gate for widgets: not every change needs an
issue first.

## Discuss first

Open an issue, or confirm with a maintainer, before writing the pull request
when the change:

- adds a new public widget
- extracts shared helpers or refactors code that other widgets import
- changes a public API, the comm protocol, or the tutorial-data contract

Those changes have side effects on other viewers, docs, release registration,
and review load. The discussion decides whether the widget or abstraction
belongs in this package, and what the smallest first pull request is.

A new widget then follows [Creating a widget](../developer/widget-creation.md).

## Incremental (no prior discussion)

Open a pull request when the change stays inside one existing widget:

- a bug fix
- a control or option used only by that widget
- docs or tests for that widget

Keep the branch to one problem. Use the
[pull-request template](https://github.com/electronmicroscopy/quantem.widget/blob/main/.github/PULL_REQUEST_TEMPLATE.md).

## Shared delivery rules

- One scientific or documentation goal per pull request.
- Branch names use a short date prefix such as `aug-19-show2d-contrast`.
- Once review starts, add follow-up commits. Do not force-push. The project
  squash-merges completed work.
- Put reviewer actions and verification evidence in the three visible template
  sections. Keep the hidden checklist out of the rendered description.

Tutorial fixtures for notebooks live in the public Hugging Face dataset
[bobleesj/quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data)
under `widget-tutorials/`. Upload and download commands are on that dataset
card. A coding agent opens that Community pull request with
[Add tutorial data](../agent-prompts.md#add-tutorial-data-hugging-face-pull-request).
The GitHub loader pull request is in
[Contribute tutorial data](../tutorials/contribute_data.md).
