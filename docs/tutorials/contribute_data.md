# Contribute tutorial data

How to add files, including the Hugging Face token
(`hf auth login`):
[bobleesj/quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data).
A coding agent can open that Community pull request. Copy the prompt
[Add tutorial data](https://huggingface.co/datasets/bobleesj/quantem-data#coding-agents)
from the dataset card.
The human creates the Write token and attests permission. The agent runs
`hf upload ... --create-pr` after `python -m check_meta` prints `ok`.

After that Hugging Face PR is merged, open a GitHub PR here only if a
notebook should load the fixture by name:

- helper in `src/quantem/widget/data/tutorials.py`
- list it on [Tutorial Datasets](../api/datasets.md)
- use it in the tutorial
- link the Hugging Face PR

A new widget is discuss-first. See [Pull requests](../maintainer/pull-requests.md).
Do not commit `.npy` or `.emd` here. Do not call `quantem.widget.io.upload`.
