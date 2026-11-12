# ShowPtycho release acceptance

Build success is separate from scientific and browser acceptance. Every widget
release must pass the installed-wheel test on physical CUDA and Apple MPS.
The release workflow builds one wheel, tests that artifact on both backends,
and publishes that same artifact only after both checks pass.

Register dedicated self-hosted runners labelled `quantem-cuda` and
`quantem-mps`, with Python 3.11 or newer and Google Chrome installed. CUDA needs
a supported NVIDIA driver and working hardware WebGPU. MPS needs Apple silicon.
Use dedicated compute slots: other GPU jobs can invalidate timing comparisons
or exhaust device memory. No registered runner means the release stays pending.

The workflow downloads the pinned public gold dataset and starts a fresh
environment and Jupyter kernel. It verifies GPU-only joint fitting with one
seed, 200 trials and the exact-gradient refinement, then draws two regions, moves and resizes one,
checks the other remains unchanged, saves calibration and restores it in a
second fresh kernel. Browser rendering must use a hardware adapter.

To run the same gate locally after installing candidate wheels:

```bash
unset PYTHONPATH
export QUANTEM_WIDGET_BROWSER=1 QUANTEM_WIDGET_HEADLESS=1
export QUANTEM_DIRECT_RELEASE=1
export QUANTEM_DIRECT_RELEASE_SOURCE=/path/to/scan.qem
export QUANTEM_WIDGET_BROWSER_OUT=/tmp/direct-widget-check
python -m pytest -q -s tests/browser/test_showptycho_release.py
```

For an H5 file with incomplete calibration, set
`QUANTEM_DIRECT_RELEASE_CALIBRATION` to a JSON object with the actual beam energy,
convergence semiangle and scan sampling. Saved detector sampling is reused;
missing sampling is fitted. Do not substitute another specimen's calibration.

The gate rejects imports from editable checkouts. It saves screenshots and
separates fit time from the live interaction completion time. Keep private-data
screenshots and raw reports outside the repository. CI uses public gold only.
Kernel benchmarks use `quantem-gpu benchmark-direct --require-installed`; they do
not measure browser paint. Record missing hardware, skipped tests and failures
explicitly, and do not publish an RC that has not passed both backend gates.

Publish and qualify the GPU dependency first, then update the widget's GPU
version bounds to that exact candidate series before building its wheel.
An unchanged version label is not proof of identical code. Keep source and
wheel fingerprints with the private release evidence.
