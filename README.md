# quantem.widget

Interactive WebGPU visualization widgets for 4D-STEM / electron microscopy:
`Show2D`, `Show3D`, `Show3DSlices`, `Show4DSTEM` (anywidget + WebGPU), plus a
standalone offline browser app (`web/`).

Published to TestPyPI as `quantem.widget`.

## Provenance
Extracted from the `quantem` monorepo (`bobleesj/quantem`, branch
`widget-show3d-show4dstem-kernels` @ `a64301db1b14`) on 2026-06-23 to a standalone
repo for independent maintenance + release cadence. The monorepo retains the full
prior development history. Release line continues at `0.0.1rc23+` (TestPyPI already
holds rc1-rc22 from the monorepo).

## Install (TestPyPI, pre-release)
```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantem.widget
```
