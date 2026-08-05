# Generated WebGPU Engine Sources

The reusable WebGPU browser-compute sources are generated into
`js/.generated/engine/` from their scientific domains in `quantem.gpu` before
frontend builds.

Edit the canonical source under `quantem.gpu/src/quantem/gpu/device/`, `io/`,
`detector/`, `dpc/`, or `ssb/`, then run:

```bash
npm run sync:webgpu
```

Do not add WebGPU engine `.ts` files back under `js/engine/`. This directory is
kept only as a maintainer note so `quantem.widget` does not own decompression,
HDF5 browser IO, Show4DSTEM reduction, or ShowPtycho SSB kernels.

`quantem.widget` owns the UI, bundling, and exported HTML runtime. The shared
kernel math and browser compute engine source belong in `quantem.gpu`,
including Show4DSTEM WebGPU IO/reductions and ShowPtycho WebGPU SSB.
