# API reference

Complete reference for the four widgets and the `load` helper. Each page has two
halves:

1. **Reference** - the constructor signature, every parameter, and every public
   method, generated directly from the source by static analysis (so it never
   drifts from the code).
2. **Interactive controls** - the UI elements the widget exposes and the synced
   trait each one drives.

## Doubles as a UI-test spec

The *Interactive controls* tables are the contract an automated agent drives the
widget against. Each row names a control, the trait it mutates, and what should
be observed after acting on it. A driving agent reads the table, performs the
action over CDP, and asserts the trait moved and the canvas repainted (non-zero,
no console error, no NaN frame). When a control changes, update the table here in
the same commit; the published page is the source of truth for both human
readers and test agents.

```{toctree}
:hidden:
show2d
show3d
show3dslices
show4dstem
load
```

## At a glance

| Widget | Class | Offline export |
|---|---|---|
| [Show2D](show2d) | `quantem.widget.show2d.Show2D` | state JSON, PNG |
| [Show3D](show3d) | `quantem.widget.show3d.Show3D` | state JSON, PNG, interactive HTML (exact / quantized) |
| [Show3DSlices](show3dslices) | `quantem.widget.show3dslices.Show3DSlices` | state JSON, PNG, interactive HTML (exact / quantized) |
| [Show4DSTEM](show4dstem) | `quantem.widget.Show4DSTEM` dispatcher | state JSON, PNG, interactive WebGPU HTML; large exports use a companion data directory |
