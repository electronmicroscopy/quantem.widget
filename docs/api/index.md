# API reference

Complete reference for the widgets, HTML export, and data loading / I/O.
Common display-chrome names and presets are documented in [Viewer UI controls](viewer-ui).
The same tables are surfaced in the developer-facing [UI Guide](../developer/ui-guide).
Widget pages have two halves:

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

## At a glance

| Widget | Class | Offline export |
|---|---|---|
| [Show1D](show1d) | `quantem.widget.show1d.Show1D` | state JSON, CSV, PNG/PDF via Python, interactive HTML |
| [Show2D](show2d) | `quantem.widget.show2d.Show2D` | state JSON, PNG, interactive HTML (`encoding="full"` / `encoding="uint8"`) |
| [Show3D](show3d) | `quantem.widget.show3d.Show3D` | state JSON, PNG, interactive HTML (`encoding="full"` / `encoding="uint8"`) |
| [Show3DSlices](show3dslices) | `quantem.widget.show3dslices.Show3DSlices` | state JSON, PNG, interactive HTML (`encoding="full"` / `encoding="uint8"`) |
| [Show4DSTEM](show4dstem) | `quantem.widget.Show4DSTEM` dispatcher | state JSON, PNG, interactive WebGPU HTML; large exports use a companion data directory |
| [ShowEDS](showeds) | `quantem.widget.showeds.ShowEDS` | state JSON, interactive HTML; large exact data use folder export, portable demos use downsampled HTML |
| [ShowDiffraction](showdiffraction) | `quantem.widget.showdiffraction.ShowDiffraction` | state JSON, PNG, interactive HTML |
| [ShowFolder](showfolder) | `quantem.widget.showfolder.ShowFolder` | selection JSON for selected microscopy files and folders |

All widget-level HTML exports follow the [HTML export](html-export) protocol.
