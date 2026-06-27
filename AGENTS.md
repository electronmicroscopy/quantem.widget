# quantem.widget

Interactive Jupyter widgets and standalone browser GUI for electron microscopy.

## User-facing export language

For docs, menus, README text, and tutorials, use science-friendly export terms:

- Say **single** for a one-file HTML export. If precision matters, say
  **exact single**.
- Say **folder** for an HTML file that reads exact data from a nearby data
  folder or URL. Avoid leading with "sidecar" in user-facing text.
- Say **downsample** or **downsampled** for reduced-shape one-file exports.
  Binning may be mentioned as the widget-specific reducer, but the preferred
  public API option is `downsample`.
- Say **encoding** for stored data representation, such as `encoding="full"` or
  `encoding="uint8"`. Avoid using "quantized" as the preferred public API name.
- It is fine to mention old/internal parameter aliases in parentheses, for
  example "folder (`mode=\"folder\"`; old alias `mode=\"sidecar\"`)".

User-facing `mode` values should stay simple: `mode="single"` or
`mode="folder"`. Do not introduce public labels such as "sidecar",
"linked folder", or "linked data folder" unless documenting a legacy alias.

The package-wide HTML export API standard is:

```python
widget.export_html(path=None, title=None, mode="single", encoding="full", downsample=None)
```

Use compatibility names only when preserving existing APIs:
`quantized=True` -> `encoding="uint8"`, `dtype="uint8"` -> `encoding="uint8"`,
`det_bin=2` -> `downsample=2`, and `binning=4` -> `downsample=4`.

Keep implementation names such as `sidecar_url`, `from_sidecar`, and
`prepare_spectrum_image_sidecar` unchanged unless doing a deliberate API
migration.

## Release Workflow

For any `quantem.widget` release or release-candidate work, follow
`docs/maintainer/widget-release.md`.

Do not create or push a `widget-v*` tag until the required local gates in that
runbook pass. For RCs intended for TestPyPI, also run the real browser/Jupyter
user-path checks called out in the runbook before tagging.

## Repository Hygiene

Keep public documentation in durable paths such as `README.md`,
`CONTRIBUTING.md`, `docs/api/`, `docs/tutorials/`, and `docs/maintainer/`.
Do not commit local session notes, one-off benchmark scripts, AppleDouble
`._*` files, generated docs builds, screenshots, or scratch files unless they
have been promoted into a documented test, example, or maintainer runbook.
