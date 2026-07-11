<!--
This template mirrors the "Widget PR checklist" in README.md. If you edit one,
update the other. Links are absolute because relative links do not resolve in
PR descriptions.

How to use: keep the Core checklist, then expand and complete only the
sections your change touches. Delete sections that do not apply — a reviewer
should only see boxes that are relevant to this PR. Either verify each box
yourself or have a coding agent verify it and check it off for you; mark an
item that is in a relevant section but does not apply as "n/a — reason".

This PR workflow follows the scientific-software packaging standards described
in: S. Lee, C. Myers, A. Yang, T. Zhang, Y. Xiao, and S. J. L. Billinge,
"Scikit-package - software packaging standards and roadmap for sharing
reproducible scientific software", Digital Discovery (2026).
https://doi.org/10.1039/d6dd00121a
-->

## Summary

<!-- What does this PR change, and why? -->

## Core checklist (every PR)

- [ ] The change includes focused tests for Python state/export behavior and
  frontend build coverage where possible; start with `PYTHONPATH=src:. pytest -q`
  and `npm run build`, or run `scripts/widget_local_signoff.sh`.
- [ ] Before committing, inspect `git status --short` and `git diff --stat`;
  do not commit generated HTML, docs builds, screenshots, local notebooks,
  private data, or machine-specific notes.
- [ ] Committed notebooks carry NO baked widget state (`metadata.widgets`) and
  pass `scripts/check_notebook_sizes.py`. The docs CI executes tutorials at
  build time (`execute_notebooks: force` in `docs/_config.yml`) and bakes
  widget state into the published HTML only — never commit a re-executed
  notebook with stored widget state, and never switch the docs build to
  `cache` mode (it silently drops widget state and blanks every widget).
- [ ] Only the sections below that this PR touches are kept; the rest are
  deleted from this description.

<details>
<summary><b>Python API and docs</b> — new widget, loader, or API change</summary>

- [ ] The widget has a small, stable Python API with NumPy-style docs, helpful
  errors, and `(row, col)` coordinate wording where positions are shown.
- [ ] Documentation includes a minimal tutorial notebook under
  [docs/tutorials](https://github.com/bobleesj/quantem.widget/tree/main/docs/tutorials)
  and an API page under
  [docs/api](https://github.com/bobleesj/quantem.widget/tree/main/docs/api)
  when a public widget or loader is added.
- [ ] Tutorial notebooks avoid unnecessary `display(...)` and extra display
  imports; let the returned widget render naturally.
- [ ] Synthetic-data generation cells in tutorial notebooks are collapsed with
  the `hide-input` cell tag and a descriptive toggle label via cell metadata
  `mystnb.code_prompt_show` (for example "Show synthetic data generation
  code"), never the default "Show code cell source". Keep widget-construction
  code and real-data loader calls visible; split a cell that mixes the two.

</details>

<details>
<summary><b>UI design and theming</b> — frontend / viewer changes</summary>

- [ ] The frontend follows the local viewer patterns instead of inventing a new
  design system; compare against
  [Show2D](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/show2d.ipynb),
  [Show3D](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/show3d.ipynb),
  [Show3DSlices](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/show3dslices.ipynb),
  [Show4DSTEM](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/show4dstem.ipynb), and
  [ShowEDS](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/showeds.ipynb). Follow the
  [widget UI protocol](https://github.com/bobleesj/quantem.widget/blob/main/docs/maintainer/widget-ui-protocol.md).
- [ ] Controls are compact and content-sized: use icon/text buttons for
  commands, switches for binary options, sliders for numeric values, menus for
  option sets, and avoid stretched empty control bars.
- [ ] Compact labels stay grouped with the control they name on mobile and
  narrow layouts. A row may wrap, but `Auto`, `Smooth`, `Zoom`, `Pan`,
  `Contrast`, `fps`, `avg`, and similar labels must not separate from their
  switch, menu, slider, or button.
- [ ] Compact widget control labels do not use decorative colons. Prefer
  `Scale`, `Color`, `Auto`, `Link`, `Zoom`, `Pan`, and `ROI` in dense toolbar
  rows; keep colons for explanatory prose and tooltips.
- [ ] Command buttons use Title Case, for example `Copy`, `Export`, `Reset`,
  `Add`, `Clear`, and `Undo`. Keep scientific acronyms and file formats
  uppercase, for example `FFT`, `ROI`, `BF`, `ADF`, `HTML`, `PNG`, and `MP4`.
- [ ] The widget supports both light and dark notebook/docs themes: all labels,
  borders, controls, plots, histograms, ROI handles, status text, and export UI
  remain readable.
- [ ] The widget has no hardcoded dark-only or light-only assumptions in plots,
  canvas backgrounds, tooltips, menus, or exported HTML.
- [ ] Histogram UI matches the existing Show2D-style interaction: compact panel,
  no extra whitespace, draggable min/max handles, fast center drag, and no
  visible lag.
- [ ] New or changed widget interactions have a matching storyboard story in
  [docs/maintainer/storyboard-&lt;widget&gt;.md](https://github.com/bobleesj/quantem.widget/blob/main/docs/maintainer/storyboard.md)
  (add stories for new behavior, update stale ones), and the storyboard
  drive-test was run for the affected widget with the driven story IDs
  reported.

</details>

<details>
<summary><b>Performance and real-time interaction</b> — drag, live controls, big data paths</summary>

- [ ] Any draggable selector has live preview separate from committed widget
  state; use refs/CSS transforms or an equivalent fast path during drag. See
  [performance notes](https://github.com/bobleesj/quantem.widget/blob/main/docs/maintainer/widget-performance.md).
- [ ] Use [Show4DSTEM](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/show4dstem.ipynb)
  detector dragging and
  [ShowEDS](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/showeds.ipynb)
  energy-band dragging as the real-time UX benchmark: aim for 60 FPS when
  feasible and keep live controls at 30 FPS or better.
- [ ] Real-time interactions are browser-driven and verified by actually
  dragging controls in JupyterLab or exported HTML, not only by reading code or
  unit tests.
- [ ] Show4DSTEM live-scope workflows append new ready `*_master.h5`
  acquisitions into the same live viewer without rebuilding the notebook. Verify
  new masters appear in the Dataset slider, partial files are skipped until
  ready, and detector/scan interactions remain real time after append. See
  [Show4DSTEM live scope folders](https://github.com/bobleesj/quantem.widget/blob/main/docs/api/show4dstem.md#live-scope-folders).
- [ ] Performance reports separate load time, widget build time, first browser
  paint, and interaction FPS/latency. Include data shape, dtype, raw size,
  backend, and any crop/bin/downsample/quantization. Prefer `verbose=True`
  output that users and agents can copy; use `quantem.widget.profile_widget`
  for profiling notebooks when possible. See
  [performance notes](https://github.com/bobleesj/quantem.widget/blob/main/docs/maintainer/widget-performance.md).
- [ ] For interaction-sensitive changes, run
  `scripts/widget_local_signoff.sh --quick --browser` for exported HTML/UI
  paths, fix issues immediately, rebuild, refresh, and redrive before claiming
  the widget is ready. See
  [Automation](https://github.com/bobleesj/quantem.widget/blob/main/docs/maintainer/automation.md) and
  [Agent signoff](https://github.com/bobleesj/quantem.widget/blob/main/docs/maintainer/widget-agent-signoff.md).
- [ ] Expensive work avoids Python/kernel round trips during pointer movement;
  use WebGPU, typed arrays, cached indexes, workers, or throttled schedulers
  where the widget interaction requires live feedback.

</details>

<details>
<summary><b>Data honesty and repo size</b> — data loaders, tutorial data, CI fixtures</summary>

- [ ] Large scientific data stays honest about precision and size: do not
  silently crop, bin, downsample, quantize, or materialize sparse zeros.
- [ ] Any binning/downsampling is explicit in the API and documentation, with
  the reducer named clearly, for example mean, sum, or display-scaled `uint8`.
- [ ] Keep `main` lightweight: small real rendered examples are fine, but large
  tutorial arrays or HTML payloads should be generated during docs builds or
  downloaded from public data hosting only when the size justifies it.
- [ ] Keep clone and install size small for microscope PCs. Real tutorial data
  belongs in public data hosting such as Hugging Face datasets, Zenodo, or
  release assets, then gets downloaded and cached by tutorial helpers at run
  time. Do not commit large real arrays, generated HTML, or rendered docs
  branches to this repository just to make examples work.
- [ ] CI should test data-loading protocol with tiny deterministic fixtures or
  monkeypatched downloads. Full real-data downloads are reserved for docs builds,
  release signoff, or local performance checks that explicitly opt in.

</details>

<details>
<summary><b>Export and saved state</b> — HTML export, widget state, sharing</summary>

- [ ] The widget exposes `export_html(path=None, title=None, mode="single",
  encoding="full", downsample=None)` when it can be exported. Follow the
  [HTML export protocol](https://github.com/bobleesj/quantem.widget/blob/main/docs/api/html-export.md).
- [ ] If the widget has an in-widget **Export** button, it uses the standard
  export traits and reports filename, mode, encoding/downsample choice, and
  output size.
- [ ] Saved Jupyter widget state works: after interacting, Cmd+S, close/reopen
  in JupyterLab, and confirm the view restores without rerunning cells when the
  environment supports saved widget state.
- [ ] Standalone HTML works without a live Python kernel, and the exported page
  preserves the intended theme, viewport, interaction state, and scale/contrast
  state.
- [ ] GitHub sharing is treated separately from live HTML: GitHub notebook
  previews should use static compressed widget pictures, never heavy live widget
  state. See [GitHub preview](https://github.com/bobleesj/quantem.widget/blob/main/docs/github-preview.md).

</details>

---

This PR workflow follows the packaging standards for reproducible scientific
software described in [scikit-package](https://doi.org/10.1039/d6dd00121a):
S. Lee, C. Myers, A. Yang, T. Zhang, Y. Xiao, and S. J. L. Billinge,
*Digital Discovery* (2026), DOI
[10.1039/d6dd00121a](https://doi.org/10.1039/d6dd00121a).
