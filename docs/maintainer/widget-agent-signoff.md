# Agent signoff

Use this workflow when a widget change must be judged by real interaction, not
only by unit tests. The goal is to make an agent drive the widgets the way a
microscopist would, fix anything that feels wrong, then show human-readable
evidence before a release.

This is deliberately not a fully automated smoke test. It is a fix-and-redrive
loop. Start from the scientific user stories in
[Storyboard](storyboard); use generated checklist packets only as
evidence containers and prompts for concrete browser actions.
When the change touches speed, heavy data, save/reopen, FFT, playback, export,
or large-panel layout, also run the real-data gates in
[Performance UI Testing](performance-ui-testing).

## When to run

Run an agent signoff before a release candidate and after changes to:

- widget layout, controls, theme colors, scale bars, histograms, or exports,
- drag/zoom/pan interactions,
- Show4DSTEM detector or virtual-image behavior,
- ShowEDS energy-band, ROI, spectrum, periodic-table, or real-data behavior,
- loader paths that affect what users see in JupyterLab or exported HTML.

For tiny Python-only changes, the normal tests may be enough. For anything users
touch with a mouse, run this workflow.

## Start a signoff packet

From a clean worktree:

```bash
scripts/widget_agent_signoff.sh
```

The script writes a timestamped packet under `/tmp/quantem-widget-agent-signoff`
by default. It contains:

- `README.md` with the exact URLs and rules for the session,
- `report.md` for the pass/fail summary,
- `checklists/*.md` with per-widget human-drive steps,
- `screenshots/`, `videos/`, and `logs/` folders for evidence.

Optional modes:

```bash
scripts/widget_agent_signoff.sh --quick   # prepare packet + build/typecheck
scripts/widget_agent_signoff.sh --full    # prepare packet + full local gates
scripts/widget_agent_signoff.sh --artifact-dir /tmp/my-signoff
```

## Agent behavior

The agent must do the following for every affected widget:

1. Open the real notebook, docs page, or exported HTML in the in-app browser.
2. Drive the widget like a human: click, drag, wheel-zoom, scrub, resize, toggle,
   export, save/reopen where relevant.
3. Watch for blank canvases, stale state, console errors, layout wrapping,
   unreadable theme colors, and pointer-to-preview lag.
4. If anything is off, patch the code immediately.
5. Rebuild, refresh the browser, and redrive the same interaction.
6. Save final evidence in the signoff packet.

Do not claim success from source inspection alone. A widget interaction is only
signed off after it has been driven in a browser after the final code change.

## Evidence standard

For each widget, attach:

- one final screenshot after nontrivial interaction,
- a short video for motion-sensitive issues such as histogram center drag,
  ShowEDS band drag, Show4DSTEM detector drag, or Show3DSlices scrub/rotate,
- console-error notes,
- load/build/first-paint timing from notebook output, widget timing traits, or
  another direct timing signal,
- FPS or debug-HUD notes when available, including whether the kernel became
  busy during the tested interaction,
- a short list of fixes made during the session.

Screenshots are enough for static layout and theme review. Use video when the
question is whether the interaction feels attached to the pointer.

## Story-driven checks

Use [Storyboard](storyboard) as the primary source of behavior to
verify. The generated `checklists/*.md` files from
`scripts/widget_agent_signoff.sh` are lower-level reminders for concrete
interactions. A good report cites storyboard IDs, then lists the browser actions
and evidence collected for each story.

The generated packet checklists cover:

- **Show2D**: histogram min/max and center drag, FFT/profile/lens/ROI toggles,
  pan, wheel zoom, scale bar, colormap, export, light/dark readability.
- **Show3D**: frame scrub, play/pause, FPS, loop/bounce, histogram, export.
- **Show3DSlices**: independent panel sizes, crosshair, slice/angle/position
  sliders, 3D view controls, histogram, FFT/log/smooth/export.
- **Show4DSTEM**: detector drag, BF/ABF/ADF, diffraction pan/zoom, virtual image
  update, histogram, WebGPU path, export.
- **ShowEDS**: energy-band center/edge drag, synchronized sliders/readouts,
  element-map update, ROI rectangle/ellipse drag, periodic-table/element UI,
  log/scale/smooth/export, real-data behavior.
- **ShowFolder**: chooser or fallback path input, real folder survey,
  thumbnail/cache reuse, file/folder selection, selection save/load, and compact
  notebook state.
- **ShowDiffraction**: center picking, spot/ring add/remove, d-spacing labels,
  contrast/histogram, frame scrub, export.

## Release gate

Before tagging a release candidate, the report should say one of:

- `PASS`: all affected widgets were driven after the last code change.
- `PASS WITH LIMITATION`: a known limitation is documented and acceptable.
- `BLOCKED`: release should wait.

Do not hide a laggy or visually broken interaction behind a passing test suite.
If the interaction is not real time, either fix it or mark the release blocked.
For Show2D and Show3D, "real time" means the report names the real dataset,
MJGOAT/workstation backend, browser frontend, first-paint time, interaction
FPS/latency method, save/reopen result, and export reopen result.
