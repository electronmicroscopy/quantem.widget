# Agent-assisted development

`quantem.widget` is the interface layer connecting scientific data and
computational algorithms. We recommend agent-assisted development so you can
spend less time writing interface code and more time discovering scientific
insight. These copyable prompts give an agent the scientific and repository
context needed to work without guessing. Replace text in angle brackets before
sending a prompt. Do not paste credentials or private data into the prompt.

## Turn images into a shareable HTML page

The most common request: "make these images into a page I can send or embed."
`quantem-widget show2d` (separate images, one panel each) and
`quantem-widget show3d` (a stack or frames to scrub) do it in one command,
with no notebook. The page is one self-contained `.html` file that opens
offline, on a phone, or inside an `<iframe>`.

```bash
quantem-widget show2d before.dm4 after.dm4 --labels before after --title "Anneal 300C" --no-open --save-dir ./share
quantem-widget show3d ./drift_frames/ --title "Drift series" --no-open --save-dir ./share
# wrote ./share/drift_series_40x512x512_exact.html (55.4 MB, full float32)
```

Already have a widget in a notebook? `viewer.export_html("page.html")` writes
the same page. See [Command line](cli.md) for every input and flag.

Copy this prompt to have an agent do it and check the result:

```text
Make a shareable quantem.widget HTML page of these images:

<absolute paths to image files, one stack file, or one folder>

What they are and what the reader should compare:
<e.g. "before/after anneal, same area" or "drift series, 1 s per frame">

Embed it in a page? <no, or the host page / report it goes into>

Read docs/cli.md first. Use `quantem-widget show2d` for separate images (one
panel each) and `quantem-widget show3d` for a stack or frames to scrub. Run it
with --no-open and --save-dir <a folder outside any source checkout>, set --title
(it names the file) and --labels when the file names are not meaningful. Pass
--sampling and --units when the files carry no calibration (TIFF, PNG, NPY),
and --no-link-contrast when the images have different value ranges. Keep the
lossless default (--encoding full, no --image-bin). If the command refuses the
size, tell me the full and uint8 sizes it names and ask before packing uint8
or binning. Never crop, filter or bin to make it fit.

Report the exact command and the "wrote <path> (<size> MB, <packing>)" line.
Then serve the folder on localhost, open the page in a real browser, check
that navigator.gpu.requestAdapter() gives real hardware (not swiftshader),
and confirm every panel shows image pixels (not blank), the scale bar matches
the calibration, and for Show3D that stepping or playing frames changes the
picture. If I asked for an embed, add an <iframe> of the page to the host page,
check it renders there too, and give me the <iframe> snippet. Say plainly what
you verified and what you did not.
```

For many pages from a script, loop over the command: each run exits 0 with
one `wrote ...` line, or exits 1 with one `quantem-widget: ...` line naming the
flag that fixes the problem.

## Open an ARINA H5 file in Show4DSTEM

Use this when you have an ARINA `*_master.h5` file and want a live Jupyter
viewer rather than an exported report.

```text
I want to open this ARINA HDF5 master in Show4DSTEM in JupyterLab:

<absolute path to *_master.h5>

Use the local quantem.widget checkout and its locally built frontend when one is
available. Do not install, publish, or test a released package in place of the
local code. Read the repository instructions and existing Show4DSTEM loading
documentation first, then use the production quantem.gpu.io.load path and
quantem.widget.Show4DSTEM.

Before choosing uint8 or uint16, audit or locate an existing audit of the exact
detector-count range. Use uint8 only if the maximum is <=255 and there are zero
pixels above 255; otherwise preserve uint16. Report the maximum, pixels above
255, shape, dtype, backend, and estimated resident memory. Do not silently bin,
crop, downsample, or quantize the data. If the unmodified data cannot fit, stop
and explain the safe choices before changing it.

Create a small notebook outside the repository's canonical docs/tutorials
sources, keep private paths and data out of Git, start or reuse JupyterLab, and
give me the exact local URL. Render the returned widget naturally without an
unnecessary display(...) call.

Test the live notebook end to end in a real browser. Confirm that the
diffraction pattern appears, moving the scan position updates it, BF/ABF/ADF
detectors update the virtual image while dragging, and contrast/zoom work.
Record the browser WebGPU adapter and report any CPU/software fallback or
console error. Leave the tested notebook open for me and summarize exactly what
was verified and what was not.
```

See [IO](api/io.md) and the
[Show4DSTEM API](api/show4dstem.md) for the underlying interfaces.

## Create a widget from a notebook or description

Use this for a new viewer or scientific interaction that does not already fit
one of the existing widgets.

```text
I want to create a quantem.widget for this scientific workflow.

Existing notebook, if available:
<absolute path or "none">

Scientific goal and user decisions the widget should support:
<describe the data, controls, outputs, and what a scientist needs to decide>

Representative data shape, dtype, units, and backend:
<details or "unknown">

Work from the local checkout. Read the repository instructions, widget UI
protocol, performance guidance, and the closest existing widget before editing.
If the notebook exists, preserve its scientific behavior and identify which
steps belong in reusable package code versus the example notebook. If it does
not exist, make the smallest synthetic or public-data notebook that captures
the workflow before implementing the production widget.

Design a small public Python API with clear scientific parameter names and a
matching browser interaction model. Preserve raw data and physical units. Do
not silently crop, bin, quantize, change coordinate conventions, or fall back
from GPU to CPU. Separate live drag previews from committed state, use
requestAnimationFrame for high-frequency pointer UI, and follow existing
traitlet, React, WebGPU, saved-state, and HTML-export patterns.

Add focused tests for the scientist's workflow and concise documentation. Keep
canonical docs/tutorial notebooks interactive; do not bake static GitHub widget
state into them. Build the local frontend, open a fresh live notebook or export,
drive every new control in a real browser, and collect pixel, state, console,
and responsiveness evidence. Show me the local result and diff for review. Do
not push or open a pull request unless I explicitly ask.
```

## Prepare a pull request

Use this after a focused change is ready for review. This prompt prepares the
branch and PR material but deliberately stops before publishing.

```text
Prepare a focused quantem.widget pull request for:

<issue URL or concise goal>

First read CONTRIBUTING.md, docs/maintainer/pull-requests.md, the
pull-request template, and any instructions nearest the changed files.
New public widgets and cross-widget refactors need an issue or maintainer
discussion first. In-widget bug fixes and features do not.

Inspect the worktree, branches, and remotes; preserve unrelated user changes.
Fetch the authoritative upstream main branch before you compare or create the
feature branch. Follow the repository's branch-naming rules; if none are
present, use a short creation-date prefix such as aug-8-show2d-contrast.

Keep one scientific or documentation goal in the PR. Stage named files only.
Exclude private data, machine-specific paths, notebook outputs, generated HTML,
screenshots, browser profiles, node_modules, and scratch artifacts unless the
PR explicitly needs them.

Run the smallest focused checks while editing and the repository-required
Python/frontend checks before handoff. For an interaction change, also drive
the actual notebook or exported HTML in a real browser and report the data
shape, dtype, backend, WebGPU adapter, interaction tested, pixel evidence,
latency or responsiveness evidence, console errors, and anything not verified.

Then show me git status, the diff summary, focused test results, browser proof,
the proposed single-line commit message, and a completed PR body using the
repository template. Inspect the final GitHub Files diff for accidental noise.
Do not push the branch or open the PR until I explicitly approve publication.
Preserve my configured Git identity and do not add a Co-authored-by trailer.
```

When publication is approved, push to the contributor's fork and let the human
open the upstream PR unless they explicitly ask the agent to do that step.
