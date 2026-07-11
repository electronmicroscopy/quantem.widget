# Widget UI protocol

Use this page when editing widget frontend controls, reviewing a widget PR, or
asking an agent to make UI changes. The goal is for every viewer to feel like
one scientific tool family, even when the underlying data are very different.

For developer-facing `ui_mode`, `show_*`, and control-visibility tables, use
the [UI Guide](../developer/ui-guide). This maintainer page is the internal protocol
for frontend wording, toolbar layout, export labels, and review checks.

## Command text

Use Title Case for command buttons and toolbar actions:

- `Copy`
- `Export`
- `Reset`
- `Add`
- `Clear`
- `Undo`
- `Save Band`
- `Center View`

Keep scientific acronyms, detector labels, and file-format names uppercase when
the uppercase form is the term users recognize:

- `FFT`
- `ROI`
- `BF`, `ABF`, `ADF`
- `HTML`, `PNG`, `GIF`, `MP4`, `CSV`, `JSON`

Do not use all-caps for ordinary commands such as `COPY`, `EXPORT`, `CLEAR`,
or `UNDO`.

## Compact control labels

Dense widget toolbars and control rows should not use decorative colons. Use:

- `Scale`
- `Color`
- `Auto`
- `Smooth`
- `Link`
- `Zoom`
- `Pan`
- `Contrast`
- `ROI`

Use colons in explanatory prose, tooltips, documentation, and status text when
they make the sentence clearer. Avoid them inside compact controls because
they waste horizontal space and create inconsistent rhythm.

## Control Pairs

Treat a compact label and the control it names as one UI unit. This applies to
labels paired with switches, dropdowns, sliders, icon buttons, and compact text
buttons:

- `Auto` + switch
- `Smooth` + switch
- `Color` + menu
- `Scale` + menu
- `fps` + slider/value
- `avg` + menu

On mobile and narrow layouts, the row may wrap, but the pair must stay
together. Do not allow a label to remain on one line while its switch or menu
wraps to the next line. Use an inline-flex pair wrapper or an equivalent stable
layout primitive with small fixed gaps.

Every compact label that names a live control (`Scale`, `Color`, `Colorbar`,
`Profile`, `FFT`, `Lens`, `Auto`, `Smooth`, and the `Link` toggles) must render
inside a pair wrapper, not as a bare sibling of its control. A bare label is the
one that drifts away from its switch or menu when the toolbar wraps. Give these
labels full-strength text (`themeColors.text`); reserve the muted text color for
status text only (export status, page status, zoom readout), never for a label
that names a control the user can toggle.

**Scoped sub-groups.** When several toggles share one governing word (for
example `Link` scoping `Zoom`, `Pan`, `Contrast`, and `Denoise`), wrap the whole
set in a single bordered sub-group led by that word. Repeating or bordering the
scope keeps it legible when the group wraps; a lone `Pan` switch that wraps away
from a distant `Link` label reads as an unscoped control.

**Mode-gated knobs get their own row.** Occasional sliders that appear only when
a mode is on (lens magnification and window size, the denoise method/σ/bin knobs,
the underlay blend/stretch knobs) belong on their own conditional row, not
appended to an always-visible row. This keeps the everyday rows short and stops a
toggled-on tool from forcing the base controls to wrap.

Browser signoff for dense controls must include at least one narrow viewport.
For Show2D and Show3D, check the main image controls, FFT controls, playback
controls, linked zoom/pan/contrast controls, and any page/column controls that
can wrap.

## Toolbar order

When a widget has these actions, prefer this order:

1. Widget-specific controls and mode switches
2. `Copy`
3. `Export`
4. Export status text
5. `Reset`

Keep existing widget-specific exceptions only when the scientific workflow
depends on them. If changing the order, compare the result against Show2D,
Show3D, Show4DSTEM, and ShowEDS before committing.

## Overflow (More) menu + always-visible active-reduction badge

Dense widgets outgrow a single toolbar. Tier the controls instead of wrapping
them off screen. Show2D is the reference implementation; Show3D, Show4DSTEM,
and ShowEDS should mirror this pattern.

**Tiering.** Keep the everyday viewer toggles top-level and push occasional
tools into a `More` overflow menu in the right action cluster (next to `View`,
`Export`, `Copy`).

- Top-level (Show2D): `Profile`, `FFT`, `Lens`, `Colorbar`, panel `Reorder`,
  `Panels`, and the per-panel hide/star controls.
- Inside `More` (Show2D): `ROI`, `Denoise` (`show_denoise`), and `Diff`.

Move controls into `More` by relocating the existing control and its
handlers/side-effects. Do not duplicate the toggle logic, and preserve each
control's render guard (for example, `ROI` only when not a gallery; `Diff`
under its existing availability condition). Render each moved control inline in
the menu with its real control type (a mode select stays a select, a toggle
stays a switch); never downgrade a mode to a bare on/off just because it moved.
Use the shared MUI `Select`/`Menu` with `themedSelect`/`themedMenuProps`; never
a native `<select>`.

**Active badge (MUST).** An active reduction or tool must never be invisible.
Compute a count of the active items inside `More` and show it as an MUI `Badge`
on the button; accent the button and the active menu items so a collapsed-away
tool is still discoverable. Reuse the existing accent idioms (the `Panels N/M`
count and the reorder-button accent).

```text
moreActiveCount = (roiActive ? 1 : 0) + (diffActive ? 1 : 0) + (showDenoise ? 1 : 0)
```

**Collapse banner (MUST).** Reduction banners (denoise, crop/pad) normally live
inside the controls block and disappear when the user collapses controls. That
would hide an active reduction, which the house rule forbids. When controls are
collapsed and a denoise or view banner is non-empty, render a compact accent
badge in the always-visible title row, mirroring the surviving `2× binned`
badge. Strip the trailing "how to undo" hint for the compact label and keep the
full banner text in the `title` tooltip.

**Collapse toggle affordance.** Give the `Controls` collapse toggle a
chevron/open-state indicator (and `aria-expanded`) so its state is legible at a
glance.

## Dynamic labels

Scientific labels can be long and data-dependent: page labels, run names, file
names, lambda/RMSE summaries, detector names, export statuses, and frame labels.
Do not let that text resize the whole toolbar or shift nearby controls while the
user scrubs pages or frames.

For dynamic labels inside dense toolbars:

- Give the label a bounded width with `overflow: hidden`, `text-overflow:
  ellipsis`, and `white-space: nowrap`.
- Keep the full value in `title` and the relevant `aria-label`.
- Use fixed or reserved widths for neighboring sliders and compact controls.
- Use tabular numbers for counters such as `2/11`, `10/11`, and `4 fps`.
- Put detailed metadata in a tooltip, panel label, report table, or secondary
  line instead of making the primary toolbar grow with every dataset.

## Export labels

Export menu labels should tell users what will be saved:

- Format or mode: `HTML`, `GIF`, `MP4`, `PNG`
- Encoding: `Exact float32`, `Exact uint16`, `Quantized uint8`
- Reduction: `Downsample 2x`, `Binned 4x`, or the widget-specific reducer
- Estimated size or render work when available

Examples:

- `HTML exact float32 (82 MB)`
- `HTML quantized uint8 (21 MB)`
- `Binned 4x uint16 (180 MB)`
- `GIF medium (1.6 MB work)`

Do not hide scientific reductions behind vague words like "small" unless the
menu also says what changed.

For animation exports, the GUI may not know the compressed GIF/MP4 size before
encoding. In that case, show estimated uncompressed RGB render work and keep
the label explicit. Do not imply this estimate is the final file size.

Keep advanced animation controls out of the primary toolbar unless they become
common user actions. The GUI should expose the simple path, usually
`GIF low/medium/high` and `MP4 low/medium/high`; Python and maintainer smoke
reports should cover advanced options such as frame labels, background color,
bounce playback, panel gap, and dry-run planning.

## Visual testing

After changing controls, rebuild and drive the widget in a browser. At minimum:

1. Open the live Jupyter widget or standalone exported HTML.
2. Toggle the controls touched by the change.
3. Verify labels wrap cleanly in a narrow viewport.
4. Check light and dark docs themes when the page is theme-sensitive.
5. Confirm there are no console errors.

For interaction-sensitive changes, follow
[Agent signoff](widget-agent-signoff) and the relevant
[Storyboard](storyboard) file.
