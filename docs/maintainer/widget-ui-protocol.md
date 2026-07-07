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
