# Show3D remote tunnel scrub timing, 2026-07-12

## Goal

Measure the slow Show3D scrub path before optimizing it. The target deployment is
JupyterLab in an Apple Silicon browser, with the kernel and data on a remote GPU workstation over an
SSH `-L` tunnel. Playback already uses a sliding buffer; interactive scrubbing
commits `slice_idx` and receives one `frame_bytes` payload per tick, so this note
separates the Python, Comm, browser decode, and paint-proxy costs for that path.
This is the measurement note for
[S3D-20](storyboard-show3d.md#s3d-20-scrub-full-resolution-movies-over-a-remote-jupyter-tunnel).

Hard rule: source data stay full resolution. Any display-side reduction must be
explicit, announced once, and reversible to native pixels.

## User story: mount now, hydrate in the background

As a remote JupyterLab user opening large experimental Show3D data through an SSH
tunnel, I need the widget shell, controls, and first canvas to mount immediately
so I can see that the notebook is alive. Full-resolution frame fetches, preview
fetches, playback cache warming, and GPU uploads may continue in the background,
but the initial widget display must not be blocked behind a full native
`frame_bytes` payload over the Jupyter Comm channel.

Acceptance criteria:

- Frame-server-backed Show3D construction syncs lightweight traits first and
  leaves initial `frame_bytes` empty.
- The browser fetches and paints the initial native frame asynchronously after
  mount.
- Slider release and zoom/settle paths keep native full-resolution pixels
  reachable.
- Any drag-time display reduction announces the active factor and how to restore
  native resolution.

## Instrumentation added

Show3D now syncs per-message timing metadata:

| Channel | Python fields | Browser fields |
| --- | --- | --- |
| `frame_bytes` scrub | `pythonPrepareMs`, `pythonWireMs`, `pythonEncodeMs`, `pythonTraitSetMs`, `sendTimeMs`, `bytes`, `slice` | `browserReceiveLatencyMs`, `jsDecodeMs`, `endToEndUiLatencyMs` |
| `_buffer_bytes` playback | `pythonPrepareMs`, `pythonEncodeMs`, `pythonTraitSetMs`, `sendTimeMs`, `bytes`, `start`, `count` | `browserReceiveLatencyMs`, `jsDecodeMs` |
| `_scrub_preview_bytes` drag preview | `pythonPrepareMs`, `pythonEncodeMs`, `sendTimeMs`, `bytes`, `fullBytes`, `factor`, `idx` | `browserReceiveLatencyMs`, `jsDecodeMs`, `endToEndUiLatencyMs` |

Browser samples are also mirrored to
`window.__quantemShow3DPerf.transportSamples` for manual inspection. The
automated benchmark mode is:

```python
widget.benchmark_request = {
    "token": "remote-scrub-20260712-1",
    "mode": "scrubTransport",
    "sampleCount": 12,
    "settleMs": 80,
    "timeoutMs": 8000,
    "label": "remote GPU tunnel Show3D scrub",
}
```

The result appears in `widget.benchmark_result`.

After the scrub-preview fix, use `mode="scrubPreviewTransport"` to measure the
drag-time preview path without forcing a native `slice_idx` commit on every
sample:

```python
widget.benchmark_request = {
    "token": "remote-preview-20260712-1",
    "mode": "scrubPreviewTransport",
    "sampleCount": 12,
    "settleMs": 80,
    "timeoutMs": 8000,
    "maxBytes": 16 * 1024 * 1024,
    "label": "remote GPU tunnel Show3D scrub preview",
}
```

## Measurement matrix

Fill this table only with real tunnel measurements from the Apple Silicon
browser to the remote GPU kernel. Do not substitute localhost numbers.

| Run | Widget/data | Shape | Panels | Bytes per scrub frame | Tunnel | Python prepare avg/p95 ms | Python encode avg/p95 ms | Browser receive avg/p95 ms | JS decode avg/p95 ms | UI latency avg/p95 ms | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-12 initial scrub | Show3D from an anonymized panel array; first 4 frames of a real panel stack | `(4, 2048, 18432)` float32 display stack from source `(9, 69, 2048, 2048)` | 9 native 2048 x 2048 panels concatenated for the current Comm path | 150,994,944 B (144 MiB) | Apple Silicon browser -> SSH tunnel -> remote GPU kernel | 0.018 / 0.046 | 65.3 / 104.6 | 806.0 / 1209.8 | 0 / 0 | 895.1 / 1248.8 | Browser benchmark `scrubTransport`; excluded one stale pre-request sample (`seq=6`) from an earlier unmounted-widget attempt. The time axis was shortened to 4 frames because the remote host had only about 8-16 GiB available; spatial pixels and per-scrub payload stayed native and unreduced. |
| 2026-07-12 post-fix smoke | Show3D synthetic full-spatial stack | `(2, 2048, 18432)` float32 | 9 native 2048 x 2048 panels concatenated | 150,994,944 B (144 MiB) native frame; 16,785,408 B preview | Apple Silicon browser -> SSH tunnel -> remote GPU kernel | not sampled | not sampled | not sampled | not sampled | not sampled | Browser-driven slider drag emitted `[Show3D scrub preview] displaying 3x reduced frames during slider drag; release the slider or zoom/settle the view to request native full resolution.` Canvas stayed mounted; frame control moved between `1/2` and `2/2`. Synthetic was used only for the post-fix smoke because the real `.npz` archive inflated the full source array before slicing and blocked the scratch kernel. |
| 2026-07-12 post-fix preview benchmark | Show3D synthetic full-spatial stack, `mode="scrubPreviewTransport"` | `(2, 2048, 18432)` float32 | 9 native 2048 x 2048 panels concatenated | 150,994,944 B (144 MiB) native frame; 16,785,408 B preview | Apple Silicon browser -> SSH tunnel -> remote GPU kernel | 753.0 / 1554.8 cold-inclusive; warm frame-1 samples ~3.9 ms | 6.9 / 9.3 | 27.7 / 51.3 | 0 / 0 | 27.7 / 51.3 | Four samples, factor 3, preview shape `683 x 6144`. Two frame-0 samples had cold preparation spikes (~1.45-1.55 s), while frame-1 warm samples prepared in ~4 ms. Transport/UI latency for the 16.8 MB preview was realtime-ish over the tunnel. |
| 2026-07-12 reference constructor profile, first frame only | `Show3D.from_panel_folders` target data, six real experimental PNG panel folders | first frame per panel, each native `2048 x 2048` | 6 | 100,663,296 B native combined frame; initial Comm frame payload 0 B | Python profiler on the remote GPU workstation | load/decode total 295.2 ms; construct 142.5 ms | n/a | n/a | n/a | total 439.7 ms | Real local time-series data path; full source stays on disk and reachable by frame server. This is the only measured path under the 0.5 s mount target. |
| 2026-07-12 reference constructor profile, eager 4 frames | Previous notebook strategy: decode four frames per six experimental panels before constructing widget | `(4, 2048, 12288)` logical six-panel native view | 6 | 100,663,296 B native combined frame; initial Comm frame payload 0 B after defer | Python profiler on the remote GPU workstation | load/decode total 403.6 ms; construct 557.6 ms | n/a | n/a | n/a | total 963.1 ms | Same real folders. Even four eager frames misses the 0.5 s target before user interaction starts. |
| 2026-07-12 reference constructor profile, eager 12 frames | Previous notebook strategy: decode twelve frames per six experimental panels before constructing widget | `(12, 2048, 12288)` logical six-panel native view | 6 | 100,663,296 B native combined frame; initial Comm frame payload 0 B after defer | Python profiler on the remote GPU workstation | load/decode total 947.6 ms; construct 1644.1 ms | n/a | n/a | n/a | total 2593.6 ms | Same real folders. This confirms background/lazy hydration is required; eager preload cannot meet 0.5 s. |

## S3D-20 drive result

On the `52186` tunnel run, the in-app browser drove the live JupyterLab widget:
manual frame-slider drag, keyboard frame step, and Play/Pause. The canvas stayed
mounted, the frame readout moved to `1/2`, and the browser logged the required
preview announcement:

```text
[Show3D scrub preview] displaying 3x reduced frames during slider drag; release the slider or zoom/settle the view to request native full resolution.
```

The top-level JupyterLab page could not reliably read
`window.__quantemShow3DPerf` because the widget code ran in a blob/module
context, so the durable measurement path is the widget-level
`benchmark_result` produced by `mode="scrubPreviewTransport"`.

## Initial finding

For the measured 144 MiB scrub frame, Python data preparation was effectively
free (`~0.02 ms`) and `tobytes()` encoding was tens of milliseconds
(`~65 ms` average). The dominant cost was transport from Python trait send to
browser receive and paint proxy: about `0.9 s` average and `1.25 s` p95 after
excluding the stale pre-request sample. This confirms the slow path is the
Comm/tunnel payload size, not device-to-host preparation or browser decode.

## Fix applied

Interactive slider drag now has a Show3D scrub-preview transport. During active
drag, JS first uses local GPU/cache/buffer/server paths. If those miss and the
view does not require browser-only transforms, JS sends a coalesced
`_scrub_preview_request` instead of committing `slice_idx` on every pointer
tick. Python returns `_scrub_preview_bytes` capped to a 16 MiB drag-time display
budget. For the measured 144 MiB frame this implies a 3x strided display
preview, about 16 MiB per drag update.

This is display-side transport reduction only. The source arrays, display stack,
and committed `frame_bytes` path remain full resolution. Releasing the slider
still commits `slice_idx` and requests the native full-resolution frame. When a
factor greater than 1 is active, Python prints and the browser logs:

```text
[Show3D scrub preview] displaying {factor}x reduced frames during slider drag; release the slider or zoom/settle the view to request native full resolution.
```

The frame-server-backed initial mount now follows the same UX principle:
Python does not put a full native `frame_bytes` payload into the initial widget
state. The frontend mounts from metadata plus the frame server URL, then fetches
and paints the initial native frame asynchronously. This changes transport
ordering only; source arrays and native frame availability are unchanged.

## Lazy real-data source

`Show3D.from_panel_folders(...)` is the large-time-series path for remote Jupyter.
It inventories the panel image folders, decodes only frame 0 from each panel for
bootstrap dimensions and contrast, and then installs a lazy full-resolution
panel source. The slider receives the full real frame count immediately. Native
frame pixels are decoded on demand by the frame server, cached with a small LRU,
and returned as exact float32 bytes. No source frame is binned, cropped, or
replaced.

This is the path that makes the 0.5 s load target plausible. The real-data
profile above measured 439.7 ms for first-frame decode plus widget construction.
The same profile showed that eager 4-frame and 12-frame notebook construction
cannot meet the target, so future large-time-series notebooks should mount via
`from_panel_folders` and let the browser/server warm the native cache in the
background.

## Interpretation checklist

- If `browserReceiveLatencyMs` dominates, the tunnel and Comm payload size are
  the floor. Optimization should reduce scrub-time transport without changing
  the stored full-resolution data.
- If `pythonPrepareMs` dominates, inspect device-to-host copies, diff/filter
  preparation, and panel concatenation.
- If `pythonEncodeMs` dominates, inspect `tobytes()` allocations and exact
  buffer reuse options.
- If `jsDecodeMs` or `endToEndUiLatencyMs` dominates, inspect browser decode,
  canvas upload, and paint scheduling.
- Compare scrub against `_buffer_bytes` playback samples to quantify what the
  existing sliding-send-buffer path already solves.
