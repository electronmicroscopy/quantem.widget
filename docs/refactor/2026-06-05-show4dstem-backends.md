# Show4DSTEM — Backend vs Backendless refactor

Architecture ledger for the phased refactor that landed on
`widget-show3d-show4dstem-kernels` 2026-06-05.

## Canonical operator API (after this refactor)

```python
from quantem.widget import load, Show4DSTEM

# Auto-pick Python compute (CUDA / MPS / CPU based on data type).
w = Show4DSTEM(load(path))

# Web compute (kernel still alive, all reductions run in browser GPU).
# Works on ANY GPU: NVIDIA, AMD, Apple, Intel.
w = Show4DSTEM(load(path), backend='web')

# Standalone HTML — no kernel needed after export.
w = Show4DSTEM(load(path), backend='web', offline_codec='bslz4',
               data_url='show4dstem-data')
w.export_html('show4dstem.html')
```

Three lines, four backend options. `backend=` is the only widget-level
switch: `None` (default; auto) or `'web'`. Python compute backend is
picked from the data type — operator never has to think about
`TorchBackend` vs `MetalRawBackend`.

### Legacy aliases (still work, soft-deprecated)

- `backend='browser'` / `backend='webgpu'` ≡ `backend='web'`
- `offline=True` ≡ `backend='web'`
- `Show4DSTEM_MACBOOK(load(path, backend='mps'), ...)` — explicit
  Mac sampling helper. `Show4DSTEM(load(path, backend='mps'))` is
  the canonical short form.

## Why

Show4DSTEM had grown two divergent classes (~3.6k LOC total) — `Show4DSTEM`
(torch on CUDA/MPS/CPU) and `Show4DSTEMMPS` (raw-Metal chunked for Phil's
19.3 GB Sample-class stacks where torch.MPS hits the >2^31-element buffer
limit). The compute paths were already abstracted in
`kernels/compute/backends.py`, but the MPS subclass duplicated lifecycle
logic (fast_vi sidecar, radial cache, multi-dataset proxy) by reaching into
`self._data.vi.*` directly. Adding a third backend (WebGPU online / offline)
would have meant carving a third parallel widget file. This refactor
collapses the abstraction so backend selection is a runtime concern, not a
class-hierarchy concern.

Tied issues: #772 (single-source quantem.live widgets/), #775 (single-source
WebGPU frontend), #746/#745/#744/#743 (MetalRaw lifecycle + perf), #747 (5D
time-series binned), #754 (backendless HTML with sibling .h5),
#740/#737/#763 (Mac MPS expansion).

## Framing — backend vs backendless

| Mode | Data lives | Compute runs | Status |
|---|---|---|---|
| **Backend / Torch** | Python torch.Tensor on CUDA / MPS / CPU | Python via torch | shipped (`TorchBackend`) |
| **Backend / MetalRaw** | Python `MPSChunked4DSTEM` (Metal unified-memory chunks) | Python via raw Metal kernels (`MetalVirtualImage`) | shipped (`MetalRawBackend`) |
| **Backendless / Web (live kernel)** | Browser GPU buffer; kernel ships uint8-packed stack via `_offline_stack` trait | Browser WebGPU (`js/engine/compute.ts`) | shipped — `Show4DSTEM(data, backend='web')` ≡ `offline=True` |
| **Backendless / HTML (no kernel)** | Browser, embedded in `<script>` block of standalone HTML | Browser WebGPU (`js/engine/compute.ts`) | shipped — `widget.export_html()` + `_pack_offline_bslz4()` |

## Phase 1 — Shipped 2026-06-05

Commits `831bb577` (refactor) + `e208cc8b` (snapshot label work).

### `kernels/compute/backend.py` (new, 109 LOC)

`ComputeBackend` `Protocol`, runtime-checkable. Required surface:

```python
backend.scan_shape    -> (int, int)
backend.det_shape     -> (int, int)
backend.n_frames      -> int
backend.device        -> str | torch.device
backend.capabilities  -> tuple[ComputeCapability, ...]

backend.frame(idx)                     -> np.ndarray (det_r, det_c)
backend.masked_sum(det_mask)           -> np.ndarray (scan_r, scan_c) float32
backend.mean_dp()                      -> np.ndarray (det_r, det_c) float32
backend.reduce_frames(idx, "mean"|"sum"|"max") -> np.ndarray (det_r, det_c) float32
backend.center_of_mass(det_mask=None)  -> (com_col, com_row) flat (N,) float32
```

Optional capability hooks are advertised via `capabilities` tuple and
called only after a `'<cap>' in backend.capabilities` check. They're NOT
part of the runtime Protocol so `isinstance(b, ComputeBackend)` passes
for any backend that implements just the required surface.

Capability strings:
- `'fast_sidecar'` — `ensure_fast_sidecar(verbose)`, `cache_fast_presets(masks)`, `fast_bin`, `has_fast`
- `'radial_cache'` — `ensure_radial_cache(row, col)`, `radial_cache_ready(row, col)`, `radial_masked_sum(...)`
- `'row_prefix_exact'` — marker that radial cache uses exact row-prefix sums (no bin)
- `'multi_dataset'` — `set_active_dataset(idx)`, `multi_n_ready`, `multi_names`, `multi_active_idx`, `multi_total()`, `set_multi_ready_callback(cb)`

### `kernels/compute/backends.py` — renames + lifecycle move

- `TorchCompute` → `TorchBackend` (alias kept for one release)
- `MetalCompute` → `MetalRawBackend` (alias kept for one release)
- `MetalRawBackend.capabilities` advertised dynamically based on what
  `ChunkedFrames` actually supports:
    - `'fast_sidecar'` always
    - `'radial_cache'` + `'row_prefix_exact'` when `data.vi.row_prefix_enabled`
    - `'multi_dataset'` when `data` has `set_active` + `on_ready`
- All MPS lifecycle methods now live ON the backend, not in the widget subclass:
  - `ensure_fast_sidecar(verbose)` (blocking; idempotent on already-binned data)
  - `cache_fast_presets({"bf": mask, ...})` → `dict[str, np.ndarray]`
  - `ensure_radial_cache(row, col, *, idle_delay_s=0.75)` (async; idempotent at the same center; cancels stale builds when center moves)
  - `radial_cache_ready(row, col)` / `radial_masked_sum(...)` / `radial_building` / `radial_error`
  - `set_active_dataset(idx)` / `multi_n_ready` / `multi_names` / `multi_active_idx` / `multi_total()` / `set_multi_ready_callback(cb)`

### `show4dstem_mps.py` — UI-only subclass (908 → 802 LOC)

`Show4DSTEMMPS` keeps:
- The four MPS traits (`fast_interaction`, `fast_interaction_ready`,
  `fast_interaction_building`, `radial_interaction_ready`,
  `radial_interaction_building`) and their observers
- The detector preset cache (BF/ABF/ADF/HAADF arrays cached in instance attrs)
- The numpy ROI-mask builder (`_detector_mask_np`)
- `set_fast_interaction(enabled, wait=)` / `wait_for_fast_interaction` /
  `wait_for_radial_interaction` operator-facing controls

Removed:
- Direct `self._data.vi.*` calls — every Metal interaction now flows through
  `self._compute.<lifecycle_method>`
- The radial-cache background thread + idle-delay polling — now owned by
  `MetalRawBackend.ensure_radial_cache`
- The multi-dataset proxy wiring details — now owned by
  `MetalRawBackend.set_multi_ready_callback`

### Verification (Phase 1)

- host CUDA path: `Show4DSTEM(load('.../device_master.h5', det_bin=4))`
  → `_compute = TorchBackend`, `capabilities = ()`. masked_sum + frame +
  mean_dp all produce expected shapes/dtypes. `live notebook publish` hook
  baked real Show4DSTEM canvas PNG into ipynb in 3.8 s. Verified
  bit-identical against pre-refactor `publish_dict` output.
- Phil MPS path (`load(backend='mps', det_bin=4)`):
  - `_compute = MetalRawBackend`, `capabilities = ('fast_sidecar',)`
  - `masked_sum(full mask)`: 13 ms (real-time)
  - `mean_dp()`: 57 ms, `frame(0)`: 0 ms
  - `virtual_image_bytes` (1 MB) + `frame_bytes` (9 KB) populated
  - Zero Chrome processes touched on Phil

## Phase 3 — Backendless / Offline (already shipped, verified post-refactor)

`widget.export_html(path)` produces a self-contained `.html` that mounts
the live anywidget JS bundle with all current widget state embedded.

The existing implementation uses:
- `_clone_for_html_export()` → builds an export-only widget with
  `_offline_stack` (uint8 quantized) + `_offline_bslz4` (HDF5
  bitshuffle+LZ4 metadata for 8-plane GPU fast path) + `_offline_bad_px`
  populated
- `embed_minimal_html()` from `ipywidgets.embed` writes the standalone HTML
- Browser-side `Show4DSTEMCompute` (`js/engine/compute.ts`) does ALL
  subsequent reductions in WebGPU: `maskedSum`, `frameAt`, `reduceFrames`,
  `maskedCoM` — same shaders the live offline-mode interactive widget uses

Post-refactor verification (host → Linux Chrome on `:1`):
- `Show4DSTEM(load('.../det_bin=4'))` → `export_html('/tmp/x.html')` →
  2 MB self-contained file in <0.1 s
- Open in Linux headed Chrome → 8 canvases mounted (BF detector + CBED +
  scale bars + histograms + sliders), full interactivity
- Title, magma CBED, virtual image, ROI mode, presets all working

Conceptually `Show4DSTEM` in offline mode IS a "WebGPUOfflineBackend" — the
Python widget object has no `self._compute` after `_clone_for_html_export`
runs; all derived compute happens in the browser. The Protocol stays
unviolated because offline-export widgets are write-only (the operator
exports them; they're never queried via `self._compute.<method>` after).

Open question: #754 (backendless HTML that auto-loads a sibling .h5).
Today's `_pack_offline_bslz4` bakes the FULL data into the HTML. The
sibling-.h5 alternative would have the HTML fetch the .h5 via `fetch()` +
do bslz4 decode in the browser. That's a future optimization; not needed
to call Phase 3 "shipped".

## Phase 2 — superseded — collapsed into `backend='web'`

The original Phase 2 sketch proposed a custom Comm RPC backend
(`WebGPUOnlineBackend`) that would push per-call masked_sum / frame /
reduce_frames requests to JS. After examining the existing offline path
we realized the same JS WebGPU shaders + `_offline_stack` trait already
deliver the full online backendless experience with a live kernel; no
Comm RPC needed. The dropped `WebGPUOnlineBackend` Python skeleton (in
bcac8e7f) was deleted in e0b92af2. The operator-facing alias for this
mode is now `backend='web'` (e.g. `Show4DSTEM(data, backend='web')`).

**Realization 2026-06-05: Phase 2 is already done.** The widget's
existing ``offline=True`` knob does exactly this — kernel ships a uint8-
packed 4D stack via the ``_offline_stack`` trait, JS mounts
``Show4DSTEMCompute``, all subsequent reductions (ROI drag, frame
scrub, ADF change) happen in the browser GPU via the same WGSL shaders
used by ``export_html``. The kernel stays alive but isn't used for
compute. "Offline" here means "compute offline from Python" not "kernel
offline."

**API:** ``Show4DSTEM(data, backend='webgpu')`` (added 2026-06-05) maps
to ``offline=True`` for a clearer name. Same runtime behavior.

**Verified end-to-end** on Sample Logic-013 (det_bin=8):
``backend='webgpu'`` → 92 MB ``_offline_stack`` packed → JS receives the
trait → mounts WebGPU pipeline → all UI compute runs browser-side.

**Universal GPU access** (any modern GPU + browser):
- NVIDIA / AMD / Intel users without a CUDA-built torch
- Apple M-series Mac users without working torch.MPS
- Laptop-first adopters without Python GPU drivers
- Anyone whose data exceeds torch.MPS's >2^31 ceiling

The earlier ``WebGPUOnlineBackend`` Python skeleton (committed in
bcac8e7f) duplicated this. Removed 2026-06-05 — the existing
``_offline_stack`` + ``Show4DSTEMCompute`` flow IS the online backend.

### How it works (no new code paths)

```python
Show4DSTEM(data, backend='webgpu')      # alias of offline=True
Show4DSTEM(data, offline=True)          # original knob, equivalent
```

Both routes call ``_pack_offline(offline)`` which:
1. Clips ``data`` to uint8 (lossless for counts <=255).
2. Stores the packed bytes in the ``_offline_stack`` traitlets.Bytes.
3. Sets ``self.offline = True``.

JS side reads ``_offline_stack`` on mount and calls
``Show4DSTEMCompute.create(stack, scanCount, detSize)`` — the same
shaders ``export_html`` uses. Every subsequent ROI drag / scrub /
ADF change runs on the browser GPU.

### Future v2 — bslz4 streaming for >2 GB stacks

Already implemented via ``offline_codec='bslz4'`` + a companion
``data_url`` directory. The ``_pack_offline_bslz4`` path chunks the
stack so it doesn't have to fit in one Comm message. Documented in
``show4dstem.py:827`` (``_pack_offline_bslz4``).

## Follow-ups (post-Phase-1)

- **#772 closure** — quantem.live `widgets/show4dstem_*.py` (~3.3 kLOC)
  becomes a `from quantem.widget.show4dstem import Show4DSTEM` re-export.
  Gated on this commit landing in widget upstream + live notebooks getting
  re-pointed. ~1 hour work, separate commit.

## Migration signoff (2026-06-05)

Signoff tag: `show4dstem-migration-signoff-2026-06-05`.

Verified paths:

- **CUDA / host:** full no-bin `_29` (`512 x 512 x 192 x 192`) loads and the
  widget constructs; multi-dataset CUDA works for full no-bin and binned stacks.
- **MPS / Phil:** gold full no-bin (`256 x 256 x 192 x 192`) opens through the
  raw-Metal viewer, the async fast sidecar becomes ready, real browser drag
  changes the view, and the browser rAF probe measured about 120 FPS.
- **WebGPU live/browser compute:** `backend="web"` runs reductions in the
  browser WebGPU path with `navigator.gpu == true`.
- **WebGPU exported HTML:** full no-bin uses HTML plus bslz4 companion chunks.
  A true single self-contained HTML file is intentionally not the large-data
  path; it is only appropriate for small inline exports.
- **WebGPU multi-dataset:** 5D stacks export as lazy bslz4 volumes. The
  Dataset slider fetches/decodes the selected volume on demand and avoids
  keeping every full no-bin dataset resident at once.

The Python API for the large-data browser path is:

```python
w = Show4DSTEM(data_or_stack5d, backend="web", offline_codec="bslz4", data_url="show4dstem-data")
w.export_html("show4dstem.html")
```

Serve `show4dstem.html` and `show4dstem-data/` from the same HTTP directory.
- Capability `'fft'` — `MetalRawBackend` could implement FFT via
  `MetalVirtualImage`'s row-prefix engine for exact mode. Today FFT is
  CPU-side numpy in Python (`np.fft.fft2`). Worth measuring.
- Capability `'com_cache'` — formalize `MetalRawBackend._com_cache` as a
  capability hook so the widget can `if 'com_cache' in caps: use_cached`.
