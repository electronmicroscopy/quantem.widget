# Native QEM detector-drag gate

Use a conda environment with the current `quantem.gpu`, this widget checkout,
CUDA, headed Chrome, and at least ten independent **512 × 512 × 192 × 192**
QEM recordings. The test loads seven, then ten, with no binning or conversion.
It reuses the existing page/Jupyter fixtures and closes its acquisitions. A stress-local headed Chrome fixture exposes a loopback-only native CDP socket; it never controls an existing browser.

```bash
npm run build
export QUANTEM_WIDGET_STRESS_QEM_DIR=/path/to/recordings
export QUANTEM_WIDGET_BROWSER_OUT=/path/to/private/evidence
QUANTEM_WIDGET_BROWSER=1 PYTHONPATH=src python -m pytest \
  tests/browser/stress/test_qem_drag.py -q -s
```

Choose the CUDA device with `QUANTEM_WIDGET_STRESS_GPU` (default `0`). Set
`DISPLAY` for the local headed desktop when the shell does not inherit it.
Without the opt-in data directory and browser flag, the test skips.

The default target is **120 new complete scientific images/s**. A miss fails
and saves the measured result; it is never silently reduced to the monitor's
refresh rate. `QUANTEM_WIDGET_STRESS_TARGET_HZ` permits an explicit different
target for another display. Idle display rAF and rAF during dragging are
reported separately and are not counted as new scientific images.

The observer counts a native image-buffer revision once, after every comparison
canvas submits that revision and its GPU work finishes by a following rAF.
Repeated draws of the same uploaded buffer do not increase the count. This
readiness measure is not a physical display scanout measurement. The test also
requires exact kernel/browser image hashes, visible PNG changes while pressed, detector
movement, unchanged native shapes, a hardware adapter, and no GPU errors.
Real mouse positions are generated at the target cadence, independently of
individual CDP acknowledgements. At most four native mouse moves are in flight,
with one latest unsent position; older unsent positions are coalesced. This is
a real held gesture, not synthetic DOM events or a serialized Playwright loop.
The report records requested/sent/acknowledged/coalesced positions, ACK latency,
actual trusted pointer delivery, and drain time. Only the fixed five-second
input interval counts; evidence screenshots and ACK drain are excluded. Browser
input coalescing can still limit delivered input. A miss is reported honestly;
this workflow rate is not an isolated maximum GPU or network throughput.

Recordings and generated notebooks, JSON, screenshots, and profiler output are
private evidence. Keep them outside version control. Do not run alongside
another GPU benchmark when comparing performance.
