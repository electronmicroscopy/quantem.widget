# Show4DSTEM automation notebooks

One notebook per test case, explicit names, explicit inline data paths. Same API
on both boxes:

```python
from quantem.widget import load, Show4DSTEM
Show4DSTEM(load(master, det_bin=2))            # single
Show4DSTEM(load([m0, m1, ...], det_bin=4))     # many (Dataset slider)
```

`load` auto-detects the backend (CUDA on a Linux GPU box, MPS on a MacBook) and
`Show4DSTEM` dispatches to the right viewer. The cell code is identical across
boxes; only the data path differs.

For autonomous four-path migration signoff, use
`docs/refactor/2026-06-06-show4dstem-agent-signoff-runbook.md` before opening
these notebooks. It covers CUDA, Phil MPS, WebGPU live/browser compute, exported
WebGPU HTML + bslz4 companion chunks, and required cleanup of temporary
JupyterLab, Chrome/Playwright, HTTP servers, and export directories.

## `linux/` - Linux CUDA (host), data `/data/sample/series/`

| Notebook | Case |
|---|---|
| `test_nobin_single.ipynb` | no bin, one dataset - full detector in VRAM, VI direct |
| `test_bin2_single.ipynb` | bin 2, one dataset |
| `test_bin4_5datasets.ipynb` | bin 4, five datasets - eager 5D, instant Dataset slider |
| `test_nobin_3datasets.ipynb` | no bin, three datasets - full-res 5D on one card (`devices=[0,1]` for more) |

## `phil/` - MacBook MPS (phil), data `/Users/macbook/data/sample/`

| Notebook | Case |
|---|---|
| `test_nobin_single.ipynb` | no bin, one dataset - full-res CBED + **bin2 VI sidecar** (real-time BF/DF) |
| `test_bin2_single.ipynb` | bin 2, one dataset - no sidecar (data already small) |
| `test_bin4_5datasets.ipynb` | bin 4, five datasets - **lazy**: ds0 now, 1..4 fill in background |

## Run

Launch JupyterLab on the box, open the notebook, Run All, and interact with the
viewer (drag the detector ROI, scrub the scan cursor, move the Dataset slider).
The viewer must render and respond live - the no-bin MPS virtual image stays
real-time once the bin2 sidecar finishes building.
