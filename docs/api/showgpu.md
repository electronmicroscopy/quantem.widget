# ShowGPU

Monitor NVIDIA VRAM, utilization, temperature and power alongside system RAM
while a notebook runs. The readings describe the **notebook server**, which
may be a different machine from the browser.

```python
from quantem.widget import ShowGPU

monitor = ShowGPU(
    users=True,      # Show compute users and their VRAM.
    expand=True,     # Show each user's process list immediately.
    size="medium",   # "small", "medium", "large", "xlarge"
    interval=1.0,    # Seconds between samples; 0.2 to 60.
    running=True,    # False pauses automatic sampling.
)
monitor
```

All arguments are optional: `ShowGPU()` uses these defaults.

## Scan inventory during acquisition

Attach the existing monitor to a started recording or acquisition controller.
This starts no detector or receiver and does not copy diffraction data:

```python
from quantem.widget import Show4DSTEM

monitor.watch(recording)  # Live DPs, scan shape, encoded bytes and staging bytes.
viewer = Show4DSTEM.stream(recording)
monitor.watch(recording, viewer=viewer)  # Click a scan name to inspect its DP/VI.
```

For a controller with processing and saving queues, watch that controller
instead of its underlying recording. Its status distinguishes receiving,
queued, processing, saving and ready scans. Supply the trigger's stop callback
when available:

```python
monitor.watch(run, viewer=viewer, stop=replay.close)
```

Each card identifies lossless **ANS** storage, the count dtype, received and
encoded DP counts, and memory for ANS data/metadata, interaction indexes and
the unencoded staging tail. Progress reflects completed receive/encode batches;
sampling does not interrupt GPU work to inspect an in-flight kernel. Source
filenames appear when the acquisition provides `source_path` metadata; otherwise
the card says **Live detector**. A detector stream need not have a source file.

**Unload** discards the selected scan's unsaved counts after confirmation.
Busy scans cannot be unloaded. Saved files remain intact, and the linked
streaming viewer updates to the remaining scans. Finish other analyses that
borrow the scan before unloading it. Live reconstruction series require the
run and pending saves to stop before retained inputs can be unloaded.
**Stop** (square icon) stops the supplied trigger first, then closes the recording;
retained scans stay available. `monitor.watch(None)` detaches without stopping
or releasing the acquisition.

If another live viewer still holds a discarded scan, **Waiting for viewers**
keeps its buffers visible until that viewer closes. **Recent activity** retains
the last 32 observed state changes and action outcomes, including automatic
removal from the retained window. Events occur at the sampling interval; brief
intermediate states can occur between samples. **Stopping** remains visible
until the receiver and processing finish. This does not delete saved files.

The scan chart shows the last 120 samples of tracked buffers. Encoded bytes
include interaction indexes; staging is listed separately. Shared codec tables,
receiver queues, allocator reserves and separate analysis sessions are excluded
from that sum, but remain included in the driver VRAM totals above. Consequently,
unloading a scan may not immediately lower driver VRAM by its buffer size.

The driver can identify processes on the notebook server, but cannot enumerate
the scientific datasets inside an arbitrary process. The scan inventory covers
the owner explicitly attached in this notebook.

## Process controls

**Terminate** requests SIGTERM after confirming the displayed PID and name.
It ends all work in that process and may discard unsaved data. Only processes
owned by the notebook's operating-system user are eligible; the notebook kernel
and its ancestors are protected. The PID's creation time is rechecked before
signalling to reject a stale selection. No process is stopped by sampling.
**Termination requested** remains pending until operating-system checks confirm
exit; disappearance from the GPU list alone does not establish process exit.
The monitor never escalates a termination request to a forced kill.

## Python controls

Change the existing monitor without creating another sampler:

```python
monitor.size = "large"  # Resize text and controls.
monitor.users = False   # Hide users and skip process queries.
monitor.interval = 2.0  # Sample less often.
monitor.running = False  # Pause; True resumes.
snapshot = monitor.refresh()  # Read once, including while paused.
snapshot = monitor.snapshot   # Read the latest sample without a new query.
monitor.close()         # Stop sampling and close the display when finished.
```

Place this cell above acquisition or reconstruction. The sampler runs in a
background thread and updates once per second. It reads the NVIDIA driver
through `quantem.gpu.device.status()` without creating a CUDA context,
allocating VRAM, synchronizing work, or releasing caches. GPU rows include
every process using that physical device. Driver indices are not CUDA logical
indices; any `CUDA_VISIBLE_DEVICES` mask is shown separately.

`users=True` adds each compute user's reported VRAM and process count.
`expand=True` shows the process lists initially; use `expand=False` to start
with just user totals. Click a user row or focus it and press Enter to toggle
the process list below it: process ID, executable name and reported VRAM.
Expanded or collapsed rows keep their state as
samples update. Only executable names are read, not command-line arguments.
`users=False` hides that breakdown and skips the process query. User totals
exclude graphics processes and driver allocations, so they need not sum to
device usage. Inaccessible process ownership is labeled **Unknown user**;
unsupported memory readings remain **unavailable**.

`size` accepts `"small"`, `"medium"`, `"large"` and `"xlarge"`. For this
text-based monitor, these select 13, 15, 17 and 20 CSS-pixel text respectively,
with matching controls and responsive cards. Choose **S / M / L / XL** or
toggle **Users** directly in the widget without rerunning the cell. The same
settings are mutable Python attributes: `monitor.size = "large"` and
`monitor.users = False`.

`ShowGPU(interval=2.0)` reduces sampling frequency. Pause/Resume controls
automatic updates; Refresh requests one sample even while paused. Browser
commands may wait for a busy kernel. A native call holding Python's GIL can
also delay samples; **No recent update** labels stale readings. Saved output
does not count as live until new samples arrive.

When finished, use `monitor.close()`. Before rerunning the creation cell, close
the previous monitor to stop its sampler. System RAM remains visible without
NVIDIA telemetry. Apple unified GPU memory is not reported as dedicated VRAM.
Use matching development versions of quantem.gpu and quantem.widget when
trying this API before release.

Memory is shown in GiB (2³⁰ bytes). NVIDIA's driver reserve can make reported
used plus free smaller than total. RAM used means total minus available;
available includes reclaimable memory. The kernel's resident host RAM is
shown separately and is not its VRAM allocation. Recent VRAM traces retain
at most 120 samples per device in the browser.

## Interactive controls

| Control | Effect | Expected observation |
| --- | --- | --- |
| Pause / Resume | `running` | Sampling pauses or resumes |
| Refresh | one sample | Sample counter and timestamp advance |
| Users | `users` | Compute-user VRAM breakdown shows or hides |
| User row | expand / collapse | Process IDs, names and VRAM appear underneath |
| S / M / L / XL | `size` | Text, controls and card layout resize |
| Scan name | select linked viewer | DP/VI switch to that retained scan |
| Unload | discard ready scan after confirmation | Scan disappears; linked viewer updates |
| Stop (square icon) | stop trigger, then recording | Retained scans remain available |
| Terminate | request SIGTERM for an eligible PID | Process exits if it accepts the signal |

```{eval-rst}
.. autoclass:: quantem.widget.showgpu.ShowGPU
   :members: watch, refresh, close
```
