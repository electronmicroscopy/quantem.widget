# ScanControl

Connect to a detector and scan controller, set the detector up, run scans and manage the
detector's files from one panel. The widget starts the steps of an acquisition runner and shows
its status; every hardware call stays in the runner, so the widget holds no addresses.

```python
import scan_runner                      # the site's runner: DCU + scan-controller steps
from quantem.widget import ScanControl

runner = scan_runner.Runner(detector_address, scan_controller_address)
panel = ScanControl(runner, size=512, binned=True, folder="downloads")
panel
```

**Go moves the electron beam.** Press it only when the operator says go, with the beam on.

| Control | What it does |
|---|---|
| Size, Binning, Output, Save HDF5 | The scan settings; a change while Ready sets the detector up again |
| Connect | Connects and sets the detector up for the settings |
| Scans, Run forever | Scans per Go |
| Go | Runs the scans (`runner.start("run", settings, n)`) |
| Stop | Ends the run after the scan in progress |
| Files | Refresh, download to the folder, delete (asks first) |

The same steps work from Python: `panel.connect()`, `panel.go()`, `panel.stop()`,
`panel.download(names)`, `panel.delete(names)`. `panel.close()` stops polling and disconnects
the scan controller.

A runner provides `start(step, *args)` (steps `connect`, `configure`, `run`, `list_files`,
`download`, `delete`; returns False while a step is running), `status()` (a namedtuple with
`state`, `message`, `dcu_ok`, `revolon_ok`, `high_voltage`, `done`, `total`, `dropped`, `files`,
`free_bytes`), `stop()` and `close()`.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.scan_control.ScanControl
   :members:
   :show-inheritance:
```
