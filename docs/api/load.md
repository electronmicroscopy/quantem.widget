# Load detector acquisitions

Reads a 4D-STEM acquisition onto CUDA or Apple Metal and returns an encoded
`quantem.gpu.io.Dataset4dstemGPU`, accepted directly by
[`Show4DSTEM`](show4dstem). Public import:

```python
from quantem.gpu.io import load
```

`quantem.widget.read_4dstem(path)` calls this function when quantem.gpu is
installed and a GPU is present, and reads the file into a quantem core
`Dataset4dstem` elsewhere; use it when the same notebook must run on machines
with and without a GPU (see [Image and acquisition I/O](io.md)).

The [QuantEM.GPU I/O guide](https://github.com/bobleesj/quantem.gpu/blob/main/docs/api/io.md) is the authoritative description of
supported sources, exactness, metadata, storage, device selection, and saving.
For image files rather than scanned detector acquisitions, see [Image and
acquisition I/O](io.md).

## Function reference

```{eval-rst}
.. autofunction:: quantem.gpu.io.load
```

```{tip}
`load` keeps native detector sampling and the source count dtype (uint32 counts
that fit in uint16 are stored as uint16). It has no detector-binning or
dtype-cast option: the acquisition stays ANS encoded on the
GPU, so a full-resolution scan already fits. Pass a list of master paths to get
one acquisition per file.
```

## Backend (CUDA / Apple Silicon)

`load` detects the native GPU automatically: an NVIDIA box loads onto **CUDA**
and a Mac loads onto **Apple Metal (MPS)**. Pass `backend="cuda"` or
`backend="mps"` to require one, and `device=` to choose a CUDA device.
Scientific loading does not silently fall back to CPU.

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

loaded = load("scan_master.h5")   # CUDA on a workstation, MPS on a MacBook
Show4DSTEM(loaded)
```

The returned acquisition is encoded, not a dense array. A 512 x 512 x 192 x 192
uint16 Arina scan, 18 GiB as a dense array, occupies about 0.1 to 2 GiB of GPU
memory depending on its counts:

```python
print(loaded.shape, loaded.dtype)      # (512, 512, 192, 192) uint16
print(loaded.logical_bytes / 2**30)    # 18.0, the dense size
print(loaded.resident_bytes / 2**30)   # encoded size on the GPU
```

The same encoded storage serves a MacBook and a workstation; there is no
separate preview or memory-reduction mode.

The same is one shell command - see [the CLI](../cli):
`quantem show4dstem scan_master.h5`.

## Several files

Pass a list of masters to load each one as its own acquisition:

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

masters = [
    "/data/session/file_001_master.h5",
    "/data/session/file_002_master.h5",
    "/data/session/file_003_master.h5",
]

acquisitions = load(masters)   # a list of Dataset4dstemGPU, one per master
viewer = Show4DSTEM(acquisitions)
```

The acquisitions are not stacked into one 5D array. `Show4DSTEM` opens them as
a comparison grid with one shared detector ROI, labels each panel with its
source file name, and requires one scan and detector shape across the list.
For a folder that is still being written, use
[`Show4DSTEM.from_folder(...)`](show4dstem.md#live-scope-folders), which opens
after the first master and appends the rest.

## Bounded reads for ROI workflows

Reconstruction, denoise, and ROI workflows read a rectangular scan region from
the loaded acquisition. The read decodes only that region into a Torch tensor
on the acquisition's GPU:

```python
from quantem.gpu.io import load

with load("scan_master.h5") as loaded:
    patch_t = loaded.read(scan_region=(160, 293, 234, 367))  # row_start, row_stop, col_start, col_stop

print(patch_t.shape)
# torch.Size([133, 133, 192, 192])
```

Stops are exclusive. `detector_region=(row_start, row_stop, col_start,
col_stop)` restricts the detector pixels the same way, and basic indexing such
as `loaded[100:164, 100:164]` returns the same kind of tensor. A complete read
is allowed only when its dense tensor fits the accelerator's working memory, so
read bounded regions instead of the whole scan.

For drift-corrected time-series work, compute the source scan box from the
shared specimen ROI plus a small halo, read that patch, then apply your
existing subpixel sampler in local patch coordinates. Do not save the sampled
patch as a new raw acquisition; drift is scan-position metadata, and detector
counts stay physically unchanged.

## Detector products

Virtual-detector images and the mean diffraction pattern are computed on the
encoded storage without a dense copy:

```python
from quantem.gpu import detector

mean_dp = detector.mean(loaded)
center, radius = detector.fit_probe(mean_dp)
bright_field = detector.bf(loaded, center=center, radius=radius)

session = detector.prepare(loaded)   # repeated queries; a list of acquisitions also works
```

## Releasing GPU memory

Close an acquisition when the last consumer is done with it, or load it in a
`with` block:

```python
loaded = load("scan_master.h5")
...
loaded.close()

with load("scan_master.h5") as loaded:
    ...
```

A viewer borrows the acquisition it shows, so close the viewer first.
