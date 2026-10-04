# Load detector acquisitions

Use `quantem.gpu.io.load` to keep the acquisition and its metadata together.
It returns `quantem.gpu.io.Dataset4dstemGPU`, accepted directly by `Show4DSTEM`:

```python
from quantem.gpu import io
from quantem.widget import Show4DSTEM

data = io.load("scan_master.h5")
Show4DSTEM(data)
```

Keep `data` open while the viewer uses it. Close it with `data.close()` after
its last use. Loading supported originals keeps exact counts ANS-encoded on
the selected CUDA or MPS device; bounded indexing decodes only the requested
working selection. There is no automatic CPU decoding fallback.

```python
pattern = data[10, 12]
crop = data[8:12, 10:16, 64:128, 64:128]
data.shape, data.dtype, data.metadata
```

For multiple acquisitions, keep separate owners rather than building a dense
5D array:

```python
series = io.load(paths, stack=False)
Show4DSTEM(series[1])
```

The [QuantEM.GPU I/O guide](https://github.com/bobleesj/quantem.gpu/blob/main/docs/api/io.md) is the authoritative description of
supported sources, exactness, metadata, storage, device selection, and saving.
For image files rather than scanned detector acquisitions, see [Image and
acquisition I/O](io.md).

## Function reference

```{eval-rst}
.. autofunction:: quantem.gpu.io.load
```
