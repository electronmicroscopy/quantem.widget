# Saving GIF and MP4 Movies

Use `quantem.widget.movie` when you already have one or more image stacks and
want a GIF or MP4 without opening a widget first. This is useful for denoising
comparisons, time-series previews, reconstruction iterations, and report-ready
movies.

```python
from quantem.widget import movie

movie.save_mp4(stack, "movie.mp4", fps=12)
movie.save_gif(stack, "movie.gif", fps=12)
```

The input stack shape is `(frame, row, col)`. Several stacks can be saved into
one comparison movie by passing a list. All stacks must have the same frame
count and spatial shape.

```python
movie.save_mp4(
    [raw, denoised],
    "raw_vs_denoised.mp4",
    labels=["Raw", "Denoised"],
    fps=12,
    cols=2,
)
```

## Choose a Backend

MP4 export accepts three backend names:

| Backend | Use it when |
| --- | --- |
| `auto` | You want the fastest available path. This is the default. |
| `cuda` | You are on an NVIDIA workstation and want CUDA/NVENC MP4 compression. |
| `cpu` | You need the portable writer, or you are not in the CUDA environment. |

`backend="auto"` selects the CUDA MP4 path only when CuPy, PyNvVideoCodec, and
an NVIDIA CUDA device are available in the current Python process. Otherwise it
uses the portable CPU writer.

```python
movie.save_mp4(
    [raw, denoised],
    "raw_vs_denoised_cuda.mp4",
    labels=["Raw", "Denoised"],
    fps=12,
    cols=2,
    backend="cuda",
)
```

GIF export is CPU-only:

```python
movie.save_gif(
    [raw, denoised],
    "raw_vs_denoised.gif",
    labels=["Raw", "Denoised"],
    fps=8,
    cols=2,
)
```

## Keep Contrast Comparable

By default, all panels share one percentile window, so raw and processed views
are visually comparable.

```python
movie.save_mp4(
    [raw, denoised],
    "comparison.mp4",
    labels=["Raw", "Denoised"],
    percentile=(1, 99),
    shared_contrast=True,
)
```

If raw frames contain edge artifacts or unusually bright acquisition glitches,
compute the shared contrast from the processed stack while still showing both
panels:

```python
movie.save_mp4(
    [raw, denoised],
    "comparison.mp4",
    labels=["Raw", "Denoised"],
    ref_stacks=[denoised],
)
```

## Save From a Widget

`Show3D.save_mp4(...)` and `Show3D.save_gif(...)` route through the same package
movie writer after rendering the current widget view.

```python
from quantem.widget import Show3D

w = Show3D(stack, fps=12)
w.save_mp4("show3d-view.mp4")
w.save_gif("show3d-view.gif")
```

Use the package-level `movie.save_mp4(...)` API when you want array-first
control over panels, labels, and backend selection. Use the widget methods when
you want the saved movie to match the current widget-rendered view.

## Performance Reference

On an NVIDIA RTX PRO 6000 Blackwell workstation, a two-panel MP4 export from an
in situ DriftCorrected time series measured:

| Path | Backend | MP4 export time | Output size | Speedup |
| --- | --- | ---: | ---: | ---: |
| Portable writer | `cpu` | 33.131 s | 11.3 MB | 1.0x |
| CUDA MP4 writer | `cuda` | 1.834 s | 9.7 MB | 18.1x |

The benchmark used 58 frames from the `800C_1.3Mx_1` time series, exported as a
two-panel raw/denoised comparison from a center `512 x 512` crop at 12 fps. Use
this table as a hardware-specific reference point, not a universal guarantee:
load time, denoising time, frame shape, codec settings, and storage all affect
end-to-end runtime.
