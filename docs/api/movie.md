# Movie Export

`quantem.widget.movie` provides the package-level movie writers used by widget
methods and downstream tools such as Denova.
For a workflow-oriented guide, see [Saving GIF and MP4 movies](../tutorials/movie_export).

```python
from quantem.widget import movie

movie.save_mp4(stack, "movie.mp4", fps=12)
movie.save_gif(stack, "movie.gif", fps=12)
```

Input can be one stack with shape `(frame, row, col)`, several stacks with shape
`(movie, frame, row, col)`, or a list of stacks. Several stacks are arranged in
a grid in one output file.

```python
movie.save_mp4(
    [raw, denoised],
    "comparison.mp4",
    labels=["Raw", "Denoised"],
    fps=12,
    cols=2,
)
```

`Show3D.save_gif(...)` and `Show3D.save_mp4(...)` keep their current behavior
but now route through this module after rendering the widget view.

MP4 export accepts `backend="auto"`, `backend="cuda"`, or `backend="cpu"`.
`auto` uses the NVIDIA CUDA MP4 path when available and otherwise uses the
portable CPU writer. The CUDA path uses NVENC internally for compression. GIF
export is CPU-only.

## MP4 Backend Benchmark

Measured on a workstation with an NVIDIA RTX PRO 6000 Blackwell GPU, using the
`800C_1.3Mx_1` in situ DriftCorrected time series. The source stack was 58
frames at `2048 x 2048`; the benchmark exported a two-panel raw/denoised
comparison from the center `512 x 512` crop at 12 fps, producing a `1024 x 534`
H.264 MP4.

| Path | Backend | MP4 export time | Output size | Speedup |
| --- | --- | ---: | ---: | ---: |
| Portable writer | `cpu` | 33.131 s | 11.3 MB | 1.0x |
| CUDA MP4 writer | `cuda` | 1.834 s | 9.7 MB | 18.1x |

For the same run, loading the full PNG stack took 10.576 s and Denova TV
denoising on the review crop took 6.448 s. The CUDA MP4 backend reduced movie
compression from the dominant cost to a small part of the end-to-end workflow.
