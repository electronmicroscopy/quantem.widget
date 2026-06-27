# Command line

Installing `quantem.widget` adds a `quantem` command. Point it at a file or a
folder and it renders the right viewer - no notebook, no Python.

```bash
quantem show ./anything/                     # auto-detect content, pick the viewer
quantem show2d scan.png                       # an image            -> Show2D
quantem show3d ./frames/                       # a folder of frames -> Show3D scrub
quantem show4dstem ./masters/                  # *_master.h5        -> live Show4DSTEM
quantem show4dstem a_master.h5 b_master.h5     # several masters    -> one 5D multi-tilt viewer
quantem show4dstem ./masters/ --html           # 4D-STEM            -> shareable offline HTML
quantem html tutorial.ipynb                    # a notebook         -> standalone interactive HTML
quantem github tutorial_github.ipynb --no-execute # a notebook copy -> GitHub preview JPEG outputs
quantem jupyter nb.ipynb                        # on the GPU box     -> prints a URL for your browser
```

## Subcommands

| Command | Input | Output |
|---|---|---|
| `quantem show <path>` | anything | auto-detects and dispatches to one of the below |
| `quantem show2d <image / folder>` | one image, or a folder | a Show2D HTML (a folder becomes a gallery) |
| `quantem show3d <folder>` | a folder of same-size frames | a Show3D scrub HTML |
| `quantem show4dstem <master(s) / folder>` | one or more `*_master.h5` | a live Show4DSTEM notebook (or `--html`) |
| `quantem html <notebook.ipynb>` | a notebook you wrote | runs it, or with `--no-execute` exports saved outputs/state, into one standalone interactive HTML |
| `quantem github <notebook.ipynb>` | a GitHub copy of a notebook | strips widget state and embeds JPEG snapshots for GitHub's notebook preview |
| `quantem jupyter [nb]` | run on the GPU box | starts JupyterLab there and prints a URL to open from your laptop browser |

`quantem jupyter` is the **run-on-a-GPU-box-from-your-laptop** workflow: run it on the
compute box, kernel + GPU stay there, and you open the printed URL in your browser (bring
your own SSH tunnel or VS Code Remote-SSH, like quantem.live). Details and troubleshooting
in [Remote JupyterLab](remote-jupyter.md).

**Images** save a standalone HTML and open in your browser. **4D-STEM** opens a
live, kernel-backed notebook by default (full real-time interaction); `--html`
instead writes a **self-contained offline viewer that runs entirely on WebGPU** -
drag detectors, switch BF/ABF/ADF, pan diffraction, all with no kernel.

Several masters (a folder, or listed explicitly) stack into **one 5D viewer with a
Dataset slider** to flip between scans. `--combined --html` writes that as one
offline file (served locally, since a `file://` page can't fetch its companion).

Everything lands in `~/Downloads` and opens automatically.

For notebook sharing, keep the full-state `.ipynb` for collaborators, use
`quantem html --no-execute` for an interactive web artifact, and use `quantem github
--no-execute` on a copy for GitHub's native notebook renderer. GitHub blob/raw pages do
not execute exported HTML; serve HTML from GitHub Pages or another static host.

## Options

| Option | Effect |
|---|---|
| `--bin N` | detector mean-bin factor for 4D-STEM (default 8) |
| `--html` | 4D-STEM: write the offline-WebGPU HTML instead of a notebook |
| `--combined` | many masters -> one 5D HTML viewer (served locally) |
| `--widget {2d,3d,4dstem}` | force a widget instead of auto-detect |
| `--out PATH` | output file or directory (default `~/Downloads`) |
| `--no-open` | write the file(s) without launching a browser or Jupyter |
| `--title`, `-v/--verbose` | page title; verbose progress |

## Backends

The loader picks the backend automatically - **CUDA** on an NVIDIA box, **Apple
Metal (MPS)** on a Mac, **CPU** otherwise. No flag needed. On a MacBook:

```bash
quantem show4dstem ./masters/ --html --bin 8
```

loads on Metal, mean-bins the detector to fit the laptop, and writes a
double-clickable HTML in seconds. The detector is **mean-binned** (not summed) so
the bright field never clips at uint8, and binning happens at load so the full
multi-gigabyte stack never has to fit in memory. See [`load`](api/load) for the
backend + binning details.
