# Run on a GPU box from your laptop browser

The way to use quantem widgets (Show2D / Show3D / Show4DSTEM, SSB, any tutorial) on real
data is: **the kernel + GPU run on the compute box, and you drive everything from
JupyterLab in your laptop browser.** anywidget talks to the kernel over the Jupyter Comm
channel, so one server serves every widget with nothing to configure per widget.

## Launch

Run this **on the GPU box** (over SSH, or in its terminal):

```bash
quantem jupyter drift_tutorial.ipynb
```

It starts JupyterLab in the `live-env` conda env (the default) and prints a URL:

```
  http://localhost:8901/lab/tree/drift_tutorial.ipynb?token=...
```

Copy that URL into your laptop browser. `Ctrl-C` stops the server. conda is auto-sourced,
so `conda` need not be on the box's login PATH.

- `path` - a notebook to open directly (omit to land in the file browser), or a directory.
- `--env <name>` - conda env on the box (default `live-env`; `--env ''` skips activation).
- `--port <n>` - serve on a fixed port (default: auto-pick a free one).
- `--no-open` - don't try to open a local browser (the usual case on a headless box).

## Reaching it from your laptop

If your laptop can already open `http://localhost:<port>` on the box (you SSH'd in with a
forwarded port, or use VS Code Remote-SSH which forwards automatically), just paste the
URL. Otherwise open the tunnel the command printed, then paste the URL:

```bash
ssh -L 8901:127.0.0.1:8901 you@your-box.stanford.edu
```

The `you@your-box` part is filled in for you. On the first launch the command auto-detects
`whoami@<fqdn>` and asks you to confirm or correct it, then saves it to
`~/.config/quantem/jupyter.json`; every later launch prints a ready-to-paste line. Edit
that file anytime to change it. This is the same bring-your-own-tunnel model as
`quantem.live` - the command itself does no SSH or tunnel work.

---

## Why this, and not a downloaded HTML?

- **`quantem jupyter`** = live kernel on the GPU box. Full interaction, full data, every
  widget. The real working surface. Cmd+S saves the notebook outputs and
  `metadata.widgets`, so a compatible JupyterLab can reopen the widget from the saved
  file without rerunning cells.
- **`quantem html notebook.ipynb --no-execute`** = a standalone, self-contained HTML page
  using the ipywidgets HTML manager. It hydrates the notebook's saved widget state and
  keeps Show2D / Show3D / Show3DSlices controls interactive in the browser with no
  kernel. Interactions in the HTML page are browser-local: changing contrast, zoom,
  frame, or toggles does not write back to the `.ipynb` or the `.html` file.
- **`quantem github notebook_github.ipynb --no-execute`** = a GitHub preview notebook.
  It strips the heavy widget state and widget MIME refs, then inserts JPEG snapshots of
  the widget UIs. Use this for GitHub's native `.ipynb` renderer, which does not run
  widget JavaScript.

Use `jupyter` to *work*, `html` to share an interactive web artifact, and `github` to
share a lightweight notebook preview on GitHub.

```{warning}
GitHub does not execute widget JavaScript in notebook previews, and GitHub blob/raw URLs
do not serve exported HTML as a runnable web page. Put `quantem html` output on GitHub
Pages or another static web host when you want the interactive HTML to run.
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| laptop browser "can't connect" | open the tunnel above (or use VS Code Remote-SSH), then paste the URL |
| `EnvironmentNameNotFound` on launch | the box's env isn't `live-env`; pass the real name with `--env <name>` |
| port clash | rerun (it auto-picks a fresh port) or pass `--port <n>` |
| widget renders but feels laggy | expected for huge frames over a slow link; only pixels cross the wire |
| reopening a saved notebook shows `Error displaying widget: model not found` or `Failed to load model class 'AnyModel'` | upgrade the JupyterLab environment to `anywidget>=0.11.0` and `jupyterlab_widgets>=3.0.10`; `quantem jupyter` enables widget-state saving for Cmd+S, but manually launched Lab still needs **Save Widget State Automatically** enabled |
