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
  widget. The real working surface.
- **`quantem html notebook.ipynb`** = a static, self-contained HTML snapshot (widgets
  baked as images). No kernel, no GPU, opens anywhere - good for *sharing a result*. It
  cannot recompute.

Use `jupyter` to *work*; use `html` to *share*.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| laptop browser "can't connect" | open the tunnel above (or use VS Code Remote-SSH), then paste the URL |
| `EnvironmentNameNotFound` on launch | the box's env isn't `live-env`; pass the real name with `--env <name>` |
| port clash | rerun (it auto-picks a fresh port) or pass `--port <n>` |
| widget renders but feels laggy | expected for huge frames over a slow link; only pixels cross the wire |
