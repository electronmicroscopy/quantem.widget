# Run on a GPU box from your laptop browser

The way to use quantem widgets (Show2D / Show3D / Show4DSTEM, SSB, any tutorial)
on real data is: **the kernel + GPU run on a compute box, and you drive everything
from JupyterLab in your laptop browser.** One SSH tunnel makes every widget work -
anywidget talks to the kernel over the Jupyter Comm channel, so there is nothing to
configure per widget.

`quantem jupyter` automates the whole thing:

```bash
quantem jupyter --host buffle --env quantem-env drift_tutorial.ipynb
```

That starts JupyterLab on `buffle`, opens an SSH tunnel, and pops the notebook up in
your local browser. `Ctrl-C` stops the server and closes the tunnel.

> Setup is the part that wastes people's time. Do the four steps below **once** and
> every future session is a single command.

---

## 0. What you need from whoever runs the box

Ask the box admin (the person sharing buffle / mallard) for:

- **An account** on the box, and which host: `buffle.stanford.edu` or `mallard.stanford.edu`.
- **Your username** on that box.
- **How auth works**: your SSH public key added to the box (preferred), and whether
  you must hop through a **login/bastion node** or use **2FA**. Stanford HPC often does.
- **The conda env name** that has `quantem` + `quantem.widget` installed (e.g. `quantem-env`),
  or permission to create your own.

Don't guess these - one wrong hostname or a missing key is the #1 time sink.

---

## 1. SSH config on your laptop (one time)

Add an entry to `~/.ssh/config` so you can type `--host buffle` instead of a long string.
Replace `YOUR_USERNAME` with the username from step 0:

```sshconfig
Host buffle
    HostName buffle.stanford.edu
    User YOUR_USERNAME
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30

Host mallard
    HostName mallard.stanford.edu
    User YOUR_USERNAME
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
```

If the box is only reachable through a login node, add `ProxyJump LOGINNODE` (and define
that `Host` too). The admin tells you whether this is needed.

No SSH key yet? Make one and send the **public** half to the admin:

```bash
ssh-keygen -t ed25519           # press enter through the prompts
cat ~/.ssh/id_ed25519.pub       # give this line to the admin
```

## 2. Prove SSH works before anything else

```bash
ssh buffle echo ok
```

You must see `ok` with **no password prompt** (key auth) and no errors. If this fails,
`quantem jupyter` cannot work - fix SSH first (wrong host, key not installed, or a
login-node jump is needed). This 10-second check saves an hour of confusion.

## 3. quantem on both ends

- **On the box** (where the kernel runs): a conda env with `quantem` + `quantem.widget`.
  A tutorial notebook's first cell usually `pip install`s these for you; otherwise:
  ```bash
  ssh buffle
  conda activate quantem-env
  pip install quantem.widget
  ```
- **On your laptop** (where the `quantem jupyter` command runs): just the launcher.
  ```bash
  pip install quantem.widget
  ```

## 4. Launch

```bash
quantem jupyter --host buffle --env quantem-env drift_tutorial.ipynb
```

- `--host` - the SSH alias from step 1 (`buffle` or `mallard`).
- `--env` - the conda env on the box (omit if quantem is in the box's base env).
- last argument - a notebook to open directly (omit to land in JupyterLab's file browser).

The command prints a `http://localhost:<port>/lab?token=...` URL and opens it. If your
browser didn't pop up (headless laptop, remote desktop), paste that URL yourself.
Leave the terminal running - it holds the tunnel. `Ctrl-C` ends the session.

---

## Why this, and not a downloaded HTML?

- **`quantem jupyter`** = live kernel on the GPU box. Full interaction, full data, every
  widget. This is the real working surface.
- **`quantem html notebook.ipynb`** = a static, self-contained HTML snapshot (widgets
  baked as images). No kernel, no GPU, opens anywhere - good for *sharing a result* or
  for someone with no box access. It cannot recompute.

Use `jupyter` to *work*; use `html` to *share*.

## Troubleshooting (the usual time-wasters)

| Symptom | Cause / fix |
|---|---|
| password prompt on `ssh buffle` | key not installed on the box - send your `.pub` to the admin |
| `Permission denied (publickey)` | wrong `User` or `IdentityFile` in `~/.ssh/config` |
| hangs, then times out | box needs a login-node `ProxyJump`, or you're off the campus VPN |
| `conda: command not found` on launch | the env isn't on the box's login-shell PATH; ask the admin for the right `--env` or activation |
| browser opens but "can't connect" | rare port clash - rerun (it auto-picks a fresh remote port), or pass `--port` |
| widget renders but feels laggy | expected for huge frames over a slow link; the compute is remote, only pixels cross the wire |
