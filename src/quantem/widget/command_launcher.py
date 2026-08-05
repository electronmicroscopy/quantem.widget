"""Write a double-click ``.command`` launcher into an exported viewer folder.

An exported ShowPtycho / Show4DSTEM WebGPU folder is a static site: an
``index.html`` viewer plus source payloads the browser fetches with HTTP Range
requests. Opening it needs either a File System Access grant or a Range-capable
local server. This module writes a self-contained macOS launcher so the user can
just double-click:

- ``<WidgetLabel>.command`` at the folder root: a zsh script that starts or
  reuses the bundled Range server for this exact folder and opens the viewer in
  Google Chrome (WebGPU needs a Chromium browser, not Safari). If the preferred
  port is already busy with another folder or unknown server, the launcher
  advances to the next free port; closing the Terminal window stops a new
  server.
- ``.viewer/serve_range.py``: a stdlib-only Range HTTP server (no third-party
  deps, uses the Mac's built-in ``/usr/bin/python3``).

The launcher is additive: double-clicking ``index.html`` and granting the folder
still works for users who prefer that. The generated files never contain private
paths - only the folder is served, from wherever the user placed it.
"""
from __future__ import annotations

import pathlib

# stdlib-only Range server; ships inside the exported folder so no install is
# needed on the viewing machine.
_SERVE_RANGE_PY = '''\
#!/usr/bin/env python3
"""Range-capable local HTTP server for a QuantEM WebGPU viewer folder."""
import argparse, os, posixpath, urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class RangeHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _serve(self):
        if urllib.parse.urlsplit(self.path).path == "/__quantem_viewer_root__":
            body = str(Path.cwd().resolve()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            return
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            path = path / "index.html"
        if not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        rng = self.headers.get("Range")
        if not rng:
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(path, "rb") as handle:
                self.wfile.write(handle.read())
            return
        start_text, _, end_text = rng.replace("bytes=", "").partition("-")
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
        end = min(end, size - 1)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(1 << 16, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        self._serve()

    def do_HEAD(self):
        self._serve()

    def _snapshot_write_path(self, *, allow_json):
        raw = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
        rel = posixpath.normpath(raw).lstrip("/")
        parts = Path(rel).parts
        if len(parts) == 2 and parts[0] == "snapshots":
            pass
        elif len(parts) == 3 and parts[1] == "snapshots":
            pass
        else:
            self.send_error(403, "writes are restricted to dataset snapshots")
            return None
        name = parts[-1]
        if allow_json and name == "snapshots.json":
            pass
        elif (
            name.startswith("snapshot_")
            and (name.endswith(".jpg") or name.endswith(".jpeg"))
        ):
            pass
        else:
            self.send_error(403, "writes are restricted to snapshots")
            return None
        root = Path.cwd().resolve()
        path = (root / rel).resolve()
        try:
            path.parent.relative_to(root)
        except ValueError:
            self.send_error(403, "invalid snapshot path")
            return None
        if path.parent.name != "snapshots":
            self.send_error(403, "invalid snapshot path")
            return None
        return path

    def do_PUT(self):
        path = self._snapshot_write_path(allow_json=True)
        if path is None:
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(body)
        self.send_response(204)
        self.end_headers()

    def do_DELETE(self):
        path = self._snapshot_write_path(allow_json=False)
        if path is None:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args()
    os.chdir(args.root)
    ThreadingHTTPServer((args.host, args.port), RangeHandler).serve_forever()
'''


def _command_script(widget_label: str, port: int, viewer_html: str) -> str:
    """zsh launcher: reuse or start the Range server, open the viewer in Chrome.

    Forces Google Chrome because WebGPU (the whole point of the viewer) does not
    run in Safari; falls back to the default browser only if Chrome is absent.
    Keeps the server tied to the Terminal window so closing it stops the server.
    """
    return f'''#!/bin/zsh
# Double-click to open the {widget_label} viewer in Chrome. No install needed.
set -u
DIR="${{0:A:h}}"
PORT="${{QUANTEM_HANDOFF_PORT:-{port}}}"
cd "$DIR"
while lsof -nP -iTCP:${{PORT}} -sTCP:LISTEN >/dev/null 2>&1; do
  EXISTING_ROOT="$(/usr/bin/python3 - "$PORT" <<'PY'
import sys
import urllib.request

url = "http://127.0.0.1:" + sys.argv[1] + "/__quantem_viewer_root__"
try:
    print(urllib.request.urlopen(url, timeout=0.5).read().decode("utf-8").strip())
except Exception:
    pass
PY
)"
  if [[ "$EXISTING_ROOT" == "$DIR" ]]; then
    URL="http://127.0.0.1:${{PORT}}/{viewer_html}"
    echo "Reusing local {widget_label} viewer server for this folder on port ${{PORT}}"
    open -a "Google Chrome" "$URL" 2>/dev/null || open "$URL"
    echo "$URL"
    exit 0
  fi
  PORT=$((PORT + 1))
done
URL="http://127.0.0.1:${{PORT}}/{viewer_html}"
echo "Starting local {widget_label} viewer. Keep this Terminal window open while viewing."
/usr/bin/python3 "$DIR/.viewer/serve_range.py" --root "$DIR" --host 127.0.0.1 --port "$PORT" &
PID=$!
trap 'kill $PID 2>/dev/null || true' INT TERM EXIT
for _ in {{1..40}}; do
  READY_ROOT="$(/usr/bin/python3 - "$PORT" <<'PY'
import sys
import urllib.request

url = "http://127.0.0.1:" + sys.argv[1] + "/__quantem_viewer_root__"
try:
    print(urllib.request.urlopen(url, timeout=0.1).read().decode("utf-8").strip())
except Exception:
    pass
PY
)"
  [[ "$READY_ROOT" == "$DIR" ]] && break
  sleep 0.025
done
open -a "Google Chrome" "$URL" 2>/dev/null || open "$URL"
echo "$URL"
wait "$PID"
'''


def write_command_launcher(
    folder: str | pathlib.Path,
    widget_label: str = "ShowPtycho",
    *,
    viewer_html: str = "index.html",
    port: int = 8791,
) -> pathlib.Path:
    """Write ``<widget_label>.command`` + ``.viewer/serve_range.py`` into ``folder``.

    ``viewer_html`` is the page the launcher opens (``index.html`` for a
    ShowPtycho folder export; the specific ``<name>.html`` for a Show4DSTEM
    export that writes a named file alongside its data). Returns the path to the
    ``.command`` file. Both files are marked executable so a double-click works
    on macOS. Safe to call repeatedly (overwrites).
    """
    root = pathlib.Path(folder).expanduser()
    viewer_dir = root / ".viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    server = viewer_dir / "serve_range.py"
    server.write_text(_SERVE_RANGE_PY, encoding="utf-8")
    server.chmod(0o755)
    # macOS only auto-runs a *.command file on double-click, so the extension is
    # fixed; the name carries the widget so a folder holding more than one viewer
    # stays unambiguous (e.g. ShowPtycho.command + Show4DSTEM.command).
    command = root / f"{widget_label}.command"
    command.write_text(
        _command_script(widget_label, port, viewer_html), encoding="utf-8"
    )
    command.chmod(0o755)
    return command
