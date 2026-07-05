#!/usr/bin/env python3
"""Serve widget reports for physical phone testing.

This is intentionally a handoff helper, not a physical-device automation claim.
It serves an existing artifact directory on ``0.0.0.0``, prints local and
Tailscale URLs, and writes a phone probe page that records viewport, scroll,
pointer, touch, and click events back to the server log.
"""

from __future__ import annotations

import argparse
import http.server
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote


PHONE_PROBE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>quantem.widget phone probe</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 20px; line-height: 1.45; color: #18202a; }
    code, pre { background: #f5f7f9; border-radius: 4px; }
    code { padding: 2px 4px; }
    pre { padding: 12px; overflow: auto; white-space: pre-wrap; }
    .box { border: 1px solid #ccd3db; padding: 16px; min-height: 220px; touch-action: none; }
  </style>
</head>
<body>
  <h1>quantem.widget phone probe</h1>
  <p>Open this page on the physical phone from the same origin as the widget.
  Drag, pinch, tap, rotate, and scroll inside the box. The server records what
  Safari sends.</p>
  <p><a href="index.html">Open signoff index</a></p>
  <div id="pad" class="box">Touch and pointer test area</div>
  <h2>Live Status</h2>
  <pre id="out">starting...</pre>
  <script>
    const out = document.getElementById("out");
    const pad = document.getElementById("pad");
    const events = [];
    function snapshot(kind, extra={}) {
      const data = {
        kind,
        time: new Date().toISOString(),
        href: location.href,
        secureContext: window.isSecureContext,
        userAgent: navigator.userAgent,
        viewport: {
          width: innerWidth,
          height: innerHeight,
          devicePixelRatio,
          visualWidth: visualViewport ? visualViewport.width : null,
          visualHeight: visualViewport ? visualViewport.height : null,
          visualScale: visualViewport ? visualViewport.scale : null,
        },
        scroll: {x: scrollX, y: scrollY},
        gpu: Boolean(navigator.gpu),
        ...extra,
      };
      events.push(data);
      while (events.length > 12) events.shift();
      out.textContent = events.map(e => JSON.stringify(e, null, 2)).join("\\n\\n");
      navigator.sendBeacon("/__phone_log", new Blob([JSON.stringify(data)], {type: "application/json"}));
    }
    ["pointerdown", "pointermove", "pointerup", "pointercancel"].forEach(name => {
      pad.addEventListener(name, event => snapshot(name, {
        pointerType: event.pointerType,
        pointerId: event.pointerId,
        isPrimary: event.isPrimary,
        x: Math.round(event.clientX),
        y: Math.round(event.clientY),
        pressure: event.pressure,
      }), {passive: true});
    });
    ["touchstart", "touchmove", "touchend", "touchcancel"].forEach(name => {
      pad.addEventListener(name, event => snapshot(name, {
        touches: event.touches.length,
        changedTouches: event.changedTouches.length,
      }), {passive: true});
    });
    ["resize", "orientationchange", "scroll"].forEach(name => {
      window.addEventListener(name, () => snapshot(name), {passive: true});
    });
    snapshot("load");
  </script>
</body>
</html>
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", 0))
        return int(sock.getsockname()[1])


def _local_ips() -> list[str]:
    ips: list[str] = []
    try:
        host = socket.gethostname()
        for item in socket.getaddrinfo(host, None, socket.AF_INET):
            ip = str(item[4][0])
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return ips


def _tailscale_ips() -> list[str]:
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _write_probe(root: Path) -> None:
    (root / "phone-probe.html").write_text(PHONE_PROBE, encoding="utf-8")


def _print_urls(port: int, root: Path) -> None:
    print("== quantem.widget phone handoff ==")
    print(f"root: {root}")
    print(f"local: http://127.0.0.1:{port}/index.html")
    for ip in _local_ips():
        print(f"lan: http://{ip}:{port}/index.html")
    for ip in _tailscale_ips():
        print(f"tailscale-http: http://{ip}:{port}/index.html")
    print(f"phone probe: http://127.0.0.1:{port}/phone-probe.html")
    print()
    print("For physical iPhone Safari WebGPU checks, prefer HTTPS through Tailscale:")
    print(f"  tailscale serve --bg --https=443 http://127.0.0.1:{port}")
    print("Then open the printed https://<machine>.<tailnet>.ts.net/index.html on the iPhone.")
    print("Stop Tailscale serve later with:")
    print("  tailscale serve --https=443 off")
    print()
    print("Server is running. Press Ctrl-C to stop.")


class _Handler(http.server.SimpleHTTPRequestHandler):
    root: Path
    log_path: Path

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(self.root), **kwargs)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/__phone_log":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}
        payload["_received_unix"] = time.time()
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        message = format % args
        print(f"{self.address_string()} - {message}")

    def translate_path(self, path: str) -> str:
        # Keep SimpleHTTPRequestHandler behavior but make URL decoding explicit.
        path = unquote(path.split("?", 1)[0].split("#", 1)[0])
        return super().translate_path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path, help="Directory containing index.html or exported widget HTML.")
    parser.add_argument("--port", type=int, default=0, help="Port to bind on 0.0.0.0. Default chooses a free port.")
    args = parser.parse_args()

    root = args.artifact_dir.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"artifact directory does not exist: {root}")
    _write_probe(root)
    port = args.port or _free_port()
    _Handler.root = root
    _Handler.log_path = root / "phone-events.ndjson"
    _print_urls(port, root)
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
