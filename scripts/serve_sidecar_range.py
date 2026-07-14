#!/usr/bin/env python3
"""Serve a Show3D folder export with HTTP Range support.

Show3D folder exports keep the viewer in ``index.html`` and the frame bytes in
``offline_stack.u8``. Browsers need byte-range requests for large stacks, and
``file://`` is not a reliable transport for that workflow.
"""

from __future__ import annotations

import argparse
import email.utils
import http.server
import mimetypes
from pathlib import Path
import posixpath
import re
import sys
import urllib.parse


CHUNK_BYTES = 1024 * 1024
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


class RangeRequestHandler(http.server.BaseHTTPRequestHandler):
    """Static file handler with single byte-range support."""

    root: Path

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_common_headers()
        self.end_headers()

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def do_GET(self) -> None:
        self._serve(send_body=True)

    def _serve(self, *, send_body: bool) -> None:
        path = self._resolve_path()
        if path is None:
            self.send_error(404, "file not found")
            return
        if path.is_dir():
            path = path / "index.html"
        if not path.is_file():
            self.send_error(404, "file not found")
            return

        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = size - 1
        partial = False
        if range_header:
            parsed = self._parse_range(range_header, size)
            if parsed is None:
                self.send_response(416)
                self._send_common_headers()
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = parsed
            partial = True

        content_length = max(0, end - start + 1)
        self.send_response(206 if partial else 200)
        self._send_common_headers()
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Last-Modified", email.utils.formatdate(path.stat().st_mtime, usegmt=True))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        if send_body:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = handle.read(min(CHUNK_BYTES, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except BrokenPipeError:
                        break
                    remaining -= len(chunk)

    def _send_common_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")

    def _resolve_path(self) -> Path | None:
        parsed = urllib.parse.urlsplit(self.path)
        raw_path = urllib.parse.unquote(parsed.path)
        norm = posixpath.normpath(raw_path)
        rel = Path(norm.lstrip("/"))
        candidate = (self.root / rel).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            return None
        return candidate

    @staticmethod
    def _parse_range(value: str, size: int) -> tuple[int, int] | None:
        match = RANGE_RE.fullmatch(value.strip())
        if not match or size < 0:
            return None
        start_text, end_text = match.groups()
        if start_text == "" and end_text == "":
            return None
        if start_text == "":
            suffix = int(end_text)
            if suffix <= 0:
                return None
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        if start >= size or end < start:
            return None
        return start, min(end, size - 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Folder export directory containing index.html.")
    parser.add_argument("--port", type=int, default=8803, help="Port to bind (default: 8803).")
    parser.add_argument("--bind", default="127.0.0.1", help="Address to bind (default: 127.0.0.1).")
    args = parser.parse_args(argv)

    root = Path(args.dir).expanduser().resolve()
    if not root.is_dir():
        print(f"serve_sidecar_range: not a directory: {root}", file=sys.stderr)
        return 2

    handler = type("ConfiguredRangeRequestHandler", (RangeRequestHandler,), {"root": root})
    server = http.server.ThreadingHTTPServer((args.bind, args.port), handler)
    url = f"http://{args.bind}:{args.port}/index.html"
    print(f"serving {root} at {url}")
    print("press Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
