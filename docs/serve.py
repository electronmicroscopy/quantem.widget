"""Gzip-compressing preview server for the built docs.

Usage: python docs/serve.py [--host HOST] [--port PORT]
"""

import argparse
import gzip
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

COMPRESSIBLE = {".html", ".js", ".css", ".json", ".svg", ".txt", ".map"}


class GzipHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        # no stale pages during preview
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            path = os.path.join(path, "index.html")
        ext = os.path.splitext(path)[1].lower()
        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "")
        if accepts_gzip and ext in COMPRESSIBLE and os.path.isfile(path):
            with open(path, "rb") as f:
                data = gzip.compress(f.read(), 6)
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    html_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build", "html")
    if not os.path.isdir(html_dir):
        raise SystemExit("docs/_build/html does not exist; run `jupyter-book build docs` first")
    os.chdir(html_dir)

    print(f"Serving docs at http://{args.host}:{args.port}/")
    ThreadingHTTPServer((args.host, args.port), GzipHandler).serve_forever()


if __name__ == "__main__":
    main()
