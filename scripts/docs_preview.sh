#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/docs_preview.sh [--no-build] [--port PORT] [--host HOST]

Build and serve the local documentation HTML.

Options:
  --no-build    Serve the existing docs/_build/html tree.
  --port PORT   Port for python -m http.server. Default: 8767.
  --host HOST   Bind host. Default: 127.0.0.1.
EOF
}

build=1
port=8767
host=127.0.0.1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --no-build) build=0 ;;
    --port) port="$2"; shift ;;
    --host) host="$2"; shift ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cd "$(dirname "$0")/.."

if [[ "$build" -eq 1 ]]; then
  jupyter-book build docs --all
fi

if [[ ! -d docs/_build/html ]]; then
  echo "docs/_build/html does not exist; rerun without --no-build" >&2
  exit 1
fi

echo "Serving docs at http://${host}:${port}/"
cd docs/_build/html
python -m http.server "$port" --bind "$host"
