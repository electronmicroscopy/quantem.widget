#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/widget_local_signoff.sh [--quick] [--full] [--browser] [--mobile] [--performance] [--skip-docs] [--artifact-dir DIR]

Canonical local automation before saying widget work is ready.

Default gates:
  - repo hygiene and size guards
  - npm run build
  - PYTHONPATH=src:. pytest -q
  - scripts/widget_html_smoke.py
  - jupyter-book build docs --all

Options:
  --quick        Run size guards, frontend build, focused HTML protocol tests,
                 and HTML export smoke. Skip full pytest/docs build.
  --full         Also run npm typecheck/test and widget_release_check.sh --skip-wheel.
  --browser      Drive generated HTML exports in Chromium, verify nonblank
                 canvases, exercise basic controls, and write screenshots.
  --mobile       With --browser, also run the browser smoke in a 390x844
                 touch viewport. This is a Chromium pre-check, not proof of
                 physical iPhone Safari behavior.
  --performance  Run real-data Show2D/Show3D export performance smoke.
  --skip-docs    Skip docs build.
  --artifact-dir DIR
                 Write reports and generated HTML under DIR. By default, use
                 /tmp/quantem-widget-local-signoff/<timestamp>.
EOF
}

mode=default
browser=0
mobile=0
performance=0
skip_docs=0
artifact_dir=""
start_unix="$(date +%s)"
branch="unknown"
commit="unknown"
docs_status="not-run"
dashboard_ready=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --quick) mode=quick ;;
    --full) mode=full ;;
    --browser) browser=1 ;;
    --mobile) mobile=1 ;;
    --performance) performance=1 ;;
    --skip-docs) skip_docs=1 ;;
    --artifact-dir)
      if [[ $# -lt 2 ]]; then
        echo "--artifact-dir requires a path" >&2
        exit 2
      fi
      artifact_dir="$2"
      shift
      ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cd "$(dirname "$0")/.."

if [[ -z "$artifact_dir" ]]; then
  artifact_dir="/tmp/quantem-widget-local-signoff/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$artifact_dir"
artifact_dir="$(cd "$artifact_dir" && pwd)"
branch="$(git branch --show-current 2>/dev/null || echo unknown)"
commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

write_dashboard() {
  local status="$1"
  local end_unix duration
  end_unix="$(date +%s)"
  duration="$((end_unix - start_unix))"
  mkdir -p "$artifact_dir"
  SIGNOFF_STATUS="$status" \
  SIGNOFF_REPOSITORY="$(pwd)" \
  SIGNOFF_BRANCH="$branch" \
  SIGNOFF_COMMIT="$commit" \
  SIGNOFF_MODE="$mode" \
  SIGNOFF_BROWSER="$browser" \
  SIGNOFF_MOBILE="$mobile" \
  SIGNOFF_PERFORMANCE="$performance" \
  SIGNOFF_DOCS_STATUS="$docs_status" \
  SIGNOFF_DURATION="$duration" \
  SIGNOFF_ARTIFACT_DIR="$artifact_dir" \
  python - <<'PY'
import json
import os
import pathlib

path = pathlib.Path(os.environ["SIGNOFF_ARTIFACT_DIR"]) / "signoff-manifest.json"
manifest = {
    "status": os.environ["SIGNOFF_STATUS"],
    "repository": os.environ["SIGNOFF_REPOSITORY"],
    "branch": os.environ["SIGNOFF_BRANCH"],
    "commit": os.environ["SIGNOFF_COMMIT"],
    "mode": os.environ["SIGNOFF_MODE"],
    "browser_smoke": os.environ["SIGNOFF_BROWSER"] == "1",
    "mobile_browser_precheck": os.environ["SIGNOFF_MOBILE"] == "1",
    "performance_smoke": os.environ["SIGNOFF_PERFORMANCE"] == "1",
    "docs_build": os.environ["SIGNOFF_DOCS_STATUS"],
    "duration_seconds": int(os.environ["SIGNOFF_DURATION"]),
    "artifact_dir": os.environ["SIGNOFF_ARTIFACT_DIR"],
}
path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY
  PYTHONPATH=src:. python scripts/widget_signoff_dashboard.py --artifact-dir "$artifact_dir" >/dev/null || true
}

finalize() {
  local exit_code="$?"
  if [[ "$dashboard_ready" -eq 1 ]]; then
    if [[ "$exit_code" -eq 0 ]]; then
      write_dashboard "pass"
    else
      write_dashboard "fail"
    fi
  fi
  exit "$exit_code"
}
trap finalize EXIT
dashboard_ready=1

echo "== quantem.widget local signoff =="
echo "repo: $(pwd)"
echo "branch: $branch"
echo "commit: $commit"
echo "artifact dir: $artifact_dir"
git status --short

echo "== size guards =="
python scripts/check_large_files.py
python scripts/check_notebook_sizes.py

echo "== stale browser artifact cleanup =="
python scripts/cleanup_browser_artifacts.py

echo "== frontend build =="
npm run build

if [[ "$mode" == "quick" ]]; then
  echo "== focused pytest =="
  PYTHONPATH=src:. pytest -q \
    tests/test_html_export_protocol.py \
    tests/test_showfolder.py \
    tests/test_automation_scripts.py
else
  echo "== full pytest =="
  PYTHONPATH=src:. pytest -q
fi

echo "== HTML export smoke matrix =="
PYTHONPATH=src:. python scripts/widget_html_smoke.py --artifact-dir "$artifact_dir/html-smoke"

echo "== ShowFolder live-folder smoke =="
PYTHONPATH=src:. python scripts/widget_showfolder_live_smoke.py --artifact-dir "$artifact_dir/showfolder-live"

if [[ "$browser" -eq 1 ]]; then
  echo "== browser-drive HTML smoke =="
  browser_args=(--artifact-dir "$artifact_dir/html-smoke")
  if [[ "$mobile" -eq 1 ]]; then
    browser_args+=(--mobile)
  fi
  PYTHONPATH=src:. python scripts/widget_browser_smoke.py "${browser_args[@]}"

  echo "== post-browser artifact cleanup =="
  python scripts/cleanup_browser_artifacts.py
fi

if [[ "$performance" -eq 1 ]]; then
  echo "== real-data performance smoke =="
  PYTHONPATH=src:. python scripts/widget_performance_smoke.py --artifact-dir "$artifact_dir/performance"

  if [[ "$browser" -eq 1 ]]; then
    echo "== browser-drive real-data performance smoke =="
    perf_browser_args=(--artifact-dir "$artifact_dir/performance")
    if [[ "$mobile" -eq 1 ]]; then
      perf_browser_args+=(--mobile)
    fi
    PYTHONPATH=src:. python scripts/widget_browser_smoke.py "${perf_browser_args[@]}"
  fi

  echo "== post-performance browser artifact cleanup =="
  python scripts/cleanup_browser_artifacts.py
fi

if [[ "$skip_docs" -eq 0 && "$mode" != "quick" ]]; then
  echo "== docs build =="
  # Match CI: docs pages bake live widget state, so skip the static preview
  # sibling that would render as a duplicate image under each widget.
  QUANTEM_WIDGET_STATIC_FALLBACK=0 jupyter-book build docs --all
  echo "== docs page size guard =="
  python scripts/check_docs_page_sizes.py
fi

if [[ "$mode" == "full" ]]; then
  echo "== frontend typecheck/test =="
  npm run typecheck
  npm test

  echo "== release check without wheel =="
  scripts/widget_release_check.sh --skip-wheel
fi

if [[ "$skip_docs" -eq 0 && "$mode" != "quick" ]]; then
  docs_status="built"
else
  docs_status="skipped"
fi

echo "== local signoff passed =="
echo "Signoff report: $artifact_dir/index.html"
