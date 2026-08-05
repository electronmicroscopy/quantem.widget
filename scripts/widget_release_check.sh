#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/widget_release_check.sh [--skip-wheel]

Runs the local quantem.widget release gates:
  - npm typecheck/test/static widget build
  - standalone browser build
  - offline browser build
  - Python compile smoke
  - local wheel build and wheel-content check

Options:
  --skip-wheel   skip the local wheel build/content check
  --help         show this help
EOF
}

skip_wheel=0
for arg in "$@"; do
  case "$arg" in
    --skip-wheel) skip_wheel=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

echo "== quantem.widget release check =="
echo "repo: $(pwd)"
echo "branch: $(git branch --show-current 2>/dev/null || echo unknown)"
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "== npm typecheck/test/build =="
npm run typecheck
npm test
npm run build
test -s src/quantem/widget/static/show2d.js
test -s src/quantem/widget/static/show4dstem.js

echo "== standalone browser build =="
(
  cd web
  npm run build
  npm run build:offline
)
test -s web/dist/index.html

echo "== stage offline browser artifact for wheel check =="
rm -rf src/quantem/widget/static/browser
mkdir -p src/quantem/widget/static/browser
cp -R web/dist/. src/quantem/widget/static/browser/
test -s src/quantem/widget/static/browser/index.html

echo "== Python compile smoke =="
python -m compileall -q src/quantem/widget/show4dstem_mps.py src/quantem/widget/__init__.py

if [[ "$skip_wheel" == "0" ]]; then
  echo "== local wheel build/content check =="
  rm -rf dist/widget-release-check
  python -m build . --no-isolation --outdir dist/widget-release-check
  python - <<'PY'
from pathlib import Path
import zipfile

wheels = sorted(Path("dist/widget-release-check").glob("quantem_widget-*.whl"))
if len(wheels) != 1:
    raise SystemExit(f"expected one wheel, found {wheels}")
wheel = wheels[0]
required = {
    "quantem/widget/static/chooselattice.js",
    "quantem/widget/static/show1d.js",
    "quantem/widget/static/show2d.js",
    "quantem/widget/static/show3d.js",
    "quantem/widget/static/show3dslices.js",
    "quantem/widget/static/show4dstem.js",
    "quantem/widget/static/showdiffraction.js",
    "quantem/widget/static/showeds.js",
    "quantem/widget/static/showptycho.js",
    "quantem/widget/static/browser/index.html",
}
with zipfile.ZipFile(wheel) as zf:
    names = set(zf.namelist())
missing = sorted(required - names)
if missing:
    raise SystemExit(f"{wheel} missing required files: {missing}")
print(f"wheel ok: {wheel}")
PY
fi

echo "ALL LOCAL RELEASE GATES PASS"
