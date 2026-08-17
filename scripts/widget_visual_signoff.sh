#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/widget_visual_signoff.sh [--quick] [--show4dstem] [--full]

Runs the opt-in widget visual signoff gates for agents.

Modes:
  --quick       Generic synthetic Jupyter visual smoke only.
  --show4dstem Include existing Show4DSTEM WebGPU browser/Jupyter smokes.
  --full        Run generic visual, Show4DSTEM browser/Jupyter smokes, and
                local release checks. Heavy real-data CUDA/MPS/offline signoff
                still follows docs/refactor/2026-06-06-show4dstem-agent-signoff-runbook.md.

Environment:
  QT_WIDGET_MIN_FPS          Minimum generic visual FPS, default 30.
  QT_WEBGPU_MIN_FPS          Minimum Show4DSTEM WebGPU FPS, default 30.
  CHROME_EXECUTABLE          Optional Chrome/Chromium path.
  QT_WEBGPU_LIVE_MASTER      Optional real-data master for WebGPU live test.
  QT_WEBGPU_LIVE_DET_BIN     Optional real-data detector bin, default 4.
  QT_WEBGPU_LIVE_DTYPE       Optional real-data dtype, default u8.
  QT_WEBGPU_LIVE_CROP        Optional real-data crop, default 96:160,96:160.
EOF
}

quick=0
show4dstem=0
full=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --quick) quick=1 ;;
    --show4dstem) show4dstem=1 ;;
    --full) full=1; show4dstem=1 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cd "$(dirname "$0")/.."

echo "== preflight =="
git branch --show-current
git rev-parse --short HEAD
git status --short

echo "== parity gates =="
python -m pytest -q \
  tests/test_fft_parity.py \
  tests/test_dpc_virtual_parity.py \
  tests/test_bslz4_offline.py \
  tests/test_load_cpu_uint8_clip.py \
  tests/test_state_dict.py

echo "== generic widget visual Jupyter smoke =="
QT_RUN_WIDGET_VISUAL_TESTS=1 python -m pytest -q tests/test_widget_visual_jupyter.py -s

if [[ "$show4dstem" -eq 1 ]]; then
  echo "== Show4DSTEM WebGPU live Jupyter smoke =="
  QT_RUN_JUPYTER_WEBGPU_TESTS=1 python -m pytest -q tests/show4dstem/test_webgpu_live_jupyter.py -s

  echo "== Show4DSTEM exported HTML WebGPU smoke =="
  QT_RUN_BROWSER_TESTS=1 python -m pytest -q tests/show4dstem/test_webgpu_browser.py -s
fi

if [[ "$full" -eq 1 ]]; then
  echo "== release build/check gate =="
  scripts/widget_release_check.sh
fi

echo "== done =="
