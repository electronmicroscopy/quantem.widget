#!/usr/bin/env bash
# Verify quantem-widget from TestPyPI in a clean environment.
#
# Run this after pushing a version tag (which triggers CI publish).
# It creates a fresh conda env, installs from TestPyPI, verifies imports,
# and opens JupyterLab for visual inspection.
#
# Usage:
#   ./scripts/verify_testpypi.sh              # install latest version
#   ./scripts/verify_testpypi.sh 0.0.4        # install specific version
#
# After verifying, press Ctrl+C to stop JupyterLab, then clean up:
#   conda env remove -n test-widget-env -y
#
set -euo pipefail

ENV_NAME="test-widget-env"
NOTEBOOK="notebooks/test_pypi/test_all_widgets.ipynb"
VERSION="${1:-}"
PIP_VERSION="${VERSION:+==$VERSION}"

# Find conda
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Wait for version to appear on TestPyPI (CI may still be uploading)
if [ -n "$VERSION" ]; then
    echo "==> Waiting for v${VERSION} on TestPyPI..."
    for i in $(seq 1 30); do
        AVAILABLE=$(pip index versions quantem-widget \
            --index-url https://test.pypi.org/simple/ 2>&1 || true)
        if echo "$AVAILABLE" | grep -q "$VERSION"; then
            echo "  Found v${VERSION} on TestPyPI."
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "  ERROR: v${VERSION} not found on TestPyPI after 5 minutes."
            echo "  Check CI: https://github.com/bobleesj/quantem.widget/actions"
            exit 1
        fi
        echo "  Not yet available, retrying in 10s... ($i/30)"
        sleep 10
    done
fi

echo "==> Removing old env (if exists)..."
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true

echo "==> Creating fresh env: $ENV_NAME (Python 3.11)..."
conda create -n "$ENV_NAME" python=3.11 -y -q

echo "==> Activating $ENV_NAME..."
conda activate "$ENV_NAME"

echo "==> Installing quantem-widget${PIP_VERSION} from TestPyPI..."
pip install -q \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "quantem-widget${PIP_VERSION}"

echo "==> Installing jupyterlab..."
pip install -q jupyterlab

echo "==> Verifying imports and JS bundles..."
python -c "
from quantem.widget import (
    Show1D, Show2D, Show3D, Show3DVolume, Show4D, Show4DSTEM,
    ShowComplex2D, Mark2D, Edit2D, Align2D, Bin,
)
import numpy as np

img = np.random.rand(64, 64).astype(np.float32)
stack = np.random.rand(5, 64, 64).astype(np.float32)
vol = np.random.rand(16, 16, 16).astype(np.float32)
stem = np.random.rand(4, 4, 16, 16).astype(np.float32)
cplx = np.random.rand(64, 64).astype(np.float32) + 1j * np.random.rand(64, 64).astype(np.float32)
line = np.random.rand(100).astype(np.float32)

for cls, data, name in [
    (Show1D, line, 'Show1D'),
    (Show2D, img, 'Show2D'),
    (Show3D, stack, 'Show3D'),
    (Show3DVolume, vol, 'Show3DVolume'),
    (Show4D, stem, 'Show4D'),
    (Show4DSTEM, stem, 'Show4DSTEM'),
    (ShowComplex2D, cplx, 'ShowComplex2D'),
    (Mark2D, img, 'Mark2D'),
    (Edit2D, img, 'Edit2D'),
    (Align2D, (img, img), 'Align2D'),
]:
    w = cls(data) if not isinstance(data, tuple) else cls(*data)
    assert len(w._esm) > 1000, f'{name} JS bundle missing'
    print(f'  {name}: OK ({len(w._esm):,} chars)')
print()
print('All imports and instantiations passed.')
"

echo ""
echo "==> Opening JupyterLab — run all cells and visually verify each widget."
echo "    Press Ctrl+C when done, then clean up with:"
echo "    conda env remove -n $ENV_NAME -y"
echo ""
jupyter lab "$NOTEBOOK"
