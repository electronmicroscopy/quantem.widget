import pathlib
import subprocess
import sys

import numpy as np
import pytest

from quantem.widget import (
    Align2D,
    Bin,
    Edit2D,
    Mark2D,
    Show2D,
    Show3D,
    Show3DVolume,
    Show4D,
    Show4DSTEM,
    ShowComplex2D,
)

def test_tool_parity_ci_gate_has_no_errors():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/check_tool_parity.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

@pytest.mark.parametrize(
    ("factory", "group"),
    [
        (lambda: Show2D(np.random.rand(8, 8).astype(np.float32)), "display"),
        (lambda: Show3D(np.random.rand(4, 8, 8).astype(np.float32)), "playback"),
        (lambda: Show3DVolume(np.random.rand(4, 8, 8).astype(np.float32)), "volume"),
        (lambda: Show4D(np.random.rand(4, 4, 8, 8).astype(np.float32)), "roi"),
        (lambda: Show4DSTEM(np.random.rand(4, 4, 8, 8).astype(np.float32)), "virtual"),
        (lambda: ShowComplex2D((np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)), "fft"),
        (lambda: Mark2D(np.random.rand(8, 8).astype(np.float32)), "points"),
        (lambda: Edit2D(np.random.rand(8, 8).astype(np.float32)), "edit"),
        (lambda: Align2D(np.random.rand(8, 8).astype(np.float32), np.random.rand(8, 8).astype(np.float32)), "alignment"),
        (lambda: Bin(np.random.rand(4, 4, 8, 8).astype(np.float32)), "binning"),
    ],
)
def test_runtime_tool_api_methods(factory, group):
    w = factory()

    assert w.set_disabled_tools([group]) is w
    assert group in w.disabled_tools

    assert w.lock_tool(group) is w
    assert group in w.disabled_tools

    assert w.unlock_tool(group) is w
    assert group not in w.disabled_tools

    assert w.set_hidden_tools([group]) is w
    assert group in w.hidden_tools

    assert w.hide_tool(group) is w
    assert group in w.hidden_tools

    assert w.show_tool(group) is w
    assert group not in w.hidden_tools

    assert w.apply_control_preset("all") is w
    assert w.hidden_tools == []
