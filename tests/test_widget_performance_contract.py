"""Small performance contracts for notebook save paths.

These are not browser FPS benchmarks. They protect the regression that matters
for Cmd+S / reopen: a lightweight widget save must finish quickly enough for
interactive notebooks and must not serialize the heavy pixel buffers.
"""

from __future__ import annotations

import time

import numpy as np

from quantem.widget import Show2D, Show3D


def test_show2d_lightweight_save_snapshot_is_fast_and_compact():
    rng = np.random.default_rng(91)
    widget = Show2D(
        rng.random((6, 768, 768), dtype=np.float32),
        display_bin="auto",
        save_state=False,
        verbose=False,
    )

    start = time.perf_counter()
    state = widget.get_state()
    elapsed = time.perf_counter() - start

    assert elapsed < 4.0
    assert "_static_fallback_jpeg" in state
    assert 1_000 < len(state["_static_fallback_jpeg"]) < 4_000_000
    assert "frame_bytes" not in state
    assert "_detail_bytes" not in state
    assert "export_payload" not in state


def test_show3d_lightweight_save_snapshot_is_fast_and_compact():
    rng = np.random.default_rng(92)
    widget = Show3D(
        rng.random((12, 256, 256), dtype=np.float32),
        save_state=False,
        title="save performance show3d",
    )

    start = time.perf_counter()
    state = widget.get_state()
    elapsed = time.perf_counter() - start

    assert elapsed < 4.0
    assert "_static_fallback_jpeg" in state
    assert 1_000 < len(state["_static_fallback_jpeg"]) < 4_000_000
    assert "frame_bytes" not in state
    assert "_buffer_bytes" not in state
    assert "_offline_stack" not in state
    assert "_offline_float_stack" not in state
    assert "export_payload" not in state
