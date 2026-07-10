"""Small performance contracts for notebook save paths.

These are not browser FPS benchmarks. They protect the regression that matters
for Cmd+S / reopen: a lightweight widget save must finish quickly enough for
interactive notebooks and must not serialize the heavy pixel buffers.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from quantem.widget import Show2D, Show3D, Show4DSTEM

ROOT = Path(__file__).resolve().parents[1]


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


def test_show2d_and_show3d_paged_sliders_use_local_preview_contract():
    """Paged slider movement must render locally before the Python trait commit."""
    show2d = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "const pageCommitRafRef" in show2d
    assert "window.requestAnimationFrame" in show2d
    assert "const activePageStart = isPaged ? pageControlIdx" in show2d
    assert "hidden_page_slots" in show2d
    assert "setHiddenPageSlotsTrait(slots)" in show2d

    assert "const pageCommitRafRef" in show3d
    assert "window.requestAnimationFrame" in show3d
    assert "const displayPageIdx = pageSliderPreviewIdx === null" in show3d
    assert "const activePageStart = isPaged ? displayPageIdx" in show3d


def test_show3d_standalone_export_has_bounded_frame_prewarm_contract():
    """Heavy standalone Show3D HTML should cache/prewarm frames with counters."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "const OFFLINE_FRAME_CACHE_BYTES" in show3d
    assert "function orderedFramePrewarmIndices" in show3d
    assert "offlineFrameCacheRef" in show3d
    assert "offlineFrameCacheLimit" in show3d
    assert "offlineFrameCacheHits" in show3d
    assert "offlineFrameCacheMisses" in show3d
    assert "offlineFramePrewarmActive" in show3d
    assert "offlineFramePrewarmDone" in show3d


def test_debug_overlay_trait_round_trips_through_state_dict():
    """Debug overlay is a normal saved widget state, not a one-off frontend flag."""
    show2d = Show2D(np.zeros((8, 8), dtype=np.float32), debug=True, verbose=False)
    assert show2d.state_dict()["debug"] is True
    show2d_restored = Show2D(np.zeros((8, 8), dtype=np.float32), verbose=False)
    show2d_restored.load_state_dict(show2d.state_dict())
    assert show2d_restored.debug is True

    show3d = Show3D(np.zeros((3, 8, 8), dtype=np.float32), debug=True, verbose=False)
    assert show3d.state_dict()["debug"] is True
    show3d_restored = Show3D(np.zeros((3, 8, 8), dtype=np.float32), verbose=False)
    show3d_restored.load_state_dict(show3d.state_dict())
    assert show3d_restored.debug is True

    show4dstem = Show4DSTEM(
        np.zeros((3, 3, 8, 8), dtype=np.float32),
        debug=True,
        precompute_virtual_images=False,
        verbose=False,
    )
    assert show4dstem.state_dict()["debug"] is True
    show4dstem_restored = Show4DSTEM(
        np.zeros((3, 3, 8, 8), dtype=np.float32),
        precompute_virtual_images=False,
        verbose=False,
    )
    show4dstem_restored.load_state_dict(show4dstem.state_dict())
    assert show4dstem_restored.debug is True


def test_debug_overlay_frontend_contract_is_wired_for_key_widgets():
    """The Python debug trait must have a visible browser hook in each widget."""
    for widget_name in ("show2d", "show3d", "show4dstem"):
        source = (ROOT / "js" / widget_name / "index.tsx").read_text(encoding="utf-8")
        assert 'useModelState<boolean>("debug")' in source
        assert "useDebugFps(Boolean(debug))" in source
        assert "data-quantem-debug-badge" in source
        assert "Debug UI FPS" in source

    for path in (
        ROOT / "src" / "quantem" / "widget" / "show2d.py",
        ROOT / "src" / "quantem" / "widget" / "show3d.py",
        ROOT / "src" / "quantem" / "widget" / "show4dstem.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "debug = traitlets.Bool(False).tag(sync=True)" in source
        assert '"debug": self.debug' in source
