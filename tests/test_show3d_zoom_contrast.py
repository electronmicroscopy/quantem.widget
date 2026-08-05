"""Show3D zoom/contrast frontend regression contracts."""

from __future__ import annotations

import pathlib


def test_packed_panel_frame_refresh_preserves_independent_contrast_on_zoom() -> None:
    """The retained zoom frame must use the settled per-panel contrast path."""
    frontend = (
        pathlib.Path(__file__).resolve().parents[1] / "js" / "show3d" / "index.tsx"
    ).read_text(encoding="utf-8")
    refresh = frontend.split(
        "const renderFloatFrameSlice =",
        1,
    )[1].split("const renderBufferedSlice", 1)[0]

    assert "const perPanelContrast" in refresh
    assert "resolvePanelRenderRange(" in refresh
    assert "engine.renderPerPanelGpuExplicit(" in refresh
    assert refresh.index("if (perPanelContrast)") < refresh.index(
        "engine.renderSlotsToImageBitmap([0]"
    )
    assert "drawMain(ctx, mainOffscreenRef.current)" in refresh


def test_offline_viewport_caches_keep_unlinked_auto_contrast_per_panel() -> None:
    """Cached scroll/zoom pixels must not reuse the packed-frame auto range."""
    frontend = (
        pathlib.Path(__file__).resolve().parents[1] / "js" / "show3d" / "index.tsx"
    ).read_text(encoding="utf-8")

    assert frontend.count(
        "const localAuto = !linkContrast ? panelStateRange : null;"
    ) == 2
    assert frontend.count("const autoRange = !linkContrast") == 2
    assert 'byteRangeSource = "auto-panel-state"' in frontend
    assert frontend.count("const panelContrastState =") == 2
    assert frontend.count(
        "const panelStateRange = !linkContrast ? panelContrastState : null;"
    ) == 1
    assert "? panelContrastState\n        : null;" in frontend
    assert 'import { packedPanelAutoByteRange } from "./packedPanelContrast";' in frontend
    assert frontend.count("packedPanelAutoByteRange(") == 2
    assert frontend.count("const localAutoBytes = !linkContrast") == 2
    assert 'byteRangeSource = "auto-panel-percentile"' in frontend
