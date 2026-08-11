"""Show3D zoom/contrast frontend regression contracts."""

from __future__ import annotations

import pathlib


def test_packed_panel_frame_refresh_preserves_independent_contrast_on_zoom() -> None:
    """The retained zoom frame must use the settled per-panel contrast path."""
    frontend = (
        pathlib.Path(__file__).resolve().parents[2] / "js" / "show3d" / "index.tsx"
    ).read_text(encoding="utf-8")
    assert "const perPanelContrast" in frontend
    assert "resolvePanelRenderRange(" in frontend
    assert "engine.renderPerPanelGpuExplicit(" in frontend
    assert "renderGpuCachedSliceDirect" in frontend
    assert "renderGpuPackedPanelTransformSlice" in frontend


def test_one_resident_renderer_keeps_unlinked_contrast_per_panel() -> None:
    """GPU and the CPU fallback share the per-panel contrast resolver."""
    frontend = (
        pathlib.Path(__file__).resolve().parents[2] / "js" / "show3d" / "index.tsx"
    ).read_text(encoding="utf-8")

    assert "const directPanelRanges =" in frontend
    assert "return resolvePanelRange(panel, bounds, sharedAutoRange);" in frontend
    assert "const renderOfflinePackedPanels2D =" in frontend
    assert "const range = resolvePanelRenderRange(" in frontend
    assert "sidecarComposite" not in frontend
    assert "packedPanelAutoByteRange" not in frontend


def test_scale_change_converts_manual_limits_before_repaint() -> None:
    """Linear limits must never be reused as logarithmic limits or vice versa."""
    frontend = (
        pathlib.Path(__file__).resolve().parents[2] / "js" / "show3d" / "index.tsx"
    ).read_text(encoding="utf-8")

    assert "const signedExpm1" in frontend
    assert "const changeLogScale = React.useCallback" in frontend
    assert "const convert = nextLogScale ? signedLog1p : signedExpm1;" in frontend
    assert "playRef.current.logScale = nextLogScale;" in frontend
    assert 'onChange={(e) => changeLogScale(e.target.value === "log")}' in frontend


def test_frame_scrub_is_raf_coalesced_and_commits_once_on_release() -> None:
    """Pointer floods repaint resident data but commit only on release."""
    root = pathlib.Path(__file__).resolve().parents[2]
    frontend = (root / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")
    colormaps = (root / "js" / "colormaps.ts").read_text(encoding="utf-8")

    assert "averageResidentSlotsInto" in colormaps
    assert "encoder.copyBufferToBuffer(slot.dataBuffer" in colormaps
    assert "const renderGpuTemporalAverageSliceDirect" in frontend
    assert "engine.averageResidentSlotsInto(scratchSlot, sourceSlots)" in frontend
    assert "window.requestAnimationFrame(paintCurrent)" in frontend
    assert "scrubToSlice(next, false)" in frontend
    assert "commitSlice(next);" in frontend
    assert "debug.scrubModelCommits" in frontend
    assert "debug.lastScrubCommitFrame = next" in frontend


def test_gpu_resident_display_controls_are_immediate_repaint_dependencies() -> None:
    """Every display-only control must re-present the current GPU frame."""
    frontend = (
        pathlib.Path(__file__).resolve().parents[2] / "js" / "show3d" / "index.tsx"
    ).read_text(encoding="utf-8")
    effect = frontend.split(
        "// Display controls repaint the current resident GPU slot immediately.",
        1,
    )[1].split("const lastTransformBurstBenchmarkTokenRef", 1)[0]

    assert "renderGpuCachedSliceDirect(idx, false)" in effect
    assert "updatePlaybackLiveControls(idx)" in effect
    for dependency in (
        "cmap",
        "panelCmaps",
        "logScale",
        "smooth",
        "imageVminPct",
        "imageVmaxPct",
        "autoContrast",
        "linkContrast",
        "panelStates",
        "vminPerPanel",
        "vmaxPerPanel",
    ):
        assert dependency in effect
