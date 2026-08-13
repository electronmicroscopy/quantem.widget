"""Show3D zoom/contrast frontend regression contracts."""

from __future__ import annotations

import pathlib

import numpy as np

from quantem.widget import Show3D


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
    assert "return resolvePanelRenderRange(" in frontend
    assert "autoVminsPerPanel" in frontend
    assert "autoVmaxsPerPanel" in frontend
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


def test_playback_uses_live_per_panel_colormap_contract() -> None:
    """Resident, CPU, and offline playback must keep panel color identity."""
    root = pathlib.Path(__file__).resolve().parents[2]
    frontend = (root / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")
    colormaps = (root / "js" / "colormaps.ts").read_text(encoding="utf-8")

    assert "const panelCmapsLiveRef = React.useRef" in frontend
    assert "panelCmapsLiveRef.current = next" in frontend
    assert "const panelLuts = livePanelCmaps.length" in frontend
    assert "panelLuts," in frontend
    assert "const panelLut = COLORMAPS[panelCmap] || lut" in frontend
    assert "panelLuts?: { name: string; lut: Uint8Array }[]" in colormaps
    assert "private namedLutBuffer" in colormaps
    assert "slot.directRegionLutNames[panel] !== lutName" in colormaps


def test_playback_uses_live_panel_columns_with_live_canvas_geometry() -> None:
    """Layout changes during playback must not retain the Play-start columns."""
    root = pathlib.Path(__file__).resolve().parents[2]
    frontend = (root / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "canvasW, canvasH, panelCols: _colsLocal" in frontend
    assert "Math.round(c.panelCols || 1)" in frontend
    assert "canvasW, canvasH, _colsLocal" in frontend


def test_auto_contrast_caches_stable_ranges_for_mixed_unit_panels() -> None:
    """Phase and count panels must not share one absolute auto window."""
    rng = np.random.default_rng(21)
    phase = rng.normal(0.0, 0.02, (4, 24, 24)).astype(np.float32)
    bright_field = rng.normal(47_000, 500, (4, 24, 24)).astype(np.float32)
    dark_field = rng.normal(1_200, 100, (4, 24, 24)).astype(np.float32)

    widget = Show3D(
        phase,
        bright_field,
        dark_field,
        link_contrast=True,
        auto_contrast=True,
        display_bin=1,
        verbose=False,
    )

    assert len(set(widget.auto_vmins)) == 1
    assert len(set(widget.auto_vmaxs)) == 1
    np.testing.assert_allclose(
        widget.auto_vmins_per_panel,
        [np.percentile(phase, 0.5), np.percentile(bright_field, 0.5), np.percentile(dark_field, 0.5)],
        rtol=0.05,
    )
    np.testing.assert_allclose(
        widget.auto_vmaxs_per_panel,
        [np.percentile(phase, 99.5), np.percentile(bright_field, 99.5), np.percentile(dark_field, 99.5)],
        rtol=0.05,
    )
    assert widget.auto_vmaxs_per_panel[0] < 1
    assert widget.auto_vmins_per_panel[1] > 40_000


def test_multi_panel_contrast_defaults_to_independent_domains() -> None:
    """Unlike scientific panels must not share an absolute clip by default."""
    phase = np.zeros((2, 8, 8), dtype=np.float32)
    counts = np.ones((2, 8, 8), dtype=np.float32) * 40_000

    widget = Show3D(phase, counts, display_bin=1, verbose=False)
    comparable = Show3D(
        counts,
        counts + 100,
        link_contrast=True,
        display_bin=1,
        verbose=False,
    )

    assert widget.link_contrast is False
    assert comparable.link_contrast is True
