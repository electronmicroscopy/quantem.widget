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


def test_show2d_local_stack_fft_cache_and_playback_contract():
    """Local slice playback keeps the prior FFT visible and reuses revisits."""
    show2d = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")
    local_stack = (ROOT / "js" / "show2d" / "localStack.ts").read_text(
        encoding="utf-8"
    )

    assert 'useModelState<number>("panel_playback_fps")' in show2d
    assert "panelPlaybackIntervalMs(panelPlaybackFps)" in show2d
    assert "galleryFftMagnitudeLruRef" in show2d
    assert "galleryFftActiveKeysRef" in show2d
    assert "galleryFftTargetKeysRef" in show2d
    assert "galleryFftComputeSerialRef" in show2d
    assert "__quantemShow2DPerf" in show2d
    assert "data-show2d-panel-playback-fps" in show2d
    assert "data-show2d-fft-cache-hits" in show2d
    assert "data-show2d-fft-computes" in show2d
    assert "fftComputing && !fftOffscreensRef.current[i]" in show2d
    assert "fftComputing && !fftMagCacheGalleryRef.current[i]" not in show2d
    assert "makeGalleryFftCacheKey" in local_stack
    assert "GALLERY_FFT_CACHE_MAX_BYTES" in local_stack
    assert "protectedKeys" in local_stack


def test_show3d_playback_dynamics_menu_uses_saved_state_traits():
    """Show3D time-series presets must be UI-accessible and state-backed."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    # C1: a scientist can create playback dynamics from the More menu, expect
    # the UI to write the durable API traits rather than keeping hidden local
    # browser-only state.
    assert "Playback Dynamics" in show3d
    assert "applyPlaybackDynamicsPreset" in show3d
    assert "Hold Key Frame" in show3d
    assert "Focus Range" in show3d
    assert 'useModelState<number[]>("playback_path")' in show3d
    assert "setPlaybackPath([...before, current, current, current, current, ...after])" in show3d
    assert "setLoopStart(Math.min(start, end))" in show3d
    assert "setBoomerang(true)" in show3d


def test_show3d_filtered_playback_waits_for_cached_display_frames():
    """Filtered playback must not flash raw frames while async filters settle."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    # C1: a user configures denoise or FFT filtering, presses Play, and expects
    # every transition to remain in the configured filtered view.
    assert "frequencyFilterCacheRef" in show3d
    assert "frequencyFilterPendingRef" in show3d
    assert "allowRawOnMiss" in show3d
    assert (
        "displayAndFrequencyFrameForIndex(next, frame, { allowRawOnMiss: false })"
        in show3d
    )
    assert "warmPlaybackDisplayFrame(next + warmDirection, next, frame)" in show3d
    assert (
        "}) || browserFilterOnRef.current || frequencyFilterIsActive"
        in show3d
    )
    assert (
        "if (frequencyFilterPendingRef.current.has(key)) "
        "return allowRawOnMiss ? frame : null;"
    ) in show3d
    assert "Do not clamp this live DOM update to loop handles" in show3d
    assert "users see \"1/18\" while the canvas was already showing a later frame" in show3d
    assert 'data-show3d-playback-count="true"' in show3d


def test_show3d_bottom_fft_layout_always_stacks_below_main_panel():
    """C1: user selects FFT Bottom, expect FFT below even for one panel."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert 'flexWrap: effectiveShowFft && fftLayoutBottom ? "wrap" : "nowrap"' in show3d
    assert 'flex: fftLayoutBottom ? "1 0 100%"' in show3d
    assert 'maxWidth: fftLayoutBottom ? "100%"' in show3d
    assert 'fftLayoutBottom && (nPanels || 1) > 1' not in show3d


def test_show2d_and_show3d_fft_zoom_labels_cover_every_interactive_fft():
    """FFT views expose their live multiplier in every supported layout."""
    figure = (ROOT / "js" / "figure.ts").read_text(encoding="utf-8")
    show2d = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    # C1: shared formatter and per-widget markers, expect every interactive FFT
    # layout to expose the same accessible N.N× zoom label contract.
    assert "export function formatZoomLabel" in figure
    assert 'data-show2d-fft-zoom-indicator={i}' in show2d
    assert 'data-show2d-fft-zoom-indicator="single"' in show2d
    assert "formatZoomLabel(getGalleryFftState(i).zoom)" in show2d
    assert "formatZoomLabel(fftZoom)" in show2d
    assert 'data-show3d-fft-zoom-indicator={panel}' in show3d
    assert show3d.count('data-show3d-fft-zoom-indicator={panel}') == 2
    assert "showZoomIndicator !== false && panelChromeVisible" in show3d
    assert "onTouchStart={handleFftInsetTouchStart}" in show3d
    assert "scheduleFftViewState({ zoom: newZoom" in show3d
    assert "`${fftZoom.toFixed(1)}×`" not in show3d


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


def test_show3d_offline_gpu_playback_owns_canvas_contract():
    """Standalone Show3D playback should not race a static canvas repaint."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "const offlineGpuPlaybackOwnsCanvas" in show3d
    assert show3d.count("if (offlineGpuPlaybackOwnsCanvas) return;") >= 2


def test_show3d_playback_row_bookmarks_current_frame_contract():
    """Show3D playback controls should let users star the current frame."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert (
        'const [bookmarkedFrames, setBookmarkedFrames] = '
        'useModelState<number[]>("bookmarked_frames")'
    ) in show3d
    assert "const normalizedBookmarkedFrames" in show3d
    assert "const toggleCurrentFrameBookmark" in show3d
    assert "aria-pressed={currentFrameBookmarked}" in show3d
    assert 'currentFrameBookmarked ? "Unstar" : "Star"' in show3d


def test_show3d_offline_packed_panel_playback_uses_per_panel_contrast_contract():
    """Packed multi-panel HTML playback must not fall back to global contrast."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")
    performance = (
        ROOT / "docs" / "maintainer" / "widget-performance.md"
    ).read_text(encoding="utf-8")
    performance_text = " ".join(performance.split())

    assert "const renderOfflinePackedPanels2D" in show3d
    assert '"offline-packed-panels-2d-per-panel"' in show3d
    assert "resolvePanelRenderRange(panel, panelRange, sharedAutoRange" in show3d
    assert "const offlinePackedPanelPlaybackUsesStaticCanvas" in show3d
    assert "!offlinePackedPanelPlaybackUsesStaticCanvas" in show3d
    assert "Show3D standalone HTML WebGPU playback blank" in performance
    assert "offlinePackedPanelPlaybackUsesStaticCanvas" in performance
    assert "active playback stays nonblank" in performance_text
    assert "not only after pause" in performance_text


def test_show3d_fft_cache_ignores_frame_delivery_counter():
    """Show3D FFT cache keys must survive repeated frame_bytes delivery."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "frame_seq can tick every time Python sends frame_bytes" in show3d
    assert "`data=${frameServerVersion || 0}`" in show3d
    assert "`seq=${frameSeq || 0}`" not in show3d
    assert "fftCacheInvalidations" in show3d
    assert "fftCacheHits" in show3d


def test_show2d_show3d_foreground_canvas_repaint_uses_cached_fft_contract():
    """Tab restore must repaint every canvas without recomputing FFT magnitudes."""
    lifecycle = (ROOT / "js" / "canvasLifecycle.ts").read_text(encoding="utf-8")
    show2d = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    # C1: foreground lifecycle events settle and coalesce before repainting.
    assert 'document.addEventListener("visibilitychange"' in lifecycle
    assert 'window.addEventListener("pageshow"' in lifecycle
    assert 'window.addEventListener("focus"' in lifecycle
    assert lifecycle.count("window.requestAnimationFrame") == 2
    assert "if (document.hidden) return" in lifecycle

    # C2: Show2D invalidates stale hidden-tab GPU work and rebuilds display
    # layers from retained image/FFT data.
    assert "const renderGeneration = ++mainCmapGenerationRef.current" in show2d
    assert "renderGeneration === mainCmapGenerationRef.current" in show2d
    assert "bitmaps.forEach(bitmap => bitmap?.close())" in show2d
    assert "data-show2d-canvas-repaint-signal" in show2d
    assert show2d.count("canvasRepaintSignal") >= 12
    show2d_single_fft_compute = show2d.split(
        "// Compute FFT magnitude (cached", 1
    )[1].split("// Clear FFT measurement", 1)[0]
    show2d_gallery_fft_compute = show2d.split(
        "// Compute FFT magnitudes for gallery mode", 1
    )[1].split("// Gallery FFT data effect", 1)[0]
    assert "canvasRepaintSignal" not in show2d_single_fft_compute
    assert "canvasRepaintSignal" not in show2d_gallery_fft_compute
    show2d_diff_fft_compute = show2d.split(
        "// FFT of a single visible diff pair", 1
    )[1].split("// Re-blit the cached diff FFT", 1)[0]
    assert "fftColormap" not in show2d_diff_fft_compute
    assert "canvasW" not in show2d_diff_fft_compute
    assert "fftSmooth" not in show2d_diff_fft_compute
    assert "setDiffFftMagVersion" in show2d_diff_fft_compute
    assert "diffFftOffscreenRef.current = offscreen" in show2d
    assert "const magnitude = diffFftMagRef.current" in show2d
    assert "const dims = diffFftDimsRef.current" in show2d
    assert "autoEnhanceFFT(magnitude, dims.width, dims.height)" in show2d
    assert "createGPUColormapEngine().then" in show2d
    assert "getGPUColormapEngine().then" not in show2d
    assert "Diff WebGPU FFT failed; using CPU worker" in show2d_diff_fft_compute
    assert "mag = result.magnitude" in show2d_diff_fft_compute
    assert "Do not shift again: the worker result is already centered" in show2d_diff_fft_compute
    assert "inputData = data.slice();\n          applyHannWindow2D(inputData, width, height);" in show2d_single_fft_compute
    assert "applyHannWindow2D(inputData, curW, curH)" in show2d_gallery_fft_compute
    assert "renderSlotsToImageBitmapAsync([fftSlot]" in show2d

    # C3: Show3D re-presents a live GPU frame or the stable 2D offscreen, while
    # bottom/inset FFT display effects reuse the cached magnitude.
    assert "renderGpuCachedSliceDirect(frameIdx, false)" in show3d
    assert "[temporary], [temporaryImgData], false" in show3d
    assert "if (renderSerial !== gpuRenderSerialRef.current) return false" in show3d
    assert 'lastVisibilityResumePath = restoredGpu ? "webgpu-cache"' in show3d
    assert "data-show3d-canvas-repaint-signal" in show3d
    assert show3d.count("canvasRepaintSignal") >= 15
    show3d_fft_compute = show3d.split(
        "// Compute FFT magnitude (expensive", 1
    )[1].split("// Clear FFT measurement", 1)[0]
    show3d_fft_display = show3d.split(
        "// Process FFT magnitude", 1
    )[1].split("// === Kymograph", 1)[0]
    assert "canvasRepaintSignal" not in show3d_fft_compute
    assert "canvasRepaintSignal" in show3d_fft_display
    show3d_resume_effect = show3d.split(
        "// A presented WebGPU texture is not a durable cache", 1
    )[1].split("// Render overlay (ROI only)", 1)[0]
    assert "gpuRenderSerialRef.current++" not in show3d_resume_effect
    assert "visibilityResumeMisses" in show3d_resume_effect
    assert 'lastVisibilityResumePath = "playback-owner"' in show3d_resume_effect
    assert "!transformActive && gpuDisplayVisibleRef.current" in show3d_resume_effect
    assert "shouldApplyClientDifference(offline, activeDiffMode)" in show3d
    assert "const extractPanelSlice = React.useCallback" in show3d
    assert "if (offlineGpuInFlight()) return" not in show3d_fft_compute
    assert "bufferRef.current = null" in show3d
    assert "nextBufferRef.current = null" in show3d
    assert "applyHannWindow2D(inputData, width, height)" in show3d_fft_compute
    assert "applyHannWindow2D(source, fftSourceW, fftSourceH)" in show3d_fft_compute
    assert "`transform=${diffMode}:${Math.max(1, Math.round(avgWindow || 1))}`" in show3d_fft_compute
    assert "ensureFftGpu, diffMode, avgWindow]" in show3d_fft_compute


def test_show1d_trace_hover_links_matching_snapshot_group():
    """Show1D point inspection should preview and pin matching image groups."""
    show1d = (ROOT / "js" / "show1d" / "index.tsx").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "api" / "show1d.md").read_text(encoding="utf-8")

    assert "snapshotGroupForPoint" in show1d
    assert "scheduleHoverAtPointer" in show1d
    assert "window.requestAnimationFrame(flushHoverFrame)" in show1d
    assert "commitScheduledHover(point, snapshotGroupForPoint(point))" in show1d
    assert "selectSnapshotGroup(snapshotGroup)" in show1d
    assert 'data-testid="show1d-snapshot-group-label"' in show1d
    assert 'data-testid="show1d-hover-readout"' in show1d
    assert 'data-testid="show1d-perf-telemetry"' in show1d
    assert "methodTickCharBudget" in show1d
    assert "__quantemShow1DPerf" in show1d
    assert "SNAPSHOT_FFT_CACHE_MAX_BYTES" in show1d
    assert 'useModelState<boolean>("controls_collapsed")' in show1d
    assert 'useModelState<boolean>("show_review")' in show1d
    assert 'useModelState<boolean>("show_trial_notes")' in show1d
    assert 'useModelState<boolean>("show_snapshot_histogram")' in show1d
    assert 'useModelState<string>("review_mode")' in show1d
    assert "optionalFiniteNumber" in show1d
    assert "snapshotFftGenerationRef.current !== generation" in show1d
    assert "snapshotFftPrewarmQueueRef" in show1d
    assert "visibleTraceSet.has(trace)" in show1d
    assert 'COLORMAPS.viridis' in show1d
    assert "csvAxisHeader" in show1d
    assert ">FFT:</Typography>" not in show1d
    assert 'flexWrap: { xs: "wrap", md: "nowrap" }' in show1d
    assert "its images and group label preview in the side panel" in api


def test_widget_performance_docs_cover_widget_stories_and_fft_cache():
    """Maintainer docs should keep the widget-wise performance story explicit."""
    performance = (ROOT / "docs" / "maintainer" / "widget-performance.md").read_text(
        encoding="utf-8"
    )
    automation = (ROOT / "docs" / "maintainer" / "automation.md").read_text(
        encoding="utf-8"
    )
    ui_testing = (
        ROOT / "docs" / "maintainer" / "performance-ui-testing.md"
    ).read_text(encoding="utf-8")
    performance_text = " ".join(performance.split())
    ui_testing_text = " ".join(ui_testing.split())

    assert "## Widget Performance Stories" in performance
    for widget in (
        "Show1D",
        "Show2D",
        "Show3D",
        "Show3DSlices",
        "Show4DSTEM",
        "ShowEDS",
        "ShowDiffraction",
        "ShowFolder",
    ):
        assert f"| {widget} |" in performance

    assert "cloud/CI" in performance_text
    assert "local heavy signoff" in performance_text
    assert "returning" in performance
    assert "cache hit" in performance
    assert "frame_seq" in performance

    assert "return-scrub cache" in automation
    assert "misses/computes" in automation
    assert "return scrub" in ui_testing_text
    assert "misses and computes stay unchanged" in ui_testing_text


def test_show3dslices_oblique_line_drag_contract_is_documented():
    """Show3DSlices oblique line editing should stay a maintained interaction."""
    show3dslices_js = (ROOT / "js" / "show3dslices" / "index.tsx").read_text(
        encoding="utf-8"
    )
    show3dslices_py = (
        ROOT / "src" / "quantem" / "widget" / "show3dslices.py"
    ).read_text(encoding="utf-8")
    api = (ROOT / "docs" / "api" / "show3dslices.md").read_text(encoding="utf-8")
    storyboard = (
        ROOT / "docs" / "maintainer" / "storyboard-show3dslices.md"
    ).read_text(encoding="utf-8")
    performance = (
        ROOT / "docs" / "maintainer" / "widget-performance.md"
    ).read_text(encoding="utf-8")
    performance_text = " ".join(performance.split())

    assert 'useModelState<{ row: number; col: number }[]>("oblique_profile_line")' in show3dslices_js
    assert "obliqueHandleDragRef" in show3dslices_js
    assert 'mode: "endpoint"' in show3dslices_js
    assert 'mode: "line"' in show3dslices_js
    assert "updateObliqueFromEndpoints" in show3dslices_js
    assert "updateObliqueFromLineDrag" in show3dslices_js
    assert "translateSegmentInsideImage" in show3dslices_js
    assert "const dxRaw = point.col - drag.origin.x" in show3dslices_js
    assert "const dyRaw = point.row - drag.origin.y" in show3dslices_js
    assert "liveFftSchedulerRef" in show3dslices_js
    assert "const scheduleLiveFft" in show3dslices_js
    assert "computeLiveFftAxis" in show3dslices_js
    assert "fftResultCacheRef" in show3dslices_js
    assert "makeFftResultCacheKey" in show3dslices_js
    assert "SHOW3DSLICES_FFT_RESULT_CACHE_MAX_BYTES" in show3dslices_js
    assert "fftCacheHits" in show3dslices_js
    assert "fftCacheMisses" in show3dslices_js
    assert "scheduleLiveFft([1], { segment: { start, stop } })" in show3dslices_js
    assert "scheduleLiveFft([0], { sliceZ: next[0] })" in show3dslices_js
    assert "const liveObliqueEditing = obliqueHandleDragRef.current !== null" in show3dslices_js
    assert "const debounceMs = liveObliqueEditing ? 16 : 80" in show3dslices_js
    assert "pausePlaybackForEdit" in show3dslices_js
    assert "oblique_profile_line" in show3dslices_py
    assert "_validate_oblique_profile_line" in show3dslices_py

    assert "Oblique line drag" in api
    assert "Drag either endpoint" in api
    assert "translate the whole cut freely in row/col" in api
    assert "Drag the oblique line endpoints" in storyboard
    assert "Drag the oblique line body" in storyboard
    assert "translates in row and col" in storyboard
    assert "oblique FFT panel follows" in storyboard
    assert "oblique line endpoint/body drags" in performance_text
    assert "oblique FFT redraw during line drag" in performance_text
    assert "FFT return-scrub cache hits" in performance_text


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
