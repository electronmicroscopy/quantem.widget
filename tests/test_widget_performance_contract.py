"""Small performance contracts for notebook save paths.

These are not browser FPS benchmarks. They protect the regression that matters
for Cmd+S / reopen: a lightweight widget save must finish quickly enough for
interactive notebooks and must not serialize the heavy pixel buffers.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from PIL import Image

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


def test_show3d_frame_server_mount_defers_initial_native_frame_wire():
    """C1: remote users see controls before the first full frame payload."""
    widget = Show3D(
        np.zeros((2, 2049, 2049), dtype=np.float32),
        offline=False,
        save_state=False,
        title="defer initial frame wire",
    )

    assert widget.frame_server_url
    assert widget.frame_bytes == b""
    assert widget.frame_seq == 0

    widget.goto(0)

    assert len(widget.frame_bytes) == 2049 * 2049 * 4
    assert widget.frame_seq == 1


def test_show3d_lazy_panel_folders_mount_without_preloading_stack(tmp_path):
    """C1: remote real-data panels mount from disk paths, frames fetch native."""
    panel_dirs = []
    for panel in range(2):
        folder = tmp_path / f"panel_{panel}"
        folder.mkdir()
        panel_dirs.append(folder)
        for idx in range(3):
            frame = np.full((8, 10), panel * 100 + idx, dtype=np.uint16)
            Image.fromarray(frame).save(folder / f"frame_{idx:03d}.png")

    widget = Show3D.from_panel_folders(
        panel_dirs,
        offline=False,
        save_state=False,
        title="lazy panel folders",
    )

    assert widget.n_slices == 3
    assert widget.n_panels == 2
    assert widget.separate_panel_frames
    assert widget.frame_server_url
    assert widget.frame_bytes == b""
    assert widget.width == 20
    assert widget.height == 8
    assert widget._lazy_panel_cache_bytes == 0
    assert len(widget._lazy_panel_cache) == 0

    widget.goto(2)

    assert widget.frame_bytes == b""
    assert widget.frame_seq >= 1

    status, frame = widget._frame_for_http(2, widget.frame_server_version, panel=1)

    assert status == 200
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (8, 10)
    assert frame.dtype == np.float32
    assert frame.nbytes == 8 * 10 * 4
    assert float(frame[0, 0]) == 102.0


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


def test_show3d_paged_side_panels_align_with_image_canvas():
    """C1: paged right-side FFT/kymograph panels keep canvas tops aligned."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "!fftLayoutBottom && controlsVisible && isPaged" in show3d
    assert "minHeight: 28" in show3d
    assert "borderBottom: `1px solid ${themeColors.border}`" in show3d
    assert "controlsVisible && isPaged" in show3d


def test_show3d_histogram_drag_repaints_single_file_exports():
    """C1: normal exported Show3D histograms react while the user drags."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")
    stress = (ROOT / "scripts" / "widget_show3d_stress.py").read_text(encoding="utf-8")

    assert show3d.count("commitOnChange={!sidecarMode}") == 2
    assert "freezeCurrentPanelContrastAsManual(panel, { min, max })" in show3d
    assert "keep the visible Auto windows as the editable manual baseline" in show3d
    assert (
        "if (sidecarMode) window.setTimeout(commitPanelRange, 0);\n"
        "                              else commitPanelRange();"
    ) in show3d
    assert (
        "if (sidecarMode) window.setTimeout(commitSharedRange, 0);\n"
        "                      else commitSharedRange();"
    ) in show3d
    assert '"histogram_contrast": histogram_contrast' in stress
    assert "manual independent histogram drag reset other panels to full range" in stress


def test_show3d_arrow_keys_are_stress_tested_for_frame_navigation():
    """C1: keyboard frame navigation remains a real browser stress gate."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")
    stress = (ROOT / "scripts" / "widget_show3d_stress.py").read_text(encoding="utf-8")

    assert 'const FRAME_NAVIGATION_KEYS = new Set(["ArrowLeft", "ArrowRight", "Home", "End"]);' in show3d
    assert "shouldIgnoreWidgetShortcut(e.target, e.key)" in show3d
    assert "rootRef.current?.focus({ preventScroll: true });" in show3d
    assert "target.closest(WIDGET_TEXT_OR_VALUE_CONTROL_SELECTOR)" in show3d
    assert "def _drive_keyboard_scrub(page, *, fps_ms: int)" in stress
    assert 'page.keyboard.press("ArrowRight")' in stress
    assert '"keyboard_scrub": keyboard_scrub' in stress
    assert "ArrowRight did not change the rendered frame" in stress


def test_show2d_folder_uint8_export_uses_per_panel_frame_files():
    """C1: large exact 4k folder exports avoid one giant browser allocation."""
    show2d_py = (ROOT / "src" / "quantem" / "widget" / "show2d.py").read_text(encoding="utf-8")
    show2d_js = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")
    profile = (ROOT / "scripts" / "widget_external_html_profile.py").read_text(encoding="utf-8")

    assert "frame_bytes_urls = traitlets.List" in show2d_py
    assert 'name = f"frame_{image_index:06d}.bin"' in show2d_py
    assert '"frame_bytes_files": frame_file_names' in show2d_py
    assert 'useModelState<string[]>("frame_bytes_urls")' in show2d_js
    assert "setFetchedFrameBytePanels(loaded.slice())" in show2d_js
    assert "data-show2d-main-canvas={i}" in show2d_js
    assert "data-show2d-selected-panel={selectedIdx}" in show2d_js
    assert "onMouseDownCapture={handleRootMouseDownCapture}" in show2d_js
    assert "const uint8FolderPreviewMode = !!uint8FolderFrameBytes ||" in show2d_js
    assert "previousSource.frameSourceKey !== frameSourceKey" in show2d_js
    assert "def _show2d_main_canvas_screenshots" in profile
    assert "visible Show2D main canvases were blank/flat" in profile
    assert "def _run_show2d_keyboard_step" in profile
    assert "show2d_keyboard_shortcuts ArrowRight did not change the selected panel" in profile


def test_show2d_show3d_offline_export_bakes_current_view_state():
    """C1: standalone HTML Export reopens with the scientist's tuned view."""
    format_ts = (ROOT / "js" / "format.ts").read_text(encoding="utf-8")
    show2d = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "quantem-current-widget-view-state" in format_ts
    assert "standaloneHtmlWithCurrentWidgetState" in format_ts
    assert "standaloneWidgetStaticHtmlFromDocument" in format_ts
    assert "application/vnd.jupyter.widget-state+json" in format_ts
    assert "application/vnd.jupyter.widget-view+json" in format_ts
    assert "updateEmbeddedWidgetStateScript" in format_ts
    assert "applyStandaloneWidgetViewState" in format_ts
    assert "stripCurrentWidgetViewState(html)" in format_ts
    assert "JSON.stringify(value)" in format_ts
    assert "replace(/</g" in format_ts

    assert "SHOW2D_STANDALONE_VIEW_STATE_KEYS" in show2d
    assert "SHOW3D_STANDALONE_VIEW_STATE_KEYS" in show3d
    for source in (show2d, show3d):
        assert "React.useLayoutEffect(() => applyStandaloneWidgetViewState(model), [model]);" in source
        assert "standaloneHtmlWithCurrentWidgetState(" in source
        assert "standaloneWidgetStaticHtmlFromDocument()" in source
    for key in ("hidden_panels", "hidden_page_slots", "contrast_preset"):
        assert f'"{key}"' in show2d
        assert f'"{key}"' in show3d
    assert '"selected_idx"' in show2d
    assert '"slice_idx"' in show3d


def test_show2d_svg_preview_reuses_export_payload_and_snaps_pixels():
    """C1: publication preview displays the same SVG that Export saves."""
    show2d = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")

    assert "type Show2DSvgPreview = Show2DSvgExport" in show2d
    assert "const buildSvgExport = React.useCallback" in show2d
    assert "const built = svgPreview && svgPreview.scale === exportScale" in show2d
    assert 'data-show2d-svg-preview-image="true"' in show2d
    assert "snapSvgPreviewToDevicePixels" in show2d
    assert "Math.ceil(layoutTop * dpr) / dpr - layoutTop" in show2d
    assert "marginTop: svgPreviewSnap.y" in show2d


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
    # C2: many pages of large panels must fit one overview FFT per panel in
    # the bounded LRU, so revisiting the first page does not recompute it.
    assert "PAGED_GALLERY_FFT_OVERVIEW_MAX_DIM = 1024" in show2d
    assert "const overviewMaxDim = isPaged" in show2d
    # C3: the page player is a visual sweep, not a slow slide show.
    assert "PAGE_PLAY_FPS_OPTIONS = [1, 2, 4, 8, 12]" in show2d
    assert "useState<number>(8)" in show2d


def test_show2d_hidden_panels_skip_hot_render_paths():
    """C1: hiding gallery panels removes them from zoom/detail/render work."""
    show2d = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")

    assert "visibleImageIndicesRef.current = visibleImageIndices;" in show2d
    assert "for (const i of visibleImageIndices) {\n        if (isRgbFlags && isRgbFlags[i]) continue; // painted directly above" in show2d
    assert "const indices = visibleImageIndices.filter(i => i >= 0 && i < capturedNImages);" in show2d
    assert "const signature = visibleImageIndices.map((i) => {" in show2d
    assert "for (const i of visibleImageIndices) {\n        const win = currentDetailWindow(i);" in show2d
    # C2: a large gallery only repaints the panels near the browser viewport
    # during zoom; offscreen panels keep their cached display until scrolled in.
    assert "const [viewportPanelIndices, setViewportPanelIndices]" in show2d
    assert 'rootMargin: "0px"' in show2d
    assert "const viewportPaintImageIndices" in show2d
    assert "const paintIndices = (" in show2d
    assert ") ? [activePanel] : viewportPaintImageIndices;" in show2d
    assert "for (const i of paintIndices) {\n      const canvas = canvasRefs.current[i];" in show2d
    assert "const beginViewInteraction" in show2d
    assert "beginViewInteraction(idx);" in show2d
    assert "if (hiddenPanelSet.has(panel)) return null;" in show2d
    assert "hiddenPanels && hiddenPanels.includes(panel)" not in show2d


def test_show2d_batch_selection_and_marker_frame_contract():
    """C1: users can shift-select panels, batch-hide, and mark full frames."""
    show2d = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert 'useModelState<number[]>("selected_panels")' in show2d
    assert "event.shiftKey" in show2d
    assert "event.metaKey || event.ctrlKey" in show2d
    assert "setPanelsHidden(selectedVisiblePanels, true)" in show2d
    assert 'case "h":' in show2d
    assert 'case "H":' in show2d
    assert "shouldIgnoreWidgetShortcut(e.target)" in show2d
    assert 'useModelState<number[]>("selected_panels")' in show3d
    assert "handlePanelSelectionMouseDown" in show3d
    assert "event.shiftKey" in show3d
    assert "event.metaKey || event.ctrlKey" in show3d
    assert "setPanelsHidden(selectedVisiblePanels, true)" in show3d
    assert 'case "h":' in show3d
    assert 'case "H":' in show3d
    assert "hasPanelChoices && selectedVisiblePanels.length > 0" in show3d
    assert "data-show3d-panel-selection={panel}" in show3d
    assert "for (const panel of batchPanels) next[panel] = value;" in show2d
    assert 'data-show2d-marker-style="around"' in show2d
    assert 'data-show3d-marker-style={markerAround ? "around" : "left"}' in show3d
    assert 'useModelState<PanelTitleStyle>("panel_title_style")' in show2d
    assert 'useModelState<PanelTitleStyle>("panel_title_style")' in show3d
    assert 'useModelState<number>("inter_panel_gap_px")' in show2d
    assert 'useModelState<string>("inter_panel_gap_color")' in show2d
    assert 'useModelState<number>("gallery_outer_border_px")' in show2d
    assert 'useModelState<string>("gallery_outer_border_color")' in show2d
    assert 'useModelState<number>("panel_inner_border_px")' in show2d
    assert 'useModelState<string>("panel_inner_border_color")' in show2d
    assert "galleryGapColorResolved" in show2d
    assert "panelInnerBorderColor" in show2d
    assert 'useModelState<MarkerMap>("row_markers")' in show2d
    assert 'useModelState<MarkerMap>("row_markers")' in show3d
    assert 'useModelState<PanelAnnotationSpec[][]>("panel_annotations")' in show2d
    assert 'useModelState<PanelAnnotationSpec[][]>("panel_annotations")' in show3d
    assert 'useModelState<PanelOverlaySpec[][]>("panel_overlays")' in show2d
    assert 'useModelState<PanelOverlaySpec[][]>("panel_overlays")' in show3d
    assert "drawPanelOverlays(ctx, panelOverlaySpecs" in show2d
    assert "drawPanelOverlays(ctx, overlaySpecs" in show3d
    assert "overlayDashPattern" in show2d
    assert "overlayDashPattern" in show3d
    assert "ctx.setLineDash(overlayDashPattern" in show2d
    assert "ctx.setLineDash(overlayDashPattern" in show3d
    assert "svgPanelOverlayElement" in show2d
    assert "svgPanelAnnotationElement" in show2d
    assert "svgInsetPlotElement" in show2d
    assert "svgColorbarElements" in show2d
    assert 'data-show2d-vector-layer="true"' in show2d
    assert "setPanelOverlays" in show2d
    assert "setPanelOverlays" in show3d
    assert "Overlay Edit" in show2d
    assert "Overlay Edit" in show3d
    assert "Reset Overlays" in show2d
    assert "Reset Overlays" in show3d
    assert "panelOverlayHit" in show2d
    assert "panelOverlayHit" in show3d
    assert "updateOverlayFromDrag" in show2d
    assert "updateOverlayFromDrag" in show3d
    assert "drawPanelOverlaySelection" in show2d
    assert "drawPanelOverlaySelection" in show3d
    assert "const shape = overlay.shape === \"rectangle\" ? \"rect\"" in show2d
    assert "const shape = overlay.shape === \"rectangle\" ? \"rect\"" in show3d
    assert "LATEX_SYMBOLS" in show2d
    assert "LATEX_SYMBOLS" in show3d
    assert "renderTextWithInlineMath" in show2d
    assert "renderTextWithInlineMath" in show3d
    assert 'expr.trim().replace(/\\\\+(?=[A-Za-z])/g, "\\\\")' in show2d
    assert 'expr.trim().replace(/\\\\+(?=[A-Za-z])/g, "\\\\")' in show3d
    assert 'data-quantem-math="true"' in show2d
    assert 'data-quantem-math="true"' in show3d
    assert "data-show2d-row-marker" in show2d
    assert "data-show2d-col-marker" in show2d
    assert "data-show2d-panel-title" in show2d
    assert "data-show2d-panel-annotation" in show2d
    assert "data-show3d-row-marker" in show3d
    assert "data-show3d-col-marker" in show3d
    assert "data-show3d-panel-title" in show3d
    assert "data-show3d-panel-annotation" in show3d
    assert "panelAnnotationSx(annotation)" in show2d
    assert "panelAnnotationSx(annotation)" in show3d
    assert "canDownloadCurrentHtml" in show3d
    assert "handleStandaloneHtmlDownload" in show3d
    assert 'mode === "hug"' in show2d
    assert 'mode === "hug"' in show3d
    assert 'sx.width = "fit-content"' in show2d
    assert 'sx.width = "fit-content"' in show3d
    assert 'mode === "panel" ? (defaults.width || "calc(100% - 56px)")' in show2d
    assert 'mode === "panel" ? (defaults.width || "calc(100% - 56px)")' in show3d
    assert "inset 0 0 0 3px ${panelMarkerColor(i)}, inset 0 0 0 5px rgba(0,0,0,0.9)" in show2d
    assert "inset 0 0 0 3px ${color}, inset 0 0 0 5px rgba(0,0,0,0.9)" in show3d


def test_show_panel_chrome_uses_rich_labels_and_collapsed_export():
    """C1: compact menus/stats render symbols/math and presentation keeps export."""
    show2d = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "showControls && controlsCollapsed && (exportEnabled || canDownloadCurrentHtml)" in show2d
    assert "showControls && controlsCollapsed && (exportEnabled || canDownloadCurrentHtml)" in show3d
    assert "(showTitle || showControls)" in show2d
    assert "ml: showTitle ? 0.75 : 0" in show2d
    assert "{panelTitleContent(panel)}" in show2d
    assert "{panelTitleContent(panel)}" in show3d
    assert "{panelTitleContent(statsIdx)}" in show2d
    assert "{panelTitleContent(st.panel)}" in show3d
    assert "(exportEnabled || canDownloadCurrentHtml || canExportStandaloneGif || canExportStandaloneMp4)" in show3d
    assert "WebCodecs H.264 support for standalone MP4 export" in show3d
    assert "browser WebCodecs H.264 when available" in show3d


def test_show3d_standalone_export_labels_exact_vs_uint8_sources():
    """C1: standalone Show3D export UI must distinguish exact and encoded sources."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert 'const standaloneHtmlMode = hasOfflineFloatStack ? "exact" : "quantized";' in show3d
    assert "Source: exact float32 embedded data." in show3d
    assert "Source: encoded uint8 standalone data. For best movie fidelity" in show3d
    assert "export/open HTML exact float32 from the live widget" in show3d
    assert "standaloneAnimationUsesEncodedUint8" in show3d
    assert "data-show3d-encoded-source-animation-warning" in show3d
    assert "GIF adds a 256-color palette step" in show3d
    assert 'encoding="full"' in show3d
    assert "Exact float32 is not embedded in this standalone page" in show3d


def test_show3d_mp4_export_defaults_are_publication_quality():
    """C1: MP4 exports default to full-quality review videos, not fast previews."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "const DEFAULT_ANIMATION_EXPORT_FPS = 8;" in show3d
    assert "const MIN_ANIMATION_TITLE_FONT_PX = 12;" in show3d
    assert "const MIN_ANIMATION_SCALE_FONT_PX = 12;" in show3d
    assert "const MIN_ANIMATION_SCALE_BAR_THICKNESS_PX = 5;" in show3d
    assert "const MIN_ANIMATION_OVERLAY_MARGIN_PX = 12;" in show3d
    assert "const [exportFps, setExportFps] = React.useState(DEFAULT_ANIMATION_EXPORT_FPS);" in show3d
    assert "const exportDurationSeconds = exportFrameIndices.length / exportFpsValue;" in show3d
    assert "frames · ${exportDurationSeconds.toFixed(1)} s at ${exportFpsValue} fps" in show3d
    assert "const animationPanelWidth = sharedPanelSource" in show3d
    assert "animationOutputScale(width, height, quality, downsample, maxEdgePx, visiblePanels, maxCols, panelGap)" in show3d
    assert "activePanels.length,\n      cols,\n      panelGapPx" in show3d
    assert "const applyMp4ExportDefaults = React.useCallback" in show3d
    assert 'setExportPanelMode("mp4");' in show3d
    assert 'setExportQuality("high");' in show3d
    assert "setExportMaxFrames(0);" in show3d
    assert "setExportFps(DEFAULT_ANIMATION_EXPORT_FPS);" in show3d
    assert 'setExportSpatialPreset("edge1024");' in show3d
    assert "Math.max(MIN_ANIMATION_TITLE_FONT_PX" in show3d
    assert "const targetBarPx = Math.min(60, panelOutW * 0.25);" in show3d
    assert "Math.max(MIN_ANIMATION_SCALE_BAR_THICKNESS_PX" in show3d
    assert "Try Size = Max edge 1024 px or 2x downsample." in show3d
    assert "onClick={applyMp4ExportDefaults}" in show3d


def test_show2d_fft_gallery_quality_labels_stay_readable():
    """C1: every Show2D FFT panel keeps SNR/peak text visible in-gallery."""
    show2d = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")

    assert "quantem-fft-panel-label-stack" in show2d
    assert "quantem-fft-quality-label" in show2d
    assert "FFT quality for ${panelLabel(i)}" in show2d
    assert 'bgcolor: "rgba(0,0,0,0.58)"' in show2d
    assert 'maxWidth: "calc(100% - 16px)"' in show2d
    assert 'textOverflow: "ellipsis"' in show2d
    assert 'alignItems: "flex-start"' in show2d
    assert 'zIndex: 6' in show2d
    assert "data-show2d-fft-resize-handle" in show2d
    assert "Resize FFT panels" in show2d
    assert "e.stopPropagation();" in show2d


def test_show3d_playback_style_menu_stays_next_to_playback_controls():
    """Show3D play-style choices belong in the playback row, not top More."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    # C1: a scientist chooses how time moves while watching playback, expect
    # the button next to the frame star and only curve-style options inside.
    assert "More playback style options" in show3d
    assert "Play Style" in show3d
    assert "applyPlaybackStylePreset" in show3d
    assert "Power In" in show3d
    assert "Power Out" in show3d
    assert "Ease In/Out" in show3d
    assert 'useModelState<number[]>("playback_path")' in show3d
    assert 'producing "Path 1"' in show3d
    assert "Loop, Bounce, fps, and range stay user-controlled" in show3d
    assert "title=\"More tools: Stats, Denoise, Filter, Sub-pixel alignment, Color, Flip, Compare\"" in show3d
    assert "Playback Dynamics" not in show3d
    assert "Hold Key Frame" not in show3d
    assert "Focus Range" not in show3d


def test_show3d_loop_off_stops_bounce_playback_contract():
    """C1: Loop is the only repeat switch, even when Bounce is enabled."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "Loop remains the master repeat control" in show3d
    assert "it must not keep ping-ponging forever" in show3d
    assert "if (pi >= pp.length) {\n            if (!c.loop) { setPlaying(false); return; }" in show3d
    assert "if (next > rangeEnd) {\n            if (!c.loop) { setPlaying(false); return; }" in show3d
    assert "else if (next < rangeStart) {\n            if (!c.loop) { setPlaying(false); return; }" in show3d


def test_show3d_frontend_playback_budget_is_sixty_fps():
    """C1: 60 fps Show3D playback stays reachable in the browser controls."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "const MAX_PLAYBACK_FPS = 60;" in show3d
    assert "effectiveFps >= 60" in show3d


def test_show3d_many_panel_zoom_uses_correct_transform_path():
    """C1: many-panel Show3D zoom must avoid known-bad direct GPU sampling."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")
    colormaps = (ROOT / "js" / "colormaps.ts").read_text(encoding="utf-8")

    assert "const renderGpuPackedPanelTransformSlice" in show3d
    assert "renderGpuPackedPanelTransformSlice(idx, false)" in show3d
    assert 'dbg.lastInteractionRenderPath = "webgpu-packed-panel-transform";' in show3d
    assert '"webgpu-grid-packed-panels-transform-direct-fragment"' in show3d
    assert 'mode !== "transformBurst"' in show3d
    assert '|| mode === "transformBurst"' in show3d
    assert "dbg.runTransformBurst = runTransformBurst;" in show3d
    assert "dbg.transformBurstResult = result;" in show3d
    assert "drawP95Ms" in show3d
    assert "MAX_INTERACTIVE_GRID_CANVAS_EDGE = 4096" in show3d
    assert "MAX_INTERACTIVE_GRID_CANVAS_PIXELS = 8_388_608" in show3d
    assert "const gridCanvasCap = isMultiPanelSource" in show3d
    assert "const [rootLayoutWidth, setRootLayoutWidth] = React.useState(0);" in show3d
    assert "const responsiveCols = rootLayoutWidth > 0" in show3d
    assert "return Math.max(1, Math.min(requestedCols, responsiveCols));" in show3d
    assert "dbg.layoutCols = _colsLocal;" in show3d
    assert "sourcePanelIndices?: number[]" in colormaps
    assert "opts.sourcePanelIndices?.[panel]" in colormaps
    assert "const hasActiveTransform = (opts.transforms || []).some" in colormaps
    assert "if (!hasActiveTransform && this.renderPackedPanelTransformComputeToCanvas" in colormaps
    assert "smooth_sample: u32" in colormaps
    assert "params.smooth_sample == 1u" in colormaps
    assert "origin_x: f32" in colormaps
    assert "origin_y: f32" in colormaps
    assert "let local_x = in.pos.x - params.origin_x;" in colormaps
    assert "let local_y = in.pos.y - params.origin_y;" in colormaps
    assert "params.flags.w == 1u" in colormaps
    assert "pu[11] = usesExplicitSourcePanels ? 1 : 0;" in colormaps
    assert "pf[6] = col * (panelW + gap);" in colormaps
    assert "pf[7] = row * (panelH + gap);" in colormaps
    assert "return vec4f(0.0, 0.0, 0.0, 1.0);" in colormaps
    assert "rgba[out_idx] = 0xFF000000u;" in colormaps
    assert "smooth?: boolean" in colormaps
    assert "function shouldSmoothDirectSample" in colormaps
    assert "projectedW >= Math.max(1, sourceWidth) * 0.75" in colormaps
    assert "pu[11] = shouldSmoothDirectSample(opts.smooth" in colormaps
    assert "smooth: c.smooth" in show3d
    assert "function samplePackedU8Viewport(" in show3d
    assert "const sample = samplePackedU8Viewport(" in show3d
    assert show3d.count("const sample = samplePackedU8Viewport(") == 2
    assert "ctx.imageSmoothingEnabled = smooth;" in show3d
    assert "if (!hasActivePanelTransform && hasPackedFrame && packedFrame)" in show3d
    assert '"webgpu-grid-separate-panels-packed-transform-fragment"' in show3d
    assert '"webgpu-grid-separate-panels-panel-slots-direct-fragment"' in show3d
    assert 'return drawCanvasTransformFallback("canvas-panel-transform");' in show3d
    assert "gpuDisplayVisibleRef.current = false;" in show3d
    assert "gpuRenderSerialRef.current++;" in show3d
    assert "const confirmOfflineStaticCanvasPresent" in show3d
    assert 'dbg.lastInitialStaticPresent = `${reason}:${phase}`;' in show3d
    assert 'confirmOfflineStaticCanvasPresent("offline-static");' in show3d
    assert "lastDrawMainPreservedGpu" in show3d
    assert '"active-view-transform"' in show3d
    assert "sidecarViewTransformActive() &&" in show3d
    assert "Restores 60 fps on the GPU-cached multi-panel path" in show3d


def test_show3d_offline_viewport_honors_smooth_toggle():
    """C1: a smooth offline zoom interpolates within, never across, panel strips."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "function samplePackedU8Viewport(" in show3d
    assert "const x1 = Math.min(maxX, x0 + 1);" in show3d
    assert "if (!smooth) return values[y0 * width + x0];" in show3d
    assert show3d.count("const sample = samplePackedU8Viewport(") == 2
    assert "ctx.imageSmoothingEnabled = smooth;" in show3d
    # C2: toggling Smooth invalidates the rendered composite rather than
    # briefly applying the new zoom sampler to a cache made with old pixels.
    assert 'smooth,' in show3d[show3d.index("const sidecarDisplayStyleKey"):show3d.index("const sidecarDisplayStyleKey") + 700]
    assert show3d.count("sidecarCompositeStyleKeyRef.current === sidecarDisplayStyleKey") == 2
    assert 'sidecarCompositeStyleKeyRef.current = "";' in show3d


def test_show3d_offline_zoom_keeps_retained_canvas_visible():
    """C1: standalone zoom never reveals a cleared WebGPU presentation frame."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert 'if (offline) return drawCanvasTransformFallback("canvas-packed-transform");' in show3d


def test_browser_smoke_guards_zoom_canvas_continuity():
    """C1: browser smoke fails when a zoom briefly exposes a cleared canvas."""
    smoke = (ROOT / "scripts" / "widget_browser_smoke.py").read_text(encoding="utf-8")

    assert "def _start_zoom_continuity_probe" in smoke
    assert "def _exercise_zoom_continuity" in smoke
    assert "zoom changed the visible canvas composition" in smoke
    assert "zoom exposed a blank or flat canvas frame" in smoke
    assert 'if widget in {"show2d", "show3d", "showptycho"}:' in smoke


def test_show3d_stress_runner_covers_exact_and_sidecar_exports():
    """C1: real-data Show3D stress must cover both export transport paths."""
    script = (ROOT / "scripts" / "widget_show3d_stress.py").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "maintainer" / "performance-ui-testing.md").read_text(encoding="utf-8")

    assert "--html" in script
    assert "--sidecar-dir" in script
    assert "--make-sidecar-from-html" in script
    assert "--independent-contrast" in script
    assert '_set_labeled_switch_with_retry(page, "Contrast", False)' in script
    assert "attempts: int = 20" in script
    assert "RangeRequestHandler" in script
    assert "offline_stack.u8" in script
    assert "_offline_float_stack" in script
    assert "first_paint_ms" in script
    assert "layoutRequestedMaxCols" in script
    assert "lastInteractionRenderPath" in script
    assert "canvas became blank after zoom/pan stress" in script
    assert "sidecar target did not request offline_stack.u8" in script
    assert "scripts/widget_show3d_stress.py" in docs
    assert "exact single-file HTML and folder sidecar paths" in docs


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
        "}) || browserFilterOnRef.current || frequencyFilterIsActive;"
        in show3d
    )
    assert (
        "if (frequencyFilterPendingRef.current.has(key)) "
        "return allowRawOnMiss ? frame : null;"
    ) in show3d
    assert "browserFilterReadyForIndex" in show3d
    assert "low-pass(raw)" in show3d
    assert (
        "if (frequencyFilterIsActive && browserFilterKnobsOn "
        "&& !browserFilterReadyForIndex(idx))"
    ) in show3d
    assert "Do not clamp this live DOM update to loop handles" in show3d
    assert "users see \"1/18\" while the canvas was already showing a later frame" in show3d
    assert 'data-show3d-playback-count="true"' in show3d
    assert "if (!offline && playing && frameTransformActive()) return;" in show3d
    assert "visible \"twitch\"" in show3d


def test_show2d_show3d_colormap_sharing_is_advanced_more_menu_state():
    """Per-panel colormap editing stays discoverable but off by default."""
    show2d = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    # C1: a scientist usually wants one colormap across comparison panels, so
    # the main toolbar should only show Color. Advanced per-panel color is
    # unlocked from More without adding a noisy "Link Color" toolbar label.
    for source in (show2d, show3d):
        assert "Color shared" in source
        assert "Toggle shared panel colormap" in source
        assert "Shared keeps one colormap for every panel" in source
        assert "setPanelCmaps([])" in source
    assert "const colorShared = !(panelCmaps && panelCmaps.length === nImages" in show2d
    assert "const colorShared = normalizedPanelCmaps.length !== Math.max(1, nPanels || 1)" in show3d
    assert "setSelectedCmap(String(e.target.value))" in show2d
    assert "setCmapForPanel(" in show3d
    assert "Link Color" not in show2d
    assert "Link Color" not in show3d


def test_show3d_subpixel_alignment_lives_in_more_and_repaints_display_only():
    """Sub-pixel alignment should be a simple More-menu display transform."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")
    show3d_py = (ROOT / "src" / "quantem" / "widget" / "show3d.py").read_text(
        encoding="utf-8"
    )

    # C1: a scientist reviews a drifting single-panel stack, enables alignment
    # from More, picks a reference frame, then scrubs/plays without rewriting a
    # notebook cell or mutating the stored stack.
    assert 'useModelState<boolean>("subpixel_align_enabled")' in show3d
    assert 'useModelState<number>("subpixel_align_reference")' in show3d
    assert "Sub-pixel align" in show3d
    assert "Needs a single-panel client-side stack" in show3d
    assert "estimateSubpixelShift(reference, frame, width, height, gpu)" in show3d
    assert "shiftFrameBilinear(" in show3d
    assert "subpixelAlignFrameForIndex(idx, averagedFrameForIndex" in show3d
    assert "subpixelAlignFrameForIndex(refIdx, averagedFrameForIndex" in show3d
    assert "refreshCurrentDisplayFrameForTransform" in show3d
    assert "More → Align button can" in show3d
    assert "Ready · press Align to use frame" in show3d
    assert "Compute alignment now and repaint the current frame" in show3d
    assert 'subpixelAlignShiftsRef.current ? "Re-align" : "Align"' in show3d
    assert "near-zero row shifts on an intentionally drifted stack" in show3d
    assert "current row" in show3d
    assert "max ${maxRow.toFixed(1)}/${maxCol.toFixed(1)} px" in show3d
    assert "const paddedW = nextPow2(width);" in show3d
    assert "return { real: realCopy, imag: imagCopy, width: paddedW, height: paddedH }" in show3d
    assert "const row = wrappedPeakOffset(peakRow, corr.height) + rowDelta;" in show3d
    assert (
        "}) || browserFilterOnRef.current || frequencyFilterIsActive || !!subpixelAlignEnabled"
        in show3d
    )
    assert "raw data stays unchanged" in show3d
    assert "subpixel_align_enabled = traitlets.Bool(False).tag(sync=True)" in show3d_py
    assert "subpixel_align_reference = traitlets.Int(0).tag(sync=True)" in show3d_py
    assert "subpixel_align: bool = False" in show3d_py


def test_show3d_remote_scrub_transport_instrumentation_contract():
    """Remote scrub measurements must split Python, Comm, decode, and paint."""
    show3d_py = (ROOT / "src" / "quantem" / "widget" / "show3d.py").read_text(
        encoding="utf-8"
    )
    show3d_js = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "frame_transport_timing" in show3d_py
    assert "buffer_transport_timing" in show3d_py
    assert "_scrub_preview_request" in show3d_py
    assert "_scrub_preview_bytes" in show3d_py
    assert "_scrub_preview_info" in show3d_py
    assert "[Show3D scrub preview]" in show3d_py
    assert "release the slider or zoom/settle the view to request native full resolution" in show3d_py
    assert "_image_header_shape_and_range" in show3d_py
    assert "placeholder = np.zeros((1, 1, 1), dtype=np.float32)" in show3d_py
    assert "source_shape=(height, width)" in show3d_py
    assert "pythonPrepareMs" in show3d_py
    assert "pythonEncodeMs" in show3d_py
    assert "pythonTraitSetMs" in show3d_py
    assert "sendTimeMs" in show3d_py

    assert 'mode === "scrubTransport"' in show3d_js
    assert 'mode === "scrubPreviewTransport"' in show3d_js
    assert 'setScrubPreviewRequest(JSON.stringify({' in show3d_js
    assert 'kind: "scrubPreview"' in show3d_js
    assert 'if (!transformActive && requestScrubPreview(next)) return;' in show3d_js
    assert "requestCommFramePreview" in show3d_js
    assert 'requestCommFramePreview(normalized, "panel-server-fallback")' in show3d_js
    assert 'requestCommFramePreview(normalized, "frame-server-fallback")' in show3d_js
    assert "release the slider or zoom/settle the view to request native full resolution" in show3d_js
    assert "const commitSlice = (idx: number) =>" in show3d_js
    assert "setSliceIdx(next);" in show3d_js
    assert "browserReceiveLatencyMs" in show3d_js
    assert "jsDecodeMs" in show3d_js
    assert "endToEndUiLatencyMs" in show3d_js
    assert "__quantemShow3DPerf" in show3d_js
    assert "_defer_initial_frame_wire" in show3d_py
    assert "can mount controls immediately" in show3d_py
    assert "if (offline || !frameServerUrl || playing) return;" in show3d_js
    assert "data-show3d-native-cache-status" in show3d_js
    assert "Native cache" in show3d_js
    assert "Preview ready" in show3d_js
    assert "Reduced preview frame" in show3d_js
    assert "INITIAL_NATIVE_PREVIEW_DELAY_MS" in show3d_js
    assert 'requestCommFramePreview(normalized, "initial-native-preview")' in show3d_js
    assert "PANEL_GPU_READY_TIMEOUT_MS" in show3d_js
    assert "fetchSeparatePanelPackedFrameFromServer" in show3d_js
    assert "cpu-packed-panel-native" in show3d_js
    assert "if (separatePanelFrames) return false;" in show3d_js
    assert "mainOffscreenSourcePanelWidthRef.current !== undefined && !rawFrameDataRef.current" in show3d_js


def test_show3d_bottom_fft_layout_always_stacks_below_main_panel():
    """C1: user selects FFT Bottom, expect FFT below even for one panel."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert 'flexWrap: effectiveShowFft && fftLayoutBottom ? "wrap" : "nowrap"' in show3d
    assert 'flex: fftLayoutBottom ? "1 0 100%"' in show3d
    assert 'maxWidth: fftLayoutBottom ? "100%"' in show3d
    assert 'fftLayoutBottom && (nPanels || 1) > 1' not in show3d


def test_show3d_right_fft_layout_tracks_wrapped_toolbar_height_contract():
    """C1: side FFT canvas should stay aligned when the main toolbar wraps."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "const toolControlsRef = React.useRef<HTMLDivElement>(null)" in show3d
    assert "const [toolControlsHeight, setToolControlsHeight] = React.useState(28)" in show3d
    assert "new ResizeObserver(measure)" in show3d
    assert 'ref={toolControlsRef} data-show3d-tool-controls="true"' in show3d
    assert 'data-show3d-fft-tool-spacer="true"' in show3d
    assert "height: !fftLayoutBottom && controlsVisible ? toolControlsHeight : \"auto\"" in show3d


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
    assert "showZoomIndicator === true && panelChromeVisible" in show3d
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


def test_show3d_sidecar_playback_prefers_ready_gpu_frames():
    """C1: a cached multi-panel folder movie avoids CPU compositing."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")
    draw_sidecar = show3d.split("const drawSidecarBitmapFrame", 1)[1].split(
        "const previousSidecarPagePaintStartRef", 1
    )[0]

    gpu_path = "if (sidecarGpuReadyRef.current && renderSidecarGpuFrame(drawIdx, reason))"
    composite_path = "const composite = sidecarCompositeReadyRef.current"
    assert gpu_path in draw_sidecar
    assert composite_path in draw_sidecar
    assert draw_sidecar.index(gpu_path) < draw_sidecar.index(composite_path)


def test_show3d_sidecar_zoom_skips_viewport_cache_contract():
    """C1: zoomed sidecar inspection must not present stale viewport cache."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "invalidateSidecarViewportCache(\"view-transform\")" in show3d
    assert "sidecarViewportSkipReason" in show3d
    assert "\"view-transform\"" in show3d
    assert "sidecarDisplayCacheDirtyRef.current = true;" in show3d
    assert "sidecar-u8-viewport-transform" in show3d
    assert "sidecar-u8-viewport-live" in show3d
    assert "if (separatePanelFrames && !sidecarMode && imageRotation % 4 !== 0) return false;" in show3d
    assert "const packedPanelSource = !separatePanelFrames;" in show3d
    assert "if (separatePanelFrames) {" in show3d
    assert 'dbg.lastInteractionRenderPath = "webgpu-panel-transform";' in show3d
    assert "offline-panel-gpu-upload" in show3d
    assert '"layout-transform"' in show3d
    assert '"compare-blink"' in show3d
    assert '"visibility-sidecar"' in show3d
    assert "sidecarViewTransformActive()) return;" in show3d
    assert "byteRangeSource = \"auto-range\"" in show3d
    assert "byteRangeSource = \"histogram-preview\"" in show3d
    assert "valueToPanelByte(autoRange.vmin)" in show3d
    assert (
        "React.useLayoutEffect(() => {\n"
        "    if (\n"
        "      !offline ||\n"
        "      !sidecarMode ||\n"
        "      !sidecarRamReady"
    ) in show3d
    assert 'drawSidecarBitmapFrame(drawIdx, false, "immediate");' in show3d
    assert "debug.sidecarDisplayStyleDirty = true;" in show3d


def test_show3d_embedded_packed_zoom_reuses_frame_cache_contract():
    """C1: zooming a packed multi-panel export must not rebuild its whole movie."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")
    style_key = show3d.split("const sidecarDisplayStyleKey", 1)[1].split(
        "const syncPlaybackPanelTransform", 1
    )[0]
    transform_sync = show3d.split("const syncPlaybackPanelTransform", 1)[1].split(
        "const clampPanelViewForDraw", 1
    )[0]

    # C1: ordinary pan/zoom does not change a packed cache key; orientation
    # transforms retain the exact legacy per-pixel viewport path.
    assert "const packedViewportTransformRequiresRebuild" in show3d
    assert "packedViewportTransformRequiresRebuild ? Number(state.zoom" in style_key
    assert "packedViewportTransformRequiresRebuild ? Number(state.panX" in style_key
    assert "packedViewportTransformRequiresRebuild ? Number(state.panY" in style_key
    # C2: a packed cache invalidates on a viewport transform only when that
    # transform changes source-pixel orientation semantics.
    assert "sidecarMode ||" in transform_sync
    assert "packedViewportTransformRequiresRebuild" in transform_sync
    # C3: packed exports draw the cached frame through the current panel view.
    assert "paintEmbeddedPackedCompositeTransform" in show3d
    assert "embedded-packed-composite-transform" in show3d


def test_show3d_sidecar_export_preserves_display_contrast_contract():
    """C1: folder export should preserve microscope contrast state."""
    show3d = (ROOT / "src" / "quantem" / "widget" / "show3d.py").read_text(encoding="utf-8")
    export_body = show3d.split("    def export_sidecar(", 1)[1].split(
        "    def _compress_html_if_requested", 1
    )[0]

    assert "export_widget.auto_contrast = False" not in export_body
    assert "export_widget.vmin = 0.0" not in export_body
    assert "\"auto_contrast\": bool(export_widget.auto_contrast)" in export_body
    assert "\"offline_mins\": list(export_widget._offline_mins or [])" in export_body
    assert "serve_sidecar_range.py --dir ." in export_body
    assert "python3 -m http.server" not in export_body


def test_show3d_folder_review_range_server_is_shipped():
    """C1: collaborators can serve folder exports without private lab scripts."""
    server = (ROOT / "scripts" / "serve_sidecar_range.py").read_text(encoding="utf-8")

    assert "Accept-Ranges" in server
    assert "Content-Range" in server
    assert "206 if partial else 200" in server
    assert 'parser.add_argument("--dir", required=True' in server


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
    assert 'thumb.getAttribute("data-index") !== "1"' in show3d
    assert show3d.count("onPointerDownCapture={handleLoopSliderPointerDownCapture}") == 2


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


def test_show3d_fft_overlay_updates_during_playback_contract():
    """C1: Show3D playback recomputes FFT from the displayed frame."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")
    show3d_fft_compute = show3d.split(
        "// Compute FFT magnitude (expensive", 1
    )[1].split("// Clear FFT measurement", 1)[0]

    assert "FFT_PLAYBACK_UPDATE_INTERVAL_MS" in show3d
    assert "const playbackFft = Boolean(playing)" in show3d_fft_compute
    assert "fftPlaybackComputeInFlightRef" in show3d_fft_compute
    assert "playbackFft ? playbackIdxRef.current" in show3d_fft_compute
    assert "if (!data && offline) data = getOfflineFrame(fftFrameIdx)" in show3d_fft_compute
    assert "if (!playbackFft) cancelled = true" in show3d_fft_compute
    assert "dbg.lastFftFrame = fftFrameIdx" in show3d_fft_compute
    assert "dbg.lastFftPlayback = playbackFft" in show3d_fft_compute
    assert "During playback keep the last FFT visible" not in show3d_fft_compute


def test_show3d_fft_overlay_top_corners_clear_panel_titles_contract():
    """C1: dragged top FFT overlays should not collide with panel title chrome."""
    show3d = (ROOT / "js" / "show3d" / "index.tsx").read_text(encoding="utf-8")

    assert "function fftOverlayTopInsetPad" in show3d
    assert "showPanelTitles === false || panelCount <= 1" in show3d
    assert show3d.count("const topInsetPad = fftOverlayTopInsetPad(") == 2
    assert "panelTop + topInsetPad" in show3d


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
