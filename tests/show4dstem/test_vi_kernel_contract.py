from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantem.widget import Show4DSTEM


def _webgpu_source(name: str) -> str:
    repo = Path(__file__).resolve().parents[2]
    return (repo / "js" / ".generated" / "engine" / name).read_text(
        encoding="utf-8"
    )


def test_show4dstem_cuda_keeps_cupy_compute_source_for_rawkernel() -> None:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is not available.")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device is not available: {exc}")


    data = cp.ones((4, 4, 12, 12), dtype=cp.uint16)
    widget = Show4DSTEM(
        data,
        precompute_virtual_images=False,
        center=(5.5, 5.5),
        bf_radius=2.0,
    )
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True

    np.testing.assert_array_equal(
        widget._fast_masked_sum(mask),
        np.full((4, 4), int(mask.sum()), dtype=np.float32),
    )


def test_show4dstem_cuda_compare_grid_uses_rawkernel_frames() -> None:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is not available.")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device is not available: {exc}")


    data = cp.ones((2, 4, 4, 12, 12), dtype=cp.uint16)
    widget = Show4DSTEM(
        data,
        precompute_virtual_images=False,
        view_mode="multiple",
        compare_max_panels=2,
        center=(5.5, 5.5),
        bf_radius=2.0,
    )
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True

    panels = widget._compare_virtual_images_for_indices([0, 1], mask)

    assert len(panels) == 2
    for panel in panels:
        np.testing.assert_array_equal(panel, np.ones((4, 4), dtype=np.float32))
    assert list(widget._cuda_compare_compute_backends) == [0, 1]

    _ = widget._compare_virtual_images_for_indices([0, 1], mask)
    assert list(widget._cuda_compare_compute_backends) == [0, 1]


def test_show4dstem_uses_public_detector_session() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "quantem"
        / "widget"
        / "show4dstem.py"
    ).read_text(encoding="utf-8")

    assert "from quantem.gpu.detector import prepare" in source
    assert "quantem.gpu.detector.compute" not in source


def test_show4dstem_webgpu_engine_has_selected_index_vi_kernel() -> None:
    repo = Path(__file__).resolve().parents[2]
    source = _webgpu_source("detector/compute/webgpu/backend.ts")
    dpc_source = _webgpu_source("dpc/compute/webgpu/kernels.ts")
    frontend = (repo / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )

    assert "const maskedSumSrc = (sg: boolean)" in source
    assert "export function buildDetectorMask" in source
    assert "export function buildFullDetectorMask" in source
    assert "export function buildScanMask" in source
    assert "arrayLength(&idx)" in source
    assert "sampleF(base + idx[j]" in source
    assert "maskedSumBuffer(mask: Uint32Array)" in source
    assert "maskedDpcBuffer(mask: Uint32Array" in source
    assert "maskedIDpcBuffer(" in source
    assert "const IDPC_POISSON_WGSL" in dpc_source
    assert "getDevice(): GPUDevice" in source
    assert "readFloatBuffer(buf: GPUBuffer" in source
    assert "const DPC_MEAN_WGSL" in dpc_source
    assert "const DPC_COMPONENT_WGSL" in dpc_source
    assert "adoptBuffer(idx: number, buffer: GPUBuffer" in (
        repo / "js" / "colormaps.ts"
    ).read_text(encoding="utf-8")
    assert "renderSlotDirectWithGpuRangeToCanvas" in (
        repo / "js" / "colormaps.ts"
    ).read_text(encoding="utf-8")
    assert "function buildDetectorMask" not in frontend
    assert "function buildScanMask" not in frontend
    assert "buildFullDetectorMask" in frontend
    assert "maskedDpc" in frontend
    assert "maskedIDpc" in frontend
    assert "iDPC" in frontend
    assert "roiBufferOnly" in frontend
    assert "dpcBufferOnly" in frontend
    assert "dpcCompareReference" in frontend
    assert "warmStandardViCache" in frontend
    assert "warmCache: () => warmCacheSummary()" in frontend
    assert "suppressViTraitRecompute" in frontend
    assert "const normalizedRadiusInner" in frontend
    assert "saveChangesIfLiveComm" in frontend
    assert "requestViPreset" in frontend
    assert '"launch_warm_cache"' in frontend
    assert "virtualGpuCanvasRef" in frontend
    assert "renderPanelSlotsDirectToCanvas" in frontend
    assert "renderSlotDirectWithGpuRangeToCanvas" in frontend


def test_show4dstem_webgpu_h5_master_loader_batches_external_decodes() -> None:
    repo = Path(__file__).resolve().parents[2]
    frontend = (repo / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )
    lazy = (repo / "js" / "show4dstem" / "lazy.ts").read_text(encoding="utf-8")
    local_h5 = _webgpu_source("io/backends/webgpu/local-h5.ts")
    compute = _webgpu_source("detector/compute/webgpu/backend.ts")
    bslz4 = _webgpu_source("io/backends/webgpu/bslz4.ts")

    assert "decodeBslz4Batch" in frontend
    assert "const decodeQueue" in frontend
    assert "decodeWorker" in frontend
    assert "pipelineMode: \"fetch-parse-decode-queue\"" in frontend
    assert "embeddedBadPxJson" in frontend
    assert "hasEmbeddedBadPx" in frontend
    assert "__QT_H5_DECODE_BATCH" in frontend
    assert "__QT_H5_FETCH_WINDOW" in frontend
    assert "__QT_H5_DECODE_QUEUE" in frontend
    assert "function show4DSTEMGlobalInt" in frontend
    assert "decodeBatch," in frontend
    assert "fetchWindow," in frontend
    assert "uploadMs:" in frontend
    assert "gpuWaitMs:" in frontend
    assert "decodeCompressedMB:" in frontend
    assert "const h5MasterInfoCache = new Map" in frontend
    assert "readH5MasterInfoCached" in frontend
    assert "const masterInfo = await readH5MasterInfoCached(sourceUrl, \"master\");" in frontend
    assert "masterInfo.dataFileCount" in frontend
    assert "const initialFetchLimit = Number.isFinite(maxDataFiles)" in frontend
    assert 'compute = await loadH5Compute(h5Url, "dataset 1/1");' in frontend
    assert "loadShow4DSTEMLocalH5Master" in frontend
    assert "setShow4DSTEMLocalFiles" in frontend
    assert "collectShow4DSTEMLocalH5Files" in frontend
    assert "__QT_H5_LOCAL_WORKERS" in frontend
    assert "__QT_H5_LOCAL_GROUP" in frontend
    assert "__QT_H5_SOURCE_SCAN_ROWS" in frontend
    assert "__QT_H5_SOURCE_SCAN_COLS" in frontend
    assert "__QT_H5_SCAN_REGION" in frontend
    assert "detBin: show4DSTEMOptionalGlobalInt(\"__QT_H5_DET_BIN\", 1, 16)" in frontend
    assert "decodeDtype: (globalThis as { __QT_H5_DECODE_DTYPE?: unknown })" in frontend
    assert "const bytesPerPixel = local.mode === 2 ? 4 : local.mode === 0 ? 2 : 1;" in frontend
    assert "const httpDecodeOverride = String((globalThis as { __QT_H5_DECODE_DTYPE?: unknown })" in frontend
    assert "localFiles: true" in frontend
    assert "h5LocalFilesGranted, h5SourceAvailable, offline" in frontend
    assert "rawChecksums:" in frontend
    assert "__sh4dRawChecksums" in frontend
    assert "checksumFrames(scanIndices" in compute
    assert "const bad = this.badPx.length ? new Set(this.badPx) : null;" in compute
    assert "bad?.has(i) ? 0" in compute
    assert "if (local.badPixels.length) created.badPx = local.badPixels;" in frontend
    assert "badPixels?: number[]" in lazy
    assert 'sourceDtype?: LazySourceDtype;' in lazy
    assert 'function lazySourceDtype(value: unknown): LazySourceDtype' in lazy
    assert 'function bytesPerLazyPixel(dtype: LazySourceDtype): number' in lazy
    assert ': "uint16";' in lazy
    assert "Array.isArray(meta.badPixels)" in lazy
    assert "created.badPx = new Uint32Array(bad);" in lazy
    assert "const sourceDtype = lazySourceDtype(this.meta.sourceDtype);" in lazy
    assert "const bb = be32(ch, 8), be = bb / bytesPerLazyPixel(sourceDtype)" in lazy
    assert "const byteLength = this.detSize * bytesPerLazyPixel(sourceDtype);" in lazy
    assert "}, sourceDtype, sourceDtype);" in lazy
    assert 'sourceDtype === "uint16"' in lazy
    assert "for (const bp of this.badPx) v[bp] = 0;" in lazy
    assert "compute.detSize === detR * detC" in frontend
    assert "local.scanCount" in frontend
    assert "rawFrame:" not in frontend
    assert "BSLZ4_LOW8_ONLY" in bslz4
    assert "BSLZ4_COOP_LOW8" in bslz4
    assert "BSLZ4_FRAME_LOW8" in bslz4
    assert "BSLZ4_LOW8_U32_SHARED" in bslz4
    assert "BSLZ4_SINGLE_PARSE_LOW8" in bslz4
    assert "BSLZ4_UPLOAD_WRITEBUFFER" in bslz4
    assert "BSLZ4_UPLOAD_MAPPED" in bslz4
    assert "BSLZ4_UPLOAD_COMBINED" in bslz4
    assert "FUSED_LOW8_WGSL" in bslz4
    assert "FUSED_COOP_LOW8_WGSL" in bslz4
    assert "FUSED_FRAME_COOP_LOW8_WGSL" in bslz4
    assert "FUSED_FRAME_U32_LOW8_WGSL" in bslz4
    assert "FUSED_FRAME_SINGLEPARSE_LOW8_WGSL" in bslz4
    assert "fused-low8-experimental" in bslz4
    assert "fused-coop-low8-experimental" in bslz4
    assert "fused-frame-coop-low8-experimental" in bslz4
    assert "fused-frame-u32-low8-experimental" in bslz4
    assert "fused-frame-singleparse-low8-experimental" in bslz4
    assert "uploadViaMapped" in bslz4
    assert "stageUploadCopies" in bslz4
    assert "decodeVariant" in local_h5
    assert 'title={h5LocalSourceStatus || "Grant local HDF5 master/data files for browser WebGPU load"}' in frontend
    assert "export async function loadShow4DSTEMLocalH5Master" in local_h5
    assert 'acquisitionMode: "local-file"' in local_h5
    assert "const READ_WORKER_SOURCE" in local_h5
    assert "new Blob([READ_WORKER_SOURCE]" in local_h5
    assert "decodeBslz4ToStack({ ...vol.chunks[0], startScan" not in frontend
    assert "const mergedSpecs = vol.chunks.map((chunk) => {" in frontend
    assert "DetectorCompute.createFromBslz4Chunked(mergedSpecs" in frontend
    assert "_h5_uint8_lossless" in (
        repo / "src" / "quantem" / "widget" / "show4dstem.py"
    ).read_text(encoding="utf-8")
    assert "h5_uint8_lossless: bool = False" in (
        repo / "src" / "quantem" / "widget" / "show4dstem.py"
    ).read_text(encoding="utf-8")
    assert 'model.get("_h5_uint8_lossless")' in frontend
    assert "__BSLZ4_LOW8_ONLY = true" in frontend
    assert "const low8Only = h5Uint8Lossless ||" in frontend

    show4dstem_py = (repo / "src" / "quantem" / "widget" / "show4dstem.py").read_text(
        encoding="utf-8"
    )
    assert "def _show4dstem_h5_webgpu_tuning" in show4dstem_py
    assert 'globalThis.__QT_H5_DECODE_DTYPE ??= "{decode_dtype}";' in show4dstem_py
    assert "globalThis.__BSLZ4_PIPELINE_STAGING ??= false;" in show4dstem_py
    assert "_inject_show4dstem_h5_webgpu_tuning(" in show4dstem_py


def test_show4dstem_webgpu_h5_prefetch_is_bounded_by_master_data_file_count() -> None:
    repo = Path(__file__).resolve().parents[2]
    frontend = (repo / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )

    start = frontend.index("const prefetchVolume = async")
    stop = frontend.index("let next = 0;", start)
    prefetch = frontend[start:stop]

    assert "let fileLimit = maxPrefetchFiles;" in prefetch
    assert "const masterInfo = await readH5MasterInfoCached(h5Urls[index]" in prefetch
    assert "masterInfo?.dataFileCount" in prefetch
    assert "fileLimit = Math.min(fileLimit" in prefetch
    assert "nextFile <= Math.min(prefetchWindow, fileLimit)" in prefetch
    assert "n <= fileLimit" in prefetch
    assert "nextFile <= fileLimit" in prefetch


def test_show4dstem_webgpu_h5_initial_load_uses_visible_loading_panels() -> None:
    repo = Path(__file__).resolve().parents[2]
    frontend = (repo / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )

    assert "offlineBackendLoading" in frontend
    assert "offlineBackendStatus" in frontend
    assert "offlineBackendError" in frontend
    assert "setOfflineBackendLoading(true);" in frontend
    assert 'setOfflineBackendStatus(h5SourceAvailable ? "Loading WebGPU source" : "Loading offline 4D-STEM data");' in frontend
    assert 'data-testid="show4dstem-offline-status"' in frontend
    assert 'Show4DSTEM load failed: ${offlineStatusText}' in frontend
    assert 'data-show4dstem-panel-loading="true"' in frontend
    assert 'offlineBackendError ? "Show4DSTEM load failed" : "Loading DP"' in frontend
    assert 'offlineBackendError ? "Show4DSTEM load failed" : "Loading virtual image"' in frontend
    assert 'renderPanelLoadingOverlay("Loading FFT")' in frontend
    assert 'renderPanelLoadingOverlay(\n              offlineBackendError ? "Show4DSTEM load failed" : "Loading DP",' in frontend
    assert 'renderPanelLoadingOverlay(\n                offlineBackendError ? "Show4DSTEM load failed" : "Loading virtual image",' in frontend
    assert 'loadH5Compute(h5Urls[clamped], `dataset ${clamped + 1}/${h5Urls.length}`)' in frontend
    assert 'loadH5Compute(h5Url, "dataset 1/1")' in frontend
    assert 'setOfflineBackendError(error instanceof Error ? error.message : String(error));' in frontend
    assert "await recomputeFrame();  // initial DP at mount" in frontend
    assert "await recomputeVI();  // initial virtual image" in frontend
    assert "showStats && !dpPanelLoading" in frontend
    assert "showStats && !viPanelLoading" in frontend
    assert "showStats && !fftPanelLoading" in frontend


def test_show4dstem_h5_preload_publishes_every_dataset_progressively() -> None:
    """Every HDF5 dataset is visited even when native uint16 residency is one."""
    repo = Path(__file__).resolve().parents[2]
    frontend = (repo / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )

    assert "const h5ResidentLimit = h5UsesNativeU16" in frontend
    assert "const h5RequestedResidentLimit = show4DSTEMGlobalInt(" in frontend
    assert "? 1\n        : h5RequestedResidentLimit" in frontend
    assert "const maxPreload = Math.max(1, h5Urls.length);" in frontend
    assert "const defaultPreload = h5Urls.length;" in frontend
    assert "while (volCache.size > h5ResidentLimit)" in frontend
    assert "if (getVol && !volIsResident(idx)) continue;" in frontend


def test_show4dstem_h5_multiple_starts_with_loading_compare_state() -> None:

    widget = Show4DSTEM(
        np.zeros((1, 1, 1, 1), dtype=np.uint8),
        h5_urls=[
            "dataset_0_master.h5",
            "dataset_1_master.h5",
            "dataset_2_master.h5",
        ],
        scan_shape=(4, 4),
        detector_shape=(8, 8),
        backend="webgpu",
        view_mode="multiple",
        compare_max_panels=3,
        precompute_virtual_images=False,
        verbose=False,
    )
    try:
        assert widget.compare_panel_count == 0
        assert widget.compare_panel_indices == []
        assert widget.compare_virtual_image_bytes == b""
        assert widget.compare_status == "Loading 3/3 browser WebGPU panels"
    finally:
        widget.close()


def test_show4dstem_frontend_virtual_image_bytes_use_react_setter() -> None:
    repo = Path(__file__).resolve().parents[2]
    frontend = (repo / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )

    assert (
        'const [virtualImageBytes, setVirtualImageBytes] = '
        'useModelState<DataView>("virtual_image_bytes");'
    ) in frontend
    assert (
        "const [frontendVirtualImageBytes, setFrontendVirtualImageBytes] = "
        "React.useState<DataView | null>(null);"
    ) in frontend
    assert "const publishVirtualImageBytes = React.useCallback" in frontend
    assert "setFrontendVirtualImageBytes(bytes);" in frontend
    assert "setVirtualImageBytes(bytes);" in frontend
    assert 'model.set("virtual_image_bytes"' not in frontend
    assert "publishVirtualImageBytes(new DataView(vi.buffer));" in frontend


def test_show4dstem_multiple_detector_drag_uses_live_gpu_compare_slots() -> None:
    repo = Path(__file__).resolve().parents[2]
    frontend = (repo / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )

    assert "requestCompareViLiveRef" in frontend
    assert "requestCompareViLive();" in frontend
    assert "requestCompareViLiveRef.current = () => {" in frontend
    live_drag = frontend.split("requestCompareViLiveRef.current = () => {", 1)[1].split(
        "requestViFinalizeRef.current",
        1,
    )[0]
    assert "await recomputeVI();" in live_drag
    assert "await recomputeCompareVI();" in live_drag
    assert live_drag.index("await recomputeVI();") < live_drag.index(
        "await recomputeCompareVI();"
    )
    assert 'type DpcGpuSource = "DPC_row" | "DPC_col" | "iDPC";' in frontend
    assert "gpuLoaded: Boolean(gpuSlots?.has(frame) && gpuEngine)" in frontend
    assert 'scaleMode === "log"' in frontend
    assert "entry.panel !== undefined || entry.gpuLoaded" in frontend
    assert "const loaded = panel !== undefined || gpuLoaded;" in frontend
    assert "onChangeCommitted={finishDpRoiInteraction}" in frontend
    assert "__sh4dLiveViStats" in frontend
    assert "gpuOnlyHotPath: true" in frontend
    assert 'publishLiveCompareViStats("paint"' in frontend
    assert "if (gpuEngine) gpuEngine.uploadLUT(colormap, lut);" in frontend
    assert "renderSlotDirectWithGpuRangeToCanvas" in frontend
    assert "renderSlotGpuRangeToOffscreen" not in frontend
    assert "let comparePersistentStack: Float32Array | null = null;" in frontend
    assert "if (getVol && !volIsResident(idx)) continue;" in frontend


def test_show4dstem_webgpu_fits_bf_disk_in_browser_for_h5_sources() -> None:
    repo = Path(__file__).resolve().parents[2]
    frontend = (repo / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )

    assert "const fitBfDiskFromMeanDp = async (): Promise<void> => {" in frontend
    assert "const ratioGuess = Math.min(detR, detC) * 0.125;" in frontend
    assert "const scanMask = new Uint32Array(scanCount);" in frontend
    assert "const dp = await compute.reduceFrames(scanMask, true);" in frontend
    assert "model.set(\"bf_radius\", edge);" in frontend
    assert "await fitBfDiskFromMeanDp().catch((error) => {" in frontend
    initial_load = frontend.split(
        "await recomputeVI();  // initial virtual image, no interaction needed",
        1,
    )[1].split(
        "requestAnimationFrame(() => { if (!disposed) { void recomputeVI();",
        1,
    )[0]
    assert initial_load.index("await fitBfDiskFromMeanDp().catch((error) => {") < (
        initial_load.index("scheduleWarmStandardViCache();")
    )
    assert "if (Math.abs(current - ratioGuess) > 0.51) return;" in frontend


def test_show4dstem_h5_export_embeds_local_bad_pixel_mask(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")

    from quantem.widget.show4dstem import _h5_bad_pixel_json_for_export

    master = tmp_path / "sample_master.h5"
    with h5py.File(master, "w") as f:
        f.create_dataset(
            "entry/instrument/detector/detectorSpecific/pixel_mask",
            data=[[0, 1], [0, 2]],
        )

    payload = _h5_bad_pixel_json_for_export("sample_master.h5", tmp_path)

    assert json.loads(payload or "") == [1, 3]
