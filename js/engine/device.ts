/// <reference types="@webgpu/types" />
// Shared WebGPU device for the 4D-STEM compute engine. ONE source of the GPU
// device + adapter limits, imported by every consumer (Show4DSTEM widget, the
// offline browser app, FFT). Keeps no UI deps so the engine is framework-agnostic.

let gpuDevice: GPUDevice | null = null;
let devicePromise: Promise<GPUDevice | null> | null = null;
let gpuInfo = "GPU";
const lostCallbacks: Array<() => void> = [];

// Register a reset to run when the GPU device is lost (process crash, tab suspend).
// Consumers (e.g. the FFT cache) use this to drop their device-bound state.
export function onGPULost(cb: () => void): void { lostCallbacks.push(cb); }

// Memoize the in-flight requestDevice so concurrent first callers share ONE device.
// Without this guard, decode + colormap + FFT + render all call getGPUDevice() before
// gpuDevice is assigned, each runs requestDevice(), and pipelines/bind-groups built on
// the loser device get submitted on the winner -> "BindGroupLayout is associated with
// [Device], cannot be used with [Device]" -> device lost -> tab GPU process crash.
export function getGPUDevice(): Promise<GPUDevice | null> {
  if (gpuDevice) return Promise.resolve(gpuDevice);
  if (!devicePromise) devicePromise = createGPUDevice();
  return devicePromise;
}

async function createGPUDevice(): Promise<GPUDevice | null> {
  if (!navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return null;
    try {
      // Newer Chrome exposes the sync `adapter.info`; older builds used the async
      // requestAdapterInfo(). Prefer the sync one, fall back to async.
      // @ts-ignore - info / requestAdapterInfo are not in all type definitions
      const info = adapter.info || (await adapter.requestAdapterInfo?.());
      if (info) {
        gpuInfo = info.description || `${info.vendor || ""} ${info.architecture || ""} ${info.device || ""}`.trim() || "Generic WebGPU Adapter";
      }
    } catch (_e) { /* adapter info not available */ }
    // Raise device limits to the adapter max. Defaults are conservative
    // (maxStorageBufferBindingSize 128 MB, maxTextureDimension2D 8192); without
    // this, buffers > 128 MB silently invalidate bind groups and wide panels fail.
    const requiredLimits: Record<string, number> = {};
    for (const key of ["maxBufferSize", "maxStorageBufferBindingSize", "maxTextureDimension2D"] as const) {
      const v = adapter.limits[key] || 0;
      if (v > 0) requiredLimits[key] = v;
    }
    const feats: GPUFeatureName[] = [];
    if (adapter.features.has("timestamp-query")) feats.push("timestamp-query");   // for kernel profiling
    if (adapter.features.has("subgroups")) feats.push("subgroups" as GPUFeatureName);   // warp reduction in maskedSum/CoM
    gpuDevice = await adapter.requestDevice({ requiredFeatures: feats, requiredLimits });
    // On loss, drop BOTH the device and the memoized promise so the next getGPUDevice()
    // rebuilds a fresh device (and consumers re-create their device-bound pipelines via onGPULost).
    gpuDevice.lost.then(() => { gpuDevice = null; devicePromise = null; lostCallbacks.forEach((cb) => cb()); });
    return gpuDevice;
  } catch { devicePromise = null; return null; }
}

export function getGPUInfo(): string { return gpuInfo; }

export function isSoftwareGPUAdapter(): boolean {
  return /swiftshader|llvmpipe|software|subzero/i.test(gpuInfo);
}
