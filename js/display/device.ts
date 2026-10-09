/// <reference types="@webgpu/types" />
// The one WebGPU device of a widget bundle. Display kernels import it directly;
// the build points the synced quantem.gpu science kernels (device/webgpu) here too,
// so buffers borrowed between detector compute and display share one device.

let gpuDevice: GPUDevice | null = null;
let devicePromise: Promise<GPUDevice | null> | null = null;
let gpuInfo = "GPU";
let lostMessage: string | null = null;
const lostCallbacks: Array<() => void> = [];

// Register a reset to run when the GPU device is lost (process crash, tab suspend).
// Consumers (e.g. the FFT cache) use this to drop their device-bound state.
// Returns the function that unregisters it, for components that unmount.
export function onGPULost(callback: () => void): () => void {
  lostCallbacks.push(callback);
  return () => {
    const index = lostCallbacks.indexOf(callback);
    if (index >= 0) lostCallbacks.splice(index, 1);
  };
}

/**
 * Why the page stopped using WebGPU, or null while it may. A device the page
 * did not destroy is lost when the GPU process crashes or its watchdog resets
 * a hung queue. A new device would rerun the same shaders on the same adapter
 * and could hang it again, so display math stays in JavaScript until the page
 * reloads.
 */
export function webGPULostReason(): string | null { return lostMessage; }

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
  if (typeof navigator === "undefined" || !navigator.gpu || lostMessage !== null) return null;
  try {
    // powerPreference selects the discrete GPU of a dual-GPU Mac. Chromium on Windows ignores it and
    // logs a console warning on every request (crbug.com/369219127), so it is left out there.
    const platform = (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData?.platform;
    const adapter = await navigator.gpu.requestAdapter(platform === "Windows" ? {} : { powerPreference: "high-performance" });
    if (!adapter) return null;
    try {
      // Newer Chrome exposes the sync `adapter.info`; older builds used the async
      // requestAdapterInfo(). Prefer the sync one, fall back to async.
      // @ts-ignore - info / requestAdapterInfo are not in all type definitions
      const info = adapter.info || (await adapter.requestAdapterInfo?.());
      if (info) {
        gpuInfo = info.description || `${info.vendor || ""} ${info.architecture || ""} ${info.device || ""}`.trim() || "Generic WebGPU Adapter";
      }
    } catch { /* adapter info not available */ }
    // Raise device limits to the adapter max. Defaults are conservative
    // (maxStorageBufferBindingSize 128 MB, maxTextureDimension2D 8192); without
    // this, buffers > 128 MB silently invalidate bind groups and wide panels fail.
    const requiredLimits: Record<string, number> = {};
    for (const key of [
      "maxBufferSize",
      "maxStorageBufferBindingSize",
      "maxTextureDimension2D",
      "maxComputeInvocationsPerWorkgroup",
      "maxComputeWorkgroupSizeX",
    ] as const) {
      const limit = adapter.limits[key] || 0;
      if (limit > 0) requiredLimits[key] = limit;
    }
    const features: GPUFeatureName[] = [];
    if (adapter.features.has("timestamp-query")) features.push("timestamp-query");   // for kernel profiling
    if (adapter.features.has("subgroups")) features.push("subgroups" as GPUFeatureName);   // warp reduction in maskedSum/CoM
    const device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits });
    gpuDevice = device;
    // On loss, drop BOTH the device and the memoized promise, then let consumers drop
    // their device-bound state. Only a device the page destroyed itself may be rebuilt.
    device.lost.then((info) => {
      if (gpuDevice === device) gpuDevice = null;
      devicePromise = null;
      if (info.reason !== "destroyed" && lostMessage === null) {
        lostMessage = info.message || "the browser reported no reason";
        console.warn(
          `quantem.widget: the WebGPU device was lost (${lostMessage}). ` +
          "Widgets on this page now draw with JavaScript; reload the page to try WebGPU again.",
        );
      }
      [...lostCallbacks].forEach((callback) => callback());
    });
    return device;
  } catch { devicePromise = null; return null; }
}

/** Adapter description for status chips, "GPU" until the device request settles. */
export function getGPUInfo(): string { return gpuInfo; }

/** Which path draws a widget's pixels: hardware WebGPU, or the Canvas2D/JS reference. */
export type RenderPath = "WebGPU" | "CPU";

/**
 * The hardware device, or null when the browser has no WebGPU or only a
 * software adapter. Display code then runs the JS reference math, which gives
 * the same pixels; a software adapter would be slower than plain JS.
 */
export async function getHardwareGPUDevice(): Promise<GPUDevice | null> {
  const device = await getGPUDevice();
  return device && !isSoftwareGPUAdapter() ? device : null;
}

/** Resolve the render path once the device request settles. */
export async function getRenderPath(): Promise<RenderPath> {
  return (await getHardwareGPUDevice()) ? "WebGPU" : "CPU";
}

/** True for CPU-emulated adapters (SwiftShader, llvmpipe), which run slower than the JS reference path. */
export function isSoftwareGPUAdapter(): boolean {
  return /swiftshader|llvmpipe|software|subzero/i.test(gpuInfo);
}

/** Error raised when browser scientific compute cannot run on hardware WebGPU. */
class WebGPUUnavailableError extends Error {
  constructor(operation: string, detail?: string) {
    super(
      `${operation} requires a hardware WebGPU adapter. ` +
      `${detail || "Enable WebGPU in a supported browser and reload the widget."}`,
    );
    this.name = "WebGPUUnavailableError";
  }
}

/** Resolve the shared hardware device or fail without a scientific CPU fallback. */
export async function requireHardwareGPUDevice(operation: string): Promise<GPUDevice> {
  const device = await getGPUDevice();
  if (!device && lostMessage !== null) {
    throw new WebGPUUnavailableError(operation, `The WebGPU device was lost (${lostMessage}); reload the page to try WebGPU again.`);
  }
  if (!device) throw new WebGPUUnavailableError(operation);
  if (isSoftwareGPUAdapter()) {
    throw new WebGPUUnavailableError(
      operation,
      `The selected adapter (${getGPUInfo()}) is software-rendered; choose a browser/GPU with hardware WebGPU.`,
    );
  }
  return device;
}
