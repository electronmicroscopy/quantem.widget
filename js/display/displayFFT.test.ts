// Every panel of a widget asks for the page's display FFT at once when its FFT
// is switched on. The callers must share one WebGPUFFT, compiled once, instead
// of each building its own pipelines while the device request is in flight.
import { expect, it, vi } from "vitest";

const device = vi.hoisted(() => ({
  limits: { minUniformBufferOffsetAlignment: 256 },
  createShaderModule: vi.fn(() => ({})),
  createComputePipeline: vi.fn(() => ({})),
}));
vi.mock("./device", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./device")>()),
  getHardwareGPUDevice: async () => device,
  requireHardwareGPUDevice: async () => device,
}));

const { getDisplayFFT } = await import("./fft");

it("gives concurrent first callers one WebGPU FFT, compiled once", async () => {
  const ffts = await Promise.all([getDisplayFFT(), getDisplayFFT(), getDisplayFFT(), getDisplayFFT()]);
  expect(ffts[0].path).toBe("WebGPU");
  for (const fft of ffts) expect(fft).toBe(ffts[0]);
  expect(device.createShaderModule).toHaveBeenCalledOnce();
  expect(device.createComputePipeline).toHaveBeenCalledTimes(5);
});
