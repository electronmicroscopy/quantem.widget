import type { Uint32ImageView } from "../display/borrowedImage";

/**
 * The exact count under the pointer of a GPU-resident virtual image.
 *
 * A compare panel of a .qem source lives only on the GPU; copying its whole
 * image to read one pixel on every pointer move would stall the hover. Each
 * request copies the four bytes of one count. A request made while a read is
 * in flight replaces any queued one, so only the latest position is read, and
 * a read is published only if no newer request came and `isCurrent()` still
 * holds (the image was not recomputed meanwhile).
 */
export function createResidentHoverReader(onError: (error: unknown) => void = console.warn) {
  type Request = { token: number; view: Uint32ImageView; pixel: number; isCurrent: () => boolean; publish: (count: number) => void };
  let token = 0;
  let pending: Request | null = null;
  let reading = false;
  let closed = false;
  let readback: GPUBuffer | null = null;
  let readbackDevice: GPUDevice | null = null;
  const drain = async () => {
    reading = true;
    while (pending && !closed) {
      const request = pending;
      pending = null;
      if (!request.isCurrent()) continue;
      const { view, pixel } = request;
      if (readbackDevice !== view.device) {
        readback?.destroy();
        readbackDevice = view.device;
        readback = view.device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      }
      const target = readback!;
      const encoder = view.device.createCommandEncoder();
      encoder.copyBufferToBuffer(view.buffer, view.byteOffset + pixel * 4, target, 0, 4);
      view.device.queue.submit([encoder.finish()]);
      await target.mapAsync(GPUMapMode.READ);
      const count = new Uint32Array(target.getMappedRange())[0];
      target.unmap();
      if (!closed && request.token === token && request.isCurrent()) request.publish(count);
    }
    reading = false;
    if (closed) readback?.destroy();
  };
  return {
    /** Read `pixel` (row * cols + col) of `view`; `publish` receives its count. */
    request(view: Uint32ImageView, pixel: number, isCurrent: () => boolean, publish: (count: number) => void) {
      pending = { token: ++token, view, pixel, isCurrent, publish };
      if (!reading) void drain().catch(error => { reading = false; if (!closed) onError(error); });
    },
    /** Drop any queued or in-flight read: the pointer left or the image changed. */
    clear() { token++; pending = null; },
    dispose() { closed = true; token++; pending = null; if (!reading) readback?.destroy(); },
  };
}
