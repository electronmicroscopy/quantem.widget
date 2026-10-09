/** A read-only display view of source-owned exact counts. The consumer must not
 * destroy or write the buffer. The owner keeps it alive until submitted work ends.
 * Display values are f32(count) / divisor, including before log/range transforms.
 */
export type Uint32ImageView = Readonly<{
  device: GPUDevice;
  buffer: GPUBuffer;
  byteOffset: number;
  count: number;
  divisor: number;
}>;

/** Validate a borrowed view before creating any bindings or issuing GPU work. */
export function validateUint32ImageView(view: Uint32ImageView, device: GPUDevice, count: number): void {
  if (view.device !== device) throw new Error('Image counts and renderer must use the same GPU device.');
  if (!Number.isSafeInteger(view.count) || view.count !== count || count < 1
    || !Number.isSafeInteger(view.byteOffset) || view.byteOffset < 0
    || view.byteOffset % device.limits.minStorageBufferOffsetAlignment !== 0
    || !Number.isSafeInteger(view.byteOffset + count * 4)
    || view.byteOffset + count * 4 > view.buffer.size
    || count * 4 > device.limits.maxStorageBufferBindingSize
    || !(view.buffer.usage & GPUBufferUsage.STORAGE)) {
    throw new Error('Supply an aligned native uint32 image slice within a storage buffer.');
  }
  if (!Number.isFinite(view.divisor) || !Number.isFinite(Math.fround(view.divisor))
    || Math.fround(view.divisor) <= 0) throw new Error('Image mean divisor must be positive and finite in float32.');
}
