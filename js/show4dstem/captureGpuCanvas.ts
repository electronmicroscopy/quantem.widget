/** Capture a freshly rendered GPU canvas only for an explicit copy/export. */
export async function captureGpuCanvas(
  device: GPUDevice,
  context: GPUCanvasContext,
  render: () => void,
): Promise<HTMLCanvasElement> {
  // Submit the render and texture copy in the same task, before presentation
  // recycles the canvas texture. No readback belongs in the pointer loop.
  render();
  const texture = context.getCurrentTexture();
  const width = texture.width, height = texture.height;
  const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
  const read = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({texture}, {buffer: read, bytesPerRow}, {width, height});
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const source = new Uint8Array(read.getMappedRange());
    const pixels = new Uint8ClampedArray(width * height * 4);
    const bgra = texture.format.startsWith("bgra");
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        const src = row * bytesPerRow + col * 4, dst = (row * width + col) * 4;
        pixels[dst] = source[src + (bgra ? 2 : 0)];
        pixels[dst + 1] = source[src + 1];
        pixels[dst + 2] = source[src + (bgra ? 0 : 2)];
        pixels[dst + 3] = source[src + 3];
      }
    }
    const canvas = document.createElement("canvas");
    canvas.width = width; canvas.height = height;
    // Keep this export surface in CPU memory: accelerated 2D canvas readback
    // can return black pixels on Vulkan even when the captured RGBA is valid.
    canvas.getContext("2d", {willReadFrequently: true})!.putImageData(new ImageData(pixels, width, height), 0, 0);
    return canvas;
  } finally {
    if (read.mapState === "mapped") read.unmap();
    read.destroy();
  }
}
