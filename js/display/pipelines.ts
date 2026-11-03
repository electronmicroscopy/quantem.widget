// Pipeline builders shared by the display modules: every display shader is a
// compute pass with an automatic layout, or a fullscreen-triangle vs/fs pair.

/** The compute pipeline of `code`'s `entryPoint`, with an automatic bind group layout. */
export function computePipeline(device: GPUDevice, code: string, entryPoint = "main"): GPUComputePipeline {
  return device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint } });
}

/** The fullscreen-triangle render pipeline of `code`'s `vs` and `fs`, drawing into `format`. */
export function renderPipeline(device: GPUDevice, code: string, format: GPUTextureFormat): GPURenderPipeline {
  const module = device.createShaderModule({ code });
  return device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
}
