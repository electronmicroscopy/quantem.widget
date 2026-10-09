"""The WebGPU display normalization picks the same colormap level as the CPU path, on the hardware adapter.

A value exactly on a level edge lands in the neighbouring level when the GPU's
float32 division is one ULP off (WGSL allows 2.5 ULP; Apple Metal uses it).
The shader's display_normalize therefore divides through an integer long
division. This test compiles the shader source of ``js/display/colormaps.ts``
on the real adapter, normalizes every level edge of several windows (and one
float32 step either side), and requires the float32 bits to equal NumPy's
correctly rounded ``(value - low) / (high - low)``, the CPU path's arithmetic
(windows wider than float32 holds are halved first, exactly, on both paths).
On an NVIDIA RTX PRO 6000 the previous shader (a plain ``/``) differed in 1435
of these 6144 values.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.browser

COLORMAPS = Path(__file__).resolve().parents[2] / "js" / "display" / "colormaps.ts"
WINDOWS = [(0, 6), (0, 1), (-1, 1), (0, 255), (3, 7), (1e-3, 2e-3), (-5000, 65535), (0.1, 0.7),
           (-3e38, 3e38), (-3.4e38, 2e38), (-1e38, 3.3e38)]
PAGE = """<!doctype html><meta charset="utf-8"><script>
window.result = (async () => {
  const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
  const device = await adapter.requestDevice();
  const inputs = new Float32Array(INPUTS);
  const code = SHADER + `
@group(0) @binding(0) var<storage, read> inputs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= arrayLength(&outputs)) { return; }
  let input = inputs[gid.x];
  outputs[gid.x] = display_normalize(input.x, input.y, input.z);
}`;
  const count = inputs.length / 4;
  const input = device.createBuffer({size: inputs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  const output = device.createBuffer({size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const read = device.createBuffer({size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(input, 0, inputs);
  const pipeline = device.createComputePipeline({layout: 'auto', compute: {module: device.createShaderModule({code}), entryPoint: 'main'}});
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [
    {binding: 0, resource: {buffer: input}}, {binding: 1, resource: {buffer: output}}]}));
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, read, 0, count * 4);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  return Array.from(new Uint32Array(read.getMappedRange().slice(0)));
})();
</script>"""


def shader_source() -> str:
    """ROUNDED_QUOTIENT_WGSL and DISPLAY_NORMALIZE_WGSL as the widget concatenates them."""
    source = COLORMAPS.read_text()
    templates = [re.search(rf"const {name} = [^`]*`([^`]*)`", source).group(1)
                 for name in ("ROUNDED_QUOTIENT_WGSL", "DISPLAY_NORMALIZE_WGSL")]
    return "".join(templates)


def edge_inputs() -> np.ndarray:
    """(value, low, high) at every level edge of each window, and one float32 step either side."""
    rows = []
    for vmin, vmax in WINDOWS:
        low, high = np.float32(vmin), np.float32(vmax)
        for edge in range(256):
            exact = np.float32(np.float64(low) + edge * (np.float64(high) - np.float64(low)) / 255)
            for value in (np.nextafter(exact, -np.inf), exact, np.nextafter(exact, np.inf)):
                rows.append((value, low, high, 0))
    return np.asarray(rows, dtype=np.float32)


def test_webgpu_level_edges_match_cpu_normalization(open_page, serve, hardware_adapter, browser_out):
    inputs = edge_inputs()
    folder = browser_out / "level_edges"
    folder.mkdir(exist_ok=True)
    page_source = PAGE.replace("INPUTS", json.dumps(inputs.ravel().tolist())).replace("SHADER", json.dumps(shader_source()))
    (folder / "index.html").write_text(page_source)
    page = open_page(f"{serve(folder)}/index.html", webgpu=True)
    hardware_adapter(page.adapter())
    gpu = np.asarray(page.evaluate("window.result"), dtype=np.uint32).view(np.float32)
    assert page.errors == [], page.errors
    value, low, high = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    # A window whose float32 span overflows is halved first, exactly, on both paths.
    with np.errstate(over="ignore"):
        scale = np.where(np.isfinite(high - low), np.float32(1), np.float32(0.5))
    # Adding +0 turns a clipped -0 into +0, as Math.max(0, -0) does on the CPU path.
    cpu = np.clip((value * scale - low * scale) / (high * scale - low * scale), np.float32(0), np.float32(1)) + np.float32(0)
    # WGSL may flush a subnormal to zero; such a value is level 0 on both paths.
    normal = cpu >= np.finfo(np.float32).tiny
    differ = np.flatnonzero((gpu.view(np.uint32) != cpu.view(np.uint32)) & (normal | (gpu != 0)))
    assert differ.size == 0, [(*inputs[i, :3].tolist(), float(gpu[i]), float(cpu[i])) for i in differ[:5]]
    levels = np.minimum(np.trunc(np.float32(gpu * np.float32(255))), 255)
    assert np.array_equal(levels, np.minimum(np.trunc(np.float32(cpu * np.float32(255))), 255))
