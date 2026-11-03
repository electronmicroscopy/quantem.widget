"""The WebGPU display FFT equals NumPy's FFT on the hardware adapter, forward and inverse.

Every FFT panel, the frequency filter and Show3D's sub-pixel alignment run
``WebGPUFFT`` from ``js/display/fft.ts``. The module is bundled with esbuild
into a page that transforms non-square power-of-two images (row and column
passes of different lengths) and transforms the spectrum back; the spectrum
must match ``numpy.fft.fft2`` to float32 precision and the round trip must
return the image. The frequency filter on WebGPU must equal its CPU path on a
non-power-of-two image (both pad to powers of two and mask the padded grid).
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.browser

REPO = Path(__file__).resolve().parents[2]
SHAPES = [(32, 64), (128, 32), (64, 256)]
ENTRY = """
import { WebGPUFFT } from "./display/fft";
import { requireHardwareGPUDevice } from "./display/device";
import { applyFrequencyFilterBrowser, applyFrequencyFilterCPU, getFrequencyFilterBackend } from "./display/frequencyFilter";
window.transform = async (images) => {
  const fft = new WebGPUFFT(await requireHardwareGPUDevice("FFT parity"));
  const results = [];
  for (const { height, width, values } of images) {
    const forward = await fft.fft2D(Float32Array.from(values), new Float32Array(values.length), width, height);
    const inverse = await fft.fft2D(forward.real, forward.imag, width, height, true);
    results.push({ real: Array.from(forward.real), imag: Array.from(forward.imag), back: Array.from(inverse.real), backImag: Array.from(inverse.imag) });
  }
  return results;
};
window.filter = async ({ height, width, values, options }) => {
  const gpu = await applyFrequencyFilterBrowser(Float32Array.from(values), width, height, options);
  return { backend: getFrequencyFilterBackend(), gpu: Array.from(gpu), cpu: Array.from(applyFrequencyFilterCPU(Float32Array.from(values), width, height, options)) };
};
"""


@pytest.fixture(scope="module")
def fft_page(browser_out) -> str:
    bundle = subprocess.run(
        ["npx", "esbuild", "--bundle", "--format=iife", "--loader=ts", f"--sourcefile={REPO / 'js' / 'entry.ts'}"],
        input=ENTRY, capture_output=True, text=True, cwd=REPO / "js", check=True,
    ).stdout
    (browser_out / "display_fft.html").write_text(f'<!doctype html><meta charset="utf-8"><script>{bundle}</script>')
    return "display_fft.html"


def test_webgpu_fft_matches_numpy_on_non_square_images(fft_page, serve, open_page, hardware_adapter, browser_out):
    rng = np.random.default_rng(0)
    images = [rng.standard_normal(shape).astype(np.float32) * 100 for shape in SHAPES]
    page = open_page(f"{serve(browser_out)}/{fft_page}", webgpu=True)
    hardware_adapter(page.adapter())
    payload = [{"height": image.shape[0], "width": image.shape[1], "values": image.ravel().tolist()} for image in images]
    results = page.evaluate("images => window.transform(images)", payload)
    errors = {}
    for image, result in zip(images, results):
        spectrum = np.fft.fft2(image.astype(np.float64))
        found = (np.array(result["real"]) + 1j * np.array(result["imag"])).reshape(image.shape)
        # float32 butterflies: error grows with log2(size) times the largest coefficient's ulp
        scale = np.abs(spectrum).max()
        errors[str(image.shape)] = (float(np.abs(found - spectrum).max() / scale),
                                    float(np.abs(np.array(result["back"]).reshape(image.shape) - image).max() / np.abs(image).max()),
                                    float(np.abs(np.array(result["backImag"])).max() / np.abs(image).max()))
    (browser_out / "display_fft.json").write_text(json.dumps(errors, indent=1))
    for forward, back, back_imag in errors.values():
        assert forward < 2e-6 and back < 2e-6 and back_imag < 2e-6, errors
    assert page.errors == [], page.errors


@pytest.mark.parametrize("options", [{"mode": "highpass", "cutoff": 0.15}, {"mode": "bandpass", "center": 0.35, "width": 0.2}], ids=["highpass", "bandpass"])
def test_frequency_filter_webgpu_equals_cpu_on_a_non_power_of_two_image(options, fft_page, serve, open_page, hardware_adapter, browser_out):
    image = (50 + 30 * np.random.default_rng(1).standard_normal((37, 53))).astype(np.float32)
    page = open_page(f"{serve(browser_out)}/{fft_page}", webgpu=True)
    hardware_adapter(page.adapter())
    result = page.evaluate("job => window.filter(job)", {"height": 37, "width": 53, "values": image.ravel().tolist(), "options": options})
    assert result["backend"] == "WebGPU"
    # float32 transforms on both sides; the mask is float32 on the GPU and float64 on the CPU
    assert np.abs(np.array(result["gpu"]) - np.array(result["cpu"])).max() / np.abs(image).max() < 2e-6
    assert page.errors == [], page.errors
