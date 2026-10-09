"""Show4DSTEM in a live Jupyter kernel, driven like a user with WebGPU on and off.

A private JupyterLab server runs one notebook per mode on the public gold
4D-STEM dataset. The same clicks and drags run in both: three scan positions,
move and resize the detector, BF and ADF presets, a colormap change and a
contrast drag. After each step the diffraction, virtual-image and FFT panels
must draw identical pixels in both modes, the virtual image must equal the
masked detector sum computed in the kernel, and the badge must name the path.

A third notebook holds a 16 x 16 x 16 x 16 synthetic with large pixels: the
scan ROI is resized from its rim and must grow by the distance dragged, and
profile points clicked on pixel centers must be those pixels' indices (the
convention the line sampler reads) and be drawn on those centers.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import nbformat
import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.browser

DP = '[data-quantem-scientific-output="show4dstem-diffraction-pattern"]'
VI = '[data-quantem-scientific-output="show4dstem-virtual-image"]'
FFT = '[data-quantem-scientific-output="show4dstem-fft"]'
REPORT = r"""
import json as _json, numpy as _np
_vi = viewer.virtual_image()
_mask = viewer._current_detector_mask().astype(_np.float64)
_points = [(0, 0), (viewer.shape_rows // 2, viewer.shape_cols // 3), (viewer.shape_rows - 1, viewer.shape_cols - 1)]
print(_json.dumps({
    "pos": [viewer.pos_row, viewer.pos_col],
    "roi": [viewer.roi_mode, viewer.roi_center_row, viewer.roi_center_col, viewer.roi_radius, viewer.roi_radius_inner],
    "vi_colormap": viewer.vi_colormap, "vi_vmin_pct": viewer.vi_vmin_pct,
    "vi_at": [float(_vi[row, col]) for row, col in _points],
    "masked_sum_at": [float((viewer.pattern(row, col).astype(_np.float64) * _mask).sum()) for row, col in _points],
}))
"""


@pytest.fixture(scope="module")
def jupyter(tmp_path_factory):
    """A private JupyterLab with a kernel running this interpreter and its sys.path."""
    root = tmp_path_factory.mktemp("live")
    kernel_dir = root / "jupyter" / "kernels" / "quantem-browser-test"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "quantem-browser-test", "language": "python",
    }))
    source = ("from quantem.widget import Show4DSTEM\nfrom quantem.widget.datasets import show4dstem_gold\n\n"
              "viewer = Show4DSTEM(show4dstem_gold(size=\"medium\"), show_fft=True)\nviewer")
    synthetic = ("import numpy as np\nfrom quantem.widget import Show4DSTEM\n\n"
                 "rows, cols = np.mgrid[0:16, 0:16]\ndata = np.full((16, 16, 16, 16), 10, np.uint16)\n"
                 "data[:, :, 5, 9] = 1000\ndata += (rows * 16 + cols)[:, :, None, None].astype(np.uint16)\n"
                 "viewer = Show4DSTEM(data)\nviewer")
    for name, cell in (("show4dstem_webgpu", source), ("show4dstem_cpu", source), ("show4dstem_pixels", synthetic)):
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(cell)])
        notebook.metadata["kernelspec"] = {"name": "quantem-browser-test", "display_name": "quantem-browser-test", "language": "python"}
        nbformat.write(notebook, root / f"{name}.ipynb")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    runtime = root / "runtime"
    # An empty config dir keeps the user's Jupyter config (save hooks, extensions) out of the run.
    env = {**os.environ, "JUPYTER_PATH": str(root / "jupyter"), "JUPYTER_RUNTIME_DIR": str(runtime),
           "JUPYTER_CONFIG_DIR": str(root / "config")}
    server = subprocess.Popen(
        [sys.executable, "-m", "jupyterlab", "--no-browser", f"--port={port}", "--ip=127.0.0.1",
         "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.disable_check_xsrf=True",
         f"--ServerApp.root_dir={root}", "--ServerApp.port_retries=0"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{url}/api/status")
            break
        except OSError:
            time.sleep(0.5)
    yield {"url": url, "runtime": runtime}
    for session in json.load(urllib.request.urlopen(f"{url}/api/sessions")):
        urllib.request.urlopen(urllib.request.Request(f"{url}/api/sessions/{session['id']}", method="DELETE"))
    server.terminate()
    server.wait(timeout=30)


def run_in_kernel(jupyter, notebook: str, code: str) -> str:
    """Run ``code`` in the notebook's kernel and return what it printed."""
    from jupyter_client import BlockingKernelClient

    sessions = json.load(urllib.request.urlopen(f"{jupyter['url']}/api/sessions"))
    kernel_id = next(session["kernel"]["id"] for session in sessions if session["path"].endswith(notebook))
    client = BlockingKernelClient(connection_file=str(jupyter["runtime"] / f"kernel-{kernel_id}.json"))
    client.load_connection_file()
    client.start_channels()
    chunks = []
    client.execute_interactive(code, timeout=120, output_hook=lambda message: chunks.append(
        message["content"].get("text", "")) if message["msg_type"] == "stream" else None)
    client.stop_channels()
    return "".join(chunks).strip()


def kernel_report(jupyter, notebook: str) -> dict:
    return json.loads(run_in_kernel(jupyter, notebook, REPORT).splitlines()[-1])


def detector_on_screen(page, out: Path):
    """Center and radius of the drawn detector, from its green outline (the view may be zoomed)."""
    locator = page.locator(DP).first
    locator.scroll_into_view_if_needed()
    box = locator.bounding_box()
    page.screenshot(path=str(out), clip=box)
    rgb = np.asarray(Image.open(out).convert("RGB"), dtype=np.int16)
    rows, cols = np.nonzero((rgb[..., 1] > rgb[..., 0] + 60) & (rgb[..., 1] > rgb[..., 2] + 60) & (rgb[..., 0] < 90))
    center_row, center_col = rows.mean(), cols.mean()
    return box["x"] + center_col, box["y"] + center_row, np.percentile(np.hypot(rows - center_row, cols - center_col), 90)


def drag(page, start, end):
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, steps=5)
    page.mouse.move(*end, steps=5)
    page.mouse.up()


def run_notebook(open_page, jupyter, notebook: str, workspace: str, *, webgpu: bool, out: Path):
    """Open the notebook in JupyterLab, run its cell and wait for the viewer."""
    page = open_page(f"{jupyter['url']}/lab/workspaces/{workspace}/tree/{notebook}", webgpu=webgpu)
    page.wait_for_selector(".jp-Notebook .jp-Cell", timeout=120_000)
    page.wait_for_timeout(3000)
    page.locator(".jp-Cell .cm-content").first.click()
    page.keyboard.press("Shift+Enter")
    page.mouse.move(5, 300)
    page.screenshot(path=str(out / f"live_{workspace}_running.png"))
    page.wait_for_selector(DP, timeout=300_000)
    page.wait_for_timeout(6000)
    return page


def drive(open_page, jupyter, mode: str, out: Path) -> dict:
    notebook = f"show4dstem_{mode}.ipynb"
    page = run_notebook(open_page, jupyter, notebook, mode, webgpu=mode == "webgpu", out=out)
    steps = {"adapter": page.adapter(), "paths": page.render_paths()}

    def record(name: str) -> None:
        # park the pointer off the panels: a click or drag leaves it over one, and the hover
        # readout drawn there (in one mode and not the other, by timing) is not data
        page.mouse.move(5, 300)
        page.wait_for_timeout(2500)
        shots = {}
        for key, selector in (("dp", DP), ("vi", VI), ("fft", FFT)):
            path = out / f"live_{mode}_{name}_{key}.png"
            page.locator(selector).first.screenshot(path=str(path))
            shots[key] = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
        page.screenshot(path=str(out / f"live_{mode}_{name}_page.png"))
        steps[name] = {"shots": shots, "kernel": kernel_report(jupyter, notebook)}

    record("initial")
    for index, (row_fraction, col_fraction) in enumerate([(0.25, 0.30), (0.62, 0.71), (0.80, 0.15)]):
        locator = page.locator(VI).first
        locator.scroll_into_view_if_needed()
        box = locator.bounding_box()
        page.mouse.click(box["x"] + col_fraction * box["width"], box["y"] + row_fraction * box["height"])
        record(f"scan_{index}")
    x, y, _ = detector_on_screen(page, out / f"live_{mode}_probe.png")
    drag(page, (x, y), (x + 40, y + 30))
    record("move_detector")
    x, y, radius = detector_on_screen(page, out / f"live_{mode}_probe.png")
    drag(page, (x + radius * 0.7071, y + radius * 0.7071), (x + radius * 1.1, y + radius * 1.1))
    record("resize_detector")
    page.get_by_text("BF", exact=True).first.click()
    record("preset_bf")
    page.get_by_text("ADF", exact=True).first.click()
    record("preset_adf")
    combos = page.locator('.jp-OutputArea-output [role="combobox"]')
    colormap_menus = [index for index in range(combos.count()) if combos.nth(index).inner_text().strip().lower() == "inferno"]
    combos.nth(colormap_menus[1]).click()
    page.locator('li[role="option"]', has_text="Viridis").first.click()
    record("colormap")
    left = page.locator(VI).first.bounding_box()["x"] - 2
    page.evaluate("""(left) => {
      const slider = Array.from(document.querySelectorAll('.jp-OutputArea-output .MuiSlider-root'))
        .find(s => s.getBoundingClientRect().left >= left && s.querySelectorAll('input[type=range]').length === 2);
      const input = slider.querySelectorAll('input[type=range]')[0];
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(input, '12');
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
    }""", left)
    record("contrast")
    return steps


def test_show4dstem_live_kernel_same_pixels_and_numbers(jupyter, open_page, hardware_adapter, browser_out):
    gpu = drive(open_page, jupyter, "webgpu", browser_out)
    cpu = drive(open_page, jupyter, "cpu", browser_out)
    hardware_adapter(gpu.pop("adapter"))
    assert cpu.pop("adapter") is None
    assert gpu.pop("paths") == ["WebGPU"] and cpu.pop("paths") == ["CPU"]
    # Equal pixels in both modes prove nothing if both are blank: every panel
    # shows data, and each panel changes when the step that drives it runs.
    for mode, steps in (("webgpu", gpu), ("cpu", cpu)):
        for step, record in steps.items():
            for key, shot in record["shots"].items():
                pixels = shot.reshape(-1, 3)
                assert pixels.std(axis=0).max() > 1 and len(np.unique(pixels, axis=0)) >= 8, (mode, step, key, "blank")
        for step_a, step_b, key in (("scan_0", "scan_1", "dp"), ("preset_bf", "preset_adf", "vi"), ("preset_adf", "colormap", "vi")):
            assert np.abs(steps[step_a]["shots"][key] - steps[step_b]["shots"][key]).max() > 0, (mode, step_a, step_b, key)
    summary = {}
    for step, gpu_step in gpu.items():
        cpu_step = cpu[step]
        for kernel in (gpu_step["kernel"], cpu_step["kernel"]):
            np.testing.assert_allclose(kernel["vi_at"], kernel["masked_sum_at"], rtol=1e-6)
        summary[step] = {key: int(np.abs(gpu_step["shots"][key] - cpu_step["shots"][key]).max()) for key in ("dp", "vi", "fft")}
        # A drag can land a fraction of a pixel apart between the two runs;
        # pixels are only comparable when both kernels hold the same detector.
        if gpu_step["kernel"]["pos"] == cpu_step["kernel"]["pos"] and gpu_step["kernel"]["roi"] == cpu_step["kernel"]["roi"]:
            assert max(summary[step].values()) <= 1, (step, summary[step])
    (browser_out / "live_show4dstem.json").write_text(json.dumps(summary, indent=1))
    assert gpu["contrast"]["kernel"]["vi_colormap"] == "viridis"
    assert gpu["contrast"]["kernel"]["vi_vmin_pct"] == 12


def pixel_center(page, selector: str, row: float, col: float, size: int = 16) -> tuple[float, float]:
    """Screen position of the center of pixel (row, col) of an unzoomed square panel."""
    box = page.locator(selector).first.bounding_box()
    return box["x"] + (col + 0.5) * box["width"] / size, box["y"] + (row + 0.5) * box["height"] / size


def switch_profile(page, panel: int) -> None:
    """Turn on the Profile switch of the diffraction (0) or virtual-image (1) panel."""
    page.evaluate("""(panel) => {
      const labels = Array.from(document.querySelectorAll('.jp-OutputArea-output p, .jp-OutputArea-output span'))
        .filter(element => element.textContent === 'Profile' && element.children.length === 0);
      labels[panel].nextElementSibling.querySelector('input').click();
    }""", panel)


def test_scan_roi_resize_follows_the_grab_and_profiles_sit_on_pixel_centers(jupyter, open_page, hardware_adapter, browser_out):
    notebook = "show4dstem_pixels.ipynb"
    page = run_notebook(open_page, jupyter, notebook, "pixels", webgpu=True, out=browser_out)
    hardware_adapter(page.adapter())
    page.locator(VI).first.scroll_into_view_if_needed()
    # A radius-4 scan ROI at (7, 7), pressed on its rim band 3 px right of the center and dragged 3 px further out
    run_in_kernel(jupyter, notebook, "viewer.vi_roi_mode = 'circle'; viewer.vi_roi_radius = 4.0; "
                                     "viewer.vi_roi_center_row = 7.0; viewer.vi_roi_center_col = 7.0; viewer.vi_roi_center = [7.0, 7.0]")
    page.wait_for_timeout(2000)
    start, end = pixel_center(page, VI, 7, 10), pixel_center(page, VI, 7, 13)
    page.mouse.move(*start)
    page.mouse.down()
    radii = []
    for step in range(1, 7):
        page.mouse.move(start[0] + (end[0] - start[0]) * step / 6, start[1])
        page.wait_for_timeout(150)
        radii.append(float(run_in_kernel(jupyter, notebook, "print(viewer.vi_roi_radius)")))
    page.mouse.up()
    page.wait_for_timeout(1500)
    page.locator(VI).first.screenshot(path=str(browser_out / "live_pixels_scan_roi.png"))
    roi = json.loads(run_in_kernel(jupyter, notebook, "import json; print(json.dumps([viewer.vi_roi_radius, viewer.vi_roi_center_row, viewer.vi_roi_center_col]))"))
    # the edge follows the pointer from the press: never below the starting radius, 4 + 3 at the end, center kept
    assert min(radii) >= 4 and roi == [7.0, 7.0, 7.0], (radii, roi)
    run_in_kernel(jupyter, notebook, "viewer.vi_roi_mode = 'off'")
    page.wait_for_timeout(1500)
    # Diffraction profile clicked on the centers of pixels (5, 2) and (5, 13) stores those indices
    page.locator(DP).first.scroll_into_view_if_needed()
    switch_profile(page, 0)
    page.wait_for_timeout(800)
    for col in (2, 13):
        page.mouse.click(*pixel_center(page, DP, 5, col))
        page.wait_for_timeout(800)
    page.wait_for_timeout(1000)
    line = json.loads(run_in_kernel(jupyter, notebook, "import json; print(json.dumps(viewer.profile_line))"))
    page.locator(DP).first.screenshot(path=str(browser_out / "live_pixels_dp_profile.png"))
    assert np.allclose([[point["row"], point["col"]] for point in line], [[5, 2], [5, 13]], atol=0.05), line
    # Virtual-image profile clicked on the centers of (3, 3) and (12, 3) is drawn through column 3's center
    page.locator(VI).first.scroll_into_view_if_needed()
    switch_profile(page, 1)
    page.wait_for_timeout(800)
    for row in (3, 12):
        page.mouse.click(*pixel_center(page, VI, row, 3))
        page.wait_for_timeout(800)
    page.mouse.move(5, 300)
    page.wait_for_timeout(1000)
    path = browser_out / "live_pixels_vi_profile.png"
    page.locator(VI).first.screenshot(path=str(path))
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    purple_cols = np.nonzero((rgb[..., 0] > 120) & (rgb[..., 2] > 200) & (rgb[..., 1] < 60))[1]
    assert purple_cols.size, "no profile line drawn"
    assert abs(np.median(purple_cols) / (rgb.shape[1] / 16) - 3.5) < 0.1, np.median(purple_cols) / (rgb.shape[1] / 16)
    # JupyterLab's debugger panel reports "No active debugger session" on its own
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors
