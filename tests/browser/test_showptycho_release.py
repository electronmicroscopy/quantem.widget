"""Installed-wheel acceptance on a real acquisition and a fresh Jupyter kernel.

Set QUANTEM_WIDGET_BROWSER=1, QUANTEM_DIRECT_RELEASE_SOURCE and optionally
QUANTEM_DIRECT_RELEASE_CALIBRATION (a JSON object). This is a hardware release
gate, not the synthetic browser smoke tier. Missing data is a failure when
QUANTEM_DIRECT_RELEASE=1. Screenshots and timings stay in the requested output.
"""

import json
import os
import re
import time

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.browser
PHASE = '[data-quantem-scientific-output="Phase"]'
IMAGE = '[data-showptycho-images] [data-quantem-scientific-output]'


def test_installed_joint_fit_draw_move_resize_and_restore(
    jupyter,
    open_page,
    hardware_adapter,
    browser_out,
):
    """Draw, move and resize one region without replacing the other or Overview."""
    if os.environ.get("QUANTEM_DIRECT_RELEASE") != "1":
        pytest.skip("Opt-in installed-wheel release gate: QUANTEM_DIRECT_RELEASE=1")
    assert os.environ.get("QUANTEM_DIRECT_RELEASE_SOURCE"), (
        "Provide the real acquisition path"
    )
    calibration = json.loads(os.environ.get("QUANTEM_DIRECT_RELEASE_CALIBRATION", "{}"))
    notebook = "showptycho_release.ipynb"
    setup = f"""from pathlib import Path
import json
import os
import time
import numpy as np
import quantem.gpu
import quantem.widget
from quantem.gpu import DirectPtycho, io
from quantem.gpu.device import resolve

for package in (quantem.gpu, quantem.widget):
    assert "site-packages" in Path(package.__file__).resolve().parts, "Install wheels and unset PYTHONPATH"
backend = resolve("auto")
assert backend == {os.environ.get("QUANTEM_DIRECT_RELEASE_BACKEND", "mps" if os.sys.platform == "darwin" else "cuda")!r}
data = io.load(os.environ['QUANTEM_DIRECT_RELEASE_SOURCE'], verbose=False)
for name, value in {calibration!r}.items():
    setattr(data.calibration, name, value)
if data.calibration.detector_sampling_mrad is None and data.calibration.detector_sampling_inv_A is None:
    data.calibration.detector_sampling()
direct = DirectPtycho(data)
started = time.perf_counter()
fit = direct.aberrations(trials=200, seed=42, compare_standard=False)
full_fit_s = time.perf_counter() - started
assert fit.optimizer["host_numerical_search"] is False
assert fit.optimizer["device"] == backend
assert len(fit.trial_records) == 200 * (len(fit.rotation_branches.branches) if fit.rotation_branches else 1)
result = direct.run(upsample="auto")
save_dir = Path.cwd() / 'calibration'
viewer = direct.show(local_tilt=True, save_dir=save_dir)
viewer.regions = []
overview = viewer.phase_bytes
viewer
"""
    jupyter.write_notebook(notebook, setup)
    page = jupyter.open(
        open_page, notebook, "direct-release", webgpu=True, ready=PHASE, out=browser_out
    )
    adapter = page.adapter()
    hardware_adapter(adapter)
    assert adapter["vendor"].lower() in {"apple", "nvidia"}, adapter
    page.locator(".showptycho-root").screenshot(
        path=str(browser_out / "showptycho_release_ready.png")
    )
    page.get_by_label("Local Tilt", exact=True).check()
    editor = page.get_by_label("Local tilt region editor")
    editor.wait_for(timeout=60_000)
    page.locator("[data-local-tilt-controls]").get_by_label("Auto", exact=True).check()
    timings = {}

    def draw(row, col, size):
        page.locator("[data-local-tilt-controls] button").filter(
            has_text=re.compile(r"^Draw$")
        ).click()
        editor.scroll_into_view_if_needed()
        box = editor.bounding_box()
        page.mouse.move(box["x"] + col * box["width"], box["y"] + row * box["height"])
        page.mouse.down()
        page.mouse.move(
            box["x"] + (col + size) * box["width"],
            box["y"] + (row + size) * box["height"],
            steps=10,
        )
        start = time.perf_counter()
        page.mouse.up()
        return start

    def completed(start, label, count):
        # Pending work is visible, and completion must arrive in the live page.
        page.get_by_label("Refitting regions", exact=True).wait_for(
            state="visible", timeout=30_000
        )
        page.get_by_label("Refitting regions", exact=True).wait_for(
            state="hidden", timeout=180_000
        )
        page.wait_for_function(
            """() => {
            const s = document.querySelector('[data-local-tilt-controls] [role=status]');
            return s && /Fits ready/.test(s.textContent)
                && !/refitting|trials/i.test(s.textContent);
        }""",
            timeout=180_000,
        )
        page.wait_for_function(
            "n => document.querySelectorAll('[data-local-tilt-result]').length === n",
            arg=count,
        )
        timings[label] = time.perf_counter() - start

    completed(draw(0.15, 0.15, 0.25), "draw_first_s", 1)
    jupyter.run_in_kernel(
        notebook,
        "first_fit = viewer.region_fits[0]; first_image = next(iter(viewer.region_images.values())).copy()",
    )
    completed(draw(0.55, 0.55, 0.25), "draw_second_s", 2)
    proof = jupyter.run_in_kernel(
        notebook,
        """
assert viewer.phase_bytes == overview
assert viewer.region_fits[0] == first_fit
np.testing.assert_array_equal(viewer.region_images[first_fit['label'] + ' · Direct'], first_image)
print('independent')
""",
    )
    assert "independent" in proof
    page.locator("[data-local-tilt-controls] button").filter(
        has_text=re.compile(r"^Move$")
    ).click()
    editor.scroll_into_view_if_needed()
    box = editor.bounding_box()
    page.mouse.move(box["x"] + 0.675 * box["width"], box["y"] + 0.675 * box["height"])
    page.mouse.down()
    page.mouse.move(
        box["x"] + 0.70 * box["width"], box["y"] + 0.70 * box["height"], steps=10
    )
    start = time.perf_counter()
    page.mouse.up()
    completed(start, "move_second_s", 2)
    # The selected rectangle's bottom-right handle remains directly draggable.
    handle = editor.locator('rect[rx="2"]').last
    bounds = handle.bounding_box()
    page.mouse.move(
        bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2
    )
    page.mouse.down()
    page.mouse.move(bounds["x"] - 24, bounds["y"] - 15, steps=10)
    start = time.perf_counter()
    page.mouse.up()
    completed(start, "resize_second_s", 2)
    proof = jupyter.run_in_kernel(
        notebook,
        """
assert viewer.phase_bytes == overview
assert viewer.region_fits[0] == first_fit
np.testing.assert_array_equal(viewer.region_images[first_fit['label'] + ' · Direct'], first_image)
assert all(len(f['fit_record']['fits']) == 1 and f['num_completed_trials'] == 200
           and f['refinement'] == 'bfgs' and f['optimizer']['host_numerical_search'] is False
           for f in viewer.region_fits)
viewer.save_calibration()
print(json.dumps({'backend': backend, 'full_fit_s': full_fit_s,
                  'region_fit_s': [f['fit_seconds'] for f in viewer.region_fits]}))
""",
    )
    timings["kernel"] = json.loads(proof.splitlines()[-1])
    screenshot = browser_out / "showptycho_release_phase.png"
    page.locator(IMAGE).first.screenshot(path=str(screenshot))
    assert np.asarray(Image.open(screenshot).convert("L")).std() > 10
    page.locator(".showptycho-root").screenshot(
        path=str(browser_out / "showptycho_release_regions.png")
    )
    # Release the first kernel's GPU allocations before a fresh-kernel reopen.
    proof = jupyter.run_in_kernel(
        notebook,
        """
Path(viewer.calibration_path).with_name('release_expected.json').write_text(json.dumps({
    'fits': viewer.region_fits, 'C10_nm': fit.aberrations['C10_nm']}))
viewer.close()
direct.close()
data.close()
print('saved')
""",
    )
    assert "saved" in proof
    reopen = (
        setup.split("started = time.perf_counter()")[0]
        + """
save_dir = Path.cwd() / 'calibration'
expected = json.loads((save_dir / 'release_expected.json').read_text())
def unexpected_refit(*args, **kwargs):
    raise AssertionError('Saved calibration should restore without refitting')
direct.aberrations = unexpected_refit
viewer = direct.show(local_tilt=True, save_dir=save_dir)
assert viewer.region_fits == expected['fits']
assert viewer.fitted_C10_nm == expected['C10_nm']
viewer
"""
    )
    jupyter.write_notebook("showptycho_restored.ipynb", reopen)
    restored_page = jupyter.open(
        open_page,
        "showptycho_restored.ipynb",
        "direct-restored",
        webgpu=True,
        ready=IMAGE,
        out=browser_out,
        collapse_input=True,
    )
    hardware_adapter(restored_page.adapter())
    restored_page.locator(".showptycho-root").screenshot(
        path=str(browser_out / "showptycho_release_restored.png")
    )
    (browser_out / "showptycho_release_timings.json").write_text(
        json.dumps(timings, indent=2)
    )
    assert [e for e in page.errors if "debugger session" not in e] == [], page.errors
