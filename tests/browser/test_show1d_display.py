"""Show1D snapshot panels in headed Chrome on the hardware adapter.

Stepping through snapshot groups whose images carry different labels mounts a
new contrast histogram for every group. The page counts the WebGPU buffers it
holds, and stepping through every group must not leave one behind. Zooming a
snapshot panel, or hovering the plot beside it, redraws the panel without
colouring its pixels again (no putImageData). With ten trials in a group the
histogram cells are narrow, and each keeps its Full / 1-99% / Reset buttons
inside its own cell without overlapping. In a live kernel, View selected as 2D
embeds a Show2D below the plot, a colormap change on the Show1D reaches the
embedded Show2D's model in the kernel, and Hide removes the view.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

COUNT_GPU_BUFFERS = """
window.__liveGpuBuffers = 0;
const createBuffer = GPUDevice.prototype.createBuffer;
GPUDevice.prototype.createBuffer = function (...args) {
  const buffer = createBuffer.apply(this, args);
  window.__liveGpuBuffers += 1;
  const destroy = buffer.destroy.bind(buffer);
  let destroyed = false;
  buffer.destroy = () => {
    if (!destroyed) window.__liveGpuBuffers -= 1;
    destroyed = true;
    destroy();
  };
  return buffer;
};
"""
COUNT_PUT_IMAGE_DATA = """
window.__putImageData = 0;
const putImageData = CanvasRenderingContext2D.prototype.putImageData;
CanvasRenderingContext2D.prototype.putImageData = function (...args) {
  window.__putImageData += 1;
  return putImageData.apply(this, args);
};
"""
GROUPS = 12


@pytest.fixture(scope="module")
def labelled_snapshots(browser_out) -> str:
    from quantem.widget import Show1D

    rng = np.random.default_rng(0)
    # Snapshots sit between the trace points, so hovering a point previews no group.
    widget = Show1D(np.stack([np.zeros(GROUPS), np.full(GROUPS, 0.1)]).astype(np.float32), labels=["loss", "error"])
    for group in range(GROUPS):
        widget.snapshot(group + 0.5, **{f"frame {group}": rng.random((96, 96)).astype(np.float32)})
    widget.export_html(browser_out / "show1d_snapshots.html")
    return "show1d_snapshots.html"


def test_stepping_snapshot_histograms_leaves_no_gpu_buffers(labelled_snapshots, serve, open_page, hardware_adapter, browser_out):
    page = open_page(f"{serve(browser_out)}/{labelled_snapshots}", webgpu=True)
    page.add_init_script(COUNT_GPU_BUFFERS)
    page.reload()
    page.wait_for_selector('[data-testid="show1d-panel-histogram"]', timeout=60_000)
    page.wait_for_timeout(3_000)
    hardware_adapter(page.adapter())
    before = page.evaluate("window.__liveGpuBuffers")
    previous = page.get_by_label("Previous snapshot")
    for _ in range(GROUPS - 1):
        previous.click()
        page.wait_for_timeout(250)
    page.wait_for_timeout(1_000)
    labels = page.locator('[data-testid="show1d-panel-histogram"] p').first.text_content()
    after = page.evaluate("window.__liveGpuBuffers")
    page.screenshot(path=str(browser_out / "show1d_histogram_steps.png"))
    assert labels == "frame 0", labels
    assert after == before, f"{after - before} WebGPU buffers left behind by {GROUPS - 1} histogram mounts"
    assert page.errors == [], page.errors


def test_plot_hover_and_zoom_redraw_snapshots_without_recolouring(labelled_snapshots, serve, open_page, hardware_adapter, browser_out):
    page = open_page(f"{serve(browser_out)}/{labelled_snapshots}", webgpu=True)
    page.add_init_script(COUNT_PUT_IMAGE_DATA)
    page.reload()
    page.wait_for_selector('[data-testid^="show1d-snapshot-panel-"] canvas', timeout=60_000)
    page.wait_for_timeout(3_000)
    hardware_adapter(page.adapter())
    start = page.evaluate("window.__putImageData")
    plot = page.locator("canvas[data-quantem-scientific-output='show1d-plot']").bounding_box()
    # Along the zero trace (plot frame top 16 px, bottom 46 px, y range padded by 4 %):
    # every move hovers a trace point, which re-renders the widget.
    trace_row = plot["y"] + 16 + (1 - 0.004 / 0.108) * (plot["height"] - 62)
    for step in range(12):
        page.mouse.move(plot["x"] + 70 + step * (plot["width"] - 80) / 12, trace_row)
        page.wait_for_timeout(80)
    hovered = page.evaluate("document.querySelector(\"canvas[data-quantem-scientific-output='show1d-plot']\").nextElementSibling.toDataURL()")
    tile = page.locator('[data-testid^="show1d-snapshot-panel-"] canvas').first
    before = tile.screenshot()
    box = tile.bounding_box()
    page.mouse.move(box["x"] + 0.4 * box["width"], box["y"] + 0.6 * box["height"])
    for _ in range(5):
        page.mouse.wheel(0, -100)
        page.wait_for_timeout(150)
    page.wait_for_timeout(500)
    zoomed = tile.screenshot()
    recoloured = page.evaluate("window.__putImageData") - start
    page.screenshot(path=str(browser_out / "show1d_hover_zoom.png"))
    assert len(hovered) > 2_000, "the plot hover marker was never drawn"
    assert zoomed != before, "the wheel did not zoom the snapshot panel"
    assert recoloured == 0, f"{recoloured} full-image recolours for 12 plot hover moves and 5 wheel steps"
    assert page.errors == [], page.errors


@pytest.fixture(scope="module")
def ten_trials(browser_out) -> str:
    from quantem.widget import Show1D

    rng = np.random.default_rng(1)
    widget = Show1D(rng.random((10, 8)).astype(np.float32), labels=[f"lambda {trial}" for trial in range(10)])
    widget.snapshot(7, **{f"lambda {trial}": rng.random((64, 64)).astype(np.float32) for trial in range(10)})
    widget.export_html(browser_out / "show1d_ten_trials.html")
    return "show1d_ten_trials.html"


def test_narrow_histograms_keep_their_buttons_inside(ten_trials, serve, open_page, hardware_adapter, browser_out):
    page = open_page(f"{serve(browser_out)}/{ten_trials}", webgpu=True)
    page.wait_for_function("document.querySelectorAll('[data-testid=\"show1d-panel-histogram\"]').length === 10", timeout=60_000)
    page.wait_for_timeout(2_000)
    hardware_adapter(page.adapter())
    page.screenshot(path=str(browser_out / "show1d_ten_histograms.png"))
    cells = page.evaluate("""() => Array.from(document.querySelectorAll('[data-testid="show1d-panel-histogram"]')).map(cell => ({
      cell: cell.getBoundingClientRect().toJSON(),
      buttons: Array.from(cell.querySelectorAll('button')).map(button => button.getBoundingClientRect().toJSON()),
    }))""")
    for index, cell in enumerate(cells):
        assert len(cell["buttons"]) == 3
        for button in cell["buttons"]:
            assert cell["cell"]["left"] - 0.5 <= button["left"] and button["right"] <= cell["cell"]["right"] + 0.5, (index, cell)
        for first, second in zip(cell["buttons"], cell["buttons"][1:]):
            overlap_x = min(first["right"], second["right"]) - max(first["left"], second["left"])
            overlap_y = min(first["bottom"], second["bottom"]) - max(first["top"], second["top"])
            assert overlap_x <= 0 or overlap_y <= 0, (index, cell)
    assert page.errors == [], page.errors


def test_view_selected_as_2d_embeds_a_linked_show2d(jupyter, open_page, hardware_adapter, browser_out):
    jupyter.write_notebook("show1d_handoff.ipynb", (
        "import numpy as np\nfrom quantem.widget import Show1D\n\n"
        "rng = np.random.default_rng(0)\nmonitor = Show1D(rng.random((2, 6)).astype(np.float32), labels=['a', 'b'])\n"
        "monitor.snapshot(3, a=rng.random((64, 64)).astype(np.float32), b=rng.random((64, 64)).astype(np.float32))\nmonitor"))
    page = jupyter.open(open_page, "show1d_handoff.ipynb", "show1d-handoff", webgpu=True,
                        ready='[data-testid="show1d-snapshot-grid"]', out=browser_out)
    hardware_adapter(page.adapter())
    page.get_by_label("Open view options").click()
    page.get_by_role("menuitem", name="View selected as 2D").click()
    region = page.get_by_role("region", name="2D view")
    region.locator('canvas[data-quantem-scientific-output="show2d-image-0"]').wait_for(timeout=60_000)
    page.wait_for_timeout(2_000)
    jupyter.run_in_kernel("show1d_handoff.ipynb", "monitor.image_cmap = 'magma'")
    page.wait_for_timeout(2_000)
    linked = jupyter.run_in_kernel("show1d_handoff.ipynb", "print(monitor.prepared_view.cmap, len(monitor.prepared_view.labels))")
    page.screenshot(path=str(browser_out / "show1d_handoff.png"), full_page=True)
    region.get_by_role("button", name="Hide", exact=True).click()
    region.wait_for(state="detached", timeout=30_000)
    cleared = jupyter.run_in_kernel("show1d_handoff.ipynb", "print(monitor.prepared_view)")
    assert linked == "magma 2", linked
    assert cleared == "None", cleared
    # JupyterLab's debugger extension reports a missing session on its own.
    assert [error for error in page.errors if "debugger session" not in error] == [], page.errors
