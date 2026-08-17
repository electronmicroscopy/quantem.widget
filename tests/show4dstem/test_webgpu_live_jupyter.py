import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import nbformat
import pytest


pytest.importorskip("jupyterlab")
pytest.importorskip("playwright.sync_api")


def _chrome_executable():
    candidates = [
        os.environ.get("CHROME_EXECUTABLE"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/opt/google/chrome/chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _notebook_source():
    master = os.environ.get("QT_WEBGPU_LIVE_MASTER")
    if master:
        det_bin = int(os.environ.get("QT_WEBGPU_LIVE_DET_BIN", "4"))
        dtype = os.environ.get("QT_WEBGPU_LIVE_DTYPE", "u8")
        crop = os.environ.get("QT_WEBGPU_LIVE_CROP", "96:160,96:160")
        full = os.environ.get("QT_WEBGPU_LIVE_FULL") == "1"
        return f"""
import numpy as np
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM
master = {master!r}
res = load(master, det_bin={det_bin}, dtype={dtype!r}, verbose=True)
data = res.data
if not {full!r}:
    r, c = {crop!r}.split(",")
    r0, r1 = [int(x) for x in r.split(":")]
    c0, c1 = [int(x) for x in c.split(":")]
    data = data[r0:r1, c0:c1]
    if data.ndim == 4:
        data = np.stack([data, data], axis=0)
w = Show4DSTEM(
    data,
    frame_dim_label="Dataset" if data.ndim == 5 else None,
    frame_labels=["first", "second"] if data.ndim == 5 and data.shape[0] == 2 else None,
    backend="webgpu",
    title="Live WebGPU Jupyter smoke",
    precompute_virtual_images=False,
    show_fft=True,
    verbose=True,
)
w
"""
    return """
import numpy as np
from quantem.widget import Show4DSTEM

data = np.zeros((2, 32, 32, 64, 64), dtype=np.uint16)
rr, cc = np.indices((32, 32))
kr, kc = np.indices((64, 64))
for frame, offset in enumerate((7, 31)):
    data[frame] = (
        offset
        + rr[..., None, None] * 3
        + cc[..., None, None] * 5
        + kr[None, None, ...] * 2
        + kc[None, None, ...]
    ) % 251

w = Show4DSTEM(
    data,
    frame_dim_label="Dataset",
    frame_labels=["first", "second"],
    backend="webgpu",
    title="Live WebGPU Jupyter smoke",
    precompute_virtual_images=False,
    show_fft=True,
    verbose=True,
)
w
"""


def _write_notebook(root):
    nb = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(_notebook_source())],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    path = root / "show4dstem_live_webgpu_agent.ipynb"
    nbformat.write(nb, path)
    return path


def _start_jupyter(root, port, token):
    return subprocess.Popen(
        [
            "jupyter",
            "lab",
            "--no-browser",
            f"--port={port}",
            "--ip=127.0.0.1",
            f"--IdentityProvider.token={token}",
            f"--ServerApp.root_dir={root}",
            "--ServerApp.open_browser=False",
            "--ServerApp.terminals_enabled=False",
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_jupyter(port, token, proc):
    import urllib.request

    url = f"http://127.0.0.1:{port}/api/status?token={token}"
    for _ in range(90):
        if proc.poll() is not None:
            raise RuntimeError(f"JupyterLab exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("JupyterLab did not become ready")


def _canvas_boxes(page):
    return page.evaluate(
        """() => [...document.querySelectorAll('canvas')].map((c, i) => {
          const r = c.getBoundingClientRect();
          return {i, x:r.x, y:r.y, w:r.width, h:r.height, width:c.width, height:c.height};
        }).filter(c => c.w > 20 && c.h > 20)"""
    )


def _drag_canvas(page, box, dx=35, dy=25):
    x = box["x"] + box["w"] * 0.52
    y = box["y"] + box["h"] * 0.52
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + dx, y + dy, steps=12)
    page.mouse.up()


def _move_dataset_slider(page, require=True):
    slider = page.evaluate(
        """() => [...document.querySelectorAll('.MuiSlider-root')].map((root, i) => {
          const r = root.getBoundingClientRect();
          return {i, rect:{x:r.x,y:r.y,w:r.width,h:r.height},
            inputs:[...root.querySelectorAll('input')].map(inp => ({min:inp.min,max:inp.max,value:inp.value}))};
        }).find(root => root.inputs.some(inp => inp.min === '0' && inp.max === '1'))"""
    )
    if slider is None:
        assert not require, "Dataset/frame slider was not found"
        return False
    slider_root = page.locator(".MuiSlider-root").nth(int(slider["i"]))
    slider_root.scroll_into_view_if_needed()
    page.wait_for_timeout(250)
    rect = slider_root.bounding_box()
    assert rect is not None
    page.mouse.move(rect["x"] + 2, rect["y"] + rect["height"] / 2)
    page.mouse.down()
    page.mouse.move(
        rect["x"] + rect["width"] - 2,
        rect["y"] + rect["height"] / 2,
        steps=10,
    )
    page.mouse.up()
    page.wait_for_timeout(800)
    return True


def _toggle_fft(page):
    clicked = page.evaluate(
        """() => {
          const nodes = [...document.querySelectorAll('p,span,label,div')];
          const label = nodes.find(n => /^FFT:?$/i.test((n.textContent || '').trim()));
          if (!label) return false;
          let root = label.parentElement;
          for (let depth = 0; root && depth < 5; depth++, root = root.parentElement) {
            const input = root.querySelector('input[type="checkbox"]');
            if (input) {
              input.click();
              return true;
            }
          }
          return false;
        }"""
    )
    assert clicked, "FFT toggle was not found"
    page.wait_for_timeout(800)


def _measure_fps(page):
    return page.evaluate(
        """async () => {
          let frames = 0;
          const start = performance.now();
          await new Promise(resolve => {
            function tick(t) {
              frames++;
              if (t - start >= 3000) resolve();
              else requestAnimationFrame(tick);
            }
            requestAnimationFrame(tick);
          });
          return frames / ((performance.now() - start) / 1000);
        }"""
    )


def _click_copy_buttons(page):
    buttons = page.get_by_role("button", name="COPY")
    count = buttons.count()
    assert count >= 2, f"Expected DP and VI COPY buttons, found {count}"
    copied = []
    for idx in range(2):
        buttons.nth(idx).click()
        page.wait_for_timeout(500)
        types = page.evaluate(
            """async () => {
              const items = await navigator.clipboard.read();
              return items.flatMap(item => item.types);
            }"""
        )
        assert "image/png" in types
        copied.append(types)
    return copied


def _exercise_dpc_backend(page):
    page.wait_for_function(
        "window.__sh4d && typeof window.__sh4d.recomputeVI === 'function' && document.body.innerText.includes('DPC')",
        timeout=60_000,
    )
    results = []
    for source in ("DPC_row", "DPC_col"):
        results.append(
            page.evaluate(
                """async (source) => {
                  const api = window.__sh4d;
                  const rows = Number(api.model.get("shape_rows") || 0);
                  const cols = Number(api.model.get("shape_cols") || 0);
                  api.model.set("vi_source", source);
                  await api.recomputeVI();
                  const view = api.model.get("virtual_image_bytes");
                  const arr = new Float32Array(view.buffer, view.byteOffset, view.byteLength / 4);
                  let sum = 0;
                  for (let i = 0; i < arr.length; i++) sum += arr[i];
                  return { source: api.model.get("vi_source"), length: arr.length, expected: rows * cols, sum };
                }""",
                source,
            )
        )
    for result in results:
        assert result["source"] in {"DPC_row", "DPC_col"}
        assert result["length"] == result["expected"]
    return results


@pytest.mark.skipif(
    os.environ.get("QT_RUN_JUPYTER_WEBGPU_TESTS") != "1",
    reason="set QT_RUN_JUPYTER_WEBGPU_TESTS=1 to run headed Jupyter WebGPU smoke tests",
)
def test_live_jupyter_webgpu_widget_interaction(tmp_path):
    from playwright.sync_api import sync_playwright

    chrome = _chrome_executable()
    if chrome is None:
        pytest.skip("Chrome/Chromium executable not found")

    root = tmp_path / "show4dstem-agent-jupyter"
    root.mkdir()
    notebook = _write_notebook(root)
    port = int(os.environ.get("QT_JUPYTER_WEBGPU_PORT", "0")) or _free_port()
    token = "show4dstem-agent-token"
    profile = Path(f"/tmp/cdp-show4dstem-live-jupyter-{port}")
    shutil.rmtree(profile, ignore_errors=True)
    server = _start_jupyter(root, port, token)
    try:
        _wait_for_jupyter(port, token, server)
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(
                str(profile),
                executable_path=chrome,
                headless=False,
                viewport={"width": 1280, "height": 900},
                args=[
                    "--enable-unsafe-webgpu",
                    "--enable-features=Vulkan,WebGPU",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
            )
            try:
                page = context.pages[0] if context.pages else context.new_page()
                url = f"http://127.0.0.1:{port}/lab/tree/{notebook.name}?token={token}"
                context.grant_permissions(
                    ["clipboard-read", "clipboard-write"],
                    origin=f"http://127.0.0.1:{port}",
                )
                page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                page.wait_for_selector(".jp-Notebook", timeout=120_000)
                page.locator(".jp-Cell").first.click()
                page.keyboard.press("Shift+Enter")
                page.wait_for_function(
                    "document.body.innerText.includes('Live WebGPU Jupyter smoke') && document.querySelectorAll('canvas').length >= 4",
                    timeout=180_000,
                )
                page.wait_for_timeout(3000)

                assert page.evaluate("!!navigator.gpu")
                canvases = _canvas_boxes(page)
                assert len(canvases) >= 4
                before = page.screenshot(full_page=False)
                for box in sorted(canvases, key=lambda c: c["w"] * c["h"], reverse=True)[:2]:
                    _drag_canvas(page, box)
                frame_slider_moved = _move_dataset_slider(
                    page,
                    require=os.environ.get("QT_WEBGPU_REQUIRE_FRAME_SLIDER", "1") != "0",
                )
                _toggle_fft(page)
                dpc_results = _exercise_dpc_backend(page)
                copied_types = _click_copy_buttons(page)
                after = page.screenshot(full_page=False)
                screenshot_changed = hashlib.sha256(before).hexdigest() != hashlib.sha256(after).hexdigest()
                assert screenshot_changed
                fps = _measure_fps(page)
                min_fps = float(os.environ.get("QT_WEBGPU_MIN_FPS", "30"))
                print(
                    json.dumps(
                        {
                            "navigator_gpu": True,
                            "canvas_count": len(canvases),
                            "frame_slider_moved": frame_slider_moved,
                            "fft_toggled": True,
                            "dpc_checked": [r["source"] for r in dpc_results],
                            "copy_png_buttons": len(copied_types),
                            "screenshot_changed": screenshot_changed,
                            "fps": fps,
                            "min_fps": min_fps,
                        },
                        indent=2,
                    )
                )
                assert fps >= min_fps, f"FPS too low: {fps:.2f}"
            finally:
                context.close()
    finally:
        with suppress(Exception):
            server.terminate()
            server.wait(timeout=10)
        with suppress(Exception):
            server.kill()
        shutil.rmtree(profile, ignore_errors=True)
