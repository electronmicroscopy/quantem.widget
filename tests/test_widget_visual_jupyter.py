"""Opt-in real-browser Jupyter visual smoke for the core widget family.

This is an agent/release gate, not default CI. It proves the browser renders
real widget canvases for Show2D, Show3D, Show3DSlices, and Show4DSTEM and that
basic pointer interaction changes pixels.
"""
from __future__ import annotations

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


def _chrome_executable() -> str | None:
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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_jupyter(root: Path, port: int, token: str) -> subprocess.Popen:
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


def _wait_for_jupyter(port: int, token: str, proc: subprocess.Popen) -> None:
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


def _notebook_source() -> str:
    return r"""
import numpy as np
from IPython.display import display
from quantem.widget import Show2D, Show3D, Show3DSlices, Show4DSTEM

rng = np.random.default_rng(42)

image = rng.normal(size=(96, 96)).astype("float32")
image[24:72, 30:66] += 4
display(Show2D(image, title="Agent Visual Show2D", offline=False, verbose=False))

z, y, x = np.indices((24, 64, 64))
volume = (
    np.exp(-(((z - 10) / 5) ** 2 + ((y - 28) / 13) ** 2 + ((x - 34) / 11) ** 2))
    + 0.3 * np.exp(-(((z - 17) / 4) ** 2 + ((y - 43) / 8) ** 2 + ((x - 20) / 9) ** 2))
).astype("float32")
display(Show3D(volume, title="Agent Visual Show3D"))
display(Show3DSlices(volume, title="Agent Visual Show3DSlices"))

scan_r, scan_c = np.indices((16, 16))
det_r, det_c = np.indices((32, 32))
stack = (
    8
    + scan_r[..., None, None] * 2
    + scan_c[..., None, None] * 3
    + det_r[None, None, ...]
    + det_c[None, None, ...]
).astype("uint16")
stack[:, :, 13:19, 13:19] += 300
display(Show4DSTEM(
    stack,
    title="Agent Visual Show4DSTEM",
    show_fft=True,
    precompute_virtual_images=False,
    verbose=False,
))
"""


def _write_notebook(root: Path) -> Path:
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
    path = root / "widget_visual_agent.ipynb"
    nbformat.write(nb, path)
    return path


def _visible_canvas_boxes(page) -> list[dict]:
    return page.evaluate(
        """() => [...document.querySelectorAll('canvas')].map((c, i) => {
          const r = c.getBoundingClientRect();
          return {i, x:r.x, y:r.y, w:r.width, h:r.height, width:c.width, height:c.height};
        }).filter(c => c.w > 20 && c.h > 20 && c.width > 0 && c.height > 0)"""
    )


def _drag(page, box: dict, dx: float = 32, dy: float = 24) -> None:
    x = box["x"] + box["w"] * 0.52
    y = box["y"] + box["h"] * 0.52
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + dx, y + dy, steps=12)
    page.mouse.up()


def _measure_fps(page) -> float:
    return float(
        page.evaluate(
            """async () => {
              let frames = 0;
              const start = performance.now();
              await new Promise(resolve => {
                function tick(t) {
                  frames++;
                  if (t - start >= 1000) resolve();
                  else requestAnimationFrame(tick);
                }
                requestAnimationFrame(tick);
              });
              return frames / ((performance.now() - start) / 1000);
            }"""
        )
    )


def _click_visible_text(page, text: str) -> bool:
    return bool(
        page.evaluate(
            """(text) => {
              const nodes = [...document.querySelectorAll('button,[role="button"],label,span,p,div')]
                .filter(n => (n.textContent || '').trim() === text);
              for (const node of nodes) {
                const r = node.getBoundingClientRect();
                if (r.width > 0 && r.height > 0) {
                  node.click();
                  return true;
                }
              }
              return false;
            }""",
            text,
        )
    )


def _click_visible_checkboxes(page, limit: int = 3) -> int:
    return int(
        page.evaluate(
            """(limit) => {
              let clicked = 0;
              const inputs = [...document.querySelectorAll('input[type="checkbox"]')];
              for (const input of inputs) {
                const host = input.closest('label') || input.parentElement || input;
                const r = host.getBoundingClientRect();
                if (r.width > 0 && r.height > 0) {
                  input.click();
                  clicked++;
                  if (clicked >= limit) break;
                }
              }
              return clicked;
            }""",
            limit,
        )
    )


@pytest.mark.skipif(
    os.environ.get("QT_RUN_WIDGET_VISUAL_TESTS") != "1",
    reason="set QT_RUN_WIDGET_VISUAL_TESTS=1 to run headed Jupyter widget visual tests",
)
def test_core_widgets_render_and_interact_in_jupyter(tmp_path):
    from playwright.sync_api import sync_playwright

    chrome = _chrome_executable()
    if chrome is None:
        pytest.skip("Chrome/Chromium executable not found")

    root = tmp_path / "widget-visual-agent"
    root.mkdir()
    notebook = _write_notebook(root)
    port = int(os.environ.get("QT_WIDGET_VISUAL_PORT", "0")) or _free_port()
    token = "widget-visual-agent-token"
    profile = Path(f"/tmp/cdp-widget-visual-{port}")
    shutil.rmtree(profile, ignore_errors=True)
    server = _start_jupyter(root, port, token)
    try:
        _wait_for_jupyter(port, token, server)
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(
                str(profile),
                executable_path=chrome,
                headless=False,
                viewport={"width": 1280, "height": 1000},
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
                page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                page.wait_for_selector(".jp-Notebook", timeout=120_000)
                page.locator(".jp-Cell").first.click()
                run_command = page.locator('[data-command="notebook:run-cell-and-select-next"]')
                assert run_command.count() == 1
                run_command.click()
                page.wait_for_function(
                    "document.body.innerText.includes('Agent Visual Show2D')"
                    " && document.body.innerText.includes('Agent Visual Show3D')"
                    " && document.body.innerText.includes('Agent Visual Show3DSlices')"
                    " && document.body.innerText.includes('Agent Visual Show4DSTEM')"
                    " && document.querySelectorAll('canvas').length >= 6",
                    timeout=180_000,
                )
                page.wait_for_timeout(2500)
                page.evaluate(
                    """() => {
                      const nodes = [...document.querySelectorAll('h1,h2,h3,p,span,div')];
                      const target = nodes.find(n => (n.textContent || '').includes('Agent Visual Show4DSTEM'));
                      if (target) target.scrollIntoView({block: 'center', inline: 'nearest'});
                    }"""
                )
                page.wait_for_timeout(500)
                boxes = _visible_canvas_boxes(page)
                assert len(boxes) >= 6
                before = page.screenshot(full_page=False)
                for box in sorted(boxes, key=lambda c: c["w"] * c["h"], reverse=True)[:4]:
                    _drag(page, box)
                    page.mouse.move(box["x"] + box["w"] * 0.5, box["y"] + box["h"] * 0.5)
                    page.mouse.wheel(0, -350)
                clicked_controls = [
                    name for name in ("ADF", "DF", "FFT", "Profile")
                    if _click_visible_text(page, name)
                ]
                clicked_checkboxes = _click_visible_checkboxes(page)
                page.wait_for_timeout(1000)
                after = page.screenshot(full_page=False)
                screenshot_changed = (
                    hashlib.sha256(before).hexdigest() != hashlib.sha256(after).hexdigest()
                )
                assert screenshot_changed
                fps = _measure_fps(page)
                min_fps = float(os.environ.get("QT_WIDGET_MIN_FPS", "30"))
                summary = {
                    "canvas_count": len(boxes),
                    "screenshot_changed": screenshot_changed,
                    "clicked_controls": clicked_controls,
                    "clicked_checkboxes": clicked_checkboxes,
                    "fps": fps,
                    "min_fps": min_fps,
                    "navigator_gpu": bool(page.evaluate("!!navigator.gpu")),
                }
                print(json.dumps(summary, indent=2))
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
