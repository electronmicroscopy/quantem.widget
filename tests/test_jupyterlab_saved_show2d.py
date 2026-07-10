from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import os
import shutil
import socket
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import nbformat
import pytest


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in version.replace("-", ".").split("."):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            digits = ""
            for char in chunk:
                if not char.isdigit():
                    break
                digits += char
            if digits:
                parts.append(int(digits))
            break
    return tuple(parts)


def test_package_requires_fixed_jupyterlab_widget_manager():
    import tomllib

    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    assert "anywidget>=0.11.0" in deps
    assert "jupyterlab_widgets>=3.0.10" in deps


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
    env = os.environ.copy()
    src = str(Path(__file__).parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
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
        env=env,
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


def _write_executed_saved_widgets_notebook(root: Path) -> tuple[Path, Path]:
    pytest.importorskip("nbclient")
    from nbclient import NotebookClient
    from nbformat.sign import NotebookNotary

    marker = root / "run-count.txt"
    repo_src = Path(__file__).parents[1] / "src"
    code = f"""
import sys
from pathlib import Path

sys.path.insert(0, {str(repo_src)!r})
marker = Path({str(marker)!r})
count = int(marker.read_text()) if marker.exists() else 0
marker.write_text(str(count + 1))

import numpy as np
from IPython.display import display
from quantem.widget import Show2D, Show3D, Show3DSlices

rng = np.random.default_rng(11)
image = rng.normal(size=(96, 96)).astype("float32")
image[30:66, 24:72] += 5
display(Show2D(image, title="Saved Show2D Reopen", size=420, verbose=False))

haadf_stack = np.stack([image + frame * 0.25 for frame in range(4)], axis=0)
display(Show2D(
    [image, haadf_stack],
    labels=["EDS map", "HAADF stack"],
    panel_frame_indices=[0, 2],
    title="Saved Mixed-Stack Show2D Reopen",
    ncols=2,
    size=280,
    save_state=True,
    verbose=False,
))

z, y, x = np.indices((18, 48, 48))
volume = (
    np.exp(-(((z - 8) / 4) ** 2 + ((y - 22) / 10) ** 2 + ((x - 24) / 9) ** 2))
    + 0.4 * np.exp(-(((z - 13) / 3) ** 2 + ((y - 31) / 7) ** 2 + ((x - 14) / 6) ** 2))
).astype("float32")
display(Show3D(volume, title="Saved Show3D Reopen", offline=True, show_fft=False, panel_width_px=280))
display(Show3DSlices(volume, title="Saved Show3DSlices Reopen", offline=True, panel_width_px=240))
"""
    nb = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(code)],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    notebook = root / "saved_widgets_reopen.ipynb"
    client = NotebookClient(nb, timeout=120, kernel_name="python3", store_widget_state=True)
    client.execute()
    assert marker.read_text() == "1"
    assert "widgets" in nb.metadata
    NotebookNotary().sign(nb)
    nbformat.write(nb, notebook)
    return notebook, marker


@pytest.mark.skipif(
    os.environ.get("QT_RUN_JUPYTER_SAVED_WIDGET_TESTS") != "1",
    reason="set QT_RUN_JUPYTER_SAVED_WIDGET_TESTS=1 to test saved widget reopen in JupyterLab",
)
def test_saved_widgets_reopen_interactive_without_cell_execution(tmp_path):
    pytest.importorskip("jupyterlab")
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    try:
        installed_anywidget = metadata.version("anywidget")
        installed = metadata.version("jupyterlab_widgets")
    except metadata.PackageNotFoundError:
        pytest.fail("anywidget and jupyterlab_widgets must be installed in the JupyterLab environment")
    assert _version_tuple(installed_anywidget) >= (0, 11, 0)
    assert _version_tuple(installed) >= (3, 0, 10)

    chrome = _chrome_executable()
    if chrome is None:
        pytest.skip("Chrome/Chromium executable not found")

    root = tmp_path / "saved-widgets"
    root.mkdir()
    notebook, marker = _write_executed_saved_widgets_notebook(root)
    port = int(os.environ.get("QT_WIDGET_REOPEN_PORT", "0")) or _free_port()
    token = "saved-widgets-token"
    profile = Path(f"/tmp/cdp-saved-widgets-{port}")
    shutil.rmtree(profile, ignore_errors=True)
    server = _start_jupyter(root, port, token)
    try:
        _wait_for_jupyter(port, token, server)
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(
                str(profile),
                executable_path=chrome,
                headless=os.environ.get("QT_WIDGET_HEADLESS") == "1",
                viewport={"width": 1200, "height": 900},
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
                page.wait_for_function(
                    """() => document.body.innerText.includes('Saved Show2D Reopen')
                      && document.body.innerText.includes('Saved Mixed-Stack Show2D Reopen')
                      && document.body.innerText.includes('Saved Show3D Reopen')
                      && document.body.innerText.includes('Saved Show3DSlices Reopen')
                      && document.querySelector('[aria-label="Frame for HAADF stack"]')
                      && document.querySelectorAll('canvas').length >= 1""",
                    timeout=180_000,
                )
                body_text = page.evaluate("document.body.innerText")
                assert "model not found" not in body_text.lower()
                stack_slider = page.locator('[aria-label="Frame for HAADF stack"]')
                assert stack_slider.get_attribute("aria-valuenow") == "2"
                stack_slider.focus()
                stack_slider.press("Home")
                page.wait_for_timeout(300)
                assert stack_slider.get_attribute("aria-valuenow") == "0"

                box = page.evaluate(
                    """() => {
                      const canvas = [...document.querySelectorAll('canvas')]
                        .map(c => ({c, r: c.getBoundingClientRect()}))
                        .filter(x => x.r.width > 50 && x.r.height > 50)[0];
                      const r = canvas.r;
                      return {x: r.x, y: r.y, w: r.width, h: r.height};
                    }"""
                )
                before = page.screenshot(full_page=False)
                page.mouse.move(box["x"] + box["w"] * 0.5, box["y"] + box["h"] * 0.5)
                page.mouse.down()
                page.mouse.move(box["x"] + box["w"] * 0.65, box["y"] + box["h"] * 0.65, steps=12)
                page.mouse.up()
                page.mouse.wheel(0, -350)
                page.wait_for_timeout(1000)
                after = page.screenshot(full_page=False)
                assert hashlib.sha256(before).hexdigest() != hashlib.sha256(after).hexdigest()
            finally:
                context.close()
    finally:
        with suppress(Exception):
            server.terminate()
            server.wait(timeout=10)
        with suppress(Exception):
            server.kill()
        shutil.rmtree(profile, ignore_errors=True)

    assert marker.read_text() == "1"
