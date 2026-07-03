"""Real-browser check: the gallery star button is hover-revealed.

Renders a 3-panel Show2D gallery in JupyterLab and asserts the star button's
contract: hidden at rest (opacity 0), revealed on panel hover (opacity 1),
and a starred panel keeps its star visible after the pointer leaves.
Opt-in like the other visual tests: QT_RUN_WIDGET_VISUAL_TESTS=1.
"""

from __future__ import annotations

import os
import shutil
import importlib.util
from pathlib import Path

import nbformat
import pytest

pytest.importorskip("jupyterlab")
pytest.importorskip("playwright.sync_api")

_VISUAL_HELPERS_PATH = Path(__file__).with_name("test_widget_visual_jupyter.py")
_VISUAL_HELPERS_SPEC = importlib.util.spec_from_file_location(
    "quantem_widget_visual_helpers",
    _VISUAL_HELPERS_PATH,
)
if _VISUAL_HELPERS_SPEC is None or _VISUAL_HELPERS_SPEC.loader is None:
    raise ImportError(f"Cannot load visual test helpers from {_VISUAL_HELPERS_PATH}")
_VISUAL_HELPERS = importlib.util.module_from_spec(_VISUAL_HELPERS_SPEC)
_VISUAL_HELPERS_SPEC.loader.exec_module(_VISUAL_HELPERS)

_chrome_executable = _VISUAL_HELPERS._chrome_executable
_free_port = _VISUAL_HELPERS._free_port
_start_jupyter = _VISUAL_HELPERS._start_jupyter
_wait_for_jupyter = _VISUAL_HELPERS._wait_for_jupyter


def _write_gallery_notebook(root: Path) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import numpy as np\n"
            "from quantem.widget import Show2D\n"
            "rng = np.random.default_rng(0)\n"
            "Show2D(rng.random((3, 64, 64)), labels=['a', 'b', 'c'], title='Star Hover Gallery')\n"
        )
    ]
    path = root / "star_hover.ipynb"
    nbformat.write(nb, path)
    return path


@pytest.mark.skipif(
    os.environ.get("QT_RUN_WIDGET_VISUAL_TESTS") != "1",
    reason="set QT_RUN_WIDGET_VISUAL_TESTS=1 to run headed Jupyter widget visual tests",
)
def test_show2d_star_button_hover_reveal(tmp_path):
    from playwright.sync_api import sync_playwright

    chrome = _chrome_executable()
    if chrome is None:
        pytest.skip("Chrome/Chromium executable not found")

    root = tmp_path / "star-hover"
    root.mkdir()
    notebook = _write_gallery_notebook(root)
    port = _free_port()
    token = "star-hover-token"
    profile = Path(f"/tmp/cdp-star-hover-{port}")
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
                args=["--no-first-run", "--no-default-browser-check"],
            )
            try:
                page = context.pages[0] if context.pages else context.new_page()
                page.goto(
                    f"http://127.0.0.1:{port}/lab/tree/{notebook.name}?token={token}",
                    wait_until="domcontentloaded", timeout=120_000,
                )
                page.wait_for_selector(".jp-Notebook", timeout=120_000)
                page.wait_for_timeout(3000)  # let the kernel connect before executing
                page.locator(".jp-Cell").first.click()
                page.keyboard.press("Shift+Enter")
                # the star buttons themselves are the render signal we care about
                page.wait_for_selector(".show2d-panel-star-button", state="attached", timeout=120_000)
                page.wait_for_timeout(1500)

                stars = page.locator(".show2d-panel-star-button")
                assert stars.count() >= 3, "star buttons missing from gallery panels"
                first_star = stars.first
                opacity_at_rest = first_star.evaluate("el => getComputedStyle(el).opacity")
                assert float(opacity_at_rest) == 0.0, (
                    f"star should be hidden at rest, opacity={opacity_at_rest}")

                # Hover the first panel -> its star fades in.
                first_canvas = page.locator("canvas").first
                first_canvas.hover()
                page.wait_for_timeout(400)  # 120 ms transition + margin
                opacity_hover = first_star.evaluate("el => getComputedStyle(el).opacity")
                assert float(opacity_hover) == 1.0, (
                    f"star should be visible on panel hover, opacity={opacity_hover}")

                # Star the panel, move the pointer far away -> gold star stays visible.
                first_star.click()
                page.mouse.move(5, 5)
                page.wait_for_timeout(400)
                opacity_starred = first_star.evaluate("el => getComputedStyle(el).opacity")
                assert float(opacity_starred) == 1.0, (
                    f"starred panel must keep its star visible, opacity={opacity_starred}")
                assert first_star.inner_text().strip() == "★"
            finally:
                context.close()
    finally:
        try:
            server.terminate()
            server.wait(timeout=10)
        except Exception:
            server.kill()
        shutil.rmtree(profile, ignore_errors=True)
