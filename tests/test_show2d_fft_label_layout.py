"""Real-browser check for non-overlapping Show2D gallery FFT labels.

The gallery FFT title and quality metrics are both enabled by default. This
opt-in check renders two local-stack panels and verifies that those two labels
remain vertically separated at desktop and narrow viewport widths.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
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


def _write_fft_gallery_notebook(root: Path) -> Path:
    code = """
import numpy as np
from quantem.widget import Show2D

y, x = np.indices((64, 64), dtype=np.float32)
base = np.sin(x * 0.55) + np.cos(y * 0.47)
stack_a = np.stack([base, np.roll(base, 2, axis=0)])
stack_b = np.stack([np.roll(base, frame, axis=1) for frame in range(3)])

Show2D(
    [stack_a, stack_b],
    labels=[
        "regularized reconstruction with eight slices",
        "alternative reconstruction with sixteen slices",
    ],
    panel_frame_indices=[1, 2],
    ncols=2,
    size=240,
    show_fft=True,
    fft_metrics=True,
    panel_title_font_size=13,
    verbose=False,
)
"""
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(code)]
    )
    path = root / "show2d_fft_label_layout.ipynb"
    nbformat.write(notebook, path)
    return path


def _assert_fft_labels_do_not_overlap(page) -> None:
    panels = page.locator("[data-show2d-fft-panel]")
    assert panels.count() >= 2, "expected at least two gallery FFT panels"
    for panel_idx in range(2):
        panel = panels.nth(panel_idx)
        title = panel.locator(".quantem-fft-panel-title")
        metrics = panel.locator(".quantem-fft-quality-label")
        title_box = title.bounding_box()
        metrics_box = metrics.bounding_box()
        panel_box = panel.bounding_box()
        assert title_box is not None and metrics_box is not None and panel_box is not None
        title_bottom = title_box["y"] + title_box["height"]
        assert metrics_box["y"] >= title_bottom + 1, (
            f"FFT metrics overlap panel {panel_idx} title: "
            f"title={title_box}, metrics={metrics_box}"
        )
        assert metrics_box["y"] + metrics_box["height"] <= (
            panel_box["y"] + panel_box["height"]
        )


def _assert_image_and_fft_panels_touch(page) -> None:
    images = page.locator("[data-show2d-image-panel]")
    ffts = page.locator("[data-show2d-fft-panel]")
    assert images.count() >= 2 and ffts.count() >= 2
    for panel_idx in range(2):
        image_box = images.nth(panel_idx).bounding_box()
        fft_box = ffts.nth(panel_idx).bounding_box()
        assert image_box is not None and fft_box is not None
        gap = fft_box["y"] - (image_box["y"] + image_box["height"])
        assert abs(gap) <= 0.5, (
            f"image and FFT panel {panel_idx} should touch vertically, got "
            f"gap={gap}px; image={image_box}, fft={fft_box}"
        )


def _assert_fft_zoom_labels(page, expected: str = "2.0×") -> None:
    panels = page.locator("[data-show2d-fft-panel]")
    zoom_labels = page.locator("[data-show2d-fft-zoom-indicator]")
    # C1: two interactive gallery FFTs, expect an accurate contained badge in
    # each panel, including Show2D's default 2.0× view.
    assert zoom_labels.count() >= 2
    for panel_idx in range(2):
        panel_box = panels.nth(panel_idx).bounding_box()
        label = zoom_labels.nth(panel_idx)
        label_box = label.bounding_box()
        assert panel_box is not None and label_box is not None
        assert label.inner_text() == expected
        assert label_box["x"] >= panel_box["x"]
        assert label_box["y"] >= panel_box["y"]
        assert label_box["x"] + label_box["width"] <= panel_box["x"] + panel_box["width"]
        assert label_box["y"] + label_box["height"] <= panel_box["y"] + panel_box["height"]


@pytest.mark.skipif(
    os.environ.get("QT_RUN_WIDGET_VISUAL_TESTS") != "1",
    reason="set QT_RUN_WIDGET_VISUAL_TESTS=1 to run headed Jupyter widget visual tests",
)
def test_show2d_gallery_fft_title_and_metrics_do_not_overlap(tmp_path) -> None:
    from playwright.sync_api import sync_playwright

    chrome = _chrome_executable()
    if chrome is None:
        pytest.skip("Chrome/Chromium executable not found")

    root = tmp_path / "show2d-fft-label-layout"
    root.mkdir()
    notebook = _write_fft_gallery_notebook(root)
    port = _free_port()
    token = "show2d-fft-label-layout-token"
    profile = Path(f"/tmp/cdp-show2d-fft-label-layout-{port}")
    shutil.rmtree(profile, ignore_errors=True)
    server = _start_jupyter(root, port, token)
    try:
        _wait_for_jupyter(port, token, server)
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
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
                    wait_until="domcontentloaded",
                    timeout=120_000,
                )
                page.wait_for_selector(".jp-Notebook", timeout=120_000)
                page.wait_for_timeout(3000)
                page.locator(".jp-Cell").first.click()
                page.keyboard.press("Shift+Enter")
                page.wait_for_selector(
                    ".quantem-fft-quality-label",
                    state="visible",
                    timeout=120_000,
                )
                page.wait_for_timeout(1000)

                _assert_fft_labels_do_not_overlap(page)
                _assert_image_and_fft_panels_touch(page)
                _assert_fft_zoom_labels(page)
                page.set_viewport_size({"width": 390, "height": 844})
                page.wait_for_timeout(500)
                _assert_fft_labels_do_not_overlap(page)
                _assert_image_and_fft_panels_touch(page)
                _assert_fft_zoom_labels(page)
            finally:
                context.close()
    finally:
        try:
            server.terminate()
            server.wait(timeout=10)
        except Exception:
            server.kill()
        shutil.rmtree(profile, ignore_errors=True)
