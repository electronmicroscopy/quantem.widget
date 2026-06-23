import os
import subprocess
import time
from contextlib import suppress

import numpy as np
import pytest


pytest.importorskip("hdf5plugin")
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


def _click_copy_buttons(page):
    buttons = page.get_by_role("button", name="COPY")
    count = buttons.count()
    assert count >= 2, f"Expected DP and VI COPY buttons, found {count}"
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


@pytest.mark.skipif(
    os.environ.get("QT_RUN_BROWSER_TESTS") != "1",
    reason="set QT_RUN_BROWSER_TESTS=1 to run headed WebGPU browser smoke tests",
)
def test_webgpu_multi_volume_export_fetches_second_volume(tmp_path):
    from playwright.sync_api import sync_playwright
    from quantem.widget import Show4DSTEM

    chrome = _chrome_executable()
    if chrome is None:
        pytest.skip("Chrome/Chromium executable not found")

    data = np.zeros((2, 8, 8, 64, 64), dtype=np.uint16)
    data[0] += 3
    data[1] += 17

    out_dir = tmp_path / "webgpu-multi"
    w = Show4DSTEM(
        data,
        frame_dim_label="Dataset",
        frame_labels=["first", "second"],
        backend="web",
        offline_codec="bslz4",
        data_url=str(out_dir),
        precompute_virtual_images=False,
        verbose=False,
    )
    w.export_html(str(out_dir / "index.html"), title="small multi-volume WebGPU")

    port = int(os.environ.get("QT_BROWSER_TEST_PORT", "8898"))
    server = subprocess.Popen(
        [
            "python3",
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(out_dir),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(1.0)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                executable_path=chrome,
                headless=False,
                args=[
                    "--enable-unsafe-webgpu",
                    "--enable-features=Vulkan,WebGPU",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
            )
            context = browser.new_context(viewport={"width": 1000, "height": 800})
            context.grant_permissions(
                ["clipboard-read", "clipboard-write"],
                origin=f"http://127.0.0.1:{port}",
            )
            try:
                page = context.new_page()
                requested = []
                page.on(
                    "request",
                    lambda request: requested.append(request.url)
                    if ("vol0/" in request.url or "vol1/" in request.url)
                    else None,
                )
                page.goto(f"http://127.0.0.1:{port}/index.html", wait_until="domcontentloaded")
                page.wait_for_function(
                    "document.body.innerText.includes('first') && document.querySelectorAll('canvas').length >= 4",
                    timeout=120_000,
                )
                page.wait_for_timeout(3000)
                assert page.evaluate("!!navigator.gpu")

                dataset_slider = page.evaluate(
                    """() => [...document.querySelectorAll('.MuiSlider-root')]
                      .map((root, i) => {
                        const r = root.getBoundingClientRect();
                        return {i, rect:{x:r.x,y:r.y,w:r.width,h:r.height},
                          inputs:[...root.querySelectorAll('input')].map(inp => ({min:inp.min,max:inp.max,value:inp.value}))};
                      })
                      .find(root => root.inputs.some(inp => inp.min === '0' && inp.max === '1'))"""
                )
                assert dataset_slider is not None
                rect = dataset_slider["rect"]
                page.mouse.move(rect["x"] + 2, rect["y"] + rect["h"] / 2)
                page.mouse.down()
                page.mouse.move(rect["x"] + rect["w"] - 2, rect["y"] + rect["h"] / 2, steps=10)
                page.mouse.up()
                page.wait_for_function("document.body.innerText.includes('second')", timeout=60_000)
                page.wait_for_timeout(3000)

                assert any("vol0/" in url for url in requested)
                assert any("vol1/" in url for url in requested)
                _click_copy_buttons(page)
            finally:
                context.close()
            browser.close()
    finally:
        with suppress(Exception):
            server.terminate()
            server.wait(timeout=5)
        with suppress(Exception):
            server.kill()
