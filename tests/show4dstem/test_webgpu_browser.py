import os
import subprocess
import time
from contextlib import suppress

import numpy as np
import pytest

from quantem.widget import Show4DSTEM


pytest.importorskip("hdf5plugin")
sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright


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


def _exercise_dpc_backend(page, expected_pixels):
    page.wait_for_function(
        "window.__sh4d && typeof window.__sh4d.dpcOnly === 'function' && typeof window.__sh4d.dpcBufferOnly === 'function' && document.body.innerText.includes('DPC')",
        timeout=60_000,
    )
    warm = page.evaluate("window.__sh4d.dpcOnly()")
    assert warm["length"] == expected_pixels
    for source in ("DPC_row", "DPC_col"):
        result = page.evaluate(
            """async ({source, expectedPixels}) => {
              const api = window.__sh4d;
              try {
                api.model.set("vi_source", source);
                await api.recomputeVI();
                for (let attempt = 0; attempt < 120; attempt++) {
                  await new Promise(resolve => requestAnimationFrame(resolve));
                  const display = window.__sh4dDpcDisplay || {};
                  const view = api.model.get("virtual_image_bytes");
                  const len = view ? (view.byteLength / 4) : 0;
                  if (display.source === source && display.rendered === true && len === expectedPixels) {
                    break;
                  }
                }
              } catch (error) {
                return {
                  source: api.model.get("vi_source"),
                  length: -1,
                  expectedPixels,
                  error: error && error.message ? error.message : String(error),
                  display: window.__sh4dDpcDisplay || {},
                };
              }
              const view = api.model.get("virtual_image_bytes");
              const arr = new Float32Array(view.buffer, view.byteOffset, view.byteLength / 4);
              let sum = 0;
              for (let i = 0; i < arr.length; i++) sum += arr[i];
              const display = window.__sh4dDpcDisplay || {};
              return {
                source: api.model.get("vi_source"),
                length: arr.length,
                expectedPixels,
                sum,
                displaySource: display.source,
                gpuBufferToDisplay: display.gpuBufferToDisplay === true,
                rendered: display.rendered === true,
                display,
                error: null,
              };
            }""",
            {"source": source, "expectedPixels": expected_pixels},
        )
        assert result["error"] is None, result
        assert result["source"] == source
        assert result["length"] == expected_pixels
        assert result["displaySource"] == source, result
        assert result["gpuBufferToDisplay"] is True, result
        assert result["rendered"] is True, result


def _exercise_roi_backend(page, expected_pixels):
    page.wait_for_function(
        "window.__sh4d && typeof window.__sh4d.roiBufferOnly === 'function' && document.body.innerText.includes('ROI')",
        timeout=60_000,
    )
    warm = page.evaluate("window.__sh4d.roiBufferOnly()")
    assert warm["available"] is True, warm
    assert warm["displayed"] is True, warm
    assert warm["length"] == expected_pixels
    result = page.evaluate(
        """async ({expectedPixels}) => {
          const api = window.__sh4d;
          try {
            api.model.set("vi_source", "roi");
            await api.recomputeVI();
            for (let attempt = 0; attempt < 120; attempt++) {
              await new Promise(resolve => requestAnimationFrame(resolve));
              const display = window.__sh4dViDisplay || {};
              const view = api.model.get("virtual_image_bytes");
              const len = view ? (view.byteLength / 4) : 0;
              if (display.source === "roi" && display.rendered === true && len === expectedPixels) {
                break;
              }
            }
          } catch (error) {
            return {
              source: api.model.get("vi_source"),
              length: -1,
              expectedPixels,
              error: error && error.message ? error.message : String(error),
              display: window.__sh4dViDisplay || {},
            };
          }
          const view = api.model.get("virtual_image_bytes");
          const arr = new Float32Array(view.buffer, view.byteOffset, view.byteLength / 4);
          let sum = 0;
          for (let i = 0; i < arr.length; i++) sum += arr[i];
          const display = window.__sh4dViDisplay || {};
          return {
            source: api.model.get("vi_source"),
            length: arr.length,
            expectedPixels,
            sum,
            displaySource: display.source,
            gpuBufferToDisplay: display.gpuBufferToDisplay === true,
            rendered: display.rendered === true,
            rangeMode: display.rangeMode,
            display,
            error: null,
          };
        }""",
        {"expectedPixels": expected_pixels},
    )
    assert result["error"] is None, result
    assert result["source"] == "roi"
    assert result["length"] == expected_pixels
    assert result["displaySource"] == "roi"
    assert result["gpuBufferToDisplay"] is True, result
    assert result["rendered"] is True, result


@pytest.mark.skipif(
    os.environ.get("QT_RUN_BROWSER_TESTS") != "1",
    reason="set QT_RUN_BROWSER_TESTS=1 to run headed WebGPU browser smoke tests",
)
def test_webgpu_multi_volume_export_fetches_second_volume(tmp_path):

    chrome = _chrome_executable()
    if chrome is None:
        pytest.skip("Chrome/Chromium executable not found")

    data = np.zeros((2, 8, 8, 64, 64), dtype=np.uint16)
    data[0] += 3
    data[1] += 17

    out_dir = tmp_path / "webgpu-multi-data"
    html_path = tmp_path / "index.html"
    w = Show4DSTEM(
        data,
        frame_dim_label="Dataset",
        frame_labels=["first", "second"],
        backend="webgpu",
        offline_codec="bslz4",
        data_url=str(out_dir),
        precompute_virtual_images=False,
        verbose=False,
    )
    w.export_html(str(html_path), title="small multi-volume WebGPU")

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
            str(tmp_path),
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
                    "--use-angle=vulkan",
                    "--ignore-gpu-blocklist",
                    "--disable-software-rasterizer",
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
                page.goto(
                    f"http://127.0.0.1:{port}/index.html",
                    wait_until="domcontentloaded",
                )
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
                slider = page.locator(".MuiSlider-root").nth(int(dataset_slider["i"]))
                slider.scroll_into_view_if_needed()
                page.wait_for_timeout(250)
                rect = slider.bounding_box()
                assert rect is not None
                page.mouse.move(rect["x"] + 2, rect["y"] + rect["height"] / 2)
                page.mouse.down()
                page.mouse.move(
                    rect["x"] + rect["width"] - 2,
                    rect["y"] + rect["height"] / 2,
                    steps=10,
                )
                page.mouse.up()
                page.wait_for_function("document.body.innerText.includes('second')", timeout=60_000)
                page.wait_for_timeout(3000)

                assert any("vol0/" in url for url in requested)
                assert any("vol1/" in url for url in requested)
                _exercise_roi_backend(page, 8 * 8)
                _exercise_dpc_backend(page, 8 * 8)
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
