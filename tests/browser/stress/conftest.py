"""Private headed Chrome with loopback CDP for bounded native input.

The surrounding browser tier supplies the existing page/Jupyter fixtures.
This stress-only override adds a native input socket without controlling any
already running browser or changing other browser tests.
"""
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def chrome():
    from playwright.sync_api import sync_playwright

    args = ["--ignore-gpu-blocklist", "--enable-features=Vulkan",
            "--enable-unsafe-webgpu", "--disable-gpu-sandbox", "--no-first-run",
            "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0"]
    env = dict(os.environ)
    icd = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if icd.exists():
        env["VK_ICD_FILENAMES"] = str(icd)
    if sys.platform == "linux":
        # GL compositing with hardware WebGPU avoids the Vulkan compositor's
        # accumulating dma-buf handles in long, multi-canvas Chrome sessions.
        args.append("--use-angle=gl")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=False,
                                               args=args, env=env)
        try:
            connection = browser.new_browser_cdp_session()
            argv = connection.send("Browser.getBrowserCommandLine")["arguments"]
            connection.detach()
            profile = Path(next(arg.split("=", 1)[1] for arg in argv
                                if arg.startswith("--user-data-dir=")))
            port = int((profile / "DevToolsActivePort").read_text().splitlines()[0])
            browser.stress_cdp_endpoint = f"http://127.0.0.1:{port}"
            yield browser
        finally:
            browser.close()
