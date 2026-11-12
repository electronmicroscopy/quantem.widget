"""Own a headed Chrome whose Playwright transport can detach during timing."""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def chrome():
    from playwright.sync_api import sync_playwright

    args = ["--ignore-gpu-blocklist", "--enable-features=Vulkan",
            "--enable-unsafe-webgpu", "--disable-gpu-sandbox", "--no-first-run",
            "--window-size=1900,2000", "--force-device-scale-factor=1",
            "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0"]
    env = dict(os.environ)
    icd = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if icd.exists():
        env["VK_ICD_FILENAMES"] = str(icd)
    if sys.platform == "linux":
        # GL compositing with hardware WebGPU avoids the Vulkan compositor's
        # accumulating dma-buf handles in long, multi-canvas Chrome sessions.
        args.append("--use-angle=gl")
    executable = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
    if not executable and sys.platform == "darwin":
        executable = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    assert executable and Path(executable).is_file(), "Install Google Chrome for headed stress tests."
    with tempfile.TemporaryDirectory(prefix="quantem-stress-chrome-") as directory:
        profile = Path(directory)
        process = subprocess.Popen([executable, *args, f"--user-data-dir={profile}", "about:blank"],
                                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        browser = None
        try:
            port_file = profile / "DevToolsActivePort"
            for _ in range(100):
                if port_file.exists() or process.poll() is not None:
                    break
                time.sleep(0.1)
            assert port_file.exists(), "Private Chrome did not expose its loopback debugging port."
            port = int(port_file.read_text().splitlines()[0])
            endpoint = f"http://127.0.0.1:{port}"
            with sync_playwright() as playwright:
                state = SimpleNamespace(stress_cdp_endpoint=endpoint, errors=[], console=[])

                def connect():
                    nonlocal browser
                    if browser is None or not browser.is_connected():
                        browser = playwright.chromium.connect_over_cdp(endpoint)
                    state.browser = browser
                    return browser

                def observe(page):
                    page.errors, page.console = state.errors, state.console
                    page.on("pageerror", lambda error: state.errors.append(str(error)[:300]))
                    page.on("console", lambda message: state.console.append(f"{message.type}: {message.text}"))
                    return page

                def reconnect(target_id):
                    for page in connect().contexts[0].pages:
                        session = page.context.new_cdp_session(page)
                        identity = session.send("Target.getTargetInfo")["targetInfo"]["targetId"]
                        session.detach()
                        if identity == target_id:
                            return observe(page)
                    raise RuntimeError("The owned stress-test page disappeared while Playwright was detached.")

                # Closing a connect_over_cdp handle disconnects; this externally
                # launched browser and its default context remain alive.
                state.disconnect = lambda: browser.close()
                state.reconnect = reconnect
                state.observe = observe
                connect()
                try:
                    yield state
                finally:
                    if browser.is_connected():
                        browser.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.fixture
def open_page(chrome):
    def open_(url, *, webgpu):
        assert webgpu, "These throughput tests require hardware WebGPU."
        # Incognito contexts created by Playwright disappear on disconnect.
        page = chrome.observe(chrome.browser.contexts[0].new_page())
        page.set_viewport_size({"width": 1600, "height": 1050})
        page.goto(url)
        return page

    return open_
