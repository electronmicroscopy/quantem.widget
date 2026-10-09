"""Headed-Chrome fixtures for the opt-in browser tier (WebGPU on and off).

Run on a machine with a display and a GPU:

    QUANTEM_WIDGET_BROWSER=1 DISPLAY=:0 python -m pytest tests/browser -s

Chrome is pointed at the NVIDIA Vulkan driver when it exists, so WebGPU runs
on the hardware adapter instead of Chrome's bundled CPU renderer.
"""

import functools
import http.server
import os
import threading
from pathlib import Path

import pytest

NVIDIA_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"
CHROME_ARGS = [
    "--ignore-gpu-blocklist",
    "--enable-features=Vulkan",
    "--enable-unsafe-webgpu",
    "--disable-gpu-sandbox",
    "--no-first-run",
    "--window-size=1700,1100",
]
# Removing the accessor makes a widget see a browser without WebGPU.
NO_WEBGPU_SCRIPT = (
    "Object.defineProperty(Navigator.prototype, 'gpu', {get: () => undefined, configurable: true});"
)
ADAPTER_INFO = """
async () => {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
  return adapter ? {vendor: adapter.info.vendor, architecture: adapter.info.architecture} : null;
}
"""
RENDER_PATHS = "() => Array.from(document.querySelectorAll('[data-render-path]')).map(e => e.dataset.renderPath)"


def pytest_configure(config):
    config.addinivalue_line("markers", "browser: headed Chrome with WebGPU on and off (opt-in)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("QUANTEM_WIDGET_BROWSER") == "1":
        return
    skip = pytest.mark.skip(reason="opt-in: set QUANTEM_WIDGET_BROWSER=1 (headed Chrome and a GPU)")
    for item in items:
        if "browser" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def browser_out(tmp_path_factory) -> Path:
    """Screenshots and JSON summaries; QUANTEM_WIDGET_BROWSER_OUT keeps them."""
    out = Path(os.environ.get("QUANTEM_WIDGET_BROWSER_OUT") or tmp_path_factory.mktemp("browser"))
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture(scope="session")
def chrome():
    from playwright.sync_api import sync_playwright

    env = dict(os.environ)
    if os.path.exists(NVIDIA_ICD):
        env["VK_ICD_FILENAMES"] = NVIDIA_ICD
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=False, args=CHROME_ARGS, env=env)
        yield browser
        browser.close()


@pytest.fixture
def open_page(chrome):
    """Open a page with WebGPU on or removed; returns the page and its error list."""
    contexts = []

    def open_(url: str, *, webgpu: bool):
        context = chrome.new_context(viewport={"width": 1600, "height": 1050}, device_scale_factor=1)
        contexts.append(context)
        if not webgpu:
            context.add_init_script(NO_WEBGPU_SCRIPT)
        page = context.new_page()
        page.errors = []
        page.on("pageerror", lambda error: page.errors.append(str(error)[:300]))
        page.goto(url)
        page.adapter = lambda: page.evaluate(ADAPTER_INFO)
        page.render_paths = lambda: page.evaluate(RENDER_PATHS)
        return page

    yield open_
    for context in contexts:
        context.close()


@pytest.fixture(scope="session")
def serve():
    """Serve a directory over HTTP (exports fetch their data; file:// blocks it)."""
    servers = []

    def start(directory: Path) -> str:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
        handler.log_message = lambda *args: None
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server in servers:
        server.shutdown()


@pytest.fixture
def hardware_adapter():
    """Assert a WebGPU run is on a hardware adapter, not Chrome's CPU renderer."""

    def check(adapter) -> None:
        assert adapter is not None, "WebGPU run found no adapter"
        assert adapter["architecture"] != "swiftshader", f"WebGPU run is on Chrome's CPU renderer: {adapter}"

    return check
