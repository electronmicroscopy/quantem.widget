"""Headed-Chrome fixtures for the opt-in browser tier (WebGPU on and off).

Run on a machine with a display and a GPU:

    QUANTEM_WIDGET_BROWSER=1 DISPLAY=:0 python -m pytest tests/browser -s

Chrome is pointed at the NVIDIA Vulkan driver when it exists, so WebGPU runs
on the hardware adapter instead of Chrome's bundled CPU renderer.
"""

import functools
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import nbformat
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
    """Open a page with WebGPU on or removed; returns the page and its error list (and console messages)."""
    contexts = []

    def open_(url: str, *, webgpu: bool):
        context = chrome.new_context(viewport={"width": 1600, "height": 1050}, device_scale_factor=1)
        contexts.append(context)
        if not webgpu:
            context.add_init_script(NO_WEBGPU_SCRIPT)
        page = context.new_page()
        page.errors = []
        page.on("pageerror", lambda error: page.errors.append(str(error)[:300]))
        page.console = []
        page.on("console", lambda message: page.console.append(f"{message.type}: {message.text}"))
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


class JupyterLab:
    """A private JupyterLab whose kernels run this interpreter and its sys.path."""

    def __init__(self, root: Path):
        self.root = root
        kernel_dir = root / "jupyter" / "kernels" / "quantem-browser-test"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "quantem-browser-test", "language": "python",
        }))
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.runtime = root / "runtime"
        # JupyterLab's first-run news prompt is a toast over the lower right of the page: it lands on a panel
        # screenshot at a position that differs between two runs, so the private settings answer it.
        settings = root / "settings"
        (settings / "@jupyterlab" / "apputils-extension").mkdir(parents=True)
        (settings / "@jupyterlab" / "apputils-extension" / "notification.jupyterlab-settings").write_text(json.dumps({"fetchNews": "false"}))
        # An empty config dir keeps the user's Jupyter config (save hooks, extensions) out of the run.
        env = {**os.environ, "JUPYTER_PATH": str(root / "jupyter"), "JUPYTER_RUNTIME_DIR": str(self.runtime),
               "JUPYTER_CONFIG_DIR": str(root / "config"), "JUPYTERLAB_SETTINGS_DIR": str(settings)}
        self.server = subprocess.Popen(
            [sys.executable, "-m", "jupyterlab", "--no-browser", f"--port={port}", "--ip=127.0.0.1",
             "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.disable_check_xsrf=True",
             f"--ServerApp.root_dir={root}", "--ServerApp.port_retries=0"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.url = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{self.url}/api/status")
                break
            except OSError:
                time.sleep(0.5)

    def write_notebook(self, name: str, *cells: str) -> None:
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(cell) for cell in cells])
        notebook.metadata["kernelspec"] = {"name": "quantem-browser-test", "display_name": "quantem-browser-test", "language": "python"}
        nbformat.write(notebook, self.root / name)

    def open(self, open_page, notebook: str, workspace: str, *, webgpu: bool, ready: str, out: Path):
        """Open the notebook in JupyterLab, run its first cell and wait for ``ready``."""
        page = open_page(f"{self.url}/lab/workspaces/{workspace}/tree/{notebook}", webgpu=webgpu)
        page.wait_for_selector(".jp-Notebook .jp-Cell", timeout=120_000)
        page.wait_for_timeout(3000)
        page.locator(".jp-Cell .cm-content").first.click()
        page.keyboard.press("Shift+Enter")
        page.mouse.move(5, 300)
        page.screenshot(path=str(out / f"live_{workspace}_running.png"))
        page.wait_for_selector(ready, timeout=300_000)
        page.wait_for_timeout(6000)
        return page

    def run_in_kernel(self, notebook: str, code: str) -> str:
        """Run ``code`` in the notebook's kernel and return what it printed."""
        from jupyter_client import BlockingKernelClient

        sessions = json.load(urllib.request.urlopen(f"{self.url}/api/sessions"))
        kernel_id = next(session["kernel"]["id"] for session in sessions if session["path"].endswith(notebook))
        client = BlockingKernelClient(connection_file=str(self.runtime / f"kernel-{kernel_id}.json"))
        client.load_connection_file()
        client.start_channels()
        chunks = []
        errors = []

        def capture(message):
            if message["msg_type"] == "stream":
                chunks.append(message["content"].get("text", ""))
            elif message["msg_type"] == "error":
                errors.append("\n".join(message["content"].get("traceback", [])))

        try:
            client.execute_interactive(code, timeout=120, output_hook=capture)
        finally:
            client.stop_channels()
        assert not errors, "\n".join(errors)
        return "".join(chunks).strip()

    def close(self) -> None:
        for session in json.load(urllib.request.urlopen(f"{self.url}/api/sessions")):
            urllib.request.urlopen(urllib.request.Request(f"{self.url}/api/sessions/{session['id']}", method="DELETE"))
        self.server.terminate()
        self.server.wait(timeout=30)


@pytest.fixture(scope="session")
def jupyter(tmp_path_factory):
    lab = JupyterLab(tmp_path_factory.mktemp("live"))
    yield lab
    lab.close()
