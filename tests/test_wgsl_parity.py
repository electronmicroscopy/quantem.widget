"""WGSL/WebGPU compute parity: the widget's browser BF/DF/CoM kernels vs a numpy reference.

The widget owns the entire WebGPU compute layer (js/engine/compute.ts: maskedSum -> virtual
image, maskedCoM -> CoM/DPC). The Python torch path is covered by test_dpc_virtual_parity.py;
this is the missing leg - it proves the WGSL output matches numpy on a deterministic fixture.

WGSL only runs on a real GPU in a browser, so this drives a headed Chrome over CDP and calls
the web app's `window.__wgslParity(scanCount, detRows, detCols)` hook. The fixture is a pure
index function (value = (s*31 + d*17) % 251) so numpy reproduces the exact bytes the JS builds.

Skips cleanly when google-chrome, websockets, or the built web dist are absent (CI without a GPU).
Run on the CUDA box: pytest tests/test_wgsl_parity.py -v
"""
import json
import shutil
import socket
import subprocess
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

websockets = pytest.importorskip("websockets")
import asyncio
import urllib.request

DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
CHROME = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
NVIDIA_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"

pytestmark = [
    pytest.mark.skipif(CHROME is None, reason="google-chrome not installed"),
    pytest.mark.skipif(not (DIST / "index.html").exists(), reason="web app not built (run: cd web && npx vite build)"),
]

SCAN, DET_ROWS, DET_COLS = 64, 32, 32


def _numpy_reference():
    """The SAME deterministic stack + BF mask the JS hook builds, plus numpy BF sum & CoM.

    value(s, d) = (s*31 + d*17) % 251 with d the row-major detector index; this is a pure
    function of indices so JS and numpy produce byte-identical input - any mismatch is a real
    kernel bug, not RNG drift.
    """
    det_size = DET_ROWS * DET_COLS
    s = np.arange(SCAN)[:, None]
    d = np.arange(det_size)[None, :]
    stack = ((s * 31 + d * 17) % 251).astype(np.uint8).reshape(SCAN, DET_ROWS, DET_COLS).astype(np.float64)
    cy, cx, radius = (DET_ROWS - 1) / 2, (DET_COLS - 1) / 2, min(DET_ROWS, DET_COLS) * 0.25
    rows = np.arange(DET_ROWS)[:, None]
    cols = np.arange(DET_COLS)[None, :]
    mask = ((rows - cy) ** 2 + (cols - cx) ** 2 <= radius * radius).astype(np.float64)
    intensity = stack * mask
    virtual = intensity.sum(axis=(1, 2))
    denom = intensity.sum(axis=(1, 2))
    com_y = (intensity * rows).sum(axis=(1, 2)) / denom
    com_x = (intensity * cols).sum(axis=(1, 2)) / denom
    return virtual, com_y, com_x


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _run_wgsl(cdp_port):
    """Connect to Chrome, call window.__wgslParity, return the parsed result dict."""
    tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json", timeout=5).read())
    page = next(t for t in tabs if t["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=2**28) as ws:
        mid = 0

        async def cmd(method, params=None):
            nonlocal mid
            mid += 1
            await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == mid:
                    return msg

        await cmd("Runtime.enable")
        expr = f"window.__wgslParity({SCAN},{DET_ROWS},{DET_COLS})"
        res = await cmd("Runtime.evaluate", {"expression": expr, "awaitPromise": True, "returnByValue": True})
        return res.get("result", {}).get("result", {}).get("value")


@pytest.fixture(scope="module")
def wgsl_result():
    """Serve the built web app, launch headed Chrome on the real GPU, call the parity hook."""
    port = _free_port()
    handler = partial(SimpleHTTPRequestHandler, directory=str(DIST))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cdp_port = _free_port()
    profile = f"/tmp/cdp-wgsl-parity-{cdp_port}"
    import os
    env = dict(os.environ)
    if Path(NVIDIA_ICD).exists():
        env["VK_ICD_FILENAMES"] = NVIDIA_ICD  # force the real NVIDIA Vulkan device, never SwiftShader
    env.setdefault("DISPLAY", ":1")
    chrome = subprocess.Popen(
        [CHROME, f"--remote-debugging-port={cdp_port}", f"--user-data-dir={profile}",
         "--no-first-run", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--use-angle=vulkan",
         "--enable-unsafe-webgpu", "--disable-gpu-sandbox", "--window-size=900,700",
         f"http://127.0.0.1:{port}/index.html"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):  # wait for CDP + page + the hook to mount
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json/version", timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        time.sleep(4)
        result = None
        for _ in range(5):
            result = asyncio.new_event_loop().run_until_complete(_run_wgsl(cdp_port))
            if result and "error" not in result:
                break
            time.sleep(2)
        yield result
    finally:
        chrome.terminate()
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)


def test_wgsl_compute_available(wgsl_result):
    assert wgsl_result is not None, "no response from __wgslParity (Chrome/CDP failed)"
    if isinstance(wgsl_result, dict) and wgsl_result.get("error"):
        pytest.skip(f"WebGPU unavailable in this Chrome: {wgsl_result['error']}")
    assert wgsl_result["scanCount"] == SCAN


def test_wgsl_masked_sum_matches_numpy(wgsl_result):
    """Virtual image (BF mask sum) - integer sum, must be effectively exact in f32."""
    if not isinstance(wgsl_result, dict) or wgsl_result.get("error"):
        pytest.skip("WebGPU unavailable")
    virtual_ref, _, _ = _numpy_reference()
    virtual_wgsl = np.array(wgsl_result["virtual"], dtype=np.float64)
    np.testing.assert_allclose(virtual_wgsl, virtual_ref, rtol=1e-5, atol=1.0)


def test_wgsl_com_matches_numpy(wgsl_result):
    """CoM (intensity-weighted centroid in detector px) - f32 division, tight tolerance."""
    if not isinstance(wgsl_result, dict) or wgsl_result.get("error"):
        pytest.skip("WebGPU unavailable")
    _, com_y_ref, com_x_ref = _numpy_reference()
    com_y = np.array(wgsl_result["comY"], dtype=np.float64)
    com_x = np.array(wgsl_result["comX"], dtype=np.float64)
    np.testing.assert_allclose(com_y, com_y_ref, atol=1e-3)
    np.testing.assert_allclose(com_x, com_x_ref, atol=1e-3)
