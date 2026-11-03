#!/usr/bin/env python3
"""Measure ShowPtycho and Show4DSTEM interactivity in a real browser and compare it with this chip's baseline.

``run`` starts its own JupyterLab and headed Chrome (private ports, settings
and profile, never a browser you are using), opens one notebook per widget on
the public gold scan (``bobleesj/quantem-data``, ``4dstem/gold_512``), drives
them with scripted pointer events over the Chrome DevTools protocol, writes a
JSON report and stops everything it started. ``compare`` checks a report
against the committed baseline for the same chip and exits 1 when a metric
got worse than the tolerance allows::

    npm run build
    python scripts/bench_interactivity.py run --out interactivity.json
    python scripts/bench_interactivity.py compare interactivity.json
    python scripts/bench_interactivity.py run --save-baseline      # re-record this chip's baseline
    python scripts/bench_interactivity.py merge a.json b.json c.json --save-baseline   # from several runs

Before any timing the page's WebGPU adapter is logged; the run stops when it
is Chrome's CPU renderer (SwiftShader), whose timings say nothing about the
GPU. All timing happens inside the page (``performance.now``): CDP only
injects pointer events, at 120 Hz.

Measured, per drag:

- ShowPtycho, C10 slider dragged back and forth at native sampling and at 2x:
  phase and FFT panel paints per second, and the request latency from the
  browser's ``ssb-preview`` message to the kernel's phase for the same id;
- Show4DSTEM on the gold scan as loaded (its uint32 counts fit 16 bits once
  the flagged pixels are replaced, so quantem.gpu stores them as the uint16
  count resident): the probe dragged across the scan, the bright-field disk
  dragged, and the annular detector's outer radius resized: kernel updates
  per second and the pointer-to-paint latency p50 / p95.

Rates are capped by the 60 Hz display. Run on a quiet GPU (no other Metal
work, nothing streaming the screen) with the window visible; the report
lists the busiest other GPU clients so a slow run can be explained.
"""

import argparse
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "scripts" / "baselines"
DATASET = "bobleesj/quantem-data"
REVISION = "5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20"
MASTER = "4dstem/gold_512/gold_004_master.h5"
VIEWPORT = {"width": 1512, "height": 982, "deviceScaleFactor": 2, "mobile": False}
MOVE_HZ = 120
DEFAULT_TOLERANCE = 0.20
# A p95 of a 3 to 6 s drag rests on a handful of samples and moves by a third between quiet runs.
TAIL_TOLERANCE = 0.50
# A change must also exceed these absolute amounts to count: a 60 Hz display quantizes rates and latencies.
MIN_DELTA = {"per_s": 2.0, "ms": 3.0}
CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
)

# Gold calibration recorded with the dataset (the tutorials use the same values).
SETUP = """
import os
import numpy as np
import torch
from quantem.gpu import io
data = io.load(os.environ["QW_BENCH_MASTER"], hot_pixel_correction="median", verbose=False)
data.calibration.semiangle_mrad = 30
data.calibration.scan_sampling_A = 0.5
data.calibration.detector_sampling_mrad = 0.46
"""
PTYCHO_CELL = """
from quantem.gpu import SSB
# A fitted gold calibration, so the panel opens on a focused image without a 30 s search.
ssb = SSB(data, aberrations={"C10_nm": -23.24, "C12_nm": 3.42, "phi12_deg": -40.2}, scan_detector_rotation_deg=358.31)
viewer = ssb.show(show_fft=True)
viewer
"""
COUNTS_CELL = """
from quantem.widget import Show4DSTEM
w = Show4DSTEM(data)
w
"""

INSTRUMENT = r"""
(() => {
  if (window.__qb) return;
  const B = window.__qb = {presents: [], frames: [], sends: [], recvs: [], ptr: [], nid: 0};
  const idOf = (c) => { if (!c.__qbid) c.__qbid = ++B.nid; return c.__qbid; };
  B.idOf = idOf;
  B.reset = () => { for (const k of ["presents", "frames", "sends", "recvs", "ptr"]) B[k].length = 0; };
  if (window.GPUCanvasContext) {
    const current = GPUCanvasContext.prototype.getCurrentTexture;
    GPUCanvasContext.prototype.getCurrentTexture = function () { B.presents.push([performance.now(), idOf(this.canvas)]); return current.call(this); };
  }
  for (const name of ["putImageData", "drawImage"]) {
    const paint = CanvasRenderingContext2D.prototype[name];
    CanvasRenderingContext2D.prototype[name] = function (...args) {
      if (this.canvas && this.canvas.isConnected) B.presents.push([performance.now(), idOf(this.canvas)]);
      return paint.apply(this, args);
    };
  }
  // ShowPtycho announces each painted panel with a 'quantem-frame' event on that panel's canvas.
  const dispatch = EventTarget.prototype.dispatchEvent;
  EventTarget.prototype.dispatchEvent = function (event) {
    if (event && event.type === "quantem-frame" && this instanceof HTMLCanvasElement) B.frames.push([performance.now(), idOf(this)]);
    return dispatch.call(this, event);
  };
  const decode = (data) => {
    try {
      let message, bytes;
      if (typeof data === "string") { bytes = data.length; message = JSON.parse(data); }
      else {
        const buffer = data instanceof ArrayBuffer ? data : data.buffer;
        bytes = buffer.byteLength;
        const view = new DataView(buffer);
        const count = Number(view.getBigUint64(0, true));
        const offsets = [];
        for (let i = 0; i < count; i++) offsets.push(Number(view.getBigUint64(8 * (i + 1), true)));
        offsets.push(buffer.byteLength);
        const text = new TextDecoder();
        const part = (i) => text.decode(new Uint8Array(buffer, offsets[i], offsets[i + 1] - offsets[i]));
        message = {header: JSON.parse(part(1)), content: JSON.parse(part(4))};
      }
      if (!message.header || message.header.msg_type !== "comm_msg") return null;
      const body = (message.content || {}).data || {};
      const state = body.state || {};
      const keys = [...Object.keys(state), ...((body.buffer_paths || []).map(path => path.join(".")))];
      let id = null;
      if (body.method === "custom" && body.content && body.content.request) id = body.content.request.id;
      if (state.result_json) { try { id = JSON.parse(state.result_json).id; } catch (error) {} }
      return [bytes, keys.join(","), id];
    } catch (error) { return null; }
  };
  const Socket = window.WebSocket;
  const Wrapped = function (...args) {
    const socket = new Socket(...args);
    socket.addEventListener("message", (event) => { const info = decode(event.data); if (info) B.recvs.push([performance.now(), ...info]); });
    const send = socket.send.bind(socket);
    socket.send = (data) => { const info = decode(data); if (info) B.sends.push([performance.now(), ...info]); return send(data); };
    return socket;
  };
  Wrapped.prototype = Socket.prototype;
  Object.assign(Wrapped, {CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3});
  window.WebSocket = Wrapped;
  for (const type of ["pointermove", "pointerdown"]) {
    window.addEventListener(type, (event) => { if (event.isTrusted) B.ptr.push(performance.now()); }, {capture: true, passive: true});
  }
})();
"""
DISPLAY_HZ = """new Promise(resolve => { let frames = 0; const start = performance.now();
  const tick = () => { frames++; if (performance.now() - start < 1000) requestAnimationFrame(tick);
    else resolve(frames * 1000 / (performance.now() - start)); }; requestAnimationFrame(tick); })"""
ADAPTER = """(async () => {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({powerPreference: "high-performance"});
  if (!adapter) return null;
  const info = adapter.info;
  return {vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
          subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize, isFallbackAdapter: info.isFallbackAdapter};
})()"""


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))] if ordered else None


def rounded(value, digits=1):
    return None if value is None else round(float(value), digits)


class CDP:
    """A minimal synchronous Chrome DevTools client for one page."""

    def __init__(self, port):
        import websocket

        targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
        page = next(target for target in targets if target["type"] == "page")
        self.socket = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=300, max_size=None,
                                                  suppress_origin=True)
        self.next_id = 0

    def send(self, method, **params):
        self.next_id += 1
        self.socket.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        return self.next_id

    def call(self, method, **params):
        wanted = self.send(method, **params)
        while True:
            message = json.loads(self.socket.recv())
            if message.get("method") == "Page.javascriptDialogOpening":
                # a "leave page?" prompt would block every later command
                self.send("Page.handleJavaScriptDialog", accept=True)
            if message.get("id") == wanted:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    def js(self, expression, timeout=None):
        result = self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in result:
            raise RuntimeError(f"page script failed: {json.dumps(result['exceptionDetails'])[:800]}")
        return result.get("result", {}).get("value")

    def wait(self, expression, timeout=240, what=None):
        stop = time.time() + timeout
        while time.time() < stop:
            value = self.js(expression)
            if value:
                return value
            time.sleep(0.25)
        raise TimeoutError(f"timed out waiting for {what or expression}")

    def mouse(self, kind, x, y, pressed=False):
        params = {"type": kind, "x": x, "y": y, "buttons": 1 if pressed else 0,
                  "button": "left" if kind != "mouseMoved" or pressed else "none"}
        if kind in ("mousePressed", "mouseReleased"):
            params["clickCount"] = 1
        return self.call("Input.dispatchMouseEvent", **params)

    def drag(self, start, points, settle=1.5):
        """Press at ``start``, move through ``points`` at MOVE_HZ, release; return the page times of the drag."""
        self.js("window.__qb.reset(), 1")
        self.mouse("mouseMoved", *start)
        time.sleep(0.05)
        began = self.js("performance.now()")
        self.mouse("mousePressed", *start, pressed=True)
        clock = time.perf_counter()
        for step, (x, y) in enumerate(points, 1):
            self.send("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y, buttons=1, button="left")
            delay = clock + step / MOVE_HZ - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        ended = self.js("performance.now()")
        self.mouse("mouseReleased", *points[-1])
        time.sleep(settle)
        return began, ended, self.js("(() => { const B = window.__qb; return {presents: B.presents, frames: B.frames, "
                                     "sends: B.sends, recvs: B.recvs, ptr: B.ptr}; })()")

    def close(self):
        self.socket.close()


class Session:
    """Private JupyterLab and Chrome processes, stopped and deleted by ``close``."""

    def __init__(self, master, chrome):
        self.root = Path(tempfile.mkdtemp(prefix="quantem-widget-interactivity-"))
        self.processes = []
        kernels = self.root / "jupyter" / "kernels" / "quantem-interactivity"
        kernels.mkdir(parents=True)
        (kernels / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "quantem-interactivity", "language": "python"}))
        settings = self.root / "settings" / "@jupyterlab" / "apputils-extension"
        settings.mkdir(parents=True)
        (settings / "notification.jupyterlab-settings").write_text(json.dumps({"fetchNews": "false"}))
        self.runtime = self.root / "runtime"
        self.lab_port = free_port()
        env = {**os.environ, "JUPYTER_PATH": str(self.root / "jupyter"), "JUPYTER_RUNTIME_DIR": str(self.runtime),
               "JUPYTER_CONFIG_DIR": str(self.root / "config"), "JUPYTERLAB_SETTINGS_DIR": str(self.root / "settings"),
               "QW_BENCH_MASTER": str(master), "QW_STATIC_FALLBACK": "0"}
        self.lab = f"http://127.0.0.1:{self.lab_port}"
        self.processes.append(subprocess.Popen(
            [sys.executable, "-m", "jupyterlab", "--no-browser", f"--port={self.lab_port}", "--ip=127.0.0.1",
             "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.disable_check_xsrf=True",
             f"--ServerApp.root_dir={self.root}", "--ServerApp.port_retries=0", "--LabApp.expose_app_in_browser=True"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))
        for _ in range(240):
            try:
                urllib.request.urlopen(f"{self.lab}/api/status")
                break
            except OSError:
                time.sleep(0.25)
        else:
            raise RuntimeError("JupyterLab did not start")
        self.cdp_port = free_port()
        self.processes.append(subprocess.Popen(
            [chrome, f"--remote-debugging-port={self.cdp_port}", f"--user-data-dir={self.root / 'chrome'}",
             "--no-first-run", "--no-default-browser-check", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
             "--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding",
             "--disable-background-timer-throttling", "--window-size=1512,1100",
             # On the main display: a second display refreshing at another rate changes every
             # animation-frame-paced number (input coalescing, display-capped rates).
             "--window-position=40,40", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.cdp_port}/json/version")
                break
            except OSError:
                time.sleep(0.25)
        else:
            raise RuntimeError("Chrome did not open its DevTools port")
        self.page = CDP(self.cdp_port)
        self.page.call("Page.enable")
        self.page.call("Runtime.enable")
        self.page.call("Emulation.setDeviceMetricsOverride", **VIEWPORT)
        self.page.call("Page.addScriptToEvaluateOnNewDocument", source=INSTRUMENT)

    def write_notebook(self, name, *cells):
        import nbformat

        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(cell.strip()) for cell in cells])
        notebook.metadata["kernelspec"] = {"name": "quantem-interactivity", "display_name": "quantem-interactivity",
                                           "language": "python"}
        nbformat.write(notebook, self.root / name)

    def open(self, name, ready, timeout=240):
        """Open ``name`` in a fresh workspace, run all its cells and wait until the page expression ``ready`` holds."""
        workspace = Path(name).stem
        self.page.call("Page.navigate", url=f"{self.lab}/lab/workspaces/{workspace}/tree/{name}?reset")
        # the previous document answers until the navigation commits, and evaluating during it fails
        loaded = (f"location.href.includes('/workspaces/{workspace}/') && document.readyState === 'complete'"
                  " && !!window.jupyterapp")
        for _ in range(240):
            try:
                if self.page.js(loaded):
                    break
            except RuntimeError:
                pass
            time.sleep(0.25)
        panel = (f"(() => {{ const app = window.jupyterapp; if (!app) return null; for (const w of app.shell.widgets('main'))"
                 f" if (w.sessionContext && w.context && w.context.path === '{name}') return w; return null; }})()")
        self.page.wait(f"(() => {{ const w = {panel}; const k = w && w.sessionContext.session && w.sessionContext.session.kernel;"
                       " return !!k && k.status === 'idle'; })()", timeout=120, what=f"an idle kernel for {name}")
        self.page.js(f"(() => {{ const app = window.jupyterapp; app.shell.activateById({panel}.id); app.shell.collapseLeft();"
                     " return app.commands.execute('notebook:run-all-cells').then(() => 1); })()")
        self.page.wait(ready, timeout=timeout, what=f"the widget in {name}")
        errors = self.page.js("[...document.querySelectorAll('.jp-OutputArea-output[data-mime-type="
                              "\"application/vnd.jupyter.stderr\"], .jp-RenderedText[data-mime-type="
                              "\"application/vnd.jupyter.error\"]')].map(e => e.innerText.slice(0, 400))")
        return errors

    def kernel(self, notebook, code):
        """Run ``code`` in the kernel of ``notebook`` and return what it printed."""
        from jupyter_client import BlockingKernelClient

        sessions = json.load(urllib.request.urlopen(f"{self.lab}/api/sessions"))
        kernel_id = next(entry["kernel"]["id"] for entry in sessions if entry["path"].endswith(notebook))
        client = BlockingKernelClient(connection_file=str(self.runtime / f"kernel-{kernel_id}.json"))
        client.load_connection_file()
        client.start_channels()
        printed, errors = [], []

        def capture(message):
            if message["msg_type"] == "stream":
                printed.append(message["content"].get("text", ""))
            elif message["msg_type"] == "error":
                errors.append("\n".join(message["content"].get("traceback", [])))

        try:
            client.execute_interactive(code, timeout=120, output_hook=capture)
        finally:
            client.stop_channels()
        if errors:
            raise RuntimeError(errors[0][-1500:])
        return "".join(printed).strip()

    def shutdown_kernels(self):
        for entry in json.load(urllib.request.urlopen(f"{self.lab}/api/sessions")):
            urllib.request.urlopen(urllib.request.Request(f"{self.lab}/api/sessions/{entry['id']}", method="DELETE"))

    def close(self):
        try:
            self.page.close()
        except Exception as error:  # noqa: BLE001 - the page may already be gone; the processes still have to stop
            print(f"closing the DevTools connection: {error}")
        try:
            self.shutdown_kernels()
        except OSError:
            pass
        for process in reversed(self.processes):
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        shutil.rmtree(self.root, ignore_errors=True)


def rect(page, selector):
    return page.js(f"(() => {{ const e = {selector}; if (!e) return null; e.scrollIntoView({{block: 'center'}});"
                   " const r = e.getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height,"
                   " id: window.__qb.idOf(e)}; })()")


def ptycho_drag(page, seconds):
    """One C10 thumb drag: panel paint rates and request latencies inside it."""
    thumb = ("(() => { for (const root of document.querySelectorAll('.MuiSlider-root')) {"
             " const p = root.parentElement; if (p && p.textContent.trim().startsWith('C10 '))"
             " return root.querySelector(\".MuiSlider-thumb[data-index='1']\") || root.querySelector('.MuiSlider-thumb'); }"
             " return null; })()")
    box = rect(page, thumb)
    if box is None:
        raise RuntimeError("ShowPtycho's C10 slider was not found")
    time.sleep(0.5)
    cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
    count = int(seconds * MOVE_HZ)
    points = [(cx + 30 * math.sin(2 * math.pi * step / (1.5 * MOVE_HZ)), cy) for step in range(count)]
    began, ended, log = page.drag((cx, cy), points)
    low, high = began + 500, ended
    by_canvas = {}
    for stamp, canvas in log["frames"]:
        if low <= stamp <= high:
            by_canvas.setdefault(canvas, []).append(stamp)
    canvases = sorted(by_canvas)
    window_s = (high - low) / 1e3
    sent = {entry[3]: entry[0] for entry in log["sends"] if entry[3] is not None and low <= entry[0] <= high}
    replied = {}
    for stamp, _bytes, _keys, ident in log["recvs"]:
        if ident in sent and ident not in replied:
            replied[ident] = stamp - sent[ident]
    latency = list(replied.values())
    return {
        "phase_fps": rounded(len(by_canvas[canvases[0]]) / window_s) if canvases else 0.0,
        "fft_fps": rounded(len(by_canvas[canvases[1]]) / window_s) if len(canvases) > 1 else 0.0,
        "requests_per_s": rounded(len(sent) / window_s),
        "request_latency_p50_ms": rounded(statistics.median(latency)) if latency else None,
        "request_latency_p95_ms": rounded(percentile(latency, 0.95)),
    }


def detector_drag(page, name, start, points, key, canvas):
    """Kernel updates per second and pointer-to-paint latency of one Show4DSTEM drag."""
    began, ended, log = page.drag(start, points)
    paints = sorted(stamp for stamp, ident in log["presents"] if ident == canvas)
    updates = [entry for entry in log["recvs"] if key in (entry[2] or "")]
    sends = sorted(entry[0] for entry in log["sends"])
    pointer = sorted(log["ptr"])
    latency = []
    for received, *_rest in updates:
        before = [stamp for stamp in sends if stamp <= received]
        after = [stamp for stamp in paints if stamp >= received]
        if not before or not after:
            continue
        cause = [stamp for stamp in pointer if stamp <= before[-1]]
        if cause:
            latency.append(after[0] - cause[-1])
    seconds = (ended - began) / 1e3
    inside = [entry for entry in updates if began <= entry[0] <= ended]
    if not inside:
        keys = sorted({entry[2] for entry in log["recvs"] if entry[2]})
        target = page.js(f"(() => {{ const e = document.elementFromPoint({start[0]}, {start[1]});"
                         " return e ? e.tagName + ' ' + (e.dataset.quantemScientificOutput || e.className || '') : null; })()")
        print(f"{name}: no {key} update during the drag; the press landed on {target}; messages seen: {keys}")
    return {
        "updates_per_s": rounded(len(inside) / seconds),
        "paints_per_s": rounded(len([stamp for stamp in paints if began <= stamp <= ended]) / seconds),
        "latency_p50_ms": rounded(statistics.median(latency)) if latency else None,
        "latency_p95_ms": rounded(percentile(latency, 0.95)),
        "bytes_per_update": int(statistics.median(entry[1] for entry in inside)) if inside else 0,
    }


def drive_ptycho(session, seconds, log):
    session.write_notebook("showptycho.ipynb", SETUP, PTYCHO_CELL)
    errors = session.open("showptycho.ipynb", "window.__qb.frames.length > 0", timeout=240)
    if errors:
        log(f"notebook output: {errors}")
    time.sleep(2.0)
    results = {}
    for factor in (1, 2):
        # set explicitly: "auto" may choose 2x on its own
        session.kernel("showptycho.ipynb", f"viewer.upsampling_factor = {factor}")
        time.sleep(3.0)
        label = "native" if factor == 1 else f"{factor}x"
        results[label] = ptycho_drag(session.page, seconds)
        log(f"ShowPtycho C10 drag {label}: {results[label]}")
    session.shutdown_kernels()
    return results


def drive_show4dstem(session, seconds, log):
    session.write_notebook("show4dstem.ipynb", SETUP, COUNTS_CELL)
    ready = ("!!document.querySelector('canvas[data-quantem-scientific-output=\"show4dstem-diffraction-pattern\"]') && "
             "!!document.querySelector('canvas[data-quantem-scientific-output=\"show4dstem-virtual-image\"]') && "
             "window.__qb.presents.length > 2")
    errors = session.open("show4dstem.ipynb", ready, timeout=300)
    if errors:
        log(f"notebook output: {errors}")
    time.sleep(3.0)
    page = session.page
    info = json.loads(session.kernel("show4dstem.ipynb", "import json; print(json.dumps(dict(det=[w.det_rows, w.det_cols], "
                                     "scan=[w.shape_rows, w.shape_cols], roi=[w.roi_center_row, w.roi_center_col, "
                                     "w.roi_radius])))"))
    panels = {}

    def locate():
        # the two panels sit side by side: scrolling one into view places both
        panels["vi"] = rect(page, "document.querySelector('canvas[data-quantem-scientific-output=\"show4dstem-virtual-image\"]')")
        panels["dp"] = rect(page, "document.querySelector('canvas[data-quantem-scientific-output=\"show4dstem-diffraction-pattern\"]')")
        return panels["vi"], panels["dp"]

    vi, dp = locate()

    def on_dp(row, col):
        dp = panels["dp"]
        return dp["x"] + (col + 0.5) * dp["w"] / info["det"][1], dp["y"] + (row + 0.5) * dp["h"] / info["det"][0]

    def on_vi(row, col):
        vi = panels["vi"]
        return vi["x"] + (col + 0.5) * vi["w"] / info["scan"][1], vi["y"] + (row + 0.5) * vi["h"] / info["scan"][0]

    count = int(seconds * MOVE_HZ)
    rows, cols = info["scan"]
    results = {}
    circle = [on_vi(rows / 2 + 0.3 * rows * math.sin(2 * math.pi * i / count),
                    cols / 2 + 0.3 * cols * math.cos(2 * math.pi * i / count)) for i in range(count)]
    results["probe"] = detector_drag(page, "probe", on_vi(rows / 2, cols / 2 + 0.3 * cols), circle, "frame_bytes",
                                     dp["id"])
    log(f"Show4DSTEM probe drag: {results['probe']}")
    session.kernel("show4dstem.ipynb", "w.apply_preset('bf')")
    time.sleep(1.0)
    vi, dp = locate()
    row, col = json.loads(session.kernel("show4dstem.ipynb", "print(json.dumps([w.roi_center_row, w.roi_center_col]))"))
    disk = [on_dp(row + 12 * math.sin(2 * math.pi * i / count), col + 12 * (1 - math.cos(2 * math.pi * i / count)))
            for i in range(count)]
    results["bf_drag"] = detector_drag(page, "bf", on_dp(row, col), disk, "virtual_image_bytes", vi["id"])
    log(f"Show4DSTEM BF drag: {results['bf_drag']}")
    session.kernel("show4dstem.ipynb", "w.apply_preset('adf')")
    time.sleep(1.0)
    vi, dp = locate()
    row, col, outer, inner = json.loads(session.kernel(
        "show4dstem.ipynb", "print(json.dumps([w.roi_center_row, w.roi_center_col, w.roi_radius, w.roi_radius_inner]))"))
    # grab the annulus between its edges, nearer the outer one, and pull the outer radius in and out
    grip = (outer + inner) / 2 + 0.3 * (outer - inner)
    ring = [on_dp(row - (grip - 15 * math.sin(math.pi * i / count)) / math.sqrt(2),
                  col - (grip - 15 * math.sin(math.pi * i / count)) / math.sqrt(2)) for i in range(count)]
    results["adf_resize"] = detector_drag(page, "adf", on_dp(row - grip / math.sqrt(2), col - grip / math.sqrt(2)),
                                          ring, "virtual_image_bytes", vi["id"])
    log(f"Show4DSTEM ADF resize: {results['adf_resize']}")
    session.shutdown_kernels()
    return results


def environment():
    def sysctl(name):
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False).stdout.strip()

    import quantem.gpu

    import quantem.widget

    def commit(package):
        root = Path(package.__file__).resolve().parents[3]
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=False)
        return result.stdout.strip()[:12] or None

    return {"chip": sysctl("machdep.cpu.brand_string") or platform.processor(),
            "macos": platform.mac_ver()[0], "python": platform.python_version(),
            "quantem_widget_commit": commit(quantem.widget), "quantem_gpu_commit": commit(quantem.gpu)}


def gpu_clients(seconds=2.0):
    """Other processes' share of the Apple GPU over ``seconds`` (IOAccelerator statistics), busiest first."""
    def snapshot():
        text = subprocess.run(["ioreg", "-l", "-w0"], capture_output=True, text=True, check=False).stdout
        totals = {}
        for block in text.split("+-o "):
            creator = re.search(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"', block)
            times = re.findall(r'"accumulatedGPUTime"=(\d+)', block)
            if creator and times:
                key = (creator.group(2), int(creator.group(1)))
                totals[key] = totals.get(key, 0) + sum(int(value) for value in times)
        return totals

    first = snapshot()
    time.sleep(seconds)
    second = snapshot()
    shares = [{"command": name, "gpu_percent": round(100 * (total - first.get((name, pid), 0)) / (seconds * 1e9), 1)}
              for (name, pid), total in second.items()]
    return sorted([entry for entry in shares if entry["gpu_percent"] >= 1], key=lambda entry: -entry["gpu_percent"])


def gold_master(download):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        root = snapshot_download(DATASET, repo_type="dataset", revision=REVISION, allow_patterns="4dstem/gold_512/*",
                                 local_files_only=not download)
    except LocalEntryNotFoundError:
        return None
    master = Path(root) / MASTER
    return master if master.is_file() else None


def find_chrome(explicit):
    for candidate in ([explicit] if explicit else []) + [os.environ.get("CHROME")] + list(CHROME_CANDIDATES):
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return candidate if Path(candidate).is_file() else shutil.which(candidate)
    return None


def flatten(report):
    """``metric -> value`` for every compared number: ``*_per_s`` and ``*_fps`` higher is better, ``*_ms`` lower."""
    flat = {}
    for widget in ("showptycho", "show4dstem"):
        for scenario, values in report.get(widget, {}).items():
            for name, value in values.items():
                if value is not None and (name.endswith(("_fps", "_per_s", "_ms"))):
                    flat[f"{widget}.{scenario}.{name}"] = value
    return flat


def regressions(report, baseline, overrides=None):
    overrides = overrides or {}
    rows, failed = [], []
    current = flatten(report)
    for name, old in flatten(baseline).items():
        if name.split(".")[0] not in report:
            continue  # a run with --widgets that left this widget out
        new = current.get(name)
        tolerance = overrides.get(name, TAIL_TOLERANCE if name.endswith("_p95_ms") else DEFAULT_TOLERANCE)
        if new is None:
            rows.append((name, old, None, tolerance, "MISSING"))
            failed.append(name)
            continue
        lower_is_better = name.endswith("_ms")
        worse = (new - old) if lower_is_better else (old - new)
        slack = MIN_DELTA["ms" if lower_is_better else "per_s"]
        if worse > tolerance * abs(old) and worse > slack:
            verdict = "WORSE"
            failed.append(name)
        elif -worse > tolerance * abs(old) and -worse > slack:
            verdict = "better (re-record the baseline?)"
        else:
            verdict = "ok"
        rows.append((name, old, new, tolerance, verdict))
    return rows, failed


def baseline_path(chip):
    return BASELINES / f"interactivity-{re.sub(r'[^a-z0-9]+', '-', chip.lower()).strip('-')}.json"


def run(arguments):
    master = gold_master(arguments.download)
    if master is None:
        print(f"The gold scan is not cached; pass --download to fetch {MASTER} (about 4.6 GB) from {DATASET}.")
        return 2
    chrome = find_chrome(arguments.chrome)
    if chrome is None:
        print("No Chrome or Chromium found; pass --chrome or set CHROME.")
        return 2
    if not (ROOT / "src" / "quantem" / "widget" / "static" / "showptycho.js").is_file():
        print("The JS bundle is missing; run `npm ci && npm run build` first.")
        return 2
    report = {"schema": "quantem.widget.interactivity-benchmark/1",
              "dataset": {"id": DATASET, "revision": REVISION, "file": MASTER},
              "environment": environment(), "viewport": VIEWPORT, "drag_seconds": arguments.seconds,
              "other_gpu_clients_before": gpu_clients()}
    print(f"chip {report['environment']['chip']}; other GPU clients: {report['other_gpu_clients_before']}")
    session = Session(master, chrome)
    try:
        session.page.call("Page.navigate", url=f"{session.lab}/lab")
        session.page.wait("!!window.jupyterapp", what="JupyterLab")
        adapter = session.page.js(ADAPTER)
        report["adapter"] = adapter
        report["display_hz"] = rounded(session.page.js(DISPLAY_HZ))
        print(f"display refresh: {report['display_hz']} Hz")
        print(f"WebGPU adapter: {adapter}")
        if (not adapter or adapter.get("architecture") == "swiftshader" or adapter.get("vendor") == "google"
                or adapter.get("isFallbackAdapter")):
            print("This is Chrome's CPU renderer (SwiftShader) or no adapter: browser timings would not describe "
                  "the GPU. Aborting.")
            return 3
        if "showptycho" in arguments.widgets:
            report["showptycho"] = drive_ptycho(session, arguments.seconds, print)
        if "show4dstem" in arguments.widgets:
            report["show4dstem"] = drive_show4dstem(session, min(arguments.seconds, 3.0), print)
    finally:
        session.close()
    report["other_gpu_clients_after"] = gpu_clients()
    rendered = json.dumps(report, indent=1) + "\n"
    targets = ([Path(arguments.out)] if arguments.out else []) + (
        [baseline_path(report["environment"]["chip"])] if arguments.save_baseline else [])
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered)
        print(f"wrote {target.name}")
    if not targets:
        print(rendered, end="")
    return 0


def compare(arguments):
    report = json.loads(Path(arguments.report).read_text())
    chip = report["environment"]["chip"]
    path = Path(arguments.baseline) if arguments.baseline else baseline_path(chip)
    if not path.is_file():
        print(f"No baseline for {chip} ({path.name}); record one on a quiet GPU with `run --save-baseline`.")
        return 2
    baseline = json.loads(path.read_text())
    overrides = {}
    for item in arguments.tolerance or []:
        name, _, value = item.partition("=")
        overrides[name] = float(value)
    rows, failed = regressions(report, baseline, overrides)
    print("| metric | baseline | current | tolerance | verdict |")
    print("|---|---:|---:|---:|---|")
    for name, old, new, tolerance, verdict in rows:
        print(f"| {name} | {old:.1f} | {'-' if new is None else f'{new:.1f}'} | {100 * tolerance:.0f} % | {verdict} |")
    rates = [data.get("display_hz") for data in (baseline, report)]
    if all(rates) and abs(rates[0] - rates[1]) > 0.1 * rates[0]:
        print(f"warning: the baseline ran on a {rates[0]:.0f} Hz display and this run on {rates[1]:.0f} Hz; "
              "rates and latencies paced by animation frames are not comparable.")
    for side, data in (("baseline", baseline), ("current", report)):
        busiest = {}
        for entry in (data.get("other_gpu_clients_before") or []) + (data.get("other_gpu_clients_after") or []):
            busiest[entry["command"]] = max(busiest.get(entry["command"], 0), entry["gpu_percent"])
        busy = [f"{command} up to {share:.0f} %" for command, share in busiest.items() if share >= 10]
        if busy:
            print(f"warning: other processes used the GPU around the {side} run ({', '.join(busy)}); "
                  "its numbers may include their work.")
    if failed:
        print(f"WORSE: {', '.join(failed)}")
        return 1
    print("No metric got worse beyond its tolerance.")
    return 0


def merge(arguments):
    """One report whose every metric is the median of the runs (the better middle value for an even count)."""
    reports = [json.loads(Path(path).read_text()) for path in arguments.reports]
    chips = {report["environment"]["chip"] for report in reports}
    if len(chips) != 1:
        print(f"The reports come from different chips: {sorted(chips)}")
        return 2
    merged = json.loads(json.dumps(reports[0]))
    for widget in ("showptycho", "show4dstem"):
        for scenario, values in merged.get(widget, {}).items():
            for name in values:
                if not name.endswith(("_fps", "_per_s", "_ms")):
                    continue
                runs = [report[widget][scenario][name] for report in reports
                        if report.get(widget, {}).get(scenario, {}).get(name) is not None]
                runs.sort(reverse=not name.endswith("_ms"))
                values[name] = runs[(len(runs) - 1) // 2] if runs else None
    merged["merged_runs"] = len(reports)
    for side in ("other_gpu_clients_before", "other_gpu_clients_after"):
        merged[side] = [client for report in reports for client in report.get(side) or []]
    target = baseline_path(chips.pop()) if arguments.save_baseline else Path(arguments.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, indent=1) + "\n")
    print(f"wrote {target.name} from {len(reports)} runs")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("run", help="drive both widgets and write a JSON report")
    measure.add_argument("--out", help="report path")
    measure.add_argument("--save-baseline", action="store_true", help="also write scripts/baselines/interactivity-<chip>.json")
    measure.add_argument("--seconds", type=float, default=6.0, help="length of each ShowPtycho drag (Show4DSTEM: at most 3)")
    measure.add_argument("--chrome", help="Chrome or Chromium executable")
    measure.add_argument("--widgets", default="showptycho,show4dstem", help="comma-separated widgets to drive")
    measure.add_argument("--download", action="store_true", help="fetch the gold scan if it is not cached")
    measure.set_defaults(handler=run)
    contrast = commands.add_parser("compare", help="exit 1 when a metric got worse than the baseline")
    contrast.add_argument("report")
    contrast.add_argument("--baseline", help="baseline JSON (default: this chip's committed baseline)")
    contrast.add_argument("--tolerance", action="append", metavar="METRIC=FRACTION",
                          help=f"per-metric change allowed (default {DEFAULT_TOLERANCE:.2f}; p95 latencies "
                               f"{TAIL_TOLERANCE:.2f})")
    contrast.set_defaults(handler=compare)
    combine = commands.add_parser("merge", help="combine several runs into one report, metric by metric")
    combine.add_argument("reports", nargs="+")
    target = combine.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", help="merged report path")
    target.add_argument("--save-baseline", action="store_true", help="write this chip's baseline")
    combine.set_defaults(handler=merge)
    arguments = parser.parse_args()
    sys.exit(arguments.handler(arguments))


if __name__ == "__main__":
    main()
