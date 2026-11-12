"""Opt-in native-resolution detector dragging across seven and ten recordings."""

import json
import math
import os
import statistics
import urllib.request
from itertools import pairwise
from pathlib import Path

import pytest
from native_pointer import _Connection, drag

# io.load of several paths returns a list of acquisitions; each owns GPU memory.
KERNEL_CLEANUP = """
if 'viewer' in globals(): viewer.close()
if 'loaded' in globals():
    for acquisition in loaded: acquisition.close()
"""
DP = '[data-quantem-scientific-output="show4dstem-diffraction-pattern"]'
VI = '[data-quantem-scientific-output="show4dstem-compare-0"]'
ADAPTER = """async () => {
  const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
  return adapter && {vendor: adapter.info.vendor, architecture: adapter.info.architecture,
    fallback: adapter.info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? false};
}"""
REPORT = """
import hashlib, json, os
import cupy as cp
print(json.dumps({
    "pid": os.getpid(),
    "gpu_pci_bus_id": cp.cuda.Device().pci_bus_id,
    "scan": [viewer.shape_rows, viewer.shape_cols],
    "detector": list(loaded[0].shape[-2:]),
    "panels": list(viewer.compare_panel_indices),
    "center": list(viewer.roi_center),
    "radii": [viewer.roi_radius, viewer.roi_radius_inner],
    "bytes": len(viewer.compare_virtual_image_bytes),
    "sha256": hashlib.sha256(viewer.compare_virtual_image_bytes).hexdigest(),
}))
"""


def _report(jupyter, notebook):
    return json.loads(jupyter.run_in_kernel(notebook, REPORT).splitlines()[-1])


def _summary(evidence, count, started, ended):
    """Count complete revisions and their distinct GPU-ready rAF timestamps."""
    revisions = {}
    completed = {}
    expected = {f"show4dstem-compare-{index}" for index in range(count)}
    for item in sorted(evidence["presentations"], key=lambda item: item["ready"]):
        if started <= item["queued"] <= item["ready"] <= ended:
            targets = revisions.setdefault(item["revision"], set())
            targets.update(item["targets"])
            if targets >= expected:
                completed.setdefault(item["revision"], item["ready"])
    complete = len(completed)
    refreshes = len(set(completed.values()))
    frames = [stamp for stamp in evidence["frames"] if started <= stamp <= ended]
    gaps = [right - left for left, right in pairwise(frames)]
    return {
        "seconds": (ended - started) / 1000,
        "new_complete_image_revisions": complete,
        "new_image_hz": complete * 1000 / (ended - started),
        "gpu_ready_refreshes": refreshes,
        "gpu_ready_refresh_hz": refreshes * 1000 / (ended - started),
        "raf_hz_during_drag": len(frames) * 1000 / (ended - started),
        "median_frame_interval_ms": statistics.median(gaps) if gaps else None,
        "p95_frame_interval_ms": (statistics.quantiles(gaps, n=100, method="inclusive")[94]
                                  if len(gaps) > 1 else gaps[0] if gaps else None),
        "trusted_pointer_events": sum(item["trusted"] for item in evidence["inputs"]
                                      if started <= item["at"] <= ended),
        "delivered_input_hz": sum(started <= item["at"] <= ended and item["trusted"]
                                  for item in evidence["inputs"]) * 1000 / (ended - started),
    }


def test_ready_refreshes_require_complete_panels_and_deduplicate_revisions():
    panels = [f"show4dstem-compare-{index}" for index in range(7)]
    evidence = {"frames": [], "inputs": [], "presentations": [
        {"queued": 10, "ready": 20, "revision": 1, "targets": panels},
        {"queued": 12, "ready": 20, "revision": 2, "targets": panels},
        {"queued": 30, "ready": 40, "revision": 3, "targets": panels[:-1]},
        {"queued": 50, "ready": 60, "revision": 3, "targets": panels[-1:]},
        {"queued": 70, "ready": 80, "revision": 1, "targets": panels},
        {"queued": 70, "ready": 80, "revision": 4, "targets": panels[:-1]},
    ]}
    result = _summary(evidence, 7, 0, 100)
    assert result["new_complete_image_revisions"] == 3
    assert result["new_image_hz"] == 30
    assert result["gpu_ready_refreshes"] == 2
    assert result["gpu_ready_refresh_hz"] == 20


def test_kernel_cleanup_closes_every_loaded_acquisition():
    """The cleanup cell runs in the stress kernel; check it on CPU here."""
    class Closable:
        closed = False

        def close(self):
            self.closed = True

    loaded = [Closable(), Closable()]
    viewer = Closable()
    exec(KERNEL_CLEANUP, {"loaded": loaded, "viewer": viewer})
    assert viewer.closed and all(acquisition.closed for acquisition in loaded)
    exec(KERNEL_CLEANUP, {})  # startup failed before either name existed


@pytest.mark.browser
@pytest.mark.skipif(not os.getenv("QUANTEM_WIDGET_STRESS_QEM_DIR"),
                    reason="set QUANTEM_WIDGET_STRESS_QEM_DIR to real .qem recordings")
@pytest.mark.parametrize("count", [7, 10])
@pytest.mark.parametrize("preset,gesture", [("bf", "move"), ("df", "move"), ("adf", "move"), ("adf", "resize"), ("large-adf", "move")])
def test_native_qem_detector_drag(count, preset, gesture, chrome, jupyter, open_page, hardware_adapter, browser_out):
    root = Path(os.environ["QUANTEM_WIDGET_STRESS_QEM_DIR"]).expanduser()
    paths = sorted({path.resolve() for path in root.glob("*.qem")})[:count]
    assert len(paths) == count, f"Provide {count} independent .qem recordings in the fixture directory"
    target = float(os.getenv("QUANTEM_WIDGET_STRESS_TARGET_HZ", "120"))
    assert math.isfinite(target) and target > 0
    device = int(os.getenv("QUANTEM_WIDGET_STRESS_GPU", "0"))
    name = f"qem_{count}_{preset}_{gesture}"
    notebook = f"stress_{name}.ipynb"
    jupyter.write_notebook(notebook, f"""
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "0")
from quantem.gpu import io
from quantem.widget import Show4DSTEM
import time
loaded = io.load({[str(path) for path in paths]!r}, device={device}, verbose=False)
viewer = Show4DSTEM(loaded, view_mode="multiple", compare_group_mode="all",
    compare_max_panels={count}, compare_columns=5, compare_grid_width_px=1200,
    panel_width_px=320, sync_resize=False, show_fft=False)
query_times = []
query = viewer._compare_images
def timed_query(*args, **kwargs):
    started = time.perf_counter()
    result = query(*args, **kwargs)
    query_times.append(time.perf_counter() - started)
    return result
viewer._compare_images = timed_query
viewer
""")
    instrument = Path(__file__).with_name("image_updates.js").read_text()

    def instrumented_page(url, *, webgpu):
        page = open_page("about:blank", webgpu=webgpu)
        page.set_viewport_size({"width": 1900, "height": 1800})
        page.add_init_script(instrument)
        page.goto(url)
        return page

    native_connection = None
    try:
        page = jupyter.open(instrumented_page, notebook, f"stress_{count}_{preset}_{gesture}", webgpu=True,
                            ready=VI, out=browser_out)
        adapter = page.evaluate(ADAPTER)
        hardware_adapter(adapter)
        assert not adapter["fallback"], f"Hardware adapter required: {adapter}"
        page.get_by_text({"large-adf": "ADF", "df": "BF"}.get(preset, preset.upper()), exact=True).click()
        if preset == "df":
            jupyter.run_in_kernel(notebook, "viewer.roi_center = [viewer.roi_center_row, min(180.0, viewer.roi_center_col + 40.0)]")
        if preset == "large-adf":
            jupyter.run_in_kernel(notebook, "with viewer.hold_sync(), viewer.hold_trait_notifications():\n    viewer.roi_radius = 88.0\n    viewer.roi_radius_inner = 48.0")
        page.wait_for_timeout(500)
        before = _report(jupyter, notebook)
        assert before["scan"] == [512, 512] and before["detector"] == [192, 192]
        assert before["panels"] == list(range(count))
        assert before["bytes"] == count * 512 * 512 * 4
        before_png = page.locator(VI).first.screenshot(path=str(browser_out / f"{name}_before.png"))
        page.locator(".jp-OutputArea").first.scroll_into_view_if_needed()
        panels_visible = page.locator('[data-quantem-scientific-output^="show4dstem-compare-"]').evaluate_all(
            "elements => elements.filter(element => { const box = element.getBoundingClientRect(); "
            "return box.width > 0 && box.height > 0 && box.top >= 0 && box.bottom <= innerHeight; }).length")
        assert panels_visible == count, "All comparison panels must be visible during the drag"
        refresh = page.evaluate("""async () => {
          const times = [];
          await new Promise(resolve => { function tick(now) {
            times.push(now); if (times.length < 61) requestAnimationFrame(tick); else resolve();
          } requestAnimationFrame(tick); });
          return times.slice(1).map((now, index) => now - times[index]);
        }""")
        page.evaluate("""options => Object.assign(scientificStress, {
          active: true, frameBytes: 512*512*4, stackBytes: options.bytes, count: options.count,
        })""", {"bytes": before["bytes"], "count": count})
        connection = page.context.new_cdp_session(page)
        target_id = connection.send("Target.getTargetInfo")["targetInfo"]["targetId"]
        connection.detach()
        targets = json.load(urllib.request.urlopen(chrome.stress_cdp_endpoint + "/json/list", timeout=10))
        target_info = next(item for item in targets if item["id"] == target_id)
        native_connection = _Connection(target_info["webSocketDebuggerUrl"], timeout=10)
        profile = os.getenv("QUANTEM_WIDGET_STRESS_PROFILE") == "1"
        if profile:
            # This native session never enables Network: browser profiling must
            # not base64-copy every scientific WebSocket payload into Python.
            native_connection.call("Profiler.enable", {}, 10)
            native_connection.call("Profiler.start", {}, 10)
            jupyter.run_in_kernel(notebook, "import cProfile; kernel_profile = cProfile.Profile(); kernel_profile.enable()")
        held = {}
        held_png = None
        held_upload = None

        def while_pressed():
            nonlocal page, held, held_png, held_upload
            # Drain and evidence collection are excluded from the five-second
            # requested input interval and its fresh scientific-image count.
            page = chrome.reconnect(target_id)
            page.wait_for_timeout(800)
            held_png = page.locator(VI).first.screenshot(
                path=str(browser_out / f"{name}_held.png"))
            for _ in range(30):
                held = _report(jupyter, notebook)
                held_upload = page.evaluate("scientificStressUploadHash()")
                if held_upload == {key: held[key] for key in ("bytes", "sha256")}:
                    break
                page.wait_for_timeout(100)
            else:
                (browser_out / f"{name}_held_failure.json").write_text(json.dumps({
                    "held": held, "uploaded": held_upload, "errors": page.errors, "console": page.console,
                    "evidence": page.evaluate("structuredClone(scientificStress)"),
                }))
                raise AssertionError("The held detector's full native image bytes did not reach every panel")

        start = ((before["center"][1] + 0.5) / 192, (before["center"][0] + 0.5) / 192)
        trajectory = None
        if gesture == "resize":
            # Grab the outer ring, not a center or an inner boundary.
            radius = before["radii"][0]
            direction = -1 if before["center"][1] + radius >= 191 else 1
            center = start
            start = (center[0] + direction * radius / 192, center[1])
            def trajectory(elapsed):
                moving = max(before["radii"][1] + 1, radius + 15 * math.sin(8 * elapsed))
                return center[0] + direction * moving / 192, center[1]
        # Playwright enables Network on its own CDP sessions and copies every
        # binary WebSocket frame, even with no websocket listener. Disconnect
        # it during measurement; native input does not enable that domain.
        chrome.disconnect()
        # Detaching may clear Playwright's emulated viewport. Measure the real
        # layout after it settles, then aim input at the actual detector panel.
        native_viewport = native_connection.call("Runtime.evaluate", {
            "expression": """(async () => {
              await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
              const rect = element => {
                const {x, y, width, height} = element.getBoundingClientRect();
                return {x, y, width, height};
              };
              const visible = box => box.width > 0 && box.height > 0 && box.x >= 0 && box.y >= 0
                && box.x + box.width <= innerWidth && box.y + box.height <= innerHeight;
              const dp = document.querySelector(__DP__);
              const panels = [...document.querySelectorAll('[data-quantem-scientific-output^="show4dstem-compare-"]')].map(rect);
              const dpBox = dp && rect(dp);
              return {width: innerWidth, height: innerHeight, device_pixel_ratio: devicePixelRatio,
                dp_box: dpBox, dp_visible: dpBox && visible(dpBox), panels,
                panels_visible: panels.filter(visible).length};
            })()""".replace("__DP__", json.dumps(DP)),
            "awaitPromise": True, "returnByValue": True,
        }, 10)["result"]["value"]
        (browser_out / f"{name}_native_viewport.json").write_text(json.dumps(native_viewport, indent=2))
        assert native_viewport["dp_visible"] and native_viewport["panels_visible"] == count, (
            "Every scientific panel must remain visible after Playwright disconnects; "
            f"see {name}_native_viewport.json"
        )
        box = native_viewport["dp_box"]
        pointer = drag(target_id, box, endpoint=chrome.stress_cdp_endpoint,
                       duration_s=5, rate_hz=target, max_inflight=4,
                       start=start, trajectory=trajectory,
                       before_release=while_pressed)
        if not chrome.browser.is_connected():
            page = chrome.reconnect(target_id)
        if profile:
            (browser_out / f"{name}.cpuprofile").write_text(json.dumps(native_connection.call("Profiler.stop", {}, 10)["profile"]))
            profile_text = jupyter.run_in_kernel(notebook, "kernel_profile.disable(); import pstats; pstats.Stats(kernel_profile).sort_stats('cumulative').print_stats(35)")
            (browser_out / f"{name}_kernel_profile.txt").write_text(profile_text)
        # Browser and input driver are on this same host; use its recorded
        # performance origin, excluding input-ACK drain and held screenshots.
        started = pointer["host_wall_start_s"] * 1000 - pointer["browser_clock"]["timeOrigin"]
        ended = started + pointer["duration_s"] * 1000
        page.evaluate("scientificStress.active = false")
        page.wait_for_timeout(500)
        evidence = page.evaluate("structuredClone(scientificStress)")
        summary = _summary(evidence, count, started, ended)
        query_times = json.loads(jupyter.run_in_kernel(notebook, "import json; print(json.dumps(query_times))").splitlines()[-1])
        summary["query_ms_median"] = statistics.median(query_times) * 1000
        summary["query_ms_max"] = max(query_times) * 1000
        summary.update(preset=preset, gesture=gesture, target_hz=target, measured_display_raf_hz=1000 / statistics.median(refresh),
                       requested_input_hz=target, native_pointer=pointer, adapter_before=adapter,
                       adapter_after=page.evaluate(ADAPTER), before=before, held=held,
                       uploaded=held_upload, errors=list(dict.fromkeys(page.errors + evidence["pageErrors"])),
                       gpu_errors=evidence["errors"], playwright_detached_during_drag=True,
                       native_viewport=native_viewport)
        summary["target_met"] = summary["gpu_ready_refresh_hz"] >= target
        (browser_out / f"{name}_metrics.json").write_text(json.dumps(summary, indent=2))
        (browser_out / f"{name}_events.json").write_text(json.dumps(evidence))
        assert pointer["status"] == "complete" and pointer["released"], pointer
        assert summary["adapter_after"] == adapter and not evidence["errors"]
        assert [error for error in summary["errors"] if "debugger session" not in error] == []
        assert held_upload == {key: held[key] for key in ("bytes", "sha256")}
        assert held["sha256"] != before["sha256"] and held_png != before_png
        assert held["center" if gesture == "move" else "radii"] != before["center" if gesture == "move" else "radii"]
        assert summary["new_complete_image_revisions"] > 0
        assert summary["target_met"], (
            f"Target missed: {summary['gpu_ready_refresh_hz']:.2f} distinct GPU-ready refreshes/s "
            f"({summary['new_image_hz']:.2f} complete image revisions/s); "
            f"target {target:g} Hz; display rAF {summary['measured_display_raf_hz']:.2f} Hz. "
            "See saved metrics; this real-input rate is not an isolated maximum-throughput measurement."
        )
    finally:
        if native_connection is not None:
            native_connection.close()
        try:
            jupyter.run_in_kernel(notebook, KERNEL_CLEANUP)
        except StopIteration:
            pass  # Startup failed before this notebook acquired a kernel.
        finally:
            # A failed close must still terminate this test's kernel and free
            # its allocations before the next stress case starts.
            sessions = json.load(urllib.request.urlopen(jupyter.url + "/api/sessions"))
            for session in sessions:
                if session["path"] == notebook:
                    urllib.request.urlopen(urllib.request.Request(
                        jupyter.url + "/api/sessions/" + session["id"], method="DELETE")).close()
