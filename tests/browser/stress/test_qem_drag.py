"""Opt-in native-resolution detector dragging across seven and ten recordings."""

import json
import math
import os
import statistics
import urllib.request
from itertools import pairwise
from pathlib import Path

import pytest
from native_pointer import drag

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
import hashlib, json
print(json.dumps({
    "scan": [viewer.shape_rows, viewer.shape_cols],
    "detector": list(loaded[0].shape[-2:]),
    "panels": list(viewer.compare_panel_indices),
    "center": list(viewer.roi_center),
    "bytes": len(viewer.compare_virtual_image_bytes),
    "sha256": hashlib.sha256(viewer.compare_virtual_image_bytes).hexdigest(),
}))
"""


def _report(jupyter, notebook):
    return json.loads(jupyter.run_in_kernel(notebook, REPORT).splitlines()[-1])


def _summary(evidence, count, started, ended):
    """Count a revision once, after every comparison canvas is GPU-ready."""
    revisions = {}
    expected = {f"show4dstem-compare-{index}" for index in range(count)}
    for item in evidence["presentations"]:
        if started <= item["queued"] <= item["ready"] <= ended:
            revisions.setdefault(item["revision"], set()).update(item["targets"])
    complete = sum(targets >= expected for targets in revisions.values())
    frames = [stamp for stamp in evidence["frames"] if started <= stamp <= ended]
    gaps = [right - left for left, right in pairwise(frames)]
    return {
        "seconds": (ended - started) / 1000,
        "new_complete_image_revisions": complete,
        "new_image_hz": complete * 1000 / (ended - started),
        "raf_hz_during_drag": 1000 / statistics.median(gaps) if gaps else 0,
        "trusted_pointer_events": sum(item["trusted"] for item in evidence["inputs"]
                                      if started <= item["at"] <= ended),
        "delivered_input_hz": sum(started <= item["at"] <= ended and item["trusted"]
                                  for item in evidence["inputs"]) * 1000 / (ended - started),
    }


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
def test_native_qem_detector_drag(count, chrome, jupyter, open_page, hardware_adapter, browser_out):
    root = Path(os.environ["QUANTEM_WIDGET_STRESS_QEM_DIR"]).expanduser()
    paths = sorted({path.resolve() for path in root.glob("*.qem")})[:count]
    assert len(paths) == count, f"Provide {count} independent .qem recordings in the fixture directory"
    target = float(os.getenv("QUANTEM_WIDGET_STRESS_TARGET_HZ", "120"))
    assert math.isfinite(target) and target > 0
    device = int(os.getenv("QUANTEM_WIDGET_STRESS_GPU", "0"))
    notebook = f"stress_qem_{count}.ipynb"
    jupyter.write_notebook(notebook, f"""
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "0")
from quantem.gpu import io
from quantem.widget import Show4DSTEM
loaded = io.load({[str(path) for path in paths]!r}, backend="cuda", device={device}, verbose=False)
viewer = Show4DSTEM(loaded, view_mode="multiple", compare_group_mode="all",
    compare_max_panels={count}, compare_cols=5, compare_grid_width_px=1200,
    panel_width_px=320, sync_resize=False, show_fft=False)
viewer
""")
    instrument = Path(__file__).with_name("image_updates.js").read_text()

    def instrumented_page(url, *, webgpu):
        page = open_page("about:blank", webgpu=webgpu)
        page.set_viewport_size({"width": 1900, "height": 1800})
        page.add_init_script(instrument)
        page.goto(url)
        return page

    try:
        page = jupyter.open(instrumented_page, notebook, f"stress_{count}", webgpu=True,
                            ready=VI, out=browser_out)
        adapter = page.evaluate(ADAPTER)
        hardware_adapter(adapter)
        assert not adapter["fallback"], f"Hardware adapter required: {adapter}"
        before = _report(jupyter, notebook)
        assert before["scan"] == [512, 512] and before["detector"] == [192, 192]
        assert before["panels"] == list(range(count))
        assert before["bytes"] == count * 512 * 512 * 4
        before_png = page.locator(VI).first.screenshot(path=str(browser_out / f"qem_{count}_before.png"))
        page.locator(".jp-OutputArea").first.scroll_into_view_if_needed()
        box = page.locator(DP).first.bounding_box()
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
        held = {}
        held_png = None
        held_upload = None

        def while_pressed():
            nonlocal held, held_png, held_upload
            # Drain and evidence collection are excluded from the five-second
            # requested input interval and its fresh scientific-image count.
            page.wait_for_timeout(800)
            held_png = page.locator(VI).first.screenshot(
                path=str(browser_out / f"qem_{count}_held.png"))
            for _ in range(30):
                held = _report(jupyter, notebook)
                held_upload = page.evaluate("scientificStressUploadHash()")
                if held_upload == {key: held[key] for key in ("bytes", "sha256")}:
                    break
                page.wait_for_timeout(100)
            else:
                raise AssertionError("The held detector's full native image bytes did not reach every panel")

        pointer = drag(target_id, box, endpoint=chrome.stress_cdp_endpoint,
                       duration_s=5, rate_hz=target, max_inflight=4,
                       before_release=while_pressed)
        # Browser and input driver are on this same host; use its recorded
        # performance origin, excluding input-ACK drain and held screenshots.
        started = pointer["host_wall_start_s"] * 1000 - pointer["browser_clock"]["timeOrigin"]
        ended = started + pointer["duration_s"] * 1000
        page.evaluate("scientificStress.active = false")
        page.wait_for_timeout(500)
        evidence = page.evaluate("structuredClone(scientificStress)")
        summary = _summary(evidence, count, started, ended)
        summary.update(target_hz=target, measured_display_raf_hz=1000 / statistics.median(refresh),
                       requested_input_hz=target, native_pointer=pointer, adapter_before=adapter,
                       adapter_after=page.evaluate(ADAPTER), before=before, held=held,
                       uploaded=held_upload, errors=page.errors, gpu_errors=evidence["errors"])
        summary["target_met"] = summary["new_image_hz"] >= target
        (browser_out / f"qem_{count}_metrics.json").write_text(json.dumps(summary, indent=2))
        (browser_out / f"qem_{count}_events.json").write_text(json.dumps(evidence))
        assert pointer["status"] == "complete" and pointer["released"], pointer
        assert summary["adapter_after"] == adapter and not evidence["errors"]
        assert [error for error in page.errors if "debugger session" not in error] == []
        assert held_upload == {key: held[key] for key in ("bytes", "sha256")}
        assert held["sha256"] != before["sha256"] and held_png != before_png
        assert held["center"] != before["center"]
        assert summary["new_complete_image_revisions"] > 0
        assert summary["target_met"], (
            f"Target missed: {summary['new_image_hz']:.2f} new complete images/s; "
            f"target {target:g} Hz; display rAF {summary['measured_display_raf_hz']:.2f} Hz. "
            "See saved metrics; this real-input rate is not an isolated maximum-throughput measurement."
        )
    finally:
        try:
            jupyter.run_in_kernel(notebook, KERNEL_CLEANUP)
        except StopIteration:
            pass  # Startup failed before this notebook acquired a kernel.
        sessions = json.load(urllib.request.urlopen(jupyter.url + "/api/sessions"))
        for session in sessions:
            if session["path"] == notebook:
                urllib.request.urlopen(urllib.request.Request(
                    jupyter.url + "/api/sessions/" + session["id"], method="DELETE")).close()
