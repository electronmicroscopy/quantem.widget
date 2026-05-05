import argparse
import sys
import tempfile
import webbrowser
from pathlib import Path

_WEBGPU_PROBE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>WebGPU check</title>
<style>
body { font: 16px -apple-system, system-ui, sans-serif; padding: 32px; max-width: 640px; }
h1 { font-size: 20px; margin: 0 0 16px; }
.box { padding: 20px; border-radius: 8px; margin-top: 12px; }
.ok { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
.bad { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
pre { font: 13px ui-monospace, monospace; background: #f5f5f5; padding: 12px; border-radius: 6px; overflow-x: auto; }
</style></head><body>
<h1>quantem.widget &mdash; WebGPU check</h1>
<div id="status" class="box">Probing&hellip;</div>
<pre id="detail"></pre>
<script>
(async () => {
    const status = document.getElementById('status');
    const detail = document.getElementById('detail');
    detail.textContent = 'User-Agent: ' + navigator.userAgent;
    if (!navigator.gpu) {
        status.className = 'box bad';
        status.textContent = 'WebGPU: NOT AVAILABLE (navigator.gpu undefined)';
        return;
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            status.className = 'box bad';
            status.textContent = 'WebGPU: NOT AVAILABLE (no adapter)';
            return;
        }
        const info = adapter.info || {};
        const parts = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' ');
        status.className = 'box ok';
        status.textContent = 'WebGPU: AVAILABLE' + (parts ? ' (' + parts + ')' : '');
        detail.textContent += '\\n\\nadapter.info:\\n' + JSON.stringify(info, null, 2);
        detail.textContent += '\\n\\nfeatures:\\n' + [...adapter.features].join(', ');
    } catch (err) {
        status.className = 'box bad';
        status.textContent = 'WebGPU: NOT AVAILABLE (' + err.message + ')';
    }
})();
</script></body></html>
"""


def _cmd_profile(args: argparse.Namespace) -> int:
    from quantem.widget.profile import profile
    profile()
    if args.no_browser:
        return 0
    html_path = Path(tempfile.gettempdir()) / "quantem_widget_webgpu.html"
    html_path.write_text(_WEBGPU_PROBE_HTML)
    print(f"\nWebGPU check  opened in browser ({html_path})")
    webbrowser.open(html_path.as_uri())
    return 0


def main(argv: list[str] | None = None) -> int:
    # Shared flags reused by the top-level parser and the `profile` subparser
    # so `widget --no-browser`, `widget profile --no-browser`, and a bare
    # `widget` (defaults to profile) all work.
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--no-browser",
        action="store_true",
        help="skip opening the WebGPU probe in the default browser",
    )

    parser = argparse.ArgumentParser(
        prog="widget",
        description="quantem.widget command-line tools",
        parents=[shared],
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser(
        "profile",
        parents=[shared],
        help="Print Python/torch/system info and open browser-side WebGPU check",
    )

    args = parser.parse_args(argv)
    if args.command in (None, "profile"):
        return _cmd_profile(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
