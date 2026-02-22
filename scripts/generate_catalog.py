"""
Visual Widget Catalog Generator

Generates an HTML page showing every quantem.widget in key states for visual QA.
Creates a test notebook, launches JupyterLab, captures screenshots of all widgets
in light and dark themes, and assembles them into a browsable HTML page.

Usage:
    python scripts/generate_catalog.py              # generate + open in browser
    python scripts/generate_catalog.py --serve       # generate + start local server
    python scripts/generate_catalog.py --light-only  # light theme only (faster)

Output:
    tests/screenshots/catalog/index.html
"""

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JUPYTER_PORT = 8877
SERVE_PORT = 8321
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = PROJECT_ROOT / "tests" / "screenshots" / "catalog"
DOCS_STATIC_DIR = PROJECT_ROOT / "docs" / "_static" / "catalog"
NOTEBOOK_PATH = PROJECT_ROOT / "tests" / "_widget_catalog.ipynb"
OUTPUT_HTML = SCREENSHOT_DIR / "index.html"

# ---------------------------------------------------------------------------
# Notebook generation
# ---------------------------------------------------------------------------

# Widget cells: list of (label, source_lines)
WIDGET_CELLS: list[tuple[str, list[str]]] = [
    # ── Show1D ────────────────────────────────────────────────────────────
    ("Show1D Default", [
        "from quantem.widget import Show1D\n",
        "Show1D(spectrum, x=energy, title='Show1D Default', x_label='Energy Loss', x_unit='eV', y_label='Counts')\n",
    ]),
    ("Show1D Multi-Trace", [
        "Show1D([spectrum, spectrum * 0.7 + 0.1, spectrum * 0.4 + 0.3],\n",
        "       x=energy, labels=['Region A', 'Region B', 'Region C'],\n",
        "       title='Show1D Multi-Trace', x_label='Energy', x_unit='eV')\n",
    ]),
    # ── Show2D ────────────────────────────────────────────────────────────
    ("Show2D Default", [
        "from quantem.widget import Show2D\n",
        "Show2D(image_128, title='Show2D Default', pixel_size=2.5)\n",
    ]),
    ("Show2D Viridis + Log", [
        "Show2D(image_128, title='Show2D Viridis+Log', cmap='viridis', log_scale=True, pixel_size=2.5)\n",
    ]),
    ("Show2D Gallery", [
        "Show2D([image_128, np.flipud(image_128), np.fliplr(image_128)],\n",
        "       labels=['Original', 'FlipUD', 'FlipLR'], title='Show2D Gallery')\n",
    ]),
    # ── Show3D ────────────────────────────────────────────────────────────
    ("Show3D Default", [
        "from quantem.widget import Show3D\n",
        "Show3D(stack_50, title='Show3D Default', fps=10, pixel_size=2.5)\n",
    ]),
    ("Show3D + ROI", [
        "w = Show3D(stack_50, title='Show3D + ROI', cmap='viridis')\n",
        "w.add_roi(row=64, col=64, shape='circle')\n",
        "w\n",
    ]),
    # ── Show3DVolume ──────────────────────────────────────────────────────
    ("Show3DVolume Default", [
        "from quantem.widget import Show3DVolume\n",
        "Show3DVolume(volume_64, title='Show3DVolume Default', cmap='viridis')\n",
    ]),
    # ── Show4D ────────────────────────────────────────────────────────────
    ("Show4D Default", [
        "from quantem.widget import Show4D\n",
        "Show4D(data_4d, title='Show4D Default')\n",
    ]),
    # ── Show4DSTEM ────────────────────────────────────────────────────────
    ("Show4DSTEM Default", [
        "from quantem.widget import Show4DSTEM\n",
        "Show4DSTEM(data_4d, title='Show4DSTEM Default', pixel_size=2.39, k_pixel_size=0.46)\n",
    ]),
    # ── ShowComplex2D ─────────────────────────────────────────────────────
    ("ShowComplex2D Amplitude", [
        "from quantem.widget import ShowComplex2D\n",
        "ShowComplex2D(complex_data, title='ShowComplex2D Amplitude', display_mode='amplitude')\n",
    ]),
    ("ShowComplex2D Phase", [
        "ShowComplex2D(complex_data, title='ShowComplex2D Phase', display_mode='phase')\n",
    ]),
    ("ShowComplex2D HSV", [
        "ShowComplex2D(complex_data, title='ShowComplex2D HSV', display_mode='hsv')\n",
    ]),
    # ── Mark2D ────────────────────────────────────────────────────────────
    ("Mark2D Default", [
        "from quantem.widget import Mark2D\n",
        "Mark2D(image_128, title='Mark2D Default', pixel_size=2.5)\n",
    ]),
    ("Mark2D + Points + ROI", [
        "w = Mark2D(image_128, title='Mark2D + Points', points=[(30,30),(42,30),(54,30)])\n",
        "w.add_roi(64, 64, shape='circle', radius=20, color='#00ff00')\n",
        "w\n",
    ]),
    # ── Edit2D ────────────────────────────────────────────────────────────
    ("Edit2D Default", [
        "from quantem.widget import Edit2D\n",
        "Edit2D(image_128, title='Edit2D Default')\n",
    ]),
    ("Edit2D Mask Mode", [
        "Edit2D(image_128, title='Edit2D Mask Mode', mode='mask')\n",
    ]),
    # ── Align2D ───────────────────────────────────────────────────────────
    ("Align2D Default", [
        "from quantem.widget import Align2D\n",
        "shifted = np.roll(np.roll(image_128, 5, axis=0), -3, axis=1)\n",
        "Align2D(image_128, shifted, title='Align2D Default', pixel_size=2.5)\n",
    ]),
    # ── Bin ───────────────────────────────────────────────────────────────
    ("Bin Default", [
        "from quantem.widget import Bin\n",
        "Bin(data_4d_small, pixel_size=2.39, k_pixel_size=0.46, device='cpu')\n",
    ]),
    # ── Tool lock / hide demos ────────────────────────────────────────────
    ("Show2D Locked Display", [
        "Show2D(image_128, title='Show2D Locked Display', disable_display=True, disable_export=True)\n",
    ]),
    ("Show3D Hidden Histogram", [
        "Show3D(stack_50, title='Show3D Hidden Histogram', hide_histogram=True, hide_stats=True)\n",
    ]),
]

WIDGET_CSS_CLASS = {
    "Show1D": ".show1d-root",
    "Show2D": ".show2d-root",
    "Show3DVolume": ".show3dvolume-root",
    "Show3D": ".show3d-root",
    "Show4DSTEM": ".show4dstem-root",
    "Show4D": ".show4d-root",
    "ShowComplex2D": ".showcomplex-root",
    "Mark2D": ".mark2d-root",
    "Edit2D": ".edit2d-root",
    "Align2D": ".align2d-root",
    "Bin": ".bin-root",
}


def _css_class_for_label(label: str) -> str:
    # Sort by length descending so "Show3DVolume" matches before "Show3D"
    for widget_name, css_class in sorted(WIDGET_CSS_CLASS.items(), key=lambda x: -len(x[0])):
        if label.startswith(widget_name):
            return css_class
    raise ValueError(f"Cannot determine CSS class for label: {label}")


def _build_widget_index():
    counts: dict[str, int] = {}
    result = []
    for label, _ in WIDGET_CELLS:
        css = _css_class_for_label(label)
        idx = counts.get(css, 0)
        counts[css] = idx + 1
        result.append((label, css, idx))
    return result


WIDGET_INDEX = _build_widget_index()


def _setup_cell_source() -> list[str]:
    return [
        "import numpy as np\n",
        "np.random.seed(42)\n",
        "\n",
        "# Hexagonal lattice of Gaussian peaks (atomic columns)\n",
        "image_128 = np.zeros((128, 128), dtype=np.float32)\n",
        "a = 12\n",
        "for i in range(-10, 15):\n",
        "    for j in range(-10, 15):\n",
        "        cx = 64 + int(i * a + j * a * 0.5)\n",
        "        cy = 64 + int(j * a * 0.866)\n",
        "        if 0 <= cx < 128 and 0 <= cy < 128:\n",
        "            y, x = np.ogrid[max(0,cy-4):min(128,cy+5), max(0,cx-4):min(128,cx+5)]\n",
        "            r2 = (x - cx)**2 + (y - cy)**2\n",
        "            image_128[max(0,cy-4):min(128,cy+5), max(0,cx-4):min(128,cx+5)] += np.exp(-r2 / 3.0)\n",
        "image_128 += np.random.normal(0, 0.02, image_128.shape).astype(np.float32)\n",
        "\n",
        "# 3D stack (50 frames with slight drift)\n",
        "stack_50 = np.stack([\n",
        "    np.roll(np.roll(image_128, int(3*np.sin(2*np.pi*i/50)), axis=0),\n",
        "            int(3*np.cos(2*np.pi*i/50)), axis=1) + np.random.normal(0, 0.01, (128, 128))\n",
        "    for i in range(50)\n",
        "]).astype(np.float32)\n",
        "\n",
        "# 3D volume (sphere in center)\n",
        "volume_64 = np.random.rand(64, 64, 64).astype(np.float32)\n",
        "z, y, x = np.mgrid[:64, :64, :64]\n",
        "volume_64 += 2.0 * np.exp(-((x-32)**2 + (y-32)**2 + (z-32)**2) / (2*8**2))\n",
        "\n",
        "# 4D data\n",
        "data_4d = np.random.rand(16, 16, 64, 64).astype(np.float32)\n",
        "yy, xx = np.mgrid[0:64, 0:64]\n",
        "r = np.sqrt((xx - 32)**2 + (yy - 32)**2)\n",
        "bf_disk = np.exp(-r**2 / 40)\n",
        "data_4d += bf_disk[np.newaxis, np.newaxis, :, :] * 3\n",
        "\n",
        "# Complex data\n",
        "complex_data = (np.random.rand(128, 128) * np.exp(1j * np.random.rand(128, 128) * 2 * np.pi)).astype(np.complex64)\n",
        "\n",
        "# 1D spectrum (EELS-like)\n",
        "energy = np.linspace(0, 800, 512).astype(np.float32)\n",
        "spectrum = np.exp(-(energy - 300)**2 / 2000) + 0.5 * np.exp(-(energy - 530)**2 / 500)\n",
        "spectrum += np.random.normal(0, 0.02, spectrum.shape)\n",
        "spectrum = spectrum.astype(np.float32)\n",
        "\n",
        "# Small 4D data for Bin (8x8 scan, 32x32 detector — fast)\n",
        "data_4d_small = np.random.rand(8, 8, 32, 32).astype(np.float32)\n",
        "yy_s, xx_s = np.mgrid[0:32, 0:32]\n",
        "bf_s = np.exp(-((xx_s - 16)**2 + (yy_s - 16)**2) / 20)\n",
        "data_4d_small += bf_s[np.newaxis, np.newaxis, :, :] * 3\n",
        "\n",
        "print('Data ready')\n",
    ]


def create_test_notebook():
    cells = []
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _setup_cell_source(),
    })
    for label, source_lines in WIDGET_CELLS:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {label}\n"],
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines,
        })
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2))
    print(f"Created test notebook: {NOTEBOOK_PATH}")


def cleanup_test_notebook():
    if NOTEBOOK_PATH.exists():
        NOTEBOOK_PATH.unlink()
        print(f"Cleaned up: {NOTEBOOK_PATH}")


# ---------------------------------------------------------------------------
# JupyterLab management
# ---------------------------------------------------------------------------

def start_jupyter():
    print("Starting JupyterLab...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "jupyter", "lab",
            f"--port={JUPYTER_PORT}", "--no-browser",
            "--NotebookApp.token=''", "--NotebookApp.password=''",
            "--ServerApp.disable_check_xsrf=True",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
    )
    import socket
    print("Waiting for JupyterLab to start...")
    for _ in range(30):
        time.sleep(1)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("localhost", JUPYTER_PORT))
            sock.close()
            if result == 0:
                print("JupyterLab is ready!")
                time.sleep(2)
                return proc
        except Exception:
            pass
    raise RuntimeError("JupyterLab failed to start within 30 seconds")


def stop_jupyter(proc):
    print("Stopping JupyterLab...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


# ---------------------------------------------------------------------------
# Theme switching
# ---------------------------------------------------------------------------

def set_theme(page, theme: str):
    print(f"  Setting {theme} theme...")
    if theme == "dark":
        page.evaluate("""() => {
            document.body.dataset.jpThemeLight = 'false';
            document.body.dataset.jpThemeName = 'JupyterLab Dark';
            document.body.classList.remove('jp-theme-light');
            document.body.classList.add('jp-theme-dark');
        }""")
    else:
        page.evaluate("""() => {
            document.body.dataset.jpThemeLight = 'true';
            document.body.dataset.jpThemeName = 'JupyterLab Light';
            document.body.classList.remove('jp-theme-dark');
            document.body.classList.add('jp-theme-light');
        }""")
    time.sleep(1)


# ---------------------------------------------------------------------------
# Screenshot capture
# ---------------------------------------------------------------------------

def capture_widgets(page, theme: str) -> list[tuple[str, str, Path]]:
    """Capture screenshots. Returns list of (label, theme, path)."""
    print(f"\nCapturing {theme} theme screenshots...")
    captured: list[tuple[str, str, Path]] = []

    for label, css_class, nth in WIDGET_INDEX:
        safe_label = label.lower().replace(" ", "_").replace("+", "").replace("/", "_")
        filename = f"{safe_label}_{theme}.png"
        filepath = SCREENSHOT_DIR / filename

        try:
            widgets = page.locator(css_class)
            widget_count = widgets.count()
            if nth >= widget_count:
                print(f"  Skip: {label} (only {widget_count} '{css_class}' found, need index {nth})")
                continue
            widget = widgets.nth(nth)
            widget.scroll_into_view_if_needed()
            time.sleep(0.5)
            widget.screenshot(path=str(filepath))
            captured.append((label, theme, filepath))
            print(f"  Saved: {filename}")
        except Exception as e:
            print(f"  Warning: Could not capture {label}: {e}")

    return captured


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _img_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def assemble_html(captured: list[tuple[str, str, Path]], light_only: bool):
    """Assemble captured screenshots into a self-contained HTML page."""
    if not captured:
        print("No screenshots to assemble.")
        return

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nAssembling HTML from {len(captured)} screenshots...")

    # Group by label to pair light/dark
    from collections import OrderedDict
    groups: OrderedDict[str, dict[str, Path]] = OrderedDict()
    for label, theme, path in captured:
        if label not in groups:
            groups[label] = {}
        groups[label][theme] = path

    # Build nav + cards
    nav_items = []
    cards = []
    for i, (label, themes) in enumerate(groups.items()):
        anchor = f"widget-{i}"
        widget_type = label.split()[0]
        nav_items.append(f'<a href="#{anchor}" class="nav-item" data-type="{widget_type}">{label}</a>')

        imgs_html = ""
        for theme_name in ["light", "dark"]:
            if theme_name not in themes:
                continue
            b64 = _img_to_base64(themes[theme_name])
            badge_class = "badge-light" if theme_name == "light" else "badge-dark"
            imgs_html += f"""
            <div class="screenshot">
                <span class="badge {badge_class}">{theme_name}</span>
                <img src="data:image/png;base64,{b64}" alt="{label} ({theme_name})" loading="lazy">
            </div>"""

        cards.append(f"""
        <div class="card" id="{anchor}" data-type="{widget_type}">
            <h2>{label}</h2>
            <div class="screenshots">{imgs_html}
            </div>
        </div>""")

    # Unique widget types for filter buttons
    widget_types = list(dict.fromkeys(label.split()[0] for label in groups))
    filter_buttons = '<button class="filter-btn active" data-filter="all">All</button>'
    for wt in widget_types:
        filter_buttons += f'<button class="filter-btn" data-filter="{wt}">{wt}</button>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>quantem.widget Visual Catalog</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; }}
  .header {{
    position: sticky; top: 0; z-index: 100;
    background: #1a1a2e; color: white; padding: 16px 24px;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }}
  .header h1 {{ font-size: 20px; font-weight: 600; white-space: nowrap; }}
  .header .meta {{ font-size: 12px; color: #888; white-space: nowrap; }}
  .filters {{ display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto; }}
  .filter-btn {{
    padding: 4px 12px; border: 1px solid #444; border-radius: 12px;
    background: transparent; color: #ccc; cursor: pointer; font-size: 12px;
    transition: all 0.15s;
  }}
  .filter-btn:hover {{ background: #333; color: white; }}
  .filter-btn.active {{ background: #4a90d9; color: white; border-color: #4a90d9; }}
  .layout {{ display: flex; min-height: calc(100vh - 60px); }}
  .sidebar {{
    width: 220px; min-width: 220px; background: white;
    border-right: 1px solid #ddd; padding: 12px 0; overflow-y: auto;
    position: sticky; top: 60px; height: calc(100vh - 60px);
  }}
  .nav-item {{
    display: block; padding: 6px 16px; font-size: 12px; color: #555;
    text-decoration: none; border-left: 3px solid transparent;
    transition: all 0.1s;
  }}
  .nav-item:hover {{ background: #f0f4ff; color: #1a1a2e; }}
  .nav-item.active {{ border-left-color: #4a90d9; background: #f0f4ff; color: #1a1a2e; font-weight: 600; }}
  .nav-item.hidden {{ display: none; }}
  .main {{ flex: 1; padding: 24px; max-width: calc(100% - 220px); }}
  .card {{
    background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    scroll-margin-top: 80px;
  }}
  .card.hidden {{ display: none; }}
  .card h2 {{ font-size: 16px; margin-bottom: 12px; color: #1a1a2e; }}
  .screenshots {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .screenshot {{ position: relative; }}
  .screenshot img {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }}
  .badge {{
    position: absolute; top: 8px; right: 8px; padding: 2px 8px;
    border-radius: 4px; font-size: 10px; font-weight: 600;
  }}
  .badge-light {{ background: #fff; color: #333; border: 1px solid #ddd; }}
  .badge-dark {{ background: #333; color: #fff; }}
  .count {{ font-size: 12px; color: #888; margin-left: 8px; }}
</style>
</head>
<body>
<div class="header">
  <h1>quantem.widget Catalog</h1>
  <span class="meta">{len(groups)} widgets &middot; {timestamp}</span>
  <div class="filters">{filter_buttons}</div>
</div>
<div class="layout">
  <nav class="sidebar">
    {"".join(nav_items)}
  </nav>
  <div class="main">
    {"".join(cards)}
  </div>
</div>
<script>
// Filter buttons
document.querySelectorAll('.filter-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const filter = btn.dataset.filter;
    document.querySelectorAll('.card').forEach(card => {{
      card.classList.toggle('hidden', filter !== 'all' && card.dataset.type !== filter);
    }});
    document.querySelectorAll('.nav-item').forEach(nav => {{
      nav.classList.toggle('hidden', filter !== 'all' && nav.dataset.type !== filter);
    }});
  }});
}});
// Highlight active nav on scroll
const observer = new IntersectionObserver(entries => {{
  entries.forEach(e => {{
    if (e.isIntersecting) {{
      document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
      const link = document.querySelector(`.nav-item[href="#${{e.target.id}}"]`);
      if (link) link.classList.add('active');
    }}
  }});
}}, {{ rootMargin: '-80px 0px -60% 0px' }});
document.querySelectorAll('.card').forEach(card => observer.observe(card));
</script>
</body>
</html>"""

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html)
    print(f"\nHTML saved: {OUTPUT_HTML}")
    print(f"  Widgets: {len(groups)}, Screenshots: {len(captured)}")


# ---------------------------------------------------------------------------
# Local server
# ---------------------------------------------------------------------------

def copy_to_docs(captured: list[tuple[str, str, Path]]):
    """Copy screenshots to docs/_static/catalog/ for Sphinx gallery page."""
    import shutil

    DOCS_STATIC_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    for label, theme, path in captured:
        safe_label = label.lower().replace(" ", "_").replace("+", "").replace("/", "_")
        dest = DOCS_STATIC_DIR / f"{safe_label}_{theme}.png"
        shutil.copy2(path, dest)
        copied += 1

    print(f"\nDocs screenshots: {copied} images copied to {DOCS_STATIC_DIR}")
    print("  Gallery page: docs/widgets/gallery.rst (update manually if widgets change)")


def serve_catalog():
    """Start a local HTTP server to serve the catalog."""
    import http.server
    import threading
    import webbrowser

    os_module = __import__("os")
    os_module.chdir(SCREENSHOT_DIR)

    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer(("localhost", SERVE_PORT), handler)

    url = f"http://localhost:{SERVE_PORT}/index.html"
    print(f"\nServing catalog at: {url}")
    print("Press Ctrl+C to stop.\n")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate visual widget catalog")
    parser.add_argument("--light-only", action="store_true", help="Skip dark theme (faster)")
    parser.add_argument("--serve", action="store_true", help="Start local server after generation")
    parser.add_argument("--serve-only", action="store_true", help="Serve existing catalog (skip generation)")
    parser.add_argument("--docs", action="store_true", help="Copy screenshots to docs/_static/catalog/ and generate gallery.rst")
    args = parser.parse_args()

    if args.serve_only:
        if not OUTPUT_HTML.exists():
            print(f"No catalog found at {OUTPUT_HTML}. Run without --serve-only first.")
            sys.exit(1)
        serve_catalog()
        return

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    jupyter_proc = None
    all_captured: list[tuple[str, str, Path]] = []

    try:
        create_test_notebook()
        jupyter_proc = start_jupyter()

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1400, "height": 900}, device_scale_factor=2)
            page = context.new_page()

            notebook_rel_path = NOTEBOOK_PATH.relative_to(PROJECT_ROOT)
            url = f"http://localhost:{JUPYTER_PORT}/lab/tree/{notebook_rel_path}"
            print(f"Opening: {url}")
            page.goto(url, timeout=60000)
            print("Waiting for JupyterLab to load...")
            time.sleep(10)

            try:
                page.keyboard.press("Escape")
                time.sleep(0.5)
            except Exception:
                pass

            set_theme(page, "light")
            print("Running all cells...")
            page.keyboard.press("Meta+Shift+C")
            time.sleep(0.3)
            page.keyboard.type("Run All Cells")
            time.sleep(0.3)
            page.keyboard.press("Enter")
            print("  Waiting for cells to execute (30s)...")
            time.sleep(30)

            all_captured.extend(capture_widgets(page, "light"))

            if not args.light_only:
                set_theme(page, "dark")
                time.sleep(2)
                all_captured.extend(capture_widgets(page, "dark"))

            print(f"\nScreenshots saved to: {SCREENSHOT_DIR}")
            browser.close()

        assemble_html(all_captured, args.light_only)

        if args.docs:
            copy_to_docs(all_captured)

    finally:
        if jupyter_proc:
            stop_jupyter(jupyter_proc)
        cleanup_test_notebook()

    if args.serve:
        serve_catalog()
    else:
        import webbrowser
        webbrowser.open(f"file://{OUTPUT_HTML}")


if __name__ == "__main__":
    main()
