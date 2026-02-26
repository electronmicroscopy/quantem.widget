"""
E2E Screenshot Capture for Align2DBulk Widget

Captures screenshots of Align2DBulk with before/after side-by-side layout.
Uses Playwright to automate the browser.

Test data: perovskite crystal lattice (50 frames) with random drift + Poisson noise.

Usage:
    python tests/capture_align2d_bulk.py

Screenshots are saved to tests/screenshots/align2d_bulk/
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

JUPYTER_PORT = 8899
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "align2d_bulk"
NOTEBOOK_PATH = Path(__file__).parent / "_test_align2d_bulk.ipynb"

FEATURE_NAMES = ["before_after", "fft_toggle"]

def create_test_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import torch\n",
                    "from quantem.widget import Show3D, Align2DBulk\n",
                    "\n",
                    "device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "n_frames, H, W = 30, 256, 256\n",
                    "lattice_a, sigma = 16.0, 1.8\n",
                    "yy = torch.arange(H, device=device, dtype=torch.float32)\n",
                    "xx = torch.arange(W, device=device, dtype=torch.float32)\n",
                    "Y, X = torch.meshgrid(yy, xx, indexing='ij')\n",
                    "base = torch.zeros((H, W), device=device)\n",
                    "for cy in torch.arange(lattice_a / 2, H, lattice_a, device=device):\n",
                    "    for cx in torch.arange(lattice_a / 2, W, lattice_a, device=device):\n",
                    "        base += torch.exp(-((Y - cy)**2 + (X - cx)**2) / (2 * sigma**2))\n",
                    "base = base / base.max()\n",
                    "np.random.seed(42)\n",
                    "true_shifts = np.column_stack([np.cumsum(np.random.randn(n_frames) * 3), np.cumsum(np.random.randn(n_frames) * 3)])\n",
                    "true_shifts -= true_shifts[0]\n",
                    "frames = []\n",
                    "for i in range(n_frames):\n",
                    "    dx, dy = true_shifts[i]\n",
                    "    grid_y = torch.linspace(-1, 1, H, device=device)\n",
                    "    grid_x = torch.linspace(-1, 1, W, device=device)\n",
                    "    gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')\n",
                    "    grid = torch.stack([gx - dx * 2.0 / W, gy - dy * 2.0 / H], dim=-1).unsqueeze(0)\n",
                    "    shifted = torch.nn.functional.grid_sample(base.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()\n",
                    "    noisy = torch.poisson(shifted.cpu() * 80 + 3) / 80\n",
                    "    frames.append(noisy.numpy())\n",
                    "raw_stack = np.stack(frames).astype(np.float32)\n",
                    "print(f'Stack: {raw_stack.shape}')\n",
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Before/after side-by-side with moving average\n",
                    "viewer = Show3D(raw_stack, title='Raw frames')\n",
                    "viewer.add_roi(128, 128, shape='rectangle')\n",
                    "viewer.roi_rectangle(200, 200)\n",
                    "bulk = Align2DBulk(raw_stack, roi=viewer.roi, max_shift=15)\n",
                    "bulk\n",
                ]
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created test notebook: {NOTEBOOK_PATH}")

def cleanup_test_notebook():
    if NOTEBOOK_PATH.exists():
        NOTEBOOK_PATH.unlink()
        print(f"Cleaned up: {NOTEBOOK_PATH}")

def start_jupyter():
    print("Starting JupyterLab...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "jupyter", "lab",
            f"--port={JUPYTER_PORT}", "--no-browser",
            "--ServerApp.token=''", "--ServerApp.password=''",
            "--ServerApp.disable_check_xsrf=True",
            "--IdentityProvider.token=''",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=Path(__file__).parent.parent,
    )
    import socket
    print("Waiting for JupyterLab to start...")
    for _ in range(30):
        time.sleep(1)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', JUPYTER_PORT))
            sock.close()
            if result == 0:
                print("JupyterLab is ready!")
                time.sleep(2)
                return proc
        except:
            pass
    raise RuntimeError("JupyterLab failed to start within 30 seconds")

def stop_jupyter(proc):
    print("Stopping JupyterLab...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except:
        proc.kill()

def capture_widgets(page, theme: str):
    print(f"Capturing {theme} theme screenshots...")
    theme_dir = SCREENSHOT_DIR / theme
    theme_dir.mkdir(parents=True, exist_ok=True)

    page.screenshot(path=str(theme_dir / "full_page.png"), full_page=True)
    print(f"  Saved: full_page.png")

    widgets = page.locator(".align2d-bulk-root")
    widget_count = widgets.count()
    print(f"  Found {widget_count} widgets")

    for i in range(min(widget_count, len(FEATURE_NAMES))):
        try:
            widget = widgets.nth(i)
            widget.scroll_into_view_if_needed()
            time.sleep(0.5)
            filename = f"{FEATURE_NAMES[i]}.png"
            widget.screenshot(path=str(theme_dir / filename))
            print(f"  Saved: {filename}")
        except Exception as e:
            print(f"  Warning: Could not capture {FEATURE_NAMES[i]}: {e}")

def main():
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    jupyter_proc = None
    try:
        create_test_notebook()
        jupyter_proc = start_jupyter()

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1400, "height": 900})
            page = context.new_page()

            notebook_rel_path = NOTEBOOK_PATH.relative_to(Path(__file__).parent.parent)
            url = f"http://localhost:{JUPYTER_PORT}/lab/tree/{notebook_rel_path}?token="
            print(f"Opening: {url}")
            page.goto(url, timeout=60000)

            print("Waiting for JupyterLab to load...")
            time.sleep(10)

            try:
                page.keyboard.press("Escape")
                time.sleep(0.5)
            except:
                pass

            # Light theme
            page.evaluate("""() => {
                document.body.dataset.jpThemeLight = 'true';
                document.body.dataset.jpThemeName = 'JupyterLab Light';
                document.body.classList.remove('jp-theme-dark');
                document.body.classList.add('jp-theme-light');
            }""")
            time.sleep(1)

            page.keyboard.press("Meta+Shift+C")
            time.sleep(0.3)
            page.keyboard.type("Run All Cells")
            time.sleep(0.3)
            page.keyboard.press("Enter")
            print("  Waiting for cells to execute...")
            time.sleep(30)

            capture_widgets(page, "light")

            # Toggle FFT on the widget via JS
            try:
                page.evaluate("""() => {
                    const switches = document.querySelectorAll('.align2d-bulk-root .MuiSwitch-input');
                    for (const sw of switches) sw.click();
                }""")
                time.sleep(2)
                theme_dir = SCREENSHOT_DIR / "light"
                widgets = page.locator(".align2d-bulk-root")
                if widgets.count() > 0:
                    widgets.nth(0).screenshot(path=str(theme_dir / "fft_toggle.png"))
                    print("  Saved: fft_toggle.png (FFT on)")
                # Toggle back off
                page.evaluate("""() => {
                    const switches = document.querySelectorAll('.align2d-bulk-root .MuiSwitch-input');
                    for (const sw of switches) sw.click();
                }""")
                time.sleep(1)
            except Exception as e:
                print(f"  Warning: FFT toggle test failed: {e}")

            # Dark theme
            page.evaluate("""() => {
                document.body.dataset.jpThemeLight = 'false';
                document.body.dataset.jpThemeName = 'JupyterLab Dark';
                document.body.classList.remove('jp-theme-light');
                document.body.classList.add('jp-theme-dark');
            }""")
            time.sleep(2)
            capture_widgets(page, "dark")

            print(f"\nScreenshots saved to: {SCREENSHOT_DIR}")
            browser.close()

    finally:
        if jupyter_proc:
            stop_jupyter(jupyter_proc)
        cleanup_test_notebook()

if __name__ == "__main__":
    main()
