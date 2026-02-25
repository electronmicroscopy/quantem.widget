"""
Open and execute all widget notebooks in JupyterLab via Playwright.

This opens each notebook in a real browser tab with a live kernel,
triggers "Restart Kernel and Run All Cells", and moves on.
Widgets render interactively because they have a live frontend connection.

Usage:
    python scripts/run_all_notebooks.py [--token TOKEN] [--port 8889] [--group all]

Groups: all, features, simple, showcase
"""

import argparse
import time
from playwright.sync_api import sync_playwright

FEATURES = [
    "show1d/show1d_all_features.ipynb",
    "show2d/show2d_all_features.ipynb",
    "show3d/show3d_all_features.ipynb",
    "show3dvolume/show3dvolume_all_features.ipynb",
    "show4d/show4d_all_features.ipynb",
    "show4dstem/show4dstem_all_features.ipynb",
    "showcomplex2d/showcomplex2d_all_features.ipynb",
    "mark2d/mark2d_all_features.ipynb",
    "edit2d/edit2d_all_features.ipynb",
    "align2d/align2d_all_features.ipynb",
    "bin/bin_all_features.ipynb",
]

SIMPLE = [
    "show1d/show1d_simple.ipynb",
    "show2d/show2d_simple.ipynb",
    "show3d/show3d_simple.ipynb",
    "show3dvolume/show3dvolume_simple.ipynb",
    "show4d/show4d_simple.ipynb",
    "show4dstem/show4dstem_simple.ipynb",
    "showcomplex2d/showcomplex2d_simple.ipynb",
    "mark2d/mark2d_simple.ipynb",
    "edit2d/edit2d_simple.ipynb",
    "align2d/align2d_simple.ipynb",
    "bin/bin_simple.ipynb",
]

SHOWCASE = [
    "collaboratorC/showcase_20260221.ipynb",
    "guoliang/showcase_guoliang.ipynb",
]


def run_notebook(page, base_url, token, nb_path, timeout_ms=120_000):
    """Open a notebook tab and run all cells."""
    url = f"{base_url}/lab/tree/{nb_path}?token={token}"
    page.goto(url)
    # Wait for notebook to fully load (kernel ready)
    page.wait_for_selector(".jp-Notebook", timeout=30_000)
    time.sleep(2)

    # Use JupyterLab menu: Run → Restart Kernel and Run All Cells
    page.keyboard.press("Control+Shift+P" if page.evaluate("() => navigator.platform.startsWith('Mac')") else "Control+Shift+P")
    time.sleep(0.5)
    # Try command palette approach
    try:
        page.fill('[placeholder="Search commands"]', "Restart Kernel and Run All")
        time.sleep(0.5)
        page.locator(".lm-Menu-itemLabel:has-text('Restart Kernel and Run All')").first.click()
        time.sleep(0.5)
        # Accept the confirmation dialog
        page.locator("button:has-text('Restart')").click(timeout=3000)
    except Exception:
        # Fallback: use keyboard shortcut or menu
        try:
            page.click('text=Run')
            time.sleep(0.3)
            page.click('text=Restart Kernel and Run All Cells')
            time.sleep(0.3)
            page.locator("button:has-text('Restart')").click(timeout=3000)
        except Exception:
            print(f"  [WARN] Could not trigger Run All for {nb_path}")
            return False

    print(f"  [RUN] {nb_path} — executing...")
    # Wait for execution to finish (cells stop showing [*])
    try:
        page.wait_for_function(
            "() => !document.querySelector('.jp-InputArea-prompt')?.textContent?.includes('*')",
            timeout=timeout_ms,
        )
    except Exception:
        print(f"  [TIMEOUT] {nb_path} — still running after {timeout_ms // 1000}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all widget notebooks in JupyterLab")
    parser.add_argument("--token", required=True, help="JupyterLab auth token")
    parser.add_argument("--port", type=int, default=8889, help="JupyterLab port")
    parser.add_argument("--group", default="all", choices=["all", "features", "simple", "showcase"],
                        help="Which notebook group to run")
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    args = parser.parse_args()

    notebooks = []
    if args.group in ("all", "features"):
        notebooks += FEATURES
    if args.group in ("all", "simple"):
        notebooks += SIMPLE
    if args.group in ("all", "showcase"):
        notebooks += SHOWCASE

    base_url = f"http://localhost:{args.port}"
    print(f"Running {len(notebooks)} notebooks on {base_url}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        context = browser.new_context(viewport={"width": 1400, "height": 900})

        for i, nb in enumerate(notebooks):
            print(f"\n[{i+1}/{len(notebooks)}] {nb}")
            page = context.new_page()
            run_notebook(page, base_url, args.token, nb)
            # Don't close the page — leave tabs open for the user to inspect

        print(f"\nDone! {len(notebooks)} notebooks running in browser tabs.")
        print("Press Ctrl+C to close browser, or leave it open to inspect widgets.")

        # Keep browser open for inspection
        try:
            input("\nPress Enter to close browser...")
        except KeyboardInterrupt:
            pass

        browser.close()


if __name__ == "__main__":
    main()
