"""ShowCIF on the page's shared hardware WebGPU device, in headed Chrome.

The atom scene draws with the one device every widget on the page shares: the
3D view and the column projection show atoms, not a cleared canvas, and no
WebGPU validation error reaches the status line.
"""

import io

import numpy as np
import pytest
from ase import Atoms
from PIL import Image

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def silicon(browser_out) -> str:
    from quantem.widget import ShowCIF
    from quantem.widget.export import write_widget_html

    crystal = Atoms("Si2", scaled_positions=[(0, 0, 0), (0.25, 0.25, 0.25)], cell=[5.43] * 3, pbc=True)
    widget = ShowCIF(crystal)
    write_widget_html(browser_out / "showcif_silicon.html", widget, "showcif")
    widget.close()
    return "showcif_silicon.html"


def distinct_colours(locator) -> int:
    pixels = np.asarray(Image.open(io.BytesIO(locator.screenshot())).convert("RGB")).reshape(-1, 3)
    return len(np.unique(pixels, axis=0))


def test_atoms_draw_on_the_shared_device(silicon, serve, open_page, hardware_adapter, browser_out):
    page = open_page(f"{serve(browser_out)}/{silicon}", webgpu=True)
    page.wait_for_selector('canvas[aria-label="3D atomic structure"]', timeout=60_000)
    page.wait_for_timeout(3_000)
    hardware_adapter(page.adapter())
    page.screenshot(path=str(browser_out / "showcif_silicon.png"), full_page=True)
    assert page.locator('[data-render-path="none"]').count() == 0
    assert distinct_colours(page.locator('canvas[aria-label="3D atomic structure"]')) > 20
    assert distinct_colours(page.locator('canvas[aria-label="Atomic column projection"]')) > 20
    assert "WebGPU error" not in page.locator(".qcif").inner_text()
    assert page.errors == [], page.errors
