"""ChooseLattice on a HiDPI screen (device pixel ratio 2), in headed Chrome.

The image and the point-marker overlay draw on backing stores of device
pixels, so neither is upscaled blurry, and a click still lands on the image
pixel under the pointer.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

IMAGE = 'canvas[data-quantem-scientific-output="choose-lattice-image"]'


@pytest.fixture(scope="module")
def lattice_page(browser_out) -> str:
    from quantem.widget import ChooseLattice
    from quantem.widget.export import write_widget_html

    rows, cols = np.indices((128, 128))
    widget = ChooseLattice((np.sin(rows / 4.0) * np.cos(cols / 5.0)).astype(np.float32))
    write_widget_html(browser_out / "chooselattice_hidpi.html", widget, "chooselattice")
    widget.close()
    return "chooselattice_hidpi.html"


def test_hidpi_canvases_and_click_coordinates(lattice_page, chrome, serve, browser_out):
    context = chrome.new_context(viewport={"width": 1200, "height": 900}, device_scale_factor=2)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)[:300]))
    page.goto(f"{serve(browser_out)}/{lattice_page}")
    page.wait_for_selector(IMAGE, timeout=60_000)
    page.wait_for_timeout(1_000)
    sizes = page.evaluate(f"""() => {{
      const image = document.querySelector('{IMAGE}');
      return [image, image.nextElementSibling].map(canvas => [canvas.width, canvas.getBoundingClientRect().width]);
    }}""")
    box = page.locator(IMAGE).bounding_box()
    # 128 image pixels fill the 512 px canvas: the centre of pixel (row 32, col 96) is at (96.5, 32.5) * 4 px.
    page.mouse.click(box["x"] + 96.5 * 4, box["y"] + 32.5 * 4)
    page.wait_for_timeout(800)
    origin = page.get_by_text("Origin:").text_content()
    page.screenshot(path=str(browser_out / "chooselattice_hidpi.png"))
    context.close()
    for backing, css in sizes:
        assert backing == 2 * css, sizes
    # The browser rounds the click to a CSS pixel, a quarter of an image pixel here.
    row, col = (float(value) for value in origin.removeprefix("Origin: (").removesuffix(")").split(", "))
    assert abs(row - 32.5) <= 0.25 and abs(col - 96.5) <= 0.25, origin
    assert errors == [], errors
