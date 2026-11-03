"""Every widget bundle mounts through anywidget's default export, with no deprecation warning.

anywidget logs "Direct export of a 'render' will likely be deprecated" for
every view of a bundle that exports ``render`` by name instead of
``export default { render }``. Each widget is embedded in a standalone page
and opened in headed Chrome; the page must draw the widget and log no such
warning.
"""

import numpy as np
import pytest
from ase import Atoms

import quantem.widget as qw
from quantem.widget.export import write_widget_html
from quantem.widget.showpdf import _pdf_class

pytestmark = pytest.mark.browser

RNG = np.random.default_rng(0)
IMAGE = RNG.random((32, 32)).astype(np.float32)
WIDGETS = {
    "chooselattice": lambda: qw.ChooseLattice(IMAGE),
    "plot2d": lambda: qw.Plot2D(IMAGE[:4, :6], x=np.arange(6), y=np.arange(4)),
    "show1d": lambda: qw.Show1D(RNG.random(64).astype(np.float32)),
    "show2d": lambda: qw.Show2D(IMAGE),
    "show3d": lambda: qw.Show3D(RNG.random((3, 16, 16)).astype(np.float32)),
    "show3dslices": lambda: qw.Show3DSlices(RNG.random((8, 8, 8)).astype(np.float32)),
    "show4dstem": lambda: qw.Show4DSTEM(RNG.integers(0, 50, (4, 4, 16, 16)).astype(np.uint16), device="cpu"),
    "showcif": lambda: qw.ShowCIF(Atoms("Si2", scaled_positions=[(0, 0, 0), (0.25, 0.25, 0.25)], cell=[5.43] * 3, pbc=True)),
    "showdiffraction": lambda: qw.ShowDiffraction(RNG.random((32, 32)).astype(np.float32), device="cpu", verbose=False),
    "showptycho": lambda: qw.ShowPtycho(np.exp(1j * IMAGE).astype(np.complex64), sampling=0.2),
}


def _showpdf():
    rr, cc = np.mgrid[0:16, 0:16]
    pattern = (np.exp(-((np.hypot(rr - 7.5, cc - 7.5) - 4.0) ** 2) / 3.0) + 0.1).astype(np.float32)
    data = np.broadcast_to(pattern, (2, 2, 16, 16)).copy()
    return qw.ShowPDF(data, find_origin=False, origin_row=7.5, origin_col=7.5)


try:  # ShowPDF needs quantem's PairDistributionFunction API (quantem pull request 248)
    _pdf_class()
    WIDGETS["showpdf"] = _showpdf
except ImportError:
    pass
# The widget's own DOM inside the embed's view container: anywidget renders into it
# (without a kernel, Show2D and Show3D draw their saved preview image).
WIDGET_DRAWN = "() => Array.from(document.querySelectorAll('.widget-subarea')).some(area => area.querySelector('canvas, button, svg, input, img'))"


@pytest.mark.parametrize("name", sorted(WIDGETS))
def test_widget_mounts_without_the_direct_export_warning(name, serve, open_page, hardware_adapter, browser_out):
    widget = WIDGETS[name]()
    write_widget_html(browser_out / f"entry_{name}.html", widget, name)
    widget.close()
    page = open_page(f"{serve(browser_out)}/entry_{name}.html", webgpu=True)
    hardware_adapter(page.adapter())
    page.wait_for_function(WIDGET_DRAWN, timeout=60_000)
    page.wait_for_timeout(1_000)
    page.screenshot(path=str(browser_out / f"entry_{name}.png"), full_page=True)
    deprecations = [message for message in page.console if "Direct export of a 'render'" in message]
    assert deprecations == [], (name, deprecations)
    assert page.errors == [], (name, page.errors)
