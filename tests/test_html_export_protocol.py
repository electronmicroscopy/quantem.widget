import inspect
import json
import pathlib

import numpy as np
import pytest
from ipywidgets.widgets.widget import _instances

from quantem.widget._diffraction_sim_crystal import require_crystal_tools
from quantem.widget.diffraction_sim import DiffractionSim
from quantem.widget.show1d import Show1D
from quantem.widget.show2d import Show2D
from quantem.widget.show3d import Show3D
from quantem.widget.show3dslices import Show3DSlices
from quantem.widget import Show4DSTEM
from quantem.widget.showdiffraction import ShowDiffraction

from quantem.widget.showpdf import ShowPDF, _pdf_class

try:  # ShowPDF needs quantem's PairDistributionFunction API (quantem pull request 248)
    _pdf_class()
except ImportError:
    ShowPDF = None

EXPORT_WIDGET_CLASSES = (Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowDiffraction) + (
    (ShowPDF,) if ShowPDF is not None else ()
)

try:  # DiffractionSim needs quantem.diffraction (not installed on Intel Macs and Windows on ARM)
    require_crystal_tools()
    EXPORT_WIDGET_CLASSES += (DiffractionSim,)
    _DIFFRACTION_SIM_MISSING = ""
except ImportError as exc:
    _DIFFRACTION_SIM_MISSING = str(exc)


def _show1d() -> Show1D:
    data = np.asarray([[3.0, 2.0, 1.0], [2.5, 1.7, 1.1]], dtype=np.float32)
    return Show1D(data, x=[0, 1, 2], labels=["loss", "validation"], title="Protocol Show1D")


def _show2d() -> Show2D:
    data = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    return Show2D(data, title="Protocol Show2D", cmap="viridis", verbose=False)


def test_show2d_html_export_preserves_per_panel_cmaps(tmp_path) -> None:
    data = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    widget = Show2D(data, title="Panel Cmaps", cmap=["inferno", "viridis"], verbose=False)

    path = widget.export_html(tmp_path / "panel-cmaps.html")
    html = path.read_text()

    assert "panel_cmaps" in html
    assert "inferno" in html
    assert "viridis" in html


def test_show3d_html_export_preserves_per_panel_cmaps(tmp_path) -> None:
    data_a = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    data_b = np.flip(data_a, axis=-1).copy()
    widget = Show3D(data_a, data_b, title="Panel Cmaps", cmap="inferno", verbose=False)
    widget.panel_cmaps = ["inferno", "viridis"]

    path = widget.export_html(tmp_path / "show3d-panel-cmaps.html")
    html = path.read_text()

    assert "panel_cmaps" in html
    assert "inferno" in html
    assert "viridis" in html


def _show3d() -> Show3D:
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    return Show3D(data, title="Protocol Show3D", cmap="magma")


def _show3dslices() -> Show3DSlices:
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    return Show3DSlices(data, title="Protocol Show3DSlices", cmap="plasma")


def _show4dstem() -> Show4DSTEM:
    data = np.arange(2 * 2 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4)
    return Show4DSTEM(data, title="Protocol Show4DSTEM", verbose=False)


def _showdiffraction() -> ShowDiffraction:
    data = np.random.rand(48, 48).astype(np.float32)
    return ShowDiffraction(data, title="Protocol ShowDiffraction", verbose=False)


def _diffractionsim() -> DiffractionSim:
    return DiffractionSim("Al (fcc)", zone_axis=[0, 0, 1], title="Protocol DiffractionSim")


def _showpdf():
    rr, cc = np.mgrid[0:16, 0:16]
    rad = np.hypot(rr - 7.5, cc - 7.5)
    pattern = (np.exp(-((rad - 4.0) ** 2) / 3.0) + 0.1).astype(np.float32)
    data = np.broadcast_to(pattern, (2, 2, 16, 16)).copy()
    return ShowPDF(data, title="Protocol ShowPDF", find_origin=False, origin_row=7.5, origin_col=7.5)


# The request payloads are the ones each toolbar sends (js/<widget>/index.tsx), with the packing its status line names.
EXPORT_WIDGET_CASES = (
    pytest.param(_show1d, {"image_bin": 2}, {}, "full float32", "Protocol Show1D", id="show1d"),
    pytest.param(_show2d, {"encoding": "uint8"}, {"encoding": "uint8"}, "uint8", "Protocol Show2D", id="show2d"),
    pytest.param(_show3d, {"encoding": "uint8", "image_bin": 2}, {"encoding": "uint8", "image_bin": 2}, "uint8, image bin 2x", "Protocol Show3D", id="show3d"),
    pytest.param(_show3dslices, {"encoding": "full"}, {"encoding": "uint8"}, "uint8", "Protocol Show3DSlices", id="show3dslices"),
    pytest.param(_show4dstem, {"dtype": "uint8"}, {"export_kind": "interactive", "dtype": "uint8", "detector_bin": 1, "scan_bin": 1, "dataset_scope": "unhidden"},
                 "uint8", "Protocol Show4DSTEM", id="show4dstem"),
    pytest.param(_showdiffraction, {}, {}, "full float32", "Protocol ShowDiffraction", id="showdiffraction"),
) + (
    (pytest.param(_showpdf, {"encoding": "uint8"}, {"mode": "single", "encoding": "uint8"},
                  "exact float32 curves + quantized uint8 profiles", "Protocol ShowPDF", id="showpdf"),)
    if ShowPDF is not None
    else ()
) + (
    pytest.param(_diffractionsim, {}, {}, "1 crystal, no image data", "Protocol DiffractionSim", id="diffractionsim",
                 marks=pytest.mark.skipif(bool(_DIFFRACTION_SIM_MISSING), reason=_DIFFRACTION_SIM_MISSING)),
)


def test_widgets_expose_structural_html_export_protocol() -> None:
    for cls in EXPORT_WIDGET_CLASSES:
        assert callable(cls.export_html)
        signature = inspect.signature(cls.export_html)
        assert "path" in signature.parameters
        assert signature.parameters["path"].default is None
        assert "title" in signature.parameters
        assert signature.parameters["title"].default is None
        assert signature.return_annotation in {pathlib.Path, "pathlib.Path"}


HTML_EXPORT_TRAITS = (
    "export_request",
    "export_status",
    "export_enabled",
    "export_payload",
    "export_payload_id",
    "export_filename",
)


def test_widgets_expose_standard_frontend_html_export_traits() -> None:
    for cls in EXPORT_WIDGET_CLASSES:
        trait_names = set(cls.class_trait_names())
        assert set(HTML_EXPORT_TRAITS).issubset(trait_names)


@pytest.mark.parametrize(("factory", "export_kwargs", "request_payload", "packing", "state_marker"), EXPORT_WIDGET_CASES)
def test_widget_export_html_writes_standalone_state(
    tmp_path: pathlib.Path,
    factory,
    export_kwargs: dict,
    request_payload: dict,
    packing: str,
    state_marker: str,
) -> None:
    widget = factory()

    assert callable(widget.export_html)
    open_models = set(_instances)
    out = widget.export_html(tmp_path / "page.html", **export_kwargs)

    # the page is written from a clone: it and its Layout must close, or every export leaves models open
    assert set(_instances) == open_models
    assert out == tmp_path / "page.html"
    assert out.exists()
    html = out.read_text()
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert 'id="quantem-widget-export-layout"' in html
    assert "application/vnd.jupyter.widget-state+json" in html
    assert state_marker in html
    assert widget.export_status.startswith(f"Exported {out.name}")


@pytest.mark.parametrize(("factory", "export_kwargs", "request_payload", "packing", "state_marker"), EXPORT_WIDGET_CASES)
def test_widget_frontend_export_request_creates_payload_and_status(
    factory,
    export_kwargs: dict,
    request_payload: dict,
    packing: str,
    state_marker: str,
) -> None:
    widget = factory()
    payload = {"id": "req-protocol", "filename": "toolbar.html", "download": True, **request_payload}

    widget.export_request = json.dumps(payload)

    assert widget.export_payload_id == "req-protocol"
    assert widget.export_filename == "toolbar.html"
    assert state_marker.encode() in widget.export_payload
    # the status names the packing the request asked for
    assert widget.export_status.startswith("Ready toolbar.html") and widget.export_status.endswith(f", {packing})")
    for expected_trait in HTML_EXPORT_TRAITS:
        assert widget.has_trait(expected_trait)


def test_a_single_file_export_opens_without_a_network(tmp_path) -> None:
    """rc41: the page loaded require.js, html-manager and anywidget from CDNs and stayed blank offline."""
    import base64
    import gzip
    import re

    page = Show2D(np.zeros((4, 4), np.float32)).export_html(tmp_path / "offline.html").read_text()
    assert "https://" not in "".join(re.findall(r"<script[^>]*>", page))
    runtime = [gzip.decompress(base64.b64decode(blob)).decode() for blob in
               re.findall(r'<script type="application/gzip" id="quantem-widget-runtime-\d">([^<]+)</script>', page)]
    assert len(runtime) == 3 and "requirejs" in runtime[0] and 'define("anywidget",["@jupyter-widgets/base"]' in runtime[1]
    assert 'define("@jupyter-widgets/base"' in runtime[2]
