from __future__ import annotations

import inspect
import json
import pathlib

import numpy as np
import pytest

from quantem.widget.export import HTML_EXPORT_TRAITS, supports_html_export
from quantem.widget.show2d import Show2D
from quantem.widget.show3d import Show3D
from quantem.widget.show3dslices import Show3DSlices
from quantem.widget.show4dstem import Show4DSTEM
from quantem.widget.showdiffraction import ShowDiffraction
from quantem.widget.showeds import ShowEDS


EXPORT_WIDGET_CLASSES = (Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS, ShowDiffraction)


def _show2d() -> Show2D:
    data = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    return Show2D(data, title="Protocol Show2D", cmap="viridis", verbose=False)


def _show3d() -> Show3D:
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    return Show3D(data, title="Protocol Show3D", cmap="magma")


def _show3dslices() -> Show3DSlices:
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    return Show3DSlices(data, title="Protocol Show3DSlices", cmap="plasma")


def _show4dstem() -> Show4DSTEM:
    data = np.arange(2 * 2 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4)
    return Show4DSTEM(data, title="Protocol Show4DSTEM", verbose=False)


def _showeds() -> ShowEDS:
    data = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    return ShowEDS(data, title="Protocol ShowEDS", band=(1, 4), roi=(0, 1, 2, 2))


def _showdiffraction() -> ShowDiffraction:
    data = np.random.rand(48, 48).astype(np.float32)
    return ShowDiffraction(data, title="Protocol ShowDiffraction", verbose=False)


EXPORT_WIDGET_CASES = (
    pytest.param(_show2d, {"encoding": "full"}, {"mode": "single", "encoding": "full"}, "Protocol Show2D", id="show2d"),
    pytest.param(_show3d, {"encoding": "full"}, {"mode": "single", "encoding": "full"}, "Protocol Show3D", id="show3d"),
    pytest.param(_show3dslices, {"encoding": "full"}, {"mode": "single", "encoding": "full"}, "Protocol Show3DSlices", id="show3dslices"),
    pytest.param(_show4dstem, {"encoding": "uint8", "downsample": 1}, {"mode": "single", "encoding": "uint8", "downsample": 1}, "Protocol Show4DSTEM", id="show4dstem"),
    pytest.param(_showeds, {"mode": "single", "encoding": "full"}, {"mode": "single", "encoding": "full"}, "Protocol ShowEDS", id="showeds"),
    pytest.param(_showdiffraction, {"encoding": "full"}, {"mode": "single", "encoding": "full"}, "Protocol ShowDiffraction", id="showdiffraction"),
)


def test_widgets_expose_structural_html_export_protocol() -> None:
    for cls in EXPORT_WIDGET_CLASSES:
        assert supports_html_export(cls)
        signature = inspect.signature(cls.export_html)
        assert "path" in signature.parameters
        assert signature.parameters["path"].default is None
        assert "title" in signature.parameters
        assert signature.parameters["title"].default is None
        assert signature.return_annotation in {pathlib.Path, "pathlib.Path"}


def test_widgets_expose_standard_frontend_html_export_traits() -> None:
    for cls in EXPORT_WIDGET_CLASSES:
        trait_names = set(cls.class_trait_names())
        assert set(HTML_EXPORT_TRAITS).issubset(trait_names)


@pytest.mark.parametrize(("factory", "export_kwargs", "request_payload", "state_marker"), EXPORT_WIDGET_CASES)
def test_widget_export_html_writes_standalone_state(
    tmp_path: pathlib.Path,
    factory,
    export_kwargs: dict,
    request_payload: dict,
    state_marker: str,
) -> None:
    widget = factory()
    filename_stem = f"{request_payload['mode']}_{request_payload.get('encoding', 'full')}_{request_payload.get('downsample', 1)}"

    assert supports_html_export(widget)
    out = widget.export_html(tmp_path / f"{filename_stem}.html", **export_kwargs)

    assert out == tmp_path / f"{filename_stem}.html"
    assert out.exists()
    html = out.read_text()
    assert "application/vnd.jupyter.widget-state+json" in html
    assert state_marker in html
    assert widget.export_status.startswith(f"Exported {out.name}")


@pytest.mark.parametrize(("factory", "export_kwargs", "request_payload", "state_marker"), EXPORT_WIDGET_CASES)
def test_widget_frontend_export_request_creates_payload_and_status(
    factory,
    export_kwargs: dict,
    request_payload: dict,
    state_marker: str,
) -> None:
    widget = factory()
    filename = f"{request_payload['mode']}_{request_payload.get('encoding', 'full')}_{request_payload.get('downsample', 1)}.html"
    payload = {"id": "req-protocol", "filename": filename, "download": True, **request_payload}

    widget.export_request = json.dumps(payload)

    assert widget.export_payload_id == "req-protocol"
    assert widget.export_filename == filename
    assert state_marker.encode() in widget.export_payload
    assert widget.export_status.startswith(f"Ready {filename}")
    for expected_trait in HTML_EXPORT_TRAITS:
        assert widget.has_trait(expected_trait)
