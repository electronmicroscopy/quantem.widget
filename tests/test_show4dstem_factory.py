from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace

import numpy as np

import quantem.widget.show4dstem_factory as factory


LoadResult = namedtuple("LoadResult", ["data", "metadata"])


def test_public_show4dstem_import_uses_factory() -> None:
    from quantem.widget import Show4DSTEM

    assert Show4DSTEM is factory.Show4DSTEM


def test_show4dstem_routes_chunked_payload_to_mps_builder(monkeypatch) -> None:
    payload = SimpleNamespace(chunks=[object()], metadata={"scan_shape": (2, 2)})
    calls = []

    def _fake_mps_builder(data, **kwargs):
        calls.append((data, kwargs))
        return "mps-viewer"

    monkeypatch.setattr(factory, "_build_mps_viewer", _fake_mps_builder)

    result = factory.Show4DSTEM(payload, title="mps")

    assert result == "mps-viewer"
    assert calls == [(payload, {"title": "mps"})]
    assert factory.show4dstem_backend_kind(payload) == "mps"


def test_show4dstem_routes_loadresult_chunked_payload_to_mps_builder(monkeypatch) -> None:
    payload = SimpleNamespace(chunks=[object()], metadata={"scan_shape": (2, 2)})
    load_result = LoadResult(payload, {"file_names": ["a"]})
    calls = []

    def _fake_mps_builder(data, **kwargs):
        calls.append((data, kwargs))
        return "mps-viewer"

    monkeypatch.setattr(factory, "_build_mps_viewer", _fake_mps_builder)

    result = factory.Show4DSTEM(load_result, verbose=False)

    assert result == "mps-viewer"
    assert calls == [(payload, {"verbose": False})]


def test_show4dstem_routes_mps_gpu_frame_proxy_to_mps_builder(monkeypatch) -> None:
    payload = SimpleNamespace(_is_gpu_frames=True, device="mps:0")

    def _fake_mps_builder(data, **kwargs):
        return {"kind": "mps", "data": data, "kwargs": kwargs}

    monkeypatch.setattr(factory, "_build_mps_viewer", _fake_mps_builder)

    result = factory.Show4DSTEM(payload, fast_interaction=True)

    assert result == {
        "kind": "mps",
        "data": payload,
        "kwargs": {"fast_interaction": True},
    }
    assert factory.is_mps_show4dstem_payload(payload)


def test_show4dstem_keeps_cuda_gpu_frame_proxy_on_base_viewer(monkeypatch) -> None:
    payload = SimpleNamespace(_is_gpu_frames=True, device="cuda:0", ndim=4)

    def _fake_base(data, **kwargs):
        return {"kind": "base", "data": data, "kwargs": kwargs}

    monkeypatch.setattr(factory, "_Show4DSTEMBase", _fake_base)

    result = factory.Show4DSTEM(payload, verbose=False)

    assert result == {"kind": "base", "data": payload, "kwargs": {"verbose": False}}
    assert factory.show4dstem_backend_kind(payload) == "base"


def test_show4dstem_labels_5d_loadresult_as_dataset_stack(monkeypatch) -> None:
    payload = SimpleNamespace(ndim=5)
    load_result = LoadResult(payload, {"file_names": ("first.h5", "second.h5")})

    def _fake_base(data, **kwargs):
        return {"kind": "base", "data": data, "kwargs": kwargs}

    monkeypatch.setattr(factory, "_Show4DSTEMBase", _fake_base)

    result = factory.Show4DSTEM(load_result, verbose=False)

    assert result["data"] is load_result
    assert result["kwargs"]["frame_dim_label"] == "Dataset"
    assert result["kwargs"]["frame_labels"] == ["first.h5", "second.h5"]
    assert result["kwargs"]["verbose"] is False


def test_show4dstem_uses_lazy_macbook_handle_directly(monkeypatch) -> None:
    from quantem.widget.multidataset_mps import LazyMacbookDatasets

    lazy = LazyMacbookDatasets(
        masters=["first_master.h5"],
        det_bin=4,
        names=["first"],
        multi=object(),
        decode=lambda path: object(),
        verbose=False,
    )

    def _fake_build_viewer(**kwargs):
        return {"kind": "lazy-mps", "kwargs": kwargs}

    monkeypatch.setattr(lazy, "build_viewer", _fake_build_viewer)

    result = factory.Show4DSTEM(lazy, ui_mode="report")

    assert result == {"kind": "lazy-mps", "kwargs": {"ui_mode": "report"}}
    assert factory.show4dstem_backend_kind(lazy) == "mps"


def test_public_show4dstem_constructs_small_binned_numpy_viewer() -> None:
    from quantem.widget import Show4DSTEM
    from quantem.widget.show4dstem import Show4DSTEM as Show4DSTEMBase

    data = np.arange(2 * 2 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4)
    widget = Show4DSTEM(
        data,
        precompute_virtual_images=False,
        verbose=False,
        ui_mode="minimal",
    )

    try:
        assert isinstance(widget, Show4DSTEMBase)
        assert widget.shape_rows == 2
        assert widget.shape_cols == 2
        assert widget.det_rows == 4
        assert widget.det_cols == 4
        assert widget.show_controls is False
    finally:
        widget.close()
