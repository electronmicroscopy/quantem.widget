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


def test_show4dstem_compare_grid_builds_virtual_image_stack() -> None:
    from quantem.widget import Show4DSTEM

    data = np.arange(5 * 3 * 4 * 6 * 6, dtype=np.uint16).reshape(5, 3, 4, 6, 6)
    widget = Show4DSTEM(
        data,
        view_mode="compare",
        compare_cols=3,
        compare_grid_width_px=720,
        compare_max_panels=4,
        frame_dim_label="Dataset",
        frame_labels=[f"scan-{idx}" for idx in range(5)],
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.view_mode == "compare"
        assert widget.compare_cols == 3
        assert widget.compare_grid_width_px == 720
        assert widget.compare_dp_mode == "average"
        assert widget.compare_panel_count == 4
        assert widget.compare_panel_indices == [0, 1, 2, 3]
        assert len(widget.compare_virtual_image_bytes) == 4 * 3 * 4 * 4
        assert "4/5 dataset panels" in widget.compare_status
        average_dp = widget.frame_bytes
        widget.compare_dp_mode = "selected"
        assert widget.compare_dp_mode == "selected"
        assert widget.frame_bytes != average_dp

        widget.compare_dp_mode = "avg"
        assert widget.compare_dp_mode == "average"

        widget.set_compare_panel_order(["scan-3", "scan-0", "scan-1", "scan-2", "scan-4"])
        assert widget.compare_panel_order == [3, 0, 1, 2, 4]
        assert widget.compare_panel_indices == [3, 0, 1, 2]

        widget.hide_compare_panel("scan-0")
        assert widget.compare_hidden_panels == [0]
        assert widget.compare_panel_indices == [3, 1, 2, 4]
        widget.show_compare_panel("scan-0")
        assert widget.compare_hidden_panels == []

        widget.star_compare_panel("scan-3")
        widget.star_compare_panel(1)
        widget.unstar_compare_panel(1)
        assert widget.compare_starred_panels == [3]

        try:
            widget.set_compare_hidden_panels([0, 1, 2, 3, 4])
        except ValueError as exc:
            assert "hide every panel" in str(exc)
        else:  # pragma: no cover - assertion helper
            raise AssertionError("Show4DSTEM allowed hiding every compare panel")

        before = widget.compare_virtual_image_bytes
        widget.apply_preset("adf")
        assert widget.compare_panel_count == 4
        assert widget.compare_virtual_image_bytes != before

        state = widget.state_dict()
        restored = Show4DSTEM(
            data,
            view_mode="compare",
            compare_cols=1,
            compare_max_panels=4,
            frame_dim_label="Dataset",
            frame_labels=[f"scan-{idx}" for idx in range(5)],
            precompute_virtual_images=False,
            verbose=False,
        )
        try:
            restored.load_state_dict(state)
            assert restored.compare_dp_mode == widget.compare_dp_mode
            assert restored.compare_grid_width_px == widget.compare_grid_width_px
            assert restored.compare_panel_order == widget.compare_panel_order
            assert restored.compare_hidden_panels == widget.compare_hidden_panels
            assert restored.compare_starred_panels == widget.compare_starred_panels
            assert restored.compare_panel_indices == widget.compare_panel_indices
        finally:
            restored.close()

        widget.view_mode = "single"
        widget._refresh_compare_virtual_images()
        assert widget.compare_panel_count == 0
        assert widget.compare_virtual_image_bytes == b""
    finally:
        widget.close()


def test_show4dstem_compare_grid_normalizes_detector_roi_preview() -> None:
    from quantem.widget import Show4DSTEM

    data = np.zeros((2, 2, 2, 4, 4), dtype=np.uint16)
    data[0] = 4
    data[1] = 7
    widget = Show4DSTEM(
        data,
        view_mode="compare",
        compare_max_panels=2,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        widget.roi_mode = "rect"
        widget.roi_center_row = 1.5
        widget.roi_center_col = 1.5
        widget.roi_width = 4
        widget.roi_height = 4
        widget._compute_virtual_image_from_roi()
        widget._refresh_compare_virtual_images()

        main_vi = np.frombuffer(widget.virtual_image_bytes, dtype=np.float32).reshape(2, 2)
        compare_vi = np.frombuffer(
            widget.compare_virtual_image_bytes, dtype=np.float32
        ).reshape(2, 2, 2)

        np.testing.assert_allclose(main_vi, np.full((2, 2), 64, dtype=np.float32))
        np.testing.assert_allclose(compare_vi[0], np.full((2, 2), 4, dtype=np.float32))
        np.testing.assert_allclose(compare_vi[1], np.full((2, 2), 7, dtype=np.float32))
    finally:
        widget.close()


def test_show4dstem_compare_grid_validates_api() -> None:
    from quantem.widget import Show4DSTEM

    data = np.zeros((2, 2, 2, 4, 4), dtype=np.uint16)

    for kwargs, message in [
        ({"view_mode": "movie"}, "view_mode"),
        ({"compare_layout": "diagonal"}, "compare_layout"),
        ({"compare_cols": -1}, "compare_cols"),
        ({"compare_grid_width_px": -1}, "compare_grid_width_px"),
        ({"compare_max_panels": 0}, "compare_max_panels"),
        ({"compare_dp_mode": "median"}, "compare_dp_mode"),
    ]:
        try:
            Show4DSTEM(data, verbose=False, **kwargs)
        except ValueError as exc:
            assert message in str(exc)
        else:  # pragma: no cover - assertion helper
            raise AssertionError(f"Show4DSTEM accepted invalid kwargs {kwargs!r}")
