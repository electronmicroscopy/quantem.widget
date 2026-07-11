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


def test_show4dstem_accepts_read_only_numpy_without_torch_warning(recwarn) -> None:
    from quantem.widget import Show4DSTEM

    data = np.arange(2 * 2 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4)
    data.setflags(write=False)
    widget = Show4DSTEM(data, precompute_virtual_images=False, verbose=False)

    try:
        assert widget.shape_rows == 2
        assert widget.shape_cols == 2
        assert widget.det_rows == 4
        assert widget.det_cols == 4
    finally:
        widget.close()

    assert not any("not writable" in str(warning.message) for warning in recwarn)


def test_show4dstem_compare_grid_builds_virtual_image_stack() -> None:
    from quantem.widget import Show4DSTEM

    data = np.arange(5 * 3 * 4 * 6 * 6, dtype=np.uint16).reshape(5, 3, 4, 6, 6)
    widget = Show4DSTEM(
        data,
        view_mode="multiple",
        compare_cols=3,
        compare_grid_width_px=720,
        compare_panel_gap_px=3,
        compare_max_panels=4,
        frame_dim_label="Dataset",
        frame_labels=[f"scan-{idx}" for idx in range(5)],
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.view_mode == "multiple"
        assert widget.compare_cols == 3
        assert widget.compare_grid_width_px == 720
        assert widget.compare_panel_gap_px == 3
        assert widget.compare_dp_mode == "average"
        assert widget.compare_group_mode == "paged"
        assert widget.compare_panel_count == 4
        assert widget.compare_panel_indices == [0, 1, 2, 3]
        assert len(widget.compare_virtual_image_bytes) == 4 * 3 * 4 * 4
        assert "4/5 dataset panels" in widget.compare_status
        widget.show_compare_all_groups()
        assert widget.compare_group_mode == "all"
        assert widget.compare_panel_count == 5
        assert widget.compare_panel_indices == [0, 1, 2, 3, 4]
        assert "all groups" in widget.compare_status
        widget.show_compare_paged_groups()
        assert widget.compare_group_mode == "paged"
        assert widget.compare_panel_indices == [0, 1, 2, 3]
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
        assert widget.compare_panel_indices == [3, 1, 2]
        widget.show_compare_panel("scan-0")
        assert widget.compare_hidden_panels == []
        assert widget.compare_panel_indices == [3, 0, 1, 2]

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
            view_mode="multiple",
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
            assert restored.compare_group_mode == widget.compare_group_mode
            assert restored.compare_grid_width_px == widget.compare_grid_width_px
            assert restored.compare_panel_gap_px == widget.compare_panel_gap_px
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


def test_show4dstem_5d_offline_save_state_embeds_inline_stack() -> None:
    from quantem.widget import Show4DSTEM

    data = np.arange(3 * 2 * 2 * 4 * 4, dtype=np.uint16).reshape(3, 2, 2, 4, 4)
    widget = Show4DSTEM(
        data,
        view_mode="multiple",
        offline=True,
        offline_dtype="uint16",
        save_state=True,
        compare_max_panels=3,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.offline is True
        assert widget._offline_gzip is True
        assert len(widget._offline_stack) > 0
        state = widget.get_state(drop_defaults=False)
        assert state["offline"] is True
        assert state["_offline_stack"] == widget._offline_stack
    finally:
        widget.close()


def test_show4dstem_compare_grid_pages_panels() -> None:
    from quantem.widget import Show4DSTEM

    data = np.arange(7 * 3 * 4 * 6 * 6, dtype=np.uint16).reshape(7, 3, 4, 6, 6)
    widget = Show4DSTEM(
        data,
        view_mode="multiple",
        compare_max_panels=3,
        frame_dim_label="Dataset",
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.compare_page_progressive_enabled is False
        assert widget.compare_page_count == 3
        assert widget.compare_page_idx == 0
        assert widget.compare_panel_indices == [0, 1, 2]
        assert "page 1/3" in widget.compare_status

        widget.set_compare_page(1)

        assert widget.compare_page_idx == 1
        assert widget.compare_panel_indices == [3, 4, 5]
        assert widget.frame_idx == 3
        assert "page 2/3" in widget.compare_status

        widget.next_compare_page()
        assert widget.compare_panel_indices == [6]
        assert widget.compare_page_idx == 2
        assert widget.frame_idx == 6

        widget.next_compare_page()
        assert widget.compare_page_idx == 2

        widget.previous_compare_page()
        assert widget.compare_panel_indices == [3, 4, 5]
        assert widget.frame_idx == 3

        widget.frame_idx = 6
        assert widget.compare_page_idx == 2
        assert widget.compare_panel_indices == [6]

        widget.frame_idx = 1
        assert widget.compare_page_idx == 0
        assert widget.compare_panel_indices == [0, 1, 2]

        widget.show_compare_all_groups()
        assert widget.compare_group_mode == "all"
        assert widget.compare_page_count == 3
        assert widget.compare_panel_count == 7
        assert widget.compare_panel_indices == [0, 1, 2, 3, 4, 5, 6]
        assert "all groups" in widget.compare_status

        widget.show_compare_paged_groups()
        assert widget.compare_group_mode == "paged"
        assert widget.compare_page_idx == 0
        assert widget.compare_panel_indices == [0, 1, 2]

        widget.hide_compare_panel(1)
        assert widget.compare_page_idx == 0
        assert widget.compare_page_count == 3
        assert widget.compare_panel_indices == [0, 2]
        assert widget.frame_idx == 0
        assert "hidden" in widget.compare_status

        widget.set_compare_page(1)
        assert widget.compare_panel_indices == [3, 4, 5]
    finally:
        widget.close()


def test_show4dstem_multiple_init_renders_compare_grid_once(monkeypatch) -> None:
    from quantem.widget import Show4DSTEM
    from quantem.widget.show4dstem import Show4DSTEM as Show4DSTEMBase

    data = np.arange(3 * 2 * 2 * 4 * 4, dtype=np.uint16).reshape(3, 2, 2, 4, 4)
    calls = {"virtual": 0, "compare": 0}
    original_virtual = Show4DSTEMBase._compute_virtual_image_from_roi
    original_compare = Show4DSTEMBase._refresh_compare_virtual_images

    def count_virtual(self):
        calls["virtual"] += 1
        return original_virtual(self)

    def count_compare(self):
        calls["compare"] += 1
        return original_compare(self)

    monkeypatch.setattr(
        Show4DSTEMBase,
        "_compute_virtual_image_from_roi",
        count_virtual,
    )
    monkeypatch.setattr(
        Show4DSTEMBase,
        "_refresh_compare_virtual_images",
        count_compare,
    )

    widget = Show4DSTEM(
        data,
        view_mode="multiple",
        compare_max_panels=3,
        center=(1.5, 1.5),
        bf_radius=2,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert calls == {"virtual": 1, "compare": 1}
        assert widget.compare_panel_count == 3
    finally:
        widget.close()


def test_show4dstem_free_reports_data_freed_for_multiple_grid() -> None:
    from quantem.widget import Show4DSTEM

    data = np.arange(2 * 2 * 2 * 4 * 4, dtype=np.uint16).reshape(2, 2, 2, 4, 4)
    widget = Show4DSTEM(
        data,
        view_mode="multiple",
        compare_max_panels=2,
        center=(1.5, 1.5),
        bf_radius=2,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.compare_panel_count == 2
        widget.free()
        assert widget.compare_panel_count == 0
        assert "data was freed" in widget.compare_status
    finally:
        widget.close()


def test_show4dstem_compare_preset_cache_reuses_multiple_grid(monkeypatch) -> None:
    from quantem.widget import Show4DSTEM
    from quantem.widget.show4dstem import Show4DSTEM as Show4DSTEMBase

    data = np.arange(3 * 2 * 2 * 6 * 6, dtype=np.uint16).reshape(3, 2, 2, 6, 6)
    widget = Show4DSTEM(
        data,
        view_mode="multiple",
        compare_max_panels=3,
        center=(2.5, 2.5),
        bf_radius=2,
        precompute_virtual_images=False,
        verbose=False,
    )

    calls = {"compare_grid": 0}
    original = Show4DSTEMBase._compare_virtual_images_for_indices

    def count_compare_grid(self, indices, mask):
        calls["compare_grid"] += 1
        return original(self, indices, mask)

    monkeypatch.setattr(
        Show4DSTEMBase,
        "_compare_virtual_images_for_indices",
        count_compare_grid,
    )
    try:
        widget._refresh_compare_virtual_images()
        assert calls["compare_grid"] == 0

        widget.apply_preset("adf")
        assert calls["compare_grid"] == 1

        widget.apply_preset("adf")
        assert calls["compare_grid"] == 1
    finally:
        widget.close()


def test_show4dstem_frame_virtual_image_uses_sparse_detector_mask(monkeypatch) -> None:
    import torch

    from quantem.widget import Show4DSTEM
    from quantem.widget.show4dstem import Show4DSTEM as Show4DSTEMBase

    data = torch.arange(2 * 4 * 4 * 16 * 16, dtype=torch.int32).to(torch.uint16)
    data = data.reshape(2, 4, 4, 16, 16)
    widget = Show4DSTEM(
        data,
        view_mode="multiple",
        compare_max_panels=2,
        center=(8, 8),
        bf_radius=4,
        precompute_virtual_images=False,
        verbose=False,
    )

    calls = {"sparse": 0}
    original = Show4DSTEMBase._sparse_masked_sum_tensor_for_frame_data

    def count_sparse(self, frame_data, mask):
        calls["sparse"] += 1
        return original(self, frame_data, mask)

    monkeypatch.setattr(
        Show4DSTEMBase,
        "_sparse_masked_sum_tensor_for_frame_data",
        count_sparse,
    )
    try:
        widget.roi_mode = "annular"
        widget.roi_center_row = 8
        widget.roi_center_col = 8
        widget.roi_radius_inner = 2
        widget.roi_radius = 4
        calls["sparse"] = 0
        vi = widget._virtual_image_for_frame(0)

        mask = widget._current_detector_mask().bool()
        expected = data[0].float()[:, :, mask].sum(dim=2).numpy()

        assert calls["sparse"] == 1
        np.testing.assert_allclose(vi, expected)
    finally:
        widget.close()


def test_torch_backend_sparse_masked_sum_matches_dense_reference() -> None:
    import torch

    from quantem.widget.kernels.compute.backends import compute_backend

    data = torch.arange(4 * 4 * 16 * 16, dtype=torch.int32).to(torch.uint16)
    data = data.reshape(4, 4, 16, 16)
    yy, xx = np.ogrid[:16, :16]
    mask = ((yy - 8) ** 2 + (xx - 8) ** 2 <= 3**2).astype(np.float32)
    backend = compute_backend(data)

    sparse = backend._sparse_masked_sum(mask)
    vi = backend.masked_sum(mask)
    expected = (data.float() * torch.as_tensor(mask)).sum(dim=(2, 3)).numpy()

    assert sparse is not None
    np.testing.assert_allclose(sparse, expected)
    np.testing.assert_allclose(vi, expected)


def test_show4dstem_compare_grid_normalizes_detector_roi_preview() -> None:
    from quantem.widget import Show4DSTEM

    data = np.zeros((2, 2, 2, 4, 4), dtype=np.uint16)
    data[0] = 4
    data[1] = 7
    widget = Show4DSTEM(
        data,
        view_mode="multiple",
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


def test_show4dstem_single_view_refreshes_after_multiple_mode() -> None:
    from quantem.widget import Show4DSTEM

    data = np.empty((2, 2, 2, 4, 4), dtype=np.uint16)
    data[0] = 1
    data[1] = 7
    widget = Show4DSTEM(
        data,
        view_mode="multiple",
        compare_max_panels=2,
        center=(1.5, 1.5),
        bf_radius=2,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        widget.frame_idx = 1
        multiple_dp = np.frombuffer(widget.frame_bytes, dtype=np.float32).reshape(4, 4)
        np.testing.assert_allclose(multiple_dp, np.full((4, 4), 4, dtype=np.float32))

        widget.view_mode = "single"
        single_dp = np.frombuffer(widget.frame_bytes, dtype=np.float32).reshape(4, 4)
        single_vi = np.frombuffer(widget.virtual_image_bytes, dtype=np.float32).reshape(2, 2)

        np.testing.assert_allclose(single_dp, np.full((4, 4), 7, dtype=np.float32))
        assert float(single_vi.min()) > 0
        np.testing.assert_allclose(single_vi, np.full((2, 2), single_vi[0, 0], dtype=np.float32))
    finally:
        widget.close()


def test_show4dstem_view_mode_legacy_aliases() -> None:
    from quantem.widget import Show4DSTEM

    data = np.zeros((2, 2, 2, 4, 4), dtype=np.uint16)

    compare_alias = Show4DSTEM(data, view_mode="compare", verbose=False)
    temporal_alias = Show4DSTEM(data, view_mode="temporal", verbose=False)
    try:
        assert compare_alias.view_mode == "multiple"
        assert temporal_alias.view_mode == "single"
    finally:
        compare_alias.close()
        temporal_alias.close()


def test_show4dstem_compare_grid_validates_api() -> None:
    from quantem.widget import Show4DSTEM

    data = np.zeros((2, 2, 2, 4, 4), dtype=np.uint16)

    for kwargs, message in [
        ({"view_mode": "movie"}, "view_mode"),
        ({"compare_layout": "diagonal"}, "compare_layout"),
        ({"compare_cols": -1}, "compare_cols"),
        ({"compare_grid_width_px": -1}, "compare_grid_width_px"),
        ({"compare_panel_gap_px": -1}, "compare_panel_gap_px"),
        ({"compare_max_panels": 0}, "compare_max_panels"),
        ({"compare_group_mode": "merged"}, "compare_group_mode"),
        ({"compare_dp_mode": "median"}, "compare_dp_mode"),
    ]:
        try:
            Show4DSTEM(data, verbose=False, **kwargs)
        except ValueError as exc:
            assert message in str(exc)
        else:  # pragma: no cover - assertion helper
            raise AssertionError(f"Show4DSTEM accepted invalid kwargs {kwargs!r}")
