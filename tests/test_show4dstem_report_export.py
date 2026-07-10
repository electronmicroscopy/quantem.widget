from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantem.widget import Show4DSTEM
from quantem.widget.data.dataset5dstem import Dataset5dstem


def test_show4dstem_export_real_space_and_detector_bin_use_mean() -> None:
    raw = np.arange(4 * 4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4, 4)
    widget = Show4DSTEM(
        torch.as_tensor(raw),
        center=(2, 2),
        bf_radius=1,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        exported = widget._export_data_array(dtype="uint16", det_bin=2, scan_bin=2)
    finally:
        widget.close()

    expected = raw.reshape(2, 2, 2, 2, 2, 2, 2, 2).mean(axis=(1, 3, 5, 7))
    expected = np.round(expected).astype(np.uint16)
    assert exported.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(exported, expected)


def test_show4dstem_export_state_scales_scan_and_detector_coordinates() -> None:
    widget = Show4DSTEM(
        torch.ones((4, 4, 8, 8), dtype=torch.uint8),
        sampling=(0.5, 0.5, 0.1, 0.1),
        center=(4, 4),
        bf_radius=2,
        precompute_virtual_images=False,
        verbose=False,
    )
    widget.pos_row = 2
    widget.pos_col = 3
    widget.vi_roi_center_row = 2
    widget.vi_roi_center_col = 2
    widget.vi_vmin = 1.0
    widget.vi_vmax = 9.0

    try:
        state = widget._export_state_for_bin(2, scan_bin=2)
    finally:
        widget.close()

    assert state["center_row"] == 2.0
    assert state["center_col"] == 2.0
    assert state["bf_radius"] == 1.0
    assert state["pos_row"] == 1.0
    assert state["pos_col"] == 1.5
    assert state["vi_roi_center_row"] == 1.0
    assert state["pixel_size"] == 1.0
    assert state["k_pixel_size"] == 0.2
    assert state["dp_vmin"] is None
    assert state["vi_vmin"] is None


def test_show4dstem_report_export_pages_lazy_data_without_raw_payload() -> None:
    calls: list[int] = []

    def make_loader(idx: int):
        def load():
            calls.append(idx)
            return torch.full((4, 4, 8, 8), idx + 1, dtype=torch.uint8)

        return load

    data = Dataset5dstem.from_lazy_loaders(
        [make_loader(idx) for idx in range(5)],
        shape=(5, 4, 4, 8, 8),
        dtype=torch.uint8,
        initial_frames={0: torch.ones((4, 4, 8, 8), dtype=torch.uint8)},
    )
    widget = Show4DSTEM(
        data,
        view_mode="multiple",
        compare_max_panels=2,
        page_budget=2,
        center=(4, 4),
        bf_radius=2,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        html = widget._html_report_export_bytes(
            dtype="uint8",
            det_bin=2,
            scan_bin=2,
            dataset_scope="unhidden",
        ).decode("utf-8")
        resident = set(data.loaded_indices())
    finally:
        widget.close()

    assert "Static report export: virtual-image PNGs only" in html
    assert "Raw interactive 4D data is not embedded" in html
    assert 'data-preset="bf"' in html
    assert 'data-preset="haadf"' in html
    assert "_offline_stack" not in html
    assert calls == [1, 2, 3, 4]
    assert resident.issubset({0, 1})


def test_show4dstem_report_export_request_prepares_download_payload() -> None:
    widget = Show4DSTEM(
        torch.ones((2, 4, 4, 8, 8), dtype=torch.uint8),
        view_mode="multiple",
        compare_max_panels=1,
        center=(4, 4),
        bf_radius=2,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        widget.export_request = json.dumps({
            "export_kind": "report",
            "mode": "uint8-bin2",
            "dtype": "uint8",
            "det_bin": 2,
            "scan_bin": 2,
            "dataset_scope": "current_page",
            "id": "report-request",
            "filename": "show4dstem-report.html",
            "download": True,
        })
        html = bytes(widget.export_payload).decode("utf-8")
    finally:
        widget.close()

    assert widget.export_payload_id == "report-request"
    assert widget.export_filename == "show4dstem-report.html"
    assert "Ready show4dstem-report.html" in widget.export_status
    assert "Static report export" in html
