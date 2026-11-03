"""Prepared detector results remain usable after later queries and exports."""

import pytest
import torch

from quantem.widget import Show4DSTEM
from quantem.widget.show4dstem.dense import DenseSession


@pytest.mark.parametrize("backend", ["widget", "gpu"])
def test_detector_results_keep_their_values_after_query_and_close(backend):
    values_t = torch.arange(5 * 7 * 16 * 16).reshape(5, 7, 16, 16)
    if backend == "gpu":
        session = pytest.importorskip("quantem.gpu.detector").prepare(values_t)
    else:
        session = DenseSession(values_t)
    mask_t = torch.zeros((16, 16), dtype=torch.bool)
    mask_t[2:6, 5:9] = True
    try:
        first = session.masked_sum(mask_t.numpy())
        frozen_t = torch.as_tensor(first).clone()
        expected_t = values_t[..., mask_t].sum(-1).float()
        torch.testing.assert_close(frozen_t, expected_t, rtol=0, atol=0)
        second = session.masked_sum((~mask_t).numpy())
        assert not torch.equal(torch.as_tensor(second), frozen_t)
    finally:
        session.close()
    torch.testing.assert_close(torch.as_tensor(first), frozen_t, rtol=0, atol=0)


def test_prepared_session_exports_results_without_packing_raw_counts(tmp_path):
    session = DenseSession(torch.arange(5 * 7 * 16 * 16).reshape(5, 7, 16, 16))
    session.metadata = {}
    viewer = Show4DSTEM(session, bright_field_center_px=(7.5, 7.5),
                       bright_field_radius_px=3)
    try:
        raw_page = tmp_path / "interactive.html"
        with pytest.raises(NotImplementedError, match="export_kind='report'"):
            viewer.export_html(raw_page)
        assert not raw_page.exists()
        report = viewer.export_html(tmp_path / "report.html", export_kind="report")
        assert "data:image/png;base64," in report.read_text()
    finally:
        viewer.free()
        viewer.close()
