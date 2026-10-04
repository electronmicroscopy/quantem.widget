"""The live comparison borrows measurements and reads bounded scan regions."""

from types import SimpleNamespace

import pytest
import torch

from quantem.widget import Show4DSTEM
from quantem.gpu.io.models import Dataset4dstemGPU


def test_resident_before_after_uses_same_scan_and_aperture(monkeypatch):
    device = 'cuda:0' if torch.cuda.is_available() else 'mps'
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Accelerator required')
    values_t = torch.arange(48*48*16*16, device=device).reshape(48,48,16,16).float() % 32
    reads = []

    def read(*, scan_region):
        r0, r1, c0, c1 = scan_region
        reads.append((r1-r0)*(c1-c0))
        return values_t[r0:r1,c0:c1]

    source = Dataset4dstemGPU(values_t, {"device": device})
    original_read = Dataset4dstemGPU.read

    def tracked_read(self, **kwargs):
        if self is source and kwargs.get("scan_region") is not None:
            r0, r1, c0, c1 = kwargs["scan_region"]
            reads.append((r1 - r0) * (c1 - c0))
        return original_read(self, **kwargs)

    monkeypatch.setattr(Dataset4dstemGPU, "read", tracked_read)
    after = Dataset4dstemGPU(values_t / 2, {"device": device})
    viewer = Show4DSTEM([source, after], center=(7.5,7.5), bf_radius=3)
    try:
        mask_t = torch.ones(16,16, dtype=torch.bool, device=device)
        image_t = torch.as_tensor(viewer._compute.masked_sum(mask_t.cpu().numpy()), device=device)
        torch.testing.assert_close(image_t, values_t.sum((-1,-2)), rtol=0, atol=0)
        viewer.pos_row, viewer.pos_col = 12, 19
        first_t = viewer._diffraction_frame_for_index(0)
        second_t = viewer._diffraction_frame_for_index(1)
        torch.testing.assert_close(first_t, values_t[12,19], rtol=0, atol=0)
        torch.testing.assert_close(second_t, first_t / 2, rtol=0, atol=0)
        viewer.frame_idx = 1
        after_t = torch.as_tensor(viewer._compute.masked_sum(mask_t.cpu().numpy()), device=device)
        torch.testing.assert_close(after_t, image_t / 2, rtol=0, atol=0)
        panels = viewer._compare_virtual_images_for_indices([0, 1], mask_t)
        torch.testing.assert_close(torch.as_tensor(panels[0], device=device),
                                   image_t / 256, rtol=0, atol=0)
        torch.testing.assert_close(torch.as_tensor(panels[1], device=device),
                                   image_t / 512, rtol=0, atol=0)
        assert max(reads) <= 32
    finally:
        viewer.close()
    torch.testing.assert_close(read(scan_region=(0,1,0,1)), values_t[:1,:1], rtol=0, atol=0)


def test_bounded_view_exposes_detector_source_for_native_reductions():
    """quantem.gpu routes resident detector sums through ANS kernels only when
    the wrapper names its owner and region; a missing name silently falls back
    to decoding the region per query."""
    from quantem.widget.show4dstem_bounded import _View

    values_t = torch.zeros(8, 8, 4, 4)
    source = SimpleNamespace(shape=values_t.shape, read=lambda *, scan_region: values_t, metadata={'device': 'cpu'})
    view = _View(source, (2, 6, 1, 5))
    assert view._detector_source is source
    assert view._detector_region == (2, 6, 1, 5)
    assert _View(source)._detector_region == (0, 8, 0, 8)
