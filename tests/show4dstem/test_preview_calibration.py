"""Saved previews retain the field of view when their display size changes."""

from types import SimpleNamespace

import numpy as np
import pytest

from quantem.widget.show4dstem import preview


@pytest.mark.parametrize("width", [64, 256, 512])
@pytest.mark.parametrize("panel_key", ["virtual", "diffraction"])
def test_preview_scale_bar_uses_source_pixels(monkeypatch, width, panel_key):
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    widget = SimpleNamespace(
        vi_smooth=False, show_scale_bar=True, scan_sampling_A=[0.3, 0.5],
        detector_sampling_mrad=[0.1, 0.2], detector_sampling_inv_A=[],
    )
    monkeypatch.setattr(preview, "virtual_rgb", lambda _: rgb)
    monkeypatch.setattr(preview, "diffraction_rgb", lambda _: rgb)
    monkeypatch.setattr(preview, "_draw_virtual_overlays", lambda *_: None)
    monkeypatch.setattr(preview, "_draw_diffraction_overlays", lambda *_: None)
    bars = []
    monkeypatch.setattr(preview, "_draw_scalebar", lambda image, step, unit: bars.append((image.width * step, unit)))
    preview._panel(widget, panel_key, width)
    expected = (48 * 0.5, "angstrom") if panel_key == "virtual" else (48 * 0.2, "mrad")
    assert bars == [expected]
    widget.show_scale_bar = False
    preview._panel(widget, panel_key, width)
    assert bars == [expected]
