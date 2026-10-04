"""Resident viewer loading policy contracts."""

from pathlib import Path
from types import SimpleNamespace

from quantem.widget.io.resident import load_resident


def test_load_resident_defaults_to_native_median_hot_pixels(monkeypatch):
    calls = []
    expected = SimpleNamespace()

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr("quantem.widget.io.resident.io.load", fake_load)

    loaded = load_resident(
        Path("scan_master.h5"),
        backend="cuda",
        representation="encoded",
        device=0,
    )

    assert loaded is expected
    assert calls == [
        (
            (Path("scan_master.h5"),),
            {
                "backend": "cuda",
                "representation": "encoded",
                "dtype": "native",
                "device": 0,
                "apply_mask": False,
                "hot_pixel_correction": "median",
                "verbose": False,
            },
        )
    ]
