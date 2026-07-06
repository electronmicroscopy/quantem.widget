import numpy as np


def test_load_stacked_u8_routes_to_direct_output_dtype(monkeypatch):
    """Public dtype='u8' must reach stacked list loads before materializing U16."""
    from quantem.widget.io import hdf5

    calls = {}

    def fake_load_impl(filepath, *args, **kwargs):
        calls["filepath"] = filepath
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((2, 1, 1, 1, 1), dtype=np.uint8),
            {"file_names": ["a", "b"]},
        )

    monkeypatch.setattr(hdf5, "_load_impl", fake_load_impl)

    hdf5.load(["a_master.h5", "b_master.h5"], dtype="u8", verbose=False)

    assert calls["filepath"] == ["a_master.h5", "b_master.h5"]
    assert calls["kwargs"]["output_dtype"] is np.uint8


def test_load_sharded_u8_routes_to_direct_output_dtype(monkeypatch):
    """devices=[...] list loads should also use the direct U8 browse path."""
    from quantem.widget.io import hdf5

    calls = {}

    def fake_load_impl(filepath, *args, **kwargs):
        calls["filepath"] = filepath
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            {0: np.zeros((1, 1, 1, 1, 1), dtype=np.uint8)},
            {"sharded": True},
        )

    monkeypatch.setattr(hdf5, "_load_impl", fake_load_impl)

    hdf5.load(
        ["a_master.h5", "b_master.h5"],
        devices=[0, 1],
        dtype="uint8",
        verbose=False,
    )

    assert calls["filepath"] == ["a_master.h5", "b_master.h5"]
    assert calls["kwargs"]["devices"] == [0, 1]
    assert calls["kwargs"]["output_dtype"] is np.uint8


def test_load_parallel_u8_routes_to_direct_output_dtype(monkeypatch):
    """gpus=/stack=False placement should not decode as U16 first."""
    from quantem.widget.io import hdf5

    calls = {}

    def fake_many_parallel(masters, *, gpus=None, max_concurrent=None, verbose=False, **kwargs):
        calls["masters"] = masters
        calls["gpus"] = gpus
        calls["max_concurrent"] = max_concurrent
        calls["verbose"] = verbose
        calls["kwargs"] = kwargs
        return [hdf5.LoadResult(np.zeros((1, 1, 1, 1), dtype=np.uint8), {})]

    monkeypatch.setattr(hdf5, "_load_many_parallel", fake_many_parallel)

    hdf5.load(
        ["a_master.h5", "b_master.h5"],
        gpus=[0, 1],
        stack=False,
        dtype="u8",
        verbose=False,
    )

    assert calls["masters"] == ["a_master.h5", "b_master.h5"]
    assert calls["gpus"] == [0, 1]
    assert calls["kwargs"]["output_dtype"] is np.uint8


def test_load_u8_does_not_override_explicit_output_dtype(monkeypatch):
    """Explicit lower-level output_dtype remains authoritative."""
    from quantem.widget.io import hdf5

    calls = {}

    def fake_load_impl(filepath, *args, **kwargs):
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((1, 1, 1), dtype=np.float16),
            {},
        )

    monkeypatch.setattr(hdf5, "_load_impl", fake_load_impl)

    hdf5.load("a_master.h5", dtype="u8", output_dtype=np.float16, verbose=False)

    assert calls["kwargs"]["output_dtype"] is np.float16
