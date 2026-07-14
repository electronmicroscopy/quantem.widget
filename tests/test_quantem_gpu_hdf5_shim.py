from __future__ import annotations


def test_widget_hdf5_reexports_quantem_gpu_loader() -> None:
    import importlib.util
    from pathlib import Path

    import quantem.gpu.io.hdf5 as gpu_hdf5

    shim_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "quantem"
        / "widget"
        / "io"
        / "hdf5.py"
    )
    spec = importlib.util.spec_from_file_location("widget_hdf5_shim", shim_path)
    assert spec is not None
    assert spec.loader is not None
    widget_hdf5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(widget_hdf5)

    assert widget_hdf5.load is gpu_hdf5.load
    assert widget_hdf5.bin is gpu_hdf5.bin
    assert widget_hdf5.LoadResult is gpu_hdf5.LoadResult
