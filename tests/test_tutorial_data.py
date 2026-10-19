import json
from pathlib import Path

import numpy as np
import pytest

from quantem.widget import datasets


def test_show2d_gold_size_maps_to_preview_stride(tmp_path, monkeypatch):
    root = tmp_path / "hf-cache"
    data_dir = root / "widget-tutorials" / "shared" / "gold-haadf" / "full"
    data_dir.mkdir(parents=True)
    np.save(data_dir / "data.npy", np.arange(64, dtype=np.float32).reshape(8, 8))
    (data_dir / "meta.json").write_text(
        json.dumps({"name": "gold_haadf_npy", "sampling": [0.2, 0.2], "units": ["nm", "nm"]})
    )
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return root

    monkeypatch.setattr(datasets, "snapshot_download", fake_snapshot_download, raising=False)

    dataset = datasets.show2d_gold(size="medium", verbose=False)

    assert dataset.array.shape == (2, 2)
    assert tuple(dataset.sampling) == (0.8, 0.8)
    assert calls == [
        {
            "repo_id": "bobleesj/quantem-data",
            "repo_type": "dataset",
            "allow_patterns": ["widget-tutorials/shared/gold-haadf/full/*"],
            "force_download": False,
        }
    ]


def test_show3d_gold_reuses_shared_haadf_source(tmp_path, monkeypatch):
    root = tmp_path / "hf-cache"
    data_dir = root / "widget-tutorials" / "shared" / "gold-haadf" / "full"
    data_dir.mkdir(parents=True)
    image = np.arange(2048 * 2048, dtype=np.float32).reshape(2048, 2048)
    np.save(data_dir / "data.npy", image)
    (data_dir / "meta.json").write_text(
        json.dumps({"name": "gold_haadf_npy", "sampling": [0.2, 0.2], "units": ["nm", "nm"]})
    )
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return root

    monkeypatch.setattr(datasets, "snapshot_download", fake_snapshot_download, raising=False)

    dataset = datasets.show3d_gold(size="small", verbose=False)

    assert dataset.array.shape == (32, 256, 256)
    assert calls[0]["allow_patterns"] == ["widget-tutorials/shared/gold-haadf/full/*"]


def test_tutorial_size_names_are_strict():
    assert datasets._normalise_tutorial_size("Small") == "small"
    try:
        datasets._normalise_tutorial_size("mini")
    except ValueError as exc:
        assert "small, medium, large, full" in str(exc)
    else:
        raise AssertionError("expected invalid tutorial size to raise")


def test_show4dstem_gold_uses_widget_tutorial_source(tmp_path, monkeypatch):
    root = tmp_path / "hf-cache"
    data_dir = root / "widget-tutorials" / "show4dstem" / "gold-128-bin8" / "full"
    data_dir.mkdir(parents=True)
    stack = np.arange(8 * 8 * 2 * 2, dtype=np.uint16).reshape(8, 8, 2, 2)
    np.save(data_dir / "data.npy", stack)
    (data_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "gold_128_npy_bin8",
                "sampling": [2.0, 2.0, 3.68, 3.68],
                "units": ["A", "A", "mrad", "mrad"],
            }
        )
    )
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return root

    monkeypatch.setattr(datasets, "snapshot_download", fake_snapshot_download, raising=False)

    dataset = datasets.show4dstem_gold(size="small", verbose=False)

    assert dataset.shape == (2, 2, 2, 2)  # a Dataset4dstemGPU on a GPU, a core Dataset4dstem elsewhere
    assert tuple(dataset.sampling) == (8.0, 8.0, 3.68, 3.68)
    assert calls[0]["allow_patterns"] == ["widget-tutorials/show4dstem/gold-128-bin8/full/*"]


def test_show1d_ducky_downloads_scoped_widget_tutorial_folder(tmp_path, monkeypatch):
    root = tmp_path / "hf-cache"
    folder = root / "widget-tutorials" / "show1d" / "ducky" / "small"
    folder.mkdir(parents=True)
    (folder / "show1d_monitor.jsonl").write_text('{"iteration": 0, "losses": {"lambda 1": 1.0}}\n')
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return root

    monkeypatch.setattr(datasets, "snapshot_download", fake_snapshot_download, raising=False)

    result = datasets.show1d_ducky(size="small", cache_dir=Path("/tmp/qw-cache"), verbose=False)

    assert result == folder
    assert calls == [
        {
            "repo_id": "bobleesj/quantem-data",
            "repo_type": "dataset",
            "allow_patterns": ["widget-tutorials/show1d/ducky/small/*"],
            "force_download": False,
            "cache_dir": "/tmp/qw-cache",
        }
    ]


def test_showdiffraction_fe3o4_uses_widget_tutorial_source(tmp_path, monkeypatch):
    root = tmp_path / "hf-cache"
    data_dir = root / "widget-tutorials" / "showdiffraction" / "fe3o4-saed" / "small"
    data_dir.mkdir(parents=True)
    np.save(data_dir / "data.npy", np.arange(16, dtype=np.float32).reshape(4, 4))
    (data_dir / "meta.json").write_text(json.dumps({"name": "fe3o4-saed"}))
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return root

    monkeypatch.setattr(datasets, "snapshot_download", fake_snapshot_download, raising=False)

    pattern = datasets.showdiffraction_fe3o4(verbose=False)

    assert pattern.shape == (4, 4)
    assert pattern.dtype == np.float32
    assert calls == [
        {
            "repo_id": "bobleesj/quantem-data",
            "repo_type": "dataset",
            "allow_patterns": ["widget-tutorials/showdiffraction/fe3o4-saed/small/*"],
            "force_download": False,
        }
    ]


def test_showdiffraction_fe3o4_raises_without_usable_source(tmp_path, monkeypatch):
    def fail_download(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(datasets, "snapshot_download", fail_download, raising=False)

    try:
        datasets.showdiffraction_fe3o4(verbose=False)
    except OSError as exc:
        assert "offline" in str(exc)
    else:
        raise AssertionError("expected diffraction tutorial loader to raise")


def test_gold_session_uses_widget_tutorial_folder(tmp_path, monkeypatch):
    root = tmp_path / "hf-cache"
    folder = root / "widget-tutorials" / "showfolder" / "gold-haadf-session" / "small"
    folder.mkdir(parents=True)
    (folder / "0010 - HAADF 15Mx Nano.emd").write_bytes(b"placeholder")
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return root

    monkeypatch.setattr(datasets, "snapshot_download", fake_snapshot_download, raising=False)

    result = datasets.gold_session(size="small", cache_dir=Path("/tmp/qw-cache"), verbose=False)

    assert result == folder
    assert calls == [
        {
            "repo_id": "bobleesj/quantem-data",
            "repo_type": "dataset",
            "allow_patterns": ["widget-tutorials/showfolder/gold-haadf-session/small/*"],
            "force_download": False,
            "cache_dir": "/tmp/qw-cache",
        }
    ]


def test_showpdf_ta_loads_the_calibrated_scan_and_the_model_gr(tmp_path, monkeypatch):
    root = tmp_path / "hf-cache"
    data_dir = root / "4dstem" / "ta_amorphous_sim"
    data_dir.mkdir(parents=True)
    np.save(data_dir / "data.npy", np.arange(2 * 2 * 4 * 4, dtype=np.float32).reshape(2, 2, 4, 4))
    (data_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "ta_amorphous_sim",
                "sampling": [1.0, 1.0, 0.01488, 0.01488],
                "units": ["pixels", "pixels", "A^-1", "A^-1"],
            }
        )
    )
    np.savez(data_dir / "ground_truth_gr.npz", r=np.array([0.0, 0.05]), gr=np.array([0.0, 1.0]))
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return root

    monkeypatch.setattr(datasets, "snapshot_download", fake_snapshot_download, raising=False)

    scan = datasets.showpdf_ta(revision="abc123", verbose=False)
    r, gr = datasets.showpdf_ta_ground_truth(revision="abc123")

    assert scan.array.shape == (2, 2, 4, 4)
    assert [float(value) for value in scan.sampling] == [1.0, 1.0, 0.01488, 0.01488]
    assert list(scan.units) == ["pixels", "pixels", "A^-1", "A^-1"]
    assert r.tolist() == [0.0, 0.05]
    assert gr.tolist() == [0.0, 1.0]
    assert calls == 2 * [
        {
            "repo_id": "bobleesj/quantem-data",
            "repo_type": "dataset",
            "allow_patterns": ["4dstem/ta_amorphous_sim/*"],
            "force_download": False,
            "revision": "abc123",
        }
    ]


def test_showpdf_ta_names_the_hugging_face_path_of_a_missing_folder(tmp_path, monkeypatch):
    # Hugging Face returns the snapshot without the folder when no file matches allow_patterns.
    monkeypatch.setattr(datasets, "snapshot_download", lambda **kwargs: tmp_path, raising=False)

    with pytest.raises(FileNotFoundError, match="bobleesj/quantem-data/4dstem/ta_amorphous_sim$"):
        datasets.showpdf_ta(verbose=False)
