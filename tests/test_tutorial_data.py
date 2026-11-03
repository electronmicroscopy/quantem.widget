import json
from pathlib import Path

import numpy as np
import pytest

from quantem.widget import datasets


def test_show2d_gold_size_maps_to_preview_stride(tmp_path, monkeypatch, capsys):
    root = tmp_path / "hf-cache"
    data_dir = root / "haadf" / "gold_haadf_npy"
    data_dir.mkdir(parents=True)
    np.save(data_dir / "data.npy", np.arange(64, dtype=np.float32).reshape(8, 8))
    (data_dir / "meta.json").write_text(
        json.dumps({"name": "gold_haadf_npy", "sampling": [0.2, 0.2], "units": ["nm", "nm"]})
    )
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return root

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    dataset = datasets.show2d_gold(size="medium", verbose=False)

    assert dataset.array.shape == (2, 2)
    assert tuple(dataset.sampling) == (0.8, 0.8)
    assert calls == [
        {
            "repo_id": "bobleesj/quantem-data",
            "repo_type": "dataset",
            "allow_patterns": ["haadf/gold_haadf_npy/*"],
            "force_download": False,
            "revision": datasets.TUTORIAL_DATA_REVISION,
        }
    ]
    # E-R3: the stride was printed only with verbose=True and never named the way back to native pixels
    assert capsys.readouterr().out == (
        "show2d_gold(size='medium') keeps every 4th pixel of the 8x8 image (2x2); size='full' loads native pixels.\n")
    datasets.show2d_gold(size="full", verbose=False)
    assert capsys.readouterr().out == ""


def test_show3d_gold_reuses_shared_haadf_source(tmp_path, monkeypatch):
    root = tmp_path / "hf-cache"
    data_dir = root / "haadf" / "gold_haadf_npy"
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

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    dataset = datasets.show3d_gold(size="small", verbose=False)

    assert dataset.array.shape == (32, 256, 256)
    assert calls[0]["allow_patterns"] == ["haadf/gold_haadf_npy/*"]


def test_tutorial_size_names_are_strict():
    assert datasets._normalize_tutorial_size("Small") == "small"
    try:
        datasets._normalize_tutorial_size("mini")
    except ValueError as exc:
        assert "small, medium, large, full" in str(exc)
    else:
        raise AssertionError("expected invalid tutorial size to raise")


def test_show4dstem_gold_uses_widget_tutorial_source(tmp_path, monkeypatch, capsys):
    root = tmp_path / "hf-cache"
    data_dir = root / "4dstem" / "gold_128_npy_bin8"
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

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    dataset = datasets.show4dstem_gold(size="small", verbose=False)

    assert dataset.shape == (2, 2, 2, 2)  # a Dataset4dstemGPU on a GPU, a core Dataset4dstem elsewhere
    assert tuple(dataset.sampling) == (8.0, 8.0, 3.68, 3.68)
    assert "keeps every 4th scan position of the 8x8 scan" in capsys.readouterr().out
    assert calls[0]["allow_patterns"] == ["4dstem/gold_128_npy_bin8/*"]


@pytest.mark.parametrize("revision,path", [
    (None, "widget/ducky"),
    ("recorded", "widget/ducky"),
])
def test_show1d_ducky_downloads_scoped_input_family(tmp_path, monkeypatch, revision, path):
    root = tmp_path / "hf-cache"
    folder = root / path
    folder.mkdir(parents=True)
    (folder / "show1d_monitor.jsonl").write_text('{"iteration": 0, "losses": {"lambda 1": 1.0}}\n')
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return root

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    result = datasets.show1d_ducky(cache_dir=Path("/tmp/qw-cache"), revision=revision, verbose=False)

    assert result == folder
    assert calls == [
        {
            "repo_id": "bobleesj/quantem-data",
            "repo_type": "dataset",
            "allow_patterns": [f"{path}/*"],
            "force_download": False,
            "revision": revision or datasets.TUTORIAL_DATA_REVISION,
            "cache_dir": "/tmp/qw-cache",
        }
    ]


@pytest.mark.parametrize("revision,path", [
    (None, "widget/gold_session"),
    ("recorded", "widget/gold_session"),
])
def test_gold_session_uses_recorded_input_family(tmp_path, monkeypatch, revision, path):
    root = tmp_path / "hf-cache"
    folder = root / path
    folder.mkdir(parents=True)
    (folder / "0010 - HAADF 15Mx Nano.emd").write_bytes(b"placeholder")
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return root

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    result = datasets.gold_session(size="small", cache_dir=Path("/tmp/qw-cache"), revision=revision, verbose=False)

    assert result == folder
    assert calls == [
        {
            "repo_id": "bobleesj/quantem-data",
            "repo_type": "dataset",
            "allow_patterns": [f"{path}/*"],
            "force_download": False,
            "revision": revision or datasets.TUTORIAL_DATA_REVISION,
            "cache_dir": "/tmp/qw-cache",
        }
    ]


@pytest.mark.parametrize("revision,path", [("staged-revision", "widget/ducky"), ("another-revision", "widget/ducky")])
def test_ducky_revision_keeps_snapshot_family_together(tmp_path, monkeypatch, revision, path):
    folder = tmp_path / path
    (folder / "snapshots").mkdir(parents=True)
    reference = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.save(folder / "snapshots/reference.npy", reference)
    (folder / "show1d_monitor.jsonl").write_text(json.dumps({"iteration": 0, "snapshots": {"reference": "snapshots/reference.npy"}}) + "\n")
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: calls.append(kwargs) or tmp_path)
    result = datasets.show1d_ducky(revision=revision, verbose=False)
    event = json.loads((result / "show1d_monitor.jsonl").read_text())
    np.testing.assert_array_equal(np.load(result / event["snapshots"]["reference"]), reference)
    assert calls[0]["allow_patterns"] == ["widget/ducky/*"]
    assert all(call["revision"] == revision for call in calls)
    assert calls[-1]["allow_patterns"] == [f"{path}/*"]


def test_session_accepts_new_location_without_changing_filenames(tmp_path, monkeypatch):
    folder = tmp_path / "widget/gold_session"
    folder.mkdir(parents=True)
    (folder / "0010 - HAADF 15Mx Nano.emd").write_bytes(b"session input")
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: calls.append(kwargs) or tmp_path)
    result = datasets.gold_session(revision="staged-revision", verbose=False)
    assert [p.name for p in result.glob("*.emd")] == ["0010 - HAADF 15Mx Nano.emd"]
    assert calls[0]["allow_patterns"] == ["widget/gold_session/*"]
    assert calls[0]["revision"] == "staged-revision"


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
