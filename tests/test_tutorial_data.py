import json

import numpy as np

from quantem.widget.data import tutorials


def test_load_tutorial_show2d_uses_calibrated_preview(tmp_path, monkeypatch):
    data_dir = tmp_path / "gold_haadf_npy"
    data_dir.mkdir()
    np.save(data_dir / "data.npy", np.arange(16, dtype=np.float32).reshape(4, 4))
    (data_dir / "meta.json").write_text(json.dumps({"name": "gold_haadf_npy", "sampling": [0.2, 0.2], "units": ["nm", "nm"]}))
    monkeypatch.setattr(tutorials, "download", lambda *args, **kwargs: data_dir)

    dataset = tutorials.load_tutorial_show2d(stride=2, verbose=False)

    np.testing.assert_array_equal(dataset.array, np.array([[0, 2], [8, 10]], dtype=np.float32))
    assert tuple(dataset.sampling) == (0.4, 0.4)
    assert tuple(dataset.units) == ("nm", "nm")


def test_load_tutorial_show3d_uses_real_image_crops(tmp_path, monkeypatch):
    data_dir = tmp_path / "gold_haadf_npy"
    data_dir.mkdir()
    image = np.arange(30 * 30, dtype=np.float32).reshape(30, 30)
    np.save(data_dir / "data.npy", image)
    (data_dir / "meta.json").write_text(json.dumps({"name": "gold_haadf_npy", "sampling": [0.2, 0.2], "units": ["nm", "nm"]}))
    monkeypatch.setattr(tutorials, "download", lambda *args, **kwargs: data_dir)

    dataset = tutorials.load_tutorial_show3d(n_frames=3, stride=1, crop_size=16, verbose=False)

    assert dataset.array.shape == (3, 16, 16)
    np.testing.assert_array_equal(dataset.array[0], image[0:16, 14:30])
    np.testing.assert_array_equal(dataset.array[-1], image[14:30, 0:16])
    assert tuple(dataset.sampling) == (1.0, 0.2, 0.2)
    assert tuple(dataset.units) == ("frame", "nm", "nm")


def test_load_tutorial_show4dstem_preserves_uint16_counts(tmp_path, monkeypatch):
    data_dir = tmp_path / "gold_128_npy_bin8"
    data_dir.mkdir()
    stack = np.arange(4 * 4 * 2 * 2, dtype=np.uint16).reshape(4, 4, 2, 2)
    stack[2, 2, 1, 1] = 4096
    np.save(data_dir / "data.npy", stack)
    (data_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "gold_128_npy_bin8",
                "sampling": [2.0, 2.0, 3.68, 3.68],
                "units": ["A", "A", "mrad", "mrad"],
                "processing": "test",
            }
        )
    )
    monkeypatch.setattr(tutorials, "download", lambda *args, **kwargs: data_dir)

    dataset = tutorials.load_tutorial_show4dstem(scan_stride=2, verbose=False)

    assert dataset.array.dtype == np.uint16
    np.testing.assert_array_equal(dataset.array, stack[::2, ::2])
    assert tuple(dataset.sampling) == (4.0, 4.0, 3.68, 3.68)


def test_create_tutorial_showfolder_folder_writes_velox_like_session(tmp_path):
    from quantem.widget.showfolder_core import build_showfolder, has_eds, scan_rotation_deg

    folder = tutorials.create_tutorial_showfolder_folder(tmp_path / "showfolder-demo")
    files = sorted(folder.glob("*.emd"))

    assert len(files) == 4
    assert sum(has_eds(path) for path in files) == 1
    assert scan_rotation_deg(folder / "0011 - HAADF 15Mx Nano 90deg.emd") == 90.0

    result = build_showfolder(folder, thumb=32)
    assert len(result.image_items) == 3
    assert len(result.eds_items) == 1
    assert result.eds_widgets == []
    assert len(result.fov_groups) == 1


def test_load_tutorial_showfolder_folder_falls_back_offline(monkeypatch, tmp_path):
    def fail_download(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(tutorials, "snapshot_download", fail_download, raising=False)
    monkeypatch.setattr(tutorials, "create_tutorial_showfolder_folder", lambda path=None: tmp_path)

    folder = tutorials.load_tutorial_showfolder_folder(verbose=False)

    assert folder == tmp_path


def test_load_tutorial_showfolder_folder_can_require_real_download(monkeypatch):
    def fail_download(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(tutorials, "snapshot_download", fail_download, raising=False)

    try:
        tutorials.load_tutorial_showfolder_folder(verbose=False, allow_fallback=False)
    except OSError as exc:
        assert "offline" in str(exc)
    else:
        raise AssertionError("expected real-data tutorial loader to raise when download fails")
