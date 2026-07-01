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
