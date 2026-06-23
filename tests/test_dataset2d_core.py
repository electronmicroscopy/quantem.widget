import numpy as np


def test_read_image_returns_core_dataset2d(tmp_path):
    from quantem.core.datastructures import Dataset2d
    from quantem.widget.io.image import read_image

    path = tmp_path / "survey.npy"
    np.save(path, np.arange(4, dtype=np.float32).reshape(2, 2))

    ds = read_image(path)

    assert isinstance(ds, Dataset2d)
    assert ds.name == "survey"
    np.testing.assert_array_equal(ds.array, np.array([[0, 1], [2, 3]], dtype=np.float32))
