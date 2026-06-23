import h5py
import numpy as np


def test_cpu_det_bin_uint8_clips_after_sum(tmp_path):
    """dtype='u8' browse loads clip binned uint16 sums instead of wrapping."""
    from quantem.widget.io import load

    master = tmp_path / "tiny_master.h5"
    data_file = tmp_path / "tiny_data_000001.h5"
    frames = np.array(
        [
            [[100, 100, 1, 2], [100, 100, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[20, 20, 30, 30], [20, 20, 30, 30], [40, 40, 50, 50], [40, 40, 50, 50]],
            [[255, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            [[0, 0, 0, 0], [0, 300, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=np.uint16,
    )
    with h5py.File(data_file, "w") as f:
        f.create_dataset("entry/data/data", data=frames)
    with h5py.File(master, "w") as f:
        f["entry/data/data_000001"] = h5py.ExternalLink(
            data_file.name, "entry/data/data"
        )
        ds = f.create_dataset(
            "entry/instrument/detector/detectorSpecific/ntrigger", data=4
        )
        ds[()]  # keep h5py from pruning an otherwise empty-looking path

    loaded = load(str(master), backend="cpu", det_bin=2, dtype="u8", verbose=False).data
    expected = np.clip(
        frames.reshape(4, 2, 2, 2, 2).sum(axis=(2, 4)), 0, 255
    ).astype(np.uint8).reshape(2, 2, 2, 2)

    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, expected)
