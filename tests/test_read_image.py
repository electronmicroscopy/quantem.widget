"""Unit tests for io.read_image - one reader, every survey-image format.

Each format is synthesized in a tmp dir with known values, read back through
read_image, and checked for exact array equality + Dataset2d return + the
right first-frame reduction and calibration. This guards the multi-format
routing so loading stays `ds = io.read_image(path)` for tif/png/npy/emd.
"""
import json

import numpy as np
import pytest

from quantem.widget.io.image import read_image, read_images
from quantem.core.datastructures import Dataset2d


# --- per-format round trips ------------------------------------------------

def test_npy_exact(tmp_path):
    arr = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
    path = tmp_path / "img.npy"
    np.save(path, arr)
    ds = read_image(path)
    assert isinstance(ds, Dataset2d)
    assert ds.name == "img"
    np.testing.assert_array_equal(ds.array, arr)


@pytest.mark.parametrize("ext,dtype,maxval", [
    ("tif", np.uint8, 255),
    ("tif", np.uint16, 65535),
    ("png", np.uint8, 255),
    ("png", np.uint16, 65535),
    ("bmp", np.uint8, 255),
])
def test_pillow_formats_exact(tmp_path, ext, dtype, maxval):
    from PIL import Image
    rng = np.random.default_rng(0)
    arr = rng.integers(0, maxval + 1, size=(64, 48), dtype=dtype)
    path = tmp_path / f"img.{ext}"
    Image.fromarray(arr).save(path)
    ds = read_image(path)
    assert isinstance(ds, Dataset2d)
    np.testing.assert_array_equal(ds.array, arr)


def test_three_dim_reduced_to_first_frame(tmp_path):
    stack = np.stack([np.full((16, 16), k, np.float32) for k in range(5)])
    path = tmp_path / "stack.npy"
    np.save(path, stack)
    ds = read_image(path)
    np.testing.assert_array_equal(ds.array, stack[0])      # first frame only
    assert ds.array.shape == (16, 16)


# --- EMD: Velox layout + non-Velox fallback --------------------------------

def test_velox_emd_sampling_and_first_frame(tmp_path):
    import h5py
    path = tmp_path / "haadf.emd"
    image = np.arange(32 * 24, dtype=np.float32).reshape(32, 24)
    velox = np.repeat(image[:, :, None], 3, axis=2)        # (H, W, N) Velox shape
    meta = {"BinaryResult": {"PixelSize": {"height": 1.5e-9, "width": 2.0e-9}}}
    blob = json.dumps(meta).encode("utf-8") + b"\x00"
    with h5py.File(path, "w") as f:
        g = f.create_group("Data/Image/abc123")
        g.create_dataset("Data", data=velox)
        g.create_dataset("Metadata", data=np.frombuffer(blob, np.uint8)[:, None])
    ds = read_image(path)
    np.testing.assert_array_equal(ds.array, image)         # first frame, (H, W)
    assert ds.sampling[0] == pytest.approx(1.5)            # meters -> nm
    assert ds.sampling[1] == pytest.approx(2.0)


def test_non_velox_emd_picks_largest_dataset(tmp_path):
    import h5py
    path = tmp_path / "drift.emd"
    series = np.stack([np.full((48, 48), k, np.uint16) for k in range(2)])  # data/drift/data
    with h5py.File(path, "w") as f:
        f.create_dataset("data/drift/data", data=series)
        f.create_dataset("data/meta/small", data=np.zeros((4, 4), np.uint8))  # decoy
    ds = read_image(path)
    np.testing.assert_array_equal(ds.array, series[0].astype(np.float32))
    assert ds.array.shape == (48, 48)


# --- folder reader (mixed formats + sizes) ---------------------------------

def test_read_images_folder_mixed(tmp_path):
    from PIL import Image
    a = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    np.save(tmp_path / "b.npy", a)
    Image.fromarray(np.full((16, 24), 7, np.uint8)).save(tmp_path / "a.png")
    (tmp_path / "notes.md").write_text("ignore me")     # non-image, skipped
    out = read_images(tmp_path)
    assert [d.name for d in out] == ["a", "b"]           # sorted by filename
    assert all(isinstance(d, Dataset2d) for d in out)
    assert out[0].array.shape == (16, 24)               # different sizes OK
    np.testing.assert_array_equal(out[1].array, a)


def test_read_images_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No supported images"):
        read_images(tmp_path)


# --- error path ------------------------------------------------------------

def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="unsupported extension"):
        read_image(path)
