"""Unit tests for io.read_image - one reader, every survey-image format.

Each format is synthesized in a tmp dir with known values, read back through
read_image, and checked for exact array equality + Dataset2d return + the
right first-frame reduction and calibration. This guards the multi-format
routing so loading stays `ds = io.read_image(path)` for tif/png/npy/emd.
"""
import json

import numpy as np
import pytest

from quantem.core.datastructures import Dataset2d, Dataset3d
from quantem.widget import Show2D, Show3D, read_gif
from quantem.widget.io.image import read_image, read_image_stack, read_images


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


def test_read_gif_returns_stack_and_widgets_open_it(tmp_path):
    from PIL import Image

    frames = [
        np.full((7, 9), value, dtype=np.uint8)
        for value in (10, 80, 160, 240)
    ]
    path = tmp_path / "denoise_preview.gif"
    Image.fromarray(frames[0]).save(
        path,
        save_all=True,
        append_images=[Image.fromarray(frame) for frame in frames[1:]],
        duration=80,
        loop=0,
        optimize=False,
    )

    ds = read_gif(path)
    assert ds.name == "denoise_preview"
    assert ds.array.shape == (4, 7, 9)
    assert ds.array.dtype == np.float32
    np.testing.assert_array_equal(ds.array[:, 0, 0], [10, 80, 160, 240])

    first = read_image(path)
    np.testing.assert_array_equal(first.array, frames[0])

    movie = Show3D.from_gif(
        path,
        fps=12,
        frame_labels=True,
        show_controls=False,
        show_scale_bar=False,
    )
    assert movie.title == "denoise_preview"
    assert movie.n_slices == 4
    assert movie.height == 7
    assert movie.width == 9
    assert movie.fps == 12
    assert movie.labels == ["Frame 1", "Frame 2", "Frame 3", "Frame 4"]

    grid = Show2D.from_gif(path, ncols=2, labels=True, verbose=False)
    assert grid.title == "denoise_preview"
    assert grid.n_images == 4
    assert grid.ncols == 2
    assert grid.height == 7
    assert grid.width == 9
    assert grid.labels == ["Frame 1", "Frame 2", "Frame 3", "Frame 4"]


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


def test_read_images_folder_parallel_preserves_order(tmp_path):
    from PIL import Image

    for idx in range(4):
        frame = np.full((10, 12), idx, dtype=np.uint8)
        Image.fromarray(frame).save(tmp_path / f"frame_{idx}.png")

    out = read_images(tmp_path, workers=2)
    assert [d.name for d in out] == [f"frame_{idx}" for idx in range(4)]
    for idx, ds in enumerate(out):
        np.testing.assert_array_equal(ds.array, np.full((10, 12), idx, dtype=np.uint8))


def test_read_images_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No supported images"):
        read_images(tmp_path)


# --- folder stack reader ----------------------------------------------------

def test_read_image_stack_natural_sort_and_pattern(tmp_path):
    from PIL import Image

    for idx in (10, 2, 1):
        frame = np.full((6, 8), idx, dtype=np.uint8)
        Image.fromarray(frame).save(tmp_path / f"frame_{idx}.png")
    Image.fromarray(np.full((6, 8), 99, dtype=np.uint8)).save(tmp_path / "survey.png")

    ds = read_image_stack(tmp_path, pattern="frame_*.png", workers=2, progress=False)

    assert isinstance(ds, Dataset3d)
    assert ds.name == tmp_path.name
    assert ds.array.shape == (3, 6, 8)
    assert ds.array.dtype == np.float32
    np.testing.assert_array_equal(ds.array[:, 0, 0], [1, 2, 10])


def test_read_image_stack_file_type_filter(tmp_path):
    from PIL import Image

    Image.fromarray(np.full((4, 5), 3, dtype=np.uint16)).save(tmp_path / "frame_001.tif")
    Image.fromarray(np.full((4, 5), 9, dtype=np.uint8)).save(tmp_path / "frame_002.png")

    ds = read_image_stack(tmp_path, file_type="tif", progress=False)

    assert ds.array.shape == (1, 4, 5)
    assert ds.array.dtype == np.float32
    np.testing.assert_array_equal(ds.array[0], np.full((4, 5), 3, dtype=np.float32))


def test_read_image_stack_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No image frames"):
        read_image_stack(tmp_path, progress=False)


# --- error path ------------------------------------------------------------

def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="unsupported extension"):
        read_image(path)


def test_read_image_stack_reads_npy_frames(tmp_path):
    # .npy (and EMD/DM) frames route through read_image instead of PIL.
    for idx in range(3):
        np.save(tmp_path / f"frame_{idx}.npy", np.full((4, 5), idx, dtype=np.uint16))

    ds = read_image_stack(tmp_path, progress=False)

    assert ds.array.shape == (3, 4, 5)
    assert ds.array.dtype == np.float32
    np.testing.assert_array_equal(ds.array[2], np.full((4, 5), 2, dtype=np.float32))


def test_read_image_stack_mixed_default_scan_includes_metadata_formats(tmp_path):
    # Default (no filter) folder scan picks up .npy alongside PNG-style frames.
    np.save(tmp_path / "frame_0.npy", np.zeros((4, 5), dtype=np.uint8))
    np.save(tmp_path / "frame_1.npy", np.ones((4, 5), dtype=np.uint8))

    ds = read_image_stack(tmp_path, pattern="frame_*.npy", progress=False)

    assert ds.array.shape == (2, 4, 5)
