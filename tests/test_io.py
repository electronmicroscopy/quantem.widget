import os

import numpy as np
import pytest

from quantem.widget.io import (
    IO,
    IOResult,
    _coerce_to_float32,
    _coerce_to_stack_3d,
    _find_best_h5_dataset,
)

try:
    import h5py  # type: ignore

    _HAS_H5PY = True
except Exception:
    h5py = None
    _HAS_H5PY = False


# =========================================================================
# IOResult basics
# =========================================================================


def test_io_result_fields():
    arr = np.zeros((32, 32), dtype=np.float32)
    r = IOResult(data=arr, pixel_size=1.5, units="Å", title="test", labels=["a"])
    assert r.data is arr
    assert r.pixel_size == 1.5
    assert r.units == "Å"
    assert r.title == "test"
    assert r.labels == ["a"]
    assert r.metadata == {}
    # Duck-typing properties
    assert r.array is arr
    assert r.name == "test"


def test_io_result_defaults():
    arr = np.ones((8, 8), dtype=np.float32)
    r = IOResult(data=arr)
    assert r.pixel_size is None
    assert r.units is None
    assert r.title == ""
    assert r.labels == []
    assert r.metadata == {}


# =========================================================================
# PNG loading
# =========================================================================


def test_io_read_png(tmp_path):
    from PIL import Image

    img = Image.fromarray(np.random.randint(0, 255, (16, 16), dtype=np.uint8))
    path = tmp_path / "test.png"
    img.save(str(path))
    result = IO.file(path)
    assert isinstance(result, IOResult)
    assert result.data.shape == (16, 16)
    assert result.data.dtype == np.float32
    assert result.title == "test"
    assert result.pixel_size is None


# =========================================================================
# TIFF loading
# =========================================================================


def test_io_read_tiff_single(tmp_path):
    from PIL import Image

    img = Image.fromarray(np.random.randint(0, 255, (20, 20), dtype=np.uint8))
    path = tmp_path / "single.tiff"
    img.save(str(path))
    result = IO.file(path)
    assert result.data.shape == (20, 20)
    assert result.data.dtype == np.float32


def test_io_read_tiff_multipage(tmp_path):
    from PIL import Image

    frames = [
        Image.fromarray(np.random.randint(0, 255, (12, 12), dtype=np.uint8))
        for _ in range(4)
    ]
    path = tmp_path / "multi.tiff"
    frames[0].save(str(path), save_all=True, append_images=frames[1:])
    result = IO.file(path)
    assert result.data.shape == (4, 12, 12)
    assert len(result.labels) == 4
    assert result.labels[0] == "multi[0]"


# =========================================================================
# EMD loading
# =========================================================================


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed")
def test_io_read_emd_auto_discover(tmp_path):
    path = tmp_path / "test.emd"
    with h5py.File(path, "w") as f:
        f.create_dataset("data/image", data=np.random.rand(32, 32).astype(np.float32))
    result = IO.file(path)
    assert result.data.shape == (32, 32)
    assert result.data.dtype == np.float32


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed")
def test_io_read_emd_explicit_path(tmp_path):
    path = tmp_path / "test.emd"
    with h5py.File(path, "w") as f:
        f.create_dataset("group/arr1", data=np.ones((8, 8), dtype=np.float32))
        f.create_dataset("group/arr2", data=np.zeros((16, 16), dtype=np.float32))
    result = IO.file(path, dataset_path="group/arr1")
    assert result.data.shape == (8, 8)


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed")
def test_io_read_emd_missing_path(tmp_path):
    path = tmp_path / "test.emd"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="not found"):
        IO.file(path, dataset_path="nonexistent")


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed")
def test_io_read_emd_3d(tmp_path):
    path = tmp_path / "stack.emd"
    with h5py.File(path, "w") as f:
        f.create_dataset("stack", data=np.random.rand(5, 16, 16).astype(np.float32))
    result = IO.file(path)
    assert result.data.shape == (5, 16, 16)
    assert len(result.labels) == 5


# =========================================================================
# NPY / NPZ loading
# =========================================================================


def test_io_read_npy(tmp_path):
    arr = np.random.rand(24, 24).astype(np.float32)
    path = tmp_path / "data.npy"
    np.save(path, arr)
    result = IO.file(path)
    np.testing.assert_array_equal(result.data, arr)
    assert result.title == "data"


def test_io_read_npz(tmp_path):
    arr = np.random.rand(10, 10).astype(np.float32)
    path = tmp_path / "data.npz"
    np.savez(path, my_array=arr)
    result = IO.file(path)
    np.testing.assert_array_equal(result.data, arr)


def test_io_read_npy_3d(tmp_path):
    arr = np.random.rand(3, 16, 16).astype(np.float32)
    path = tmp_path / "stack.npy"
    np.save(path, arr)
    result = IO.file(path)
    assert result.data.shape == (3, 16, 16)
    assert len(result.labels) == 3


# =========================================================================
# HDF5 dataset discovery
# =========================================================================


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed")
def test_io_find_best_h5_prefer_2d(tmp_path):
    path = tmp_path / "test.emd"
    with h5py.File(path, "w") as f:
        f.create_dataset("small_2d", data=np.ones((4, 4), dtype=np.float32))
        f.create_dataset("big_3d", data=np.ones((10, 10, 10), dtype=np.float32))
    with h5py.File(path, "r") as f:
        ds_path, ds = _find_best_h5_dataset(f, prefer_ndim=2)
        # The 2D dataset should be preferred despite 3D being bigger
        assert ds_path == "small_2d"


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed")
def test_io_find_best_h5_prefer_3d(tmp_path):
    path = tmp_path / "test.emd"
    with h5py.File(path, "w") as f:
        f.create_dataset("small_2d", data=np.ones((4, 4), dtype=np.float32))
        f.create_dataset("big_3d", data=np.ones((10, 10, 10), dtype=np.float32))
    with h5py.File(path, "r") as f:
        ds_path, ds = _find_best_h5_dataset(f, prefer_ndim=3)
        assert ds_path == "big_3d"


# =========================================================================
# Error cases
# =========================================================================


def test_io_read_nonexistent():
    with pytest.raises(ValueError, match="does not exist"):
        IO.file("/nonexistent/file.png")


def test_io_read_folder_unknown_file_type(tmp_path):
    (tmp_path / "a.png").touch()
    with pytest.raises(ValueError, match="Unknown file_type"):
        IO.folder(tmp_path, file_type="xyz")


def test_io_read_folder_empty(tmp_path):
    with pytest.raises(ValueError, match="No PNG files"):
        IO.folder(tmp_path, file_type="png")


def test_io_read_folder_nonexistent():
    with pytest.raises(ValueError, match="does not exist"):
        IO.folder("/nonexistent/folder", file_type="png")


def test_io_read_folder_shape_mismatch(tmp_path):
    from PIL import Image

    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(str(tmp_path / "a.png"))
    Image.fromarray(np.zeros((20, 20), dtype=np.uint8)).save(str(tmp_path / "b.png"))
    with pytest.raises(ValueError, match="Inconsistent"):
        IO.folder(tmp_path, file_type="png")


# =========================================================================
# Folder loading
# =========================================================================


def test_io_read_folder_png(tmp_path):
    from PIL import Image

    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (8, 8), dtype=np.uint8))
        img.save(str(tmp_path / f"img_{i:02d}.png"))
    result = IO.folder(tmp_path, file_type="png")
    assert result.data.shape == (3, 8, 8)
    assert len(result.labels) == 3
    assert result.title == tmp_path.name


def test_io_read_folder_npy(tmp_path):
    for i in range(2):
        np.save(tmp_path / f"arr_{i}.npy", np.random.rand(12, 12).astype(np.float32))
    result = IO.folder(tmp_path, file_type="npy")
    assert result.data.shape == (2, 12, 12)


# =========================================================================
# Coercion helpers
# =========================================================================


def test_io_coerce_complex():
    arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    result = _coerce_to_float32(arr)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.abs(arr))


def test_io_coerce_to_stack_3d_from_2d():
    arr = np.zeros((16, 16), dtype=np.float32)
    result = _coerce_to_stack_3d(arr)
    assert result.shape == (1, 16, 16)


def test_io_coerce_to_stack_3d_from_4d():
    arr = np.zeros((2, 3, 8, 8), dtype=np.float32)
    result = _coerce_to_stack_3d(arr)
    assert result.shape == (6, 8, 8)


def test_io_coerce_to_stack_3d_1d_raises():
    arr = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError, match="at least 2D"):
        _coerce_to_stack_3d(arr)


# =========================================================================
# supported_formats
# =========================================================================


def test_io_supported_formats():
    fmts = IO.supported_formats()
    assert isinstance(fmts, list)
    assert fmts == sorted(fmts)
    # Native formats always present
    for ext in ["png", "tiff", "emd", "npy", "npz"]:
        assert ext in fmts


# =========================================================================
# Widget acceptance (IOResult duck typing)
# =========================================================================


def test_io_result_accepted_by_show2d():
    from quantem.widget import Show2D

    result = IOResult(
        data=np.random.rand(16, 16).astype(np.float32),
        pixel_size=2.5,
        title="IO Test",
    )
    w = Show2D(result)
    assert w.title == "IO Test"
    assert w.pixel_size == 2.5
    assert w.height == 16
    assert w.width == 16


def test_io_result_accepted_by_show3d():
    from quantem.widget import Show3D

    result = IOResult(
        data=np.random.rand(4, 16, 16).astype(np.float32),
        pixel_size=1.0,
        title="Stack IO",
        labels=["f0", "f1", "f2", "f3"],
    )
    w = Show3D(result)
    assert w.title == "Stack IO"
    assert w.pixel_size == 1.0
    assert w.n_slices == 4
    assert w.labels == ["f0", "f1", "f2", "f3"]


def test_io_result_accepted_by_mark2d():
    from quantem.widget import Mark2D

    result = IOResult(
        data=np.random.rand(20, 20).astype(np.float32),
        pixel_size=3.0,
        title="Markers",
    )
    w = Mark2D(result)
    assert w.title == "Markers"
    assert w.pixel_size == 3.0


def test_io_result_accepted_by_edit2d():
    from quantem.widget import Edit2D

    result = IOResult(
        data=np.random.rand(16, 16).astype(np.float32),
        title="Edit",
    )
    w = Edit2D(result)
    assert w.title == "Edit"


def test_io_result_accepted_by_show3dvolume():
    from quantem.widget import Show3DVolume

    result = IOResult(
        data=np.random.rand(8, 8, 8).astype(np.float32),
        pixel_size=2.0,
        title="Volume",
    )
    w = Show3DVolume(result)
    assert w.title == "Volume"
    assert w.pixel_size == 2.0


def test_io_result_accepted_by_showcomplex2d():
    from quantem.widget import ShowComplex2D

    complex_data = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    result = IOResult(
        data=complex_data,
        pixel_size=1.5,
        title="Complex",
    )
    w = ShowComplex2D(result)
    assert w.title == "Complex"
    assert w.pixel_size == 1.5


def test_io_result_2d_accepted_by_show3d():
    """Show3D should accept a 2D IOResult (wraps to single frame)."""
    from quantem.widget import Show3D

    result = IOResult(
        data=np.random.rand(16, 16).astype(np.float32),
        title="Single Frame",
    )
    w = Show3D(result)
    assert w.n_slices == 1
    assert w.title == "Single Frame"


def test_io_result_with_labels_show2d():
    """Show2D should use labels from IOResult when available."""
    from quantem.widget import Show2D

    result = IOResult(
        data=np.random.rand(3, 16, 16).astype(np.float32),
        title="Gallery",
        labels=["A", "B", "C"],
    )
    w = Show2D(result)
    assert w.labels == ["A", "B", "C"]
    assert w.n_images == 3


# =========================================================================
# IO.file with list of files/folders
# =========================================================================


def test_io_read_list_of_files(tmp_path):
    from PIL import Image

    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(np.random.randint(0, 255, (10, 10), dtype=np.uint8)).save(str(p))
        paths.append(p)
    result = IO.file(paths)
    assert result.data.shape == (3, 10, 10)
    assert len(result.labels) == 3


def test_io_read_list_single_file(tmp_path):
    """List with one file returns that file's result (not stacked)."""
    arr = np.random.rand(8, 8).astype(np.float32)
    p = tmp_path / "solo.npy"
    np.save(p, arr)
    result = IO.file([p])
    assert result.data.shape == (8, 8)
    assert result.title == "solo"


def test_io_read_empty_list():
    with pytest.raises(ValueError, match="Empty"):
        IO.file([])


def test_io_read_list_of_folders(tmp_path):
    """List of folders → reads each folder and stacks."""
    from PIL import Image

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    for i in range(2):
        Image.fromarray(np.random.randint(0, 255, (8, 8), dtype=np.uint8)).save(
            str(dir_a / f"img_{i}.png")
        )
    for i in range(3):
        Image.fromarray(np.random.randint(0, 255, (8, 8), dtype=np.uint8)).save(
            str(dir_b / f"img_{i}.png")
        )
    result = IO.file([dir_a, dir_b])
    assert result.data.shape == (5, 8, 8)
    assert len(result.labels) == 5


def test_io_read_list_mixed_files_and_folders(tmp_path):
    """List mixing files and folders → reads all and stacks."""
    from PIL import Image

    # A folder with 2 images
    sub = tmp_path / "sub"
    sub.mkdir()
    for i in range(2):
        Image.fromarray(np.random.randint(0, 255, (8, 8), dtype=np.uint8)).save(
            str(sub / f"img_{i}.png")
        )
    # A standalone file
    standalone = tmp_path / "solo.npy"
    np.save(standalone, np.random.rand(8, 8).astype(np.float32))
    result = IO.file([sub, standalone])
    assert result.data.shape == (3, 8, 8)
    assert len(result.labels) == 3


# =========================================================================
# IO.folder auto-detect
# =========================================================================


def test_io_folder_auto_detects_type(tmp_path):
    from PIL import Image

    for i in range(2):
        Image.fromarray(np.random.randint(0, 255, (8, 8), dtype=np.uint8)).save(
            str(tmp_path / f"f_{i}.png")
        )
    result = IO.folder(tmp_path)
    assert result.data.shape == (2, 8, 8)


def test_io_read_folder_auto_detect_type(tmp_path):
    """read_folder without file_type auto-detects when all files are same type."""
    for i in range(3):
        np.save(tmp_path / f"arr_{i}.npy", np.random.rand(6, 6).astype(np.float32))
    result = IO.folder(tmp_path)
    assert result.data.shape == (3, 6, 6)


def test_io_read_folder_mixed_types_raises(tmp_path):
    from PIL import Image

    np.save(tmp_path / "data.npy", np.random.rand(8, 8).astype(np.float32))
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(str(tmp_path / "img.png"))
    with pytest.raises(ValueError, match="mixed file types"):
        IO.folder(tmp_path)


# =========================================================================
# Recursive folder loading
# =========================================================================


def test_io_read_folder_recursive(tmp_path):
    from PIL import Image

    sub1 = tmp_path / "sub1"
    sub2 = tmp_path / "sub2"
    sub1.mkdir()
    sub2.mkdir()
    for i, folder in enumerate([sub1, sub2]):
        Image.fromarray(
            np.random.randint(0, 255, (8, 8), dtype=np.uint8)
        ).save(str(folder / f"img_{i}.png"))
    # Non-recursive should find nothing in root
    with pytest.raises(ValueError):
        IO.folder(tmp_path, file_type="png", recursive=False)
    # Recursive should find both
    result = IO.folder(tmp_path, file_type="png", recursive=True)
    assert result.data.shape == (2, 8, 8)


def test_io_read_folder_recursive_auto_detect(tmp_path):
    sub = tmp_path / "nested"
    sub.mkdir()
    for i in range(2):
        np.save(sub / f"arr_{i}.npy", np.random.rand(6, 6).astype(np.float32))
    result = IO.folder(tmp_path, recursive=True)
    assert result.data.shape == (2, 6, 6)


# =========================================================================
# IOResult.describe() — diff mode
# =========================================================================


def _make_5d_result_with_metadata(n_frames=3):
    """Helper: create an IOResult with frame_metadata for describe tests."""
    return IOResult(
        data=np.zeros((n_frames, 4, 4, 2, 2), dtype=np.float32),
        labels=[f"scan_{i:02d}" for i in range(n_frames)],
        frame_metadata=[
            {
                "entry/instrument/detector/count_time": 0.001 * (i + 1),
                "entry/instrument/detector/photon_energy": 12000.0,
                "entry/instrument/detector/beam_center_x": 48.0,
                "entry/instrument/detector/ntrigger": 65536,
                "entry/sample/data_collection_date": f"2026-02-{20 + i}",
            }
            for i in range(n_frames)
        ],
    )


def test_describe_diff_true_filters_constant_columns(capsys):
    """diff=True (default) should hide columns where all frames have the same value."""
    r = _make_5d_result_with_metadata()
    r.describe()  # diff=True by default
    out = capsys.readouterr().out
    # count_time and data_collection_date differ — should appear
    assert "count_time" in out
    assert "data_collection_date" in out
    # photon_energy is constant — should NOT appear
    assert "photon_energy" not in out
    assert "beam_center_x" not in out
    assert "ntrigger" not in out


def test_describe_diff_false_shows_all_columns(capsys):
    """diff=False should show all matched columns including constants."""
    r = _make_5d_result_with_metadata()
    r.describe(diff=False)
    out = capsys.readouterr().out
    assert "count_time" in out
    assert "photon_energy" in out
    assert "data_collection_date" in out


def test_describe_diff_all_identical(capsys):
    """When all values are identical across frames, describe(diff=True) prints a message."""
    r = IOResult(
        data=np.zeros((2, 4, 4, 2, 2), dtype=np.float32),
        labels=["a", "b"],
        frame_metadata=[
            {"entry/instrument/detector/count_time": 0.001},
            {"entry/instrument/detector/count_time": 0.001},
        ],
    )
    r.describe()  # diff=True, but count_time is the same
    out = capsys.readouterr().out
    assert "identical" in out.lower()
    assert "diff=False" in out


def test_describe_single_frame_shows_all(capsys):
    """Single frame: diff filtering is skipped (nothing to compare), shows all columns."""
    r = IOResult(
        data=np.zeros((1, 4, 4, 2, 2), dtype=np.float32),
        labels=["scan_00"],
        frame_metadata=[
            {
                "entry/instrument/detector/count_time": 0.001,
                "entry/instrument/detector/photon_energy": 12000.0,
            }
        ],
    )
    r.describe()  # diff=True, but only 1 frame → shows all
    out = capsys.readouterr().out
    assert "count_time" in out
    assert "photon_energy" in out


def test_describe_no_metadata(capsys):
    """No frame_metadata and no metadata → informative message."""
    r = IOResult(data=np.zeros((4, 4), dtype=np.float32))
    r.describe()
    out = capsys.readouterr().out
    assert "No per-frame metadata" in out


def test_describe_single_file_metadata(capsys):
    """Single-file metadata (no frame_metadata) prints key-value pairs."""
    r = IOResult(
        data=np.zeros((4, 4), dtype=np.float32),
        metadata={
            "entry/instrument/detector/count_time": 0.001,
            "entry/instrument/detector/description": "Dectris ARINA Si",
        },
    )
    r.describe()
    out = capsys.readouterr().out
    assert "count_time: 0.001" in out
    assert "description: Dectris ARINA Si" in out


def test_describe_all_keys_default(capsys):
    """When keys=None, all metadata keys are used (not just hardcoded subset)."""
    r = IOResult(
        data=np.zeros((2, 4, 4, 2, 2), dtype=np.float32),
        labels=["a", "b"],
        frame_metadata=[
            {"entry/custom/my_field": 1.0, "entry/custom/other": "same"},
            {"entry/custom/my_field": 2.0, "entry/custom/other": "same"},
        ],
    )
    r.describe()  # diff=True: should show my_field (differs), not other (same)
    out = capsys.readouterr().out
    assert "my_field" in out
    assert "other" not in out


def test_describe_custom_keys(capsys):
    """Custom keys parameter selects specific columns."""
    r = _make_5d_result_with_metadata()
    r.describe(keys=["photon_energy"], diff=False)
    out = capsys.readouterr().out
    assert "photon_energy" in out
    # Other keys should not appear
    assert "count_time" not in out


def test_describe_no_matching_keys(capsys):
    """Keys that don't match any metadata paths → helpful message with available keys."""
    r = _make_5d_result_with_metadata()
    r.describe(keys=["nonexistent_field"])
    out = capsys.readouterr().out
    assert "No matching" in out
    assert "Available" in out


# =========================================================================
# CPU arina backend
# =========================================================================


def _make_arina_files(tmp_path, n_frames=16, det_rows=64, det_cols=64, dtype="uint16"):
    """Create synthetic arina master + chunk files with bitshuffle+LZ4 compression."""
    if not _HAS_H5PY:
        pytest.skip("h5py not available")
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pytest.skip("hdf5plugin not available")
    rng = np.random.default_rng(42)
    data = rng.integers(0, 1000, (n_frames, det_rows, det_cols), dtype=dtype)
    prefix = "test_scan"
    # Write chunk file
    chunk_path = tmp_path / f"{prefix}_data_000001.h5"
    with h5py.File(chunk_path, "w") as f:
        f.create_dataset(
            "entry/data/data",
            data=data,
            chunks=(1, det_rows, det_cols),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
    # Write master file
    master_path = tmp_path / f"{prefix}_master.h5"
    with h5py.File(master_path, "w") as f:
        f["entry/data/data_000001"] = h5py.ExternalLink(
            chunk_path.name, "entry/data/data"
        )
        spec = f.create_group("entry/instrument/detector/detectorSpecific")
        spec.create_dataset("ntrigger", data=n_frames)
        spec.create_dataset("x_pixels_in_detector", data=det_cols)
        spec.create_dataset("y_pixels_in_detector", data=det_rows)
    return str(master_path), data


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not available")
def test_cpu_arina_load_no_binning(tmp_path):
    """CPU backend loads arina data without binning, matching raw data."""
    master_path, expected = _make_arina_files(tmp_path)
    from quantem.widget.io import _load_arina_cpu
    result = _load_arina_cpu(master_path, det_bin=1)
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not available")
def test_cpu_arina_load_with_binning(tmp_path):
    """CPU backend bins correctly — matches numpy reference."""
    det_rows, det_cols = 64, 64
    n_frames = 16  # 4x4 → auto scan_shape inferred
    master_path, raw_data = _make_arina_files(
        tmp_path, n_frames=n_frames, det_rows=det_rows, det_cols=det_cols
    )
    bin_factor = 2
    from quantem.widget.io import _load_arina_cpu
    result = _load_arina_cpu(master_path, det_bin=bin_factor)
    # Auto scan_shape: 16 frames → (4, 4), so result is (4, 4, 32, 32)
    assert result.shape == (4, 4, 32, 32)
    # Compute reference with numpy and compare flattened
    raw_f32 = raw_data.astype(np.float32)
    ref = (
        raw_f32.reshape(n_frames, det_rows // bin_factor, bin_factor, det_cols // bin_factor, bin_factor)
        .mean(axis=(2, 4))
    )
    np.testing.assert_allclose(result.reshape(ref.shape), ref, rtol=1e-5)


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not available")
def test_cpu_arina_load_with_scan_shape(tmp_path):
    """CPU backend reshapes to 4D when scan_shape is given."""
    n_frames = 16  # 4x4 scan
    master_path, _ = _make_arina_files(tmp_path, n_frames=n_frames)
    from quantem.widget.io import _load_arina_cpu
    result = _load_arina_cpu(master_path, det_bin=2, scan_shape=(4, 4))
    assert result.ndim == 4
    assert result.shape[0] == 4
    assert result.shape[1] == 4


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not available")
def test_cpu_arina_via_io_class(tmp_path):
    """IO.arina_file with backend='cpu' uses CPU fallback end-to-end."""
    master_path, expected = _make_arina_files(tmp_path)
    result = IO.arina_file(master_path, det_bin=1, backend="cpu")
    assert isinstance(result, IOResult)
    assert result.data.shape == expected.shape
    np.testing.assert_array_equal(result.data, expected)


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not available")
def test_cpu_arina_auto_fallback(tmp_path, monkeypatch):
    """backend='auto' falls back to CPU when no GPU is available."""
    monkeypatch.setattr("quantem.widget.io._detect_gpu_backend", lambda: None)
    master_path, expected = _make_arina_files(tmp_path)
    result = IO.arina_file(master_path, det_bin=1, backend="auto")
    assert isinstance(result, IOResult)
    np.testing.assert_array_equal(result.data, expected)


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not available")
def test_cpu_bin_mean_correctness():
    """_cpu_bin_mean matches numpy reference binning."""
    from quantem.widget.io import _cpu_bin_mean
    rng = np.random.default_rng(123)
    src = rng.random((8, 64, 64)).astype(np.float32)
    for bf in [2, 4, 8]:
        result = _cpu_bin_mean(src, bf)
        ref = (
            src.reshape(8, 64 // bf, bf, 64 // bf, bf)
            .mean(axis=(2, 4))
        )
        np.testing.assert_allclose(result, ref, rtol=1e-6)


def test_arina_to_show4dstem_pipeline(tmp_path):
    """IO.arina_file → Show4DSTEM end-to-end: data, title, and stats survive the handoff."""
    from quantem.widget import Show4DSTEM
    master_path, raw_data = _make_arina_files(tmp_path, n_frames=16, det_rows=32, det_cols=32)
    result = IO.arina_file(master_path, det_bin=1, backend="cpu")
    w = Show4DSTEM(result)
    assert w.shape_rows * w.shape_cols == 16
    assert w.det_rows == 32
    assert w.det_cols == 32
    assert w.title == "test_scan"
    assert len(w.dp_stats) == 4
    assert w.dp_stats[1] >= 0  # min >= 0 for uint16 source


# =========================================================================
# Real arina fixture (binned from Korean Sample C1, committed to repo)
# =========================================================================

ARINA_FIXTURE = os.path.join(
    os.path.dirname(__file__), "data", "arina_fixture", "test_arina_master.h5"
)


def test_real_arina_cpu_load():
    """CPU backend loads real arina data (binned fixture) without error."""
    from quantem.widget.io import _load_arina_cpu
    result = _load_arina_cpu(ARINA_FIXTURE, det_bin=1)
    assert result.shape == (16, 24, 24)
    assert result.dtype == np.uint32


def test_real_arina_io_class():
    """IO.arina_file loads real arina fixture end-to-end."""
    result = IO.arina_file(ARINA_FIXTURE, det_bin=1, backend="cpu")
    assert isinstance(result, IOResult)
    assert result.data.shape == (16, 24, 24)
    assert result.title == "test_arina"
    assert isinstance(result.metadata, dict)
    assert len(result.metadata) > 0  # real metadata from detector


def test_real_arina_to_show4dstem():
    """Real arina fixture → Show4DSTEM: full pipeline with real detector metadata."""
    from quantem.widget import Show4DSTEM
    result = IO.arina_file(ARINA_FIXTURE, det_bin=1, backend="cpu")
    w = Show4DSTEM(result, scan_shape=(4, 4))
    assert w.shape_rows == 4
    assert w.shape_cols == 4
    assert w.det_rows == 24
    assert w.det_cols == 24
    assert w.title == "test_arina"
    assert len(w.dp_stats) == 4
    # Virtual image should be computable
    w.roi_circle(radius=5)
    w.roi_center_row = 12
    w.roi_center_col = 12
    vi_stats = w.vi_stats
    assert len(vi_stats) == 4
    assert vi_stats[2] >= vi_stats[1]  # max >= min
