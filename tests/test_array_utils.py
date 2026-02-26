import numpy as np
import pytest
import torch

from quantem.widget.array_utils import get_array_backend, to_numpy, bin2d

def test_backend_numpy():
    arr = np.array([1, 2, 3])
    assert get_array_backend(arr) == "numpy"

def test_backend_torch():
    tensor = torch.tensor([1, 2, 3])
    assert get_array_backend(tensor) == "torch"

def test_backend_list():
    assert get_array_backend([1, 2, 3]) == "unknown"

def test_to_numpy_from_numpy():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = to_numpy(arr)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, arr)

def test_to_numpy_from_torch():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = to_numpy(tensor)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, [1.0, 2.0, 3.0])

def test_to_numpy_from_torch_2d():
    tensor = torch.rand(10, 10)
    result = to_numpy(tensor)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10)

def test_to_numpy_dtype_conversion():
    arr = np.array([1, 2, 3], dtype=np.int32)
    result = to_numpy(arr, dtype=np.float32)
    assert result.dtype == np.float32

def test_to_numpy_from_list():
    result = to_numpy([1, 2, 3])
    assert isinstance(result, np.ndarray)


# === bin2d ===

def test_bin2d_2d_shape():
    arr = np.ones((8, 8), dtype=np.float32)
    result = bin2d(arr, factor=2)
    assert result.shape == (4, 4)

def test_bin2d_3d_shape():
    arr = np.ones((3, 8, 8), dtype=np.float32)
    result = bin2d(arr, factor=2)
    assert result.shape == (3, 4, 4)

def test_bin2d_values_correct():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = bin2d(arr, factor=2)
    assert result.shape == (2, 2)
    # Top-left 2x2: mean(0,1,4,5) = 2.5
    assert result[0, 0] == pytest.approx(2.5)
    # Top-right 2x2: mean(2,3,6,7) = 4.5
    assert result[0, 1] == pytest.approx(4.5)

def test_bin2d_trimming():
    arr = np.ones((7, 9), dtype=np.float32)
    result = bin2d(arr, factor=2)
    assert result.shape == (3, 4)

def test_bin2d_dtype_float32():
    arr = np.ones((8, 8), dtype=np.float64)
    result = bin2d(arr, factor=2)
    assert result.dtype == np.float32

def test_bin2d_factor_4():
    arr = np.ones((16, 16), dtype=np.float32) * 3.0
    result = bin2d(arr, factor=4)
    assert result.shape == (4, 4)
    assert np.allclose(result, 3.0)

def test_bin2d_torch_input():
    tensor = torch.ones(8, 8)
    result = bin2d(tensor, factor=2)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4)
    assert np.allclose(result, 1.0)

def test_bin2d_3d_values():
    arr = np.ones((2, 4, 4), dtype=np.float32) * 5.0
    result = bin2d(arr, factor=2)
    assert result.shape == (2, 2, 2)
    assert np.allclose(result, 5.0)
