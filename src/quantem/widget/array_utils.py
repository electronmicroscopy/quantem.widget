"""
Array utilities for handling NumPy, CuPy, and PyTorch arrays uniformly.

This module provides utilities to convert arrays from different backends
into NumPy arrays for widget processing.
"""

from typing import Any, Literal
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


ArrayBackend = Literal["numpy", "cupy", "torch", "unknown"]


def get_array_backend(data: Any) -> ArrayBackend:
    """
    Detect the array backend of the input data.

    Parameters
    ----------
    data : array-like
        Input array (NumPy, CuPy, PyTorch, or other).

    Returns
    -------
    str
        One of: "numpy", "cupy", "torch", "unknown"
    """
    # Check PyTorch first (has both .numpy and .detach methods)
    if hasattr(data, "detach") and hasattr(data, "numpy"):
        return "torch"
    # Check CuPy (has .get() or __cuda_array_interface__)
    if hasattr(data, "__cuda_array_interface__"):
        return "cupy"
    if hasattr(data, "get") and hasattr(data, "__array__"):
        # CuPy arrays have .get() to transfer to CPU
        type_name = type(data).__module__
        if "cupy" in type_name:
            return "cupy"
    # Check NumPy
    if isinstance(data, np.ndarray):
        return "numpy"
    return "unknown"


def to_numpy(data: Any, dtype: np.dtype | None = None) -> np.ndarray:
    """
    Convert any array-like (NumPy, CuPy, PyTorch) to a NumPy array.

    Parameters
    ----------
    data : array-like
        Input array from any supported backend.
    dtype : np.dtype, optional
        Target dtype for the output array. If None, preserves original dtype.

    Returns
    -------
    np.ndarray
        NumPy array with the same data.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget.array_utils import to_numpy
    >>>
    >>> # NumPy passthrough
    >>> arr = np.random.rand(10, 10)
    >>> result = to_numpy(arr)
    >>>
    >>> # CuPy conversion (if available)
    >>> import cupy as cp
    >>> gpu_arr = cp.random.rand(10, 10)
    >>> cpu_arr = to_numpy(gpu_arr)
    >>>
    >>> # PyTorch conversion (if available)
    >>> import torch
    >>> tensor = torch.rand(10, 10)
    >>> arr = to_numpy(tensor)
    """
    backend = get_array_backend(data)

    if backend == "torch":
        # PyTorch tensor: detach from graph, move to CPU, convert to numpy
        result = data.detach().cpu().numpy()

    elif backend == "cupy":
        # CuPy array: use .get() to transfer to CPU
        if hasattr(data, "get"):
            result = data.get()
        else:
            # Fallback for __cuda_array_interface__
            import cupy as cp

            result = cp.asnumpy(data)

    elif backend == "numpy":
        # NumPy array: passthrough (may copy if dtype changes)
        result = data

    else:
        # Unknown backend: try np.asarray as fallback
        result = np.asarray(data)

    # Apply dtype conversion if specified
    if dtype is not None:
        result = np.asarray(result, dtype=dtype)

    return result


def bin2d(data, factor: int = 2) -> np.ndarray:
    """
    Spatial mean-pooling (binning) for 2D or 3D arrays.

    Parameters
    ----------
    data : array-like
        Input array with shape ``(H, W)`` or ``(N, H, W)``.
    factor : int, default 2
        Bin factor. Pixels that don't fit complete blocks are trimmed.

    Returns
    -------
    np.ndarray
        Binned array with shape ``(H//factor, W//factor)`` or
        ``(N, H//factor, W//factor)``, dtype float32.
    """
    arr = to_numpy(data).astype(np.float32)
    if arr.ndim == 2:
        h, w = arr.shape
        oh, ow = h // factor, w // factor
        trimmed = arr[:oh * factor, :ow * factor]
        return trimmed.reshape(oh, factor, ow, factor).mean(axis=(1, 3))
    # 3D: (N, H, W)
    n, h, w = arr.shape
    oh, ow = h // factor, w // factor
    trimmed = arr[:, :oh * factor, :ow * factor]
    return trimmed.reshape(n, oh, factor, ow, factor).mean(axis=(2, 4)).astype(np.float32)


def apply_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """
    Apply sub-pixel shift using bilinear interpolation.

    Uses ``torch.nn.functional.grid_sample`` on GPU when torch is available,
    falls back to numpy bilinear interpolation otherwise.

    Parameters
    ----------
    img : np.ndarray
        2D image, float32.
    dy : float
        Shift in y (rows).
    dx : float
        Shift in x (columns).

    Returns
    -------
    np.ndarray
        Shifted image, same shape, float32. Out-of-bounds pixels are zero.
    """
    if _HAS_TORCH:
        h, w = img.shape
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        t = torch.as_tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        base_y = torch.linspace(-1, 1, h, device=device)
        base_x = torch.linspace(-1, 1, w, device=device)
        gy, gx = torch.meshgrid(base_y, base_x, indexing="ij")
        grid = torch.stack([gx - dx * 2.0 / w, gy - dy * 2.0 / h], dim=-1).unsqueeze(0)
        result = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return result.squeeze().cpu().numpy()
    h, w = img.shape
    y_src = np.arange(h, dtype=np.float64) - dy
    x_src = np.arange(w, dtype=np.float64) - dx
    yy, xx = np.meshgrid(y_src, x_src, indexing="ij")
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    fy = (yy - y0).astype(np.float32)
    fx = (xx - x0).astype(np.float32)
    valid = (y0 >= 0) & (y0 + 1 < h) & (x0 >= 0) & (x0 + 1 < w)
    y0c = np.clip(y0, 0, h - 2)
    x0c = np.clip(x0, 0, w - 2)
    result = (img[y0c, x0c] * (1 - fy) * (1 - fx)
              + img[y0c, x0c + 1] * (1 - fy) * fx
              + img[y0c + 1, x0c] * fy * (1 - fx)
              + img[y0c + 1, x0c + 1] * fy * fx)
    result[~valid] = 0.0
    return result.astype(np.float32)


def _resize_image(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image using bilinear interpolation (pure numpy, no scipy)."""
    h, w = img.shape

    if h == target_h and w == target_w:
        return img

    y_new = np.linspace(0, h - 1, target_h)
    x_new = np.linspace(0, w - 1, target_w)
    x_grid, y_grid = np.meshgrid(x_new, y_new)

    y0 = np.floor(y_grid).astype(int)
    x0 = np.floor(x_grid).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)

    fy = y_grid - y0
    fx = x_grid - x0

    result = (
        img[y0, x0] * (1 - fy) * (1 - fx) +
        img[y0, x1] * (1 - fy) * fx +
        img[y1, x0] * fy * (1 - fx) +
        img[y1, x1] * fy * fx
    )
    return result.astype(img.dtype)
