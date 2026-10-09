"""Array utilities for widgets. NumPy + PyTorch input."""

import numpy as np
import torch

from quantem.widget.adapters import core as core_adapter


def to_numpy(data, dtype: np.dtype | None = None) -> np.ndarray:
    """Convert NumPy / PyTorch / a quantem dataset (numpy- or tensor-backed) to NumPy.

    Upcasts torch dtypes numpy can't represent (bfloat16, float8) to float32 first
    so the user sees their data instead of "Got unsupported ScalarType BFloat16".
    """
    if core_adapter.is_dataset(data):
        data = core_adapter.as_array(data)
    if isinstance(data, torch.Tensor):
        if data.dtype == torch.bfloat16 or str(data.dtype).startswith("torch.float8"):
            data = data.to(torch.float32)
        result = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        result = data
    elif hasattr(data, "get") and type(data).__module__.split(".", 1)[0] == "cupy":
        # a CuPy array, recognised by its module so cupy is never imported here
        result = data.get()
    else:
        # Last-resort fallback covers RgbImage.__array__, dlpack-compatible objects, etc.
        try:
            result = np.asarray(data)
        except (TypeError, ValueError, RuntimeError) as error:
            raise TypeError(
                f"to_numpy expected a NumPy, PyTorch, or CuPy array, got {type(data).__name__}."
            ) from error
    if dtype is not None:
        result = np.asarray(result, dtype=dtype)
    return result


def _resize_image(image: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    """Center-pad image to (target_rows, target_cols) with the image MEDIAN. For gallery
    alignment when panels (or a 0/90 pair vs its larger corrected merge) differ in size.
    Median, not zero: a black border skews percentile/auto contrast and reads as data;
    the median matches the image brightness so ``auto`` works without manual vmin/vmax."""
    rows, cols = image.shape[-2:]
    if rows == target_rows and cols == target_cols:
        return image
    pad_top = (target_rows - rows) // 2
    pad_bottom = target_rows - rows - pad_top
    pad_left = (target_cols - cols) // 2
    pad_right = target_cols - cols - pad_left
    fill = float(np.median(image)) if image.size else 0.0
    return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=fill)


def bin2d(image: np.ndarray, factor: int, mode: str = "mean") -> np.ndarray:
    """Reduce 2D image by integer binning factor. mean or sum of f×f blocks.

    Trailing rows and columns that do not fill a whole block are dropped.
    """
    if factor <= 1:
        return image
    rows, cols = image.shape[-2:]
    kept_rows, kept_cols = rows - rows % factor, cols - cols % factor
    image = image[..., :kept_rows, :kept_cols]
    blocks = image.reshape(*image.shape[:-2], kept_rows // factor, factor, kept_cols // factor, factor)
    if mode == "sum":
        return blocks.sum(axis=(-3, -1))
    return blocks.mean(axis=(-3, -1))


def _b64_safe(raw: bytes) -> bytes:
    """Pad a bytes buffer to a multiple of 3 so its base64 encoding never needs
    `=` padding. The jupyter-book / nbconvert static-HTML embed emits UNPADDED
    base64 for widget binary buffers, and the jupyter-widgets html-manager
    decoder is strict ("Invalid string. Length must be a multiple of 4"). A
    buffer whose length is a multiple of 3 base64-encodes to a multiple of 4
    chars with no padding, so it survives the embed. The JS reads frame_bytes by
    explicit n_images x height x width, so the <=2 trailing zero bytes are
    ignored. Without this, any Show2D/Show3D/Show4DSTEM offline export whose
    buffer length isn't a multiple of 3 fails to mount in the static docs.
    """
    pad = (-len(raw)) % 3
    return raw + b"\x00" * pad if pad else raw
