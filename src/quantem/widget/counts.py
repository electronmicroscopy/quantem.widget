"""Exact sums and maxima of integer counts in torch, on every device.

Counts are summed in int64 and become float only at the small reduced output,
so a virtual image or a binned pattern of counts is exact. Every widget that
reduces integer data with torch calls ``exact_sum`` or ``exact_max`` here.

Apple MPS reduces integers differently: on an Intel Mac's AMD GPU (torch 2.2)
an int64 sum or maximum along an axis aborts the process, and an int32 sum
wraps past 2**31 without an error. There the counts are summed in int32 blocks
too short to overflow and the block sums add up in int64 (``int32_block_sum``),
so the totals are exact there too. CUDA and the CPU sum in int64 directly.
"""

import numpy as np
import torch

CHUNK_BYTES = 256 << 20  # transient budget of one reduction pass
INT32_MAX = 2**31 - 1
TORCH_HAS_UINT16 = hasattr(torch, "uint16")  # torch 2.2, the newest for Intel Macs, has none


def host_dtype_for_torch(dtype) -> np.dtype:
    """The NumPy dtype a value of ``dtype`` crosses into torch as, holding every value exactly.

    uint32 becomes int64 (no torch version reduces uint32 on every device) and
    uint16 becomes int32 on torch 2.2, which has no uint16; other dtypes cross
    as they are.
    """
    dtype = np.dtype(dtype)
    if dtype == np.uint32:
        return np.dtype(np.int64)
    if dtype == np.uint16 and not TORCH_HAS_UINT16:
        return np.dtype(np.int32)
    return dtype


def tensor_dtype(dtype) -> torch.dtype:
    """The torch dtype that holds every value of the NumPy ``dtype`` (``host_dtype_for_torch``)."""
    return torch.from_numpy(np.empty(0, host_dtype_for_torch(dtype))).dtype


def counts_tensor(values: np.ndarray, device) -> torch.Tensor:
    """A host array as a tensor on ``device`` that every torch version reduces exactly.

    The array is widened as ``host_dtype_for_torch`` says, and copied when it
    is read-only, as ``torch.from_numpy`` needs a writable buffer. Otherwise
    the tensor shares the array's memory on the CPU.
    """
    host = host_dtype_for_torch(values.dtype)
    if host != values.dtype or not values.flags.writeable:
        values = values.astype(host)
    return torch.from_numpy(values).to(device)


def exact_sum(values_t: torch.Tensor, dim: int | tuple[int, ...], bound: int | None = None) -> torch.Tensor:
    """Exact int64 sum of integer ``values_t`` along ``dim`` (an axis or a tuple of axes).

    ``bound`` is the largest magnitude a value can have (``count_bound``). When
    it is given, and always on Apple MPS, the sum runs in int32 blocks; a tuple
    of axes is then reduced one axis at a time, the bound growing by each
    reduced length. A caller that sums many chunks of one tensor passes the
    bound once instead of rescanning every chunk.
    """
    if bound is None and values_t.device.type == "mps":
        bound = count_bound(values_t)
    if bound is None:
        # cast before the sum: some devices do not implement mixed uint16/int64 sums
        return values_t.to(torch.int64).sum(dim=dim)
    for axis in sorted((dim,) if isinstance(dim, int) else dim, reverse=True):
        length = values_t.shape[axis]
        values_t = int32_block_sum(values_t, axis, bound)
        bound *= length
    return values_t


def exact_max(values_t: torch.Tensor, bound: int | None = None) -> torch.Tensor:
    """Exact int64 maximum of integer ``values_t`` over dim 0, through int32 on Apple MPS."""
    if bound is None and values_t.device.type == "mps":
        bound = count_bound(values_t)
    if bound is None:
        return values_t.to(torch.int64).amax(dim=0)
    return int32_block_max(values_t, bound)


def detector_bin_mean(values_t: torch.Tensor, det_bin: int) -> np.ndarray:
    """Mean of every ``det_bin`` x ``det_bin`` detector block of a ``(rows, cols, det_rows, det_cols)`` tensor.

    Returns host float32. Counts sum exactly (``exact_sum``) and the total is
    divided once in float64, so a binned pattern is the exact mean rounded once;
    float data averages in float32. ``det_bin=1`` returns the values as float32.
    """
    if det_bin == 1:
        return values_t.float().cpu().numpy()
    rows, cols, det_rows, det_cols = values_t.shape
    blocks_t = values_t.reshape(rows, cols, det_rows // det_bin, det_bin, det_cols // det_bin, det_bin)
    if torch.is_floating_point(blocks_t):
        return blocks_t.float().mean(dim=(3, 5)).cpu().numpy()
    return (exact_sum(blocks_t, dim=(3, 5)).cpu().numpy() / (det_bin * det_bin)).astype(np.float32)


def count_bound(data_t: torch.Tensor) -> int:
    """Largest magnitude a value of the integer tensor ``data_t`` can have.

    Up to 16 bits the dtype bounds it (no pass over the data); wider integers,
    such as uint16 counts that torch 2.2 holds as int32, are bounded by their
    actual minimum and maximum. ``int32_block_sum`` sizes its blocks with it.
    The extremes are taken in ``CHUNK_BYTES`` slices: on an Intel Mac's AMD GPU
    one MPS maximum over 2**27 values came back wrong.
    """
    if data_t.dtype == torch.bool:
        return 1
    if data_t.element_size() <= 2:
        info = torch.iinfo(data_t.dtype)
        return max(info.max, -info.min)
    values_t = data_t.flatten()
    step = CHUNK_BYTES // 8
    slices = [values_t[start : start + step] for start in range(0, values_t.numel(), step)]
    return max((max(int(slice_t.amax()), -int(slice_t.amin())) for slice_t in slices), default=0)


# ---------------------------------------------------------------------------
# int32 block reductions (the Apple MPS path)
# ---------------------------------------------------------------------------


def int32_block_sum(values_t: torch.Tensor, dim: int, bound: int) -> torch.Tensor:
    """Exact int64 sum of integer ``values_t`` along ``dim`` built from int32 sums.

    A block of ``INT32_MAX // bound`` values of magnitude at most ``bound``
    cannot leave int32, so each block sums exactly in int32; the block sums
    then add up elementwise in int64. Values beyond int32 are split into 16-bit
    limbs, ``value = high * 65536 + low`` with ``0 <= low < 65536``, and each
    limb is summed this way.
    """
    if bound > INT32_MAX:
        values_t = values_t.to(torch.int64)
        low_t = values_t & 0xFFFF
        high_t = (values_t - low_t) // 65536  # an exact quotient, so truncating division equals the floor
        return int32_block_sum(high_t, dim, bound // 65536 + 1) * 65536 + int32_block_sum(low_t, dim, 0xFFFF)
    length = values_t.shape[dim]
    total_shape = values_t.shape[:dim] + values_t.shape[dim + 1 :]
    total_t = torch.zeros(total_shape, dtype=torch.int64, device=values_t.device)
    step = max(1, INT32_MAX // max(1, bound))
    for start in range(0, length, step):
        block_t = values_t.narrow(dim, start, min(step, length - start)).to(torch.int32)
        total_t += block_t.sum(dim=dim, dtype=torch.int32).to(torch.int64)
    return total_t


def int32_block_max(values_t: torch.Tensor, bound: int) -> torch.Tensor:
    """Exact int64 maximum of integer ``values_t`` over dim 0 with int32 reductions.

    Values within int32 reduce as int32. Wider values compare limb by limb:
    the largest high limb, then the largest low limb among the rows that hold
    it.
    """
    if bound <= INT32_MAX:
        return values_t.to(torch.int32).amax(dim=0).to(torch.int64)
    values_t = values_t.to(torch.int64)
    low_t = values_t & 0xFFFF
    high_t = (values_t - low_t) // 65536
    high_max_t = int32_block_max(high_t, bound // 65536 + 1)
    low_max_t = torch.where(high_t == high_max_t, low_t, -1).to(torch.int32).amax(dim=0).to(torch.int64)
    return high_max_t * 65536 + low_max_t
