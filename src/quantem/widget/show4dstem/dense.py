"""Detector reductions of a dense 4D tensor, in torch on the device the tensor lives on.

This is the widget's definition of the math every Show4DSTEM view shows
(virtual images, mean pattern, scan-ROI patterns); quantem.gpu's encoded
kernels must give the same numbers. Integer counts are summed exactly in int64
on every device (``quantem.widget.counts``, int32 blocks on Apple MPS) and turn
into float only at the small reduced output, so a virtual image of counts is
exact. A mean divides the exact total once in float64 and rounds once to
float32. Float intensities sum in float64, except on Apple MPS, which has no
float64 and sums in float32.

Every reduction runs in chunks of scan positions so its wide transient (the
int64 or float64 copy of the selected pixels) stays inside ``CHUNK_BYTES``.
"""

import numpy as np
import torch

from quantem.widget.counts import CHUNK_BYTES, count_bound, exact_max, exact_sum


class DenseSession:
    """Detector queries over a ``(scan_row, scan_col, det_row, det_col)`` tensor.

    Answers the queries Show4DSTEM asks a detector session: ``frame``,
    ``mean_dp``, ``masked_sum`` and ``reduce_frames``. Dense data carries no
    detector flags, so ``detector_validity`` is None.
    """

    detector_validity = None

    def __init__(self, data_t: torch.Tensor):
        """Pick the exact accumulator for the data's dtype and device and view the scan as ``(positions, pixels)``."""
        self.scan_shape = (int(data_t.shape[0]), int(data_t.shape[1]))
        self.detector_shape = (int(data_t.shape[2]), int(data_t.shape[3]))
        self.num_frames = self.scan_shape[0] * self.scan_shape[1]
        self.device = data_t.device
        self.integer = not torch.is_floating_point(data_t)
        if self.integer:
            self.accumulator = torch.int64
        else:
            # MPS has no float64
            self.accumulator = torch.float32 if data_t.device.type == "mps" else torch.float64
        # largest count magnitude, set only where integer sums run in int32 blocks
        self.count_bound = count_bound(data_t) if self.integer and data_t.device.type == "mps" else None
        self._flat_t = data_t.reshape(self.num_frames, -1)

    def frame(self, index: int) -> np.ndarray:
        """The pattern at flat row-major scan ``index``, in the data's dtype."""
        pattern_t = self._flat_t[int(index)].reshape(self.detector_shape)
        if pattern_t.dtype == torch.bfloat16:
            pattern_t = pattern_t.float()  # NumPy has no bfloat16
        return pattern_t.cpu().numpy()

    def mean_dp(self) -> np.ndarray:
        """Mean pattern: exact total over every scan position divided once in float64, as float32."""
        total_t = self._chunked_sum(None)
        return self._divide(total_t, self.num_frames).reshape(self.detector_shape)

    def masked_sum(self, mask: np.ndarray) -> np.ndarray:
        """Virtual image: the sum of the ``mask`` pixels at each scan position, float32 ``scan_shape``.

        Only the selected detector pixels are read, so a small BF disk costs a
        fraction of the full stack.
        """
        selected_t = torch.as_tensor(np.flatnonzero(np.asarray(mask, dtype=bool)), device=self.device)
        if selected_t.numel() == 0:
            return np.zeros(self.scan_shape, dtype=np.float32)
        image_t = torch.empty(self.num_frames, dtype=self.accumulator, device=self.device)
        step = max(1, CHUNK_BYTES // (selected_t.numel() * 8))
        for start in range(0, self.num_frames, step):
            chunk_t = self._flat_t[start : start + step].index_select(1, selected_t)
            image_t[start : start + step] = self._sum(chunk_t, dim=1)
        return image_t.cpu().numpy().astype(np.float32).reshape(self.scan_shape)

    def reduce_frames(self, indices, mode: str = "mean") -> np.ndarray:
        """The patterns at flat scan ``indices`` reduced by ``mean``, ``sum`` or ``max``.

        Counts give exact uint64 ``sum`` and ``max``; float data gives float32.
        ``mean`` is float32 everywhere, the exact total divided once in float64.
        """
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if mode == "max":
            maximum_t = None
            step = max(1, CHUNK_BYTES // (self._flat_t.shape[1] * 8))
            for start in range(0, indices.size, step):
                rows_t = torch.as_tensor(indices[start : start + step], device=self.device)
                patterns_t = self._flat_t.index_select(0, rows_t)
                chunk_t = exact_max(patterns_t, self.count_bound) if self.integer else patterns_t.to(self.accumulator).amax(dim=0)
                maximum_t = chunk_t if maximum_t is None else torch.maximum(maximum_t, chunk_t)
            return self._exact(maximum_t).reshape(self.detector_shape)
        total_t = self._chunked_sum(indices)
        if mode == "sum":
            return self._exact(total_t).reshape(self.detector_shape)
        if mode == "mean":
            return self._divide(total_t, indices.size).reshape(self.detector_shape)
        raise ValueError(f"Unknown frame reduction {mode!r}; use mean, sum, or max.")

    def close(self) -> None:
        """Drop the reference to the data so the widget's ``free`` can release it."""
        self._flat_t = None

    # ---------------------------------------------------------------
    # Primitives
    # ---------------------------------------------------------------

    def _chunked_sum(self, indices) -> torch.Tensor:
        """Sum of the patterns at ``indices`` (every scan position when None) in the wide accumulator."""
        total_t = torch.zeros(self._flat_t.shape[1], dtype=self.accumulator, device=self.device)
        step = max(1, CHUNK_BYTES // (self._flat_t.shape[1] * 8))
        count = self.num_frames if indices is None else indices.size
        for start in range(0, count, step):
            if indices is None:
                chunk_t = self._flat_t[start : start + step]
            else:
                chunk_t = self._flat_t.index_select(0, torch.as_tensor(indices[start : start + step], device=self.device))
            total_t += self._sum(chunk_t, dim=0)
        return total_t

    def _sum(self, values_t: torch.Tensor, dim: int) -> torch.Tensor:
        """``values_t`` summed along ``dim`` in the wide accumulator, exactly for counts on every device."""
        if self.integer:
            return exact_sum(values_t, dim, self.count_bound)
        return values_t.to(self.accumulator).sum(dim=dim)

    def _divide(self, total_t: torch.Tensor, count: int) -> np.ndarray:
        """``total / count`` in float64 on the host, rounded once to float32."""
        return (total_t.cpu().numpy().astype(np.float64) / max(1, count)).astype(np.float32)

    def _exact(self, total_t: torch.Tensor) -> np.ndarray:
        """An integer total as uint64, a float total as float32."""
        values = total_t.cpu().numpy()
        return values.astype(np.uint64) if self.integer else values.astype(np.float32)
