"""quantem.widget.io.bin: CuPy or Torch in, same type out."""

import numpy as np
import pytest
import torch

from quantem.widget.io import bin

cp = pytest.importorskip("cupy")


def test_torch_4d_bin_returns_torch():
    """Bin a torch 4D-STEM cube; output stays torch on the same device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_t = torch.ones(4, 4, 8, 8, device=device)
    out_t = bin(data_t, factor=2, axes="detector", reduction="sum")
    assert type(out_t) is torch.Tensor
    assert out_t.device == data_t.device
    assert out_t.shape == (4, 4, 4, 4)
    assert float(out_t[0, 0, 0, 0]) == 4.0

    out_scan_t = bin(data_t, factor=2, axes="scan", reduction="sum")
    assert type(out_scan_t) is torch.Tensor
    assert out_scan_t.shape == (2, 2, 8, 8)
    assert float(out_scan_t[0, 0, 0, 0]) == 4.0


def test_cupy_4d_bin_returns_cupy():
    """Bin a cupy 4D-STEM cube; output stays cupy."""
    data_cp = cp.ones((4, 4, 8, 8), dtype=cp.float32)
    out_cp = bin(data_cp, factor=2, axes="detector", reduction="sum")
    assert isinstance(out_cp, cp.ndarray)
    assert type(out_cp) is type(data_cp)
    assert out_cp.shape == (4, 4, 4, 4)
    assert float(out_cp[0, 0, 0, 0]) == 4.0

    out_scan_cp = bin(data_cp, factor=2, axes="scan", reduction="sum")
    assert isinstance(out_scan_cp, cp.ndarray)
    assert out_scan_cp.shape == (2, 2, 8, 8)


def test_torch_and_cupy_same_values():
    """Same host data: torch bin and cupy bin match numerically."""
    host = np.random.default_rng(0).random((4, 4, 8, 8), dtype=np.float32)
    out_cp = bin(cp.asarray(host), factor=2, axes="detector", reduction="sum")
    out_t = bin(torch.as_tensor(host), factor=2, axes="detector", reduction="sum")
    assert isinstance(out_cp, cp.ndarray)
    assert type(out_t) is torch.Tensor
    np.testing.assert_allclose(
        cp.asnumpy(out_cp), out_t.cpu().numpy(), rtol=1e-5, atol=1e-5
    )
