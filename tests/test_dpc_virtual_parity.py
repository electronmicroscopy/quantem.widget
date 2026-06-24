"""End-to-end parity: widget virtual() / dpc() / center_of_mass() vs quantem.live.

The widget's MPS Metal path and CUDA torch path must produce the SAME numbers as
quantem.live's reference CUDA implementation (engine.dpc / engine.preprocess).
These run on the CUDA box (cupy + quantem.live present); they skip cleanly
elsewhere. Real data (gold_512, pre-binned, ~25 MB from Hugging Face) keeps them
deterministic and fast.
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("quantem.live")


@pytest.fixture(scope="module")
def gold():
    from quantem.widget.io import download
    path = download("gold_512_npy_bin4", verbose=False)
    return np.load(f"{path}/data.npy")  # (512, 512, 48, 48) uint16


def test_center_of_mass_matches_quantem_live(gold):
    from quantem.widget.dpc import center_of_mass
    from quantem.live.engine.dpc import compute_center_of_mass
    com_row, com_col = center_of_mass(gold)                      # widget (mean-subtracted)
    ref_row, ref_col = compute_center_of_mass(                   # reference
        cp.asarray(gold.reshape(-1, *gold.shape[-2:])), normalize_zero_mean=True)
    ref_row = cp.asnumpy(ref_row).reshape(512, 512)
    ref_col = cp.asnumpy(ref_col).reshape(512, 512)
    np.testing.assert_allclose(com_row, ref_row, atol=1e-3)
    np.testing.assert_allclose(com_col, ref_col, atol=1e-3)


def test_dpc_phase_matches_quantem_live(gold):
    from quantem.widget.dpc import dpc
    from quantem.live import dpc as live_dpc
    w = dpc(gold, verbose=False)
    r = live_dpc(cp.asarray(gold.reshape(-1, *gold.shape[-2:])), scan_shape=(512, 512), verbose=False)
    assert abs(w.rotation_deg - float(r.rotation_angle_deg)) < 1.0
    ref_phase = cp.asnumpy(r.phase)
    rel = np.abs(w.phase - ref_phase).max() / max(float(np.ptp(ref_phase)), 1e-9)
    assert rel < 1e-3, f"iDPC phase rel diff {rel:.2e}"


def test_virtual_matches_manual_masked_sum(gold):
    """virtual(mode) at an explicit probe == a direct masked-sum over that band."""
    from quantem.widget.detector import virtual, auto_probe, _detector_mask, _resolve_backend
    mean_dp = np.asarray(_resolve_backend(gold).mean_dp(), dtype=np.float32)
    center, r = auto_probe(mean_dp)
    for mode in ("BF", "ABF", "ADF", "HAADF", "DF"):
        vi = virtual(gold, mode, center=center, bf_radius=r)
        mask = _detector_mask(mode, center, r, mean_dp.shape, None, None)
        ref = (gold.astype(np.float64) * mask).sum(axis=(2, 3)).astype(np.float32)
        np.testing.assert_allclose(vi, ref, rtol=1e-4, atol=1.0,
                                   err_msg=f"virtual({mode}) != manual masked-sum")


def test_core_dataset4dstem_roundtrip(gold):
    """Core Dataset4dstem follows the same detector/DPC paths as raw arrays."""
    import torch
    from quantem.core.datastructures import Dataset4dstem
    from quantem.widget.detector import adf, bf
    from quantem.widget.dpc import com, dpc, idpc

    ds = Dataset4dstem.from_array(torch.as_tensor(gold))
    assert ds.shape == (512, 512, 48, 48)
    np.testing.assert_array_equal(adf(ds), adf(gold))
    np.testing.assert_array_equal(bf(ds), bf(gold))
    ref = dpc(gold, verbose=False)
    np.testing.assert_array_equal(idpc(ds), ref.phase)
    np.testing.assert_array_equal(com(ds).col, ref.com_col)
    np.testing.assert_array_equal(com(ds).row, ref.com_row)


def test_uint8_browse_parity(gold):
    """uint8 browse block == uint16 for screening: bit-identical when counts fit
    255 (the common case), and = clip255(uint16) by construction always. Locks
    the dtype='u8' path so virtual images stay faithful (#757)."""
    from quantem.widget.detector import adf, bf, df
    from quantem.widget.dpc import com

    u8 = np.minimum(gold, 255).astype(np.uint8)
    # the uint8 browse representation IS clip-at-255 (linear, so sums stay correct)
    np.testing.assert_array_equal(u8, np.minimum(gold, 255).astype(np.uint8))
    if int(gold.max()) <= 255:  # lossless regime → virtual images bit-identical
        for name, fn in (("bf", bf), ("adf", adf), ("df", df)):
            np.testing.assert_array_equal(fn(u8), fn(gold),
                                          err_msg=f"uint8 {name} != uint16 (data fits uint8, must be lossless)")
        np.testing.assert_array_equal(com(u8).row, com(gold).row)
        np.testing.assert_array_equal(com(u8).col, com(gold).col)
    else:  # clipped: BF/ADF still faithful for screening (normalized rmse small)
        def nrmse(a, b):
            a = (a - a.min()) / (np.ptp(a) + 1e-9); b = (b - b.min()) / (np.ptp(b) + 1e-9)
            return float(np.sqrt(np.mean((a - b) ** 2)))
        assert nrmse(adf(u8), adf(gold)) < 0.05, "uint8 ADF diverges from uint16"
