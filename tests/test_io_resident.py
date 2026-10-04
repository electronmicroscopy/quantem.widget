"""Complete native counts remain usable after the acquisition file is closed."""

import os

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1",
    reason="Set QUANTEM_CUDA_ANS_TEST=1 in an owned CUDA test window.",
)


@pytest.mark.parametrize("representation", ["packed", "encoded"])
def test_resident_detector_and_diffraction_match_native_counts(tmp_path, representation):
    from quantem.widget.io.resident import load_resident, prepare_resident

    counts = np.random.default_rng(4).integers(0, 65536, (16, 16, 32, 32), dtype=np.uint16)
    source = tmp_path / "counts.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("entry/data/data", data=counts)
    with load_resident(source, representation=representation, device=0) as loaded:
        session = prepare_resident(loaded)
        try:
            source.unlink()
            row, col = np.mgrid[:32, :32]
            for mask in [(row-16)**2+(col-16)**2 <= 8**2,
                         ((row-16)**2+(col-16)**2 >= 8**2)]:
                actual = session.masked_sum_exact(mask).reshape(16, 16)
                expected = counts[:, :, mask].sum(axis=2, dtype=np.uint64)
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(session.frame(255).reshape(32, 32), counts[15, 15])
            np.testing.assert_array_equal(
                session.mean_dp().reshape(32, 32), counts.mean(axis=(0, 1)).astype(np.float32),
            )
            com_row, com_col = session.center_of_mass()
            total = counts.sum(axis=(-2, -1), dtype=np.uint64)
            expected_row = (
                (counts * row).sum(axis=(-2, -1), dtype=np.uint64) / total
            ).astype(np.float32)
            expected_col = (
                (counts * col).sum(axis=(-2, -1), dtype=np.uint64) / total
            ).astype(np.float32)
            np.testing.assert_allclose(com_row, expected_row, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(com_col, expected_col, rtol=2e-6, atol=2e-6)
            assert loaded.representation.value == representation
        finally:
            session.close()
