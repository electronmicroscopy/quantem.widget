import numpy as np


def test_detector_virtual_accepts_core_dataset4dstem():
    from quantem.core.datastructures import Dataset4dstem
    from quantem.gpu.detector import virtual

    data = np.arange(2 * 2 * 4 * 4, dtype=np.float32).reshape(2, 2, 4, 4)
    ds = Dataset4dstem.from_array(data)

    np.testing.assert_array_equal(
        virtual(ds, "BF", center=(1.5, 1.5), bf_radius=1.0),
        virtual(data, "BF", center=(1.5, 1.5), bf_radius=1.0),
    )


def test_dpc_center_of_mass_accepts_core_dataset4dstem():
    from quantem.core.datastructures import Dataset4dstem
    from quantem.gpu.dpc import center_of_mass

    data = np.arange(2 * 2 * 4 * 4, dtype=np.float32).reshape(2, 2, 4, 4)
    ds = Dataset4dstem.from_array(data)

    core_row, core_col = center_of_mass(ds)
    raw_row, raw_col = center_of_mass(data)

    np.testing.assert_array_equal(core_row, raw_row)
    np.testing.assert_array_equal(core_col, raw_col)
