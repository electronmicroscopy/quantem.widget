import h5py


def test_is_master_ready_rejects_non_hdf5_external_chunk(tmp_path):
    """Existing but corrupt linked chunk files are not browse-ready."""
    from quantem.gpu.io import inspect

    master = tmp_path / "bad_master.h5"
    data_file = tmp_path / "bad_data_000001.h5"
    data_file.write_bytes(b"not an hdf5 file")

    with h5py.File(master, "w") as f:
        f["entry/data/data_000001"] = h5py.ExternalLink(
            data_file.name, "entry/data/data"
        )

    assert not inspect(str(master)).ready
